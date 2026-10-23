"""
Reranking — a second, sharper look at a small candidate set.

Fusion is cheap and blunt: Reciprocal Rank Fusion knows only *positions*
in three lists, never how emphatic any of them was, and never anything
about the query and the passage together. That is fine for producing 50
plausible candidates and wrong for choosing the 5 that go in the prompt.

A cross-encoder is the textbook second stage and measurably the largest
single win in published benchmarks. It is also a model download and tens
to hundreds of milliseconds per query. So this module does both:

* **A feature reranker that always works.** No model, no download, sub-
  millisecond. It scores the query against each candidate on signals
  fusion structurally cannot see — whole-phrase matches, how tightly the
  query terms cluster in the passage, how much of the query is covered,
  whether the title matches, and where in the document the passage sits.

* **An optional cross-encoder**, used when one is configured and
  loadable, falling back to the features otherwise.

The reranker only ever **reorders** its input. It cannot introduce a
passage that retrieval did not return, and it cannot resurrect one that
authorization removed — so it can never widen what a user sees.
"""

from __future__ import annotations

import logging
import math
import os
import re
from dataclasses import dataclass
from typing import Any

log = logging.getLogger("agent.rerank")

# Blend: how much the reranker is allowed to disagree with fusion.
# 1.0 would discard the retrieval signal entirely, which throws away the
# dense and graph evidence; 0 would do nothing.
RERANK_WEIGHT = float(os.environ.get("ALEX_RERANK_WEIGHT", "0.65"))
# Only the head of the candidate list is worth a second look.
RERANK_DEPTH = int(os.environ.get("ALEX_RERANK_DEPTH", "30"))
# A candidate matching essentially all of a query's distinctive mass —
# an identifier, a ticket number, a contract reference — is not a
# ranking opinion, it is the answer. Above this threshold the candidate
# is placed first rather than blended, because no amount of general
# similarity elsewhere should outrank an exact match on something that
# occurs once in the corpus.
DECISIVE_RARITY = float(os.environ.get("ALEX_DECISIVE_RARITY", "0.9"))

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_\-]*")
_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "what", "who",
    "how", "why", "when", "where", "our", "are", "was", "were", "has",
    "have", "does", "did", "about", "into", "over", "all", "any", "can",
    "you", "your", "please", "need", "want", "tell", "give", "show",
    "would", "should", "could", "there", "their", "them", "than", "then",
}


def _tokens(text: str) -> list[str]:
    return [
        t for t in _TOKEN_RE.findall((text or "").lower())
        if t not in _STOPWORDS and len(t) > 2
    ]


@dataclass
class Features:
    """Why a passage scored the way it did — surfaced for transparency."""

    phrase: float = 0.0
    coverage: float = 0.0
    proximity: float = 0.0
    title: float = 0.0
    position: float = 0.0
    density: float = 0.0
    rare: float = 0.0

    def total(self) -> float:
        # Weights reflect how much each signal says about *this* passage
        # answering *this* query.
        #
        # `rare` dominates deliberately. A query term that occurs in one
        # document out of a hundred thousand — an identifier, a ticket
        # number, a customer name — makes that document the answer
        # almost by definition. BM25 is supposed to capture this through
        # IDF but its length normalisation can bury it: the document
        # that *contains* the identifier is longer for that reason, and
        # every one of its terms is scaled down as a result.
        return (
            3.0 * self.rare
            + 2.2 * self.phrase
            + 1.8 * self.coverage
            + 1.0 * self.proximity
            + 1.2 * self.title
            + 0.4 * self.position
            + 0.6 * self.density
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "rare": round(self.rare, 4),
            "phrase": round(self.phrase, 4),
            "coverage": round(self.coverage, 4),
            "proximity": round(self.proximity, 4),
            "title": round(self.title, 4),
            "position": round(self.position, 4),
            "density": round(self.density, 4),
            "total": round(self.total(), 4),
        }


def score_features(
    query: str,
    title: str,
    text: str,
    chunk_index: int = 0,
    rarity: dict[str, float] | None = None,
) -> Features:
    """Score one passage against one query on signals RRF cannot see.

    ``rarity`` maps a query term to how rare it is in the corpus (0..1,
    where 1 means it appears in essentially one document). Supplied by
    the caller because only the index knows it.
    """
    q_tokens = _tokens(query)
    if not q_tokens:
        return Features()
    q_set = set(q_tokens)
    body = (text or "").lower()
    body_tokens = _TOKEN_RE.findall(body)
    title_set = set(_tokens(title))

    f = Features()

    # Whole-phrase containment: the strongest single signal there is.
    q_phrase = " ".join(q_tokens)
    if len(q_tokens) > 1 and q_phrase in " ".join(body_tokens):
        f.phrase = 1.0
    else:
        # Longest contiguous run of query terms present in the passage.
        best = 0
        for size in range(len(q_tokens), 1, -1):
            joined = " ".join(body_tokens)
            if any(
                " ".join(q_tokens[i : i + size]) in joined
                for i in range(len(q_tokens) - size + 1)
            ):
                best = size
                break
        f.phrase = best / len(q_tokens) if best else 0.0

    present = q_set & set(body_tokens)
    f.coverage = len(present) / len(q_set)

    # Proximity: how tightly the matched terms cluster. Terms scattered
    # across a long passage usually mean it is about something else.
    positions = [i for i, t in enumerate(body_tokens) if t in q_set]
    if len(positions) > 1:
        span = positions[-1] - positions[0] + 1
        f.proximity = min(len(positions) / span, 1.0)
    elif positions:
        f.proximity = 0.5

    if title_set:
        f.title = len(q_set & title_set) / len(q_set)

    # Earlier chunks of a document carry its summary and definitions.
    f.position = 1.0 / (1.0 + 0.35 * max(chunk_index, 0))

    # Density guards against a long passage matching by sheer length.
    if body_tokens:
        f.density = min(len(present) / math.sqrt(len(body_tokens)), 1.0)

    # Rare-term match: the share of the query's *distinctiveness* this
    # passage accounts for. Weighting by rarity rather than counting
    # matches is what makes one identifier outweigh four common words.
    if rarity:
        wanted = sum(rarity.get(t, 0.0) for t in q_set)
        if wanted > 0:
            f.rare = sum(rarity.get(t, 0.0) for t in present) / wanted

    return f


# ---------------------------------------------------------------------------
# Optional cross-encoder
# ---------------------------------------------------------------------------

_CROSS_ENCODER: Any = None
_CROSS_ENCODER_TRIED = False


def _cross_encoder():
    """Load the configured cross-encoder once, or give up quietly.

    Set ``ALEX_RERANK_MODEL`` to a sentence-transformers CrossEncoder
    name to enable it. Absent or unloadable, the feature reranker is
    used — the caller never has to care which ran.
    """
    global _CROSS_ENCODER, _CROSS_ENCODER_TRIED
    if _CROSS_ENCODER_TRIED:
        return _CROSS_ENCODER
    _CROSS_ENCODER_TRIED = True
    name = os.environ.get("ALEX_RERANK_MODEL", "").strip()
    if not name:
        return None
    try:
        from sentence_transformers import CrossEncoder

        _CROSS_ENCODER = CrossEncoder(name)
        log.info("cross-encoder reranker loaded: %s", name)
    except Exception:  # noqa: BLE001
        log.warning(
            "cross-encoder %r could not be loaded; using feature reranking",
            name,
        )
        _CROSS_ENCODER = None
    return _CROSS_ENCODER


def reset() -> None:
    """Forget the loaded model (tests, config change)."""
    global _CROSS_ENCODER, _CROSS_ENCODER_TRIED
    _CROSS_ENCODER = None
    _CROSS_ENCODER_TRIED = False


def _normalise(values: list[float]) -> list[float]:
    """Scale to 0..1. Reranker scores are not calibrated, so only the
    ordering within this candidate set is meaningful."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [0.5] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def rerank(query: str, results: list, *, depth: int = RERANK_DEPTH,
           weight: float = RERANK_WEIGHT,
           rarity: dict[str, float] | None = None) -> list:
    """Reorder retrieved passages by a sharper query-passage score.

    Only the top ``depth`` are rescored; the tail keeps its fused order
    beneath them. The final score blends the reranker with the fused
    score rather than replacing it, so dense and graph evidence still
    counts for something and one bad reranker signal cannot bury a
    passage every retriever agreed on.
    """
    if not results or not (query or "").strip():
        return results
    head, tail = results[:depth], results[depth:]

    pairs = [
        (
            getattr(getattr(r, "chunk", None), "title", "") or "",
            getattr(getattr(r, "chunk", None), "text", "") or "",
        )
        for r in head
    ]

    raw: list[float] = []
    feature_list: list[Features | None] = []
    model = _cross_encoder()
    if model is not None:
        try:
            raw = [
                float(x)
                for x in model.predict([(query, f"{t}\n{b}") for t, b in pairs])
            ]
            feature_list = [None] * len(head)
        except Exception:  # noqa: BLE001
            log.warning("cross-encoder failed mid-query; using features")
            raw = []
    if not raw:
        feature_list = [
            score_features(
                query, t, b,
                getattr(getattr(r, "chunk", None), "chunk_index", 0),
                rarity=rarity,
            )
            for (t, b), r in zip(pairs, head, strict=False)
        ]
        raw = [f.total() for f in feature_list if f is not None]

    rr = _normalise([float(x) for x in raw])
    fused = _normalise([float(getattr(r, "score", 0.0)) for r in head])

    scored = []
    for i, r in enumerate(head):
        blended = weight * rr[i] + (1.0 - weight) * fused[i]
        feats = feature_list[i] if i < len(feature_list) else None
        decisive = bool(feats is not None and feats.rare >= DECISIVE_RARITY)
        # Record why, so the Transparency surface can show it.
        try:
            r.rerank_score = round(rr[i], 4)
            if feats is not None:
                r.rerank_features = feats.to_dict()
            if decisive:
                r.rerank_decisive = True
        except Exception:  # noqa: BLE001
            pass
        scored.append((decisive, blended, i, r))

    # Decisive matches first; then blended score; then the original fused
    # order, which keeps the sort stable and means reranking degrades to
    # "no change" rather than to noise.
    scored.sort(key=lambda x: (not x[0], -x[1], x[2]))
    return [r for _d, _s, _i, r in scored] + tail


__all__ = ["rerank", "score_features", "Features", "reset", "RERANK_WEIGHT"]
