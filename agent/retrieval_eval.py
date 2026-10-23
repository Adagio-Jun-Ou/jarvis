"""
Retrieval evaluation — measuring the thing before tuning it.

Every retrieval improvement is a hypothesis: contextual chunking,
reranking, a different fusion constant. Without a harness you ship the
hypothesis and find out months later, from a user, that answers got
worse for one class of question. This module is the harness.

It is deliberately **retrieval-only**. Generation quality is a separate
(harder, noisier) question; if the right passage never reaches the
prompt, no amount of prompting recovers it, so this measures whether it
does. Metrics are the standard ones:

* **Recall@k** — did any expected document reach the top *k*? The
  headline number, because a passage the model never sees cannot be
  used.
* **MRR** — how high did the *first* correct document land? Sensitive
  to ordering in a way recall is not.
* **nDCG@k** — rewards getting several relevant documents high, with
  logarithmic discount for position.
* **Precision@k** — how much of what was retrieved was actually wanted.
  Low precision is context budget spent on noise.

A case may also declare ``forbidden`` documents: material the query must
*not* surface. That is how an access-control or disambiguation
regression is caught, and it is scored separately because retrieving a
forbidden document is a different kind of failure from missing a wanted
one.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("agent.retrieval_eval")

DEFAULT_K = 5


@dataclass
class Case:
    """One golden query and what it should find."""

    query: str
    # Documents that satisfy the query. Matched on doc_id or source_ref
    # so a golden set survives re-ingestion (doc_ids are content-derived
    # but source_refs are what a human actually writes down).
    expected: list[str] = field(default_factory=list)
    # Documents this query must NOT surface.
    forbidden: list[str] = field(default_factory=list)
    # Free-form label for slicing results ("exact-match", "multi-hop").
    kind: str = "general"
    note: str = ""


@dataclass
class CaseResult:
    case: Case
    retrieved: list[str]
    recall: float
    precision: float
    mrr: float
    ndcg: float
    leaked: list[str]
    latency_ms: float

    @property
    def passed(self) -> bool:
        """A case passes when it found something wanted and leaked nothing."""
        return self.recall > 0 and not self.leaked

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.case.query,
            "kind": self.case.kind,
            "expected": list(self.case.expected),
            "retrieved": list(self.retrieved),
            "recall": round(self.recall, 4),
            "precision": round(self.precision, 4),
            "mrr": round(self.mrr, 4),
            "ndcg": round(self.ndcg, 4),
            "leaked": list(self.leaked),
            "latency_ms": round(self.latency_ms, 2),
            "passed": self.passed,
        }


@dataclass
class Report:
    results: list[CaseResult]
    k: int = DEFAULT_K
    label: str = ""

    def _mean(self, attr: str) -> float:
        if not self.results:
            return 0.0
        return sum(getattr(r, attr) for r in self.results) / len(self.results)

    @property
    def recall(self) -> float:
        return self._mean("recall")

    @property
    def precision(self) -> float:
        return self._mean("precision")

    @property
    def mrr(self) -> float:
        return self._mean("mrr")

    @property
    def ndcg(self) -> float:
        return self._mean("ndcg")

    @property
    def leaks(self) -> int:
        return sum(1 for r in self.results if r.leaked)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    def by_kind(self) -> dict[str, dict[str, float]]:
        """Metrics sliced by case kind — an average hides which class of
        query a change actually broke."""
        buckets: dict[str, list[CaseResult]] = {}
        for r in self.results:
            buckets.setdefault(r.case.kind, []).append(r)
        return {
            kind: {
                "n": len(rs),
                "recall": round(sum(x.recall for x in rs) / len(rs), 4),
                "mrr": round(sum(x.mrr for x in rs) / len(rs), 4),
                "ndcg": round(sum(x.ndcg for x in rs) / len(rs), 4),
            }
            for kind, rs in sorted(buckets.items())
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "k": self.k,
            "cases": len(self.results),
            "passed": self.passed,
            "leaks": self.leaks,
            "recall_at_k": round(self.recall, 4),
            "precision_at_k": round(self.precision, 4),
            "mrr": round(self.mrr, 4),
            "ndcg_at_k": round(self.ndcg, 4),
            "by_kind": self.by_kind(),
            "results": [r.to_dict() for r in self.results],
        }

    def summary(self) -> str:
        lines = [
            f"{self.label or 'retrieval'} @k={self.k}: "
            f"{self.passed}/{len(self.results)} cases passed"
            + (f", {self.leaks} LEAKED" if self.leaks else ""),
            f"  recall@{self.k}={self.recall:.3f}  mrr={self.mrr:.3f}  "
            f"ndcg@{self.k}={self.ndcg:.3f}  precision@{self.k}={self.precision:.3f}",
        ]
        for kind, m in self.by_kind().items():
            lines.append(
                f"    {kind:<14} n={m['n']:<3} recall={m['recall']:.3f} "
                f"mrr={m['mrr']:.3f} ndcg={m['ndcg']:.3f}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _identifiers(hit) -> set[str]:
    """Every name a golden set might refer to this passage by."""
    chunk = getattr(hit, "chunk", None)
    if chunk is None:
        return set()
    return {
        str(getattr(chunk, "doc_id", "")),
        str(getattr(chunk, "source_ref", "")),
        str(getattr(chunk, "title", "")),
    } - {""}


def score_case(case: Case, hits: list, k: int) -> tuple[float, float, float, float, list[str]]:
    """Recall@k, precision@k, MRR, nDCG@k, and anything leaked."""
    top = hits[:k]
    wanted = set(case.expected)
    forbidden = set(case.forbidden)

    # Relevance is per DOCUMENT, not per chunk. Several chunks of the
    # same document can legitimately be retrieved; crediting each one
    # inflates DCG above the ideal and makes the metric meaningless
    # (an nDCG above 1.0 is the tell). Only the highest-ranked chunk of
    # each expected document counts.
    matched_ranks: list[int] = []
    credited: set[str] = set()
    leaked: list[str] = []
    for rank, hit in enumerate(top, start=1):
        ids = _identifiers(hit)
        newly = (ids & wanted) - credited
        if newly:
            credited |= newly
            matched_ranks.append(rank)
        if ids & forbidden:
            leaked.append(next(iter(sorted(ids & forbidden))))

    # A forbidden document anywhere in the full result set is a leak, not
    # just in the top k — the caller still receives it.
    for hit in hits[k:]:
        ids = _identifiers(hit)
        if ids & forbidden:
            leaked.append(next(iter(sorted(ids & forbidden))))

    recall = len(credited) / len(wanted) if wanted else 1.0
    # Precision counts retrieved slots that were wanted, so a duplicate
    # chunk of a wanted document is not "wrong" — but it is not a second
    # hit either.
    precision = len(matched_ranks) / len(top) if top else 0.0
    mrr = 1.0 / matched_ranks[0] if matched_ranks else 0.0

    dcg = sum(1.0 / math.log2(rank + 1) for rank in matched_ranks)
    ideal = sum(
        1.0 / math.log2(i + 1) for i in range(1, min(len(wanted), k) + 1)
    )
    ndcg = (dcg / ideal) if ideal else 0.0

    return recall, precision, mrr, ndcg, sorted(set(leaked))


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def evaluate(
    cases: list[Case],
    *,
    retrieve,
    k: int = DEFAULT_K,
    label: str = "",
) -> Report:
    """Run every case through ``retrieve(query, k)`` and score it.

    ``retrieve`` is injected so the same harness can measure a config
    change, a reranker, or a different backend without knowing about
    any of them.
    """
    results: list[CaseResult] = []
    for case in cases:
        started = time.perf_counter()
        try:
            hits = retrieve(case.query, k) or []
        except Exception:  # noqa: BLE001
            log.exception("retrieval failed for %r", case.query)
            hits = []
        latency = (time.perf_counter() - started) * 1000
        recall, precision, mrr, ndcg, leaked = score_case(case, hits, k)
        results.append(
            CaseResult(
                case=case,
                retrieved=[next(iter(sorted(_identifiers(h))), "") for h in hits[:k]],
                recall=recall,
                precision=precision,
                mrr=mrr,
                ndcg=ndcg,
                leaked=leaked,
                latency_ms=latency,
            )
        )
    return Report(results=results, k=k, label=label)


def compare(baseline: Report, candidate: Report) -> dict[str, Any]:
    """Did the candidate actually help?

    Reports per-metric deltas and, importantly, the cases that
    **regressed** — an average can improve while a whole class of query
    gets worse, which is the failure mode that reaches users.
    """
    by_query = {r.case.query: r for r in baseline.results}
    regressed, improved = [], []
    for r in candidate.results:
        before = by_query.get(r.case.query)
        if before is None:
            continue
        if r.ndcg < before.ndcg - 1e-9 or (before.recall > 0 and r.recall == 0):
            regressed.append(
                {"query": r.case.query, "kind": r.case.kind,
                 "ndcg": [round(before.ndcg, 4), round(r.ndcg, 4)]}
            )
        elif r.ndcg > before.ndcg + 1e-9:
            improved.append(
                {"query": r.case.query, "kind": r.case.kind,
                 "ndcg": [round(before.ndcg, 4), round(r.ndcg, 4)]}
            )
    return {
        "baseline": baseline.label,
        "candidate": candidate.label,
        "delta": {
            "recall": round(candidate.recall - baseline.recall, 4),
            "mrr": round(candidate.mrr - baseline.mrr, 4),
            "ndcg": round(candidate.ndcg - baseline.ndcg, 4),
            "precision": round(candidate.precision - baseline.precision, 4),
            "leaks": candidate.leaks - baseline.leaks,
        },
        "improved": improved,
        "regressed": regressed,
        "verdict": (
            "regression"
            if candidate.leaks > baseline.leaks
            or candidate.ndcg < baseline.ndcg - 0.01
            else "improvement"
            if candidate.ndcg > baseline.ndcg + 0.01
            else "no material change"
        ),
    }


def load_cases(path: str) -> list[Case]:
    """Read a golden set from JSON."""
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    entries = raw.get("cases", raw) if isinstance(raw, dict) else raw
    return [
        Case(
            query=str(e["query"]),
            expected=list(e.get("expected") or []),
            forbidden=list(e.get("forbidden") or []),
            kind=str(e.get("kind") or "general"),
            note=str(e.get("note") or ""),
        )
        for e in entries
        if e.get("query")
    ]


__all__ = [
    "Case",
    "CaseResult",
    "Report",
    "evaluate",
    "compare",
    "score_case",
    "load_cases",
]
