"""
The best-practice library — how this organisation does things.

This is deliberately **not** RAG. Retrieval answers "what does the
company know that resembles your question?" and hands the model some
text that may or may not be used. A practice answers a different
question: "how must this be done here?" — and the answer is normative.

The distinction shows up in four places:

* **Precedence.** Practices enter the prompt in the T1 directive slice,
  directly under the constitution and above retrieved passages. A
  retrieved document is information; a practice is an instruction.

* **Matching, not similarity.** A practice declares what it *applies
  to*: subject phrases, and — more importantly — **capabilities**. A
  practice bound to ``email.send`` fires whenever the agent is about to
  send mail, however the user phrased the request. Semantic similarity
  alone would miss that, which is exactly how "we always CC legal on
  contract mail" gets quietly skipped.

* **Authority.** ``mandatory`` practices must be followed and must be
  cited; ``recommended`` should be followed unless there is a reason;
  ``advisory`` is context. The prompt states the difference explicitly
  so the model does not flatten them into suggestions.

* **Compliance is checkable.** :func:`compliance_report` inspects an
  answer for the mandatory practices that were in play, so "did it
  follow policy?" is answerable after the fact rather than assumed.

Practices are versioned and superseded, never edited in place, so
"which version of the policy was in force when this was decided?" has
an answer. Visibility runs through :mod:`agent.access` like everything
else.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("agent.practices")

# Authority levels, strongest first. Order matters: it drives both
# prompt ordering and the trim order when the budget is tight.
MANDATORY = "mandatory"
RECOMMENDED = "recommended"
ADVISORY = "advisory"
AUTHORITIES = (MANDATORY, RECOMMENDED, ADVISORY)
AUTHORITY_RANK = {name: i for i, name in enumerate(AUTHORITIES)}

ENTERPRISE_SCOPE = "enterprise"
DEFAULT_CLASSIFICATION = "internal"

MAX_IN_PROMPT = 6
GUIDANCE_BUDGET_CHARS = 2400

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_\-]*")
_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "what", "who",
    "how", "why", "when", "where", "our", "are", "was", "were", "has",
    "have", "does", "did", "about", "into", "over", "all", "any", "can",
    "should", "would", "please", "need", "want", "make", "get",
}


@dataclass
class Practice:
    """One rule of the house."""

    practice_id: str
    title: str
    guidance: str
    authority: str = RECOMMENDED
    # What situations this governs.
    applies_to: list[str] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)
    # Concrete help for the model.
    rationale: str = ""
    examples: list[str] = field(default_factory=list)
    anti_patterns: list[str] = field(default_factory=list)
    # Governance.
    scope: str = ENTERPRISE_SCOPE
    classification: str = DEFAULT_CLASSIFICATION
    owner: str = ""
    version: int = 1
    supersedes: str = ""
    superseded_by: str = ""
    active: bool = True
    source_ref: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    embedding: list[float] = field(default_factory=list)

    @property
    def is_mandatory(self) -> bool:
        return self.authority == MANDATORY

    @property
    def current(self) -> bool:
        return self.active and not self.superseded_by

    def to_dict(self, *, with_body: bool = True) -> dict[str, Any]:
        out = {
            "practice_id": self.practice_id,
            "title": self.title,
            "authority": self.authority,
            "applies_to": list(self.applies_to),
            "capabilities": list(self.capabilities),
            "scope": self.scope,
            "classification": self.classification,
            "owner": self.owner,
            "version": self.version,
            "active": self.active,
            "current": self.current,
            "superseded_by": self.superseded_by,
            "source_ref": self.source_ref,
        }
        if with_body:
            out.update(
                {
                    "guidance": self.guidance,
                    "rationale": self.rationale,
                    "examples": list(self.examples),
                    "anti_patterns": list(self.anti_patterns),
                }
            )
        return out


@dataclass
class Match:
    """A practice that applies, and why."""

    practice: Practice
    score: float
    reason: str  # 'capability' | 'subject' | 'semantic' | 'always'

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.practice.to_dict(),
            "score": round(self.score, 4),
            "matched_on": self.reason,
        }


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


class _DevLibrary:
    def __init__(self) -> None:
        self.practices: dict[str, Practice] = {}
        self.version = 0

    def put(self, p: Practice) -> None:
        self.practices[p.practice_id] = p
        self.version += 1

    def all(self) -> list[Practice]:
        return list(self.practices.values())


def _dev_library() -> _DevLibrary:
    from alex.testing import get_fakes

    fk = get_fakes()
    lib = getattr(fk, "_practice_library", None)
    if lib is None:
        lib = _DevLibrary()
        fk._practice_library = lib
    return lib


def _is_dev() -> bool:
    from alex.config import get_settings

    return get_settings().mode == "dev"


def _embed(text: str) -> list[float]:
    from agent.rag import _embed as cached

    return cached(text)


def _tokens(text: str) -> set[str]:
    return {
        t for t in _TOKEN_RE.findall((text or "").lower())
        if t not in _STOPWORDS and len(t) > 2
    }


TABLE = "media.control.best_practices"


def upsert(
    sql,
    *,
    title: str,
    guidance: str,
    authority: str = RECOMMENDED,
    applies_to: list[str] | None = None,
    capabilities: list[str] | None = None,
    rationale: str = "",
    examples: list[str] | None = None,
    anti_patterns: list[str] | None = None,
    scope: str = ENTERPRISE_SCOPE,
    classification: str = DEFAULT_CLASSIFICATION,
    owner: str = "",
    practice_id: str = "",
    supersedes: str = "",
    source_ref: str = "",
) -> Practice:
    """Publish a practice.

    Revising a practice **supersedes** rather than edits: pass
    ``supersedes`` with the old id and the previous version stays on
    record, so a past decision can always be judged against the policy
    that was actually in force at the time.
    """
    title = (title or "").strip()
    guidance = (guidance or "").strip()
    if not title or not guidance:
        raise ValueError("a practice needs both a title and guidance")
    if authority not in AUTHORITY_RANK:
        authority = RECOMMENDED

    version = 1
    if supersedes:
        old = get(sql, supersedes)
        if old is not None:
            version = old.version + 1
            old.superseded_by = practice_id or "pending"
            old.active = False
            _put(sql, old)

    practice = Practice(
        practice_id=practice_id or f"bp-{uuid.uuid4().hex[:12]}",
        title=title,
        guidance=guidance,
        authority=authority,
        applies_to=[s.strip() for s in (applies_to or []) if s.strip()],
        capabilities=[c.strip() for c in (capabilities or []) if c.strip()],
        rationale=rationale.strip(),
        examples=[e for e in (examples or []) if e],
        anti_patterns=[a for a in (anti_patterns or []) if a],
        scope=scope,
        classification=classification,
        owner=owner,
        version=version,
        supersedes=supersedes,
        source_ref=source_ref,
    )
    practice.embedding = _embed(
        f"{title}\n{guidance}\n{' '.join(practice.applies_to)}"
    )
    if supersedes:
        old = get(sql, supersedes)
        if old is not None:
            old.superseded_by = practice.practice_id
            _put(sql, old)
    _put(sql, practice)
    log.info(
        "practice published %s v%d (%s)",
        practice.practice_id, practice.version, practice.authority,
    )
    return practice


def _put(sql, p: Practice) -> None:
    """Persist a practice. UPDATE when the row exists, INSERT when it
    does not — retire()/revise() mutate an *existing* row and write it
    back, and a plain INSERT hit the PRIMARY KEY constraint in
    production, making every retirement and revision a 500. Dev's
    in-memory dict masked it: exactly the dev-passes/prod-fails class
    this codebase exists to catch.
    """
    if _is_dev():
        _dev_library().put(p)
        return
    params = {
        "id": p.practice_id, "t": p.title, "g": p.guidance,
        "a": p.authority, "ap": p.applies_to, "cap": p.capabilities,
        "r": p.rationale, "ex": p.examples, "anti": p.anti_patterns,
        "sc": p.scope, "cl": p.classification, "o": p.owner,
        "v": p.version, "sup": p.supersedes, "supby": p.superseded_by,
        "act": p.active, "src": p.source_ref,
    }
    existing = sql(
        f"SELECT practice_id FROM {TABLE} WHERE practice_id = :id",
        {"id": p.practice_id},
    )
    if existing:
        sql(
            f"UPDATE {TABLE} SET "
            "  title = :t, guidance = :g, authority = :a, applies_to = :ap, "
            "  capabilities = :cap, rationale = :r, examples = :ex, "
            "  anti_patterns = :anti, scope = :sc, classification = :cl, "
            "  owner = :o, version = :v, supersedes = :sup, "
            "  superseded_by = :supby, active = :act, source_ref = :src, "
            "  updated_at = current_timestamp() "
            "WHERE practice_id = :id",
            params,
        )
        return
    sql(
        f"INSERT INTO {TABLE} "
        "(practice_id, title, guidance, authority, applies_to, capabilities, "
        " rationale, examples, anti_patterns, scope, classification, owner, "
        " version, supersedes, superseded_by, active, source_ref, updated_at) "
        "VALUES (:id, :t, :g, :a, :ap, :cap, :r, :ex, :anti, :sc, :cl, :o, "
        "        :v, :sup, :supby, :act, :src, current_timestamp())",
        params,
    )


def _all(sql) -> list[Practice]:
    if _is_dev():
        return _dev_library().all()
    try:
        rows = sql(f"SELECT * FROM {TABLE}", {}) or []
    except Exception:  # noqa: BLE001
        log.debug("practice library read failed", exc_info=True)
        return []
    out: list[Practice] = []
    for r in rows:
        out.append(
            Practice(
                practice_id=r.get("practice_id", ""),
                title=r.get("title", ""),
                guidance=r.get("guidance", ""),
                authority=r.get("authority", RECOMMENDED),
                applies_to=list(r.get("applies_to") or []),
                capabilities=list(r.get("capabilities") or []),
                rationale=r.get("rationale", "") or "",
                examples=list(r.get("examples") or []),
                anti_patterns=list(r.get("anti_patterns") or []),
                scope=r.get("scope", ENTERPRISE_SCOPE),
                classification=r.get("classification", DEFAULT_CLASSIFICATION),
                owner=r.get("owner", "") or "",
                version=int(r.get("version") or 1),
                supersedes=r.get("supersedes", "") or "",
                superseded_by=r.get("superseded_by", "") or "",
                active=bool(r.get("active", True)),
                source_ref=r.get("source_ref", "") or "",
            )
        )
    return out


def get(sql, practice_id: str) -> Practice | None:
    for p in _all(sql):
        if p.practice_id == practice_id:
            return p
    return None


def retire(sql, practice_id: str) -> bool:
    """Take a practice out of force without deleting the record."""
    p = get(sql, practice_id)
    if p is None:
        return False
    p.active = False
    p.updated_at = time.time()
    _put(sql, p)
    return True


def visible(sql, principal, *, include_retired: bool = False) -> list[Practice]:
    """Practices this principal may read, newest authority first."""
    from agent import access

    out = []
    for p in _all(sql):
        if not include_retired and not p.current:
            continue
        if not access.can_read(
            principal, scope=p.scope, classification=p.classification
        ).allowed:
            continue
        out.append(p)
    out.sort(key=lambda p: (AUTHORITY_RANK[p.authority], p.title.lower()))
    return out


# ---------------------------------------------------------------------------
# Matching — the part that makes this guidance rather than search
# ---------------------------------------------------------------------------


def _cosine(a: list[float], b: list[float]) -> float:
    import math

    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(x * x for x in b)) or 1e-9
    return dot / (na * nb)


SEMANTIC_FLOOR = 0.28


def match(
    sql,
    principal,
    *,
    query: str = "",
    capabilities_in_play: list[str] | None = None,
    limit: int = MAX_IN_PROMPT,
) -> list[Match]:
    """Which practices govern this situation?

    Three independent triggers, in descending reliability:

    1. **Capability** — the agent is about to use a tool the practice
       governs. This fires regardless of wording and is why a mandatory
       practice cannot be phrased around.
    2. **Subject** — meaningful overlap with the practice's declared
       subjects.
    3. **Semantic** — embedding similarity, for practices whose
       subjects were written loosely.

    A practice with neither subjects nor capabilities is *unconditional*
    (house style, tone, formatting) and always applies.
    """
    caps = {c.strip() for c in (capabilities_in_play or []) if c}
    q_tokens = _tokens(query)
    q_embed = _embed(query) if query.strip() else []

    matches: list[Match] = []
    for p in visible(sql, principal):
        if caps and set(p.capabilities) & caps:
            matches.append(Match(p, 1.0, "capability"))
            continue
        if not p.applies_to and not p.capabilities:
            matches.append(Match(p, 0.5, "always"))
            continue
        if p.capabilities and not p.applies_to:
            # Governs a tool that is not in play — stay out of the prompt.
            continue
        subject_tokens: set[str] = set()
        for subject in p.applies_to:
            subject_tokens |= _tokens(subject)
        hits = q_tokens & subject_tokens
        if hits:
            # Presence is the signal, not proportion. Declaring a subject
            # means "this practice governs that subject", so one declared
            # subject appearing in the request is a match. Scoring by
            # matched/declared would have penalised a practice for listing
            # more subjects — the opposite of what an author intends.
            coverage = len(hits) / max(len(subject_tokens), 1)
            matches.append(Match(p, 0.6 + 0.4 * coverage, "subject"))
            continue
        sim = _cosine(q_embed, p.embedding) if q_embed else 0.0
        if sim >= SEMANTIC_FLOOR:
            matches.append(Match(p, 0.3 + 0.3 * sim, "semantic"))

    # Mandatory first, then score. A mandatory practice must never be
    # crowded out of the prompt by a better-matching advisory one.
    matches.sort(
        key=lambda m: (AUTHORITY_RANK[m.practice.authority], -m.score)
    )
    return matches[:limit]


# ---------------------------------------------------------------------------
# The prompt slice
# ---------------------------------------------------------------------------


def guidance_block(
    matches: list[Match], *, max_chars: int = GUIDANCE_BUDGET_CHARS
) -> str:
    """Render matched practices as directives for the T1 slice.

    The wording is deliberately imperative and states the authority
    difference, because a model given a list of "relevant documents"
    treats every line as optional context. Mandatory practices are
    rendered first and are the last thing dropped when the budget bites.
    """
    if not matches:
        return ""
    mandatory = [m for m in matches if m.practice.is_mandatory]
    others = [m for m in matches if not m.practice.is_mandatory]

    lines: list[str] = [
        "HOW THIS ORGANISATION WORKS (the best-practice library — this is "
        "operating guidance, not background reading):"
    ]
    used = len(lines[0])

    def render(m: Match) -> str:
        p = m.practice
        head = f"- [{p.practice_id}] {p.title} ({p.authority.upper()})"
        body = [f"  {p.guidance}"]
        if p.rationale:
            body.append(f"  Why: {p.rationale}")
        if p.anti_patterns:
            body.append(f"  Do not: {'; '.join(p.anti_patterns[:2])}")
        return "\n".join([head, *body])

    if mandatory:
        lines.append(
            "MANDATORY — you must follow each of these, and name the "
            "practice you applied in your answer:"
        )
        used += len(lines[-1])
        for m in mandatory:
            entry = render(m)
            lines.append(entry)
            used += len(entry) + 1
    if others and used < max_chars:
        lines.append(
            "RECOMMENDED / ADVISORY — follow unless the user's request "
            "gives you a specific reason not to:"
        )
        used += len(lines[-1])
        for m in others:
            entry = render(m)
            if used + len(entry) > max_chars:
                break
            lines.append(entry)
            used += len(entry) + 1
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Compliance
# ---------------------------------------------------------------------------


def compliance_report(
    answer: str, matches: list[Match]
) -> dict[str, Any]:
    """Did the answer honour the mandatory practices that were in play?

    A cheap, deterministic check: a mandatory practice is *cited* when
    its id or title appears in the answer. This is not a judge — it
    catches the common failure (the model silently ignored the policy)
    and feeds the review surface, which is where a human decides.
    """
    text = (answer or "").lower()
    mandatory = [m.practice for m in matches if m.practice.is_mandatory]
    cited, uncited = [], []
    for p in mandatory:
        hit = p.practice_id.lower() in text or p.title.lower() in text
        (cited if hit else uncited).append(
            {"practice_id": p.practice_id, "title": p.title}
        )
    return {
        "mandatory_in_play": len(mandatory),
        "cited": cited,
        "uncited": uncited,
        "compliant": not uncited,
    }
