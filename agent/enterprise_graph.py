"""
Enterprise vector knowledge graph — what the organisation *is*, and how
each user relates to it.

Where :mod:`agent.rag` retrieves *passages*, this module holds
**structured understanding**: the entities an enterprise is made of
(people, teams, customers, products, systems, projects, policies) and
the typed, time-bounded relationships between them — plus the edges
that connect a **user** to those entities, which is what makes recall
personal rather than generic.

Three things make this a *vector* knowledge graph rather than a table
of triples:

1. **Entities are embedded.** :func:`resolve_entity` finds the entity a
   phrase refers to by cosine similarity, so "the finance team", "Finance",
   and "FinOps" can resolve to one node without an exact-match lookup.
   Alias lists sharpen this further.

2. **Edges are temporal.** Every edge carries ``valid_from`` /
   ``valid_to``. Facts are *superseded*, never silently overwritten, so
   "who owns Project Atlas" has a correct answer for any point in time
   and a stale fact can never masquerade as current. Temporal validity
   is the difference between a graph that ages well and one that rots.

3. **Scopes compose.** Every node and edge belongs to a scope:
   ``enterprise`` (shared organisational truth) or ``user:<email>``
   (what Alex knows about working with one person). Queries read the
   **union** of the scopes a caller may see, so a user's private context
   is layered on top of enterprise knowledge without leaking sideways
   to another user.

Dev/test stores the graph in memory on ``alex.testing.get_fakes()``;
production writes to Lakebase (``kg_entities`` / ``kg_edges_v2``) and
mirrors to ``media.control.kg_mirror``.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from alex.config import get_settings
from alex.testing import get_fakes

log = logging.getLogger("agent.enterprise_graph")

ENTERPRISE_SCOPE = "enterprise"
DEFAULT_HOPS = 2
MAX_HOPS = 4
MAX_EDGES = 200
RESOLVE_THRESHOLD = 0.25

# Entity types the extractor is asked to use. Keeping the vocabulary
# closed keeps the graph queryable; "other" is the escape hatch.
ENTITY_TYPES = (
    "person", "team", "customer", "product", "project",
    "system", "policy", "metric", "document", "other",
)

EXTRACTION_PROMPT = """You extract an enterprise knowledge graph from text.

Return JSON only, in this exact shape:
{"entities": [{"name": "...", "type": "person|team|customer|product|project|system|policy|metric|document|other", "summary": "one line"}],
 "edges": [{"src": "...", "rel": "owns|reports_to|works_on|uses|depends_on|part_of|responsible_for|related_to", "dst": "...", "evidence": "<short quote>", "confidence": 0.0-1.0}]}

Rules:
- Use names exactly as they appear in the text.
- Only assert relationships the text actually states.
- Omit anything you are unsure about rather than guessing.
"""

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_\-]*")
_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "what", "who",
    "how", "why", "when", "where", "our", "are", "was", "were", "has",
    "have", "does", "did", "about", "into", "over", "all", "any", "can",
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Entity:
    entity_id: str
    name: str
    etype: str = "other"
    summary: str = ""
    scope: str = ENTERPRISE_SCOPE
    aliases: list[str] = field(default_factory=list)
    embedding: list[float] = field(default_factory=list)
    source_ref: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "name": self.name,
            "type": self.etype,
            "summary": self.summary,
            "scope": self.scope,
            "aliases": list(self.aliases),
            "source_ref": self.source_ref,
        }


@dataclass
class Edge:
    edge_id: str
    src: str
    rel: str
    dst: str
    scope: str = ENTERPRISE_SCOPE
    evidence: str = ""
    confidence: float = 0.7
    source_ref: str = ""
    valid_from: float = field(default_factory=time.time)
    valid_to: float | None = None
    superseded_by: str | None = None

    @property
    def current(self) -> bool:
        return self.valid_to is None and self.superseded_by is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "src": self.src,
            "rel": self.rel,
            "dst": self.dst,
            "scope": self.scope,
            "evidence": self.evidence,
            "confidence": round(float(self.confidence), 3),
            "source_ref": self.source_ref,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "current": self.current,
        }


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


class _DevGraph:
    """In-memory graph. ``version`` increments on every write so the
    adjacency cache below invalidates the moment a fact changes.

    ``token`` is a per-instance identity folded into the cache keys:
    ``reset_fakes()`` installs a fresh graph whose version restarts at
    0, and a version-only key would serve the *previous* graph's
    adjacency to the new one — test B traversing test A's edges, or a
    dev session reading another session's graph after a reload.
    """

    def __init__(self) -> None:
        self.entities: dict[str, Entity] = {}
        self.edges: dict[str, Edge] = {}
        self.version = 0
        self.token = uuid.uuid4().hex

    def entities_in(self, scopes: list[str]) -> list[Entity]:
        return [e for e in self.entities.values() if e.scope in scopes]

    def edges_in(self, scopes: list[str]) -> list[Edge]:
        return [e for e in self.edges.values() if e.scope in scopes]

    def touch(self) -> None:
        self.version += 1


def _dev_graph() -> _DevGraph:
    fk = get_fakes()
    g = getattr(fk, "_enterprise_graph", None)
    if g is None:
        g = _DevGraph()
        fk._enterprise_graph = g
    return g


def _is_dev() -> bool:
    return get_settings().mode == "dev"


def _embed(text: str) -> list[float]:
    # Shares :mod:`agent.rag`'s memoised embedder, so a query embedded
    # for retrieval is not embedded again for entity resolution.
    from agent.rag import _embed as _cached_embed

    return _cached_embed(text)


# Adjacency is rebuilt only when the graph changes. Traversal is the
# hot path (every turn's context block walks it), and rebuilding the
# index per call made it O(edges) even for a one-hop lookup.
_ADJACENCY_CACHE: dict[tuple, tuple[int, dict[str, list[Edge]]]] = {}
_ADJACENCY_CACHE_MAX = 16

def _adjacency(
    conn, scopes: list[str], include_history: bool
) -> dict[str, list[Edge]]:
    key = (tuple(sorted(scopes)), include_history)
    if _is_dev():
        g = _dev_graph()
        version, cache_key = g.version, (g.token, *key)
    else:
        version, cache_key = None, key
    if version is not None:
        cached = _ADJACENCY_CACHE.get(cache_key)
        if cached and cached[0] == version:
            return cached[1]
    edges = _all_edges(conn, scopes)
    if not include_history:
        edges = [e for e in edges if e.current]
    index: dict[str, list[Edge]] = {}
    for e in edges:
        index.setdefault(_norm(e.src), []).append(e)
        index.setdefault(_norm(e.dst), []).append(e)
    for bucket in index.values():
        bucket.sort(key=lambda x: -x.confidence)
    if version is not None:
        if len(_ADJACENCY_CACHE) >= _ADJACENCY_CACHE_MAX:
            _ADJACENCY_CACHE.clear()
        _ADJACENCY_CACHE[cache_key] = (version, index)
    return index


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(x * x for x in b)) or 1e-9
    return dot / (na * nb)


def _norm(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").strip()).lower()


# Entity embeddings as an L2-normalised matrix, cached per graph version
# so resolving a phrase is one matrix-vector product rather than a
# Python loop over every node.
_ENTITY_MATRIX_CACHE: dict[tuple, tuple[int, Any]] = {}


def _entity_matrix(scopes: list[str], pool: list[Entity]):
    try:
        import numpy as np
    except ImportError:
        return None
    dims = {len(e.embedding) for e in pool if e.embedding}
    if len(dims) != 1:
        return None
    if _is_dev():
        g = _dev_graph()
        version, cache_key = g.version, (g.token, tuple(sorted(scopes)))
    else:
        version, cache_key = None, (tuple(sorted(scopes)),)
    if version is not None:
        cached = _ENTITY_MATRIX_CACHE.get(cache_key)
        if cached and cached[0] == version and cached[1].shape[0] == len(pool):
            return cached[1]
    width = dims.copy().pop()
    mat = np.array(
        [e.embedding if e.embedding else [0.0] * width for e in pool],
        dtype="float32",
    )
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat = mat / norms
    if version is not None:
        if len(_ENTITY_MATRIX_CACHE) > 16:
            _ENTITY_MATRIX_CACHE.clear()
        _ENTITY_MATRIX_CACHE[cache_key] = (version, mat)
    return mat


def _similarities(
    conn, scopes: list[str], pool: list[Entity], qe: list[float]
) -> list[float]:
    """Cosine similarity of ``qe`` against every entity in ``pool``."""
    mat = _entity_matrix(scopes, pool)
    if mat is not None and qe and len(qe) == mat.shape[1]:
        import numpy as np

        q = np.asarray(qe, dtype="float32")
        qn = float(np.linalg.norm(q)) or 1e-9
        return (mat @ (q / qn)).tolist()
    return [_cosine(qe, e.embedding) for e in pool]


# ---------------------------------------------------------------------------
# Scopes
# ---------------------------------------------------------------------------


def user_scope(user_email: str) -> str:
    return f"user:{(user_email or '').strip().lower()}"


def visible_scopes(user_email: str = "", extra: list[str] | None = None) -> list[str]:
    """Enterprise truth plus this user's private layer."""
    scopes = [ENTERPRISE_SCOPE]
    if user_email:
        scopes.append(user_scope(user_email))
    for s in extra or []:
        if s and s not in scopes:
            scopes.append(s)
    return scopes


# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------


def upsert_entity(
    conn,
    name: str,
    *,
    etype: str = "other",
    summary: str = "",
    scope: str = ENTERPRISE_SCOPE,
    aliases: list[str] | None = None,
    source_ref: str = "",
) -> Entity:
    """Create or update an entity node (idempotent on scope+name)."""
    name = (name or "").strip()
    if not name:
        raise ValueError("entity name is required")
    if etype not in ENTITY_TYPES:
        etype = "other"
    existing = get_entity(conn, name, scopes=[scope])
    text = f"{name} {etype} {summary} {' '.join(aliases or [])}".strip()
    if existing:
        existing.etype = etype if etype != "other" else existing.etype
        existing.summary = summary or existing.summary
        for a in aliases or []:
            if a and a not in existing.aliases:
                existing.aliases.append(a)
        existing.embedding = _embed(
            f"{existing.name} {existing.etype} {existing.summary} "
            f"{' '.join(existing.aliases)}"
        )
        existing.updated_at = time.time()
        _put_entity(conn, existing)
        return existing
    ent = Entity(
        entity_id="ent-" + uuid.uuid4().hex[:16],
        name=name,
        etype=etype,
        summary=summary,
        scope=scope,
        aliases=list(aliases or []),
        embedding=_embed(text),
        source_ref=source_ref,
    )
    _put_entity(conn, ent)
    return ent


def _put_entity(conn, ent: Entity) -> None:
    if _is_dev():
        g = _dev_graph()
        g.entities[ent.entity_id] = ent
        g.touch()
        return
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO kg_entities
              (entity_id, name, etype, summary, scope, aliases, embedding,
               source_ref, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s::vector, %s, now(), now())
            ON CONFLICT (scope, lower(name)) DO UPDATE
              SET etype = EXCLUDED.etype, summary = EXCLUDED.summary,
                  aliases = EXCLUDED.aliases, embedding = EXCLUDED.embedding,
                  updated_at = now()
            """,
            (
                ent.entity_id, ent.name, ent.etype, ent.summary, ent.scope,
                json.dumps(ent.aliases), ent.embedding, ent.source_ref,
            ),
        )


def get_entity(conn, name: str, *, scopes: list[str] | None = None) -> Entity | None:
    """Exact (case-insensitive) lookup by name or alias."""
    key = _norm(name)
    if not key:
        return None
    scopes = scopes if scopes is not None else [ENTERPRISE_SCOPE]
    for ent in _all_entities(conn, scopes):
        if _norm(ent.name) == key or any(_norm(a) == key for a in ent.aliases):
            return ent
    return None


def _pg_vector(raw) -> list[float]:
    """Parse pgvector's text representation into a list (see rag._pg_vector)."""
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [float(x) for x in raw]
    try:
        parsed = json.loads(str(raw))
        return [float(x) for x in parsed] if isinstance(parsed, list) else []
    except (ValueError, TypeError):
        return []


def _all_entities(conn, scopes: list[str]) -> list[Entity]:
    if _is_dev():
        return _dev_graph().entities_in(scopes)
    # The embedding column is selected: without it every prod entity
    # resolves by exact/alias match only — the embedding fallback in
    # resolve_entity silently scores every candidate 0.
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT entity_id, name, etype, summary, scope, aliases, source_ref,
                   embedding
              FROM kg_entities WHERE scope = ANY(%s)
            """,
            (scopes,),
        )
        return [
            Entity(
                entity_id=r[0], name=r[1], etype=r[2] or "other",
                summary=r[3] or "", scope=r[4],
                aliases=json.loads(r[5]) if r[5] else [],
                source_ref=r[6] or "",
                embedding=_pg_vector(r[7]),
            )
            for r in cur.fetchall()
        ]


def resolve_entity(
    conn, text: str, *, scopes: list[str] | None = None, k: int = 3
) -> list[dict[str, Any]]:
    """Find the entities a phrase most likely refers to.

    Tries exact/alias match first (a name the caller typed is
    authoritative), then falls back to embedding similarity so
    paraphrases still land on the right node.
    """
    text = (text or "").strip()
    if not text:
        return []
    scopes = scopes if scopes is not None else [ENTERPRISE_SCOPE]
    pool = _all_entities(conn, scopes)
    if not pool:
        return []
    exact = get_entity(conn, text, scopes=scopes)
    out: list[dict[str, Any]] = []
    if exact:
        out.append({**exact.to_dict(), "match": "exact", "similarity": 1.0})
    # Token overlap catches "the Atlas project" -> "Project Atlas".
    terms = {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS}
    qe = _embed(text)
    sims = _similarities(conn, scopes, pool, qe)
    scored: list[tuple[float, Entity, str]] = []
    for i, ent in enumerate(pool):
        if exact and ent.entity_id == exact.entity_id:
            continue
        name_terms = set(_TOKEN_RE.findall(ent.name.lower()))
        overlap = len(terms & name_terms) / max(len(name_terms), 1)
        sim = sims[i]
        combined = max(sim, overlap)
        if combined >= RESOLVE_THRESHOLD:
            scored.append((combined, ent, "lexical" if overlap >= sim else "vector"))
    scored.sort(key=lambda x: -x[0])
    for sim, ent, how in scored[: max(0, k - len(out))]:
        out.append({**ent.to_dict(), "match": how, "similarity": round(sim, 4)})
    return out


# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------


def upsert_edge(
    conn,
    src: str,
    rel: str,
    dst: str,
    *,
    scope: str = ENTERPRISE_SCOPE,
    evidence: str = "",
    confidence: float = 0.7,
    source_ref: str = "",
    supersede_conflicting: bool = True,
) -> Edge:
    """Assert a relationship, closing the validity of any it replaces.

    When ``supersede_conflicting`` is set (the default) an existing
    current edge with the same ``src`` and ``rel`` but a different
    ``dst`` is closed — ``valid_to`` is stamped and ``superseded_by``
    points at the new edge. The old fact stays queryable as history and
    stops being returned as current.
    """
    src, dst, rel = (src or "").strip(), (dst or "").strip(), (rel or "").strip()
    if not (src and rel and dst):
        raise ValueError("src, rel and dst are all required")
    now = time.time()
    edge = Edge(
        edge_id="edge-" + uuid.uuid4().hex[:16],
        src=src, rel=rel, dst=dst, scope=scope,
        evidence=evidence, confidence=float(confidence),
        source_ref=source_ref, valid_from=now,
    )
    for other in _all_edges(conn, [scope]):
        if not other.current:
            continue
        same = _norm(other.src) == _norm(src) and _norm(other.rel) == _norm(rel)
        if same and _norm(other.dst) == _norm(dst):
            # Re-asserting the same fact: refresh confidence, keep identity.
            other.confidence = max(other.confidence, float(confidence))
            other.evidence = evidence or other.evidence
            _put_edge(conn, other)
            return other
        if same and supersede_conflicting:
            other.valid_to = now
            other.superseded_by = edge.edge_id
            _put_edge(conn, other)
    _put_edge(conn, edge)
    return edge


def _put_edge(conn, edge: Edge) -> None:
    if _is_dev():
        g = _dev_graph()
        g.edges[edge.edge_id] = edge
        g.touch()
        return
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO kg_edges_v2
              (edge_id, src, rel, dst, scope, evidence, confidence,
               source_ref, valid_from, valid_to, superseded_by)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s,
                    to_timestamp(%s), to_timestamp(%s), %s)
            ON CONFLICT (edge_id) DO UPDATE
              SET confidence = EXCLUDED.confidence,
                  evidence = EXCLUDED.evidence,
                  valid_to = EXCLUDED.valid_to,
                  superseded_by = EXCLUDED.superseded_by
            """,
            (
                edge.edge_id, edge.src, edge.rel, edge.dst, edge.scope,
                edge.evidence, edge.confidence, edge.source_ref,
                edge.valid_from, edge.valid_to, edge.superseded_by,
            ),
        )


def _all_edges(conn, scopes: list[str]) -> list[Edge]:
    if _is_dev():
        return _dev_graph().edges_in(scopes)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT edge_id, src, rel, dst, scope, evidence, confidence,
                   source_ref, extract(epoch from valid_from),
                   extract(epoch from valid_to), superseded_by
              FROM kg_edges_v2 WHERE scope = ANY(%s)
            """,
            (scopes,),
        )
        return [
            Edge(
                edge_id=r[0], src=r[1], rel=r[2], dst=r[3], scope=r[4],
                evidence=r[5] or "", confidence=float(r[6] or 0.0),
                source_ref=r[7] or "", valid_from=float(r[8] or 0.0),
                valid_to=float(r[9]) if r[9] else None, superseded_by=r[10],
            )
            for r in cur.fetchall()
        ]


def neighbors(
    conn,
    entity: str,
    *,
    scopes: list[str] | None = None,
    hops: int = DEFAULT_HOPS,
    limit: int = 40,
    include_history: bool = False,
    rel: str = "",
) -> list[dict[str, Any]]:
    """Breadth-first traversal out from ``entity``.

    Each returned edge carries the ``hop`` distance at which it was
    reached, so a caller can weight direct facts above distant ones.
    Only *current* edges are traversed unless ``include_history``.
    """
    start = _norm(entity)
    if not start:
        return []
    scopes = scopes if scopes is not None else [ENTERPRISE_SCOPE]
    hops = max(1, min(int(hops), MAX_HOPS))
    adjacency = _adjacency(conn, scopes, include_history)
    if start not in adjacency:
        return []
    want_rel = _norm(rel) if rel else ""

    seen_nodes = {start}
    seen_edges: set[str] = set()
    frontier = [start]
    out: list[dict[str, Any]] = []
    for hop in range(1, hops + 1):
        nxt: list[str] = []
        for node in frontier:
            # Buckets are pre-sorted by confidence when the index is built.
            for e in adjacency.get(node, ()):
                if e.edge_id in seen_edges:
                    continue
                seen_edges.add(e.edge_id)
                if want_rel and _norm(e.rel) != want_rel:
                    # Filtered out of the result, but still traversable so
                    # a relation filter cannot disconnect the graph.
                    for side in (_norm(e.src), _norm(e.dst)):
                        if side not in seen_nodes:
                            seen_nodes.add(side)
                            nxt.append(side)
                    continue
                out.append({**e.to_dict(), "hop": hop})
                if len(out) >= min(limit, MAX_EDGES):
                    return out
                for side in (_norm(e.src), _norm(e.dst)):
                    if side not in seen_nodes:
                        seen_nodes.add(side)
                        nxt.append(side)
        if not nxt:
            break
        frontier = nxt
    return out


def subgraph(
    conn,
    query: str,
    *,
    user_email: str = "",
    hops: int = DEFAULT_HOPS,
    limit: int = 24,
) -> dict[str, Any]:
    """Resolve the entities in a question and return their neighbourhood.

    This is the graph half of retrieval: it answers "what does the
    organisation know that is *connected* to this?", which vector search
    over passages cannot do.
    """
    scopes = visible_scopes(user_email)
    roots = resolve_entity(conn, query, scopes=scopes, k=3)
    edges: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root in roots:
        for e in neighbors(
            conn, root["name"], scopes=scopes, hops=hops, limit=limit
        ):
            if e["edge_id"] in seen:
                continue
            seen.add(e["edge_id"])
            edges.append(e)
            if len(edges) >= limit:
                break
        if len(edges) >= limit:
            break
    return {"query": query, "entities": roots, "edges": edges}


# ---------------------------------------------------------------------------
# The user <-> enterprise memory graph
# ---------------------------------------------------------------------------


def link_user(
    conn,
    user_email: str,
    rel: str,
    entity: str,
    *,
    evidence: str = "",
    confidence: float = 0.8,
    etype: str = "other",
) -> dict[str, Any]:
    """Record how a user relates to an enterprise entity.

    The edge is written in the user's private scope (so it never leaks
    to a colleague) while pointing at a node in the shared enterprise
    scope — this is precisely the seam that makes recall personal:
    "your team", "your customer", "the project you own".
    """
    scope = user_scope(user_email)
    upsert_entity(
        conn, user_email, etype="person", scope=scope,
        summary="The signed-in user.",
    )
    if not get_entity(conn, entity, scopes=[ENTERPRISE_SCOPE]):
        upsert_entity(conn, entity, etype=etype, scope=ENTERPRISE_SCOPE)
    edge = upsert_edge(
        conn, user_email, rel, entity, scope=scope,
        evidence=evidence, confidence=confidence,
    )
    return edge.to_dict()


def user_context(
    conn, user_email: str, *, limit: int = 12
) -> list[dict[str, Any]]:
    """Everything the graph knows about this user, most confident first."""
    scope = user_scope(user_email)
    edges = [e for e in _all_edges(conn, [scope]) if e.current]
    edges.sort(key=lambda e: (-e.confidence, -e.valid_from))
    return [e.to_dict() for e in edges[:limit]]


def context_block(
    conn, query: str, *, user_email: str = "", max_chars: int = 1200
) -> str:
    """Render graph facts for the prompt: user layer first, then enterprise."""
    parts: list[str] = []
    personal = user_context(conn, user_email, limit=6) if user_email else []
    if personal:
        parts.append(
            "ABOUT THIS USER: "
            + "; ".join(f"{e['src']} {e['rel']} {e['dst']}" for e in personal)
        )
    sub = subgraph(conn, query, user_email=user_email, limit=10)
    if sub["edges"]:
        facts = "; ".join(
            f"{e['src']} {e['rel']} {e['dst']}" for e in sub["edges"]
        )
        parts.append(f"RELATED ENTERPRISE FACTS: {facts}")
    block = "\n".join(parts)
    return block[:max_chars]


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def extract_and_store(
    conn,
    text: str,
    *,
    llm_invoke,
    scope: str = ENTERPRISE_SCOPE,
    source_ref: str = "",
) -> dict[str, Any]:
    """Extract entities/edges from text with the LLM and store them.

    ``llm_invoke`` is a callable taking a message list and returning the
    model's text (the same seam :mod:`agent.tools.impls` uses), so this
    works against the dev model and Databricks alike. A malformed
    response yields an empty graph rather than an exception — extraction
    is best-effort enrichment, never a hard dependency of a turn.
    """
    if not (text or "").strip():
        return {"entities": 0, "edges": 0}
    try:
        raw = llm_invoke(
            [
                {"role": "system", "content": EXTRACTION_PROMPT},
                {"role": "user", "content": text[:6000]},
            ]
        )
        payload = _parse_json(raw)
    except Exception:  # noqa: BLE001
        log.debug("graph extraction failed", exc_info=True)
        return {"entities": 0, "edges": 0}

    n_ent = 0
    for ent in payload.get("entities") or []:
        if not isinstance(ent, dict) or not (ent.get("name") or "").strip():
            continue
        try:
            upsert_entity(
                conn,
                str(ent["name"]),
                etype=str(ent.get("type") or "other"),
                summary=str(ent.get("summary") or ""),
                scope=scope,
                source_ref=source_ref,
            )
            n_ent += 1
        except ValueError:
            continue
    n_edge = 0
    for e in payload.get("edges") or []:
        if not isinstance(e, dict):
            continue
        try:
            upsert_edge(
                conn,
                str(e.get("src") or ""),
                str(e.get("rel") or "related_to"),
                str(e.get("dst") or ""),
                scope=scope,
                evidence=str(e.get("evidence") or "")[:500],
                confidence=float(e.get("confidence") or 0.6),
                source_ref=source_ref,
            )
            n_edge += 1
        except (ValueError, TypeError):
            continue
    log.info("graph extraction stored %d entities / %d edges", n_ent, n_edge)
    return {"entities": n_ent, "edges": n_edge}


def _parse_json(raw: Any) -> dict[str, Any]:
    """Parse a model response that should be JSON, tolerating fences."""
    if isinstance(raw, dict):
        return raw
    text = str(raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\s*|\s*```$", "", text, flags=re.IGNORECASE)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return {}
    try:
        parsed = json.loads(text[start : end + 1])
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def stats(conn, *, scopes: list[str] | None = None) -> dict[str, Any]:
    """Counts for the Transparency surface."""
    scopes = scopes if scopes is not None else [ENTERPRISE_SCOPE]
    ents = _all_entities(conn, scopes)
    edges = _all_edges(conn, scopes)
    by_type: dict[str, int] = {}
    for e in ents:
        by_type[e.etype] = by_type.get(e.etype, 0) + 1
    return {
        "entities": len(ents),
        "edges": len(edges),
        "current_edges": sum(1 for e in edges if e.current),
        "historical_edges": sum(1 for e in edges if not e.current),
        "entity_types": by_type,
        "scopes": scopes,
    }
