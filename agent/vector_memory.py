"""
Vector memory: the agent's per-user semantic memory.

Two implementations:
  * Production: pgvector on Lakebase.
  * Dev / test: in-memory dicts through `alex.testing.get_fakes()`.

API:
  store(conn, owner, kind, content, source_ref, embedding=None)
  recall(conn, owner, query_embedding, k=10, threshold=0.7) -> list[MemoryBead]
  decay_and_supersede(conn, owner) -> int   # nightly job

The schema (Lakebase DDL) is owned by `pipelines/memory_consolidation.py`
because it is built at runtime, not at bundle deploy. The Delta mirror
of these rows is in `media.control.vector_memory_mirror`.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
import uuid
from dataclasses import dataclass, field

from alex.config import get_settings
from alex.testing import get_fakes

log = logging.getLogger("agent.vector_memory")


@dataclass
class MemoryBead:
    id: str
    owner_email: str
    kind: str  # 'preference' | 'fact' | 'skill_cue' | 'project'
    content: str
    salience: float = 0.5
    source_ref: str = ""
    embedding: list[float] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_recalled: float | None = None
    superseded_by: str | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "owner_email": self.owner_email,
            "kind": self.kind,
            "content": self.content,
            "salience": self.salience,
            "source_ref": self.source_ref,
            "created_at": self.created_at,
            "last_recalled": self.last_recalled,
            "superseded_by": self.superseded_by,
        }


# ---------------------------------------------------------------------------
# Dev / test: in-memory storage. The shape mirrors the Lakebase schema.
# ---------------------------------------------------------------------------


class _DevStore:
    def __init__(self) -> None:
        self._beads: list[MemoryBead] = []
        self._by_id: dict[str, MemoryBead] = {}

    def add(self, b: MemoryBead) -> None:
        self._beads.append(b)
        self._by_id[b.id] = b

    def get(self, bid: str) -> MemoryBead | None:
        return self._by_id.get(bid)

    def for_owner(self, owner: str) -> list[MemoryBead]:
        return [
            b for b in self._beads if b.owner_email == owner and not b.superseded_by
        ]


def _dev_store() -> _DevStore:
    fk = get_fakes()
    store = getattr(fk, "_vector_memory_store", None)
    if store is None:
        store = _DevStore()
        fk._vector_memory_store = store
    return store


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(x * x for x in b)) or 1e-9
    return dot / (na * nb)


_DEV_EMBED_DIM = 96


# The dev embedding model. A real embedding endpoint, not a hash: the
# hashing trick below can only match shared vocabulary, which means a
# question phrased differently from the document scores zero however
# obviously it is about the same thing. Measured on real enterprise
# documents, that was the single weakest retrieval class.
DEV_EMBED_MODEL = os.environ.get("ALEX_EMBED_MODEL", "nomic-embed-text")
DEV_EMBED_URL = (
    os.environ.get("OLLAMA_EMBED_URL", "").strip()
    or "http://127.0.0.1:11434/api/embed"
)
EMBED_BATCH = int(os.environ.get("ALEX_EMBED_BATCH", "128"))
# Embedding requests in flight at once. The backend spends most of a
# request waiting, so a little concurrency is most of the throughput.
EMBED_CONCURRENCY = int(os.environ.get("ALEX_EMBED_CONCURRENCY", "4"))
EMBED_RETRIES = int(os.environ.get("ALEX_EMBED_RETRIES", "3"))
EMBED_RETRY_BACKOFF_S = float(os.environ.get("ALEX_EMBED_BACKOFF", "0.5"))
# Set to "1" to force the offline hashing fallback (CI without a model).
_FORCE_HASH = os.environ.get("ALEX_EMBED_HASH_ONLY", "").strip() == "1"

_embed_backend_warned = False

#: Which rows of the last embed_many() call came from the hash fallback,
#: PER THREAD. Out of band because the wire format is a plain vector
#: list; thread-local because two concurrent turns embedding at once
#: would otherwise read each other's flags — and mis-cache exactly the
#: hash vectors the flags exist to keep out of the cache.
_EMBED_TLS = threading.local()


def last_embed_was_fallback() -> list[bool]:
    """The fallback flags for THIS thread's most recent embed_many()."""
    return list(getattr(_EMBED_TLS, "flags", []) or [])


def _hash_embed(text: str) -> list[float]:
    """The offline fallback: a signed bag-of-token-hashes vector.

    Cosine over this reflects *shared vocabulary* only — it is a sparse
    lexical signal wearing a dense coat. Kept so the system still works
    with no embedding service, but it is not semantic and should not be
    mistaken for it.
    """
    import hashlib
    import re

    vec = [0.0] * _DEV_EMBED_DIM
    for tok in re.findall(r"[a-z0-9]+", (text or "").lower()):
        if len(tok) < 2:
            continue
        hv = int.from_bytes(hashlib.md5(tok.encode()).digest()[:4], "big")
        vec[hv % _DEV_EMBED_DIM] += 1.0 if (hv >> 8) & 1 else -1.0
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm else vec


class EmbeddingUnavailable(RuntimeError):
    """The embedding backend could not be reached and no substitute is
    acceptable. Raised only in strict mode — see :func:`embed_many`."""


def _ollama_embed(texts: list[str]) -> list[list[float]] | None:
    """Embed a batch through the local model, or None if unavailable.

    Retries, because the alternative to a retry here is not a slower
    answer — it is a batch of hash vectors in a corpus of real ones. A
    single blip mid-ingest is enough to leave the index holding two
    incompatible embedding spaces.
    """
    global _embed_backend_warned
    if _FORCE_HASH:
        return None
    import json as _json
    import time
    import urllib.request

    payload = _json.dumps({"model": DEV_EMBED_MODEL, "input": texts}).encode()
    last: Exception | None = None
    for attempt in range(EMBED_RETRIES):
        req = urllib.request.Request(
            DEV_EMBED_URL, data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = _json.loads(resp.read())
            vectors = data.get("embeddings")
            if isinstance(vectors, list) and len(vectors) == len(texts):
                return [list(map(float, v)) for v in vectors]
            last = ValueError(
                f"expected {len(texts)} embeddings, got "
                f"{len(vectors) if isinstance(vectors, list) else type(vectors)}"
            )
        except Exception as exc:  # noqa: BLE001
            last = exc
        if attempt + 1 < EMBED_RETRIES:
            time.sleep(EMBED_RETRY_BACKOFF_S * (2**attempt))

    if not _embed_backend_warned:
        _embed_backend_warned = True
        log.warning(
            "embedding model %r unavailable at %s after %d attempts (%s); "
            "falling back to the lexical hash. Retrieval will not be semantic.",
            DEV_EMBED_MODEL, DEV_EMBED_URL, EMBED_RETRIES, last,
        )
    return None


def _map_batches(fn, batches: list[list[str]]) -> list[list]:
    """Apply ``fn`` to each batch, concurrently, **in order**.

    Order is not incidental: the caller lines these vectors back up with
    its own texts by position, so a reordered result silently attaches
    every embedding to the wrong document. ``Executor.map`` yields in
    submission order, which is what makes the concurrency safe here.

    Concurrency is worth having because an embedding request costs a
    fixed few hundred milliseconds of round-trip almost regardless of
    how many texts it carries, so a single-threaded loop spends most of
    its time waiting. Measured locally: 39 texts/s sequential against 87
    with four workers.
    """
    if len(batches) <= 1 or EMBED_CONCURRENCY <= 1:
        return [fn(b) for b in batches]
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(
        max_workers=min(EMBED_CONCURRENCY, len(batches))
    ) as pool:
        return list(pool.map(fn, batches))


def embed_many(texts: list[str], *, strict: bool = False) -> list[list[float]]:
    """Embed a batch. Batching is not an optimisation here — embedding a
    hundred thousand chunks one HTTP call at a time is the difference
    between minutes and hours.

    ``strict`` decides what happens when the backend cannot be reached,
    and the right answer differs by caller. A **query** should degrade:
    a lexical answer beats an error page, and it affects one request. An
    **ingest** must not: the hash fallback is a different embedding space
    (96 dimensions against the model's 768), so a single failed batch
    leaves the corpus holding two kinds of vector that cannot be compared
    with each other. The index detects the mismatch and abandons vector
    search *for the whole corpus* — one blip during a bulk load silently
    costs semantic retrieval everywhere. Loud failure and a retry is the
    cheaper outcome, so ingestion passes ``strict=True``.

    Returns a list parallel to ``texts``. A row is ``None`` when it came
    from the hash fallback — callers that memoise (the RAG embed cache)
    must not cache those: a fallback vector is a property of the
    backend's *availability*, not of the text, and caching one poisons
    every future query for that text with a vector from the wrong space.
    """
    if not texts:
        return []
    batches = [
        texts[i : i + EMBED_BATCH] for i in range(0, len(texts), EMBED_BATCH)
    ]
    settings = get_settings()

    if settings.mode != "dev":
        from databricks.sdk import WorkspaceClient

        ws = WorkspaceClient()
        endpoint = os.environ.get(
            "ALEX_EMBEDDING_ENDPOINT", "databricks-gte-large-en"
        )

        def serve(batch: list[str]) -> list[list[float]]:
            resp = ws.serving_endpoints.query(name=endpoint, inputs=batch)
            data = getattr(resp, "data", None) or []
            if len(data) != len(batch):
                # A misrouted or truncated response must be loud: the
                # dev path validates the count, and the prod path
                # silently zip-truncating is exactly how empty vectors
                # end up in a corpus that asked for strict mode.
                raise EmbeddingUnavailable(
                    f"endpoint {endpoint} returned {len(data)} embeddings "
                    f"for {len(batch)} inputs"
                )
            out = []
            for d in data:
                emb = getattr(d, "embedding", None)
                if not emb:
                    raise EmbeddingUnavailable(
                        f"endpoint {endpoint} returned an empty embedding"
                    )
                out.append(list(map(float, emb)))
            return out

        return [v for group in _map_batches(serve, batches) for v in group]

    def local(batch: list[str]) -> list[tuple[list[float], bool]]:
        vectors = _ollama_embed(batch)
        if vectors:
            return [(v, False) for v in vectors]
        if strict and not _FORCE_HASH:
            raise EmbeddingUnavailable(
                f"{DEV_EMBED_MODEL!r} at {DEV_EMBED_URL} could not embed "
                f"{len(batch)} texts after {EMBED_RETRIES} attempts. Refusing "
                "to substitute hash vectors: they are a different embedding "
                "space, and mixing them into the corpus disables vector "
                "search for all of it."
            )
        return [(_hash_embed(t), True) for t in batch]

    # The fallback flag rides WITH each vector through the ordered map,
    # not on a shared side list: with concurrent batches, extend() on a
    # shared list appends in COMPLETION order while the vectors flatten
    # in submission order — the flags could line up with the wrong rows,
    # and the embed cache would cache exactly the hash vectors it exists
    # to reject.
    pairs = [p for group in _map_batches(local, batches) for p in group]
    flat = [v for v, _fb in pairs]
    _EMBED_TLS.flags = [fb for _v, fb in pairs]
    return flat


def _embed_query(_text: str) -> list[float]:
    """Embed one text. See :func:`embed_many` for the batch path."""
    return embed_many([_text or ""])[0]


def store(
    conn,
    owner: str,
    kind: str,
    content: str,
    source_ref: str = "",
    embedding: list[float] | None = None,
) -> MemoryBead:
    """Persist a memory bead. In dev this is a dict; in production it's a
    pgvector INSERT with the embedding as a vector column."""
    settings = get_settings()
    if embedding is None:
        embedding = _embed_query(content)
    bead = MemoryBead(
        id=str(uuid.uuid4()),
        owner_email=owner,
        kind=kind,
        content=content,
        salience=0.5,
        source_ref=source_ref,
        embedding=embedding,
    )
    if settings.mode == "dev":
        _dev_store().add(bead)
        return bead
    # Production: pgvector INSERT
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO vector_memory
              (id, owner_email, kind, content, salience, source_ref, embedding, created_at, last_recalled, superseded_by)
            VALUES (%s, %s, %s, %s, %s, %s, %s::vector, now(), NULL, NULL)
            """,
            (bead.id, owner, kind, content, bead.salience, source_ref, embedding),
        )
    return bead


def recall(
    conn,
    owner: str,
    query_text: str = "",
    query_embedding: list[float] | None = None,
    k: int = 10,
    min_salience: float = 0.1,
) -> list[MemoryBead]:
    """Return up to k memory beads for `owner`, ranked by similarity to the
    query (or by salience if no embedding is given).

    When a query is supplied, beads with salience >= 0.5 (the default for
    store) are always considered even if their cosine similarity is low
    — a relevant preference should not be hidden behind a cold vector.
    """
    settings = get_settings()
    beads = (
        _dev_store().for_owner(owner)
        if settings.mode == "dev"
        else _pg_recall(conn, owner, k * 4, min_salience)
    )
    if query_text or query_embedding is not None:
        qe = query_embedding or _embed_query(query_text)
        scored = [(b, _cosine(qe, b.embedding)) for b in beads if b.embedding]
        scored.sort(key=lambda x: (-x[1], -x[0].salience))
        out = [b for b, s in scored[:k] if s > 0.05 or b.salience >= 0.5]
        if len(out) < k:
            # Pad with beads that could NOT be scored — no embedding,
            # e.g. pre-vector rows — never with beads the similarity
            # threshold just rejected: returning a dissimilar bead as
            # if recalled is worse than returning fewer.
            seen = {b.id for b in out}
            unscored = [b for b in beads if not b.embedding]
            unscored.sort(key=lambda b: (-b.salience, -b.created_at))
            for b in unscored:
                if b.id in seen:
                    continue
                out.append(b)
                if len(out) >= k:
                    break
        return out
    beads = [b for b in beads if b.salience >= min_salience]
    beads.sort(key=lambda b: (-b.salience, -b.created_at))
    return beads[:k]


def _pg_recall(conn, owner: str, k: int, min_salience: float) -> list[MemoryBead]:
    # The embedding column is selected: without it every prod bead is
    # constructed with embedding=[] and recall degrades to salience
    # ranking — silently, because the empty vectors fail no check.
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, owner_email, kind, content, salience, source_ref,
                   created_at, last_recalled, superseded_by, embedding
              FROM vector_memory
             WHERE owner_email = %s AND superseded_by IS NULL AND salience >= %s
             ORDER BY salience DESC, created_at DESC
             LIMIT %s
            """,
            (owner, min_salience, k),
        )
        rows = cur.fetchall()
    out: list[MemoryBead] = []
    for r in rows:
        out.append(
            MemoryBead(
                id=r[0],
                owner_email=r[1],
                kind=r[2],
                content=r[3],
                salience=float(r[4]),
                source_ref=r[5] or "",
                created_at=r[6].timestamp() if r[6] else 0.0,
                last_recalled=r[7].timestamp() if r[7] else None,
                superseded_by=r[8],
                embedding=_pg_vector(r[9]),
            )
        )
    return out


def _pg_vector(raw) -> list[float]:
    """Parse pgvector's text representation into a list (see rag._pg_vector)."""
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [float(x) for x in raw]
    try:
        import json as _json

        parsed = _json.loads(str(raw))
        return [float(x) for x in parsed] if isinstance(parsed, list) else []
    except (ValueError, TypeError):
        return []


def supersede(conn, bead_id: str, by_bead_id: str) -> None:
    """Mark `bead_id` as superseded by `by_bead_id`. Used by the
    consolidation job when a newer bead has the same meaning."""
    settings = get_settings()
    if settings.mode == "dev":
        b = _dev_store().get(bead_id)
        if b:
            b.superseded_by = by_bead_id
        return
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE vector_memory SET superseded_by = %s WHERE id = %s",
            (by_bead_id, bead_id),
        )


def decay_and_supersede(conn, owner: str, *, decay: float = 0.98) -> int:
    """Nightly decay: multiply salience by `decay`. Returns the row count
    affected. Called from `pipelines/memory_apply.py`."""
    settings = get_settings()
    if settings.mode == "dev":
        n = 0
        for b in _dev_store().for_owner(owner):
            b.salience *= decay
            n += 1
        return n
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE vector_memory SET salience = salience * %s "
            "WHERE owner_email = %s AND superseded_by IS NULL",
            (decay, owner),
        )
        return cur.rowcount
