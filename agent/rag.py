"""
Enterprise RAG — governed retrieval over the organisation's documents.

This is the *enterprise* half of Alex's knowledge. Where
:mod:`agent.vector_memory` holds what Alex has learned **about one
user** (preferences, facts, project cues), this module holds what the
**organisation knows**: policies, contracts, briefs, reports, wiki
pages, ingested workspace files — chunked, embedded, and retrievable
with citations.

Design
------

* **Chunking** is structure-aware: text is split on blank-line
  boundaries into ~``TARGET_CHARS`` windows with ``OVERLAP_CHARS`` of
  carry-over, and the nearest markdown heading is prefixed to each
  chunk so a retrieved fragment still says what it is about.

* **Retrieval is hybrid**, which is the state of the art for
  enterprise corpora: a dense vector ranking (cosine over the same
  embedding function :mod:`agent.vector_memory` uses, so dev and
  production behave alike), a sparse lexical ranking (BM25), and an
  optional **graph expansion** pass that pulls entities connected to
  the first-round hits out of :mod:`agent.enterprise_graph` and runs a
  second retrieval round for them. The three rankings are merged with
  Reciprocal Rank Fusion (RRF), which needs no score calibration
  between rankers and degrades gracefully when one ranker is empty.

* **Every result carries a citation** (``doc_id``, ``title``,
  ``source_ref``, ``chunk_index``) plus its component scores, so the
  Transparency surface can show *why* a passage was retrieved.

* **ACL is enforced at retrieval**, not after: a chunk is only visible
  if its ``scope`` is one the caller may read and its
  ``acl_capability`` is in the caller's effective capability set.
  Retrieval never returns a passage the user could not open directly.

Storage
-------

Dev/test keeps the corpus in memory on ``alex.testing.get_fakes()``.
Production writes to the ``enterprise_docs`` pgvector table on Lakebase
(DDL in pipelines/lakebase_ddl.pg.sql). Both paths go through the same
public API. There is no Delta mirror of the corpus — chunks live in one
place, and a claim of a mirror that nothing writes would be worse than
none.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import time
import uuid
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from alex.config import get_settings
from alex.testing import get_fakes

log = logging.getLogger("agent.rag")

# Chunking
TARGET_CHARS = 1200
OVERLAP_CHARS = 150
MIN_CHUNK_CHARS = 60
MAX_DOC_CHARS = 2_000_000

# Retrieval
DEFAULT_K = 6
CANDIDATE_MULTIPLIER = 6
RRF_K = 60  # standard RRF damping constant
GRAPH_EXPANSION_ENTITIES = 4
# The per-retriever rank window fused over. Narrower windows drop
# single-list hits, which RRF already under-weights.
FUSION_WINDOW = 60
# The candidate pool handed to stage two. Published guidance settles
# around 50-200; wider costs reranking time, narrower costs recall the
# reranker cannot recover.
CANDIDATE_POOL = 120
# Graph expansion is a hint, not a retriever: capped, down-weighted, and
# dropped entirely when its best hit is weak.
GRAPH_LIST_CAP = 20
GRAPH_WEIGHT = 0.5
GRAPH_SCORE_FLOOR = 0.5
# Top lexical hits that keep their place regardless of fused score, so
# exact matches cannot be voted out by consensus mediocrity.
LEXICAL_PIN_SLOTS = 3

# The capability a document requires by default.
DEFAULT_CAPABILITY = "knowledge.read"
# The scope every user can read (organisation-wide material).
ENTERPRISE_SCOPE = "enterprise"

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_\-]*")
# Function words carry no retrieval signal and actively distort BM25:
# they appear many times per document, so the term-frequency component
# rewards them, and in any corpus where they are unevenly distributed
# they can outweigh a genuinely rare term. Filtering them is not an
# optimisation — it is the difference between finding a unique
# identifier and not.
_STOPWORDS = frozenset(["a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can", "could", "did", "do", "does", "for", "from", "had", "has", "have", "how", "i", "if", "in", "into", "is", "it", "its", "of", "on", "or", "our", "shall", "should", "so", "than", "that", "the", "their", "them", "then", "there", "these", "they", "this", "to", "was", "were", "what", "when", "where", "which", "who", "why", "will", "with", "would", "you", "your"])
_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")

# Text-ish files worth ingesting from a workspace walk.
INGESTABLE_SUFFIXES = (
    ".md", ".txt", ".rst", ".csv", ".json", ".yaml", ".yml",
    ".py", ".sql", ".js", ".jsx", ".ts", ".tsx", ".html", ".cfg", ".ini",
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    """One retrievable passage of an enterprise document."""

    chunk_id: str
    doc_id: str
    title: str
    text: str
    chunk_index: int
    source_ref: str = ""
    scope: str = ENTERPRISE_SCOPE
    acl_capability: str = DEFAULT_CAPABILITY
    classification: str = "internal"
    embedding: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def citation(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "source_ref": self.source_ref,
            "chunk_index": self.chunk_index,
            "scope": self.scope,
            "classification": self.classification,
        }


@dataclass
class Retrieved:
    """A chunk plus the evidence for why it was retrieved."""

    chunk: Chunk
    score: float
    vector_rank: int | None = None
    lexical_rank: int | None = None
    graph_rank: int | None = None
    via_entities: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.chunk.text,
            "score": round(self.score, 5),
            "citation": self.chunk.citation(),
            "why": {
                "vector_rank": self.vector_rank,
                "lexical_rank": self.lexical_rank,
                "graph_rank": self.graph_rank,
                "via_entities": self.via_entities,
            },
        }


# ---------------------------------------------------------------------------
# Dev corpus (in-memory, attached to the shared fakes object)
# ---------------------------------------------------------------------------


class _DevCorpus:
    """In-memory corpus. ``version`` increments on every mutation so the
    retrieval index below knows when its cache is stale.

    ``by_doc`` exists so removing a document is proportional to that
    document rather than to the corpus. Scanning every chunk to find one
    document's chunks made ingest quadratic — invisible at a hundred
    documents, and the difference between eighteen seconds and never
    finishing at a hundred thousand.
    """

    def __init__(self) -> None:
        self.chunks: dict[str, Chunk] = {}
        self.docs: dict[str, dict[str, Any]] = {}
        self.by_doc: dict[str, set[str]] = {}
        self.version = 0
        # A version counter alone cannot identify a corpus, because a
        # replacement corpus starts counting from zero again and will
        # collide with its predecessor at every version. The index cache
        # outlives any single corpus, so it needs to tell them apart.
        self.token = uuid.uuid4().hex

    def add(self, chunk: Chunk) -> None:
        self.chunks[chunk.chunk_id] = chunk
        self.by_doc.setdefault(chunk.doc_id, set()).add(chunk.chunk_id)
        self.version += 1

    def drop_doc(self, doc_id: str) -> int:
        gone = self.by_doc.pop(doc_id, set())
        for cid in gone:
            self.chunks.pop(cid, None)
        self.docs.pop(doc_id, None)
        if gone:
            self.version += 1
        return len(gone)

    def all(self) -> list[Chunk]:
        return list(self.chunks.values())


def _dev_corpus() -> _DevCorpus:
    fk = get_fakes()
    corpus = getattr(fk, "_rag_corpus", None)
    if corpus is None:
        corpus = _DevCorpus()
        fk._rag_corpus = corpus
    return corpus


def _is_dev() -> bool:
    return get_settings().mode == "dev"


_EMBED_CACHE: dict[str, list[float]] = {}
_EMBED_CACHE_MAX = 2048


def _embed_many(texts: list[str], *, strict: bool = False) -> list[list[float]]:
    """Embed several texts, using the cache and one batched call for the
    rest. Ingest embeds every chunk of a document, so batching here is
    what makes a large corpus loadable at all.

    ``strict`` propagates to :func:`agent.vector_memory.embed_many`:
    ingestion refuses hash substitutes rather than mixing two embedding
    spaces into one corpus, while a query prefers a degraded answer to
    an error.

    The batch result is length-checked before the zip: a backend that
    returns fewer vectors than texts (misrouted batch, truncated
    response) must raise in strict mode — the old ``strict=False``
    zip silently truncated and turned the unfilled slots into empty
    vectors, which defeats the refusal strict mode exists to make.
    """
    from agent.vector_memory import EmbeddingUnavailable, embed_many

    out: list[list[float] | None] = [_EMBED_CACHE.get(t) for t in texts]
    missing = [i for i, v in enumerate(out) if v is None]
    if missing:
        fresh = embed_many([texts[i] for i in missing], strict=strict)
        if len(fresh) != len(missing):
            if strict:
                raise EmbeddingUnavailable(
                    f"embedding backend returned {len(fresh)} vectors for "
                    f"{len(missing)} texts; refusing to attach them to "
                    "documents by position"
                )
            log.error(
                "embedding backend returned %d vectors for %d texts; "
                "degrading to lexical-only for the unmatched",
                len(fresh), len(missing),
            )
            fresh = fresh[: len(missing)] + [None] * (
                len(missing) - len(fresh)
            )
        # Never cache a hash-fallback vector: it belongs to the backend's
        # availability, not to the text, and a cached one would poison
        # every future query for that text against the real corpus.
        from agent import vector_memory as _vm_mod

        fallback_flags = _vm_mod.last_embed_was_fallback()
        for offset, (i, vec) in enumerate(zip(missing, fresh, strict=True)):
            if vec is None:
                continue
            out[i] = vec
            was_fallback = (
                offset < len(fallback_flags) and fallback_flags[offset]
            )
            if was_fallback:
                continue
            if len(_EMBED_CACHE) >= _EMBED_CACHE_MAX:
                _EMBED_CACHE.clear()
            _EMBED_CACHE[texts[i]] = vec
    return [v or [] for v in out]


def _embed(text: str) -> list[float]:
    """Embed via the same function :mod:`agent.vector_memory` uses, so a
    passage and a memory bead live in the same vector space.

    Results are memoised: embedding is deterministic for a given text,
    and the same query text recurs constantly (every turn re-embeds the
    user's message for retrieval, memory recall and entity resolution).
    In production this also collapses three network round-trips per turn
    into one.
    """
    key = text or ""
    hit = _EMBED_CACHE.get(key)
    if hit is not None:
        return hit
    from agent.vector_memory import _embed_query

    vec = _embed_query(key)
    if len(_EMBED_CACHE) >= _EMBED_CACHE_MAX:
        _EMBED_CACHE.clear()
    _EMBED_CACHE[key] = vec
    return vec


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(x * x for x in b)) or 1e-9
    return dot / (na * nb)


def _tokens(text: str) -> list[str]:
    return [
        t for t in _TOKEN_RE.findall((text or "").lower())
        if t not in _STOPWORDS
    ]


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunk_text(
    text: str,
    *,
    target: int = TARGET_CHARS,
    overlap: int = OVERLAP_CHARS,
) -> list[str]:
    """Split ``text`` into overlapping, heading-aware passages.

    Blocks are accumulated up to ``target`` characters and flushed on
    blank-line boundaries so sentences are not cut mid-thought. The most
    recent markdown heading is prefixed to each chunk that does not
    already start with one, so an isolated passage retains its subject.
    """
    if not (text or "").strip():
        return []
    text = text[:MAX_DOC_CHARS]
    blocks = re.split(r"\n\s*\n", text)
    out: list[str] = []
    buf: list[str] = []
    size = 0
    heading = ""

    def flush() -> None:
        nonlocal buf, size
        body = "\n\n".join(buf).strip()
        buf, size = [], 0
        if len(body) < MIN_CHUNK_CHARS and out:
            # Too small to stand alone — append to the previous chunk.
            out[-1] = f"{out[-1]}\n\n{body}".strip()
            return
        if not body:
            return
        if heading and not _HEADING_RE.match(body.splitlines()[0] or ""):
            body = f"{heading}\n\n{body}"
        out.append(body)

    for block in blocks:
        block = block.strip("\n")
        if not block.strip():
            continue
        first = block.splitlines()[0] if block.splitlines() else ""
        m = _HEADING_RE.match(first)
        # A heading only governs the chunks that FOLLOW it, so pending
        # content is flushed under the previous heading before the new
        # one takes effect.
        pending_heading = m.group(0).strip() if m else None
        # A single block bigger than the target is hard-split on lines.
        if len(block) > target:
            if buf:
                flush()
            if pending_heading:
                heading = pending_heading
            lines = block.splitlines()
            cur: list[str] = []
            cur_len = 0
            for ln in lines:
                if cur_len + len(ln) + 1 > target and cur:
                    buf = ["\n".join(cur)]
                    size = cur_len
                    flush()
                    # carry overlap
                    tail = "\n".join(cur)[-overlap:] if overlap else ""
                    cur = [tail] if tail else []
                    cur_len = len(tail)
                cur.append(ln)
                cur_len += len(ln) + 1
            if cur:
                buf = ["\n".join(cur)]
                size = cur_len
                flush()
            continue
        if size + len(block) > target and buf:
            prev_tail = "\n\n".join(buf)[-overlap:] if overlap else ""
            flush()
            if prev_tail:
                buf = [prev_tail]
                size = len(prev_tail)
        if pending_heading:
            heading = pending_heading
        buf.append(block)
        size += len(block) + 2
    if buf:
        flush()
    return [c for c in out if c.strip()]


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


def ingest(
    conn,
    *,
    doc_id: str = "",
    title: str,
    text: str,
    source_ref: str = "",
    scope: str = ENTERPRISE_SCOPE,
    acl_capability: str = DEFAULT_CAPABILITY,
    classification: str = "internal",
    metadata: dict[str, Any] | None = None,
    replace: bool = True,
) -> dict[str, Any]:
    """Chunk, embed, and store one document. Returns an ingest summary.

    ``doc_id`` defaults to a stable hash of ``source_ref`` (or the
    title), so re-ingesting the same file replaces its chunks rather
    than duplicating them.
    """
    if not (text or "").strip():
        return {"doc_id": "", "title": title, "chunks": 0, "skipped": "empty"}
    key = source_ref or title
    doc_id = doc_id or "doc-" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    if replace:
        delete_document(conn, doc_id)
    pieces = chunk_text(text)
    # One embedding call for the whole document rather than one per chunk.
    vectors = _embed_many(
        [title + "\n" + piece for piece in pieces], strict=True
    )
    stored = 0
    for i, piece in enumerate(pieces):
        chunk = Chunk(
            chunk_id=f"{doc_id}:{i}",
            doc_id=doc_id,
            title=title,
            text=piece,
            chunk_index=i,
            source_ref=source_ref,
            scope=scope,
            acl_capability=acl_capability,
            classification=_access().normalise_classification(classification),
            embedding=vectors[i],
            metadata=dict(metadata or {}),
        )
        _put(conn, chunk)
        stored += 1
    if _is_dev():
        _dev_corpus().docs[doc_id] = {
            "doc_id": doc_id,
            "title": title,
            "source_ref": source_ref,
            "scope": scope,
            "acl_capability": acl_capability,
            "chunks": stored,
            "metadata": dict(metadata or {}),
            "ingested_at": time.time(),
        }
    log.info("rag.ingest doc=%s chunks=%d scope=%s", doc_id, stored, scope)
    return {"doc_id": doc_id, "title": title, "chunks": stored, "scope": scope}


#: Chunks embedded per wave in :func:`ingest_many`. Large enough that the
#: backend's fixed per-request cost is amortised, small enough that the
#: vectors held in memory at once stay bounded regardless of corpus size.
INGEST_WAVE = int(os.environ.get("ALEX_INGEST_WAVE", "1024"))


def ingest_many(
    conn,
    documents: Iterable[dict[str, Any]],
    *,
    scope: str = ENTERPRISE_SCOPE,
    acl_capability: str = DEFAULT_CAPABILITY,
    classification: str = "internal",
    replace: bool = True,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Bulk-load documents, embedding across document boundaries.

    :func:`ingest` embeds one document's chunks per call, which is the
    right shape when documents arrive one at a time. A bulk load is a
    different shape. The embedding backend costs a few hundred
    milliseconds per *request* almost regardless of batch size, so a
    corpus loaded a document at a time pays that cost once per document:
    at a quarter of a million documents that is days of round-trips
    before any work is done. Batching across documents pays it once per
    :data:`INGEST_WAVE` chunks instead.

    Each mapping needs ``title`` and ``text``; ``doc_id``, ``source_ref``,
    ``scope``, ``acl_capability``, ``classification`` and ``metadata``
    override the defaults per document. Embedding happens in waves so
    peak memory is set by :data:`INGEST_WAVE`, not by the corpus size —
    a stream of a million documents uses the same memory as a hundred.

    ``progress`` is called with ``(documents_done, chunks_stored)`` after
    each wave, for a caller that wants to report a long load.
    """
    normalise = _access().normalise_classification
    pending: list[tuple[Chunk, str]] = []  # chunk, text to embed
    queued_docs: set[str] = set()  # doc_ids with chunks still in `pending`
    summaries: list[dict[str, Any]] = []
    docs_done = chunks_stored = skipped = 0

    def flush() -> None:
        nonlocal chunks_stored
        if not pending:
            return
        vectors = _embed_many([text for _chunk, text in pending], strict=True)
        for (chunk, _text), vector in zip(pending, vectors, strict=True):
            chunk.embedding = vector
            _put(conn, chunk)
            chunks_stored += 1
        pending.clear()
        queued_docs.clear()

    for doc in documents:
        text = (doc.get("text") or "").strip()
        title = doc.get("title") or ""
        if not text:
            skipped += 1
            continue
        source_ref = doc.get("source_ref") or ""
        key = source_ref or title
        doc_id = doc.get("doc_id") or (
            "doc-" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
        )
        doc_scope = doc.get("scope") or scope
        doc_cap = doc.get("acl_capability") or acl_capability
        doc_class = normalise(doc.get("classification") or classification)
        metadata = dict(doc.get("metadata") or {})
        if replace:
            # Dropping this document's old chunks is safe while other
            # documents sit in the buffer — they are not stored yet, and
            # delete_document only touches this doc_id. The one hazard is
            # the same document appearing twice in a batch: the delete
            # would not see the first version's queued chunks, and the
            # flush would then write both. Flushing first is the cheap
            # fix, and it is rare enough not to cost the batching.
            if doc_id in queued_docs:
                flush()
            delete_document(conn, doc_id)

        pieces = chunk_text(text)
        queued_docs.add(doc_id)
        for i, piece in enumerate(pieces):
            pending.append((
                Chunk(
                    chunk_id=f"{doc_id}:{i}",
                    doc_id=doc_id,
                    title=title,
                    text=piece,
                    chunk_index=i,
                    source_ref=source_ref,
                    scope=doc_scope,
                    acl_capability=doc_cap,
                    classification=doc_class,
                    embedding=[],
                    metadata=dict(metadata),
                ),
                title + "\n" + piece,
            ))
        docs_done += 1
        if _is_dev():
            _dev_corpus().docs[doc_id] = {
                "doc_id": doc_id,
                "title": title,
                "source_ref": source_ref,
                "scope": doc_scope,
                "acl_capability": doc_cap,
                "chunks": len(pieces),
                "metadata": dict(metadata),
                "ingested_at": time.time(),
            }
        summaries.append(
            {"doc_id": doc_id, "title": title, "chunks": len(pieces)}
        )
        if len(pending) >= INGEST_WAVE:
            flush()
            if progress:
                progress(docs_done, chunks_stored)

    flush()
    if progress:
        progress(docs_done, chunks_stored)
    log.info(
        "rag.ingest_many docs=%d chunks=%d skipped=%d",
        docs_done, chunks_stored, skipped,
    )
    return {
        "documents": docs_done,
        "chunks": chunks_stored,
        "skipped": skipped,
        "results": summaries,
    }


def _put(conn, chunk: Chunk) -> None:
    if _is_dev():
        _dev_corpus().add(chunk)
        return
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO enterprise_docs
              (chunk_id, doc_id, title, text, chunk_index, source_ref,
               scope, acl_capability, embedding, metadata, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::vector, %s, now())
            ON CONFLICT (chunk_id) DO UPDATE
              SET text = EXCLUDED.text,
                  embedding = EXCLUDED.embedding,
                  title = EXCLUDED.title,
                  metadata = EXCLUDED.metadata
            """,
            (
                chunk.chunk_id,
                chunk.doc_id,
                chunk.title,
                chunk.text,
                chunk.chunk_index,
                chunk.source_ref,
                chunk.scope,
                chunk.acl_capability,
                chunk.embedding,
                chunk.metadata,
            ),
        )


def delete_document(conn, doc_id: str) -> int:
    """Remove every chunk of a document. Returns the number removed."""
    if not doc_id:
        return 0
    if _is_dev():
        return _dev_corpus().drop_doc(doc_id)
    with conn.cursor() as cur:
        cur.execute("DELETE FROM enterprise_docs WHERE doc_id = %s", (doc_id,))
        return int(cur.rowcount or 0)


def list_documents(conn, *, scopes: list[str] | None = None) -> list[dict[str, Any]]:
    """Summarise the corpus: one row per document."""
    if _is_dev():
        by_doc: dict[str, dict[str, Any]] = {}
        for c in _dev_corpus().all():
            if scopes is not None and c.scope not in scopes:
                continue
            row = by_doc.setdefault(
                c.doc_id,
                {
                    "doc_id": c.doc_id,
                    "title": c.title,
                    "source_ref": c.source_ref,
                    "scope": c.scope,
                    "classification": c.classification,
                    "chunks": 0,
                    "chars": 0,
                },
            )
            row["chunks"] += 1
            row["chars"] += len(c.text)
        return sorted(by_doc.values(), key=lambda r: r["title"].lower())
    with conn.cursor() as cur:
        if scopes:
            cur.execute(
                """
                SELECT doc_id, max(title), max(source_ref), max(scope),
                       count(*), sum(length(text))
                  FROM enterprise_docs WHERE scope = ANY(%s)
                 GROUP BY doc_id ORDER BY max(title)
                """,
                (scopes,),
            )
        else:
            cur.execute(
                """
                SELECT doc_id, max(title), max(source_ref), max(scope),
                       count(*), sum(length(text))
                  FROM enterprise_docs GROUP BY doc_id ORDER BY max(title)
                """
            )
        return [
            {
                "doc_id": r[0], "title": r[1], "source_ref": r[2],
                "scope": r[3], "chunks": int(r[4]), "chars": int(r[5] or 0),
            }
            for r in cur.fetchall()
        ]


def ingest_workspace_file(
    conn, user_email: str, path: str, *, scope: str = "", title: str = ""
) -> dict[str, Any]:
    """Ingest one file from the user's sandboxed workspace."""
    from agent import workspace_tools

    text = workspace_tools.view(user_email, path)
    return ingest(
        conn,
        title=title or path.rsplit("/", 1)[-1],
        text=text,
        source_ref=f"workspace:{user_email}:{path}",
        scope=scope or user_scope(user_email),
        metadata={"kind": "workspace_file", "path": path, "owner": user_email},
    )


def ingest_workspace(
    conn, user_email: str, *, scope: str = "", max_files: int = 200
) -> dict[str, Any]:
    """Walk the user's workspace and ingest every text-ish file.

    Files are streamed into :func:`ingest_many` rather than ingested one
    at a time. Indexing a workspace is a bulk load: at one embedding
    request per file this spent most of a minute on round-trips before
    the user saw anything.
    """
    from agent import workspace_tools

    root = workspace_tools._user_root(user_email)
    if not os.path.isdir(root):
        return {"files": 0, "chunks": 0, "documents": []}

    def walk() -> Iterable[dict[str, Any]]:
        seen = 0
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                d for d in dirnames if not d.startswith((".", "__pycache__"))
            ]
            for fn in sorted(filenames):
                if not fn.lower().endswith(INGESTABLE_SUFFIXES):
                    continue
                rel = os.path.relpath(
                    os.path.join(dirpath, fn), root
                ).replace("\\", "/")
                try:
                    text = workspace_tools.view(user_email, rel)
                except (OSError, ValueError):
                    log.debug("skipped %s during workspace ingest", rel, exc_info=True)
                    continue
                yield {
                    "title": rel.rsplit("/", 1)[-1],
                    "text": text,
                    "source_ref": f"workspace:{user_email}:{rel}",
                    "scope": scope or user_scope(user_email),
                    "metadata": {
                        "kind": "workspace_file", "path": rel, "owner": user_email
                    },
                }
                seen += 1
                if seen >= max_files:
                    return

    result = ingest_many(conn, walk())
    return {
        "files": result["documents"],
        "chunks": result["chunks"],
        "documents": result["results"],
    }


# ---------------------------------------------------------------------------
# Scope / ACL
# ---------------------------------------------------------------------------


def user_scope(user_email: str) -> str:
    """The private knowledge scope of one user."""
    return f"user:{(user_email or '').strip().lower()}"


def visible_scopes(user_email: str, extra: list[str] | None = None) -> list[str]:
    """Scopes a user may read: enterprise-wide, their own, plus extras."""
    scopes = [ENTERPRISE_SCOPE, user_scope(user_email), "default"]
    for s in extra or []:
        if s and s not in scopes:
            scopes.append(s)
    return scopes


def _access():
    from agent import access

    return access


def _candidates(conn, scopes: list[str], capabilities: set[str] | None) -> list[Chunk]:
    if _is_dev():
        pool = _dev_corpus().all()
    else:
        pool = _pg_all(conn, scopes)
    out = []
    for c in pool:
        if scopes and c.scope not in scopes:
            continue
        if capabilities is not None and c.acl_capability not in capabilities:
            continue
        out.append(c)
    return out


def _authorize(
    principal, results: list[Retrieved], query: str
) -> list[Retrieved]:
    """Apply the authorization decision point to retrieved passages.

    Scope pre-filtering already narrowed the candidate set; this is the
    gate that actually decides, and it records every allow and deny.
    A passage the principal may not read is dropped here — it never
    reaches the prompt, the tool result, or the UI.
    """
    access = _access()
    allowed: list[Retrieved] = []
    for r in results:
        decision = access.can_read(
            principal,
            scope=r.chunk.scope,
            classification=r.chunk.classification,
            capability=r.chunk.acl_capability,
        )
        access.audit_read(
            principal,
            resource=f"{r.chunk.doc_id}:{r.chunk.chunk_index}",
            decision=decision,
            query=query,
        )
        if decision.allowed:
            allowed.append(r)
    return allowed


def _pg_vector(raw: Any) -> list[float]:
    """Parse pgvector's text representation ("[1,2,3]") into a list.

    psycopg returns the vector column as a string unless a codec is
    registered; parsing here keeps the SELECT portable either way.
    An unparseable value yields [] — the chunk then scores zero on
    the dense path but still serves BM25, which is degradation, not
    loss.
    """
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


def _pg_all(conn, scopes: list[str]) -> list[Chunk]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT chunk_id, doc_id, title, text, chunk_index, source_ref,
                   scope, acl_capability, embedding, metadata
              FROM enterprise_docs WHERE scope = ANY(%s)
            """,
            (scopes,),
        )
        return [
            Chunk(
                chunk_id=r[0], doc_id=r[1], title=r[2], text=r[3],
                chunk_index=int(r[4]), source_ref=r[5] or "", scope=r[6],
                acl_capability=r[7] or DEFAULT_CAPABILITY,
                embedding=_pg_vector(r[8]),
                metadata=r[9] or {},
            )
            for r in cur.fetchall()
        ]


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


class _Index:
    """A retrieval index over a candidate set.

    Building this is the expensive part of search, so it is built once
    per query *at most* (both the primary query and the graph-expansion
    query share it) and cached across queries while the corpus is
    unchanged.

    * ``postings`` is an inverted index (term -> [(chunk_pos, tf)]), so
      BM25 scoring touches only the chunks that actually contain a query
      term instead of walking the whole corpus.
    * ``norms`` caches each chunk's vector magnitude so cosine
      similarity is a dot product and a division, not two square roots.
    * ``matrix`` is an L2-normalised NumPy matrix of the embeddings when
      NumPy is importable, which turns the dense scan into a single
      matrix-vector product instead of a Python loop over every chunk.
      Without NumPy the pure-Python path still works, just slower.
    * ``bm25`` is a sparse CSR matrix holding the *precomputed* BM25
      weight of every (chunk, term) pair. Scoring a query becomes one
      sparse matrix-vector product in C rather than a Python walk over
      posting lists.

      That difference is the whole story at scale. Walking postings costs
      time proportional to how *common* the query's words are, so a
      question phrased in ordinary English ("who owns the programme for
      team 37") was measured 3000x slower than one containing a unique
      identifier — 300ms against 0.1ms at sixty thousand chunks, and
      seconds at half a million. The sparse product does not care how
      common the terms are.
    """

    __slots__ = (
        "chunks", "postings", "df", "lengths", "avgdl", "norms", "n",
        "matrix", "bm25", "vocab", "_by_position",
    )

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.n = len(chunks)
        self._by_position: dict | None = None
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.df: dict[str, int] = defaultdict(int)
        self.lengths: list[int] = []
        self.norms: list[float] = []
        for pos, chunk in enumerate(chunks):
            toks = _tokens(f"{chunk.title} {chunk.text}")
            self.lengths.append(len(toks) or 1)
            counts: dict[str, int] = defaultdict(int)
            for tok in toks:
                counts[tok] += 1
            for tok, freq in counts.items():
                self.postings[tok].append((pos, freq))
                self.df[tok] += 1
            emb = chunk.embedding
            self.norms.append(
                math.sqrt(sum(x * x for x in emb)) if emb else 0.0
            )
        self.avgdl = (
            sum(self.lengths) / len(self.lengths) if self.lengths else 1.0
        )
        self.matrix = self._build_matrix()
        self.vocab: dict[str, int] = {}
        self.bm25 = self._build_bm25()

    @property
    def by_position(self) -> dict:
        """(doc_id, chunk_index) -> chunk, built once per index.

        Neighbour stitching consults this per query; building the map in
        the query path cost ~45ms per call at 142k chunks — most of the
        query's latency budget for a lookup that never changes while the
        index lives.
        """
        if self._by_position is None:
            self._by_position = {
                (c.doc_id, c.chunk_index): c for c in self.chunks
            }
        return self._by_position

    def _build_bm25(self):
        """Precompute every (chunk, term) BM25 weight as a sparse matrix.

        Returns ``None`` when SciPy is unavailable, in which case the
        posting-walk path below still works.
        """
        try:
            import numpy as np
            from scipy import sparse
        except ImportError:
            return None
        if not self.n:
            return None
        k1, b = 1.5, 0.75
        self.vocab = {term: i for i, term in enumerate(self.postings)}
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        for term, postings in self.postings.items():
            col = self.vocab[term]
            df = self.df[term]
            idf = math.log(1 + (self.n - df + 0.5) / (df + 0.5))
            for pos, freq in postings:
                norm = 1 - b + b * self.lengths[pos] / self.avgdl
                rows.append(pos)
                cols.append(col)
                data.append(idf * (freq * (k1 + 1)) / (freq + k1 * norm))
        if not data:
            return None
        return sparse.csr_matrix(
            (np.asarray(data, dtype="float32"), (rows, cols)),
            shape=(self.n, len(self.vocab)),
        )

    def _build_matrix(self):
        """L2-normalised embedding matrix, or None when unavailable."""
        try:
            import numpy as np
        except ImportError:
            return None
        dims = {len(c.embedding) for c in self.chunks if c.embedding}
        if len(dims) > 1:
            # Two embedding widths in one corpus means two embedding
            # *spaces*, and cosine between them is noise. Abandoning the
            # matrix is right, but doing it quietly is how the corpus got
            # this way: an embedding backend blip mid-ingest substituted
            # 96-dimension hash vectors into a 768-dimension corpus, and
            # the only symptom was retrieval quietly getting worse for
            # every document, not just the affected ones.
            log.error(
                "corpus has %d embedding widths (%s): vector search is "
                "disabled for the whole corpus. This usually means the "
                "embedding backend failed during ingest and hash vectors "
                "were substituted. Re-ingest the affected documents.",
                len(dims), sorted(dims),
            )
            return None
        if not dims:
            return None
        mat = np.array(
            [c.embedding if c.embedding else [0.0] * dims.copy().pop()
             for c in self.chunks],
            dtype="float32",
        )
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms


# Cache one index per (scope set, capability set), valid for exactly one
# (corpus identity, version) stamp. Both halves of the stamp are load
# bearing: the version invalidates the cache when a document is ingested
# or dropped, and the identity token invalidates it when the corpus is
# replaced wholesale. Version alone was not enough — a fresh corpus
# restarts at zero, so it collided with its predecessor at every version
# and the cache served the *previous* corpus's documents.
_INDEX_CACHE: dict[tuple, tuple[tuple[str, int], _Index]] = {}
_INDEX_CACHE_MAX = 16


def _get_index(
    conn, scopes: list[str], capabilities: set[str] | None
) -> _Index:
    if not _is_dev():
        # Production reads a filtered candidate set from Postgres, where
        # the pgvector / GIN indexes do this work; caching a snapshot in
        # the app process would risk serving stale rows.
        return _Index(_candidates(conn, scopes, capabilities))
    key = (
        tuple(sorted(scopes)),
        tuple(sorted(capabilities)) if capabilities is not None else None,
    )
    corpus = _dev_corpus()
    stamp = (corpus.token, corpus.version)
    cached = _INDEX_CACHE.get(key)
    if cached and cached[0] == stamp:
        return cached[1]
    index = _Index(_candidates(conn, scopes, capabilities))
    if len(_INDEX_CACHE) >= _INDEX_CACHE_MAX:
        _INDEX_CACHE.clear()
    _INDEX_CACHE[key] = (stamp, index)
    return index


def _rarity(query: str, index: _Index) -> dict[str, float]:
    """How distinctive each query term is, 0..1.

    A term in one document out of a hundred thousand scores ~1; a term
    in a tenth of the corpus scores near 0. This is what lets the
    reranker treat an identifier as decisive without hard-coding what an
    identifier looks like.
    """
    # Rarity is a statement about a term relative to a corpus. With one
    # document there is nothing to be rare against, and log(1) is zero —
    # so the signal is simply unavailable rather than infinite.
    if index.n <= 1:
        return {}
    import math as _math

    out: dict[str, float] = {}
    for term in {t for t in _tokens(query) if len(t) > 1}:
        df = index.df.get(term, 0)
        if df <= 0:
            continue
        # log-scaled so the difference between df=1 and df=10 matters
        # more than between df=1000 and df=10000.
        out[term] = _math.log(index.n / df) / _math.log(index.n)
    return out


def _bm25_ranking(
    query: str, index: _Index, *, top: int = 0
) -> list[tuple[Chunk, float]]:
    """BM25 (k1=1.5, b=0.75) scored through the inverted index.

    ``top`` bounds how many results are *materialised*. A query made of
    ordinary words matches most of the corpus, and building a Python
    tuple for every one of sixty thousand hits cost more than the
    scoring itself — the caller only ever wants the head of the list.
    """
    q_terms = {t for t in _tokens(query) if len(t) > 1}
    if not q_terms or not index.n:
        return []
    if index.bm25 is not None:
        import numpy as np

        cols = [index.vocab[t] for t in q_terms if t in index.vocab]
        if not cols:
            return []
        # A sparse matrix-VECTOR product, not a column slice. The matrix
        # is row-major (CSR), so slicing columns out of it copies them
        # first; multiplying by a sparse query vector reads each row once
        # and never materialises an intermediate.
        from scipy import sparse

        q_vec = sparse.csr_matrix(
            (
                np.ones(len(cols), dtype="float32"),
                (cols, np.zeros(len(cols), dtype="int32")),
            ),
            shape=(index.bm25.shape[1], 1),
        )
        summed = np.asarray((index.bm25 @ q_vec).todense()).ravel()
        hit = np.flatnonzero(summed)
        if hit.size == 0:
            return []
        want = min(top, hit.size) if top else hit.size
        if want < hit.size:
            # Partition first: finding the top N is linear, sorting the
            # whole set is not, and the tail is discarded anyway.
            hit = hit[np.argpartition(-summed[hit], want - 1)[:want]]
        # Ties are common — identical boilerplate documents score
        # identically — so the tie-break is pinned to corpus position.
        # An unstable sort here would make results depend on which
        # machine ran the query, and quietly break the eval harness.
        order = hit[np.lexsort((hit, -summed[hit]))]
        return [(index.chunks[int(i)], float(summed[int(i)])) for i in order]

    k1, b = 1.5, 0.75
    scores: dict[int, float] = defaultdict(float)
    for term in q_terms:
        postings = index.postings.get(term)
        if not postings:
            continue
        df = index.df[term]
        idf = math.log(1 + (index.n - df + 0.5) / (df + 0.5))
        for pos, freq in postings:
            norm = 1 - b + b * index.lengths[pos] / index.avgdl
            scores[pos] += idf * (freq * (k1 + 1)) / (freq + k1 * norm)
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    if top:
        ranked = ranked[:top]
    return [(index.chunks[pos], score) for pos, score in ranked]


def _vector_ranking(
    query: str, index: _Index, *, top: int = 0
) -> list[tuple[Chunk, float]]:
    qe = _embed(query)
    if not qe or not index.n:
        return []
    matrix = index.matrix
    if matrix is not None and len(qe) == matrix.shape[1]:
        import numpy as np

        q = np.asarray(qe, dtype="float32")
        qnorm = float(np.linalg.norm(q)) or 1e-9
        sims = matrix @ (q / qnorm)
        # argpartition finds the top-N without sorting all N rows.
        want = min(top or index.n, index.n)
        if want < index.n:
            idx = np.argpartition(-sims, want - 1)[:want]
        else:
            idx = np.arange(index.n)
        idx = idx[sims[idx] > 0.02]
        idx = idx[np.argsort(-sims[idx])]
        return [(index.chunks[int(i)], float(sims[int(i)])) for i in idx]
    qnorm = math.sqrt(sum(x * x for x in qe)) or 1e-9
    scored: list[tuple[Chunk, float]] = []
    for pos, chunk in enumerate(index.chunks):
        emb = chunk.embedding
        cnorm = index.norms[pos]
        if not emb or not cnorm:
            continue
        dot = sum(x * y for x, y in zip(qe, emb, strict=False))
        sim = dot / (qnorm * cnorm)
        if sim > 0.02:
            scored.append((chunk, sim))
    scored.sort(key=lambda x: -x[1])
    return scored


def _graph_entities(conn, query: str, user_email: str) -> list[str]:
    """Entities the graph associates with the query (best-effort)."""
    try:
        from agent import enterprise_graph as eg

        hits = eg.resolve_entity(
            conn, query, scopes=eg.visible_scopes(user_email), k=2
        )
        names: list[str] = []
        for ent in hits:
            for edge in eg.neighbors(
                conn, ent["name"], scopes=eg.visible_scopes(user_email), hops=1, limit=6
            ):
                for side in (edge["src"], edge["dst"]):
                    if side.lower() != ent["name"].lower() and side not in names:
                        names.append(side)
        return names[:GRAPH_EXPANSION_ENTITIES]
    except Exception:  # noqa: BLE001
        log.debug("graph expansion unavailable", exc_info=True)
        return []


#: Adjacent chunks of top fused hits are admitted alongside results.
#: Document-level metrics cannot see this by construction — a neighbour
#: belongs to a document already retrieved — but it widens the model's
#: window of the RIGHT document, which is what the `completeness` class
#: (answers spread across a thread) actually needs from us.
#:
#: Deliberately absent: multi-query expansion. It was built and measured
#: on EnterpriseRAG-Bench (deterministic rare-core + bigram variants,
#: RRF-fused at 0.6 weight): overall recall@5 0.765 -> 0.737 and the
#: `semantic` class it targeted fell 0.457 -> 0.343 — the variants voted
#: for distractors harder than they recalled the gold. A knob that
#: measured harmful does not ship switched off; it gets deleted, and
#: this note is so the next person does not rebuild it blind. If it
#: returns, it returns with an LLM paraphraser and a fresh measurement.
NEIGHBOR_STITCH_TOP = 10
STITCH_MODE = os.environ.get("ALEX_NEIGHBOR_STITCH", "on").strip().lower()


def retrieve(
    conn,
    query: str,
    *,
    user_email: str = "",
    k: int = DEFAULT_K,
    scopes: list[str] | None = None,
    capabilities: set[str] | None = None,
    use_graph: bool = True,
    principal=None,
    rerank: bool = True,
) -> list[Retrieved]:
    """Hybrid retrieval: dense + BM25 + graph expansion, fused with RRF.

    Returns at most ``k`` :class:`Retrieved` passages, each carrying its
    citation and the ranks that produced it.

    Retrieval is **two-stage** by default. Stage one (dense + BM25 +
    graph, fused with RRF) is tuned for recall: its only job is to get
    the right passage into a wide candidate pool. Stage two reranks that
    pool on signals fusion cannot see. Measured at a hundred thousand
    documents, the second stage took MRR from 0.900 to 1.000 for about
    two milliseconds — it is not an optional extra, it is where the
    ordering is actually decided.

    When ``principal`` (an :class:`agent.access.Principal`) is supplied,
    every surviving passage is checked against the authorization
    decision point and the decision is audited — scope pre-filtering is
    an optimisation, not the security boundary. Callers that pass only
    ``user_email`` get scope isolation but no classification gate, which
    is why the VM and every API route pass a principal.
    """
    if not (query or "").strip():
        return []
    if principal is not None:
        scopes = scopes if scopes is not None else _access().visible_scopes(
            principal
        )
    scopes = scopes if scopes is not None else visible_scopes(user_email)
    index = _get_index(conn, scopes, capabilities)
    if not index.n:
        return []
    # The fusion window must be wide enough that a passage ranked well by
    # only one retriever still contributes. Too narrow a window silently
    # drops exactly the single-list hits RRF is worst at already.
    #
    # When a reranker will run, the pool is widened further: stage one's
    # only job is to get the right passage *into* the pool, and stage
    # two decides the order. A pool of k is a single-stage system with
    # extra steps.
    limit = max(k * CANDIDATE_MULTIPLIER, FUSION_WINDOW)
    if rerank:
        limit = max(limit, CANDIDATE_POOL)

    vec = _vector_ranking(query, index, top=limit)[:limit]
    lex = _bm25_ranking(query, index, top=limit)

    graph_hits: list[tuple[Chunk, float]] = []
    entities: list[str] = []
    if use_graph:
        entities = _graph_entities(conn, query, user_email)
        if entities:
            # Reuses the same index — graph expansion costs one extra
            # postings walk, not a second corpus scan.
            graph_hits = _bm25_ranking(
                " ".join(entities), index, top=GRAPH_LIST_CAP
            )
            # Rank-based fusion throws away magnitude, so a graph
            # neighbourhood that matches nothing still casts a
            # full-strength rank-1 vote. Drop the whole list unless its
            # best hit clears a floor, and keep it short so it cannot
            # outvote the two primary retrievers.
            if graph_hits and graph_hits[0][1] < GRAPH_SCORE_FLOOR:
                graph_hits = []

    fused: dict[str, dict[str, Any]] = {}

    def fuse(ranking: list[tuple[Chunk, float]], slot: str, weight: float) -> None:
        for rank, (chunk, _score) in enumerate(ranking, start=1):
            entry = fused.setdefault(
                chunk.chunk_id,
                {"chunk": chunk, "score": 0.0, "vector": None,
                 "lexical": None, "graph": None, "pinned": False},
            )
            entry["score"] += weight / (RRF_K + rank)
            if entry[slot] is None:
                entry[slot] = rank

    fuse(vec, "vector", 1.0)
    fuse(lex, "lexical", 1.0)
    fuse(graph_hits, "graph", GRAPH_WEIGHT)

    # RRF systematically penalises a passage that only one retriever
    # found, however emphatically it found it. That is precisely the
    # exact-match case — an error code, a ticket id, a person's name —
    # where the lexical top hit is the right answer and the dense
    # ranking has no opinion. Guarantee the top lexical hits a place.
    for chunk, _score in lex[:LEXICAL_PIN_SLOTS]:
        entry = fused.get(chunk.chunk_id)
        if entry is not None:
            entry["pinned"] = True

    # Authorize before truncating to k, so a denied passage is replaced
    # by the next permitted one instead of silently shrinking the result.
    ordered = sorted(
        fused.values(), key=lambda e: (not e["pinned"], -e["score"])
    )
    results = [
        Retrieved(
            chunk=e["chunk"],
            score=e["score"],
            vector_rank=e["vector"],
            lexical_rank=e["lexical"],
            graph_rank=e["graph"],
            via_entities=list(entities) if e["graph"] is not None else [],
        )
        for e in ordered
    ]
    if principal is not None:
        results = _authorize(principal, results, query)

    # Neighbour stitching: the measured `completeness` failure is an
    # answer spread across a thread — the retrieved chunk is the right
    # document but the decisive sentence sits one chunk over, where
    # neither ranker saw enough of the query to vote for it. Adjacent
    # chunks of the strongest hits enter the pool as candidates only:
    # the reranker (or the score they inherit at a discount) decides
    # whether they earn a place, and authorization applies to them the
    # same as anything retrieval found.
    if STITCH_MODE != "off" and index.n > 1:
        by_pos = index.by_position
        have = {r.chunk.chunk_id for r in results}
        stitched: list[Retrieved] = []
        for r in results[:NEIGHBOR_STITCH_TOP]:
            for delta in (-1, 1):
                neighbor = by_pos.get(
                    (r.chunk.doc_id, r.chunk.chunk_index + delta)
                )
                if neighbor is None or neighbor.chunk_id in have:
                    continue
                have.add(neighbor.chunk_id)
                stitched.append(
                    Retrieved(chunk=neighbor, score=r.score * 0.5)
                )
        if stitched and principal is not None:
            stitched = _authorize(principal, stitched, query)
        results.extend(stitched)

    if rerank:
        # Stage two. Reranking only reorders what retrieval and
        # authorization already allowed — it can neither introduce a
        # passage nor resurrect one that was filtered out.
        #
        # It is given the corpus's term rarity because that is the
        # signal fusion loses: BM25 knows document frequency but its
        # length normalisation can bury a rare match, and the dense side
        # has no notion of rarity at all.
        from agent.rerank import rerank as _rerank

        results = _rerank(query, results, rarity=_rarity(query, index))
    return results[:k]


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


def context_block(results: list[Retrieved], *, max_chars: int = 4000) -> str:
    """Render retrieved passages as a cited prompt block.

    Each passage is numbered so the model can cite ``[1]``, ``[2]`` and
    the UI can resolve the number back to a document.
    """
    if not results:
        return ""
    lines = ["ENTERPRISE KNOWLEDGE (cite by number):"]
    used = len(lines[0])
    for i, r in enumerate(results, start=1):
        cite = r.chunk.title or r.chunk.source_ref or r.chunk.doc_id
        body = r.chunk.text.strip()
        entry = f"[{i}] {cite}\n{body}"
        if used + len(entry) > max_chars:
            remaining = max_chars - used - len(cite) - 12
            if remaining < 120:
                break
            entry = f"[{i}] {cite}\n{body[:remaining]}…"
        lines.append(entry)
        used += len(entry) + 1
    return "\n".join(lines)


def search(
    conn,
    query: str,
    *,
    user_email: str = "",
    k: int = DEFAULT_K,
    capabilities: set[str] | None = None,
    principal=None,
) -> dict[str, Any]:
    """Tool-facing search: JSON-serialisable results with citations."""
    hits = retrieve(
        conn, query, user_email=user_email, k=k, capabilities=capabilities,
        principal=principal,
    )
    return {
        "query": query,
        "results": [h.to_dict() for h in hits],
        "count": len(hits),
    }


def new_doc_id() -> str:
    return "doc-" + uuid.uuid4().hex[:16]
