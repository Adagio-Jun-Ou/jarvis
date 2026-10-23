"""
Memory helpers — per-user vector + graph memory.

This is a thin client wrapper around the agent's memory API. The
backend storage is Lakebase (pgvector + edges table) on Databricks;
the SDK is the client-side convenience for "remember this" and
"what do you know about me".
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alex_sdk.client import Alex

log = logging.getLogger("alex_sdk.memory")


@dataclass
class MemoryBead:
    id: str
    owner_email: str
    kind: str
    content: str
    salience: float = 0.5
    source_ref: str = ""
    created_at: str | None = None
    last_recalled: str | None = None
    superseded_by: str | None = None


@dataclass
class GraphEdge:
    edge_id: str
    owner_email: str
    src: str
    rel: str
    dst: str
    evidence: str = ""
    confidence: float = 0.5


class Memory:
    """Per-user memory API. Each instance is bound to one user (the
    SDK consumer's session identity). All writes are scoped to that
    user; reads are too.

    The implementation talks to the app backend over HTTP. The
    backend is the source of truth; the SDK does not embed or
    vectorize locally.
    """

    def __init__(self, alex: Alex, owner_email: str) -> None:
        self._alex = alex
        self.owner = owner_email
        self._client = alex._http

    # ---- vector -----------------------------------------------------------

    def remember(
        self, kind: str, content: str, *, source_ref: str = "", salience: float = 0.5
    ) -> MemoryBead:
        """Persist a memory bead. The backend is responsible for
        embedding and (de-)duplication."""
        r = self._client.post(
            "/api/sdk/remember",
            json={
                "kind": kind,
                "content": content,
                "source_ref": source_ref,
                "salience": salience,
            },
        )
        r.raise_for_status()
        return MemoryBead(
            id=str(uuid.uuid4()),
            owner_email=self.owner,
            kind=kind,
            content=content,
            salience=salience,
            source_ref=source_ref,
        )

    def recall(self, query: str, *, k: int = 10) -> list[MemoryBead]:
        """Top-k memory beads relevant to `query`."""
        r = self._client.post("/api/sdk/recall", json={"query": query, "k": k})
        r.raise_for_status()
        return [
            MemoryBead(
                id=str(uuid.uuid4()),
                owner_email=self.owner,
                kind=m.get("kind", ""),
                content=m.get("content", ""),
                salience=float(m.get("salience") or 0.0),
                source_ref=m.get("source_ref", "") or "",
                created_at=m.get("created_at"),
            )
            for m in r.json().get("memories", [])
        ]

    def forget(self, item_id: str, *, reason: str = "user_requested") -> None:
        """Request a memory or edge be marked superseded. The
        consolidation job honors the request; the call is async."""
        r = self._client.post(
            "/api/transparency/forget", json={"item_id": item_id, "reason": reason}
        )
        r.raise_for_status()

    # ---- graph ------------------------------------------------------------

    def add_edge(
        self, src: str, rel: str, dst: str, *, evidence: str = "", confidence: float = 0.5
    ) -> GraphEdge:
        """Create a knowledge-graph edge between `src` and `dst`."""
        r = self._client.post(
            "/api/sdk/add_edge",
            json={
                "src": src,
                "rel": rel,
                "dst": dst,
                "evidence": evidence,
                "confidence": confidence,
            },
        )
        r.raise_for_status()
        return GraphEdge(
            edge_id=str(uuid.uuid4()),
            owner_email=self.owner,
            src=src,
            rel=rel,
            dst=dst,
            evidence=evidence,
            confidence=confidence,
        )

    def query(self, text: str, *, limit: int = 8) -> list[GraphEdge]:
        """Edges related to any entity in `text`.

        The app backend has no dedicated graph-query route yet; this
        reads the user's knowledge-graph mirror via
        ``GET /api/transparency/graph`` and filters client-side.
        """
        r = self._client.get("/api/transparency/graph")
        r.raise_for_status()
        needle = text.lower()
        out: list[GraphEdge] = []
        for e in r.json().get("edges", []):
            hay = f"{e.get('src', '')} {e.get('rel', '')} {e.get('dst', '')}".lower()
            if not needle or any(tok in hay for tok in needle.split()):
                out.append(
                    GraphEdge(
                        edge_id=str(uuid.uuid4()),
                        owner_email=self.owner,
                        src=e.get("src", ""),
                        rel=e.get("rel", ""),
                        dst=e.get("dst", ""),
                        evidence=e.get("evidence", "") or "",
                        confidence=float(e.get("confidence") or 0.0),
                    )
                )
            if len(out) >= limit:
                break
        return out
