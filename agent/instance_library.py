"""
Instance library: the curated per-employee corpus. Reads are cheap
(a single SELECT); writes happen via the `media.control.instance_library`
table and are governed by `pipelines/library_sync.py`.

The agent reads this when answering questions where the company's
approved file is the source of truth.
"""

from __future__ import annotations

import logging

log = logging.getLogger("agent.instance_library")


def library_for(sql, user_email: str) -> list[dict]:
    """Return the user's instance library as a list of dicts."""
    if not user_email:
        return []
    try:
        rows = sql(
            """
            SELECT il.file_id, il.pinned_summary, il.added_at,
                   af.path, af.sha256, af.size_bytes, af.source_channel,
                   af.approved_at, af.expires_at
              FROM media.control.instance_library il
              JOIN media.control.approved_files af ON af.file_id = il.file_id
             WHERE il.owner_email = :u
               AND af.revoked = false
               AND (af.expires_at IS NULL OR af.expires_at > current_timestamp())
             ORDER BY il.added_at DESC
             LIMIT 100
            """,
            parameters={"u": user_email},
        )
    except Exception:
        log.exception("instance_library.library_for failed")
        return []
    return [dict(r) for r in rows]


def verify_hosted(sql, path: str, content_hash: str) -> bool:
    """Verify that a file at `path` with `content_hash` is on the approved
    list. Returns True if found and not revoked."""
    if not path or not content_hash:
        return False
    try:
        rows = sql(
            """
            SELECT 1 FROM media.control.approved_files
             WHERE path = :p AND sha256 = :h AND revoked = false
             LIMIT 1
            """,
            parameters={"p": path, "h": content_hash},
        )
    except Exception:
        log.exception("instance_library.verify_hosted failed")
        return False
    return bool(rows)


def context_gist(library: list[dict]) -> str:
    """A short summary of the user's instance library, suitable for the
    system prompt. Returns an empty string if the user has no library."""
    if not library:
        return ""
    parts = ["YOUR APPROVED FILES (instance library):"]
    for entry in library[:8]:
        s = entry.get("pinned_summary") or entry.get("path", "")
        parts.append(f"- {s[:120]}")
    return "\n".join(parts)
