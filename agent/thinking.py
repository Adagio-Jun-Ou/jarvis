"""
Thinking archive: persist the raw chain-of-thought to the append-only
`media.control.thinking_traces` table and compute a small decision
digest for the next turn's prompt.

The function `archive` is the only function the agent calls after a turn
that produced visible reasoning.
"""

from __future__ import annotations

import logging
import uuid

log = logging.getLogger("agent.thinking")


def archive(
    sql, user_email: str, session_id: str, phase: str, raw_thinking: str, digest: str
) -> None:
    """Persist the raw thinking and digest. The trace id is returned to
    the caller if it wants to reference it later."""
    if not user_email or not session_id or not raw_thinking:
        return
    # Bound the raw_thinking to 60k chars. The column allows more but
    # the convention is "60k chars raw, 80 token digest".
    bounded = raw_thinking[:60_000]
    sql(
        """
        INSERT INTO media.control.thinking_traces
          (id, ts, user_email, session_id, turn_no, phase, raw_thinking, digest)
        VALUES (:id, current_timestamp(), :u, :s, NULL, :p, :r, :d)
        """,
        parameters={
            "id": str(uuid.uuid4()),
            "u": user_email,
            "s": session_id,
            "p": phase,
            "r": bounded,
            "d": (digest or "")[:4000],
        },
    )


def digest_only_for_context(summary: dict) -> str:
    """Compute a short, prompt-cache-friendly digest of the turn. The
    digest is a few sentences — what the model decided and why."""
    if not isinstance(summary, dict):
        return ""
    s = str(summary.get("summary") or "").strip()
    if not s:
        return ""
    # Hard cap: ~80 tokens ~= 320 chars.
    if len(s) > 320:
        s = s[:317] + "..."
    return s
