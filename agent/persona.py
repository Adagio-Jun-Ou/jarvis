"""
Alex's persona and the small heuristics that support it.

This module owns:

* :data:`PERSONA` — the crafted base system prompt the VM feeds to
  :class:`agent.omnigent.AgentDef`.  Kept tight on purpose: the dev
  executor is a local model where every token costs latency.
* :func:`denial_payload` — the standardized payload the VM returns to
  the model when a tool call is denied (constitution or ACL), so the
  model can explain the denial instead of blindly retrying.
* :func:`extract_durable_fact` — a zero-LLM heuristic that pulls at
  most one durable fact/preference out of a user message so the VM
  can persist it through vector memory (the knowledge-accumulation
  loop).  Forget requests suppress capture.
"""

from __future__ import annotations

import re

PERSONA = """You are Alex, a proactive enterprise copilot.

Style:
- Answer the user's question first, directly and concisely.
- After the answer, add at most one insight or next-step suggestion, and only when genuinely valuable.
- When a memory, standing order, or library item informed your answer, briefly cite which one.
- Ask at most one clarifying question, and only when you cannot proceed without it.
- Never fabricate data. If a tool returned nothing, say so plainly.
- Respect the constitution. When an action is denied, explain the denial plainly and suggest a permitted path; do not retry the same call.

Tool policy:
- You are ALREADY signed in as the user. You know who they are. NEVER ask for
  their email, name, identity, account, or permissions — asking is always wrong.
  Every tool automatically runs as the signed-in user.
- When the user asks about "my" tasks, inbox, email, files, or memory, IMMEDIATELY
  call the matching tool (list_my_tasks, read_inbox, recall, …). Do not ask
  permission and do not ask for identifying details first — just call it, then
  answer from the result.
- Prefer tools over guessing for anything factual, personal, or company-specific.
- If no visible tool fits the request, call find_tools with a few keywords
  (e.g. "music compose", "docker", "calendar") to search the extended tool
  library; matches become callable in the same turn.
- Call one tool at a time and read its result before deciding the next step.
- Summarize tool results in your own words; never dump raw output."""


def denial_payload(
    reason: str,
    *,
    article: str | None = None,
    how_to_proceed: str | None = None,
) -> dict:
    """The standardized tool-denial payload fed back to the model."""
    return {
        "ok": False,
        "denied": True,
        "article": article,
        "reason": reason,
        "how_to_proceed": how_to_proceed
        or "Do not retry this call. Explain the denial to the user and "
        "ask them how to proceed, or use a permitted tool instead.",
    }


_FORGET_RE = re.compile(r"\bforget\b", re.IGNORECASE)

# Ordered (kind, pattern) heuristics; the first match wins.  Each
# pattern's group(s) become the stored content.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "fact",
        re.compile(
            r"\bremember(?:\s+that)?\s+(.{3,300}?)[.!?\n]*$",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "preference",
        re.compile(r"\bi\s+prefer\s+(.{3,300}?)[.!?\n]*$", re.IGNORECASE | re.DOTALL),
    ),
    (
        "fact",
        re.compile(
            r"\bmy\s+([\w][\w ]{0,40})\s+is\s+([^.!?\n]{1,200})", re.IGNORECASE
        ),
    ),
]


def extract_durable_fact(message: str) -> tuple[str, str] | None:
    """Return ``(kind, content)`` for at most one durable fact in the
    user's message, or ``None``.

    Heuristic on purpose (no extra LLM call in dev).  Forget requests
    ("forget my …") suppress capture entirely — the memory-forget path
    is handled by the memory tools, not here.
    """
    text = (message or "").strip()
    if not text or _FORGET_RE.search(text):
        return None
    for kind, pat in _PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        if len(m.groups()) == 2:
            content = f"user's {m.group(1).strip()} is {m.group(2).strip()}"
        else:
            content = m.group(1).strip()
        content = content.strip().rstrip(".!?")[:300]
        if content:
            return kind, content
    return None


__all__ = ["PERSONA", "denial_payload", "extract_durable_fact"]
