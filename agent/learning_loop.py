"""
Active skill hints: when the user types a message, the agent can match it
against its per-user learned skills and prepend a short, prompt-cached
hint that says "for this kind of task, follow this proven procedure".

The skills themselves are written by `pipelines/skill_reflection.py`
(OBSERVE → GENERALIZE) and ratified by the user/governor before they
become 'active'. This module only READS.
"""

from __future__ import annotations

import logging

log = logging.getLogger("agent.learning_loop")


def active_skill_hints(
    sql, owner_email: str, last_user_msg: str, *, budget_chars: int = 600
) -> str:
    """Return a compact string of the user's active skills that match the
    last user message. Empty string if no skills match.

    `sql` is a callable (statement, parameters) -> list[dict]."""
    if not owner_email or not last_user_msg:
        return ""
    msg_tokens = set(_tokens(last_user_msg))
    if not msg_tokens:
        return ""
    try:
        rows = sql(
            """
            SELECT skill_id, name, trigger_pattern, procedure
              FROM media.control.skills
             WHERE owner_email = :u
               AND status = 'active'
               AND (last_used IS NULL OR last_used > current_timestamp() - INTERVAL 90 DAYS)
            """,
            parameters={"u": owner_email},
        )
    except Exception:
        log.exception("skill_hints: SELECT failed")
        return ""
    matches: list[tuple[float, dict]] = []
    for r in rows:
        score = _match_score(
            msg_tokens, r.get("trigger_pattern") or "", r.get("name") or ""
        )
        if score > 0:
            matches.append((score, r))
    if not matches:
        return ""
    matches.sort(key=lambda x: -x[0])
    lines = ["LEARNED PROCEDURES (follow these; you ratified them):"]
    used = 0
    for _, r in matches:
        proc = r.get("procedure")
        proc_str = " → ".join(proc) if isinstance(proc, list) else str(proc or "")
        line = f"- {r.get('name','')}: {proc_str}"
        if used + len(line) + 1 > budget_chars:
            break
        lines.append(line)
        used += len(line) + 1
    if len(lines) == 1:
        return ""
    return "\n".join(lines)


# High-frequency function words (>2 chars, so not already dropped by the
# length filter) that otherwise cause false matches: e.g. a skill trigger
# "send the report" would match an unrelated message just because both
# contain "the". Shared by learning_loop and agent.anticipation.
_STOPWORDS = frozenset({
    "the", "and", "for", "you", "your", "with", "have", "has", "had", "this",
    "that", "these", "those", "from", "are", "was", "were", "will", "would",
    "can", "could", "should", "our", "their", "its", "but", "not", "all",
    "any", "been", "they", "them", "then", "than", "out", "get", "got",
    "into", "per", "via", "about", "please", "just", "let", "make", "made",
    "need", "want", "some", "such", "who", "what", "when", "where", "how",
    "why", "which", "here", "there", "also", "one", "two", "new", "now",
})


def _keep(t: str) -> bool:
    return len(t) > 2 and t not in _STOPWORDS


def _tokens(text: str) -> list[str]:
    out: list[str] = []
    cur = []
    for ch in text.lower():
        if ch.isalnum():
            cur.append(ch)
        else:
            if cur:
                t = "".join(cur)
                if _keep(t):
                    out.append(t)
                cur = []
    if cur:
        t = "".join(cur)
        if _keep(t):
            out.append(t)
    return out


def _match_score(msg_tokens: set, trigger: str, name: str) -> float:
    pat_tokens = set(_tokens(trigger + " " + name))
    if not pat_tokens:
        return 0.0
    inter = msg_tokens & pat_tokens
    if not inter:
        return 0.0
    return len(inter) / len(pat_tokens)
