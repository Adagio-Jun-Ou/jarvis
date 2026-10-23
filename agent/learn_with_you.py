"""
Learn-with-you: capture a correction (user edited a prior AI draft) and
distill it into a preference memory and a candidate skill.

The full OBSERVE → GENERALIZE → RATIFY pipeline lives in
`pipelines/skill_reflection.py`. This module is the *capture* step that
runs in the agent's turn — record the diff, write the preference to
vector memory, and surface a "would you like to ratify this as a skill?"
prompt only if the user already had a similar memory.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass

log = logging.getLogger("agent.learn_with_you")


@dataclass
class Correction:
    prompt: str
    on_result: str


def capture_correction(
    store_fn: Callable, user_email: str, original: str, edited: str, ref: str = "turn"
) -> Correction:
    """Record a user correction. `store_fn(owner, kind, content, source_ref)`
    is provided by the caller (typically `vector_memory.store`).

    Returns a `Correction` describing what to do with the captured diff:
      * `prompt` is what the UI shows ("Saved as a preference. Want this
        to become a skill?").
      * `on_result` is a follow-up the UI may dispatch. The agent does
        NOT call it directly — the user clicks to ratify.

    Crucially, this function only WRITES the preference; it does NOT
    self-grant a skill. The skill is proposed (status='proposed') by the
    nightly reflection job and ratified by the user/governor.
    """
    if not user_email or not edited:
        return Correction(prompt="", on_result="")
    if not edited.strip() or edited.strip() == (original or "").strip():
        return Correction(prompt="", on_result="")
    diff = _short_diff(original or "", edited)
    if not diff:
        return Correction(prompt="", on_result="")
    try:
        store_fn(
            user_email,
            "preference",
            f"When the user edits AI drafts in this style, they prefer: {diff}",
            ref or f"turn-{int(time.time())}",
        )
    except Exception:
        log.exception("learn_with_you.store failed")
        return Correction(prompt="", on_result="")
    return Correction(
        prompt="Saved as a preference.",
        on_result=(
            "ratify_skill_suggested" if _looks_skillable(diff) else "preference_only"
        ),
    )


def teach_routine(store_fn, user_email: str, prompt: str, context: str = "") -> str:
    """Capture a /teach command. The user is explicitly walking the
    agent through a procedure.

    Returns the routine_id."""
    if not user_email or not prompt:
        return ""
    rid = str(uuid.uuid4())
    try:
        store_fn(user_email, "taught_routine", prompt, f"taught-{rid}")
    except Exception:
        log.exception("teach_routine.store failed")
        return ""
    return rid


def match_routine(
    sql, user_email: str, message: str, *, threshold: float = 0.3
) -> dict | None:
    """Find the taught routine that best lexically matches the message.

    Taught routines are stored by :func:`teach_routine` (via
    ``POST /api/teach``) as ``kind='taught_routine'`` rows in the
    vector-memory mirror.  Returns ``{"routine_id", "steps", "score"}``
    for the best match at or above ``threshold``, else ``None``.
    ``sql`` is a callable ``(statement, parameters) -> list[dict]``.
    """
    if not user_email or not message:
        return None
    try:
        rows = (
            sql(
                "SELECT content, source_ref "
                "FROM media.control.vector_memory_mirror "
                "WHERE owner_email = :u AND kind = 'taught_routine' "
                "AND superseded_by IS NULL",
                {"u": user_email},
            )
            or []
        )
    except Exception:
        log.debug("match_routine query failed", exc_info=True)
        return None
    from agent.learning_loop import _tokens

    msg_tokens = set(_tokens(message))
    if not msg_tokens:
        return None
    best: dict | None = None
    best_score = 0.0
    for r in rows:
        content = str(r.get("content") or "")
        routine_tokens = set(_tokens(content))
        if not routine_tokens:
            continue
        score = len(msg_tokens & routine_tokens) / len(msg_tokens)
        if score >= threshold and score > best_score:
            ref = str(r.get("source_ref") or "")
            best = {
                "routine_id": ref.removeprefix("taught-") or ref,
                "steps": content,
                "score": round(score, 3),
            }
            best_score = score
    return best


def routine_context_block(routine: dict) -> str:
    """Render a matched routine as a prompt context block."""
    steps = str((routine or {}).get("steps") or "").strip()
    if not steps:
        return ""
    return (
        "TAUGHT ROUTINE (the user explicitly taught you this procedure "
        "— follow these steps, using your tools where they apply):\n"
        + steps[:800]
    )


def should_suggest(user_email: str, last_msg: str) -> bool:
    """Heuristic: should the agent offer a teaching moment?
    Conservative — we do not pester. Used to gate the in-chat tip."""
    if not last_msg:
        return False
    low = last_msg.lower()
    return any(
        t in low
        for t in (
            "show me how",
            "explain how",
            "what's the right way",
            "what's the process",
            "can you walk me through",
        )
    )


def suggestion_feedback(
    sql, user_email: str, suggestion_id: str, feedback: str, comment: str = ""
) -> None:
    """Record the user's response to a suggestion. `feedback` is one of
    'helpful' | 'unhelpful' | 'inaccurate' | 'other'."""
    sql(
        """
        INSERT INTO media.control.suggestion_feedback
          (id, user_email, suggestion_id, feedback, comment)
        VALUES (:id, :u, :s, :f, :c)
        """,
        parameters={
            "id": str(uuid.uuid4()),
            "u": user_email,
            "s": suggestion_id,
            "f": feedback,
            "c": (comment or "")[:1000],
        },
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _short_diff(original: str, edited: str) -> str:
    """Return a short human-readable description of the diff.
    This is not a full diff; it's a one-line summary for the preference."""
    o = (original or "").strip()
    e = edited.strip()
    if not o:
        return f"new content ({len(e)} chars)"
    if e.startswith(o):
        return f"appended: {e[len(o):].strip()[:120]}"
    if o.startswith(e):
        return f"trimmed leading by {len(o) - len(e)} chars"
    # crude line-level diff
    o_lines = [line.strip() for line in o.splitlines() if line.strip()]
    e_lines = [line.strip() for line in e.splitlines() if line.strip()]
    if o_lines and e_lines and o_lines[0] != e_lines[0]:
        return f"first line changed: {o_lines[0][:60]!r} -> {e_lines[0][:60]!r}"
    return f"edited ({len(o)} -> {len(e)} chars)"


def _looks_skillable(diff: str) -> bool:
    """Heuristic: is this a structural correction (worth proposing as a
    skill) or just a one-off preference?"""
    if "first line changed" in diff:
        return True
    if "appended" in diff and ":" in diff:
        return True
    return False
