"""
Goal anticipation: a deterministic, per-turn "what likely matters now"
context slice.

Before each turn the VM calls :func:`anticipation_block` which computes
cheap signals from the warehouse:

* overdue or due-soon (48h) P1/P2 tasks owned by the user,
* unread P1 inbox items whose subject matches an open task,
* an unconsumed standing-order brief.

The result is a compact block (max ~400 chars of signals) with an
explicit instruction so the model leads with it only when it is
relevant to the user's message.  Every failure degrades to "" — the
slice is strictly additive and never breaks a turn.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

log = logging.getLogger("agent.anticipation")

MAX_SIGNAL_CHARS = 400
DUE_SOON_HOURS = 48

_HEADER = "WHAT LIKELY MATTERS NOW:"
_FOOTER = (
    "(If relevant to the user's message, lead with it; "
    "otherwise ignore silently.)"
)


def _rows(sql, stmt: str, params: dict) -> list[dict]:
    try:
        return list(sql(stmt, params) or [])
    except Exception:  # noqa: BLE001
        log.debug("anticipation query failed", exc_info=True)
        return []


def _parse_ts(value) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if not value:
        return None
    text = str(value).strip().replace("T", " ")
    for cut in ("+", "Z"):
        if cut in text:
            text = text.split(cut)[0]
    try:
        return datetime.fromisoformat(text.strip())
    except ValueError:
        return None


def _tokens(text: str) -> set[str]:
    from agent.learning_loop import _tokens as toks

    return set(toks(text or ""))


def _truthy(value) -> bool:
    return str(value).strip().lower() in ("true", "1", "yes")


def anticipation_block(sql, user_email: str, now: datetime | None = None) -> str:
    """Compute the compact anticipation slice for the user, or ""."""
    if not user_email:
        return ""
    # DB timestamps (due_at, etc.) are UTC and _parse_ts strips the tz to a
    # naive value, so the reference clock must also be UTC — datetime.now()
    # (local) mis-windowed overdue/due-soon by the server's UTC offset.
    now = now or datetime.utcnow()
    signals: list[str] = []

    tasks = _rows(
        sql,
        "SELECT id, title, status, priority, due_at "
        "FROM media.alex.tasks WHERE owner_email = :u",
        {"u": user_email},
    )
    # Live = not terminal: the table carries board lanes (todo/backlog/
    # review) alongside the legacy statuses, and enumerating only the
    # legacy pair made every board task invisible to anticipation.
    open_tasks = [
        t for t in tasks
        if str(t.get("status", "")) not in ("done", "cancelled")
    ]
    urgent: list[tuple[int, str]] = []
    for t in open_tasks:
        try:
            pri = int(t.get("priority") or 9)
        except (TypeError, ValueError):
            pri = 9
        if pri > 2:
            continue
        due = _parse_ts(t.get("due_at"))
        if due is None:
            continue
        title = str(t.get("title") or "")[:80]
        if due <= now:
            urgent.append((0, f"- OVERDUE P{pri} task: {title}"))
        elif due <= now + timedelta(hours=DUE_SOON_HOURS):
            urgent.append((1, f"- P{pri} task due soon: {title}"))
    urgent.sort(key=lambda x: x[0])
    signals.extend(s for _, s in urgent[:3])

    # Unread P1 inbox items whose subject matches an open task.
    task_tokens: set[str] = set()
    for t in open_tasks:
        task_tokens |= _tokens(str(t.get("title") or ""))
    if task_tokens:
        inbox = _rows(
            sql,
            "SELECT email_id, priority, subject "
            "FROM media.alex.inbox_view WHERE owner_email = :u",
            {"u": user_email},
        )
        matched = 0
        for m in inbox:
            try:
                pri = int(m.get("priority") or 9)
            except (TypeError, ValueError):
                pri = 9
            subject = str(m.get("subject") or "")
            if pri == 1 and (_tokens(subject) & task_tokens):
                signals.append(f"- P1 email matches an open task: {subject[:80]}")
                matched += 1
                if matched >= 2:
                    break

    # An unconsumed standing-order brief. The table has no `title` column
    # (see control_plane.sql); label it with the compiled_context summary.
    briefs = _rows(
        sql,
        "SELECT order_id, compiled_context, consumed "
        "FROM media.control.standing_order_briefs WHERE user_email = :u",
        {"u": user_email},
    )
    for b in briefs:
        if not _truthy(b.get("consumed")):
            signals.append(
                f"- Unconsumed standing-order brief: {str(b.get('compiled_context') or '')[:80]}"
            )
            break

    if not signals:
        return ""
    body = ""
    for s in signals:
        if len(body) + len(s) + 1 > MAX_SIGNAL_CHARS:
            break
        body += ("\n" if body else "") + s
    if not body:
        return ""
    return f"{_HEADER}\n{body}\n{_FOOTER}"


__all__ = ["anticipation_block"]
