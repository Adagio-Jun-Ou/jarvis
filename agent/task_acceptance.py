"""Acceptance criteria — the human's definition of done.

The plan (:mod:`agent.task_plan`) is the agent's roadmap; acceptance
criteria are the *person's* bar for the work. The asymmetry between them
is the control property and it is deliberate:

* people WRITE the criteria, through the board API;
* the agent may only MARK a criterion ``met`` or ``unmet``, with the
  evidence — it cannot add, remove, or reword them. An agent that could
  edit its own definition of done would grade its own homework.

The finish gate in :mod:`agent.taskboard` reports every criterion not
marked ``met`` in the run summary — the text the reviewer reads when
deciding review->done. ``unmet`` is a first-class answer, not a
failure: "I checked, and it does not hold, because…" is exactly what a
reviewer needs said out loud.

Binding: like the plan tools, ``check_acceptance`` resolves its task
from the invoking VM's session, never from a model-supplied argument.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("agent.task_acceptance")

ACCEPTANCE_TABLE = "media.control.task_acceptance"
STATUSES = ("open", "met", "unmet")
MAX_CRITERIA = 8
MAX_TEXT = 300
MAX_NOTE = 400


def _sql(stmt: str, params: dict | None = None) -> list[dict]:
    from agent.tools.impls import _sql as impl_sql

    return impl_sql(stmt, params)


def get_criteria(task_id: str) -> list[dict[str, Any]]:
    if not task_id:
        return []
    rows = _sql(
        f"SELECT crit_no, text, status, note FROM {ACCEPTANCE_TABLE} "
        "WHERE task_id = :t",
        {"t": task_id},
    )
    rows.sort(key=lambda r: int(r.get("crit_no") or 0))
    return [
        {
            "crit_no": int(r["crit_no"]),
            "text": str(r.get("text") or ""),
            "status": str(r.get("status") or "open"),
            "note": str(r.get("note") or ""),
        }
        for r in rows
    ]


def set_criteria(task_id: str, criteria: list[str]) -> list[dict[str, Any]]:
    """Replace the task's criteria (the human write path).

    Replacement resets statuses to ``open`` on purpose: rewording the bar
    invalidates any evidence gathered against the old wording.
    """
    texts = [str(c).strip()[:MAX_TEXT] for c in (criteria or []) if str(c).strip()]
    if len(texts) > MAX_CRITERIA:
        raise ValueError(
            f"{len(texts)} criteria is over the {MAX_CRITERIA}-criterion cap"
        )
    _sql(f"DELETE FROM {ACCEPTANCE_TABLE} WHERE task_id = :t", {"t": task_id})
    for i, text in enumerate(texts, start=1):
        _sql(
            f"INSERT INTO {ACCEPTANCE_TABLE} "
            "(task_id, crit_no, text, status, note, updated_at) "
            "VALUES (:t, :n, :x, 'open', '', current_timestamp())",
            {"t": task_id, "n": i, "x": text},
        )
    return get_criteria(task_id)


def mark(
    task_id: str, crit_no: int, status: str, note: str = ""
) -> dict[str, Any] | None:
    """Record whether one criterion holds (the agent's — or a person's —
    verdict). ``met`` should carry the evidence; ``unmet`` what stands in
    the way."""
    status = (status or "").strip().lower()
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}")
    _sql(
        f"UPDATE {ACCEPTANCE_TABLE} SET status = :s, note = :no, "
        "updated_at = current_timestamp() "
        "WHERE task_id = :t AND crit_no = :n",
        {"s": status, "no": (note or "")[:MAX_NOTE], "t": task_id,
         "n": int(crit_no)},
    )
    for crit in get_criteria(task_id):
        if crit["crit_no"] == int(crit_no):
            return crit
    return None


def not_met(task_id: str) -> list[dict[str, Any]]:
    """Criteria the reviewer cannot yet rely on: unchecked or failing."""
    return [c for c in get_criteria(task_id) if c["status"] != "met"]


def prompt_block(task_id: str) -> str:
    """The criteria rendered for a run prompt."""
    criteria = get_criteria(task_id)
    if not criteria:
        return ""
    marks = {"met": "[x]", "unmet": "[!]", "open": "[ ]"}
    lines = [
        f"  {marks.get(c['status'], '[ ]')} {c['crit_no']}. {c['text']}"
        + (f" — {c['note']}" if c["note"] else "")
        for c in criteria
    ]
    return (
        "ACCEPTANCE CRITERIA (the task counts as done only when every one "
        "of these holds; verify each with check_acceptance — 'met' with "
        "the evidence, 'unmet' with what stands in the way):\n"
        + "\n".join(lines)
    )


# ---------------------------------------------------------------------------
# The agent-facing tool — session-bound, mark-only
# ---------------------------------------------------------------------------


def _bound_task() -> str:
    from agent.swarm import current_vm
    from agent.task_plan import task_for_session

    vm = current_vm()
    if vm is None:
        return ""
    return task_for_session(getattr(vm, "session_id", ""))


def check_acceptance(
    crit_no: int,
    status: str,
    note: str = "",
    *,
    user_email: str = "",
    instance_scope: str = "default",
) -> dict[str, Any]:
    """Record whether one acceptance criterion of your current task holds:
    'met' with the evidence, 'unmet' with what stands in the way. You
    cannot add or change the criteria themselves — the crew writes those."""
    task_id = _bound_task()
    if not task_id:
        return {"ok": False, "error": "this session has no task"}
    try:
        crit = mark(task_id, crit_no, status, note)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    if crit is None:
        return {"ok": False, "error": f"no criterion {crit_no} on this task"}
    _emit(task_id)
    return {"ok": True, "criterion": crit}


def _emit(task_id: str) -> None:
    try:
        from agent.swarm import current_vm
        from alex.events import acceptance_event

        vm = current_vm()
        if vm is not None:
            vm.pump.emit(
                acceptance_event(task_id=task_id, criteria=get_criteria(task_id))
            )
    except Exception:  # noqa: BLE001
        log.debug("acceptance event emit failed", exc_info=True)
