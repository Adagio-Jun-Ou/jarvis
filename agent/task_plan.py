"""The plan — a task run's structured spine.

Square-one purpose of this system: help employees complete tasks. A run
that streams prose helps nobody *steer*; the published pattern the
multi-agent literature converges on is plan-then-execute with
interleaved replanning — the agent declares concrete steps before
acting, checks them off as it works, and may replace the plan when the
situation changes. The plan is a first-class, durable artifact:

* the **crew** reads the same rows the agent writes, live, so a person
  steers against the plan instead of against a guess;
* the **run** resumes against it — an interrupted attempt's remaining
  steps frame the next attempt, which is plan continuity across process
  death;
* the **finish gate** is honest about staleness: a run that ends with
  pending steps says so in its summary instead of quietly declaring
  victory, because the known failure mode of plan-first agents is
  following a stale roadmap into a silent failure.

Binding: the plan tools resolve their task from the *session* of the VM
that invoked them (`task-<id>`), never from a model-supplied argument —
a model that could name a task id could rewrite another crew's plan.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("agent.task_plan")

PLANS_TABLE = "media.control.task_plans"
STATUSES = ("pending", "doing", "done", "skipped", "blocked")
MAX_STEPS = 12
MAX_TITLE = 200
MAX_NOTE = 400


def _sql(stmt: str, params: dict | None = None) -> list[dict]:
    from agent.tools.impls import _sql as impl_sql

    return impl_sql(stmt, params)


def task_for_session(session_id: str) -> str:
    """The task a session is bound to, or "".

    Board runs use ``task-<id>`` sessions; anything else has no plan.
    """
    sid = str(session_id or "")
    return sid[5:] if sid.startswith("task-") else ""


def get_plan(task_id: str) -> list[dict[str, Any]]:
    if not task_id:
        return []
    rows = _sql(
        f"SELECT step_no, title, status, note FROM {PLANS_TABLE} "
        "WHERE task_id = :t",
        {"t": task_id},
    )
    rows.sort(key=lambda r: int(r.get("step_no") or 0))
    return [
        {
            "step_no": int(r["step_no"]),
            "title": str(r.get("title") or ""),
            "status": str(r.get("status") or "pending"),
            "note": str(r.get("note") or ""),
        }
        for r in rows
    ]


def set_plan(task_id: str, steps: list[str]) -> list[dict[str, Any]]:
    """Replace the task's plan. A replan is a replacement, not an edit —
    the thread records that it happened; these rows are only ever the
    current intention."""
    titles = [str(s).strip()[:MAX_TITLE] for s in (steps or []) if str(s).strip()]
    if not titles:
        raise ValueError("a plan needs at least one step")
    if len(titles) > MAX_STEPS:
        raise ValueError(
            f"{len(titles)} steps is over the {MAX_STEPS}-step cap — "
            "over-decomposition is latency wearing a to-do list"
        )
    _sql(f"DELETE FROM {PLANS_TABLE} WHERE task_id = :t", {"t": task_id})
    for i, title in enumerate(titles, start=1):
        _sql(
            f"INSERT INTO {PLANS_TABLE} "
            "(task_id, step_no, title, status, note, updated_at) "
            "VALUES (:t, :n, :ti, 'pending', '', current_timestamp())",
            {"t": task_id, "n": i, "ti": title},
        )
    return get_plan(task_id)


def update_step(
    task_id: str, step_no: int, status: str, note: str = ""
) -> dict[str, Any] | None:
    status = (status or "").strip().lower()
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}")
    _sql(
        f"UPDATE {PLANS_TABLE} SET status = :s, note = :no, "
        "updated_at = current_timestamp() "
        "WHERE task_id = :t AND step_no = :n",
        {"s": status, "no": (note or "")[:MAX_NOTE], "t": task_id,
         "n": int(step_no)},
    )
    for step in get_plan(task_id):
        if step["step_no"] == int(step_no):
            return step
    return None


def unfinished(task_id: str) -> list[dict[str, Any]]:
    """Steps that are not done or deliberately skipped."""
    return [
        s for s in get_plan(task_id) if s["status"] not in ("done", "skipped")
    ]


def prompt_block(task_id: str) -> str:
    """The current plan rendered for a run prompt.

    This is what gives plan continuity across process death: a restarted
    run sees where the last attempt's roadmap stood, not a blank page.
    """
    plan = get_plan(task_id)
    if not plan:
        return ""
    marks = {"done": "[x]", "skipped": "[-]", "doing": "[~]",
             "blocked": "[!]", "pending": "[ ]"}
    lines = [
        f"  {marks.get(s['status'], '[ ]')} {s['step_no']}. {s['title']}"
        + (f" — {s['note']}" if s["note"] else "")
        for s in plan
    ]
    return "EXISTING PLAN (continue it, update statuses as you go):\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# The agent-facing tools — session-bound, never task-id-from-the-model
# ---------------------------------------------------------------------------


def _bound_task() -> str:
    from agent.swarm import current_vm

    vm = current_vm()
    if vm is None:
        return ""
    return task_for_session(getattr(vm, "session_id", ""))


def _emit_plan(vm, task_id: str) -> None:
    try:
        from alex.events import plan_event

        vm.pump.emit(plan_event(task_id=task_id, steps=get_plan(task_id)))
    except Exception:  # noqa: BLE001
        log.debug("plan event emit failed", exc_info=True)


def plan_task(
    steps: list, *, user_email: str = "", instance_scope: str = "default"
) -> dict[str, Any]:
    """Declare (or replace) the step-by-step plan for the task you are
    working. Call this FIRST with 2–7 concrete steps, before acting; call
    it again if the situation genuinely changes. Steps are visible live
    to the whole crew."""
    from agent.swarm import current_vm

    task_id = _bound_task()
    if not task_id:
        return {
            "ok": False,
            "error": "plans belong to board-task sessions; this session "
                     "has no task",
        }
    try:
        plan = set_plan(task_id, list(steps or []))
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    _emit_plan(current_vm(), task_id)
    return {"ok": True, "task_id": task_id, "plan": plan}


def check_plan_step(
    step_no: int,
    status: str,
    note: str = "",
    *,
    user_email: str = "",
    instance_scope: str = "default",
) -> dict[str, Any]:
    """Update one step of your current task's plan as you work:
    'doing' when you start it, 'done' when finished, 'skipped' (with a
    note saying why) when it turned out unnecessary, 'blocked' when a
    human decision is needed."""
    from agent.swarm import current_vm

    task_id = _bound_task()
    if not task_id:
        return {"ok": False, "error": "this session has no task plan"}
    try:
        step = update_step(task_id, step_no, status, note)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    if step is None:
        return {"ok": False, "error": f"no step {step_no} in the plan"}
    _emit_plan(current_vm(), task_id)
    return {"ok": True, "step": step}
