"""The plan — a task run's structured spine.

The properties under test are the ones that make plan-then-execute safe
in an enterprise system rather than a demo:

* the plan tools bind to the invoking VM's session task and CANNOT be
  aimed at another task — a model that could name a task id could
  rewrite another crew's plan;
* the plan is durable and frames the next run (continuity across
  process death);
* a run that declares victory with open steps is called on it in the
  summary the reviewer reads — the silent-stale-plan failure mode,
  named by the pattern literature, made loud.
"""

from __future__ import annotations

import pytest

from agent import task_plan
from agent import taskboard as tb
from agent.omnigent import MockExecutor
from agent.vm import make_vm


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("ALEX_ENV", "dev")
    from alex import config, testing

    config._settings = None
    testing._fakes = None
    tb.reset_runs()
    yield
    config._settings = None
    testing._fakes = None
    tb.reset_runs()


@pytest.fixture
def sql(env):
    from alex.testing import get_fakes
    from app._dev_seed import (
        TASK_CREW_COLS,
        TASK_MESSAGES_COLS,
        TASK_PLANS_COLS,
        TASK_RUNS_COLS,
    )

    wh = get_fakes().warehouse
    wh.register(
        "media.alex.tasks",
        ["id", "owner_email", "assignee_email", "title", "description",
         "status", "priority", "due_at", "source", "source_ref",
         "created_at", "updated_at", "closed_at"],
        pk=["id"],
    )
    wh.register("media.control.task_crew", TASK_CREW_COLS)
    wh.register("media.control.task_messages", TASK_MESSAGES_COLS,
                pk=["message_id"])
    wh.register("media.control.task_runs", TASK_RUNS_COLS, pk=["run_id"])
    wh.register("media.control.task_plans", TASK_PLANS_COLS,
                pk=["task_id", "step_no"])

    def _sql(stmt, params=None):
        res = wh.execute(stmt, params or {})
        as_dicts = getattr(res, "as_dicts", None)
        return as_dicts() if callable(as_dicts) else list(res or [])

    return _sql


# ---- the store ------------------------------------------------------------


def test_a_plan_round_trips_in_order(sql):
    task_plan.set_plan("t-1", ["find the rate card", "compare Q2 vs Q3",
                               "draft the summary"])
    plan = task_plan.get_plan("t-1")

    assert [s["step_no"] for s in plan] == [1, 2, 3]
    assert plan[1]["title"] == "compare Q2 vs Q3"
    assert all(s["status"] == "pending" for s in plan)


def test_a_replan_replaces_rather_than_appends(sql):
    task_plan.set_plan("t-1", ["a", "b", "c", "d"])
    task_plan.set_plan("t-1", ["only this now"])

    plan = task_plan.get_plan("t-1")
    assert len(plan) == 1 and plan[0]["title"] == "only this now"


def test_over_decomposition_is_refused(sql):
    with pytest.raises(ValueError, match="cap"):
        task_plan.set_plan("t-1", [f"step {i}" for i in range(20)])


def test_step_updates_persist_with_notes(sql):
    task_plan.set_plan("t-1", ["a", "b"])
    task_plan.update_step("t-1", 2, "blocked", "needs legal sign-off")

    step = task_plan.get_plan("t-1")[1]
    assert step["status"] == "blocked"
    assert step["note"] == "needs legal sign-off"


def test_an_invalid_status_is_refused(sql):
    task_plan.set_plan("t-1", ["a"])
    with pytest.raises(ValueError):
        task_plan.update_step("t-1", 1, "finished-ish")


# ---- session binding: the security property -------------------------------


def test_the_tool_binds_to_the_sessions_task_only(sql):
    from agent import swarm

    vm = make_vm(session_id="task-t-mine", user_email="demo@alex.dev",
                 executor=MockExecutor())
    token = swarm.set_current_vm(vm)
    try:
        out = task_plan.plan_task(["step one", "step two"])
    finally:
        swarm.reset_current_vm(token)

    assert out["ok"] and out["task_id"] == "t-mine"
    assert len(task_plan.get_plan("t-mine")) == 2


def test_a_chat_session_cannot_write_any_plan(sql):
    """No task binding, no plan — the tool refuses rather than guessing."""
    from agent import swarm

    vm = make_vm(session_id="ordinary-chat", user_email="demo@alex.dev",
                 executor=MockExecutor())
    token = swarm.set_current_vm(vm)
    try:
        out = task_plan.plan_task(["anything"])
    finally:
        swarm.reset_current_vm(token)

    assert out["ok"] is False and "task" in out["error"]


def test_there_is_no_argument_that_names_another_task(sql):
    """The spoof surface must not exist: the tool signature takes steps
    and statuses, never a task id."""
    import inspect

    for fn in (task_plan.plan_task, task_plan.check_plan_step):
        params = set(inspect.signature(fn).parameters)
        assert "task_id" not in params, (
            f"{fn.__name__} accepts a task_id — a model could rewrite "
            "another crew's plan"
        )


# ---- continuity and the staleness gate ------------------------------------


def _task(sql, task_id, title="Reconcile the ledger"):
    sql("INSERT INTO media.alex.tasks (id, owner_email, title, status) "
        "VALUES (:i, :o, :t, 'in_progress')",
        {"i": task_id, "o": "dana@alex.dev", "t": title})


class P:
    email = "dana@alex.dev"
    is_admin = False


def test_an_existing_plan_frames_the_next_run(sql):
    """Plan continuity across process death: the restarted run sees where
    the roadmap stood, not a blank page."""
    _task(sql, "t-cont")
    task_plan.set_plan("t-cont", ["gather statements", "reconcile", "report"])
    task_plan.update_step("t-cont", 1, "done", "statements in /work")

    seen = {}

    class EchoVM:
        def subscribe_sse(self, fn):
            return lambda: None

        def run_turn(self, *, text, **_kw):
            seen["prompt"] = text
            return "continuing", []

        def close(self):
            pass

    tb.start_run(sql, P(), "t-cont", background=False,
                 vm_factory=lambda **kw: EchoVM())

    prompt = seen["prompt"]
    assert "EXISTING PLAN" in prompt
    assert "[x] 1. gather statements" in prompt
    assert "[ ] 2. reconcile" in prompt
    assert "plan_task" in prompt, "the plan-first instruction is missing"


def test_finishing_with_open_steps_is_said_out_loud(sql):
    """The silent-stale-plan failure, made loud: the reviewer deciding
    review->done reads the unmet count in the summary itself."""
    _task(sql, "t-stale")
    task_plan.set_plan("t-stale", ["step A", "step B", "step C"])
    task_plan.update_step("t-stale", 1, "done")

    class QuickVM:
        def subscribe_sse(self, fn):
            return lambda: None

        def run_turn(self, *, text, **_kw):
            return "All wrapped up!", []

        def close(self):
            pass

    run = tb.start_run(sql, P(), "t-stale", background=False,
                       vm_factory=lambda **kw: QuickVM())

    assert "2 planned step(s) not completed" in run.summary
    assert "step B" in run.summary
    bodies = [m["body"] for m in tb.messages(sql, "t-stale")]
    assert any("not completed" in b for b in bodies), (
        "the crew's thread does not show the unmet steps"
    )


def test_a_fully_worked_plan_finishes_clean(sql):
    _task(sql, "t-clean")
    task_plan.set_plan("t-clean", ["a", "b"])
    task_plan.update_step("t-clean", 1, "done")
    task_plan.update_step("t-clean", 2, "skipped", "already handled upstream")

    class QuickVM:
        def subscribe_sse(self, fn):
            return lambda: None

        def run_turn(self, *, text, **_kw):
            return "Done properly.", []

        def close(self):
            pass

    run = tb.start_run(sql, P(), "t-clean", background=False,
                       vm_factory=lambda **kw: QuickVM())

    assert "not completed" not in run.summary


# ---- visibility -----------------------------------------------------------


def test_plan_changes_stream_to_the_watchers(sql):
    from agent import swarm

    vm = make_vm(session_id="task-t-live", user_email="demo@alex.dev",
                 executor=MockExecutor())
    seen = []
    vm.subscribe_sse(seen.append)
    token = swarm.set_current_vm(vm)
    try:
        task_plan.plan_task(["one", "two"])
        task_plan.check_plan_step(1, "doing")
    finally:
        swarm.reset_current_vm(token)

    plans = [e for e in seen if getattr(e, "type", "") == "plan"]
    assert len(plans) >= 2, "plan changes did not reach the stream"
    last = plans[-1].body["steps"]
    assert last[0]["status"] == "doing"
