"""Elicitation — the agent asking a human and waiting for the answer.

The properties that matter:

* **The wait is real and the answer reaches the waiter.** ask_user
  parks; a crew member's answer (from another thread, like the route)
  wakes it with the text.
* **Silence is guidance, not approval.** A timeout returns
  proceed-on-your-own-stated-assumption; the question row stays open.
* **The write asymmetry.** The agent can ask; only a person resolves.
  The tool takes no task_id (session-bound); the grant layer gives the
  agent SP no UPDATE on the questions table.
* **Waiting is free.** The governor is credited the parked time — a
  deliberating human must not eat the run's wall clock.
* **Stop cancels.** A stopped run wakes any parked question as
  cancelled instead of leaving it to time out.
* **Continuation.** A turn ended by the BUDGET with plan steps still
  open re-enters (bounded, progress-gated); a turn the model ended by
  itself does not.
"""

from __future__ import annotations

import threading
import time

import pytest

from agent import task_plan
from agent import task_questions as tq
from agent import taskboard as tb
from agent.omnigent import MockExecutor
from agent.vm import make_vm


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("ALEX_ENV", "dev")
    monkeypatch.setenv("ALEX_ASK_WAIT_S", "2.0")
    monkeypatch.setenv("ALEX_ASK_POLL_S", "0.05")
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
        TASK_ACCEPTANCE_COLS,
        TASK_CREW_COLS,
        TASK_DEPS_COLS,
        TASK_MESSAGES_COLS,
        TASK_PLANS_COLS,
        TASK_QUESTIONS_COLS,
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
    wh.register("media.control.task_acceptance", TASK_ACCEPTANCE_COLS,
                pk=["task_id", "crit_no"])
    wh.register("media.control.task_deps", TASK_DEPS_COLS,
                pk=["task_id", "blocked_by"])
    wh.register("media.control.task_questions", TASK_QUESTIONS_COLS,
                pk=["question_id"])

    def _sql(stmt, params=None):
        res = wh.execute(stmt, params or {})
        as_dicts = getattr(res, "as_dicts", None)
        return as_dicts() if callable(as_dicts) else list(res or [])

    return _sql


class P:
    email = "dana@alex.dev"
    is_admin = False


def _task(sql, task_id, title="A task", status="todo", owner="dana@alex.dev"):
    sql("INSERT INTO media.alex.tasks (id, owner_email, title, status) "
        "VALUES (:i, :o, :t, :s)",
        {"i": task_id, "o": owner, "t": title, "s": status})


def _bound(session_id):
    from agent import swarm

    vm = make_vm(session_id=session_id, user_email="dana@alex.dev",
                 executor=MockExecutor())
    return swarm, vm


# ---- the tool --------------------------------------------------------------


def test_ask_outside_a_task_session_points_back_to_chat(sql):
    swarm, vm = _bound("chat-123")
    token = swarm.set_current_vm(vm)
    try:
        out = tq.ask_user("Which quarter?")
    finally:
        swarm.reset_current_vm(token)
    assert out["ok"] is False
    assert "chat" in out["error"]


def test_ask_and_answer_round_trip_wakes_the_waiter(sql):
    _task(sql, "t-q")
    swarm, vm = _bound("task-t-q")
    token = swarm.set_current_vm(vm)
    got = {}

    def _answer_soon():
        deadline = time.time() + 3
        while time.time() < deadline:
            open_qs = tq.open_for_task(sql, "t-q")
            if open_qs:
                tq.answer(sql, P(), "t-q", open_qs[0]["question_id"],
                          "Q3, the fiscal one")
                return
            time.sleep(0.02)

    t = threading.Thread(target=_answer_soon, daemon=True)
    t.start()
    try:
        got = tq.ask_user("Which quarter should the report cover?",
                          ["Q2", "Q3"])
    finally:
        swarm.reset_current_vm(token)
    t.join(timeout=3)
    assert got["ok"] and got["answered"] is True
    assert got["answer"] == "Q3, the fiscal one"
    assert got["answered_by"] == "dana@alex.dev"
    # Both sides of the exchange are in the thread — the record.
    bodies = [m["body"] for m in tb.messages(sql, "t-q")]
    assert any("Which quarter" in b for b in bodies)
    assert any("Q3, the fiscal one" in b for b in bodies)


def test_timeout_returns_guidance_and_leaves_the_question_open(sql, monkeypatch):
    monkeypatch.setenv("ALEX_ASK_WAIT_S", "0.15")
    _task(sql, "t-slow")
    swarm, vm = _bound("task-t-slow")
    token = swarm.set_current_vm(vm)
    try:
        out = tq.ask_user("Ship it now or wait for legal?")
    finally:
        swarm.reset_current_vm(token)
    assert out["ok"] and out["answered"] is False
    assert "best judgment" in out["guidance"]
    assert "approval" in out["guidance"], (
        "silence must be explicitly framed as not-approval"
    )
    assert len(tq.open_for_task(sql, "t-slow")) == 1


def test_the_wait_is_credited_back_to_the_governor(sql, monkeypatch):
    monkeypatch.setenv("ALEX_ASK_WAIT_S", "0.3")
    _task(sql, "t-clock")
    swarm, vm = _bound("task-t-clock")
    from agent.execution_governor import ExecutionGovernor

    gov = ExecutionGovernor(goal="test")
    vm._governor = gov
    token = swarm.set_current_vm(vm)
    try:
        tq.ask_user("Proceed?")
    finally:
        swarm.reset_current_vm(token)
    assert gov._wait_credit_s >= 0.25, "parked time must be credited"
    assert gov.elapsed < 0.25, "credited time must leave the clock"


def test_validation_question_required_and_option_cap(sql):
    _task(sql, "t-v")
    swarm, vm = _bound("task-t-v")
    token = swarm.set_current_vm(vm)
    try:
        assert tq.ask_user("")["ok"] is False
        out = tq.ask_user("Pick one", ["a", "b", "c", "d", "e"])
    finally:
        swarm.reset_current_vm(token)
    assert out["ok"] is False and "options" in out["error"]


def test_answering_twice_is_refused(sql):
    _task(sql, "t-two")
    sql("INSERT INTO media.control.task_questions "
        "(question_id, task_id, run_id, asked_by, question, options, status, "
        " asked_at) VALUES ('q-1', 't-two', '', 'alex', 'Which?', '[]', "
        "'open', current_timestamp())")
    tq.answer(sql, P(), "t-two", "q-1", "the first")
    with pytest.raises(ValueError, match="already answered"):
        tq.answer(sql, P(), "t-two", "q-1", "the second")


def test_stop_run_cancels_the_parked_question(sql):
    _task(sql, "t-stop", status="in_progress")

    class ParkingVM:
        session_id = "task-t-stop"

        def subscribe_sse(self, fn):
            return lambda: None

        def run_turn(self, *, text, **_kw):
            from agent import swarm

            token = swarm.set_current_vm(self)
            try:
                self.asked = tq.ask_user("May I delete the old rows?")
            finally:
                swarm.reset_current_vm(token)
            return "ended", []

        def close(self):
            pass

    vm = ParkingVM()
    run = tb.start_run(sql, P(), "t-stop", background=True,
                       vm_factory=lambda **kw: vm)
    deadline = time.time() + 3
    while time.time() < deadline and not tq.open_for_task(sql, "t-stop"):
        time.sleep(0.02)
    assert tq.open_for_task(sql, "t-stop"), "the question should be parked"
    assert tb.stop_run(sql, "t-stop") is True
    deadline = time.time() + 3
    while time.time() < deadline and not hasattr(vm, "asked"):
        time.sleep(0.02)
    assert vm.asked.get("cancelled") is True
    assert tq.open_for_task(sql, "t-stop") == []
    assert run.status == "stopped"


# ---- the control surface ---------------------------------------------------


def test_ask_user_takes_no_task_id_and_is_registered(sql):
    import inspect

    from agent.guardrails import TOOL_CAPABILITY
    from agent.tools.registry import TOOL_REGISTRY

    assert "ask_user" in TOOL_REGISTRY
    params = set(inspect.signature(tq.ask_user).parameters)
    assert "task_id" not in params, (
        "a model that could name a task could park questions on another "
        "crew's board"
    )
    assert TOOL_CAPABILITY["ask_user"] == "tasks.update"
    from agent.vm import _PLAN_MODE_TOOL_EXCEPTIONS

    assert "ask_user" in _PLAN_MODE_TOOL_EXCEPTIONS, (
        "clarifying with the crew is exactly what a planning pass is for"
    )


def test_the_agent_grant_cannot_resolve_questions():
    import re
    from pathlib import Path

    grants = (Path(__file__).resolve().parents[2] / "governance"
              / "grants.sql").read_text(encoding="utf-8")
    m = re.search(
        r"GRANT\s+([A-Z,\s]+)\s+ON\s+media\.control\.task_questions\s+"
        r"TO\s+sp-alex-agent", grants)
    assert m, "the agent needs SELECT, INSERT on task_questions"
    perms = {p.strip() for p in m.group(1).split(",")}
    assert perms == {"SELECT", "INSERT"}, (
        f"the agent may ask, never resolve; got {perms}"
    )


def test_open_questions_ride_the_board_and_the_prompt(sql):
    _task(sql, "t-b")
    sql("INSERT INTO media.control.task_questions "
        "(question_id, task_id, run_id, asked_by, question, options, status, "
        " asked_at) VALUES ('q-b', 't-b', '', 'alex', 'Blue or green?', "
        "'[]', 'open', current_timestamp())")
    board = tb.visible_tasks(sql, P())
    mine = next(t for t in board if t["task_id"] == "t-b")
    assert mine["open_questions"] == 1
    prompt = tb.build_task_prompt(
        {"task_id": "t-b", "title": "A task"}, [])
    assert "STILL UNANSWERED" in prompt
    assert "Blue or green?" in prompt
    assert "Do not re-ask" in prompt


def test_governor_scores_delegation_as_an_act():
    from agent.execution_governor import CAPABILITY_POWER

    assert CAPABILITY_POWER.get("agents.delegate", 0) >= 2, (
        "spawning governed VMs is not a read"
    )


def test_the_answer_route_end_to_end(env):
    from fastapi.testclient import TestClient

    from alex.identity import mint_dev_token
    from app.app import app

    client = TestClient(app)
    headers = {
        "X-Forwarded-Access-Token": mint_dev_token(
            "demo@alex.dev", scopes=["sql"]
        ),
        "Content-Type": "application/json",
    }
    created = client.post("/api/board/tasks", headers=headers,
                          json={"title": "Route task"})
    assert created.status_code == 200, created.text
    tid = created.json()["task_id"]

    from alex.testing import get_fakes

    get_fakes().warehouse.execute(
        "INSERT INTO media.control.task_questions "
        "(question_id, task_id, run_id, asked_by, question, options, status, "
        " asked_at) VALUES ('q-r', :t, '', 'alex', 'Proceed?', :opts, "
        "'open', current_timestamp())",
        {"t": tid, "opts": '["yes","no"]'},
    )
    detail = client.get(f"/api/board/tasks/{tid}", headers=headers).json()
    assert any(q["question_id"] == "q-r" for q in detail["questions"])

    r = client.post(f"/api/board/tasks/{tid}/questions/q-r/answer",
                    headers=headers, json={"answer": "yes"})
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "answered"
    assert r.json()["answered_by"] == "demo@alex.dev"

    thread = client.get(f"/api/board/tasks/{tid}/messages",
                        headers=headers).json()
    assert any(m["body"] == "yes" for m in thread["messages"])
    assert thread["questions"][0]["status"] == "answered"
    # A second answer is a 400, not a silent overwrite.
    again = client.post(f"/api/board/tasks/{tid}/questions/q-r/answer",
                        headers=headers, json={"answer": "no"})
    assert again.status_code == 400


# ---- continuation: long horizon by re-entry --------------------------------


class BudgetedVM:
    """Ends every turn 'on budget' until the plan is finished."""

    session_id = "task-t-long"

    def __init__(self, sql):
        self.sql = sql
        self.turns = 0

    def subscribe_sse(self, fn):
        return lambda: None

    def last_governor_state(self):
        return {
            "elapsed_s": 12.0, "tool_calls": 5, "llm_turns": 12,
            "denials": 0,
            "budget": {"max_llm_turns": 12},
            "halted": {"pathology": "budget_turns"},
        }

    def run_turn(self, *, text, **_kw):
        self.turns += 1
        if self.turns == 1:
            task_plan.set_plan("t-long", ["gather", "draft", "send"])
            task_plan.update_step("t-long", 1, "done")
        elif self.turns == 2:
            task_plan.update_step("t-long", 2, "done")
            task_plan.update_step("t-long", 3, "done")
        return f"turn {self.turns} output", []

    def close(self):
        pass


def test_budget_ended_turns_continue_until_the_plan_is_done(sql):
    _task(sql, "t-long")
    vm = BudgetedVM(sql)
    run = tb.start_run(sql, P(), "t-long", background=False,
                       vm_factory=lambda **kw: vm)
    assert vm.turns == 2, "the budget ended turn 1 with open steps — re-enter"
    assert run.status == "done"
    assert "not completed" not in run.summary
    kinds = [e.get("type") for e in run.events]
    assert "run_usage" in kinds, "resource usage must be in the durable record"
    bodies = [m["body"] for m in tb.messages(sql, "t-long")]
    assert any("Continuing" in b for b in bodies)


def test_a_turn_the_model_ended_itself_is_not_re_entered(sql):
    _task(sql, "t-self")

    class SelfEndingVM(BudgetedVM):
        session_id = "task-t-self"

        def last_governor_state(self):
            return {"elapsed_s": 3.0, "tool_calls": 2, "llm_turns": 4,
                    "denials": 0, "budget": {"max_llm_turns": 12},
                    "halted": None}

        def run_turn(self, *, text, **_kw):
            self.turns += 1
            task_plan.set_plan("t-self", ["a", "b"])
            return "stopped by choice", []

    vm = SelfEndingVM(sql)
    run = tb.start_run(sql, P(), "t-self", background=False,
                       vm_factory=lambda **kw: vm)
    assert vm.turns == 1, "the model's own stop is a statement, not a budget"
    assert "not completed" in run.summary, "the staleness gate still reports"


def test_unmet_criteria_get_one_verification_retry(sql):
    """Check-verified retry: a run that ends with criteria it marked
    UNMET gets one focused turn to make them hold before review."""
    from agent import task_acceptance as ta

    _task(sql, "t-verify")
    ta.set_criteria("t-verify", ["totals reconcile"])
    ta.mark("t-verify", 1, "unmet", "off by 402.10")

    class VerifyVM(BudgetedVM):
        session_id = "task-t-verify"

        def last_governor_state(self):
            return {"halted": None, "llm_turns": 3,
                    "budget": {"max_llm_turns": 12}}

        def run_turn(self, *, text, **_kw):
            self.turns += 1
            if self.turns == 2:
                assert "UNMET" in text and "totals reconcile" in text
                assert "402.10" in text, "the agent's own note frames the retry"
                ta.mark("t-verify", 1, "met", "recomputed; diff is zero")
            return f"turn {self.turns}", []

    vm = VerifyVM(sql)
    run = tb.start_run(sql, P(), "t-verify", background=False,
                       vm_factory=lambda **kw: vm)
    assert vm.turns == 2
    assert "not verified met" not in run.summary, "the retry made it hold"
    bodies = [m["body"] for m in tb.messages(sql, "t-verify")]
    assert any("Verification retry" in b for b in bodies)


def test_verification_retry_happens_exactly_once(sql):
    from agent import task_acceptance as ta

    _task(sql, "t-stillno")
    ta.set_criteria("t-stillno", ["the API contract holds"])
    ta.mark("t-stillno", 1, "unmet", "endpoint 404s")

    class StubbornVM(BudgetedVM):
        session_id = "task-t-stillno"

        def last_governor_state(self):
            return {"halted": None, "llm_turns": 3,
                    "budget": {"max_llm_turns": 12}}

        def run_turn(self, *, text, **_kw):
            self.turns += 1
            return f"turn {self.turns}", []

    vm = StubbornVM(sql)
    run = tb.start_run(sql, P(), "t-stillno", background=False,
                       vm_factory=lambda **kw: vm)
    assert vm.turns == 2, "a second identical failure is a fact, not a loop"
    assert "not verified met" in run.summary, (
        "still unmet after the retry — the reviewer hears it"
    )


def test_a_continuation_with_no_progress_stops(sql):
    _task(sql, "t-stuck")

    class StuckVM(BudgetedVM):
        session_id = "task-t-stuck"

        def run_turn(self, *, text, **_kw):
            self.turns += 1
            if self.turns == 1:
                task_plan.set_plan("t-stuck", ["a", "b", "c"])
            return "no movement", []

    vm = StuckVM(sql)
    tb.start_run(sql, P(), "t-stuck", background=False,
                 vm_factory=lambda **kw: vm)
    assert vm.turns == 2, (
        "one re-entry to try; a second with zero closed steps is a runaway"
    )
