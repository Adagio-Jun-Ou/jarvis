"""The long-horizon pass from the OSS-harness research: the reclaiming
dispatcher (Hermes kanban), observation-aware stuck detection
(OpenHands / OpenCode doom_loop), and non-degrading compaction lineage
(Hermes incremental summary update).

Properties pinned:

* A task whose latest run was INTERRUPTED (a crash) is auto-restarted;
  ``failed`` and ``stopped`` — deliberate outcomes — never are. The
  attempt ceiling is a circuit breaker with a once-only thread note.
* Polling (same call, different result) is not a loop; the same call
  with the same result still halts, and even polling has a ceiling.
* A re-compaction UPDATES the previous summary instead of concatenating
  digests of digests forever; the no-model fallback chain is bounded.
"""

from __future__ import annotations

import pytest

from agent import taskboard as tb
from agent.execution_governor import Budget, ExecutionGovernor


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

    def _sql(stmt, params=None):
        res = wh.execute(stmt, params or {})
        as_dicts = getattr(res, "as_dicts", None)
        return as_dicts() if callable(as_dicts) else list(res or [])

    return _sql


def _task(sql, task_id, status="todo"):
    sql("INSERT INTO media.alex.tasks (id, owner_email, title, status) "
        "VALUES (:i, 'dana@alex.dev', 'A task', :s)",
        {"i": task_id, "s": status})


def _run_row(sql, run_id, task_id, status, started_at):
    sql(
        "INSERT INTO media.control.task_runs "
        "(run_id, task_id, session_id, started_by, status, started_at) "
        "VALUES (:r, :t, :s, 'dana@alex.dev', :st, :at)",
        {"r": run_id, "t": task_id, "s": f"task-{task_id}", "st": status,
         "at": started_at},
    )


class QuickVM:
    def __init__(self):
        self.turns = 0

    def subscribe_sse(self, fn):
        return lambda: None

    def run_turn(self, *, text, **_kw):
        self.turns += 1
        return "restarted and finished", []

    def close(self):
        pass


# ---- the dispatcher --------------------------------------------------------


def test_interrupted_work_is_restarted_deliberate_ends_are_not(sql):
    _task(sql, "t-crash")
    _task(sql, "t-stopped")
    _task(sql, "t-failed")
    _run_row(sql, "r-1", "t-crash", "interrupted", "2026-09-01 10:00:00")
    _run_row(sql, "r-2", "t-stopped", "stopped", "2026-09-01 10:00:00")
    _run_row(sql, "r-3", "t-failed", "failed", "2026-09-01 10:00:00")

    started = tb.dispatch_once(sql, vm_factory=lambda **kw: QuickVM())
    assert started == ["t-crash"], (
        "only the crash restarts; stopped/failed are people's decisions"
    )
    bodies = [m["body"] for m in tb.messages(sql, "t-crash")]
    assert any("Auto-restarting" in b for b in bodies)


def test_the_attempt_ceiling_is_a_circuit_breaker_with_one_note(sql):
    _task(sql, "t-dying")
    for i in range(3):  # ALEX_DISPATCH_MAX_ATTEMPTS default 3 — at cap
        _run_row(sql, f"r-{i}", "t-dying", "interrupted",
                 f"2026-09-01 0{i}:00:00")

    assert tb.dispatch_once(sql, vm_factory=lambda **kw: QuickVM()) == []
    assert tb.dispatch_once(sql, vm_factory=lambda **kw: QuickVM()) == []
    notes = [
        m["body"] for m in tb.messages(sql, "t-dying")
        if "Auto-restart gave up" in m["body"]
    ]
    assert len(notes) == 1, "the breaker note lands once, not every sweep"


def test_the_dispatcher_respects_the_concurrency_cap(sql, monkeypatch):
    monkeypatch.setenv("ALEX_MAX_CONCURRENT_RUNS", "0")
    _task(sql, "t-crash2")
    _run_row(sql, "r-9", "t-crash2", "interrupted", "2026-09-01 10:00:00")
    assert tb.dispatch_once(sql, vm_factory=lambda **kw: QuickVM()) == []


# ---- observation-aware stuck detection -------------------------------------


def test_polling_with_changing_results_is_not_a_loop(env):
    gov = ExecutionGovernor(goal="poll", budget=Budget(max_repeats=3))
    for i in range(4):
        v = gov.before_tool("n8n_execution_status", {"id": "ex-1"},
                            "workflow.read")
        assert v.action == "continue", f"poll {i} wrongly stopped: {v.reason}"
        gov.record(kind="tool", name="n8n_execution_status",
                   args={"id": "ex-1"}, capability="workflow.read",
                   result={"status": f"running-{i}"})
    # But even patient polling has a ceiling (2x max_repeats).
    for i in range(4, 6):
        gov.record(kind="tool", name="n8n_execution_status",
                   args={"id": "ex-1"}, capability="workflow.read",
                   result={"status": f"running-{i}"})
    v = gov.before_tool("n8n_execution_status", {"id": "ex-1"},
                        "workflow.read")
    assert v.action == "halt" and v.pathology == "repetition"


def test_the_same_call_with_the_same_answer_still_halts(env):
    gov = ExecutionGovernor(goal="loop", budget=Budget(max_repeats=3))
    for _ in range(3):
        gov.record(kind="tool", name="view", args={"p": "a.txt"},
                   capability="files.read", result="the same 12 lines")
    v = gov.before_tool("view", {"p": "a.txt"}, "files.read")
    assert v.action == "halt" and v.pathology == "repetition"
    assert "same result" in v.reason


def test_recording_no_observation_keeps_the_strict_reading(env):
    """Legacy callers that never pass a result must not get the longer
    polling leash — no observation means no evidence of progress."""
    gov = ExecutionGovernor(goal="loop", budget=Budget(max_repeats=3))
    for _ in range(3):
        gov.record(kind="tool", name="view", args={"p": "a.txt"},
                   capability="files.read")
    assert gov.before_tool("view", {"p": "a.txt"}, "files.read").action == "halt"


# ---- non-degrading compaction lineage --------------------------------------


def _long_history():
    hist = [{"role": "user", "content": "the task"}]
    hist += [{"role": "assistant", "content": f"step {i}"} for i in range(30)]
    return hist


def test_recompaction_updates_the_prior_summary(env):
    from agent.harness.compaction import compact_history

    prompts = []

    def llm(messages):
        prompts.append(messages[0]["content"])
        return "- merged working summary"

    once, ran = compact_history(_long_history(), llm_invoke=llm,
                                trigger=10, keep_tail=4)
    assert ran and "WORKING SUMMARY" not in prompts[0]

    grown = list(once) + [
        {"role": "assistant", "content": f"more {i}"} for i in range(30)
    ]
    twice, ran = compact_history(grown, llm_invoke=llm,
                                 trigger=10, keep_tail=4)
    assert ran
    assert "WORKING SUMMARY" in prompts[1], (
        "the second compaction must hand the first summary back for update"
    )
    head = twice[0]["content"]
    assert head.count("- merged working summary") == 1, (
        "updated, not concatenated — no digests of digests"
    )
    assert "(compaction #2)" in head


def test_the_no_model_fallback_chain_is_bounded(env):
    from agent.harness import compaction
    from agent.harness.compaction import compact_history

    hist = _long_history()
    hist[0] = {"role": "system",
               "content": "EARLIER CONVERSATION DIGEST (compaction #1) — x\n"
                          + ("old line\n" * 3000)}
    hist.insert(1, {"role": "user", "content": "the task"})
    out, ran = compact_history(hist, trigger=10, keep_tail=4)
    assert ran
    assert len(out[0]["content"]) < compaction._FALLBACK_CHAIN_CHARS + 2000
