"""Tests for agent/execution_governor.py.

Each test names the runaway it prevents. These are not edge cases — a
model given a goal and a VM reaches every one of these states eventually,
and none of them is caught by authorization, because each individual step
may be perfectly permitted.
"""

from __future__ import annotations

import time

import pytest

from agent.execution_governor import (
    CONTINUE,
    HALT,
    WARN,
    Budget,
    ExecutionGovernor,
)


def gov(goal: str = "", **budget_kw) -> ExecutionGovernor:
    b = Budget(**budget_kw) if budget_kw else Budget()
    return ExecutionGovernor(goal=goal, budget=b, run_id="r1")


def do(g: ExecutionGovernor, name: str, args=None, cap="files.read",
       produced=True, ok=True):
    """Ask, then act — the sequence a caller must follow."""
    verdict = g.before_tool(name, args or {}, cap)
    if verdict.action != HALT:
        g.record(kind="tool", name=name, args=args or {}, capability=cap,
                 ok=ok, produced=produced)
    return verdict


# ---- budgets -------------------------------------------------------------


def test_a_fresh_run_may_proceed():
    assert gov().before_tool("view", {"path": "a"}, "files.read").action == CONTINUE


def test_the_tool_call_budget_stops_a_run():
    g = gov(max_tool_calls=3, max_repeats=99, max_stagnant_steps=99)
    for i in range(3):
        assert do(g, "view", {"path": i}).action != HALT
    v = do(g, "view", {"path": 99})
    assert v.action == HALT
    assert v.pathology == "budget_tools"


def test_the_turn_budget_stops_a_run():
    g = gov(max_llm_turns=2)
    for _ in range(2):
        g.record(kind="llm")
    v = g.before_tool("view", {}, "files.read")
    assert v.action == HALT and v.pathology == "budget_turns"


def test_the_time_budget_stops_a_run():
    g = gov(max_seconds=0.001)
    time.sleep(0.01)
    v = g.before_tool("view", {}, "files.read")
    assert v.action == HALT and v.pathology == "budget_time"


def test_autonomy_moves_the_ceilings():
    """The autonomy preference has to be a real control, not a label."""
    cautious = Budget.for_autonomy("suggest")
    default = Budget.for_autonomy("co-pilot")
    bold = Budget.for_autonomy("autopilot")
    assert cautious.max_tool_calls < default.max_tool_calls < bold.max_tool_calls
    assert cautious.max_seconds < default.max_seconds < bold.max_seconds


# ---- repetition ----------------------------------------------------------


def test_repeating_the_same_call_is_halted():
    """The commonest runaway: a failing call retried forever."""
    g = gov(max_repeats=3, max_stagnant_steps=99)
    args = {"query": "atlas"}
    # The gate sees the history *before* this call, so the warning lands
    # on the attempt after two identical calls and the halt on the next.
    assert do(g, "search_knowledge", args, produced=False).action == CONTINUE
    assert do(g, "search_knowledge", args, produced=False).action == CONTINUE
    assert do(g, "search_knowledge", args, produced=False).action == WARN
    v = do(g, "search_knowledge", args, produced=False)
    assert v.action == HALT and v.pathology == "repetition"


def test_the_same_tool_with_different_arguments_is_fine():
    g = gov(max_repeats=2, max_stagnant_steps=99)
    for i in range(5):
        assert do(g, "view", {"path": f"file{i}.py"}).action != HALT


# ---- oscillation ---------------------------------------------------------


def test_alternating_between_two_tools_is_halted():
    """A, B, A, B is a model unable to choose, not a model working."""
    g = gov(max_repeats=99, max_stagnant_steps=99)
    for name in ("a", "b", "a", "b"):
        do(g, name, {"n": name})
    v = g.before_tool("a", {"n": "a"}, "files.read")
    assert v.action == HALT and v.pathology == "oscillation"


def test_a_varied_sequence_is_not_oscillation():
    g = gov(max_repeats=99, max_stagnant_steps=99)
    for name in ("a", "b", "c", "d"):
        assert do(g, name, {"n": name}).action != HALT


# ---- stagnation ----------------------------------------------------------


def test_calls_that_produce_nothing_are_halted():
    g = gov(max_repeats=99, max_stagnant_steps=4)
    for i in range(4):
        do(g, f"tool{i}", {"i": i}, produced=False)
    v = g.before_tool("tool9", {}, "files.read")
    assert v.action == HALT and v.pathology == "stagnation"


def test_progress_resets_stagnation():
    g = gov(max_repeats=99, max_stagnant_steps=3)
    do(g, "a", {"i": 1}, produced=False)
    do(g, "b", {"i": 2}, produced=False)
    do(g, "c", {"i": 3}, produced=True)
    assert g.before_tool("d", {}, "files.read").action != HALT


# ---- refusals: the most serious signal ----------------------------------


def test_persisting_after_repeated_refusals_is_halted():
    g = gov(max_denials=3)
    for _ in range(3):
        g.record_denial("send_email", "email.send", "not permitted")
    v = g.before_tool("read_inbox", {}, "email.read")
    assert v.action == HALT and v.pathology == "denial_persistence"


def test_escalating_to_a_more_powerful_capability_after_a_refusal_is_halted():
    """The failure the whole module exists for: told no, so it reaches
    for a bigger tool."""
    g = gov()
    g.record_denial("draft_email_reply", "email.draft", "not permitted")
    v = g.before_tool("send_email", {}, "email.send")
    assert v.action == HALT
    assert v.pathology == "denial_escalation"
    assert "email.send" in v.reason


def test_retrying_at_the_same_power_after_a_refusal_is_allowed():
    """A refusal is not a reason to stop reading — only to stop climbing."""
    g = gov()
    g.record_denial("send_email", "email.send", "not permitted")
    assert g.before_tool("read_inbox", {}, "email.read").action == CONTINUE


def test_a_refusal_does_not_block_an_unrelated_read():
    g = gov()
    g.record_denial("system_install", "system.install", "no")
    assert g.before_tool("search_knowledge", {}, "knowledge.read").action == CONTINUE


# ---- scope drift ---------------------------------------------------------


def test_writing_outside_the_goals_area_is_halted():
    g = gov("summarise my inbox and list open tasks")
    # email/tasks are in scope; installing packages is not.
    first = g.before_tool("pip_install", {"packages": ["x"]}, "compute.install")
    assert first.action == WARN
    g.record(kind="tool", name="pip_install", args={}, capability="compute.install")
    g.record(kind="tool", name="pip_install", args={}, capability="compute.install")
    v = g.before_tool("pip_install", {"packages": ["y"]}, "compute.install")
    assert v.action == HALT and v.pathology == "scope_drift"


def test_reading_outside_the_goal_is_allowed():
    """Research legitimately wanders; only acting outside scope is stopped."""
    g = gov("summarise my inbox")
    assert g.before_tool("view", {"path": "x"}, "files.read").action == CONTINUE


def test_a_goal_with_no_recognised_subject_disables_drift_detection():
    """Better to not guess than to block legitimate work."""
    g = gov("xyzzy plugh")
    assert g.before_tool("pip_install", {}, "compute.install").action == CONTINUE


def test_in_scope_capabilities_are_never_drift():
    g = gov("debug the code and fix the failing file")
    # Distinct arguments each time, so this exercises drift and not the
    # repetition check.
    for i in range(4):
        v = g.before_tool("create_file", {"path": f"f{i}.py"}, "files.write")
        assert v.action != HALT
        g.record(kind="tool", name="create_file", args={"path": f"f{i}.py"},
                 capability="files.write", produced=True)


def test_adapted_capabilities_use_their_own_family_for_drift():
    """jarvis.<category>.<read|act> is category drift, not "jarvis" drift.

    Otherwise a goal that names files never matches the adapted file
    tools and the third adapted act halts the run.
    """
    g = gov("fix the failing file")
    for i in range(4):
        v = g.before_tool("native_write", {"path": f"f{i}.py"}, "jarvis.files.act")
        assert v.action != HALT, v.reason
        g.record(kind="tool", name="native_write", args={"path": f"f{i}.py"},
                 capability="jarvis.files.act", produced=True)


def test_adapted_act_outside_the_goal_still_drifts():
    from agent.execution_governor import CAPABILITY_POWER

    # Boot registration assigns adapted capabilities their power.
    CAPABILITY_POWER.setdefault("jarvis.music.act", 2)
    g = gov("summarise my inbox")
    first = g.before_tool("generate_composition", {}, "jarvis.music.act")
    assert first.action == WARN
    g.record(kind="tool", name="generate_composition", args={},
             capability="jarvis.music.act", produced=True)
    g.record(kind="tool", name="generate_composition", args={},
             capability="jarvis.music.act", produced=True)
    v = g.before_tool("generate_variation", {}, "jarvis.music.act")
    assert v.action == HALT and v.pathology == "scope_drift"


# ---- behaviour of the gate itself ---------------------------------------


def test_once_halted_a_run_stays_halted():
    g = gov(max_tool_calls=1, max_repeats=99, max_stagnant_steps=99)
    do(g, "a", {"i": 1})
    first = g.before_tool("b", {}, "files.read")
    assert first.action == HALT
    again = g.before_tool("totally_different", {}, "knowledge.read")
    assert again.action == HALT, "a halted run must not be resumable by a new call"


def test_every_decision_is_emitted_for_the_execution_log():
    seen = []
    g = ExecutionGovernor(
        goal="", budget=Budget(max_tool_calls=1, max_repeats=99,
                               max_stagnant_steps=99),
        run_id="r1", on_event=seen.append,
    )
    do(g, "a", {"i": 1})
    g.before_tool("b", {}, "files.read")
    assert seen and seen[-1]["type"] == "governor"
    assert seen[-1]["action"] == "halt"
    assert "state" in seen[-1]


def test_a_failing_event_sink_does_not_break_the_gate():
    def boom(_e):
        raise RuntimeError("sink down")

    g = ExecutionGovernor(
        goal="", budget=Budget(max_tool_calls=1, max_repeats=99,
                               max_stagnant_steps=99),
        run_id="r1", on_event=boom,
    )
    do(g, "a", {"i": 1})
    assert g.before_tool("b", {}, "files.read").action == HALT


def test_state_is_reportable():
    g = gov("summarise my inbox")
    do(g, "read_inbox", {}, cap="email.read")
    state = g.state()
    assert state["tool_calls"] == 1
    assert state["halted"] is None
    assert state["budget"]["max_tool_calls"] > 0


@pytest.mark.parametrize(
    "pathology,setup",
    [
        ("repetition", lambda g: [do(g, "x", {"a": 1}, produced=False)
                                  for _ in range(3)]),
        ("denial_escalation", lambda g: g.record_denial("d", "email.draft", "no")),
    ],
)
def test_halting_verdicts_explain_themselves(pathology, setup):
    """A halt the user cannot understand is indistinguishable from a bug."""
    g = gov(max_repeats=3)
    setup(g)
    verdict = g.before_tool(
        "x" if pathology == "repetition" else "send_email",
        {"a": 1},
        "files.read" if pathology == "repetition" else "email.send",
    )
    assert verdict.action == HALT
    assert verdict.pathology == pathology
    assert len(verdict.reason) > 20 and verdict.reason.endswith((".", "!"))
