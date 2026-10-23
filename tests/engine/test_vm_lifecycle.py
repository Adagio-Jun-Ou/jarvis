"""
Tests for the VM lifecycle: construction, audit logging, close.

The VM is the canonical process model in Alex 2.0.  Every public
method goes through ``_invoke_tool`` which calls the audit log.
These tests verify the lifecycle without going through the
FastAPI route layer.
"""
from __future__ import annotations

import pytest

from agent.omnigent import MockExecutor, ToolCallRequest
from agent.tools import TOOL_REGISTRY
from agent.vm import VM, make_vm


@pytest.fixture
def vm() -> VM:
    return make_vm(
        session_id="s1",
        user_email="alice@x.com",
        instance_scope="default",
        executor=MockExecutor(),
    )


def test_vm_construction_sets_session_fields(vm: VM) -> None:
    assert vm.session_id == "s1"
    assert vm.user_email == "alice@x.com"
    assert vm.instance_scope == "default"


def test_vm_pump_is_named_after_session(vm: VM) -> None:
    assert vm.pump.name == "vm:s1"


def test_vm_agent_def_has_all_registered_tools(vm: VM) -> None:
    # Deferred (library) tools are registered and governed but only
    # enter a turn by relevance/pinning, so they are not in the
    # default working set.
    for name, spec in TOOL_REGISTRY.items():
        if getattr(spec, "deferred", False):
            continue
        assert name in vm.agent_def.tools


def test_vm_agent_def_has_constitution_policy(vm: VM) -> None:
    assert "constitution" in vm.agent_def.policies


def test_vm_invoke_tool_runs_and_audits(vm: VM) -> None:
    seen: list = []
    vm.subscribe_sse(seen.append)
    s = vm.open_browser_session()
    t = vm.open_browser_tab(s.session_id, url="alex://welcome")
    request = ToolCallRequest(
        name="browser_navigate",
        args={"session_id": s.session_id, "tab_id": t.tab_id, "url": "alex://welcome"},
    )
    result = vm._invoke_tool(request)
    # The result is a successful navigate dict.
    assert result["ok"] is True
    # The event pump saw a tool event + an artifact.
    types = [e.type for e in seen]
    assert "tool" in types
    assert "artifact" in types


def test_vm_invoke_tool_unknown_tool_returns_error(vm: VM) -> None:
    seen: list = []
    vm.subscribe_sse(seen.append)
    request = ToolCallRequest(name="nuke_it", args={})
    result = vm._invoke_tool(request)
    assert result["ok"] is False
    assert any(e.type == "error" for e in seen)


def test_vm_invoke_tool_audit_log_writes_a_row(vm: VM) -> None:
    """A successful tool call writes a row to media.alex.agent_actions."""
    s = vm.open_browser_session()
    t = vm.open_browser_tab(s.session_id, url="alex://welcome")
    request = ToolCallRequest(
        name="browser_navigate",
        args={"session_id": s.session_id, "tab_id": t.tab_id, "url": "alex://welcome"},
    )
    vm._invoke_tool(request)
    rows = vm._session.audit_log._wh.execute(
        "SELECT tool, ok FROM media.alex.agent_actions"
    ).as_dicts()
    assert any(r["tool"] == "browser_navigate" and r["ok"] == 1 for r in rows)


def test_vm_invoke_tool_audit_log_deny(vm: VM) -> None:
    """A permission-denied tool call writes a row with ok=0."""
    # The dev browser OS lets any agent navigate, but if we hand it
    # a user-controlled tab, it denies.  Set agent_piloting=False.
    s = vm.browser_os.create_session(user_email="alice@x.com", instance_scope="default")
    t = vm.browser_os.create_tab(s.session_id, url="alex://welcome")
    vm.browser_os.set_user_control(s.session_id, t.tab_id, mode="co-drive", user_email="alice@x.com")
    request = ToolCallRequest(
        name="browser_navigate",
        args={"session_id": s.session_id, "tab_id": t.tab_id, "url": "alex://docs"},
    )
    vm._invoke_tool(request)
    rows = vm._session.audit_log._wh.execute(
        "SELECT tool, ok FROM media.alex.agent_actions"
    ).as_dicts()
    assert any(r["tool"] == "browser_navigate" and r["ok"] == 0 for r in rows)


def test_vm_close_is_idempotent(vm: VM) -> None:
    vm.close()
    vm.close()
    assert vm._closed is True


def test_vm_run_turn_after_close_raises(vm: VM) -> None:
    vm.close()
    with pytest.raises(RuntimeError):
        vm.run_turn(text="hi")


def test_vm_context_manager_closes(vm: VM) -> None:
    with vm as v:
        assert v is vm
    assert vm._closed is True


def test_vm_open_browser_session_emits_artifact(vm: VM) -> None:
    seen: list = []
    vm.subscribe_sse(seen.append)
    s = vm.open_browser_session()
    assert s.user_email == "alice@x.com"
    assert any(e.type == "artifact" for e in seen)


def test_vm_list_browser_sessions_returns_only_owned(vm: VM) -> None:
    vm.open_browser_session()
    vm.open_browser_session()
    sessions = vm.list_browser_sessions()
    assert len(sessions) == 2
    # A second VM for a different user sees none.
    other = make_vm(
        session_id="s2",
        user_email="bob@x.com",
        instance_scope="default",
        executor=MockExecutor(),
    )
    assert other.list_browser_sessions() == []


def test_vm_close_browser_session_emits_artifact(vm: VM) -> None:
    seen: list = []
    vm.subscribe_sse(seen.append)
    s = vm.open_browser_session()
    vm.close_browser_session(s.session_id)
    assert any(
        e.type == "artifact" and e.path.endswith("/closed") for e in seen
    )


# ---- governance does not depend on how the tool was reached --------------
# The governor used to be built only by run_turn, so every other way into
# a tool ran unbounded: the /api/tools/{name} route, the SDK, and a
# LangGraph node calling through agent.langchain_tools. Those are runs
# too — a caller looping on the invoke endpoint is the exact shape this
# layer exists to stop.


def test_a_tool_reached_without_a_turn_is_still_governed(vm: VM) -> None:
    assert vm._governor is None, "precondition: no turn has been opened"

    vm._invoke_tool(ToolCallRequest(name="list_my_tasks", args={}))

    assert vm._governor is not None, "a tool ran with no governor at all"


def test_the_budget_is_shared_across_direct_invocations(vm: VM) -> None:
    """One governor per session, not one per call — otherwise the budget
    resets every time and bounds nothing."""
    vm._invoke_tool(ToolCallRequest(name="list_my_tasks", args={}))
    first = vm._governor

    vm._invoke_tool(ToolCallRequest(name="list_my_tasks", args={}))

    assert vm._governor is first


def test_a_runaway_direct_caller_is_halted(vm: VM) -> None:
    outcomes = [
        str(vm._invoke_tool(ToolCallRequest(name="list_my_tasks", args={})))
        for _ in range(60)
    ]

    assert any(
        "budget" in o.lower() or "halt" in o.lower() or "stopped" in o.lower()
        for o in outcomes
    ), "60 identical direct invocations ran unbounded"


def test_malformed_calls_are_governed_not_invisible(vm: VM) -> None:
    """The governor sees failed calls too: a caller hammering a tool
    with invalid args used to run unbounded because the failure path
    never recorded a step. Repetition/stagnation must catch it."""
    outcomes = [
        str(
            vm._invoke_tool(
                ToolCallRequest(name="list_my_tasks", args={"bogus": i})
            )
        )
        for i in range(60)
    ]

    failed = [s for s in vm._governor.steps if s.kind == "tool" and not s.ok]
    assert failed, "failed calls were invisible to the governor"
    assert any(
        "halt" in o.lower() or "budget" in o.lower() or "stagnation" in o.lower()
        for o in outcomes
    ), "60 malformed invocations ran unbounded"


def test_unknown_tools_are_governed_and_audited(vm: VM) -> None:
    """Garbage tool names are steps too — a loop on them is exactly the
    runaway shape the governor exists to stop, and oversight should
    see the attempts."""
    for _ in range(5):
        vm._invoke_tool(ToolCallRequest(name="no_such_tool_xyz", args={}))

    assert any(s.name == "no_such_tool_xyz" for s in vm._governor.steps), (
        "unknown-tool calls were invisible to the governor"
    )


def test_opening_a_turn_still_gets_its_own_governor(vm: VM) -> None:
    """The lazy one must not outlive its usefulness: a real turn gets a
    fresh budget sized for that turn, not whatever a previous direct
    caller left behind."""
    vm._invoke_tool(ToolCallRequest(name="list_my_tasks", args={}))
    lazy = vm._governor

    vm.run_turn(text="hello")

    assert vm._governor is not lazy


# ---- mid-run steering ------------------------------------------------------


def test_steer_outside_a_turn_is_refused_not_queued(vm: VM) -> None:
    """No turn, no steering: the caller learns the note was not
    delivered instead of believing it was read."""
    assert vm.steer("do it differently") is False
    assert vm.drain_steering() == []


def test_steer_during_a_turn_is_queued_and_drained(vm: VM) -> None:
    """A note arriving mid-turn is queued, then taken by the loop's
    drain — the next safe point, never mid-tool."""

    vm._turn_active = True
    try:
        assert vm.steer("focus on billing", author="bob@x.com") is True
        assert vm.steer("", author="bob@x.com") is False, "empty notes are noise"
        notes = vm.drain_steering()
        assert notes == [("bob@x.com", "focus on billing")]
        assert vm.drain_steering() == [], "drain must empty the mailbox"
    finally:
        vm._turn_active = False


def test_a_stopped_turn_rejects_steering(vm: VM) -> None:
    vm._turn_active = True
    vm._turn_active = False
    assert vm.steer("too late") is False


def test_steering_reaches_the_react_loop_history(vm: VM) -> None:
    """The full chain: a note queued mid-turn becomes a user message
    the model sees on its next pass, and the loop's iteration budget
    is not charged for the correction pass."""
    from agent.omnigent import MockExecutor

    # Scripted executor: first pass asks for a tool, the next pass
    # finishes with text. The steering note is drained after the first
    # tool completes, mimicking a crew member reading the work.
    ex = MockExecutor()
    ex.enqueue_tool_call("list_my_tasks", {})
    ex.enqueue_response("Done with the corrected plan.")

    from agent.harness.react_loop import ReactLoop

    loop = ReactLoop(
        executor=ex,
        agent_def=vm._agent_def,
        history=[],
        system_prompt="You are Alex.",
        pump=vm._pump,
        constitution_adapter=vm._agent_def.policies["constitution"],
        poll_steering=lambda: [("bob@x.com", "focus on billing")],
    )
    loop._vm = vm

    vm._turn_active = True
    try:
        final, history = loop.run()
    finally:
        vm._turn_active = False

    steered = [
        h for h in history
        if h.get("role") == "user" and "steering from bob@x.com" in str(h.get("content"))
    ]
    assert steered, "the note never reached the model's history"


def test_continuous_steering_cannot_make_the_loop_eternal(vm: VM) -> None:
    """The steering refund is capped: a crew member posting on every
    pass used to keep the refund ahead of the iteration counter, and
    the run never terminated. The cap gives each correction one extra
    pass, up to a finite number of extensions."""
    from agent.omnigent import MockExecutor

    ex = MockExecutor()
    # Every pass asks for the same tool, forever — the model never
    # finishes. Only the refund cap ends this.
    for _ in range(30):
        ex.enqueue_tool_call("list_my_tasks", {})

    from agent.harness.react_loop import ReactLoop

    loop = ReactLoop(
        executor=ex,
        agent_def=vm._agent_def,
        history=[],
        system_prompt="You are Alex.",
        pump=vm._pump,
        constitution_adapter=vm._agent_def.policies["constitution"],
        # Steering available on every poll — the adversarial case.
        poll_steering=lambda: [("bob@x.com", "again")],
    )
    loop._vm = vm

    vm._turn_active = True
    try:
        final, history = loop.run(max_turns=8)
    finally:
        vm._turn_active = False

    # The loop terminated (rather than hanging) and the budget was
    # exceeded exactly once — 8 work passes + 3 refunded corrections.
    calls = [h for h in history if h.get("role") == "tool_call"]
    assert len(calls) <= 8 + 3, (
        f"{len(calls)} tool passes ran; the refund made the loop unbounded"
    )


# ---- capability filtering must fail closed -------------------------------
# The VM does not gate at call time: it filters the tool list instead,
# which is the stronger property — an unauthorised tool is never offered
# rather than refused after the model picks it. But it also means this is
# the ONLY place that decides. `guardrails.guarded()`, which several
# docstrings name as the call-time enforcement point, has no production
# call sites at all, so there is no second line here.
#
# Two inputs used to disable the filter entirely:
#   * a user with no granted capabilities  -> every tool offered
#   * any failure reading the ACL store    -> every tool offered
# The second meant a control-plane outage escalated every user to
# run_python and vm_exec.


def _caps_for(vm_obj, rows_or_error):
    import agent.vm as vmmod

    vmmod._ACL_CACHE.clear()

    class Warehouse:
        def execute(self, *a, **k):
            if isinstance(rows_or_error, Exception):
                raise rows_or_error

            rows = rows_or_error

            class R:
                def as_dicts(self):
                    return rows

            return R()

    return vm_obj._query_capabilities(Warehouse())


def test_a_user_with_no_grants_gets_no_tools(vm: VM) -> None:
    """An empty grant is a real answer: it means nothing, not everything."""
    assert _caps_for(vm, []) == set()


def test_granted_capabilities_are_returned(vm: VM) -> None:
    assert _caps_for(vm, [{"capability": "tasks.read"}]) == {"tasks.read"}


def test_an_acl_outage_denies_everything_in_production(vm: VM, monkeypatch) -> None:
    """A control-plane outage must not escalate anyone. An agent that can
    do nothing is the correct outcome."""
    monkeypatch.setenv("ALEX_ENV", "production")
    from alex import config

    config._settings = None
    try:
        assert _caps_for(vm, RuntimeError("warehouse down")) == set()
    finally:
        config._settings = None


def test_an_absent_acl_table_stays_permissive_in_dev(vm: VM) -> None:
    """The one case where unfiltered is right: a local machine with no
    control plane, where the alternative is an agent that can do nothing
    and a developer who cannot work."""
    assert _caps_for(vm, RuntimeError("no such table")) is None


def test_the_tool_list_is_actually_filtered_by_the_result(vm: VM, monkeypatch) -> None:
    """The end of the chain: the decision has to reach the model."""
    monkeypatch.setattr(vm, "_allowed_capabilities", lambda: {"tasks.read"})
    exposed = set(vm._build_agent_def().tools)

    assert "list_my_tasks" in exposed
    assert "run_python" not in exposed
    assert "vm_exec" not in exposed


def test_no_capabilities_means_no_dangerous_tools_are_offered(
    vm: VM, monkeypatch
) -> None:
    monkeypatch.setattr(vm, "_allowed_capabilities", lambda: set())
    exposed = set(vm._build_agent_def().tools)

    assert exposed == set(), f"tools offered to a principal with no grants: {exposed}"


def test_a_broken_warehouse_denies_everything_in_production(
    vm: VM, monkeypatch
) -> None:
    """The companion case to the outage test one level down: when the
    warehouse cannot even be *acquired* (bad config, credential error
    constructing the client), production must still deny rather than
    treat it as "no filter"."""
    import agent.vm as vmmod

    monkeypatch.setenv("ALEX_ENV", "production")
    from alex import config

    config._settings = None
    vmmod._ACL_CACHE.clear()
    try:

        def boom():
            raise RuntimeError("warehouse client construction failed")

        monkeypatch.setattr(vmmod, "_vm_warehouse", boom)
        assert vm._allowed_capabilities() == set()
    finally:
        config._settings = None
        vmmod._ACL_CACHE.clear()


def test_a_broken_warehouse_stays_permissive_in_dev(vm: VM, monkeypatch) -> None:
    """Dev keeps the permissive path: a local machine with no control
    plane is the one case unfiltered is right."""
    import agent.vm as vmmod

    monkeypatch.setenv("ALEX_ENV", "dev")
    from alex import config

    config._settings = None
    vmmod._ACL_CACHE.clear()
    try:

        def boom():
            raise RuntimeError("no warehouse configured")

        monkeypatch.setattr(vmmod, "_vm_warehouse", boom)
        assert vm._allowed_capabilities() is None
    finally:
        config._settings = None
        vmmod._ACL_CACHE.clear()
