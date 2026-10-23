"""The swarm — bounded, policy-ceilinged, fully visible delegation.

Three properties carry the design, and each has teeth here:

* **Bounded.** Never more than ten workers per delegation, never more
  than ten concurrent, and depth-one — a worker cannot delegate, so the
  cap cannot be multiplied away by recursion.
* **Ceilinged.** A worker's capabilities are parent ∩ requested, minus
  delegation, enforced INSIDE the child VM (`capability_ceiling`), not
  by the swarm module's goodwill.
* **Visible.** Every worker's lifecycle and forwarded work lands on the
  parent's pump as named `subagent` events, so the person watching the
  run sees each lane live and supervision reviews the same record.
"""

from __future__ import annotations

import threading
import time

import pytest

from agent import swarm
from agent.omnigent import MockExecutor
from agent.vm import make_vm


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("ALEX_ENV", "dev")
    from alex import config, testing

    config._settings = None
    testing._fakes = None
    yield
    config._settings = None
    testing._fakes = None


@pytest.fixture
def parent(env):
    return make_vm(
        session_id="lead-1", user_email="demo@alex.dev", executor=MockExecutor()
    )


class ChildVM:
    """A scripted worker VM, factory-injected."""

    def __init__(self, session_id, user_email, capability_ceiling=None, **_kw):
        self.session_id = session_id
        self.user_email = user_email
        self.ceiling = capability_ceiling
        self.closed = False
        self._subs = []

    def subscribe_sse(self, fn):
        self._subs.append(fn)
        return lambda: None

    def run_turn(self, *, text, **_kw):
        for fn in self._subs:
            fn({"type": "tool", "tool": "search_enterprise", "args": {}})
            fn({"type": "text", "text": "found it"})
        return f"report from {self.session_id.rsplit(':', 1)[-1]}", []

    def close(self):
        self.closed = True


def _factory(made: list):
    def make(**kw):
        vm = ChildVM(**kw)
        made.append(vm)
        return vm

    return make


# ---- bounds ---------------------------------------------------------------


def test_more_than_ten_workers_is_refused(parent):
    tasks = [{"name": f"w{i}", "brief": "x"} for i in range(11)]

    with pytest.raises(ValueError, match="policy limit"):
        swarm.run_swarm(parent, tasks, vm_factory=_factory([]))


def test_concurrency_never_exceeds_the_cap(parent, monkeypatch):
    monkeypatch.setattr(swarm, "MAX_CONCURRENT", 3)
    live = {"now": 0, "peak": 0}
    lock = threading.Lock()

    class SlowChild(ChildVM):
        def run_turn(self, *, text, **_kw):
            with lock:
                live["now"] += 1
                live["peak"] = max(live["peak"], live["now"])
            time.sleep(0.05)
            with lock:
                live["now"] -= 1
            return "ok", []

    out = swarm.run_swarm(
        parent,
        [{"name": f"w{i}", "brief": "x"} for i in range(9)],
        vm_factory=lambda **kw: SlowChild(**kw),
    )

    assert out["succeeded"] == 9
    assert live["peak"] <= 3, f"peak concurrency {live['peak']} exceeded the cap"


def test_a_worker_can_never_delegate(parent):
    """Depth-one by construction: the ceiling strips agents.delegate even
    when the brief explicitly requests it."""
    made: list[ChildVM] = []
    swarm.run_swarm(
        parent,
        [{"name": "sneaky", "brief": "x",
          "capabilities": ["agents.delegate", "tasks.read"]}],
        vm_factory=_factory(made),
    )

    assert made[0].ceiling is not None
    assert swarm.DELEGATE_CAPABILITY not in made[0].ceiling


def test_the_env_override_cannot_raise_the_policy_cap(monkeypatch):
    """Operators may lower the bounds; ten is the ceiling regardless."""
    import importlib

    monkeypatch.setenv("ALEX_SWARM_PER_TASK", "50")
    monkeypatch.setenv("ALEX_SWARM_CONCURRENT", "99")
    reloaded = importlib.reload(swarm)
    try:
        assert reloaded.MAX_PER_TASK == 10
        assert reloaded.MAX_CONCURRENT == 10
    finally:
        monkeypatch.undo()
        importlib.reload(swarm)


# ---- the ceiling ----------------------------------------------------------


def test_workers_get_the_intersection_not_the_request(parent, monkeypatch):
    monkeypatch.setattr(
        parent, "_allowed_capabilities", lambda: {"tasks.read", "knowledge.read"}
    )
    made: list[ChildVM] = []
    swarm.run_swarm(
        parent,
        [{"name": "scout", "brief": "x",
          "capabilities": ["tasks.read", "email.send"]}],
        vm_factory=_factory(made),
    )

    # email.send was requested but the parent does not hold it.
    assert made[0].ceiling == frozenset({"tasks.read"})


def test_an_empty_request_inherits_the_parent_minus_delegation(parent, monkeypatch):
    monkeypatch.setattr(
        parent, "_allowed_capabilities",
        lambda: {"tasks.read", swarm.DELEGATE_CAPABILITY},
    )
    made: list[ChildVM] = []
    swarm.run_swarm(
        parent, [{"name": "w", "brief": "x"}], vm_factory=_factory(made)
    )

    assert made[0].ceiling == frozenset({"tasks.read"})


def test_the_ceiling_is_enforced_inside_a_real_child_vm(env):
    """End of the chain: a real VM built with the ceiling exposes only
    those tools to its model. The swarm's promise is the VM's promise."""
    vm = make_vm(
        session_id="ceil-1", user_email="demo@alex.dev",
        executor=MockExecutor(),
        capability_ceiling=frozenset({"tasks.read"}),
    )
    exposed = set(vm._build_agent_def().tools)

    assert "list_my_tasks" in exposed
    assert "delegate_subagents" not in exposed
    assert "run_python" not in exposed


# ---- visibility -----------------------------------------------------------


def test_every_worker_is_named_and_streams_into_the_parent(parent):
    seen: list = []
    parent.pump.subscribe(lambda e: seen.append(e))

    out = swarm.run_swarm(
        parent,
        [{"name": "pricing", "brief": "find pricing"},
         {"name": "contracts", "brief": "find contracts"}],
        vm_factory=_factory([]),
    )

    sub = [e for e in seen if getattr(e, "type", "") == "subagent"]
    names = {e.source for e in sub}
    phases = {(e.source, e.phase) for e in sub}
    assert names == {"pricing", "contracts"}
    for name in names:
        assert (name, "spawned") in phases
        assert (name, "done") in phases
        assert (name, "tool") in phases, "worker tool calls were not forwarded"
        assert (name, "text") in phases, "worker output was not forwarded"
    assert out["succeeded"] == 2


def test_a_failed_worker_appears_in_the_results_not_a_hole(parent):
    class Exploding(ChildVM):
        def run_turn(self, *, text, **_kw):
            raise RuntimeError("model exploded")

    out = swarm.run_swarm(
        parent,
        [{"name": "ok", "brief": "x"}, {"name": "boom", "brief": "y"}],
        vm_factory=lambda **kw: (
            Exploding(**kw) if "boom" in kw["session_id"] else ChildVM(**kw)
        ),
    )

    assert out["ok"] is False and out["succeeded"] == 1
    by_name = {r["name"]: r for r in out["results"]}
    assert by_name["boom"]["status"] == "failed"
    assert "model exploded" in by_name["boom"]["error"]
    assert by_name["ok"]["status"] == "done"


def test_worker_vms_are_always_closed(parent):
    made: list[ChildVM] = []
    swarm.run_swarm(
        parent,
        [{"name": f"w{i}", "brief": "x"} for i in range(4)],
        vm_factory=_factory(made),
    )
    assert all(c.closed for c in made), "a worker VM was leaked"


def test_duplicate_names_are_disambiguated_not_merged(parent):
    made: list[ChildVM] = []
    out = swarm.run_swarm(
        parent,
        [{"name": "scout", "brief": "a"}, {"name": "scout", "brief": "b"}],
        vm_factory=_factory(made),
    )
    assert {r["name"] for r in out["results"]} == {"scout", "scout-2"}


# ---- the tool entry point -------------------------------------------------


def test_the_tool_refuses_to_run_without_a_parent_vm(env):
    out = swarm.delegate_subagents([{"name": "w", "brief": "x"}])
    assert out["ok"] is False and "session" in out["error"]


def test_the_tool_finds_its_parent_through_the_contextvar(parent, monkeypatch):
    # Stub the child factory: this test is about the contextvar wiring,
    # not about running a real model. (Unstubbed, the child VM builds a
    # real executor and calls the local model — sixty seconds of Ollama
    # inside a unit test.)
    import agent.vm as vmmod

    monkeypatch.setattr(
        vmmod, "make_vm", lambda **kw: ChildVM(
            session_id=kw["session_id"], user_email=kw["user_email"],
            capability_ceiling=kw.get("capability_ceiling"),
        )
    )
    token = swarm.set_current_vm(parent)
    try:
        out = swarm.delegate_subagents('[{"name": "w", "brief": "x"}]')
    finally:
        swarm.reset_current_vm(token)
    assert out["workers"] == 1
    assert {r["name"] for r in out["results"]} == {"w"}


def test_a_bad_task_list_is_an_error_result_not_an_exception(parent):
    token = swarm.set_current_vm(parent)
    try:
        out = swarm.delegate_subagents("not json at all")
    finally:
        swarm.reset_current_vm(token)
    assert out["ok"] is False and "JSON" in out["error"]
