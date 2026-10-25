"""The VM's tool working set stays bounded and free of duplicates.

Every tool schema is prompt the model re-reads on every turn, and
selection accuracy degrades sharply as the count grows. Two failures are
easy to ship and hard to notice: an unbounded remote catalogue, and a
remote server that re-exposes tools the agent already has natively.
"""

from __future__ import annotations

import pytest

from agent import mcp_client as mc


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("ALEX_ENV", "dev")
    from alex import config, testing

    config._settings = None
    testing._fakes = None
    mc.reset()
    mc.set_connection_factory(None)
    yield
    config._settings = None
    testing._fakes = None
    mc.reset()
    mc.set_connection_factory(None)


def make_vm(monkeypatch, remote_tools):
    """A VM whose MCP discovery returns exactly ``remote_tools``."""
    from agent import vm as vm_mod
    from agent.omnigent import FunctionTool

    def fake_mcp(self):
        return {
            name: FunctionTool(
                name=name, description="remote",
                callable=lambda **_kw: None, scopes={"mcp.use"},
            )
            for name in remote_tools
        }

    monkeypatch.setattr(vm_mod.VM, "_mcp_tools", fake_mcp)
    return vm_mod.make_vm(session_id="s1", user_email="demo@alex.dev")


def test_remote_tools_join_the_working_set(monkeypatch, env):
    vm = make_vm(monkeypatch, ["mcp__github__create_issue"])
    try:
        assert "mcp__github__create_issue" in vm.agent_def.tools
    finally:
        vm.close()


def test_a_remote_duplicate_of_a_builtin_is_dropped(monkeypatch, env):
    """A server re-exposing `search_knowledge` doubles the prompt cost and
    teaches the model to pick at random."""
    vm = make_vm(
        monkeypatch,
        ["mcp__alexself__search_knowledge", "mcp__github__create_issue"],
    )
    try:
        names = set(vm.agent_def.tools)
        assert "mcp__alexself__search_knowledge" not in names
        assert "mcp__github__create_issue" in names
        # The native one is still there.
        assert "search_knowledge" in names
    finally:
        vm.close()


def test_remote_tools_only_fill_the_room_that_is_left(monkeypatch, env):
    """A large remote catalogue cannot inflate the prompt without bound.

    Built-ins are not evicted to make space — they are governed, audited
    and known — so the guarantee is that remote tools take only the
    headroom under the cap.
    """
    from agent import vm as vm_mod

    vm = make_vm(monkeypatch, [f"mcp__srv__tool{i}" for i in range(500)])
    try:
        names = set(vm.agent_def.tools)
        builtin = {n for n in names if not n.startswith("mcp__")}
        remote = names - builtin
        assert len(remote) <= max(
            vm_mod.MAX_TOOLS_IN_PROMPT - len(builtin), 0
        )
        assert len(names) <= max(len(builtin), vm_mod.MAX_TOOLS_IN_PROMPT)
    finally:
        vm.close()


def test_builtins_are_never_evicted_by_remote_tools(monkeypatch, env):
    """Built-ins are governed, audited and known; they keep their place."""
    from agent import vm as vm_mod

    monkeypatch.setattr(vm_mod, "MAX_TOOLS_IN_PROMPT", 5)
    vm = make_vm(monkeypatch, [f"mcp__srv__tool{i}" for i in range(50)])
    try:
        names = set(vm.agent_def.tools)
        assert not any(n.startswith("mcp__") for n in names), (
            "with no room left, remote tools must yield to built-ins"
        )
        assert names, "the built-in set must survive"
    finally:
        vm.close()


def test_discovery_failure_leaves_the_builtins_intact(monkeypatch, env):
    from agent import vm as vm_mod

    def boom(self):
        raise RuntimeError("mcp down")

    monkeypatch.setattr(vm_mod.VM, "_mcp_tools", boom)
    vm = vm_mod.make_vm(session_id="s1", user_email="demo@alex.dev")
    try:
        # A broken integration must not take the agent's own tools with it.
        assert "search_knowledge" in vm.agent_def.tools
    finally:
        vm.close()
