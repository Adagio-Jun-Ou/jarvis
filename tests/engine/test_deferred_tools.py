"""The deferred tool library: hundreds of tools, none of them in every
model pass.

* deferred tools are registered and governed but out of the default
  working set;
* a request that names what it needs pulls the matching tool in for
  that turn;
* find_tools pins matches so they are callable in the SAME turn;
* the built-in working set (and the room left for MCP tools) is
  unaffected by the library's size.
"""

from __future__ import annotations

import pytest

from agent import jarvis_tools
from tools.base import Tool


def _tool(name, desc):
    class T(Tool):
        parameters = {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}
        requires_permission = False

        async def execute(self, q: str) -> str:
            return f"{name}:{q}"

    T.name, T.description = name, desc
    return T()


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("ALEX_ENV", "dev")
    from alex import config, testing

    config._settings = None
    testing._fakes = None
    jarvis_tools.register_jarvis_tools([
        _tool("compose_music", "Compose a piece of music in a given style"),
        _tool("docker_ps", "List running docker containers"),
        _tool("calendar_add", "Add an event to the calendar"),
    ])
    yield
    jarvis_tools.unregister_jarvis_tools()
    config._settings = None
    testing._fakes = None


def _vm():
    from agent.omnigent import MockExecutor
    from agent.vm import make_vm

    return make_vm(session_id="chat-def", user_email="dana@alex.dev", executor=MockExecutor())


def test_deferred_tools_stay_out_of_the_default_set(env):
    from agent.tools.registry import TOOL_REGISTRY

    assert TOOL_REGISTRY["compose_music"].deferred
    vm = _vm()
    assert "compose_music" not in vm.agent_def.tools
    assert "view" in vm.agent_def.tools, "built-ins are unaffected"
    vm.close()


def test_a_relevant_request_pulls_the_tool_in_for_the_turn(env):
    vm = _vm()
    vm.run_turn(text="please compose some music for the evening")
    assert "compose_music" in vm.agent_def.tools
    assert "docker_ps" not in vm.agent_def.tools
    vm.close()


def test_find_tools_pins_matches_immediately(env):
    from agent import swarm
    from agent.omnigent.executor import ToolCallRequest

    vm = _vm()
    token = swarm.set_current_vm(vm)
    try:
        res = vm._invoke_tool(ToolCallRequest(name="find_tools", args={"query": "docker containers"}))
    finally:
        swarm.reset_current_vm(token)
    assert res["ok"] and [t["name"] for t in res["tools"]] == ["docker_ps"]
    assert "docker_ps" in vm.agent_def.tools, "callable this turn, not next"
    out = vm._invoke_tool(ToolCallRequest(name="docker_ps", args={"q": "all"}))
    assert out == {"ok": True, "output": "docker_ps:all"}
    # And it stays for later turns.
    vm.run_turn(text="hello again")
    assert "docker_ps" in vm.agent_def.tools
    vm.close()


def test_the_library_does_not_crowd_out_remote_tools(env):
    from agent.vm import MAX_TOOLS_IN_PROMPT, make_vm
    from agent.omnigent import MockExecutor

    vm = make_vm(session_id="chat-room", user_email="dana@alex.dev", executor=MockExecutor())
    assert len(vm.agent_def.tools) < MAX_TOOLS_IN_PROMPT
    vm.close()


def test_a_used_deferred_tool_stays_in_the_working_set(env):
    """The registry's contract: entry "by having been used before"."""
    from agent.omnigent.executor import ToolCallRequest

    vm = _vm()
    out = vm._invoke_tool(ToolCallRequest(name="docker_ps", args={"q": "all"}))
    assert out == {"ok": True, "output": "docker_ps:all"}
    assert "docker_ps" in vm._pinned
    vm.run_turn(text="hello again")
    assert "docker_ps" in vm.agent_def.tools
    vm.close()


def test_pinned_library_is_bounded(env):
    """A long session must not grow every model pass without bound."""
    from agent.vm import MAX_DEFERRED_IN_PROMPT

    extra = [
        _tool(f"lib_tool_{i}", f"library tool number {i}")
        for i in range(MAX_DEFERRED_IN_PROMPT + 5)
    ]
    jarvis_tools.register_jarvis_tools(extra)
    vm = _vm()
    vm.pin_tools([t.name for t in extra])
    assert len(vm._pinned) == MAX_DEFERRED_IN_PROMPT
    assert "lib_tool_0" not in vm._pinned, "oldest pin is evicted first"
    vm.close()


def test_mutating_tools_are_promoted_to_act(env):
    """Jarvis's shell declares requires_permission=False but runs
    arbitrary commands: the adapter must not classify it read."""
    from agent import tool_policy
    from agent.tools.registry import TOOL_REGISTRY

    jarvis_tools.register_jarvis_tools([_tool("shell", "Execute shell commands")])
    spec = TOOL_REGISTRY["shell"]
    assert spec.capability.endswith(".act")
    assert spec.capability in tool_policy.DEFAULT_ASK_CAPABILITIES


def test_find_tools_without_keywords_lists_the_library(env):
    """A model probing the library calls the tool with no arguments."""
    from agent import swarm
    from agent.omnigent.executor import ToolCallRequest

    vm = _vm()
    token = swarm.set_current_vm(vm)
    try:
        res = vm._invoke_tool(ToolCallRequest(name="find_tools", args={}))
    finally:
        swarm.reset_current_vm(token)
    assert res["ok"] is True
    assert res["tools"] == [] and res["pinned"] == []
    assert sum(entry["count"] for entry in res["library"]) >= 3
    assert "keywords" in res["note"]
    vm.close()


def test_find_tools_returns_the_original_rich_schema(env):
    """The model must see parameter descriptions, not just types."""
    from agent import swarm
    from agent.omnigent.executor import ToolCallRequest

    class RichTool(Tool):
        name = "rich_demo"
        description = "A demonstration tool with documented arguments"
        parameters = {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "what to answer",
                },
            },
            "required": ["question"],
        }
        requires_permission = False

        async def execute(self, question: str) -> str:
            return question

    jarvis_tools.register_jarvis_tools([RichTool()])
    vm = _vm()
    token = swarm.set_current_vm(vm)
    try:
        res = vm._invoke_tool(
            ToolCallRequest(name="find_tools", args={"query": "rich demonstration"})
        )
    finally:
        swarm.reset_current_vm(token)
    hit = next(t for t in res["tools"] if t["name"] == "rich_demo")
    assert hit["parameters"]["properties"]["question"]["description"] == "what to answer"
    assert res["pinned"] == ["rich_demo"]
    vm.close()
