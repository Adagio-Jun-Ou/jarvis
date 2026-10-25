"""Jarvis's native tools under the engine's governance.

* a Tool subclass becomes a governed ToolSpec: synthesized signature
  (required vs optional, typed), original JSON schema carried through,
  ToolResult mapped to {"ok": ...};
* requires_permission -> an .act capability (power 2, policy-ASK);
  outside-world categories -> untrusted results;
* discovery registers a real, large slice of Jarvis without exceptions;
* the adapted tool runs end to end through the VM.
"""

from __future__ import annotations

import inspect

import pytest

from agent import jarvis_tools
from tools.base import Tool, ToolResult


class EchoTool(Tool):
    name = "echo_demo"
    description = "Echo a message a number of times"
    parameters = {
        "type": "object",
        "properties": {
            "message": {"type": "string", "description": "what to say"},
            "times": {"type": "integer", "description": "repeat count"},
        },
        "required": ["message"],
    }
    requires_permission = False

    async def execute(self, message: str, times: int | None = None) -> str:
        return " ".join([message] * (times or 1))


class DangerTool(Tool):
    name = "danger_demo"
    description = "Something consequential"
    parameters = {"type": "object", "properties": {"target": {"type": "string"}},
                  "required": ["target"]}
    requires_permission = True

    async def execute(self, target: str) -> ToolResult:
        return ToolResult(success=False, error=f"refused {target}")


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("ALEX_ENV", "dev")
    from alex import config, testing

    config._settings = None
    testing._fakes = None
    yield
    jarvis_tools.unregister_jarvis_tools()
    config._settings = None
    testing._fakes = None


def test_a_tool_becomes_a_governed_spec(env):
    from agent.execution_governor import CAPABILITY_POWER
    from agent.guardrails import TOOL_CAPABILITY
    from agent.tool_policy import decision
    from agent.tools.registry import TOOL_REGISTRY

    reg = jarvis_tools.register_jarvis_tools([EchoTool(), DangerTool()])
    assert reg["echo_demo"] == "jarvis.misc.read"
    assert reg["danger_demo"] == "jarvis.misc.act"
    spec = TOOL_REGISTRY["echo_demo"]
    sig = inspect.signature(spec.callable)
    assert sig.parameters["message"].default is inspect.Parameter.empty
    assert sig.parameters["times"].default is None
    assert sig.parameters["times"].annotation is int
    assert spec.input_schema["properties"]["message"]["description"] == "what to say"
    assert TOOL_CAPABILITY["danger_demo"] == "jarvis.misc.act"
    assert CAPABILITY_POWER["jarvis.misc.act"] == 2
    assert decision("danger_demo", "jarvis.misc.act") == "ask"
    # Idempotent.
    again = jarvis_tools.register_jarvis_tools([EchoTool()])
    assert again == {"echo_demo": "jarvis.misc.read"}


def test_results_map_to_the_engine_shape(env):
    jarvis_tools.register_jarvis_tools([EchoTool(), DangerTool()])
    from agent.tools.registry import TOOL_REGISTRY

    out = TOOL_REGISTRY["echo_demo"].callable(message="hi", times=2, user_email="d@x")
    assert out == {"ok": True, "output": "hi hi"}
    out = TOOL_REGISTRY["danger_demo"].callable(target="prod")
    assert out["ok"] is False and "refused prod" in out["error"]


def test_the_adapted_tool_runs_through_the_vm(env):
    jarvis_tools.register_jarvis_tools([EchoTool()])
    from agent.omnigent import MockExecutor
    from agent.omnigent.executor import ToolCallRequest
    from agent.vm import make_vm

    vm = make_vm(session_id="chat-jt", user_email="dana@alex.dev", executor=MockExecutor())
    assert "echo_demo" not in vm.agent_def.tools, "adapted tools are deferred by default"
    vm.pin_tools(["echo_demo"])
    assert "echo_demo" in vm.agent_def.tools
    schema = vm.agent_def.tools["echo_demo"].tool_schema()["parameters"]
    assert schema["required"] == ["message"]
    res = vm._invoke_tool(ToolCallRequest(name="echo_demo", args={"message": "yo"}))
    assert res == {"ok": True, "output": "yo"}
    bad = vm._invoke_tool(ToolCallRequest(name="echo_demo", args={"nope": 1}))
    assert bad["ok"] is False, "unknown args are refused before execution"
    vm.close()


def test_discovery_registers_a_large_slice_of_jarvis(env):
    found = jarvis_tools.discover()
    names = {t.name for t in found}
    assert len(found) >= 40, f"only {len(found)} jarvis tools discovered"
    reg = jarvis_tools.register_jarvis_tools(found)
    assert len(reg) >= 40
    assert all(c.startswith("jarvis.") for c in reg.values())
    # Screen and web reads are outside-authored content.
    from agent.tools.registry import TOOL_REGISTRY

    for n in ("see_screen", "web_search"):
        if n in names:
            assert TOOL_REGISTRY[n if n in TOOL_REGISTRY else f"jarvis_{n}"].untrusted_output
