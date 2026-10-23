"""
Tests for the SDK core decorators and Agent validation.

The SDK lives in `sdk/alex_sdk/`. We add the `sdk/` directory to
`sys.path` so `import alex_sdk` works without an install step.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sdk"))

import alex_sdk
import pytest
from alex_sdk.core import Agent, capability, skill, tool
from alex_sdk.workflow import workflow


@pytest.fixture(autouse=True)
def _reset_registry():
    """The module-level _REGISTRY is shared across tests. Reset it
    between tests so capability declarations don't leak."""
    from alex_sdk import core as _core
    _core._REGISTRY = _core._Registry()
    yield
    _core._REGISTRY = _core._Registry()


def test_capability_registers():
    capability("tasks.create")
    from alex_sdk import core as _core
    assert _core._REGISTRY.has_capability("tasks.create")


def test_tool_binds_capability():
    capability("tasks.create")
    @tool(capability="tasks.create")
    def create_task(title: str) -> str:
        return title
    assert create_task("hi") == "hi"
    assert create_task.__alex__["capability"] == "tasks.create"


def test_tool_rejects_undeclared_capability():
    with pytest.raises(ValueError, match="undeclared capability"):
        @tool(capability="does.not.exist")
        def f() -> None:
            pass


def test_skill_requires_declared_capabilities():
    capability("email.draft")
    @skill("weekly-brief", trigger="weekly", uses=["email.draft"])
    def f() -> None:
        pass
    with pytest.raises(ValueError, match="undeclared capability 'foo'"):
        @skill("bad", trigger="x", uses=["foo"])
        def g() -> None:
            pass


def test_agent_validates_tools_and_workflows():
    capability("analytics.read")
    capability("email.draft")
    @tool(capability="email.draft", confirm=True)
    def send_email():
        return "ok"
    weekly = (workflow("weekly")
              .step("gather", capability="analytics.read")
              .confirm("send", tool=send_email))
    agent = Agent(name="x", tools=[send_email], workflows=[weekly])
    # Should not raise
    agent._validate()


def test_agent_validation_fails_for_undeclared_tool_capability():
    """The @tool decorator is the line of defense: an undeclared
    capability fails at decoration time, before Agent._validate even
    runs. This is intentional — we want the SDK to fail fast."""
    with pytest.raises(ValueError, match="undeclared capability 'foo.bar'"):
        @tool(capability="foo.bar")
        def f():
            return "x"


def test_agent_validation_fails_for_undeclared_workflow_capability():
    capability("email.draft")
    @tool(capability="email.draft")
    def f():
        return "x"
    weekly = workflow("weekly").step("send", tool=f).step("other", capability="missing.cap")
    agent = Agent(name="x", tools=[f], workflows=[weekly])
    with pytest.raises(ValueError, match="undeclared capability 'missing.cap'"):
        agent._validate()


def test_workflow_to_n8n_shape():
    capability("c1")
    w = workflow("x", description="weekly").step("a", capability="c1")
    w.confirm("b", capability="c1")
    n8n = w.to_n8n()
    assert n8n["name"] == "x"
    assert n8n["nodes"][0]["requires_dual_approval"] is False
    assert n8n["nodes"][1]["requires_dual_approval"] is True


def test_workflow_to_native_plan_shape():
    capability("c1")
    w = workflow("x").step("a", capability="c1")
    plan = w.to_native_plan()
    assert plan["name"] == "x"
    assert plan["plan"][0]["two_phase"] is False


def test_version_is_string():
    assert isinstance(alex_sdk.__version__, str)
    assert alex_sdk.__version__ == "2.0.0"
