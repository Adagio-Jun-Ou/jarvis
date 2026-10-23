"""Alex's tools inside LangChain and LangGraph.

The interop is only worth having if it does not cost the controls. A
wrapper that called the registry's functions directly would be a hole
straight through the ACL, the governor, the constitution and the audit
log — every one of which lives in `VM._invoke_tool`, not in the tool.

So these tests are mostly not about LangChain. They are about proving
that a tool called from inside somebody else's graph is governed exactly
as one called from an Alex turn.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from agent import langchain_tools as lct
from agent.omnigent import MockExecutor
from agent.vm import make_vm

pytestmark = pytest.mark.skipif(
    not lct.langchain_available(), reason="langchain-core not installed"
)


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
def vm(env):
    return make_vm(
        session_id="lc-session", user_email="demo@alex.dev", executor=MockExecutor()
    )


def _tool(tools, name):
    return next(t for t in tools if t.name == name)


# ---- what gets exported --------------------------------------------------


def test_tools_are_exported_as_langchain_tools(vm):
    from langchain_core.tools import BaseTool

    tools = lct.governed_tools(vm)

    assert tools, "no tools exported at all"
    assert all(isinstance(t, BaseTool) for t in tools)
    assert {t.name for t in tools} <= set(__import__(
        "agent.tools", fromlist=["TOOL_REGISTRY"]
    ).TOOL_REGISTRY)


def test_identity_parameters_are_not_in_the_exported_schema(vm):
    """A model that can see `user_email` is a model that will try to set
    it. The VM overrides it anyway, but the schema is where the attempt
    starts."""
    for tool in lct.governed_tools(vm):
        fields = set(tool.args_schema.model_fields)
        assert not (fields & lct.INJECTED), f"{tool.name} exposes {fields & lct.INJECTED}"


def test_the_export_is_filtered_by_the_principals_acl(vm, monkeypatch):
    """L0: a tool the principal may not use is not offered, rather than
    refused after the model has chosen it."""
    monkeypatch.setattr(vm, "_allowed_capabilities", lambda: {"tasks.read"})

    names = {t.name for t in lct.governed_tools(vm)}

    assert "list_my_tasks" in names
    assert "send_email" not in names
    assert "vm_exec" not in names


def test_names_can_narrow_the_export_but_never_widen_it(vm, monkeypatch):
    """Asking for a tool is not the same as being entitled to it."""
    monkeypatch.setattr(vm, "_allowed_capabilities", lambda: {"tasks.read"})

    names = {t.name for t in lct.governed_tools(vm, names=["send_email"])}

    assert names == set(), "an ACL-excluded tool was exported on request"


# ---- the governance actually applies -------------------------------------


def test_a_tool_call_goes_through_the_vm_not_the_raw_callable(vm, monkeypatch):
    """The whole design: the registry entry is the unguarded function,
    the VM is the guard."""
    seen = {}

    def spy(request):
        seen["name"] = request.name
        seen["args"] = dict(request.args or {})
        return {"ok": True}

    monkeypatch.setattr(vm, "_invoke_tool", spy)
    _tool(lct.governed_tools(vm), "list_my_tasks").invoke({"status": "open"})

    assert seen["name"] == "list_my_tasks"
    assert seen["args"] == {"status": "open"}


def test_a_caller_cannot_impersonate_through_the_bridge(vm, monkeypatch):
    """Identity is dropped before it reaches the VM, so the attempt does
    not even appear in the arguments."""
    seen = {}
    monkeypatch.setattr(
        vm, "_invoke_tool", lambda r: seen.update(args=dict(r.args or {})) or {"ok": True}
    )

    tool = _tool(lct.governed_tools(vm), "list_my_tasks")
    tool.func(status="open", user_email="ceo@alex.dev", instance_scope="other")

    assert "user_email" not in seen["args"]
    assert "instance_scope" not in seen["args"]


def test_a_constitutional_block_reaches_the_caller_as_a_refusal(env):
    """L3 runs inside `_invoke_tool`, so it runs here too — and comes
    back as a result the agent can reason about, not an escape."""
    from alex.testing import get_fakes

    wh = get_fakes().warehouse
    wh.register(
        "media.control.constitution",
        ["article_id", "article_type", "title", "text", "machine_rule",
         "severity", "version", "superseded_by", "effective_from"],
        pk=["article_id"],
    )
    wh.register(
        "media.control.constitution_violations",
        ["id", "ts", "user_email", "session_id", "article_id", "action",
         "disposition", "detail"],
        pk=["id"],
    )
    wh.execute(
        "INSERT INTO media.control.constitution (article_id, article_type, "
        "title, text, machine_rule, severity, effective_from) "
        "VALUES (:a, 'prohibition', 'No shell', 'No shell commands.', :r, "
        "'blocking', :ef)",
        {
            "a": "A9",
            "r": '{"deny_capabilities": ["compute.shell"]}',
            "ef": datetime.now(timezone.utc) - timedelta(days=1),
        },
    )
    governed = make_vm(
        session_id="lc-law", user_email="demo@alex.dev",
        executor=MockExecutor(), warehouse=wh,
    )
    assert governed._session.constitution.articles, "the fixture did not load"

    result = _tool(lct.governed_tools(governed), "vm_exec").invoke(
        {"argv": ["rm", "-rf", "/"]}
    )

    text = str(result)
    assert "A9" in text or "constitution" in text.lower(), (
        f"a prohibited tool ran through the bridge unblocked: {text[:200]}"
    )
    rows = wh.execute(
        "SELECT article_id, disposition FROM media.control.constitution_violations", {}
    ).as_dicts()
    assert rows and rows[0]["disposition"] == "blocked", "the denial left no record"


def test_the_governor_halts_a_runaway_graph(vm):
    """L2 counts calls on the VM, so a LangGraph loop that thrashes is
    stopped by the same budget an Alex turn is."""
    tool = _tool(lct.governed_tools(vm), "list_my_tasks")

    outcomes = [str(tool.invoke({"status": "open"})) for _ in range(60)]

    assert any(
        "budget" in o.lower() or "halt" in o.lower() or "stopped" in o.lower()
        for o in outcomes
    ), "60 identical calls ran unbounded through the bridge"


# ---- message translation -------------------------------------------------


def test_tool_calls_survive_the_round_trip():
    """A model that cannot see what it already asked for asks again."""
    from langchain_core.messages import AIMessage

    from agent.langgraph_agent import to_alex_message

    wire = to_alex_message(
        AIMessage(
            content="",
            tool_calls=[{"name": "send_email", "args": {"to": "a@b.c"},
                         "id": "c1", "type": "tool_call"}],
        )
    )

    assert wire["role"] == "assistant"
    assert wire["tool_calls"][0]["function"]["name"] == "send_email"
    assert wire["tool_calls"][0]["function"]["arguments"] == {"to": "a@b.c"}


def test_a_tool_result_carries_which_call_it_answered():
    """With parallel tool calls, a bare result forces the model to guess."""
    from langchain_core.messages import ToolMessage

    from agent.langgraph_agent import to_alex_message

    wire = to_alex_message(
        ToolMessage(content="2 tasks", name="list_my_tasks", tool_call_id="c1")
    )

    assert wire["role"] == "tool"
    assert wire["name"] == "list_my_tasks"
    assert wire["tool_call_id"] == "c1"


def test_system_and_human_messages_map_across():
    from langchain_core.messages import HumanMessage, SystemMessage

    from agent.langgraph_agent import to_alex_message

    assert to_alex_message(SystemMessage(content="be brief")) == {
        "role": "system", "content": "be brief"
    }
    assert to_alex_message(HumanMessage(content="hello")) == {
        "role": "user", "content": "hello"
    }


def test_a_model_response_without_a_call_id_still_gets_one():
    """LangGraph pairs the ToolMessage back by id; an empty one silently
    breaks multi-tool turns."""
    from agent.langgraph_agent import _to_langchain_ai
    from alex.llm import LLMResponse, ToolCall

    ai = _to_langchain_ai(
        LLMResponse(content="", tool_calls=[ToolCall(name="x", args={}, id="")])
    )

    assert ai.tool_calls[0]["id"], "a tool call went out with no id"


# ---- the graph -----------------------------------------------------------


def test_a_langgraph_agent_runs_a_governed_tool_and_answers(vm, monkeypatch):
    from agent import langgraph_agent as lg
    from alex import llm as llmmod
    from alex.llm import LLMResponse, ToolCall

    if not lg.langgraph_available():
        pytest.skip("langgraph not installed")

    turns = {"n": 0}
    offered: list[str] = []

    class Scripted:
        def invoke(self, messages, tools=None, temperature=0.1, **kw):
            turns["n"] += 1
            if turns["n"] == 1:
                offered.extend(t["function"]["name"] for t in (tools or []))
                return LLMResponse(
                    content="",
                    tool_calls=[ToolCall(name="list_my_tasks", args={}, id="c1")],
                )
            return LLMResponse(content="You have no open tasks.")

    llmmod.set_llm(Scripted())
    try:
        answer = lg.run(vm, "what are my tasks?")
    finally:
        llmmod.reset_llm()

    assert turns["n"] == 2, "the graph did not complete a tool round trip"
    assert "list_my_tasks" in offered, "governed tools were not bound to the model"
    assert answer == "You have no open tasks."
