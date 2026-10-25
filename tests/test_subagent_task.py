"""Tests for the subagent Task tool (Claude Code Task semantics).

Covers: tool filtering per subagent type, isolated execution through the
real AgentLoop with a scripted LLM, summary-only return, and
registration/parent injection.
"""
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.base import Tool  # noqa: E402
from tools.subagent_task import (  # noqa: E402
    EXPLORE_TOOLS,
    SubagentTaskTool,
    register_subagent_tools,
    set_agent_for_subagent,
)


class FakeTool(Tool):
    name = "native_read"
    description = "Fake read tool"
    parameters = {"type": "object", "properties": {"path": {"type": "string"}}}
    requires_permission = False

    def __init__(self, result: str = "fake content"):
        self.result = result
        self.calls: list[dict] = []

    async def execute(self, **params) -> str:
        self.calls.append(params)
        return self.result


class FakeWriteTool(FakeTool):
    name = "terminal_execute"
    description = "Fake terminal tool"


class FakeTaskTool(FakeTool):
    name = "task"
    description = "Recursion bait"


class FakeAgent:
    def __init__(self, tools: dict[str, Any] | None = None, llm: Any = None):
        self.tools = tools or {}
        self.llm = llm
        self.backend_llm = None
        self.permissions = None


class ScriptedLLM:
    """Returns scripted chat_with_tools responses in order."""

    def __init__(self, responses: list[dict]):
        self.responses = list(responses)

    async def chat_with_tools(self, user_input="", system_prompt="",
                              conversation_history=None, tools=None):
        if not self.responses:
            return {"message": {"content": "done"}}
        return self.responses.pop(0)


def _tool_call_response(name: str, args: dict, call_id: str = "call_1") -> dict:
    return {"message": {
        "content": "",
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }],
    }}


def _final_response(text: str) -> dict:
    return {"message": {"content": text}}


@pytest.fixture(autouse=True)
def _reset_parent_ref():
    set_agent_for_subagent(None)
    yield
    set_agent_for_subagent(None)


class TestValidationAndSelection:
    def test_missing_agent_errors(self):
        tool = SubagentTaskTool(agent=None)
        output = asyncio.new_event_loop().run_until_complete(
            tool.execute(prompt="do something"))
        assert "not initialized" in output

    def test_empty_prompt_errors(self):
        agent = FakeAgent()
        tool = SubagentTaskTool(agent=agent)
        output = asyncio.new_event_loop().run_until_complete(tool.execute(prompt="  "))
        assert "requires a prompt" in output

    def test_explore_selects_readonly_tools(self):
        read, write, task = FakeTool(), FakeWriteTool(), FakeTaskTool()
        agent = FakeAgent(tools={"native_read": read, "terminal_execute": write,
                                 "task": task})
        tool = SubagentTaskTool(agent=agent)
        selected = tool._select_tools(agent, "explore")
        assert set(selected) == {"native_read"}
        assert "native_read" in EXPLORE_TOOLS

    def test_general_excludes_task_tool(self):
        read, write, task = FakeTool(), FakeWriteTool(), FakeTaskTool()
        agent = FakeAgent(tools={"native_read": read, "terminal_execute": write,
                                 "task": task})
        tool = SubagentTaskTool(agent=agent)
        selected = tool._select_tools(agent, "general")
        assert "task" not in selected
        assert set(selected) == {"native_read", "terminal_execute"}

    def test_needs_permission_never_blocks(self):
        assert SubagentTaskTool().needs_permission(prompt="x") is False


class TestIsolatedExecution:
    def test_tool_roundtrip_and_summary_only(self):
        read = FakeTool(result="file content here")
        llm = ScriptedLLM([
            _tool_call_response("native_read", {"path": "a.txt"}),
            _final_response("Report: a.txt contains file content here."),
        ])
        agent = FakeAgent(tools={"native_read": read}, llm=llm)
        tool = SubagentTaskTool(agent=agent)

        output = asyncio.new_event_loop().run_until_complete(
            tool.execute(prompt="read a.txt and report", type="explore"))

        assert output == "Report: a.txt contains file content here."
        assert read.calls == [{"path": "a.txt"}]

    def test_summary_excludes_intermediate_chatter(self):
        read = FakeTool()
        llm = ScriptedLLM([
            {"message": {"content": "Let me check the file...",
                         "tool_calls": [{
                             "id": "c1", "type": "function",
                             "function": {"name": "native_read",
                                          "arguments": "{\"path\": \"a.txt\"}"}}]}},
            _final_response("Final: 3 matches found."),
        ])
        agent = FakeAgent(tools={"native_read": read}, llm=llm)
        tool = SubagentTaskTool(agent=agent)

        output = asyncio.new_event_loop().run_until_complete(
            tool.execute(prompt="count matches"))
        assert output == "Final: 3 matches found."
        assert "Let me check" not in output

    def test_no_llm_errors(self):
        agent = FakeAgent(tools={"native_read": FakeTool()}, llm=None)
        tool = SubagentTaskTool(agent=agent)
        output = asyncio.new_event_loop().run_until_complete(
            tool.execute(prompt="x"))
        assert "no LLM client" in output

    def test_no_matching_tools_errors(self):
        agent = FakeAgent(tools={"terminal_execute": FakeWriteTool()},
                          llm=ScriptedLLM([]))
        tool = SubagentTaskTool(agent=agent)
        output = asyncio.new_event_loop().run_until_complete(
            tool.execute(prompt="x", type="explore"))
        assert "no tools available" in output

    def test_llm_failure_returns_error_string(self):
        class BrokenLLM:
            async def chat_with_tools(self, **kwargs):
                raise RuntimeError("backend down")

        agent = FakeAgent(tools={"native_read": FakeTool()}, llm=BrokenLLM())
        tool = SubagentTaskTool(agent=agent)
        output = asyncio.new_event_loop().run_until_complete(
            tool.execute(prompt="x"))
        assert "Subagent failed" in output


class TestRegistration:
    def test_register_sets_parent_and_registers(self):
        registered = []

        class Registry:
            def register_tools(self, tools):
                registered.extend(tools)

        agent = FakeAgent()
        count = register_subagent_tools(Registry(), agent)
        assert count == 1
        assert len(registered) == 1
        assert registered[0].name == "task"
        assert registered[0].agent is agent