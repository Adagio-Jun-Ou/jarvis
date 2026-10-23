"""Tests for the agent task list (Claude Code TodoWrite semantics).

Covers the store (single in_progress, status validation, bounds,
rendering, next_action), the update_tasks tool, and the completion-
evaluator gate that keeps the agentic loop working while tasks remain
unfinished.
"""
import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.task_list import (  # noqa: E402
    AgentTaskList, get_task_list, reset_task_list,
)


@pytest.fixture(autouse=True)
def _fresh_task_list():
    reset_task_list()
    yield
    reset_task_list()


class TestAgentTaskListStore:
    def test_update_replaces_list(self):
        tl = AgentTaskList()
        tl.update([{"content": "first", "status": "pending"}])
        tl.update([
            {"content": "a", "status": "completed"},
            {"content": "b", "status": "pending"},
        ])
        tasks = tl.get_tasks()
        assert [t.content for t in tasks] == ["a", "b"]
        assert tasks[0].status == "completed"

    def test_single_in_progress_enforced(self):
        tl = AgentTaskList()
        tl.update([
            {"content": "one", "status": "in_progress"},
            {"content": "two", "status": "in_progress"},
        ])
        statuses = [t.status for t in tl.get_tasks()]
        assert statuses == ["in_progress", "pending"]

    def test_invalid_status_becomes_pending(self):
        tl = AgentTaskList()
        tl.update([{"content": "x", "status": "banana"}])
        assert tl.get_tasks()[0].status == "pending"

    def test_empty_content_skipped(self):
        tl = AgentTaskList()
        tl.update([
            {"content": "   ", "status": "pending"},
            {"content": "keep", "status": "pending"},
        ])
        assert [t.content for t in tl.get_tasks()] == ["keep"]

    def test_bounds_never_drop_active_work(self):
        tl = AgentTaskList()
        raw = [{"content": f"t{i}", "status": "completed"} for i in range(30)]
        raw.append({"content": "active", "status": "in_progress"})
        tl.update(raw)
        tasks = tl.get_tasks()
        assert len(tasks) <= AgentTaskList.MAX_TASKS
        assert any(t.content == "active" for t in tasks), "active task was evicted"

    def test_next_action_prefers_in_progress(self):
        tl = AgentTaskList()
        tl.update([
            {"content": "pending first", "status": "pending"},
            {"content": "current", "status": "in_progress"},
        ])
        assert tl.next_action() == "current"

    def test_next_action_none_when_all_terminal(self):
        tl = AgentTaskList()
        tl.update([
            {"content": "done", "status": "completed"},
            {"content": "dropped", "status": "cancelled"},
        ])
        assert tl.next_action() is None
        assert tl.has_unfinished() is False

    def test_to_prompt_renders_markers_and_directive(self):
        tl = AgentTaskList()
        tl.update([
            {"content": "write tests", "status": "completed"},
            {"content": "run suite", "status": "in_progress"},
            {"content": "commit", "status": "pending"},
        ])
        rendered = tl.to_prompt()
        assert "[✓] write tests" in rendered
        assert "[►] run suite" in rendered
        assert "[ ] commit" in rendered
        assert "2 remaining" in rendered
        assert "update_tasks" in rendered

    def test_to_prompt_empty_without_list(self):
        assert AgentTaskList().to_prompt() == ""

    def test_clear(self):
        tl = AgentTaskList()
        tl.update([{"content": "x", "status": "pending"}])
        tl.clear()
        assert tl.get_tasks() == []
        assert tl.to_prompt() == ""

    def test_singleton(self):
        first = get_task_list()
        assert get_task_list() is first
        reset_task_list()
        assert get_task_list() is not first


class TestUpdateTasksTool:
    def test_tool_updates_and_summarizes(self):
        from tools.task_list import UpdateTasksTool

        tool = UpdateTasksTool()
        assert tool.name == "update_tasks"
        result = asyncio.run(tool.execute(tasks=[
            {"content": "step one", "status": "in_progress"},
            {"content": "step two", "status": "pending"},
        ]))
        assert "2 remaining" in result
        assert "[►] step one" in result

    def test_tool_empty_clears(self):
        from tools.task_list import UpdateTasksTool

        tool = UpdateTasksTool()
        asyncio.run(tool.execute(tasks=[
            {"content": "x", "status": "pending"},
        ]))
        result = asyncio.run(tool.execute(tasks=[]))
        assert "cleared" in result.lower()
        assert get_task_list().get_tasks() == []


class TestCompletionEvaluatorGate:
    def _agent_with_backend(self, reply: str):
        from core.agent import JarvisAgent

        agent = JarvisAgent.__new__(JarvisAgent)
        agent._recursive_self_model = None
        agent.backend_llm = AsyncMock()
        agent.backend_llm.chat = AsyncMock(return_value={
            "message": {"role": "assistant", "content": reply},
        })
        return agent

    def test_unfinished_tasks_block_complete(self):
        agent = self._agent_with_backend("COMPLETE")
        get_task_list().update([
            {"content": "finish the migration", "status": "in_progress"},
        ])
        is_complete, next_action = asyncio.new_event_loop().run_until_complete(
            agent._evaluate_task_completion("do work", "partial results")
        )
        assert is_complete is False
        assert "finish the migration" in next_action
        # The backend evaluator is not even consulted while tasks remain
        agent.backend_llm.chat.assert_not_called()

    def test_all_tasks_done_restores_normal_evaluation(self):
        agent = self._agent_with_backend("COMPLETE")
        get_task_list().update([
            {"content": "done thing", "status": "completed"},
        ])
        is_complete, _ = asyncio.new_event_loop().run_until_complete(
            agent._evaluate_task_completion("do work", "results ready")
        )
        assert is_complete is True
        agent.backend_llm.chat.assert_called_once()

    def test_no_task_list_is_inert(self):
        agent = self._agent_with_backend("COMPLETE")
        is_complete, _ = asyncio.new_event_loop().run_until_complete(
            agent._evaluate_task_completion("simple question", "answer ready")
        )
        assert is_complete is True