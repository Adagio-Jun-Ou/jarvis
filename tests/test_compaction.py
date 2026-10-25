"""Tests for model-triggered compaction (Stage I).

Covers: ConversationMemory.compact() (summary replacement, recent
verbatim, long-term storage, small-history no-op, fallback path) and
the compact_context tool.
"""
import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.memory import ConversationMemory  # noqa: E402
from tools.compact_context import (  # noqa: E402
    CompactContextTool,
    register_compact_tools,
    set_agent_for_compact,
)


@pytest.fixture(autouse=True)
def _reset_ref():
    set_agent_for_compact(None)
    yield
    set_agent_for_compact(None)


def _fill(memory: ConversationMemory, pairs: int) -> None:
    for i in range(pairs):
        memory.add_user_message(f"user message number {i} with some content")
        memory.add_assistant_message(f"assistant reply number {i}")


class TestCompaction:
    def test_small_history_is_noop(self):
        memory = ConversationMemory(use_vector_memory=False)
        _fill(memory, 3)
        stats = memory.compact(keep_recent=10)
        assert stats["compacted"] is False
        assert len(memory.history) == 6

    def test_compaction_replaces_old_with_summary(self):
        memory = ConversationMemory(use_vector_memory=False)
        _fill(memory, 15)
        total_before = len(memory.history)

        stats = memory.compact(keep_recent=6)

        assert stats["compacted"] is True
        assert stats["messages_removed"] == total_before - 6
        assert stats["tokens_after"] <= stats["tokens_before"]
        # summary message + kept recent
        assert len(memory.history) == 7
        assert "compacted" in memory.history[0]["content"].lower()
        assert "assistant reply number 14" in memory.history[-1]["content"]

    def test_summary_stored_in_long_term(self):
        memory = ConversationMemory(use_vector_memory=False)
        _fill(memory, 15)
        memory.compact(keep_recent=6)
        assert memory.long_term_memory
        assert memory.long_term_memory[-1]["importance"] == 0.8

    def test_token_count_recomputed(self):
        memory = ConversationMemory(use_vector_memory=False)
        _fill(memory, 15)
        memory.compact(keep_recent=6)
        recomputed = sum(
            memory._estimate_tokens(m.get("content", ""))
            for m in memory.history
        )
        assert memory._total_tokens == recomputed

    def test_fallback_when_summarizer_unavailable(self, monkeypatch):
        memory = ConversationMemory(use_vector_memory=False)
        _fill(memory, 15)

        import core.conversation_summarizer as cs
        def broken():
            raise RuntimeError("summarizer down")
        monkeypatch.setattr(cs, "get_conversation_summarizer", broken)

        stats = memory.compact(keep_recent=6)
        assert stats["compacted"] is True
        assert "User:" in stats["summary"] or "Assistant:" in stats["summary"]

    def test_get_messages_for_llm_after_compact(self):
        memory = ConversationMemory(use_vector_memory=False)
        _fill(memory, 15)
        memory.compact(keep_recent=4)
        messages = memory.get_messages_for_llm()
        assert len(messages) == 5
        assert "compact" in messages[0]["content"].lower()


class TestCompactTool:
    def test_tool_compacts_memory(self):
        memory = ConversationMemory(use_vector_memory=False)
        _fill(memory, 15)

        class FakeAgent:
            pass

        agent = FakeAgent()
        agent.memory = memory
        tool = CompactContextTool(agent=agent)

        output = asyncio.new_event_loop().run_until_complete(
            tool.execute(keep_recent=6))
        assert "Context compacted" in output
        assert "24 old messages" in output

    def test_tool_noop_message(self):
        memory = ConversationMemory(use_vector_memory=False)
        _fill(memory, 2)
        agent = type("A", (), {"memory": memory})()
        tool = CompactContextTool(agent=agent)
        output = asyncio.new_event_loop().run_until_complete(tool.execute())
        assert "Nothing to compact" in output

    def test_tool_missing_agent(self):
        tool = CompactContextTool(agent=None)
        output = asyncio.new_event_loop().run_until_complete(tool.execute())
        assert "not initialized" in output

    def test_tool_memory_without_compact(self):
        agent = type("A", (), {"memory": object()})()
        tool = CompactContextTool(agent=agent)
        output = asyncio.new_event_loop().run_until_complete(tool.execute())
        assert "does not support compaction" in output

    def test_register_sets_parent(self):
        registered = []

        class Registry:
            def register_tools(self, tools):
                registered.extend(tools)

        agent = type("A", (), {"memory": ConversationMemory(use_vector_memory=False)})()
        count = register_compact_tools(Registry(), agent)
        assert count == 1
        assert registered[0].name == "compact_context"
        assert registered[0].agent is agent


class TestAgentLoopWiring:
    def test_summarize_if_needed_called_between_steps(self):
        """AgentLoop.execute must compact when the message list grows."""
        import json
        from core.agent_loop import AgentLoop

        class ReadTool:
            name = "native_read"
            description = "read"
            parameters = {"type": "object", "properties": {}}
            requires_permission = False

            async def execute(self, **kwargs):
                return "x"

        class LoopLLM:
            def __init__(self):
                self.turns = 0

            async def chat_with_tools(self, user_input="", system_prompt="",
                                      conversation_history=None, tools=None):
                self.turns += 1
                if self.turns <= 3:
                    return {"message": {
                        "content": "",
                        "tool_calls": [{
                            "id": f"c{self.turns}", "type": "function",
                            "function": {"name": "native_read",
                                         "arguments": json.dumps({"path": "a"})},
                        }],
                    }}
                return {"message": {"content": "done"}}

        llm = LoopLLM()
        loop = AgentLoop(llm_client=llm, tool_registry={"native_read": ReadTool()},
                         max_steps=10)

        summarized = {"count": 0}
        original = loop.summarize_if_needed

        async def tracking(messages, **kwargs):
            summarized["count"] += 1
            return await original(messages, max_messages=2, keep_recent=1)

        loop.summarize_if_needed = tracking

        result = asyncio.new_event_loop().run_until_complete(
            loop.execute(messages=[{"role": "user", "content": "go"}],
                         system_prompt="sys"))
        assert summarized["count"] >= 2
        assert result.total_tool_calls == 3