"""Compact Context tool — model-triggered conversation compaction.

Claude Code's /compact, exposed as a tool: when the agent notices the
conversation has grown long (or the user asks it to compact), it calls
this to replace older messages with a structured summary while keeping
recent context verbatim. Frees the context window without losing facts
(the summary is also stored in long-term memory).
"""
import logging
import threading
from typing import Any

from tools.base import Tool

logger = logging.getLogger(__name__)

_agent_ref: Any = None
_agent_ref_lock = threading.Lock()


def set_agent_for_compact(agent: Any) -> None:
    """Inject the parent agent (memory access) for compaction."""
    global _agent_ref
    with _agent_ref_lock:
        _agent_ref = agent


def get_compact_parent() -> Any:
    with _agent_ref_lock:
        return _agent_ref


class CompactContextTool(Tool):
    name = "compact_context"
    description = (
        "Compact the conversation context: replaces older messages with "
        "a structured summary (recent messages are kept verbatim). Use "
        "when the conversation is long, context feels cluttered, or a "
        "fresh sub-task does not need the full history. The summary is "
        "preserved in long-term memory."
    )
    parameters = {
        "type": "object",
        "properties": {
            "keep_recent": {
                "type": "integer",
                "description": "Messages to keep verbatim (default 10, max 40).",
            },
        },
    }
    requires_permission = False

    def __init__(self, agent: Any = None):
        self._agent = agent

    @property
    def agent(self) -> Any:
        return self._agent or get_compact_parent()

    async def execute(self, keep_recent: int = 10, **kwargs) -> str:
        agent = self.agent
        if agent is None:
            return "Error: compaction infrastructure not initialized."
        memory = getattr(agent, "memory", None)
        if memory is None or not hasattr(memory, "compact"):
            return "Error: conversation memory does not support compaction."

        keep = max(2, min(int(keep_recent or 10), 40))
        try:
            stats = memory.compact(keep_recent=keep)
        except Exception as e:
            from core.helpers import safe_error_message
            logger.error(f"Compaction failed: {e}")
            return f"Compaction failed: {safe_error_message(e)}"

        if not stats.get("compacted"):
            return f"Nothing to compact: {stats.get('reason', 'history is small')}."
        return (
            f"Context compacted: {stats['messages_removed']} old messages "
            f"replaced with a summary ({stats['tokens_before']} -> "
            f"{stats['tokens_after']} tokens). "
            f"Summary: {stats['summary'][:500]}"
        )


TOOLS = [CompactContextTool()]


def register_compact_tools(registry, agent: Any) -> int:
    """Set the parent agent and register the compact_context tool."""
    set_agent_for_compact(agent)
    tool = CompactContextTool(agent=agent)
    if hasattr(registry, "register_tools"):
        registry.register_tools([tool])
    else:
        registry.register(tool)
    logger.info("Registered compact_context tool")
    return 1


__all__ = [
    "CompactContextTool",
    "set_agent_for_compact",
    "get_compact_parent",
    "register_compact_tools",
    "TOOLS",
]
