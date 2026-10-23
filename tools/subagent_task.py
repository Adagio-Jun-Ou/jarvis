"""Subagent Task tool — Claude Code Task semantics for Jarvis.

Spawns an isolated agent loop that receives ONLY the given prompt, runs
its own tool-calling loop with a restricted tool set, and returns a
single final summary. The parent conversation stays clean: intermediate
tool chatter never enters the main context (the core value of the
Claude Code Task tool — context isolation).

Subagent types:
- ``explore``: read-only tools. Safe for reconnaissance.
- ``general``: the same tools as the parent (minus ``task`` itself, to
  prevent unbounded recursion), inheriting the parent's permission
  manager so destructive actions still gate.

The subagent runs on the parent's LLM client (or backend LLM when
available) via the existing AgentLoop runtime — no duplicate executor.
"""
import logging
import threading
from typing import Any, Optional

from tools.base import Tool

logger = logging.getLogger(__name__)

# Claude Code-style tool presets.
EXPLORE_TOOLS = (
    "native_read", "native_list", "read_file", "list_files",
    "glob", "grep", "search_files", "web_search", "web_fetch",
    "system_sense", "check_jobs",
)
GENERAL_EXCLUDE = ("task",)

_SUBAGENT_SYSTEM = (
    "You are a subagent of Jarvis, working on a single focused task in "
    "isolation. You have tools available. Investigate autonomously and "
    "finish with a concise, complete report of findings or results — the "
    "report is the ONLY thing returned to the caller, so include every "
    "fact it needs and no process narration. Do not ask questions; make "
    "reasonable assumptions and state them."
)

_agent_ref: Any = None
_agent_ref_lock = threading.Lock()


def set_agent_for_subagent(agent: Any) -> None:
    """Inject the parent agent (LLM, tools, permissions) for task runs."""
    global _agent_ref
    with _agent_ref_lock:
        _agent_ref = agent


def get_subagent_parent() -> Any:
    with _agent_ref_lock:
        return _agent_ref


class SubagentTaskTool(Tool):
    name = "task"
    description = (
        "Spawn an isolated subagent for a focused multi-step task "
        "(reconnaissance, searching, analysis). The subagent has its own "
        "context and returns only a final summary, keeping the main "
        "conversation clean. Use type='explore' for read-only "
        "investigation, 'general' for tasks that may act. Include "
        "everything the subagent needs in the prompt — it cannot see "
        "this conversation."
    )
    parameters = {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Complete, self-contained task description for the subagent.",
            },
            "type": {
                "type": "string",
                "enum": ["explore", "general"],
                "description": "Subagent type: explore (read-only) or general.",
            },
            "max_steps": {
                "type": "integer",
                "description": "Maximum tool-calling steps (default 10, max 30).",
            },
        },
        "required": ["prompt"],
    }
    requires_permission = False

    def __init__(self, agent: Any = None):
        self._agent = agent

    @property
    def agent(self) -> Any:
        return self._agent or get_subagent_parent()

    async def execute(self, prompt: str = "", type: str = "explore",
                      max_steps: int = 10, **kwargs) -> str:
        if not prompt or not prompt.strip():
            return "Error: task requires a prompt."
        agent = self.agent
        if agent is None:
            return "Error: subagent infrastructure not initialized."

        steps = max(1, min(int(max_steps or 10), 30))
        sub_type = (type or "explore").lower()

        llm = getattr(agent, "backend_llm", None) or getattr(agent, "llm", None)
        if llm is None:
            return "Error: no LLM client available for subagent."

        tools = self._select_tools(agent, sub_type)
        if not tools:
            return "Error: no tools available for subagent."

        from core.agent_loop import AgentLoop
        from core.helpers import safe_error_message

        loop = AgentLoop(
            llm_client=llm,
            tool_registry=tools,
            permission_manager=getattr(agent, "permissions", None),
            max_steps=steps,
            show_thinking=False,
            show_tool_calls=False,
            auto_approve_tools=False,
        )

        try:
            result = await loop.execute(
                messages=[{"role": "user", "content": prompt}],
                system_prompt=_SUBAGENT_SYSTEM,
                tools=list(tools.keys()),
            )
        except Exception as e:
            logger.error(f"Subagent task failed: {e}")
            return f"Subagent failed: {safe_error_message(e)}"

        from core.agent_loop import FinishReason
        if getattr(result, "finish_reason", None) == FinishReason.ERROR:
            error_text = self._extract_error(result)
            logger.warning(f"Subagent ({sub_type}) ran into an error: {error_text}")
            return f"Subagent failed: {error_text}"

        summary = self._extract_summary(result)
        tool_calls = getattr(result, "total_tool_calls", 0)
        logger.info(
            f"Subagent ({sub_type}) finished: {tool_calls} tool calls, "
            f"{len(result.steps)} steps, summary={len(summary)} chars"
        )
        return summary or "Subagent finished without producing a report."

    def _select_tools(self, agent: Any, sub_type: str) -> dict[str, Any]:
        available = getattr(agent, "tools", {}) or {}
        if sub_type == "general":
            return {
                name: tool for name, tool in available.items()
                if name not in GENERAL_EXCLUDE
            }
        return {
            name: tool for name, tool in available.items()
            if name in EXPLORE_TOOLS
        }

    def _extract_summary(self, result: Any) -> str:
        """Final assistant text across steps (last non-empty wins)."""
        for step in reversed(getattr(result, "steps", [])):
            msg = getattr(step, "assistant_message", None)
            text = getattr(msg, "text", "") if msg else ""
            if text and text.strip():
                return text.strip()
        return ""

    def _extract_error(self, result: Any) -> str:
        """Pull the error text AgentLoop stores on an ERROR finish."""
        for step in reversed(getattr(result, "steps", [])):
            msg = getattr(step, "assistant_message", None)
            text = getattr(msg, "text", "") if msg else ""
            if text and text.strip():
                return text.strip().strip("[]").removeprefix("Error: ").strip()
        return "agent loop error"

    def needs_permission(self, **kwargs) -> bool:
        # Reconnaissance is safe; general inherits the parent's gate per
        # tool inside AgentLoop, so the wrapper itself never blocks.
        return False


TOOLS = [SubagentTaskTool()]


def register_subagent_tools(registry, agent: Any) -> int:
    """Set the parent agent and register the task tool."""
    set_agent_for_subagent(agent)
    tool = SubagentTaskTool(agent=agent)
    if hasattr(registry, "register_tools"):
        registry.register_tools([tool])
    else:
        registry.register(tool)
    logger.info("Registered subagent task tool")
    return 1


__all__ = [
    "SubagentTaskTool",
    "set_agent_for_subagent",
    "get_subagent_parent",
    "register_subagent_tools",
    "EXPLORE_TOOLS",
    "TOOLS",
]
