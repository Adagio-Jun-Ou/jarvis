"""
ReactLoop — the single-turn orchestrator the VM delegates to.

Responsibilities:

1. Build an Omnigent :class:`AgentDef` for this turn from the
   :class:`agent.tools.registry.TOOL_REGISTRY`.
2. Attach the VM's :class:`ConstitutionAdapter` to the agent's
   ``policies`` so every tool call goes through the constitution.
3. Drive the executor (Ollama or Databricks) through one
   ReAct turn using :func:`agent.omnigent._alex_bridge.run_turn`.
4. Translate every :class:`ExecutorEvent` to an
   :class:`alex.events.Event` and emit it through the VM's
   :class:`EventPump` so SSE subscribers and the audit log both
   see the same stream.

The :class:`ReactLoop` is intentionally small.  All the real work
happens in the executor, the bridge, the constitution adapter, and
the event pump.  This module is the wiring.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from agent.omnigent import (
    AgentDef,
    Executor,
    ToolCallRequest,
)
from agent.omnigent._alex_bridge import executor_event_to_agent_event, run_turn

from .constitution_adapter import ConstitutionAdapter
from .event_pump import EventPump

logger = logging.getLogger("alex.harness.react_loop")


class ReactLoop:
    """One ReAct turn. Construct fresh per turn; the VM owns the
    long-lived executor + event pump + session.
    """

    def __init__(
        self,
        *,
        executor: Executor,
        agent_def: AgentDef,
        history: list[dict[str, Any]],
        system_prompt: str | None,
        pump: EventPump,
        constitution_adapter: ConstitutionAdapter,
        poll_steering: Callable[[], list[tuple[str, str]]] | None = None,
        on_llm_pass: Callable[..., None] | None = None,
    ) -> None:
        self._executor = executor
        self._agent_def = agent_def
        self._history = history
        self._system_prompt = system_prompt
        self._pump = pump
        self._constitution_adapter = constitution_adapter
        # Mid-run steering: drain the VM's mailbox between ReAct
        # iterations and inject the notes as user messages, so a human
        # on a task crew can course-correct without stopping the run.
        self._poll_steering = poll_steering
        # Governor accounting: called once per executor pass so
        # llm_turns is a real number instead of a permanent zero.
        self._on_llm_pass = on_llm_pass

    def run(self, max_turns: int = 8) -> tuple[str, list[dict[str, Any]]]:
        """Drive one ReAct turn.  Returns ``(final_response, history)``.

        Synchronous wrapper around the async bridge so the VM can be
        driven from anywhere: a FastAPI route on the event loop, a
        task-board worker thread, a standing-order run, or a tool call
        nested inside a turn.  :func:`run_coroutine` picks the right
        strategy for the calling thread — a bare ``asyncio.run`` here
        raised ``cannot be called from a running event loop`` and left
        the turn unawaited whenever a loop was already spinning.
        """
        from agent.harness.sync_bridge import run_coroutine

        return run_coroutine(self._arun(max_turns))

    async def _arun(self, max_turns: int) -> tuple[str, list[dict[str, Any]]]:
        """The async body.  The bridge iterates the ReAct loop
        internally (tool call → tool result → next LLM round) up to
        ``max_turns`` rounds and appends the assistant's final
        response to the history, so a single bridge call is one
        complete turn.
        """
        sys_prompt = self._system_prompt or self._agent_def.prompt or ""
        last, _ = await run_turn(
            executor=self._executor,
            agent_def=self._agent_def,
            history=self._history,
            system_prompt=sys_prompt,
            on_tool_call=self._dispatch_tool_call,
            emit=lambda ev: self._pump.emit(ev),
            max_iterations=max_turns,
            poll_steering=self._poll_steering,
            on_llm_pass=self._on_llm_pass,
        )
        return last, self._history

    def _dispatch_tool_call(self, request: ToolCallRequest) -> Any:
        """Run a single tool call through the registry.  The VM
        owns the actual invocation (gate + audit + event-pump emit)
        so this method is a thin pass-through.
        """
        vm = getattr(self, "_vm", None)
        if vm is None:
            raise RuntimeError("ReactLoop has no VM; set self._vm before run()")
        return vm._invoke_tool(request)


__all__ = ["ReactLoop", "executor_event_to_agent_event"]
