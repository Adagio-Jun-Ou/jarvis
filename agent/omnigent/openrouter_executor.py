"""OpenRouterExecutor — the local-machine LLM executor.

One ``Executor`` per backend, same as the others: this one drives
:class:`alex.llm.OpenRouterLLM` (OpenAI wire, real token streaming,
native tool calls) and converts each :class:`alex.llm.LLMDelta` into
the executor events the bridge understands. Message flattening and
tool-schema conversion are shared with the other executors so a run's
history replays identically whichever backend answers.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from .async_utils import aiter_any
from .databricks_executor import _convert_messages, _convert_tools_to_openai
from .executor import (
    Executor,
    ExecutorConfig,
    ExecutorError,
    PassUsage,
    TextChunk,
    ToolCallRequest,
    TurnComplete,
)

logger = logging.getLogger(__name__)


def _get_openrouter_llm():
    from alex.llm import OpenRouterLLM

    return OpenRouterLLM()


class OpenRouterExecutor(Executor):
    """LLM executor backed by OpenRouter (OpenAI-compatible API)."""

    def __init__(self, llm: Any | None = None) -> None:
        self._llm = llm if llm is not None else _get_openrouter_llm()

    def supports_streaming(self) -> bool:
        return True

    def supports_tool_calling(self) -> bool:
        return True

    async def run_turn(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system_prompt: str,
        config: ExecutorConfig | None = None,
    ) -> AsyncIterator[Any]:
        cfg = config or ExecutorConfig()
        if cfg.model:
            model = cfg.model
        else:
            from alex.config import get_settings

            model = get_settings().default_llm_endpoint
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": _convert_messages(messages, system_prompt),
            "max_tokens": cfg.max_tokens,
            "temperature": cfg.temperature,
        }
        oai_tools = _convert_tools_to_openai(tools) if tools else None
        if oai_tools:
            kwargs["tools"] = oai_tools
        try:
            started = time.monotonic()
            async for delta in aiter_any(self._llm.stream(**kwargs)):
                if delta.text:
                    yield TextChunk(text=delta.text)
                if delta.tool_call:
                    tc = delta.tool_call
                    yield ToolCallRequest(
                        name=getattr(tc, "name", ""),
                        args=getattr(tc, "args", {}) or {},
                    )
                if delta.finish_reason == "tool_calls":
                    if getattr(delta, "usage", None):
                        yield PassUsage(usage=dict(delta.usage))
                    return
                if delta.finish_reason == "stop":
                    if getattr(delta, "usage", None):
                        yield PassUsage(usage=dict(delta.usage))
                    yield TurnComplete(
                        response=delta.full_text or "",
                        usage=dict(getattr(delta, "usage", None) or {}) or None,
                    )
                    return
            duration = (time.monotonic() - started) * 1000.0
            logger.warning(
                "OpenRouterExecutor: stream ended without finish_reason after %.0fms",
                duration,
            )
            yield ExecutorError(message="Stream ended without finish_reason")
        except Exception as exc:  # noqa: BLE001
            logger.error("OpenRouterExecutor: stream failed: %s", exc)
            yield ExecutorError(message=f"LLM stream error: {exc}")
