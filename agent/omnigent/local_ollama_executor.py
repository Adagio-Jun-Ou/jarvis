"""LocalOllamaExecutor — dev LLM executor backed by Ollama Cloud.

Adapted from the upstream ``omnigent`` pattern of having one
``Executor`` subclass per backend.  Alex 2.0 ships two executors:

* :class:`DatabricksExecutor` for production
  (Databricks Model Serving via :mod:`alex.llm`).
* :class:`LocalOllamaExecutor` for dev/test (Ollama Cloud via
  :mod:`alex.llm`).

Both classes expose the same ``run_turn`` interface and yield the
same :class:`ExecutorEvent` types, so :mod:`agent.omnigent._alex_bridge`
doesn't care which one is in play.  The dev executor is the default
in the conftest fixtures and in ``app/_dev_seed.py`` so the rest of
the agent can be exercised end-to-end on a laptop without a real
Databricks workspace.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from .async_utils import aiter_any
from .databricks_executor import _convert_messages
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


def _get_ollama_llm():
    """Return an :class:`OllamaCloudLLM` instance (dev fallback)."""
    from alex.llm import OllamaCloudLLM

    return OllamaCloudLLM()


class LocalOllamaExecutor(Executor):
    """LLM executor backed by :class:`alex.llm.OllamaCloudLLM`.

    In dev, the agent runs against Ollama Cloud (the same key Alex 2.0
    was originally wired to).  The executor converts :class:`LLMDelta`
    events from :meth:`OllamaCloudLLM.stream` into
    :class:`ExecutorEvent` instances.  Tool calling is supported when
    the upstream Ollama model advertises it; otherwise the agent
    degrades to a single-turn :class:`TurnComplete`.
    """

    def __init__(self, llm: Any | None = None) -> None:
        self._llm = llm if llm is not None else _get_ollama_llm()

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
        oai_messages = _to_oai_messages(messages, system_prompt)
        oai_tools = _to_oai_tools(tools) if tools else None
        kwargs: dict[str, Any] = {
            "model": cfg.model or None,
            "messages": oai_messages,
            "max_tokens": cfg.max_tokens,
            "temperature": cfg.temperature,
        }
        if oai_tools:
            kwargs["tools"] = oai_tools
        try:
            started = time.monotonic()
            async for delta in aiter_any(self._llm.stream(**kwargs)):
                if delta.text:
                    yield TextChunk(text=delta.text)
                if delta.tool_call:
                    tc = delta.tool_call
                    if isinstance(tc, dict):
                        name = tc.get("name", "")
                        args = tc.get("arguments", {}) or tc.get("args", {})
                    else:
                        name = getattr(tc, "name", "")
                        args = getattr(tc, "args", {}) or {}
                    yield ToolCallRequest(name=name, args=args)
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
                "LocalOllamaExecutor: stream ended without finish_reason after %.0fms",
                duration,
            )
            yield ExecutorError(message="Stream ended without finish_reason")
        except Exception as exc:  # noqa: BLE001
            logger.error("LocalOllamaExecutor: stream failed: %s", exc)
            yield ExecutorError(message=f"LLM stream error: {exc}")


def _to_oai_messages(
    messages: list[dict[str, Any]], system_prompt: str
) -> list[dict[str, Any]]:
    """Convert bridge-shaped history to OpenAI chat messages.

    Delegates to :func:`agent.omnigent.databricks_executor._convert_messages`
    so tool-call linkage is preserved: a ``tool_call`` entry becomes an
    assistant message with ``tool_calls`` and the following
    ``tool_result`` becomes a ``tool`` message with the matching
    ``tool_call_id``, instead of an orphaned ``tool`` message.
    """
    out = _convert_messages(messages, system_prompt)
    # Ollama difference vs the OpenAI wire format: in REQUEST messages
    # Ollama expects assistant tool_calls `arguments` as a JSON OBJECT,
    # not a serialized string — a string makes /api/chat 400 with
    # "Value looks like object, but can't find closing '}' symbol".
    for m in out:
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    fn["arguments"] = json.loads(args) if args else {}
                except json.JSONDecodeError:
                    fn["arguments"] = {}
    return out


def _to_oai_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for t in tools:
        fn = {
            "name": t.get("name", ""),
            "description": t.get("description", ""),
            "parameters": t.get("parameters", {"type": "object", "properties": {}}),
        }
        out.append({"type": "function", "function": fn})
    return out
