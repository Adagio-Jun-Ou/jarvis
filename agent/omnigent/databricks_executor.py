"""DatabricksExecutor — LLM execution via the Databricks FM API.

Adapted from ``omnigent/inner/databricks_executor.py``
(https://github.com/omnigent-ai/omnigent, Apache-2.0).  The
adaptations:

* **Replaced** the OpenAI SDK + ``~/.databrickscfg`` + ``databricks-sdk``
  credential chain with a call into
  :class:`alex.llm.DatabricksServingLLM`.  Alex 2.0 already maintains
  the canonical LLM callable (single key, single base URL, single
  retry policy), so the executor delegates to it and the executor
  inherits the same retry / timeout / cost-tracking behaviour as the
  hand-rolled ``agent/agent.py`` loop it replaces.
* **Removed** ``CompactionComplete``-related session-state management
  (we persist our own history through
  :mod:`agent.session_manager`).
* **Kept** the streaming ``TextChunk`` / ``ToolCallRequest`` /
  ``TurnComplete`` event shape so :mod:`agent.omnigent._alex_bridge`
  can translate :class:`ExecutorEvent` instances into our
  ``alex.events`` SSE protocol unchanged.

Streaming support is provided by :class:`alex.llm.DatabricksServingLLM`,
which yields ``LLMDelta`` objects as OpenAI streaming chunks arrive.
The executor converts them into Omnigent's ``TextChunk`` /
``ToolCallRequest`` / ``TurnComplete`` event types.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from .async_utils import aiter_any
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


def _get_databricks_llm():
    """Return a :class:`DatabricksServingLLM` instance.

    Lazy import so the rest of the package is importable in
    environments where ``databricks-sdk`` is not installed (e.g. a
    unit test that uses :class:`MockExecutor`).
    """
    from alex.llm import DatabricksServingLLM

    return DatabricksServingLLM()


def _get_ollama_llm():
    """Return an :class:`OllamaCloudLLM` instance (dev fallback)."""
    from alex.llm import OllamaCloudLLM

    return OllamaCloudLLM()


class DatabricksExecutor(Executor):
    """LLM executor backed by Databricks Model Serving.

    The executor streams from
    :meth:`alex.llm.DatabricksServingLLM.stream` and converts each
    :class:`alex.llm.LLMDelta` into an :class:`ExecutorEvent`.
    Tool-call deltas are accumulated across chunks and emitted as a
    single :class:`ToolCallRequest` once the stream signals a
    ``finish_reason="tool_calls"`` (Databricks OpenAI-compatible API
    follows the OpenAI streaming tool-call convention).
    """

    def __init__(self, llm: Any | None = None) -> None:
        self._llm = llm if llm is not None else _get_databricks_llm()

    def supports_streaming(self) -> bool:
        return True

    def supports_tool_calling(self) -> bool:
        return True

    def _session_key(self, messages: list[dict[str, Any]]) -> str:
        if messages:
            sid = messages[-1].get("session_id")
            if sid:
                return str(sid)
            meta = messages[-1].get("metadata", {})
            if isinstance(meta, dict) and meta.get("session_id"):
                return str(meta["session_id"])
        return "default"

    async def run_turn(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system_prompt: str,
        config: ExecutorConfig | None = None,
    ) -> AsyncIterator[Any]:
        cfg = config or ExecutorConfig()
        # The endpoint name comes from config, not a literal: the old
        # hard-coded "databricks-claude-sonnet-4-6" silently bypassed
        # Settings.default_llm_endpoint (and named an endpoint the
        # workspace does not have), so changing the configured model
        # changed nothing about what the executor actually called.
        if cfg.model:
            model = cfg.model
        else:
            from alex.config import get_settings

            model = get_settings().default_llm_endpoint
        oai_messages = _convert_messages(messages, system_prompt)
        oai_tools = _convert_tools_to_openai(tools) if tools else None
        kwargs: dict[str, Any] = {
            "model": model,
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
            # Stream ended without an explicit finish_reason; surface what
            # we have but warn — a missing finish_reason means the turn
            # may be incomplete.
            duration = (time.monotonic() - started) * 1000.0
            logger.warning(
                "DatabricksExecutor: stream ended without finish_reason after %.0fms",
                duration,
            )
            yield ExecutorError(message="Stream ended without finish_reason")
        except Exception as exc:  # noqa: BLE001
            logger.error("DatabricksExecutor: stream failed: %s", exc)
            yield ExecutorError(message=f"LLM stream error: {exc}")


def _convert_tools_to_openai(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for tool in tools:
        raw_name = tool.get("name")
        raw_description = tool.get("description")
        fn: dict[str, Any] = {
            "name": raw_name if isinstance(raw_name, str) else "",
            "description": raw_description if isinstance(raw_description, str) else "",
        }
        if "parameters" in tool:
            fn["parameters"] = tool["parameters"]
        else:
            fn["parameters"] = {"type": "object", "properties": {}}
        out.append({"type": "function", "function": fn})
    return out


def _convert_messages(
    messages: list[dict[str, Any]],
    system_prompt: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if system_prompt:
        out.append({"role": "system", "content": system_prompt})

    i = 0
    while i < len(messages):
        msg = messages[i]
        raw_role = msg.get("role")
        role = raw_role if isinstance(raw_role, str) else ""
        content = msg.get("content") if msg.get("content") is not None else ""

        if role == "user":
            out.append(
                {"role": "user", "content": str(content) if content else "(empty)"}
            )
        elif role == "assistant":
            out.append(
                {"role": "assistant", "content": str(content) if content else "(empty)"}
            )
        elif role == "tool_call":
            # Represent a run of tool_call/tool_result pairs as PLAIN
            # CONVERSATIONAL CONTEXT (one assistant note + one user
            # results block) rather than the strict OpenAI
            # tool_calls/tool-role protocol.  Empirically the protocol
            # form makes some served models (e.g. Kimi via Ollama)
            # return an empty synthesis turn far more often; plain-text
            # tool context is understood by every model and is much more
            # reliable.  The model still emits tool calls via the
            # protocol on the way *out* — this only affects how prior
            # turns are replayed back to it.
            names: list[str] = []
            results: list[str] = []
            while i < len(messages) and messages[i].get("role") == "tool_call":
                c = messages[i].get("content")
                if isinstance(c, str):
                    try:
                        c = json.loads(c)
                    except (json.JSONDecodeError, TypeError):
                        c = {}
                tname = (
                    c.get("tool")
                    if isinstance(c, dict) and isinstance(c.get("tool"), str)
                    else "tool"
                )
                names.append(tname)
                if i + 1 < len(messages) and messages[i + 1].get("role") == "tool_result":
                    raw_tr = messages[i + 1].get("content")
                    tr = raw_tr if raw_tr is not None else ""
                    if not isinstance(tr, str):
                        tr = json.dumps(tr)
                    results.append(f"- {tname}: {tr}")
                    i += 2
                else:
                    results.append(f"- {tname}: (no result)")
                    i += 1
            out.append(
                {"role": "assistant", "content": f"(used tools: {', '.join(names)})"}
            )
            out.append(
                {"role": "user", "content": "[tool results]\n" + "\n".join(results)}
            )
            continue
        elif role == "tool_result":
            tr_content = content if isinstance(content, str) else json.dumps(content)
            out.append({"role": "user", "content": f"[tool result] {tr_content}"})
        else:
            out.append({"role": "user", "content": str(content)})
        i += 1
    return out
