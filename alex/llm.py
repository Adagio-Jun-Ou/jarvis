"""
LLM client for the Alex system.

This module is the only place the rest of the codebase talks to an LLM.
Two implementations exist:

  * `OllamaCloudLLM` — used in dev/test. Hits Ollama's cloud API directly.
    Authenticated by an API key. Always available during local development,
    no Databricks workspace required.

  * `DatabricksServingLLM` — the production implementation. Wraps the
    Databricks Model Serving endpoint via the Databricks SDK. The endpoint
    is selected from the agent's `models.py` catalog and routed through
    AI Gateway.

The contract is the same for both: `invoke(messages, **kwargs) -> LLMResponse`
where `LLMResponse.content` is the model's text output and `LLMResponse.tool_calls`
is a list of structured tool calls. Both backends are also streaming-capable
via `stream(messages) -> Iterator[LLMDelta]`.

In dev mode the system loads `OLLAMA_API_KEY` (or the hard-coded dev key
in this file) and uses `OllamaCloudLLM`. Tests can replace the singleton
with a fake via `set_llm(...)`.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from alex.config import get_settings
from alex.errors import UpstreamError
from alex.retry import call_with_retry

log = logging.getLogger("alex.llm")


@dataclass
class ToolCall:
    name: str
    args: dict = field(default_factory=dict)
    id: str = ""


@dataclass
class LLMResponse:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = None
    model: str = ""
    usage: dict = field(default_factory=dict)


@dataclass
class LLMDelta:
    text: str = ""
    tool_call: ToolCall | None = None
    done: bool = False
    # Set on the final delta of a stream: "stop" or "tool_calls".
    finish_reason: str | None = None
    # Accumulated text of the whole turn, set on the final delta.
    full_text: str = ""
    # Token accounting, set on the final delta when the provider reports
    # it ({"prompt_tokens": .., "completion_tokens": ..}). Empty dict
    # when unknown — never invented.
    usage: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Ollama cloud implementation (dev / local / test)
# ---------------------------------------------------------------------------


# Default dev model. Override with ALEX_CLOUD_MODEL (or OLLAMA_MODEL) to
# match a model that is actually pulled locally. Production uses the
# catalog from `agent.models`.
DEFAULT_CLOUD_MODEL = (
    os.environ.get("ALEX_CLOUD_MODEL", "").strip()
    or os.environ.get("OLLAMA_MODEL", "").strip()
    or "gpt-oss:20b"
)
# Dev LLM endpoint. Defaults to a local Ollama daemon so dev mode is
# fully offline. Point OLLAMA_CHAT_URL at https://ollama.com/api/chat
# (with OLLAMA_API_KEY set) to use Ollama Cloud instead.
OLLAMA_CHAT_URL = (
    os.environ.get("OLLAMA_CHAT_URL", "").strip()
    or "http://127.0.0.1:11434/api/chat"
)
# Local models can take minutes on prompt eval for large contexts;
# this is the max silence between chunks, not a total-call cap.
DEFAULT_LLM_TIMEOUT_S = float(os.environ.get("ALEX_LLM_TIMEOUT", "300"))
# Ollama defaults to a 4096-token context; the agent's system prompt +
# tool schemas need more. Overridable for small-VRAM machines.
OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "16384"))


def _resolve_local_model(model: str) -> str:
    """If we're talking to a local Ollama daemon and the configured model
    isn't pulled, fall back to the first locally available chat model so
    dev mode works out of the box. Cloud URLs are returned unchanged."""
    if "://127.0.0.1" not in OLLAMA_CHAT_URL and "://localhost" not in OLLAMA_CHAT_URL:
        return model
    try:
        tags_url = OLLAMA_CHAT_URL.rsplit("/api/", 1)[0] + "/api/tags"
        with urllib.request.urlopen(tags_url, timeout=3) as r:
            names = [m.get("name", "") for m in json.loads(r.read()).get("models", [])]
    except Exception:
        return model
    if model in names or any(n.split(":")[0] == model for n in names):
        return model
    chat_models = [n for n in names if "bge" not in n and "embed" not in n]

    def _rank(n: str) -> int:
        low = n.lower()
        for i, kw in enumerate(("hermes", "llama", "qwen", "gemma", "mistral")):
            if kw in low:
                return i
        return 99

    chat_models.sort(key=_rank)
    if chat_models:
        logging.getLogger("alex.llm").warning(
            "model %r not pulled locally; falling back to %r", model, chat_models[0]
        )
        return chat_models[0]
    return model


def _parse_hermes_tool_calls(text: str) -> list[ToolCall]:
    """Extract Hermes-format tool calls from raw model text:
    ``<tool_call>{"name": ..., "arguments": {...}}</tool_call>``."""
    import re

    out: list[ToolCall] = []
    # Capture everything up to the closing tag rather than `\{.*?\}`: the
    # non-greedy brace form stops at the FIRST '}', so a tool call with
    # nested-object arguments ({"arguments":{"a":{"b":1}}}) produced invalid
    # JSON and was silently dropped.
    for m in re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL):
        try:
            d = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if not isinstance(d, dict):
            continue
        name = str(d.get("name") or "")
        args = d.get("arguments") or d.get("args") or {}
        if not isinstance(args, dict):
            args = {"_raw": str(args)}
        if name:
            out.append(ToolCall(name=name, args=args))
    return out


def _strip_hermes_tool_calls(text: str) -> str:
    import re

    return re.sub(
        r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL
    ).strip()


def _ollama_headers() -> dict:
    headers = {"Content-Type": "application/json"}
    key = os.environ.get("OLLAMA_API_KEY", "").strip()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def _ollama_request(payload: dict, *, timeout: float = 60.0) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(OLLAMA_CHAT_URL, data=body, headers=_ollama_headers())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise UpstreamError(f"Ollama HTTP {e.code}: {body}") from e
    except urllib.error.URLError as e:
        raise UpstreamError(f"Ollama connection error: {e}") from e


def _ollama_request_stream(payload: dict, *, timeout: float = 60.0) -> Iterator[dict]:
    """Stream NDJSON responses from Ollama's /api/chat endpoint."""
    body = json.dumps({**payload, "stream": True}).encode("utf-8")
    req = urllib.request.Request(OLLAMA_CHAT_URL, data=body, headers=_ollama_headers())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for line in r:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise UpstreamError(f"Ollama HTTP {e.code}: {body}") from e
    except urllib.error.URLError as e:
        raise UpstreamError(f"Ollama connection error: {e}") from e


class OllamaCloudLLM:
    """The LLM client used in dev and tests. Production never instantiates this."""

    def __init__(
        self, model: str = DEFAULT_CLOUD_MODEL, *, timeout: float | None = None
    ) -> None:
        self.model = model
        self.timeout = timeout if timeout is not None else DEFAULT_LLM_TIMEOUT_S
        self._resolved: str | None = None

    def _effective_model(self, override: str | None = None) -> str:
        """The model actually sent to Ollama. Resolved lazily (and once)
        against the local daemon's pulled models; see _resolve_local_model."""
        if override:
            return override
        if self._resolved is None:
            self._resolved = _resolve_local_model(self.model)
        return self._resolved

    def invoke(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        temperature: float = 0.1,
        max_tokens: int | None = None,
        response_format: dict | None = None,
    ) -> LLMResponse:
        payload: dict = {
            "model": self._effective_model(),
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_ctx": OLLAMA_NUM_CTX},
        }
        if tools:
            payload["tools"] = tools
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens
        if response_format:
            payload["format"] = response_format
        data = call_with_retry(
            lambda: _ollama_request(payload, timeout=self.timeout),
            attempts=3,
            retriable=(UpstreamError,),
        )
        return self._to_response(data)

    def stream(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        tools: list[dict] | None = None,
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ) -> Iterator[LLMDelta]:
        """Stream the LLM response. ``model`` is accepted (and
        ignored if ``None``) so callers that pass an explicit model
        override — including the Omnigent executors — don't have
        to special-case which LLM client they're talking to."""
        effective_model = self._effective_model(model)
        payload: dict = {
            "model": effective_model,
            "messages": messages,
            "options": {"temperature": temperature, "num_ctx": OLLAMA_NUM_CTX},
        }
        if tools:
            payload["tools"] = tools
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens
        acc: list[str] = []
        saw_tool_call = False
        holding_text = False  # true once a Hermes-style <tool_call> tag appears
        for chunk in _ollama_request_stream(payload, timeout=self.timeout):
            # Ollama can emit an {"error": ...} object mid-stream (model
            # not found, context exceeded, rate limit). Surface it as a
            # real error instead of silently ending the stream.
            if chunk.get("error"):
                raise UpstreamError(f"Ollama stream error: {chunk['error']}")
            if chunk.get("done"):
                full = "".join(acc)
                # Hermes-format fallback: models whose Ollama template
                # lacks native tool support emit
                # <tool_call>{"name":...,"arguments":...}</tool_call>
                # as plain text. Parse those so any local model can
                # drive tools.
                if not saw_tool_call and tools and "<tool_call>" in full:
                    parsed = _parse_hermes_tool_calls(full)
                    for tc in parsed:
                        saw_tool_call = True
                        yield LLMDelta(tool_call=tc)
                    full = _strip_hermes_tool_calls(full)
                yield LLMDelta(
                    done=True,
                    finish_reason="tool_calls" if saw_tool_call else "stop",
                    full_text=full,
                    usage={
                        "prompt_tokens": int(chunk.get("prompt_eval_count") or 0),
                        "completion_tokens": int(chunk.get("eval_count") or 0),
                    },
                )
                return
            msg = chunk.get("message") or {}
            text = msg.get("content") or ""
            if text:
                acc.append(text)
                # Once a <tool_call> tag shows up, stop forwarding raw
                # text to the UI; the accumulated tail is parsed at done.
                if not holding_text and "<tool_call" in "".join(acc[-8:]):
                    holding_text = True
                if not holding_text:
                    yield LLMDelta(text=text)
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args else {}
                    except json.JSONDecodeError:
                        args = {"_raw": args}
                else:
                    args = args or {}
                saw_tool_call = True
                yield LLMDelta(
                    tool_call=ToolCall(
                        name=fn.get("name", ""),
                        args=args,
                        id=str(tc.get("id", "")),
                    )
                )

    @staticmethod
    def _to_response(data: dict) -> LLMResponse:
        if isinstance(data, dict) and data.get("error"):
            raise UpstreamError(f"Ollama error: {data['error']}")
        msg = data.get("message") or {}
        text = msg.get("content") or ""
        tcs: list[ToolCall] = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args else {}
                except json.JSONDecodeError:
                    args = {"_raw": args}
            tcs.append(
                ToolCall(
                    name=fn.get("name", ""), args=args or {}, id=str(tc.get("id", ""))
                )
            )
        return LLMResponse(
            content=text,
            tool_calls=tcs,
            raw=data,
            model=data.get("model", ""),
            usage={
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
            },
        )


# ---------------------------------------------------------------------------
# OpenRouter implementation (the local-machine backend)
# ---------------------------------------------------------------------------

OPENROUTER_URL = os.environ.get(
    "OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions"
)
DEFAULT_OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "z-ai/glm-5.3-flash")


def llm_backend() -> str:
    """Which LLM backend this process talks to: openrouter | ollama |
    databricks | mock. Explicit via ALEX_LLM_BACKEND; otherwise
    openrouter when a key is present, else the local Ollama daemon."""
    explicit = os.environ.get("ALEX_LLM_BACKEND", "").strip().lower()
    if explicit:
        return explicit
    if os.environ.get("OPENROUTER_API_KEY", "").strip():
        return "openrouter"
    return "ollama"


def _openrouter_headers() -> dict:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise UpstreamError("OPENROUTER_API_KEY is not set")
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
        "HTTP-Referer": os.environ.get("OPENROUTER_REFERER", "http://localhost"),
        "X-Title": os.environ.get("OPENROUTER_TITLE", "Jarvis"),
    }


def _openrouter_request(payload: dict, *, timeout: float) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(OPENROUTER_URL, data=body, headers=_openrouter_headers())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise UpstreamError(
            f"OpenRouter HTTP {e.code}: {e.read().decode('utf-8', errors='replace')[:800]}"
        ) from e
    except urllib.error.URLError as e:
        raise UpstreamError(f"OpenRouter connection error: {e}") from e


def _openrouter_request_stream(payload: dict, *, timeout: float) -> Iterator[dict]:
    """Server-sent events from the OpenAI-compatible endpoint."""
    body = json.dumps({**payload, "stream": True}).encode("utf-8")
    req = urllib.request.Request(OPENROUTER_URL, data=body, headers=_openrouter_headers())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                try:
                    yield json.loads(data)
                except json.JSONDecodeError:
                    continue
    except urllib.error.HTTPError as e:
        raise UpstreamError(
            f"OpenRouter HTTP {e.code}: {e.read().decode('utf-8', errors='replace')[:800]}"
        ) from e
    except urllib.error.URLError as e:
        raise UpstreamError(f"OpenRouter connection error: {e}") from e


def _oai_tool_calls(raw: list | None) -> list[ToolCall]:
    out: list[ToolCall] = []
    for tc in raw or []:
        fn = tc.get("function") or {}
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args) if args else {}
            except json.JSONDecodeError:
                args = {"_raw": args}
        out.append(ToolCall(name=fn.get("name", ""), args=args or {}, id=str(tc.get("id", ""))))
    return out


class OpenRouterLLM:
    """OpenAI-wire-compatible client for OpenRouter: real token
    streaming, native tool calls, usage on the final chunk. The one
    LLM client the local (personal) deployment uses."""

    def __init__(self, model: str = DEFAULT_OPENROUTER_MODEL, *, timeout: float | None = None) -> None:
        self.model = model
        self.timeout = timeout if timeout is not None else DEFAULT_LLM_TIMEOUT_S

    def _payload(self, messages, *, model, tools, temperature, max_tokens) -> dict:
        payload: dict = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        return payload

    def invoke(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        temperature: float = 0.1,
        max_tokens: int | None = None,
        response_format: dict | None = None,
    ) -> LLMResponse:
        payload = self._payload(messages, model=None, tools=tools,
                                temperature=temperature, max_tokens=max_tokens)
        if response_format:
            payload["response_format"] = response_format
        data = call_with_retry(
            lambda: _openrouter_request(payload, timeout=self.timeout),
            attempts=3, retriable=(UpstreamError,),
        )
        if data.get("error"):
            raise UpstreamError(f"OpenRouter error: {data['error']}")
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        usage = data.get("usage") or {}
        return LLMResponse(
            content=msg.get("content") or "",
            tool_calls=_oai_tool_calls(msg.get("tool_calls")),
            raw=data,
            model=data.get("model", self.model),
            usage={
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "completion_tokens": int(usage.get("completion_tokens") or 0),
            },
        )

    def stream(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        tools: list[dict] | None = None,
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ) -> Iterator[LLMDelta]:
        payload = self._payload(messages, model=model, tools=tools,
                                temperature=temperature, max_tokens=max_tokens)
        payload["stream_options"] = {"include_usage": True}
        acc: list[str] = []
        pending: dict[int, dict] = {}  # tool calls accumulate across chunks by index
        usage: dict = {}
        for chunk in _openrouter_request_stream(payload, timeout=self.timeout):
            if chunk.get("error"):
                raise UpstreamError(f"OpenRouter stream error: {chunk['error']}")
            if chunk.get("usage"):
                u = chunk["usage"]
                usage = {
                    "prompt_tokens": int(u.get("prompt_tokens") or 0),
                    "completion_tokens": int(u.get("completion_tokens") or 0),
                }
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                text = delta.get("content") or ""
                if text:
                    acc.append(text)
                    yield LLMDelta(text=text)
                for tc in delta.get("tool_calls") or []:
                    idx = int(tc.get("index") or 0)
                    slot = pending.setdefault(idx, {"id": "", "name": "", "args": ""})
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["name"] += fn["name"]
                    if fn.get("arguments"):
                        slot["args"] += fn["arguments"]
        full = "".join(acc)
        saw_tool_call = False
        for idx in sorted(pending):
            slot = pending[idx]
            try:
                args = json.loads(slot["args"]) if slot["args"] else {}
            except json.JSONDecodeError:
                args = {"_raw": slot["args"]}
            if slot["name"]:
                saw_tool_call = True
                yield LLMDelta(tool_call=ToolCall(name=slot["name"], args=args, id=slot["id"]))
        if not saw_tool_call and tools and "<tool_call>" in full:
            for tc in _parse_hermes_tool_calls(full):
                saw_tool_call = True
                yield LLMDelta(tool_call=tc)
            full = _strip_hermes_tool_calls(full)
        yield LLMDelta(
            done=True,
            finish_reason="tool_calls" if saw_tool_call else "stop",
            full_text=full,
            usage=usage,
        )


# ---------------------------------------------------------------------------
# Databricks serving implementation (production)
# ---------------------------------------------------------------------------


class DatabricksServingLLM:
    """Thin wrapper over Databricks Model Serving. Production-only.

    The model is selected by endpoint name (matches the catalog in
    `agent/models.py`). The Databricks SDK handles auth via the OBO token
    (production) or the workspace client (dev/staging). AI Gateway provides
    rate limits, PII guardrails, and payload logging."""

    def __init__(
        self,
        endpoint: str = "alex-agent",
        *,
        max_tokens: int = 1024,
        temperature: float = 0.1,
        sdk_client=None,
    ) -> None:
        self.endpoint = endpoint
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._sdk = sdk_client  # may be None; created lazily

    def _client(self):
        if self._sdk is None:
            from databricks.sdk import WorkspaceClient

            self._sdk = WorkspaceClient()
        return self._sdk

    def invoke(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict | None = None,
    ) -> LLMResponse:
        # The Databricks OpenAI-compatible chat completions endpoint accepts
        # the same shape as the OpenAI Chat API. We pass messages verbatim
        # and let Gateway handle guardrails.
        from databricks.sdk.service.serving import ChatMessage

        cm = [
            ChatMessage(role=m["role"], content=m.get("content", "")) for m in messages
        ]
        req_kwargs: dict = dict(
            model=self.endpoint,
            messages=cm,
            max_tokens=max_tokens or self.max_tokens,
            temperature=self.temperature if temperature is None else temperature,
        )
        if response_format:
            req_kwargs["response_format"] = response_format
        try:
            resp = self._client().serving_endpoints.query(**req_kwargs)
        except Exception as e:
            raise UpstreamError(f"serving endpoint failed: {e}") from e
        # The response shape matches OpenAI ChatCompletion.
        text = ""
        tcs: list[ToolCall] = []
        for choice in resp.choices or []:
            msg = choice.message or {}
            # `or {}` makes msg a dict when the SDK returns no message;
            # a dict has no .content, so attribute access must be soft.
            text += getattr(msg, "content", None) or ""
            for tc in getattr(msg, "tool_calls", None) or []:
                fn = getattr(tc, "function", None) or {}
                args = getattr(fn, "arguments", None) or "{}"
                try:
                    parsed = json.loads(args) if isinstance(args, str) else args
                except json.JSONDecodeError:
                    parsed = {"_raw": str(args)}
                tcs.append(
                    ToolCall(
                        name=getattr(fn, "name", ""),
                        args=parsed,
                        id=str(getattr(tc, "id", "")),
                    )
                )
        return LLMResponse(
            content=text,
            tool_calls=tcs,
            raw=resp,
            model=self.endpoint,
            usage={
                "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0)
                if resp.usage
                else 0,
                "completion_tokens": getattr(resp.usage, "completion_tokens", 0)
                if resp.usage
                else 0,
            },
        )

    def stream(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[LLMDelta]:
        """Stream from a Databricks serving endpoint.

        ``model`` is accepted (and ignored if ``None``) so callers
        that pass an explicit model override — including the
        Omnigent executors — don't have to special-case which
        LLM client they're talking to.

        SDK reality check: databricks-sdk's ``serving_endpoints.query``
        with ``stream=True`` returns a single ``QueryEndpointResponse``
        (it forces the JSON Accept header and does ``from_dict``), not
        an iterable of chunks — iterating it raised TypeError on the
        first production turn. Until the SDK ships a real SSE client,
        streaming here is a one-shot ``invoke`` surfaced as a single
        delta: the ReAct loop, tool calls and audit trail all work
        unchanged; only token-by-token display is lost.
        """
        response = self.invoke(
            messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        # LLMResponse carries the answer in `content` — `.full_text` was
        # a phantom attribute that made every production stream die as
        # an executor error before the first token reached anyone.
        text = response.content or ""
        if text:
            yield LLMDelta(text=text)
        for call in response.tool_calls or []:
            yield LLMDelta(tool_call=call)
        yield LLMDelta(
            done=True,
            finish_reason="tool_calls" if response.tool_calls else "stop",
            full_text=text,
            usage=dict(getattr(response, "usage", None) or {}),
        )


# ---------------------------------------------------------------------------
# Singleton resolution
# ---------------------------------------------------------------------------


_llm: Any | None = None


def get_llm() -> Any:
    """Return the configured LLM for the current process.

    Dev/test: `OllamaCloudLLM` (always available).
    Production: `DatabricksServingLLM` with the default endpoint from
    settings — validated against the approved catalog in
    `agent.models`, because a catalog nobody checks is a README claim,
    not a control. An endpoint name that is not an approved serving
    endpoint fails fast here rather than 500-ing on the first turn.
    Tests can override via `set_llm(...)`."""
    global _llm
    if _llm is not None:
        return _llm
    settings = get_settings()
    backend = llm_backend()
    if backend == "openrouter":
        _require_approved_endpoint(settings.default_llm_endpoint)
        _llm = OpenRouterLLM(model=settings.default_llm_endpoint)
    elif backend == "ollama" or settings.mode == "dev":
        model = os.environ.get("ALEX_CLOUD_MODEL", DEFAULT_CLOUD_MODEL)
        _llm = OllamaCloudLLM(model=model)
    else:
        _require_approved_endpoint(settings.default_llm_endpoint)
        _llm = DatabricksServingLLM(endpoint=settings.default_llm_endpoint)
    return _llm


def _require_approved_endpoint(endpoint: str) -> None:
    """Fail closed when the configured endpoint is not in the catalog.

    The catalog (`agent.models.MODEL_CATALOG`) is the governance list of
    approved serving endpoints; `resolve_model` checks it for
    user-requested models, but the *configured* endpoint went straight
    through — so `ALEX_DEFAULT_LLM_ENDPOINT=anything` bypassed the one
    gate the catalog exists to be. Both dev-catalog endpoints and the
    raw serving endpoint names are accepted, so operators can set
    either form.
    """
    from agent.models import MODEL_CATALOG

    approved = {r.endpoint for r in MODEL_CATALOG.values()} | set(MODEL_CATALOG)
    if endpoint not in approved:
        raise ValueError(
            f"configured default LLM endpoint {endpoint!r} is not in the "
            f"approved catalog. Approved: {sorted(approved)}. Set "
            "ALEX_DEFAULT_LLM_ENDPOINT to one of these; the catalog is a "
            "governance control, not a suggestion."
        )


def set_llm(client) -> None:
    global _llm
    _llm = client


def reset_llm() -> None:
    global _llm
    _llm = None


# ---------------------------------------------------------------------------
# Tool schema helpers
# ---------------------------------------------------------------------------


def tool_schema(name: str, description: str, parameters: dict) -> dict:
    """Convert a Python function signature into the JSON schema Ollama expects
    for a tool. Mirrors the OpenAI function-calling shape."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }
