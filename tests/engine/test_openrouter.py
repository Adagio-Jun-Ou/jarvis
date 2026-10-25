"""The OpenRouter backend — the local machine's LLM.

* the SSE stream is parsed into token deltas, accumulated tool calls
  (index-merged across chunks), usage, and a correct finish_reason;
* the executor emits the same event shape as every other executor
  (PassUsage on tool passes too; TurnComplete only on stop);
* the factory is keyed on the backend, not on dev/prod mode;
* the approved-endpoint gate accepts the OpenRouter catalog.
"""

from __future__ import annotations

import asyncio

import pytest

from alex import llm as llm_mod


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("ALEX_ENV", "dev")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.delenv("ALEX_LLM_BACKEND", raising=False)
    from alex import config

    config._settings = None
    yield
    config._settings = None


def _sse(chunks):
    def fake(payload, *, timeout):
        yield from chunks
    return fake


def test_stream_parses_text_tool_calls_and_usage(env, monkeypatch):
    chunks = [
        {"choices": [{"delta": {"content": "Let me "}}]},
        {"choices": [{"delta": {"content": "check."}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "call_1", "function": {"name": "view", "arguments": "{\"pa"}}]}}]},
        {"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": "th\": \"a.txt\"}"}}]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        {"choices": [], "usage": {"prompt_tokens": 120, "completion_tokens": 9}},
    ]
    monkeypatch.setattr(llm_mod, "_openrouter_request_stream", _sse(chunks))
    deltas = list(llm_mod.OpenRouterLLM().stream(
        [{"role": "user", "content": "hi"}], tools=[{"type": "function"}]
    ))
    text = "".join(d.text for d in deltas if d.text)
    assert text == "Let me check."
    calls = [d.tool_call for d in deltas if d.tool_call]
    assert len(calls) == 1
    assert calls[0].name == "view" and calls[0].args == {"path": "a.txt"}
    assert calls[0].id == "call_1"
    final = deltas[-1]
    assert final.done and final.finish_reason == "tool_calls"
    assert final.usage == {"prompt_tokens": 120, "completion_tokens": 9}


def test_stream_stop_carries_full_text(env, monkeypatch):
    chunks = [
        {"choices": [{"delta": {"content": "done"}, "finish_reason": "stop"}]},
    ]
    monkeypatch.setattr(llm_mod, "_openrouter_request_stream", _sse(chunks))
    deltas = list(llm_mod.OpenRouterLLM().stream([{"role": "user", "content": "q"}]))
    assert deltas[-1].finish_reason == "stop" and deltas[-1].full_text == "done"


def test_invoke_maps_the_openai_response(env, monkeypatch):
    monkeypatch.setattr(llm_mod, "_openrouter_request", lambda payload, *, timeout: {
        "model": "z-ai/glm-5.3-flash",
        "choices": [{"message": {"content": "answer", "tool_calls": [
            {"id": "c1", "function": {"name": "view", "arguments": "{\"path\": \"x\"}"}}]}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2},
    })
    r = llm_mod.OpenRouterLLM().invoke([{"role": "user", "content": "q"}])
    assert r.content == "answer" and r.tool_calls[0].args == {"path": "x"}
    assert r.usage["prompt_tokens"] == 5


def test_missing_key_is_a_clean_upstream_error(env, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY")
    with pytest.raises(llm_mod.UpstreamError, match="OPENROUTER_API_KEY"):
        llm_mod._openrouter_headers()


def test_executor_emits_pass_usage_and_turn_complete(env, monkeypatch):
    from agent.omnigent import OpenRouterExecutor
    from agent.omnigent.executor import PassUsage, TextChunk, ToolCallRequest, TurnComplete

    class FakeLLM:
        def stream(self, **kw):
            yield llm_mod.LLMDelta(text="ok")
            yield llm_mod.LLMDelta(tool_call=llm_mod.ToolCall(name="view", args={"p": 1}))
            yield llm_mod.LLMDelta(done=True, finish_reason="tool_calls", full_text="ok",
                                   usage={"prompt_tokens": 1, "completion_tokens": 1})

    async def _collect():
        out = []
        async for ev in OpenRouterExecutor(llm=FakeLLM()).run_turn(
            [{"role": "user", "content": "hi"}], [], "sys"
        ):
            out.append(ev)
        return out

    events = asyncio.run(_collect())
    kinds = [type(e).__name__ for e in events]
    assert kinds == ["TextChunk", "ToolCallRequest", "PassUsage"], kinds
    assert not any(isinstance(e, TurnComplete) for e in events), (
        "a tool-call pass ends the stream without TurnComplete"
    )
    assert isinstance(events[0], TextChunk) and isinstance(events[1], ToolCallRequest)
    assert isinstance(events[2], PassUsage)


def test_factory_is_keyed_on_backend_not_mode(env, monkeypatch):
    from agent.harness import executor_for
    from agent.omnigent import LocalOllamaExecutor, OpenRouterExecutor

    assert isinstance(executor_for("dev"), OpenRouterExecutor), (
        "a key present -> OpenRouter, even in dev"
    )
    monkeypatch.setenv("ALEX_LLM_BACKEND", "ollama")
    assert isinstance(executor_for("production"), LocalOllamaExecutor)
    with pytest.raises(ValueError):
        executor_for("nope")


def test_default_endpoint_follows_the_backend_and_is_approved(env):
    from alex.config import get_settings

    assert get_settings().default_llm_endpoint == "z-ai/glm-5.3-flash"
    llm_mod._require_approved_endpoint("z-ai/glm-5.3-flash")  # must not raise
    from agent.models import resolve_model

    assert resolve_model("auto", "what time is standup?").endpoint == "z-ai/glm-5.3-flash"
    assert resolve_model("auto", "analyze the contract rights").endpoint == "z-ai/glm-5.3"


def test_get_llm_returns_openrouter_when_key_present(env):
    llm_mod.reset_llm()
    try:
        assert isinstance(llm_mod.get_llm(), llm_mod.OpenRouterLLM)
    finally:
        llm_mod.reset_llm()
