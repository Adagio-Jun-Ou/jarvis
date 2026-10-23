"""Tests for the LLM client (Ollama cloud in dev, Databricks in production)."""
from __future__ import annotations

import pytest

from alex.config import reset_settings
from alex.llm import (
    DEFAULT_CLOUD_MODEL,
    OllamaCloudLLM,
    get_llm,
    reset_llm,
    set_llm,
    tool_schema,
)


@pytest.fixture(autouse=True)
def _reset_llm_singleton():
    reset_llm()
    yield
    reset_llm()


def test_get_llm_returns_ollama_in_dev_mode():
    reset_settings(mode="dev")
    llm = get_llm()
    assert isinstance(llm, OllamaCloudLLM)
    assert llm.model == DEFAULT_CLOUD_MODEL


def test_get_llm_respects_cloud_model_env(monkeypatch):
    monkeypatch.setenv("ALEX_CLOUD_MODEL", "gpt-oss:120b")
    reset_settings(mode="dev")
    llm = get_llm()
    assert llm.model == "gpt-oss:120b"


# ---- the approved-model gate ----------------------------------------------
# MODEL_CATALOG is a governance control: the list of serving endpoints
# the organisation approved. The configured default endpoint used to go
# straight through it — ALEX_DEFAULT_LLM_ENDPOINT=anything bypassed the
# one gate the catalog exists to be.


def test_an_unapproved_production_endpoint_is_refused(monkeypatch):
    monkeypatch.setenv("ALEX_DEFAULT_LLM_ENDPOINT", "some-random-endpoint")
    monkeypatch.setenv("ALEX_LLM_BACKEND", "databricks")
    reset_settings(mode="production")
    with pytest.raises(ValueError, match="approved catalog"):
        get_llm()


def test_an_approved_production_endpoint_is_accepted(monkeypatch):
    from agent.models import MODEL_CATALOG

    approved = MODEL_CATALOG["claude-sonnet"].endpoint
    monkeypatch.setenv("ALEX_DEFAULT_LLM_ENDPOINT", approved)
    monkeypatch.setenv("ALEX_LLM_BACKEND", "databricks")
    reset_settings(mode="production")
    llm = get_llm()
    assert llm.endpoint == approved


def test_the_default_production_endpoint_is_in_the_catalog():
    """The shipped default must itself be approved — a default that
    fails its own gate would brick every fresh production boot."""
    from agent.models import MODEL_CATALOG
    from alex.config import get_settings

    reset_settings(mode="production")
    settings = get_settings()
    endpoints = {r.endpoint for r in MODEL_CATALOG.values()}
    assert settings.default_llm_endpoint in endpoints


@pytest.mark.integration
def test_ollama_invoke_basic_text():
    """Smoke test: can the dev harness call the real Ollama cloud endpoint?
    This is a live-network test; in offline CI it will fail. We mark it
    as 'integration' so the default `pytest` run skips it."""
    llm = OllamaCloudLLM(model=DEFAULT_CLOUD_MODEL, timeout=30)
    resp = llm.invoke(
        messages=[{"role": "user", "content": "Reply with just the word PONG."}],
        temperature=0.0,
        max_tokens=512,
    )
    assert "PONG" in resp.content.upper()
    assert resp.model


@pytest.mark.integration
def test_ollama_invoke_with_tool_call():
    """The harness must be able to elicit a structured tool call."""
    llm = OllamaCloudLLM(model=DEFAULT_CLOUD_MODEL, timeout=30)
    tools = [tool_schema(
        name="get_weather",
        description="Get the current weather for a city.",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    )]
    resp = llm.invoke(
        messages=[{"role": "user", "content": "What's the weather in Paris?"}],
        tools=tools,
        temperature=0.0,
        max_tokens=512,
    )
    # The model may either answer in prose or call the tool. If it calls,
    # the call must have a valid name and a 'city' argument.
    if resp.tool_calls:
        tc = resp.tool_calls[0]
        assert tc.name == "get_weather"
        assert "city" in tc.args
        assert isinstance(tc.args["city"], str)


@pytest.mark.integration
def test_ollama_stream_yields_deltas():
    llm = OllamaCloudLLM(model=DEFAULT_CLOUD_MODEL, timeout=30)
    deltas = list(llm.stream(
        messages=[{"role": "user", "content": "Count: 1 2 3"}],
        temperature=0.0,
        max_tokens=512,
    ))
    # We expect at least one text delta and a final done delta
    assert any(d.text for d in deltas)
    assert any(d.done for d in deltas)
    # The model may echo or count. We just require the response to be
    # non-empty; specific digit assertions are flaky across model
    # versions.
    full = "".join(d.text for d in deltas if d.text)
    assert full.strip()


def test_tool_schema_shape():
    s = tool_schema(
        name="noop",
        description="does nothing",
        parameters={"type": "object", "properties": {}},
    )
    assert s["type"] == "function"
    assert s["function"]["name"] == "noop"
    assert s["function"]["parameters"]["type"] == "object"


def test_set_llm_replaces_singleton():
    """Tests can swap in a fake LLM without going to the network."""
    class FakeLLM:
        def invoke(self, *a, **kw):
            from alex.llm import LLMResponse
            return LLMResponse(content="FAKE")
    set_llm(FakeLLM())
    assert get_llm().invoke([]).content == "FAKE"


def test_stream_surfaces_midstream_ollama_error(monkeypatch):
    """An {"error": ...} object mid-stream becomes a real UpstreamError,
    not a silently-ended stream."""
    import alex.llm as llm
    from alex.errors import UpstreamError

    monkeypatch.setattr(
        llm, "_ollama_request_stream",
        lambda payload, timeout=60.0: iter([{"error": "model 'x' not found"}]),
    )
    client = llm.OllamaCloudLLM(model="x")
    with pytest.raises(UpstreamError, match="model 'x' not found"):
        list(client.stream(messages=[{"role": "user", "content": "hi"}]))


def test_to_response_surfaces_error_key():
    from alex.errors import UpstreamError
    from alex.llm import OllamaCloudLLM

    with pytest.raises(UpstreamError, match="rate limit"):
        OllamaCloudLLM._to_response({"error": "rate limit exceeded"})
