"""Tests for the SDK client (streaming, event parsing, hooks)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sdk"))

import httpx
import pytest
from alex_sdk.client import AgentEvent, Alex, _iter_sse


@pytest.fixture
def _alex():
    a = Alex(base_url="http://testserver", token="t")
    yield a
    a.close()


def test_agentevent_from_dict():
    d = {"type": "text", "text": "Hello"}
    ev = AgentEvent.from_dict(d)
    assert ev.type == "text"
    assert ev.text == "Hello"


def test_agentevent_confirm_required():
    d = {"type": "confirmation_required", "token": "abc12345", "summary": "Send"}
    ev = AgentEvent.from_dict(d, client=None, confirm_url="")
    assert ev.type == "confirmation_required"
    assert ev.token == "abc12345"
    # No client -> confirm is a no-op (returns None)
    assert ev.confirm() is None


def test_parse_handles_done_sentinel(_alex):
    ev = _alex._parse("[DONE]")
    assert ev is not None
    assert ev.type == "done"


def test_parse_handles_raw_text(_alex):
    ev = _alex._parse("raw text delta")
    assert ev is not None
    assert ev.type == "text"
    assert ev.text == "raw text delta"


def test_parse_handles_json_event(_alex):
    raw = json.dumps({"type": "tool", "tool": "x", "args": {"a": 1}})
    ev = _alex._parse(raw)
    assert ev is not None
    assert ev.type == "tool"
    assert ev.tool == "x"
    assert ev.args == {"a": 1}


def test_pre_event_hook_can_observe(_alex):
    seen = []
    @_alex.hook("pre_event")
    def h(ev: AgentEvent) -> None:
        seen.append(ev.type)
    _alex._parse(json.dumps({"type": "text", "text": "x"}))
    assert seen == ["text"]


def test_pre_event_hook_can_veto_by_raising(_alex):
    @_alex.hook("pre_event")
    def veto(ev: AgentEvent) -> None:
        if ev.type == "law_block":
            raise RuntimeError("vetoed by hook")
    with pytest.raises(RuntimeError, match="vetoed"):
        _alex._parse(json.dumps({"type": "law_block", "article": "A1", "detail": "no"}))


def test_iter_sse_yields_each_data_line():
    """The SSE parser must yield each `data: ...` line once and
    handle multi-byte UTF-8 that may split across read boundaries."""
    from unittest.mock import MagicMock

    # Build a fake httpx response whose iter_bytes yields chunks that
    # intentionally split a multi-byte UTF-8 character.
    body = "data: {\"type\": \"text\", \"text\": \"héllo\"}\n\ndata: [DONE]\n\n"
    chunks = [body[i:i + 5].encode("utf-8") for i in range(0, len(body), 5)]
    resp = MagicMock()
    resp.iter_bytes = lambda: iter(chunks)

    out = list(_iter_sse(resp))
    assert out == ['{"type": "text", "text": "héllo"}', "[DONE]"]


def test_sessions_endpoint_parses_json(_alex, monkeypatch):
    """Alex.sessions() returns a list of dicts."""
    captured = {}

    def _mock_handler(request):
        captured["url"] = request.url
        return httpx.Response(200, json={"sessions": [
            {"session_id": "s1", "title": "T1", "kind": "personal"},
        ]})

    transport = httpx.MockTransport(_mock_handler)
    a = Alex(base_url="http://testserver", token="t")
    a._http = httpx.Client(base_url="http://testserver", headers={"Authorization": "Bearer t"},
                           transport=transport, timeout=5.0)
    out = a.sessions()
    assert out == [{"session_id": "s1", "title": "T1", "kind": "personal"}]
    a.close()


def test_dev_token_is_a_string():
    tok = Alex.dev_token("alice@x.com")
    assert isinstance(tok, str)
    # Three dotted segments
    assert len(tok.split(".")) == 3


def test_cli_auth_includes_obo_header():
    # `alex skills` / `alex cap list` build raw requests via cli._auth; it
    # must include X-Forwarded-Access-Token or those commands 401 in prod.
    from types import SimpleNamespace

    from alex_sdk.cli import _auth
    h = _auth(SimpleNamespace(token="tok"))
    assert h["Authorization"] == "Bearer tok"
    assert h["X-Forwarded-Access-Token"] == "tok"
    assert _auth(SimpleNamespace(token="")) == {}
