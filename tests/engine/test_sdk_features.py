"""Tests for the SDK's product-parity features: tools/teach/feedback,
reasoning streams, Session.correct, MCP conversion, and the new CLI
subcommands. All HTTP goes through httpx.MockTransport."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sdk"))

import httpx
import pytest
from alex_sdk.client import AgentEvent, Alex, EventType, ToolInfo
from alex_sdk.mcp import to_mcp_tools

MANIFEST = {
    "tools": [
        {
            "name": "search_email",
            "description": "Search the user's mailbox",
            "inputSchema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
        {"name": "create_task", "description": "Create a task"},
    ]
}


def _sse(*events: dict | str) -> bytes:
    out = []
    for e in events:
        data = e if isinstance(e, str) else json.dumps(e)
        out.append(f"data: {data}\n\n")
    return "".join(out).encode()


def _alex_with(handler) -> Alex:
    return Alex(
        base_url="http://testserver",
        token="t",
        transport=httpx.MockTransport(handler),
    )


# ---------------------------------------------------------------------------
# Alex.tools()
# ---------------------------------------------------------------------------


def test_tools_returns_toolinfo_list():
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/api/tools"
        assert req.method == "GET"
        assert req.headers["X-Forwarded-Access-Token"] == "t"
        return httpx.Response(200, json=MANIFEST)

    with _alex_with(handler) as alex:
        tools = alex.tools()
    assert [t.name for t in tools] == ["search_email", "create_task"]
    assert isinstance(tools[0], ToolInfo)
    assert tools[0].description == "Search the user's mailbox"
    assert tools[0].input_schema["required"] == ["query"]
    assert tools[1].input_schema == {}


def test_tools_empty_manifest():
    with _alex_with(lambda r: httpx.Response(200, json={"tools": []})) as alex:
        assert alex.tools() == []


# ---------------------------------------------------------------------------
# Alex.teach() / Alex.feedback()
# ---------------------------------------------------------------------------


def test_teach_posts_prompt():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["path"] = req.url.path
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"routine_id": "r1", "status": "stored"})

    with _alex_with(handler) as alex:
        r = alex.teach("To file an expense: open Workday", context="finance")
    assert seen["path"] == "/api/teach"
    assert seen["body"] == {"prompt": "To file an expense: open Workday", "context": "finance"}
    assert r == {"routine_id": "r1", "status": "stored"}


def test_feedback_posts_verdict():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["path"] = req.url.path
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"status": "recorded"})

    with _alex_with(handler) as alex:
        r = alex.feedback("sugg-1", "inaccurate", comment="wrong date")
    assert seen["path"] == "/api/feedback"
    assert seen["body"] == {
        "suggestion_id": "sugg-1",
        "feedback": "inaccurate",
        "comment": "wrong date",
    }
    assert r == {"status": "recorded"}


def test_feedback_rejects_bad_verdict_client_side():
    def handler(req: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("must not reach the network")

    with _alex_with(handler) as alex:
        with pytest.raises(ValueError, match="helpful"):
            alex.feedback("sugg-1", "meh")


# ---------------------------------------------------------------------------
# Streaming: new event types + show_reasoning + correct()
# ---------------------------------------------------------------------------

STREAM_EVENTS = [
    {"type": "reasoning", "text": "thinking about the inbox"},
    {"type": "tool", "tool": "search_email", "args": {"query": "urgent"}, "phase": "start"},
    {"type": "memory", "text": "remembered a preference"},
    {"type": "law_block", "article": "A3", "detail": "audit required"},
    {"type": "artifact", "path": "outputs/report.md"},
    {"type": "text", "text": "Here is your summary."},
    {"type": "done"},
]


def _chat_handler(seen: dict):
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/sessions":
            return httpx.Response(200, json={"session_id": "s1"})
        assert req.url.path == "/api/chat"
        seen["body"] = json.loads(req.content)
        return httpx.Response(
            200,
            content=_sse(*STREAM_EVENTS),
            headers={"content-type": "text/event-stream"},
        )

    return handler


def test_run_show_reasoning_and_typed_events():
    seen: dict = {}
    with _alex_with(_chat_handler(seen)) as alex:
        session = alex.session("t")
        events = list(session.run("summarize", show_reasoning=True))
    assert seen["body"]["show_reasoning"] is True
    types = [ev.type for ev in events]
    assert types == ["reasoning", "tool", "memory", "law_block", "artifact", "text", "done"]
    # EventType compares equal to the wire strings.
    assert events[0].type == EventType.REASONING
    assert events[3].type == EventType.LAW_BLOCK
    # .data helpers strip the discriminator.
    assert events[0].data == {"text": "thinking about the inbox"}
    assert events[1].data["args"] == {"query": "urgent"}
    assert events[3].article == "A3" and events[3].detail == "audit required"
    assert events[4].path == "outputs/report.md"


def test_run_custom_inputs_passthrough():
    seen: dict = {}
    with _alex_with(_chat_handler(seen)) as alex:
        session = alex.session("t")
        list(session.run("hi", custom_inputs={"k": "v"}))
    assert seen["body"]["custom_inputs"] == {"k": "v"}


def test_correct_posts_edited_output():
    seen: dict = {}
    with _alex_with(_chat_handler(seen)) as alex:
        session = alex.session("t")
        events = list(session.correct("fixed text", original_output="draft text"))
    body = seen["body"]
    assert body["custom_inputs"]["edited_output"] == "fixed text"
    assert body["custom_inputs"]["original_output"] == "draft text"
    assert body["text"]  # a follow-up prompt is always sent
    assert events[-1].type == EventType.DONE


def test_correct_omits_empty_original():
    seen: dict = {}
    with _alex_with(_chat_handler(seen)) as alex:
        session = alex.session("t")
        list(session.correct("fixed text"))
    assert "original_output" not in seen["body"]["custom_inputs"]


def test_sse_parses_all_new_event_types():
    a = Alex(base_url="http://x", token="t")
    try:
        for etype in (
            "text",
            "reasoning",
            "tool",
            "artifact",
            "law_block",
            "memory",
            "confirmation_required",
            "error",
            "done",
        ):
            ev = a._parse(json.dumps({"type": etype}))
            assert isinstance(ev, AgentEvent)
            assert ev.type == etype
            assert ev.type == EventType(etype)
    finally:
        a.close()


# ---------------------------------------------------------------------------
# MCP conversion
# ---------------------------------------------------------------------------


def test_to_mcp_tools_from_manifest_dict():
    defs = to_mcp_tools(MANIFEST)
    assert defs[0] == MANIFEST["tools"][0]
    # Missing schema gets the default empty-object schema.
    assert defs[1]["inputSchema"] == {"type": "object", "properties": {}}


def test_to_mcp_tools_from_list_and_snake_case():
    defs = to_mcp_tools(
        [
            {"name": "a", "input_schema": {"type": "object", "properties": {"x": {}}}},
            {"description": "nameless — dropped"},
            "not a dict",
            {"name": "  "},
        ]
    )
    assert len(defs) == 1
    assert defs[0]["name"] == "a"
    assert defs[0]["inputSchema"]["properties"] == {"x": {}}


def test_to_mcp_tools_pure_and_empty():
    assert to_mcp_tools({}) == []
    assert to_mcp_tools([]) == []
    assert to_mcp_tools(None) == []
    manifest = {"tools": [{"name": "a"}]}
    before = json.dumps(manifest)
    to_mcp_tools(manifest)
    assert json.dumps(manifest) == before  # no mutation


def test_serve_mcp_is_wired_to_a_real_server():
    """``serve_mcp`` builds a real MCP server (see
    tests/test_sdk_mcp_server.py for the behavioural coverage). Without
    the optional ``mcp`` package it must fail with a actionable message
    rather than an obscure ImportError."""
    import alex_sdk.mcp as m

    assert callable(m.build_mcp_server)
    try:
        import mcp  # noqa: F401
    except ImportError:
        with pytest.raises(NotImplementedError, match="mcp"):
            m.build_mcp_server(None)
        return
    # With mcp installed, a bad client surfaces its own error, not a stub.
    with pytest.raises(Exception) as exc:
        m.build_mcp_server(None)
    assert not isinstance(exc.value, NotImplementedError)


# ---------------------------------------------------------------------------
# CLI: alex tools / alex teach / alex chat --reasoning
# ---------------------------------------------------------------------------


def _patched_client(monkeypatch, handler):
    from alex_sdk import cli

    def factory(args):
        return Alex(
            base_url=args.app_url,
            token=args.token or "t",
            transport=httpx.MockTransport(handler),
        )

    monkeypatch.setattr(cli, "_client", factory)
    return cli


def test_cli_tools_table(monkeypatch, capsys):
    cli = _patched_client(monkeypatch, lambda r: httpx.Response(200, json=MANIFEST))
    assert cli.main(["tools"]) == 0
    out = capsys.readouterr().out
    assert "search_email  Search the user's mailbox" in out
    assert "create_task" in out


def test_cli_teach(monkeypatch, capsys):
    def handler(req: httpx.Request) -> httpx.Response:
        assert json.loads(req.content)["prompt"] == "do the thing"
        return httpx.Response(200, json={"routine_id": "r9", "status": "stored"})

    cli = _patched_client(monkeypatch, handler)
    assert cli.main(["teach", "do the thing"]) == 0
    assert "r9" in capsys.readouterr().out


def test_cli_chat_reasoning_flag(monkeypatch, capsys):
    seen: dict = {}
    cli = _patched_client(monkeypatch, _chat_handler(seen))
    assert cli.main(["chat", "summarize", "--reasoning"]) == 0
    out = capsys.readouterr().out
    assert seen["body"]["show_reasoning"] is True
    assert "[reasoning] thinking about the inbox" in out
    assert "\x1b[2m" in out  # dimmed
    assert "Here is your summary." in out


def test_cli_chat_hides_reasoning_by_default(monkeypatch, capsys):
    seen: dict = {}
    cli = _patched_client(monkeypatch, _chat_handler(seen))
    assert cli.main(["chat", "summarize"]) == 0
    out = capsys.readouterr().out
    assert seen["body"]["show_reasoning"] is False
    assert "[reasoning]" not in out
    assert "Here is your summary." in out


# ---------------------------------------------------------------------------
# Alex.call_tool() — MCP call-tool seam
# ---------------------------------------------------------------------------


def test_call_tool_posts_args_and_returns_result():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["path"] = req.url.path
        seen["method"] = req.method
        seen["body"] = json.loads(req.content)
        return httpx.Response(
            200,
            json={"tool": "create_spreadsheet", "result": {"ok": True, "path": "outputs/q1.xlsx"}},
        )

    with _alex_with(handler) as alex:
        result = alex.call_tool(
            "create_spreadsheet", filename="q1", headers=["M", "R"], rows=[["Jan", 100]]
        )
    assert seen["path"] == "/api/tools/create_spreadsheet"
    assert seen["method"] == "POST"
    assert seen["body"] == {"args": {"filename": "q1", "headers": ["M", "R"], "rows": [["Jan", 100]]}}
    assert result == {"ok": True, "path": "outputs/q1.xlsx"}


def test_session_messages_fetches_persisted_history():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["path"] = req.url.path
        seen["method"] = req.method
        return httpx.Response(200, json={
            "session_id": "s-42",
            "messages": [
                {"turn_no": 0, "role": "user", "content": "hi"},
                {"turn_no": 1, "role": "assistant", "content": "hello"},
            ],
        })

    with _alex_with(handler) as alex:
        msgs = alex.session("s-42").messages()
    assert seen["path"] == "/api/sessions/s-42/messages"
    assert seen["method"] == "GET"
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[1]["content"] == "hello"


def test_resume_binds_existing_session_without_creating():
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append((req.method, req.url.path))
        return httpx.Response(200, json={"messages": []})

    with _alex_with(handler) as alex:
        s = alex.resume("s-existing")
        assert s.id == "s-existing"
        s.messages()
    # resume() must NOT POST /api/sessions (no new session created).
    assert ("POST", "/api/sessions") not in calls
    assert ("GET", "/api/sessions/s-existing/messages") in calls
