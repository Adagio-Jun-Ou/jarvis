"""Tests for the event protocol."""
from __future__ import annotations

from alex.events import (
    Event,
    artifact_event,
    confirmation_event,
    done_event,
    error_event,
    law_block_event,
    parse,
    reasoning_event,
    text_event,
    tool_event,
)


def test_text_event_sse_roundtrip():
    e = text_event("Hello, world")
    line = e.to_sse().rstrip()
    assert line.startswith("data: ")
    parsed = parse(line)
    assert parsed is not None
    assert parsed.type == "text"
    assert parsed.text == "Hello, world"


def test_done_event_sse_roundtrip():
    e = done_event(session_id="abc")
    parsed = parse(e.to_sse().rstrip())
    assert parsed.type == "done"
    assert parsed.session_id == "abc"


def test_tool_event_roundtrip():
    e = tool_event("check_rights", args={"asset_id": "A1"}, result="CLEARED")
    parsed = parse(e.to_sse().rstrip())
    assert parsed.type == "tool"
    assert parsed.tool == "check_rights"
    assert parsed.args == {"asset_id": "A1"}
    assert parsed.result == "CLEARED"


def test_reasoning_event_roundtrip():
    e = reasoning_event("plan", "User asked about A1")
    parsed = parse(e.to_sse().rstrip())
    assert parsed.type == "reasoning"
    assert parsed.phase == "plan"
    assert parsed.text == "User asked about A1"


def test_law_block_event_roundtrip():
    e = law_block_event("A4", "check_rights not performed")
    parsed = parse(e.to_sse().rstrip())
    assert parsed.type == "law_block"
    assert parsed.article == "A4"
    assert parsed.detail == "check_rights not performed"


def test_confirmation_event_roundtrip():
    e = confirmation_event(token="tok-1", summary="Send draft", body={"to": "x@y.com"})
    parsed = parse(e.to_sse().rstrip())
    assert parsed.type == "confirmation_required"
    assert parsed.token == "tok-1"
    assert parsed.summary == "Send draft"
    assert parsed.body == {"to": "x@y.com"}


def test_artifact_event_roundtrip():
    e = artifact_event("outputs/report.md", "file")
    parsed = parse(e.to_sse().rstrip())
    assert parsed.type == "artifact"
    assert parsed.path == "outputs/report.md"


def test_error_event_roundtrip():
    e = error_event("E_BAD", "bad input")
    parsed = parse(e.to_sse().rstrip())
    assert parsed.type == "error"
    assert parsed.code == "E_BAD"
    assert parsed.message == "bad input"


def test_parse_handles_sentinel():
    assert parse("data: [DONE]") is None
    assert parse("data: ") is None
    assert parse("") is None
    assert parse("not-data-line") is None


def test_parse_handles_unknown_type_as_extras():
    e = Event(type="future_type", extras={"foo": "bar"})
    parsed = parse(e.to_sse().rstrip())
    assert parsed is not None
    assert parsed.type == "future_type"
    assert parsed.extras.get("foo") == "bar"


def test_parse_handles_legacy_raw_text():
    """Old clients used to emit raw text as the SSE payload. We still
    parse it as a text event for backwards compatibility."""
    parsed = parse("data: hello world")
    assert parsed is not None
    assert parsed.type == "text"
    assert parsed.text == "hello world"


def test_no_newline_in_serialized_event():
    e = text_event("line1\nline2")
    s = e.to_sse()
    # The only \n should be the trailing \n\n terminator
    body = s.split("data: ")[1]
    assert "\n" not in body.rstrip(), "event body must be a single line"


def test_json_compact_no_whitespace():
    e = tool_event("t", args={"a": 1})
    body = e.to_sse().split("data: ")[1].rstrip()
    assert ": " not in body, "must use compact JSON"
