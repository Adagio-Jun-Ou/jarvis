r"""Hermes tool-call parsing must handle nested-object arguments.

The old `\{.*?\}` regex stopped at the first '}', so any tool call whose
arguments contained a nested object produced invalid JSON and was silently
dropped — the agent would appear to make no tool call at all.
"""
from alex.llm import _parse_hermes_tool_calls


def test_nested_object_arguments_parse():
    text = ('<tool_call>{"name": "create_spreadsheet", '
            '"arguments": {"data": {"rows": [[1, 2]], "meta": {"title": "Q1"}}}}'
            '</tool_call>')
    calls = _parse_hermes_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "create_spreadsheet"
    assert calls[0].args["data"]["meta"]["title"] == "Q1"


def test_multiple_tool_calls_still_parse():
    text = ('<tool_call>{"name": "a", "arguments": {}}</tool_call>'
            'noise'
            '<tool_call>{"name": "b", "arguments": {"x": {"y": 1}}}</tool_call>')
    calls = _parse_hermes_tool_calls(text)
    assert [c.name for c in calls] == ["a", "b"]
    assert calls[1].args["x"]["y"] == 1


def test_malformed_tool_call_is_skipped_not_crash():
    text = '<tool_call>{not json}</tool_call><tool_call>{"name":"ok","arguments":{}}</tool_call>'
    calls = _parse_hermes_tool_calls(text)
    assert [c.name for c in calls] == ["ok"]
