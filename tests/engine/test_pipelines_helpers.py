"""Tests for the pipeline helpers (pipelines/_helpers.py)."""
from __future__ import annotations

import pytest

from pipelines import _helpers as h


def test_parameterized_rejects_fstring():
    # An f-string would be:   "SELECT 'foo{a}bar'"
    # The check matches  '{name}' inside a quoted string.
    stmt = "SELECT '{a}' FROM x"
    with pytest.raises(ValueError, match="f-string"):
        h.parameterized(stmt)


def test_parameterized_passes_clean():
    stmt = "SELECT 1 FROM x WHERE a = :a"
    s, p = h.parameterized(stmt, a=1)
    assert s == stmt
    assert p == {"a": 1}


def test_parse_json_response_handles_fences():
    raw = '```json\n{"x": 1}\n```'
    assert h.parse_json_response(raw) == {"x": 1}


def test_parse_json_response_handles_prose():
    raw = 'Here you go: {"x": 1}'
    assert h.parse_json_response(raw) == {"x": 1}


def test_parse_json_response_handles_invalid():
    assert h.parse_json_response("not json") is None
    assert h.parse_json_response("") is None
    assert h.parse_json_response("[1, 2, 3]") is None  # not a dict


def test_with_backoff_eventually_succeeds():
    calls = {"n": 0}
    def f():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("nope")
        return "ok"
    assert h.with_backoff(f, attempts=5, base_delay=0.001) == "ok"
    assert calls["n"] == 3


def test_with_backoff_propagates_after_attempts():
    def f():
        raise ValueError("nope")
    with pytest.raises(ValueError):
        h.with_backoff(f, attempts=3, base_delay=0.001)


def test_coalesce_columns():
    out = h.coalesce_columns({"a": 1}, {"a": 0, "b": 2})
    assert out == {"a": 1, "b": 2}
