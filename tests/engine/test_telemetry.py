"""Telemetry export: off by default, structured JSON on demand,
OTel when installed — and never able to cost a run.
"""

from __future__ import annotations

import json
import logging

import pytest

from agent import telemetry


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("ALEX_ENV", "dev")
    monkeypatch.delenv("ALEX_TELEMETRY", raising=False)
    monkeypatch.delenv("ALEX_OTEL_ENDPOINT", raising=False)
    from alex import config, testing

    config._settings = None
    testing._fakes = None
    yield
    config._settings = None
    testing._fakes = None


def test_off_by_default_no_line_is_written(env, caplog):
    with caplog.at_level(logging.INFO, logger="alex.telemetry"):
        telemetry.emit("task_run_turn", run_id="r1", tool_calls=3)
    assert not caplog.records


def test_log_mode_writes_one_clean_json_line(env, monkeypatch, caplog):
    monkeypatch.setenv("ALEX_TELEMETRY", "log")
    with caplog.at_level(logging.INFO, logger="alex.telemetry"):
        telemetry.emit(
            "task_run_turn",
            run_id="r1", tool_calls=3, elapsed_s=7.5,
            skipped=None,                       # dropped
            blob="x" * 500,                     # truncated: attrs, not content
        )
    assert len(caplog.records) == 1
    row = json.loads(caplog.records[0].message)
    assert row["event"] == "task_run_turn"
    assert row["tool_calls"] == 3 and row["elapsed_s"] == 7.5
    assert "skipped" not in row
    assert len(row["blob"]) == 200, "attribute values are bounded"


def test_otlp_without_the_stack_degrades_to_log_lines(env, monkeypatch, caplog):
    monkeypatch.setenv("ALEX_TELEMETRY", "otlp")
    monkeypatch.setenv("ALEX_OTEL_ENDPOINT", "http://collector.invalid:4318")
    with caplog.at_level(logging.INFO, logger="alex.telemetry"):
        telemetry.emit("task_run_finished", run_id="r1", status="done")
    # opentelemetry is not installed in this repo: the event must still
    # land, as a log line, and nothing raised.
    lines = [r for r in caplog.records if r.name == "alex.telemetry"]
    assert len(lines) == 1
    assert json.loads(lines[0].message)["status"] == "done"


def test_emit_never_raises_on_hostile_values(env, monkeypatch):
    monkeypatch.setenv("ALEX_TELEMETRY", "log")

    class Weird:
        def __str__(self):
            return "weird-object"

    telemetry.emit("x", thing=Weird(), n=float("inf"))  # must not raise


def test_the_run_path_emits_turn_accounting(env, monkeypatch, caplog):
    monkeypatch.setenv("ALEX_TELEMETRY", "log")
    from agent import taskboard as tb
    from alex.testing import get_fakes
    from app._dev_seed import RUN_METRICS_COLS

    wh = get_fakes().warehouse
    wh.register("media.ops.run_metrics", RUN_METRICS_COLS,
                pk=["run_id", "turn"])

    def sql(stmt, params=None):
        res = wh.execute(stmt, params or {})
        as_dicts = getattr(res, "as_dicts", None)
        return as_dicts() if callable(as_dicts) else list(res or [])

    run = tb.TaskRun(run_id="run-tel", task_id="t-tel",
                     session_id="task-t-tel", started_by="dana@alex.dev")

    class VMStub:
        def last_governor_state(self):
            return {"elapsed_s": 4.0, "waited_s": 0.0, "tool_calls": 2,
                    "llm_turns": 3, "denials": 0,
                    "usage": {"prompt_tokens": 900, "completion_tokens": 40,
                              "pass_ms": []},
                    "budget": {}, "halted": None}

    with caplog.at_level(logging.INFO, logger="alex.telemetry"):
        tb._emit_usage(sql, run, VMStub(), turn=1)
    rows = [json.loads(r.message) for r in caplog.records
            if r.name == "alex.telemetry"]
    assert any(
        r["event"] == "task_run_turn" and r["prompt_tokens"] == 900
        for r in rows
    )
