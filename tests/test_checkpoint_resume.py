"""Tests for adaptive-loop checkpoint persistence and resume (Stage G).

Covers: disk persistence, disk restore after "restart", interrupted-turn
detection (non-terminal phase), resume context rendering, and the resume
event emitted by run_persistent_stream.
"""
import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.adaptive_agent_loop import (  # noqa: E402
    AdaptiveAgentLoop,
    ExecutionPhase,
    TurnCheckpoint,
)


@pytest.fixture()
def loop_with_tmp_checkpoint(tmp_path, monkeypatch):
    loop = AdaptiveAgentLoop()
    monkeypatch.setattr(
        loop, "_checkpoint_path",
        tmp_path / "adaptive_loop" / "last_checkpoint.json",
    )
    return loop


class TestCheckpointPersistence:
    def test_checkpoint_writes_to_disk(self, loop_with_tmp_checkpoint):
        loop = loop_with_tmp_checkpoint
        loop._phase = ExecutionPhase.EXECUTING
        loop.checkpoint("do the thing", tool_calls=[{"name": "native_read"}])

        assert loop._checkpoint_path.exists()
        data = json.loads(loop._checkpoint_path.read_text(encoding="utf-8"))
        assert data["user_input"] == "do the thing"
        assert data["phase"] == "executing"

    def test_restore_after_restart(self, loop_with_tmp_checkpoint, monkeypatch):
        loop = loop_with_tmp_checkpoint
        loop._phase = ExecutionPhase.EXECUTING
        loop.checkpoint("long task", tool_calls=[{"name": "terminal_execute"}])

        # Simulate process restart: brand-new loop, same checkpoint path.
        fresh = AdaptiveAgentLoop()
        monkeypatch.setattr(fresh, "_checkpoint_path", loop._checkpoint_path)

        restored = fresh.restore_checkpoint()
        assert restored is not None
        assert restored.user_input == "long task"
        assert restored.phase == "executing"
        assert restored.tool_calls == [{"name": "terminal_execute"}]

    def test_restore_no_file_returns_none(self, loop_with_tmp_checkpoint):
        assert loop_with_tmp_checkpoint.restore_checkpoint() is None

    def test_roundtrip_from_dict(self):
        cp = TurnCheckpoint(turn_number=3, phase="thinking", user_input="x",
                            tool_calls=[{"name": "a"}], cwd="/tmp")
        rebuilt = TurnCheckpoint.from_dict(cp.to_dict())
        assert rebuilt.turn_number == 3
        assert rebuilt.user_input == "x"
        assert rebuilt.tool_calls == [{"name": "a"}]
        assert rebuilt.cwd == "/tmp"

    def test_from_dict_tolerates_missing_keys(self):
        rebuilt = TurnCheckpoint.from_dict({})
        assert rebuilt.turn_number == 0
        assert rebuilt.phase == "idle"


class TestInterruptedDetection:
    def test_completed_turn_is_not_interrupted(self, loop_with_tmp_checkpoint):
        loop = loop_with_tmp_checkpoint
        loop._phase = ExecutionPhase.COMPLETED
        loop.checkpoint("done")
        assert loop.interrupted_checkpoint() is None

    def test_failed_turn_is_not_interrupted(self, loop_with_tmp_checkpoint):
        loop = loop_with_tmp_checkpoint
        loop._phase = ExecutionPhase.FAILED
        loop.checkpoint("bad")
        assert loop.interrupted_checkpoint() is None

    def test_thinking_turn_is_interrupted(self, loop_with_tmp_checkpoint):
        loop = loop_with_tmp_checkpoint
        loop._phase = ExecutionPhase.THINKING
        loop.checkpoint("in flight")
        interrupted = loop.interrupted_checkpoint()
        assert interrupted is not None
        assert interrupted.user_input == "in flight"

    def test_interrupted_after_restart_uses_disk(self, loop_with_tmp_checkpoint,
                                                 monkeypatch):
        loop = loop_with_tmp_checkpoint
        loop._phase = ExecutionPhase.EXECUTING
        loop.checkpoint("crashed mid-run")

        fresh = AdaptiveAgentLoop()
        monkeypatch.setattr(fresh, "_checkpoint_path", loop._checkpoint_path)
        interrupted = fresh.interrupted_checkpoint()
        assert interrupted is not None
        assert interrupted.user_input == "crashed mid-run"


class TestResumeContext:
    def test_empty_when_completed(self, loop_with_tmp_checkpoint):
        loop = loop_with_tmp_checkpoint
        loop._phase = ExecutionPhase.COMPLETED
        loop.checkpoint("done")
        assert loop.resume_context() == ""

    def test_renders_request_and_tools(self, loop_with_tmp_checkpoint):
        loop = loop_with_tmp_checkpoint
        loop._phase = ExecutionPhase.EXECUTING
        loop.checkpoint(
            "build the report",
            tool_calls=[{"name": "native_read"}, {"name": "terminal_execute"}],
            tool_results=[
                {"tool": "native_read", "success": True, "output": "data"},
                {"tool": "terminal_execute", "success": False, "output": "boom"},
            ],
        )
        ctx = loop.resume_context()
        assert "Resumed Session" in ctx
        assert "build the report" in ctx
        assert "native_read" in ctx
        assert "terminal_execute" in ctx
        assert "data" in ctx

    def test_last_checkpoint_lookup_by_turn(self, loop_with_tmp_checkpoint):
        loop = loop_with_tmp_checkpoint
        loop._turn_count = 1
        loop._phase = ExecutionPhase.COMPLETED
        loop.checkpoint("first")
        loop._turn_count = 2
        loop._phase = ExecutionPhase.THINKING
        loop.checkpoint("second")

        assert loop.last_checkpoint().user_input == "second"
        assert loop.last_checkpoint(turn_number=1).user_input == "first"
        assert loop.last_checkpoint(turn_number=99) is None


class TestResumeEvent:
    def test_run_persistent_stream_emits_resume_event(self, loop_with_tmp_checkpoint):
        loop = loop_with_tmp_checkpoint
        loop._phase = ExecutionPhase.EXECUTING
        loop.checkpoint("interrupted work")
        loop._phase = ExecutionPhase.THINKING  # as the crashed turn left it

        events = []

        async def run():
            async for event in loop.run_persistent_stream(
                user_input="new request",
                system_prompt="sys",
                tools={},
                max_iterations=1,
            ):
                events.append(event)

        asyncio.new_event_loop().run_until_complete(run())
        resume_events = [e for e in events if e.get("type") == "resume"]
        assert len(resume_events) == 1
        # The checkpoint recorded its phase when saved (executing).
        assert resume_events[0]["phase"] == "executing"
        assert resume_events[0]["user_input"] == "interrupted work"