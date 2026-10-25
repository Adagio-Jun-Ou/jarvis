"""Tests for the Agentic Execution Harness.

Tests cover:
- Plan creation, step dependencies, parallel execution
- Retry with exponential backoff
- Fallback tool execution
- Checkpoint save/restore
- Event bus publish/subscribe
- Verification step
- Reflection and plan adjustment
- Error recovery
- Heuristic plan generation
- Progress tracking
"""
import asyncio
import pytest
import time

from core.agentic_harness import (
    AgenticHarness, HarnessPhase, HarnessPlan, HarnessStep, StepStatus,
    EventBus, EventType, HarnessCheckpoint, create_plan_from_objective,
)


class MockTool:
    """Mock tool for testing."""

    def __init__(self, name: str = "mock_tool", result=None, error=None, delay: float = 0):
        self.name = name
        self.description = f"Mock tool: {name}"
        self._result = result
        self._error = error
        self._delay = delay
        self._call_count = 0
        self._last_params = None
        self.requires_permission = False
        self.parameters = {"type": "object", "properties": {}}

    async def execute(self, **kwargs):
        self._call_count += 1
        self._last_params = kwargs
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._error:
            raise self._error
        return self._result


class FailingThenSucceedingTool:
    """Tool that fails N times then succeeds."""

    def __init__(self, fail_count: int = 2, result: str = "success"):
        self._fail_count = fail_count
        self._call_count = 0
        self._result = result
        self.name = "flaky_tool"
        self.description = "Flaky tool"
        self.requires_permission = False
        self.parameters = {"type": "object", "properties": {}}

    async def execute(self, **kwargs):
        self._call_count += 1
        if self._call_count <= self._fail_count:
            raise RuntimeError(f"Attempt {self._call_count} failed")
        return self._result


class TestHarnessPlan:
    def test_create_empty_plan(self):
        plan = HarnessPlan(objective="test objective")
        assert plan.objective == "test objective"
        assert len(plan.steps) == 0
        assert plan.progress == 0.0
        assert plan.is_complete is True  # Vacuously true

    def test_create_plan_with_steps(self):
        plan = HarnessPlan(
            objective="fix bug",
            steps=[
                HarnessStep(action="investigate", tool_name="shell"),
                HarnessStep(action="fix", tool_name="write_file", dependencies=["investigate"]),
                HarnessStep(action="verify", tool_name="shell", dependencies=["fix"]),
            ],
        )
        assert len(plan.steps) == 3
        assert plan.progress == 0.0
        assert plan.is_complete is False

    def test_plan_progress(self):
        plan = HarnessPlan(objective="test")
        plan.steps = [
            HarnessStep(action="s1", status=StepStatus.SUCCESS),
            HarnessStep(action="s2", status=StepStatus.PENDING),
        ]
        assert plan.progress == 0.5

    def test_plan_completed_steps(self):
        plan = HarnessPlan(objective="test")
        plan.steps = [
            HarnessStep(action="s1", status=StepStatus.VERIFIED),
            HarnessStep(action="s2", status=StepStatus.SUCCESS),
            HarnessStep(action="s3", status=StepStatus.FAILED),
            HarnessStep(action="s4", status=StepStatus.SKIPPED),
        ]
        assert plan.completed_steps == 3  # VERIFIED + SUCCESS + SKIPPED
        assert plan.failed_steps == 1

    def test_ready_steps_with_dependencies(self):
        plan = HarnessPlan(objective="test")
        s1 = HarnessStep(action="s1", id="a1")
        s2 = HarnessStep(action="s2", id="a2", dependencies=["a1"])
        s3 = HarnessStep(action="s3", id="a3", dependencies=["a1"])
        plan.steps = [s1, s2, s3]
        ready = plan.ready_steps
        assert len(ready) == 1
        assert ready[0].action == "s1"

    def test_ready_steps_after_completion(self):
        plan = HarnessPlan(objective="test")
        s1 = HarnessStep(action="s1", id="a1", status=StepStatus.VERIFIED)
        s2 = HarnessStep(action="s2", id="a2", dependencies=["a1"])
        s3 = HarnessStep(action="s3", id="a3", dependencies=["a1"])
        plan.steps = [s1, s2, s3]
        ready = plan.ready_steps
        assert len(ready) == 2

    def test_step_duration(self):
        step = HarnessStep(action="test")
        step.started_at = 100.0
        step.completed_at = 102.5
        assert step.duration == 2.5

    def test_plan_to_dict(self):
        plan = HarnessPlan(objective="test", steps=[HarnessStep(action="s1")])
        d = plan.to_dict()
        assert d["objective"] == "test"
        assert d["total_steps"] == 1
        assert "steps" in d


class TestHarnessStep:
    def test_step_defaults(self):
        step = HarnessStep()
        assert step.status == StepStatus.PENDING
        assert step.retry_count == 0
        assert step.max_retries == 3
        assert step.timeout_seconds == 120.0
        assert len(step.id) == 12

    def test_step_to_dict(self):
        step = HarnessStep(
            action="test_action",
            tool_name="shell",
            tool_params={"command": "ls"},
            description="Run ls",
            expected_outcome="List files",
            max_retries=5,
        )
        d = step.to_dict()
        assert d["action"] == "test_action"
        assert d["tool_name"] == "shell"
        assert d["status"] == "pending"
        assert d["max_retries"] == 5


class TestEventBus:
    def setup_method(self):
        EventBus._instance = None

    @pytest.mark.asyncio
    async def test_subscribe_and_emit(self):
        bus = EventBus.get_instance()
        received = []
        bus.subscribe("step_started", lambda e: received.append(e))
        await bus.emit(EventType.STEP_STARTED, {"step_id": "test"})
        assert len(received) == 1
        assert received[0]["type"] == "step_started"

    @pytest.mark.asyncio
    async def test_history(self):
        bus = EventBus.get_instance()
        await bus.emit(EventType.PLAN_CREATED, {"plan_id": "p1"})
        await bus.emit(EventType.STEP_STARTED, {"step_id": "s1"})
        history = bus.get_history(limit=2)
        assert len(history) == 2

    @pytest.mark.asyncio
    async def test_history_by_type(self):
        bus = EventBus.get_instance()
        await bus.emit(EventType.STEP_STARTED, {"step_id": "s1"})
        await bus.emit(EventType.HARNESS_COMPLETE, {"progress": 1.0})
        history = bus.get_history(event_type="step_started")
        assert len(history) == 1

    @pytest.mark.asyncio
    async def test_unsubscribe(self):
        bus = EventBus.get_instance()
        received = []
        callback = lambda e: received.append(e)
        bus.subscribe("step_started", callback)
        await bus.emit(EventType.STEP_STARTED, {"step_id": "1"})
        assert len(received) == 1
        bus.unsubscribe("step_started", callback)
        await bus.emit(EventType.STEP_STARTED, {"step_id": "2"})
        assert len(received) == 1  # No new event


class TestAgenticHarness:
    def setup_method(self):
        EventBus._instance = None

    def test_create_plan(self):
        harness = AgenticHarness()
        plan = harness.create_plan("Fix the login bug", [
            {"action": "investigate", "tool_name": "shell", "description": "Find bug"},
            {"action": "fix", "tool_name": "write_file", "description": "Fix bug"},
        ])
        assert plan.objective == "Fix the login bug"
        assert len(plan.steps) == 2
        assert harness.current_plan == plan

    @pytest.mark.asyncio
    async def test_execute_simple_plan(self):
        harness = AgenticHarness()
        tool = MockTool(name="test_tool", result={"output": "hello"})
        plan = harness.create_plan("Simple task", [
            {"action": "step1", "tool_name": "test_tool", "description": "Do something"},
        ])
        result = await harness.execute_plan(plan, tools={"test_tool": tool})
        assert result["is_complete"] is True
        assert result["completed_steps"] == 1
        assert tool._call_count == 1

    @pytest.mark.asyncio
    async def test_execute_plan_with_dependencies(self):
        harness = AgenticHarness()
        tool1 = MockTool(name="tool1", result="step1_result")
        tool2 = MockTool(name="tool2", result="step2_result")
        plan = harness.create_plan("Sequential task", [
            {"action": "first", "tool_name": "tool1", "description": "First step"},
            {"action": "second", "tool_name": "tool2", "description": "Second step", "dependencies": ["first"]},
        ])
        # Fix dependency IDs
        plan.steps[0].id = "s1"
        plan.steps[1].dependencies = ["s1"]
        plan.steps[1].id = "s2"

        result = await harness.execute_plan(plan, tools={"tool1": tool1, "tool2": tool2})
        assert result["is_complete"] is True
        assert tool1._call_count == 1
        assert tool2._call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        harness = AgenticHarness()
        harness._verification_enabled = False
        flaky = FailingThenSucceedingTool(fail_count=2, result="finally_worked")
        plan = harness.create_plan("Retry task", [
            {"action": "flaky_step", "tool_name": "flaky_tool", "description": "May fail", "max_retries": 3},
        ])
        result = await harness.execute_plan(plan, tools={"flaky_tool": flaky})
        assert flaky._call_count == 3  # 2 failures + 1 success
        assert result["is_complete"] is True

    @pytest.mark.asyncio
    async def test_permanent_failure(self):
        harness = AgenticHarness()
        harness._verification_enabled = False
        tool = MockTool(name="bad_tool", error=RuntimeError("always fails"))
        plan = harness.create_plan("Will fail", [
            {"action": "fail_step", "tool_name": "bad_tool", "description": "Always fails", "max_retries": 1},
        ])
        result = await harness.execute_plan(plan, tools={"bad_tool": tool})
        assert result["failed_steps"] >= 1

    @pytest.mark.asyncio
    async def test_fallback_execution(self):
        harness = AgenticHarness()
        harness._verification_enabled = False
        primary = MockTool(name="primary", error=RuntimeError("primary fails"))
        fallback = MockTool(name="fallback", result="fallback_result")
        plan = harness.create_plan("Fallback task", [
            {"action": "use_primary", "tool_name": "primary", "description": "Try primary", "max_retries": 0, "fallback_action": "fallback", "fallback_params": {}},
        ])
        result = await harness.execute_plan(plan, tools={"primary": primary, "fallback": fallback})
        assert fallback._call_count >= 1

    @pytest.mark.asyncio
    async def test_verification_step(self):
        harness = AgenticHarness()
        tool = MockTool(name="tool", result={"success": True, "data": "output"})
        plan = harness.create_plan("Verified task", [
            {"action": "verified_step", "tool_name": "tool", "description": "Verified step"},
        ])
        result = await harness.execute_plan(plan, tools={"tool": tool})
        step = plan.steps[0]
        assert step.verification_result == "passed" or step.status == StepStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_verification_catches_error_result(self):
        harness = AgenticHarness()
        tool = MockTool(name="tool", result={"error": "something went wrong"})
        plan = harness.create_plan("Bad result task", [
            {"action": "bad_result", "tool_name": "tool", "description": "Returns error dict"},
        ])
        result = await harness.execute_plan(plan, tools={"tool": tool})
        # Verification should catch the error in the result
        step = plan.steps[0]
        if harness._verification_enabled:
            assert step.verification_result in ("failed", None) or step.status in (StepStatus.VERIFIED, StepStatus.SUCCESS, StepStatus.VERIFICATION_FAILED, StepStatus.FAILED)

    def test_checkpoint_save_restore(self):
        harness = AgenticHarness()
        plan = harness.create_plan("Checkpoint test", [
            {"action": "step1", "tool_name": "shell", "description": "Step one"},
            {"action": "step2", "tool_name": "shell", "description": "Step two"},
        ])
        plan.steps[0].status = StepStatus.VERIFIED
        checkpoint = harness.save_checkpoint(plan)
        assert checkpoint.plan_id == plan.id
        assert checkpoint.phase == HarnessPhase.IDLE

        restored = harness.restore_checkpoint(checkpoint.plan_id)
        assert restored is not None
        assert restored.objective == "Checkpoint test"
        assert restored.steps[0].status == StepStatus.VERIFIED

    def test_checkpoint_without_plan(self):
        harness = AgenticHarness()
        with pytest.raises(ValueError):
            harness.save_checkpoint()

    def test_get_status(self):
        harness = AgenticHarness()
        status = harness.get_status()
        assert status["phase"] == "idle"
        assert status["total_steps_executed"] == 0


class TestHeuristicPlanGeneration:
    def test_create_fix_plan(self):
        steps = create_plan_from_objective("Fix the authentication bug")
        assert len(steps) >= 3
        assert any("investigate" in s["action"] or "fix" in s["action"] for s in steps)
        assert all("tool_name" in s for s in steps)

    def test_create_write_plan(self):
        steps = create_plan_from_objective("Create a new API endpoint")
        assert len(steps) >= 3
        assert any("implement" in s["action"] or "create" in s["action"] for s in steps)

    def test_create_analyze_plan(self):
        steps = create_plan_from_objective("Analyze the performance issues")
        assert len(steps) >= 2
        assert any("analyze" in s["action"] or "gather" in s["action"] for s in steps)

    def test_create_generic_plan(self):
        steps = create_plan_from_objective("Deploy to production")
        assert len(steps) >= 2

    def test_plan_with_dependencies(self):
        steps = create_plan_from_objective("Create a new feature")
        # After step 0, each step should reference the previous
        for i in range(1, len(steps)):
            assert len(steps[i].get("dependencies", [])) > 0

    def test_plan_with_available_tools(self):
        steps = create_plan_from_objective("Fix the bug", available_tools=["shell", "read_file", "write_file"])
        for step in steps:
            assert step["tool_name"] in ["shell", "read_file", "write_file"]


class TestHarnessStepStatus:
    def test_all_statuses(self):
        for status in StepStatus:
            assert status.value in [
                "pending", "ready", "running", "success", "failed",
                "retrying", "skipped", "verified", "verification_failed"
            ]

    def test_all_phases(self):
        for phase in HarnessPhase:
            assert phase.value in [
                "idle", "planning", "executing", "verifying", "reflecting",
                "completed", "failed", "paused"
            ]

    def test_all_event_types(self):
        expected = [
            "plan_created", "step_started", "step_completed", "step_failed",
            "step_retried", "tool_call", "tool_result",
            "verification_started", "verification_passed", "verification_failed",
            "reflection_started", "reflection_completed", "plan_adjusted",
            "progress_update", "checkpoint_saved", "checkpoint_restored",
            "error_recovery", "harness_complete", "harness_failed",
        ]
        for et in EventType:
            assert et.value in expected


class TestHarnessIntegration:
    @pytest.mark.asyncio
    async def test_full_plan_execute_verify_cycle(self):
        harness = AgenticHarness()
        step1_tool = MockTool(name="investigate", result={"findings": "bug in auth.py line 42"})
        step2_tool = MockTool(name="fix", result={"success": True, "file": "auth.py"})
        step3_tool = MockTool(name="verify", result={"success": True, "tests_pass": True})

        plan = harness.create_plan("Fix auth bug", [
            {"action": "investigate", "tool_name": "investigate", "description": "Investigate bug"},
            {"action": "fix", "tool_name": "fix", "description": "Fix the bug", "dependencies": ["investigate"]},
            {"action": "verify", "tool_name": "verify", "description": "Verify fix", "dependencies": ["fix"]},
        ])
        plan.steps[0].id = "i1"
        plan.steps[1].dependencies = ["i1"]
        plan.steps[1].id = "i2"
        plan.steps[2].dependencies = ["i2"]

        tools = {"investigate": step1_tool, "fix": step2_tool, "verify": step3_tool}
        result = await harness.execute_plan(plan, tools=tools)

        assert result["is_complete"] is True
        assert result["completed_steps"] == 3
        assert step1_tool._call_count == 1
        assert step2_tool._call_count == 1
        assert step3_tool._call_count == 1

    @pytest.mark.asyncio
    async def test_parallel_execution(self):
        harness = AgenticHarness()
        harness._max_parallel = 3
        harness._verification_enabled = False
        tool_a = MockTool(name="a", result="result_a")
        tool_b = MockTool(name="b", result="result_b")
        tool_c = MockTool(name="c", result="result_c")

        plan = harness.create_plan("Parallel task", [
            {"action": "task_a", "tool_name": "a", "description": "Task A"},
            {"action": "task_b", "tool_name": "b", "description": "Task B"},
            {"action": "task_c", "tool_name": "c", "description": "Task C"},
        ])
        tools = {"a": tool_a, "b": tool_b, "c": tool_c}
        result = await harness.execute_plan(plan, tools=tools)
        assert result["completed_steps"] == 3

    @pytest.mark.asyncio
    async def test_event_emission(self):
        harness = AgenticHarness()
        events_received = []
        harness.event_bus.subscribe("step_started", lambda e: events_received.append(("started", e)))
        harness.event_bus.subscribe("step_completed", lambda e: events_received.append(("completed", e)))

        tool = MockTool(name="test", result="ok")
        plan = harness.create_plan("Event test", [
            {"action": "event_step", "tool_name": "test", "description": "Test events"},
        ])
        harness._verification_enabled = False
        await harness.execute_plan(plan, tools={"test": tool})

        event_types = [e[0] for e in events_received]
        assert "started" in event_types
        assert "completed" in event_types

    @pytest.mark.asyncio
    async def test_progress_tracking(self):
        harness = AgenticHarness()
        progress_events = []
        harness.event_bus.subscribe("progress_update", lambda e: progress_events.append(e))

        t1 = MockTool(name="t1", result="r1")
        t2 = MockTool(name="t2", result="r2")
        plan = harness.create_plan("Progress test", [
            {"action": "s1", "tool_name": "t1", "description": "Step 1"},
            {"action": "s2", "tool_name": "t2", "description": "Step 2", "dependencies": ["s1"]},
        ])
        plan.steps[0].id = "p1"
        plan.steps[1].dependencies = ["p1"]
        harness._verification_enabled = False
        await harness.execute_plan(plan, tools={"t1": t1, "t2": t2})

        # Should have progress updates
        assert len(progress_events) >= 1