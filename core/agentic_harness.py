"""Agentic Execution Harness for Jarvis.

Modern agentic execution engine inspired by OpenCode, Claude Code, and
state-of-the-art agent architectures. Implements the Plan-Execute-Verify
loop with structured error recovery, retry with backoff, parallel tool
execution, checkpoint/resume, and verification steps.

Key design principles:
1. Plan-Execute-Verify: Every task goes through planning, execution, and
   verification phases before proceeding
2. Structured error recovery: Retry with exponential backoff, fallback
   tools, and re-planning on failure
3. Checkpoint/resume: Save execution state and resume from failures
4. Progressive verification: Verify tool outputs before proceeding
5. Streaming events: Event bus for external observation and UI updates
6. Timeout management: Per-tool timeouts with graceful cancellation
7. Parallel execution: Run independent tools concurrently
8. Self-correction: LLM-powered reflection and re-planning on errors

Based on research:
- OpenCode (Anomaly, 2025) — multi-session, LSP-enabled, streaming
- ReAct (Yao et al., 2023) — Reason+Act pattern
- Reflexion (Shinn et al., 2023) — self-reflective agent loop
- LATS (Zhou et al., 2023) — tree-structured agent search
- AgentCoder (Huang et al., 2023) — task decomposition and verification
- SWE-Agent (Yang et al., 2024) — autonomous software engineering
- Devon (2024) — plan-execute-verify cycle
- Aider (Paul Gauthier, 2024) — search-replace editing with git integration
- Claude Code (Anthropic, 2025) — agentic coding with tool use
"""
import asyncio
import hashlib
import json
import logging
import re
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional, TYPE_CHECKING

from core.helpers import safe_error_message

if TYPE_CHECKING:
    from .agent import JarvisAgent

logger = logging.getLogger(__name__)


class HarnessPhase(Enum):
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    REFLECTING = "reflecting"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


class StepStatus(Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"
    SKIPPED = "skipped"
    VERIFIED = "verified"
    VERIFICATION_FAILED = "verification_failed"


class EventType(Enum):
    PLAN_CREATED = "plan_created"
    STEP_STARTED = "step_started"
    STEP_COMPLETED = "step_completed"
    STEP_FAILED = "step_failed"
    STEP_RETRIED = "step_retried"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    VERIFICATION_STARTED = "verification_started"
    VERIFICATION_PASSED = "verification_passed"
    VERIFICATION_FAILED = "verification_failed"
    REFLECTION_STARTED = "reflection_started"
    REFLECTION_COMPLETED = "reflection_completed"
    PLAN_ADJUSTED = "plan_adjusted"
    PROGRESS_UPDATE = "progress_update"
    CHECKPOINT_SAVED = "checkpoint_saved"
    CHECKPOINT_RESTORED = "checkpoint_restored"
    ERROR_RECOVERY = "error_recovery"
    HARNESS_COMPLETE = "harness_complete"
    HARNESS_FAILED = "harness_failed"


@dataclass
class HarnessStep:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    action: str = ""
    tool_name: str = ""
    tool_params: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    expected_outcome: str = ""
    dependencies: list[str] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    result: Any = None
    error: str | None = None
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: float = 120.0
    started_at: float | None = None
    completed_at: float | None = None
    verification_result: str | None = None
    fallback_action: str | None = None
    fallback_params: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return 0.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "action": self.action,
            "tool_name": self.tool_name,
            "tool_params": self.tool_params,
            "description": self.description,
            "expected_outcome": self.expected_outcome,
            "dependencies": self.dependencies,
            "status": self.status.value,
            "error": self.error,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "timeout_seconds": self.timeout_seconds,
            "duration": round(self.duration, 3),
            "verification_result": self.verification_result,
        }


@dataclass
class HarnessPlan:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    objective: str = ""
    steps: list[HarnessStep] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    context: dict[str, Any] = field(default_factory=dict)
    max_iterations: int = 10
    current_step_index: int = 0
    reflection_count: int = 0
    max_reflections: int = 3

    @property
    def completed_steps(self) -> int:
        return sum(1 for s in self.steps if s.status in (StepStatus.SUCCESS, StepStatus.VERIFIED, StepStatus.SKIPPED))

    @property
    def failed_steps(self) -> int:
        return sum(1 for s in self.steps if s.status in (StepStatus.FAILED, StepStatus.VERIFICATION_FAILED))

    @property
    def progress(self) -> float:
        if not self.steps:
            return 0.0
        return self.completed_steps / len(self.steps)

    @property
    def is_complete(self) -> bool:
        return all(s.status in (StepStatus.SUCCESS, StepStatus.VERIFIED, StepStatus.SKIPPED) for s in self.steps)

    @property
    def ready_steps(self) -> list[HarnessStep]:
        completed_ids = {s.id for s in self.steps if s.status in (StepStatus.SUCCESS, StepStatus.VERIFIED, StepStatus.SKIPPED)}
        return [
            s for s in self.steps
            if s.status == StepStatus.PENDING
            and all(dep in completed_ids for dep in s.dependencies)
        ]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "objective": self.objective,
            "total_steps": len(self.steps),
            "completed_steps": self.completed_steps,
            "failed_steps": self.failed_steps,
            "progress": round(self.progress, 3),
            "current_step": self.current_step_index,
            "reflection_count": self.reflection_count,
            "is_complete": self.is_complete,
            "steps": [s.to_dict() for s in self.steps],
        }


@dataclass
class HarnessCheckpoint:
    plan_id: str
    phase: HarnessPhase
    plan: dict
    timestamp: float = field(default_factory=time.time)
    checksum: str = ""

    def __post_init__(self):
        content = json.dumps(self.plan, sort_keys=True)
        self.checksum = hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "phase": self.phase.value,
            "plan": self.plan,
            "timestamp": self.timestamp,
            "checksum": self.checksum,
        }


class EventBus:
    _instance: Optional["EventBus"] = None
    _instance_lock = threading.Lock()

    def __init__(self):
        self._subscribers: dict[str, list[Callable]] = {}
        self._history: deque[dict] = deque(maxlen=1000)
        self._max_history = 1000

    @classmethod
    def get_instance(cls) -> "EventBus":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = EventBus()
            return cls._instance
        return cls._instance

    def subscribe(self, event_type: str | EventType, callback: Callable) -> None:
        key = event_type.value if isinstance(event_type, EventType) else event_type
        if key not in self._subscribers:
            self._subscribers[key] = []
        self._subscribers[key].append(callback)

    def unsubscribe(self, event_type: str | EventType, callback: Callable) -> None:
        key = event_type.value if isinstance(event_type, EventType) else event_type
        if key in self._subscribers:
            self._subscribers[key] = [cb for cb in self._subscribers[key] if cb != callback]

    async def emit(self, event_type: EventType, data: dict | None = None) -> None:
        event = {
            "type": event_type.value,
            "data": data or {},
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
        }
        self._history.append(event)
        key = event_type.value
        for callback in self._subscribers.get(key, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                logger.debug(f"Event callback error for {key}: {safe_error_message(e)}")

    def get_history(self, event_type: str | None = None, limit: int = 50) -> list[dict]:
        events = list(self._history)
        if event_type:
            events = [e for e in events if e["type"] == event_type]
        return events[-limit:]


class AgenticHarness:
    MAX_STEPS_PER_PLAN = 50
    MAX_DEPENDENCY_DEPTH = 10
    MAX_PLAN_AGE_SECONDS = 86400  # 24 hours

    def __init__(self, agent: Optional["JarvisAgent"] = None):
        self.agent = agent
        self.plans: dict[str, HarnessPlan] = {}
        self.current_plan: HarnessPlan | None = None
        self.phase = HarnessPhase.IDLE
        self.event_bus = EventBus.get_instance()
        self._checkpoints: deque[HarnessCheckpoint] = deque(maxlen=50)
        self._retry_backoff_base = 1.0
        self._retry_backoff_max = 30.0
        self._default_timeout = 120.0
        self._max_parallel = 3
        self._verification_enabled = True
        self._reflection_enabled = True
        self._total_steps_executed = 0
        self._total_tools_called = 0
        self._total_errors = 0
        self._total_retries = 0
        self._start_time: float | None = None
        self._plan_timestamps: dict[str, float] = {}

    def validate_plan(self, plan: HarnessPlan) -> tuple[bool, list[str]]:
        """Validate a plan for correctness before execution.
        
        Checks for: cycles, depth, missing deps, max steps.
        Returns (is_valid, list_of_warnings).
        """
        warnings = []
        errors = []

        if len(plan.steps) > self.MAX_STEPS_PER_PLAN:
            errors.append(f"Plan has {len(plan.steps)} steps (max {self.MAX_STEPS_PER_PLAN})")

        step_ids = {s.id for s in plan.steps}
        for step in plan.steps:
            for dep in step.dependencies:
                if dep not in step_ids:
                    errors.append(f"Step '{step.id}' depends on unknown step '{dep}'")

                    continue

            if step.id in step.dependencies:
                errors.append(f"Step '{step.id}' has self-dependency")

        step_map = {s.id: s for s in plan.steps}
        def get_depth(sid: str, visited: set | None = None) -> int:
            if visited is None:
                visited = set()
            if sid in visited:
                return float('inf')
            visited.add(sid)
            s = step_map.get(sid)
            if not s or not s.dependencies:
                return 0
            max_d = max(get_depth(d, set(visited)) for d in s.dependencies) + 1
            return max_d

        for step in plan.steps:
            depth = get_depth(step.id)
            if depth == float('inf'):
                errors.append(f"Circular dependency detected involving step '{step.id}'")
                break
            if depth > self.MAX_DEPENDENCY_DEPTH:
                warnings.append(f"Step '{step.id}' has dependency depth {depth} (max {self.MAX_DEPENDENCY_DEPTH})")

        for step in plan.steps:
            if step.max_retries < 0:
                errors.append(f"Step '{step.id}' has negative max_retries ({step.max_retries})")
            if step.max_retries > 10:
                warnings.append(f"Step '{step.id}' has high max_retries ({step.max_retries})")
            if step.timeout_seconds < 1:
                errors.append(f"Step '{step.id}' has timeout < 1s ({step.timeout_seconds})")
            if step.timeout_seconds > 3600:
                warnings.append(f"Step '{step.id}' has timeout > 3600s ({step.timeout_seconds})")

        for step in plan.steps:
            if not step.tool_name and not step.action:
                warnings.append(f"Step '{step.id}' has no tool_name or action")

        is_valid = len(errors) == 0
        return is_valid, errors + warnings

    def create_plan(self, objective: str, steps: list[dict] | None = None, max_iterations: int = 10) -> HarnessPlan:
        plan = HarnessPlan(
            objective=objective,
            max_iterations=max_iterations,
        )
        if steps:
            step_ids = []
            for i, step_data in enumerate(steps):
                deps = step_data.get("dependencies", [])
                if isinstance(deps, list) and deps and isinstance(deps[0], int):
                    deps = [step_ids[d] for d in deps if d < len(step_ids)]
                step = HarnessStep(
                    action=step_data.get("action", f"step_{i}"),
                    tool_name=step_data.get("tool_name", ""),
                    tool_params=step_data.get("tool_params", {}),
                    description=step_data.get("description", ""),
                    expected_outcome=step_data.get("expected_outcome", ""),
                    dependencies=deps,
                    max_retries=step_data.get("max_retries", 3),
                    timeout_seconds=step_data.get("timeout_seconds", self._default_timeout),
                    fallback_action=step_data.get("fallback_action"),
                    fallback_params=step_data.get("fallback_params", {}),
                )
                plan.steps.append(step)
                step_ids.append(step.id)

        is_valid, validation_issues = self.validate_plan(plan)
        if not is_valid:
            logger.error(f"Plan validation failed: {validation_issues}")
            for issue in validation_issues:
                if "Circular" in issue or "unknown step" in issue:
                    logger.error(f"  CRITICAL: {issue}")

        self.plans[plan.id] = plan
        self.current_plan = plan
        self._start_time = time.time()
        self._plan_timestamps[plan.id] = time.time()
        return plan

    def estimate_plan_duration(self, plan: HarnessPlan) -> dict[str, float]:
        """Estimate execution duration based on step timeouts and parallelism.
        
        Returns dict with min/max/expected duration estimates in seconds.
        """
        if not plan.steps:
            return {"min": 0.0, "max": 0.0, "expected": 0.0}

        step_map = {s.id: s for s in plan.steps}
        completed_ids = {s.id for s in plan.steps if s.status in (StepStatus.SUCCESS, StepStatus.VERIFIED)}
        remaining = [s for s in plan.steps if s.id not in completed_ids]

        def get_max_depth(sid: str, visited: set | None = None) -> int:
            if visited is None:
                visited = set()
            if sid in visited:
                return 0
            visited.add(sid)
            s = step_map.get(sid)
            if not s or not s.dependencies:
                return 1
            deps_remaining = [d for d in s.dependencies if d not in completed_ids]
            if not deps_remaining:
                return 1
            return 1 + max(get_max_depth(d, visited) for d in deps_remaining)

        levels: dict[int, list[float]] = {}
        for step in remaining:
            depth = get_max_depth(step.id)
            if depth not in levels:
                levels[depth] = []
            levels[depth].append(step.timeout_seconds)

        min_duration = 0.0
        max_duration = 0.0
        expected_duration = 0.0

        for depth in sorted(levels.keys()):
            level_timeouts = levels[depth]
            parallel_time = max(level_timeouts) if level_timeouts else self._default_timeout
            sequential_time = sum(level_timeouts)
            min_duration += parallel_time * 0.3
            max_duration += sequential_time
            expected_duration += parallel_time * 0.6

        return {
            "min": round(min_duration, 1),
            "max": round(max_duration, 1),
            "expected": round(expected_duration, 1),
        }

    def detect_dead_steps(self, plan: HarnessPlan) -> list[str]:
        """Detect steps that can never execute due to failed dependencies.
        
        Returns list of dead step IDs.
        """
        failed_ids = {s.id for s in plan.steps if s.status in (StepStatus.FAILED, StepStatus.VERIFICATION_FAILED)}
        dead_ids = set()

        def check_dead(sid: str, visited: set | None = None) -> bool:
            if visited is None:
                visited = set()
            if sid in visited:
                return True
            visited.add(sid)
            step = next((s for s in plan.steps if s.id == sid), None)
            if not step:
                return False
            if sid in failed_ids:
                return True
            for dep in step.dependencies:
                if dep in failed_ids:
                    return True
                if check_dead(dep, visited):
                    return True
            return False

        for step in plan.steps:
            if step.status == StepStatus.PENDING and check_dead(step.id):
                dead_ids.add(step.id)

        return list(dead_ids)

    def cleanup_expired_plans(self) -> int:
        """Remove plans older than MAX_PLAN_AGE_SECONDS. Returns count removed."""
        now = time.time()
        removed = 0
        for plan_id, ts in list(self._plan_timestamps.items()):
            if now - ts > self.MAX_PLAN_AGE_SECONDS:
                if plan_id in self.plans:
                    del self.plans[plan_id]
                    removed += 1
                del self._plan_timestamps[plan_id]
        if self.current_plan and self.current_plan.id not in self.plans:
            self.current_plan = None
        return removed

    async def execute_plan(self, plan: HarnessPlan | None = None, tools: dict[str, Any] | None = None) -> dict:
        if plan is None:
            plan = self.current_plan
        if plan is None:
            return {"error": "No plan to execute"}

        self.phase = HarnessPhase.EXECUTING
        await self.event_bus.emit(EventType.PLAN_CREATED, plan.to_dict())

        max_iterations = plan.max_iterations
        iteration = 0

        while not plan.is_complete and iteration < max_iterations:
            iteration += 1

            ready_steps = plan.ready_steps
            if not ready_steps:
                pending = [s for s in plan.steps if s.status == StepStatus.PENDING]
                if pending:
                    stuck_deps = [s for s in pending if s.dependencies]
                    logger.warning(f"Plan stuck: {len(stuck_deps)} steps with unmet dependencies")
                    for s in pending:
                        s.status = StepStatus.FAILED
                        s.error = "Unmet dependencies"
                    break
                else:
                    break

            parallel_steps = ready_steps[:self._max_parallel]

            if len(parallel_steps) == 1:
                step = parallel_steps[0]
                result = await self._execute_step(step, tools or {})
                if result is None and step.fallback_action:
                    result = await self._execute_fallback(step, tools or {})
            else:
                tasks = [self._execute_step(s, tools or {}) for s in parallel_steps]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for step, result in zip(parallel_steps, results):
                    if isinstance(result, Exception):
                        step.status = StepStatus.FAILED
                        step.error = safe_error_message(result)
                        if step.fallback_action:
                            await self._execute_fallback(step, tools or {})

            plan.current_step_index = plan.completed_steps

            if self._verification_enabled:
                await self._verify_recent_steps(plan)

            await self.event_bus.emit(EventType.PROGRESS_UPDATE, {
                "plan_id": plan.id,
                "progress": plan.progress,
                "completed": plan.completed_steps,
                "failed": plan.failed_steps,
                "total": len(plan.steps),
            })

            needs_reflection = plan.failed_steps > 0 and self._reflection_enabled and plan.reflection_count < plan.max_reflections
            if needs_reflection:
                adjusted = await self._reflect_and_adjust(plan)
                if not adjusted:
                    break

        self.phase = HarnessPhase.COMPLETED if plan.is_complete else HarnessPhase.FAILED
        await self.event_bus.emit(
            EventType.HARNESS_COMPLETE if plan.is_complete else EventType.HARNESS_FAILED,
            {"plan_id": plan.id, "objective": plan.objective, "progress": plan.progress},
        )

        return plan.to_dict()

    async def _execute_step(self, step: HarnessStep, tools: dict[str, Any]) -> Any:
        step.status = StepStatus.RUNNING
        step.started_at = time.time()
        self._total_steps_executed += 1

        await self.event_bus.emit(EventType.STEP_STARTED, {"step_id": step.id, "action": step.action, "tool": step.tool_name})

        result = await self._execute_with_retry(step, tools)

        step.completed_at = time.time()
        if result is not None and not isinstance(result, Exception):
            step.result = result
            step.status = StepStatus.SUCCESS
            await self.event_bus.emit(EventType.STEP_COMPLETED, {"step_id": step.id, "duration": step.duration})
        else:
            if step.retry_count >= step.max_retries:
                step.status = StepStatus.FAILED
                await self.event_bus.emit(EventType.STEP_FAILED, {"step_id": step.id, "error": step.error})
            return None

        self._total_tools_called += 1
        return result

    async def _execute_with_retry(self, step: HarnessStep, tools: dict[str, Any]) -> Any:
        tool_name = step.tool_name
        if not tool_name or tool_name not in tools:
            step.error = f"Tool '{tool_name}' not available"
            step.status = StepStatus.FAILED
            return None

        tool = tools[tool_name]
        for attempt in range(step.max_retries + 1):
            try:
                await self.event_bus.emit(EventType.TOOL_CALL, {
                    "step_id": step.id,
                    "tool": tool_name,
                    "params": step.tool_params,
                    "attempt": attempt,
                })

                if asyncio.iscoroutinefunction(tool.execute):
                    result = await asyncio.wait_for(
                        tool.execute(**step.tool_params),
                        timeout=step.timeout_seconds,
                    )
                else:
                    loop = asyncio.get_event_loop()
                    result = await asyncio.wait_for(
                        loop.run_in_executor(None, lambda: tool.execute(**step.tool_params)),
                        timeout=step.timeout_seconds,
                    )

                await self.event_bus.emit(EventType.TOOL_RESULT, {
                    "step_id": step.id,
                    "tool": tool_name,
                    "success": True,
                })
                return result

            except asyncio.TimeoutError:
                step.retry_count += 1
                step.error = f"Timeout after {step.timeout_seconds}s (attempt {attempt + 1})"
                self._total_errors += 1
                logger.warning(f"Step {step.id} timed out: {step.error}")
                if attempt < step.max_retries:
                    backoff = min(self._retry_backoff_base * (2 ** attempt), self._retry_backoff_max)
                    await asyncio.sleep(backoff)

            except Exception as e:
                step.retry_count += 1
                step.error = safe_error_message(e)
                self._total_errors += 1
                self._total_retries += 1
                logger.warning(f"Step {step.id} error (attempt {attempt + 1}): {step.error}")
                if attempt < step.max_retries:
                    backoff = min(self._retry_backoff_base * (2 ** attempt), self._retry_backoff_max)
                    await asyncio.sleep(backoff)
                    await self.event_bus.emit(EventType.STEP_RETRIED, {"step_id": step.id, "attempt": attempt + 1})
                else:
                    await self.event_bus.emit(EventType.ERROR_RECOVERY, {"step_id": step.id, "error": step.error, "retries_exhausted": True})

        return None

    async def _execute_fallback(self, step: HarnessStep, tools: dict[str, Any]) -> Any:
        if not step.fallback_action:
            return None
        await self.event_bus.emit(EventType.ERROR_RECOVERY, {
            "step_id": step.id,
            "fallback": step.fallback_action,
        })
        fallback_tool = tools.get(step.fallback_action)
        if not fallback_tool:
            return None
        try:
            if asyncio.iscoroutinefunction(fallback_tool.execute):
                result = await fallback_tool.execute(**step.fallback_params)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: fallback_tool.execute(**step.fallback_params))
            if result is not None and not isinstance(result, Exception):
                step.result = result
                step.status = StepStatus.SUCCESS
                step.completed_at = time.time()
                return result
        except Exception as e:
            logger.warning(f"Fallback for step {step.id} failed: {safe_error_message(e)}")
        return None

    async def _verify_recent_steps(self, plan: HarnessPlan) -> None:
        recent_success = [s for s in plan.steps if s.status == StepStatus.SUCCESS and s.verification_result is None]
        if not recent_success:
            return

        self.phase = HarnessPhase.VERIFYING
        for step in recent_success[-3:]:
            await self.event_bus.emit(EventType.VERIFICATION_STARTED, {"step_id": step.id})
            verified = self._verify_step_result(step)
            if verified:
                step.verification_result = "passed"
                step.status = StepStatus.VERIFIED
                await self.event_bus.emit(EventType.VERIFICATION_PASSED, {"step_id": step.id})
            else:
                step.verification_result = "failed"
                step.status = StepStatus.VERIFICATION_FAILED
                await self.event_bus.emit(EventType.VERIFICATION_FAILED, {"step_id": step.id})

    def _verify_step_result(self, step: HarnessStep) -> bool:
        if step.error is not None:
            return False
        result = step.result
        if isinstance(result, dict):
            if result.get("error"):
                return False
            if result.get("success") is False:
                return False
        if isinstance(result, str) and len(result) < 1:
            return False
        return True

    async def _reflect_and_adjust(self, plan: HarnessPlan) -> bool:
        self.phase = HarnessPhase.REFLECTING
        plan.reflection_count += 1
        await self.event_bus.emit(EventType.REFLECTION_STARTED, {"plan_id": plan.id, "reflection_count": plan.reflection_count})

        failed = [s for s in plan.steps if s.status in (StepStatus.FAILED, StepStatus.VERIFICATION_FAILED)]

        for step in failed:
            if step.retry_count < step.max_retries:
                step.status = StepStatus.PENDING
                step.error = None
            elif step.fallback_action:
                step.status = StepStatus.PENDING
                step.error = None

        adjusted = any(s.status == StepStatus.PENDING for s in failed)
        if adjusted:
            await self.event_bus.emit(EventType.REFLECTION_COMPLETED, {"plan_id": plan.id, "adjustments": len(failed)})
        return adjusted

    def save_checkpoint(self, plan: HarnessPlan | None = None) -> HarnessCheckpoint:
        plan = plan or self.current_plan
        if plan is None:
            raise ValueError("No plan to checkpoint")
        checkpoint = HarnessCheckpoint(
            plan_id=plan.id,
            phase=self.phase,
            plan=plan.to_dict(),
        )
        self._checkpoints.append(checkpoint)
        self.event_bus._history.append({"type": "checkpoint_saved", "data": checkpoint.to_dict(), "timestamp": time.time()})
        return checkpoint

    def restore_checkpoint(self, checkpoint_id: str | None = None) -> HarnessPlan | None:
        if checkpoint_id:
            checkpoint = next((c for c in self._checkpoints if c.plan_id == checkpoint_id), None)
        else:
            checkpoint = self._checkpoints[-1] if self._checkpoints else None
        if not checkpoint:
            return None

        plan_data = checkpoint.plan
        plan = HarnessPlan(
            id=plan_data["id"],
            objective=plan_data["objective"],
            max_iterations=plan_data.get("max_iterations", 10),
            current_step_index=plan_data.get("current_step", 0),
            reflection_count=plan_data.get("reflection_count", 0),
        )
        for step_data in plan_data.get("steps", []):
            status_str = step_data.get("status", "pending")
            try:
                status = StepStatus(status_str)
            except ValueError:
                status = StepStatus.PENDING
            step = HarnessStep(
                id=step_data["id"],
                action=step_data.get("action", ""),
                tool_name=step_data.get("tool_name", ""),
                tool_params=step_data.get("tool_params", {}),
                description=step_data.get("description", ""),
                expected_outcome=step_data.get("expected_outcome", ""),
                dependencies=step_data.get("dependencies", []),
                status=status,
                max_retries=step_data.get("max_retries", 3),
                timeout_seconds=step_data.get("timeout_seconds", 120),
                fallback_action=step_data.get("fallback_action"),
                fallback_params=step_data.get("fallback_params", {}),
            )
            plan.steps.append(step)

        self.plans[plan.id] = plan
        self.current_plan = plan
        self.phase = checkpoint.phase
        return plan

    def get_status(self) -> dict:
        plan = self.current_plan
        status = {
            "phase": self.phase.value,
            "current_plan": plan.to_dict() if plan else None,
            "total_steps_executed": self._total_steps_executed,
            "total_tools_called": self._total_tools_called,
            "total_errors": self._total_errors,
            "total_retries": self._total_retries,
            "checkpoints_saved": len(self._checkpoints),
            "event_history_size": len(self.event_bus._history),
            "uptime_seconds": round(time.time() - self._start_time, 1) if self._start_time else 0,
            "plans_stored": len(self.plans),
        }
        if plan:
            status["plan_duration_estimate"] = self.estimate_plan_duration(plan)
            status["dead_steps"] = self.detect_dead_steps(plan)
        return status


def create_plan_from_objective(objective: str, available_tools: list[str] | None = None) -> list[dict]:
    """Create a plan from an objective using heuristic decomposition.

    This provides a simple plan generator for common task types.
    For complex objectives, the LLM should generate the plan instead.
    """
    steps = []
    objective_lower = objective.lower()

    if any(kw in objective_lower for kw in ["create", "write", "new", "add", "build"]):
        steps = [
            {"action": "research", "description": "Research existing patterns and requirements", "tool_name": "web_search", "tool_params": {"query": objective}, "expected_outcome": "Relevant patterns found"},
            {"action": "plan", "description": "Plan the implementation structure", "tool_name": "read_file", "tool_params": {}, "expected_outcome": "Implementation plan created"},
            {"action": "implement", "description": f"Implement: {objective}", "tool_name": "write_file", "tool_params": {}, "expected_outcome": "Code written successfully"},
            {"action": "verify", "description": "Verify the implementation works", "tool_name": "shell", "tool_params": {"command": "test"}, "expected_outcome": "Tests pass"},
        ]
    elif any(kw in objective_lower for kw in ["fix", "debug", "solve", "resolve", "troubleshoot"]):
        steps = [
            {"action": "investigate", "description": "Investigate the issue", "tool_name": "shell", "tool_params": {}, "expected_outcome": "Issue identified"},
            {"action": "diagnose", "description": "Diagnose root cause", "tool_name": "read_file", "tool_params": {}, "expected_outcome": "Root cause found"},
            {"action": "fix", "description": f"Fix: {objective}", "tool_name": "write_file", "tool_params": {}, "expected_outcome": "Fix applied"},
            {"action": "verify", "description": "Verify the fix works", "tool_name": "shell", "tool_params": {"command": "test"}, "expected_outcome": "Fix verified"},
        ]
    elif any(kw in objective_lower for kw in ["analyze", "examine", "review", "investigate", "understand"]):
        steps = [
            {"action": "gather", "description": "Gather relevant information", "tool_name": "read_file", "tool_params": {}, "expected_outcome": "Information gathered"},
            {"action": "analyze", "description": f"Analyze: {objective}", "tool_name": "shell", "tool_params": {}, "expected_outcome": "Analysis complete"},
            {"action": "report", "description": "Generate analysis report", "tool_name": "write_file", "tool_params": {}, "expected_outcome": "Report generated"},
        ]
    else:
        steps = [
            {"action": "understand", "description": f"Understand objective: {objective}", "tool_name": "read_file", "tool_params": {}, "expected_outcome": "Objective understood"},
            {"action": "execute", "description": f"Execute: {objective}", "tool_name": "shell", "tool_params": {}, "expected_outcome": "Task executed"},
            {"action": "verify", "description": "Verify result", "tool_name": "shell", "tool_params": {}, "expected_outcome": "Result verified"},
        ]

    if available_tools:
        for i, step in enumerate(steps):
            if step["tool_name"] not in available_tools:
                fallback_map = {
                    "web_search": "shell",
                    "read_file": "shell",
                    "write_file": "shell",
                    "shell": "read_file",
                }
                step["tool_name"] = fallback_map.get(step["tool_name"], available_tools[0] if available_tools else "shell")

    for i in range(1, len(steps)):
        steps[i]["dependencies"] = [steps[i - 1]["action"]]

    return steps


async def llm_generate_plan(objective: str, llm_client: Any = None, available_tools: list[str] | None = None) -> list[dict]:
    """Generate an execution plan using LLM decomposition.

    Uses the LLM to decompose the objective into structured steps with
    proper tool selection, dependencies, and verification criteria.

    Falls back to heuristic plan generation if LLM is unavailable.
    """
    if llm_client is None:
        return create_plan_from_objective(objective, available_tools)

    tools_list = ", ".join(available_tools) if available_tools else "shell, read_file, write_file"

    prompt = f"""Decompose this objective into a structured execution plan with 3-7 steps.

Objective: {objective}

Available tools: {tools_list}

Return a JSON array of steps. Each step must have:
- "action": short identifier (e.g., "investigate", "implement", "verify")
- "tool_name": one of the available tools
- "tool_params": object with tool parameters (can be {{}} to fill later)
- "description": what this step does
- "expected_outcome": what success looks like
- "dependencies": array of action names this step depends on (earlier step actions)
- "max_retries": number (default 3)
- "timeout_seconds": number (default 120)

Example:
```json
[
  {{"action": "investigate", "tool_name": "shell", "tool_params": {{"command": "find . -name '*.py'"}}, "description": "Find relevant files", "expected_outcome": "Files identified", "dependencies": [], "max_retries": 2, "timeout_seconds": 30}},
  {{"action": "implement", "tool_name": "write_file", "tool_params": {{}}, "description": "Write the implementation", "expected_outcome": "Code written", "dependencies": ["investigate"], "max_retries": 3, "timeout_seconds": 120}},
  {{"action": "verify", "tool_name": "shell", "tool_params": {{"command": "python -m pytest"}}, "description": "Run tests", "expected_outcome": "Tests pass", "dependencies": ["implement"], "max_retries": 2, "timeout_seconds": 60}}
]
```

Return ONLY the JSON array, no other text."""

    try:
        if hasattr(llm_client, 'chat'):
            response = await asyncio.to_thread(
                llm_client.chat,
                messages=[{"role": "user", "content": prompt}],
            )
        elif hasattr(llm_client, 'generate'):
            response = await asyncio.to_thread(
                llm_client.generate,
                prompt,
            )
        else:
            return create_plan_from_objective(objective, available_tools)

        text = response if isinstance(response, str) else str(response)
        json_match = re.search(r'\[.*\]', text, re.DOTALL)
        if not json_match:
            return create_plan_from_objective(objective, available_tools)

        steps = json.loads(json_match.group())
        if not isinstance(steps, list) or len(steps) == 0:
            return create_plan_from_objective(objective, available_tools)

        validated = []
        for i, step in enumerate(steps):
            if not isinstance(step, dict):
                continue
            validated.append({
                "action": str(step.get("action", f"step_{i}"))[:50],
                "tool_name": str(step.get("tool_name", "shell"))[:30],
                "tool_params": step.get("tool_params", {}),
                "description": str(step.get("description", ""))[:200],
                "expected_outcome": str(step.get("expected_outcome", ""))[:200],
                "dependencies": step.get("dependencies", []) if isinstance(step.get("dependencies"), list) else [],
                "max_retries": max(0, min(10, int(step.get("max_retries", 3)))),
                "timeout_seconds": max(1, min(3600, float(step.get("timeout_seconds", 120)))),
            })

        if not validated:
            return create_plan_from_objective(objective, available_tools)

        return validated

    except Exception as e:
        logger.warning(f"LLM plan generation failed: {safe_error_message(e)}, falling back to heuristic")
        return create_plan_from_objective(objective, available_tools)


_harness_singleton: AgenticHarness | None = None
_harness_singleton_lock = threading.Lock()


def get_agentic_harness(agent: Optional["JarvisAgent"] = None) -> AgenticHarness:
    """Thread-safe process-wide AgenticHarness.

    REST endpoints used to construct a fresh ``AgenticHarness()`` per
    request, so plans created by /api/harness/plan were invisible to
    /api/harness/execute and checkpoints never accumulated. One shared
    instance makes the endpoint family coherent; ``agent`` late-binds
    like the adaptive-loop singleton.
    """
    global _harness_singleton
    if _harness_singleton is None:
        with _harness_singleton_lock:
            if _harness_singleton is None:
                _harness_singleton = AgenticHarness(agent=agent)
            return _harness_singleton
    if _harness_singleton.agent is None and agent is not None:
        with _harness_singleton_lock:
            if _harness_singleton.agent is None:
                _harness_singleton.agent = agent
    return _harness_singleton


def reset_agentic_harness() -> None:
    """Drop the singleton (tests / reconfiguration)."""
    global _harness_singleton
    with _harness_singleton_lock:
        _harness_singleton = None