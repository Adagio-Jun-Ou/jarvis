"""Self-Correction Engine for Autonomous Agent Operations.

Provides intelligent error recovery, goal decomposition, and autonomous
self-improvement for long-running agentic deployments.

Key features:
- Goal decomposition into executable steps
- Error analysis and recovery strategies
- Automatic retry with strategy modification
- Learning from failures
- Progress tracking and checkpointing
- Strategy adaptation based on outcomes

Example:
    engine = get_self_correction_engine()
    
    # Decompose a goal
    plan = engine.decompose_goal("Analyze all Python files and create a report")
    
    # Execute with auto-recovery
    result = await engine.execute_with_recovery(
        goal="Deploy the application to production",
        max_retries=5,
        recovery_strategies=["retry", "rollback", "alternative"],
    )
    
    # Learn from failure
    engine.record_outcome(
        goal_id="goal_123",
        success=False,
        error="connection timeout",
        strategy_used="retry",
    )
"""
import json
from core.helpers import atomic_write_json, safe_error_message, read_json_with_integrity
import inspect
import logging
import re
import secrets
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable
from collections import defaultdict

logger = logging.getLogger(__name__)


class GoalStatus(Enum):
    PENDING = "pending"
    DECOMPOSING = "decomposing"
    EXECUTING = "executing"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRY = "retry"


class RecoveryStrategy(Enum):
    RETRY = "retry"
    ROLLBACK = "rollback"
    ALTERNATIVE = "alternative"
    ESCALATE = "escalate"
    SKIP = "skip"
    DECOMPOSE_FURTHER = "decompose_further"


class ErrorType(Enum):
    TRANSIENT = "transient"
    PERMANENT = "permanent"
    RESOURCE = "resource"
    PERMISSION = "permission"
    TIMEOUT = "timeout"
    VALIDATION = "validation"
    DEPENDENCY = "dependency"
    UNKNOWN = "unknown"


@dataclass
class ErrorPattern:
    pattern: str
    error_type: ErrorType
    recovery_strategies: list[RecoveryStrategy]
    suggested_action: str = ""


@dataclass
class ExecutionStep:
    step_id: str
    name: str
    description: str
    action: str
    params: dict[str, Any] = field(default_factory=dict)
    status: StepStatus = StepStatus.PENDING
    dependencies: list[str] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: float = 300.0
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    rollback_action: str | None = None
    alternative_action: str | None = None

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "name": self.name,
            "description": self.description,
            "action": self.action,
            "params": self.params,
            "status": self.status.value,
            "dependencies": self.dependencies,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "timeout_seconds": self.timeout_seconds,
            "result": self.result,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "rollback_action": self.rollback_action,
            "alternative_action": self.alternative_action,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExecutionStep":
        return cls(
            step_id=data["step_id"],
            name=data["name"],
            description=data.get("description", ""),
            action=data["action"],
            params=data.get("params", {}),
            status=StepStatus(data.get("status", "pending")),
            dependencies=data.get("dependencies", []),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3),
            timeout_seconds=data.get("timeout_seconds", 300.0),
            result=data.get("result", {}),
            error=data.get("error"),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            rollback_action=data.get("rollback_action"),
            alternative_action=data.get("alternative_action"),
        )


@dataclass
class ExecutionPlan:
    goal_id: str
    goal: str
    steps: list[ExecutionStep]
    context: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    current_step_idx: int = 0
    rollback_stack: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "goal_id": self.goal_id,
            "goal": self.goal,
            "steps": [s.to_dict() for s in self.steps],
            "context": self.context,
            "created_at": self.created_at.isoformat(),
            "current_step_idx": self.current_step_idx,
            "rollback_stack": self.rollback_stack,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExecutionPlan":
        return cls(
            goal_id=data["goal_id"],
            goal=data["goal"],
            steps=[ExecutionStep.from_dict(s) for s in data.get("steps", [])],
            context=data.get("context", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            current_step_idx=data.get("current_step_idx", 0),
            rollback_stack=data.get("rollback_stack", []),
        )

    def get_ready_steps(self) -> list[ExecutionStep]:
        completed_ids = {s.step_id for s in self.steps if s.status == StepStatus.COMPLETED}
        ready = []
        for step in self.steps:
            if step.status != StepStatus.PENDING:
                continue
            if all(dep in completed_ids for dep in step.dependencies):
                ready.append(step)
        return ready

    def get_progress(self) -> float:
        if not self.steps:
            return 0.0
        completed = sum(1 for s in self.steps if s.status == StepStatus.COMPLETED)
        return completed / len(self.steps) * 100


@dataclass
class OutcomeRecord:
    goal_id: str
    goal: str
    success: bool
    error: str | None
    error_type: ErrorType | None
    strategy_used: RecoveryStrategy | None
    recovery_successful: bool
    steps_completed: int
    steps_total: int
    duration_seconds: float
    timestamp: datetime = field(default_factory=datetime.now)
    learned_lessons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "goal_id": self.goal_id,
            "goal": self.goal,
            "success": self.success,
            "error": self.error,
            "error_type": self.error_type.value if self.error_type else None,
            "strategy_used": self.strategy_used.value if self.strategy_used else None,
            "recovery_successful": self.recovery_successful,
            "steps_completed": self.steps_completed,
            "steps_total": self.steps_total,
            "duration_seconds": self.duration_seconds,
            "timestamp": self.timestamp.isoformat(),
            "learned_lessons": self.learned_lessons,
        }


class SelfCorrectionEngine:
    MAX_PLAN_AGE_HOURS = 72
    MAX_OUTCOMES = 1000
    MAX_ACTIVE_PLANS = 50
    OUTCOME_LEARNING_WINDOW_DAYS = 30
    
    def __init__(self, storage_path: Path | None = None, llm_client=None):
        self.storage_path = storage_path or Path.home() / ".jarvis" / "self_correction"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.llm = llm_client
        self._plans: dict[str, ExecutionPlan] = {}
        self._outcomes: list[OutcomeRecord] = []
        self._error_patterns: list[ErrorPattern] = self._init_error_patterns()
        self._action_handlers: dict[str, Callable] = {}
        self._error_learning: dict[str, dict] = {}
        self._strategy_stats: dict[str, dict] = {}
        
        self._lock = threading.RLock()
        self._max_outcomes = self.MAX_OUTCOMES
        self._learning_window_days = self.OUTCOME_LEARNING_WINDOW_DAYS
        
        self._load_state()

    def _init_error_patterns(self) -> list[ErrorPattern]:
        return [
            ErrorPattern(
                pattern=r"timeout|timed out|time out",
                error_type=ErrorType.TIMEOUT,
                recovery_strategies=[RecoveryStrategy.RETRY, RecoveryStrategy.ALTERNATIVE],
                suggested_action="Increase timeout or use alternative approach",
            ),
            ErrorPattern(
                pattern=r"connection refused|network unreachable|connection reset",
                error_type=ErrorType.TRANSIENT,
                recovery_strategies=[RecoveryStrategy.RETRY, RecoveryStrategy.ROLLBACK],
                suggested_action="Wait and retry, check network connectivity",
            ),
            ErrorPattern(
                pattern=r"permission denied|access denied|forbidden|unauthorized",
                error_type=ErrorType.PERMISSION,
                recovery_strategies=[RecoveryStrategy.ESCALATE],
                suggested_action="Check permissions or request access",
            ),
            ErrorPattern(
                pattern=r"not found|does not exist|no such file",
                error_type=ErrorType.PERMANENT,
                recovery_strategies=[RecoveryStrategy.ALTERNATIVE, RecoveryStrategy.SKIP],
                suggested_action="Verify resource exists or use alternative",
            ),
            ErrorPattern(
                pattern=r"out of memory|cannot allocate|resource exhausted",
                error_type=ErrorType.RESOURCE,
                recovery_strategies=[RecoveryStrategy.ROLLBACK, RecoveryStrategy.DECOMPOSE_FURTHER],
                suggested_action="Free resources or break into smaller steps",
            ),
            ErrorPattern(
                pattern=r"invalid|validation failed|schema error",
                error_type=ErrorType.VALIDATION,
                recovery_strategies=[RecoveryStrategy.ALTERNATIVE],
                suggested_action="Check input format and constraints",
            ),
            ErrorPattern(
                pattern=r"dependency|prerequisite|require",
                error_type=ErrorType.DEPENDENCY,
                recovery_strategies=[RecoveryStrategy.ROLLBACK, RecoveryStrategy.DECOMPOSE_FURTHER],
                suggested_action="Install or satisfy dependency first",
            ),
        ]

    def _load_state(self) -> None:
        plans_file = self.storage_path / "plans.json"
        if plans_file.exists():
            try:
                data = read_json_with_integrity(plans_file)
                for plan_data in data.get("plans", []):
                    plan = ExecutionPlan.from_dict(plan_data)
                    self._plans[plan.goal_id] = plan
                logger.info(f"Loaded {len(self._plans)} execution plans")
            except Exception as e:
                logger.error(f"Failed to load plans: {e}")

        outcomes_file = self.storage_path / "outcomes.json"
        if outcomes_file.exists():
            try:
                data = read_json_with_integrity(outcomes_file)
                for rec in data.get("outcomes", []):
                    outcome = OutcomeRecord(
                        goal_id=rec["goal_id"],
                        goal=rec["goal"],
                        success=rec["success"],
                        error=rec.get("error"),
                        error_type=ErrorType(rec["error_type"]) if rec.get("error_type") else None,
                        strategy_used=RecoveryStrategy(rec["strategy_used"]) if rec.get("strategy_used") else None,
                        recovery_successful=rec.get("recovery_successful", False),
                        steps_completed=rec.get("steps_completed", 0),
                        steps_total=rec.get("steps_total", 0),
                        duration_seconds=rec.get("duration_seconds", 0),
                        timestamp=datetime.fromisoformat(rec["timestamp"]) if rec.get("timestamp") else datetime.now(),
                        learned_lessons=rec.get("learned_lessons", []),
                    )
                    self._outcomes.append(outcome)
                logger.info(f"Loaded {len(self._outcomes)} outcome records")
            except Exception as e:
                logger.error(f"Failed to load outcomes: {e}")

    def _save_state(self) -> None:
        try:
            atomic_write_json(self.storage_path / "plans.json", {
                    "plans": [p.to_dict() for p in self._plans.values()],
                })

            atomic_write_json(self.storage_path / "outcomes.json", {
                    "outcomes": [o.to_dict() for o in self._outcomes[-self._max_outcomes:]],
                })
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def register_handler(self, action: str, handler: Callable) -> None:
        self._action_handlers[action] = handler

    def decompose_goal(
        self,
        goal: str,
        context: dict[str, Any] | None = None,
        max_steps: int = 10,
    ) -> ExecutionPlan:
        """Decompose a goal synchronously (heuristics first, sync LLMs only).

        Jarvis's OllamaClient is async and cannot be awaited from this sync
        entry point. Historically the un-awaited coroutine crashed the
        response parse and the failure was swallowed, silently making LLM
        decomposition a permanent no-op. Async clients are now detected and
        skipped cleanly; use :meth:`decompose_goal_async` for LLM-driven
        decomposition.
        """
        goal_id = f"goal_{secrets.token_hex(8)}"

        steps = self._decompose_with_heuristics(goal, context)

        if len(steps) < max_steps and self.llm:
            llm_steps = self._decompose_with_llm_sync(goal, context)
            if llm_steps:
                steps = llm_steps

        return self._finalize_plan(goal_id, goal, steps, max_steps, context)

    async def decompose_goal_async(
        self,
        goal: str,
        context: dict[str, Any] | None = None,
        max_steps: int = 10,
    ) -> ExecutionPlan:
        """Decompose a goal, using an LLM when one is attached.

        Supports async clients (the real OllamaClient) and routes the call
        through the shared api_manager, so background goal decomposition is
        quota'd and cached like every other subsystem's LLM spend.
        """
        goal_id = f"goal_{secrets.token_hex(8)}"

        steps = self._decompose_with_heuristics(goal, context)

        if len(steps) < max_steps and self.llm:
            llm_steps = await self._decompose_with_llm(goal, context)
            if llm_steps:
                steps = llm_steps

        return self._finalize_plan(goal_id, goal, steps, max_steps, context)

    def _finalize_plan(
        self,
        goal_id: str,
        goal: str,
        steps: list[ExecutionStep],
        max_steps: int,
        context: dict[str, Any] | None,
    ) -> ExecutionPlan:
        plan = ExecutionPlan(
            goal_id=goal_id,
            goal=goal,
            steps=steps[:max_steps],
            context=context or {},
        )

        with self._lock:
            self._plans[goal_id] = plan
            self._save_state()

        logger.info(f"Decomposed goal '{goal}' into {len(plan.steps)} steps")
        return plan

    def _decompose_with_heuristics(self, goal: str, context: dict[str, Any] | None) -> list[ExecutionStep]:
        steps = []
        goal_lower = goal.lower()
        
        if any(kw in goal_lower for kw in ["and", "then", "after", "before"]):
            parts = re.split(r'\s+(?:and|then|after)\s+', goal, flags=re.IGNORECASE)
            for i, part in enumerate(parts):
                if not part.strip():
                    continue
                step_id = f"step_{i+1}"
                steps.append(ExecutionStep(
                    step_id=step_id,
                    name=part.strip()[:50],
                    description=part.strip(),
                    action=self._infer_action(part.strip()),
                    dependencies=[f"step_{i}"] if i > 0 else [],
                ))
        
        if not steps:
            patterns = [
                (r"analyze\s+(.+)", "analyze", "Analyze {0}"),
                (r"create\s+(.+)", "create", "Create {0}"),
                (r"update\s+(.+)", "update", "Update {0}"),
                (r"delete\s+(.+)", "delete", "Delete {0}"),
                (r"search\s+(?:for\s+)?(.+)", "search", "Search for {0}"),
                (r"deploy\s+(.+)", "deploy", "Deploy {0}"),
                (r"test\s+(.+)", "test", "Test {0}"),
                (r"run\s+(.+)", "run", "Run {0}"),
                (r"build\s+(.+)", "build", "Build {0}"),
            ]
            
            for pattern, action, name_template in patterns:
                match = re.search(pattern, goal_lower)
                if match:
                    target = match.group(1)
                    steps.append(ExecutionStep(
                        step_id="step_1",
                        name=name_template.format(target),
                        description=goal,
                        action=action,
                        params={"target": target},
                    ))
                    break
        
        if not steps:
            steps.append(ExecutionStep(
                step_id="step_1",
                name=goal[:50],
                description=goal,
                action="execute",
                params={"goal": goal},
            ))
        
        return steps

    _DECOMPOSITION_SYSTEM_PROMPT = "You are a task decomposition engine. Respond only with valid JSON."

    async def _coordinated_chat(self, prompt: str) -> dict | None:
        """LLM call through the shared api_manager (cache + quota), with a
        direct-call fallback only if the manager itself is unavailable.

        Returns the chat response dict on success, or None when the call was
        refused (rate limit / queue full / timeout / LLM error).
        """
        try:
            from core.api_manager import get_api_manager, Priority as APIPriority, RequestType

            manager = get_api_manager(self.llm)
            response = await manager.submit(
                subsystem="self_correction",
                priority=APIPriority.LEARNING,
                user_input=prompt,
                system_prompt=self._DECOMPOSITION_SYSTEM_PROMPT,
                request_type=RequestType.CHAT,
                timeout=60.0,
            )
            if response.success:
                return response.content
            logger.info(
                f"[self_correction] coordinated LLM call refused: {response.error}"
            )
            return None
        except Exception as e:
            logger.debug(f"api_manager unavailable, direct LLM call: {e}")
            # Direct fallback: honour both client shapes this engine has
            # historically accepted.
            if hasattr(self.llm, 'chat'):
                response = await self.llm.chat(
                    user_input=prompt,
                    system_prompt=self._DECOMPOSITION_SYSTEM_PROMPT,
                )
                return response
            if hasattr(self.llm, 'generate'):
                result = self.llm.generate(prompt, max_tokens=1000)
                if inspect.isawaitable(result):
                    result = await result
                return {"message": {"content": str(result)}}
            return None

    def _parse_step_json(self, response_text) -> list[ExecutionStep] | None:
        """Parse an LLM response into execution steps; None if unparseable."""
        try:
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if not json_match:
                return None
            data = json.loads(json_match.group())
            steps = []
            for i, item in enumerate(data):
                steps.append(ExecutionStep(
                    step_id=f"step_{i+1}",
                    name=item.get("name", f"Step {i+1}"),
                    description=item.get("description", ""),
                    action=item.get("action", "execute"),
                    params=item.get("params", {}),
                    dependencies=item.get("dependencies", []),
                ))
            return steps
        except Exception as e:
            logger.debug(f"LLM decomposition parse failed: {e}")
            return None

    def _decomposition_prompt(self, goal: str, context: dict[str, Any] | None) -> str:
        return f"""Break down this goal into executable steps.

Goal: {goal}

Context: {json.dumps(context or {}, indent=2)[:500]}

Output as JSON array:
[{{"name": "...", "description": "...", "action": "...", "params": {{}}, "dependencies": []}}]

Steps:"""

    async def _decompose_with_llm(self, goal: str, context: dict[str, Any] | None) -> list[ExecutionStep] | None:
        """LLM decomposition for async clients, via the api_manager."""
        if not self.llm:
            return None

        try:
            response = await self._coordinated_chat(self._decomposition_prompt(goal, context))
            if response is None:
                return None
            if isinstance(response, dict):
                response = response.get("message", {}).get("content", "")
            return self._parse_step_json(response)
        except Exception as e:
            logger.debug(f"LLM decomposition failed: {e}")
            return None

    def _decompose_with_llm_sync(self, goal: str, context: dict[str, Any] | None) -> list[ExecutionStep] | None:
        """LLM decomposition for sync decompose_goal — sync LLMs only.

        With an async client (Jarvis's OllamaClient), the old code called
        ``chat(...)`` synchronously, fed the un-awaited coroutine into
        ``re.search`` — TypeError — and the broad except swallowed it, so
        LLM decomposition silently never worked and the coroutine leaked.
        Async responses are now detected and closed cleanly; async clients
        get decomposition via decompose_goal_async instead.
        """
        if not self.llm:
            return None

        try:
            if hasattr(self.llm, 'generate'):
                response = self.llm.generate(self._decomposition_prompt(goal, context), max_tokens=1000)
            elif hasattr(self.llm, 'chat'):
                response = self.llm.chat([{"role": "user", "content": self._decomposition_prompt(goal, context)}])
            else:
                return None

            if inspect.isawaitable(response):
                # Async client from a sync entry point: cannot proceed.
                # close() avoids the 'coroutine was never awaited' warning.
                response.close()
                logger.debug(
                    "self_correction: async LLM client cannot be used from "
                    "sync decompose_goal; use decompose_goal_async"
                )
                return None

            if isinstance(response, dict):
                response = response.get("content", "")

            return self._parse_step_json(response)
        except Exception as e:
            logger.debug(f"LLM decomposition failed: {e}")
            return None

    def _infer_action(self, description: str) -> str:
        desc_lower = description.lower()
        action_map = {
            "analyze": ["analyze", "examine", "investigate", "study"],
            "create": ["create", "make", "generate", "build", "write"],
            "update": ["update", "modify", "change", "edit", "update"],
            "delete": ["delete", "remove", "clear", "drop"],
            "search": ["search", "find", "lookup", "query"],
            "deploy": ["deploy", "publish", "release", "ship"],
            "test": ["test", "verify", "check", "validate"],
            "run": ["run", "execute", "start", "launch"],
        }
        
        for action, keywords in action_map.items():
            if any(kw in desc_lower for kw in keywords):
                return action
        
        return "execute"

    def classify_error(self, error: str) -> ErrorType:
        error_lower = error.lower()
        for pattern in self._error_patterns:
            if re.search(pattern.pattern, error_lower, re.IGNORECASE):
                return pattern.error_type
        return ErrorType.UNKNOWN

    def get_recovery_strategies(self, error: str) -> list[RecoveryStrategy]:
        error_lower = error.lower()
        for pattern in self._error_patterns:
            if re.search(pattern.pattern, error_lower, re.IGNORECASE):
                return pattern.recovery_strategies
        return [RecoveryStrategy.RETRY, RecoveryStrategy.ESCALATE]

    def get_suggested_action(self, error: str) -> str:
        error_lower = error.lower()
        for pattern in self._error_patterns:
            if re.search(pattern.pattern, error_lower, re.IGNORECASE):
                return pattern.suggested_action
        return "Review error and determine appropriate action"

    async def execute_step(
        self,
        step: ExecutionStep,
        plan: ExecutionPlan,
    ) -> tuple[bool, dict[str, Any]]:
        handler = self._action_handlers.get(step.action)
        
        if handler:
            try:
                import asyncio
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(step.params, plan.context)
                else:
                    result = handler(step.params, plan.context)
                return True, result if isinstance(result, dict) else {"output": str(result)}
            except Exception as e:
                return False, {"error": safe_error_message(e)}
        
        return False, {"error": f"No handler for action: {step.action}"}

    async def execute_with_recovery(
        self,
        goal: str,
        max_retries: int = 3,
        recovery_strategies: list[str] | None = None,
        context: dict[str, Any] | None = None,
        on_progress: Callable | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        # Async decomposition: this is the production entry point
        # (agentic_orchestrator), so LLM-driven decomposition actually works
        # here and is quota'd through the api_manager.
        plan = await self.decompose_goal_async(goal, context)
        start_time = datetime.now()
        
        max_loop_iterations = len(plan.steps) * 3 + 1
        loop_count = 0
        
        while True:
            loop_count += 1
            if loop_count > max_loop_iterations:
                logger.error(f"Self-correction loop exceeded max iterations ({max_loop_iterations}), aborting")
                return False, {"error": f"Execution loop exceeded max iterations ({max_loop_iterations})", "plan": plan.to_dict()}
            
            ready_steps = plan.get_ready_steps()
            
            if not ready_steps:
                all_done = all(s.status in (StepStatus.COMPLETED, StepStatus.SKIPPED) for s in plan.steps)
                any_failed = any(s.status == StepStatus.FAILED for s in plan.steps)
                
                if all_done:
                    duration = (datetime.now() - start_time).total_seconds()
                    outcome = OutcomeRecord(
                        goal_id=plan.goal_id,
                        goal=goal,
                        success=not any_failed,
                        error=None if not any_failed else "Some steps failed",
                        error_type=None,
                        strategy_used=None,
                        recovery_successful=False,
                        steps_completed=sum(1 for s in plan.steps if s.status == StepStatus.COMPLETED),
                        steps_total=len(plan.steps),
                        duration_seconds=duration,
                    )
                    with self._lock:
                        self._outcomes.append(outcome)
                        self._save_state()
                    
                    if on_progress:
                        on_progress(plan.get_progress(), "completed", None)
                    
                    return not any_failed, {"plan": plan.to_dict()}
                
                if any_failed:
                    failed_step = next(s for s in plan.steps if s.status == StepStatus.FAILED)
                    success, new_plan = await self._handle_failure(
                        plan, failed_step, recovery_strategies
                    )
                    if not success:
                        return False, {"error": failed_step.error, "plan": plan.to_dict()}
                    if new_plan:
                        plan = new_plan
                continue
            
            for step in ready_steps:
                step.status = StepStatus.RUNNING
                step.started_at = datetime.now()
                
                if on_progress:
                    on_progress(plan.get_progress(), "executing", step.name)
                
                success, result = await self.execute_step(step, plan)
                
                step.completed_at = datetime.now()
                step.result = result
                
                if success:
                    step.status = StepStatus.COMPLETED
                    plan.rollback_stack.append(step.step_id)
                else:
                    step.error = result.get("error", "Unknown error")
                    step.status = StepStatus.FAILED
                    
                    if step.retry_count < step.max_retries:
                        step.retry_count += 1
                        step.status = StepStatus.RETRY
                    
                    self._save_state()
                    break
                
                self._save_state()

    async def _handle_failure(
        self,
        plan: ExecutionPlan,
        failed_step: ExecutionStep,
        allowed_strategies: list[str] | None,
    ) -> tuple[bool, ExecutionPlan | None]:
        error = failed_step.error or "Unknown error"
        self.classify_error(error)
        strategies = self.get_recovery_strategies(error)
        
        if allowed_strategies:
            strategies = [s for s in strategies if s.value in allowed_strategies]
        
        for strategy in strategies:
            if strategy == RecoveryStrategy.RETRY:
                failed_step.status = StepStatus.PENDING
                failed_step.retry_count += 1
                return True, None
            
            elif strategy == RecoveryStrategy.ROLLBACK:
                if failed_step.rollback_action:
                    rollback_step = ExecutionStep(
                        step_id=f"rollback_{secrets.token_hex(4)}",
                        name=f"Rollback: {failed_step.name}",
                        description=f"Rollback for {failed_step.step_id}",
                        action=failed_step.rollback_action,
                        params=failed_step.params,
                    )
                    plan.steps.append(rollback_step)
                    failed_step.status = StepStatus.PENDING
                    return True, plan
            
            elif strategy == RecoveryStrategy.ALTERNATIVE:
                if failed_step.alternative_action:
                    failed_step.action = failed_step.alternative_action
                    failed_step.status = StepStatus.PENDING
                    return True, None
            
            elif strategy == RecoveryStrategy.DECOMPOSE_FURTHER:
                sub_steps = self._decompose_with_heuristics(failed_step.description, plan.context)
                if sub_steps:
                    for i, sub in enumerate(sub_steps):
                        sub.step_id = f"{failed_step.step_id}_{i+1}"
                        sub.dependencies = [failed_step.step_id] if i > 0 else [d for d in failed_step.dependencies if d in {s.step_id for s in plan.steps}]
                    plan.steps.extend(sub_steps)
                    failed_step.status = StepStatus.SKIPPED
                    return True, plan
            
            elif strategy == RecoveryStrategy.SKIP:
                failed_step.status = StepStatus.SKIPPED
                return True, None
            
            elif strategy == RecoveryStrategy.ESCALATE:
                return False, None
        
        return False, None

    def record_outcome(
        self,
        goal_id: str,
        success: bool,
        error: str | None = None,
        strategy_used: RecoveryStrategy | None = None,
        recovery_successful: bool = False,
        lessons: list[str] | None = None,
    ) -> None:
        plan = self._plans.get(goal_id)
        if not plan:
            return
        
        error_type = self.classify_error(error) if error else None
        
        outcome = OutcomeRecord(
            goal_id=goal_id,
            goal=plan.goal,
            success=success,
            error=error,
            error_type=error_type,
            strategy_used=strategy_used,
            recovery_successful=recovery_successful,
            steps_completed=sum(1 for s in plan.steps if s.status == StepStatus.COMPLETED),
            steps_total=len(plan.steps),
            duration_seconds=0,
            learned_lessons=lessons or [],
        )
        
        with self._lock:
            self._outcomes.append(outcome)
            self._save_state()

    def get_success_rate_by_strategy(self) -> dict[str, float]:
        strategy_stats = defaultdict(lambda: {"success": 0, "total": 0})
        
        for outcome in self._outcomes:
            if outcome.strategy_used:
                key = outcome.strategy_used.value
                strategy_stats[key]["total"] += 1
                if outcome.recovery_successful:
                    strategy_stats[key]["success"] += 1
        
        return {
            strategy: stats["success"] / stats["total"] if stats["total"] > 0 else 0.0
            for strategy, stats in strategy_stats.items()
        }

    def get_common_errors(self, limit: int = 10) -> list[tuple[str, ErrorType, int]]:
        error_counts = defaultdict(lambda: {"count": 0, "type": ErrorType.UNKNOWN})
        
        for outcome in self._outcomes:
            if outcome.error:
                key = outcome.error[:100]
                error_counts[key]["count"] += 1
                error_counts[key]["type"] = outcome.error_type or ErrorType.UNKNOWN
        
        sorted_errors = sorted(error_counts.items(), key=lambda x: -x[1]["count"])
        return [(e[0], e[1]["type"], e[1]["count"]) for e in sorted_errors[:limit]]

    def get_plan(self, goal_id: str) -> ExecutionPlan | None:
        return self._plans.get(goal_id)

    def list_plans(self, status: GoalStatus | None = None) -> list[ExecutionPlan]:
        with self._lock:
            plans = list(self._plans.values())
            if status:
                plans = [p for p in plans if any(s.status.value == status.value for s in p.steps)]
            return plans

    def cancel_plan(self, goal_id: str) -> bool:
        with self._lock:
            plan = self._plans.get(goal_id)
            if not plan:
                return False
            
            for step in plan.steps:
                if step.status == StepStatus.RUNNING:
                    step.status = StepStatus.FAILED
                    step.error = "Cancelled"
            
            self._save_state()
            return True

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            total_goals = len(self._plans)
            total_outcomes = len(self._outcomes)
            
            success_count = sum(1 for o in self._outcomes if o.success)
            
            error_type_counts = defaultdict(int)
            for o in self._outcomes:
                if o.error_type:
                    error_type_counts[o.error_type.value] += 1
            
            return {
                "total_goals": total_goals,
                "total_outcomes": total_outcomes,
                "success_rate": success_count / total_outcomes if total_outcomes > 0 else 0.0,
                "by_error_type": dict(error_type_counts),
                "strategy_success_rates": self.get_success_rate_by_strategy(),
            }

    def cleanup_expired_plans(self) -> int:
        """Remove plans older than MAX_PLAN_AGE_HOURS. Returns count removed."""
        now = datetime.now()
        cutoff = now - timedelta(hours=self.MAX_PLAN_AGE_HOURS)
        removed = 0
        with self._lock:
            to_remove = []
            for goal_id, plan in self._plans.items():
                plan_time = plan.created_at
                if plan_time.tzinfo is not None:
                    plan_time = plan_time.replace(tzinfo=None)
                if plan_time < cutoff:
                    to_remove.append(goal_id)
            for goal_id in to_remove:
                del self._plans[goal_id]
                removed += 1
            if removed:
                self._save_state()
        return removed

    def prune_outcomes(self) -> int:
        """Prune outcomes beyond the learning window. Returns count pruned."""
        now = datetime.now()
        from datetime import timedelta
        cutoff = now - timedelta(days=self._learning_window_days)
        with self._lock:
            original = len(self._outcomes)
            self._outcomes = [
                o for o in self._outcomes
                if o.timestamp.replace(tzinfo=None) >= cutoff
            ]
            pruned = original - len(self._outcomes)
            if pruned:
                self._save_state()
            return pruned

    def enforce_plan_limit(self) -> int:
        """Enforce MAX_ACTIVE_PLANS limit by removing oldest plans. Returns count removed."""
        with self._lock:
            if len(self._plans) <= self.MAX_ACTIVE_PLANS:
                return 0
            sorted_plans = sorted(
                self._plans.items(),
                key=lambda x: x[1].created_at,
            )
            to_remove = len(self._plans) - self.MAX_ACTIVE_PLANS
            for goal_id, _ in sorted_plans[:to_remove]:
                del self._plans[goal_id]
            self._save_state()
            return to_remove

    def learn_from_outcomes(self) -> dict[str, dict]:
        """Analyze outcomes to learn error patterns and optimal strategies.
        
        Returns dict of learned patterns: {error_pattern: {best_strategy, success_rate, ...}}
        """
        with self._lock:
            pattern_outcomes: dict[str, list[OutcomeRecord]] = defaultdict(list)
            for outcome in self._outcomes:
                if outcome.error and outcome.error_type:
                    key = f"{outcome.error_type.value}:{outcome.error[:50]}"
                    pattern_outcomes[key].append(outcome)

            learned: dict[str, dict] = {}
            for pattern_key, outcomes_list in pattern_outcomes.items():
                strategy_success: dict[str, list[bool]] = defaultdict(list)
                for o in outcomes_list:
                    if o.strategy_used:
                        strategy_success[o.strategy_used.value].append(o.recovery_successful)

                best_strategy = None
                best_rate = 0.0
                for strat, results in strategy_success.items():
                    rate = sum(results) / len(results) if results else 0.0
                    if rate > best_rate:
                        best_rate = rate
                        best_strategy = strat

                learned[pattern_key] = {
                    "count": len(outcomes_list),
                    "best_strategy": best_strategy,
                    "best_strategy_rate": best_rate,
                    "error_type": pattern_key.split(":")[0],
                }

            self._error_learning = learned
            return learned

    def get_best_recovery_strategy(self, error: str) -> RecoveryStrategy | None:
        """Get the best learned recovery strategy for an error.
        
        Uses learned outcomes to suggest the most successful strategy.
        Falls back to pattern-based strategies if no learning data exists.
        """
        error_type = self.classify_error(error)
        error_key = f"{error_type.value}:{error[:50]}"

        learned = self._error_learning.get(error_key)
        if learned and learned.get("best_strategy"):
            try:
                return RecoveryStrategy(learned["best_strategy"])
            except ValueError:
                pass

        strategies = self.get_recovery_strategies(error)
        return strategies[0] if strategies else None

    def get_analytics(self) -> dict[str, Any]:
        """Comprehensive analytics about plans, outcomes, and learning."""
        with self._lock:
            total = len(self._outcomes)
            success = sum(1 for o in self._outcomes if o.success)
            
            avg_duration = 0.0
            if total > 0:
                avg_duration = sum(o.duration_seconds for o in self._outcomes) / total

            error_type_dist = defaultdict(int)
            for o in self._outcomes:
                if o.error_type:
                    error_type_dist[o.error_type.value] += 1

            strategy_dist: dict[str, dict] = {}
            for o in self._outcomes:
                if o.strategy_used:
                    key = o.strategy_used.value
                    if key not in strategy_dist:
                        strategy_dist[key] = {"total": 0, "recovery_success": 0}
                    strategy_dist[key]["total"] += 1
                    if o.recovery_successful:
                        strategy_dist[key]["recovery_success"] += 1

            plan_status_dist: dict[str, int] = defaultdict(int)
            for plan in self._plans.values():
                all_steps_done = all(
                    s.status in (StepStatus.COMPLETED, StepStatus.SKIPPED)
                    for s in plan.steps
                )
                any_failed = any(s.status == StepStatus.FAILED for s in plan.steps)
                if all_steps_done and not any_failed:
                    plan_status_dist["completed"] += 1
                elif any_failed:
                    plan_status_dist["failed"] += 1
                else:
                    plan_status_dist["in_progress"] += 1

            return {
                "total_outcomes": total,
                "success_count": success,
                "success_rate": success / total if total > 0 else 0.0,
                "avg_duration_seconds": round(avg_duration, 2),
                "error_type_distribution": dict(error_type_dist),
                "strategy_distribution": strategy_dist,
                "plan_status_distribution": dict(plan_status_dist),
                "learned_patterns": len(self._error_learning),
                "active_plans": len(self._plans),
                "outcomes_stored": total,
            }


_self_correction_engine: SelfCorrectionEngine | None = None
_engine_lock = threading.Lock()


def get_self_correction_engine(llm_client=None) -> SelfCorrectionEngine:
    global _self_correction_engine
    if _self_correction_engine is None:
        with _engine_lock:
            if _self_correction_engine is None:
                _self_correction_engine = SelfCorrectionEngine(llm_client=llm_client)
    return _self_correction_engine


def decompose_goal(goal: str, **kwargs) -> ExecutionPlan:
    """Sync decomposition (heuristic; sync LLMs only)."""
    return get_self_correction_engine().decompose_goal(goal, **kwargs)


async def decompose_goal_async(goal: str, **kwargs) -> ExecutionPlan:
    """Async decomposition (supports async LLM clients via api_manager)."""
    return await get_self_correction_engine().decompose_goal_async(goal, **kwargs)


async def execute_with_recovery(goal: str, **kwargs) -> tuple[bool, dict]:
    return await get_self_correction_engine().execute_with_recovery(goal, **kwargs)