"""Lobster-style Composable Workflow Shell for Jarvis.

A shell abstraction for composing pipelines of tools, subagents, and operations
with pipe-like syntax and parallel/sequential execution patterns.

Key features:
- Composable pipeline syntax with | operator
- Parallel (||) and sequential (>>) composition
- Conditionals and branching
- Error handling with fallbacks
- State propagation between stages
- Redis-like pub/sub for inter-pipeline communication

Example:
    pipeline = (
        Pipeline("search_and_summarize")
        | WebSearchTool()
        | ExtractResults()
        || SummarizeTool()  # Run in parallel
        | CombineResults()
        >> SaveToFile()     # Run after previous completes
    )
    
    result = await pipeline.execute("quantum computing advances")
"""
import asyncio
import inspect
import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class StageStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StageResult(Generic[T]):
    value: T | None = None
    error: Exception | None = None
    status: StageStatus = StageStatus.PENDING
    metadata: dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0

    @property
    def success(self) -> bool:
        return self.status == StageStatus.COMPLETED


@dataclass
class PipelineContext:
    workflow_id: str
    stage_data: dict[str, StageResult] = field(default_factory=dict)
    shared_state: dict[str, Any] = field(default_factory=dict)
    parent_context: 'PipelineContext | None' = None

    def get(self, key: str, default: Any = None) -> Any:
        if key in self.shared_state:
            return self.shared_state[key]
        if self.parent_context:
            return self.parent_context.get(key, default)
        return default

    def set(self, key: str, value: Any) -> None:
        self.shared_state[key] = value

    def get_stage_result(self, stage_name: str) -> StageResult | None:
        if stage_name in self.stage_data:
            return self.stage_data[stage_name]
        if self.parent_context:
            return self.parent_context.get_stage_result(stage_name)
        return None


class Stage(ABC, Generic[T, R]):
    name: str = "stage"
    description: str = ""
    requires: list[str] = field(default_factory=list)
    provides: list[str] = field(default_factory=list)

    @abstractmethod
    async def execute(self, input_data: T, context: PipelineContext) -> StageResult[R]:
        pass

    def __or__(self, other: 'Stage') -> 'SequentialPipeline':
        return SequentialPipeline(
            name=f"{self.name}_{other.name}",
            stages=[self, other],
        )

    def __rshift__(self, other: 'Stage') -> 'SequentialPipeline':
        return SequentialPipeline(
            name=f"{self.name}_{other.name}",
            stages=[self, other],
        )

    def __ror__(self, other: 'Stage') -> 'SequentialPipeline':
        return SequentialPipeline(
            name=f"{other.name}_{self.name}",
            stages=[other, self],
        )


class FunctionStage(Stage[T, R]):
    def __init__(
        self,
        func: Callable[[T, PipelineContext], Awaitable[R] | R],
        name: str | None = None,
        description: str = "",
    ):
        self._func = func
        self.name = name or func.__name__
        self.description = description or func.__doc__ or ""

    async def execute(self, input_data: T, context: PipelineContext) -> StageResult[R]:
        try:
            start = time.perf_counter()

            if inspect.iscoroutinefunction(self._func):
                result = await self._func(input_data, context)
            else:
                result = self._func(input_data, context)

            duration_ms = (time.perf_counter() - start) * 1000

            return StageResult(
                value=result,
                status=StageStatus.COMPLETED,
                duration_ms=duration_ms,
            )
        except Exception as e:
            logger.error(f"Stage {self.name} failed: {e}")
            return StageResult(
                error=e,
                status=StageStatus.FAILED,
            )


class FallbackStage(Stage[T, R]):
    def __init__(self, primary: Stage[T, R], fallback: Stage[T, R]):
        self.primary = primary
        self.fallback = fallback
        self.name = f"{primary.name}_fallback_{fallback.name}"
        self.description = f"Try {primary.name}, fallback to {fallback.name}"

    async def execute(self, input_data: T, context: PipelineContext) -> StageResult[R]:
        result = await self.primary.execute(input_data, context)

        if result.success:
            return result

        logger.info(f"Stage {self.primary.name} failed, trying fallback {self.fallback.name}")
        return await self.fallback.execute(input_data, context)


class ConditionalStage(Stage[T, R]):
    def __init__(
        self,
        condition: Callable[[T, PipelineContext], bool],
        if_stage: Stage[T, R],
        else_stage: Stage[T, R] | None = None,
        name: str | None = None,
    ):
        self._condition = condition
        self._if_stage = if_stage
        self._else_stage = else_stage
        self.name = name or f"conditional_{if_stage.name}"

    async def execute(self, input_data: T, context: PipelineContext) -> StageResult[R]:
        try:
            if self._condition(input_data, context):
                return await self._if_stage.execute(input_data, context)
            elif self._else_stage:
                return await self._else_stage.execute(input_data, context)
            else:
                return StageResult(
                    status=StageStatus.SKIPPED,
                    metadata={"reason": "condition_false"},
                )
        except Exception as e:
            return StageResult(error=e, status=StageStatus.FAILED)


class RetryStage(Stage[T, R]):
    def __init__(
        self,
        stage: Stage[T, R],
        max_retries: int = 3,
        backoff_seconds: float = 1.0,
        exponential: bool = True,
    ):
        self.stage = stage
        self.name = f"retry_{stage.name}"
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.exponential = exponential

    async def execute(self, input_data: T, context: PipelineContext) -> StageResult[R]:
        last_error = None

        for attempt in range(self.max_retries + 1):
            result = await self.stage.execute(input_data, context)

            if result.success:
                return result

            last_error = result.error

            if attempt < self.max_retries:
                delay = self.backoff_seconds * (2 ** attempt if self.exponential else 1)
                await asyncio.sleep(delay)
                logger.warning(f"Retry {attempt + 1}/{self.max_retries} for {self.stage.name}")

        return StageResult(
            error=last_error,
            status=StageStatus.FAILED,
            metadata={"retries": self.max_retries},
        )


class TimeoutStage(Stage[T, R]):
    def __init__(self, stage: Stage[T, R], timeout_seconds: float):
        self.stage = stage
        self.name = f"timeout_{stage.name}"
        self.timeout_seconds = timeout_seconds

    async def execute(self, input_data: T, context: PipelineContext) -> StageResult[R]:
        try:
            return await asyncio.wait_for(
                self.stage.execute(input_data, context),
                timeout=self.timeout_seconds,
            )
        except asyncio.TimeoutError:
            return StageResult(
                error=asyncio.TimeoutError(f"Stage {self.stage.name} timed out after {self.timeout_seconds}s"),
                status=StageStatus.FAILED,
                metadata={"timeout_seconds": self.timeout_seconds},
            )


class Pipeline(Stage[T, list]):
    def __init__(self, name: str, description: str = "", stages: list[Stage] | None = None):
        self.name = name
        self.description = description
        self._stages: list[Stage] = stages or []
        self._context_class = PipelineContext

    def add_stage(self, stage: Stage) -> 'Pipeline':
        self._stages.append(stage)
        return self

    def __or__(self, other: 'Pipeline') -> 'SequentialPipeline':
        if isinstance(other, Pipeline):
            return SequentialPipeline(
                name=f"{self.name}_{other.name}",
                stages=self._stages + other._stages,
            )
        return SequentialPipeline(
            name=self.name,
            stages=self._stages + [other],
        )

    def __rshift__(self, other: Stage) -> 'SequentialPipeline':
        return self.__or__(other)

    def __ior__(self, other: Stage) -> 'SequentialPipeline':
        return self.__or__(other)

    async def execute(self, input_data: T, context: PipelineContext | None = None) -> StageResult[list]:
        if context is None:
            import uuid
            context = PipelineContext(workflow_id=str(uuid.uuid4()))

        results = []
        current_data = input_data

        for stage in self._stages:
            stage_result = await stage.execute(current_data, context)
            context.stage_data[stage.name] = stage_result
            results.append(stage_result)

            if not stage_result.success:
                return StageResult(
                    error=stage_result.error,
                    status=StageStatus.FAILED,
                    metadata={
                        "failed_stage": stage.name,
                        "completed_stages": len([r for r in results if r.success]),
                    },
                )

            current_data = stage_result.value

        return StageResult(
            value=[r.value for r in results if r.value is not None],
            status=StageStatus.COMPLETED,
            metadata={"stages": len(results)},
        )


class SequentialPipeline(Pipeline):
    def __init__(self, name: str = "sequential", description: str = "", stages: list[Stage] | None = None):
        super().__init__(name, description, stages)

    async def execute(self, input_data: T, context: PipelineContext | None = None) -> StageResult[list]:
        if context is None:
            import uuid
            context = PipelineContext(workflow_id=str(uuid.uuid4()))

        results = []
        current_data = input_data
        start_time = time.perf_counter()

        for i, stage in enumerate(self._stages):
            logger.debug(f"Executing stage {i+1}/{len(self._stages)}: {stage.name}")

            stage_result = await stage.execute(current_data, context)
            context.stage_data[stage.name] = stage_result
            results.append(stage_result)

            if not stage_result.success and not isinstance(stage, (FallbackStage, RetryStage)):
                return StageResult(
                    error=stage_result.error,
                    status=StageStatus.FAILED,
                    metadata={
                        "failed_stage": stage.name,
                        "stage_index": i,
                        "completed_stages": i,
                    },
                )

            current_data = stage_result.value

        duration_ms = (time.perf_counter() - start_time) * 1000

        return StageResult(
            value=[r.value for r in results if r.value is not None],
            status=StageStatus.COMPLETED,
            duration_ms=duration_ms,
            metadata={
                "stages": len(results),
                "workflow_id": context.workflow_id,
            },
        )


class ParallelPipeline(Pipeline):
    def __init__(self, name: str = "parallel", description: str = "", stages: list[Stage] | None = None):
        super().__init__(name, description, stages)

    def __or__(self, other: Stage) -> 'ParallelPipeline':
        if isinstance(other, ParallelPipeline):
            return ParallelPipeline(
                f"{self.name}_{other.name}",
                stages=self._stages + other._stages,
            )
        return self.add_stage(other)

    async def execute(self, input_data: T, context: PipelineContext | None = None) -> StageResult[list]:
        if context is None:
            import uuid
            context = PipelineContext(workflow_id=str(uuid.uuid4()))

        start_time = time.perf_counter()

        tasks = [stage.execute(input_data, context) for stage in self._stages]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            stage = self._stages[i]
            if isinstance(result, Exception):
                context.stage_data[stage.name] = StageResult(
                    error=result,
                    status=StageStatus.FAILED,
                )
            else:
                context.stage_data[stage.name] = result

        values = []
        errors = []

        for result in results:
            if isinstance(result, Exception):
                errors.append(result)
            elif isinstance(result, StageResult):
                if result.success:
                    values.append(result.value)
                elif result.error:
                    errors.append(result.error)

        duration_ms = (time.perf_counter() - start_time) * 1000

        if errors and len(errors) > len(results) // 2:
            return StageResult(
                error=errors[0],
                status=StageStatus.FAILED,
                duration_ms=duration_ms,
                metadata={"errors": [str(e) for e in errors]},
            )

        return StageResult(
            value=values,
            status=StageStatus.COMPLETED,
            duration_ms=duration_ms,
            metadata={
                "stages": len(results),
                "successful": len(values),
                "failed": len(errors),
            },
        )


def parallel(*stages: Stage) -> ParallelPipeline:
    return ParallelPipeline(stages=list(stages))


class WorkflowShell:
    """Pipeline registry and execution manager for Lobster-style workflows.
    
    Features:
    - Register and reuse named pipelines
    - Execute pipelines by name
    - Pipeline composition with DSL
    - State management across executions
    - Pub/sub for inter-pipeline communication
    """

    def __init__(self, storage_path: Path | None = None):
        self.storage_path = storage_path or Path.home() / ".jarvis" / "workflows"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self._pipelines: dict[str, Pipeline] = {}
        self._results: dict[str, StageResult] = {}
        self._subscribers: dict[str, list[Callable]] = {}
        self._lock = threading.RLock()
        self._max_history = 1000
        self._max_subscribers_per_event = 20
        self._execution_history: deque[dict] = deque(maxlen=self._max_history)

    def register(self, name: str, pipeline: Pipeline) -> None:
        with self._lock:
            self._pipelines[name] = pipeline
            logger.info(f"Registered pipeline: {name}")

    def unregister(self, name: str) -> bool:
        with self._lock:
            if name in self._pipelines:
                del self._pipelines[name]
                return True
            return False

    def get(self, name: str) -> Pipeline | None:
        with self._lock:
            return self._pipelines.get(name)

    def compose(self, *pipeline_names: str, name: str | None = None) -> Pipeline:
        stages = []
        for pname in pipeline_names:
            pipeline = self._pipelines.get(pname)
            if pipeline:
                stages.extend(pipeline._stages)

        composed_name = name or "_".join(pipeline_names)
        return SequentialPipeline(composed_name, stages=stages)

    async def execute(
        self,
        name: str,
        input_data: Any,
        context: PipelineContext | None = None,
    ) -> StageResult:
        pipeline = self._pipelines.get(name)
        if not pipeline:
            return StageResult(
                error=ValueError(f"Pipeline not found: {name}"),
                status=StageStatus.FAILED,
            )

        start_time = time.perf_counter()
        result = await pipeline.execute(input_data, context)
        duration_ms = (time.perf_counter() - start_time) * 1000

        with self._lock:
            self._results[name] = result
            self._execution_history.append({
                "pipeline": name,
                "timestamp": datetime.now().isoformat(),
                "success": result.success,
                "duration_ms": duration_ms,
            })

        self._publish(f"pipeline.{name}.complete", result)

        return result

    def subscribe(self, event: str, handler: Callable) -> None:
        with self._lock:
            if event not in self._subscribers:
                self._subscribers[event] = []
            if len(self._subscribers[event]) < self._max_subscribers_per_event:
                self._subscribers[event].append(handler)

    def unsubscribe(self, event: str, handler: Callable) -> None:
        with self._lock:
            if event in self._subscribers:
                self._subscribers[event] = [h for h in self._subscribers[event] if h != handler]

    def _publish(self, event: str, data: Any) -> None:
        with self._lock:
            handlers = list(self._subscribers.get(event, []))
        for handler in handlers:
            try:
                if inspect.iscoroutinefunction(handler):
                    from core.helpers import create_tracked_task
                    create_tracked_task(handler(event, data), name=f"workflow_handler_{event}")
                else:
                    handler(event, data)
            except Exception as e:
                logger.error(f"Handler error for {event}: {e}")

    def list_pipelines(self) -> list[str]:
        with self._lock:
            return list(self._pipelines.keys())

    def get_result(self, name: str) -> StageResult | None:
        with self._lock:
            return self._results.get(name)

    def get_history(self, limit: int = 100) -> list[dict]:
        with self._lock:
            return list(self._execution_history)[-limit:]

    def clear_history(self) -> None:
        with self._lock:
            self._execution_history.clear()

    def save_pipeline(self, name: str, path: Path | None = None) -> bool:
        return False

    def load_pipeline(self, name: str, path: Path) -> bool:
        return False


_pipeline_shell: WorkflowShell | None = None
_shell_lock = threading.Lock()


def get_workflow_shell() -> WorkflowShell:
    global _pipeline_shell
    if _pipeline_shell is None:
        with _shell_lock:
            if _pipeline_shell is None:
                _pipeline_shell = WorkflowShell()
    return _pipeline_shell


def stage(func: Callable) -> FunctionStage:
    return FunctionStage(func)


def retry(stage: Stage, max_retries: int = 3, backoff: float = 1.0) -> RetryStage:
    return RetryStage(stage, max_retries, backoff)


def timeout(stage: Stage, seconds: float) -> TimeoutStage:
    return TimeoutStage(stage, seconds)


def fallback(primary: Stage, backup: Stage) -> FallbackStage:
    return FallbackStage(primary, backup)


def when(
    condition: Callable[[T, PipelineContext], bool],
    if_stage: Stage,
    else_stage: Stage | None = None,
) -> ConditionalStage:
    return ConditionalStage(condition, if_stage, else_stage)
