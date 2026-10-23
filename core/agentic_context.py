"""Enhanced Agentic Context and Execution System for Jarvis.

This module provides:
1. Rich context with project/workspace awareness
2. Lazy execution and early exit patterns
3. Tool input pre-validation
4. Result caching and deduplication
5. Agentic decision-making in workflows
6. Project-aware execution planning

Key Classes:
- AgenticContext: Rich context with project, goals, history, dependencies
- LazyStage: Stages that can validate and skip execution
- ValidationMixin: Input pre-validation for tools
- AgenticPipeline: Full agentic execution flow

Example:
    context = AgenticContext(
        project_path="/home/user/Research Project",
        goal="Improve simulation accuracy",
    )
    
    # Lazy validation - skip if not needed
    stage = LazyStage(
        name="run_tests",
        execute_fn=run_tests,
        precondition=lambda ctx: ctx.has_modified_files(),
        lazy_exit=True,
    )
    
    # Full agentic loop
    result = await agentic_execute(
        context=context,
        goal="Add cardiac tissue template",
        max_iterations=10,
    )
"""
import asyncio
import inspect
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class ExecutionContext(Enum):
    PROJECT = "project"
    GOAL = "goal"
    TASK = "task"
    WORKFLOW = "workflow"


@dataclass
class ProjectContext:
    """Project-specific context."""
    project_path: str | None = None
    project_name: str = ""
    language: str = ""
    frameworks: list[str] = field(default_factory=list)
    entry_points: list[str] = field(default_factory=list)
    architecture: dict[str, Any] = field(default_factory=dict)
    dependencies: dict[str, list[str]] = field(default_factory=dict)
    last_modified: datetime | None = None
    git_branch: str = ""
    git_status: str = ""
    
    def has_file(self, path: str) -> bool:
        """Check if a file exists in project architecture safely.
        
        Prevents path traversal attacks by resolving paths.
        """
        if not self.project_path:
            return False
        
        try:
            from pathlib import Path
            
            # Resolve to absolute path and check it's within project
            full_path = (Path(self.project_path) / path).resolve()
            project_root = Path(self.project_path).resolve()
            
            # Prevent path traversal
            if not str(full_path).startswith(str(project_root)):
                return False
            
            # Check if in architecture keys
            arch_keys = [str(Path(self.project_path) / p) for p in self.architecture.keys()]
            return str(full_path) in arch_keys or path in self.architecture
        except Exception:
            return False
    
    def get_file_dependencies(self, path: str) -> list[str]:
        return self.dependencies.get(path, [])


@dataclass
class GoalContext:
    """Goal and progress tracking."""
    goal_id: str = ""
    goal_statement: str = ""
    parent_goal: str | None = None
    subtasks: list[str] = field(default_factory=list)
    completed_subtasks: list[str] = field(default_factory=list)
    progress: float = 0.0
    priority: int = 3
    deadline: datetime | None = None
    blockers: list[str] = field(default_factory=list)
    
    def is_complete(self) -> bool:
        return self.progress >= 100.0
    
    def add_subtask(self, subtask_id: str) -> None:
        if subtask_id not in self.subtasks:
            self.subtasks.append(subtask_id)
    
    def complete_subtask(self, subtask_id: str) -> None:
        if subtask_id in self.subtasks and subtask_id not in self.completed_subtasks:
            self.completed_subtasks.append(subtask_id)
            if self.subtasks:
                self.progress = len(self.completed_subtasks) / len(self.subtasks) * 100


def _get_max_history_steps() -> int:
    """Get max history steps from config."""
    try:
        from core.config import Config
        config = Config.load()
        return getattr(config.agent, 'max_history_steps', 100)
    except Exception:
        return 100


MAX_HISTORY_STEPS = _get_max_history_steps()


@dataclass
class ExecutionHistory:
    """History of executed steps with results."""
    steps: list[dict] = field(default_factory=list)
    current_step: int = 0
    max_steps: int = MAX_HISTORY_STEPS
    
    def add_step(
        self,
        step_id: str,
        step_type: str,
        action: str,
        result: dict | None = None,
        success: bool = True,
        duration_ms: float = 0.0,
    ) -> None:
        self.steps.append({
            "step_id": step_id,
            "step_type": step_type,
            "action": action,
            "result": result,
            "success": success,
            "duration_ms": duration_ms,
            "timestamp": datetime.now().isoformat(),
        })
        self.current_step = len(self.steps) - 1
        # Enforce max size
        if len(self.steps) > self.max_steps:
            self.steps = self.steps[-self.max_steps:]
    
    def get_recent_results(self, limit: int = 5) -> list[dict]:
        return self.steps[-limit:]
    
    def get_results_for_action(self, action_pattern: str) -> list[dict]:
        import re
        pattern = re.compile(action_pattern, re.IGNORECASE)
        return [s for s in self.steps if pattern.search(s.get("action", ""))]
    
    def last_success_for(self, action: str) -> dict | None:
        for step in reversed(self.steps):
            if step.get("action") == action and step.get("success"):
                return step.get("result")
        return None


@dataclass
class DependencyGraph:
    """Track dependencies between execution steps."""
    nodes: dict[str, dict] = field(default_factory=dict)
    edges: dict[str, list[str]] = field(default_factory=dict)
    
    def add_node(self, node_id: str, node_type: str, metadata: dict | None = None) -> None:
        self.nodes[node_id] = {
            "type": node_type,
            "metadata": metadata or {},
            "status": "pending",
        }
        if node_id not in self.edges:
            self.edges[node_id] = []
    
    def add_edge(self, from_id: str, to_id: str) -> None:
        if from_id not in self.edges:
            self.edges[from_id] = []
        if to_id not in self.edges[from_id]:
            self.edges[from_id].append(to_id)
    
    def get_dependencies(self, node_id: str) -> list[str]:
        return [n for n, edges in self.edges.items() if node_id in edges]
    
    def get_dependents(self, node_id: str) -> list[str]:
        return self.edges.get(node_id, [])
    
    def mark_complete(self, node_id: str) -> None:
        if node_id in self.nodes:
            self.nodes[node_id]["status"] = "complete"
    
    def is_ready(self, node_id: str) -> bool:
        deps = self.get_dependencies(node_id)
        return all(
            self.nodes.get(d, {}).get("status") == "complete"
            for d in deps
        )
    
    def get_ready_nodes(self) -> list[str]:
        return [
            n for n, data in self.nodes.items()
            if data["status"] == "pending" and self.is_ready(n)
        ]


@dataclass
class ContextAwareness:
    """Captures context awareness for user interaction.

    Used to provide transparent context to users about
    what the agent knows, why tools are selected, and
    what progress is being made.
    """

    query: str = ""
    relevant_memories: list[dict] = field(default_factory=list)
    active_context: dict[str, Any] = field(default_factory=dict)
    cross_model_insights: list[dict] = field(default_factory=list)
    tool_recommendations: list[dict] = field(default_factory=list)
    confidence: float = 0.0
    reasoning: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AgenticContext:
    """Rich context for agentic execution.
    
    Provides:
    - Project context (files, architecture, dependencies)
    - Goal context (progress, subtasks, blockers)
    - Execution history (past steps, results)
    - Dependency graph (what depends on what)
    - Shared state (variables passed between stages)
    - Tool call cache (avoid redundant calls)
    """
    
    workflow_id: str = ""
    execution_context: ExecutionContext = ExecutionContext.TASK
    
    project: ProjectContext = field(default_factory=ProjectContext)
    goal: GoalContext = field(default_factory=GoalContext)
    history: ExecutionHistory = field(default_factory=ExecutionHistory)
    dependencies: DependencyGraph = field(default_factory=DependencyGraph)
    
    shared_state: dict[str, Any] = field(default_factory=dict)
    stage_results: dict[str, Any] = field(default_factory=dict)
    tool_cache: dict[str, Any] = field(default_factory=dict)
    variables: dict[str, Any] = field(default_factory=dict)
    
    metadata: dict[str, Any] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    _MAX_SHARED_STATE = 500
    _MAX_TOOL_CACHE = 200
    _MAX_STAGE_RESULTS = 200
    _MAX_VARIABLES = 200
    
    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            if key in self.shared_state:
                return self.shared_state[key]
            if key in self.variables:
                return self.variables[key]
        return default
    
    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self.shared_state[key] = value
            if len(self.shared_state) > self._MAX_SHARED_STATE:
                oldest = list(self.shared_state.keys())[:len(self.shared_state) - self._MAX_SHARED_STATE]
                for k in oldest:
                    del self.shared_state[k]
    
    def set_variable(self, key: str, value: Any) -> None:
        with self._lock:
            self.variables[key] = value
            if len(self.variables) > self._MAX_VARIABLES:
                oldest = list(self.variables.keys())[:len(self.variables) - self._MAX_VARIABLES]
                for k in oldest:
                    del self.variables[k]
    
    def get_stage_result(self, stage_name: str) -> Any | None:
        return self.stage_results.get(stage_name)
    
    def set_stage_result(self, stage_name: str, result: Any) -> None:
        with self._lock:
            self.stage_results[stage_name] = result
            if len(self.stage_results) > self._MAX_STAGE_RESULTS:
                oldest = list(self.stage_results.keys())[:len(self.stage_results) - self._MAX_STAGE_RESULTS]
                for k in oldest:
                    del self.stage_results[k]
    
    def cache_tool_result(self, tool_name: str, params_hash: str, result: Any) -> None:
        with self._lock:
            key = f"{tool_name}:{params_hash}"
            self.tool_cache[key] = {
                "result": result,
                "timestamp": datetime.now(),
            }
            # Enforce max cache size (keep 500 most recent)
            if len(self.tool_cache) > 500:
                # Remove oldest entries
                sorted_keys = sorted(
                    self.tool_cache.keys(),
                    key=lambda k: self.tool_cache[k].get("timestamp", datetime.min)
                )
                for old_key in sorted_keys[:100]:
                    del self.tool_cache[old_key]

    def get_cached_tool_result(self, tool_name: str, params_hash: str) -> Any | None:
        with self._lock:
            key = f"{tool_name}:{params_hash}"
            cached = self.tool_cache.get(key)
            if cached:
                age = (datetime.now() - cached["timestamp"]).total_seconds()
                if age < 300:
                    return cached["result"]
        return None
    
    def has_modified_files(self) -> bool:
        return bool(self.shared_state.get("modified_files"))
    
    def has_run_tests(self) -> bool:
        return any(s.get("action") == "run_tests" for s in self.history.steps)
    
    def get_last_test_result(self) -> dict | None:
        return self.history.last_success_for("run_tests")
    
    def requires_rebuild(self) -> bool:
        if not self.has_modified_files():
            return False
        modified = self.shared_state.get("modified_files", [])
        rebuild_patterns = [".py", ".js", ".ts", ".go", ".rs", ".cpp", ".c"]
        return any(any(f.endswith(ext) for ext in rebuild_patterns) for f in modified)
    
    def can_proceed_with(self, step_id: str) -> bool:
        return self.dependencies.is_ready(step_id)
    
    def to_summary(self) -> dict:
        return {
            "workflow_id": self.workflow_id,
            "execution_context": self.execution_context.value,
            "project_name": self.project.project_name,
            "goal_progress": self.goal.progress,
            "steps_completed": len(self.history.steps),
            "shared_state_keys": list(self.shared_state.keys()),
            "dependency_nodes": len(self.dependencies.nodes),
            "cache_entries": len(self.tool_cache),
        }
    
    def reset(self) -> None:
        """Reset context for new interaction."""
        self.history = ExecutionHistory()
        self.shared_state = {}
        self.stage_results = {}
        self.tool_cache = {}
        self.variables = {}
    
    async def capture_context(
        self,
        query: str,
        agent: Any = None,
        include_cross_model: bool = False,
    ) -> "ContextAwareness":
        """Capture context awareness for transparent interaction.

        Args:
            query: User query
            agent: Optional agent for memory/context access
            include_cross_model: Include cross-model insights

        Returns:
            ContextAwareness object
        """
        awareness = ContextAwareness(query=query)
        
        if agent:
            try:
                if hasattr(agent, "memory") and hasattr(agent.memory, "get_messages_for_llm"):
                    recent = agent.memory.get_messages_for_llm()[-5:]
                    awareness.relevant_memories = [
                        {"role": m.get("role", ""), "content": m.get("content", "")[:100]}
                        for m in recent
                    ]
                
                if hasattr(agent, "tools"):
                    awareness.tool_recommendations = [
                        {"name": name, "available": True}
                        for name in list(agent.tools.keys())[:10]
                    ]
                
                if include_cross_model:
                    try:
                        from core.unified_context_engine import get_context_engine
                        engine = get_context_engine()
                        pack = await engine.assemble_context(query=query, max_tokens=500)
                        awareness.active_context = {
                            "entries": len(pack.entries),
                            "sources": [s.value for s in pack.sources_included],
                        }
                    except Exception as e:
                        logger.warning(f"Failed to assemble context for awareness: {e}")
                
                awareness.confidence = 0.85
                awareness.reasoning = f"Captured context for: {query[:50]}"
            except Exception as e:
                logger.warning(f"Failed to capture context awareness: {e}")
        
        return awareness
    
    def format_context_intro(self, awareness: ContextAwareness | None) -> str:
        """Format context introduction for display.
        
        Args:
            awareness: Context awareness object
            
        Returns:
            Formatted string
        """
        if awareness is None:
            return ""
        
        parts = []
        
        if awareness.relevant_memories:
            parts.append(f"_Contextawareness: {len(awareness.relevant_memories)} recent interactions_")
        
        if awareness.tool_recommendations:
            tool_names = [t["name"] for t in awareness.tool_recommendations[:3]]
            parts.append(f"_Available tools: {', '.join(tool_names)}_")
        
        if parts:
            return "\n📊 **Context Awareness**\n" + "\n".join(parts) + "\n"
        
        return ""
    
    def capture_tool_reasoning(
        self,
        tool_name: str,
        tool_args: dict,
        context: ContextAwareness | None,
    ) -> str:
        """Capture and format tool selection reasoning.
        
        Args:
            tool_name: Name of tool being used
            tool_args: Tool arguments
            context: Context awareness
            
        Returns:
            Reasoning string
        """
        if context is None:
            return ""
        
        tool_info = next(
            (t for t in context.tool_recommendations if t["name"] == tool_name),
            None,
        )
        
        if tool_info:
            return f"Selected {tool_name} based on: {context.query[:50]}"
        
        return ""
    
    def format_completion(
        self,
        completed: list[str],
        tools_used: list[str],
        total_steps: int,
    ) -> str:
        """Format completion message.
        
        Args:
            completed: List of completed step descriptions
            tools_used: List of tools used
            total_steps: Total steps taken
            
        Returns:
            Formatted completion string
        """
        msg = f"\n✅ **Task Completed** ({total_steps} iterations)\n"
        if completed:
            msg += "_Steps:_" + "\n".join(f"  - {s}" for s in completed[:3]) + "\n"
        if tools_used:
            unique_tools = list(dict.fromkeys(tools_used))
            msg += f"_Tools: {', '.join(unique_tools[:5])}_\n"
        return msg
    
    def format_progress(
        self,
        completed: list[str],
        remaining: str,
        iteration: int,
        max_iterations: int,
    ) -> str:
        """Format progress message.
        
        Args:
            completed: Completed steps
            remaining: Remaining action
            iteration: Current iteration
            max_iterations: Max iterations
            
        Returns:
            Progress string
        """
        msg = f"\n🔄 **Progress** (iteration {iteration}/{max_iterations})\n"
        if completed:
            msg += "_Completed:_" + "\n".join(f"  - {s}" for s in completed[:2]) + "\n"
        if remaining:
            msg += f"_Next: {remaining[:80]}_\n"
        return msg
    
    def format_gap_analysis(
        self,
        original_goal: str,
        current_state: str,
        missing: str,
    ) -> str:
        """Format gap analysis for transparency.
        
        Args:
            original_goal: Original user goal
            current_state: Current progress state
            missing: What's missing/needed
            
        Returns:
            Gap analysis string
        """
        return (
            f"\n🔍 **Gap Analysis**\n"
            f"_Goal: {original_goal}..._\n"
            f"_Status: {current_state[:80]}..._\n"
            f"_Missing: {missing[:60]}_\n"
        )
    
    def record_tool_execution(
        self,
        tool_name: str,
        tool_args: dict,
        result: str,
        success: bool,
    ) -> None:
        """Record a tool execution in history.
        
        Args:
            tool_name: Tool name
            tool_args: Tool arguments
            result: Execution result
            success: Whether execution succeeded
        """
        step = {
            "action": f"tool:{tool_name}",
            "args": tool_args,
            "result": result[:500] if result else "",
            "success": success,
            "timestamp": datetime.now().isoformat(),
        }
        self.history.steps.append(step)
    
    def increment_iteration(self) -> int:
        """Increment iteration counter.
        
        Returns:
            New iteration count
        """
        self.shared_state["iteration"] = self.shared_state.get("iteration", 0) + 1
        return self.shared_state["iteration"]


class ValidationStatus(Enum):
    """Status of validation."""
    VALID = "valid"
    INVALID = "invalid"
    NEEDS_INPUT = "needs_input"
    SKIP = "skip"


@dataclass
class ValidationResult:
    """Result of validation with details."""
    result: ValidationStatus
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    missing_params: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    
    @property
    def is_valid(self) -> bool:
        """Check if validation passed."""
        return self.result in (ValidationStatus.VALID, ValidationStatus.SKIP)


class InputValidator(ABC):
    """Base class for tool input validation."""
    
    @abstractmethod
    def validate(self, params: dict, context: AgenticContext) -> ValidationResult:
        """Validate input parameters before execution."""
        pass
    
    def get_missing_params(self, params: dict, required: list[str]) -> list[str]:
        return [p for p in required if p not in params or params[p] is None]
    
    def get_type_errors(self, params: dict, types: dict[str, type]) -> list[str]:
        errors = []
        for key, expected_type in types.items():
            if key in params and params[key] is not None:
                if not isinstance(params[key], expected_type):
                    errors.append(f"{key} must be {expected_type.__name__}, got {type(params[key]).__name__}")
        return errors


class ToolInputValidator(InputValidator):
    """Validate tool inputs with schema and context awareness."""
    
    def __init__(
        self,
        required_params: list[str] | None = None,
        optional_params: list[str] | None = None,
        param_types: dict[str, type] | None = None,
        param_validators: dict[str, Callable] | None = None,
        preconditions: list[Callable] | None = None,
    ):
        self.required_params = required_params or []
        self.optional_params = optional_params or []
        self.param_types = param_types or {}
        self.param_validators = param_validators or {}
        self.preconditions = preconditions or []
    
    def validate(self, params: dict, context: AgenticContext) -> ValidationResult:
        errors = []
        warnings = []
        suggestions = []
        missing = self.get_missing_params(params, self.required_params)
        
        if missing:
            errors.append(f"Missing required parameters: {', '.join(missing)}")
        
        type_errors = self.get_type_errors(params, self.param_types)
        errors.extend(type_errors)
        
        for param_name, validator in self.param_validators.items():
            if param_name in params:
                try:
                    result = validator(params[param_name], context)
                    if result:
                        warnings.append(result)
                except Exception as e:
                    errors.append(f"Validation failed for {param_name}: {e}")
        
        for precondition in self.preconditions:
            try:
                if not precondition(params, context):
                    errors.append(f"Precondition not met: {precondition.__name__}")
            except Exception as e:
                warnings.append(f"Precondition check failed: {e}")
        
        if errors:
            return ValidationResult(
                result=ValidationStatus.INVALID,
                errors=errors,
                suggestions=suggestions,
            )
        
        if warnings:
            return ValidationResult(
                result=ValidationStatus.VALID,
                warnings=warnings,
            )
        
        return ValidationResult(result=ValidationStatus.VALID)


@dataclass
class StageExecutionResult(Generic[T]):
    """Result from stage execution with lazy exit support."""
    value: T | None = None
    status: str = "completed"
    error: Exception | None = None
    duration_ms: float = 0.0
    cached: bool = False
    skipped: bool = False
    skip_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def success(self) -> bool:
        return self.status == "completed" and not self.skipped


class LazyStage(ABC, Generic[T, R]):
    """Stage with lazy execution and early exit support.
    
    A LazyStage can:
    1. Validate inputs before execution
    2. Check preconditions and skip if not needed
    3. Use cached results when available
    4. Report early termination reasons
    """
    
    name: str = "lazy_stage"
    description: str = ""
    
    def __init__(
        self,
        precondition: Callable[[AgenticContext], bool] | None = None,
        skip_condition: Callable[[AgenticContext], bool] | None = None,
        cache_key_fn: Callable[[AgenticContext], str | None] | None = None,
        early_exit_fn: Callable[[AgenticContext], bool] | None = None,
        validator: InputValidator | None = None,
        timeout_seconds: float = 300.0,
        retry_count: int = 0,
    ):
        self.precondition = precondition
        self.skip_condition = skip_condition
        self.cache_key_fn = cache_key_fn
        self.early_exit_fn = early_exit_fn
        self.validator = validator
        self.timeout_seconds = timeout_seconds
        self.retry_count = retry_count
        self.retry_delay: float = 1.0
        self.retry_backoff: float = 2.0
    
    async def validate_inputs(self, params: dict, context: AgenticContext) -> ValidationResult:
        if self.validator:
            return self.validator.validate(params, context)
        return ValidationResult(result=ValidationStatus.VALID)
    
    def should_skip(self, context: AgenticContext) -> tuple[bool, str | None]:
        if self.skip_condition:
            if self.skip_condition(context):
                return True, "Skip condition met"
        
        if self.precondition:
            if not self.precondition(context):
                return True, "Precondition not met"
        
        return False, None
    
    def get_cache_key(self, params: dict, context: AgenticContext) -> str | None:
        if self.cache_key_fn:
            return self.cache_key_fn(context)
        
        import hashlib
        import json
        key_data = {
            "stage": self.name,
            "params": {k: str(v) for k, v in params.items()},
            "project": context.project.project_name,
            "goal": context.goal.goal_statement[:50] if context.goal else None,
        }
        return hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()[:16]
    
    def should_early_exit(self, context: AgenticContext) -> bool:
        if self.early_exit_fn:
            return self.early_exit_fn(context)
        return False
    
    @abstractmethod
    async def execute(self, params: dict, context: AgenticContext) -> StageExecutionResult[R]:
        """Execute the stage. Override in subclasses."""
        pass
    
    async def run(self, params: dict, context: AgenticContext) -> StageExecutionResult[R]:
        """Run stage with lazy execution and early exit."""
        start_time = time.perf_counter()
        
        should_skip, skip_reason = self.should_skip(context)
        if should_skip:
            return StageExecutionResult(
                status="skipped",
                skipped=True,
                skip_reason=skip_reason,
                duration_ms=(time.perf_counter() - start_time) * 1000,
            )
        
        if self.should_early_exit(context):
            return StageExecutionResult(
                status="early_exit",
                skipped=True,
                skip_reason="Early exit condition met",
                duration_ms=(time.perf_counter() - start_time) * 1000,
            )
        
        validation = await self.validate_inputs(params, context)
        if validation.result == ValidationStatus.INVALID:
            return StageExecutionResult(
                status="validation_failed",
                error=ValueError("; ".join(validation.errors)),
                metadata={"validation_errors": validation.errors},
                duration_ms=(time.perf_counter() - start_time) * 1000,
            )
        
        if validation.result == ValidationStatus.NEEDS_INPUT:
            return StageExecutionResult(
                status="needs_input",
                metadata={
                    "missing_params": validation.missing_params,
                    "suggestions": validation.suggestions,
                },
                duration_ms=(time.perf_counter() - start_time) * 1000,
            )
        
        cache_key = self.get_cache_key(params, context)
        if cache_key:
            cached_result = context.get_cached_tool_result(self.name, cache_key)
            if cached_result:
                return StageExecutionResult(
                    value=cached_result,
                    status="completed",
                    cached=True,
                    duration_ms=(time.perf_counter() - start_time) * 1000,
                )
        
        # Execute with retry support
        current_delay = self.retry_delay
        last_error = None
        
        for attempt in range(self.retry_count + 1):
            try:
                if self.timeout_seconds:
                    result = await asyncio.wait_for(
                        self.execute(params, context),
                        timeout=self.timeout_seconds,
                    )
                else:
                    result = await self.execute(params, context)
                
                if cache_key and result.value is not None:
                    context.cache_tool_result(self.name, cache_key, result.value)
                
                result.duration_ms = (time.perf_counter() - start_time) * 1000
                return result
                
            except asyncio.TimeoutError:
                last_error = TimeoutError(f"Stage {self.name} timed out after {self.timeout_seconds}s")
                if attempt < self.retry_count:
                    logger.warning(f"[{self.name}] Timeout on attempt {attempt + 1}, retrying in {current_delay:.1f}s")
                    await asyncio.sleep(current_delay)
                    current_delay *= self.retry_backoff
                continue
            except Exception as e:
                last_error = e
                if attempt < self.retry_count:
                    logger.warning(f"[{self.name}] Error on attempt {attempt + 1}: {e}, retrying in {current_delay:.1f}s")
                    await asyncio.sleep(current_delay)
                    current_delay *= self.retry_backoff
                continue
        
        # All retries exhausted
        return StageExecutionResult(
            status="failed",
            error=last_error or Exception("Unknown error"),
            duration_ms=(time.perf_counter() - start_time) * 1000,
        )


class FunctionLazyStage(LazyStage):
    """LazyStage wrapper for functions."""
    
    def __init__(
        self,
        func: Callable,
        name: str | None = None,
        description: str = "",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.func = func
        self.name = name or func.__name__
        self.description = description or func.__doc__ or ""
    
    async def execute(self, params: dict, context: AgenticContext) -> StageExecutionResult:
        try:
            if inspect.iscoroutinefunction(self.func):
                result = await self.func(params, context)
            else:
                result = self.func(params, context)
            
            return StageExecutionResult(
                value=result,
                status="completed",
            )
        except Exception as e:
            return StageExecutionResult(
                status="failed",
                error=e,
            )


class AgenticPipeline:
    """Pipeline with full agentic decision-making.
    
    Features:
    - Lazy execution (skip stages when possible)
    - Early exit on failures
    - Validation before execution
    - Context-aware caching
    - Iterative refinement with evaluation
    
    Example:
        pipeline = AgenticPipeline("code_improvement")
        pipeline.add_stage(validate_stage)
        pipeline.add_stage(build_stage, skip_if=no_code_changes)
        pipeline.add_stage(test_stage, precondition=has_tests)
        pipeline.add_stage(review_stage)
        
        result = await pipeline.execute(context)
    """
    
    def __init__(
        self,
        name: str,
        description: str = "",
        max_iterations: int = 5,
        enable_parallel: bool = False,
        fail_fast: bool = True,
    ):
        self.name = name
        self.description = description
        self.max_iterations = max_iterations
        self.enable_parallel = enable_parallel
        self.fail_fast = fail_fast
        self._stages: list[LazyStage] = []
        self._stage_ids: dict[str, int] = {}
        # Stage ids in execution order; kept parallel to _stages so a duplicate
        # id can never desync index lookups during execute().
        self._ordered_stage_ids: list[str] = []
        # Declared at build time, registered into the context's dependency graph
        # when execute() runs (no context exists yet while stages are added).
        self._declared_dependencies: dict[str, list[str]] = {}

    def add_stage(
        self,
        stage: LazyStage,
        stage_id: str | None = None,
        depends_on: list[str] | None = None,
    ) -> "AgenticPipeline":
        stage_id = stage_id or stage.name

        # Keep ids unique so _stage_ids and _stages stay in lockstep.
        if stage_id in self._stage_ids:
            suffix = 2
            while f"{stage_id}_{suffix}" in self._stage_ids:
                suffix += 1
            stage_id = f"{stage_id}_{suffix}"

        self._stages.append(stage)
        self._stage_ids[stage_id] = len(self._stages) - 1
        self._ordered_stage_ids.append(stage_id)

        if depends_on:
            self._declared_dependencies[stage_id] = list(depends_on)

        return self

    def _register_dependencies(self, context: "AgenticContext") -> None:
        """Register stages and their declared dependencies on the context graph.

        Every stage becomes a node so mark_complete()/is_ready() can track it.
        An edge runs dependency -> dependent, matching DependencyGraph semantics
        where get_dependencies(x) returns the nodes pointing at x.
        """
        for stage_id in self._ordered_stage_ids:
            if stage_id not in context.dependencies.nodes:
                context.dependencies.add_node(stage_id, "pipeline_stage", {"pipeline": self.name})

        for stage_id, dep_ids in self._declared_dependencies.items():
            for dep_id in dep_ids:
                if dep_id in self._stage_ids:
                    context.dependencies.add_edge(dep_id, stage_id)
                else:
                    logger.warning(
                        f"Pipeline '{self.name}': stage '{stage_id}' declares unknown "
                        f"dependency '{dep_id}' — ignoring"
                    )
    
    def stage(
        self,
        name: str | None = None,
        precondition: Callable | None = None,
        skip_if: Callable | None = None,
        cache_key: Callable | None = None,
    ) -> Callable:
        """Decorator to add a function as a stage."""
        def decorator(func: Callable) -> Callable:
            stage = FunctionLazyStage(
                func=func,
                name=name,
                precondition=precondition,
                skip_condition=skip_if,
                cache_key_fn=cache_key,
            )
            self.add_stage(stage)
            return func
        return decorator
    
    async def execute(
        self,
        context: AgenticContext,
        params: dict | None = None,
        on_stage_complete: Callable | None = None,
        on_stage_skip: Callable | None = None,
        on_error: Callable | None = None,
    ) -> StageExecutionResult:
        """Execute the pipeline with lazy evaluation."""
        params = params or {}
        start_time = time.perf_counter()

        self._register_dependencies(context)

        results = []
        current_data = params
        all_skipped = True

        for i, stage in enumerate(self._stages):
            stage_id = self._ordered_stage_ids[i] if i < len(self._ordered_stage_ids) else f"stage_{i}"
            
            should_skip, skip_reason = stage.should_skip(context)
            
            if should_skip:
                if on_stage_skip:
                    try:
                        on_stage_skip(stage_id, skip_reason, context)
                    except Exception as e:
                        logger.warning(f"on_stage_skip callback failed for stage {stage_id}: {e}")
                results.append(StageExecutionResult(
                    status="skipped",
                    skipped=True,
                    skip_reason=skip_reason,
                ))
                continue
            
            all_skipped = False
            
            if not context.can_proceed_with(stage_id):
                results.append(StageExecutionResult(
                    status="blocked",
                    skipped=True,
                    skip_reason="Dependencies not complete",
                ))
                if self.fail_fast:
                    return StageExecutionResult(
                        status="blocked",
                        error=RuntimeError(f"Stage {stage_id} blocked by incomplete dependencies"),
                        metadata={"stage_id": stage_id, "results": [r.status for r in results]},
                        duration_ms=(time.perf_counter() - start_time) * 1000,
                    )
                continue
            
            stage_result = await stage.run(current_data, context)
            
            context.history.add_step(
                step_id=stage_id,
                step_type=stage.__class__.__name__,
                action=stage.name,
                result=stage_result.value if stage_result.success else None,
                success=stage_result.success,
                duration_ms=stage_result.duration_ms,
            )
            
            context.dependencies.mark_complete(stage_id)
            
            if stage_result.success:
                context.set_stage_result(stage_id, stage_result.value)
                if stage_result.value is not None:
                    current_data = {**current_data, **stage_result.value}
                    
                if on_stage_complete:
                    try:
                        on_stage_complete(stage_id, stage_result, context)
                    except Exception as e:
                        logger.warning(f"on_stage_complete callback failed for stage {stage_id}: {e}")
            else:
                results.append(stage_result)
                
                if on_error:
                    try:
                        on_error(stage_id, stage_result.error, context)
                    except Exception as e:
                        logger.warning(f"on_error callback failed for stage {stage_id}: {e}")
                
                if self.fail_fast and stage_result.status in ("failed", "timeout", "validation_failed"):
                    return StageExecutionResult(
                        status="failed",
                        error=stage_result.error,
                        metadata={
                            "failed_stage": stage_id,
                            "stage_index": i,
                            "results": [r.status for r in results],
                        },
                        duration_ms=(time.perf_counter() - start_time) * 1000,
                    )
        
        if all_skipped:
            return StageExecutionResult(
                status="skipped",
                skipped=True,
                skip_reason="All stages skipped",
                metadata={"results": [r.status for r in results]},
                duration_ms=(time.perf_counter() - start_time) * 1000,
            )
        
        return StageExecutionResult(
            value={"results": results, "final_data": current_data},
            status="completed",
            metadata={
                "stages_executed": len([r for r in results if not r.skipped]),
                "stages_skipped": len([r for r in results if r.skipped]),
            },
            duration_ms=(time.perf_counter() - start_time) * 1000,
        )


# Common validators
common_validators = {
    "project_path": ToolInputValidator(
        required_params=["project_path"],
        param_types={"project_path": str},
        preconditions=[
            lambda p, c: Path(p["project_path"]).exists() if "project_path" in p else False
        ]
    ),
    "file_path": ToolInputValidator(
        required_params=["file_path"],
        param_types={"file_path": str},
    ),
    "task_id": ToolInputValidator(
        required_params=["task_id"],
        param_types={"task_id": str},
    ),
}


_agentic_context_instance: AgenticContext | None = None
_agentic_context_lock = threading.Lock()


def get_agentic_context() -> AgenticContext:
    """Thread-safe singleton getter for AgenticContext."""
    global _agentic_context_instance
    if _agentic_context_instance is None:
        with _agentic_context_lock:
            if _agentic_context_instance is None:
                _agentic_context_instance = AgenticContext(workflow_id="default")
    return _agentic_context_instance


# Convenience functions
def lazy_stage(
    func: Callable,
    name: str | None = None,
    **kwargs,
) -> FunctionLazyStage:
    """Create a lazy stage from a function."""
    return FunctionLazyStage(func, name=name, **kwargs)


def create_context(
    project_path: str | None = None,
    goal: str | None = None,
    **kwargs,
) -> AgenticContext:
    """Create an agentic context with project and goal."""
    import uuid
    
    context = AgenticContext(
        workflow_id=str(uuid.uuid4())[:8],
        **kwargs,
    )
    
    if project_path:
        context.project.project_path = project_path
        context.project.project_name = Path(project_path).name
    
    if goal:
        context.goal.goal_statement = goal
    
    return context