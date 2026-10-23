"""Sub-Agent Orchestration for Jarvis.

A lightweight, practical multi-agent system that integrates with Jarvis's
existing tool system. Designed for real-world use:

1. Task Decomposition: Break down complex requests into sub-tasks
2. Agent Selection: Match tasks to capable agents (researcher, coder, etc.)
3. Parallel Execution: Run independent tasks concurrently
4. Result Synthesis: Combine outputs into final answer

ARCHITECTURE:
    User Request → Orchestrator → Task Planner
                              ↓
                        ┌─────┼─────┐
                        ↓     ↓     ↓
                    Agent1 Agent2 Agent3  (parallel)
                        ↓     ↓     ↓
                        └─────┼─────┘
                              ↓
                        Synthesizer → Final Result
                        
                    ModuleBridge ←→ ContextCompressor (bidirectional)

Hardware-optimized for AMD HX 370:
- GPU/NPU-aware parallelism tuning
- Memory pressure handling
- Adaptive task scheduling

KEY DESIGN DECISIONS:
- Uses Jarvis's existing tool system (no duplication)
- LLM-based task decomposition when available
- Heuristic fallback for decomposition
- Respects context limits
- Progress tracking via optional callback
- Event-driven coordination with context system
"""
import asyncio
import atexit
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable
import re

from core.helpers import safe_error_message
from core.module_bridge import (
    get_bridge,
    BridgeEvent,
    ModuleState,
    ModuleHandle,
)

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="subagent_")
atexit.register(_executor.shutdown, wait=False)


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentRole(Enum):
    ORCHESTRATOR = "orchestrator"
    RESEARCHER = "researcher"
    CODER = "coder"
    ANALYZER = "analyzer"
    EXECUTOR = "executor"
    SYNTHESIZER = "synthesizer"


@dataclass
class SubTask:
    """A task to be executed by an agent."""
    id: str
    title: str
    description: str
    dependencies: list[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent: str | None = None
    result: Any = None
    error: str | None = None
    priority: int = 5
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    retries: int = 0
    max_retries: int = 2
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "dependencies": self.dependencies,
            "status": self.status.value,
            "assigned_agent": self.assigned_agent,
            "error": self.error,
            "priority": self.priority,
            "retries": self.retries,
        }


@dataclass
class AgentConfig:
    """Configuration for an agent type."""
    id: str
    role: AgentRole
    name: str
    description: str
    capabilities: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    system_prompt: str = ""


# Default agent configurations
DEFAULT_AGENTS = [
    AgentConfig(
        id="researcher",
        role=AgentRole.RESEARCHER,
        name="Research Agent",
        description="Gathers information, searches documentation, finds resources",
        capabilities=["search", "read", "fetch", "analyze"],
    ),
    AgentConfig(
        id="coder",
        role=AgentRole.CODER,
        name="Code Agent",
        description="Writes, modifies, debugs, and explains code",
        capabilities=["code", "debug", "refactor", "test", "explain"],
    ),
    AgentConfig(
        id="analyzer",
        role=AgentRole.ANALYZER,
        name="Analysis Agent",
        description="Analyzes data, patterns, logs, and outputs insights",
        capabilities=["analyze", "summarize", "compare", "evaluate"],
    ),
    AgentConfig(
        id="executor",
        role=AgentRole.EXECUTOR,
        name="Executor Agent",
        description="Executes tools, commands, and performs actions",
        capabilities=["execute", "run", "apply", "modify"],
    ),
    AgentConfig(
        id="synthesizer",
        role=AgentRole.SYNTHESIZER,
        name="Synthesis Agent",
        description="Combines results, creates summaries, produces final outputs",
        capabilities=["synthesize", "summarize", "format", "report"],
    ),
]


class TaskPlanner:
    """Plans and decomposes tasks."""

    # Patterns for heuristic decomposition
    DECOMPOSITION_PATTERNS = [
        (r'(.+?)\s+then\s+(.+)', ['sequential', 2]),
        (r'(.+?)\s+and\s+also\s+(.+)', ['parallel', 2]),
        (r'(.+?)\s+after\s+that\s+(.+)', ['sequential', 2]),
        (r'first\s+(.+?)\s*,?\s*then\s+(.+)', ['sequential', 2]),
        (r'(.+?)\s+while\s+(.+)', ['parallel', 2]),
        (r'(.+?)\s+simultaneously\s+(.+)', ['parallel', 2]),
        (r'please\s+(.+?\.)(.+?\.)(.+?\.)', ['sequential', 3]),
    ]

    # Cache for decomposition results (simple LRU-like)
    _decompose_cache: dict[str, tuple[list[SubTask], float]] = {}
    _cache_max_size = 200   # More cached decompositions
    _cache_ttl = 600.0  # 10 minutes
    _cache_lock = threading.Lock()

    def __init__(self, llm_client=None):
        self.llm = llm_client

    def _get_cache_key(self, objective: str) -> str:
        """Generate cache key for decomposition."""
        # Normalize objective for caching
        normalized = " ".join(objective.lower().split()[:20])
        return f"decomp_{hash(normalized)}"

    def _check_cache(self, objective: str) -> list[SubTask] | None:
        """Check cache for existing decomposition."""
        import time as time_module
        key = self._get_cache_key(objective)
        with self._cache_lock:
            if key in self._decompose_cache:
                tasks, timestamp = self._decompose_cache[key]
                if time_module.time() - timestamp < self._cache_ttl:
                    return tasks
        return None

    def _store_cache(self, objective: str, tasks: list[SubTask]) -> None:
        """Store decomposition in cache."""
        import time as time_module
        key = self._get_cache_key(objective)

        with self._cache_lock:
            # Simple cache eviction
            if len(self._decompose_cache) >= self._cache_max_size:
                sorted_keys = sorted(
                    self._decompose_cache.keys(),
                    key=lambda k: self._decompose_cache[k][1]
                )
                for old_key in sorted_keys[:10]:
                    del self._decompose_cache[old_key]

            self._decompose_cache[key] = (tasks, time_module.time())

    async def decompose(
        self,
        objective: str,
        context: dict | None = None,
        max_tasks: int = 8,
    ) -> list[SubTask]:
        """Decompose an objective into sub-tasks.
        
        Args:
            objective: The main objective to accomplish
            context: Optional context (files, previous results, etc.)
            max_tasks: Maximum number of tasks to create
        
        Returns:
            List of SubTask objects
        """
        # Check cache first
        cached = self._check_cache(objective)
        if cached:
            logger.debug(f"Using cached decomposition for: {objective[:50]}...")
            return cached

        # Try LLM-based decomposition first
        if self.llm:
            try:
                tasks = await self._llm_decompose(objective, context, max_tasks)
                if tasks:
                    self._store_cache(objective, tasks)
                    return tasks
            except Exception as e:
                logger.debug(f"LLM decomposition failed: {e}")

        # Fallback to heuristic decomposition
        tasks = self._heuristic_decompose(objective, context)
        self._store_cache(objective, tasks)
        return tasks

    async def _llm_decompose(
        self,
        objective: str,
        context: dict | None,
        max_tasks: int,
    ) -> list[SubTask] | None:
        """Use LLM to decompose objective."""
        prompt = f"""Break down this request into independent sub-tasks.

Request: {objective}

Context: {json.dumps(context or {}, indent=2)[:500]}

Rules:
1. Each task should be achievable independently (except for dependencies)
2. Specify dependencies between tasks
3. Max {max_tasks} tasks
4. Each task needs: id, title, description, dependencies (list of task ids)

Output as JSON array:
[{{"id": "task_1", "title": "...", "description": "...", "dependencies": []}}]

Tasks:"""

        try:
            if hasattr(self.llm, 'generate'):
                response = self.llm.generate(prompt, max_tokens=800)
            elif hasattr(self.llm, 'chat'):
                response = self.llm.chat([{"role": "user", "content": prompt}])
                if isinstance(response, dict):
                    response = response.get("content", "")
            else:
                return None

            # Extract JSON
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                task_data = json.loads(json_match.group())

                tasks = []
                for t in task_data[:max_tasks]:
                    tasks.append(SubTask(
                        id=t.get("id", f"task_{len(tasks)+1}"),
                        title=t.get("title", "Task"),
                        description=t.get("description", ""),
                        dependencies=t.get("dependencies", []),
                        priority=t.get("priority", 5),
                    ))

                # Validate dependencies
                task_ids = {t.id for t in tasks}
                for task in tasks:
                    task.dependencies = [d for d in task.dependencies if d in task_ids]

                return tasks
        except Exception as e:
            logger.debug(f"LLM decomposition error: {e}")

        return None

    def _heuristic_decompose(
        self,
        objective: str,
        context: dict | None,
    ) -> list[SubTask]:
        """Decompose using simple heuristics."""
        objective_lower = objective.lower()

        # Check for decomposition patterns
        for pattern, (mode, count) in self.DECOMPOSITION_PATTERNS:
            match = re.search(pattern, objective_lower, re.IGNORECASE)
            if match:
                tasks = []
                for i, group in enumerate(match.groups()):
                    if group:
                        tasks.append(SubTask(
                            id=f"task_{i+1}",
                            title=group.strip().capitalize()[:50],
                            description=group.strip(),
                            dependencies=[] if i == 0 else ["task_1"] if mode == 'sequential' else [],
                        ))
                if len(tasks) >= 2:
                    return tasks

        # Split by common delimiters
        for delim in [', and ', '; ', '. Then ', '. Also ']:
            if delim in objective_lower:
                parts = objective.split(delim)
                if 2 <= len(parts) <= 5:
                    return [
                        SubTask(
                            id=f"task_{i+1}",
                            title=part.strip().capitalize()[:50],
                            description=part.strip(),
                            dependencies=[f"task_{i}"] if i > 0 else [],
                        )
                        for i, part in enumerate(parts)
                        if part.strip()
                    ]

        # Single task fallback
        return [SubTask(
            id="task_1",
            title=objective[:50],
            description=objective,
        )]

    def build_dag(self, tasks: list[SubTask]) -> list[list[SubTask]]:
        """Build execution levels from task dependencies.
        
        Returns a list of levels where each level contains tasks
        that can be executed in parallel.
        """
        if not tasks:
            return []

        # Build dependency graph
        task_map = {t.id: t for t in tasks}
        levels = []
        remaining = set(t.id for t in tasks)
        completed = set()

        while remaining:
            # Find tasks with all dependencies met
            level = []
            for tid in list(remaining):
                task = task_map[tid]
                if all(dep in completed for dep in task.dependencies):
                    level.append(task)

            if not level:
                # Circular dependency - add remaining tasks
                logger.warning("Circular dependency detected, adding remaining tasks")
                level = [task_map[tid] for tid in remaining]

            levels.append(level)

            for task in level:
                completed.add(task.id)
                remaining.discard(task.id)

        return levels


class AgentPool:
    """Pool of available agents."""

    def __init__(self, llm_client=None, tools: dict | None = None):
        self.llm = llm_client
        self.tools = tools or {}
        self.agents: dict[str, AgentConfig] = {}
        self._busy_agents: set[str] = set()
        self._lock = threading.Lock()

        # Initialize default agents
        for config in DEFAULT_AGENTS:
            self.agents[config.id] = config

    def add_agent(self, config: AgentConfig):
        """Add a custom agent configuration."""
        with self._lock:
            self.agents[config.id] = config

    def select_agent(self, task: SubTask) -> str | None:
        """Select the best agent for a task.
        
        Returns agent ID or None if no suitable agent found.
        """
        best_agent = None
        best_score = 0.0

        description_lower = task.description.lower()

        for agent_id, config in self.agents.items():
            if agent_id in self._busy_agents:
                continue

            score = self._score_agent(config, description_lower)

            if score > best_score:
                best_score = score
                best_agent = agent_id

        return best_agent

    def _score_agent(self, config: AgentConfig, description: str) -> float:
        """Score how well an agent matches a task description."""
        score = 0.5  # Base score

        # Role-based keywords
        role_keywords = {
            AgentRole.RESEARCHER: ["search", "find", "lookup", "research", "gather", "information"],
            AgentRole.CODER: ["code", "implement", "debug", "fix", "write", "program", "function"],
            AgentRole.ANALYZER: ["analyze", "examine", "evaluate", "assess", "review"],
            AgentRole.EXECUTOR: ["execute", "run", "perform", "apply", "do"],
            AgentRole.SYNTHESIZER: ["synthesize", "summarize", "combine", "merge", "report"],
        }

        keywords = role_keywords.get(config.role, [])
        matches = sum(1 for kw in keywords if kw in description)
        score += matches * 0.15

        # Capability matching
        for cap in config.capabilities:
            if cap in description:
                score += 0.1

        return min(score, 1.0)

    def mark_busy(self, agent_id: str):
        """Mark an agent as busy."""
        with self._lock:
            self._busy_agents.add(agent_id)

    def mark_available(self, agent_id: str):
        """Mark an agent as available."""
        with self._lock:
            self._busy_agents.discard(agent_id)

    def get_available_agents(self) -> list[str]:
        """Get list of available agent IDs."""
        with self._lock:
            return [aid for aid in self.agents if aid not in self._busy_agents]


class SubAgentOrchestrator:
    """Orchestrates sub-agent execution for complex tasks.
    
    Features:
    - LLM-powered task decomposition
    - Parallel and sequential execution modes
    - Progress tracking and reporting
    - Bidirectional sensing with ContextCompressor via ModuleBridge
    
    Usage:
        orchestrator = SubAgentOrchestrator(llm_client=llm, tools=agent.tools)
        
        result = await orchestrator.execute(
            "Search for Python files, analyze each one, and create a summary"
        )
        
        print(result["final_answer"])
    """

    def __init__(
        self,
        llm_client=None,
        tools: dict | None = None,
        max_parallel: int = 3,
        progress_callback: Callable[[str, float, dict], None] | None = None,
        hardware_config: Any = None,
    ):
        self.llm = llm_client
        self.tools = tools or {}
        self.max_parallel = max_parallel
        self.progress_callback = progress_callback
        self._hardware_config = hardware_config

        self._init_hardware()

        self.planner = TaskPlanner(llm_client)
        self.pool = AgentPool(llm_client, tools)

        self.tasks: dict[str, SubTask] = {}
        self.results: dict[str, Any] = {}
        self._lock = asyncio.Lock()
        self.MAX_RESULTS = 500

        self._progress_file = Path.home() / ".jarvis" / "subagent_progress.json"

        self._bridge: ModuleHandle | None = None
        self._memory_pressure = False
        self._available_tokens = float('inf')
        self._init_bridge()

    def _init_hardware(self) -> None:
        """Initialize hardware-aware scheduling."""
        try:
            from core.hardware import get_hardware_detector, get_memory_pool
            self._hardware = get_hardware_detector()
            self._memory_pool = get_memory_pool(self._hardware_config)

            if self._hardware.use_gpu():
                self.max_parallel = min(self.max_parallel, 4)
                logger.info(f"GPU detected, max parallel tasks: {self.max_parallel}")

            if self._hardware.use_npu():
                logger.info("NPU available for accelerated inference")

            cpu_count = self._hardware.accelerators[0].compute_units if self._hardware.accelerators else 8
            self.max_parallel = min(self.max_parallel, max(2, cpu_count // 4))

            logger.debug(f"Sub-agent parallelism: {self.max_parallel}")

        except Exception as e:
            logger.debug(f"Hardware detection not available: {e}")
            self._hardware = None
            self._memory_pool = None

    def _init_bridge(self) -> None:
        """Initialize connection to module bridge."""
        try:
            bridge = get_bridge()
            self._bridge = bridge.register_module("SubAgentOrchestrator")
            bridge.connect_sub_agent_orchestrator(self)

            # Subscribe to memory pressure events from ContextCompressor
            self._bridge.subscribe(BridgeEvent.MEMORY_PRESSURE, self._on_memory_pressure)
            self._bridge.subscribe(BridgeEvent.COMPRESSION_DONE, self._on_compression_done)
            self._bridge.subscribe(BridgeEvent.COMPRESSION_NEEDED, self._on_compression_needed)

            logger.debug("SubAgentOrchestrator connected to module bridge")
        except Exception as e:
            logger.debug(f"Bridge connection deferred: {e}")

    def _on_memory_pressure(self, message) -> None:
        """Handle memory pressure event from ContextCompressor."""
        data = message.data
        self._memory_pressure = True
        self._available_tokens = data.get('available_tokens', 0)
        logger.info(f"Memory pressure detected: {data.get('level', 0):.1%} used, {self._available_tokens} tokens available")

    def _on_compression_done(self, message) -> None:
        """Handle compression completion - memory recovered."""
        self._memory_pressure = False
        data = message.data
        logger.debug(f"Memory recovered: {data.get('tokens_freed', 0)} tokens freed")

    def _on_compression_needed(self, message) -> None:
        """Handle compression request - may need to reduce parallelism."""
        urgency = message.data.get('urgency', 0.5)
        if urgency > 0.7 and self.max_parallel > 1:
            # Reduce parallelism to help memory situation
            self.max_parallel = max(1, self.max_parallel - 1)
            logger.info(f"Reduced parallelism to {self.max_parallel} due to memory pressure")

    def on_memory_pressure(self, pressure_level: float, available_tokens: int) -> None:
        """Called by bridge when context compressor is under pressure."""
        self._memory_pressure = True
        self._available_tokens = available_tokens

        # Adjust parallelism based on pressure
        if pressure_level > 0.9:
            self.max_parallel = 1
        elif pressure_level > 0.8:
            self.max_parallel = min(2, self.max_parallel)

    def on_memory_recovered(self, freed_tokens: int) -> None:
        """Called by bridge when compression completes."""
        self._memory_pressure = False
        self._available_tokens += freed_tokens

    def should_delegate_or_compress(self) -> str:
        """Determine strategy based on current state."""
        if self._memory_pressure and self._available_tokens < 2000:
            return "compress_first"
        elif self._memory_pressure:
            return "delegate_sequential"
        else:
            return "delegate_parallel"

    async def execute(
        self,
        objective: str,
        context: dict | None = None,
        mode: str = "auto",
    ) -> dict:
        """Execute an objective using sub-agents.
        
        Args:
            objective: The goal to accomplish
            context: Additional context (files, data, etc.)
            mode: Execution mode - "auto", "parallel", "sequential"
        
        Returns:
            Dict with:
            - success: bool
            - final_answer: str
            - tasks: dict of task_id -> status
            - results: dict of task_id -> output
            - duration: float
        """
        start_time = time.time()

        self._report_progress("Planning", 0.1, {"phase": "decomposition"})

        # 1. Decompose
        tasks = await self.planner.decompose(objective, context)

        if not tasks:
            return {
                "success": False,
                "final_answer": "Could not decompose objective into tasks",
                "tasks": {},
                "results": {},
                "duration": time.time() - start_time,
            }

        # 2. Build execution DAG
        dag = self.planner.build_dag(tasks)

        # 3. Initialize task tracking
        async with self._lock:
            self.tasks = {t.id: t for t in tasks}
            self.results = {}

        # 4. Execute
        self._report_progress(f"Executing {len(tasks)} tasks", 0.2, {"phase": "execution"})

        # Emit event for task delegation
        if self._bridge:
            self._bridge.emit(
                BridgeEvent.TASK_DELEGATED,
                {
                    "objective": objective[:200],
                    "task_count": len(tasks),
                    "mode": mode,
                    "max_parallel": self.max_parallel,
                }
            )
            self._bridge.update_state(ModuleState.BUSY, {
                "tasks": len(tasks),
                "mode": mode,
            })

        try:
            if mode == "parallel":
                await self._execute_parallel(dag)
            elif mode == "sequential":
                await self._execute_sequential(dag)
            else:
                await self._execute_auto(dag)
        except Exception as e:
            logger.error(f"Execution failed: {e}")
            return {
                "success": False,
                "final_answer": f"Execution error: {safe_error_message(e)}",
                "tasks": {t.id: t.status.value for t in tasks},
                "results": self.results,
                "duration": time.time() - start_time,
            }

        # 5. Synthesize
        self._report_progress("Synthesizing results", 0.9, {"phase": "synthesis"})

        final_result = await self._synthesize(objective)

        self._report_progress("Complete", 1.0, {"phase": "done"})

        # Emit completion event
        if self._bridge:
            self._bridge.emit(
                BridgeEvent.TASK_COMPLETED,
                {
                    "objective": objective[:200],
                    "success": all(t.status == TaskStatus.COMPLETED for t in tasks),
                    "task_count": len(tasks),
                    "result_length": len(final_result),
                }
            )
            self._bridge.update_state(ModuleState.ACTIVE)

        return {
            "success": all(t.status == TaskStatus.COMPLETED for t in tasks),
            "final_answer": final_result,
            "tasks": {t.id: t.status.value for t in tasks},
            "results": self.results,
            "duration": time.time() - start_time,
        }

    async def _execute_auto(self, dag: list[list[SubTask]]):
        """Execute with automatic parallel/sequential decision."""
        total_tasks = sum(len(level) for level in dag)
        completed = 0

        for level_idx, level in enumerate(dag):
            # Decide parallel vs sequential based on level size
            if len(level) <= self.max_parallel and len(level) > 1:
                # Parallel execution
                tasks_str = ", ".join(t.id for t in level)
                self._report_progress(
                    f"Running {len(level)} tasks in parallel: {tasks_str}",
                    0.2 + 0.7 * (completed / total_tasks),
                    {"level": level_idx, "parallel": True}
                )
                await asyncio.gather(
                    *[self._execute_task(t) for t in level],
                    return_exceptions=True
                )
            else:
                # Sequential execution
                for task in level:
                    self._report_progress(
                        f"Running {task.id}: {task.title[:30]}",
                        0.2 + 0.7 * (completed / total_tasks),
                        {"task": task.id}
                    )
                    await self._execute_task(task)

            completed += len(level)

    async def _execute_parallel(self, dag: list[list[SubTask]]):
        """Execute all levels with maximum parallelism."""
        for level in dag:
            await asyncio.gather(
                *[self._execute_task(t) for t in level],
                return_exceptions=True
            )

    async def _execute_sequential(self, dag: list[list[SubTask]]):
        """Execute all tasks sequentially."""
        for level in dag:
            for task in level:
                await self._execute_task(task)

    async def _execute_task(self, task: SubTask) -> bool:
        """Execute a single task.
        
        Returns True if successful, False otherwise.
        """
        task.status = TaskStatus.RUNNING
        task.started_at = time.time()

        # Check dependencies
        for dep_id in task.dependencies:
            if dep_id in self.tasks:
                dep_task = self.tasks[dep_id]
                if dep_task.status != TaskStatus.COMPLETED:
                    task.status = TaskStatus.FAILED
                    task.error = f"Dependency {dep_id} not completed"
                    return False

        # Select agent
        agent_id = self.pool.select_agent(task)

        if not agent_id:
            # Try to execute without specific agent (direct tool use or LLM call)
            success = await self._execute_with_llm(task)
        else:
            self.pool.mark_busy(agent_id)
            task.assigned_agent = agent_id

            try:
                success = await self._execute_with_agent(task, agent_id)
            finally:
                self.pool.mark_available(agent_id)

        task.completed_at = time.time()
        return success

    async def _execute_with_agent(self, task: SubTask, agent_id: str) -> bool:
        """Execute task with a specific agent."""
        config = self.pool.agents[agent_id]

        # Get dependency results
        dep_results = {}
        for dep_id in task.dependencies:
            if dep_id in self.results:
                dep_results[dep_id] = self.results[dep_id]

        # Build prompt
        prompt = self._build_agent_prompt(config, task, dep_results)

        try:
            if not self.llm:
                task.status = TaskStatus.FAILED
                task.error = "No LLM client available"
                return False

            result = await self._call_llm(prompt)

            task.result = result
            async with self._lock:
                self.results[task.id] = result
                if len(self.results) > self.MAX_RESULTS:
                    oldest_keys = list(self.results.keys())[:len(self.results) - self.MAX_RESULTS]
                    for k in oldest_keys:
                        del self.results[k]
            task.status = TaskStatus.COMPLETED

            return True

        except Exception as e:
            logger.error(f"Task {task.id} failed: {e}")
            task.status = TaskStatus.FAILED
            task.error = str(e)

            # Retry logic
            if task.retries < task.max_retries:
                task.retries += 1
                task.status = TaskStatus.PENDING
                return await self._execute_task(task)

            return False

    async def _execute_with_llm(self, task: SubTask) -> bool:
        """Execute task directly with LLM (no specific agent)."""
        dep_results = {}
        for dep_id in task.dependencies:
            if dep_id in self.results:
                dep_results[dep_id] = self.results[dep_id]

        prompt = f"""You are helping with a sub-task.

Task: {task.title}
Description: {task.description}

{f"Context from previous tasks: {json.dumps(dep_results, indent=2)[:1000]}" if dep_results else ""}

Complete this task. Provide a clear, concise answer:"""

        try:
            result = await self._call_llm(prompt)

            task.result = result
            async with self._lock:
                self.results[task.id] = result
                if len(self.results) > self.MAX_RESULTS:
                    oldest_keys = list(self.results.keys())[:len(self.results) - self.MAX_RESULTS]
                    for k in oldest_keys:
                        del self.results[k]
            task.status = TaskStatus.COMPLETED

            return True
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            return False

    def _build_agent_prompt(
        self,
        config: AgentConfig,
        task: SubTask,
        dep_results: dict,
    ) -> str:
        """Build the prompt for an agent."""
        context_str = ""
        if dep_results:
            context_str = f"\n\nResults from previous tasks:\n{json.dumps(dep_results, indent=2)[:1500]}"

        tools_str = ""
        if config.tools and self.tools:
            available = [t for t in config.tools if t in self.tools]
            if available:
                tools_str = f"\n\nAvailable tools: {', '.join(available)}"

        return f"""You are {config.name}.
Role: {config.role.value}
Description: {config.description}
{tools_str}

Your task:
{task.description}
{context_str}

Complete this task and provide your answer:"""

    async def _call_llm(self, prompt: str) -> str:
        """Make an LLM call."""
        if hasattr(self.llm, 'generate'):
            result = self.llm.generate(prompt, max_tokens=2000)
            # Handle both async and sync generate methods
            if hasattr(result, '__await__'):
                result = await result
            return str(result)
        elif hasattr(self.llm, 'chat'):
            result = self.llm.chat([{"role": "user", "content": prompt}])
            # Handle both async and sync chat methods
            if hasattr(result, '__await__'):
                result = await result
            if isinstance(result, dict):
                return result.get("content", str(result))
            return str(result)
        else:
            raise ValueError("LLM client must have 'generate' or 'chat' method")

    async def _synthesize(self, objective: str) -> str:
        """Synthesize final result from all task results."""
        if not self.results:
            return "No results to synthesize."

        if len(self.results) == 1:
            result = list(self.results.values())[0]
            if isinstance(result, str):
                return result
            return json.dumps(result, indent=2)[:1000]

        # Combine results
        combined = []
        for task_id, result in self.results.items():
            task = self.tasks.get(task_id)
            if task:
                result_str = result if isinstance(result, str) else json.dumps(result, indent=2)
                combined.append(f"**{task.title}**:\n{result_str[:500]}")

        if not combined:
            return "No successful results."

        # Try LLM synthesis
        if self.llm:
            prompt = f"""Combine these results into a final answer for the objective.

Objective: {objective}

Results:
{chr(10).join(combined)}

Final answer:"""

            try:
                return await self._call_llm(prompt)
            except Exception as e:
                logger.debug(f"LLM call failed in synthesizer: {e}")

        # Fallback: just combine
        return "\n\n".join(combined)

    def _report_progress(self, message: str, progress: float, details: dict):
        """Report progress to callback."""
        if self.progress_callback:
            try:
                self.progress_callback(message, progress, details)
            except Exception as e:
                logger.debug(f"Progress callback error: {e}")

        logger.info(f"[SubAgent] {message} ({progress*100:.0f}%)")

    def get_stats(self) -> dict:
        """Get orchestrator statistics."""
        completed = sum(1 for t in self.tasks.values() if t.status == TaskStatus.COMPLETED)
        failed = sum(1 for t in self.tasks.values() if t.status == TaskStatus.FAILED)
        pending = sum(1 for t in self.tasks.values() if t.status == TaskStatus.PENDING)
        running = sum(1 for t in self.tasks.values() if t.status == TaskStatus.RUNNING)

        return {
            "total_tasks": len(self.tasks),
            "completed": completed,
            "failed": failed,
            "pending": pending,
            "running": running,
            "available_agents": self.pool.get_available_agents().__len__(),
        }


# Singleton
_orchestrator: SubAgentOrchestrator | None = None
_orchestrator_lock = threading.Lock()


def get_orchestrator(
    llm_client=None,
    tools: dict | None = None,
) -> SubAgentOrchestrator:
    """Get or create the orchestrator singleton."""
    global _orchestrator
    if _orchestrator is None:
        with _orchestrator_lock:
            if _orchestrator is None:
                _orchestrator = SubAgentOrchestrator(
                    llm_client=llm_client,
                    tools=tools,
                )
    return _orchestrator


def has_orchestrator() -> bool:
    """Check if orchestrator is initialized."""
    return _orchestrator is not None
