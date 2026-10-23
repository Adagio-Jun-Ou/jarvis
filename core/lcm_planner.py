"""LCM Planner - Task decomposition and structured planning.

Implements the "write_todos" pattern from LangChain Deep Agents:
1. Receive complex request
2. Decompose into structured sub-tasks
3. Route each sub-task to appropriate LLM
4. Track progress and handle failures
5. Synthesize results into coherent response

Architecture:
┌──────────────────────────────────────────────────────────────────┐
│                      LCM PLANNER                                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│   Complex Request                                                  │
│       │                                                            │
│       ▼                                                            │
│   ┌──────────────┐                                                │
│   │  DECOMPOSE   │  Break into sub-tasks with dependencies        │
│   │  (Backend)   │  Identify model for each sub-task              │
│   └──────┬───────┘                                                │
│          │                                                         │
│          ▼                                                         │
│   ┌──────────────┐                                                │
│   │  PLAN        │  Create DAG with execution order               │
│   │  (LCM)       │  Identify parallelizable tasks                 │
│   └──────┬───────┘                                                │
│          │                                                         │
│          ▼                                                         │
│   ┌──────────────┐                                                │
│   │  EXECUTE     │  Run tasks in waves (parallel where possible)  │
│   │  (Router)    │  Track progress and handle failures            │
│   └──────┬───────┘                                                │
│          │                                                         │
│          ▼                                                         │
│   ┌──────────────┐                                                │
│   │  SYNTHESIZE  │  Combine results into coherent response        │
│   │  (Primary)   │  Apply LCM post-shield validation               │
│   └──────┬───────┘                                                │
│          │                                                         │
│          ▼                                                         │
│   Final Response                                                   │
│                                                                    │
└──────────────────────────────────────────────────────────────────┘

Integration with existing systems:
- Uses PrefrontalCortex for concept-level decomposition
- Uses DualModelOrchestrator for model routing
- Uses LCMShield for pre/post validation
- Uses ContextCoordinator for context assembly
- Uses KnowledgeGraph for relevant context injection
"""
import asyncio
import hashlib
import json
import logging
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

_planner: Optional["LCMPlanner"] = None
_planner_lock = threading.Lock()


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class TaskModel(Enum):
    PRIMARY = "primary"
    BACKEND = "backend"
    PARALLEL = "parallel"


class TaskComplexity(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


@dataclass
class PlanTask:
    """A single task in a plan."""
    id: str
    description: str
    model: TaskModel = TaskModel.PRIMARY
    complexity: TaskComplexity = TaskComplexity.SIMPLE
    status: TaskStatus = TaskStatus.PENDING
    depends_on: list[str] = field(default_factory=list)
    result: str = ""
    error: str = ""
    latency_ms: float = 0.0
    tool_calls: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class Plan:
    """A structured execution plan."""
    id: str
    original_request: str
    tasks: list[PlanTask] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None
    execution_waves: list[list[str]] = field(default_factory=list)
    synthesis: str = ""
    total_latency_ms: float = 0.0
    
    def get_progress(self) -> dict:
        """Get plan progress."""
        total = len(self.tasks)
        completed = sum(1 for t in self.tasks if t.status == TaskStatus.COMPLETED)
        failed = sum(1 for t in self.tasks if t.status == TaskStatus.FAILED)
        running = sum(1 for t in self.tasks if t.status == TaskStatus.RUNNING)
        return {
            "total": total,
            "completed": completed,
            "failed": failed,
            "running": running,
            "pending": total - completed - failed - running,
            "progress_pct": (completed / total * 100) if total > 0 else 0,
        }


@dataclass
class PlannerStats:
    total_plans: int = 0
    completed_plans: int = 0
    failed_plans: int = 0
    avg_tasks_per_plan: float = 0.0
    avg_plan_latency_ms: float = 0.0


class LCMPlanner:
    """LCM-powered task decomposition and planning.
    
    Decomposes complex requests into structured execution plans,
    routes sub-tasks to the appropriate model, and synthesizes results.
    
    The planner works in 4 phases:
    1. DECOMPOSE: Break request into sub-tasks
    2. PLAN: Create execution DAG with dependency ordering
    3. EXECUTE: Run tasks in dependency waves
    4. SYNTHESIZE: Combine results into final response
    """
    
    def __init__(self):
        self._stats = PlannerStats()
        self._lock = threading.RLock()
        self._active_plans: OrderedDict[str, Plan] = OrderedDict()
        self._max_active_plans = 20
        
        self._task_templates: OrderedDict[str, dict] = OrderedDict()
        self._max_templates = 50
    
    async def create_plan(
        self,
        request: str,
        context: dict | None = None,
        backend_llm=None,
        primary_llm=None,
    ) -> Plan:
        """Create an execution plan for a complex request.
        
        Uses the backend LLM (GLM-5) for decomposition since it excels
        at structured analysis.
        """
        plan_id = f"plan_{hashlib.sha256(request.encode()).hexdigest()[:8]}_{int(time.time()*1000)}"
        
        plan = Plan(
            id=plan_id,
            original_request=request,
        )
        
        tasks = await self._decompose(request, context, backend_llm)
        plan.tasks = tasks
        
        plan.execution_waves = self._compute_waves(tasks)
        
        with self._lock:
            self._stats.total_plans += 1
            if self._stats.total_plans == 1:
                self._stats.avg_tasks_per_plan = float(len(tasks))
            else:
                self._stats.avg_tasks_per_plan = (
                    self._stats.avg_tasks_per_plan * 0.9 + len(tasks) * 0.1
                )
            self._active_plans[plan_id] = plan
            while len(self._active_plans) > self._max_active_plans:
                self._active_plans.popitem(last=False)
        
        logger.info(f"Plan {plan_id}: {len(tasks)} tasks in {len(plan.execution_waves)} waves")
        return plan
    
    async def execute_plan(
        self,
        plan: Plan,
        backend_llm=None,
        primary_llm=None,
        dual_model=None,
        agent=None,
    ) -> Plan:
        """Execute a plan by running tasks in dependency waves."""
        plan.status = TaskStatus.RUNNING
        start_time = time.time()
        
        for wave_idx, wave in enumerate(plan.execution_waves):
            logger.info(f"Plan {plan.id}: Wave {wave_idx + 1}/{len(plan.execution_waves)} - {len(wave)} tasks")
            
            wave_tasks = []
            for task_id in wave:
                task = next((t for t in plan.tasks if t.id == task_id), None)
                if task and task.status == TaskStatus.PENDING:
                    wave_tasks.append(task)
            
            if wave_tasks:
                results = await asyncio.gather(
                    *[
                        self._execute_task(task, plan, backend_llm, primary_llm, dual_model, agent)
                        for task in wave_tasks
                    ],
                    return_exceptions=True,
                )
                
                for task, result in zip(wave_tasks, results):
                    if isinstance(result, Exception):
                        task.status = TaskStatus.FAILED
                        task.error = str(result)
                    elif isinstance(result, PlanTask):
                        pass
        
        completed = sum(1 for t in plan.tasks if t.status == TaskStatus.COMPLETED)
        failed = sum(1 for t in plan.tasks if t.status == TaskStatus.FAILED)
        
        if completed > 0:
            plan.synthesis = self._synthesize_results(plan)
        
        plan.status = TaskStatus.COMPLETED if failed == 0 else TaskStatus.FAILED
        plan.completed_at = datetime.now()
        plan.total_latency_ms = (time.time() - start_time) * 1000
        
        with self._lock:
            if plan.status == TaskStatus.COMPLETED:
                self._stats.completed_plans += 1
            else:
                self._stats.failed_plans += 1
            self._stats.avg_plan_latency_ms = (
                self._stats.avg_plan_latency_ms * 0.9 + plan.total_latency_ms * 0.1
            )
        
        logger.info(f"Plan {plan.id}: {plan.status.value} in {plan.total_latency_ms:.0f}ms")
        return plan
    
    async def _decompose(
        self,
        request: str,
        context: dict | None,
        backend_llm,
    ) -> list[PlanTask]:
        """Decompose a request into sub-tasks.
        
        Uses backend LLM for structured decomposition when available,
        falls back to heuristic decomposition otherwise.
        """
        if backend_llm:
            try:
                tasks = await self._llm_decompose(request, context, backend_llm)
                if tasks:
                    return tasks
            except Exception as e:
                logger.warning(f"LLM decomposition failed: {e}, using heuristics")
        
        return self._heuristic_decompose(request)
    
    async def _llm_decompose(
        self,
        request: str,
        context: dict | None,
        backend_llm,
    ) -> list[PlanTask]:
        """Use backend LLM to decompose a request."""
        sanitized_request = request.replace("```", "''").replace("<|", "").replace("|>", "")[:2000]
        context_str = ""
        if context:
            context_str = f"\nContext: {json.dumps(context, default=str)[:500]}"
        
        prompt = f"""Decompose this request into structured sub-tasks.

Request: {sanitized_request}{context_str}

Return a JSON array of tasks. Each task should have:
- "id": short unique id (e.g., "task_1")
- "description": what this task should do
- "model": "primary" for chat/creative, "backend" for code/math/analysis
- "depends_on": list of task IDs this depends on (empty [] if none)
- "complexity": "simple", "moderate", or "complex"

Example:
[
  {{"id": "task_1", "description": "Search for relevant information", "model": "primary", "depends_on": [], "complexity": "simple"}},
  {{"id": "task_2", "description": "Analyze the findings", "model": "backend", "depends_on": ["task_1"], "complexity": "moderate"}}
]

Return ONLY the JSON array, no other text."""

        response = await backend_llm.chat(
            user_input=prompt,
            system_prompt="You are a task decomposition specialist. Break down complex requests into clear, actionable sub-tasks.",
            conversation_history=[],
        )
        
        content = response if isinstance(response, str) else str(response)
        
        return self._parse_tasks_json(content)
    
    def _parse_tasks_json(self, content: str) -> list[PlanTask]:
        """Parse LLM response into PlanTask objects."""
        try:
            json_start = content.find("[")
            json_end = content.rfind("]") + 1
            if json_start < 0 or json_end <= json_start:
                return []
            
            json_str = content[json_start:json_end]
            tasks_data = json.loads(json_str)
            
            tasks = []
            for td in tasks_data:
                if isinstance(td, dict):
                    model_str = td.get("model", "primary")
                    model = TaskModel.BACKEND if model_str == "backend" else TaskModel.PRIMARY
                    
                    complexity_str = td.get("complexity", "simple")
                    complexity = TaskComplexity.MODERATE if complexity_str == "moderate" else (
                        TaskComplexity.COMPLEX if complexity_str == "complex" else TaskComplexity.SIMPLE
                    )
                    
                    tasks.append(PlanTask(
                        id=td.get("id", f"task_{len(tasks)}"),
                        description=td.get("description", ""),
                        model=model,
                        complexity=complexity,
                        depends_on=td.get("depends_on", []),
                    ))
            
            return tasks
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Failed to parse tasks JSON: {e}")
            return []
    
    def _heuristic_decompose(self, request: str) -> list[PlanTask]:
        """Decompose using heuristics when LLM is unavailable."""
        tasks = []
        
        connectors = [" and then ", " then ", " and ", ", then ", "; then "]
        parts = [request]
        
        for connector in connectors:
            new_parts = []
            for part in parts:
                if connector.lower() in part.lower():
                    new_parts.extend(re.split(re.escape(connector), part, flags=re.IGNORECASE))
                else:
                    new_parts.append(part)
            parts = new_parts
        
        if len(parts) <= 1:
            return [PlanTask(
                id="task_1",
                description=request,
                model=TaskModel.PRIMARY,
                complexity=TaskComplexity.SIMPLE,
            )]
        
        tasks = []
        prev_task_id = None
        
        for i, part in enumerate(parts):
            desc = part.strip()
            if not desc:
                continue
            
            model = TaskModel.PRIMARY
            complexity = TaskComplexity.SIMPLE
            
            code_keywords = ["code", "program", "function", "debug", "implement", "write", "script"]
            analysis_keywords = ["analyze", "compare", "evaluate", "calculate", "compute"]
            math_keywords = ["math", "equation", "formula", "solve", "calculate"]
            
            desc_lower = desc.lower()
            if any(kw in desc_lower for kw in code_keywords + analysis_keywords + math_keywords):
                model = TaskModel.BACKEND
                complexity = TaskComplexity.MODERATE
            
            task_id = f"task_{len(tasks) + 1}"
            depends_on = [prev_task_id] if prev_task_id else []
            
            tasks.append(PlanTask(
                id=task_id,
                description=desc,
                model=model,
                complexity=complexity,
                depends_on=depends_on,
            ))
            prev_task_id = task_id
        
        return tasks if tasks else [PlanTask(
            id="task_1",
            description=request,
            model=TaskModel.PRIMARY,
        )]
    
    def _compute_waves(self, tasks: list[PlanTask]) -> list[list[str]]:
        """Compute parallel execution waves from task dependencies."""
        waves = []
        remaining = {t.id: t for t in tasks}
        completed = set()
        
        while remaining:
            wave = []
            for tid, task in list(remaining.items()):
                if all(dep in completed for dep in task.depends_on):
                    wave.append(tid)
            
            if not wave:
                remaining_ids = list(remaining.keys())
                for tid in remaining_ids:
                    task = remaining[tid]
                    unmet = [d for d in task.depends_on if d in remaining]
                    if unmet:
                        logger.warning(
                            f"Breaking cyclic dependency: task {tid} drops deps {unmet}"
                        )
                        task.depends_on = [d for d in task.depends_on if d not in remaining]
                wave = [tid for tid in remaining if all(d not in remaining for d in remaining[tid].depends_on)]
                if not wave:
                    wave = list(remaining.keys())[:3]
                    logger.warning(f"Force-executing {len(wave)} tasks with unmet dependencies")
            
            for tid in wave:
                completed.add(tid)
                del remaining[tid]
            
            waves.append(wave)
        
        return waves
    
    async def _execute_task(
        self,
        task: PlanTask,
        plan: Plan,
        backend_llm=None,
        primary_llm=None,
        dual_model=None,
        agent=None,
    ) -> PlanTask:
        """Execute a single task in the plan."""
        task.status = TaskStatus.RUNNING
        start_time = time.time()
        
        try:
            context_parts = []
            for dep_id in task.depends_on:
                dep = next((t for t in plan.tasks if t.id == dep_id), None)
                if dep and dep.result:
                    context_parts.append(f"[{dep_id} result] {dep.result[:500]}")
            
            task_context = "\n".join(context_parts) if context_parts else ""
            
            prompt = task.description
            if task_context:
                prompt = f"Context from previous steps:\n{task_context}\n\nCurrent task: {task.description}"
            
            if agent:
                result = await self._execute_via_agent(task, prompt, agent)
            elif dual_model:
                from core.dual_model import TaskType
                task_type = TaskType.ANALYSIS if task.model == TaskModel.BACKEND else TaskType.CHAT
                result = await dual_model.route_task(
                    task_type=task_type,
                    prompt=prompt,
                    system_prompt="Complete this task concisely and accurately.",
                )
            elif task.model == TaskModel.BACKEND and backend_llm:
                response = await backend_llm.chat(
                    user_input=prompt,
                    system_prompt="You are a technical expert. Complete the assigned task precisely.",
                    conversation_history=[],
                )
                result = response if isinstance(response, str) else str(response)
            elif primary_llm:
                response = await primary_llm.chat(
                    user_input=prompt,
                    system_prompt="Complete this task concisely and helpfully.",
                    conversation_history=[],
                )
                result = response if isinstance(response, str) else str(response)
            else:
                result = f"[No LLM available for task: {task.description}]"
            
            task.result = result
            task.status = TaskStatus.COMPLETED
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            logger.error(f"Task {task.id} failed: {e}")
        
        task.latency_ms = (time.time() - start_time) * 1000
        return task
    
    async def _execute_via_agent(self, task: PlanTask, prompt: str, agent) -> str:
        """Execute a task using the JarvisAgent's think+execute pipeline."""
        proposal, tool_calls = await agent.think(prompt)
        
        if tool_calls:
            results = []
            for tool_name, tool_args in tool_calls:
                try:
                    result = await agent.execute_tool(tool_name, tool_args)
                    results.append(f"[{tool_name}] {result.output}")
                except Exception as e:
                    results.append(f"[{tool_name} ERROR] {e}")
            return "\n".join(results)
        
        return proposal.response or proposal.thought or ""
    
    def _synthesize_results(self, plan: Plan) -> str:
        """Synthesize task results into a coherent response.

        Falls back to concatenation when no LLM is available for synthesis.
        When an LLM is available, uses it to intelligently merge results.
        """
        completed_tasks = [t for t in plan.tasks if t.status == TaskStatus.COMPLETED]
        
        if not completed_tasks:
            return "No tasks completed successfully."
        
        if len(completed_tasks) == 1:
            return completed_tasks[0].result

        if self.primary_llm is not None:
            try:
                return self._llm_synthesize(plan, completed_tasks)
            except Exception as e:
                logger.warning(f"LLM synthesis failed, falling back to concatenation: {e}")

        parts = []
        for task in completed_tasks:
            header = f"**{task.description}**"
            result_preview = task.result[:800] if len(task.result) > 800 else task.result
            parts.append(f"{header}\n{result_preview}")
        
        return "\n\n---\n\n".join(parts)

    def _llm_synthesize(self, plan: Plan, completed_tasks: list) -> str:
        """Use LLM to synthesize multiple task results into a coherent response."""
        results_text = ""
        for task in completed_tasks:
            results_text += f"\n## {task.description}\n{task.result[:600]}\n"

        import asyncio

        async def _synthesize():
            prompt = f"""Synthesize the following sub-task results into a single coherent response.

Original request: {plan.original_request[:500]}

Sub-task results:
{results_text[:3000]}

Instructions:
1. Combine the results into a unified response that addresses the original request
2. Remove redundancy between sub-tasks
3. Maintain all important details from each sub-task
4. Structure the response logically
5. Do NOT add information not present in the sub-task results"""

            if hasattr(self.primary_llm, 'chat'):
                result = await self.primary_llm.chat(
                    user_input=prompt,
                    system_prompt="You are a synthesis engine. Produce only the synthesized response.",
                )
            else:
                return self._fallback_synthesize(completed_tasks)

            if isinstance(result, dict):
                return result.get("content", "") or result.get("response", "")
            return str(result)

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return self._fallback_synthesize(completed_tasks)
            return loop.run_until_complete(_synthesize())
        except RuntimeError:
            return self._fallback_synthesize(completed_tasks)

    def _fallback_synthesize(self, completed_tasks: list) -> str:
        parts = []
        for task in completed_tasks:
            header = f"**{task.description}**"
            result_preview = task.result[:800] if len(task.result) > 800 else task.result
            parts.append(f"{header}\n{result_preview}")
        return "\n\n---\n\n".join(parts)
    
    def get_plan(self, plan_id: str) -> Plan | None:
        """Get an active plan by ID."""
        with self._lock:
            return self._active_plans.get(plan_id)
    
    def get_stats(self) -> dict:
        """Get planner statistics."""
        with self._lock:
            return {
                "total_plans": self._stats.total_plans,
                "completed_plans": self._stats.completed_plans,
                "failed_plans": self._stats.failed_plans,
                "avg_tasks_per_plan": round(self._stats.avg_tasks_per_plan, 1),
                "avg_plan_latency_ms": round(self._stats.avg_plan_latency_ms, 0),
                "active_plans": len(self._active_plans),
                "task_templates": len(self._task_templates),
            }


def get_lcm_planner() -> LCMPlanner:
    """Get the global LCM Planner instance."""
    global _planner, _planner_lock
    if _planner is None:
        with _planner_lock:
            if _planner is None:
                _planner = LCMPlanner()
    return _planner