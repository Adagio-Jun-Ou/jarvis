"""Intelligent Agent Orchestrator - Multi-Model Autonomous Coordination.

This module enables each model (Primary/Backend) to act as an autonomous agent,
decomposing complex tasks into subtasks, planning workflows, and coordinating
execution across the dual-model architecture.

Architecture:
┌─────────────────────────────────────────────────────────────────────────────┐
│                     INTELLIGENT AGENT ORCHESTRATOR                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     TASK DECOMPOSITION                               │   │
│  │                                                                       │   │
│  │  User Request → Planner Agent (Backend) → Subtask List              │   │
│  │                    ↓                                                  │   │
│  │              Task Graph (DAG)                                        │   │
│  │                    ↓                                                  │   │
│  │         ┌─────────┼─────────┐                                        │   │
│  │         ↓         ↓         ↓                                        │   │
│  │     Subtask1   Subtask2   Subtask3                                   │   │
│  │         ↓         ↓         ↓                                        │   │
│  │      Agent1     Agent2     Agent3                                     │   │
│  │         │         │         │                                         │   │
│  │         └─────────┴─────────┘                                         │   │
│  │                   ↓                                                    │   │
│  │            Result Synthesis                                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Model Agents:                                                               │
│    PRIMARY Agent (Kimi): Creative, Conversational, User-facing               │
│    BACKEND Agent (GLM-5): Code, Math, Analysis, Reasoning                   │
│    Both can:                                                                │
│      - Decompose complex tasks                                              │
│      - Delegate subtasks to each other                                       │
│      - Execute parallel workflows                                           │
│      - Call tools and reason about results                                   │
│                                                                              │
│  Collaboration Patterns:                                                    │
│    1. DECOMPOSE: Backend breaks down complex request                        │
│    2. DELEGATE: Primary delegates creative tasks to Backend                │
│    3. PARALLEL: Both work on independent subtasks                           │
│    4. SEQUENTIAL: One passes output to the next                             │
│    5. SYNTHESIZE: Combine outputs into coherent response                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
"""
import asyncio
import json
import logging
import threading
import uuid
from core.helpers import safe_error_message
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.dual_model import TaskType

logger = logging.getLogger(__name__)

# Singleton
_intelligent_orchestrator: Optional["IntelligentOrchestrator"] = None
_lock = threading.Lock()


class SubtaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    DELEGATED = "delegated"


class SubtaskType(Enum):
    TOOL_CALL = "tool_call"
    REASONING = "reasoning"
    CODE_GENERATION = "code_generation"
    ANALYSIS = "analysis"
    CREATION = "creation"
    VERIFICATION = "verification"
    SYNTHESIS = "synthesis"


@dataclass
class Subtask:
    """A subtask that can be executed by an agent."""
    id: str
    name: str
    description: str
    subtask_type: SubtaskType
    status: SubtaskStatus = SubtaskStatus.PENDING
    
    # Dependencies
    depends_on: list[str] = field(default_factory=list)
    
    # Execution
    assigned_model: str = "auto"  # "primary", "backend", "auto", "parallel"
    tool_calls: list[tuple[str, dict]] = field(default_factory=list)
    reasoning_prompt: str = ""
    
    # Results
    result: str = ""
    error: str = ""
    metadata: dict = field(default_factory=dict)
    
    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None


@dataclass  
class WorkflowPlan:
    """A plan for executing a complex request."""
    id: str
    original_request: str
    subtasks: list[Subtask] = field(default_factory=list)
    
    # Execution order (topological sort of subtask DAG)
    execution_order: list[str] = field(default_factory=list)
    
    # Status
    status: str = "pending"
    current_subtask_idx: int = 0
    
    # Context from unified context engine
    unified_context: str = ""
    
    # Results
    final_result: str = ""
    synthesis: str = ""
    
    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None


class IntelligentOrchestrator:
    """Orchestrates complex tasks across the dual-model architecture.
    
    Each model can act as an agent:
    - PRIMARY: Good at creative tasks, user communication, high-level planning
    - BACKEND: Good at code, math, analysis, systematic execution
    
    The orchestrator:
    1. Decomposes complex requests into subtasks
    2. Assigns subtasks to appropriate model agents
    3. Coordinates parallel and sequential execution
    4. Synthesizes results from multiple agents
    """
    
    def __init__(self, llm_client, backend_llm, dual_model):
        self.llm = llm_client
        self.backend_llm = backend_llm
        self.dual_model = dual_model
        
        self._active_workflows: dict[str, WorkflowPlan] = {}
        self._max_active_workflows = 20
        self._workflow_history: deque[WorkflowPlan] = deque(maxlen=50)  # Bounded deque
        self._lock = threading.Lock()
        
        # Tool execution - wired from agent
        self._tool_executor: Any | None = None
        self._tool_schemas: list[dict] = []
        
        # Context provider - wired from context_integration
        self._context_provider: Any | None = None
        
        # Track model capabilities
        self.primary_capabilities = ["chat", "creative", "planning", "synthesis", "user_facing"]
        self.backend_capabilities = ["code", "math", "analysis", "reasoning", "verification"]
    
    async def plan_and_execute(self, request: str, context: dict | None = None) -> str:
        """Plan and execute a complex request using dual-model coordination.
        
        Steps:
        1. BACKEND: Decompose request into subtasks (analytical thinking)
        2. Assign subtasks to appropriate agents
        3. Execute subtasks (parallel where possible)
        4. PRIMARY: Synthesize results into coherent response
        """
        workflow_id = f"wf_{uuid.uuid4().hex[:8]}"
        
        # Extract unified context if available
        unified_context = ""
        if context and context.get("unified_context"):
            unified_context = context["unified_context"]
        
        try:
            # Step 1: Decompose request into subtasks using Backend
            decomposition = await self._decompose_request(request, context)
            
            if not decomposition.get("subtasks"):
                # No decomposition needed - simple request
                return await self._execute_simple(request, context)
            
            # Step 2: Create workflow plan
            plan = self._create_workflow_plan(workflow_id, request, decomposition, unified_context)
            with self._lock:
                if len(self._active_workflows) >= self._max_active_workflows:
                    evicted_id = None
                    for wid, wf in self._active_workflows.items():
                        if getattr(wf, 'status', None) not in ('running', 'executing', 'pending'):
                            evicted_id = wid
                            break
                    if evicted_id:
                        evicted = self._active_workflows.pop(evicted_id, None)
                        if evicted:
                            self._workflow_history.append(evicted)
                            logger.info(f"Evicted completed/idle workflow {evicted_id} to stay under limit")
                    else:
                        oldest_id = next(iter(self._active_workflows), None)
                        if oldest_id:
                            evicted = self._active_workflows.pop(oldest_id, None)
                            if evicted:
                                self._workflow_history.append(evicted)
                                logger.warning(f"Evicted running workflow {oldest_id} to stay under limit {self._max_active_workflows}")
                self._active_workflows[workflow_id] = plan
            
            # Step 3: Execute subtasks
            await self._execute_workflow(plan)
            
            # Step 4: Synthesize results
            synthesis = await self._synthesize_results(plan, context)
            
            # Cleanup
            with self._lock:
                completed_plan = self._active_workflows.pop(workflow_id, None)
                if completed_plan:
                    self._workflow_history.append(completed_plan)  # Bounded by deque maxlen
            
            return synthesis
            
        except Exception as e:
            logger.error(f"Workflow {workflow_id} failed: {e}")
            # Fallback to simple execution
            return await self._execute_simple(request, context)
    
    async def _decompose_request(self, request: str, context: dict | None = None) -> dict:
        """Use Backend model to decompose complex request into subtasks."""
        if not self.backend_llm:
            return {"subtasks": []}
        
        # Build context section if available
        context_section = ""
        if context and context.get("unified_context"):
            context_section = f"""
Relevant context from my knowledge:
{context["unified_context"]}

"""
        
        prompt = f"""Analyze this request and break it down into subtasks if it requires multiple steps.
{context_section}Request: "{request}"

Instructions:
1. If this is a simple request (single tool call, straightforward question), return: {{"simple": true}}
2. If it requires multiple steps, break into subtasks:
   {{
     "simple": false,
     "subtasks": [
       {{"name": "descriptive_name", "type": "tool_call|reasoning|code|analysis|creation|verification", "description": "what this subtask does", "model": "primary|backend|auto", "depends_on": []}}
     ],
     "execution_mode": "sequential|parallel|mixed"
   }}

Types: 
- tool_call: Execute a specific tool
- reasoning: Think through a problem (needs model output)
- code: Write or analyze code
- analysis: Analyze data or information  
- creation: Create something new
- verification: Check previous results

Model assignment:
- primary: Creative, conversational, planning
- backend: Code, math, analysis, verification
- auto: System decides based on type

Depends_on: List of subtask names that must complete first (for sequential execution)

Respond with ONLY the JSON, no explanation."""

        try:
            response = await self.backend_llm.chat(
                user_input=prompt,
                system_prompt="You are a task decomposition assistant. Analyze requests and break them into structured subtasks. Respond only with valid JSON.",
            )
            content = response.get("content", "") if response else ""
            
            # Extract JSON
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                return json.loads(json_match.group())
            
        except Exception as e:
            logger.warning(f"Decomposition failed: {e}")
        
        return {"subtasks": []}
    
    def _create_workflow_plan(self, workflow_id: str, request: str, decomposition: dict, unified_context: str = "") -> WorkflowPlan:
        """Create a workflow plan from decomposition."""
        subtasks = []
        
        for i, st in enumerate(decomposition.get("subtasks", [])):
            subtask = Subtask(
                id=f"{workflow_id}_st{i}",
                name=st.get("name", f"subtask_{i}"),
                description=st.get("description", ""),
                subtask_type=SubtaskType(st.get("type", "reasoning")),
                assigned_model=st.get("model", "auto"),
                depends_on=st.get("depends_on", []),
            )
            subtasks.append(subtask)
        
        # Create execution order (topological sort)
        execution_order = self._topological_sort(subtasks)
        
        return WorkflowPlan(
            id=workflow_id,
            original_request=request,
            subtasks=subtasks,
            execution_order=execution_order,
            unified_context=unified_context,
        )
    
    def _topological_sort(self, subtasks: list[Subtask]) -> list[str]:
        """Sort subtasks by dependencies - adds all ready items per iteration."""
        order = []
        remaining = {st.name: st for st in subtasks}
        completed = set()
        
        while remaining:
            batch = []
            for name, st in list(remaining.items()):
                if all(dep in completed for dep in st.depends_on):
                    batch.append(name)
            
            if not batch and remaining:
                order.extend(remaining.keys())
                break
            
            for name in batch:
                order.append(name)
                completed.add(name)
                del remaining[name]
        
        return order
    
    async def _execute_workflow(self, plan: WorkflowPlan) -> None:
        """Execute subtasks in order, respecting dependencies."""
        plan.status = "running"
        
        # Group subtasks by execution wave for parallel execution
        waves = self._compute_execution_waves(plan)
        
        for wave in waves:
            # Execute all subtasks in this wave in parallel
            tasks = []
            for subtask_name in wave:
                subtask = next((st for st in plan.subtasks if st.name == subtask_name), None)
                if subtask:
                    tasks.append(self._execute_subtask(plan, subtask))
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
    
    def _compute_execution_waves(self, plan: WorkflowPlan) -> list[list[str]]:
        """Compute parallel execution waves from dependency DAG."""
        waves = []
        remaining = {st.name: st for st in plan.subtasks}
        completed = set()
        
        while remaining:
            wave = []
            for name, st in list(remaining.items()):
                if all(dep in completed for dep in st.depends_on):
                    wave.append(name)
            
            if not wave:
                # Circular dependency - add remaining
                wave = list(remaining.keys())
            
            waves.append(wave)
            for name in wave:
                completed.add(name)
                del remaining[name]
        
        return waves
    
    async def _execute_subtask(self, plan: WorkflowPlan, subtask: Subtask) -> None:
        """Execute a single subtask."""
        subtask.status = SubtaskStatus.RUNNING
        subtask.started_at = datetime.now()
        
        try:
            # Assign model based on subtask type
            model = subtask.assigned_model
            if model == "auto":
                model = self._select_model_for_subtask(subtask)
            
            if subtask.subtask_type == SubtaskType.TOOL_CALL:
                result = await self._execute_tool_subtask(subtask)
            elif subtask.subtask_type == SubtaskType.REASONING:
                result = await self._execute_reasoning_subtask(subtask, model)
            elif subtask.subtask_type == SubtaskType.CODE_GENERATION:
                result = await self._execute_code_subtask(subtask)
            elif subtask.subtask_type == SubtaskType.ANALYSIS:
                result = await self._execute_analysis_subtask(subtask)
            else:
                result = await self._execute_generic_subtask(subtask, model)
            
            subtask.result = result
            subtask.status = SubtaskStatus.COMPLETED
            subtask.completed_at = datetime.now()
            
        except Exception as e:
            subtask.status = SubtaskStatus.FAILED
            subtask.error = str(e)
            logger.error(f"Subtask {subtask.name} failed: {e}")
    
    def _select_model_for_subtask(self, subtask: Subtask) -> str:
        """Select best model for a subtask type."""
        model_map = {
            SubtaskType.REASONING: "backend",
            SubtaskType.CODE_GENERATION: "backend",
            SubtaskType.ANALYSIS: "backend",
            SubtaskType.VERIFICATION: "backend",
            SubtaskType.CREATION: "primary",
            SubtaskType.TOOL_CALL: "primary",
            SubtaskType.SYNTHESIS: "primary",
        }
        return model_map.get(subtask.subtask_type, "primary")
    
    async def _execute_tool_subtask(self, subtask: Subtask) -> str:
        """Execute a tool-based subtask."""
        if self._tool_executor is None:
            return "Tool execution not available - no executor wired"
        
        if not subtask.tool_calls:
            return f"No tool calls specified for subtask: {subtask.name}"
        
        try:
            results = []
            for tool_name, tool_args in subtask.tool_calls:
                result = await self._tool_executor(tool_name, tool_args)
                results.append(f"**{tool_name}:** {result}")
            return "\n\n".join(results)
        except Exception as e:
            logger.error(f"Tool execution failed for {subtask.name}: {e}")
            return f"Tool execution error: {safe_error_message(e)}"
    
    async def _execute_reasoning_subtask(self, subtask: Subtask, model: str) -> str:
        """Execute a reasoning subtask using specified model."""
        llm = self.backend_llm if model == "backend" else self.llm
        
        prompt = f"""Complete this reasoning task:

Task: {subtask.description}

Think through this systematically and provide your reasoning and conclusion."""
        
        response = await llm.chat(user_input=prompt)
        return response.get("content", "") if response else ""
    
    async def _execute_code_subtask(self, subtask: Subtask) -> str:
        """Execute a code-related subtask."""
        if not self.backend_llm:
            return "Backend model not available for code task"
        
        prompt = f"""Complete this code task:

Task: {subtask.description}

Provide the code solution with explanations."""
        
        response = await self.backend_llm.chat(user_input=prompt)
        return response.get("content", "") if response else ""
    
    async def _execute_analysis_subtask(self, subtask: Subtask) -> str:
        """Execute an analysis subtask."""
        if not self.backend_llm:
            return "Backend model not available for analysis"
        
        prompt = f"""Analyze the following:

Task: {subtask.description}

Provide a thorough analysis with key findings."""
        
        response = await self.backend_llm.chat(user_input=prompt)
        return response.get("content", "") if response else ""
    
    async def _execute_generic_subtask(self, subtask: Subtask, model: str) -> str:
        """Execute a generic subtask."""
        llm = self.backend_llm if model == "backend" else self.llm
        
        prompt = f"""Complete this task:

Task: {subtask.description}

Provide a comprehensive response."""
        
        response = await llm.chat(user_input=prompt)
        return response.get("content", "") if response else ""
    
    async def _synthesize_results(self, plan: WorkflowPlan, context: dict | None = None) -> str:
        """Use Primary model to synthesize all results into coherent response."""
        # Collect all subtask results
        results = []
        for st in plan.subtasks:
            if st.status == SubtaskStatus.COMPLETED and st.result:
                results.append(f"**{st.name}:**\n{st.result}")
        
        if not results:
            return "No results to synthesize."
        
        # Get system prompt from context if available
        system_prompt_base = "You are a helpful assistant."
        if context and context.get("system_prompt"):
            system_prompt_base = context["system_prompt"]
        
        # Build context section if available
        context_section = ""
        if plan.unified_context:
            context_section = f"""
When synthesizing, draw from this relevant context about the user and our conversations:
{plan.unified_context}

"""
        
        # Use Primary for synthesis (creative, coherent output)
        synthesis_prompt = f"""Synthesize these results into a coherent, helpful response for the user.
{context_section}Original Request: {plan.original_request}

Results:
{chr(10).join(results)}

Provide a natural, conversational response that integrates all the results seamlessly. If you referenced any context about the user or past conversations, acknowledge that continuity naturally."""
        
        try:
            response = await self.llm.chat(
                user_input=synthesis_prompt,
                system_prompt=system_prompt_base,
            )
            return response.get("content", "") if response else chr(10).join(results)
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return chr(10).join(results)
    
    async def _execute_simple(self, request: str, context: dict | None = None) -> str:
        """Execute a simple request without decomposition."""
        # Build context section if available
        context_section = ""
        if context and context.get("unified_context"):
            context_section = f"""
Relevant context from our conversations:
{context["unified_context"]}

"""
        
        # Get system prompt from context if available
        system_prompt = context.get("system_prompt", "") if context else ""
        if context_section:
            if system_prompt:
                system_prompt = f"{system_prompt}\n\n{context_section}"
            else:
                system_prompt = f"You are a helpful assistant.\n\n{context_section}"
        
        # Route to appropriate single model
        if self.dual_model:
            task_type = self._classify_simple_task(request)
            return await self.dual_model.route_task(
                task_type=task_type,
                prompt=request,
                system_prompt=system_prompt,
                context=context or {},
            )
        
        # Fallback to primary
        if not system_prompt:
            system_prompt = "You are a helpful assistant."
        
        response = await self.llm.chat(
            user_input=request,
            system_prompt=system_prompt,
        )
        return response.get("content", "") if response else ""
    
    def _classify_simple_task(self, request: str) -> "TaskType":
        """Classify a simple task for routing."""
        from core.dual_model import TaskType
        
        request_lower = request.lower()
        
        # Code-related tasks -> Backend (GLM-5)
        if any(kw in request_lower for kw in ["code", "function", "implement", "debug", "write a script", "program"]):
            return TaskType.CODE
        
        # Math-related tasks -> Backend (GLM-5)
        if any(kw in request_lower for kw in ["calculate", "compute", "math", "solve", "sqrt", "divide", "multiply"]):
            return TaskType.MATH
        
        # Analysis-related tasks -> Backend (GLM-5)
        if any(kw in request_lower for kw in ["analyze", "compare", "evaluate", "assess", "examine", "investigate"]):
            return TaskType.ANALYSIS
        
        # Reasoning tasks -> Backend (GLM-5)
        if any(kw in request_lower for kw in ["deduce", "infer", "conclude", "reason", "logic"]):
            return TaskType.REASONING
        
        # Creative tasks -> Primary (Kimi)
        if any(kw in request_lower for kw in ["write a story", "tell me a story", "create", "compose", "design", "generate ideas"]):
            return TaskType.CREATIVE
        
        # Default to chat -> Primary (Kimi)
        return TaskType.CHAT
    
    def get_active_workflows(self) -> list[dict]:
        """Get status of active workflows."""
        with self._lock:
            return [
                {
                    "id": wf.id,
                    "request": wf.original_request[:100],
                    "status": wf.status,
                    "subtasks": len(wf.subtasks),
                    "completed": sum(1 for st in wf.subtasks if st.status == SubtaskStatus.COMPLETED),
                }
                for wf in self._active_workflows.values()
            ]

    def get_workflow_status(self, workflow_id: str) -> dict | None:
        """Get detailed status of a workflow."""
        with self._lock:
            wf = self._active_workflows.get(workflow_id) or next(
                (w for w in self._workflow_history if w.id == workflow_id), None
            )
            
            if not wf:
                return None
            
            return {
                "id": wf.id,
                "request": wf.original_request,
                "status": wf.status,
                "subtasks": [
                    {
                        "name": st.name,
                        "type": st.subtask_type.value,
                        "status": st.status.value,
                        "model": st.assigned_model,
                        "result": st.result[:200] if st.result else None,
                        "error": st.error,
                    }
                    for st in wf.subtasks
                ],
                "created_at": wf.created_at.isoformat(),
                "completed_at": wf.completed_at.isoformat() if wf.completed_at else None,
            }


def get_intelligent_orchestrator() -> "IntelligentOrchestrator":
    """Get singleton instance of IntelligentOrchestrator.
    
    Returns the initialized instance if available, otherwise None.
    Use initialize_intelligent_orchestrator() to create the instance.
    """
    global _intelligent_orchestrator
    with _lock:
        return _intelligent_orchestrator


def initialize_intelligent_orchestrator(llm_client, backend_llm, dual_model) -> IntelligentOrchestrator:
    """Initialize the intelligent orchestrator with LLM clients."""
    global _intelligent_orchestrator
    with _lock:
        _intelligent_orchestrator = IntelligentOrchestrator(llm_client, backend_llm, dual_model)
    return _intelligent_orchestrator

# ==================== WORKFLOW LEARNING INTEGRATION ====================
# These methods are added to enable embedding-based workflow learning

def _store_workflow_embedding(self, workflow_id: str, request: str, decomposition: dict, success: bool):
    """Store workflow in embedding interconnect for learning."""
    try:
        from core.embedding_interconnect import get_embedding_interconnect
        interconnect = get_embedding_interconnect()
        if hasattr(interconnect, 'store_workflow'):
            interconnect.store_workflow(workflow_id, request, decomposition, success)
            logger.debug(f"Stored workflow embedding: {workflow_id}")
    except Exception as e:
        logger.warning(f"Failed to store workflow embedding: {e}")


def _retrieve_similar_workflows(self, request: str, top_k: int = 3) -> list:
    """Retrieve similar past workflows from embeddings."""
    try:
        from core.embedding_interconnect import get_embedding_interconnect
        interconnect = get_embedding_interconnect()
        if hasattr(interconnect, 'get_workflow_patterns'):
            return interconnect.get_workflow_patterns(request, top_k=top_k)
    except Exception as e:
        logger.warning(f"Failed to retrieve similar workflows: {e}")
    return []


def _get_workflow_embedding_interconnect(self):
    """Lazy load embedding interconnect for workflow learning."""
    try:
        from core.embedding_interconnect import get_embedding_interconnect
        return get_embedding_interconnect()
    except Exception as e:
        logger.warning(f"Embedding interconnect not available: {e}")
        return None


# Add these methods to the IntelligentOrchestrator class
IntelligentOrchestrator._store_workflow_embedding = _store_workflow_embedding
IntelligentOrchestrator._retrieve_similar_workflows = _retrieve_similar_workflows
IntelligentOrchestrator._get_workflow_embedding_interconnect = _get_workflow_embedding_interconnect
