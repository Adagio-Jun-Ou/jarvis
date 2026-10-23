"""Multi-Model Agentic Integration - Ties agentic work to dual-model architecture.

This module provides the integration between agentic project work and Jarvis's
dual-model system, ensuring proper model routing for different task types.

Architecture Integration:
┌─────────────────────────────────────────────────────────────────┐
│                    AGENTIC MULTI-MODEL                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  MODEL-AWARE TASK ROUTING                 │   │
│  │                                                            │   │
│  │  Task Type         → Model Choice                         │   │
│  │  ────────────────────────────────────────                 │   │
│  │  Decomposition      → Backend (code/math/analysis)       │   │
│  │  Plan Execution     → Backend (reasoning/verification)   │   │
│  │  Code Generation    → Backend (GLM-5 thinking)           │   │
│  │  Code Review        → Backend (analysis)                  │   │
│  │  Creative Writing   → Primary (creative/chat)             │   │
│  │  Documentation      → Primary (creative/explanation)      │   │
│  │  User Communication  → Primary (chat)                     │   │
│  │  Vision/Image       → Primary (vision)                    │   │
│  │                                                                 │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    DUAL MODEL AGENTIC                     │   │
│  │                                                            │   │
│  │  project_decompose:  Backend (analysis)                   │   │
│  │  project_execute:    Routed by task type                  │   │
│  │  project_file_write: Backend (code)                       │   │
│  │  project_test:       Backend (analysis)                   │   │
│  │  project_commit:     Backend (analysis)                   │   │
│  │                                                                 │
│  │  Workflow Stages:                                          │   │
│  │  ErdosProblemStage    → Backend (math)                    │   │
│  │  MusicCompositionStage → Primary (creative) + Backend    │   │
│  │  DaydreamExploration  → Primary (explore) + Backend      │   │
│  │  ToolPracticeStage    → Backend (analysis)                │   │
│  │  VerificationStage    → Backend (verification)            │   │
│  │                                                                 │
│  └─────────────────────────────────────────────────────────────┘
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
"""
import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from core.agentic_context import (
    AgenticContext, StageExecutionResult,
)
from core.task_decomposition import TaskDecomposition
from core.execution_planner import ExecutionPlanner

logger = logging.getLogger(__name__)

# Type alias for dual model types
try:
    from core.dual_model import TaskType, ModelRole, get_dual_model_orchestrator
    HAS_DUAL_MODEL = True
except ImportError:
    HAS_DUAL_MODEL = False
    TaskType = None
    ModelRole = None
    get_dual_model_orchestrator = None


# Task type to model routing for agentic work
AGENTIC_TASK_ROUTING = {
    # Project tasks
    "project_init": "primary",      # Status overview - primary
    "project_status": "primary",    # User-facing status
    "project_decompose": "backend", # Goal analysis - needs reasoning
    "project_execute": "backend",    # Execution planning
    "project_file_read": "primary", # Simple file read
    "project_file_write": "backend", # Code generation
    "project_test": "backend",       # Test analysis
    "project_commit": "backend",     # Analysis for commit
    
    # Decomposition tasks
    "decompose_goal": "backend",
    "analyze_impact": "backend",
    "suggest_next": "primary",
    
    # Execution tasks
    "create_file": "backend",
    "modify_file": "backend",
    "run_tests": "backend",
    "run_command": "backend",
    "review_code": "backend",
    "git_commit": "backend",
    
    # Subsystem tasks
    "erdos_problem": "backend",      # Math
    "music_composition": "primary",   # Creative (with backend for synthesis)
    "daydream_exploration": "primary",
    "insight_processing": "backend",
    "knowledge_graph": "backend",
    "tool_practice": "backend",
    "verification": "backend",
    
    # Creative/communication
    "documentation": "primary",
    "user_communication": "primary",
    "creative_writing": "primary",
}


@dataclass
class ModelChoice:
    """Result of model selection for a task."""
    model: str  # "primary", "backend", "parallel"
    task_type: Optional[str] = None
    reason: str = ""
    confidence: float = 1.0


class MultiModelAgentic:
    """Integrates agentic work with the dual-model architecture.
    
    Routes tasks to the appropriate model based on:
    1. Task type (code/math → Backend, creative/chat → Primary)
    2. Semantic similarity to past successful outputs
    3. Current model availability and load
    
    Provides:
    - Model-aware task decomposition
    - Routed execution planning
    - Dual-model code review
    - Parallel validation
    """
    
    def __init__(
        self,
        primary_llm=None,
        backend_llm=None,
        dual_model=None,
    ):
        self.primary_llm = primary_llm
        self.backend_llm = backend_llm
        self.dual_model = dual_model
        
        self._decomposer = None
        self._planner = None
        self._lock = threading.RLock()
        
        if backend_llm:
            self._decomposer = TaskDecomposition(llm_client=backend_llm)
            self._planner = ExecutionPlanner(llm_client=backend_llm)
    
    def choose_model_for_task(
        self,
        task_type: str,
        context: AgenticContext | None = None,
        prompt: str | None = None,
    ) -> ModelChoice:
        """Choose the best model for a task.
        
        Args:
            task_type: Type of task being performed
            context: Agentic context (optional)
            prompt: Task prompt for semantic routing (optional)
        
        Returns:
            ModelChoice with model selection and reasoning
        """
        routing = AGENTIC_TASK_ROUTING.get(task_type, "primary")
        
        if HAS_DUAL_MODEL and self.dual_model and prompt:
            try:
                from core.dual_model import TaskType as DMTaskType
                
                task_type_map = {
                    "decompose_goal": DMTaskType.ANALYSIS,
                    "analyze_impact": DMTaskType.ANALYSIS,
                    "suggest_next": DMTaskType.CHAT,
                    "create_file": DMTaskType.CODE,
                    "modify_file": DMTaskType.CODE,
                    "run_tests": DMTaskType.VERIFICATION,
                    "review_code": DMTaskType.ANALYSIS,
                    "erdos_problem": DMTaskType.MATH,
                    "music_composition": DMTaskType.CREATIVE,
                    "documentation": DMTaskType.CREATIVE,
                }
                
                dm_task_type = task_type_map.get(task_type)
                if dm_task_type:
                    semantic_model, confidence = self.dual_model._semantic_route(prompt, dm_task_type)
                    if confidence > 0.7:
                        routing = semantic_model.lower()
                        return ModelChoice(
                            model=routing,
                            task_type=task_type,
                            reason=f"Semantic routing (confidence: {confidence:.2f})",
                            confidence=confidence,
                        )
            except Exception as e:
                logger.debug(f"Semantic routing failed: {e}")
        
        reason_map = {
            "primary": "User-facing/creative task",
            "backend": "Code/math/analysis task",
            "parallel": "Task benefits from multiple perspectives",
        }
        
        return ModelChoice(
            model=routing,
            task_type=task_type,
            reason=reason_map.get(routing, f"Task type requires {routing}"),
            confidence=0.8,
        )
    
    def get_llm_for_task(self, model_choice: ModelChoice):
        """Get the appropriate LLM client for a model choice."""
        if model_choice.model == "backend":
            return self.backend_llm
        return self.primary_llm
    
    async def decompose_goal(
        self,
        goal: str,
        context: AgenticContext,
        use_backend: bool = True,
    ) -> dict:
        """Decompose a goal using the appropriate model.
        
        Always uses Backend (GLM-5) for decomposition since it requires
        rigorous analysis and reasoning.
        
        Args:
            goal: High-level goal to decompose
            context: Agentic context
            use_backend: Force backend model (default True)
        
        Returns:
            Decomposition result with tasks
        """
        self.choose_model_for_task("decompose_goal", context, goal)
        llm = self.backend_llm if use_backend else self.primary_llm
        
        if not llm:
            return {"success": False, "error": "No LLM available for decomposition"}
        
        if not self._decomposer:
            self._decomposer = TaskDecomposition(llm_client=llm)
        
        from core.task_decomposition import DecompositionContext
        
        decomp_ctx = DecompositionContext(
            project_name=context.project.project_name,
            project_path=context.project.project_path,
            existing_epics=[],  # Would populate from workspace
            architecture_overview="\n".join(
                f"{k}: {v['description']}" 
                for k, v in list(context.project.architecture.items())[:20]
            ),
            relevant_files=list(context.project.architecture.keys())[:30],
        )
        
        result = await self._decomposer.decompose(goal, decomp_ctx, depth=3)
        
        if result.success:
            result_dict = {
                "success": True,
                "epic": result.epic.title if result.epic else None,
                "stories": len(result.epic.stories) if result.epic else 0,
                "tasks": sum(len(s.tasks) for s in result.epic.stories) if result.epic else 0,
                "total_minutes": result.total_estimated_minutes,
                "risks": result.risks,
                "recommendations": result.recommendations,
                "model_used": "backend" if use_backend else "primary",
            }
            
            if context.goal:
                context.goal.subtasks = [
                    t.title
                    for s in result.epic.stories
                    for t in s.tasks
                ] if result.epic else []
            
            return result_dict
        
        return {
            "success": False,
            "error": result.analysis,
            "model_used": "backend" if use_backend else "primary",
        }
    
    async def execute_code_task(
        self,
        task: str,
        context: AgenticContext,
        code_type: str = "python",
    ) -> StageExecutionResult:
        """Execute a code-related task using Backend model.
        
        Backend (GLM-5) is used for all code tasks because:
        - Better at structured reasoning
        - Thinking mode for complex problems
        - Trained for code generation
        
        Args:
            task: Task description
            context: Agentic context
            code_type: Programming language
        
        Returns:
            Execution result
        """
        self.choose_model_for_task("modify_file", context, task)
        
        if not self.backend_llm:
            return StageExecutionResult(
                status="failed",
                error=RuntimeError("Backend LLM not available for code tasks"),
            )
        
        start_time = datetime.now()
        
        try:
            prompt = f"""Task: {task}

Project Context:
- Language: {context.project.language}
- Frameworks: {', '.join(context.project.frameworks)}
- Entry points: {', '.join(context.project.entry_points[:5])}

Recent changes:
{context.shared_state.get('recent_changes', 'None')}

Generate code to complete this task. Include file paths in the output."""

            response = await self.backend_llm.chat(
                user_input=prompt,
                system_prompt=f"You are an expert {code_type} developer. Generate clean, well-documented code.",
            )
            
            from core.helpers import get_response_content
            content = get_response_content(response)
            
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            return StageExecutionResult(
                value={"code": content, "model": "backend"},
                status="completed",
                duration_ms=duration_ms,
                metadata={"task": task, "code_type": code_type},
            )
            
        except Exception as e:
            return StageExecutionResult(
                status="failed",
                error=e,
                duration_ms=(datetime.now() - start_time).total_seconds() * 1000,
            )
    
    async def review_code(
        self,
        code: str,
        context: AgenticContext,
        focus: str = "correctness",
    ) -> StageExecutionResult:
        """Review code using Backend model.
        
        Args:
            code: Code to review
            context: Agentic context
            focus: Review focus (correctness, security, performance)
        
        Returns:
            Review result
        """
        self.choose_model_for_task("review_code", context, code[:500])
        
        if not self.backend_llm:
            return StageExecutionResult(
                status="failed",
                error=RuntimeError("Backend LLM not available for code review"),
            )
        
        start_time = datetime.now()
        
        try:
            prompt = f"""Review this code for {focus}:

```
{code}
```

Project context: {context.project.language}

Provide:
1. ISSUES: List any problems found
2. SUGGESTIONS: Improvement recommendations
3. APPROVED: yes/no"""

            response = await self.backend_llm.chat(
                user_input=prompt,
                system_prompt="You are a senior code reviewer. Be thorough but practical.",
            )
            
            from core.helpers import get_response_content
            content = get_response_content(response)
            
            approved = "APPROVED: yes" in content or "approved: yes" in content.lower()
            
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            return StageExecutionResult(
                value={
                    "approved": approved,
                    "review": content,
                    "focus": focus,
                    "model": "backend",
                },
                status="completed",
                duration_ms=duration_ms,
            )
            
        except Exception as e:
            return StageExecutionResult(
                status="failed",
                error=e,
                duration_ms=(datetime.now() - start_time).total_seconds() * 1000,
            )
    
    async def parallel_validate(
        self,
        content: str,
        context: AgenticContext,
        validation_type: str = "general",
    ) -> dict:
        """Get validation from both models in parallel.
        
        Useful for:
        - Cross-checking analysis
        - Multiple perspectives on a problem
        - Conflict detection
        
        Args:
            content: Content to validate
            context: Agentic context
            validation_type: Type of validation
        
        Returns:
            Dict with primary and backend perspectives
        """
        if not HAS_DUAL_MODEL or not self.dual_model:
            if self.backend_llm:
                result = await self.backend_llm.chat(
                    user_input=f"Validate this {validation_type}:\n{content}",
                    system_prompt="You are a validation engine. Check for errors and issues.",
                )
                from core.helpers import get_response_content, safe_error_message
                return {"backend": get_response_content(result), "primary": None}
            return {"error": "No models available for validation"}
        
        try:
            from core.dual_model import TaskType
            
            results = await self.dual_model.parallel_execute(
                primary_task=(TaskType.CREATIVE, f"Review this from a user perspective:\n{content[:2000]}"),
                backend_task=(TaskType.VERIFICATION, f"Validate this {validation_type} for correctness:\n{content[:2000]}"),
            )
            
            return {
                "primary": results.get("primary"),
                "backend": results.get("backend"),
                "validation_type": validation_type,
            }
            
        except Exception as e:
            logger.error(f"Parallel validation failed: {e}")
            return {"error": safe_error_message(e)}
    
    def get_stats(self) -> dict:
        """Get statistics about model usage for agentic tasks."""
        return {
            "backend_available": self.backend_llm is not None,
            "primary_available": self.primary_llm is not None,
            "dual_model_available": self.dual_model is not None,
            "decomposer_initialized": self._decomposer is not None,
            "planner_initialized": self._planner is not None,
        }


# Singleton
_multi_model_agentic: MultiModelAgentic | None = None
_multi_model_lock = threading.Lock()


def get_multi_model_agentic(
    primary_llm=None,
    backend_llm=None,
    dual_model=None,
) -> MultiModelAgentic:
    """Get or create multi-model agentic integration."""
    global _multi_model_agentic
    
    with _multi_model_lock:
        if _multi_model_agentic is None:
            _multi_model_agentic = MultiModelAgentic(
                primary_llm=primary_llm,
                backend_llm=backend_llm,
                dual_model=dual_model,
            )
        else:
            if primary_llm is not None:
                _multi_model_agentic.primary_llm = primary_llm
            if backend_llm is not None:
                _multi_model_agentic.backend_llm = backend_llm
            if dual_model is not None:
                _multi_model_agentic.dual_model = dual_model
    
    return _multi_model_agentic


def init_multi_model_agentic(
    primary_llm,
    backend_llm,
    dual_model=None,
) -> MultiModelAgentic:
    """Initialize multi-model agentic with all models."""
    global _multi_model_agentic
    
    _multi_model_agentic = MultiModelAgentic(
        primary_llm=primary_llm,
        backend_llm=backend_llm,
        dual_model=dual_model,
    )
    
    logger.info("Multi-model agentic integration initialized")
    logger.info(f"  Primary: {'available' if primary_llm else 'not available'}")
    logger.info(f"  Backend: {'available' if backend_llm else 'not available'}")
    logger.info(f"  Dual model: {'available' if dual_model else 'not available'}")
    
    return _multi_model_agentic