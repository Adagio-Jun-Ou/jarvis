"""Dual Model Integration for Core Subsystems.

This module provides the integration layer between the dual-model architecture
and Jarvis's core intelligence subsystems:

┌─────────────────────────────────────────────────────────────────────────────┐
│                        DUAL MODEL SUBSYSTEM INTEGRATION                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     SUBSYSTEM MODEL ROUTING                          │   │
│  │                                                                       │   │
│  │  Daydream Engine:                                                     │   │
│  │    • Creative exploration → PRIMARY (generates ideas)                │   │
│  │    • Analysis → BACKEND (evaluates ideas)                            │   │
│  │    • Mode: SEQUENTIAL (create, then evaluate)                        │   │
│  │                                                                       │   │
│  │  Erdős Engine:                                                        │   │
│  │    • Problem exploration → BACKEND (math reasoning)                  │   │
│  │    • Insight generation → PARALLEL (both generate insights)          │   │
│  │    • Proof verification → BACKEND (rigorous checking)                │   │
│  │                                                                       │   │
│  │  Evolution Orchestrator:                                              │   │
│  │    • Proposal generation → PRIMARY (creative ideas)                  │   │
│  │    • Impact analysis → BACKEND (system analysis)                     │   │
│  │    • Mode: SEQUENTIAL (propose, then analyze)                        │   │
│  │                                                                       │   │
│  │  Self-Refine:                                                         │   │
│  │    • Code analysis → BACKEND (static analysis)                       │   │
│  │    • Fix generation → PARALLEL (both suggest fixes)                  │   │
│  │    • Impact evaluation → BACKEND (testing/reasoning)                 │   │
│  │                                                                       │   │
│  │  Insight Bridge:                                                      │   │
│  │    • Pattern extraction → PARALLEL (both find patterns)              │   │
│  │    • Evolution guidance → BACKEND (analytical mapping)               │   │
│  │    • Knowledge generation → PRIMARY (human-readable output)          │   │
│  │                                                                       │   │
│  │  Learning Pipeline:                                                   │   │
│  │    • Pattern discovery → PARALLEL (both analyze)                     │   │
│  │    • Strategy evaluation → BACKEND (analytical comparison)           │   │
│  │    • Knowledge storage → BACKEND (structured storage)                │   │
│  │                                                                       │   │
│  │  Proactive Engine:                                                    │   │
│  │    • Curiosity generation → PRIMARY (creative curiosity)             │   │
│  │    • Analysis tasks → BACKEND (reasoning tasks)                      │   │
│  │    • Self-reflection → PARALLEL (both perspectives)                  │   │
│  │                                                                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Shared Context: Both models share a unified context via ModelCoordinator   │
│  Cross-Pollination: Insights from one model enrich the other               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
"""
import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from core.helpers import get_response_content

logger = logging.getLogger(__name__)

# Singleton
_dual_subsystem_integration: Optional["DualSubsystemIntegration"] = None
_lock = threading.Lock()


class SubsystemTask(Enum):
    """Types of tasks each subsystem performs."""
    # Daydream
    CREATIVE_EXPLORATION = "creative_exploration"
    IDEA_EVALUATION = "idea_evaluation"
    
    # Erdős
    MATH_REASONING = "math_reasoning"
    INSIGHT_GENERATION = "insight_generation"
    PROOF_VERIFICATION = "proof_verification"
    
    # Evolution
    PROPOSAL_GENERATION = "proposal_generation"
    IMPACT_ANALYSIS = "impact_analysis"
    
    # Self-Refine
    CODE_ANALYSIS = "code_analysis"
    FIX_GENERATION = "fix_generation"
    IMPACT_EVALUATION = "impact_evaluation"
    
    # Insight Bridge
    PATTERN_EXTRACTION = "pattern_extraction"
    EVOLUTION_GUIDANCE = "evolution_guidance"
    KNOWLEDGE_GENERATION = "knowledge_generation"
    
    # Learning
    PATTERN_DISCOVERY = "pattern_discovery"
    STRATEGY_EVALUATION = "strategy_evaluation"
    
    # Proactive
    CURIOSITY_GENERATION = "curiosity_generation"
    SELF_REFLECTION = "self_reflection"


# Human-readable descriptions for each task type
SUBSYSTEM_TASK_DESCRIPTIONS = {
    # Daydream
    SubsystemTask.CREATIVE_EXPLORATION: "Exploring creative ideas and possibilities",
    SubsystemTask.IDEA_EVALUATION: "Evaluating and refining generated ideas",
    
    # Erdős
    SubsystemTask.MATH_REASONING: "Solving mathematical problems and proofs",
    SubsystemTask.INSIGHT_GENERATION: "Generating insights from reasoning",
    SubsystemTask.PROOF_VERIFICATION: "Verifying mathematical proofs",
    
    # Evolution
    SubsystemTask.PROPOSAL_GENERATION: "Generating improvement proposals",
    SubsystemTask.IMPACT_ANALYSIS: "Analyzing impact of proposed changes",
    
    # Self-Refine
    SubsystemTask.CODE_ANALYSIS: "Analyzing code for improvements",
    SubsystemTask.FIX_GENERATION: "Generating code fixes and patches",
    SubsystemTask.IMPACT_EVALUATION: "Evaluating fix impact and safety",
    
    # Insight Bridge
    SubsystemTask.PATTERN_EXTRACTION: "Extracting patterns from data",
    SubsystemTask.EVOLUTION_GUIDANCE: "Guiding evolutionary improvements",
    SubsystemTask.KNOWLEDGE_GENERATION: "Generating structured knowledge",
    
    # Learning
    SubsystemTask.PATTERN_DISCOVERY: "Discovering new patterns and strategies",
    SubsystemTask.STRATEGY_EVALUATION: "Evaluating learning strategies",
    
    # Proactive
    SubsystemTask.CURIOSITY_GENERATION: "Generating curious questions",
    SubsystemTask.SELF_REFLECTION: "Reflecting on system state",
}

# Subsystem display names and descriptions
SUBSYSTEM_INFO = {
    "daydream": {"name": "Daydream Engine", "emoji": "💭", "role": "Idle-time exploration and creative synthesis"},
    "erdos": {"name": "Erdős Engine", "emoji": "🔢", "role": "Mathematical problem solving"},
    "evolution": {"name": "Evolution", "emoji": "🧬", "role": "Self-improvement and adaptation"},
    "self_refine": {"name": "Self-Refine", "emoji": "🔧", "role": "Code analysis and refinement"},
    "insight_bridge": {"name": "Insight Bridge", "emoji": "🌉", "role": "Pattern extraction and knowledge synthesis"},
    "learning": {"name": "Learning Pipeline", "emoji": "📚", "role": "Continuous learning and strategy discovery"},
    "proactive": {"name": "Proactive Engine", "emoji": "🎯", "role": "Anticipatory intelligence and curiosity"},
}


# Task routing: which model handles which task type
SUBSYSTEM_TASK_ROUTING = {
    # Daydream: Primary creates, Backend evaluates (sequential)
    SubsystemTask.CREATIVE_EXPLORATION: "primary",
    SubsystemTask.IDEA_EVALUATION: "backend",
    
    # Erdős: Backend dominates (math), but both generate insights (parallel)
    SubsystemTask.MATH_REASONING: "backend",
    SubsystemTask.INSIGHT_GENERATION: "parallel",  # Both contribute
    SubsystemTask.PROOF_VERIFICATION: "backend",
    
    # Evolution: Primary proposes, Backend analyzes (sequential)
    SubsystemTask.PROPOSAL_GENERATION: "primary",
    SubsystemTask.IMPACT_ANALYSIS: "backend",
    
    # Self-Refine: Backend dominates, both suggest fixes (parallel)
    SubsystemTask.CODE_ANALYSIS: "backend",
    SubsystemTask.FIX_GENERATION: "parallel",  # Both contribute
    SubsystemTask.IMPACT_EVALUATION: "backend",
    
    # Insight Bridge: Both extract patterns (parallel), Backend maps, Primary writes
    SubsystemTask.PATTERN_EXTRACTION: "parallel",
    SubsystemTask.EVOLUTION_GUIDANCE: "backend",
    SubsystemTask.KNOWLEDGE_GENERATION: "primary",
    
    # Learning: Both discover (parallel), Backend evaluates
    SubsystemTask.PATTERN_DISCOVERY: "parallel",
    SubsystemTask.STRATEGY_EVALUATION: "backend",
    
    # Proactive: Primary curious, Backend reflective (parallel)
    SubsystemTask.CURIOSITY_GENERATION: "primary",
    SubsystemTask.SELF_REFLECTION: "parallel",  # Both perspectives
}

# Sequential task pairs: when these tasks are called in sequence, mark as collaborative
SEQUENTIAL_PAIRS = {
    # daydream_explore: creative then evaluate
    (SubsystemTask.CREATIVE_EXPLORATION, SubsystemTask.IDEA_EVALUATION): "sequential",
    # evolution: propose then analyze  
    (SubsystemTask.PROPOSAL_GENERATION, SubsystemTask.IMPACT_ANALYSIS): "sequential",
    # self_refine: analyze then fix
    (SubsystemTask.CODE_ANALYSIS, SubsystemTask.FIX_GENERATION): "sequential",
    # insight: extract then guide
    (SubsystemTask.PATTERN_EXTRACTION, SubsystemTask.EVOLUTION_GUIDANCE): "sequential",
    # learning: discover then evaluate
    (SubsystemTask.PATTERN_DISCOVERY, SubsystemTask.STRATEGY_EVALUATION): "sequential",
}


@dataclass
class SubsystemCall:
    """A call from a subsystem to the dual-model system."""
    subsystem: str
    task: SubsystemTask
    prompt: str
    context: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Results
    primary_result: Optional[str] = None
    backend_result: Optional[str] = None
    synthesis: Optional[str] = None
    model_used: Optional[str] = None
    latency_ms: float = 0.0


@dataclass
class SubsystemStats:
    """Statistics per subsystem."""
    total_calls: int = 0
    primary_calls: int = 0
    backend_calls: int = 0
    parallel_calls: int = 0
    sequential_calls: int = 0
    total_latency_ms: float = 0.0
    errors: int = 0


class DualSubsystemIntegration:
    """Integrates dual-model architecture with all core subsystems.
    
    This class provides:
    1. Unified API for subsystems to use dual-model capabilities
    2. Task routing based on subsystem task types
    3. Shared context between models
    4. Statistics tracking per subsystem
    5. Reports to the global orchestrator for unified stats
    """
    
    def __init__(
        self,
        primary_llm=None,
        backend_llm=None,
        model_coordinator=None,
    ):
        self.primary_llm = primary_llm
        self.backend_llm = backend_llm
        self.model_coordinator = model_coordinator
        
        # Per-subsystem statistics
        self._stats: dict[str, SubsystemStats] = {}
        self._call_history: list[SubsystemCall] = []
        self._lock = threading.RLock()
        
        # Shared context between models for each subsystem
        self._subsystem_context: dict[str, dict] = {}
        
        # Initialized subsystems
        self._subsystems: dict[str, Any] = {}
        
        # Track last task per subsystem for sequential detection
        self._last_task: dict[str, SubsystemTask] = {}
        
    def register_subsystem(
        self,
        name: str,
        subsystem: Any,
        primary_tasks: list[SubsystemTask],
        backend_tasks: list[SubsystemTask],
    ) -> None:
        """Register a subsystem with its task routing."""
        self._subsystems[name] = {
            "instance": subsystem,
            "primary_tasks": primary_tasks,
            "backend_tasks": backend_tasks,
        }
        self._stats[name] = SubsystemStats()
        self._subsystem_context[name] = {}
        logger.info(f"Registered subsystem '{name}' with dual-model integration")
    
    def get_model_for_task(self, task: SubsystemTask) -> str:
        """Get which model should handle a task: 'primary', 'backend', or 'parallel'."""
        return SUBSYSTEM_TASK_ROUTING.get(task, "backend")
    
    async def call(
        self,
        subsystem: str,
        task: SubsystemTask,
        prompt: str,
        context: dict | None = None,
        system_prompt: str = "",
    ) -> str:
        """Make a dual-model call from a subsystem.
        
        Automatically routes to the appropriate model(s) based on task type.
        
        Args:
            subsystem: Name of the calling subsystem
            task: Type of task being performed
            prompt: The input prompt
            context: Additional context
            system_prompt: System instruction
            
        Returns:
            Response from the model(s)
        """
        call = SubsystemCall(
            subsystem=subsystem,
            task=task,
            prompt=prompt,
            context=context or {},
        )
        
        model_routing = self.get_model_for_task(task)
        start_time = time.time()
        
        try:
            if model_routing == "parallel":
                result = await self._execute_parallel(call, system_prompt)
            elif model_routing == "primary":
                result = await self._execute_primary(call, system_prompt)
            else:
                result = await self._execute_backend(call, system_prompt)
            
            call.synthesis = result
            latency = (time.time() - start_time) * 1000
            with self._lock:
                stats = self._stats.get(subsystem, SubsystemStats())
                stats.total_calls += 1
                stats.total_latency_ms += latency
                # Increment the appropriate call type counter
                if model_routing == "primary":
                    stats.primary_calls += 1
                elif model_routing == "backend":
                    stats.backend_calls += 1
                else:
                    stats.parallel_calls += 1
                
                # Auto-detect sequential collaboration patterns
                last_task = self._last_task.get(subsystem)
                if last_task:
                    pair_key = (last_task, task)
                    if pair_key in SEQUENTIAL_PAIRS:
                        stats.sequential_calls += 1
                        # Report to global orchestrator
                        self._report_sequential_to_orchestrator(call.latency_ms)
                        logger.info(f"Sequential collaboration DETECTED: {subsystem} {last_task.value} -> {task.value}")
                    else:
                        logger.debug(f"No sequential match: {last_task.value} -> {task.value} (checking {len(SEQUENTIAL_PAIRS)} pairs)")
                
                self._last_task[subsystem] = task
                
                self._stats[subsystem] = stats
                self._call_history.append(call)
                if len(self._call_history) > 500:
                    self._call_history = self._call_history[-500:]
                
                # Report to global orchestrator for unified stats
                self._report_to_orchestrator(model_routing, call.latency_ms)
            
            # Log activity for dashboard
            self._log_subsystem_activity(subsystem, task, model_routing, latency, result[:100] if result else "")
            
            return result
            
        except Exception as e:
            logger.error(f"Dual-subsystem call failed: {subsystem}/{task.value}: {e}")
            with self._lock:
                stats = self._stats.get(subsystem, SubsystemStats())
                stats.errors += 1
                self._stats[subsystem] = stats
            raise
    
    async def _execute_primary(self, call: SubsystemCall, system_prompt: str) -> str:
        """Execute on primary model."""
        if not self.primary_llm:
            raise RuntimeError("Primary LLM not configured")
        
        start = time.time()
        response = await self.primary_llm.chat(
            user_input=call.prompt,
            system_prompt=system_prompt or "You are a creative AI assistant.",
            conversation_history=[],
        )
        
        call.primary_result = get_response_content(response) if isinstance(response, dict) else str(response)
        call.model_used = "primary"
        call.latency_ms = (time.time() - start) * 1000
        
        # Store embedding for semantic routing
        self._store_embedding("primary", call.prompt, call.primary_result)
        
        # Share context with model coordinator
        if self.model_coordinator:
            self.model_coordinator.set_shared_context(
                f"{call.subsystem}_last_primary",
                {"result": call.primary_result[:500], "task": call.task.value},
                "primary"
            )
        
        return call.primary_result
    
    async def _execute_backend(self, call: SubsystemCall, system_prompt: str) -> str:
        """Execute on backend model."""
        if not self.backend_llm:
            raise RuntimeError("Backend LLM not configured")
        
        start = time.time()
        response = await self.backend_llm.chat(
            user_input=call.prompt,
            system_prompt=system_prompt or "You are an analytical AI assistant focused on precise reasoning.",
            conversation_history=[],
        )
        
        call.backend_result = get_response_content(response) if isinstance(response, dict) else str(response)
        
        # Extract thinking if available (GLM-5 specific)
        thinking = response.get("thinking", "") if isinstance(response, dict) else ""
        if thinking:
            call.backend_result = f"<thinking>\n{thinking}\n</thinking>\n{call.backend_result}"
        
        call.model_used = "backend"
        call.latency_ms = (time.time() - start) * 1000
        
        # Store embedding for semantic routing
        self._store_embedding("backend", call.prompt, call.backend_result)
        
        # Share context with model coordinator
        if self.model_coordinator:
            self.model_coordinator.set_shared_context(
                f"{call.subsystem}_last_backend",
                {"result": call.backend_result[:500], "task": call.task.value},
                "backend"
            )
        
        return call.backend_result
    
    async def _execute_parallel(self, call: SubsystemCall, system_prompt: str) -> str:
        """Execute on both models in parallel."""
        start_time = time.time()
        primary_task = self._execute_primary(call, system_prompt)
        backend_task = self._execute_backend(call, system_prompt)
        
        results = await asyncio.gather(primary_task, backend_task, return_exceptions=True)
        
        call.primary_result = results[0] if not isinstance(results[0], Exception) else None
        call.backend_result = results[1] if not isinstance(results[1], Exception) else None
        call.model_used = "parallel"
        call.latency_ms = (time.time() - start_time) * 1000
        
        # Synthesize results
        if call.primary_result and call.backend_result:
            synthesis = await self._synthesize(call.primary_result, call.backend_result, call.task)
            # Store synthesis embedding
            self._store_embedding("synthesis", call.prompt, synthesis)
            return synthesis
        elif call.primary_result:
            return call.primary_result
        elif call.backend_result:
            return call.backend_result
        else:
            raise RuntimeError("Both models failed in parallel execution")
    
    async def _synthesize(self, primary_result: str, backend_result: str, task: SubsystemTask) -> str:
        """Synthesize results from both models into a unified response.
        
        The user doesn't care which model produced the output - they want
        a coherent, helpful response. We intelligently merge or select the best
        output based on task type and content quality.
        """
        if not primary_result and not backend_result:
            return ""
        if not primary_result:
            return backend_result
        if not backend_result:
            return primary_result
        
        insight_tasks = [SubsystemTask.INSIGHT_GENERATION, SubsystemTask.PATTERN_EXTRACTION, SubsystemTask.SELF_REFLECTION]
        combined_tasks = [SubsystemTask.FIX_GENERATION, SubsystemTask.PATTERN_DISCOVERY]
        
        if task in insight_tasks:
            primary_lines = len(primary_result.split('\n'))
            backend_lines = len(backend_result.split('\n'))
            if backend_lines > primary_lines * 2:
                return backend_result
            if primary_lines > backend_lines * 2:
                return primary_result
            combined = primary_result.rstrip()
            if not combined.endswith('.'):
                combined += '.'
            return f"{combined}\n\n{backend_result}"
        elif task in combined_tasks:
            combined = []
            seen = set()
            for line in primary_result.split('\n'):
                line = line.strip()
                if line and line not in seen:
                    combined.append(line)
                    seen.add(line)
            for line in backend_result.split('\n'):
                line = line.strip()
                if line and line not in seen:
                    combined.append(line)
                    seen.add(line)
            return '\n'.join(combined[:50])
        else:
            if len(backend_result) > len(primary_result) * 1.3:
                return backend_result
            if len(primary_result) > len(backend_result) * 1.3:
                return primary_result
            return backend_result

    def _store_embedding(self, model: str, prompt: str, response: str) -> None:
        """Store embedding via the global orchestrator for semantic routing."""
        try:
            from core.dual_model import get_dual_model_orchestrator
            orchestrator = get_dual_model_orchestrator()
            if orchestrator:
                orchestrator._store_model_embedding(model, prompt, response)
        except Exception as e:
            logger.debug(f"Could not store embedding: {e}")

    def _report_to_orchestrator(self, model_routing: str, latency_ms: float) -> None:
        """Report call to global orchestrator for unified stats."""
        try:
            from core.dual_model import get_dual_model_orchestrator
            orchestrator = get_dual_model_orchestrator()
            if orchestrator:
                orchestrator.record_subsystem_call(model_routing, latency_ms)
        except Exception as e:
            logger.debug(f"Could not report to orchestrator: {e}")

    def _report_sequential_to_orchestrator(self, latency_ms: float) -> None:
        """Report sequential collaboration to global orchestrator."""
        try:
            from core.dual_model import get_dual_model_orchestrator
            orchestrator = get_dual_model_orchestrator()
            if orchestrator:
                orchestrator.record_sequential_call(latency_ms)
        except Exception as e:
            logger.debug(f"Could not report sequential to orchestrator: {e}")

    # Convenience methods for each subsystem
    
    async def daydream_explore(self, prompt: str, context: dict | None = None) -> str:
        """Creative exploration for daydream engine."""
        ideas = await self.call("daydream", SubsystemTask.CREATIVE_EXPLORATION, prompt, context,
                               system_prompt="You are imagining possibilities. Be creative and generative.")
        evaluation = await self.call("daydream", SubsystemTask.IDEA_EVALUATION,
                                     f"Evaluate these ideas for feasibility and potential:\n\n{ideas}",
                                     context,
                                     system_prompt="You are evaluating ideas critically. Be analytical.")
        combined = []
        if ideas:
            combined.append(ideas)
        if evaluation:
            ev_lines = evaluation.split('\n')
            combined.extend([l for l in ev_lines if l.strip() and not l.strip().startswith('I')][:5])
        return '\n\n'.join(combined) if combined else (ideas or evaluation or "")
    
    async def erdos_solve(self, problem: str, context: dict | None = None) -> str:
        """Math problem solving for Erdős engine."""
        return await self.call("erdos", SubsystemTask.MATH_REASONING, problem, context,
                              system_prompt="You are solving a mathematical problem. Show your reasoning step by step.")
    
    async def erdos_insight(self, discovery: str, context: dict | None = None) -> str:
        """Generate insights from a discovery (both models contribute)."""
        return await self.call("erdos", SubsystemTask.INSIGHT_GENERATION, 
                              f"What insights can we derive from this discovery?\n\n{discovery}",
                              context)
    
    async def evolution_propose(self, area: str, context: dict | None = None) -> str:
        """Generate evolution proposals (primary creative)."""
        return await self.call("evolution", SubsystemTask.PROPOSAL_GENERATION,
                              f"Propose a creative improvement for: {area}",
                              context,
                              system_prompt="You are proposing improvements. Be creative and forward-thinking.")
    
    async def evolution_analyze(self, proposal: str, context: dict | None = None) -> str:
        """Analyze impact of a proposal (backend analytical)."""
        return await self.call("evolution", SubsystemTask.IMPACT_ANALYSIS,
                              f"Analyze the impact and risks of this proposal:\n\n{proposal}",
                              context,
                              system_prompt="You are analyzing proposals critically. Be thorough.")
    
    async def refine_analyze(self, code: str, context: dict | None = None) -> str:
        """Code analysis for self-refine."""
        return await self.call("self_refine", SubsystemTask.CODE_ANALYSIS, code, context,
                              system_prompt="You are analyzing code. Identify issues and potential improvements.")
    
    async def refine_fix(self, issue: str, context: dict | None = None) -> str:
        """Generate fix suggestions (both models contribute)."""
        return await self.call("self_refine", SubsystemTask.FIX_GENERATION,
                              f"Suggest fixes for:\n\n{issue}",
                              context)
    
    async def insight_extract(self, data: str, context: dict | None = None) -> str:
        """Extract patterns from data (both models)."""
        return await self.call("insight_bridge", SubsystemTask.PATTERN_EXTRACTION,
                              f"Extract patterns and insights from:\n\n{data}",
                              context)
    
    async def insight_guidance(self, pattern: str, context: dict | None = None) -> str:
        """Generate evolution guidance from a pattern."""
        return await self.call("insight_bridge", SubsystemTask.EVOLUTION_GUIDANCE,
                              f"How should this pattern guide system evolution?\n\n{pattern}",
                              context)
    
    async def learning_discover(self, experience: str, context: dict | None = None) -> str:
        """Discover patterns from experience (both models)."""
        return await self.call("learning", SubsystemTask.PATTERN_DISCOVERY,
                              f"What patterns can we learn from this experience?\n\n{experience}",
                              context)
    
    async def proactive_curious(self, topic: str, context: dict | None = None) -> str:
        """Generate curiosity-driven exploration."""
        return await self.call("proactive", SubsystemTask.CURIOSITY_GENERATION,
                              f"What should we explore about: {topic}?",
                              context,
                              system_prompt="You are curious and creative. Suggest interesting directions to explore.")
    
    def get_stats(self) -> dict:
        """Get statistics for all subsystems."""
        with self._lock:
            return {
                subsystem: {
                    "total_calls": stats.total_calls,
                    "primary_calls": stats.primary_calls,
                    "backend_calls": stats.backend_calls,
                    "parallel_calls": stats.parallel_calls,
                    "sequential_calls": stats.sequential_calls,
                    "avg_latency_ms": stats.total_latency_ms / max(1, stats.total_calls),
                    "errors": stats.errors,
                }
                for subsystem, stats in self._stats.items()
            }
    
    def get_summary_stats(self) -> dict:
        """Get summary statistics across all subsystems."""
        with self._lock:
            total_calls = 0
            total_primary = 0
            total_backend = 0
            total_parallel = 0
            total_sequential = 0
            total_errors = 0
            
            for stats in self._stats.values():
                total_calls += stats.total_calls
                total_primary += stats.primary_calls
                total_backend += stats.backend_calls
                total_parallel += stats.parallel_calls
                total_sequential += stats.sequential_calls
                total_errors += stats.errors
            
            return {
                "total_calls": total_calls,
                "primary_calls": total_primary,
                "backend_calls": total_backend,
                "parallel_calls": total_parallel,
                "sequential_calls": total_sequential,
                "errors": total_errors,
                "active_subsystems": len([s for s in self._stats.values() if s.total_calls > 0]),
            }
    
    def get_recent_calls(self, subsystem: str | None = None, limit: int = 20) -> list[dict]:
        """Get recent calls, optionally filtered by subsystem."""
        with self._lock:
            calls = self._call_history[-limit:]
            if subsystem:
                calls = [c for c in calls if c.subsystem == subsystem]
            return [
                {
                    "subsystem": c.subsystem,
                    "task": c.task.value,
                    "model_used": c.model_used,
                    "latency_ms": c.latency_ms,
                    "timestamp": c.timestamp.isoformat(),
                    "prompt_preview": c.prompt[:100] + "..." if len(c.prompt) > 100 else c.prompt,
                    "primary_result": c.primary_result[:200] if c.primary_result else None,
                    "backend_result": c.backend_result[:200] if c.backend_result else None,
                }
                for c in calls
            ]
    
    def track_sequential(self, subsystem: str, task1: "SubsystemTask", task2: "SubsystemTask") -> None:
        """Track a sequential collaboration between two tasks.
        
        Call this when two tasks are performed in sequence by different models.
        For example: daydream_explore calls CREATIVE_EXPLORATION (primary) then
        IDEA_EVALUATION (backend) - this is a sequential collaboration.
        """
        with self._lock:
            stats = self._stats.get(subsystem, SubsystemStats())
            stats.sequential_calls += 1
            self._stats[subsystem] = stats
            logger.debug(f"Sequential collaboration tracked: {subsystem}/{task1.value} -> {task2.value}")
    
    def _log_subsystem_activity(self, subsystem: str, task: SubsystemTask, 
                                  model_routing: str, latency_ms: float, preview: str) -> None:
        """Log subsystem activity to API manager for dashboard display."""
        try:
            from core.api_manager import get_api_manager
            manager = get_api_manager()
            
            # Get description for this task
            task_desc = SUBSYSTEM_TASK_DESCRIPTIONS.get(task, task.value)
            subsystem_info = SUBSYSTEM_INFO.get(subsystem, {"name": subsystem.title(), "emoji": "⚙️"})
            
            # Build descriptive activity message
            emoji = subsystem_info.get("emoji", "⚙️")
            model_name = "Kimi K2.5" if model_routing == "primary" else "GLM-5" if model_routing == "backend" else "Both"
            activity_preview = f"{emoji} {task_desc}"
            
            manager.log_activity(
                subsystem=subsystem,
                priority="learning",
                preview=activity_preview,
                latency_ms=latency_ms,
                model=model_name,
                status="complete",
            )
        except Exception as e:
            logger.debug(f"Failed to log subsystem activity: {e}")
    
    def get_subsystem_descriptions(self) -> dict[str, dict]:
        """Get human-readable descriptions for all subsystems."""
        return SUBSYSTEM_INFO.copy()
    
    def get_task_description(self, task: SubsystemTask) -> str:
        """Get human-readable description for a task type."""
        return SUBSYSTEM_TASK_DESCRIPTIONS.get(task, task.value)


def get_dual_subsystem_integration(
    primary_llm=None,
    backend_llm=None,
    model_coordinator=None,
) -> DualSubsystemIntegration:
    """Get or create the global DualSubsystemIntegration."""
    global _dual_subsystem_integration
    
    with _lock:
        if _dual_subsystem_integration is None:
            _dual_subsystem_integration = DualSubsystemIntegration(
                primary_llm=primary_llm,
                backend_llm=backend_llm,
                model_coordinator=model_coordinator,
            )
        else:
            if primary_llm is not None:
                _dual_subsystem_integration.primary_llm = primary_llm
            if backend_llm is not None:
                _dual_subsystem_integration.backend_llm = backend_llm
            if model_coordinator is not None:
                _dual_subsystem_integration.model_coordinator = model_coordinator
    
    return _dual_subsystem_integration