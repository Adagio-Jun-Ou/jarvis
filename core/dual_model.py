"""Dual Model Collaboration System.

Provides seamless cooperation between Primary (Kimi K2.5) and Backend (GLM-5) models
with embedding-based semantic interconnect.

Architecture:
┌──────────────────────────────────────────────────────────────────┐
│                     DUAL MODEL ORCHESTRATOR                       │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────┐     ┌─────────────────────────────────┐    │
│  │   PRIMARY LLM   │     │        BACKEND LLM              │    │
│  │   (Kimi K2.5)   │     │        (GLM-5)                  │    │
│  │                 │     │                                 │    │
│  │  • Chat         │     │  • Code                         │    │
│  │  • Creative     │     │  • Math                         │    │
│  │  • User-facing  │     │  • Reasoning                    │    │
│  │                 │     │  • Analysis                     │    │
│  └────────┬────────┘     └───────────────┬─────────────────┘    │
│           │                              │                      │
│           └──────────┬───────────────────┘                      │
│                      │                                          │
│              ┌───────▼───────┐                                  │
│              │   EMBEDDER    │  ← Semantic Interconnect         │
│              │ (Jina v2)     │    768-dim embeddings            │
│              └───────┬───────┘    Similarity matching           │
│                      │                                          │
│              ┌───────▼───────┐                                  │
│              │ SHARED CONTEXT │                                 │
│              └───────────────┘                                  │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘

The Embedder serves as the semantic interconnect:
- Semantic-based task routing
- Cross-model output similarity analysis
- Learning from successful outputs
- Pattern recognition across model outputs

Collaboration Modes:
1. PARALLEL: Both models work simultaneously on different aspects
2. SEQUENTIAL: One model's output feeds into the other
3. COLLABORATIVE: Models discuss and synthesize together
4. ADAPTIVE: System chooses best model based on task

Usage:
    from core.dual_model import get_dual_model_orchestrator
    
    orchestrator = get_dual_model_orchestrator()
    
    # Route to best model
    response = await orchestrator.route_task("code", "Write a sorting function")
    
    # Parallel execution
    results = await orchestrator.parallel_execute(
        primary_task=("creative", "Write a poem"),
        backend_task=("analysis", "Analyze the structure")
    )
    
    # Collaborative thinking
    synthesis = await orchestrator.think_together(
        topic="Optimal algorithm design",
        query="What's the best approach for large datasets?"
    )
"""
import asyncio
import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.prefrontal_cortex import PrefrontalCortex

import numpy as np

from core.helpers import get_response_content
from .embedding_interconnect import get_embedding_interconnect

# Lazy import for PFC to avoid circular dependency
def _get_pfc() -> "PrefrontalCortex | None":
    """Lazy import and retrieval of Prefrontal Cortex singleton.
    
    Returns:
        PrefrontalCortex instance if available, None otherwise.
        
    Uses defer importing to avoid circular dependencies.
    """
    try:
        from core.prefrontal_cortex import get_prefrontal_cortex
        return get_prefrontal_cortex()
    except Exception as e:
        logger.debug(f"Prefrontal cortex import failed: {e}")
        return None

logger = logging.getLogger(__name__)

# Singleton
_dual_model_orchestrator: Optional["DualModelOrchestrator"] = None
_lock = threading.Lock()

# Context window limits for inter-model communication
# Prevents context overflow when passing outputs between models
MAX_INTER_MODEL_CONTEXT = 4000  # ~1000 tokens (4 chars/token average)
MAX_PROMPT_CACHE = 500  # Max chars to cache for successful outputs


class CollaborationMode(Enum):
    """How models collaborate on a task."""
    PARALLEL = "parallel"          # Both work simultaneously
    SEQUENTIAL = "sequential"      # One after another
    COLLABORATIVE = "collaborative" # Discussion between models
    ADAPTIVE = "adaptive"          # System chooses best model


class TaskType(Enum):
    """Types of tasks that can be routed."""
    CHAT = "chat"
    CREATIVE = "creative"
    CODE = "code"
    MATH = "math"
    ANALYSIS = "analysis"
    REASONING = "reasoning"
    REFINEMENT = "refinement"
    VERIFICATION = "verification"
    THINKING = "thinking"
    VISION = "vision"
    ABSTRACTION = "abstraction"
    SYNTHESIS = "synthesis"
    META_REASONING = "meta_reasoning"
    PATTERN = "pattern"


class ModelRole(Enum):
    """Role of each model."""
    PRIMARY = "primary"    # Kimi K2.5 - user-facing, creative
    BACKEND = "backend"    # GLM-5 - code, math, reasoning, analysis


@dataclass
class ModelConfig:
    """Configuration for a model."""
    model_id: str
    role: ModelRole
    capabilities: list[str]
    preferred_tasks: list[TaskType]
    temperature: float = 0.7
    max_tokens: int = 4096
    supports_thinking: bool = False
    supports_vision: bool = False


@dataclass
class DualModelRequest:
    """A request to the dual-model system."""
    id: str
    task_type: TaskType
    prompt: str
    system_prompt: str = ""
    context: dict = field(default_factory=dict)
    mode: CollaborationMode = CollaborationMode.ADAPTIVE
    priority: int = 5
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Execution tracking
    primary_result: Optional[str] = None
    backend_result: Optional[str] = None
    synthesis: Optional[str] = None
    model_used: Optional[str] = None
    semantic_model: Optional[str] = None  # Model selected by semantic routing
    semantic_confidence: float = 0.0  # Confidence of semantic match


@dataclass
class DualModelStats:
    """Statistics for dual-model operations."""
    total_requests: int = 0
    primary_requests: int = 0
    backend_requests: int = 0
    audio_requests: int = 0  # Audio model calls (Whisper/TTS)
    parallel_requests: int = 0
    collaborative_requests: int = 0
    sequential_requests: int = 0
    cross_model_calls: int = 0
    avg_primary_latency_ms: float = 0.0
    avg_backend_latency_ms: float = 0.0
    errors: int = 0
    primary_requests_per_min: int = 0
    backend_requests_per_min: int = 0
    audio_requests_per_min: int = 0
    total_requests_per_min: int = 0


# Model configurations — ROLE TEMPLATES.
# The model_id here is a DEFAULT only; DualModelOrchestrator.__init__ replaces
# it with the ACTUAL model id from the live clients (primary_llm.model /
# backend_llm.model), which come from config.yaml (llm.model / backend_llm.model).
# Config.yaml is the single source of truth for which models run; these
# templates define each role's capabilities and task routing.
PRIMARY_CONFIG = ModelConfig(
    model_id="glm-5.3-flash",
    role=ModelRole.PRIMARY,
    capabilities=["conversation", "creative", "reasoning", "analysis", "meta_reasoning", "abstraction", "vision"],
    preferred_tasks=[TaskType.CHAT, TaskType.CREATIVE, TaskType.ABSTRACTION, TaskType.META_REASONING, TaskType.PATTERN, TaskType.VISION],
    temperature=0.7,
    supports_thinking=False,
    supports_vision=True,
)

BACKEND_CONFIG = ModelConfig(
    model_id="kimi-k2.7-code",
    role=ModelRole.BACKEND,
    capabilities=["code", "math", "reasoning", "analysis", "thinking", "synthesis"],
    preferred_tasks=[TaskType.CODE, TaskType.MATH, TaskType.ANALYSIS, TaskType.REASONING, TaskType.REFINEMENT, TaskType.VERIFICATION, TaskType.THINKING, TaskType.SYNTHESIS],
    temperature=0.3,
    supports_thinking=True,
)

# Task routing matrix - dual model architecture
TASK_ROUTING = {
    TaskType.CHAT: ModelRole.PRIMARY,
    TaskType.CREATIVE: ModelRole.PRIMARY,
    TaskType.CODE: ModelRole.BACKEND,
    TaskType.MATH: ModelRole.BACKEND,
    TaskType.ANALYSIS: ModelRole.BACKEND,
    TaskType.REASONING: ModelRole.BACKEND,
    TaskType.REFINEMENT: ModelRole.BACKEND,
    TaskType.VERIFICATION: ModelRole.BACKEND,
    TaskType.THINKING: ModelRole.BACKEND,
    TaskType.VISION: ModelRole.PRIMARY,
    TaskType.ABSTRACTION: ModelRole.PRIMARY,
    TaskType.SYNTHESIS: ModelRole.BACKEND,
    TaskType.META_REASONING: ModelRole.PRIMARY,
    TaskType.PATTERN: ModelRole.PRIMARY,
}


class DualModelOrchestrator:
    """Orchestrates collaboration between Primary and Backend models.
    
    Dual-Model Architecture:
    - Primary (Kimi K2.5): Chat, creative, user-facing, vision, meta-reasoning, abstractions, patterns
    - Backend (GLM-5): Code, math, reasoning, analysis, synthesis, refinement
    
    The embedder serves as the semantic interconnect between both models,
    enabling:
    - Semantic-based task routing
    - Cross-model output similarity analysis
    - Learning from successful outputs
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

        # Role templates; the model_id is replaced by the LIVE client's model
        # (from config.yaml) so display, logging and think_together chains
        # always name the model actually serving requests — config.yaml is
        # the single source of truth, not these defaults. Only str values are
        # trusted (test doubles have Mock .model attributes).
        primary_id = getattr(primary_llm, "model", None)
        backend_id = getattr(backend_llm, "model", None)
        self.primary_config = replace(
            PRIMARY_CONFIG,
            model_id=primary_id if isinstance(primary_id, str) and primary_id else PRIMARY_CONFIG.model_id,
        )
        self.backend_config = replace(
            BACKEND_CONFIG,
            model_id=backend_id if isinstance(backend_id, str) and backend_id else BACKEND_CONFIG.model_id,
        )
        
        self._stats = DualModelStats()
        self._request_history: deque[DualModelRequest] = deque(maxlen=100)
        self._call_log: list[dict] = []
        self.MAX_CALL_LOG = 500
        self._call_log_lock = threading.Lock()
        self._lock = threading.RLock()
        
        # Per-minute rate tracking (timestamps of calls) — bounded deques
        self._primary_history: deque[float] = deque(maxlen=1000)
        self._backend_history: deque[float] = deque(maxlen=1000)
        self._audio_history: deque[float] = deque(maxlen=1000)
        self._parallel_history: deque[float] = deque(maxlen=1000)
        
        self._shared_context: dict[str, Any] = {}
        self._context_lock = threading.RLock()
        self._max_shared_context_keys = 100
        
        # Use embedding interconnect for all embedding operations
        # This ensures both models share the same embedding space via IPC
        self._interconnect = get_embedding_interconnect()
        
        self._on_primary_response: Optional[Callable] = None
        self._on_backend_response: Optional[Callable] = None
        self._on_collaboration: Optional[Callable] = None
        
        self._running = False
        
    async def start(self):
        """Start the orchestrator."""
        self._running = True
        logger.info("Dual-model orchestrator started")
        logger.info(f"  Primary: {self.primary_config.model_id} ({', '.join(t.value for t in self.primary_config.preferred_tasks)})")
        logger.info(f"  Backend: {self.backend_config.model_id} ({', '.join(t.value for t in self.backend_config.preferred_tasks)})")
        try:
            # Test that embedding interconnect is working
            test_emb = self._interconnect.encode("test")
            logger.info(f"  Embedding interconnect ready (dim={len(test_emb)}, ipc={getattr(self._interconnect, '_using_ipc', False)})")
        except Exception as e:
            logger.warning(f"  Embedding interconnect unavailable: {e}")
    
    async def stop(self):
        """Stop the orchestrator."""
        self._running = False
        logger.info("Dual-model orchestrator stopped")
    
    def set_callbacks(
        self,
        on_primary_response: Optional[Callable] = None,
        on_backend_response: Optional[Callable] = None,
        on_collaboration: Optional[Callable] = None,
    ):
        """Set callbacks for model responses."""
        self._on_primary_response = on_primary_response
        self._on_backend_response = on_backend_response
        self._on_collaboration = on_collaboration
    
    def get_model_for_task(self, task_type: TaskType) -> ModelRole:
        """Determine which model should handle a task."""
        return TASK_ROUTING.get(task_type, ModelRole.PRIMARY)
    
    def _get_cross_model_context(self, prompt: str, from_model: str) -> str:
        """Retrieve relevant context from the OTHER model's knowledge.
        
        The embedding interconnect lets models 'see' what the other knows:
        - If routing to backend, retrieve relevant context from primary's past outputs
        - If routing to primary, retrieve relevant context from backend's past outputs
        """
        try:
            return self._interconnect.get_cross_model_context(prompt, from_model)
        except Exception as e:
            logger.warning(f"Cross-model context retrieval failed: {e}")
            return ""

    def _store_model_embedding(self, model: str, prompt: str, response: str, quality_score: float = 1.0):
        """Store embedding of model output for semantic routing and learning.
        
        This enables the embedder to serve as an interconnect between models:
        - Store successful outputs for later retrieval
        - Enable semantic similarity matching
        - Learn which model handles which types of queries well
        
        Context window safety:
        - Only stores first MAX_PROMPT_CACHE chars of prompt
        - Embeddings are 768-dim vectors (small, constant size)
        - Max 50 successful outputs per model
        """
        if not prompt or not response:
            return
        try:
            self._interconnect.store_model_output(model, prompt, response, quality_score)
            self._interconnect.store_conversation(model, response)
            logger.info(f"Stored embedding for {model}: prompt_len={len(prompt[:MAX_PROMPT_CACHE])}")
        except Exception as e:
            logger.warning(f"Failed to store model embedding: {e}")
    
    def _semantic_route(self, prompt: str, task_type: TaskType) -> tuple[str, float]:
        """Use embedding similarity and PFC guidance to determine best model.
        
        Returns (model_role, confidence) based on:
        1. PFC concept predictions (if available)
        2. Semantic similarity to past successful outputs
        3. Task-type defaults
        """
        logger.debug(f"[ROUTING] Semantic routing for task_type={task_type.value}, prompt_len={len(prompt)}")
        
        try:
            prompt_emb = self._interconnect.encode(prompt)
            
            primary_outputs = self._interconnect.get_model_outputs("primary")
            backend_outputs = self._interconnect.get_model_outputs("backend")
            
            primary_scores: list[tuple[float, float]] = []
            backend_scores: list[tuple[float, float]] = []
            
            for stored_prompt, quality, stored_emb in primary_outputs:
                sim = float(np.dot(prompt_emb, stored_emb) /
                           (np.linalg.norm(prompt_emb) * np.linalg.norm(stored_emb) + 1e-8))
                if sim > 0.7:
                    primary_scores.append((sim, quality))
                    logger.debug(f"[ROUTING] Primary match: sim={sim:.3f}, quality={quality:.2f}")
            
            for stored_prompt, quality, stored_emb in backend_outputs:
                sim = float(np.dot(prompt_emb, stored_emb) /
                           (np.linalg.norm(prompt_emb) * np.linalg.norm(stored_emb) + 1e-8))
                if sim > 0.7:
                    backend_scores.append((sim, quality))
                    logger.debug(f"[ROUTING] Backend match: sim={sim:.3f}, quality={quality:.2f}")
            
            # Aggregate scores
            if primary_scores:
                primary_conf = sum(s * q for s, q in primary_scores) / len(primary_scores)
            else:
                primary_conf = 0.0
            
            if backend_scores:
                backend_conf = sum(s * q for s, q in backend_scores) / len(backend_scores)
            else:
                backend_conf = 0.0
            
            # Score-based routing
            if backend_conf > primary_conf:
                return "backend", backend_conf
            return "primary", primary_conf
            
        except Exception as e:
            logger.debug(f"Semantic routing failed: {e}")
            # Fall back to task type
            if task_type in [TaskType.CODE, TaskType.MATH, TaskType.ANALYSIS, 
                            TaskType.REASONING, TaskType.VERIFICATION, TaskType.THINKING]:
                return "backend", 0.5
            return "primary", 0.5

    def _find_similar_successful_model(self, prompt: str) -> Optional[str]:
        """Find which model handled similar prompts successfully.
        
        Uses the embedder interconnect to route based on semantic similarity
        to past successful outputs.
        """
        try:
            prompt_emb = self._interconnect.encode(prompt)
            
            best_model = None
            best_score = 0.0
            
            for model in ("primary", "backend", "synthesis"):
                outputs = self._interconnect.get_model_outputs(model)
                for stored_prompt, quality, stored_emb in outputs:
                    sim = float(np.dot(prompt_emb, stored_emb) / 
                               (np.linalg.norm(prompt_emb) * np.linalg.norm(stored_emb) + 1e-8))
                    weighted_sim = sim * quality
                    if weighted_sim > best_score and sim > 0.75:
                        best_score = weighted_sim
                        best_model = model
            
            return best_model
        except Exception as e:
            logger.debug(f"Interconnect model selection failed: {e}")
            return None
    
    async def route_task(
        self,
        task_type: TaskType,
        prompt: str,
        system_prompt: str = "",
        context: dict | None = None,
        mode: CollaborationMode = CollaborationMode.ADAPTIVE,
    ) -> str:
        """Route a task to the appropriate model using semantic interconnect.
        
        The embedder serves as the TRUE interconnect:
        1. Embed query semantically
        2. Compare to past successful outputs
        3. Route to model with highest semantic similarity
        4. Store embeddings of prompts and responses for learning
        """
        # SEMANTIC ROUTING: Use embedding to decide model
        model_role, confidence = self._semantic_route(prompt, task_type)
        
        # Determine collaboration mode
        if mode == CollaborationMode.ADAPTIVE:
            # High confidence semantic match -> use that model directly
            if confidence > 0.75:
                # Override mode to use single model (backend or primary)
                # We'll pass this to _execute_single which picks based on model_role
                routing_reason = f"semantic_match:{model_role}:{confidence:.2f}"
                logger.info(f"Semantic routing to {model_role} (confidence={confidence:.2f})")
            else:
                # Fall back to task-type routing
                mode = self._determine_mode(task_type, prompt)
                routing_reason = f"task_type:{task_type.value}"
        else:
            routing_reason = f"explicit:{mode.value}"
        
        request = DualModelRequest(
            id=f"req_{uuid.uuid4().hex[:8]}",
            task_type=task_type,
            prompt=prompt,
            system_prompt=system_prompt,
            context=context or {},
            mode=mode,
        )
        request.context["routing_reason"] = routing_reason
        request.semantic_model = model_role
        request.semantic_confidence = confidence
        
        if context and "conversation" in context:
            for msg in context.get("conversation", [])[-10:]:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                if content:
                    self._interconnect.store_conversation(role, content)
        
        similar_model = self._find_similar_successful_model(prompt)
        if similar_model:
            request.context["routing_hint"] = f"Similar to past {similar_model} success"
        
        with self._lock:
            self._stats.total_requests += 1
            self._request_history.append(request)
        
        try:
            if mode == CollaborationMode.PARALLEL:
                return await self._execute_parallel(request)
            elif mode == CollaborationMode.SEQUENTIAL:
                return await self._execute_sequential(request)
            elif mode == CollaborationMode.COLLABORATIVE:
                return await self._execute_collaborative(request)
            else:
                return await self._execute_single(request)
        except Exception as e:
            logger.error(f"Dual-model request failed: {e}")
            with self._lock:
                self._stats.errors += 1
            raise
    
    def _determine_mode(self, task_type: TaskType, prompt: str) -> CollaborationMode:
        """Determine the best collaboration mode for a task.
        
        By default, both models should contribute - this is the key to
        dual-model intelligence. We only use single model when:
        - Very simple chat (just greeting/acknowledgment)
        - System maintenance tasks
        """
        # Check prompt complexity
        word_count = len(prompt.split())
        is_simple = word_count < 10
        
        # Simple chat/greetings -> single model (fast response)
        if task_type == TaskType.CHAT and is_simple:
            return CollaborationMode.ADAPTIVE  # Single model for simple chat
        
        # MATH and VERIFICATION: Backend only, no synthesis needed
        if task_type in [TaskType.MATH, TaskType.VERIFICATION]:
            return CollaborationMode.ADAPTIVE  # Backend handles alone
        
        # CODE: Backend for implementation
        if task_type == TaskType.CODE:
            return CollaborationMode.SEQUENTIAL  # Primary understands, Backend implements
        
        # ANALYSIS/REASONING: Both models think
        if task_type in [TaskType.ANALYSIS, TaskType.REASONING]:
            return CollaborationMode.PARALLEL
        
        # CREATIVE: Primary creates
        if task_type == TaskType.CREATIVE:
            return CollaborationMode.ADAPTIVE  # Primary handles creative
        
        # Default: single model for chat
        return CollaborationMode.ADAPTIVE
    
    async def _execute_single(self, request: DualModelRequest) -> str:
        """Execute on a single model with embedding storage and cross-model context."""
        # Use semantic routing decision if high confidence
        if request.semantic_confidence > 0.75 and request.semantic_model:
            model_role = ModelRole.BACKEND if request.semantic_model == "backend" else ModelRole.PRIMARY
            logger.debug(f"Using semantic routing: {model_role.value} (confidence={request.semantic_confidence:.2f})")
        else:
            model_role = self.get_model_for_task(request.task_type)
        
        # Get cross-model context - can this model benefit from what the OTHER model knows?
        # Skip cross-model context for MATH tasks - GLM-5 should just do math
        cross_context = ""
        if request.task_type not in [TaskType.MATH, TaskType.VERIFICATION]:
            cross_context = self._get_cross_model_context(request.prompt, model_role.value)
        
        if model_role == ModelRole.PRIMARY:
            request.model_used = self.primary_config.model_id
            with self._lock:
                self._stats.primary_requests += 1
            self._log_call("primary")
            
            result = await self._call_primary(request.prompt, request.system_prompt, cross_context)
            request.primary_result = result
            self._store_model_embedding("primary", request.prompt, result)
            
            if self._on_primary_response:
                try:
                    self._on_primary_response(request)
                except Exception as e:
                    logger.debug(f"Primary response callback failed: {e}")
        else:
            request.model_used = self.backend_config.model_id
            with self._lock:
                self._stats.backend_requests += 1
            self._log_call("backend")
            
            result = await self._call_backend(request.prompt, request.system_prompt, cross_context)
            request.backend_result = result
            self._store_model_embedding("backend", request.prompt, result)
            
            if self._on_backend_response:
                try:
                    self._on_backend_response(request)
                except Exception as e:
                    logger.debug(f"Backend response callback failed: {e}")
        
        return result
    
    async def _execute_parallel(self, request: DualModelRequest) -> str:
        """Execute on both models in parallel with embedding storage."""
        with self._lock:
            self._stats.parallel_requests += 1
        self._log_call("parallel")
        
        # Get cross-model context for each model
        primary_context = self._get_cross_model_context(request.prompt, "primary")
        backend_context = self._get_cross_model_context(request.prompt, "backend")
        
        primary_task = self._call_primary(request.prompt, request.system_prompt, primary_context)
        backend_task = self._call_backend(request.prompt, request.system_prompt, backend_context)
        
        results = await asyncio.gather(
            primary_task, backend_task, return_exceptions=True
        )
        
        primary_result = results[0] if not isinstance(results[0], Exception) else None
        backend_result = results[1] if not isinstance(results[1], Exception) else None
        
        request.primary_result = str(primary_result) if primary_result is not None else None
        request.backend_result = str(backend_result) if backend_result is not None else None
        
        if primary_result:
            self._store_model_embedding("primary", request.prompt, str(primary_result))
        if backend_result:
            self._store_model_embedding("backend", request.prompt, str(backend_result))
        
        with self._lock:
            self._stats.cross_model_calls += 1
        
        if primary_result and backend_result:
            synthesis = await self._synthesize_results(
                str(primary_result), str(backend_result), request.task_type
            )
            request.synthesis = synthesis
            self._store_model_embedding("synthesis", request.prompt, synthesis)
            return synthesis
        elif primary_result:
            return str(primary_result)
        elif backend_result:
            return str(backend_result)
        else:
            raise RuntimeError("Both models failed to produce results")
    
    async def _execute_sequential(self, request: DualModelRequest) -> str:
        """Execute sequentially with embedding storage for cross-model learning."""
        with self._lock:
            self._stats.sequential_requests += 1
        
        backend_leads = request.task_type in [TaskType.MATH, TaskType.VERIFICATION]
        
        if backend_leads:
            # Backend leads - MATH tasks should NOT get cross-model context
            # GLM-5 handles math directly, no need for Kimi's conversational context
            request.model_used = "backend"
            backend_result = await self._call_backend(request.prompt, request.system_prompt)
            request.backend_result = backend_result
            self._store_model_embedding("backend", request.prompt, backend_result)
            
            with self._lock:
                self._stats.cross_model_calls += 1
            
            # Return backend result directly - no synthesis needed for math
            return backend_result
        else:
            # Primary leads, then backend analyzes
            primary_result = await self._call_primary(request.prompt, request.system_prompt)
            request.primary_result = primary_result
            request.model_used = "primary"
            self._store_model_embedding("primary", request.prompt, primary_result)
            
            # For chat: don't append technical analysis, just return the natural response
            if request.task_type == TaskType.CHAT:
                return str(primary_result)
            
            # For other tasks: primary explores, backend analyzes
            truncated_primary = primary_result[:MAX_INTER_MODEL_CONTEXT] if len(primary_result) > MAX_INTER_MODEL_CONTEXT else primary_result
            
            sanitized_prompt = request.prompt[:500].replace("---", "—")
            sanitized_primary = truncated_primary[:MAX_INTER_MODEL_CONTEXT].replace("---", "—")
            
            backend_prompt = f"""Based on this initial response, provide a detailed technical analysis:

Original prompt (summarized): {sanitized_prompt}

Initial response: {sanitized_primary}

Provide technical refinement, code implementation, or detailed analysis:"""
            
            backend_result = await self._call_backend(backend_prompt, "You are a technical expert refining ideas.")
            request.backend_result = backend_result
            self._store_model_embedding("backend", request.prompt, backend_result)
            
            with self._lock:
                self._stats.cross_model_calls += 1
            
            # Return synthesized result combining both perspectives
            return f"{primary_result}\n\n**Technical Refinement:** {backend_result}"
    
    async def _execute_collaborative(self, request: DualModelRequest, rounds: int = 2) -> str:
        """Execute with model collaboration through discussion and embedding storage.

        `rounds` used to be hardcoded to 2 here while think_together() accepted a
        rounds argument and reported it back in its callback — so callers were
        told their value had been honoured when it had not.
        """
        with self._lock:
            self._stats.collaborative_requests += 1

        if self.model_coordinator:
            chain = await self.model_coordinator.think_together(
                topic=request.task_type.value,
                initial_query=request.prompt,
                models=[self.primary_config.model_id, self.backend_config.model_id],
                rounds=rounds,
            )
            
            if chain and chain.turns:
                request.synthesis = chain.synthesis
                
                # ReasoningChain.turns is list[dict] — add_turn() appends
                # {"model", "thinking", "output", "timestamp"}. This used to read
                # turn.model_id / turn.content as attributes, which raises
                # AttributeError on the first turn, so collaborative mode never
                # completed and no cross-model embedding was ever stored.
                for turn in chain.turns:
                    if turn.get("model") == self.primary_config.model_id:
                        model_name = "primary"
                    else:
                        model_name = "backend"
                    self._store_model_embedding(model_name, request.prompt, turn.get("output", ""))
                
                if chain.synthesis:
                    self._store_model_embedding("collaboration", request.prompt, chain.synthesis)
                
                with self._lock:
                    self._stats.cross_model_calls += len(chain.turns)
                return chain.synthesis
        
        result = await self._execute_parallel(request)
        self._store_model_embedding("collaboration", request.prompt, result)
        return result
    
    async def _call_model(
        self,
        model_type: str,
        prompt: str,
        system_prompt: str = "",
        cross_model_context: str = "",
        max_retries: int = 2,
        timeout: float = 120.0,
        default_system: str = "You are a helpful AI assistant.",
        extract_thinking: bool = False,
    ) -> str:
        """Call a model with retry and timeout.

        Args:
            model_type: "primary" or "backend"
            prompt: User input prompt
            system_prompt: System prompt override
            cross_model_context: Context from the other model
            max_retries: Number of retry attempts
            timeout: Timeout in seconds
            default_system: Default system prompt if none provided
            extract_thinking: If True, extract thinking tags (backend behavior)

        Returns:
            Model response string
        """
        llm = None
        with self._lock:
            llm = self.primary_llm if model_type == "primary" else self.backend_llm
        label = model_type.upper()
        stat_key = f"avg_{model_type}_latency_ms"

        if not llm:
            raise RuntimeError(f"{label} LLM not configured")

        if cross_model_context:
            if len(cross_model_context) > 2000:
                cross_model_context = cross_model_context[:2000] + "...[truncated]"
            prompt = f"[Cross-model context]: {cross_model_context}\n\n{prompt}"

        effective_prompt = system_prompt or default_system
        last_error = None
        retry_delay = 1.0

        for attempt in range(max_retries + 1):
            start = time.time()
            try:
                response = await asyncio.wait_for(
                    llm.chat(
                        user_input=prompt,
                        system_prompt=effective_prompt,
                        conversation_history=[],
                    ),
                    timeout=timeout,
                )

                latency = (time.time() - start) * 1000
                with self._lock:
                    current = getattr(self._stats, stat_key, 0)
                    setattr(self._stats, stat_key, current * 0.9 + latency * 0.1)

                if attempt > 0:
                    logger.info(f"[{label}] Recovered on attempt {attempt + 1}")

                logger.debug(f"[{label}] Success: {latency:.0f}ms, {len(str(response))} chars")

                if isinstance(response, dict):
                    content = get_response_content(response)
                    if content is None:
                        content = ""
                    if extract_thinking:
                        thinking = response.get("thinking", "") or ""
                        if thinking:
                            return f"<thinking>\n{thinking}\n</thinking>\n\n{content}"
                    return content
                result = str(response) if response is not None else ""
                return result

            except asyncio.TimeoutError:
                last_error = TimeoutError(f"{label} timeout after {timeout}s")
                logger.warning(f"[{label}] Timeout on attempt {attempt + 1}/{max_retries + 1}")
                if attempt < max_retries:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2

            except Exception as e:
                last_error = e
                logger.error(f"[{label}] Error on attempt {attempt + 1}: {e}")
                if attempt < max_retries:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2

        raise RuntimeError(f"{label} model failed after {max_retries + 1} attempts") from last_error

    async def _call_primary(
        self,
        prompt: str,
        system_prompt: str = "",
        cross_model_context: str = "",
        max_retries: int = 2,
        timeout: float = 120.0,
    ) -> str:
        """Call the primary model with retry and timeout."""
        return await self._call_model(
            model_type="primary",
            prompt=prompt,
            system_prompt=system_prompt,
            cross_model_context=cross_model_context,
            max_retries=max_retries,
            timeout=timeout,
            default_system="You are a helpful AI assistant.",
            extract_thinking=False,
        )

    async def _call_backend(
        self,
        prompt: str,
        system_prompt: str = "",
        cross_model_context: str = "",
        max_retries: int = 2,
        timeout: float = 180.0,
    ) -> str:
        """Call the backend model with retry and timeout."""
        return await self._call_model(
            model_type="backend",
            prompt=prompt,
            system_prompt=system_prompt,
            cross_model_context=cross_model_context,
            max_retries=max_retries,
            timeout=timeout,
            default_system="You are an expert AI assistant.",
            extract_thinking=True,
        )

    async def _synthesize_results(
        self,
        primary_result: str,
        backend_result: str,
        task_type: TaskType,
    ) -> str:
        """Synthesize results from both models into a unified response.
        
        Context window safety:
        - Truncates results to MAX_INTER_MODEL_CONTEXT chars
        - Combined output stays under MAX_INTER_MODEL_CONTEXT
        - Uses semantic similarity to select best content when both models contribute
        """
        half_budget = MAX_INTER_MODEL_CONTEXT // 2
        
        if not primary_result and not backend_result:
            return ""
        if not primary_result:
            return backend_result[:MAX_INTER_MODEL_CONTEXT]
        if not backend_result:
            return primary_result[:MAX_INTER_MODEL_CONTEXT]
        
        primary_truncated = primary_result[:half_budget]
        backend_truncated = backend_result[:half_budget]
        
        if task_type in [TaskType.CODE, TaskType.MATH, TaskType.ANALYSIS, TaskType.VERIFICATION]:
            return backend_truncated
        elif task_type in [TaskType.CREATIVE, TaskType.CHAT]:
            return primary_truncated
        elif task_type == TaskType.REASONING:
            if len(backend_truncated) > len(primary_truncated) * 1.5:
                return backend_truncated
            return primary_truncated
        else:
            if len(backend_truncated) > len(primary_truncated):
                return backend_truncated
            return primary_truncated
    
    async def parallel_execute(
        self,
        primary_task: tuple[TaskType, str],
        backend_task: tuple[TaskType, str],
    ) -> dict[str, str | None]:
        """Execute different tasks on both models in parallel.
        
        Args:
            primary_task: (task_type, prompt) for primary model
            backend_task: (task_type, prompt) for backend model
            
        Returns:
            Dict with 'primary' and 'backend' results (may be None on failure)
        """
        primary_type, primary_prompt = primary_task
        backend_type, backend_prompt = backend_task
        
        results = await asyncio.gather(
            self.route_task(primary_type, primary_prompt),
            self.route_task(backend_type, backend_prompt),
            return_exceptions=True,
        )
        
        primary_result = results[0] if not isinstance(results[0], Exception) else None
        backend_result = results[1] if not isinstance(results[1], Exception) else None
        
        return {
            "primary": str(primary_result) if primary_result is not None else None,
            "backend": str(backend_result) if backend_result is not None else None,
        }
    
    async def think_together(
        self,
        topic: str,
        query: str,
        rounds: int = 2,
    ) -> str:
        """Have both models think together on a topic."""
        request = DualModelRequest(
            id=f"collab_{int(time.time() * 1000)}",
            task_type=TaskType.ANALYSIS,
            prompt=query,
            mode=CollaborationMode.COLLABORATIVE,
        )
        
        result = await self._execute_collaborative(request, rounds=rounds)

        if self._on_collaboration:
            try:
                self._on_collaboration({
                    "topic": topic,
                    "query": query,
                    "rounds": rounds,
                    "result": result,
                })
            except Exception as e:
                logger.debug(f"Collaboration callback failed: {e}")
        
        return result
    
    def set_shared_context(self, key: str, value: Any, model: str = "system"):
        """Set a value in shared context."""
        with self._context_lock:
            if key not in self._shared_context:
                self._shared_context[key] = {}
            
            # Limit models per key to prevent unbounded growth
            if len(self._shared_context[key]) >= 10:
                oldest_model = min(
                    self._shared_context[key].items(),
                    key=lambda x: x[1].get("timestamp", ""),
                    default=(None, None)
                )
                if oldest_model[0]:
                    del self._shared_context[key][oldest_model[0]]
            
            self._shared_context[key][model] = {
                "value": value,
                "timestamp": datetime.now().isoformat(),
            }
            while len(self._shared_context) > self._max_shared_context_keys:
                if not self._shared_context:
                    break
                oldest_key = min(
                    self._shared_context.keys(),
                    key=lambda k: min(
                        (v.get("timestamp", "") for v in self._shared_context[k].values()),
                        default=""
                    )
                )
                del self._shared_context[oldest_key]
    
    def get_shared_context(self, key: str) -> Optional[Any]:
        """Get a value from shared context."""
        with self._context_lock:
            if key in self._shared_context:
                # Return the latest value from any model
                data = self._shared_context[key]
                if data:
                    latest = max(data.items(), key=lambda x: x[1].get("timestamp", ""))
                    return latest[1].get("value")
        return None
    
    def get_all_shared_context(self) -> dict:
        """Get all shared context."""
        with self._context_lock:
            return dict(self._shared_context)
    
    def _log_call(self, model: str) -> None:
        """Log a model call for per-minute rate tracking."""
        from datetime import datetime
        now_ts = datetime.now().timestamp()
        with self._call_log_lock:
            self._call_log.append({
                "model": model,
                "timestamp": now_ts,
            })
            five_min_ago = now_ts - 300
            self._call_log = [c for c in self._call_log if c.get("timestamp", 0) > five_min_ago]
            if len(self._call_log) > self.MAX_CALL_LOG:
                self._call_log = self._call_log[-self.MAX_CALL_LOG:]
    
    def get_stats(self) -> dict:
        """Get orchestrator statistics.
        
        Acquires _call_log_lock before _lock to maintain consistent lock ordering
        and avoid potential deadlocks.
        """
        with self._call_log_lock:
            from datetime import datetime
            one_min_ago = datetime.now().timestamp() - 60
            recent_calls = [
                c for c in self._call_log 
                if c.get("timestamp", 0) > one_min_ago
            ]
            _primary_from_log = sum(1 for c in recent_calls if c.get("model") == "primary")
            _backend_from_log = sum(1 for c in recent_calls if c.get("model") == "backend")
            _parallel_from_log = sum(1 for c in recent_calls if c.get("model") == "parallel")
        
        with self._lock:
            now = time.time()
            cutoff = now - 60
            primary_recent = sum(1 for t in self._primary_history if t > cutoff)
            backend_recent = sum(1 for t in self._backend_history if t > cutoff)
            audio_recent = sum(1 for t in self._audio_history if t > cutoff) if hasattr(self, '_audio_history') else 0
            parallel_recent = sum(1 for t in self._parallel_history if t > cutoff)
            
            return {
                "total_requests": self._stats.total_requests,
                "primary_requests": self._stats.primary_requests,
                "backend_requests": self._stats.backend_requests,
                "audio_requests": self._stats.audio_requests if hasattr(self._stats, 'audio_requests') else 0,
                "parallel_requests": self._stats.parallel_requests,
                "collaborative_requests": self._stats.collaborative_requests,
                "sequential_requests": self._stats.sequential_requests,
                "cross_model_calls": self._stats.cross_model_calls,
                "avg_primary_latency_ms": round(self._stats.avg_primary_latency_ms, 2),
                "avg_backend_latency_ms": round(self._stats.avg_backend_latency_ms, 2),
                "errors": self._stats.errors,
                "primary_connected": self.primary_llm is not None,
                "backend_connected": self.backend_llm is not None,
                "audio_connected": True,  # Audio model (Whisper/TTS) availability
                "shared_context_keys": len(self._shared_context),
                "primary_requests_per_min": primary_recent,
                "backend_requests_per_min": backend_recent,
                "audio_requests_per_min": audio_recent,
                "total_requests_per_min": primary_recent + backend_recent + audio_recent + parallel_recent,
            }

    def record_subsystem_call(self, model_routing: str, latency_ms: float) -> None:
        """Record a call from a subsystem for unified stats.
        
        Called by DualSubsystemIntegration to contribute to global stats.
        Also tracks per-minute rates.
        
        Args:
            model_routing: 'primary', 'backend', 'audio', or 'parallel'
            latency_ms: Latency in milliseconds
        """
        with self._lock:
            self._stats.total_requests += 1
            if model_routing == "primary":
                self._stats.primary_requests += 1
                self._primary_history.append(time.time())
                self._stats.avg_primary_latency_ms = (
                    self._stats.avg_primary_latency_ms * 0.9 + latency_ms * 0.1
                )
            elif model_routing == "backend":
                self._stats.backend_requests += 1
                self._backend_history.append(time.time())
                self._stats.avg_backend_latency_ms = (
                    self._stats.avg_backend_latency_ms * 0.9 + latency_ms * 0.1
                )
            elif model_routing == "audio":
                self._stats.audio_requests += 1
                self._audio_history.append(time.time())
            elif model_routing == "parallel":
                self._stats.parallel_requests += 1
                self._parallel_history.append(time.time())
                self._stats.avg_primary_latency_ms = (
                    self._stats.avg_primary_latency_ms * 0.9 + latency_ms * 0.05
                )
                self._stats.avg_backend_latency_ms = (
                    self._stats.avg_backend_latency_ms * 0.9 + latency_ms * 0.05
                )
            
            # Trim histories to prevent memory bloat
            # Deques auto-trim via maxlen; just remove entries older than cutoff
            cutoff = time.time() - 120
            while self._primary_history and self._primary_history[0] < cutoff:
                self._primary_history.popleft()
            while self._backend_history and self._backend_history[0] < cutoff:
                self._backend_history.popleft()
            while self._audio_history and self._audio_history[0] < cutoff:
                self._audio_history.popleft()
            while self._parallel_history and self._parallel_history[0] < cutoff:
                self._parallel_history.popleft()

    def record_audio_call(self, latency_ms: float) -> None:
        """Record an audio model call (Whisper/TTS).
        
        Args:
            latency_ms: Latency in milliseconds
        """
        self.record_subsystem_call("audio", latency_ms)

    def record_sequential_call(self, latency_ms: float) -> None:
        """Record a sequential collaboration call.
        
        Called when two models collaborate in sequence (one leads, other follows).
        """
        with self._lock:
            self._stats.sequential_requests += 1
            self._stats.cross_model_calls += 1

    def record_collaborative_call(self, latency_ms: float) -> None:
        """Record a collaborative call (think_together)."""
        with self._lock:
            self._stats.collaborative_requests += 1
            self._stats.cross_model_calls += 1


def get_dual_model_orchestrator(
    primary_llm=None,
    backend_llm=None,
    model_coordinator=None,
) -> DualModelOrchestrator:
    """Get or create the global DualModelOrchestrator."""
    global _dual_model_orchestrator
    
    with _lock:
        if _dual_model_orchestrator is None:
            _dual_model_orchestrator = DualModelOrchestrator(
                primary_llm=primary_llm,
                backend_llm=backend_llm,
                model_coordinator=model_coordinator,
            )
        else:
            if primary_llm is not None:
                with _dual_model_orchestrator._lock:
                    _dual_model_orchestrator.primary_llm = primary_llm
            if backend_llm is not None:
                with _dual_model_orchestrator._lock:
                    _dual_model_orchestrator.backend_llm = backend_llm
            if model_coordinator is not None:
                _dual_model_orchestrator.model_coordinator = model_coordinator
    
    return _dual_model_orchestrator


def init_dual_model(
    primary_llm,
    backend_llm,
    model_coordinator=None,
) -> DualModelOrchestrator:
    """Initialize the dual-model system."""
    orchestrator = get_dual_model_orchestrator(
        primary_llm=primary_llm,
        backend_llm=backend_llm,
        model_coordinator=model_coordinator,
    )
    return orchestrator