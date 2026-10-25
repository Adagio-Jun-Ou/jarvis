"""Model Coordinator - Connective layer between multiple LLM models.

Enables models to communicate THROUGH the Jarvis stack:
- Shared context and memory
- Structured handoff protocol
- Capability-based routing
- Cross-model reasoning chains

This is NOT an external API bridge - it's an internal coordination layer
that allows Kimi K2.5 and GLM-5 to work together seamlessly.
"""
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace

from datetime import datetime
from enum import Enum
from typing import Any, Optional
import json

from core.helpers import get_response_content, get_response_thinking, safe_error_message

logger = logging.getLogger(__name__)

_model_coordinator: Optional["ModelCoordinator"] = None
_coordinator_lock = threading.Lock()


class ModelRole(str, Enum):
    """Roles for different models."""
    PRIMARY = "primary"          # Main conversational model (Kimi K2.5)
    BACKEND = "backend"          # Backend tasks, code, analysis (GLM-5)
    REASONING = "reasoning"      # Complex reasoning chains
    VISION = "vision"            # Vision/image processing
    SPECIALIZED = "specialized"  # Domain-specific tasks


class HandoffReason(str, Enum):
    """Reasons for model handoff."""
    CAPABILITY = "capability"        # Model has specific capability
    SPECIALIZATION = "specialization" # Domain specialization
    LOAD_BALANCE = "load_balance"    # Distribute load
    FALLBACK = "fallback"            # Primary unavailable
    COLLABORATION = "collaboration"  # Multi-model reasoning
    USER_REQUEST = "user_request"    # User specified model


class MessageType(str, Enum):
    """Types of inter-model messages."""
    HANDOFF = "handoff"              # Transfer context to another model
    QUERY = "query"                  # Ask another model something
    SYNTHESIZE = "synthesize"        # Combine outputs from multiple models
    VERIFY = "verify"                # Ask another model to verify
    REFINE = "refine"                # Ask for refinement
    THINK_TOGETHER = "think_together" # Collaborative reasoning


@dataclass
class ModelConfig:
    """Configuration for a model."""
    model_id: str
    role: ModelRole
    capabilities: list[str]
    context_window: int
    supports_thinking: bool = False
    supports_vision: bool = False
    preferred_tasks: list[str] = field(default_factory=list)
    temperature: float = 0.7
    max_tokens: int = 4096


@dataclass
class InterModelMessage:
    """A message passed between models."""
    id: str
    message_type: MessageType
    from_model: str
    to_model: str
    context: dict[str, Any]
    query: str
    reasoning_chain: list[dict] = field(default_factory=list)
    handoff_reason: HandoffReason | None = None
    created_at: datetime = field(default_factory=datetime.now)
    response: str | None = None
    response_from: str | None = None
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "message_type": self.message_type.value,
            "from_model": self.from_model,
            "to_model": self.to_model,
            "context": self.context,
            "query": self.query,
            "reasoning_chain": self.reasoning_chain,
            "handoff_reason": self.handoff_reason.value if self.handoff_reason else None,
            "created_at": self.created_at.isoformat(),
            "response": self.response,
            "response_from": self.response_from,
            "timestamp": self.timestamp,
        }


@dataclass
class ReasoningChain:
    """A chain of reasoning across multiple models."""
    id: str
    topic: str
    turns: list[dict] = field(default_factory=list)
    synthesis: str | None = None
    complete: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    
    _max_turns = 50
    _turns_deque: deque = field(default_factory=lambda: deque(maxlen=50))
    _use_deque: bool = False
    
    def add_turn(self, model: str, thinking: str, output: str):
        """Add a reasoning turn with bound checking."""
        turn = {
            "model": model,
            "thinking": thinking[:2000],
            "output": output[:2000],
            "timestamp": datetime.now().isoformat(),
        }
        self.turns.append(turn)
        if len(self.turns) > self._max_turns:
            self.turns = self.turns[-self._max_turns:]
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "topic": self.topic,
            "turns": self.turns,
            "synthesis": self.synthesis,
            "complete": self.complete,
            "created_at": self.created_at.isoformat(),
        }


class SharedContext:
    """Shared context that all models can access.
    
    This is the key integration point - models communicate through
    this shared context, not by calling each other's APIs directly.
    """
    
    def __init__(self):
        self._context: dict[str, Any] = {}
        self._memory_bridge: Any = None
        self._context_manager: Any = None
        self._lock = threading.RLock()
        self._message_log: list[InterModelMessage] = []
        self._reasoning_chains: dict[str, ReasoningChain] = {}
        self._max_context_keys = 100
        self._max_reasoning_chains = 50
    
    def attach_memory(self, memory) -> None:
        """Attach the unified memory system."""
        self._memory_bridge = memory
    
    def attach_context_manager(self, context_manager) -> None:
        """Attach the context manager."""
        self._context_manager = context_manager
    
    def set(self, key: str, value: Any, model_id: str = "system") -> None:
        """Set a shared context value."""
        with self._lock:
            self._context[key] = {
                "value": value,
                "set_by": model_id,
                "timestamp": datetime.now().isoformat(),
            }
            # Enforce bounds
            while len(self._context) > self._max_context_keys:
                oldest_key = min(
                    self._context.keys(),
                    key=lambda k: self._context[k].get("timestamp", ""),
                    default=None
                )
                if oldest_key:
                    del self._context[oldest_key]
                else:
                    break
    
    def get(self, key: str) -> Any | None:
        """Get a shared context value."""
        with self._lock:
            entry = self._context.get(key)
            if entry is None:
                return None
            return entry.get("value")
    
    def get_all(self) -> dict[str, Any]:
        """Get all shared context."""
        with self._lock:
            return {k: v["value"] for k, v in self._context.items()}
    
    def log_message(self, message: InterModelMessage) -> None:
        """Log an inter-model message."""
        with self._lock:
            self._message_log.append(message)
            if len(self._message_log) > 100:
                self._message_log = self._message_log[-100:]
    
    def get_messages(self, limit: int = 20) -> list[dict]:
        """Get recent inter-model messages."""
        with self._lock:
            return [m.to_dict() for m in self._message_log[-limit:]]
    
    def create_reasoning_chain(self, topic: str) -> ReasoningChain:
        """Create a new reasoning chain for collaborative thinking."""
        import hashlib
        chain_id = hashlib.sha256(f"{topic}{time.time()}".encode()).hexdigest()[:12]
        chain = ReasoningChain(id=chain_id, topic=topic)
        with self._lock:
            self._reasoning_chains[chain_id] = chain
            # Enforce bounds
            while len(self._reasoning_chains) > self._max_reasoning_chains:
                oldest_id = min(
                    self._reasoning_chains.keys(),
                    key=lambda k: self._reasoning_chains[k].created_at,
                    default=None
                )
                if oldest_id:
                    del self._reasoning_chains[oldest_id]
                else:
                    break
        return chain
    
    def get_reasoning_chain(self, chain_id: str) -> ReasoningChain | None:
        """Get a reasoning chain by ID."""
        with self._lock:
            return self._reasoning_chains.get(chain_id)


def _live_model_id(client, default: str) -> str:
    """Extract the model id from a live client, trusting only str values.

    Test doubles (MagicMock/AsyncMock) expose .model as a Mock — using that
    as a dict key would corrupt the registry; fall back to the role default.
    """
    model = getattr(client, "model", None)
    return model if isinstance(model, str) and model else default


class ModelCoordinator:
    """Coordinates multiple LLM models through shared context.
    
    This is the connective layer between models. Models don't call
    each other directly - they communicate through this coordinator
    which manages shared state, handoffs, and reasoning chains.
    
    Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │                    Model Coordinator                      │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
    │  │Shared Context│  │Memory Bridge│  │Handoff Queue│      │
    │  └─────────────┘  └─────────────┘  └─────────────┘      │
    └─────────────────────────────────────────────────────────┘
              ↑                    ↑                    ↑
              │                    │                    │
    ┌─────────┴────────┐  ┌───────┴────────┐  ┌───────┴────────┐
    │   Kimi K2.5      │  │     GLM-5      │  │   Qwen3-VL     │
    │   (Primary)      │  │    (Backend)   │  │    (Vision)    │
    └──────────────────┘  └────────────────┘  └────────────────┘
    
    All models access the same shared context, memory, and can
    participate in reasoning chains together.
    """
    
    DEFAULT_MODELS = {
        # ROLE TEMPLATE keyed by the historical primary id. __init__ replaces
        # this entry (and the backend one) with the LIVE model ids from the
        # actual clients — config.yaml (llm.model / backend_llm.model) is the
        # single source of truth for which models run.
        "kimi-k2.5": ModelConfig(
            model_id="kimi-k2.5",
            role=ModelRole.PRIMARY,
            capabilities=["conversation", "reasoning", "creative", "analysis", "meta_reasoning", "abstraction", "vision"],
            context_window=131072,
            supports_thinking=False,
            supports_vision=True,
            preferred_tasks=["chat", "creative", "analysis", "abstraction", "meta_reasoning", "pattern", "vision"],
            temperature=0.7,
        ),
        "glm-5": ModelConfig(
            model_id="glm-5",
            role=ModelRole.BACKEND,
            capabilities=["code", "math", "reasoning", "analysis", "thinking", "synthesis"],
            context_window=131072,
            supports_thinking=True,
            preferred_tasks=["code", "math", "refinement", "verification", "synthesis"],
            temperature=0.3,
        ),
        "qwen3.5:397b": ModelConfig(
            model_id="qwen3.5:397b",
            role=ModelRole.PRIMARY,
            capabilities=["conversation", "reasoning", "vision", "code"],
            context_window=131072,
            supports_thinking=False,
            supports_vision=True,
            preferred_tasks=["chat", "analysis", "vision"],
            temperature=0.7,
        ),
        "qwen3-vl:235b": ModelConfig(
            model_id="qwen3-vl:235b",
            role=ModelRole.VISION,
            capabilities=["vision", "image_analysis", "ocr", "multimodal"],
            context_window=32768,
            supports_vision=True,
            preferred_tasks=["vision", "image", "ocr"],
            temperature=0.5,
        ),
    }

    # ROLE TOKENS in the routing template. The class keeps the historical
    # ids; __init__ builds the live TASK_ROUTING by substituting the actual
    # configured model ids for these tokens ("qwen*" entries are shared
    # fallback names and pass through untouched).
    TASK_ROUTING_TEMPLATE = {
        "chat": ["kimi-k2.5", "qwen3.5:397b", "glm-5"],
        "code": ["glm-5", "kimi-k2.5"],
        "math": ["glm-5", "kimi-k2.5"],
        "creative": ["kimi-k2.5", "qwen3.5:397b"],
        "analysis": ["glm-5", "kimi-k2.5"],
        "reasoning": ["glm-5", "kimi-k2.5"],
        "vision": ["kimi-k2.5", "qwen3-vl:235b"],
        "verification": ["glm-5"],
        "refinement": ["glm-5", "kimi-k2.5"],
        "thinking": ["glm-5"],
        "abstraction": ["kimi-k2.5"],
        "synthesis": ["glm-5"],
        "meta_reasoning": ["kimi-k2.5"],
        "pattern": ["kimi-k2.5"],
    }

    # Backward-compat alias: instances get their resolved routing; readers of
    # the class attribute see the template.
    TASK_ROUTING = TASK_ROUTING_TEMPLATE

    def __init__(
        self,
        primary_llm=None,
        backend_llm=None,
        vision_llm=None,
    ):
        self.primary_llm = primary_llm
        self.backend_llm = backend_llm or primary_llm
        self.vision_llm = vision_llm

        # Live model ids: config.yaml (llm.model / backend_llm.model) is the
        # single source of truth. All routing, registry entries and status
        # displays resolve through these. Only str values are trusted — test
        # doubles (MagicMock) have .model attributes that are not strings.
        self.primary_model_id: str = _live_model_id(primary_llm, "glm-5.3-flash")
        self.backend_model_id: str = _live_model_id(backend_llm, "kimi-k2.7-code")
        if self.backend_llm is primary_llm or self.backend_llm is None:
            # No separate backend: both roles resolve to the primary model
            self.backend_model_id = self.primary_model_id

        # Registry: substitute the live ids for the role templates
        self.models: dict[str, ModelConfig] = dict(self.DEFAULT_MODELS)
        for template_id, live_id in (
            ("kimi-k2.5", self.primary_model_id),
            ("glm-5", self.backend_model_id),
        ):
            if live_id == template_id:
                continue
            template = self.DEFAULT_MODELS[template_id]
            self.models.pop(template_id, None)
            self.models[live_id] = replace(template, model_id=live_id)

        # Live routing: same substitution over the template
        sub = {"kimi-k2.5": self.primary_model_id, "glm-5": self.backend_model_id}
        self.TASK_ROUTING: dict[str, list[str]] = {
            task: [sub.get(m, m) for m in routes]
            for task, routes in self.TASK_ROUTING_TEMPLATE.items()
        }
        # Deduplicate while preserving order (e.g. primary == backend)
        for task in self.TASK_ROUTING:
            seen: set[str] = set()
            self.TASK_ROUTING[task] = [
                m for m in self.TASK_ROUTING[task]
                if not (m in seen or seen.add(m))
            ]

        self.shared_context = SharedContext()

        self._active_model: str = self.primary_model_id
        self._handoff_queue: list[InterModelMessage] = []
        self._reasoning_chains: dict[str, ReasoningChain] = {}
        self._lock = threading.RLock()

        self._stats = {
            "total_handoffs": 0,
            "successful_handoffs": 0,
            "cross_model_calls": 0,
            "reasoning_chains": 0,
        }

        self._on_handoff_callback = None
        self._on_message_callback = None

    def set_callbacks(
        self,
        on_handoff=None,
        on_message=None,
    ) -> None:
        """Set callbacks for model coordination events."""
        self._on_handoff_callback = on_handoff
        self._on_message_callback = on_message

    def get_model_for_task(self, task_type: str) -> str:
        """Get the best model for a given task type."""
        models = self.TASK_ROUTING.get(task_type, [self.primary_model_id])
        if not models:
            return self.primary_model_id
        return models[0]
    
    def get_llm_for_model(self, model_id: str):
        """Get the LLM client for a model."""
        config = self.models.get(model_id)
        if not config:
            return self.primary_llm
        
        if config.role == ModelRole.BACKEND:
            return self.backend_llm
        elif config.role == ModelRole.VISION:
            return self.vision_llm or self.primary_llm
        return self.primary_llm
    
    async def handoff(
        self,
        to_model: str,
        context: dict[str, Any],
        query: str,
        reason: HandoffReason = HandoffReason.COLLABORATION,
        reasoning_chain_id: str | None = None,
    ) -> InterModelMessage:
        """Hand off from current model to another model.
        
        This creates a structured handoff where the receiving model
        gets the full context and can continue the work.
        """
        with self._lock:
            from_model = self._active_model
        
        message = InterModelMessage(
            id=f"msg_{int(time.time() * 1000)}",
            message_type=MessageType.HANDOFF,
            from_model=from_model,
            to_model=to_model,
            context=context,
            query=query,
            handoff_reason=reason,
        )
        
        if reasoning_chain_id:
            chain = self.shared_context.get_reasoning_chain(reasoning_chain_id)
            if chain:
                message.reasoning_chain = chain.turns.copy()
        
        self.shared_context.log_message(message)
        
        with self._lock:
            self._stats["total_handoffs"] += 1
            self._stats["successful_handoffs"] += 1  # Mark as successful
            self._active_model = to_model
        
        if self._on_handoff_callback:
            try:
                self._on_handoff_callback(message)
            except Exception as e:
                logger.error(f"Handoff callback error: {e}")
        
        logger.info(f"Model handoff: {from_model} -> {to_model} ({reason.value})")
        
        return message
    
    async def query_other_model(
        self,
        target_model: str,
        query: str,
        context: dict[str, Any] | None = None,
        include_reasoning: bool = False,
    ) -> str:
        """Query another model while staying in current model context.
        
        This is for asking another model something, not handing off.
        The response comes back to the current model.
        """
        with self._lock:
            current_model = self._active_model
        llm = self.get_llm_for_model(target_model)
        
        if not llm:
            return f"Model {target_model} not available"
        
        messages = []
        
        if context:
            shared = self.shared_context.get_all()
            if shared:
                context_str = json.dumps(shared, indent=2)[:2000]
                messages.append({
                    "role": "system",
                    "content": f"Shared context from other model:\n{context_str}",
                })
        
        messages.append({"role": "user", "content": query})
        
        try:
            with self._lock:
                self._stats["cross_model_calls"] += 1
            
            response = await llm.chat(
                user_input=query,
                system_prompt=f"You are responding to a query from another AI model ({current_model}). Be concise and helpful.",
                conversation_history=messages if len(messages) > 1 else [],
            )
            
            content = get_response_content(response) if isinstance(response, dict) else str(response)
            
            message = InterModelMessage(
                id=f"query_{int(time.time() * 1000)}",
                message_type=MessageType.QUERY,
                from_model=current_model,
                to_model=target_model,
                context=context or {},
                query=query,
                response=content,
                response_from=target_model,
            )
            self.shared_context.log_message(message)
            
            return content
            
        except Exception as e:
            logger.error(f"Cross-model query failed: {e}")
            return f"Error querying {target_model}: {safe_error_message(e)}"
    
    async def think_together(
        self,
        topic: str,
        initial_query: str,
        models: list[str] | None = None,
        rounds: int = 2,
    ) -> ReasoningChain:
        """Have multiple models think together on a topic.
        
        This creates a reasoning chain where models build on each other's
        thinking, creating a richer synthesis.
        """
        if models is None:
            models = [self.primary_model_id, self.backend_model_id]
        
        chain = self.shared_context.create_reasoning_chain(topic)
        
        with self._lock:
            self._stats["reasoning_chains"] += 1
        
        current_query = initial_query
        accumulated_context = {}
        
        for round_num in range(rounds):
            for model_id in models:
                llm = self.get_llm_for_model(model_id)
                if not llm:
                    continue
                
                model_config = self.models.get(model_id)
                
                prompt = f"""Topic: {topic}

Previous reasoning from other models:
{json.dumps(accumulated_context, indent=2) if accumulated_context else 'None - this is the first turn'}

Current question: {current_query}

Provide your reasoning and answer. {'Show your thinking process.' if model_config and model_config.supports_thinking else 'Be concise.'}"""
                
                try:
                    response = await llm.chat(
                        user_input=prompt,
                        system_prompt="You are collaborating with other AI models. Build on their insights and add your own perspective.",
                        conversation_history=[],
                    )
                    
                    content = get_response_content(response) if isinstance(response, dict) else str(response)
                    
                    thinking = get_response_thinking(response)
                    
                    chain.add_turn(
                        model=model_id,
                        thinking=thinking[:500] if thinking else "",
                        output=content,
                    )
                    
                    accumulated_context[model_id] = {
                        "round": round_num + 1,
                        "key_insight": content[:300],
                    }
                    
                    current_query = f"Building on that, what's another perspective on: {topic}"
                    
                except Exception as e:
                    logger.error(f"Think-together error for {model_id}: {e}")
        
        chain.complete = True
        
        if not self.primary_llm:
            chain.synthesis = "Synthesis unavailable - no primary model configured"
            logger.warning("Think-together synthesis failed: no primary model")
            return chain
        
        synthesis_prompt = f"""Synthesize these reasoning turns into a comprehensive answer:

{json.dumps(chain.turns, indent=2)}

Provide a clear, well-structured synthesis."""
        
        try:
            llm = self.primary_llm
            response = await llm.chat(
                user_input=synthesis_prompt,
                system_prompt="You are synthesizing insights from multiple AI models into a coherent answer.",
                conversation_history=[],
            )
            chain.synthesis = get_response_content(response) if isinstance(response, dict) else str(response)
        except Exception as e:
            chain.synthesis = "Synthesis failed"
            logger.error(f"Synthesis error: {e}")
        
        return chain
    
    def set_shared_context(self, key: str, value: Any, model_id: str = "system") -> None:
        """Set a value in shared context."""
        self.shared_context.set(key, value, model_id)
    
    def get_shared_context(self, key: str) -> Any | None:
        """Get a value from shared context."""
        return self.shared_context.get(key)
    
    def get_active_model(self) -> str:
        """Get the currently active model."""
        return self._active_model
    
    def get_connected_models(self) -> list[str]:
        """Get list of models with actual LLM connections."""
        connected = []
        if self.primary_llm is not None:
            connected.append(self._active_model)
        if self.backend_llm is not None and self.backend_llm != self.primary_llm:
            if self.backend_model_id not in connected:
                connected.append(self.backend_model_id)
        if self.vision_llm is not None and self.vision_llm != self.primary_llm:
            connected.append("qwen3-vl:235b")
        return connected

    def get_dual_model_status(self) -> dict:
        """Get detailed dual-model architecture status."""
        return {
            "enabled": self.backend_llm is not None and self.backend_llm != self.primary_llm,
            "primary": {
                "model": self.primary_model_id,
                "role": "chat, creative, user-facing, meta-reasoning, abstractions, vision",
                "connected": self.primary_llm is not None,
                "tasks": ["chat", "creative", "abstraction", "meta_reasoning", "pattern", "vision"],
            },
            "backend": {
                "model": self.backend_model_id,
                "role": "code, math, reasoning, analysis, refinement, synthesis",
                "connected": self.backend_llm is not None and self.backend_llm != self.primary_llm,
                "tasks": ["code", "math", "analysis", "reasoning", "refinement", "verification", "thinking", "synthesis"],
            },
            "routing": self.TASK_ROUTING,
            "collaboration": {
                "enabled": True,
                "cross_model_calls": self._stats.get("cross_model_calls", 0),
                "total_handoffs": self._stats.get("total_handoffs", 0),
            },
        }
    
    def get_tri_model_status(self) -> dict:
        """Alias for backward compatibility."""
        return self.get_dual_model_status()
    
    def get_stats(self) -> dict:
        """Get coordination statistics."""
        with self._lock:
            connected = self.get_connected_models()
            return {
                **self._stats,
                "active_model": self._active_model,
                "available_models": list(self.models.keys()),
                "connected_models": connected,  # Models with actual LLM clients
                "primary_connected": self.primary_llm is not None,
                "backend_connected": self.backend_llm is not None and self.backend_llm != self.primary_llm,
                "recent_messages": len(self.shared_context._message_log),
                "reasoning_chains": len(self.shared_context._reasoning_chains),
            }
    
    def register_model(self, config: ModelConfig) -> None:
        """Register a new model configuration."""
        self.models[config.model_id] = config
    
    def get_model_config(self, model_id: str) -> ModelConfig | None:
        """Get configuration for a model."""
        return self.models.get(model_id)


def get_model_coordinator(
    primary_llm=None,
    backend_llm=None,
    vision_llm=None,
) -> ModelCoordinator:
    """Get or create the global ModelCoordinator instance."""
    global _model_coordinator, _coordinator_lock
    
    if _model_coordinator is None:
        with _coordinator_lock:
            if _model_coordinator is None:
                _model_coordinator = ModelCoordinator(
                    primary_llm=primary_llm,
                    backend_llm=backend_llm,
                    vision_llm=vision_llm,
                )
    else:
        # Update existing instance with new LLMs
        with _coordinator_lock:
            if primary_llm is not None:
                _model_coordinator.primary_llm = primary_llm
            if backend_llm is not None:
                _model_coordinator.backend_llm = backend_llm
            if vision_llm is not None:
                _model_coordinator.vision_llm = vision_llm
    
    return _model_coordinator


async def handoff_to_backend(query: str, context: dict | None = None) -> str:
    """Convenience function to hand off to backend model."""
    coordinator = get_model_coordinator()
    await coordinator.handoff(
        to_model=coordinator.backend_model_id,
        context=context or {},
        query=query,
        reason=HandoffReason.SPECIALIZATION,
    )

    llm = coordinator.get_llm_for_model(coordinator.backend_model_id)
    if not llm:
        return "Backend model not available"
    
    response = await llm.chat(
        user_input=query,
        system_prompt="Continue from previous context. The primary model has handed off to you for specialized processing.",
        conversation_history=[],
    )
    
    return get_response_content(response) if isinstance(response, dict) else str(response)