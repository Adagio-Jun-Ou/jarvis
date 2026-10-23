"""Context Manager - Integration layer for Jarvis context systems.

This module provides a unified interface for:
1. Context compression (automatic when approaching token limits)
2. Sub-agent orchestration (for complex multi-step requests)
3. Integration with Jarvis's existing memory system
4. Bidirectional module sensing via ModuleBridge
5. Self-evaluation integration for ethical awareness
6. Checkpoint permissioning for frontier domains
7. Cross-stack context propagation

USAGE IN JARVIS:
    # In agent.py initialization:
    from core.context_manager import init_context_manager
    
    init_context_manager(llm_client=self.llm, agent=self)
    
    # Before API calls:
    from core.context_manager import get_optimized_context
    
    messages = get_optimized_context(messages, max_tokens=16000)
    
    # For complex requests:
    from core.context_manager import should_use_subagents, execute_with_subagents
    
    if should_use_subagents(request):
        result = await execute_with_subagents(request, context)
        
    # Check module status (bidirectional sensing):
    from core.context_manager import get_module_status
    
    status = get_module_status()  # Returns health of all connected modules
    
    # Set subsystem context (for LLM call tracking):
    from core.context_manager import subsystem_context
    
    async with subsystem_context("Erdős"):
        result = await llm.chat(...)  # Automatically tracked
"""

import contextvars
import logging
import threading
from core.helpers import safe_error_message
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

logger = logging.getLogger(__name__)

_current_subsystem: contextvars.ContextVar[str] = contextvars.ContextVar("current_subsystem", default="Chat")
_current_user_intent: contextvars.ContextVar[str] = contextvars.ContextVar("current_user_intent", default="")
_current_context: contextvars.ContextVar[dict] = contextvars.ContextVar("current_context", default=None)
_current_model: contextvars.ContextVar[str] = contextvars.ContextVar("current_model", default="kimi-k2.5")

_context_compressor = None
_sub_agent_orchestrator = None
_self_eval_engine = None
_checkpoint_manager = None
_model_coordinator = None
_initialized = False
_lock = threading.Lock()

DEFAULT_ACTIVE_TOKEN_LIMIT = 12000
DEFAULT_MAX_TOKENS = 16000
SUBAGENT_THRESHOLD = 3


@dataclass
class RequestContext:
    """Context about a request for decision making."""
    request: str
    estimated_tokens: int
    detected_tasks: int
    use_subagents: bool
    needs_compression: bool
    complexity_score: float
    reason: str = ""


@dataclass
class ContextSnapshot:
    """Snapshot of current context state."""
    subsystem: str
    user_intent: str
    conversation_turn: int
    active_tools: list[str]
    ethical_score: float
    pending_checkpoints: int
    timestamp: datetime


def init_context_manager(
    llm_client=None,
    agent=None,
    active_token_limit: int = DEFAULT_ACTIVE_TOKEN_LIMIT,
    enable_self_eval: bool = True,
    enable_checkpoints: bool = True,
) -> bool:
    """Initialize the context management system.
    
    Call this once during Jarvis startup.
    
    Args:
        llm_client: The LLM client for summarization
        agent: The JarvisAgent instance (for tool access)
        active_token_limit: Token limit for active context
        enable_self_eval: Enable self-evaluation engine
        enable_checkpoints: Enable checkpoint permissioning
    
    Returns:
        True if initialization successful
    """
    global _context_compressor, _sub_agent_orchestrator, _self_eval_engine, _checkpoint_manager, _initialized

    with _lock:
        if _initialized:
            logger.debug("Context manager already initialized")
            return True

        try:
            from core.context_compressor import get_context_compressor

            _context_compressor = get_context_compressor(
                budget=active_token_limit,
            )

            tools = {}
            if agent and hasattr(agent, 'tools'):
                tools = agent.tools

            from core.sub_agent import get_orchestrator
            _sub_agent_orchestrator = get_orchestrator(
                llm_client=llm_client,
                tools=tools,
            )

            if enable_self_eval:
                try:
                    from core.self_evaluation import get_self_evaluation_engine
                    _self_eval_engine = get_self_evaluation_engine()
                    if llm_client:
                        _self_eval_engine.llm = llm_client
                    logger.info("Self-evaluation engine enabled")
                except Exception as e:
                    logger.debug(f"Self-evaluation not available: {e}")

            if enable_checkpoints:
                try:
                    from core.checkpoint_permission import get_checkpoint_manager
                    _checkpoint_manager = get_checkpoint_manager()
                    if llm_client:
                        _checkpoint_manager.llm = llm_client
                    logger.info("Checkpoint permission manager enabled")
                except Exception as e:
                    logger.debug(f"Checkpoint manager not available: {e}")

            from core.module_bridge import connect_modules
            connect_modules()

            _initialized = True
            logger.info("Context manager initialized successfully")
            return True

        except Exception as e:
            logger.warning(f"Failed to initialize context manager: {e}")
            return False


def analyze_request(
    request: str,
    messages: list[dict] | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> RequestContext:
    """Analyze a request to determine the best handling strategy.
    
    Args:
        request: The user's request
        messages: Current conversation context
        max_tokens: Token limit for context
    
    Returns:
        RequestContext with recommendations
    """
    global _context_compressor

    # Estimate tokens
    request_tokens = len(request) // 4
    context_tokens = 0

    if messages:
        from core.context_compressor import TokenCounter
        context_tokens = TokenCounter.count_messages(messages)

    total_tokens = request_tokens + context_tokens

    # Detect task complexity
    task_count = _detect_task_count(request)

    # Check if compression needed
    needs_compression = total_tokens > max_tokens * 0.85

    # Decide on sub-agents
    use_subagents = task_count >= SUBAGENT_THRESHOLD

    # Calculate complexity
    complexity_score = 0.0
    indicators = [
        (" and then ", 0.2),
        (" also ", 0.15),
        (" first ", 0.1),
        (" after that ", 0.15),
        (" in parallel ", 0.2),
        (" multiple ", 0.1),
        (" each ", 0.1),
        (" all ", 0.1),
        (" simultaneously ", 0.2),
        (" comprehensive ", 0.15),
    ]

    request_lower = request.lower()
    for indicator, score in indicators:
        if indicator in request_lower:
            complexity_score += score

    complexity_score = min(1.0, complexity_score)

    # Build reason
    reasons = []
    if needs_compression:
        reasons.append(f"context at {total_tokens}/{max_tokens} tokens")
    if use_subagents:
        reasons.append(f"{task_count} tasks detected")

    return RequestContext(
        request=request,
        estimated_tokens=total_tokens,
        detected_tasks=task_count,
        use_subagents=use_subagents,
        needs_compression=needs_compression,
        complexity_score=complexity_score,
        reason="; ".join(reasons) if reasons else "simple request",
    )


def _detect_task_count(request: str) -> int:
    """Detect number of tasks in a request."""
    request_lower = request.lower()

    # Count conjunctions that suggest multiple tasks
    count = 1  # At least one task

    separators = [
        " and then ", " then ", " after that ",
        " also ", " additionally ", " moreover ",
        " first ", " second ", " third ",
        " step 1", " step 2", " step 3",
    ]

    for sep in separators:
        count += request_lower.count(sep)

    # Detect numbered lists
    import re
    numbered = re.findall(r'\b\d+[\).\]]', request)
    count = max(count, len(numbered))

    # Cap at reasonable number
    return min(count, 10)


def get_optimized_context(
    messages: list[dict],
    query: str | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    model: str | None = None,
) -> list[dict]:
    """Get optimized context for LLM API call.
    
    This is the main entry point for context optimization.
    It will compress if needed, check for proactive summarization,
    and retrieve relevant memories.
    
    Args:
        messages: Current conversation messages
        query: Optional search query for memory retrieval
        max_tokens: Maximum tokens to include
        model: Model name for model-specific thresholds
    
    Returns:
        Optimized list of messages
    """
    global _context_compressor, _initialized

    if not _initialized or _context_compressor is None:
        return _simple_truncate(messages, max_tokens)

    try:
        from core.tokens import get_token_counter

        counter = get_token_counter(model)
        total_tokens = counter.count_messages(messages)

        if total_tokens <= counter.compression_threshold_tokens:
            if counter.needs_proactive_summarize(messages):
                result = _context_compressor.proactive_summarize(model)
                if result.get("summarized", 0) > 0:
                    logger.info(f"Proactive summarization: {result}")
            return messages

        for msg in messages:
            _context_compressor.add_message(
                role=msg.get("role", "user"),
                content=msg.get("content", ""),
            )

        optimized = _context_compressor.get_messages_for_llm(max_tokens=max_tokens)
        _context_compressor.clear(keep_important=True)

        return optimized

    except Exception as e:
        logger.warning(f"Context optimization failed: {e}")
        return _simple_truncate(messages, max_tokens)


def check_context_status(messages: list[dict], model: str | None = None) -> dict:
    """Check context status including compression and summarization needs.
    
    Args:
        messages: Current conversation messages
        model: Model name for thresholds
    
    Returns:
        Dict with context status information
    """
    from core.tokens import get_token_counter
    
    counter = get_token_counter(model)
    total_tokens = counter.count_messages(messages)
    compression_threshold = counter.compression_threshold_tokens
    proactive_threshold = counter.proactive_summarize_tokens
    
    return {
        "total_tokens": total_tokens,
        "max_tokens": counter.max_context,
        "compression_threshold": compression_threshold,
        "proactive_threshold": proactive_threshold,
        "usage_ratio": total_tokens / counter.max_context,
        "needs_compression": total_tokens >= compression_threshold,
        "needs_proactive_summarize": total_tokens >= proactive_threshold,
        "available_tokens": counter.max_context - counter.reserved_output - total_tokens,
    }


def _simple_truncate(messages: list[dict], max_tokens: int) -> list[dict]:
    """Simple truncation fallback."""
    from core.context_compressor import TokenCounter

    # Keep last messages that fit
    result = []
    total = 0

    for msg in reversed(messages):
        msg_tokens = TokenCounter.count_messages([msg])
        if total + msg_tokens <= max_tokens:
            result.insert(0, msg)
            total += msg_tokens
        else:
            break

    return result


async def execute_with_subagents(
    request: str,
    context: dict | None = None,
    progress_callback: Callable[[str, float, dict], None] | None = None,
) -> dict:
    """Execute a complex request using sub-agents.
    
    Args:
        request: The request to execute
        context: Optional context (files, data, etc.)
        progress_callback: Optional callback for progress updates
    
    Returns:
        Dict with:
        - success: bool
        - final_answer: str
        - tasks: dict of task statuses
        - duration: float
    """
    global _sub_agent_orchestrator, _initialized

    if not _initialized or _sub_agent_orchestrator is None:
        return {
            "success": False,
            "final_answer": "Sub-agent system not initialized",
            "tasks": {},
            "duration": 0.0,
        }

    try:
        result = await _sub_agent_orchestrator.execute(
            objective=request,
            context=context,
            mode="auto",
        )
        return result
    except Exception as e:
        logger.error(f"Sub-agent execution failed: {e}")
        return {
            "success": False,
            "final_answer": f"Execution error: {safe_error_message(e)}",
            "tasks": {},
            "duration": 0.0,
        }


def should_use_subagents(request: str) -> bool:
    """Quick check if sub-agents should be used for a request."""
    analysis = analyze_request(request)
    return analysis.use_subagents


def store_interaction(
    user_input: str,
    assistant_response: str,
    importance: float | None = None,
):
    """Store an interaction for future context.
    
    Call this after each turn to build the memory.
    """
    global _context_compressor

    if _context_compressor is None:
        return

    try:
        from core.context_compressor import ImportanceScorer

        if importance is None:
            importance = ImportanceScorer.score(user_input)

        _context_compressor.add_message(
            role="user",
            content=user_input,
            importance=importance,
        )
        _context_compressor.add_message(
            role="assistant",
            content=assistant_response,
            importance=importance * 0.8,  # Slightly lower for responses
        )
    except Exception as e:
        logger.debug(f"Failed to store interaction: {e}")


def get_context_stats() -> dict:
    """Get context management statistics."""
    global _context_compressor, _sub_agent_orchestrator, _initialized

    stats = {
        "initialized": _initialized,
        "compressor": None,
        "orchestrator": None,
    }

    if _context_compressor:
        stats["compressor"] = _context_compressor.get_stats()

    if _sub_agent_orchestrator:
        stats["orchestrator"] = _sub_agent_orchestrator.get_stats()

    return stats


def reset_context():
    """Reset context for new conversation."""
    global _context_compressor

    if _context_compressor:
        _context_compressor.clear(keep_important=False)


# Convenience function for backward compatibility
def compress_if_needed(
    messages: list[dict],
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> tuple[list[dict], bool]:
    """Compress messages if needed.
    
    Returns:
        Tuple of (compressed_messages, was_compressed)
    """
    optimized = get_optimized_context(messages, max_tokens=max_tokens)
    was_compressed = len(optimized) < len(messages)
    return optimized, was_compressed


def get_module_status() -> dict:
    """Get status of all connected modules via bridge.
    
    Returns status from:
    - ContextCompressor
    - SubAgentOrchestrator
    - ModuleBridge events and pending operations
    """
    try:
        from core.module_bridge import get_bridge
        bridge = get_bridge()

        return {
            "modules": bridge.get_all_status(),
            "pending": bridge.get_pending_operations(),
            "recent_events": bridge.get_event_history(limit=10),
            "memory_pressure": bridge.is_memory_pressure(),
            "can_delegate": bridge.can_delegate_tasks(),
        }
    except Exception as e:
        logger.debug(f"Failed to get module status: {e}")
        return {
            "modules": {},
            "pending": {},
            "error": safe_error_message(e),
        }


def request_compression_from_subagent(reason: str = "sub-agent request") -> bool:
    """Allow sub-agent to request context compression.
    
    Called when sub-agent needs more context space for its work.
    """
    try:
        from core.module_bridge import get_bridge
        bridge = get_bridge()
        return bridge.request_compression("SubAgentOrchestrator", reason)
    except Exception as e:
        logger.debug(f"Failed to request compression: {e}")
        return False


def connect_bridge() -> bool:
    """Explicitly connect all modules via the bridge.
    
    Call this after all modules are initialized to enable
    bidirectional sensing.
    """
    try:
        from core.module_bridge import connect_modules
        connect_modules()
        logger.info("Module bridge connected for bidirectional sensing")
        return True
    except Exception as e:
        logger.warning(f"Failed to connect bridge: {e}")
        return False


# Initialize lazily on first use
def _ensure_initialized():
    """Ensure context manager is initialized."""
    global _initialized

    if not _initialized:
        logger.warning("Context manager not initialized - using defaults")
        # Try to initialize with defaults
        init_context_manager()


# Auto-initialize when module is imported if possible
pass


# ============================================================================
# CONTEXT PROPAGATION API - Cross-stack context awareness
# ============================================================================

def set_subsystem(name: str):
    """Set the current subsystem context for LLM call tracking."""
    _current_subsystem.set(name)


def get_subsystem() -> str:
    """Get the current subsystem context."""
    return _current_subsystem.get()


def set_user_intent(intent: str):
    """Set the current user intent/goal."""
    _current_user_intent.set(intent)


def get_user_intent() -> str:
    """Get the current user intent."""
    return _current_user_intent.get()


def add_context(key: str, value: Any):
    """Add context data."""
    ctx = _current_context.get() or {}
    ctx[key] = value
    _current_context.set(ctx)


def get_context(key: str = None, default: Any = None) -> Any:
    """Get context data."""
    ctx = _current_context.get() or {}
    if key is None:
        return ctx
    return ctx.get(key, default)


def clear_context():
    """Clear current context."""
    _current_context.set({})


class subsystem_context:
    """Context manager for setting subsystem during LLM calls.
    
    Integrates with:
    - Self-evaluation (tracks calls per subsystem)
    - Checkpoint permissioning (detects frontier domains)
    - Memory hierarchy (organizes by subsystem)
    
    Usage:
        async with subsystem_context("Erdős"):
            result = await llm.chat(...)  # Automatically tracked
    """
    
    def __init__(self, name: str, user_intent: str = ""):
        self.name = name
        self.user_intent = user_intent
        self._token = None
        self._intent_token = None
        self._prev_subsystem = None
        self._prev_intent = None
    
    def __enter__(self):
        self._prev_subsystem = _current_subsystem.get()
        self._prev_intent = _current_user_intent.get()
        self._token = _current_subsystem.set(self.name)
        if self.user_intent:
            self._intent_token = _current_user_intent.set(self.user_intent)
        return self
    
    def __exit__(self, *args):
        if self._token:
            _current_subsystem.reset(self._token)
        if self._intent_token:
            _current_user_intent.reset(self._intent_token)
    
    async def __aenter__(self):
        return self.__enter__()
    
    async def __aexit__(self, *args):
        self.__exit__(*args)


def ctx(subsystem: str, user_intent: str = "") -> subsystem_context:
    """Shorthand for subsystem_context."""
    return subsystem_context(subsystem, user_intent)


# ============================================================================
# SELF-EVALUATION API
# ============================================================================

def record_output_for_eval(
    output: str,
    input_preview: str = "",
    subsystem: str = "",
    action_type: str = "response",
) -> dict:
    """Record output for self-evaluation.
    
    Returns evaluation result if available.
    """
    if not _self_eval_engine:
        return {"evaluated": False, "reason": "self-eval not enabled"}
    
    from core.self_evaluation import ActionType
    
    try:
        action_enum = ActionType(action_type)
    except ValueError:
        action_enum = ActionType.RESPONSE
    
    record = _self_eval_engine.evaluate_action(
        action_type=action_enum,
        description=f"Output from {subsystem or get_subsystem()}",
        input_preview=input_preview[:500],
        output_preview=output[:500],
        subsystem=subsystem or get_subsystem(),
    )
    
    return {
        "evaluated": True,
        "score": record.overall_score,
        "flags": record.flags,
        "id": record.id,
    }


def get_ethical_score() -> float:
    """Get current ethical alignment score."""
    if not _self_eval_engine:
        return 1.0
    status = _self_eval_engine.get_status()
    return status.get("overall_score", 1.0)


def get_self_eval_status() -> dict:
    """Get self-evaluation engine status."""
    if not _self_eval_engine:
        return {"enabled": False}
    return {"enabled": True, **_self_eval_engine.get_status()}


# ============================================================================
# CHECKPOINT PERMISSION API
# ============================================================================

def check_frontier(operation: str, context: dict = None) -> dict:
    """Check if operation enters a frontier domain.
    
    Returns checkpoint info if needed, else empty dict.
    """
    if not _checkpoint_manager:
        return {"requires_checkpoint": False}
    
    from core.checkpoint_permission import detect_frontier_domain
    
    domain = detect_frontier_domain(operation, context or {})
    
    if domain:
        return {
            "requires_checkpoint": True,
            "domain": domain.value,
            "operation": operation,
        }
    
    return {"requires_checkpoint": False}


async def request_checkpoint(
    domain: str,
    operation: str,
    context: dict = None,
    impact_preview: str = "",
) -> bool:
    """Request checkpoint permission for frontier operation.
    
    Returns True if approved, False if denied.
    """
    if not _checkpoint_manager:
        return True
    
    from core.checkpoint_permission import FrontierDomain
    
    try:
        domain_enum = FrontierDomain(domain)
    except ValueError:
        return True
    
    checkpoint = await _checkpoint_manager.request_permission(
        domain=domain_enum,
        operation=operation,
        context=context or {},
        impact_preview=impact_preview,
    )
    
    return checkpoint.status.value in ["approved", "auto_approved"]


def get_pending_checkpoints() -> list[dict]:
    """Get all pending checkpoints requiring user response."""
    if not _checkpoint_manager:
        return []
    return _checkpoint_manager.get_pending_checkpoints()


def respond_to_checkpoint(checkpoint_id: str, approved: bool, reason: str = "") -> bool:
    """Respond to a pending checkpoint."""
    if not _checkpoint_manager:
        return False
    return _checkpoint_manager.respond_to_checkpoint(checkpoint_id, approved, reason)


class frontier_checkpoint:
    """Context manager for operations in frontier domains.
    
    Automatically checks for frontier domain and requests permission if needed.
    
    Usage:
        async with frontier_checkpoint("send email", {"to": "user@example.com"}) as cp:
            if cp.is_approved():
                # Proceed with operation
                send_email(...)
    """
    
    def __init__(self, operation: str, context: dict = None, impact: str = ""):
        self.operation = operation
        self.context = context or {}
        self.impact = impact
        self.approved = False
        self._domain = None
    
    async def __aenter__(self):
        check = check_frontier(self.operation, self.context)
        
        if check.get("requires_checkpoint"):
            self._domain = check.get("domain")
            self.approved = await request_checkpoint(
                domain=self._domain,
                operation=self.operation,
                context=self.context,
                impact_preview=self.impact,
            )
        else:
            self.approved = True
        
        return self
    
    async def __aexit__(self, *args):
        pass
    
    def is_approved(self) -> bool:
        return self.approved
    
    def get_domain(self) -> str:
        return self._domain


# ============================================================================
# INTEGRATED STATUS API
# ============================================================================

def get_full_context_status() -> dict:
    """Get comprehensive status of all context systems."""
    status = {
        "initialized": _initialized,
        "subsystem": get_subsystem(),
        "user_intent": get_user_intent(),
        "current_model": get_current_model(),
        "ethical_score": get_ethical_score(),
        "pending_checkpoints": len(get_pending_checkpoints()),
        "memory": get_context_stats(),
        "self_eval": get_self_eval_status(),
        "checkpoint_manager": _get_checkpoint_status(),
        "model_coordinator": get_model_coordinator_status(),
        "timestamp": datetime.now().isoformat(),
    }
    return status


def _get_checkpoint_status() -> dict:
    """Safely get checkpoint manager status."""
    if _checkpoint_manager is None:
        return {"enabled": False, "stats": {}}
    try:
        return {
            "enabled": True,
            "stats": _checkpoint_manager.get_stats() if hasattr(_checkpoint_manager, 'get_stats') else {}
        }
    except Exception as e:
        logger.debug(f"Error getting checkpoint stats: {e}")
        return {"enabled": True, "stats": {}, "error": safe_error_message(e)}


def get_current_model() -> str:
    """Get the current active model."""
    return _current_model.get()


def set_current_model(model: str) -> None:
    """Set the current active model."""
    _current_model.set(model)


def get_model_coordinator_status() -> dict:
    """Get model coordinator status."""
    if _model_coordinator is None:
        return {"enabled": False}
    return {"enabled": True, **_model_coordinator.get_stats()}


def init_model_coordinator(primary_llm=None, backend_llm=None, memory=None) -> None:
    """Initialize the model coordinator."""
    global _model_coordinator
    try:
        from core.model_coordinator import get_model_coordinator
        _model_coordinator = get_model_coordinator(
            primary_llm=primary_llm,
            backend_llm=backend_llm,
        )
        if memory:
            _model_coordinator.shared_context.attach_memory(memory)
        logger.info("Model coordinator initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize model coordinator: {e}")


def share_context_to_models(key: str, value: Any) -> None:
    """Share a context value with all models."""
    if _model_coordinator:
        _model_coordinator.set_shared_context(key, value, get_subsystem())


def get_shared_context(key: str) -> Any:
    """Get a value from the shared context."""
    if _model_coordinator:
        return _model_coordinator.get_shared_context(key)
    return None


def get_context_snapshot() -> ContextSnapshot:
    """Get a snapshot of current context state."""
    return ContextSnapshot(
        subsystem=get_subsystem(),
        user_intent=get_user_intent(),
        conversation_turn=len(get_context("conversation_history", [])),
        active_tools=get_context("active_tools", []),
        ethical_score=get_ethical_score(),
        pending_checkpoints=len(get_pending_checkpoints()),
        timestamp=datetime.now(),
    )
