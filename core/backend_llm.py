"""Backend LLM Manager for Jarvis.

Provides access to the secondary LLM (GLM-5) for:
- CLI operations
- Self-refinement tasks
- Code review and editing
- Background analysis
- Terminal/shell operations
- File operations

The backend model (GLM-5) has access to dangerous tools for code/math/automation.
The primary model (Kimi K2.5) is restricted to safe user-facing interactions.

Model Permission Tiers:
- BACKEND (GLM-5): Full access to dangerous tools (shell, terminal, files)
- PRIMARY (Kimi K2.5): Safe tools only (no shell, terminal, file deletion)

Tool Permission Model:
- GLOBAL_DENIED_TOOLS: Never allowed for ANY model (unless explicitly enabled)
- DANGEROUS_TOOLS: Only BACKEND can use (shell, terminal, delete_file, etc.)
- SAFE_TOOLS: Both PRIMARY and BACKEND can use
"""
import logging
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any

from core.config import Config
from core.helpers import get_response_content
from core.llm import OllamaClient

logger = logging.getLogger(__name__)

_backend_client: OllamaClient | None = None
_backend_lock = threading.Lock()
_primary_client: OllamaClient | None = None
_primary_lock = threading.Lock()
_concept_client: OllamaClient | None = None
_concept_lock = threading.Lock()


class ModelTier(Enum):
    """Permission tiers for model access."""
    PRIMARY = "primary"      # Restricted - Kimi K2.5 (user-facing, safe tools only)
    BACKEND = "backend"      # Full access - GLM-5 (code/math/terminal/files)


@dataclass
class ModelInfo:
    """Information about an LLM model."""
    name: str
    tier: ModelTier
    is_vision_capable: bool
    context_window: int
    allowed_tool_categories: set
    denied_tools: set
    status: str = "unknown"


# ==============================================================================
# TOOL PERMISSION CONFIGURATION
# ==============================================================================
# This defines which tools each model can use.
#
# PHILOSOPHY:
# - Permission-required tools require explicit user authorization for BOTH models
# - Permission-required categories require auth for BOTH models
# - Most tools are available to both models without restriction
# - Evolution category requires auth for both PRIMARY and BACKEND
# ==============================================================================

# Tools that require explicit user permission for BOTH models
# These are dangerous operations that always need authorization
PERMISSION_REQUIRED_TOOLS = {
    # Shell/Terminal operations - always need permission
    "shell",
    "terminal_execute",
    "execute_command",
    "run_shell",
    "run_terminal",
    "bash",
    "cmd",
    
    # File operations (destructive) - always need permission
    "delete_file",
    "file_delete",
    "remove_file",
    "rm_file",
    
    # System modifications - always need permission
    "install_package",
    "self_edit",
    "evolution",
    "modify_system",
}

# Categories that require explicit user permission for BOTH models
PERMISSION_REQUIRED_CATEGORIES = {
    "terminal",      # Terminal access - requires auth for both
    "file",          # File operations - requires auth for both
    "system",        # System operations - requires auth for both
    "evolution",     # Self-evolution - requires auth for both
}

# Backward compatibility - no globally denied tools
GLOBAL_DENIED_TOOLS = set()

# Tools denied for PRIMARY (must handoff to BACKEND for these)
PRIMARY_DENIED_TOOLS = PERMISSION_REQUIRED_TOOLS

# Backward compatibility alias
DANGEROUS_TOOLS = PERMISSION_REQUIRED_TOOLS

# Safe tool categories - BOTH models can use without auth
SAFE_CATEGORIES = {
    # Core categories
    "web",           # Web search, fetch
    "data",          # Data analysis
    "meta",          # Meta operations
    "multimodal",    # Multimodal operations
    "home",          # Home automation
    "productivity",  # Productivity tools
    "memory",        # Memory operations
    "refinement",    # Code refinement
    
    # Additional safe categories for both
    "browser",       # Browser control
    "screen",        # Screen automation
    "automation",    # Automation
    "security",      # Security tools
    "cli",           # CLI tools
    "tools",         # Tool management
    "spatial",       # Spatial operations
    "daydream",      # Daydream engine
    "analysis",      # Analysis tools
    "learning",      # Learning tools
}

# Backend gets safe categories + permission-required categories (with auth)
BACKEND_ALLOWED_CATEGORIES = SAFE_CATEGORIES | PERMISSION_REQUIRED_CATEGORIES

# Primary gets only safe categories (no terminal, file, system, evolution)
PRIMARY_ALLOWED_CATEGORIES = SAFE_CATEGORIES


def get_model_permissions(tier: ModelTier) -> tuple[set, set]:
    """Get allowed tool categories and denied tools for a model tier.
    
    Permission Model:
    - BACKEND: Can access all categories, but permission-required tools/categories
               still need user authorization at runtime
    - PRIMARY: Safe categories only, permission-required tools denied (need handoff)
    
    Returns:
        Tuple of (allowed_categories, denied_tools)
    """
    if tier == ModelTier.BACKEND:
        # Backend has access to all categories including permission-required ones
        # But permission-required tools/categories still need user authorization
        return BACKEND_ALLOWED_CATEGORIES, GLOBAL_DENIED_TOOLS
    else:
        # Primary is denied permission-required tools and permission-required categories
        return PRIMARY_ALLOWED_CATEGORIES, PRIMARY_DENIED_TOOLS | GLOBAL_DENIED_TOOLS


def check_tool_permission(tool_name: str, tool_category: str, tier: ModelTier) -> bool:
    """Check if a model tier has permission to use a tool.
    
    Args:
        tool_name: Name of the tool
        tool_category: Category of the tool
        tier: Model permission tier
        
    Returns:
        True if allowed, False if denied
    """
    allowed_categories, denied_tools = get_model_permissions(tier)
    
    if tool_name.lower() in denied_tools:
        return False
    
    if tool_category.lower() not in allowed_categories:
        return False
    
    return True


def get_primary_llm(config: Config | None = None, circuit_breaker=None) -> OllamaClient:
    """Get or create the primary LLM client (Qwen-3.5 for interactive chat).
    
    Args:
        config: Optional config override
        circuit_breaker: Optional circuit breaker for resilience
    """
    global _primary_client
    if _primary_client is None:
        with _primary_lock:
            if _primary_client is None:
                if config is None:
                    config = Config.load()
                _primary_client = OllamaClient(
                    model=config.llm.model,
                    temperature=config.llm.temperature,
                    api_key=config.llm.api_key,
                    use_cloud=config.llm.use_cloud,
                    fallback_models=config.llm.fallback_models,
                    max_tokens=config.llm.max_tokens,
                    circuit_breaker=circuit_breaker,
                )
                logger.info(f"Primary LLM initialized: {config.llm.model}")
    return _primary_client


def get_backend_llm(config: Config | None = None, circuit_breaker=None) -> OllamaClient:
    """Get or create the backend LLM client (GLM-5 for CLI, refinement, analysis).
    
    Args:
        config: Optional config override
        circuit_breaker: Optional circuit breaker for resilience
    """
    global _backend_client
    if _backend_client is None:
        with _backend_lock:
            if _backend_client is None:
                if config is None:
                    config = Config.load()
                api_key = config.backend_llm.api_key or config.llm.api_key
                _backend_client = OllamaClient(
                    model=config.backend_llm.model,
                    temperature=config.backend_llm.temperature,
                    api_key=api_key,
                    use_cloud=config.backend_llm.use_cloud,
                    fallback_models=list(config.backend_llm.fallback_models or []),
                    max_tokens=4096,
                    circuit_breaker=circuit_breaker,
                )
                logger.info(f"Backend LLM initialized: {config.backend_llm.model}")
    return _backend_client


def get_concept_llm(config: Config | None = None, circuit_breaker=None) -> OllamaClient:
    """Get or create the concept LLM client.

    In the dual-model architecture, concept tasks are handled by the backend LLM (GLM-5).
    This function is kept for backward compatibility and routes to backend_llm.
    
    Args:
        config: Optional config override
        circuit_breaker: Optional circuit breaker for resilience
    """
    return get_backend_llm(config, circuit_breaker)


def get_llm_for_task(task_type: str, config: Config | None = None) -> OllamaClient:
    """Get the appropriate LLM client for a given task type.
    
    Task types:
    - "interactive" -> Primary LLM (Kimi K2.5)
    - "chat" -> Primary LLM
    - "vision" -> Primary LLM (vision-enabled)
    - "cli" -> Backend LLM (GLM-5)
    - "refine" -> Backend LLM
    - "code_review" -> Backend LLM
    - "analysis" -> Backend LLM
    - "daydream" -> Backend LLM
    - "abstraction" -> Primary LLM (Kimi K2.5)
    - "synthesis" -> Backend LLM (GLM-5)
    - "meta_reasoning" -> Primary LLM
    - "pattern" -> Primary LLM
    - "code" -> Backend LLM
    - "math" -> Backend LLM
    - "reasoning" -> Backend LLM
    - "verification" -> Backend LLM
    - "thinking" -> Backend LLM
    """
    primary_tasks = {"abstraction", "meta_reasoning", "pattern", "vision", "chat", "creative", "interactive"}
    backend_tasks = {"cli", "refine", "code_review", "analysis", "daydream", "self_edit", "tools", "synthesis", "code", "math", "reasoning", "refinement", "verification", "thinking"}

    if task_type in primary_tasks:
        return get_primary_llm(config)
    if task_type in backend_tasks:
        return get_backend_llm(config)
    return get_primary_llm(config)


async def call_backend(
    prompt: str,
    system_prompt: str = "You are a helpful AI assistant focused on accurate, concise responses.",
    config: Config | None = None,
) -> str:
    """Convenience function for backend LLM calls.
    
    Returns the response content as a string.
    """
    llm = get_backend_llm(config)
    result = await llm.chat(user_input=prompt, system_prompt=system_prompt)
    return get_response_content(result)


async def call_primary(
    prompt: str,
    system_prompt: str = "You are a helpful AI assistant.",
    config: Config | None = None,
) -> str:
    """Convenience function for primary LLM calls.
    
    Returns the response content as a string.
    """
    llm = get_primary_llm(config)
    result = await llm.chat(user_input=prompt, system_prompt=system_prompt)
    return get_response_content(result)


async def call_backend_stream(
    prompt: str,
    system_prompt: str = "You are a helpful AI assistant focused on accurate, concise responses.",
    config: Config | None = None,
    tools: list[dict] | None = None,
    temperature: float = 0.3,
):
    """Stream backend LLM calls with thinking/tool support.
    
    Yields:
        dict with type:
        - {"type": "thinking", "text": "..."} - GLM-5 thinking content
        - {"type": "text", "text": "..."} - Response text
        - {"type": "tool_call", "name": "...", "arguments": {...}} - Tool call
        - {"type": "finish", "reason": "stop"} - Completion
    """
    llm = get_backend_llm(config)
    
    if hasattr(llm, 'chat_stream_with_tools'):
        async for chunk in llm.chat_stream_with_tools(
            user_input=prompt,
            system_prompt=system_prompt,
            tools=tools,
            temperature=temperature,
        ):
            yield chunk
    elif hasattr(llm, 'chat_stream'):
        async for chunk in llm.chat_stream(
            user_input=prompt,
            system_prompt=system_prompt,
        ):
            # chat_stream yields strings, not dicts
            if isinstance(chunk, dict):
                if chunk.get("type") == "thinking":
                    yield {"type": "thinking", "text": chunk.get("content", "")}
                elif chunk.get("type") == "content":
                    yield {"type": "text", "text": chunk.get("content", "")}
                elif chunk.get("done"):
                    yield {"type": "finish", "reason": "stop"}
            else:
                # String chunk
                yield {"type": "text", "text": str(chunk)}
    else:
        result = await llm.chat(user_input=prompt, system_prompt=system_prompt)
        content = get_response_content(result)
        
        if result.get("thinking"):
            yield {"type": "thinking", "text": result["thinking"]}
        
        yield {"type": "text", "text": content}
        yield {"type": "finish", "reason": "stop"}


def reset_clients():
    """Reset all LLM clients (useful for config changes)."""
    global _backend_client, _primary_client, _concept_client
    with _backend_lock:
        _backend_client = None
    with _primary_lock:
        _primary_client = None
    with _concept_lock:
        _concept_client = None


def set_circuit_breakers(primary_breaker=None, backend_breaker=None, concept_breaker=None):
    """Set circuit breakers for existing LLM clients.
    
    This allows setting circuit breakers after clients are initialized.
    """
    global _primary_client, _backend_client, _concept_client
    
    if _primary_client and primary_breaker:
        _primary_client._circuit_breaker = primary_breaker
        logger.info("Circuit breaker attached to primary LLM")
    
    if _backend_client and backend_breaker:
        _backend_client._circuit_breaker = backend_breaker
        logger.info("Circuit breaker attached to backend LLM")
    
    if _concept_client and concept_breaker:
        _concept_client._circuit_breaker = concept_breaker
        logger.info("Circuit breaker attached to concept LLM")


def get_model_status() -> dict[str, Any]:
    """Get status information about all models.
    
    Returns:
        Dictionary with primary, backend, and concept model info
    """
    config = Config.load()
    
    # Get permissions for each tier
    primary_cats, primary_denied = get_model_permissions(ModelTier.PRIMARY)
    backend_cats, backend_denied = get_model_permissions(ModelTier.BACKEND)
    
    status = {
        "primary": {
            "model": config.llm.model,
            "tier": ModelTier.PRIMARY.value,
            "vision_capable": True,
            "context_window": 128000,
            "use_cloud": config.llm.use_cloud,
            "allowed_categories": list(primary_cats),
            "denied_tools": list(primary_denied),
            "initialized": _primary_client is not None,
        },
        "backend": {
            "model": config.backend_llm.model,
            "tier": ModelTier.BACKEND.value,
            "vision_capable": False,
            "context_window": 131072,
            "use_cloud": config.backend_llm.use_cloud,
            "allowed_categories": list(backend_cats),
            "denied_tools": list(backend_denied),
            "initialized": _backend_client is not None,
        },
        "routing": {
            "backend_tasks": [
                "cli", "refine", "code_review", "analysis", 
                "daydream", "self_edit", "tools", "terminal", "shell"
            ],
            "primary_tasks": [
                "interactive", "chat", "vision", "home", "web"
            ],
            "concept_tasks": [
                "abstraction", "synthesis", "meta_reasoning", "pattern"
            ],
        },
    }
    
    if hasattr(config, 'concept_llm') and config.concept_llm.enabled:
        status["concept"] = {
            "model": config.concept_llm.model,
            "tier": "concept",
            "vision_capable": False,
            "context_window": 131072,
            "use_cloud": config.concept_llm.use_cloud,
            "capabilities": ["meta_reasoning", "abstraction", "synthesis", "pattern_recognition"],
            "initialized": _concept_client is not None,
        }
    
    return status


def get_tier_for_task(task_type: str) -> ModelTier:
    """Get the model tier for a given task type.
    
    Args:
        task_type: The type of task being performed
        
    Returns:
        ModelTier.PRIMARY or ModelTier.BACKEND
    """
    backend_tasks = {"cli", "refine", "code_review", "analysis", 
                     "daydream", "self_edit", "tools"}
    
    if task_type in backend_tasks:
        return ModelTier.BACKEND
    return ModelTier.PRIMARY


def get_current_model_name(tier: ModelTier) -> str:
    """Get the model name for a tier.
    
    Args:
        tier: Model tier
        
    Returns:
        Model name string
    """
    config = Config.load()
    if tier == ModelTier.BACKEND:
        return config.backend_llm.model
    return config.llm.model
