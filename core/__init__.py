"""Jarvis Core Module.

Public names are resolved lazily (PEP 562). Importing ``core`` previously pulled
in the entire agent stack eagerly — roughly 850ms — and every consumer paid it,
even one that only wanted a leaf utility like ``core.helpers.safe_error_message``
(0.013ms of actual work). Attribute access now imports only the submodule that
provides the requested name.

Behaviour is unchanged for callers: ``from core import JarvisAgent``,
``core.NPU_AVAILABLE``, ``from core import *`` and ``core.agent`` all work as
before. The optional-subsystem ``*_AVAILABLE`` flags are still booleans, and
names from a subsystem that fails to import are still ``None``. The work simply
happens on first use instead of at import time.
"""
from importlib import import_module

# --- Always-available exports: public name -> providing submodule -----------

_CORE_EXPORTS: dict[str, str] = {
    "JarvisAgent": ".agent",
    "OllamaClient": ".llm",
    "ConversationMemory": ".memory",
    "PermissionManager": ".permissions",
    "Config": ".config",
    # thinking
    "ThinkingEngine": ".thinking",
    "ThinkingMode": ".thinking",
    "Thought": ".thinking",
    "ConversationContext": ".thinking",
    # prompts
    "PromptManager": ".prompts",
    "get_base_prompt": ".prompts",
    "get_thinking_prompt": ".prompts",
    "create_prompt_manager_with_awareness": ".prompts",
    # awareness
    "AwarenessManager": ".awareness",
    "AwarenessLevel": ".awareness",
    # self awareness
    "SelfAwarenessEngine": ".self_awareness",
    "create_self_awareness": ".self_awareness",
    "ModelIdentity": ".self_awareness",
    "EnvironmentContext": ".self_awareness",
    "SessionState": ".self_awareness",
    # hardware
    "HardwareDetector": ".hardware",
    "GPUAccelerator": ".hardware",
    "NPUAccelerator": ".hardware",
    "MemoryPoolManager": ".hardware",
    "get_hardware_detector": ".hardware",
    "get_gpu_accelerator": ".hardware",
    "get_memory_pool": ".hardware",
    # accelerator
}

# Names provided by two submodules. The eager module imported .npu_optimizations
# after .accelerator, so the former won whenever it was importable; that order is
# preserved here. When it is not importable we now fall back to .accelerator
# rather than binding None, which is what the old except-branch did to a
# perfectly usable class.
# (All two-provider names were provided by deleted dead modules.)
_SHADOWED_EXPORTS: dict[str, tuple[str, ...]] = {}


def _gpu_unavailable() -> bool:
    """Fallback for is_gpu_available when GPU simulation cannot be imported."""
    return False


def _npu_unavailable() -> bool:
    """Fallback for is_npu_available when the NPU accelerator is unavailable."""
    return False


# --- Optional subsystems ----------------------------------------------------
#
# flag -> (submodule, ((public_name, source_name), ...), {public_name: fallback})
# Anything omitted from the fallback map degrades to None, matching the old
# try/except blocks.

_OPTIONAL_GROUPS: dict[str, tuple[str, tuple[tuple[str, str], ...], dict]] = {
    "GPU_SIMULATION_AVAILABLE": (
        ".gpu_simulation",
        (
            ("GPUSimulationEngine", "GPUSimulationEngine"),
            ("GPUBackend", "GPUBackend"),
            ("GPUInfo", "GPUInfo"),
            ("SimulationStats", "SimulationStats"),
            ("get_gpu_simulation", "get_gpu_simulation"),
            ("is_gpu_available", "is_gpu_available"),
            ("get_gpu_backend", "get_gpu_backend"),
        ),
        {"is_gpu_available": _gpu_unavailable},
    ),
    "NPU_AVAILABLE": (
        ".npu_accelerator",
        (
            ("XDNAAccelerator", "NPUAccelerator"),
            ("NPUInfo", "NPUInfo"),
            ("XDNAController", "XDNAController"),
            ("get_npu", "get_npu"),
            ("is_npu_available", "is_npu_available"),
        ),
        {"is_npu_available": _npu_unavailable},
    ),
    "NPU_MODELS_AVAILABLE": (
        ".npu_models",
        (
            ("FastEmbeddingModel", "FastEmbeddingModel"),
            ("FastVoiceTokenizer", "FastVoiceTokenizer"),
            ("FileRetriever", "FileRetriever"),
            ("EmbeddingResult", "EmbeddingResult"),
            ("VoiceTokenResult", "VoiceTokenResult"),
            ("get_embedding", "get_embedding"),
            ("get_voice", "get_voice"),
            ("get_retriever", "get_retriever"),
        ),
        {},
    ),
    "NPU_VOICE_AVAILABLE": (
        ".npu_voice",
        (
            ("NPUVoiceTokenizer", "NPUVoiceTokenizer"),
            ("NPUVoiceProcessor", "NPUVoiceProcessor"),
            ("NPUVoiceResult", "NPUVoiceResult"),
            ("get_voice_tokenizer", "get_voice_tokenizer"),
            ("get_voice_processor", "get_voice_processor"),
        ),
        {},
    ),
    "WORKFLOW_SHELL_AVAILABLE": (
        ".workflow_shell",
        (
            ("WorkflowShell", "WorkflowShell"),
            ("Pipeline", "Pipeline"),
            ("SequentialPipeline", "SequentialPipeline"),
            ("ParallelPipeline", "ParallelPipeline"),
            ("Stage", "Stage"),
            ("FunctionStage", "FunctionStage"),
            ("get_workflow_shell", "get_workflow_shell"),
        ),
        {},
    ),
    "SUB_AGENT_COMM_AVAILABLE": (
        ".sub_agent_comm",
        (
            ("SubAgentCommunication", "SubAgentCommunication"),
            ("SessionManager", "SessionManager"),
            ("Message", "Message"),
            ("MessageType", "MessageType"),
            ("get_sub_agent_comm", "get_sub_agent_comm"),
        ),
        {},
    ),
    "SKILLS_REGISTRY_AVAILABLE": (
        ".skills_registry",
        (
            ("SkillsRegistry", "SkillsRegistry"),
            ("Skill", "Skill"),
            ("SkillType", "SkillType"),
            ("get_skills_registry", "get_skills_registry"),
        ),
        {},
    ),
}

# Reverse index: public name -> owning flag. Built without importing anything.
_OPTIONAL_OWNER: dict[str, str] = {
    public: flag
    for flag, (_module, _pairs, _fallbacks) in _OPTIONAL_GROUPS.items()
    for public, _source in _pairs
}


def _load_optional_group(flag: str) -> None:
    """Import an optional subsystem and bind its names (or their fallbacks)."""
    module_name, pairs, fallbacks = _OPTIONAL_GROUPS[flag]
    namespace = globals()
    try:
        module = import_module(module_name, __name__)
    except ImportError:
        namespace[flag] = False
        for public, _source in pairs:
            namespace[public] = fallbacks.get(public, None)
    else:
        namespace[flag] = True
        for public, source in pairs:
            namespace[public] = getattr(module, source, fallbacks.get(public, None))


def __getattr__(name: str):
    """Resolve a public name on first access (PEP 562)."""
    namespace = globals()

    if name in _SHADOWED_EXPORTS:
        for module_name in _SHADOWED_EXPORTS[name]:
            try:
                value = getattr(import_module(module_name, __name__), name)
            except (ImportError, AttributeError):
                continue
            namespace[name] = value
            return value
        namespace[name] = None
        return None

    if name in _CORE_EXPORTS:
        value = getattr(import_module(_CORE_EXPORTS[name], __name__), name)
        namespace[name] = value
        return value

    flag = name if name in _OPTIONAL_GROUPS else _OPTIONAL_OWNER.get(name)
    if flag is not None:
        _load_optional_group(flag)
        return namespace[name]

    # Fall back to treating the attribute as a submodule, so that `core.agent`
    # resolves without a preceding `import core.agent`.
    if not name.startswith("_"):
        try:
            return import_module(f".{name}", __name__)
        except ImportError:
            pass

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(
        set(globals())
        | set(_CORE_EXPORTS)
        | set(_SHADOWED_EXPORTS)
        | set(_OPTIONAL_GROUPS)
        | set(_OPTIONAL_OWNER)
    )


__all__ = [
    "JarvisAgent",
    "OllamaClient",
    "ConversationMemory",
    "PermissionManager",
    "Config",
    "ThinkingEngine",
    "ThinkingMode",
    "Thought",
    "ConversationContext",
    "PromptManager",
    "get_base_prompt",
    "get_thinking_prompt",
    "create_prompt_manager_with_awareness",
    "AwarenessManager",
    "AwarenessLevel",
    "SelfAwarenessEngine",
    "create_self_awareness",
    "ModelIdentity",
    "EnvironmentContext",
    "SessionState",
    "HardwareDetector",
    "GPUAccelerator",
    "NPUAccelerator",
    "MemoryPoolManager",
    "get_hardware_detector",
    "get_gpu_accelerator",
    "get_memory_pool",
    "GPUSimulationEngine",
    "GPUBackend",
    "GPUInfo",
    "SimulationStats",
    "get_gpu_simulation",
    "is_gpu_available",
    "get_gpu_backend",
    "GPU_SIMULATION_AVAILABLE",
    "XDNAAccelerator",
    "NPUInfo",
    "XDNAController",
    "get_npu",
    "is_npu_available",
    "NPU_AVAILABLE",
    "FastEmbeddingModel",
    "FastVoiceTokenizer",
    "FileRetriever",
    "EmbeddingResult",
    "VoiceTokenResult",
    "get_embedding",
    "get_voice",
    "get_retriever",
    "NPU_MODELS_AVAILABLE",
    "WorkflowShell",
    "Pipeline",
    "SequentialPipeline",
    "ParallelPipeline",
    "Stage",
    "FunctionStage",
    "get_workflow_shell",
    "WORKFLOW_SHELL_AVAILABLE",
    "SubAgentCommunication",
    "SessionManager",
    "Message",
    "MessageType",
    "get_sub_agent_comm",
    "SUB_AGENT_COMM_AVAILABLE",
    "SkillsRegistry",
    "Skill",
    "SkillType",
    "get_skills_registry",
    "SKILLS_REGISTRY_AVAILABLE",
]
