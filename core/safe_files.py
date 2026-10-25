"""Unified Safe Files Registry for Self-Modification.

This module provides a single source of truth for which files
Jarvis is allowed to modify through self-edit, self-refine,
evolution, and other automated systems.

`SelfEditEngine` consumes these constants, so this module is the one place to
change self-modification policy. It previously sat unimported while
`self_edit.py` carried a second, divergent copy — the two disagreed about
whether `self_edit.py`, `vector_memory.py` and `knowledge_graph.py` were
editable. Reconciled here in the safe direction: critical always wins, and the
forbidden-pattern list is the union of both.
"""
import re
from pathlib import Path
from typing import Set, Tuple

# Files that should NEVER be auto-modified. These either gate access
# (auth/permissions/credentials), define the sandbox that constrains
# self-modification itself (self_edit, prompt_sandbox, lcm_shield,
# context_guardian), or are load-bearing enough that a bad edit is
# unrecoverable without manual intervention (agent, llm, memory, run.py).
CRITICAL_FILES: Set[str] = {
    "core/security_manager.py",
    "core/auth.py",
    "core/auth_middleware.py",
    "core/credentials.py",
    "core/permissions.py",
    "run.py",
    "config.py",
    "core/agent.py",
    "core/llm.py",
    "core/memory.py",
    "core/self_edit.py",
    "core/context_guardian.py",
    "core/vector_memory.py",
    "core/knowledge_graph.py",
    "core/prompt_sandbox.py",
    "core/lcm_shield.py",
}

# Files that are safe for self-modification. Union of what this registry and
# SelfEditEngine each declared, minus anything in CRITICAL_FILES — critical
# always wins, which is what removed self_edit.py, vector_memory.py and
# knowledge_graph.py that the old registry had wrongly marked editable.
SAFE_EDIT_FILES: Set[str] = {
    "core/auto_refine.py",
    "core/autonomous_evolution.py",
    "core/backend_llm.py",
    "core/canvas.py",
    "core/daydream_engine.py",
    "core/daydream_sketch.py",
    "core/daydream_verification.py",
    "core/discovery_notifier.py",
    "core/dm_security.py",
    "core/docker_sandbox.py",
    "core/dual_helpers.py",
    "core/dual_model.py",
    "core/dual_subsystem.py",
    "core/dynamic_memory.py",
    "core/erdos_engine.py",
    "core/evolution_orchestrator.py",
    "core/gpu_simulation.py",
    "core/insight_bridge.py",
    "core/intuition_bridge.py",
    "core/learning_pipeline.py",
    "core/memory_hub.py",
    "core/model_coordinator.py",
    "core/module_bridge.py",
    "core/npu_accelerator.py",
    "core/npu_integration.py",
    "core/npu_optimizations.py",
    "core/resource_monitor.py",
    "core/self_refine.py",
    "core/shifting_memory.py",
    "core/skills_registry.py",
    "core/sub_agent_comm.py",
    "core/tool_learning.py",
    "core/unified_memory.py",
    "core/workflow_shell.py",
    "tools/evolution_tools.py",
    # Web UI components — ViewGroup mode for UI self-refinement
    "web/src/App.css",
    "web/src/App.tsx",
    "web/src/index.css",
    "web/src/components/ActivityFeed.tsx",
    "web/src/components/ChatTab.css",
    "web/src/components/ChatTab.tsx",
    "web/src/components/ConversationSidebar.css",
    "web/src/components/ConversationSidebar.tsx",
    "web/src/components/DashboardTab.tsx",
    "web/src/components/JarvisAvatar.tsx",
    "web/src/components/LiveVisualization.tsx",
    "web/src/components/MediaInput.css",
    "web/src/components/MediaInput.tsx",
    "web/src/components/SelfEvaluationPanel.tsx",
    "web/src/components/TerminalTab.css",
    "web/src/components/TerminalTab.tsx",
    "web/src/components/VisualTab.tsx",
}

# Patterns that are NEVER safe, as (regex source, description) pairs.
# Union of what this registry and SelfEditEngine each carried separately.
# Matched with re.DOTALL so multi-line constructs such as `compile(...exec)`
# cannot evade detection by wrapping across lines.
NEVER_SAFE_PATTERN_SOURCES: list[tuple[str, str]] = [
    (r"exec\s*\(", "exec() - code execution"),
    (r"eval\s*\(", "eval() - code evaluation"),
    (r"__import__\s*\(", "__import__() - dynamic import"),
    # Bare form as well as the subprocess-specific one: a helper that forwards
    # shell=True would otherwise slip past the narrower pattern.
    (r"shell\s*=\s*True", "shell=True - shell injection risk"),
    (r"subprocess\.(call|run|Popen).*shell\s*=\s*True", "subprocess with shell=True"),
    (r"os\.system\s*\(", "os.system() - system command"),
    (r"shutil\.rmtree\s*\(", "rmtree - recursive delete"),
    (r"rm\s+-rf", "rm -rf - recursive delete"),
    (r"pickle\.loads?\s*\(", "pickle.loads - deserialization"),
    (r"marshal\.loads?\s*\(", "marshal.loads - deserialization"),
    (r"compile\s*\(.*exec", "compile/exec - dynamic code"),
    (r"\binput\s*\(", "input() - blocking input"),
    (r"trust_remote_code\s*=\s*True", "trust_remote_code - arbitrary model code"),
    (r"os\.environ\s*\[", "os.environ[] - secret access"),
    (r"getattr\s*\(\s*__builtins__", "getattr(__builtins__) - sandbox escape"),
    (r"globals\s*\(\s*\)", "globals() - namespace access"),
    (r"locals\s*\(\s*\)", "locals() - namespace access"),
    (r"__builtins__", "__builtins__ - sandbox escape"),
    (r"c?import\s+os\s*$", "bare os import"),
    (r"socket\s*\.\s*socket", "raw socket"),
    (r"io\.open\s*\(", "io.open - file access"),
    (r"pathlib\.Path\s*\(\s*['\"]/etc/passwd", "/etc/passwd access"),
    (r"pathlib\.Path\s*\(\s*['\"]/etc/shadow", "/etc/shadow access"),
    (r"breakpoint\s*\(", "breakpoint() - debugger"),
    (r"pdb\.set_trace", "pdb.set_trace - debugger"),
    (r"import\s+pdb\b", "import pdb - debugger"),
]

# Patterns that require caution (warn, do not block).
CAUTION_PATTERN_SOURCES: list[tuple[str, str]] = [
    (r"subprocess\.(call|run|Popen)", "subprocess"),
    (r"open\s*\([^)]*['\"][wab]\+?['\"]", "file write"),
    (r"import\s+selenium", "selenium"),
    (r"import\s+requests", "requests"),
    (r"\.delete\s*\(", "delete operation"),
    (r"\.remove\s*\(", "remove operation"),
    (r"DROP\s+TABLE", "DROP TABLE"),
    (r"DELETE\s+FROM", "DELETE FROM"),
]

NEVER_SAFE_PATTERNS = [
    (re.compile(src, re.IGNORECASE | re.DOTALL | re.MULTILINE), desc)
    for src, desc in NEVER_SAFE_PATTERN_SOURCES
]

CAUTION_PATTERNS = [
    (re.compile(src, re.IGNORECASE), desc)
    for src, desc in CAUTION_PATTERN_SOURCES
]

AUTO_APPROVABLE_EDIT_TYPES: Set[str] = {
    "documentation", "refactor", "optimization",
}

NEVER_AUTO_APPROVE_TYPES: Set[str] = {
    "feature", "bug_fix", "security", "evolution",
}


class PermissionLevel:
    AUTO_APPROVE = "auto_approve"
    QUICK_APPROVE = "quick_approve"
    FULL_REVIEW = "full_review"
    NEVER_AUTO = "never_auto"


def is_safe_file(file_path: str) -> bool:
    """Check if a file is safe for self-modification."""
    path = Path(file_path)
    parts = path.parts
    
    possible_paths = [
        str(path),
        "/".join(parts[-2:]) if len(parts) >= 2 else str(path),
        "/".join(parts[-3:]) if len(parts) >= 3 else str(path),
    ]
    
    for p in possible_paths:
        if p in SAFE_EDIT_FILES:
            return True
    return False


def is_critical_file(file_path: str) -> bool:
    """Check if a file is critical and should never be modified."""
    path = Path(file_path)
    parts = path.parts
    
    possible_paths = [
        str(path),
        "/".join(parts[-2:]) if len(parts) >= 2 else str(path),
    ]
    
    for p in possible_paths:
        if p in CRITICAL_FILES:
            return True
    return False


def contains_forbidden_pattern(code: str) -> Tuple[bool, str]:
    """Check if code contains forbidden patterns."""
    for pattern, desc in NEVER_SAFE_PATTERNS:
        if pattern.search(code):
            return True, desc
    return False, ""


def contains_caution_pattern(code: str) -> Tuple[bool, str]:
    """Check if code contains caution patterns."""
    for pattern, desc in CAUTION_PATTERNS:
        if pattern.search(code):
            return True, desc
    return False, ""


def get_file_risk_level(file_path: str) -> str:
    """Get the risk level for modifying a file."""
    if is_critical_file(file_path):
        return "critical"
    if not is_safe_file(file_path):
        return "high"
    
    path_str = str(Path(file_path))
    if any(x in path_str for x in ["interface", "web", "api"]):
        return "medium"
    if any(x in path_str for x in ["evolution", "daydream", "insight", "learning"]):
        return "low"
    return "low"


def get_edit_permission_level(edit_type: str, impact: str) -> str:
    """Get the permission level required for an edit."""
    if impact == "critical":
        return PermissionLevel.NEVER_AUTO
    
    if edit_type in NEVER_AUTO_APPROVE_TYPES or edit_type in ["feature", "bug_fix", "security", "evolution"]:
        return PermissionLevel.FULL_REVIEW
    
    if edit_type in AUTO_APPROVABLE_EDIT_TYPES:
        if impact == "low":
            return PermissionLevel.AUTO_APPROVE
        elif impact == "medium":
            return PermissionLevel.QUICK_APPROVE
        return PermissionLevel.FULL_REVIEW
    
    if impact == "low":
        return PermissionLevel.QUICK_APPROVE
    return PermissionLevel.FULL_REVIEW
