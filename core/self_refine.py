"""Self-Refinement Agent - Enhanced autonomous code improvement system.

This module provides Jarvis with the ability to:
1. Deeply analyze its own codebase
2. Identify improvement opportunities
3. Generate safe fixes and enhancements
4. Execute refinements with full visibility
5. Learn from each refinement cycle

Architecture:
- AgentLayer: Provides full codebase visibility and analysis
- RefinementScheduler: Manages background refinement cycles
- SafeExecutor: Applies changes with rollback capability
- LearningFeedback: Learns from each refinement outcome
"""
import ast
import asyncio
import hashlib
import json
import logging
import re
import sys
import threading
import time
from collections import defaultdict
from core.safe_files import (
    CRITICAL_FILES as REGISTRY_CRITICAL_FILES,
    SAFE_EDIT_FILES,
)
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Callable, Any

from core.dual_helpers import self_refine_analyze, self_refine_fix
from core.helpers import atomic_write_json, get_response_content, create_tracked_task, safe_error_message, read_json_with_integrity

logger = logging.getLogger(__name__)


class RefinementState(Enum):
    IDLE = "idle"
    SCANNING = "scanning"
    ANALYZING = "analyzing"
    PROPOSING = "proposing"
    EXECUTING = "executing"
    TESTING = "testing"
    DEBUGGING = "debugging"
    ROLLING_BACK = "rolling_back"
    LEARNING = "learning"
    PAUSED = "paused"


class RefinementPriority(Enum):
    CRITICAL = 1    # Bugs, security issues
    HIGH = 2        # Performance, reliability
    MEDIUM = 3      # Code quality, maintainability
    LOW = 4         # Style, minor improvements
    ENHANCEMENT = 5 # New features, optimizations


@dataclass
class RefinementTask:
    """A single refinement task."""
    id: str
    title: str
    description: str
    file_path: str
    priority: RefinementPriority
    category: str
    original_code: str
    proposed_code: str
    reasoning: str
    risk_level: str
    dependencies: list[str] = field(default_factory=list)
    test_required: bool = True
    rollback_code: str | None = None
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.now)
    executed_at: datetime | None = None
    result: str | None = None
    impact_score: float = 0.0
    confidence: float = 0.0
    source: str = "auto"  # auto, intuition, user, schedule
    line_number: int = 0


@dataclass
class RefinementSession:
    """A refinement session with context."""
    id: str
    started_at: datetime
    state: RefinementState
    files_analyzed: int = 0
    issues_found: int = 0
    tasks_generated: int = 0
    tasks_executed: int = 0
    tasks_succeeded: int = 0
    tasks_failed: int = 0
    tasks_debugged: int = 0
    debug_successes: int = 0
    current_task: str | None = None
    current_file: str | None = None
    progress: float = 0.0
    log: list[str] = field(default_factory=list)
    MAX_LOG_ENTRIES = 500

    def add_log(self, message: str):
        """Add a message to session log with bounds."""
        self.log.append(message)
        if len(self.log) > self.MAX_LOG_ENTRIES:
            self.log = self.log[-self.MAX_LOG_ENTRIES:]


@dataclass
class DebugSession:
    """A debugging session for a failed task."""
    id: str
    task_id: str
    started_at: datetime
    error_type: str
    error_message: str
    error_traceback: str | None = None
    file_path: str | None = None
    line_number: int | None = None
    attempts: int = 0
    max_attempts: int = 3
    fixes_tried: list[str] = field(default_factory=list)
    fix_applied: str | None = None
    success: bool = False
    resolved_at: datetime | None = None
    root_cause: str | None = None
    solution: str | None = None


class AgentLayer:
    """Provides full codebase visibility and analysis capabilities.
    
    This layer gives the self-refinement system the same visibility
    that an AI assistant would have - able to read, analyze, and
    understand the entire codebase structure.
    """

    # Self-modification policy comes from core/safe_files.py, the single registry
    # shared with SelfEditEngine. This module used to carry a fourth private copy
    # whose SAFE_FILES listed core/knowledge_graph.py — which the registry treats
    # as CRITICAL — so self-refine considered it freely editable.
    SAFE_FILES = SAFE_EDIT_FILES
    CRITICAL_FILES = REGISTRY_CRITICAL_FILES

    def __init__(self, base_path: Path):
        self.base_path = base_path
        self._file_cache: dict[str, tuple[str, float]] = {}
        self._ast_cache: dict[str, tuple[ast.AST, str]] = {}
        self._MAX_FILE_CACHE = 100
        self._MAX_AST_CACHE = 50
        self._dependencies: dict[str, set[str]] = {}
        self._last_analysis: datetime | None = None
        self._cache_lock = threading.Lock()

    def get_file_content(self, file_path: str, use_cache: bool = True) -> str | None:
        """Get file content with caching."""
        full_path = self.base_path / file_path

        if not full_path.exists():
            return None

        mtime = full_path.stat().st_mtime

        if use_cache:
            with self._cache_lock:
                if file_path in self._file_cache:
                    cached_content, cached_mtime = self._file_cache[file_path]
                    if cached_mtime == mtime:
                        return cached_content

        try:
            content = full_path.read_text(encoding='utf-8')
            with self._cache_lock:
                self._file_cache[file_path] = (content, mtime)
                if len(self._file_cache) > self._MAX_FILE_CACHE:
                    oldest_key = next(iter(self._file_cache))
                    del self._file_cache[oldest_key]
            return content
        except Exception as e:
            logger.debug(f"Failed to read {file_path}: {e}")
            return None

    def get_file_ast(self, file_path: str) -> ast.AST | None:
        """Parse file into AST with caching."""
        content = self.get_file_content(file_path)
        if not content:
            return None

        cache_key = hashlib.sha256(content.encode()).hexdigest()

        with self._cache_lock:
            if file_path in self._ast_cache:
                cached_ast, cached_key = self._ast_cache[file_path]
                if cached_key == cache_key:
                    return cached_ast

        try:
            tree = ast.parse(content)
            with self._cache_lock:
                self._ast_cache[file_path] = (tree, cache_key)
                if len(self._ast_cache) > self._MAX_AST_CACHE:
                    oldest_key = next(iter(self._ast_cache))
                    del self._ast_cache[oldest_key]
            return tree
        except SyntaxError as e:
            logger.debug(f"Syntax error in {file_path}: {e}")
            return None

    def get_all_python_files(self) -> list[str]:
        """Get all Python files in the codebase."""
        files = []
        # Exclude scanning tools that contain pattern definitions (false positives)
        scanner_files = {"auto_refine.py", "self_refine.py"}
        for py_file in self.base_path.rglob("*.py"):
            rel_path = str(py_file.relative_to(self.base_path))
            if not any(ex in rel_path for ex in ["__pycache__", ".venv", "venv", "test_", "tests/"]):
                # Skip scanner files - they contain pattern definitions that look like risks
                if py_file.name in scanner_files:
                    continue
                files.append(rel_path)
        return sorted(files)

    def get_codebase_structure(self) -> dict:
        """Analyze codebase structure."""
        structure = {
            "directories": {},
            "files": {},
            "imports": defaultdict(set),
            "exports": defaultdict(set),
            "classes": {},
            "functions": {},
            "total_lines": 0,
            "total_files": 0,
        }

        for py_file in self.get_all_python_files():
            tree = self.get_file_ast(py_file)
            if not tree:
                continue

            content = self.get_file_content(py_file)
            if content:
                structure["total_lines"] += len(content.split("\n"))
                structure["total_files"] += 1

            structure["files"][py_file] = {
                "classes": [],
                "functions": [],
                "imports": [],
                "lines": len(content.split("\n")) if content else 0,
            }

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    structure["files"][py_file]["classes"].append(node.name)
                    structure["classes"][node.name] = py_file

                elif isinstance(node, ast.FunctionDef):
                    structure["files"][py_file]["functions"].append(node.name)
                    structure["functions"][node.name] = py_file

                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        structure["files"][py_file]["imports"].append(alias.name)
                        structure["imports"][py_file].add(alias.name)

                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        imp = f"{module}.{alias.name}" if module else alias.name
                        structure["files"][py_file]["imports"].append(imp)
                        structure["imports"][py_file].add(imp)

        structure["imports"] = {k: list(v) for k, v in structure["imports"].items()}
        structure["exports"] = dict(structure["exports"])

        return structure

    def analyze_dependencies(self) -> dict[str, set[str]]:
        """Build dependency graph."""
        if self._dependencies:
            return self._dependencies

        for py_file in self.get_all_python_files():
            content = self.get_file_content(py_file)
            if not content:
                continue

            deps = set()

            # Find local imports
            for match in re.finditer(r'^(?:from|import)\s+((?:core|tools|interfaces)\.[^\s]+)', content, re.MULTILINE):
                dep = match.group(1).split('.')[0]
                deps.add(dep)

            self._dependencies[py_file] = deps

        return self._dependencies

    def is_safe_to_modify(self, file_path: str) -> tuple[bool, str]:
        """Check if a file is safe to modify.

        The registry keys are POSIX-style ("core/agent.py") but callers pass
        paths from get_all_python_files(), which on Windows produces backslashes
        ("core\\agent.py"). Both set-membership tests then missed and the
        function fell through to (True, "moderate") — self-refine considered
        core/agent.py, core/llm.py and run.py freely editable on this platform.
        Normalise before comparing, and check critical *first* so that a file
        appearing in both lists is refused rather than approved.
        """
        key = str(file_path).replace("\\", "/")

        if key in self.CRITICAL_FILES:
            return False, "critical - requires explicit approval"
        if key in self.SAFE_FILES:
            return True, "safe"
        if "test" in key.lower():
            return False, "test file - modify with caution"
        return True, "moderate"

    def get_file_context(self, file_path: str, line_number: int, context_lines: int = 20) -> str:
        """Get context around a line in a file."""
        content = self.get_file_content(file_path)
        if not content:
            return ""

        lines = content.split("\n")
        start = max(0, line_number - context_lines - 1)
        end = min(len(lines), line_number + context_lines)

        context = []
        for i in range(start, end):
            prefix = ">>> " if i == line_number - 1 else "    "
            context.append(f"{prefix}{i+1}: {lines[i]}")

        return "\n".join(context)

    def get_raw_context(self, file_path: str, line_number: int, context_lines: int = 10) -> str:
        """Get raw code context around a line (for use in original_code)."""
        content = self.get_file_content(file_path)
        if not content:
            return ""

        lines = content.split("\n")
        start = max(0, line_number - context_lines - 1)
        end = min(len(lines), line_number + context_lines)

        return "\n".join(lines[start:end])

    def find_similar_code(self, pattern: str, file_patterns: list[str] | None = None) -> list[dict]:
        """Find similar code patterns across the codebase."""
        results = []

        for py_file in self.get_all_python_files():
            if file_patterns and not any(p in py_file for p in file_patterns):
                continue

            content = self.get_file_content(py_file)
            if not content:
                continue

            for match in re.finditer(pattern, content, re.MULTILINE | re.IGNORECASE):
                line_num = content[:match.start()].count("\n") + 1
                results.append({
                    "file": py_file,
                    "line": line_num,
                    "match": match.group(0),
                    "context": self.get_file_context(py_file, line_num, 5),
                })

        return results

    def analyze_code_patterns(self) -> dict:
        """Analyze common code patterns and potential improvements."""
        patterns = {
            "todo_comments": [],
            "fixme_comments": [],
            "security_risks": [],
            "performance_issues": [],
            "code_smells": [],
            "missing_type_hints": [],
            "missing_docstrings": [],
            "duplicate_code": [],
        }

        for py_file in self.get_all_python_files():
            content = self.get_file_content(py_file)
            tree = self.get_file_ast(py_file)

            if not content or not tree:
                continue

            # Find TODO/FIXME
            for match in re.finditer(r'#\s*(TODO|FIXME):?\s*(.+)', content, re.IGNORECASE):
                line_num = content[:match.start()].count("\n") + 1
                item = {
                    "file": py_file,
                    "line": line_num,
                    "text": match.group(2).strip(),
                }
                if "TODO" in match.group(1).upper():
                    patterns["todo_comments"].append(item)
                else:
                    patterns["fixme_comments"].append(item)

            # Security patterns - use word boundaries and actual function calls
            # Match exec(, eval(, etc. only when they're actual function calls, not methods
            for match in re.finditer(r'(?<![.\w])(exec|eval)\s*\(', content):
                line_num = content[:match.start()].count("\n") + 1
                line_start = content.rfind('\n', 0, match.start()) + 1
                line_end = content.find('\n', match.start())
                line_text = content[line_start:line_end] if line_end != -1 else content[line_start:]
                
                # Skip if preceded by dot (method call like model.eval())
                pos_in_line = match.start() - line_start
                if pos_in_line > 0 and line_text[pos_in_line - 1] == '.':
                    continue
                
                # Skip if in a tuple/list/string (pattern definitions)
                if any(indicator in line_text for indicator in ['("', "('", '["', "['", '"exec', '"eval', "'exec", "'eval"]):
                    continue
                
                # Skip false positives (comments)
                stripped = line_text.strip()
                if stripped.startswith('#'):
                    continue
                    
                patterns["security_risks"].append({
                    "file": py_file,
                    "line": line_num,
                    "pattern": match.group(1) + "()",
                    "context": self.get_file_context(py_file, line_num, 3),
                })
            
            # Match shell=True only in actual subprocess calls
            for match in re.finditer(r'subprocess\..*shell\s*=\s*True', content):
                line_num = content[:match.start()].count("\n") + 1
                patterns["security_risks"].append({
                    "file": py_file,
                    "line": line_num,
                    "pattern": "shell=True in subprocess",
                    "context": self.get_file_context(py_file, line_num, 3),
                })
            
            # Skip pickle.load entirely - too many false positives for trusted cached data
            # pickle.loads() from untrusted sources is the real risk, but that's harder to detect

            # Performance patterns
            for match in re.finditer(r'(for\s+\w+\s+in\s+range\(len\(|\.append\s*\([^)]*\)\s+in\s*.*for|\+\s*=\s*.*\s+in\s+.*for)', content):
                line_num = content[:match.start()].count("\n") + 1
                patterns["performance_issues"].append({
                    "file": py_file,
                    "line": line_num,
                    "pattern": match.group(1),
                    "context": self.get_raw_context(py_file, line_num, 5),
                })

            # AST-based analysis
            for node in ast.walk(tree):
                # Missing docstrings - only for public functions/classes, skip private/internal
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                    # Skip private methods (starting with _)
                    if node.name.startswith('_'):
                        continue
                    # Skip common patterns (getters, setters, properties)
                    if node.name in ['get', 'set', 'run', 'main', 'test']:
                        continue
                    # Check for docstring
                    if not (node.body and isinstance(node.body[0], ast.Expr) and
                            isinstance(node.body[0].value, ast.Constant)):
                        # Limit to 50 missing docstrings per file
                        if len([p for p in patterns["missing_docstrings"] if p["file"] == py_file]) < 50:
                            patterns["missing_docstrings"].append({
                                "file": py_file,
                                "line": node.lineno,
                                "name": node.name,
                            "type": "class" if isinstance(node, ast.ClassDef) else "function",
                        })

                # Missing type hints
                if isinstance(node, ast.FunctionDef):
                    if not node.returns and not any(isinstance(n, ast.AnnAssign) for n in ast.walk(node)):
                        # Simple heuristic: function without return annotation
                        pass  # Skip for now, too noisy

        return patterns


class SelfRefineAgent:
    """Enhanced self-refinement agent with full codebase visibility.
    
    This agent operates autonomously but can be guided by:
    - Intuition bridge insights
    - Scheduled refinement cycles
    - User requests
    - Performance metrics
    """

    MAX_PENDING_TASKS = 50
    MAX_EXECUTED_PER_CYCLE = 5
    MAX_DEBUG_ATTEMPTS = 3
    MAX_COMPLETED_TASKS = 200  # Limit history of completed tasks
    MAX_DEBUG_SESSIONS = 50    # Limit debug session history
    MAX_LEARNED_FIXES = 100    # Limit learned fix patterns
    ANALYSIS_INTERVAL = 300  # 5 minutes
    REFINEMENT_INTERVAL = 600  # 10 minutes

    # Common error patterns for self-debugging
    ERROR_PATTERNS = {
        "import_error": {
            "pattern": r"ImportError|ModuleNotFoundError|cannot import name",
            "handler": "_debug_import_error",
        },
        "attribute_error": {
            "pattern": r"AttributeError",
            "handler": "_debug_attribute_error",
        },
        "type_error": {
            "pattern": r"TypeError",
            "handler": "_debug_type_error",
        },
        "key_error": {
            "pattern": r"KeyError",
            "handler": "_debug_key_error",
        },
        "index_error": {
            "pattern": r"IndexError",
            "handler": "_debug_index_error",
        },
        "value_error": {
            "pattern": r"ValueError",
            "handler": "_debug_value_error",
        },
        "name_error": {
            "pattern": r"NameError",
            "handler": "_debug_name_error",
        },
        "syntax_error": {
            "pattern": r"SyntaxError|IndentationError",
            "handler": "_debug_syntax_error",
        },
        "test_failure": {
            "pattern": r"FAILED|AssertionError|assert",
            "handler": "_debug_test_failure",
        },
        "timeout_error": {
            "pattern": r"TimeoutError|timed out",
            "handler": "_debug_timeout_error",
        },
    }

    def __init__(self, llm_client, base_path: Path | None = None):
        self.base_path = base_path or Path(__file__).parent.parent
        self.llm = llm_client
        self.agent_layer = AgentLayer(self.base_path)

        # State
        self._state = RefinementState.IDLE
        self._session: RefinementSession | None = None
        self._tasks: dict[str, RefinementTask] = {}
        self._task_counter = 0
        self._completed_tasks: list[RefinementTask] = []
        self._debug_sessions: dict[str, DebugSession] = {}
        self._debug_counter = 0
        self._learned_fixes: dict[str, str] = {}  # error_pattern -> fix

        # Callbacks
        self._permission_callback: Callable | None = None
        self._notify_callback: Callable | None = None

        # Background task
        self._running = False
        self._task: asyncio.Task | None = None

        # Lock
        self._lock = threading.RLock()

        # API manager for coordinated calls
        self._api_manager = None
        self._subsystem_name = "self_refine"

        # Storage
        self.storage_path = Path.home() / ".jarvis" / "self_refine"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Load state
        self._load()

        logger.info(f"SelfRefineAgent initialized with {len(self._tasks)} pending tasks")

    def _get_api_manager(self):
        """Get or create API manager for coordinated LLM calls."""
        if self._api_manager is None and self.llm:
            try:
                from core.api_manager import get_api_manager
                self._api_manager = get_api_manager(self.llm)
            except Exception as e:
                logger.warning(f"Failed to create API manager: {e}")
        return self._api_manager

    async def _llm_call(self, user_input: str, system_prompt: str, priority: str = "refinement") -> dict:
        """Make coordinated LLM call with rate limiting."""
        from core.api_manager import Priority as APIPriority

        priority_map = {
            "interactive": APIPriority.INTERACTIVE,
            "learning": APIPriority.LEARNING,
            "refinement": APIPriority.REFINEMENT,
            "maintenance": APIPriority.MAINTENANCE,
        }

        api_manager = self._get_api_manager()
        if api_manager is not None:
            return await api_manager.submit_chat(
                subsystem=self._subsystem_name,
                priority=priority_map.get(priority, APIPriority.REFINEMENT),
                user_input=user_input,
                system_prompt=system_prompt,
            )

        if self.llm:
            from core.llm import subsystem_context
            async with subsystem_context("Self-Refine"):
                return await self.llm.chat(user_input=user_input, system_prompt=system_prompt)

        raise RuntimeError("No LLM client available")

    def set_permission_callback(self, callback: Callable):
        """Set callback for permission requests."""
        self._permission_callback = callback

    def set_notify_callback(self, callback: Callable):
        """Set callback for notifications."""
        self._notify_callback = callback

    # Agentic permission settings
    AUTO_APPROVE_LOW_RISK = True      # Auto-approve documentation, style
    AUTO_APPROVE_SAFE_REFACTOR = True  # Auto-approve safe refactors
    AUTO_APPROVE_PERFORMANCE = True    # Auto-approve performance fixes
    AUTO_APPROVE_FIXME = True          # Auto-approve FIXME fixes
    AUTO_DEBUG_ENABLED = True          # Auto-debug failed changes
    MAX_AUTO_APPROVES_PER_SESSION = 20  # Rate limit
    REQUIRE_TEST_FOR_HIGH_RISK = True  # Run tests for high-risk changes

    def set_agentic_settings(
        self,
        auto_approve_low_risk: bool | None = None,
        auto_approve_safe_refactor: bool | None = None,
        auto_approve_performance: bool | None = None,
        auto_approve_fixme: bool | None = None,
        auto_debug: bool | None = None,
        max_per_session: int | None = None,
    ):
        """Configure agentic permission settings."""
        if auto_approve_low_risk is not None:
            self.AUTO_APPROVE_LOW_RISK = auto_approve_low_risk
        if auto_approve_safe_refactor is not None:
            self.AUTO_APPROVE_SAFE_REFACTOR = auto_approve_safe_refactor
        if auto_approve_performance is not None:
            self.AUTO_APPROVE_PERFORMANCE = auto_approve_performance
        if auto_approve_fixme is not None:
            self.AUTO_APPROVE_FIXME = auto_approve_fixme
        if auto_debug is not None:
            self.AUTO_DEBUG_ENABLED = auto_debug
        if max_per_session is not None:
            self.MAX_AUTO_APPROVES_PER_SESSION = max_per_session
        logger.info(f"Agentic settings: low_risk={self.AUTO_APPROVE_LOW_RISK}, "
                   f"refactor={self.AUTO_APPROVE_SAFE_REFACTOR}, debug={self.AUTO_DEBUG_ENABLED}")

    def get_agentic_settings(self) -> dict:
        """Get current agentic settings."""
        return {
            "auto_approve_low_risk": self.AUTO_APPROVE_LOW_RISK,
            "auto_approve_safe_refactor": self.AUTO_APPROVE_SAFE_REFACTOR,
            "auto_approve_performance": self.AUTO_APPROVE_PERFORMANCE,
            "auto_approve_fixme": self.AUTO_APPROVE_FIXME,
            "auto_debug_enabled": self.AUTO_DEBUG_ENABLED,
            "max_auto_approves_per_session": self.MAX_AUTO_APPROVES_PER_SESSION,
            "require_test_for_high_risk": self.REQUIRE_TEST_FOR_HIGH_RISK,
        }

    def get_state(self) -> RefinementState:
        """Get current state."""
        return self._state

    def get_session(self) -> RefinementSession | None:
        """Get current session."""
        return self._session

    def get_pending_tasks(self, limit: int = 20) -> list[RefinementTask]:
        """Get pending tasks sorted by priority."""
        with self._lock:
            pending = [t for t in self._tasks.values() if t.status == "pending"]
            pending.sort(key=lambda t: (t.priority.value, -t.impact_score))
            return pending[:limit]

    def get_stats(self) -> dict:
        """Get refinement statistics."""
        with self._lock:
            return {
                "state": self._state.value,
                "running": self._running,
                "pending_tasks": len([t for t in self._tasks.values() if t.status == "pending"]),
                "completed_tasks": len(self._completed_tasks),
                "failed_tasks": len([t for t in self._completed_tasks if t.status == "failed"]),
                "debug_sessions": len(self._debug_sessions),
                "current_session": self._session.id if self._session else None,
                "total_analyzed": self._session.files_analyzed if self._session else 0,
            }

    async def run_system_health_check(self) -> dict:
        """Run a comprehensive system health check and debug analysis.
        
        Moved from daydream to self-refine for active monitoring.
        Returns findings that may generate refinement tasks.
        """
        findings = []
        
        # 1. Check for logged errors and analyze them
        try:
            log_dir = self.base_path / "logs"
            if log_dir.exists():
                jarvis_log = log_dir / "jarvis.log"
                if jarvis_log.exists():
                    log_content = await asyncio.to_thread(jarvis_log.read_text, errors='ignore')
                    error_patterns = [
                        r'\[ERROR\].*',
                        r'\[WARNING\].*(?:failed|error|exception)',
                        r'Traceback.*',
                    ]
                    errors_found = []
                    for pattern in error_patterns:
                        matches = re.findall(pattern, log_content[-50000:], re.IGNORECASE)
                        errors_found.extend(matches[-5:])
                    
                    if errors_found:
                        findings.append({
                            "type": "log_errors",
                            "severity": "medium",
                            "count": len(errors_found),
                            "description": f"Found {len(errors_found)} recent errors in logs",
                            "details": errors_found[:3],
                        })
        except Exception as e:
            logger.debug(f"Log analysis error: {e}")
        
        # 2. Check subsystem health
        try:
            from core.health import get_health_manager
            health_mgr = get_health_manager()
            health_data = health_mgr.get_all_health()
            subsystems = health_data.get("subsystems", {})
            unhealthy = [name for name, data in subsystems.items() 
                         if data.get('status') in ('unhealthy', 'stopped')]
            if unhealthy:
                findings.append({
                    "type": "subsystem_health",
                    "severity": "high",
                    "subsystems": unhealthy,
                    "description": f"Unhealthy subsystems: {', '.join(unhealthy)}",
                })
        except Exception as e:
            logger.warning(f"Failed to check subsystem health: {e}")
        
        # 3. Check for memory pressure
        try:
            from core.dynamic_memory import get_dynamic_memory_manager
            mm = get_dynamic_memory_manager()
            mem_status = mm.get_status()
            if mem_status.get("pressure") in ["high", "critical"]:
                findings.append({
                    "type": "memory_pressure",
                    "severity": "high",
                    "pressure": mem_status.get("pressure"),
                    "description": f"Memory pressure: {mem_status.get('pressure')}",
                })
        except Exception as e:
            logger.warning(f"Failed to check memory pressure: {e}")
        
        # 4. Analyze tool performance
        try:
            from core.tool_learning import get_tool_learning_engine
            engine = get_tool_learning_engine()
            competencies = engine.get_all_competencies()
            low_performers = [name for name, comp in competencies.items() 
                              if hasattr(comp, 'successful_uses') and hasattr(comp, 'total_uses')
                              and comp.total_uses > 0 and (comp.successful_uses / comp.total_uses) < 0.5]
            if low_performers:
                findings.append({
                    "type": "tool_performance",
                    "severity": "low",
                    "tools": low_performers[:5],
                    "description": f"Low performer tools: {', '.join(low_performers[:3])}",
                })
        except Exception as e:
            logger.warning(f"Failed to analyze tool performance: {e}")
        
        # 5. Check for stuck tasks
        with self._lock:
            pending = [t for t in self._tasks.values() if t.status == "pending"]
            if len(pending) > 10:
                findings.append({
                    "type": "task_backlog",
                    "severity": "medium",
                    "count": len(pending),
                    "description": f"Many pending refinement tasks: {len(pending)}",
                })
        
        # 6. Analyze dual-model activity for issues
        try:
            from core.dual_model import get_dual_model_orchestrator
            orchestrator = get_dual_model_orchestrator()
            if orchestrator:
                stats = orchestrator.get_stats()
                error_rate = stats.get("errors", 0) / max(1, stats.get("total_requests", 1))
                if error_rate > 0.1:  # More than 10% errors
                    findings.append({
                        "type": "dual_model_errors",
                        "severity": "high",
                        "error_rate": error_rate,
                        "errors": stats.get("errors", 0),
                        "total": stats.get("total_requests", 0),
                        "description": f"High dual-model error rate: {error_rate:.1%}",
                    })
        except Exception as e:
            logger.warning(f"Failed to check dual-model error rate: {e}")
        
        if findings:
            logger.info(f"System health check found {len(findings)} issues")
            self._create_health_check_tasks(findings)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "findings": findings,
            "healthy": len([f for f in findings if f.get("severity") in ("high", "critical")]) == 0,
        }

    def _create_health_check_tasks(self, findings: list) -> None:
        """Create refinement tasks from health check findings."""
        for finding in findings:
            if finding.get("severity") in ("high", "critical"):
                with self._lock:
                    if len(self._tasks) < self.MAX_PENDING_TASKS:
                        self._task_counter += 1
                        task = RefinementTask(
                            id=f"task_{self._task_counter:06d}",
                            title=f"Health: {finding.get('type', 'Unknown issue')}",
                            description=finding.get("description", ""),
                            file_path="system",
                            priority=RefinementPriority.HIGH if finding.get("severity") == "critical" else RefinementPriority.MEDIUM,
                            category="health",
                            original_code="",
                            proposed_code="",
                            reasoning=f"System health check identified: {finding.get('description', '')}",
                            risk_level="medium",
                            source="health_check",
                        )
                        self._tasks[task.id] = task
                        logger.info(f"Created health check task: {task.id}")

    async def scan_now(self) -> dict:
        """Manually trigger a scan and generate tasks.
        
        Returns dict with scan results.
        """
        try:
            with self._lock:
                self._state = RefinementState.SCANNING
            
            patterns = self.agent_layer.analyze_code_patterns()
            structure = self.agent_layer.get_codebase_structure()
            
            total_issues = sum(len(v) for v in patterns.values())
            
            tasks = await self._generate_tasks_from_patterns(patterns)
            tasks_added = 0
            
            for task in tasks:
                with self._lock:
                    if len(self._tasks) < self.MAX_PENDING_TASKS:
                        self._task_counter += 1
                        task.id = f"task_{self._task_counter:06d}"
                        self._tasks[task.id] = task
                        tasks_added += 1
            
            with self._lock:
                self._state = RefinementState.IDLE
            self._save()
            
            return {
                "success": True,
                "files_scanned": structure.get("total_files", 0),
                "issues_found": total_issues,
                "tasks_generated": tasks_added,
                "patterns": {k: len(v) for k, v in patterns.items()},
            }
        except Exception as e:
            self._state = RefinementState.IDLE
            logger.error(f"Scan error: {e}")
            return {"success": False, "error": safe_error_message(e)}

    async def start(self):
        """Start the refinement agent."""
        if self._running:
            return

        self._running = True
        self._task = create_tracked_task(self._run_loop(), name="self_refine_loop")
        logger.info("Self-refinement agent started")

    async def stop(self):
        """Stop the refinement agent."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._save()
        logger.info("Self-refinement agent stopped")

    async def _run_loop(self):
        """Background refinement loop."""
        while self._running:
            try:
                # Start new session
                await self._run_session()

                # Run system health check after each session
                await self.run_system_health_check()

                # Wait before next cycle
                await asyncio.sleep(self.REFINEMENT_INTERVAL)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Refinement loop error: {e}")
                with self._lock:
                    self._state = RefinementState.IDLE
                await asyncio.sleep(60)

    async def _run_session(self):
        """Run a refinement session."""
        session_id = f"session_{int(time.time())}"
        self._session = RefinementSession(
            id=session_id,
            started_at=datetime.now(),
            state=RefinementState.SCANNING,
        )
        self._session.add_log(f"Session started at {datetime.now().isoformat()}")

        try:
            # Phase 1: Analyze codebase
            with self._lock:
                self._state = RefinementState.SCANNING
            self._session.add_log("Scanning codebase...")

            patterns = self.agent_layer.analyze_code_patterns()
            self._session.files_analyzed = self.agent_layer.get_codebase_structure()["total_files"]
            self._session.issues_found = sum(len(v) for v in patterns.values())
            self._session.add_log(f"Found {self._session.issues_found} potential issues in {self._session.files_analyzed} files")

            # Phase 2: Generate tasks from patterns
            with self._lock:
                self._state = RefinementState.ANALYZING
            self._session.add_log("Analyzing patterns...")

            tasks = await self._generate_tasks_from_patterns(patterns)
            for task in tasks:
                with self._lock:
                    if len(self._tasks) < self.MAX_PENDING_TASKS:
                        self._task_counter += 1
                        task.id = f"task_{self._task_counter:06d}"
                        self._tasks[task.id] = task
                        self._session.tasks_generated += 1

            self._session.add_log(f"Generated {len(tasks)} refinement tasks")

            # Phase 3: Execute high-priority tasks
            with self._lock:
                self._state = RefinementState.EXECUTING
            pending = self.get_pending_tasks(limit=self.MAX_EXECUTED_PER_CYCLE)

            for task in pending:
                if not self._running:
                    break

                self._session.current_task = task.id
                self._session.current_file = task.file_path
                self._session.add_log(f"Executing task {task.id}: {task.title}")

                # Request permission for modifications
                can_proceed = await self._request_permission(task)

                if can_proceed:
                    success, result = await self._execute_task(task)
                    self._session.tasks_executed += 1
                    if success:
                        self._session.tasks_succeeded += 1
                        self._session.add_log(f"Task {task.id} succeeded: {result[:100]}")
                    else:
                        self._session.tasks_failed += 1
                        self._session.add_log(f"Task {task.id} failed: {result[:100]}")
                else:
                    self._session.add_log(f"Task {task.id} skipped (no permission)")

                self._session.progress = self._session.tasks_executed / max(1, len(pending))

            # Phase 4: Learning
            with self._lock:
                self._state = RefinementState.LEARNING
            self._session.add_log("Processing learnings...")
            await self._process_learnings()

            # Complete session
            self._session.state = RefinementState.IDLE
            self._session.add_log(f"Session completed at {datetime.now().isoformat()}")
            with self._lock:
                self._state = RefinementState.IDLE

        except Exception as e:
            self._session.add_log(f"Session error: {safe_error_message(e)}")
            logger.error(f"Refinement session error: {e}")
            with self._lock:
                self._state = RefinementState.IDLE

        finally:
            self._save()

    async def _generate_tasks_from_patterns(self, patterns: dict) -> list[RefinementTask]:
        """Generate refinement tasks from analyzed patterns."""
        tasks = []

        # Security risks - always high priority
        for risk in patterns.get("security_risks", [])[:10]:
            safe, reason = self.agent_layer.is_safe_to_modify(risk["file"])
            tasks.append(RefinementTask(
                id="",
                title=f"Security: {risk['pattern']} in {risk['file']}",
                description=f"Potential security risk: {risk['pattern']} at line {risk['line']}",
                file_path=risk["file"],
                priority=RefinementPriority.CRITICAL,
                category="security",
                original_code=risk.get("context", ""),
                proposed_code="",  # Will be generated
                reasoning=f"Security risk detected: {risk['pattern']} can lead to code injection or other vulnerabilities",
                risk_level="high" if not safe else "medium",
                confidence=0.95,
                impact_score=0.95,
            ))

        # Performance issues
        for issue in patterns.get("performance_issues", [])[:10]:
            safe, reason = self.agent_layer.is_safe_to_modify(issue["file"])
            tasks.append(RefinementTask(
                id="",
                title=f"Performance: {issue['pattern'][:30]}... in {issue['file']}",
                description=f"Performance improvement opportunity at line {issue['line']}",
                file_path=issue["file"],
                priority=RefinementPriority.HIGH,
                category="performance",
                original_code=issue.get("context", ""),
                proposed_code="",
                reasoning=f"Performance pattern detected: {issue['pattern']}",
                risk_level="low" if safe else "medium",
                confidence=0.8,
                impact_score=0.7,
                line_number=issue.get("line", 0),
            ))

        # TODOs - convert to tasks
        for todo in patterns.get("todo_comments", [])[:5]:
            safe, reason = self.agent_layer.is_safe_to_modify(todo["file"])
            tasks.append(RefinementTask(
                id="",
                title=f"TODO: {todo['text'][:50]}",
                description=f"TODO comment at line {todo['line']}: {todo['text']}",
                file_path=todo["file"],
                priority=RefinementPriority.LOW,
                category="todo",
                original_code="",
                proposed_code="",
                reasoning="TODO comment found - consider implementing",
                risk_level="low" if safe else "medium",
                confidence=0.6,
                impact_score=0.4,
            ))

        # FIXMEs - higher priority
        for fixme in patterns.get("fixme_comments", [])[:5]:
            safe, reason = self.agent_layer.is_safe_to_modify(fixme["file"])
            tasks.append(RefinementTask(
                id="",
                title=f"FIXME: {fixme['text'][:50]}",
                description=f"FIXME comment at line {fixme['line']}: {fixme['text']}",
                file_path=fixme["file"],
                priority=RefinementPriority.HIGH,
                category="fixme",
                original_code="",
                proposed_code="",
                reasoning="FIXME comment found - needs attention",
                risk_level="medium" if safe else "high",
                confidence=0.7,
                impact_score=0.7,
            ))

        # Missing docstrings
        for item in patterns.get("missing_docstrings", [])[:10]:
            safe, reason = self.agent_layer.is_safe_to_modify(item["file"])
            tasks.append(RefinementTask(
                id="",
                title=f"Docstring: {item['name']} in {item['file']}",
                description=f"{item['type'].capitalize()} '{item['name']}' missing docstring",
                file_path=item["file"],
                priority=RefinementPriority.LOW,
                category="documentation",
                original_code="",
                proposed_code="",
                reasoning="Adding docstring improves code clarity",
                risk_level="low",
                confidence=0.5,
                impact_score=0.3,
            ))

        return tasks

    async def _request_permission(self, task: RefinementTask) -> bool:
        """Request permission to execute a task.
        
        Uses agentic settings to determine if task can be auto-approved.
        Returns True if permission is granted (either auto or via callback).
        """
        # Check auto-approval rules based on task category and risk
        can_auto_approve = False
        auto_reason = ""
        
        # Documentation tasks are always safe
        if task.category == "documentation" and self.AUTO_APPROVE_LOW_RISK:
            can_auto_approve = True
            auto_reason = "documentation (low risk)"
        
        # Performance optimizations
        elif task.category == "performance" and self.AUTO_APPROVE_PERFORMANCE:
            safe, _ = self.agent_layer.is_safe_to_modify(task.file_path)
            if safe and task.risk_level in ("low", "medium"):
                can_auto_approve = True
                auto_reason = f"performance fix in safe file ({task.risk_level} risk)"
        
        # FIXME comments
        elif task.category == "fixme" and self.AUTO_APPROVE_FIXME:
            safe, _ = self.agent_layer.is_safe_to_modify(task.file_path)
            if safe:
                can_auto_approve = True
                auto_reason = "FIXME resolution in safe file"
        
        # Refactoring
        elif task.category in ("refactor", "code_quality") and self.AUTO_APPROVE_SAFE_REFACTOR:
            safe, _ = self.agent_layer.is_safe_to_modify(task.file_path)
            if safe and task.risk_level == "low":
                can_auto_approve = True
                auto_reason = f"safe refactor ({task.risk_level} risk)"
        
        # Low risk with high confidence
        elif task.risk_level == "low" and task.confidence > 0.85:
            can_auto_approve = True
            auto_reason = f"low risk, high confidence ({task.confidence:.0%})"
        
        # Security issues - always require review but auto-approve safe file fixes
        elif task.category == "security":
            safe, _ = self.agent_layer.is_safe_to_modify(task.file_path)
            if safe and task.confidence > 0.9:
                can_auto_approve = True
                auto_reason = "security fix in safe file"
        
        # Log auto-approval decision
        if can_auto_approve:
            logger.info(f"Auto-approving task {task.id}: {auto_reason}")
            if self._notify_callback:
                try:
                    await self._notify_callback({
                        "type": "auto_approval",
                        "task_id": task.id,
                        "reason": auto_reason,
                        "file": task.file_path,
                        "category": task.category,
                    })
                except Exception as e:
                    logger.warning(f"Failed to record auto-approval log: {e}")
            return True
        
        # Check permission callback for non-auto-approved tasks
        if self._permission_callback:
            return await self._permission_callback(task)

        # Default: require permission for medium/high risk
        return task.risk_level == "low"

    async def _execute_task(self, task: RefinementTask) -> tuple[bool, str]:
        """Execute a refinement task."""
        logger.info(f"Executing task {task.id}: {task.title}")
        with self._lock:
            task.status = "executing"
            task.executed_at = datetime.now()

        try:
            file_path = self.base_path / task.file_path

            if not file_path.exists():
                logger.warning(f"Task {task.id}: File not found: {task.file_path}")
                return False, f"File not found: {task.file_path}"

            # Backup original
            original = await asyncio.to_thread(file_path.read_text)
            with self._lock:
                task.rollback_code = original
            
            # Populate original_code from file if missing or contains markers
            if not task.original_code or ">>>" in task.original_code:
                # Try to get context from file using line number if available
                line_num = getattr(task, 'line_number', 0)
                if line_num == 0 and hasattr(task, 'description') and 'line' in task.description:
                    import re
                    match = re.search(r'line (\d+)', task.description)
                    if match:
                        line_num = int(match.group(1))
                if line_num > 0:
                    context = self.agent_layer.get_raw_context(task.file_path, line_num, 5)
                    if context:
                        task.original_code = context
                        logger.debug(f"Task {task.id}: Populated original_code from file (line {line_num})")

            # Generate fix if not provided
            if not task.proposed_code:
                logger.debug(f"Task {task.id}: Generating fix...")
                task.proposed_code = await self._generate_fix(task, original)

            if not task.proposed_code:
                logger.warning(f"Task {task.id}: Could not generate fix")
                with self._lock:
                    task.status = "failed"
                    task.result = "Could not generate fix"
                self._save()
                return False, "Could not generate fix"

            # Apply the change — use single replacement, not global replace
            if task.original_code and task.original_code in original:
                new_content = original.replace(task.original_code, task.proposed_code, 1)
            elif task.category == "documentation":
                new_content = original + "\n\n" + task.proposed_code
            else:
                logger.warning(f"Task {task.id}: Cannot apply fix - no valid original_code to replace")
                logger.debug(f"Task {task.id}: original_code length={len(task.original_code)}, in_content={task.original_code[:100] in original if task.original_code else False}")
                with self._lock:
                    task.status = "failed"
                    task.result = "Cannot apply fix: no valid original_code to replace"
                self._save()
                return False, "Cannot apply fix: no valid original_code to replace"

            await asyncio.to_thread(file_path.write_text, new_content)

            # Run tests if required
            if task.test_required:
                test_success, test_output = await self._run_tests(task.file_path)
                if not test_success:
                    # Try self-debugging before giving up
                    logger.info(f"Tests failed for {task.id}, attempting self-debug...")
                    debug_success, debug_result = await self.self_debug(task.id, test_output)
                    
                    if debug_success:
                        # Debug fixed it, verify tests pass now
                        test_success, test_output = await self._run_tests(task.file_path)
                        if test_success:
                            task.status = "completed"
                            task.result = f"Successfully applied refinement after debugging: {debug_result}"
                            with self._lock:
                                self._completed_tasks.append(task)
                                if len(self._completed_tasks) > self.MAX_COMPLETED_TASKS:
                                    self._completed_tasks = self._completed_tasks[-self.MAX_COMPLETED_TASKS:]
                                if task.id in self._tasks:
                                    del self._tasks[task.id]
                            return True, task.result
                    
                    # Still failing, rollback
                    if file_path.exists():
                        await asyncio.to_thread(file_path.write_text, original)
                    with self._lock:
                        task.status = "failed"
                        task.result = f"Tests failed after debug attempts: {test_output[:200]}"
                    return False, task.result

            with self._lock:
                task.status = "completed"
                task.result = "Successfully applied refinement"
            logger.info(f"Task {task.id}: Successfully applied refinement")

            # Move to completed
            with self._lock:
                self._completed_tasks.append(task)
                if len(self._completed_tasks) > self.MAX_COMPLETED_TASKS:
                    self._completed_tasks = self._completed_tasks[-self.MAX_COMPLETED_TASKS:]
                if task.id in self._tasks:
                    del self._tasks[task.id]
            self._save()

            return True, task.result

        except Exception as e:
            with self._lock:
                task.status = "failed"
                task.result = safe_error_message(e)
            logger.error(f"Task {task.id} failed: {e}")
            self._save()
            return False, safe_error_message(e)

    async def _generate_fix(self, task: RefinementTask, content: str) -> str:
        """Generate a fix for the task using LLM."""
        if task.category == "documentation":
            if "missing docstring" in task.description.lower():
                return await self._generate_docstring(task, content)

        elif task.category == "performance":
            if "range(len(" in task.original_code or "range(len(" in task.description:
                proposed = self._fix_range_len(task, content)
                if proposed:
                    return proposed

        prompt = f"""Generate a minimal, safe fix for this code issue.

File: {task.file_path}
Category: {task.category}
Issue: {task.title}
Description: {task.description}

Current code around the issue:
```
{task.original_code[:500]}
```

Provide only the fixed code, no explanations. Keep changes minimal.
"""
        try:
            return await self_refine_fix(
                issue=f"{task.title}: {task.description}\nCode:\n{task.original_code[:500]}",
                context={"file_path": task.file_path, "category": task.category},
            )
        except Exception:
            if self.llm:
                try:
                    response = await self._llm_call(
                        user_input=prompt,
                        system_prompt="You are an expert Python developer. Generate minimal, safe code fixes.",
                        priority="refinement",
                    )
                    return get_response_content(response)
                except Exception as e:
                    logger.error(f"LLM fix generation failed: {e}")

        return ""

    async def _generate_docstring(self, task: RefinementTask, content: str) -> str:
        """Generate a docstring by analyzing the actual code structure."""
        try:
            name = task.title.split(": ")[1].split(" in ")[0]
        except (IndexError, ValueError):
            name = "item"

        doc_type = "class" if "class" in task.description.lower() else "function"

        parsed = self._parse_code_element(content, name, doc_type)
        if parsed:
            if doc_type == "function":
                return self._build_function_docstring(name, parsed)
            else:
                return self._build_class_docstring(name, parsed)

        if self.llm:
            return await self._llm_docstring(task, content, name, doc_type)

        raise RuntimeError(f"Cannot generate docstring for {name}: no LLM available and code parsing failed")

    def _fix_range_len(self, task: RefinementTask, content: str) -> str | None:
        """Fix range(len(x)) pattern to enumerate(x) or direct iteration.
        
        Returns the replacement for the original_code context block, not the full file.
        """
        import re
        
        if not task.original_code:
            return None
        
        context = task.original_code
        lines = context.split('\n')
        modified = False
        
        for i, line in enumerate(lines):
            # Match: for i in range(len(seq)) or for i in range(len(self.seq))
            # Handles: range(len(var)), range(len(self.var)), range(len(var)-1), etc.
            match = re.search(r'for\s+(\w+)\s+in\s+range\(len\(([\w.]+)\)([^)]*)\)', line)
            if match:
                idx_var = match.group(1)
                seq_var = match.group(2)  # Could be 'var' or 'self.var'
                offset = match.group(3).strip()  # e.g., "-1" or ""
                
                # For simple range(len(seq)) without offset, convert to enumerate
                if not offset:
                    # Use a safe variable name based on the sequence
                    safe_name = seq_var.replace('.', '_').replace('self_', '')
                    new_line = re.sub(
                        r'for\s+\w+\s+in\s+range\(len\([\w.]+\)\):',
                        f'for {idx_var}, item_{safe_name} in enumerate({seq_var}):',
                        line
                    )
                    
                    if new_line != line:
                        lines[i] = new_line
                        modified = True
                        
                        # Update following lines if they use seq[idx_var]
                        for j in range(i + 1, min(i + 5, len(lines))):
                            if f'{seq_var}[{idx_var}]' in lines[j]:
                                lines[j] = lines[j].replace(f'{seq_var}[{idx_var}]', f'item_{safe_name}')
        
        if modified:
            return '\n'.join(lines)
        return None

    def _parse_code_element(self, content: str, name: str, doc_type: str) -> dict | None:
        """Parse code to extract function/class structure."""
        try:
            import ast
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if doc_type == "function" and isinstance(node, ast.FunctionDef) and node.name == name:
                    return {
                        "args": [arg.arg for arg in node.args.args if arg.arg != "self"],
                        "defaults": [ast.unparse(d) if isinstance(d, ast.AST) else str(d) 
                                    for d in node.args.defaults],
                        "returns": ast.unparse(node.returns) if node.returns else None,
                        "docstring": ast.get_docstring(node) or "",
                    }
                elif doc_type == "class" and isinstance(node, ast.ClassDef) and node.name == name:
                    attributes = []
                    for item in node.body:
                        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                            attributes.append({
                                "name": item.target.id,
                                "type": ast.unparse(item.annotation) if item.annotation else "Any"
                            })
                    return {
                        "attributes": attributes,
                        "methods": [n.name for n in node.body if isinstance(n, ast.FunctionDef)],
                        "docstring": ast.get_docstring(node) or "",
                    }
        except Exception as e:
            logger.warning(f"Failed to parse class definition: {e}")
        return None

    def _build_function_docstring(self, name: str, parsed: dict) -> str:
        """Build a docstring from parsed function data."""
        lines = [f"{name} - "]
        
        if parsed["args"]: 
            lines.append("")
            lines.append("Args:")
            for arg in parsed["args"]:
                lines.append(f"    {arg}: Description needed.")

        if parsed["returns"]:
            lines.append("")
            lines.append("Returns:")
            lines.append(f"    {parsed['returns']}: Description needed.")
        
        lines.append("")
        lines.append('"""')
        return '"""' + "\n".join(lines)

    def _build_class_docstring(self, name: str, parsed: dict) -> str:
        """Build a docstring from parsed class data."""
        lines = [f"{name} - "]
        
        if parsed["attributes"]:
            lines.append("")
            lines.append("Attributes:")
            for attr in parsed["attributes"]:
                lines.append(f"    {attr['name']} ({attr['type']}): Description needed.")
        
        lines.append("")
        lines.append('"""')
        return '"""' + "\n".join(lines)

    async def _llm_docstring(self, task: RefinementTask, content: str, name: str, doc_type: str) -> str:
        """Generate docstring using LLM with code context."""
        prompt = f"""Generate a comprehensive docstring for this {doc_type}.

Name: {name}
Original code context:
```
{content[:500]}
```

Generate a docstring that:
1. Describes the purpose clearly
2. Documents all parameters/attributes
3. Documents return values for functions
4. Uses Google-style docstring format

Output ONLY the docstring (including the triple quotes), no other text."""

        response = await self._llm_call(
            user_input=prompt,
            system_prompt="You are a Python documentation expert. Generate clear, accurate docstrings.",
            priority="refinement",
        )
        docstring = get_response_content(response)
        if docstring:
            return docstring
        raise RuntimeError(f"LLM failed to generate docstring for {name}")

    async def _run_tests(self, file_path: str) -> tuple[bool, str]:
        """Run tests for a modified file."""
        import os

        test_path = None
        try:
            test_file = file_path.replace("core/", "tests/test_").replace(".py", ".py")
            test_path = self.base_path / test_file

            if not test_path.exists():
                return True, "No tests for this file"

            env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "pytest", str(test_path), "-v", "--tb=short",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.base_path),
                env=env,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return False, "Tests timed out"

            return proc.returncode == 0, stdout.decode() + stderr.decode()

        except Exception as e:
            return False, safe_error_message(e)

    async def self_debug(self, task_id: str, error_output: str) -> tuple[bool, str]:
        """Attempt to self-debug a failed task.
        
        This method analyzes the error, identifies root cause, and attempts
        to generate and apply a fix.
        
        Args:
            task_id: The failed task ID
            error_output: Error message/traceback from the failure
            
        Returns:
            Tuple of (success, message)
        """
        with self._lock:
            task = self._tasks.get(task_id) or self._find_completed_task(task_id)
            if not task:
                return False, f"Task {task_id} not found"
            
            self._debug_counter += 1
            debug_id = f"debug_{self._debug_counter:06d}"
            
            # Identify error type
            error_type = self._identify_error_type(error_output)
            
            # Create debug session
            debug_session = DebugSession(
                id=debug_id,
                task_id=task_id,
                started_at=datetime.now(),
                error_type=error_type,
                error_message=error_output[:500],
                error_traceback=error_output,
            )
            self._debug_sessions[debug_id] = debug_session
            if len(self._debug_sessions) > self.MAX_DEBUG_SESSIONS:
                # Remove oldest sessions
                oldest_keys = sorted(self._debug_sessions.keys())[:len(self._debug_sessions) - self.MAX_DEBUG_SESSIONS]
                for k in oldest_keys:
                    del self._debug_sessions[k]
        
        self._state = RefinementState.DEBUGGING
        logger.info(f"Starting self-debug for task {task_id}: {error_type}")
        
        try:
            # Extract error context
            await self._extract_error_context(debug_session, error_output)
            
            # Check learned fixes first
            with self._lock:
                learned_fix = self._learned_fixes.get(error_type)
            
            if learned_fix:
                debug_session.fixes_tried.append(f"learned:{learned_fix[:50]}")
                
                success, result = await self._apply_debug_fix(
                    debug_session, task, learned_fix, is_learned=True
                )
                if success:
                    debug_session.success = True
                    debug_session.solution = learned_fix
                    return True, f"Applied learned fix: {result}"
            
            # Try auto-diagnosis
            diagnosis = await self._diagnose_error(debug_session, task, error_output)
            debug_session.root_cause = diagnosis.get("root_cause")
            
            # Generate fix using LLM
            while debug_session.attempts < debug_session.max_attempts:
                debug_session.attempts += 1
                
                fix = await self._generate_debug_fix(
                    debug_session, task, error_output, diagnosis
                )
                
                if not fix:
                    continue
                
                debug_session.fixes_tried.append(fix[:100])
                
                success, result = await self._apply_debug_fix(
                    debug_session, task, fix
                )
                
                if success:
                    debug_session.success = True
                    debug_session.fix_applied = fix
                    debug_session.solution = result
                    debug_session.resolved_at = datetime.now()
                    
                    # Learn this fix for future use
                    with self._lock:
                        self._learned_fixes[error_type] = fix
                        if len(self._learned_fixes) > self.MAX_LEARNED_FIXES:
                            oldest_keys = list(self._learned_fixes.keys())[:len(self._learned_fixes) - self.MAX_LEARNED_FIXES]
                            for k in oldest_keys:
                                del self._learned_fixes[k]
                    
                    return True, f"Debug successful: {result}"
                
                # Update diagnosis based on what didn't work
                diagnosis = await self._diagnose_error(
                    debug_session, task, error_output, previous_fix=fix
                )
            
            return False, f"Could not debug after {debug_session.attempts} attempts"
            
        except Exception as e:
            logger.error(f"Self-debug error: {e}")
            return False, f"Debug error: {e}"
        finally:
            self._state = RefinementState.IDLE
            self._save()

    def _find_completed_task(self, task_id: str) -> RefinementTask | None:
        """Find a task in completed tasks."""
        for task in self._completed_tasks:
            if task.id == task_id:
                return task
        return None

    def _identify_error_type(self, error_output: str) -> str:
        """Identify the type of error from output."""
        for error_type, config in self.ERROR_PATTERNS.items():
            if re.search(config["pattern"], error_output, re.IGNORECASE):
                return error_type
        return "unknown"

    async def _extract_error_context(self, session: DebugSession, error_output: str):
        """Extract file path and line number from error."""
        # Python traceback pattern
        traceback_pattern = r'File "([^"]+)", line (\d+)'
        matches = list(re.finditer(traceback_pattern, error_output))
        
        if matches:
            # Get the last match (actual error location)
            last_match = matches[-1]
            session.file_path = last_match.group(1)
            session.line_number = int(last_match.group(2))
        
        # Also check for test file references
        test_pattern = r'(tests/\S+\.py)'
        test_match = re.search(test_pattern, error_output)
        if test_match and not session.file_path:
            session.file_path = test_match.group(1)

    async def _diagnose_error(
        self, 
        session: DebugSession, 
        task: RefinementTask,
        error_output: str,
        previous_fix: str | None = None
    ) -> dict:
        """Diagnose the error and identify root cause."""
        diagnosis = {
            "error_type": session.error_type,
            "root_cause": None,
            "suggestions": [],
        }
        
        # Get handler for error type
        error_type = session.error_type
        if error_type in self.ERROR_PATTERNS:
            handler_name = self.ERROR_PATTERNS[error_type]["handler"]
            handler = getattr(self, handler_name, None)
            if handler:
                diagnosis = await handler(session, task, error_output)
        
        # If no specific handler or diagnosis incomplete, use LLM
        if not diagnosis.get("root_cause") and self.llm:
            diagnosis = await self._llm_diagnose(session, task, error_output, previous_fix)
        
        return diagnosis

    async def _llm_diagnose(
        self,
        session: DebugSession,
        task: RefinementTask,
        error_output: str,
        previous_fix: str | None = None
    ) -> dict:
        """Use LLM to diagnose the error."""
        code_context = ""
        if session.file_path:
            file_path = self.base_path / session.file_path
            if file_path.exists():
                content = await asyncio.to_thread(file_path.read_text)
                lines = content.split("\n")
                start = max(0, (session.line_number or 1) - 10)
                end = min(len(lines), (session.line_number or 1) + 10)
                code_context = "\n".join(
                    f"{i+1}: {line}" for i, line in enumerate(lines[start:end], start=start)
                )
        
        prompt = f"""Analyze this error and provide a diagnosis.

Error Type: {session.error_type}
Error Output:
```
{error_output[:2000]}
```

Code Context (from {session.file_path}):
```
{code_context}
```

Task: {task.title}
Previous Fix Attempted: {previous_fix or "None"}

Provide:
1. Root cause (one sentence)
2. Suggested fix (the exact code change needed)
3. Explanation (why this fix will work)

Format as JSON:
{{"root_cause": "...", "suggested_fix": "...", "explanation": "..."}}
"""
        
        try:
            diagnosis_result = await self_refine_analyze(
                code=f"Error: {error_output[:2000]}\n\nCode:\n{code_context}",
                context={
                    "error_type": session.error_type,
                    "file_path": session.file_path,
                    "task": task.title,
                    "previous_fix": previous_fix,
                },
            )
            if diagnosis_result:
                try:
                    json_match = re.search(r'\{[^{}]+\}', diagnosis_result, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group(0))
                except Exception as e:
                    logger.warning(f"Failed to parse diagnosis JSON: {e}")
        except Exception as e:
            logger.debug(f"Dual model diagnose failed: {e}")
        
        try:
            response = await self._llm_call(
                user_input=prompt,
                system_prompt="You are an expert Python debugger. Analyze errors and provide precise fixes.",
                priority="refinement",
            )
            content = get_response_content(response)
            
            json_match = re.search(r'\{[^{}]+\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
        except Exception as e:
            logger.debug(f"LLM diagnosis failed: {e}")
        
        return {"root_cause": "Unknown", "suggested_fix": "", "explanation": ""}

    async def _generate_debug_fix(
        self,
        session: DebugSession,
        task: RefinementTask,
        error_output: str,
        diagnosis: dict
    ) -> str:
        """Generate a fix for the debug session."""
        # Use suggested fix from diagnosis if available
        if diagnosis.get("suggested_fix"):
            return diagnosis["suggested_fix"]
        
        # Check if we have a learned fix
        if session.error_type in self._learned_fixes:
            return self._learned_fixes[session.error_type]
        
        # Generate specific fix based on error type
        handler_name = f"_fix_{session.error_type}"
        handler = getattr(self, handler_name, None)
        if handler:
            return await handler(session, task, error_output)
        
        return ""

    async def _apply_debug_fix(
        self,
        session: DebugSession,
        task: RefinementTask,
        fix: str,
        is_learned: bool = False
    ) -> tuple[bool, str]:
        """Apply a debug fix and verify it works."""
        if not session.file_path:
            return False, "No file path identified"
        
        file_path = self.base_path / session.file_path
        if not file_path.exists():
            return False, f"File not found: {session.file_path}"
        
        try:
            original = await asyncio.to_thread(file_path.read_text)
            
            # Apply the fix
            if session.line_number:
                lines = original.split("\n")
                line_idx = session.line_number - 1
                
                if 0 <= line_idx < len(lines):
                    # Insert or replace fix at line
                    if fix.startswith("++"):  # Insert after
                        lines.insert(line_idx + 1, fix[2:])
                    elif fix.startswith("--"):  # Delete line
                        del lines[line_idx]
                    elif fix.startswith("->"):  # Replace with
                        lines[line_idx] = fix[2:]
                    else:
                        # Default: replace the line
                        lines[line_idx] = fix
                    
                    new_content = "\n".join(lines)
                else:
                    return False, f"Line number out of range: {session.line_number}"
            else:
                # No line number - append fix or find pattern
                new_content = original + "\n" + fix
            
            # Write the fix
            await asyncio.to_thread(file_path.write_text, new_content)
            
            # Run tests or syntax check
            if "test" in session.file_path.lower():
                # It's a test file, just check syntax
                try:
                    compile(new_content, session.file_path, 'exec')
                    return True, "Syntax check passed"
                except SyntaxError as e:
                    await asyncio.to_thread(file_path.write_text, original)
                    return False, f"Syntax error after fix: {e}"
            else:
                # Run tests
                test_success, test_output = await self._run_tests(session.file_path)
                if test_success:
                    # Also check if module can be imported
                    try:
                        proc = await asyncio.create_subprocess_exec(
                            sys.executable, "-c", f"import {session.file_path.replace('/', '.').replace('.py', '')}",
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                            cwd=str(self.base_path),
                        )
                        try:
                            stdout, stderr_data = await asyncio.wait_for(proc.communicate(), timeout=10)
                        except asyncio.TimeoutError:
                            proc.kill()
                            await proc.wait()
                            raise TimeoutError("Import check timed out")
                        if proc.returncode != 0:
                            await asyncio.to_thread(file_path.write_text, original)
                            return False, f"Import failed: {stderr_data.decode()}"
                    except TimeoutError:
                        await asyncio.to_thread(file_path.write_text, original)
                        return False, "Import check timed out"
                    except Exception as e:
                        await asyncio.to_thread(file_path.write_text, original)
                        return False, f"Import check failed: {e}"
                    
                    return True, "Tests passed"
                else:
                    # Rollback
                    await asyncio.to_thread(file_path.write_text, original)
                    return False, f"Tests still failing: {test_output[:200]}"
                    
        except Exception as e:
            return False, f"Failed to apply fix: {e}"

    # Specific error handlers
    async def _debug_import_error(
        self, session: DebugSession, task: RefinementTask, error_output: str
    ) -> dict:
        """Debug import errors."""
        diagnosis = {"root_cause": None, "suggested_fix": None, "suggestions": []}
        
        # Parse missing module
        import_match = re.search(r"No module named '([^']+)'|cannot import name '([^']+)'", error_output)
        if import_match:
            missing = import_match.group(1) or import_match.group(2)
            diagnosis["root_cause"] = f"Missing or incorrect import: {missing}"
            diagnosis["suggestions"].append(f"Check if '{missing}' is spelled correctly")
            diagnosis["suggestions"].append(f"Check if '{missing}' module/package is installed")
            
            # Check if it's a local import issue
            if session.file_path and "cannot import name" in error_output:
                # Look for the import in the file
                file_path = self.base_path / session.file_path
                if file_path.exists():
                    content = file_path.read_text(encoding='utf-8')
                    import_lines = [l for l in content.split("\n") if "import" in l and missing in l]
                    if import_lines:
                        diagnosis["suggestions"].append(f"Check import statement: {import_lines[0]}")
        
        return diagnosis

    async def _debug_attribute_error(
        self, session: DebugSession, task: RefinementTask, error_output: str
    ) -> dict:
        """Debug attribute errors."""
        diagnosis = {"root_cause": None, "suggested_fix": None, "suggestions": []}
        
        # Parse missing attribute
        attr_match = re.search(r"'(\w+)' object has no attribute '(\w+)'", error_output)
        if attr_match:
            obj_type = attr_match.group(1)
            attr_name = attr_match.group(2)
            diagnosis["root_cause"] = f"Object of type '{obj_type}' has no attribute '{attr_name}'"
            diagnosis["suggestions"].append(f"Check if '{attr_name}' is a valid attribute for {obj_type}")
            diagnosis["suggestions"].append("Consider using hasattr() to check before access")
        
        return diagnosis

    async def _debug_type_error(
        self, session: DebugSession, task: RefinementTask, error_output: str
    ) -> dict:
        """Debug type errors."""
        diagnosis = {"root_cause": None, "suggested_fix": None, "suggestions": []}
        
        # Parse type error details
        if "unsupported operand" in error_output:
            diagnosis["root_cause"] = "Incompatible types in operation"
        elif "missing required positional argument" in error_output:
            diagnosis["root_cause"] = "Function called with wrong number of arguments"
        elif "not callable" in error_output:
            diagnosis["root_cause"] = "Attempting to call a non-callable object"
        elif "NoneType" in error_output:
            diagnosis["root_cause"] = "Operation on None value - check for null/None"
        
        return diagnosis

    async def _debug_key_error(
        self, session: DebugSession, task: RefinementTask, error_output: str
    ) -> dict:
        """Debug key errors."""
        diagnosis = {"root_cause": None, "suggested_fix": None, "suggestions": []}
        
        # Parse missing key
        key_match = re.search(r"KeyError: '?([^'\n]+)'?", error_output)
        if key_match:
            missing_key = key_match.group(1)
            diagnosis["root_cause"] = f"Dictionary missing key: '{missing_key}'"
            diagnosis["suggestions"].append(f"Check if key '{missing_key}' exists before access")
            diagnosis["suggestions"].append(f"Use dict.get('{missing_key}', default) for safe access")
        
        return diagnosis

    async def _debug_index_error(
        self, session: DebugSession, task: RefinementTask, error_output: str
    ) -> dict:
        """Debug index errors."""
        diagnosis: dict[str, Any] = {"root_cause": None, "suggested_fix": None, "suggestions": []}
        diagnosis["root_cause"] = "List/sequence index out of range"
        diagnosis["suggestions"] = ["Check list length before indexing", "Use bounds checking"]
        return diagnosis

    async def _debug_value_error(
        self, session: DebugSession, task: RefinementTask, error_output: str
    ) -> dict:
        """Debug value errors."""
        diagnosis: dict[str, Any] = {"root_cause": None, "suggested_fix": None, "suggestions": []}
        diagnosis["root_cause"] = "Invalid value passed to operation"
        diagnosis["suggestions"] = ["Validate input values", "Check for edge cases"]
        return diagnosis

    async def _debug_name_error(
        self, session: DebugSession, task: RefinementTask, error_output: str
    ) -> dict:
        """Debug name errors."""
        diagnosis = {"root_cause": None, "suggested_fix": None, "suggestions": []}
        
        # Parse undefined name
        name_match = re.search(r"name '(\w+)' is not defined", error_output)
        if name_match:
            undefined_name = name_match.group(1)
            diagnosis["root_cause"] = f"Undefined variable/function: '{undefined_name}'"
            diagnosis["suggestions"].append(f"Define '{undefined_name}' before use")
            diagnosis["suggestions"].append(f"Check for typo in '{undefined_name}'")
        
        return diagnosis

    async def _debug_syntax_error(
        self, session: DebugSession, task: RefinementTask, error_output: str
    ) -> dict:
        """Debug syntax errors."""
        diagnosis = {"root_cause": None, "suggested_fix": None, "suggestions": []}
        
        # Parse syntax error
        if "unexpected indent" in error_output.lower():
            diagnosis["root_cause"] = "Indentation error - check for extra spaces/tabs"
        elif "expected" in error_output.lower():
            diagnosis["root_cause"] = "Missing syntax element (colon, parenthesis, etc.)"
        else:
            diagnosis["root_cause"] = "Python syntax error"
        
        diagnosis["suggestions"].append("Check indentation matches Python conventions")
        diagnosis["suggestions"].append("Verify all brackets/parentheses are closed")
        
        return diagnosis

    async def _debug_test_failure(
        self, session: DebugSession, task: RefinementTask, error_output: str
    ) -> dict:
        """Debug test failures."""
        diagnosis = {"root_cause": None, "suggested_fix": None, "suggestions": []}
        
        # Parse assertion error
        assert_match = re.search(r"assert\s+(.+?)(?:\n|$)", error_output)
        if assert_match:
            diagnosis["root_cause"] = f"Assertion failed: {assert_match.group(1)[:50]}"
        
        # Check for comparison operators
        if "==" in error_output or "!=" in error_output:
            diagnosis["suggestions"].append("Check expected vs actual values")
        
        diagnosis["suggestions"].append("Review test logic and expected outcomes")
        
        return diagnosis

    async def _debug_timeout_error(
        self, session: DebugSession, task: RefinementTask, error_output: str
    ) -> dict:
        """Debug timeout errors."""
        diagnosis: dict[str, Any] = {"root_cause": None, "suggested_fix": None, "suggestions": []}
        diagnosis["root_cause"] = "Operation timed out - possible infinite loop or slow operation"
        diagnosis["suggestions"] = [
            "Check for infinite loops",
            "Optimize slow operations",
            "Increase timeout if operation is expected to be slow"
        ]
        return diagnosis

    # Fix generators
    async def _fix_import_error(
        self, session: DebugSession, task: RefinementTask, error_output: str
    ) -> str:
        """Generate fix for import error."""
        if not session.file_path:
            return ""
        
        # Try to find and fix the import
        import_match = re.search(r"No module named '([^']+)'", error_output)
        if import_match:
            missing = import_match.group(1)
            # Check for typos
            common_typos = {
                "numpy": "np",
                "pandas": "pd",
                "matplotlib.pyplot": "plt",
            }
            if missing in common_typos:
                # This might be an alias issue
                return f"# Check if you meant to import as {common_typos[missing]}"
        
        return ""

    def get_debug_stats(self) -> dict:
        """Get debugging statistics."""
        with self._lock:
            total = len(self._debug_sessions)
            successful = sum(1 for s in self._debug_sessions.values() if s.success)
            by_type = defaultdict(int)
            for session in self._debug_sessions.values():
                by_type[session.error_type] += 1
            
            return {
                "total_debug_sessions": total,
                "successful_debugs": successful,
                "success_rate": successful / total if total > 0 else 0,
                "by_error_type": dict(by_type),
                "learned_fixes": len(self._learned_fixes),
            }

    def get_recent_debug_sessions(self, limit: int = 10) -> list[dict]:
        """Get recent debug sessions for display."""
        with self._lock:
            sessions = sorted(
                self._debug_sessions.values(),
                key=lambda s: s.started_at,
                reverse=True
            )[:limit]
            
            return [
                {
                    "id": s.id,
                    "task_id": s.task_id,
                    "error_type": s.error_type,
                    "error_message": s.error_message[:100],
                    "file_path": s.file_path,
                    "line_number": s.line_number,
                    "attempts": s.attempts,
                    "success": s.success,
                    "root_cause": s.root_cause,
                    "solution": s.solution[:100] if s.solution else None,
                }
                for s in sessions
            ]

    async def _process_learnings(self):
        """Process learnings from this session."""
        # Record insights to intuition bridge
        try:
            from .intuition_bridge import get_intuition_bridge, InsightType
            bridge = get_intuition_bridge()

            if self._session:
                # Record session outcome
                bridge.record_insight(
                    insight_type=InsightType.LEARNING,
                    source="self_refine",
                    content=f"Refinement session completed: {self._session.tasks_succeeded}/{self._session.tasks_executed} tasks succeeded",
                    confidence=0.8,
                    impact=0.6,
                    metadata={
                        "session_id": self._session.id,
                        "files_analyzed": self._session.files_analyzed,
                        "issues_found": self._session.issues_found,
                        "tasks_generated": self._session.tasks_generated,
                    },
                )

                # Record failures as strategy failures
                if self._session.tasks_failed > 0:
                    bridge.record_insight(
                        insight_type=InsightType.STRATEGY_FAILURE,
                        source="self_refine",
                        content=f"{self._session.tasks_failed} refinement tasks failed",
                        confidence=0.7,
                        impact=0.5,
                    )

        except Exception as e:
            logger.debug(f"Failed to record learnings: {e}")

    def rollback_task(self, task_id: str) -> tuple[bool, str]:
        """Rollback a completed task."""
        with self._lock:
            for task in self._completed_tasks:
                if task.id == task_id:
                    if not task.rollback_code:
                        return False, "No rollback code available"
                    task.status = "rolled_back"
                    rollback_code = task.rollback_code
                    file_path = self.base_path / task.file_path
                    break
            else:
                return False, f"Task {task_id} not found"

        try:
            file_path.write_text(rollback_code, encoding='utf-8')
            return True, f"Rolled back {task.file_path}"
        except Exception as e:
            with self._lock:
                for task in self._completed_tasks:
                    if task.id == task_id:
                        task.status = "failed"
                        break
            return False, safe_error_message(e)

    def add_intuition_task(self, insight_type: str, content: str, target: str, priority: int = 3) -> str:
        """Add a task from the intuition bridge."""
        with self._lock:
            self._task_counter += 1
            task_id = f"task_{self._task_counter:06d}"

            priority_map = {
                1: RefinementPriority.CRITICAL,
                2: RefinementPriority.HIGH,
                3: RefinementPriority.MEDIUM,
                4: RefinementPriority.LOW,
                5: RefinementPriority.ENHANCEMENT,
            }

            task = RefinementTask(
                id=task_id,
                title=f"Intuition: {content[:50]}",
                description=content,
                file_path=target,
                priority=priority_map.get(priority, RefinementPriority.MEDIUM),
                category="intuition",
                original_code="",
                proposed_code="",
                reasoning=f"Generated from intuition: {insight_type}",
                risk_level="medium",
                source="intuition",
            )

            self._tasks[task_id] = task
            self._save()

            return task_id

    def _load(self):
        """Load state from storage."""
        state_file = self.storage_path / "self_refine_state.json"
        if state_file.exists():
            try:
                data = read_json_with_integrity(state_file)

                for task_data in data.get("pending_tasks", []):
                    task = RefinementTask(
                        id=task_data["id"],
                        title=task_data["title"],
                        description=task_data["description"],
                        file_path=task_data["file_path"],
                        priority=RefinementPriority(task_data["priority"]),
                        category=task_data["category"],
                        original_code=task_data.get("original_code", ""),
                        proposed_code=task_data.get("proposed_code", ""),
                        reasoning=task_data.get("reasoning", ""),
                        risk_level=task_data.get("risk_level", "medium"),
                        status=task_data.get("status", "pending"),
                        impact_score=task_data.get("impact_score", 0.5),
                        confidence=task_data.get("confidence", 0.5),
                        source=task_data.get("source", "auto"),
                    )
                    self._tasks[task.id] = task
                    self._task_counter = max(self._task_counter, int(task.id.split("_")[1]))

                logger.info(f"Loaded {len(self._tasks)} pending tasks")
                
                # Reset stuck tasks: any task in "executing" state wasn't completed
                # and should be retried
                stuck_count = 0
                for task in self._tasks.values():
                    if task.status == "executing":
                        task.status = "pending"
                        task.executed_at = None
                        stuck_count += 1
                if stuck_count > 0:
                    logger.info(f"Reset {stuck_count} stuck tasks from 'executing' to 'pending'")

            except Exception as e:
                logger.error(f"Failed to load self-refine state: {e}")

    def _save(self):
        """Save state to storage."""
        state_file = self.storage_path / "self_refine_state.json"
        try:
            data = {
                "pending_tasks": [
                    {
                        "id": t.id,
                        "title": t.title,
                        "description": t.description,
                        "file_path": t.file_path,
                        "priority": t.priority.value,
                        "category": t.category,
                        "original_code": t.original_code,
                        "proposed_code": t.proposed_code,
                        "reasoning": t.reasoning,
                        "risk_level": t.risk_level,
                        "status": t.status,
                        "impact_score": t.impact_score,
                        "confidence": t.confidence,
                        "source": t.source,
                    }
                    for t in self._tasks.values()
                ],
                "completed_count": len(self._completed_tasks),
                "last_save": datetime.now().isoformat(),
            }
            atomic_write_json(state_file, data)
        except Exception as e:
            logger.error(f"Failed to save self-refine state: {e}")


# Singleton
_self_refine_agent: Optional[SelfRefineAgent] = None
_agent_lock = threading.Lock()


def get_self_refine_agent(llm_client=None) -> SelfRefineAgent:
    """Get the self-refine agent singleton."""
    global _self_refine_agent
    if _self_refine_agent is None:
        with _agent_lock:
            if _self_refine_agent is None:
                if llm_client is None:
                    try:
                        from core.llm import OllamaClient
                        llm_client = OllamaClient()
                    except Exception as e:
                        logger.warning(f"Could not create default LLM client: {e}")
                        raise ValueError("LLM client required for first initialization")
                _self_refine_agent = SelfRefineAgent(llm_client)
    return _self_refine_agent


def has_self_refine_agent() -> bool:
    """Check if self-refine agent is initialized."""
    return _self_refine_agent is not None
