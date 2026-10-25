"""Execution Context — Persistent runtime state for Jarvis agent operations.

Provides a living execution environment that persists across conversation turns:
- Working directory tracking (follows cd, pushd, popd)
- Environment variable management
- Process lifecycle tracking
- File operation audit log
- Session state persistence (save/load)
- Health heartbeat and crash recovery

This is the "body" that lets Jarvis actually LIVE on the computer —
remembering where it is, what it was doing, and what files it touched.
"""
import logging
import os
import platform
import re
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.helpers import atomic_write_json, path_is_within, read_json_with_integrity, safe_error_message

logger = logging.getLogger(__name__)

_CD_PATTERN = re.compile(r'^cd\s+([^\s;&|]+)')
_COMPOUND_CD_PATTERN = re.compile(r'&&\s*cd\s+([^\s;&|]+)')
_PUSHD_PATTERN = re.compile(r'^pushd\s+([^\s;&|]+)')
_SECRET_PATTERNS = ("SECRET", "PASSWORD", "TOKEN", "KEY", "CREDENTIAL", "PRIVATE")
_PROTECTED_DIRS = frozenset({"/etc", "/root", "/boot", "/sys", "/proc", "/dev", "/sbin", "/bin"})
_PROTECTED_DIRS_WRITE = frozenset({"/etc", "/root", "/boot", "/sys", "/proc", "/dev", "/sbin", "/bin", "/usr/bin", "/usr/sbin"})
_BLOCKED_EXTENSIONS = frozenset({".exe", ".dll", ".so", ".dylib", ".bin", ".pem", ".key", ".p12", ".pfx"})


@dataclass
class ProcessRecord:
    pid: int
    command: str
    started_at: float
    working_dir: str
    exit_code: int | None = None
    ended_at: float | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""

    def to_dict(self) -> dict:
        return {
            "pid": self.pid,
            "command": self.command,
            "started_at": self.started_at,
            "working_dir": self.working_dir,
            "exit_code": self.exit_code,
            "ended_at": self.ended_at,
            "stdout_tail": self.stdout_tail[:500],
            "stderr_tail": self.stderr_tail[:500],
        }

    @property
    def is_running(self) -> bool:
        return self.exit_code is None

    @property
    def duration_seconds(self) -> float:
        end = self.ended_at or time.time()
        return end - self.started_at


@dataclass
class FileOperation:
    op_type: str  # read, write, create, delete, move, copy
    path: str
    timestamp: float
    size_bytes: int = 0
    success: bool = True
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "op_type": self.op_type,
            "path": self.path,
            "timestamp": self.timestamp,
            "size_bytes": self.size_bytes,
            "success": self.success,
            "error": self.error,
        }


class ExecutionContext:
    """Persistent execution context that lives across conversation turns.

    This is the key difference between a chatbot and an agent:
    a chatbot forgets where it is; an agent remembers.
    
    Usage:
        ctx = get_execution_context()
        ctx.cd("/home/user/jarvis")
        result = ctx.run("ls -la")
        print(ctx.cwd)  # Still /home/user/jarvis next turn
    
    Performance optimizations (v2.1):
    - Cached run environment (invalidated on env/cwd change)
    - Module-level compiled regex patterns
    - Fine-grained locking to reduce contention
    - Lazy env build for subprocess (no redaction overhead)
    """

    MAX_PROCESS_HISTORY = 200
    MAX_FILE_LOG = 500
    MAX_ENV_SIZE = 100

    def __init__(self, workspace_root: Path | None = None):
        self._lock = threading.RLock()
        self._cwd = str(workspace_root or Path.home())
        self._env: dict[str, str] = {}
        self._process_history: list[ProcessRecord] = []
        self._file_log: list[FileOperation] = []
        self._session_start = time.time()
        self._last_activity = time.time()
        self._turn_count = 0
        self._command_count = 0
        self._total_output_bytes = 0
        self._health_check_interval = 60.0
        self._running_processes: dict[int, subprocess.Popen] = {}
        self._workspace_root = str(workspace_root or Path.home())
        self._state_path = Path.home() / ".jarvis" / "execution_context.json"
        self._persist_interval = 30.0
        self._last_persist = 0.0
        self._loaded = False
        self._run_env_cache: dict[str, str] | None = None
        self._run_env_cache_dirty = True

        os.makedirs(self._state_path.parent, exist_ok=True)

    def load_state(self) -> None:
        """Load persisted execution context from disk."""
        if self._loaded:
            return
        try:
            data = read_json_with_integrity(self._state_path)
            if data:
                self._cwd = data.get("cwd", self._cwd)
                saved_env = data.get("env", {})
                self._env.update({k: v for k, v in saved_env.items() if isinstance(v, str)})
                self._command_count = data.get("command_count", 0)
                self._total_output_bytes = data.get("total_output_bytes", 0)
                logger.info(f"ExecutionContext loaded: cwd={self._cwd}, commands={self._command_count}")
        except Exception as e:
            logger.debug(f"Could not load execution context: {e}")
        finally:
            self._loaded = True

    def save_state(self) -> None:
        """Persist execution context to disk."""
        now = time.time()
        if now - self._last_persist < self._persist_interval:
            return
        try:
            data = {
                "cwd": self._cwd,
                "env": dict(self._env),
                "command_count": self._command_count,
                "total_output_bytes": self._total_output_bytes,
                "last_activity": self._last_activity,
                "version": 2,
            }
            atomic_write_json(self._state_path, data)
            self._last_persist = now
        except Exception as e:
            logger.debug(f"Could not save execution context: {e}")

    @property
    def cwd(self) -> str:
        return self._cwd

    def cd(self, path: str) -> tuple[bool, str]:
        """Change working directory. Returns (success, new_cwd_or_error)."""
        with self._lock:
            expanded = os.path.expanduser(path)
            if not os.path.isabs(expanded):
                expanded = os.path.normpath(os.path.join(self._cwd, expanded))

            expanded = os.path.realpath(expanded)

            if not os.path.isdir(expanded):
                return False, f"Directory not found: {expanded}"

            # path_is_within, not startswith: a bare prefix test also denies
            # legitimate siblings such as /etcetera, /devops or /bind.
            if any(path_is_within(expanded, p) for p in _PROTECTED_DIRS):
                return False, f"Access denied: {expanded}"

            old_cwd = self._cwd
            self._cwd = expanded
            self._run_env_cache_dirty = True
            self._last_activity = time.time()
            self.save_state()

        try:
            from core.event_bus import get_event_bus
            get_event_bus().publish("context.cwd_changed", {
                "old_cwd": old_cwd,
                "new_cwd": expanded,
            }, source="execution_context")
        except Exception as e:
            logger.debug(f"CWD change event publish failed: {e}")

        return True, self._cwd

    def set_env(self, key: str, value: str) -> None:
        """Set an environment variable for subsequent commands."""
        with self._lock:
            if len(self._env) >= self.MAX_ENV_SIZE:
                oldest_key = next(iter(self._env))
                del self._env[oldest_key]
            self._env[key] = str(value)
            self._run_env_cache_dirty = True
            self._last_activity = time.time()
            self.save_state()

    def get_env(self, key: str, default: str | None = None) -> str | None:
        """Get an environment variable (context env overrides os env)."""
        with self._lock:
            if key in self._env:
                return self._env[key]
            return os.environ.get(key, default)

    def get_full_env(self) -> dict[str, str]:
        """Get merged environment (os env + context overrides) with secret redaction."""
        env = dict(os.environ)
        with self._lock:
            env.update(self._env)
            env["PWD"] = self._cwd
        safe_env = {}
        for k, v in env.items():
            if any(p in k.upper() for p in _SECRET_PATTERNS):
                safe_env[k] = "***REDACTED***"
            else:
                safe_env[k] = v
        return safe_env

    def _get_run_env(self) -> dict[str, str]:
        """Get merged environment for subprocess execution (cached, no redaction)."""
        if not self._run_env_cache_dirty and self._run_env_cache is not None:
            return self._run_env_cache

        env = dict(os.environ)
        with self._lock:
            env.update(self._env)
            env["PWD"] = self._cwd
        self._run_env_cache = env
        self._run_env_cache_dirty = False
        return env

    def run(self, command: str, timeout: float = 60.0, cwd: str | None = None) -> dict[str, Any]:
        """Execute a shell command in the execution context.

        Returns dict with: stdout, stderr, exit_code, duration, timed_out
        """
        with self._lock:
            self._turn_count += 1
            self._command_count += 1
            self._last_activity = time.time()
            work_dir = cwd or self._cwd
            if not os.path.isdir(work_dir):
                work_dir = str(Path.home())
                self._cwd = work_dir

        env = self._get_run_env()

        record = ProcessRecord(
            pid=0,
            command=command,
            started_at=time.time(),
            working_dir=work_dir,
        )

        timed_out = False
        try:
            proc = subprocess.Popen(
                command,
                shell=True,
                cwd=work_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            record.pid = proc.pid

            with self._lock:
                self._running_processes[proc.pid] = proc

            try:
                stdout_bytes, stderr_bytes = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                proc.kill()
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass
                stdout_bytes, stderr_bytes = proc.communicate(timeout=5)

            record.exit_code = proc.returncode
            record.ended_at = time.time()
            record.stdout_tail = stdout_bytes.decode("utf-8", errors="replace")[-2000:]
            record.stderr_tail = stderr_bytes.decode("utf-8", errors="replace")[-1000:]

            total_size = len(stdout_bytes) + len(stderr_bytes)
            with self._lock:
                self._total_output_bytes += total_size
                self._process_history.append(record)
                if len(self._process_history) > self.MAX_PROCESS_HISTORY:
                    self._process_history = self._process_history[-self.MAX_PROCESS_HISTORY:]

            self._track_cwd_change(command, work_dir)

        except Exception as e:
            record.exit_code = -1
            record.ended_at = time.time()
            record.stderr_tail = safe_error_message(e)
            with self._lock:
                self._process_history.append(record)
            return {
                "stdout": "",
                "stderr": safe_error_message(e),
                "exit_code": -1,
                "duration": record.duration_seconds,
                "timed_out": timed_out,
                "command": command,
                "cwd": work_dir,
            }
        finally:
            with self._lock:
                self._running_processes.pop(record.pid, None)

        self.save_state()

        return {
            "stdout": record.stdout_tail,
            "stderr": record.stderr_tail,
            "exit_code": record.exit_code,
            "duration": round(record.duration_seconds, 3),
            "timed_out": timed_out,
            "command": command,
            "cwd": work_dir,
            "pid": record.pid,
        }

    def _track_cwd_change(self, command: str, current_cwd: str) -> None:
        """Track CWD changes from shell commands (cd, pushd, compound commands)."""
        cmd_stripped = command.strip()

        cd_match = _CD_PATTERN.match(cmd_stripped)
        if cd_match:
            self.cd(cd_match.group(1).strip())
            return

        compound_cd = _COMPOUND_CD_PATTERN.search(cmd_stripped)
        if compound_cd:
            self.cd(compound_cd.group(1).strip())
            return

        pushd_match = _PUSHD_PATTERN.match(cmd_stripped)
        if pushd_match:
            self.cd(pushd_match.group(1).strip())

    def log_file_op(self, op_type: str, path: str, size: int = 0, success: bool = True, error: str | None = None) -> None:
        """Log a file operation."""
        with self._lock:
            self._file_log.append(FileOperation(
                op_type=op_type, path=path, timestamp=time.time(),
                size_bytes=size, success=success, error=error,
            ))
            if len(self._file_log) > self.MAX_FILE_LOG:
                self._file_log = self._file_log[-self.MAX_FILE_LOG:]

    def read_file(self, path: str, max_bytes: int = 10 * 1024 * 1024, start_line: int = 0, end_line: int = 0) -> dict[str, Any]:
        """Read a file from the execution context's workspace.

        Resolves relative paths against cwd. Returns file content or error.
        """
        expanded = self._resolve_path(path)
        if not expanded:
            return {"error": f"Path not accessible: {path}", "success": False}

        try:
            p = Path(expanded)
            if not p.exists():
                self.log_file_op("read", path, success=False, error="not found")
                return {"error": f"File not found: {path}", "success": False}
            if p.is_dir():
                return {
                    "content": "\n".join(f"{item.name}/" if item.is_dir() else item.name for item in sorted(p.iterdir())),
                    "path": str(p),
                    "is_dir": True,
                    "success": True,
                }

            size = p.stat().st_size
            if size > max_bytes:
                self.log_file_op("read", path, size, success=False, error="too large")
                return {"error": f"File too large: {size} bytes (max {max_bytes})", "success": False}

            raw = p.read_bytes()
            if b"\x00" in raw[:8192]:
                self.log_file_op("read", path, size, success=True)
                return {"error": "Binary file", "path": str(p), "size": size, "success": False}

            content = raw.decode("utf-8", errors="replace")

            if start_line > 0 or end_line > 0:
                lines = content.splitlines()
                start = max(0, start_line - 1)
                end = end_line if end_line > 0 else len(lines)
                content = "\n".join(f"{i+1}: {lines[i]}" for i in range(start, min(end, len(lines))))

            self.log_file_op("read", path, size, success=True)
            return {
                "content": content,
                "path": str(p),
                "size": size,
                "lines": content.count("\n") + 1,
                "success": True,
            }
        except Exception as e:
            self.log_file_op("read", path, success=False, error=safe_error_message(e))
            return {"error": safe_error_message(e), "success": False}

    def write_file(self, path: str, content: str, mode: str = "write") -> dict[str, Any]:
        """Write a file within the execution context.

        Args:
            path: File path (relative to cwd or absolute)
            content: Content to write
            mode: 'write', 'append', or 'prepend'

        Returns dict with success status, path, and size.
        """
        expanded = self._resolve_path(path, for_write=True)
        if not expanded:
            return {"error": f"Path not writable: {path}", "success": False}

        p = Path(expanded)
        if p.suffix.lower() in _BLOCKED_EXTENSIONS:
            return {"error": f"Blocked file extension: {p.suffix}", "success": False}

        try:
            p.parent.mkdir(parents=True, exist_ok=True)

            if mode == "append" and p.exists():
                existing = p.read_text(encoding="utf-8", errors="replace")
                content = existing + "\n" + content
            elif mode == "prepend" and p.exists():
                existing = p.read_text(encoding="utf-8", errors="replace")
                content = content + "\n" + existing

            tmp_path = p.with_suffix(p.suffix + ".tmp")
            tmp_path.write_text(content, encoding="utf-8")
            tmp_path.replace(p)

            size = p.stat().st_size
            self.log_file_op("write", path, size, success=True)
            return {"path": str(p), "size": size, "mode": mode, "success": True}
        except Exception as e:
            self.log_file_op("write", path, success=False, error=safe_error_message(e))
            return {"error": safe_error_message(e), "success": False}

    def _resolve_path(self, path: str, for_write: bool = False) -> str | None:
        """Resolve a path, checking access permissions."""
        expanded = os.path.expanduser(path)
        if not os.path.isabs(expanded):
            expanded = os.path.normpath(os.path.join(self._cwd, expanded))

        expanded = os.path.realpath(expanded)

        protected = _PROTECTED_DIRS_WRITE if for_write else _PROTECTED_DIRS
        # path_is_within, not startswith: a bare prefix test also denies
        # legitimate siblings such as /etcetera, /devops or /bind.
        if any(path_is_within(expanded, p) for p in protected):
            return None

        return expanded

    def kill_process(self, pid: int) -> bool:
        """Kill a tracked process."""
        with self._lock:
            proc = self._running_processes.get(pid)
            if proc:
                try:
                    os.killpg(os.getpgid(pid), signal.SIGTERM)
                    time.sleep(0.5)
                    try:
                        os.killpg(os.getpgid(pid), signal.SIGKILL)
                    except (ProcessLookupError, OSError):
                        pass
                    return True
                except (ProcessLookupError, OSError):
                    return False
        return False

    def get_running_processes(self) -> list[dict]:
        """Get list of currently running processes."""
        with self._lock:
            return [
                {"pid": pid, "command": p.args if hasattr(p, 'args') else str(p)}
                for pid, p in self._running_processes.items()
            ]

    def get_recent_commands(self, limit: int = 20) -> list[dict]:
        """Get recent command history."""
        with self._lock:
            return [r.to_dict() for r in self._process_history[-limit:]]

    def get_recent_files(self, limit: int = 20) -> list[dict]:
        """Get recent file operations."""
        with self._lock:
            return [f.to_dict() for f in self._file_log[-limit:]]

    def get_status(self) -> dict[str, Any]:
        """Get comprehensive execution context status."""
        with self._lock:
            uptime = time.time() - self._session_start
            return {
                "cwd": self._cwd,
                "env_count": len(self._env),
                "command_count": self._command_count,
                "turn_count": self._turn_count,
                "total_output_bytes": self._total_output_bytes,
                "running_processes": len(self._running_processes),
                "process_history_size": len(self._process_history),
                "file_log_size": len(self._file_log),
                "uptime_seconds": round(uptime, 1),
                "last_activity": self._last_activity,
                "platform": platform.system(),
                "hostname": platform.node(),
            }

    def heartbeat(self) -> dict[str, Any]:
        """Check health and return status. Called periodically."""
        with self._lock:
            self._last_activity = time.time()
            alive_procs = []
            dead_pids = []
            for pid, proc in list(self._running_processes.items()):
                if proc.poll() is not None:
                    dead_pids.append(pid)
                else:
                    alive_procs.append(pid)
            for pid in dead_pids:
                del self._running_processes[pid]

        self.save_state()
        return {
            "healthy": True,
            "cwd": self._cwd,
            "running": len(alive_procs),
            "commands_executed": self._command_count,
            "uptime_hours": round((time.time() - self._session_start) / 3600, 2),
        }

    def shutdown(self) -> None:
        """Graceful shutdown: kill orphan processes, save state."""
        with self._lock:
            for pid, proc in list(self._running_processes.items()):
                try:
                    proc.terminate()
                    proc.wait(timeout=3)
                except Exception as e:
                    logger.debug(f"Process terminate failed: {e}")
                    try:
                        proc.kill()
                    except Exception as e2:
                        logger.debug(f"Process kill failed: {e2}")
            self._running_processes.clear()
        self.save_state()
        logger.info(f"ExecutionContext shutdown: {self._command_count} commands, {self._turn_count} turns")


    def reset(self) -> None:
        """Reset the execution context (careful!)."""
        with self._lock:
            for pid, proc in list(self._running_processes.items()):
                try:
                    proc.kill()
                except Exception as e:
                    logger.debug(f"Process kill during reset failed: {e}")
            self._running_processes.clear()
            self._process_history.clear()
            self._file_log.clear()
            self._env.clear()
            self._cwd = self._workspace_root
            self._command_count = 0
            self._total_output_bytes = 0
            self._turn_count = 0
            self._last_activity = time.time()
            self._run_env_cache_dirty = True
        self.save_state()


_execution_context: ExecutionContext | None = None
_context_lock = threading.Lock()


def get_execution_context() -> ExecutionContext:
    """Thread-safe singleton for ExecutionContext."""
    global _execution_context
    if _execution_context is None:
        with _context_lock:
            if _execution_context is None:
                _execution_context = ExecutionContext()
                _execution_context.load_state()
                import atexit
                atexit.register(_execution_context.shutdown)
    return _execution_context