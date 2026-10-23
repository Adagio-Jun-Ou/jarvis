"""Process Sandbox Manager — Isolation and resource limits for tool execution.

Provides layered sandboxing for Jarvis tool execution:
- Resource limits (CPU time, memory, file size, process count)
- Output size caps (prevent log flooding)
- Directory confinement (chroot-like path restriction)
- Temporary workspace provisioning with auto-cleanup
- Per-sandbox resource accounting
- Timeout enforcement with process group kill

This is critical for 24/7 operation — a runaway subprocess can't be
allowed to consume all system resources.
"""
import logging
import os
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Any

from core.helpers import safe_error_message

# POSIX-only. The rlimit-based isolation below degrades to timeout + output
# caps on platforms without it (notably Windows), rather than making the whole
# sandbox module unimportable.
try:
    import resource
except ImportError:  # pragma: no cover - platform dependent
    resource = None

HAS_RLIMIT = resource is not None

logger = logging.getLogger(__name__)


@dataclass
class SandboxLimits:
    cpu_seconds: float = 60.0
    memory_mb: int = 0
    max_file_size_mb: int = 0
    max_processes: int = 0
    max_output_bytes: int = 5 * 1024 * 1024
    timeout_seconds: float = 120.0
    allowed_paths: list[str] | None = None
    blocked_commands: list[str] | None = None
    env_override: dict[str, str] | None = None

    def to_dict(self) -> dict:
        return {
            "cpu_seconds": self.cpu_seconds,
            "memory_mb": self.memory_mb,
            "max_file_size_mb": self.max_file_size_mb,
            "max_processes": self.max_processes,
            "max_output_bytes": self.max_output_bytes,
            "timeout_seconds": self.timeout_seconds,
            "allowed_paths": self.allowed_paths,
        }


@dataclass
class SandboxResult:
    exit_code: int
    stdout: str
    stderr: str
    duration: float
    timed_out: bool
    memory_peak_mb: float
    sandbox_id: str
    command: str
    limits_applied: bool
    output_truncated: bool
    killed_reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "exit_code": self.exit_code,
            "stdout_len": len(self.stdout),
            "stderr_len": len(self.stderr),
            "duration": round(self.duration, 3),
            "timed_out": self.timed_out,
            "memory_peak_mb": round(self.memory_peak_mb, 2),
            "sandbox_id": self.sandbox_id,
            "limits_applied": self.limits_applied,
            "output_truncated": self.output_truncated,
            "killed_reason": self.killed_reason,
        }


@dataclass
class SandboxInfo:
    sandbox_id: str
    command: str
    started_at: float
    limits: SandboxLimits
    status: str = "running"
    pid: int = 0
    ended_at: float | None = None

    @property
    def duration(self) -> float:
        end = self.ended_at or time.time()
        return end - self.started_at

    def to_dict(self) -> dict:
        return {
            "sandbox_id": self.sandbox_id,
            "command": self.command[:200],
            "started_at": self.started_at,
            "duration": round(self.duration, 2),
            "status": self.status,
            "pid": self.pid,
            "limits": self.limits.to_dict(),
        }


class SandboxManager:
    """Manages isolated execution environments for tool commands.

    Each sandbox gets:
    1. Resource limits (via ulimit / resource module)
    2. A temporary working directory (auto-cleaned)
    3. Output size caps
    4. Process count limits
    5. Command blocklist enforcement
    6. Timeout with process group kill

    Thread-safe. Tracks all active sandboxes.
    """

    DEFAULT_LIMITS = SandboxLimits()

    BLOCKED_COMMANDS = frozenset({
        "rm -rf /", "mkfs", "dd if=/dev/zero", "fork bomb",
        ":(){:|:&};:", "chmod -R 777 /", "chown -R",
        "wget http", "curl http:// | sh", "curl -sSf | bash",
        "pip install --user", "npm install -g",
    })

    BLOCKED_PATTERNS = [
        (lambda c: "rm" in c and "-rf" in c and "/" in c and ("&&" in c or "|" in c or ";" in c), "destructive rm pipeline"),
        (lambda c: "curl" in c and ("| sh" in c or "| bash" in c or "| sudo" in c), "piped remote execution"),
        (lambda c: "wget" in c and ("| sh" in c or "| bash" in c), "piped remote execution"),
        (lambda c: "dd" in c and ("of=/dev/" in c or "if=/dev/zero" in c), "destructive dd"),
        (lambda c: ":()" in c and ":|:&" in c, "fork bomb"),
        (lambda c: "chmod" in c and "777" in c and "/" in c, "dangerous chmod"),
        (lambda c: "mkfs" in c, "filesystem format"),
        (lambda c: "> /dev/sd" in c, "direct device write"),
    ]

    MAX_ACTIVE_SANDBOXES = 50
    MAX_HISTORY = 200
    CLEANUP_INTERVAL = 60.0

    def __init__(self):
        self._lock = threading.RLock()
        self._active: dict[str, SandboxInfo] = {}
        self._history: list[SandboxInfo] = []
        self._total_executions = 0
        self._total_timeouts = 0
        self._total_killed = 0
        self._total_output_truncated = 0
        self._start_time = time.time()
        self._sandbox_counter = 0

    def _next_id(self) -> str:
        with self._lock:
            self._sandbox_counter += 1
            return f"sandbox_{self._sandbox_counter}_{int(time.time())}"

    def _check_command_blocked(self, command: str) -> str | None:
        cmd_lower = command.strip().lower()
        for blocked in self.BLOCKED_COMMANDS:
            if blocked.lower() in cmd_lower:
                return f"Blocked command pattern: {blocked}"
        for pattern_check, reason in self.BLOCKED_PATTERNS:
            try:
                if pattern_check(command):
                    return f"Blocked: {reason}"
            except Exception as e:
                logger.debug(f"Blocked pattern check failed: {e}")
        return None

    def _check_path_allowed(self, command: str, limits: SandboxLimits) -> str | None:
        if not limits.allowed_paths:
            return None
        import re
        path_patterns = [r'(?:cd|ls|cat|head|tail|grep|find|touch|mkdir|cp|mv)\s+([^\s;&|]+)']
        for pattern in path_patterns:
            for match in re.finditer(pattern, command):
                target = match.group(1)
                if not any(target.startswith(p) for p in limits.allowed_paths):
                    return f"Path not in allowed list: {target}"
        return None

    def create_sandbox(self, command: str, limits: SandboxLimits | None = None) -> tuple[str, str | None]:
        with self._lock:
            if len(self._active) >= self.MAX_ACTIVE_SANDBOXES:
                return "", "Too many active sandboxes"

        if limits is None:
            limits = self.DEFAULT_LIMITS

        block_reason = self._check_command_blocked(command)
        if block_reason:
            return "", block_reason

        path_reason = self._check_path_allowed(command, limits)
        if path_reason:
            return "", path_reason

        sandbox_id = self._next_id()
        info = SandboxInfo(
            sandbox_id=sandbox_id,
            command=command,
            started_at=time.time(),
            limits=limits,
        )
        with self._lock:
            self._active[sandbox_id] = info
            self._total_executions += 1

        return sandbox_id, None

    def execute(self, command: str, limits: SandboxLimits | None = None, cwd: str | None = None, env: dict[str, str] | None = None) -> SandboxResult:
        sandbox_id, error = self.create_sandbox(command, limits)
        if error:
            return SandboxResult(
                exit_code=-1, stdout="", stderr=error, duration=0,
                timed_out=False, memory_peak_mb=0, sandbox_id="",
                command=command, limits_applied=False, output_truncated=False,
                killed_reason=error,
            )

        if limits is None:
            limits = self.DEFAULT_LIMITS

        info = self._active.get(sandbox_id)
        if not info:
            return SandboxResult(
                exit_code=-1, stdout="", stderr="Sandbox not found", duration=0,
                timed_out=False, memory_peak_mb=0, sandbox_id=sandbox_id,
                command=command, limits_applied=False, output_truncated=False,
            )

        tmpdir = tempfile.mkdtemp(prefix=f"jarvis_sandbox_{sandbox_id}_")
        work_dir = cwd or tmpdir
        run_env = dict(os.environ)
        if env:
            run_env.update(env)
        if limits.env_override:
            run_env.update(limits.env_override)
        run_env["HOME"] = tmpdir
        run_env["TMPDIR"] = tmpdir

        if not os.path.isdir(work_dir):
            work_dir = tmpdir

        timed_out = False
        output_truncated = False
        killed_reason = None
        start = time.time()
        pid = 0

        try:
            proc = subprocess.Popen(
                command,
                shell=True,
                cwd=work_dir,
                env=run_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
                preexec_fn=self._set_limits(limits) if os.name != "nt" else None,
            )
            pid = proc.pid
            info.pid = pid

            try:
                stdout_bytes, stderr_bytes = proc.communicate(timeout=limits.timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                killed_reason = "timeout"
                proc.kill()
                try:
                    # Process groups are POSIX-only; proc.kill() above already
                    # covers the direct child on other platforms.
                    if hasattr(os, "killpg") and hasattr(os, "getpgid"):
                        os.killpg(os.getpgid(pid), signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass
                try:
                    stdout_bytes, stderr_bytes = proc.communicate(timeout=5)
                except Exception as e:
                    logger.debug(f"Process communicate after kill failed: {e}")
                    stdout_bytes, stderr_bytes = b"", b""

            if len(stdout_bytes) > limits.max_output_bytes:
                stdout_bytes = stdout_bytes[:limits.max_output_bytes]
                output_truncated = True
                self._total_output_truncated += 1

            if len(stderr_bytes) > limits.max_output_bytes // 2:
                stderr_bytes = stderr_bytes[:limits.max_output_bytes // 2]
                output_truncated = True

            stdout_str = stdout_bytes.decode("utf-8", errors="replace")
            stderr_str = stderr_bytes.decode("utf-8", errors="replace")

            memory_peak = 0
            if proc.returncode is not None and proc.returncode < 0:
                if abs(proc.returncode) == signal.SIGKILL or abs(proc.returncode) == 9:
                    if not timed_out:
                        killed_reason = "memory_limit"
                        self._total_killed += 1

        except Exception as e:
            stdout_str = ""
            stderr_str = safe_error_message(e)
            proc = type('Proc', (), {'returncode': -1})()
            output_truncated = False
            memory_peak = 0

        finally:
            info.status = "completed"
            info.ended_at = time.time()
            info.pid = 0
            with self._lock:
                self._active.pop(sandbox_id, None)
                self._history.append(info)
                if len(self._history) > self.MAX_HISTORY:
                    self._history = self._history[-self.MAX_HISTORY:]

            try:
                import shutil
                shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception as e:
                logger.debug(f"Tmpdir cleanup failed: {e}")
        if timed_out:
            self._total_timeouts += 1

        return SandboxResult(
            exit_code=proc.returncode if hasattr(proc, 'returncode') else -1,
            stdout=stdout_str,
            stderr=stderr_str,
            duration=time.time() - start,
            timed_out=timed_out,
            memory_peak_mb=memory_peak,
            sandbox_id=sandbox_id,
            command=command,
            limits_applied=True,
            output_truncated=output_truncated,
            killed_reason=killed_reason,
        )

    def _set_limits(self, limits: SandboxLimits):
        if not HAS_RLIMIT:
            return None

        def _apply():
            try:
                if limits.cpu_seconds > 0:
                    resource.setrlimit(resource.RLIMIT_CPU, (int(limits.cpu_seconds), int(limits.cpu_seconds)))
            except (ValueError, OSError):
                pass
            try:
                if limits.memory_mb > 0:
                    mem_bytes = limits.memory_mb * 1024 * 1024
                    resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
            except (ValueError, OSError):
                pass
            try:
                if limits.max_file_size_mb > 0:
                    file_bytes = limits.max_file_size_mb * 1024 * 1024
                    resource.setrlimit(resource.RLIMIT_FSIZE, (file_bytes, file_bytes))
            except (ValueError, OSError):
                pass
            try:
                if limits.max_processes > 0 and limits.max_processes < 100:
                    resource.setrlimit(resource.RLIMIT_NPROC, (limits.max_processes, limits.max_processes))
            except (ValueError, OSError):
                pass
        return _apply

    def kill_sandbox(self, sandbox_id: str) -> bool:
        with self._lock:
            info = self._active.get(sandbox_id)
            if not info or info.pid == 0:
                return False
            pid = info.pid
        try:
            if hasattr(os, "killpg") and hasattr(os, "getpgid"):
                # POSIX: kill the whole process group so grandchildren die too.
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            else:
                # Windows has no process groups here; terminate the child directly.
                os.kill(pid, signal.SIGTERM)
            self._total_killed += 1
            info.status = "killed"
            info.ended_at = time.time()
            return True
        except (ProcessLookupError, OSError):
            return False

    def get_active_sandbox(self, sandbox_id: str) -> SandboxInfo | None:
        with self._lock:
            return self._active.get(sandbox_id)

    def get_active_sandboxes(self) -> list[dict]:
        with self._lock:
            return [info.to_dict() for info in self._active.values()]

    def get_history(self, limit: int = 50) -> list[dict]:
        with self._lock:
            return [info.to_dict() for info in self._history[-limit:]]

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "active_sandboxes": len(self._active),
                "total_executions": self._total_executions,
                "total_timeouts": self._total_timeouts,
                "total_killed": self._total_killed,
                "total_output_truncated": self._total_output_truncated,
                "history_size": len(self._history),
                "uptime_seconds": round(time.time() - self._start_time, 1),
                "max_active": self.MAX_ACTIVE_SANDBOXES,
            }

    def cleanup_stale(self, max_age_seconds: float = 600) -> int:
        now = time.time()
        killed = 0
        with self._lock:
            stale_ids = []
            for sid, info in self._active.items():
                if now - info.started_at > max_age_seconds:
                    stale_ids.append(sid)
        for sid in stale_ids:
            if self.kill_sandbox(sid):
                killed += 1
        return killed


_sandbox_manager: SandboxManager | None = None
_sandbox_lock = threading.Lock()


def get_sandbox_manager() -> SandboxManager:
    global _sandbox_manager
    if _sandbox_manager is None:
        with _sandbox_lock:
            if _sandbox_manager is None:
                _sandbox_manager = SandboxManager()
    return _sandbox_manager