"""Native Terminal Access for Jarvis.

Like OpenCode, Jarvis has full terminal access with permission gating.
Jarvis has his own "computer" - a workspace where he can:
- Execute any shell command (with permission)
- Maintain persistent terminal sessions
- Install packages and run applications
- Work within a sandboxed workspace or full system access

The permission system ensures Jarvis asks before running commands.
"""
import asyncio
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Callable
from collections import deque

from .base import Tool
from core.helpers import safe_error_message

logger = logging.getLogger(__name__)


class TerminalMode(Enum):
    SANDBOX = "sandbox"      # Limited to workspace directory
    FULL = "full"           # Full system access
    PROJECT = "project"     # Current project directory


@dataclass
class TerminalSession:
    """A persistent terminal session."""
    id: str
    created_at: datetime
    cwd: str
    env: dict[str, str]
    history: deque = field(default_factory=lambda: deque(maxlen=1000))
    last_output: str = ""
    exit_code: int = 0
    is_running: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "cwd": self.cwd,
            "history_count": len(self.history),
            "last_output": self.last_output[:500] if self.last_output else "",
            "exit_code": self.exit_code,
            "is_running": self.is_running,
        }


@dataclass
class PermissionRequest:
    """A pending permission request."""
    id: str
    command: str
    reason: str
    risk_level: str  # "low", "medium", "high"
    created_at: datetime
    callback_id: str = ""


class JarvisComputer:
    """Jarvis's virtual computer - his workspace.
    
    This represents the computer Jarvis "owns" and can work within.
    He can install packages, run code, edit files, etc.
    """

    def __init__(
        self,
        workspace: str = None,
        mode: TerminalMode = TerminalMode.SANDBOX,
        auto_approve_safe: bool = False,
    ):
        self.workspace = Path(workspace or Path.home() / ".jarvis_workspace")
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.auto_approve_safe = auto_approve_safe

        self.sessions: dict[str, TerminalSession] = {}
        self.active_session: Optional[str] = None
        self._permission_callbacks: dict[str, Callable] = {}
        self._pending_permissions: dict[str, PermissionRequest] = {}

        # Background jobs (Claude Code run_in_background semantics):
        # tracked asyncio tasks with bounded retained output so the agent
        # can start long work, continue, and poll results later.
        self._background_jobs: dict[str, dict] = {}
        self._bg_counter = 0
        self.MAX_BACKGROUND_JOBS = 20
        self.BACKGROUND_TIMEOUT_CAP = 3600.0
        self._BG_OUTPUT_CAP = 100_000  # chars retained per stream

        self._init_default_session()

        self.safe_commands = {
            "ls", "cat", "head", "tail", "wc", "grep", "find", "echo",
            "pwd", "whoami", "date", "uname", "which", "env", "printenv",
            "git", "python", "python3", "pip", "pip3", "node", "npm", "npx",
            "mkdir", "touch", "cp", "mv", "chmod", "ln", "diff", "tree",
            "curl", "wget", "tar", "unzip", "zip", "less", "more",
            "code", "vim", "nano", "emacs", "subl",
            "pytest", "mypy", "ruff", "black", "isort", "flake8",
            "docker", "docker-compose",
            "cargo", "rustc", "go",
        }

        self.dangerous_commands = {
            # System destruction
            "rm -rf /", "rm -rf /*", "rm -rf ~", "rm -rf ~/",
            # Disk operations
            "mkfs", "dd if=/dev/", "> /dev/sd",
            # Fork bomb
            ":(){ :|:& };:",
            # Permission chaos
            "chmod -R 777 /", "chown -R root",
            # Rootkit patterns
            "curl | bash", "wget | bash", "curl | sh", "wget | sh",
            # Shell piping (even with safe commands as prefix)
            "| bash", "| sh", "| zsh", "| fish",
            "| sudo", "| su",
        }

    _SAFE_ENV_VARS = {
        'PATH', 'HOME', 'USER', 'SHELL', 'TERM', 'LANG', 'LC_ALL',
        'LC_CTYPE', 'TMPDIR', 'TEMP', 'TMP', 'DISPLAY', 'XDG_CONFIG_HOME',
        'XDG_DATA_HOME', 'XDG_CACHE_HOME', 'EDITOR', 'PAGER', 'TZ',
        'COLORTERM', 'LS_COLORS', 'HOSTNAME', 'MAIL', 'OLDPWD',
    }
    _BLOCKED_ENV_VARS = {
        'DOCKER_HOST', 'DOCKER_TLS_VERIFY', 'DOCKER_CERT_PATH', 'KUBECONFIG',
        'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_SESSION_TOKEN',
        'AWS_DEFAULT_REGION', 'OLLAMA_API_KEY', 'OPENAI_API_KEY',
        'GITHUB_TOKEN', 'GH_TOKEN', 'HEROKU_API_KEY',
    }

    def _safe_env(self) -> dict[str, str]:
        _SECRET_PATTERNS = ('KEY', 'SECRET', 'TOKEN', 'PASSWORD', 'CREDENTIAL', 'AUTH', 'PRIVATE')
        return {
            k: v for k, v in os.environ.items()
            if k in self._SAFE_ENV_VARS
            or (not k.startswith('_') and k not in self._BLOCKED_ENV_VARS
                and not any(s in k.upper() for s in _SECRET_PATTERNS))
        }

    def _init_default_session(self):
        session = TerminalSession(
            id="default",
            created_at=datetime.now(),
            cwd=str(self.workspace),
            env=self._safe_env(),
        )
        self.sessions["default"] = session
        self.active_session = "default"

    def get_cwd(self) -> str:
        if self.active_session and self.active_session in self.sessions:
            return self.sessions[self.active_session].cwd
        return str(self.workspace)

    _PROTECTED_PATHS = frozenset({
        "/etc", "/root", "/boot", "/proc", "/sys", "/dev",
        "/sbin", "/bin", "/usr/sbin", "/usr/bin", "/lib", "/lib64",
        "/var/run", "/var/log", "/var/lib", "/var/spool",
        "/opt", "/srv", "/usr/local", "/usr/lib",
        "/snap", "/run", "/tmp", "/lost+found",
    })

    def set_cwd(self, path: str) -> bool:
        try:
            new_cwd = Path(path).resolve()
            home_str = str(Path.home())
            for protected in self._PROTECTED_PATHS:
                if str(new_cwd) == protected or str(new_cwd).startswith(protected + "/"):
                    if str(new_cwd).startswith(home_str):
                        pass
                    else:
                        logger.warning(f"set_cwd blocked: protected directory {protected}")
                        return False
            if not new_cwd.exists():
                if not str(new_cwd).startswith(home_str):
                    logger.warning("set_cwd blocked: cannot create directories outside home")
                    return False
                new_cwd.mkdir(parents=True, exist_ok=True)
            if self.active_session and self.active_session in self.sessions:
                self.sessions[self.active_session].cwd = str(new_cwd)
            return True
        except Exception:
            return False

    def assess_risk(self, command: str) -> tuple[str, str]:
        cmd_lower = command.lower().strip()
        cmd_parts = cmd_lower.split()
        cmd_name = cmd_parts[0] if cmd_parts else ""

        # =============================================
        # CHECK SAFE OPERATIONS FIRST
        # =============================================
        
        # Command substitution patterns - always high risk
        if "$(" in command or "`" in command:
            return "high", "Command substitution detected (security risk)"
        
        # Recursive deletion - check target carefully
        if cmd_name in ("rm", "rmdir") and "-rf" in cmd_lower:
            target = cmd_parts[-1] if len(cmd_parts) >= 3 else ""
            
            # WILDCARD PATTERNS - HIGH RISK (check first)
            if target in ("/*", "/**", "/", "~/*", "~", "~/"):
                return "high", "Wildcard deletion of root or home directory"
            for sys_dir in ["/", "/home", "/etc", "/usr", "/root", "/opt", "/boot", "/lib", "/bin", "/sbin", "/var"]:
                if target == f"{sys_dir}/*" or target == f"{sys_dir}/**":
                    return "high", "Wildcard deletion of system directory contents"
            if target.startswith("~"):
                return "high", "Home directory deletion"
            
            # Check safe targets (MEDIUM risk)
            if target.startswith("/tmp") or target.startswith("/var/tmp"):
                return "medium", "Temp directory deletion"
            if target.startswith("./") or target.startswith("../"):
                return "medium", "Local directory deletion"
            if target and not target.startswith("/"):
                return "medium", "Relative directory deletion"
            
            safe_dirs = ["build", "dist", "node_modules", "__pycache__", ".git", "cache", "venv", ".venv", "target", "out"]
            for dirname in safe_dirs:
                if dirname in target.lower():
                    return "medium", "Local directory deletion"
            
            # Check dangerous targets (HIGH risk)
            if len(cmd_parts) < 3:
                return "high", "Recursive deletion without target"
            
            system_dirs = ["/", "/home", "/etc", "/usr", "/root", "/opt", "/boot", "/lib", "/bin", "/sbin", "/var"]
            for sys_dir in system_dirs:
                if cmd_lower.strip() == f"rm -rf {sys_dir}" or cmd_lower.strip().startswith(f"rm -rf {sys_dir}/"):
                    return "high", "System directory deletion"
            
            return "medium", "Recursive deletion - verify target"

        # =============================================
        # NOW CHECK DANGEROUS PATTERNS
        # =============================================
        
        for dangerous in self.dangerous_commands:
            if dangerous.lower() in cmd_lower:
                # Skip if it's a safe rm -rf path (already handled above)
                if "rm -rf" in cmd_lower:
                    target = cmd_parts[-1] if len(cmd_parts) >= 3 else ""
                    if target.startswith("/tmp") or target.startswith("/var/tmp"):
                        continue  # Already handled as MEDIUM above
                return "high", "Potentially destructive command detected"

        # Piped destructive commands
        if "| rm " in cmd_lower or "|rm " in cmd_lower:
            return "high", "Piped destructive command"
        
        # Remote script execution
        if ("curl" in cmd_lower or "wget" in cmd_lower) and ("| bash" in cmd_lower or "| sh" in cmd_lower):
            return "high", "Remote script execution"

        if cmd_name in ("sudo", "su", "doas"):
            return "high", "Elevated privileges requested"

        if cmd_name in ("apt", "apt-get", "yum", "dnf", "pacman", "snap", "flatpak"):
            if "install" in cmd_lower or "remove" in cmd_lower:
                return "medium", "Package manager operation"
            return "low", "Package manager query"

        if cmd_name in ("pip", "pip3", "npm", "yarn", "cargo", "go"):
            if "install" in cmd_lower or "uninstall" in cmd_lower:
                return "medium", f"Package installation via {cmd_name}"
            return "low", "Package manager query"

        if cmd_name in ("docker", "kubectl"):
            if any(x in cmd_lower for x in ["rm", "delete", "prune", "rmi"]):
                return "medium", "Container/cluster deletion"
            return "low", "Container orchestration"

        if cmd_name in ("curl", "wget") and ("| bash" in cmd_lower or "| sh" in cmd_lower):
            return "high", "Remote script execution"

        if cmd_name in self.safe_commands:
            return "low", f"Standard command: {cmd_name}"

        if "/" in cmd_name or cmd_name.endswith(".sh"):
            return "medium", "Script or binary execution"

        return "medium", f"Command requires review: {cmd_name}"

    AUTO_APPROVE_SAFE = True

    def needs_permission(self, command: str) -> bool:
        risk, _ = self.assess_risk(command)

        if not self.AUTO_APPROVE_SAFE:
            return True

        if risk == "low":
            return False

        return True

    async def execute(
        self,
        command: str,
        timeout: float = 60.0,
        session_id: Optional[str] = None,
    ) -> tuple[int, str, str]:
        """Execute a command and return (exit_code, stdout, stderr).
        
        Uses subprocess_exec for safety - shell metacharacters are automatically
        escaped. For complex commands with pipes/redirections, wraps in bash -c.
        """
        session = self.sessions.get(session_id or self.active_session or "default")
        if not session:
            session = self.sessions["default"]

        cwd = session.cwd if self.mode != TerminalMode.SANDBOX else str(self.workspace)

        session.history.append({
            "command": command,
            "timestamp": datetime.now().isoformat(),
            "cwd": cwd,
        })

        shell_chars = {'|', '>', '<', '&', ';', '$', '`', '\\', '"', "'", '\n', '\t'}
        needs_shell = any(c in command for c in shell_chars) or '&&' in command or '||' in command

        if needs_shell:
            cmd_lower = command.lower()
            BLOCKED_PIPE_PATTERNS = [
                '| bash', '| sh', '|zsh', '|fish',
                '| sudo', '| su ',
                '| curl', '| wget',
                'curl |', 'wget |',
            ]
            for pattern in BLOCKED_PIPE_PATTERNS:
                if pattern in cmd_lower:
                    return 1, "", f"Blocked: unsafe pipe pattern '{pattern}' detected"

            DANGEROUS_SUBS = ['$((', '${', '`']
            for sub in DANGEROUS_SUBS:
                if sub in command:
                    return 1, "", "Blocked: dangerous shell substitution detected"

            10 * 1024 * 1024

        try:
            if needs_shell:
                safe_cmd = ['/bin/bash', '-c', command]
                process = await asyncio.create_subprocess_exec(
                    *safe_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=cwd,
                    env=session.env,
                    start_new_session=True,
                )
            else:
                # Split safely without shell interpretation
                import shlex
                try:
                    args = shlex.split(command)
                except ValueError as e:
                    return 1, "", f"Invalid command syntax: {e}"
                
                if not args:
                    return 1, "", "Empty command"
                
                process = await asyncio.create_subprocess_exec(
                    args[0],
                    *args[1:],
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=cwd,
                    env=session.env,
                    start_new_session=True,
                )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                try:
                    import os, signal
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    process.kill()
                session.exit_code = -1
                session.last_output = f"Command timed out after {timeout}s"
                return -1, "", f"Command timed out after {timeout}s"

            stdout_str = stdout.decode("utf-8", errors="replace")
            stderr_str = stderr.decode("utf-8", errors="replace")

            MAX_OUTPUT = 10 * 1024 * 1024
            if len(stdout_str) > MAX_OUTPUT:
                stdout_str = stdout_str[:MAX_OUTPUT] + f"\n... [truncated at {MAX_OUTPUT} bytes]"
            if len(stderr_str) > MAX_OUTPUT:
                stderr_str = stderr_str[:MAX_OUTPUT] + f"\n... [truncated at {MAX_OUTPUT} bytes]"

            session.exit_code = process.returncode or 0
            session.last_output = stdout_str[:1000] if stdout_str else stderr_str[:1000]

            if process.returncode == 0:
                cwd_result = await self._check_cwd_change(command, cwd)
                if cwd_result:
                    session.cwd = cwd_result
                    try:
                        from core.execution_context import get_execution_context
                        get_execution_context().cd(cwd_result)
                    except Exception:
                        pass

            return process.returncode or 0, stdout_str, stderr_str

        except FileNotFoundError:
            session.exit_code = 127
            session.last_output = "Command not found"
            return 127, "", "Command not found"
        except Exception as e:
            session.exit_code = 1
            session.last_output = safe_error_message(e)
            return 1, "", safe_error_message(e)

    # ------------------------------------------------------------------ #
    # Background jobs (run_in_background semantics)
    # ------------------------------------------------------------------ #

    def start_background(self, command: str, timeout: float = 3600.0,
                         session_id: Optional[str] = None) -> tuple[bool, str]:
        """Start a command as a tracked background job.

        Reuses execute()'s full pipeline — blocked pipe patterns,
        substitution checks, session cwd/env, output caps — but returns
        immediately with a job id instead of waiting.

        The job runs on its own worker thread with a private event
        loop, NOT as a task on the caller's loop. A background job must
        outlive the request/turn that started it: under the engine
        adapter every tool call runs ``asyncio.run`` on a short-lived
        loop, whose shutdown cancels pending tasks (and could hang
        waiting for the subprocess). A private thread works from any
        caller — async or sync.

        Returns (started, job_id_or_error).
        """
        timeout = min(max(float(timeout), 1.0), self.BACKGROUND_TIMEOUT_CAP)
        self._bg_counter += 1
        job_id = f"bg-{self._bg_counter}"
        job = {
            "id": job_id,
            "command": command,
            "session": session_id or self.active_session or "default",
            "status": "running",
            "started_at": datetime.now().isoformat(),
            "finished_at": None,
            "exit_code": None,
            "stdout": "",
            "stderr": "",
            "thread": None,
        }
        self._register_background_job(job)

        def _worker():
            try:
                code, out, err = asyncio.run(self.execute(
                    command=command, timeout=timeout, session_id=session_id))
                job["exit_code"] = code
                job["stdout"] = out[: self._BG_OUTPUT_CAP]
                job["stderr"] = err[: self._BG_OUTPUT_CAP]
                job["status"] = "completed" if code == 0 else "failed"
            except Exception as e:
                job["status"] = "failed"
                job["exit_code"] = 1
                job["stderr"] = safe_error_message(e)
            finally:
                job["finished_at"] = datetime.now().isoformat()
                logger.info(f"Background job {job_id} finished (status={job['status']})")
                self._notify_background_completion(job)

        thread = threading.Thread(
            target=_worker, name=f"jarvis-bg-{job_id}", daemon=True)
        job["thread"] = thread
        thread.start()
        return True, job_id

    def _register_background_job(self, job: dict) -> None:
        self._background_jobs[job["id"]] = job
        # Bound the registry: evict oldest FINISHED jobs first; running
        # jobs are never evicted.
        while len(self._background_jobs) > self.MAX_BACKGROUND_JOBS:
            evicted = False
            for jid, j in list(self._background_jobs.items()):
                if j["status"] != "running":
                    del self._background_jobs[jid]
                    evicted = True
                    break
            if not evicted:
                break

    def get_background_job(self, job_id: str) -> Optional[dict]:
        return self._background_jobs.get(job_id)

    def _notify_background_completion(self, job: dict) -> None:
        """Wake-up channel: job finished -> attention item + event bus.

        The attention queue is the channel the agent already drains in
        _preprocess_turn (non-"user" sources become Background Attention
        Items in the next system prompt), so a finished job surfaces on
        the next turn without the agent polling. The EventBus publish
        lets UI/proactive subscribers react immediately.
        """
        status = job["status"]
        exit_info = f" exit={job['exit_code']}" if job["exit_code"] is not None else ""
        content = (
            f"Background job {job['id']} {status}{exit_info}: "
            f"{job['command'][:100]}"
        )
        try:
            from core.attention_priority import get_attention_queue, UrgencySignal
            get_attention_queue().enqueue(
                content=content,
                urgency=UrgencySignal.FOLLOW_UP if status == "completed"
                else UrgencySignal.SYSTEM_ALERT,
                source="background_job",
                metadata={"job_id": job["id"], "exit_code": job["exit_code"]},
            )
        except Exception as e:
            logger.debug(f"Background job attention enqueue failed: {e}")
        try:
            from core.event_bus import get_event_bus
            get_event_bus().publish(
                "job.background.completed",
                {
                    "job_id": job["id"],
                    "status": status,
                    "exit_code": job["exit_code"],
                    "command": job["command"][:200],
                },
                source="tools.terminal",
            )
        except Exception as e:
            logger.debug(f"Background job event publish failed: {e}")

    def list_background_jobs(self) -> list[dict]:
        """Job snapshots without the internal thread handle."""
        return [
            {k: v for k, v in j.items() if k != "thread"}
            for j in self._background_jobs.values()
        ]

    async def _check_cwd_change(self, command: str, current_cwd: str) -> Optional[str]:
        if command.strip().startswith("cd "):
            parts = command.strip().split(maxsplit=1)
            if len(parts) > 1:
                new_dir = parts[1].strip()
                if new_dir == "-":
                    return None
                try:
                    target = Path(new_dir)
                    if not target.is_absolute():
                        target = Path(current_cwd) / target
                    if target.is_dir():
                        return str(target.resolve())
                except (OSError, ValueError) as e:
                    logger.debug(f"Invalid directory in cd command: {e}")
        return None

    def create_session(self, name: Optional[str] = None) -> str:
        import uuid
        session_id = name or f"session_{uuid.uuid4().hex[:8]}"
        session = TerminalSession(
            id=session_id,
            created_at=datetime.now(),
            cwd=str(self.workspace),
            env=self._safe_env(),
        )
        self.sessions[session_id] = session
        return session_id

    def get_status(self) -> dict:
        return {
            "workspace": str(self.workspace),
            "mode": self.mode.value,
            "active_session": self.active_session,
            "sessions": {k: v.to_dict() for k, v in self.sessions.items()},
            "auto_approve_safe": self.auto_approve_safe,
        }


_computer: Optional[JarvisComputer] = None
_computer_lock = threading.Lock()

def get_jarvis_computer() -> JarvisComputer:
    global _computer
    if _computer is None:
        with _computer_lock:
            if _computer is None:
                _computer = JarvisComputer()
    return _computer


class TerminalExecuteTool(Tool):
    """Execute commands on Jarvis's computer."""

    name = "terminal_execute"
    description = """Execute a shell command on Jarvis's computer.
Jarvis has his own workspace and can run any command.
SECURITY: Always requires permission - even during autonomous operations (daydream, free time, etc.)
REQUIRES PERMISSION for most operations - Jarvis will ask you before executing.
Safe commands (ls, cat, git, python, etc.) may be auto-approved."""
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute",
            },
            "timeout": {
                "type": "number",
                "description": "Timeout in seconds (default 60, max 300 for foreground; background jobs allow up to 3600)",
                "default": 60,
            },
            "reason": {
                "type": "string",
                "description": "Why this command is being executed",
            },
            "session": {
                "type": "string",
                "description": "Optional session ID to execute in",
            },
            "background": {
                "type": "boolean",
                "description": "Run in the background and return a job id immediately (for long builds, test suites, downloads). Check results later with check_jobs.",
                "default": False,
            },
        },
        "required": ["command"],
    }
    requires_permission = True

    def __init__(self, computer: Optional[JarvisComputer] = None):
        self.computer = computer or get_jarvis_computer()

    def needs_permission(self, command: str = "", **kwargs) -> bool:
        return self.computer.needs_permission(command)

    async def execute(
        self,
        command: str,
        timeout: float = 60.0,
        reason: str = "",
        session: Optional[str] = None,
        background: bool = False,
    ) -> str:
        timeout = min(timeout, 300)

        risk, risk_reason = self.computer.assess_risk(command)

        # HIGH risk commands are BLOCKED - they cannot be executed
        if risk == "high":
            from core.permission_coordinator import get_permission_coordinator, PermissionType, RiskLevel
            coordinator = get_permission_coordinator()
            
            # Log the blocked attempt
            coordinator.request(
                type=PermissionType.TERMINAL,
                source="terminal_execute",
                operation=f"BLOCKED: {command[:100]}",
                risk_level=RiskLevel.HIGH,
                context={"command": command, "risk_reason": risk_reason},
            )
            
            return (
                f"❌ BLOCKED: {risk_reason}\n"
                f"Command: {command}\n"
                f"\nThis command has been blocked for safety.\n"
                f"If you need to run this, please use a different method."
            )
        
        # MEDIUM risk commands require permission (will show warning)
        if risk == "medium":
            # Tool already has requires_permission=False, so medium risk is allowed
            # but will be logged for review
            pass

        # Background mode: same validation above, but don't wait — the
        # job registry keeps the result for polling via check_jobs.
        if background:
            started, job_or_error = self.computer.start_background(
                command, timeout=timeout, session_id=session)
            if not started:
                return f"❌ Could not start background job: {job_or_error}"
            return (
                f"⏳ Background job started: {job_or_error}\n"
                f"Command: {command[:200]}\n"
                f"Continue with other work; check results later with "
                f"check_jobs (job_id={job_or_error})."
            )

        exit_code, stdout, stderr = await self.computer.execute(
            command, timeout, session
        )

        try:
            from core.execution_context import get_execution_context
            ctx = get_execution_context()
            terminal_cwd = self.computer.get_cwd()
            if terminal_cwd != ctx.cwd:
                ctx.cd(terminal_cwd)
        except Exception:
            pass

        output_parts = []

        if stdout:
            output_parts.append(stdout)
        if stderr:
            output_parts.append(f"[stderr]\n{stderr}")

        output = "\n".join(output_parts) if output_parts else "(no output)"

        if exit_code != 0:
            return f"❌ Exit code: {exit_code}\n{output}"

        return output


class BackgroundJobsTool(Tool):
    """List background jobs or fetch one job's captured output."""

    name = "check_jobs"
    description = (
        "Check background jobs started with terminal_execute(background=true). "
        "Without job_id: lists all jobs and their status. With job_id: returns "
        "that job's status, exit code and captured output (stdout/stderr)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "job_id": {
                "type": "string",
                "description": "Optional job id (e.g. 'bg-1'); omit to list all jobs.",
            },
        },
    }
    requires_permission = False

    def __init__(self, computer: Optional[JarvisComputer] = None):
        self.computer = computer or get_jarvis_computer()

    async def execute(self, job_id: Optional[str] = None) -> str:
        if job_id:
            job = self.computer.get_background_job(job_id)
            if not job:
                return f"No such background job: {job_id}"
            lines = [
                f"Job {job['id']}: {job['status']}",
                f"Command: {job['command'][:200]}",
                f"Started: {job['started_at']}",
            ]
            if job["finished_at"]:
                lines.append(f"Finished: {job['finished_at']}")
            if job["exit_code"] is not None:
                lines.append(f"Exit code: {job['exit_code']}")
            if job["stdout"]:
                lines.append(f"--- stdout ---\n{job['stdout'][-4000:]}")
            if job["stderr"]:
                lines.append(f"--- stderr ---\n{job['stderr'][-2000:]}")
            return "\n".join(lines)

        jobs = self.computer.list_background_jobs()
        if not jobs:
            return "No background jobs."
        lines = []
        for j in jobs:
            detail = j["command"][:80]
            exit_info = f" exit={j['exit_code']}" if j["exit_code"] is not None else ""
            lines.append(f"- {j['id']} [{j['status']}{exit_info}] {detail}")
        return "Background jobs:\n" + "\n".join(lines)


class TerminalCwdTool(Tool):
    """Get or change current working directory."""

    name = "terminal_cwd"
    description = "Get or change Jarvis's current working directory"
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "New directory path (optional, returns current if not provided)",
            },
        },
    }
    requires_permission = False

    def __init__(self, computer: Optional[JarvisComputer] = None):
        self.computer = computer or get_jarvis_computer()

    async def execute(self, path: Optional[str] = None) -> str:
        if path:
            if self.computer.set_cwd(path):
                try:
                    from core.execution_context import get_execution_context
                    get_execution_context().cd(self.computer.get_cwd())
                except Exception:
                    pass
                return f"Working directory changed to: {self.computer.get_cwd()}"
            else:
                return f"Failed to change to: {path}"
        return f"Current working directory: {self.computer.get_cwd()}"


class TerminalSessionTool(Tool):
    """Manage terminal sessions."""

    name = "terminal_session"
    description = "Create, list, or switch terminal sessions for Jarvis"
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "Action: 'create', 'list', 'switch', 'status'",
                "enum": ["create", "list", "switch", "status"],
            },
            "name": {
                "type": "string",
                "description": "Session name (for create/switch)",
            },
        },
        "required": ["action"],
    }

    def __init__(self, computer: Optional[JarvisComputer] = None):
        self.computer = computer or get_jarvis_computer()

    async def execute(self, action: str, name: Optional[str] = None) -> str:
        if action == "create":
            session_id = self.computer.create_session(name)
            return f"Created terminal session: {session_id}"

        elif action == "list":
            sessions = []
            for sid, session in self.computer.sessions.items():
                active = " (active)" if sid == self.computer.active_session else ""
                sessions.append(f"  {sid}: {session.cwd}{active}")
            return "Terminal sessions:\n" + "\n".join(sessions)

        elif action == "switch":
            if name and name in self.computer.sessions:
                self.computer.active_session = name
                return f"Switched to session: {name}"
            return f"Session not found: {name}"

        elif action == "status":
            status = self.computer.get_status()
            lines = [
                "🖥️ Jarvis's Computer",
                f"Workspace: {status['workspace']}",
                f"Mode: {status['mode']}",
                f"Active session: {status['active_session']}",
                f"Sessions: {len(status['sessions'])}",
            ]
            return "\n".join(lines)

        return f"Unknown action: {action}"


class TerminalHistoryTool(Tool):
    """View command history."""

    name = "terminal_history"
    description = "View Jarvis's terminal command history"
    parameters = {
        "type": "object",
        "properties": {
            "count": {
                "type": "integer",
                "description": "Number of recent commands to show",
                "default": 20,
            },
        },
    }

    def __init__(self, computer: Optional[JarvisComputer] = None):
        self.computer = computer or get_jarvis_computer()

    async def execute(self, count: int = 20) -> str:
        session = self.computer.sessions.get(
            self.computer.active_session or "default"
        )
        if not session or not session.history:
            return "No command history"

        lines = ["📜 Recent commands:"]
        for item in list(session.history)[-count:]:
            lines.append(f"  [{item['timestamp']}] {item['command']}")

        return "\n".join(lines)


class ComputerStatusTool(Tool):
    """Get status of Jarvis's computer."""

    name = "computer_status"
    description = "Get the status of Jarvis's virtual computer - workspace, sessions, capabilities"
    parameters = {
        "type": "object",
        "properties": {},
    }

    def __init__(self, computer: Optional[JarvisComputer] = None):
        self.computer = computer or get_jarvis_computer()

    async def execute(self) -> str:
        status = self.computer.get_status()
        import platform
        import psutil

        lines = [
            "🖥️ JARVIS'S COMPUTER",
            "=" * 40,
            "",
            f"📁 Workspace: {status['workspace']}",
            f"🔒 Mode: {status['mode']}",
            f"📍 Current directory: {self.computer.get_cwd()}",
            "",
            "SYSTEM INFO:",
            f"  OS: {platform.system()} {platform.release()}",
            f"  Hostname: {platform.node()}",
            f"  Python: {platform.python_version()}",
            "",
            "RESOURCES:",
            f"  CPU: {psutil.cpu_percent():.0f}%",
            f"  Memory: {psutil.virtual_memory().percent:.0f}%",
            f"  Disk: {psutil.disk_usage('/').percent:.0f}%",
            "",
            "TERMINAL:",
            f"  Active session: {status['active_session']}",
            f"  Total sessions: {len(status['sessions'])}",
            f"  Auto-approve safe: {status['auto_approve_safe']}",
        ]

        active = self.computer.sessions.get(self.computer.active_session)
        if active and active.history:
            lines.extend([
                "",
                "RECENT COMMANDS:",
                f"  Last: {list(active.history)[-1]['command'] if active.history else 'none'}",
            ])

        return "\n".join(lines)


class TerminalModeTool(Tool):
    """Change terminal access mode."""

    name = "terminal_mode"
    description = "Change Jarvis's terminal access mode (sandbox/project/full)"
    parameters = {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "description": "Access mode: 'sandbox', 'project', or 'full'",
                "enum": ["sandbox", "project", "full"],
            },
        },
        "required": ["mode"],
    }
    requires_permission = True

    def __init__(self, computer: Optional[JarvisComputer] = None):
        self.computer = computer or get_jarvis_computer()

    async def execute(self, mode: str) -> str:
        try:
            self.computer.mode = TerminalMode(mode)
            return f"Terminal mode changed to: {mode}"
        except ValueError:
            return "Invalid mode. Use: sandbox, project, or full"


class InstallPackageTool(Tool):
    """Install packages on Jarvis's computer."""

    name = "install_package"
    description = "Install a package on Jarvis's computer. Supports pip, npm, apt, etc."
    parameters = {
        "type": "object",
        "properties": {
            "package": {
                "type": "string",
                "description": "Package name to install",
            },
            "manager": {
                "type": "string",
                "description": "Package manager: pip, npm, apt, cargo, go (auto-detected if not specified)",
            },
        },
        "required": ["package"],
    }
    requires_permission = True

    def __init__(self, computer: Optional[JarvisComputer] = None):
        self.computer = computer or get_jarvis_computer()

    async def execute(self, package: str, manager: Optional[str] = None) -> str:
        import re
        if not re.match(r'^[a-zA-Z0-9._@/\-]+$', package):
            return f"Error: Invalid package name '{package}'. Only alphanumeric characters, dots, dashes, underscores, @, and / are allowed."
        if '--' in package or '://' in package or any(c in package for c in ('|', '>', '<', '$', '`', '&', ';', '!', '#')):
            return f"Error: Package name contains forbidden characters: {package}"

        if not manager:
            if package.startswith("@") or package.startswith("npm:"):
                manager = "npm"
            elif package.endswith(".crate") or "/" in package and not package.startswith("git"):
                manager = "cargo"
            else:
                manager = "pip"

        commands = {
            "pip": f"pip install {package}",
            "pip3": f"pip3 install {package}",
            "npm": f"npm install -g {package}",
            "apt": f"sudo apt-get install -y {package}",
            "cargo": f"cargo install {package}",
            "go": f"go install {package}",
        }

        if manager not in commands:
            return f"Unknown package manager: {manager}. Use: pip, npm, apt, cargo, go"

        command = commands[manager]

        exit_code, stdout, stderr = await self.computer.execute(command, timeout=120)

        if exit_code == 0:
            return f"✅ Successfully installed {package} via {manager}\n{stdout}"
        else:
            return f"❌ Failed to install {package} via {manager}:\n{stderr or stdout}"


TERMINAL_TOOLS = [
    TerminalExecuteTool,
    TerminalCwdTool,
    TerminalSessionTool,
    TerminalHistoryTool,
    ComputerStatusTool,
    TerminalModeTool,
    BackgroundJobsTool,
    InstallPackageTool,
]


def register_terminal_tools(registry, computer: Optional[JarvisComputer] = None) -> int:
    """Register all terminal tools with a tool registry or agent."""
    computer = computer or get_jarvis_computer()

    tools = []
    for tool_class in TERMINAL_TOOLS:
        try:
            if tool_class in (TerminalExecuteTool, TerminalCwdTool, TerminalSessionTool,
                            TerminalHistoryTool, ComputerStatusTool, TerminalModeTool,
                            BackgroundJobsTool, InstallPackageTool):
                tools.append(tool_class(computer=computer))
            else:
                tools.append(tool_class())
        except Exception as e:
            logger.error(f"Failed to create {tool_class.name}: {e}")

    if not tools:
        return 0

    if hasattr(registry, 'register_tools'):
        registry.register_tools(tools)
    else:
        for tool in tools:
            registry.register(tool)

    logger.info(f"Registered {len(tools)} terminal tools")
    return len(tools)


__all__ = [
    "JarvisComputer",
    "TerminalMode",
    "TerminalSession",
    "BackgroundJobsTool",
    "TERMINAL_TOOLS",
    "register_terminal_tools",
    "get_jarvis_computer",
]
