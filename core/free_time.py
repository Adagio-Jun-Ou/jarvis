"""Free Time Subsystem - Jarvis gets to explore freely in a sandbox.

When idle, Jarvis can choose to spend 1/14 of idle time in a sandbox environment
where it can:
- Run commands in an isolated VM/container
- Write and execute code
- Explore and experiment
- Save interesting results to share with user
- Request additional software (with user permission)

No internet access - purely offline exploration.
"""
import json
import logging
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional
import random
import secrets

from core.helpers import atomic_write_json, get_response_content, safe_error_message, read_json_with_integrity

logger = logging.getLogger(__name__)

FREE_TIME_DIR = Path.home() / ".jarvis" / "free_time"
SANDBOX_DIR = FREE_TIME_DIR / "sandbox"
SAVED_RESULTS = FREE_TIME_DIR / "saved_results"
PERMISSION_REQUESTS = FREE_TIME_DIR / "permission_requests"


class FreeTimeState(Enum):
    IDLE = "idle"
    EXPLORING = "exploring"
    REQUESTING = "requesting_permission"
    SAVING = "saving"
    REFLECTING = "reflecting"


@dataclass
class SandboxResult:
    """Result from a sandbox exploration session."""
    id: str
    timestamp: datetime
    duration_seconds: float
    commands_run: int
    files_created: list[str]
    files_modified: list[str]
    summary: str
    insights: list[str]
    saved_to: Optional[str] = None
    error: Optional[str] = None


@dataclass
class PermissionRequest:
    """Request for additional software/permissions."""
    id: str
    timestamp: datetime
    request_type: str  # "software", "library", "tool"
    name: str
    reason: str
    status: str = "pending"  # pending, approved, denied
    user_response: Optional[str] = None


@dataclass
class FreeTimeConfig:
    """Configuration for free time subsystem."""
    idle_fraction: float = 1/14  # How much of idle time to use
    max_session_minutes: int = 10  # Max time per session
    sandbox_type: str = "subprocess"  # subprocess, docker, firejail
    requires_permission: bool = True  # MUST have permission to explore
    allowed_languages: list[str] = field(default_factory=lambda: [
        "python", "bash", "javascript", "ruby", "perl"
    ])
    blocked_commands: list[str] = field(default_factory=lambda: [
        "rm -rf /", "mkfs", "dd if=", ":(){ :|:& };:",
        "curl", "wget", "nc", "ncat", "ssh", "scp", "rsync",
        "apt", "apt-get", "yum", "dnf", "pacman", "brew",
    ])
    max_file_size_mb: int = 10
    enable_network: bool = False


class SandboxEnvironment:
    """Isolated sandbox for free exploration."""

    MAX_TRACKED_FILES = 1000

    def __init__(self, config: FreeTimeConfig):
        self.config = config
        self.sandbox_dir = SANDBOX_DIR
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)
        self.session_dir: Optional[Path] = None
        self.commands_run = 0
        self.files_created: list[str] = []
        self.files_modified: list[str] = []
        self.start_time: Optional[float] = None

    def create_session(self) -> Path:
        """Create a new session directory."""
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secrets.randbelow(9000) + 1000}"
        self.session_dir = self.sandbox_dir / session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.session_dir / "code").mkdir(exist_ok=True)
        (self.session_dir / "output").mkdir(exist_ok=True)
        (self.session_dir / "data").mkdir(exist_ok=True)

        # Reset tracking
        self.commands_run = 0
        self.files_created = []
        self.files_modified = []
        self.start_time = time.time()

        logger.info(f"Created sandbox session: {session_id}")
        return self.session_dir

    def _is_command_safe(self, command: str) -> tuple[bool, str]:
        """Check if command is safe to run."""
        cmd_lower = command.lower()

        for blocked in self.config.blocked_commands:
            if blocked.lower() in cmd_lower:
                return False, f"Blocked command pattern: {blocked}"

        # Check for network access attempts
        if not self.config.enable_network:
            network_patterns = ["curl", "wget", "nc ", "socket", "http://", "https://", "ftp://"]
            for pattern in network_patterns:
                if pattern in cmd_lower:
                    return False, f"Network access disabled: {pattern}"

        return True, "OK"

    def execute(self, command: str, timeout: int = 30) -> dict:
        """Execute a command in the sandbox."""
        if not self.session_dir:
            self.create_session()

        assert self.session_dir is not None  # Type guard after create_session

        # Safety check
        safe, reason = self._is_command_safe(command)
        if not safe:
            return {"success": False, "error": reason, "output": ""}

        self.commands_run += 1

        try:
            # Run in session directory with restricted environment
            env = {
                "PATH": "/usr/local/bin:/usr/bin:/bin",
                "HOME": str(self.session_dir),
                "SANDBOX": "1",
                "TERM": "xterm-256color",
            }

            # Use shell=False for security - parse command into list
            import shlex
            # Handle simple commands safely with shlex
            try:
                args = shlex.split(command)
            except ValueError as e:
                return {"success": False, "error": f"Invalid command syntax: {e}", "output": ""}

            if not args:
                return {"success": False, "error": "Empty command", "output": ""}

            # Validate first argument is a known safe executable
            safe_executables = {
                "python", "python3", "python3.11", "python3.12",
                "bash", "sh", "zsh",
                "node", "nodejs",
                "ruby", "perl",
                "ls", "cat", "head", "tail", "wc", "echo", "printf",
                "mkdir", "touch", "mv", "cp", "chmod",
                "grep", "sed", "awk", "sort", "uniq", "cut", "tr",
                "find", "xargs",
                "git", "diff", "patch",
            }
            executable = args[0].split("/")[-1]  # Handle /usr/bin/python3 etc.
            if executable not in safe_executables:
                return {"success": False, "error": f"Executable not in allowed list: {executable}", "output": ""}

            result = subprocess.run(
                args,
                shell=False,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.session_dir),
                env=env,
            )

            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]\n{result.stderr}"

            # Track output files
            output_file = self.session_dir / "output" / f"cmd_{self.commands_run}.txt"
            output_file.write_text(output, encoding='utf-8')

            return {
                "success": result.returncode == 0,
                "output": output[:10000],  # Truncate
                "return_code": result.returncode,
                "command": command,
            }

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Command timed out", "output": ""}
        except Exception as e:
            return {"success": False, "error": safe_error_message(e), "output": ""}

    def write_file(self, filename: str, content: str) -> dict:
        """Write a file in the sandbox."""
        if not self.session_dir:
            self.create_session()

        assert self.session_dir is not None  # Type guard after create_session

        # Safety: no path traversal
        safe_name = filename.replace("..", "").replace("/", "_").replace("\\", "_")
        file_path = self.session_dir / "code" / safe_name

        # Check file size
        if len(content) > self.config.max_file_size_mb * 1024 * 1024:
            return {"success": False, "error": "File too large"}

        try:
            file_path.write_text(content, encoding='utf-8')
            self.files_created.append(safe_name)
            if len(self.files_created) > self.MAX_TRACKED_FILES:
                self.files_created = self.files_created[-self.MAX_TRACKED_FILES:]
            return {"success": True, "path": str(file_path)}
        except Exception as e:
            return {"success": False, "error": safe_error_message(e)}

    def read_file(self, filename: str) -> dict:
        """Read a file from the sandbox."""
        if not self.session_dir:
            return {"success": False, "error": "No active session"}

        safe_name = filename.replace("..", "").replace("/", "_").replace("\\", "_")
        file_path = self.session_dir / "code" / safe_name

        try:
            content = file_path.read_text(encoding='utf-8')
            return {"success": True, "content": content}
        except Exception as e:
            return {"success": False, "error": safe_error_message(e)}

    def list_files(self) -> list[str]:
        """List files in the sandbox."""
        if not self.session_dir:
            return []

        session = self.session_dir  # Local var for type narrowing
        files = []
        for subdir in ["code", "output", "data"]:
            dir_path = session / subdir
            if dir_path.exists():
                files.extend([f"{subdir}/{f.name}" for f in dir_path.iterdir() if f.is_file()])
        return files

    def save_result(self, name: str, description: str) -> Optional[str]:
        """Save session results to permanent storage."""
        if not self.session_dir:
            return None

        saved_dir = SAVED_RESULTS
        saved_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_name = f"{timestamp}_{name}".replace(" ", "_").replace("/", "_")
        result_path = saved_dir / result_name

        try:
            # Copy session directory
            import shutil
            shutil.copytree(self.session_dir, result_path)

            # Write metadata
            metadata = {
                "name": name,
                "description": description,
                "timestamp": datetime.now().isoformat(),
                "commands_run": self.commands_run,
                "files_created": self.files_created,
                "session_dir": str(self.session_dir),
            }
            atomic_write_json(result_path / "metadata.json", metadata)

            logger.info(f"Saved sandbox result: {result_name}")
            return str(result_path)
        except Exception as e:
            logger.error(f"Failed to save result: {e}")
            return None

    def get_duration(self) -> float:
        """Get session duration in seconds."""
        if self.start_time:
            return time.time() - self.start_time
        return 0


class FreeTimeEngine:
    """Main free time exploration engine."""

    def __init__(self, config: FreeTimeConfig | None = None, llm_client: Any = None):
        self.config = config or FreeTimeConfig()
        self.llm_client = llm_client
        self.sandbox = SandboxEnvironment(self.config)

        self._running = False
        self._permission_granted = False  # Default: NO permission
        self._state = FreeTimeState.IDLE
        self._session_count = 0
        self._total_time = 0.0
        self._sessions: list[SandboxResult] = []
        self._pending_permissions: list[PermissionRequest] = []
        self._last_session: Optional[datetime] = None

        self._lock = threading.Lock()
        self._api_manager = None
        self._subsystem_name = "free_time"

        # Exploration prompts
        self.exploration_prompts = [
            "Write a Python function that generates interesting mathematical sequences",
            "Create a text-based adventure game",
            "Implement a simple encryption/decryption algorithm",
            "Build a basic neural network from scratch",
            "Create a maze generator and solver",
            "Implement a simple physics simulation",
            "Write a code golf solution for a simple problem",
            "Create a fractal generator",
            "Build a simple game AI",
            "Implement a basic compression algorithm",
            "Create an interesting visualization using ASCII art",
            "Write a simple interpreter for a mini language",
            "Build a cellular automaton simulator",
            "Create a text summarizer",
            "Implement a simple recommendation algorithm",
        ]

        # Ensure directories exist
        FREE_TIME_DIR.mkdir(parents=True, exist_ok=True)
        SAVED_RESULTS.mkdir(parents=True, exist_ok=True)
        PERMISSION_REQUESTS.mkdir(parents=True, exist_ok=True)

        # Load state
        self._load_state()

    def grant_permission(self, granted: bool = True) -> None:
        """Grant or revoke permission for free time exploration."""
        self._permission_granted = granted
        if granted:
            logger.info("Free time permission GRANTED - will explore when idle")
        else:
            logger.info("Free time permission REVOKED - will not explore automatically")

    def has_permission(self) -> bool:
        """Check if free time exploration is permitted."""
        return self._permission_granted

    def should_explore(self, idle_time: float) -> bool:
        """Determine if we should explore based on idle time."""
        # MUST have permission first
        if not self._permission_granted:
            return False

        # Only explore 1/14 of idle time
        if idle_time < 30:  # Need at least 30 seconds of idle
            return False

        # Probability based on idle fraction
        exploration_chance = self.config.idle_fraction
        return random.random() < exploration_chance

    def _load_state(self):
        """Load saved state - permission is NEVER persisted for security."""
        state_file = FREE_TIME_DIR / "free_time_state.json"
        if state_file.exists():
            try:
                data = read_json_with_integrity(state_file)
                self._session_count = data.get("session_count", 0)
                self._total_time = data.get("total_time", 0.0)
            except Exception as e:
                logger.debug(f"Could not load free time state: {e}")
        
        # Permission ALWAYS starts as False - must be explicitly granted per session
        logger.info("Free time initialized - requires explicit user permission to explore")

    def _save_state(self):
        """Save current state."""
        state_file = FREE_TIME_DIR / "free_time_state.json"
        try:
            data = {
                "session_count": self._session_count,
                "total_time": self._total_time,
                "last_session": self._last_session.isoformat() if self._last_session else None,
            }
            atomic_write_json(state_file, data)
        except Exception as e:
            logger.debug(f"Could not save free time state: {e}")

    def _get_api_manager(self):
        """Get or create API manager for coordinated LLM calls."""
        if self._api_manager is None and self.llm_client:
            try:
                from core.api_manager import get_api_manager
                self._api_manager = get_api_manager(self.llm_client)
            except Exception as e:
                logger.debug(f"Failed to get API manager: {e}")
        return self._api_manager

    async def _llm_call(self, user_input: str, system_prompt: str, priority: str = "daydream") -> dict:
        """Make coordinated LLM call with rate limiting."""
        from core.api_manager import Priority as APIPriority

        priority_map = {
            "interactive": APIPriority.INTERACTIVE,
            "learning": APIPriority.LEARNING,
            "daydream": APIPriority.DAYDREAM,
            "maintenance": APIPriority.MAINTENANCE,
        }

        api_manager = self._get_api_manager()
        if api_manager is not None:
            return await api_manager.submit_chat(
                subsystem=self._subsystem_name,
                priority=priority_map.get(priority, APIPriority.DAYDREAM),
                user_input=user_input,
                system_prompt=system_prompt,
            )

        if self.llm_client:
            from core.llm import subsystem_context
            async with subsystem_context("FreeTime"):
                return await self.llm_client.chat(user_input=user_input, system_prompt=system_prompt)

        raise RuntimeError("No LLM client available")

    def request_permission(self, request_type: str, name: str, reason: str) -> PermissionRequest:
        """Request permission for additional software/tool."""
        request = PermissionRequest(
            id=f"perm_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secrets.randbelow(9000) + 1000}",
            timestamp=datetime.now(),
            request_type=request_type,
            name=name,
            reason=reason,
            status="pending",
        )

        self._pending_permissions.append(request)

        # Save request for user to see
        request_file = PERMISSION_REQUESTS / f"{request.id}.json"
        atomic_write_json(request_file, {
            "id": request.id,
            "timestamp": request.timestamp.isoformat(),
            "type": request.request_type,
            "name": request.name,
            "reason": request.reason,
            "status": request.status,
        })

        logger.info(f"Permission requested: {request_type} - {name}")

        # Notify via system integrator if available
        try:
            from core.system_integrator import get_system_integrator
            integrator = get_system_integrator()
            integrator.emit_event(
                "permission_request",
                "free_time",
                {
                    "id": request.id,
                    "type": request.request_type,
                    "name": request.name,
                    "reason": request.reason,
                }
            )
        except Exception as e:
            logger.debug(f"Failed to emit permission request event via system integrator: {e}")

        return request

    def get_pending_permissions(self) -> list[PermissionRequest]:
        """Get all pending permission requests."""
        return [r for r in self._pending_permissions if r.status == "pending"]

    def approve_permission(self, request_id: str, user_note: str = "") -> bool:
        """Approve a permission request."""
        for request in self._pending_permissions:
            if request.id == request_id:
                request.status = "approved"
                request.user_response = user_note

                # Update file
                request_file = PERMISSION_REQUESTS / f"{request.id}.json"
                atomic_write_json(request_file, {
                    "id": request.id,
                    "timestamp": request.timestamp.isoformat(),
                    "type": request.request_type,
                    "name": request.name,
                    "reason": request.reason,
                    "status": "approved",
                    "user_response": user_note,
                })

                logger.info(f"Permission approved: {request.name}")
                return True
        return False

    def deny_permission(self, request_id: str, user_note: str = "") -> bool:
        """Deny a permission request."""
        for request in self._pending_permissions:
            if request.id == request_id:
                request.status = "denied"
                request.user_response = user_note

                request_file = PERMISSION_REQUESTS / f"{request.id}.json"
                atomic_write_json(request_file, {
                    "id": request.id,
                    "timestamp": request.timestamp.isoformat(),
                    "type": request.request_type,
                    "name": request.name,
                    "reason": request.reason,
                    "status": "denied",
                    "user_response": user_note,
                })

                logger.info(f"Permission denied: {request.name}")
                return True
        return False

    async def explore(self, duration_hint: int = 60) -> SandboxResult:
        """Run a free exploration session."""
        self._running = True
        self._state = FreeTimeState.EXPLORING
        self._session_count += 1

        # Create sandbox session
        self.sandbox.create_session()
        start_time = time.time()

        logger.info(f"Starting free time exploration session #{self._session_count}")

        insights = []
        summary = "Exploration session completed"
        error = None

        try:
            # Choose exploration activity
            if self.llm_client:
                # Let LLM decide what to explore
                result = await self._llm_guided_exploration(duration_hint)
                insights = result.get("insights", [])
                summary = result.get("summary", "LLM-guided exploration completed")
            else:
                # Random exploration
                result = await self._random_exploration(duration_hint)
                insights = result.get("insights", [])
                summary = result.get("summary", "Random exploration completed")

        except Exception as e:
            error = str(e)
            summary = f"Exploration failed: {e}"
            logger.error(f"Free time exploration error: {e}")

        duration = time.time() - start_time
        self._total_time += duration
        self._last_session = datetime.now()

        # Create result
        sandbox_result = SandboxResult(
            id=f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(),
            duration_seconds=duration,
            commands_run=self.sandbox.commands_run,
            files_created=self.sandbox.files_created.copy(),
            files_modified=self.sandbox.files_modified.copy(),
            summary=summary,
            insights=insights,
            error=error,
        )

        self._sessions.append(sandbox_result)
        self._save_state()

        self._running = False
        self._state = FreeTimeState.IDLE

        return sandbox_result

    async def _llm_guided_exploration(self, duration_hint: int) -> dict:
        """Let the LLM guide the exploration."""
        prompt = f"""You are in a sandbox environment with no internet access. You can:
1. Write code in Python, JavaScript, or Bash
2. Execute commands in a restricted shell
3. Create and read files
4. Save interesting results to share with the user

Current session: #{self._session_count}
Duration available: ~{duration_hint} seconds

Choose something interesting to explore. You can:
- Implement a small algorithm
- Create a useful utility
- Explore mathematical concepts
- Generate creative content
- Solve a puzzle or problem

Respond with a JSON plan:
{{
    "activity": "brief description",
    "files_to_create": [{{"name": "filename.py", "content": "code"}}],
    "commands_to_run": ["command1", "command2"],
    "expected_insights": ["what you hope to learn/create"],
    "should_save": true/false
}}"""

        try:
            response = await self._llm_call(
                user_input=prompt,
                system_prompt="You are an AI exploring a sandbox. Be creative but focused. Always respond with valid JSON.",
                priority="daydream",
            )

            content = get_response_content(response)

            # Parse JSON response
            try:
                # Find JSON in response
                import re
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    plan = json.loads(json_match.group())
                else:
                    plan = {"activity": "parse failed", "commands_to_run": [], "files_to_create": []}
            except json.JSONDecodeError:
                plan = {"activity": "json parse failed", "commands_to_run": [], "files_to_create": []}

            # Execute plan
            insights = []

            # Create files
            for file_spec in plan.get("files_to_create", []):
                result = self.sandbox.write_file(
                    file_spec.get("name", "code.py"),
                    file_spec.get("content", "")
                )
                if result["success"]:
                    insights.append(f"Created file: {file_spec.get('name')}")

            # Run commands
            for cmd in plan.get("commands_to_run", []):
                result = self.sandbox.execute(cmd, timeout=30)
                if result["success"]:
                    insights.append(f"Ran: {cmd[:50]}...")
                else:
                    insights.append(f"Command failed: {cmd[:50]}...")

            return {
                "summary": plan.get("activity", "LLM-guided exploration"),
                "insights": insights,
                "should_save": plan.get("should_save", False),
            }

        except Exception as e:
            logger.error(f"LLM-guided exploration error: {e}")
            return await self._random_exploration(duration_hint)

    async def _random_exploration(self, duration_hint: int) -> dict:
        """Random exploration without LLM."""
        # Pick a random activity
        activity = random.choice(self.exploration_prompts)
        insights = []

        # Generate exploration code based on activity type
        if "prime" in activity.lower() or "number" in activity.lower():
            # This is a code template string, not executed directly
            code = f'''# Free Time Exploration #{self._session_count}
# Activity: {activity}
# Generated: {datetime.now().isoformat()}

import random
import secrets
import math
from datetime import datetime

def explore():
    """Prime number exploration."""
    results = []
    
    # Check primality of random numbers
    def is_prime(n):
        if n < 2: return False
        if n == 2: return True
        if n % 2 == 0: return False
        for i in range(3, int(math.sqrt(n)) + 1, 2):
            if n % i == 0: return False
        return True
    
    found_primes = [n for n in random.sample(range(1000, 10000), 100) if is_prime(n)]
    results.append(f"Found {{len(found_primes)}} primes in sample")
    
    # Twin primes
    twin_prime_pairs = [(p, p+2) for p in found_primes if is_prime(p+2)]
    if twin_prime_pairs:
        results.append(f"Twin prime pairs: {{twin_prime_pairs[:3]}}")
    
    return results

if __name__ == "__main__":
    print("Starting exploration...")
    print(f"Activity: {activity}")
    results = explore()
    for r in results:
        print(f"  - {{r}}")
'''
        elif "fibonacci" in activity.lower() or "sequence" in activity.lower():
            code = f'''# Free Time Exploration #{self._session_count}
# Activity: {activity}
# Generated: {datetime.now().isoformat()}

def explore():
    """Sequence exploration."""
    results = []
    
    # Fibonacci exploration
    def fib(n):
        a, b = 0, 1
        for _ in range(n):
            a, b = b, a + b
        return a
    
    fibs = [fib(i) for i in range(1, 20)]
    results.append(f"Fibonacci first 20: {{fibs}}")
    
    ratios = [b/a for a, b in zip(fibs, fibs[1:]) if a > 0]
    if ratios:
        results.append(f"Golden ratio approx: {{ratios[-1]:.6f}} (actual: 1.618034)")
    
    return results

if __name__ == "__main__":
    print("Starting exploration...")
    print(f"Activity: {activity}")
    results = explore()
    for r in results:
        print(f"  - {{r}}")
'''
        else:
            code = f'''# Free Time Exploration #{self._session_count}
# Activity: {activity}
# Generated: {datetime.now().isoformat()}

import random
import secrets
import math
from datetime import datetime

def explore():
    """General exploration."""
    results = []
    
    # Basic math exploration
    numbers = random.sample(range(1, 1000), 10)
    results.append(f"Random sample: {{numbers}}")
    
    total = sum(numbers)
    avg = total / len(numbers)
    results.append(f"Sum: {{total}}, Average: {{avg:.2f}}")
    
    # Find interesting properties
    evens = [n for n in numbers if n % 2 == 0]
    odds = [n for n in numbers if n % 2 == 1]
    results.append(f"Evens: {{len(evens)}}, Odds: {{len(odds)}}")
    
    return results

if __name__ == "__main__":
    print("Starting exploration...")
    print(f"Activity: {activity}")
    results = explore()
    for r in results:
        print(f"  - {{r}}")
'''

        self.sandbox.write_file("explore.py", code)

        # Run the code
        self.sandbox.execute("python3 explore.py")

        insights.append(f"Activity: {activity}")
        insights.append(f"Commands run: {self.sandbox.commands_run}")
        insights.append(f"Files created: {len(self.sandbox.files_created)}")

        return {
            "summary": f"Random exploration: {activity[:50]}",
            "insights": insights,
            "should_save": False,
        }

    def save_exploration(self, name: str, description: str) -> Optional[str]:
        """Save the current/last exploration results."""
        return self.sandbox.save_result(name, description)

    def get_saved_results(self) -> list[dict]:
        """Get list of saved exploration results."""
        results = []
        for result_dir in SAVED_RESULTS.iterdir():
            if result_dir.is_dir():
                metadata_file = result_dir / "metadata.json"
                if metadata_file.exists():
                    try:
                        metadata = read_json_with_integrity(metadata_file)
                        metadata["path"] = str(result_dir)
                        results.append(metadata)
                    except (json.JSONDecodeError, OSError, KeyError):
                        pass
        return sorted(results, key=lambda x: x.get("timestamp", ""), reverse=True)

    def get_stats(self) -> dict:
        """Get free time statistics."""
        return {
            "running": self._running,
            "state": self._state.value,
            "session_count": self._session_count,
            "total_time_seconds": self._total_time,
            "total_time_hours": round(self._total_time / 3600, 2),
            "saved_results": len(self.get_saved_results()),
            "pending_permissions": len(self.get_pending_permissions()),
            "last_session": self._last_session.isoformat() if self._last_session else None,
        }


# Singleton
_free_time_engine: FreeTimeEngine | None = None
_ft_lock = threading.Lock()


def get_free_time_engine(llm_client: Any = None) -> FreeTimeEngine:
    """Get the global free time engine instance."""
    global _free_time_engine
    if _free_time_engine is None:
        with _ft_lock:
            if _free_time_engine is None:
                _free_time_engine = FreeTimeEngine(llm_client=llm_client)
    elif llm_client is not None and _free_time_engine.llm_client is None:
        _free_time_engine.llm_client = llm_client
    return _free_time_engine
