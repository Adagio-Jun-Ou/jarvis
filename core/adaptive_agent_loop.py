"""Adaptive Agent Loop — Persistent, self-healing execution for 24/7 Jarvis.

This is the runtime that makes Jarvis LIVE on the computer:
- Persistent execution context (CWD, env, process history)
- Self-healing tool execution with retry and fallback
- Checkpoint/resume across conversation turns
- Adaptive context management (auto-compress when needed)
- Real system awareness (what's running, what changed)
- Long-running task support with progress streaming

Like OpenCode but for AI — the agent doesn't just chat, it DOES.
"""
import asyncio
import json
import logging
import os
import threading
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator

from core.helpers import safe_error_message
from core.execution_context import get_execution_context

logger = logging.getLogger(__name__)


def _assess_command_risk(command: str) -> tuple[str, str]:
    """Classify a shell command's risk using the terminal tool's own rules.

    Delegates to JarvisComputer.assess_risk so there is one definition of
    "high risk" rather than a second copy that can drift. Fails closed: if the
    assessor cannot be loaded, the command is treated as high risk rather than
    waved through.
    """
    try:
        from tools.terminal import JarvisComputer
    except Exception as e:  # pragma: no cover - import guard
        logger.error(f"Cannot load command risk assessor, refusing command: {e}")
        return "high", "risk assessor unavailable"

    try:
        return JarvisComputer().assess_risk(command)
    except Exception as e:
        logger.error(f"Command risk assessment failed, refusing command: {e}")
        return "high", "risk assessment failed"


def _permission_deny_level() -> Any:
    """The PermissionLevel sentinel meaning 'explicitly denied', resolved
    lazily to avoid an import cycle with core.permissions."""
    try:
        from core.permissions import PermissionLevel

        return PermissionLevel.DENY
    except Exception:  # pragma: no cover - import guard
        logger.error("Cannot load PermissionLevel; treating unknown as deny")
        return "deny"


class ExecutionPhase(Enum):
    IDLE = "idle"
    THINKING = "thinking"
    EXECUTING = "executing"
    WAITING_APPROVAL = "waiting_approval"
    REFLECTING = "reflecting"
    ERROR_RECOVERY = "error_recovery"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TurnCheckpoint:
    """Saves the full state of an agent turn for resume."""
    turn_number: int
    phase: str
    user_input: str
    assistant_response: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    tool_results: list[dict] = field(default_factory=list)
    cwd: str = ""
    timestamp: float = field(default_factory=time.time)
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "turn_number": self.turn_number,
            "phase": self.phase,
            "user_input": self.user_input[:500],
            "assistant_response": self.assistant_response[:500],
            "tool_calls": self.tool_calls[-10:],
            "tool_results": self.tool_results[-10:],
            "cwd": self.cwd,
            "timestamp": self.timestamp,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TurnCheckpoint":
        """Rebuild from a to_dict() payload (missing keys tolerated —
        checkpoints outlive the code that wrote them)."""
        return cls(
            turn_number=int(data.get("turn_number", 0)),
            phase=str(data.get("phase", "idle")),
            user_input=str(data.get("user_input", "")),
            assistant_response=str(data.get("assistant_response", "")),
            tool_calls=list(data.get("tool_calls") or []),
            tool_results=list(data.get("tool_results") or []),
            cwd=str(data.get("cwd", "")),
            timestamp=float(data.get("timestamp", 0.0)),
            error=data.get("error"),
        )


class AdaptiveAgentLoop:
    """Self-healing, persistent agent loop that gives Jarvis real system presence.

    Key differences from the basic agent loop:
    1. EXECUTION CONTEXT: Maintains working directory, env vars, process history
    2. SELF-HEALING: Auto-retries failed tool calls with different strategies
    3. CHECKPOINT/RESUME: Saves state between turns, resumes after crashes
    4. ADAPTIVE CONTEXT: Auto-compresses context window when approaching limits
    5. SYSTEM AWARENESS: Knows what processes are running, what files changed
    6. LONG-RUNNING TASKS: Supports multi-minute operations with progress updates
    """

    MAX_RETRIES = 3
    RETRY_DELAYS = [1.0, 3.0, 5.0]
    MAX_CACHED_RESULTS = 100

    def __init__(self, agent=None, llm_client=None):
        self.agent = agent
        self.llm = llm_client
        self._ctx = get_execution_context()
        self._phase = ExecutionPhase.IDLE
        self._turn_count = 0
        self._checkpoints: list[TurnCheckpoint] = []
        self._tool_result_cache: dict[str, dict] = {}
        self._error_history: list[dict] = []
        self._total_tool_calls = 0
        self._total_errors = 0
        self._total_recoveries = 0
        self._start_time = time.time()
        self._last_input_time = 0.0
        self._consecutive_errors = 0
        self._max_consecutive_errors = 5
        self._checkpoint_path = Path(
            os.environ.get("JARVIS_ADAPTIVE_CHECKPOINT_DIR",
                           str(Path.home() / ".jarvis" / "adaptive_loop"))
        ) / "last_checkpoint.json"

    @property
    def phase(self) -> str:
        return self._phase.value

    def get_status(self) -> dict[str, Any]:
        return {
            "phase": self._phase.value,
            "turn_count": self._turn_count,
            "total_tool_calls": self._total_tool_calls,
            "total_errors": self._total_errors,
            "total_recoveries": self._total_recoveries,
            "consecutive_errors": self._consecutive_errors,
            "uptime_seconds": round(time.time() - self._start_time, 1),
            "last_input_ago": round(time.time() - self._last_input_time, 1) if self._last_input_time else None,
            "context": self._ctx.get_status(),
            "checkpoints_saved": len(self._checkpoints),
        }

    def checkpoint(self, user_input: str, response: str = "", tool_calls: list | None = None, tool_results: list | None = None, error: str | None = None) -> TurnCheckpoint:
        """Save a turn checkpoint for potential resume."""
        cp = TurnCheckpoint(
            turn_number=self._turn_count,
            phase=self._phase.value,
            user_input=user_input,
            assistant_response=response,
            tool_calls=tool_calls or [],
            tool_results=tool_results or [],
            cwd=self._ctx.cwd,
            error=error,
        )
        self._checkpoints.append(cp)
        if len(self._checkpoints) > 50:
            self._checkpoints = self._checkpoints[-50:]
        self._persist_checkpoint(cp)
        return cp

    def _persist_checkpoint(self, cp: TurnCheckpoint) -> None:
        """Atomically write the latest checkpoint to disk (best effort)."""
        try:
            self._checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(cp.to_dict(), default=str, ensure_ascii=False)
            temp_path = self._checkpoint_path.with_suffix(".tmp")
            temp_path.write_text(payload, encoding="utf-8")
            temp_path.replace(self._checkpoint_path)
        except Exception as e:
            logger.debug(f"Checkpoint persistence failed: {e}")

    def restore_checkpoint(self) -> TurnCheckpoint | None:
        """Load the last persisted checkpoint (disk, else in-memory).

        Returns None when there is nothing to restore. A restored
        checkpoint is appended to the in-memory list so ``last_checkpoint``
        and resume-context rendering work after a process restart.
        """
        cp: TurnCheckpoint | None = None
        try:
            if self._checkpoint_path.exists():
                data = json.loads(self._checkpoint_path.read_text(encoding="utf-8"))
                cp = TurnCheckpoint.from_dict(data)
        except Exception as e:
            logger.warning(f"Checkpoint restore failed: {e}")
            cp = None

        if cp is None and self._checkpoints:
            cp = self._checkpoints[-1]

        if cp is not None and cp not in self._checkpoints:
            self._checkpoints.append(cp)
        return cp

    def last_checkpoint(self, turn_number: int | None = None) -> TurnCheckpoint | None:
        """Most recent checkpoint, optionally for a specific turn."""
        if turn_number is None:
            return self._checkpoints[-1] if self._checkpoints else None
        for cp in reversed(self._checkpoints):
            if cp.turn_number == turn_number:
                return cp
        return None

    def interrupted_checkpoint(self) -> TurnCheckpoint | None:
        """A checkpoint whose turn never reached a terminal phase.

        Phase is written at turn start (THINKING) and again at turn end
        (COMPLETED/FAILED), so a persisted non-terminal phase means the
        process died mid-turn. After a restart the in-memory list is
        empty, so fall back to the on-disk checkpoint.
        """
        cp = self._checkpoints[-1] if self._checkpoints else self.restore_checkpoint()
        if cp is None:
            return None
        terminal = {
            ExecutionPhase.COMPLETED.value,
            ExecutionPhase.FAILED.value,
            ExecutionPhase.IDLE.value,
        }
        return None if cp.phase in terminal else cp

    def resume_context(self) -> str:
        """Human-readable continuation context for the system prompt.

        Empty string when the last turn completed normally; otherwise a
        compact summary of what was in flight so the next turn can pick
        up instead of restarting.
        """
        cp = self.interrupted_checkpoint()
        if cp is None:
            return ""
        lines = [
            "## Resumed Session",
            f"A previous turn ({cp.phase}) was interrupted before completing.",
            f"Interrupted request: {cp.user_input[:300]}",
        ]
        if cp.tool_calls:
            names = []
            for tc in cp.tool_calls[-5:]:
                name = tc.get("name", "?") if isinstance(tc, dict) else str(tc)
                names.append(str(name))
            lines.append(f"Tool calls already made: {', '.join(names)}")
        if cp.tool_results:
            finished = [
                r for r in cp.tool_results[-5:]
                if isinstance(r, dict) and r.get("success")
            ]
            if finished:
                lines.append(
                    "Completed tool results so far: " + "; ".join(
                        f"{r.get('tool', '?')} -> {str(r.get('output', ''))[:120]}"
                        for r in finished
                    )
                )
        if cp.cwd:
            lines.append(f"Working directory: {cp.cwd}")
        lines.append("Decide whether to continue that work or ask the user.")
        return "\n".join(lines)

    async def execute_tool_with_recovery(
        self,
        tool_name: str,
        tool_input: dict,
        tools: dict[str, Any] | None = None,
        timeout: float = 120.0,
    ) -> dict[str, Any]:
        """Execute a tool with automatic retry and fallback strategies.

        This is the core method that makes Jarvis's tool use reliable.
        It handles:
        - Unknown tools (suggests alternatives)
        - Permission errors (escalates to user)
        - Timeouts (retries with shorter scope)
        - Input errors (auto-corrects common mistakes)
        - Process crashes (restarts gracefully)
        """
        if tools is None and self.agent:
            tools = getattr(self.agent, 'tools', {})

        if not tools or tool_name not in tools:
            suggestion = self._suggest_alternative_tool(tool_name, tools or {})
            return {
                "success": False,
                "error": f"Unknown tool: {tool_name}",
                "suggestion": suggestion,
                "available_tools": list(tools.keys())[:20] if tools else [],
            }

        tool = tools[tool_name]

        # Check if this is a shell/file command — use execution context
        # (see _execute_shell_tool: risk assessment is re-applied there, since
        # routing around the tool object also routes around its blocklist)
        if tool_name in ("terminal_execute", "shell", "run_shell", "execute_command"):
            return self._execute_shell_tool(tool_name, tool_input, timeout)

        if tool_name in ("read_file", "native_read", "cat"):
            return self._execute_read_tool(tool_input)

        if tool_name in ("write_file", "native_write", "edit"):
            return self._execute_write_tool(tool_input)

        # Generic tool execution with retry
        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                result = await self._execute_tool_raw(tool, tool_name, tool_input, timeout)
                if result.get("success", False):
                    self._consecutive_errors = 0
                    self._total_tool_calls += 1
                    return result

                last_error = result.get("error", "Unknown error")

                # Some errors are not retryable
                if any(kw in last_error.lower() for kw in ["permission denied", "not found", "invalid", "blocked"]):
                    break

                if attempt < self.MAX_RETRIES - 1:
                    delay = self.RETRY_DELAYS[min(attempt, len(self.RETRY_DELAYS) - 1)]
                    await asyncio.sleep(delay)

            except asyncio.TimeoutError:
                last_error = f"Tool '{tool_name}' timed out after {timeout}s"
                self._total_errors += 1
                if attempt == self.MAX_RETRIES - 1:
                    break
            except Exception as e:
                last_error = safe_error_message(e)
                self._total_errors += 1
                if attempt == self.MAX_RETRIES - 1:
                    break

        self._consecutive_errors += 1
        self._error_history.append({
            "tool": tool_name,
            "error": last_error,
            "timestamp": time.time(),
            "attempt": attempt + 1,
        })

        # Try fallback
        fallback_result = await self._try_fallback(tool_name, tool_input, tools or {})
        if fallback_result:
            self._total_recoveries += 1
            return fallback_result

        return {"success": False, "error": last_error, "retries": attempt + 1}

    def _execute_shell_tool(self, tool_name: str, tool_input: dict, timeout: float) -> dict:
        """Execute shell commands through the execution context.

        The execution context is used here rather than the tool object so that
        the working directory persists across turns — but that also bypassed
        TerminalExecuteTool.execute(), which is where the high-risk blocklist
        lives. `rm -rf /` and command substitution were classified high-risk and
        refused on the tool path while running unchecked here. Re-apply the same
        assessment before anything reaches subprocess.

        Permission gating (added after the sibling-call fix): the decision is
        delegated to the terminal tool's own needs_permission() — the same
        single source of truth the agent's execute_tool path uses — and the
        grant is checked against the agent's PermissionManager with the same
        contract (DENY refuses, no stored grant requires the user's approval).
        Medium-risk commands (package installs, docker ops, script execution)
        previously ran ungated on this path while being gated on the tool path.
        """
        command = tool_input.get("command", "")
        if not command:
            return {"success": False, "error": "No command provided"}

        risk, risk_reason = _assess_command_risk(command)
        if risk == "high":
            logger.warning(f"Blocked high-risk command via adaptive loop: {risk_reason}")
            return {
                "success": False,
                "error": f"BLOCKED: {risk_reason}",
                "output": (
                    f"BLOCKED: {risk_reason}\nCommand: {command}\n"
                    "This command is refused for safety regardless of caller."
                ),
                "exit_code": -1,
            }

        # Medium-risk commands: same gate the tool path enforces. The tool
        # object is the policy oracle (poka-yoke — no duplicated rules here);
        # the grant lives in the agent's PermissionManager, same check the
        # agent's execute_tool performs: DENY refuses, None (no stored grant)
        # requires the user's approval, anything else proceeds.
        if risk == "medium" and self.agent is not None:
            tool = getattr(self.agent, "tools", {}).get("terminal_execute")
            if tool is not None and hasattr(tool, "needs_permission"):
                try:
                    needs_perm = bool(tool.needs_permission(command=command))
                except Exception as e:
                    logger.debug(f"needs_permission failed, treating as required: {e}")
                    needs_perm = True
                if needs_perm:
                    permission = None
                    perm_mgr = getattr(self.agent, "permissions", None)
                    if perm_mgr is not None and hasattr(perm_mgr, "check_permission"):
                        permission = perm_mgr.check_permission("terminal_execute")
                    if permission == _permission_deny_level():
                        return {
                            "success": False,
                            "error": "Permission denied",
                            "output": "Permission denied by user preference.",
                            "exit_code": -1,
                        }
                    if permission is None:
                        logger.info(
                            f"Permission required via adaptive loop: {command[:80]}"
                        )
                        return {
                            "success": False,
                            "error": "PERMISSION_REQUIRED",
                            "output": (
                                f"PERMISSION_REQUIRED: {risk_reason}\n"
                                f"Command: {command}\n"
                                "Approve terminal_execute to run this command."
                            ),
                            "exit_code": -1,
                        }

        # Update CWD if the user passed cwd
        if "cwd" in tool_input and tool_input["cwd"]:
            success, new_cwd = self._ctx.cd(tool_input["cwd"])
            if not success:
                return {"success": False, "error": f"Cannot change to directory: {new_cwd}"}

        cmd_timeout = min(float(tool_input.get("timeout", timeout)), 300.0)
        result = self._ctx.run(command, timeout=cmd_timeout)

        success = result.get("exit_code", -1) == 0 and not result.get("timed_out", False)

        output_parts = []
        if result.get("stdout"):
            output_parts.append(result["stdout"])
        if result.get("stderr"):
            output_parts.append(f"[stderr] {result['stderr']}")

        combined_output = "\n".join(output_parts)
        if len(combined_output) > 10000:
            combined_output = combined_output[:10000] + "\n... (output truncated)"

        return {
            "success": success,
            "output": combined_output,
            "exit_code": result.get("exit_code", -1),
            "timed_out": result.get("timed_out", False),
            "duration": result.get("duration", 0),
            "cwd": result.get("cwd", self._ctx.cwd),
        }

    def _execute_read_tool(self, tool_input: dict) -> dict:
        """Execute file read through the execution context."""
        path = tool_input.get("path", tool_input.get("file", ""))
        if not path:
            return {"success": False, "error": "No path provided"}

        start_line = int(tool_input.get("start_line", 0))
        end_line = int(tool_input.get("end_line", 0))
        result = self._ctx.read_file(path, start_line=start_line, end_line=end_line)

        if result.get("success", False):
            return {
                "success": True,
                "output": result.get("content", ""),
                "path": result.get("path", ""),
                "lines": result.get("lines", 0),
            }
        return result

    def _execute_write_tool(self, tool_input: dict) -> dict:
        """Execute file write through the execution context."""
        path = tool_input.get("path", tool_input.get("file", ""))
        content = tool_input.get("content", tool_input.get("text", ""))
        mode = tool_input.get("mode", "write")

        if not path:
            return {"success": False, "error": "No path provided"}
        if not content and mode == "write":
            return {"success": False, "error": "No content provided"}

        result = self._ctx.write_file(path, content, mode=mode)
        return result

    async def _execute_tool_raw(self, tool: Any, tool_name: str, tool_input: dict, timeout: float) -> dict:
        """Execute a tool object directly with timeout."""
        try:
            if hasattr(tool, 'safe_execute'):
                try:
                    result = await asyncio.wait_for(
                        tool.safe_execute(**tool_input),
                        timeout=timeout,
                    )
                    if isinstance(result, dict) and "success" in result:
                        return result
                    if hasattr(result, 'to_dict'):
                        return result.to_dict()
                    if isinstance(result, str):
                        return {"success": True, "output": result}
                    return {"success": True, "output": str(result)}
                except asyncio.TimeoutError:
                    return {"success": False, "error": f"Tool '{tool_name}' timed out after {timeout}s"}
                except Exception as e:
                    logger.debug(f"safe_execute fallback for '{tool_name}': {safe_error_message(e)}")

            if asyncio.iscoroutinefunction(tool.execute):
                result = await asyncio.wait_for(tool.execute(**tool_input), timeout=timeout)
            else:
                result = await asyncio.wait_for(
                    asyncio.to_thread(tool.execute, **tool_input),
                    timeout=timeout,
                )

            if isinstance(result, dict):
                if "success" in result:
                    return result
                return {"success": True, "output": json.dumps(result)[:10000]}
            if isinstance(result, str):
                return {"success": True, "output": result[:10000]}
            return {"success": True, "output": str(result)[:10000]}

        except asyncio.TimeoutError:
            return {"success": False, "error": f"Tool '{tool_name}' timed out after {timeout}s"}
        except TypeError as e:
            return {"success": False, "error": f"Tool input error: {safe_error_message(e)}"}
        except Exception as e:
            return {"success": False, "error": safe_error_message(e)}

    def _suggest_alternative_tool(self, tool_name: str, tools: dict[str, Any]) -> str | None:
        """Suggest an alternative tool when the requested one isn't found."""
        from core.tool_registry import find_best_tool_for_task
        name_lower = tool_name.lower().replace("-", "_").replace(" ", "_")
        if name_lower in tools:
            return name_lower

        for registered_name in tools:
            if registered_name.replace("_", "").replace("-", "") == name_lower.replace("_", "").replace("-", ""):
                return registered_name

        results = find_best_tool_for_task(tool_name.replace("_", " "))
        if results:
            return results[0][0]
        return None

    async def _try_fallback(self, failed_tool: str, tool_input: dict, tools: dict[str, Any]) -> dict | None:
        """Try a fallback tool when the primary fails."""
        FALLBACKS = {
            "terminal_execute": "shell",
            "shell": "terminal_execute",
            "read_file": "native_read",
            "native_read": "read_file",
            "write_file": "native_write",
            "native_write": "write_file",
            "web_search": "browser_navigate",
            "data_analysis": "terminal_execute",
        }

        fallback_name = FALLBACKS.get(failed_tool)
        if fallback_name and fallback_name in tools:
            logger.info(f"Trying fallback tool '{fallback_name}' for failed '{failed_tool}'")
            try:
                fallback_tool = tools[fallback_name]
                if asyncio.iscoroutinefunction(fallback_tool.execute):
                    result = await fallback_tool.execute(**tool_input)
                else:
                    result = fallback_tool.execute(**tool_input)
                if isinstance(result, dict) and result.get("success"):
                    return result
            except Exception as e:
                logger.debug(f"Fallback tool '{fallback_name}' failed: {safe_error_message(e)}")
        return None

    async def run_persistent(
        self,
        user_input: str,
        system_prompt: str,
        messages: list[dict] | None = None,
        tools: dict[str, Any] | None = None,
        max_iterations: int = 10,
        on_progress: Any = None,
    ) -> dict[str, Any]:
        """Run a persistent agent loop that maintains context across iterations.

        This is the main entry point for the adaptive agent. It:
        1. Loads execution context (CWD, env, process history)
        2. Runs the LLM with system prompt including context awareness
        3. Executes tool calls with self-healing recovery
        4. Checkpoints state for resume capability
        5. Auto-compresses context when approaching limits
        6. Returns complete results with metadata

        Args:
            user_input: The user's message
            system_prompt: System instructions
            messages: Conversation history
            tools: Available tools dict
            max_iterations: Maximum tool-call iterations
            on_progress: Progress callback

        Returns:
            Dict with response, tool_calls, tool_results, checkpoint, metadata
        """
        if tools is None and self.agent:
            tools = getattr(self.agent, 'tools', {})

        self._turn_count += 1
        self._last_input_time = time.time()
        self._phase = ExecutionPhase.THINKING

        if messages is None:
            messages = []

        context_awareness = self._build_context_awareness()
        enhanced_prompt = f"{system_prompt}\n\n{context_awareness}" if context_awareness else system_prompt

        all_tool_calls = []
        all_tool_results = []
        response_text = ""
        iterations = 0

        self.checkpoint(user_input)
        self._ctx.save_state()

        try:
            for iteration in range(max_iterations):
                iterations = iteration + 1
                self._phase = ExecutionPhase.THINKING

                think_result = await self._think(user_input, enhanced_prompt, messages, tools)

                if not think_result:
                    self._phase = ExecutionPhase.COMPLETED
                    response_text = "I'm unable to process that right now. Please try again."
                    break

                content, tool_calls = think_result

                if not tool_calls:
                    response_text = content or ""
                    self._phase = ExecutionPhase.COMPLETED
                    break

                self._phase = ExecutionPhase.EXECUTING
                self._total_tool_calls += 1

                for tc in tool_calls:
                    tool_name = tc.get("name", tc.get("function", {}).get("name", ""))
                    tool_args = tc.get("arguments", tc.get("function", {}).get("arguments", {}))

                    if isinstance(tool_args, str):
                        try:
                            tool_args = json.loads(tool_args)
                        except json.JSONDecodeError:
                            tool_args = {}

                    result = await self.execute_tool_with_recovery(
                        tool_name, tool_args, tools, timeout=120.0,
                    )

                    all_tool_calls.append({"name": tool_name, "arguments": tool_args})
                    all_tool_results.append({
                        "tool": tool_name,
                        "success": result.get("success", False),
                        "output": result.get("output", result.get("error", "")),
                    })

                    messages.append({
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{
                            "id": f"call_{iterations}_{len(all_tool_calls)}",
                            "type": "function",
                            "function": {"name": tool_name, "arguments": json.dumps(tool_args)},
                        }],
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": f"call_{iterations}_{len(all_tool_calls)}",
                        "content": result.get("output", result.get("error", ""))[:4000],
                    })

                    if on_progress:
                        on_progress({
                            "type": "tool_result",
                            "tool": tool_name,
                            "success": result.get("success", False),
                            "iteration": iteration + 1,
                        })

                # Check if we should continue
                all_failed = all(not r.get("success", False) for r in all_tool_results[-len(tool_calls):])
                if all_failed and self._consecutive_errors >= self._max_consecutive_errors:
                    self._phase = ExecutionPhase.ERROR_RECOVERY
                    response_text = "I've encountered multiple errors. Let me try a different approach."
                    break

                # Auto-compress context if getting long
                if len(messages) > 20:
                    messages = self._compress_messages(messages)

            if not response_text and all_tool_results:
                response_text = "Task completed."

        except Exception as e:
            self._phase = ExecutionPhase.FAILED
            self._consecutive_errors += 1
            response_text = f"Error during execution: {safe_error_message(e)}"
            logger.error(f"AdaptiveAgentLoop error: {traceback.format_exc()}")

        self.checkpoint(
            user_input=user_input,
            response=response_text,
            tool_calls=all_tool_calls,
            tool_results=all_tool_results,
        )

        return {
            "response": response_text,
            "tool_calls": all_tool_calls,
            "tool_results": all_tool_results,
            "iterations": iterations,
            "phase": self._phase.value,
            "context": self._ctx.get_status(),
            "errors": self._error_history[-5:],
            "success": self._phase != ExecutionPhase.FAILED,
        }

    async def _think(self, user_input: str, system_prompt: str, messages: list, tools: dict) -> tuple[str, list] | None:
        """Call the LLM and extract tool calls."""
        if not self.llm:
            return None

        try:
            if hasattr(self.llm, 'chat_with_tools'):
                tool_schemas = self._build_tool_schemas(tools)
                response = await self.llm.chat_with_tools(
                    user_input=user_input,
                    system_prompt=system_prompt,
                    conversation_history=messages,
                    tools=tool_schemas,
                )
                content = ""
                if isinstance(response, dict):
                    content = response.get("content", "") or response.get("message", {}).get("content", "")
                    tool_calls = response.get("tool_calls", []) or response.get("message", {}).get("tool_calls", [])
                    return content, tool_calls
                elif isinstance(response, str):
                    return response, []

            elif hasattr(self.llm, 'chat'):
                response = await self.llm.chat(
                    user_input=user_input,
                    system_prompt=system_prompt,
                    conversation_history=messages,
                )
                content = response if isinstance(response, str) else str(response)
                return content, []

        except Exception as e:
            logger.error(f"Think error: {safe_error_message(e)}")
            self._total_errors += 1

        return None

    def _build_tool_schemas(self, tools: dict) -> list[dict]:
        """Build tool schemas from tool registry."""
        schemas = []
        for name, tool in (tools or {}).items():
            desc = getattr(tool, 'description', '') or f"Execute {name}"
            params = getattr(tool, 'parameters', {})
            if isinstance(params, dict) and "properties" not in params:
                properties = {k: {"type": "string", "description": v} for k, v in params.items()}
                params = {"type": "object", "properties": properties}
            schemas.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": desc,
                    "parameters": params or {"type": "object", "properties": {}},
                }
            })
        return schemas

    def _build_context_awareness(self) -> str:
        """Build context awareness information for the system prompt."""
        ctx = self._ctx.get_status()
        parts = [
            f"## Current Execution Context",
            f"Working directory: {ctx['cwd']}",
            f"Commands executed: {ctx['command_count']}",
            f"Session uptime: {ctx['uptime_seconds']}s",
        ]

        recent_cmds = self._ctx.get_recent_commands(3)
        if recent_cmds:
            parts.append("Recent commands:")
            for cmd in recent_cmds:
                exit_code = cmd.get("exit_code", "?")
                duration = cmd.get("duration", 0)
                parts.append(f"  $ {cmd.get('command', '?')} (exit={exit_code}, {duration}s)")

        recent_files = self._ctx.get_recent_files(3)
        if recent_files:
            parts.append("Recent file operations:")
            for f in recent_files:
                parts.append(f"  {f['op_type']}: {f['path']}")

        return "\n".join(parts)

    def _compress_messages(self, messages: list[dict]) -> list[dict]:
        """Compress old messages to fit context window."""
        if len(messages) <= 10:
            return messages

        # Keep first 3 (system + recent context) and last 7 (conversation)
        system_messages = [m for m in messages[:3] if m.get("role") == "system"]
        recent = messages[-7:]

        # Summarize middle messages
        middle = messages[3:-7]
        if middle:
            summary_parts = []
            for m in middle:
                role = m.get("role", "unknown")
                content = m.get("content", "")
                if isinstance(content, str) and len(content) > 200:
                    content = content[:200] + "..."
                summary_parts.append(f"[{role}] {content}")

            summary = f"[Earlier conversation summary]\n" + "\n".join(summary_parts[:10])
            system_messages.append({"role": "system", "content": summary})

        return system_messages + recent

    def recover_from_error(self, error: str, context: dict | None = None) -> dict[str, Any]:
        """Attempt to recover from an error state.

        Enhanced (v2.1) with:
        - More granular error classification
        - Context-aware strategy selection
        - Partial output preservation guidance
        - Error deduplication
        """
        self._phase = ExecutionPhase.ERROR_RECOVERY
        self._total_errors += 1
        self._consecutive_errors += 1

        error_lower = error.lower()
        recovery_strategies = []
        context_hints = {}

        if "timeout" in error_lower or "timed out" in error_lower:
            recovery_strategies.append("retry_shorter")
            recovery_strategies.append("fallback_tool")
            recovery_strategies.append("reduce_scope")
            context_hints["suggested_timeout"] = 30.0
        elif "permission" in error_lower or "access denied" in error_lower or "forbidden" in error_lower:
            recovery_strategies.append("request_approval")
            recovery_strategies.append("alternative_path")
            recovery_strategies.append("skip_step")
        elif "not found" in error_lower or "no such file" in error_lower or "does not exist" in error_lower:
            recovery_strategies.append("search_alternative")
            recovery_strategies.append("list_available")
            recovery_strategies.append("create_missing")
        elif "connection" in error_lower or "refused" in error_lower or "network" in error_lower:
            recovery_strategies.append("retry_with_backoff")
            recovery_strategies.append("offline_mode")
            recovery_strategies.append("cache_result")
        elif "memory" in error_lower or "oom" in error_lower or "out of memory" in error_lower:
            recovery_strategies.append("context_reduction")
            recovery_strategies.append("simpler_query")
        elif "rate limit" in error_lower or "too many" in error_lower or "429" in error_lower:
            recovery_strategies.append("retry_with_backoff")
            recovery_strategies.append("cache_result")
            context_hints["backoff_seconds"] = 10.0
        elif "json" in error_lower or "parse" in error_lower or "decode" in error_lower:
            recovery_strategies.append("retry_clean_input")
            recovery_strategies.append("simpler_query")
        elif "interrupt" in error_lower or "cancelled" in error_lower or "cancel" in error_lower:
            recovery_strategies.append("resume_from_checkpoint")
            recovery_strategies.append("skip_step")
        else:
            recovery_strategies.append("retry")
            recovery_strategies.append("fallback")

        if self._consecutive_errors >= self._max_consecutive_errors:
            self._phase = ExecutionPhase.FAILED
            return {
                "recovered": False,
                "message": "Too many consecutive errors. Stopping execution.",
                "consecutive_errors": self._consecutive_errors,
                "strategies": recovery_strategies,
                "context_hints": context_hints,
            }

        return {
            "recovered": True,
            "message": f"Attempting recovery: {recovery_strategies[0]}",
            "consecutive_errors": self._consecutive_errors,
            "strategies": recovery_strategies,
            "context_hints": context_hints,
        }

    def recover_mid_stream(
        self,
        error: str,
        partial_output: str = "",
        tool_calls_made: list[dict] | None = None,
        tool_results: list[dict] | None = None,
    ) -> dict[str, Any]:
        """Recover from a mid-stream error, preserving partial output.

        This is the key method for handling errors during long-running
        streaming tasks where the user has already received partial output.

        Returns:
            Dict with recovery plan including preserved partial output.
        """
        recovery = self.recover_from_error(error, {
            "partial_output_len": len(partial_output),
            "tool_calls_made": len(tool_calls_made or []),
        })

        recovery["partial_output"] = partial_output
        recovery["partial_output_len"] = len(partial_output)
        recovery["tool_calls_made"] = tool_calls_made or []
        recovery["tool_results"] = tool_results or []

        if partial_output:
            recovery["message"] += f" ({len(partial_output)} chars preserved)"

        return recovery


    async def run_persistent_stream(
        self,
        user_input: str,
        system_prompt: str,
        messages: list[dict] | None = None,
        tools: dict[str, Any] | None = None,
        max_iterations: int = 10,
    ) -> AsyncIterator[dict]:
        """Streaming version of run_persistent that yields events as they happen.

        Yields dict events:
        - {"type": "thinking", "text": "..."}
        - {"type": "tool_call", "tool": "name", "arguments": {...}}
        - {"type": "tool_result", "tool": "name", "success": bool, "output": "..."}
        - {"type": "text", "text": "..."}
        - {"type": "recovery", "strategy": "...", "attempt": int}
        - {"type": "resume", "turn_number": int, "phase": str}
        - {"type": "complete", "success": bool, "iterations": int, "tool_calls": int}
        - {"type": "error", "error": "..."}
        """
        if tools is None and self.agent:
            tools = getattr(self.agent, 'tools', {})

        self._turn_count += 1
        self._last_input_time = time.time()
        self._phase = ExecutionPhase.THINKING

        if messages is None:
            messages = []

        # Crash recovery: if the last persisted turn never completed,
        # surface it so this turn can pick up where it stopped.
        resume_ctx = ""
        try:
            interrupted = self.interrupted_checkpoint()
            if interrupted is not None:
                resume_ctx = self.resume_context()
                if resume_ctx:
                    yield {"type": "resume", "turn_number": interrupted.turn_number,
                           "phase": interrupted.phase,
                           "user_input": interrupted.user_input[:200]}
                    logger.info(
                        f"Resuming from interrupted turn {interrupted.turn_number} "
                        f"(phase={interrupted.phase})"
                    )
        except Exception as e:
            logger.debug(f"Interrupted-checkpoint check failed: {e}")

        context_awareness = self._build_context_awareness()
        enhanced_prompt = f"{system_prompt}\n\n{context_awareness}" if context_awareness else system_prompt
        if resume_ctx:
            enhanced_prompt = f"{enhanced_prompt}\n\n{resume_ctx}"

        all_tool_calls = []
        all_tool_results = []

        self.checkpoint(user_input)
        self._ctx.save_state()

        try:
            for iteration in range(max_iterations):
                self._phase = ExecutionPhase.THINKING
                yield {"type": "thinking", "text": f"Iteration {iteration + 1}/{max_iterations}"}

                think_result = await self._think(user_input, enhanced_prompt, messages, tools)

                if not think_result:
                    yield {"type": "text", "text": "I'm unable to process that right now."}
                    self._phase = ExecutionPhase.COMPLETED
                    break

                content, tool_calls = think_result

                if not tool_calls:
                    self._phase = ExecutionPhase.COMPLETED
                    if content:
                        for i in range(0, len(content), 20):
                            yield {"type": "text", "text": content[i:i+20]}
                            await asyncio.sleep(0.01)
                    break

                self._phase = ExecutionPhase.EXECUTING
                self._total_tool_calls += 1

                for tc in tool_calls:
                    tool_name = tc.get("name", tc.get("function", {}).get("name", ""))
                    tool_args = tc.get("arguments", tc.get("function", {}).get("arguments", {}))

                    if isinstance(tool_args, str):
                        try:
                            tool_args = json.loads(tool_args)
                        except json.JSONDecodeError:
                            yield {"type": "error", "error": f"Invalid JSON args for {tool_name}"}
                            continue

                    yield {"type": "tool_call", "tool": tool_name, "arguments": tool_args}

                    result = await self.execute_tool_with_recovery(
                        tool_name, tool_args, tools, timeout=120.0,
                    )

                    all_tool_calls.append({"name": tool_name, "arguments": tool_args})
                    all_tool_results.append({
                        "tool": tool_name,
                        "success": result.get("success", False),
                        "output": result.get("output", result.get("error", "")),
                    })

                    yield {
                        "type": "tool_result",
                        "tool": tool_name,
                        "success": result.get("success", False),
                        "output": result.get("output", result.get("error", ""))[:2000],
                    }

                    if not result.get("success", False) and result.get("retries", 0) > 0:
                        yield {
                            "type": "recovery",
                            "strategy": "retry",
                            "attempt": result.get("retries", 0),
                        }

                    messages.append({
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{
                            "id": f"call_{iteration}_{len(all_tool_calls)}",
                            "type": "function",
                            "function": {"name": tool_name, "arguments": json.dumps(tool_args)},
                        }],
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": f"call_{iteration}_{len(all_tool_calls)}",
                        "content": result.get("output", result.get("error", ""))[:4000],
                    })

                all_failed = all(not r.get("success", False) for r in all_tool_results[-len(tool_calls):])
                if all_failed and self._consecutive_errors >= self._max_consecutive_errors:
                    self._phase = ExecutionPhase.ERROR_RECOVERY
                    yield {"type": "error", "error": "Multiple consecutive errors. Stopping."}
                    break

                if len(messages) > 20:
                    messages = self._compress_messages(messages)

        except Exception as e:
            self._phase = ExecutionPhase.FAILED
            self._consecutive_errors += 1
            yield {"type": "error", "error": safe_error_message(e)}
            logger.error(f"AdaptiveAgentLoop stream error: {traceback.format_exc()}")

        # A turn that exhausted max_iterations (or stopped on repeated tool
        # errors) still ENDED — mark it terminal so the persisted phase
        # cannot masquerade as a crash for the next turn.
        if self._phase in (ExecutionPhase.THINKING, ExecutionPhase.EXECUTING,
                           ExecutionPhase.ERROR_RECOVERY):
            self._phase = (
                ExecutionPhase.FAILED
                if self._consecutive_errors >= self._max_consecutive_errors
                else ExecutionPhase.COMPLETED
            )

        self.checkpoint(
            user_input=user_input,
            response="",
            tool_calls=all_tool_calls,
            tool_results=all_tool_results,
        )

        yield {
            "type": "complete",
            "success": self._phase != ExecutionPhase.FAILED,
            "iterations": iteration + 1 if 'iteration' in locals() else 0,
            "tool_calls": len(all_tool_calls),
            "phase": self._phase.value,
        }


_adaptive_loop: AdaptiveAgentLoop | None = None
_loop_lock = threading.Lock()


def get_adaptive_agent_loop(agent=None, llm_client=None) -> AdaptiveAgentLoop:
    """Thread-safe singleton for AdaptiveAgentLoop with late binding.

    A cached instance missing a dependency is UPGRADED when a later
    caller supplies it. Without this, the read-only status endpoint
    (which calls with no llm_client) could create the singleton first
    and permanently poison it: ``_think()`` returns None whenever
    ``self.llm`` is falsy, so every subsequent agent-loop chat would
    silently produce no output for the life of the process.
    """
    global _adaptive_loop
    with _loop_lock:
        if _adaptive_loop is None:
            _adaptive_loop = AdaptiveAgentLoop(agent=agent, llm_client=llm_client)
        else:
            if _adaptive_loop.agent is None and agent is not None:
                _adaptive_loop.agent = agent
            if _adaptive_loop.llm is None and llm_client is not None:
                _adaptive_loop.llm = llm_client
        return _adaptive_loop


def reset_adaptive_agent_loop() -> None:
    """Drop the singleton (tests / reconfiguration)."""
    global _adaptive_loop
    with _loop_lock:
        _adaptive_loop = None