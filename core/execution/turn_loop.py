"""Turn-Loop Execution System.

Implements stateful execution loops with turn tracking, stop reasons,
and context management, inspired by Claw Code patterns.

Features:
- Turn-loop management with max turns
- Stop reasons (completed, permission_required, error, max_turns)
- Structured turn results
- Event streaming during execution
"""
import asyncio
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterator, Optional

logger = logging.getLogger(__name__)


class StopReason(Enum):
    """Reasons for stopping a turn loop."""
    COMPLETED = "completed"
    PERMISSION_REQUIRED = "permission_required"
    ERROR = "error"
    MAX_TURNS = "max_turns"
    USER_CANCEL = "user_cancel"
    TIMEOUT = "timeout"


class StreamEventType(Enum):
    """Types of stream events during execution."""
    TURN_START = "turn_start"
    COMMAND_MATCH = "command_match"
    TOOL_MATCH = "tool_match"
    COMMAND_EXECUTE = "command_execute"
    TOOL_EXECUTE = "tool_execute"
    PERMISSION_DENIAL = "permission_denial"
    TURN_COMPLETE = "turn_complete"
    ERROR = "error"


@dataclass
class StreamEvent:
    """A stream event during execution."""
    event_type: StreamEventType
    timestamp: float
    data: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.event_type.value,
            "timestamp": self.timestamp,
            "data": self.data,
        }


@dataclass
class TurnResult:
    """Result of a single turn in the execution loop."""
    turn: int
    output: str
    stop_reason: StopReason
    matched_commands: list[str] = field(default_factory=list)
    matched_tools: list[str] = field(default_factory=list)
    executed_commands: list[str] = field(default_factory=list)
    executed_tools: list[str] = field(default_factory=list)
    permission_denials: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    end_time: float = field(default_factory=time.time)
    
    @property
    def duration(self) -> float:
        """Time taken for this turn in seconds."""
        return self.end_time - self.start_time


@dataclass
class PermissionDenial:
    """A permission denial for a tool/command."""
    name: str
    kind: str  # 'command' or 'tool'
    reason: str
    severity: str = "hard"  # 'soft', 'hard', 'critical'
    requires_approval: bool = True
    context: dict[str, Any] = field(default_factory=dict)


class ExecutionRegistry:
    """Registry for commands and tools with unified execution.
    
    Manages registration and execution of both commands and tools
    with permission checking and error handling.
    """
    
    def __init__(self):
        self._commands: dict[str, Callable] = {}
        self._tools: dict[str, Callable] = {}
        self._command_metadata: dict[str, dict] = {}
        self._tool_metadata: dict[str, dict] = {}
        self._lock = threading.Lock()
    
    def register_command(
        self,
        name: str,
        executor: Callable,
        metadata: Optional[dict] = None,
    ) -> None:
        """Register a command executor."""
        with self._lock:
            self._commands[name] = executor
            self._command_metadata[name] = metadata or {}
        
        logger.debug(f"Registered command: {name}")
    
    def register_tool(
        self,
        name: str,
        executor: Callable,
        metadata: Optional[dict] = None,
    ) -> None:
        """Register a tool executor."""
        with self._lock:
            self._tools[name] = executor
            self._tool_metadata[name] = metadata or {}
        
        logger.debug(f"Registered tool: {name}")
    
    def get_command(self, name: str) -> Optional[Callable]:
        """Get a command executor by name."""
        return self._commands.get(name)
    
    def get_tool(self, name: str) -> Optional[Callable]:
        """Get a tool executor by name."""
        return self._tools.get(name)
    
    def execute(
        self,
        kind: str,
        name: str,
        input_data: Any,
    ) -> tuple[Any, Optional[Exception]]:
        """Execute a command or tool.
        
        Args:
            kind: 'command' or 'tool'
            name: Name of command/tool
            input_data: Input to executor
            
        Returns:
            Tuple of (result, error)
        """
        if kind == "command":
            executor = self._commands.get(name)
        else:
            executor = self._tools.get(name)
        
        if executor is None:
            return None, ValueError(f"{kind} '{name}' not found")
        
        try:
            if asyncio.iscoroutinefunction(executor):
                # Async executor - needs to be awaited separately
                return executor, None
            else:
                result = executor(input_data)
                return result, None
        except Exception as e:
            logger.error(f"Error executing {kind} '{name}': {e}")
            return None, e
    
    def list_commands(self) -> list[str]:
        """List all registered commands."""
        return list(self._commands.keys())
    
    def list_tools(self) -> list[str]:
        """List all registered tools."""
        return list(self._tools.keys())
    
    def get_command_metadata(self, name: str) -> dict:
        """Get metadata for a command."""
        return self._command_metadata.get(name, {})
    
    def get_tool_metadata(self, name: str) -> dict:
        """Get metadata for a tool."""
        return self._tool_metadata.get(name, {})


class TurnLoopManager:
    """Manages turn-loop execution for commands and tools.
    
    Implements the stateful execution pattern from Claw Code with:
    - Turn tracking
    - Stop reasons
    - Permission denials
    - Event streaming
    
    Usage:
        manager = TurnLoopManager(registry)
        
        # Run turn loop
        results = manager.run_turn_loop(
            prompt="Calculate fibonacci",
            max_turns=5
        )
        
        # Stream events
        for event in manager.stream_turn_loop(prompt="Analyze data"):
            print(event.event_type, event.data)
    """
    
    MAX_TURNS = 10
    DEFAULT_TIMEOUT = 300.0  # 5 minutes
    
    # Tools/commands that require explicit permission
    PERMISSION_GATED = {
        "bash", "shell", "execute", "delete", "write_file",
        "system", "restart", "shutdown", "modify_settings",
    }
    
    # Destructive tools that always require approval
    DESTRUCTIVE_TOOLS = {
        "delete", "remove", "format", "wipe", "reset",
        "shutdown", "restart", "kill", "terminate",
    }
    
    def __init__(self, registry: Optional[ExecutionRegistry] = None):
        self.registry = registry or ExecutionRegistry()
        self._history: deque[TurnResult] = deque(maxlen=100)
        self._lock = threading.Lock()
        self._event_subscribers: list[Callable[[StreamEvent], None]] = []
    
    def subscribe_to_events(self, callback: Callable[[StreamEvent], None]) -> None:
        """Subscribe to stream events."""
        self._event_subscribers.append(callback)
    
    def unsubscribe_from_events(self, callback: Callable[[StreamEvent], None]) -> None:
        """Unsubscribe from stream events."""
        if callback in self._event_subscribers:
            self._event_subscribers.remove(callback)
    
    def _emit_event(self, event_type: StreamEventType, data: dict[str, Any]) -> StreamEvent:
        """Emit a stream event to all subscribers."""
        event = StreamEvent(
            event_type=event_type,
            timestamp=time.time(),
            data=data,
        )
        
        for callback in self._event_subscribers:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Event subscriber error: {e}")
        
        return event
    
    def check_permissions(
        self,
        name: str,
        kind: str,
        context: Optional[dict] = None,
    ) -> Optional[PermissionDenial]:
        """Check if a command/tool requires permission.
        
        Args:
            name: Command/tool name
            kind: 'command' or 'tool'
            context: Execution context
            
        Returns:
            PermissionDenial if permission required, None otherwise
        """
        name_lower = name.lower()
        
        # Check for destructive tools
        if name_lower in self.DESTRUCTIVE_TOOLS:
            return PermissionDenial(
                name=name,
                kind=kind,
                reason=f"'{name}' is a destructive operation",
                severity="critical",
                requires_approval=True,
                context=context or {},
            )
        
        # Check for permission-gated tools
        if name_lower in self.PERMISSION_GATED:
            return PermissionDenial(
                name=name,
                kind=kind,
                reason=f"'{name}' requires explicit permission",
                severity="hard",
                requires_approval=True,
                context=context or {},
            )
        
        return None
    
    def run_turn_loop(
        self,
        prompt: str,
        matched_commands: list[str] = None,
        matched_tools: list[str] = None,
        max_turns: int = None,
        timeout: float = None,
        context: Optional[dict] = None,
    ) -> list[TurnResult]:
        """Run a turn-loop execution.
        
        Args:
            prompt: The user prompt
            matched_commands: Commands matched by router
            matched_tools: Tools matched by router
            max_turns: Maximum turns (default: MAX_TURNS)
            timeout: Timeout in seconds (default: DEFAULT_TIMEOUT)
            context: Execution context
            
        Returns:
            List of turn results
        """
        max_turns = max_turns or self.MAX_TURNS
        timeout = timeout or self.DEFAULT_TIMEOUT
        matched_commands = matched_commands or []
        matched_tools = matched_tools or []
        
        results: list[TurnResult] = []
        start_time = time.time()
        
        for turn in range(max_turns):
            # Check timeout
            if time.time() - start_time > timeout:
                result = TurnResult(
                    turn=turn + 1,
                    output="",
                    stop_reason=StopReason.TIMEOUT,
                    errors=["Execution timed out"],
                )
                results.append(result)
                break
            
            # Emit turn start event
            self._emit_event(StreamEventType.TURN_START, {
                "turn": turn + 1,
                "prompt": prompt,
            })
            
            # Initialize turn result
            result = TurnResult(
                turn=turn + 1,
                output="",
                stop_reason=StopReason.COMPLETED,
                matched_commands=matched_commands,
                matched_tools=matched_tools,
            )
            
            # Execute commands
            for cmd_name in matched_commands:
                denial = self.check_permissions(cmd_name, "command", context)
                
                if denial:
                    result.permission_denials.append(denial.to_dict())
                    self._emit_event(StreamEventType.PERMISSION_DENIAL, {
                        "name": cmd_name,
                        "kind": "command",
                        "reason": denial.reason,
                    })
                    continue
                
                self._emit_event(StreamEventType.COMMAND_EXECUTE, {
                    "name": cmd_name,
                    "turn": turn + 1,
                })
                
                output, error = self.registry.execute("command", cmd_name, prompt)
                
                if error:
                    result.errors.append(f"Command '{cmd_name}' error: {error}")
                    result.stop_reason = StopReason.ERROR
                else:
                    result.executed_commands.append(cmd_name)
            
            # Execute tools
            for tool_name in matched_tools:
                denial = self.check_permissions(tool_name, "tool", context)
                
                if denial:
                    result.permission_denials.append(denial.to_dict())
                    self._emit_event(StreamEventType.PERMISSION_DENIAL, {
                        "name": tool_name,
                        "kind": "tool",
                        "reason": denial.reason,
                    })
                    continue
                
                self._emit_event(StreamEventType.TOOL_EXECUTE, {
                    "name": tool_name,
                    "turn": turn + 1,
                })
                
                output, error = self.registry.execute("tool", tool_name, prompt)
                
                if error:
                    result.errors.append(f"Tool '{tool_name}' error: {error}")
                    result.stop_reason = StopReason.ERROR
                else:
                    result.executed_tools.append(tool_name)
            
            # Determine stop reason
            if result.permission_denials:
                result.stop_reason = StopReason.PERMISSION_REQUIRED
            elif result.errors:
                result.stop_reason = StopReason.ERROR
            
            result.end_time = time.time()
            results.append(result)
            
            # Emit turn complete event
            self._emit_event(StreamEventType.TURN_COMPLETE, {
                "turn": turn + 1,
                "stop_reason": result.stop_reason.value,
                "duration": result.duration,
            })
            
            # Stop if not completed
            if result.stop_reason != StopReason.COMPLETED:
                break
        
        # Store in history
        with self._lock:
            for result in results:
                self._history.append(result)
        
        # Handle max turns
        if len(results) >= max_turns and results[-1].stop_reason == StopReason.COMPLETED:
            results[-1].stop_reason = StopReason.MAX_TURNS
        
        return results
    
    def stream_turn_loop(
        self,
        prompt: str,
        matched_commands: list[str] = None,
        matched_tools: list[str] = None,
        max_turns: int = None,
    ) -> Iterator[StreamEvent]:
        """Stream turn-loop execution events.
        
        Yields stream events as execution progresses.
        
        Args:
            prompt: The user prompt
            matched_commands: Commands matched by router
            matched_tools: Tools matched by router
            max_turns: Maximum turns
            
        Yields:
            StreamEvent objects
        """
        max_turns = max_turns or self.MAX_TURNS
        matched_commands = matched_commands or []
        matched_tools = matched_tools or []
        
        # Temporarily collect events
        events: list[StreamEvent] = []
        
        def collect_events(event: StreamEvent):
            events.append(event)
        
        self.subscribe_to_events(collect_events)
        
        try:
            # Run turn loop (events will be collected)
            self.run_turn_loop(
                prompt=prompt,
                matched_commands=matched_commands,
                matched_tools=matched_tools,
                max_turns=max_turns,
            )
            
            # Yield collected events
            for event in events:
                yield event
        finally:
            self.unsubscribe_from_events(collect_events)
    
    def get_history(self, limit: int = 10) -> list[TurnResult]:
        """Get turn history."""
        with self._lock:
            return list(self._history)[-limit:]
    
    def clear_history(self) -> None:
        """Clear turn history."""
        with self._lock:
            self._history.clear()
        logger.info("Turn history cleared")


# Singleton instances
_execution_registry: Optional[ExecutionRegistry] = None
_turn_loop_manager: Optional[TurnLoopManager] = None
_registry_lock = threading.Lock()


def get_execution_registry() -> ExecutionRegistry:
    """Get the singleton execution registry."""
    global _execution_registry
    if _execution_registry is None:
        with _registry_lock:
            if _execution_registry is None:
                _execution_registry = ExecutionRegistry()
    return _execution_registry


def get_turn_loop_manager() -> TurnLoopManager:
    """Get the singleton turn loop manager."""
    global _turn_loop_manager
    if _turn_loop_manager is None:
        with _registry_lock:
            if _turn_loop_manager is None:
                _turn_loop_manager = TurnLoopManager(get_execution_registry())
    return _turn_loop_manager