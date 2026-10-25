# Claw Code Analysis - Learnings for Jarvis

## Overview
The Claw Code project is a Python/Rust port of an agent harness system (Claude Code). It provides excellent patterns for command execution, tool orchestration, and runtime management that Jarvis can adopt.

## Key Architecture Patterns

### 1. **Command/Tool Routing System**
**Pattern**: Intelligent routing of prompts to appropriate commands/tools based on token matching and scoring.

```python
# Claw Code Pattern
@dataclass(frozen=True)
class RoutedMatch:
    kind: str  # 'command' or 'tool'
    name: str
    source_hint: str
    score: int

def route_prompt(self, prompt: str, limit: int = 5) -> list[RoutedMatch]:
    tokens = {token.lower() for token in prompt.replace('/', ' ').replace('-', ' ').split()}
    # Score matches based on token presence
    # Return top-N matches
```

**Application to Jarvis**:
- Implement intelligent routing in `intelligence_hub.py`
- Score and route queries to appropriate subsystems
- Support multi-match execution (execute top-N relevant commands/tools)

### 2. **Turn-Loop Execution**
**Pattern**: Stateful execution loop with turn tracking, stop reasons, and context management.

```python
def run_turn_loop(self, prompt: str, max_turns: int = 3) -> list[TurnResult]:
    results: list[TurnResult] = []
    for turn in range(max_turns):
        result = engine.submit_message(turn_prompt, commands, tools, denials)
        results.append(result)
        if result.stop_reason != 'completed':
            break
    return results
```

**Application to Jarvis**:
- Enhance `agent.py` with formal turn-loop management
- Support stop reasons: 'completed', 'permission_required', 'error', 'max_turns'
- Track turn state for resumable execution

### 3. **Permission Context System**
**Pattern**: Gate access to destructive/privileged tools with permission contexts.

```python
@dataclass
class ToolPermissionContext:
    allowed_tools: frozenset[str]
    denied_tools: frozenset[str]
    denied_prefixes: frozenset[str]

def infer_permission_denials(self, matches: list[RoutedMatch]) -> list[PermissionDenial]:
    if 'bash' in match.name.lower():
        return PermissionDenial(tool_name=match.name, reason='destructive')
```

**Application to Jarvis**:
- Already have permission system in `agent.py`
- Enhance with formal denial reasons and user approval flow
- Support permission inheritance for tool chains

### 4. **Session Persistence & History**
**Pattern**: Structured transcript storage with flush semantics.

```python
class HistoryLog:
    def add(self, kind: str, message: str):
        self.entries.append(HistoryEntry(kind=kind, message=message, timestamp=time.time()))
    
    def as_markdown(self) -> str:
        return '\n'.join(f'- [{e.kind}] {e.message}' for e in self.entries)

class TranscriptStore:
    def persist_session(self) -> str:
        # Save session to file, return path
    def flush(self):
        # Clear in-memory transcript after persistence
```

**Application to Jarvis**:
- Already have `ConversationMemory`
- Add structured history log with kind/message/timestamp
- Support session flush and load for resumable conversations

### 5. **Execution Registry**
**Pattern**: Separate registries for command and tool execution.

```python
class ExecutionRegistry:
    def command(self, name: str) -> CommandExecutor:
        return self._commands.get(name)
    
    def tool(self, name: str) -> ToolExecutor:
        return self._tools.get(name)
    
    def execute(self, kind: str, name: str, input: str) -> ExecutionResult:
        if kind == 'command':
            return self.command(name).execute(input)
        else:
            return self.tool(name).execute(input)
```

**Application to Jarvis**:
- Already have `tool_registry`
- Add command registry for meta-commands (not tools)
- Support both with unified execution interface

### 6. **Bootstrap/Runtime Graph**
**Pattern**: Structured system initialization with stages.

```python
def build_bootstrap_graph() -> BootstrapGraph:
    return BootstrapGraph(
        stages=[
            'system_init',
            'context_init',
            'tool_pool_init',
            'command_graph_init',
            'session_ready',
        ]
    )
```

**Application to Jarvis**:
- Already have subsystem initialization in `run.py`
- Make it more structured with explicit stages
- Support stage-by-stage health checks

### 7. **Remote Runtime Modes**
**Pattern**: Multiple execution modes (remote, SSH, teleport, deep-link).

```python
def run_remote_mode(self, target: str) -> RuntimeResult:
    # Execute on remote server
    
def run_ssh_mode(self, target: str) -> RuntimeResult:
    # Execute via SSH

def run_teleport_mode(self, target: str) -> RuntimeResult:
    # Execute via teleport
```

**Application to Jarvis**:
- Add remote execution capabilities
- Support execution on other machines
- Could integrate with existing `remote_tools`

### 8. **Stream Events**
**Pattern**: Event-based communication during execution.

```python
def stream_submit_message(self, prompt: str, ...) -> Iterator[dict]:
    yield {'type': 'turn_start', 'turn': 1}
    yield {'type': 'command_execute', 'command': 'analyze'}
    yield {'type': 'tool_execute', 'tool': 'read_file'}
    yield {'type': 'turn_complete', 'result': '...'}
```

**Application to Jarvis**:
- Already have streaming in `run_step_stream()`
- Add structured event types for better observability
- Support event subscribers (for UI, logging, metrics)

### 9. **Context Management**
**Pattern**: Rich context object passed through execution.

```python
@dataclass
class PortContext:
    python_version: str
    implementation: str
    platform_name: str
    python_file_count: int
    archive_available: bool
    workspace_root: Path
    permission_context: ToolPermissionContext
```

**Application to Jarvis**:
- Enhance `ContextPack` with more metadata
- Include platform info, file counts, permission context
- Pass through all tool/command executions

### 10. **Query Engine**
**Pattern**: Central query processing with matching and execution.

```python
class QueryEnginePort:
    def submit_message(self, prompt: str, commands, tools, denials) -> TurnResult:
        # Match commands/tools
        # Execute in order
        # Return structured result
    
    def stream_submit_message(self, ...) -> Iterator[dict]:
        # Stream execution events
```

**Application to Jarvis**:
- Already have `agent.run_step_stream()`
- Add formal `QueryEngine` abstraction
- Support structured results and streaming

## Architecture Comparison

### Claw Code Structure
```
src/
├── main.py              # CLI entrypoint
├── commands.py          # Command registry & execution
├── tools.py             # Tool registry & execution
├── runtime.py           # Turn loop & session management
├── context.py           # Context management
├── permissions.py       # Permission contexts
├── history.py           # History log
├── session_store.py     # Persistence
├── query_engine.py      # Query routing & execution
├── bootstrap_graph.py   # Initialization stages
└── execution_registry.py # Unified execution
```

### Jarvis Structure (Current)
```
core/
├── agent.py             # Main orchestrator
├── llm.py               # LLM client
├── memory.py            # Conversation memory
├── tool_registry.py     # Tool registry
├── intelligence_hub.py  # Cross-subsystem coordination
├── reasoning_engine.py  # Multi-step reasoning
├── lazy_loader.py       # Deferred imports
└── ... (many more)
```

### Jarvis Structure (Proposed)
```
core/
├── agent.py             # Main orchestrator (enhanced)
├── execution/
│   ├── turn_loop.py     # Turn-loop management
│   ├── execution_registry.py  # Unified command/tool execution
│   └── stream_events.py # Structured event streaming
├── routing/
│   ├── query_engine.py  # Enhanced query routing
│   ├── command_router.py # Command routing
│   └── tool_router.py   # Tool routing
├── context/
│   ├── port_context.py  # Rich context objects
│   └── permission_context.py # Permission management
├── session/
│   ├── history_log.py   # Structured history
│   ├── session_store.py # Persistence
│   └── transcript.py    # Transcript management
├── bootstrap/
│   ├── stages.py        # Initialization stages
│   └── health_checks.py # Stage health checks
└── ... (existing modules)
```

## Immediate Integrations for Jarvis

### 1. Add Turn-Loop Management
```python
# core/execution/turn_loop.py
@dataclass
class TurnResult:
    turn: int
    output: str
    stop_reason: str  # 'completed', 'permission_required', 'error', 'max_turns'
    matched_commands: list[str]
    matched_tools: list[str]
    permission_denials: list[PermissionDenial]

class TurnLoopManager:
    def run_turn_loop(self, prompt: str, max_turns: int = 5) -> list[TurnResult]:
        # Implement turn-loop with stop reasons
```

### 2. Add Permission Denials
```python
# core/context/permission_context.py
@dataclass
class PermissionDenial:
    tool_name: str
    reason: str
    severity: str  # 'soft', 'hard', 'critical'
    
class PermissionContext:
    def check_permission(self, tool_name: str) -> PermissionDenial | None:
        if tool_name in DESTRUCTIVE_TOOLS:
            return PermissionDenial(tool_name, 'destructive operation', 'hard')
        return None
```

### 3. Add Structured History Log
```python
# core/session/history_log.py
class HistoryLog:
    def add(self, kind: str, message: str, metadata: dict = None):
        self.entries.append(HistoryEntry(
            kind=kind,
            message=message, 
            timestamp=time.time(),
            metadata=metadata or {}
        ))
```

### 4. Add Stream Events
```python
# core/execution/stream_events.py
class StreamEvent(Enum):
    TURN_START = "turn_start"
    COMMAND_EXECUTE = "command_execute"
    TOOL_EXECUTE = "tool_execute"
    PERMISSION_DENIAL = "permission_denial"
    TURN_COMPLETE = "turn_complete"

def emit_event(event_type: StreamEvent, data: dict) -> dict:
    return {
        'type': event_type.value,
        'timestamp': time.time(),
        'data': data
    }
```

### 5. Add Execution Registry
```python
# core/execution/execution_registry.py
class ExecutionRegistry:
    def __init__(self):
        self._commands: dict[str, CommandExecutor] = {}
        self._tools: dict[str, ToolExecutor] = {}
    
    def execute(self, kind: str, name: str, input: str) -> ExecutionResult:
        if kind == 'command':
            return self._commands[name].execute(input)
        else:
            return self._tools[name].execute(input)
```

## Benefits for Jarvis

1. **Better Error Handling**: Turn-loop with stop reasons allows graceful degradation
2. **Permission System**: Clear denial reasons help users understand restrictions
3. **Session Persistence**: Resume conversations across restarts
4. **Event Streaming**: Better observability for debugging and UI updates
5. **Structured Context**: Pass rich context through all executions
6. **Modular Architecture**: Clean separation of concerns
7. **Testable Components**: Each module has clear responsibilities

## Next Steps

1. Create `core/execution/` directory with turn_loop.py
2. Create `core/routing/` directory with enhanced routing
3. Create `core/session/` directory with history_log.py
4. Add PermissionDenial data structure to existing permissions.py
5. Integrate turn-loop into agent.run_step_stream()
6. Add stream events to existing WebSocket interface
7. Write tests for new components
8. Update AGENTS.md with new architecture

## Key Takeaway

Claw Code demonstrates clean separation between:
- **Routing** (which command/tool to use)
- **Execution** (how to run it)
- **Context** (what state to pass)
- **Session** (how to persist/restore)
- **Permissions** (what to gate)

This modular approach would make Jarvis more maintainable and extensible.