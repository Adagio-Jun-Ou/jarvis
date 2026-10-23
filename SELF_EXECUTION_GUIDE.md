# Jarvis Self-Execution and Refinement Guide

## Overview

Jarvis is designed as a self-aware AI assistant capable of running autonomously, monitoring its own performance, and refining its behavior over time. This document explains how Jarvis executes, evaluates, and improves itself.

## 1. Execution Model

### Startup Sequence

```
run.py
  ├── Load Config (core/config.py)
  ├── Create JarvisAgent (core/agent.py)
  ├── Register Tools (tools/*.py)
  ├── Start Background Subsystems
  │   ├── Evolution Orchestrator (every 4 hours)
  │   ├── Erdos Engine (math exploration)
  │   ├── Daydream Engine (idle exploration)
  │   ├── Defense Agent (security monitoring)
  │   ├── Learning Pipeline
  │   ├── Memory Manager
  │   └── Free Time Engine
  └── Start Interface (cli/voice/web/terminal)
```

### Key Execution Flows

#### 1. Main Agent Loop
```python
# core/agent.py - JarvisAgent
async def chat(self, message: str) -> str:
    # 1. Check memory for context
    context = self.memory.get_relevant(message)
    
    # 2. Build prompt with tools
    prompt = self._build_prompt(message, context)
    
    # 3. Call LLM (Ollama or Cloud)
    response = await self.llm.generate(prompt)
    
    # 4. Execute any tool calls
    if has_tool_calls(response):
        results = await self._execute_tools(response)
        response = await self._continue_with_results(results)
    
    # 5. Store in memory
    self.memory.add(message, response)
    
    # 6. Trigger learning/reflection
    self._maybe_reflect(message, response)
    
    return response
```

#### 2. Background Subsystem Orchestration
```python
# core/system_integrator.py
class SystemIntegrator:
    # Dependency graph for subsystem startup order
    # - evolution_orchestrator: root (no deps)
    # - erdos_engine: depends on evolution_orchestrator
    # - defense: root (no deps)
    # - learning_pipeline: depends on erdos_engine, defense
    # - daydream_engine: depends on erdos_engine
```

## 2. Self-Evaluation Mechanisms

### 2.1 Observability System
Location: `core/observability.py`

```python
# Health checks registered for each subsystem
observability.register_health_check('erdos_engine', lambda: (erdos._running, {'problems': len(erdos.problems)}))
observability.register_health_check('learning_pipeline', lambda: (pipeline._running, {'sessions': len(pipeline.sessions)}))
observability.register_health_check('defense', lambda: (defense._running, {'detected': defense._stats.get('threats_detected', 0)}))
```

### 2.2 Memory Pressure Monitoring
Location: `core/memory_cleanup.py`

```python
# Automatic cleanup when memory pressure detected
memory_mgr.register_pressure_callback(memory_pressure_response)

# Pressure levels trigger different actions:
# - LOW: Standard operation
# - MEDIUM: Start cleanup of caches
# - HIGH: Aggressive cleanup, reduce parallelism
# - CRITICAL: Emergency cleanup, may stop non-essential subsystems
```

### 2.3 Performance Metrics
Tracked automatically:
- Tool execution times
- LLM response times
- Memory usage patterns
- Token consumption
- Error rates

## 3. Self-Refinement Pipeline

### 3.1 Evolution Orchestrator
Location: `core/evolution_orchestrator.py`

Runs every 4 hours (configurable):
1. Collects metrics from all subsystems
2. Identifies improvement opportunities
3. Proposes changes to safe files
4. Requires approval for modifications (if permissions enabled)

Safe files that can be auto-modified:
```python
SAFE_FILES = [
    'core/erdos_engine.py',
    'core/insight_bridge.py',
    'core/shifting_memory.py',
    'core/evolution_orchestrator.py',
    'core/daydream_engine.py',
    'core/discovery_notifier.py',
    'core/daydream_verification.py',
    'tools/evolution_tools.py',
]
```

### 3.2 Daydream Engine
Location: `core/daydream_engine.py`

Runs during idle time:
1. Picks unsolved or interesting problems
2. Explores multiple approaches in parallel
3. Tests hypotheses using sandboxed execution
4. Validates discoveries through verification
5. Publishes successful insights to knowledge graph

### 3.3 Erdős Engine
Location: `core/erdos_engine.py`

Mathematical problem exploration:
1. Maintains queue of interesting problems
2. Explores patterns and connections
3. Generates solutions and proof attempts
4. Feeds insights to evolution system

### 3.4 Insight Bridge
Location: `core/insight_bridge.py`

Connects discoveries to actionable improvements:
```
ErdosEngine ──┐
              ├──▶ InsightBridge ──▶ KnowledgeGraph
Daydream ─────┘          │
                         ▼
                EvolutionOrchestrator
```

## 4. Self-Modification

### 4.1 Self-Edit System
Location: `core/self_edit.py`

```python
class SelfEditor:
    # Can only modify files in SAFE_FILES
    # Creates backups before modification
    # Tracks all changes in history
    
    async def propose_edit(self, file_path: str, changes: dict) -> str:
        # 1. Validate file is safe
        # 2. Create backup
        # 3. Apply changes
        # 4. Validate syntax
        # 5. If failed, rollback
```

### 4.2 Auto-Refine
Location: `core/auto_refine.py`

```python
class AutoRefineManager:
    # Analyzes code for improvements
    # Suggests optimizations
    # Can auto-apply safe improvements
    # Tracks performance before/after
```

## 5. Learning Pipeline

### 5.1 Reinforcement Learning
Location: `core/reinforcement_learning.py`

- Tracks user feedback (explicit and implicit)
- Adjusts behavior based on outcomes
- Maintains reward tables for actions

### 5.2 Tool Learning
Location: `core/tool_learning.py`

- Discovers optimal tool combinations
- Learns from failed tool calls
- Adapts to user patterns

### 5.3 Persistent Memory
Location: `core/persistent_memory.py`

- Long-term storage of user preferences
- Cross-session knowledge retention
- Fact extraction and storage

## 6. Security Model

### 6.1 Permission System
Location: `core/permissions.py`

```python
class PermissionLevel(Enum):
    ALLOW = "allow"       # Pre-approved
    ASK = "ask"           # Single approval
    DENY = "deny"         # Always blocked
```

Tools declare their permission requirement:
```python
class DockerTool(Tool):
    requires_permission = True
```

### 6.2 Defense Agent
Location: `core/defense.py`

- Monitors for suspicious patterns
- Blocks malicious requests
- Learns from attack attempts
- Periodic security review

### 6.3 DM Security (New)
Location: `core/dm_security.py`

- Device pairing with approval codes
- Rate limiting
- Trust levels for devices
- Encrypted direct messaging

## 7. Hardware Optimization

### 7.1 Hardware Detection
Location: `core/hardware.py`

Auto-detects and uses available acceleration:
- **CPU**: Always available, parallel processing
- **NPU (XDNA)**: AI acceleration for supported operations
- **GPU**: Currently unsupported for gfx1150 (needs ROCm 7.3+)

### 7.2 NPU Integration
Location: `core/npu_accelerator.py`

```python
# ONNX Runtime with Vitis AI
# Accelerates:
# - Voice tokenization (275x realtime)
# - Embedding generation
# - Small model inference
```

## 8. Running Jarvis

### Basic Startup
```bash
cd /home/user/jarvis
./venv/bin/python run.py --interface cli
```

### Web Interface
```bash
./venv/bin/python run.py --interface web --port 8080
# Access at http://localhost:8080
```

### With Cloud LLM
```bash
export OLLAMA_API_KEY="your-glm-5-api-key"
./venv/bin/python run.py --interface web
```

### Debug Mode
```bash
./venv/bin/python run.py --interface cli --debug
```

## 9. Monitoring and Debugging

### Logs
```bash
# Main log
tail -f /home/user/jarvis/logs/jarvis.log

# Subsystem-specific
grep "ErdosEngine" /home/user/jarvis/logs/jarvis.log
grep "Daydream" /home/user/jarvis/logs/jarvis.log
```

### State Inspection
```python
# In Python:
from core.erdos_engine import get_erdos_engine
erdos = get_erdos_engine()
print(erdos.problems)  # Active problems

from core.memory_cleanup import get_memory_manager
mem = get_memory_manager()
print(mem.get_stats())  # Memory stats
```

### Health Check
```python
from core.observability import get_observability
obs = get_observability()
print(obs.get_health_report())  # All subsystem health
```

## 10. Configuration Files

### Main Config
Location: `~/.jarvis/config.yaml`

```yaml
identity:
  name: "Jarvis"
  
llm:
  model: "glm-4-9b-chat"
  ollama_host: "http://localhost:11434"
  
thinking:
  mode: "auto"  # auto, think, no-think
  
automation:
  docker:
    enabled: true
  n8n:
    enabled: false
```

### Storage Paths
```
~/.jarvis/
├── config.yaml           # Main config
├── memory/               # Conversation memory
├── knowledge/            # Knowledge graph
├── skills_registry/      # Skills database
├── canvas/               # Canvas workspaces
├── dm_security/          # Device security
├── workflows/            # Saved workflows
├── sandbox/              # Sandbox workspace
└── logs/                 # Runtime logs
```

## 11. Extending Jarvis

### Adding a New Tool
```python
# tools/my_tool.py
from tools.base import Tool

class MyTool(Tool):
    name = "my_tool"
    description = "Does something useful"
    parameters = {
        "type": "object",
        "properties": {
            "input": {"type": "string"}
        },
        "required": ["input"]
    }
    requires_permission = False
    
    async def execute(self, input: str) -> str:
        return f"Processed: {input}"
```

### Register in run.py
```python
from tools.my_tool import MyTool
agent.register_tools([MyTool()])
```

### Adding a Background Subsystem
```python
# core/my_subsystem.py
class MySubsystem:
    def __init__(self):
        self._running = False
    
    async def start(self):
        self._running = True
        while self._running:
            await self._do_work()
            await asyncio.sleep(60)

# In run.py:
from core.my_subsystem import get_my_subsystem
subsystem = get_my_subsystem()
await subsystem.start()
```

## 12. Troubleshooting

### LLM Not Responding
```bash
# Check Ollama
curl http://localhost:11434/api/tags

# Or check cloud API key
echo $OLLAMA_API_KEY
```

### High Memory Usage
```bash
# Check memory manager
grep "Memory pressure" logs/jarvis.log

# Reduce parallelism in config
# Or add cleanup handlers
```

### Subsystem Not Starting
```bash
# Check health
grep "ERROR" logs/jarvis.log | grep -E "(Erdos|Daydream|Defense)"

# Check dependencies in system_integrator
```

## Summary

Jarvis operates as a continuous self-improving system:
1. **Execution**: Multi-interface agent with background subsystems
2. **Evaluation**: Real-time monitoring, metrics, and health checks
3. **Refinement**: Evolution orchestrator, daydream engine, and auto-refine
4. **Learning**: Reinforcement learning, tool learning, persistent memory
5. **Security**: Permission system, defense agent, DM pairing

The key insight is that Jarvis doesn't just respond to queries—it continuously explores, learns, and improves its own code and behavior within carefully defined safety boundaries.