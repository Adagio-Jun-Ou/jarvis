# Jarvis Architecture Review
## Cognitive Information Flow, API Integration & Agentic Deployment

Generated: 2026-02-24

---

## Executive Summary

This review covers:
1. **Cognitive Information Flow** - How data flows through Jarvis
2. **API Call Integration** - External/internal API architecture
3. **Agentic Deployment** - Autonomous operations and multi-agent coordination

### Key Findings

| Area | Status | Issues Found |
|------|--------|--------------|
| Core Flow | ✅ Working | Minor gaps in agent_loop integration |
| API Layer | ⚠️ Needs Work | Circuit breakers not wired, no auth middleware |
| Agentic Ops | ✅ Working | AgentLoop not yet connected to WebSocket |

---

## 1. Cognitive Information Flow

### 1.1 Input Processing Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  WebSocket  │────▶│  JarvisAgent│────▶│    Memory   │
│  REST API   │     │.run_step_   │     │   System    │
│             │     │  stream()   │     │             │
└─────────────┘     └─────────────┘     └─────────────┘
       │                   │                   │
       │                   ▼                   ▼
       │           ┌─────────────┐     ┌─────────────┐
       │           │   Think()   │     │   Vector    │
       │           │   Method    │────▶│   Memory    │
       │           └─────────────┘     │   (HNSW)    │
       │                   │           └─────────────┘
       │                   ▼
       │           ┌─────────────┐
       │           │    LLM      │
       │           │  Qwen-3.5   │
       │           │  GLM-5      │
       │           └─────────────┘
       │                   │
       ▼                   ▼
┌─────────────┐     ┌─────────────┐
│   Token     │◀────│   Tool      │
│   Stream    │     │  Execution  │
└─────────────┘     └─────────────┘
```

**Key Code Paths:**

| Step | File:Line | Method |
|------|-----------|--------|
| WebSocket receive | `web_react.py:2474-2536` | `run_step_stream()` |
| Input processing | `agent.py:694-720` | `run_step_stream()` |
| System prompt | `agent.py:379-424` | `get_system_prompt()` |
| LLM call | `agent.py:430-469` | `think()` |
| Tool execution | `agent.py:471-545` | `execute_tool()` |
| Memory update | `agent.py:855-872` | Persistent memory |

### 1.2 LLM Integration - Dual Model Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                    MODEL ROUTING                                │
├───────────────────────────┬────────────────────────────────────┤
│      PRIMARY (Tier 1)     │        BACKEND (Tier 2)            │
│      qwen3.5:397b         │           glm-5                     │
├───────────────────────────┼────────────────────────────────────┤
│ ✓ Interactive chat        │ ✗ No interactive chat              │
│ ✓ Vision/images           │ ✗ No vision                        │
│ ✓ Screen automation       │ ✗ No screen automation             │
│ ✓ All tool categories     │ ✓ Limited: web, data, meta, refine │
│ ✓ Browser control         │ ✗ No shell/terminal                │
│ ✓ Full autonomy           │ ✓ Code review, analysis, daydream  │
└───────────────────────────┴────────────────────────────────────┘
```

**Task Routing (backend_llm.py:167-171):**
```python
backend_tasks = {"cli", "refine", "code_review", "analysis", 
                 "daydream", "self_edit", "tools"}
# Routes to GLM-5 for backend tasks, Qwen-3.5 for everything else
```

### 1.3 Memory Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     MEMORY LAYERS                                │
├──────────────┬──────────────────────────────────────────────────┤
│   Layer      │  Description                        │ Budget     │
├──────────────┼─────────────────────────────────────┼────────────┤
│   WORKING    │ Recent conversation history         │    40%     │
│   SESSION    │ Important marked messages           │    30%     │
│   EPISODIC   │ Vector memory retrieval (HNSW)      │    20%     │
│   SEMANTIC   │ Knowledge graph context             │    10%     │
└──────────────┴─────────────────────────────────────┴────────────┘
```

**Bounds Enforced:**
- MAX_IMPORTANT_MESSAGES = 100
- MAX_TOPIC_IMPORTANCE = 200
- MAX_LONG_TERM_MEMORY = 100
- DEFAULT_MAX_TOKENS = 28,000

### 1.4 Streaming Flow (NEW - chat_stream_with_tools)

**File: `llm.py:404-520`**

The new `chat_stream_with_tools` method yields structured events:

```python
# Thinking content from GLM-5
{"type": "thinking", "text": "Let me analyze..."}

# Response content
{"type": "text", "text": "The answer is..."}

# Tool call request
{"type": "tool_call", "name": "search", "call_id": "...", "arguments": {...}}

# Completion signal
{"type": "finish", " reason": "stop|tool-calls|error"}
```

This enables OpenCode-style continuous agent loops with:
- Real-time thinking display
- Automatic tool execution
- Multi-turn conversation handling

---

## 2. API Integration

### 2.1 External API Calls

| Endpoint | Purpose | Auth |
|----------|---------|------|
| `/api/chat` (Ollama) | LLM inference | Bearer token |
| Cloud: `https://ollama.com/api/chat` | GLM-5 fallback | API key |

**API Key Handling:**
- Environment: `OLLAMA_API_KEY`
- Config: `config.yaml` → `llm.api_key`
- Auto-enables cloud mode when key present

### 2.2 Internal REST Endpoints (77 total)

**Key Endpoints:**

| Category | Endpoint | Purpose |
|----------|----------|---------|
| Health | `GET /health` | Basic health check |
| Health | `GET /api/health` | Subsystem health |
| Status | `GET /api/status` | Full system status |
| Models | `GET /api/models` | Primary/backend model info |
| Chat | `POST /api/chat` | SSE streaming chat |
| Chat | `WS /ws` | WebSocket real-time chat |
| Conversations | `GET/POST /api/conversations/*` | CRUD operations |
| Tools | `GET /api/tools` | List available tools |
| Goals | `GET/POST /api/goals` | Autonomous goals |
| Swarm | `GET /api/swarm/status` | Multi-agent status |
| Circuit Breaker | `GET /api/circuit-breakers` | Resilience status |
| Rate Limiter | `GET /api/rate-limiter/stats` | Rate limit stats |

### 2.3 Issues Found

#### HIGH: Circuit Breakers Not Integrated

**Location:** `run.py:503-510`, `llm.py`

Circuit breakers are created but NOT wrapping LLM calls:
```python
# run.py - Created but unused
primary_breaker = get_circuit_breaker("llm_primary", ...)
backend_breaker = get_circuit_breaker("llm_backend", ...)

# llm.py - No circuit breaker usage
async def _make_request_with_failover(...):
    # Direct HTTP call without circuit breaker protection
```

**Fix Required:**
```python
# Wrap LLM calls with circuit breaker
async with primary_breaker:
    response = await self.client.post(...)
```

#### MEDIUM: No Authentication Middleware

REST endpoints have no authentication. Anyone can access:
- `/api/conversations` - All conversation history
- `/api/tools` - Tool inventory
- `/api/models` - Model configuration

#### MEDIUM: Rate Limiter Not Applied

Rate limiter is configured but not applied as middleware to WebSocket or REST.

---

## 3. Agentic Deployment

### 3.1 Agent Loop (NEW - agent_loop.py)

**Status:** Implemented but NOT YET CONNECTED

The `AgentLoop` class provides:
- Continuous iteration until task completion
- Automatic tool execution
- Structured streaming events
- Permission gating

**Connection Points:**
```python
# Create agent loop
loop = AgentLoop(
    llm_client=backend_llm,      # Can use GLM-5
    tool_registry=agent.tools,    # From JarvisAgent
    permission_manager=agent.permissions,
)

# Execute with streaming
async for event in loop.execute_stream(messages, system_prompt):
    # event types: thinking, text, tool_call, tool_result, complete
```

**Missing Integration:**
- WebSocket handler uses `agent.run_step_stream()` directly
- Should optionally use `AgentLoop` for multi-step tasks

### 3.2 Autonomous Operations

```
┌─────────────────────────────────────────────────────────────────┐
│                   AUTONOMOUS SUBSYSTEMS                          │
├───────────────┬─────────────────────────────────────────────────┤
│   Subsystem   │   Purpose                    │   Trigger        │
├───────────────┼──────────────────────────────┼──────────────────┤
│ daydream      │ Idle-time learning           │ 15s idle         │
│ erdos         │ Math problem exploration     │ Background       │
│ evolution     │ Self-improvement proposals   │ Periodic         │
│ defense       │ Security monitoring          │ Continuous       │
│ learning      │ Continuous learning pipeline │ Background       │
│ goals         │ Long-term goal tracking      │ Periodic check   │
└───────────────┴──────────────────────────────┴──────────────────┘
```

**Daydream Conditions (daydream_engine.py:76-86):**
- Idle > 15 seconds
- Job queue empty
- No pending work
- CPU/memory available

### 3.3 Multi-Agent Coordination

**Swarm System (swarm.py):**
- MAX_AGENTS = 20
- Heartbeat timeout: 120s
- Task queue with priority
- Byzantine consensus (tolerates 1 faulty of 4 nodes)

**Claims System (claims.py):**
- Work ownership tracking
- Byzantine fault tolerance
- TTL-based claim expiration (default 30 min)
- MAX_CLAIMS = 1000

### 3.4 Startup Sequence

```
Phase 1: Core Setup
├── Load config.yaml
├── Apply CLI overrides
├── Create JarvisAgent
└── Register 15+ tool categories

Phase 2: Agent Startup
├── agent.start()
├── api_manager.start()
└── Initialize memory/LLM

Phase 3: Background Subsystems
├── evolution_orchestrator.start_evolution()
├── erdos_engine.start_background_learning()
├── defense_agent.start_monitoring()
├── learning_pipeline.start_pipeline()
├── daydream_engine.start()
├── state_synchronizer.start()
└── observability.start()

Phase 4: Health & Memory
├── health_manager.start()
├── memory_watchdog.start()
└── persistent_memory.start()

Phase 5: Interface
└── web_interface.run() / cli.run()
```

---

## 4. Recommendations

### 4.1 Critical Fixes

1. **Wire Circuit Breakers** (HIGH)
   - Wrap `_make_request_with_failover` in circuit breaker
   - Create separate breakers for primary/backend LLM

2. **Connect Agent Loop** (HIGH)
   - Add option to use `AgentLoop` for multi-step tasks
   - Wire to WebSocket for thinking display

3. **Add Authentication Middleware** (MEDIUM)
   - API key validation for REST endpoints
   - Session token for WebSocket

### 4.2 Enhancements

1. **Rate Limiter Middleware**
   - Apply to WebSocket message rate
   - Per-client rate limiting

2. **Thinking Content Display**
   - Update `web_react.py` to handle `type: "thinking"` events
   - Style thinking differently from response text

3. **Agent Loop Integration**
   - Create endpoint `/api/agent/execute` for multi-step tasks
   - Use backend LLM (GLM-5) for code analysis

### 4.3 Missing Integration

| Component | Status | Action |
|-----------|--------|--------|
| AgentLoop → WebSocket | ❌ | Create bridge for streaming |
| Circuit Breaker → LLM | ❌ | Wrap HTTP calls |
| Rate Limiter → WebSocket | ❌ | Add middleware |
| Thinking Display → UI | ❌ | Handle thinking events |

---

## 5. File Reference

| File | Purpose | Lines |
|------|---------|-------|
| `core/llm.py` | LLM client with streaming | 1-580 |
| `core/backend_llm.py` | Model routing | 1-331 |
| `core/agent.py` | Main agent orchestration | 1-1200+ |
| `core/agent_loop.py` | OpenCode-style loop | 1-593 |
| `core/memory.py` | Conversation memory | 1-400+ |
| `core/vector_memory.py` | HNSW semantic search | 1-1000+ |
| `core/swarm.py` | Multi-agent coordination | 1-800+ |
| `core/claims.py` | Work ownership | 1-520 |
| `core/daydream_engine.py` | Idle-time learning | 1-500+ |
| `interfaces/web_react.py` | Web API & WebSocket | 1-2600+ |
| `run.py` | Startup sequence | 1-600+ |

---

## 6. Next Steps

1. **Immediate**: Wire circuit breakers to LLM calls
2. **Short-term**: Connect AgentLoop to WebSocket for multi-step tasks
3. **Short-term**: Add thinking content handling in UI
4. **Medium-term**: Add authentication middleware
5. **Medium-term**: Apply rate limiting to WebSocket