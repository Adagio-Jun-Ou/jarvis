# Jarvis Development Guidelines

## Commands

### Run Tests
```bash
./venv/bin/python -m pytest tests/ -v
./venv/bin/python -m pytest tests/ -q --tb=short  # Quick
./venv/bin/python -m pytest tests/test_new_modules.py -v  # Phase 1 modules
./venv/bin/python -m pytest tests/test_phase1_modules.py -v  # Phase 1 detailed
./venv/bin/python -m pytest tests/test_security_guardian.py -v  # Security & Guardian
./venv/bin/python -m pytest tests/test_summarizer.py -v  # Conversation Summarizer
./venv/bin/python -m pytest tests/test_prompt_sandbox.py -v  # Prompt Sandbox
./venv/bin/python -m pytest tests/test_hybrid_renderer.py -v  # Hybrid Music Renderer
./venv/bin/python -m pytest tests/test_music21_theory.py -v  # Music21 Theory Engine
./venv/bin/python -m pytest tests/test_security_audit_v2.py -v  # Security Audit v2
./venv/bin/python -m pytest tests/test_agent_architecture_hardened.py -v  # Agent Hardening
./venv/bin/python -m pytest tests/test_runtime_execution.py -v  # Runtime & Execution
./venv/bin/python -m pytest tests/test_bio_systems.py -v  # Bio-physiological systems
JARVIS_LOCAL_EMBEDDER=1 ./venv/bin/python -m pytest tests/test_embedding_system.py -v  # With real embeddings
```

### Type Check
```bash
./venv/bin/python -m py_compile core/*.py
./venv/bin/python -m py_compile tools/*.py
```

### Lint
```bash
./venv/bin/ruff check core/ tools/ 2>/dev/null || ./venv/bin/flake8 core/ tools/ 2>/dev/null || echo "Install ruff or flake8"
```

### Start Jarvis
```bash
./venv/bin/python run.py --interface web
./venv/bin/python run.py --interface cli
```

### Deploy (GLM-5 Cloud API)
```bash
export OLLAMA_API_KEY="your-glm-5-api-key"
./venv/bin/python run.py --interface web --port 8080
curl http://localhost:8080/health
```

## Architecture

### Core Modules (42 files)
- `agent.py` - Main JarvisAgent orchestrating LLM, memory, tools, bio-integration
- `llm.py` - OllamaClient with failover support, token optimizer integration
- `memory.py` - ConversationMemory with vector memory support
- `erdos_engine.py` - Math problem exploration (daydream)
- `insight_bridge.py` - Math insights → evolution guidance, knowledge graph
- `evolution_orchestrator.py` - Self-improvement with permissions
- `daydream_engine.py` - Idle-time intensive exploration
- `daydream_verification.py` - Verify daydream solutions
- `daydream_sketch.py` - Quick insight capture and testing
- `discovery_notifier.py` - Share discoveries naturally
- `self_edit.py` - Code modification with permission
- `resource_monitor.py` - Laptop resource optimization

### New Phase 1 Modules (6 files)
- `vector_memory.py` - HNSW vector search (150x faster retrieval)
- `knowledge_graph.py` - PageRank + community detection for insights
- `claims.py` - Work ownership with Byzantine consensus
- `swarm.py` - Multi-agent coordination with task distribution
- `token_optimizer.py` - 30-50% token reduction via caching + patterns
- `mcp_server.py` - Model Context Protocol server with stdio transport.
  Run as an MCP server for external clients: `python run.py --mcp`
  (read-only tools by default; set `JARVIS_MCP_ALLOW_WRITES=1` to
  register write-capable tools like `jarvis_memory_store`).

### Security & Context Systems (7 files)
- `lcm_shield.py` - Pre/post-request validation against injection and hallucination
- `lcm_planner.py` - Task decomposition with dependency DAG and execution waves
- `three_layer_context.py` - L1 Hot/L2 Warm/L3 Cold memory with self-editing blocks
- `reflect_edit.py` - Post-interaction self-improvement loop + council deliberation
- `semantic_contracts.py` - Typed I/O contracts between models
- `audio_bridge.py` - EnCodec audio tokenizer with semantic projector
- `context_guardian.py` - Proactive context overflow detection and auto-eviction
- `conversation_summarizer.py` - Compression before eviction (inspired by Letta)
- `prompt_sandbox.py` - Cross-model prompt isolation and injection filtering

### Bio-Physiological Architecture (4 files)
- `bio_homeostasis.py` - Autonomic regulation: NT setpoints, HPA stress response, vagal tone, energy budget, allostatic drift, stochastic fluctuation
- `circadian_rhythm.py` - 24h clock: performance curves, hormone cycling, ultradian BRAC, sleep pressure
- `neuroplasticity.py` - Synaptic adaptation: LTP/LTD, STDP, pruning, myelination, critical periods (50K synapses)
- `bio_integration.py` - Cross-module integration: unified PhysiologicalState, response style, circadian→homeostasis modulation
- Cross-wired with: `neuroscience.py` (NT sync), `emotional_state.py` (arousal/valence bridge), `agent.py` (prompt injection, stress/reward hooks)

### Predictive Processing & Active Inference (2 files)
- `predictive_processing.py` - Inner generative model: 3-level hierarchical prediction (abstract/context/sensory), prediction error computation, precision weighting, free energy minimization, belief updates, CurvatureDrive (curiosity/mastery/coherence/novelty)
- `cognitive_state_machine.py` - Autonomous state machine: AWAKE→CONTEMPLATING→DREAMING→EVOLVING transitions driven by idle time, circadian phase, stress, prediction error, intrinsic drives

### Consciousness Architecture (4 files)
- `interoception.py` - Predictive interoception engine: body state prediction, interoceptive error, feeling generation (valence/arousal/dominance), feeling quality classification (10 types), awareness tracking, attention modulation (narrow/broad)
- `global_workspace.py` - Baars' Global Workspace Theory: coalition competition (5 types), ignition threshold, broadcasting, cross-modality boost, recency boost, arousal-modulated threshold, workspace state tracking, decay, reverberation, contextual amplification
- `recursive_self_model.py` - Hofstadter's strange loop / HOT: L0-L3 self-model hierarchy, metacognitive judgments, confidence calibration, uncertainty awareness, meta-narrative, observation-driven mode transitions (automatic→monitored→reflective→meta_aware), cognitive load assessment, conflict detection, introspection
- `stream_consciousness.py` - James' stream of consciousness: continuous inner narrative, thought chaining (9 types, 6 relations), temporal windows, associative generation, workspace/feeling injection, coherence tracking, rhythm modulation, thread forking, sub-stream merging, topic drift detection, dream-mode surreal chaining

### Emotion Regulation Architecture (1 file)
- `emotion_regulation.py` - Barrett/Gross emotion regulation engine: 8 strategies (situation selection, modification, attentional deployment, cognitive reappraisal, response modulation, acceptance, suppression, amplification, body budget restoration), body budget tracking (energy/glucose/oxygen), strategy effectiveness tracking, reappraisal narrative generation, allostatic load monitoring

### Hybrid Music Renderer
- `music_hybrid_renderer.py` - Neural + programmatic synthesis pipeline
  - FluidSynth (GM SoundFont) → professional instrument timbres on CPU
  - pretty_midi → MIDI manipulation and rendering
  - librosa → spectral analysis, key detection, tempo estimation
  - music21 → music theory integration
  - Composer-specific instrument mapping (Bach→harpsichord, Chopin→piano, etc.)
  - Falls back to sine synthesis if FluidSynth unavailable
  - Integrated into `music_creation_engine.py` as primary audio path

### Music21 Theory Engine
- `music21_theory_engine.py` - Proper music theory backed by music21
  - Key validation, scale generation, chord construction
  - Thematic transformations (inversion, retrograde, augmentation, diminution)
  - Voice leading check (leap detection, parallel fifths)
  - Cadence generation (perfect, plagal, deceptive, half)
  - Fallback to built-in theory when music21 not available

### Runtime Execution Architecture (Phase 2)
- `execution_context.py` - Persistent CWD, env vars, process tracking, file ops audit, heartbeat
- `adaptive_agent_loop.py` - Self-healing agent runtime with retry/fallback/streaming
- `system_awareness.py` - Machine perception (CPU, memory, disk, processes, services, Docker)
- `watchdog.py` - Background health monitor (resource alerts, zombie cleanup, event log)

### Security & Audit Systems (Session 2 Fixes)
- **CRITICAL FIX**: `security/stack_defense.py` `_add_violation()` recursion → `.append()`
- **CRITICAL FIX**: `agent.py` `_get_tool_names()` recursion → `list(self.tools.keys())`
- **CRITICAL FIX**: `agent.py` `RequestQueue` future abandoned on outer exception
- **CRITICAL FIX**: Terminal execute hardened blocklist + timeout bounds (1-300s)
- **HIGH FIX**: Agent `_evaluate_task_completion` — "incomplete" no longer matches "complete"
- **HIGH FIX**: Short commands with dangerous keywords no longer bypass safety layers
- **HIGH FIX**: Session store bounded to 1000 entries with LRU eviction
- **MEDIUM FIX**: Embedding model now loads (fixed `trust_remote_code` + `custom_st.py` double-kwarg)
- **MEDIUM FIX**: `_active_workflows` bounded to 20 with eviction
- **MEDIUM FIX**: Unified context cache uses OrderedDict (O(1) LRU vs O(n))
- **MEDIUM FIX**: Prompt sandbox failure now logs warning instead of silent pass

### Key Flows
1. **Math → Evolution**: Erdős → Insight Bridge → Evolution Orchestrator
2. **Daydream → Verification**: Daydream Engine → Verification → Discovery
3. **Self-Edit**: Propose → Approve → Apply → Rollback
4. **Insight → Knowledge**: Insight Bridge → Knowledge Graph → PageRank
5. **Multi-Agent**: Swarm Coordinator → Claims → Task Distribution
6. **Agentic Loop**: Execute tool → Evaluate completion → Continue if incomplete (max 5 iterations)
7. **Bio-Physiological**: Homeostasis NT setpoints → Circadian modulation → Neuroplasticity adaptation → Prompt context injection
8. **Neuroscience Bridge**: Neuroscience NTs ↔ Homeostasis NTs (40/60 blend), Arousal/valence → EmotionalStateManager
9. **Stress/Reward**: User urgency words → Bio stressor → Homeostasis disruption → Regulatory actions → Response style adaptation; Task complete → Dopamine reward → Neuroplasticity LTP strengthening
10. **Predictive Processing**: Input → 3-level generative model prediction → Error computation → Belief update → Drive modulation (curiosity/mastery/coherence/novelty) → Attention weighting
11. **Cognitive State Machine**: Idle time → AWAKE→CONTEMPLATING→DREAMING→EVOLVING transitions, gated by circadian phase, stress, prediction error, intrinsic drives
12. **Consciousness Pipeline**: Bio NTs → Interoception (feeling generation) → Global Workspace (competition/ignition) → Stream of Consciousness (thought chaining) → Recursive Self-Model (metacognition) → System prompt injection
13. **Emotion Regulation Pipeline**: Feeling state → Strategy selection (8 strategies) → Body budget regulation → Allostasis prediction → Anticipatory signals → Bio feedback

### Agentic Workflow
- After tool execution, Jarvis evaluates if the task is complete
- Uses backend model (GLM-5) to assess: "COMPLETE" vs "CONTINUE: <next action>"
- Configured via `config.agent.max_agentic_iterations` (default: 5)
- Implemented in `run_step_stream()` for both single and parallel tool execution
- **Task list** (`update_tasks` tool, Claude Code TodoWrite semantics):
  the agent maintains an explicit decomposition (full-state replace,
  exactly one in_progress) that is injected into the system prompt and
  consulted by the completion evaluator — while tasks remain unfinished
  the loop continues (even on the backend-fail-open path). The
  iteration budget extends with unfinished tasks: `min(max(base,
  remaining + 2), 20)`.
- **Background jobs** (`terminal_execute(background=true)` + `check_jobs`
  tool, Claude Code run_in_background semantics): long builds/tests run
  as tracked async tasks with bounded output retention; the system
  prompt shows a "## Background Jobs" section so results are picked up
  on a later turn without polling. Runs through the same safety
  pipeline and permission gate as foreground commands.

### Long-horizon Harness Research (2026-09)
Research basis: Anthropic "Building effective agents" (workflows vs
agents; agents = LLM + tools + environmental feedback in a loop, with
stopping conditions) and the Claude Code best-practices harness
(TodoWrite task tracking, run_in_background + output check,
explore→plan→implement, verification loops, auto-compaction, subagents,
adversarial review). Built and verified (see NEXT_STEPS for commits):
- **Subagent `task` tool** (`tools/subagent_task.py`): isolated agent
  loop, restricted tools (explore = read-only; general = all minus
  `task`), returns only the final summary.
- **Checkpoint resume** (`core/adaptive_agent_loop.py`): checkpoints
  persist to disk; an interrupted turn (non-terminal phase) emits a
  `resume` event next turn and injects continuation context.
  Path override for tests: `JARVIS_ADAPTIVE_CHECKPOINT_DIR`.
- **Model compaction**: `compact_context` tool +
  `ConversationMemory.compact()`; `AgentLoop.execute` auto-summarizes
  between steps.
- **Shared harness singleton**: `get_agentic_harness()` — /api/harness
  routes and the agent share plans/checkpoints.
- **Background wake-up**: job completion posts a `background_job`
  attention item (next-turn prompt pickup) + `job.background.completed`
  EventBus event.
Remaining gaps tracked in NEXT_STEPS: lcm_planner execution, proactive
WebSocket wake-up, harness plan disk persistence.

### Tool Usage (SKILLS.md)
- SKILLS.md provides detailed patterns for tool chaining
- Jarvis can use multiple tools in sequence autonomously
- When in shell context, all other tools remain available
- Use `read_file` not `shell cat` for file reading
- Tools are loaded into system prompt via PromptManager

### Composable Workflows (SKILLS-WORKFLOWS.md)
- **Workflow Shell** - Composable pipelines with `|`, `||`, `>>` operators
- **Workflow Scheduler** - Cron-like scheduling for recurring tasks
- **Agentic Orchestrator** - Long-running autonomous deployments with self-recovery
- **n8n Integration** - Visual workflow automation (auto-started on port 5678)

#### Workflow Systems
1. `core/workflow_shell.py` - Lobster-style composable pipelines
2. `core/workflow_scheduler.py` - Scheduled/recurring tasks
3. `core/agentic_orchestrator.py` - Goal-driven autonomous operations
4. `tools/n8n.py` + `core/n8n_manager.py` - Visual workflow automation

#### n8n Tool Actions
- `list` - List all workflows
- `get` - Get workflow details
- `create` - Create new workflow
- `execute` - Execute a workflow
- `activate/deactivate` - Enable/disable workflows
- `design` - Generate workflow from description

### Memory Bounds
All collections have max sizes enforced. Check MAX_* constants.

### Thread Safety
All singleton getters use double-checked locking with threading.Lock() / threading.RLock().

### Background Bio Tasks
- `bio_tick` (60s interval, LOW priority): Advances homeostasis + circadian + neuroplasticity tick
- `neuroplasticity_consolidation` (3600s interval, LOW priority): LTP/LTD strengthening, pruning, myelination
- `cognitive_state_tick` (30s interval, LOW priority): Evaluates state transitions (AWAKE→CONTEMPLATING→DREAMING→EVOLVING) based on idle time, circadian, stress, prediction error
- `interoception_tick` (60s interval, LOW priority): Predictive interoception body state prediction + feeling generation + attention modulation
- `global_workspace_compete` (45s interval, LOW priority): Coalition competition, ignition, broadcasting
- `stream_consciousness` (30s interval, LOW priority): Generate next stream thought (awareness→workspace→stream chain)
- `self_model_reflection` (120s interval, LOW priority): Recursive self-model observation + mode transition
- `emotion_regulation_tick` (90s interval, LOW priority): Feeling-based strategy selection + body budget regulation
- `allostasis_prediction` (60s interval, LOW priority): Predict upcoming body budget needs + anticipatory regulation

## Tests
- `tests/test_comprehensive.py` - 59 tests
- `tests/test_integration_flow.py` - 16 tests (Erdős→Bridge→Evolution)
- `tests/test_self_awareness.py` - 37 tests
- `tests/test_runtime_integrity.py` - 16 tests
- `tests/test_new_modules.py` - 43 tests (vector_memory, knowledge_graph, claims, swarm, token_optimizer, mcp_server)
- `tests/test_phase1_modules.py` - 67 tests (LCM Shield, Planner, Three-Layer, Reflect-Edit, Contracts, Audio Bridge)
- `tests/test_security_guardian.py` - 22 tests (Context Guardian, memory security, LLM security, read-only blocks)
- `tests/test_summarizer.py` - 11 tests (Conversation Summarizer)
- `tests/test_prompt_sandbox.py` - 12 tests (Prompt Sandbox)
- `tests/test_health_check.py` - 10 tests (Health Checker)
- `tests/test_hybrid_renderer.py` - 20 tests (Hybrid Renderer, FluidSynth, composition)
- `tests/test_music21_theory.py` - 24 tests (Music21 Theory Engine, scales, chords, voice leading)
- `tests/test_security_audit_v2.py` - 16 tests (StackDefense, SessionManager, Agent tool safety, evaluation)
- `tests/test_bio_systems.py` - 65 tests (Homeostasis, Circadian, Neuroplasticity, Integration, Neuroscience bridge, Emotional bridge, Background tasks)
- `tests/test_fundamental_architecture.py` - 37 tests (PredictiveProcessing, CurvatureDrive, CognitiveStateMachine, AllostaticDrift, Integration)
- `tests/test_consciousness_modules.py` - 47 tests (Interoception, Modulator, GlobalWorkspace, RecursiveSelfModel, StreamConsciousness, ModuleIntegration)
- `tests/test_emotion_regulation.py` - 46 tests (EmotionRegulation, AllostasisPredictor, GlobalWorkspaceDeepening, StreamDeepening, SelfModelDeepening, RegulationLoopIntegration)

### Embedding Interconnect Thread Safety
- Use `interconnect.get_model_outputs(model)` instead of `interconnect._model_outputs[model]`
- Use `interconnect.get_last_responses()` instead of `interconnect._last_responses`
- Use `interconnect.get_stats()` for counts instead of `len(interconnect._model_outputs[model])`
- `_model_outputs` stores 3-tuples `(prompt, quality, embedding)`, not 2-tuples

## Safe File Editing
Only files in `self_edit.SAFE_FILES` can be modified by Jarvis:
- `core/erdos_engine.py`
- `core/insight_bridge.py`
- `core/shifting_memory.py`
- `core/evolution_orchestrator.py`
- `core/daydream_engine.py`
- `core/discovery_notifier.py`
- `core/daydream_verification.py`
- `tools/evolution_tools.py`

## Singleton Getters
All new modules follow the thread-safe singleton pattern:
```python
_example: Example | None = None
_lock = threading.Lock()

def get_example() -> Example:
    global _example
    if _example is None:
        with _lock:
            if _example is None:
                _example = Example()
    return _example
```

## Integration Points
- `insight_bridge.py` → `knowledge_graph.add_insight()` - Auto-indexes insights
- `memory.py` → `get_vector_memory()` - Optional semantic search
- `llm.py` → `get_token_optimizer()` - Token compression ready
- `run.py` → Background subsystems (Erdős, Daydream, Evolution, Defense)
- `music_creation_engine.py` → `music_hybrid_renderer.py` - Primary audio path (FluidSynth + SoundFont)
- `music_perception.py` → librosa when available - Enhanced key/tempo detection

## Known Security Issues (Unfixed, Require Design Decisions)
- **RCE via terminal/execute**: The endpoint allows arbitrary shell commands. Blocklist is imperfect. Requires either auth enforcement or sandboxing.
- **WebAuthn signature verification skipped**: `mobile_auth.py` `verify_authentication` never verifies the cryptographic signature. Requires proper WebAuthn implementation.
- **CSRF bypass on missing Origin header**: The CSRF middleware skips validation when Origin is absent.
- **Auth disabled by default**: `require_auth=False` is the default. Set `JARVIS_API_KEY` env var to enable.
- **Information disclosure in error responses**: Multiple endpoints return `str(e)` in error messages.
- **Path traversal in terminal CWD**: `/api/terminal/cwd` allows setting CWD to any directory.

## Test Status
- **570+ tests passing** across 16+ test files  
- Bio-physiological: `test_bio_systems.py` (65 tests: homeostasis, circadian, neuroplasticity, integration, bridges)
- Phase 2: `test_runtime_execution.py` (115 tests), `test_agent_architecture_hardened.py` (88 tests)
- Pre-existing failure: `test_new_modules.py::test_thread_safety` hangs (infinite join)
- Pre-existing failure: embedding tests need `JARVIS_LOCAL_EMBEDDER=1` for real model (otherwise zero vectors)
- Pre-existing failure: Some embedding resilience tests fail without real model

### Bio-Physiological API Endpoints
- `GET /api/bio/state` - Full physiological state (arousal, valence, NTs, stress, circadian)
- `GET /api/bio/style` - Adaptive response style (detail_level, tone, energy)
- `POST /api/bio/stressor` - Apply stressor (body: {intensity, source})
- `POST /api/bio/reward` - Apply reward (body: {magnitude, neurotransmitter})
- `GET /api/bio/circadian` - Circadian rhythm state (phase, capacity, sleep pressure)
- `GET /api/bio/neuroplasticity` - Neuroplasticity statistics (synapses, LTP/LTD, pruning)
- `POST /api/bio/consolidate` - Trigger memory consolidation (body: {duration_factor})
- `GET /api/bio/homeostasis` - Full homeostasis state (NTs, autonomic, energy, regulatory actions)

### Predictive Processing & Cognitive State API Endpoints
- `GET /api/predictive/state` - Prediction error, free energy, drives, curiosity topics
- `POST /api/predictive/process` - Process input and compute prediction errors
- `GET /api/predictive/curiosity` - Top curiosity-driven topics for exploration
- `GET /api/cognitive/state` - Cognitive state (awake/contemplating/dreaming/evolving)
- `POST /api/cognitive/force` - Force cognitive state transition (body: {state})

### Emotion Regulation & Allostasis API Endpoints
- `GET /api/emotion_regulation/state` - Regulation engine state, budget, strategy effectiveness, history
- `GET /api/allostasis/state` - Allostatic load, stress level, anticipatory signals

### Consciousness Architecture API Endpoints
- `GET /api/interoception/state` - Interoception engine state, feeling history, attention modulator
- `GET /api/workspace/state` - Global workspace state, current broadcast, ignition history
- `GET /api/self-model/state` - Recursive self-model state, meta-narrative, mode
- `GET /api/stream/state` - Stream of consciousness state, recent thoughts, coherence

### Phase 2 Key Changes (This Session)
- **AdaptiveAgentLoop streaming**: New `run_persistent_stream()` method yields events (thinking, tool_call, tool_result, recovery, text, complete)
- **Chat streaming recovery**: When `run_step_stream` throws, AdaptiveAgentLoop's `recover_from_error()` attempts a retry with strategy-specific context reduction
- **Consecutive error tracking**: Web handler tracks consecutive agent errors, disables retry after 2 consecutive failures, resets on success
- **Terminal `needs_permission()`**: Low-risk commands (ls, pwd, git) auto-approved; medium/high still require permission
- **Dynamic `needs_permission()` in agent.py**: `execute_tool()` now calls `tool.needs_permission(**args)` if the method exists
- **CWD sync**: Terminal cd, terminal_cwd tool, and shell commands sync through ExecutionContext — persistent across turns
- **CWD-aware file resolution**: Native file tools resolve relative paths against ExecutionContext CWD when available
- **Tool input validation**: `ToolRegistryValidator.validate_tool_input()` called before tool execution, with 50KB max arg size
- **System prompt injection**: Execution context (CWD, recent commands, system resources) injected into system prompt
- **Malformed args protection**: JSON parse errors in tool args now return error instead of empty dict
- **Silent error logging**: All `except Exception: pass` replaced with `logger.debug()` for observability
- **Content-Length bypass**: Malformed Content-Length headers now return 400 instead of being silently ignored
- **Agentic loop break messaging**: Mid-stream break now yields user-visible message instead of silent stop
- **Broadcast error throttling**: Background broadcast task logs errors and uses 0.5s backoff instead of 0.1s
- **Compound CWD tracking**: `cd dir && cmd` and `cmd && cd dir` patterns tracked by ExecutionContext
- **ExecutionContext atexit**: Graceful shutdown kills orphan processes and persists state on process exit
- **ExecutionContext shutdown**: New `shutdown()` method terminates running processes and saves state
- **Runtime dashboard**: New "Runtime" tab in Cognitive-AI Dashboard with execution context, awareness, commands, files, adaptive loop, watchdog, heartbeat, and runtime controls (cd, execute)
- **File operation audit**: Native file tools log read/write operations to ExecutionContext
- **Tool failure diagnostics**: When all tools fail in `run_step_stream`, execution context diagnostics are shown
- **WS connections endpoint**: New `/api/runtime/connections` for monitoring WebSocket connection health