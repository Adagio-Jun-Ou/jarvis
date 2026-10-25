# Jarvis Architecture & Status Report

**Updated:** 2026-05-24
**Status:** All Systems Operational
**Test Results:** 5576 passed, 46 skipped, 0 failed

---

## System Overview

Jarvis is a comprehensive cognitive AI platform integrating dual-model LLM routing, consciousness simulation, bio-physiological modeling, and autonomous tool execution.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            JARVIS CORE                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐  ┌─────────────────┐  ┌──────────────────────────┐    │
│  │   AGENT     │  │    LLM CLIENT   │  │       MEMORY             │    │
│  │             │  │                 │  │                          │    │
│  │ • JarvisAgent│ │ • OllamaClient │  │ • ConversationMemory     │    │
│  │ • 4,856 lines│ │ • Cloud API    │  │ • VectorMemory (HNSW)    │    │
│  │ • Dual-model │  │ • Failover     │  │ • UnifiedMemory          │    │
│  │ • Execution  │  │ • Token Optim. │  │ • MemoryHub              │    │
│  └─────────────┘  └─────────────────┘  │ • AdaptiveContextWindow  │    │
│                                          └──────────────────────────┘    │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    BIO-PHYSIOLOGICAL                             │   │
│  │                                                                  │   │
│  │  Homeostasis → Circadian → Neuroplasticity → BioIntegration     │   │
│  │       ↓              ↓              ↓              ↓            │   │
│  │   NT setpoints    24h cycle     synaptic     unified context    │   │
│  │   HPA stress      BRAC          LTP/LTD      prompt injection   │   │
│  │   vagal tone      chronotype    50K synapses  response style    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    CONSCIOUSNESS ARCHITECTURE                    │   │
│  │                                                                  │   │
│  │  Interoception → GlobalWorkspace → StreamConsciousness → RSM    │   │
│  │       ↓                ↓                  ↓                ↓     │   │
│  │  body state      coalition bid      thought chain     metacog...│   │
│  │  feeling gen     ignition           temporal window   confidence│   │
│  │  attention       broadcasting       topic drift       self-model│   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌────────────────────┐  ┌─────────────────────────────────────────┐   │
│  │   PREDICTIVE       │  │            SECURITY                      │   │
│  │                    │  │                                          │   │
│  │ • 3-level gen.mod. │  │ • Auth + Session + API Key              │   │
│  │ • Free energy min  │  │ • CSRF + XSS + CSP                      │   │
│  │ • CuriosityDrive   │  │ • DLP + MITM + Branch Detect            │   │
│  │ • CogStateMachine  │  │ • StackDefense + Upload Validation      │   │
│  └────────────────────┘  │ • Sandbox + SSRF Protection             │   │
│                            └─────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                            TOOLS LAYER                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Core: System, Web, Browser, NativeFiles, Project, Oracle, Docker       │
│  Data: DataTools (pandas, charts, CSV/JSON)                            │
│  Specialized: PhysicsSim, Spatial, ArtPhilosophy, Music                 │
│  Cognitive: MetaTools, EvolutionTools, LearningTools, NPU Tools        │
│  Multimodal: VisionTool, Transcription, Speech, Token Tools             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Module Inventory (281 Core Modules)

### Core Infrastructure (22 modules)
auth, auth_middleware, mobile_auth, config, agent, llm, memory, permissions,
background, session, events, resilience, performance, resource_monitor,
observability, dashboard, api_schema, deployment, state_sync, module_bridge,
context_manager, connection_pool

### Hardware (9 modules)
hardware, accelerator, gpu_compute, gpu_simulation, npu_accelerator,
npu_integration, npu_mlir, npu_models, npu_optimizations

### Memory (18 modules)
vector_memory, semantic_memory, persistent_memory, unified_memory,
shifting_memory, memory_manager, memory_cleanup, memory_hub, memory_router,
memory_consolidation, memory_protection, memory_watchdog, adaptive_context,
conversation_summarizer, context_compressor, three_layer_context,
context_enrichment, context_cascade

### Cognitive (14 modules)
thinking, prompts, awareness, self_awareness, knowledge_graph,
context_coordinator, proactive, self_reflection, emotional_state,
sentiment_analyzer, user_model, goal_tracker, thought_sense,
cognitive_psychology

### Bio-Physiological (6 modules)
bio_homeostasis, circadian_rhythm, neuroplasticity, bio_integration,
neuroscience, cognitive_state_machine

### Consciousness (7 modules)
interoception, global_workspace, recursive_self_model,
stream_consciousness, emotion_regulation, emotional_memory,
affective_context

### Predictive Processing (2 modules)
predictive_processing, cognitive_state_machine, concept_drift

### Learning (6 modules)
learning_pipeline, tool_learning, reinforcement_learning,
vulnerability_learning, orchestrated_learning, cognitive

### Security (12 modules)
security/dlp, security/branch_detection, security/mitm_protection,
security/stack_defense, security/audit, security/upload_validation,
security/session_security, security/ssrf_protection, defense, audit,
prompt_sandbox, safe_files

### Tool Modules (23 modules)
system, web, browser_tools, native_files, project, docker, data_tools,
n8n, physics_sim, spatial, art_philosophy, meta_tools, evolution_tools,
learning_tools, security_tools, npu_tools, token_tools, transcribe,
screen_tools, refine_tools, oracle, terminal, base

### Interfaces (4 modules)
cli, terminal, voice, web_react

### Speech (4 modules)
codec, conversation, npu_whisper, context_voice

### Additional modules (25+)
agent_loop, adaptive_agent_loop, execution_context, execution_router,
workflow_shell, workflow_scheduler, agentic_orchestrator,
intelligent_orchestrator, sub_agent, subagent, swarm, claims, mcp_server,
dual_model, model_coordinator, oracle_integration, oracle_bridge,
music_creation_engine, music_hybrid_renderer, music_perception,
music_evolution, music_theory, music_intent, music_refinement,
concept_model

---

## Hardware Configuration

```
Detected Hardware:
Component             Type           Memory    Status
────────────────────────────────────────────────
CPU                   multi-core        unified      ACTIVE
AMD Radeon iGPU (gfx1150) GPU            shared    FALLBACK (Vulkan via Ollama)
XDNA NPU              4 AI cores     512MB     ACTIVE
```

Primary Accelerator: NPU (XDNA)
GPU Status: gfx1150 supported via Vulkan backend (Ollama)
Memory Pool: unified memory total, shared available

---

## Test Results Summary

**Total: 5576 passed, 46 skipped, 0 failed across 176 test files**

### Test Coverage by Area

| Area | Test File | Tests |
|------|----------|-------|
| Bio-Physiological | test_bio_systems.py | 108 |
| Consciousness | test_consciousness_modules.py | 47 |
| Emotion Regulation | test_emotion_regulation.py | 46 |
| Fundamental Architecture | test_fundamental_architecture.py | 37 |
| Phase 1 Modules | test_phase1_modules.py | 67 |
| Security | test_security_audit_v2.py | 16 |
| Security Hardening | test_security_hardening.py | 30+ |
| Runtime/Execution | test_runtime_execution.py | 162 |
| Runtime Integrity | test_runtime_integrity.py | 38 |
| Agent Architecture | test_agent_architecture_hardened.py | 88 |
| Memory/Context | test_memory_context_integration.py | 80+ |
| Embedding System | test_embedding_*.py | 40+ (46 skipped) |

### Test Count by Category  
- Comprehensive/New Modules: 200+
- Integration/E2E: 100+
- Security: 100+
- Bio/Consciousness: 200+
- Runtime/Execution: 200+
- Tools: 100+
- Embedding: 100+

---

## Performance Metrics

| Metric | Value |
|--------|-------|
| Voice Tokenization | 275x realtime (NPU) |
| Vector Search (HNSW) | 150x faster than linear |
| Token Optimization | 30-50% reduction |
| Active Context Limit | 96,000 tokens |
| System Prompt Cache TTL | 300s (5 min) |
| Tool Cache TTL | 600s (10 min) |

---

## Known Issues

1. **GPU (gfx1150)**: AMD Radeon iGPU supported via Vulkan (Ollama) but not via PyTorch/ROCm
2. **Embedding model**: torch/sentence-transformers not installed in Python 3.14 venv (disk quota). 46 embedding tests skip via graceful degradation.
3. **Auth disabled by default**: `require_auth=False` is intentional for personal/local use. Set `JARVIS_API_KEY` env var to enable.
4. **RCE via terminal/execute**: Shell commands are inherently powerful. Mitigated by permission system, timeout bounds, and blocklist.

---

## Quick Start

```bash
cd /home/user/jarvis
source venv/bin/activate

# Run all tests
./venv/bin/python -m pytest tests/ -v

# Start Jarvis
./venv/bin/python run.py --interface cli
./venv/bin/python run.py --interface web --port 8080
```

---

## Recent Changes (v3.0)

### This Session
- Fixed 17 test failures (missing deps, zero-vector embedding, empty-track audio)
- Security: 28 `str(e)` → `safe_error_message()` replacements, CORS NameError fix
- Thread safety: Lock→RLock + guards in bio_homeostasis, circadian_rhythm
- Bugfix: emotion_regulation `abs()` bug, double allostatic recovery, tuple `.get()` crash, vector rebuild data loss
- Refactor: Extracted shared `_preprocess_turn()` from agent (193 lines saved)
- Rate limiter: Degrades gracefully instead of failing closed
- Dead code: Removed 48 lines unreachable tone-shaping filters

### Previous Session (HEAD~2)
- Music system overhaul: MuseScore soundfont, tone shaping, chord parsing fixes
- Context system hardening: AdaptiveThresholds, PressurePredictor, MultiBudgetGuardian
- Knowledge Graph enrichment, Oracle circuit breaker, per-endpoint rate limits
- AgentLoop parallel tool execution + context guardian priority eviction
