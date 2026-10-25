# Jarvis v3.0 — Project Status & Roadmap

**Last Updated:** 2026-10-19
**Build Status:** 5572 passed, 49 skipped, 14 failed (all host-environment, see below)

## Cycle: engine merge + Jarvis tool adapter (2026-10-19)

The enterprise agent engine (agent/, alex/, app/, sdk/, pipelines/,
workflows/, governance/) now lives in this repo and runs locally via
`run_local.py`; Jarvis's original stack is preserved alongside it.
Claude's last commit adapted Jarvis's native tools into the engine's
governed registry as a deferred library. This cycle audited that seam
line by line and fixed what reading found:

- **Background jobs broke under the adapter**: `terminal_execute(
  background=true)` created an asyncio task on the caller's loop; the
  adapter runs each tool call through `asyncio.run` on a throwaway
  loop, whose shutdown cancelled the task (and could hang waiting for
  the subprocess — measured). Jobs now run on a private worker
  thread, so they outlive the call in both stacks.
- **Governance**: the dual-ACL was applied only at exposure time —
  adapted tools have no `guarded()` wrapper, so a direct API/SDK call
  bypassed it. The VM now checks the capability at *invocation* time.
  `shell` (declares `requires_permission=False` but runs arbitrary
  commands) and the project/approval tools were read/power 0; an
  `_ALWAYS_ACT` set promotes them to `.act`/power 2/policy-ASK.
- **Adapter families**: music and project silently failed to build
  (wrong symbols); system sense was missing (and Jarvis's own registry
  mislabelled `system_sense`/`system_watch` as SCREEN). Discovery is
  now 112 tools; meta tools stay out deliberately (they need Jarvis's
  LLM client).
- **Capability granularity**: 56 tools collapsed into `jarvis.misc.*`
  (one ACL entry covering unrelated tools); a family→category fallback
  splits them. Local file readers are now fenced as untrusted, matching
  the engine's own `view`.
- **Deferred library**: `find_tools` returned reduced signature-only
  schemas (no descriptions/enums) — it now returns the original JSON
  schema. Pinning was unbounded (every turn carried every pin) — now a
  bounded LRU. Successful use pins a tool for later turns ("used
  before" was promised but unimplemented). MCP discovery was being
  re-run on every turn's working-set rebuild — now TTL-cached.
- **Test isolation**: app-lifespan tests registered Jarvis tools into
  the global registry and never cleaned up, so later doc/VM-lifecycle
  tests failed in full-suite order (they passed alone). conftest now
  unregisters after each test.
- Docs: `docs/engine/10_JARVIS_TOOL_ADAPTER.md` (design, governance,
  knobs, limits); the generated TOOLS.md notes the extended library.

### Second pass — deep engine audit (same cycle)

Continued line-by-line review of the engine after the adapter fixes:

- **Governor false halts**: `jarvis.<category>.*` capabilities were
  judged as family `"jarvis"`, so the third adapted act tool in any run
  halted as scope drift even when it served the goal. Capabilities now
  resolve to their real category family.
- **Dead internal bypass**: `AGENT_INTERNAL_TOOLS` held a tool *name*
  (`log_agent_action`) while the gate receives a *capability*
  (`agent.log`) — the bypass never fired. Both spellings now checked.
- **Approvals**: a stopped run kept a worker parked on the question for
  the full 180s window; it now refuses immediately (fail closed).
- **`OPENROUTER_MODEL` was ignored**: `Settings` hardcoded the default
  endpoint, so the documented knob did nothing unless
  `ALEX_DEFAULT_LLM_ENDPOINT` was also set. Fixed and verified through
  `get_settings()`.
- **Fusion anchoring was dead in production**: the context engine calls
  `fuse()` without a Lakebase connection and the resulting exception was
  swallowed; `resolve_anchors` now opens a short-lived connection when
  the caller has none. Dev was unaffected (its graph ignores the conn).
- **Overlapping turns on one VM**: `run_turn` interleaved history,
  governor, and working set. Turns are now serialized per VM (RLock;
  different VMs still run in parallel).
- **Turn persistence race**: concurrent chats read the same base index
  and `append_turn`'s insert-or-drop silently lost a pair. The
  read-modify-write is serialized and derives the next index from the
  max existing one (gaps stay gaps).
- **Deploy gate was advisory**: `deploy()` claimed to refuse on a failed
  eval but never called it; production deploys now require the
  `eval_result` and refuse a non-passing gate. Both in-repo callers pass
  their result.
- **`find_tools` with no keywords** returned an error; it now returns
  the library inventory (categories, counts, examples), which is what a
  model probing for capabilities actually needs.
- **Doctor**: `scripts/engine/doctor.py` — offline check of backend/key
  presence, store writability, optional arms, adapted-library
  registration, dev ACL count, and constitution articles.

Known engine limits found but not changed: standing-order single-flight
is process-atomic (in-process lock + DB heartbeat check); two replicas
can interleave between the check and the INSERT in a narrow window. The
default deployment is one local process, so this is documented rather
than redesigned (a portable atomic claim needs a warehouse-level
constraint). `meta_tools` stays unadapted (needs Jarvis's LLM client).

## Cycle: Claude Code-style harness research + build (2026-09-12)

Research: Anthropic "Building effective agents" + Claude Code best
practices, plus a line-by-line audit of Jarvis's existing harness
(agent_loop, adaptive_agent_loop, agentic_harness, orchestrator,
lcm_planner, sub_agent). Anthropic's agent definition: an LLM using
tools in a loop with environmental feedback and stopping conditions;
the harness patterns that make long-horizon work reliable are explicit
task decomposition (TodoWrite), background execution with output
checks (run_in_background), verification loops, and context
compaction. Four stages landed:

### Stage A — adaptive-loop singleton fix (2a47b62)
Real production bug: `/api/adaptive-loop/status` created the singleton
with `llm=None`, and `_think()` returns None whenever `self.llm` is
falsy — so a first dashboard visit poisoned every subsequent
agent-loop chat for the life of the process. The getter now
late-binds missing dependencies (upgrade-only, never thrashes).

### Stage B — TodoWrite-style task list (2bf17d7)
- `core/task_list.py`: bounded, thread-safe store; full-state replace;
  exactly one in_progress (extras demoted); status validation;
  active work never evicted; prompt rendering with markers.
- `update_tasks` tool (tools/task_list.py), registered at boot.
- System prompt "## Task List" section, ordered right after the
  execution context.
- Completion-evaluator gate: while unfinished tasks exist the loop
  continues instead of accepting COMPLETE — including the backend
  fail-open path.

### Stage C — task-aware iteration budget (13f693e)
Both agentic loops (`run_step_stream` single- and parallel-tool paths)
compute the bound per iteration: `min(max(base, remaining + 2), 20)`,
so a decomposed 8-task run can actually finish while a hard cap keeps
runaway control. Base stays `config.agent.max_agentic_iterations`.

### Stage D — background jobs + standby visibility (0e0be0c)
- `JarvisComputer.start_background()` reuses the full terminal safety
  pipeline (blocked pipes/substitutions, session cwd/env, output caps)
  and returns a job id; jobs are tracked asyncio tasks.
- `terminal_execute(background=true)` + `check_jobs` tool; registry
  bounded (finished evicted first, running never).
- "## Background Jobs" prompt section gives standby pickup: finished
  results surface on the next turn without polling.

### Stage E — background completion wake-up (25f3145)
A finished job now posts an attention-queue item (`source=
background_job`, FOLLOW_UP on success / SYSTEM_ALERT on failure) —
the exact channel `_preprocess_turn` drains into the "## Background
Attention Items" prompt section — plus a `job.background.completed`
EventBus event for UI/proactive subscribers.

### Stage F — subagent Task tool (cfeb6c2)
`tools/subagent_task.py`: Claude Code Task semantics. `task(prompt,
type=explore|general, max_steps)` spawns the existing AgentLoop with a
fresh message list, restricted tools (explore = read-only preset;
general = all minus `task` to bound recursion), and returns ONLY the
final summary — intermediate exploration never enters the parent
context. Parent LLM (backend preferred) + permission manager are
inherited; AgentLoop ERROR finishes are surfaced as failures.

### Stage G — checkpoint restore/resume (dc18f59)
Adaptive-loop turn checkpoints now persist atomically to
`~/.jarvis/adaptive_loop/last_checkpoint.json` (override:
`JARVIS_ADAPTIVE_CHECKPOINT_DIR`). `restore_checkpoint()` /
`interrupted_checkpoint()` detect a turn whose phase never reached
terminal (process died mid-turn); `run_persistent_stream` emits a
`resume` event and injects a "## Resumed Session" continuation block
(request, tool calls made, completed results, cwd). Turn-end phase is
normalized so an exhausted iteration budget is not mistaken for a
crash. Suite isolation via conftest env var (tests used to pollute the
real `~/.jarvis`).

### Stage H — shared AgenticHarness (5f1c22e)
`/api/harness/*` endpoints each built a fresh `AgenticHarness`, so
plans created by /plan were invisible to /execute and checkpoints
never accumulated. Added `get_agentic_harness()` singleton
(late-binding, thread-safe); all 6 route sites plus
`JarvisAgent.get_agentic_harness()` now share one instance.

### Stage I — model-triggered compaction (738275a)
`ConversationMemory.compact(keep_recent)`: replaces older messages
with a ConversationSummarizer block (heuristic fallback), keeps recent
verbatim, recomputes tokens, stores the summary in long-term memory.
New `compact_context` tool exposes it to the model (Claude Code
/compact). The previously-dead `AgentLoop.summarize_if_needed` is now
called between steps so long tool-executing runs can't grow the
message list without bound.

### Remaining harness gaps (audited, not yet built)
- **lcm_planner execution**: planner produces DAGs but nothing executes
  them; `run_agentic` / `agentic_orchestrator` boot still dead.
- **Background completion push**: wake-up is prompt-section pickup at
  the next turn; no WebSocket push that starts a turn proactively.
- **Harness plan persistence**: singleton fixes coherence, but plans/
  checkpoints are still in-memory only (lost on restart).

## Cycle: ground-up refinement, second pass (2026-09-06)

Executing the refinements the first-pass audit catalogued (4cdb7e1,
2891a34):

### Backend token savings
- **Slim backend prompt**: the backend model (code/math/analysis
  routing) received the full ~30-section conversational prompt — it
  now gets identity + execution context + attention + style + goals
  only (~60% smaller per routed call). Wired at both route_task sites,
  cached with the main prompt's TTL pattern.

### Dead on-the-fly machinery removed
- **BBB / ATP metabolism / E-I balance** (236 lines + 156 test lines):
  ticked every cycle mutating only their own internal state. Verified
  zero consumers — even the dashboard reads circadian's capacity, not
  homeostasis's. The "ATP gates cognitive throughput" intent was
  dropped on the floor from the day it was built.
- **"Regulatory:" prompt line** (17 internal action-name strings): the
  LLM cannot act on "increase_arousal_stimulation". Still on the
  dashboard; no longer in the prompt.
- **16 transitively-dead modules** deleted (9,206 lines): symbolic
  composers v1-v3, dashboard, performance, master_practice, etc. —
  scanned for import-path AND bare-name references first.

### Intent bugs fixed (code that was built deliberately, then never worked)
- **Inert style adjustments in _bio_context**: four blocks (CSM
  processing depth, affective bias framing, load-balancer throttle,
  urgency discount) mutated the style dict AFTER the "Response style:"
  line had already been rendered — so every one of them was dead.
  Style now renders after all adjustments; depth/bias/throttle
  actually reach the prompt.

### Batch B (completed — 21,465 lines deleted in 94a6238)
Modules referenced by test files but not by production: auth, events,
subagent + sub_agent_comm, resilience, response_cache, skill_tracker,
subsystem_health, unified_sync, evaluator_optimizer, input_validation,
llm_coordinated, llm_queue, memory_hub_enhancements, natural_output,
speech_formatter, deployment, canvas, dm_security, docker_sandbox,
context_voice, api_schema, agentic_integration, fluid_integration +
fluid_information_flow, accelerator, npu family (npu_accelerator ←
npu_whisper, npu_integration, npu_mlir, npu_optimizations, onnx_voice),
react_engine, vulnerability_learning — all deleted (65 files,
21,465 lines), with 44 test classes and 11 test methods that were
their last consumers removed from 33 shared test files.

KEPT deliberately (documented capability, zero runtime cost, decide
later): music21_theory_engine (test-covered, advertised in AGENTS.md,
never wired into production — wiring it is a feature decision, not
cleanup). Also kept: the NPU_AVAILABLE optional group with its False
fallback (tools/system_sense.py reads is_npu_available).

MCP completed (c937493): mcp_server was a tested protocol core with no
transport. Added the stdio transport (`python run.py --mcp`), protocol
purity (stdout reserved for JSON-RPC, sys.stdout redirected to stderr
before logging setup), notification/invalid-request handling per spec,
and a write gate (write-capable tools only with
JARVIS_MCP_ALLOW_WRITES=1). Tests: test_mcp_stdio.py (8) + existing
MCP tests; end-to-end smoke via `python -m core.mcp_server` verified.

core/__init__.py export map cleaned: accelerator always-exports
removed, _SHADOWED_EXPORTS emptied (all providers deleted), 8 optional
groups for deleted modules removed, 56 __all__ entries filtered.
Verified: no missing names in __all__, JarvisAgent resolves.

Verification: 5,572 tests collect with zero errors after all deletions
(full-suite collect-only check) — the strongest possible proof that
nothing production-relevant was removed.

Second-pass extras (0e68edb, 289019d, 7c74f0d):
- test_live_integration.py: standalone live harness was auto-collected
  by pytest and errored 10x at setup (its `test` fixture never
  existed). Layer functions renamed to check_* — the harness still
  runs standalone as designed, pytest ignores it.
- unified_memory._save() self-heals its storage directory (a removed
  pytest tmp dir caused an ENOENT flake in the reload regression test).
- npu_mlir deleted (12KB, last zero-ref module of the batch).

## Cycle: ground-up behavioral review (2026-09-06)

Five parallel swarm audits (bio consumption trace, prompt assembly,
tick economics, consciousness effect, dead-module inventory), every
critical claim verified by reading before fixing. The standard: a
feature matters only if it changes LLM text, gates a decision, or
acts — everything else is theater. Landed in 529cc7f:

### Bugs found and fixed (all regression-tested)
- **CSM never woke from DREAMING**: the (DREAMING, AWAKE) transition
  edge was missing — user input was silently rejected while chatting.
- **Short messages never reset idle**: notify_user_input sat inside
  `if not is_simple` — the machine drifted to DREAMING mid-conversation.
  Now every turn notifies user presence at the _preprocess_turn
  chokepoint.
- **Physiology ran at several times real speed**: 5 tick sites + every
  prompt-build read advanced bio time; bio_tick double-advanced
  circadian. tick() is wall-clock anchored; reads use a read-only
  _peek_state() cache. Verified: 60s gap -> exactly 1.0 min.

### Honest verdicts on the bio layer
LOAD-BEARING (keep): homeostasis arousal/valence/stress (prompt text +
CSM gate + ignition threshold), circadian capacity/fatigue/operational
phase (prompt + gates), interoception chain (Feeling line), the
metacognition gate, stream inner-speech read.

COSMETIC (computed, never consumed — candidates for future removal):
BloodBrainBarrier + ATP + E/I balance (~230 lines whose
get_cognitive_capacity() output is silently replaced by circadian's),
circadian zeitgeber/chronotype/light-exposure (zero callers), the 17
regulatory-action strings (no executor exists — they only become
prompt adjectives), most of the 50K-synapse machinery (serves one
get_connected topic walk), hormones/energy/vagal fields in bio_state
(read by nobody).

### Tick economics
CUT: health_check (60 HTTP probes/hr), load_balancer, autonomous_goals,
file_change_monitor, proactive_intelligence, butler_personality
(re-observed the same interaction — sentiment double-count).
DEMOTED: the whole consciousness chain to 300s (safe: wall-clock
anchoring + on-demand prompt reads). ~70% of idle-hour executions cut.

### Prompt trim (30-45% of chars, zero behavioral loss)
Deleted stat dumps (ignition counters, Phi/WM floats, uptime); gated
lines behind interestingness (self-model floats unless mode changed,
stream stats only when degraded, budget only when unhealthy,
CPU/RAM only when constrained). Inner-speech read widened 2-of-6 ->
3-of-18: the LLM now actually sees what the stream generates.

### Known-dead inventory (mechanical removal pass queued)
~55 transitively-dead core modules (npu family, composers v1-v3,
auth vs auth_middleware, events vs event_bus, subagent trio,
mcp_server, music21_theory_engine — advertised in AGENTS.md but never
imported in production). 276 modules total; ~35 form the provably
behavior-changing core. Also queued: backend model pays full system
prompt on dual-routed calls — a slim backend variant would cut ~40%
of its token spend.

## Cycle: consciousness revival, research-grade inner life (2026-09-04)

Audited every consciousness module against its primary literature and found
most advertised features were dead code — never called, or computed but
never consumed. This cycle revived them and grounded the implementations.

### Dead paths revived
- 8 consciousness paths had zero production callers (dream thoughts,
  temporal-window consolidation, forking, sub-stream merging, recency,
  contextual amplification, self-model L2/L3 observation, workspace
  reverberation injection). Each is now driven by a background tick or
  agent-loop hook with a bio/cognitive-state trigger.
- **The GWT broadcast → stream pipeline was silently disabled in
  production**: `_last_ignited_type` vs `_last_ignition_type` crashed
  every `compete()` call and was swallowed by the tick's debug-level
  handler. One word, entire pipeline dark.

### Research-grounded upgrades (commits 07acf07, 0421b25, c3d12e1)
- **Stream of consciousness** — dialogic inner speech per Fernyhough's
  dialogism (VISQ-tagged counter-voices with speaker attribution),
  condensation-by-load (Vygotsky: under body-budget load, inner speech
  drops to predicative fragments), DES sampling (Heavey & Hurlburt: inner
  speech is a *minority* of conscious moments — now probed every 30s and
  compared against the empirical 5-category distribution).
- **Global workspace ignition** — replaced the one-shot soft cutoff with
  Del Cul, Baillet & Dehaene 2007's nonlinear bifurcation: clearly
  supra-threshold input (≥ threshold×1.3) ignites in a single pass;
  near-threshold input accumulates leaky evidence across ticks
  (probabilistic sigmoid zone); same-type repeats inside a 4s refractory
  window are attenuated ×0.6 (repetition suppression, Grill-Spector 2006)
  instead of threshold-elevated. Evidence map bounded.
- **Metacognition** — type-2 SDT per Fleming & Lau 2014: AUROC2
  discrimination (rank-based, bias-free) + Brier-reliability calibration,
  index = 0.6·disc + 0.4·calib, reliable only at N≥40 (full meta-d' ML
  fitting needs hundreds of trials — the rolling 50-window cannot support
  it). Injected into the system prompt with the actual inner monologue.

### Also this cycle
- Physical world model for objects/spatial reasoning (e9f26fc); oracle
  route split (6977d66); agentic-loop completion-judgment fix + safety
  gate (211f059); session security (5 capabilities); model-id single
  source of truth; frontend TS fixes + bundle rebuild.

### Next research queue
- Fuller meta-d′ once judgment history exceeds ~200 samples (proper ML
  fit becomes viable; current AUROC2 proxy is the right tool at N=50).
- Social-context triggers for second-person self-talk: the stream now
  *generates* "you can do this" under load/failure (15d13a7), but the
  Dolcos & Albarracin social-evaluative trigger (more self-talk in
  social contexts) is not yet wired to user presence.

### Also this cycle (2)
- Second-person self-talk generation: `generate_self_instruction()`
  in the stream, emerging organically under condensation load, after
  tool failures, at high arousal, and more frequently for 120s after
  user input (social-evaluative trigger, rate 0.39 vs 0.24 verified)
  — the templates existed but nothing produced the speech (15d13a7,
  a18d955).
- Calibration-gated verification: tool outcomes feed judge_confidence
  (estimate = recent per-tool success rate), and when the type-2 index
  is reliably negative, COMPLETE verdicts on failure-laden contexts are
  downgraded to verification steps; the system prompt names the
  behavior so the model knows to double-check (15d13a7).
- Broken-contract fixes from a swarm audit (990b5dd): surprise-gated
  dreaming (LOW_SURPRISE) and mastery-gated evolution (HIGH_MASTERY →
  EVOLVING) were both unreachable — prediction errors were never
  surfaced in get_state(), and record_insight had zero callers; the
  condensation budget term was a constant 0.5 (wrong key); pp.tick
  (belief decay + idle curiosity) had no caller; allostatic
  apply_stressor never spiked load; DES distribution was write-only.

### Next research queue
- Fuller meta-d′ once judgment history exceeds ~200 samples (proper ML
  fit becomes viable; current AUROC2 proxy is the right tool at N=50).
- Prediction.weighted_error computed property has no consumers — fold
  into free-energy aggregation or delete.
- InteroceptionEngine.predict_body_state is a redundant getter; a real
  N-step-ahead body prediction would justify its existence.

## Cycle: mind-diffusion cross-pollination (2026-09-05)

Integration of concepts from C:\projects\mind-diffusion — a diffusion-
model latent "mind" simulator (VAE 64x64 brain images -> 8x8x4 latent,
conditional DDPM/DDIM, trajectory modes, coupled mind-world knot loop).

### What landed (f8d4aa8)
- **Knot loop bicameral dial** -> stream counter-voice: set_bicameral()
  blends the "other" voice between opposing (balanced, b~0.3) and
  ECHOING the self-voice (Jaynes' bicameral mind, b->1). Driven by
  cognitive state: dreaming 0.8, contemplating 0.5, awake 0.3.
- **Whole-loop tightening**: at b>0.6 the self-voice itself repeats
  with p=(b-0.3) — mind-diffusion's finding that b=0.9 locks the loop
  into self-echo (pixel change 4x lower than balanced b=0.3).
- **Self-echo streak** = knot tightening; environment-driven thoughts
  (workspace/feeling) reset it — the knot is the mind hearing itself.
- **Walled-garden pull-back**: streak >= 4 jumps topic via CONNECTOME
  spread activation — one step through neuroplasticity's LTP graph
  (the real connectome), i.e. the manifold pull keeping the loop
  in-distribution.

### Queued from mind-diffusion (not yet built)
- **MindJEPA over terminal sessions**: LANDED (049f255) —
  `core/jepa_world_model.py`, faithful port of mind_jepa_bridge's
  latent-only world model (EMA target + GRU context + action-conditioned
  predictor + VICReg-lite), trained online every 90s on Jarvis's own
  shell history; one-step-ahead cosine error feeds PredictiveProcessor
  as a context-level error. Learning verified: structured vs shuffled
  data → median error 0.41-0.61 vs 0.82-0.91. Two measurement-caught
  bugs fixed and regression-tested (train/eval GRU context mismatch;
  learning rate undertrained at tick cadence). Remaining refinement:
  feed real session data long enough to see error shape on actual
  Jarvis usage; consider exposing per-channel world errors (jepa vs
  physics schemas) distinctly in get_state.
- **Dream GIF rendering**: render dreaming-state stream trajectories
  through mind-diffusion's trained VAE checkpoints (vae.pt/diffusion.pt)
  as dashboard visuals — "what Jarvis's dream looks like". Optional
  tool, cross-project dependency.
- **DDIM-slerp thought transitions**: interpolate between embedding
  centroids instead of template-pick topic jumps (needs the vector
  memory embeddings to expose an interpolation path).

## Cycle: review, repair, integrate (2026-07-25)

Windows became a first-class host this cycle (see `SETUP_WINDOWS.md`), which
surfaced a class of latent bugs that a Linux-only CI could never reach.

### Crashes fixed
- `safe_error_message` used but never imported in `defense.py`, `health.py`,
  `tool_learning.py` — the exception handlers themselves raised `NameError`,
  turning handled errors into unhandled 500s. Fallout from an earlier bulk
  `str(e)` → `safe_error_message()` sweep that missed the imports.
- `JSONResponse`/`security_headers` were locals of `_setup_middleware()` but used
  inside `_setup_routes()` — 6 sites on the auth-reject, rate-limit and
  payload-too-large paths of `/api/terminal/*` and `/api/chat`.
- `core/music_intent.py`, `core/music_refinement.py`: `threading.Lock()` at module
  scope with no `import threading` — both modules were **unimportable**.
- `core/sandbox.py`: module-level `import resource` (POSIX-only) made the whole
  module unimportable on Windows, blocking 41 tests. Now degrades to timeout +
  output caps; `os.killpg`/`getpgid` guarded.
- `npu_whisper.encode_for_npu()` referenced an undeclared `sample_rate`.
- `AgenticPipeline.add_stage(depends_on=…)` referenced an undefined `context`,
  had the dependency edge **inverted**, and never registered graph nodes — so a
  declared dependency could never become ready. Rebuilt; also fixed a stage-index
  desync on duplicate stage ids.

### Silent corruption fixed
- `context_compressor.get_messages_for_llm()` returned conversation history **out
  of order**: it ranked by importance then restored chronology using `created_at`
  (which collides at the wall clock's ~15.6ms granularity) keyed by message
  *content* (which collides on repeated text). Now ranks by insertion position.
- `execution_router` measured durations with `time.time()`, reporting `0.0ms` for
  any sub-15ms dispatch. Ten sites moved to `perf_counter()`.
- `event_bus` async delivery used bare `loop.create_task()`; asyncio holds only a
  weak reference, so **events could be garbage-collected before delivery**. Fixed
  there and at 4 other sites via the existing `create_tracked_task` helper.
- 140 production file-I/O sites had no explicit encoding — on Windows that means
  cp1252, so any non-ASCII payload corrupts or raises on persist/load, including
  `memory.py`, `knowledge_graph.py`, `persistent_memory.py`, `audit.py`.
- **`peft` was an undeclared dependency of jina-embeddings-v5.** Without it the
  model fails to load and `encode()` returns **zero vectors**, which makes every
  pair perfectly similar — semantic search, vector memory and knowledge-graph
  similarity all return plausible-looking garbage. Declared, and the fallback now
  logs once at ERROR instead of a per-call warning.

### Security
- **Sandbox escape (latent).** `native_files.resolve_path()` hardcoded
  `is_in_workspace = True` on the relative-path branch, disabling both the
  traversal guard and the sandbox guard below it — `../escape.txt` escaped the
  workspace on *every* platform. Existing tests only covered absolute paths.
  Fixed, with parametrized regression tests.
- **Six independent protected-path blocklists** (`native_files`, `terminal`,
  `docker.py`, `docker_sandbox`, `execution_context`, `web`) all used POSIX
  string-prefix matching: inert on Windows, and on Linux they falsely denied
  siblings like `/etcetera` and `/devops`. Added one shared *mechanism* in
  `core/helpers.py` (`is_protected_path`, `path_is_within`,
  `platform_protected_roots`); each site keeps its own policy list.
- `/run/docker.sock` was missing from the docker mount blocklist — on modern
  Linux `/var/run` symlinks to `/run`, so the resolved socket path fell outside
  the list. That is a direct container escape.
- `core/security/ssrf_protection.py` was written and tested but **not exported
  and never called**; `tools/web.py` had its own weaker check. The shared
  protector is now layered after the local checks (it can add a denial, never
  relax one), adding internal service ports, IPv6 internal space, internal-only
  TLDs, embedded credentials and suspicious query encoding.

### Integration
- **`core/__init__.py` is now lazy (PEP 562).** It used to eagerly import the
  whole agent stack, so `from core.helpers import safe_error_message` — a leaf
  utility — cost **852ms**, of which 851.98ms was the package `__init__`.
  `import core` went 804ms → **2.8ms**; the test suite got ~29% faster. All 125
  exports verified byte-identical against a pre-refactor snapshot.
- **The shared embedding server is now started.** `EmbeddingInterconnect` is
  designed around one model shared over IPC, but nothing ever called
  `start_server_if_needed()`, so Jarvis and every sub-agent worker silently fell
  back to loading its own ~239M-parameter copy (~1GB each). Wired into `run.py`
  as a best-effort startup step; the local fallback still applies if it fails.
- `run.py` forces UTF-8 on stdout/stderr. Without it the banner emoji raised
  `UnicodeEncodeError` under cp1252 and killed the process **after** every
  subsystem had already started.
- `requirements.txt` gained 5 genuinely-required but undeclared deps:
  `python-multipart` (FastAPI `Form(...)`, without which `/api/chat` 500s),
  `psutil` (17 modules), `aiofiles`, `scipy`, `peft`.

### CI
The lint gate was **failing** on 19 undefined names, and the test job ran with
`continue-on-error: true` so failures never blocked. Both fixed: the blocking
ruff gate now covers `F821,F822,F811,F601,F602,B006,B023,B033,PLE,RUF006`
(clean), a broader advisory pass reports the remaining backlog, and tests gate
on `-m "not requires_host_env"`.

### Remaining failures (14, all host-environment)
Missing `gh`/`cargo`/`rustc`, no MuseScore soundfont, no fluidsynth, no
`~/.jarvis/secrets/`, Windows file locking, and Git-Bash POSIX path style. Those
that assert facts about the developer's machine are now marked
`requires_host_env` and excluded from CI.

## Completed Previous Cycle

### Test Remediation
- Installed 11 missing dependencies (cryptography, bcrypt, scipy, fluidsynth, pretty_midi, psutil, requests, aiofiles, rich, scikit-learn, pillow)
- Fixed genuine bug: `music_hybrid_renderer.py` — `np.max()` on zero-size audio for empty tracks
- 46 embedding tests gracefully skip when torch/sentence-transformers unavailable

### Security Hardening
- **Information disclosure**: Replaced 28 `str(e)` with `safe_error_message()` across 9 tool modules
- **WebAuthn**: Verified signature verification now implements Ed25519 + ECDSA (was previously documented as broken)
- **CSRF**: Verified middleware returns 403 on missing Origin/Referer (fixed in prior session)
- **CORS**: Fixed `_allowed_origins` → `_allowed_cors_origins` NameError that crashed cross-origin requests
- **Rate limiter**: Changed fatal-503-on-error to graceful degradation (fail-open)
- **CWD security**: Verified directory blocks system paths, restricts to home

### Critical Bugfixes
- **emotion_regulation.py**: Fixed `abs()` bug — worsening negative valence was scored as improvement
- **interoception.py**: Removed double allostatic load recovery (was ~3x intended rate)
- **context_cascade.py**: Fixed tuple `.get()` AttributeError on every cross-ref lookup
- **vector_memory.py**: `_rebuild_index` now preserves vector data (was silently discarded)
- **three_layer_context.py**: Fixed `top_k`→`k` param + `store_insight`→`add_memory`
- **execution_router.py**: Fixed wrong `llm.chat()` signature (positional→keyword args)
- **adaptive_agent_loop.py**: Removed `asyncio.to_thread` wrapping async LLM calls
- **execution_context.py**: Falls back to home directory when saved CWD is stale

### Thread Safety
- **bio_homeostasis.py**: Lock→RLock, added guards to 8 getter methods + fixed partial lock scope
- **circadian_rhythm.py**: Lock→RLock, added guards to 6 getter methods + input clamping
- **emotion_regulation/interoception**: Double-checked locking for all singletons

### Code Quality
- Extracted shared `_preprocess_turn()` from `run_step()`/`run_step_stream()` — saved 193 lines
- Removed 48 lines dead tone-shaping code in `music_hybrid_renderer.py`
- Removed duplicate `_generate_session_token` in `auth_middleware.py` (was invalidating tokens)
- Removed `_get_static_follow_up` dead code from agent.py
- agent.py reduced from 5027 → 4856 lines

## Architecture Status

| Subsystem | Modules | Tests | Health |
|-----------|---------|-------|--------|
| Core Agent | 22 | 200+ | ✅ Stable |
| Memory/Context | 18 | 80+ | ✅ Stable |
| Bio-Physiological | 6 | 108 | ✅ Stable |
| Consciousness | 7 | 93 | ✅ Stable |
| Predictive Processing | 3 | 37 | ✅ Stable |
| Security | 12 | 100+ | ⚠️ Some known limitations |
| Tools | 23 | 100+ | ✅ Stable |
| Runtime/Execution | 8 | 200+ | ✅ Stable |
| Embedding/LLM | 8 | 40+ | ✅ torch + peft installed, embeddings verified semantic |
| Interfaces | 4 | 50+ | ✅ Stable |
| Music System | 6 | 50+ | ✅ Stable (soundfont/fluidsynth optional) |

### Immediate (2026-09 cycle)

- [x] **Physical world model** (the "GEN1.5" research ask) — `core/world_model.py`
      (2026-09-04). A symbolic/probabilistic world model giving Jarvis native
      physical understanding, adopting the ARCHITECTURE of physical-AI world
      models (NVIDIA Cosmos WFM arXiv:2501.03575 / DeepMind Genie lineage):
      persistent scene belief → forward prediction → prediction error as the
      learning signal, seeded with Spelke's core knowledge. 12 typed schemas
      with per-schema precision: permanence, continuity (teleport detection),
      solidity, gravity/support (buoyancy respected), containment (co-movement
      + hiding), inertia, conservation, contact-before-effect causality,
      agentivity (unforced motion upgrades entities toward agents), Gibson/
      Gaver affordances (perceivable/hidden/false, calibrated by action
      outcomes with novelty reward on discovery), RCC-flavoured qualitative
      spatial relations, and decaying hypotheses about hidden state (RSSM
      analogue). Pure symbolic mechanics — zero LLM calls, deterministic.
      Integration: `PredictiveProcessor.process_world_model_errors()` (exact
      analogue of process_interoceptive_errors — world beliefs at level 2,
      physical-world attention at level 3, curiosity boost); agent init block
      + live_state getter; background `world_model_tick` (60s LOW); episodic
      memory of user-asserted physical events; scene belief injected into the
      system prompt with an "interesting-not-wrong" framing for schema
      violations. New endpoints: `GET /api/world/state`, `POST /api/world/event`.
      Tests: `tests/test_world_model.py` (41) — every schema's violation
      detection, buoyancy exemption, agentivity calibration, spatial
      contradiction, hypothesis decay, once-only error consumption, full
      world→predictive-processor pipeline, agent/background/prompt wiring
      pins. 156+ tests pass across the affected suites with zero regressions.

## Remaining Work

### Immediate
- [x] ~~Install torch/sentence-transformers~~ — done; `peft` was also required
- [x] ~~Fix remaining `str(e)` in logger.error calls~~ — **decided: leave them
      by design** (2026-08-31). A census found 769 `{e}`-interpolation sites in
      `logger.error`/`logger.warning` across 180 files. These are *internal*
      logs — the trusted-internal surface where exception detail is exactly
      what debugging needs. The earlier `str(e)` sweep was about *user-facing
      HTTP responses* (`safe_error_message()`), where the same content is an
      information-disclosure channel to attackers. Logs are not that channel.
      A blanket rewrite would trade real debuggability for no security gain.
      If a specific deployment treats log files as untrusted, revisit per-file
      then — not as a tree-wide sweep.
- [x] ~~Enforce `rate_limit_requests_per_minute` in SecurityMiddleware~~ — done this
      cycle. It is now a per-IP sliding window over the monotonic clock
      (loopback-exempt), enforced in both `validate_request()` and
      `validate_websocket_connection()` (the WebSocket path previously had no
      rate check at all). The app-config value now flows
      config.yaml → run.py → WebInterface → the middleware; the shipped default
      was corrected 120 → 2000 because the dashboard polls at 200ms (300
      req/min) and the old 120 would have 429'd the UI. Tests:
      `tests/test_auth_middleware.py::TestRateLimitEnforcement`.
- [x] ~~`tools/web.py::is_safe_url` DNS-rebinding TOCTOU~~ — done this cycle.
      `_pinned_fetch()` connects to the *validated* IP (URL host swapped via
      `httpx.URL.copy_with`, which anyio never re-resolves for IP literals)
      while `Host` (IDNA form) and TLS SNI (`sni_hostname` extension) keep the
      original name, so vhosts/certs behave unchanged. Validation and connect
      share one authoritative resolution —
      `SSRFProtector.resolve_validated_ips()` (new, exported) — so there is no
      second lookup to attack. Redirects are re-validated and re-pinned per hop,
      and the three duplicated redirect loops (search/fetch/image-download)
      collapsed onto the one primitive. Multi-IP fallback preserved.
      Tests: `tests/test_web_safety.py::TestPinnedFetch/TestResolveValidatedIps`.
- [x] ~~Monotonic clock for `execution_router`'s circuit breaker~~ — done this
      cycle, and extended to the whole defect class: all five breaker/cooldown
      implementations (`execution_router.CircuitBreaker`, `circuit_breaker.py`,
      `oracle_integration.OracleCircuitBreaker`, `graceful_degradation`,
      `resilience.py`) now time recovery windows with `time.monotonic()`.
      Wall-clock stamps are kept only as display twins where `get_status()`
      exposes them. Clock-jump regression tests added to
      `test_enhanced_router.py`, `test_resilience.py`,
      `test_telemetry_circuit.py`, `test_oracle_resilience.py`.

### Medium Term
- [x] ~~Split `agent.py` `get_system_prompt()` into dedicated `core/system_prompt.py`~~ —
      done this cycle. The 586-line method assembled ~30 context sections
      inline; assembly now lives in `core/system_prompt.SystemPromptBuilder`
      (one section method per source, order pinned in `_SECTION_ORDER`,
      per-section try/except with the original debug-log wording, mid-group
      appends preserved so a failed gather never discards blocks already
      added). The agent keeps only genuinely agent-owned concerns: the prompt
      cache (TTL + three-layer version invalidation), the self-awareness
      refresh/session update, and base-prompt selection. Also moved off the
      agent: `_get_oracle_context` (with its cache — now builder state),
      `_get_cognitive_alignment_context` (now `cognitive_alignment_context`),
      and `_get_bio_context_for_prompt`. agent.py: 4872 → ~4020 lines.
      Contract pinned by `tests/test_system_prompt.py` (18 tests: degradation,
      order, partial-failure semantics, consume-once insight, oracle caching,
      agent cache/TTL/L2-version semantics).
- [x] ~~Add `needs_permission()` to `terminal_execute` for dangerous commands~~ —
      and, deeper: the **adaptive loop's shell path** re-implemented only the
      HIGH-risk block of the terminal policy, so MEDIUM commands (package
      installs, docker ops, script execution) ran ungated there while being
      gated on the tool path — two paths, two policies. The gate now consults
      the terminal tool's own `needs_permission()` (single policy source,
      poka-yoke) and the agent's PermissionManager with the same contract
      `execute_tool` enforces (DENY refuses / no stored grant requires
      approval / stored allow proceeds). Fails closed if the policy oracle
      breaks. Tests: `test_review_refinements.py::TestAdaptiveLoopPermissionGate`
      (8 tests incl. real-tool parity and fail-closed).
- [x] ~~Fix `memory_hub.py` random embedding fallback~~ — `encode()` used to
      fabricate a random 768-dim vector when the embedder was unavailable and
      **cache it like a real one**, so every retrieval score became noise that
      looked plausible (the same failure class as the peft zero-vector
      incident). Now: honest `None`, nothing cached, conversation re-ranking
      degrades to deterministic lexical (Jaccard word-overlap) scoring, a
      cooldown-bounded warning instead of silence, and the pointless
      query-encode in `_retrieve_from_vector` (result discarded; search uses
      VectorMemory's own embedder) removed. Tests:
      `test_review_refinements.py::TestMemoryHubEmbeddingHonesty` (7 tests,
      incl. an RNG trap proving no fallback path exists).
- [x] ~~Add auth rate-limiting for `login_with_password`~~ — done (commit e3694a4)
- [x] ~~Break up `_setup_routes()`~~ — done this cycle: the 8,090-line method
      (~280 routes) is now a thin registrar plus 16 contiguous domain group
      methods (`_setup_frontend_routes` … `_setup_websocket_routes`), each a
      verbatim lift with unchanged decorators and registration order (smoke
      checked: 290 routes, no duplicates, key paths present). Two real bugs
      surfaced and fixed during the split: (1) route handlers call *sibling*
      handlers — the `/api/dashboard` and `/api/flow/live` aggregators compose
      ~12 other endpoints — which worked only as scope-siblings and would have
      NameError'd behind their try/excepts after the split; the ruff gate
      caught it before it shipped, and the calls now go through an explicit
      `self._route_handlers` registry populated per group at setup;
      (2) `/api/health` was registered twice, so FastAPI's first-match-wins
      made the rich `health_dashboard` permanently unreachable — now at
      `/api/health/dashboard`.
      Remaining monoliths: `_setup_oracle_routes()` 2,490 — **split this cycle**
      (2026-08-31) into a thin registrar + 9 domain methods, anchor-driven
      verbatim lift, route parity verified (360 routes before/after, zero
      duplicates); the split surfaced and fixed the `get_hostname`
      cross-group closure dependency (8 mobile-auth call sites would have
      NameError'd behind their try/excepts) and a missing StreamingResponse
      import for the mobile SSE path. Remaining: `run.py::main()` 1,283,
      and `agent.py` `run_step_stream` 658 / `think` 527.
      **`run.py::main()` split — deferred with rationale (2026-08-31):** an
      AST survey found 136 locals with heavy cross-phase references
      (daydream attached to agent callbacks, orchestrators referencing each
      other, subsystems wired both directions). A parameter-passing split
      would be a large rewrite of the boot path for readability only; the
      honest shape is a `BootContext` dataclass carrying the phases, which
      is a deliberate design task, not a mechanical lift. Deferred until
      someone is about to modify the boot sequence anyway.
- [x] ~~Register `tools/native_files.py` (1,127 LOC) with the agent, or retire it~~ —
      **RETIRED** (2026-08-31). File operations are owned by `tools/system.py`
      (registered file tools), `tools/terminal.py` (shell) and the
      ExecutionContext CWD; this module's own "~/.jarvis/workspace" model is a
      second, conflicting CWD model — registering it would recreate the
      divergent-policy problem the review cycle eliminated elsewhere. Labelled
      DEPRECATED at the top of the file; kept on disk because
      `tests/test_native_files.py` and `tests/test_security_hardening.py` pin
      its sandbox-escape regression coverage. Its exports remain in
      `tools/__init__.py` for the tests' direct imports.
- [x] ~~WebSocket CSRF protection for chat endpoint~~ — done (2026-08-31), and
      the review found worse holes in the same area first:
      **(a)** A WS handshake is a GET, so the HTTP CSRF middleware (POST/PUT/
      PATCH/DELETE only) never covered it, and none of the three WS endpoints
      (`/ws`, `/ws/api-activity`, `/api/mobile/ws`) checked Origin. Browsers
      always send Origin on a WS handshake and do NOT enforce same-origin on
      WS — any website in the user's browser could connect and chat as the
      user. New `_ws_origin_allowed()` guard (one policy: Origin absent →
      allowed for non-browser clients; present → netloc must equal Host or be
      in `JARVIS_CORS_ORIGINS`, exact equality, the localhost.evil.com
      suffix trick pinned by test). Tests: `tests/test_ws_origin_guard.py` (12).
      **(b)** The per-IP rate limiter added last cycle **never fired**: its
      per-client sliding window was an OrderedDict keyed by *client_ip*, so
      every request overwrote the single entry and `len(stamps)` was always 1.
      Now keyed by a monotonic sequence; the 3 pre-existing failing tests
      (test_remote_ip_blocked_after_limit, test_windows_are_per_ip,
      test_websocket_connection_rate_limited) pass again.
      **(c)** The HTTP CSRF allow-list match was dead code too: entries from
      `JARVIS_CORS_ORIGINS` are full origins ("https://host"), but the check
      compared them against the request's *netloc* — never equal, so
      configured origins were 403'd on POST. Both sides now normalize to
      netloc.
- [x] ~~vector_memory degraded-embedder fallback (design decision from the
      embedding-health section)~~ — decided and implemented as **search-time
      lexical fallback** (2026-08-31). `search()` consults the interconnect's
      `health()` (which names `returns_zero_vectors`) and rejects zero/
      deggenerate query vectors; when degraded it ranks by deterministic
      Jaccard word overlap instead of ranking on noise — zero vectors made
      every pair perfectly similar (the peft incident), and the hash fallback
      is pseudo-random per text. Results are flagged
      `metadata["lexical_fallback"]=True` so consumers can tell the basis
      apart. `store()` still accepts text while degraded (preserved for
      re-embedding). A test-caught bug: `_lexical_search` ignored
      `min_score` (fixed). Tests:
      `tests/test_vector_lexical_fallback.py` (12: zero-vector refusal,
      health-signal trust, word-overlap ranking, honest-empty, min_score,
      filter_fn, healthy-path-still-HNSW).
- [x] ~~Empty package directories left over from scaffolding~~ — removed
      (`core/channels/`, `core/gateway/`, `core/plugin_sdk/`); verified
      empty and unreferenced first.
- [x] ~~`core/security/session_security.py` wiring (the "needs your sign-off"
      item below)~~ — **wired this cycle** (2026-08-31, commit d958cda) with
      the SSRF layering rules: advisory-only (it can add a denial, never
      grant), fail-open on its own errors, SessionManager's return contract
      untouched. The capability that layers cleanly — per-IP progressive
      brute-force lockout — is live: failed validations feed it, session
      creation consults it, success clears it. This closes the hole where an
      IP hammering failed logins could just create fresh sessions forever
      (the old per-session lockout reset on every new session). The four
      store-owned capabilities (user-agent binding, concurrent limits,
      revocation, fixation prevention) still want the caller contract to
      carry user_agent — the remaining live-auth-path change.
- [x] ~~User-agent binding (the second of the five)~~ — wired (2026-08-31,
      commit c377a17). Session pins a 200-char UA snippet at creation;
      validation with a different browser refuses ("user-agent mismatch")
      and feeds record_failed_attempt (hijack probes drive the same per-IP
      progressive lockout as password brute-force). Policy is the security
      layer's `_verify_user_agent` (allows version drift, refuses different
      browsers) but the store is SessionManager's own metadata — no
      dual-store tokens. Advisory: absent UA / unpinned session /
      unavailable layer degrade open. WS handshake extracts and pins UA;
      facade wrappers carry it through. Remaining capabilities:
      concurrent-session limits, explicit revocation surface, fixation
      prevention — all straightforward against SessionManager's own store.
- [x] ~~Concurrent-session limit, revocation, fixation prevention (the
      remaining three)~~ — **all five capabilities now live** (2026-08-31,
      commits a52e11d + d3ed939). Concurrent limit: an IP above the layer's
      `max_concurrent_sessions` is refused at creation (loopback exempt —
      the local operator must never lock themselves out); my own test
      caught the off-by-one before commit. Revocation: `revoke_session`
      remembers revoked ids in a bounded 2000-entry set, so replaying one
      reports "Session revoked" (evidence of reuse) instead of "Invalid
      session"; `revoke_ip_sessions` bulk-revokes per IP. Fixation
      prevention: verified structural — ids are always server-generated
      `token_urlsafe(32)`, no client-supplied id parameter exists. The
      five capabilities now run entirely on SessionManager's own store
      with the layer's policy constants/comparison logic as the single
      source — session_security.py's separate runtime store is unused,
      which dissolves the dual-store problem rather than working around it.
- [x] ~~run_step / run_step_stream safety divergence~~ — closed (2026-08-31,
      commit 211f059): both paths now call the shared `_turn_safety_gate()`
      (LCM pre-shield + guardian auto-eviction, one chokepoint). Previously
      the voice/CLI path skipped both — an injection the web UI refused
      would execute over voice. The capability divergence (run_step stays
      single-shot for spoken output) is documented as a product choice, not
      a safety gap.
- [x] ~~Agentic-loop completion evaluation~~ — **fixed this cycle**
      (2026-08-31, commit ce766b3): `_evaluate_task_completion` had NEVER
      worked (phantom `stream=` kwarg raised TypeError into the broad except;
      the response-shape check looked for a top-level "content" key the
      Ollama client never returns), so every tool-using turn ended after the
      first round of tools and `max_agentic_iterations` was dead config. Also
      swept the 17 remaining `except: pass` sites in agent.py to context-aware
      `logger.debug`, and closed the parallel/single-tool sanitize gap.
- [x] ~~api_manager refusal contract~~ — **fixed this cycle** (2026-08-31,
      commit a4fef6b): `submit_chat()` used to return refusals shaped exactly
      like answers, so the 8 partially-adopted modules processed "Rate limit
      exceeded" as model output. Refusals now carry `_refused: True`, and
      `get_response_content`/`get_response_thinking` (the single extraction
      chokepoint all 23 consumer sites use) return their default for refused
      responses.
- [x] ~~`_setup_oracle_routes()` 2,490 lines~~ — split this cycle
      (2026-08-31, commit 6977d66) into a thin registrar + 9 domain methods;
      route parity verified (360 before/after, zero duplicates); surfaced and
      fixed the `get_hostname` cross-group closure dependency and a missing
      mobile SSE import.

### Orphan triage

Started at 34 modules (~15,470 LOC) imported by nothing. Now **27 (~12,940)**:
four wired into production, the rest labelled in-code so the next reader does not
re-investigate them. Decision per module below. **Wired** = now reachable from
production. **Superseded/deprecated/dormant** = kept on disk and labelled, never
deleted.

Every remaining orphan has been verified to import and construct its entry point
— nothing is blocked by breakage, only by product or design judgement. Two of
them (`music_intent`, `music_refinement`) were outright unimportable until this
cycle's missing-`threading` fix.

| Module | LOC | Decision |
|---|---|---|
| `core/safe_files.py` | 188 | **WIRED** — now the authoritative self-modification registry; `SelfEditEngine` consumes it (see below) |
| `core/pfc_observer.py` | 134 | **WIRED** — `insight_bridge.process_insight()` now reports insights to the Prefrontal Cortex |
| `core/security/ssrf_protection.py` | 387 | **WIRED** — layered into `tools/web.py::is_safe_url` |
| `core/embedding_server.py` | 312 | **WIRED** — started from `run.py`; was the reason every process loaded its own model |
| `core/cohesive_composer.py` | 710 | Deprecated by its own docstring — leave |
| `core/fluid_integration.py` | 475 | Deprecated by its own docstring — leave |
| `core/symbolic_composer{,_v2,_v3}.py` | 1,962 | Superseded by `_v4` (the only one with importers); labelled |
| `core/session.py` | ~700 | Unreachable — shadowed by the `core/session/` package; deprecated on purpose |
| `core/lazy_loader.py`, `core/performance.py` | 518 | **LABELLED SUPERSEDED** — the lazy-loading half is covered by `core/__init__.py` (PEP 562); `performance.py`'s metrics half is covered by the live `observability.py` (4 importers), `telemetry_aggregator.py` (2) and `metrics.py` (1) |
| `core/gpu_compute.py` | 817 | **LABELLED DORMANT** — not superseded, but unreachable by design: `run.py` disables every GPU backend at startup and torch is the CPU build. Revisit if the hardware/driver situation changes |
| `core/embedding_recommendations.py` | 273 | **RESOLVED** — not a module at all: a design note held in two string constants. Its three top-level `print()` calls ran at *import* time, so importing it dumped ~270 lines to stdout; now behind a `__main__` guard. Read it with `python -m core.embedding_recommendations` |
| `core/dashboard.py` | 212 | **LABELLED SUPERSEDED** — the production dashboard is the web UI served by `interfaces/web_react.py` (`/api/dashboard` + the React front end) |
| `core/agentic_project_manager.py`, `core/autonomous_worker.py` | 1,488 | 24/7 autonomous work. Substantial and coherent — needs a product decision on whether Jarvis should run unattended projects |
| `core/intelligence_hub.py` | 515 | Cross-subsystem knowledge sharing. Overlaps `insight_bridge` + `embedding_interconnect`; decide which owns the role |
| `core/fluid_information_flow.py` | 873 | **Two scoring bugs fixed; wiring still a judgement call.** Designed as a drop-in for `insight_bridge`'s deterministic classification (its methods map 1:1). Evaluating it for that role found: (1) `confidence` was the winner's raw boosted score clamped to 1.0, so it reported total certainty regardless of evidence — now the winner's share of total score, damped by the Shannon informativeness of the type distribution; (2) with zero keyword matches every level got the same sigmoid baseline, so the winner was decided by static weights alone and "high" (weight 1.0) always won — unrelated text scored `high`/`1.0`, and `irrelevant` could never win because its keyword list is empty. Absent evidence now returns `irrelevant`/`0.0`. Note its classifier only fires on *exact* multi-word signature phrases ("found pattern", "improved bound"), returning a flat prior for most natural phrasing — worth addressing before wiring |
| `core/reasoning_engine.py` | 519 | **Not an overlap — a unique capability.** Despite its "Enhanced Prefrontal Cortex" docstring it shares nothing with `prefrontal_cortex.py`, which is a latent *concept* model (~100M-param LCM, concept encoding). This is symbolic inference: abductive/deductive/inductive/analogical hypothesis generation, hypothesis testing with confidence update, and causal relations. It is the **only** such implementation in the tree (`compositional_reasoning.py` is music-specific). Verified working end to end; fixed a defect where `reason()` read only a key literally named `"evidence"`, silently discarding ordinary context so hypotheses were built from nothing. Natural consumer is `erdos_engine`, which tracks `strategies_tried` but does no hypothesis or causal reasoning — wiring it changes problem-solving behaviour, so left for you |
| `core/semantic_memory.py` | 520 | **Superseded in its original role** — a TF-IDF store built explicitly "without requiring heavy ML libraries", i.e. the no-torch fallback. torch + jina-v5 now work, so `vector_memory` (HNSW + real embeddings) covers it. Still the only lexical-similarity implementation in the tree, so it is the natural basis for a degraded-mode search if you ever want ranked results instead of zero vectors when the embedder dies — see below |
| `core/execution/turn_loop.py`, `core/bootstrap/stages.py` | 1,085 | Alternative turn-loop and startup sequencing; `agent.run_step*` and `run.py::main()` own these today |
| `core/master_practice.py`, `core/music_intent.py`, `core/music_refinement.py`, `core/musical_designer.py` | 2,664 | **All four verified functional** — each imports and its entry point constructs (`music_intent`/`music_refinement` were outright unimportable until this cycle's missing-`threading` fix). `music_creation_engine` imports 7 music modules but none of these. They are feature modules — intent-driven composition, iterative refinement, design architecture, practice — so wiring them changes *what Jarvis composes*. That is a creative decision, not a defect, so left for you. They are ready if you want them |
| `core/structured_logging.py` | 264 | **WIRED** — `request_id_middleware` in `web_react.py` gives every HTTP request a correlation id (honours inbound `X-Request-ID`, echoes it back). Fixed a latent bug first: `RequestContext.__exit__` looked up `token.var.old_value`, raising `AttributeError` into a bare `except`, so context vars were **never reset** and request ids leaked between requests |
| `core/runtime_integrity.py` | 291 | **WIRED** — started from `run.py`. Complements `startup_validator` (one-time pre-flight) by continuously hashing the files self-edit is forbidden to touch, which matters because Jarvis rewrites its own code. Now sources its watch list from `safe_files.CRITICAL_FILES` (was a private 6-entry copy vs the registry's 16, so most protected files went unmonitored). Fixed three bugs while wiring: `check_all_now()` called `run_until_complete()` on the already-running loop then `asyncio.run()`, both raising, leaving two orphaned coroutines per call; `start()` built its monitor coroutine before checking for a loop, so a sync caller got `_running=True`, no task and an un-awaited coroutine; and the monitor was started *before* the baseline hashes were taken |
| `core/llm_coordinated.py` | 198 | **Not superseded** — it is the adoption path *for* `api_manager` (a mixin giving any module `await self.llm_call(...)` with priority, quota and cache). Unused because most subsystems never adopted coordination. See "API manager bypass" below |
| `core/session/history_log.py` | 461 | **LABELLED SUPERSEDED** — `core/session_persistence.py` is in production (integration_harness, startup_validator, web) and strictly more capable: atomic writes, HMAC integrity, tool-boundary checkpointing, crash resume. The one capability with no equivalent there is `HistoryLog.as_markdown()` transcript rendering plus `get_by_kind`/`get_by_time_range` — worth porting if transcript export is ever wanted |

### `core/security/session_security.py` (345) — additive, not redundant

Earlier triage called this "overlaps `auth_middleware`". That was wrong. They
overlap only on the basics (create / validate / expire / IP-bind); on five
security dimensions `auth_middleware` has **no implementation at all**:

| Capability | `auth_middleware` | `session_security` |
|---|---|---|
| User-agent binding (hijack detection) | absent | yes |
| Concurrent session limiting | absent | yes |
| Session revocation (incl. bulk per-user) | absent | yes |
| Session fixation prevention | absent | yes |
| Progressive brute-force delays | absent | yes |

Not wired unilaterally, for one structural reason rather than doubt about
value: it owns its own session store, so unlike the SSRF protector it cannot be
layered as an advisory check that only adds denials — session *creation* has to
route through it. That is a change to the live auth path and wants your sign-off.

Suggested shape if you want it: have `auth_middleware.SessionManager.create_session()`
and `validate_session()` delegate to `get_session_security()`, keeping the
existing return contract so callers are untouched.

### Deep defect audit (2026-07-25)

An 8-way parallel audit of the live subsystems, with every finding handed to an
independent skeptic instructed to refute it: **48 candidates → 32 confirmed, 16
refuted** (mostly on reachability — the refutations were doing real work).
Highlights, all empirically reproduced before fixing:

**Security**
- `self_refine.is_safe_to_modify()` compared POSIX-style registry keys against
  Windows paths, so *every* membership test missed and it fell through to
  "editable" — self-refine considered `core/agent.py`, `core/llm.py` and `run.py`
  freely modifiable. It was also a **fourth** divergent copy of the file lists,
  whose SAFE list contained `core/knowledge_graph.py` (CRITICAL in the registry).
  Now normalises separators, consumes `core/safe_files.py`, and checks critical first.
- CSRF origin check was `A or (B and C)` — `and` binds tighter — where B/C was a
  prefix+port test, so `localhost.evil.com:8080` matched allowed `localhost:8080`
  and bypassed CSRF for every POST/PUT/PATCH/DELETE. Now exact netloc equality.
- CORS **failed open**: an unconfigured allow-list (the shipped default) meant
  "reflect every Origin", and GETs are not covered by the CSRF gate. Now
  deny-by-default with same-origin still permitted.
- `JARVIS_CORS_ORIGINS` was silently ignored by CSRF: the parsed set was a local,
  while the middleware read `getattr(self, '_allowed_cors_origins', set())`,
  which nothing ever assigned.
- `PermissionManager.check_permission()` had **no branch for a stored `"deny"`**,
  so `deny_permission(permanently=True)` read back as `None` ("undecided").
- `agent_loop`'s permission key `f"tool:{name}"` matched neither the exact-key
  store nor the config prefix map, so it could only ever return `None` — both
  grants and denies were invisible and the `== deny` branch was dead code.
- Six of seven `add_tool_result` call sites in `agent.py` stored **raw** tool
  output, bypassing `_sanitize_tool_output` — the indirect-prompt-injection
  barrier. All now route through `_remember_tool_result`.
- `mobile_auth` referenced `self._rp_id` (real attribute: `self.rp_id`) in ten
  places inside a `try/except Exception`, so the WebAuthn origin-mismatch check
  never ran and verification failed with a misleading message.
- `InputSanitizer.sanitize_string`'s `strict` flag is accepted and never read. The
  audit flagged this as a bug; it is not — `TestInputSanitizerAlwaysRuns` pins the
  unconditional filtering deliberately. Reverted the behavioural "fix" and
  corrected the *docstring* instead, which had it backwards. Worth knowing the
  cost: every request body has `;`, `&`, `|`, `$`, backticks and words like
  "curl " deleted, so code snippets and `$` amounts in chat are mutilated. If
  that matters, gate it on `strict` **and** update that test — do not just flip it.
- Rate-limiter concurrency slots leaked: the HTTP middleware released outside a
  `try/finally` (so any route exception leaked one), and none of the three
  WebSocket handlers released at all — permanent lockout after enough connections.

**Correctness / data loss**
- **12 sites sliced a `collections.deque`**, which raises `TypeError`. Several sat
  inside broad `except Exception`, so features became permanent silent no-ops —
  the learning pipeline's whole reflection phase among them. Found by sweeping for
  the pattern rather than fixing only the two the audit reported; a test now
  asserts the tree stays clean.
- `VectorMemory._load_index` restored the HNSW graph but not `metadata_index` /
  `importance_scores`, so after any restart every search returned empty metadata —
  the text was on disk the whole time.
- `VectorMemory._rebuild_index` passed `VectorEntry` objects where `insert()`
  expects a vector, *after* swapping in the empty index — the resulting TypeError
  destroyed every stored vector. Now builds into a local index and only publishes
  it once migration succeeds.
- **Four consumers** (`memory_hub`, `three_layer_context`, `unified_memory` ×2)
  treated `search()`'s `(id, score, metadata)` tuples as objects. `hasattr` is
  always False on a tuple and `.get()` raises, so they silently returned empty
  content — one with a hard-coded score of 1.0 that out-ranked every real result.
- `UnifiedMemory._load` never rebuilt the word index, so keyword search returned
  nothing at all after a restart.
- `os.getloadavg()` is POSIX-only and was called unconditionally, so **every**
  resource snapshot raised on Windows and pressure-based throttling never engaged.
  (It now visibly throttles under real memory pressure.)
- `tool_learning._execute_practice_scenario` fell off the end returning `None` on
  both the success and exception paths, so every practice run raised
  `AttributeError` in its caller.

### Embedding health is now observable

The `peft` incident exposed a whole failure *class*, not just a missing package:
when the embedding model fails to load, `encode()` returns zero vectors. Every
pair is then perfectly similar, so vector memory, the knowledge graph and
insight retrieval keep answering — with confident nonsense — instead of failing.
Nothing could detect that state from outside the process.

Added `ConceptEmbedder.health()` → `EmbeddingInterconnect.health()` →
`EmbeddingClient.health()`, registered as an `embeddings` subsystem in `run.py`
so it reaches the health manager and `/api/status`. It reports `degraded`,
`returns_zero_vectors: true`, and the underlying load errors (naming the missing
dependency), and deliberately does **not** trigger a model load, so polling
stays cheap.

Optional follow-up: when degraded, fall back to lexical ranking rather than
returning zero vectors. `core/semantic_memory.py`'s TF-IDF implementation is the
obvious basis. Not done — it needs a design decision, because TF-IDF vectors do
not fit the 768-dim contract the HNSW index is built around, so it would have to
be a search-time fallback in `vector_memory.search()` rather than an encode-time
substitution.

### API manager bypass — cost and rate-limit exposure

`core/api_manager.py` provides prioritisation, per-subsystem quotas, rate
limiting and response caching, and `run.py` starts it. But most subsystems never
call through it:

- **14 modules bypass it entirely**: `adaptive_agent_loop`, `agent`,
  `agentic_project_manager`, `autonomous_worker`, `checkpoint_permission`,
  `execution_planner`, `execution_router`, `growth`, `intelligent_orchestrator`,
  `proactive_engine`, `self_correction`, `self_evaluation`, `sub_agent`,
  `task_decomposition`
- **8 more are partially adopted** (both paths in the same module):
  `auto_refine`, `cognitive`, `erdos_engine`, `free_time`, `learning_pipeline`,
  `meta`, `self_refine`, `thinking`

**Update (this cycle) — an empirical re-audit found the "24/7 spend" claim
partly stale, then fixed what was real:**

- `self_evaluation`'s singleton is constructed **without an LLM** and
  `start()` is never called from `run.py`, so its evaluation/reflection loops
  do not run at all in the production boot path. `proactive_intelligence` and
  `daydream_engine` make **zero** direct LLM calls. `growth`'s
  `SelfGrowthTool` is constructed in `run.py` without an `llm_client`, so its
  LLM suggestion path is likewise dormant. The four named "always-on" spenders
  were, in practice, mostly not spending.
- A **real bug** was found in `self_correction` instead: sync
  `decompose_goal()` called the async `OllamaClient.chat()`, fed the
  un-awaited coroutine into `re.search` (TypeError), and the broad except
  swallowed it — LLM decomposition had *silently never worked*, and the
  coroutine leaked. Now: sync `decompose_goal()` detects async clients and
  skips cleanly (closing the coroutine), new `decompose_goal_async()` awaits
  them properly, and `execute_with_recovery()` (the production entry point via
  `agentic_orchestrator`) takes the async path — so decomposition now actually
  works *and* is quota'd.
- Routed through the api_manager (manager-first, direct fallback only when the
  manager itself is unavailable, refusal degrades instead of bypassing):
  `self_evaluation._llm_evaluate` + `_enhance_reflection_with_llm`,
  `growth._get_llm_suggestions`, `self_correction._decompose_with_llm` —
  all at `Priority.LEARNING` (20/min cap + 60s response cache, so repeated
  identical evaluations become free).
- Still open: the partially-adopted modules above (the live spend surface now
  that the always-on four are capped), and the user-facing paths
  (`agent`, `execution_router`, `adaptive_agent_loop`) which arguably *should*
  stay direct for latency. `llm_coordinated.LLMCoordinated` remains the
  migration vehicle for the rest.
- Tests: `tests/test_api_manager_routing.py` pins the routing contract
  (manager called with the right subsystem/priority, refusal never bypasses to
  a direct call, direct fallback works without a manager, and the
  sync/async decomposition fix).

### Long Term
- [ ] Module consolidation: merge 30+ overlapping memory/context modules, and the
      parallel families that never converged — `symbolic_composer` v1–v4,
      `memory`/`unified_memory`, `agent_loop`/`adaptive_agent_loop`,
      `rate_limiter`/`unified_rate_limiter`, `context`/`adaptive_context`/`unified_context`.
- [ ] Sandbox isolation via Docker/Firejail for autonomous execution
- [ ] Persistent session tokens (currently invalidate on restart)
- [x] ~~`core/session.py` (26KB) is unreachable — shadowed by the `core/session/`
      package~~ — deleted (2026-08-31). Python always preferred the package over
      the shadowed module, so the file was provably dead (nothing imported it;
      the two `test_untested_modules.py` imports resolved to the package's
      `Message` all along). `test_untested_modules.py`'s TestSession docstring
      now states that honestly.
- [ ] On Windows, `atomic_write_json` and the SSL key chmod to `0600`, which is a
      no-op — NTFS uses ACLs. Secrets Jarvis writes are not permission-protected
      there the way they are on Linux.
