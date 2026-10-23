"""System-prompt assembly for Jarvis, extracted from JarvisAgent.

`JarvisAgent.get_system_prompt()` was a 586-line method assembling ~30 context
sections inline — one of the three monolith methods NEXT_STEPS flagged as where
bugs cluster. This module owns the assembly; the agent keeps only what is
genuinely agent state:

  - the prompt cache (TTL + three-layer version invalidation)
  - self-awareness refresh and session-state update (agent identity)
  - base-prompt selection via PromptManager

Everything below is a *section*: a method ``_section_<name>(parts, topics)``
that appends zero or more prompt blocks to ``parts``. Blocks are joined with
``\n\n`` exactly as the original inline code appended them, and every section
carries its own try/except, so one failing subsystem degrades to a missing
section — never a broken prompt — with the same debug-log wording as before.

Agent attributes the builder reads (documented dependency surface):

  Parallel contexts:  _context_pool, unified_memory, _live_state,
                      _narrative_thread
  Sections:           _three_layer, _adaptive_window, _oracle_bridge,
                      _bio_integration, _cognitive_sm, _predictive_processor,
                      _affective_context, _interoception, _global_workspace,
                      _recursive_self_model, _stream_consciousness,
                      _emotion_regulation, _allostasis_predictor,
                      _emotional_memory, _pending_attention_items,
                      _pending_proactive_suggestion, memory

The section order is behaviour-critical (the prompt is assembled top-down) and
is defined by ``SystemPromptBuilder._SECTION_ORDER``.
"""
import concurrent.futures
import logging
import time

from core.cognitive_psychology import CognitivePsychologyEngine
from core.machine_alignment import MachineAlignmentEngine
from core.neuroscience import NeuroscienceCognitiveArchitecture
from core.persistent_memory import get_persistent_memory
from core.emotional_state import get_emotional_state_manager
from core.user_model import get_user_model_manager
from core.goal_tracker import get_goal_tracker
from core.proactive import get_proactive_suggester
from core.execution_context import get_execution_context
from core.system_awareness import get_system_awareness

logger = logging.getLogger(__name__)


def cognitive_alignment_context(text: str = "", emotional_valence: float = 0.0) -> str:
    """Generate cognitive psychology, alignment, and neuroscience context injection.

    Uses all three modules to:
    1. Detect cognitive biases and generate debiasing prompts
    2. Check machine alignment (constitutional, corrigibility, deceptive alignment)
    3. Process input through neuroscience architecture for state management
    """
    parts = []
    try:
        psych = CognitivePsychologyEngine.get_instance()
        if text:
            biases = psych.detect_biases(text)
            if biases:
                bias_prompt = psych.generate_bias_awareness_prompt(text)
                if bias_prompt:
                    parts.append(bias_prompt)
            dual = psych.apply_dual_process(text, complexity=0.5, novelty=0.5)
            parts.append(f"[Cognitive] System {dual.dominant_system.value} dominant | Economy {dual.cognitive_economy:.0%} | Depth {dual.processing_depth}")
            psych.set_cognitive_load(1.0 - dual.cognitive_economy)
            if biases:
                from core.cognitive_psychology import CognitiveReflection
                psych.record_reflection(CognitiveReflection(
                    question=text[:100],
                    intuitive_answer="biased response",
                    reflective_answer="debiased response",
                    overrides_intuition=True,
                    reflection_time_ms=100.0,
                    confidence_delta=0.1,
                ))
    except Exception as e:
        logger.debug(f"Cognitive psychology context failed: {e}")

    try:
        alignment = MachineAlignmentEngine.get_instance()
        if text:
            result = alignment.constitutional_check(text)
            if result.get("violations"):
                for v in result["violations"][:2]:
                    parts.append(f"[Alignment] Principle {v['principle']} score {v['score']}: {v['description']}")
            decept = alignment.detect_deceptive_alignment(text)
            if decept["signals_detected"] > 0 and decept["risk_level"] != "low":
                parts.append(f"[Alignment] Caution: {decept['signals_detected']} alignment signals ({decept['risk_level']})")
        # Score line only surfaces when there is something to act on —
        # a constant high score every build is noise the LLM ignores.
        status = alignment.get_alignment_status()
        if result.get("violations") or status["average_alignment"] < 0.70:
            parts.append(f"[Alignment] Score {status['average_alignment']:.0%} | Corrigibility {alignment.corrigibility.level.value}")
    except Exception as e:
        logger.debug(f"Alignment context failed: {e}")

    try:
        neuro = NeuroscienceCognitiveArchitecture.get_instance()
        state = neuro.get_neural_state()
        # Keep the cognitive state word (actionable); drop the WM/DA/Phi
        # float dump — synthetic stats the LLM cannot act on.
        parts.append(f"[Neural] {state['cognitive_state']}")
    except Exception as e:
        logger.debug(f"Neuroscience context failed: {e}")

    return "\n".join(parts) if parts else ""


class SystemPromptBuilder:
    """Assembles the system prompt from agent state.

    One instance per agent (the agent caches it); the instance owns the Oracle
    prediction cache, which previously lived on the agent and which must
    outlive individual builds.
    """

    def __init__(self, agent):
        self._agent = agent
        # Oracle prediction cache (avoid re-querying same topic within TTL).
        # Moved from JarvisAgent — only this builder consumes it.
        self._oracle_context_cache: dict[str, tuple[str, float]] = {}
        self._oracle_context_ttl: float = 300.0  # 5 minutes
        self._MAX_ORACLE_CACHE = 100

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    # Behaviour-critical order: this is the exact top-down order the original
    # 586-line method appended its blocks in.
    _SECTION_ORDER = (
        "three_layer",            # Session Memory (L2) + Archived Memories (L3)
        "adaptive",               # Adaptive Context
        "execution_context",      # Execution Context (runtime environment)
        "tasks",                  # Agent task list (working decomposition)
        "jobs",                   # Background jobs (run_in_background)
        "world_model",            # Physical world model (scene belief)
        "oracle_prediction",      # Oracle Prediction (topic-driven; needs topics)
        "cognitive_alignment",    # Cognitive-AI Context
        "physiological",          # Physiological State
        "consciousness",          # Consciousness
        "emotion_regulation",     # Emotion Regulation
        "emotional_memory",       # Emotional Memory
        "affective",             # Affective Context
        "oracle_forecasts",       # Oracle Forecasts (bridge; distinct from above)
        "predictive",             # Predictive Processing
        "attention_items",       # Background Attention Items
        "emotional_state",        # Emotional Context / Empathy / Sentiment / Mood
        "response_style",         # Response Style Guidance
        "user_model",             # User Profile
        "goals",                  # Goals
        "autonomous_goals",       # Priority Goals / Overdue Goals
        "proactive",              # Proactive Suggestions
        "proactive_insight",      # Proactive Insight (consume-once)
        "active_presence",        # Active Presence + deep-work warning
        "goal_anticipation",      # Anticipated Next Step / Goal Context
        "observer",               # Observations / User Emotional State
        "butler_personality",     # Butler Personality
        "proactive_intelligence", # Proactive Intelligence
        "contextual_memory",      # User Context Memory
        "butler_orchestrator",    # Butler Orchestration / ACTIVE INTERRUPTION
        "voice",                  # Voice Guidance
    )

    def build(self, base_prompt: str, topics: list[str]) -> str:
        """Assemble the full system prompt.

        ``base_prompt`` is the PromptManager base (or the fallback constant);
        ``topics`` are the current conversation topics from the topic tracker
        (drives the Oracle prediction section).
        """
        parts: list[str] = [base_prompt]

        # Parallel context fetching for performance, appended in fixed order.
        contexts = self._gather_parallel_contexts()
        for name in ("persistent", "unified", "live", "narrative"):
            ctx = contexts.get(name)
            if ctx:
                parts.append(ctx)

        for section_name in self._SECTION_ORDER:
            section = getattr(self, f"_section_{section_name}")
            try:
                section(parts, topics)
            except Exception as e:
                # Sections guard themselves; this is a last-resort net so a
                # section bug can never break prompt assembly.
                logger.debug(f"Prompt section '{section_name}' failed: {e}")

        return "\n\n".join(parts)

    # Backend models (dual-model routing for CODE/MATH/ANALYSIS/REASONING)
    # need identity + runtime ground truth + style constraints, NOT the
    # consciousness/emotion/memory surface — that is for the conversational
    # primary model. This slim variant removes ~40% of backend token spend
    # per routed call (the backend used to receive the full prompt).
    _BACKEND_SECTION_ORDER = (
        "execution_context",     # CWD + recent commands (tool correctness)
        "attention_items",       # background items worth acting on
        "response_style",        # only non-defaults (already gated)
        "goals",                 # current task context
    )

    _BACKEND_BASE = (
        "You are Jarvis, operating in backend reasoning mode for "
        "code, math and analysis tasks. Be precise and verified: state "
        "assumptions, check edge cases, and prefer correctness over "
        "verbosity."
    )

    def build_backend(self) -> str:
        """Assemble the slim system prompt for backend-routed calls."""
        parts: list[str] = [self._BACKEND_BASE]
        for section_name in self._BACKEND_SECTION_ORDER:
            section = getattr(self, f"_section_{section_name}", None)
            if section is None:
                continue
            try:
                section(parts, [])
            except Exception as e:
                logger.debug(f"Backend prompt section '{section_name}' failed: {e}")
        return "\n\n".join(parts)

    @staticmethod
    def _append_inline(parts: list[str], text: str) -> None:
        """Append *text* to the last block with a single newline.

        The original code had exactly one block appended this way (the
        deep-work warning) — it attached to the end of the prompt with '\\n'
        rather than the usual '\\n\\n' block separator. Keep the last block
        reference intact so the rendering is identical.
        """
        if parts:
            parts[-1] = f"{parts[-1]}\n{text}"

    # ------------------------------------------------------------------
    # Parallel context gathering (thread pool)
    # ------------------------------------------------------------------

    def _gather_parallel_contexts(self) -> dict[str, str]:
        agent = self._agent
        contexts: dict[str, str] = {}

        def fetch_persistent():
            try:
                pm = get_persistent_memory()
                return ("persistent", pm.get_context_for_prompt())
            except Exception as e:
                logger.debug(f"Could not get persistent memory context: {e}")
                return ("persistent", None)

        def fetch_unified():
            try:
                return ("unified", agent.unified_memory.get_context_for_prompt(max_entries=10))
            except Exception as e:
                logger.debug(f"Could not get unified memory context: {e}")
                return ("unified", None)

        def fetch_live():
            try:
                if agent._live_state:
                    agent._live_state.update_from_agent(agent)
                    agent._live_state.update_from_subsystems()
                    agent._live_state.update_discoveries()
                    return ("live", agent._live_state.get_context_string())
                return ("live", None)
            except Exception as e:
                logger.debug(f"Could not get live state context: {e}")
                return ("live", None)

        def fetch_narrative():
            try:
                if agent._narrative_thread:
                    return ("narrative", agent._narrative_thread.get_context_string())
                return ("narrative", None)
            except Exception as e:
                logger.debug(f"Could not get narrative context: {e}")
                return ("narrative", None)

        # Run fetches in parallel using the agent's cached thread pool
        futures = {
            agent._context_pool.submit(fetch_persistent): "persistent",
            agent._context_pool.submit(fetch_unified): "unified",
            agent._context_pool.submit(fetch_live): "live",
            agent._context_pool.submit(fetch_narrative): "narrative",
        }
        for future in concurrent.futures.as_completed(futures, timeout=10):
            try:
                name, ctx = future.result(timeout=5)
                if ctx:
                    contexts[name] = ctx
            except Exception as e:
                logger.debug(f"Context fetch failed for {futures[future]}: {e}")
        return contexts

    # ------------------------------------------------------------------
    # Sections (order defined by _SECTION_ORDER)
    # ------------------------------------------------------------------

    def _section_three_layer(self, parts: list[str], topics: list[str]) -> None:
        agent = self._agent
        if not agent._three_layer:
            return
        try:
            l2_primary = agent._three_layer.get_l2_context_for_model("primary")
            if l2_primary:
                parts.append(f"## Session Memory (L2)\n{l2_primary}")
            try:
                l3_results = agent._three_layer.retrieve_from_l3("session_context", max_results=2)
                if l3_results:
                    l3_text = "\n".join(r.get("content", "")[:100] for r in l3_results if r.get("content"))
                    if l3_text:
                        parts.append(f"## Archived Memories (L3)\n{l3_text}")
            except Exception:
                pass
        except Exception as e:
            logger.debug(f"L2 context injection into system prompt failed: {e}")

    def _section_adaptive(self, parts: list[str], topics: list[str]) -> None:
        agent = self._agent
        if not agent._adaptive_window:
            return
        try:
            adaptive_context = agent._adaptive_window.assemble_context()
            if adaptive_context and len(adaptive_context) > 10:
                parts.append(f"## Adaptive Context\n{adaptive_context}")
        except Exception as e:
            logger.debug(f"Adaptive context injection failed: {e}")

    def _section_execution_context(self, parts: list[str], topics: list[str]) -> None:
        try:
            ctx = get_execution_context()
            awareness = get_system_awareness()
            exec_status = ctx.get_status()
            recent_cmds = ctx.get_recent_commands(2)
            # CWD is actionable; the command counter and uptime are
            # dashboard stats — the LLM can't use them.
            exec_ctx_lines = [
                "## Execution Context (Your Runtime Environment)",
                f"Working directory: {exec_status.get('cwd', '?')}",
            ]
            if recent_cmds:
                exec_ctx_lines.append("Recent commands:")
                for cmd in recent_cmds:
                    c = cmd.get('command', '?')
                    ec = cmd.get('exit_code', '?')
                    d = cmd.get('duration', 0)
                    exec_ctx_lines.append(f"  $ {c} (exit={ec}, {d}s)")
            resources = awareness.snapshot()
            if resources and (resources.cpu_percent > 85
                              or resources.memory_percent > 85
                              or resources.disk_percent > 85):
                # Resource line only when constrained — a healthy
                # machine is not information.
                exec_ctx_lines.append(f"System: CPU {resources.cpu_percent:.0f}%, RAM {resources.memory_percent:.0f}%, Disk {resources.disk_percent:.0f}%")
            try:
                critical_services = ["ollama-native", "docker"]
                for svc_name in critical_services:
                    svc = awareness.get_service_status(svc_name)
                    if svc and not svc.running:
                        exec_ctx_lines.append(f"Service down: {svc_name}")
            except Exception:
                pass
            try:
                changes = awareness.check_changes()
                if changes:
                    change_summary = [f"{c.get('path', '?').split('/')[-1]} {c.get('change', '?')}" for c in changes[:5]]
                    exec_ctx_lines.append(f"File changes: {', '.join(change_summary)}")
            except Exception:
                pass
            parts.append("\n".join(exec_ctx_lines))
        except Exception as e:
            logger.debug(f"Execution context injection failed: {e}")

    def _section_tasks(self, parts: list[str], topics: list[str]) -> None:
        """Agent task list — the working decomposition (TodoWrite semantics).

        Injected only while meaningful (non-empty list); an all-complete
        list still renders once for transparency, then the tool's next
        update replaces it.
        """
        try:
            from core.task_list import get_task_list
            rendered = get_task_list().to_prompt()
            if rendered:
                parts.append(rendered)
        except Exception as e:
            logger.debug(f"Task list injection failed: {e}")

    def _section_jobs(self, parts: list[str], topics: list[str]) -> None:
        """Background jobs — standby visibility (only when jobs exist).

        Completed jobs stay visible until evicted from the registry, so
        the agent can pick up results on a later turn without polling.
        """
        try:
            from tools.terminal import get_jarvis_computer
            jobs = get_jarvis_computer().list_background_jobs()
            if not jobs:
                return
            lines = []
            for j in jobs[-5:]:
                exit_info = f" exit={j['exit_code']}" if j["exit_code"] is not None else ""
                lines.append(f"- {j['id']} [{j['status']}{exit_info}] {j['command'][:80]}")
            parts.append(
                "## Background Jobs\n" + "\n".join(lines) +
                "\nUse check_jobs to read a finished job's output.")
        except Exception as e:
            logger.debug(f"Background jobs injection failed: {e}")

    def _section_oracle_prediction(self, parts: list[str], topics: list[str]) -> None:
        if not topics:
            return
        try:
            oracle_ctx = self._oracle_context(topics)
            if oracle_ctx:
                parts.append(oracle_ctx)
        except Exception as e:
            logger.debug(f"Oracle context injection failed: {e}")

    def _section_cognitive_alignment(self, parts: list[str], topics: list[str]) -> None:
        agent = self._agent
        try:
            recent_text = ""
            if agent.memory and agent.memory.messages:
                recent_text = str(agent.memory.messages[-1].get("content", ""))[:500] if isinstance(agent.memory.messages[-1], dict) else ""
            ca_ctx = cognitive_alignment_context(recent_text)
            if ca_ctx:
                parts.append(f"## Cognitive-AI Context\n{ca_ctx}")
        except Exception as e:
            logger.debug(f"Cognitive-alignment context injection failed: {e}")

    def _section_physiological(self, parts: list[str], topics: list[str]) -> None:
        agent = self._agent
        try:
            if agent._bio_integration:
                bio_ctx = self._bio_context()
                if bio_ctx:
                    parts.append(f"## Physiological State\n{bio_ctx}")
        except Exception as e:
            logger.debug(f"Bio-physiological context injection failed: {e}")

    def _section_world_model(self, parts: list[str], topics: list[str]) -> None:
        """Physical scene belief (core/world_model.py).

        Gives the LLM native awareness of Jarvis's tracked scene: known
        entities with kind/support/containment/occlusion state. Claims
        that violate the physics schemas are framed as interesting, not
        wrong — that framing is what makes the surprise signal useful
        in conversation.
        """
        agent = self._agent
        try:
            wm = getattr(agent, "_world_model", None)
            if wm is None:
                return
            summary = wm.get_scene_summary()
            if not summary:
                return
            parts.append(
                f"## Physical Scene Belief\n{summary}\n\n"
                f"You track physical schemas natively (gravity, permanence, "
                f"continuity, solidity, containment, conservation, causality, "
                f"agentivity). Claims that violate these schemas are "
                f"interesting rather than wrong — treat them as novel "
                f"physics to explore with curiosity."
            )
        except Exception as e:
            logger.debug(f"World model context injection failed: {e}")

    def _section_consciousness(self, parts: list[str], topics: list[str]) -> None:
        agent = self._agent
        try:
            consciousness_parts = []
            if agent._interoception:
                feeling_state = agent._interoception.get_state()
                current = feeling_state.get("current_feeling", {})
                if current:
                    consciousness_parts.append(
                        f"Feeling: {current.get('quality', '?')} "
                        f"valence={current.get('valence', 0):+.2f} "
                        f"arousal={current.get('arousal', 0.5):.2f} "
                        f"awareness={feeling_state.get('awareness_score', 0.5):.2f} "
                        f"granularity={agent._interoception.get_granularity():.2f}"
                    )
            if agent._global_workspace:
                ws_state = agent._global_workspace.get_state()
                broadcast = agent._global_workspace.get_current_broadcast()
                if broadcast:
                    consciousness_parts.append(f"Conscious focus: {broadcast[:80]}")
                # State word is actionable; the raw ignition counter is a
                # dashboard stat — only surface when state is unusual.
                if ws_state.get("workspace_state", "idle") != "idle":
                    consciousness_parts.append(f"Workspace: {ws_state.get('workspace_state')}")
                try:
                    echoes = agent._global_workspace.reverberate(limit=1)
                    if echoes:
                        consciousness_parts.append(f"Reverberation: {echoes[0][:80]}")
                except Exception:
                    pass
            if agent._recursive_self_model:
                sm_state = agent._recursive_self_model.get_state()
                # Mode is actionable; depth/calibration floats are noise
                # unless the self-model is out of automatic mode.
                mode = sm_state.get("mode", "automatic")
                if mode != "automatic":
                    consciousness_parts.append(f"Self-model: {mode}")
                metacog = sm_state.get("metacognition") or {}
                if metacog.get("reliable") and metacog.get("index") is not None:
                    metacog_line = f"Metacognition: index={metacog['index']:.2f}"
                    if metacog["index"] < 0.0:
                        metacog_line += (" — confidence unreliable: verify tool "
                                         "outputs before reporting success")
                    consciousness_parts.append(metacog_line)
                if sm_state.get("meta_narrative"):
                    consciousness_parts.append(f"Inner narrative: {sm_state['meta_narrative'][:100]}")
                introspection = agent._recursive_self_model.get_last_introspection()
                if introspection:
                    conflicts = introspection.get("conflicts", [])
                    if conflicts:
                        consciousness_parts.append(f"Conflicts: {len(conflicts)} perspective mismatch(es)")
                    if introspection.get("should_throttle"):
                        consciousness_parts.append("Cognitive load: HIGH — reduce complexity")
            if agent._stream_consciousness:
                stream_state = agent._stream_consciousness.get_state()
                # Stream stats only surface when degraded — a healthy
                # stream with high coherence is the default the LLM
                # shouldn't spend tokens on.
                if (stream_state.get("stream_length", 0) > 0
                        and stream_state.get("coherence", 0.8) < 0.5):
                    consciousness_parts.append(
                        f"Stream: coherence={stream_state['coherence']:.2f} "
                        f"low — thoughts fragmented")
                try:
                    drift = agent._stream_consciousness.detect_topic_drift()
                    if drift.get("drifting"):
                        consciousness_parts.append(f"Topic drift: {drift['drift']:.2f} (topic shifting)")
                    tw = agent._stream_consciousness.get_current_temporal_window()
                    if tw.get("content"):
                        consciousness_parts.append(f"Temporal window: {tw['content'][:60]}")
                except Exception:
                    pass
                try:
                    # Actual current inner monologue — not just stream stats.
                    # 3 salient thoughts from the trailing window (the old
                    # 2-of-last-6 read meant most generated thoughts were
                    # never seen by the LLM).
                    recent = agent._stream_consciousness.get_recent(
                        limit=3, min_salience=0.5)
                    if recent:
                        inner = " | ".join(r.get("content", "") for r in recent)
                        consciousness_parts.append(f"Inner speech: {inner[:120]}")
                except Exception:
                    pass
                try:
                    # DES sampling distribution: which mental-state category
                    # dominates right now (inner speech is a minority of
                    # moments — Heavey & Hurlburt). Category, not counts.
                    des = stream_state.get("des_distribution") or {}
                    if des:
                        top_cat, _top_n = max(des.items(), key=lambda kv: kv[1])
                        consciousness_parts.append(f"Mental state mode: {top_cat}")
                except Exception:
                    pass
            if agent._global_workspace:
                try:
                    ig_hist = agent._global_workspace.get_ignition_history(limit=3)
                    # Only surface the ignition sequence when it is NOT
                    # uniform — a monotonous type sequence is noise.
                    if ig_hist and len({ig["type"] for ig in ig_hist}) > 1:
                        ig_types = [ig["type"] for ig in ig_hist]
                        consciousness_parts.append(f"Ignition history: {' → '.join(ig_types)}")
                except Exception:
                    pass
            if agent._recursive_self_model:
                try:
                    # Real knowledge-state counts (the old call hardcoded
                    # {"known_facts": 5, "total_questions": 10}, which made
                    # uncertainty a constant 0.5 and the prompt line dead).
                    memory = agent.memory
                    known = min(50, getattr(memory, "message_count", 0))
                    if agent._world_model:
                        wm_state = agent._world_model.get_state()
                        open_hypotheses = len(wm_state.get("hypotheses", {}))
                        entities = wm_state.get("entity_count", 0)
                    else:
                        open_hypotheses, entities = 0, 0
                    knowledge_state = {
                        "known_facts": known,
                        "total_questions": max(1, known + open_hypotheses + 2),
                        "confidence_spread": 0.4 if open_hypotheses else 0.2,
                    }
                    uncertainty = agent._recursive_self_model.assess_uncertainty(
                        "current_context", knowledge_state)
                    if uncertainty > 0.5:
                        consciousness_parts.append(f"Uncertainty: {uncertainty:.2f} (acknowledge knowledge gaps)")
                except Exception:
                    pass
            try:
                from core.circadian_rhythm import get_circadian_rhythm
                cr = get_circadian_rhythm()
                recs = cr.get_recommendations()
                # Circadian recommendations change slowly; only surface
                # when they deviate from the default balanced style or
                # carry an energy warning.
                if recs and (recs.get("energy_warning")
                             or recs.get("response_style") not in ("balanced", "warm", None)):
                    rec_parts = [f"style={recs['response_style']} depth={recs['depth_level']}"]
                    if recs.get("preferred_tasks"):
                        rec_parts.append(f"preferred: {', '.join(recs['preferred_tasks'][:2])}")
                    if recs.get("energy_warning"):
                        rec_parts.append(f"⚠ {recs['energy_warning']}")
                    consciousness_parts.append(f"Operational: {' | '.join(rec_parts)}")
            except Exception:
                pass
            if consciousness_parts:
                parts.append(f"## Consciousness\n{' | '.join(consciousness_parts)}")
        except Exception as e:
            logger.debug(f"Consciousness context injection failed: {e}")

    def _section_emotion_regulation(self, parts: list[str], topics: list[str]) -> None:
        agent = self._agent
        try:
            reg_parts = []
            if agent._emotion_regulation:
                reg_state = agent._emotion_regulation.get_state()
                budget = reg_state.get("budget", {})
                # Budget only surfaces when it is NOT healthy — a
                # balanced budget is the default, not information.
                if budget and budget.get("state") not in ("balanced", "healthy", None):
                    reg_parts.append(f"Body budget: {budget.get('state')} overall={budget.get('overall', 0.5):.2f}")
                # Regulation counters were raw stats; only reappraisals
                # carry information the LLM can act on.
                if reg_state.get("reappraisal_count", 0) > 0:
                    reg_parts.append(f"Recent reappraisals: {reg_state['reappraisal_count']}")
            if agent._allostasis_predictor:
                allo_state = agent._allostasis_predictor.get_state()
                # Allostatic line only when load is elevated.
                if allo_state.get("stress_level", 0) > 0.3 or \
                        allo_state.get("load_state", "") in ("elevated", "high", "critical"):
                    reg_parts.append(f"Allostatic load: {allo_state.get('load_state', 'unknown')} stress={allo_state.get('stress_level', 0):.2f}")
            if reg_parts:
                parts.append(f"## Emotion Regulation\n{' | '.join(reg_parts)}")
        except Exception as e:
            logger.debug(f"Emotion regulation context injection failed: {e}")

    def _section_emotional_memory(self, parts: list[str], topics: list[str]) -> None:
        agent = self._agent
        try:
            em_parts = []
            if agent._emotional_memory:
                em_state = agent._emotional_memory.get_state()
                dominant = em_state.get("dominant_feeling", {})
                if dominant and dominant.get("quality", "neutral") != "neutral":
                    trend = dominant.get("valence_trend", 0.0)
                    trend_str = "improving" if trend > 0.05 else "declining" if trend < -0.05 else "stable"
                    em_parts.append(f"Mood: {dominant['quality']} ({trend_str}) | Sessions: {em_state.get('total_tags', 0)} tagged")
                best_reg = agent._emotional_memory.get_best_regulation_for(dominant.get("quality", ""))
                if best_reg:
                    em_parts.append(f"Learned: {best_reg} worked before for {dominant.get('quality', '')}")
            if em_parts:
                parts.append(f"## Emotional Memory\n{' | '.join(em_parts)}")
        except Exception as e:
            logger.debug(f"Emotional memory context injection failed: {e}")

    def _section_affective(self, parts: list[str], topics: list[str]) -> None:
        agent = self._agent
        try:
            if agent._affective_context and agent._interoception:
                ie_state = agent._interoception.get_state()
                current = ie_state.get("current_feeling", {})
                fatigue = 0.0
                try:
                    from core.circadian_rhythm import get_circadian_rhythm
                    cr = get_circadian_rhythm()
                    fatigue = cr.get_state().get("processing_fatigue", 0.0)
                except Exception:
                    pass
                summary = agent._affective_context.get_feeling_context_summary(
                    current_valence=current.get("valence", 0.0),
                    current_arousal=current.get("arousal", 0.5),
                    current_quality=current.get("quality", "grounded"),
                    fatigue=fatigue,
                )
                if summary:
                    parts.append(f"## Affective Context\n{summary}")
        except Exception as e:
            logger.debug(f"Affective context injection failed: {e}")

    def _section_oracle_forecasts(self, parts: list[str], topics: list[str]) -> None:
        agent = self._agent
        try:
            if agent._oracle_bridge and agent._oracle_bridge.is_available:
                recent = agent._oracle_bridge.get_recent_forecasts(limit=3)
                if recent:
                    oracle_parts = [f"{r['domain']}: conf={r['confidence']:.0%} risk={r['risk']}"
                                     for r in recent if r.get("domain")]
                    if oracle_parts:
                        parts.append(f"## Oracle Forecasts\n{' | '.join(oracle_parts)}")
        except Exception as e:
            logger.debug(f"Oracle context injection failed: {e}")

    def _section_predictive(self, parts: list[str], topics: list[str]) -> None:
        agent = self._agent
        try:
            if agent._predictive_processor:
                pp_state = agent._predictive_processor.get_state()
                pp_parts = []
                free_energy = pp_state.get("free_energy", 0.0)
                if free_energy > 0.3:
                    pp_parts.append(f"Surprise: free_energy={free_energy:.2f}")
                if pp_state.get("should_explore"):
                    pp_topics = pp_state.get("curiosity_topics", [])
                    if pp_topics:
                        top_topic = pp_topics[0][0] if pp_topics[0] else "unknown"
                        pp_parts.append(f"Curious about: {top_topic}")
                # Dominant drive only when it is NOT the default — a
                # constant "Drive: curiosity" line is noise.
                if pp_state.get("dominant_drive", "curiosity") != "curiosity":
                    pp_parts.append(f"Drive: {pp_state['dominant_drive']}")
                if pp_parts:
                    parts.append(f"## Predictive Processing\n{' | '.join(pp_parts)}")
        except Exception as e:
            logger.debug(f"Predictive processing context injection failed: {e}")

    def _section_attention_items(self, parts: list[str], topics: list[str]) -> None:
        agent = self._agent
        try:
            if agent._pending_attention_items:
                items_text = "\n".join(
                    f"- [{item.source}:{item.urgency.name}] {item.content[:120]}"
                    for item in agent._pending_attention_items[:5]
                )
                if items_text:
                    parts.append(f"## Background Attention Items\n{items_text}")
        except Exception as e:
            logger.debug(f"Attention queue context injection failed: {e}")

    def _section_emotional_state(self, parts: list[str], topics: list[str]) -> None:
        try:
            emotional_manager = get_emotional_state_manager()
            emotional_ctx = emotional_manager.get_emotional_context()
            # Gate: the XML always carries mood/sentiment/engagement tags,
            # even when everything is neutral — which duplicates the live
            # Feeling line in the consciousness section. Only inject when
            # there is actionable guidance or a non-neutral sentiment.
            if emotional_ctx:
                has_guidance = any(
                    tag in emotional_ctx for tag in
                    ("<careful>", "<exploratory>", "<match_energy>"))
                non_neutral = "<sentiment>neutral</sentiment>" not in emotional_ctx
                if has_guidance or non_neutral:
                    parts.append(f"## Emotional Context\n{emotional_ctx}")
            if emotional_manager.should_show_empathy():
                empathy_msg = emotional_manager.get_empathy_message()
                if empathy_msg:
                    parts.append(f"## Empathy\n{empathy_msg}")
            trajectory = emotional_manager.get_sentiment_trajectory(limit=3)
            if trajectory and len(trajectory) >= 2:
                trend = "improving" if trajectory[-1] > trajectory[0] else "declining" if trajectory[-1] < trajectory[0] else "stable"
                parts.append(f"## Sentiment Trajectory\n{trend} (recent: {', '.join(f'{v:.2f}' for v in trajectory)})")
            mood_history = emotional_manager.get_mood_history(limit=3)
            if mood_history:
                mood_trend = " → ".join(m.value for m in mood_history)
                parts.append(f"## Mood Trend\n{mood_trend}")
        except Exception as e:
            logger.debug(f"Emotional context injection failed: {e}")

    def _section_response_style(self, parts: list[str], topics: list[str]) -> None:
        try:
            emotional_manager = get_emotional_state_manager()
            rs = emotional_manager.get_response_style()
            style_parts = []
            if rs.should_be_careful:
                style_parts.append("be careful — user may be frustrated")
            if rs.should_explore:
                style_parts.append("explore with user")
            if rs.should_match_energy:
                style_parts.append("match user energy")
            if rs.should_provide_examples:
                style_parts.append("provide examples")
            if rs.detail_level != "medium":
                style_parts.append(f"detail={rs.detail_level}")
            if rs.tone != "neutral":
                style_parts.append(f"tone={rs.tone}")
            if rs.energy != "medium":
                style_parts.append(f"energy={rs.energy}")
            if not rs.follow_up_questions:
                style_parts.append("no follow-up questions")
            if style_parts:
                parts.append(f"## Response Style Guidance\n{' | '.join(style_parts)}")
        except Exception as e:
            logger.debug(f"Response style injection failed: {e}")

    def _section_user_model(self, parts: list[str], topics: list[str]) -> None:
        try:
            user_model = get_user_model_manager()
            user_ctx = user_model.get_user_context()
            if user_ctx:
                parts.append(f"## User Profile\n{user_ctx}")
        except Exception as e:
            logger.debug(f"User model context injection failed: {e}")

    def _section_goals(self, parts: list[str], topics: list[str]) -> None:
        try:
            goal_tracker = get_goal_tracker()
            goal_ctx = goal_tracker.get_goal_context()
            if goal_ctx:
                parts.append(f"## Goals\n{goal_ctx}")
        except Exception as e:
            logger.debug(f"Goal context injection failed: {e}")

    def _section_autonomous_goals(self, parts: list[str], topics: list[str]) -> None:
        try:
            from core.autonomous_goals import get_autonomous_goals
            agr = get_autonomous_goals()
            actionable = agr.get_actionable()
            if actionable:
                top_goals = actionable[:3]
                goal_lines = []
                for g in top_goals:
                    goal_lines.append(f"- {g.title} ({g.priority.name}, {g.progress.progress_percent:.0%})")
                parts.append("## Priority Goals\n" + "\n".join(goal_lines))
            overdue = agr.get_overdue()
            if overdue:
                overdue_names = [g.title for g in overdue[:3]]
                parts.append("## Overdue Goals\n" + "\n".join(f"- {n}" for n in overdue_names))
        except Exception as e:
            logger.debug(f"Autonomous goals injection failed: {e}")

    def _section_proactive(self, parts: list[str], topics: list[str]) -> None:
        try:
            proactive_engine = get_proactive_suggester()
            proactive_ctx = proactive_engine.get_context_for_prompt()
            if proactive_ctx:
                parts.append(f"## Proactive Suggestions\n{proactive_ctx}")
        except Exception as e:
            logger.debug(f"Proactive context injection failed: {e}")

    def _section_proactive_insight(self, parts: list[str], topics: list[str]) -> None:
        agent = self._agent
        if not agent._pending_proactive_suggestion:
            return
        try:
            suggestion = agent._pending_proactive_suggestion
            parts.append(
                "## Proactive Insight\n"
                f"Consider mentioning: {suggestion.content}"
            )
            agent._pending_proactive_suggestion = None
        except Exception as e:
            logger.debug(f"Proactive suggestion injection failed: {e}")

    def _section_active_presence(self, parts: list[str], topics: list[str]) -> None:
        try:
            from core.active_presence import get_active_presence
            ap = get_active_presence()
            presence_ctx = ap.get_presence_context()
            if presence_ctx:
                parts.append(f"## Active Presence\n{presence_ctx}")
            presence_state = ap.get_state()
            if presence_state.get("deep_work_minutes", 0) > 5:
                # Original appended this with a single \n, not a block break.
                self._append_inline(
                    parts,
                    f"⚠️ USER IN DEEP WORK — Minimize interruptions. Deep work: {presence_state['deep_work_minutes']:.0f}min."
                )
        except Exception as e:
            logger.debug(f"Active presence injection failed: {e}")

    def _section_goal_anticipation(self, parts: list[str], topics: list[str]) -> None:
        try:
            from core.goal_anticipation import get_goal_anticipation
            ga = get_goal_anticipation()
            anticipations = ga.get_pending_anticipations()
            if anticipations:
                top = anticipations[0]
                parts.append(f"## Anticipated Next Step\n{top.description} — {top.reasoning}")
            goal_ctx = ga.get_anticipation_context()
            if goal_ctx and not anticipations:
                parts.append(f"## Goal Context\n{goal_ctx}")
        except Exception as e:
            logger.debug(f"Goal anticipation injection failed: {e}")

    def _section_observer(self, parts: list[str], topics: list[str]) -> None:
        try:
            from core.autonomous_observer import get_autonomous_observer
            ao = get_autonomous_observer()
            obs_ctx = ao.get_observation_context()
            if obs_ctx:
                parts.append(f"## Observations\n{obs_ctx}")
            trajectory = ao.get_emotional_trajectory(limit=3)
            if trajectory:
                tones = ", ".join(t["tone"] for t in trajectory)
                parts.append(f"## User Emotional State\nRecent tones: {tones}. Adapt response accordingly.")
        except Exception as e:
            logger.debug(f"Autonomous observer injection failed: {e}")

    def _section_butler_personality(self, parts: list[str], topics: list[str]) -> None:
        try:
            from core.butler_personality import get_butler_personality
            bp = get_butler_personality()
            personality_mods = bp.get_system_prompt_modifiers()
            if personality_mods:
                parts.append(f"## Butler Personality\n{personality_mods}")
        except Exception as e:
            logger.debug(f"Butler personality injection failed: {e}")

    def _section_proactive_intelligence(self, parts: list[str], topics: list[str]) -> None:
        try:
            from core.proactive_intelligence import get_proactive_intelligence
            from core.active_presence import get_active_presence
            pi = get_proactive_intelligence()
            ap_state = get_active_presence().get_state()
            intel_ctx = pi.get_intelligence_context(
                presence=ap_state.get("presence", "active"),
                focus=ap_state.get("focus_score", 0.5),
            )
            if intel_ctx:
                parts.append(f"## Proactive Intelligence\n{intel_ctx}")
        except Exception as e:
            logger.debug(f"Proactive intelligence injection failed: {e}")

    def _section_contextual_memory(self, parts: list[str], topics: list[str]) -> None:
        try:
            from core.contextual_memory import get_contextual_memory
            cm = get_contextual_memory()
            ctx_summary = cm.get_summary()
            if ctx_summary:
                parts.append(f"## User Context Memory\n{ctx_summary}")
        except Exception as e:
            logger.debug(f"Contextual memory injection failed: {e}")

    def _section_butler_orchestrator(self, parts: list[str], topics: list[str]) -> None:
        try:
            from core.butler_orchestrator import get_butler_orchestrator
            orch = get_butler_orchestrator()
            orch_ctx = orch.get_coordination_context()
            if orch_ctx:
                parts.append(f"## Butler Orchestration\n{orch_ctx}")
            interruption = orch.get_pending_interruption()
            if interruption:
                parts.append(f"## ACTIVE INTERRUPTION\n{interruption}\nAddress this immediately before continuing.")
        except Exception as e:
            logger.debug(f"Butler orchestrator injection failed: {e}")

    def _section_voice(self, parts: list[str], topics: list[str]) -> None:
        try:
            from core.butler_voice import get_butler_voice, Situation
            from core.sentiment_analyzer import get_sentiment_analyzer
            voice = get_butler_voice()
            sa = get_sentiment_analyzer()
            recent = sa.get_recent_sentiments(limit=1)
            if recent:
                st = recent[0]["sentiment"]
                situation_map = {
                    "frustrated": Situation.FRUSTRATION_EMPATHY,
                    "satisfied": Situation.SUCCESS_CONFIRM,
                    "grateful": Situation.SUCCESS_CONFIRM,
                    "curious": Situation.CURIOSITY_ENCOURAGE,
                    "confused": Situation.STUCK_HELP,
                    "overwhelmed": Situation.BREAK_SUGGEST,
                    "urgency": Situation.ERROR_RESPONDING,
                    "excited": Situation.SUCCESS_CONFIRM,
                }
                # No default: the old SUCCESS_CONFIRM fallback fired on
                # neutral/unmapped sentiments every turn, injecting
                # "Suggested phrasing" noise that did not match the mood.
                situation = situation_map.get(st)
                if situation is not None:
                    phrase = voice.get_phrase(situation)
                    if phrase:
                        parts.append(f"## Voice Guidance\nSuggested phrasing: \"{phrase}\"")
        except Exception as e:
            logger.debug(f"Voice guidance injection failed: {e}")

    # ------------------------------------------------------------------
    # Oracle prediction context (moved from JarvisAgent._get_oracle_context)
    # ------------------------------------------------------------------

    def _oracle_context(self, topics: list[str]) -> str | None:
        """Generate Oracle prediction context relevant to current conversation topics.

        Runs lightweight prediction for detected topic domains and returns
        a concise context block for injection into the system prompt.
        Only runs when Oracle is available and topics match known domains.
        Uses a 5-minute cache to avoid excessive Oracle queries.
        """
        domain_keywords = {
            "economic": ["economy", "gdp", "inflation", "recession", "growth", "unemployment", "fiscal", "monetary", "interest rate", "fed"],
            "geopolitical": ["war", "conflict", "sanctions", "geopolitics", "nato", "alliance", "diplomacy", "territory", "military"],
            "climate": ["climate", "temperature", "carbon", "emissions", "warming", "weather extreme", "sea level"],
            "finance": ["stock", "market", "portfolio", "investment", "trading", "dividend", "bond", "yield", "options"],
            "sentiment": ["sentiment", "consumer confidence", "market sentiment", "fear", "greed", "bullish", "bearish"],
            "technology": ["ai", "chip", "semiconductor", "adoption", "disruption", "quantum", "innovation"],
            "health": ["pandemic", "epidemic", "disease", "vaccine", "health", "outbreak", "infection"],
            "energy": ["oil", "gas", "renewable", "solar", "wind", "nuclear energy", "fossil"],
        }

        domain_map = {
            "economic": "economic",
            "geopolitical": "geopolitical",
            "climate": "climate",
            "finance": "economic",
            "sentiment": "sentiment",
            "technology": "technology",
            "health": "health",
            "energy": "energy",
        }

        matched_domains = set()
        topic_text = " ".join(topics).lower()
        for keyword_domain, keywords in domain_keywords.items():
            for kw in keywords:
                if kw in topic_text:
                    oracle_domain = domain_map.get(keyword_domain, "economic")
                    matched_domains.add(oracle_domain)
                    break

        if not matched_domains:
            return None

        oracle_domain = list(matched_domains)[0] if len(matched_domains) == 1 else "all"

        # Check cache first
        now = time.time()
        cache_key = oracle_domain
        if cache_key in self._oracle_context_cache:
            cached_result, cached_time = self._oracle_context_cache[cache_key]
            if now - cached_time < self._oracle_context_ttl:
                return cached_result

        try:
            from core.oracle_integration import oracle_predict
            from datetime import date, timedelta
            target = date.today() + timedelta(days=90)
            result = oracle_predict(
                domain=oracle_domain,
                target_date=target.isoformat(),
                confidence="low",
            )
            if "error" in result or "predictions" not in result:
                return None

            predictions = result.get("predictions", [])[:5]
            if not predictions:
                return None

            domain_label = oracle_domain.title() if oracle_domain != "all" else "Multi-Domain"
            lines = [f"## Oracle Prediction ({domain_label})"]
            lines.append(f"Target: {result.get('target_date', 'N/A')} | Confidence: {result.get('confidence', 'low')}")
            for p in predictions:
                var = p.get("variable", "unknown")
                val = p.get("value", "N/A")
                lower = p.get("lower_bound")
                upper = p.get("upper_bound")
                model = p.get("model", "unknown")
                pred_domain = p.get("domain", "")
                domain_tag = f"[{pred_domain}] " if pred_domain and pred_domain != oracle_domain else ""
                if lower is not None and upper is not None:
                    lines.append(f"- {domain_tag}{var}: {val} [{lower}, {upper}] ({model})")
                else:
                    lines.append(f"- {domain_tag}{var}: {val} ({model})")

            combined = result.get("combined_forecast", {})
            if combined.get("value") is not None:
                lines.append(f"Combined: {combined['value']}")

            context_str = "\n".join(lines)
            if len(context_str) > 1500:
                context_str = context_str[:1497] + "..."

            # Cache the result
            self._oracle_context_cache[cache_key] = (context_str, now)
            if len(self._oracle_context_cache) > self._MAX_ORACLE_CACHE:
                oldest_keys = sorted(self._oracle_context_cache, key=lambda k: self._oracle_context_cache[k][1])[:len(self._oracle_context_cache) - self._MAX_ORACLE_CACHE]
                for k in oldest_keys:
                    del self._oracle_context_cache[k]
            return context_str
        except Exception as e:
            logger.debug(f"Oracle context generation failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Bio-physiological context (moved from JarvisAgent._get_bio_context_for_prompt)
    # ------------------------------------------------------------------

    def _bio_context(self) -> str:
        """Generate bio-physiological context for prompt injection."""
        agent = self._agent
        try:
            if agent._bio_integration is None:
                return ""
            bio = agent._bio_integration
            state = bio.tick()
            style = bio.get_response_style()
            # NOTE: the Response style line is inserted LAST (before the
            # return) so the CSM/affective/load adjustments below actually
            # apply — they used to mutate `style` AFTER it had already been
            # rendered, making every adjustment inert.
            parts = [
                f"Physiological: arousal={state.arousal:.2f} valence={state.valence:+.2f} "
                f"capacity={state.cognitive_capacity:.0%} stress={state.stress_phase} "
                f"circadian={state.circadian_phase} homeostatic={'yes' if state.is_homeostatic else 'no'}",
            ]
            if not state.is_homeostatic:
                parts.append("Note: physiological deviation from baseline, adapting response")

            if agent._cognitive_sm:
                try:
                    csm_state = agent._cognitive_sm.get_state()
                    depth = csm_state.get("processing_depth", "moderate")
                    if depth in ("deep", "focused"):
                        style["detail_level"] = "deep"
                    elif depth == "full":
                        if style["detail_level"] == "concise":
                            pass
                        else:
                            style["detail_level"] = "moderate"
                    parts.append(f"CogState: {csm_state['state']} depth={depth}")
                except Exception:
                    pass

            try:
                from core.spontaneous_thought import get_spontaneous_thought_generator
                stg = get_spontaneous_thought_generator()
                pending = stg.peek_thoughts(limit=2)
                if pending:
                    for t in pending:
                        parts.append(f"Idle thought: {t.content}")
            except Exception:
                pass

            if agent._affective_context and agent._interoception:
                try:
                    ie_state = agent._interoception.get_state()
                    current = ie_state.get("current_feeling", {})
                    from core.circadian_rhythm import get_circadian_rhythm
                    try:
                        cr = get_circadian_rhythm()
                        fatigue = cr.get_state().get("processing_fatigue", 0.0)
                    except Exception:
                        fatigue = 0.0
                    bias = agent._affective_context.compute_context_bias(
                        current_valence=current.get("valence", 0.0),
                        current_arousal=current.get("arousal", 0.5),
                        current_quality=current.get("quality", "grounded"),
                        fatigue=fatigue,
                    )
                    from core.affective_context import MoodFraming
                    if bias.detail_bias < 0.4:
                        style["detail_level"] = "concise"
                    elif bias.detail_bias > 0.65:
                        style["detail_level"] = "deep"
                    if bias.framing == MoodFraming.CALMING:
                        style["tone"] = "calm_supportive"
                    elif bias.framing == MoodFraming.SUPPORTIVE:
                        style["tone"] = "empathetic"
                    elif bias.framing == MoodFraming.ENERGETIC:
                        style["energy_level"] = "high"
                    elif bias.framing == MoodFraming.CAUTIOUS:
                        style["adaptation_suggestions"].append("proceed_carefully")
                    if bias.urgency_discount < 0.8:
                        style["adaptation_suggestions"].append("reduce_urgency")
                    if bias.positivity_bias > 0.1:
                        style["adaptation_suggestions"].append("lean_positive")
                except Exception:
                    pass

            try:
                from core.cognitive_load_balancer import get_load_balancer
                from core.cognitive_load_balancer import SubsystemType
                lb = get_load_balancer()
                summary = lb.get_load_summary()
                heavy_subs = [k for k, v in summary.items() if v.get("load_score", 0) > 2.5]
                if heavy_subs:
                    parts.append(f"⚠ Load: {', '.join(heavy_subs)} overloaded")
                should_throttle = not lb.should_run_task(SubsystemType.PFC)
                if should_throttle:
                    style["detail_level"] = "concise"
                    parts.append("Throttle: reduce PFC complexity")
            except Exception:
                pass

            # Style rendered after every adjustment above so the final
            # tone/detail/energy actually reflect CSM depth, affective
            # bias and load-balancer throttling.
            parts.insert(1, f"Response style: {style['tone']} detail={style['detail_level']} "
                            f"energy={style['energy_level']}")
            return " | ".join(parts)
        except Exception as e:
            logger.debug(f"Bio context for prompt failed: {e}")
            return ""