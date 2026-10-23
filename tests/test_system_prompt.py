"""Regression tests for the system-prompt extraction (core/system_prompt.py).

`JarvisAgent.get_system_prompt()` was a 586-line monolith assembling ~30
context sections inline. The extraction moved assembly into
`SystemPromptBuilder` while the agent keeps cache + identity concerns. These
tests pin the behavioural contract:

  1. Full degradation: an agent whose subsystems are all absent yields
     EXACTLY the base prompt — one failing subsystem can never corrupt the
     whole prompt.
  2. Section order: the prompt is assembled top-down in the original order.
  3. Mid-group appends: when a multi-block section fails halfway, the blocks
     already appended survive (matching the original's incremental appends).
  4. The single-\\n deep-work warning attaches inline, not as a new block.
  5. Proactive Insight is consume-once.
  6. The Oracle prediction cache prevents re-querying within TTL.

All subsystem singletons are neutralised with NullSingleton so tests are
hermetic — no real subsystem state leaks into assertions.
"""
import concurrent.futures
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class NullSingleton:
    """Any attribute is a method returning None — a hermetic stand-in for any
    subsystem singleton."""

    def __getattr__(self, name):
        def _method(*args, **kwargs):
            return None
        return _method

    def __call__(self, *args, **kwargs):
        return None


def _neutralize_subsystems(monkeypatch):
    """Point every subsystem getter the builder touches at a NullSingleton."""
    # Module-level imports inside core.system_prompt
    for name in (
        "get_persistent_memory",
        "get_emotional_state_manager",
        "get_user_model_manager",
        "get_goal_tracker",
        "get_proactive_suggester",
        "get_execution_context",
        "get_system_awareness",
    ):
        monkeypatch.setattr(f"core.system_prompt.{name}", lambda: NullSingleton())
    # Class singletons used by cognitive_alignment_context
    for name in (
        "core.cognitive_psychology.CognitivePsychologyEngine.get_instance",
        "core.machine_alignment.MachineAlignmentEngine.get_instance",
        "core.neuroscience.NeuroscienceCognitiveArchitecture.get_instance",
    ):
        monkeypatch.setattr(name, lambda: NullSingleton())
    # Lazy imports inside sections — patch at their home modules
    for name in (
        "core.circadian_rhythm.get_circadian_rhythm",
        "core.autonomous_goals.get_autonomous_goals",
        "core.active_presence.get_active_presence",
        "core.goal_anticipation.get_goal_anticipation",
        "core.autonomous_observer.get_autonomous_observer",
        "core.butler_personality.get_butler_personality",
        "core.proactive_intelligence.get_proactive_intelligence",
        "core.contextual_memory.get_contextual_memory",
        "core.butler_orchestrator.get_butler_orchestrator",
        "core.butler_voice.get_butler_voice",
        "core.sentiment_analyzer.get_sentiment_analyzer",
    ):
        monkeypatch.setattr(name, lambda: NullSingleton())
    # Dynamic-state sections added 2026-09: task list + background jobs.
    # They read global singletons that other tests may have populated, so
    # the degradation contract needs them explicitly empty.
    monkeypatch.setattr("core.task_list.get_task_list", lambda: _EmptyTasks())
    monkeypatch.setattr("tools.terminal.get_jarvis_computer", lambda: _EmptyJobs())


class _EmptyTasks:
    def to_prompt(self) -> str:
        return ""


class _EmptyJobs:
    def list_background_jobs(self) -> list:
        return []


class FakeAgent:
    """Duck-typed minimal agent: every handle absent, like a stripped-down
    or partially-initialised JarvisAgent."""

    def __init__(self, **overrides):
        self.unified_memory = None
        self._live_state = None
        self._narrative_thread = None
        self._three_layer = None
        self._adaptive_window = None
        self._oracle_bridge = None
        self._bio_integration = None
        self._cognitive_sm = None
        self._predictive_processor = None
        self._affective_context = None
        self._interoception = None
        self._global_workspace = None
        self._recursive_self_model = None
        self._stream_consciousness = None
        self._emotion_regulation = None
        self._allostasis_predictor = None
        self._emotional_memory = None
        self._pending_attention_items = []
        self._pending_proactive_suggestion = None
        self.memory = None
        self._context_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        for key, value in overrides.items():
            setattr(self, key, value)

    def close(self):
        self._context_pool.shutdown(wait=False)


class TestDegradationContract:
    def test_fully_degraded_agent_yields_exactly_the_base_prompt(self, monkeypatch):
        from core.system_prompt import SystemPromptBuilder

        _neutralize_subsystems(monkeypatch)
        agent = FakeAgent()
        try:
            builder = SystemPromptBuilder(agent)
            result = builder.build("BASE-PROMPT", [])
        finally:
            agent.close()

        assert result == "BASE-PROMPT"

    def test_neutralized_builder_survives_topics(self, monkeypatch):
        """Topics alone must not break anything when Oracle is absent."""
        from core.system_prompt import SystemPromptBuilder

        monkeypatch.setattr(
            "core.oracle_integration.oracle_predict",
            lambda **kw: {"error": "oracle unavailable"},
        )
        _neutralize_subsystems(monkeypatch)
        agent = FakeAgent()
        try:
            builder = SystemPromptBuilder(agent)
            result = builder.build("BASE", ["inflation", "interest rate"])
        finally:
            agent.close()

        # The oracle section degrades to the generation-failure debug path.
        assert result == "BASE"


class TestSectionOrder:
    def test_declared_order_matches_the_original_append_order(self):
        from core.system_prompt import SystemPromptBuilder

        assert SystemPromptBuilder._SECTION_ORDER == (
            "three_layer", "adaptive", "execution_context",
            "tasks",  # deliberate 2026-09 addition: agent task list
            "jobs",   # deliberate 2026-09 addition: background jobs
            "world_model",
            "oracle_prediction",
            "cognitive_alignment", "physiological", "consciousness",
            "emotion_regulation", "emotional_memory", "affective", "oracle_forecasts",
            "predictive", "attention_items", "emotional_state", "response_style",
            "user_model", "goals", "autonomous_goals", "proactive", "proactive_insight",
            "active_presence", "goal_anticipation", "observer", "butler_personality",
            "proactive_intelligence", "contextual_memory", "butler_orchestrator",
            "voice",
        )

    def test_blocks_appear_in_order(self, monkeypatch):
        from core.system_prompt import SystemPromptBuilder

        _neutralize_subsystems(monkeypatch)
        agent = FakeAgent(
            _three_layer=SimpleNamespace(
                get_l2_context_for_model=lambda model: "warm session memory",
                retrieve_from_l3=lambda kind, max_results=2: [],
            ),
            _adaptive_window=SimpleNamespace(
                assemble_context=lambda: "x" * 20,
            ),
        )
        try:
            builder = SystemPromptBuilder(agent)
            result = builder.build("BASE", [])
        finally:
            agent.close()

        l2_pos = result.index("## Session Memory (L2)")
        adaptive_pos = result.index("## Adaptive Context")
        assert result.startswith("BASE")
        assert 0 < l2_pos < adaptive_pos


class TestMidGroupFailureSemantics:
    def test_partially_failed_section_keeps_earlier_blocks(self, monkeypatch):
        """The original appended blocks inside one try as it went: a failure
        midway left earlier blocks in place. The extraction must preserve
        that, not lose the whole group."""
        from core.system_prompt import SystemPromptBuilder

        class FlakyEmotionalManager:
            def get_emotional_context(self):
                return "user seems tired"

            def should_show_empathy(self):
                raise RuntimeError("boom")

        monkeypatch.setattr(
            "core.system_prompt.get_emotional_state_manager",
            lambda: FlakyEmotionalManager(),
        )
        _neutralize_subsystems(monkeypatch)  # everything else stays quiet
        monkeypatch.setattr(
            "core.system_prompt.get_emotional_state_manager",
            lambda: FlakyEmotionalManager(),
        )

        agent = FakeAgent()
        try:
            builder = SystemPromptBuilder(agent)
            parts: list[str] = []
            builder._section_emotional_state(parts, [])
        finally:
            agent.close()

        assert parts == ["## Emotional Context\nuser seems tired"]


class TestProactiveInsightConsumeOnce:
    def test_suggestion_injected_once_and_cleared(self):
        from core.system_prompt import SystemPromptBuilder

        agent = FakeAgent(
            _pending_proactive_suggestion=SimpleNamespace(content="the new tool"),
        )
        builder = SystemPromptBuilder(agent)

        parts: list[str] = []
        builder._section_proactive_insight(parts, [])
        assert parts == ["## Proactive Insight\nConsider mentioning: the new tool"]
        assert agent._pending_proactive_suggestion is None

        # Second run: nothing left to consume.
        parts2: list[str] = []
        builder._section_proactive_insight(parts2, [])
        assert parts2 == []


class TestDeepWorkInlineWarning:
    def test_warning_attaches_with_single_newline(self, monkeypatch):
        """The deep-work warning was appended with '\\n', not the usual block
        separator — it must land at the end of the last block, not form a new
        one."""
        from core.system_prompt import SystemPromptBuilder

        presence = SimpleNamespace(
            get_presence_context=lambda: "at desk",
            get_state=lambda: {"deep_work_minutes": 42},
        )
        monkeypatch.setattr(
            "core.active_presence.get_active_presence", lambda: presence
        )
        _neutralize_subsystems(monkeypatch)
        monkeypatch.setattr(
            "core.active_presence.get_active_presence", lambda: presence
        )

        agent = FakeAgent()
        try:
            builder = SystemPromptBuilder(agent)
            result = builder.build("BASE", [])
        finally:
            agent.close()

        assert result.endswith(
            "## Active Presence\nat desk\n"
            "⚠️ USER IN DEEP WORK — Minimize interruptions. Deep work: 42min."
        )
        # Single \n, not a \n\n block break, before the warning.
        assert "\n⚠️ USER IN DEEP WORK" in result
        assert "\n\n⚠️ USER IN DEEP WORK" not in result

    def test_no_warning_below_threshold(self, monkeypatch):
        from core.system_prompt import SystemPromptBuilder

        presence = SimpleNamespace(
            get_presence_context=lambda: "at desk",
            get_state=lambda: {"deep_work_minutes": 3},
        )
        _neutralize_subsystems(monkeypatch)
        monkeypatch.setattr(
            "core.active_presence.get_active_presence", lambda: presence
        )

        agent = FakeAgent()
        try:
            builder = SystemPromptBuilder(agent)
            result = builder.build("BASE", [])
        finally:
            agent.close()

        assert result.endswith("## Active Presence\nat desk")
        assert "DEEP WORK" not in result


class TestOracleContextCache:
    def _patch_predict(self, monkeypatch, result):
        calls = []

        def fake_predict(**kwargs):
            calls.append(kwargs)
            return result

        monkeypatch.setattr("core.oracle_integration.oracle_predict", fake_predict)
        return calls

    def test_topic_matching_and_cache(self, monkeypatch):
        from core.system_prompt import SystemPromptBuilder

        self._patch_predict(monkeypatch, {
            "predictions": [{"variable": "gdp", "value": "3%", "model": "test"}],
            "target_date": "2026-11-28",
            "confidence": "low",
        })

        agent = FakeAgent()
        try:
            builder = SystemPromptBuilder(agent)
            first = builder._oracle_context(["inflation", "interest rate"])
            second = builder._oracle_context(["inflation", "gdp"])
        finally:
            agent.close()

        assert first is not None
        assert "## Oracle Prediction (Economic)" in first
        assert "gdp: 3% (test)" in first
        assert second == first  # cached

    def test_no_match_yields_none(self, monkeypatch):
        from core.system_prompt import SystemPromptBuilder

        calls = self._patch_predict(monkeypatch, {})
        agent = FakeAgent()
        try:
            builder = SystemPromptBuilder(agent)
            assert builder._oracle_context(["cooking pasta"]) is None
        finally:
            agent.close()
        assert calls == []

    def test_error_response_yields_none_and_is_not_cached(self, monkeypatch):
        from core.system_prompt import SystemPromptBuilder

        self._patch_predict(monkeypatch, {"error": "oracle unavailable"})
        agent = FakeAgent()
        try:
            builder = SystemPromptBuilder(agent)
            assert builder._oracle_context(["inflation"]) is None
            # Nothing cached under the key — a retry is possible.
            assert "economic" not in builder._oracle_context_cache
        finally:
            agent.close()


class TestParallelContexts:
    def test_live_state_context_flows_through(self, monkeypatch):
        from core.system_prompt import SystemPromptBuilder

        _neutralize_subsystems(monkeypatch)

        updates = []
        live_state = SimpleNamespace(
            update_from_agent=lambda agent: updates.append("agent"),
            update_from_subsystems=lambda: updates.append("subsystems"),
            update_discoveries=lambda: updates.append("discoveries"),
            get_context_string=lambda: "live context snapshot",
        )
        agent = FakeAgent(_live_state=live_state)
        try:
            builder = SystemPromptBuilder(agent)
            result = builder.build("BASE", [])
        finally:
            agent.close()

        assert "live context snapshot" in result
        assert updates == ["agent", "subsystems", "discoveries"]


class TestAgentMethodDelegation:
    """The thin agent method: cache, identity refresh, base selection."""

    @staticmethod
    def _bare_agent():
        from core.agent import JarvisAgent

        agent = JarvisAgent.__new__(JarvisAgent)
        agent._tools_lock = threading.Lock()
        agent._system_prompt = None
        agent._system_prompt_time = None
        agent._system_prompt_ttl = 300.0
        agent._last_l2_version = 0
        agent._three_layer = None
        agent.self_awareness = None
        agent.prompt_manager = None
        agent.topic_tracker = None
        agent._session_id = "test-session"
        agent._turn_count = 0
        agent._tools_used_this_session = []
        agent.memory = None
        agent.tools = {}
        agent._prompt_builder = None
        return agent

    def test_fallback_base_prompt_and_builder_delegation(self, monkeypatch):
        captured = {}

        def fake_build(self, base_prompt, topics):
            captured["base"] = base_prompt
            captured["topics"] = topics
            return f"BUILT({base_prompt})"

        monkeypatch.setattr(
            "core.system_prompt.SystemPromptBuilder.build", fake_build
        )

        agent = self._bare_agent()
        result = agent.get_system_prompt()

        assert result == "BUILT(You are Jarvis, an AI assistant.)"
        assert captured["topics"] == []

    def test_real_base_prompt_and_topics_flow_through(self, monkeypatch):
        captured = {}

        def fake_build(self, base_prompt, topics):
            captured["base"] = base_prompt
            captured["topics"] = topics
            return "BUILT"

        monkeypatch.setattr(
            "core.system_prompt.SystemPromptBuilder.build", fake_build
        )

        class FakeSelfAwareness:
            def refresh_environment(self):
                captured["refreshed"] = True

            def update_session_state(self, **kwargs):
                captured["session_state"] = kwargs

        class FakePromptManager:
            def get_base_prompt(self):
                return "MANAGED-BASE"

        class FakeTopicTracker:
            def get_top_topics(self):
                return ["ai", "inflation"]

        class FakeMemory:
            message_count = 7

        agent = self._bare_agent()
        agent.self_awareness = FakeSelfAwareness()
        agent.prompt_manager = FakePromptManager()
        agent.topic_tracker = FakeTopicTracker()
        agent.memory = FakeMemory()

        result = agent.get_system_prompt()

        assert result == "BUILT"
        assert captured["base"] == "MANAGED-BASE"
        assert captured["topics"] == ["ai", "inflation"]
        assert captured["refreshed"] is True
        assert captured["session_state"]["session_id"] == "test-session"
        assert captured["session_state"]["messages_count"] == 7

    def test_prompt_cached_within_ttl(self, monkeypatch):
        import time as _time

        monkeypatch.setattr(
            "core.system_prompt.SystemPromptBuilder.build",
            lambda self, base, topics: "BUILT",
        )

        agent = self._bare_agent()
        first = agent.get_system_prompt()
        assert first == "BUILT"

        # Second call within TTL: cache hit, no rebuild.
        assert agent.get_system_prompt() is first
        assert agent._system_prompt_time > 0

    def test_prompt_rebuilt_after_ttl(self, monkeypatch):
        import time as _time

        builds = []

        def fake_build(self, base, topics):
            builds.append(len(builds))
            return f"BUILT-{len(builds)}"

        monkeypatch.setattr(
            "core.system_prompt.SystemPromptBuilder.build", fake_build
        )

        agent = self._bare_agent()
        assert agent.get_system_prompt() == "BUILT-1"
        # Age the cache past the TTL.
        agent._system_prompt_time = _time.time() - 400.0
        assert agent.get_system_prompt() == "BUILT-2"

    def test_l2_version_change_invalidates_cache(self, monkeypatch):
        """Three-layer version semantics: the first build records nothing;
        the first cache-CHECK records the version and rebuilds if it changed;
        subsequent checks with an unchanged version are cache hits."""
        builds = []

        def fake_build(self, base, topics):
            builds.append(len(builds))
            return "BUILT"

        monkeypatch.setattr(
            "core.system_prompt.SystemPromptBuilder.build", fake_build
        )

        class FakeThreeLayer:
            def __init__(self):
                self.version = 1

            def get_version(self):
                return self.version

        agent = self._bare_agent()
        agent._three_layer = FakeThreeLayer()

        # First build: no cache; _last_l2_version untouched (still 0).
        agent.get_system_prompt()
        assert len(builds) == 1
        assert agent._last_l2_version == 0

        # First cache check: version 1 != 0 -> rebuild, version recorded.
        agent.get_system_prompt()
        assert len(builds) == 2
        assert agent._last_l2_version == 1

        # Unchanged version: pure cache hit, no rebuild.
        agent.get_system_prompt()
        assert len(builds) == 2

        # New version: invalidates the cache on the next check.
        agent._three_layer.version = 2
        agent.get_system_prompt()
        assert len(builds) == 3
        assert agent._last_l2_version == 2

    def test_builder_is_reused_across_calls(self, monkeypatch):
        monkeypatch.setattr(
            "core.system_prompt.SystemPromptBuilder.build",
            lambda self, base, topics: "BUILT",
        )

        agent = self._bare_agent()
        b1 = agent._get_prompt_builder()
        b2 = agent._get_prompt_builder()
        assert b1 is b2

        # The builder must survive get_system_prompt calls (oracle cache).
        agent.get_system_prompt()
        assert agent._get_prompt_builder() is b1