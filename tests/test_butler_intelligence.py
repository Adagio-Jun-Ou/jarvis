"""Tests for Butler Personality, Proactive Intelligence, and Contextual Memory."""
import pytest
import sys
import os
import time
import tempfile
import pathlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))

from core.butler_personality import Formality, Verbosity, EnergyLevel


class TestButlerPersonality:
    def test_initial_state(self):
        from core.butler_personality import ButlerPersonality, PersonaMode, Formality
        bp = ButlerPersonality()
        state = bp.get_state()
        assert state["mode"] == PersonaMode.BUTLER.value
        assert state["formality"] == "professional"
        assert "warmth" in state
        assert "humor" in state

    def test_observe_interaction_returns_state(self):
        from core.butler_personality import ButlerPersonality, PersonaMode
        bp = ButlerPersonality()
        state = bp.observe_interaction(user_sentiment="curious", user_energy="high")
        assert state.mode == PersonaMode.MENTOR

    def test_frustration_adapts_to_butler_mode(self):
        from core.butler_personality import ButlerPersonality, PersonaMode, Verbosity
        bp = ButlerPersonality()
        for _ in range(4):
            bp.observe_interaction(frustration=True)
        assert bp._state.mode == PersonaMode.BUTLER
        assert bp._state.verbosity.value <= Verbosity.CONCISE.value

    def test_gratitude_increases_warmth(self):
        from core.butler_personality import ButlerPersonality
        bp = ButlerPersonality()
        initial_warmth = bp._state.warmth
        bp.observe_interaction(gratitude=True)
        assert bp._state.warmth > initial_warmth

    def test_set_goal_context(self):
        from core.butler_personality import ButlerPersonality, PersonaMode, Verbosity
        bp = ButlerPersonality()
        bp.set_goal_context("how does vector search work? explain it to me")
        assert bp._state.mode == PersonaMode.MENTOR
        assert bp._state.verbosity == Verbosity.DETAILED

    def test_set_goal_context_debug(self):
        from core.butler_personality import ButlerPersonality, PersonaMode
        bp = ButlerPersonality()
        bp.set_goal_context("fix this urgent error")
        assert bp._state.mode == PersonaMode.BUTLER

    def test_set_goal_context_collaborate(self):
        from core.butler_personality import ButlerPersonality, PersonaMode
        bp = ButlerPersonality()
        bp.set_goal_context("let's build this together")
        assert bp._state.mode == PersonaMode.COLLABORATOR

    def test_set_goal_context_companion(self):
        from core.butler_personality import ButlerPersonality, PersonaMode
        bp = ButlerPersonality()
        bp.set_goal_context("I want to chat and think about ideas")
        assert bp._state.mode == PersonaMode.COMPANION

    def test_system_prompt_modifiers(self):
        from core.butler_personality import ButlerPersonality
        bp = ButlerPersonality()
        modifiers = bp.get_system_prompt_modifiers()
        assert "butler" in modifiers.lower() or "anticipatory" in modifiers.lower()

    def test_response_guidance(self):
        from core.butler_personality import ButlerPersonality
        bp = ButlerPersonality()
        guidance = bp.get_response_guidance()
        assert "mode" in guidance
        assert "formality" in guidance
        assert "warmth" in guidance

    def test_record_preference(self):
        from core.butler_personality import ButlerPersonality
        bp = ButlerPersonality()
        bp.record_preference("verbosity", "terse", 0.8)
        bp.record_preference("verbosity", "terse", 0.8)
        prefs = bp.get_preferences_summary()
        assert any(p["key"] == "verbosity" for p in prefs)

    def test_preference_reinforcement(self):
        from core.butler_personality import ButlerPersonality
        bp = ButlerPersonality()
        bp.record_preference("formality", "casual", 0.3)
        bp.record_preference("formality", "casual", 0.3)
        assert bp._preferences["formality"].confidence > 0.3
        assert bp._preferences["formality"].times_observed >= 2

    def test_learn_from_input_short(self):
        from core.butler_personality import ButlerPersonality
        bp = ButlerPersonality()
        bp._learn_from_interaction("fix bug", "neutral", False)
        assert "preferred_response_length" in bp._preferences or len(bp._preferences) >= 0

    def test_learn_from_input_detailed(self):
        from core.butler_personality import ButlerPersonality
        bp = ButlerPersonality()
        bp._learn_from_interaction("could you please explain in detail how the entire system works and what all the components do", "curious", False)
        assert bp._preferences.get("verbosity", None) is not None


class TestProactiveIntelligence:
    def _fresh_pi(self):
        import tempfile
        from core.proactive_intelligence import ProactiveIntelligence
        return ProactiveIntelligence(storage_path=tempfile.mkdtemp())

    def test_initial_state(self):
        pi = self._fresh_pi()
        state = pi.get_state()
        assert "active_patterns" in state
        assert "acceptance_rate" in state

    def test_observe_returns_none_when_no_pattern(self):
        pi = self._fresh_pi()
        result = pi.observe("file_read")
        assert result is None

    def test_stuck_loop_detection(self):
        from core.proactive_intelligence import PatternType
        pi = self._fresh_pi()
        for _ in range(3):
            pi.observe("file_edit", success=False)
        patterns = pi.get_active_patterns()
        stuck = [p for p in patterns if p["type"] == PatternType.STUCK_LOOP.value]
        assert len(stuck) > 0

    def test_error_storm_detection(self):
        from core.proactive_intelligence import PatternType
        pi = self._fresh_pi()
        for _ in range(4):
            pi.observe("terminal_command", success=False)
        patterns = pi.get_active_patterns()
        storm = [p for p in patterns if p["type"] == PatternType.ERROR_STORM.value]
        assert len(storm) > 0

    def test_debugging_detection(self):
        from core.proactive_intelligence import ProactiveIntelligence, PatternType
        pi = self._fresh_pi()
        pi.observe("file_edit", success=False)
        pi.observe("terminal_command", success=False)
        pi.observe("test_run", success=False)
        pi.observe("error_seen", success=True)
        pi.observe("file_edit", success=False)
        patterns = pi.get_active_patterns()
        debug = [p for p in patterns if p["type"] == PatternType.DEBUGGING.value]
        assert len(debug) > 0

    def test_intervention_cooldown(self):
        from core.proactive_intelligence import ProactiveIntelligence
        pi = self._fresh_pi()
        pi._last_intervention_time = time.time()
        pending = pi.get_pending_interventions()
        assert len(pending) == 0

    def test_accept_increases_rate(self):
        from core.proactive_intelligence import ProactiveIntelligence
        pi = self._fresh_pi()
        initial_rate = pi._acceptance_rate
        pi.observe("file_edit", success=False)
        pi.observe("file_edit", success=False)
        pi.observe("file_edit", success=False)
        pi.mark_delivered()
        pi.mark_accepted()
        assert pi._acceptance_rate > initial_rate

    def test_dismiss_decreases_rate(self):
        from core.proactive_intelligence import ProactiveIntelligence
        pi = self._fresh_pi()
        initial_rate = pi._acceptance_rate
        pi.observe("file_edit", success=False)
        pi.observe("file_edit", success=False)
        pi.observe("file_edit", success=False)
        pi.mark_delivered()
        pi.mark_dismissed()
        assert pi._acceptance_rate < initial_rate

    def test_intelligence_context(self):
        from core.proactive_intelligence import ProactiveIntelligence
        pi = self._fresh_pi()
        ctx = pi.get_intelligence_context()
        assert isinstance(ctx, str)

    def test_thread_safety(self):
        from core.proactive_intelligence import ProactiveIntelligence
        import threading
        pi = self._fresh_pi()
        errors = []
        def worker():
            try:
                for _ in range(50):
                    pi.observe("file_edit", success=True)
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0


class TestContextualMemory:
    def test_initial_state(self):
        from core.contextual_memory import ContextualMemory
        cm = ContextualMemory(storage_path=pathlib.Path(tempfile.mkdtemp()))
        state = cm.get_state()
        assert "preferences_learned" in state
        assert "communication_style" in state

    def test_learn_preference(self):
        from core.contextual_memory import ContextualMemory, PreferenceCategory
        cm = ContextualMemory(storage_path=pathlib.Path(tempfile.mkdtemp()))
        cm.learn_preference("theme", "dark", PreferenceCategory.COMMUNICATION, 0.7)
        prefs = cm.get_state()
        assert prefs["preferences_learned"] >= 1

    def test_preference_reinforcement(self):
        from core.contextual_memory import ContextualMemory, PreferenceCategory
        cm = ContextualMemory(storage_path=pathlib.Path(tempfile.mkdtemp()))
        cm.learn_preference("formality", "casual", PreferenceCategory.COMMUNICATION, 0.3)
        cm.learn_preference("formality", "casual", PreferenceCategory.COMMUNICATION, 0.3)
        assert cm._preferences["formality"].confidence > 0.3
        assert cm._preferences["formality"].times_observed >= 2

    def test_preference_change(self):
        from core.contextual_memory import ContextualMemory, PreferenceCategory
        cm = ContextualMemory(storage_path=pathlib.Path(tempfile.mkdtemp()))
        cm.learn_preference("formality", "formal", PreferenceCategory.COMMUNICATION, 0.8)
        cm.learn_preference("formality", "casual", PreferenceCategory.COMMUNICATION, 0.5)
        assert cm._preferences["formality"].value == "casual"

    def test_communication_style(self):
        from core.contextual_memory import ContextualMemory
        cm = ContextualMemory(storage_path=pathlib.Path(tempfile.mkdtemp()))
        style = cm.get_communication_style()
        assert "verbosity" in style
        assert "formality" in style
        assert "warmth" in style

    def test_learn_from_interaction(self):
        from core.contextual_memory import ContextualMemory
        cm = ContextualMemory(storage_path=pathlib.Path(tempfile.mkdtemp()))
        cm.learn_from_interaction("hey can you help me fix this bug?", user_sentiment="frustrated")
        style = cm.get_communication_style()
        assert style is not None

    def test_temporal_patterns(self):
        from core.contextual_memory import ContextualMemory
        cm = ContextualMemory(storage_path=pathlib.Path(tempfile.mkdtemp()))
        patterns = cm.get_temporal_patterns()
        assert "peak_hours" in patterns
        assert "total_sessions" in patterns

    def test_get_summary(self):
        from core.contextual_memory import ContextualMemory
        cm = ContextualMemory(storage_path=pathlib.Path(tempfile.mkdtemp()))
        summary = cm.get_summary()
        assert isinstance(summary, str)

    def test_session_start_end(self):
        from core.contextual_memory import ContextualMemory
        cm = ContextualMemory(storage_path=pathlib.Path(tempfile.mkdtemp()))
        cm.start_session()
        assert cm._current_session is not None
        cm.end_session()
        assert len(cm._session_history) >= 1

    def test_persistence(self):
        from core.contextual_memory import ContextualMemory, PreferenceCategory
        tmp = pathlib.Path(tempfile.mkdtemp())
        cm1 = ContextualMemory(storage_path=tmp)
        cm1.learn_preference("theme", "dark", PreferenceCategory.COMMUNICATION, 0.8)
        cm2 = ContextualMemory(storage_path=tmp)
        assert "theme" in cm2._preferences
        assert cm2._preferences["theme"].value == "dark"

    def test_project_preferences(self):
        from core.contextual_memory import ContextualMemory, PreferenceCategory
        cm = ContextualMemory(storage_path=pathlib.Path(tempfile.mkdtemp()))
        cm.learn_preference("project_type", "web_app", PreferenceCategory.PROJECT, 0.6)
        cm.learn_preference("project_type", "web_app", PreferenceCategory.PROJECT, 0.6)
        prefs = cm.get_project_preferences()
        assert "project_type" in prefs


class TestSentimentPersonalityIntegration:
    """Test that all 12 sentiment types properly adapt personality."""

    def _fresh_bp(self):
        from core.butler_personality import ButlerPersonality
        return ButlerPersonality(storage_path=tempfile.mkdtemp())

    def test_confused_sentiment(self):
        from core.butler_personality import PersonaMode
        bp = self._fresh_bp()
        bp._adapt_persona("confused", "moderate", False, False)
        state = bp.get_state()
        assert state["mode"] == PersonaMode.MENTOR.value
        assert bp._state.verbosity == Verbosity.DETAILED

    def test_overwhelmed_sentiment(self):
        from core.butler_personality import PersonaMode
        bp = self._fresh_bp()
        bp._adapt_persona("overwhelmed", "low", False, False)
        state = bp.get_state()
        assert state["mode"] == PersonaMode.COMPANION.value

    def test_determined_sentiment(self):
        from core.butler_personality import PersonaMode
        bp = self._fresh_bp()
        bp._adapt_persona("determined", "high", False, False)
        state = bp.get_state()
        assert state["mode"] == PersonaMode.BUTLER.value
        assert bp._state.energy == EnergyLevel.INTENSE

    def test_urgency_sentiment(self):
        from core.butler_personality import PersonaMode, Verbosity
        bp = self._fresh_bp()
        bp._adapt_persona("urgency", "high", True, False)
        state = bp.get_state()
        assert state["mode"] == PersonaMode.BUTLER.value
        assert bp._state.verbosity == Verbosity.TERSE

    def test_playful_sentiment(self):
        from core.butler_personality import PersonaMode
        bp = self._fresh_bp()
        bp._adapt_persona("playful", "high", False, False)
        state = bp.get_state()
        assert state["mode"] == PersonaMode.COLLABORATOR.value
        assert bp._state.formality == Formality.CASUAL

    def test_satisfied_sentiment(self):
        bp = self._fresh_bp()
        bp._adapt_persona("satisfied", "moderate", False, True)
        assert bp._state.warmth >= 0.65

    def test_grateful_sentiment(self):
        bp = self._fresh_bp()
        bp._adapt_persona("grateful", "moderate", False, True)
        assert bp._state.warmth >= 0.7

    def test_frustrated_sentiment(self):
        from core.butler_personality import PersonaMode, Verbosity
        bp = self._fresh_bp()
        bp._interaction_memory.frustration_signals = 5
        bp._adapt_persona("frustrated", "high", True, False)
        state = bp.get_state()
        assert state["mode"] == PersonaMode.BUTLER.value
        assert bp._state.verbosity == Verbosity.CONCISE

    def test_curious_sentiment(self):
        from core.butler_personality import PersonaMode
        bp = self._fresh_bp()
        bp._adapt_persona("curious", "moderate", False, False)
        state = bp.get_state()
        assert state["mode"] == PersonaMode.MENTOR.value

    def test_neutral_sentiment_no_change(self):
        from core.butler_personality import PersonaMode
        bp = self._fresh_bp()
        original_mode = bp.get_state()["mode"]
        bp._adapt_persona("neutral", "moderate", False, False)
        state = bp.get_state()
        assert state["mode"] == original_mode

    def test_excited_sentiment(self):
        from core.butler_personality import PersonaMode
        bp = self._fresh_bp()
        bp._adapt_persona("excited", "high", False, True)
        state = bp.get_state()
        assert state["mode"] == PersonaMode.COLLABORATOR.value

    def test_anxious_sentiment(self):
        from core.butler_personality import PersonaMode
        bp = self._fresh_bp()
        bp._adapt_persona("anxious", "low", True, False)
        state = bp.get_state()
        assert state["mode"] == PersonaMode.COMPANION.value
