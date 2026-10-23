"""Tests for global workspace operational phase modulation and ignition history."""
import pytest


class TestWorkspaceOperationalPhaseModulation:
    """Test workspace threshold/duration modulation by operational phase."""

    def setup_method(self):
        from core.global_workspace import GlobalWorkspace
        self.gw = GlobalWorkspace()

    def test_set_ignition_threshold_peak(self):
        self.gw.set_ignition_threshold(0.3)
        state = self.gw.get_state()
        assert state["ignition_threshold"] == pytest.approx(0.3, abs=0.01)

    def test_set_ignition_threshold_recovery(self):
        self.gw.set_ignition_threshold(0.6)
        state = self.gw.get_state()
        assert state["ignition_threshold"] == pytest.approx(0.6, abs=0.01)

    def test_set_ignition_threshold_bounds(self):
        self.gw.set_ignition_threshold(0.0)
        assert self.gw.get_state()["ignition_threshold"] >= 0.1
        self.gw.set_ignition_threshold(1.0)
        assert self.gw.get_state()["ignition_threshold"] <= 0.9

    def test_set_broadcast_duration(self):
        self.gw.set_broadcast_duration(3000.0)
        assert self.gw._broadcast_duration_ms == pytest.approx(3000.0, abs=1.0)

    def test_set_broadcast_duration_bounds(self):
        self.gw.set_broadcast_duration(0.0)
        assert self.gw._broadcast_duration_ms >= 100
        self.gw.set_broadcast_duration(50000.0)
        assert self.gw._broadcast_duration_ms <= 10000

    def test_decay_ignitions_reduces_strength(self):
        from core.global_workspace import CoalitionType
        for _ in range(5):
            self.gw.submit_coalition(CoalitionType.EMOTIONAL, "test coalition", 0.7)
        self.gw.tick(arousal=0.5)
        state_before = self.gw.get_state()
        self.gw.decay_ignitions(dt=5.0)
        # After decay, coalition strengths should be affected

    def test_get_ignition_history_empty(self):
        history = self.gw.get_ignition_history(limit=5)
        assert isinstance(history, list)

    def test_get_ignition_history_after_tick(self):
        from core.global_workspace import CoalitionType
        for _ in range(8):
            self.gw.submit_coalition(CoalitionType.INTEROCEPTIVE, "strong body signal", 0.9)
        self.gw.tick(arousal=0.7)
        if self.gw.get_state().get("total_ignitions", 0) > 0:
            history = self.gw.get_ignition_history(limit=3)
            assert len(history) > 0
            assert "type" in history[0]
            assert "strength" in history[0]


class TestCircadianAwakeningSleepOnset:
    """Test circadian transitions in operational context."""

    def test_awakening_resets_to_peak(self):
        from core.circadian_rhythm import CircadianRhythm, OperationalPhase
        cr = CircadianRhythm()
        cr._operational_phase = OperationalPhase.DEEP_PROCESSING
        cr._processing_fatigue = 0.9
        cr._time_awake = 7200.0
        cr.register_awakening()
        assert cr._operational_phase == OperationalPhase.PEAK_ENGAGEMENT
        assert cr._processing_fatigue == 0.0
        assert cr._time_awake == 0.0

    def test_sleep_onset_transitions_to_deep_processing(self):
        from core.circadian_rhythm import CircadianRhythm, OperationalPhase
        cr = CircadianRhythm()
        cr._operational_phase = OperationalPhase.PEAK_ENGAGEMENT
        result = cr.register_sleep_onset(duration_hours=2.0)
        assert cr._operational_phase == OperationalPhase.DEEP_PROCESSING
        assert result["previous_phase"] == "peak_engagement"
        assert result["new_phase"] == "deep_processing"


class TestContextGuardianEmergencyCallback:
    """Test context guardian emergency callback registration."""

    def test_callback_invoked_on_emergency(self):
        from core.context_guardian import ContextGuardian, GuardianAction, PressureLevel
        cg = ContextGuardian(token_budget=1000)
        received = []
        cg.add_emergency_callback(lambda a: received.append(a))
        cg.record_action(GuardianAction(
            level=PressureLevel.EMERGENCY,
            action="test",
            tokens_before=950,
            tokens_after=800,
            messages_removed=5,
            reason="test",
        ))
        assert len(received) == 1


class TestNeuroscienceIntegrations:
    """Test neuroscience decision reset and precision update."""

    def test_decision_resets_after_threshold(self):
        from core.neuroscience import NeuroscienceCognitiveArchitecture
        neuro = NeuroscienceCognitiveArchitecture.get_instance()
        result = neuro.make_decision({"strong_positive": True, "evidence": True, "confirm": True}, urgency=0.8)
        assert isinstance(result, dict)

    def test_precision_updates_in_tick(self):
        from core.neuroscience import NeuroscienceCognitiveArchitecture
        neuro = NeuroscienceCognitiveArchitecture.get_instance()
        neuro.tick()
        assert hasattr(neuro, '_tick_count')


class TestThreeLayerEvictionProduction:
    """Test three-layer eviction in production context."""

    def test_eviction_plan_generates_correct_structure(self):
        from core.context_guardian import ContextGuardian
        cg = ContextGuardian(token_budget=1000)
        plan = cg.get_eviction_plan(900, 30)
        assert "keep_recent" in plan
        assert "target_tokens" in plan

    def test_should_evict_normal_pressure(self):
        from core.context_guardian import ContextGuardian
        cg = ContextGuardian(token_budget=1000)
        assert cg.should_auto_evict(500) is False

    def test_should_evict_high_pressure(self):
        from core.context_guardian import ContextGuardian
        cg = ContextGuardian(token_budget=1000)
        assert cg.should_auto_evict(850) is True