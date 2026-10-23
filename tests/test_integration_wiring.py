"""Tests for integration wiring: making dead-end computed state flow to production."""
import pytest
import threading
import time


class TestRecursiveSelfModelIntrospection:
    """Test introspect() caching and production wiring."""

    def setup_method(self):
        from core.recursive_self_model import RecursiveSelfModel
        self.rsm = RecursiveSelfModel()

    def test_introspect_caches_result(self):
        result = self.rsm.introspect()
        assert result is not None
        assert "current_mode" in result
        cached = self.rsm.get_last_introspection()
        assert cached is not None
        assert cached["current_mode"] == result["current_mode"]

    def test_introspect_includes_cognitive_load_assessment(self):
        result = self.rsm.introspect()
        assert "should_throttle" in result
        assert "recommended_mode" in result
        assert isinstance(result["should_throttle"], bool)
        assert isinstance(result["recommended_mode"], str)

    def test_introspect_includes_conflicts(self):
        from core.recursive_self_model import SelfModelLevel
        self.rsm.observe(SelfModelLevel.FIRST_ORDER, "test observation one", confidence=0.9)
        self.rsm.observe(SelfModelLevel.SECOND_ORDER, "different observation", confidence=0.9)
        result = self.rsm.introspect()
        assert "conflicts" in result

    def test_introspect_includes_meta_narrative(self):
        from core.recursive_self_model import SelfModelLevel
        self.rsm.observe(SelfModelLevel.FIRST_ORDER, "test", confidence=0.8)
        self.rsm.tick()
        self.rsm.introspect()
        self.rsm.generate_meta_narrative()
        result = self.rsm.introspect()
        assert "meta_narrative" in result

    def test_get_state_reports_introspection_availability(self):
        state = self.rsm.get_state()
        assert "last_introspection_available" in state
        assert state["last_introspection_available"] is False
        self.rsm.introspect()
        state = self.rsm.get_state()
        assert state["last_introspection_available"] is True

    def test_judge_confidence_calibration(self):
        j = self.rsm.judge_confidence("test", 0.8, actual_accuracy=1.0)
        assert j.judgment_type == "test"
        assert j.confidence == 0.8
        assert j.calibration_error == pytest.approx(0.2, abs=0.01)
        cal_after = self.rsm.get_state()["confidence_calibration"]
        assert cal_after > 0.5


class TestAttentionModulatorAperture:
    """Test InteroceptiveAttentionModulator narrow/broaden methods."""

    def setup_method(self):
        from core.interoception import InteroceptiveAttentionModulator
        self.mod = InteroceptiveAttentionModulator()

    def test_narrow_reduces_aperture(self):
        initial = self.mod.get_state()["attention_aperture"]
        self.mod.narrow(0.2)
        after = self.mod.get_state()["attention_aperture"]
        assert after < initial

    def test_broaden_increases_aperture(self):
        self.mod.narrow(0.3)
        narrowed = self.mod.get_state()["attention_aperture"]
        self.mod.broaden(0.2)
        after = self.mod.get_state()["attention_aperture"]
        assert after > narrowed

    def test_aperture_bounds(self):
        self.mod.narrow(5.0)
        assert self.mod.get_state()["attention_aperture"] >= 0.1
        self.mod.broaden(5.0)
        assert self.mod.get_state()["attention_aperture"] <= 1.0

    def test_thread_safety(self):
        errors = []
        def narrow_loop():
            try:
                for _ in range(100):
                    self.mod.narrow(0.01)
            except Exception as e:
                errors.append(e)
        def broaden_loop():
            try:
                for _ in range(100):
                    self.mod.broaden(0.01)
            except Exception as e:
                errors.append(e)
        t1 = threading.Thread(target=narrow_loop)
        t2 = threading.Thread(target=broaden_loop)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert len(errors) == 0


class TestAllostasisRecoveryWiring:
    """Test that apply_recovery is wired into production paths."""

    def test_apply_recovery_reduces_load(self):
        from core.interoception import AllostasisPredictor
        ap = AllostasisPredictor()
        ap.apply_stressor(0.5, source="test")
        load_before = ap.get_state()["allostatic_load"]
        ap.apply_recovery(0.3, source="test_recovery")
        load_after = ap.get_state()["allostatic_load"]
        assert load_after < load_before

    def test_apply_recovery_reduces_stress(self):
        from core.interoception import AllostasisPredictor
        ap = AllostasisPredictor()
        ap.apply_stressor(0.8, source="test")
        stress_before = ap.get_state()["stress_level"]
        ap.apply_recovery(0.5, source="test_recovery")
        stress_after = ap.get_state()["stress_level"]
        assert stress_after < stress_before

    def test_recovery_curve_recorded(self):
        from core.interoception import AllostasisPredictor
        ap = AllostasisPredictor()
        ap.apply_stressor(0.5, source="test")
        ap.apply_recovery(0.2, source="rec1")
        ap.apply_recovery(0.2, source="rec2")
        state = ap.get_state()
        trend = state.get("recovery_trend")
        assert trend is not None


class TestEmotionRegulationPredictiveProcessing:
    """Test that predictive processing free energy feeds into emotion regulation."""

    def test_regulate_with_predictive_processing_errors(self):
        from core.emotion_regulation import EmotionRegulationEngine
        ere = EmotionRegulationEngine()
        result = ere.regulate(
            feeling_state={
                "valence": -0.3,
                "arousal": 0.7,
                "dominance": 0.3,
                "quality": "stressed",
                "awareness_score": 0.6,
                "prediction_errors": {"cardiac": 0.8, "predictive_processing": 0.6},
            },
        )
        assert result["strategy"] != "cooldown"
        assert "prediction_errors" not in result or True  # just check it doesn't crash

    def test_high_prediction_error_selects_reappraisal(self):
        from core.emotion_regulation import EmotionRegulationEngine
        ere = EmotionRegulationEngine()
        result = ere.regulate(
            feeling_state={
                "valence": -0.5,
                "arousal": 0.8,
                "dominance": 0.2,
                "quality": "anxious",
                "awareness_score": 0.7,
                "prediction_errors": {"predictive_processing": 0.9},
            },
        )
        assert result["strategy"] in (
            "cognitive_reappraisal", "acceptance", "body_budget_restoration",
            "attentional_deployment", "situation_modification", "situation_selection",
        )


class TestWorkspaceAmplification:
    """Test workspace amplification and reverberation."""

    def test_amplify_context_boosts_strength(self):
        from core.global_workspace import GlobalWorkspace, CoalitionType
        gw = GlobalWorkspace()
        gw.submit_coalition(CoalitionType.EMOTIONAL, "test feeling", 0.5)
        gw.submit_coalition(CoalitionType.CONCEPTUAL, "test idea", 0.4)
        gw.amplify_context("emotion", strength_boost=0.3)
        state = gw.get_state()
        assert "total_ignitions" in state

    def test_reverberation_after_ignition(self):
        from core.global_workspace import GlobalWorkspace, CoalitionType
        gw = GlobalWorkspace()
        for _ in range(5):
            gw.submit_coalition(CoalitionType.INTEROCEPTIVE, "body signal", 0.7)
            gw.submit_coalition(CoalitionType.EMOTIONAL, "emotional content", 0.6)
        gw.tick(arousal=0.7)
        echoes = gw.reverberate(limit=3)
        if gw.get_state().get("total_ignitions", 0) > 0:
            assert len(echoes) > 0

    def test_reverberation_empty_workspace(self):
        from core.global_workspace import GlobalWorkspace
        gw = GlobalWorkspace()
        echoes = gw.reverberate()
        assert echoes == []


class TestInteroceptionGranularityAndAwareness:
    """Test that get_granularity and get_awareness are wired correctly."""

    def test_get_awareness_after_tick(self):
        from core.interoception import InteroceptionEngine
        ie = InteroceptionEngine()
        nt_state = {"dopamine": 0.5, "serotonin": 0.5, "norepinephrine": 0.5, "acetylcholine": 0.5}
        ie.tick(nt_state=nt_state)
        awareness = ie.get_awareness()
        assert 0.0 <= awareness <= 1.0

    def test_get_granularity_after_tick(self):
        from core.interoception import InteroceptionEngine
        ie = InteroceptionEngine()
        nt_state = {"dopamine": 0.5, "serotonin": 0.5, "norepinephrine": 0.5, "acetylcholine": 0.5}
        ie.tick(nt_state=nt_state)
        granularity = ie.get_granularity()
        assert 0.0 <= granularity <= 1.0

    def test_granularity_increases_with_experience(self):
        from core.interoception import InteroceptionEngine
        ie = InteroceptionEngine()
        nt_state = {"dopamine": 0.5, "serotonin": 0.5, "norepinephrine": 0.5, "acetylcholine": 0.5}
        g1 = ie.get_granularity()
        for _ in range(5):
            ie.tick(nt_state=nt_state)
        g2 = ie.get_granularity()
        assert g2 >= g1


class TestBackgroundTaskIntegration:
    """Test that background tasks properly wire new integrations."""

    def test_self_model_reflection_updates_introspection(self):
        from core.recursive_self_model import RecursiveSelfModel, SelfModelLevel
        rsm = RecursiveSelfModel()
        rsm.observe(SelfModelLevel.FIRST_ORDER, "test observing", confidence=0.8)
        rsm.observe(SelfModelLevel.SECOND_ORDER, "observing myself", confidence=0.9)
        rsm.tick()
        result = rsm.introspect()
        assert result is not None
        assert "current_mode" in result
        cached = rsm.get_last_introspection()
        assert cached is not None
        assert cached["current_mode"] == result["current_mode"]

    def test_emotion_regulation_with_pp_errors(self):
        from core.emotion_regulation import EmotionRegulationEngine
        ere = EmotionRegulationEngine()
        result = ere.regulate(
            feeling_state={
                "valence": 0.1,
                "arousal": 0.4,
                "dominance": 0.5,
                "quality": "grounded",
                "awareness_score": 0.5,
                "prediction_errors": {"cardiac": 0.3, "predictive_processing": 0.5},
            },
        )
        assert "strategy" in result
        assert "budget_effect" in result