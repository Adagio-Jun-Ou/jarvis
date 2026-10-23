"""Tests for circadian rhythm register_awakening and register_sleep_onset wiring."""
import pytest


class TestCircadianAwakening:
    """Test register_awakening production wiring."""

    def setup_method(self):
        from core.circadian_rhythm import CircadianRhythm
        self.rhythm = CircadianRhythm()

    def test_awakening_sets_peak_engagement(self):
        from core.circadian_rhythm import OperationalPhase
        self.rhythm._operational_phase = OperationalPhase.DEEP_PROCESSING
        self.rhythm.register_awakening()
        state = self.rhythm.get_state()
        assert state["operational_phase"] == "peak_engagement"

    def test_awakening_resets_fatigue(self):
        self.rhythm._processing_fatigue = 0.8
        self.rhythm.register_awakening()
        assert self.rhythm._processing_fatigue == 0.0

    def test_awakening_resets_time_awake(self):
        self.rhythm._time_awake = 3600.0
        self.rhythm.register_awakening()
        assert self.rhythm._time_awake == 0.0


class TestCircadianSleepOnset:
    """Test register_sleep_onset production wiring."""

    def setup_method(self):
        from core.circadian_rhythm import CircadianRhythm, OperationalPhase
        self.rhythm = CircadianRhythm()
        self.OperationalPhase = OperationalPhase

    def test_sleep_onset_sets_deep_processing(self):
        self.rhythm.register_sleep_onset(duration_hours=2.0)
        state = self.rhythm.get_state()
        assert state["operational_phase"] == "deep_processing"

    def test_sleep_onset_returns_previous_phase(self):
        from core.circadian_rhythm import OperationalPhase
        self.rhythm._operational_phase = OperationalPhase.PEAK_ENGAGEMENT
        result = self.rhythm.register_sleep_onset(duration_hours=2.0)
        assert result["previous_phase"] == "peak_engagement"
        assert result["new_phase"] == "deep_processing"

    def test_sleep_onset_preserves_fatigue(self):
        self.rhythm._processing_fatigue = 0.6
        result = self.rhythm.register_sleep_onset()
        assert result["processing_fatigue"] == 0.6


class TestCognitivePsychologyWiring:
    """Test cognitive psychology production wiring."""

    def test_set_cognitive_load(self):
        from core.cognitive_psychology import CognitivePsychologyEngine
        psych = CognitivePsychologyEngine.get_instance()
        psych.set_cognitive_load(0.7)
        state = psych.to_dict()
        assert "bias_profile" in state
        assert isinstance(state["bias_profile"].get("cognitive_load", 0), float)

    def test_record_reflection(self):
        from core.cognitive_psychology import CognitivePsychologyEngine, CognitiveReflection, CognitiveBias
        psych = CognitivePsychologyEngine.get_instance()
        initial_score = psych.get_cognitive_reflection_score()
        psych.record_reflection(CognitiveReflection(
            question="test question",
            intuitive_answer="intuitive",
            reflective_answer="reflective",
            overrides_intuition=True,
            reflection_time_ms=500.0,
            confidence_delta=0.3,
            bias_type=None,
        ))
        new_score = psych.get_cognitive_reflection_score()
        assert isinstance(new_score, float)

    def test_add_and_validate_mental_model(self):
        from core.cognitive_psychology import CognitivePsychologyEngine, MentalModel, MentalModelType
        psych = CognitivePsychologyEngine.get_instance()
        model = MentalModel(
            domain="test_domain",
            model_type=MentalModelType.CAUSAL,
            assumptions=["A causes B"],
            predictions=["B will follow A"],
            evidence_for=["observed A→B"],
            evidence_against=[],
        )
        psych.add_mental_model(model)
        result = psych.validate_mental_model("test_domain", new_evidence=["C also causes B"])
        assert result["status"] in ("validated", "needs_repair", "no_model")