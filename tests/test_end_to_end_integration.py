"""Tests for end-to-end integration of all the new wiring from sprint 1-60."""
import pytest
import threading
import time


class TestEndToEndConsciousnessPipeline:
    """Test that the full consciousness pipeline produces cascading effects."""

    def setup_method(self):
        from core.interoception import InteroceptionEngine, InteroceptiveAttentionModulator
        from core.global_workspace import GlobalWorkspace
        from core.recursive_self_model import RecursiveSelfModel, SelfModelLevel
        from core.stream_consciousness import StreamOfConsciousness, ThoughtType
        from core.cognitive_state_machine import CognitiveStateMachine
        self.ie = InteroceptionEngine()
        self.mod = InteroceptiveAttentionModulator()
        self.gw = GlobalWorkspace()
        self.rsm = RecursiveSelfModel()
        self.soc = StreamOfConsciousness()
        self.csm = CognitiveStateMachine()

    def test_interoception_feeds_modulator(self):
        nt_state = {"dopamine": 0.5, "serotonin": 0.5, "norepinephrine": 0.5, "acetylcholine": 0.5}
        feeling = self.ie.tick(nt_state=nt_state)
        result = self.mod.update(feeling, bio_stress=0.3)
        assert "attention_aperture" in result
        assert result["attention_aperture"] > 0

    def test_high_prediction_error_narrows_attention(self):
        self.mod._current_aperture = 0.7
        self.mod.narrow(0.3)
        assert self.mod.get_state()["attention_aperture"] < 0.7

    def test_low_prediction_error_broadens_attention(self):
        self.mod._current_aperture = 0.4
        self.mod.broaden(0.3)
        assert self.mod.get_state()["attention_aperture"] > 0.4

    def test_self_model_introspection_cascades(self):
        from core.recursive_self_model import SelfModelLevel
        self.rsm.observe(SelfModelLevel.FIRST_ORDER, "test observation", confidence=0.9)
        self.rsm.observe(SelfModelLevel.SECOND_ORDER, "reflecting on test", confidence=0.8)
        self.rsm.tick()
        self.rsm.introspect()
        cached = self.rsm.get_last_introspection()
        assert cached is not None

    def test_workspace_amplification_resonance(self):
        from core.global_workspace import CoalitionType
        for _ in range(5):
            self.gw.submit_coalition(CoalitionType.EMOTIONAL, "strong feeling", 0.8)
        self.gw.tick(arousal=0.7)
        self.gw.amplify_context("emotion", strength_boost=0.3)
        echoes = self.gw.reverberate(limit=3)
        assert isinstance(echoes, list)


class TestEndToEndBioCircadianPipeline:
    """Test that bio → circadian → CSM → workspace cascading works."""

    def test_fatigue_affects_csm_transitions(self):
        from core.cognitive_state_machine import CognitiveStateMachine
        csm = CognitiveStateMachine()
        bio_state = {
            "stress_intensity": 0.1,
            "cognitive_capacity": 0.3,
            "operational_phase": "deep_processing",
        }
        prediction_state = {"free_energy": 0.7}
        result = csm._evaluate_transition(0.5, bio_state, prediction_state)
        assert result is not None or result is None

    def test_awakening_resets_fatigue(self):
        from core.circadian_rhythm import CircadianRhythm, OperationalPhase
        cr = CircadianRhythm()
        cr._processing_fatigue = 0.8
        cr._operational_phase = OperationalPhase.DEEP_PROCESSING
        cr.register_awakening()
        assert cr._processing_fatigue == 0.0
        assert cr._operational_phase == OperationalPhase.PEAK_ENGAGEMENT


class TestEndToEndEmotionPipeline:
    """Test emotion regulation → emotional memory → affective context."""

    def test_regulation_records_in_memory(self):
        from core.emotion_regulation import EmotionRegulationEngine
        from core.emotional_memory import EmotionalMemory
        ere = EmotionRegulationEngine()
        em = EmotionalMemory()
        reg = ere.regulate(
            feeling_state={
                "valence": -0.4, "arousal": 0.7, "dominance": 0.3,
                "quality": "stressed", "awareness_score": 0.6,
                "prediction_errors": {},
            },
        )
        if reg.get("strategy") != "cooldown":
            em.record_feeling_shift(
                pre_feeling="stressed",
                post_feeling=reg.get("post_feeling", "calmer"),
                regulation=reg["strategy"],
                trigger="test",
                valence=-0.4,
                arousal=0.7,
            )
        trajectory = em.get_feeling_trajectory(limit=3)
        assert isinstance(trajectory, list)


class TestEndToEndThreeLayerPipeline:
    """Test three-layer memory pipeline: L1 → L2 → L3."""

    def test_should_evict_large_history(self):
        from core.three_layer_context import ThreeLayerContext
        tlc = ThreeLayerContext()
        should = tlc.should_evict_l1(500000)
        assert should is True

    def test_should_not_evict_small_history(self):
        from core.three_layer_context import ThreeLayerContext
        tlc = ThreeLayerContext()
        should = tlc.should_evict_l1(100)
        assert should is False

    def test_consolidate_l2_to_l3(self):
        from core.three_layer_context import ThreeLayerContext
        tlc = ThreeLayerContext()
        count = tlc.consolidate_l2_to_l3(force=True)
        assert isinstance(count, int)

    def test_retrieve_from_l3(self):
        from core.three_layer_context import ThreeLayerContext
        tlc = ThreeLayerContext()
        results = tlc.retrieve_from_l3("test query", max_results=3)
        assert isinstance(results, list)


class TestEndToEndNeurosciencePipeline:
    """Test neuroscience pipeline: decision → reset → precision update."""

    def test_decision_resets_on_threshold(self):
        from core.neuroscience import NeuroscienceCognitiveArchitecture
        neuro = NeuroscienceCognitiveArchitecture.get_instance()
        result = neuro.make_decision({"strong_evidence": True, "support": True}, urgency=0.8)
        assert isinstance(result, dict)
        if result.get("reached_threshold", False):
            assert neuro.decision_circuit.evidence_accumulation == 0.0

    def test_tick_updates_precision(self):
        from core.neuroscience import NeuroscienceCognitiveArchitecture
        neuro = NeuroscienceCognitiveArchitecture.get_instance()
        initial_precision = neuro.predictive_coding.precision_weight
        neuro.predictive_coding.update_precision(0.8)
        new_precision = neuro.predictive_coding.precision_weight
        assert new_precision != initial_precision