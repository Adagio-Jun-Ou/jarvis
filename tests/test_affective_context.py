"""Tests for Affective Context System."""

import pytest
import time
import threading
from core.affective_context import (
    AffectiveContextSystem,
    MoodFraming,
    FeelingTrend,
    FeelingPrediction,
    ContextBias,
    get_affective_context,
    reset_affective_context,
)


@pytest.fixture(autouse=True)
def cleanup():
    reset_affective_context()
    yield
    reset_affective_context()


class TestAffectiveContextSystem:
    def test_init(self):
        ac = AffectiveContextSystem()
        state = ac.get_state()
        assert state["snapshots"] == 0
        assert state["last_prediction"] is None
        assert state["last_bias"] is None
        assert state["avg_resonance"] == 0.0

    def test_record_feeling_snapshot(self):
        ac = AffectiveContextSystem()
        ac.record_feeling_snapshot(0.5, 0.7, "joyful")
        assert ac.get_state()["snapshots"] == 1
        ac.record_feeling_snapshot(-0.3, 0.8, "anxious")
        assert ac.get_state()["snapshots"] == 2

    def test_predict_feeling_insufficient_data(self):
        ac = AffectiveContextSystem()
        pred = ac.predict_feeling(0.0, 0.5, "neutral")
        assert pred.confidence < 0.3
        assert pred.trend == FeelingTrend.STABLE
        assert pred.method == "insufficient_data"

    def test_predict_feeling_with_trajectory(self):
        ac = AffectiveContextSystem()
        for i in range(10):
            val = -0.5 + i * 0.1
            ac.record_feeling_snapshot(val, 0.5, "grounded")
        pred = ac.predict_feeling(0.3, 0.5, "grounded")
        assert pred.confidence > 0.3
        assert pred.trend in (FeelingTrend.IMPROVING, FeelingTrend.STABLE)
        assert pred.predicted_valence > 0.0

    def test_predict_feeling_declining(self):
        ac = AffectiveContextSystem()
        for i in range(10):
            val = 0.5 - i * 0.1
            ac.record_feeling_snapshot(val, 0.6, "stressed")
        pred = ac.predict_feeling(-0.3, 0.6, "stressed")
        assert pred.trend in (FeelingTrend.DECLINING, FeelingTrend.VOLATILE)
        assert pred.predicted_valence < -0.1

    def test_predict_feeling_volatile(self):
        ac = AffectiveContextSystem()
        values = [0.5, -0.3, 0.4, -0.5, 0.3, -0.2, 0.6, -0.4]
        for i, v in enumerate(values):
            ac.record_feeling_snapshot(v, 0.7, "anxious")
        pred = ac.predict_feeling(-0.2, 0.7, "anxious")
        assert pred.trend == FeelingTrend.VOLATILE
        assert pred.confidence < 0.7

    def test_predict_feeling_clamped(self):
        ac = AffectiveContextSystem()
        for i in range(10):
            ac.record_feeling_snapshot(0.8 + i * 0.05, 0.5, "joyful")
        pred = ac.predict_feeling(0.9, 0.5, "joyful", horizon_minutes=30)
        assert pred.predicted_valence <= 1.0
        assert pred.predicted_arousal >= 0.0

    def test_compute_context_bias_neutral(self):
        ac = AffectiveContextSystem()
        bias = ac.compute_context_bias(0.0, 0.5, "neutral", fatigue=0.0)
        assert bias.framing == MoodFraming.NEUTRAL
        assert 0.7 <= bias.empathy_level <= 0.9
        assert bias.urgency_discount == 1.0

    def test_compute_context_bias_stressed(self):
        ac = AffectiveContextSystem()
        bias = ac.compute_context_bias(-0.4, 0.8, "stressed", fatigue=0.3)
        assert bias.framing == MoodFraming.SUPPORTIVE
        assert bias.empathy_level > 1.0
        assert bias.urgency_discount < 1.0
        assert bias.positivity_bias > 0.0

    def test_compute_context_bias_overwhelmed(self):
        ac = AffectiveContextSystem()
        bias = ac.compute_context_bias(-0.6, 0.9, "overwhelmed", fatigue=0.8)
        assert bias.framing == MoodFraming.CALMING
        assert bias.empathy_level > 1.2
        assert bias.urgency_discount < 0.7
        assert bias.detail_bias < 0.5

    def test_compute_context_bias_joyful(self):
        ac = AffectiveContextSystem()
        bias = ac.compute_context_bias(0.6, 0.6, "joyful", fatigue=0.0)
        assert bias.framing == MoodFraming.OPTIMISTIC
        assert bias.empathy_level < 0.9
        assert bias.urgency_discount == 1.0

    def test_compute_context_bias_excited_high_arousal(self):
        ac = AffectiveContextSystem()
        bias = ac.compute_context_bias(0.5, 0.9, "excited", fatigue=0.0)
        assert bias.framing == MoodFraming.ENERGETIC
        assert bias.detail_bias < 0.5

    def test_compute_context_bias_fatigue_reduces_detail(self):
        ac = AffectiveContextSystem()
        bias_no_fatigue = ac.compute_context_bias(0.0, 0.5, "neutral", fatigue=0.0)
        bias_fatigue = ac.compute_context_bias(0.0, 0.5, "neutral", fatigue=0.8)
        assert bias_fatigue.detail_bias < bias_no_fatigue.detail_bias

    def test_compute_context_bias_approach_suggestions(self):
        ac = AffectiveContextSystem()
        for quality, expected_type in [
            ("joyful", "enthusiasm"),
            ("stressed", "gentle"),
            ("overwhelmed", "steps"),
            ("angry", "pressure"),
            ("curious", "action"),
        ]:
            bias = ac.compute_context_bias(0.0, 0.5, quality, fatigue=0.0)
            assert isinstance(bias.suggested_approach, str)
            assert len(bias.suggested_approach) > 5

    def test_compute_resonance(self):
        ac = AffectiveContextSystem()
        r1 = ac.compute_resonance(0.5, 0.5, 0.5, 0.5)
        assert r1 > 0.9
        r2 = ac.compute_resonance(0.5, 0.5, -0.5, 0.9)
        assert r2 < 0.5

    def test_compute_resonance_history(self):
        ac = AffectiveContextSystem()
        ac.compute_resonance(0.3, 0.5, 0.3, 0.5)
        ac.compute_resonance(0.5, 0.5, 0.1, 0.5)
        state = ac.get_state()
        assert state["avg_resonance"] > 0.0

    def test_get_feeling_context_summary(self):
        ac = AffectiveContextSystem()
        for i in range(8):
            ac.record_feeling_snapshot(-0.2 + i * 0.05, 0.6, "grounded")
        summary = ac.get_feeling_context_summary(0.1, 0.6, "grounded", fatigue=0.2)
        assert "Mood:" in summary
        assert "Approach:" in summary

    def test_get_feeling_context_summary_stressed(self):
        ac = AffectiveContextSystem()
        for i in range(8):
            ac.record_feeling_snapshot(-0.3 - i * 0.05, 0.8, "stressed")
        summary = ac.get_feeling_context_summary(-0.5, 0.8, "stressed", fatigue=0.4)
        assert "Urgency: reduced" in summary

    def test_get_feeling_context_summary_forecast(self):
        ac = AffectiveContextSystem()
        for i in range(8):
            ac.record_feeling_snapshot(0.2 - i * 0.05, 0.7, "anxious")
        summary = ac.get_feeling_context_summary(-0.1, 0.7, "anxious", fatigue=0.2)
        assert len(summary) > 10

    def test_singleton(self):
        ac1 = get_affective_context()
        ac2 = get_affective_context()
        assert ac1 is ac2

    def test_reset(self):
        ac1 = get_affective_context()
        ac1.record_feeling_snapshot(0.5, 0.5, "grounded")
        reset_affective_context()
        ac2 = get_affective_context()
        assert ac2.get_state()["snapshots"] == 0

    def test_thread_safety(self):
        ac = AffectiveContextSystem()
        errors = []

        def worker(thread_id):
            try:
                for i in range(50):
                    val = (thread_id * 0.1 + i * 0.01) % 1.0 - 0.5
                    ac.record_feeling_snapshot(val, 0.5, "neutral")
                    ac.predict_feeling(val, 0.5, "neutral")
                    ac.compute_context_bias(val, 0.5, "neutral")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_framing_map_coverage(self):
        ac = AffectiveContextSystem()
        for quality in ["joyful", "excited", "grounded", "calm", "anxious",
                        "stressed", "frustrated", "overwhelmed", "curious",
                        "bored", "sad", "angry", "fearful", "neutral"]:
            bias = ac.compute_context_bias(0.0, 0.5, quality, fatigue=0.0)
            assert isinstance(bias.framing, MoodFraming)

    def test_valence_slope_computation(self):
        ac = AffectiveContextSystem()
        slope = ac._compute_slope([(0.0, 0.0), (0.5, 60.0), (1.0, 120.0)])
        assert slope > 0

    def test_volatility_computation(self):
        ac = AffectiveContextSystem()
        low_vol = ac._compute_volatility([(0.5, 0.0), (0.51, 1.0), (0.52, 2.0)])
        high_vol = ac._compute_volatility([(0.5, 0.0), (-0.5, 1.0), (0.5, 2.0)])
        assert high_vol > low_vol


class TestFeelingPredictionDataclass:
    def test_defaults(self):
        pred = FeelingPrediction()
        assert pred.predicted_valence == 0.0
        assert pred.predicted_arousal == 0.5
        assert pred.confidence == 0.3
        assert pred.trend == FeelingTrend.STABLE

    def test_custom(self):
        pred = FeelingPrediction(
            predicted_valence=0.8,
            predicted_arousal=0.9,
            predicted_quality="excited",
            confidence=0.7,
            trend=FeelingTrend.IMPROVING,
        )
        assert pred.predicted_valence == 0.8
        assert pred.trend == FeelingTrend.IMPROVING


class TestContextBiasDataclass:
    def test_defaults(self):
        bias = ContextBias()
        assert bias.framing == MoodFraming.NEUTRAL
        assert bias.empathy_level == 0.5
        assert bias.urgency_discount == 1.0

    def test_custom(self):
        bias = ContextBias(
            framing=MoodFraming.SUPPORTIVE,
            empathy_level=1.3,
            urgency_discount=0.6,
        )
        assert bias.framing == MoodFraming.SUPPORTIVE
        assert bias.urgency_discount == 0.6


class TestCrossModuleIntegration:
    def test_affective_context_with_emotional_memory(self):
        from core.emotional_memory import EmotionalMemory
        from core.affective_context import AffectiveContextSystem
        em = EmotionalMemory()
        ac = AffectiveContextSystem()

        for i in range(10):
            val = 0.3 - i * 0.08
            em.tag_interaction(valence=val, arousal=0.7, quality="anxious",
                              awareness=0.6, topic="test")
            ac.record_feeling_snapshot(val, 0.7, "anxious")

        dominant = em.get_dominant_feeling()
        bias = ac.compute_context_bias(
            dominant["avg_valence"],
            dominant["avg_arousal"],
            dominant["quality"],
        )
        assert bias.framing in (MoodFraming.SUPPORTIVE, MoodFraming.CALMING)
        assert bias.empathy_level > 1.0

    def test_prediction_informs_regulation_choice(self):
        from core.affective_context import AffectiveContextSystem
        ac = AffectiveContextSystem()

        for i in range(8):
            ac.record_feeling_snapshot(-0.3 - i * 0.05, 0.8, "stressed")

        pred = ac.predict_feeling(-0.5, 0.8, "stressed")
        assert pred.trend in (FeelingTrend.DECLINING, FeelingTrend.VOLATILE)

        bias = ac.compute_context_bias(-0.5, 0.8, "stressed", fatigue=0.5)
        assert bias.urgency_discount < 1.0
        assert bias.empathy_level > 1.0