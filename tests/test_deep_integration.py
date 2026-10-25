"""Tests for deep integration: affective context → response style,
emotional memory → vector search recall boost, workspace → attention."""

import pytest
import time
import threading


class TestAffectiveContextResponseStyle:
    def test_anxious_state_produces_supportive_style(self):
        from core.affective_context import AffectiveContextSystem, MoodFraming
        ac = AffectiveContextSystem()
        bias = ac.compute_context_bias(-0.3, 0.8, "anxious", fatigue=0.2)
        assert bias.framing in (MoodFraming.SUPPORTIVE, MoodFraming.CALMING)
        assert bias.empathy_level > 1.0
        assert bias.urgency_discount < 1.0
        assert "gentle" in bias.suggested_approach.lower() or "acknowledge" in bias.suggested_approach.lower()

    def test_overwhelmed_state_reduces_detail(self):
        from core.affective_context import AffectiveContextSystem, MoodFraming
        ac = AffectiveContextSystem()
        bias = ac.compute_context_bias(-0.6, 0.9, "overwhelmed", fatigue=0.7)
        assert bias.detail_bias < 0.5
        assert bias.framing == MoodFraming.CALMING
        assert bias.urgency_discount < 0.7

    def test_joyful_state_is_optimistic(self):
        from core.affective_context import AffectiveContextSystem, MoodFraming
        ac = AffectiveContextSystem()
        bias = ac.compute_context_bias(0.6, 0.5, "joyful", fatigue=0.1)
        assert bias.framing == MoodFraming.OPTIMISTIC
        assert bias.empathy_level < 1.0
        assert "enthusiasm" in bias.suggested_approach.lower() or "possibilit" in bias.suggested_approach.lower()

    def test_excited_high_arousal_becomes_energetic(self):
        from core.affective_context import AffectiveContextSystem, MoodFraming
        ac = AffectiveContextSystem()
        bias = ac.compute_context_bias(0.5, 0.9, "excited", fatigue=0.1)
        assert bias.framing == MoodFraming.ENERGETIC
        assert bias.detail_bias < 0.5

    def test_fatigued_state_reduces_detail(self):
        from core.affective_context import AffectiveContextSystem
        ac = AffectiveContextSystem()
        no_fatigue = ac.compute_context_bias(0.0, 0.5, "neutral", fatigue=0.0)
        high_fatigue = ac.compute_context_bias(0.0, 0.5, "neutral", fatigue=0.8)
        assert high_fatigue.detail_bias < no_fatigue.detail_bias

    def test_prediction_declining_trajectory(self):
        from core.affective_context import AffectiveContextSystem, FeelingTrend
        ac = AffectiveContextSystem()
        for i in range(10):
            ac.record_feeling_snapshot(0.3 - i * 0.07, 0.6, "grounded")
        pred = ac.predict_feeling(-0.2, 0.6, "grounded")
        assert pred.trend in (FeelingTrend.DECLINING, FeelingTrend.VOLATILE)

    def test_prediction_improving_trajectory(self):
        from core.affective_context import AffectiveContextSystem, FeelingTrend
        ac = AffectiveContextSystem()
        for i in range(10):
            ac.record_feeling_snapshot(-0.3 + i * 0.08, 0.5, "grounded")
        pred = ac.predict_feeling(0.3, 0.5, "grounded")
        assert pred.trend == FeelingTrend.IMPROVING
        assert pred.confidence > 0.4

    def test_full_style_pipeline_stressed(self):
        from core.affective_context import AffectiveContextSystem
        from core.cognitive_state_machine import CognitiveStateMachine, CognitiveState
        ac = AffectiveContextSystem()
        csm = CognitiveStateMachine()
        csm.state = CognitiveState.AWAKE
        
        bias = ac.compute_context_bias(-0.4, 0.8, "stressed", fatigue=0.3)
        style = {"detail_level": "moderate", "tone": "balanced", "energy_level": "normal", "adaptation_suggestions": []}
        
        if bias.detail_bias < 0.4:
            style["detail_level"] = "concise"
        if bias.framing.value in ("calming", "supportive"):
            style["tone"] = "empathetic" if bias.framing.value == "supportive" else "calm_supportive"
        if bias.urgency_discount < 0.8:
            style["adaptation_suggestions"].append("reduce_urgency")
        
        assert style["detail_level"] in ("concise", "moderate")
        assert style["tone"] in ("empathetic", "calm_supportive")


class TestEmotionalMemoryRecallIntegration:
    def test_recall_boost_with_current_mood(self):
        from core.emotional_memory import EmotionalMemory
        em = EmotionalMemory()
        for i in range(5):
            em.tag_interaction(valence=0.5, arousal=0.6, quality="happy")
        for i in range(5):
            em.tag_interaction(valence=-0.3, arousal=0.8, quality="stressed")
        
        tags = list(em._affective_tags)
        happy_boosts = em.get_recall_boost(0.5, 0.6, tags)
        assert any(b > 1.0 for b in happy_boosts[:5])
        stressed_boosts = em.get_recall_boost(-0.3, 0.8, tags)
        assert any(b > 1.0 for b in stressed_boosts[5:])


class TestRegulationLearningIntegration:
    def test_regulation_learning_from_background(self):
        from core.emotion_regulation import EmotionRegulationEngine
        from core.emotional_memory import EmotionalMemory
        ere = EmotionRegulationEngine()
        em = EmotionalMemory()
        
        feeling = {"valence": -0.4, "arousal": 0.7, "quality": "anxious",
                   "dominance": 0.3, "awareness_score": 0.6, "prediction_errors": {}}
        result = ere.regulate(feeling)
        
        post_quality = result.get("post_feeling", "")
        strategy = result.get("strategy", "")
        if post_quality and post_quality != "anxious":
            em.record_feeling_shift(
                pre_feeling="anxious",
                post_feeling=post_quality,
                regulation=strategy,
                trigger="test_integration",
                valence=-0.4,
                arousal=0.7,
            )
        
        best = em.get_best_regulation_for("anxious")
        if best:
            assert isinstance(best, str)
            assert len(best) > 0

    def test_emotion_regulation_returns_all_fields(self):
        from core.emotion_regulation import EmotionRegulationEngine
        ere = EmotionRegulationEngine()
        feeling = {"valence": -0.3, "arousal": 0.7, "quality": "stressed"}
        result = ere.regulate(feeling)
        assert "strategy" in result
        assert "effects" in result
        assert "post_feeling" in result
        assert "budget_effect" in result
        assert isinstance(result["post_feeling"], str)
        assert isinstance(result["budget_effect"], float)


class TestBioIntegrationNewFields:
    def test_operational_phase_in_cognitive_context(self):
        from core.bio_integration import BioIntegration
        from core.circadian_rhythm import CircadianRhythm, OperationalPhase
        bio = BioIntegration()
        circ = CircadianRhythm()
        circ._operational_phase = OperationalPhase.SUSTAINED_LOAD
        bio.initialize(circadian=circ)
        ctx = bio.get_cognitive_context()
        assert "processing_fatigue" in ctx["physiological_state"]
        assert "operational_phase" in ctx["physiological_state"]

    def test_fatigue_affects_response_style(self):
        from core.bio_integration import BioIntegration
        from core.circadian_rhythm import CircadianRhythm
        bio = BioIntegration()
        circ = CircadianRhythm()
        for _ in range(20):
            circ.register_user_interaction()
            circ.tick(dt_minutes=1.0)
        circ._processing_fatigue = 0.85
        circ._operational_phase = type(circ._operational_phase).RECOVERY_CONSOLIDATION
        bio.initialize(circadian=circ)
        style = bio.get_response_style()
        energy_levels_with_fatigue = {"fatigued", "depleted"}
        assert style["energy_level"] in energy_levels_with_fatigue or "consolidation" in str(style.get("adaptation_suggestions", []))


class TestOracleBridgeIntegration:
    def test_oracle_affective_context_end_to_end(self):
        from core.oracle_bridge import OracleBridge
        from core.affective_context import AffectiveContextSystem, MoodFraming
        ob = OracleBridge()
        ac = AffectiveContextSystem()
        
        uncertainty = ob.get_uncertainty_for_affect(domain="technology")
        assert "prediction_uncertainty" in uncertainty
        
        ac.record_feeling_snapshot(-0.1, 0.6, "grounded")
        bias = ac.compute_context_bias(
            -0.1, 0.6, "grounded",
            fatigue=uncertainty["prediction_uncertainty"] * 0.5,
        )
        assert isinstance(bias.framing, MoodFraming)

    def test_oracle_heuristic_domains(self):
        from core.oracle_bridge import OracleBridge
        ob = OracleBridge()
        signals = ob.get_cross_domain_signals(["technology", "science"])
        assert len(signals) == 2
        for s in signals:
            assert "domain" in s
            assert "confidence" in s