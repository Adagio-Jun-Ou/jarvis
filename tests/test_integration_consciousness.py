"""Comprehensive integration tests for the full consciousness pipeline.

Tests that all new modules wire together correctly:
1. Emotional Memory ↔ Affective Context ↔ Interoception
2. Operational Rhythm ↔ Cognitive State Machine ↔ Bio Integration
3. Oracle Bridge ↔ Affective Context (uncertainty signals)
4. Full pipeline: Bio → Interoception → Emotional Memory → Affective Context → System Prompt
"""

import pytest
import time
import threading


class TestEmotionalMemoryAffectiveContextIntegration:
    def test_feeling_trajectory_propagates(self):
        from core.emotional_memory import EmotionalMemory
        from core.affective_context import AffectiveContextSystem
        em = EmotionalMemory()
        ac = AffectiveContextSystem()

        for i in range(12):
            valence = -0.5 + i * 0.1
            em.tag_interaction(valence=valence, arousal=0.6, quality="grounded")
            ac.record_feeling_snapshot(valence, 0.6, "grounded")

        dominant = em.get_dominant_feeling()
        assert dominant["avg_valence"] > -0.2

        pred = ac.predict_feeling(dominant["avg_valence"], 0.6, dominant["quality"])
        assert pred.confidence > 0.4
        assert pred.trend.value in ("improving", "stable")

    def test_regulation_learning_informs_context(self):
        from core.emotional_memory import EmotionalMemory
        from core.affective_context import AffectiveContextSystem, MoodFraming
        em = EmotionalMemory()
        ac = AffectiveContextSystem()

        em.record_feeling_shift("anxious", "grounded", regulation="cognitive_reappraisal")
        em.record_feeling_shift("anxious", "calm", regulation="cognitive_reappraisal")
        em.record_feeling_shift("stressed", "grounded", regulation="acceptance")

        best = em.get_best_regulation_for("anxious")
        assert best == "cognitive_reappraisal"

        bias = ac.compute_context_bias(-0.3, 0.7, "anxious")
        assert bias.framing in (MoodFraming.SUPPORTIVE, MoodFraming.CALMING)

    def test_recall_boost_integration(self):
        from core.emotional_memory import EmotionalMemory, AffectiveTag
        em = EmotionalMemory()
        for i in range(5):
            em.tag_interaction(valence=0.5, arousal=0.6, quality="joyful")
        for i in range(5):
            em.tag_interaction(valence=-0.3, arousal=0.8, quality="anxious")

        all_tags = list(em._affective_tags)
        boosts_positive = em.get_recall_boost(0.5, 0.6, all_tags)
        boosts_negative = em.get_recall_boost(-0.3, 0.8, all_tags)

        assert any(b > 1.0 for b in boosts_positive)
        assert any(b > 1.0 for b in boosts_negative)


class TestOperationalRhythmCSMIntegration:
    def test_deep_processing_triggers_dreaming(self):
        from core.cognitive_state_machine import CognitiveStateMachine, CognitiveState
        csm = CognitiveStateMachine()
        csm.state = CognitiveState.CONTEMPLATING
        csm._last_user_input_time = time.time() - 400

        result = csm._evaluate_transition(
            idle=400,
            bio_state={"stress_phase": "recovery", "operational_phase": "deep_processing",
                        "processing_fatigue": 0.3},
            prediction_state=None,
        )
        assert result is not None
        assert result[0] == CognitiveState.DREAMING

    def test_peak_engagement_exits_dreaming(self):
        from core.cognitive_state_machine import CognitiveStateMachine, CognitiveState
        csm = CognitiveStateMachine()
        csm.state = CognitiveState.DREAMING
        csm._state_entry_time = time.time() - 2000

        result = csm._evaluate_transition(
            idle=2000,
            bio_state={"operational_phase": "peak_engagement", "processing_fatigue": 0.1},
            prediction_state=None,
        )
        assert result is not None
        assert result[0] == CognitiveState.AWAKE

    def test_fatigue_gate_with_operational_phase(self):
        from core.cognitive_state_machine import CognitiveStateMachine, CognitiveState
        from core.circadian_rhythm import CircadianRhythm, OperationalPhase
        csm = CognitiveStateMachine()
        cr = CircadianRhythm()
        cr._processing_fatigue = 0.8
        cr._operational_phase = OperationalPhase.RECOVERY_CONSOLIDATION

        csm.state = CognitiveState.CONTEMPLATING
        csm._last_user_input_time = time.time() - 60

        result = csm._evaluate_transition(
            idle=60,
            bio_state={
                "stress_phase": "recovery",
                "operational_phase": "recovery_consolidation",
                "processing_fatigue": 0.8,
            },
            prediction_state=None,
        )
        assert result is not None
        assert result[0] == CognitiveState.DREAMING


class TestBioIntegrationFatigueFields:
    def test_physiological_state_populated_from_circadian(self):
        from core.bio_integration import BioIntegration
        from core.circadian_rhythm import CircadianRhythm, OperationalPhase
        bio = BioIntegration()
        circ = CircadianRhythm()
        circ._processing_fatigue = 0.6
        circ._operational_phase = OperationalPhase.SUSTAINED_LOAD
        bio.initialize(circadian=circ)

        state = bio.tick(dt=1.0)
        assert hasattr(state, "processing_fatigue")
        assert hasattr(state, "operational_phase")

    def test_response_style_uses_fatigue(self):
        from core.bio_integration import BioIntegration
        from core.circadian_rhythm import CircadianRhythm
        bio = BioIntegration()
        circ = CircadianRhythm()
        circ._processing_fatigue = 0.8
        bio.initialize(circadian=circ)

        style = bio.get_response_style()
        assert style["energy_level"] == "fatigued"
        assert any("consolidation" in s for s in style["adaptation_suggestions"])


class TestOracleBridgeAffectiveContext:
    def test_oracle_uncertainty_feeds_affect(self):
        from core.oracle_bridge import OracleBridge
        from core.affective_context import AffectiveContextSystem
        ob = OracleBridge()
        ac = AffectiveContextSystem()

        for domain in ["technology", "economics", "geopolitics"]:
            uncertainty = ob.get_uncertainty_for_affect(domain=domain)
            assert "prediction_uncertainty" in uncertainty
            assert 0.0 <= uncertainty["prediction_uncertainty"] <= 1.0

    def test_high_risk_adjusts_context(self):
        from core.oracle_bridge import OracleBridge
        from core.affective_context import AffectiveContextSystem, MoodFraming
        ob = OracleBridge()
        ac = AffectiveContextSystem()

        signals = ob.get_cross_domain_signals()
        for s in signals:
            if s.get("risk") == "high":
                bias = ac.compute_context_bias(-0.2, 0.6, "anxious")
                assert bias.urgency_discount < 1.0


class TestFullConsciousnessPipeline:
    def test_full_pipeline_positive_state(self):
        from core.bio_homeostasis import BioHomeostasis
        from core.circadian_rhythm import CircadianRhythm
        from core.bio_integration import BioIntegration
        from core.interoception import InteroceptionEngine
        from core.emotional_memory import EmotionalMemory
        from core.affective_context import AffectiveContextSystem, MoodFraming

        homeo = BioHomeostasis()
        circ = CircadianRhythm()
        bio = BioIntegration()
        bio.initialize(homeostasis=homeo, circadian=circ)

        ie = InteroceptionEngine()
        em = EmotionalMemory()
        ac = AffectiveContextSystem()

        for _ in range(5):
            bio.tick(dt=1.0)
            nt_state = {"dopamine": 0.5, "serotonin": 0.6, "norepinephrine": 0.4}
            ie.tick(nt_state=nt_state)
            feeling_state = ie.get_state()
            current = feeling_state.get("current_feeling", {})
            em.tag_interaction(
                valence=current.get("valence", 0.0),
                arousal=current.get("arousal", 0.5),
                quality=current.get("quality", "grounded"),
                awareness=feeling_state.get("awareness_score", 0.5),
            )
            ac.record_feeling_snapshot(
                current.get("valence", 0.0),
                current.get("arousal", 0.5),
                current.get("quality", "grounded"),
            )

        dominant = em.get_dominant_feeling()
        assert "quality" in dominant
        bias = ac.compute_context_bias(
            dominant.get("avg_valence", 0.0),
            dominant.get("avg_arousal", 0.5),
            dominant.get("quality", "neutral"),
        )
        assert isinstance(bias.framing, MoodFraming)

    def test_full_pipeline_stress_recovery(self):
        from core.bio_homeostasis import BioHomeostasis
        from core.circadian_rhythm import CircadianRhythm
        from core.bio_integration import BioIntegration
        from core.interoception import InteroceptionEngine
        from core.emotional_memory import EmotionalMemory
        from core.affective_context import AffectiveContextSystem

        homeo = BioHomeostasis()
        circ = CircadianRhythm()
        bio = BioIntegration()
        bio.initialize(homeostasis=homeo, circadian=circ)

        ie = InteroceptionEngine()
        em = EmotionalMemory()
        ac = AffectiveContextSystem()

        for i in range(10):
            if i < 3:
                homeo.apply_stressor(0.8, "test_stress")
            bio.tick(dt=1.0)
            nt_state = {"dopamine": 0.5, "serotonin": 0.6, "norepinephrine": 0.4}
            ie.tick(nt_state=nt_state)
            feeling = ie.get_state().get("current_feeling", {})
            em.tag_interaction(
                valence=feeling.get("valence", 0.0),
                arousal=feeling.get("arousal", 0.5),
                quality=feeling.get("quality", "grounded"),
            )
            ac.record_feeling_snapshot(
                feeling.get("valence", 0.0),
                feeling.get("arousal", 0.5),
                feeling.get("quality", "grounded"),
            )

        em.record_feeling_shift("stressed", "grounded",
                                  regulation="cognitive_reappraisal",
                                  trigger="stress_recovery")
        history = em.get_regulation_history("stressed")
        assert len(history) > 0
        best = em.get_best_regulation_for("stressed")
        assert best is not None


class TestConcurrentAccess:
    def test_concurrent_pipeline_access(self):
        from core.emotional_memory import EmotionalMemory
        from core.affective_context import AffectiveContextSystem
        em = EmotionalMemory()
        ac = AffectiveContextSystem()
        errors = []

        def worker(tid):
            try:
                for i in range(30):
                    val = (tid * 0.1 + i * 0.02) % 1.0 - 0.5
                    em.tag_interaction(valence=val, arousal=0.5, quality="grounded")
                    ac.record_feeling_snapshot(val, 0.5, "grounded")
                    ac.predict_feeling(val, 0.5, "grounded")
                    ac.compute_context_bias(val, 0.5, "grounded")
                    em.get_state()
                    ac.get_state()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        assert em.get_state()["total_tags"] == 150


class TestBackgroundTaskIntegration:
    def test_emotional_memory_background_task(self):
        import asyncio
        from core.background import create_default_scheduler

        scheduler = create_default_scheduler()
        task_names = [t.name for t in scheduler._tasks.values()]
        assert "emotional_memory_session" in task_names

    def test_bio_tick_includes_circadian(self):
        from core.background import create_default_scheduler
        scheduler = create_default_scheduler()
        task_names = [t.name for t in scheduler._tasks.values()]
        assert "bio_tick" in task_names

    def test_all_consciousness_background_tasks(self):
        from core.background import create_default_scheduler
        scheduler = create_default_scheduler()
        task_names = [t.name for t in scheduler._tasks.values()]
        expected = [
            "bio_tick",
            "cognitive_state_tick",
            "interoception_tick",
            "global_workspace_compete",
            "stream_consciousness",
            "self_model_reflection",
            "emotion_regulation_tick",
            "allostasis_prediction",
            "emotional_memory_session",
        ]
        for name in expected:
            assert name in task_names, f"Missing background task: {name}"


class TestEmotionalRecallBoostIntegration:
    def test_recall_boost_applied_to_search(self):
        from core.emotional_memory import EmotionalMemory
        from core.vector_memory import VectorMemory
        em = EmotionalMemory()
        for i in range(10):
            em.tag_interaction(valence=0.5, arousal=0.6, quality="joyful")
        for i in range(10):
            em.tag_interaction(valence=-0.3, arousal=0.8, quality="anxious")
        tags = list(em._affective_tags)
        boosts = em.get_recall_boost(0.5, 0.6, tags)
        assert len(boosts) == len(tags)
        positive_boosts = [b for i, b in enumerate(boosts)
                          if list(tags)[i].valence > 0]
        assert any(b > 1.0 for b in positive_boosts)

    def test_recall_boost_negative_mood(self):
        from core.emotional_memory import EmotionalMemory
        em = EmotionalMemory()
        for i in range(10):
            em.tag_interaction(valence=-0.4, arousal=0.8, quality="anxious")
        tags = list(em._affective_tags)
        boosts = em.get_recall_boost(-0.4, 0.8, tags)
        assert len(boosts) == len(tags)
        assert any(b > 1.0 for b in boosts)

    def test_search_with_emotional_boosts_parameter(self):
        from core.vector_memory import VectorMemory
        import tempfile
        from pathlib import Path
        vm = VectorMemory(storage_path=Path(tempfile.mkdtemp()))
        vm.store("happy memory", metadata={"importance": 0.8})
        vm.store("sad memory", metadata={"importance": 0.8})
        results = vm.search("memory", k=2, emotional_boosts=[1.3, 0.85])
        assert len(results) <= 2


class TestAffectiveContextResponseStyleIntegration:
    def test_stressed_mood_changes_response_style(self):
        from core.affective_context import AffectiveContextSystem, MoodFraming
        ac = AffectiveContextSystem()
        bias = ac.compute_context_bias(-0.4, 0.8, "stressed")
        style = {"detail_level": "moderate", "tone": "balanced", "energy_level": "normal",
                 "adaptation_suggestions": []}
        if bias.detail_bias < 0.4:
            style["detail_level"] = "concise"
        if bias.framing == MoodFraming.CALMING:
            style["tone"] = "calm_supportive"
        assert style["detail_level"] == "concise" or style["tone"] == "calm_supportive"

    def test_csm_depth_modulates_detail(self):
        from core.cognitive_state_machine import CognitiveStateMachine, CognitiveState
        csm = CognitiveStateMachine()
        csm.state = CognitiveState.DREAMING
        depth = csm.get_processing_depth()
        style = {"detail_level": "moderate"}
        if depth in ("deep", "focused"):
            style["detail_level"] = "deep"
        assert style["detail_level"] == "deep"

    def test_awake_depth_stays_moderate(self):
        from core.cognitive_state_machine import CognitiveStateMachine, CognitiveState
        csm = CognitiveStateMachine()
        csm.state = CognitiveState.AWAKE
        depth = csm.get_processing_depth()
        style = {"detail_level": "moderate"}
        if depth in ("deep", "focused"):
            style["detail_level"] = "deep"
        elif depth == "full":
            style["detail_level"] = "moderate"
        assert style["detail_level"] == "moderate"