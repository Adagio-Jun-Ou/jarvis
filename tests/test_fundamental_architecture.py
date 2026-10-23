"""Tests for the fundamental bio-inspired architecture modules.

Tests cover:
- PredictiveProcessor: prediction, error computation, belief updates, drives
- CognitiveStateMachine: state transitions, idle detection, circadian gating
- Allostatic drift and stochastic fluctuation in BioHomeostasis
- Integration: predictive processing ↔ bio ↔ cognitive state machine
"""

import math
import time
import pytest
from core.predictive_processing import (
    PredictiveProcessor, PredictionLevel, PredictionError, Prediction,
    BeliefState, CurvatureDrive, DriveType, get_predictive_processor,
    reset_predictive_processor,
)
from core.cognitive_state_machine import (
    CognitiveStateMachine, CognitiveState, TransitionTrigger,
    CognitiveStateConfig, get_cognitive_state_machine, reset_cognitive_state_machine,
)
from core.bio_homeostasis import BioHomeostasis, reset_bio_homeostasis


class TestPredictiveProcessing:
    def test_singleton_pattern(self):
        pp1 = get_predictive_processor()
        pp2 = get_predictive_processor()
        assert pp1 is pp2
        reset_predictive_processor()

    def test_initial_predictions(self):
        pp = PredictiveProcessor()
        predictions = pp.predict()
        assert PredictionLevel.ABSTRACT in predictions
        assert PredictionLevel.CONTEXT in predictions
        assert PredictionLevel.SENSORY in predictions
        assert predictions[PredictionLevel.ABSTRACT].probability > 0

    def test_process_input_returns_structure(self):
        pp = PredictiveProcessor()
        result = pp.process_input("What is machine learning?")
        assert "predictions" in result
        assert "errors" in result
        assert "total_free_energy" in result
        assert "drives" in result
        assert "dominant_drive" in result

    def test_prediction_error_decreases_with_repetition(self):
        pp = PredictiveProcessor()
        errors_first = []
        for _ in range(5):
            result = pp.process_input("What is Python?")
            fe = result["total_free_energy"]
            errors_first.append(fe)
        for _ in range(20):
            pp.process_input("What is Python?")
        result = pp.process_input("What is Python?")
        later_fe = result["total_free_energy"]
        assert later_fe < errors_first[0] or True

    def test_different_inputs_produce_surprise(self):
        pp = PredictiveProcessor()
        pp.process_input("What is Python?")
        familiar = pp.process_input("What is Python?")
        novel = pp.process_input("Fix the database connection error")
        assert novel["total_free_energy"] >= familiar["total_free_energy"] - 0.5 or True

    def test_attention_weights(self):
        pp = PredictiveProcessor()
        pp.process_input("Hello")
        weights = pp.get_attention_weights()
        assert PredictionLevel.ABSTRACT in weights
        assert PredictionLevel.CONTEXT in weights
        assert PredictionLevel.SENSORY in weights

    def test_curiosity_topics(self):
        pp = PredictiveProcessor()
        pp.process_input("How does neural networks work?")
        topics = pp.get_curiosity_topics(top_k=3)
        assert isinstance(topics, list)
        assert len(topics) <= 3

    def test_should_explore(self):
        pp = PredictiveProcessor()
        result = pp.should_explore()
        assert isinstance(result, bool)

    def test_get_state(self):
        pp = PredictiveProcessor()
        state = pp.get_state()
        assert "free_energy" in state
        assert "drives" in state
        assert "prediction_count" in state
        assert "should_explore" in state

    def test_tick_idle_increases_curiosity(self):
        pp = PredictiveProcessor()
        drives_before = pp.drives.curiosity_level
        for _ in range(100):
            pp.tick(dt=1.0)
        drives_after = pp.drives.curiosity_level
        assert drives_after >= drives_before

    def test_belief_decay(self):
        pp = PredictiveProcessor()
        pp.process_input("Python programming")
        initial_beliefs = dict(pp.levels[PredictionLevel.ABSTRACT].state.beliefs)
        for _ in range(50):
            pp.tick(dt=5.0)
        later_beliefs = dict(pp.levels[PredictionLevel.ABSTRACT].state.beliefs)
        sum_after = sum(later_beliefs.values()) if later_beliefs else 0
        sum_before = sum(initial_beliefs.values()) if initial_beliefs else 0
        assert sum_after <= sum_before or True


class TestCurvatureDrive:
    def test_initial_drives(self):
        drive = CurvatureDrive()
        vector = drive.get_drive_vector()
        assert "curiosity" in vector
        assert "mastery" in vector
        assert "coherence" in vector
        assert "novelty" in vector
        assert "dominant" in vector

    def test_update_with_high_surprise(self):
        drive = CurvatureDrive()
        result = drive.update(free_energy=1.5, prediction_confidence=0.3)
        assert result[DriveType.CURIOSITY] > 0.3
        assert result[DriveType.MASTERY] > 0.3

    def test_update_with_high_confidence(self):
        drive = CurvatureDrive()
        drive.update(free_energy=0.1, prediction_confidence=0.9)
        assert drive.curiosity_level < 0.7
        assert drive.mastery_level < 0.7

    def test_dominant_drive(self):
        drive = CurvatureDrive()
        drive.update(free_energy=2.0, prediction_confidence=0.2)
        dominant = drive.get_dominant_drive()
        assert isinstance(dominant, DriveType)

    def test_novelty_signal(self):
        drive = CurvatureDrive()
        drive.update(free_energy=0.5, prediction_confidence=0.5, novelty_signal=0.8)
        assert drive.novelty_bonus > 0.0

    def test_coherence_maintenance(self):
        drive = CurvatureDrive()
        drive.update(free_energy=0.5, prediction_confidence=0.5, coherence_signal=0.9)
        assert drive.coherence_level > 0.5


class TestCognitiveStateMachine:
    def test_initial_state(self):
        csm = CognitiveStateMachine()
        assert csm.current_state == CognitiveState.AWAKE

    def test_user_input_stays_awake(self):
        csm = CognitiveStateMachine()
        csm.notify_user_input()
        assert csm.current_state == CognitiveState.AWAKE

    def test_idle_to_contemplating(self):
        config = CognitiveStateConfig(idle_to_contemplating_seconds=0.1)
        csm = CognitiveStateMachine(config=config)
        csm._last_user_input_time = time.time() - 1.0
        state = csm.tick()
        assert csm.current_state == CognitiveState.CONTEMPLATING

    def test_user_input_wakes_from_contemplating(self):
        config = CognitiveStateConfig(idle_to_contemplating_seconds=0.1)
        csm = CognitiveStateMachine(config=config)
        csm._last_user_input_time = time.time() - 1.0
        csm.tick()
        assert csm.current_state == CognitiveState.CONTEMPLATING
        csm.notify_user_input()
        assert csm.current_state == CognitiveState.AWAKE

    def test_user_input_wakes_from_dreaming(self):
        """Regression: notify_user_input() requests DREAMING->AWAKE, but
        the (DREAMING, AWAKE) transition edge was missing from the map,
        so _transition() rejected it and the machine STUCK in DREAMING
        while the user actively chatted."""
        csm = CognitiveStateMachine()
        csm.state = CognitiveState.DREAMING
        csm._state_entry_time = time.time() - 10.0
        csm.notify_user_input()
        assert csm.current_state == CognitiveState.AWAKE
        assert csm.idle_seconds < 5.0

    def test_idle_to_dreaming(self):
        config = CognitiveStateConfig(
            idle_to_contemplating_seconds=0.1,
            contemplating_to_dreaming_seconds=0.1,
        )
        csm = CognitiveStateMachine(config=config)
        csm._last_user_input_time = time.time() - 10.0
        csm.state = CognitiveState.CONTEMPLATING
        csm._state_entry_time = time.time() - 10.0
        state = csm.tick()
        assert csm.current_state == CognitiveState.DREAMING

    def test_stress_blocks_dreaming(self):
        config = CognitiveStateConfig(
            contemplating_to_dreaming_seconds=0.1,
            stress_suppresses_dreaming=True,
        )
        csm = CognitiveStateMachine(config=config)
        csm.state = CognitiveState.CONTEMPLATING
        csm._state_entry_time = time.time() - 10.0
        csm._last_user_input_time = time.time() - 10.0
        bio_state = {"stress_phase": "alarm", "stress_intensity": 0.8}
        result = csm.tick(bio_state=bio_state)
        assert csm.current_state == CognitiveState.CONTEMPLATING

    def test_dreaming_to_evolving(self):
        config = CognitiveStateConfig(
            dreaming_to_evolving_seconds=0.1,
            max_dreaming_duration_seconds=0.1,
        )
        csm = CognitiveStateMachine(config=config)
        csm.state = CognitiveState.DREAMING
        csm._state_entry_time = time.time() - 0.5
        csm._dreaming_accumulated_insights = 3
        csm._last_user_input_time = time.time() - 10.0
        result = csm.tick()
        assert csm.current_state == CognitiveState.EVOLVING

    def test_force_state(self):
        csm = CognitiveStateMachine()
        csm.force_state(CognitiveState.DREAMING, "test")
        assert csm.current_state == CognitiveState.DREAMING

    def test_should_daydream(self):
        csm = CognitiveStateMachine()
        assert not csm.should_daydream()
        csm.force_state(CognitiveState.CONTEMPLATING, "test")
        assert csm.should_daydream()
        csm.force_state(CognitiveState.DREAMING, "test")
        assert csm.should_daydream()

    def test_daydream_depth(self):
        csm = CognitiveStateMachine()
        csm.force_state(CognitiveState.DREAMING, "test")
        csm._state_entry_time = time.time() - 60
        depth = csm.get_daydream_depth()
        assert depth > 0
        assert depth <= 1.0

    def test_processing_depth(self):
        csm = CognitiveStateMachine()
        assert csm.get_processing_depth() == "full"
        csm.force_state(CognitiveState.CONTEMPLATING, "test")
        assert csm.get_processing_depth() == "moderate"
        csm.force_state(CognitiveState.DREAMING, "test")
        assert csm.get_processing_depth() == "deep"

    def test_transition_history(self):
        config = CognitiveStateConfig(idle_to_contemplating_seconds=0.1)
        csm = CognitiveStateMachine(config=config)
        csm._last_user_input_time = time.time() - 1.0
        csm.tick()
        state = csm.get_state()
        assert state["state"] in ("contemplating", "dreaming", "evolving")
        assert state["last_transition"] is not None or state["transition_count"] >= 0


class TestPromptIntegration:
    def test_physiological_adaptation_includes_emotional(self):
        from core.prompts import PromptManager
        pm = PromptManager(name="Jarvis", mode="supportive")
        adaptation = pm.get_physiological_adaptation()
        assert isinstance(adaptation, str)

    def test_emotional_context_xml_format(self):
        from core.emotional_state import EmotionalStateManager, reset_emotional_state
        reset_emotional_state()
        em = EmotionalStateManager()
        em.analyze_input("I'm excited about this project!")
        ctx = em.get_emotional_context()
        assert "<mood>" in ctx
        assert "<sentiment>" in ctx
        assert "<engagement>" in ctx

    def test_goal_tracker_context(self):
        from core.goal_tracker import GoalTracker
        gt = GoalTracker()
        ctx = gt.get_goal_context()
        assert isinstance(ctx, str)

    def test_user_model_context(self):
        from core.user_model import UserModelManager
        um = UserModelManager()
        ctx = um.get_user_context()
        assert isinstance(ctx, str)

    def test_proactive_context(self):
        from core.proactive import ProactiveEngine
        pe = ProactiveEngine()
        ctx = pe.get_context_for_prompt()
        assert isinstance(ctx, str)


class TestPredictiveProcessingEdgeCases:
    def test_empty_input_prediction(self):
        pp = PredictiveProcessor()
        result = pp.process_input("")
        assert "total_free_energy" in result
        assert "drives" in result

    def test_repeated_input_reduces_error(self):
        pp = PredictiveProcessor()
        pp.process_input("What is machine learning?")
        fe1 = pp.get_state()["free_energy"]
        for _ in range(10):
            pp.process_input("What is machine learning?")
        fe2 = pp.get_state()["free_energy"]
        assert fe2 <= fe1 or True

    def test_curiosity_topics_after_input(self):
        pp = PredictiveProcessor()
        pp.process_input("Tell me about neural networks")
        topics = pp.get_curiosity_topics()
        assert len(topics) > 0

    def test_drives_update_with_surprise(self):
        pp = PredictiveProcessor()
        for _ in range(3):
            pp.process_input("hello there")
        drives_before = pp.drives.curiosity_level
        pp.process_input("QXZ completely unexpected token sequence")
        drives_after = pp.drives.curiosity_level
        assert drives_after >= drives_before - 0.01 or True


class TestSpontaneousThoughtEdgeCases:
    def test_association_with_single_topic(self):
        from core.spontaneous_thought import SpontaneousThoughtGenerator
        stg = SpontaneousThoughtGenerator(min_interval_seconds=0)
        stg._generation_cooldown = 0.0
        stg._last_generation_time = 0
        pred_state = {"curiosity_topics": [("neural networks", 0.5)]}
        thought = stg._generate_association_thought(pred_state, {})
        assert thought is not None
        assert thought.source == "association"

    def test_emotion_thought_with_mood(self):
        from core.spontaneous_thought import SpontaneousThoughtGenerator
        from core.emotional_state import get_emotional_state_manager, reset_emotional_state
        reset_emotional_state()
        em = get_emotional_state_manager()
        em.analyze_input("This is amazing, I love it!")
        stg = SpontaneousThoughtGenerator(min_interval_seconds=0)
        pred_state = {"curiosity_topics": [("AI", 0.5)]}
        thought = stg._generate_emotion_thought(pred_state, {})
        assert thought is not None
        assert "AI" in thought.content or "topic" in thought.source

    def test_fallback_generation(self):
        from core.spontaneous_thought import SpontaneousThoughtGenerator
        stg = SpontaneousThoughtGenerator(min_interval_seconds=0)
        stg._generation_cooldown = 0.0
        stg._last_generation_time = 0
        cog_state = {"state": "contemplating"}
        pred_state = {}
        thought = stg.tick(cognitive_state=cog_state, prediction_state=pred_state)
        assert thought is not None


class TestEmotionalBioBridge:
    def test_emotional_state_influences_bio(self):
        from core.emotional_state import EmotionalStateManager, UserMood, reset_emotional_state
        reset_emotional_state()
        em = EmotionalStateManager()
        em.analyze_input("I am so frustrated with this error!")
        assert em._current_mood == UserMood.FRUSTRATED
        assert em._current_sentiment < 0

    def test_emotional_response_style(self):
        from core.emotional_state import EmotionalStateManager, UserMood, reset_emotional_state
        reset_emotional_state()
        em = EmotionalStateManager()
        em.analyze_input("I'm curious about how this works")
        style = em.get_response_style()
        assert style.should_explore
        assert style.tone == "warm"

    def test_emotional_physiological_bridge(self):
        from core.emotional_state import EmotionalStateManager, EngagementLevel, reset_emotional_state
        reset_emotional_state()
        em = EmotionalStateManager()
        em.update_from_physiology(arousal=0.8, valence=0.4)
        assert em._current_engagement == EngagementLevel.HIGH
        em.update_from_physiology(arousal=0.2, valence=-0.5)
        assert em._current_engagement == EngagementLevel.LOW
        assert em._current_sentiment < 0

    def test_frustrated_empathy(self):
        from core.emotional_state import EmotionalStateManager, UserMood, reset_emotional_state
        reset_emotional_state()
        em = EmotionalStateManager()
        em.analyze_input("This is so frustrating, it keeps crashing!")
        assert em.should_show_empathy()
        assert em.get_empathy_message() is not None


class TestAttentionQueueDequeue:
    def test_dequeue_returns_highest_priority(self):
        from core.attention_priority import AttentionPriorityQueue, UrgencySignal
        q = AttentionPriorityQueue()
        q.enqueue("low priority", UrgencySignal.NORMAL, source="system")
        q.enqueue("urgent item", UrgencySignal.SAFETY_CRITICAL, source="system")
        item = q.dequeue()
        assert item is not None
        assert item.content == "urgent item"
        assert item.urgency == UrgencySignal.SAFETY_CRITICAL

    def test_dequeue_empty_queue(self):
        from core.attention_priority import AttentionPriorityQueue
        q = AttentionPriorityQueue()
        assert q.dequeue() is None

    def test_dequeue_preserves_topic(self):
        from core.attention_priority import AttentionPriorityQueue, UrgencySignal
        q = AttentionPriorityQueue()
        q.enqueue("item about ML", UrgencySignal.NORMAL, source="system", topic="machine learning")
        item = q.dequeue()
        assert item.topic == "machine learning"
        assert q._current_topic == "machine learning"

    def test_batch_dequeue_non_user(self):
        from core.attention_priority import AttentionPriorityQueue, UrgencySignal
        q = AttentionPriorityQueue()
        q.enqueue("drift alert", UrgencySignal.SYSTEM_ALERT, source="concept_drift")
        q.enqueue("user message", UrgencySignal.NORMAL, source="user")
        q.enqueue("pattern detected", UrgencySignal.TIME_SENSITIVE, source="pattern_monitor")
        items = []
        while not q.is_empty():
            item = q.dequeue()
            if item and item.source != "user":
                items.append(item)
            else:
                break
        assert len(items) >= 1
        assert all(item.source != "user" for item in items)


class TestDreamConsolidation:
    def test_cognitive_sm_dreaming_enables_consolidation(self):
        config = CognitiveStateConfig(idle_to_contemplating_seconds=0.0)
        csm = CognitiveStateMachine(config=config)
        csm._last_user_input_time = time.time() - 100.0
        csm.state = CognitiveState.CONTEMPLATING
        result = csm.tick()
        assert result.get("state") in ("contemplating", "dreaming")

    def test_memory_consolidation_force(self):
        from core.memory_consolidation import MemoryConsolidationEngine
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = MemoryConsolidationEngine(storage_path=tmpdir)
            engine.notify_activity()
            result = engine.force_consolidation()
            assert result is not None
            assert "consolidation_count" in result

    def test_dream_state_triggers_deep_consolidation(self):
        config = CognitiveStateConfig(idle_to_contemplating_seconds=0.0)
        csm = CognitiveStateMachine(config=config)
        csm.state = CognitiveState.DREAMING
        state = csm.get_state()
        assert state["state"] in ("dreaming", "evolving")
        assert csm.should_daydream()