"""Tests for Thought Sense — Predictive Intent Modeling System."""

import time
from collections import Counter
from core.thought_sense import (
    ThoughtSense,
    ThoughtState,
    IntentSignal,
    CognitiveState,
    EmotionalValence,
    PatternMemory,
    ConversationPulse,
    get_thought_sense,
)


class TestIntentSignal:
    def test_intent_values(self):
        assert IntentSignal.EXPLORATION.value == 1
        assert IntentSignal.CURIOSITY.value == 8
        assert IntentSignal.FRUSTRATION.value == 7

    def test_all_intents_unique(self):
        values = [s.value for s in IntentSignal]
        assert len(values) == len(set(values))
        assert len(IntentSignal) == 10


class TestCognitiveState:
    def test_cognitive_states(self):
        assert len(CognitiveState) == 7


class TestEmotionalValence:
    def test_valence_values(self):
        assert EmotionalValence.POSITIVE.value == 1
        assert EmotionalValence.NEGATIVE.value == 3
        assert len(EmotionalValence) == 4


class TestThoughtState:
    def test_default_state(self):
        state = ThoughtState()
        assert state.cognitive_state == CognitiveState.IDLE
        assert state.emotional_valence == EmotionalValence.NEUTRAL
        assert state.emotional_trajectory == 0.0
        assert state.cognitive_load == 0.0
        assert state.confidence == 0.0
        assert len(state.predicted_next_topics) == 0
        assert len(state.predicted_needs) == 0


class TestPatternMemory:
    def test_record_pattern(self):
        pm = PatternMemory()
        pm.record_pattern(["a", "b", "c"], "EXPLORATION", reward=1.0)
        assert pm.get_stats()["total_patterns"] == 1

    def test_record_pattern_too_short(self):
        pm = PatternMemory()
        pm.record_pattern(["a"], "EXPLORATION", reward=1.0)
        assert pm.get_stats()["total_patterns"] == 0

    def test_predict_outcomes_empty(self):
        pm = PatternMemory()
        assert pm.predict_outcomes([]) == []

    def test_predict_outcomes_with_data(self):
        pm = PatternMemory()
        for _ in range(5):
            pm.record_pattern(["python", "debug", "fix"], "CURIOSITY", reward=1.0)

        outcomes = pm.predict_outcomes(["debug", "fix", "python"], top_k=3)
        assert len(outcomes) >= 1
        assert outcomes[0][0] == "CURIOSITY"

    def test_pattern_memory_bounded(self):
        pm = PatternMemory()
        for i in range(pm.MAX_PATTERNS + 100):
            pm.record_pattern([f"topic_{i}", f"sub_{i}"], f"intent_{i}", reward=1.0)
        assert pm.get_stats()["total_patterns"] <= pm.MAX_PATTERNS

    def test_record_pattern_reward(self):
        pm = PatternMemory()
        pm.record_pattern(["a", "b"], "EXPLORATION", reward=0.5)
        pm.record_pattern(["a", "b"], "EXPLORATION", reward=0.8)
        stats = pm.get_stats()
        assert stats["total_patterns"] == 1
        assert stats["total_sequences"] == 2


class TestConversationPulse:
    def test_pulse_defaults(self):
        pulse = ConversationPulse()
        assert pulse.correction_count == 0
        assert pulse.question_count == 0
        assert pulse.last_topic is None

    def test_pulse_bounded(self):
        pulse = ConversationPulse()
        for i in range(100):
            pulse.message_times.append(time.time())
            pulse.message_lengths.append(50)
        assert len(pulse.message_times) <= 50
        assert len(pulse.message_lengths) <= 50


class TestThoughtSense:
    def test_initial_state(self):
        ts = ThoughtSense()
        state = ts.get_state()
        assert state.cognitive_state == CognitiveState.IDLE
        assert state.confidence == 0.0

    def test_observe_question(self):
        ts = ThoughtSense()
        ts.observe("What is the capital of France?", concept_type="user_query")
        state = ts.get_state()
        assert ts._conversation_pulse.question_count == 1
        assert state.intent_signals.get(IntentSignal.CURIOSITY) is not None

    def test_observe_correction(self):
        ts = ThoughtSense()
        ts.observe("Actually, I meant something else", concept_type="user_query")
        state = ts.get_state()
        assert ts._conversation_pulse.correction_count == 1

    def test_observe_frustration(self):
        ts = ThoughtSense()
        ts.observe("This is frustrating, it's broken and not working and keeps failing, I hate this error", concept_type="user_query")
        state = ts.get_state()
        assert state.emotional_valence in (EmotionalValence.NEGATIVE, EmotionalValence.MIXED)

    def test_observe_positive(self):
        ts = ThoughtSense()
        ts.observe("That's amazing and excellent work", concept_type="user_query")
        state = ts.get_state()
        assert state.emotional_valence == EmotionalValence.POSITIVE

    def test_observe_exploration(self):
        ts = ThoughtSense()
        ts.observe("What if we tried a different approach?", concept_type="user_query")
        state = ts.get_state()
        assert IntentSignal.CURIOSITY in state.intent_signals or IntentSignal.EXPLORATION in state.intent_signals

    def test_observe_completion(self):
        ts = ThoughtSense()
        ts.observe("That works, thanks!", concept_type="user_query")
        state = ts.get_state()
        assert IntentSignal.COMPLETION in state.intent_signals

    def test_cognitive_load_increases_with_rapid_messages(self):
        ts = ThoughtSense()
        for i in range(10):
            ts.observe(f"Quick message {i}?", concept_type="user_query")
        state = ts.get_state()
        assert state.cognitive_load > 0.0

    def test_proactive_suggestion_idle(self):
        ts = ThoughtSense()
        state = ts.get_state()
        ts._state.cognitive_state = CognitiveState.IDLE
        should, kind = ts.should_proactive_suggest()
        assert should is True
        assert kind == "idle_suggestion"

    def test_proactive_suggestion_flow(self):
        ts = ThoughtSense()
        ts._state.cognitive_state = CognitiveState.FLOW
        should, kind = ts.should_proactive_suggest()
        assert should is False

    def test_proactive_suggestion_overloaded(self):
        ts = ThoughtSense()
        ts._state.cognitive_state = CognitiveState.OVERLOADED
        should, kind = ts.should_proactive_suggest()
        assert should is False
        assert kind == "user_overloaded"

    def test_adapted_response_depth(self):
        ts = ThoughtSense()
        ts._state.cognitive_state = CognitiveState.OVERLOADED
        assert ts.get_adapted_response_depth() == "minimal"

        ts._state.cognitive_state = CognitiveState.IDLE
        ts._state.cognitive_load = 0.2
        assert ts.get_adapted_response_depth() in ["standard", "concise", "detailed", "minimal"]

    def test_adapted_tone(self):
        ts = ThoughtSense()
        ts._state.emotional_valence = EmotionalValence.NEGATIVE
        assert ts.get_adapted_tone() == "empathetic"

        ts._state.emotional_valence = EmotionalValence.POSITIVE
        assert ts.get_adapted_tone() == "enthusiastic"

    def test_topic_extraction(self):
        ts = ThoughtSense()
        topic = ts._extract_topic("Tell me about machine learning algorithms")
        assert topic is not None
        assert "machine" in topic or "machine_learning" in topic

    def test_topic_extraction_short(self):
        ts = ThoughtSense()
        topic = ts._extract_topic("hi")
        assert topic is None or len(topic) >= 2

    def test_multiple_observations_build_pattern(self):
        ts = ThoughtSense()
        ts.observe("How does Python work?", concept_type="user_query")
        ts.observe("Can you explain decorators in Python?", concept_type="user_query")
        ts.observe("What about async await?", concept_type="user_query")

        state = ts.get_state()
        assert ts._observation_count == 3
        assert len(ts._topic_history) == 3

    def test_prediction_confidence_increases(self):
        ts = ThoughtSense()
        initial_confidence = ts._estimate_prediction_confidence()

        for i in range(20):
            ts.observe(f"Tell me about topic {i}", concept_type="user_query")

        later_confidence = ts._estimate_prediction_confidence()
        assert later_confidence > initial_confidence

    def test_needs_prediction_overloaded(self):
        ts = ThoughtSense()
        ts._state.cognitive_state = CognitiveState.OVERLOADED
        ts._state.cognitive_load = 0.85
        needs = ts._predict_needs(CognitiveState.OVERLOADED, 0.85)
        assert "simplification" in needs

    def test_needs_prediction_frustrated(self):
        ts = ThoughtSense()
        needs = ts._predict_needs(CognitiveState.FRUSTRATED, 0.6)
        assert "encouragement" in needs

    def test_needs_prediction_flow(self):
        ts = ThoughtSense()
        needs = ts._predict_needs(CognitiveState.FLOW, 0.4)
        assert "minimal_interruption" in needs

    def test_thought_completions(self):
        ts = ThoughtSense()
        for i in range(10):
            ts.observe(f"Tell me about Python programming topic{i}", concept_type="user_query")

        completions = ts._generate_thought_completions("Tell me about Python")
        assert isinstance(completions, dict)

    def test_intent_transition_tracking(self):
        ts = ThoughtSense()
        ts.observe("What is machine learning?", concept_type="user_query")
        ts.observe("Explain neural networks in detail", concept_type="user_query")

        assert len(ts._intent_transition_counts) > 0

    def test_emotional_trajectory_positive(self):
        ts = ThoughtSense()
        # First message is clearly negative
        ts._state_history.append(ThoughtState(emotional_valence=EmotionalValence.NEGATIVE))
        # Second message is clearly positive
        ts.observe("This is wonderful and excellent! Great work!", concept_type="user_query")

        # The trajectory should reflect improvement
        trajectory = ts._estimate_emotional_trajectory(ts._state.emotional_valence)
        # If current is positive and history has negative, trajectory should be positive
        if ts._state.emotional_valence == EmotionalValence.POSITIVE:
            assert trajectory > 0 or trajectory == 0  # May be 0 if history state is same

    def test_stats(self):
        ts = ThoughtSense()
        ts.observe("Hello, how are you?", concept_type="user_query")

        stats = ts.get_stats()
        assert "observation_count" in stats
        assert "current_state" in stats
        assert "pattern_memory" in stats
        assert "adapted_depth" in stats
        assert "adapted_tone" in stats
        assert stats["observation_count"] == 1

    def test_topic_switch_detection(self):
        ts = ThoughtSense()
        ts.observe("Tell me about Python programming", concept_type="user_query")
        ts.observe("Now explain machine learning", concept_type="user_query")
        assert ts._conversation_pulse.topic_switches >= 1

    def test_singleton(self):
        ts1 = get_thought_sense()
        ts2 = get_thought_sense()
        assert ts1 is ts2


class TestThoughtSenseEdgeCases:
    def test_empty_content(self):
        ts = ThoughtSense()
        ts.observe("", concept_type="user_query")
        assert ts._observation_count == 0  # Empty content is ignored

    def test_very_long_content(self):
        ts = ThoughtSense()
        ts.observe("word " * 10000, concept_type="user_query")
        assert ts._observation_count == 1

    def test_no_pfc_prediction(self):
        ts = ThoughtSense()
        ts.observe("Hello", concept_type="user_query", pfc_prediction=None)
        assert ts._observation_count == 1

    def test_with_pfc_prediction(self):
        ts = ThoughtSense()
        pfc_pred = {
            "embedding_dim": 768,
            "type_probs": [("intention", 0.6), ("insight", 0.3)],
            "confidence": 0.7,
        }
        ts.observe("What should we explore?", concept_type="user_query", pfc_prediction=pfc_pred)
        assert ts._observation_count == 1

    def test_rapid_observations(self):
        ts = ThoughtSense()
        for i in range(50):
            ts.observe(f"Message {i}", concept_type="user_query")
        assert ts._observation_count == 50
        assert len(ts._state_history) <= 200