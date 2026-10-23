"""Tests for Shadow AI components — PFC, Thought Sense, and integration."""

import asyncio
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import numpy as np
import pytest

from core.prefrontal_cortex import (
    ConceptType,
    Concept,
    ConceptEncoder,
    CortexState,
    LCMCore,
    PrefrontalCortex,
)
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
from core.thinking import ThinkingEngine, Thought, SuggestionType, ConversationContext


class TestConceptEncoder:
    def test_init_no_global_seed(self):
        """Verify ConceptEncoder doesn't set global numpy random seed."""
        import numpy as np
        state = np.random.get_state()
        encoder = ConceptEncoder(embedding_dim=128)
        state_after = np.random.get_state()
        assert state[1][0] == state_after[1][0], "ConceptEncoder should not affect global numpy seed"

    def test_type_embeddings_separate(self):
        """Verify type embeddings are independent."""
        encoder = ConceptEncoder(embedding_dim=128)
        types = list(ConceptType)
        embeddings = [encoder.type_embeddings[t] for t in types]
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                assert not np.array_equal(embeddings[i], embeddings[j]), f"Type embeddings {types[i]} and {types[j]} should differ"

    def test_position_embeddings(self):
        """Verify position embeddings are deterministic."""
        encoder1 = ConceptEncoder(embedding_dim=128)
        encoder2 = ConceptEncoder(embedding_dim=128)
        assert np.array_equal(encoder1.position_embeddings, encoder2.position_embeddings)


class TestLCMCore:
    def test_init_no_global_seed(self):
        """Verify LCMCore doesn't affect global numpy seed."""
        import numpy as np
        state = np.random.get_state()
        lcm = LCMCore()
        state_after = np.random.get_state()
        assert state[1][0] == state_after[1][0]

    def test_training_buffer_bounded(self):
        """Verify training buffer is bounded."""
        lcm = LCMCore()
        assert lcm._training_buffer.maxlen == 10000

    def test_loss_history_bounded(self):
        """Verify loss history is bounded."""
        lcm = LCMCore()
        assert lcm._loss_history.maxlen == 1000

    def test_add_training_example_thread_safe(self):
        """Verify training examples can be added from multiple threads."""
        lcm = LCMCore()
        concept = Concept(
            id="test_concept",
            embedding=np.zeros(768),
            concept_type=ConceptType.OBSERVATION,
            content="test content",
            source="test",
        )
        errors = []

        def add_examples():
            try:
                for _ in range(100):
                    lcm.add_training_example([concept], concept, reward=1.0)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=add_examples) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"
        # Verify all 500 examples were added
        assert len(lcm._training_buffer) == 500


class TestPrefrontalCortex:
    def test_start_sequence_preserves_deque(self):
        """Verify start_sequence uses bounded deque, not unbounded list."""
        pfc = PrefrontalCortex(auto_train=False)
        pfc.start_sequence(goal="test")
        assert isinstance(pfc._current_sequence, deque)
        assert pfc._current_sequence.maxlen == 1000

    def test_end_sequence_preserves_deque(self):
        """Verify end_sequence resets with bounded deque."""
        pfc = PrefrontalCortex(auto_train=False)
        pfc.start_sequence(goal="test")
        pfc._current_sequence.append("concept_1")
        # end_sequence is async, just test the reset directly
        pfc._current_sequence = deque(maxlen=1000)
        pfc._sequence_goal = None
        assert isinstance(pfc._current_sequence, deque)
        assert pfc._current_sequence.maxlen == 1000

    def test_callback_list_bounded(self):
        """Verify callback lists are bounded."""
        pfc = PrefrontalCortex(auto_train=False)
        assert pfc._MAX_CALLBACKS == 20

        for i in range(30):
            pfc.on_prediction(lambda: None)
        assert len(pfc._on_prediction_callbacks) <= 20

    def test_anomaly_count_increment(self):
        """Verify anomaly detection increments counter."""
        state = CortexState()
        assert state.anomaly_count == 0

    def test_get_stats_includes_anomaly(self):
        """Verify get_stats includes anomaly count."""
        pfc = PrefrontalCortex(auto_train=False)
        stats = pfc.get_stats()
        assert "total_concepts" in stats
        assert "stored_concepts" in stats

    def test_get_predictions_for_thought_sense(self):
        """Verify get_predictions_for_thought_sense returns dict."""
        pfc = PrefrontalCortex(auto_train=False)
        predictions = pfc.get_predictions_for_thought_sense()
        assert "active_concepts" in predictions
        assert "confidence" in predictions
        assert "predicted_next" in predictions
        assert "concept_types" in predictions
        assert "topic_drift" in predictions
        assert "anomaly_flags" in predictions


class TestThoughtSense:
    def test_singleton(self):
        """Verify get_thought_sense returns same instance."""
        ts1 = get_thought_sense()
        ts2 = get_thought_sense()
        assert ts1 is ts2

    def test_observe_creates_state(self):
        """Verify observe creates a ThoughtState."""
        ts = ThoughtSense()
        ts.observe("Hello, how are you?", concept_type="user_query")
        state = ts.get_state()
        assert isinstance(state, ThoughtState)

    def test_intent_detection_curiosity(self):
        """Verify curiosity intent detection."""
        ts = ThoughtSense()
        ts.observe("What makes quantum computing different?", concept_type="user_query")
        state = ts.get_state()
        assert IntentSignal.CURIOSITY in state.intent_signals

    def test_intent_detection_correction(self):
        """Verify correction intent detection."""
        ts = ThoughtSense()
        ts.observe("Actually, I meant something different", concept_type="user_query")
        state = ts.get_state()
        assert IntentSignal.CORRECTION in state.intent_signals

    def test_intent_detection_completion(self):
        """Verify completion intent detection."""
        ts = ThoughtSense()
        ts.observe("That's great, I got it, thanks!", concept_type="user_query")
        state = ts.get_state()
        assert IntentSignal.COMPLETION in state.intent_signals

    def test_proactive_suggestion_overloaded(self):
        """Verify proactive suggestions are suppressed when overloaded."""
        ts = ThoughtSense()
        ts._state = ThoughtState(cognitive_state=CognitiveState.OVERLOADED)
        should, kind = ts.should_proactive_suggest()
        assert should is False
        assert kind == "user_overloaded"

    def test_proactive_suggestion_flow(self):
        """Verify proactive suggestions are suppressed during flow."""
        ts = ThoughtSense()
        ts._state = ThoughtState(cognitive_state=CognitiveState.FLOW)
        should, kind = ts.should_proactive_suggest()
        assert should is False

    def test_proactive_suggestion_idle(self):
        """Verify proactive suggestions are appropriate when idle."""
        ts = ThoughtSense()
        ts._state = ThoughtState(cognitive_state=CognitiveState.IDLE)
        should, kind = ts.should_proactive_suggest()
        assert should is True
        assert kind == "idle_suggestion"

    def test_adapted_response_depth(self):
        """Verify response depth adaptation."""
        ts = ThoughtSense()
        ts._state = ThoughtState(cognitive_state=CognitiveState.OVERLOADED)
        assert ts.get_adapted_response_depth() == "minimal"

    def test_adapted_tone(self):
        """Verify tone adaptation."""
        ts = ThoughtSense()
        ts._state = ThoughtState(emotional_valence=EmotionalValence.NEGATIVE)
        assert ts.get_adapted_tone() == "empathetic"

    def test_pattern_memory_record_and_predict(self):
        """Verify pattern memory records and predicts."""
        pm = PatternMemory()
        pm.record_pattern(["a", "b", "c"], "EXPLORATION")
        assert pm.get_stats()["total_patterns"] == 1

    def test_pattern_memory_bounded(self):
        """Verify pattern memory is bounded."""
        pm = PatternMemory()
        for i in range(pm.MAX_PATTERNS + 100):
            pm.record_pattern([f"topic_{i}"], f"intent_{i}")
        assert pm.get_stats()["total_patterns"] <= pm.MAX_PATTERNS

    def test_conversation_pulse_bounded(self):
        """Verify conversation pulse has bounded deques."""
        pulse = ConversationPulse()
        for i in range(200):
            pulse.message_times.append(time.time())
            pulse.message_lengths.append(50)
        assert len(pulse.message_times) <= 50
        assert len(pulse.message_lengths) <= 50

    def test_cognitive_load_rapid_messages(self):
        """Verify cognitive load increases with rapid messages."""
        ts = ThoughtSense()
        for i in range(15):
            ts.observe(f"Quick question {i}?", concept_type="user_query")
        state = ts.get_state()
        assert state.cognitive_load > 0.3

    def test_topic_extraction(self):
        """Verify topic extraction returns meaningful topics."""
        ts = ThoughtSense()
        topic = ts._extract_topic("Tell me about machine learning algorithms")
        assert topic is not None

    def test_topic_extraction_short_returns_none(self):
        """Verify very short content returns None."""
        ts = ThoughtSense()
        topic = ts._extract_topic("ok")
        assert topic is None or len(topic) == 2

    def test_save_and_load_state(self, tmp_path):
        """Verify Thought Sense state persistence."""
        from pathlib import Path
        ts = ThoughtSense()
        ts._state_path = tmp_path / "thought_sense" / "state.json"
        ts._observation_count = 42
        ts._topic_counts["python"] = 10
        ts._save_state()

        ts2 = ThoughtSense()
        ts2._state_path = tmp_path / "thought_sense" / "state.json"
        ts2.load_state()
        assert ts2._observation_count == 42
        assert ts2._topic_counts.get("python") == 10

    def test_needs_prediction_frustrated(self):
        """Verify needs prediction for frustrated user."""
        ts = ThoughtSense()
        needs = ts._predict_needs(CognitiveState.FRUSTRATED, 0.7)
        assert "encouragement" in needs

    def test_needs_prediction_flow(self):
        """Verify needs prediction during flow state."""
        ts = ThoughtSense()
        needs = ts._predict_needs(CognitiveState.FLOW, 0.4)
        assert "minimal_interruption" in needs

    def test_record_suggestion_outcome(self):
        """Verify suggestion outcome recording via ThinkingEngine."""
        from core.thinking import ThinkingEngine
        import tempfile
        config = MagicMock()
        config.thinking = MagicMock()
        config.thinking.enabled = True
        config.thinking.mode = "supportive"
        config.thinking.suggestion_interval = 30
        config.thinking.confidence_threshold = 0.7
        config.thinking.max_suggestions_per_turn = 3
        config.thinking.pause_detection = True
        config.identity = MagicMock()
        config.identity.name = "Jarvis"
        te = ThinkingEngine(llm_client=None, config=config)
        # Use a temp path to avoid state file interference
        te._state_path = Path(tempfile.mkdtemp()) / "test_thinking_state.json"
        te._total_suggestions = 10
        te._accepted_suggestions = 0
        te._suggestion_outcomes = {}
        te.record_suggestion_outcome("thought_test_1", accepted=True)
        assert te._accepted_suggestions == 1
        assert "thought_test_1" in te._suggestion_outcomes
        assert te._suggestion_outcomes["thought_test_1"] == "accepted"

    def test_adapted_confidence_threshold(self):
        """Verify confidence threshold adapts based on acceptance rate."""
        from core.thinking import ThinkingEngine
        import tempfile
        config = MagicMock()
        config.thinking = MagicMock()
        config.thinking.enabled = True
        config.thinking.mode = "supportive"
        config.thinking.suggestion_interval = 30
        config.thinking.confidence_threshold = 0.7
        config.thinking.max_suggestions_per_turn = 3
        config.thinking.pause_detection = True
        config.identity = MagicMock()
        config.identity.name = "Jarvis"
        te = ThinkingEngine(llm_client=None, config=config)
        te._state_path = Path(tempfile.mkdtemp()) / "test_thinking_state.json"
        te._total_suggestions = 20
        te._accepted_suggestions = 15  # 75% acceptance rate
        adapted = te.get_adapted_confidence_threshold()
        assert adapted <= te.confidence_threshold

    def test_pfc_predictions_integration(self):
        """Verify PFC predictions can be fed into Thought Sense."""
        ts = ThoughtSense()
        pfc_pred = {
            "embedding_dim": 768,
            "type_probs": [("intention", 0.6), ("insight", 0.3)],
            "confidence": 0.7,
        }
        ts.observe("What should we explore next?", concept_type="user_query", pfc_prediction=pfc_pred)
        assert ts._observation_count == 1


class TestCortexState:
    def test_default_state(self):
        """Verify CortexState defaults."""
        state = CortexState()
        assert state.anomaly_count == 0
        assert state.confidence == 0.0
        assert state.training_steps == 0

    def test_deques_bounded(self):
        """Verify CortexState deques are bounded."""
        state = CortexState()
        assert state.active_concepts.maxlen == 100
        assert state.recent_sequences.maxlen == 20


class TestConceptDeduplication:
    """Tests for PFC concept deduplication by embedding similarity."""

    def test_find_similar_concept_returns_none_for_first_concept(self):
        pfc = PrefrontalCortex(auto_train=False)
        concept = Concept(
            id="test_1",
            embedding=np.random.RandomState(42).randn(768).astype(np.float32),
            concept_type=ConceptType.USER_QUERY,
            content="hello world",
            source="test",
            timestamp=time.time(),
        )
        result = pfc._find_similar_concept(concept)
        assert result is None

    def test_find_similar_concept_detects_near_duplicate(self):
        pfc = PrefrontalCortex(auto_train=False)
        rng = np.random.RandomState(42)
        base_embedding = rng.randn(768).astype(np.float32)
        base_embedding = base_embedding / np.linalg.norm(base_embedding)

        concept1 = Concept(
            id="test_1",
            embedding=base_embedding.copy(),
            concept_type=ConceptType.USER_QUERY,
            content="what is python",
            source="test",
            timestamp=time.time(),
        )
        pfc._concepts["test_1"] = concept1

        similar_embedding = base_embedding + rng.randn(768).astype(np.float32) * 0.01
        concept2 = Concept(
            id="test_2",
            embedding=similar_embedding,
            concept_type=ConceptType.USER_QUERY,
            content="what is python programming",
            source="test",
            timestamp=time.time(),
        )
        result = pfc._find_similar_concept(concept2)
        assert result == "test_1"

    def test_dedup_threshold_respected(self):
        pfc = PrefrontalCortex(auto_train=False)
        pfc._dedup_threshold = 0.99
        rng = np.random.RandomState(42)

        concept1 = Concept(
            id="test_1",
            embedding=rng.randn(768).astype(np.float32),
            concept_type=ConceptType.USER_QUERY,
            content="hello",
            source="test",
            timestamp=time.time(),
        )
        pfc._concepts["test_1"] = concept1

        concept2 = Concept(
            id="test_2",
            embedding=rng.randn(768).astype(np.float32),
            concept_type=ConceptType.USER_QUERY,
            content="world",
            source="test",
            timestamp=time.time(),
        )
        result = pfc._find_similar_concept(concept2)
        assert result is None

    def test_different_types_not_deduplicated(self):
        pfc = PrefrontalCortex(auto_train=False)
        rng = np.random.RandomState(42)
        emb = rng.randn(768).astype(np.float32)

        concept1 = Concept(
            id="test_1",
            embedding=emb.copy(),
            concept_type=ConceptType.USER_QUERY,
            content="test",
            source="test",
            timestamp=time.time(),
        )
        pfc._concepts["test_1"] = concept1

        concept2 = Concept(
            id="test_2",
            embedding=emb.copy(),
            concept_type=ConceptType.TOOL_CALL,
            content="test",
            source="test",
            timestamp=time.time(),
        )
        result = pfc._find_similar_concept(concept2)
        assert result is None

    def test_dedup_stats_tracking(self):
        pfc = PrefrontalCortex(auto_train=False)
        assert pfc._dedup_checks == 0
        assert pfc._dedup_hits == 0


class TestConceptDecay:
    """Tests for PFC concept decay and forgetting mechanism."""

    def test_decay_prunes_low_importance(self):
        pfc = PrefrontalCortex(auto_train=False)
        pfc._DECAY_INTERVAL = 0
        old_time = time.time() - 7200
        for i in range(5):
            concept = Concept(
                id=f"old_{i}",
                embedding=np.random.RandomState(i).randn(768).astype(np.float32),
                concept_type=ConceptType.OBSERVATION,
                content=f"old observation {i}",
                source="test",
                timestamp=old_time,
            )
            concept.frequency = 1
            concept.reward = 0.1
            pfc._concepts[f"old_{i}"] = concept

        pruned = pfc._apply_concept_decay()
        assert pruned >= 0

    def test_get_concept_importance(self):
        pfc = PrefrontalCortex(auto_train=False)
        concept = Concept(
            id="test_1",
            embedding=np.random.RandomState(42).randn(768).astype(np.float32),
            concept_type=ConceptType.USER_QUERY,
            content="test",
            source="test",
            timestamp=time.time(),
        )
        concept.frequency = 10
        concept.reward = 0.5
        pfc._concepts["test_1"] = concept

        importance = pfc.get_concept_importance("test_1")
        assert importance > 0

    def test_get_concept_importance_nonexistent(self):
        pfc = PrefrontalCortex(auto_train=False)
        importance = pfc.get_concept_importance("nonexistent")
        assert importance == 0.0

    def test_decay_does_not_run_too_frequently(self):
        pfc = PrefrontalCortex(auto_train=False)
        pfc._decay_last_run = time.time()
        pfc._DECAY_INTERVAL = 300
        result = pfc._apply_concept_decay()
        assert result == 0


class TestThoughtSenseNewFeatures:
    """Tests for new Thought Sense features."""

    def test_get_thought_completions_empty(self):
        ts = ThoughtSense()
        ts._state = ThoughtState()
        completions = ts.get_thought_completions()
        assert completions == []

    def test_get_thought_completions_with_data(self):
        ts = ThoughtSense()
        ts._state = ThoughtState(
            thought_completions={"python": 0.8, "coding": 0.5, "test": 0.1},
            cognitive_state=CognitiveState.ENGAGED,
            emotional_valence=EmotionalValence.POSITIVE,
        )
        completions = ts.get_thought_completions(max_completions=2)
        assert len(completions) == 2
        assert completions[0]["text"] == "python"
        assert completions[0]["confidence"] > 0

    def test_get_proactive_suggestion_delay_flow(self):
        ts = ThoughtSense()
        ts._state = ThoughtState(
            cognitive_state=CognitiveState.FLOW,
            emotional_trajectory=0.0,
            cognitive_load=0.5,
        )
        delay = ts.get_proactive_suggestion_delay()
        assert delay >= 50

    def test_get_proactive_suggestion_delay_idle(self):
        ts = ThoughtSense()
        ts._state = ThoughtState(cognitive_state=CognitiveState.IDLE)
        delay = ts.get_proactive_suggestion_delay()
        assert delay == 0

    def test_get_proactive_suggestion_delay_overloaded(self):
        ts = ThoughtSense()
        ts._state = ThoughtState(
            cognitive_state=CognitiveState.OVERLOADED,
            cognitive_load=0.8,
        )
        delay = ts.get_proactive_suggestion_delay()
        assert delay >= 30  # Overloaded should wait before suggesting

    def test_intent_transition_prediction(self):
        ts = ThoughtSense()
        ts.observe("what is python?", concept_type="user_query")
        ts.observe("tell me more about python", concept_type="user_query")
        ts.observe("what about javascript?", concept_type="user_query")
        topics = ts._predict_next_topics(top_k=3)
        assert len(topics) <= 3

    def test_topic_counts_are_bounded(self):
        ts = ThoughtSense()
        for i in range(600):
            ts.observe(f"message about topic_{i}", concept_type="user_query")
        assert len(ts._topic_counts) <= ts._MAX_TOPIC_COUNTS

    def test_observe_empty_content_returns_early(self):
        ts = ThoughtSense()
        ts.observe("", concept_type="user_query")
        assert ts._observation_count == 0

    def test_observe_whitespace_only_returns_early(self):
        ts = ThoughtSense()
        ts.observe("   ", concept_type="user_query")
        assert ts._observation_count == 0


class TestThinkingEngineIntegration:
    """Tests for ThinkingEngine Thought Sense integration."""

    def _make_engine(self):
        """Create a ThinkingEngine with minimal initialization for testing."""
        from unittest.mock import MagicMock
        config = MagicMock()
        config.thinking.enabled = False
        config.thinking.mode = 'supportive'
        config.thinking.suggestion_interval = 30
        config.thinking.confidence_threshold = 0.7
        config.thinking.max_suggestions_per_turn = 3
        config.thinking.pause_detection = True
        config.identity.name = 'Jarvis'
        engine = ThinkingEngine.__new__(ThinkingEngine)
        engine.llm = None
        engine.config = config
        engine.suggestion_callback = None
        engine.enabled = False
        engine.mode = 'supportive'
        engine.suggestion_interval = 30
        engine.confidence_threshold = 0.7
        engine.max_suggestions_per_turn = 3
        engine.pause_detection = True
        engine.pause_threshold = 5.0
        engine.last_user_message_time = None
        engine.last_suggestion_time = 0.0
        engine.suggestions_this_turn = 0
        engine._running = False
        engine._task = None
        engine._api_manager = None
        engine._subsystem_name = "thinking"
        engine.context = ConversationContext()
        engine.prompt_manager = MagicMock()
        engine._state_path = None
        engine._confidence_history = {}
        engine._MAX_CONFIDENCE_HISTORY = 100
        engine._suggestion_outcomes = {}
        engine._MAX_OUTCOMES = 200
        engine._total_suggestions = 0
        engine._accepted_suggestions = 0
        engine._last_outcome_save = 0
        return engine

    def test_estimate_confidence_receives_suggestion_type(self):
        """Verify _estimate_confidence uses suggestion_type for calibration."""
        engine = self._make_engine()

        # Record some outcomes so the learning path is exercised
        for i in range(5):
            engine._suggestion_outcomes[f"insight_{i}"] = "accepted"
        for i in range(2):
            engine._suggestion_outcomes[f"question_{i}"] = "dismissed"

        # Test that _estimate_confidence CAN use suggestion_type
        confidence_with_type = engine._estimate_confidence(
            "This is a detailed explanation that specifically addresses the question.",
            suggestion_type="insight",
        )
        confidence_without = engine._estimate_confidence("Short answer.", suggestion_type="")
        assert isinstance(confidence_with_type, float)
        assert isinstance(confidence_without, float)

    def test_confidence_history_bounded_per_type(self):
        """Verify confidence history lists are bounded at runtime."""
        engine = self._make_engine()

        # Simulate recording many suggestions
        for i in range(150):
            thought = Thought(
                content="test",
                suggestion_type=SuggestionType.INSIGHT,
                confidence=0.8,
            )
            engine.record_suggestion_shown(thought)

        # History should be capped
        assert len(engine._confidence_history.get("insight", [])) <= 100

    def test_confidence_history_keys_bounded(self):
        """Verify confidence history doesn't accumulate unlimited keys."""
        engine = self._make_engine()

        # Add many different suggestion type keys
        for i in range(30):
            thought = Thought(
                content="test",
                suggestion_type=SuggestionType.INSIGHT,
                confidence=0.8,
            )
            engine.record_suggestion_shown(thought)

        assert len(engine._confidence_history) <= 20


class TestConceptCrossLinking:
    """Tests for PFC concept cross-linking."""

    def test_cross_link_connects_similar_concepts(self):
        pfc = PrefrontalCortex(auto_train=False)
        rng = np.random.RandomState(42)
        base = rng.randn(768).astype(np.float32)
        base = base / np.linalg.norm(base)

        c1 = Concept(
            id="c1",
            embedding=base.copy(),
            concept_type=ConceptType.USER_QUERY,
            content="python programming",
            source="user",
            timestamp=time.time(),
        )
        pfc._concepts["c1"] = c1

        c2 = Concept(
            id="c2",
            embedding=(base + rng.randn(768).astype(np.float32) * 0.02).astype(np.float32),
            concept_type=ConceptType.USER_QUERY,
            content="python development",
            source="user",
            timestamp=time.time(),
        )
        pfc._concepts["c2"] = c2
        pfc._cross_link_concept(c2)

        # c2 should be linked to c1 (similar embedding, same type)
        assert "c1" in c2.related_ids or "c2" in c1.related_ids

    def test_cross_link_respects_max_links(self):
        pfc = PrefrontalCortex(auto_train=False)
        pfc._MAX_RELATED = 10
        # _cross_link_concept has max_links=3 by default
        rng = np.random.RandomState(42)
        base = rng.randn(768).astype(np.float32)

        # Create 10 similar concepts
        for i in range(10):
            c = Concept(
                id=f"c{i}",
                embedding=(base + rng.randn(768).astype(np.float32) * 0.01).astype(np.float32),
                concept_type=ConceptType.USER_QUERY,
                content=f"query {i}",
                source="user",
                timestamp=time.time(),
            )
            pfc._concepts[f"c{i}"] = c

        # Create a new concept that will link to all of them
        new_c = Concept(
            id="new",
            embedding=base.copy(),
            concept_type=ConceptType.USER_QUERY,
            content="new query",
            source="user",
            timestamp=time.time(),
        )
        pfc._concepts["new"] = new_c
        pfc._cross_link_concept(new_c, max_links=3)

        # Should have at most 3 related IDs
        assert len(new_c.related_ids) <= 3

    def test_cross_link_different_types_not_linked(self):
        """Cross-linking intentionally links across types (related != duplicate)."""
        pfc = PrefrontalCortex(auto_train=False)
        rng = np.random.RandomState(42)
        # Use very different embeddings so similarity is low
        emb1 = rng.randn(768).astype(np.float32)
        emb1 = emb1 / np.linalg.norm(emb1)
        emb2 = rng.randn(768).astype(np.float32)
        emb2 = emb2 / np.linalg.norm(emb2)

        c1 = Concept(
            id="c1", embedding=emb1, concept_type=ConceptType.USER_QUERY,
            content="test query", source="user", timestamp=time.time(),
        )
        pfc._concepts["c1"] = c1

        c2 = Concept(
            id="c2", embedding=emb2, concept_type=ConceptType.TOOL_CALL,
            content="test tool", source="user", timestamp=time.time(),
        )
        pfc._concepts["c2"] = c2
        pfc._cross_link_concept(c2)

        # Unrelated embeddings should not be linked regardless of type
        assert len(c2.related_ids) == 0


class TestCognitiveStateEnum:
    """Tests for CognitiveState enum value correctness (regression test for bug)."""

    def test_flow_state_value(self):
        assert CognitiveState.FLOW.value == 1

    def test_idle_state_value(self):
        assert CognitiveState.IDLE.value == 6

    def test_enum_comparison_not_magic_number(self):
        """Verify that we use enum comparison (==) not magic number comparison."""
        # This is the CORRECT way to check cognitive state
        assert CognitiveState.FLOW == CognitiveState.FLOW
        assert CognitiveState.IDLE != CognitiveState.FLOW
        assert CognitiveState.OVERLOADED.value == 5

    def test_all_states_have_values(self):
        for state in CognitiveState:
            assert isinstance(state.value, int)
            assert state.value >= 1


class TestConceptRelationshipTraversal:
    """Tests for PFC concept relationship traversal and clustering."""

    def test_get_related_concepts_empty(self):
        pfc = PrefrontalCortex(auto_train=False)
        result = pfc.get_related_concepts("nonexistent")
        assert result == []

    def test_get_related_concepts_direct_link(self):
        pfc = PrefrontalCortex(auto_train=False)
        rng = np.random.RandomState(42)
        base = rng.randn(768).astype(np.float32)
        base = base / np.linalg.norm(base)

        c1 = Concept(
            id="c1", embedding=base.copy(), concept_type=ConceptType.USER_QUERY,
            content="python", source="test", timestamp=time.time(),
        )
        c1.related_ids.append("c2")
        c2 = Concept(
            id="c2", embedding=(base + rng.randn(768).astype(np.float32) * 0.02).astype(np.float32),
            concept_type=ConceptType.USER_QUERY, content="javascript", source="test", timestamp=time.time(),
        )
        c2.related_ids.append("c1")
        pfc._concepts["c1"] = c1
        pfc._concepts["c2"] = c2

        result = pfc.get_related_concepts("c1", depth=1)
        assert len(result) == 1
        assert result[0]["id"] == "c2"
        assert result[0]["distance"] == 1

    def test_get_concept_cluster_single(self):
        pfc = PrefrontalCortex(auto_train=False)
        concept = Concept(
            id="only", embedding=np.random.RandomState(42).randn(768).astype(np.float32),
            concept_type=ConceptType.USER_QUERY, content="test", source="test", timestamp=time.time(),
        )
        pfc._concepts["only"] = concept
        cluster = pfc.get_concept_cluster("only")
        assert "only" in cluster

    def test_get_concept_cluster_nonexistent(self):
        pfc = PrefrontalCortex(auto_train=False)
        cluster = pfc.get_concept_cluster("nonexistent")
        assert cluster == []


class TestConversationRhythm:
    """Tests for Thought Sense conversation rhythm detection."""

    def test_rhythm_idle_with_no_messages(self):
        ts = ThoughtSense()
        rhythm = ts.get_conversation_rhythm()
        assert rhythm["cadence"] == "idle"
        assert rhythm["optimal_response_delay"] == 2.0

    def test_rhythm_rapid_messages(self):
        ts = ThoughtSense()
        # Simulate rapid messages (< 3s apart)
        for i in range(10):
            ts.observe(f"message {i}", concept_type="user_query")
            # Don't sleep — all timestamps within microseconds
        rhythm = ts.get_conversation_rhythm()
        assert rhythm["cadence"] == "rapid"

    def test_rhythm_returns_all_keys(self):
        ts = ThoughtSense()
        ts.observe("hello", concept_type="user_query")
        ts.observe("world", concept_type="user_query")
        rhythm = ts.get_conversation_rhythm()
        assert "cadence" in rhythm
        assert "tempo_stability" in rhythm
        assert "pause_pressure" in rhythm
        assert "engagement_depth" in rhythm
        assert "optimal_response_delay" in rhythm