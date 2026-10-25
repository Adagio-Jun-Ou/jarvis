"""Tests for emotional state, mood history, and sentiment trajectory production wiring."""
import pytest


class TestEmotionalStateWiring:
    """Test emotional state methods in production."""

    def setup_method(self):
        from core.emotional_state import EmotionalStateManager
        self.em = EmotionalStateManager()

    def test_should_show_empathy_default(self):
        result = self.em.should_show_empathy()
        assert isinstance(result, bool)

    def test_get_empathy_message_default(self):
        msg = self.em.get_empathy_message()
        assert msg is None or isinstance(msg, str)

    def test_get_sentiment_trajectory_empty(self):
        trajectory = self.em.get_sentiment_trajectory(limit=5)
        assert isinstance(trajectory, list)

    def test_get_mood_history_empty(self):
        history = self.em.get_mood_history(limit=5)
        assert isinstance(history, list)

    def test_summarize_emotional_state(self):
        summary = self.em.summarize_emotional_state()
        assert isinstance(summary, str)

    def test_analyze_input(self):
        from core.emotional_state import get_emotional_state_manager
        em = get_emotional_state_manager()
        signals = em.analyze_input("I'm feeling really frustrated with this problem")
        assert hasattr(signals, "sentiment")
        assert hasattr(signals, "engagement")


class TestClaimRegistryMaintenance:
    """Test claims maintenance production wiring."""

    def test_cleanup_expired_exists(self):
        from core.claims import ClaimsRegistry
        assert hasattr(ClaimsRegistry, 'cleanup_expired')

    def test_heartbeat_exists(self):
        from core.claims import ClaimsRegistry
        assert hasattr(ClaimsRegistry, 'heartbeat')


class TestObservabilityMetrics:
    """Test observability metrics API."""

    def test_get_all_metrics(self):
        from core.observability import MetricsRegistry
        obs = MetricsRegistry()
        metrics = obs.get_all_metrics()
        assert isinstance(metrics, dict)


class TestWorkspaceRhythmProduction:
    """Test stream consciousness rhythm modulation."""

    def test_rhythm_from_phase(self):
        from core.stream_consciousness import StreamOfConsciousness
        soc = StreamOfConsciousness()
        soc.set_rhythm(200.0)
        state = soc.get_state()
        assert state["rhythm_ms"] == 200.0

    def test_fork_and_merge_workflow(self):
        from core.stream_consciousness import StreamOfConsciousness, ThoughtType
        soc = StreamOfConsciousness()
        soc.add_thought("start", ThoughtType.INTENTION, salience=0.6)
        a, b = soc.fork("branch", topic="test")
        assert a is not None
        assert b is not None
        merged = soc.merge_substreams(["path A", "path B"], topic="merged", salience=0.7)
        assert merged is not None

    def test_fork_count_increases(self):
        from core.stream_consciousness import StreamOfConsciousness, ThoughtType
        soc = StreamOfConsciousness()
        soc.add_thought("start", ThoughtType.INTENTION, salience=0.6)
        fc1 = soc.get_state()["fork_count"]
        soc.fork("branch 1", topic="t1")
        fc2 = soc.get_state()["fork_count"]
        assert fc2 == fc1 + 1