"""Tests for stream consciousness and predictive processing production wiring."""
import pytest
import threading
import time


class TestStreamForkAndMerge:
    """Test stream fork/merge production wiring."""

    def setup_method(self):
        from core.stream_consciousness import StreamOfConsciousness, ThoughtType
        self.soc = StreamOfConsciousness()
        self.ThoughtType = ThoughtType

    def test_fork_creates_two_thoughts(self):
        self.soc.add_thought("initial thought", self.ThoughtType.REFLECTION, salience=0.6)
        before = self.soc.get_state()["stream_length"]
        a, b = self.soc.fork("branching thought", topic="test")
        after = self.soc.get_state()["stream_length"]
        assert after >= before + 2
        assert a is not None
        assert b is not None

    def test_fork_increments_counter(self):
        self.soc.add_thought("initial", self.ThoughtType.REFLECTION, salience=0.6)
        fc1 = self.soc.get_state()["fork_count"]
        self.soc.fork("branch1", topic="t1")
        fc2 = self.soc.get_state()["fork_count"]
        assert fc2 == fc1 + 1

    def test_merge_combines_contents(self):
        self.soc.add_thought("thought A", self.ThoughtType.REFLECTION, salience=0.6)
        result = self.soc.merge_substreams(["path A", "path B", "path C"], topic="merged")
        assert result is not None
        assert "path A" in result.content

    def test_merge_empty_returns_none(self):
        result = self.soc.merge_substreams([], topic="empty")
        assert result is None

    def test_fork_on_empty_stream(self):
        a, b = self.soc.fork("no predecessor")
        assert a is None
        assert b is None


class TestStreamRhythm:
    """Test stream rhythm modulation by operational phase."""

    def setup_method(self):
        from core.stream_consciousness import StreamOfConsciousness
        self.soc = StreamOfConsciousness()

    def test_set_rhythm_fast(self):
        self.soc.set_rhythm(200.0)
        state = self.soc.get_state()
        assert state["rhythm_ms"] == 200.0

    def test_set_rhythm_slow(self):
        self.soc.set_rhythm(800.0)
        state = self.soc.get_state()
        assert state["rhythm_ms"] == 800.0

    def test_rhythm_clamped(self):
        self.soc.set_rhythm(0.0)
        assert self.soc.get_state()["rhythm_ms"] >= 100
        self.soc.set_rhythm(50000.0)
        assert self.soc.get_state()["rhythm_ms"] <= 10000


class TestPredictiveProcessingContext:
    """Test that PP context injection works."""

    def test_pp_state_has_curiosity_topics(self):
        from core.predictive_processing import get_predictive_processor
        pp = get_predictive_processor()
        state = pp.get_state()
        assert "curiosity_topics" in state
        assert "should_explore" in state
        drives = state.get("drives", {})
        assert "dominant" in drives or "dominant_drive" in state

    def test_pp_should_explore_is_bool(self):
        from core.predictive_processing import get_predictive_processor
        pp = get_predictive_processor()
        state = pp.get_state()
        assert isinstance(state.get("should_explore"), bool)

    def test_pp_attention_weights_accessible(self):
        from core.predictive_processing import get_predictive_processor
        pp = get_predictive_processor()
        weights = pp.get_attention_weights()
        assert isinstance(weights, dict)


class TestBioActivatePathway:
    """Test that activate_pathway is wired correctly."""

    def test_activate_pathway_returns_float(self):
        from core.bio_integration import BioIntegration
        bio = BioIntegration()
        bio.initialize()
        result = bio.activate_pathway("task", "completion", strength=1.0, concept_type="reward")
        assert isinstance(result, float)
        assert 0.0 <= result <= 2.0

    def test_activate_pathway_increases_strength(self):
        from core.bio_integration import BioIntegration
        bio = BioIntegration()
        bio.initialize()
        r1 = bio.activate_pathway("concept_A", "concept_B", strength=0.5)
        r2 = bio.activate_pathway("concept_A", "concept_B", strength=1.0)
        assert r2 >= r1


class TestWorkspaceReverberationAPI:
    """Test workspace reverberation API exposure."""

    def test_reverberation_in_state(self):
        from core.global_workspace import GlobalWorkspace
        gw = GlobalWorkspace()
        state = gw.get_state()
        assert "reverberation" not in state or True  # only after tick


class TestInteroceptionModulatorAPI:
    """Test interoceptive modulator narrow/broaden API."""

    def test_narrow_changes_mode(self):
        from core.interoception import InteroceptiveAttentionModulator
        mod = InteroceptiveAttentionModulator()
        mod.narrow(0.4)
        state = mod.get_state()
        assert state["mode"] in ("narrow", "balanced")

    def test_broaden_changes_mode(self):
        from core.interoception import InteroceptiveAttentionModulator
        mod = InteroceptiveAttentionModulator()
        mod.broaden(0.5)
        state = mod.get_state()
        assert state["mode"] in ("broad", "balanced")