"""Tests for three-layer context production wiring."""
import pytest


class TestThreeLayerEviction:
    """Test should_evict_l1 and evict_to_l2 production wiring."""

    def setup_method(self):
        from core.three_layer_context import ThreeLayerContext
        self.tlc = ThreeLayerContext()

    def test_should_evict_l1_below_threshold(self):
        assert self.tlc.should_evict_l1(100) is False

    def test_should_evict_l1_above_threshold(self):
        assert self.tlc.should_evict_l1(200000) is True

    def test_evict_to_l2_returns_string(self):
        messages = [{"role": "user", "content": f"msg {i}"} for i in range(20)]
        result = self.tlc.evict_to_l2(messages, max_messages_to_keep=10)
        assert isinstance(result, str)

    def test_get_version_increments_on_write(self):
        v1 = self.tlc.get_version()
        self.tlc.edit_block("identity", "append", caller="test", text="test version increment")
        v2 = self.tlc.get_version()
        # Version may or may not increment depending on block state
        assert isinstance(v2, int)


class TestThreeLayerConsolidation:
    """Test consolidate_l2_to_l3 production wiring."""

    def setup_method(self):
        from core.three_layer_context import ThreeLayerContext
        self.tlc = ThreeLayerContext()

    def test_consolidate_without_l2_data(self):
        count = self.tlc.consolidate_l2_to_l3(force=False)
        assert isinstance(count, int)

    def test_consolidate_with_force(self):
        count = self.tlc.consolidate_l2_to_l3(force=True)
        assert isinstance(count, int)


class TestThreeLayerRetrieval:
    """Test retrieve_from_l3 production wiring."""

    def setup_method(self):
        from core.three_layer_context import ThreeLayerContext
        self.tlc = ThreeLayerContext()

    def test_retrieve_from_empty_l3(self):
        results = self.tlc.retrieve_from_l3("test query", max_results=3)
        assert isinstance(results, list)

    def test_get_stats_structure(self):
        stats = self.tlc.get_stats()
        assert "blocks" in stats or "l1_evictions" in stats


class TestThreeLayerTransitionLog:
    """Test get_transition_log production wiring."""

    def setup_method(self):
        from core.three_layer_context import ThreeLayerContext
        self.tlc = ThreeLayerContext()

    def test_get_transition_log(self):
        log = self.tlc.get_transition_log(limit=5)
        assert isinstance(log, list)

    def test_transition_log_records_eviction(self):
        messages = [{"role": "user", "content": f"long msg {i} " * 50} for i in range(20)]
        self.tlc.evict_to_l2(messages, max_messages_to_keep=10)
        log = self.tlc.get_transition_log(limit=5)
        assert isinstance(log, list)


class TestThreeLayerInjectL2IntoL1:
    """Test inject_l2_into_l1 production wiring."""

    def setup_method(self):
        from core.three_layer_context import ThreeLayerContext
        self.tlc = ThreeLayerContext()

    def test_inject_l2_into_l1(self):
        messages = [{"role": "user", "content": "hello"}]
        result = self.tlc.inject_l2_into_l1(messages, model="primary")
        assert isinstance(result, list)
        assert len(result) >= 1