"""Tests for ContextCoordinator unified context system."""
import pytest


def test_context_coordinator_import():
    """Test that context_coordinator module imports correctly."""
    from core.context_coordinator import (
        ContextCoordinator,
        ContextBootstrap,
        ContextConflictResolver,
        CacheCoordinator,
        UnifiedContextResult,
        get_context_coordinator,
        get_context_bootstrap,
        get_cache_coordinator,
    )
    
    assert ContextCoordinator is not None
    assert ContextBootstrap is not None
    assert ContextConflictResolver is not None
    assert CacheCoordinator is not None
    assert UnifiedContextResult is not None


def test_context_bootstrap_init_order():
    """Test that bootstrap has correct initialization order."""
    from core.context_coordinator import ContextBootstrap
    
    bootstrap = ContextBootstrap()
    
    # Check INIT_ORDER has expected subsystems
    names = [name for name, _, _, _ in bootstrap.INIT_ORDER]
    
    assert "embedding_interconnect" in names
    assert "memory" in names
    assert "knowledge_graph" in names
    assert "prefrontal_cortex" in names
    assert "unified_context_engine" in names
    assert "context_cascade" in names
    assert "agentic_context" in names
    
    # Check dependency order
    indices = {name: i for i, name in enumerate(names)}
    
    # PFC depends on embedding
    pfc_idx = indices["prefrontal_cortex"]
    emb_idx = indices["embedding_interconnect"]
    assert emb_idx < pfc_idx, "embedding should initialize before PFC"
    
    # Context cascade depends on PFC and engine
    cascade_idx = indices["context_cascade"]
    engine_idx = indices["unified_context_engine"]
    assert pfc_idx < cascade_idx, "PFC should initialize before cascade"
    assert engine_idx < cascade_idx, "engine should initialize before cascade"


def test_unified_context_result():
    """Test UnifiedContextResult data structure."""
    from core.context_coordinator import UnifiedContextResult
    
    result = UnifiedContextResult(
        messages=[{"role": "system", "content": "test"}],
        entries=["entry1", "entry2"],
        sources_used={"memory", "cascade"},
        total_tokens=100,
        conflicts_resolved=2,
        timing_ms={"total": 50.0, "engine": 30.0},
    )
    
    assert len(result.messages) == 1
    assert len(result.entries) == 2
    assert "memory" in result.sources_used
    assert result.total_tokens == 100
    assert result.conflicts_resolved == 2
    
    summary = result.to_summary()
    assert "memory" in summary
    assert "100" in summary  # token count


def test_conflict_resolver_precedence():
    """Test ContextConflictResolver precedence rules."""
    from core.context_coordinator import ContextConflictResolver
    
    resolver = ContextConflictResolver()
    
    # Test precedence levels
    assert resolver.get_precedence("persistent_memory") == 10
    assert resolver.get_precedence("knowledge_graph") == 8
    assert resolver.get_precedence("working_memory") == 7
    assert resolver.get_precedence("cascade") == 4
    assert resolver.get_precedence("unknown") == 0


def test_conflict_resolver_detect():
    """Test conflict detection."""
    from core.context_coordinator import ContextConflictResolver, ContextConflict
    
    resolver = ContextConflictResolver()
    
    # Create mock entries with conflicts
    class MockEntry:
        def __init__(self, key, value, source):
            self.key = key
            self.value = value
            self.source = source
    
    entries = [
        MockEntry("theme", "dark", "working_memory"),
        MockEntry("theme", "light", "persistent_memory"),
        MockEntry("theme", "system", "cascade"),
    ]
    
    conflicts = resolver.detect_conflicts(entries)
    
    assert len(conflicts) == 1
    assert conflicts[0].topic == "theme"
    assert "working_memory" in conflicts[0].sources
    assert "persistent_memory" in conflicts[0].sources


def test_cache_coordinator_registration():
    """Test CacheCoordinator cache registration."""
    from core.context_coordinator import CacheCoordinator
    
    coordinator = CacheCoordinator(max_total_mb=100)
    
    # Mock cache
    class MockCache:
        def __init__(self):
            self._data = {"a": 1, "b": 2, "c": 3}
        def __len__(self):
            return len(self._data)
        def clear(self):
            self._data.clear()
        def popitem(self, last=True):
            return self._data.popitem(last=last)
    
    cache = MockCache()
    coordinator.register("test_cache", cache, max_mb=10)
    
    stats = coordinator.get_stats()
    
    assert "test_cache" in stats["cache_details"]
    assert stats["caches"] == 1


def test_cache_coordinator_pressure():
    """Test memory pressure detection."""
    from core.context_coordinator import CacheCoordinator
    
    coordinator = CacheCoordinator(max_total_mb=1)  # Very low limit
    
    class BigCache:
        def __init__(self, size):
            self.size = size
            self._data = [i for i in range(size * 1000)]
        def __len__(self):
            return self.size
        def clear(self):
            self._data.clear()
    
    cache = BigCache(1000)  # Large cache
    coordinator.register("big_cache", cache, max_mb=50)
    
    # Should detect pressure with low limit
    assert coordinator.is_under_pressure()


def test_context_coordinator_observe():
    """Test ContextCoordinator.observe() method."""
    from core.context_coordinator import get_context_coordinator
    
    coordinator = get_context_coordinator()
    
    result = coordinator.observe(
        trigger_type="user_query",
        content="test query",
        source="user",
    )
    
    assert result is not None
    assert isinstance(result.timing_ms, dict)
    assert "total" in result.timing_ms


def test_context_coordinator_stats():
    """Test ContextCoordinator statistics."""
    from core.context_coordinator import get_context_coordinator
    
    coordinator = get_context_coordinator()
    
    stats = coordinator.get_stats()
    
    assert "subsystems_loaded" in stats
    assert "memory_pressure" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
