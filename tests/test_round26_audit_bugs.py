"""Round 26 Audit Bug Tests.

Tests for bugs found and fixed in Round 26 of the deep audit:
1. unified_context.py: Broken double-checked locking singleton pattern
2. docker_sandbox.py: _max_logs AttributeError at init (order bug)
3. auth_middleware.py: _blocked_ips unbounded set
4. memory_consolidation.py: Late deque import, unbounded dicts, thread safety
5. concept_model.py: _request_history unbounded list
6. cognitive.py: _temporal_index unbounded list, connections growing without bounds
7. permissions.py: session_permissions unbounded dict
"""
import threading
import time
import pytest


class TestUnifiedContextSingleton:
    """Test that get_unified_context uses proper double-checked locking."""

    def test_singleton_returns_same_instance(self):
        from core.unified_context import get_unified_context
        import core.unified_context as uc_module

        original = uc_module._unified_context
        uc_module._unified_context = None

        ctx1 = get_unified_context()
        ctx2 = get_unified_context()
        assert ctx1 is ctx2

        uc_module._unified_context = original

    def test_singleton_thread_safety(self):
        """Test that concurrent access doesn't create multiple instances."""
        from core.unified_context import get_unified_context
        import core.unified_context as uc_module

        original = uc_module._unified_context
        uc_module._unified_context = None

        results = []
        barrier = threading.Barrier(8)

        def get_ctx():
            barrier.wait(timeout=5)
            ctx = get_unified_context()
            results.append(id(ctx))

        threads = [threading.Thread(target=get_ctx) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(set(results)) == 1, "Multiple instances created under concurrent access"

        uc_module._unified_context = original




class TestAuthMiddlewareBlockedIPsBound:
    """Test auth_middleware.py _blocked_ips is bounded."""

    def test_blocked_ips_is_bounded(self):
        from core.auth_middleware import SecurityMiddleware
        auth = SecurityMiddleware()
        assert hasattr(auth, '_MAX_BLOCKED_IPS')

        for i in range(auth._MAX_BLOCKED_IPS + 100):
            auth.block_ip(f"192.168.{i // 256}.{i % 256}")

        assert len(auth._blocked_ips) <= auth._MAX_BLOCKED_IPS


class TestMemoryConsolidationBound:
    """Test memory_consolidation.py bounded collections and thread safety."""

    def test_memory_weights_is_ordered_dict(self):
        from core.memory_consolidation import MemoryConsolidationEngine
        from collections import OrderedDict
        engine = MemoryConsolidationEngine()
        assert isinstance(engine._memory_weights, OrderedDict)

    def test_memory_weights_bounded(self):
        from core.memory_consolidation import MemoryConsolidationEngine
        engine = MemoryConsolidationEngine()
        max_w = engine._MAX_MEMORY_WEIGHTS

        for i in range(max_w + 100):
            engine.register_memory(f"mem_{i}", weight=0.5)

        assert len(engine._memory_weights) <= max_w

    def test_access_counts_safe_on_missing_key(self):
        from core.memory_consolidation import MemoryConsolidationEngine
        engine = MemoryConsolidationEngine()

        engine.register_memory("test_1", weight=0.5)
        engine.register_memory("test_1", weight=0.6)
        assert engine._access_counts.get("test_1", 0) == 2

        engine.record_access("nonexistent")
        assert engine._access_counts.get("nonexistent", 0) == 1

    def test_deque_imported_at_top(self):
        import core.memory_consolidation as mc
        assert hasattr(mc, 'deque')

    def test_concurrent_register_memory(self):
        """Test concurrent additions don't cause errors."""
        from core.memory_consolidation import MemoryConsolidationEngine
        engine = MemoryConsolidationEngine()
        errors = []

        def register_batch(start):
            try:
                for i in range(100):
                    engine.register_memory(f"concurrent_{start}_{i}", weight=0.5)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=register_batch, args=(j,)) for j in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0
        assert len(engine._memory_weights) <= engine._MAX_MEMORY_WEIGHTS


class TestConceptModelRequestHistory:
    """Test concept_model.py _request_history is bounded deque."""

    def test_request_history_is_bounded_deque(self):
        from core.concept_model import ConceptModelOrchestrator
        from collections import deque
        orchestrator = ConceptModelOrchestrator()
        assert isinstance(orchestrator._request_history, deque)
        assert orchestrator._request_history.maxlen == 100




class TestPermissionsSessionBounds:
    """Test permissions.py session_permissions is bounded."""

    def test_session_permissions_is_ordered_dict(self):
        from core.permissions import PermissionManager
        from collections import OrderedDict
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PermissionManager(Path(tmpdir))
            assert isinstance(pm.session_permissions, OrderedDict)

    def test_session_permissions_bounded(self):
        from core.permissions import PermissionManager, PermissionLevel
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PermissionManager(Path(tmpdir))
            max_perm = pm._MAX_SESSION_PERMISSIONS

            for i in range(max_perm + 100):
                pm.grant_permission(f"action_{i}", PermissionLevel.ALLOW_SESSION)

            assert len(pm.session_permissions) <= max_perm


class TestUnifiedContextStructure:
    """Test unified_context structure is correct."""

    def test_context_layers_exist(self):
        from core.unified_context import UnifiedContext
        ctx = UnifiedContext()
        assert "working" in ctx.layers
        assert "project" in ctx.layers
        assert "knowledge" in ctx.layers
        assert "episodic" in ctx.layers
        assert "session" in ctx.layers
        assert "semantic" in ctx.layers

    def test_truncate_messages_respects_budget(self):
        from core.unified_context import UnifiedContext
        ctx = UnifiedContext()
        messages = [
            {"role": "system", "content": "Short"},
            {"role": "system", "content": "A" * 10000},
        ]
        truncated = ctx._truncate_messages(messages, max_tokens=100)
        total_chars = sum(len(m.get("content", "")) for m in truncated)
        assert total_chars <= 100 * 4 + 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])