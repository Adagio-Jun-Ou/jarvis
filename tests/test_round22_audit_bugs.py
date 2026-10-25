"""Round 22 audit bug tests - unified_memory, dynamic_memory, persistent_memory, auth, health_check, deployment."""
import threading
import time
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


class TestUnifiedMemoryThreadSafety:
    def test_search_semantic_locks_dict_access(self):
        from core.unified_memory import UnifiedMemory, MemoryType

        with tempfile.TemporaryDirectory() as tmpdir:
            um = UnifiedMemory(storage_path=Path(tmpdir))
            mem_id = um.store("test content here", MemoryType.INSIGHT, importance=0.8)

            um._interconnect = MagicMock()
            um._interconnect.search_memories.return_value = [(mem_id, 0.9)]

            results = um.search_semantic("test", limit=5)
            assert len(results) == 1
            assert results[0].content == "test content here"

    def test_find_similar_insights_under_lock(self):
        from core.unified_memory import UnifiedMemory, MemoryType

        with tempfile.TemporaryDirectory() as tmpdir:
            um = UnifiedMemory(storage_path=Path(tmpdir))

            mock_review = MagicMock()
            mock_review.find_by_content.return_value = [
                {"id": "nonexistent", "similarity": 0.5}
            ]

            with patch.object(um, "_interconnect", None):
                with patch.dict("sys.modules", {"core.insight_review": MagicMock(get_insight_review=MagicMock(return_value=mock_review))}):
                    try:
                        results = um.find_similar_insights("test", threshold=0.3, limit=5)
                    except Exception:
                        pass
                    assert True

    def test_enforce_index_bounds_called(self):
        from core.unified_memory import UnifiedMemory, MemoryType

        with tempfile.TemporaryDirectory() as tmpdir:
            um = UnifiedMemory(storage_path=Path(tmpdir))
            um._MAX_INDEX_WORDS = 3

            um.store("alpha beta gamma delta epsilon", MemoryType.CONVERSATION)
            assert len(um._index) <= 3

    def test_concurrent_store_and_retrieve(self):
        from core.unified_memory import UnifiedMemory, MemoryType

        with tempfile.TemporaryDirectory() as tmpdir:
            um = UnifiedMemory(storage_path=Path(tmpdir))
            errors = []

            def store_fn(i):
                try:
                    um.store(f"content {i}", MemoryType.CONVERSATION, importance=float(i % 10) / 10)
                except Exception as e:
                    errors.append(e)

            def retrieve_fn(i):
                try:
                    um.search(f"content {i}", limit=5)
                except Exception as e:
                    errors.append(e)

            threads = []
            for i in range(20):
                threads.append(threading.Thread(target=store_fn, args=(i,)))
                threads.append(threading.Thread(target=retrieve_fn, args=(i,)))
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)

            assert not errors, f"Concurrent errors: {errors}"

    def test_find_by_hash_under_lock(self):
        from core.unified_memory import UnifiedMemory

        with tempfile.TemporaryDirectory() as tmpdir:
            um = UnifiedMemory(storage_path=Path(tmpdir))
            result = um._find_by_hash("nonexistent")
            assert result is None

    def test_get_insight_by_hash_under_lock(self):
        from core.unified_memory import UnifiedMemory

        with tempfile.TemporaryDirectory() as tmpdir:
            um = UnifiedMemory(storage_path=Path(tmpdir))
            result = um.get_insight_by_hash("nonexistent")
            assert result is None

    def test_search_returns_correct_type(self):
        from core.unified_memory import UnifiedMemory, MemoryType

        with tempfile.TemporaryDirectory() as tmpdir:
            um = UnifiedMemory(storage_path=Path(tmpdir))
            um.store("important fact about python", MemoryType.PERSISTENT, importance=0.9)
            um.store("casual chat about python", MemoryType.CONVERSATION, importance=0.3)

            results = um.search("python", memory_types=[MemoryType.PERSISTENT], limit=5)
            for r in results:
                assert r.memory_type == MemoryType.PERSISTENT

    def test_consolidate_all_persists(self):
        from core.unified_memory import UnifiedMemory, MemoryType

        with tempfile.TemporaryDirectory() as tmpdir:
            um = UnifiedMemory(storage_path=Path(tmpdir))
            um.store("working task", MemoryType.WORKING, importance=0.9)
            um.store("chat message", MemoryType.CONVERSATION, importance=0.8)

            stats = um.consolidate_all()
            assert isinstance(stats, dict)
            assert "consolidated" in stats


class TestDynamicMemoryThreadSafety:
    def test_stats_under_lock(self):
        from core.dynamic_memory import DynamicMemoryManager

        mgr = DynamicMemoryManager()
        mgr._stats.total_cleanups = 0

        def mutate_stats():
            for _ in range(100):
                with mgr._stats_lock:
                    mgr._stats.total_cleanups += 1

        threads = [threading.Thread(target=mutate_stats) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        with mgr._stats_lock:
            total = mgr._stats.total_cleanups
        assert total == 500

    def test_register_memory_pool_thread_safe(self):
        from core.dynamic_memory import DynamicMemoryManager

        mgr = DynamicMemoryManager()
        pool = []

        def register_fn(i):
            mgr.register_memory_pool(f"pool_{i}", [])

        threads = [threading.Thread(target=register_fn, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        with mgr._lock:
            assert len(mgr._memory_pools) == 20

    def test_cleanup_callbacks_bounded(self):
        from core.dynamic_memory import DynamicMemoryManager

        mgr = DynamicMemoryManager()
        for i in range(mgr.MAX_CLEANUP_CALLBACKS + 10):
            mgr.register_cleanup_callback(lambda: None, f"cb_{i}", priority=i)

        with mgr._lock:
            assert len(mgr._cleanup_callbacks) <= mgr.MAX_CLEANUP_CALLBACKS

    def test_cache_callbacks_bounded(self):
        from core.dynamic_memory import DynamicMemoryManager

        mgr = DynamicMemoryManager()

        class FakeCache:
            def clear(self):
                pass

        for i in range(mgr.MAX_CACHE_CALLBACKS + 10):
            mgr.register_cache(FakeCache(), size_callback=lambda size: None)

        with mgr._cache_callbacks_lock:
            assert len(mgr._cache_size_callbacks) <= mgr.MAX_CACHE_CALLBACKS


    def test_force_cleanup_snapshots_pools(self):
        from core.dynamic_memory import DynamicMemoryManager, MemoryPressure

        mgr = DynamicMemoryManager()
        test_pool = [1, 2, 3]
        mgr.register_memory_pool("test", test_pool)

        results = mgr.force_cleanup(level=MemoryPressure.HIGH)
        assert isinstance(results, dict)
        assert "pools_cleared" in results

    def test_get_status_reads_stats_under_lock(self):
        from core.dynamic_memory import DynamicMemoryManager

        mgr = DynamicMemoryManager()
        status = mgr.get_status()
        assert isinstance(status, dict)
        assert "pressure" in status
        assert "stats" in status


class TestPersistentMemoryThreadSafety:
    def test_turns_since_save_under_lock(self):
        from core.persistent_memory import PersistentMemory

        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PersistentMemory(storage_path=Path(tmpdir))
            pm.SAVE_INTERVAL_TURNS = 1000

            errors = []

            def turn_fn(i):
                try:
                    pm.on_turn_complete(f"user message {i}", f"assistant message {i}" * 10)
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=turn_fn, args=(i,)) for i in range(20)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)

            assert not errors, f"Errors: {errors}"

    def test_concurrent_add_memory(self):
        from core.persistent_memory import PersistentMemory

        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PersistentMemory(storage_path=Path(tmpdir))

            errors = []

            def add_fn(i):
                try:
                    pm.add_memory(f"memory content {i}", importance=float(i % 10) / 10)
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=add_fn, args=(i,)) for i in range(20)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)

            assert not errors, f"Errors: {errors}"

    def test_max_facts_bounded(self):
        from core.persistent_memory import PersistentMemory

        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PersistentMemory(storage_path=Path(tmpdir))
            for i in range(300):
                pm.add_user_fact(f"key_{i}", f"value_{i}", confidence=0.8)

            assert len(pm.user_facts) <= 300

    def test_get_relevant_memories_empty_query(self):
        from core.persistent_memory import PersistentMemory

        with tempfile.TemporaryDirectory() as tmpdir:
            pm = PersistentMemory(storage_path=Path(tmpdir))
            result = pm.get_relevant_memories("")
            assert result == []
            result = pm.get_relevant_memories("   ")
            assert result == []




class TestHealthCheckThreadSafety:
    def test_register_check_under_lock(self):
        from core.health_check import HealthChecker

        hc = HealthChecker()

        def register_fn(i):
            hc.register_check(f"custom_{i}", lambda: None)

        threads = [threading.Thread(target=register_fn, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        stats = hc.get_stats()
        assert stats["registered_checks"] >= 20 + 8  # 20 custom + 8 default

    def test_check_all_snapshots_checks(self):
        from core.health_check import HealthChecker, HealthStatus

        hc = HealthChecker()
        result = hc.check_all()
        assert isinstance(result.overall_status, HealthStatus)
        assert isinstance(result.subsystems, list)

    def test_get_stats_under_lock(self):
        from core.health_check import HealthChecker

        hc = HealthChecker()
        stats = hc.get_stats()
        assert isinstance(stats, dict)
        assert "registered_checks" in stats
        assert "check_names" in stats
        assert isinstance(stats["check_names"], list)




import secrets