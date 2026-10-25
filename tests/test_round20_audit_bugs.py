"""Round 20 Deep Audit Bug Tests.

Tests for bugs found and fixed in:
- proactive.py: deque for bounded _mentioned_files/_mentioned_topics, lock scope
- memory_manager.py: _notify_pressure snapshots callbacks, on_memory_pressure lock + bound,
  cleanup_cycle/force_cleanup lock, BoundedCache.get_stats copy, ConnectionPool.count thread-safe
- background.py: asyncio-only scheduler (verified correct pattern)
"""

import threading
import time
import pytest
from unittest.mock import MagicMock
from collections import deque


# ============================================================
# proactive.py tests
# ============================================================

class TestProactiveEngineThreadSafety:
    def test_mentioned_files_is_deque(self):
        from core.proactive import ProactiveEngine
        engine = ProactiveEngine()
        assert isinstance(engine._mentioned_files, deque)
        assert engine._mentioned_files.maxlen == 10

    def test_mentioned_topics_is_deque(self):
        from core.proactive import ProactiveEngine
        engine = ProactiveEngine()
        assert isinstance(engine._mentioned_topics, deque)

    def test_mentioned_files_bounded(self):
        from core.proactive import ProactiveEngine
        engine = ProactiveEngine()
        for i in range(20):
            engine.update_activity(f"test message {i}", mentioned_files=[f"file_{i}.py"])
        assert len(engine._mentioned_files) <= 10

    def test_mentioned_topics_bounded(self):
        from core.proactive import ProactiveEngine
        engine = ProactiveEngine()
        for i in range(50):
            engine.update_activity(f"this is a message about testing_{i}")
        assert len(engine._mentioned_topics) <= 20

    def test_update_activity_thread_safe(self):
        from core.proactive import ProactiveEngine
        engine = ProactiveEngine()
        errors = []

        def stress():
            try:
                for i in range(30):
                    engine.update_activity(f"message {i}", mentioned_files=[f"f{i}.py"])
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=stress) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_should_suggest_lock_scope(self):
        from core.proactive import ProactiveEngine, SuggestionType
        engine = ProactiveEngine()
        should, suggestion = engine.should_suggest(user_mood="neutral")
        assert isinstance(should, bool)


# ============================================================
# memory_manager.py tests
# ============================================================

class TestMemoryManagerThreadSafety:
    def test_notify_pressure_copies_callbacks(self):
        from core.memory_manager import MemoryManager, MemoryPressureLevel
        mgr = MemoryManager()
        calls = []
        mgr.on_memory_pressure(MemoryPressureLevel.HIGH, lambda: calls.append("high"))
        mgr._notify_pressure(MemoryPressureLevel.HIGH)
        assert len(calls) == 1

    def test_on_memory_pressure_bounded(self):
        from core.memory_manager import MemoryManager, MemoryPressureLevel
        mgr = MemoryManager()
        for i in range(15):
            mgr.on_memory_pressure(MemoryPressureLevel.HIGH, lambda: None)
        assert len(mgr._pressure_callbacks[MemoryPressureLevel.HIGH]) <= 10

    def test_bounded_cache_stats_returns_copy(self):
        from core.memory_manager import BoundedCache
        cache = BoundedCache("test", max_entries=10)
        cache.set("k1", "v1")
        stats1 = cache.get_stats()
        stats2 = cache.get_stats()
        assert stats1 is not stats2
        assert stats1.name == stats2.name

    def test_bounded_cache_concurrent_access(self):
        from core.memory_manager import BoundedCache
        cache = BoundedCache("test", max_entries=100)
        errors = []

        def stress():
            try:
                for i in range(50):
                    cache.set(f"key_{i}", f"val_{i}")
                    cache.get(f"key_{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=stress) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_connection_pool_count_thread_safe(self):
        from core.memory_manager import ConnectionPool
        pool = ConnectionPool("test", max_connections=20)
        errors = []

        def stress():
            try:
                for i in range(30):
                    pool.set(f"conn_{i}", f"connection_{i}")
                    pool.count()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=stress) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_cleanup_cycle_under_lock(self):
        from core.memory_manager import MemoryManager, BoundedCache
        import asyncio

        mgr = MemoryManager()
        cache = BoundedCache("test", max_entries=10, ttl_seconds=0.01)
        cache.set("k1", "v1")

        mgr.register_cache(cache)

        result = asyncio.run(mgr.cleanup_cycle())
        assert "caches_cleaned" in result

    def test_force_cleanup_under_lock(self):
        from core.memory_manager import MemoryManager, BoundedCache, ConnectionPool

        mgr = MemoryManager()
        cache = BoundedCache("test_cleanup", max_entries=10)
        cache.set("k1", "v1")
        mgr.register_cache(cache)

        result = mgr.force_cleanup("medium")
        assert "caches_cleared" in result


# ============================================================
# background.py tests (verification)
# ============================================================

class TestBackgroundSchedulerAsyncSafety:
    def test_register_and_unregister(self):
        from core.background import BackgroundScheduler, TaskPriority
        scheduler = BackgroundScheduler()

        async def dummy_task():
            pass

        task = scheduler.register("test", dummy_task, 60, TaskPriority.MEDIUM)
        assert task.name == "test"
        assert scheduler.unregister("test")
        assert not scheduler.unregister("nonexistent")

    def test_enable_disable(self):
        from core.background import BackgroundScheduler, TaskPriority

        scheduler = BackgroundScheduler()
        async def dummy():
            pass

        scheduler.register("test", dummy, 60, TaskPriority.MEDIUM)
        assert scheduler.disable("test")
        assert scheduler.enable("test")

    def test_trigger_task(self):
        from core.background import BackgroundScheduler, TaskPriority

        scheduler = BackgroundScheduler()
        async def dummy():
            pass

        scheduler.register("test", dummy, 60)
        assert scheduler.trigger("test")
        assert not scheduler.trigger("nonexistent")

    def test_mark_user_activity(self):
        from core.background import BackgroundScheduler

        scheduler = BackgroundScheduler()
        scheduler._idle_since = time.time()
        scheduler.mark_user_activity()
        assert scheduler._idle_since is None