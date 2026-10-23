"""Tests for memory cleanup module."""
import asyncio
import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock


class TestMemoryManager:
    """Test MemoryManager functionality."""

    def test_import(self):
        """Test module imports correctly."""
        from core.memory_cleanup import MemoryManager
        assert MemoryManager is not None

    def test_get_memory_pressure(self):
        """Test memory pressure detection."""
        from core.memory_cleanup import MemoryManager, MemoryPressureLevel

        mm = MemoryManager()
        level, free_mb = mm.get_memory_pressure()

        assert isinstance(level, MemoryPressureLevel)
        assert isinstance(free_mb, int)
        assert free_mb >= 0

    def test_register_cleanup_handler(self):
        """Test cleanup handler registration."""
        from core.memory_cleanup import MemoryManager, CleanupPriority

        mm = MemoryManager()

        handler_called = []

        def test_handler():
            handler_called.append(True)
            return {"items_removed": 5, "bytes_freed": 1024, "cleaned": True}

        mm.register_cleanup_handler(test_handler, CleanupPriority.MEDIUM)

        assert len(mm._cleanup_handlers) == 1

    def test_register_pressure_callback(self):
        """Test pressure callback registration."""
        from core.memory_cleanup import MemoryManager

        mm = MemoryManager()

        callback_called = []

        def test_callback(level, free_mb):
            callback_called.append((level, free_mb))

        mm.register_pressure_callback(test_callback)

        assert len(mm._pressure_callbacks) == 1

    @pytest.mark.asyncio
    async def test_run_cleanup(self):
        """Test running cleanup."""
        from core.memory_cleanup import MemoryManager, CleanupPriority

        mm = MemoryManager()

        handler_results = []

        async def test_handler():
            handler_results.append("cleaned")
            return {"items_removed": 10, "bytes_freed": 2048, "cleaned": True}

        mm.register_cleanup_handler(test_handler, CleanupPriority.MEDIUM)

        result = await mm.run_cleanup(reason="test")

        assert result["items_removed"] == 10
        assert len(handler_results) == 1

    @pytest.mark.asyncio
    async def test_cleanup_priority_filtering(self):
        """Test cleanup respects priority."""
        from core.memory_cleanup import MemoryManager, CleanupPriority

        mm = MemoryManager()

        calls = []

        def low_handler():
            calls.append("low")
            return {"items_removed": 1, "bytes_freed": 100, "cleaned": True}

        def high_handler():
            calls.append("high")
            return {"items_removed": 1, "bytes_freed": 100, "cleaned": True}

        mm.register_cleanup_handler(low_handler, CleanupPriority.LOW)
        mm.register_cleanup_handler(high_handler, CleanupPriority.HIGH)

        await mm.run_cleanup(reason="test", priority=CleanupPriority.HIGH)

        assert "high" in calls
        assert "low" not in calls

    def test_get_stats(self):
        """Test statistics retrieval."""
        from core.memory_cleanup import MemoryManager

        mm = MemoryManager()

        stats = mm.get_stats()

        assert "total_items_removed" in stats
        assert "current_pressure" in stats

    @pytest.mark.asyncio
    async def test_start_stop(self):
        """Test starting and stopping."""
        from core.memory_cleanup import MemoryManager

        mm = MemoryManager()

        await mm.start()
        assert mm._running is True

        await mm.stop()
        assert mm._running is False


class TestCleanupHandlers:
    """Test cleanup handler factories."""

    def test_conversation_cleanup_handler(self):
        """Test conversation cleanup handler."""
        from core.memory_cleanup import create_conversation_cleanup_handler

        mock_memory = MagicMock()
        mock_memory.history = [1, 2, 3, 4, 5]
        mock_memory.important_messages = {0: 0.9, 1: 0.5, 10: 0.3}
        mock_memory._trim_history = MagicMock()
        mock_memory._trim_if_needed = MagicMock()

        handler = create_conversation_cleanup_handler(mock_memory)
        result = handler()

        # Either _trim_history or _trim_if_needed should be called
        assert mock_memory._trim_history.called or mock_memory._trim_if_needed.called
        assert "items_removed" in result

    @pytest.mark.asyncio
    async def test_cache_cleanup_handler(self):
        """Test cache cleanup handler."""
        from core.memory_cleanup import create_cache_cleanup_handler

        mock_cache = MagicMock()
        mock_cache._cache = {}
        mock_cache._lock = asyncio.Lock()
        mock_cache.ttl_seconds = 300

        handler = create_cache_cleanup_handler(mock_cache)
        result = await handler()

        assert "items_removed" in result
        assert "bytes_freed" in result

    def test_file_cleanup_handler(self):
        """Test file cleanup handler."""
        from core.memory_cleanup import create_file_cleanup_handler

        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir)

            old_file = test_dir / "old.txt"
            old_file.write_text("old content", encoding='utf-8')
            import os
            old_time = time.time() - 3600 * 48
            os.utime(old_file, (old_time, old_time))

            new_file = test_dir / "new.txt"
            new_file.write_text("new content", encoding='utf-8')

            handler = create_file_cleanup_handler(test_dir, max_age_hours=24)
            result = handler()

            assert result["items_removed"] >= 1
            assert not old_file.exists()
            assert new_file.exists()

    def test_log_cleanup_handler(self):
        """Test log cleanup handler."""
        from core.memory_cleanup import create_log_cleanup_handler

        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)

            for i in range(15):
                log_file = log_dir / f"jarvis_{i:03d}.log"
                log_file.write_text(f"log {i}", encoding='utf-8')

            handler = create_log_cleanup_handler(log_dir, max_files=10, max_age_days=7)
            result = handler()

            remaining = list(log_dir.glob("*.log"))
            assert len(remaining) <= 10


class TestIntegration:
    """Integration tests."""

    def test_get_memory_manager_singleton(self):
        """Test memory manager singleton."""
        from core.memory_cleanup import get_memory_manager

        mm1 = get_memory_manager()
        mm2 = get_memory_manager()

        assert mm1 is mm2

    @pytest.mark.asyncio
    async def test_pressure_level_enum(self):
        """Test pressure level enum values."""
        from core.memory_cleanup import MemoryPressureLevel

        assert MemoryPressureLevel.NONE.value == "none"
        assert MemoryPressureLevel.LOW.value == "low"
        assert MemoryPressureLevel.MEDIUM.value == "medium"
        assert MemoryPressureLevel.HIGH.value == "high"
        assert MemoryPressureLevel.CRITICAL.value == "critical"


class TestCleanupPriority:
    """Test cleanup priority ordering."""

    def test_priority_ordering(self):
        """Test priorities have correct ordering."""
        from core.memory_cleanup import CleanupPriority

        assert CleanupPriority.LOW.value < CleanupPriority.MEDIUM.value
        assert CleanupPriority.MEDIUM.value < CleanupPriority.HIGH.value
        assert CleanupPriority.HIGH.value < CleanupPriority.CRITICAL.value
