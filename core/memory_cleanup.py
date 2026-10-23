"""Memory Management and Cleanup System.

Coordinates memory cleanup across all Jarvis subsystems:
- Conversation memory trimming
- Archive management
- Cache cleanup
- Resource-aware cleanup scheduling
- Memory pressure response
"""
import asyncio
import gc
import logging
import threading
import time

from core.helpers import atomic_write_json, create_tracked_task, read_json_with_integrity
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


class CleanupPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class MemoryPressureLevel(Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class CleanupStats:
    items_removed: int = 0
    bytes_freed: int = 0
    collections_cleaned: int = 0
    last_cleanup: datetime | None = None
    cleanup_duration: float = 0.0


@dataclass
class MemoryThresholds:
    low_pressure_mb: int = 500
    medium_pressure_mb: int = 250
    high_pressure_mb: int = 100
    critical_pressure_mb: int = 50


class MemoryManager:
    """Central memory management and cleanup coordinator.
    
    Features:
    - Periodic cleanup scheduling
    - Memory pressure monitoring
    - Coordinated cleanup across subsystems
    - Configurable cleanup policies
    """

    DEFAULT_CLEANUP_INTERVAL = 300
    PRESSURE_CHECK_INTERVAL = 30
    MAX_HANDLERS = 50
    MAX_PRESSURE_CALLBACKS = 20

    def __init__(
        self,
        storage_path: Path | None = None,
        thresholds: MemoryThresholds | None = None,
    ):
        self.storage_path = storage_path or Path.home() / ".jarvis" / "memory_mgmt"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.thresholds = thresholds or MemoryThresholds()

        self._running = False
        self._cleanup_task: asyncio.Task | None = None
        self._pressure_task: asyncio.Task | None = None
        self._lock = threading.RLock()

        self._cleanup_handlers: list[tuple[CleanupPriority, Callable]] = []
        self._pressure_callbacks: list[Callable] = []

        self._stats = CleanupStats()
        self._last_pressure = MemoryPressureLevel.NONE

        self._psutil = None
        try:
            import psutil
            self._psutil = psutil
        except ImportError:
            pass

        self._load_config()

    def _load_config(self) -> None:
        """Load cleanup configuration."""
        config_file = self.storage_path / "cleanup_config.json"
        if config_file.exists():
            try:
                data = read_json_with_integrity(config_file)
                self.thresholds = MemoryThresholds(
                    low_pressure_mb=data.get("low_pressure_mb", 500),
                    medium_pressure_mb=data.get("medium_pressure_mb", 250),
                    high_pressure_mb=data.get("high_pressure_mb", 100),
                    critical_pressure_mb=data.get("critical_pressure_mb", 50),
                )
            except Exception as e:
                logger.warning(f"Failed to load cleanup config: {e}")

    def _save_config(self) -> None:
        """Save cleanup configuration."""
        config_file = self.storage_path / "cleanup_config.json"
        try:
            atomic_write_json(config_file, {
                    "low_pressure_mb": self.thresholds.low_pressure_mb,
                    "medium_pressure_mb": self.thresholds.medium_pressure_mb,
                    "high_pressure_mb": self.thresholds.high_pressure_mb,
                    "critical_pressure_mb": self.thresholds.critical_pressure_mb,
                })
        except Exception as e:
            logger.warning(f"Failed to save cleanup config: {e}")

    def register_cleanup_handler(self, handler: Callable, priority: CleanupPriority = CleanupPriority.MEDIUM) -> None:
        if len(self._cleanup_handlers) >= self.MAX_HANDLERS:
            return
        self._cleanup_handlers.append((priority, handler))
        self._cleanup_handlers.sort(key=lambda x: x[0].value, reverse=True)

    def register_collection(self, name: str, items, window=None) -> None:
        """Register a collection for memory management (compatibility with ShiftingMemoryManager)."""
        pass

    def register_pressure_callback(self, callback: Callable) -> None:
        if len(self._pressure_callbacks) >= self.MAX_PRESSURE_CALLBACKS:
            return
        self._pressure_callbacks.append(callback)

    def get_memory_pressure(self) -> tuple[MemoryPressureLevel, int]:
        """Get current memory pressure level and free memory in MB."""
        try:
            if self._psutil:
                mem = self._psutil.virtual_memory()
                free_mb = mem.available // (1024 * 1024)
            else:
                meminfo = Path("/proc/meminfo")
                if meminfo.exists():
                    with open(meminfo, encoding='utf-8') as f:
                        for line in f:
                            if line.startswith("MemAvailable:"):
                                free_kb = int(line.split()[1])
                                free_mb = free_kb // 1024
                                break
                        else:
                            free_mb = 500
                else:
                    free_mb = 500

            if free_mb < self.thresholds.critical_pressure_mb:
                level = MemoryPressureLevel.CRITICAL
            elif free_mb < self.thresholds.high_pressure_mb:
                level = MemoryPressureLevel.HIGH
            elif free_mb < self.thresholds.medium_pressure_mb:
                level = MemoryPressureLevel.MEDIUM
            elif free_mb < self.thresholds.low_pressure_mb:
                level = MemoryPressureLevel.LOW
            else:
                level = MemoryPressureLevel.NONE

            return level, free_mb
        except Exception as e:
            logger.debug(f"Failed to get memory pressure: {e}")
            return MemoryPressureLevel.NONE, 500

    async def start(self) -> None:
        """Start the memory manager."""
        if self._running:
            return

        self._running = True
        self._cleanup_task = create_tracked_task(self._cleanup_loop(), name="cleanup_loop")
        self._pressure_task = create_tracked_task(self._pressure_monitor_loop(), name="pressure_monitor_loop")
        logger.info("Memory manager started")

    async def stop(self) -> None:
        """Stop the memory manager."""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        if self._pressure_task:
            self._pressure_task.cancel()
            try:
                await self._pressure_task
            except asyncio.CancelledError:
                pass
        logger.info("Memory manager stopped")

    async def _cleanup_loop(self) -> None:
        """Periodic cleanup loop."""
        while self._running:
            try:
                await asyncio.sleep(self.DEFAULT_CLEANUP_INTERVAL)
                await self.run_cleanup(reason="scheduled")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")

    async def _pressure_monitor_loop(self) -> None:
        """Monitor memory pressure and trigger cleanup if needed."""
        while self._running:
            try:
                await asyncio.sleep(self.PRESSURE_CHECK_INTERVAL)

                level, free_mb = self.get_memory_pressure()

                if level != self._last_pressure:
                    logger.info(f"Memory pressure changed: {self._last_pressure.value} -> {level.value} (free: {free_mb}MB)")

                    for callback in self._pressure_callbacks:
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback(level, free_mb)
                            else:
                                callback(level, free_mb)
                        except Exception as e:
                            logger.error(f"Pressure callback error: {e}")

                    self._last_pressure = level

                if level in [MemoryPressureLevel.HIGH, MemoryPressureLevel.CRITICAL]:
                    await self.run_cleanup(
                        reason=f"pressure_{level.value}",
                        priority=CleanupPriority.HIGH if level == MemoryPressureLevel.HIGH else CleanupPriority.CRITICAL,
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Pressure monitor error: {e}")

    async def run_cleanup(
        self,
        reason: str = "manual",
        priority: CleanupPriority = CleanupPriority.MEDIUM,
    ) -> dict:
        """Run cleanup across all registered handlers."""
        start_time = time.time()
        total_removed = 0
        total_bytes = 0
        cleaned_count = 0

        logger.info(f"Running cleanup (reason={reason}, priority={priority.name})")

        for handler_priority, handler in self._cleanup_handlers:
            if handler_priority.value >= priority.value:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        result = await handler()
                    else:
                        result = handler()

                    if result:
                        total_removed += result.get("items_removed", 0)
                        total_bytes += result.get("bytes_freed", 0)
                        if result.get("cleaned"):
                            cleaned_count += 1
                except Exception as e:
                    logger.error(f"Cleanup handler error: {e}")

        if priority.value >= CleanupPriority.HIGH.value:
            collected = gc.collect()
            logger.debug(f"Garbage collected {collected} objects")

        duration = time.time() - start_time

        with self._lock:
            self._stats.items_removed += total_removed
            self._stats.bytes_freed += total_bytes
            self._stats.collections_cleaned += cleaned_count
            self._stats.last_cleanup = datetime.now()
            self._stats.cleanup_duration = duration

        logger.info(
            f"Cleanup complete: removed {total_removed} items, "
            f"freed {total_bytes / 1024 / 1024:.1f}MB in {duration:.2f}s"
        )

        return {
            "reason": reason,
            "items_removed": total_removed,
            "bytes_freed": total_bytes,
            "collections_cleaned": cleaned_count,
            "duration": duration,
        }

    def get_stats(self) -> dict:
        """Get cleanup statistics."""
        with self._lock:
            return {
                "total_items_removed": self._stats.items_removed,
                "total_bytes_freed": self._stats.bytes_freed,
                "total_collections_cleaned": self._stats.collections_cleaned,
                "last_cleanup": self._stats.last_cleanup.isoformat() if self._stats.last_cleanup else None,
                "last_cleanup_duration": self._stats.cleanup_duration,
                "current_pressure": self._last_pressure.value,
                "registered_handlers": len(self._cleanup_handlers),
            }

    async def cleanup_cycle(self) -> dict:
        """Run a cleanup cycle (compatibility with memory_manager)."""
        return await self.run_cleanup(reason="cycle")


def create_conversation_cleanup_handler(memory) -> Callable:
    """Create cleanup handler for conversation memory."""
    def cleanup_conversation() -> dict:
        initial_len = len(memory.history)
        if hasattr(memory, '_trim_if_needed'):
            memory._trim_if_needed()
        elif hasattr(memory, '_trim_history'):
            memory._trim_history()
        memory_pruned = initial_len - len(memory.history)

        initial_important = len(memory.important_messages)
        memory.important_messages = {
            k: v for k, v in memory.important_messages.items()
            if k < len(memory.history)
        }
        important_pruned = initial_important - len(memory.important_messages)

        return {
            "items_removed": memory_pruned + important_pruned,
            "bytes_freed": 0,
            "cleaned": memory_pruned > 0 or important_pruned > 0,
        }
    return cleanup_conversation


def create_cache_cleanup_handler(cache) -> Callable:
    """Create cleanup handler for tool cache."""
    async def cleanup_cache() -> dict:
        initial_size = len(cache._cache)

        async with cache._lock:
            current_time = time.time()
            expired = [
                k for k, (v, t) in cache._cache.items()
                if current_time - t > cache.ttl_seconds
            ]
            for k in expired:
                del cache._cache[k]

        pruned = initial_size - len(cache._cache)

        estimated_bytes = pruned * 1024

        return {
            "items_removed": pruned,
            "bytes_freed": estimated_bytes,
            "cleaned": pruned > 0,
        }
    return cleanup_cache


def create_file_cleanup_handler(directory: Path, max_age_hours: int = 24, pattern: str = "*") -> Callable:
    """Create cleanup handler for temporary files."""
    def cleanup_files() -> dict:
        if not directory.exists():
            return {"items_removed": 0, "bytes_freed": 0, "cleaned": False}

        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        removed = 0
        bytes_freed = 0

        for file_path in directory.glob(pattern):
            if file_path.is_file():
                try:
                    mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if mtime < cutoff:
                        size = file_path.stat().st_size
                        file_path.unlink()
                        removed += 1
                        bytes_freed += size
                except Exception as e:
                    logger.debug(f"Failed to clean {file_path}: {e}")

        return {
            "items_removed": removed,
            "bytes_freed": bytes_freed,
            "cleaned": removed > 0,
        }
    return cleanup_files


def create_log_cleanup_handler(log_dir: Path, max_files: int = 10, max_age_days: int = 7) -> Callable:
    """Create cleanup handler for log files."""
    def cleanup_logs() -> dict:
        if not log_dir.exists():
            return {"items_removed": 0, "bytes_freed": 0, "cleaned": False}

        removed = 0
        bytes_freed = 0

        log_files = sorted(
            log_dir.glob("*.log*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        cutoff = datetime.now() - timedelta(days=max_age_days)

        for i, log_file in enumerate(log_files):
            try:
                mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                if i >= max_files or mtime < cutoff:
                    size = log_file.stat().st_size
                    log_file.unlink()
                    removed += 1
                    bytes_freed += size
            except Exception as e:
                logger.debug(f"Failed to clean log {log_file}: {e}")

        return {
            "items_removed": removed,
            "bytes_freed": bytes_freed,
            "cleaned": removed > 0,
        }
    return cleanup_logs


_memory_manager: MemoryManager | None = None
_memory_manager_lock = threading.Lock()


def get_memory_manager() -> MemoryManager:
    """Get global memory manager instance (thread-safe)."""
    global _memory_manager
    if _memory_manager is None:
        with _memory_manager_lock:
            if _memory_manager is None:
                _memory_manager = MemoryManager()
    return _memory_manager
