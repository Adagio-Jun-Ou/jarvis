"""Memory management system for Jarvis.

Provides centralized memory tracking, limits enforcement, and cleanup
mechanisms optimized for memory-constrained with AMD Ryzen AI CPU/GPU.
"""
import asyncio
import gc
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from core.helpers import create_tracked_task
from typing import Any, Callable
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class MemoryPressureLevel(Enum):
    """Memory pressure levels."""
    NORMAL = "normal"
    WARNING = "warning"      # 70% of limit
    HIGH = "high"           # 85% of limit
    CRITICAL = "critical"   # 95% of limit


@dataclass
class MemoryConfig:
    """Memory configuration for 32GB system."""
    total_memory_gb: float = 32.0
    max_system_memory_percent: float = 0.8      # 80% of total - generous for 32GB
    max_dataframe_cache_mb: float = 2000.0     # 2GB for dataframes
    max_embedding_cache_entries: int = 50000    # Rich embedding cache
    max_topic_entries: int = 5000              # More topic tracking
    max_context_entries: int = 2000             # Larger context window
    max_session_tools_tracked: int = 500        # More tool history
    max_sql_connections: int = 20              # More DB connections
    max_llm_connections: int = 200              # More LLM connections
    connection_idle_timeout_seconds: float = 600.0  # 10 min idle
    cache_cleanup_interval_seconds: float = 120.0   # 2 min cleanup
    pressure_check_interval_seconds: float = 60.0   # 1 min pressure check


@dataclass
class ComponentMemoryStats:
    """Memory stats for a component."""
    name: str
    size_bytes: int = 0
    entry_count: int = 0
    max_entries: int = 0
    max_size_bytes: int = 0
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    evictions: int = 0


class BoundedCache:
    """Thread-safe bounded cache with TTL and memory limit."""

    def __init__(
        self,
        name: str,
        max_entries: int = 1000,
        max_size_bytes: int | None = None,
        ttl_seconds: float = 300.0,
        on_evict: Callable[[str, Any], None] | None = None,
    ):
        self.name = name
        self.max_entries = max_entries
        self.max_size_bytes = max_size_bytes
        self.ttl_seconds = ttl_seconds
        self.on_evict = on_evict

        self._cache: OrderedDict[str, tuple[Any, float, int]] = OrderedDict()
        self._lock = threading.RLock()
        self._current_size = 0
        self._stats = ComponentMemoryStats(
            name=name,
            max_entries=max_entries,
            max_size_bytes=max_size_bytes or 0,
        )

    def get(self, key: str) -> Any | None:
        """Get item from cache, returns None if expired or missing."""
        with self._lock:
            if key not in self._cache:
                return None

            value, timestamp, size = self._cache[key]

            if time.time() - timestamp > self.ttl_seconds:
                self._evict(key, value, size)
                return None

            self._cache.move_to_end(key)
            self._stats.last_accessed = time.time()
            return value

    def set(self, key: str, value: Any, size_bytes: int = 0) -> bool:
        """Set item in cache, returns False if rejected."""
        with self._lock:
            if key in self._cache:
                _, _, old_size = self._cache[key]
                self._current_size -= old_size
                del self._cache[key]

            if self.max_size_bytes and size_bytes > self.max_size_bytes:
                logger.warning(f"Item too large for cache {self.name}: {size_bytes} > {self.max_size_bytes}")
                return False

            while self._should_evict(size_bytes):
                if not self._evict_oldest():
                    break

            self._cache[key] = (value, time.time(), size_bytes)
            self._current_size += size_bytes

            self._stats.entry_count = len(self._cache)
            self._stats.size_bytes = self._current_size

            return True

    def _should_evict(self, incoming_size: int) -> bool:
        if len(self._cache) >= self.max_entries:
            return True
        if self.max_size_bytes and self._current_size + incoming_size > self.max_size_bytes:
            return True
        return False

    def _evict_oldest(self) -> bool:
        if not self._cache:
            return False

        key = next(iter(self._cache))
        value, _, size = self._cache[key]
        self._evict(key, value, size)
        return True

    def _evict(self, key: str, value: Any, size: int) -> None:
        if self.on_evict:
            try:
                self.on_evict(key, value)
            except Exception as e:
                logger.warning(f"Eviction callback error: {e}")

        if key in self._cache:
            del self._cache[key]
        self._current_size -= size
        self._stats.evictions += 1

    def clear(self) -> int:
        """Clear all entries, return count cleared."""
        with self._lock:
            count = len(self._cache)
            if self.on_evict:
                for key, (value, _, _) in list(self._cache.items()):
                    try:
                        self.on_evict(key, value)
                    except Exception as e:
                        logger.debug(f"Cache eviction callback failed: {e}")

            self._cache.clear()
            self._current_size = 0
            self._stats.entry_count = 0
            self._stats.size_bytes = 0
            return count

    def cleanup_expired(self) -> int:
        """Remove expired entries, return count removed."""
        with self._lock:
            expired = []
            now = time.time()

            for key, (_, timestamp, _) in self._cache.items():
                if now - timestamp > self.ttl_seconds:
                    expired.append(key)

            for key in expired:
                value, _, size = self._cache[key]
                self._evict(key, value, size)

            return len(expired)

    def get_stats(self) -> ComponentMemoryStats:
        with self._lock:
            return ComponentMemoryStats(
                name=self._stats.name,
                size_bytes=self._stats.size_bytes,
                entry_count=self._stats.entry_count,
                max_entries=self._stats.max_entries,
                max_size_bytes=self._stats.max_size_bytes,
                created_at=self._stats.created_at,
                last_accessed=self._stats.last_accessed,
                evictions=self._stats.evictions,
            )


class ConnectionPool:
    """Bounded connection pool with idle timeout."""

    def __init__(
        self,
        name: str,
        max_connections: int = 10,
        idle_timeout_seconds: float = 300.0,
        on_close: Callable[[Any], None] | None = None,
    ):
        self.name = name
        self.max_connections = max_connections
        self.idle_timeout_seconds = idle_timeout_seconds
        self.on_close = on_close

        self._connections: dict[str, tuple[Any, float]] = {}
        self._lock = threading.RLock()

    def get(self, key: str) -> Any | None:
        """Get connection if exists and not expired."""
        with self._lock:
            if key not in self._connections:
                return None

            conn, last_used = self._connections[key]

            if time.time() - last_used > self.idle_timeout_seconds:
                self._close_connection(key, conn)
                return None

            self._connections[key] = (conn, time.time())
            return conn

    def set(self, key: str, connection: Any) -> bool:
        """Store connection, evict oldest if at capacity."""
        with self._lock:
            if key in self._connections:
                old_conn, _ = self._connections[key]
                self._close_connection(key, old_conn)

            while len(self._connections) >= self.max_connections:
                oldest_key = min(
                    self._connections.keys(),
                    key=lambda k: self._connections[k][1]
                )
                old_conn, _ = self._connections[oldest_key]
                self._close_connection(oldest_key, old_conn)

            self._connections[key] = (connection, time.time())
            return True

    def remove(self, key: str) -> bool:
        """Remove and close connection."""
        with self._lock:
            if key not in self._connections:
                return False

            conn, _ = self._connections[key]
            self._close_connection(key, conn)
            return True

    def _close_connection(self, key: str, conn: Any) -> None:
        if self.on_close:
            try:
                self.on_close(conn)
            except Exception as e:
                logger.warning(f"Connection close error: {e}")

        if key in self._connections:
            del self._connections[key]

    def cleanup_idle(self) -> int:
        """Close idle connections, return count closed."""
        with self._lock:
            now = time.time()
            idle_keys = [
                k for k, (_, last_used) in self._connections.items()
                if now - last_used > self.idle_timeout_seconds
            ]

            for key in idle_keys:
                conn, _ = self._connections[key]
                self._close_connection(key, conn)

            return len(idle_keys)

    def clear(self) -> int:
        """Close all connections."""
        with self._lock:
            count = len(self._connections)
            for key, (conn, _) in list(self._connections.items()):
                self._close_connection(key, conn)
            return count

    def count(self) -> int:
        with self._lock:
            return len(self._connections)


class MemoryManager:
    """Central memory management for Jarvis.

    Tracks memory across components, enforces limits, and provides
    cleanup mechanisms for memory-constrained.
    """

    def __init__(self, config: MemoryConfig | None = None):
        self.config = config or MemoryConfig()

        self._caches: dict[str, BoundedCache] = {}
        self._pools: dict[str, ConnectionPool] = {}
        self._trackers: dict[str, list] = {}

        self._pressure_callbacks: dict[MemoryPressureLevel, list[Callable]] = {
            level: [] for level in MemoryPressureLevel
        }

        self._cleanup_task: asyncio.Task | None = None
        self._running = False

        self._lock = threading.RLock()

        self._last_pressure = MemoryPressureLevel.NORMAL
        self._component_stats: dict[str, ComponentMemoryStats] = {}

    def register_cache(self, cache: BoundedCache) -> None:
        """Register a cache for tracking."""
        with self._lock:
            self._caches[cache.name] = cache
            self._component_stats[cache.name] = cache.get_stats()

    def create_cache(
        self,
        name: str,
        max_entries: int = 1000,
        max_size_bytes: int | None = None,
        ttl_seconds: float = 300.0,
        on_evict: Callable | None = None,
    ) -> BoundedCache:
        """Create and register a bounded cache."""
        cache = BoundedCache(
            name=name,
            max_entries=max_entries,
            max_size_bytes=max_size_bytes,
            ttl_seconds=ttl_seconds,
            on_evict=on_evict,
        )
        self.register_cache(cache)
        return cache

    def register_pool(self, pool: ConnectionPool) -> None:
        """Register a connection pool for tracking."""
        with self._lock:
            self._pools[pool.name] = pool

    def create_pool(
        self,
        name: str,
        max_connections: int = 10,
        idle_timeout_seconds: float = 300.0,
        on_close: Callable | None = None,
    ) -> ConnectionPool:
        """Create and register a connection pool."""
        pool = ConnectionPool(
            name=name,
            max_connections=max_connections,
            idle_timeout_seconds=idle_timeout_seconds,
            on_close=on_close,
        )
        self.register_pool(pool)
        return pool

    def register_tracker(self, name: str, max_entries: int) -> None:
        """Register a bounded list tracker."""
        with self._lock:
            self._trackers[name] = []

    def track_append(self, name: str, item: Any, max_entries: int | None = None) -> None:
        """Append to tracked list with automatic trimming."""
        with self._lock:
            if name not in self._trackers:
                max_e = max_entries or self.config.max_session_tools_tracked
                self._trackers[name] = []

            max_e = max_entries or self.config.max_session_tools_tracked
            tracker = self._trackers[name]
            tracker.append(item)

            while len(tracker) > max_e:
                tracker.pop(0)

    def get_tracker(self, name: str) -> list:
        """Get tracked list copy."""
        with self._lock:
            return self._trackers.get(name, []).copy()

    def clear_tracker(self, name: str) -> int:
        """Clear tracker, return count cleared."""
        with self._lock:
            if name in self._trackers:
                count = len(self._trackers[name])
                self._trackers[name].clear()
                return count
            return 0

    def on_memory_pressure(self, level: MemoryPressureLevel, callback: Callable) -> None:
        """Register callback for memory pressure events."""
        with self._lock:
            if len(self._pressure_callbacks[level]) < 10:
                self._pressure_callbacks[level].append(callback)

    def get_memory_usage(self) -> dict:
        """Get current memory usage stats."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()

            return {
                "rss_mb": memory_info.rss / (1024 * 1024),
                "vms_mb": memory_info.vms / (1024 * 1024),
                "percent": process.memory_percent(),
            }
        except ImportError:
            return {"rss_mb": 0, "vms_mb": 0, "percent": 0}

    def check_pressure(self) -> MemoryPressureLevel:
        """Check current memory pressure level."""
        usage = self.get_memory_usage()
        used_gb = usage["rss_mb"] / 1024
        max_gb = self.config.total_memory_gb * self.config.max_system_memory_percent
        percent = used_gb / max_gb if max_gb > 0 else 0

        if percent >= 0.95:
            level = MemoryPressureLevel.CRITICAL
        elif percent >= 0.85:
            level = MemoryPressureLevel.HIGH
        elif percent >= 0.70:
            level = MemoryPressureLevel.WARNING
        else:
            level = MemoryPressureLevel.NORMAL

        if level != self._last_pressure:
            self._last_pressure = level
            self._notify_pressure(level)

        return level

    def _notify_pressure(self, level: MemoryPressureLevel) -> None:
        """Notify callbacks of pressure change."""
        with self._lock:
            callbacks = list(self._pressure_callbacks.get(level, []))
        for callback in callbacks:
            try:
                callback()
            except Exception as e:
                logger.warning(f"Pressure callback error: {e}")

    async def cleanup_cycle(self) -> dict:
        """Run a cleanup cycle, return stats."""
        stats = {
            "caches_cleaned": 0,
            "connections_closed": 0,
            "trackers_trimmed": 0,
        }

        with self._lock:
            caches = list(self._caches.values())
            pools = list(self._pools.values())

        for cache in caches:
            stats["caches_cleaned"] += cache.cleanup_expired()

        for pool in pools:
            stats["connections_closed"] += pool.cleanup_idle()

        gc.collect()

        return stats

    def force_cleanup(self, level: str = "medium") -> dict:
        """Force cleanup at specified level."""
        stats = {
            "caches_cleared": 0,
            "connections_closed": 0,
            "gc_collected": 0,
        }

        with self._lock:
            caches = list(self._caches.values())
            pools = list(self._pools.values())
            tracker_names = list(self._trackers.keys())

        if level == "light":
            for cache in caches:
                stats["caches_cleared"] += cache.cleanup_expired()

        elif level == "medium":
            for cache in caches:
                stats["caches_cleared"] += cache.cleanup_expired()

            for pool in pools:
                stats["connections_closed"] += pool.cleanup_idle()

        elif level == "heavy":
            for cache in caches:
                stats["caches_cleared"] += cache.clear()

            for pool in pools:
                stats["connections_closed"] += pool.clear()

            for name in tracker_names:
                stats["trackers_trimmed"] = self.clear_tracker(name)

        stats["gc_collected"] = gc.collect()

        return stats

    async def start(self) -> None:
        """Start background cleanup task."""
        if self._running:
            return

        self._running = True
        self._cleanup_task = create_tracked_task(self._cleanup_loop(), name="memory_cleanup_loop")

    async def stop(self) -> None:
        """Stop background cleanup and clear all."""
        self._running = False

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        self.force_cleanup("heavy")

    async def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while self._running:
            try:
                await asyncio.sleep(self.config.cache_cleanup_interval_seconds)
                await self.cleanup_cycle()
                self.check_pressure()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Cleanup error: {e}")

    def get_stats(self) -> dict:
        """Get comprehensive memory stats."""
        return {
            "config": {
                "total_memory_gb": self.config.total_memory_gb,
                "max_system_percent": self.config.max_system_memory_percent,
            },
            "usage": self.get_memory_usage(),
            "pressure": self._last_pressure.value,
            "caches": {
                name: {
                    "entries": cache.get_stats().entry_count,
                    "size_bytes": cache.get_stats().size_bytes,
                    "evictions": cache.get_stats().evictions,
                }
                for name, cache in self._caches.items()
            },
            "pools": {
                name: pool.count()
                for name, pool in self._pools.items()
            },
            "trackers": {
                name: len(items)
                for name, items in self._trackers.items()
            },
        }


_manager: MemoryManager | None = None
_manager_lock = threading.Lock()


def get_memory_limits_manager() -> MemoryManager:
    """Get global memory limits manager instance (thread-safe).
    
    Note: This is the memory LIMITS/TRACKING manager from memory_manager.py.
    For memory CLEANUP/ARCHIVING, use get_memory_manager() from memory_cleanup.py.
    """
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = MemoryManager()
    return _manager


def get_memory_manager() -> MemoryManager:
    """Deprecated: Use get_memory_limits_manager() for clarity.
    
    This alias exists for backward compatibility but will be removed.
    Prefer get_memory_limits_manager() to distinguish from
    get_memory_manager() in memory_cleanup.py.
    """
    return get_memory_limits_manager()


def create_memory_manager(config: MemoryConfig | None = None) -> MemoryManager:
    """Create and set global memory manager."""
    global _manager
    _manager = MemoryManager(config)
    return _manager


@contextmanager
def memory_tracked(component: str):
    """Context manager for tracking component memory usage."""
    manager = get_memory_limits_manager()
    start = time.time()
    try:
        yield manager
    finally:
        duration = time.time() - start
        logger.debug(f"{component} completed in {duration:.3f}s")
