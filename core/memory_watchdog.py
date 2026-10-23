"""Memory watchdog for aggressive memory management.

Monitors memory usage and triggers cleanup when thresholds are exceeded.
Provides proactive memory management to prevent OOM situations.
"""
import asyncio
import gc
import logging
import threading
import time
import weakref
from dataclasses import dataclass
from enum import Enum
from core.helpers import create_tracked_task
from typing import Any, Callable

logger = logging.getLogger(__name__)


class MemoryState(Enum):
    """Memory state levels."""
    NORMAL = "normal"
    WARNING = "warning"      # 70% of limit
    HIGH = "high"            # 80% of limit  
    CRITICAL = "critical"    # 90% of limit


@dataclass
class MemoryThresholds:
    """Memory thresholds for different actions."""
    warning_percent: float = 0.70
    high_percent: float = 0.80
    critical_percent: float = 0.90
    max_rss_gb: float = 16.0
    

@dataclass
class CleanupStats:
    """Statistics from a cleanup operation."""
    caches_cleared: int = 0
    gc_collected: int = 0
    callbacks_called: int = 0
    rss_before_mb: float = 0.0
    rss_after_mb: float = 0.0
    freed_mb: float = 0.0


class MemoryWatchdog:
    """Watchdog for monitoring and managing memory.
    
    Features:
    - Periodic memory monitoring
    - Automatic cleanup on pressure
    - Callback registration for pressure events
    - Integration with BoundedCache and ConnectionPool
    - Global singleton for system-wide management
    """
    
    _instance = None
    _lock = threading.Lock()
    MAX_CALLBACKS = 30
    
    def __init__(
        self,
        thresholds: MemoryThresholds | None = None,
        check_interval: float = 30.0,
        auto_cleanup: bool = True,
    ):
        self.thresholds = thresholds or MemoryThresholds()
        self.check_interval = check_interval
        self.auto_cleanup = auto_cleanup
        
        self._state = MemoryState.NORMAL
        self._running = False
        self._task: asyncio.Task | None = None
        self._last_check: float = 0.0
        self._last_cleanup: float = 0.0
        
        self._cleanup_callbacks: list[Callable[[], None]] = []
        self._pressure_callbacks: dict[MemoryState, list[Callable[[], None]]] = {
            MemoryState.WARNING: [],
            MemoryState.HIGH: [],
            MemoryState.CRITICAL: [],
        }
        
        self._cleanup_targets: weakref.WeakSet[Any] = weakref.WeakSet()
        
        self._stats = {
            "checks": 0,
            "cleanups": 0,
            "state_changes": 0,
            "total_freed_mb": 0.0,
        }
        self._lock = threading.RLock()
    
    @classmethod
    def get_instance(cls) -> "MemoryWatchdog":
        """Get or create the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def register_cleanup_target(self, target: Any) -> None:
        """Register an object with a clear_cache() or cleanup() method."""
        self._cleanup_targets.add(target)

    def unregister_cleanup_target(self, target: Any) -> bool:
        """Unregister a cleanup target."""
        if target in self._cleanup_targets:
            self._cleanup_targets.discard(target)
            return True
        return False

    def register_callback(self, callback: Callable[[], None], state: MemoryState | None = None) -> None:
        """Register a callback for cleanup or specific pressure state."""
        with self._lock:
            if state is None:
                if len(self._cleanup_callbacks) >= self.MAX_CALLBACKS:
                    return
                self._cleanup_callbacks.append(callback)
            else:
                self._pressure_callbacks[state].append(callback)

    def unregister_callback(self, callback: Callable[[], None], state: MemoryState | None = None) -> bool:
        """Unregister a callback."""
        with self._lock:
            if state is None:
                if callback in self._cleanup_callbacks:
                    self._cleanup_callbacks.remove(callback)
                    return True
            else:
                if state in self._pressure_callbacks and callback in self._pressure_callbacks[state]:
                    self._pressure_callbacks[state].remove(callback)
                    return True
        return False

    def unregister_pressure_callback(self, callback: Callable[[], None], state: MemoryState) -> bool:
        """Unregister a pressure callback for a specific state."""
        return self.unregister_callback(callback, state)
    
    def get_memory_usage(self) -> dict:
        """Get current memory usage."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                "rss_mb": memory_info.rss / (1024 * 1024),
                "vms_mb": memory_info.vms / (1024 * 1024),
                "percent": process.memory_percent(),
                "available_mb": psutil.virtual_memory().available / (1024 * 1024),
            }
        except ImportError:
            return {"rss_mb": 0, "vms_mb": 0, "percent": 0, "available_mb": 0}
    
    def check_memory(self) -> MemoryState:
        """Check current memory state."""
        usage = self.get_memory_usage()
        rss_gb = usage["rss_mb"] / 1024
        max_gb = self.thresholds.max_rss_gb
        percent = rss_gb / max_gb if max_gb > 0 else 0
        
        new_state = MemoryState.NORMAL
        if percent >= self.thresholds.critical_percent:
            new_state = MemoryState.CRITICAL
        elif percent >= self.thresholds.high_percent:
            new_state = MemoryState.HIGH
        elif percent >= self.thresholds.warning_percent:
            new_state = MemoryState.WARNING
        
        self._stats["checks"] += 1
        
        if new_state != self._state:
            old_state = self._state
            with self._lock:
                self._state = new_state
                self._stats["state_changes"] += 1
            logger.info(f"Memory state changed: {old_state.value} -> {new_state.value} ({percent:.1%})")
            self._notify_pressure(new_state)
            
            if self.auto_cleanup and new_state.value in ("high", "critical"):
                self.force_cleanup(new_state)
        
        self._last_check = time.time()
        return new_state
    
    def _notify_pressure(self, state: MemoryState) -> None:
        """Notify callbacks of pressure state."""
        with self._lock:
            callbacks = list(self._pressure_callbacks.get(state, []))
        for callback in callbacks:
            try:
                callback()
            except Exception as e:
                logger.warning(f"Pressure callback error: {e}")
    
    def force_cleanup(self, state: MemoryState | None = None) -> CleanupStats:
        """Force a cleanup operation."""
        stats = CleanupStats()
        stats.rss_before_mb = self.get_memory_usage()["rss_mb"]
        
        with self._lock:
            targets = list(self._cleanup_targets)
            cleanup_cbs = list(self._cleanup_callbacks)
        
        for target in targets:
            try:
                if hasattr(target, 'clear_cache'):
                    target.clear_cache()
                    stats.caches_cleared += 1
                elif hasattr(target, 'cleanup'):
                    target.cleanup()
                    stats.caches_cleared += 1
                elif hasattr(target, 'cleanup_expired'):
                    stats.caches_cleared += target.cleanup_expired()
            except Exception as e:
                logger.debug(f"Cleanup target error: {e}")
        
        for callback in cleanup_cbs:
            try:
                callback()
                stats.callbacks_called += 1
            except Exception as e:
                logger.debug(f"Cleanup callback error: {e}")
        
        if state in (MemoryState.HIGH, MemoryState.CRITICAL):
            stats.gc_collected = gc.collect()
        else:
            stats.gc_collected = gc.collect(0)
        
        stats.rss_after_mb = self.get_memory_usage()["rss_mb"]
        stats.freed_mb = max(0, stats.rss_before_mb - stats.rss_after_mb)
        
        self._stats["cleanups"] += 1
        self._stats["total_freed_mb"] += stats.freed_mb
        self._last_cleanup = time.time()
        
        level = state.value if state else "normal"
        logger.info(f"Memory cleanup ({level}): cleared {stats.caches_cleared} caches, "
                   f"gc collected {stats.gc_collected}, freed {stats.freed_mb:.1f}MB")
        
        return stats
    
    async def start(self) -> None:
        """Start the watchdog background task."""
        if self._running:
            return
        
        self._running = True
        self._task = create_tracked_task(self._monitor_loop(), name="memory_watchdog_loop")
        logger.info(f"Memory watchdog started (check interval: {self.check_interval}s)")
    
    async def stop(self) -> None:
        """Stop the watchdog."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Memory watchdog stopped")
    
    async def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            try:
                await asyncio.sleep(self.check_interval)
                self.check_memory()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Memory watchdog error: {e}")
    
    def get_stats(self) -> dict:
        """Get watchdog statistics."""
        with self._lock:
            stats_copy = dict(self._stats)
            state_value = self._state.value
        return {
            **stats_copy,
            "state": state_value,
            "last_check": self._last_check,
            "last_cleanup": self._last_cleanup,
            "current_usage": self.get_memory_usage(),
            "thresholds": {
                "warning": f"{self.thresholds.warning_percent:.0%}",
                "high": f"{self.thresholds.high_percent:.0%}",
                "critical": f"{self.thresholds.critical_percent:.0%}",
                "max_rss_gb": self.thresholds.max_rss_gb,
            },
        }
    
    def set_max_memory(self, max_gb: float) -> None:
        """Update max RSS memory limit."""
        self.thresholds.max_rss_gb = max_gb


def get_memory_watchdog() -> MemoryWatchdog:
    """Get the global memory watchdog instance."""
    return MemoryWatchdog.get_instance()


def register_for_cleanup(target: Any) -> None:
    """Register an object for automatic cleanup."""
    get_memory_watchdog().register_cleanup_target(target)


async def start_memory_watchdog(
    max_memory_gb: float = 16.0,
    check_interval: float = 30.0,
) -> MemoryWatchdog:
    """Start the global memory watchdog."""
    watchdog = get_memory_watchdog()
    watchdog.set_max_memory(max_memory_gb)
    watchdog.check_interval = check_interval
    await watchdog.start()
    return watchdog