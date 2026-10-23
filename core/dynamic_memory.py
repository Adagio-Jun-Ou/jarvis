"""Dynamic Memory Manager for Jarvis.

System-aware memory management:
- Monitors total system memory (RAM + swap)
- Sets soft limits based on 24GB threshold for 32GB system
- Aggressive cleanup when approaching limit
- Coordinates with all subsystems for memory release
- Dynamic cache size adjustment based on pressure
"""
import asyncio
import gc
import logging
import threading
import time
import weakref
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from core.helpers import create_tracked_task
from typing import Any, Callable

logger = logging.getLogger(__name__)


class MemoryPressure(Enum):
    """System memory pressure levels."""
    NORMAL = "normal"          # < 60% of max
    ELEVATED = "elevated"      # 60-75% of max
    HIGH = "high"              # 75-85% of max
    CRITICAL = "critical"      # 85-95% of max
    EMERGENCY = "emergency"    # > 95% of max


@dataclass
class SystemMemoryInfo:
    """Current system memory state."""
    total_ram_mb: float = 0.0
    available_ram_mb: float = 0.0
    used_ram_mb: float = 0.0
    ram_percent: float = 0.0
    
    total_swap_mb: float = 0.0
    used_swap_mb: float = 0.0
    swap_percent: float = 0.0
    
    process_rss_mb: float = 0.0
    process_percent: float = 0.0
    
    total_available_mb: float = 0.0  # RAM + Swap available
    pressure: MemoryPressure = MemoryPressure.NORMAL
    
    @property
    def combined_percent(self) -> float:
        """Combined memory usage percentage."""
        if self.total_ram_mb + self.total_swap_mb == 0:
            return 0.0
        total_used = self.used_ram_mb + self.used_swap_mb
        total_mb = self.total_ram_mb + self.total_swap_mb
        return (total_used / total_mb) * 100


@dataclass
class DynamicMemoryConfig:
    """Configuration for dynamic memory management."""
    # Total system memory we can use (24GB for a 32GB system)
    max_memory_mb: float = 24 * 1024.0
    
    # Pressure thresholds (percentage of max_memory_mb)
    elevated_threshold: float = 0.60   # 14.4 GB
    high_threshold: float = 0.75       # 18 GB
    critical_threshold: float = 0.85   # 20.4 GB
    emergency_threshold: float = 0.95  # 22.8 GB
    
    # Cleanup targets at each level
    normal_cache_size: int = 10000
    elevated_cache_size: int = 5000
    high_cache_size: int = 2500
    critical_cache_size: int = 1000
    emergency_cache_size: int = 250
    
    # Cleanup intervals
    normal_check_interval: float = 60.0
    elevated_check_interval: float = 30.0
    high_check_interval: float = 15.0
    critical_check_interval: float = 5.0
    emergency_check_interval: float = 2.0


@dataclass
class MemoryManagerStats:
    """Statistics for memory manager."""
    total_cleanups: int = 0
    emergency_cleanups: int = 0
    memory_freed_mb: float = 0.0
    cache_adjustments: int = 0
    gc_collections: int = 0
    pressure_changes: int = 0
    last_pressure: MemoryPressure = MemoryPressure.NORMAL
    last_cleanup: datetime | None = None
    uptime_seconds: float = 0.0


_PRESSURE_ORDER = {
    MemoryPressure.NORMAL: 0,
    MemoryPressure.ELEVATED: 1,
    MemoryPressure.HIGH: 2,
    MemoryPressure.CRITICAL: 3,
    MemoryPressure.EMERGENCY: 4,
}


class DynamicMemoryManager:
    MAX_CLEANUP_CALLBACKS = 50
    MAX_CACHE_CALLBACKS = 20
    """Dynamic memory manager with system-aware pressure handling.
    
    Coordinates memory cleanup across all Jarvis subsystems based on
    actual system memory (32GB RAM + 8GB swap, with 24GB max for Jarvis).
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __init__(self, config: DynamicMemoryConfig | None = None):
        self.config = config or DynamicMemoryConfig()
        
        self._running = False
        self._task: asyncio.Task | None = None
        self._pressure = MemoryPressure.NORMAL
        self._start_time: float = time.time()
        
        self._stats = MemoryManagerStats()
        self._stats_lock = threading.Lock()
        
        # Cleanup callbacks with priority
        self._cleanup_callbacks: list[tuple[int, Callable, str]] = []
        
        # Cache references for dynamic sizing
        self._caches: weakref.WeakSet[Any] = weakref.WeakSet()
        self._cache_size_callbacks: list[Callable[[int], None]] = []
        self._cache_callbacks_lock = threading.Lock()
        
        # Large objects that can be released
        self._releasable_objects: weakref.WeakSet[Any] = weakref.WeakSet()
        
        # Memory pools
        self._memory_pools: dict[str, list] = {}
        
        # psutil for system memory
        self._psutil = None
        try:
            import psutil
            self._psutil = psutil
        except ImportError:
            logger.warning("psutil not available, memory monitoring limited")
    
    @classmethod
    def get_instance(cls) -> "DynamicMemoryManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def get_system_memory(self) -> SystemMemoryInfo:
        """Get current system memory state."""
        info = SystemMemoryInfo()
        
        if self._psutil:
            try:
                # Virtual memory (RAM)
                vm = self._psutil.virtual_memory()
                info.total_ram_mb = vm.total / (1024 * 1024)
                info.available_ram_mb = vm.available / (1024 * 1024)
                info.used_ram_mb = vm.used / (1024 * 1024)
                info.ram_percent = vm.percent
                
                # Swap memory
                swap = self._psutil.swap_memory()
                info.total_swap_mb = swap.total / (1024 * 1024)
                info.used_swap_mb = swap.used / (1024 * 1024)
                info.swap_percent = swap.percent if swap.total > 0 else 0.0
                
                # Process memory
                process = self._psutil.Process()
                mem_info = process.memory_info()
                info.process_rss_mb = mem_info.rss / (1024 * 1024)
                info.process_percent = process.memory_percent()
                
                # Total available
                info.total_available_mb = info.available_ram_mb + (info.total_swap_mb - info.used_swap_mb)
                
                # Determine pressure based on process memory vs config max
                info.pressure = self._calculate_pressure(info.process_rss_mb)
                
            except Exception as e:
                logger.debug(f"Error getting system memory: {e}")
        else:
            # Fallback: just track process memory
            try:
                import resource
                usage = resource.getrusage(resource.RUSAGE_SELF)
                info.process_rss_mb = usage.ru_maxrss / 1024  # Convert from KB
            except Exception as e:
                logger.debug(f"Resource usage check failed: {e}")
        
        return info
    
    def _calculate_pressure(self, process_mb: float) -> MemoryPressure:
        """Calculate memory pressure level."""
        ratio = process_mb / self.config.max_memory_mb
        
        if ratio >= self.config.emergency_threshold:
            return MemoryPressure.EMERGENCY
        elif ratio >= self.config.critical_threshold:
            return MemoryPressure.CRITICAL
        elif ratio >= self.config.high_threshold:
            return MemoryPressure.HIGH
        elif ratio >= self.config.elevated_threshold:
            return MemoryPressure.ELEVATED
        return MemoryPressure.NORMAL
    
    def get_target_cache_size(self) -> int:
        """Get target cache size based on current pressure."""
        with self._lock:
            pressure = self._pressure
        sizes = {
            MemoryPressure.NORMAL: self.config.normal_cache_size,
            MemoryPressure.ELEVATED: self.config.elevated_cache_size,
            MemoryPressure.HIGH: self.config.high_cache_size,
            MemoryPressure.CRITICAL: self.config.critical_cache_size,
            MemoryPressure.EMERGENCY: self.config.emergency_cache_size,
        }
        return sizes[pressure]
    
    def register_cleanup_callback(self, callback: Callable, name: str, priority: int = 50) -> None:
        with self._lock:
            if len(self._cleanup_callbacks) >= self.MAX_CLEANUP_CALLBACKS:
                return
            self._cleanup_callbacks.append((priority, callback, name))
            self._cleanup_callbacks.sort(key=lambda x: x[0])
        logger.debug(f"Registered cleanup callback: {name} (priority {priority})")
    

    def unregister_cleanup_callback(self, name: str) -> bool:
        """Unregister a cleanup callback by name.
        
        Args:
            name: Unique name of callback to remove
            
        Returns:
            True if callback was found and removed
        """
        with self._lock:
            original_len = len(self._cleanup_callbacks)
            self._cleanup_callbacks = [(p, cb, n) for p, cb, n in self._cleanup_callbacks if n != name]
            return len(self._cleanup_callbacks) < original_len

    def register_cache(self, cache: Any, size_callback: Callable[[int], None] | None = None) -> None:
        with self._cache_callbacks_lock:
            self._caches.add(cache)
            if size_callback:
                if len(self._cache_size_callbacks) < self.MAX_CACHE_CALLBACKS:
                    self._cache_size_callbacks.append(size_callback)
    
    def register_releasable(self, obj: Any) -> None:
        """Register an object that can be released in emergencies."""
        self._releasable_objects.add(obj)
    
    def register_memory_pool(self, name: str, pool: list) -> None:
        """Register a memory pool for tracking."""
        with self._lock:
            self._memory_pools[name] = pool
    
    def force_cleanup(self, level: MemoryPressure | None = None) -> dict:
        """Force immediate cleanup."""
        before = self.get_system_memory()
        
        results = {
            "before_mb": before.process_rss_mb,
            "callbacks_called": 0,
            "caches_cleared": 0,
            "pools_cleared": 0,
            "objects_released": 0,
            "gc_collected": 0,
        }
        
        # Call cleanup callbacks
        with self._lock:
            sync_callbacks = list(self._cleanup_callbacks)
        for priority, callback, name in sync_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    # For async callbacks, we skip in sync context
                    pass
                else:
                    callback()
                    results["callbacks_called"] += 1
            except Exception as e:
                logger.debug(f"Cleanup callback {name} error: {e}")
        
        # Clear caches
        target_size = self.get_target_cache_size() if level else 0
        for cache in list(self._caches):
            try:
                if hasattr(cache, 'clear'):
                    cache.clear()
                    results["caches_cleared"] += 1
                elif hasattr(cache, 'clear_cache'):
                    cache.clear_cache()
                    results["caches_cleared"] += 1
                elif hasattr(cache, 'resize') and level in (MemoryPressure.HIGH, MemoryPressure.CRITICAL, MemoryPressure.EMERGENCY):
                    cache.resize(target_size)
                    results["caches_cleared"] += 1
            except Exception as e:
                logger.debug(f"Cache clear error: {e}")
        
        # Notify cache size callbacks
        with self._cache_callbacks_lock:
            callbacks_snapshot = list(self._cache_size_callbacks)
        for callback in callbacks_snapshot:
            try:
                callback(target_size)
                results["cache_adjustments"] = results.get("cache_adjustments", 0) + 1
            except Exception as e:
                logger.debug(f"Cache size callback error: {e}")
        
        # Clear memory pools
        with self._lock:
            pools_snapshot = dict(self._memory_pools)
        for name, pool in pools_snapshot.items():
            try:
                pool.clear()
                results["pools_cleared"] += 1
            except Exception as e:
                logger.debug(f"Pool {name} clear error: {e}")
        
        # Release large objects in emergency
        if level in (MemoryPressure.CRITICAL, MemoryPressure.EMERGENCY):
            for obj in list(self._releasable_objects):
                try:
                    if hasattr(obj, 'release'):
                        obj.release()
                        results["objects_released"] += 1
                    elif hasattr(obj, 'clear'):
                        obj.clear()
                        results["objects_released"] += 1
                except Exception as e:
                    logger.debug(f"Object release error: {e}")
        
        # Garbage collection
        gc_level = 2 if level in (MemoryPressure.CRITICAL, MemoryPressure.EMERGENCY) else 0
        results["gc_collected"] = gc.collect(gc_level)
        
        # Update stats
        after = self.get_system_memory()
        freed = max(0, before.process_rss_mb - after.process_rss_mb)
        
        with self._stats_lock:
            self._stats.total_cleanups += 1
            self._stats.memory_freed_mb += freed
            self._stats.gc_collections += 1
            self._stats.last_cleanup = datetime.now()
            
            if level in (MemoryPressure.CRITICAL, MemoryPressure.EMERGENCY):
                self._stats.emergency_cleanups += 1
        
        results["after_mb"] = after.process_rss_mb
        results["freed_mb"] = freed
        
        logger.info(
            f"Memory cleanup ({level.value if level else 'manual'}): "
            f"{before.process_rss_mb:.1f}MB -> {after.process_rss_mb:.1f}MB "
            f"(freed {freed:.1f}MB, callbacks={results['callbacks_called']}, "
            f"caches={results['caches_cleared']}, gc={results['gc_collected']})"
        )
        
        return results
    
    async def async_cleanup(self, level: MemoryPressure | None = None) -> dict:
        """Async cleanup that handles async callbacks."""
        before = self.get_system_memory()
        
        results = {
            "before_mb": before.process_rss_mb,
            "callbacks_called": 0,
            "async_callbacks": 0,
            "caches_cleared": 0,
            "gc_collected": 0,
        }
        
        # Call cleanup callbacks
        with self._lock:
            async_callbacks = list(self._cleanup_callbacks)
        for priority, callback, name in async_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                    results["async_callbacks"] += 1
                else:
                    callback()
                    results["callbacks_called"] += 1
            except Exception as e:
                logger.debug(f"Cleanup callback {name} error: {e}")
        
        # Clear caches
        self.get_target_cache_size()
        for cache in list(self._caches):
            try:
                if hasattr(cache, 'clear'):
                    cache.clear()
                    results["caches_cleared"] += 1
                elif hasattr(cache, 'clear_cache'):
                    cache.clear_cache()
                    results["caches_cleared"] += 1
            except Exception as e:
                logger.debug(f"Cache clear error: {e}")
        
        # Garbage collection
        gc_level = 2 if level in (MemoryPressure.CRITICAL, MemoryPressure.EMERGENCY) else 0
        results["gc_collected"] = gc.collect(gc_level)
        
        after = self.get_system_memory()
        freed = max(0, before.process_rss_mb - after.process_rss_mb)
        
        results["after_mb"] = after.process_rss_mb
        results["freed_mb"] = freed
        
        with self._stats_lock:
            self._stats.total_cleanups += 1
            self._stats.memory_freed_mb += freed
            self._stats.last_cleanup = datetime.now()
        
        return results
    
    def adjust_cache_sizes(self, target_size: int) -> None:
        with self._cache_callbacks_lock:
            callbacks_snapshot = list(self._cache_size_callbacks)
        for callback in callbacks_snapshot:
            try:
                callback(target_size)
            except Exception as e:
                logger.debug(f"Cache size callback error: {e}")
        with self._stats_lock:
            self._stats.cache_adjustments += 1
        logger.debug(f"Adjusted cache sizes to {target_size}")
    
    def get_check_interval(self) -> float:
        """Get check interval based on current pressure."""
        with self._lock:
            pressure = self._pressure
        intervals = {
            MemoryPressure.NORMAL: self.config.normal_check_interval,
            MemoryPressure.ELEVATED: self.config.elevated_check_interval,
            MemoryPressure.HIGH: self.config.high_check_interval,
            MemoryPressure.CRITICAL: self.config.critical_check_interval,
            MemoryPressure.EMERGENCY: self.config.emergency_check_interval,
        }
        return intervals[pressure]
    
    async def start(self) -> None:
        """Start the memory monitoring loop."""
        if self._running:
            return
        
        self._running = True
        self._start_time = time.time()
        self._task = create_tracked_task(self._monitor_loop(), name="dynamic_memory_monitor")
        
        initial = self.get_system_memory()
        logger.info(
            f"Dynamic memory manager started (max: {self.config.max_memory_mb/1024:.1f}GB, "
            f"process: {initial.process_rss_mb:.1f}MB, "
            f"system: {initial.ram_percent:.1f}% RAM, {initial.swap_percent:.1f}% swap)"
        )
    
    async def stop(self) -> None:
        """Stop the memory monitoring loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Dynamic memory manager stopped")
    
    async def _monitor_loop(self) -> None:
        while self._running:
            try:
                interval = self.get_check_interval()
                await asyncio.sleep(interval)
                
                info = self.get_system_memory()
                new_pressure = info.pressure
                
                with self._lock:
                    old_pressure = self._pressure
                
                if new_pressure != old_pressure:
                    with self._lock:
                        self._pressure = new_pressure
                    with self._stats_lock:
                        self._stats.pressure_changes += 1
                        self._stats.last_pressure = new_pressure
                    
                    logger.info(
                        f"Memory pressure changed: {old_pressure.value} -> {new_pressure.value} "
                        f"(process: {info.process_rss_mb:.1f}MB, "
                        f"RAM: {info.ram_percent:.1f}%, swap: {info.swap_percent:.1f}%)"
                    )
                    
                    new_cache_size = self.get_target_cache_size()
                    self.adjust_cache_sizes(new_cache_size)
                    
                    if _PRESSURE_ORDER.get(new_pressure, 0) > _PRESSURE_ORDER.get(old_pressure, 0):
                        await self.async_cleanup(new_pressure)
                
                with self._lock:
                    current_pressure = self._pressure
                if current_pressure in (MemoryPressure.HIGH, MemoryPressure.CRITICAL, MemoryPressure.EMERGENCY):
                    await self.async_cleanup(current_pressure)
                
                with self._stats_lock:
                    self._stats.uptime_seconds = time.time() - self._start_time
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Memory monitor error: {e}")
                await asyncio.sleep(5.0)
    
    def get_status(self) -> dict:
        """Get current memory status."""
        info = self.get_system_memory()
        with self._lock:
            pressure = self._pressure
        with self._stats_lock:
            stats = {
                "total_cleanups": self._stats.total_cleanups,
                "emergency_cleanups": self._stats.emergency_cleanups,
                "memory_freed_mb": self._stats.memory_freed_mb,
                "cache_adjustments": self._stats.cache_adjustments,
                "pressure_changes": self._stats.pressure_changes,
                "uptime_seconds": self._stats.uptime_seconds,
            }
        return {
            "pressure": pressure.value,
            "process": {
                "rss_mb": info.process_rss_mb,
                "percent": info.process_percent,
            },
            "system": {
                "ram_total_mb": info.total_ram_mb,
                "ram_available_mb": info.available_ram_mb,
                "ram_used_mb": info.used_ram_mb,
                "ram_percent": info.ram_percent,
                "swap_total_mb": info.total_swap_mb,
                "swap_used_mb": info.used_swap_mb,
                "swap_percent": info.swap_percent,
                "combined_percent": info.combined_percent,
            },
            "config": {
                "max_memory_mb": self.config.max_memory_mb,
                "target_cache_size": self.get_target_cache_size(),
            },
            "stats": stats,
            "thresholds": {
                "elevated": f"{self.config.elevated_threshold * 100:.0f}%",
                "high": f"{self.config.high_threshold * 100:.0f}%",
                "critical": f"{self.config.critical_threshold * 100:.0f}%",
                "emergency": f"{self.config.emergency_threshold * 100:.0f}%",
            },
        }


def get_dynamic_memory_manager() -> DynamicMemoryManager:
    """Get the global dynamic memory manager instance."""
    return DynamicMemoryManager.get_instance()


async def start_dynamic_memory_manager(
    max_memory_mb: float = 24 * 1024,
    **kwargs
) -> DynamicMemoryManager:
    """Start the dynamic memory manager with configured max memory."""
    config = DynamicMemoryConfig(max_memory_mb=max_memory_mb, **kwargs)
    manager = get_dynamic_memory_manager()
    manager.config = config
    await manager.start()
    return manager