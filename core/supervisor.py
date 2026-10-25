"""Server Supervisor for Jarvis.

Provides:
- Process health monitoring with auto-restart
- Graceful shutdown handling
- Memory pressure management (24GB max for 32GB system)
- LLM fallback on failures
- Runtime maintenance tasks
- Crash recovery
"""
import asyncio
import gc
import logging
import signal
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from core.helpers import create_tracked_task
from typing import Any, Callable

logger = logging.getLogger(__name__)


class SupervisorState(Enum):
    STARTING = "starting"
    RUNNING = "running"
    DEGRADED = "degraded"
    RESTARTING = "restarting"
    STOPPING = "stopping"
    STOPPED = "stopped"


@dataclass
class RestartPolicy:
    max_restarts: int = 5
    restart_window_seconds: float = 300.0
    backoff_base: float = 2.0
    backoff_max: float = 60.0
    initial_delay: float = 1.0


@dataclass
class MemoryConfig:
    # 32GB system with 8GB swap, max 24GB for Jarvis
    max_memory_mb: float = 24 * 1024.0
    critical_threshold: float = 0.85  # 20.4 GB
    emergency_threshold: float = 0.95  # 22.8 GB
    check_interval: float = 30.0


@dataclass
class SupervisorStats:
    starts: int = 0
    restarts: int = 0
    crashes: int = 0
    graceful_shutdowns: int = 0
    memory_cleanups: int = 0
    emergency_cleanups: int = 0
    llm_fallbacks: int = 0
    uptime_seconds: float = 0.0
    last_crash: datetime | None = None
    last_restart: datetime | None = None
    last_memory_cleanup: datetime | None = None
    restart_timestamps: list[float] = field(default_factory=list)


class ServerSupervisor:
    """Supervisor for keeping Jarvis running reliably.
    
    Features:
    - Auto-restart on crashes with backoff
    - Graceful shutdown on SIGTERM/SIGINT
    - Dynamic memory pressure management (24GB max for 32GB system)
    - LLM fallback management
    - Health monitoring
    - Maintenance task scheduling
    """
    
    _instance = None
    _lock = threading.Lock()
    
    MAX_RESTART_TIMESTAMPS = 50
    MAX_MAINTENANCE_TASKS = 20
    MAX_SHUTDOWN_HANDLERS = 20

    def __init__(
        self,
        restart_policy: RestartPolicy | None = None,
        memory_config: MemoryConfig | None = None,
        health_check_interval: float = 60.0,
    ):
        self.restart_policy = restart_policy or RestartPolicy()
        self.memory_config = memory_config or MemoryConfig()
        self.health_check_interval = health_check_interval
        
        self._state = SupervisorState.STOPPED
        self._stats = SupervisorStats()
        self._running = False
        self._shutdown_requested = False
        self._state_lock = threading.RLock()
        self._subsystems: dict[str, Any] = {}
        self._health_checkers: dict[str, Callable] = {}
        self._maintenance_tasks: list[Callable] = []
        self._shutdown_handlers: list[Callable] = []
        self._restart_count = 0
        self._last_restart_time = 0.0
        self._task: asyncio.Task | None = None
        self._memory_task: asyncio.Task | None = None
        self._maintenance_task: asyncio.Task | None = None
        self._start_time: float = 0.0
        self._loop: asyncio.AbstractEventLoop | None = None
        
        self._fallback_llm_models = ["glm-5", "qwen3-vl:235b", "qwen3.5:397b"]
        self._current_model_idx = 0
        
        self._dynamic_memory = None
        
    @classmethod
    def get_instance(cls) -> "ServerSupervisor":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def register_subsystem(self, name: str, instance: Any, health_checker: Callable | None = None) -> None:
        with self._state_lock:
            self._subsystems[name] = instance
            if health_checker:
                self._health_checkers[name] = health_checker
        logger.debug(f"Registered subsystem: {name}")
    
    def unregister_subsystem(self, name: str) -> bool:
        """Unregister a subsystem."""
        with self._state_lock:
            if name in self._subsystems:
                del self._subsystems[name]
                self._health_checkers.pop(name, None)
                logger.debug(f"Unregistered subsystem: {name}")
                return True
            return False
    
    def register_maintenance_task(self, task: Callable) -> None:
        with self._state_lock:
            if len(self._maintenance_tasks) >= self.MAX_MAINTENANCE_TASKS:
                self._maintenance_tasks = self._maintenance_tasks[-(self.MAX_MAINTENANCE_TASKS - 1):]
            self._maintenance_tasks.append(task)
        logger.debug("Registered maintenance task")
    
    def unregister_maintenance_task(self, task: Callable) -> bool:
        """Unregister a maintenance task."""
        with self._state_lock:
            if task in self._maintenance_tasks:
                self._maintenance_tasks.remove(task)
                logger.debug("Unregistered maintenance task")
                return True
            return False
    
    def register_shutdown_handler(self, handler: Callable) -> None:
        with self._state_lock:
            if len(self._shutdown_handlers) >= self.MAX_SHUTDOWN_HANDLERS:
                self._shutdown_handlers = self._shutdown_handlers[-(self.MAX_SHUTDOWN_HANDLERS - 1):]
            self._shutdown_handlers.append(handler)
        logger.debug("Registered shutdown handler")
    
    def unregister_shutdown_handler(self, handler: Callable) -> bool:
        """Unregister a shutdown handler."""
        with self._state_lock:
            if handler in self._shutdown_handlers:
                self._shutdown_handlers.remove(handler)
                logger.debug("Unregistered shutdown handler")
                return True
            return False
    
    def get_memory_usage(self) -> dict:
        """Get comprehensive memory usage including system memory."""
        try:
            import psutil
            process = psutil.Process()
            mem = process.memory_info()
            vm = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            return {
                "rss_mb": mem.rss / (1024 * 1024),
                "vms_mb": mem.vms / (1024 * 1024),
                "percent": process.memory_percent(),
                "system_ram_percent": vm.percent,
                "system_ram_available_mb": vm.available / (1024 * 1024),
                "system_swap_percent": swap.percent,
                "system_swap_used_mb": swap.used / (1024 * 1024),
            }
        except Exception:
            return {"rss_mb": 0, "vms_mb": 0, "percent": 0}
    
    def get_system_memory_status(self) -> dict:
        """Get full system memory status for 32GB RAM + 8GB swap."""
        try:
            import psutil
            vm = psutil.virtual_memory()
            swap = psutil.swap_memory()
            process = psutil.Process()
            
            total_ram_mb = vm.total / (1024 * 1024)
            total_swap_mb = swap.total / (1024 * 1024)
            total_used_mb = vm.used / (1024 * 1024) + swap.used / (1024 * 1024)
            available_mb = vm.available / (1024 * 1024) + (swap.total - swap.used) / (1024 * 1024)
            
            return {
                "total_ram_mb": total_ram_mb,
                "total_swap_mb": total_swap_mb,
                "total_used_mb": total_used_mb,
                "used_ram_mb": vm.used / (1024 * 1024),
                "used_swap_mb": swap.used / (1024 * 1024),
                "available_mb": available_mb,
                "process_rss_mb": process.memory_info().rss / (1024 * 1024),
                "process_percent": process.memory_percent(),
                "ram_percent": vm.percent,
                "swap_percent": swap.percent,
            }
        except Exception:
            return {}
    
    def check_memory_pressure(self) -> tuple[bool, float, str]:
        """Check memory pressure against 24GB limit."""
        usage = self.get_memory_usage()
        rss_mb = usage["rss_mb"]
        max_mb = self.memory_config.max_memory_mb
        
        ratio = rss_mb / max_mb if max_mb > 0 else 0
        
        if ratio >= self.memory_config.emergency_threshold:
            level = "EMERGENCY"
            is_pressure = True
        elif ratio >= self.memory_config.critical_threshold:
            level = "CRITICAL"
            is_pressure = True
        else:
            level = "NORMAL"
            is_pressure = False
        
        return is_pressure, rss_mb, level
    
    async def force_memory_cleanup(self, level: str = "NORMAL") -> dict:
        """Force memory cleanup with dynamic manager integration."""
        before = self.get_memory_usage()
        
        # Use dynamic memory manager if available
        if self._dynamic_memory:
            try:
                from core.dynamic_memory import MemoryPressure
                pressure_map = {
                    "NORMAL": None,
                    "CRITICAL": MemoryPressure.CRITICAL,
                    "EMERGENCY": MemoryPressure.EMERGENCY,
                }
                result = await self._dynamic_memory.async_cleanup(pressure_map.get(level))
                self._stats.memory_cleanups += 1
                if level in ("CRITICAL", "EMERGENCY"):
                    self._stats.emergency_cleanups += 1
                self._stats.last_memory_cleanup = datetime.now()
                
                result["system"] = self.get_system_memory_status()
                return result
            except Exception as e:
                logger.warning(f"Dynamic memory cleanup failed, falling back: {e}")
        
        # Fallback cleanup
        with self._state_lock:
            subsystems_snapshot = dict(self._subsystems)
            tasks_snapshot = list(self._maintenance_tasks)
        for name, subsystem in subsystems_snapshot.items():
            try:
                if hasattr(subsystem, 'cleanup'):
                    await subsystem.cleanup() if asyncio.iscoroutinefunction(subsystem.cleanup) else subsystem.cleanup()
                elif hasattr(subsystem, 'clear_cache'):
                    await subsystem.clear_cache() if asyncio.iscoroutinefunction(subsystem.clear_cache) else subsystem.clear_cache()
                elif hasattr(subsystem, 'clear'):
                    await subsystem.clear() if asyncio.iscoroutinefunction(subsystem.clear) else subsystem.clear()
            except Exception as e:
                logger.debug(f"Cleanup error for {name}: {e}")
        
        for task in tasks_snapshot:
            try:
                if asyncio.iscoroutinefunction(task):
                    await task()
                else:
                    task()
            except Exception as e:
                logger.debug(f"Maintenance task error: {e}")
        
        # Aggressive GC for critical/emergency
        gc_level = 2 if level in ("CRITICAL", "EMERGENCY") else 0
        collected = gc.collect(gc_level)
        
        after = self.get_memory_usage()
        freed = max(0, before["rss_mb"] - after["rss_mb"])
        
        self._stats.memory_cleanups += 1
        if level in ("CRITICAL", "EMERGENCY"):
            self._stats.emergency_cleanups += 1
        self._stats.last_memory_cleanup = datetime.now()
        
        logger.info(
            f"Memory cleanup ({level}): {before['rss_mb']:.1f}MB -> {after['rss_mb']:.1f}MB "
            f"(freed {freed:.1f}MB, gc={collected})"
        )
        
        return {
            "before_mb": before["rss_mb"],
            "after_mb": after["rss_mb"],
            "freed_mb": freed,
            "gc_collected": collected,
            "level": level,
            "system": self.get_system_memory_status(),
        }
    
    def get_next_fallback_model(self) -> str | None:
        with self._state_lock:
            self._current_model_idx += 1
            if self._current_model_idx < len(self._fallback_llm_models):
                model = self._fallback_llm_models[self._current_model_idx]
                self._stats.llm_fallbacks += 1
                logger.warning(f"Falling back to LLM model: {model}")
                return model
            self._current_model_idx = 0
            return None
    
    def reset_model_fallback(self) -> None:
        with self._state_lock:
            self._current_model_idx = 0
    
    def record_crash(self, error: Exception) -> None:
        self._stats.crashes += 1
        self._stats.last_crash = datetime.now()
        logger.error(f"Recorded crash: {error}")
    
    def should_restart(self) -> tuple[bool, float]:
        now = time.time()
        window = self.restart_policy.restart_window_seconds
        self._stats.restart_timestamps = [
            t for t in self._stats.restart_timestamps
            if now - t < window
        ]
        if len(self._stats.restart_timestamps) > self.MAX_RESTART_TIMESTAMPS:
            self._stats.restart_timestamps = self._stats.restart_timestamps[-self.MAX_RESTART_TIMESTAMPS:]
        
        if len(self._stats.restart_timestamps) >= self.restart_policy.max_restarts:
            logger.error(f"Max restarts ({self.restart_policy.max_restarts}) reached in window")
            return False, 0.0
        
        if self._stats.restart_timestamps:
            count = len(self._stats.restart_timestamps)
            delay = min(
                self.restart_policy.initial_delay * (self.restart_policy.backoff_base ** count),
                self.restart_policy.backoff_max
            )
        else:
            delay = self.restart_policy.initial_delay
        
        return True, delay
    
    def record_restart(self) -> None:
        self._stats.restarts += 1
        self._stats.last_restart = datetime.now()
        self._stats.restart_timestamps.append(time.time())
    
    async def health_check(self) -> dict[str, bool]:
        with self._state_lock:
            checkers = dict(self._health_checkers)
        results = {}
        for name, checker in checkers.items():
            try:
                if asyncio.iscoroutinefunction(checker):
                    results[name] = await checker()
                else:
                    results[name] = checker()
            except Exception as e:
                results[name] = False
                logger.warning(f"Health check failed for {name}: {e}")
        return results
    
    def _setup_signal_handlers(self) -> None:
        def handle_shutdown(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown")
            self._shutdown_requested = True
            # Schedule shutdown on the event loop instead of creating task from signal handler
            if self._loop and not self._loop.is_closed():
                from core.helpers import create_tracked_task
                self._loop.call_soon_threadsafe(
                    lambda: create_tracked_task(self.shutdown(), name="supervisor_shutdown")
                )
        
        signal.signal(signal.SIGTERM, handle_shutdown)
        signal.signal(signal.SIGINT, handle_shutdown)
        logger.debug("Signal handlers registered")
    
    async def start(self) -> None:
        with self._state_lock:
            if self._running:
                return
            self._running = True
            self._state = SupervisorState.STARTING
            self._start_time = time.time()
            self._loop = asyncio.get_running_loop()
            self._stats.starts += 1
        
        self._setup_signal_handlers()
        
        # Start dynamic memory manager
        try:
            from core.dynamic_memory import start_dynamic_memory_manager
            self._dynamic_memory = await start_dynamic_memory_manager(
                max_memory_mb=self.memory_config.max_memory_mb
            )
            logger.info(f"Dynamic memory manager started (max: {self.memory_config.max_memory_mb/1024:.1f}GB)")
        except Exception as e:
            logger.warning(f"Dynamic memory manager not available: {e}")
        
        self._task = create_tracked_task(self._supervisor_loop(), name="supervisor_loop")
        self._memory_task = create_tracked_task(self._memory_monitor_loop(), name="memory_monitor_loop")
        self._maintenance_task = create_tracked_task(self._maintenance_loop(), name="maintenance_loop")
        
        with self._state_lock:
            self._state = SupervisorState.RUNNING
        
        # Log system memory status
        sys_mem = self.get_system_memory_status()
        logger.info(
            f"Server supervisor started "
            f"(RAM: {sys_mem.get('ram_percent', 0):.1f}%, "
            f"Swap: {sys_mem.get('swap_percent', 0):.1f}%, "
            f"Process: {sys_mem.get('process_rss_mb', 0):.1f}MB)"
        )
    
    async def shutdown(self, timeout: float = 30.0) -> None:
        if not self._running:
            return
        
        with self._state_lock:
            self._state = SupervisorState.STOPPING
            self._shutdown_requested = True
            self._running = False
            handlers = list(self._shutdown_handlers)
            subsystems_snapshot = dict(self._subsystems)
        
        logger.info("Starting graceful shutdown...")
        
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await asyncio.wait_for(handler(), timeout=timeout / 2)
                else:
                    handler()
            except asyncio.CancelledError:
                logger.warning("Shutdown handler cancelled")
            except asyncio.TimeoutError:
                logger.warning("Shutdown handler timed out")
            except Exception as e:
                logger.error(f"Shutdown handler error: {e}")
        
        for name, subsystem in subsystems_snapshot.items():
            try:
                if hasattr(subsystem, 'stop'):
                    if asyncio.iscoroutinefunction(subsystem.stop):
                        await asyncio.wait_for(subsystem.stop(), timeout=5.0)
                    else:
                        subsystem.stop()
                elif hasattr(subsystem, 'close'):
                    if asyncio.iscoroutinefunction(subsystem.close):
                        await asyncio.wait_for(subsystem.close(), timeout=5.0)
                    else:
                        subsystem.close()
            except asyncio.CancelledError:
                logger.debug(f"Subsystem {name} stop cancelled")
            except Exception as e:
                logger.warning(f"Error stopping {name}: {e}")
        
        for task in [self._task, self._memory_task, self._maintenance_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        self._stats.graceful_shutdowns += 1
        self._stats.uptime_seconds = time.time() - self._start_time
        self._state = SupervisorState.STOPPED
        
        logger.info(f"Shutdown complete. Uptime: {self._stats.uptime_seconds:.1f}s")
    
    async def _supervisor_loop(self) -> None:
        while True:
            with self._state_lock:
                if not self._running or self._shutdown_requested:
                    break
            try:
                await asyncio.sleep(self.health_check_interval)
                
                with self._state_lock:
                    current_state = self._state
                    _is_running = self._running
                
                if current_state == SupervisorState.RUNNING:
                    results = await self.health_check()
                    unhealthy = [k for k, v in results.items() if not v]
                    
                    if unhealthy:
                        with self._state_lock:
                            self._state = SupervisorState.DEGRADED
                        logger.warning(f"Unhealthy subsystems: {unhealthy}")
                    else:
                        self._stats.uptime_seconds = time.time() - self._start_time
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Supervisor loop error: {e}")
    
    async def _memory_monitor_loop(self) -> None:
        while True:
            with self._state_lock:
                if not self._running or self._shutdown_requested:
                    break
            try:
                interval = self.memory_config.check_interval
                await asyncio.sleep(interval)
                
                is_pressure, rss_mb, level = self.check_memory_pressure()
                
                if is_pressure:
                    max_mb = self.memory_config.max_memory_mb
                    logger.warning(
                        f"Memory {level}: {rss_mb:.1f}MB / {max_mb:.1f}MB "
                        f"({rss_mb/max_mb*100:.1f}%)"
                    )
                    await self.force_memory_cleanup(level)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Memory monitor error: {e}")
    
    async def _maintenance_loop(self) -> None:
        maintenance_interval = 300.0
        
        while True:
            with self._state_lock:
                if not self._running or self._shutdown_requested:
                    break
            try:
                await asyncio.sleep(maintenance_interval)
                
                self._stats.restart_timestamps = [
                    t for t in self._stats.restart_timestamps
                    if time.time() - t < self.restart_policy.restart_window_seconds
                ]
                if len(self._stats.restart_timestamps) > self.MAX_RESTART_TIMESTAMPS:
                    self._stats.restart_timestamps = self._stats.restart_timestamps[-self.MAX_RESTART_TIMESTAMPS:]
                
                gc.collect(0)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Maintenance loop error: {e}")
    
    def get_status(self) -> dict:
        sys_mem = self.get_system_memory_status()
        with self._state_lock:
            state_value = self._state.value
            subsystem_keys = list(self._subsystems.keys())
            current_idx = self._current_model_idx
        return {
            "state": state_value,
            "uptime_seconds": self._stats.uptime_seconds,
            "stats": {
                "starts": self._stats.starts,
                "restarts": self._stats.restarts,
                "crashes": self._stats.crashes,
                "graceful_shutdowns": self._stats.graceful_shutdowns,
                "memory_cleanups": self._stats.memory_cleanups,
                "emergency_cleanups": self._stats.emergency_cleanups,
                "llm_fallbacks": self._stats.llm_fallbacks,
            },
            "memory": {
                "process": self.get_memory_usage(),
                "system": sys_mem,
                "config": {
                    "max_memory_mb": self.memory_config.max_memory_mb,
                    "critical_threshold": f"{self.memory_config.critical_threshold * 100:.0f}%",
                    "emergency_threshold": f"{self.memory_config.emergency_threshold * 100:.0f}%",
                },
            },
            "subsystems": subsystem_keys,
            "current_llm_model": self._fallback_llm_models[current_idx] if current_idx < len(self._fallback_llm_models) else None,
            "dynamic_memory_active": self._dynamic_memory is not None,
        }


def get_supervisor() -> ServerSupervisor:
    return ServerSupervisor.get_instance()


async def start_supervisor(**kwargs) -> ServerSupervisor:
    supervisor = get_supervisor()
    await supervisor.start()
    return supervisor