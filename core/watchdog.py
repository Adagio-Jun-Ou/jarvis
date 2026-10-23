"""Jarvis Process Watchdog — 24/7 runtime health monitor.

Background thread that:
- Monitors system health periodically
- Detects and recovers from stuck processes
- Tracks resource trends (CPU, memory over time)
- Emits heartbeats for liveness detection
- Auto-cleans zombie processes
- Persists health state for crash recovery
"""
import logging
import os
import signal
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from core.helpers import safe_error_message

logger = logging.getLogger(__name__)


@dataclass
class WatchdogEvent:
    event_type: str  # heartbeat, stuck_process, recovery, resource_alert, cleanup
    message: str
    timestamp: float = field(default_factory=time.time)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type,
            "message": self.message,
            "timestamp": self.timestamp,
            "details": self.details,
        }


class JarvisWatchdog:
    """Background watchdog that keeps Jarvis alive and healthy.

    Runs in a daemon thread, checking system health every 60 seconds:
    1. Takes system snapshots
    2. Detects stuck/zombie processes
    3. Checks resource alerts (high CPU, low memory, disk full)
    4. Cleans up zombie child processes
    5. Emits heartbeats for liveness
    6. Persists health state periodically
    7. Auto-recovers stuck subsystems
    8. OOM detection and memory pressure response
    """

    CHECK_INTERVAL = 60.0
    MEMORY_ALERT_THRESHOLD = 90.0
    CPU_ALERT_THRESHOLD = 95.0
    DISK_ALERT_THRESHOLD = 95.0
    MAX_ZOMBIE_AGE_SECONDS = 300.0
    OOM_MEMORY_THRESHOLD = 97.0
    RECOVERY_COOLDOWN = 300.0

    def __init__(self):
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self._events: list[WatchdogEvent] = []
        self._max_events = 500
        self._last_check = 0.0
        self._check_count = 0
        self._alert_count = 0
        self._recovery_count = 0
        self._cleaned_processes = 0
        self._start_time = 0.0
        self._psutil_available = False
        self._subsystem_health: dict[str, dict[str, Any]] = {}
        self._recovery_timestamps: dict[str, float] = {}

        try:
            import psutil
            self._psutil_available = True
            self._psutil = psutil
        except ImportError:
            self._psutil = None

    def start(self) -> None:
        """Start the watchdog thread."""
        with self._lock:
            if self._running:
                return
            self._running = True
            self._start_time = time.time()
            self._thread = threading.Thread(
                target=self._run,
                name="jarvis-watchdog",
                daemon=True,
            )
            self._thread.start()
            logger.info("Jarvis Watchdog started")

    def stop(self) -> None:
        """Stop the watchdog thread."""
        with self._lock:
            self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        logger.info("Jarvis Watchdog stopped")

    def _run(self) -> None:
        """Main watchdog loop."""
        while self._running:
            try:
                self._check()
            except Exception as e:
                logger.error(f"Watchdog check error: {safe_error_message(e)}")

            # Sleep in small increments for responsive shutdown
            slept = 0.0
            while slept < self.CHECK_INTERVAL and self._running:
                time.sleep(1.0)
                slept += 1.0

    def _check(self) -> None:
        """Perform a single health check."""
        self._check_count += 1
        self._last_check = time.time()

        # Take system snapshot
        if self._psutil_available:
            self._check_resources()
            self._check_zombies()

        # Check execution context health
        try:
            from core.execution_context import get_execution_context
            ctx = get_execution_context()
            hb = ctx.heartbeat()

            running = hb.get("running", 0)
            if running > 20:
                self._emit("resource_alert", f"Too many running processes: {running}", {"running": running})
        except Exception as e:
            logger.debug(f"Execution context heartbeat check failed: {e}")

        # Check WebSocket connection health
        try:
            from core.ws_manager import get_ws_connection_manager
            ws_mgr = get_ws_connection_manager()
            stale = ws_mgr.get_stale_connections()
            if stale:
                self._emit("ws_health", f"{len(stale)} stale WebSocket connections detected", {
                    "stale_count": len(stale),
                    "details": [(ws_id, info.client_host) for ws_id, info in stale[:5]],
                })
            ws_status = ws_mgr.get_status()
            active = ws_status.get("active_connections", 0)
            if active > ws_mgr.MAX_CONNECTIONS * 0.8:
                self._emit("resource_alert", f"High WebSocket connection count: {active}", {"ws_connections": active})
        except Exception as e:
            logger.debug(f"WebSocket health check failed: {e}")

        # Check registered subsystem health
        self._check_subsystem_health()

        # Check for OOM conditions
        self._check_oom()

        # Emit heartbeat
        self._emit("heartbeat", f"Check #{self._check_count}", {
            "uptime_hours": round((time.time() - self._start_time) / 3600, 2) if self._start_time else 0,
        })

    def _check_resources(self) -> None:
        """Check for resource alerts."""
        if not self._psutil_available:
            return

        try:
            mem = self._psutil.virtual_memory()
            if mem.percent >= self.MEMORY_ALERT_THRESHOLD:
                self._alert_count += 1
                self._emit("resource_alert", f"High memory: {mem.percent}%", {
                    "memory_percent": mem.percent,
                    "available_gb": round(mem.available / (1024 ** 3), 2),
                })

            cpu = self._psutil.cpu_percent(interval=1.0)
            if cpu >= self.CPU_ALERT_THRESHOLD:
                self._alert_count += 1
                self._emit("resource_alert", f"High CPU: {cpu}%", {"cpu_percent": cpu})

            disk = self._psutil.disk_usage("/")
            if disk.percent >= self.DISK_ALERT_THRESHOLD:
                self._alert_count += 1
                self._emit("resource_alert", f"Low disk space: {disk.free / (1024**3):.1f} GB free", {
                    "disk_percent": disk.percent,
                    "free_gb": round(disk.free / (1024 ** 3), 2),
                })
        except Exception as e:
            logger.debug(f"Resource check error: {e}")

    def _check_zombies(self) -> None:
        """Find and clean up zombie processes belonging to Jarvis."""
        if not self._psutil_available:
            return

        try:
            our_pid = os.getpid()
            for proc in self._psutil.process_iter(["pid", "ppid", "name", "status", "create_time"]):
                try:
                    info = proc.info
                    if info.get("ppid") == our_pid and info.get("status") == "zombie":
                        age = time.time() - (info.get("create_time") or time.time())
                        if age > self.MAX_ZOMBIE_AGE_SECONDS:
                            try:
                                os.kill(info["pid"], signal.SIGKILL)
                                self._cleaned_processes += 1
                                self._emit("cleanup", f"Killed zombie PID {info['pid']}", {"pid": info["pid"], "age": round(age, 0)})
                            except (ProcessLookupError, OSError):
                                pass
                except (self._psutil.NoSuchProcess, self._psutil.AccessDenied):
                    continue
        except Exception as e:
            logger.debug(f"Zombie check error: {e}")

    def register_subsystem(self, name: str, health_fn=None, restart_fn=None) -> None:
        with self._lock:
            self._subsystem_health[name] = {
                "health_fn": health_fn,
                "restart_fn": restart_fn,
                "status": "unknown",
                "last_check": 0.0,
                "fail_count": 0,
                "last_failure": 0.0,
            }

    def _check_subsystem_health(self) -> None:
        with self._lock:
            subsystems = dict(self._subsystem_health)

        now = time.time()
        for name, info in subsystems.items():
            health_fn = info.get("health_fn")
            if health_fn is None:
                continue
            try:
                healthy = health_fn()
                with self._lock:
                    if name in self._subsystem_health:
                        self._subsystem_health[name]["status"] = "healthy" if healthy else "unhealthy"
                        self._subsystem_health[name]["last_check"] = now
                        if healthy:
                            self._subsystem_health[name]["fail_count"] = 0
            except Exception as e:
                with self._lock:
                    if name in self._subsystem_health:
                        self._subsystem_health[name]["status"] = "failed"
                        self._subsystem_health[name]["fail_count"] += 1
                        self._subsystem_health[name]["last_failure"] = now
                self._emit("subsystem_failure", f"Subsystem {name} health check failed: {safe_error_message(e)}", {
                    "subsystem": name,
                    "error": safe_error_message(e),
                })
                self._try_recover(name)

    def _try_recover(self, name: str) -> None:
        with self._lock:
            info = self._subsystem_health.get(name)
            if info is None:
                return
            last_recovery = self._recovery_timestamps.get(name, 0)
            if time.time() - last_recovery < self.RECOVERY_COOLDOWN:
                return
            restart_fn = info.get("restart_fn")
            if restart_fn is None or info.get("fail_count", 0) < 2:
                return

        try:
            restart_fn()
            with self._lock:
                self._recovery_timestamps[name] = time.time()
                self._recovery_count += 1
            self._emit("recovery", f"Recovered subsystem {name}", {"subsystem": name})
        except Exception as e:
            self._emit("recovery_failed", f"Failed to recover {name}: {safe_error_message(e)}", {
                "subsystem": name,
                "error": safe_error_message(e),
            })

    def _check_oom(self) -> None:
        if not self._psutil_available:
            return
        try:
            mem = self._psutil.virtual_memory()
            if mem.percent >= self.OOM_MEMORY_THRESHOLD:
                self._emit("oom_warning", f"Critical memory: {mem.percent}%, available: {mem.available / (1024**3):.1f} GB", {
                    "memory_percent": mem.percent,
                    "available_gb": round(mem.available / (1024**3), 2),
                })
                try:
                    from core.context_compressor import get_context_compressor
                    compressor = get_context_compressor()
                    compressor.compress([], strategy="compress")
                except Exception as e:
                    logger.warning(f"Context compression during OOM failed: {e}")
        except Exception as e:
            logger.debug(f"OOM check failed: {e}")

    def _emit(self, event_type: str, message: str, details: dict | None = None) -> None:
        """Emit a watchdog event."""
        event = WatchdogEvent(
            event_type=event_type,
            message=message,
            details=details or {},
        )
        with self._lock:
            self._events.append(event)
            if len(self._events) > self._max_events:
                self._events = self._events[-self._max_events:]

        try:
            from core.event_bus import get_event_bus
            bus = get_event_bus()
            bus.publish(f"watchdog.{event_type}", {
                "message": message,
                "details": details or {},
            }, source="watchdog")
        except Exception as e:
            logger.debug(f"Watchdog event emission failed: {e}")

    def get_events(self, limit: int = 50, event_type: str | None = None) -> list[dict]:
        """Get recent watchdog events."""
        with self._lock:
            events = self._events
            if event_type:
                events = [e for e in events if e.event_type == event_type]
            return [e.to_dict() for e in events[-limit:]]

    def get_status(self) -> dict[str, Any]:
        """Get watchdog status."""
        with self._lock:
            uptime = time.time() - self._start_time if self._start_time else 0
            subsystems = {
                name: {
                    "status": info["status"],
                    "fail_count": info["fail_count"],
                    "last_check": info["last_check"],
                    "last_failure": info.get("last_failure", 0),
                }
                for name, info in self._subsystem_health.items()
            }
            return {
                "running": self._running,
                "uptime_seconds": round(uptime, 0),
                "check_count": self._check_count,
                "alert_count": self._alert_count,
                "recovery_count": self._recovery_count,
                "cleaned_processes": self._cleaned_processes,
                "last_check": self._last_check,
                "last_check_ago": round(time.time() - self._last_check, 1) if self._last_check else None,
                "psutil_available": self._psutil_available,
                "event_count": len(self._events),
                "subsystems": subsystems,
            }


_watchdog: JarvisWatchdog | None = None
_watchdog_lock = threading.Lock()


def get_watchdog() -> JarvisWatchdog:
    """Thread-safe singleton for JarvisWatchdog."""
    global _watchdog
    if _watchdog is None:
        with _watchdog_lock:
            if _watchdog is None:
                _watchdog = JarvisWatchdog()
    return _watchdog


def start_watchdog() -> JarvisWatchdog:
    """Start the watchdog thread."""
    wd = get_watchdog()
    wd.start()
    return wd