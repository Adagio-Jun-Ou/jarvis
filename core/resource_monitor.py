"""Resource Monitor - Monitor and optimize laptop resource usage.

Tracks CPU, memory, and disk usage to help Jarvis optimize its behavior
for laptop constraints.
"""
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

logger = logging.getLogger(__name__)


@dataclass
class ResourceSnapshot:
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_percent: float
    process_count: int
    load_average: tuple[float, float, float]


@dataclass
class ResourceThresholds:
    cpu_high: float = 80.0
    cpu_critical: float = 95.0
    memory_high: float = 80.0
    memory_critical: float = 90.0
    disk_high: float = 85.0
    disk_critical: float = 95.0


class ResourceMonitor:
    """Monitor system resources for laptop optimization."""

    MAX_HISTORY = 100
    CHECK_INTERVAL = 30.0

    def __init__(self, thresholds: ResourceThresholds | None = None):
        self.thresholds = thresholds or ResourceThresholds()
        self.history: deque[ResourceSnapshot] = deque(maxlen=self.MAX_HISTORY)
        self._lock = threading.RLock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._callbacks: list[Callable] = []

        self._psutil = None
        try:
            import psutil
            self._psutil = psutil
        except ImportError:
            logger.debug("psutil not available, using basic monitoring")

    def _get_load_average(self) -> tuple[float, float, float]:
        """1/5/15-minute load average, or zeros where the OS has no concept of it.

        POSIX-only via os.getloadavg(); psutil emulates it on Windows, so prefer
        that when present before falling back to zeros.
        """
        getloadavg = getattr(os, "getloadavg", None)
        if getloadavg is not None:
            try:
                return getloadavg()
            except OSError:
                pass
        psutil_getloadavg = getattr(self._psutil, "getloadavg", None) if self._psutil else None
        if psutil_getloadavg is not None:
            try:
                return psutil_getloadavg()
            except Exception:
                pass
        return (0.0, 0.0, 0.0)

    def _get_snapshot(self) -> ResourceSnapshot:
        """Get current resource snapshot."""
        now = datetime.now()

        if self._psutil:
            cpu = self._psutil.cpu_percent(interval=0.1)
            mem = self._psutil.virtual_memory()
            disk = self._psutil.disk_usage(os.path.abspath(os.sep))
            process_count = len(self._psutil.pids())
            # os.getloadavg() does not exist on Windows. It was called
            # unconditionally, raising AttributeError out of every snapshot, so
            # `history` never received a sample and every resource-pressure
            # check (is_under_pressure / should_throttle / get_recommendations)
            # failed — throttling silently never engaged.
            load = self._get_load_average()

            return ResourceSnapshot(
                timestamp=now,
                cpu_percent=cpu,
                memory_percent=mem.percent,
                memory_used_gb=mem.used / (1024**3),
                memory_total_gb=mem.total / (1024**3),
                disk_percent=disk.percent,
                process_count=process_count,
                load_average=load,
            )
        else:
            try:
                with open('/proc/loadavg', encoding='utf-8') as f:
                    load_data = f.read().split()
                    if len(load_data) >= 3:
                        load = (float(load_data[0]), float(load_data[1]), float(load_data[2]))
                    else:
                        load = (0.0, 0.0, 0.0)
            except (FileNotFoundError, ValueError, IndexError):
                load = (0.0, 0.0, 0.0)

            try:
                with open('/proc/meminfo', encoding='utf-8') as f:
                    meminfo = dict(line.split(':', 1) for line in f if ':' in line)
                    
                    mem_total_str = meminfo.get('MemTotal', '0 kB')
                    mem_free_str = meminfo.get('MemFree', '0 kB')
                    mem_available_str = meminfo.get('MemAvailable', mem_free_str)
                    
                    def parse_mem(value: str) -> float:
                        parts = value.strip().split()
                        if len(parts) >= 1:
                            val = float(parts[0])
                            if len(parts) >= 2 and parts[1] == 'kB':
                                val = val / (1024 ** 2)
                            elif len(parts) >= 2 and parts[1] == 'mB':
                                val = val / 1024
                            return val
                        return 0.0
                    
                    mem_total = parse_mem(mem_total_str)
                    mem_available = parse_mem(mem_available_str)
                    
                    if mem_total <= 0:
                        mem_total = 0.0
                        mem_available = 0.0
                    
                    mem_used = mem_total - mem_available
                    mem_percent = (mem_used / mem_total) * 100 if mem_total > 0 else 0.0
            except (FileNotFoundError, ValueError, KeyError, IndexError) as e:
                logger.debug(f"Error reading meminfo: {e}")
                mem_total = 0.0
                mem_used = 0.0
                mem_percent = 0.0

            return ResourceSnapshot(
                timestamp=now,
                cpu_percent=0.0,
                memory_percent=mem_percent,
                memory_used_gb=mem_used,
                memory_total_gb=mem_total,
                disk_percent=0.0,
                process_count=0,
                load_average=load,
            )

    def get_current(self) -> ResourceSnapshot:
        """Get current resource state."""
        return self._get_snapshot()

    def is_under_pressure(self) -> tuple[bool, str]:
        """Check if system is under resource pressure."""
        snap = self._get_snapshot()

        if snap.memory_percent > self.thresholds.memory_critical:
            return True, f"Critical memory pressure: {snap.memory_percent:.1f}%"
        if snap.memory_percent > self.thresholds.memory_high:
            return True, f"High memory usage: {snap.memory_percent:.1f}%"

        if snap.cpu_percent > self.thresholds.cpu_critical:
            return True, f"Critical CPU usage: {snap.cpu_percent:.1f}%"
        if snap.cpu_percent > self.thresholds.cpu_high:
            return True, f"High CPU usage: {snap.cpu_percent:.1f}%"

        if snap.disk_percent > self.thresholds.disk_critical:
            return True, f"Critical disk usage: {snap.disk_percent:.1f}%"

        return False, "System resources OK"

    def get_recommendations(self) -> list[str]:
        """Get recommendations for optimizing resource usage."""
        recommendations = []
        snap = self._get_snapshot()

        if snap.memory_percent > self.thresholds.memory_high:
            recommendations.append("Reduce background learning intensity")
            recommendations.append("Trigger memory cleanup")

        if snap.cpu_percent > self.thresholds.cpu_high:
            recommendations.append("Slow down daydream frequency")
            recommendations.append("Reduce parallel task count")

        if snap.disk_percent > self.thresholds.disk_high:
            recommendations.append("Clear old logs and caches")
            recommendations.append("Archive old memory files")

        return recommendations

    def should_throttle(self) -> bool:
        """Check if Jarvis should throttle its activities."""
        under_pressure, _ = self.is_under_pressure()
        return under_pressure

    def start_monitoring(self) -> None:
        """Start background monitoring."""
        with self._lock:
            if self._running:
                return
            self._running = True

        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("Resource monitor started")

    def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Resource monitor stopped")

    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            try:
                snap = self._get_snapshot()

                with self._lock:
                    self.history.append(snap)
                    callbacks_snapshot = list(self._callbacks)

                under_pressure, msg = self.is_under_pressure()
                if under_pressure:
                    logger.warning(f"Resource pressure: {msg}")
                    for callback in callbacks_snapshot:
                        try:
                            callback(snap, msg)
                        except Exception as e:
                            logger.debug(f"Callback error: {e}")

            except Exception as e:
                logger.debug(f"Monitor error: {e}")

            time.sleep(self.CHECK_INTERVAL)

    def register_callback(self, callback: Callable) -> None:
        """Register callback for pressure events."""
        with self._lock:
            self._callbacks.append(callback)

    def unregister_callback(self, callback: Callable) -> bool:
        """Unregister a pressure callback."""
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)
                return True
        return False

    def get_statistics(self) -> dict:
        """Get resource statistics."""
        with self._lock:
            if not self.history:
                return {"status": "no_data"}

            recent = list(self.history)[-10:]
            
            if not recent:
                return {"status": "no_data"}

            avg_cpu = sum(s.cpu_percent for s in recent) / len(recent) if recent else 0.0
            avg_mem = sum(s.memory_percent for s in recent) / len(recent) if recent else 0.0
            max_cpu = max((s.cpu_percent for s in recent), default=0.0)
            max_mem = max((s.memory_percent for s in recent), default=0.0)

            return {
                "current_cpu": recent[-1].cpu_percent,
                "current_memory": recent[-1].memory_percent,
                "average_cpu": avg_cpu,
                "average_memory": avg_mem,
                "max_cpu": max_cpu,
                "max_memory": max_mem,
                "samples": len(self.history),
                "under_pressure": self.should_throttle(),
            }


_monitor: ResourceMonitor | None = None
_monitor_lock = threading.Lock()


def get_resource_monitor() -> ResourceMonitor:
    """Get global resource monitor instance (thread-safe)."""
    global _monitor
    if _monitor is None:
        with _monitor_lock:
            if _monitor is None:
                _monitor = ResourceMonitor()
    return _monitor
