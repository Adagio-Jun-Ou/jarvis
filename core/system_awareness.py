"""System Awareness — Jarvis's perception of the machine it lives on.

Provides real-time visibility into:
- Running processes and services
- Resource utilization (CPU, memory, disk, network)
- File system changes since last check
- Active network connections
- Docker containers status
- System uptime and load
- Recent system log events

This is what makes Jarvis an agent that LIVES on the computer,
not just a chatbot that talks about it.
"""
import logging
import os
import platform
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ResourceSnapshot:
    timestamp: float = field(default_factory=time.time)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_gb: float = 0.0
    memory_total_gb: float = 0.0
    disk_percent: float = 0.0
    disk_free_gb: float = 0.0
    load_1m: float = 0.0
    load_5m: float = 0.0
    load_15m: float = 0.0
    open_files: int = 0
    process_count: int = 0

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "cpu_percent": round(self.cpu_percent, 1),
            "memory_percent": round(self.memory_percent, 1),
            "memory_used_gb": round(self.memory_used_gb, 2),
            "memory_total_gb": round(self.memory_total_gb, 2),
            "disk_percent": round(self.disk_percent, 1),
            "disk_free_gb": round(self.disk_free_gb, 2),
            "load_1m": round(self.load_1m, 2),
            "load_5m": round(self.load_5m, 2),
            "load_15m": round(self.load_15m, 2),
            "open_files": self.open_files,
            "process_count": self.process_count,
        }


@dataclass
class ServiceStatus:
    name: str
    active: bool
    sub_state: str = ""
    enabled: bool = False
    uptime_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "active": self.active,
            "sub_state": self.sub_state,
            "enabled": self.enabled,
            "uptime_seconds": round(self.uptime_seconds, 0),
        }


class SystemAwareness:
    """Real-time system perception for Jarvis agent.

    Like having eyes on the machine — knows what's running,
    what resources are available, and what changed recently.
    """

    SNAPSHOT_INTERVAL = 30.0
    MAX_SNAPSHOTS = 2880  # 24h at 30s intervals

    def __init__(self):
        self._snapshots: deque[ResourceSnapshot] = deque(maxlen=self.MAX_SNAPSHOTS)
        self._last_snapshot_time = 0.0
        self._watched_paths: dict[str, float] = {}
        self._watched_services: list[str] = []
        self._change_log: deque[dict] = deque(maxlen=1000)
        self._lock = threading.RLock()
        self._psutil_available = False
        self._start_time = time.time()

        try:
            import psutil
            self._psutil_available = True
            self._psutil = psutil
        except ImportError:
            self._psutil = None

    def snapshot(self) -> ResourceSnapshot:
        """Take a resource snapshot of the current system state."""
        snap = ResourceSnapshot()

        if self._psutil_available:
            try:
                cpu = self._psutil.cpu_percent(interval=0.1)
                mem = self._psutil.virtual_memory()
                disk = self._psutil.disk_usage("/")
                load = os.getloadavg() if hasattr(os, "getloadavg") else (0, 0, 0)

                snap.cpu_percent = cpu
                snap.memory_percent = mem.percent
                snap.memory_used_gb = mem.used / (1024 ** 3)
                snap.memory_total_gb = mem.total / (1024 ** 3)
                snap.disk_percent = disk.percent
                snap.disk_free_gb = disk.free / (1024 ** 3)
                snap.load_1m = load[0]
                snap.load_5m = load[1]
                snap.load_15m = load[2]
                snap.open_files = len(self._psutil.net_connections()) if hasattr(self._psutil, 'net_connections') else 0
                snap.process_count = len(self._psutil.pids())
            except Exception as e:
                logger.debug(f"psutil snapshot error: {e}")
        else:
            try:
                output = subprocess.check_output(
                    ["free", "-m"], timeout=5, text=True
                ).strip().split("\n")
                if len(output) >= 2:
                    parts = output[1].split()
                    total_mb = int(parts[1])
                    used_mb = int(parts[2])
                    snap.memory_total_gb = total_mb / 1024
                    snap.memory_used_gb = used_mb / 1024
                    snap.memory_percent = (used_mb / total_mb * 100) if total_mb > 0 else 0
            except Exception as e:
                logger.debug(f"Memory stats parse failed: {e}")

            try:
                load = os.getloadavg()
                snap.load_1m = load[0]
                snap.load_5m = load[1]
                snap.load_15m = load[2]
            except (AttributeError, OSError):
                pass

            if hasattr(os, "sysconf"):
                try:
                    snap.cpu_percent = snap.load_1m * 100 / max(os.sysconf("SC_NPROCESSORS_ONLN"), 1)
                except (ValueError, OSError):
                    pass

        with self._lock:
            self._snapshots.append(snap)
            self._last_snapshot_time = time.time()

        return snap

    def get_service_status(self, service_name: str) -> ServiceStatus | None:
        """Check status of a systemd service."""
        try:
            result = subprocess.run(
                ["systemctl", "show", service_name, "--property=ActiveState,SubState,UnitFileState,ActiveEnterTimestamp"],
                capture_output=True, text=True, timeout=10,
            )
            props = {}
            for line in result.stdout.strip().split("\n"):
                if "=" in line:
                    k, v = line.split("=", 1)
                    props[k] = v

            active = props.get("ActiveState", "") == "active"
            sub_state = props.get("SubState", "")
            enabled = props.get("UnitFileState", "") == "enabled"
            uptime = 0.0
            ts_str = props.get("ActiveEnterTimestamp", "")
            if ts_str:
                try:
                    from datetime import datetime as dt
                    ts = dt.strptime(ts_str[:19], "%a %Y-%m-%d %H:%M:%S")
                    uptime = (datetime.now() - ts).total_seconds()
                except Exception as e:
                    logger.debug(f"Service uptime parse failed: {e}")

            return ServiceStatus(
                name=service_name,
                active=active,
                sub_state=sub_state,
                enabled=enabled,
                uptime_seconds=uptime,
            )
        except Exception as e:
            logger.debug(f"Service status check failed for {service_name}: {e}")
            return None

    def get_services_status(self, services: list[str] | None = None) -> list[ServiceStatus]:
        """Check multiple services."""
        if services is None:
            services = self._watched_services
        results = []
        for svc in services:
            status = self.get_service_status(svc)
            if status:
                results.append(status)
        return results

    def watch_service(self, service_name: str) -> None:
        """Add a service to the watch list."""
        if service_name not in self._watched_services:
            self._watched_services.append(service_name)

    def get_running_processes(self, top_n: int = 20, sort_by: str = "cpu") -> list[dict]:
        """Get top running processes."""
        if self._psutil_available:
            try:
                procs = []
                for p in self._psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent", "status", "create_time"]):
                    try:
                        info = p.info
                        procs.append({
                            "pid": info["pid"],
                            "name": info["name"],
                            "cpu": round(info.get("cpu_percent") or 0, 1),
                            "memory": round(info.get("memory_percent") or 0, 1),
                            "status": info.get("status", "?"),
                        })
                    except (self._psutil.NoSuchProcess, self._psutil.AccessDenied):
                        continue

                key = "cpu" if sort_by == "cpu" else "memory"
                procs.sort(key=lambda p: p.get(key, 0), reverse=True)
                return procs[:top_n]
            except Exception as e:
                logger.debug(f"Process listing error: {e}")
        return []

    def get_docker_containers(self) -> list[dict]:
        """Get Docker container status."""
        try:
            result = subprocess.run(
                ["docker", "ps", "-a", "--format", "{{.Names}}\t{{.Status}}\t{{.Ports}}"],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode != 0:
                return []
            containers = []
            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                parts = line.split("\t")
                containers.append({
                    "name": parts[0] if len(parts) > 0 else "?",
                    "status": parts[1] if len(parts) > 1 else "?",
                    "ports": parts[2] if len(parts) > 2 else "",
                })
            return containers
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return []

    def watch_path(self, path: str) -> None:
        """Watch a path for changes (file/directory)."""
        expanded = os.path.expanduser(path)
        if os.path.exists(expanded):
            try:
                mtime = os.path.getmtime(expanded)
                self._watched_paths[expanded] = mtime
            except OSError:
                pass

    def check_changes(self) -> list[dict]:
        """Check watched paths for changes since last check."""
        changes = []
        with self._lock:
            for path, last_mtime in list(self._watched_paths.items()):
                try:
                    if not os.path.exists(path):
                        changes.append({
                            "path": path, "change": "deleted",
                            "timestamp": time.time(),
                        })
                        del self._watched_paths[path]
                        continue

                    current_mtime = os.path.getmtime(path)
                    if current_mtime > last_mtime:
                        self._watched_paths[path] = current_mtime
                        changes.append({
                            "path": path, "change": "modified",
                            "timestamp": current_mtime,
                        })
                except OSError:
                    pass

            for c in changes:
                self._change_log.append(c)

        return changes

    def get_awareness_report(self) -> dict[str, Any]:
        """Get a complete awareness report of the system."""
        snap = self.snapshot()

        hostname = platform.node()
        kernel = platform.release()
        arch = platform.machine()
        uptime_s = time.time() - self._start_time

        report = {
            "hostname": hostname,
            "platform": platform.system(),
            "arch": arch,
            "kernel": kernel,
            "python": platform.python_version(),
            "uptime_agent_seconds": round(uptime_s, 0),
            "resources": snap.to_dict(),
            "watched_changes": len(self._change_log),
            "psutil_available": self._psutil_available,
        }

        if self._watched_services:
            report["services"] = [s.to_dict() for s in self.get_services_status()]

        return report

    def get_resource_trend(self, minutes: int = 30) -> dict[str, Any]:
        """Get resource utilization trend over time."""
        cutoff = time.time() - (minutes * 60)
        with self._lock:
            recent = [s for s in self._snapshots if s.timestamp >= cutoff]

        if not recent:
            return {"error": "No data in time range", "samples": 0}

        avg_cpu = sum(s.cpu_percent for s in recent) / len(recent)
        avg_mem = sum(s.memory_percent for s in recent) / len(recent)
        max_cpu = max(s.cpu_percent for s in recent)
        max_mem = max(s.memory_percent for s in recent)

        return {
            "samples": len(recent),
            "duration_minutes": minutes,
            "cpu_avg": round(avg_cpu, 1),
            "cpu_max": round(max_cpu, 1),
            "memory_avg": round(avg_mem, 1),
            "memory_max": round(max_mem, 1),
            "trend": "rising" if recent[-1].cpu_percent > recent[0].cpu_percent + 10 else
                     "falling" if recent[-1].cpu_percent < recent[0].cpu_percent - 10 else "stable",
        }


_system_awareness: SystemAwareness | None = None
_awareness_lock = threading.Lock()


def get_system_awareness() -> SystemAwareness:
    """Thread-safe singleton for SystemAwareness."""
    global _system_awareness
    if _system_awareness is None:
        with _awareness_lock:
            if _system_awareness is None:
                _system_awareness = SystemAwareness()
    return _system_awareness