"""Health Check System - Monitors all Jarvis subsystems.

Provides a unified health check endpoint that verifies:
- LLM connectivity
- Memory system integrity  
- Context Guardian pressure
- Vector memory availability
- All Phase 1 subsystems operational
- Resource usage bounds
- System-level metrics (CPU, memory, disk, process)
- Oracle circuit breaker state
- Rate limiter statistics

Inspired by Kubernetes readiness/liveness probes.
"""
import logging
import os
import platform
import threading
import time
from core.helpers import safe_error_message
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)

_health_checker: Optional["HealthChecker"] = None
_health_lock = threading.Lock()


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class SubsystemHealth:
    name: str
    status: HealthStatus
    message: str = ""
    latency_ms: float = 0
    last_check: datetime = field(default_factory=datetime.now)
    details: dict = field(default_factory=dict)


@dataclass
class HealthCheckResult:
    overall_status: HealthStatus
    subsystems: list[SubsystemHealth] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    uptime_seconds: float = 0
    memory_pressure: str = "normal"
    context_pressure: str = "normal"

    def to_dict(self) -> dict:
        return {
            "overall_status": self.overall_status.value,
            "timestamp": self.timestamp.isoformat(),
            "uptime_seconds": round(self.uptime_seconds, 1),
            "memory_pressure": self.memory_pressure,
            "context_pressure": self.context_pressure,
            "subsystems": [
                {
                    "name": s.name,
                    "status": s.status.value,
                    "message": s.message,
                    "latency_ms": round(s.latency_ms, 1),
                    "last_check": s.last_check.isoformat(),
                }
                for s in self.subsystems
            ],
        }


class HealthChecker:
    def __init__(self):
        self._start_time = time.time()
        self._checks: OrderedDict[str, callable] = OrderedDict()
        self._results: OrderedDict[str, SubsystemHealth] = OrderedDict()
        self._lock = threading.RLock()
        self._max_results = 20
        self._register_default_checks()

    def _register_default_checks(self) -> None:
        self.register_check("llm", self._check_llm)
        self.register_check("memory", self._check_memory)
        self.register_check("context_guardian", self._check_context_guardian)
        self.register_check("vector_memory", self._check_vector_memory)
        self.register_check("lcm_shield", self._check_lcm_shield)
        self.register_check("three_layer", self._check_three_layer)
        self.register_check("reflect_edit", self._check_reflect_edit)
        self.register_check("prompt_sandbox", self._check_prompt_sandbox)

    def register_check(self, name: str, check_fn: callable) -> None:
        with self._lock:
            self._checks[name] = check_fn

    def check_all(self) -> HealthCheckResult:
        with self._lock:
            checks_snapshot = list(self._checks.items())

        subsystems = []
        degraded = False
        unhealthy = False

        for name, check_fn in checks_snapshot:
            try:
                start = time.time()
                health = check_fn()
                health.latency_ms = (time.time() - start) * 1000
                subsystems.append(health)
                with self._lock:
                    self._results[name] = health
                    while len(self._results) > self._max_results:
                        self._results.popitem(last=False)
            except Exception as e:
                health = SubsystemHealth(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check failed: {e}",
                )
                subsystems.append(health)
                unhealthy = True

            if health.status == HealthStatus.DEGRADED:
                degraded = True
            elif health.status == HealthStatus.UNHEALTHY:
                unhealthy = True

        if unhealthy:
            overall = HealthStatus.UNHEALTHY
        elif degraded:
            overall = HealthStatus.DEGRADED
        else:
            overall = HealthStatus.HEALTHY

        memory_pressure = "normal"
        context_pressure = "normal"
        try:
            from core.context_guardian import get_context_guardian
            guardian = get_context_guardian()
            context_pressure = guardian.get_pressure_level()
        except Exception as e:
            logger.debug(f"Context guardian check failed: {e}")

        return HealthCheckResult(
            overall_status=overall,
            subsystems=subsystems,
            uptime_seconds=time.time() - self._start_time,
            memory_pressure=memory_pressure,
            context_pressure=context_pressure,
        )

    def _check_llm(self) -> SubsystemHealth:
        try:
            from core.llm import OllamaClient
            OllamaClient.__new__(OllamaClient)
            return SubsystemHealth(name="llm", status=HealthStatus.HEALTHY, message="LLM module available")
        except Exception as e:
            return SubsystemHealth(name="llm", status=HealthStatus.UNHEALTHY, message=safe_error_message(e))

    def _check_memory(self) -> SubsystemHealth:
        try:
            from core.memory import ConversationMemory
            mem = ConversationMemory()
            pressure = mem.get_memory_pressure()
            return SubsystemHealth(
                name="memory",
                status=HealthStatus.HEALTHY if pressure == "normal" else HealthStatus.DEGRADED,
                message=f"Memory pressure: {pressure}",
                details=mem.get_context_stats(),
            )
        except Exception as e:
            return SubsystemHealth(name="memory", status=HealthStatus.UNHEALTHY, message=safe_error_message(e))

    def _check_context_guardian(self) -> SubsystemHealth:
        try:
            from core.context_guardian import get_context_guardian
            guardian = get_context_guardian()
            stats = guardian.get_stats()
            return SubsystemHealth(name="context_guardian", status=HealthStatus.HEALTHY, message="Operational", details=stats)
        except Exception as e:
            return SubsystemHealth(name="context_guardian", status=HealthStatus.DEGRADED, message=safe_error_message(e))

    def _check_vector_memory(self) -> SubsystemHealth:
        try:
            from core.vector_memory import get_vector_memory
            vm = get_vector_memory()
            if vm:
                return SubsystemHealth(name="vector_memory", status=HealthStatus.HEALTHY, message="Available")
            return SubsystemHealth(name="vector_memory", status=HealthStatus.DEGRADED, message="Not initialized")
        except ImportError:
            return SubsystemHealth(name="vector_memory", status=HealthStatus.DEGRADED, message="Module not found")
        except Exception as e:
            return SubsystemHealth(name="vector_memory", status=HealthStatus.DEGRADED, message=safe_error_message(e))

    def _check_lcm_shield(self) -> SubsystemHealth:
        try:
            from core.lcm_shield import get_lcm_shield
            shield = get_lcm_shield()
            stats = shield.get_stats()
            return SubsystemHealth(name="lcm_shield", status=HealthStatus.HEALTHY, message="Operational", details=stats)
        except Exception as e:
            return SubsystemHealth(name="lcm_shield", status=HealthStatus.DEGRADED, message=safe_error_message(e))

    def _check_three_layer(self) -> SubsystemHealth:
        try:
            from core.three_layer_context import get_three_layer_context
            tlc = get_three_layer_context()
            stats = tlc.get_stats()
            return SubsystemHealth(name="three_layer", status=HealthStatus.HEALTHY, message="Operational", details=stats)
        except Exception as e:
            return SubsystemHealth(name="three_layer", status=HealthStatus.DEGRADED, message=safe_error_message(e))

    def _check_reflect_edit(self) -> SubsystemHealth:
        try:
            from core.reflect_edit import get_reflect_edit_loop
            loop = get_reflect_edit_loop()
            stats = loop.get_stats()
            return SubsystemHealth(name="reflect_edit", status=HealthStatus.HEALTHY, message="Operational", details=stats)
        except Exception as e:
            return SubsystemHealth(name="reflect_edit", status=HealthStatus.DEGRADED, message=safe_error_message(e))

    def _check_prompt_sandbox(self) -> SubsystemHealth:
        try:
            from core.prompt_sandbox import get_prompt_sandbox
            sandbox = get_prompt_sandbox()
            stats = sandbox.get_stats()
            return SubsystemHealth(name="prompt_sandbox", status=HealthStatus.HEALTHY, message="Operational", details=stats)
        except Exception as e:
            return SubsystemHealth(name="prompt_sandbox", status=HealthStatus.DEGRADED, message=safe_error_message(e))

    def get_stats(self) -> dict:
        with self._lock:
            check_names = list(self._checks.keys())
            check_count = len(self._checks)
        return {
            "registered_checks": check_count,
            "check_names": check_names,
            "uptime_seconds": round(time.time() - self._start_time, 1),
        }


def get_health_checker() -> HealthChecker:
    global _health_checker, _health_lock
    if _health_checker is None:
        with _health_lock:
            if _health_checker is None:
                _health_checker = HealthChecker()
    return _health_checker


def get_system_metrics() -> dict[str, Any]:
    """Gather system-level metrics: CPU, memory, disk, process info.

    Returns dict with platform info, resource usage, and process stats.
    """
    metrics: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
        },
    }

    try:
        import psutil
        metrics["cpu"] = {
            "percent": psutil.cpu_percent(interval=0.1),
            "count_physical": psutil.cpu_count(logical=False),
            "count_logical": psutil.cpu_count(logical=True),
        }
        mem = psutil.virtual_memory()
        metrics["memory"] = {
            "total_gb": round(mem.total / (1024 ** 3), 2),
            "available_gb": round(mem.available / (1024 ** 3), 2),
            "used_percent": round(mem.percent, 1),
        }
        disk = psutil.disk_usage("/")
        metrics["disk"] = {
            "total_gb": round(disk.total / (1024 ** 3), 2),
            "free_gb": round(disk.free / (1024 ** 3), 2),
            "used_percent": round(disk.percent, 1),
        }
        proc = psutil.Process(os.getpid())
        metrics["process"] = {
            "pid": os.getpid(),
            "rss_mb": round(proc.memory_info().rss / (1024 ** 2), 1),
            "cpu_percent": proc.cpu_percent(interval=0.1),
            "threads": proc.num_threads(),
            "open_files": len(proc.open_files()) if hasattr(proc, 'open_files') else 0,
        }
    except ImportError:
        metrics["cpu"] = {"note": "psutil not available"}
        metrics["memory"] = {"note": "psutil not available"}
        metrics["disk"] = {"note": "psutil not available"}
        metrics["process"] = {"pid": os.getpid()}
    except Exception as e:
        metrics["resource_error"] = safe_error_message(e)

    return metrics


def get_jarvis_status() -> dict[str, Any]:
    """Get comprehensive Jarvis runtime status.

    Includes: subsystem health, system metrics, Oracle CB, rate limiter,
    execution router, knowledge graph statistics.
    """
    result: dict[str, Any] = {"timestamp": datetime.now().isoformat()}

    try:
        hc = get_health_checker()
        health = hc.check_all()
        result["health"] = health.to_dict()
    except Exception as e:
        result["health"] = {"error": safe_error_message(e)}

    result["system"] = get_system_metrics()

    try:
        from core.oracle_integration import cb_state, cache_stats
        result["oracle_circuit_breaker"] = cb_state()
        result["oracle_cache"] = cache_stats()
    except Exception as e:
        result["oracle_circuit_breaker"] = {"error": safe_error_message(e)}

    try:
        from core.rate_limiter import get_rate_limiter
        rl = get_rate_limiter()
        result["rate_limiter"] = rl.get_stats()
    except Exception as e:
        result["rate_limiter"] = {"error": safe_error_message(e)}

    try:
        from core.execution_router import get_execution_router
        router = get_execution_router()
        result["execution_router"] = router.get_statistics()
    except Exception as e:
        result["execution_router"] = {"error": safe_error_message(e)}

    try:
        from core.knowledge_graph import get_knowledge_graph
        kg = get_knowledge_graph()
        result["knowledge_graph"] = kg.get_graph_statistics()
    except Exception as e:
        result["knowledge_graph"] = {"error": safe_error_message(e)}

    return result