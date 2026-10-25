"""Health check system for Jarvis subsystems.

Provides startup validation, health monitoring, and graceful degradation
when subsystems fail to initialize.
"""
import asyncio
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from core.helpers import create_tracked_task, safe_error_message
from typing import Any, Callable

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    STARTING = "starting"
    STOPPED = "stopped"


@dataclass
class SubsystemHealth:
    """Health information for a subsystem."""
    name: str
    status: HealthStatus = HealthStatus.STOPPED
    last_check: datetime | None = None
    last_error: str | None = None
    error_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status.value,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "last_error": self.last_error,
            "error_count": self.error_count,
            "metadata": self.metadata,
        }


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    healthy: bool
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class HealthManager:
    """Manages health checks for all subsystems.
    
    Features:
    - Startup validation with dependency ordering
    - Periodic health monitoring
    - Graceful degradation when subsystems fail
    - Health endpoints for monitoring
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __init__(self, check_interval: float = 60.0):
        self.check_interval = check_interval
        self._lock = threading.RLock()
        self._subsystems: dict[str, SubsystemHealth] = {}
        self._checkers: dict[str, Callable[[], HealthCheckResult]] = {}
        self._dependencies: dict[str, list[str]] = {}
        self._startup_order: list[str] = []
        self._running = False
        self._task: asyncio.Task | None = None
        self._start_time: datetime | None = None
    
    @classmethod
    def get_instance(cls) -> "HealthManager":
        """Get the singleton health manager."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def register(
        self,
        name: str,
        health_check: Callable[[], HealthCheckResult] | None = None,
        dependencies: list[str] | None = None,
    ) -> None:
        """Register a subsystem for health monitoring.
        
        Args:
            name: Subsystem name
            health_check: Optional async health check function
            dependencies: List of subsystems this depends on (must start first)
        """
        with self._lock:
            self._subsystems[name] = SubsystemHealth(name=name, status=HealthStatus.STOPPED)
            if health_check:
                self._checkers[name] = health_check
            if dependencies:
                self._dependencies[name] = dependencies
            self._update_startup_order()
        logger.debug(f"Registered subsystem: {name}")
    
    def _update_startup_order(self) -> None:
        """Calculate startup order based on dependencies."""
        visited = set()
        order = []
        
        def visit(name: str) -> None:
            if name in visited:
                return
            visited.add(name)
            for dep in self._dependencies.get(name, []):
                visit(dep)
            order.append(name)
        
        for name in self._subsystems:
            visit(name)
        
        self._startup_order = order
    
    def mark_starting(self, name: str) -> None:
        """Mark a subsystem as starting."""
        with self._lock:
            if name in self._subsystems:
                self._subsystems[name].status = HealthStatus.STARTING
                self._subsystems[name].last_check = datetime.now()
        logger.info(f"Subsystem starting: {name}")
    
    def mark_healthy(
        self,
        name: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Mark a subsystem as healthy."""
        with self._lock:
            if name in self._subsystems:
                self._subsystems[name].status = HealthStatus.HEALTHY
                self._subsystems[name].last_check = datetime.now()
                self._subsystems[name].last_error = None
                if metadata:
                    self._subsystems[name].metadata.update(metadata)
        logger.info(f"Subsystem healthy: {name}")
    
    def mark_degraded(
        self,
        name: str,
        reason: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Mark a subsystem as degraded (partially working)."""
        with self._lock:
            if name in self._subsystems:
                self._subsystems[name].status = HealthStatus.DEGRADED
                self._subsystems[name].last_check = datetime.now()
                self._subsystems[name].last_error = reason
                self._subsystems[name].error_count += 1
                if metadata:
                    self._subsystems[name].metadata.update(metadata)
        logger.warning(f"Subsystem degraded: {name} - {reason}")
    
    def mark_unhealthy(
        self,
        name: str,
        error: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Mark a subsystem as unhealthy."""
        with self._lock:
            if name in self._subsystems:
                self._subsystems[name].status = HealthStatus.UNHEALTHY
                self._subsystems[name].last_check = datetime.now()
                self._subsystems[name].last_error = error
                self._subsystems[name].error_count += 1
                if metadata:
                    self._subsystems[name].metadata.update(metadata)
        logger.error(f"Subsystem unhealthy: {name} - {error}")
    
    def mark_stopped(self, name: str) -> None:
        """Mark a subsystem as stopped."""
        with self._lock:
            if name in self._subsystems:
                self._subsystems[name].status = HealthStatus.STOPPED
                self._subsystems[name].last_check = datetime.now()
        logger.info(f"Subsystem stopped: {name}")
    
    def get_status(self, name: str) -> HealthStatus:
        """Get the health status of a subsystem."""
        with self._lock:
            if name in self._subsystems:
                return self._subsystems[name].status
        return HealthStatus.UNHEALTHY
    
    def is_healthy(self, name: str) -> bool:
        """Check if a subsystem is healthy or degraded."""
        status = self.get_status(name)
        return status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)
    
    def get_all_health(self) -> dict[str, Any]:
        """Get health status of all subsystems."""
        with self._lock:
            subsystems = {
                name: health.to_dict()
                for name, health in self._subsystems.items()
            }
            startup_order = list(self._startup_order)
            overall = self._get_overall_status()
            start_time = self._start_time

        return {
            "overall": overall.value,
            "uptime_seconds": (datetime.now() - start_time).total_seconds() if start_time else 0,
            "subsystems": subsystems,
            "startup_order": startup_order,
        }
    
    def _get_overall_status(self) -> HealthStatus:
        """Get overall system health status."""
        if not self._subsystems:
            return HealthStatus.STARTING

        statuses = [h.status for h in self._subsystems.values()]

        if any(s == HealthStatus.UNHEALTHY for s in statuses):
            with self._lock:
                unhealthy_critical = any(
                    self._subsystems[n].status == HealthStatus.UNHEALTHY
                    for n in ["agent", "llm", "memory"]
                    if n in self._subsystems
                )
            if unhealthy_critical:
                return HealthStatus.UNHEALTHY

        if any(s == HealthStatus.DEGRADED for s in statuses):
            return HealthStatus.DEGRADED

        if all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY

        if any(s == HealthStatus.STARTING for s in statuses):
            return HealthStatus.STARTING

        return HealthStatus.DEGRADED
    
    async def check_subsystem(self, name: str) -> HealthCheckResult:
        """Run health check for a specific subsystem."""
        with self._lock:
            if name not in self._checkers:
                return HealthCheckResult(healthy=True, message="No checker registered")
            checker = self._checkers[name]
        
        try:
            if asyncio.iscoroutinefunction(checker):
                result = await checker()
            else:
                result = checker()
            
            if result.healthy:
                self.mark_healthy(name, result.metadata)
            else:
                self.mark_degraded(name, result.message, result.metadata)
            
            return result
        except Exception as e:
            self.mark_unhealthy(name, safe_error_message(e))
            return HealthCheckResult(healthy=False, message=safe_error_message(e))
    
    async def check_all(self) -> dict[str, HealthCheckResult]:
        """Run health checks for all subsystems."""
        with self._lock:
            checker_names = list(self._checkers.keys())
        results = {}
        for name in checker_names:
            results[name] = await self.check_subsystem(name)
        return results
    
    async def validate_startup(self) -> dict[str, bool]:
        """Validate that required subsystems are healthy.
        
        Returns dict of subsystem name -> is_healthy
        """
        results = {}

        with self._lock:
            startup_order = list(self._startup_order)
            dependencies = dict(self._dependencies)

        for name in startup_order:
            status = self.get_status(name)
            is_healthy = status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)
            results[name] = is_healthy

            deps = dependencies.get(name, [])
            failed_deps = [d for d in deps if not results.get(d, False)]

            if failed_deps:
                logger.warning(
                    f"Subsystem {name} has failed dependencies: {failed_deps}"
                )

        return results
    
    async def start(self) -> None:
        """Start the health monitoring loop."""
        if self._running:
            return
        
        self._running = True
        self._start_time = datetime.now()
        self._task = create_tracked_task(self._monitor_loop(), name="health_monitor_loop")
        logger.info("Health manager started")
    
    async def stop(self) -> None:
        """Stop the health monitoring loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Health manager stopped")
    
    async def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            try:
                await asyncio.sleep(self.check_interval)
                await self.check_all()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Health check error: {e}")


def get_health_manager() -> HealthManager:
    """Get the global health manager instance."""
    return HealthManager.get_instance()


def register_subsystem(
    name: str,
    health_check: Callable[[], HealthCheckResult] | None = None,
    dependencies: list[str] | None = None,
) -> None:
    """Register a subsystem for health monitoring."""
    get_health_manager().register(name, health_check, dependencies)


def mark_healthy(name: str, metadata: dict[str, Any] | None = None) -> None:
    """Mark a subsystem as healthy."""
    get_health_manager().mark_healthy(name, metadata)


def mark_degraded(name: str, reason: str, metadata: dict[str, Any] | None = None) -> None:
    """Mark a subsystem as degraded."""
    get_health_manager().mark_degraded(name, reason, metadata)


def mark_unhealthy(name: str, error: str, metadata: dict[str, Any] | None = None) -> None:
    """Mark a subsystem as unhealthy."""
    get_health_manager().mark_unhealthy(name, error, metadata)


def mark_stopped(name: str) -> None:
    """Mark a subsystem as stopped."""
    get_health_manager().mark_stopped(name)