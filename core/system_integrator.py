"""System Integrator - Orchestrates all Jarvis subsystems.

Provides a unified interface for:
- Cross-subsystem communication
- Health monitoring
- Coordinated shutdown
- Event propagation
- Resource-aware throttling
"""
import asyncio
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from core.helpers import create_tracked_task
from typing import Any, Callable

logger = logging.getLogger(__name__)


class SubsystemState(Enum):
    UNINITIALIZED = "uninitialized"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class EventPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class SystemEvent:
    name: str
    source: str
    data: dict
    priority: EventPriority = EventPriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.now)
    handled: bool = False


@dataclass
class SubsystemRegistry:
    name: str
    instance: Any
    state: SubsystemState = SubsystemState.UNINITIALIZED
    dependencies: list[str] = field(default_factory=list)
    error_count: int = 0
    last_error: str | None = None
    start_time: datetime | None = None
    health_check_interval: float = 60.0
    _last_health_check: datetime | None = None


class SystemIntegrator:
    """Central orchestrator for all Jarvis subsystems.
    
    Responsibilities:
    1. Initialize subsystems in dependency order
    2. Monitor subsystem health
    3. Propagate events between subsystems
    4. Coordinate graceful shutdown
    5. Throttle based on resource pressure
    """

    MAX_EVENT_QUEUE = 100
    HEALTH_CHECK_INTERVAL = 30.0
    MAX_SUBSYSTEM_ERRORS = 5
    MAX_THROTTLE_CALLBACKS = 20

    def __init__(self):
        self._subsystems: dict[str, SubsystemRegistry] = {}
        self._event_handlers: dict[str, list[Callable]] = {}
        self._event_queue: asyncio.Queue = asyncio.Queue(maxsize=self.MAX_EVENT_QUEUE)
        self._running = False
        self._lock = threading.RLock()
        self._integrator_task: asyncio.Task | None = None
        self._health_task: asyncio.Task | None = None

        self._resource_throttle = False
        self._throttle_callbacks: list[Callable] = []

        self._metrics = {
            "events_processed": 0,
            "events_dropped": 0,
            "health_checks": 0,
            "subsystem_restarts": 0,
            "uptime_start": None,
        }

    def register_subsystem(
        self,
        name: str,
        instance: Any,
        dependencies: list[str] | None = None,
        health_check_interval: float = 60.0,
    ) -> None:
        """Register a subsystem with the integrator."""
        with self._lock:
            self._subsystems[name] = SubsystemRegistry(
                name=name,
                instance=instance,
                dependencies=dependencies or [],
                health_check_interval=health_check_interval,
            )
            logger.info(f"Registered subsystem: {name}")

    def on_event(self, event_name: str, handler: Callable) -> None:
        """Register an event handler."""
        with self._lock:
            if event_name not in self._event_handlers:
                self._event_handlers[event_name] = []
            self._event_handlers[event_name].append(handler)

    def unregister_event_handler(self, event_name: str, handler: Callable) -> bool:
        """Unregister an event handler."""
        with self._lock:
            if event_name not in self._event_handlers:
                return False
            handlers = self._event_handlers[event_name]
            if handler in handlers:
                handlers.remove(handler)
                if not handlers:
                    del self._event_handlers[event_name]
                return True
            return False

    def set_broadcast_callback(self, callback: Callable[[dict], None]) -> None:
        """Set a callback to broadcast events to WebSocket clients."""
        self._broadcast_callback = callback

    async def broadcast_event(self, event_type: str, data: dict) -> None:
        """Broadcast an event to all connected WebSocket clients."""
        if hasattr(self, '_broadcast_callback') and self._broadcast_callback:
            try:
                self._broadcast_callback({"type": event_type, **data})
            except Exception as e:
                logger.debug(f"Broadcast error: {e}")

    def emit_event(
        self,
        name: str,
        source: str,
        data: dict,
        priority: EventPriority = EventPriority.NORMAL,
    ) -> bool:
        """Emit an event to be processed by handlers."""
        event = SystemEvent(
            name=name,
            source=source,
            data=data,
            priority=priority,
        )

        # Broadcast permission events to WebSocket
        if name in ["permission_request", "permission_pending"] and hasattr(self, '_broadcast_callback'):
            try:
                self._broadcast_callback({"type": name, **data})
            except Exception as e:
                logger.debug(f"Broadcast error: {e}")

        try:
            self._event_queue.put_nowait(event)
            return True
        except asyncio.QueueFull:
            with self._lock:
                self._metrics["events_dropped"] += 1
            logger.warning(f"Event queue full, dropped: {name}")
            return False

    async def start(self) -> None:
        """Start the integrator and all subsystems."""
        if self._running:
            return

        self._running = True
        self._metrics["uptime_start"] = datetime.now()

        start_order = self._resolve_dependency_order()

        for name in start_order:
            try:
                await self._start_subsystem(name)
            except Exception as e:
                logger.error(f"Failed to start {name}: {e}")
                self._subsystems[name].state = SubsystemState.ERROR
                self._subsystems[name].error_count += 1
                self._subsystems[name].last_error = str(e)

        self._integrator_task = create_tracked_task(self._event_loop(), name="system_integrator_event_loop")
        self._health_task = create_tracked_task(self._health_check_loop(), name="system_integrator_health_check")

        logger.info(f"System integrator started with {len(start_order)} subsystems")

    async def stop(self) -> None:
        """Stop all subsystems gracefully."""
        if not self._running:
            return

        self._running = False

        if self._integrator_task:
            self._integrator_task.cancel()
            try:
                await self._integrator_task
            except asyncio.CancelledError:
                pass

        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass

        stop_order = list(reversed(self._resolve_dependency_order()))

        for name in stop_order:
            try:
                await self._stop_subsystem(name)
            except Exception as e:
                logger.error(f"Error stopping {name}: {e}")

        logger.info("System integrator stopped")

    def _resolve_dependency_order(self) -> list[str]:
        """Resolve subsystem start order based on dependencies."""
        visited = set()
        order = []

        def visit(name: str, path: set[str]):
            if name in path:
                raise ValueError(f"Circular dependency detected: {name}")
            if name in visited:
                return

            if name not in self._subsystems:
                logger.warning(f"Unknown subsystem in dependencies: {name}")
                return

            path.add(name)

            for dep in self._subsystems[name].dependencies:
                visit(dep, path)

            path.remove(name)
            visited.add(name)
            order.append(name)

        for name in self._subsystems:
            visit(name, set())

        return order

    async def _start_subsystem(self, name: str) -> None:
        """Start a single subsystem."""
        registry = self._subsystems[name]
        registry.state = SubsystemState.STARTING

        instance = registry.instance

        if hasattr(instance, 'start'):
            if asyncio.iscoroutinefunction(instance.start):
                await instance.start()
            else:
                instance.start()
        elif hasattr(instance, 'start_pipeline'):
            if asyncio.iscoroutinefunction(instance.start_pipeline):
                await instance.start_pipeline()
            else:
                instance.start_pipeline()
        elif hasattr(instance, 'start_background_learning'):
            if asyncio.iscoroutinefunction(instance.start_background_learning):
                await instance.start_background_learning()
            else:
                instance.start_background_learning()
        elif hasattr(instance, 'start_monitoring'):
            instance.start_monitoring()
        elif hasattr(instance, 'start_evolution'):
            if asyncio.iscoroutinefunction(instance.start_evolution):
                await instance.start_evolution()
            else:
                instance.start_evolution()

        registry.state = SubsystemState.RUNNING
        registry.start_time = datetime.now()

        self.emit_event(
            "subsystem_started",
            "integrator",
            {"subsystem": name},
            EventPriority.HIGH,
        )

        logger.info(f"Started subsystem: {name}")

    async def _stop_subsystem(self, name: str) -> None:
        """Stop a single subsystem."""
        registry = self._subsystems[name]
        registry.state = SubsystemState.STOPPING

        instance = registry.instance

        if hasattr(instance, 'stop'):
            if asyncio.iscoroutinefunction(instance.stop):
                await instance.stop()
            else:
                instance.stop()
        elif hasattr(instance, 'stop_pipeline'):
            if asyncio.iscoroutinefunction(instance.stop_pipeline):
                await instance.stop_pipeline()
            else:
                instance.stop_pipeline()
        elif hasattr(instance, 'stop_background_learning'):
            if asyncio.iscoroutinefunction(instance.stop_background_learning):
                await instance.stop_background_learning()
            else:
                instance.stop_background_learning()
        elif hasattr(instance, 'stop_monitoring'):
            instance.stop_monitoring()
        elif hasattr(instance, 'stop_evolution'):
            if asyncio.iscoroutinefunction(instance.stop_evolution):
                await instance.stop_evolution()
            else:
                instance.stop_evolution()

        registry.state = SubsystemState.STOPPED

        self.emit_event(
            "subsystem_stopped",
            "integrator",
            {"subsystem": name},
            EventPriority.HIGH,
        )

        logger.info(f"Stopped subsystem: {name}")

    async def _event_loop(self) -> None:
        """Process events from the queue."""
        while self._running:
            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=1.0
                )

                await self._process_event(event)
                self._metrics["events_processed"] += 1

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Event processing error: {e}")

    async def _process_event(self, event: SystemEvent) -> None:
        """Process a single event."""
        with self._lock:
            handlers = list(self._event_handlers.get(event.name, []))

        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
                event.handled = True
            except Exception as e:
                logger.error(f"Event handler error for {event.name}: {e}")

        if event.name in ["resource_pressure", "resource_critical"]:
            await self._handle_resource_event(event)

    async def _handle_resource_event(self, event: SystemEvent) -> None:
        """Handle resource pressure events."""
        pressure_level = event.data.get("level", "normal")

        if pressure_level == "critical":
            self._resource_throttle = True
            await self._throttle_subsystems(aggressive=True)

            with self._lock:
                throttle_cbs = list(self._throttle_callbacks)
            for callback in throttle_cbs:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(True, pressure_level)
                    else:
                        callback(True, pressure_level)
                except Exception as e:
                    logger.error(f"Throttle callback error: {e}")

        elif pressure_level == "high":
            self._resource_throttle = True
            await self._throttle_subsystems(aggressive=False)
        else:
            self._resource_throttle = False
            await self._unthrottle_subsystems()

    async def _throttle_subsystems(self, aggressive: bool = False) -> None:
        """Throttle background subsystems."""
        throttle_targets = ["daydream_engine", "erdos_engine", "learning_pipeline"]

        for name in throttle_targets:
            if name in self._subsystems:
                registry = self._subsystems[name]
                if registry.state == SubsystemState.RUNNING:
                    instance = registry.instance
                    if hasattr(instance, 'throttle'):
                        instance.throttle(aggressive=aggressive)
                    elif hasattr(instance, 'pause'):
                        if asyncio.iscoroutinefunction(instance.pause):
                            await instance.pause()
                        else:
                            instance.pause()
                        registry.state = SubsystemState.PAUSED

        logger.info(f"Throttled subsystems (aggressive={aggressive})")

    async def _unthrottle_subsystems(self) -> None:
        """Resume throttled subsystems."""
        throttle_targets = ["daydream_engine", "erdos_engine", "learning_pipeline"]

        for name in throttle_targets:
            if name in self._subsystems:
                registry = self._subsystems[name]
                if registry.state == SubsystemState.PAUSED:
                    instance = registry.instance
                    if hasattr(instance, 'resume'):
                        if asyncio.iscoroutinefunction(instance.resume):
                            await instance.resume()
                        else:
                            instance.resume()
                        registry.state = SubsystemState.RUNNING

        logger.info("Unthrottled subsystems")

    async def _health_check_loop(self) -> None:
        """Periodic health check for all subsystems."""
        while self._running:
            try:
                await asyncio.sleep(self.HEALTH_CHECK_INTERVAL)

                for name, registry in self._subsystems.items():
                    if registry.state != SubsystemState.RUNNING:
                        continue

                    now = datetime.now()
                    if registry._last_health_check:
                        elapsed = (now - registry._last_health_check).total_seconds()
                        if elapsed < registry.health_check_interval:
                            continue

                    registry._last_health_check = now

                    health_ok = await self._check_subsystem_health(name, registry)

                    if not health_ok:
                        registry.error_count += 1

                        if registry.error_count >= self.MAX_SUBSYSTEM_ERRORS:
                            logger.error(f"Subsystem {name} exceeded max errors, marking as error state")
                            registry.state = SubsystemState.ERROR
                            self.emit_event(
                                "subsystem_error",
                                "integrator",
                                {"subsystem": name, "errors": registry.error_count},
                                EventPriority.CRITICAL,
                            )
                    else:
                        registry.error_count = max(0, registry.error_count - 1)

                self._metrics["health_checks"] += 1

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")

    async def _check_subsystem_health(self, name: str, registry: SubsystemRegistry) -> bool:
        """Check health of a single subsystem."""
        instance = registry.instance

        try:
            if hasattr(instance, 'is_healthy'):
                if asyncio.iscoroutinefunction(instance.is_healthy):
                    return await instance.is_healthy()
                else:
                    return instance.is_healthy()

            if hasattr(instance, 'get_status'):
                if asyncio.iscoroutinefunction(instance.get_status):
                    status = await instance.get_status()
                else:
                    status = instance.get_status()
                return status.get("running", True)

            if registry.start_time:
                uptime = (datetime.now() - registry.start_time).total_seconds()
                return uptime > 0

            return True

        except Exception as e:
            registry.last_error = str(e)
            logger.warning(f"Health check failed for {name}: {e}")
            return False

    def register_throttle_callback(self, callback: Callable) -> None:
        with self._lock:
            if len(self._throttle_callbacks) >= self.MAX_THROTTLE_CALLBACKS:
                return
            self._throttle_callbacks.append(callback)

    def unregister_throttle_callback(self, callback: Callable) -> bool:
        """Unregister a throttle callback."""
        with self._lock:
            if callback in self._throttle_callbacks:
                self._throttle_callbacks.remove(callback)
                return True
        return False

    def get_status(self) -> dict:
        """Get comprehensive system status."""
        with self._lock:
            subsystems = {}
            for name, registry in self._subsystems.items():
                subsystems[name] = {
                    "state": registry.state.value,
                    "error_count": registry.error_count,
                    "last_error": registry.last_error,
                    "uptime": (
                        (datetime.now() - registry.start_time).total_seconds()
                        if registry.start_time else 0
                    ),
                }

            uptime = 0
            if self._metrics["uptime_start"]:
                uptime = (datetime.now() - self._metrics["uptime_start"]).total_seconds()

            throttled = self._resource_throttle

        return {
            "running": self._running,
            "throttled": throttled,
            "uptime_seconds": uptime,
            "events_processed": self._metrics["events_processed"],
            "events_dropped": self._metrics["events_dropped"],
            "health_checks": self._metrics["health_checks"],
            "subsystem_restarts": self._metrics["subsystem_restarts"],
            "event_queue_size": self._event_queue.qsize(),
            "subsystems": subsystems,
        }

    def get_subsystem(self, name: str) -> Any | None:
        """Get a subsystem instance by name."""
        with self._lock:
            registry = self._subsystems.get(name)
            return registry.instance if registry else None

    async def restart_subsystem(self, name: str) -> bool:
        """Restart a specific subsystem."""
        if name not in self._subsystems:
            return False

        try:
            await self._stop_subsystem(name)
            await asyncio.sleep(1)
            await self._start_subsystem(name)
            self._metrics["subsystem_restarts"] += 1
            return True
        except Exception as e:
            logger.error(f"Failed to restart {name}: {e}")
            return False


_integrator: SystemIntegrator | None = None
_integrator_lock = threading.Lock()


def get_system_integrator() -> SystemIntegrator:
    """Get global system integrator instance (thread-safe)."""
    global _integrator
    if _integrator is None:
        with _integrator_lock:
            if _integrator is None:
                _integrator = SystemIntegrator()
    return _integrator
