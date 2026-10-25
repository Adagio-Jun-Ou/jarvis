"""Bidirectional Module Bridge for Context System.

Provides event-driven communication and state sensing between:
- ContextCompressor: Memory and compression
- SubAgentOrchestrator: Multi-agent coordination
- ContextManager: Unified integration layer

This enables modules to be aware of each other's state and react accordingly.

SENSING DIRECTION:
    ContextCompressor ←→ ModuleBridge ←→ SubAgentOrchestrator
                                ↓
                         ContextManager
    
EVENTS:
    - memory_pressure: Emitted when approaching context limits
    - task_delegated: Emitted when task assigned to sub-agent
    - task_completed: Emitted when sub-agent finishes
    - compression_done: Emitted after compression cycle
    - context_restored: Emitted when archived context retrieved
    - agent_available: Emitted when agent pool has capacity
"""
import asyncio
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class BridgeEvent(Enum):
    MEMORY_PRESSURE = "memory_pressure"
    TASK_DELEGATED = "task_delegated"
    TASK_COMPLETED = "task_completed"
    COMPRESSION_DONE = "compression_done"
    CONTEXT_RESTORED = "context_restored"
    AGENT_AVAILABLE = "agent_available"
    SUB_TASK_CREATED = "sub_task_created"
    IMPORTANCE_UPDATED = "importance_updated"
    COMPRESSION_NEEDED = "compression_needed"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    THINKING = "thinking"
    RESPONSE = "response"
    STATE_CHANGE = "state_change"


@dataclass
class BridgeMessage:
    event: BridgeEvent
    source: str
    data: dict[str, Any]
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "event": self.event.value,
            "source": self.source,
            "data": self.data,
            "timestamp": self.timestamp,
        }


class ModuleState:
    ACTIVE = "active"
    IDLE = "idle"
    BUSY = "busy"
    COMPRESSING = "compressing"
    DELEGATING = "delegating"


@dataclass
class ModuleHealth:
    name: str
    state: str = ModuleState.IDLE
    last_heartbeat: float = field(default_factory=time.time)
    metrics: dict[str, Any] = field(default_factory=dict)

    def is_healthy(self, timeout: float = 60.0) -> bool:
        return time.time() - self.last_heartbeat < timeout


class ModuleBridge:
    _instance: Optional["ModuleBridge"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "ModuleBridge":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        # Use regular set instead of WeakSet to keep bound methods alive
        self._subscribers: dict[BridgeEvent, set] = defaultdict(set)
        self._module_health: dict[str, ModuleHealth] = {}
        self._event_history: list[BridgeMessage] = []
        self._max_history = 100
        self._async_queue: asyncio.Queue | None = None
        self._async_loop: asyncio.AbstractEventLoop | None = None

        self._context_compressor_ref: Any = None
        self._sub_agent_orchestrator_ref: Any = None
        self._pending_compressions = 0
        self._pending_tasks = 0
        self._last_memory_pressure = 0.0
        self._compression_requests: list[dict] = []

        self._state_lock = threading.RLock()

    def register_module(self, name: str) -> "ModuleHandle":
        with self._state_lock:
            if name not in self._module_health:
                self._module_health[name] = ModuleHealth(name=name)
            else:
                self._module_health[name].last_heartbeat = time.time()
        logger.debug(f"Module registered: {name}")
        return ModuleHandle(self, name)

    def connect_context_compressor(self, compressor: Any) -> None:
        self._context_compressor_ref = compressor
        logger.info("ContextCompressor connected to bridge")

    def connect_sub_agent_orchestrator(self, orchestrator: Any) -> None:
        self._sub_agent_orchestrator_ref = orchestrator
        logger.info("SubAgentOrchestrator connected to bridge")

    def subscribe(
        self,
        event: BridgeEvent,
        callback: Callable[[BridgeMessage], None],
    ) -> None:
        self._subscribers[event].add(callback)
        logger.debug(f"Subscriber added for {event.value}")

    def unsubscribe(
        self,
        event: BridgeEvent,
        callback: Callable[[BridgeMessage], None],
    ) -> None:
        self._subscribers[event].discard(callback)

    def emit(self, event: BridgeEvent, source: str, data: dict[str, Any]) -> None:
        message = BridgeMessage(event=event, source=source, data=data)

        with self._state_lock:
            self._event_history.append(message)
            if len(self._event_history) > self._max_history:
                self._event_history = self._event_history[-self._max_history:]

            self._handle_event_locally(event, data, source)

            callbacks = list(self._subscribers.get(event, set()))

        for callback in callbacks:
            try:
                callback(message)
            except Exception as e:
                logger.warning(f"Callback error for {event.value}: {e}")

        if self._async_queue:
            try:
                self._async_queue.put_nowait(message)
            except asyncio.QueueFull:
                pass

    def _handle_event_locally(self, event: BridgeEvent, data: dict, source: str) -> None:
        # Called under self._state_lock from emit()
        if event == BridgeEvent.MEMORY_PRESSURE:
            self._last_memory_pressure = time.time()
            self._pending_compressions += 1
            if self._sub_agent_orchestrator_ref:
                self._notify_orchestrator_pressure(data)

        elif event == BridgeEvent.COMPRESSION_DONE:
            self._pending_compressions = max(0, self._pending_compressions - 1)
            if self._sub_agent_orchestrator_ref:
                self._notify_orchestrator_recovered(data)

        elif event == BridgeEvent.TASK_DELEGATED:
            self._pending_tasks += 1

        elif event == BridgeEvent.TASK_COMPLETED:
            self._pending_tasks = max(0, self._pending_tasks - 1)
            if self._context_compressor_ref:
                self._store_task_result(data)

    def _notify_orchestrator_pressure(self, data: dict) -> None:
        try:
            orchestrator = self._sub_agent_orchestrator_ref
            if orchestrator and hasattr(orchestrator, 'on_memory_pressure'):
                orchestrator.on_memory_pressure(
                    pressure_level=data.get('level', 0.8),
                    available_tokens=data.get('available_tokens', 0),
                )
        except Exception as e:
            logger.warning(f"Failed to notify orchestrator of pressure: {e}")

    def _notify_orchestrator_recovered(self, data: dict) -> None:
        try:
            orchestrator = self._sub_agent_orchestrator_ref
            if orchestrator and hasattr(orchestrator, 'on_memory_recovered'):
                orchestrator.on_memory_recovered(
                    freed_tokens=data.get('tokens_freed', 0),
                )
        except Exception as e:
            logger.warning(f"Failed to notify orchestrator of recovery: {e}")

    def _store_task_result(self, data: dict) -> None:
        try:
            compressor = self._context_compressor_ref
            if compressor and hasattr(compressor, 'add_message'):
                task_id = data.get('task_id', 'unknown')
                result = data.get('result', '')
                importance = data.get('importance', 0.7)

                compressor.add_message(
                    role="system",
                    content=f"[Sub-agent task completed] {task_id}: {str(result)[:500]}",
                    importance=importance,
                )
        except Exception as e:
            logger.warning(f"Failed to store task result: {e}")

    def request_compression(self, source: str, reason: str, urgency: float = 0.5) -> bool:
        with self._state_lock:
            request = {
                "source": source,
                "reason": reason,
                "urgency": urgency,
                "timestamp": time.time(),
            }
            self._compression_requests.append(request)
            if len(self._compression_requests) > 10:
                self._compression_requests = self._compression_requests[-10:]

        self.emit(
            BridgeEvent.COMPRESSION_NEEDED,
            source=source,
            data={"reason": reason, "urgency": urgency},
        )

        if self._context_compressor_ref:
            try:
                if hasattr(self._context_compressor_ref, 'compress_if_needed'):
                    return self._context_compressor_ref.compress_if_needed()
            except Exception as e:
                logger.warning(f"Compression request failed: {e}")

        return False

    def get_compression_requests(self) -> list[dict]:
        with self._state_lock:
            return list(self._compression_requests)

    def update_heartbeat(self, module_name: str, state: str = ModuleState.ACTIVE, metrics: dict | None = None) -> None:
        with self._state_lock:
            if module_name in self._module_health:
                health = self._module_health[module_name]
                health.last_heartbeat = time.time()
                health.state = state
                if metrics:
                    health.metrics.update(metrics)

    def get_module_status(self, module_name: str) -> dict | None:
        with self._state_lock:
            if module_name in self._module_health:
                health = self._module_health[module_name]
                return {
                    "name": health.name,
                    "state": health.state,
                    "healthy": health.is_healthy(),
                    "metrics": dict(health.metrics),
                }
        return None

    def get_all_status(self) -> dict[str, dict]:
        with self._state_lock:
            return {
                name: {
                    "name": health.name,
                    "state": health.state,
                    "healthy": health.is_healthy(),
                    "metrics": dict(health.metrics),
                }
                for name, health in self._module_health.items()
            }

    def get_pending_operations(self) -> dict:
        with self._state_lock:
            return {
                "pending_compressions": self._pending_compressions,
                "pending_tasks": self._pending_tasks,
                "compression_requests": len(self._compression_requests),
                "last_memory_pressure": self._last_memory_pressure,
            }

    def get_event_history(self, limit: int = 20) -> list[dict]:
        with self._state_lock:
            return [m.to_dict() for m in self._event_history[-limit:]]

    def is_memory_pressure(self) -> bool:
        with self._state_lock:
            return (
                self._pending_compressions > 0 or
                time.time() - self._last_memory_pressure < 30
            )

    def can_delegate_tasks(self) -> bool:
        with self._state_lock:
            return self._pending_compressions == 0

    def set_async_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._async_loop = loop
        self._async_queue = asyncio.Queue(maxsize=100)

    async def wait_for_event(
        self,
        event: BridgeEvent,
        timeout: float = 30.0,
    ) -> BridgeMessage | None:
        if not self._async_queue:
            return None

        start = time.time()
        while time.time() - start < timeout:
            try:
                msg = await asyncio.wait_for(
                    self._async_queue.get(),
                    timeout=min(1.0, timeout - (time.time() - start)),
                )
                if msg.event == event:
                    return msg
            except asyncio.TimeoutError:
                continue
        return None


class ModuleHandle:
    def __init__(self, bridge: ModuleBridge, name: str):
        self._bridge = bridge
        self._name = name

    def emit(self, event: BridgeEvent, data: dict[str, Any]) -> None:
        self._bridge.emit(event, self._name, data)

    def subscribe(self, event: BridgeEvent, callback: Callable[[BridgeMessage], None]) -> None:
        self._bridge.subscribe(event, callback)

    def unsubscribe(self, event: BridgeEvent, callback: Callable[[BridgeMessage], None]) -> None:
        self._bridge.unsubscribe(event, callback)

    def update_state(self, state: str, metrics: dict | None = None) -> None:
        self._bridge.update_heartbeat(self._name, state, metrics)

    def request_compression(self, reason: str, urgency: float = 0.5) -> bool:
        return self._bridge.request_compression(self._name, reason, urgency)

    def get_status(self) -> dict | None:
        return self._bridge.get_module_status(self._name)


_bridge: ModuleBridge | None = None
_bridge_lock = threading.Lock()


def get_bridge() -> ModuleBridge:
    global _bridge
    if _bridge is None:
        with _bridge_lock:
            if _bridge is None:
                _bridge = ModuleBridge()
    return _bridge


def connect_modules() -> ModuleBridge:
    bridge = get_bridge()

    try:
        from core.context_compressor import get_context_compressor
        compressor = get_context_compressor()
        bridge.connect_context_compressor(compressor)
    except Exception as e:
        logger.debug(f"ContextCompressor not yet available: {e}")

    try:
        from core.sub_agent import get_orchestrator
        orchestrator = get_orchestrator()
        bridge.connect_sub_agent_orchestrator(orchestrator)
    except Exception as e:
        logger.debug(f"SubAgentOrchestrator not yet available: {e}")

    return bridge
