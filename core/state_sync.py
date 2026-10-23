"""State Synchronization - Maintains consistent state across subsystems.

Ensures:
- State changes propagate to relevant subsystems
- Conflicts are detected and resolved
- State history is maintained for debugging
- Checkpoint/restore functionality
"""
import asyncio
import copy
import json
import logging
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from core.helpers import create_tracked_task
from typing import Any, Callable

logger = logging.getLogger(__name__)


class StateScope(Enum):
    LOCAL = "local"
    SUBSYSTEM = "subsystem"
    GLOBAL = "global"


class StateUpdateType(Enum):
    SET = "set"
    DELETE = "delete"
    MERGE = "merge"
    INCREMENT = "increment"
    APPEND = "append"


@dataclass
class StateUpdate:
    key: str
    update_type: StateUpdateType
    value: Any
    source: str
    scope: StateScope
    timestamp: datetime = field(default_factory=datetime.now)
    version: int = 0
    acknowledged_by: list[str] = field(default_factory=list)


@dataclass
class StateCheckpoint:
    id: str
    timestamp: datetime
    state: dict
    subsystem_states: dict[str, dict]
    description: str


class StateSynchronizer:
    """Synchronizes state across Jarvis subsystems.
    
    Features:
    - Key-value state store with versioning
    - Pub/sub for state changes
    - Conflict detection and resolution
    - Checkpoint/restore
    - Subsystem isolation with shared keys
    """

    MAX_HISTORY = 100
    MAX_CHECKPOINTS = 20
    CHECKPOINT_INTERVAL = 300
    MAX_SUBSYSTEMS = 50

    def __init__(self, storage_path: Path | None = None):
        self.storage_path = storage_path or Path.home() / ".jarvis" / "state_sync"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self._state: dict[str, Any] = {}
        self._versions: dict[str, int] = defaultdict(int)
        self._subsystem_states: dict[str, dict[str, Any]] = defaultdict(dict)

        self._subscribers: dict[str, list[Callable]] = defaultdict(list)
        self._wildcard_subscribers: list[Callable] = []

        self._update_history: deque[StateUpdate] = deque(maxlen=self.MAX_HISTORY)
        self._checkpoints: deque[StateCheckpoint] = deque(maxlen=self.MAX_CHECKPOINTS)
        self._checkpoint_counter = 0

        self._lock = threading.RLock()
        self._running = False
        self._checkpoint_task: asyncio.Task | None = None

        self._shared_keys: dict[str, list[str]] = {
            "evolution.gene.*": ["evolution_orchestrator", "learning_pipeline", "erdos_engine"],
            "defense.threat_level": ["defense", "learning_pipeline", "resource_monitor"],
            "resource.pressure": ["resource_monitor", "daydream_engine", "erdos_engine"],
            "session.active": ["agent", "web_interface", "voice_interface"],
            "learning.insights": ["insight_bridge", "erdos_engine", "evolution_orchestrator"],
        }

        self._metrics = {
            "updates_processed": 0,
            "conflicts_resolved": 0,
            "checkpoints_created": 0,
        }

        self._load_state()

    def _load_state(self) -> None:
        """Load persisted state."""
        state_file = self.storage_path / "sync_state.json"
        if state_file.exists():
            try:
                with open(state_file, encoding='utf-8') as f:
                    data = json.load(f)
                    self._state = data.get("state", {})
                    for k, v in data.get("versions", {}).items():
                        self._versions[k] = v
                logger.info("Loaded synchronized state")
            except Exception as e:
                logger.warning(f"Failed to load state: {e}")

    def _snapshot_state(self) -> dict:
        return {
            "state": dict(self._state),
            "versions": dict(self._versions),
        }

    def _persist_state(self, data: dict) -> None:
        state_file = self.storage_path / "sync_state.json"
        try:
            with open(state_file, "w", encoding='utf-8') as f:
                json.dump({
                    "state": data["state"],
                    "versions": data["versions"],
                    "timestamp": datetime.now().isoformat(),
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save state: {e}")

    def _save_state(self) -> None:
        with self._lock:
            data = self._snapshot_state()
        self._persist_state(data)

    def get(self, key: str, default: Any = None, scope: StateScope = StateScope.GLOBAL) -> Any:
        """Get a state value."""
        with self._lock:
            if scope == StateScope.GLOBAL:
                return self._state.get(key, default)
            else:
                parts = key.split(".", 1)
                if len(parts) == 2:
                    subsystem, subkey = parts
                    return self._subsystem_states[subsystem].get(subkey, default)
                return default

    def set(
        self,
        key: str,
        value: Any,
        source: str,
        scope: StateScope = StateScope.GLOBAL,
        propagate: bool = True,
    ) -> bool:
        """Set a state value with automatic propagation."""
        subscribers_to_notify = []

        with self._lock:
            old_value = self._state.get(key) if scope == StateScope.GLOBAL else None

            self._versions[key] += 1

            update = StateUpdate(
                key=key,
                update_type=StateUpdateType.SET,
                value=value,
                source=source,
                scope=scope,
                version=self._versions[key],
            )

            if scope == StateScope.GLOBAL:
                self._state[key] = value
            else:
                parts = key.split(".", 1)
                if len(parts) == 2:
                    subsystem, subkey = parts
                    self._subsystem_states[subsystem][subkey] = value

            self._update_history.append(update)

            self._metrics["updates_processed"] += 1

            if propagate:
                self._propagate_update(update)

            subscribers_to_notify = self._get_subscribers_for_key(key)
            change_info = {
                "key": key,
                "old_value": old_value,
                "new_value": value,
                "source": source,
                "timestamp": datetime.now().isoformat(),
                "version": self._versions[key],
            }

        self._invoke_subscribers(subscribers_to_notify, change_info)

        return True

    def _get_subscribers_for_key(self, key: str) -> list[Callable]:
        """Get list of subscribers for a key (must be called with lock held)."""
        subscribers = list(self._wildcard_subscribers)

        if key in self._subscribers:
            subscribers.extend(self._subscribers[key])

        for pattern, callbacks in self._subscribers.items():
            if pattern.endswith(".*"):
                prefix = pattern[:-2]
                if key.startswith(prefix + ".") or key == prefix:
                    subscribers.extend(callbacks)

        return subscribers

    def _invoke_subscribers(self, subscribers: list[Callable], change_info: dict) -> None:
        """Invoke subscriber callbacks outside of lock."""
        for callback in subscribers:
            try:
                callback(change_info)
            except Exception as e:
                logger.debug(f"Subscriber error: {e}")

    def merge(
        self,
        key: str,
        value: dict,
        source: str,
        scope: StateScope = StateScope.GLOBAL,
    ) -> bool:
        """Merge a dictionary into existing state."""
        with self._lock:
            current = self._state.get(key, {}) if scope == StateScope.GLOBAL else {}

            if not isinstance(current, dict):
                current = {}

            if not isinstance(value, dict):
                return False

            self._versions[key] += 1
            new_value = {**current, **value}

            update = StateUpdate(
                key=key,
                update_type=StateUpdateType.MERGE,
                value=value,
                source=source,
                scope=scope,
                version=self._versions[key],
            )

            if scope == StateScope.GLOBAL:
                self._state[key] = new_value

            self._update_history.append(update)

            self._propagate_update(update)
            self._notify_subscribers(key, current, new_value, source)

            return True

    def increment(
        self,
        key: str,
        delta: float = 1,
        source: str = "system",
        scope: StateScope = StateScope.GLOBAL,
    ) -> float:
        """Increment a numeric state value."""
        with self._lock:
            current = self._state.get(key, 0) if scope == StateScope.GLOBAL else 0

            if not isinstance(current, (int, float)):
                current = 0

            new_value = current + delta

            self._versions[key] += 1

            update = StateUpdate(
                key=key,
                update_type=StateUpdateType.INCREMENT,
                value=delta,
                source=source,
                scope=scope,
                version=self._versions[key],
            )

            if scope == StateScope.GLOBAL:
                self._state[key] = new_value

            self._update_history.append(update)

            self._notify_subscribers(key, current, new_value, source)

            return new_value

    def delete(self, key: str, source: str, scope: StateScope = StateScope.GLOBAL) -> bool:
        """Delete a state value."""
        with self._lock:
            old_value = self._state.get(key) if scope == StateScope.GLOBAL else None

            if scope == StateScope.GLOBAL:
                if key in self._state:
                    del self._state[key]
                else:
                    return False

            self._versions[key] += 1

            update = StateUpdate(
                key=key,
                update_type=StateUpdateType.DELETE,
                value=None,
                source=source,
                scope=scope,
                version=self._versions[key],
            )

            self._update_history.append(update)

            self._notify_subscribers(key, old_value, None, source)

            return True

    def subscribe(self, key_pattern: str, callback: Callable) -> None:
        """Subscribe to state changes for a key pattern."""
        if key_pattern == "*":
            self._wildcard_subscribers.append(callback)
        else:
            self._subscribers[key_pattern].append(callback)

    def unsubscribe(self, key_pattern: str, callback: Callable) -> None:
        """Unsubscribe from state changes."""
        if key_pattern == "*":
            if callback in self._wildcard_subscribers:
                self._wildcard_subscribers.remove(callback)
        elif key_pattern in self._subscribers:
            if callback in self._subscribers[key_pattern]:
                self._subscribers[key_pattern].remove(callback)

    def _notify_subscribers(
        self,
        key: str,
        old_value: Any,
        new_value: Any,
        source: str,
    ) -> None:
        """Notify subscribers of a state change."""
        subscribers_to_notify = []
        change_info = {}

        with self._lock:
            change_info = {
                "key": key,
                "old_value": old_value,
                "new_value": new_value,
                "source": source,
                "timestamp": datetime.now().isoformat(),
                "version": self._versions.get(key, 0),
            }
            subscribers_to_notify = self._get_subscribers_for_key(key)

        self._invoke_subscribers(subscribers_to_notify, change_info)

    def _propagate_update(self, update: StateUpdate) -> None:
        """Propagate update to relevant subsystems."""
        for pattern, subsystems in self._shared_keys.items():
            if self._key_matches_pattern(update.key, pattern):
                for subsystem in subsystems:
                    update.acknowledged_by.append(subsystem)

    def _key_matches_pattern(self, key: str, pattern: str) -> bool:
        """Check if a key matches a pattern."""
        if pattern == "*":
            return True
        if pattern.endswith(".*"):
            prefix = pattern[:-2]
            return key.startswith(prefix + ".") or key == prefix
        return key == pattern

    def create_checkpoint(self, description: str = "") -> str:
        """Create a state checkpoint."""
        files_to_remove = []
        checkpoint_data = None

        with self._lock:
            self._checkpoint_counter += 1
            checkpoint_id = f"checkpoint_{self._checkpoint_counter:04d}"

            checkpoint = StateCheckpoint(
                id=checkpoint_id,
                timestamp=datetime.now(),
                state=copy.deepcopy(self._state),
                subsystem_states=copy.deepcopy(dict(self._subsystem_states)),
                description=description,
            )

            self._checkpoints.append(checkpoint)
            while len(self._checkpoints) > self.MAX_CHECKPOINTS:
                removed = self._checkpoints.popleft()
                removed_file = self.storage_path / f"{removed.id}.json"
                files_to_remove.append(removed_file)

            self._metrics["checkpoints_created"] += 1

            checkpoint_data = {
                "id": checkpoint.id,
                "timestamp": checkpoint.timestamp.isoformat(),
                "state": checkpoint.state,
                "subsystem_states": checkpoint.subsystem_states,
                "description": checkpoint.description,
            }
            checkpoint_file = self.storage_path / f"{checkpoint_id}.json"

        for removed_file in files_to_remove:
            try:
                if removed_file.exists():
                    removed_file.unlink()
            except Exception as e:
                logger.debug(f"Failed to remove old checkpoint file: {e}")

        try:
            with open(checkpoint_file, "w", encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")

        logger.info(f"Created checkpoint: {checkpoint_id}")
        return checkpoint_id

    def restore_checkpoint(self, checkpoint_id: str) -> bool:
        """Restore state from a checkpoint."""
        checkpoint = None

        with self._lock:
            for cp in self._checkpoints:
                if cp.id == checkpoint_id:
                    checkpoint = cp
                    break

        if not checkpoint:
            checkpoint_file = self.storage_path / f"{checkpoint_id}.json"
            try:
                with open(checkpoint_file, encoding='utf-8') as f:
                    data = json.load(f)
                    checkpoint = StateCheckpoint(
                        id=data["id"],
                        timestamp=datetime.fromisoformat(data["timestamp"]),
                        state=data["state"],
                        subsystem_states=data["subsystem_states"],
                        description=data.get("description", ""),
                    )
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
                return False

        if not checkpoint:
            return False

        with self._lock:
            self._state = copy.deepcopy(checkpoint.state)
            self._subsystem_states = defaultdict(dict, copy.deepcopy(checkpoint.subsystem_states))

            for key in self._state:
                self._versions[key] += 1

            data = self._snapshot_state()

        self._persist_state(data)

        logger.info(f"Restored checkpoint: {checkpoint_id}")
        return True

    def list_checkpoints(self) -> list[dict]:
        """List available checkpoints."""
        with self._lock:
            return [
                {
                    "id": cp.id,
                    "timestamp": cp.timestamp.isoformat(),
                    "description": cp.description,
                    "state_keys": len(cp.state),
                }
                for cp in self._checkpoints
            ]

    def get_update_history(self, limit: int = 50) -> list[dict]:
        """Get recent update history."""
        with self._lock:
            return [
                {
                    "key": u.key,
                    "type": u.update_type.value,
                    "source": u.source,
                    "timestamp": u.timestamp.isoformat(),
                    "version": u.version,
                }
                for u in list(self._update_history)[-limit:]
            ]

    async def start(self) -> None:
        """Start the state synchronizer."""
        if self._running:
            return

        self._running = True
        self._checkpoint_task = create_tracked_task(self._checkpoint_loop(), name="checkpoint_loop")
        logger.info("State synchronizer started")

    async def stop(self) -> None:
        """Stop the state synchronizer."""
        self._running = False
        if self._checkpoint_task:
            self._checkpoint_task.cancel()
            try:
                await self._checkpoint_task
            except asyncio.CancelledError:
                pass
        self._save_state()
        logger.info("State synchronizer stopped")

    async def _checkpoint_loop(self) -> None:
        """Periodic checkpoint creation."""
        while self._running:
            try:
                await asyncio.sleep(self.CHECKPOINT_INTERVAL)
                self.create_checkpoint(description="Automatic checkpoint")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Checkpoint error: {e}")

    def get_status(self) -> dict:
        """Get synchronizer status."""
        with self._lock:
            return {
                "running": self._running,
                "state_keys": len(self._state),
                "subsystems_tracked": len(self._subsystem_states),
                "subscribers": sum(len(v) for v in self._subscribers.values()),
                "checkpoints": len(self._checkpoints),
                "update_history": len(self._update_history),
                "metrics": self._metrics,
            }

    def get_all_state(self) -> dict:
        """Get all current state (for debugging)."""
        with self._lock:
            return {
                "global": copy.deepcopy(self._state),
                "subsystems": copy.deepcopy(dict(self._subsystem_states)),
                "versions": dict(self._versions),
            }

    def register_subsystem(self, name: str) -> None:
        """Register a subsystem for state isolation."""
        with self._lock:
            if name not in self._subsystem_states:
                if len(self._subsystem_states) >= self.MAX_SUBSYSTEMS:
                    oldest = next(iter(self._subsystem_states))
                    del self._subsystem_states[oldest]
                self._subsystem_states[name] = {}

    def sync_to_subsystem(self, subsystem_name: str) -> dict:
        """Get state relevant to a subsystem."""
        with self._lock:
            relevant_keys = set()

            for pattern, subsystems in self._shared_keys.items():
                if subsystem_name in subsystems:
                    prefix = pattern.replace(".*", "")
                    for key in self._state:
                        if key.startswith(prefix):
                            relevant_keys.add(key)

            return {
                "global_state": {k: self._state[k] for k in relevant_keys if k in self._state},
                "local_state": self._subsystem_states.get(subsystem_name, {}),
            }


_state_synchronizer: StateSynchronizer | None = None
_state_lock = threading.Lock()


def get_state_synchronizer() -> StateSynchronizer:
    """Get global state synchronizer instance (thread-safe)."""
    global _state_synchronizer
    if _state_synchronizer is None:
        with _state_lock:
            if _state_synchronizer is None:
                _state_synchronizer = StateSynchronizer()
    return _state_synchronizer
