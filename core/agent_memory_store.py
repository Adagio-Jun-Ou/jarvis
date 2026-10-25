"""Agent Memory Store — persistent key-value store with TTL, namespace isolation, and event bus integration.

Provides:
- AgentMemoryStore: thread-safe KV store with TTL, namespaces, LRU eviction
- Namespaced keys (ns:key) for isolation
- Optional TTL per key (auto-expiry)
- Change events published to Event Bus
- Snapshot/restore for crash recovery
"""
import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class StoreEntry:
    value: Any
    namespace: str
    key: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    expires_at: float | None = None
    access_count: int = 0

    @property
    def is_expired(self) -> bool:
        return self.expires_at is not None and time.time() > self.expires_at

    def touch(self) -> None:
        self.access_count += 1
        self.updated_at = time.time()


class AgentMemoryStore:
    """Thread-safe key-value store with TTL, namespaces, and LRU eviction.

    Keys are namespaced: store.put("ns", "key", value) → stored as "ns:key"
    """

    def __init__(
        self,
        max_keys: int = 10000,
        default_ttl: float | None = None,
        max_key_size: int = 65536,
        max_value_size: int = 1048576,
    ):
        self._max_keys = max_keys
        self._default_ttl = default_ttl
        self._max_key_size = max_key_size
        self._max_value_size = max_value_size
        self._lock = threading.RLock()
        self._store: OrderedDict[str, StoreEntry] = OrderedDict()
        self._stats = {
            "total_puts": 0,
            "total_gets": 0,
            "total_deletes": 0,
            "total_expired": 0,
            "total_evicted": 0,
        }

    def put(
        self,
        namespace: str,
        key: str,
        value: Any,
        ttl: float | None = None,
    ) -> bool:
        if len(key) > self._max_key_size:
            logger.debug(f"Key too long: {len(key)} > {self._max_key_size}")
            return False
        val_str = json.dumps(value) if not isinstance(value, str) else value
        if len(val_str) > self._max_value_size:
            logger.debug(f"Value too large: {len(val_str)} > {self._max_value_size}")
            return False

        full_key = f"{namespace}:{key}"
        effective_ttl = ttl if ttl is not None else self._default_ttl
        expires_at = (time.time() + effective_ttl) if effective_ttl else None

        with self._lock:
            if full_key in self._store:
                self._store.move_to_end(full_key)
                existing = self._store[full_key]
                existing.value = value
                existing.updated_at = time.time()
                existing.expires_at = expires_at
            else:
                while len(self._store) >= self._max_keys:
                    self._store.popitem(last=False)
                    self._stats["total_evicted"] += 1
                self._store[full_key] = StoreEntry(
                    value=value,
                    namespace=namespace,
                    key=key,
                    expires_at=expires_at,
                )
            self._stats["total_puts"] += 1

        self._emit_change("store.put", namespace, key)
        return True

    def get(self, namespace: str, key: str, default: Any = None) -> Any:
        full_key = f"{namespace}:{key}"
        with self._lock:
            entry = self._store.get(full_key)
            if entry is None:
                self._stats["total_gets"] += 1
                return default
            if entry.is_expired:
                del self._store[full_key]
                self._stats["total_expired"] += 1
                self._stats["total_gets"] += 1
                return default
            entry.touch()
            self._store.move_to_end(full_key)
            self._stats["total_gets"] += 1
            return entry.value

    def delete(self, namespace: str, key: str) -> bool:
        full_key = f"{namespace}:{key}"
        with self._lock:
            if full_key in self._store:
                del self._store[full_key]
                self._stats["total_deletes"] += 1
                self._emit_change("store.delete", namespace, key)
                return True
            return False

    def exists(self, namespace: str, key: str) -> bool:
        full_key = f"{namespace}:{key}"
        with self._lock:
            entry = self._store.get(full_key)
            if entry is None:
                return False
            if entry.is_expired:
                del self._store[full_key]
                self._stats["total_expired"] += 1
                return False
            return True

    def list_keys(self, namespace: str | None = None) -> list[str]:
        with self._lock:
            self._expire_sweep_locked()
            if namespace:
                return [e.key for e in self._store.values() if e.namespace == namespace]
            return [e.key for e in self._store.values()]

    def list_namespaces(self) -> list[str]:
        with self._lock:
            self._expire_sweep_locked()
            seen = set()
            for e in self._store.values():
                seen.add(e.namespace)
            return sorted(seen)

    def get_namespace_stats(self, namespace: str) -> dict[str, Any]:
        with self._lock:
            self._expire_sweep_locked()
            entries = [e for e in self._store.values() if e.namespace == namespace]
            return {
                "count": len(entries),
                "total_accesses": sum(e.access_count for e in entries),
            }

    def clear_namespace(self, namespace: str) -> int:
        with self._lock:
            to_remove = [k for k, e in self._store.items() if e.namespace == namespace]
            for k in to_remove:
                del self._store[k]
            self._stats["total_deletes"] += len(to_remove)
            return len(to_remove)

    def cleanup(self) -> int:
        with self._lock:
            return self._expire_sweep_locked()

    def _expire_sweep_locked(self) -> int:
        now = time.time()
        expired = [k for k, e in self._store.items() if e.expires_at and now > e.expires_at]
        for k in expired:
            del self._store[k]
        self._stats["total_expired"] += len(expired)
        return len(expired)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            self._expire_sweep_locked()
            return {
                "entries": {
                    k: {
                        "value": e.value,
                        "namespace": e.namespace,
                        "key": e.key,
                        "created_at": e.created_at,
                        "updated_at": e.updated_at,
                        "expires_at": e.expires_at,
                        "access_count": e.access_count,
                    }
                    for k, e in self._store.items()
                },
                "stats": dict(self._stats),
                "max_keys": self._max_keys,
            }

    def restore(self, data: dict[str, Any]) -> int:
        with self._lock:
            self._store.clear()
            count = 0
            for k, entry_data in data.get("entries", {}).items():
                self._store[k] = StoreEntry(
                    value=entry_data["value"],
                    namespace=entry_data["namespace"],
                    key=entry_data["key"],
                    created_at=entry_data.get("created_at", time.time()),
                    updated_at=entry_data.get("updated_at", time.time()),
                    expires_at=entry_data.get("expires_at"),
                    access_count=entry_data.get("access_count", 0),
                )
                count += 1
            if "stats" in data:
                self._stats.update(data["stats"])
            return count

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total_keys": len(self._store),
                "max_keys": self._max_keys,
                "namespaces": self.list_namespaces(),
                "stats": dict(self._stats),
            }

    def _emit_change(self, event_type: str, namespace: str, key: str) -> None:
        try:
            from core.event_bus import get_event_bus
            bus = get_event_bus()
            bus.publish(event_type, {"namespace": namespace, "key": key})
        except Exception as e:
            logger.debug(f"Memory store event emission failed: {e}")


_store: AgentMemoryStore | None = None
_store_lock = threading.Lock()


def get_agent_memory_store(**kwargs) -> AgentMemoryStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = AgentMemoryStore(**kwargs)
    return _store