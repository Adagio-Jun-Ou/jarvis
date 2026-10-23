"""WebSocket Connection Manager — Keepalive, stale detection, and lifecycle.

Manages WebSocket connection health for 24/7 Jarvis operation:
- Server-side ping/pong with timeout detection
- Stale connection identification and cleanup
- Per-connection metadata tracking (connect time, last activity, client info)
- Connection event logging for observability
- Graceful connection shutdown
"""
import logging
import time
import threading
from dataclasses import dataclass, field
from typing import Any

from fastapi import WebSocket

logger = logging.getLogger(__name__)


@dataclass
class WSConnectionInfo:
    ws: WebSocket
    connected_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    last_pong: float = 0.0
    last_ping_sent: float = 0.0
    client_host: str = "unknown"
    session_id: str | None = None
    message_count: int = 0
    ping_count: int = 0
    pong_count: int = 0
    missed_pongs: int = 0

    @property
    def age_seconds(self) -> float:
        return time.time() - self.connected_at

    @property
    def idle_seconds(self) -> float:
        return time.time() - self.last_activity

    def to_dict(self) -> dict:
        return {
            "client_host": self.client_host,
            "connected_at": self.connected_at,
            "age_seconds": round(self.age_seconds, 1),
            "idle_seconds": round(self.idle_seconds, 1),
            "last_pong_ago": round(time.time() - self.last_pong, 1) if self.last_pong else None,
            "message_count": self.message_count,
            "ping_count": self.ping_count,
            "pong_count": self.pong_count,
            "missed_pongs": self.missed_pongs,
            "session_id": self.session_id,
        }


class WSConnectionManager:
    """Manages WebSocket connection lifecycle with keepalive support.

    Features:
    - Server-side ping with configurable interval
    - Stale connection detection (missed pongs)
    - Max idle time enforcement
    - Max connection lifetime enforcement
    - Connection event logging
    - Thread-safe operations

    Usage:
        mgr = get_ws_connection_manager()
        info = mgr.register(websocket, client_host="10.0.0.1")
        mgr.record_activity(ws_id)
        mgr.record_pong(ws_id)
        stale = mgr.get_stale_connections()
    """

    PING_INTERVAL = 30.0
    MISSED_PONG_LIMIT = 3
    MAX_IDLE_SECONDS = 300.0
    MAX_LIFETIME_SECONDS = 86400.0 * 7
    MAX_CONNECTIONS = 100
    MAX_EVENTS = 1000

    def __init__(self):
        self._lock = threading.RLock()
        self._connections: dict[str, WSConnectionInfo] = {}
        self._events: list[dict] = []
        self._total_connects = 0
        self._total_disconnects = 0
        self._total_timeouts = 0
        self._start_time = time.time()

    def register(self, ws: WebSocket, client_host: str = "unknown", session_id: str | None = None) -> WSConnectionInfo:
        ws_id = str(id(ws))
        info = WSConnectionInfo(
            ws=ws,
            client_host=client_host,
            session_id=session_id,
        )
        with self._lock:
            self._connections[ws_id] = info
            self._total_connects += 1
        self._log_event("connect", ws_id, {"client": client_host})
        return info

    def unregister(self, ws: WebSocket, reason: str = "disconnect") -> None:
        ws_id = str(id(ws))
        with self._lock:
            info = self._connections.pop(ws_id, None)
            if info:
                self._total_disconnects += 1
        if info:
            self._log_event("disconnect", ws_id, {
                "reason": reason,
                "age": round(info.age_seconds, 1),
                "messages": info.message_count,
            })

    def record_activity(self, ws: WebSocket) -> None:
        ws_id = str(id(ws))
        with self._lock:
            info = self._connections.get(ws_id)
            if info:
                info.last_activity = time.time()
                info.message_count += 1

    def record_pong(self, ws: WebSocket) -> None:
        ws_id = str(id(ws))
        with self._lock:
            info = self._connections.get(ws_id)
            if info:
                info.last_pong = time.time()
                info.pong_count += 1
                info.missed_pongs = 0

    def record_ping_sent(self, ws: WebSocket) -> None:
        ws_id = str(id(ws))
        with self._lock:
            info = self._connections.get(ws_id)
            if info:
                info.last_ping_sent = time.time()
                info.ping_count += 1

    def get_stale_connections(self) -> list[tuple[str, WSConnectionInfo]]:
        stale = []
        with self._lock:
            for ws_id, info in self._connections.items():
                is_stale = False
                reason = ""
                if info.missed_pongs >= self.MISSED_PONG_LIMIT:
                    is_stale = True
                    reason = f"missed_pongs={info.missed_pongs}"
                elif info.idle_seconds > self.MAX_IDLE_SECONDS:
                    is_stale = True
                    reason = f"idle={info.idle_seconds:.0f}s"
                elif info.age_seconds > self.MAX_LIFETIME_SECONDS:
                    is_stale = True
                    reason = f"lifetime={info.age_seconds:.0f}s"
                if is_stale:
                    stale.append((ws_id, info))
                    self._total_timeouts += 1
                    self._log_event("stale_detected", ws_id, {"reason": reason})
        return stale

    def increment_missed_pongs(self) -> list[str]:
        with self._lock:
            due = []
            now = time.time()
            for ws_id, info in self._connections.items():
                if info.last_ping_sent > 0 and info.last_pong < info.last_ping_sent:
                    if now - info.last_ping_sent > self.PING_INTERVAL * 2:
                        info.missed_pongs += 1
                        due.append(ws_id)
            return due

    def get_connection_info(self, ws: WebSocket) -> WSConnectionInfo | None:
        with self._lock:
            return self._connections.get(str(id(ws)))

    def get_all_connections(self) -> list[dict]:
        with self._lock:
            return [info.to_dict() for info in self._connections.values()]

    def get_connection_count(self) -> int:
        with self._lock:
            return len(self._connections)

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "active_connections": len(self._connections),
                "total_connects": self._total_connects,
                "total_disconnects": self._total_disconnects,
                "total_timeouts": self._total_timeouts,
                "uptime_seconds": round(time.time() - self._start_time, 1),
                "ping_interval": self.PING_INTERVAL,
                "missed_pong_limit": self.MISSED_PONG_LIMIT,
                "max_idle_seconds": self.MAX_IDLE_SECONDS,
            }

    def get_events(self, limit: int = 50, event_type: str | None = None) -> list[dict]:
        with self._lock:
            events = self._events
            if event_type:
                events = [e for e in events if e.get("type") == event_type]
            return events[-limit:]

    def _log_event(self, event_type: str, ws_id: str, details: dict | None = None) -> None:
        event = {
            "type": event_type,
            "ws_id": ws_id,
            "timestamp": time.time(),
            "details": details or {},
        }
        with self._lock:
            self._events.append(event)
            if len(self._events) > self.MAX_EVENTS:
                self._events = self._events[-self.MAX_EVENTS:]


_ws_manager: WSConnectionManager | None = None
_ws_manager_lock = threading.Lock()


def get_ws_connection_manager() -> WSConnectionManager:
    global _ws_manager
    if _ws_manager is None:
        with _ws_manager_lock:
            if _ws_manager is None:
                _ws_manager = WSConnectionManager()
    return _ws_manager