"""Session Persistence Manager — Crash-resilient session state for 24/7 Jarvis.

Provides:
- Atomic session state saving (write-to-tmp + rename)
- Conversation checkpointing at every tool call boundary
- Session resume after crash (auto-detect stale session)
- Session metadata (start time, turn count, last activity)
- Multi-session management (named sessions)
- Automatic stale session cleanup
- HMAC integrity verification

This ensures Jarvis can survive process crashes, reboots, and OOM kills
without losing conversation context or execution state.
"""
import json
import logging
import os
import threading
import time
import hashlib
import hmac
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

SESSIONS_DIR = Path.home() / ".jarvis" / "sessions"
HMAC_KEY_ENV = "JARVIS_SESSION_HMAC_KEY"
MAX_SESSION_SIZE_BYTES = 50 * 1024 * 1024
MAX_SESSIONS = 100
MAX_TURNS_PER_SESSION = 10000
CHECKPOINT_INTERVAL = 30.0


@dataclass
class SessionTurn:
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)
    tool_calls: list[dict] = field(default_factory=list)
    tool_results: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content[:50000],
            "timestamp": self.timestamp,
            "tool_calls": self.tool_calls[-20:],
            "tool_results": self.tool_results[-20:],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SessionTurn":
        return cls(
            role=data.get("role", "user"),
            content=data.get("content", ""),
            timestamp=data.get("timestamp", time.time()),
            tool_calls=data.get("tool_calls", []),
            tool_results=data.get("tool_results", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class SessionMetadata:
    session_id: str
    title: str = "Untitled Session"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    turn_count: int = 0
    total_tokens: int = 0
    total_tool_calls: int = 0
    is_active: bool = True
    crash_recovery: bool = False
    source: str = "web"

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "turn_count": self.turn_count,
            "total_tokens": self.total_tokens,
            "total_tool_calls": self.total_tool_calls,
            "is_active": self.is_active,
            "crash_recovery": self.crash_recovery,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SessionMetadata":
        return cls(
            session_id=data.get("session_id", ""),
            title=data.get("title", "Untitled Session"),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            turn_count=data.get("turn_count", 0),
            total_tokens=data.get("total_tokens", 0),
            total_tool_calls=data.get("total_tool_calls", 0),
            is_active=data.get("is_active", True),
            crash_recovery=data.get("crash_recovery", False),
            source=data.get("source", "web"),
        )


class SessionPersistence:
    """Manages persistent session state with crash recovery.

    Features:
    - Atomic writes (tmp + rename) prevent corruption
    - HMAC integrity verification detects tampering
    - Auto-detection of stale sessions for crash recovery
    - Checkpointing at tool call boundaries
    - LRU-based session eviction when max count reached
    - Thread-safe operations

    Usage:
        sp = get_session_persistence()
        sp.create_session("my_session")
        sp.add_turn("my_session", "user", "hello")
        sp.add_turn("my_session", "assistant", "hi there")
        sp.save_session("my_session")
        
        # After crash:
        stale = sp.find_stale_sessions()
        for sid in stale:
            sp.resume_session(sid)
    """

    def __init__(self, sessions_dir: Path | None = None):
        self._dir = sessions_dir or SESSIONS_DIR
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._sessions: OrderedDict[str, SessionMetadata] = OrderedDict()
        self._active_writes: set[str] = set()
        self._last_checkpoint: dict[str, float] = {}
        self._hmac_key = os.environ.get(HMAC_KEY_ENV, "jarvis_default_hmac_key_v1")
        self._total_saves = 0
        self._total_loads = 0
        self._total_crash_recoveries = 0
        self._start_time = time.time()
        self._load_session_index()

    def _load_session_index(self) -> None:
        index_path = self._dir / "index.json"
        if index_path.exists():
            try:
                data = json.loads(index_path.read_text(encoding='utf-8'))
                for sid, meta_dict in data.get("sessions", {}).items():
                    self._sessions[sid] = SessionMetadata.from_dict(meta_dict)
            except Exception as e:
                logger.debug(f"Could not load session index: {e}")

    def _save_session_index(self) -> None:
        index_path = self._dir / "index.json"
        try:
            data = {"sessions": {sid: m.to_dict() for sid, m in self._sessions.items()}}
            tmp = index_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2), encoding='utf-8')
            tmp.replace(index_path)
        except Exception as e:
            logger.debug(f"Could not save session index: {e}")

    def _session_path(self, session_id: str) -> Path:
        valid_id = "".join(c for c in session_id if c.isalnum() or c in "_-")
        if not valid_id:
            valid_id = hashlib.sha256(session_id.encode()).hexdigest()[:16]
        return self._dir / f"{valid_id}.json"

    def _compute_hmac(self, data: str) -> str:
        return hmac.new(self._hmac_key.encode(), data.encode(), hashlib.sha256).hexdigest()

    def _atomic_write(self, path: Path, data: dict) -> None:
        content = json.dumps(data, indent=2, ensure_ascii=False)
        size = len(content.encode("utf-8"))
        if size > MAX_SESSION_SIZE_BYTES:
            turns = data.get("turns", [])
            if len(turns) > 100:
                data["turns"] = turns[-100:]
                data["truncated"] = True
            content = json.dumps(data, indent=2, ensure_ascii=False)

        mac = self._compute_hmac(content)
        wrapped = {"_hmac": mac, "_version": 2, "data": data}

        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(wrapped, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
        self._total_saves += 1

    def _atomic_read(self, path: Path) -> dict | None:
        try:
            raw = path.read_text(encoding="utf-8")
            wrapped = json.loads(raw)
            if not isinstance(wrapped, dict):
                return None
            if "_hmac" in wrapped and "_version" in wrapped:
                data_json = json.dumps(wrapped["data"], indent=2, ensure_ascii=False)
                expected_mac = self._compute_hmac(data_json)
                if not hmac.compare_digest(wrapped["_hmac"], expected_mac):
                    logger.warning(f"HMAC mismatch for {path.name}")
                    return None
                return wrapped["data"]
            return wrapped
        except Exception as e:
            logger.debug(f"Could not read session {path}: {e}")
            return None

    def create_session(self, session_id: str | None = None, title: str = "New Session", source: str = "web") -> str:
        if session_id is None:
            session_id = f"session_{int(time.time())}_{os.urandom(4).hex()}"

        with self._lock:
            if session_id in self._sessions:
                return session_id

            if len(self._sessions) >= MAX_SESSIONS:
                oldest = next(iter(self._sessions))
                self.delete_session(oldest)

            meta = SessionMetadata(
                session_id=session_id,
                title=title,
                source=source,
            )
            self._sessions[session_id] = meta
            self._save_session_index()

        return session_id

    def add_turn(self, session_id: str, role: str, content: str, tool_calls: list[dict] | None = None, tool_results: list[dict] | None = None, metadata: dict | None = None) -> None:
        with self._lock:
            meta = self._sessions.get(session_id)
            if not meta:
                return

            meta.turn_count += 1
            meta.updated_at = time.time()
            if tool_calls:
                meta.total_tool_calls += len(tool_calls)
            self._sessions.move_to_end(session_id)

        data = self.load_session_data(session_id) or {"turns": [], "metadata": meta.to_dict()}
        turns = data.get("turns", [])
        if len(turns) >= MAX_TURNS_PER_SESSION:
            turns = turns[-(MAX_TURNS_PER_SESSION - 100):]

        turn = SessionTurn(
            role=role,
            content=content,
            tool_calls=tool_calls or [],
            tool_results=tool_results or [],
            metadata=metadata or {},
        )
        turns.append(turn.to_dict())
        data["turns"] = turns
        data["metadata"] = meta.to_dict()

        now = time.time()
        last = self._last_checkpoint.get(session_id, 0)
        if now - last >= CHECKPOINT_INTERVAL or (tool_calls and len(tool_calls) > 0):
            self.save_session_data(session_id, data)
            self._last_checkpoint[session_id] = now

    def save_session_data(self, session_id: str, data: dict) -> bool:
        path = self._session_path(session_id)
        meta = self._sessions.get(session_id)
        if meta:
            data["metadata"] = meta.to_dict()
        try:
            with self._lock:
                self._active_writes.add(session_id)
            self._atomic_write(path, data)
            return True
        except Exception as e:
            logger.debug(f"Could not save session {session_id}: {e}")
            return False
        finally:
            with self._lock:
                self._active_writes.discard(session_id)

    def load_session_data(self, session_id: str) -> dict | None:
        path = self._session_path(session_id)
        if not path.exists():
            return None
        with self._lock:
            self._total_loads += 1
        return self._atomic_read(path)

    def delete_session(self, session_id: str) -> bool:
        path = self._session_path(session_id)
        with self._lock:
            self._sessions.pop(session_id, None)
            self._last_checkpoint.pop(session_id, None)
            self._save_session_index()
        try:
            if path.exists():
                path.unlink()
            return True
        except Exception as e:
            logger.debug(f"Could not delete session file: {e}")
            return False

    def list_sessions(self) -> list[dict]:
        with self._lock:
            sorted_sessions = sorted(
                self._sessions.values(),
                key=lambda m: m.updated_at,
                reverse=True,
            )
            return [m.to_dict() for m in sorted_sessions]

    def get_session_metadata(self, session_id: str) -> SessionMetadata | None:
        return self._sessions.get(session_id)

    def find_stale_sessions(self, max_idle_seconds: float = 3600) -> list[str]:
        now = time.time()
        stale = []
        with self._lock:
            for sid, meta in self._sessions.items():
                if meta.is_active and (now - meta.updated_at) > max_idle_seconds:
                    stale.append(sid)
        return stale

    def resume_session(self, session_id: str) -> dict | None:
        data = self.load_session_data(session_id)
        if not data:
            return None

        with self._lock:
            meta = self._sessions.get(session_id)
            if meta:
                meta.is_active = True
                meta.crash_recovery = True
                meta.updated_at = time.time()
                self._total_crash_recoveries += 1
                self._save_session_index()
            data["crash_recovery"] = True

        logger.info(f"Session {session_id} resumed after crash")
        return data

    def mark_inactive(self, session_id: str) -> None:
        with self._lock:
            meta = self._sessions.get(session_id)
            if meta:
                meta.is_active = False
                self._save_session_index()

    def cleanup_old_sessions(self, max_age_days: float = 30) -> int:
        cutoff = time.time() - (max_age_days * 86400)
        removed = 0
        with self._lock:
            to_remove = []
            for sid, meta in self._sessions.items():
                if not meta.is_active and meta.updated_at < cutoff:
                    to_remove.append(sid)
        for sid in to_remove:
            if self.delete_session(sid):
                removed += 1
        return removed

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total_sessions": len(self._sessions),
                "active_sessions": sum(1 for m in self._sessions.values() if m.is_active),
                "total_saves": self._total_saves,
                "total_loads": self._total_loads,
                "total_crash_recoveries": self._total_crash_recoveries,
                "uptime_seconds": round(time.time() - self._start_time, 1),
                "sessions_dir": str(self._dir),
                "max_sessions": MAX_SESSIONS,
            }


_session_persistence: SessionPersistence | None = None
_sp_lock = threading.Lock()


def get_session_persistence() -> SessionPersistence:
    global _session_persistence
    if _session_persistence is None:
        with _sp_lock:
            if _session_persistence is None:
                _session_persistence = SessionPersistence()
    return _session_persistence