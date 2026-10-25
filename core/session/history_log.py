"""SUPERSEDED: Session Persistence and History Log.

`core/session_persistence.py` is the implementation in production — it is used
by integration_harness, startup_validator and the web interface, and is strictly
more capable: atomic writes, HMAC integrity verification, checkpointing at tool
boundaries, crash resume and stale-session detection. This module's SessionStore
is a simpler subset with no importers.

Kept for reference; do not import in new code. The one capability here with no
equivalent in session_persistence is HistoryLog.as_markdown() — structured
transcript rendering, plus get_by_kind()/get_by_time_range() filtering — which
is worth porting if transcript export is ever wanted.

Provides structured session management with:
- History log with kind/message/timestamp
- Session persistence across restarts
- Session load/restore capabilities
- Structured transcript management

Inspired by Claw Code session patterns.
"""
import json
from core.helpers import atomic_write_json, read_json_with_integrity
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class HistoryEntry:
    """A single entry in the history log."""
    kind: str  # 'context', 'command', 'tool', 'error', 'turn', etc.
    message: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "message": self.message,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HistoryEntry":
        return cls(
            kind=data["kind"],
            message=data["message"],
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {}),
        )


class HistoryLog:
    """Structured history log with kind/message/timestamp.
    
    Tracks all events during a session with structured entries.
    Supports querying by kind, time range, and metadata.
    
    Usage:
        log = HistoryLog()
        log.add("tool", "read_file", {"path": "/etc/config"})
        log.add("command", "analyze", {"target": "data.csv"})
        
        # Get all tool events
        tool_events = log.get_by_kind("tool")
        
        # Get as markdown
        print(log.as_markdown())
    """
    
    MAX_ENTRIES = 1000
    
    def __init__(self, max_entries: int = None):
        self._entries: deque[HistoryEntry] = deque(maxlen=max_entries or self.MAX_ENTRIES)
        self._lock = threading.Lock()
        self._entry_counts: dict[str, int] = {}
    
    def add(
        self,
        kind: str,
        message: str,
        metadata: dict[str, Any] = None,
    ) -> HistoryEntry:
        """Add an entry to the history log.
        
        Args:
            kind: Entry type (context, command, tool, error, turn, etc.)
            message: Entry message
            metadata: Optional metadata
            
        Returns:
            The created history entry
        """
        entry = HistoryEntry(
            kind=kind,
            message=message,
            metadata=metadata or {},
        )
        
        with self._lock:
            self._entries.append(entry)
            self._entry_counts[kind] = self._entry_counts.get(kind, 0) + 1
        
        logger.debug(f"History entry: [{kind}] {message[:100]}")
        return entry
    
    def get_entries(self, limit: int = None) -> list[HistoryEntry]:
        """Get all entries, optionally limited."""
        with self._lock:
            if limit is None:
                return list(self._entries)
            return list(self._entries)[-limit:]
    
    def get_by_kind(self, kind: str, limit: int = None) -> list[HistoryEntry]:
        """Get entries by kind."""
        with self._lock:
            matches = [e for e in self._entries if e.kind == kind]
            if limit is None:
                return matches
            return matches[-limit:]
    
    def get_by_time_range(
        self,
        start_time: float,
        end_time: float,
    ) -> list[HistoryEntry]:
        """Get entries within a time range."""
        with self._lock:
            return [
                e for e in self._entries
                if start_time <= e.timestamp <= end_time
            ]
    
    def get_entry_counts(self) -> dict[str, int]:
        """Get counts by entry kind."""
        with self._lock:
            return dict(self._entry_counts)
    
    def get_last_entry(self, kind: str = None) -> Optional[HistoryEntry]:
        """Get the last entry, optionally filtered by kind."""
        with self._lock:
            if not self._entries:
                return None
            
            if kind is None:
                return self._entries[-1]
            
            for entry in reversed(self._entries):
                if entry.kind == kind:
                    return entry
            
            return None
    
    def as_markdown(self, limit: int = None) -> str:
        """Render history as markdown."""
        entries = self.get_entries(limit)
        
        if not entries:
            return "## History\n\nNo entries."
        
        lines = ["## History", ""]
        
        for entry in entries:
            timestamp_str = datetime.fromtimestamp(entry.timestamp).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            
            if entry.metadata:
                metadata_str = f" ({entry.metadata})"
            else:
                metadata_str = ""
            
            lines.append(
                f"- [{timestamp_str}] **{entry.kind}**: {entry.message}{metadata_str}"
            )
        
        return "\n".join(lines)
    
    def to_json(self) -> str:
        """Serialize history to JSON."""
        return json.dumps({
            "entries": [e.to_dict() for e in self._entries],
            "counts": self.get_entry_counts(),
        }, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> "HistoryLog":
        """Deserialize history from JSON."""
        data = json.loads(json_str)
        log = cls()
        
        for entry_data in data.get("entries", []):
            entry = HistoryEntry.from_dict(entry_data)
            log._entries.append(entry)
        
        log._entry_counts = data.get("counts", {})
        return log
    
    def clear(self) -> None:
        """Clear all history entries."""
        with self._lock:
            self._entries.clear()
            self._entry_counts.clear()
        logger.info("History cleared")


@dataclass
class SessionMetadata:
    """Metadata for a persisted session."""
    session_id: str
    created_at: float
    updated_at: float
    turn_count: int
    entry_count: int
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "turn_count": self.turn_count,
            "entry_count": self.entry_count,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionMetadata":
        return cls(
            session_id=data["session_id"],
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            turn_count=data.get("turn_count", 0),
            entry_count=data.get("entry_count", 0),
            metadata=data.get("metadata", {}),
        )


class SessionStore:
    """Session persistence with load/restore capabilities.
    
    Supports:
    - Saving sessions to disk
    - Loading sessions from disk
    - Session history management
    - Session metadata tracking
    
    Usage:
        store = SessionStore(Path.home() / ".jarvis" / "sessions")
        
        # Create new session
        session_id = store.create_session()
        
        # Add to history
        store.add_entry(session_id, "tool", "read_file", {"path": "/etc/config"})
        
        # Save session
        store.save_session(session_id)
        
        # Load session
        store.load_session(session_id)
    """
    
    DEFAULT_SESSION_DIR = Path.home() / ".jarvis" / "sessions"
    
    def __init__(self, session_dir: Path = None):
        self.session_dir = session_dir or self.DEFAULT_SESSION_DIR
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        self._active_sessions: dict[str, HistoryLog] = {}
        self._session_metadata: dict[str, SessionMetadata] = {}
        self._lock = threading.Lock()
    
    def create_session(self, metadata: dict[str, Any] = None) -> str:
        """Create a new session.
        
        Args:
            metadata: Optional session metadata
            
        Returns:
            Session ID
        """
        import hashlib
        session_id = hashlib.sha256(f"{time.time()}".encode()).hexdigest()[:12]
        
        session_metadata = SessionMetadata(
            session_id=session_id,
            created_at=time.time(),
            updated_at=time.time(),
            turn_count=0,
            entry_count=0,
            metadata=metadata or {},
        )
        
        with self._lock:
            self._active_sessions[session_id] = HistoryLog()
            self._session_metadata[session_id] = session_metadata
        
        logger.info(f"Created session: {session_id}")
        return session_id
    
    def add_entry(
        self,
        session_id: str,
        kind: str,
        message: str,
        metadata: dict[str, Any] = None,
    ) -> Optional[HistoryEntry]:
        """Add entry to session history.
        
        Args:
            session_id: Session identifier
            kind: Entry kind
            message: Entry message
            metadata: Optional metadata
            
        Returns:
            Created entry, or None if session not found
        """
        with self._lock:
            if session_id not in self._active_sessions:
                logger.warning(f"Session not found: {session_id}")
                return None
            
            entry = self._active_sessions[session_id].add(kind, message, metadata)
            
            # Update metadata
            if self._session_metadata[session_id].turn_count == 0:
                self._session_metadata[session_id].created_at = entry.timestamp
            self._session_metadata[session_id].updated_at = entry.timestamp
            self._session_metadata[session_id].entry_count += 1
            
            if kind == "turn":
                self._session_metadata[session_id].turn_count += 1
            
            return entry
    
    def get_history(self, session_id: str) -> Optional[HistoryLog]:
        """Get history for a session."""
        with self._lock:
            return self._active_sessions.get(session_id)
    
    def get_metadata(self, session_id: str) -> Optional[SessionMetadata]:
        """Get metadata for a session."""
        with self._lock:
            return self._session_metadata.get(session_id)
    
    def save_session(self, session_id: str) -> Optional[Path]:
        """Save session to disk.
        
        Args:
            session_id: Session to save
            
        Returns:
            Path to saved file, or None if session not found
        """
        with self._lock:
            if session_id not in self._active_sessions:
                logger.warning(f"Session not found: {session_id}")
                return None
            
            history = self._active_sessions[session_id]
            metadata = self._session_metadata[session_id]
        
        # Save to file
        session_file = self.session_dir / f"{session_id}.json"
        
        data = {
            "metadata": metadata.to_dict(),
            "history": history.to_json(),
        }
        
        atomic_write_json(session_file, data)
        
        logger.info(f"Saved session: {session_id}")
        return session_file
    
    def load_session(self, session_id: str) -> bool:
        """Load session from disk.
        
        Args:
            session_id: Session to load
            
        Returns:
            True if successful, False otherwise
        """
        session_file = self.session_dir / f"{session_id}.json"
        
        if not session_file.exists():
            logger.warning(f"Session file not found: {session_id}")
            return False
        
        try:
            data = read_json_with_integrity(session_file)
            
            metadata = SessionMetadata.from_dict(data["metadata"])
            history = HistoryLog.from_json(data["history"])
            
            with self._lock:
                self._active_sessions[session_id] = history
                self._session_metadata[session_id] = metadata
            
            logger.info(f"Loaded session: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return False
    
    def list_sessions(self) -> list[str]:
        """List all saved session IDs."""
        return [
            f.stem for f in self.session_dir.glob("*.json")
        ]
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session.
        
        Args:
            session_id: Session to delete
            
        Returns:
            True if successful, False otherwise
        """
        session_file = self.session_dir / f"{session_id}.json"
        
        with self._lock:
            if session_id in self._active_sessions:
                del self._active_sessions[session_id]
            
            if session_id in self._session_metadata:
                del self._session_metadata[session_id]
        
        if session_file.exists():
            session_file.unlink()
            logger.info(f"Deleted session: {session_id}")
        
        return True
    
    def get_active_session_ids(self) -> list[str]:
        """Get all active session IDs."""
        with self._lock:
            return list(self._active_sessions.keys())
    
    def flush_session(self, session_id: str) -> bool:
        """Flush (save and keep active) a session."""
        path = self.save_session(session_id)
        return path is not None


# Singleton
_session_store: Optional[SessionStore] = None
_store_lock = threading.Lock()


def get_session_store() -> SessionStore:
    """Get the singleton session store."""
    global _session_store
    if _session_store is None:
        with _store_lock:
            if _session_store is None:
                _session_store = SessionStore()
    return _session_store