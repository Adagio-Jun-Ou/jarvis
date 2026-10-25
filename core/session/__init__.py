"""Jarvis Session Module.

Provides session management with:
- History log with kind/message/timestamp
- Session persistence across restarts
- Session load/restore capabilities
- Structured transcript management
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from .history_log import (
    HistoryEntry,
    HistoryLog,
    SessionMetadata,
    SessionStore,
    get_session_store,
)


@dataclass
class Message:
    """A message in the conversation."""
    id: str
    role: str
    content: str
    timestamp: datetime
    parent_id: Optional[str] = None
    children: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    reverted: bool = False


__all__ = [
    "HistoryEntry",
    "HistoryLog",
    "Message",
    "SessionMetadata",
    "SessionStore",
    "get_session_store",
]