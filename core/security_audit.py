"""Security Audit Logger - Structured logging for security events.

Provides a centralized log of all security-relevant actions including:
- Path traversal attempts
- Rate limit breaches  
- Self-edit requests and outcomes
- Prompt injection detections
- Context pressure events
- Authentication events
- API key rotations
"""
import json
import logging
import threading
import time
from collections import OrderedDict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_audit_logger: Optional["SecurityAuditLogger"] = None
_audit_lock = threading.Lock()


class SecurityEvent(Enum):
    PATH_TRAVERSAL_BLOCKED = "path_traversal_blocked"
    RATE_LIMIT_BREACH = "rate_limit_breach"
    SELF_EDIT_REQUEST = "self_edit_request"
    SELF_EDIT_APPROVED = "self_edit_approved"
    SELF_EDIT_REJECTED = "self_edit_rejected"
    PROMPT_INJECTION_DETECTED = "prompt_injection_detected"
    CONTEXT_PRESSURE = "context_pressure"
    CONTEXT_EVICTED = "context_evicted"
    AUTH_SUCCESS = "auth_success"
    AUTH_FAILURE = "auth_failure"
    API_KEY_MASKED = "api_key_masked"
    GUARDIAN_ACTION = "guardian_action"
    CIRCUIT_BREAKER_OPEN = "circuit_breaker_open"
    SANDBOX_FILTER = "sandbox_filter"


class SecurityAuditLogger:
    def __init__(self, log_file: Path | None = None, max_events: int = 1000):
        self._log_file = log_file or Path.home() / ".jarvis" / "security_audit.jsonl"
        self._max_events = max_events
        self._events: OrderedDict[str, dict] = OrderedDict()
        self._lock = threading.RLock()
        self._counts: dict[str, int] = {}
        self._ensure_log_dir()

    def _ensure_log_dir(self) -> None:
        try:
            self._log_file.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"Failed to create log directory: {e}")

    def log_event(self, event: SecurityEvent, details: dict | None = None, severity: str = "info") -> None:
        event_record = {
            "timestamp": datetime.now().isoformat(),
            "event": event.value,
            "severity": severity,
            "details": details or {},
        }

        with self._lock:
            event_id = f"{event.value}_{int(time.time()*1000)}"
            self._events[event_id] = event_record
            while len(self._events) > self._max_events:
                self._events.popitem(last=False)

            self._counts[event.value] = self._counts.get(event.value, 0) + 1

        log_level = logging.WARNING if severity in ("warning", "high", "critical") else logging.INFO
        logger.log(log_level, f"Security: {event.value} - {details}")

        try:
            with open(self._log_file, "a", encoding='utf-8') as f:
                f.write(json.dumps(event_record, default=str) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write security event to log: {e}")

    def get_recent_events(self, limit: int = 50, event_type: str | None = None) -> list[dict]:
        with self._lock:
            events = list(self._events.values())
            if event_type:
                events = [e for e in events if e["event"] == event_type]
            return events[-limit:]

    def get_counts(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counts)

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "total_events": len(self._events),
                "event_counts": dict(self._counts),
                "log_file": str(self._log_file),
            }


def get_security_audit_logger() -> SecurityAuditLogger:
    global _audit_logger, _audit_lock
    if _audit_logger is None:
        with _audit_lock:
            if _audit_logger is None:
                _audit_logger = SecurityAuditLogger()
    return _audit_logger