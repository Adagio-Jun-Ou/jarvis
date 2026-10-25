"""Audit logging for sensitive operations.

Provides:
- Immutable audit trail
- Operation categorization
- User/origin tracking
- Compliance-friendly output
"""
import hashlib
import json
import logging
import threading
from core.helpers import safe_error_message
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class AuditCategory(Enum):
    """Categories of auditable operations."""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    FILE_ACCESS = "file_access"
    FILE_MODIFY = "file_modify"
    FILE_DELETE = "file_delete"
    SHELL_EXECUTE = "shell_execute"
    WEB_REQUEST = "web_request"
    CONFIG_CHANGE = "config_change"
    SYSTEM_MODIFY = "system_modify"
    DATA_ACCESS = "data_access"
    DATA_EXPORT = "data_export"
    PERMISSION_CHANGE = "permission_change"
    TOOL_EXECUTE = "tool_execute"
    API_CALL = "api_call"


class AuditSeverity(Enum):
    """Severity levels for audit events."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """An auditable event."""
    timestamp: datetime
    category: AuditCategory
    severity: AuditSeverity
    operation: str
    user_id: str | None
    client_ip: str | None
    session_id: str | None
    request_id: str | None
    resource: str | None
    details: dict[str, Any] = field(default_factory=dict)
    outcome: str = "success"
    error_message: str | None = None
    duration_ms: float | None = None
    
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "category": self.category.value,
            "severity": self.severity.value,
            "operation": self.operation,
            "user_id": self.user_id,
            "client_ip": self.client_ip,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "resource": self.resource,
            "details": self.details,
            "outcome": self.outcome,
            "error_message": self.error_message,
            "duration_ms": self.duration_ms,
        }
    
    def compute_hash(self, previous_hash: str | None = None) -> str:
        """Compute a hash for chain verification."""
        data = json.dumps(self.to_dict(), sort_keys=True)
        combined = f"{previous_hash or ''}{data}"
        return hashlib.sha256(combined.encode()).hexdigest()


class AuditLogger:
    """Audit logger with file persistence and chain verification.
    
    Features:
    - JSON Lines format for easy parsing
    - Hash chain for tamper detection
    - Automatic rotation
    - Compliance-ready output
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __init__(
        self,
        log_dir: Path | None = None,
        max_file_size_mb: float = 100.0,
        max_files: int = 30,
        enabled: bool = True,
    ):
        self.log_dir = log_dir or Path.home() / ".jarvis" / "audit"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.max_file_size_mb = max_file_size_mb
        self.max_files = max_files
        self.enabled = enabled
        
        self._current_file: Path | None = None
        self._file_lock = threading.RLock()
        self._last_hash: str | None = None
        self._event_count: int = 0
        self._max_queue_size = 1000
        
        self._init_log_file()
    
    @classmethod
    def get_instance(cls) -> "AuditLogger":
        """Get the singleton audit logger."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def _init_log_file(self) -> None:
        """Initialize or rotate log file."""
        today = datetime.now().strftime("%Y-%m-%d")
        self._current_file = self.log_dir / f"audit-{today}.jsonl"
        self._last_hash = self._read_last_hash()
        self._write_queue: deque[dict] = deque(maxlen=self._max_queue_size)
        self._write_lock = threading.Lock()
        self._pending_write = False
    
    def _write_pending_events(self) -> None:
        """Write pending events to file (called from background thread or on threshold)."""
        if not self._write_queue:
            return
        
        with self._write_lock:
            events = list(self._write_queue)
            self._write_queue.clear()
        
        if not events:
            return
        
        with self._file_lock:
            for entry in events:
                try:
                    with open(self._current_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(entry, default=str) + "\n")
                except Exception as e:
                    logger.error(f"Failed to write audit log: {e}")
                    # Re-add to queue on failure
                    with self._write_lock:
                        self._write_queue.extendleft(reversed(events))
                    return
    
    def _flush_if_needed(self) -> None:
        """Flush pending events if queue is full."""
        if len(self._write_queue) >= self._max_queue_size:
            self._write_pending_events()
    
    def _read_last_hash(self) -> str | None:
        """Read the last hash from the current log file."""
        if not self._current_file or not self._current_file.exists():
            return None
        
        try:
            with open(self._current_file, 'r', encoding='utf-8') as f:
                last_line = None
                for line in f:
                    if line.strip():
                        last_line = line
                if last_line:
                    data = json.loads(last_line)
                    return data.get("_chain_hash")
        except Exception as e:
            logger.debug(f"Chain hash read failed: {e}")
        return None
    
    def _check_rotation(self) -> None:
        """Check if log rotation is needed."""
        if not self._current_file or not self._current_file.exists():
            return
        
        size_mb = self._current_file.stat().st_size / (1024 * 1024)
        if size_mb >= self.max_file_size_mb:
            self._rotate()
    
    def _rotate(self) -> None:
        """Rotate the current log file."""
        if not self._current_file:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        rotated = self._current_file.with_suffix(f".{timestamp}.jsonl")
        self._current_file.rename(rotated)
        
        self._init_log_file()
        self._cleanup_old_files()
        logger.info(f"Rotated audit log to {rotated}")
    
    def _cleanup_old_files(self) -> None:
        """Remove old log files beyond the retention limit."""
        log_files = sorted(
            self.log_dir.glob("audit-*.jsonl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        for old_file in log_files[self.max_files:]:
            try:
                old_file.unlink()
                logger.debug(f"Removed old audit log: {old_file}")
            except Exception as e:
                logger.warning(f"Failed to remove old audit log: {e}")
    
    def log(
        self,
        category: AuditCategory,
        operation: str,
        user_id: str | None = None,
        client_ip: str | None = None,
        session_id: str | None = None,
        request_id: str | None = None,
        resource: str | None = None,
        details: dict[str, Any] | None = None,
        severity: AuditSeverity = AuditSeverity.INFO,
        outcome: str = "success",
        error_message: str | None = None,
        duration_ms: float | None = None,
    ) -> None:
        """Log an auditable event.
        
        Args:
            category: Category of operation
            operation: Description of operation
            user_id: User performing operation
            client_ip: Client IP address
            session_id: Session identifier
            request_id: Request identifier
            resource: Resource being accessed/modified
            details: Additional details
            severity: Event severity
            outcome: "success", "failure", or "denied"
            error_message: Error message if failed
            duration_ms: Duration of operation in ms
        """
        if not self.enabled:
            return
        
        event = AuditEvent(
            timestamp=datetime.now(),
            category=category,
            severity=severity,
            operation=operation,
            user_id=user_id,
            client_ip=client_ip,
            session_id=session_id,
            request_id=request_id,
            resource=resource,
            details=details or {},
            outcome=outcome,
            error_message=error_message,
            duration_ms=duration_ms,
        )
        
        chain_hash = event.compute_hash(self._last_hash)
        entry = event.to_dict()
        entry["_chain_hash"] = chain_hash
        entry["_prev_hash"] = self._last_hash
        
        # Update last hash and queue event atomically under lock
        with self._write_lock:
            self._last_hash = chain_hash
            self._write_queue.append(entry)
            self._event_count += 1
        
        # Check rotation (fast, just stat call)
        with self._file_lock:
            self._check_rotation()
        
        # Flush if queue is full
        self._flush_if_needed()
        
        if severity == AuditSeverity.CRITICAL:
            logger.critical(f"AUDIT: {category.value} - {operation} - {outcome}")
        elif severity == AuditSeverity.WARNING:
            logger.warning(f"AUDIT: {category.value} - {operation} - {outcome}")
    
    def log_file_access(
        self,
        operation: str,
        path: str,
        user_id: str | None = None,
        outcome: str = "success",
        error_message: str | None = None,
    ) -> None:
        """Log a file access event."""
        category = AuditCategory.FILE_ACCESS
        if "write" in operation.lower() or "modify" in operation.lower():
            category = AuditCategory.FILE_MODIFY
        elif "delete" in operation.lower():
            category = AuditCategory.FILE_DELETE
        
        self.log(
            category=category,
            operation=operation,
            resource=path,
            user_id=user_id,
            outcome=outcome,
            error_message=error_message,
        )
    
    def log_shell_execute(
        self,
        command: str,
        user_id: str | None = None,
        outcome: str = "success",
        error_message: str | None = None,
    ) -> None:
        """Log a shell command execution."""
        self.log(
            category=AuditCategory.SHELL_EXECUTE,
            operation="shell_execute",
            resource=command[:200],
            user_id=user_id,
            severity=AuditSeverity.WARNING,
            outcome=outcome,
            error_message=error_message,
            details={"command_preview": command[:100]},
        )
    
    def log_tool_execute(
        self,
        tool_name: str,
        args: dict[str, Any],
        user_id: str | None = None,
        outcome: str = "success",
        error_message: str | None = None,
        duration_ms: float | None = None,
    ) -> None:
        """Log a tool execution."""
        safe_args = {k: str(v)[:50] for k, v in args.items() if not k.startswith('_')}
        
        self.log(
            category=AuditCategory.TOOL_EXECUTE,
            operation=f"tool:{tool_name}",
            user_id=user_id,
            details={"args": safe_args},
            outcome=outcome,
            error_message=error_message,
            duration_ms=duration_ms,
        )
    
    def log_permission_change(
        self,
        action: str,
        permission: str,
        granted_to: str,
        granted_by: str | None = None,
    ) -> None:
        """Log a permission change."""
        self.log(
            category=AuditCategory.PERMISSION_CHANGE,
            operation=action,
            user_id=granted_by,
            resource=permission,
            details={"granted_to": granted_to},
            severity=AuditSeverity.WARNING,
        )
    
    def verify_chain(self, file_path: Path | None = None) -> tuple[bool, list[str]]:
        """Verify the hash chain of a log file.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        file_path = file_path or self._current_file
        if not file_path or not file_path.exists():
            return True, []
        
        errors = []
        prev_hash = None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    
                    try:
                        entry = json.loads(line)
                        stored_hash = entry.pop("_chain_hash", None)
                        stored_prev = entry.pop("_prev_hash", None)
                        
                        computed_hash = hashlib.sha256(
                            f"{prev_hash or ''}{json.dumps(entry, sort_keys=True)}".encode()
                        ).hexdigest()
                        
                        if stored_hash and computed_hash != stored_hash:
                            errors.append(f"Line {line_num}: Hash mismatch (tampered entry)")
                        
                        if stored_prev != prev_hash:
                            errors.append(f"Line {line_num}: Chain broken")
                        
                        prev_hash = stored_hash
                    except json.JSONDecodeError:
                        errors.append(f"Line {line_num}: Invalid JSON")
        except Exception as e:
            return False, [safe_error_message(e)]
        
        return len(errors) == 0, errors
    
    def get_stats(self) -> dict[str, Any]:
        """Get audit logger statistics."""
        return {
            "enabled": self.enabled,
            "event_count": self._event_count,
            "current_file": str(self._current_file) if self._current_file else None,
            "log_dir": str(self.log_dir),
            "max_file_size_mb": self.max_file_size_mb,
            "max_files": self.max_files,
        }


def get_audit_logger() -> AuditLogger:
    """Get the global audit logger instance."""
    return AuditLogger.get_instance()


def audit_log(category: AuditCategory, operation: str, **kwargs) -> None:
    """Convenience function for audit logging."""
    get_audit_logger().log(category, operation, **kwargs)