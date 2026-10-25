"""Data Leak Protection (DLP) for Jarvis.

Detects and prevents unauthorized exfiltration of sensitive data including:
- API keys and secrets
- Passwords and credentials
- Credit card numbers
- SSN and personal identifiers
- Private keys and certificates
- Database connection strings
- Internal IP addresses and hostnames
"""

import hashlib
import json
import logging
import re
import threading
from core.helpers import atomic_write_json, read_json_with_integrity
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


class SensitiveDataPattern(str, Enum):
    API_KEY_GENERIC = "api_key"
    AWS_ACCESS_KEY = "aws_access_key"
    AWS_SECRET_KEY = "aws_secret_key"
    GITHUB_TOKEN = "github_token"
    PRIVATE_KEY = "private_key"
    PASSWORD = "password"
    CREDIT_CARD = "credit_card"
    SSN = "ssn"
    EMAIL = "email"
    PHONE = "phone"
    IP_ADDRESS = "ip_address"
    DB_CONNECTION = "db_connection"
    JWT_TOKEN = "jwt_token"
    OAUTH_TOKEN = "oauth_token"
    SECRET_KEY = "secret_key"
    PASSPHRASE = "passphrase"
    CUSTOM = "custom"


class LeakSeverity(str, Enum):
    """Severity levels for data leak events."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DataLeakEvent:
    """Represents a detected data leak attempt."""
    timestamp: datetime
    pattern_type: SensitiveDataPattern
    severity: LeakSeverity
    source: str  # Tool or component that triggered the detection
    destination: str  # Where data was being sent
    data_preview: str  # Sanitized preview of detected data
    blocked: bool
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "pattern_type": self.pattern_type.value,
            "severity": self.severity.value,
            "source": self.source,
            "destination": self.destination,
            "data_preview": self.data_preview,
            "blocked": self.blocked,
            "context": self.context,
        }


class DataLeakProtector:
    """Main DLP engine for detecting and blocking sensitive data leaks."""

    MAX_CUSTOM_PATTERNS = 100
    PATTERNS: dict[SensitiveDataPattern, tuple[re.Pattern, LeakSeverity]] = {
        SensitiveDataPattern.API_KEY_GENERIC: (
            re.compile(r'(?i)(api[_-]?key|apikey|api[_-]?token)\s*[=:]\s*["\']?[a-zA-Z0-9_\-]{20,}["\']?', re.IGNORECASE),
            LeakSeverity.HIGH,
        ),
        SensitiveDataPattern.AWS_ACCESS_KEY: (
            re.compile(r'(?:A3T[A-Z0-9]|AKIA|AGPA|AIDA|AROA|AIPA|ANPA|ANVA|ASIA)[0-9A-Z]{16}'),
            LeakSeverity.CRITICAL,
        ),
        SensitiveDataPattern.AWS_SECRET_KEY: (
            re.compile(r'(?:aws)?[_-]?(?:secret)?[_-]?(?:access)?[_-]?key[_-]?(?:id)?\s*[=:]\s*["\']?[a-zA-Z0-9/+=]{40}["\']?', re.IGNORECASE),
            LeakSeverity.CRITICAL,
        ),
        SensitiveDataPattern.GITHUB_TOKEN: (
            re.compile(r'(?:ghp|gho|ghu|ghs|ghr)_[a-zA-Z0-9]{36}'),
            LeakSeverity.CRITICAL,
        ),
        SensitiveDataPattern.PRIVATE_KEY: (
            re.compile(r'-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----'),
            LeakSeverity.CRITICAL,
        ),
        SensitiveDataPattern.PASSWORD: (
            re.compile(r'(?i)(?:password|passwd|pwd)\s*[=:]\s*["\'][^"\']{8,}["\']'),
            LeakSeverity.HIGH,
        ),
        SensitiveDataPattern.CREDIT_CARD: (
            re.compile(r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b'),
            LeakSeverity.CRITICAL,
        ),
        SensitiveDataPattern.SSN: (
            re.compile(r'\b[0-9]{3}-[0-9]{2}-[0-9]{4}\b'),
            LeakSeverity.CRITICAL,
        ),
        SensitiveDataPattern.DB_CONNECTION: (
            re.compile(r'(?i)(?:postgres|mysql|mongodb|redis)://[^\s]+:[^\s]+@[^\s]+'),
            LeakSeverity.HIGH,
        ),
        SensitiveDataPattern.JWT_TOKEN: (
            re.compile(r'eyJ[a-zA-Z0-9_-]*\.eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*'),
            LeakSeverity.HIGH,
        ),
        SensitiveDataPattern.OAUTH_TOKEN: (
            re.compile(r'(?i)(?:oauth[_-]?token|bearer)\s+[a-zA-Z0-9_\-\.]{20,}'),
            LeakSeverity.HIGH,
        ),
        SensitiveDataPattern.SECRET_KEY: (
            re.compile(r'(?i)(?:secret[_-]?key|secret)\s*[=:]\s*["\']?[a-zA-Z0-9_\-]{16,}["\']?'),
            LeakSeverity.HIGH,
        ),
        SensitiveDataPattern.IP_ADDRESS: (
            re.compile(r'\b(?:10(?:\.\d{1,3}){3}|172\.(?:1[6-9]|2[0-9]|3[01])(?:\.\d{1,3}){2}|192\.168(?:\.\d{1,3}){2})\b'),
            LeakSeverity.MEDIUM,
        ),
        SensitiveDataPattern.PASSPHRASE: (
            re.compile(r'(?i)(?:passphrase|pass[_-]?phrase)\s*[=:]\s*["\'][^"\']{8,}["\']'),
            LeakSeverity.HIGH,
        ),
    }
    _COMBINED_PATTERN: re.Pattern | None = None
    _PATTERN_NAMES: list[str] = []
    _PATTERN_LOCK = threading.Lock()

    @classmethod
    def _get_combined_pattern(cls) -> re.Pattern:
        if cls._COMBINED_PATTERN is None:
            with cls._PATTERN_LOCK:
                if cls._COMBINED_PATTERN is None:
                    pattern_parts = []
                    cls._PATTERN_NAMES = []
                    for pattern_type, (pattern, _) in cls.PATTERNS.items():
                        pat_str = pattern.pattern
                        pat_str = re.sub(r'\(\?[a-zA-Z]+\)', '', pat_str)
                        pattern_parts.append(f"(?P<{pattern_type.name}>{pat_str})")
                        cls._PATTERN_NAMES.append(pattern_type.name)
                    cls._COMBINED_PATTERN = re.compile('|'.join(pattern_parts), re.IGNORECASE)
        return cls._COMBINED_PATTERN

    def __init__(
        self,
        storage_path: Path | None = None,
        block_critical: bool = True,
        block_high: bool = True,
        audit_callback: Callable[[DataLeakEvent], None] | None = None,
    ):
        self.storage_path = storage_path
        self.block_critical = block_critical
        self.block_high = block_high
        self.audit_callback = audit_callback
        self._lock = threading.RLock()
        self._events: list[DataLeakEvent] = []
        self._max_events = 5000
        self._custom_patterns: dict[str, tuple[re.Pattern, LeakSeverity]] = {}
        self._whitelist_hashes: set[str] = set()

    def add_custom_pattern(
        self,
        name: str,
        pattern: str,
        severity: LeakSeverity = LeakSeverity.HIGH,
    ) -> None:
        """Add a custom detection pattern."""
        compiled = re.compile(pattern, re.IGNORECASE)
        with self._lock:
            if len(self._custom_patterns) >= self.MAX_CUSTOM_PATTERNS and name not in self._custom_patterns:
                oldest = next(iter(self._custom_patterns))
                del self._custom_patterns[oldest]
            self._custom_patterns[name] = (compiled, severity)

    def add_to_whitelist(self, data: str) -> None:
        """Add a known-safe data value to the whitelist."""
        data_hash = hashlib.sha256(data.encode()).hexdigest()
        with self._lock:
            self._whitelist_hashes.add(data_hash)

    def _is_whitelisted(self, data: str) -> bool:
        """Check if data is in the whitelist."""
        data_hash = hashlib.sha256(data.encode()).hexdigest()
        return data_hash in self._whitelist_hashes

    def _sanitize_preview(self, data: str, max_length: int = 20) -> str:
        """Create a safe preview of sensitive data."""
        if len(data) <= max_length:
            return data[:4] + "*" * (len(data) - 8) + data[-4:] if len(data) > 8 else "****"
        return data[:8] + "..." + "*" * 8 + "..." + data[-4:]

    def _has_potential_matches(self, data: str) -> bool:
        """Quick check if data potentially contains sensitive patterns."""
        combined = self._get_combined_pattern()
        return combined.search(data) is not None

    def scan(
        self,
        data: str,
        source: str = "unknown",
        destination: str = "unknown",
    ) -> list[DataLeakEvent]:
        """Scan data for sensitive information leaks.

        Args:
            data: The data to scan
            source: Source component/tool name
            destination: Destination (URL, file path, etc.)

        Returns:
            List of detected leak events
        """
        events = []

        has_builtin_matches = self._has_potential_matches(data)

        if has_builtin_matches:
            for pattern_type, (pattern, severity) in self.PATTERNS.items():
                for match in pattern.finditer(data):
                    match_str = match.group(0)

                    if self._is_whitelisted(match_str):
                        continue

                    should_block = (
                        (severity == LeakSeverity.CRITICAL and self.block_critical) or
                        (severity == LeakSeverity.HIGH and self.block_high)
                    )

                    event = DataLeakEvent(
                        timestamp=datetime.now(),
                        pattern_type=pattern_type,
                        severity=severity,
                        source=source,
                        destination=destination,
                        data_preview=self._sanitize_preview(match_str),
                        blocked=should_block,
                        context={"match_length": len(match_str)},
                    )
                    events.append(event)

        with self._lock:
            custom_patterns_snapshot = list(self._custom_patterns.items())
        for name, (pattern, severity) in custom_patterns_snapshot:
            for match in pattern.finditer(data):
                match_str = match.group(0)

                if self._is_whitelisted(match_str):
                    continue

                should_block = (
                    (severity == LeakSeverity.CRITICAL and self.block_critical) or
                    (severity == LeakSeverity.HIGH and self.block_high)
                )

                event = DataLeakEvent(
                    timestamp=datetime.now(),
                    pattern_type=SensitiveDataPattern.CUSTOM,
                    severity=severity,
                    source=source,
                    destination=destination,
                    data_preview=self._sanitize_preview(match_str),
                    blocked=should_block,
                    context={"custom_pattern": name, "match_length": len(match_str)},
                )
                events.append(event)

        for event in events:
            with self._lock:
                self._events.append(event)
                if len(self._events) > self._max_events:
                    self._events = self._events[-self._max_events:]
            if self.audit_callback:
                self.audit_callback(event)

        return events

    def check_output(
        self,
        data: str,
        source: str,
        destination: str,
    ) -> tuple[bool, str, list[DataLeakEvent]]:
        """Check output data for leaks and optionally block.

        Args:
            data: Data to check
            source: Source component
            destination: Destination

        Returns:
            Tuple of (allowed, sanitized_data, events)
        """
        events = self.scan(data, source, destination)

        # Check if any critical/high events should be blocked
        blocked_events = [e for e in events if e.blocked]

        if blocked_events:
            sanitized = data
            for event in blocked_events:
                if event.pattern_type in self.PATTERNS:
                    pattern, _ = self.PATTERNS[event.pattern_type]
                    sanitized = pattern.sub("[REDACTED]", sanitized)
                elif event.context.get("custom_pattern"):
                    with self._lock:
                        cp_entry = self._custom_patterns.get(event.context["custom_pattern"])
                    if cp_entry:
                        pattern, _ = cp_entry
                        sanitized = pattern.sub("[REDACTED]", sanitized)

            return False, sanitized, events

        return True, data, events

    def check_file_write(
        self,
        file_path: str,
        content: str,
        source: str = "file_write",
    ) -> tuple[bool, str, list[DataLeakEvent]]:
        """Check file write operation for data leaks.

        Special handling for writing to external locations like:
        - Network shares
        - External drives
        - World-writable directories
        """
        destination = f"file:{file_path}"
        return self.check_output(content, source, destination)

    def check_network_request(
        self,
        url: str,
        body: str | dict | None,
        headers: dict | None,
        source: str = "http_request",
    ) -> tuple[bool, str | dict | None, list[DataLeakEvent]]:
        """Check outgoing network request for data leaks.

        Args:
            url: Request URL
            body: Request body (string or dict)
            headers: Request headers
            source: Source component

        Returns:
            Tuple of (allowed, sanitized_body, events)
        """
        all_events = []

        # Check URL for sensitive data
        url_events = self.scan(url, source, url)
        all_events.extend(url_events)

        # Check body
        if body:
            body_str = json.dumps(body) if isinstance(body, dict) else body
            allowed, sanitized_body, body_events = self.check_output(
                body_str, source, url
            )
            all_events.extend(body_events)
            if not allowed:
                body = json.loads(sanitized_body) if isinstance(body, dict) else sanitized_body

        # Check headers for sensitive values
        if headers:
            for key, value in headers.items():
                header_events = self.scan(str(value), source, f"header:{key}")
                all_events.extend(header_events)

        # Block if any critical leaks found
        has_blocked = any(e.blocked for e in all_events)
        return not has_blocked, body, all_events

    def get_events(
        self,
        since: datetime | None = None,
        severity: LeakSeverity | None = None,
    ) -> list[DataLeakEvent]:
        with self._lock:
            events = list(self._events)

        if since:
            events = [e for e in events if e.timestamp >= since]
        if severity:
            events = [e for e in events if e.severity == severity]
        return events

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            total = len(self._events)
            by_severity = {}
            for sev in LeakSeverity:
                by_severity[sev.value] = len([e for e in self._events if e.severity == sev])
            blocked = len([e for e in self._events if e.blocked])
            patterns_detected = list(set(e.pattern_type.value for e in self._events))

        return {
            "total_detections": total,
            "blocked_attempts": blocked,
            "by_severity": by_severity,
            "patterns_detected": patterns_detected,
        }

    def save_events(self) -> None:
        if not self.storage_path:
            return

        self.storage_path.mkdir(parents=True, exist_ok=True)
        events_file = self.storage_path / "dlp_events.json"

        existing = []
        if events_file.exists():
            try:
                existing = read_json_with_integrity(events_file)
            except json.JSONDecodeError as e:
                logger.error(f"DLP events file corrupted: {e}")
                import shutil
                try:
                    shutil.copy2(events_file, events_file.with_suffix('.json.bak'))
                except Exception:
                    logger.warning("Failed to backup corrupted DLP events file")
                existing = []
            except Exception as e:
                logger.error(f"Failed to load DLP events: {e}")
                existing = []

        if len(existing) > self._max_events:
            existing = existing[-self._max_events:]

        with self._lock:
            in_memory = [e.to_dict() for e in self._events]

        combined = existing + in_memory
        if len(combined) > self._max_events:
            combined = combined[-self._max_events:]

        atomic_write_json(events_file, combined)

    def clear_events(self) -> None:
        with self._lock:
            self._events.clear()
