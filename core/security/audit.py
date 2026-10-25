"""Security Audit Logging for Jarvis.

Provides comprehensive security audit logging including:
- Event logging and tracking
- Security metrics and reporting
- Alert management
- Compliance reporting
"""

import json
from core.helpers import atomic_write_json, read_json_with_integrity
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


class AuditLevel(str, Enum):
    """Audit event severity levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AuditCategory(str, Enum):
    """Categories of audit events."""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    DATA_EXFILTRATION = "data_exfiltration"
    NETWORK = "network"
    SYSTEM = "system"
    TOOL_EXECUTION = "tool_execution"
    PERMISSION = "permission"
    SECURITY = "security"
    CONFIGURATION = "configuration"


@dataclass
class AuditEvent:
    """Represents a security audit event."""
    timestamp: datetime
    level: AuditLevel
    category: AuditCategory
    action: str
    description: str
    source: str  # Component that generated the event
    user: str | None = None
    target: str | None = None  # Target of the action
    details: dict[str, Any] = field(default_factory=dict)
    session_id: str | None = None
    ip_address: str | None = None
    success: bool = True
    duration_ms: int | None = None

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "category": self.category.value,
            "action": self.action,
            "description": self.description,
            "source": self.source,
            "user": self.user,
            "target": self.target,
            "details": self.details,
            "session_id": self.session_id,
            "ip_address": self.ip_address,
            "success": self.success,
            "duration_ms": self.duration_ms,
        }


class SecurityAuditor:
    """Comprehensive security audit logging system."""

    def __init__(
        self,
        storage_path: Path | None = None,
        retention_days: int = 30,
        max_events: int = 10000,
        alert_callback: Callable[[AuditEvent], None] | None = None,
    ):
        self.storage_path = storage_path
        self.retention_days = retention_days
        self.max_events = max_events
        self.alert_callback = alert_callback

        self._events: list[AuditEvent] = []
        self._alert_rules: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

        self._stats: dict[str, dict[str, int]] = {
            "by_level": {level.value: 0 for level in AuditLevel},
            "by_category": {cat.value: 0 for cat in AuditCategory},
            "by_source": {},
        }

    def log(
        self,
        level: AuditLevel,
        category: AuditCategory,
        action: str,
        description: str,
        source: str,
        **kwargs,
    ) -> AuditEvent:
        """Log a security audit event.

        Args:
            level: Event severity level
            category: Event category
            action: Action that occurred
            description: Human-readable description
            source: Source component
            **kwargs: Additional event fields

        Returns:
            The created audit event
        """
        event = AuditEvent(
            timestamp=datetime.now(),
            level=level,
            category=category,
            action=action,
            description=description,
            source=source,
            **kwargs,
        )

        with self._lock:
            self._events.append(event)

            self._stats["by_level"][level.value] += 1
            self._stats["by_category"][category.value] += 1
            if source not in self._stats["by_source"]:
                self._stats["by_source"][source] = 0
            self._stats["by_source"][source] += 1

            if len(self._events) > self.max_events:
                self._events = self._events[-self.max_events:]

        self._check_alerts(event)

        return event

    def log_authentication(
        self,
        action: str,
        success: bool,
        user: str | None = None,
        ip_address: str | None = None,
        details: dict | None = None,
    ) -> AuditEvent:
        """Log an authentication event."""
        level = AuditLevel.INFO if success else AuditLevel.WARNING
        return self.log(
            level=level,
            category=AuditCategory.AUTHENTICATION,
            action=action,
            description=f"Authentication {'succeeded' if success else 'failed'} for {user or 'unknown'}",
            source="auth",
            user=user,
            ip_address=ip_address,
            success=success,
            details=details or {},
        )

    def log_authorization(
        self,
        action: str,
        resource: str,
        granted: bool,
        user: str | None = None,
        reason: str | None = None,
    ) -> AuditEvent:
        """Log an authorization event."""
        level = AuditLevel.INFO if granted else AuditLevel.WARNING
        return self.log(
            level=level,
            category=AuditCategory.AUTHORIZATION,
            action=action,
            description=f"Access {'granted' if granted else 'denied'} to {resource}",
            source="authz",
            user=user,
            target=resource,
            success=granted,
            details={"reason": reason} if reason else {},
        )

    def log_data_access(
        self,
        action: str,
        resource: str,
        user: str | None = None,
        details: dict | None = None,
    ) -> AuditEvent:
        """Log a data access event."""
        return self.log(
            level=AuditLevel.INFO,
            category=AuditCategory.DATA_ACCESS,
            action=action,
            description=f"Data accessed: {resource}",
            source="data",
            user=user,
            target=resource,
            details=details or {},
        )

    def log_data_modification(
        self,
        action: str,
        resource: str,
        changes: dict | None = None,
        user: str | None = None,
    ) -> AuditEvent:
        """Log a data modification event."""
        return self.log(
            level=AuditLevel.WARNING,
            category=AuditCategory.DATA_MODIFICATION,
            action=action,
            description=f"Data modified: {resource}",
            source="data",
            user=user,
            target=resource,
            details={"changes": changes} if changes else {},
        )

    def log_security_event(
        self,
        action: str,
        description: str,
        level: AuditLevel = AuditLevel.WARNING,
        source: str = "security",
        details: dict | None = None,
    ) -> AuditEvent:
        """Log a security event."""
        return self.log(
            level=level,
            category=AuditCategory.SECURITY,
            action=action,
            description=description,
            source=source,
            details=details or {},
        )

    def log_tool_execution(
        self,
        tool_name: str,
        action: str,
        success: bool,
        duration_ms: int | None = None,
        user: str | None = None,
        details: dict | None = None,
    ) -> AuditEvent:
        """Log a tool execution event."""
        level = AuditLevel.INFO if success else AuditLevel.ERROR
        return self.log(
            level=level,
            category=AuditCategory.TOOL_EXECUTION,
            action=action,
            description=f"Tool '{tool_name}' executed",
            source=tool_name,
            user=user,
            success=success,
            duration_ms=duration_ms,
            details=details or {},
        )

    def log_permission_change(
        self,
        action: str,
        permission: str,
        granted: bool,
        user: str | None = None,
        target_user: str | None = None,
    ) -> AuditEvent:
        """Log a permission change."""
        return self.log(
            level=AuditLevel.WARNING,
            category=AuditCategory.PERMISSION,
            action=action,
            description=f"Permission '{permission}' {'granted' if granted else 'revoked'}",
            source="permission",
            user=user,
            target=target_user,
            details={"permission": permission, "granted": granted},
        )

    def add_alert_rule(
        self,
        rule_name: str,
        conditions: dict[str, Any],
        callback: Callable[[AuditEvent], None] | None = None,
    ) -> None:
        """Add an alert rule.

        Args:
            rule_name: Name of the rule
            conditions: Conditions that trigger the alert
                - level: AuditLevel or list of levels
                - category: AuditCategory or list of categories
                - action: str or list of actions
                - source: str or list of sources
                - count_threshold: int - trigger after N occurrences
                - time_window_seconds: int - count within this window
            callback: Optional callback for this rule (uses global callback if None)
        """
        self._alert_rules[rule_name] = {
            "conditions": conditions,
            "callback": callback or self.alert_callback,
            "trigger_count": 0,
            "last_triggered": None,
        }

    def _check_alerts(self, event: AuditEvent) -> None:
        """Check if an event triggers any alert rules."""
        with self._lock:
            rules_snapshot = list(self._alert_rules.items())
        for rule_name, rule in rules_snapshot:
            conditions = rule["conditions"]

            # Check level
            if "level" in conditions:
                levels = conditions["level"]
                if isinstance(levels, AuditLevel):
                    levels = [levels]
                if event.level not in levels:
                    continue

            # Check category
            if "category" in conditions:
                categories = conditions["category"]
                if isinstance(categories, AuditCategory):
                    categories = [categories]
                if event.category not in categories:
                    continue

            # Check action
            if "action" in conditions:
                actions = conditions["action"]
                if isinstance(actions, str):
                    actions = [actions]
                if event.action not in actions:
                    continue

            # Check source
            if "source" in conditions:
                sources = conditions["source"]
                if isinstance(sources, str):
                    sources = [sources]
                if event.source not in sources:
                    continue

            # Check count threshold
            if "count_threshold" in conditions:
                rule["trigger_count"] += 1
                threshold = conditions["count_threshold"]

                if "time_window_seconds" in conditions:
                    window = timedelta(seconds=conditions["time_window_seconds"])
                    with self._lock:
                        recent = [
                            e for e in self._events
                            if e.timestamp >= datetime.now() - window
                            and self._matches_conditions(e, conditions)
                        ]
                    if len(recent) < threshold:
                        continue
                else:
                    if rule["trigger_count"] < threshold:
                        continue

            # Trigger alert
            if rule["callback"]:
                try:
                    rule["callback"](event)
                except Exception as e:
                    logger.debug(f"Alert callback error (rule '{rule_name}'): {e}")

            rule["last_triggered"] = datetime.now()

    def _matches_conditions(self, event: AuditEvent, conditions: dict) -> bool:
        """Check if an event matches conditions."""
        if "level" in conditions:
            levels = conditions["level"]
            if isinstance(levels, AuditLevel):
                levels = [levels]
            if event.level not in levels:
                return False

        if "category" in conditions:
            categories = conditions["category"]
            if isinstance(categories, AuditCategory):
                categories = [categories]
            if event.category not in categories:
                return False

        return True

    def get_events(
        self,
        level: AuditLevel | None = None,
        category: AuditCategory | None = None,
        source: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        """Get events with optional filters."""
        with self._lock:
            events = list(self._events)

        if level:
            events = [e for e in events if e.level == level]
        if category:
            events = [e for e in events if e.category == category]
        if source:
            events = [e for e in events if e.source == source]
        if since:
            events = [e for e in events if e.timestamp >= since]
        if until:
            events = [e for e in events if e.timestamp <= until]

        return events[-limit:]

    def get_stats(
        self,
        since: datetime | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            events = list(self._events)
        if since:
            events = [e for e in events if e.timestamp >= since]

        by_level = {level.value: 0 for level in AuditLevel}
        by_category = {cat.value: 0 for cat in AuditCategory}
        by_source: dict[str, int] = {}

        for event in events:
            by_level[event.level.value] += 1
            by_category[event.category.value] += 1
            by_source[event.source] = by_source.get(event.source, 0) + 1

        failed_events = len([e for e in events if not e.success])

        return {
            "total_events": len(events),
            "failed_events": failed_events,
            "by_level": by_level,
            "by_category": by_category,
            "by_source": by_source,
            "alert_rules": len(self._alert_rules),
        }

    def generate_report(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> dict[str, Any]:
        with self._lock:
            events = list(self._events)
        events = [
            e for e in events
            if start_time <= e.timestamp <= end_time
        ]

        # Summary
        summary = {
            "period_start": start_time.isoformat(),
            "period_end": end_time.isoformat(),
            "total_events": len(events),
            "failed_events": len([e for e in events if not e.success]),
            "critical_events": len([e for e in events if e.level == AuditLevel.CRITICAL]),
            "warning_events": len([e for e in events if e.level == AuditLevel.WARNING]),
        }

        # Authentication events
        auth_events = [e for e in events if e.category == AuditCategory.AUTHENTICATION]
        auth_summary = {
            "total_attempts": len(auth_events),
            "successful": len([e for e in auth_events if e.success]),
            "failed": len([e for e in auth_events if not e.success]),
        }

        # Authorization events
        authz_events = [e for e in events if e.category == AuditCategory.AUTHORIZATION]
        authz_summary = {
            "total_requests": len(authz_events),
            "granted": len([e for e in authz_events if e.success]),
            "denied": len([e for e in authz_events if not e.success]),
        }

        # Data events
        data_events = [e for e in events if e.category in (
            AuditCategory.DATA_ACCESS,
            AuditCategory.DATA_MODIFICATION,
            AuditCategory.DATA_EXFILTRATION,
        )]
        data_summary = {
            "total_access": len([e for e in data_events if e.category == AuditCategory.DATA_ACCESS]),
            "total_modifications": len([e for e in data_events if e.category == AuditCategory.DATA_MODIFICATION]),
            "exfiltration_attempts": len([e for e in data_events if e.category == AuditCategory.DATA_EXFILTRATION]),
        }

        # Security events
        security_events = [e for e in events if e.category == AuditCategory.SECURITY]
        security_summary = {
            "total_events": len(security_events),
            "by_action": {},
        }
        for event in security_events:
            security_summary["by_action"][event.action] = \
                security_summary["by_action"].get(event.action, 0) + 1

        return {
            "summary": summary,
            "authentication": auth_summary,
            "authorization": authz_summary,
            "data": data_summary,
            "security": security_summary,
            "events": [e.to_dict() for e in events[-100:]],  # Last 100 events
        }

    def save_events(self) -> None:
        """Save events to storage."""
        if not self.storage_path:
            return

        self.storage_path.mkdir(parents=True, exist_ok=True)
        events_file = self.storage_path / "audit_events.json"

        with self._lock:
            existing = []
            if events_file.exists():
                try:
                    existing = read_json_with_integrity(events_file)
                except json.JSONDecodeError as e:
                    logger.error(f"Audit events file corrupted: {e}")
                    import shutil
                    try:
                        shutil.copy2(events_file, events_file.with_suffix('.json.bak'))
                    except Exception as e:
                        logger.debug(f"Failed to backup corrupted audit events file: {e}")
                except Exception as e:
                    logger.error(f"Failed to load audit events: {e}")

            all_events = existing + [e.to_dict() for e in self._events]

        # Remove duplicates (by timestamp + action + source)
        seen = set()
        unique_events = []
        for event in reversed(all_events):
            key = f"{event['timestamp']}:{event['action']}:{event['source']}"
            if key not in seen:
                seen.add(key)
                unique_events.append(event)

        # Sort by timestamp and limit
        unique_events.sort(key=lambda x: x["timestamp"])

        # Apply retention
        cutoff = datetime.now() - timedelta(days=self.retention_days)
        unique_events = [
            e for e in unique_events
            if datetime.fromisoformat(e["timestamp"]) >= cutoff
        ]

        atomic_write_json(events_file, unique_events)

    def clear_events(self) -> None:
        with self._lock:
            self._events.clear()
            self._stats["by_level"] = {level.value: 0 for level in AuditLevel}
            self._stats["by_category"] = {cat.value: 0 for cat in AuditCategory}
            self._stats["by_source"] = {}
