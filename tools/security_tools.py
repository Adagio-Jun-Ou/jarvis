"""Security Tools for Jarvis.

Provides security-related tools that integrate with the Jarvis agent:
- DLP scanning and management
- Branch detection monitoring
- MITM protection status
- Security audit queries
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from tools.base import Tool as BaseTool
from core.security import (
    DataLeakProtector,
    DataBranchMonitor,
    MITMProtector,
    StackDefense,
    SecurityAuditor,
    AuditLevel,
    AuditCategory,
    AnomalySeverity,
)

logger = logging.getLogger(__name__)


class SecurityScannerTool(BaseTool):
    """Tool for scanning data for sensitive information."""

    name = "security_scan"
    description = "Scan data for sensitive information like API keys, passwords, and secrets"
    requires_permission = False

    def __init__(self, dlp: DataLeakProtector):
        self.dlp = dlp

    def get_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {
                        "type": "string",
                        "description": "Data to scan for sensitive information",
                    },
                    "source": {
                        "type": "string",
                        "description": "Source identifier (e.g., 'user_input', 'file_read')",
                        "default": "user_input",
                    },
                },
                "required": ["data"],
            },
        }

    async def execute(self, data: str, source: str = "user_input", **kwargs) -> str:
        """Scan data for sensitive information."""
        events = self.dlp.scan(data, source, "output")

        if not events:
            return "No sensitive information detected."

        result = [f"Detected {len(events)} potential data leak(s):\n"]
        for event in events:
            result.append(f"  - {event.pattern_type.value}: {event.severity.value}")
            result.append(f"    Preview: {event.data_preview}")
            result.append(f"    Blocked: {event.blocked}")
            result.append("")

        return "\n".join(result)


class SecurityStatusTool(BaseTool):
    """Tool for checking security status."""

    name = "security_status"
    description = "Get current security status and statistics"
    requires_permission = False

    def __init__(
        self,
        dlp: DataLeakProtector | None = None,
        branch_monitor: DataBranchMonitor | None = None,
        mitm_protector: MITMProtector | None = None,
        stack_defense: StackDefense | None = None,
        auditor: SecurityAuditor | None = None,
    ):
        self.dlp = dlp
        self.branch_monitor = branch_monitor
        self.mitm_protector = mitm_protector
        self.stack_defense = stack_defense
        self.auditor = auditor

    def get_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "component": {
                        "type": "string",
                        "description": "Specific component to check (dlp, branch, mitm, stack, audit, all)",
                        "default": "all",
                    },
                },
            },
        }

    async def execute(self, component: str = "all", **kwargs) -> str:
        """Get security status."""
        results = []

        if component in ("all", "dlp") and self.dlp:
            stats = self.dlp.get_stats()
            results.append("=== Data Leak Protection ===")
            results.append(f"Total detections: {stats['total_detections']}")
            results.append(f"Blocked attempts: {stats['blocked_attempts']}")
            results.append(f"By severity: {stats['by_severity']}")
            results.append("")

        if component in ("all", "branch") and self.branch_monitor:
            stats = self.branch_monitor.get_stats()
            results.append("=== Branch Detection ===")
            results.append(f"Total flows: {stats['total_flows']}")
            results.append(f"Total anomalies: {stats['total_anomalies']}")
            results.append(f"Unique destinations: {stats['unique_destinations']}")
            results.append("")

        if component in ("all", "mitm") and self.mitm_protector:
            stats = self.mitm_protector.get_stats()
            results.append("=== MITM Protection ===")
            results.append(f"Total threats: {stats['total_threats']}")
            results.append(f"Active sessions: {stats['active_sessions']}")
            results.append(f"TLS enforced: {stats['tls_enforced']}")
            results.append("")

        if component in ("all", "stack") and self.stack_defense:
            stats = self.stack_defense.get_stats()
            results.append("=== Stack Defense ===")
            results.append(f"Total violations: {stats['total_violations']}")
            results.append(f"Active allocations: {stats['active_allocations']}")
            results.append(f"Strict mode: {stats['strict_mode']}")
            results.append("")

        if component in ("all", "audit") and self.auditor:
            stats = self.auditor.get_stats()
            results.append("=== Security Audit ===")
            results.append(f"Total events: {stats['total_events']}")
            results.append(f"Failed events: {stats['failed_events']}")
            results.append(f"By level: {stats['by_level']}")
            results.append("")

        if not results:
            return "No security components available."

        return "\n".join(results)


class SecurityAuditTool(BaseTool):
    """Tool for querying security audit logs.
    
    This is a read-only operation that provides visibility into security events.
    No permission needed for viewing audit logs.
    """

    name = "security_audit"
    description = "Query security audit logs for events (read-only)"
    requires_permission = False  # Read-only operation

    def __init__(self, auditor: SecurityAuditor):
        self.auditor = auditor

    def get_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "level": {
                        "type": "string",
                        "description": "Filter by level (debug, info, warning, error, critical)",
                    },
                    "category": {
                        "type": "string",
                        "description": "Filter by category (authentication, authorization, data_access, etc.)",
                    },
                    "since_hours": {
                        "type": "integer",
                        "description": "Events from the last N hours",
                        "default": 24,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of events to return",
                        "default": 20,
                    },
                },
            },
        }

    async def execute(
        self,
        level: str | None = None,
        category: str | None = None,
        since_hours: int = 24,
        limit: int = 20,
        **kwargs
    ) -> str:
        """Query audit logs."""
        since = datetime.now() - timedelta(hours=since_hours)

        level_enum = None
        if level:
            try:
                level_enum = AuditLevel(level.lower())
            except ValueError:
                pass

        category_enum = None
        if category:
            try:
                category_enum = AuditCategory(category.lower())
            except ValueError:
                pass

        events = self.auditor.get_events(
            level=level_enum,
            category=category_enum,
            since=since,
            limit=limit,
        )

        if not events:
            return "No audit events found matching the criteria."

        result = [f"Found {len(events)} audit event(s):\n"]
        for event in events:
            result.append(f"[{event.timestamp.strftime('%Y-%m-%d %H:%M:%S')}]")
            result.append(f"  Level: {event.level.value}")
            result.append(f"  Category: {event.category.value}")
            result.append(f"  Action: {event.action}")
            result.append(f"  Source: {event.source}")
            result.append(f"  Success: {event.success}")
            if event.description:
                result.append(f"  Description: {event.description}")
            result.append("")

        return "\n".join(result)


class SecurityReportTool(BaseTool):
    """Tool for generating security reports.
    
    This is a read-only operation that provides security status visibility.
    No permission needed for viewing security reports.
    """

    name = "security_report"
    description = "Generate a comprehensive security report (read-only)"
    requires_permission = False  # Read-only operation

    def __init__(self, auditor: SecurityAuditor):
        self.auditor = auditor

    def get_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "period_hours": {
                        "type": "integer",
                        "description": "Report period in hours",
                        "default": 24,
                    },
                },
            },
        }

    async def execute(self, period_hours: int = 24, **kwargs) -> str:
        """Generate security report."""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=period_hours)

        report = self.auditor.generate_report(start_time, end_time)

        result = [
            "=== Security Report ===",
            f"Period: {report['summary']['period_start']} to {report['summary']['period_end']}",
            "",
            "=== Summary ===",
            f"Total events: {report['summary']['total_events']}",
            f"Failed events: {report['summary']['failed_events']}",
            f"Critical events: {report['summary']['critical_events']}",
            f"Warning events: {report['summary']['warning_events']}",
            "",
            "=== Authentication ===",
            f"Total attempts: {report['authentication']['total_attempts']}",
            f"Successful: {report['authentication']['successful']}",
            f"Failed: {report['authentication']['failed']}",
            "",
            "=== Authorization ===",
            f"Total requests: {report['authorization']['total_requests']}",
            f"Granted: {report['authorization']['granted']}",
            f"Denied: {report['authorization']['denied']}",
            "",
            "=== Data Access ===",
            f"Total access: {report['data']['total_access']}",
            f"Modifications: {report['data']['total_modifications']}",
            f"Exfiltration attempts: {report['data']['exfiltration_attempts']}",
            "",
            "=== Security Events ===",
            f"Total: {report['security']['total_events']}",
        ]

        if report['security']['by_action']:
            result.append("By action:")
            for action, count in report['security']['by_action'].items():
                result.append(f"  - {action}: {count}")

        return "\n".join(result)


class DataFlowMonitorTool(BaseTool):
    """Tool for monitoring data flows.
    
    This is a read-only operation that provides visibility into data patterns.
    No permission needed for monitoring.
    """

    name = "data_flow_monitor"
    description = "Monitor and check data flows for anomalies (read-only)"
    requires_permission = False  # Read-only operation

    def __init__(self, branch_monitor: DataBranchMonitor):
        self.branch_monitor = branch_monitor

    def get_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "Action to perform (stats, anomalies, history)",
                        "enum": ["stats", "anomalies", "history"],
                    },
                    "severity": {
                        "type": "string",
                        "description": "Filter anomalies by severity (info, warning, error, critical)",
                    },
                },
                "required": ["action"],
            },
        }

    async def execute(
        self,
        action: str = "",
        severity: str | None = None,
        **kwargs
    ) -> str:
        """Monitor data flows."""
        if action == "stats":
            stats = self.branch_monitor.get_stats()
            return (
                f"Data Flow Statistics:\n"
                f"  Total flows: {stats['total_flows']}\n"
                f"  Total anomalies: {stats['total_anomalies']}\n"
                f"  Unique destinations: {stats['unique_destinations']}\n"
                f"  Authorized destinations: {stats['authorized_destinations']}\n"
                f"  By type: {stats['flows_by_type']}"
            )

        elif action == "anomalies":
            severity_enum = None
            if severity:
                try:
                    severity_enum = AnomalySeverity(severity.lower())
                except ValueError:
                    pass

            anomalies = self.branch_monitor.get_anomalies(severity=severity_enum)

            if not anomalies:
                return "No anomalies found."

            result = [f"Found {len(anomalies)} anomalies:\n"]
            for anomaly in anomalies[-10:]:  # Last 10
                result.append(f"[{anomaly.timestamp.strftime('%H:%M:%S')}] {anomaly.severity.value}")
                result.append(f"  Type: {anomaly.anomaly_type.value}")
                result.append(f"  Description: {anomaly.description}")
                result.append(f"  Action: {anomaly.action_taken}")
                result.append("")

            return "\n".join(result)

        elif action == "history":
            events = self.branch_monitor.get_flow_history(limit=20)

            if not events:
                return "No flow history found."

            result = [f"Last {len(events)} data flows:\n"]
            for event in events:
                result.append(f"[{event.timestamp.strftime('%H:%M:%S')}] {event.branch_type.value}")
                result.append(f"  {event.source} -> {event.destination}")
                result.append(f"  Size: {event.data_size} bytes")
                result.append("")

            return "\n".join(result)

        return f"Unknown action: {action}"


class WhitelistManagerTool(BaseTool):
    """Tool for managing DLP whitelist."""

    name = "dlp_whitelist"
    description = "Manage data leak protection whitelist"
    requires_permission = True

    def __init__(self, dlp: DataLeakProtector):
        self.dlp = dlp

    def get_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "Action to perform",
                        "enum": ["add", "stats"],
                    },
                    "data": {
                        "type": "string",
                        "description": "Data to add to whitelist (for 'add' action)",
                    },
                },
                "required": ["action"],
            },
        }

    async def execute(self, action: str, data: str | None = None, **kwargs) -> str:
        """Manage DLP whitelist."""
        if action == "add":
            if not data:
                return "Error: 'data' parameter required for 'add' action"
            self.dlp.add_to_whitelist(data)
            return "Data added to whitelist successfully."

        elif action == "stats":
            stats = self.dlp.get_stats()
            return (
                f"DLP Statistics:\n"
                f"  Total detections: {stats['total_detections']}\n"
                f"  Blocked attempts: {stats['blocked_attempts']}\n"
                f"  By severity: {stats['by_severity']}\n"
                f"  Patterns detected: {stats['patterns_detected']}"
            )

        return f"Unknown action: {action}"


def get_security_tools(
    storage_path: Path | None = None,
    enable_dlp: bool = True,
    enable_branch_monitor: bool = True,
    enable_mitm_protection: bool = True,
    enable_stack_defense: bool = True,
) -> tuple[list[BaseTool], dict[str, Any]]:
    """Create and return security tools and components.

    Args:
        storage_path: Path for storing security data
        enable_dlp: Enable data leak protection
        enable_branch_monitor: Enable branch detection
        enable_mitm_protection: Enable MITM protection
        enable_stack_defense: Enable stack defense

    Returns:
        Tuple of (tools list, components dict)
    """
    security_path = storage_path / "security" if storage_path else None

    # Initialize components
    dlp = DataLeakProtector(storage_path=security_path / "dlp" if security_path else None)
    branch_monitor = DataBranchMonitor(storage_path=security_path / "branch" if security_path else None)
    mitm_protector = MITMProtector(storage_path=security_path / "mitm" if security_path else None)
    stack_defense = StackDefense(storage_path=security_path / "stack" if security_path else None)
    auditor = SecurityAuditor(storage_path=security_path / "audit" if security_path else None)

    # Create tools
    tools = [
        SecurityScannerTool(dlp),
        SecurityStatusTool(dlp, branch_monitor, mitm_protector, stack_defense, auditor),
        SecurityAuditTool(auditor),
        SecurityReportTool(auditor),
        DataFlowMonitorTool(branch_monitor),
        WhitelistManagerTool(dlp),
    ]

    components = {
        "dlp": dlp,
        "branch_monitor": branch_monitor,
        "mitm_protector": mitm_protector,
        "stack_defense": stack_defense,
        "auditor": auditor,
    }

    return tools, components
