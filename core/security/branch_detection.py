"""Unauthorized Data Branch Detection for Jarvis.

Monitors and detects anomalous data flows that may indicate:
- Unauthorized data exfiltration paths
- Data branching to unexpected destinations
- Data duplication/cloning attempts
- Side-channel data leakage
- Process isolation violations
"""

import hashlib
import logging
import threading
from core.helpers import atomic_write_json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable
from collections import defaultdict

logger = logging.getLogger(__name__)


class FlowDirection(str, Enum):
    """Direction of data flow."""
    INBOUND = "inbound"
    OUTBOUND = "outbound"
    INTERNAL = "internal"


class BranchType(str, Enum):
    """Types of data branches."""
    FILE_WRITE = "file_write"
    NETWORK_SEND = "network_send"
    CLIPBOARD = "clipboard"
    PROCESS_PIPE = "process_pipe"
    MEMORY_SHARE = "memory_share"
    SOCKET = "socket"
    DATABASE = "database"
    API_CALL = "api_call"
    ENVIRONMENT = "environment"


class AnomalyType(str, Enum):
    """Types of branch anomalies."""
    UNAUTHORIZED_DESTINATION = "unauthorized_destination"
    UNEXPECTED_VOLUME = "unexpected_volume"
    UNUSUAL_TIMING = "unusual_timing"
    DATA_DUPLICATION = "data_duplication"
    SENSITIVE_FLOW = "sensitive_flow"
    POLICY_VIOLATION = "policy_violation"
    ISOLATION_BREACH = "isolation_breach"


class AnomalySeverity(str, Enum):
    """Severity of branch anomalies."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class DataFlowEvent:
    """Represents a data flow event."""
    timestamp: datetime
    branch_type: BranchType
    direction: FlowDirection
    source: str
    destination: str
    data_hash: str  # Hash of the data for tracking
    data_size: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "branch_type": self.branch_type.value,
            "direction": self.direction.value,
            "source": self.source,
            "destination": self.destination,
            "data_hash": self.data_hash,
            "data_size": self.data_size,
            "metadata": self.metadata,
        }


@dataclass
class BranchAnomaly:
    """Represents a detected branch anomaly."""
    timestamp: datetime
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    description: str
    flow_event: DataFlowEvent
    related_events: list[DataFlowEvent] = field(default_factory=list)
    action_taken: str = "logged"
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "anomaly_type": self.anomaly_type.value,
            "severity": self.severity.value,
            "description": self.description,
            "flow_event": self.flow_event.to_dict(),
            "related_events": [e.to_dict() for e in self.related_events],
            "action_taken": self.action_taken,
            "context": self.context,
        }


class DataBranchMonitor:
    """Monitors data flows for unauthorized branching."""

    MAX_FLOW_HISTORY = 10000
    MAX_HASH_EVENTS = 100
    MAX_DESTINATION_EVENTS = 100
    MAX_ANOMALIES = 500
    MAX_HASH_KEYS = 500
    MAX_DESTINATION_KEYS = 500
    MAX_VOLUME_KEYS = 500

    def __init__(
        self,
        storage_path: Path | None = None,
        authorized_destinations: set[str] | None = None,
        enforce_destination_whitelist: bool = False,
        max_data_per_minute: int = 10 * 1024 * 1024,  # 10MB
        audit_callback: Callable[[BranchAnomaly], None] | None = None,
    ):
        self.storage_path = storage_path
        self.authorized_destinations = authorized_destinations or set()
        self.enforce_destination_whitelist = enforce_destination_whitelist
        self.max_data_per_minute = max_data_per_minute
        self.audit_callback = audit_callback
        self._lock = threading.RLock()

        # Flow tracking
        self._flow_history: list[DataFlowEvent] = []
        self._data_hashes: dict[str, list[DataFlowEvent]] = defaultdict(list)
        self._destinations: dict[str, list[DataFlowEvent]] = defaultdict(list)

        # Volume tracking per minute
        self._volume_window: dict[str, list[tuple[datetime, int]]] = defaultdict(list)

        # Anomaly tracking
        self._anomalies: list[BranchAnomaly] = []

        self._BRANCH_TYPE_TO_POLICY = {
            BranchType.NETWORK_SEND: "network",
            BranchType.SOCKET: "network",
            BranchType.FILE_WRITE: "file",
            BranchType.DATABASE: "file",
            BranchType.PROCESS_PIPE: "process",
            BranchType.ENVIRONMENT: "process",
            BranchType.CLIPBOARD: "network",
            BranchType.MEMORY_SHARE: "process",
            BranchType.API_CALL: "network",
        }

        # Policy rules
        self._policies: dict[str, dict[str, Any]] = {
            "network": {
                "allow_http": True,
                "allow_https": True,
                "block_ports": [23, 21, 25],  # Telnet, FTP, SMTP
                "require_tls": True,
            },
            "file": {
                "allowed_paths": ["~"],  # Home directory by default
                "blocked_paths": ["/etc/passwd", "/etc/shadow", "~/.ssh"],
                "max_file_size": 100 * 1024 * 1024,  # 100MB
            },
            "process": {
                "allow_pipes": True,
                "block_system_commands": ["rm -rf", "mkfs", "dd if="],
            },
        }

    def add_authorized_destination(self, destination: str) -> None:
        with self._lock:
            self.authorized_destinations.add(destination)

    def remove_authorized_destination(self, destination: str) -> None:
        with self._lock:
            self.authorized_destinations.discard(destination)

    def set_policy(self, branch_type: str, policy: dict[str, Any]) -> None:
        with self._lock:
            self._policies[branch_type] = policy

    def _hash_data(self, data: str | bytes) -> str:
        """Create a hash of data for tracking."""
        if isinstance(data, str):
            data = data.encode()
        return hashlib.sha256(data).hexdigest()[:16]

    def _clean_old_volume_entries(self, key: str) -> None:
        """Remove volume entries older than 1 minute."""
        cutoff = datetime.now() - timedelta(minutes=1)
        self._volume_window[key] = [
            (ts, size) for ts, size in self._volume_window[key]
            if ts > cutoff
        ]

    def _get_volume_last_minute(self, key: str) -> int:
        """Get total volume for a key in the last minute."""
        self._clean_old_volume_entries(key)
        return sum(size for _, size in self._volume_window[key])

    def track_flow(
        self,
        branch_type: BranchType,
        direction: FlowDirection,
        source: str,
        destination: str,
        data: str | bytes,
        metadata: dict[str, Any] | None = None,
    ) -> DataFlowEvent:
        """Track a data flow event.

        Args:
            branch_type: Type of data branch
            direction: Direction of flow
            source: Source identifier
            destination: Destination identifier
            data: The data being transferred
            metadata: Additional metadata

        Returns:
            DataFlowEvent tracking record
        """
        data_hash = self._hash_data(data)
        data_size = len(data) if isinstance(data, (str, bytes)) else 0

        event = DataFlowEvent(
            timestamp=datetime.now(),
            branch_type=branch_type,
            direction=direction,
            source=source,
            destination=destination,
            data_hash=data_hash,
            data_size=data_size,
            metadata=metadata or {},
        )

        with self._lock:
            self._flow_history.append(event)
            self._data_hashes[data_hash].append(event)
            self._destinations[destination].append(event)

            if len(self._flow_history) > self.MAX_FLOW_HISTORY:
                self._flow_history = self._flow_history[-self.MAX_FLOW_HISTORY:]
            if len(self._data_hashes[data_hash]) > self.MAX_HASH_EVENTS:
                self._data_hashes[data_hash] = self._data_hashes[data_hash][-self.MAX_HASH_EVENTS:]
            if len(self._destinations[destination]) > self.MAX_DESTINATION_EVENTS:
                self._destinations[destination] = self._destinations[destination][-self.MAX_DESTINATION_EVENTS:]
            if len(self._data_hashes) > self.MAX_HASH_KEYS:
                oldest_keys = list(self._data_hashes.keys())[:len(self._data_hashes) - self.MAX_HASH_KEYS]
                for k in oldest_keys:
                    del self._data_hashes[k]
            if len(self._destinations) > self.MAX_DESTINATION_KEYS:
                oldest_keys = list(self._destinations.keys())[:len(self._destinations) - self.MAX_DESTINATION_KEYS]
                for k in oldest_keys:
                    del self._destinations[k]

            volume_key = f"{source}:{branch_type.value}"
            self._volume_window[volume_key].append((event.timestamp, data_size))
            if len(self._volume_window) > self.MAX_VOLUME_KEYS:
                oldest_keys = list(self._volume_window.keys())[:len(self._volume_window) - self.MAX_VOLUME_KEYS]
                for k in oldest_keys:
                    del self._volume_window[k]

        return event

    def check_branch_policy(
        self,
        branch_type: BranchType,
        destination: str,
        data: str | bytes,
    ) -> list[BranchAnomaly]:
        """Check if a data branch violates policy.

        Args:
            branch_type: Type of branch
            destination: Destination of data
            data: Data being transferred

        Returns:
            List of detected anomalies
        """
        anomalies = []
        data_size = len(data) if isinstance(data, (str, bytes)) else 0
        data_hash = self._hash_data(data)

        prelim_event = DataFlowEvent(
            timestamp=datetime.now(),
            branch_type=branch_type,
            direction=FlowDirection.OUTBOUND,
            source="unknown",
            destination=destination,
            data_hash=data_hash,
            data_size=data_size,
        )

        with self._lock:
            policy_key = self._BRANCH_TYPE_TO_POLICY.get(branch_type)
            policy = self._policies.get(policy_key, {}) if policy_key else {}

            if self.enforce_destination_whitelist and destination not in self.authorized_destinations:
                anomaly = BranchAnomaly(
                    timestamp=datetime.now(),
                    anomaly_type=AnomalyType.UNAUTHORIZED_DESTINATION,
                    severity=AnomalySeverity.WARNING,
                    description=f"Data flow to unauthorized destination: {destination}",
                    flow_event=prelim_event,
                    action_taken="logged",
                )
                anomalies.append(anomaly)

            volume_key = f"unknown:{branch_type.value}"
            current_volume = self._get_volume_last_minute(volume_key)
            if current_volume + data_size > self.max_data_per_minute:
                anomaly = BranchAnomaly(
                    timestamp=datetime.now(),
                    anomaly_type=AnomalyType.UNEXPECTED_VOLUME,
                    severity=AnomalySeverity.WARNING,
                    description=f"Data volume exceeds limit: {current_volume + data_size} > {self.max_data_per_minute}",
                    flow_event=prelim_event,
                    action_taken="logged",
                )
                anomalies.append(anomaly)

            if data_hash in self._data_hashes:
                existing_destinations = set(
                    e.destination for e in self._data_hashes[data_hash]
                )
                if destination not in existing_destinations and len(existing_destinations) >= 1:
                    anomaly = BranchAnomaly(
                        timestamp=datetime.now(),
                        anomaly_type=AnomalyType.DATA_DUPLICATION,
                        severity=AnomalySeverity.WARNING,
                        description="Data duplication detected: same data going to multiple destinations",
                        flow_event=prelim_event,
                        related_events=list(self._data_hashes[data_hash])[-3:],
                        action_taken="logged",
                    )
                    anomalies.append(anomaly)

            if branch_type == BranchType.NETWORK_SEND:
                if not destination.startswith(("http://", "https://")):
                    if not policy.get("allow_http", True):
                        anomaly = BranchAnomaly(
                            timestamp=datetime.now(),
                            anomaly_type=AnomalyType.POLICY_VIOLATION,
                            severity=AnomalySeverity.ERROR,
                            description=f"Non-HTTP network destination: {destination}",
                            flow_event=prelim_event,
                            action_taken="logged",
                        )
                        anomalies.append(anomaly)

                if policy.get("require_tls", True) and destination.startswith("http://"):
                    anomaly = BranchAnomaly(
                        timestamp=datetime.now(),
                        anomaly_type=AnomalyType.POLICY_VIOLATION,
                        severity=AnomalySeverity.WARNING,
                        description=f"Non-TLS HTTP connection to: {destination}",
                        flow_event=prelim_event,
                        action_taken="logged",
                    )
                    anomalies.append(anomaly)

            elif branch_type == BranchType.FILE_WRITE:
                blocked_paths = policy.get("blocked_paths", [])
                for blocked in blocked_paths:
                    if blocked.replace("~", str(Path.home())) in destination:
                        anomaly = BranchAnomaly(
                            timestamp=datetime.now(),
                            anomaly_type=AnomalyType.POLICY_VIOLATION,
                            severity=AnomalySeverity.CRITICAL,
                            description=f"Write to blocked path: {destination}",
                            flow_event=prelim_event,
                            action_taken="blocked",
                        )
                        anomalies.append(anomaly)

                max_size = policy.get("max_file_size", 100 * 1024 * 1024)
                if data_size > max_size:
                    anomaly = BranchAnomaly(
                        timestamp=datetime.now(),
                        anomaly_type=AnomalyType.UNEXPECTED_VOLUME,
                        severity=AnomalySeverity.WARNING,
                        description=f"File size exceeds limit: {data_size} > {max_size}",
                        flow_event=prelim_event,
                        action_taken="logged",
                    )
                    anomalies.append(anomaly)

            elif branch_type == BranchType.PROCESS_PIPE:
                blocked_commands = policy.get("block_system_commands", [])
                data_str = data.decode() if isinstance(data, bytes) else data
                for blocked in blocked_commands:
                    if blocked in data_str:
                        anomaly = BranchAnomaly(
                            timestamp=datetime.now(),
                            anomaly_type=AnomalyType.POLICY_VIOLATION,
                            severity=AnomalySeverity.CRITICAL,
                            description=f"Blocked system command detected: {blocked}",
                            flow_event=prelim_event,
                            action_taken="blocked",
                        )
                        anomalies.append(anomaly)

            for anomaly in anomalies:
                self._anomalies.append(anomaly)
                if len(self._anomalies) > self.MAX_ANOMALIES:
                    self._anomalies = self._anomalies[-self.MAX_ANOMALIES:]

        for anomaly in anomalies:
            if self.audit_callback:
                self.audit_callback(anomaly)

        return anomalies

    def get_flow_history(
        self,
        branch_type: BranchType | None = None,
        direction: FlowDirection | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[DataFlowEvent]:
        with self._lock:
            events = list(self._flow_history)

        if branch_type:
            events = [e for e in events if e.branch_type == branch_type]
        if direction:
            events = [e for e in events if e.direction == direction]
        if since:
            events = [e for e in events if e.timestamp >= since]

        return events[-limit:]

    def get_anomalies(
        self,
        severity: AnomalySeverity | None = None,
        anomaly_type: AnomalyType | None = None,
        since: datetime | None = None,
    ) -> list[BranchAnomaly]:
        with self._lock:
            anomalies = list(self._anomalies)

        if severity:
            anomalies = [a for a in anomalies if a.severity == severity]
        if anomaly_type:
            anomalies = [a for a in anomalies if a.anomaly_type == anomaly_type]
        if since:
            anomalies = [a for a in anomalies if a.timestamp >= since]

        return anomalies

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            total_flows = len(self._flow_history)
            total_anomalies = len(self._anomalies)

            flows_by_type: dict[str, int] = defaultdict(int)
            for event in self._flow_history:
                flows_by_type[event.branch_type.value] += 1

            anomalies_by_severity: dict[str, int] = defaultdict(int)
            for anomaly in self._anomalies:
                anomalies_by_severity[anomaly.severity.value] += 1

        return {
            "total_flows": total_flows,
            "total_anomalies": total_anomalies,
            "flows_by_type": dict(flows_by_type),
            "anomalies_by_severity": dict(anomalies_by_severity),
            "unique_destinations": len(self._destinations),
            "authorized_destinations": len(self.authorized_destinations),
        }

    def save_state(self) -> None:
        if not self.storage_path:
            return

        self.storage_path.mkdir(parents=True, exist_ok=True)

        with self._lock:
            flows_data = [e.to_dict() for e in self._flow_history[-1000:]]
            anomalies_data = [a.to_dict() for a in self._anomalies[-100:]]

        flows_file = self.storage_path / "branch_flows.json"
        atomic_write_json(flows_file, flows_data)

        anomalies_file = self.storage_path / "branch_anomalies.json"
        atomic_write_json(anomalies_file, anomalies_data)

    def clear_history(self) -> None:
        with self._lock:
            self._flow_history.clear()
            self._data_hashes.clear()
            self._destinations.clear()
            self._volume_window.clear()
