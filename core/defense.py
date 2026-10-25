"""Data Defense Agent for Jarvis.

A comprehensive defense system that monitors, detects, and responds to threats
against both user data and Jarvis itself. Inspired by modern security operations
centers (SOC) with automated threat intelligence.

Features:
- Continuous threat monitoring
- Anomaly detection in system behavior
- Self-protection for Jarvis binaries and configs
- Automatic threat response
- Integration with evolution system for adaptive defense
"""
import asyncio
import hashlib
import logging
from core.helpers import atomic_write_json, create_tracked_task, read_json_with_integrity, safe_error_message
import math
import re
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)
from core.prefrontal_cortex import get_prefrontal_cortex, ConceptType


class ThreatLevel(Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatType(Enum):
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_EXFILTRATION = "data_exfiltration"
    MALICIOUS_CODE = "malicious_code"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"
    INJECTION_ATTACK = "injection_attack"
    RESOURCE_ABUSE = "resource_abuse"
    INTEGRITY_VIOLATION = "integrity_violation"
    CONFIGURATION_CHANGE = "configuration_change"
    SUSPICIOUS_PROCESS = "suspicious_process"
    NETWORK_ANOMALY = "network_anomaly"
    FILE_INTEGRITY = "file_integrity"


class DefenseAction(Enum):
    LOG = "log"
    ALERT = "alert"
    BLOCK = "block"
    QUARANTINE = "quarantine"
    TERMINATE = "terminate"
    ISOLATE = "isolate"
    RESTORE = "restore"
    LEARN = "learn"


@dataclass
class ThreatEvent:
    id: str
    timestamp: datetime
    threat_type: ThreatType
    level: ThreatLevel
    source: str
    description: str
    details: dict
    action_taken: DefenseAction
    resolved: bool = False
    resolved_at: datetime | None = None
    resolution: str | None = None
    similar_count: int = 0
    patterns_matched: list[str] = field(default_factory=list)


@dataclass
class FileIntegrityRecord:
    path: str
    hash: str
    size: int
    modified_time: float
    permissions: int
    last_checked: datetime
    is_protected: bool = False


@dataclass
class BehaviorBaseline:
    metric_name: str
    baseline_value: float
    std_deviation: float
    samples: int
    last_updated: datetime
    threshold_high: float
    threshold_low: float


@dataclass
class DefenseRule:
    id: str
    name: str
    threat_type: ThreatType
    conditions: dict
    actions: list[DefenseAction]
    enabled: bool = True
    priority: int = 50
    created_at: datetime = field(default_factory=datetime.now)
    triggered_count: int = 0


class DataDefenseAgent:
    """Comprehensive data defense and self-protection system.
    
    Monitors system behavior, detects threats, and takes automated
    defensive actions to protect both user data and Jarvis itself.
    """

    MAX_EVENTS = 1000
    MAX_BASELINE_SAMPLES = 1000
    MAX_INTEGRITY_RECORDS = 500
    BASELINE_WINDOW = timedelta(hours=24)

    def __init__(
        self,
        jarvis_root: Path | None = None,
        llm_client: Any = None,
        alert_callback: Callable | None = None,
    ):
        self.jarvis_root = jarvis_root or Path(__file__).parent.parent
        self.llm_client = llm_client
        self.alert_callback = alert_callback

        self.events: deque[ThreatEvent] = deque(maxlen=self.MAX_EVENTS)
        self.integrity_records: dict[str, FileIntegrityRecord] = {}
        self._integrity_order: deque[str] = deque(maxlen=self.MAX_INTEGRITY_RECORDS)
        self.baselines: dict[str, BehaviorBaseline] = {}
        self.rules: dict[str, DefenseRule] = {}
        self.blocked_entities: set[str] = set()
        self.quarantined_files: dict[str, datetime] = {}

        self._event_counter = 0
        self._running = False
        self._monitor_task: asyncio.Task | None = None
        self._lock = threading.RLock()

        self._stats = {
            "threats_detected": 0,
            "threats_blocked": 0,
            "threats_quarantined": 0,
            "false_positives": 0,
            "auto_resolved": 0,
            "baseline_updates": 0,
        }

        self._init_default_rules()
        self._load_state()

    def _init_default_rules(self) -> None:
        default_rules = [
            DefenseRule(
                id="rule_001",
                name="Unauthorized Jarvis Modification",
                threat_type=ThreatType.INTEGRITY_VIOLATION,
                conditions={"path_pattern": "jarvis/**", "change_type": "modified"},
                actions=[DefenseAction.ALERT, DefenseAction.BLOCK, DefenseAction.RESTORE],
                priority=90,
            ),
            DefenseRule(
                id="rule_002",
                name="Suspicious Process Injection",
                threat_type=ThreatType.MALICIOUS_CODE,
                conditions={"memory_write": True, "target": "jarvis_process"},
                actions=[DefenseAction.ALERT, DefenseAction.TERMINATE, DefenseAction.ISOLATE],
                priority=95,
            ),
            DefenseRule(
                id="rule_003",
                name="Excessive Resource Usage",
                threat_type=ThreatType.RESOURCE_ABUSE,
                conditions={"cpu_percent": 90, "duration": "5m"},
                actions=[DefenseAction.ALERT, DefenseAction.LEARN],
                priority=40,
            ),
            DefenseRule(
                id="rule_004",
                name="Sensitive Data Access Pattern",
                threat_type=ThreatType.DATA_EXFILTRATION,
                conditions={"file_pattern": "*.key,*.pem,.env,*secret*", "access_count": 10},
                actions=[DefenseAction.ALERT, DefenseAction.BLOCK],
                priority=80,
            ),
            DefenseRule(
                id="rule_005",
                name="Injection Attack Detection",
                threat_type=ThreatType.INJECTION_ATTACK,
                conditions={"input_pattern": "sql_injection,xss,command_injection"},
                actions=[DefenseAction.BLOCK, DefenseAction.LOG, DefenseAction.LEARN],
                priority=85,
            ),
            DefenseRule(
                id="rule_006",
                name="Configuration Tampering",
                threat_type=ThreatType.CONFIGURATION_CHANGE,
                conditions={"path_pattern": "*.yaml,*.json,*.conf", "change_type": "modified"},
                actions=[DefenseAction.ALERT, DefenseAction.LEARN],
                priority=70,
            ),
            DefenseRule(
                id="rule_007",
                name="Anomalous Network Activity",
                threat_type=ThreatType.NETWORK_ANOMALY,
                conditions={"connection_rate": 100, "unique_hosts": 50},
                actions=[DefenseAction.ALERT, DefenseAction.BLOCK],
                priority=75,
            ),
            DefenseRule(
                id="rule_008",
                name="Brute Force Detection",
                threat_type=ThreatType.UNAUTHORIZED_ACCESS,
                conditions={"failed_attempts": 5, "window": "1m"},
                actions=[DefenseAction.BLOCK, DefenseAction.ALERT],
                priority=80,
            ),
        ]

        for rule in default_rules:
            self.rules[rule.id] = rule

    def _load_state(self) -> None:
        state_file = self.jarvis_root / "logs" / "defense_state.json"
        if state_file.exists():
            try:
                data = read_json_with_integrity(state_file)
                self._stats = data.get("stats", self._stats)
                self.blocked_entities = set(data.get("blocked_entities", []))
                logger.info(f"Loaded defense state: {self._stats['threats_detected']} threats detected")
            except Exception as e:
                logger.warning(f"Failed to load defense state: {e}")

    def _save_state(self) -> None:
        state_file = self.jarvis_root / "logs" / "defense_state.json"
        state_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            atomic_write_json(state_file, {
                    "stats": self._stats,
                    "blocked_entities": list(self.blocked_entities),
                    "last_updated": datetime.now().isoformat(),
                })
        except Exception as e:
            logger.warning(f"Failed to save defense state: {e}")

    async def start_monitoring(self) -> None:
        if self._running:
            return

        self._running = True
        self._monitor_task = create_tracked_task(self._monitoring_loop(), name="defense_monitoring_loop")
        logger.info("Data defense agent started - monitoring active")

    async def stop_monitoring(self) -> None:
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        self._save_state()
        logger.info("Data defense agent stopped")

    async def _monitoring_loop(self) -> None:
        while self._running:
            try:
                await self._check_file_integrity()
                await self._check_system_behavior()
                await self._cleanup_quarantine()

                await asyncio.sleep(60)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Defense monitoring error: {e}")
                await asyncio.sleep(30)

    def scan_file(self, file_path: Path | str) -> dict:
        """Scan a file for threats."""
        path = Path(file_path)
        if not path.exists():
            return {"status": "not_found", "path": str(path)}

        results = {
            "path": str(path),
            "size": path.stat().st_size,
            "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
            "threats": [],
            "risk_score": 0,
        }

        try:
            content = path.read_bytes()
            results["hash"] = hashlib.sha256(content).hexdigest()

            suspicious_patterns = [
                (rb"eval\s*\(", "eval usage", 30),
                (rb"exec\s*\(", "exec usage", 30),
                (rb"__import__", "dynamic import", 20),
                (rb"subprocess\.(call|run|Popen)", "subprocess call", 15),
                (rb"os\.system", "os.system call", 25),
                (rb"base64\.b64decode", "base64 decoding", 10),
                (rb"pickle\.loads?", "pickle deserialization", 35),
                (rb"marshal\.loads?", "marshal deserialization", 30),
                (rb"\.decode\(['\"]utf-8['\"]\)", "string decoding", 5),
                (rb"password\s*[=:]", "password variable", 40),
                (rb"api[_-]?key\s*[=:]", "api key variable", 40),
                (rb"secret\s*[=:]", "secret variable", 35),
                (rb"token\s*[=:]", "token variable", 30),
                (rb"-----BEGIN.*PRIVATE KEY-----", "private key", 50),
                (rb"curl\s+", "curl usage", 10),
                (rb"wget\s+", "wget usage", 10),
                (rb"/etc/passwd", "passwd file access", 45),
                (rb"/etc/shadow", "shadow file access", 50),
            ]

            for pattern, desc, score in suspicious_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    results["threats"].append({
                        "type": "suspicious_pattern",
                        "description": desc,
                        "score": score,
                    })
                    results["risk_score"] += score

            if results["risk_score"] >= 50:
                results["status"] = "high_risk"
            elif results["risk_score"] >= 25:
                results["status"] = "medium_risk"
            elif results["risk_score"] >= 10:
                results["status"] = "low_risk"
            else:
                results["status"] = "clean"

        except Exception as e:
            results["error"] = safe_error_message(e)
            results["status"] = "scan_error"

        return results

    def scan_input(self, input_text: str, context: str = "general") -> dict:
        """Scan user input for injection attacks and malicious patterns."""
        results = {
            "input_length": len(input_text),
            "threats": [],
            "risk_score": 0,
            "action": "allow",
        }

        injection_patterns = [
            (r";\s*rm\s", "command injection attempt", 80),
            (r";\s*del\s", "command injection attempt", 80),
            (r";\s*format\s", "command injection attempt", 75),
            (r"\|\s*sh\b", "shell pipe injection", 70),
            (r"\|\s*bash\b", "bash pipe injection", 70),
            (r"\$\([^)]+\)", "command substitution", 50),
            (r"`[^`]+`", "backtick execution", 50),
            (r"UNION\s+SELECT", "SQL injection", 85),
            (r"OR\s+1\s*=\s*1", "SQL injection", 90),
            (r"OR\s+'[^']*'\s*=\s*'[^']*'", "SQL injection", 85),
            (r";\s*DROP\s+", "SQL injection", 95),
            (r"<\s*script", "XSS attempt", 70),
            (r"javascript:", "XSS attempt", 75),
            (r"on\w+\s*=", "event handler injection", 65),
            (r"{{\s*[^}]+\s*}}", "template injection", 60),
            (r"{%\s*[^%]+\s*%}", "template injection", 60),
            (r"\.\./", "path traversal", 70),
            (r"\.\.\\", "path traversal", 70),
            (r"/etc/(passwd|shadow|hosts)", "sensitive file access", 80),
            (r"\\\\", "UNC path injection", 60),
            (r">\s*/dev/", "device write attempt", 75),
        ]

        for pattern, desc, score in injection_patterns:
            if re.search(pattern, input_text, re.IGNORECASE):
                results["threats"].append({
                    "type": "injection_pattern",
                    "description": desc,
                    "pattern": pattern,
                    "score": score,
                })
                results["risk_score"] += score

        if results["risk_score"] >= 50:
            results["action"] = "block"
            self._record_threat(
                ThreatType.INJECTION_ATTACK,
                ThreatLevel.HIGH,
                "input_scanner",
                f"Blocked malicious input: {results['threats'][0]['description']}",
                {"input_preview": input_text[:100], "threats": results["threats"]},
                DefenseAction.BLOCK,
            )
        elif results["risk_score"] >= 25:
            results["action"] = "sanitize"

        return results

    def protect_file(self, file_path: Path | str) -> bool:
        """Add a file to integrity protection."""
        path = Path(file_path)
        if not path.exists():
            return False

        try:
            stat_info = path.stat()
            content = path.read_bytes()

            record = FileIntegrityRecord(
                path=str(path),
                hash=hashlib.sha256(content).hexdigest(),
                size=stat_info.st_size,
                modified_time=stat_info.st_mtime,
                permissions=stat_info.st_mode,
                last_checked=datetime.now(),
                is_protected=True,
            )

            self.integrity_records[str(path)] = record
            self._integrity_order.append(str(path))
            
            # Clean up integrity_records for evicted keys
            max_cleanup_iterations = self.MAX_INTEGRITY_RECORDS + 1
            while len(self.integrity_records) > self.MAX_INTEGRITY_RECORDS and max_cleanup_iterations > 0:
                max_cleanup_iterations -= 1
                # Find oldest keys not in _integrity_order
                order_set = set(self._integrity_order)
                removed = False
                for key in list(self.integrity_records.keys()):
                    if key not in order_set:
                        del self.integrity_records[key]
                        removed = True
                        break
                if not removed:
                    # All keys are in order, evict the oldest
                    if self._integrity_order:
                        oldest_key = self._integrity_order[0]
                        self.integrity_records.pop(oldest_key, None)
                        self._integrity_order.popleft()
                    else:
                        # _integrity_order is empty but records exceed limit — force evict oldest by dict key
                        if self.integrity_records:
                            oldest_key = next(iter(self.integrity_records))
                            del self.integrity_records[oldest_key]
                        else:
                            break
            
            return True
        except Exception as e:
            logger.warning(f"Failed to protect file {path}: {e}")
            return False

    async def _check_file_integrity(self) -> None:
        """Check protected files for integrity violations."""
        for path_str, record in list(self.integrity_records.items()):
            path = Path(path_str)
            if not path.exists():
                if record.is_protected:
                    self._record_threat(
                        ThreatType.FILE_INTEGRITY,
                        ThreatLevel.HIGH,
                        "integrity_monitor",
                        f"Protected file deleted: {path_str}",
                        {"original_size": record.size, "original_hash": record.hash},
                        DefenseAction.ALERT,
                    )
                continue

            try:
                current_stat = path.stat()
                current_hash = hashlib.sha256(path.read_bytes()).hexdigest()

                if current_hash != record.hash:
                    self._record_threat(
                        ThreatType.FILE_INTEGRITY,
                        ThreatLevel.HIGH if record.is_protected else ThreatLevel.MEDIUM,
                        "integrity_monitor",
                        f"File modified: {path_str}",
                        {
                            "original_hash": record.hash,
                            "current_hash": current_hash,
                            "original_size": record.size,
                            "current_size": current_stat.st_size,
                        },
                        DefenseAction.ALERT,
                    )
                    record.hash = current_hash
                    record.size = current_stat.st_size
                    record.modified_time = current_stat.st_mtime
                    record.last_checked = datetime.now()

            except Exception as e:
                logger.warning(f"Integrity check failed for {path_str}: {e}")

    async def _check_system_behavior(self) -> None:
        """Monitor system behavior for anomalies."""
        try:
            import psutil
            
            # Use asyncio.to_thread to avoid blocking event loop
            def get_system_stats():
                cpu = psutil.cpu_percent(interval=1)
                mem = psutil.virtual_memory()
                pids = len(psutil.pids())
                return cpu, mem, pids
            
            cpu_percent, memory, process_count = await asyncio.to_thread(get_system_stats)

            self._update_baseline("cpu_percent", cpu_percent)
            self._update_baseline("memory_percent", memory.percent)

            for name, baseline in self.baselines.items():
                current = cpu_percent if name == "cpu_percent" else memory.percent
                if current > baseline.threshold_high:
                    self._record_threat(
                        ThreatType.RESOURCE_ABUSE,
                        ThreatLevel.MEDIUM,
                        "behavior_monitor",
                        f"High {name}: {current:.1f}% (threshold: {baseline.threshold_high:.1f}%)",
                        {"current": current, "threshold": baseline.threshold_high},
                        DefenseAction.ALERT,
                    )
                elif current < baseline.threshold_low and name == "memory_percent":
                    self._record_threat(
                        ThreatType.ANOMALOUS_BEHAVIOR,
                        ThreatLevel.LOW,
                        "behavior_monitor",
                        f"Unusually low {name}: {current:.1f}%",
                        {"current": current, "baseline": baseline.baseline_value},
                        DefenseAction.LOG,
                    )

            self._update_baseline("process_count", process_count)

            # Process iteration is quick, doesn't need to_thread
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    if proc.info['cpu_percent'] and proc.info['cpu_percent'] > 80:
                        self._update_baseline(f"process_cpu_{proc.info['name']}", proc.info['cpu_percent'])
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

        except ImportError:
            logger.debug("psutil not available for system monitoring")
        except Exception as e:
            logger.warning(f"System behavior check error: {e}")

    def _update_baseline(self, metric_name: str, value: float) -> None:
        """Update behavior baseline with new sample."""
        if metric_name in self.baselines:
            baseline = self.baselines[metric_name]
            new_samples = baseline.samples + 1

            if new_samples <= self.MAX_BASELINE_SAMPLES:
                old_mean = baseline.baseline_value
                baseline.baseline_value = old_mean + (value - old_mean) / new_samples
                baseline.std_deviation = math.sqrt(
                    ((new_samples - 2) / max(new_samples - 1, 1)) * baseline.std_deviation ** 2 +
                    (value - old_mean) ** 2 / max(new_samples, 1)
                ) if new_samples > 1 else 0
            else:
                alpha = 0.01
                baseline.baseline_value = alpha * value + (1 - alpha) * baseline.baseline_value
                baseline.std_deviation = abs(value - baseline.baseline_value) * math.sqrt(alpha) + baseline.std_deviation * math.sqrt(1 - alpha)

            baseline.threshold_high = baseline.baseline_value + 3 * baseline.std_deviation
            baseline.threshold_low = max(0, baseline.baseline_value - 3 * baseline.std_deviation)
            baseline.samples = new_samples
            baseline.last_updated = datetime.now()
            self._stats["baseline_updates"] += 1
        else:
            self.baselines[metric_name] = BehaviorBaseline(
                metric_name=metric_name,
                baseline_value=value,
                std_deviation=0,
                samples=1,
                last_updated=datetime.now(),
                threshold_high=value * 1.5,
                threshold_low=value * 0.5,
            )

    def _record_threat(
        self,
        threat_type: ThreatType,
        level: ThreatLevel,
        source: str,
        description: str,
        details: dict,
        action: DefenseAction,
    ) -> ThreatEvent:
        self._event_counter += 1
        event_id = f"threat_{self._event_counter:06d}"

        event = ThreatEvent(
            id=event_id,
            timestamp=datetime.now(),
            threat_type=threat_type,
            level=level,
            source=source,
            description=description,
            details=details,
            action_taken=action,
        )

        with self._lock:
            self.events.append(event)
            self._stats["threats_detected"] += 1

            if action == DefenseAction.BLOCK:
                self._stats["threats_blocked"] += 1
            elif action == DefenseAction.QUARANTINE:
                self._stats["threats_quarantined"] += 1

        logger.warning(f"Threat detected [{level.value}]: {description}")

        # Observe threat with Prefrontal Cortex
        try:
            pfc = get_prefrontal_cortex()
            if pfc and pfc._running:
                from core.helpers import create_tracked_task
                create_tracked_task(
                    pfc.observe(
                        content=f"Threat {threat_type.value}: {description[:200]}",
                        concept_type=ConceptType.OBSERVATION,
                        source="defense",
                        metadata={"level": level.value, "action": action.value, "source": source},
                    ),
                    name="pfc_observe_defense_threat",
                )
        except Exception as e:
            logger.debug(f"Failed to observe threat via PFC: {e}")

        if self.alert_callback and level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            from core.helpers import create_tracked_task
            create_tracked_task(self._send_alert(event), name="defense_send_alert")

        return event

    async def _send_alert(self, event: ThreatEvent) -> None:
        if self.alert_callback:
            try:
                await self.alert_callback(event)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

    async def _cleanup_quarantine(self) -> None:
        expired = [
            path for path, time in self.quarantined_files.items()
            if datetime.now() - time > timedelta(days=7)
        ]
        for path in expired:
            del self.quarantined_files[path]

    def block_entity(self, entity: str) -> None:
        self.blocked_entities.add(entity)
        self._save_state()
        logger.info(f"Blocked entity: {entity}")

    def unblock_entity(self, entity: str) -> bool:
        if entity in self.blocked_entities:
            self.blocked_entities.remove(entity)
            self._save_state()
            logger.info(f"Unblocked entity: {entity}")
            return True
        return False

    def quarantine_file(self, file_path: Path | str) -> bool:
        path = Path(file_path)
        if not path.exists():
            return False

        try:
            quarantine_dir = self.jarvis_root / "logs" / "quarantine"
            quarantine_dir.mkdir(parents=True, exist_ok=True)

            quarantine_path = quarantine_dir / f"{path.name}.{int(time.time())}"
            path.rename(quarantine_path)

            self.quarantined_files[str(quarantine_path)] = datetime.now()
            self._stats["threats_quarantined"] += 1

            logger.info(f"Quarantined file: {path} -> {quarantine_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to quarantine {path}: {e}")
            return False

    def restore_file(self, quarantine_name: str, restore_path: Path | str) -> bool:
        quarantine_dir = self.jarvis_root / "logs" / "quarantine"
        quarantine_file = quarantine_dir / quarantine_name

        if not quarantine_file.exists():
            return False

        try:
            quarantine_file.rename(restore_path)
            if str(quarantine_file) in self.quarantined_files:
                del self.quarantined_files[str(quarantine_file)]
            logger.info(f"Restored file: {quarantine_file} -> {restore_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to restore {quarantine_file}: {e}")
            return False

    def add_rule(self, rule: DefenseRule) -> None:
        self.rules[rule.id] = rule

    def remove_rule(self, rule_id: str) -> bool:
        if rule_id in self.rules:
            del self.rules[rule_id]
            return True
        return False

    def get_threat_summary(self, hours: int = 24) -> dict:
        cutoff = datetime.now() - timedelta(hours=hours)

        with self._lock:
            recent_events = [e for e in self.events if e.timestamp > cutoff]

        by_type: dict[str, int] = defaultdict(int)
        by_level: dict[str, int] = defaultdict(int)
        by_action: dict[str, int] = defaultdict(int)

        for event in recent_events:
            by_type[event.threat_type.value] += 1
            by_level[event.level.value] += 1
            by_action[event.action_taken.value] += 1

        return {
            "period_hours": hours,
            "total_threats": len(recent_events),
            "by_type": dict(by_type),
            "by_level": dict(by_level),
            "by_action": dict(by_action),
            "resolved_count": sum(1 for e in recent_events if e.resolved),
            "critical_count": by_level.get("critical", 0),
            "high_count": by_level.get("high", 0),
        }

    def get_defense_status(self) -> dict:
        return {
            "running": self._running,
            "protected_files": len(self.integrity_records),
            "blocked_entities": len(self.blocked_entities),
            "quarantined_files": len(self.quarantined_files),
            "rules_count": len(self.rules),
            "baselines": len(self.baselines),
            "stats": self._stats,
            "recent_threats": [
                {
                    "id": e.id,
                    "type": e.threat_type.value,
                    "level": e.level.value,
                    "description": e.description[:100],
                    "timestamp": e.timestamp.isoformat(),
                }
                for e in list(self.events)[-10:]
            ],
        }

    def get_recent_events(self, limit: int = 50) -> list[dict]:
        with self._lock:
            return [
                {
                    "id": e.id,
                    "timestamp": e.timestamp.isoformat(),
                    "threat_type": e.threat_type.value,
                    "level": e.level.value,
                    "source": e.source,
                    "description": e.description,
                    "action_taken": e.action_taken.value,
                    "resolved": e.resolved,
                }
                for e in list(self.events)[-limit:]
            ]

    def resolve_threat(self, event_id: str, resolution: str) -> bool:
        with self._lock:
            for event in self.events:
                if event.id == event_id and not event.resolved:
                    event.resolved = True
                    event.resolved_at = datetime.now()
                    event.resolution = resolution
                    self._stats["auto_resolved"] += 1
                    return True
        return False

    def protect_jarvis_core(self) -> dict:
        """Protect critical Jarvis files."""
        core_files = [
            "core/agent.py",
            "core/llm.py",
            "core/memory.py",
            "core/security/dlp.py",
            "core/security/audit.py",
            "run.py",
            "main.py",
            "config.yaml",
        ]

        results = {"protected": [], "failed": [], "skipped": []}

        for file_path in core_files:
            full_path = self.jarvis_root / file_path
            if full_path.exists():
                if self.protect_file(full_path):
                    results["protected"].append(file_path)
                else:
                    results["failed"].append(file_path)
            else:
                results["skipped"].append(file_path)

        self._save_state()
        return results

    def periodic_defense_review(self) -> dict:
        """Periodic review of defense logs for evolution insights.
        
        Called periodically (not on idle) to analyze security patterns
        and generate evolution guidance.
        
        Returns:
            dict with insights, patterns, and evolution suggestions
        """
        review = {
            "timestamp": datetime.now().isoformat(),
            "threats_analyzed": 0,
            "patterns_found": [],
            "evolution_suggestions": [],
            "security_posture": "healthy",
        }

        with self._lock:
            recent_events = list(self.events)[-100:]

        review["threats_analyzed"] = len(recent_events)

        if not recent_events:
            review["security_posture"] = "secure"
            return review

        by_type: dict[str, int] = defaultdict(int)
        by_level: dict[str, int] = defaultdict(int)
        by_source: dict[str, int] = defaultdict(int)

        for event in recent_events:
            by_type[event.threat_type.value] += 1
            by_level[event.level.value] += 1
            by_source[event.source] += 1

        high_critical = by_level.get("high", 0) + by_level.get("critical", 0)
        if high_critical > 5:
            review["security_posture"] = "concerning"
            review["evolution_suggestions"].append({
                "type": "increase_security_sensitivity",
                "reason": f"{high_critical} high/critical threats detected",
                "gene": "security_sensitivity",
                "delta": 0.1,
            })

        repeated_threats = [t for t, count in by_type.items() if count > 3]
        if repeated_threats:
            review["patterns_found"].append({
                "pattern": "repeated_threat_type",
                "types": repeated_threats,
            })
            review["evolution_suggestions"].append({
                "type": "adaptive defense",
                "reason": f"Repeated threats: {repeated_threats}",
            })

        common_sources = [(s, c) for s, c in by_source.items() if c > 2]
        if common_sources:
            review["patterns_found"].append({
                "pattern": "common_attack_sources",
                "sources": [s for s, _ in sorted(common_sources, key=lambda x: -x[1])[:5]],
            })

        resolved_rate = sum(1 for e in recent_events if e.resolved) / max(1, len(recent_events))
        if resolved_rate < 0.5:
            review["evolution_suggestions"].append({
                "type": "improve_response",
                "reason": f"Low resolution rate: {resolved_rate:.1%}",
            })

        try:
            from core.insight_bridge import get_insight_bridge
            bridge = get_insight_bridge()

            for suggestion in review["evolution_suggestions"]:
                if "gene" in suggestion:
                    bridge.process_insight(
                        raw_insight=f"Defense review: {suggestion['reason']}",
                        source_problem="periodic_defense_review",
                        source_category="security",
                        problem_solved=False,
                    )
        except Exception as e:
            logger.debug(f"Insight bridge not available for defense review: {e}")

        logger.info(f"Defense review complete: {review['threats_analyzed']} threats, posture={review['security_posture']}")

        return review


_defense_agent: DataDefenseAgent | None = None
_defense_lock = threading.Lock()


def get_defense_agent() -> DataDefenseAgent:
    """Get global defense agent instance (thread-safe)."""
    global _defense_agent
    if _defense_agent is None:
        with _defense_lock:
            if _defense_agent is None:
                _defense_agent = DataDefenseAgent()
    return _defense_agent
