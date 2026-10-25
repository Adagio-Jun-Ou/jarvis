"""Cross-Module Telemetry Aggregator - Unified stats across all Jarvis subsystems.

Collects, normalizes, and aggregates telemetry from:
- Memory (ConversationMemory, VectorMemory, ThreeLayer, Consolidation)
- Cognition (AttentionQueue, DriftDetector, ThoughtSense, LoadBalancer)
- Security (LCMShield, PromptSandbox, BranchDetection, MITM, StackDefense, DLP)
- LLM (OllamaClient, DualModel, TokenOptimizer)
- Evolution (EvolutionOrchestrator, DaydreamEngine, ErdosEngine)
- Infrastructure (Supervisor, HealthManager, Observability, Metrics)
- Context (AdaptiveWindow, MemoryRouter, ContextGuardian)

Provides:
- Unified /api/telemetry endpoint data
- Per-subsystem health scores (0-100)
- Cross-module correlation detection
- Anomaly detection across subsystem metrics
- Rolling time-window aggregation
"""
import logging
import threading
import time
from core.helpers import safe_error_message
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable

logger = logging.getLogger(__name__)


class SubsystemCategory(Enum):
    MEMORY = "memory"
    COGNITION = "cognition"
    SECURITY = "security"
    LLM = "llm"
    EVOLUTION = "evolution"
    INFRASTRUCTURE = "infrastructure"
    CONTEXT = "context"


class HealthScore(Enum):
    EXCELLENT = auto()
    GOOD = auto()
    DEGRADED = auto()
    POOR = auto()
    CRITICAL = auto()


@dataclass
class SubsystemTelemetry:
    name: str
    category: SubsystemCategory
    health_score: float
    health_label: str
    metrics: dict[str, Any]
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "category": self.category.value,
            "health_score": round(self.health_score, 1),
            "health_label": self.health_label,
            "metrics": self.metrics,
            "timestamp": self.timestamp,
        }


@dataclass
class AnomalyEvent:
    subsystem: str
    metric_name: str
    expected: float
    actual: float
    deviation: float
    severity: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "subsystem": self.subsystem,
            "metric": self.metric_name,
            "expected": round(self.expected, 2),
            "actual": round(self.actual, 2),
            "deviation": round(self.deviation, 2),
            "severity": self.severity,
            "timestamp": self.timestamp,
        }


class TelemetryAggregator:
    _MAX_HISTORY = 100
    _MAX_ANOMALIES = 50
    _MAX_COLLECTORS = 30
    _ANOMALY_THRESHOLD = 2.0

    def __init__(self):
        self._lock = threading.RLock()
        self._collectors: dict[str, tuple[SubsystemCategory, Callable[[], dict]]] = {}
        self._last_snapshot: dict[str, SubsystemTelemetry] = {}
        self._history: deque[dict[str, SubsystemTelemetry]] = deque(maxlen=self._MAX_HISTORY)
        self._anomalies: deque[AnomalyEvent] = deque(maxlen=self._MAX_ANOMALIES)
        self._baseline: dict[str, dict[str, float]] = {}
        self._collection_count: int = 0
        self._last_collect_time: float = 0
        self._collect_interval: float = 30.0

    def register(
        self,
        name: str,
        category: SubsystemCategory,
        collector: Callable[[], dict],
    ) -> bool:
        with self._lock:
            if len(self._collectors) >= self._MAX_COLLECTORS and name not in self._collectors:
                oldest = next(iter(self._collectors))
                del self._collectors[oldest]
            self._collectors[name] = (category, collector)
        return True

    def unregister(self, name: str) -> bool:
        with self._lock:
            if name in self._collectors:
                del self._collectors[name]
                self._last_snapshot.pop(name, None)
                self._baseline.pop(name, None)
                return True
            return False

    def _compute_health(self, metrics: dict) -> tuple[float, str]:
        if not metrics:
            return 50.0, "DEGRADED"

        scores = []

        if "errors" in metrics and "total" in metrics:
            total = max(1, metrics["total"])
            error_rate = metrics["errors"] / total
            scores.append(100 * (1 - error_rate))

        if "success_rate" in metrics:
            scores.append(metrics["success_rate"] * 100)

        if "cache_hits" in metrics and "total_routes" in metrics:
            total = max(1, metrics["total_routes"])
            hit_rate = metrics["cache_hits"] / total
            scores.append(hit_rate * 100)

        if "blocked" in metrics and "total_sandboxed" in metrics:
            total = max(1, metrics["total_sandboxed"])
            block_rate = metrics["blocked"] / total
            scores.append(max(0, 100 - block_rate * 200))

        if "active_goals" in metrics:
            scores.append(min(100, metrics["active_goals"] * 10))

        if "running" in metrics:
            scores.append(100 if metrics["running"] else 0)

        if "health" in metrics:
            scores.append(metrics["health"] * 100)

        if "utilization" in metrics:
            util = metrics["utilization"]
            if util < 0.8:
                scores.append(100)
            elif util < 0.95:
                scores.append(50)
            else:
                scores.append(10)

        if not scores:
            avg = 75.0
        else:
            avg = sum(scores) / len(scores)

        avg = max(0.0, min(100.0, avg))

        if avg >= 90:
            label = "EXCELLENT"
        elif avg >= 70:
            label = "GOOD"
        elif avg >= 50:
            label = "DEGRADED"
        elif avg >= 25:
            label = "POOR"
        else:
            label = "CRITICAL"

        return avg, label

    def _detect_anomalies(self, name: str, metrics: dict):
        if name not in self._baseline:
            return

        baseline = self._baseline[name]
        for key, expected in baseline.items():
            if key not in metrics:
                continue
            actual = metrics[key]
            if expected == 0:
                continue
            deviation = abs(actual - expected) / abs(expected)
            if deviation > self._ANOMALY_THRESHOLD:
                severity = "warning" if deviation < 3.0 else "critical"
                self._anomalies.append(AnomalyEvent(
                    subsystem=name,
                    metric_name=key,
                    expected=expected,
                    actual=actual,
                    deviation=deviation,
                    severity=severity,
                ))

    def _update_baseline(self, name: str, metrics: dict):
        numeric_keys = {k: v for k, v in metrics.items() if isinstance(v, (int, float)) and v >= 0}
        if name not in self._baseline:
            self._baseline[name] = numeric_keys
        else:
            alpha = 0.1
            for k, v in numeric_keys.items():
                if k in self._baseline[name]:
                    self._baseline[name][k] = (1 - alpha) * self._baseline[name][k] + alpha * v
                else:
                    self._baseline[name][k] = v

    def collect(self) -> dict[str, SubsystemTelemetry]:
        with self._lock:
            collectors = dict(self._collectors)

        snapshot = {}
        for name, (category, collector) in collectors.items():
            try:
                metrics = collector()
            except Exception as e:
                logger.debug(f"Telemetry collection failed for {name}: {e}")
                metrics = {"error": safe_error_message(e)[:100]}

            health, label = self._compute_health(metrics)
            telemetry = SubsystemTelemetry(
                name=name,
                category=category,
                health_score=health,
                health_label=label,
                metrics=metrics,
            )
            snapshot[name] = telemetry

            with self._lock:
                self._detect_anomalies(name, metrics)
                self._update_baseline(name, metrics)

        with self._lock:
            self._last_snapshot = snapshot
            self._history.append(snapshot)
            self._collection_count += 1
            self._last_collect_time = time.time()

        return snapshot

    def get_snapshot(self) -> dict[str, SubsystemTelemetry]:
        with self._lock:
            if not self._last_snapshot or (time.time() - self._last_collect_time > self._collect_interval):
                pass
            return dict(self._last_snapshot)

    def get_summary(self) -> dict:
        with self._lock:
            snapshot = dict(self._last_snapshot)
            anomalies = list(self._anomalies)
            collection_count = self._collection_count

        if not snapshot:
            return {"status": "no_data", "subsystems": 0}

        category_health = {}
        for name, tel in snapshot.items():
            cat = tel.category.value
            if cat not in category_health:
                category_health[cat] = []
            category_health[cat].append(tel.health_score)

        category_scores = {}
        overall_scores = []
        for cat, scores in category_health.items():
            avg = sum(scores) / len(scores) if scores else 0
            category_scores[cat] = round(avg, 1)
            overall_scores.extend(scores)

        overall = round(sum(overall_scores) / len(overall_scores), 1) if overall_scores else 0

        recent_anomalies = [a.to_dict() for a in anomalies[-10:]]

        return {
            "overall_health": overall,
            "overall_label": self._score_to_label(overall),
            "category_health": category_scores,
            "subsystems": {name: tel.to_dict() for name, tel in snapshot.items()},
            "subsystem_count": len(snapshot),
            "collection_count": collection_count,
            "recent_anomalies": recent_anomalies,
            "anomaly_count": len(anomalies),
        }

    def _score_to_label(self, score: float) -> str:
        if score >= 90:
            return "EXCELLENT"
        elif score >= 70:
            return "GOOD"
        elif score >= 50:
            return "DEGRADED"
        elif score >= 25:
            return "POOR"
        return "CRITICAL"

    def get_history(self, limit: int = 10) -> list[dict]:
        with self._lock:
            return [
                {name: tel.to_dict() for name, tel in snap.items()}
                for snap in list(self._history)[-limit:]
            ]

    def get_anomalies(self, limit: int = 20) -> list[dict]:
        with self._lock:
            return [a.to_dict() for a in list(self._anomalies)[-limit:]]

    def force_collect(self) -> dict:
        return {name: tel.to_dict() for name, tel in self.collect().items()}


_telemetry: TelemetryAggregator | None = None
_telemetry_lock = threading.Lock()


def get_telemetry_aggregator() -> TelemetryAggregator:
    global _telemetry
    if _telemetry is None:
        with _telemetry_lock:
            if _telemetry is None:
                _telemetry = TelemetryAggregator()
    return _telemetry


def register_standard_collectors() -> None:
    aggregator = get_telemetry_aggregator()

    def _safe_get(getter_fn, default=None):
        try:
            result = getter_fn()
            return result if result is not None else default
        except Exception as e:
            logger.debug(f"Telemetry collector error: {e}")
            return default

    try:
        from core.memory_consolidation import get_consolidation_engine
        engine = get_consolidation_engine()
        aggregator.register("memory_consolidation", SubsystemCategory.MEMORY, lambda: {
            "consolidation_count": engine._consolidation_count,
            "running": engine._running,
            "reports": len(engine._reports) if hasattr(engine, '_reports') else 0,
        })
    except Exception as e:
        logger.debug(f"Telemetry: memory_consolidation unavailable: {e}")

    try:
        from core.attention_priority import get_attention_queue
        aq = get_attention_queue()
        aggregator.register("attention_queue", SubsystemCategory.COGNITION, lambda: {
            "size": _safe_get(lambda: aq.size(), 0),
            "total_enqueued": _safe_get(lambda: aq.get_stats().get("total_enqueued", 0), 0),
        })
    except Exception as e:
        logger.debug(f"Telemetry: attention_queue unavailable: {e}")

    try:
        from core.concept_drift import get_drift_detector
        dd = get_drift_detector()
        aggregator.register("concept_drift", SubsystemCategory.COGNITION, lambda: {
            "total_observations": _safe_get(lambda: dd.get_stats().get("total_observations", 0), 0),
            "active_alerts": _safe_get(lambda: dd.get_stats().get("active_alerts", 0), 0),
        })
    except Exception as e:
        logger.debug(f"Telemetry: concept_drift unavailable: {e}")

    try:
        from core.cognitive_load_balancer import get_load_balancer
        lb = get_load_balancer()
        aggregator.register("load_balancer", SubsystemCategory.COGNITION, lambda: {
            **_safe_get(lambda: lb.get_stats(), {}),
        })
    except Exception as e:
        logger.debug(f"Telemetry: load_balancer unavailable: {e}")

    try:
        from core.prompt_sandbox import get_prompt_sandbox
        sb = get_prompt_sandbox()
        aggregator.register("prompt_sandbox", SubsystemCategory.SECURITY, lambda: {
            **_safe_get(lambda: sb.get_stats(), {}),
        })
    except Exception as e:
        logger.debug(f"Telemetry: prompt_sandbox unavailable: {e}")

    try:
        from core.memory_router import get_memory_router
        mr = get_memory_router()
        aggregator.register("memory_router", SubsystemCategory.CONTEXT, lambda: {
            **_safe_get(lambda: mr.get_stats(), {}),
        })
    except Exception as e:
        logger.debug(f"Telemetry: memory_router unavailable: {e}")

    try:
        from core.adaptive_context import get_adaptive_window
        aw = get_adaptive_window()
        aggregator.register("adaptive_context", SubsystemCategory.CONTEXT, lambda: {
            **_safe_get(lambda: aw.get_stats(), {}),
        })
    except Exception as e:
        logger.debug(f"Telemetry: adaptive_context unavailable: {e}")

    try:
        from core.self_reflection import get_self_reflection
        sr = get_self_reflection()
        aggregator.register("self_reflection", SubsystemCategory.COGNITION, lambda: {
            "avg_quality": _safe_get(lambda: sr.get_average_quality(), 0.5),
        })
    except Exception as e:
        logger.debug(f"Telemetry: self_reflection unavailable: {e}")

    try:
        from core.supervisor import get_supervisor
        sup = get_supervisor()
        aggregator.register("supervisor", SubsystemCategory.INFRASTRUCTURE, lambda: {
            **_safe_get(lambda: sup.get_status().get("stats", {}), {}),
        })
    except Exception as e:
        logger.debug(f"Telemetry: supervisor unavailable: {e}")

    logger.info(f"Registered {len(aggregator._collectors)} telemetry collectors")