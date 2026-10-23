"""Metrics and Observability System for Jarvis.

Provides:
- Performance metrics collection
- Subsystem health monitoring
- Alert management
- Metric aggregation and export
- Dashboard-ready data
"""
import asyncio
import logging
import threading
from collections import defaultdict
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from core.helpers import create_tracked_task, safe_error_message
from typing import Any, Callable

logger = logging.getLogger(__name__)


class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricPoint:
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    labels: dict[str, str] = field(default_factory=dict)


@dataclass
class Alert:
    id: str
    name: str
    severity: AlertSeverity
    message: str
    source: str
    fired_at: datetime
    resolved_at: datetime | None = None
    acknowledged: bool = False
    data: dict = field(default_factory=dict)


class Metric:
    """Base metric class."""

    MAX_POINTS = 1000

    def __init__(self, name: str, metric_type: MetricType, description: str = "", max_points: int = MAX_POINTS):
        self.name = name
        self.metric_type = metric_type
        self.description = description
        self._points: deque[MetricPoint] = deque(maxlen=max_points)
        self._lock = threading.Lock()

    def record(self, value: float, labels: dict[str, str] | None = None) -> None:
        point = MetricPoint(value=value, labels=labels or {})
        with self._lock:
            self._points.append(point)

    def get_points(self, since: datetime | None = None) -> list[MetricPoint]:
        with self._lock:
            if since:
                return [p for p in self._points if p.timestamp >= since]
            return list(self._points)


class Counter(Metric):
    """Monotonically increasing counter."""

    def __init__(self, name: str, description: str = "", max_points: int = Metric.MAX_POINTS):
        super().__init__(name, MetricType.COUNTER, description, max_points)
        self._value = 0.0

    def increment(self, amount: float = 1.0, labels: dict[str, str] | None = None) -> None:
        with self._lock:
            self._value += amount
            self._points.append(MetricPoint(value=self._value, labels=labels or {}))

    def get_value(self) -> float:
        with self._lock:
            return self._value


class Gauge(Metric):
    """Value that can go up or down."""

    def __init__(self, name: str, description: str = "", max_points: int = Metric.MAX_POINTS):
        super().__init__(name, MetricType.GAUGE, description, max_points)
        self._value = 0.0

    def set(self, value: float, labels: dict[str, str] | None = None) -> None:
        with self._lock:
            self._value = value
            self._points.append(MetricPoint(value=value, labels=labels or {}))

    def get_value(self) -> float:
        with self._lock:
            return self._value


class Histogram(Metric):
    """Distribution of values."""

    DEFAULT_BUCKETS = [0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0]

    def __init__(self, name: str, description: str = "", buckets: list[float] | None = None, max_points: int = Metric.MAX_POINTS):
        super().__init__(name, MetricType.HISTOGRAM, description, max_points)
        self._buckets = buckets or self.DEFAULT_BUCKETS
        self._bucket_counts: dict[float, int] = {b: 0 for b in self._buckets}
        self._bucket_counts[float('inf')] = 0
        self._sum = 0.0
        self._count = 0

    def observe(self, value: float, labels: dict[str, str] | None = None) -> None:
        with self._lock:
            self._sum += value
            self._count += 1

            for bucket in self._buckets:
                if value <= bucket:
                    self._bucket_counts[bucket] += 1
            self._bucket_counts[float('inf')] += 1

            self._points.append(MetricPoint(value=value, labels=labels or {}))

    def get_summary(self) -> dict:
        with self._lock:
            return {
                "count": self._count,
                "sum": self._sum,
                "mean": self._sum / self._count if self._count > 0 else 0,
                "buckets": dict(self._bucket_counts),
            }


class MetricsRegistry:
    """Central registry for all metrics."""

    MAX_POINTS_PER_METRIC = 1000
    METRIC_RETENTION_HOURS = 24

    def __init__(self, storage_path: Path | None = None):
        self.storage_path = storage_path or Path.home() / ".jarvis" / "metrics"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self._metrics: dict[str, Metric] = {}
        self._lock = threading.RLock()

        self._counters: dict[str, Counter] = {}
        self._gauges: dict[str, Gauge] = {}
        self._histograms: dict[str, Histogram] = {}

        self._subsystem_metrics: dict[str, dict[str, Metric]] = defaultdict(dict)

        self._register_default_metrics()

    def _register_default_metrics(self) -> None:
        """Register built-in metrics."""
        self.register_counter(
            "jarvis_requests_total",
            "Total number of requests processed",
        )
        self.register_counter(
            "jarvis_requests_errors",
            "Total number of request errors",
        )
        self.register_counter(
            "jarvis_tool_calls_total",
            "Total tool invocations",
        )
        self.register_counter(
            "jarvis_tool_errors_total",
            "Total tool execution errors",
        )
        self.register_gauge(
            "jarvis_memory_usage_bytes",
            "Current memory usage",
        )
        self.register_gauge(
            "jarvis_active_sessions",
            "Number of active sessions",
        )
        self.register_histogram(
            "jarvis_request_duration_seconds",
            "Request processing duration",
        )
        self.register_histogram(
            "jarvis_tool_duration_seconds",
            "Tool execution duration",
        )
        self.register_gauge(
            "jarvis_llm_tokens_used",
            "Total LLM tokens used",
        )
        self.register_counter(
            "jarvis_evolution_cycles_total",
            "Total evolution cycles completed",
        )
        self.register_gauge(
            "jarvis_math_problems_attempted",
            "Math problems attempted by Erdos engine",
        )
        self.register_gauge(
            "jarvis_math_problems_solved",
            "Math problems solved",
        )
        self.register_gauge(
            "jarvis_daydream_count",
            "Number of daydream sessions",
        )
        self.register_counter(
            "jarvis_threats_detected",
            "Security threats detected",
        )
        self.register_counter(
            "jarvis_defense_reviews",
            "Defense reviews completed",
        )
        self.register_counter(
            "jarvis_discoveries_shared",
            "Discoveries shared with user",
        )

    def register_counter(self, name: str, description: str = "") -> Counter:
        with self._lock:
            counter = Counter(name, description)
            self._metrics[name] = counter
            self._counters[name] = counter
            return counter

    def register_gauge(self, name: str, description: str = "") -> Gauge:
        with self._lock:
            gauge = Gauge(name, description)
            self._metrics[name] = gauge
            self._gauges[name] = gauge
            return gauge

    def register_histogram(self, name: str, description: str = "", buckets: list[float] | None = None) -> Histogram:
        with self._lock:
            histogram = Histogram(name, description, buckets)
            self._metrics[name] = histogram
            self._histograms[name] = histogram
            return histogram

    def counter(self, name: str) -> Counter | None:
        return self._counters.get(name)

    def gauge(self, name: str) -> Gauge | None:
        return self._gauges.get(name)

    def histogram(self, name: str) -> Histogram | None:
        return self._histograms.get(name)

    def increment_counter(self, name: str, amount: float = 1.0, labels: dict[str, str] | None = None) -> None:
        counter = self._counters.get(name)
        if counter:
            counter.increment(amount, labels)
        else:
            logger.warning(f"Counter not found: {name}")

    def set_gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        gauge = self._gauges.get(name)
        if gauge:
            gauge.set(value, labels)
        else:
            logger.warning(f"Gauge not found: {name}")

    def observe_histogram(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        histogram = self._histograms.get(name)
        if histogram:
            histogram.observe(value, labels)
        else:
            logger.warning(f"Histogram not found: {name}")

    def get_all_metrics(self) -> dict[str, Any]:
        """Get all metrics for export."""
        with self._lock:
            result = {}

            for name, counter in self._counters.items():
                result[name] = {
                    "type": "counter",
                    "value": counter.get_value(),
                    "description": counter.description,
                }

            for name, gauge in self._gauges.items():
                result[name] = {
                    "type": "gauge",
                    "value": gauge.get_value(),
                    "description": gauge.description,
                }

            for name, histogram in self._histograms.items():
                summary = histogram.get_summary()
                result[name] = {
                    "type": "histogram",
                    "summary": summary,
                    "description": histogram.description,
                }

            return result

    def get_metric_history(self, name: str, since: datetime | None = None) -> list[dict]:
        """Get history for a specific metric."""
        metric = self._metrics.get(name)
        if not metric:
            return []

        points = metric.get_points(since)
        return [
            {
                "value": p.value,
                "timestamp": p.timestamp.isoformat(),
                "labels": p.labels,
            }
            for p in points[-self.MAX_POINTS_PER_METRIC:]
        ]

    def register_subsystem_metrics(self, subsystem: str, metrics: dict[str, Metric]) -> None:
        """Register metrics for a specific subsystem."""
        with self._lock:
            self._subsystem_metrics[subsystem] = metrics

    def get_subsystem_metrics(self, subsystem: str) -> dict[str, Any]:
        """Get all metrics for a subsystem."""
        metrics = self._subsystem_metrics.get(subsystem, {})
        return {
            name: {
                "type": metric.metric_type.value,
                "description": metric.description,
            }
            for name, metric in metrics.items()
        }

    def cleanup_old_points(self) -> None:
        """Remove old metric points. Deque already handles max points, this removes by time."""
        cutoff = datetime.now() - timedelta(hours=self.METRIC_RETENTION_HOURS)

        with self._lock:
            for metric in self._metrics.values():
                with metric._lock:
                    while metric._points and metric._points[0].timestamp < cutoff:
                        metric._points.popleft()


class AlertManager:
    """Manages alerts and notifications."""

    MAX_ALERTS = 100
    ALERT_TTL = timedelta(hours=24)

    def __init__(self, storage_path: Path | None = None):
        self.storage_path = storage_path or Path.home() / ".jarvis" / "alerts"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self._alerts: deque[Alert] = deque(maxlen=AlertManager.MAX_ALERTS)
        self._alert_counter = 0
        self._lock = threading.RLock()

        self._alert_handlers: list[Callable] = []
        self._MAX_ALERT_HANDLERS = 20
        self._silenced_alerts: set[str] = set()
        self._MAX_SILENCED_ALERTS = 100

    def fire_alert(
        self,
        name: str,
        severity: AlertSeverity,
        message: str,
        source: str,
        data: dict | None = None,
    ) -> Alert:
        """Fire a new alert."""
        with self._lock:
            self._alert_counter += 1
            alert_id = f"alert_{self._alert_counter:04d}"

            alert = Alert(
                id=alert_id,
                name=name,
                severity=severity,
                message=message,
                source=source,
                fired_at=datetime.now(),
                data=data or {},
            )

            self._alerts.append(alert)

            if name not in self._silenced_alerts:
                for handler in self._alert_handlers:
                    try:
                        handler(alert)
                    except Exception as e:
                        logger.error(f"Alert handler error: {e}")

            logger.log(
                self._severity_to_log_level(severity),
                f"Alert [{name}] from {source}: {message}"
            )

            return alert

    def _severity_to_log_level(self, severity: AlertSeverity) -> int:
        mapping = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.ERROR: logging.ERROR,
            AlertSeverity.CRITICAL: logging.CRITICAL,
        }
        return mapping.get(severity, logging.INFO)

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        with self._lock:
            for alert in self._alerts:
                if alert.id == alert_id and not alert.resolved_at:
                    alert.resolved_at = datetime.now()
                    return True
            return False

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        with self._lock:
            for alert in self._alerts:
                if alert.id == alert_id:
                    alert.acknowledged = True
                    return True
            return False

    def unsilence_alert(self, name: str) -> None:
        """Unsilence alerts by name."""
        with self._lock:
            self._silenced_alerts.discard(name)

    def get_active_alerts(self) -> list[dict]:
        """Get all active (unresolved) alerts."""
        with self._lock:
            return [
                {
                    "id": alert.id,
                    "name": alert.name,
                    "severity": alert.severity.value,
                    "message": alert.message,
                    "source": alert.source,
                    "fired_at": alert.fired_at.isoformat(),
                    "acknowledged": alert.acknowledged,
                }
                for alert in self._alerts
                if not alert.resolved_at
            ]

    def get_all_alerts(self, limit: int = 50) -> list[dict]:
        """Get recent alerts."""
        with self._lock:
            return [
                {
                    "id": alert.id,
                    "name": alert.name,
                    "severity": alert.severity.value,
                    "message": alert.message,
                    "source": alert.source,
                    "fired_at": alert.fired_at.isoformat(),
                    "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None,
                    "acknowledged": alert.acknowledged,
                }
                for alert in list(self._alerts)[-limit:]
            ]

    def cleanup_old_alerts(self) -> int:
        """Remove alerts older than TTL. Returns count of removed alerts."""
        cutoff = datetime.now() - self.ALERT_TTL
        removed = 0
        with self._lock:
            original_len = len(self._alerts)
            self._alerts = deque(
                (alert for alert in self._alerts
                 if alert.resolved_at is None or alert.resolved_at >= cutoff),
                maxlen=AlertManager.MAX_ALERTS,
            )
            removed = original_len - len(self._alerts)
        return removed

    def register_handler(self, handler: Callable) -> None:
        """Register an alert handler."""
        with self._lock:
            self._alert_handlers.append(handler)
            if len(self._alert_handlers) > self._MAX_ALERT_HANDLERS:
                self._alert_handlers = self._alert_handlers[-self._MAX_ALERT_HANDLERS:]

    def silence_alert(self, name: str) -> None:
        """Silence alerts by name."""
        with self._lock:
            self._silenced_alerts.add(name)
            if len(self._silenced_alerts) > self._MAX_SILENCED_ALERTS:
                excess = list(self._silenced_alerts)[:len(self._silenced_alerts) - self._MAX_SILENCED_ALERTS]
                for item in excess:
                    self._silenced_alerts.discard(item)


class ObservabilitySystem:
    """Unified observability system combining metrics and alerts."""

    HEALTH_CHECK_INTERVAL = 60.0

    def __init__(self, storage_path: Path | None = None):
        base_path = storage_path or Path.home() / ".jarvis"

        self.metrics = MetricsRegistry(base_path / "metrics")
        self.alerts = AlertManager(base_path / "alerts")

        self._running = False
        self._health_task: asyncio.Task | None = None
        self._health_checks: dict[str, Callable] = {}
        self._lock = threading.RLock()

        self._component_status: dict[str, dict] = {}

    def register_health_check(self, component: str, check_func: Callable) -> None:
        """Register a health check function for a component."""
        with self._lock:
            self._health_checks[component] = check_func

    def update_component_status(self, component: str, status: str, details: dict | None = None) -> None:
        """Update status for a component."""
        with self._lock:
            self._component_status[component] = {
                "status": status,
                "updated_at": datetime.now().isoformat(),
                "details": details or {},
            }

    async def start(self) -> None:
        """Start the observability system."""
        if self._running:
            return

        self._running = True
        self._health_task = create_tracked_task(self._health_check_loop(), name="observability_health_check")
        logger.info("Observability system started")

    async def stop(self) -> None:
        """Stop the observability system."""
        self._running = False
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
        logger.info("Observability system stopped")

    async def _health_check_loop(self) -> None:
        """Periodic health check for all components."""
        while self._running:
            try:
                await asyncio.sleep(self.HEALTH_CHECK_INTERVAL)

                for component, check_func in self._health_checks.items():
                    try:
                        if asyncio.iscoroutinefunction(check_func):
                            healthy, details = await check_func()
                        else:
                            healthy, details = check_func()

                        status = "healthy" if healthy else "unhealthy"
                        self.update_component_status(component, status, details)

                        if not healthy:
                            self.alerts.fire_alert(
                                name=f"{component}_unhealthy",
                                severity=AlertSeverity.WARNING,
                                message=f"Component {component} is unhealthy",
                                source="observability",
                                data=details,
                            )

                    except Exception as e:
                        self.update_component_status(component, "error", {"error": safe_error_message(e)})
                        self.alerts.fire_alert(
                            name=f"{component}_check_error",
                            severity=AlertSeverity.ERROR,
                            message=f"Health check failed for {component}: {e}",
                            source="observability",
                        )

                self.metrics.cleanup_old_points()
                self.alerts.cleanup_old_alerts()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error: {e}")

    def get_dashboard_data(self) -> dict:
        """Get all data needed for a dashboard."""
        return {
            "metrics": self.metrics.get_all_metrics(),
            "alerts": self.alerts.get_active_alerts(),
            "components": self._component_status,
            "timestamp": datetime.now().isoformat(),
        }

    def export_metrics_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []

        for name, metric in self.metrics.get_all_metrics().items():
            metric_type = metric.get("type", "gauge")
            description = metric.get("description", "")
            value = metric.get("value", 0)

            lines.append(f"# HELP {name} {description}")
            lines.append(f"# TYPE {name} {metric_type}")
            lines.append(f"{name} {value}")

        return "\n".join(lines)


_observability: ObservabilitySystem | None = None
_observability_lock = threading.Lock()


def get_observability() -> ObservabilitySystem:
    """Get global observability system instance (thread-safe)."""
    global _observability
    if _observability is None:
        with _observability_lock:
            if _observability is None:
                _observability = ObservabilitySystem()
    return _observability


def get_metrics() -> MetricsRegistry:
    """Get the metrics registry."""
    return get_observability().metrics


def get_alerts() -> AlertManager:
    """Get the alert manager."""
    return get_observability().alerts
