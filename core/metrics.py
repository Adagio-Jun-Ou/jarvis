"""Metrics collection for bounded collections and memory monitoring.

Provides real-time monitoring of collection sizes, memory usage,
and health metrics for all bounded collections in Jarvis.
"""
import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CollectionMetrics:
    """Metrics for a single bounded collection."""
    name: str
    current_size: int
    max_size: int
    utilization: float  # 0.0 to 1.0
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "current_size": self.current_size,
            "max_size": self.max_size,
            "utilization": round(self.utilization, 3),
            "last_updated": self.last_updated.isoformat(),
        }


class MetricsCollector:
    """Collects and reports metrics for bounded collections.
    
    Tracks:
    - Collection sizes (learning_pipeline sessions, defense integrity_records, etc.)
    - Memory bounds checking
    - Utilization percentages
    - Warning thresholds
    
    Usage:
        metrics = get_metrics_collector()
        metrics.register("learning_pipeline.sessions", sessions_deque, 100)
        metrics.register("defense.integrity_records", integrity_dict, 500)
        
        # Get alerts
        alerts = metrics.check_thresholds()
        
        # Get all metrics
        all_metrics = metrics.get_all_metrics()
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __init__(self, warning_threshold: float = 0.8, critical_threshold: float = 0.95):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self._collections: dict[str, tuple[Any, int]] = {}  # name -> (collection, max_size)
        self._size_functions: dict[str, callable] = {}  # name -> size_function
        self._last_check: datetime | None = None
        
    @classmethod
    def get_instance(cls) -> "MetricsCollector":
        """Get the singleton metrics collector."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def register(self, name: str, collection: Any, max_size: int, size_func: callable = None) -> None:
        """Register a collection for monitoring.
        
        Args:
            name: Unique identifier for this collection
            collection: The collection object (list, dict, deque, etc.)
            max_size: Maximum allowed size
            size_func: Optional function to get size (default: len())
        """
        self._collections[name] = (collection, max_size)
        if size_func:
            self._size_functions[name] = size_func
        logger.debug(f"Registered metrics for {name} (max: {max_size})")
    
    def get_size(self, name: str) -> int:
        """Get current size of a registered collection."""
        if name not in self._collections:
            return 0
        
        collection, _ = self._collections[name]
        
        if name in self._size_functions:
            return self._size_functions[name](collection)
        
        # Default size functions for common types
        if isinstance(collection, deque):
            return len(collection)
        elif isinstance(collection, dict):
            return len(collection)
        elif isinstance(collection, list):
            return len(collection)
        elif hasattr(collection, '__len__'):
            return len(collection)
        
        return 0
    
    def get_metrics(self, name: str) -> CollectionMetrics | None:
        """Get metrics for a specific collection."""
        if name not in self._collections:
            return None
        
        _, max_size = self._collections[name]
        current_size = self.get_size(name)
        utilization = current_size / max_size if max_size > 0 else 0.0
        
        return CollectionMetrics(
            name=name,
            current_size=current_size,
            max_size=max_size,
            utilization=utilization,
        )
    
    def get_all_metrics(self) -> dict[str, CollectionMetrics]:
        """Get metrics for all registered collections."""
        return {name: self.get_metrics(name) for name in self._collections}
    
    def check_thresholds(self) -> list[dict]:
        """Check all collections against thresholds.
        
        Returns:
            List of alerts for collections exceeding thresholds
        """
        alerts = []
        
        for name in self._collections:
            metrics = self.get_metrics(name)
            if metrics is None:
                continue
            
            if metrics.utilization >= self.critical_threshold:
                alerts.append({
                    "level": "critical",
                    "name": name,
                    "message": f"{name} at {metrics.utilization*100:.1f}% capacity ({metrics.current_size}/{metrics.max_size})",
                    "utilization": metrics.utilization,
                })
            elif metrics.utilization >= self.warning_threshold:
                alerts.append({
                    "level": "warning",
                    "name": name,
                    "message": f"{name} at {metrics.utilization*100:.1f}% capacity ({metrics.current_size}/{metrics.max_size})",
                    "utilization": metrics.utilization,
                })
        
        self._last_check = datetime.now()
        return alerts
    
    def get_summary(self) -> dict[str, Any]:
        """Get a summary of all collection states."""
        metrics = self.get_all_metrics()
        
        if not metrics:
            return {
                "total_collections": 0,
                "total_items": 0,
                "total_capacity": 0,
                "overall_utilization": 0.0,
                "warnings": 0,
                "critical": 0,
            }
        
        total_items = sum(m.current_size for m in metrics.values())
        total_capacity = sum(m.max_size for m in metrics.values())
        overall_utilization = total_items / total_capacity if total_capacity > 0 else 0.0
        
        alerts = self.check_thresholds()
        warnings = sum(1 for a in alerts if a["level"] == "warning")
        critical = sum(1 for a in alerts if a["level"] == "critical")
        
        return {
            "total_collections": len(metrics),
            "total_items": total_items,
            "total_capacity": total_capacity,
            "overall_utilization": round(overall_utilization, 3),
            "warnings": warnings,
            "critical": critical,
            "last_check": self._last_check.isoformat() if self._last_check else None,
            "collections": {name: m.to_dict() for name, m in metrics.items()},
        }


def get_metrics_collector() -> MetricsCollector:
    """Get the singleton metrics collector."""
    return MetricsCollector.get_instance()


# Convenience function for health checks
def check_collection_health() -> dict[str, Any]:
    """Check health of all registered collections.
    
    This can be used as a health check callback:
    
        health_manager.register("metrics", check_collection_health)
    """
    from core.health import HealthCheckResult
    
    metrics = get_metrics_collector()
    alerts = metrics.check_thresholds()
    summary = metrics.get_summary()
    
    has_critical = any(a["level"] == "critical" for a in alerts)
    has_warning = any(a["level"] == "warning" for a in alerts)
    
    if has_critical:
        return HealthCheckResult(
            healthy=False,
            message=f"{summary['critical']} collections at critical capacity",
            metadata=summary,
        )
    elif has_warning:
        return HealthCheckResult(
            healthy=True,
            message=f"{summary['warnings']} collections approaching capacity",
            metadata=summary,
        )
    else:
        return HealthCheckResult(
            healthy=True,
            message=f"All {summary['total_collections']} collections healthy",
            metadata=summary,
        )