"""Graceful Degradation Manager - Fallback behavior when subsystems fail.

When optional subsystems (Memory Router, Adaptive Context, Thought Sense,
Load Balancer, Drift Detector, etc.) become unavailable or error, this manager
provides typed fallback responses so the core agent loop never breaks.

Philosophy: The agent must always be able to respond, even if every
optional subsystem is down. This module provides the safety net.
"""
import logging
import threading
import time
from collections import OrderedDict
from enum import Enum, auto
from typing import Any, Callable

logger = logging.getLogger(__name__)


class DegradationLevel(Enum):
    FULL = auto()
    DEGRADED = auto()
    MINIMAL = auto()
    EMERGENCY = auto()


class SubsystemStatus(Enum):
    HEALTHY = auto()
    DEGRADED = auto()
    FAILED = auto()
    UNAVAILABLE = auto()


_DEGRADATION_FALLBACKS = {
    "memory_router": {
        "fallback_route": {"tier": "L2_WARM", "confidence": 0.5, "reason": "fallback"},
        "fallback_classify": "CONVERSATIONAL",
    },
    "adaptive_context": {
        "fallback_assemble": "",
        "fallback_strategy": "FOCUSED",
    },
    "thought_sense": {
        "fallback_state": "ENGAGED",
        "fallback_cognitive": "ENGAGED",
    },
    "load_balancer": {
        "fallback_should_run": True,
        "fallback_throttle": 0.0,
    },
    "drift_detector": {
        "fallback_topic": "",
        "fallback_drift": None,
    },
    "attention_queue": {
        "fallback_urgency": "NORMAL",
        "fallback_size": 0,
    },
    "consolidation": {
        "fallback_running": False,
    },
    "knowledge_graph": {
        "fallback_insights": [],
    },
    "vector_memory": {
        "fallback_search": [],
    },
    "prompt_sandbox": {
        "fallback_sanitize": None,
    },
}


class GracefulDegradationManager:
    _MAX_FAILURE_HISTORY = 200
    _MAX_REGISTERED = 30
    _FAILURE_COOLDOWN = 60.0

    def __init__(self):
        self._lock = threading.RLock()
        self._subsystem_status: dict[str, SubsystemStatus] = {}
        self._failure_counts: dict[str, int] = {}
        # Monotonic stamps — the cooldown decision must not shift with a
        # wall-clock step. (The failure *history* below stays wall-clock
        # because it is display-only.)
        self._last_failure_time: dict[str, float] = {}
        self._failure_history: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self._fallbacks: dict[str, dict] = dict(_DEGRADATION_FALLBACKS)
        self._custom_wrappers: dict[str, Callable] = {}
        self._degradation_level = DegradationLevel.FULL
        self._total_calls: int = 0
        self._fallback_calls: int = 0

    def register_subsystem(self, name: str, fallback_data: dict | None = None):
        with self._lock:
            if fallback_data:
                self._fallbacks[name] = fallback_data
            if name not in self._subsystem_status:
                self._subsystem_status[name] = SubsystemStatus.UNAVAILABLE

    def mark_healthy(self, name: str):
        with self._lock:
            self._subsystem_status[name] = SubsystemStatus.HEALTHY
            self._failure_counts[name] = 0

    def mark_degraded(self, name: str):
        with self._lock:
            self._subsystem_status[name] = SubsystemStatus.DEGRADED

    def mark_failed(self, name: str, error: str = ""):
        with self._lock:
            self._subsystem_status[name] = SubsystemStatus.FAILED
            self._failure_counts[name] = self._failure_counts.get(name, 0) + 1
            self._last_failure_time[name] = time.monotonic()
            wall = time.time()
            self._failure_history[f"{name}:{wall:.0f}"] = (wall, error[:100])
            while len(self._failure_history) > self._MAX_FAILURE_HISTORY:
                self._failure_history.popitem(last=False)
            logger.warning(f"Subsystem '{name}' marked as FAILED: {error[:80]}")

    def mark_unavailable(self, name: str):
        with self._lock:
            self._subsystem_status[name] = SubsystemStatus.UNAVAILABLE

    def get_status(self, name: str) -> SubsystemStatus:
        with self._lock:
            return self._subsystem_status.get(name, SubsystemStatus.UNAVAILABLE)

    def is_available(self, name: str) -> bool:
        status = self.get_status(name)
        return status in (SubsystemStatus.HEALTHY, SubsystemStatus.DEGRADED)

    def get_fallback(self, name: str, key: str, default: Any = None) -> Any:
        with self._lock:
            fallbacks = self._fallbacks.get(name, {})
            return fallbacks.get(key, default)

    def wrap_call(self, name: str, fn: Callable, fallback_key: str, *args, **kwargs) -> Any:
        with self._lock:
            self._total_calls += 1
            status = self._subsystem_status.get(name, SubsystemStatus.UNAVAILABLE)

        if status in (SubsystemStatus.FAILED, SubsystemStatus.UNAVAILABLE):
            cooldown_remaining = 0
            with self._lock:
                last_fail = self._last_failure_time.get(name, 0.0)
                cooldown_remaining = self._FAILURE_COOLDOWN - (
                    time.monotonic() - last_fail
                )
                failure_count = self._failure_counts.get(name, 0)

            if cooldown_remaining > 0 and failure_count >= 3:
                with self._lock:
                    self._fallback_calls += 1
                return self.get_fallback(name, fallback_key)

        try:
            result = fn(*args, **kwargs)
            self.mark_healthy(name)
            return result
        except Exception as e:
            self.mark_failed(name, str(e))
            with self._lock:
                self._fallback_calls += 1
            return self.get_fallback(name, fallback_key)

    def compute_degradation_level(self) -> DegradationLevel:
        with self._lock:
            total = len(self._subsystem_status)
            if total == 0:
                return DegradationLevel.FULL

            healthy = sum(1 for s in self._subsystem_status.values() if s == SubsystemStatus.HEALTHY)
            _degraded = sum(1 for s in self._subsystem_status.values() if s == SubsystemStatus.DEGRADED)
            failed = sum(1 for s in self._subsystem_status.values() if s == SubsystemStatus.FAILED)

            healthy_ratio = healthy / total
            failed_ratio = failed / total

            if failed_ratio > 0.5:
                level = DegradationLevel.EMERGENCY
            elif failed_ratio > 0.25:
                level = DegradationLevel.MINIMAL
            elif healthy_ratio < 0.5:
                level = DegradationLevel.DEGRADED
            else:
                level = DegradationLevel.FULL

            self._degradation_level = level
            return level

    def get_summary(self) -> dict:
        level = self.compute_degradation_level()
        with self._lock:
            return {
                "degradation_level": level.name,
                "total_subsystems": len(self._subsystem_status),
                "healthy": sum(1 for s in self._subsystem_status.values() if s == SubsystemStatus.HEALTHY),
                "degraded": sum(1 for s in self._subsystem_status.values() if s == SubsystemStatus.DEGRADED),
                "failed": sum(1 for s in self._subsystem_status.values() if s == SubsystemStatus.FAILED),
                "unavailable": sum(1 for s in self._subsystem_status.values() if s == SubsystemStatus.UNAVAILABLE),
                "total_calls": self._total_calls,
                "fallback_calls": self._fallback_calls,
                "fallback_rate": round(self._fallback_calls / max(1, self._total_calls), 3),
                "subsystem_details": {
                    name: status.name for name, status in self._subsystem_status.items()
                },
            }

    def reset(self):
        with self._lock:
            for name in self._subsystem_status:
                self._subsystem_status[name] = SubsystemStatus.UNAVAILABLE
            self._failure_counts.clear()
            self._last_failure_time.clear()
            self._degradation_level = DegradationLevel.FULL


_manager: GracefulDegradationManager | None = None
_manager_lock = threading.Lock()


def get_degradation_manager() -> GracefulDegradationManager:
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = GracefulDegradationManager()
                for name in _DEGRADATION_FALLBACKS:
                    _manager.register_subsystem(name)
    return _manager