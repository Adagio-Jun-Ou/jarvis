"""Health Router — intelligent request routing based on subsystem health.

Provides:
- HealthRouter: routes requests to healthy backends, falls back on degradation
- Backend health tracking with configurable thresholds
- Weighted random selection based on health scores
- Automatic circuit-breaking integration
- Event bus notifications on route changes
"""
import logging
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class Backend:
    name: str
    health_score: float = 1.0
    last_checked: float = field(default_factory=time.time)
    total_requests: int = 0
    total_errors: int = 0
    consecutive_errors: int = 0
    consecutive_successes: int = 0
    last_error_time: float = 0.0
    avg_latency: float = 0.0
    circuit_open: bool = False
    circuit_open_until: float = 0.0


@dataclass
class RouteDecision:
    backend: str
    health_score: float
    reason: str = ""
    fallback: bool = False
    latency_estimate: float = 0.0


class HealthRouter:
    """Routes requests based on backend health scores.

    Features:
    - Weighted random selection proportional to health
    - Circuit breaker per backend (opens after consecutive errors)
    - Half-open recovery (tries one request after cooldown)
    - Health check callbacks for live monitoring
    - Event bus integration for route change notifications
    """

    def __init__(
        self,
        error_threshold: int = 5,
        recovery_threshold: int = 3,
        circuit_cooldown: float = 30.0,
        min_health: float = 0.1,
        health_check_interval: float = 60.0,
        max_backends: int = 50,
    ):
        self._error_threshold = error_threshold
        self._recovery_threshold = recovery_threshold
        self._circuit_cooldown = circuit_cooldown
        self._min_health = min_health
        self._health_check_interval = health_check_interval
        self._max_backends = max_backends
        self._lock = threading.RLock()
        self._backends: dict[str, Backend] = {}
        self._health_checks: dict[str, Callable[[], float]] = {}
        self._priority_backends: list[str] = []
        self._stats = {
            "total_routes": 0,
            "total_fallbacks": 0,
            "total_circuit_opens": 0,
        }

    def register_backend(
        self,
        name: str,
        health_fn: Callable[[], float] | None = None,
        priority: bool = False,
    ) -> None:
        with self._lock:
            if len(self._backends) >= self._max_backends and name not in self._backends:
                oldest = min(self._backends, key=lambda k: self._backends[k].last_checked)
                del self._backends[oldest]
            self._backends[name] = Backend(name=name)
            if health_fn:
                self._health_checks[name] = health_fn
            if priority:
                if name not in self._priority_backends:
                    self._priority_backends.append(name)

    def unregister_backend(self, name: str) -> None:
        with self._lock:
            self._backends.pop(name, None)
            self._health_checks.pop(name, None)
            if name in self._priority_backends:
                self._priority_backends.remove(name)

    def route(self, required_backends: list[str] | None = None) -> RouteDecision:
        with self._lock:
            self._stats["total_routes"] += 1
            self._update_circuits_locked()

            candidates = required_backends or list(self._backends.keys())
            healthy = []
            for name in candidates:
                be = self._backends.get(name)
                if be is None:
                    continue
                if be.circuit_open:
                    if time.time() < be.circuit_open_until:
                        continue
                    else:
                        be.circuit_open = False
                        be.consecutive_errors = 0
                if be.health_score >= self._min_health:
                    healthy.append(be)

            if not healthy:
                fallback_candidates = [
                    self._backends[n] for n in candidates if n in self._backends
                ]
                if fallback_candidates:
                    be = max(fallback_candidates, key=lambda b: b.health_score)
                    self._stats["total_fallbacks"] += 1
                    return RouteDecision(
                        backend=be.name,
                        health_score=be.health_score,
                        reason="fallback_no_healthy",
                        fallback=True,
                        latency_estimate=be.avg_latency,
                    )
                return RouteDecision(
                    backend="none",
                    health_score=0.0,
                    reason="no_backends",
                    fallback=True,
                )

            priority_healthy = [h for h in healthy if h.name in self._priority_backends]
            if priority_healthy:
                pool = priority_healthy
            else:
                pool = healthy

            total_health = sum(b.health_score for b in pool)
            if total_health <= 0:
                be = pool[0]
            else:
                r = random.random() * total_health
                cumulative = 0.0
                be = pool[0]
                for b in pool:
                    cumulative += b.health_score
                    if r <= cumulative:
                        be = b
                        break

            return RouteDecision(
                backend=be.name,
                health_score=be.health_score,
                reason="weighted_health",
                latency_estimate=be.avg_latency,
            )

    def report_success(self, backend_name: str, latency: float = 0.0) -> None:
        with self._lock:
            be = self._backends.get(backend_name)
            if be is None:
                return
            be.total_requests += 1
            be.consecutive_errors = 0
            be.consecutive_successes += 1
            if latency > 0:
                if be.avg_latency == 0:
                    be.avg_latency = latency
                else:
                    be.avg_latency = 0.7 * be.avg_latency + 0.3 * latency
            be.health_score = min(1.0, be.health_score + 0.1)
            if be.consecutive_successes >= self._recovery_threshold and be.circuit_open:
                be.circuit_open = False
                be.health_score = 0.5

    def report_error(self, backend_name: str, latency: float = 0.0) -> None:
        with self._lock:
            be = self._backends.get(backend_name)
            if be is None:
                return
            be.total_requests += 1
            be.total_errors += 1
            be.consecutive_errors += 1
            be.consecutive_successes = 0
            be.last_error_time = time.time()
            be.health_score = max(0.0, be.health_score - 0.2)
            if latency > 0:
                if be.avg_latency == 0:
                    be.avg_latency = latency
                else:
                    be.avg_latency = 0.7 * be.avg_latency + 0.3 * latency
            if be.consecutive_errors >= self._error_threshold and not be.circuit_open:
                be.circuit_open = True
                be.circuit_open_until = time.time() + self._circuit_cooldown
                self._stats["total_circuit_opens"] += 1
                self._emit_event("health_router.circuit_open", backend_name)

    def get_backend_status(self, name: str) -> dict[str, Any] | None:
        with self._lock:
            be = self._backends.get(name)
            if be is None:
                return None
            return {
                "name": be.name,
                "health_score": round(be.health_score, 3),
                "total_requests": be.total_requests,
                "total_errors": be.total_errors,
                "consecutive_errors": be.consecutive_errors,
                "avg_latency": round(be.avg_latency, 4),
                "circuit_open": be.circuit_open,
                "circuit_open_until": be.circuit_open_until,
                "priority": be.name in self._priority_backends,
            }

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "backends": {
                    n: self.get_backend_status(n) for n in self._backends
                },
                "priority_backends": list(self._priority_backends),
                "stats": dict(self._stats),
                "error_threshold": self._error_threshold,
                "recovery_threshold": self._recovery_threshold,
            }

    def _update_circuits_locked(self) -> None:
        now = time.time()
        for name, fn in self._health_checks.items():
            be = self._backends.get(name)
            if be is None:
                continue
            if now - be.last_checked > self._health_check_interval:
                try:
                    score = fn()
                    be.health_score = max(0.0, min(1.0, score))
                    be.last_checked = now
                except Exception as e:
                    be.health_score = max(0.0, be.health_score - 0.1)
                    be.last_checked = now
                    logger.debug(f"Health check failed for backend {name}: {e}")

    def _emit_event(self, event_type: str, backend_name: str) -> None:
        try:
            from core.event_bus import get_event_bus
            get_event_bus().publish(event_type, {"backend": backend_name})
        except Exception as e:
            logger.debug(f"Health event emission failed for {backend_name}: {e}")


_router: HealthRouter | None = None
_router_lock = threading.Lock()


def get_health_router(**kwargs) -> HealthRouter:
    global _router
    if _router is None:
        with _router_lock:
            if _router is None:
                _router = HealthRouter(**kwargs)
    return _router