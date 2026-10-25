"""Tests for core/health_router.py — HealthRouter."""
import time
import threading

import pytest

from core.health_router import HealthRouter, Backend, RouteDecision, get_health_router


class TestHealthRouterBasic:
    def test_register_and_route(self):
        hr = HealthRouter()
        hr.register_backend("llm_primary")
        hr.register_backend("llm_backup")
        decision = hr.route()
        assert decision.backend in ("llm_primary", "llm_backup")
        assert decision.reason == "weighted_health"

    def test_route_to_healthy_backend(self):
        hr = HealthRouter()
        hr.register_backend("healthy")
        hr.report_success("healthy", latency=0.1)
        decision = hr.route()
        assert decision.backend == "healthy"
        assert decision.health_score > 0

    def test_route_no_backends(self):
        hr = HealthRouter()
        decision = hr.route()
        assert decision.backend == "none"
        assert decision.reason == "no_backends"

    def test_unregister_backend(self):
        hr = HealthRouter()
        hr.register_backend("temp")
        hr.unregister_backend("temp")
        decision = hr.route()
        assert decision.backend == "none"


class TestHealthRouterPriority:
    def test_priority_backend_preferred(self):
        hr = HealthRouter()
        hr.register_backend("priority_be", priority=True)
        hr.register_backend("regular_be")
        be = hr._backends["priority_be"]
        be.health_score = 0.9
        hr._backends["regular_be"].health_score = 0.9
        decisions = [hr.route().backend for _ in range(100)]
        assert "priority_be" in decisions

    def test_required_backends(self):
        hr = HealthRouter()
        hr.register_backend("a")
        hr.register_backend("b")
        hr.register_backend("c")
        decision = hr.route(required_backends=["a", "b"])
        assert decision.backend in ("a", "b")


class TestHealthRouterCircuitBreaker:
    def test_circuit_opens_after_errors(self):
        hr = HealthRouter(error_threshold=3)
        hr.register_backend("flaky")
        for _ in range(3):
            hr.report_error("flaky", latency=1.0)
        decision = hr.route()
        assert decision.reason == "fallback_no_healthy"

    def test_circuit_recovers_after_cooldown(self):
        hr = HealthRouter(error_threshold=3, circuit_cooldown=0.05)
        hr.register_backend("flaky")
        for _ in range(3):
            hr.report_error("flaky")
        decision1 = hr.route()
        assert decision1.reason == "fallback_no_healthy"
        time.sleep(0.08)
        decision2 = hr.route()
        assert decision2.backend == "flaky"

    def test_circuit_recover_on_success(self):
        hr = HealthRouter(error_threshold=3, recovery_threshold=3)
        hr.register_backend("recovering")
        for _ in range(3):
            hr.report_error("recovering")
        assert hr._backends["recovering"].circuit_open is True
        hr._backends["recovering"].circuit_open_until = time.time() - 1
        hr._backends["recovering"].circuit_open = False
        for _ in range(3):
            hr.report_success("recovering")
        status = hr.get_backend_status("recovering")
        assert status["health_score"] > 0


class TestHealthRouterHealthScores:
    def test_success_increases_health(self):
        hr = HealthRouter()
        hr.register_backend("be")
        initial = hr._backends["be"].health_score
        hr.report_success("be")
        assert hr._backends["be"].health_score >= initial

    def test_error_decreases_health(self):
        hr = HealthRouter()
        hr.register_backend("be")
        hr.report_error("be")
        assert hr._backends["be"].health_score < 1.0

    def test_health_clamped(self):
        hr = HealthRouter()
        hr.register_backend("be")
        for _ in range(20):
            hr.report_success("be")
        assert hr._backends["be"].health_score <= 1.0
        for _ in range(20):
            hr.report_error("be")
        assert hr._backends["be"].health_score >= 0.0

    def test_latency_ema(self):
        hr = HealthRouter()
        hr.register_backend("be")
        hr.report_success("be", latency=0.1)
        assert hr._backends["be"].avg_latency == 0.1
        hr.report_success("be", latency=0.2)
        assert 0.1 < hr._backends["be"].avg_latency < 0.2


class TestHealthRouterWeightedSelection:
    def test_weighted_distribution(self):
        hr = HealthRouter()
        hr.register_backend("fast")
        hr.register_backend("slow")
        hr._backends["fast"].health_score = 1.0
        hr._backends["slow"].health_score = 0.1
        counts = {"fast": 0, "slow": 0}
        for _ in range(1000):
            d = hr.route(required_backends=["fast", "slow"])
            counts[d.backend] += 1
        assert counts["fast"] > counts["slow"]

    def test_all_equal_health(self):
        hr = HealthRouter()
        hr.register_backend("a")
        hr.register_backend("b")
        counts = {"a": 0, "b": 0}
        for _ in range(200):
            d = hr.route()
            counts[d.backend] += 1
        assert counts["a"] > 50
        assert counts["b"] > 50


class TestHealthRouterFallback:
    def test_fallback_when_all_unhealthy(self):
        hr = HealthRouter(min_health=0.5)
        hr.register_backend("low1")
        hr.register_backend("low2")
        hr._backends["low1"].health_score = 0.1
        hr._backends["low2"].health_score = 0.05
        decision = hr.route()
        assert decision.fallback is True
        assert decision.reason == "fallback_no_healthy"

    def test_fallback_picks_best(self):
        hr = HealthRouter(min_health=0.5)
        hr.register_backend("better")
        hr.register_backend("worse")
        hr._backends["better"].health_score = 0.4
        hr._backends["worse"].health_score = 0.1
        decision = hr.route()
        assert decision.backend == "better"


class TestHealthRouterHealthCheck:
    def test_health_check_callback(self):
        hr = HealthRouter(health_check_interval=0.01)
        score = [1.0]
        hr.register_backend("dynamic", health_fn=lambda: score[0])
        hr._backends["dynamic"].last_checked = 0
        hr._update_circuits_locked()
        assert hr._backends["dynamic"].health_score >= 1.0
        score[0] = 0.3
        hr._backends["dynamic"].last_checked = 0
        hr._update_circuits_locked()
        assert hr._backends["dynamic"].health_score <= 0.3

    def test_health_check_failure_degrades(self):
        def broken():
            raise RuntimeError("down")
        hr = HealthRouter(health_check_interval=0.01)
        hr.register_backend("broken", health_fn=broken)
        hr._backends["broken"].health_score = 1.0
        hr._backends["broken"].last_checked = 0
        hr._update_circuits_locked()
        assert hr._backends["broken"].health_score < 1.0


class TestHealthRouterStatus:
    def test_get_backend_status(self):
        hr = HealthRouter()
        hr.register_backend("be")
        hr.report_success("be", latency=0.1)
        status = hr.get_backend_status("be")
        assert status is not None
        assert status["name"] == "be"
        assert status["total_requests"] == 1
        assert status["avg_latency"] > 0

    def test_get_status(self):
        hr = HealthRouter()
        hr.register_backend("a")
        hr.register_backend("b", priority=True)
        status = hr.get_status()
        assert "backends" in status
        assert "priority_backends" in status
        assert "stats" in status
        assert "b" in status["priority_backends"]

    def test_missing_backend_status(self):
        hr = HealthRouter()
        assert hr.get_backend_status("missing") is None


class TestHealthRouterThreadSafety:
    def test_concurrent_routing(self):
        hr = HealthRouter()
        hr.register_backend("be1")
        hr.register_backend("be2")
        results = []
        errors = []

        def worker():
            for _ in range(50):
                try:
                    d = hr.route()
                    results.append(d.backend)
                    if d.backend != "none":
                        hr.report_success(d.backend, latency=0.01)
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        assert len(results) > 0


class TestGetHealthRouter:
    def test_singleton(self):
        a = get_health_router()
        b = get_health_router()
        assert a is b