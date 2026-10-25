"""Tests for system health, metrics, and status endpoints."""
import os
import tempfile
import threading

import pytest

os.environ.setdefault("JARVIS_HOME", tempfile.mkdtemp())

from core.health_check import (
    get_health_checker,
    get_system_metrics,
    get_jarvis_status,
    HealthChecker,
    HealthStatus,
)


class TestSystemMetrics:
    def test_returns_dict(self):
        metrics = get_system_metrics()
        assert isinstance(metrics, dict)

    def test_has_timestamp(self):
        metrics = get_system_metrics()
        assert "timestamp" in metrics

    def test_has_platform(self):
        metrics = get_system_metrics()
        assert "platform" in metrics
        assert "system" in metrics["platform"]
        assert "python_version" in metrics["platform"]

    def test_has_cpu_info(self):
        metrics = get_system_metrics()
        assert "cpu" in metrics

    def test_has_memory_info(self):
        metrics = get_system_metrics()
        assert "memory" in metrics

    def test_has_disk_info(self):
        metrics = get_system_metrics()
        assert "disk" in metrics

    def test_has_process_info(self):
        metrics = get_system_metrics()
        assert "process" in metrics
        assert "pid" in metrics["process"]

    def test_psutil_metrics_if_available(self):
        try:
            import psutil
            metrics = get_system_metrics()
            assert "percent" in metrics["cpu"]
            assert "total_gb" in metrics["memory"]
            assert "rss_mb" in metrics["process"]
        except ImportError:
            pytest.skip("psutil not available")


class TestJarvisStatus:
    def test_returns_dict(self):
        status = get_jarvis_status()
        assert isinstance(status, dict)

    def test_has_timestamp(self):
        status = get_jarvis_status()
        assert "timestamp" in status

    def test_has_health(self):
        status = get_jarvis_status()
        assert "health" in status

    def test_has_system(self):
        status = get_jarvis_status()
        assert "system" in status

    def test_has_oracle_circuit_breaker(self):
        status = get_jarvis_status()
        assert "oracle_circuit_breaker" in status

    def test_has_rate_limiter(self):
        status = get_jarvis_status()
        assert "rate_limiter" in status

    def test_has_knowledge_graph(self):
        status = get_jarvis_status()
        assert "knowledge_graph" in status

    def test_health_subsystem_data(self):
        status = get_jarvis_status()
        health = status.get("health", {})
        if "error" not in health:
            assert "overall_status" in health or "subsystem_health" in health


class TestHealthCheckerRegistration:
    def test_register_custom_check(self):
        hc = HealthChecker()

        def custom_check():
            from core.health_check import SubsystemHealth
            return SubsystemHealth(
                name="custom",
                status=HealthStatus.HEALTHY,
                message="Custom check OK",
            )

        hc.register_check("custom", custom_check)
        result = hc.check_all()
        names = [s.name for s in result.subsystems]
        assert "custom" in names

    def test_custom_check_degraded(self):
        hc = HealthChecker()

        def degraded_check():
            from core.health_check import SubsystemHealth
            return SubsystemHealth(
                name="degraded_test",
                status=HealthStatus.DEGRADED,
                message="Partially broken",
            )

        hc.register_check("degraded_test", degraded_check)
        result = hc.check_all()
        assert result.overall_status == HealthStatus.DEGRADED

    def test_custom_check_unhealthy(self):
        hc = HealthChecker()

        def unhealthy_check():
            from core.health_check import SubsystemHealth
            return SubsystemHealth(
                name="broken_test",
                status=HealthStatus.UNHEALTHY,
                message="Completely broken",
            )

        hc.register_check("broken_test", unhealthy_check)
        result = hc.check_all()
        assert result.overall_status == HealthStatus.UNHEALTHY

    def test_check_stats(self):
        hc = HealthChecker()
        stats = hc.get_stats()
        assert "registered_checks" in stats
        assert "uptime_seconds" in stats


class TestHealthCheckerSingleton:
    def test_get_health_checker(self):
        hc = get_health_checker()
        assert hc is not None
        assert isinstance(hc, HealthChecker)

    def test_singleton(self):
        hc1 = get_health_checker()
        hc2 = get_health_checker()
        assert hc1 is hc2


class TestHealthCheckResultSerialization:
    def test_to_dict(self):
        hc = HealthChecker()
        result = hc.check_all()
        d = result.to_dict()
        assert "overall_status" in d
        assert "timestamp" in d
        assert "uptime_seconds" in d
        assert "subsystems" in d

    def test_subsystem_serialization(self):
        hc = HealthChecker()
        result = hc.check_all()
        d = result.to_dict()
        for sub in d["subsystems"]:
            assert "name" in sub
            assert "status" in sub
            assert "message" in sub


class TestHealthCheckThreadSafety:
    def test_concurrent_checks(self):
        hc = HealthChecker()
        errors = []
        results = []

        def run_check():
            try:
                r = hc.check_all()
                results.append(r.overall_status)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=run_check) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(results) == 5