"""Tests for Health Checker system."""
import pytest


class TestHealthChecker:
    def test_import(self):
        from core.health_check import HealthChecker, HealthStatus, get_health_checker
        assert HealthChecker is not None
        assert HealthStatus.HEALTHY.value == "healthy"

    def test_singleton(self):
        from core.health_check import get_health_checker
        h1 = get_health_checker()
        h2 = get_health_checker()
        assert h1 is h2

    def test_check_all(self):
        from core.health_check import get_health_checker, HealthStatus
        checker = get_health_checker()
        result = checker.check_all()
        assert result.overall_status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED, HealthStatus.UNHEALTHY]
        assert result.to_dict()["overall_status"] in ["healthy", "degraded", "unhealthy"]
        assert len(result.subsystems) > 0

    def test_check_memory(self):
        from core.health_check import HealthChecker, HealthStatus
        checker = HealthChecker()
        health = checker._check_memory()
        assert health.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED, HealthStatus.UNHEALTHY]
        assert health.name == "memory"

    def test_check_context_guardian(self):
        from core.health_check import HealthChecker, HealthStatus
        checker = HealthChecker()
        health = checker._check_context_guardian()
        assert health.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
        assert health.name == "context_guardian"

    def test_check_lcm_shield(self):
        from core.health_check import HealthChecker, HealthStatus
        checker = HealthChecker()
        health = checker._check_lcm_shield()
        assert health.name == "lcm_shield"

    def test_check_prompt_sandbox(self):
        from core.health_check import HealthChecker, HealthStatus
        checker = HealthChecker()
        health = checker._check_prompt_sandbox()
        assert health.name == "prompt_sandbox"

    def test_check_three_layer(self):
        from core.health_check import HealthChecker, HealthStatus
        checker = HealthChecker()
        health = checker._check_three_layer()
        assert health.name == "three_layer"

    def test_get_stats(self):
        from core.health_check import get_health_checker
        checker = get_health_checker()
        stats = checker.get_stats()
        assert "registered_checks" in stats
        assert stats["registered_checks"] >= 5

    def test_register_custom_check(self):
        from core.health_check import HealthChecker, HealthStatus, SubsystemHealth
        checker = HealthChecker()
        def my_check():
            return SubsystemHealth(name="custom", status=HealthStatus.HEALTHY, message="OK")
        checker.register_check("custom", my_check)
        result = checker.check_all()
        custom = [s for s in result.subsystems if s.name == "custom"]
        assert len(custom) == 1
        assert custom[0].status == HealthStatus.HEALTHY