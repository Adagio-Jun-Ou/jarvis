"""Test Graceful Degradation Manager."""

import threading
import time
import pytest
from core.graceful_degradation import (
    GracefulDegradationManager,
    DegradationLevel,
    SubsystemStatus,
    get_degradation_manager,
)


class TestGracefulDegradation:
    def setup_method(self):
        self.mgr = GracefulDegradationManager()

    def test_register_subsystem(self):
        self.mgr.register_subsystem("test_sys")
        assert self.mgr.get_status("test_sys") == SubsystemStatus.UNAVAILABLE

    def test_mark_healthy(self):
        self.mgr.register_subsystem("sys1")
        self.mgr.mark_healthy("sys1")
        assert self.mgr.get_status("sys1") == SubsystemStatus.HEALTHY

    def test_mark_failed(self):
        self.mgr.register_subsystem("sys2")
        self.mgr.mark_failed("sys2", "connection error")
        assert self.mgr.get_status("sys2") == SubsystemStatus.FAILED

    def test_mark_degraded(self):
        self.mgr.register_subsystem("sys3")
        self.mgr.mark_degraded("sys3")
        assert self.mgr.get_status("sys3") == SubsystemStatus.DEGRADED

    def test_is_available(self):
        self.mgr.register_subsystem("sys4")
        assert self.mgr.is_available("sys4") is False  # UNAVAILABLE
        self.mgr.mark_healthy("sys4")
        assert self.mgr.is_available("sys4") is True
        self.mgr.mark_degraded("sys4")
        assert self.mgr.is_available("sys4") is True
        self.mgr.mark_failed("sys4")
        assert self.mgr.is_available("sys4") is False

    def test_get_fallback(self):
        fallback = self.mgr.get_fallback("memory_router", "fallback_route")
        assert fallback is not None
        assert fallback["tier"] == "L2_WARM"

    def test_get_fallback_default(self):
        fallback = self.mgr.get_fallback("nonexistent", "key", default="safe")
        assert fallback == "safe"

    def test_wrap_call_success(self):
        self.mgr.register_subsystem("wrapped")
        result = self.mgr.wrap_call("wrapped", lambda: 42, "fallback_route")
        assert result == 42
        assert self.mgr.get_status("wrapped") == SubsystemStatus.HEALTHY

    def test_wrap_call_failure_returns_fallback(self):
        self.mgr.register_subsystem("failing", {"fallback_classify": "CONVERSATIONAL"})
        self.mgr.mark_healthy("failing")
        result = self.mgr.wrap_call(
            "failing",
            lambda: (_ for _ in ()).throw(RuntimeError("boom")),
            "fallback_classify",
        )
        assert result == "CONVERSATIONAL"

    def test_wrap_call_failed_subsystem_returns_fallback(self):
        self.mgr.register_subsystem("already_failed", {"fallback_classify": "CONVERSATIONAL"})
        for _ in range(3):
            self.mgr.mark_failed("already_failed", "error")
        result = self.mgr.wrap_call(
            "already_failed",
            lambda: "should_not_see",
            "fallback_classify",
        )
        assert result == "CONVERSATIONAL"

    def test_degradation_level_full(self):
        self.mgr.register_subsystem("a")
        self.mgr.register_subsystem("b")
        self.mgr.mark_healthy("a")
        self.mgr.mark_healthy("b")
        level = self.mgr.compute_degradation_level()
        assert level == DegradationLevel.FULL

    def test_degradation_level_emergency(self):
        for i in range(6):
            self.mgr.register_subsystem(f"e{i}")
            self.mgr.mark_failed(f"e{i}", f"error {i}")
        level = self.mgr.compute_degradation_level()
        assert level == DegradationLevel.EMERGENCY

    def test_degradation_level_degraded(self):
        for i in range(4):
            self.mgr.register_subsystem(f"d{i}")
            self.mgr.mark_healthy(f"d{i}")
        self.mgr.register_subsystem("d_fail")
        self.mgr.mark_failed("d_fail", "err")
        self.mgr.register_subsystem("d_fail2")
        self.mgr.mark_failed("d_fail2", "err2")
        level = self.mgr.compute_degradation_level()
        assert level in (DegradationLevel.FULL, DegradationLevel.DEGRADED, DegradationLevel.MINIMAL)

    def test_get_summary(self):
        self.mgr.register_subsystem("sum1")
        self.mgr.mark_healthy("sum1")
        summary = self.mgr.get_summary()
        assert "degradation_level" in summary
        assert "healthy" in summary
        assert "failed" in summary
        assert summary["healthy"] == 1

    def test_reset(self):
        self.mgr.register_subsystem("reset_test")
        self.mgr.mark_failed("reset_test", "error")
        self.mgr.reset()
        assert self.mgr.get_status("reset_test") == SubsystemStatus.UNAVAILABLE

    def test_custom_fallback_data(self):
        self.mgr.register_subsystem("custom_sys", {"my_key": "my_value"})
        assert self.mgr.get_fallback("custom_sys", "my_key") == "my_value"

    def test_thread_safety(self):
        errors = []

        def stress(n):
            try:
                for i in range(n):
                    name = f"thread_{n}_{i}"
                    self.mgr.register_subsystem(name)
                    self.mgr.mark_healthy(name)
                    self.mgr.wrap_call(name, lambda: 42, "fallback_classify")
                    self.mgr.mark_failed(name, "test")
                    self.mgr.compute_degradation_level()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=stress, args=(20,)) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_singleton(self):
        m1 = get_degradation_manager()
        m2 = get_degradation_manager()
        assert m1 is m2