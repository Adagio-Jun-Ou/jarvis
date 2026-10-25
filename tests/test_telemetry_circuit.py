"""Test Telemetry Aggregator and Circuit Breaker enhancements."""

import threading
import time
import pytest
from core.telemetry_aggregator import (
    TelemetryAggregator,
    SubsystemCategory,
    SubsystemTelemetry,
    AnomalyEvent,
    get_telemetry_aggregator,
)
from core.circuit_breaker import (
    CircuitBreaker,
    CircuitConfig,
    CircuitState,
    CircuitBreakerManager,
    get_circuit_breaker,
)


class TestTelemetryAggregator:
    def setup_method(self):
        self.aggregator = TelemetryAggregator()

    def test_register_collector(self):
        result = self.aggregator.register(
            "test_sys", SubsystemCategory.MEMORY, lambda: {"total": 100, "errors": 5}
        )
        assert result is True

    def test_collect(self):
        self.aggregator.register("sys1", SubsystemCategory.MEMORY, lambda: {"total": 100, "errors": 5})
        self.aggregator.register("sys2", SubsystemCategory.COGNITION, lambda: {"running": True})
        snapshot = self.aggregator.collect()
        assert "sys1" in snapshot
        assert "sys2" in snapshot
        assert snapshot["sys1"].metrics["total"] == 100
        assert snapshot["sys2"].category == SubsystemCategory.COGNITION

    def test_health_score_computation(self):
        self.aggregator.register("healthy", SubsystemCategory.LLM, lambda: {"total": 100, "errors": 2})
        snapshot = self.aggregator.collect()
        assert snapshot["healthy"].health_score > 0

    def test_health_labels(self):
        self.aggregator.register("perfect", SubsystemCategory.LLM, lambda: {"running": True})
        self.aggregator.register("failing", SubsystemCategory.LLM, lambda: {"total": 100, "errors": 100})
        snapshot = self.aggregator.collect()
        assert snapshot["perfect"].health_label in ("EXCELLENT", "GOOD")
        assert snapshot["failing"].health_label in ("CRITICAL", "POOR")

    def test_get_snapshot(self):
        self.aggregator.register("test", SubsystemCategory.SECURITY, lambda: {"blocked": 10, "total_sandboxed": 50})
        self.aggregator.collect()
        snap = self.aggregator.get_snapshot()
        assert "test" in snap

    def test_get_summary(self):
        self.aggregator.register("s1", SubsystemCategory.MEMORY, lambda: {"running": True})
        self.aggregator.register("s2", SubsystemCategory.COGNITION, lambda: {"success_rate": 0.95})
        self.aggregator.collect()
        summary = self.aggregator.get_summary()
        assert "overall_health" in summary
        assert "category_health" in summary
        assert "subsystems" in summary
        assert summary["subsystem_count"] == 2

    def test_anomaly_detection(self):
        call_count = [0]
        def metrics_fn():
            call_count[0] += 1
            if call_count[0] < 3:
                return {"total": 100, "errors": 5}
            else:
                return {"total": 100, "errors": 95}

        self.aggregator.register("anomaly_test", SubsystemCategory.LLM, metrics_fn)
        self.aggregator.collect()
        self.aggregator.collect()
        snapshot = self.aggregator.collect()

        from core.telemetry_aggregator import TelemetryAggregator
        anomalies = list(self.aggregator._anomalies)
        assert len(anomalies) > 0

    def test_unregister(self):
        self.aggregator.register("temp", SubsystemCategory.INFRASTRUCTURE, lambda: {"running": True})
        assert self.aggregator.unregister("temp") is True
        assert self.aggregator.unregister("nonexistent") is False

    def test_max_collectors(self):
        for i in range(TelemetryAggregator._MAX_COLLECTORS + 5):
            self.aggregator.register(f"sys_{i}", SubsystemCategory.LLM, lambda: {"total": 100})
        with self.aggregator._lock:
            count = len(self.aggregator._collectors)
        assert count <= TelemetryAggregator._MAX_COLLECTORS

    def test_get_history(self):
        self.aggregator.register("hist", SubsystemCategory.MEMORY, lambda: {"running": True})
        self.aggregator.collect()
        self.aggregator.collect()
        history = self.aggregator.get_history(limit=2)
        assert len(history) == 2

    def test_get_anomalies(self):
        anomalies = self.aggregator.get_anomalies()
        assert isinstance(anomalies, list)

    def test_force_collect(self):
        self.aggregator.register("force", SubsystemCategory.SECURITY, lambda: {"total": 10})
        result = self.aggregator.force_collect()
        assert "force" in result

    def test_collector_exception_handling(self):
        def broken_collector():
            raise ValueError("test error")

        self.aggregator.register("broken", SubsystemCategory.LLM, broken_collector)
        snapshot = self.aggregator.collect()
        assert "broken" in snapshot
        assert "error" in snapshot["broken"].metrics

    def test_singleton(self):
        t1 = get_telemetry_aggregator()
        t2 = get_telemetry_aggregator()
        assert t1 is t2

    def test_thread_safety(self):
        errors = []

        def register_and_collect(n):
            try:
                for i in range(n):
                    self.aggregator.register(f"t_sys_{n}_{i}", SubsystemCategory.LLM, lambda: {"total": i * 10})
                self.aggregator.collect()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=register_and_collect, args=(10,)) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestCircuitBreakerEnhanced:
    def test_create_breaker(self):
        cb = CircuitBreaker(name="test_llm")
        assert cb.state == CircuitState.CLOSED

    def test_success_keeps_closed(self):
        cb = CircuitBreaker(name="test_success")
        for _ in range(10):
            cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_failures_open_circuit(self):
        cb = CircuitBreaker(name="test_failures", config=CircuitConfig(failure_threshold=3))
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_recovery_from_open(self):
        cb = CircuitBreaker(name="test_recovery", config=CircuitConfig(
            failure_threshold=2,
            timeout_seconds=0.1,
            success_threshold=1,
        ))
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        time.sleep(0.15)
        state = cb.state
        assert state == CircuitState.HALF_OPEN

        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_half_open_limits_calls(self):
        cb = CircuitBreaker(name="test_half_open", config=CircuitConfig(
            failure_threshold=2,
            timeout_seconds=0.1,
            half_open_max_calls=1,
        ))
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN

        assert cb._can_execute() is True
        assert cb._can_execute() is False

    def test_stats(self):
        cb = CircuitBreaker(name="test_stats")
        cb.record_success()
        cb.record_failure()
        stats = cb.stats
        assert stats.success_count == 1
        assert stats.failure_count == 1

    def test_reset(self):
        cb = CircuitBreaker(name="test_reset", config=CircuitConfig(failure_threshold=2))
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        cb.reset()
        assert cb.state == CircuitState.CLOSED

    def test_wall_clock_step_does_not_alter_recovery(self, monkeypatch):
        """Regression: recovery was timed with time.time(), so a wall-clock
        step (NTP correction, manual change) could trip recovery early or pin
        the breaker open indefinitely. Recovery must follow the monotonic
        clock only."""
        cb = CircuitBreaker(
            name="test_clock_step",
            config=CircuitConfig(failure_threshold=1, timeout_seconds=60.0),
        )
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        real_time = time.time()

        # A decade forward must not 'elapse' the 60s recovery window.
        monkeypatch.setattr(time, "time", lambda: real_time + 3650 * 86400.0)
        assert cb.state == CircuitState.OPEN, (
            "future wall-clock step tripped recovery early"
        )

        # A decade backward must not pin it open beyond real elapsed time
        # either; with no real time passed the breaker stays OPEN.
        monkeypatch.setattr(time, "time", lambda: real_time - 3650 * 86400.0)
        assert cb.state == CircuitState.OPEN

        # Real monotonic elapsed time exceeding the window recovers it,
        # regardless of the lying wall clock.
        cb_short = CircuitBreaker(
            name="test_clock_step_short",
            config=CircuitConfig(failure_threshold=1, timeout_seconds=0.01),
        )
        cb_short.record_failure()
        assert cb_short.state == CircuitState.OPEN
        time.sleep(0.02)
        assert cb_short.state == CircuitState.HALF_OPEN, (
            "backward wall-clock step pinned the breaker open"
        )

    def test_manager(self):
        mgr = CircuitBreakerManager.get_instance()
        cb1 = mgr.get_or_create("test_mgr_1")
        cb2 = mgr.get_or_create("test_mgr_1")
        assert cb1 is cb2

        stats = mgr.get_all_stats()
        assert "test_mgr_1" in stats

    def test_convenience_function(self):
        cb = get_circuit_breaker("test_convenience")
        assert cb is not None

    def test_thread_safety(self):
        cb = CircuitBreaker("test_threads", config=CircuitConfig(failure_threshold=50))

        def hammer(n):
            for i in range(n):
                cb.record_success()
                cb.record_failure()
                _ = cb.state

        threads = [threading.Thread(target=hammer, args=(100,)) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert cb.stats.total_successes > 0
        assert cb.stats.total_failures > 0