"""Tests for Oracle integration hardening: circuit breaker, call cache, retry."""
import os
import tempfile
import time
import threading

import pytest

os.environ.setdefault("JARVIS_HOME", tempfile.mkdtemp())

from core.oracle_integration import (
    OracleCircuitBreaker,
    OracleCallCache,
    get_oracle_circuit_breaker,
    get_oracle_call_cache,
    oracle_retry_call,
    cb_state,
    cache_stats,
)


class TestOracleCircuitBreaker:
    def test_initial_state_closed(self):
        cb = OracleCircuitBreaker()
        assert cb.state == "closed"

    def test_allow_call_when_closed(self):
        cb = OracleCircuitBreaker()
        assert cb.allow_call() is True

    def test_transitions_to_open_after_failures(self):
        cb = OracleCircuitBreaker(failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == "open"

    def test_blocks_calls_when_open(self):
        cb = OracleCircuitBreaker(failure_threshold=2, recovery_timeout=999)
        cb.record_failure()
        cb.record_failure()
        assert cb.allow_call() is False

    def test_transitions_to_half_open_after_timeout(self):
        cb = OracleCircuitBreaker(failure_threshold=2, recovery_timeout=0.01)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "open"
        time.sleep(0.02)
        assert cb.state == "half_open"

    def test_allows_limited_calls_in_half_open(self):
        cb = OracleCircuitBreaker(failure_threshold=2, recovery_timeout=0.01, half_open_max=1)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.02)
        assert cb.allow_call() is True
        assert cb.allow_call() is False

    def test_half_open_success_closes_circuit(self):
        cb = OracleCircuitBreaker(failure_threshold=2, recovery_timeout=0.01, half_open_max=2)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.02)
        assert cb.allow_call() is True
        cb.record_success()
        assert cb.allow_call() is True
        cb.record_success()
        assert cb.state == "closed"

    def test_half_open_failure_reopens(self):
        cb = OracleCircuitBreaker(failure_threshold=2, recovery_timeout=0.01, half_open_max=1)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.02)
        cb.allow_call()
        cb.record_failure()
        assert cb.state == "open"

    def test_success_decrements_failure_count(self):
        cb = OracleCircuitBreaker(failure_threshold=5)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb._failure_count == 1

    def test_reset(self):
        cb = OracleCircuitBreaker(failure_threshold=2)
        cb.record_failure()
        cb.record_failure()
        cb.reset()
        assert cb.state == "closed"
        assert cb._failure_count == 0

    def test_wall_clock_step_does_not_alter_recovery(self, monkeypatch):
        """Regression: recovery was timed with time.time(); a wall-clock step
        could trip recovery early or pin the breaker open. Recovery must
        follow the monotonic clock only."""
        cb = OracleCircuitBreaker(failure_threshold=1, recovery_timeout=60.0)
        cb.record_failure()
        assert cb.allow_call() is False

        real_time = time.time()

        # A decade forward must not 'elapse' the 60s recovery window.
        monkeypatch.setattr(time, "time", lambda: real_time + 3650 * 86400.0)
        assert cb.allow_call() is False, (
            "future wall-clock step tripped recovery early"
        )

        # A decade backward: with no real time passed, still blocked.
        monkeypatch.setattr(time, "time", lambda: real_time - 3650 * 86400.0)
        assert cb.state == "open"

        # Real monotonic elapsed time beyond a short window recovers it,
        # regardless of the lying wall clock.
        cb_short = OracleCircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
        cb_short.record_failure()
        assert cb_short.allow_call() is False
        time.sleep(0.02)
        assert cb_short.allow_call() is True, (
            "backward wall-clock step pinned the breaker open"
        )

    def test_get_status(self):
        cb = OracleCircuitBreaker(failure_threshold=3, recovery_timeout=60.0)
        cb.record_failure()
        status = cb.get_status()
        assert "state" in status
        assert "failure_count" in status
        assert "threshold" in status
        assert status["failure_count"] == 1

    def test_total_calls_tracking(self):
        cb = OracleCircuitBreaker()
        cb.record_success()
        cb.record_success()
        cb.record_failure()
        status = cb.get_status()
        assert status["total_calls"] == 3
        assert status["total_failures"] == 1

    def test_thread_safety(self):
        cb = OracleCircuitBreaker(failure_threshold=100)
        errors = []

        def record_successes():
            try:
                for _ in range(50):
                    cb.record_success()
            except Exception as e:
                errors.append(e)

        def record_failures():
            try:
                for _ in range(50):
                    cb.record_failure()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=record_successes),
            threading.Thread(target=record_failures),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert cb._total_calls == 100


class TestOracleCallCache:
    def test_set_and_get(self):
        cache = OracleCallCache(ttl=60)
        cache.set("key1", {"result": "value"})
        assert cache.get("key1") == {"result": "value"}

    def test_miss_returns_none(self):
        cache = OracleCallCache(ttl=60)
        assert cache.get("nonexistent") is None

    def test_ttl_expiry(self):
        cache = OracleCallCache(ttl=0.01)
        cache.set("key1", "value")
        time.sleep(0.02)
        assert cache.get("key1") is None

    def test_invalidate_key(self):
        cache = OracleCallCache(ttl=60)
        cache.set("key1", "value1")
        cache.invalidate("key1")
        assert cache.get("key1") is None

    def test_invalidate_all(self):
        cache = OracleCallCache(ttl=60)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.invalidate()
        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_eviction_on_overflow(self):
        cache = OracleCallCache(ttl=60, max_size=5)
        for i in range(10):
            cache.set(f"key_{i}", f"value_{i}")
        assert len(cache._cache) <= 5

    def test_hit_miss_tracking(self):
        cache = OracleCallCache(ttl=60)
        cache.set("k", "v")
        cache.get("k")
        cache.get("k")
        cache.get("miss1")
        stats = cache.stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1

    def test_hit_rate_calculation(self):
        cache = OracleCallCache(ttl=60)
        cache.set("k", "v")
        cache.get("k")
        cache.get("miss")
        stats = cache.stats()
        assert stats["hit_rate"] == 0.5

    def test_stats(self):
        cache = OracleCallCache(ttl=300, max_size=100)
        stats = cache.stats()
        assert "size" in stats
        assert "max_size" in stats
        assert "ttl" in stats
        assert "hits" in stats
        assert "misses" in stats

    def test_thread_safety(self):
        cache = OracleCallCache(ttl=60, max_size=500)
        errors = []

        def writer():
            try:
                for i in range(100):
                    cache.set(f"key_{i}", f"val_{i}")
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for i in range(100):
                    cache.get(f"key_{i}")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors


class TestOracleRetryCall:
    def test_success_no_retry(self):
        call_count = 0

        def succeeds():
            nonlocal call_count
            call_count += 1
            return "ok"

        cb = get_oracle_circuit_breaker()
        cb.reset()
        result = oracle_retry_call(succeeds, max_retries=3)
        assert result == "ok"
        assert call_count == 1

    def test_retry_on_failure(self):
        call_count = 0

        def fails_twice():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("timeout")
            return "ok"

        cb = get_oracle_circuit_breaker()
        cb.reset()
        result = oracle_retry_call(fails_twice, max_retries=3, base_delay=0.01, max_delay=0.05)
        assert result == "ok"
        assert call_count == 3

    def test_max_retries_exhausted(self):
        def always_fails():
            raise ConnectionError("down")

        cb = get_oracle_circuit_breaker()
        cb.reset()
        with pytest.raises(ConnectionError):
            oracle_retry_call(always_fails, max_retries=2, base_delay=0.01, max_delay=0.05)

    def test_circuit_breaker_blocks(self):
        global_cb = get_oracle_circuit_breaker()
        global_cb.reset()
        for _ in range(global_cb._failure_threshold + 1):
            global_cb.record_failure()

        with pytest.raises(RuntimeError, match="circuit breaker"):
            oracle_retry_call(lambda: "ok", max_retries=0)

        global_cb.reset()

    def test_exponential_backoff(self):
        delays = []
        call_count = 0

        def check_delay():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TimeoutError("slow")
            return "ok"

        result = oracle_retry_call(
            check_delay,
            max_retries=3,
            base_delay=0.01,
            max_delay=1.0,
            backoff_factor=2.0,
        )
        assert result == "ok"


class TestSingletons:
    def test_circuit_breaker_singleton(self):
        cb1 = get_oracle_circuit_breaker()
        cb2 = get_oracle_circuit_breaker()
        assert cb1 is cb2

    def test_call_cache_singleton(self):
        c1 = get_oracle_call_cache()
        c2 = get_oracle_call_cache()
        assert c1 is c2

    def test_cb_state_function(self):
        state = cb_state()
        assert "state" in state
        assert "failure_count" in state

    def test_cache_stats_function(self):
        stats = cache_stats()
        assert "size" in stats
        assert "hits" in stats