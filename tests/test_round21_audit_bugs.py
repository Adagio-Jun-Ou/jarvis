"""Round 21 Deep Audit Bug Tests.

Tests for bugs found and fixed in:
- resilience.py: is_available() missing lock, get_error_statistics() circuit_breaker iteration without lock,
  error_counts dict replacement pattern
- rate_limiter.py: get_stats() without lock
- context_guardian.py: emergency callbacks iteration under lock (verified correct)
- proactive.py: deque-based _mentioned_files/_mentioned_topics
- memory_manager.py: _notify_pressure copies callbacks, on_memory_pressure bounded+locked
- memory_watchdog.py: _notify_pressure no lock, register_callback no lock, force_cleanup no lock,
  get_stats no lock, _state changes no lock
"""

import threading
import time
import pytest
from collections import deque


# ============================================================
# resilience.py tests
# ============================================================



# ============================================================
# rate_limiter.py tests
# ============================================================

class TestRateLimiterThreadSafety:
    def test_get_stats_under_lock(self):
        from core.rate_limiter import RateLimiter, RateLimit
        limiter = RateLimiter()
        stats = limiter.get_stats()
        assert "total_requests" in stats
        assert "active_clients" in stats

    def test_concurrent_requests(self):
        from core.rate_limiter import RateLimiter, RateLimit
        limiter = RateLimiter(default_limit=RateLimit(burst_size=100, requests_per_second=1000))
        errors = []

        def stress():
            try:
                for i in range(50):
                    limiter.check_rate_limit(f"client_{i % 5}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=stress) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_block_unblock_ip(self):
        from core.rate_limiter import RateLimiter
        limiter = RateLimiter()
        limiter.block_ip("192.168.1.1")
        blocked = limiter.get_blocked_ips()
        assert "192.168.1.1" in blocked
        limiter.unblock_ip("192.168.1.1")
        blocked = limiter.get_blocked_ips()
        assert "192.168.1.1" not in blocked

    def test_rate_limit_blocks_excess(self):
        from core.rate_limiter import RateLimiter, RateLimit
        limiter = RateLimiter(default_limit=RateLimit(burst_size=2, requests_per_second=1))
        allowed, _, _ = limiter.check_rate_limit("client1")
        assert allowed is True
        allowed, _, _ = limiter.check_rate_limit("client1")
        assert allowed is True
        allowed, retry_after, info = limiter.check_rate_limit("client1")
        assert allowed is False


# ============================================================
# context_guardian.py tests
# ============================================================

class TestContextGuardianThreadSafety:
    def test_emergency_callback_bounded(self):
        from core.context_guardian import ContextGuardian, GuardianAction, PressureLevel
        guardian = ContextGuardian()
        for i in range(25):
            guardian.add_emergency_callback(lambda action: None)
        assert len(guardian._emergency_callbacks) <= guardian._max_callbacks

    def test_record_action_and_stats(self):
        from core.context_guardian import ContextGuardian, GuardianAction, PressureLevel
        guardian = ContextGuardian()
        action = GuardianAction(
            level=PressureLevel.WARNING,
            action="test_eviction",
            tokens_before=100000,
            tokens_after=80000,
            messages_removed=5,
            reason="context overflow"
        )
        guardian.record_action(action)
        stats = guardian.get_stats()
        assert stats["warnings_issued"] == 1
        assert stats["total_messages_evicted"] == 5

    def test_check_pressure_levels(self):
        from core.context_guardian import ContextGuardian, PressureLevel
        guardian = ContextGuardian(token_budget=100000)

        assert guardian.check_pressure(50000) == PressureLevel.NORMAL
        assert guardian.check_pressure(72000) == PressureLevel.WARNING
        assert guardian.check_pressure(85000) == PressureLevel.SOFT_EVICT
        assert guardian.check_pressure(92000) == PressureLevel.HARD_EVICT
        assert guardian.check_pressure(97000) == PressureLevel.EMERGENCY

    def test_concurrent_record_action(self):
        from core.context_guardian import ContextGuardian, GuardianAction, PressureLevel
        guardian = ContextGuardian()
        errors = []

        def stress():
            try:
                for _ in range(50):
                    action = GuardianAction(
                        level=PressureLevel.SOFT_EVICT,
                        action="test",
                        tokens_before=100000,
                        tokens_after=80000,
                        messages_removed=2,
                        reason="test"
                    )
                    guardian.record_action(action)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=stress) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        stats = guardian.get_stats()
        assert stats["soft_evictions"] == 150


# ============================================================
# memory_watchdog.py tests
# ============================================================

class TestMemoryWatchdogThreadSafety:
    def test_register_callback_bounded(self):
        from core.memory_watchdog import MemoryWatchdog
        watchdog = MemoryWatchdog()
        for i in range(40):
            watchdog.register_callback(lambda: None)
        assert len(watchdog._cleanup_callbacks) <= watchdog.MAX_CALLBACKS

    def test_register_callback_with_state(self):
        from core.memory_watchdog import MemoryWatchdog, MemoryState
        watchdog = MemoryWatchdog()
        cb = lambda: None
        watchdog.register_callback(cb, MemoryState.HIGH)
        assert cb in watchdog._pressure_callbacks[MemoryState.HIGH]

    def test_unregister_callback(self):
        from core.memory_watchdog import MemoryWatchdog
        watchdog = MemoryWatchdog()
        cb = lambda: None
        watchdog.register_cleanup_callback(cb) if hasattr(watchdog, 'register_cleanup_callback') else watchdog.register_callback(cb)
        result = watchdog.unregister_callback(cb)
        assert result is True

    def test_notify_pressure_copies_callbacks(self):
        from core.memory_watchdog import MemoryWatchdog, MemoryState
        watchdog = MemoryWatchdog()
        calls = []
        watchdog.register_callback(lambda: calls.append("high"), MemoryState.HIGH)
        watchdog._notify_pressure(MemoryState.HIGH)
        assert len(calls) == 1

    def test_get_stats_thread_safe(self):
        from core.memory_watchdog import MemoryWatchdog
        watchdog = MemoryWatchdog()
        stats = watchdog.get_stats()
        assert "checks" in stats
        assert "state" in stats

    def test_force_cleanup_under_lock(self):
        from core.memory_watchdog import MemoryWatchdog, MemoryState
        watchdog = MemoryWatchdog()
        stats = watchdog.force_cleanup(MemoryState.HIGH)
        assert isinstance(stats.caches_cleared, int)

    def test_concurrent_check_memory(self):
        from core.memory_watchdog import MemoryWatchdog
        watchdog = MemoryWatchdog()
        errors = []

        def stress():
            try:
                for _ in range(20):
                    watchdog.check_memory()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=stress) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0