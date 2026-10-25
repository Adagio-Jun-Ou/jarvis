"""Tests for core/unified_rate_limiter.py — UnifiedRateLimiter."""
import threading
import time

import pytest

from core.rate_limiter import RateLimit, RateLimiter
from core.streaming_pipeline import BackpressureController, BackpressureConfig
from core.unified_rate_limiter import UnifiedRateLimiter, get_unified_rate_limiter, UnifiedRateDecision


class TestUnifiedRateLimiter:
    def test_check_allows_normal_request(self):
        rl = RateLimiter()
        bp = BackpressureController(BackpressureConfig(tokens_per_second=1000, max_burst=1000))
        url = UnifiedRateLimiter(rl, bp)
        decision = url.check("client1", "/api/chat")
        assert decision.allowed is True

    def test_check_blocks_rate_limited_client(self):
        rl = RateLimiter(default_limit=RateLimit(requests_per_second=1, burst_size=1, requests_per_minute=1))
        rl.check_rate_limit("client1", "/api/chat")
        bp = BackpressureController(BackpressureConfig(tokens_per_second=1000, max_burst=1000))
        url = UnifiedRateLimiter(rl, bp)
        decision = url.check("client1", "/api/chat")
        assert decision.allowed is False
        assert decision.reason in ("rate_limit", "minute_limit")

    def test_backpressure_block(self):
        rl = RateLimiter()
        bp = BackpressureController(BackpressureConfig(tokens_per_second=1, max_burst=1))
        bp.acquire(tokens=1.0, timeout=0.1)
        url = UnifiedRateLimiter(rl, bp)
        decision = url.check("client1", "/api/chat")
        assert decision.allowed is False
        assert decision.reason == "backpressure"

    def test_release_frees_slots(self):
        rl = RateLimiter()
        bp = BackpressureController(BackpressureConfig(tokens_per_second=1000, max_burst=1000))
        url = UnifiedRateLimiter(rl, bp)
        url.check("client1", "/api/chat")
        url.release("client1")
        assert bp._active == 0

    def test_get_headers_allowed(self):
        rl = RateLimiter()
        bp = BackpressureController(BackpressureConfig(tokens_per_second=1000, max_burst=1000))
        url = UnifiedRateLimiter(rl, bp)
        decision = url.check("client1", "/api/chat")
        headers = url.get_headers(decision)
        assert "X-RateLimit-Remaining" in headers

    def test_get_headers_blocked(self):
        rl = RateLimiter()
        bp = BackpressureController(BackpressureConfig(tokens_per_second=1, max_burst=1))
        bp.acquire(tokens=1.0, timeout=0.1)
        url = UnifiedRateLimiter(rl, bp)
        decision = url.check("client1", "/api/chat")
        headers = url.get_headers(decision)
        assert "Retry-After" in headers
        assert "X-RateLimit-Reason" in headers

    def test_get_status(self):
        rl = RateLimiter()
        bp = BackpressureController()
        url = UnifiedRateLimiter(rl, bp)
        status = url.get_status()
        assert "rate_limiter" in status
        assert "backpressure" in status
        assert "stats" in status

    def test_stats_tracking(self):
        rl = RateLimiter()
        bp = BackpressureController(BackpressureConfig(tokens_per_second=1000, max_burst=1000))
        url = UnifiedRateLimiter(rl, bp)
        url.check("c1", "/")
        url.check("c2", "/")
        status = url.get_status()
        assert status["stats"]["total_checked"] == 2
        assert status["stats"]["allowed"] == 2

    def test_reset_all(self):
        rl = RateLimiter()
        bp = BackpressureController(BackpressureConfig(tokens_per_second=1, max_burst=1))
        url = UnifiedRateLimiter(rl, bp)
        url.check("c1", "/")
        url.reset_all()
        bp_status = bp.get_status()
        assert bp_status["tokens_available"] > 0

    def test_thread_safety(self):
        rl = RateLimiter()
        bp = BackpressureController(BackpressureConfig(tokens_per_second=10000, max_burst=10000))
        url = UnifiedRateLimiter(rl, bp)
        allowed_count = [0]
        blocked_count = [0]

        def worker(tid):
            for i in range(10):
                d = url.check(f"t{tid}", "/api/chat")
                if d.allowed:
                    allowed_count[0] += 1
                    url.release(f"t{tid}")
                else:
                    blocked_count[0] += 1

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert allowed_count[0] + blocked_count[0] == 50

    def test_endpoint_specific_limits(self):
        rl = RateLimiter()
        rl.set_endpoint_limit("/api/expensive", RateLimit(
            requests_per_second=1, requests_per_minute=1, burst_size=1, concurrent_requests=1
        ))
        bp = BackpressureController(BackpressureConfig(tokens_per_second=1000, max_burst=1000))
        url = UnifiedRateLimiter(rl, bp)
        d1 = url.check("c1", "/api/expensive")
        assert d1.allowed is True
        url.check("c1", "/api/expensive")
        d2 = url.check("c1", "/api/expensive")
        assert d2.allowed is False

    def test_client_isolation(self):
        rl = RateLimiter()
        bp = BackpressureController(BackpressureConfig(tokens_per_second=1000, max_burst=1000))
        url = UnifiedRateLimiter(rl, bp)
        d1 = url.check("client_a", "/api/chat")
        d2 = url.check("client_b", "/api/chat")
        assert d1.allowed is True
        assert d2.allowed is True

    def test_decision_has_endpoint_and_client(self):
        rl = RateLimiter()
        bp = BackpressureController(BackpressureConfig(tokens_per_second=1000, max_burst=1000))
        url = UnifiedRateLimiter(rl, bp)
        d = url.check("bob", "/api/predict")
        assert d.endpoint == "/api/predict"
        assert d.client_id == "bob"


class TestUnifiedRateDecision:
    def test_default_disallowed(self):
        d = UnifiedRateDecision(allowed=False, reason="test")
        assert d.allowed is False
        assert d.reason == "test"

    def test_default_allowed(self):
        d = UnifiedRateDecision(allowed=True)
        assert d.allowed is True
        assert d.retry_after == 0.0


class TestGetUnifiedRateLimiter:
    def test_singleton(self):
        from core.unified_rate_limiter import _unified
        a = get_unified_rate_limiter()
        b = get_unified_rate_limiter()
        assert a is b