"""Tests for per-endpoint rate limiting."""
import os
import tempfile
import time
import threading

import pytest

os.environ.setdefault("JARVIS_HOME", tempfile.mkdtemp())

from core.rate_limiter import (
    RateLimit,
    RateLimiter,
    ENDPOINT_LIMITS,
    configure_endpoint_limits,
    get_rate_limiter,
)


class TestEndpointRateLimits:
    def test_endpoint_limits_defined(self):
        assert "/api/chat" in ENDPOINT_LIMITS
        assert "/api/predict" in ENDPOINT_LIMITS
        assert "/api/execute" in ENDPOINT_LIMITS
        assert "/api/health" in ENDPOINT_LIMITS

    def test_chat_limits_are_strict(self):
        chat_limit = ENDPOINT_LIMITS["/api/chat"]
        assert chat_limit.requests_per_second <= 10
        assert chat_limit.concurrent_requests <= 10

    def test_execute_limits_are_very_strict(self):
        exec_limit = ENDPOINT_LIMITS["/api/execute"]
        assert exec_limit.requests_per_second <= 2
        assert exec_limit.concurrent_requests <= 3

    def test_health_limits_are_relaxed(self):
        health_limit = ENDPOINT_LIMITS["/api/health"]
        assert health_limit.requests_per_second >= 5

    def test_configure_endpoint_limits(self):
        limiter = RateLimiter()
        configured = configure_endpoint_limits(limiter)
        assert "/api/chat" in configured
        assert "/api/predict" in configured

    def test_endpoint_limit_enforcement(self):
        limiter = RateLimiter()
        limiter.set_endpoint_limit("/api/test", RateLimit(
            requests_per_second=1.0,
            requests_per_minute=3,
            requests_per_hour=10,
            burst_size=2,
            concurrent_requests=1,
        ))
        allowed1, _, _ = limiter.check_rate_limit("client1", "/api/test")
        assert allowed1
        allowed2, _, _ = limiter.check_rate_limit("client1", "/api/test")
        allowed3, _, _ = limiter.check_rate_limit("client1", "/api/test")
        allowed4, _, _ = limiter.check_rate_limit("client1", "/api/test")
        assert not allowed4

    def test_different_endpoints_independent(self):
        limiter = RateLimiter()
        limiter.set_endpoint_limit("/slow", RateLimit(
            requests_per_second=1.0,
            requests_per_minute=2,
            requests_per_hour=10,
            burst_size=1,
            concurrent_requests=1,
        ))
        limiter.set_endpoint_limit("/fast", RateLimit(
            requests_per_second=100.0,
            requests_per_minute=1000,
            requests_per_hour=10000,
            burst_size=100,
            concurrent_requests=100,
        ))
        allowed_slow, _, _ = limiter.check_rate_limit("c1", "/slow")
        allowed_fast, _, _ = limiter.check_rate_limit("c2", "/fast")
        assert allowed_slow
        assert allowed_fast

    def test_endpoint_limit_defaults_to_global(self):
        limiter = RateLimiter()
        allowed, _, _ = limiter.check_rate_limit("client1", "/api/unknown_endpoint")
        assert allowed


class TestRateLimiterThreadSafety:
    def test_concurrent_checks(self):
        limiter = RateLimiter()
        limiter.set_endpoint_limit("/api/test", RateLimit(
            requests_per_second=100.0,
            requests_per_minute=1000,
            requests_per_hour=10000,
            burst_size=500,
            concurrent_requests=100,
        ))
        results = []
        errors = []

        def check():
            try:
                for _ in range(50):
                    allowed, _, _ = limiter.check_rate_limit(
                        f"client_{threading.current_thread().name}",
                        "/api/test",
                    )
                    results.append(allowed)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=check, name=f"t{i}") for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(results) == 200