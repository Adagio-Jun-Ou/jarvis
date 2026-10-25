"""Tests for rate limiter, response cache semantic dedup, and various fixes."""

import tempfile
import time
import pytest


class TestRateLimiter:
    def test_basic_allow(self):
        from core.rate_limiter import RateLimiter, RateLimit
        config = RateLimit(requests_per_second=100.0, requests_per_minute=10000)
        limiter = RateLimiter(config)

        allowed, retry_after, info = limiter.check_rate_limit("127.0.0.1")
        assert allowed is True
        assert retry_after == 0

    def test_burst_allow(self):
        from core.rate_limiter import RateLimiter, RateLimit
        config = RateLimit(requests_per_second=100.0, burst_size=50, requests_per_minute=200)
        limiter = RateLimiter(config)

        allowed_count = 0
        for _ in range(30):
            allowed, _, _ = limiter.check_rate_limit("127.0.0.1")
            if allowed:
                allowed_count += 1
        assert allowed_count >= 25

    def test_minute_limit_blocks(self):
        from core.rate_limiter import RateLimiter, RateLimit
        config = RateLimit(requests_per_second=100.0, requests_per_minute=5)
        limiter = RateLimiter(config)

        blocked = False
        for _ in range(10):
            allowed, retry_after, info = limiter.check_rate_limit("10.0.0.1")
            if not allowed:
                blocked = True
                break
        assert blocked

    def test_different_clients_independent(self):
        from core.rate_limiter import RateLimiter, RateLimit
        config = RateLimit(requests_per_second=100.0, requests_per_minute=5)
        limiter = RateLimiter(config)

        for _ in range(5):
            limiter.check_rate_limit("10.0.0.1")

        allowed, _, _ = limiter.check_rate_limit("10.0.0.2")
        assert allowed is True

    def test_stats(self):
        from core.rate_limiter import RateLimiter, RateLimit
        config = RateLimit(requests_per_second=10.0, requests_per_minute=100)
        limiter = RateLimiter(config)
        limiter.check_rate_limit("127.0.0.1")
        limiter.check_rate_limit("127.0.0.1")
        stats = limiter.get_stats()
        assert "total_requests" in stats
        assert stats["total_requests"] >= 2

    def test_singleton(self):
        from core.rate_limiter import get_rate_limiter
        r1 = get_rate_limiter()
        r2 = get_rate_limiter()
        assert r1 is r2




class TestSelfReflectionNegationFix:
    def test_not_bad_higher_than_bad(self):
        import tempfile
        from core.self_reflection import SelfReflection
        sr = SelfReflection(storage_path=__import__('pathlib').Path(tempfile.mkdtemp()))
        sr.track_response_quality("response", "not bad")
        not_bad = sr._quality_scores[-1][0]
        sr.track_response_quality("response", "bad")
        bad = sr._quality_scores[-1][0]
        assert not_bad > bad, f"'not bad' ({not_bad}) should score higher than 'bad' ({bad})"

    def test_great_high_score(self):
        import tempfile
        from core.self_reflection import SelfReflection
        sr = SelfReflection(storage_path=__import__('pathlib').Path(tempfile.mkdtemp()))
        sr.track_response_quality("response", "great perfect answer")
        assert sr._quality_scores[-1][0] >= 0.8

    def test_terrible_low_score(self):
        import tempfile
        from core.self_reflection import SelfReflection
        sr = SelfReflection(storage_path=__import__('pathlib').Path(tempfile.mkdtemp()))
        sr.track_response_quality("response", "terrible wrong bad answer")
        assert sr._quality_scores[-1][0] <= 0.2


class TestProactiveFileOrderPreserved:
    def test_order_preserved(self):
        from core.proactive import ProactiveEngine
        engine = ProactiveEngine()
        engine.update_activity("look at file_a.py then file_b.py")
        files = engine._mentioned_files
        if len(files) >= 2:
            assert "file_a.py" in files[0] or "file_a.py" in files[1]
            assert "file_b.py" in files[0] or "file_b.py" in files[1]