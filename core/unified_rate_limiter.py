"""Unified Rate Limiting — bridges RateLimiter + BackpressureController + web middleware.

Provides:
- UnifiedRateLimiter: single entry point for all rate decisions (per-endpoint, per-client, backpressure)
- RateLimitMiddleware: drop-in Starlette/FastAPI middleware
- Rate limit headers in responses (X-RateLimit-*)
- Streaming pipeline rate limit integration
"""
import logging
import threading
from dataclasses import dataclass
from typing import Any

from core.rate_limiter import RateLimiter, get_rate_limiter
from core.streaming_pipeline import BackpressureController, BackpressureConfig

logger = logging.getLogger(__name__)


@dataclass
class UnifiedRateDecision:
    allowed: bool
    retry_after: float = 0.0
    reason: str = ""
    tokens_remaining: float = 0.0
    minute_remaining: int = 0
    hour_remaining: int = 0
    concurrent_active: int = 0
    endpoint: str = ""
    client_id: str = ""


class UnifiedRateLimiter:
    """Unified rate limiter combining endpoint, client, and backpressure.

    Decision flow:
    1. Check backpressure (global throttling)
    2. Check client IP block
    3. Check per-endpoint rate limits
    4. Return unified decision
    """

    def __init__(
        self,
        rate_limiter: RateLimiter | None = None,
        backpressure: BackpressureController | None = None,
        backpressure_config: BackpressureConfig | None = None,
    ):
        self._limiter = rate_limiter or get_rate_limiter()
        self._bp = backpressure or BackpressureController(backpressure_config)
        self._lock = threading.Lock()
        self._request_times: dict[str, list[float]] = {}
        self._stats = {
            "total_checked": 0,
            "backpressure_rejects": 0,
            "rate_limit_rejects": 0,
            "allowed": 0,
        }

    def check(
        self,
        client_id: str,
        endpoint: str = "/",
        cost: float = 1.0,
    ) -> UnifiedRateDecision:
        self._stats["total_checked"] += 1

        bp_ok = self._bp.acquire(tokens=cost * 0.1, timeout=0.01)
        if not bp_ok:
            self._stats["backpressure_rejects"] += 1
            return UnifiedRateDecision(
                allowed=False,
                retry_after=0.1,
                reason="backpressure",
                endpoint=endpoint,
                client_id=client_id,
            )

        ok, retry_after, info = self._limiter.check_rate_limit(
            client_id, endpoint, cost=int(cost)
        )

        if not ok:
            self._stats["rate_limit_rejects"] += 1
            return UnifiedRateDecision(
                allowed=False,
                retry_after=retry_after,
                reason=info.get("reason", "rate_limit"),
                tokens_remaining=info.get("tokens_remaining", 0),
                minute_remaining=info.get("minute_remaining", 0),
                hour_remaining=info.get("hour_remaining", 0),
                concurrent_active=info.get("concurrent", 0),
                endpoint=endpoint,
                client_id=client_id,
            )

        self._stats["allowed"] += 1
        return UnifiedRateDecision(
            allowed=True,
            tokens_remaining=info.get("tokens_remaining", 0),
            minute_remaining=info.get("minute_remaining", 0),
            hour_remaining=info.get("hour_remaining", 0),
            concurrent_active=info.get("concurrent", 0),
            endpoint=endpoint,
            client_id=client_id,
        )

    def release(self, client_id: str) -> None:
        self._limiter.release(client_id)
        self._bp.release()

    def get_headers(self, decision: UnifiedRateDecision) -> dict[str, str]:
        headers = {
            "X-RateLimit-Remaining": str(int(decision.tokens_remaining)),
            "X-RateLimit-Minute-Remaining": str(decision.minute_remaining),
            "X-RateLimit-Hour-Remaining": str(decision.hour_remaining),
        }
        if not decision.allowed:
            headers["Retry-After"] = str(int(decision.retry_after) + 1)
            headers["X-RateLimit-Reason"] = decision.reason
        return headers

    def get_status(self) -> dict[str, Any]:
        return {
            "rate_limiter": self._limiter.get_stats(),
            "backpressure": self._bp.get_status(),
            "stats": dict(self._stats),
        }

    def reset_all(self) -> None:
        self._limiter.reset_all()
        self._bp._tokens = self._bp._max_burst


_unified: UnifiedRateLimiter | None = None
_unified_lock = threading.Lock()


def get_unified_rate_limiter(
    rate_limiter: RateLimiter | None = None,
    backpressure: BackpressureController | None = None,
) -> UnifiedRateLimiter:
    global _unified
    if _unified is None:
        with _unified_lock:
            if _unified is None:
                _unified = UnifiedRateLimiter(rate_limiter, backpressure)
    return _unified