"""Rate limiting for API endpoints.

Provides:
- Token bucket algorithm for smooth rate limiting
- Per-client and global rate limits
- Sliding window limits
- Configurable per-endpoint limits
"""
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class RateLimitType(Enum):
    """Types of rate limits."""
    REQUESTS = "requests"
    TOKENS = "tokens"
    CONCURRENT = "concurrent"


@dataclass
class RateLimit:
    """Rate limit configuration."""
    requests_per_second: float = 100.0
    requests_per_minute: int = 10000
    requests_per_hour: int = 100000
    burst_size: int = 500
    concurrent_requests: int = 100


@dataclass
class ClientState:
    """Rate limit state for a client."""
    tokens: float = 0.0
    last_update: float = 0.0
    minute_count: int = 0
    minute_start: float = 0.0
    hour_count: int = 0
    hour_start: float = 0.0
    concurrent: int = 0
    blocked_until: float = 0.0


class RateLimiter:
    """Token bucket rate limiter with sliding windows.
    
    Features:
    - Token bucket for smooth per-second limiting
    - Sliding window for per-minute/hour limits
    - Concurrent request limiting
    - Automatic cleanup of old clients
    """
    
    def __init__(
        self,
        default_limit: RateLimit | None = None,
        max_clients: int = 1000,
        cleanup_interval: float = 60.0,
        max_blocked_ips: int = 1000,  # Limit blocked IPs
    ):
        self.default_limit = default_limit or RateLimit()
        self.max_clients = max_clients
        self.cleanup_interval = cleanup_interval
        self.max_blocked_ips = max_blocked_ips
        
        self._clients: OrderedDict[str, ClientState] = OrderedDict()
        self._lock = threading.RLock()
        self._endpoint_limits: dict[str, RateLimit] = {}
        self._blocked_ips: OrderedDict[str, None] = OrderedDict()  # Use OrderedDict for FIFO
        self._stats = {
            "total_requests": 0,
            "blocked_requests": 0,
            "active_clients": 0,
        }
    
    def set_endpoint_limit(self, endpoint: str, limit: RateLimit) -> None:
        """Set a specific rate limit for an endpoint."""
        with self._lock:
            self._endpoint_limits[endpoint] = limit
    
    def block_ip(self, ip: str) -> None:
        """Block an IP address."""
        with self._lock:
            # Use OrderedDict for FIFO eviction
            if ip in self._blocked_ips:
                del self._blocked_ips[ip]  # Remove if exists (re-add to end)
            self._blocked_ips[ip] = None
            # Enforce limit (remove oldest)
            while len(self._blocked_ips) > self.max_blocked_ips:
                self._blocked_ips.popitem(last=False)
        logger.warning(f"Blocked IP: {ip}")
    
    def unblock_ip(self, ip: str) -> None:
        """Unblock an IP address."""
        with self._lock:
            self._blocked_ips.pop(ip, None)
    
    def get_blocked_ips(self) -> set[str]:
        """Get copy of blocked IPs."""
        with self._lock:
            return set(self._blocked_ips.keys())
    
    def _get_or_create_client(self, client_id: str) -> ClientState:
        """Get or create client state, with LRU eviction."""
        if client_id in self._clients:
            self._clients.move_to_end(client_id)
            return self._clients[client_id]
        
        while len(self._clients) >= self.max_clients:
            self._clients.popitem(last=False)
        
        now = time.time()
        state = ClientState(
            tokens=float(self.default_limit.burst_size),
            last_update=now,
            minute_start=now,
            hour_start=now,
        )
        self._clients[client_id] = state
        return state
    
    def _cleanup_old_clients(self) -> None:
        """Remove clients that haven't been seen recently."""
        now = time.time()
        max_age = 3600  # 1 hour
        
        to_remove = [
            client_id for client_id, state in self._clients.items()
            if now - state.last_update > max_age
        ]
        
        for client_id in to_remove:
            del self._clients[client_id]
        
        if to_remove:
            logger.debug(f"Cleaned up {len(to_remove)} old rate limit clients")
    
    def _maybe_cleanup(self) -> None:
        """Periodically cleanup old clients (call occasionally)."""
        # Cleanup every ~1000 calls
        if self._stats["total_requests"] % 1000 == 0:
            self._cleanup_old_clients()
    
    def check_rate_limit(
        self,
        client_id: str,
        endpoint: str | None = None,
        cost: int = 1,
    ) -> tuple[bool, float, dict[str, Any]]:
        """Check if a request is allowed.
        
        Args:
            client_id: Client identifier (IP, user ID, etc.)
            endpoint: Optional endpoint for specific limits
            cost: Cost of this request (default 1)
            
        Returns:
            Tuple of (allowed, retry_after_seconds, info_dict)
        """
        with self._lock:
            self._stats["total_requests"] += 1
            
            # Periodic cleanup
            self._maybe_cleanup()
            
            if client_id in self._blocked_ips:
                self._stats["blocked_requests"] += 1
                return False, 60.0, {"reason": "blocked_ip"}
            
            if endpoint is not None:
                limit = self._endpoint_limits.get(endpoint)
            else:
                limit = None
            if limit is None:
                limit = self.default_limit
            
            state = self._get_or_create_client(client_id)
            now = time.time()
            
            if now < state.blocked_until:
                retry_after = state.blocked_until - now
                self._stats["blocked_requests"] += 1
                return False, retry_after, {"reason": "blocked", "retry_after": retry_after}
            
            tokens_to_add = (now - state.last_update) * limit.requests_per_second
            state.tokens = min(
                state.tokens + tokens_to_add,
                limit.burst_size
            )
            state.last_update = now
            
            if now - state.minute_start >= 60:
                state.minute_count = 0
                state.minute_start = now
            
            if now - state.hour_start >= 3600:
                state.hour_count = 0
                state.hour_start = now
            
            info = {
                "tokens_remaining": state.tokens,
                "minute_remaining": limit.requests_per_minute - state.minute_count,
                "hour_remaining": limit.requests_per_hour - state.hour_count,
                "concurrent": state.concurrent,
            }
            
            if state.concurrent >= limit.concurrent_requests:
                self._stats["blocked_requests"] += 1
                return False, 1.0, {"reason": "concurrent_limit", **info}
            
            if state.tokens < cost:
                retry_after = (cost - state.tokens) / limit.requests_per_second
                self._stats["blocked_requests"] += 1
                return False, retry_after, {"reason": "rate_limit", **info}
            
            if state.minute_count >= limit.requests_per_minute:
                retry_after = 60 - (now - state.minute_start)
                self._stats["blocked_requests"] += 1
                return False, retry_after, {"reason": "minute_limit", **info}
            
            if state.hour_count >= limit.requests_per_hour:
                retry_after = 3600 - (now - state.hour_start)
                self._stats["blocked_requests"] += 1
                return False, retry_after, {"reason": "hour_limit", **info}
            
            state.tokens -= cost
            state.minute_count += 1
            state.hour_count += 1
            state.concurrent += 1
            
            return True, 0.0, info
    
    def release(self, client_id: str) -> None:
        """Release a concurrent request slot."""
        with self._lock:
            if client_id in self._clients:
                state = self._clients[client_id]
                state.concurrent = max(0, state.concurrent - 1)
    
    def get_stats(self) -> dict[str, Any]:
        """Get rate limiter statistics."""
        with self._lock:
            return {
                **dict(self._stats),
                "active_clients": len(self._clients),
                "blocked_ips": len(self._blocked_ips),
                "endpoint_limits": list(self._endpoint_limits.keys()),
            }
    
    def reset_client(self, client_id: str) -> None:
        """Reset rate limit state for a client."""
        with self._lock:
            if client_id in self._clients:
                del self._clients[client_id]
    
    def reset_all(self) -> None:
        """Reset all rate limit state."""
        with self._lock:
            self._clients.clear()
            self._blocked_ips.clear()
            self._stats = {
                "total_requests": 0,
                "blocked_requests": 0,
                "active_clients": 0,
            }


def rate_limit_middleware(
    limiter: RateLimiter,
    get_client_id: Callable[[Any], str],
    endpoint: str | None = None,
):
    """Create a rate limiting middleware function.
    
    Usage with FastAPI:
        @app.middleware("http")
        async def rate_limit(request: Request, call_next):
            client_id = request.client.host if request.client else "unknown"
            allowed, retry_after, info = limiter.check_rate_limit(client_id)
            
            if not allowed:
                return JSONResponse(
                    status_code=429,
                    content={"error": "Rate limit exceeded", "retry_after": retry_after}
                )
            
            try:
                return await call_next(request)
            finally:
                limiter.release(client_id)
    """
    def middleware(request, call_next):
        client_id = get_client_id(request)
        allowed, retry_after, info = limiter.check_rate_limit(client_id, endpoint)
        
        if not allowed:
            return {
                "status_code": 429,
                "body": {
                    "error": "Rate limit exceeded",
                    "retry_after": retry_after,
                    "info": info,
                },
            }
        
        return None
    
    return middleware


_default_limiter: RateLimiter | None = None
_limiter_lock = threading.Lock()


def get_rate_limiter() -> RateLimiter:
    """Get the global rate limiter instance."""
    global _default_limiter
    if _default_limiter is None:
        with _limiter_lock:
            if _default_limiter is None:
                _default_limiter = RateLimiter()
    return _default_limiter


ENDPOINT_LIMITS = {
    "/api/chat": RateLimit(
        requests_per_second=5.0,
        requests_per_minute=60,
        requests_per_hour=500,
        burst_size=10,
        concurrent_requests=5,
    ),
    "/api/predict": RateLimit(
        requests_per_second=2.0,
        requests_per_minute=30,
        requests_per_hour=200,
        burst_size=5,
        concurrent_requests=3,
    ),
    "/api/diagnose": RateLimit(
        requests_per_second=1.0,
        requests_per_minute=15,
        requests_per_hour=100,
        burst_size=3,
        concurrent_requests=2,
    ),
    "/api/analyze-image": RateLimit(
        requests_per_second=1.0,
        requests_per_minute=10,
        requests_per_hour=60,
        burst_size=3,
        concurrent_requests=2,
    ),
    "/api/execute": RateLimit(
        requests_per_second=1.0,
        requests_per_minute=20,
        requests_per_hour=100,
        burst_size=3,
        concurrent_requests=2,
    ),
    "/api/terminal/cwd": RateLimit(
        requests_per_second=2.0,
        requests_per_minute=30,
        requests_per_hour=200,
        burst_size=5,
        concurrent_requests=3,
    ),
    "/api/oracle": RateLimit(
        requests_per_second=1.0,
        requests_per_minute=15,
        requests_per_hour=100,
        burst_size=3,
        concurrent_requests=2,
    ),
    "/api/portfolio": RateLimit(
        requests_per_second=1.0,
        requests_per_minute=15,
        requests_per_hour=100,
        burst_size=3,
        concurrent_requests=2,
    ),
    "/api/options": RateLimit(
        requests_per_second=2.0,
        requests_per_minute=30,
        requests_per_hour=200,
        burst_size=5,
        concurrent_requests=3,
    ),
    "/api/router": RateLimit(
        requests_per_second=2.0,
        requests_per_minute=30,
        requests_per_hour=200,
        burst_size=5,
        concurrent_requests=3,
    ),
    "/api/cursor": RateLimit(
        requests_per_second=10.0,
        requests_per_minute=120,
        requests_per_hour=1000,
        burst_size=20,
        concurrent_requests=10,
    ),
    "/api/health": RateLimit(
        requests_per_second=10.0,
        requests_per_minute=300,
        requests_per_hour=5000,
        burst_size=30,
        concurrent_requests=20,
    ),
}


def configure_endpoint_limits(limiter: RateLimiter | None = None) -> dict[str, str]:
    """Apply per-endpoint rate limits to the global rate limiter.

    Returns mapping of endpoints to their configured limits.
    """
    rl = limiter or get_rate_limiter()
    configured = {}
    for endpoint, limit in ENDPOINT_LIMITS.items():
        rl.set_endpoint_limit(endpoint, limit)
        configured[endpoint] = (
            f"{limit.requests_per_second}/s, {limit.requests_per_minute}/min, "
            f"{limit.requests_per_hour}/hr, burst={limit.burst_size}, "
            f"concurrent={limit.concurrent_requests}"
        )
    return configured