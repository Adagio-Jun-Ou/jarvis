"""Circuit breaker for resilient LLM calls.

Implements the circuit breaker pattern to prevent cascading failures
when the LLM provider is experiencing issues.

States:
- CLOSED: Normal operation, requests pass through
- OPEN: Failing, requests are rejected immediately
- HALF_OPEN: Testing if service recovered, limited requests allowed
"""
import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Callable, Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitStats:
    """Statistics for a circuit breaker."""
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure: datetime | None = None
    last_success: datetime | None = None
    last_state_change: datetime | None = None
    total_requests: int = 0
    total_failures: int = 0
    total_successes: int = 0
    rejected_requests: int = 0
    
    def to_dict(self) -> dict:
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure": self.last_failure.isoformat() if self.last_failure else None,
            "last_success": self.last_success.isoformat() if self.last_success else None,
            "last_state_change": self.last_state_change.isoformat() if self.last_state_change else None,
            "total_requests": self.total_requests,
            "total_failures": self.total_failures,
            "total_successes": self.total_successes,
            "rejected_requests": self.rejected_requests,
        }


@dataclass
class CircuitConfig:
    """Configuration for a circuit breaker."""
    failure_threshold: int = 5          # Failures before opening
    success_threshold: int = 3          # Successes to close from half-open
    timeout_seconds: float = 60.0       # Time in OPEN before HALF_OPEN
    half_open_max_calls: int = 1        # Max calls allowed in HALF_OPEN
    reset_timeout_seconds: float = 300.0  # Reset counters after this time


class CircuitOpenError(Exception):
    """Raised when circuit is open."""
    pass


class CircuitBreaker(Generic[T]):
    """Circuit breaker for resilient operations.
    
    Usage:
        breaker = CircuitBreaker(
            name="llm_api",
            failure_threshold=5,
            timeout_seconds=60.0,
        )
        
        async def call_llm():
            return await breaker.call(llm_client.chat, prompt)
    """
    
    def __init__(
        self,
        name: str,
        config: CircuitConfig | None = None,
        fallback: Callable[[], T] | None = None,
    ):
        self.name = name
        self.config = config or CircuitConfig()
        self._fallback = fallback
        self._stats = CircuitStats()
        self._lock = threading.RLock()
        self._half_open_calls = 0
        # Monotonic stamp for state transitions — immune to wall-clock steps
        # (NTP corrections, manual changes). `stats.last_failure` remains a
        # wall-clock datetime for display.
        self._last_failure_time: float = 0

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            if self._stats.state == CircuitState.OPEN:
                if time.monotonic() - self._last_failure_time >= self.config.timeout_seconds:
                    self._transition_to(CircuitState.HALF_OPEN)
            return self._stats.state
    
    @property
    def stats(self) -> CircuitStats:
        """Get circuit statistics (thread-safe copy)."""
        with self._lock:
            return CircuitStats(
                state=self._stats.state,
                failure_count=self._stats.failure_count,
                success_count=self._stats.success_count,
                total_requests=self._stats.total_requests,
                total_failures=self._stats.total_failures,
                total_successes=self._stats.total_successes,
                last_failure=self._stats.last_failure,
                last_success=self._stats.last_success,
                last_state_change=self._stats.last_state_change,
                rejected_requests=self._stats.rejected_requests,
            )
    
    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        old_state = self._stats.state
        if old_state == new_state:
            return
        
        self._stats.state = new_state
        self._stats.last_state_change = datetime.now()
        
        if new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
        
        logger.info(f"Circuit [{self.name}] {old_state.value} -> {new_state.value}")
    
    def _record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            self._stats.success_count += 1
            self._stats.total_successes += 1
            self._stats.total_requests += 1
            self._stats.last_success = datetime.now()
            
            if self._stats.state == CircuitState.HALF_OPEN:
                if self._stats.success_count >= self.config.success_threshold:
                    self._stats.failure_count = 0
                    self._stats.success_count = 0
                    self._transition_to(CircuitState.CLOSED)
            elif self._stats.state == CircuitState.CLOSED:
                self._stats.failure_count = 0
    
    def record_success(self) -> None:
        """Public method to record a successful call."""
        self._record_success()
    
    def _record_failure(self, error: Exception) -> None:
        """Record a failed call."""
        with self._lock:
            self._stats.failure_count += 1
            self._stats.total_failures += 1
            self._stats.total_requests += 1
            self._stats.last_failure = datetime.now()
            self._last_failure_time = time.monotonic()
            
            if self._stats.state == CircuitState.HALF_OPEN:
                self._transition_to(CircuitState.OPEN)
            elif self._stats.state == CircuitState.CLOSED:
                if self._stats.failure_count >= self.config.failure_threshold:
                    self._transition_to(CircuitState.OPEN)
    
    def record_failure(self, error: Exception | None = None) -> None:
        """Public method to record a failed call."""
        self._record_failure(error or Exception("Unknown error"))
    
    def _can_execute(self) -> bool:
        """Check if a call can be executed."""
        with self._lock:
            state = self.state
            
            if state == CircuitState.CLOSED:
                return True
            
            if state == CircuitState.HALF_OPEN:
                if self._half_open_calls < self.config.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False
            
            return False
    
    async def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute a function with circuit breaker protection.
        
        Args:
            func: Async function to call
            *args: Arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result of the function
            
        Raises:
            CircuitOpenError: If circuit is open
        """
        if not self._can_execute():
            with self._lock:
                self._stats.rejected_requests += 1
            if self._fallback:
                logger.debug(f"Circuit [{self.name}] open, using fallback")
                return self._fallback()
            raise CircuitOpenError(f"Circuit [{self.name}] is open")
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                # Run synchronous function in executor to avoid blocking
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(None, lambda: func(*args, **kwargs))
            
            self._record_success()
            return result
            
        except Exception as e:
            self._record_failure(e)
            if self._fallback:
                logger.warning(f"Circuit [{self.name}] failure, using fallback: {e}")
                return self._fallback()
            raise
    
    def reset(self) -> None:
        """Reset the circuit breaker."""
        with self._lock:
            self._stats = CircuitStats()
            self._half_open_calls = 0
            self._last_failure_time = 0
            logger.info(f"Circuit [{self.name}] reset")
    
    def force_open(self) -> None:
        """Force the circuit open (for maintenance)."""
        with self._lock:
            self._transition_to(CircuitState.OPEN)
            self._last_failure_time = time.monotonic()
    
    def force_close(self) -> None:
        """Force the circuit closed."""
        with self._lock:
            self._stats.failure_count = 0
            self._transition_to(CircuitState.CLOSED)


class CircuitBreakerManager:
    """Manager for multiple circuit breakers.
    
    Provides a centralized registry for circuit breakers across
    different services.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __init__(self):
        self._breakers: dict[str, CircuitBreaker] = {}
        self._breaker_lock = threading.RLock()
    
    @classmethod
    def get_instance(cls) -> "CircuitBreakerManager":
        """Get the singleton manager."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def get_or_create(
        self,
        name: str,
        config: CircuitConfig | None = None,
        fallback: Callable | None = None,
    ) -> CircuitBreaker:
        """Get or create a circuit breaker."""
        with self._breaker_lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name, config, fallback)
            return self._breakers[name]
    
    def get(self, name: str) -> CircuitBreaker | None:
        """Get a circuit breaker by name."""
        with self._breaker_lock:
            return self._breakers.get(name)
    
    def get_all_stats(self) -> dict[str, dict]:
        """Get stats for all circuit breakers."""
        with self._breaker_lock:
            return {
                name: breaker.stats.to_dict()
                for name, breaker in self._breakers.items()
            }
    
    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        with self._breaker_lock:
            breakers = list(self._breakers.values())
        for breaker in breakers:
            breaker.reset()


def get_circuit_breaker(
    name: str,
    config: CircuitConfig | None = None,
    fallback: Callable | None = None,
) -> CircuitBreaker:
    """Get or create a circuit breaker."""
    return CircuitBreakerManager.get_instance().get_or_create(name, config, fallback)


def get_all_circuit_stats() -> dict[str, dict]:
    """Get stats for all circuit breakers."""
    return CircuitBreakerManager.get_instance().get_all_stats()