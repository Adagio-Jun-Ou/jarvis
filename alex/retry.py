"""
Retry / backoff helpers shared by the pipelines and the agent's external calls.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from typing import TypeVar

log = logging.getLogger("alex.retry")

T = TypeVar("T")


def _compute_delay(attempt: int, base_delay: float, max_delay: float) -> float:
    """Exponential backoff with jitter, clamped so the result never exceeds
    max_delay. The jitter (up to x1.3) is applied *before* the final clamp;
    applying the cap only before jitter let a max_delay of 8s produce ~10.4s
    sleeps, silently violating the documented ceiling."""
    raw = min(max_delay, base_delay * (2 ** (attempt - 1)))
    jittered = raw * (0.7 + 0.6 * random.random())
    return min(max_delay, jittered)


def call_with_retry(
    fn: Callable[[], T],
    *,
    attempts: int = 4,
    base_delay_seconds: float = 0.5,
    max_delay_seconds: float = 8.0,
    retriable: tuple[type[BaseException], ...] = (Exception,),
    on_failure: Callable[[BaseException, int], None] | None = None,
) -> T:
    """Call `fn` with exponential backoff + jitter. Re-raise on final attempt."""
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    last: BaseException | None = None
    for i in range(1, attempts + 1):
        try:
            return fn()
        except retriable as e:
            last = e
            if i == attempts:
                break
            delay = _compute_delay(i, base_delay_seconds, max_delay_seconds)
            log.warning(
                "attempt %d/%d failed: %s; retrying in %.2fs", i, attempts, e, delay
            )
            if on_failure:
                try:
                    on_failure(e, i)
                except Exception:  # pragma: no cover
                    pass
            time.sleep(delay)
    assert last is not None
    raise last
