"""Tests for the retry helper."""
from __future__ import annotations

import pytest

from alex.retry import _compute_delay, call_with_retry


def test_success_first_try():
    calls = {"n": 0}
    def f():
        calls["n"] += 1
        return 42
    assert call_with_retry(f, attempts=3) == 42
    assert calls["n"] == 1


def test_success_after_retries():
    calls = {"n": 0}
    def f():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("nope")
        return "ok"
    assert call_with_retry(f, attempts=5, base_delay_seconds=0.001) == "ok"
    assert calls["n"] == 3


def test_exhausted_raises():
    def f():
        raise ValueError("always")
    with pytest.raises(ValueError, match="always"):
        call_with_retry(f, attempts=3, base_delay_seconds=0.001)


def test_delay_never_exceeds_max_even_with_full_jitter(monkeypatch):
    # random()==1.0 is the worst case (jitter x1.3). The clamp must still
    # keep the delay at or below the documented ceiling.
    import alex.retry as retry
    monkeypatch.setattr(retry.random, "random", lambda: 1.0)
    for attempt in range(1, 8):
        d = _compute_delay(attempt, base_delay=0.5, max_delay=8.0)
        assert d <= 8.0


def test_delay_grows_but_is_bounded():
    lo = _compute_delay(1, base_delay=0.5, max_delay=8.0)
    hi = _compute_delay(5, base_delay=0.5, max_delay=8.0)
    assert 0 < lo <= 8.0
    assert 0 < hi <= 8.0


def test_non_retriable_propagates_immediately():
    calls = {"n": 0}
    def f():
        calls["n"] += 1
        raise TypeError("not retriable")
    with pytest.raises(TypeError):
        call_with_retry(
            f, attempts=5, base_delay_seconds=0.001,
            retriable=(ValueError,),
        )
    assert calls["n"] == 1
