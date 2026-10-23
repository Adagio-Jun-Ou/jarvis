"""Tests for agent/harness/sync_bridge.py.

The failure this exists to prevent is silent: ``asyncio.run`` inside a
thread that already has a loop raises, the coroutine is never awaited,
and the turn dies with a confusing RuntimeError instead of doing the
work. So the tests drive it from every context the VM is actually
reached from.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from agent.harness.sync_bridge import loop_is_running, run_coroutine


async def add(a: int, b: int) -> int:
    await asyncio.sleep(0)
    return a + b


async def boom() -> None:
    await asyncio.sleep(0)
    raise ValueError("inner failure")


def test_runs_from_a_plain_thread():
    assert not loop_is_running()
    assert run_coroutine(add(2, 3)) == 5


def test_runs_from_inside_a_running_loop():
    """The case that used to raise `cannot be called from a running
    event loop` and leave the coroutine unawaited."""

    async def outer():
        assert loop_is_running()
        # Blocking inside a loop is exactly what the VM's synchronous
        # surface does when a route calls it.
        return await asyncio.to_thread(run_coroutine, add(20, 22))

    assert asyncio.run(outer()) == 42


def test_runs_from_a_worker_thread_spawned_by_a_loop():
    """The task-board / standing-order shape: a route on the loop spawns
    a thread, and the thread drives a turn."""
    result: dict = {}

    async def outer():
        def work():
            result["value"] = run_coroutine(add(1, 1))

        t = threading.Thread(target=work)
        t.start()
        t.join(timeout=10)

    asyncio.run(outer())
    assert result["value"] == 2


def test_an_exception_propagates_to_the_caller():
    with pytest.raises(ValueError, match="inner failure"):
        run_coroutine(boom())


def test_an_exception_propagates_from_inside_a_loop():
    async def outer():
        return await asyncio.to_thread(run_coroutine, boom())

    with pytest.raises(ValueError, match="inner failure"):
        asyncio.run(outer())


def test_loop_is_running_reports_the_calling_thread():
    assert loop_is_running() is False

    async def inner():
        return loop_is_running()

    assert asyncio.run(inner()) is True


def test_nested_use_still_completes():
    """A tool call inside a turn can reach the bridge again."""

    async def inner_work():
        return await asyncio.to_thread(run_coroutine, add(3, 4))

    assert run_coroutine(inner_work()) == 7
