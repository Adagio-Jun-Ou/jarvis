"""
Running a coroutine from wherever the caller happens to be.

The VM's public surface is synchronous, but it is now reached from four
different contexts: a FastAPI route (running on the event loop), a
background task-board thread, a standing-order thread, and a tool call
nested inside a turn. ``asyncio.run`` only works in the first *and only*
if that thread has no loop of its own — anywhere else it raises
``RuntimeError: asyncio.run() cannot be called from a running event
loop`` and the coroutine is silently never awaited.

:func:`run_coroutine` removes that footgun: if the calling thread has a
running loop, the coroutine is executed on a short-lived worker thread
with its own loop, and the caller blocks on the result exactly as it
would have. Callers stay plain synchronous code and stop caring which
context they are in.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from typing import Any

log = logging.getLogger("agent.harness.sync_bridge")

DEFAULT_TIMEOUT_S = 900.0


def loop_is_running() -> bool:
    """Is the calling thread already inside an event loop?"""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def run_coroutine(coro, *, timeout: float | None = DEFAULT_TIMEOUT_S) -> Any:
    """Run ``coro`` to completion and return its result, from any thread.

    The coroutine is always awaited — including on the error path — so a
    caller can never leak a "coroutine was never awaited" warning and an
    unfinished turn.
    """
    if not loop_is_running():
        return asyncio.run(coro)

    # A loop is already spinning on this thread; hand the work to one
    # that is free. `max_workers=1` keeps this to a single short-lived
    # thread rather than a pool that outlives the call.
    #
    # NOT a `with` block: the context manager joins the worker on exit,
    # so a hung coroutine kept the caller blocked long PAST the timeout
    # — the timeout raised and then shutdown waited anyway, which is a
    # timeout that does not time out. On the timeout path the pool is
    # released without waiting; the daemonized worker finishes (or dies
    # with the process) on its own.
    pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="alex-sync-bridge"
    )
    future = pool.submit(asyncio.run, coro)
    try:
        result = future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        log.error(
            "sync-bridge coroutine exceeded %.0fs; abandoning the worker",
            timeout or 0.0,
        )
        pool.shutdown(wait=False, cancel_futures=True)
        raise TimeoutError(
            f"agent turn exceeded {timeout:.0f}s in the sync bridge"
        ) from None
    pool.shutdown(wait=True)
    return result


__all__ = ["run_coroutine", "loop_is_running"]
