"""Async helpers that avoid default-executor shutdown edge cases.

Adapted from ``omnigent/inner/async_utils.py`` (Apache-2.0).  Used by
:mod:`agent.omnigent.databricks_executor` and
:mod:`agent.omnigent.local_ollama_executor` to bridge blocking SDK
calls into the async event loop without using
``loop.run_in_executor`` (whose default thread pool can hang on
shutdown in tests that repeatedly create/close loops).
"""

from __future__ import annotations

import asyncio
import queue as sync_queue
import threading
from collections.abc import Callable
from typing import Any


async def aiter_any(it: Any):
    """Async-iterate over either an async iterable or a plain sync
    iterable/generator.  Sync iterables are drained on a background
    thread so a blocking LLM stream never stalls the event loop.

    The pump is cooperative: when the consumer abandons the generator
    (ExecutorError break, client disconnect), the finally block sets a
    stop event and closes the underlying iterator. Without it, the
    pump thread blocked forever on a full queue and the LLM HTTP
    stream stayed open — one leaked thread + socket per cancelled
    turn until the socket timeout.
    """
    if hasattr(it, "__aiter__"):
        async for item in it:
            yield item
        return

    _end = object()
    q: sync_queue.Queue[tuple[str, Any]] = sync_queue.Queue(maxsize=64)
    stop = threading.Event()

    def _pump() -> None:
        try:
            for item in it:
                if stop.is_set():
                    break
                try:
                    q.put(("item", item), timeout=0.05)
                except sync_queue.Full:
                    continue
        except BaseException as exc:
            if not stop.is_set():
                try:
                    q.put(("error", exc), timeout=0.05)
                except sync_queue.Full:
                    pass
        finally:
            if not stop.is_set():
                try:
                    q.put(("end", _end), timeout=0.05)
                except sync_queue.Full:
                    pass
            close = getattr(it, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001
                    pass

    thread = threading.Thread(target=_pump, daemon=True)
    thread.start()
    try:
        while True:
            try:
                kind, payload = q.get_nowait()
            except sync_queue.Empty:
                await asyncio.sleep(0.001)
                continue
            if kind == "error":
                raise payload
            if kind == "end":
                return
            yield payload
    finally:
        stop.set()
        thread.join(timeout=2)


async def run_sync_on_thread(
    fn: Callable[..., Any],
    /,
    *args: Any,
    **kwargs: Any,
) -> Any:
    result_queue: sync_queue.Queue[tuple[str, Any]] = sync_queue.Queue()

    def _runner() -> None:
        try:
            result = fn(*args, **kwargs)
        except BaseException as exc:
            result_queue.put(("error", exc))
            return
        result_queue.put(("result", result))

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    try:
        while True:
            try:
                kind, payload = result_queue.get_nowait()
            except sync_queue.Empty:
                await asyncio.sleep(0.001)
                continue
            if kind == "error":
                raise payload
            return payload
    finally:
        thread.join(timeout=0)
