"""
EventPump — the SSE event subscriber registry used by the VM.

The Alex 2.0 SSE protocol (see ``alex/events.py``) is a single ordered
stream of ``Event`` instances.  The :class:`VM` writes events to the
pump; the FastAPI ``/api/chat`` and ``/api/browser/sessions/{sid}/stream``
routes read from the pump to feed their ``StreamingResponse``.

Design:

* One pump per VM instance (one VM per user session).
* Subscribers are callables ``(Event) -> None``.  The FastAPI route
  registers a subscriber that pushes the event onto an
  ``asyncio.Queue``; the route's generator yields from the queue.
* Subscribers that raise are logged and removed so a single bad
  subscriber never wedges the VM.
* The pump is **synchronous** on the write side (so the agent loop
  doesn't have to be async) and the FastAPI route bridges the
  sync->async boundary with a small polling loop.

This is the *only* event sink in the system.  The audit log reads
the same event stream to write to ``media.alex.agent_actions``; the
agentic browser operations push through it for tool calls; the
constitution interceptor pushes ``law_block`` events through it.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

from alex.events import Event

logger = logging.getLogger("alex.harness.event_pump")

Subscriber = Callable[[Event], None]


class EventPump:
    """Thread-safe fan-out of :class:`Event` instances to subscribers.

    Subscribers are added with :meth:`subscribe` and removed with
    :meth:`unsubscribe`.  :meth:`emit` fans out to every subscriber
    in registration order; exceptions in one subscriber are logged
    and the subscriber is removed so the rest of the stream keeps
    flowing.
    """

    def __init__(self, *, name: str = "default") -> None:
        self._name = name
        self._lock = threading.Lock()
        self._subscribers: list[tuple[str, Subscriber]] = []
        self._buffer: list[Event] = []
        self._buffer_max = 1024

    @property
    def name(self) -> str:
        return self._name

    def subscribe(self, subscriber: Subscriber, *, tag: str = "") -> Callable[[], None]:
        """Register a subscriber. Returns an unsubscribe callable."""
        with self._lock:
            self._subscribers.append((tag, subscriber))

        def _unsub() -> None:
            self.unsubscribe(subscriber)

        return _unsub

    def unsubscribe(self, subscriber: Subscriber) -> None:
        with self._lock:
            self._subscribers = [
                (t, s) for (t, s) in self._subscribers if s is not subscriber
            ]

    def emit(self, event: Event) -> None:
        """Fan an event out to every subscriber. Thread-safe."""
        with self._lock:
            subs = list(self._subscribers)
            self._buffer.append(event)
            if len(self._buffer) > self._buffer_max:
                self._buffer = self._buffer[-self._buffer_max :]
        for _tag, sub in subs:
            try:
                sub(event)
            except Exception:  # noqa: BLE001
                logger.exception("event pump subscriber raised; removing")
                self.unsubscribe(sub)

    def emit_dict(self, type_: str, **payload: Any) -> Event:
        """Convenience: build an Event from a dict-shaped payload and emit it."""
        ev = Event(type=type_, **payload)
        self.emit(ev)
        return ev

    def recent(self, n: int = 200) -> list[Event]:
        """Return the last *n* events from the rolling buffer."""
        with self._lock:
            return list(self._buffer[-n:])

    def __len__(self) -> int:
        with self._lock:
            return len(self._subscribers)
