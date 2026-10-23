"""Event Bus — Decoupled publish/subscribe for cross-module communication.

Provides a thread-safe event bus that all Jarvis subsystems can use
to communicate without tight coupling:

- Watchdog emits resource_alert → WS manager pushes to client
- Sandbox emits execution_complete → Session records it
- ExecutionContext emits cwd_changed → UI updates
- AdaptiveAgentLoop emits recovery_attempt → Dashboard shows it

Features:
- Type-safe event channels
- Synchronous and async subscribers
- Event buffering (no subscriber left behind)
- Subscription management (subscribe, unsubscribe, once)
- Event history and replay
- Metrics (events/sec, subscriber counts, dropped events)
"""
import asyncio
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable

from core.helpers import create_tracked_task

logger = logging.getLogger(__name__)


@dataclass
class Event:
    channel: str
    data: dict
    timestamp: float = field(default_factory=time.time)
    source: str = ""
    event_id: str = ""

    def to_dict(self) -> dict:
        return {
            "channel": self.channel,
            "data": self.data,
            "timestamp": self.timestamp,
            "source": self.source,
            "event_id": self.event_id,
        }


class EventBus:
    """Thread-safe publish/subscribe event bus for Jarvis subsystems.

    Usage:
        bus = get_event_bus()
        
        # Subscribe
        bus.subscribe("resource_alert", my_handler)
        
        # Publish
        bus.publish("resource_alert", {"cpu": 95, "message": "High CPU"})
        
        # Unsubscribe
        bus.unsubscribe("resource_alert", my_handler)
    """

    MAX_BUFFER_PER_CHANNEL = 1000
    MAX_SUBSCRIBERS_PER_CHANNEL = 50
    MAX_HISTORY = 5000

    def __init__(self):
        self._lock = threading.RLock()
        self._subscribers: dict[str, deque[Callable]] = defaultdict(deque)
        self._async_subscribers: dict[str, deque[Callable]] = defaultdict(deque)
        self._once_subscribers: dict[str, list[Callable]] = defaultdict(list)
        self._buffers: dict[str, deque[Event]] = defaultdict(lambda: deque(maxlen=self.MAX_BUFFER_PER_CHANNEL))
        self._history: deque[Event] = deque(maxlen=self.MAX_HISTORY)
        self._event_counter = 0
        self._total_published = 0
        self._total_delivered = 0
        self._total_dropped = 0
        self._start_time = time.time()

    def subscribe(self, channel: str, handler: Callable, once: bool = False) -> str:
        sub_id = f"sub_{id(handler)}_{channel}"
        with self._lock:
            if once:
                self._once_subscribers[channel].append(handler)
            else:
                if len(self._subscribers[channel]) >= self.MAX_SUBSCRIBERS_PER_CHANNEL:
                    self._subscribers[channel].popleft()
                self._subscribers[channel].append(handler)
        return sub_id

    def subscribe_async(self, channel: str, handler: Callable) -> str:
        sub_id = f"asub_{id(handler)}_{channel}"
        with self._lock:
            if len(self._async_subscribers[channel]) >= self.MAX_SUBSCRIBERS_PER_CHANNEL:
                self._async_subscribers[channel].popleft()
            self._async_subscribers[channel].append(handler)
        return sub_id

    def unsubscribe(self, channel: str, handler: Callable) -> bool:
        with self._lock:
            subs = self._subscribers.get(channel, deque())
            try:
                subs.remove(handler)
                return True
            except ValueError:
                pass
            asubs = self._async_subscribers.get(channel, deque())
            try:
                asubs.remove(handler)
                return True
            except ValueError:
                pass
            once_subs = self._once_subscribers.get(channel, [])
            try:
                once_subs.remove(handler)
                return True
            except ValueError:
                pass
        return False

    def publish(self, channel: str, data: dict, source: str = "") -> Event:
        with self._lock:
            self._event_counter += 1
            event = Event(
                channel=channel,
                data=data,
                source=source,
                event_id=f"evt_{self._event_counter}",
            )
            self._total_published += 1
            self._history.append(event)
            self._buffers[channel].append(event)

            handlers = list(self._subscribers.get(channel, []))
            async_handlers = list(self._async_subscribers.get(channel, []))
            once_handlers = list(self._once_subscribers.pop(channel, []))

        for handler in handlers:
            try:
                handler(event)
                self._total_delivered += 1
            except Exception as e:
                logger.debug(f"Event handler error on {channel}: {e}")
                self._total_dropped += 1

        for handler in once_handlers:
            try:
                handler(event)
                self._total_delivered += 1
            except Exception as e:
                logger.debug(f"Once handler error on {channel}: {e}")
                self._total_dropped += 1

        if async_handlers:
            try:
                # Presence check: raises RuntimeError when publish() is called
                # from a non-async thread, which routes handlers to a new loop.
                asyncio.get_running_loop()
                for handler in async_handlers:
                    try:
                        # Tracked: a bare create_task() reference can be GC'd
                        # before it runs, silently dropping the event.
                        create_tracked_task(
                            self._call_async(handler, event),
                            name=f"event_bus:{channel}",
                        )
                        self._total_delivered += 1
                    except RuntimeError:
                        thread = threading.Thread(
                            target=self._run_async_in_new_loop,
                            args=(handler, event),
                            daemon=True,
                        )
                        thread.start()
                        self._total_delivered += 1
            except RuntimeError:
                pass

        return event

    async def _call_async(self, handler: Callable, event: Event) -> None:
        try:
            await handler(event)
        except Exception as e:
            logger.debug(f"Async event handler error: {e}")
            self._total_dropped += 1

    def _run_async_in_new_loop(self, handler: Callable, event: Event) -> None:
        try:
            loop = asyncio.new_event_loop()
            loop.run_until_complete(handler(event))
            loop.close()
        except Exception as e:
            logger.debug(f"Async handler in new loop error: {e}")

    def get_buffer(self, channel: str, limit: int = 50) -> list[dict]:
        with self._lock:
            return [e.to_dict() for e in list(self._buffers.get(channel, []))[-limit:]]

    def get_history(self, channel: str | None = None, limit: int = 50) -> list[dict]:
        with self._lock:
            events = self._history
            if channel:
                events = [e for e in events if e.channel == channel]
            return [e.to_dict() for e in list(events)[-limit:]]

    def get_channels(self) -> list[str]:
        with self._lock:
            all_channels = set(self._subscribers.keys()) | set(self._async_subscribers.keys()) | set(self._buffers.keys())
            return sorted(all_channels)

    def get_subscriber_count(self, channel: str) -> int:
        with self._lock:
            return len(self._subscribers.get(channel, [])) + len(self._async_subscribers.get(channel, []))

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            uptime = time.time() - self._start_time
            events_per_sec = self._total_published / max(1, uptime)
            return {
                "total_published": self._total_published,
                "total_delivered": self._total_delivered,
                "total_dropped": self._total_dropped,
                "events_per_second": round(events_per_sec, 2),
                "channels": self.get_channels(),
                "history_size": len(self._history),
                "uptime_seconds": round(uptime, 1),
            }

    def clear(self) -> None:
        with self._lock:
            self._buffers.clear()
            self._history.clear()
            self._subscribers.clear()
            self._async_subscribers.clear()
            self._once_subscribers.clear()


_event_bus: EventBus | None = None
_event_bus_lock = threading.Lock()


def get_event_bus() -> EventBus:
    global _event_bus
    if _event_bus is None:
        with _event_bus_lock:
            if _event_bus is None:
                _event_bus = EventBus()
            return _event_bus
    return _event_bus