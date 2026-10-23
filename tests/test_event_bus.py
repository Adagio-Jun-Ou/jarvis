"""Tests for Event Bus system.

Covers:
- Publish/subscribe lifecycle
- Once subscriptions
- Async subscriptions
- Event buffering
- Event history
- Channel management
- Subscriber limits
- Metrics (published, delivered, dropped)
- Thread safety
- Unsubscribe
- Clear
"""
import time
import threading
from pathlib import Path

import pytest

sys_path = str(Path(__file__).parent.parent / "src")
import sys
if sys_path not in sys.path:
    sys_path not in sys.path and sys.path.insert(0, sys_path)

from core.event_bus import EventBus, Event, get_event_bus


class TestEvent:
    def test_event_creation(self):
        e = Event(channel="test", data={"key": "value"}, source="module")
        assert e.channel == "test"
        assert e.data == {"key": "value"}
        assert e.source == "module"

    def test_event_to_dict(self):
        e = Event(channel="test", data={"x": 1}, source="mod", event_id="evt_1")
        d = e.to_dict()
        assert d["channel"] == "test"
        assert d["data"]["x"] == 1
        assert d["source"] == "mod"
        assert d["event_id"] == "evt_1"

    def test_event_timestamp(self):
        e = Event(channel="test", data={})
        assert e.timestamp > 0


class TestEventBusSubscribe:
    def setup_method(self):
        self.bus = EventBus()

    def test_subscribe_and_publish(self):
        received = []
        self.bus.subscribe("test", lambda e: received.append(e))
        self.bus.publish("test", {"msg": "hello"})
        assert len(received) == 1
        assert received[0].data["msg"] == "hello"

    def test_multiple_subscribers(self):
        r1, r2 = [], []
        self.bus.subscribe("test", lambda e: r1.append(e))
        self.bus.subscribe("test", lambda e: r2.append(e))
        self.bus.publish("test", {"x": 1})
        assert len(r1) == 1
        assert len(r2) == 1

    def test_unsubscribe(self):
        received = []
        handler = lambda e: received.append(e)
        self.bus.subscribe("test", handler)
        self.bus.publish("test", {"x": 1})
        assert len(received) == 1
        self.bus.unsubscribe("test", handler)
        self.bus.publish("test", {"x": 2})
        assert len(received) == 1

    def test_unsubscribe_nonexistent(self):
        result = self.bus.unsubscribe("nonexistent", lambda e: None)
        assert result is False

    def test_subscribe_once(self):
        received = []
        self.bus.subscribe("test", lambda e: received.append(e), once=True)
        self.bus.publish("test", {"x": 1})
        self.bus.publish("test", {"x": 2})
        assert len(received) == 1

    def test_channel_isolation(self):
        r1, r2 = [], []
        self.bus.subscribe("channel_a", lambda e: r1.append(e))
        self.bus.subscribe("channel_b", lambda e: r2.append(e))
        self.bus.publish("channel_a", {"x": "a"})
        assert len(r1) == 1
        assert len(r2) == 0

    def test_handler_exception_doesnt_stop_others(self):
        r1, r2 = [], []

        def bad_handler(e):
            raise ValueError("oops")

        self.bus.subscribe("test", bad_handler)
        self.bus.subscribe("test", lambda e: r2.append(e))
        self.bus.publish("test", {"x": 1})
        assert len(r2) == 1


class TestEventBusBuffer:
    def setup_method(self):
        self.bus = EventBus()

    def test_buffer_stores_events(self):
        self.bus.publish("test", {"x": 1})
        self.bus.publish("test", {"x": 2})
        buf = self.bus.get_buffer("test")
        assert len(buf) >= 2

    def test_buffer_channel_isolation(self):
        self.bus.publish("a", {"x": 1})
        self.bus.publish("b", {"x": 2})
        buf_a = self.bus.get_buffer("a")
        buf_b = self.bus.get_buffer("b")
        assert any(e["data"]["x"] == 1 for e in buf_a)
        assert any(e["data"]["x"] == 2 for e in buf_b)

    def test_buffer_limit(self):
        for i in range(1200):
            self.bus.publish("test", {"i": i})
        buf = self.bus.get_buffer("test", limit=2000)
        assert len(buf) <= self.bus.MAX_BUFFER_PER_CHANNEL


class TestEventBusHistory:
    def setup_method(self):
        self.bus = EventBus()

    def test_history_stores_events(self):
        self.bus.publish("test", {"x": 1})
        history = self.bus.get_history()
        assert len(history) >= 1

    def test_history_channel_filter(self):
        self.bus.publish("a", {"x": 1})
        self.bus.publish("b", {"x": 2})
        history_a = self.bus.get_history(channel="a")
        assert all(e["channel"] == "a" for e in history_a)

    def test_history_limit(self):
        for i in range(100):
            self.bus.publish("ch", {"i": i})
        history = self.bus.get_history(limit=10)
        assert len(history) <= 10


class TestEventBusMetrics:
    def setup_method(self):
        self.bus = EventBus()

    def test_published_count(self):
        self.bus.publish("test", {"x": 1})
        self.bus.publish("test", {"x": 2})
        status = self.bus.get_status()
        assert status["total_published"] >= 2

    def test_delivered_count(self):
        self.bus.subscribe("test", lambda e: None)
        self.bus.publish("test", {"x": 1})
        status = self.bus.get_status()
        assert status["total_delivered"] >= 1

    def test_dropped_count_on_error(self):
        def bad(e):
            raise RuntimeError("fail")
        self.bus.subscribe("test", bad)
        self.bus.publish("test", {"x": 1})
        status = self.bus.get_status()
        assert status["total_dropped"] >= 1

    def test_events_per_second(self):
        status = self.bus.get_status()
        assert "events_per_second" in status

    def test_channels_list(self):
        self.bus.publish("alpha", {})
        self.bus.publish("beta", {})
        channels = self.bus.get_channels()
        assert "alpha" in channels
        assert "beta" in channels

    def test_subscriber_count(self):
        self.bus.subscribe("test", lambda e: None)
        self.bus.subscribe("test", lambda e: None)
        count = self.bus.get_subscriber_count("test")
        assert count == 2

    def test_status_fields(self):
        status = self.bus.get_status()
        assert "total_published" in status
        assert "total_delivered" in status
        assert "total_dropped" in status
        assert "channels" in status
        assert "uptime_seconds" in status
        assert "history_size" in status


class TestEventBusSubLimits:
    def setup_method(self):
        self.bus = EventBus()

    def test_max_subscribers_eviction(self):
        handlers = []
        for i in range(self.bus.MAX_SUBSCRIBERS_PER_CHANNEL + 5):
            handlers.append(lambda e, i=i: i)
            self.bus.subscribe("test", handlers[-1])
        count = self.bus.get_subscriber_count("test")
        assert count <= self.bus.MAX_SUBSCRIBERS_PER_CHANNEL


class TestEventBusClear:
    def setup_method(self):
        self.bus = EventBus()

    def test_clear_removes_everything(self):
        self.bus.subscribe("test", lambda e: None)
        self.bus.publish("test", {"x": 1})
        self.bus.clear()
        assert self.bus.get_subscriber_count("test") == 0
        assert len(self.bus.get_history()) == 0


class TestEventBusThreadSafety:
    def test_concurrent_publish(self):
        bus = EventBus()
        received = []
        bus.subscribe("test", lambda e: received.append(e.data.get("i")))
        errors = []

        def publish_many(start):
            try:
                for i in range(start, start + 50):
                    bus.publish("test", {"i": i})
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=publish_many, args=(i * 50,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0
        assert len(received) == 200

    def test_singleton(self):
        b1 = get_event_bus()
        b2 = get_event_bus()
        assert b1 is b2