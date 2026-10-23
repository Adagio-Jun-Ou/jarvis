"""Tests for core/streaming_pipeline.py — BackpressureController, StreamingPipeline, FlowControlledStream, AdaptiveThrottle."""
import asyncio
import time
import threading

import pytest


# ─── BackpressureController ──────────────────────────────────────────────────

from core.streaming_pipeline import (
    BackpressureConfig,
    BackpressureController,
    StreamingPipeline,
    FlowControlledStream,
    AdaptiveThrottle,
)


class TestBackpressureController:
    def test_acquire_and_release(self):
        bc = BackpressureController(BackpressureConfig(tokens_per_second=100, max_burst=100))
        assert bc.acquire(tokens=1.0, timeout=1.0) is True
        assert bc._active == 1
        bc.release()
        assert bc._active == 0

    def test_acquire_exceeds_burst_fails(self):
        bc = BackpressureController(BackpressureConfig(tokens_per_second=10, max_burst=10))
        assert bc.acquire(tokens=10.0, timeout=0.1) is True
        assert bc.acquire(tokens=1.0, timeout=0.05) is False

    def test_acquire_refills_over_time(self):
        bc = BackpressureController(BackpressureConfig(tokens_per_second=1000, max_burst=10))
        assert bc.acquire(tokens=10.0, timeout=0.1) is True
        time.sleep(0.02)
        assert bc.acquire(tokens=1.0, timeout=0.5) is True

    def test_release_without_acquire_safe(self):
        bc = BackpressureController()
        bc.release()
        assert bc._active == 0

    def test_pause_blocks_acquire(self):
        bc = BackpressureController()
        bc.pause()
        assert bc.acquire(timeout=0.05) is False

    def test_resume_allows_acquire(self):
        bc = BackpressureController()
        bc.pause()
        bc.resume()
        assert bc.acquire(timeout=0.5) is True

    def test_set_rate(self):
        bc = BackpressureController(BackpressureConfig(tokens_per_second=100, max_burst=100))
        bc.set_rate(500)
        assert bc._rate == 500

    def test_get_status(self):
        bc = BackpressureController()
        bc.acquire(timeout=0.5)
        status = bc.get_status()
        assert "tokens_available" in status
        assert "rate" in status
        assert "active_requests" in status
        assert "paused" in status
        assert "stats" in status

    def test_stats_tracking(self):
        bc = BackpressureController()
        bc.acquire(timeout=0.5)
        bc.release()
        assert bc.get_status()["stats"]["total_acquired"] == 1
        assert bc.get_status()["stats"]["total_released"] == 1

    def test_timeout_returns_false(self):
        bc = BackpressureController(BackpressureConfig(tokens_per_second=1, max_burst=1))
        assert bc.acquire(tokens=1.0, timeout=0.1) is True
        assert bc.acquire(tokens=1.0, timeout=0.01) is False
        assert bc.get_status()["stats"]["total_throttled"] >= 1


@pytest.mark.asyncio
class TestBackpressureControllerAsync:
    async def test_acquire_async(self):
        bc = BackpressureController(BackpressureConfig(tokens_per_second=100, max_burst=100))
        result = await bc.acquire_async(tokens=1.0, timeout=1.0)
        assert result is True
        bc.release()

    async def test_acquire_async_timeout(self):
        bc = BackpressureController(BackpressureConfig(tokens_per_second=1, max_burst=1))
        assert await bc.acquire_async(tokens=1.0, timeout=0.1) is True
        result = await bc.acquire_async(tokens=1.0, timeout=0.01)
        assert result is False

    async def test_pause_resume_async(self):
        bc = BackpressureController()
        bc.pause()
        result = await bc.acquire_async(timeout=0.05)
        assert result is False
        bc.resume()
        result = await bc.acquire_async(timeout=0.5)
        assert result is True


# ─── StreamingPipeline ────────────────────────────────────────────────────────

class TestStreamingPipeline:
    def test_add_stage_returns_self(self):
        p = StreamingPipeline(name="test")
        result = p.add_stage("upper", lambda x: x.upper())
        assert result is p

    @pytest.mark.asyncio
    async def test_process_single_stage(self):
        p = StreamingPipeline()
        p.add_stage("upper", lambda x: x.upper())
        result = await p.process("hello")
        assert result.data == "HELLO"
        assert result.stage_name == "upper"
        assert result.error is None

    @pytest.mark.asyncio
    async def test_process_multi_stage(self):
        p = StreamingPipeline()
        p.add_stage("upper", lambda x: x.upper())
        p.add_stage("reverse", lambda x: x[::-1])
        result = await p.process("hello")
        assert result.data == "OLLEH"
        assert result.stage_name == "reverse"

    @pytest.mark.asyncio
    async def test_process_error_in_stage(self):
        def bad_stage(x):
            raise ValueError("oops")

        p = StreamingPipeline()
        p.add_stage("bad", bad_stage)
        result = await p.process("data")
        assert result.error is not None
        assert "bad" in result.error
        assert p._errors == 1

    @pytest.mark.asyncio
    async def test_process_async_stage(self):
        async def async_upper(x):
            return x.upper()

        p = StreamingPipeline()
        p.add_stage("async_upper", async_upper)
        result = await p.process("hello")
        assert result.data == "HELLO"

    @pytest.mark.asyncio
    async def test_empty_pipeline_passthrough(self):
        p = StreamingPipeline()
        result = await p.process("data")
        assert result.data == "data"
        assert result.stage_name == "passthrough"

    @pytest.mark.asyncio
    async def test_process_stream(self):
        async def source():
            for i in range(5):
                yield f"item_{i}"

        p = StreamingPipeline()
        p.add_stage("upper", lambda x: x.upper())
        results = []
        async for r in p.process_stream(source(), max_items=5):
            results.append(r)
        assert len(results) == 5
        assert results[0].data == "ITEM_0"

    def test_get_status(self):
        p = StreamingPipeline()
        p.add_stage("s1", lambda x: x)
        status = p.get_status()
        assert status["name"] == "pipeline"
        assert status["status"] == "idle"
        assert "s1" in status["stages"]
        assert status["processed"] == 0


# ─── FlowControlledStream ────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestFlowControlledStream:
    async def test_basic_stream(self):
        async def source():
            for i in range(5):
                yield i

        stream = FlowControlledStream(source(), buffer_size=10)
        await stream.start()
        await asyncio.sleep(0.1)
        items = []
        async for item in stream:
            items.append(item)
        assert len(items) == 5
        assert items == list(range(5))

    async def test_pause_resume(self):
        async def source():
            for i in range(10):
                yield i
                await asyncio.sleep(0.01)

        stream = FlowControlledStream(source(), buffer_size=20)
        await stream.start()
        await asyncio.sleep(0.05)
        stream.pause()
        await asyncio.sleep(0.05)
        assert stream._paused is True
        stream.resume()
        items = []
        async for item in stream:
            items.append(item)
        assert len(items) >= 5

    async def test_stop_early(self):
        async def source():
            for i in range(100):
                yield i
                await asyncio.sleep(0.01)

        stream = FlowControlledStream(source(), buffer_size=10)
        await stream.start()
        await asyncio.sleep(0.05)
        stream.stop()
        assert stream._finished is True

    async def test_buffer_overflow_drop_oldest(self):
        async def fast_source():
            for i in range(50):
                yield i

        stream = FlowControlledStream(fast_source(), buffer_size=5)
        await stream.start()
        await asyncio.sleep(0.1)
        status = stream.get_status()
        assert status["produced"] == 50
        assert status["dropped"] >= 0

    async def test_status(self):
        async def source():
            for i in range(3):
                yield i

        stream = FlowControlledStream(source())
        await stream.start()
        await asyncio.sleep(0.2)
        status = stream.get_status()
        assert "name" in status
        assert "produced" in status
        assert "consumed" in status
        assert "buffer_size" in status


# ─── AdaptiveThrottle ─────────────────────────────────────────────────────────

class TestAdaptiveThrottle:
    def test_initial_delay(self):
        at = AdaptiveThrottle(min_delay=0.001, max_delay=1.0)
        assert at.get_delay() == 0.001

    def test_increase_on_high_latency(self):
        at = AdaptiveThrottle(min_delay=0.001, max_delay=1.0, target_latency=0.1)
        at.record_latency(0.5)
        at.record_latency(0.5)
        at.record_latency(0.5)
        assert at.get_delay() > 0.001

    def test_decrease_on_low_latency(self):
        at = AdaptiveThrottle(min_delay=0.001, max_delay=1.0, target_latency=0.1, decrease_factor=0.3)
        at.record_latency(0.5)
        at.record_latency(0.5)
        at.record_latency(0.5)
        increased = at.get_delay()
        at._latencies.clear()
        at.record_latency(0.001)
        at.record_latency(0.001)
        at.record_latency(0.001)
        decreased = at.get_delay()
        assert decreased < increased

    def test_clamp_max_delay(self):
        at = AdaptiveThrottle(min_delay=0.001, max_delay=0.05, target_latency=0.001)
        for _ in range(20):
            at.record_latency(10.0)
        assert at.get_delay() <= 0.05

    def test_clamp_min_delay(self):
        at = AdaptiveThrottle(min_delay=0.001, max_delay=1.0, target_latency=10.0)
        for _ in range(20):
            at.record_latency(0.0001)
        assert at.get_delay() >= 0.001

    def test_get_status(self):
        at = AdaptiveThrottle()
        at.record_latency(0.05)
        status = at.get_status()
        assert "current_delay_ms" in status
        assert "total_items" in status
        assert "avg_latency_ms" in status
        assert "target_latency_ms" in status

    @pytest.mark.asyncio
    async def test_async_wait(self):
        at = AdaptiveThrottle(min_delay=0.001, max_delay=0.01)
        start = time.time()
        await at.wait()
        elapsed = time.time() - start
        assert elapsed < 0.1


# ─── Integration ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestStreamingIntegration:
    async def test_pipeline_with_backpressure(self):
        bc = BackpressureController(BackpressureConfig(tokens_per_second=1000, max_burst=1000))
        p = StreamingPipeline(name="integrated", backpressure=bc)

        async def enrich(x):
            return x * 2

        p.add_stage("double", enrich)
        p.add_stage("stringify", lambda x: f"result={x}")

        result = await p.process(21)
        assert result.data == "result=42"

    async def test_stream_with_flow_control(self):
        async def source():
            for i in range(10):
                yield i

        stream = FlowControlledStream(source(), buffer_size=20)
        await stream.start()
        results = []
        async for item in stream:
            results.append(item)
        assert len(results) == 10
        assert results == list(range(10))

    async def test_adaptive_throttle_with_pipeline(self):
        at = AdaptiveThrottle(min_delay=0.001, max_delay=0.01, target_latency=0.05)

        async def timed_stage(x):
            at.record_latency(0.02)
            return x + 1

        p = StreamingPipeline(name="throttled")
        p.add_stage("increment", timed_stage)

        for i in range(5):
            result = await p.process(i)
            assert result.data == i + 1

        assert p._processed == 5

    async def test_backpressure_pause_resume_stream(self):
        bc = BackpressureController(BackpressureConfig(tokens_per_second=10000, max_burst=10000))

        async def source():
            for i in range(20):
                yield i
                await asyncio.sleep(0.005)

        async def double(x):
            return x * 2

        p = StreamingPipeline(name="paused", backpressure=bc)
        p.add_stage("double", double)

        bc.pause()
        results = []
        async for r in p.process_stream(source(), max_items=3):
            results.append(r)
            if len(results) == 3:
                break
        assert len(results) <= 3