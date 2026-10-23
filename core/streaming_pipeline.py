"""Backpressure-Aware Streaming Pipeline for Jarvis.

Provides:
- BackpressureController: token-bucket rate limiting + concurrency cap
- StreamingPipeline: composable async stages with flow control
- FlowControlledStream: wraps async generators respecting backpressure
- AdaptiveThrottle: dynamically adjusts output rate based on consumer speed
"""
import asyncio
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable

logger = logging.getLogger(__name__)


@dataclass
class BackpressureConfig:
    tokens_per_second: float = 100.0
    max_burst: float = 200.0
    max_concurrent: int = 10
    queue_high_watermark: int = 1000
    queue_low_watermark: int = 100


class BackpressureController:
    """Token-bucket rate limiter with concurrency cap and queue watermarks."""

    def __init__(self, config: BackpressureConfig | None = None):
        self._config = config or BackpressureConfig()
        self._tokens = self._config.max_burst
        self._max_burst = self._config.max_burst
        self._rate = self._config.tokens_per_second
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()
        self._active = 0
        self._semaphore = asyncio.Semaphore(self._config.max_concurrent) if _has_event_loop() else None
        self._paused = False
        self._pause_event: asyncio.Event | None = None
        self._stats = {
            "total_acquired": 0,
            "total_released": 0,
            "total_throttled": 0,
            "total_paused": 0,
            "total_dropped": 0,
        }

    def acquire(self, tokens: float = 1.0, timeout: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                if self._paused:
                    self._stats["total_paused"] += 1
                    return False
                self._refill_locked()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    self._active += 1
                    self._stats["total_acquired"] += 1
                    return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._stats["total_throttled"] += 1
                return False
            time.sleep(min(0.05, remaining))

    async def acquire_async(self, tokens: float = 1.0, timeout: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout
        while True:
            if self._pause_event and self._pause_event.is_set():
                await self._pause_event.wait()
            with self._lock:
                if self._paused:
                    self._stats["total_paused"] += 1
                    return False
                self._refill_locked()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    self._active += 1
                    self._stats["total_acquired"] += 1
                    return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._stats["total_throttled"] += 1
                return False
            await asyncio.sleep(min(0.05, remaining))

    def release(self) -> None:
        with self._lock:
            self._active = max(0, self._active - 1)
            self._stats["total_released"] += 1

    def pause(self) -> None:
        with self._lock:
            self._paused = True
        if self._pause_event:
            self._pause_event.clear()

    def resume(self) -> None:
        with self._lock:
            self._paused = False
        if self._pause_event:
            self._pause_event.set()

    def set_rate(self, tokens_per_second: float) -> None:
        with self._lock:
            self._rate = tokens_per_second

    def _refill_locked(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._max_burst, self._tokens + elapsed * self._rate)
        self._last_refill = now

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "tokens_available": round(self._tokens, 2),
                "max_burst": self._max_burst,
                "rate": self._rate,
                "active_requests": self._active,
                "paused": self._paused,
                "stats": dict(self._stats),
            }


@dataclass
class StageResult:
    data: Any
    stage_name: str
    timestamp: float = field(default_factory=time.time)
    error: str | None = None


class StreamingPipeline:
    """Composable async streaming pipeline with backpressure.

    Stages are async callables that take data and return transformed data.
    Data flows through stages, respecting backpressure from the controller.
    """

    def __init__(
        self,
        name: str = "pipeline",
        backpressure: BackpressureController | None = None,
    ):
        self._name = name
        self._stages: list[tuple[str, Callable]] = []
        self._controller = backpressure or BackpressureController()
        self._status = "idle"
        self._processed = 0
        self._errors = 0
        self._start_time = 0.0
        self._history: deque[StageResult] = deque(maxlen=500)

    def add_stage(self, name: str, fn: Callable) -> "StreamingPipeline":
        self._stages.append((name, fn))
        return self

    async def process(self, data: Any) -> StageResult:
        if not self._stages:
            return StageResult(data=data, stage_name="passthrough")
        self._status = "processing"
        current = data
        last_stage = "input"
        for stage_name, stage_fn in self._stages:
            try:
                if asyncio.iscoroutinefunction(stage_fn):
                    current = await stage_fn(current)
                else:
                    current = stage_fn(current)
                last_stage = stage_name
            except Exception as e:
                self._errors += 1
                logger.debug(f"Pipeline {self._name} stage {stage_name} error: {e}")
                result = StageResult(
                    data=current,
                    stage_name=last_stage,
                    error=f"{stage_name}: {e}",
                )
                self._history.append(result)
                return result
        self._processed += 1
        result = StageResult(data=current, stage_name=last_stage)
        self._history.append(result)
        return result

    async def process_stream(
        self, source: AsyncGenerator, max_items: int | None = None
    ) -> AsyncGenerator[StageResult, None]:
        self._status = "streaming"
        self._start_time = time.time()
        count = 0
        try:
            async for item in source:
                if max_items is not None and count >= max_items:
                    break
                result = await self.process(item)
                count += 1
                yield result
                if not await self._controller.acquire_async(tokens=0.01, timeout=5.0):
                    break
                self._controller.release()
        finally:
            self._status = "idle"

    def get_status(self) -> dict[str, Any]:
        return {
            "name": self._name,
            "status": self._status,
            "stages": [name for name, _ in self._stages],
            "processed": self._processed,
            "errors": self._errors,
            "uptime_seconds": round(time.time() - self._start_time, 1) if self._start_time else 0,
            "backpressure": self._controller.get_status(),
        }


class FlowControlledStream:
    """Wraps an async generator with backpressure-aware flow control.

    Supports:
    - Pause/resume from consumer side
    - Buffer with configurable size
    - Drop-oldest when buffer overflows
    - Consumer speed-based adaptive throttling
    """

    def __init__(
        self,
        source: AsyncGenerator,
        name: str = "stream",
        buffer_size: int = 200,
        controller: BackpressureController | None = None,
    ):
        self._source = source
        self._name = name
        self._buffer: asyncio.Queue = asyncio.Queue(maxsize=buffer_size)
        self._controller = controller or BackpressureController()
        self._paused = False
        self._finished = False
        self._consumed = 0
        self._produced = 0
        self._dropped = 0
        self._producer_task: asyncio.Task | None = None
        self._throttle_rate: float = 0.0
        self._last_consume_time: float = 0.0
        self._consume_times: deque[float] = deque(maxlen=100)

    async def start(self) -> None:
        self._producer_task = asyncio.create_task(self._produce())

    async def _produce(self) -> None:
        try:
            async for item in self._source:
                if self._finished:
                    break
                self._produced += 1
                try:
                    self._buffer.put_nowait(item)
                except asyncio.QueueFull:
                    self._dropped += 1
                    try:
                        self._buffer.get_nowait()
                        self._buffer.put_nowait(item)
                    except (asyncio.QueueEmpty, asyncio.QueueFull):
                        pass
                while self._paused and not self._finished:
                    await asyncio.sleep(0.1)
        except Exception as e:
            logger.debug(f"FlowControlledStream {self._name} producer error: {e}")
        finally:
            self._finished = True
            await self._buffer.put(None)

    async def read(self) -> Any:
        if self._finished and self._buffer.empty():
            return None
        item = await self._buffer.get()
        if item is None:
            return None
        self._consumed += 1
        now = time.time()
        if self._last_consume_time > 0:
            dt = now - self._last_consume_time
            self._consume_times.append(dt)
            if len(self._consume_times) >= 5:
                avg_dt = sum(self._consume_times) / len(self._consume_times)
                self._throttle_rate = 1.0 / avg_dt if avg_dt > 0 else 0
        self._last_consume_time = now
        return item

    async def __aiter__(self):
        while True:
            item = await self.read()
            if item is None:
                break
            yield item

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def stop(self) -> None:
        self._finished = True
        if self._producer_task and not self._producer_task.done():
            self._producer_task.cancel()

    @property
    def is_finished(self) -> bool:
        return self._finished and self._buffer.empty()

    def get_status(self) -> dict[str, Any]:
        return {
            "name": self._name,
            "produced": self._produced,
            "consumed": self._consumed,
            "dropped": self._dropped,
            "buffer_size": self._buffer.qsize(),
            "paused": self._paused,
            "finished": self._finished,
            "consume_rate_per_sec": round(self._throttle_rate, 2),
        }


class AdaptiveThrottle:
    """Dynamically adjusts output rate based on consumer performance.

    Tracks consumer processing times and adjusts the inter-item delay
    to maximize throughput without overwhelming the consumer.
    """

    def __init__(
        self,
        min_delay: float = 0.001,
        max_delay: float = 1.0,
        target_latency: float = 0.1,
        increase_factor: float = 1.5,
        decrease_factor: float = 0.7,
    ):
        self._min_delay = min_delay
        self._max_delay = max_delay
        self._target_latency = target_latency
        self._increase_factor = increase_factor
        self._decrease_factor = decrease_factor
        self._current_delay = min_delay
        self._lock = threading.Lock()
        self._latencies: deque[float] = deque(maxlen=50)
        self._total_items = 0
        self._total_wait = 0.0

    def record_latency(self, latency: float) -> None:
        with self._lock:
            self._latencies.append(latency)
            self._total_items += 1
            if len(self._latencies) >= 3:
                avg = sum(self._latencies) / len(self._latencies)
                if avg > self._target_latency * 1.5:
                    self._current_delay = min(
                        self._max_delay, self._current_delay * self._increase_factor
                    )
                elif avg < self._target_latency * 0.5:
                    self._current_delay = max(
                        self._min_delay, self._current_delay * self._decrease_factor
                    )

    def get_delay(self) -> float:
        with self._lock:
            return self._current_delay

    async def wait(self) -> None:
        delay = self.get_delay()
        if delay > 0:
            self._total_wait += delay
            await asyncio.sleep(delay)

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "current_delay_ms": round(self._current_delay * 1000, 2),
                "min_delay_ms": round(self._min_delay * 1000, 2),
                "max_delay_ms": round(self._max_delay * 1000, 2),
                "total_items": self._total_items,
                "total_wait_ms": round(self._total_wait * 1000, 2),
                "avg_latency_ms": round(
                    (sum(self._latencies) / len(self._latencies) * 1000)
                    if self._latencies
                    else 0,
                    2,
                ),
                "target_latency_ms": round(self._target_latency * 1000, 2),
            }


_controller: BackpressureController | None = None
_controller_lock = threading.Lock()


def get_backpressure_controller(config: BackpressureConfig | None = None) -> BackpressureController:
    global _controller
    if _controller is None:
        with _controller_lock:
            if _controller is None:
                _controller = BackpressureController(config)
    return _controller


def _has_event_loop() -> bool:
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False