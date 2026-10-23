"""API Call Manager for Jarvis.

Coordinates all LLM API calls across subsystems to:
1. Batch similar requests together
2. Implement rate limiting and throttling
3. Prioritize interactive vs background calls
4. Deduplicate redundant requests
5. Cache responses for repeated queries
6. Provide backpressure for overloaded systems

This prevents subsystems (Erdős engine, learning pipeline, self-refine, etc.)
from overwhelming the API with concurrent requests.
"""
import asyncio
from collections import deque
import hashlib
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from core.helpers import create_tracked_task, safe_error_message
from typing import Any, Optional, Callable
from functools import wraps

logger = logging.getLogger(__name__)


class Priority(Enum):
    INTERACTIVE = 1    # User-facing, needs fast response
    LEARNING = 2       # Background learning
    REFINEMENT = 3     # Code refinement
    DAYDREAM = 4       # Idle exploration
    MAINTENANCE = 5    # Background maintenance


class RequestType(Enum):
    CHAT = "chat"
    CHAT_STREAM = "chat_stream"
    CHAT_WITH_TOOLS = "chat_with_tools"
    EMBEDDING = "embedding"


@dataclass
class APIRequest:
    id: str
    request_type: RequestType
    priority: Priority
    subsystem: str
    user_input: str
    system_prompt: str
    conversation_history: list[dict] | None
    tools: list[dict] | None
    callback: Optional[Callable] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    batch_id: str | None = None
    retry_count: int = 0


@dataclass
class APIResponse:
    request_id: str
    success: bool
    content: Any
    error: str | None = None
    from_cache: bool = False
    duration_ms: float = 0.0


@dataclass
class RateLimit:
    max_requests: int
    window_seconds: float
    _requests: list = field(default_factory=list)

    def can_proceed(self) -> bool:
        now = time.time()
        self._requests = [t for t in self._requests if now - t < self.window_seconds]
        return len(self._requests) < self.max_requests

    def record(self):
        self._requests.append(time.time())


@dataclass
class BatchWindow:
    window_ms: float = 100.0
    max_batch_size: int = 5
    requests: list = field(default_factory=list)
    started_at: float | None = None


class APICallManager:
    """Manages all LLM API calls for Jarvis subsystems.
    
    Features:
    - Request batching with time windows
    - Priority-based queuing
    - Rate limiting per subsystem
    - Response caching
    - Request deduplication
    - Backpressure when overloaded
    """

    DEFAULT_RATE_LIMITS = {
        Priority.INTERACTIVE: RateLimit(max_requests=60, window_seconds=60.0),
        Priority.LEARNING: RateLimit(max_requests=20, window_seconds=60.0),
        Priority.REFINEMENT: RateLimit(max_requests=10, window_seconds=60.0),
        Priority.DAYDREAM: RateLimit(max_requests=5, window_seconds=60.0),
        Priority.MAINTENANCE: RateLimit(max_requests=5, window_seconds=60.0),
    }

    CACHE_TTL = {
        RequestType.EMBEDDING: 3600,    # 1 hour for embeddings
        RequestType.CHAT: 60,           # 1 minute for chat
        RequestType.CHAT_WITH_TOOLS: 0, # No caching for tool calls
    }

    def __init__(
        self,
        llm_client,
        batch_window_ms: float = 100.0,
        max_batch_size: int = 5,
        max_queue_size: int = 100,
        enable_caching: bool = True,
    ):
        self.llm = llm_client
        self.batch_window_ms = batch_window_ms
        self.max_batch_size = max_batch_size
        self.max_queue_size = max_queue_size
        self.enable_caching = enable_caching

        self._request_queue: list[APIRequest] = []
        self._pending_requests: dict[str, APIRequest] = {}
        self._max_pending = 500
        self._batch_windows: dict[RequestType, BatchWindow] = {}

        self._rate_limits = dict(self.DEFAULT_RATE_LIMITS)
        self._subsystem_limits: dict[str, RateLimit] = {}

        self._cache: dict[str, tuple[Any, float]] = {}
        self._cache_lock = threading.Lock()

        self._lock = threading.RLock()
        self._async_lock = asyncio.Lock()
        self._stats_lock = threading.Lock()

        self._running = False
        self._processor_task: asyncio.Task | None = None

        self._activity_log: deque[dict] = deque(maxlen=100)
        self._activity_log_lock = threading.Lock()

        with self._stats_lock:
            self._stats = {
                "total_requests": 0,
                "batched_requests": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "rate_limited": 0,
                "errors": 0,
                "avg_latency_ms": 0.0,
            }

    def set_subsystem_limit(self, subsystem: str, max_requests: int, window_seconds: float):
        """Set rate limit for a specific subsystem."""
        self._subsystem_limits[subsystem] = RateLimit(
            max_requests=max_requests,
            window_seconds=window_seconds,
        )

    def _get_cache_key(self, request: APIRequest) -> str:
        """Generate cache key for a request."""
        if not self.enable_caching:
            return ""

        content = f"{request.request_type.value}:{request.user_input}:{request.system_prompt}"
        if request.conversation_history:
            history_hash = hashlib.sha256(
                str([m.get("role", "") + ":" + m.get("content", "")[:100] for m in request.conversation_history[:5]]).encode()
            ).hexdigest()[:16]
            content += f":hist_{history_hash}"
        if request.tools:
            tool_names = sorted([t.get("name", "") for t in request.tools])
            content += f":tools_{','.join(tool_names)}"

        return hashlib.sha256(content.encode()).hexdigest()

    def _get_cached(self, cache_key: str, request_type: RequestType) -> Any | None:
        """Get cached response if valid."""
        if not cache_key or not self.enable_caching:
            return None

        ttl = self.CACHE_TTL.get(request_type, 0)
        if ttl == 0:
            return None

        with self._cache_lock:
            if cache_key in self._cache:
                cached, timestamp = self._cache[cache_key]
                if time.time() - timestamp < ttl:
                    return cached
                else:
                    del self._cache[cache_key]
        return None

    def _set_cached(self, cache_key: str, response: Any):
        """Cache a response."""
        if not cache_key or not self.enable_caching:
            return

        with self._cache_lock:
            self._cache[cache_key] = (response, time.time())

            if len(self._cache) > 1000:
                oldest = min(self._cache.items(), key=lambda x: x[1][1])
                del self._cache[oldest[0]]

    def _check_rate_limit(self, priority: Priority, subsystem: str) -> bool:
        """Check if request can proceed based on rate limits."""
        if not self._rate_limits[priority].can_proceed():
            return False

        if subsystem in self._subsystem_limits:
            if not self._subsystem_limits[subsystem].can_proceed():
                return False

        return True

    def _record_rate_limit(self, priority: Priority, subsystem: str):
        """Record request for rate limiting."""
        self._rate_limits[priority].record()
        if subsystem in self._subsystem_limits:
            self._subsystem_limits[subsystem].record()

    async def submit(
        self,
        subsystem: str,
        priority: Priority,
        user_input: str,
        system_prompt: str,
        conversation_history: list[dict] | None = None,
        tools: list[dict] | None = None,
        request_type: RequestType = RequestType.CHAT,
        wait_for_result: bool = True,
        timeout: float = 120.0,
    ) -> APIResponse:
        """Submit an API request for processing.
        
        Args:
            subsystem: Which subsystem is making the request (erdos, learning, etc.)
            priority: Request priority
            user_input: User input text
            system_prompt: System prompt
            conversation_history: Optional conversation history
            tools: Optional tools for tool calling
            request_type: Type of API request
            wait_for_result: Whether to wait for the result
            timeout: Maximum time to wait for result
            
        Returns:
            APIResponse with result or error
        """
        import uuid
        request_id = f"{subsystem}_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}"

        request = APIRequest(
            id=request_id,
            request_type=request_type,
            priority=priority,
            subsystem=subsystem,
            user_input=user_input,
            system_prompt=system_prompt,
            conversation_history=conversation_history,
            tools=tools,
        )

        # Check cache first
        cache_key = self._get_cache_key(request)
        cached = self._get_cached(cache_key, request_type)
        if cached is not None:
            with self._stats_lock:
                self._stats["cache_hits"] += 1
            return APIResponse(
                request_id=request_id,
                success=True,
                content=cached,
                from_cache=True,
            )

        with self._stats_lock:
            self._stats["cache_misses"] += 1

        # Check rate limit
        if not self._check_rate_limit(priority, subsystem):
            with self._stats_lock:
                self._stats["rate_limited"] += 1
            return APIResponse(
                request_id=request_id,
                success=False,
                content=None,
                error="Rate limit exceeded",
            )

        # If processor is running, queue the request
        if self._running and wait_for_result:
            result_future = asyncio.Future()
            request.callback = lambda r: result_future.set_result(r) if not result_future.done() else None

            with self._lock:
                if len(self._request_queue) >= self.max_queue_size:
                    return APIResponse(
                        request_id=request_id,
                        success=False,
                        content=None,
                        error="Request queue full",
                    )
                self._request_queue.append(request)
                self._request_queue.sort(key=lambda r: r.priority.value)
                if len(self._request_queue) > self.max_queue_size:
                    self._request_queue = self._request_queue[-self.max_queue_size:]
                self._pending_requests[request_id] = request

            with self._stats_lock:
                self._stats["total_requests"] += 1

            try:
                result = await asyncio.wait_for(result_future, timeout=timeout)
                return result
            except asyncio.TimeoutError:
                return APIResponse(
                    request_id=request_id,
                    success=False,
                    content=None,
                    error="Request timed out",
                )

        # Direct execution (processor not running or not waiting)
        with self._stats_lock:
            self._stats["total_requests"] += 1
        return await self._execute_request(request, cache_key)

    async def submit_chat(
        self,
        subsystem: str,
        priority: Priority,
        user_input: str,
        system_prompt: str,
        **kwargs,
    ) -> dict:
        """Submit a chat request and return the response dict.

        Refusals (rate limit / queue full / timeout / LLM error) are marked
        ``"_refused": True`` so callers can distinguish a refusal from a
        genuine model answer whose literal text happens to be the error
        string. Previously a refusal came back shaped exactly like a
        successful response, so partially-adopted modules processed
        "Rate limit exceeded" as if the model had said it. Callers that
        want the degrade-don't-bypass behaviour should check the flag
        and fall back to their default, NOT re-call the LLM directly
        (that would bypass the very quota the refusal is enforcing).
        """
        response = await self.submit(
            subsystem=subsystem,
            priority=priority,
            user_input=user_input,
            system_prompt=system_prompt,
            request_type=RequestType.CHAT,
            **kwargs,
        )
        if response.success and isinstance(response.content, dict):
            response.content["_from_cache"] = response.from_cache
            response.content["_refused"] = False
            return response.content
        return {
            "message": {"content": response.error or "Request failed"},
            "_from_cache": False,
            "_refused": True,
        }

    async def start(self):
        """Start the request processor."""
        if self._running:
            return

        self._running = True
        self._processor_task = create_tracked_task(self._process_loop(), name="api_manager_process_loop")
        logger.info("API Call Manager started")

    async def stop(self):
        """Stop the request processor."""
        self._running = False
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
        logger.info("API Call Manager stopped")

    async def _process_loop(self):
        """Process queued requests."""
        while self._running:
            try:
                await asyncio.sleep(0.01)

                with self._lock:
                    if not self._request_queue:
                        continue

                    batch = self._get_batch()

                if batch:
                    await self._process_batch(batch)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in process loop: {e}")

    def _get_batch(self) -> list[APIRequest]:
        """Get a batch of requests to process."""
        time.time()
        batch = []

        to_remove = []
        for request in list(self._request_queue):
            if len(batch) >= self.max_batch_size:
                break

            priority_limit = self._rate_limits.get(request.priority)
            if priority_limit and not priority_limit.can_proceed():
                continue

            to_remove.append(request)
            request.started_at = datetime.now()
            batch.append(request)

        for request in to_remove:
            try:
                self._request_queue.remove(request)
            except ValueError:
                pass

        if len(batch) > 1:
            with self._stats_lock:
                self._stats["batched_requests"] += len(batch) - 1

        return batch

    async def _process_batch(self, batch: list[APIRequest]):
        """Process a batch of requests in parallel."""
        if not batch:
            return

        if len(batch) == 1:
            await self._process_single_request(batch[0])
            return

        tasks = [self._process_single_request(req) for req in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch request {batch[i].id} failed: {result}")

    async def _process_single_request(self, request: APIRequest) -> None:
        """Process a single API request."""
        try:
            self._record_rate_limit(request.priority, request.subsystem)

            start = time.time()

            from core.llm import subsystem_context
            async with subsystem_context(request.subsystem):
                if request.request_type == RequestType.CHAT:
                    result = await self.llm.chat(
                        user_input=request.user_input,
                        system_prompt=request.system_prompt,
                        conversation_history=request.conversation_history,
                    )
                elif request.request_type == RequestType.CHAT_WITH_TOOLS:
                    content, tool_calls = await self.llm.chat_with_tools(
                        user_input=request.user_input,
                        system_prompt=request.system_prompt,
                        conversation_history=request.conversation_history,
                        tools=request.tools,
                    )
                    result = {"content": content, "tool_calls": tool_calls}
                else:
                    result = await self.llm.chat(
                        user_input=request.user_input,
                        system_prompt=request.system_prompt,
                    )

            duration = (time.time() - start) * 1000

            cache_key = self._get_cache_key(request)
            self._set_cached(cache_key, result)

            response = APIResponse(
                request_id=request.id,
                success=True,
                content=result,
                duration_ms=duration,
            )

            self._update_stats(duration)

        except Exception as e:
            logger.error(f"Request {request.id} failed: {e}")
            with self._stats_lock:
                self._stats["errors"] += 1
            response = APIResponse(
                request_id=request.id,
                success=False,
                content=None,
                error=safe_error_message(e),
            )

        request.completed_at = datetime.now()

        if request.callback:
            try:
                if asyncio.iscoroutinefunction(request.callback):
                    await request.callback(response)
                else:
                    request.callback(response)
            except Exception as e:
                logger.error(f"Callback error: {e}")

        with self._lock:
            if request.id in self._pending_requests:
                del self._pending_requests[request.id]
            if len(self._pending_requests) > self._max_pending:
                oldest_keys = list(self._pending_requests.keys())[:len(self._pending_requests) - self._max_pending]
                for k in oldest_keys:
                    del self._pending_requests[k]

    def _update_stats(self, duration_ms: float):
        """Update average latency stats."""
        with self._stats_lock:
            total = self._stats["total_requests"]
            old_avg = self._stats["avg_latency_ms"]
            self._stats["avg_latency_ms"] = old_avg + (duration_ms - old_avg) / max(total, 1)

    async def _execute_request(self, request: APIRequest, cache_key: str = "") -> APIResponse:
        """Execute a single request directly (bypassing queue)."""
        try:
            self._record_rate_limit(request.priority, request.subsystem)

            start = time.time()

            # Log start of activity
            self.log_activity(
                subsystem=request.subsystem,
                priority=request.priority.name.lower(),
                preview=request.user_input[:60],
                status="active",
            )

            if request.request_type == RequestType.CHAT:
                from core.llm import subsystem_context
                async with subsystem_context(request.subsystem):
                    result = await self.llm.chat(
                        user_input=request.user_input,
                        system_prompt=request.system_prompt,
                        conversation_history=request.conversation_history,
                    )
            elif request.request_type == RequestType.CHAT_WITH_TOOLS:
                from core.llm import subsystem_context
                async with subsystem_context(request.subsystem):
                    content, tool_calls = await self.llm.chat_with_tools(
                        user_input=request.user_input,
                        system_prompt=request.system_prompt,
                        conversation_history=request.conversation_history,
                        tools=request.tools,
                    )
                result = {"content": content, "tool_calls": tool_calls}
            else:
                from core.llm import subsystem_context
                async with subsystem_context(request.subsystem):
                    result = await self.llm.chat(
                        user_input=request.user_input,
                        system_prompt=request.system_prompt,
                    )

            duration = (time.time() - start) * 1000

            if cache_key:
                self._set_cached(cache_key, result)

            self._update_stats(duration)

            # Log completion
            self.log_activity(
                subsystem=request.subsystem,
                priority=request.priority.name.lower(),
                preview=request.user_input[:60],
                latency_ms=duration,
                cached=False,
                status="complete",
            )

            return APIResponse(
                request_id=request.id,
                success=True,
                content=result,
                duration_ms=duration,
            )

        except Exception as e:
            logger.error(f"Request {request.id} failed: {e}")
            with self._stats_lock:
                self._stats["errors"] += 1

            # Log error
            self.log_activity(
                subsystem=request.subsystem,
                priority=request.priority.name.lower(),
                preview=f"ERROR: {str(e)[:50]}",
                status="error",
            )

            return APIResponse(
                request_id=request.id,
                success=False,
                content=None,
                error=safe_error_message(e),
            )

    def get_stats(self) -> dict:
        """Get manager statistics."""
        with self._lock:
            queue_size = len(self._request_queue)
            pending_count = len(self._pending_requests)
            cache_size = len(self._cache)
        with self._stats_lock:
            stats_snapshot = dict(self._stats)
            stats_snapshot["queue_size"] = queue_size
            stats_snapshot["pending_requests"] = pending_count
            stats_snapshot["cache_size"] = cache_size
        return stats_snapshot

    def log_activity(self, subsystem: str, priority: str, preview: str,
                     latency_ms: float = 0, cached: bool = False, status: str = "complete",
                     model: str | None = None, tokens: int = 0):
        """Log an activity for the CLI panel."""
        with self._activity_log_lock:
            self._activity_log.appendleft({
                "subsystem": subsystem,
                "priority": priority,
                "preview": preview[:100] if preview else "",
                "latency": int(latency_ms),
                "cached": cached,
                "status": status,
                "timestamp": datetime.now().isoformat(),
                "model": model,
                "tokens": tokens,
            })

    def get_recent_activities(self, limit: int = 20) -> list[dict]:
        """Get recent API activities."""
        with self._activity_log_lock:
            return list(self._activity_log)[:limit]

    def get_active_subsystems(self) -> dict[str, dict]:
        """Get currently active subsystems."""
        active = {}
        now = datetime.now()
        with self._activity_log_lock:
            for activity in list(self._activity_log)[:20]:
                subsystem = activity.get("subsystem", "unknown")
                if subsystem not in active:
                    ts = activity.get("timestamp", "")
                    try:
                        ts_dt = datetime.fromisoformat(ts) if ts else now
                        age_ms = (now - ts_dt).total_seconds() * 1000
                        active[subsystem] = {
                            "active": age_ms < 5000,
                            "last_seen": ts,
                        }
                    except Exception:
                        active[subsystem] = {"active": False, "last_seen": ""}
        return active

    def clear_cache(self):
        """Clear the response cache."""
        with self._cache_lock:
            self._cache.clear()


_default_manager: APICallManager | None = None
_manager_lock = threading.Lock()


def get_api_manager(llm_client=None) -> APICallManager:
    """Get or create the default API call manager."""
    global _default_manager
    if _default_manager is None:
        with _manager_lock:
            if _default_manager is None:
                if llm_client is None:
                    raise ValueError("LLM client required for first initialization")
                _default_manager = APICallManager(llm_client)
    return _default_manager


def api_call(
    subsystem: str,
    priority: Priority = Priority.INTERACTIVE,
):
    """Decorator to wrap async functions with API call management.
    
    Usage:
        @api_call("erdos", Priority.LEARNING)
        async def my_function(llm, prompt):
            return await llm.chat(...)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            manager = get_api_manager()
            if manager is None:
                return await func(*args, **kwargs)

            user_input = kwargs.get("user_input", "") or (args[1] if len(args) > 1 else "")
            system_prompt = kwargs.get("system_prompt", "") or (args[2] if len(args) > 2 else "")

            result = await manager.submit_chat(
                subsystem=subsystem,
                priority=priority,
                user_input=str(user_input),
                system_prompt=str(system_prompt),
            )
            return result
        return wrapper
    return decorator
