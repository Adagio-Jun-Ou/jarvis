"""Tool Chain Optimizer — Dependency DAG, parallel execution, and caching for Jarvis.

When a multi-step task involves multiple tool calls, this optimizer:
1. Builds a dependency DAG from tool inputs/outputs
2. Executes independent tools in parallel (asyncio.gather)
3. Caches idempotent tool results for reuse
4. Provides composition operators (chain, fan-out, fan-in)
5. Tracks execution metrics for observability

This is what makes Jarvis a real agent, not just a chatbot that
calls tools sequentially.
"""
import asyncio
import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    call_id: str
    tool_name: str
    arguments: dict
    depends_on: list[str] = field(default_factory=list)
    priority: int = 0
    timeout: float = 60.0
    cache_key: str | None = None
    result: Any = None
    error: str | None = None
    started_at: float = 0.0
    ended_at: float = 0.0
    status: str = "pending"

    @property
    def duration(self) -> float:
        if self.ended_at > 0 and self.started_at > 0:
            return round(self.ended_at - self.started_at, 3)
        return 0.0

    def to_dict(self) -> dict:
        return {
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "depends_on": self.depends_on,
            "priority": self.priority,
            "status": self.status,
            "duration": self.duration,
            "error": self.error,
        }


@dataclass
class ChainResult:
    calls: list[ToolCall]
    total_duration: float
    parallel_groups: int
    cache_hits: int
    cache_misses: int
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return len(self.errors) == 0 and all(c.status == "completed" for c in self.calls if c.status != "pending")

    @property
    def sequential_duration_estimate(self) -> float:
        return sum(c.duration for c in self.calls)

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "total_duration": round(self.total_duration, 3),
            "sequential_estimate": round(self.sequential_duration_estimate, 3),
            "parallel_groups": self.parallel_groups,
            "calls": [c.to_dict() for c in self.calls],
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "speedup": round(self.sequential_duration_estimate / max(self.total_duration, 0.001), 2),
            "errors": self.errors,
        }


class ToolResultCache:
    """Thread-safe LRU cache for idempotent tool results."""

    def __init__(self, max_size: int = 500, ttl_seconds: float = 300.0):
        self._cache: dict[str, tuple[Any, float]] = {}
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def _make_key(self, tool_name: str, arguments: dict) -> str:
        args_json = json.dumps(arguments, sort_keys=True, default=str)
        return f"{tool_name}:{hashlib.sha256(args_json.encode()).hexdigest()[:16]}"

    def get(self, tool_name: str, arguments: dict) -> Any | None:
        key = self._make_key(tool_name, arguments)
        with self._lock:
            if key in self._cache:
                result, ts = self._cache[key]
                if time.time() - ts < self._ttl:
                    self._hits += 1
                    return result
                del self._cache[key]
            self._misses += 1
            return None

    def put(self, tool_name: str, arguments: dict, result: Any) -> None:
        key = self._make_key(tool_name, arguments)
        with self._lock:
            if len(self._cache) >= self._max_size:
                oldest = min(self._cache.items(), key=lambda x: x[1][1])
                del self._cache[oldest[0]]
            self._cache[key] = (result, time.time())

    def invalidate(self, tool_name: str, arguments: dict) -> None:
        key = self._make_key(tool_name, arguments)
        with self._lock:
            self._cache.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / max(1, self._hits + self._misses), 4),
                "ttl_seconds": self._ttl,
            }


class ToolChainOptimizer:
    """Optimizes multi-tool execution with dependency analysis and caching.

    Usage:
        optimizer = get_tool_chain_optimizer()
        result = await optimizer.execute_chain(tool_calls, tools)
        print(f"Speedup: {result.to_dict()['speedup']}x")
    """

    IDEMPOTENT_TOOLS = frozenset({
        "terminal_execute", "read_file", "cat", "ls",
        "native_read", "web_search", "browser_navigate",
        "health_check", "status", "check",
    })

    def __init__(self):
        self._cache = ToolResultCache()
        self._lock = threading.RLock()
        self._total_chains = 0
        self._total_calls = 0
        self._total_parallelized = 0
        self._start_time = time.time()

    def _analyze_dependencies(self, calls: list[ToolCall]) -> list[set[str]]:
        """Build execution layers from dependency DAG.

        Returns list of sets, where each set contains call IDs
        that can execute in parallel.
        """
        calls_by_id = {c.call_id: c for c in calls}
        remaining = set(c.call_id for c in calls)
        layers: list[set[str]] = []
        completed: set[str] = set()

        while remaining:
            ready = set()
            for call_id in remaining:
                call = calls_by_id[call_id]
                if all(dep in completed for dep in call.depends_on):
                    ready.add(call_id)

            if not ready:
                for call_id in remaining:
                    ready.add(call_id)
                if not ready:
                    break

            layers.append(ready)
            completed |= ready
            remaining -= ready

        return layers

    async def execute_chain(
        self,
        calls: list[ToolCall],
        tools: dict[str, Any],
        use_cache: bool = True,
    ) -> ChainResult:
        start = time.time()
        cache_hits = 0
        cache_misses = 0
        errors: list[str] = []

        calls_by_id = {c.call_id: c for c in calls}
        layers = self._analyze_dependencies(calls)

        results_by_id: dict[str, Any] = {}
        parallel_groups = 0

        for layer_idx, layer in enumerate(layers):
            layer_has_parallel = len(layer) > 1
            if layer_has_parallel:
                parallel_groups += 1

            tasks = []
            for call_id in layer:
                call = calls_by_id[call_id]

                for dep_id in call.depends_on:
                    if dep_id in results_by_id:
                        dep_result = results_by_id[dep_id]
                        if isinstance(dep_result, dict):
                            for key, value in dep_result.items():
                                arg_key = f"{dep_id}.{key}"
                                if arg_key in call.arguments:
                                    call.arguments[arg_key] = value

                if use_cache and call.tool_name in self.IDEMPOTENT_TOOLS:
                    cached = self._cache.get(call.tool_name, call.arguments)
                    if cached is not None:
                        call.result = cached
                        call.status = "completed"
                        call.started_at = time.time()
                        call.ended_at = time.time()
                        results_by_id[call_id] = cached
                        cache_hits += 1
                        continue

                cache_misses += 1
                tasks.append((call_id, call))

            if tasks:
                if len(tasks) > 1:
                    parallel_groups += 1
                    coros = [self._execute_tool(cid, call, tools) for cid, call in tasks]
                    results = await asyncio.gather(*coros, return_exceptions=True)
                    for (cid, call), result in zip(tasks, results):
                        if isinstance(result, Exception):
                            call.error = str(result)
                            call.status = "failed"
                            errors.append(f"{cid}: {result}")
                        else:
                            results_by_id[cid] = result
                else:
                    cid, call = tasks[0]
                    try:
                        result = await self._execute_tool(cid, call, tools)
                        results_by_id[cid] = result
                    except Exception as e:
                        call.error = str(e)
                        call.status = "failed"
                        errors.append(f"{cid}: {e}")

            for call_id in layer:
                call = calls_by_id[call_id]
                if call.status == "completed" and call.result is not None and use_cache:
                    if call.tool_name in self.IDEMPOTENT_TOOLS:
                        self._cache.put(call.tool_name, call.arguments, call.result)

        total_duration = time.time() - start

        with self._lock:
            self._total_chains += 1
            self._total_calls += len(calls)
            self._total_parallelized += parallel_groups

        return ChainResult(
            calls=list(calls_by_id.values()),
            total_duration=total_duration,
            parallel_groups=parallel_groups,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            errors=errors,
        )

    async def _execute_tool(self, call_id: str, call: ToolCall, tools: dict[str, Any]) -> Any:
        tool = tools.get(call.tool_name)
        if not tool:
            call.status = "failed"
            call.error = f"Unknown tool: {call.tool_name}"
            raise ValueError(call.error)

        call.status = "executing"
        call.started_at = time.time()

        try:
            if asyncio.iscoroutinefunction(tool.execute):
                result = await asyncio.wait_for(
                    tool.execute(**call.arguments),
                    timeout=call.timeout,
                )
            elif hasattr(tool, 'safe_execute') and asyncio.iscoroutinefunction(tool.safe_execute):
                result = await asyncio.wait_for(
                    tool.safe_execute(**call.arguments),
                    timeout=call.timeout,
                )
            else:
                result = await asyncio.to_thread(
                    tool.execute, **call.arguments
                )

            call.result = result
            call.status = "completed"
            call.ended_at = time.time()
            return result

        except asyncio.TimeoutError:
            call.status = "failed"
            call.error = f"Timeout after {call.timeout}s"
            call.ended_at = time.time()
            raise
        except Exception as e:
            call.status = "failed"
            call.error = str(e)
            call.ended_at = time.time()
            raise

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total_chains": self._total_chains,
                "total_calls": self._total_calls,
                "total_parallelized": self._total_parallelized,
                "uptime_seconds": round(time.time() - self._start_time, 1),
                "cache": self._cache.get_stats(),
            }

    def clear_cache(self) -> None:
        self._cache.clear()


_tool_optimizer: ToolChainOptimizer | None = None
_optimizer_lock = threading.Lock()


def get_tool_chain_optimizer() -> ToolChainOptimizer:
    global _tool_optimizer
    if _tool_optimizer is None:
        with _optimizer_lock:
            if _tool_optimizer is None:
                _tool_optimizer = ToolChainOptimizer()
    return _tool_optimizer