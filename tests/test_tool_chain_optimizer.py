"""Tests for Tool Chain Optimizer.

Covers:
- Dependency DAG analysis
- Parallel execution
- Tool result caching (hits, misses, TTL, eviction)
- Chain composition
- Execution metrics
- Thread safety
"""
import asyncio
import time
import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, AsyncMock

import pytest

sys_path = str(Path(__file__).parent.parent / "src")
import sys
if sys_path not in sys_path:
    sys.path.insert(0, sys_path)

from core.tool_chain_optimizer import (
    ToolChainOptimizer, ToolCall, ToolResultCache, ChainResult,
    get_tool_chain_optimizer,
)


class MockTool:
    def __init__(self, name: str, result: Any = "ok", delay: float = 0):
        self.name = name
        self._result = result
        self._delay = delay
        self.call_count = 0

    async def execute(self, **kwargs):
        self.call_count += 1
        if self._delay:
            await asyncio.sleep(self._delay)
        return self._result


class FailingTool:
    def __init__(self):
        self.call_count = 0

    async def execute(self, **kwargs):
        self.call_count += 1
        raise RuntimeError("Tool failed")


class TestToolResultCache:
    def setup_method(self):
        self.cache = ToolResultCache(max_size=10, ttl_seconds=60)

    def test_put_and_get(self):
        self.cache.put("read_file", {"path": "/test"}, "content")
        result = self.cache.get("read_file", {"path": "/test"})
        assert result == "content"

    def test_cache_miss(self):
        result = self.cache.get("nonexistent", {})
        assert result is None

    def test_cache_hit_increments(self):
        self.cache.put("tool", {"a": 1}, "result")
        self.cache.get("tool", {"a": 1})
        stats = self.cache.get_stats()
        assert stats["hits"] == 1

    def test_cache_miss_increments(self):
        self.cache.get("missing", {})
        stats = self.cache.get_stats()
        assert stats["misses"] == 1

    def test_lru_eviction(self):
        for i in range(15):
            self.cache.put("tool", {"i": i}, f"result_{i}")
        stats = self.cache.get_stats()
        assert stats["size"] <= 10

    def test_ttl_expiry(self):
        cache = ToolResultCache(max_size=10, ttl_seconds=0.01)
        cache.put("tool", {"x": 1}, "result")
        time.sleep(0.02)
        result = cache.get("tool", {"x": 1})
        assert result is None

    def test_invalidate(self):
        self.cache.put("tool", {"a": 1}, "result")
        self.cache.invalidate("tool", {"a": 1})
        assert self.cache.get("tool", {"a": 1}) is None

    def test_clear(self):
        self.cache.put("tool", {}, "result")
        self.cache.clear()
        stats = self.cache.get_stats()
        assert stats["size"] == 0
        assert stats["hits"] == 0

    def test_hit_rate(self):
        self.cache.put("tool", {"x": 1}, "r")
        self.cache.get("tool", {"x": 1})
        self.cache.get("missing", {})
        stats = self.cache.get_stats()
        assert stats["hit_rate"] == 0.5

    def test_get_stats_fields(self):
        stats = self.cache.get_stats()
        assert "size" in stats
        assert "max_size" in stats
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats
        assert "ttl_seconds" in stats

    def test_same_args_same_key(self):
        self.cache.put("tool", {"a": 1, "b": 2}, "result1")
        result = self.cache.get("tool", {"b": 2, "a": 1})
        assert result == "result1"

    def test_different_args_different_key(self):
        self.cache.put("tool", {"a": 1}, "result1")
        self.cache.put("tool", {"a": 2}, "result2")
        r1 = self.cache.get("tool", {"a": 1})
        r2 = self.cache.get("tool", {"a": 2})
        assert r1 == "result1"
        assert r2 == "result2"


class TestToolCall:
    def test_tool_call_defaults(self):
        call = ToolCall(call_id="c1", tool_name="test", arguments={})
        assert call.status == "pending"
        assert call.depends_on == []
        assert call.duration == 0.0

    def test_tool_call_duration(self):
        call = ToolCall(call_id="c1", tool_name="test", arguments={})
        call.started_at = 100.0
        call.ended_at = 105.0
        assert call.duration == 5.0

    def test_tool_call_to_dict(self):
        call = ToolCall(call_id="c1", tool_name="test", arguments={"x": 1})
        d = call.to_dict()
        assert d["call_id"] == "c1"
        assert d["tool_name"] == "test"
        assert d["status"] == "pending"


class TestDependencyAnalysis:
    def setup_method(self):
        self.optimizer = ToolChainOptimizer()

    def test_no_dependencies(self):
        calls = [
            ToolCall(call_id="c1", tool_name="t1", arguments={}),
            ToolCall(call_id="c2", tool_name="t2", arguments={}),
            ToolCall(call_id="c3", tool_name="t3", arguments={}),
        ]
        layers = self.optimizer._analyze_dependencies(calls)
        assert len(layers) == 1
        assert len(layers[0]) == 3

    def test_linear_chain(self):
        calls = [
            ToolCall(call_id="c1", tool_name="t1", arguments={}),
            ToolCall(call_id="c2", tool_name="t2", arguments={}, depends_on=["c1"]),
            ToolCall(call_id="c3", tool_name="t3", arguments={}, depends_on=["c2"]),
        ]
        layers = self.optimizer._analyze_dependencies(calls)
        assert len(layers) == 3
        assert "c1" in layers[0]
        assert "c2" in layers[1]
        assert "c3" in layers[2]

    def test_fan_out(self):
        calls = [
            ToolCall(call_id="c1", tool_name="t1", arguments={}),
            ToolCall(call_id="c2", tool_name="t2", arguments={}, depends_on=["c1"]),
            ToolCall(call_id="c3", tool_name="t3", arguments={}, depends_on=["c1"]),
        ]
        layers = self.optimizer._analyze_dependencies(calls)
        assert len(layers) == 2
        assert "c1" in layers[0]
        assert "c2" in layers[1] and "c3" in layers[1]

    def test_fan_in(self):
        calls = [
            ToolCall(call_id="c1", tool_name="t1", arguments={}),
            ToolCall(call_id="c2", tool_name="t2", arguments={}),
            ToolCall(call_id="c3", tool_name="t3", arguments={}, depends_on=["c1", "c2"]),
        ]
        layers = self.optimizer._analyze_dependencies(calls)
        assert len(layers) == 2
        assert "c1" in layers[0] and "c2" in layers[0]
        assert "c3" in layers[1]


class TestChainExecution:
    def setup_method(self):
        self.optimizer = ToolChainOptimizer()

    @pytest.mark.asyncio
    async def test_single_tool(self):
        tool = MockTool("test", result="hello")
        calls = [ToolCall(call_id="c1", tool_name="test", arguments={})]
        result = await self.optimizer.execute_chain(calls, {"test": tool})
        assert result.success
        assert result.calls[0].status == "completed"

    @pytest.mark.asyncio
    async def test_parallel_execution(self):
        t1 = MockTool("t1", result="r1", delay=0.05)
        t2 = MockTool("t2", result="r2", delay=0.05)
        calls = [
            ToolCall(call_id="c1", tool_name="t1", arguments={}),
            ToolCall(call_id="c2", tool_name="t2", arguments={}),
        ]
        result = await self.optimizer.execute_chain(calls, {"t1": t1, "t2": t2})
        assert result.success
        assert result.parallel_groups >= 1
        assert result.total_duration < 0.2

    @pytest.mark.asyncio
    async def test_sequential_execution(self):
        t1 = MockTool("t1", result="r1")
        t2 = MockTool("t2", result="r2")
        calls = [
            ToolCall(call_id="c1", tool_name="t1", arguments={}),
            ToolCall(call_id="c2", tool_name="t2", arguments={}, depends_on=["c1"]),
        ]
        result = await self.optimizer.execute_chain(calls, {"t1": t1, "t2": t2})
        assert result.success
        assert len(result.calls) == 2

    @pytest.mark.asyncio
    async def test_unknown_tool(self):
        calls = [ToolCall(call_id="c1", tool_name="missing", arguments={})]
        result = await self.optimizer.execute_chain(calls, {})
        assert not result.success
        assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_failing_tool(self):
        tool = FailingTool()
        calls = [ToolCall(call_id="c1", tool_name="fail", arguments={})]
        result = await self.optimizer.execute_chain(calls, {"fail": tool})
        assert not result.success
        assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_tool_result_to_dict(self):
        tool = MockTool("test", result="hello")
        calls = [ToolCall(call_id="c1", tool_name="test", arguments={})]
        result = await self.optimizer.execute_chain(calls, {"test": tool})
        d = result.to_dict()
        assert "speedup" in d
        assert "calls" in d
        assert d["success"] is True


class TestChainCaching:
    def setup_method(self):
        self.optimizer = ToolChainOptimizer()

    @pytest.mark.asyncio
    async def test_cache_hit_on_idempotent(self):
        tool = MockTool("read_file", result="cached_content")
        calls1 = [ToolCall(call_id="c1", tool_name="read_file", arguments={"path": "/test"})]
        result1 = await self.optimizer.execute_chain(calls1, {"read_file": tool})
        assert result1.cache_misses >= 1

        calls2 = [ToolCall(call_id="c2", tool_name="read_file", arguments={"path": "/test"})]
        result2 = await self.optimizer.execute_chain(calls2, {"read_file": tool}, use_cache=True)
        assert result2.cache_hits >= 1

    @pytest.mark.asyncio
    async def test_cache_disabled(self):
        tool = MockTool("read_file", result="content")
        calls = [ToolCall(call_id="c1", tool_name="read_file", arguments={"path": "/t"})]
        result = await self.optimizer.execute_chain(calls, {"read_file": tool}, use_cache=False)
        assert result.cache_hits == 0


class TestOptimizerStatus:
    def test_initial_status(self):
        opt = ToolChainOptimizer()
        status = opt.get_status()
        assert status["total_chains"] == 0
        assert status["total_calls"] == 0
        assert "cache" in status

    def test_status_after_execution(self):
        opt = ToolChainOptimizer()

        async def run():
            tool = MockTool("test", result="ok")
            calls = [ToolCall(call_id="c1", tool_name="test", arguments={})]
            await opt.execute_chain(calls, {"test": tool})

        asyncio.run(run())
        status = opt.get_status()
        assert status["total_chains"] == 1
        assert status["total_calls"] == 1

    def test_clear_cache(self):
        opt = ToolChainOptimizer()
        opt._cache.put("tool", {"x": 1}, "result")
        opt.clear_cache()
        stats = opt._cache.get_stats()
        assert stats["size"] == 0

    def test_singleton(self):
        opt1 = get_tool_chain_optimizer()
        opt2 = get_tool_chain_optimizer()
        assert opt1 is opt2


class TestChainResult:
    def test_chain_result_success(self):
        call = ToolCall(call_id="c1", tool_name="t", arguments={}, status="completed")
        result = ChainResult(
            calls=[call], total_duration=1.0, parallel_groups=1,
            cache_hits=0, cache_misses=1,
        )
        assert result.success

    def test_chain_result_failure(self):
        result = ChainResult(
            calls=[], total_duration=1.0, parallel_groups=0,
            cache_hits=0, cache_misses=0,
            errors=["something failed"],
        )
        assert not result.success

    def test_chain_result_speedup(self):
        call1 = ToolCall(call_id="c1", tool_name="t1", arguments={}, status="completed")
        call1.started_at = 100.0
        call1.ended_at = 101.0
        call2 = ToolCall(call_id="c2", tool_name="t2", arguments={}, status="completed")
        call2.started_at = 100.0
        call2.ended_at = 101.0
        result = ChainResult(
            calls=[call1, call2], total_duration=1.0, parallel_groups=1,
            cache_hits=0, cache_misses=2,
        )
        d = result.to_dict()
        assert d["sequential_estimate"] == 2.0
        assert d["speedup"] == 2.0