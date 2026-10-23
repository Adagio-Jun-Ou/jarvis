"""Tests for Enhanced Execution Router, Circuit Breaker, and Event Emitter."""
import asyncio
import json
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch

from core.execution_router import (
    ExecutionRouter,
    ExecutionStrategy,
    RoutingDecision,
    ExecutionResult,
    EnhancedExecutionRouter,
    CircuitBreaker,
    ExecutionEventEmitter,
)


class TestCircuitBreaker:
    def test_initial_state_closed(self):
        cb = CircuitBreaker()
        assert cb.can_execute("direct") is True

    def test_record_success_stays_closed(self):
        cb = CircuitBreaker()
        cb.record_success("direct")
        assert cb.can_execute("direct") is True
        circuit = cb._get_circuit("direct")
        assert circuit["state"] == CircuitBreaker.State.CLOSED

    def test_opens_after_threshold(self):
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            cb.record_failure("harness")
        assert cb.can_execute("harness") is False
        circuit = cb._get_circuit("harness")
        assert circuit["state"] == CircuitBreaker.State.OPEN

    def test_open_rejects_immediately(self):
        cb = CircuitBreaker(failure_threshold=1)
        cb.record_failure("harness")
        assert cb.can_execute("harness") is False

    def test_half_open_after_timeout(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
        cb.record_failure("harness")
        time.sleep(0.02)
        assert cb.can_execute("harness") is True
        circuit = cb._get_circuit("harness")
        assert circuit["state"] == CircuitBreaker.State.HALF_OPEN

    def test_half_open_success_closes(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
        cb.record_failure("harness")
        time.sleep(0.02)
        cb.can_execute("harness")
        cb.record_success("harness")
        circuit = cb._get_circuit("harness")
        assert circuit["state"] == CircuitBreaker.State.CLOSED

    def test_half_open_failure_reopens(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
        cb.record_failure("harness")
        time.sleep(0.02)
        cb.can_execute("harness")
        cb.record_failure("harness")
        circuit = cb._get_circuit("harness")
        assert circuit["state"] == CircuitBreaker.State.OPEN

    def test_separate_circuits_per_strategy(self):
        cb = CircuitBreaker(failure_threshold=1)
        cb.record_failure("harness")
        assert cb.can_execute("harness") is False
        assert cb.can_execute("direct") is True

    def test_get_status(self):
        cb = CircuitBreaker()
        cb.record_failure("harness")
        status = cb.get_status()
        assert "harness" in status
        assert status["harness"]["failures"] == 1

    def test_half_open_max_calls(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01, half_open_max=1)
        cb.record_failure("harness")
        time.sleep(0.02)
        assert cb.can_execute("harness") is True
        cb.record_half_open_call("harness")
        assert cb.can_execute("harness") is False

    def test_wall_clock_forward_jump_cannot_trip_recovery_early(self, monkeypatch):
        """Regression: recovery used to be timed with time.time(), so a clock
        step forward (NTP correction, manual change) instantly 'elapsed' the
        recovery window. The breaker must decide on the monotonic clock only.
        """
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=60.0)
        cb.record_failure("harness")
        assert cb.can_execute("harness") is False

        real_time = time.time()
        monkeypatch.setattr(time, "time", lambda: real_time + 10_000_000.0)

        # Wall clock says years passed; the breaker must still be OPEN.
        assert cb.can_execute("harness") is False

    def test_wall_clock_backward_jump_cannot_pin_breaker_open(self, monkeypatch):
        """Regression: a backward clock step made last_failure appear to be in
        the future, pinning the breaker OPEN far beyond the recovery window.
        Real monotonic elapsed time must still recover it."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
        cb.record_failure("harness")
        assert cb.can_execute("harness") is False

        real_time = time.time()
        monkeypatch.setattr(time, "time", lambda: real_time - 10_000_000.0)

        time.sleep(0.02)  # real elapsed time exceeds recovery_timeout

        assert cb.can_execute("harness") is True
        circuit = cb._get_circuit("harness")
        assert circuit["state"] == CircuitBreaker.State.HALF_OPEN

    def test_status_exposes_wall_clock_and_open_duration(self):
        """get_status() keeps last_failure as a wall-clock value for display
        and adds open_for_s (monotonic elapsed) — without exposing the
        monotonic stamp itself."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=60.0)
        cb.record_failure("harness")

        status = cb.get_status()
        entry = status["harness"]

        assert entry["state"] == "open"
        assert abs(entry["last_failure"] - time.time()) < 5.0
        assert 0.0 <= entry["open_for_s"] < 5.0
        assert "last_failure_mono" not in entry


class TestExecutionEventEmitter:
    @pytest.mark.asyncio
    async def test_emit_event(self):
        emitter = ExecutionEventEmitter()
        await emitter.emit("test_event", {"key": "value"})
        assert emitter.event_count == 1

    @pytest.mark.asyncio
    async def test_get_events(self):
        emitter = ExecutionEventEmitter()
        await emitter.emit("event_1", {})
        await emitter.emit("event_2", {})
        events = emitter.get_events()
        assert len(events) == 2

    @pytest.mark.asyncio
    async def test_get_events_filtered(self):
        emitter = ExecutionEventEmitter()
        await emitter.emit("routing_decision", {})
        await emitter.emit("execution_started", {})
        events = emitter.get_events(event_type="routing_decision")
        assert len(events) == 1
        assert events[0]["type"] == "routing_decision"

    @pytest.mark.asyncio
    async def test_get_events_limit(self):
        emitter = ExecutionEventEmitter()
        for i in range(10):
            await emitter.emit("event", {"i": i})
        events = emitter.get_events(limit=3)
        assert len(events) == 3

    @pytest.mark.asyncio
    async def test_listener_callback(self):
        emitter = ExecutionEventEmitter()
        received = []
        async def callback(event):
            received.append(event)
        emitter.add_listener(callback)
        await emitter.emit("test", {"data": True})
        assert len(received) == 1
        assert received[0]["type"] == "test"

    @pytest.mark.asyncio
    async def test_sync_listener(self):
        emitter = ExecutionEventEmitter()
        received = []
        def callback(event):
            received.append(event)
        emitter.add_listener(callback)
        await emitter.emit("test", {"data": True})
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_remove_listener(self):
        emitter = ExecutionEventEmitter()
        received = []
        def callback(event):
            received.append(event)
        emitter.add_listener(callback)
        emitter.remove_listener(callback)
        await emitter.emit("test", {})
        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_event_log_bounded(self):
        emitter = ExecutionEventEmitter()
        for i in range(600):
            await emitter.emit("event", {"i": i})
        assert emitter.event_count <= 500

    @pytest.mark.asyncio
    async def test_listener_error_doesnt_break(self):
        emitter = ExecutionEventEmitter()
        async def bad_callback(event):
            raise ValueError("callback error")
        emitter.add_listener(bad_callback)
        await emitter.emit("test", {})
        assert emitter.event_count == 1


class TestEnhancedExecutionRouter:
    def setup_method(self):
        EnhancedExecutionRouter._instance = None
        ExecutionRouter._instance = None

    def test_enhanced_has_circuit_breaker(self):
        router = EnhancedExecutionRouter()
        assert router.circuit_breaker is not None

    def test_enhanced_has_event_emitter(self):
        router = EnhancedExecutionRouter()
        assert router.event_emitter is not None

    def test_get_instance_creates_enhanced(self):
        r1 = EnhancedExecutionRouter.get_instance()
        r2 = EnhancedExecutionRouter.get_instance()
        assert r1 is r2
        EnhancedExecutionRouter._instance = None

    @pytest.mark.asyncio
    async def test_execute_with_circuit_breaker(self):
        mock_agent = MagicMock()
        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(return_value="Response")
        mock_agent.llm = mock_llm
        mock_agent.get_system_prompt = MagicMock(return_value="System")
        mock_agent.backend_llm = None
        mock_agent._intelligent_orchestrator = None
        mock_agent._agentic_harness = None
        mock_agent.memory = MagicMock()
        mock_agent.memory.get_messages_for_llm = MagicMock(return_value=[])
        mock_agent.memory.get_recent_messages = MagicMock(return_value=[])
        mock_agent.tools = {}
        mock_agent.permissions = None
        mock_agent.config = MagicMock()
        mock_agent.config.agent.auto_approve_tools = False
        mock_agent._tool_schemas = []

        router = EnhancedExecutionRouter(agent=mock_agent)
        result = await router.execute("Hello", force_strategy=ExecutionStrategy.DIRECT)
        assert result.success is True
        assert result.strategy_used == ExecutionStrategy.DIRECT

    @pytest.mark.asyncio
    async def test_enhanced_emits_events(self):
        mock_agent = MagicMock()
        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(return_value="Response")
        mock_agent.llm = mock_llm
        mock_agent.get_system_prompt = MagicMock(return_value="System")
        mock_agent.backend_llm = None
        mock_agent._intelligent_orchestrator = None
        mock_agent._agentic_harness = None
        mock_agent.memory = MagicMock()
        mock_agent.memory.get_messages_for_llm = MagicMock(return_value=[])
        mock_agent.memory.get_recent_messages = MagicMock(return_value=[])
        mock_agent.tools = {}
        mock_agent.permissions = None
        mock_agent.config = MagicMock()
        mock_agent.config.agent.auto_approve_tools = False
        mock_agent._tool_schemas = []

        router = EnhancedExecutionRouter(agent=mock_agent)
        await router.execute("Hello", force_strategy=ExecutionStrategy.DIRECT)
        assert router.event_emitter.event_count >= 2

    def test_get_full_status(self):
        router = EnhancedExecutionRouter()
        status = router.get_full_status()
        assert "total_routings" in status
        assert "circuit_breaker" in status
        assert "event_count" in status
        assert "recent_events" in status

    @pytest.mark.asyncio
    async def test_circuit_breaker_blocks_failing_strategy(self):
        router = EnhancedExecutionRouter()
        router._llm_routing_enabled = False
        router._circuit_breaker_enabled = True
        router.circuit_breaker._failure_threshold = 1
        router.circuit_breaker._recovery_timeout = 9999.0

        router.circuit_breaker.record_failure("harness")
        assert router.circuit_breaker.can_execute("harness") is False

        mock_agent = MagicMock()
        mock_agent._intelligent_orchestrator = None
        mock_agent._agentic_harness = None
        router.agent = mock_agent

        result = await router.execute("Build step by step", force_strategy=ExecutionStrategy.HARNESS)
        assert result.success is False


class TestLLMRouting:
    def setup_method(self):
        ExecutionRouter._instance = None
        EnhancedExecutionRouter._instance = None

    @pytest.mark.asyncio
    async def test_route_with_llm_falls_back_to_heuristic(self):
        router = EnhancedExecutionRouter()
        decision = await router.route_with_llm("What is Python?")
        assert decision.strategy is not None
        assert decision.confidence > 0

    @pytest.mark.asyncio
    async def test_route_with_llm_high_confidence_skips_llm(self):
        router = EnhancedExecutionRouter()
        decision = await router.route_with_llm("hi")
        assert decision.strategy == ExecutionStrategy.DIRECT

    @pytest.mark.asyncio
    async def test_route_with_llm_with_mock_llm(self):
        mock_agent = MagicMock()
        mock_llm = AsyncMock()
        response_json = json.dumps({
            "strategy": "harness",
            "confidence": 0.8,
            "reasoning": "Multi-step build task",
            "estimated_steps": 4,
        })
        mock_llm.chat = AsyncMock(return_value=response_json)
        mock_agent.llm = mock_llm
        mock_agent.backend_llm = None

        router = EnhancedExecutionRouter(agent=mock_agent)
        decision = await router.route_with_llm(
            "Build the authentication system and then write tests for it and verify they pass"
        )
        assert decision.strategy in (ExecutionStrategy.HARNESS, ExecutionStrategy.AGENTIC_LOOP)
        assert "Heuristic" in decision.reasoning or decision.confidence > 0

    @pytest.mark.asyncio
    async def test_route_with_llm_invalid_response_uses_heuristic(self):
        mock_agent = MagicMock()
        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(return_value="Invalid response text")
        mock_agent.llm = mock_llm
        mock_agent.backend_llm = None

        router = EnhancedExecutionRouter(agent=mock_agent)
        decision = await router.route_with_llm("Build step by step and verify")
        assert decision.strategy is not None


class TestExecutionRouterRouteWithLLM:
    def setup_method(self):
        ExecutionRouter._instance = None

    @pytest.mark.asyncio
    async def test_route_with_llm_on_base_router(self):
        router = ExecutionRouter()
        decision = await router.route_with_llm("What is Python?")
        assert decision.strategy is not None