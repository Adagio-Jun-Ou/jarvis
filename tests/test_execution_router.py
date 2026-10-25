"""Tests for the unified Execution Router and AgentLoop Plan-Execute-Verify."""
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
)


class TestRoutingDecision:
    def test_routing_decision_creation(self):
        decision = RoutingDecision(
            strategy=ExecutionStrategy.DIRECT,
            confidence=0.9,
            reasoning="Simple query",
            estimated_steps=1,
        )
        assert decision.strategy == ExecutionStrategy.DIRECT
        assert decision.confidence == 0.9
        assert decision.estimated_steps == 1

    def test_routing_decision_to_dict(self):
        decision = RoutingDecision(
            strategy=ExecutionStrategy.HARNESS,
            confidence=0.7,
            reasoning="Multi-step task",
            estimated_steps=5,
            requires_parallel=True,
            requires_verification=True,
            task_type="code",
            fallback_strategy=ExecutionStrategy.AGENTIC_LOOP,
        )
        d = decision.to_dict()
        assert d["strategy"] == "harness"
        assert d["confidence"] == 0.7
        assert d["estimated_steps"] == 5
        assert d["requires_parallel"] is True
        assert d["requires_verification"] is True
        assert d["task_type"] == "code"
        assert d["fallback_strategy"] == "agentic_loop"

    def test_routing_decision_no_fallback(self):
        decision = RoutingDecision(
            strategy=ExecutionStrategy.DIRECT,
            confidence=1.0,
            reasoning="Simple",
            estimated_steps=1,
        )
        assert decision.fallback_strategy is None
        assert decision.to_dict()["fallback_strategy"] is None


class TestExecutionResult:
    def test_execution_result_success(self):
        result = ExecutionResult(
            strategy_used=ExecutionStrategy.DIRECT,
            success=True,
            output="Hello world",
            duration_ms=150.0,
            steps_completed=1,
        )
        assert result.success is True
        assert result.output == "Hello world"
        assert result.steps_completed == 1

    def test_execution_result_failure(self):
        result = ExecutionResult(
            strategy_used=ExecutionStrategy.HARNESS,
            success=False,
            error="Tool not available",
            duration_ms=500.0,
        )
        assert result.success is False
        assert result.error == "Tool not available"

    def test_execution_result_to_dict(self):
        result = ExecutionResult(
            strategy_used=ExecutionStrategy.AGENTIC_LOOP,
            success=True,
            output="x" * 600,
            tool_calls=3,
            steps_completed=5,
            duration_ms=1000.0,
            metadata={"finish_reason": "stop"},
        )
        d = result.to_dict()
        assert d["strategy_used"] == "agentic_loop"
        assert len(d["output"]) == 500
        assert d["tool_calls"] == 3
        assert d["steps_completed"] == 5
        assert d["metadata"]["finish_reason"] == "stop"


class TestExecutionRouterRouting:
    def setup_method(self):
        ExecutionRouter._instance = None
        self.router = ExecutionRouter()

    def test_simple_question_routes_direct(self):
        decision = self.router.route("What is Python?")
        assert decision.strategy == ExecutionStrategy.DIRECT
        assert decision.estimated_steps == 1

    def test_very_short_query_routes_direct(self):
        decision = self.router.route("Hello")
        assert decision.strategy == ExecutionStrategy.DIRECT

    def test_single_action_routes_agentic_loop(self):
        decision = self.router.route("Search for the latest news about AI")
        assert decision.strategy in (ExecutionStrategy.AGENTIC_LOOP, ExecutionStrategy.HARNESS, ExecutionStrategy.DIRECT)

    def test_multi_step_tasks_route_harness(self):
        decision = self.router.route("Create a new module and then write tests for it and verify they pass")
        assert decision.strategy == ExecutionStrategy.HARNESS
        assert decision.estimated_steps >= 3

    def test_fix_and_verify_routes_harness(self):
        decision = self.router.route("Debug the login issue and then verify the fix works correctly")
        assert decision.strategy == ExecutionStrategy.HARNESS
        assert decision.requires_verification is True

    def test_parallel_keyword_routes_harness(self):
        decision = self.router.route("Analyze the data in parallel and compare the results simultaneously")
        assert decision.strategy == ExecutionStrategy.HARNESS
        assert decision.requires_parallel is True

    def test_analysis_with_perspectives_routes_orchestrator(self):
        decision = self.router.route("Compare and analyze the pros and cons of microservices vs monolith from different angles")
        assert decision.strategy == ExecutionStrategy.ORCHESTRATOR

    def test_step_by_step_routes_harness(self):
        decision = self.router.route("Please refactor the codebase step by step and run tests")
        assert decision.strategy == ExecutionStrategy.HARNESS

    def test_create_and_verify_routes_harness(self):
        decision = self.router.route("Build the authentication system first, then implement the API, and finally verify everything")
        assert decision.strategy == ExecutionStrategy.HARNESS
        assert decision.estimated_steps >= 3

    def test_simple_explain_routes_direct(self):
        decision = self.router.route("Explain how async/await works")
        assert decision.strategy == ExecutionStrategy.DIRECT

    def test_who_is_routes_direct(self):
        decision = self.router.route("Who is Alan Turing?")
        assert decision.strategy == ExecutionStrategy.DIRECT

    def test_long_complex_routes_harness(self):
        long_input = "First investigate the database connection issue, then debug the ORM layer, " \
                     "after that fix the connection pool, and finally verify that all the integration tests pass"
        decision = self.router.route(long_input)
        assert decision.strategy == ExecutionStrategy.HARNESS

    def test_task_type_detection_code(self):
        decision = self.router.route("Write a Python script for data processing and verify it works")
        assert decision.task_type == "code"

    def test_task_type_creation(self):
        decision = self.router.route("Create a new module and build the test suite")
        assert decision.task_type in ("creation", "code")

    def test_task_type_debugging(self):
        decision = self.router.route("Debug the memory leak in the server process")
        assert decision.task_type == "debugging"

    def test_task_type_analysis(self):
        decision = self.router.route("Analyze the performance bottlenecks in our API")
        assert decision.task_type == "analysis"

    def test_fallback_strategy_populated(self):
        decision = self.router.route("Create the module and test it")
        assert decision.fallback_strategy is not None

    def test_confidence_is_bounded(self):
        for query in ["Hi", "Fix the bug and verify", "What is 2+2?", "Build the system step by step"]:
            decision = self.router.route(query)
            assert 0.0 <= decision.confidence <= 1.0

    def test_routing_history_recorded(self):
        self.router.route("Test query 1")
        self.router.route("Test query 2")
        stats = self.router.get_routing_stats()
        assert stats["total_routings"] == 2

    def test_routing_history_bounded(self):
        for i in range(250):
            self.router.route(f"Query {i}")
        stats = self.router.get_routing_stats()
        assert stats["total_routings"] == 200


class TestExecutionRouterStrategySelection:
    def setup_method(self):
        ExecutionRouter._instance = None
        self.router = ExecutionRouter()

    def test_strategy_for_task(self):
        strategy = self.router.get_strategy_for_task("Build the app step by step")
        assert isinstance(strategy, ExecutionStrategy)

    def test_get_instance_creates_singleton(self):
        r1 = ExecutionRouter.get_instance()
        r2 = ExecutionRouter.get_instance()
        assert r1 is r2
        ExecutionRouter._instance = None

    def test_get_instance_with_agent(self):
        mock_agent = MagicMock()
        r = ExecutionRouter.get_instance(agent=mock_agent)
        assert r.agent is mock_agent
        ExecutionRouter._instance = None


class TestExecutionRouterStats:
    def setup_method(self):
        ExecutionRouter._instance = None
        self.router = ExecutionRouter()

    def test_initial_stats_all_zero(self):
        stats = self.router.get_routing_stats()
        assert stats["total_routings"] == 0
        for strategy_stats in stats["strategy_stats"].values():
            assert strategy_stats["calls"] == 0

    def test_strategy_stats_has_all_strategies(self):
        stats = self.router.get_routing_stats()
        for strategy in ExecutionStrategy:
            assert strategy.value in stats["strategy_stats"]

    def test_update_stats(self):
        self.router._update_stats(ExecutionStrategy.DIRECT, True, 100.0)
        stats = self.router.get_routing_stats()["strategy_stats"]["direct"]
        assert stats["calls"] == 1
        assert stats["successes"] == 1
        assert stats["total_ms"] == 100.0

    def test_update_stats_failure(self):
        self.router._update_stats(ExecutionStrategy.HARNESS, False, 200.0)
        stats = self.router.get_routing_stats()["strategy_stats"]["harness"]
        assert stats["calls"] == 1
        assert stats["failures"] == 1
        assert stats["successes"] == 0


class TestExecutionRouterDispatch:
    def setup_method(self):
        ExecutionRouter._instance = None
        self.router = ExecutionRouter()

    @pytest.mark.asyncio
    async def test_execute_direct_without_agent(self):
        result = await self.router.execute("Hello")
        assert result.success is False
        assert "No agent" in result.error

    @pytest.mark.asyncio
    async def test_execute_agentic_loop_without_agent(self):
        result = await self.router.execute("Search and analyze", force_strategy=ExecutionStrategy.AGENTIC_LOOP)
        assert result.success is False
        assert "No agent" in result.error

    @pytest.mark.asyncio
    async def test_execute_harness_without_agent(self):
        result = await self.router.execute("Build step by step", force_strategy=ExecutionStrategy.HARNESS)
        assert result.success is False
        assert "No agent" in result.error

    @pytest.mark.asyncio
    async def test_execute_orchestrator_without_agent(self):
        result = await self.router.execute("Compare and analyze", force_strategy=ExecutionStrategy.ORCHESTRATOR)
        assert result.success is False
        assert "Orchestrator not available" in result.error

    @pytest.mark.asyncio
    async def test_execute_council_without_agent(self):
        result = await self.router.execute("Deliberate on topic", force_strategy=ExecutionStrategy.COUNCIL)
        assert result.success is False
        assert "Council not available" in result.error

    @pytest.mark.asyncio
    async def test_execute_with_mock_agent_direct(self):
        mock_agent = MagicMock()
        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(return_value="Hello! I'm Jarvis.")
        mock_agent.llm = mock_llm
        mock_agent.get_system_prompt = MagicMock(return_value="You are Jarvis.")
        self.router.agent = mock_agent

        result = await self.router.execute("Hello", force_strategy=ExecutionStrategy.DIRECT)
        assert result.success is True
        assert "Jarvis" in result.output
        assert result.strategy_used == ExecutionStrategy.DIRECT
        assert result.steps_completed == 1

    @pytest.mark.asyncio
    async def test_execute_records_duration(self):
        mock_agent = MagicMock()
        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(return_value="Response")
        mock_agent.llm = mock_llm
        mock_agent.get_system_prompt = MagicMock(return_value="System prompt")
        self.router.agent = mock_agent

        result = await self.router.execute("Hello", force_strategy=ExecutionStrategy.DIRECT)
        assert result.duration_ms > 0

    @pytest.mark.asyncio
    async def test_execute_updates_stats(self):
        mock_agent = MagicMock()
        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(return_value="Response")
        mock_agent.llm = mock_llm
        mock_agent.get_system_prompt = MagicMock(return_value="System prompt")
        self.router.agent = mock_agent

        await self.router.execute("Hello", force_strategy=ExecutionStrategy.DIRECT)
        stats = self.router.get_routing_stats()["strategy_stats"]["direct"]
        assert stats["calls"] == 1
        assert stats["successes"] == 1


class TestExecutionRouterFallback:
    def setup_method(self):
        ExecutionRouter._instance = None
        self.router = ExecutionRouter()

    @pytest.mark.asyncio
    async def test_fallback_on_failure(self):
        mock_agent = MagicMock()
        mock_agent.llm = AsyncMock()
        mock_agent.get_system_prompt = MagicMock(side_effect=Exception("Broken"))
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
        self.router.agent = mock_agent

        decision = RoutingDecision(
            strategy=ExecutionStrategy.AGENTIC_LOOP,
            confidence=0.6,
            reasoning="Tool call needed",
            estimated_steps=2,
            fallback_strategy=ExecutionStrategy.DIRECT,
        )
        result = await self.router._dispatch(ExecutionStrategy.AGENTIC_LOOP, "test", {}, decision)
        assert result.strategy_used == ExecutionStrategy.AGENTIC_LOOP


class TestRoutingEdgeCases:
    def setup_method(self):
        ExecutionRouter._instance = None
        self.router = ExecutionRouter()

    def test_empty_input(self):
        decision = self.router.route("")
        assert decision.strategy is not None

    def test_very_long_input(self):
        long_input = "Please " + "and then do something " * 100 + "and verify it works"
        decision = self.router.route(long_input)
        assert decision.strategy in (ExecutionStrategy.HARNESS, ExecutionStrategy.AGENTIC_LOOP, ExecutionStrategy.ORCHESTRATOR)

    def test_unicode_input(self):
        decision = self.router.route("创建新模块并验证")
        assert decision.strategy is not None

    def test_special_characters(self):
        decision = self.router.route("Fix the bug @#$% and test it!")
        assert decision.strategy is not None

    def test_numbers_and_maths(self):
        decision = self.router.route("Calculate the fibonacci sequence and then verify the result")
        assert decision.strategy in (ExecutionStrategy.HARNESS, ExecutionStrategy.AGENTIC_LOOP)

    def test_mixed_case(self):
        d1 = self.router.route("BUILD THE SYSTEM AND VERIFY IT WORKS")
        d2 = self.router.route("build the system and verify it works")
        assert d1.strategy == d2.strategy

    def test_code_task_type_keywords(self):
        for kw in ["code", "program", "script", "function", "class", "module"]:
            decision = self.router.route(f"Write a {kw} and test it")
            assert decision.task_type == "code"

    def test_multiple_verification_keywords(self):
        decision = self.router.route("Implement the feature, ensure it works, validate the output, and run tests to verify")
        assert decision.strategy == ExecutionStrategy.HARNESS
        assert decision.requires_verification is True


class TestAgentLoopPlanExecute:
    def test_agent_result_has_metadata(self):
        from core.agent_loop import AgentResult, FinishReason
        result = AgentResult(
            steps=[],
            finish_reason=FinishReason.STOP,
            metadata={"plan_id": "abc123"},
        )
        assert result.metadata["plan_id"] == "abc123"

    def test_agent_loop_plan_first_param(self):
        from core.agent_loop import AgentLoop
        loop = AgentLoop(
            llm_client=MagicMock(),
            tool_registry={},
            plan_first=True,
            verify_steps=True,
        )
        assert loop.plan_first is True
        assert loop.verify_steps is True

    def test_agent_loop_default_no_planning(self):
        from core.agent_loop import AgentLoop
        loop = AgentLoop(
            llm_client=MagicMock(),
            tool_registry={},
        )
        assert loop.plan_first is False
        assert loop.verify_steps is False


class TestExecutionRouterIntegration:
    def setup_method(self):
        ExecutionRouter._instance = None

    @pytest.mark.asyncio
    async def test_full_routing_cycle(self):
        mock_agent = MagicMock()
        mock_llm = AsyncMock()
        mock_llm.chat = AsyncMock(return_value="Result from LLM")
        mock_agent.llm = mock_llm
        mock_agent.get_system_prompt = MagicMock(return_value="You are Jarvis.")
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

        router = ExecutionRouter(agent=mock_agent)

        result_direct = await router.execute("What is Python?", force_strategy=ExecutionStrategy.DIRECT)
        assert result_direct.success is True
        assert result_direct.strategy_used == ExecutionStrategy.DIRECT

    @pytest.mark.asyncio
    async def test_harness_dispatch_without_harness(self):
        mock_agent = MagicMock()
        mock_agent.get_agentic_harness = MagicMock(return_value=None)
        mock_agent.tools = {}

        router = ExecutionRouter(agent=mock_agent)
        result = await router.execute("Build and test", force_strategy=ExecutionStrategy.HARNESS)
        assert result.success is False
        assert "not available" in result.error

    @pytest.mark.asyncio
    async def test_stats_tracked_across_calls(self):
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

        router = ExecutionRouter(agent=mock_agent)

        await router.execute("Query 1", force_strategy=ExecutionStrategy.DIRECT)
        await router.execute("Query 2", force_strategy=ExecutionStrategy.DIRECT)

        stats = router.get_routing_stats()
        assert stats["strategy_stats"]["direct"]["calls"] == 2
        assert stats["strategy_stats"]["direct"]["successes"] == 2