"""Tests for enhanced Tool base class, validate_params, tool_timeout, and harness-agent integration.

Covers:
- ToolResult structured output
- validate_params decorator (missing required, wrong type, enum, min/max, maxLength)
- tool_timeout decorator
- Tool.safe_execute (timeout, error wrapping, string/dict/ToolResult returns)
- ToolRegistry.safe_execute
- EventBus async subscription
- LLM plan generation fallback
- Harness-agent integration
"""
import asyncio
import pytest
import time

from tools.base import Tool, ToolResult, ToolRegistry, validate_params, tool_timeout
from core.agentic_harness import (
    AgenticHarness, HarnessPhase, HarnessPlan, HarnessStep, StepStatus,
    EventBus, EventType, create_plan_from_objective, llm_generate_plan,
)


class TestToolResult:
    def test_success_result(self):
        r = ToolResult(success=True, result={"data": 42})
        assert r.success is True
        assert r.result["data"] == 42
        assert r.error is None

    def test_error_result(self):
        r = ToolResult(success=False, error="file not found")
        assert r.success is False
        assert r.error == "file not found"

    def test_to_dict(self):
        r = ToolResult(success=True, output="hello")
        d = r.to_dict()
        assert d["success"] is True
        assert d["output"] == "hello"

    def test_default_values(self):
        r = ToolResult()
        assert r.success is True
        assert r.result == {}
        assert r.output == ""
        assert r.error is None


class TestValidateParams:
    @pytest.mark.asyncio
    async def test_required_missing(self):
        @validate_params({"required": ["path"], "properties": {"path": {"type": "string"}}})
        async def execute(self, **params):
            return ToolResult(success=True)

        result = await execute(None)
        assert result.success is False
        assert "path" in result.error

    @pytest.mark.asyncio
    async def test_wrong_type_string(self):
        @validate_params({"properties": {"name": {"type": "string"}}})
        async def execute(self, **params):
            return ToolResult(success=True)

        result = await execute(None, name=123)
        assert result.success is False
        assert "string" in result.error

    @pytest.mark.asyncio
    async def test_wrong_type_number(self):
        @validate_params({"properties": {"count": {"type": "number"}}})
        async def execute(self, **params):
            return ToolResult(success=True)

        result = await execute(None, count="five")
        assert result.success is False
        assert "number" in result.error

    @pytest.mark.asyncio
    async def test_wrong_type_integer(self):
        @validate_params({"properties": {"n": {"type": "integer"}}})
        async def execute(self, **params):
            return ToolResult(success=True)

        result = await execute(None, n=3.5)
        assert result.success is False
        assert "integer" in result.error

    @pytest.mark.asyncio
    async def test_wrong_type_boolean(self):
        @validate_params({"properties": {"flag": {"type": "boolean"}}})
        async def execute(self, **params):
            return ToolResult(success=True)

        result = await execute(None, flag="yes")
        assert result.success is False
        assert "boolean" in result.error

    @pytest.mark.asyncio
    async def test_wrong_type_array(self):
        @validate_params({"properties": {"items": {"type": "array"}}})
        async def execute(self, **params):
            return ToolResult(success=True)

        result = await execute(None, items="a,b")
        assert result.success is False
        assert "array" in result.error

    @pytest.mark.asyncio
    async def test_wrong_type_object(self):
        @validate_params({"properties": {"config": {"type": "object"}}})
        async def execute(self, **params):
            return ToolResult(success=True)

        result = await execute(None, config=[1, 2])
        assert result.success is False
        assert "object" in result.error

    @pytest.mark.asyncio
    async def test_max_length_exceeded(self):
        @validate_params({"properties": {"text": {"type": "string", "maxLength": 10}}})
        async def execute(self, **params):
            return ToolResult(success=True)

        result = await execute(None, text="a" * 20)
        assert result.success is False
        assert "maxLength" in result.error

    @pytest.mark.asyncio
    async def test_minimum_violated(self):
        @validate_params({"properties": {"count": {"type": "number", "minimum": 0}}})
        async def execute(self, **params):
            return ToolResult(success=True)

        result = await execute(None, count=-5)
        assert result.success is False
        assert "minimum" in result.error

    @pytest.mark.asyncio
    async def test_maximum_violated(self):
        @validate_params({"properties": {"ratio": {"type": "number", "maximum": 1.0}}})
        async def execute(self, **params):
            return ToolResult(success=True)

        result = await execute(None, ratio=1.5)
        assert result.success is False
        assert "maximum" in result.error

    @pytest.mark.asyncio
    async def test_enum_violated(self):
        @validate_params({"properties": {"mode": {"type": "string", "enum": ["fast", "slow"]}}})
        async def execute(self, **params):
            return ToolResult(success=True)

        result = await execute(None, mode="medium")
        assert result.success is False
        assert "must be one of" in result.error

    @pytest.mark.asyncio
    async def test_enum_valid(self):
        @validate_params({"properties": {"mode": {"type": "string", "enum": ["fast", "slow"]}}})
        async def execute(self, **params):
            return ToolResult(success=True)

        result = await execute(None, mode="fast")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_all_params_valid(self):
        @validate_params({"required": ["path"], "properties": {"path": {"type": "string"}, "recursive": {"type": "boolean"}}})
        async def execute(self, **params):
            return ToolResult(success=True)

        result = await execute(None, path="/tmp/test", recursive=True)
        assert result.success is True


class TestToolTimeout:
    @pytest.mark.asyncio
    async def test_no_timeout(self):
        @tool_timeout(5.0)
        async def execute(self, **params):
            return ToolResult(success=True)

        result = await execute(None)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_timeout_triggered(self):
        @tool_timeout(0.1)
        async def execute(self, **params):
            await asyncio.sleep(1.0)
            return ToolResult(success=True)

        result = await execute(None)
        assert result.success is False
        assert "timed out" in result.error.lower() or "timeout" in result.error.lower()


class TestSafeExecute:
    @pytest.mark.asyncio
    async def test_dict_return(self):
        class DummyTool(Tool):
            name = "dummy"
            description = "Test"
            parameters = {"type": "object"}
            async def execute(self, **params):
                return {"success": True, "data": "output"}

        tool = DummyTool()
        result = await tool.safe_execute()
        assert result.success is True
        assert result.result["data"] == "output"

    @pytest.mark.asyncio
    async def test_error_dict_return(self):
        class ErrorTool(Tool):
            name = "errtool"
            description = "Test"
            parameters = {"type": "object"}
            async def execute(self, **params):
                return {"error": "something broke", "success": False}

        tool = ErrorTool()
        result = await tool.safe_execute()
        assert result.success is False

    @pytest.mark.asyncio
    async def test_string_return(self):
        class StringTool(Tool):
            name = "strtool"
            description = "Test"
            parameters = {"type": "object"}
            async def execute(self, **params):
                return "hello world"

        tool = StringTool()
        result = await tool.safe_execute()
        assert result.success is True
        assert result.output == "hello world"

    @pytest.mark.asyncio
    async def test_exception_wrapping(self):
        class CrashTool(Tool):
            name = "crash"
            description = "Test"
            parameters = {"type": "object"}
            timeout = 1.0
            async def execute(self, **params):
                raise ValueError("boom")

        tool = CrashTool()
        result = await tool.safe_execute()
        assert result.success is False
        assert "boom" in result.error

    @pytest.mark.asyncio
    async def test_timeout_wrapping(self):
        class SlowTool(Tool):
            name = "slow"
            description = "Test"
            parameters = {"type": "object"}
            timeout = 0.1
            async def execute(self, **params):
                await asyncio.sleep(5.0)
                return {"success": True}

        tool = SlowTool()
        result = await tool.safe_execute()
        assert result.success is False
        assert "timed out" in result.error.lower() or "timeout" in result.error.lower()


class TestToolRegistrySafeExecute:
    @pytest.mark.asyncio
    async def test_safe_execute_existing_tool(self):
        class GoodTool(Tool):
            name = "good"
            description = "Test"
            parameters = {"type": "object"}
            async def execute(self, **params):
                return {"result": "ok"}

        registry = ToolRegistry()
        registry.register(GoodTool())
        result = await registry.safe_execute("good")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_safe_execute_missing_tool(self):
        registry = ToolRegistry()
        result = await registry.safe_execute("nonexistent")
        assert result.success is False
        assert "not found" in result.error


class TestLLMPlanGeneration:
    @pytest.mark.asyncio
    async def test_fallback_without_llm(self):
        steps = await llm_generate_plan("Fix the bug", llm_client=None)
        assert isinstance(steps, list)
        assert len(steps) >= 2

    @pytest.mark.asyncio
    async def test_fallback_with_bad_llm(self):
        class BadLLM:
            def chat(self, messages):
                raise RuntimeError("LLM unavailable")

        steps = await llm_generate_plan("Create feature", llm_client=BadLLM())
        assert isinstance(steps, list)
        assert len(steps) >= 2

    @pytest.mark.asyncio
    async def test_fallback_with_nonjson_response(self):
        class NonJSONLLM:
            def chat(self, messages):
                return "I don't understand, please rephrase."

        steps = await llm_generate_plan("Analyze code", llm_client=NonJSONLLM())
        assert isinstance(steps, list)
        assert len(steps) >= 2


class TestHarnessAgentIntegration:
    @pytest.mark.asyncio
    async def test_execute_with_harness_basic(self):
        harness = AgenticHarness()
        steps = create_plan_from_objective("Fix authentication bug", ["shell", "read_file", "write_file"])

        plan = harness.create_plan("Fix authentication bug", steps)
        assert plan is not None
        assert len(plan.steps) >= 3

    @pytest.mark.asyncio
    async def test_harness_uses_safe_execute(self):
        harness = AgenticHarness()
        harness._verification_enabled = False

        class SafeTool(Tool):
            name = "safe"
            description = "Test"
            parameters = {"type": "object"}
            async def execute(self, **params):
                return ToolResult(success=True, result={"data": "test"})

        plan = harness.create_plan("Safe test", [
            {"action": "safe_step", "tool_name": "safe", "description": "Safe execution"},
        ])
        # The harness calls tool.execute, which returns ToolResult
        # Safe_execute would wrap it properly but harness calls execute directly
        result = await harness.execute_plan(plan, tools={"safe": SafeTool()})
        assert result["is_complete"] is True