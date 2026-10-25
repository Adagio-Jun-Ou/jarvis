"""Tests for 2025 AI Architecture Improvements.

Tests for:
1. Evaluator-Optimizer Loop (Anthropic pattern)
2. Structured ReAct Engine with persistent traces
3. Adaptive Thinking Budget
4. API Manager parallel batch processing
5. LCM Planner LLM synthesis
"""
import asyncio
import threading
import time
import pytest








class TestAPIManagerParallelBatch:
    """Test that API Manager processes batches in parallel."""

    def test_process_single_request_method_exists(self):
        from core.api_manager import APICallManager
        assert hasattr(APICallManager, '_process_single_request')

    def test_process_batch_calls_gather(self):
        import inspect
        from core.api_manager import APICallManager
        source = inspect.getsource(APICallManager._process_batch)
        assert 'asyncio.gather' in source
        assert 'return_exceptions' in source


class TestLCMPlannerSynthesis:
    """Test LCM Planner LLM-powered synthesis."""

    def test_synthesis_fallback_single_task(self):
        from core.lcm_planner import LCMPlanner, Plan, PlanTask, TaskStatus
        planner = LCMPlanner()

        plan = Plan(id="test_plan", original_request="test")
        task = PlanTask(id="t1", description="Do thing", status=TaskStatus.COMPLETED, result="Done")
        plan.tasks = [task]

        result = planner._synthesize_results(plan)
        assert result == "Done"

    def test_synthesis_fallback_no_tasks(self):
        from core.lcm_planner import LCMPlanner, Plan
        planner = LCMPlanner()

        plan = Plan(id="test_plan", original_request="test")
        result = planner._synthesize_results(plan)
        assert "No tasks completed" in result

    def test_synthesis_concatenation_multiple_tasks(self):
        from core.lcm_planner import LCMPlanner, Plan, PlanTask, TaskStatus
        planner = LCMPlanner()

        plan = Plan(id="test_plan", original_request="test")
        plan.tasks = [
            PlanTask(id="t1", description="Task A", status=TaskStatus.COMPLETED, result="Result A"),
            PlanTask(id="t2", description="Task B", status=TaskStatus.COMPLETED, result="Result B"),
        ]

        result = planner._fallback_synthesize(plan.tasks)
        assert "Task A" in result
        assert "Task B" in result
        assert "Result A" in result
        assert "Result B" in result

    def test_llm_synthesize_method_exists(self):
        from core.lcm_planner import LCMPlanner
        assert hasattr(LCMPlanner, '_llm_synthesize')
        assert hasattr(LCMPlanner, '_fallback_synthesize')




if __name__ == "__main__":
    pytest.main([__file__, "-v"])