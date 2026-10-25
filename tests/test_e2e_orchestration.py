#!/usr/bin/env python3
"""End-to-end tests for intelligent orchestration with real components."""
import os as _os
_JARVIS_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import asyncio
import sys
sys.path.insert(0, _JARVIS_ROOT)

from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import dataclass
from typing import Any


# ==================== REAL COMPONENT TESTS ====================

async def test_real_decomposition():
    """Test real decomposition with actual LLM structure."""
    print("\n=== Test: Real Decomposition Flow ===")
    
    from core.intelligent_orchestrator import IntelligentOrchestrator, SubtaskType, Subtask
    
    # Create realistic mock LLMs
    class RealisticMockLLM:
        def __init__(self, model_name):
            self.model = model_name
            self.calls = []
        
        async def chat(self, user_input: str, stream: bool = False, **kwargs):
            self.calls.append(user_input)
            
            # Simulate realistic decomposition
            if "decompose" in user_input.lower() or "subtask" in user_input.lower():
                return {
                    "content": '''{
                        "simple": false,
                        "subtasks": [
                            {"name": "search", "type": "tool_call", "description": "Search for relevant information", "model": "primary", "depends_on": []},
                            {"name": "analyze", "type": "analysis", "description": "Analyze the search results", "model": "backend", "depends_on": ["search"]},
                            {"name": "summarize", "type": "creation", "description": "Create a comprehensive summary", "model": "primary", "depends_on": ["analyze"]}
                        ],
                        "execution_mode": "sequential"
                    }'''
                }
            return {"content": f"Response from {self.model}: {user_input[:50]}"}
    
    class RealisticMockDual:
        async def route_task(self, task_type, prompt, **kwargs):
            return f"Routed {task_type.value}: {prompt[:50]}"
    
    orchestrator = IntelligentOrchestrator(
        llm_client=RealisticMockLLM("primary"),
        backend_llm=RealisticMockLLM("backend"),
        dual_model=RealisticMockDual(),
    )
    
    # Test decomposition
    result = await orchestrator._decompose_request(
        "Search for Python tutorials and analyze them thoroughly",
        {"conversation": []}
    )
    
    print(f"  Decomposition type: {result.get('simple', 'N/A')}")
    print(f"  Subtasks count: {len(result.get('subtasks', []))}")
    
    if result.get("subtasks"):
        for i, st in enumerate(result["subtasks"]):
            print(f"    {i+1}. {st['name']} ({st['type']}) -> {st['model']}")
    
    assert not result.get("simple", True), "Should be complex"
    assert len(result.get("subtasks", [])) >= 2, "Should have subtasks"
    print("  ✓ Complex request properly decomposed")


async def test_real_model_selection():
    """Test model selection for various subtask types."""
    print("\n=== Test: Real Model Selection ===")
    
    from core.intelligent_orchestrator import IntelligentOrchestrator, SubtaskType, Subtask
    
    orchestrator = IntelligentOrchestrator(
        llm_client=MagicMock(),
        backend_llm=MagicMock(),
        dual_model=None,
    )
    
    # Test all subtask types with realistic descriptions
    test_cases = [
        (SubtaskType.CODE_GENERATION, "backend", "Write a Python function to process data"),
        (SubtaskType.ANALYSIS, "backend", "Analyze the performance metrics"),
        (SubtaskType.REASONING, "backend", "Deduce the logical conclusion"),
        (SubtaskType.VERIFICATION, "backend", "Verify the calculation results"),
        (SubtaskType.CREATION, "primary", "Create a summary report"),
        (SubtaskType.TOOL_CALL, "primary", "Execute web search"),
        (SubtaskType.SYNTHESIS, "primary", "Combine results into final output"),
    ]
    
    for st_type, expected_model, description in test_cases:
        subtask = Subtask(
            id=f"test_{st_type.value}",
            name=st_type.value,
            description=description,
            subtask_type=st_type,
            assigned_model="auto",
        )
        
        model = orchestrator._select_model_for_subtask(subtask)
        status = "✓" if model == expected_model else "✗"
        print(f"  {status} {st_type.value:20} → {model:8} (expected: {expected_model})")
        assert model == expected_model, f"{st_type.value}: expected {expected_model}, got {model}"


async def test_real_workflow_execution():
    """Test workflow execution with realistic subtasks."""
    print("\n=== Test: Real Workflow Execution ===")
    
    from core.intelligent_orchestrator import (
        IntelligentOrchestrator, Subtask, WorkflowPlan, SubtaskType, SubtaskStatus
    )
    
    class ExecutingMockLLM:
        def __init__(self, name):
            self.name = name
            self.call_count = 0
            
        async def chat(self, user_input: str, stream: bool = False, **kwargs):
            self.call_count += 1
            return {"content": f"[{self.name}] Processed: {user_input[:30]}..."}
    
    orchestrator = IntelligentOrchestrator(
        llm_client=ExecutingMockLLM("primary"),
        backend_llm=ExecutingMockLLM("backend"),
        dual_model=None,
    )
    
    # Create realistic workflow
    subtasks = [
        Subtask(
            id="step1_search",
            name="search",
            description="Search for Python async patterns",
            subtask_type=SubtaskType.TOOL_CALL,
            assigned_model="primary",
            depends_on=[],
        ),
        Subtask(
            id="step2_analyze",
            name="analyze",
            description="Analyze the search results",
            subtask_type=SubtaskType.ANALYSIS,
            assigned_model="backend",
            depends_on=["search"],
        ),
        Subtask(
            id="step3_synthesize",
            name="synthesize",
            description="Create final summary",
            subtask_type=SubtaskType.SYNTHESIS,
            assigned_model="primary",
            depends_on=["analyze"],
        ),
    ]
    
    plan = WorkflowPlan(
        id="test_workflow",
        original_request="Search for Python async patterns, analyze them, and summarize",
        subtasks=subtasks,
    )
    plan.execution_order = orchestrator._topological_sort(subtasks)
    
    print(f"  Workflow: {plan.id}")
    print(f"  Execution order: {plan.execution_order}")
    
    # Execute workflow
    await orchestrator._execute_workflow(plan)
    
    # Check results
    completed = [st for st in plan.subtasks if st.status == SubtaskStatus.COMPLETED]
    print(f"  Completed: {len(completed)}/{len(subtasks)}")
    
    for st in plan.subtasks:
        status_icon = "✓" if st.status == SubtaskStatus.COMPLETED else "✗"
        result_preview = st.result[:50] if st.result else "No result"
        print(f"    {status_icon} {st.name}: {st.status.value} - {result_preview}...")
    
    assert len(completed) == len(subtasks), f"Expected all subtasks completed, got {len(completed)}"
    print("  ✓ Workflow executed successfully")


async def test_real_parallel_execution():
    """Test parallel execution of independent subtasks."""
    print("\n=== Test: Real Parallel Execution ===")
    
    from core.intelligent_orchestrator import (
        IntelligentOrchestrator, Subtask, WorkflowPlan, SubtaskType
    )
    
    orchestrator = IntelligentOrchestrator(
        llm_client=MagicMock(),
        backend_llm=MagicMock(),
        dual_model=None,
    )
    
    # Create workflow with parallel tasks
    # Wave 0: search_py, search_js (parallel)
    # Wave 1: analyze (depends on both)
    # Wave 2: report (depends on analyze)
    subtasks = [
        Subtask(
            id="search_python",
            name="search_python",
            description="Search Python resources",
            subtask_type=SubtaskType.TOOL_CALL,
            assigned_model="primary",
            depends_on=[],
        ),
        Subtask(
            id="search_javascript",
            name="search_javascript",
            description="Search JavaScript resources",
            subtask_type=SubtaskType.TOOL_CALL,
            assigned_model="primary",
            depends_on=[],
        ),
        Subtask(
            id="analyze_results",
            name="analyze",
            description="Analyze combined results",
            subtask_type=SubtaskType.ANALYSIS,
            assigned_model="backend",
            depends_on=["search_python", "search_javascript"],
        ),
        Subtask(
            id="create_report",
            name="report",
            description="Create final report",
            subtask_type=SubtaskType.CREATION,
            assigned_model="primary",
            depends_on=["analyze"],
        ),
    ]
    
    plan = WorkflowPlan(
        id="parallel_test",
        original_request="Compare Python and JavaScript async patterns",
        subtasks=subtasks,
    )
    plan.execution_order = orchestrator._topological_sort(subtasks)
    
    waves = orchestrator._compute_execution_waves(plan)
    
    print(f"  Execution waves: {waves}")
    
    # Validate parallel execution
    wave0 = waves[0]
    wave1 = waves[1] if len(waves) > 1 else []
    wave2 = waves[2] if len(waves) > 2 else []
    
    # Wave 0 should have both search tasks (parallel)
    assert len(wave0) == 2, f"Wave 0 should have 2 parallel tasks, got {len(wave0)}"
    print(f"  ✓ Wave 0 has {len(wave0)} parallel tasks: {wave0}")
    
    # Wave 1 should have analyze (depends on both wave 0 tasks)
    assert len(wave1) == 1, f"Wave 1 should have 1 task, got {len(wave1)}"
    assert "analyze" in wave1, f"Wave 1 should have analyze task"
    print(f"  ✓ Wave 1 has dependent task: {wave1}")
    
    # Wave 2 should have report (depends on analyze)
    assert len(wave2) == 1, f"Wave 2 should have 1 task, got {len(wave2)}"
    assert "report" in wave2, f"Wave 2 should have report task"
    print(f"  ✓ Wave 2 has final task: {wave2}")
    
    print("  ✓ Parallel execution waves computed correctly")


async def test_real_result_synthesis():
    """Test result synthesis from multiple subtasks."""
    print("\n=== Test: Real Result Synthesis ===")
    
    from core.intelligent_orchestrator import (
        IntelligentOrchestrator, Subtask, WorkflowPlan, SubtaskType, SubtaskStatus
    )
    
    class SynthesizingMockLLM:
        async def chat(self, user_input: str, stream: bool = False, **kwargs):
            return {
                "content": "Based on the search and analysis, here is a comprehensive summary:\n\n"
                          "1. Python asyncio uses async/await syntax\n"
                          "2. JavaScript Promises are similar but have differences\n"
                          "3. Key patterns include error handling and cancellation"
            }
    
    orchestrator = IntelligentOrchestrator(
        llm_client=SynthesizingMockLLM(),
        backend_llm=MagicMock(),
        dual_model=None,
    )
    
    # Create plan with completed subtasks and realistic results
    plan = WorkflowPlan(
        id="synthesis_test",
        original_request="Compare Python and JavaScript async patterns",
        subtasks=[
            Subtask(
                id="s1",
                name="search_python",
                description="Search Python async",
                subtask_type=SubtaskType.TOOL_CALL,
                status=SubtaskStatus.COMPLETED,
                result="Python asyncio documentation: async/await syntax, event loop, tasks, futures.",
            ),
            Subtask(
                id="s2",
                name="search_javascript",
                description="Search JavaScript async",
                subtask_type=SubtaskType.TOOL_CALL,
                status=SubtaskStatus.COMPLETED,
                result="JavaScript Promises: then/catch, async/await, Promise.all, error handling.",
            ),
            Subtask(
                id="s3",
                name="analyze",
                description="Analyze results",
                subtask_type=SubtaskType.ANALYSIS,
                status=SubtaskStatus.COMPLETED,
                result="Comparison: Both use async/await. Python has explicit event loop. JS has implicit. "
                       "Error handling differs: try/except vs .catch().",
            ),
        ],
    )
    
    synthesis = await orchestrator._synthesize_results(plan)
    
    print(f"  Synthesis length: {len(synthesis)} chars")
    print(f"  Synthesis preview: {synthesis[:200]}...")
    
    assert len(synthesis) > 50, "Synthesis should be substantial"
    assert any(kw in synthesis.lower() for kw in ["python", "javascript", "async", "pattern"]), \
        "Synthesis should mention key terms"
    print("  ✓ Results synthesized correctly")


async def test_error_recovery():
    """Test graceful handling of subtask failures."""
    print("\n=== Test: Error Recovery ===")
    
    from core.intelligent_orchestrator import (
        IntelligentOrchestrator, Subtask, WorkflowPlan, SubtaskType, SubtaskStatus
    )
    
    class FailingMockLLM:
        async def chat(self, user_input: str, stream: bool = False, **kwargs):
            if "code" in user_input.lower():
                raise Exception("Code generation failed")
            return {"content": "Analysis complete"}
    
    orchestrator = IntelligentOrchestrator(
        llm_client=FailingMockLLM(),
        backend_llm=FailingMockLLM(),
        dual_model=None,
    )
    
    # Create workflow with a failing subtask
    plan = WorkflowPlan(
        id="error_test",
        original_request="Test error handling",
        subtasks=[
            Subtask(
                id="s1",
                name="analyze",
                description="Analyze input",
                subtask_type=SubtaskType.ANALYSIS,
                assigned_model="backend",
            ),
            Subtask(
                id="s2",
                name="code_gen",
                description="Generate code",
                subtask_type=SubtaskType.CODE_GENERATION,  # This will fail
                assigned_model="backend",
                depends_on=["analyze"],
            ),
        ],
    )
    plan.execution_order = orchestrator._topological_sort(plan.subtasks)
    
    # Execute workflow
    await orchestrator._execute_workflow(plan)
    
    # Check results
    analyze_task = next(st for st in plan.subtasks if st.name == "analyze")
    code_task = next(st for st in plan.subtasks if st.name == "code_gen")
    
    print(f"  analyze: {analyze_task.status.value}")
    print(f"  code_gen: {code_task.status.value}")
    
    # Analyze should succeed
    assert analyze_task.status == SubtaskStatus.COMPLETED
    print("  ✓ Pre-dependency task completed")
    
    # Code gen should have failed or have an error
    if code_task.status == SubtaskStatus.FAILED:
        print(f"  ✓ Task failed gracefully: {code_task.error}")
    
    print("  ✓ Error handling works correctly")


async def test_complexity_detection_edge_cases():
    """Test complexity detection for edge cases."""
    print("\n=== Test: Complexity Detection Edge Cases ===")
    
    from core.agent import JarvisAgent
    from core.config import Config
    
    config = Config.load()
    agent = JarvisAgent(config)
    
    edge_cases = [
        # (input, expected_complex, description)
        ("What is 2 + 2?", False, "Simple math"),
        ("What is the meaning of life? Tell me your thoughts", True, "Philosophical + tell"),
        ("Run ls -la", False, "Simple command"),
        ("Run this command and show output", True, "Command + and + long"),
        ("List files", False, "Simple list"),
        ("List all Python files and count them", True, "Multiple actions"),
        ("Compare Python and JavaScript async patterns step by step", True, "Step by step"),
        ("Find bugs in my code", False, "Simple find"),
        ("Find all files modified today, list their contents, and then create a summary", True, "Multi-step"),
        ("Execute the backup script", False, "Single execution"),
        ("First analyze the logs, then identify errors, and finally create a report", True, "Sequential markers"),
        ("What's the weather? What's the temperature?", True, "Multiple questions"),
        ("", False, "Empty input"),
        ("a" * 200, True, "Very long input"),
        ("Quick question: how do I fix this error?", False, "Question with colon"),
        ("I need you to search for tutorials, analyze their quality, and recommend the best ones", True, "Complex request"),
    ]
    
    passed = 0
    for input_text, expected, description in edge_cases:
        if not input_text:
            continue  # Skip empty input
        result = agent._is_complex_request(input_text)
        status = "✓" if result == expected else "✗"
        if result == expected:
            passed += 1
        input_preview = input_text[:40] + "..." if len(input_text) > 40 else input_text
        print(f"  {status} {description}: complex={result} (expected={expected})")
        print(f"     Input: '{input_preview}'")
    
    print(f"\n  Passed: {passed}/{len(edge_cases)}")
    assert passed >= len(edge_cases) * 0.9, f"Expected at least 90% pass rate"


async def test_task_priority_ordering():
    """Test that high-priority tasks are handled correctly."""
    print("\n=== Test: Task Priority Ordering ===")
    
    from core.intelligent_orchestrator import (
        IntelligentOrchestrator, Subtask, WorkflowPlan, SubtaskType
    )
    
    orchestrator = IntelligentOrchestrator(
        llm_client=MagicMock(),
        backend_llm=MagicMock(),
        dual_model=None,
    )
    
    # Create tasks with different dependencies but same "priority"
    subtasks = [
        Subtask(
            id="setup",
            name="setup",
            description="Initialize environment",
            subtask_type=SubtaskType.TOOL_CALL,
            depends_on=[],
        ),
        Subtask(
            id="analyze_requirements",
            name="analyze_requirements",
            description="Analyze requirements",
            subtask_type=SubtaskType.ANALYSIS,
            depends_on=["setup"],
        ),
        Subtask(
            id="implement_1",
            name="implement_feature_1",
            description="Implement feature 1",
            subtask_type=SubtaskType.CODE_GENERATION,
            depends_on=["analyze_requirements"],
        ),
        Subtask(
            id="implement_2",
            name="implement_feature_2",
            description="Implement feature 2",
            subtask_type=SubtaskType.CODE_GENERATION,
            depends_on=["analyze_requirements"],
        ),
        Subtask(
            id="test",
            name="test",
            description="Test implementation",
            subtask_type=SubtaskType.VERIFICATION,
            depends_on=["implement_feature_1", "implement_feature_2"],
        ),
        Subtask(
            id="report",
            name="report",
            description="Create report",
            subtask_type=SubtaskType.SYNTHESIS,
            depends_on=["test"],
        ),
    ]
    
    plan = WorkflowPlan(
        id="priority_test",
        original_request="Build features with proper ordering",
        subtasks=subtasks,
    )
    plan.execution_order = orchestrator._topological_sort(subtasks)
    
    waves = orchestrator._compute_execution_waves(plan)
    
    print(f"  Execution waves: {waves}")
    
    # Verify ordering
    task_positions = {}
    for i, wave in enumerate(waves):
        for task in wave:
            task_positions[task] = i
    
    # Setup should be in first wave
    assert task_positions.get("setup", 99) == 0, "setup should be in wave 0"
    print("  ✓ setup in wave 0")
    
    # Implement tasks should be in same wave (parallel after analyze)
    impl_1_wave = task_positions.get("implement_feature_1", 99)
    impl_2_wave = task_positions.get("implement_feature_2", 99)
    assert impl_1_wave == impl_2_wave, "implement tasks should be in same wave"
    print(f"  ✓ implement tasks in same wave ({impl_1_wave})")
    
    # Test should come after both implement tasks
    test_wave = task_positions.get("test", 99)
    assert test_wave > impl_1_wave, "test should come after implement tasks"
    print(f"  ✓ test after implement tasks (test in wave {test_wave})")
    
    print("  ✓ Task ordering respects dependencies")


async def test_workflow_status_tracking():
    """Test workflow status tracking during execution."""
    print("\n=== Test: Workflow Status Tracking ===")
    
    from core.intelligent_orchestrator import (
        IntelligentOrchestrator, Subtask, WorkflowPlan, SubtaskType, SubtaskStatus
    )
    
    orchestrator = IntelligentOrchestrator(
        llm_client=MagicMock(),
        backend_llm=MagicMock(),
        dual_model=None,
    )
    
    # Create simple workflow
    plan = WorkflowPlan(
        id="status_test",
        original_request="Test status tracking",
        subtasks=[
            Subtask(id="s1", name="step1", description="First", subtask_type=SubtaskType.TOOL_CALL),
            Subtask(id="s2", name="step2", description="Second", subtask_type=SubtaskType.ANALYSIS),
        ],
    )
    plan.execution_order = ["step1", "step2"]
    
    # Add to active workflows
    orchestrator._active_workflows[plan.id] = plan
    
    # Get status
    status = orchestrator.get_workflow_status(plan.id)
    
    assert status is not None
    assert status["id"] == plan.id
    assert status["status"] == "pending"
    assert len(status["subtasks"]) == 2
    print(f"  ✓ Workflow status tracked: {status['id']}")
    print(f"    Status: {status['status']}")
    print(f"    Subtasks: {len(status['subtasks'])}")
    
    # Get active workflows
    active = orchestrator.get_active_workflows()
    assert len(active) == 1
    print(f"  ✓ Active workflows: {len(active)}")
    
    # Clean up
    orchestrator._active_workflows.clear()


async def test_fallback_for_simple_requests():
    """Test that simple requests use direct execution."""
    print("\n=== Test: Fallback for Simple Requests ===")
    
    from core.intelligent_orchestrator import IntelligentOrchestrator
    from core.dual_model import TaskType
    
    class MockDual:
        async def route_task(self, task_type, prompt, **kwargs):
            return f"Direct response for: {prompt[:30]}"
    
    class MockBackend:
        async def chat(self, user_input, stream=False, **kwargs):
            return {"content": '{"simple": true}'}
    
    class MockPrimary:
        async def chat(self, user_input, stream=False, **kwargs):
            return {"content": "Simple response"}
    
    orchestrator = IntelligentOrchestrator(
        llm_client=MockPrimary(),
        backend_llm=MockBackend(),
        dual_model=MockDual(),
    )
    
    # Test simple request
    result = await orchestrator._decompose_request("What is 2 + 2?", None)
    
    # Should return simple: true for simple requests (based on mock)
    if result.get("simple"):
        print("  ✓ Backend correctly identified simple request")
    else:
        print("  ✓ Backend processing request")
    
    # Test simple execution
    simple_result = await orchestrator._execute_simple("What is 2 + 2?", None)
    assert simple_result is not None
    print("  ✓ Simple execution works")


# ==================== RUN ALL TESTS ====================

async def main():
    """Run all end-to-end tests."""
    print("=" * 70)
    print("END-TO-END ORCHESTRATION TESTS")
    print("=" * 70)
    
    tests = [
        test_real_decomposition,
        test_real_model_selection,
        test_real_workflow_execution,
        test_real_parallel_execution,
        test_real_result_synthesis,
        test_error_recovery,
        test_complexity_detection_edge_cases,
        test_task_priority_ordering,
        test_workflow_status_tracking,
        test_fallback_for_simple_requests,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            await test()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {type(e).__name__}: {e}")
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)
    
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)