#!/usr/bin/env python3
"""Test intelligent orchestrator and agentic workflow."""
import os as _os
_JARVIS_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import asyncio
import sys
sys.path.insert(0, _JARVIS_ROOT)

from core.intelligent_orchestrator import (
    IntelligentOrchestrator,
    Subtask,
    SubtaskType,
    SubtaskStatus,
    WorkflowPlan,
)


class MockLLM:
    """Mock LLM for testing."""
    
    def __init__(self, name="mock"):
        self.model = name
        self.call_count = 0
    
    async def chat(self, user_input: str, stream: bool = False, **kwargs):
        self.call_count += 1
        
        # Simulate decomposition for complex requests
        if "and then" in user_input.lower() or "step by step" in user_input.lower():
            return {
                "content": '''{
                    "simple": false,
                    "subtasks": [
                        {"name": "step1_search", "type": "tool_call", "description": "Search for information", "model": "primary", "depends_on": []},
                        {"name": "step2_analyze", "type": "analysis", "description": "Analyze the results", "model": "backend", "depends_on": ["step1_search"]},
                        {"name": "step3_summarize", "type": "creation", "description": "Create summary", "model": "primary", "depends_on": ["step2_analyze"]}
                    ],
                    "execution_mode": "sequential"
                }'''
            }
        
        # Simple request
        if "simple" in user_input.lower():
            return {"content": '{"simple": true}'}
        
        # Default response
        return {"content": f"Response from {self.model}: Processed '{user_input[:50]}...'"}


class MockDualModel:
    """Mock dual model for testing."""
    
    def __init__(self):
        self.stats = {}
    
    def record_subsystem_call(self, subsystem: str, tokens: int):
        self.stats[subsystem] = self.stats.get(subsystem, 0) + 1
    
    async def route_task(self, task_type, prompt: str, **kwargs):
        return f"Routed task result for: {prompt[:50]}..."


async def test_workflow_plan_creation():
    """Test that workflow plans are created correctly."""
    print("\n=== Test: Workflow Plan Creation ===")
    
    orchestrator = IntelligentOrchestrator(
        llm_client=MockLLM("primary"),
        backend_llm=MockLLM("backend"),
        dual_model=MockDualModel(),
    )
    
    # Test subtask creation
    subtask = Subtask(
        id="test_st1",
        name="test_subtask",
        description="A test subtask",
        subtask_type=SubtaskType.ANALYSIS,
        assigned_model="backend",
    )
    
    plan = WorkflowPlan(
        id="test_wf",
        original_request="Test request",
        subtasks=[subtask],
    )
    
    assert plan.status == "pending"
    assert len(plan.subtasks) == 1
    assert plan.subtasks[0].name == "test_subtask"
    print("✓ Workflow plan structure works correctly")


async def test_topological_sort():
    """Test that dependency ordering works."""
    print("\n=== Test: Topological Sort ===")
    
    orchestrator = IntelligentOrchestrator(
        llm_client=MockLLM("primary"),
        backend_llm=MockLLM("backend"),
        dual_model=MockDualModel(),
    )
    
    # Create subtasks with dependencies
    subtasks = [
        Subtask(id="st1", name="first", description="", subtask_type=SubtaskType.TOOL_CALL, depends_on=[]),
        Subtask(id="st2", name="second", description="", subtask_type=SubtaskType.ANALYSIS, depends_on=["first"]),
        Subtask(id="st3", name="third", description="", subtask_type=SubtaskType.CREATION, depends_on=["second"]),
        Subtask(id="st4", name="parallel", description="", subtask_type=SubtaskType.VERIFICATION, depends_on=[]),
    ]
    
    order = orchestrator._topological_sort(subtasks)
    
    # first and parallel should be before second
    assert order.index("first") < order.index("second"), f"Order: {order}"
    assert order.index("parallel") < order.index("second"), f"Order: {order}"
    # second should be before third
    assert order.index("second") < order.index("third"), f"Order: {order}"
    
    print(f"✓ Topological sort correct: {order}")


async def test_execution_waves():
    """Test that parallel execution waves are computed correctly."""
    print("\n=== Test: Execution Waves ===")
    
    orchestrator = IntelligentOrchestrator(
        llm_client=MockLLM("primary"),
        backend_llm=MockLLM("backend"),
        dual_model=MockDualModel(),
    )
    
    # Create a plan with dependencies
    plan = WorkflowPlan(
        id="wf_test",
        original_request="Test",
        subtasks=[
            Subtask(id="st1", name="a", description="", subtask_type=SubtaskType.TOOL_CALL, depends_on=[]),
            Subtask(id="st2", name="b", description="", subtask_type=SubtaskType.TOOL_CALL, depends_on=[]),
            Subtask(id="st3", name="c", description="", subtask_type=SubtaskType.ANALYSIS, depends_on=["a", "b"]),
            Subtask(id="st4", name="d", description="", subtask_type=SubtaskType.CREATION, depends_on=["c"]),
        ],
    )
    plan.execution_order = orchestrator._topological_sort(plan.subtasks)
    
    waves = orchestrator._compute_execution_waves(plan)
    
    # Wave 0: a, b (no dependencies)
    # Wave 1: c (depends on a, b)
    # Wave 2: d (depends on c)
    
    assert len(waves) >= 2, f"Waves: {waves}"
    print(f"✓ Execution waves computed: {waves}")
    
    # First wave should have tasks with no dependencies
    first_wave = waves[0]
    assert "a" in first_wave or "b" in first_wave


async def test_model_selection():
    """Test that subtasks are assigned to correct models."""
    print("\n=== Test: Model Selection ===")
    
    orchestrator = IntelligentOrchestrator(
        llm_client=MockLLM("primary"),
        backend_llm=MockLLM("backend"),
        dual_model=MockDualModel(),
    )
    
    # Test model selection for different subtask types
    tests = [
        (SubtaskType.CODE_GENERATION, "backend"),
        (SubtaskType.ANALYSIS, "backend"),
        (SubtaskType.REASONING, "backend"),
        (SubtaskType.VERIFICATION, "backend"),
        (SubtaskType.CREATION, "primary"),
        (SubtaskType.TOOL_CALL, "primary"),
        (SubtaskType.SYNTHESIS, "primary"),
    ]
    
    for st_type, expected_model in tests:
        subtask = Subtask(
            id="test",
            name="test",
            description="",
            subtask_type=st_type,
            assigned_model="auto",
        )
        model = orchestrator._select_model_for_subtask(subtask)
        assert model == expected_model, f"{st_type}: expected {expected_model}, got {model}"
        print(f"  ✓ {st_type.value} → {model}")


async def test_simple_request_detection():
    """Test decomposition of simple vs complex requests."""
    print("\n=== Test: Simple Request Detection ===")
    
    orchestrator = IntelligentOrchestrator(
        llm_client=MockLLM("primary"),
        backend_llm=MockLLM("backend"),
        dual_model=MockDualModel(),
    )
    
    # Simple requests should return empty subtasks
    result = await orchestrator._decompose_request("What is 2 + 2?", None)
    
    # The mock LLM returns simple: true for simple requests
    print(f"  Simple request decomposition: {result}")


async def test_complex_request_decomposition():
    """Test that complex requests are decomposed."""
    print("\n=== Test: Complex Request Decomposition ===")
    
    backend = MockLLM("backend")
    orchestrator = IntelligentOrchestrator(
        llm_client=MockLLM("primary"),
        backend_llm=backend,
        dual_model=MockDualModel(),
    )
    
    # Complex request should trigger decomposition
    result = await orchestrator._decompose_request(
        "Search for Python tutorials and then analyze them step by step",
        None
    )
    
    # Backend mock should have been called
    assert backend.call_count > 0, "Backend LLM should be called for decomposition"
    print(f"  ✓ Backend LLM called for decomposition (call_count: {backend.call_count})")
    print(f"  Decomposition result: {result}")


async def test_task_classification():
    """Test simple task classification."""
    print("\n=== Test: Task Classification ===")
    
    orchestrator = IntelligentOrchestrator(
        llm_client=MockLLM("primary"),
        backend_llm=MockLLM("backend"),
        dual_model=MockDualModel(),
    )
    
    from core.dual_model import TaskType
    
    tests = [
        ("Write a Python function to sort a list", TaskType.CODE),
        ("Calculate the square root of 144", TaskType.MATH),
        ("Analyze the performance of this algorithm", TaskType.ANALYSIS),
        ("What is the weather today?", TaskType.CHAT),
        ("Tell me a creative story", TaskType.CHAT),  # creative defaults to chat
    ]
    
    for request, expected_type in tests:
        task_type = orchestrator._classify_simple_task(request)
        assert task_type == expected_type, f"{request}: expected {expected_type}, got {task_type}"
        print(f"  ✓ '{request[:40]}...' → {task_type.value}")


async def test_workflow_status():
    """Test workflow status tracking."""
    print("\n=== Test: Workflow Status ===")
    
    orchestrator = IntelligentOrchestrator(
        llm_client=MockLLM("primary"),
        backend_llm=MockLLM("backend"),
        dual_model=MockDualModel(),
    )
    
    # No active workflows initially
    active = orchestrator.get_active_workflows()
    assert len(active) == 0, f"Should have no active workflows, got: {active}"
    print("  ✓ No active workflows initially")


async def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Intelligent Orchestrator")
    print("=" * 60)
    
    tests = [
        test_workflow_plan_creation,
        test_topological_sort,
        test_execution_waves,
        test_model_selection,
        test_simple_request_detection,
        test_complex_request_decomposition,
        test_task_classification,
        test_workflow_status,
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
            print(f"  ✗ ERROR: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)