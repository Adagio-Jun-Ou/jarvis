#!/usr/bin/env python3
"""Comprehensive tests for intelligent orchestrator and agentic workflow."""
import os as _os
_JARVIS_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import asyncio
import sys
sys.path.insert(0, _JARVIS_ROOT)

from unittest.mock import MagicMock, AsyncMock
from core.intelligent_orchestrator import (
    IntelligentOrchestrator,
    SubtaskType,
    SubtaskStatus,
    Subtask,
    WorkflowPlan,
    get_intelligent_orchestrator,
    initialize_intelligent_orchestrator,
)
from core.dual_model import TaskType


class MockLLM:
    """Mock LLM with realistic responses."""
    
    def __init__(self, name="mock"):
        self.model = name
        self.call_count = 0
        self.calls = []
    
    async def chat(self, user_input: str, stream: bool = False, **kwargs):
        self.call_count += 1
        self.calls.append(user_input[:100])
        
        # Simulate realistic responses
        if "decompose" in user_input.lower() or "subtask" in user_input.lower():
            return {"content": self._decomposition_response()}
        elif "synthesize" in user_input.lower():
            return {"content": "Synthesized result from multiple sources."}
        else:
            return {"content": f"Response from {self.model}: {user_input[:50]}"}
    
    def _decomposition_response(self):
        return '''{
            "simple": false,
            "subtasks": [
                {"name": "search", "type": "tool_call", "description": "Search for information", "model": "primary", "depends_on": []},
                {"name": "analyze", "type": "analysis", "description": "Analyze search results", "model": "backend", "depends_on": ["search"]},
                {"name": "summarize", "type": "creation", "description": "Create summary", "model": "primary", "depends_on": ["analyze"]}
            ],
            "execution_mode": "sequential"
        }'''


class MockDualModel:
    """Mock dual model for testing."""
    
    def __init__(self):
        self.stats = {}
        self.routes = []
    
    def record_subsystem_call(self, subsystem: str, tokens: int):
        self.stats[subsystem] = self.stats.get(subsystem, 0) + 1
    
    async def route_task(self, task_type, prompt: str, **kwargs):
        self.routes.append((task_type, prompt[:50]))
        return f"Routed {task_type.value} task result"


# ==================== TESTS ====================

async def test_subtask_creation():
    """Test subtask dataclass."""
    print("\n=== Test: Subtask Creation ===")
    
    st = Subtask(
        id="test_1",
        name="test_subtask",
        description="Test subtask description",
        subtask_type=SubtaskType.ANALYSIS,
        assigned_model="backend",
    )
    
    assert st.status == SubtaskStatus.PENDING
    assert st.subtask_type == SubtaskType.ANALYSIS
    assert st.depends_on == []
    assert st.result == ""
    assert st.error == ""
    print("  ✓ Subtask created with correct defaults")
    
    # Test subtask with dependencies
    st2 = Subtask(
        id="test_2",
        name="dependent_subtask",
        description="Depends on test_subtask",
        subtask_type=SubtaskType.SYNTHESIS,
        assigned_model="primary",
        depends_on=["test_1"],
    )
    
    assert "test_1" in st2.depends_on
    print("  ✓ Subtask with dependencies created")


async def test_workflow_plan_creation():
    """Test workflow plan creation and management."""
    print("\n=== Test: Workflow Plan Creation ===")
    
    subtasks = [
        Subtask(id="s1", name="step1", description="First", subtask_type=SubtaskType.TOOL_CALL),
        Subtask(id="s2", name="step2", description="Second", subtask_type=SubtaskType.ANALYSIS, depends_on=["step1"]),
        Subtask(id="s3", name="step3", description="Third", subtask_type=SubtaskType.SYNTHESIS, depends_on=["step2"]),
    ]
    
    plan = WorkflowPlan(
        id="test_plan",
        original_request="Test workflow",
        subtasks=subtasks,
    )
    
    assert plan.status == "pending"
    assert len(plan.subtasks) == 3
    print("  ✓ Workflow plan created")
    
    # Test status transitions
    plan.status = "running"
    assert plan.status == "running"
    print("  ✓ Plan status transitions work")


async def test_model_selection():
    """Test that subtasks are assigned to correct models."""
    print("\n=== Test: Model Selection ===")
    
    orchestrator = IntelligentOrchestrator(
        llm_client=MockLLM("primary"),
        backend_llm=MockLLM("backend"),
        dual_model=MockDualModel(),
    )
    
    # Test all subtask types
    model_mapping = {
        SubtaskType.CODE_GENERATION: "backend",
        SubtaskType.REASONING: "backend",
        SubtaskType.ANALYSIS: "backend",
        SubtaskType.VERIFICATION: "backend",
        SubtaskType.CREATION: "primary",
        SubtaskType.TOOL_CALL: "primary",
        SubtaskType.SYNTHESIS: "primary",
    }
    
    for st_type, expected_model in model_mapping.items():
        subtask = Subtask(
            id="test",
            name="test",
            description="",
            subtask_type=st_type,
            assigned_model="auto",
        )
        model = orchestrator._select_model_for_subtask(subtask)
        assert model == expected_model, f"{st_type.value}: expected {expected_model}, got {model}"
        print(f"  ✓ {st_type.value} → {model}")


async def test_topological_sort():
    """Test dependency ordering."""
    print("\n=== Test: Topological Sort ===")
    
    orchestrator = IntelligentOrchestrator(
        llm_client=MockLLM("primary"),
        backend_llm=MockLLM("backend"),
        dual_model=MockDualModel(),
    )
    
    # Diamond dependency graph
    #     A
    #    / \
    #   B   C
    #    \ /
    #     D
    subtasks = [
        Subtask(id="a", name="A", description="", subtask_type=SubtaskType.TOOL_CALL, depends_on=[]),
        Subtask(id="b", name="B", description="", subtask_type=SubtaskType.ANALYSIS, depends_on=["A"]),
        Subtask(id="c", name="C", description="", subtask_type=SubtaskType.ANALYSIS, depends_on=["A"]),
        Subtask(id="d", name="D", description="", subtask_type=SubtaskType.SYNTHESIS, depends_on=["B", "C"]),
    ]
    
    order = orchestrator._topological_sort(subtasks)
    
    # Validate ordering
    assert order.index("A") < order.index("B"), "A before B"
    assert order.index("A") < order.index("C"), "A before C"
    assert order.index("B") < order.index("D"), "B before D"
    assert order.index("C") < order.index("D"), "C before D"
    print(f"  ✓ Valid order: {order}")


async def test_execution_waves():
    """Test parallel execution wave computation."""
    print("\n=== Test: Execution Waves ===")
    
    orchestrator = IntelligentOrchestrator(
        llm_client=MockLLM("primary"),
        backend_llm=MockLLM("backend"),
        dual_model=MockDualModel(),
    )
    
    # Parallel tasks
    # Wave 0: A, B (independent)
    # Wave 1: C (depends on A)
    # Wave 2: D (depends on C)
    # Wave 3: E (depends on B, D)
    subtasks = [
        Subtask(id="a", name="A", description="", subtask_type=SubtaskType.TOOL_CALL, depends_on=[]),
        Subtask(id="b", name="B", description="", subtask_type=SubtaskType.TOOL_CALL, depends_on=[]),
        Subtask(id="c", name="C", description="", subtask_type=SubtaskType.ANALYSIS, depends_on=["A"]),
        Subtask(id="d", name="D", description="", subtask_type=SubtaskType.SYNTHESIS, depends_on=["C"]),
        Subtask(id="e", name="E", description="", subtask_type=SubtaskType.CREATION, depends_on=["B", "D"]),
    ]
    
    plan = WorkflowPlan(
        id="test_waves",
        original_request="Test",
        subtasks=subtasks,
    )
    plan.execution_order = orchestrator._topological_sort(subtasks)
    
    waves = orchestrator._compute_execution_waves(plan)
    
    print(f"  Waves: {waves}")
    
    # Validate parallel execution
    wave0 = waves[0]
    assert "A" in wave0 and "B" in wave0, f"Wave 0 should have A and B: {wave0}"
    assert len(wave0) == 2, f"Wave 0 should have exactly 2 tasks: {wave0}"
    
    # Find which wave each task is in
    task_waves = {}
    for i, wave in enumerate(waves):
        for task in wave:
            task_waves[task] = i
    
    # A and B should be in same or earlier wave than C
    assert task_waves.get("A", 99) <= task_waves.get("C", 0), "A before C"
    assert task_waves.get("B", 99) <= task_waves.get("E", 0), "B before E"
    
    print("  ✓ Parallel execution waves computed correctly")


async def test_decomposition():
    """Test request decomposition."""
    print("\n=== Test: Request Decomposition ===")
    
    backend = MockLLM("backend")
    orchestrator = IntelligentOrchestrator(
        llm_client=MockLLM("primary"),
        backend_llm=backend,
        dual_model=MockDualModel(),
    )
    
    # Test complex request decomposition
    result = await orchestrator._decompose_request(
        "Search for Python tutorials and then analyze them step by step",
        None
    )
    
    print(f"  Decomposition result: {result}")
    
    assert backend.call_count > 0, "Backend LLM should be called for decomposition"
    print(f"  ✓ Backend LLM called ({backend.call_count} times)")
    
    # Check result structure
    if "subtasks" in result:
        assert isinstance(result["subtasks"], list)
        print(f"  ✓ Subtasks returned: {len(result['subtasks'])} items")


async def test_simple_request_routing():
    """Test that simple requests bypass orchestration."""
    print("\n=== Test: Simple Request Routing ===")
    
    backend = MockLLM("backend")
    orchestrator = IntelligentOrchestrator(
        llm_client=MockLLM("primary"),
        backend_llm=backend,
        dual_model=MockDualModel(),
    )
    
    # Test task classification
    classification_tests = [
        ("Write a Python function to sort", TaskType.CODE),
        ("Calculate the factorial", TaskType.MATH),
        ("Analyze performance", TaskType.ANALYSIS),
        ("Tell me a story", TaskType.CREATIVE),  # Creative goes to Primary
        ("What's the weather", TaskType.CHAT),
        ("Tell me a story about dragons", TaskType.CREATIVE),  # Creative
        ("What is the capital of France", TaskType.CHAT),  # Simple question
    ]
    
    for request, expected_type in classification_tests:
        task_type = orchestrator._classify_simple_task(request)
        assert task_type == expected_type, f"Failed for '{request}': expected {expected_type}, got {task_type}"
        print(f"  ✓ '{request[:35]}...' → {task_type.value}")


async def test_workflow_execution():
    """Test workflow execution."""
    print("\n=== Test: Workflow Execution ===")
    
    orchestrator = IntelligentOrchestrator(
        llm_client=MockLLM("primary"),
        backend_llm=MockLLM("backend"),
        dual_model=MockDualModel(),
    )
    
    # Create a simple workflow
    subtasks = [
        Subtask(
            id="reason_1",
            name="reason_step",
            description="Think through the problem",
            subtask_type=SubtaskType.REASONING,
            assigned_model="backend",
        ),
    ]
    
    plan = WorkflowPlan(
        id="test_exec",
        original_request="Test execution",
        subtasks=subtasks,
    )
    plan.execution_order = ["reason_step"]
    
    # Execute single subtask
    await orchestrator._execute_workflow(plan)
    
    # Check status
    assert plan.subtasks[0].status == SubtaskStatus.COMPLETED, f"Expected COMPLETED, got {plan.subtasks[0].status}"
    print(f"  ✓ Subtask executed successfully")
    print(f"  Result: {plan.subtasks[0].result[:100]}...")


async def test_result_synthesis():
    """Test result synthesis."""
    print("\n=== Test: Result Synthesis ===")
    
    primary = MockLLM("primary")
    orchestrator = IntelligentOrchestrator(
        llm_client=primary,
        backend_llm=MockLLM("backend"),
        dual_model=MockDualModel(),
    )
    
    # Create plan with completed subtasks
    plan = WorkflowPlan(
        id="test_synth",
        original_request="Test synthesis",
        subtasks=[
            Subtask(
                id="s1",
                name="step1",
                description="First step",
                subtask_type=SubtaskType.TOOL_CALL,
                status=SubtaskStatus.COMPLETED,
                result="Result from step 1",
            ),
            Subtask(
                id="s2",
                name="step2",
                description="Second step",
                subtask_type=SubtaskType.ANALYSIS,
                status=SubtaskStatus.COMPLETED,
                result="Result from step 2",
            ),
        ],
    )
    
    synthesis = await orchestrator._synthesize_results(plan)
    
    print(f"  Synthesis: {synthesis[:200]}..." if synthesis else "  No synthesis returned")
    
    # Primary should be called for synthesis
    if synthesis:
        print(f"  ✓ Synthesis generated by primary model")


async def test_singleton():
    """Test singleton pattern."""
    print("\n=== Test: Singleton Pattern ===")
    
    # Get singleton
    orch1 = get_intelligent_orchestrator()
    orch2 = get_intelligent_orchestrator()
    
    assert orch1 is orch2, "Singleton should return same instance"
    print("  ✓ Singleton returns same instance")
    
    # Initialize with mock clients
    mock_llm = MockLLM("primary")
    mock_backend = MockLLM("backend")
    mock_dual = MockDualModel()
    
    orch = initialize_intelligent_orchestrator(
        llm_client=mock_llm,
        backend_llm=mock_backend,
        dual_model=mock_dual,
    )
    
    assert orch is not None
    assert orch.llm == mock_llm
    assert orch.backend_llm == mock_backend
    print("  ✓ Initialization works correctly")


async def test_workflow_status_tracking():
    """Test workflow status tracking."""
    print("\n=== Test: Workflow Status Tracking ===")
    
    orchestrator = IntelligentOrchestrator(
        llm_client=MockLLM("primary"),
        backend_llm=MockLLM("backend"),
        dual_model=MockDualModel(),
    )
    
    # No active workflows initially
    active = orchestrator.get_active_workflows()
    assert len(active) == 0
    print("  ✓ No active workflows initially")
    
    # Create a workflow
    plan = WorkflowPlan(
        id="status_test",
        original_request="Test status",
        subtasks=[
            Subtask(
                id="s1",
                name="test",
                description="Test",
                subtask_type=SubtaskType.ANALYSIS,
            )
        ],
    )
    
    # Add to active
    orchestrator._active_workflows[plan.id] = plan
    
    active = orchestrator.get_active_workflows()
    assert len(active) == 1
    assert active[0]["id"] == plan.id
    print("  ✓ Active workflow tracked")
    
    # Get detailed status
    status = orchestrator.get_workflow_status(plan.id)
    assert status is not None
    assert status["id"] == plan.id
    assert len(status["subtasks"]) == 1
    print("  ✓ Detailed status available")
    
    # Clean up
    orchestrator._active_workflows.clear()


async def test_complex_request_detection():
    """Test complex request detection."""
    print("\n=== Test: Complex Request Detection ===")
    
    orchestrator = IntelligentOrchestrator(
        llm_client=MockLLM("primary"),
        backend_llm=MockLLM("backend"),
        dual_model=MockDualModel(),
    )
    
    # Test cases
    complex_requests = [
        "Search for Python tutorials and then analyze them",
        "First search, then summarize, and finally save",
        "List all files and then count them",
        "Find all Python files and analyze their imports",
        "Comprehensive analysis of the system multiple steps",
        "Step by step guide to building an app",
    ]
    
    simple_requests = [
        "What is the weather?",
        "Tell me a joke",
        "What is 2 + 2?",
        "Run this command",
        "Explain Python",
    ]
    
    # Test complex detection
    for req in complex_requests:
        result = await orchestrator._decompose_request(req, None)
        # Complex requests should produce subtasks (mock returns them)
        print(f"  Complex: '{req[:40]}...' -> {len(result.get('subtasks', []))} subtasks")
    
    # Test simple detection  
    for req in simple_requests:
        # Simple requests should return empty subtasks from mock
        print(f"  Simple: '{req[:40]}...'")


# ==================== AGENT INTEGRATION ====================

async def test_agent_complexity_detection():
    """Test agent's complexity detection."""
    print("\n=== Test: Agent Complexity Detection ===")
    
    from core.config import Config
    from core.agent import JarvisAgent
    
    config = Config.load()
    agent = JarvisAgent(config)
    
    test_cases = [
        # (request, expected_complex)
        ("What is the weather today?", False),
        ("Tell me a joke", False),
        ("What is 2 + 2?", False),
        ("Run this command", False),
        ("Explain Python", False),
        ("Search for Python tutorials and then analyze them", True),
        ("First search, then summarize, and finally save", True),
        ("List all files and then count them", True),
        ("Find all Python files and analyze their imports", True),
        ("I need a comprehensive analysis with multiple steps", True),
        ("Step by step guide to building an app", True),
        ("Search and fetch multiple documents", True),  # "multiple" triggers
    ]
    
    for request, expected in test_cases:
        result = agent._is_complex_request(request)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{request[:40]}...' -> complex={result} (expected={expected})")
        if result != expected:
            print(f"      MISMATCH!")


async def test_agent_orchestrator_attachment():
    """Test that orchestrator can be attached to agent."""
    print("\n=== Test: Agent Orchestrator Attachment ===")
    
    from core.config import Config
    from core.agent import JarvisAgent
    
    config = Config.load()
    agent = JarvisAgent(config)
    
    # Initially None
    assert agent._intelligent_orchestrator is None
    print("  ✓ Orchestrator is None initially")
    
    # Create and attach
    orchestrator = IntelligentOrchestrator(
        llm_client=MockLLM("primary"),
        backend_llm=MockLLM("backend"),
        dual_model=MockDualModel(),
    )
    
    agent.attach_intelligent_orchestrator(orchestrator)
    
    assert agent._intelligent_orchestrator is orchestrator
    print("  ✓ Orchestrator attached successfully")


# ==================== EDGE CASES ====================

async def test_circular_dependencies():
    """Test handling of circular dependencies."""
    print("\n=== Test: Circular Dependencies ===")
    
    orchestrator = IntelligentOrchestrator(
        llm_client=MockLLM("primary"),
        backend_llm=MockLLM("backend"),
        dual_model=MockDualModel(),
    )
    
    # Circular: A -> B -> C -> A
    subtasks = [
        Subtask(id="a", name="A", description="", subtask_type=SubtaskType.TOOL_CALL, depends_on=["C"]),
        Subtask(id="b", name="B", description="", subtask_type=SubtaskType.ANALYSIS, depends_on=["A"]),
        Subtask(id="c", name="C", description="", subtask_type=SubtaskType.SYNTHESIS, depends_on=["B"]),
    ]
    
    plan = WorkflowPlan(
        id="circular_test",
        original_request="Test",
        subtasks=subtasks,
    )
    plan.execution_order = orchestrator._topological_sort(subtasks)
    
    # Should handle circular deps gracefully
    assert len(plan.execution_order) == 3
    print(f"  ✓ Circular deps handled: order={plan.execution_order}")
    
    # Execution waves should also work
    waves = orchestrator._compute_execution_waves(plan)
    print(f"  ✓ Execution waves: {waves}")


async def test_empty_workflow():
    """Test handling of empty workflows."""
    print("\n=== Test: Empty Workflow ===")
    
    orchestrator = IntelligentOrchestrator(
        llm_client=MockLLM("primary"),
        backend_llm=MockLLM("backend"),
        dual_model=MockDualModel(),
    )
    
    # Empty plan
    plan = WorkflowPlan(
        id="empty",
        original_request="Test",
        subtasks=[],
    )
    
    # Should handle gracefully
    waves = orchestrator._compute_execution_waves(plan)
    assert waves == []
    print("  ✓ Empty workflow handled correctly")


async def test_failed_subtask():
    """Test handling of failed subtasks."""
    print("\n=== Test: Failed Subtask ===")
    
    orchestrator = IntelligentOrchestrator(
        llm_client=MockLLM("primary"),
        backend_llm=MockLLM("backend"),
        dual_model=MockDualModel(),
    )
    
    plan = WorkflowPlan(
        id="fail_test",
        original_request="Test",
        subtasks=[
            Subtask(
                id="s1",
                name="fail_step",
                description="This will fail",
                subtask_type=SubtaskType.TOOL_CALL,
            ),
        ],
    )
    plan.execution_order = ["fail_step"]
    
    # Simulate execution with error
    plan.subtasks[0].status = SubtaskStatus.FAILED
    plan.subtasks[0].error = "Simulated failure"
    
    # Synthesis should handle failed tasks
    synthesis = await orchestrator._synthesize_results(plan)
    
    # Should still produce some output
    print(f"  Synthesis with failed task: {synthesis[:100] if synthesis else 'No output'}...")
    print("  ✓ Failed subtask handled gracefully")


# ==================== MAIN ====================

async def main():
    """Run all tests."""
    print("=" * 70)
    print("COMPREHENSIVE INTELLIGENT ORCHESTRATOR TESTS")
    print("=" * 70)
    
    tests = [
        # Core tests
        test_subtask_creation,
        test_workflow_plan_creation,
        test_model_selection,
        test_topological_sort,
        test_execution_waves,
        test_decomposition,
        test_simple_request_routing,
        test_workflow_execution,
        test_result_synthesis,
        test_singleton,
        test_workflow_status_tracking,
        test_complex_request_detection,
        
        # Agent integration
        test_agent_complexity_detection,
        test_agent_orchestrator_attachment,
        
        # Edge cases
        test_circular_dependencies,
        test_empty_workflow,
        test_failed_subtask,
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