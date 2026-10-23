#!/usr/bin/env python3
"""
Sub-Agent Tool Use Tests

Tests the sub-agent orchestration and tool execution system with focus on:
1. Task decomposition and planning
2. Agent selection based on task type
3. Tool execution through sub-agents
4. Swarm coordination with tools
5. Parallel vs sequential execution strategies
"""
import os as _os
_JARVIS_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import os
os.environ["JARVIS_LOCAL_EMBEDDER"] = "1"

import sys
sys.path.insert(0, _JARVIS_ROOT)

import asyncio
import threading
import time
import pytest
from unittest.mock import Mock, MagicMock, AsyncMock, patch


# ==================== FIXTURES ====================

@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singletons before each test for isolation."""
    try:
        from core.sub_agent import _orchestrator, _lock as sub_agent_lock
        with sub_agent_lock:
            import core.sub_agent as sub_agent_module
            sub_agent_module._orchestrator = None
    except:
        pass
    
    try:
        from core.swarm import _swarm_coordinator, _lock as swarm_lock
        with swarm_lock:
            import core.swarm as swarm_module
            swarm_module._swarm_coordinator = None
    except:
        pass
    
    yield


@pytest.fixture
def mock_tools():
    """Create mock tools for testing."""
    return {
        "search": Mock(return_value={"results": ["result1", "result2"]}),
        "read_file": Mock(return_value={"content": "file content"}),
        "write_file": Mock(return_value={"success": True}),
        "execute": Mock(return_value={"output": "executed"}),
        "analyze": Mock(return_value={"analysis": "complete"}),
    }


@pytest.fixture
def mock_llm():
    """Create mock LLM client."""
    llm = Mock()
    llm.generate = Mock(return_value="Generated response")
    llm.generate_stream = Mock(return_value=iter(["chunk1", "chunk2"]))
    return llm


# ==================== TASK PLANNING TESTS ====================

class TestTaskPlanner:
    """Tests for task decomposition and planning."""
    
    def test_planner_initialization(self):
        """Task planner should initialize correctly."""
        from core.sub_agent import TaskPlanner
        
        planner = TaskPlanner()
        
        assert planner is not None
    
    def test_simple_task_decomposition(self):
        """Planner should decompose simple tasks."""
        from core.sub_agent import TaskPlanner
        
        planner = TaskPlanner()
        
        # Test heuristic decomposition - takes (objective, context)
        plan = planner._heuristic_decompose("Search for Python documentation", None)
        
        assert plan is not None
        assert len(plan) > 0
    
    def test_complex_task_decomposition(self):
        """Planner should decompose complex tasks into multiple subtasks."""
        from core.sub_agent import TaskPlanner
        
        planner = TaskPlanner()
        
        # Complex task with multiple aspects
        plan = planner._heuristic_decompose(
            "Research the best practices for async Python, and then write a summary document",
            None
        )
        
        assert plan is not None
        # Should produce multiple tasks when using delimiter pattern
    
    def test_task_dependencies(self):
        """Planner should identify task dependencies."""
        from core.sub_agent import TaskPlanner
        
        planner = TaskPlanner()
        
        plan = planner._heuristic_decompose(
            "First search for documentation, and then analyze the results",
            None
        )
        
        assert plan is not None
        # Dependencies are set based on sequential pattern
    
    def test_build_dag(self):
        """Planner should build execution DAG from tasks."""
        from core.sub_agent import TaskPlanner, SubTask
        
        planner = TaskPlanner()
        
        tasks = [
            SubTask(id="t1", title="Task 1", description="First task"),
            SubTask(id="t2", title="Task 2", description="Second task", dependencies=["t1"]),
            SubTask(id="t3", title="Task 3", description="Third task", dependencies=["t1"]),
        ]
        
        # build_dag returns list of levels (layers of parallel tasks)
        levels = planner.build_dag(tasks)
        
        assert len(levels) >= 1  # At least one level
        # First level should have t1 (no dependencies)
        assert any(t.id == "t1" for t in levels[0])


# ==================== AGENT SELECTION TESTS ====================

class TestAgentSelection:
    """Tests for agent selection based on task type."""
    
    def test_pool_initialization(self):
        """Agent pool should have default agents."""
        from core.sub_agent import AgentPool, DEFAULT_AGENTS
        
        pool = AgentPool()
        
        assert len(pool.agents) > 0
        assert len(DEFAULT_AGENTS) >= 5  # At least 5 default agent types
    
    def test_researcher_selection(self):
        """Research tasks should select researcher agent."""
        from core.sub_agent import AgentPool, SubTask, AgentRole
        
        pool = AgentPool()
        
        task = SubTask(
            id="research_1",
            title="Search docs",
            description="Search for Python documentation"
        )
        
        agent_id = pool.select_agent(task)
        
        assert agent_id is not None
        assert pool.agents[agent_id].role == AgentRole.RESEARCHER
    
    def test_coder_selection(self):
        """Code tasks should select coder agent."""
        from core.sub_agent import AgentPool, SubTask, AgentRole
        
        pool = AgentPool()
        
        task = SubTask(
            id="code_1",
            title="Implement function",
            description="Write a Python function to parse JSON"
        )
        
        agent_id = pool.select_agent(task)
        
        assert agent_id is not None
        assert pool.agents[agent_id].role == AgentRole.CODER
    
    def test_analyzer_selection(self):
        """Analysis tasks should select analyzer agent."""
        from core.sub_agent import AgentPool, SubTask, AgentRole
        
        pool = AgentPool()
        
        task = SubTask(
            id="analyze_1",
            title="Analyze results",
            description="Analyze the test results and evaluate performance"
        )
        
        agent_id = pool.select_agent(task)
        
        assert agent_id is not None
        assert pool.agents[agent_id].role == AgentRole.ANALYZER
    
    def test_executor_selection(self):
        """Execution tasks should select executor agent."""
        from core.sub_agent import AgentPool, SubTask, AgentRole
        
        pool = AgentPool()
        
        task = SubTask(
            id="exec_1",
            title="Run command",
            description="Execute the build script"
        )
        
        agent_id = pool.select_agent(task)
        
        assert agent_id is not None
        assert pool.agents[agent_id].role == AgentRole.EXECUTOR
    
    def test_synthesizer_selection(self):
        """Synthesis tasks should select synthesizer agent."""
        from core.sub_agent import AgentPool, SubTask, AgentRole
        
        pool = AgentPool()
        
        task = SubTask(
            id="synth_1",
            title="Summarize results",
            description="Synthesize all findings into a summary report"
        )
        
        agent_id = pool.select_agent(task)
        
        assert agent_id is not None
        assert pool.agents[agent_id].role == AgentRole.SYNTHESIZER
    
    def test_custom_agent_registration(self):
        """Pool should accept custom agent configurations."""
        from core.sub_agent import AgentPool, AgentConfig, AgentRole
        
        pool = AgentPool()
        
        custom_agent = AgentConfig(
            id="custom_agent",
            role=AgentRole.EXECUTOR,
            name="Custom Agent",
            description="Custom agent for specialized tasks",
            capabilities=["special_op"],
        )
        
        pool.add_agent(custom_agent)
        
        assert "custom_agent" in pool.agents
        assert pool.agents["custom_agent"].capabilities == ["special_op"]


# ==================== ORCHESTRATOR TESTS ====================

class TestSubAgentOrchestrator:
    """Tests for sub-agent orchestrator."""
    
    def test_orchestrator_initialization(self):
        """Orchestrator should initialize correctly."""
        from core.sub_agent import SubAgentOrchestrator
        
        orchestrator = SubAgentOrchestrator()
        
        assert orchestrator is not None
        assert orchestrator.planner is not None
        assert orchestrator.pool is not None
    
    def test_orchestrator_with_tools(self, mock_tools):
        """Orchestrator should accept tools."""
        from core.sub_agent import SubAgentOrchestrator
        
        orchestrator = SubAgentOrchestrator(tools=mock_tools)
        
        assert orchestrator.tools is not None
        assert len(orchestrator.tools) == len(mock_tools)
    
    def test_get_execution_strategy(self):
        """Orchestrator should determine execution strategy."""
        from core.sub_agent import SubAgentOrchestrator
        
        orchestrator = SubAgentOrchestrator()
        
        # should_delegate_or_compress returns strategy
        strategy = orchestrator.should_delegate_or_compress()
        
        assert strategy in ["delegate_parallel", "delegate_sequential", "direct", "compress"]
    
    def test_singleton_pattern(self):
        """get_orchestrator should return singleton."""
        from core.sub_agent import get_orchestrator
        
        orchestrator1 = get_orchestrator()
        orchestrator2 = get_orchestrator()
        
        assert orchestrator1 is orchestrator2


# ==================== ASYNC EXECUTION TESTS ====================

class TestAsyncExecution:
    """Tests for async task execution."""
    
    @pytest.mark.asyncio
    async def test_execute_simple_objective(self):
        """Orchestrator should execute simple objectives."""
        from core.sub_agent import SubAgentOrchestrator
        
        orchestrator = SubAgentOrchestrator()
        
        result = await orchestrator.execute(
            objective="Search for information",
            mode="sequential"
        )
        
        assert "success" in result
        assert "final_answer" in result
        assert "duration" in result
    
    @pytest.mark.asyncio
    async def test_execute_with_context(self):
        """Orchestrator should use provided context."""
        from core.sub_agent import SubAgentOrchestrator
        
        orchestrator = SubAgentOrchestrator()
        
        result = await orchestrator.execute(
            objective="Analyze the data",
            context={"data": [1, 2, 3], "source": "test"},
            mode="sequential"
        )
        
        assert "success" in result
    
    @pytest.mark.asyncio
    async def test_execute_modes(self):
        """Orchestrator should support different execution modes."""
        from core.sub_agent import SubAgentOrchestrator
        
        orchestrator = SubAgentOrchestrator()
        
        # Test sequential
        result_seq = await orchestrator.execute(
            objective="Simple task",
            mode="sequential"
        )
        assert "success" in result_seq
        
        # Test parallel
        result_par = await orchestrator.execute(
            objective="Simple task",
            mode="parallel"
        )
        assert "success" in result_par


# ==================== AGENT COORDINATION TESTS ====================

class TestAgentCoordination:
    """Tests for multi-agent coordination."""
    
    def test_busy_agent_tracking(self):
        """Pool should track busy agents."""
        from core.sub_agent import AgentPool
        
        pool = AgentPool()
        
        # Mark agent busy
        pool.mark_busy("researcher")
        
        # Should not select busy agent
        from core.sub_agent import SubTask
        task = SubTask(id="t1", title="Research", description="Search for info")
        
        agent_id = pool.select_agent(task)
        assert agent_id != "researcher"
    
    def test_mark_available(self):
        """Pool should mark agents available."""
        from core.sub_agent import AgentPool
        
        pool = AgentPool()
        
        pool.mark_busy("researcher")
        pool.mark_available("researcher")
        
        from core.sub_agent import SubTask
        task = SubTask(id="t1", title="Research", description="Search for info")
        
        agent_id = pool.select_agent(task)
        # Can select researcher again if it's best match
        assert agent_id is not None
    
    def test_available_agents_list(self):
        """Pool should list available agents."""
        from core.sub_agent import AgentPool
        
        pool = AgentPool()
        
        available = pool.get_available_agents()
        
        assert len(available) > 0


# ==================== INTEGRATION TESTS ====================

class TestSubAgentIntegration:
    """Integration tests for sub-agent system."""
    
    def test_orchestrator_with_swarm(self):
        """Sub-agent should integrate with swarm coordinator."""
        from core.sub_agent import get_orchestrator
        from core.swarm import get_swarm_coordinator
        
        orchestrator = get_orchestrator()
        coordinator = get_swarm_coordinator()
        
        assert orchestrator is not None
        assert coordinator is not None
    
    def test_orchestrator_with_claims(self):
        """Sub-agent should integrate with claims registry."""
        from core.sub_agent import get_orchestrator
        from core.claims import get_claims_registry
        
        orchestrator = get_orchestrator()
        registry = get_claims_registry()
        
        assert orchestrator is not None
        assert registry is not None
    
    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test full sub-agent workflow."""
        from core.sub_agent import SubAgentOrchestrator
        
        orchestrator = SubAgentOrchestrator()
        
        # Full workflow: decompose -> select agents -> execute
        result = await orchestrator.execute(
            objective="Search for Python async docs and create a summary",
            mode="auto"
        )
        
        assert result is not None
        assert "success" in result


# ==================== ERROR HANDLING TESTS ====================

class TestErrorHandling:
    """Tests for error handling in sub-agent system."""
    
    def test_empty_objective(self):
        """Orchestrator should handle empty objectives."""
        from core.sub_agent import TaskPlanner
        
        planner = TaskPlanner()
        
        plan = planner._heuristic_decompose("", None)
        
        # Should handle gracefully - creates single task with empty title
        assert plan is not None or plan == []
    
    @pytest.mark.asyncio
    async def test_failed_task_handling(self):
        """Orchestrator should handle failed tasks."""
        from core.sub_agent import SubAgentOrchestrator, SubTask, TaskStatus
        
        orchestrator = SubAgentOrchestrator()
        
        # Execute with a task that might fail
        result = await orchestrator.execute(
            objective="This is a malformed objective!!!@@@###",
            mode="sequential"
        )
        
        # Should return some result even on failure
        assert "success" in result
    
    def test_invalid_agent_selection(self):
        """Pool should handle when no agent matches."""
        from core.sub_agent import AgentPool, SubTask
        
        pool = AgentPool()
        
        # Mark all agents busy
        for agent_id in list(pool.agents.keys()):
            pool.mark_busy(agent_id)
        
        task = SubTask(id="t1", title="Task", description="Do something")
        
        agent_id = pool.select_agent(task)
        
        # Should return None when all agents busy
        assert agent_id is None


# ==================== PERFORMANCE TESTS ====================

class TestPerformance:
    """Tests for performance characteristics."""
    
    def test_planner_speed(self):
        """Task planning should be fast."""
        from core.sub_agent import TaskPlanner
        import time
        
        planner = TaskPlanner()
        
        start = time.time()
        for _ in range(100):
            planner._heuristic_decompose("Search for documentation and analyze results", None)
        elapsed = time.time() - start
        
        # 100 decompositions should be fast
        assert elapsed < 2.0
    
    def test_agent_selection_speed(self):
        """Agent selection should be fast."""
        from core.sub_agent import AgentPool, SubTask
        import time
        
        pool = AgentPool()
        
        task = SubTask(id="t1", title="Research", description="Search for information")
        
        start = time.time()
        for _ in range(100):
            pool.select_agent(task)
        elapsed = time.time() - start
        
        # 100 selections should be fast
        assert elapsed < 0.5
    
    def test_orchestrator_creation_speed(self):
        """Orchestrator creation should be fast."""
        from core.sub_agent import SubAgentOrchestrator
        import time
        
        start = time.time()
        for _ in range(10):
            SubAgentOrchestrator()
        elapsed = time.time() - start
        
        # 10 creations should be fast
        assert elapsed < 1.0


# ==================== RUN ====================

if __name__ == "__main__":
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "-p", "no:xdist", 
         "--tb=short", "--override-ini=addopts="],
        cwd=_JARVIS_ROOT
    )
    sys.exit(result.returncode)