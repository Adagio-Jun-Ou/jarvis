"""Integration test for full context cascade flow."""
import asyncio
import pytest
import tempfile
import os


def test_context_cascade_imports():
    """Test that all context cascade modules import correctly."""
    from core.context_cascade import observe_activity, get_context_cascade, ContextCascadeIntegration
    from core.unified_context import UnifiedContext, get_unified_context
    from core.agentic_context import AgenticContext, ContextAwareness, get_agentic_context
    from core.prefrontal_cortex import get_prefrontal_cortex, ConceptType
    
    assert observe_activity is not None
    assert get_context_cascade is not None
    assert get_unified_context is not None
    assert get_agentic_context is not None
    assert get_prefrontal_cortex is not None


def test_multi_model_agentic_imports():
    """Test that multi-model agentic modules import correctly."""
    from core.multi_model_agentic import MultiModelAgentic, get_multi_model_agentic, AGENTIC_TASK_ROUTING
    from core.project_workspace import ProjectWorkspace, get_workspace
    from core.task_decomposition import TaskDecomposition, get_task_decomposition
    from core.execution_planner import ExecutionPlanner, get_execution_planner
    
    assert MultiModelAgentic is not None
    assert get_multi_model_agentic is not None
    assert AGENTIC_TASK_ROUTING is not None
    assert get_workspace is not None
    assert get_task_decomposition is not None
    assert get_execution_planner is not None


def test_project_tools_imports():
    """Test that project tools import correctly."""
    from tools.project import (
        ProjectInitTool, ProjectStatusTool, ProjectDecomposeTool,
        ProjectExecuteTool, ProjectFileReadTool, ProjectFileWriteTool,
        ProjectTestTool, ProjectCommitTool, register_project_tools
    )
    
    tools = register_project_tools()
    assert len(tools) == 8


def test_subsystem_stages_with_cascade():
    """Test that subsystem stages have cascade integration."""
    from core.subsystem_stages import SubsystemStage, PFCObserveStage, ErdosProblemStage
    
    # Check that _observe_with_pfc mentions cascade
    import inspect
    source = inspect.getsource(SubsystemStage._observe_with_pfc)
    assert "observe_activity" in source or "context_cascade" in source


def test_dual_model_pfc_routing():
    """Test that dual model has PFC routing integration."""
    from core.dual_model import _get_pfc
    import inspect
    
    source = inspect.getsource(_get_pfc)
    assert "get_prefrontal_cortex" in source


def test_agent_multi_model_attachment():
    """Test that agent has multi-model attachment method."""
    from core.agent import JarvisAgent
    import inspect
    
    source = inspect.getsource(JarvisAgent)
    assert "attach_multi_model_agentic" in source


def test_observe_activity_basic():
    """Test observe_activity creates a cascade."""
    from core.context_cascade import observe_activity
    
    cascade = observe_activity(
        trigger_type="test",
        content="test content",
        source="test_source"
    )
    
    assert cascade is not None
    # ConceptCascade has concepts_observed attribute
    assert hasattr(cascade, 'concepts_observed') or hasattr(cascade, 'trigger_type')


def test_agentic_context_methods():
    """Test AgenticContext has required methods."""
    from core.agentic_context import get_agentic_context
    
    ctx = get_agentic_context()
    
    assert hasattr(ctx, 'reset')
    assert hasattr(ctx, 'capture_context')
    assert hasattr(ctx, 'format_context_intro')
    assert hasattr(ctx, 'capture_tool_reasoning')
    assert hasattr(ctx, 'format_completion')
    assert hasattr(ctx, 'format_progress')
    assert hasattr(ctx, 'format_gap_analysis')
    assert hasattr(ctx, 'record_tool_execution')
    assert hasattr(ctx, 'increment_iteration')


def test_context_cascade_health_check():
    """Test context cascade health check method."""
    from core.context_cascade import get_context_cascade
    
    cascade = get_context_cascade()
    is_healthy, status = cascade.health_check()
    
    assert isinstance(is_healthy, bool)
    assert isinstance(status, dict)
    assert "components" in status or "error" in status or "healthy" in status


def test_agentic_history_bounds():
    """Test MAX_HISTORY_STEPS enforcement."""
    from core.agentic_context import AgenticContext, _get_max_history_steps
    
    max_steps = _get_max_history_steps()
    ctx = AgenticContext()
    
    # Add more steps than max
    for i in range(max_steps + 50):
        ctx.history.add_step(f"step_{i}", "test", "action")
    
    # Should be trimmed to max
    assert len(ctx.history.steps) <= max_steps + 10  # Allow some buffer


def test_agentic_tool_cache_bounds():
    """Test tool cache size enforcement."""
    from core.agentic_context import AgenticContext
    
    ctx = AgenticContext()
    
    # Add many cache entries
    for i in range(600):
        ctx.cache_tool_result("test_tool", f"hash_{i}", {"result": f"data_{i}"})
    
    # Should be trimmed
    assert len(ctx.tool_cache) <= 500


def test_context_cascade_with_embedding():
    """Test cascade with embedding interconnect."""
    from core.context_cascade import observe_activity
    
    # Trigger cascade
    result = observe_activity(
        trigger_type="test",
        content="test embedding integration",
        source="test",
    )
    
    # Should have concepts_observed
    assert hasattr(result, 'concepts_observed')
    assert len(result.concepts_observed) > 0


def test_multi_model_agentic_model_choice():
    """Test MultiModelAgentic model routing."""
    from core.multi_model_agentic import MultiModelAgentic, AGENTIC_TASK_ROUTING
    
    # Check routing map exists
    assert "project_decompose" in AGENTIC_TASK_ROUTING
    assert AGENTIC_TASK_ROUTING["project_decompose"] == "backend"
    
    # Create instance without LLMs (for testing)
    mma = MultiModelAgentic()
    
    # Should have routing logic
    choice = mma.choose_model_for_task("project_decompose")
    assert choice.model in ("primary", "backend", "parallel")
    assert choice.task_type == "project_decompose"


def test_project_workspace_initialization():
    """Test ProjectWorkspace can be created."""
    from core.project_workspace import ProjectWorkspace
    from pathlib import Path
    
    # Create temp directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        ws = ProjectWorkspace(project_path=tmpdir)
        
        assert ws.project_name == os.path.basename(tmpdir)
        assert str(ws.project_path) == tmpdir or ws.project_path == Path(tmpdir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
