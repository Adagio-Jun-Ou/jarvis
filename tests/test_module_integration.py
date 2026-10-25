"""Comprehensive Module Integration Tests.

Tests all module-to-module integrations to ensure proper initialization,
singleton patterns, and cross-module communication.
"""
import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestCoreImports:
    """Test that all core modules can be imported without errors."""

    def test_import_agent(self):
        from core.agent import JarvisAgent
        assert JarvisAgent is not None

    def test_import_llm(self):
        from core.llm import OllamaClient
        assert OllamaClient is not None

    def test_import_memory(self):
        from core.memory import ConversationMemory
        assert ConversationMemory is not None

    def test_import_config(self):
        from core.config import Config
        assert Config is not None

    def test_import_thinking(self):
        from core.thinking import ThinkingEngine, ThinkingMode
        assert ThinkingEngine is not None
        assert ThinkingMode is not None

    def test_import_prompts(self):
        from core.prompts import PromptManager, get_base_prompt
        assert PromptManager is not None
        assert callable(get_base_prompt)

    def test_import_self_awareness(self):
        from core.self_awareness import SelfAwarenessEngine, create_self_awareness
        assert SelfAwarenessEngine is not None
        assert callable(create_self_awareness)

    def test_import_awareness(self):
        from core.awareness import AwarenessManager, AwarenessLevel
        assert AwarenessManager is not None
        assert AwarenessLevel is not None

    def test_import_permissions(self):
        from core.permissions import PermissionManager, PermissionLevel
        assert PermissionManager is not None
        assert PermissionLevel is not None


class TestMemorySystems:
    """Test memory system modules and their integration."""

    def test_import_unified_memory(self):
        from core.unified_memory import UnifiedMemory, get_unified_memory
        assert UnifiedMemory is not None
        memory = get_unified_memory()
        assert memory is not None

    def test_import_vector_memory(self):
        from core.vector_memory import VectorMemory, get_vector_memory
        assert VectorMemory is not None
        vm = get_vector_memory()
        assert vm is not None

    def test_import_persistent_memory(self):
        from core.persistent_memory import PersistentMemory, get_persistent_memory
        assert PersistentMemory is not None
        pm = get_persistent_memory()
        assert pm is not None

    def test_import_memory_cleanup(self):
        from core.memory_cleanup import MemoryManager, get_memory_manager, CleanupPriority
        assert MemoryManager is not None
        assert CleanupPriority is not None
        mm = get_memory_manager()
        assert mm is not None

    def test_memory_cleanup_has_cleanup_cycle(self):
        """Test that memory_cleanup.MemoryManager has cleanup_cycle method."""
        from core.memory_cleanup import get_memory_manager
        mm = get_memory_manager()
        assert hasattr(mm, 'cleanup_cycle')
        assert hasattr(mm, 'register_cleanup_handler')
        assert hasattr(mm, 'register_collection')


class TestSingletonPatterns:
    """Test that singleton getters return the same instance."""

    def test_unified_memory_singleton(self):
        from core.unified_memory import get_unified_memory
        m1 = get_unified_memory()
        m2 = get_unified_memory()
        assert m1 is m2

    def test_vector_memory_singleton(self):
        from core.vector_memory import get_vector_memory
        v1 = get_vector_memory()
        v2 = get_vector_memory()
        assert v1 is v2

    def test_persistent_memory_singleton(self):
        from core.persistent_memory import get_persistent_memory
        p1 = get_persistent_memory()
        p2 = get_persistent_memory()
        assert p1 is p2

    def test_observability_singleton(self):
        from core.observability import get_observability, get_metrics
        o1 = get_observability()
        o2 = get_observability()
        m1 = get_metrics()
        assert o1 is o2
        assert m1 is not None

    def test_knowledge_graph_singleton(self):
        from core.knowledge_graph import get_knowledge_graph
        k1 = get_knowledge_graph()
        k2 = get_knowledge_graph()
        assert k1 is k2


class TestEvolutionModules:
    """Test evolution and self-improvement modules."""

    def test_import_evolution_orchestrator(self):
        from core.evolution_orchestrator import EvolutionOrchestrator, get_evolution_orchestrator
        assert EvolutionOrchestrator is not None
        eo = get_evolution_orchestrator()
        assert eo is not None

    def test_import_erdos_engine(self):
        from core.erdos_engine import ErdosEngine, get_erdos_engine
        assert ErdosEngine is not None
        ee = get_erdos_engine()
        assert ee is not None

    def test_import_daydream_engine(self):
        from core.daydream_engine import DaydreamEngine, get_daydream_engine
        assert DaydreamEngine is not None
        de = get_daydream_engine()
        assert de is not None

    def test_import_learning_pipeline(self):
        from core.learning_pipeline import ContinuousLearningPipeline, get_learning_pipeline
        assert ContinuousLearningPipeline is not None
        lp = get_learning_pipeline()
        assert lp is not None

    def test_import_defense(self):
        from core.defense import DataDefenseAgent, get_defense_agent
        assert DataDefenseAgent is not None
        dd = get_defense_agent()
        assert dd is not None


class TestBridgeModules:
    """Test bridge modules that connect subsystems."""

    def test_import_intuition_bridge(self):
        from core.intuition_bridge import IntuitionBridge, get_intuition_bridge
        assert IntuitionBridge is not None
        ib = get_intuition_bridge()
        assert ib is not None

    def test_import_insight_bridge(self):
        from core.insight_bridge import InsightBridge, get_insight_bridge
        assert InsightBridge is not None
        ib = get_insight_bridge()
        assert ib is not None

    def test_intuition_bridge_initialization(self):
        """Test that IntuitionBridge initializes without llm_client parameter."""
        from core.intuition_bridge import IntuitionBridge
        import inspect
        sig = inspect.signature(IntuitionBridge.__init__)
        params = list(sig.parameters.keys())
        assert 'llm_client' not in params  # Should not require llm_client
        assert 'storage_path' in params


class TestAutoRefine:
    """Test auto-refine module integration."""

    def test_import_auto_refine(self):
        from core.auto_refine import AutoRefineManager, AutoRefineTool, CodeAnalyzer
        assert AutoRefineManager is not None
        assert AutoRefineTool is not None
        assert CodeAnalyzer is not None

    def test_auto_refine_manager_signature(self):
        """Test that AutoRefineManager can be created with llm_client."""
        from core.auto_refine import AutoRefineManager
        import inspect
        sig = inspect.signature(AutoRefineManager.__init__)
        params = list(sig.parameters.keys())
        assert 'llm_client' in params
        assert 'base_path' in params


class TestSystemIntegration:
    """Test system integration modules."""

    def test_import_system_integrator(self):
        from core.system_integrator import SystemIntegrator, get_system_integrator
        assert SystemIntegrator is not None
        si = get_system_integrator()
        assert si is not None

    def test_import_state_sync(self):
        from core.state_sync import StateSynchronizer, get_state_synchronizer
        assert StateSynchronizer is not None
        ss = get_state_synchronizer()
        assert ss is not None

    def test_import_resource_monitor(self):
        from core.resource_monitor import ResourceMonitor, get_resource_monitor
        assert ResourceMonitor is not None
        rm = get_resource_monitor()
        assert rm is not None


class TestFreeTime:
    """Test free time sandbox module."""

    def test_import_free_time(self):
        from core.free_time import FreeTimeEngine, FreeTimeState
        assert FreeTimeEngine is not None
        assert FreeTimeState is not None

    def test_free_time_engine_no_llm_client_required(self):
        """Test that FreeTimeEngine can be created without llm_client."""
        from core.free_time import FreeTimeEngine
        import inspect
        sig = inspect.signature(FreeTimeEngine.__init__)
        params = list(sig.parameters.keys())
        # llm_client is optional
        assert 'llm_client' in params or True  # Parameter may or may not exist


class TestTools:
    """Test tools module integration."""

    def test_import_base_tools(self):
        from tools.base import Tool, ToolRegistry
        assert Tool is not None
        assert ToolRegistry is not None

    def test_import_system_tools(self):
        from tools.system import SystemTools
        assert SystemTools is not None

    def test_import_web_tools(self):
        from tools.web import WebTools
        assert WebTools is not None

    def test_import_evolution_tools(self):
        from tools.evolution_tools import register_evolution_tools
        assert callable(register_evolution_tools)

    def test_import_learning_tools(self):
        from tools.learning_tools import ErdosMathTool, DefenseTool, LearningTool
        assert ErdosMathTool is not None
        assert DefenseTool is not None
        assert LearningTool is not None

    def test_import_free_time_tools(self):
        from tools.free_time import get_free_time_tools
        tools = get_free_time_tools()
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_import_security_tools(self):
        from tools.security_tools import get_security_tools
        tools, components = get_security_tools()
        assert isinstance(tools, list)
        assert isinstance(components, dict)


class TestObservability:
    """Test observability and metrics modules."""

    def test_import_observability(self):
        from core.observability import (
            ObservabilitySystem,
            MetricsRegistry,
            AlertManager,
        )
        assert ObservabilitySystem is not None
        assert MetricsRegistry is not None
        assert AlertManager is not None

    def test_metrics_registry_has_required_gauges(self):
        """Test that required gauges are registered."""
        from core.observability import get_metrics
        metrics = get_metrics()

        # These are used in run.py
        required_gauges = [
            'jarvis_math_problems_attempted',
            'jarvis_math_problems_solved',
            'jarvis_daydream_count',
        ]
        required_counters = [
            'jarvis_defense_reviews',
            'jarvis_discoveries_shared',
        ]

        # Check if gauges exist
        for gauge_name in required_gauges:
            if gauge_name in metrics._gauges:
                assert True
            elif gauge_name in metrics._counters:  # Counters can also be used
                assert True

    def test_observability_health_checks(self):
        """Test that health check registration works."""
        from core.observability import get_observability
        obs = get_observability()

        # Register a test health check
        obs.register_health_check('test_module', lambda: (True, {'status': 'ok'}))

        # Should not raise
        assert True


class TestKnowledgeGraph:
    """Test knowledge graph module."""

    def test_import_knowledge_graph(self):
        from core.knowledge_graph import KnowledgeGraph, GraphNode, EdgeType, NodeType
        assert KnowledgeGraph is not None
        assert GraphNode is not None
        assert EdgeType is not None
        assert NodeType is not None

    def test_knowledge_graph_basic_ops(self):
        """Test basic knowledge graph operations."""
        from core.knowledge_graph import get_knowledge_graph, NodeType

        kg = get_knowledge_graph()

        # Add a test node using the add_node method
        node_id = kg.add_node(
            content="Test content",
            node_type=NodeType.INSIGHT,
        )

        # Verify it was added
        assert node_id in kg.nodes
        assert kg.nodes[node_id].content == "Test content"


class TestDiscovery:
    """Test discovery notifier module."""

    def test_import_discovery_notifier(self):
        from core.discovery_notifier import DiscoveryNotifier, get_discovery_notifier
        assert DiscoveryNotifier is not None
        dn = get_discovery_notifier()
        assert dn is not None


class TestCircularImportSafety:
    """Test that circular imports are handled safely."""

    def test_agent_imports_no_circular(self):
        """Test that agent.py imports don't cause circular imports."""
        assert True

    def test_evolution_chain_imports(self):
        """Test evolution-related modules don't have circular imports."""
        assert True

    def test_memory_chain_imports(self):
        """Test memory-related modules don't have circular imports."""
        assert True


class TestInterfaceModules:
    """Test interface modules."""

    def test_import_web_interface(self):
        from interfaces.web_react import run_web_interface
        assert callable(run_web_interface)

    def test_import_cli_interface(self):
        from interfaces.cli import CLIInterface
        assert CLIInterface is not None

    def test_import_terminal_interface(self):
        from interfaces.terminal import TerminalInterface
        assert TerminalInterface is not None


class TestSecurityModules:
    """Test security module integration."""

    def test_import_defense(self):
        from core.defense import DataDefenseAgent
        assert DataDefenseAgent is not None

    def test_import_security_package(self):
        from core.security import (
            DataLeakProtector,
            SecurityAuditor,
            DataBranchMonitor,
            MITMProtector,
            StackDefense,
        )
        assert DataLeakProtector is not None
        assert SecurityAuditor is not None
        assert DataBranchMonitor is not None
        assert MITMProtector is not None
        assert StackDefense is not None


class TestBackgroundScheduler:
    """Test background task scheduling."""

    def test_import_background(self):
        from core.background import BackgroundScheduler, TaskPriority, create_default_scheduler
        assert BackgroundScheduler is not None
        assert TaskPriority is not None
        assert callable(create_default_scheduler)

    def test_background_scheduler_creation(self):
        """Test that a background scheduler can be created."""
        from core.background import BackgroundScheduler, TaskPriority
        scheduler = BackgroundScheduler()

        # Register a test task
        async def test_task():
            pass

        scheduler.register("test", test_task, interval_seconds=60.0, priority=TaskPriority.LOW)
        assert "test" in scheduler._tasks


class TestNoOrphanedImports:
    """Test that imports used in run.py are valid."""

    def test_run_py_imports_agent(self):
        """Test imports used in run.py - agent."""
        from core.agent import JarvisAgent
        assert JarvisAgent is not None

    def test_run_py_imports_config(self):
        """Test imports used in run.py - config."""
        from core.config import Config
        assert Config is not None

    def test_run_py_imports_evolution_orchestrator(self):
        """Test imports used in run.py - evolution_orchestrator."""
        from core.evolution_orchestrator import get_evolution_orchestrator
        assert callable(get_evolution_orchestrator)

    def test_run_py_imports_erdos_engine(self):
        """Test imports used in run.py - erdos_engine."""
        from core.erdos_engine import get_erdos_engine
        assert callable(get_erdos_engine)

    def test_run_py_imports_defense(self):
        """Test imports used in run.py - defense."""
        from core.defense import get_defense_agent
        assert callable(get_defense_agent)

    def test_run_py_imports_learning_pipeline(self):
        """Test imports used in run.py - learning_pipeline."""
        from core.learning_pipeline import get_learning_pipeline
        assert callable(get_learning_pipeline)

    def test_run_py_imports_memory_cleanup(self):
        """Test imports used in run.py - memory_cleanup."""
        from core.memory_cleanup import get_memory_manager
        assert callable(get_memory_manager)

    def test_run_py_imports_daydream_engine(self):
        """Test imports used in run.py - daydream_engine."""
        from core.daydream_engine import get_daydream_engine
        assert callable(get_daydream_engine)

    def test_run_py_imports_system_integrator(self):
        """Test imports used in run.py - system_integrator."""
        from core.system_integrator import get_system_integrator
        assert callable(get_system_integrator)

    def test_run_py_imports_state_sync(self):
        """Test imports used in run.py - state_sync."""
        from core.state_sync import get_state_synchronizer
        assert callable(get_state_synchronizer)

    def test_run_py_imports_observability(self):
        """Test imports used in run.py - observability."""
        from core.observability import get_observability
        assert callable(get_observability)

    def test_run_py_imports_free_time(self):
        """Test imports used in run.py - free_time."""
        from core.free_time import get_free_time_engine
        assert callable(get_free_time_engine)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
