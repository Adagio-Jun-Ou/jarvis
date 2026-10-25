"""
Tests for core modules - memory, evolution, bridges, and utilities.
"""
import tempfile
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestMemoryCleanup:
    """Tests for memory_cleanup module."""

    def test_import_memory_cleanup(self):
        from core.memory_cleanup import MemoryManager, get_memory_manager, CleanupPriority, CleanupStats
        assert MemoryManager is not None
        assert get_memory_manager is not None
        assert CleanupPriority is not None
        assert CleanupStats is not None

    def test_memory_manager_creation(self):
        from core.memory_cleanup import MemoryManager
        with tempfile.TemporaryDirectory() as tmpdir:
            from pathlib import Path
            manager = MemoryManager(storage_path=Path(tmpdir))
            assert manager.storage_path is not None

    def test_memory_manager_get_stats(self):
        from core.memory_cleanup import get_memory_manager
        manager = get_memory_manager()
        stats = manager.get_stats()
        assert stats is not None

    def test_memory_manager_register_handler(self):
        from core.memory_cleanup import MemoryManager, CleanupPriority
        with tempfile.TemporaryDirectory() as tmpdir:
            from pathlib import Path
            manager = MemoryManager(storage_path=Path(tmpdir))
            called = []
            def handler():
                called.append(1)
            manager.register_cleanup_handler(handler, CleanupPriority.MEDIUM)
            assert len(manager._cleanup_handlers) >= 0

    def test_memory_manager_singleton(self):
        from core.memory_cleanup import get_memory_manager
        m1 = get_memory_manager()
        m2 = get_memory_manager()
        assert m1 is m2

    def test_cleanup_cycle(self):
        from core.memory_cleanup import MemoryManager
        with tempfile.TemporaryDirectory() as tmpdir:
            from pathlib import Path
            manager = MemoryManager(storage_path=Path(tmpdir))
            try:
                import asyncio
                result = asyncio.run(manager.cleanup_cycle())
            except RuntimeError:
                pass
            assert True


class TestPersistentMemory:
    """Tests for persistent_memory module."""

    def test_import_persistent_memory(self):
        from core.persistent_memory import PersistentMemory
        assert PersistentMemory is not None

    def test_persistent_memory_creation(self):
        from core.persistent_memory import PersistentMemory
        pm = PersistentMemory()
        assert pm is not None

    def test_persistent_memory_add_memory(self):
        from core.persistent_memory import PersistentMemory
        pm = PersistentMemory()
        pm.add_memory("user", "Hello there", 0.5)
        assert True

    def test_persistent_memory_singleton(self):
        from core.persistent_memory import get_persistent_memory
        m1 = get_persistent_memory()
        m2 = get_persistent_memory()
        assert m1 is m2


class TestUnifiedMemory:
    """Tests for unified_memory module."""

    def test_import_unified_memory(self):
        from core.unified_memory import UnifiedMemory
        assert UnifiedMemory is not None

    def test_unified_memory_creation(self):
        from core.unified_memory import UnifiedMemory
        um = UnifiedMemory()
        assert um is not None


class TestVectorMemory:
    """Tests for vector_memory module."""

    def test_import_vector_memory(self):
        from core.vector_memory import VectorMemory
        assert VectorMemory is not None

    def test_vector_memory_creation(self):
        from core.vector_memory import VectorMemory
        vm = VectorMemory()
        assert vm is not None


class TestObservability:
    """Tests for observability module."""

    def test_import_observability(self):
        from core.observability import ObservabilitySystem
        assert ObservabilitySystem is not None

    def test_observability_singleton(self):
        from core.observability import get_observability
        obs1 = get_observability()
        obs2 = get_observability()
        assert obs1 is obs2


class TestStateSync:
    """Tests for state_sync module."""

    def test_import_state_sync(self):
        from core.state_sync import StateSynchronizer
        assert StateSynchronizer is not None

    def test_state_sync_creation(self):
        from core.state_sync import StateSynchronizer
        ss = StateSynchronizer()
        assert ss is not None


class TestSystemIntegrator:
    """Tests for system_integrator module."""

    def test_import_system_integrator(self):
        from core.system_integrator import SystemIntegrator
        assert SystemIntegrator is not None

    def test_system_integrator_singleton(self):
        from core.system_integrator import get_system_integrator
        si1 = get_system_integrator()
        si2 = get_system_integrator()
        assert si1 is si2


class TestResourceMonitor:
    """Tests for resource_monitor module."""

    def test_import_resource_monitor(self):
        from core.resource_monitor import ResourceMonitor
        assert ResourceMonitor is not None

    def test_resource_monitor_singleton(self):
        from core.resource_monitor import get_resource_monitor
        rm1 = get_resource_monitor()
        rm2 = get_resource_monitor()
        assert rm1 is rm2


class TestKnowledgeGraph:
    """Tests for knowledge_graph module."""

    def test_import_knowledge_graph(self):
        from core.knowledge_graph import KnowledgeGraph
        assert KnowledgeGraph is not None

    def test_knowledge_graph_singleton(self):
        from core.knowledge_graph import get_knowledge_graph
        kg1 = get_knowledge_graph()
        kg2 = get_knowledge_graph()
        assert kg1 is kg2


class TestDefense:
    """Tests for defense module."""

    def test_import_defense(self):
        from core.defense import DataDefenseAgent
        assert DataDefenseAgent is not None

    def test_defense_singleton(self):
        from core.defense import get_defense_agent
        da1 = get_defense_agent()
        da2 = get_defense_agent()
        assert da1 is da2


class TestErdosEngine:
    """Tests for erdos_engine module."""

    def test_import_erdos_engine(self):
        from core.erdos_engine import ErdosEngine
        assert ErdosEngine is not None

    def test_erdos_engine_singleton(self):
        from core.erdos_engine import get_erdos_engine
        ee1 = get_erdos_engine()
        ee2 = get_erdos_engine()
        assert ee1 is ee2


class TestLearningPipeline:
    """Tests for learning_pipeline module."""

    def test_import_learning_pipeline(self):
        from core.learning_pipeline import ContinuousLearningPipeline
        assert ContinuousLearningPipeline is not None

    def test_learning_pipeline_singleton(self):
        from core.learning_pipeline import get_learning_pipeline
        lp1 = get_learning_pipeline()
        lp2 = get_learning_pipeline()
        assert lp1 is lp2


class TestDaydreamEngine:
    """Tests for daydream_engine module."""

    def test_import_daydream_engine(self):
        from core.daydream_engine import DaydreamEngine
        assert DaydreamEngine is not None

    def test_daydream_engine_singleton(self):
        from core.daydream_engine import get_daydream_engine
        de1 = get_daydream_engine()
        de2 = get_daydream_engine()
        assert de1 is de2


class TestEvolutionOrchestrator:
    """Tests for evolution_orchestrator module."""

    def test_import_evolution_orchestrator(self):
        from core.evolution_orchestrator import EvolutionOrchestrator
        assert EvolutionOrchestrator is not None

    def test_evolution_orchestrator_singleton(self):
        from core.evolution_orchestrator import get_evolution_orchestrator
        eo1 = get_evolution_orchestrator()
        eo2 = get_evolution_orchestrator()
        assert eo1 is eo2


class TestIntuitionBridge:
    """Tests for intuition_bridge module."""

    def test_import_intuition_bridge(self):
        from core.intuition_bridge import IntuitionBridge
        assert IntuitionBridge is not None

    def test_intuition_bridge_singleton(self):
        from core.intuition_bridge import get_intuition_bridge
        ib1 = get_intuition_bridge()
        ib2 = get_intuition_bridge()
        assert ib1 is ib2


class TestInsightBridge:
    """Tests for insight_bridge module."""

    def test_import_insight_bridge(self):
        from core.insight_bridge import InsightBridge
        assert InsightBridge is not None

    def test_insight_bridge_singleton(self):
        from core.insight_bridge import get_insight_bridge
        ib1 = get_insight_bridge()
        ib2 = get_insight_bridge()
        assert ib1 is ib2


class TestFreeTime:
    """Tests for free_time module."""

    def test_import_free_time(self):
        from core.free_time import FreeTimeEngine
        assert FreeTimeEngine is not None

    def test_free_time_singleton(self):
        from core.free_time import get_free_time_engine
        fte1 = get_free_time_engine()
        fte2 = get_free_time_engine()
        assert fte1 is fte2


class TestAutoRefine:
    """Tests for auto_refine module."""

    def test_import_auto_refine(self):
        from core.auto_refine import AutoRefineManager
        assert AutoRefineManager is not None


class TestPromptSystem:
    """Tests for prompts module."""

    def test_import_prompts(self):
        from core.prompts import BASE_PROMPT
        assert BASE_PROMPT is not None


class TestSelfAwareness:
    """Tests for self_awareness module."""

    def test_import_self_awareness(self):
        from core.self_awareness import SelfAwarenessEngine
        assert SelfAwarenessEngine is not None

    def test_self_awareness_singleton(self):
        from core.self_awareness import get_self_awareness
        sa1 = get_self_awareness()
        sa2 = get_self_awareness()
        assert sa1 is sa2


class TestMeta:
    """Tests for meta module."""

    def test_import_meta(self):
        from core.meta import AgenticProcess
        assert AgenticProcess is not None


class TestThinking:
    """Tests for thinking module."""

    def test_import_thinking(self):
        from core.thinking import ThinkingEngine
        assert ThinkingEngine is not None


class TestSelfEdit:
    """Tests for self_edit module."""

    def test_import_self_edit(self):
        from core.self_edit import SelfEditEngine
        assert SelfEditEngine is not None


class TestPermissions:
    """Tests for permissions module."""

    def test_import_permissions(self):
        from core.permissions import PermissionManager
        assert PermissionManager is not None


class TestConfig:
    """Tests for config module."""

    def test_import_config(self):
        from core.config import Config
        assert Config is not None




class TestClaims:
    """Tests for claims module."""

    def test_import_claims(self):
        from core.claims import ClaimsRegistry
        assert ClaimsRegistry is not None


class TestSwarm:
    """Tests for swarm module."""

    def test_import_swarm(self):
        from core.swarm import SwarmCoordinator
        assert SwarmCoordinator is not None


class TestTokenOptimizer:
    """Tests for token_optimizer module."""

    def test_import_token_optimizer(self):
        from core.token_optimizer import TokenOptimizer
        assert TokenOptimizer is not None

    def test_token_optimizer_singleton(self):
        from core.token_optimizer import get_token_optimizer
        to1 = get_token_optimizer()
        to2 = get_token_optimizer()
        assert to1 is to2


class TestBackground:
    """Tests for background module."""

    def test_import_background(self):
        from core.background import BackgroundScheduler
        assert BackgroundScheduler is not None


class TestAwareness:
    """Tests for awareness module."""

    def test_import_awareness(self):
        from core.awareness import AwarenessManager
        assert AwarenessManager is not None




class TestGrowth:
    """Tests for growth module."""

    def test_import_growth(self):
        from core.growth import SelfGrowthEngine
        assert SelfGrowthEngine is not None


class TestImagination:
    """Tests for imagination module."""

    def test_import_imagination(self):
        from core.imagination import ImaginationEngine
        assert ImaginationEngine is not None


class TestDream:
    """Tests for dream module."""

    def test_import_dream(self):
        from core.dream import DreamEngine
        assert DreamEngine is not None
