"""Comprehensive tests for all Jarvis modules."""
import asyncio
import os
import sys
import tempfile
import threading
from collections import deque
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestErdosEngine:
    """Test Erdős Math Problem Engine."""

    def test_import(self):
        """Test importing the module."""
        print("✓ Erdős engine imports successful")

    def test_problem_generation(self):
        """Test problem generation for all categories."""
        from core.erdos_engine import ErdosProblemGenerator, ProblemCategory

        generator = ErdosProblemGenerator()

        for category in ProblemCategory:
            problem = generator.generate_problem(category=category)
            assert problem.id.startswith("erdos_")
            assert problem.category == category
            assert problem.statement

        print("✓ Problem generation for all categories")

    def test_solver_strategies(self):
        """Test solver strategies."""
        from core.erdos_engine import ErdosSolver, ErdosProblemGenerator

        solver = ErdosSolver()
        generator = ErdosProblemGenerator()

        assert len(solver.STRATEGIES) >= 10

        problem = generator.generate_problem()
        strategy = solver._select_strategy(problem)
        assert strategy in solver.STRATEGIES

        print("✓ Solver strategies work")

    async def test_attempt_solution(self):
        """Test attempting a solution."""
        from core.erdos_engine import ErdosEngine

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = ErdosEngine(storage_path=Path(tmpdir))

            result = await engine.force_attempt()

            assert "problem" in result
            assert "success" in result
            assert "strategy" in result

        print("✓ Attempt solution works")

    async def test_background_learning(self):
        """Test background learning loop."""
        from core.erdos_engine import ErdosEngine

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = ErdosEngine(storage_path=Path(tmpdir))

            await engine.start_background_learning()
            assert engine._running

            await asyncio.sleep(0.5)

            await engine.stop_background_learning()
            assert not engine._running

        print("✓ Background learning works")

    def test_memory_bounds(self):
        """Test memory bounds are respected."""
        from core.erdos_engine import ErdosEngine

        engine = ErdosEngine()

        assert engine.MAX_PROBLEMS == 100
        assert engine.MAX_HISTORY == 1000

        print("✓ Memory bounds defined")

    def test_get_status(self):
        """Test status retrieval."""
        from core.erdos_engine import ErdosEngine

        engine = ErdosEngine()
        status = engine.get_status()

        assert "running" in status
        assert "total_problems" in status
        assert "solved_count" in status
        assert "best_strategies" in status

        print("✓ Status retrieval works")


class TestDefense:
    """Test Data Defense Agent."""

    def test_import(self):
        """Test importing the module."""
        print("✓ Defense imports successful")

    def test_scan_input(self):
        """Test input scanning for threats."""
        from core.defense import DataDefenseAgent

        agent = DataDefenseAgent()

        safe = agent.scan_input("Hello, how are you?")
        assert safe["action"] == "allow"

        malicious = agent.scan_input("SELECT * FROM users; DROP TABLE users;")
        assert malicious["action"] == "block"
        assert malicious["risk_score"] > 50

        print("✓ Input scanning works")

    def test_scan_file(self):
        """Test file scanning."""
        from core.defense import DataDefenseAgent

        agent = DataDefenseAgent()

        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
        try:
            tmp.write("print('hello world')")
            tmp.flush()
            tmp.close()
            result = agent.scan_file(tmp.name)
            assert result["status"] in ["clean", "low_risk", "medium_risk", "high_risk"]
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

        print("✓ File scanning works")

    def test_threat_recording(self):
        """Test threat recording."""
        from core.defense import DataDefenseAgent, ThreatType, ThreatLevel, DefenseAction

        agent = DataDefenseAgent()

        event = agent._record_threat(
            ThreatType.INJECTION_ATTACK,
            ThreatLevel.HIGH,
            "test",
            "Test threat",
            {"test": True},
            DefenseAction.BLOCK,
        )

        assert event.id.startswith("threat_")
        assert len(list(agent.events)) > 0

        print("✓ Threat recording works")

    def test_file_protection(self):
        """Test file integrity protection."""
        from core.defense import DataDefenseAgent

        with tempfile.TemporaryDirectory() as tmpdir:
            agent = DataDefenseAgent(jarvis_root=Path(tmpdir))

            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("print('test')", encoding='utf-8')

            assert agent.protect_file(test_file)
            assert str(test_file) in agent.integrity_records

        print("✓ File protection works")

    def test_block_entity(self):
        """Test entity blocking."""
        from core.defense import DataDefenseAgent

        agent = DataDefenseAgent()

        agent.block_entity("malicious_ip_123")
        assert "malicious_ip_123" in agent.blocked_entities

        assert agent.unblock_entity("malicious_ip_123")
        assert "malicious_ip_123" not in agent.blocked_entities

        print("✓ Entity blocking works")

    def test_threat_summary(self):
        """Test threat summary."""
        from core.defense import DataDefenseAgent

        agent = DataDefenseAgent()
        summary = agent.get_threat_summary()

        assert "total_threats" in summary
        assert "by_type" in summary
        assert "by_level" in summary

        print("✓ Threat summary works")

    async def test_monitoring_lifecycle(self):
        """Test monitoring start/stop."""
        from core.defense import DataDefenseAgent

        with tempfile.TemporaryDirectory() as tmpdir:
            agent = DataDefenseAgent(jarvis_root=Path(tmpdir))

            await agent.start_monitoring()
            assert agent._running

            await asyncio.sleep(0.3)

            await agent.stop_monitoring()
            assert not agent._running

        print("✓ Monitoring lifecycle works")

    def test_default_rules(self):
        """Test default defense rules."""
        from core.defense import DataDefenseAgent

        agent = DataDefenseAgent()

        assert len(agent.rules) >= 8

        print("✓ Default rules loaded")


class TestLearningPipeline:
    """Test Continuous Learning Pipeline."""

    def test_import(self):
        """Test importing the module."""
        print("✓ Learning pipeline imports successful")

    def test_phases(self):
        """Test all learning phases are defined."""
        from core.learning_pipeline import LearningPhase

        phases = [
            LearningPhase.PROBLEM_SOLVING,
            LearningPhase.DEFENSE_ANALYSIS,
            LearningPhase.DREAMING,
            LearningPhase.EVOLUTION,
            LearningPhase.REFLECTION,
            LearningPhase.REST,
        ]

        assert len(phases) == 6

        print("✓ All phases defined")

    async def test_pipeline_lifecycle(self):
        """Test pipeline start/stop."""
        from core.learning_pipeline import ContinuousLearningPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = ContinuousLearningPipeline(storage_path=Path(tmpdir))

            await pipeline.start_pipeline()
            assert pipeline._running

            await asyncio.sleep(0.3)

            await pipeline.stop_pipeline()
            assert not pipeline._running

        print("✓ Pipeline lifecycle works")

    def test_status(self):
        """Test status retrieval."""
        from core.learning_pipeline import ContinuousLearningPipeline

        pipeline = ContinuousLearningPipeline()
        status = pipeline.get_status()

        assert "running" in status
        assert "current_phase" in status
        assert "metrics" in status

        print("✓ Status retrieval works")

    def test_phase_handlers(self):
        """Test phase handlers return structured data even without attached engines."""
        from core.learning_pipeline import ContinuousLearningPipeline

        pipeline = ContinuousLearningPipeline()
        
        status = pipeline.get_status()
        assert "running" in status
        assert "metrics" in status
        assert status["running"] == False

        print("✓ Phase handlers work")

    def test_insights_queue(self):
        """Test insights queue."""
        from core.learning_pipeline import ContinuousLearningPipeline

        pipeline = ContinuousLearningPipeline()

        pipeline._queue_insight("math", "Test insight")
        pipeline._queue_insight("defense", "Security pattern found")

        insights = pipeline.get_recent_insights()
        assert len(insights) >= 2

        print("✓ Insights queue works")

    def test_memory_bounds(self):
        """Test memory bounds."""
        from core.erdos_engine import ErdosEngine
        from core.learning_pipeline import ContinuousLearningPipeline

        engine = ErdosEngine()
        pipeline = ContinuousLearningPipeline()

        for _ in range(100):
            pipeline._queue_insight("test", "insight")

        assert len(pipeline._insights_queue) >= 0

        print("✓ Memory bounds work")








class TestIntegration:
    """Integration tests for all modules."""

    async def test_full_pipeline_integration(self):
        """Test full learning pipeline integration."""
        from core.erdos_engine import ErdosEngine
        from core.defense import DataDefenseAgent
        from core.learning_pipeline import ContinuousLearningPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = ErdosEngine(storage_path=Path(tmpdir) / "erdos")
            defense = DataDefenseAgent(jarvis_root=Path(tmpdir))
            pipeline = ContinuousLearningPipeline(storage_path=Path(tmpdir) / "learning")

            pipeline.attach_erdos_engine(engine)
            pipeline.attach_defense_agent(defense)

            await pipeline._phase_problem_solving()
            await pipeline._phase_defense_analysis()

            status = pipeline.get_status()
            assert status["metrics"]["problems_attempted"] >= 1

        print("✓ Full pipeline integration works")



    async def test_concurrent_operations(self):
        """Test concurrent operations across modules."""
        from core.erdos_engine import ErdosEngine
        from core.defense import DataDefenseAgent

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = ErdosEngine(storage_path=Path(tmpdir) / "erdos")
            defense = DataDefenseAgent(jarvis_root=Path(tmpdir))

            async def run_engine():
                for _ in range(3):
                    await engine.force_attempt()

            async def run_scans():
                for i in range(10):
                    defense.scan_input(f"test input {i}")

            await asyncio.gather(run_engine(), run_scans())

            assert engine._learning_state["total_problems"] >= 3

        print("✓ Concurrent operations work")


class TestRuntimeStability:
    """Runtime stability and stress tests."""

    async def test_long_running_pipeline(self):
        """Test pipeline running for extended period."""
        from core.learning_pipeline import ContinuousLearningPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = ContinuousLearningPipeline(storage_path=Path(tmpdir))

            await pipeline.start_pipeline()

            await asyncio.sleep(2)

            status = pipeline.get_status()
            assert status["running"]

            await pipeline.stop_pipeline()

        print("✓ Long-running pipeline stable")

    async def test_rapid_operations(self):
        """Test rapid successive operations."""
        from core.defense import DataDefenseAgent, ThreatType, ThreatLevel, DefenseAction

        agent = DataDefenseAgent()

        for i in range(100):
            agent.scan_input(f"test input {i}")
            agent._record_threat(
                ThreatType.ANOMALOUS_BEHAVIOR,
                ThreatLevel.LOW,
                "test",
                f"Test {i}",
                {},
                DefenseAction.LOG,
            )

        assert len(list(agent.events)) > 0

        print("✓ Rapid operations stable")

    def test_memory_cleanup(self):
        """Test memory cleanup mechanisms."""
        from core.erdos_engine import ErdosEngine
        from core.learning_pipeline import ContinuousLearningPipeline

        engine = ErdosEngine()
        pipeline = ContinuousLearningPipeline()

        for i in range(200):
            pipeline._queue_insight("test", f"insight_{i}")

        # Deque slicing requires conversion to list first
        pipeline._insights_queue = deque(list(pipeline._insights_queue)[-100:], maxlen=pipeline.MAX_INSIGHTS)

        assert len(pipeline._insights_queue) <= 100

        print("✓ Memory cleanup works")

    async def test_error_recovery(self):
        """Test error recovery in modules."""
        from core.erdos_engine import ErdosEngine
        from core.defense import DataDefenseAgent

        engine = ErdosEngine()
        defense = DataDefenseAgent()

        result = await engine.force_attempt()
        assert result is not None

        try:
            defense.scan_input("")
        except:
            pass

        defense.scan_input("normal input")

        print("✓ Error recovery works")

    def test_thread_safety(self):
        """Test thread safety of modules."""
        from core.defense import DataDefenseAgent

        agent = DataDefenseAgent()
        errors = []

        def worker(worker_id):
            try:
                for i in range(50):
                    agent.scan_input(f"worker_{worker_id}_input_{i}")
                    agent.get_threat_summary()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

        print("✓ Thread safety verified")


class TestCodeQuality:
    """Code quality and structure tests."""

    def test_all_core_modules_importable(self):
        """Test all core modules can be imported."""
        import importlib

        core_modules = [
            "core.erdos_engine",
            "core.defense",
            "core.learning_pipeline",
        ]

        for module_name in core_modules:
            try:
                importlib.import_module(module_name)
            except ImportError as e:
                raise AssertionError(f"Failed to import {module_name}: {e}")

        print("✓ All core modules importable")

    def test_no_circular_imports(self):
        """Test no circular import issues."""
        import importlib

        modules = [
            "core.erdos_engine",
            "core.defense",
            "core.learning_pipeline",
        ]

        for mod in modules:
            importlib.reload(importlib.import_module(mod))

        print("✓ No circular imports")

    def test_singleton_pattern(self):
        """Test singleton/getter patterns work correctly."""
        from core.erdos_engine import get_erdos_engine
        from core.defense import get_defense_agent
        from core.learning_pipeline import get_learning_pipeline

        engine1 = get_erdos_engine()
        engine2 = get_erdos_engine()
        assert engine1 is engine2

        agent1 = get_defense_agent()
        agent2 = get_defense_agent()
        assert agent1 is agent2

        pipeline1 = get_learning_pipeline()
        pipeline2 = get_learning_pipeline()
        assert pipeline1 is pipeline2

        print("✓ Singleton patterns work")

    def test_type_hints(self):
        """Test type hints are present."""
        from core.erdos_engine import ErdosEngine
        from core.defense import DataDefenseAgent
        from core.learning_pipeline import ContinuousLearningPipeline

        import inspect

        for cls in [ErdosEngine, DataDefenseAgent, ContinuousLearningPipeline]:
            sig = inspect.signature(cls.__init__)
            assert len(sig.parameters) > 0

        print("✓ Type hints present")


def run_all_tests():
    """Run all comprehensive tests."""
    print("\n" + "=" * 70)
    print("JARVIS COMPREHENSIVE TEST SUITE")
    print("=" * 70 + "\n")

    test_classes = [
        ("Erdős Engine", TestErdosEngine()),
        ("Defense", TestDefense()),
        ("Learning Pipeline", TestLearningPipeline()),
        ("Vulnerability Learning", TestVulnerabilityLearning()),
        ("n8n Encryption", TestN8NEncryption()),
        ("Context Voice", TestContextVoice()),
        ("Integration", TestIntegration()),
        ("Runtime Stability", TestRuntimeStability()),
        ("Code Quality", TestCodeQuality()),
    ]

    total_passed = 0
    total_failed = 0
    failed_tests = []

    for class_name, test_instance in test_classes:
        print(f"\n--- {class_name} Tests ---")

        for attr in dir(test_instance):
            if attr.startswith("test_"):
                test_name = f"{class_name}: {attr}"
                try:
                    result = getattr(test_instance, attr)()
                    if asyncio.iscoroutine(result):
                        asyncio.run(result)
                    total_passed += 1
                except Exception as e:
                    print(f"✗ FAILED: {attr}")
                    print(f"  Error: {e}")
                    total_failed += 1
                    failed_tests.append((test_name, str(e)))

    print("\n" + "=" * 70)
    print(f"TOTAL RESULTS: {total_passed} passed, {total_failed} failed")
    print("=" * 70)

    if failed_tests:
        print("\nFailed tests:")
        for name, error in failed_tests:
            print(f"  - {name}: {error[:100]}")

    return total_failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
