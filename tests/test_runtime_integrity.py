"""Runtime integrity and stability tests for Jarvis."""
import asyncio
import sys
import tempfile
import threading
import time
import tracemalloc
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestMemoryIntegrity:
    """Test memory integrity across modules."""

    def test_erdos_memory_bounds_enforced(self):
        """Test that Erdos engine enforces memory bounds."""
        from core.erdos_engine import ErdosEngine

        engine = ErdosEngine()

        for _ in range(300):
            asyncio.run(engine.force_attempt())

        assert len(engine.problems) <= engine.MAX_PROBLEMS, \
            f"Problems {len(engine.problems)} exceeds max {engine.MAX_PROBLEMS}"
        assert len(engine.history) <= engine.MAX_HISTORY, \
            f"History {len(engine.history)} exceeds max {engine.MAX_HISTORY}"

        print(f"✓ Erdos memory bounds enforced: {len(engine.problems)} problems, {len(engine.history)} history")

    def test_defense_event_bounds(self):
        """Test that defense agent enforces event bounds."""
        from core.defense import DataDefenseAgent, ThreatType, ThreatLevel, DefenseAction

        agent = DataDefenseAgent()

        for i in range(2000):
            agent._record_threat(
                ThreatType.ANOMALOUS_BEHAVIOR,
                ThreatLevel.LOW,
                "test",
                f"Test {i}",
                {},
                DefenseAction.LOG,
            )

        assert len(list(agent.events)) <= agent.MAX_EVENTS, \
            f"Events {len(list(agent.events))} exceeds max {agent.MAX_EVENTS}"

        print(f"✓ Defense event bounds enforced: {len(list(agent.events))} events")

    def test_learning_pipeline_bounds(self):
        """Test that learning pipeline enforces bounds."""
        from core.learning_pipeline import ContinuousLearningPipeline

        pipeline = ContinuousLearningPipeline()

        for _ in range(200):
            asyncio.run(pipeline._phase_problem_solving())

        assert len(pipeline.sessions) <= pipeline.MAX_SESSIONS, \
            f"Sessions {len(pipeline.sessions)} exceeds max {pipeline.MAX_SESSIONS}"

        print(f"✓ Pipeline bounds enforced: {len(pipeline.sessions)} sessions")


    def test_no_memory_leak_under_load(self):
        """Test for memory leaks under sustained load."""
        from core.erdos_engine import ErdosEngine
        from core.defense import DataDefenseAgent

        tracemalloc.start()

        engine = ErdosEngine()
        agent = DataDefenseAgent()

        baseline = tracemalloc.get_traced_memory()[0]

        for _ in range(100):
            asyncio.run(engine.force_attempt())
            agent.scan_input("test input")

        after_load = tracemalloc.get_traced_memory()[0]
        growth = (after_load - baseline) / 1024  # KB

        # Allow up to 5MB growth for 100 ops (LSTM models and caches may allocate memory)
        assert growth < 5000, f"Memory grew by {growth:.1f} KB (potential leak)"

        tracemalloc.stop()
        print(f"✓ No memory leak: {growth:.1f} KB growth after 100 ops")


class TestThreadSafety:
    """Test thread safety of modules."""

    def test_concurrent_erdos_operations(self):
        """Test concurrent Erdos operations."""
        from core.erdos_engine import ErdosEngine

        engine = ErdosEngine()
        errors = []

        def worker(worker_id):
            try:
                for _ in range(20):
                    asyncio.run(engine.force_attempt())
            except Exception as e:
                errors.append((worker_id, str(e)))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"
        print("✓ Concurrent Erdos operations: 5 threads x 20 ops = 100 total")

    def test_concurrent_defense_operations(self):
        """Test concurrent defense operations."""
        from core.defense import DataDefenseAgent

        agent = DataDefenseAgent()
        errors = []

        def worker(worker_id):
            try:
                for i in range(100):
                    agent.scan_input(f"worker_{worker_id}_input_{i}")
                    agent.get_threat_summary()
            except Exception as e:
                errors.append((worker_id, str(e)))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"
        print("✓ Concurrent defense operations: 10 threads x 200 ops = 2000 total")

    def test_concurrent_pipeline_operations(self):
        """Test concurrent pipeline operations."""
        from core.learning_pipeline import ContinuousLearningPipeline

        pipeline = ContinuousLearningPipeline()
        errors = []

        def worker(worker_id):
            try:
                for _ in range(10):
                    asyncio.run(pipeline._phase_problem_solving())
            except Exception as e:
                errors.append((worker_id, str(e)))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"
        print("✓ Concurrent pipeline operations: 5 threads x 10 ops = 50 total")


class TestAsyncCancellation:
    """Test proper async cancellation handling."""

    async def test_erdos_cancellation(self):
        """Test Erdos engine handles cancellation."""
        from core.erdos_engine import ErdosEngine

        engine = ErdosEngine()

        await engine.start_background_learning()
        assert engine._running

        await asyncio.sleep(0.5)

        await engine.stop_background_learning()
        assert not engine._running

        print("✓ Erdos cancellation handled cleanly")

    async def test_defense_cancellation(self):
        """Test defense agent handles cancellation."""
        from core.defense import DataDefenseAgent

        with tempfile.TemporaryDirectory() as tmpdir:
            agent = DataDefenseAgent(jarvis_root=Path(tmpdir))

            await agent.start_monitoring()
            assert agent._running

            await asyncio.sleep(0.3)

            await agent.stop_monitoring()
            assert not agent._running

        print("✓ Defense cancellation handled cleanly")

    async def test_pipeline_cancellation(self):
        """Test learning pipeline handles cancellation."""
        from core.learning_pipeline import ContinuousLearningPipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = ContinuousLearningPipeline(storage_path=Path(tmpdir))

            await pipeline.start_pipeline()
            assert pipeline._running

            await asyncio.sleep(0.3)

            await pipeline.stop_pipeline()
            assert not pipeline._running

        print("✓ Pipeline cancellation handled cleanly")


class TestErrorRecovery:
    """Test error recovery across modules."""

    def test_erdos_error_recovery(self):
        """Test Erdos engine recovers from errors."""
        from core.erdos_engine import ErdosEngine

        engine = ErdosEngine()

        result = asyncio.run(engine.force_attempt())
        assert result is not None

        result = asyncio.run(engine.force_attempt(problem_id="nonexistent"))
        assert result is not None

        print("✓ Erdos error recovery works")

    def test_defense_error_recovery(self):
        """Test defense agent handles malformed input."""
        from core.defense import DataDefenseAgent

        agent = DataDefenseAgent()

        agent.scan_input("")
        agent.scan_input("x" * 10000)
        agent.scan_input("null\x00bytes")

        assert agent.get_defense_status() is not None

        print("✓ Defense error recovery works")




class TestPerformanceBaseline:
    """Test performance baselines."""

def run_all_tests():
    """Run all runtime integrity tests."""
    print("\n" + "=" * 70)
    print("JARVIS RUNTIME INTEGRITY TESTS")
    print("=" * 70 + "\n")

    test_classes = [
        ("Memory Integrity", TestMemoryIntegrity()),
        ("Thread Safety", TestThreadSafety()),
        ("Async Cancellation", TestAsyncCancellation()),
        ("Error Recovery", TestErrorRecovery()),
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
