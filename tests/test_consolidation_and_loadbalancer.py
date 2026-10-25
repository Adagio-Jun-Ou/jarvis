"""Tests for Memory Consolidation Engine and Cognitive Load Balancer."""
import time
import threading
import pytest

from core.memory_consolidation import (
    MemoryConsolidationEngine,
    ConsolidationPhase,
    ConsolidationReport,
    get_consolidation_engine,
)
from core.cognitive_load_balancer import (
    CognitiveLoadBalancer,
    SubsystemType,
    LoadLevel,
    SubsystemLoad,
    get_load_balancer,
)


class TestMemoryConsolidationEngine:
    def test_initial_state(self):
        engine = MemoryConsolidationEngine(storage_path="/tmp/jarvis_test_consolidation")
        assert not engine._running
        assert engine._consolidation_count == 0

    def test_register_memory(self):
        engine = MemoryConsolidationEngine(storage_path="/tmp/jarvis_test_consolidation")
        engine.register_memory("mem_001", weight=0.8, tags={"math", "ramsey"})
        assert "mem_001" in engine._memory_weights
        assert engine._memory_weights["mem_001"] == 0.8
        assert "mem_001" in engine._tag_index.get("math", set())
        assert "mem_001" in engine._tag_index.get("ramsey", set())

    def test_register_memory_updates_max_weight(self):
        engine = MemoryConsolidationEngine(storage_path="/tmp/jarvis_test_consolidation")
        engine.register_memory("mem_001", weight=0.3)
        engine.register_memory("mem_001", weight=0.7)
        assert engine._memory_weights["mem_001"] == 0.7

    def test_register_memory_keeps_higher_weight(self):
        engine = MemoryConsolidationEngine(storage_path="/tmp/jarvis_test_consolidation")
        engine.register_memory("mem_001", weight=0.8)
        engine.register_memory("mem_001", weight=0.5)
        assert engine._memory_weights["mem_001"] == 0.8

    def test_record_access(self):
        engine = MemoryConsolidationEngine(storage_path="/tmp/jarvis_test_consolidation")
        engine.register_memory("mem_001", weight=0.5)
        initial_weight = engine._memory_weights["mem_001"]
        engine.record_access("mem_001")
        assert engine._memory_weights["mem_001"] > initial_weight

    def test_record_access_unknown_memory(self):
        engine = MemoryConsolidationEngine(storage_path="/tmp/jarvis_test_consolidation")
        engine.record_access("unknown")
        assert engine._access_counts["unknown"] == 1

    def test_replay_phase(self):
        engine = MemoryConsolidationEngine(storage_path="/tmp/jarvis_test_consolidation")
        engine.register_memory("mem_001", weight=0.9)
        engine.register_memory("mem_002", weight=0.7)
        engine.register_memory("mem_003", weight=0.3)

        report = engine._phase_replay()
        assert report.phase == ConsolidationPhase.REPLAY
        assert report.memories_strengthened > 0

    def test_decay_phase(self):
        engine = MemoryConsolidationEngine(storage_path="/tmp/jarvis_test_consolidation")
        engine.register_memory("mem_001", weight=0.01, tags={"test"})
        engine.register_memory("mem_002", weight=0.9)

        report = engine._phase_decay()
        assert report.memories_decayed >= 1

    def test_cross_link_phase(self):
        engine = MemoryConsolidationEngine(storage_path="/tmp/jarvis_test_consolidation")
        engine.register_memory("mem_001", weight=0.8, tags={"math", "ramsey"})
        engine.register_memory("mem_002", weight=0.7, tags={"math", "ramsey"})
        engine.register_memory("mem_003", weight=0.6, tags={"math", "graph"})

        report = engine._phase_cross_link()
        assert report.phase == ConsolidationPhase.CROSS_LINK

    def test_compress_phase(self):
        engine = MemoryConsolidationEngine(storage_path="/tmp/jarvis_test_consolidation")
        for i in range(5):
            mid = f"mem_{i:03d}"
            engine.register_memory(mid, weight=0.7, tags={"cluster_a"})
            for j in range(5):
                if j != i:
                    engine._cross_links[mid].add(f"mem_{j:03d}")

        report = engine._phase_compress()
        assert report.phase == ConsolidationPhase.COMPRESS

    def test_reindex_phase(self):
        engine = MemoryConsolidationEngine(storage_path="/tmp/jarvis_test_consolidation")
        engine.register_memory("mem_001", weight=0.8, tags={"math"})
        engine.register_memory("mem_002", weight=0.01, tags={"stale"})

        report = engine._phase_reindex()
        assert report.phase == ConsolidationPhase.REINDEX

    def test_force_consolidation(self):
        engine = MemoryConsolidationEngine(storage_path="/tmp/jarvis_test_consolidation")
        engine.register_memory("mem_001", weight=0.8, tags={"test"})

        stats = engine.force_consolidation()
        assert stats["consolidation_count"] >= 1
        assert "recent_reports" in stats

    def test_get_related(self):
        engine = MemoryConsolidationEngine(storage_path="/tmp/jarvis_test_consolidation")
        engine.register_memory("mem_001", weight=0.8, tags={"math"})
        engine.register_memory("mem_002", weight=0.7, tags={"math"})
        engine._cross_links["mem_001"].add("mem_002")
        engine._cross_links["mem_002"].add("mem_001")

        related = engine.get_related("mem_001")
        assert "mem_002" in related

    def test_get_compressed_knowledge(self):
        engine = MemoryConsolidationEngine(storage_path="/tmp/jarvis_test_consolidation")
        engine._compressed_facts["c1"] = {"member_count": 5}
        knowledge = engine.get_compressed_knowledge()
        assert len(knowledge) >= 1

    def test_consolidation_stats(self):
        engine = MemoryConsolidationEngine(storage_path="/tmp/jarvis_test_consolidation")
        stats = engine.get_consolidation_stats()
        assert "running" in stats
        assert "tracked_memories" in stats
        assert "avg_weight" in stats

    def test_notify_activity(self):
        engine = MemoryConsolidationEngine(storage_path="/tmp/jarvis_test_consolidation")
        old_activity = engine._last_activity
        time.sleep(0.01)
        engine.notify_activity()
        assert engine._last_activity > old_activity

    def test_start_stop(self):
        engine = MemoryConsolidationEngine(storage_path="/tmp/jarvis_test_consolidation")
        engine._CONSOLIDATION_INTERVAL = 3600
        engine.start()
        assert engine._running
        engine.stop()
        assert not engine._running

    def test_singleton(self):
        engine = get_consolidation_engine()
        assert engine is not None
        assert isinstance(engine, MemoryConsolidationEngine)


class TestSubsystemLoad:
    def test_load_score_empty(self):
        load = SubsystemLoad(subsystem=SubsystemType.ERDOS)
        assert load.load_score == 0.0

    def test_load_score_with_tasks(self):
        load = SubsystemLoad(subsystem=SubsystemType.PFC, active_tasks=3, pending_tasks=2)
        score = load.load_score
        assert score > 0

    def test_load_level_idle(self):
        load = SubsystemLoad(subsystem=SubsystemType.ERDOS)
        assert load.load_level == LoadLevel.IDLE

    def test_load_level_critical(self):
        load = SubsystemLoad(
            subsystem=SubsystemType.ERDOS,
            active_tasks=10,
            pending_tasks=10,
            cpu_estimate=100,
            memory_estimate_mb=500,
        )
        assert load.load_level == LoadLevel.CRITICAL

    def test_load_level_moderate(self):
        load = SubsystemLoad(
            subsystem=SubsystemType.ERDOS,
            active_tasks=2,
            pending_tasks=1,
            cpu_estimate=30,
            memory_estimate_mb=100,
        )
        assert load.load_level in (LoadLevel.LIGHT, LoadLevel.MODERATE)


class TestCognitiveLoadBalancer:
    def test_initial_state(self):
        lb = CognitiveLoadBalancer()
        assert not lb._running
        assert lb._global_throttle == 1.0

    def test_all_subsystems_initialized(self):
        lb = CognitiveLoadBalancer()
        for st in SubsystemType:
            assert st in lb._subsystem_loads

    def test_notify_task_start(self):
        lb = CognitiveLoadBalancer()
        initial = lb._subsystem_loads[SubsystemType.ERDOS].active_tasks
        lb.notify_task_start(SubsystemType.ERDOS, task_cost=1.0)
        assert lb._subsystem_loads[SubsystemType.ERDOS].active_tasks == initial + 1

    def test_notify_task_complete(self):
        lb = CognitiveLoadBalancer()
        lb.notify_task_start(SubsystemType.ERDOS)
        lb.notify_task_complete(SubsystemType.ERDOS, duration=0.5)
        assert lb._subsystem_loads[SubsystemType.ERDOS].completed_tasks >= 1

    def test_notify_task_fail(self):
        lb = CognitiveLoadBalancer()
        lb.notify_task_start(SubsystemType.ERDOS)
        initial = lb._subsystem_loads[SubsystemType.ERDOS].active_tasks
        lb.notify_task_fail(SubsystemType.ERDOS)
        assert lb._subsystem_loads[SubsystemType.ERDOS].active_tasks == max(0, initial - 1)

    def test_should_run_task_idle(self):
        lb = CognitiveLoadBalancer()
        assert lb.should_run_task(SubsystemType.ERDOS)

    def test_should_run_task_overloaded(self):
        lb = CognitiveLoadBalancer()
        load = lb._subsystem_loads[SubsystemType.ERDOS]
        load.active_tasks = 50
        load.cpu_estimate = 200
        load.memory_estimate_mb = 1000
        assert not lb.should_run_task(SubsystemType.ERDOS)

    def test_get_subsystem_throttle(self):
        lb = CognitiveLoadBalancer()
        throttle = lb.get_subsystem_throttle(SubsystemType.PFC)
        assert 0 <= throttle <= 1.0

    def test_get_optimal_batch_size_idle(self):
        lb = CognitiveLoadBalancer()
        batch = lb.get_optimal_batch_size(SubsystemType.ERDOS, max_batch=20)
        assert batch >= 1

    def test_get_optimal_batch_size_heavy(self):
        lb = CognitiveLoadBalancer()
        load = lb._subsystem_loads[SubsystemType.ERDOS]
        load.active_tasks = 50
        load.cpu_estimate = 200
        batch = lb.get_optimal_batch_size(SubsystemType.ERDOS, max_batch=20)
        assert batch >= 0

    def test_get_load_summary(self):
        lb = CognitiveLoadBalancer()
        summary = lb.get_load_summary()
        assert "running" in summary
        assert "global_throttle" in summary
        assert "subsystems" in summary
        assert len(summary["subsystems"]) == len(SubsystemType)

    def test_rebalance_history(self):
        lb = CognitiveLoadBalancer()
        history = lb.get_rebalance_history()
        assert isinstance(history, list)

    def test_start_stop(self):
        lb = CognitiveLoadBalancer()
        lb._CHECK_INTERVAL = 3600
        lb.start()
        assert lb._running
        lb.stop()
        assert not lb._running

    def test_singleton(self):
        lb = get_load_balancer()
        assert lb is not None
        assert isinstance(lb, CognitiveLoadBalancer)

    def test_snapshot_recording(self):
        lb = CognitiveLoadBalancer()
        lb._record_snapshot()
        assert len(lb._load_history) == 1

    def test_multiple_snapshots(self):
        lb = CognitiveLoadBalancer()
        for _ in range(10):
            lb._record_snapshot()
        assert len(lb._load_history) == 10


class TestConsolidationThreadSafety:
    def test_concurrent_registration(self):
        engine = MemoryConsolidationEngine(storage_path="/tmp/jarvis_test_consolidation")
        errors = []

        def register(idx):
            try:
                engine.register_memory(f"mem_{idx}", weight=0.5 + (idx % 5) * 0.1, tags={f"tag_{idx % 3}"})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=register, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(engine._memory_weights) == 50


class TestLoadBalancerThreadSafety:
    def test_concurrent_task_notifications(self):
        lb = CognitiveLoadBalancer()
        errors = []

        def notify(idx):
            try:
                subsystem = list(SubsystemType)[idx % len(SubsystemType)]
                lb.notify_task_start(subsystem)
                lb.notify_task_complete(subsystem, duration=0.1)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=notify, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors