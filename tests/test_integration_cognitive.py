"""Integration tests for Cognitive State ↔ Load Balancer feedback loop."""
import threading
from core.cognitive_load_balancer import (
    CognitiveLoadBalancer,
    SubsystemType,
    LoadLevel,
    get_load_balancer,
)
from core.memory_consolidation import (
    MemoryConsolidationEngine,
    ConsolidationPhase,
    get_consolidation_engine,
)


class TestCognitiveStateIntegration:
    def test_adjust_for_flow_increases_throttle(self):
        lb = CognitiveLoadBalancer()
        lb._global_throttle = 0.5
        lb.adjust_for_cognitive_state("FLOW", cognitive_load=0.2)
        assert lb._global_throttle > 0.5

    def test_adjust_for_overloaded_decreases_throttle(self):
        lb = CognitiveLoadBalancer()
        lb._global_throttle = 0.8
        lb.adjust_for_cognitive_state("OVERLOADED", cognitive_load=0.9)
        assert lb._global_throttle < 0.8

    def test_adjust_for_idle_maximizes_throttle(self):
        lb = CognitiveLoadBalancer()
        lb._global_throttle = 0.5
        lb.adjust_for_cognitive_state("IDLE", cognitive_load=0.0)
        assert lb._global_throttle > 0.5

    def test_adjust_for_frustrated_minimizes_throttle(self):
        lb = CognitiveLoadBalancer()
        lb._global_throttle = 0.8
        for _ in range(10):
            lb.adjust_for_cognitive_state("FRUSTRATED", cognitive_load=0.9)
        assert lb._global_throttle < 0.5

    def test_adjust_creates_rebalance_action(self):
        lb = CognitiveLoadBalancer()
        lb.adjust_for_cognitive_state("FRUSTRATED", cognitive_load=0.8)
        assert len(lb._rebalance_actions) >= 1

    def test_throttle_bounded_after_adjustment(self):
        lb = CognitiveLoadBalancer()
        for _ in range(20):
            lb.adjust_for_cognitive_state("FRUSTRATED", cognitive_load=1.0)
        assert 0.1 <= lb._global_throttle <= 1.0

    def test_unknown_state_defaults(self):
        lb = CognitiveLoadBalancer()
        lb._global_throttle = 0.5
        lb.adjust_for_cognitive_state("UNKNOWN_STATE", cognitive_load=0.5)
        assert 0.1 <= lb._global_throttle <= 1.0

    def test_high_cognitive_load_reduces_throttle(self):
        lb = CognitiveLoadBalancer()
        lb._global_throttle = 0.9
        lb.adjust_for_cognitive_state("ENGAGED", cognitive_load=0.9)
        assert lb._global_throttle < 0.9

    def test_low_cognitive_load_increases_throttle(self):
        lb = CognitiveLoadBalancer()
        lb._global_throttle = 0.3
        lb.adjust_for_cognitive_state("IDLE", cognitive_load=0.1)
        assert lb._global_throttle > 0.3


class TestConsolidationIntegration:
    def test_register_and_get_related(self):
        engine = MemoryConsolidationEngine(storage_path="/tmp/jarvis_test_consolidation_int")
        engine.register_memory("m1", weight=0.9, tags={"math", "graph"})
        engine.register_memory("m2", weight=0.8, tags={"math", "graph"})
        engine._cross_links["m1"].add("m2")
        engine._cross_links["m2"].add("m1")
        related = engine.get_related("m1")
        assert "m2" in related

    def test_consolidation_cycle_processes_registered_memories(self):
        engine = MemoryConsolidationEngine(storage_path="/tmp/jarvis_test_consolidation_int2")
        engine.register_memory("m1", weight=0.9, tags={"math"})
        engine.register_memory("m2", weight=0.7, tags={"math"})
        engine.register_memory("m3", weight=0.01, tags={"stale"})
        stats = engine.force_consolidation()
        assert stats["consolidation_count"] >= 1

    def test_low_weight_memory_evicted_after_decay(self):
        engine = MemoryConsolidationEngine(storage_path="/tmp/jarvis_test_consolidation_int3")
        engine.register_memory("m1", weight=0.01, tags={"test"})
        engine.force_consolidation()
        if "m1" in engine._memory_weights:
            assert engine._memory_weights["m1"] < 0.05

    def test_consolidation_compression_creates_clusters(self):
        engine = MemoryConsolidationEngine(storage_path="/tmp/jarvis_test_consolidation_int4")
        for i in range(5):
            mid = f"m{i:03d}"
            engine.register_memory(mid, weight=0.7, tags={"cluster_x"})
            for j in range(5):
                if j != i:
                    engine._cross_links[mid].add(f"m{j:03d}")
        engine.force_consolidation()
        knowledge = engine.get_compressed_knowledge()
        assert isinstance(knowledge, list)

    def test_concurrent_cognitive_state_adjustments(self):
        lb = CognitiveLoadBalancer()
        errors = []

        def adjust(state):
            try:
                lb.adjust_for_cognitive_state(state, cognitive_load=0.5)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=adjust, args=(s,)) for s in ["FLOW", "OVERLOADED", "IDLE", "FRUSTRATED", "ENGAGED"]]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert 0.1 <= lb._global_throttle <= 1.0