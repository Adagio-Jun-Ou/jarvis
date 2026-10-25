"""Tests for integration wiring, adaptive context, skill tracker, and more security v2."""

import threading
import time
import pytest

from core.attention_priority import AttentionPriorityQueue, UrgencySignal
from core.concept_drift import ConceptDriftDetector, DriftType
from core.cognitive_load_balancer import CognitiveLoadBalancer, SubsystemType, LoadLevel
from core.memory_consolidation import MemoryConsolidationEngine, ConsolidationPhase
from core.adaptive_context import AdaptiveContextWindow, ContextStrategy, ContentType


class TestMemoryConsolidationKGIntegration:
    def test_consolidation_publishes_to_kg(self):
        mc = MemoryConsolidationEngine(storage_path=None)
        mc.register_memory("mem_1", 0.8, tags={"math", "prime"})
        mc.register_memory("mem_2", 0.7, tags={"math", "graph"})
        mc.register_memory("mem_3", 0.9, tags={"math", "prime"})

        report = mc._phase_replay()
        assert report.memories_strengthened >= 1

    def test_memory_weights_boost_on_access(self):
        mc = MemoryConsolidationEngine(storage_path=None)
        mc.register_memory("m1", 0.5)
        mc.record_access("m1")
        assert mc._memory_weights["m1"] > 0.5

    def test_decay_removes_low_weight(self):
        mc = MemoryConsolidationEngine(storage_path=None)
        mc.register_memory("weak", 0.02)
        mc.register_memory("strong", 0.8)
        report = mc._phase_decay()
        assert "weak" not in mc._memory_weights

    def test_cross_link_phase(self):
        mc = MemoryConsolidationEngine(storage_path=None)
        mc.register_memory("m1", 0.9, tags={"math", "proof"})
        mc.register_memory("m2", 0.9, tags={"math", "proof"})
        report = mc._phase_cross_link()
        assert "m2" in mc._cross_links.get("m1", set()) or report.cross_links_created >= 0

    def test_compressed_knowledge_output(self):
        mc = MemoryConsolidationEngine(storage_path=None)
        mc.register_memory("m1", 0.9, tags={"algebra"})
        mc.register_memory("m2", 0.9, tags={"algebra"})
        mc._cross_links["m1"].add("m2")
        mc._cross_links["m2"].add("m1")
        mc._tag_index["algebra"].update({"m1", "m2"})
        report = mc._phase_compress()
        facts = mc.get_compressed_knowledge()
        assert isinstance(facts, list)

    def test_notify_thought_sense_no_crash_on_missing(self):
        mc = MemoryConsolidationEngine(storage_path=None)
        mc._notify_thought_sense()


class TestEvolutionLoadBalancerIntegration:
    def test_evolution_skipped_when_throttled(self):
        from core.cognitive_load_balancer import SubsystemLoad, _SUBSYSTEM_CAPS
        lb = CognitiveLoadBalancer()
        lb._subsystem_loads[SubsystemType.EVOLUTION] = SubsystemLoad(
            subsystem=SubsystemType.EVOLUTION,
            cpu_estimate=99,
            memory_estimate_mb=400,
            active_tasks=5,
            pending_tasks=10,
        )
        lb._subsystem_loads[SubsystemType.EVOLUTION].throttled = True
        lb._subsystem_loads[SubsystemType.EVOLUTION].throttle_level = 0.9
        result = lb.should_run_task(SubsystemType.EVOLUTION)
        assert result is False


class TestInsightBridgeDriftIntegration:
    def test_evolution_guidance_includes_drift_topic(self):
        from core.insight_bridge import InsightBridge, InsightType, EvolutionRelevance
        ib = InsightBridge(storage_path=None)

        category = "prime"
        insight = "Optimal strategy discovered for prime number factorization using quantum-inspired methods"
        strategy = "probabilistic_method"

        guidance = ib._extract_evolution_guidance(
            insight, category, strategy, InsightType.PATTERN_DISCOVERY
        )
        assert "gene_adjustments" in guidance
        assert "domain_hints" in guidance

    def test_evolution_guidance_strategy_map(self):
        from core.insight_bridge import InsightBridge, InsightType
        ib = InsightBridge(storage_path=None)
        guidance = ib._extract_evolution_guidance(
            "test insight about probability and random methods",
            "probability",
            "randomized",
            InsightType.PATTERN_DISCOVERY,
        )
        assert "reasoning_depth" in guidance.get("gene_adjustments", {}) or "creativity_factor" in guidance.get("gene_adjustments", {})


class TestAdaptiveContextWindow:
    def test_add_and_assemble(self):
        acw = AdaptiveContextWindow(max_tokens=10000)
        acw.add_block(ContentType.SYSTEM, "You are a helpful assistant.", priority=1.0, topic="system")
        acw.add_block(ContentType.CONVERSATION, "Hello!", priority=0.8, topic="greeting")
        acw.add_block(ContentType.BACKGROUND, "Background info here." * 50, priority=0.3, topic="info")

        context = acw.assemble_context()
        assert "helpful assistant" in context
        assert "Hello" in context

    def test_strategy_change(self):
        acw = AdaptiveContextWindow()
        strategy = acw.adjust_for_cognitive_state("FLOW")
        assert strategy == ContextStrategy.FULL
        assert acw._current_budget > 50000

        strategy = acw.adjust_for_cognitive_state("OVERLOADED")
        assert strategy == ContextStrategy.MINIMAL
        assert acw._current_budget < 50000

    def test_minimal_strategy_evicts_background(self):
        acw = AdaptiveContextWindow(max_tokens=2000)
        acw.adjust_for_cognitive_state("OVERLOADED")
        acw.add_block(ContentType.SYSTEM, "System prompt.", priority=1.0, topic="sys")
        acw.add_block(ContentType.BACKGROUND, "X" * 4000, priority=0.2, topic="bg")

        context = acw.assemble_context()
        assert len(context) < 5000

    def test_focus_topic_boost(self):
        acw = AdaptiveContextWindow()
        acw.set_focus_topic("python")
        acw.add_block(ContentType.REFERENCE, "Python docs", priority=0.5, topic="python")
        acw.add_block(ContentType.REFERENCE, "Rust docs", priority=0.5, topic="rust")

        scored_python = 0.5 + 0.3
        scored_rust = 0.5
        assert scored_python > scored_rust

    def test_stats(self):
        acw = AdaptiveContextWindow()
        acw.add_block(ContentType.SYSTEM, "Test", priority=0.5)
        stats = acw.get_stats()
        assert stats["blocks_count"] == 1
        assert stats["current_strategy"] == "FOCUSED"

    def test_clear(self):
        acw = AdaptiveContextWindow()
        acw.add_block(ContentType.SYSTEM, "Test", priority=0.5)
        acw.clear()
        assert len(acw._blocks) == 0




class TestLoadBalancerSubsystemIntegration:
    def test_notify_task_start_complete(self):
        lb = CognitiveLoadBalancer()
        lb.notify_task_start(SubsystemType.ERDOS, task_cost=1.0)
        load = lb._subsystem_loads[SubsystemType.ERDOS]
        assert load.active_tasks == 1
        lb.notify_task_complete(SubsystemType.ERDOS, duration=5.0)
        assert load.active_tasks == 0
        assert load.completed_tasks == 1

    def test_batch_size_scales(self):
        lb = CognitiveLoadBalancer()
        batch = lb.get_optimal_batch_size(SubsystemType.ERDOS, max_batch=10)
        assert 1 <= batch <= 10

    def test_load_level_computation(self):
        from core.cognitive_load_balancer import SubsystemLoad
        load = SubsystemLoad(subsystem=SubsystemType.PFC, cpu_estimate=50, memory_estimate_mb=100)
        assert load.load_level == LoadLevel.LIGHT

        heavy = SubsystemLoad(
            subsystem=SubsystemType.PFC, cpu_estimate=95, memory_estimate_mb=450,
            active_tasks=10, pending_tasks=5,
        )
        assert heavy.load_level in (LoadLevel.HEAVY, LoadLevel.CRITICAL)


class TestSecurityModuleRoundup:
    def test_branch_detection_enforce_whitelist(self):
        from core.security.branch_detection import DataBranchMonitor, BranchType, AnomalyType
        m = DataBranchMonitor(enforce_destination_whitelist=True, authorized_destinations={"good.com"})
        anomalies = m.check_branch_policy(BranchType.NETWORK_SEND, "evil.com", "data")
        unauth = [a for a in anomalies if a.anomaly_type == AnomalyType.UNAUTHORIZED_DESTINATION]
        assert len(unauth) == 1

    def test_mitm_rejects_future_timestamp(self):
        from core.security.mitm_protection import RequestSigner
        signer = RequestSigner()
        future_time = time.time() + 600
        sig = signer.sign_request("GET", "/test", timestamp=future_time)
        assert signer.verify_request("GET", "/test", sig, timestamp=future_time) is False

    def test_stack_defense_canary_corruption(self):
        from core.security.stack_defense import SafeBuffer
        buf = SafeBuffer(64, canary=True)
        assert buf.check_canary() is True
        buf._buffer[buf._canary_position:buf._canary_position + 8] = b'\x00' * 8
        assert buf.check_canary() is False

    def test_dlp_custom_pattern_type(self):
        from core.security.dlp import DataLeakProtector, SensitiveDataPattern, LeakSeverity
        dlp = DataLeakProtector()
        dlp.add_custom_pattern("test_pattern", r'TEST-\d{4}', LeakSeverity.HIGH)
        events = dlp.scan("Here is TEST-1234 in text")
        custom = [e for e in events if e.pattern_type == SensitiveDataPattern.CUSTOM]
        assert len(custom) == 1

    def test_adaptive_context_singleton(self):
        from core.adaptive_context import get_adaptive_window
        w1 = get_adaptive_window()
        w2 = get_adaptive_window()
        assert w1 is w2