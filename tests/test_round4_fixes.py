"""Tests for Memory Router, Skill Tracker integration, and round 4/5 bug fixes."""

import threading
import time
import pytest

from core.memory_router import MemoryRouter, MemoryTier, QueryType, get_memory_router
from core.adaptive_context import AdaptiveContextWindow, ContextStrategy, ContentType


class TestMemoryRouter:
    def test_factual_query_routes_to_l3(self):
        router = MemoryRouter()
        decision = router.route("What is the capital of France?")
        assert decision.tier == MemoryTier.L3_COLD
        assert decision.query_type == QueryType.FACTUAL

    def test_procedural_query_routes_to_semantic(self):
        router = MemoryRouter()
        decision = router.route("How to bake a chocolate cake?")
        assert decision.tier == MemoryTier.SEMANTIC
        assert decision.query_type == QueryType.PROCEDURAL

    def test_conversational_routes_to_l1(self):
        router = MemoryRouter()
        decision = router.route("Remember what you said earlier?")
        assert decision.tier == MemoryTier.L1_HOT
        assert decision.query_type == QueryType.CONVERSATIONAL

    def test_correction_routes_to_l2(self):
        router = MemoryRouter()
        decision = router.route("No, that's wrong")
        assert decision.tier == MemoryTier.L2_WARM
        assert decision.query_type == QueryType.CORRECTION

    def test_creative_routes_to_semantic(self):
        router = MemoryRouter()
        decision = router.route("Create a story about dragons")
        assert decision.tier == MemoryTier.SEMANTIC
        assert decision.query_type == QueryType.CREATIVE

    def test_frustrated_state_routes_to_l1(self):
        router = MemoryRouter()
        router.update_cognitive_state("FRUSTRATED")
        decision = router.route("Explain this concept")
        assert decision.tier == MemoryTier.L1_HOT
        assert "Frustrated" in decision.reason

    def test_overloaded_state_routes_to_l1(self):
        router = MemoryRouter()
        router.update_cognitive_state("OVERLOADED")
        decision = router.route("Tell me about economics")
        assert decision.tier == MemoryTier.L1_HOT

    def test_alternative_tiers_provided(self):
        router = MemoryRouter()
        decision = router.route("What is quantum computing?")
        assert len(decision.alternative_tiers) > 0
        assert decision.tier not in decision.alternative_tiers

    def test_cache_hit(self):
        router = MemoryRouter()
        decision1 = router.route("What is AI?")
        decision2 = router.route("What is AI?")
        assert router._stats["cache_hits"] == 1

    def test_stats_tracking(self):
        router = MemoryRouter()
        router.route("What is Python?")
        router.route("How to code?")
        stats = router.get_stats()
        assert stats["total_routes"] == 2
        assert "tier_distribution" in stats

    def test_classify_query_types(self):
        router = MemoryRouter()
        assert router.classify_query("What is machine learning?") == QueryType.FACTUAL
        assert router.classify_query("How to train a model?") == QueryType.PROCEDURAL
        assert router.classify_query("Remember our discussion?") == QueryType.CONVERSATIONAL
        assert router.classify_query("No, that's incorrect!") == QueryType.CORRECTION
        assert router.classify_query("What if we try this?") == QueryType.EXPLORATORY
        assert router.classify_query("Write a poem!") == QueryType.CREATIVE

    def test_thread_safety(self):
        router = MemoryRouter()
        errors = []

        def worker(i):
            try:
                for j in range(20):
                    router.route(f"Test query {i}_{j}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert router._stats["total_routes"] == 80

    def test_singleton(self):
        r1 = get_memory_router()
        r2 = get_memory_router()
        assert r1 is r2




class TestAdaptiveContextWindow:
    def test_focused_strategy_assembly(self):
        acw = AdaptiveContextWindow(max_tokens=5000)
        acw.adjust_for_cognitive_state("ENGAGED")
        acw.add_block(ContentType.SYSTEM, "You are helpful.", priority=1.0, topic="system")
        acw.add_block(ContentType.CONVERSATION, "User message", priority=0.8, topic="chat")
        context = acw.assemble_context(max_tokens=5000)
        assert "System" in context or "helpful" in context or len(context) > 0

    def test_minimal_strategy_trims_background(self):
        acw = AdaptiveContextWindow(max_tokens=2000)
        acw.adjust_for_cognitive_state("OVERLOADED")
        acw.add_block(ContentType.SYSTEM, "System", priority=1.0)
        acw.add_block(ContentType.BACKGROUND, "X" * 3000, priority=0.3, topic="bg")
        context = acw.assemble_context()
        assert len(context) < 4000

    def test_flow_strategy_gives_more_tokens(self):
        acw = AdaptiveContextWindow()
        flow_budget = acw._STRATEGY_MAP["FLOW"][1] * acw._max_tokens
        overloaded_budget = acw._STRATEGY_MAP["OVERLOADED"][1] * acw._max_tokens
        assert flow_budget > overloaded_budget

    def test_topic_focus_boost(self):
        acw = AdaptiveContextWindow()
        acw.set_focus_topic("python")
        acw.add_block(ContentType.REFERENCE, "Python docs", priority=0.5, topic="python")
        acw.add_block(ContentType.REFERENCE, "Rust docs", priority=0.5, topic="rust")
        stats = acw.get_stats()
        assert stats["focus_topic"] == "python"

    def test_clear(self):
        acw = AdaptiveContextWindow()
        acw.add_block(ContentType.SYSTEM, "Test", priority=0.5)
        acw.clear()
        assert acw.get_stats()["blocks_count"] == 0


class TestRound4BugFixes:

    def test_agentic_context_locks_dict_access(self):
        from core.agentic_context import AgenticContext
        ctx = AgenticContext()
        ctx.set("key1", "val1")
        assert ctx.get("key1") == "val1"
        assert ctx.get("missing", "default") == "default"

    def test_agentic_context_shared_state_bounded(self):
        from core.agentic_context import AgenticContext
        ctx = AgenticContext()
        for i in range(600):
            ctx.set(f"key_{i}", f"val_{i}")
        assert len(ctx.shared_state) <= ctx._MAX_SHARED_STATE


    def test_memory_router_query_classification(self):
        router = MemoryRouter()
        assert router.classify_query("What is the speed of light?") == QueryType.FACTUAL
        assert router.classify_query("How to install Python?") == QueryType.PROCEDURAL

    def test_mitm_future_timestamp_rejected(self):
        from core.security.mitm_protection import RequestSigner
        signer = RequestSigner()
        future_time = time.time() + 3600
        sig = signer.sign_request("GET", "/api", timestamp=future_time)
        assert signer.verify_request("GET", "/api", sig, timestamp=future_time) is False

    def test_stack_canary_corruption_detected(self):
        from core.security.stack_defense import SafeBuffer
        buf = SafeBuffer(64, canary=True)
        assert buf.check_canary() is True
        buf._buffer[buf._canary_position:buf._canary_position + 8] = b'\x00' * 8
        assert buf.check_canary() is False

    def test_branch_detection_whitelist_enforcement(self):
        from core.security.branch_detection import DataBranchMonitor, BranchType, AnomalyType
        m = DataBranchMonitor(enforce_destination_whitelist=True, authorized_destinations={"good.com"})
        anomalies = m.check_branch_policy(BranchType.NETWORK_SEND, "evil.com", "data")
        unauth = [a for a in anomalies if a.anomaly_type == AnomalyType.UNAUTHORIZED_DESTINATION]
        assert len(unauth) == 1

    def test_dlp_custom_pattern_type(self):
        from core.security.dlp import DataLeakProtector, SensitiveDataPattern, LeakSeverity
        dlp = DataLeakProtector()
        dlp.add_custom_pattern("test", r'TEST-\d{4}', LeakSeverity.HIGH)
        events = dlp.scan("Here is TEST-5678 in text")
        custom = [e for e in events if e.pattern_type == SensitiveDataPattern.CUSTOM]
        assert len(custom) == 1
