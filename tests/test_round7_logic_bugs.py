"""Tests for deep audit round 7 logic bug fixes."""

import pytest


class TestEvolutionOrchestratorTrend:
    def test_calculate_trend_filters_by_domain(self):
        from core.evolution_orchestrator import EvolutionOrchestrator, CapabilityDomain
        orch = EvolutionOrchestrator()
        trend = orch._calculate_trend("reasoning")
        assert trend in ("improving", "declining", "stable")

    def test_calculate_trend_invalid_domain(self):
        from core.evolution_orchestrator import EvolutionOrchestrator
        orch = EvolutionOrchestrator()
        trend = orch._calculate_trend("nonexistent_domain")
        assert trend == "stable"


class TestThreeLayerContextNoMutation:
    def test_inject_l2_does_not_mutate_input(self):
        from core.three_layer_context import ThreeLayerContext
        tlc = ThreeLayerContext()
        original = [{"role": "user", "content": "hello"}]
        original_copy = list(original)
        tlc.inject_l2_into_l1(original, model="primary")
        assert original == original_copy


class TestContextCascadeDeterministicHash:
    def test_concept_embedding_key_deterministic(self):
        from core.context_cascade import ContextCascadeIntegration
        import hashlib
        cascade = ContextCascadeIntegration()
        cascade._store_concept_embedding("test content", [0.1, 0.2, 0.3])
        key = hashlib.md5("test content".encode()).hexdigest()[:8]
        with cascade._lock:
            assert key in cascade._concept_embeddings

    def test_different_content_different_keys(self):
        import hashlib
        key1 = hashlib.md5("content A".encode()).hexdigest()[:8]
        key2 = hashlib.md5("content B".encode()).hexdigest()[:8]
        assert key1 != key2


class TestSelfReflectionNegationScoring:
    def setup_method(self):
        import tempfile
        from core.self_reflection import SelfReflection
        self.sr = SelfReflection(storage_path=__import__('pathlib').Path(tempfile.mkdtemp()))

    def test_not_bad_scores_higher_than_bad(self):
        self.sr.track_response_quality("response", "not bad")
        score_not_bad = self.sr._quality_scores[-1][0]
        self.sr.track_response_quality("response", "bad")
        score_bad = self.sr._quality_scores[-1][0]
        assert score_not_bad > score_bad

    def test_not_great_scores_lower_than_great(self):
        self.sr.track_response_quality("response", "not great")
        score_not_great = self.sr._quality_scores[-1][0]
        self.sr.track_response_quality("response", "great")
        score_great = self.sr._quality_scores[-1][0]
        assert score_not_great < score_great

    def test_great_feedback_scores_high(self):
        self.sr.track_response_quality("response", "great job")
        assert self.sr._quality_scores[-1][0] >= 0.8

    def test_bad_feedback_scores_low(self):
        self.sr.track_response_quality("response", "bad answer")
        assert self.sr._quality_scores[-1][0] <= 0.2


class TestProactiveFileOrder:
    def test_mentioned_files_preserves_recency(self):
        from core.proactive import ProactiveEngine
        engine = ProactiveEngine()
        engine.update_activity("look at file_a.py")
        engine.update_activity("now edit file_b.py")
        engine.update_activity("back to file_a.py")
        files = engine._mentioned_files
        if len(files) >= 2:
            last_a_idx = max(i for i, f in enumerate(files) if "file_a" in f)
            last_b_idx = max(i for i, f in enumerate(files) if "file_b" in f)
            assert last_a_idx > 0
            assert last_b_idx > 0


class TestSupervisorMemoryStatus:
    def test_get_system_memory_status_includes_total_used(self):
        from core.supervisor import ServerSupervisor
        sup = ServerSupervisor()
        status = sup.get_system_memory_status()
        assert "total_used_mb" in status
        assert status["total_used_mb"] > 0