"""Regression tests for VectorMemory's search-time lexical fallback.

Design (from NEXT_STEPS.md, decided this cycle): when the embedding
model is degraded, encode() produces either zero vectors (every pair
perfectly similar — the peft incident) or hash noise (plausible-looking
garbage). search() must refuse to rank on either and fall back to
deterministic word-overlap scoring, flagged in metadata so consumers can
tell the ranking basis apart.

store() may keep accepting entries while degraded (text preserved for a
later re-embed), but search() must never return noise-as-ranking.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class TestEmbedderUsable:
    @staticmethod
    def _vm(embedding_fn=None):
        from core.vector_memory import VectorMemory

        return VectorMemory(embedding_fn=embedding_fn)

    def test_zero_vector_query_rejected(self):
        vm = self._vm(embedding_fn=lambda text: [0.0] * 768)
        vm._interconnect = None
        assert vm._embedder_usable([0.0] * 768) is False

    def test_real_vector_accepted(self):
        vm = self._vm(embedding_fn=lambda text: [0.1] * 768)
        vm._interconnect = None
        assert vm._embedder_usable([0.1] * 768) is True

    def test_health_reporting_zero_vectors_rejected(self):
        """The interconnect's health() names the degraded state — trust it."""
        vm = self._vm()
        vm._interconnect = type("FakeIC", (), {
            "encode": lambda self, t: [0.0] * 768,
            "health": lambda self: {"returns_zero_vectors": True},
        })()
        assert vm._embedder_usable() is False

    def test_health_reporting_healthy_accepted(self):
        vm = self._vm()
        vm._interconnect = type("FakeIC", (), {
            "encode": lambda self, t: [0.1] * 768,
            "health": lambda self: {"returns_zero_vectors": False},
        })()
        assert vm._embedder_usable([0.1] * 768) is True

    def test_no_embedder_at_all_rejected(self):
        vm = self._vm(embedding_fn=None)
        vm._interconnect = None
        assert vm._embedder_usable() is False


class TestSearchLexicalFallback:
    @staticmethod
    def _vm_degraded():
        """VectorMemory whose embedder returns zero vectors (degraded)."""
        from core.vector_memory import VectorMemory

        vm = VectorMemory(embedding_fn=lambda text: [0.0] * 768)
        vm._interconnect = None
        return vm

    def test_zero_vector_search_does_not_rank_noise(self):
        """With a zero-vector embedder, old behaviour scored every entry
        ~1.0 (perfectly similar). Must now return lexical results or [],
        never confident nonsense."""
        vm = self._vm_degraded()
        vm.store("python testing best practices", {"kind": "note"})
        vm.store("cooking pasta with tomatoes", {"kind": "note"})

        results = vm.search("python testing", k=5)
        # The matching entry is found by word overlap...
        texts = [r[2].get("text", "") for r in results]
        assert any("python" in t for t in texts), results
        # ...and results are flagged so consumers know the basis changed.
        assert all(r[2].get("lexical_fallback") is True for r in results)

    def test_lexical_ranks_by_word_overlap(self):
        vm = self._vm_degraded()
        vm.store("python pytest runner failed", {"kind": "a"})
        vm.store("completely unrelated dinner plans", {"kind": "b"})

        results = vm.search("python pytest", k=5)
        assert results, "lexical fallback returned nothing"
        assert "pytest" in results[0][2]["text"]
        assert results[0][1] > 0.0

    def test_lexical_unrelated_entries_score_zero(self):
        vm = self._vm_degraded()
        vm.store("random unrelated content here", {})
        results = vm.search("quantum entanglement", k=5)
        # Word overlap is an honest zero: no results, not noise.
        assert results == []

    def test_lexical_respects_min_score(self):
        vm = self._vm_degraded()
        vm.store("one shared word only", {})
        results = vm.search("one shared word only different query", k=5, min_score=0.99)
        assert results == []

    def test_lexical_respects_filter_fn(self):
        vm = self._vm_degraded()
        vm.store("python script", {"kind": "keep"})
        vm.store("python script", {"kind": "skip"})
        results = vm.search("python script", k=5,
                            filter_fn=lambda md: md.get("kind") == "skip")
        assert len(results) == 1
        assert results[0][2]["kind"] == "skip"

    def test_healthy_embedder_still_uses_hnsw(self):
        """The fallback must not engage when embeddings are real."""
        import random

        from core.vector_memory import VectorMemory

        rng = random.Random(42)

        def healthy(text):
            # Deterministic-ish semantic-ish vector: same text -> same vector
            rng2 = random.Random(hash(text) & 0xFFFFFFFF)
            return [rng2.uniform(-1, 1) for _ in range(64)]

        vm = VectorMemory(embedding_fn=healthy)
        vm._interconnect = None
        # Force index dim to match the fake embedder
        from core.vector_memory import HNSWIndex
        vm.index = HNSWIndex(dim=64)

        vm.store("python testing", {})
        vm.store("cooking pasta", {})
        results = vm.search("python testing", k=2)
        assert results
        # No fallback flag: ranked by real (fake-but-nonzero) embeddings
        assert not any(r[2].get("lexical_fallback") for r in results)

    def test_lexical_helpers_bounded(self):
        from core.vector_memory import VectorMemory

        assert VectorMemory._lexical_similarity(
            VectorMemory._lexical_tokens("python pytest"), "pytest") > 0.0
        assert VectorMemory._lexical_similarity(
            VectorMemory._lexical_tokens("python"), "") == 0.0
        sim = VectorMemory._lexical_similarity(
            VectorMemory._lexical_tokens("same words same words"), "same words")
        assert 0.0 < sim <= 1.0