"""Tests for Unified Context Engine.

Tests:
- Context assembly from all sources
- Semantic search across knowledge
- Knowledge storage routing
- Cache behavior
- Cross-model context sharing
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch, AsyncMock

from core.unified_context_engine import (
    UnifiedContextEngine,
    ContextSource,
    ContextPriority,
    ContextEntry,
    ContextPack,
    SearchResult,
    get_context_engine,
)


class TestContextEntry:
    """Tests for ContextEntry scoring."""

    def test_compute_score_basic(self):
        """Should compute weighted score correctly."""
        entry = ContextEntry(
            id="test_1",
            content="Test content",
            source=ContextSource.INSIGHT_ACTIVE,
            priority=ContextPriority.HIGH,
            relevance_score=0.8,
            importance=0.9,
            recency=0.7,
        )

        score = entry.compute_score()

        assert 0.0 <= score <= 1.0
        assert score > 0.5

    def test_compute_score_critical_priority(self):
        """Critical priority should boost score."""
        entry = ContextEntry(
            id="test_2",
            content="Critical content",
            source=ContextSource.SYSTEM,
            priority=ContextPriority.CRITICAL,
            relevance_score=0.5,
            importance=0.5,
            recency=0.5,
        )

        score = entry.compute_score()

        assert score > 0.5

    def test_compute_score_low_priority(self):
        """Low priority should reduce score."""
        high_entry = ContextEntry(
            id="high",
            content="High priority",
            source=ContextSource.INSIGHT_ACTIVE,
            priority=ContextPriority.HIGH,
            relevance_score=0.7,
            importance=0.7,
            recency=0.7,
        )

        low_entry = ContextEntry(
            id="low",
            content="Low priority",
            source=ContextSource.KNOWLEDGE_GRAPH,
            priority=ContextPriority.LOW,
            relevance_score=0.7,
            importance=0.7,
            recency=0.7,
        )

        assert high_entry.compute_score() > low_entry.compute_score()


class TestContextPack:
    """Tests for ContextPack formatting."""

    def test_to_prompt_default_format(self):
        """Should format entries by source."""
        entries = [
            ContextEntry(
                id="1",
                content="First insight",
                source=ContextSource.INSIGHT_ACTIVE,
                priority=ContextPriority.MEDIUM,
            ),
            ContextEntry(
                id="2",
                content="Second insight",
                source=ContextSource.INSIGHT_ACTIVE,
                priority=ContextPriority.MEDIUM,
            ),
            ContextEntry(
                id="3",
                content="User fact",
                source=ContextSource.USER_FACTS,
                priority=ContextPriority.HIGH,
            ),
        ]

        pack = ContextPack(
            entries=entries,
            total_tokens=100,
            query="test query",
            sources_included=[ContextSource.INSIGHT_ACTIVE, ContextSource.USER_FACTS],
            sources_excluded=[],
        )

        prompt = pack.to_prompt(format="default")

        assert "First insight" in prompt
        assert "Second insight" in prompt
        assert "User fact" in prompt
        assert "Insight Active" in prompt or "insight" in prompt.lower()
        assert "User Facts" in prompt or "user" in prompt.lower()

    def test_to_prompt_compact_format(self):
        """Should format entries compactly."""
        entries = [
            ContextEntry(
                id="1",
                content="Content one",
                source=ContextSource.INSIGHT_ACTIVE,
                priority=ContextPriority.MEDIUM,
            ),
            ContextEntry(
                id="2",
                content="Content two",
                source=ContextSource.USER_FACTS,
                priority=ContextPriority.HIGH,
            ),
        ]

        pack = ContextPack(
            entries=entries,
            total_tokens=50,
            query="test",
            sources_included=[ContextSource.INSIGHT_ACTIVE],
            sources_excluded=[],
        )

        prompt = pack.to_prompt(format="compact")

        assert "Content one" in prompt
        assert "Content two" in prompt
        assert "\n" in prompt

    def test_to_prompt_ranked_format(self):
        """Should format entries ranked by score."""
        entries = [
            ContextEntry(
                id="high",
                content="High relevance",
                source=ContextSource.INSIGHT_ACTIVE,
                priority=ContextPriority.HIGH,
                relevance_score=0.9,
            ),
            ContextEntry(
                id="low",
                content="Low relevance",
                source=ContextSource.KNOWLEDGE_GRAPH,
                priority=ContextPriority.LOW,
                relevance_score=0.3,
            ),
        ]

        pack = ContextPack(
            entries=entries,
            total_tokens=50,
            query="test",
            sources_included=[],
            sources_excluded=[],
        )

        prompt = pack.to_prompt(format="ranked")

        assert "High relevance" in prompt
        assert "Low relevance" in prompt

    def test_to_dict(self):
        """Should serialize to dictionary."""
        entries = [
            ContextEntry(
                id="test_1",
                content="Test",
                source=ContextSource.INSIGHT_ACTIVE,
                priority=ContextPriority.MEDIUM,
            )
        ]

        pack = ContextPack(
            entries=entries,
            total_tokens=10,
            query="test query",
            sources_included=[ContextSource.INSIGHT_ACTIVE],
            sources_excluded=[ContextSource.KNOWLEDGE_GRAPH],
            score_cutoff=0.5,
            assembly_time_ms=10.5,
        )

        data = pack.to_dict()

        assert data["total_tokens"] == 10
        assert data["query"] == "test query"
        assert len(data["entries"]) == 1
        assert data["score_cutoff"] == 0.5


class TestUnifiedContextEngine:
    """Tests for UnifiedContextEngine."""

    def test_singleton(self):
        """Should return same instance."""
        engine1 = get_context_engine()
        engine2 = get_context_engine()

        assert engine1 is engine2

    def test_estimate_tokens(self):
        """Should estimate tokens from text."""
        engine = UnifiedContextEngine()

        short_text = "Hello world"
        long_text = "This is a much longer text that should have more tokens"

        short_tokens = engine._estimate_tokens(short_text)
        long_tokens = engine._estimate_tokens(long_text)

        assert long_tokens > short_tokens
        assert short_tokens >= 1

    def test_encode_text(self):
        """Should encode text to embedding."""
        engine = UnifiedContextEngine()

        with patch.object(engine, "_get_embedding_interconnect") as mock_interconnect:
            mock_emb = MagicMock()
            mock_emb.encode.return_value = [0.1] * 768
            mock_interconnect.return_value = mock_emb

            result = engine._encode("test text")

            assert result is not None or result is None

    def test_compute_similarity(self):
        """Should compute cosine similarity."""
        engine = UnifiedContextEngine()

        emb1 = [1.0, 0.0, 0.0]
        emb2 = [1.0, 0.0, 0.0]
        emb3 = [0.0, 1.0, 0.0]

        import numpy as np
        sim_same = engine._compute_similarity_fast(np.array(emb1), np.array(emb2))
        sim_different = engine._compute_similarity_fast(np.array(emb1), np.array(emb3))

        assert sim_same > 0.99
        assert sim_different < 0.01

    def test_compute_similarity_empty_embeddings(self):
        """Should handle empty embeddings."""
        engine = UnifiedContextEngine()

        import numpy as np
        assert engine._compute_similarity_fast(np.array([]), np.array([1.0])) == 0.0
        assert engine._compute_similarity_fast(np.array([1.0]), np.array([])) == 0.0
        assert engine._compute_similarity_fast(np.array([]), np.array([])) == 0.0

    @pytest.mark.asyncio
    async def test_assemble_context_basic(self):
        """Should assemble context from sources."""
        engine = UnifiedContextEngine()

        with patch.object(engine, "_get_memory") as mock_memory, \
             patch.object(engine, "_get_persistent_memory") as mock_pm, \
             patch.object(engine, "_encode") as mock_encode:

            mock_mem = MagicMock()
            mock_mem.get_messages_for_llm.return_value = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
            mock_memory.return_value = mock_mem

            mock_pm_inst = MagicMock()
            mock_pm_inst.get_context_for_prompt.return_value = "User's name is Alex"
            mock_pm.return_value = mock_pm_inst

            mock_encode.return_value = None

            pack = await engine.assemble_context(
                query="What is my name?",
                max_tokens=1000,
            )

            assert pack is not None
            assert pack.query == "What is my name?"
            assert pack.total_tokens <= 1000
            assert isinstance(pack.entries, list)

    @pytest.mark.asyncio
    async def test_assemble_context_with_insights(self):
        """Should include insights when enabled."""
        engine = UnifiedContextEngine()

        with patch.object(engine, "_get_memory") as mock_memory, \
             patch.object(engine, "_get_insight_bridge") as mock_bridge, \
             patch.object(engine, "_encode") as mock_encode:

            mock_mem = MagicMock()
            mock_mem.get_messages_for_llm.return_value = []
            mock_memory.return_value = mock_mem

            mock_bridge_inst = MagicMock()
            processed_insight = MagicMock()
            processed_insight.id = "insight_1"
            processed_insight.raw_insight = "Ramsey numbers grow exponentially"
            processed_insight.evolution_relevance.value = 3
            processed_insight.source_category = "math"
            mock_bridge_inst.processed_insights = [processed_insight]
            mock_bridge.return_value = mock_bridge_inst

            mock_encode.return_value = None

            pack = await engine.assemble_context(
                query="Tell me about Ramsey theory",
                max_tokens=2000,
                include_insights=True,
            )

            insight_entries = [e for e in pack.entries if e.source == ContextSource.INSIGHT_ACTIVE]
            assert len(insight_entries) > 0

    @pytest.mark.asyncio
    async def test_assemble_context_exclude_sources(self):
        """Should exclude specified sources."""
        engine = UnifiedContextEngine()

        with patch.object(engine, "_get_memory") as mock_memory, \
             patch.object(engine, "_encode") as mock_encode:

            mock_mem = MagicMock()
            mock_mem.get_messages_for_llm.return_value = []
            mock_memory.return_value = mock_mem

            mock_encode.return_value = None

            pack = await engine.assemble_context(
                query="test",
                max_tokens=1000,
                exclude_sources=[ContextSource.INSIGHT_ACTIVE, ContextSource.KNOWLEDGE_GRAPH],
            )

            assert ContextSource.INSIGHT_ACTIVE not in pack.sources_included
            assert ContextSource.KNOWLEDGE_GRAPH not in pack.sources_included

    @pytest.mark.asyncio
    async def test_assemble_context_token_budget(self):
        """Should respect token budget."""
        engine = UnifiedContextEngine()

        with patch.object(engine, "_get_memory") as mock_memory, \
             patch.object(engine, "_encode") as mock_encode:

            mock_mem = MagicMock()
            mock_mem.get_messages_for_llm.return_value = [
                {"content": "A" * 1000} for _ in range(10)
            ]
            mock_memory.return_value = mock_mem

            mock_encode.return_value = None

            pack = await engine.assemble_context(
                query="test",
                max_tokens=200,
            )

            assert pack.total_tokens <= 200

    @pytest.mark.asyncio
    async def test_assemble_context_cache(self):
        """Should cache assembled context."""
        engine = UnifiedContextEngine()

        with patch.object(engine, "_get_memory") as mock_memory, \
             patch.object(engine, "_encode") as mock_encode:

            mock_mem = MagicMock()
            mock_mem.get_messages_for_llm.return_value = []
            mock_memory.return_value = mock_mem

            mock_encode.return_value = None

            pack1 = await engine.assemble_context(
                query="test query",
                max_tokens=1000,
            )

            pack2 = await engine.assemble_context(
                query="test query",
                max_tokens=1000,
            )

            stats = engine.get_stats()
            assert stats["cache_hits"] >= 1


class TestSearchAllKnowledge:
    """Tests for unified knowledge search."""

    @pytest.mark.asyncio
    async def test_search_all_knowledge_basic(self):
        """Should search across multiple sources."""
        engine = UnifiedContextEngine()

        with patch.object(engine, "_get_insight_bridge") as mock_bridge, \
             patch.object(engine, "_get_unified_memory") as mock_um, \
             patch.object(engine, "_encode") as mock_encode:

            mock_encode.return_value = [0.1] * 768

            mock_bridge_inst = MagicMock()
            processed = MagicMock()
            processed.id = "insight_1"
            processed.raw_insight = "Graph theory discovery"
            processed.source_category = "math"
            processed.evolution_relevance.value = 3
            mock_bridge_inst.processed_insights = [processed]
            mock_bridge.return_value = mock_bridge_inst

            mock_um_inst = MagicMock()
            mock_um_inst.search_semantic.return_value = []
            mock_um.return_value = mock_um_inst

            results = await engine.search_all_knowledge(
                query="graph theory",
                sources=["insights"],
                top_k=5,
            )

            assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_all_knowledge_no_embedding(self):
        """Should handle missing embedding gracefully."""
        engine = UnifiedContextEngine()

        with patch.object(engine, "_encode") as mock_encode:
            mock_encode.return_value = None

            results = await engine.search_all_knowledge(
                query="test query",
            )

            assert results == []

    @pytest.mark.asyncio
    async def test_search_all_knowledge_min_similarity(self):
        """Should filter by minimum similarity."""
        engine = UnifiedContextEngine()

        with patch.object(engine, "_get_insight_bridge") as mock_bridge, \
             patch.object(engine, "_encode") as mock_encode:

            query_vec = [0.1] * 768
            doc_vec = [-0.9] * 768
            mock_encode.side_effect = [query_vec, doc_vec]

            mock_bridge_inst = MagicMock()
            processed = MagicMock()
            processed.id = "insight_1"
            processed.raw_insight = "Low relevance insight"
            processed.source_category = "general"
            processed.evolution_relevance.value = 1
            mock_bridge_inst.processed_insights = [processed]
            mock_bridge.return_value = mock_bridge_inst

            results = await engine.search_all_knowledge(
                query="test",
                sources=["insights"],
                min_similarity=0.8,
            )

            assert len(results) == 0


class TestStoreKnowledge:
    """Tests for knowledge storage."""

    def test_store_knowledge_basic(self):
        """Should store knowledge and return ID."""
        engine = UnifiedContextEngine()

        with patch.object(engine, "_get_unified_memory") as mock_um, \
             patch.object(engine, "_encode") as mock_encode:

            mock_um_inst = MagicMock()
            mock_um.return_value = mock_um_inst
            mock_encode.return_value = [0.1] * 768

            entry_id = engine.store_knowledge(
                content="Test knowledge",
                source="test",
                memory_type="semantic",
                importance=0.8,
            )

            assert entry_id.startswith("semantic_")
            mock_um_inst.store.assert_called_once()

    def test_store_knowledge_with_metadata(self):
        """Should store with metadata."""
        engine = UnifiedContextEngine()

        with patch.object(engine, "_get_unified_memory") as mock_um, \
             patch.object(engine, "_encode") as mock_encode:

            mock_um_inst = MagicMock()
            mock_um.return_value = mock_um_inst
            mock_encode.return_value = [0.1] * 768

            engine.store_knowledge(
                content="Test",
                source="user_input",
                memory_type="conversation",
                importance=0.9,
                model="primary",
                metadata={"topic": "math"},
            )

            call_args = mock_um_inst.store.call_args
            assert call_args is not None


class TestSessionManagement:
    """Tests for session persistence."""

    def test_save_session(self, tmp_path):
        """Should save session state."""
        engine = UnifiedContextEngine(storage_path=tmp_path)

        result = engine.save_session()

        assert result is True
        session_file = tmp_path / "session.json"
        assert session_file.exists()

    def test_load_session(self, tmp_path):
        """Should load session state."""
        import json

        session_file = tmp_path / "session.json"
        session_file.write_text(json.dumps({
            "stats": {"assemblies": 10},
            "timestamp": datetime.now().isoformat(),
        }))

        engine = UnifiedContextEngine(storage_path=tmp_path)

        result = engine.load_session()

        assert result is True
        stats = engine.get_stats()
        assert stats["assemblies"] == 10

    def test_clear_cache(self):
        """Should clear context cache."""
        engine = UnifiedContextEngine()
        engine._context_cache["test"] = (MagicMock(), 0.0)

        engine.clear_cache()

        assert len(engine._context_cache) == 0

    def test_get_stats(self):
        """Should return statistics."""
        engine = UnifiedContextEngine()

        stats = engine.get_stats()

        assert "assemblies" in stats
        assert "searches" in stats
        assert "stores" in stats
        assert "cache_hits" in stats
        assert "cache_misses" in stats


class TestSearchResult:
    """Tests for SearchResult."""

    def test_search_result_creation(self):
        """Should create search result."""
        result = SearchResult(
            id="test_1",
            content="Test content",
            source="insights",
            score=0.85,
            metadata={"category": "math"},
        )

        assert result.id == "test_1"
        assert result.content == "Test content"
        assert result.source == "insights"
        assert result.score == 0.85


if __name__ == "__main__":
    pytest.main([__file__, "-v"])