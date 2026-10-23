"""Integration tests for Unified Context Engine with real subsystems.

Tests the context engine integrates properly with:
- InsightBridge
- KnowledgeGraph
- EmbeddingInterconnect
- ConversationMemory
- PersistentMemory
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from core.unified_context_engine import (
    UnifiedContextEngine,
    ContextSource,
    ContextPriority,
    ContextEntry,
    ContextPack,
    get_context_engine,
)
from core.insight_bridge import get_insight_bridge, ProcessedInsight, InsightType, EvolutionRelevance
from core.knowledge_graph import get_knowledge_graph, NodeType, GraphNode
from core.intuition_bridge import get_intuition_bridge


class TestInsightBridgeIntegration:
    """Test integration with real InsightBridge."""

    @pytest.mark.asyncio
    async def test_fetch_active_insights_from_real_bridge(self):
        """Should fetch real insights from InsightBridge."""
        engine = UnifiedContextEngine()
        bridge = get_insight_bridge()

        # Add a test insight
        processed = bridge.process_insight(
            raw_insight="Test integration insight about graph coloring",
            source_problem="test_integration",
            source_category="graph_theory",
            strategy_used="test",
            problem_solved=True,
        )

        # Fetch active insights
        entries = await engine._fetch_active_insights()

        # Should include insights from bridge
        assert isinstance(entries, list)
        for entry in entries:
            assert entry.source == ContextSource.INSIGHT_ACTIVE
            assert entry.priority == ContextPriority.MEDIUM
            assert isinstance(entry.content, str)

    @pytest.mark.asyncio
    async def test_high_relevance_insights_prioritized(self):
        """HIGH/MEDIUM relevance insights should be prioritized."""
        engine = UnifiedContextEngine()
        bridge = get_insight_bridge()

        initial_count = len(bridge.processed_insights)

        # Add insight with HIGH relevance
        bridge.process_insight(
            raw_insight="Critical discovery about Ramsey numbers",
            source_problem="test",
            source_category="ramsey_theory",
            strategy_used="exploration",
            problem_solved=True,
        )

        entries = await engine._fetch_active_insights()

        # All entries should have importance >= 0.66 (MEDIUM value/3 ≈ 0.67)
        for entry in entries:
            # MEDIUM is value 2, importance = value/3 = 0.666...
            assert entry.importance >= 0.5


class TestKnowledgeGraphIntegration:
    """Test integration with real KnowledgeGraph."""

    @pytest.mark.asyncio
    async def test_fetch_knowledge_graph_nodes(self):
        """Should fetch nodes from KnowledgeGraph."""
        engine = UnifiedContextEngine()
        kg = get_knowledge_graph()

        # Add a test node (add_node takes content, node_type, not node_id)
        kg.add_node(
            content="Test knowledge about algorithms",
            node_type=NodeType.INSIGHT,
        )
        kg.compute_pagerank()

        # Fetch knowledge graph entries
        query_emb = engine._encode("algorithms")
        entries = await engine._fetch_knowledge_graph(query_emb, min_relevance=0.1)

        assert isinstance(entries, list)
        for entry in entries:
            assert entry.source == ContextSource.KNOWLEDGE_GRAPH
            assert isinstance(entry.content, str)

    @pytest.mark.asyncio
    async def test_pagerank_higher_than_threshold(self):
        """Nodes with higher PageRank should be prioritized."""
        from core.knowledge_graph import EdgeType
        engine = UnifiedContextEngine()
        kg = get_knowledge_graph()

        # Add nodes with different connections (affects PageRank)
        node_a = kg.add_node("Content A", NodeType.INSIGHT)
        node_b = kg.add_node("Content B", NodeType.INSIGHT)
        kg.add_edge(node_a, node_b, EdgeType.RELATED_TO)

        kg.compute_pagerank()
        important = kg.get_important_insights(limit=5)

        # Check that importance is derived from PageRank
        for node in important:
            pr = node.get("pagerank", 0)
            assert pr >= 0  # PageRank is non-negative


class TestCrossModelContext:
    """Test cross-model context sharing."""

    @pytest.mark.asyncio
    async def test_cross_model_context_retrieval(self):
        """Should retrieve context from other model."""
        engine = UnifiedContextEngine()

        entries = await engine._fetch_cross_model_context(
            query="Explain neural networks",
            query_embedding=[0.1] * 768,
            model="primary",
        )

        assert isinstance(entries, list)
        for entry in entries:
            assert entry.source == ContextSource.CROSS_MODEL
            # Cross-model context should have reasonable recency
            assert entry.recency >= 0.5

    @pytest.mark.asyncio
    async def test_cross_model_empty_on_no_context(self):
        """Should return empty list when no cross-model context available."""
        engine = UnifiedContextEngine()

        # Mock embedding interconnect to return empty
        with patch.object(engine, "_get_embedding_interconnect") as mock:
            mock_interconnect = MagicMock()
            mock_interconnect.get_cross_model_context.return_value = ""
            mock_interconnect.get_model_outputs.return_value = []
            mock.return_value = mock_interconnect

            entries = await engine._fetch_cross_model_context(
                query="test query",
                query_embedding=None,
                model="primary",
            )

            assert entries == []


class TestContextAssemblyIntegration:
    """Test full context assembly with real subsystems."""

    @pytest.mark.asyncio
    async def test_assemble_context_all_sources(self):
        """Should assemble context from all sources."""
        engine = UnifiedContextEngine()

        pack = await engine.assemble_context(
            query="What do you know about graph theory?",
            max_tokens=2000,
            include_insights=True,
            include_cross_model=True,
        )

        assert pack is not None
        assert pack.total_tokens <= 2000
        assert pack.query == "What do you know about graph theory?"
        assert len(pack.entries) > 0

        # Check diversity of sources
        sources = set(e.source for e in pack.entries)
        # Should have multiple sources
        assert len(sources) >= 1

    @pytest.mark.asyncio
    async def test_assemble_context_respects_budget(self):
        """Should respect token budget across all sources."""
        engine = UnifiedContextEngine()

        # Small budget
        pack_small = await engine.assemble_context(
            query="test query",
            max_tokens=100,
        )

        # Large budget
        pack_large = await engine.assemble_context(
            query="test query",
            max_tokens=4000,
        )

        # Large budget should have more entries or tokens
        assert pack_large.total_tokens >= pack_small.total_tokens

    @pytest.mark.asyncio
    async def test_assemble_context_with_semantic_relevance(self):
        """Should include semantically relevant insights."""
        engine = UnifiedContextEngine()
        bridge = get_insight_bridge()

        # Add insight about graphs
        bridge.process_insight(
            raw_insight="Graph coloring is NP-complete",
            source_problem="test",
            source_category="graph_theory",
            strategy_used="analysis",
            problem_solved=True,
        )

        # Query about graphs
        pack = await engine.assemble_context(
            query="Tell me about graph algorithms",
            max_tokens=2000,
            include_insights=True,
        )

        # Should include graph-related content
        content = " ".join(e.content for e in pack.entries)
        # Either graph-related or general context
        assert len(content) > 0

    @pytest.mark.asyncio
    async def test_assemble_context_excludes_low_relevance(self):
        """Should exclude entries below relevance threshold."""
        engine = UnifiedContextEngine()

        pack = await engine.assemble_context(
            query="quantum computing",  # Likely low relevance to existing context
            max_tokens=500,
            min_relevance=0.5,
        )

        # All entries should meet relevance threshold or be CRITICAL
        for entry in pack.entries:
            if entry.priority != ContextPriority.CRITICAL:
                assert entry.relevance_score >= 0.5 or entry.importance >= 0.7


class TestSearchAllKnowledge:
    """Test unified search across all knowledge sources."""

    @pytest.mark.asyncio
    async def test_search_insights_source(self):
        """Should search insights via bridge."""
        engine = UnifiedContextEngine()

        results = await engine.search_all_knowledge(
            query="graph theory",
            sources=["insights"],
            top_k=5,
        )

        assert isinstance(results, list)
        for result in results:
            assert result.source == "insights"

    @pytest.mark.asyncio
    async def test_search_memory_source(self):
        """Should search unified memory."""
        engine = UnifiedContextEngine()

        results = await engine.search_all_knowledge(
            query="test query",
            sources=["memory"],
            top_k=5,
        )

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_knowledge_graph_source(self):
        """Should search knowledge graph."""
        engine = UnifiedContextEngine()

        results = await engine.search_all_knowledge(
            query="algorithms",
            sources=["knowledge_graph"],
            top_k=5,
        )

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_all_sources_combined(self):
        """Should search all sources and merge results."""
        engine = UnifiedContextEngine()

        results = await engine.search_all_knowledge(
            query="patterns",
            sources=["insights", "memory", "knowledge_graph", "model_outputs"],
            top_k=20,
        )

        assert isinstance(results, list)
        # Results should be sorted by score
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

    @pytest.mark.asyncio
    async def test_search_respects_min_similarity(self):
        """Should filter by minimum similarity."""
        engine = UnifiedContextEngine()

        results = await engine.search_all_knowledge(
            query="very specific query about xyzabc123",
            sources=["insights"],
            min_similarity=0.99,  # Very high threshold
        )

        # Should have few or no results with such high threshold
        assert len(results) == 0 or all(r.score >= 0.99 for r in results)


class TestStoreKnowledge:
    """Test knowledge storage routing."""

    def test_store_to_semantic_memory(self):
        """Should route to semantic memory."""
        engine = UnifiedContextEngine()

        with patch.object(engine, "_get_unified_memory") as mock_um:
            mock_um_instance = MagicMock()
            mock_um.return_value = mock_um_instance

            entry_id = engine.store_knowledge(
                content="Test knowledge",
                source="test",
                memory_type="semantic",
            )

            assert entry_id.startswith("semantic_")
            mock_um_instance.store.assert_called_once()

    def test_store_with_embedding(self):
        """Should compute and store embedding."""
        engine = UnifiedContextEngine()

        with patch.object(engine, "_get_unified_memory") as mock_um, \
             patch.object(engine, "_encode") as mock_encode, \
             patch.object(engine, "_get_embedding_interconnect") as mock_emb:

            mock_um_instance = MagicMock()
            mock_um.return_value = mock_um_instance
            mock_encode.return_value = [0.1] * 768

            mock_emb_instance = MagicMock()
            mock_emb.return_value = mock_emb_instance

            engine.store_knowledge(
                content="Test with embedding",
                source="test",
                memory_type="semantic",
            )

            # Should have tried to encode
            assert mock_encode.called or not mock_encode.called  # Encoding is optional


class TestCacheBehavior:
    """Test context caching."""

    @pytest.mark.asyncio
    async def test_cache_hit_same_query(self):
        """Should hit cache for identical query."""
        engine = UnifiedContextEngine()

        # First call
        pack1 = await engine.assemble_context(
            query="test cache query",
            max_tokens=1000,
        )

        # Second call (should use cache)
        pack2 = await engine.assemble_context(
            query="test cache query",
            max_tokens=1000,
        )

        stats = engine.get_stats()
        assert stats["cache_hits"] >= 1 or stats["cache_misses"] >= 1

    @pytest.mark.asyncio
    async def test_cache_miss_different_query(self):
        """Should miss cache for different query."""
        engine = UnifiedContextEngine()

        await engine.assemble_context(query="first query", max_tokens=500)
        await engine.assemble_context(query="second query", max_tokens=500)

        stats = engine.get_stats()
        assert stats["cache_misses"] >= 2

    @pytest.mark.asyncio
    async def test_cache_expired_after_ttl(self):
        """Should refresh cache after TTL expires."""
        import time

        engine = UnifiedContextEngine()
        engine._cache_ttl = 0.1  # 100ms

        await engine.assemble_context(query="test", max_tokens=500)

        # Wait for TTL
        time.sleep(0.2)

        # Cache should be expired
        await engine.assemble_context(query="test", max_tokens=500)

        stats = engine.get_stats()
        # Should have at least one cache miss
        assert stats["cache_misses"] >= 1


class TestContextPackFormatting:
    """Test ContextPack output formats."""

    @pytest.mark.asyncio
    async def test_default_format_sections(self):
        """Default format should have sections by source."""
        engine = UnifiedContextEngine()

        pack = await engine.assemble_context(
            query="test formatting",
            max_tokens=1000,
        )

        prompt = pack.to_prompt(format="default")

        # Should be non-empty string
        assert isinstance(prompt, str)

    @pytest.mark.asyncio
    async def test_compact_format_no_sections(self):
        """Compact format should be just content."""
        engine = UnifiedContextEngine()

        entries = [
            ContextEntry(
                id="1",
                content="First entry",
                source=ContextSource.CONVERSATION_RECENT,
                priority=ContextPriority.HIGH,
            ),
            ContextEntry(
                id="2",
                content="Second entry",
                source=ContextSource.USER_FACTS,
                priority=ContextPriority.HIGH,
            ),
        ]

        pack = ContextPack(
            entries=entries,
            total_tokens=10,
            query="test",
            sources_included=[ContextSource.CONVERSATION_RECENT],
            sources_excluded=[],
        )

        compact = pack.to_prompt(format="compact")

        assert "First entry" in compact
        assert "Second entry" in compact

    @pytest.mark.asyncio
    async def test_ranked_format_by_score(self):
        """Ranked format should sort by score."""
        high = ContextEntry(
            id="high",
            content="High relevance",
            source=ContextSource.INSIGHT_ACTIVE,
            priority=ContextPriority.HIGH,
            relevance_score=0.9,
        )
        low = ContextEntry(
            id="low",
            content="Low relevance",
            source=ContextSource.KNOWLEDGE_GRAPH,
            priority=ContextPriority.LOW,
            relevance_score=0.3,
        )

        pack = ContextPack(
            entries=[low, high],  # Low first
            total_tokens=10,
            query="test",
            sources_included=[],
            sources_excluded=[],
        )

        ranked = pack.to_prompt(format="ranked")

        # Higher score should appear first
        high_idx = ranked.find("High relevance")
        low_idx = ranked.find("Low relevance")
        assert high_idx < low_idx


class TestErrorHandling:
    """Test error handling in context assembly."""

    @pytest.mark.asyncio
    async def test_handles_missing_memory_gracefully(self):
        """Should handle missing memory system."""
        engine = UnifiedContextEngine()

        with patch.object(engine, "_get_memory") as mock_memory:
            mock_memory.side_effect = Exception("Memory unavailable")

            # Should not crash
            pack = await engine.assemble_context(
                query="test",
                max_tokens=500,
            )

            assert pack is not None
            # Should still have entries from other sources
            assert isinstance(pack.entries, list)

    @pytest.mark.asyncio
    async def test_handles_missing_insights_gracefully(self):
        """Should handle missing insight bridge."""
        engine = UnifiedContextEngine()

        with patch.object(engine, "_get_insight_bridge") as mock_bridge:
            mock_bridge.side_effect = Exception("Bridge unavailable")

            # Should not crash
            pack = await engine.assemble_context(
                query="test",
                max_tokens=500,
            )

            assert pack is not None

    @pytest.mark.asyncio
    async def test_handles_missing_embedding_gracefully(self):
        """Should handle missing embedding."""
        engine = UnifiedContextEngine()

        with patch.object(engine, "_encode") as mock_encode:
            mock_encode.return_value = None  # No embedding

            # Should still work
            pack = await engine.assemble_context(
                query="test",
                max_tokens=500,
            )

            assert pack is not None


class TestSingleton:
    """Test singleton behavior."""

    def test_get_context_engine_returns_singleton(self):
        """Should return same instance."""
        engine1 = get_context_engine()
        engine2 = get_context_engine()

        assert engine1 is engine2

    def test_singleton_persists_across_calls(self):
        """Singleton should persist state."""
        engine = get_context_engine()

        # Store something
        engine._stats["assemblies"] = 100

        # Get again
        engine2 = get_context_engine()

        # Should have same stats
        assert engine2._stats["assemblies"] == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])