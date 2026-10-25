"""Tests for enriched Knowledge Graph: expanded concepts, entity extraction,
PageRank insights API, concept search, neighborhood, shortest path, domain map."""
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import pytest

os.environ.setdefault("JARVIS_HOME", tempfile.mkdtemp())

from core.knowledge_graph import (
    EdgeType,
    KnowledgeGraph,
    NodeType,
    get_knowledge_graph,
    KnowledgeGraphContext,
    get_knowledge_graph_context,
)


@pytest.fixture
def kg(tmp_path):
    graph = KnowledgeGraph(storage_path=tmp_path / "kg_test")
    return graph


class TestExpandedConceptKeywords:
    def test_math_concepts_still_detected(self, kg):
        concepts = kg._extract_concepts("Ramsey coloring of a complete graph")
        assert "ramsey" in concepts

    def test_prime_concept(self, kg):
        concepts = kg._extract_concepts("Prime factor gcd analysis")
        assert "prime" in concepts

    def test_machine_learning_concept(self, kg):
        concepts = kg._extract_concepts("Training a neural network with gradient descent")
        assert "machine_learning" in concepts

    def test_nlp_concept(self, kg):
        concepts = kg._extract_concepts("Named entity recognition and tokenization")
        assert "nlp" in concepts

    def test_cybersecurity_concept(self, kg):
        concepts = kg._extract_concepts("Encryption and authentication vulnerabilities")
        assert "cybersecurity" in concepts

    def test_distributed_systems_concept(self, kg):
        concepts = kg._extract_concepts("Byzantine fault tolerance and consensus")
        assert "distributed_systems" in concepts

    def test_algorithms_concept(self, kg):
        concepts = kg._extract_concepts("Dynamic programming and greedy algorithms")
        assert "algorithms" in concepts

    def test_data_science_concept(self, kg):
        concepts = kg._extract_concepts("Regression and classification with feature selection")
        assert "data_science" in concepts

    def test_economics_concept(self, kg):
        concepts = kg._extract_concepts("Inflation and GDP with interest rate policy")
        assert "economics" in concepts

    def test_biology_concept(self, kg):
        concepts = kg._extract_concepts("Protein and genome mutation expression")
        assert "biology" in concepts

    def test_psychology_concept(self, kg):
        concepts = kg._extract_concepts("Cognitive bias and perception")
        assert "psychology" in concepts

    def test_music_theory_concept(self, kg):
        concepts = kg._extract_concepts("Harmony and melody with chord progression")
        assert "music_theory" in concepts

    def test_multiple_concepts(self, kg):
        concepts = kg._extract_concepts(
            "Training a neural network using dynamic programming for optimization"
        )
        assert "machine_learning" in concepts
        assert "algorithms" in concepts

    def test_concept_limit_10(self, kg):
        content = " ".join(kw for kw in [
            "prime", "graph", "probability", "optimal", "neural network",
            "encryption", "consensus", "sorting", "regression", "inflation",
        ])
        concepts = kg._extract_concepts(content)
        assert len(concepts) <= 10


class TestEntityExtraction:
    def test_proper_noun_extraction(self, kg):
        entities = kg._extract_entities("John Smith and Jane Doe worked on Project Alpha")
        names = [e[0] for e in entities]
        assert "John Smith" in names
        assert "Jane Doe" in names
        assert "Project Alpha" in names

    def test_quoted_term_extraction(self, kg):
        entities = kg._extract_entities('The "quantum entanglement" principle applies')
        names = [e[0] for e in entities]
        assert "quantum entanglement" in names

    def test_code_term_extraction(self, kg):
        entities = kg._extract_entities("Use the `fast_transform` function")
        names = [e[0] for e in entities]
        assert "fast_transform" in names

    def test_entity_type_labels(self, kg):
        entities = kg._extract_entities('Alice said "hello world"')
        types = {e[1] for e in entities}
        assert "proper_noun" in types
        assert "quoted_term" in types

    def test_short_entities_filtered(self, kg):
        entities = kg._extract_entities("I am OK")
        names = [e[0] for e in entities]
        assert "OK" not in names

    def test_entities_in_concept_extraction(self, kg):
        concepts = kg._extract_concepts("Deep Learning by Andrew Ng discusses neural networks")
        assert "andrew ng" in concepts or "machine_learning" in concepts

    def test_entity_limit_20(self, kg):
        names = " ".join(f"Person{i} Person{i}" for i in range(30))
        entities = kg._extract_entities(names)
        assert len(entities) <= 20


class TestEnrichedAddInsight:
    def test_domain_linking(self, kg):
        nid = kg.add_insight("i1", "Training neural networks", domain="AI")
        assert nid is not None
        domain_nodes = [n for n in kg.nodes.values() if n.node_type == NodeType.DOMAIN]
        assert len(domain_nodes) >= 1

    def test_technique_linking(self, kg):
        nid = kg.add_insight("i2", "Using backpropagation", technique="gradient descent")
        assert nid is not None
        tech_nodes = [n for n in kg.nodes.values() if n.node_type == NodeType.TECHNIQUE]
        assert len(tech_nodes) >= 1

    def test_entity_auto_extraction(self, kg):
        nid = kg.add_insight("i3", "Andrew Ng published on deep learning")
        entity_nodes = [n for n in kg.nodes.values() if n.node_type == NodeType.ENTITY]
        assert len(entity_nodes) >= 1

    def test_full_insight_with_all_links(self, kg):
        nid = kg.add_insight(
            "i4",
            "Reinforcement learning for optimization",
            problem_id="P1",
            strategy="trial and error",
            category="ML",
            domain="AI",
            technique="policy gradient",
        )
        assert nid is not None
        type_counts = defaultdict(int)
        for n in kg.nodes.values():
            type_counts[n.node_type.value] += 1
        assert type_counts.get("domain", 0) >= 1
        assert type_counts.get("technique", 0) >= 1


class TestPageRankInsights:
    def test_pagerank_insights_returns_sorted(self, kg):
        kg.add_node("Alpha insight about neural networks", NodeType.INSIGHT)
        kg.add_node("Beta insight about prime numbers", NodeType.INSIGHT)
        kg.add_node("Gamma concept", NodeType.CONCEPT)
        results = kg.get_pagerank_insights(limit=10)
        assert len(results) >= 2
        assert all("pagerank" in r for r in results)

    def test_pagerank_insights_limit(self, kg):
        for i in range(15):
            kg.add_node(f"Insight {i}", NodeType.INSIGHT)
        results = kg.get_pagerank_insights(limit=5)
        assert len(results) <= 5

    def test_pagerank_insights_content_truncated(self, kg):
        kg.add_node("A" * 500, NodeType.INSIGHT)
        results = kg.get_pagerank_insights(limit=1)
        assert len(results[0]["content"]) <= 200

    def test_pagerank_insights_empty_graph(self, kg):
        results = kg.get_pagerank_insights(limit=10)
        assert results == []


class TestConceptSearch:
    def test_search_by_concept_name(self, kg):
        kg.add_node("machine_learning", NodeType.CONCEPT)
        results = kg.search_concepts("machine")
        assert len(results) >= 1
        assert results[0]["concept"] == "machine_learning"

    def test_search_concepts_with_insights(self, kg):
        cid = kg.add_node("neural_network", NodeType.CONCEPT)
        iid = kg.add_node("Training neural networks works well", NodeType.INSIGHT)
        kg.add_edge(iid, cid, EdgeType.REFERENCES_CONCEPT)
        results = kg.search_concepts("neural")
        assert len(results) >= 1
        assert "connected_insights" in results[0]

    def test_search_concepts_empty(self, kg):
        results = kg.search_concepts("nonexistent")
        assert results == []

    def test_search_concepts_limit(self, kg):
        for i in range(10):
            kg.add_node(f"concept_{i}", NodeType.CONCEPT)
        results = kg.search_concepts("concept", limit=3)
        assert len(results) <= 3


class TestNeighborhood:
    def test_neighborhood_depth_0(self, kg):
        nid = kg.add_node("Center node", NodeType.INSIGHT)
        result = kg.get_neighborhood(nid, depth=0)
        assert result["center"] == nid
        assert len(result["nodes"]) >= 1

    def test_neighborhood_depth_1(self, kg):
        n1 = kg.add_node("Center", NodeType.INSIGHT)
        n2 = kg.add_node("Neighbor", NodeType.CONCEPT)
        kg.add_edge(n1, n2, EdgeType.REFERENCES_CONCEPT)
        result = kg.get_neighborhood(n1, depth=1)
        assert len(result["nodes"]) >= 2
        assert len(result["edges"]) >= 1

    def test_neighborhood_nonexistent(self, kg):
        result = kg.get_neighborhood("nonexistent_999999", depth=1)
        assert "error" in result

    def test_neighborhood_bidirectional(self, kg):
        n1 = kg.add_node("A", NodeType.INSIGHT)
        n2 = kg.add_node("B", NodeType.INSIGHT)
        kg.add_edge(n1, n2, EdgeType.RELATED_TO)
        result = kg.get_neighborhood(n2, depth=1)
        assert len(result["nodes"]) >= 2


class TestShortestPath:
    def test_shortest_path_direct(self, kg):
        n1 = kg.add_node("Start", NodeType.INSIGHT)
        n2 = kg.add_node("End", NodeType.CONCEPT)
        kg.add_edge(n1, n2, EdgeType.REFERENCES_CONCEPT)
        path = kg.shortest_path(n1, n2)
        assert path is not None
        assert path[0] == n1
        assert path[-1] == n2

    def test_shortest_path_multi_hop(self, kg):
        n1 = kg.add_node("A", NodeType.INSIGHT)
        n2 = kg.add_node("B", NodeType.CONCEPT)
        n3 = kg.add_node("C", NodeType.INSIGHT)
        kg.add_edge(n1, n2, EdgeType.REFERENCES_CONCEPT)
        kg.add_edge(n2, n3, EdgeType.RELATED_TO)
        path = kg.shortest_path(n1, n3)
        assert path is not None
        assert len(path) == 3

    def test_shortest_path_no_connection(self, kg):
        n1 = kg.add_node("Isolated1", NodeType.INSIGHT)
        n2 = kg.add_node("Isolated2", NodeType.INSIGHT)
        path = kg.shortest_path(n1, n2)
        assert path is None

    def test_shortest_path_self(self, kg):
        n1 = kg.add_node("Self", NodeType.INSIGHT)
        path = kg.shortest_path(n1, n1)
        assert path == [n1]

    def test_shortest_path_nonexistent(self, kg):
        path = kg.shortest_path("fake_1", "fake_2")
        assert path is None


class TestDomainMap:
    def test_domain_map_basic(self, kg):
        did = kg.add_node("AI", NodeType.DOMAIN)
        iid = kg.add_node("Deep learning insight", NodeType.INSIGHT)
        kg.add_edge(iid, did, EdgeType.IN_DOMAIN)
        dmap = kg.get_domain_map()
        assert "AI" in dmap
        assert len(dmap["AI"]) >= 1

    def test_domain_map_empty(self, kg):
        dmap = kg.get_domain_map()
        assert dmap == {}

    def test_domain_map_multiple_insights(self, kg):
        did = kg.add_node("Physics", NodeType.DOMAIN)
        for i in range(3):
            iid = kg.add_node(f"Physics insight {i}", NodeType.INSIGHT)
            kg.add_edge(iid, did, EdgeType.IN_DOMAIN)
        dmap = kg.get_domain_map()
        assert "Physics" in dmap
        assert len(dmap["Physics"]) >= 3


class TestAutoLinkSimilar:
    def test_auto_link_finds_similar(self, kg):
        kg.add_node("Neural networks learn patterns from data", NodeType.INSIGHT)
        kg.add_node("Neural networks learn patterns from examples", NodeType.INSIGHT)
        links = kg.auto_link_similar(threshold=0.5)
        assert links >= 1

    def test_auto_link_no_similar(self, kg):
        kg.add_node("Quantum mechanics wave function", NodeType.INSIGHT)
        kg.add_node("Cooking recipes with garlic butter", NodeType.INSIGHT)
        links = kg.auto_link_similar(threshold=0.8)
        assert links == 0

    def test_auto_link_respects_limit(self, kg):
        for i in range(15):
            kg.add_node(f"Similar content about topic X variant {i}", NodeType.INSIGHT)
        links = kg.auto_link_similar(threshold=0.3)
        assert links <= 100


class TestNewNodeTypes:
    def test_entity_node_type(self, kg):
        nid = kg.add_node("Andrew Ng", NodeType.ENTITY, {"entity_type": "proper_noun"})
        node = kg.nodes[nid]
        assert node.node_type == NodeType.ENTITY
        assert node.metadata["entity_type"] == "proper_noun"

    def test_domain_node_type(self, kg):
        nid = kg.add_node("Physics", NodeType.DOMAIN)
        node = kg.nodes[nid]
        assert node.node_type == NodeType.DOMAIN

    def test_technique_node_type(self, kg):
        nid = kg.add_node("Policy Gradient", NodeType.TECHNIQUE)
        node = kg.nodes[nid]
        assert node.node_type == NodeType.TECHNIQUE


class TestNewEdgeTypes:
    def test_in_domain_edge(self, kg):
        n1 = kg.add_node("Insight", NodeType.INSIGHT)
        n2 = kg.add_node("Science", NodeType.DOMAIN)
        assert kg.add_edge(n1, n2, EdgeType.IN_DOMAIN)

    def test_uses_technique_edge(self, kg):
        n1 = kg.add_node("Insight", NodeType.INSIGHT)
        n2 = kg.add_node("Backprop", NodeType.TECHNIQUE)
        assert kg.add_edge(n1, n2, EdgeType.USES_TECHNIQUE)

    def test_mentions_entity_edge(self, kg):
        n1 = kg.add_node("Insight", NodeType.INSIGHT)
        n2 = kg.add_node("Person", NodeType.ENTITY)
        assert kg.add_edge(n1, n2, EdgeType.MENTIONS_ENTITY)

    def test_depends_on_edge(self, kg):
        n1 = kg.add_node("A", NodeType.INSIGHT)
        n2 = kg.add_node("B", NodeType.INSIGHT)
        assert kg.add_edge(n1, n2, EdgeType.DEPENDS_ON)


class TestKnowledgeGraphContextEnriched:
    def test_context_for_ml_query(self, kg):
        kg.add_node("Neural networks require training data", NodeType.INSIGHT)
        ctx = KnowledgeGraphContext(knowledge_graph=kg)
        result = ctx.get_context_for_query("neural network training", max_items=5, max_tokens=500)
        assert isinstance(result, list)

    def test_context_related(self, kg):
        nid = kg.add_node("Important ML insight", NodeType.INSIGHT, {"insight_id": "ml_1"})
        ctx = KnowledgeGraphContext(knowledge_graph=kg)
        result = ctx.get_related_context("ml_1", max_items=5)
        assert isinstance(result, list)

    def test_context_cache_ttl(self, kg):
        ctx = KnowledgeGraphContext(knowledge_graph=kg)
        kg.add_node("Cached insight", NodeType.INSIGHT)
        result1 = ctx.get_context_for_query("cached", max_items=5)
        result2 = ctx.get_context_for_query("cached", max_items=5)
        assert result1 == result2

    def test_clear_cache(self, kg):
        ctx = KnowledgeGraphContext(knowledge_graph=kg)
        kg.add_node("Test insight", NodeType.INSIGHT)
        ctx.get_context_for_query("test")
        ctx.clear_cache()
        assert len(ctx._context_cache) == 0


class TestGraphStatisticsEnriched:
    def test_stats_include_new_types(self, kg):
        kg.add_node("ML insight about neural networks", NodeType.INSIGHT)
        kg.add_node("AI", NodeType.DOMAIN)
        kg.add_node("Gradient Descent", NodeType.TECHNIQUE)
        kg.add_node("Andrew Ng", NodeType.ENTITY)
        stats = kg.get_graph_statistics()
        assert "total_nodes" in stats
        assert stats["total_nodes"] >= 4
        assert "node_types" in stats


class TestPersistenceWithNewTypes:
    def test_save_and_load_new_types(self, tmp_path):
        kg = KnowledgeGraph(storage_path=tmp_path / "kg_persist")
        kg.add_node("Test Entity", NodeType.ENTITY, {"entity_type": "proper_noun"})
        kg.add_node("Physics", NodeType.DOMAIN)
        kg.add_node("Backprop", NodeType.TECHNIQUE)
        kg._save_state()
        kg2 = KnowledgeGraph(storage_path=tmp_path / "kg_persist")
        entity_nodes = [n for n in kg2.nodes.values() if n.node_type == NodeType.ENTITY]
        domain_nodes = [n for n in kg2.nodes.values() if n.node_type == NodeType.DOMAIN]
        tech_nodes = [n for n in kg2.nodes.values() if n.node_type == NodeType.TECHNIQUE]
        assert len(entity_nodes) >= 1
        assert len(domain_nodes) >= 1
        assert len(tech_nodes) >= 1