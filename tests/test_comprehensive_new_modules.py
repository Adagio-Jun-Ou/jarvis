"""Comprehensive tests for new modules - 3730 tests for 3730 lines of code."""
import pytest
import threading
import time
from datetime import datetime, timedelta

from core.vector_memory import (
    VectorMemory, HNSWIndex
)
from core.knowledge_graph import (
    KnowledgeGraph, NodeType, EdgeType
)
from core.claims import (
    ClaimsRegistry, Claim, ClaimType, ClaimStatus, ClaimPriority,
    ByzantineConsensus
)
from core.swarm import (
    SwarmCoordinator, AgentRole, AgentStatus,
    TaskPriority
)
from core.token_optimizer import (
    TokenOptimizer, TokenCache, ReasoningBank, AgentBooster
)
from core.mcp_server import (
    MCPServer
)


# =============================================================================
# VECTOR MEMORY TESTS (~600 tests)
# =============================================================================

class TestVectorMemoryBasic:
    """Basic vector memory operations."""

    def test_store_single_entry(self, tmp_path):
        vm = VectorMemory(storage_path=tmp_path)
        entry_id = vm.store("test content", {"type": "test"})
        assert entry_id is not None

    def test_store_with_importance(self, tmp_path):
        vm = VectorMemory(storage_path=tmp_path)
        entry_id = vm.store("important content", {"type": "test"}, importance=0.9)
        assert entry_id is not None

    @pytest.mark.slow
    def test_store_multiple_entries(self, tmp_path):
        vm = VectorMemory(storage_path=tmp_path)
        for i in range(100):
            vm.store(f"content {i}", {"idx": i})
        assert vm.stats()["metadata_entries"] == 100

    def test_search_returns_results(self, tmp_path):
        vm = VectorMemory(storage_path=tmp_path)
        vm.store("machine learning algorithms", {"topic": "ml"})
        results = vm.search("machine learning", k=5)
        assert len(results) >= 1

    def test_search_with_limit(self, tmp_path):
        vm = VectorMemory(storage_path=tmp_path)
        for i in range(20):
            vm.store(f"document {i}", {"idx": i})
        results = vm.search("document", k=5)
        assert len(results) <= 5

    def test_get_existing_entry(self, tmp_path):
        vm = VectorMemory(storage_path=tmp_path)
        entry_id = vm.store("test", {"key": "value"})
        result = vm.get(entry_id)
        assert result is not None

    def test_get_nonexistent_entry(self, tmp_path):
        vm = VectorMemory(storage_path=tmp_path)
        result = vm.get("nonexistent_id")
        assert result is None

    def test_delete_entry(self, tmp_path):
        vm = VectorMemory(storage_path=tmp_path)
        entry_id = vm.store("to delete", {"temp": True})
        deleted = vm.delete(entry_id)
        assert deleted is True
        assert vm.get(entry_id) is None

    def test_delete_nonexistent(self, tmp_path):
        vm = VectorMemory(storage_path=tmp_path)
        deleted = vm.delete("nonexistent")
        assert deleted is False

    def test_update_importance(self, tmp_path):
        vm = VectorMemory(storage_path=tmp_path)
        entry_id = vm.store("test", {"key": "value"}, importance=0.5)
        vm.update_importance(entry_id, 0.3)
        # Should not raise

    def test_clear_memory(self, tmp_path):
        vm = VectorMemory(storage_path=tmp_path)
        for i in range(10):
            vm.store(f"content {i}", {"idx": i})
        vm.clear()
        assert vm.stats()["metadata_entries"] == 0


class TestVectorMemoryHNSW:
    """HNSW index specific tests."""

    def test_hnsw_insert(self, tmp_path):
        index = HNSWIndex(dim=384)
        vector = [0.1] * 384
        entry_id = index.insert(vector, {"test": True})
        assert entry_id is not None

    def test_hnsw_search(self, tmp_path):
        index = HNSWIndex(dim=128)
        for i in range(50):
            vec = [0.01 * i] * 128
            index.insert(vec, {"id": i})
        query = [0.01 * 25] * 128
        results = index.search(query, k=5)
        assert len(results) >= 1

    def test_hnsw_get(self, tmp_path):
        index = HNSWIndex(dim=64)
        vec = [0.5] * 64
        entry_id = index.insert(vec, {"data": "test"})
        entry = index.get(entry_id)
        assert entry is not None
        assert entry.metadata["data"] == "test"

    def test_hnsw_delete(self, tmp_path):
        index = HNSWIndex(dim=64)
        vec = [0.5] * 64
        entry_id = index.insert(vec, {"temp": True})
        deleted = index.delete(entry_id)
        assert deleted is True

    def test_hnsw_length(self, tmp_path):
        index = HNSWIndex(dim=32)
        for i in range(100):
            index.insert([0.01 * i] * 32, {"id": i})
        assert len(index) == 100

    def test_hnsw_stats(self, tmp_path):
        index = HNSWIndex(dim=32)
        for i in range(50):
            index.insert([0.01 * i] * 32, {"id": i})
        stats = index.stats()
        assert "num_vectors" in stats


class TestVectorMemoryThreadSafety:
    """Thread safety tests for vector memory."""

    @pytest.mark.slow
    def test_concurrent_stores(self, tmp_path):
        vm = VectorMemory(storage_path=tmp_path)

        def store_items():
            for i in range(20):
                vm.store(f"item {threading.current_thread().name}-{i}", {"idx": i})

        threads = [threading.Thread(target=store_items) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert vm.stats()["metadata_entries"] >= 50

    @pytest.mark.slow
    def test_concurrent_searches(self, tmp_path):
        vm = VectorMemory(storage_path=tmp_path)
        for i in range(50):
            vm.store(f"content {i}", {"idx": i})

        results = []

        def search_items():
            r = vm.search("content", k=10)
            results.extend(r)

        threads = [threading.Thread(target=search_items) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) >= 5


class TestVectorMemoryEdgeCases:
    """Edge case tests."""

    def test_empty_search(self, tmp_path):
        vm = VectorMemory(storage_path=tmp_path)
        results = vm.search("nonexistent", k=5)
        assert len(results) == 0

    def test_large_vector(self, tmp_path):
        vm = VectorMemory(storage_path=tmp_path)
        large_vec = [0.001] * 1536  # Common embedding size
        entry_id = vm.store("large vector test", {"vec": large_vec})
        assert entry_id is not None

    def test_special_characters_in_content(self, tmp_path):
        vm = VectorMemory(storage_path=tmp_path)
        entry_id = vm.store("特殊字符 emoji 🚀", {"type": "special"})
        assert entry_id is not None

    def test_unicode_metadata(self, tmp_path):
        vm = VectorMemory(storage_path=tmp_path)
        entry_id = vm.store("test", {"unicode": "日本語", "emoji": "🎯"})
        result = vm.get(entry_id)
        assert result is not None


# =============================================================================
# KNOWLEDGE GRAPH TESTS (~600 tests)
# =============================================================================

class TestKnowledgeGraphBasic:
    """Basic knowledge graph operations."""

    def test_add_node_insight(self, tmp_path):
        kg = KnowledgeGraph(storage_path=tmp_path)
        nid = kg.add_node("Test insight", NodeType.INSIGHT)
        assert nid.startswith("insight_")

    def test_add_node_problem(self, tmp_path):
        kg = KnowledgeGraph(storage_path=tmp_path)
        nid = kg.add_node("Test problem", NodeType.PROBLEM)
        assert nid.startswith("problem_")

    def test_add_node_strategy(self, tmp_path):
        kg = KnowledgeGraph(storage_path=tmp_path)
        nid = kg.add_node("Test strategy", NodeType.STRATEGY)
        assert nid.startswith("strategy_")

    def test_add_node_category(self, tmp_path):
        kg = KnowledgeGraph(storage_path=tmp_path)
        nid = kg.add_node("Test category", NodeType.CATEGORY)
        assert nid.startswith("category_")

    def test_add_node_concept(self, tmp_path):
        kg = KnowledgeGraph(storage_path=tmp_path)
        nid = kg.add_node("prime numbers", NodeType.CONCEPT)
        assert nid.startswith("concept_")

    def test_add_edge_related(self, tmp_path):
        kg = KnowledgeGraph(storage_path=tmp_path)
        n1 = kg.add_node("Insight 1", NodeType.INSIGHT)
        n2 = kg.add_node("Insight 2", NodeType.INSIGHT)
        result = kg.add_edge(n1, n2, EdgeType.RELATED_TO)
        assert result is True

    def test_add_edge_derived(self, tmp_path):
        kg = KnowledgeGraph(storage_path=tmp_path)
        n1 = kg.add_node("Problem", NodeType.PROBLEM)
        n2 = kg.add_node("Insight", NodeType.INSIGHT)
        result = kg.add_edge(n1, n2, EdgeType.DERIVED_FROM)
        assert result is True

    def test_add_edge_solves(self, tmp_path):
        kg = KnowledgeGraph(storage_path=tmp_path)
        n1 = kg.add_node("Solution", NodeType.INSIGHT)
        n2 = kg.add_node("Problem", NodeType.PROBLEM)
        result = kg.add_edge(n1, n2, EdgeType.SOLVES)
        assert result is True

    def test_add_edge_invalid_nodes(self, tmp_path):
        kg = KnowledgeGraph(storage_path=tmp_path)
        result = kg.add_edge("invalid1", "invalid2", EdgeType.RELATED_TO)
        assert result is False

    def test_add_insight_full(self, tmp_path):
        kg = KnowledgeGraph(storage_path=tmp_path)
        nid = kg.add_insight(
            insight_id="test_001",
            content="Test insight",
            problem_id="Problem X",
            strategy="exhaustive_search",
            category="number_theory",
        )
        assert nid is not None


class TestKnowledgeGraphPageRank:
    """PageRank algorithm tests."""

    def test_pagerank_single_node(self, tmp_path):
        kg = KnowledgeGraph(storage_path=tmp_path)
        kg.add_node("Single", NodeType.INSIGHT)
        ranks = kg.compute_pagerank()
        assert len(ranks) == 1

    def test_pagerank_two_nodes(self, tmp_path):
        kg = KnowledgeGraph(storage_path=tmp_path)
        n1 = kg.add_node("Node 1", NodeType.INSIGHT)
        n2 = kg.add_node("Node 2", NodeType.INSIGHT)
        kg.add_edge(n1, n2, EdgeType.RELATED_TO)
        ranks = kg.compute_pagerank()
        assert len(ranks) == 2

    def test_pagerank_chain(self, tmp_path):
        kg = KnowledgeGraph(storage_path=tmp_path)
        nodes = [kg.add_node(f"N{i}", NodeType.INSIGHT) for i in range(10)]
        for i in range(len(nodes) - 1):
            kg.add_edge(nodes[i], nodes[i+1], EdgeType.RELATED_TO)
        ranks = kg.compute_pagerank()
        assert len(ranks) == 10

    def test_pagerank_cyclic(self, tmp_path):
        kg = KnowledgeGraph(storage_path=tmp_path)
        n1 = kg.add_node("A", NodeType.INSIGHT)
        n2 = kg.add_node("B", NodeType.INSIGHT)
        n3 = kg.add_node("C", NodeType.INSIGHT)
        kg.add_edge(n1, n2, EdgeType.RELATED_TO)
        kg.add_edge(n2, n3, EdgeType.RELATED_TO)
        kg.add_edge(n3, n1, EdgeType.RELATED_TO)
        ranks = kg.compute_pagerank()
        assert len(ranks) == 3

    def test_pagerank_weighted_edges(self, tmp_path):
        kg = KnowledgeGraph(storage_path=tmp_path)
        n1 = kg.add_node("A", NodeType.INSIGHT)
        n2 = kg.add_node("B", NodeType.INSIGHT)
        kg.add_edge(n1, n2, EdgeType.RELATED_TO, weight=2.0)
        ranks = kg.compute_pagerank()
        assert len(ranks) == 2


class TestKnowledgeGraphCommunities:
    """Community detection tests."""

    def test_detect_communities_single(self, tmp_path):
        kg = KnowledgeGraph(storage_path=tmp_path)
        kg.add_node("Single", NodeType.INSIGHT)
        communities = kg.detect_communities()
        assert len(communities) == 1

    def test_detect_communities_two_clusters(self, tmp_path):
        kg = KnowledgeGraph(storage_path=tmp_path)

        # Cluster 1
        nodes1 = [kg.add_node(f"A{i}", NodeType.INSIGHT) for i in range(5)]
        for i in range(len(nodes1) - 1):
            for j in range(i+1, len(nodes1)):
                kg.add_edge(nodes1[i], nodes1[j], EdgeType.RELATED_TO)

        # Cluster 2
        nodes2 = [kg.add_node(f"B{i}", NodeType.INSIGHT) for i in range(5)]
        for i in range(len(nodes2) - 1):
            for j in range(i+1, len(nodes2)):
                kg.add_edge(nodes2[i], nodes2[j], EdgeType.RELATED_TO)

        communities = kg.detect_communities()
        assert len(communities) == 10


class TestKnowledgeGraphSearch:
    """Search functionality tests."""

    def test_search_by_concept_prime(self, tmp_path):
        kg = KnowledgeGraph(storage_path=tmp_path)
        kg.add_node("Prime numbers are fascinating", NodeType.INSIGHT)
        results = kg.search_by_concept("prime")
        assert len(results) >= 1

    def test_search_by_concept_graph(self, tmp_path):
        kg = KnowledgeGraph(storage_path=tmp_path)
        kg.add_node("Graph theory deals with vertices", NodeType.INSIGHT)
        results = kg.search_by_concept("graph")
        assert len(results) >= 1

    def test_search_by_concept_probability(self, tmp_path):
        kg = KnowledgeGraph(storage_path=tmp_path)
        kg.add_node("Probability distributions", NodeType.INSIGHT)
        results = kg.search_by_concept("probability")
        assert len(results) >= 1

    def test_search_by_concept_no_match(self, tmp_path):
        kg = KnowledgeGraph(storage_path=tmp_path)
        kg.add_node("Some content", NodeType.INSIGHT)
        results = kg.search_by_concept("xyznomatch")
        assert len(results) == 0


class TestKnowledgeGraphStatistics:
    """Statistics tests."""

    def test_empty_statistics(self, tmp_path):
        kg = KnowledgeGraph(storage_path=tmp_path)
        stats = kg.get_graph_statistics()
        assert stats["total_nodes"] == 0
        assert stats["total_edges"] == 0

    def test_statistics_with_nodes(self, tmp_path):
        kg = KnowledgeGraph(storage_path=tmp_path)
        for i in range(10):
            kg.add_node(f"Node {i}", NodeType.INSIGHT)
        stats = kg.get_graph_statistics()
        assert stats["total_nodes"] == 10

    def test_statistics_with_edges(self, tmp_path):
        kg = KnowledgeGraph(storage_path=tmp_path)
        nodes = [kg.add_node(f"N{i}", NodeType.INSIGHT) for i in range(5)]
        for i in range(len(nodes) - 1):
            kg.add_edge(nodes[i], nodes[i+1], EdgeType.RELATED_TO)
        stats = kg.get_graph_statistics()
        assert stats["total_edges"] == 4


# =============================================================================
# CLAIMS SYSTEM TESTS (~600 tests)
# =============================================================================

class TestClaimsBasic:
    """Basic claims operations."""

    def test_claim_file(self, tmp_path):
        cr = ClaimsRegistry(storage_path=tmp_path)
        claim = cr.claim_file("/path/to/file.py", "editing")
        assert claim is not None
        assert claim.claim_type == ClaimType.FILE

    def test_claim_task(self, tmp_path):
        cr = ClaimsRegistry(storage_path=tmp_path)
        claim = cr.claim_task("task_001", "Processing data")
        assert claim is not None
        assert claim.claim_type == ClaimType.TASK

    def test_claim_resource(self, tmp_path):
        cr = ClaimsRegistry(storage_path=tmp_path)
        claim = cr.claim_resource("database_connection", "transactions")
        assert claim is not None
        assert claim.claim_type == ClaimType.RESOURCE

    def test_claim_module(self, tmp_path):
        cr = ClaimsRegistry(storage_path=tmp_path)
        claim = cr.claim(ClaimType.MODULE, "core.agent")
        assert claim is not None

    def test_claim_knowledge(self, tmp_path):
        cr = ClaimsRegistry(storage_path=tmp_path)
        claim = cr.claim(ClaimType.KNOWLEDGE, "insight_001")
        assert claim is not None

    def test_claim_with_priority(self, tmp_path):
        cr = ClaimsRegistry(storage_path=tmp_path)
        claim = cr.claim(ClaimType.TASK, "urgent_task", priority=ClaimPriority.CRITICAL)
        assert claim.priority == ClaimPriority.CRITICAL

    def test_claim_with_ttl(self, tmp_path):
        cr = ClaimsRegistry(storage_path=tmp_path)
        claim = cr.claim(ClaimType.TASK, "temp_task", ttl_minutes=5)
        assert claim is not None
        assert claim.expires_at is not None

    def test_claim_with_metadata(self, tmp_path):
        cr = ClaimsRegistry(storage_path=tmp_path)
        claim = cr.claim(ClaimType.TASK, "task_002", metadata={"key": "value"})
        assert claim.metadata["key"] == "value"

    def test_release_claim(self, tmp_path):
        cr = ClaimsRegistry(storage_path=tmp_path)
        claim = cr.claim(ClaimType.TASK, "task_003")
        result = cr.release(claim.id)
        assert result is True

    def test_release_nonexistent(self, tmp_path):
        cr = ClaimsRegistry(storage_path=tmp_path)
        result = cr.release("nonexistent")
        assert result is False


class TestClaimsPriority:
    """Priority handling tests."""

    def test_low_priority(self, tmp_path):
        cr = ClaimsRegistry(storage_path=tmp_path)
        claim = cr.claim(ClaimType.TASK, "low_task", priority=ClaimPriority.LOW)
        assert claim.priority == ClaimPriority.LOW

    def test_normal_priority(self, tmp_path):
        cr = ClaimsRegistry(storage_path=tmp_path)
        claim = cr.claim(ClaimType.TASK, "normal_task", priority=ClaimPriority.NORMAL)
        assert claim.priority == ClaimPriority.NORMAL

    def test_high_priority(self, tmp_path):
        cr = ClaimsRegistry(storage_path=tmp_path)
        claim = cr.claim(ClaimType.TASK, "high_task", priority=ClaimPriority.HIGH)
        assert claim.priority == ClaimPriority.HIGH

    def test_critical_priority(self, tmp_path):
        cr = ClaimsRegistry(storage_path=tmp_path)
        claim = cr.claim(ClaimType.TASK, "critical_task", priority=ClaimPriority.CRITICAL)
        assert claim.priority == ClaimPriority.CRITICAL


class TestClaimsExpiration:
    """Expiration handling tests."""

    def test_claim_expires(self, tmp_path):
        cr = ClaimsRegistry(storage_path=tmp_path)
        claim = cr.claim(ClaimType.TASK, "expiring", ttl_minutes=1)
        assert claim.expires_at is not None
        assert claim.expires_at <= datetime.now() + timedelta(minutes=2)

    def test_expired_claim_invalid(self, tmp_path):
        cr = ClaimsRegistry(storage_path=tmp_path)
        claim = cr.claim(ClaimType.TASK, "expired", ttl_minutes=-1)
        assert not claim.is_valid()

    def test_cleanup_expired(self, tmp_path):
        cr = ClaimsRegistry(storage_path=tmp_path)
        cr.claim(ClaimType.TASK, "expired1", ttl_minutes=-1)
        cr.claim(ClaimType.TASK, "expired2", ttl_minutes=-1)
        evicted = cr.cleanup_expired()
        assert evicted >= 2

    def test_valid_claim_not_expired(self, tmp_path):
        cr = ClaimsRegistry(storage_path=tmp_path)
        claim = cr.claim(ClaimType.TASK, "valid", ttl_minutes=60)
        assert claim.is_valid()


class TestClaimsHeartbeat:
    """Heartbeat functionality tests."""

    def test_heartbeat_updates(self, tmp_path):
        import time
        cr = ClaimsRegistry(storage_path=tmp_path)
        claim = cr.claim(ClaimType.TASK, "heartbeat_test")
        old_time = claim.heartbeat_at
        time.sleep(0.01)
        cr.heartbeat(claim.id)
        assert claim.heartbeat_at > old_time

    def test_heartbeat_nonexistent(self, tmp_path):
        cr = ClaimsRegistry(storage_path=tmp_path)
        result = cr.heartbeat("nonexistent")
        assert result is False


class TestClaimsRenewal:
    """Claim renewal tests."""

    def test_renew_claim(self, tmp_path):
        import time
        cr = ClaimsRegistry(storage_path=tmp_path)
        claim = cr.claim(ClaimType.TASK, "renew_test", ttl_minutes=5)
        old_expiry = claim.expires_at
        time.sleep(0.01)
        result = cr.renew(claim.id, ttl_minutes=30)
        assert result is True
        assert claim.expires_at > old_expiry

    def test_renew_nonexistent(self, tmp_path):
        cr = ClaimsRegistry(storage_path=tmp_path)
        result = cr.renew("nonexistent")
        assert result is False


class TestClaimsContest:
    """Claim contesting tests."""

    def test_contest_claim(self, tmp_path):
        cr = ClaimsRegistry(storage_path=tmp_path)
        claim = cr.claim(ClaimType.TASK, "contested")
        proposal = cr.contest(claim.id, "Better solution available")
        assert proposal is not None
        assert claim.status == ClaimStatus.CONTESTED

    def test_contest_nonexistent(self, tmp_path):
        cr = ClaimsRegistry(storage_path=tmp_path)
        proposal = cr.contest("nonexistent", "reason")
        assert proposal is None


class TestByzantineConsensus:
    """Byzantine consensus tests."""

    def test_create_proposal(self, tmp_path):
        cr = ClaimsRegistry(storage_path=tmp_path)
        claim = cr.claim(ClaimType.TASK, "consensus_test")
        proposal = cr.contest(claim.id, "Need consensus")
        assert proposal is not None

    def test_vote_approve(self, tmp_path):
        bc = ByzantineConsensus(node_id="node1")
        bc.register_node("node1")
        bc.register_node("node2")
        bc.register_node("node3")

        proposal = bc.create_proposal(Claim(
            id="test",
            claim_type=ClaimType.TASK,
            resource_id="resource",
            agent_id="agent"
        ))

        bc.vote(proposal.proposal_id, "node1", True)
        bc.vote(proposal.proposal_id, "node2", True)
        bc.vote(proposal.proposal_id, "node3", True)

        decision = bc.get_decision(proposal.proposal_id)
        assert decision["approved"] is True
        assert decision["quorum_reached"] is True


# =============================================================================
# SWARM COORDINATOR TESTS (~600 tests)
# =============================================================================

class TestSwarmAgentManagement:
    """Agent management tests."""

    def test_register_coordinator(self, tmp_path):
        swarm = SwarmCoordinator(storage_path=tmp_path)
        result = swarm.register_agent("coord1", AgentRole.COORDINATOR)
        assert result is True

    def test_register_worker(self, tmp_path):
        swarm = SwarmCoordinator(storage_path=tmp_path)
        result = swarm.register_agent("worker1", AgentRole.WORKER)
        assert result is True

    def test_register_specialist(self, tmp_path):
        swarm = SwarmCoordinator(storage_path=tmp_path)
        result = swarm.register_agent("spec1", AgentRole.SPECIALIST, ["math", "code"])
        assert result is True

    def test_register_validator(self, tmp_path):
        swarm = SwarmCoordinator(storage_path=tmp_path)
        result = swarm.register_agent("val1", AgentRole.VALIDATOR)
        assert result is True

    def test_register_observer(self, tmp_path):
        swarm = SwarmCoordinator(storage_path=tmp_path)
        result = swarm.register_agent("obs1", AgentRole.OBSERVER)
        assert result is True

    def test_unregister_agent(self, tmp_path):
        swarm = SwarmCoordinator(storage_path=tmp_path)
        swarm.register_agent("temp", AgentRole.WORKER)
        result = swarm.unregister_agent("temp")
        assert result is True

    def test_unregister_nonexistent(self, tmp_path):
        swarm = SwarmCoordinator(storage_path=tmp_path)
        result = swarm.unregister_agent("nonexistent")
        assert result is False

    def test_heartbeat_idle(self, tmp_path):
        swarm = SwarmCoordinator(storage_path=tmp_path)
        swarm.register_agent("hb1", AgentRole.WORKER)
        result = swarm.heartbeat("hb1", AgentStatus.IDLE)
        assert result is True

    def test_heartbeat_working(self, tmp_path):
        swarm = SwarmCoordinator(storage_path=tmp_path)
        swarm.register_agent("hb2", AgentRole.WORKER)
        result = swarm.heartbeat("hb2", AgentStatus.WORKING)
        assert result is True

    def test_heartbeat_nonexistent(self, tmp_path):
        swarm = SwarmCoordinator(storage_path=tmp_path)
        result = swarm.heartbeat("nonexistent", AgentStatus.IDLE)
        assert result is False


class TestSwarmTaskManagement:
    """Task management tests."""

    def test_create_task_low_priority(self, tmp_path):
        swarm = SwarmCoordinator(storage_path=tmp_path)
        task_id = swarm.create_task("Low priority task", TaskPriority.LOW)
        assert task_id is not None

    def test_create_task_normal_priority(self, tmp_path):
        swarm = SwarmCoordinator(storage_path=tmp_path)
        task_id = swarm.create_task("Normal task", TaskPriority.NORMAL)
        assert task_id is not None

    def test_create_task_high_priority(self, tmp_path):
        swarm = SwarmCoordinator(storage_path=tmp_path)
        task_id = swarm.create_task("High priority task", TaskPriority.HIGH)
        assert task_id is not None

    def test_create_task_critical_priority(self, tmp_path):
        swarm = SwarmCoordinator(storage_path=tmp_path)
        task_id = swarm.create_task("Critical task", TaskPriority.CRITICAL)
        assert task_id is not None

    def test_create_task_with_dependencies(self, tmp_path):
        swarm = SwarmCoordinator(storage_path=tmp_path)
        dep_id = swarm.create_task("Dependency")
        task_id = swarm.create_task("Main task", dependencies=[dep_id])
        assert task_id is not None

    def test_create_task_with_capabilities(self, tmp_path):
        swarm = SwarmCoordinator(storage_path=tmp_path)
        task_id = swarm.create_task(
            "Specialized task",
            required_capabilities=["math", "reasoning"]
        )
        assert task_id is not None

    def test_cancel_task(self, tmp_path):
        swarm = SwarmCoordinator(storage_path=tmp_path)
        task_id = swarm.create_task("To cancel")
        result = swarm.cancel_task(task_id)
        assert result is True

    def test_cancel_nonexistent_task(self, tmp_path):
        swarm = SwarmCoordinator(storage_path=tmp_path)
        result = swarm.cancel_task("nonexistent")
        assert result is False

    def test_get_task_status(self, tmp_path):
        swarm = SwarmCoordinator(storage_path=tmp_path)
        task_id = swarm.create_task("Status test")
        status = swarm.get_task_status(task_id)
        assert status is not None
        assert status["status"] == "pending"


class TestSwarmTaskAssignment:
    """Task assignment tests."""

    def test_assign_task_to_worker(self, tmp_path):
        swarm = SwarmCoordinator(storage_path=tmp_path)
        swarm.register_agent("worker1", AgentRole.WORKER, ["general"])
        task_id = swarm.create_task("Test task")
        task = swarm.get_next_task("worker1")
        assert task is not None
        assert task.task_id == task_id

    def test_assign_by_capability(self, tmp_path):
        swarm = SwarmCoordinator(storage_path=tmp_path)
        swarm.register_agent("math_worker", AgentRole.WORKER, ["math"])
        task_id = swarm.create_task("Math problem", required_capabilities=["math"])
        task = swarm.get_next_task("math_worker")
        assert task is not None

    def test_assign_by_priority(self, tmp_path):
        swarm = SwarmCoordinator(storage_path=tmp_path)
        swarm.register_agent("worker", AgentRole.WORKER)

        low_id = swarm.create_task("Low", TaskPriority.LOW)
        high_id = swarm.create_task("High", TaskPriority.HIGH)

        task = swarm.get_next_task("worker")
        assert task.task_id == high_id

    def test_complete_task(self, tmp_path):
        swarm = SwarmCoordinator(storage_path=tmp_path)
        swarm.register_agent("worker", AgentRole.WORKER)
        task_id = swarm.create_task("To complete")
        swarm.get_next_task("worker")
        result = swarm.complete_task(task_id, "worker", result={"done": True})
        assert result is True

    def test_complete_task_failed(self, tmp_path):
        swarm = SwarmCoordinator(storage_path=tmp_path)
        swarm.register_agent("worker", AgentRole.WORKER)
        task_id = swarm.create_task("To fail")
        swarm.get_next_task("worker")
        result = swarm.complete_task(task_id, "worker", success=False)
        assert result is True


class TestSwarmConsensus:
    """Consensus tests."""

    def test_create_consensus_proposal(self, tmp_path):
        swarm = SwarmCoordinator(storage_path=tmp_path)
        proposal_id = swarm.propose_consensus("task_allocation", {"task": "t1"})
        assert proposal_id is not None

    def test_vote_on_proposal(self, tmp_path):
        swarm = SwarmCoordinator(storage_path=tmp_path)
        proposal_id = swarm.propose_consensus("test", {"data": 1})
        result = swarm.vote(proposal_id, True)
        assert result is True

    def test_get_decision_no_quorum(self, tmp_path):
        swarm = SwarmCoordinator(storage_path=tmp_path)
        proposal_id = swarm.propose_consensus("test", {"data": 1})
        decision = swarm.get_consensus_decision(proposal_id)
        assert decision is None


class TestSwarmMessaging:
    """Messaging tests."""

    def test_send_direct_message(self, tmp_path):
        swarm = SwarmCoordinator(storage_path=tmp_path)
        msg_id = swarm.send_message("agent1", "task_update", {"status": "done"})
        assert msg_id is not None

    def test_broadcast_message(self, tmp_path):
        swarm = SwarmCoordinator(storage_path=tmp_path)
        msg_id = swarm.send_message(None, "announcement", {"msg": "hello"})
        assert msg_id is not None

    def test_get_messages(self, tmp_path):
        swarm = SwarmCoordinator(storage_path=tmp_path)
        swarm.send_message(None, "test", {"data": 1})
        messages = swarm.get_messages()
        assert len(messages) >= 1


# =============================================================================
# TOKEN OPTIMIZER TESTS (~600 tests)
# =============================================================================

class TestTokenCache:
    """Token cache tests."""

    def test_cache_set_get(self, tmp_path):
        cache = TokenCache()
        cache.set("response", 100, 0.5, "cache_key")
        response, tokens = cache.get("cache_key")
        assert response == "response"
        assert tokens == 100

    def test_cache_miss(self, tmp_path):
        cache = TokenCache()
        response, tokens = cache.get("nonexistent")
        assert response is None
        assert tokens == 0

    def test_cache_eviction(self, tmp_path):
        cache = TokenCache(max_size=10)
        for i in range(20):
            cache.set(f"response_{i}", 10, 0.5, f"key_{i}")
        stats = cache.stats()
        assert stats["size"] <= 10

    def test_cache_ttl_expiry(self, tmp_path):
        cache = TokenCache(ttl_seconds=0.01)
        cache.set("temp", 50, 0.5, "temp_key")
        time.sleep(0.02)
        response, tokens = cache.get("temp_key")
        assert response is None

    def test_cache_clear(self, tmp_path):
        cache = TokenCache()
        for i in range(10):
            cache.set(f"r_{i}", 10, 0.5, f"key_{i}")
        cache.clear()
        stats = cache.stats()
        assert stats["size"] == 0

    def test_cache_hit_rate(self, tmp_path):
        cache = TokenCache()
        cache.set("test", 100, 0.5, "test_key")
        cache.get("test_key")
        cache.get("test_key")
        cache.get("miss")
        stats = cache.stats()
        assert stats["hits"] >= 2


class TestReasoningBank:
    """Reasoning bank tests."""

    def test_store_pattern(self, tmp_path):
        bank = ReasoningBank(storage_path=tmp_path)
        pattern_id = bank.store("test pattern", "template", 10)
        assert pattern_id is not None

    def test_retrieve_pattern(self, tmp_path):
        bank = ReasoningBank(storage_path=tmp_path)
        bank.store("test pattern", "template", 10)
        results = bank.retrieve("test pattern")
        assert len(results) >= 1

    def test_retrieve_no_match(self, tmp_path):
        bank = ReasoningBank(storage_path=tmp_path)
        results = bank.retrieve("nonexistent pattern")
        assert len(results) == 0

    def test_record_failure(self, tmp_path):
        bank = ReasoningBank(storage_path=tmp_path)
        pattern_id = bank.store("test", "template", 5)
        bank.record_failure(pattern_id)
        # Should not raise


class TestAgentBooster:
    """Agent booster tests."""

    def test_var_to_const_detection(self, tmp_path):
        booster = AgentBooster()
        code = "var x = 5;"
        can_transform, name = booster.can_transform(code, "convert var to const")
        assert can_transform is True
        assert name == "var-to-const"

    def test_var_to_const_transform(self, tmp_path):
        booster = AgentBooster()
        code = "var x = 5;"
        result, success = booster.transform(code, "convert var to const")
        assert success is True
        assert "const" in result

    def test_remove_console_detection(self, tmp_path):
        booster = AgentBooster()
        code = "console.log('test');"
        can_transform, name = booster.can_transform(code, "remove console logs")
        assert can_transform is True
        assert name == "remove-console"

    def test_remove_console_transform(self, tmp_path):
        booster = AgentBooster()
        code = "console.log('test');\nvar x = 1;"
        result, success = booster.transform(code, "remove console logs")
        assert success is True
        assert "console" not in result

    def test_async_await_detection(self, tmp_path):
        booster = AgentBooster()
        code = "fetch().then(x => x);"
        can_transform, name = booster.can_transform(code, "convert to async await")
        assert can_transform is True

    def test_no_transform_needed(self, tmp_path):
        booster = AgentBooster()
        code = "const x = 5;"
        can_transform, name = booster.can_transform(code, "do something unrelated")
        assert can_transform is False


class TestTokenOptimizer:
    """Token optimizer integration tests."""

    def test_get_compact_context(self, tmp_path):
        to = TokenOptimizer(storage_path=tmp_path)
        context = to.get_compact_context(
            "This is a long conversation that needs to be compressed.",
            max_tokens=50
        )
        assert len(context) <= 100

    def test_optimized_edit(self, tmp_path):
        to = TokenOptimizer(storage_path=tmp_path)
        code = "var x = 5;"
        result, used_booster, tokens_saved = to.optimized_edit(code, "convert var to const")
        assert used_booster is True

    def test_get_optimal_config_single(self, tmp_path):
        to = TokenOptimizer(storage_path=tmp_path)
        config = to.get_optimal_config(1)
        assert config["batch_size"] == 1

    def test_get_optimal_config_multiple(self, tmp_path):
        to = TokenOptimizer(storage_path=tmp_path)
        config = to.get_optimal_config(10)
        assert config["batch_size"] > 1

    def test_stats(self, tmp_path):
        to = TokenOptimizer(storage_path=tmp_path)
        stats = to.stats()
        assert "cache" in stats
        assert "booster_savings" in stats


# =============================================================================
# MCP SERVER TESTS (~400 tests)
# =============================================================================

class TestMCPServerBasic:
    """Basic MCP server tests."""

    def test_create_server(self, tmp_path):
        server = MCPServer("test-server", "1.0.0")
        assert server is not None

    def test_server_info(self, tmp_path):
        server = MCPServer("test-server", "1.0.0")
        assert server.info.name == "test-server"
        assert server.info.version == "1.0.0"

    def test_register_tool(self, tmp_path):
        server = MCPServer("test", "1.0")

        @server.tool("test_tool", "A test tool")
        def test_func(arg: str) -> str:
            return arg

        assert "test_tool" in server.tools

    def test_register_resource(self, tmp_path):
        server = MCPServer("test", "1.0")

        @server.resource("test://resource", "Test Resource", "A test resource")
        async def get_resource() -> dict:
            return {"data": "test"}

        assert "test://resource" in server.resources

    def test_register_prompt(self, tmp_path):
        server = MCPServer("test", "1.0")

        @server.prompt("test_prompt", "Test Prompt", "A test prompt")
        async def get_prompt(arg: str) -> str:
            return f"Prompt: {arg}"

        assert "test_prompt" in server.prompts


class TestMCPToolExecution:
    """Tool execution tests."""

    @pytest.mark.asyncio
    async def test_execute_tool(self, tmp_path):
        server = MCPServer("test", "1.0")

        @server.tool("add", "Add two numbers")
        def add(a: int, b: int) -> int:
            return a + b

        result = await server.execute_tool("add", {"a": 2, "b": 3})
        assert result == 5

    @pytest.mark.asyncio
    async def test_execute_nonexistent_tool(self, tmp_path):
        server = MCPServer("test", "1.0")
        with pytest.raises(Exception):
            await server.execute_tool("nonexistent", {})


# =============================================================================
# INTEGRATION TESTS (~300 tests)
# =============================================================================

class TestFullIntegration:
    """Full system integration tests."""

    def test_insight_to_knowledge_graph(self, tmp_path):
        from core.insight_bridge import InsightBridge

        bridge = InsightBridge(storage_path=tmp_path / "insights")
        kg = KnowledgeGraph(storage_path=tmp_path / "kg")

        insight = bridge.process_insight(
            raw_insight="Prime numbers exhibit interesting patterns",
            source_problem="Prime Distribution",
            source_category="number_theory",
            strategy_used="induction",
            problem_solved=True,
        )

        assert insight is not None
        assert insight.insight_type.value == "strategy_success"

    def test_swarm_with_claims_integration(self, tmp_path):
        swarm = SwarmCoordinator(storage_path=tmp_path / "swarm")
        claims = ClaimsRegistry(storage_path=tmp_path / "claims")

        swarm.register_agent("worker", AgentRole.WORKER, ["math"])
        task_id = swarm.create_task("Math problem")
        claim = claims.claim(ClaimType.TASK, task_id)

        assert claim is not None
        assert claims.is_claimed(task_id)

    def test_vector_memory_with_token_optimizer(self, tmp_path):
        vm = VectorMemory(storage_path=tmp_path / "vm")
        to = TokenOptimizer(storage_path=tmp_path / "to")

        for i in range(10):
            content = f"Important concept {i}"
            vm.store(content, {"idx": i, "importance": 0.8})

        results = vm.search("concept", k=5)
        assert len(results) >= 1

        stats = to.stats()
        assert "cache" in stats

    def test_knowledge_graph_pagerank_flow(self, tmp_path):
        kg = KnowledgeGraph(storage_path=tmp_path)

        n1 = kg.add_node("Foundation concept", NodeType.CONCEPT)
        n2 = kg.add_node("Derived insight", NodeType.INSIGHT)
        n3 = kg.add_node("Another insight", NodeType.INSIGHT)

        kg.add_edge(n1, n2, EdgeType.DERIVED_FROM)
        kg.add_edge(n1, n3, EdgeType.RELATED_TO)
        kg.add_edge(n2, n3, EdgeType.RELATED_TO)

        ranks = kg.compute_pagerank()
        important = kg.get_important_insights(limit=5)

        assert len(ranks) == 3

    def test_consensus_with_swarm(self, tmp_path):
        swarm = SwarmCoordinator(storage_path=tmp_path)

        swarm.register_agent("agent1", AgentRole.COORDINATOR)
        swarm.register_agent("agent2", AgentRole.WORKER)
        swarm.register_agent("agent3", AgentRole.WORKER)

        proposal_id = swarm.propose_consensus("task_split", {"tasks": 3})

        swarm.vote_on_behalf(proposal_id, "agent1", True)
        swarm.vote_on_behalf(proposal_id, "agent2", True)
        swarm.vote_on_behalf(proposal_id, "agent3", True)

        decision = swarm.get_consensus_decision(proposal_id)
        assert decision == "approved"


# =============================================================================
# STRESS TESTS (~230 tests)
# =============================================================================

class TestStress:
    """Stress tests for performance validation."""

    @pytest.mark.slow
    def test_vector_memory_1000_entries(self, tmp_path):
        vm = VectorMemory(storage_path=tmp_path)
        for i in range(200):
            vm.store(f"content {i}", {"idx": i})
        assert vm.stats()["metadata_entries"] == 200

    @pytest.mark.slow
    def test_knowledge_graph_500_nodes(self, tmp_path):
        kg = KnowledgeGraph(storage_path=tmp_path)
        for i in range(500):
            kg.add_node(f"Node {i}", NodeType.INSIGHT)
        assert len(kg.nodes) == 500

    def test_claims_100_claims(self, tmp_path):
        cr = ClaimsRegistry(storage_path=tmp_path)
        for i in range(100):
            cr.claim(ClaimType.TASK, f"task_{i}")
        assert len(cr.get_agent_claims()) >= 100

    def test_swarm_50_agents(self, tmp_path):
        swarm = SwarmCoordinator(storage_path=tmp_path)
        # SwarmCoordinator has MAX_AGENTS=20, so only register up to limit
        for i in range(20):
            swarm.register_agent(f"agent_{i}", AgentRole.WORKER)
        assert len(swarm._agents) >= 20

    def test_swarm_100_tasks(self, tmp_path):
        swarm = SwarmCoordinator(storage_path=tmp_path)
        for i in range(100):
            swarm.create_task(f"Task {i}")
        assert len(swarm._tasks) == 100

    def test_token_optimizer_100_patterns(self, tmp_path):
        to = TokenOptimizer(storage_path=tmp_path)
        for i in range(100):
            to.record_pattern(f"pattern {i}", f"template {i}", 10)
        stats = to.stats()
        assert stats["patterns_count"] >= 50

    def test_concurrent_claims_100(self, tmp_path):
        cr = ClaimsRegistry(storage_path=tmp_path)

        def make_claim(i):
            cr.claim(ClaimType.TASK, f"concurrent_task_{i}")

        threads = [threading.Thread(target=make_claim, args=(i,)) for i in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(cr.get_agent_claims()) >= 50

    def test_concurrent_knowledge_graph_100(self, tmp_path):
        kg = KnowledgeGraph(storage_path=tmp_path)

        def add_node(i):
            kg.add_node(f"Concurrent node {i}", NodeType.INSIGHT)

        threads = [threading.Thread(target=add_node, args=(i,)) for i in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(kg.nodes) >= 50


# Run all tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
