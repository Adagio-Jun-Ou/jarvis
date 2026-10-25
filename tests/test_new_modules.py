"""Tests for Knowledge Graph, Claims System, and Swarm Coordinator."""
import threading
import time
from datetime import datetime, timedelta

from core.knowledge_graph import (
    KnowledgeGraph,
    NodeType,
    EdgeType,
)
from core.claims import (
    ClaimsRegistry,
    ClaimType,
    ClaimStatus,
)
from core.swarm import (
    SwarmCoordinator,
    AgentRole,
    AgentStatus,
    TaskStatus,
    TaskPriority,
)


class TestKnowledgeGraph:
    def test_add_node(self, tmp_path):
        kg = KnowledgeGraph(storage_path=tmp_path)

        node_id = kg.add_node("Test insight about primes", NodeType.INSIGHT)

        assert node_id.startswith("insight_")
        assert node_id in kg.nodes
        assert kg.nodes[node_id].content == "Test insight about primes"

    def test_add_edge(self, tmp_path):
        kg = KnowledgeGraph(storage_path=tmp_path)

        node1 = kg.add_node("Insight 1", NodeType.INSIGHT)
        node2 = kg.add_node("Insight 2", NodeType.INSIGHT)

        result = kg.add_edge(node1, node2, EdgeType.RELATED_TO, 0.8)

        assert result is True
        assert len(kg.edge_details) == 1

    def test_pagerank(self, tmp_path):
        kg = KnowledgeGraph(storage_path=tmp_path)

        n1 = kg.add_node("Node 1", NodeType.INSIGHT)
        n2 = kg.add_node("Node 2", NodeType.INSIGHT)
        n3 = kg.add_node("Node 3", NodeType.INSIGHT)

        kg.add_edge(n1, n2, EdgeType.RELATED_TO)
        kg.add_edge(n2, n3, EdgeType.RELATED_TO)
        kg.add_edge(n3, n1, EdgeType.RELATED_TO)

        ranks = kg.compute_pagerank()

        assert len(ranks) == 3
        assert all(0 <= r <= 1 for r in ranks.values())

    def test_community_detection(self, tmp_path):
        kg = KnowledgeGraph(storage_path=tmp_path)

        nodes = [kg.add_node(f"Node {i}", NodeType.INSIGHT) for i in range(10)]

        for i in range(5):
            for j in range(i + 1, 5):
                kg.add_edge(nodes[i], nodes[j], EdgeType.RELATED_TO)

        for i in range(5, 10):
            for j in range(i + 1, 10):
                kg.add_edge(nodes[i], nodes[j], EdgeType.RELATED_TO)

        communities = kg.detect_communities()

        assert len(communities) == 10

    def test_add_insight_with_relations(self, tmp_path):
        kg = KnowledgeGraph(storage_path=tmp_path)

        insight_id = kg.add_insight(
            insight_id="test_001",
            content="Prime numbers show interesting patterns",
            problem_id="Twin Prime Conjecture",
            strategy="exhaustive_search",
            category="number_theory",
        )

        assert insight_id.startswith("insight_")
        assert len(kg.nodes) > 1

    def test_get_important_insights(self, tmp_path):
        kg = KnowledgeGraph(storage_path=tmp_path)

        for i in range(10):
            kg.add_node(f"Insight {i}", NodeType.INSIGHT)

        important = kg.get_important_insights(limit=5)

        assert len(important) <= 5

    def test_search_by_concept(self, tmp_path):
        kg = KnowledgeGraph(storage_path=tmp_path)

        kg.add_node("Prime numbers are fascinating", NodeType.INSIGHT)
        kg.add_node("Graph theory applications", NodeType.INSIGHT)

        results = kg.search_by_concept("prime")

        assert len(results) >= 1

    def test_thread_safety(self, tmp_path):
        kg = KnowledgeGraph(storage_path=tmp_path)
        kg.SIMILARITY_THRESHOLD = 0.99
        kg._interconnect = False

        def add_nodes():
            for i in range(100):
                kg.add_node(f"Node {threading.current_thread().name}-{i}-{time.time()}", NodeType.INSIGHT)

        threads = [threading.Thread(target=add_nodes) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        alive = [t for t in threads if t.is_alive()]
        assert len(kg.nodes) >= 100
        assert len(alive) == 0, f"{len(alive)} threads still alive (possible deadlock)"

    def test_node_eviction(self, tmp_path):
        kg = KnowledgeGraph(storage_path=tmp_path)

        for i in range(30):
            kg.add_node(f"UniqueNode{i}xyz", NodeType.INSIGHT)

        kg._evict_low_pagerank_nodes(10)

        assert len(kg.nodes) == 20


class TestClaimsRegistry:
    def test_claim_resource(self, tmp_path):
        registry = ClaimsRegistry(storage_path=tmp_path)

        claim = registry.claim(ClaimType.FILE, "/path/to/file.py")

        assert claim is not None
        assert claim.resource_id == "/path/to/file.py"
        assert claim.status == ClaimStatus.ACTIVE

    def test_claim_already_claimed(self, tmp_path):
        registry = ClaimsRegistry(storage_path=tmp_path)

        claim1 = registry.claim(ClaimType.TASK, "task_001")

        claim2 = registry.claim(ClaimType.TASK, "task_001")

        assert claim1 is not None
        assert claim2 is not None
        assert claim1.id == claim2.id

    def test_release_claim(self, tmp_path):
        registry = ClaimsRegistry(storage_path=tmp_path)

        claim = registry.claim(ClaimType.RESOURCE, "database_connection")
        result = registry.release(claim.id)

        assert result is True
        assert registry.get_claim(claim.id) is None

    def test_claim_expiration(self, tmp_path):
        registry = ClaimsRegistry(storage_path=tmp_path)

        claim = registry.claim(
            ClaimType.TASK,
            "task_002",
            ttl_minutes=-1
        )

        assert claim.expires_at < datetime.now()
        assert not claim.is_valid()

    def test_heartbeat(self, tmp_path):
        registry = ClaimsRegistry(storage_path=tmp_path)

        claim = registry.claim(ClaimType.FILE, "test.py")
        old_heartbeat = claim.heartbeat_at

        time.sleep(0.01)
        result = registry.heartbeat(claim.id)

        assert result is True
        assert claim.heartbeat_at > old_heartbeat

    def test_renew_claim(self, tmp_path):
        registry = ClaimsRegistry(storage_path=tmp_path)

        claim = registry.claim(ClaimType.TASK, "task_003", ttl_minutes=5)
        old_expires = claim.expires_at

        result = registry.renew(claim.id, ttl_minutes=30)

        assert result is True
        assert claim.expires_at > old_expires

    def test_is_claimed(self, tmp_path):
        registry = ClaimsRegistry(storage_path=tmp_path)

        assert not registry.is_claimed("resource_001")

        registry.claim(ClaimType.RESOURCE, "resource_001")

        assert registry.is_claimed("resource_001")

    def test_get_agent_claims(self, tmp_path):
        registry = ClaimsRegistry(storage_path=tmp_path)

        registry.claim(ClaimType.TASK, "task_a")
        registry.claim(ClaimType.TASK, "task_b")

        claims = registry.get_agent_claims()

        assert len(claims) >= 2

    def test_contest_claim(self, tmp_path):
        registry = ClaimsRegistry(storage_path=tmp_path)

        claim = registry.claim(ClaimType.TASK, "contested_task")
        proposal = registry.contest(claim.id, "Better solution available")

        assert proposal is not None
        assert claim.status == ClaimStatus.CONTESTED

    def test_claim_helpers(self, tmp_path):
        registry = ClaimsRegistry(storage_path=tmp_path)

        file_claim = registry.claim_file("test.py", "editing")
        task_claim = registry.claim_task("task_001", "Processing data")
        resource_claim = registry.claim_resource("gpu_0", "Training model")

        assert file_claim.claim_type == ClaimType.FILE
        assert task_claim.claim_type == ClaimType.TASK
        assert resource_claim.claim_type == ClaimType.RESOURCE

    def test_thread_safety(self, tmp_path):
        registry = ClaimsRegistry(storage_path=tmp_path)

        def make_claims():
            for i in range(50):
                registry.claim(ClaimType.TASK, f"task_{threading.current_thread().name}-{i}")

        threads = [threading.Thread(target=make_claims) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        claims = registry.get_agent_claims()
        assert len(claims) >= 200


class TestSwarmCoordinator:
    def test_register_agent(self, tmp_path):
        swarm = SwarmCoordinator(storage_path=tmp_path)

        result = swarm.register_agent(
            "agent_001",
            AgentRole.WORKER,
            ["math", "reasoning"]
        )

        assert result is True
        assert "agent_001" in swarm._agents

    def test_unregister_agent(self, tmp_path):
        swarm = SwarmCoordinator(storage_path=tmp_path)

        swarm.register_agent("agent_002", AgentRole.WORKER)
        result = swarm.unregister_agent("agent_002")

        assert result is True
        assert "agent_002" not in swarm._agents

    def test_heartbeat(self, tmp_path):
        swarm = SwarmCoordinator(storage_path=tmp_path)

        swarm.register_agent("agent_003", AgentRole.WORKER)
        result = swarm.heartbeat("agent_003", AgentStatus.IDLE)

        assert result is True
        assert swarm._agents["agent_003"].status == AgentStatus.IDLE

    def test_create_task(self, tmp_path):
        swarm = SwarmCoordinator(storage_path=tmp_path)

        task_id = swarm.create_task("Solve problem X", TaskPriority.HIGH)

        assert task_id.startswith("task_")
        assert task_id in swarm._tasks

    def test_get_next_task(self, tmp_path):
        swarm = SwarmCoordinator(storage_path=tmp_path)

        swarm.register_agent("worker_001", AgentRole.WORKER, ["math"])
        task_id = swarm.create_task(
            "Calculate primes",
            TaskPriority.NORMAL,
            required_capabilities=["math"]
        )

        task = swarm.get_next_task("worker_001")

        assert task is not None
        assert task.task_id == task_id
        assert task.status == TaskStatus.ASSIGNED

    def test_complete_task(self, tmp_path):
        swarm = SwarmCoordinator(storage_path=tmp_path)

        swarm.register_agent("worker_002", AgentRole.WORKER, ["math"])
        task_id = swarm.create_task("Process data")
        swarm.get_next_task("worker_002")

        result = swarm.complete_task(task_id, "worker_002", result={"output": 42})

        assert result is True
        assert swarm._tasks[task_id].status == TaskStatus.COMPLETED
        assert swarm._tasks[task_id].result == {"output": 42}

    def test_cancel_task(self, tmp_path):
        swarm = SwarmCoordinator(storage_path=tmp_path)

        task_id = swarm.create_task("Cancelled task")
        result = swarm.cancel_task(task_id)

        assert result is True
        assert swarm._tasks[task_id].status == TaskStatus.CANCELLED

    def test_task_dependencies(self, tmp_path):
        swarm = SwarmCoordinator(storage_path=tmp_path)

        swarm.register_agent("worker_003", AgentRole.WORKER)

        dep_id = swarm.create_task("Dependency task")
        task_id = swarm.create_task("Main task", dependencies=[dep_id])

        task = swarm.get_next_task("worker_003")
        assert task is not None
        assert task.task_id == dep_id

    def test_consensus(self, tmp_path):
        swarm = SwarmCoordinator(storage_path=tmp_path)

        proposal_id = swarm.propose_consensus("task_allocation", {"task": "important_job"})

        assert proposal_id is not None
        assert swarm.get_consensus_decision(proposal_id) is None

        swarm.vote(proposal_id, True)

    def test_messaging(self, tmp_path):
        swarm = SwarmCoordinator(storage_path=tmp_path)

        msg_id = swarm.send_message(
            "agent_004",
            "task_update",
            {"status": "completed"}
        )

        assert msg_id.startswith("msg_")

        swarm._broadcast_message("announcement", {"message": "hello"})

        messages = swarm.get_messages()
        assert len(messages) >= 2

    def test_cleanup_stale_agents(self, tmp_path):
        swarm = SwarmCoordinator(storage_path=tmp_path)

        swarm.register_agent("stale_agent", AgentRole.WORKER)
        swarm._agents["stale_agent"].last_heartbeat = datetime.now() - timedelta(minutes=5)

        cleaned = swarm.cleanup_stale_agents(timeout_seconds=60)

        assert cleaned >= 1
        assert "stale_agent" not in swarm._agents

    def test_get_swarm_status(self, tmp_path):
        swarm = SwarmCoordinator(storage_path=tmp_path)

        swarm.register_agent("agent_005", AgentRole.WORKER)
        swarm.create_task("Test task")

        status = swarm.get_swarm_status()

        assert "total_agents" in status
        assert "total_tasks" in status
        assert status["total_agents"] >= 1
        assert status["total_tasks"] >= 1

    def test_thread_safety(self, tmp_path):
        swarm = SwarmCoordinator(storage_path=tmp_path)

        def register_agents():
            for i in range(20):
                name = f"agent_{threading.current_thread().name}_{i}"
                swarm.register_agent(name, AgentRole.WORKER)

        def create_tasks():
            for i in range(20):
                swarm.create_task(f"Task {threading.current_thread().name}-{i}")

        threads = [
            threading.Thread(target=register_agents),
            threading.Thread(target=create_tasks),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(swarm._agents) >= 20
        assert len(swarm._tasks) >= 20


class TestEndToEndFlows:
    def test_insight_to_knowledge_flow(self, tmp_path):
        from core.insight_bridge import get_insight_bridge

        bridge = get_insight_bridge()
        bridge.storage_path = tmp_path / "insights"
        bridge.storage_path.mkdir(parents=True, exist_ok=True)

        insight = bridge.process_insight(
            raw_insight="Discovered a pattern in prime number distribution",
            source_problem="Twin Prime Conjecture",
            source_category="number_theory",
            strategy_used="induction",
            problem_solved=True,
        )

        assert insight is not None

        stats = bridge.get_insight_statistics()
        assert stats["total_insights"] >= 1

    def test_vector_memory_operations(self, tmp_path):
        from core.vector_memory import VectorMemory

        vm = VectorMemory(storage_path=tmp_path / "vm_test")

        for i in range(10):
            vm.store(f"content {i}", {"idx": i, "importance": 0.5})

        stats = vm.stats()
        assert stats["metadata_entries"] == 10

        results = vm.search("content", k=5)
        assert len(results) >= 1

    def test_token_optimizer_caching(self, tmp_path):
        from core.token_optimizer import TokenOptimizer

        to = TokenOptimizer(storage_path=tmp_path)

        to.cache_response("test response", 100, 0.5, "cache_key_123")
        cached, tokens = to.get_cached_response("cache_key_123")

        assert cached == "test response"
        assert tokens == 100

    def test_agent_booster_transform(self, tmp_path):
        from core.token_optimizer import AgentBooster

        booster = AgentBooster()

        code = "var x = 5;"
        can_transform, name = booster.can_transform(code, "convert var to const")

        assert can_transform is True
        assert name == "var-to-const"

        result, success = booster.transform(code, "convert var to const")
        assert success is True
        assert "const" in result

    def test_claims_with_expiration(self, tmp_path):
        from core.claims import ClaimsRegistry, ClaimType
        from datetime import datetime, timedelta

        registry = ClaimsRegistry(storage_path=tmp_path)

        claim = registry.claim(
            ClaimType.RESOURCE,
            "gpu_memory",
            ttl_minutes=1
        )

        assert claim is not None
        expires = claim.expires_at
        assert expires is not None
        assert expires <= datetime.now() + timedelta(minutes=2)

        evicted = registry.cleanup_expired()
        assert evicted >= 0

    def test_swarm_task_lifecycle(self, tmp_path):
        from core.swarm import SwarmCoordinator, AgentRole, TaskPriority

        swarm = SwarmCoordinator(storage_path=tmp_path)

        swarm.register_agent("worker_1", AgentRole.WORKER, ["math", "reasoning"])
        swarm.register_agent("worker_2", AgentRole.WORKER, ["code", "testing"])

        task1 = swarm.create_task("Math problem", TaskPriority.HIGH, required_capabilities=["math"])
        task2 = swarm.create_task("Code review", TaskPriority.NORMAL, required_capabilities=["code"])

        t1 = swarm.get_next_task("worker_1")
        assert t1 is not None
        assert t1.task_id == task1

        swarm.complete_task(task1, "worker_1", result={"solution": 42}, success=True)

        swarm.cancel_task(task2)

        status = swarm.get_swarm_status()
        assert status["tasks_by_status"]["completed"] >= 1
        assert status["tasks_by_status"]["cancelled"] >= 1

    def test_knowledge_graph_concepts(self, tmp_path):
        from core.knowledge_graph import KnowledgeGraph, NodeType, EdgeType

        kg = KnowledgeGraph(storage_path=tmp_path)

        n1 = kg.add_node("Prime numbers have unique factorization", NodeType.INSIGHT)
        n2 = kg.add_node("Graph coloring requires minimum colors", NodeType.INSIGHT)

        kg.add_edge(n1, n2, EdgeType.RELATED_TO, 0.5)

        results = kg.search_by_concept("prime")
        assert len(results) >= 1

        results = kg.search_by_concept("graph")
        assert len(results) >= 1

    def test_mcp_tools_registration(self, tmp_path):
        from core.mcp_server import MCPServer

        server = MCPServer("test-server", "1.0")

        @server.tool("test_tool", "A test tool")
        def test_func(arg1: str, arg2: int = 5) -> dict:
            return {"result": f"{arg1}:{arg2}"}

        assert "test_tool" in server.tools

class TestIntegration:
    def test_knowledge_graph_claims(self, tmp_path):
        kg = KnowledgeGraph(storage_path=tmp_path / "kg")
        registry = ClaimsRegistry(storage_path=tmp_path / "claims")

        insight_id = kg.add_insight(
            insight_id="insight_001",
            content="Important discovery about Ramsey numbers",
            category="graph_theory"
        )

        claim = registry.claim(ClaimType.KNOWLEDGE, insight_id)

        assert claim is not None

        kg.compute_pagerank()
        important = kg.get_important_insights()

        assert len(important) >= 1

    def test_swarm_with_claims(self, tmp_path):
        swarm = SwarmCoordinator(storage_path=tmp_path / "swarm")
        registry = ClaimsRegistry(storage_path=tmp_path / "claims")

        swarm.register_agent("worker_001", AgentRole.WORKER, ["math"])

        task_id = swarm.create_task("Process mathematical insights")
        task_claim = registry.claim(ClaimType.TASK, task_id)

        assert task_claim is not None

        swarm.get_next_task("worker_001")
        swarm.complete_task(task_id, "worker_001", success=True)

        registry.release(task_claim.id)

        assert not registry.is_claimed(task_id)
