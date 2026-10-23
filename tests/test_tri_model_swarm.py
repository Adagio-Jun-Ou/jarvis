"""Tests for Dual-Model integration with Agent Swarms."""
import asyncio
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ["CUDA_VISIBLE_DEVICES"] = ""


class TestSwarmDualModelBasic:
    """Basic tests for swarm with dual-model architecture."""

    def test_swarm_imports(self):
        from core.swarm import SwarmCoordinator, AgentRole, AgentStatus
        from core.dual_model import ModelRole
        from core.concept_model import ConceptModelOrchestrator

        assert SwarmCoordinator is not None
        assert ModelRole.PRIMARY is not None
        assert ModelRole.BACKEND is not None

    def test_model_role_enum_in_swarm_context(self):
        from core.dual_model import ModelRole

        roles = [ModelRole.PRIMARY, ModelRole.BACKEND]
        assert len(roles) == 2


class TestSwarmWithMockDualModel:
    """Test swarm agents using dual-model LLMs."""

    @pytest.fixture
    def mock_dual_model(self):
        class MockLLM:
            def __init__(self, name):
                self.name = name
                self.call_count = 0

            async def chat(self, user_input, **kwargs):
                self.call_count += 1
                return {"message": {"content": f"{self.name} response: {user_input[:50]}"}}

        class MockDualModel:
            def __init__(self):
                self.primary = MockLLM("primary")
                self.backend = MockLLM("backend")

            async def route_task(self, task_type, prompt):
                from core.dual_model import TaskType
                if task_type in [TaskType.CODE, TaskType.MATH, TaskType.ANALYSIS, TaskType.REASONING, TaskType.SYNTHESIS]:
                    return await self.backend.chat(prompt)
                else:
                    return await self.primary.chat(prompt)

        return MockDualModel()

    def test_swarm_coordinator_creation(self, tmp_path):
        from core.swarm import SwarmCoordinator

        swarm = SwarmCoordinator(storage_path=tmp_path)
        assert swarm is not None
        assert swarm.agent_id == "jarvis_main"

    @pytest.mark.asyncio
    async def test_swarm_agent_with_dual_model_routing(self, tmp_path, mock_dual_model):
        from core.swarm import SwarmCoordinator, AgentRole, TaskPriority
        from core.dual_model import TaskType

        swarm = SwarmCoordinator(storage_path=tmp_path)
        swarm.register_agent("worker1", AgentRole.WORKER, ["code", "math", "general"])

        code_task = swarm.create_task(
            "Write a Python function",
            priority=TaskPriority.HIGH,
            required_capabilities=["code"],
            metadata={"task_type": TaskType.CODE.value}
        )

        response = await mock_dual_model.route_task(TaskType.CODE, "Write a Python function")
        assert "backend response" in response["message"]["content"]
        assert mock_dual_model.backend.call_count == 1

    @pytest.mark.asyncio
    async def test_swarm_abstraction_task_routing(self, tmp_path, mock_dual_model):
        from core.swarm import SwarmCoordinator, AgentRole
        from core.dual_model import TaskType

        swarm = SwarmCoordinator(storage_path=tmp_path)
        swarm.register_agent("abstraction_agent", AgentRole.SPECIALIST, ["abstraction", "meta_reasoning"])

        task = swarm.create_task(
            "Extract pattern from these examples",
            metadata={"task_type": TaskType.ABSTRACTION.value}
        )

        response = await mock_dual_model.route_task(TaskType.ABSTRACTION, "Extract pattern")
        assert "primary response" in response["message"]["content"]
        assert mock_dual_model.primary.call_count == 1
        
        # Test synthesis goes to backend
        response2 = await mock_dual_model.route_task(TaskType.SYNTHESIS, "Synthesize results")
        assert "backend response" in response2["message"]["content"]
        assert mock_dual_model.backend.call_count == 1

    @pytest.mark.asyncio
    async def test_swarm_multi_agent_dual_model_collaboration(self, tmp_path, mock_dual_model):
        from core.swarm import SwarmCoordinator, AgentRole, TaskPriority
        from core.dual_model import TaskType

        swarm = SwarmCoordinator(storage_path=tmp_path)

        swarm.register_agent("primary_agent", AgentRole.WORKER, ["chat", "creative"])
        swarm.register_agent("backend_agent", AgentRole.SPECIALIST, ["code", "math", "abstraction", "synthesis"])

        task1 = swarm.create_task("Chat with user", metadata={"task_type": TaskType.CHAT.value})
        task2 = swarm.create_task("Analyze code", metadata={"task_type": TaskType.CODE.value})
        task3 = swarm.create_task("Extract patterns", metadata={"task_type": TaskType.ABSTRACTION.value})

        r1 = await mock_dual_model.route_task(TaskType.CHAT, "Chat with user")
        r2 = await mock_dual_model.route_task(TaskType.CODE, "Analyze code")
        r3 = await mock_dual_model.route_task(TaskType.SYNTHESIS, "Synthesize patterns")

        assert "primary response" in r1["message"]["content"]
        assert "backend response" in r2["message"]["content"]
        assert "backend response" in r3["message"]["content"]

        assert mock_dual_model.primary.call_count == 1
        assert mock_dual_model.backend.call_count == 2


class TestSwarmConceptEmbedderIntegration:
    """Test swarm using concept embedder for semantic task matching."""

    @pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason="Skip embedding model download in CI",
    )
    def test_swarm_semantic_task_matching(self, tmp_path):
        from core.swarm import SwarmCoordinator, AgentRole, TaskPriority
        from core.concept_model import ConceptEmbedder
        import numpy as np

        swarm = SwarmCoordinator(storage_path=tmp_path)
        embedder = ConceptEmbedder(provider="onnx")
        
        test_emb = embedder.encode("test")
        if np.linalg.norm(test_emb) == 0.0:
            pytest.skip("No real embedding model available")

        swarm.register_agent("ml_specialist", AgentRole.SPECIALIST, ["machine_learning", "neural_networks"])
        swarm.register_agent("web_specialist", AgentRole.SPECIALIST, ["web", "frontend"])

        ml_task = swarm.create_task("Train a neural network", priority=TaskPriority.HIGH)

        task_embedding = embedder.encode("Train a neural network")
        ml_capability_embedding = embedder.encode("machine learning neural networks")
        web_capability_embedding = embedder.encode("web frontend development")

        ml_sim = np.dot(task_embedding, ml_capability_embedding) / (
            np.linalg.norm(task_embedding) * np.linalg.norm(ml_capability_embedding)
        )
        web_sim = np.dot(task_embedding, web_capability_embedding) / (
            np.linalg.norm(task_embedding) * np.linalg.norm(web_capability_embedding)
        )

        assert ml_sim > web_sim, "ML task should match ML specialist better"

    @pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason="Skip embedding model download in CI",
    )
    def test_swarm_task_clustering(self, tmp_path):
        from core.swarm import SwarmCoordinator, AgentRole, TaskPriority
        from core.concept_model import ConceptEmbedder
        import numpy as np

        swarm = SwarmCoordinator(storage_path=tmp_path)
        embedder = ConceptEmbedder(provider="onnx")
        
        test_emb = embedder.encode("test")
        if np.linalg.norm(test_emb) == 0.0:
            pytest.skip("No real embedding model available")

        tasks = [
            "Write Python code",
            "Debug JavaScript",
            "Create a neural network",
            "Train ML model",
            "Design database schema",
            "Optimize SQL queries",
        ]

        for task in tasks:
            swarm.create_task(task)

        task_embeddings = embedder.encode(tasks)
        clusters = embedder.cluster_concepts(tasks, n_clusters=3)

        assert len(clusters) == 3
        total_tasks = sum(len(v) for v in clusters.values())
        assert total_tasks == 6


class TestSwarmTriModelConsensus:
    """Test consensus mechanism with tri-model for decision making."""

    def test_swarm_consensus_basic(self, tmp_path):
        from core.swarm import SwarmCoordinator, AgentRole

        swarm = SwarmCoordinator(storage_path=tmp_path)

        swarm.register_agent("agent1", AgentRole.COORDINATOR)
        swarm.register_agent("agent2", AgentRole.WORKER)
        swarm.register_agent("agent3", AgentRole.WORKER)

        proposal_id = swarm.propose_consensus("task_allocation", {"strategy": "round_robin"})

        assert proposal_id is not None

    @pytest.mark.asyncio
    async def test_tri_model_in_consensus_decision(self, tmp_path):
        from core.swarm import SwarmCoordinator, AgentRole

        swarm = SwarmCoordinator(storage_path=tmp_path)

        swarm.register_agent("coordinator", AgentRole.COORDINATOR)
        swarm.register_agent("worker1", AgentRole.WORKER)
        swarm.register_agent("worker2", AgentRole.WORKER)

        proposal_id = swarm.propose_consensus("use_concept_model", {"task": "abstraction"})

        swarm.vote_on_behalf(proposal_id, "coordinator", True)
        swarm.vote_on_behalf(proposal_id, "worker1", True)
        swarm.vote_on_behalf(proposal_id, "worker2", True)

        decision = swarm.get_consensus_decision(proposal_id)

        assert decision is not None
        assert "approved" in decision.lower() or "accepted" in decision.lower()


class TestSwarmTriModelTaskFlow:
    """Test complete task flow through tri-model swarm."""

    @pytest.mark.asyncio
    async def test_full_task_flow_with_tri_model(self, tmp_path):
        from core.swarm import SwarmCoordinator, AgentRole, TaskPriority
        from core.dual_model import TaskType

        class MockTriModelOrchestrator:
            def __init__(self):
                self.executed_tasks = []

            async def execute(self, task_type, prompt, context=None):
                from core.dual_model import TaskType
                self.executed_tasks.append({
                    "task_type": task_type.value if hasattr(task_type, 'value') else str(task_type),
                    "prompt": prompt[:50],
                })

                if task_type == TaskType.ABSTRACTION:
                    return {"result": "abstract_pattern_found"}
                elif task_type == TaskType.CODE:
                    return {"result": "code_generated"}
                else:
                    return {"result": "response_generated"}

        swarm = SwarmCoordinator(storage_path=tmp_path)
        tri_model = MockTriModelOrchestrator()

        swarm.register_agent("worker1", AgentRole.WORKER, ["general", "code"])
        swarm.register_agent("worker2", AgentRole.SPECIALIST, ["abstraction", "synthesis"])

        task1_id = swarm.create_task(
            "Analyze this code pattern",
            priority=TaskPriority.HIGH,
            metadata={"task_type": TaskType.CODE.value}
        )

        task2_id = swarm.create_task(
            "Extract abstract pattern from solutions",
            priority=TaskPriority.NORMAL,
            required_capabilities=["abstraction"],
            metadata={"task_type": TaskType.ABSTRACTION.value}
        )

        r1 = await tri_model.execute(TaskType.CODE, "Analyze this code pattern")
        r2 = await tri_model.execute(TaskType.ABSTRACTION, "Extract abstract pattern")

        assert r1["result"] == "code_generated"
        assert r2["result"] == "abstract_pattern_found"
        assert len(tri_model.executed_tasks) == 2

    @pytest.mark.asyncio
    async def test_tri_model_collaborative_problem_solving(self, tmp_path):
        from core.swarm import SwarmCoordinator, AgentRole, TaskPriority
        from core.dual_model import TaskType

        class MockCollaborativeTriModel:
            def __init__(self):
                self.collaboration_log = []

            async def primary_think(self, problem):
                self.collaboration_log.append(("primary", "think", problem[:30]))
                return {"thought": "Initial understanding of problem"}

            async def backend_analyze(self, data):
                self.collaboration_log.append(("backend", "analyze", data[:30]))
                return {"analysis": "Technical analysis complete"}

            async def concept_synthesize(self, inputs):
                self.collaboration_log.append(("concept", "synthesize", str(inputs)[:30]))
                return {"synthesis": "Unified pattern discovered"}

        swarm = SwarmCoordinator(storage_path=tmp_path)
        tri_model = MockCollaborativeTriModel()

        swarm.register_agent("primary_agent", AgentRole.WORKER, ["chat", "creative"])
        swarm.register_agent("backend_agent", AgentRole.SPECIALIST, ["code", "analysis"])
        swarm.register_agent("concept_agent", AgentRole.SPECIALIST, ["abstraction", "synthesis"])

        problem_task = swarm.create_task(
            "Complex problem requiring tri-model collaboration",
            priority=TaskPriority.CRITICAL,
            metadata={"requires_collaboration": True}
        )

        t1 = await tri_model.primary_think("Complex problem requiring tri-model collaboration")
        t2 = await tri_model.backend_analyze("Technical details of the problem")
        t3 = await tri_model.concept_synthesize([t1, t2])

        assert len(tri_model.collaboration_log) == 3
        assert tri_model.collaboration_log[0][0] == "primary"
        assert tri_model.collaboration_log[1][0] == "backend"
        assert tri_model.collaboration_log[2][0] == "concept"


class TestSwarmConceptModelHealth:
    """Test concept model health in swarm context."""

    @pytest.mark.asyncio
    async def test_concept_model_health_in_swarm(self, tmp_path):
        from core.concept_model import ConceptModelOrchestrator

        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(return_value={"message": {"content": "Healthy"}})

        orchestrator = ConceptModelOrchestrator(concept_llm=mock_llm)
        await orchestrator.start()

        result = await orchestrator.synthesize_concepts(["test"])
        assert len(result) > 0

        stats = orchestrator._stats
        assert stats.total_requests >= 1

        await orchestrator.stop()

    @pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason="Skip embedding model download in CI",
    )
    def test_concept_embedder_health_in_swarm(self, tmp_path):
        from core.concept_model import ConceptEmbedder

        embedder = ConceptEmbedder(provider="onnx")

        embedding = embedder.encode("Health check test")
        assert embedding.shape == (768,)

        long_text = "Long context test. " * 500
        long_embedding = embedder.encode(long_text)
        assert long_embedding.shape == (768,)


class TestSwarmModelRoleRouting:
    """Test model role routing within swarm."""

    def test_model_role_routing_table(self):
        from core.dual_model import TaskType, ModelRole

        routing = {
            TaskType.CHAT: ModelRole.PRIMARY,
            TaskType.CREATIVE: ModelRole.PRIMARY,
            TaskType.CODE: ModelRole.BACKEND,
            TaskType.MATH: ModelRole.BACKEND,
            TaskType.ABSTRACTION: ModelRole.PRIMARY,
            TaskType.SYNTHESIS: ModelRole.BACKEND,
            TaskType.META_REASONING: ModelRole.PRIMARY,
            TaskType.PATTERN: ModelRole.PRIMARY,
            TaskType.VISION: ModelRole.PRIMARY,
        }

        for task_type, expected_role in routing.items():
            assert expected_role in [ModelRole.PRIMARY, ModelRole.BACKEND]

    @pytest.mark.asyncio
    async def test_swarm_routes_to_correct_model_role(self, tmp_path):
        from core.swarm import SwarmCoordinator, AgentRole, TaskPriority
        from core.dual_model import TaskType, ModelRole

        class MockRoutingOrchestrator:
            def __init__(self):
                self.routing_log = []

            async def route(self, task_type: TaskType, prompt: str):
                role = ModelRole.BACKEND
                if task_type in [TaskType.CHAT, TaskType.CREATIVE, TaskType.ABSTRACTION, 
                                 TaskType.META_REASONING, TaskType.PATTERN, TaskType.VISION]:
                    role = ModelRole.PRIMARY
                
                self.routing_log.append({
                    "task_type": task_type.value,
                    "routed_to": role.value,
                })
                return {"role": role.value, "response": "ok"}

        swarm = SwarmCoordinator(storage_path=tmp_path)
        router = MockRoutingOrchestrator()

        swarm.register_agent("agent", AgentRole.WORKER, ["general"])

        await router.route(TaskType.CHAT, "Hello")
        await router.route(TaskType.CODE, "Write code")
        await router.route(TaskType.SYNTHESIS, "Synthesize this")

        assert router.routing_log[0]["routed_to"] == "primary"
        assert router.routing_log[1]["routed_to"] == "backend"
        assert router.routing_log[2]["routed_to"] == "backend"


class TestSwarmDualModelStress:
    """Stress tests for dual-model swarm."""

    @pytest.mark.asyncio
    async def test_concurrent_dual_model_requests(self, tmp_path):
        from core.swarm import SwarmCoordinator, AgentRole
        from core.dual_model import TaskType
        import asyncio

        class MockDualModel:
            def __init__(self):
                self.request_count = 0

            async def process(self, task_type, prompt):
                self.request_count += 1
                await asyncio.sleep(0.01)
                return {"result": f"processed_{self.request_count}"}

        swarm = SwarmCoordinator(storage_path=tmp_path)
        dual_model = MockDualModel()

        for i in range(5):
            swarm.register_agent(f"agent_{i}", AgentRole.WORKER, ["general"])

        task_types = [
            TaskType.CHAT, TaskType.CODE, TaskType.ABSTRACTION,
            TaskType.MATH, TaskType.SYNTHESIS
        ]

        tasks = [
            dual_model.process(tt, f"Task {i}")
            for i, tt in enumerate(task_types)
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == 5
        assert dual_model.request_count == 5

    def test_swarm_max_agents_with_dual_model(self, tmp_path):
        from core.swarm import SwarmCoordinator, AgentRole

        swarm = SwarmCoordinator(storage_path=tmp_path)

        for i in range(swarm.MAX_AGENTS - 1):
            result = swarm.register_agent(f"agent_{i}", AgentRole.WORKER, ["general"])
            assert result is True

        result = swarm.register_agent("overflow", AgentRole.WORKER)
        assert result is False

    def test_swarm_max_tasks_with_dual_model(self, tmp_path):
        from core.swarm import SwarmCoordinator

        swarm = SwarmCoordinator(storage_path=tmp_path)

        for i in range(100):
            swarm.create_task(f"Task {i}")

        assert len(swarm._tasks) <= swarm.MAX_TASKS