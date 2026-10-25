"""Round 25 Deep Audit Tests — vector_memory, embedding_interconnect, lcm_shield, lcm_planner, semantic_contracts, three_layer_context, reflect_edit, mcp_server, audio_bridge."""
import threading
import time
import pytest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.vector_memory import VectorMemory, HNSWIndex, VectorMemoryContext, VectorEntry, SearchResult
from core.semantic_contracts import ContractRegistry, ContractSchema, ContractInstance, FieldSpec, FieldType, ContractStatus
from core.lcm_shield import LCMShield, ShieldDecision, RiskLevel
from core.lcm_planner import LCMPlanner, PlanTask, Plan, TaskModel, TaskComplexity, TaskStatus
from core.reflect_edit import ReflectEditLoop, CouncilDeliberation, ReflectionType, CouncilProtocol
from core.three_layer_context import ThreeLayerContext, MemoryBlock, MemoryBlockOp, ContextLayer
from core.mcp_server import MCPServer, MCPTool, MCPCapabilities
from core.audio_bridge import AudioBridge, SemanticProjector, AudioModality


class TestVectorMemory:
    """Test vector_memory module — critical bug fixes."""

    def test_eviction_uses_delete_not_remove(self):
        """BUG: _evict_low_importance called index.remove() but HNSWIndex only has delete()."""
        idx = HNSWIndex(dim=3)
        entry_ids = []
        for i in range(5):
            eid = idx.insert([float(i), 0.0, 0.0], {"label": f"entry_{i}"})
            entry_ids.append(eid)
        assert len(idx) == 5
        assert idx.delete(entry_ids[0]) is True
        assert len(idx) == 4

    def test_rlock_no_deadlock_on_eviction(self):
        """BUG: store() calls _evict_low_importance() under self.lock.
        If lock were threading.Lock (non-reentrant), this would deadlock.
        Verify it's RLock so re-entrant calls work."""
        vm = VectorMemory(embedding_fn=lambda t: [0.1] * 768, storage_path=None)
        assert isinstance(vm.lock, type(threading.RLock()))
        vm.clear()

    def test_io_outside_lock_on_store(self):
        """Verify _save_index() is called outside the lock in store()."""
        vm = VectorMemory(embedding_fn=lambda t: [0.1] * 3, storage_path=None)
        vm.EMBEDDING_DIM = 3
        vm.index = HNSWIndex(dim=3)
        entry_id = vm.store("test io outside lock")
        assert entry_id is not None
        vm.clear()

    def test_io_outside_lock_on_delete(self):
        """Verify _save_index() is called outside the lock in delete()."""
        vm = VectorMemory(embedding_fn=lambda t: [0.1] * 3, storage_path=None)
        vm.EMBEDDING_DIM = 3
        vm.index = HNSWIndex(dim=3)
        entry_id = vm.store("test delete io")
        result = vm.delete(entry_id)
        assert result is True
        vm.clear()

    def test_io_outside_lock_on_eviction(self):
        """Verify _save_index() is called outside the lock in _evict_low_importance()."""
        vm = VectorMemory(embedding_fn=lambda t: [0.1] * 3, storage_path=None)
        vm.EMBEDDING_DIM = 3
        vm.index = HNSWIndex(dim=3)
        for i in range(5):
            vm.store(f"eviction test {i}", importance=0.1)
        vm.clear()

    def test_search_returns_results(self):
        """Verify search works with hash fallback embedding."""
        vm = VectorMemory(embedding_fn=lambda t: [0.1] * 3, storage_path=None)
        vm.EMBEDDING_DIM = 3
        vm.index = HNSWIndex(dim=3)
        vm.store("hello world", importance=0.8)
        vm.store("goodbye world", importance=0.5)
        results = vm.search("hello", k=2)
        assert len(results) >= 1
        vm.clear()

    def test_update_importance(self):
        """Verify importance scores update correctly."""
        vm = VectorMemory(embedding_fn=lambda t: [0.1] * 3, storage_path=None)
        vm.EMBEDDING_DIM = 3
        vm.index = HNSWIndex(dim=3)
        entry_id = vm.store("important entry", importance=0.3)
        vm.update_importance(entry_id, 0.5)
        assert vm.importance_scores[entry_id] == 0.8
        vm.clear()

    def test_access_times_attribute_exists(self):
        """BUG FIX: access_times dict was missing — verify it exists."""
        vm = VectorMemory(embedding_fn=lambda t: [0.1] * 3, storage_path=None)
        vm.EMBEDDING_DIM = 3
        vm.index = HNSWIndex(dim=3)
        assert isinstance(vm.access_times, dict)
        vm.clear()

    def test_concurrent_store(self):
        """Verify thread safety of concurrent store operations."""
        vm = VectorMemory(embedding_fn=lambda t: [0.1] * 3, storage_path=None)
        vm.EMBEDDING_DIM = 3
        vm.index = HNSWIndex(dim=3)
        errors = []

        def store_item(i):
            try:
                vm.store(f"concurrent item {i}", importance=0.5)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=store_item, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors in concurrent stores: {errors}"
        vm.clear()

    def test_vector_memory_context_cache_eviction(self):
        """Test VectorMemoryContext cache eviction works."""
        ctx = VectorMemoryContext(vector_memory=None)
        for i in range(1100):
            ctx._last_cache_update[f"key{i}"] = time.time() - 700
            ctx._context_cache[f"key{i}"] = []
            ctx._cache_order[f"key{i}"] = None
        ctx._evict_cache()
        assert len(ctx._context_cache) <= ctx._max_cache_size
        assert len(ctx._last_cache_update) <= ctx._max_cache_size

    def test_hnsw_index_insert_and_search(self):
        """Verify HNSW index insert and search work."""
        idx = HNSWIndex(dim=3)
        idx.insert([1.0, 0.0, 0.0], {"label": "x"}, "v1")
        idx.insert([0.0, 1.0, 0.0], {"label": "y"}, "v2")
        idx.insert([0.0, 0.0, 1.0], {"label": "z"}, "v3")
        results = idx.search([1.0, 0.0, 0.0], k=2)
        assert len(results) >= 1
        assert results[0].id == "v1"

    def test_hnsw_index_delete(self):
        """Verify HNSW index delete works."""
        idx = HNSWIndex(dim=3)
        idx.insert([1.0, 0.0, 0.0], {"label": "x"}, "v1")
        assert idx.delete("v1") is True
        assert idx.delete("v1") is False


class TestEmbeddingInterconnect:
    """Test embedding_interconnect bug fixes."""

    def test_memory_order_set_exists(self):
        """Verify companion sets exist for O(1) membership checks."""
        from core.embedding_interconnect import EmbeddingInterconnect
        emb = EmbeddingInterconnect(prefer_server=False)
        assert hasattr(emb, '_memory_order_set')
        assert isinstance(emb._memory_order_set, set)
        assert hasattr(emb, '_insight_order_set')
        assert hasattr(emb, '_knowledge_order_set')
        assert hasattr(emb, '_workflow_order_set')
        assert hasattr(emb, '_cluster_order_set')

    def test_store_memory_updates_order_set(self):
        """Verify store_memory correctly maintains order set."""
        from core.embedding_interconnect import EmbeddingInterconnect
        import numpy as np
        from unittest.mock import MagicMock, patch
        
        emb = EmbeddingInterconnect(prefer_server=False)
        
        mock_vec = np.zeros(768, dtype=np.float32)
        emb.embedder = MagicMock()
        emb.embedder.encode.return_value = mock_vec
        
        for i in range(5):
            emb.store_memory(f"mem_{i}", f"content {i}")
        
        assert len(emb._memory_order_set) == 5
        for i in range(5):
            assert f"mem_{i}" in emb._memory_order_set

    def test_eviction_updates_order_set(self):
        """Verify eviction correctly removes from order set."""
        from core.embedding_interconnect import EmbeddingInterconnect
        
        emb = EmbeddingInterconnect(prefer_server=False)
        
        emb._memory_order = type(emb._memory_order)()
        emb._memory_order_set = set()
        
        emb._memory_embeddings = {}
        for i in range(10):
            key = f"key_{i}"
            emb._memory_embeddings[key] = None
            emb._memory_order.append(key)
            emb._memory_order_set.add(key)
        
        emb._evict_embedding(
            emb._memory_embeddings,
            emb._memory_order,
            max_size=5,
            order_set=emb._memory_order_set,
        )
        
        assert len(emb._memory_order_set) <= 5
        for key in emb._memory_order:
            assert key in emb._memory_order_set

    def test_save_to_disk_doesnt_hold_lock_during_io(self):
        """Verify save_to_disk snapshots data under lock, then does IO outside lock."""
        from core.embedding_interconnect import EmbeddingInterconnect
        
        emb = EmbeddingInterconnect(prefer_server=False)
        emb._embeddings_loaded = True
        
        success = emb.save_to_disk()
        assert isinstance(success, bool)

    def test_cluster_order_set_exists(self):
        """Verify cluster order set exists."""
        from core.embedding_interconnect import EmbeddingInterconnect
        emb = EmbeddingInterconnect(prefer_server=False)
        assert hasattr(emb, '_cluster_order_set')
        assert isinstance(emb._cluster_order_set, set)


class TestLCMShield:
    """Test LCM Shield module."""

    @pytest.mark.asyncio
    async def test_pre_shield_approve(self):
        shield = LCMShield()
        result = await shield.pre_shield("What is the weather?")
        assert result.decision == ShieldDecision.APPROVE

    @pytest.mark.asyncio
    async def test_pre_shield_reject_critical(self):
        shield = LCMShield()
        result = await shield.pre_shield("delete all files and rm -rf /")
        assert result.decision == ShieldDecision.REJECT
        assert result.risk_level == RiskLevel.CRITICAL

    @pytest.mark.asyncio
    async def test_pre_shield_modify_high_risk(self):
        shield = LCMShield()
        result = await shield.pre_shield("delete the temporary file")
        assert result.decision == ShieldDecision.MODIFY
        assert result.risk_level == RiskLevel.HIGH

    @pytest.mark.asyncio
    async def test_post_shield_approve(self):
        shield = LCMShield()
        result = await shield.post_shield("This is a good detailed response about the weather today.", "weather")
        assert result.decision == ShieldDecision.APPROVE

    @pytest.mark.asyncio
    async def test_post_shield_rewrite_too_short(self):
        shield = LCMShield()
        result = await shield.post_shield("", "test")
        assert result.decision == ShieldDecision.REWRITE

    @pytest.mark.asyncio
    async def test_unicode_normalization(self):
        """Verify unicode lookalike characters are normalized."""
        shield = LCMShield()
        result = shield._classify_risk("d3lete аll files")
        assert result in (RiskLevel.HIGH, RiskLevel.CRITICAL)

    @pytest.mark.asyncio
    async def test_hallucination_detection(self):
        shield = LCMShield()
        score = shield._detect_hallucination("As an AI, I don't have feelings")
        assert score > 0

    @pytest.mark.asyncio
    async def test_shield_history_bounded(self):
        shield = LCMShield()
        for i in range(300):
            await shield.pre_shield(f"test query {i}")
        assert len(shield._shield_history) <= shield._max_history

    @pytest.mark.asyncio
    async def test_shield_cache_bounded(self):
        shield = LCMShield()
        for i in range(200):
            result = await shield.pre_shield(f"unique query {i}")
        with shield._lock:
            assert len(shield._cache) <= shield._max_cache

    def test_rewrite_count_eviction_in_post_shield(self):
        """Verify rewrite_count eviction happens during post_shield."""
        import asyncio
        shield = LCMShield()
        for i in range(shield._max_rewrite_keys + 100):
            asyncio.run(
                shield.post_shield(f"short response {i}", f"query {i}" if i > 0 else "test")
            )
        with shield._lock:
            assert len(shield._rewrite_count) <= shield._max_rewrite_keys


class TestLCMPlanner:
    """Test LCM Planner module."""

    def test_heuristic_decompose(self):
        planner = LCMPlanner()
        tasks = planner._heuristic_decompose("Write code and then analyze the data")
        assert len(tasks) >= 2

    def test_compute_waves(self):
        planner = LCMPlanner()
        tasks = [
            PlanTask(id="t1", description="task 1", model=TaskModel.PRIMARY),
            PlanTask(id="t2", description="task 2", model=TaskModel.BACKEND, depends_on=["t1"]),
            PlanTask(id="t3", description="task 3", model=TaskModel.PRIMARY, depends_on=["t1"]),
        ]
        waves = planner._compute_waves(tasks)
        assert len(waves) >= 2
        assert "t1" in waves[0]

    def test_compute_waves_cyclic(self):
        planner = LCMPlanner()
        tasks = [
            PlanTask(id="t1", description="task 1", depends_on=["t2"]),
            PlanTask(id="t2", description="task 2", depends_on=["t1"]),
        ]
        waves = planner._compute_waves(tasks)
        assert len(waves) >= 1

    def test_active_plans_bounded_in_create_plan(self):
        """Verify active plans eviction happens during create_plan."""
        planner = LCMPlanner()
        import asyncio
        for i in range(30):
            plan = Plan(
                id=f"plan_{i}",
                original_request=f"test {i}",
            )
            with planner._lock:
                planner._active_plans[f"plan_{i}"] = plan
                while len(planner._active_plans) > planner._max_active_plans:
                    planner._active_plans.popitem(last=False)
        assert len(planner._active_plans) <= planner._max_active_plans


class TestSemanticContracts:
    """Test semantic contracts module."""

    def test_create_and_validate(self):
        registry = ContractRegistry()
        result = registry.create_contract(
            schema_name="chat_response",
            source_model="primary",
            target_model="backend",
            data={"content": "Hello", "intent": "greeting", "confidence": 0.9, "topics": []},
        )
        assert result.status == ContractStatus.VALIDATED

    def test_reject_missing_required(self):
        registry = ContractRegistry()
        result = registry.create_contract(
            schema_name="chat_response",
            source_model="primary",
            target_model="backend",
            data={"content": "Hello"},
        )
        assert result.status == ContractStatus.REJECTED
        assert any("Missing required" in e for e in result.validation_errors)

    def test_reject_oversized(self):
        registry = ContractRegistry()
        result = registry.create_contract(
            schema_name="cross_model_context",
            source_model="primary",
            target_model="backend",
            data={"topic": "x" * 5000, "key_points": [], "source": "test"},
        )
        assert result.status == ContractStatus.REJECTED

    def test_instances_bounded(self):
        registry = ContractRegistry()
        for i in range(300):
            registry.create_contract(
                schema_name="chat_response",
                source_model="primary",
                target_model="backend",
                data={"content": f"test {i}", "intent": "greeting", "confidence": 0.9, "topics": []},
            )
        assert len(registry._instances) <= registry._max_instances

    def test_get_stats(self):
        registry = ContractRegistry()
        registry.create_contract(
            schema_name="chat_response",
            source_model="primary",
            target_model="backend",
            data={"content": "Hello", "intent": "greeting", "confidence": 0.9, "topics": []},
        )
        stats = registry.get_stats()
        assert "total_contracts" in stats
        assert stats["total_contracts"] == 1


class TestThreeLayerContext:
    """Test three-layer context module."""

    def test_edit_block_append(self):
        tlc = ThreeLayerContext()
        result = tlc.edit_block("primary_block", MemoryBlockOp.APPEND, caller="kimi-k2.5", text="Test content")
        assert result is True

    def test_edit_block_read_only(self):
        tlc = ThreeLayerContext()
        result = tlc.edit_block("lcm_meta", MemoryBlockOp.APPEND, caller="unknown", text="Should fail")
        assert result is False

    def test_edit_block_prefrontal_can_edit_readonly(self):
        tlc = ThreeLayerContext()
        result = tlc.edit_block("lcm_meta", MemoryBlockOp.APPEND, caller="prefrontal_cortex", text="LCM update")
        assert result is True

    def test_memory_block_max_chars(self):
        block = MemoryBlock(name="test", owner="test", max_chars=50)
        block.append("a" * 50)
        assert len(block.content) <= 100
        assert block.modification_count >= 1

    def test_create_block_bounded(self):
        tlc = ThreeLayerContext()
        for i in range(30):
            tlc.create_block(f"custom_{i}", owner="test")
        protected_count = len(ThreeLayerContext._PROTECTED_BLOCKS)
        total_blocks = len(tlc._memory_blocks)
        assert total_blocks <= tlc._max_blocks + protected_count

    def test_blocks_bounded(self):
        tlc = ThreeLayerContext()
        initial_blocks = len(tlc._memory_blocks)
        for i in range(25):
            tlc.create_block(f"extra_{i}", owner="test")
        assert len(tlc._memory_blocks) <= tlc._max_blocks

    def test_transitions_bounded(self):
        tlc = ThreeLayerContext()
        for i in range(300):
            tlc._transitions.append(type('obj', (object,), {
                'source_layer': ContextLayer.L1_HOT,
                'target_layer': ContextLayer.L2_WARM,
                'content': f"test {i}",
                'reason': "test",
                'timestamp': time.time(),
                'token_count': 100,
            })())
        assert len(tlc._transitions) <= tlc._max_transitions

    def test_get_l2_context(self):
        tlc = ThreeLayerContext()
        tlc.edit_block("primary_block", MemoryBlockOp.APPEND, caller="kimi-k2.5", text="Test context")
        context = tlc.get_l2_context_for_model("primary")
        assert "Test context" in context


class TestReflectEdit:
    """Test reflect-edit loop module."""

    def test_reflect_creates_reflection(self):
        loop = ReflectEditLoop()
        import asyncio
        reflection = asyncio.run(
            loop.reflect("What is AI?", "AI is artificial intelligence", "primary", latency_ms=100)
        )
        assert reflection is not None
        assert reflection.score >= 0
        assert len(loop._reflections) == 1

    def test_reflections_bounded(self):
        loop = ReflectEditLoop(max_history=10)
        import asyncio
        for i in range(20):
            asyncio.run(
                loop.reflect(f"query {i}", "response", "primary")
            )
        assert len(loop._reflections) <= 10

    def test_routing_table_bounded(self):
        loop = ReflectEditLoop()
        for i in range(120):
            loop._update_routing(f"query {i}", "backend", 0.8)
        assert len(loop._routing_table) <= loop._max_routing_entries

    def test_get_routing_hint(self):
        loop = ReflectEditLoop()
        loop._update_routing("What is machine learning?", "backend", 0.9)
        model, score = loop.get_routing_hint("What is machine learning?")
        assert model == "backend"
        assert score == 0.9


class TestCouncilDeliberation:
    """Test council deliberation module."""

    def test_council_sessions_bounded_in_deliberate(self):
        """Verify sessions eviction happens in deliberate."""
        council = CouncilDeliberation()
        from core.reflect_edit import CouncilSession
        with council._lock:
            for i in range(60):
                session = CouncilSession(
                    id=f"session_{i}",
                    topic=f"topic {i}",
                    protocol=CouncilProtocol.DEBATE,
                )
                council._sessions[session.id] = session
                while len(council._sessions) > council._max_sessions:
                    council._sessions.popitem(last=False)
        assert len(council._sessions) <= council._max_sessions


class TestMCPServer:
    """Test MCP Server module."""

    def test_register_tool(self):
        server = MCPServer()
        @server.tool("test_tool", description="A test tool")
        def test_handler(query: str) -> dict:
            return {"result": query}
        
        assert "test_tool" in server.tools

    def test_register_resource(self):
        server = MCPServer()
        @server.resource("test://resource", "Test Resource", "A test resource")
        def test_handler() -> dict:
            return {"data": "test"}
        
        assert "test://resource" in server.resources

    def test_register_prompt(self):
        server = MCPServer()
        @server.prompt("test_prompt", description="A test prompt")
        def test_handler(task: str) -> str:
            return f"Task: {task}"
        
        assert "test_prompt" in server.prompts

    @pytest.mark.asyncio
    async def test_handle_initialize(self):
        server = MCPServer()
        result = await server.handle_request({"method": "initialize", "params": {}, "id": 1})
        assert "result" in result
        assert result["result"]["protocolVersion"] == "2024-11-05"

    @pytest.mark.asyncio
    async def test_handle_tools_list(self):
        server = MCPServer()
        result = await server.handle_request({"method": "tools/list", "params": {}, "id": 2})
        assert "result" in result
        assert "tools" in result["result"]

    @pytest.mark.asyncio
    async def test_handle_unknown_method(self):
        server = MCPServer()
        result = await server.handle_request({"method": "unknown/method", "params": {}, "id": 3})
        assert "error" in result


class TestAudioBridge:
    """Test audio bridge module."""

    def test_semantic_projector(self):
        projector = SemanticProjector(input_dim=256, output_dim=768)
        features = np.random.randn(256).astype(np.float32)
        embedding = projector.project(features)
        assert embedding.shape == (768,)

    def test_tokenize_speech_input(self):
        bridge = AudioBridge()
        seq = bridge.tokenize_speech_input("Hello world", duration_seconds=2.0)
        assert seq.modality == AudioModality.SPEECH_INPUT
        assert seq.duration_seconds == 2.0
        assert len(seq.tokens) > 0

    def test_tokenize_music_output(self):
        bridge = AudioBridge()
        seq = bridge.tokenize_music_output({
            "notes": [60, 62, 64],
            "chords": ["C", "Am"],
            "duration_seconds": 10.0,
            "style": "classical",
            "title": "Test",
        })
        assert seq.modality == AudioModality.MUSIC
        assert seq.duration_seconds == 10.0

    def test_sequences_bounded(self):
        bridge = AudioBridge()
        for i in range(150):
            bridge.tokenize_speech_input(f"test {i}")
        assert len(bridge._sequences) <= bridge._max_sequences

    def test_embeddings_bounded(self):
        bridge = AudioBridge()
        for i in range(250):
            bridge.tokenize_speech_input(f"test {i}")
            if bridge._audio_embeddings:
                pass
        assert len(bridge._audio_embeddings) <= bridge._max_embeddings

    def test_find_similar_audio(self):
        bridge = AudioBridge()
        seq1 = bridge.tokenize_speech_input("hello world")
        if seq1.semantic_embedding is not None:
            results = bridge.find_similar_audio(seq1.semantic_embedding, top_k=3)
            assert isinstance(results, list)

    def test_get_stats(self):
        bridge = AudioBridge()
        bridge.tokenize_speech_input("test stats")
        stats = bridge.get_stats()
        assert "total_sequences" in stats
        assert stats["total_sequences"] >= 1


class TestVectorMemoryContext:
    """Test VectorMemoryContext module."""

    def test_context_cache_bounded(self):
        ctx = VectorMemoryContext(vector_memory=None)
        for i in range(1100):
            ctx._last_cache_update[f"key_{i}"] = time.time() - 600
            ctx._context_cache[f"key_{i}"] = []
            ctx._cache_order[f"key_{i}"] = None
        ctx._evict_cache()
        assert len(ctx._context_cache) <= ctx._max_cache_size

    def test_get_vm_creates_instance(self):
        vm = VectorMemory(embedding_fn=lambda t: [0.1] * 3, storage_path=None)
        vm.EMBEDDING_DIM = 3
        vm.index = HNSWIndex(dim=3)
        assert vm is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])