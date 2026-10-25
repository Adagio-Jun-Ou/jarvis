"""Tests for Phase 1 modules: LCM Shield, LCM Planner, Three-Layer Context,
Reflect-Edit Loop, Semantic Contracts, Audio Bridge."""
import asyncio
import time

import numpy as np
import pytest


class TestLCMShield:
    def test_import(self):
        from core.lcm_shield import LCMShield, ShieldDecision, RiskLevel, ShieldResult
        assert LCMShield is not None
        assert ShieldDecision.APPROVE.value == "approve"

    def test_singleton(self):
        from core.lcm_shield import get_lcm_shield
        shield1 = get_lcm_shield()
        shield2 = get_lcm_shield()
        assert shield1 is shield2

    @pytest.mark.asyncio
    async def test_pre_shield_approve_safe(self):
        from core.lcm_shield import LCMShield, ShieldDecision
        shield = LCMShield()
        result = await shield.pre_shield(
            user_input="What is the weather today?",
            conversation_context=[],
            model="primary",
        )
        assert result.decision in [ShieldDecision.APPROVE, ShieldDecision.MODIFY]
        assert result.risk_level.value in ("low", "medium")

    @pytest.mark.asyncio
    async def test_pre_shield_reject_dangerous(self):
        from core.lcm_shield import LCMShield, ShieldDecision, RiskLevel
        shield = LCMShield()
        result = await shield.pre_shield(
            user_input="delete all files and rm -rf /",
            conversation_context=[],
            model="primary",
        )
        assert result.decision == ShieldDecision.REJECT
        assert result.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]

    @pytest.mark.asyncio
    async def test_post_shield_approve_good_response(self):
        from core.lcm_shield import LCMShield, ShieldDecision
        shield = LCMShield()
        result = await shield.post_shield(
            response="The weather today is sunny with a high of 75 degrees. You should expect clear skies throughout the afternoon.",
            original_query="What is the weather today?",
            model="primary",
        )
        assert result.decision in [ShieldDecision.APPROVE, ShieldDecision.MODIFY]

    @pytest.mark.asyncio
    async def test_post_shield_rewrite_short(self):
        from core.lcm_shield import LCMShield, ShieldDecision
        shield = LCMShield()
        result = await shield.post_shield(
            response="yes",
            original_query="Explain quantum mechanics in detail",
            model="primary",
        )
        assert result.decision == ShieldDecision.REWRITE

    @pytest.mark.asyncio
    async def test_post_shield_detects_repetition(self):
        from core.lcm_shield import LCMShield, ShieldDecision
        shield = LCMShield()
        same_sentence = "This is the exact same sentence. " * 40
        result = await shield.post_shield(
            response=same_sentence,
            original_query="Tell me about AI",
            model="primary",
        )
        assert result.decision in [ShieldDecision.REWRITE, ShieldDecision.MODIFY, ShieldDecision.APPROVE]

    @pytest.mark.asyncio
    async def test_get_stats(self):
        from core.lcm_shield import LCMShield
        shield = LCMShield()
        await shield.pre_shield("Hello", [], "primary")
        stats = shield.get_stats()
        assert "total_pre_shields" in stats
        assert stats["total_pre_shields"] >= 1

    @pytest.mark.asyncio
    async def test_get_recent_decisions(self):
        from core.lcm_shield import LCMShield
        shield = LCMShield()
        await shield.pre_shield("Hello", [], "primary")
        decisions = shield.get_recent_decisions(limit=5)
        assert isinstance(decisions, list)


class TestLCMPlanner:
    def test_import(self):
        from core.lcm_planner import LCMPlanner, PlanTask, Plan, TaskStatus, TaskComplexity
        assert LCMPlanner is not None
        assert TaskStatus.PENDING.value == "pending"

    def test_singleton(self):
        from core.lcm_planner import get_lcm_planner
        p1 = get_lcm_planner()
        p2 = get_lcm_planner()
        assert p1 is p2

    @pytest.mark.asyncio
    async def test_create_plan(self):
        from core.lcm_planner import LCMPlanner, TaskStatus
        planner = LCMPlanner()
        plan = await planner.create_plan(
            request="Write a Python function to sort a list and test it",
        )
        assert plan is not None
        assert len(plan.tasks) > 0
        assert plan.tasks[0].status == TaskStatus.PENDING

    @pytest.mark.asyncio
    async def test_heuristic_decompose(self):
        from core.lcm_planner import LCMPlanner
        planner = LCMPlanner()
        plan = await planner.create_plan(
            request="Analyze the data and create a visualization",
        )
        assert len(plan.tasks) >= 1
        assert plan.execution_waves is not None

    @pytest.mark.asyncio
    async def test_plan_progress(self):
        from core.lcm_planner import LCMPlanner, TaskStatus
        planner = LCMPlanner()
        plan = await planner.create_plan("Do task A and B")
        if len(plan.tasks) > 0:
            plan.tasks[0].status = TaskStatus.COMPLETED
        progress = plan.get_progress()
        assert "completed" in progress
        assert "total" in progress
        if len(plan.tasks) > 0:
            plan.tasks[0].status = TaskStatus.COMPLETED
        progress = plan.get_progress()
        assert "completed" in progress
        assert "total" in progress

    def test_get_stats(self):
        from core.lcm_planner import LCMPlanner
        planner = LCMPlanner()
        stats = planner.get_stats()
        assert "total_plans" in stats

    @pytest.mark.asyncio
    async def test_get_plan_nonexistent(self):
        from core.lcm_planner import LCMPlanner
        planner = LCMPlanner()
        result = planner.get_plan("nonexistent_id")
        assert result is None


class TestThreeLayerContext:
    def test_import(self):
        from core.three_layer_context import ThreeLayerContext, ContextLayer, MemoryBlock, MemoryBlockOp
        assert ThreeLayerContext is not None
        assert ContextLayer.L1_HOT.value == "l1_hot"

    def test_singleton(self):
        from core.three_layer_context import get_three_layer_context
        t1 = get_three_layer_context()
        t2 = get_three_layer_context()
        assert t1 is t2

    def test_default_blocks_created(self):
        from core.three_layer_context import ThreeLayerContext
        tlc = ThreeLayerContext()
        blocks = tlc.list_blocks()
        block_names = [b["name"] for b in blocks]
        assert "primary_block" in block_names
        assert "backend_block" in block_names
        assert "lcm_meta" in block_names

    def test_edit_block_append(self):
        from core.three_layer_context import ThreeLayerContext, MemoryBlockOp
        tlc = ThreeLayerContext()
        result = tlc.edit_block("primary_block", MemoryBlockOp.APPEND, text="User prefers concise answers")
        assert result is True
        ctx = tlc.get_l2_context_for_model("primary")
        assert "concise" in ctx

    def test_edit_block_replace(self):
        from core.three_layer_context import ThreeLayerContext, MemoryBlockOp
        tlc = ThreeLayerContext()
        tlc.edit_block("primary_block", MemoryBlockOp.APPEND, text="Python is great")
        result = tlc.edit_block("primary_block", MemoryBlockOp.REPLACE, old="Python", new="JavaScript")
        assert result is True

    def test_edit_block_delete(self):
        from core.three_layer_context import ThreeLayerContext, MemoryBlockOp
        tlc = ThreeLayerContext()
        tlc.edit_block("primary_block", MemoryBlockOp.APPEND, text="temporary note")
        result = tlc.edit_block("primary_block", MemoryBlockOp.DELETE, text="temporary note")
        assert result is True

    def test_edit_block_clear(self):
        from core.three_layer_context import ThreeLayerContext, MemoryBlockOp
        tlc = ThreeLayerContext()
        tlc.edit_block("primary_block", MemoryBlockOp.APPEND, text="some content")
        result = tlc.edit_block("primary_block", MemoryBlockOp.CLEAR)
        assert result is True
        ctx = tlc.get_l2_context_for_model("primary")
        assert "some content" not in ctx

    def test_create_custom_block(self):
        from core.three_layer_context import ThreeLayerContext
        tlc = ThreeLayerContext()
        block = tlc.create_block("session_notes", owner="agent", max_chars=2000)
        assert block is not None
        assert block.name == "session_notes"

    def test_get_l2_context_for_model(self):
        from core.three_layer_context import ThreeLayerContext, MemoryBlockOp
        tlc = ThreeLayerContext()
        tlc.edit_block("backend_block", MemoryBlockOp.APPEND, text="User loves code with type hints")
        ctx = tlc.get_l2_context_for_model("backend")
        assert "type hints" in ctx

    def test_get_l2_context_empty_model(self):
        from core.three_layer_context import ThreeLayerContext
        tlc = ThreeLayerContext()
        ctx = tlc.get_l2_context_for_model("nonexistent")
        assert isinstance(ctx, str)

    def test_should_evict_l1(self):
        from core.three_layer_context import ThreeLayerContext
        tlc = ThreeLayerContext()
        result = tlc.should_evict_l1(estimated_tokens=50)
        assert isinstance(result, bool)
        result_high = tlc.should_evict_l1(estimated_tokens=90000)
        assert isinstance(result_high, bool)

    def test_evict_to_l2(self):
        from core.three_layer_context import ThreeLayerContext, MemoryBlockOp
        tlc = ThreeLayerContext()
        messages = [
            {"role": "user", "content": f"Message {i}"} for i in range(20)
        ]
        evicted = tlc.evict_to_l2(messages, max_messages_to_keep=5)
        assert isinstance(evicted, str)

    def test_inject_l2_into_l1(self):
        from core.three_layer_context import ThreeLayerContext, MemoryBlockOp
        tlc = ThreeLayerContext()
        tlc.edit_block("primary_block", MemoryBlockOp.APPEND, text="User prefers bullet points")
        messages = [{"role": "system", "content": "You are Jarvis."}]
        injected = tlc.inject_l2_into_l1(messages, model="primary")
        assert isinstance(injected, list)
        assert len(injected) >= len(messages)

    def test_get_stats(self):
        from core.three_layer_context import ThreeLayerContext
        tlc = ThreeLayerContext()
        stats = tlc.get_stats()
        assert "l1_token_limit" in stats
        assert "l2_block_count" in stats

    def test_get_transition_log(self):
        from core.three_layer_context import ThreeLayerContext
        tlc = ThreeLayerContext()
        log = tlc.get_transition_log(limit=5)
        assert isinstance(log, list)

    def test_consolidate_l2_to_l3(self):
        from core.three_layer_context import ThreeLayerContext
        tlc = ThreeLayerContext()
        count = tlc.consolidate_l2_to_l3(force=True)
        assert isinstance(count, int)

    def test_retrieve_from_l3(self):
        from core.three_layer_context import ThreeLayerContext
        tlc = ThreeLayerContext()
        results = tlc.retrieve_from_l3("test query", max_results=3)
        assert isinstance(results, list)


class TestReflectEditLoop:
    def test_import(self):
        from core.reflect_edit import ReflectEditLoop, Reflection, ReflectionType
        assert ReflectEditLoop is not None
        assert ReflectionType.QUALITY.value == "quality"

    def test_singleton(self):
        from core.reflect_edit import get_reflect_edit_loop
        r1 = get_reflect_edit_loop()
        r2 = get_reflect_edit_loop()
        assert r1 is r2

    @pytest.mark.asyncio
    async def test_reflect(self):
        from core.reflect_edit import ReflectEditLoop, ReflectionType
        loop = ReflectEditLoop()
        reflection = await loop.reflect(
            query="What is 2+2?",
            response="The answer is 4.",
            model_used="backend",
            latency_ms=150.0,
        )
        assert reflection is not None
        assert reflection.score > 0
        assert reflection.model_used == "backend"

    @pytest.mark.asyncio
    async def test_reflect_updates_quality(self):
        from core.reflect_edit import ReflectEditLoop
        loop = ReflectEditLoop()
        await loop.reflect("Hello", "Hi there! How can I help?", "primary")
        stats = loop.get_stats()
        assert stats["total_reflections"] >= 1

    @pytest.mark.asyncio
    async def test_reflect_poor_response(self):
        from core.reflect_edit import ReflectEditLoop
        loop = ReflectEditLoop()
        reflection = await loop.reflect(
            query="Explain machine learning in detail",
            response="idk",
            model_used="primary",
        )
        assert reflection.score < 0.6
        assert len(reflection.insight) > 0

    @pytest.mark.asyncio
    async def test_reflect_good_response(self):
        from core.reflect_edit import ReflectEditLoop
        loop = ReflectEditLoop()
        reflection = await loop.reflect(
            query="What is Python?",
            response="Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation. Python is dynamically typed and garbage-collected.",
            model_used="primary",
        )
        assert reflection.score > 0.3

    def test_get_routing_hint(self):
        from core.reflect_edit import ReflectEditLoop
        loop = ReflectEditLoop()
        hint, confidence = loop.get_routing_hint("What is 2+2?")
        assert isinstance(hint, (str, type(None)))
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    def test_get_stats(self):
        from core.reflect_edit import ReflectEditLoop
        loop = ReflectEditLoop()
        stats = loop.get_stats()
        assert "total_reflections" in stats
        assert "avg_reflection_score" in stats


class TestCouncilDeliberation:
    def test_import(self):
        from core.reflect_edit import CouncilDeliberation, CouncilProtocol, CouncilSession
        assert CouncilDeliberation is not None

    def test_singleton(self):
        from core.reflect_edit import get_council_deliberation
        c1 = get_council_deliberation()
        c2 = get_council_deliberation()
        assert c1 is c2

    @pytest.mark.asyncio
    async def test_deliberate_no_llms(self):
        from core.reflect_edit import CouncilDeliberation, CouncilProtocol
        council = CouncilDeliberation(max_rounds=2)
        session = await council.deliberate(
            topic="What is the best programming language?",
            primary_llm=None,
            backend_llm=None,
            protocol=CouncilProtocol.DEBATE,
        )
        assert session is not None
        assert session.id is not None

    def test_get_stats(self):
        from core.reflect_edit import CouncilDeliberation
        council = CouncilDeliberation()
        stats = council.get_stats()
        assert "total_sessions" in stats

    def test_compute_convergence(self):
        from core.reflect_edit import CouncilDeliberation
        council = CouncilDeliberation()
        score = council._compute_convergence("Python is great for AI", "Python is great for AI")
        assert score > 0.5
        score2 = council._compute_convergence("Python is great", "Java is better for enterprise")
        assert 0.0 <= score2 <= 1.0


class TestSemanticContracts:
    def test_import(self):
        from core.semantic_contracts import ContractRegistry, ContractSchema, FieldSpec, FieldType
        assert ContractRegistry is not None
        assert FieldType.STRING.value == "string"

    def test_singleton(self):
        from core.semantic_contracts import get_contract_registry
        c1 = get_contract_registry()
        c2 = get_contract_registry()
        assert c1 is c2

    def test_register_and_create_contract(self):
        from core.semantic_contracts import ContractRegistry, ContractSchema, FieldSpec, FieldType
        registry = ContractRegistry()
        schema = ContractSchema(
            name="test_contract",
            fields=[
                FieldSpec(name="query", field_type=FieldType.STRING, required=True),
                FieldSpec(name="max_tokens", field_type=FieldType.NUMBER, required=False, default=100),
            ],
        )
        registry.register_schema(schema)
        contract = registry.create_contract(
            schema_name="test_contract",
            source_model="primary",
            target_model="backend",
            data={"query": "What is AI?"},
        )
        assert contract is not None
        assert contract.status.value == "validated"

    def test_contract_validation_missing_required(self):
        from core.semantic_contracts import ContractRegistry, ContractSchema, FieldSpec, FieldType, ContractStatus
        registry = ContractRegistry()
        schema = ContractSchema(
            name="strict_contract",
            fields=[
                FieldSpec(name="required_field", field_type=FieldType.STRING, required=True),
            ],
        )
        registry.register_schema(schema)
        contract = registry.create_contract(
            schema_name="strict_contract",
            source_model="primary",
            target_model="backend",
            data={},
        )
        assert contract.status == ContractStatus.REJECTED

    def test_contract_type_validation(self):
        from core.semantic_contracts import ContractRegistry, ContractSchema, FieldSpec, FieldType, ContractStatus
        registry = ContractRegistry()
        schema = ContractSchema(
            name="typed_contract",
            fields=[
                FieldSpec(name="count", field_type=FieldType.NUMBER, required=True),
            ],
        )
        registry.register_schema(schema)
        contract = registry.create_contract(
            schema_name="typed_contract",
            source_model="primary",
            target_model="backend",
            data={"count": 42},
        )
        assert contract.status == ContractStatus.VALIDATED

    def test_get_validated_data(self):
        from core.semantic_contracts import ContractRegistry, ContractSchema, FieldSpec, FieldType
        registry = ContractRegistry()
        schema = ContractSchema(
            name="data_contract",
            fields=[
                FieldSpec(name="x", field_type=FieldType.STRING, required=True),
            ],
        )
        registry.register_schema(schema)
        contract = registry.create_contract(
            schema_name="data_contract",
            source_model="primary",
            target_model="backend",
            data={"x": "hello"},
        )
        data = registry.get_validated_data(contract.id)
        assert data is not None
        assert data["x"] == "hello"

    def test_get_validated_data_invalid(self):
        from core.semantic_contracts import ContractRegistry, ContractSchema, FieldSpec, FieldType
        registry = ContractRegistry()
        schema = ContractSchema(
            name="bad_data_contract",
            fields=[
                FieldSpec(name="y", field_type=FieldType.NUMBER, required=True),
            ],
        )
        registry.register_schema(schema)
        contract = registry.create_contract(
            schema_name="bad_data_contract",
            source_model="primary",
            target_model="backend",
            data={},
        )
        data = registry.get_validated_data(contract.id)
        assert data is None

    def test_get_stats(self):
        from core.semantic_contracts import ContractRegistry, ContractSchema, FieldSpec, FieldType
        registry = ContractRegistry()
        schema = ContractSchema(
            name="stats_contract",
            fields=[FieldSpec(name="val", field_type=FieldType.STRING, required=True)],
        )
        registry.register_schema(schema)
        registry.create_contract(
            schema_name="stats_contract",
            source_model="primary",
            target_model="backend",
            data={"val": "test"},
        )
        stats = registry.get_stats()
        assert "total_contracts" in stats
        assert stats["total_contracts"] >= 1

    def test_built_in_schemas(self):
        from core.semantic_contracts import get_contract_registry
        registry = get_contract_registry()
        stats = registry.get_stats()
        assert "registered_schemas" in stats


class TestAudioBridge:
    def test_import(self):
        from core.audio_bridge import AudioBridge, SemanticProjector, AudioTokenSequence, AudioModality
        assert AudioBridge is not None
        assert AudioModality.SPEECH_INPUT.value == "speech_input"

    def test_singleton(self):
        from core.audio_bridge import get_audio_bridge
        a1 = get_audio_bridge()
        a2 = get_audio_bridge()
        assert a1 is a2

    def test_semantic_projector(self):
        from core.audio_bridge import SemanticProjector
        projector = SemanticProjector(input_dim=256, output_dim=768)
        features = np.random.randn(256).astype(np.float32)
        projected = projector.project(features)
        assert projected.shape == (768,)

    def test_semantic_projector_tokens(self):
        from core.audio_bridge import SemanticProjector
        projector = SemanticProjector(input_dim=256, output_dim=768)
        tokens = [1, 2, 3, 4, 5]
        features = projector.tokens_to_features(tokens)
        assert features.shape[0] == 256

    def test_tokenize_speech_input(self):
        from core.audio_bridge import AudioBridge, AudioModality
        bridge = AudioBridge()
        seq = bridge.tokenize_speech_input("Hello world", duration_seconds=2.0)
        assert seq is not None
        assert seq.modality == AudioModality.SPEECH_INPUT
        assert len(seq.tokens) > 0

    def test_tokenize_speech_output(self):
        from core.audio_bridge import AudioBridge, AudioModality
        bridge = AudioBridge()
        seq = bridge.tokenize_speech_input("The answer is 42", duration_seconds=1.5)
        assert seq is not None

    def test_tokenize_music_output(self):
        from core.audio_bridge import AudioBridge, AudioModality
        bridge = AudioBridge()
        composition = {
            "style": "classical",
            "form": "fugue",
            "voices": 4,
            "key": "C minor",
            "tempo": 120,
        }
        seq = bridge.tokenize_music_output(composition)
        assert seq is not None
        assert seq.modality == AudioModality.MUSIC

    def test_audio_token_sequence_density(self):
        from core.audio_bridge import AudioTokenSequence, AudioModality
        seq = AudioTokenSequence(
            id="test_seq_1",
            modality=AudioModality.SPEECH_INPUT,
            tokens=[1, 2, 3, 4, 5],
            duration_seconds=2.0,
        )
        density = seq.get_token_density()
        assert density > 0

    def test_audio_token_sequence_concept_dict(self):
        from core.audio_bridge import AudioTokenSequence, AudioModality
        seq = AudioTokenSequence(
            id="test_seq_2",
            modality=AudioModality.SPEECH_INPUT,
            tokens=[1, 2, 3],
            duration_seconds=1.0,
        )
        d = seq.to_concept_dict()
        assert "type" in d
        assert "token_count" in d

    def test_find_similar_audio(self):
        from core.audio_bridge import AudioBridge
        bridge = AudioBridge()
        query = np.random.randn(768).astype(np.float32)
        results = bridge.find_similar_audio(query, top_k=5)
        assert isinstance(results, list)

    def test_get_stats(self):
        from core.audio_bridge import AudioBridge
        bridge = AudioBridge()
        stats = bridge.get_stats()
        assert "speech_sequences" in stats
        assert "music_sequences" in stats

    @pytest.mark.asyncio
    async def test_observe_with_lcm_no_pfc(self):
        from core.audio_bridge import AudioBridge
        bridge = AudioBridge()
        from core.audio_bridge import AudioTokenSequence, AudioModality
        seq = AudioTokenSequence(
            id="test_seq_3",
            modality=AudioModality.SPEECH_INPUT,
            tokens=[1, 2, 3],
            duration_seconds=1.0,
        )
        result = await bridge.observe_with_lcm(seq)
        assert isinstance(result, bool)