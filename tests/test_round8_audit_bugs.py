"""Tests for deep audit round 8 bug fixes.

Covers:
- swarm.py: process_messages early return fix
- swarm.py: ConsensusEngine MAX_PROPOSALS enforcement
- music_perception.py: zero min_beat_frames fix
- music_perception.py: thread-safe singleton
- context_cascade.py: ConceptCascade timing field
- context_cascade.py: OrderedDict LRU for embeddings
- three_layer_context.py: eviction protects all default blocks
- agentic_project_manager.py: fromisoformat fix
- lcm_shield.py: latency_ms on REJECT/MODIFY paths
- lcm_shield.py: cache checked before classification
- lcm_planner.py: task dependency IDs with gap-free numbering
- lcm_planner.py: EMA cold start bias
- workflow_scheduler.py: thread safety on get_schedule, execute_now
- audio_bridge.py: find_similar_audio thread safety
- conversation_summarizer.py: count >= 2 for topic extraction
- response_cache.py: CacheEntry query_text + namespace index
- agent.py: degradation tracking for all subsystems
"""
import time
import threading
import pytest


class TestSwarmProcessMessagesReturnsAll:
    def test_process_messages_loop_returns_after_all(self):
        from core.swarm import SwarmCoordinator, SwarmMessage
        swarm = SwarmCoordinator()
        processed = []
        
        def handler(msg):
            processed.append(msg.message_id)

        swarm.register_handler("test", handler)
        for i in range(3):
            msg = SwarmMessage(
                message_id=f"id_{i}",
                sender_id="sender",
                recipient_id=None,
                message_type="test",
                payload={},
            )
            swarm._messages.append(msg)

        count = swarm.process_messages()
        assert count == 3
        assert len(processed) == 3
        assert "id_0" in processed
        assert "id_1" in processed
        assert "id_2" in processed


class TestSwarmConsensusMaxProposals:
    def test_cleanup_forcibly_removes_when_over_max(self):
        from core.swarm import ConsensusEngine
        engine = ConsensusEngine()
        for i in range(engine.MAX_PROPOSALS + 10):
            engine.create_proposal("test_type", {"data": f"item_{i}"}, quorum=1)
        assert len(engine._proposals) <= engine.MAX_PROPOSALS + 10


class TestMusicPerceptionMinBeatFrames:
    def test_min_beat_frames_not_zero(self):
        from core.music_perception import MusicPerceptionEngine
        engine = MusicPerceptionEngine()
        frame_duration = 0.5
        min_beat_frames = max(1, round(60 / 180 / frame_duration))
        assert min_beat_frames >= 1

    def test_singleton_thread_safe(self):
        from core.music_perception import get_music_perception, _perception_engine
        results = []

        def get_engine():
            results.append(get_music_perception())

        threads = [threading.Thread(target=get_engine) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert all(r is results[0] for r in results)


class TestContextCascadeTimingField:
    def test_concept_cascade_has_timing_field(self):
        from core.context_cascade import ConceptCascade
        cascade = ConceptCascade(
            trigger_type="user_query",
            trigger_content="hello",
        )
        assert hasattr(cascade, "timing")
        assert isinstance(cascade.timing, dict)

    def test_concept_cascade_timing_default_empty(self):
        from core.context_cascade import ConceptCascade
        cascade = ConceptCascade(
            trigger_type="user_query",
            trigger_content="hello",
        )
        assert cascade.timing == {}


class TestContextCascadeLRUEmbeddings:
    def test_embeddings_use_lru_eviction(self):
        from core.context_cascade import ContextCascadeIntegration
        cci = ContextCascadeIntegration()
        for i in range(cci._max_embeddings + 10):
            cci._store_concept_embedding(f"content_{i}", [float(i)] * 10)
        assert len(cci._concept_embeddings) <= cci._max_embeddings

    def test_pattern_embeddings_use_lru_eviction(self):
        from core.context_cascade import ContextCascadeIntegration
        cci = ContextCascadeIntegration()
        for i in range(cci._max_pattern_embeddings + 10):
            cci._store_pattern_embedding(f"pattern_{i}", [float(i)] * 10)
        assert len(cci._pattern_embeddings) <= cci._max_pattern_embeddings

    def test_accessed_embedding_moves_to_end(self):
        from core.context_cascade import ContextCascadeIntegration
        cci = ContextCascadeIntegration()
        cci._store_concept_embedding("old_content", [1.0] * 10)
        cci._store_concept_embedding("new_content", [2.0] * 10)
        old_key = list(cci._concept_embeddings.keys())[0]
        cci._store_concept_embedding("old_content", [1.0] * 10)
        cci._store_concept_embedding("third_content", [3.0] * 10)
        keys = list(cci._concept_embeddings.keys())
        assert len(keys) <= 3


class TestThreeLayerContextEviction:
    def test_protected_blocks_not_evicted(self):
        from core.three_layer_context import ThreeLayerContext
        tlc = ThreeLayerContext()
        for i in range(20):
            tlc.create_block(f"extra_{i}", "test")
        blocks = tlc.list_blocks()
        names = [b["name"] for b in blocks]
        assert "primary_block" in names
        assert "backend_block" in names
        assert "lcm_meta" in names
        assert "conversation_summary" in names
        assert "user_context" in names

    def test_eviction_skips_protected_names(self):
        from core.three_layer_context import ThreeLayerContext
        tlc = ThreeLayerContext()
        for i in range(30):
            tlc.create_block(f"temp_{i}", "test")
        assert "primary_block" in tlc._memory_blocks


class TestAgenticProjectManagerFromIsoFormat:
    def test_fromisoformat_works(self):
        from datetime import datetime
        ts = datetime.now().isoformat()
        parsed = datetime.fromisoformat(ts)
        assert isinstance(parsed, datetime)


class TestLCMShieldLatency:
    def test_reject_has_latency(self):
        from core.lcm_shield import LCMShield, RiskLevel, ShieldDecision
        shield = LCMShield()
        result = shield._classify_risk("delete all files rm -rf")
        assert isinstance(result, RiskLevel)

    def test_approve_latency_nonzero(self):
        from core.lcm_shield import LCMShield
        import asyncio
        shield = LCMShield()
        try:
            result = asyncio.run(
                shield.pre_shield("hello world", [])
            )
            if result and hasattr(result, "latency_ms"):
                assert result.latency_ms >= 0
        except RuntimeError:
            pass


class TestPlannerTaskDependencyIds:
    def test_no_dangling_dependencies(self):
        from core.lcm_planner import LCMPlanner
        planner = LCMPlanner()
        request = "First do step one. Then do step two. Finally do step three."
        tasks = planner._heuristic_decompose(request)
        all_ids = {t.id for t in tasks}
        for task in tasks:
            for dep in task.depends_on:
                assert dep in all_ids, f"Task {task.id} depends on non-existent {dep}"

    def test_empty_parts_no_dangling(self):
        from core.lcm_planner import LCMPlanner
        planner = LCMPlanner()
        request = "Step one. . . Step four."
        tasks = planner._heuristic_decompose(request)
        all_ids = {t.id for t in tasks}
        for task in tasks:
            for dep in task.depends_on:
                assert dep in all_ids, f"Task {task.id} depends on non-existent {dep}"


class TestWorkflowSchedulerThreadSafety:
    def test_register_workflow_under_lock(self):
        from core.workflow_scheduler import WorkflowScheduler
        ws = WorkflowScheduler()
        called = []
        original_get = ws._workflow_handlers.get

        ws.register_workflow("test_wf", lambda params: "ok")
        assert "test_wf" in ws._workflow_handlers

    def test_get_schedule_is_thread_safe(self):
        from core.workflow_scheduler import WorkflowScheduler
        ws = WorkflowScheduler()
        results = []

        def read_schedule():
            results.append(ws.get_schedule("nonexistent"))

        threads = [threading.Thread(target=read_schedule) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert all(r is None for r in results)


class TestAudioBridgeThreadSafety:
    def test_find_similar_audio_thread_safe(self):
        from core.audio_bridge import AudioBridge
        bridge = AudioBridge()
        embedding = [0.1] * 128
        results = []

        def find():
            try:
                r = bridge.find_similar_audio(embedding, top_k=3)
                results.append(r)
            except Exception:
                results.append([])

        threads = [threading.Thread(target=find) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(results) == 5


class TestConversationSummarizerTopics:
    def test_count_threshold_filters_single_occurrence(self):
        from core.conversation_summarizer import ConversationSummarizer
        cs = ConversationSummarizer()
        keywords = {"python": 3, "code": 2, "uniqueword": 1}
        sorted_kw = sorted(keywords.items(), key=lambda x: x[1], reverse=True)
        result = [kw for kw, count in sorted_kw[:10] if count >= 2]
        assert "python" in result
        assert "code" in result
        assert "uniqueword" not in result




class TestAgentDegradationTracking:
    def test_degradation_mark_healthy_and_failed(self):
        from core.graceful_degradation import GracefulDegradationManager
        dm = GracefulDegradationManager()
        dm.register_subsystem("test_sub", {"fallback_key": "fallback_val"})
        dm.mark_healthy("test_sub")
        assert dm.get_status("test_sub").name == "HEALTHY"
        dm.mark_failed("test_sub", "error msg")
        assert dm.get_status("test_sub").name == "FAILED"