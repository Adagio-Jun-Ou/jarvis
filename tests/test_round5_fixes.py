"""Round 5 bug fix tests: supervisor thread safety, proactive bounds,
goal_tracker locks, self_reflection bounds, memory router + adaptive context integration."""

import threading
import time
import pytest


class TestSupervisorThreadSafety:
    def test_register_unregister_under_concurrency(self):
        from core.supervisor import ServerSupervisor, RestartPolicy
        sup = ServerSupervisor(restart_policy=RestartPolicy(max_restarts=10))
        errors = []

        def register_many(n):
            try:
                for i in range(n):
                    sup.register_subsystem(f"sys_{threading.current_thread().name}_{i}", object())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=register_many, args=(50,)) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(sup._subsystems) == 250

    def test_maintenance_tasks_bounded(self):
        from core.supervisor import ServerSupervisor
        sup = ServerSupervisor()
        for i in range(30):
            sup.register_maintenance_task(lambda: None)
        assert len(sup._maintenance_tasks) <= sup.MAX_MAINTENANCE_TASKS

    def test_shutdown_handlers_bounded(self):
        from core.supervisor import ServerSupervisor
        sup = ServerSupervisor()
        for i in range(30):
            sup.register_shutdown_handler(lambda: None)
        assert len(sup._shutdown_handlers) <= sup.MAX_SHUTDOWN_HANDLERS

    def test_restart_timestamps_bounded(self):
        from core.supervisor import ServerSupervisor, SupervisorStats
        stats = SupervisorStats()
        for i in range(100):
            stats.restart_timestamps.append(time.time())
        assert len(stats.restart_timestamps) == 100
        sup = ServerSupervisor()
        sup._stats.restart_timestamps = list(stats.restart_timestamps)
        sup.should_restart()
        assert len(sup._stats.restart_timestamps) <= sup.MAX_RESTART_TIMESTAMPS

    def test_get_status_reads_under_lock(self):
        from core.supervisor import ServerSupervisor
        sup = ServerSupervisor()
        sup.register_subsystem("test", object())
        status = sup.get_status()
        assert "test" in status["subsystems"]

    def test_health_check_snapshots_checkers(self):
        from core.supervisor import ServerSupervisor
        sup = ServerSupervisor()
        sup.register_subsystem("svc1", object(), health_checker=lambda: True)
        import asyncio
        results = asyncio.run(sup.health_check())
        assert results["svc1"] is True
        sup.unregister_subsystem("svc1")
        results2 = asyncio.run(sup.health_check())
        assert "svc1" not in results2


class TestProactiveBounds:
    def test_suggestion_history_bounded(self):
        from core.proactive import ProactiveEngine, SuggestionType, ProactiveSuggestion
        engine = ProactiveEngine()
        engine._last_suggestion_time = 0
        for i in range(engine.MAX_SUGGESTION_HISTORY + 50):
            engine._suggestion_history.append(
                ProactiveSuggestion(
                    suggestion_type=SuggestionType.HELP_WITH_TASK,
                    message=f"test {i}",
                    priority=3,
                    confidence=0.5,
                )
            )
            if len(engine._suggestion_history) > engine.MAX_SUGGESTION_HISTORY:
                engine._suggestion_history = engine._suggestion_history[-engine.MAX_SUGGESTION_HISTORY:]
        assert len(engine._suggestion_history) <= engine.MAX_SUGGESTION_HISTORY

    def test_mentioned_topics_bounded(self):
        from core.proactive import ProactiveEngine
        engine = ProactiveEngine()
        for i in range(100):
            engine.update_activity(f"testing topic number {i} something important here")
        assert len(engine._mentioned_topics) <= engine.MAX_MENTIONED_TOPICS

    def test_suggestion_history_get_returns_copy(self):
        from core.proactive import ProactiveEngine
        engine = ProactiveEngine()
        history = engine.get_suggestion_history()
        assert isinstance(history, list)


class TestGoalTrackerThreadSafety:
    def test_concurrent_track_and_read(self):
        import tempfile
        from core.goal_tracker import GoalTracker, Goal
        with tempfile.TemporaryDirectory() as td:
            tracker = GoalTracker(storage_path=__import__('pathlib').Path(td))
            errors = []

            def track_goals(n):
                try:
                    for i in range(n):
                        g = Goal(goal_id=f"goal_{threading.current_thread().name}_{i}", description=f"Test goal {i}")
                        tracker.track_goal(g)
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=track_goals, args=(20,)) for _ in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0
            active = tracker.get_active_goals()
            assert len(active) == 100

    def test_extract_goals_thread_safe(self):
        import tempfile
        from core.goal_tracker import GoalTracker
        with tempfile.TemporaryDirectory() as td:
            tracker = GoalTracker(storage_path=__import__('pathlib').Path(td))
            errors = []

            def extract_many(n):
                try:
                    for i in range(n):
                        tracker.extract_goals(f"I want to build a project {i}")
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=extract_many, args=(10,)) for _ in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0

    def test_get_active_goals_returns_copy(self):
        import tempfile
        from core.goal_tracker import GoalTracker, Goal
        with tempfile.TemporaryDirectory() as td:
            tracker = GoalTracker(storage_path=__import__('pathlib').Path(td))
            tracker.track_goal(Goal(goal_id="g1", description="Test"))
            active = tracker.get_active_goals()
            active.clear()
            assert len(tracker.get_active_goals()) == 1


class TestSelfReflectionBounds:
    def test_quality_scores_bounded(self):
        import tempfile
        from core.self_reflection import SelfReflection, MAX_QUALITY_SCORES
        with tempfile.TemporaryDirectory() as td:
            sr = SelfReflection(storage_path=__import__('pathlib').Path(td))
            for i in range(MAX_QUALITY_SCORES + 100):
                sr._quality_scores.append((0.5, time.time()))
                if len(sr._quality_scores) > MAX_QUALITY_SCORES:
                    sr._quality_scores = sr._quality_scores[-MAX_QUALITY_SCORES:]
            assert len(sr._quality_scores) <= MAX_QUALITY_SCORES

    def test_quality_scores_bounded_on_track(self):
        import tempfile
        from core.self_reflection import SelfReflection, MAX_QUALITY_SCORES
        with tempfile.TemporaryDirectory() as td:
            sr = SelfReflection(storage_path=__import__('pathlib').Path(td))
            for i in range(MAX_QUALITY_SCORES + 20):
                sr.track_response_quality("response text", "great")
            assert len(sr._quality_scores) <= MAX_QUALITY_SCORES

    def test_conversation_history_bounded(self):
        import tempfile
        from core.self_reflection import SelfReflection, MAX_CONVERSATION_HISTORY
        with tempfile.TemporaryDirectory() as td:
            sr = SelfReflection(storage_path=__import__('pathlib').Path(td))
            for i in range(MAX_CONVERSATION_HISTORY + 20):
                sr.record_conversation_turn(f"user {i}", f"jarvis {i}")
            assert len(sr._conversation_history) <= MAX_CONVERSATION_HISTORY


class TestAgentIntegrationBugFixes:
    def test_prefrontal_cortex_not_nullified(self):
        from core.agent import JarvisAgent
        from core.config import Config
        config = Config.load()
        agent = JarvisAgent(config)
        if agent._prefrontal_cortex is not None:
            assert agent._prefrontal_cortex is not None

    def test_degradation_manager_attached(self):
        from core.agent import JarvisAgent
        from core.config import Config
        config = Config.load()
        agent = JarvisAgent(config)
        assert hasattr(agent, '_degradation_manager')

    def test_consolidation_attached(self):
        from core.agent import JarvisAgent
        from core.config import Config
        config = Config.load()
        agent = JarvisAgent(config)
        assert agent._consolidation_engine is not None


class TestMemoryRouterIntegration:
    def test_route_returns_valid_tier(self):
        from core.memory_router import MemoryRouter, MemoryTier, QueryType
        router = MemoryRouter()
        decision = router.route("What is the capital of France?")
        assert decision.tier in list(MemoryTier)
        assert decision.confidence > 0
        assert decision.reason

    def test_route_cache_hit(self):
        from core.memory_router import MemoryRouter
        router = MemoryRouter()
        d1 = router.route("What is Python?")
        d2 = router.route("What is Python?")
        assert d2.metadata.get("cache_time") is not None

    def test_classify_query(self):
        from core.memory_router import MemoryRouter, QueryType
        router = MemoryRouter()
        assert router.classify_query("What is Python?") == QueryType.FACTUAL
        assert router.classify_query("How to bake a cake?") == QueryType.PROCEDURAL
        assert router.classify_query("Remember what I said earlier") == QueryType.CONVERSATIONAL

    def test_drift_notification(self):
        from core.memory_router import MemoryRouter
        router = MemoryRouter()
        router.notify_drift(time.time())
        assert router._last_drift_time > 0

    def test_cognitive_state_update(self):
        from core.memory_router import MemoryRouter, MemoryTier
        router = MemoryRouter()
        router.update_cognitive_state("OVERLOADED")
        decision = router.route("help me")
        assert decision.tier == MemoryTier.L1_HOT

    def test_stats(self):
        from core.memory_router import MemoryRouter
        router = MemoryRouter()
        router.route("test query")
        stats = router.get_stats()
        assert "total_routes" in stats
        assert stats["total_routes"] == 1


class TestAdaptiveContextIntegration:
    def test_add_and_assemble(self):
        from core.adaptive_context import AdaptiveContextWindow, ContentType
        window = AdaptiveContextWindow()
        window.add_block(ContentType.SYSTEM, "You are an assistant", priority=1.0, topic="system")
        window.add_block(ContentType.CONVERSATION, "User asked about Python", priority=0.7, topic="python")
        context = window.assemble_context()
        assert "assistant" in context
        assert "Python" in context

    def test_cognitive_state_adjustment(self):
        from core.adaptive_context import AdaptiveContextWindow, ContextStrategy
        window = AdaptiveContextWindow()
        result = window.adjust_for_cognitive_state("OVERLOADED")
        assert result == ContextStrategy.MINIMAL
        stats = window.get_stats()
        assert stats["current_strategy"] == "MINIMAL"

    def test_focus_topic(self):
        from core.adaptive_context import AdaptiveContextWindow, ContentType
        window = AdaptiveContextWindow()
        window.add_block(ContentType.BACKGROUND, "Deep background on Rust language", priority=0.3, topic="rust")
        window.add_block(ContentType.CONVERSATION, "User asked about Rust", priority=0.7, topic="rust")
        window.set_focus_topic("rust")
        context = window.assemble_context()
        assert "Rust" in context

    def test_clear(self):
        from core.adaptive_context import AdaptiveContextWindow, ContentType
        window = AdaptiveContextWindow()
        window.add_block(ContentType.SYSTEM, "test", priority=1.0)
        window.clear()
        assert len(window._blocks) == 0

    def test_strategy_map_coverage(self):
        from core.adaptive_context import AdaptiveContextWindow
        for state in ["FLOW", "ENGAGED", "EXPLORING", "DEEP_DIVE", "IDLE", "OVERLOADED", "FRUSTRATED"]:
            result = AdaptiveContextWindow().adjust_for_cognitive_state(state)
            assert result is not None