"""Integration tests for butler API endpoints and enhanced features."""
import pytest
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))


class TestButlerAPIEndpoints:
    def test_presence_state_has_deep_work(self):
        from core.active_presence import ActivePresenceEngine
        engine = ActivePresenceEngine()
        state = engine.get_state()
        assert "deep_work_minutes" in state
        assert "distraction_count" in state
        assert "deep_work_periods" in state["rhythm"]
        assert "breaks_taken" in state["rhythm"]
        assert "longest_focus_minutes" in state["rhythm"]

    def test_presence_deep_work_transition(self):
        from core.active_presence import ActivePresenceEngine, UserPresence, PresenceTransition
        engine = ActivePresenceEngine()
        engine._state.deep_work_start = time.time() - 700
        engine._state.presence = UserPresence.FOCUSED
        engine._state.focus_score = 0.8
        result = engine.tick()
        assert result["presence"] in (UserPresence.DEEP_WORK.value, UserPresence.FOCUSED.value)

    def test_presence_checkin_cooldown(self):
        from core.active_presence import ActivePresenceEngine, UserPresence, CheckIn
        engine = ActivePresenceEngine()
        engine._state.presence = UserPresence.ACTIVE
        engine._state.last_checkin_time = time.time()
        ci = CheckIn(reason="test", message="test", urgency="low")
        engine._state.pending_checkins.append(ci)
        result = engine.should_check_in()
        assert result is None

    def test_presence_distracted_context(self):
        from core.active_presence import ActivePresenceEngine, UserPresence
        engine = ActivePresenceEngine()
        engine._state.presence = UserPresence.DISTRACTED
        engine._state.distraction_count = 5
        ctx = engine.get_presence_context()
        assert "distracted" in ctx.lower()

    def test_presence_deep_work_context(self):
        from core.active_presence import ActivePresenceEngine, UserPresence
        engine = ActivePresenceEngine()
        engine._state.presence = UserPresence.DEEP_WORK
        engine._state.deep_work_start = time.time() - 600
        ctx = engine.get_presence_context()
        assert "deep work" in ctx.lower() or "deep_work" in ctx.lower()

    def test_goal_anticipation_detail_patterns(self):
        from core.goal_anticipation import GoalAnticipationEngine
        engine = GoalAnticipationEngine()
        antic = engine.record_action("code_edit", "error in module.py traceback")
        if antic:
            assert antic.suggested_action is not None
            assert antic.confidence > 0

    def test_goal_anticipation_schedule(self):
        from core.goal_anticipation import GoalAnticipationEngine
        engine = GoalAnticipationEngine()
        result = engine.tick()
        assert isinstance(result, dict)
        assert "active_anticipations" in result or "total_anticipated" in result

    def test_goal_anticipation_sequence(self):
        from core.goal_anticipation import GoalAnticipationEngine, ActionSequence
        engine = GoalAnticipationEngine()
        for _ in range(3):
            engine.record_action("code_edit")
            engine.record_action("run_tests")
        engine._state.sequence_history.append(ActionSequence(actions=["code_edit", "run_tests", "commit"]))
        engine._state.sequence_history[-1].count = 3
        antic = engine._anticipate_from_sequence()
        if antic:
            assert antic.source.value == "sequence"

    def test_goal_anticipation_extra_goals(self):
        from core.goal_anticipation import GoalAnticipationEngine
        engine = GoalAnticipationEngine()
        engine.set_active_goals(["debug the api"])
        antic = engine.record_action("code_edit")
        if antic:
            assert isinstance(antic.description, str)

    def test_observer_emotional_tone(self):
        from core.autonomous_observer import AutonomousObserver, ObservationType
        observer = AutonomousObserver()
        obs = observer.observe_emotional_tone("frustrated", intensity=0.7, trigger="test failure")
        assert obs.obs_type == ObservationType.EMOTIONAL
        state = observer.get_state()
        assert "emotional_tone" in state
        assert state["emotional_tone"] == "frustrated"

    def test_observer_emotional_trajectory(self):
        from core.autonomous_observer import AutonomousObserver
        observer = AutonomousObserver()
        observer.observe_emotional_tone("curious", intensity=0.4)
        observer.observe_emotional_tone("excited", intensity=0.8)
        trajectory = observer.get_emotional_trajectory(limit=2)
        assert len(trajectory) == 2
        assert trajectory[0]["tone"] == "curious"

    def test_observer_file_change(self):
        from core.autonomous_observer import AutonomousObserver, ObservationType
        observer = AutonomousObserver()
        obs = observer.observe_file_change("/src/module.py", "modified")
        assert obs.obs_type == ObservationType.FILE_CHANGE
        assert "module.py" in obs.content

    def test_observer_network_event(self):
        from core.autonomous_observer import AutonomousObserver, ObservationType
        observer = AutonomousObserver()
        obs = observer.observe_network_event("api_timeout", "API call timed out")
        assert obs.obs_type == ObservationType.NETWORK

    def test_observer_anomaly(self):
        from core.autonomous_observer import AutonomousObserver, ObservationType, ObservationSalience
        observer = AutonomousObserver()
        obs = observer.observe_anomaly("cpu_spike", "CPU at 99%", severity=0.9)
        assert obs.obs_type == ObservationType.ANOMALY
        assert obs.salience == ObservationSalience.CRITICAL
        assert obs.notifiable is True

    def test_observer_context_includes_emotional(self):
        from core.autonomous_observer import AutonomousObserver
        observer = AutonomousObserver()
        observer.observe_emotional_tone("happy", intensity=0.6)
        ctx = observer.get_observation_context()
        assert "emotional" in ctx.lower() or "happy" in ctx.lower()

    def test_integration_presence_anticipation_observer(self):
        from core.active_presence import ActivePresenceEngine
        from core.goal_anticipation import GoalAnticipationEngine
        from core.autonomous_observer import AutonomousObserver, ObservationSalience

        presence = ActivePresenceEngine()
        anticipation = GoalAnticipationEngine()
        observer = AutonomousObserver()

        presence.record_user_activity("typing")
        anticipation.record_action("code_edit", "fixing security bug")
        observer.observe_emotional_tone("focused", intensity=0.7, trigger="deep coding")
        observer.observe_action_outcome("test", success=True, details="4 tests passed")

        p_state = presence.get_state()
        a_state = anticipation.get_state()
        o_state = observer.get_state()

        assert "presence" in p_state
        assert "active_anticipations" in a_state
        assert "emotional_tone" in o_state

    def test_vector_memory_save_batching(self):
        import tempfile
        import pathlib
        from core.vector_memory import VectorMemory

        def simple_embed(text):
            import hashlib
            h = hashlib.sha256(text.encode()).digest()
            return [b / 255.0 for b in h] * 12

        with tempfile.TemporaryDirectory() as td:
            vm = VectorMemory(storage_path=pathlib.Path(td), embedding_fn=simple_embed)
            start = time.time()
            for i in range(50):
                vm.store(f"test batch item {i}")
            elapsed = time.time() - start
            vm.flush()
            vm.clear()
            assert elapsed < 30


class TestButlerStressTest:
    def test_presence_rapid_activities(self):
        from core.active_presence import ActivePresenceEngine
        engine = ActivePresenceEngine()
        for i in range(100):
            engine.record_user_activity("typing" if i % 2 == 0 else "message")
        state = engine.get_state()
        assert state["consecutive_messages"] == 100

    def test_anticipation_many_actions(self):
        from core.goal_anticipation import GoalAnticipationEngine
        engine = GoalAnticipationEngine()
        for i in range(50):
            engine.record_action("code_edit")
            engine.record_action("run_tests")
        state = engine.get_state()
        assert state["total_anticipated"] > 0

    def test_observer_many_observations(self):
        from core.autonomous_observer import AutonomousObserver
        observer = AutonomousObserver()
        for i in range(50):
            observer.observe_system(f"observation {i}")
        assert observer._observation_count >= 50

    def test_presence_tick_loop(self):
        from core.active_presence import ActivePresenceEngine
        engine = ActivePresenceEngine()
        for _ in range(10):
            engine.tick()
        state = engine.get_state()
        assert "presence" in state

    def test_anticipation_tick_loop(self):
        from core.goal_anticipation import GoalAnticipationEngine
        engine = GoalAnticipationEngine()
        for _ in range(5):
            result = engine.tick()
            assert isinstance(result, dict)

    def test_observer_tick_loop(self):
        from core.autonomous_observer import AutonomousObserver
        observer = AutonomousObserver()
        for _ in range(10):
            result = observer.tick()
            assert isinstance(result, dict)

    def test_thread_safety_presence(self):
        from core.active_presence import ActivePresenceEngine
        import threading
        engine = ActivePresenceEngine()
        errors = []

        def worker():
            try:
                for _ in range(50):
                    engine.record_user_activity("typing")
                    engine.tick()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_thread_safety_anticipation(self):
        from core.goal_anticipation import GoalAnticipationEngine
        import threading
        engine = GoalAnticipationEngine()
        errors = []

        def worker():
            try:
                for _ in range(50):
                    engine.record_action("code_edit")
                    engine.tick()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_thread_safety_observer(self):
        from core.autonomous_observer import AutonomousObserver
        import threading
        observer = AutonomousObserver()
        errors = []

        def worker():
            try:
                for _ in range(50):
                    observer.observe_system("test")
                    observer.tick()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0


class TestEndToEndButlerPipeline:
    """End-to-end test: sentiment→personality→voice→orchestrator context."""

    def test_frustration_pipeline(self):
        from core.sentiment_analyzer import SentimentAnalyzer
        from core.butler_personality import ButlerPersonality, PersonaMode
        import tempfile

        sa = SentimentAnalyzer()
        bp = ButlerPersonality(storage_path=tempfile.mkdtemp())

        result = sa.analyze("This is incredibly frustrating! Nothing works!")
        assert result.sentiment.value in ("frustrated", "urgency")
        assert result.escalation_risk > 0

        bp.observe_interaction(
            "This is incredibly frustrating!",
            user_sentiment=result.sentiment.value,
            user_energy="high",
            frustration=True,
        )
        state = bp.get_state()
        assert state["mode"] in (PersonaMode.BUTLER.value, PersonaMode.COMPANION.value)

    def test_curiosity_pipeline(self):
        from core.sentiment_analyzer import SentimentAnalyzer
        from core.butler_personality import ButlerPersonality, PersonaMode
        import tempfile

        sa = SentimentAnalyzer()
        bp = ButlerPersonality(storage_path=tempfile.mkdtemp())

        result = sa.analyze("How does this algorithm work? I'd love to understand it better.")
        assert result.sentiment.value == "curious"

        bp._adapt_persona(result.sentiment.value, "moderate", False, False)
        assert bp._state.mode == PersonaMode.MENTOR

    def test_playful_pipeline(self):
        from core.sentiment_analyzer import SentimentAnalyzer
        from core.butler_personality import ButlerPersonality, Formality
        import tempfile

        sa = SentimentAnalyzer()
        bp = ButlerPersonality(storage_path=tempfile.mkdtemp())

        result = sa.analyze("lol that was so funny!")
        assert result.sentiment.value in ("playful", "excited")

        bp._adapt_persona(result.sentiment.value, "high", False, False)
        assert bp._state.formality == Formality.CASUAL

    def test_orchestrator_interruption_pipeline(self):
        from core.butler_orchestrator import ButlerOrchestrator, OrchestratedAction, SignalPriority

        orch = ButlerOrchestrator()
        assert orch.get_pending_interruption() is None

        critical = OrchestratedAction(
            action="error_storm",
            reason="Error cascade detected",
            priority=SignalPriority.CRITICAL,
            source_modules=["proactive"],
            message="Switching to error recovery mode",
        )
        orch._actions.append(critical)
        msg = orch.get_pending_interruption()
        assert msg is not None
        assert "CRITICAL" in msg
        assert critical.delivered is True

        assert orch.get_pending_interruption() is None

    def test_sentiment_trajectory(self):
        from core.sentiment_analyzer import SentimentAnalyzer

        sa = SentimentAnalyzer()
        sa.analyze("Hello")
        sa.analyze("Wait, what?")
        sa.analyze("I'm really frustrated with this!")
        trajectory = sa.get_trajectory()
        assert len(trajectory.sentiments) >= 3

    def test_personality_persistence_round_trip(self):
        import tempfile
        from core.butler_personality import ButlerPersonality

        tmp = tempfile.mkdtemp()
        bp1 = ButlerPersonality(storage_path=tmp)
        bp1.observe_interaction("Hey there!", user_sentiment="playful", user_energy="high")
        bp1.flush()

        bp2 = ButlerPersonality(storage_path=tmp)
        state2 = bp2.get_state()
        assert state2["mode"] == bp1.get_state()["mode"]

    def test_stress_concurrent_sentiment_personality(self):
        import threading
        import tempfile
        from core.sentiment_analyzer import SentimentAnalyzer
        from core.butler_personality import ButlerPersonality

        errors = []
        sa = SentimentAnalyzer()
        bp = ButlerPersonality(storage_path=tempfile.mkdtemp())

        def worker(i):
            try:
                result = sa.analyze(f"Test message {i} with frustrated tone")
                bp.observe_interaction(
                    f"Test message {i}",
                    user_sentiment=result.sentiment.value,
                    user_energy="moderate",
                )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        assert len(sa._recent_sentiments) > 0

    def test_stress_concurrent_orchestrator_interruption(self):
        import threading
        from core.butler_orchestrator import ButlerOrchestrator, OrchestratedAction, SignalPriority

        orch = ButlerOrchestrator()
        errors = []

        def writer(i):
            try:
                action = OrchestratedAction(
                    action=f"test_action_{i}",
                    reason=f"Reason {i}",
                    priority=SignalPriority.CRITICAL if i % 3 == 0 else SignalPriority.SUGGEST,
                    source_modules=["test"],
                    message=f"Action {i}",
                )
                orch._actions.append(action)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(20):
                    orch.get_pending_interruption()
                    orch.get_coordination_context()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(20)]
        threads.append(threading.Thread(target=reader))
        threads.append(threading.Thread(target=reader))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
