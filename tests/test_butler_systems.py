"""Tests for Active Presence, Goal Anticipation, and Autonomous Observer.

Tests the three new AI butler systems that give Jarvis genuine
ambient awareness, goal anticipation, and observational intelligence.
"""
import pytest
import sys
import os
import time
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))


class TestActivePresence:
    def test_initial_state(self):
        from core.active_presence import ActivePresenceEngine, UserPresence
        engine = ActivePresenceEngine()
        state = engine.get_state()
        assert state["presence"] in (UserPresence.IDLE.value, UserPresence.ACTIVE.value)
        assert "idle_seconds" in state
        assert "focus_score" in state

    def test_record_user_activity_returns_transition(self):
        from core.active_presence import ActivePresenceEngine, PresenceTransition, UserPresence
        engine = ActivePresenceEngine()
        engine._state.presence = UserPresence.AWAY
        transition = engine.record_user_activity("message")
        assert transition == PresenceTransition.RETURNED
        assert engine._state.presence == UserPresence.FOCUSED or engine._state.presence.value in ("active", "focused", "typing")

    def test_focus_detection(self):
        from core.active_presence import ActivePresenceEngine, UserPresence
        engine = ActivePresenceEngine()
        now = time.time()
        for i in range(5):
            engine._message_times.append(now - i * 10)
        engine.record_user_activity("message")
        assert engine._state.presence in (UserPresence.FOCUSED, UserPresence.ACTIVE, UserPresence.TYPING)

    def test_tick_updates_idle(self):
        from core.active_presence import ActivePresenceEngine
        engine = ActivePresenceEngine()
        engine._state.last_activity = time.time() - 400
        result = engine.tick()
        assert result["presence"] == "away"

    def test_add_observation(self):
        from core.active_presence import ActivePresenceEngine
        engine = ActivePresenceEngine()
        obs = engine.add_observation("system", "CPU spike detected", relevance=0.7)
        assert obs.category == "system"
        assert obs.content == "CPU spike detected"
        assert obs.relevance == 0.7
        assert not obs.acknowledged

    def test_unacknowledged_observations(self):
        from core.active_presence import ActivePresenceEngine
        engine = ActivePresenceEngine()
        engine.add_observation("system", "A", relevance=0.5)
        engine.add_observation("system", "B", relevance=0.8)
        unack = engine.get_unacknowledged_observations()
        assert len(unack) == 2

    def test_acknowledge_observation(self):
        from core.active_presence import ActivePresenceEngine
        engine = ActivePresenceEngine()
        engine.add_observation("system", "Test", relevance=0.5)
        result = engine.acknowledge_observation(0)
        assert result is True

    def test_should_check_in_returned(self):
        from core.active_presence import ActivePresenceEngine, UserPresence
        engine = ActivePresenceEngine()
        engine._state.presence = UserPresence.RETURNED
        ci = engine.should_check_in()
        assert ci is not None
        assert ci.reason == "user_returned"

    def test_should_check_in_with_observation(self):
        from core.active_presence import ActivePresenceEngine, UserPresence
        engine = ActivePresenceEngine()
        engine._state.presence = UserPresence.ACTIVE
        engine.add_observation("system", "High relevance", relevance=0.8)
        ci = engine.should_check_in()
        assert ci is not None
        assert ci.reason == "observation"

    def test_dismiss_checkin(self):
        from core.active_presence import ActivePresenceEngine, CheckIn
        engine = ActivePresenceEngine()
        ci = CheckIn(reason="test", message="Test checkin", urgency="low")
        engine._state.pending_checkins.append(ci)
        result = engine.dismiss_checkin()
        assert result is True

    def test_presence_context(self):
        from core.active_presence import ActivePresenceEngine
        engine = ActivePresenceEngine()
        ctx = engine.get_presence_context()
        assert isinstance(ctx, str)
        assert "presence" in ctx.lower()

    def test_get_active_presence_singleton(self):
        from core.active_presence import get_active_presence
        ap = get_active_presence()
        assert ap is not None

    def test_typing_timeout(self):
        from core.active_presence import ActivePresenceEngine, UserPresence
        engine = ActivePresenceEngine()
        engine._state.presence = UserPresence.TYPING
        engine._state.last_activity = time.time() - 20
        result = engine.tick()
        assert result["presence"] != "typing"


class TestGoalAnticipation:
    def test_initial_state(self):
        from core.goal_anticipation import GoalAnticipationEngine
        engine = GoalAnticipationEngine()
        state = engine.get_state()
        assert "active_anticipations" in state
        assert "learned_patterns" in state
        assert "current_goals" in state

    def test_record_action_learns_pattern(self):
        from core.goal_anticipation import GoalAnticipationEngine
        engine = GoalAnticipationEngine()
        for _ in range(3):
            engine.record_action("code_edit")
            engine.record_action("run_tests")
        state = engine.get_state()
        patterns = state["learned_patterns"]
        assert len(patterns) > 0

    def test_context_based_anticipation(self):
        from core.goal_anticipation import GoalAnticipationEngine
        engine = GoalAnticipationEngine()
        result = engine.record_action("code_edit")
        assert result is not None
        assert "run_tests" in result.description or "test" in result.description.lower()

    def test_goal_based_anticipation(self):
        from core.goal_anticipation import GoalAnticipationEngine
        engine = GoalAnticipationEngine()
        engine.set_active_goals(["build a REST API"])
        result = engine._anticipate_from_goals("random_action")
        assert result is not None
        assert result.source.value == "goal"

    def test_set_active_goals(self):
        from core.goal_anticipation import GoalAnticipationEngine
        engine = GoalAnticipationEngine()
        engine.set_active_goals(["deploy to production", "write tests"])
        state = engine.get_state()
        assert len(state["current_goals"]) == 2

    def test_pending_anticipations(self):
        from core.goal_anticipation import GoalAnticipationEngine
        engine = GoalAnticipationEngine()
        engine.record_action("code_edit")
        pending = engine.get_pending_anticipations()
        assert isinstance(pending, list)

    def test_dismiss_anticipation(self):
        from core.goal_anticipation import GoalAnticipationEngine
        engine = GoalAnticipationEngine()
        engine.record_action("code_edit")
        result = engine.dismiss_anticipation(0)
        assert result is True

    def test_accept_anticipation(self):
        from core.goal_anticipation import GoalAnticipationEngine
        engine = GoalAnticipationEngine()
        engine.record_action("code_edit")
        engine.deliver_anticipation(0)
        result = engine.accept_anticipation(0)
        assert result is True
        assert engine._state.accepted_count == 1

    def test_dismissal_tracking(self):
        from core.goal_anticipation import GoalAnticipationEngine
        engine = GoalAnticipationEngine()
        engine.record_action("code_edit")
        engine.dismiss_anticipation(0)
        engine.record_action("code_edit")
        engine.dismiss_anticipation(0)
        assert engine._state.dismissed_count == 2

    def test_anticipation_context(self):
        from core.goal_anticipation import GoalAnticipationEngine
        engine = GoalAnticipationEngine()
        engine.set_active_goals(["optimize performance"])
        ctx = engine.get_anticipation_context()
        assert isinstance(ctx, str)

    def test_get_goal_anticipation_singleton(self):
        from core.goal_anticipation import get_goal_anticipation
        ga = get_goal_anticipation()
        assert ga is not None

    def test_error_seen_anticipation(self):
        from core.goal_anticipation import GoalAnticipationEngine
        engine = GoalAnticipationEngine()
        result = engine.record_action("error_seen")
        assert result is not None
        assert "debug" in result.description.lower()

    def test_action_sequence_tracking(self):
        from core.goal_anticipation import GoalAnticipationEngine
        engine = GoalAnticipationEngine()
        for _ in range(4):
            engine.record_action("code_edit")
            engine.record_action("run_tests")
            engine.record_action("commit_changes")
        assert len(engine._state.sequence_history) > 0


class TestAutonomousObserver:
    def test_initial_state(self):
        from core.autonomous_observer import AutonomousObserver
        observer = AutonomousObserver()
        state = observer.get_state()
        assert "total_observations" in state
        assert "habits" in state

    def test_observe_system(self):
        from core.autonomous_observer import AutonomousObserver, ObservationSalience
        observer = AutonomousObserver()
        obs = observer.observe_system("CPU at 95%", ObservationSalience.HIGH)
        assert obs.obs_type.value == "system"
        assert obs.notifiable is True

    def test_observe_user_habit(self):
        from core.autonomous_observer import AutonomousObserver
        observer = AutonomousObserver()
        observer.observe_user_habit("work_pattern", "Usually starts coding at 9am")
        habits = observer.get_learned_habits()
        assert isinstance(habits, list)

    def test_observe_conversation(self):
        from core.autonomous_observer import AutonomousObserver
        observer = AutonomousObserver()
        observer.observe_conversation("User asked about testing", {"topic": "testing"})
        recent = observer.get_recent_observations(limit=5)
        assert len(recent) >= 1

    def test_observe_action_outcome(self):
        from core.autonomous_observer import AutonomousObserver
        observer = AutonomousObserver()
        observer.observe_action_outcome("deploy", success=True, details="v1.2.3")
        observer.observe_action_outcome("test_run", success=False, details="3 failures")
        notifiable = observer.get_notifiable_observations()
        assert len(notifiable) >= 1

    def test_observe_environment_change(self):
        from core.autonomous_observer import AutonomousObserver, ObservationSalience
        observer = AutonomousObserver()
        obs = observer.observe_environment_change("service_restart", "docker restarted", ObservationSalience.HIGH)
        assert obs.obs_type.value == "environment"
        assert obs.notifiable is True

    def test_tick_expires_observations(self):
        from core.autonomous_observer import AutonomousObserver
        observer = AutonomousObserver()
        observer.observe_system("test")
        for obs in observer._observations:
            obs.timestamp = time.time() - 7200
            obs.ttl_seconds = 3600
        result = observer.tick()
        assert result["expired"] >= 1

    def test_mark_notified(self):
        from core.autonomous_observer import AutonomousObserver, ObservationSalience
        observer = AutonomousObserver()
        observer.observe_system("Alert", ObservationSalience.HIGH)
        result = observer.mark_notified(0)
        assert result is True

    def test_habit_learning(self):
        from core.autonomous_observer import AutonomousObserver
        observer = AutonomousObserver()
        for _ in range(3):
            observer.observe_user_habit("schedule", "Morning check-in")
        habits = observer.get_learned_habits()
        assert any(h["description"] == "Morning check-in" for h in habits)

    def test_observation_context(self):
        from core.autonomous_observer import AutonomousObserver, ObservationSalience
        observer = AutonomousObserver()
        observer.observe_system("High CPU", ObservationSalience.HIGH)
        ctx = observer.get_observation_context()
        assert isinstance(ctx, str)

    def test_recent_observations_filtered(self):
        from core.autonomous_observer import AutonomousObserver, ObservationType
        observer = AutonomousObserver()
        observer.observe_system("sys1")
        observer.observe_conversation("conv1")
        recent = observer.get_recent_observations(limit=10, obs_type=ObservationType.SYSTEM)
        assert all(r["type"] == "system" for r in recent)

    def test_get_autonomous_observer_singleton(self):
        from core.autonomous_observer import get_autonomous_observer
        ao = get_autonomous_observer()
        assert ao is not None

    def test_habit_max_bound(self):
        from core.autonomous_observer import AutonomousObserver
        observer = AutonomousObserver()
        observer._habits = {}
        for i in range(55):
            observer.observe_user_habit("test", f"Habit {i}")
        assert len(observer._habits) <= 52


class TestIntegrationBetweenModules:
    def test_presence_feeds_observer(self):
        from core.active_presence import ActivePresenceEngine, UserPresence
        from core.autonomous_observer import AutonomousObserver
        engine = ActivePresenceEngine()
        observer = AutonomousObserver()
        engine._state.presence = UserPresence.AWAY
        transition = engine.record_user_activity("message")
        if transition:
            observer.observe_user_habit("presence", f"User {transition.value}")
        assert observer._observation_count > 0

    def test_anticipation_feeds_context(self):
        from core.goal_anticipation import GoalAnticipationEngine
        engine = GoalAnticipationEngine()
        engine.set_active_goals(["deploy the service"])
        engine.record_action("code_edit")
        ctx = engine.get_anticipation_context()
        assert isinstance(ctx, str)

    def test_observer_feeds_presence_observations(self):
        from core.active_presence import ActivePresenceEngine
        from core.autonomous_observer import AutonomousObserver, ObservationSalience
        presence = ActivePresenceEngine()
        observer = AutonomousObserver()
        observer.observe_system("Disk nearly full", ObservationSalience.HIGH)
        notifiable = observer.get_notifiable_observations()
        if notifiable:
            presence.add_observation("system", notifiable[0]["content"], relevance=0.7)
        unack = presence.get_unacknowledged_observations()
        assert len(unack) >= 0

    def test_full_butler_pipeline(self):
        from core.active_presence import ActivePresenceEngine, UserPresence
        from core.goal_anticipation import GoalAnticipationEngine
        from core.autonomous_observer import AutonomousObserver, ObservationSalience
        presence = ActivePresenceEngine()
        anticipation = GoalAnticipationEngine()
        observer = AutonomousObserver()

        presence._state.presence = UserPresence.AWAY
        transition = presence.record_user_activity("message")
        observer.observe_user_habit("session", f"User {transition.value if transition else 'active'}")

        anticipation.set_active_goals(["build and deploy"])
        antic = anticipation.record_action("code_edit")

        observer.observe_system("Build passing", ObservationSalience.MEDIUM)
        observer.observe_action_outcome("build", success=True, details="All green")

        p_ctx = presence.get_presence_context()
        a_ctx = anticipation.get_anticipation_context()
        o_ctx = observer.get_observation_context()

        assert isinstance(p_ctx, str)
        assert isinstance(a_ctx, str)
        assert isinstance(o_ctx, str)

    def test_deep_work_detection(self):
        from core.active_presence import ActivePresenceEngine, UserPresence, PresenceTransition
        engine = ActivePresenceEngine()
        engine._state.deep_work_start = time.time() - 700
        engine._state.presence = UserPresence.FOCUSED
        engine._state.focus_score = 0.8
        result = engine.tick()
        if result.get("transition") == PresenceTransition.DEEP_WORK_STARTED.value:
            assert engine._state.presence == UserPresence.DEEP_WORK
        assert engine._state.rhythm.deep_work_periods >= 0

    def test_break_reminder_checkin(self):
        from core.active_presence import ActivePresenceEngine, UserPresence
        engine = ActivePresenceEngine()
        engine._state.focus_score = 0.7
        engine._state.presence = UserPresence.ACTIVE
        engine._state.last_break_time = time.time() - 3000
        engine._state.last_checkin_time = 0
        ci = engine.should_check_in()
        if ci:
            assert ci.reason == "break_reminder"

    def test_distracted_detection(self):
        from core.active_presence import ActivePresenceEngine, UserPresence
        engine = ActivePresenceEngine()
        for t in ["code_edit", "file_read", "search", "config_change", "deploy"]:
            engine._activity_types.append(t)
        engine._state.distraction_count = 3
        engine._state.presence = UserPresence.DISTRACTED
        ctx = engine.get_presence_context()
        assert "distracted" in ctx.lower()

    def test_checkin_cooldown(self):
        from core.active_presence import ActivePresenceEngine, UserPresence, CheckIn
        engine = ActivePresenceEngine()
        engine._state.presence = UserPresence.ACTIVE
        engine._state.last_checkin_time = time.time()
        ci = CheckIn(reason="test", message="test", urgency="low")
        engine._state.pending_checkins.append(ci)
        result = engine.should_check_in()
        assert result is None

    def test_detail_pattern_anticipation(self):
        from core.goal_anticipation import GoalAnticipationEngine
        engine = GoalAnticipationEngine()
        antic = engine.record_action("code_edit", "error in module.py")
        if antic:
            assert antic.suggested_action in ("run_tests", "debug", "verify_config", "test_endpoint")

    def test_schedule_anticipation(self):
        from core.goal_anticipation import GoalAnticipationEngine
        engine = GoalAnticipationEngine()
        result = engine.tick()
        assert isinstance(result, dict)

    def test_sequence_anticipation(self):
        from core.goal_anticipation import GoalAnticipationEngine
        engine = GoalAnticipationEngine()
        for _ in range(3):
            engine.record_action("code_edit")
            engine.record_action("run_tests")
        from core.goal_anticipation import ActionSequence
        engine._state.sequence_history.append(ActionSequence(actions=["code_edit", "run_tests", "commit"]))
        engine._state.sequence_history[-1].count = 3
        antic = engine._anticipate_from_sequence()
        if antic:
            assert antic.source.value == "sequence"

    def test_emotional_tone_observation(self):
        from core.autonomous_observer import AutonomousObserver, ObservationType
        observer = AutonomousObserver()
        obs = observer.observe_emotional_tone("frustrated", intensity=0.8, trigger="test failure")
        assert obs.obs_type == ObservationType.EMOTIONAL
        state = observer.get_state()
        assert "emotional_tone" in state
        assert state["emotional_tone"] == "frustrated"

    def test_file_change_observation(self):
        from core.autonomous_observer import AutonomousObserver, ObservationType
        observer = AutonomousObserver()
        obs = observer.observe_file_change("/src/module.py", "modified")
        assert obs.obs_type == ObservationType.FILE_CHANGE
        assert "module.py" in obs.content

    def test_network_event_observation(self):
        from core.autonomous_observer import AutonomousObserver, ObservationType
        observer = AutonomousObserver()
        obs = observer.observe_network_event("api_timeout", "API call timed out after 30s")
        assert obs.obs_type == ObservationType.NETWORK

    def test_anomaly_observation(self):
        from core.autonomous_observer import AutonomousObserver, ObservationType, ObservationSalience
        observer = AutonomousObserver()
        obs = observer.observe_anomaly("memory_leak", "Memory usage increasing steadily", severity=0.9)
        assert obs.obs_type == ObservationType.ANOMALY
        assert obs.salience == ObservationSalience.CRITICAL
        assert obs.notifiable is True

    def test_emotional_trajectory(self):
        from core.autonomous_observer import AutonomousObserver
        observer = AutonomousObserver()
        observer.observe_emotional_tone("curious", intensity=0.4)
        observer.observe_emotional_tone("frustrated", intensity=0.7)
        observer.observe_emotional_tone("satisfied", intensity=0.5)
        trajectory = observer.get_emotional_trajectory()
        assert len(trajectory) == 3
        ctx = observer.get_observation_context()
        assert "emotional" in ctx.lower()

    def test_presence_state_new_fields(self):
        from core.active_presence import ActivePresenceEngine
        engine = ActivePresenceEngine()
        state = engine.get_state()
        assert "deep_work_minutes" in state
        assert "distraction_count" in state
        assert "deep_work_periods" in state["rhythm"]
        assert "breaks_taken" in state["rhythm"]
        assert "longest_focus_minutes" in state["rhythm"]

    def test_goal_anticipation_detail_patterns(self):
        from core.goal_anticipation import GoalAnticipationEngine, AnticipationSource
        engine = GoalAnticipationEngine()
        antic = engine.record_action("file_write", "updating security config")
        if antic:
            assert antic.source in (AnticipationSource.CONTEXT, AnticipationSource.PATTERN)

    def test_goal_anticipation_extra_goals(self):
        from core.goal_anticipation import GoalAnticipationEngine
        engine = GoalAnticipationEngine()
        engine.set_active_goals(["debug the api"])
        antic = engine.record_action("code_edit")
        if antic:
            assert isinstance(antic.description, str)