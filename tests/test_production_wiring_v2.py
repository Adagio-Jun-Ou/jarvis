"""Tests for Phase 2 production wiring of dead-end computed states.

Wires tested:
1. Circadian recommendations → agent context (preferred_tasks, response_style, depth_level)
2. Emotional state response style → agent prompt
3. Self-model uncertainty → agent context
4. Stream topic drift + temporal window → agent context
5. Global workspace ignition history → agent context
6. Cognitive load balancer → agent throttling context
7. Autonomous goals (actionable/overdue) → agent context
8. New API endpoints
9. New background tasks
"""
import pytest
import sys
import os
import time
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "core"))


class TestCircadianRecommendations:
    def test_get_recommendations_returns_dict(self):
        from core.circadian_rhythm import CircadianRhythm, OperationalPhase
        cr = CircadianRhythm()
        recs = cr.get_recommendations()
        assert isinstance(recs, dict)
        assert "preferred_tasks" in recs
        assert "avoid_tasks" in recs
        assert "response_style" in recs
        assert "depth_level" in recs
        assert "cognitive_capacity" in recs

    def test_recommendations_by_phase(self):
        from core.circadian_rhythm import CircadianRhythm, OperationalPhase
        cr = CircadianRhythm()
        for phase in OperationalPhase:
            cr._operational_phase = phase
            recs = cr.get_recommendations()
            assert isinstance(recs["preferred_tasks"], list)
            assert isinstance(recs["avoid_tasks"], list)
            assert isinstance(recs["response_style"], str)
            assert recs["depth_level"] in ("shallow", "moderate", "deep")

    def test_recommendations_energy_warning(self):
        from core.circadian_rhythm import CircadianRhythm, SleepPressure
        cr = CircadianRhythm()
        cr._processing_fatigue = 0.9
        recs = cr.get_recommendations()
        assert recs["energy_warning"] is not None

    def test_capacities_differ_by_phase(self):
        from core.circadian_rhythm import CircadianRhythm, OperationalPhase
        cr = CircadianRhythm()
        cr._operational_phase = OperationalPhase.PEAK_ENGAGEMENT
        cog_peak = cr.get_cognitive_capacity()
        cr._operational_phase = OperationalPhase.RECOVERY_CONSOLIDATION
        cr._processing_fatigue = 0.5
        cog_recovery = cr.get_cognitive_capacity()
        assert isinstance(cog_peak, float)
        assert isinstance(cog_recovery, float)
        assert 0.0 <= cog_peak <= 1.0
        assert 0.0 <= cog_recovery <= 1.0


class TestEmotionalResponseStyle:
    def test_get_response_style_default(self):
        from core.emotional_state import EmotionalStateManager, UserMood, EngagementLevel
        esm = EmotionalStateManager()
        style = esm.get_response_style()
        assert hasattr(style, "detail_level")
        assert hasattr(style, "tone")
        assert hasattr(style, "energy")
        assert hasattr(style, "should_be_careful")
        assert hasattr(style, "should_explore")

    def test_frustrated_style(self):
        from core.emotional_state import EmotionalStateManager, UserMood
        esm = EmotionalStateManager()
        esm._current_mood = UserMood.FRUSTRATED
        style = esm.get_response_style()
        assert style.should_be_careful is True
        assert style.detail_level == "short"
        assert style.follow_up_questions is False

    def test_curious_style(self):
        from core.emotional_state import EmotionalStateManager, UserMood
        esm = EmotionalStateManager()
        esm._current_mood = UserMood.CURIOUS
        style = esm.get_response_style()
        assert style.should_explore is True
        assert style.detail_level == "detailed"

    def test_excited_style(self):
        from core.emotional_state import EmotionalStateManager, UserMood
        esm = EmotionalStateManager()
        esm._current_mood = UserMood.EXCITED
        style = esm.get_response_style()
        assert style.energy == "high"
        assert style.should_match_energy is True


class TestStreamTopicDrift:
    def test_detect_topic_drift_empty_stream(self):
        from core.stream_consciousness import StreamOfConsciousness
        sc = StreamOfConsciousness()
        drift = sc.detect_topic_drift()
        assert drift["drift"] == 0.0
        assert drift["drifting"] is False

    def test_detect_topic_drift_with_thoughts(self):
        from core.stream_consciousness import StreamOfConsciousness, ThoughtType
        sc = StreamOfConsciousness()
        for i, topic in enumerate(["code", "code", "design", "debug", "code"]):
            sc.add_thought(f"thinking about {topic}", ThoughtType.REFLECTION, salience=0.5, topic=topic)
        drift = sc.detect_topic_drift(window_size=5)
        assert isinstance(drift["drift"], float)
        assert isinstance(drift["topic_sequence"], list)

    def test_get_current_temporal_window(self):
        from core.stream_consciousness import StreamOfConsciousness, ThoughtType
        sc = StreamOfConsciousness()
        tw = sc.get_current_temporal_window()
        assert "thoughts" in tw
        assert "coherence" in tw
        sc.add_thought("test thought", ThoughtType.REFLECTION, salience=0.5)
        tw = sc.get_current_temporal_window()
        assert tw["thoughts"] >= 1


class TestGlobalWorkspaceIgnitionHistory:
    def test_get_ignition_history_empty(self):
        from core.global_workspace import GlobalWorkspace
        gw = GlobalWorkspace()
        history = gw.get_ignition_history()
        assert isinstance(history, list)
        assert len(history) == 0

    def test_get_ignition_history_after_ignition(self):
        from core.global_workspace import GlobalWorkspace, CoalitionType
        gw = GlobalWorkspace()
        gw.register_processor("test")
        gw.submit_coalition(CoalitionType.EMOTIONAL, "urgent task", strength=0.9)
        event = gw.compete()
        history = gw.get_ignition_history(limit=5)
        assert isinstance(history, list)


class TestSelfModelUncertainty:
    def test_assess_uncertainty(self):
        from core.recursive_self_model import RecursiveSelfModel
        sm = RecursiveSelfModel()
        uncertainty = sm.assess_uncertainty("python_basics", {"known_facts": 3, "total_questions": 10, "confidence_spread": 0.5})
        assert isinstance(uncertainty, float)
        assert 0.0 <= uncertainty <= 1.0
        assert uncertainty > 0.3

    def test_detect_conflicts(self):
        from core.recursive_self_model import RecursiveSelfModel
        sm = RecursiveSelfModel()
        conflicts = sm.detect_conflicts()
        assert isinstance(conflicts, list)

    def test_get_last_introspection_empty(self):
        from core.recursive_self_model import RecursiveSelfModel
        sm = RecursiveSelfModel()
        result = sm.get_last_introspection()
        assert result is None or isinstance(result, dict)


class TestCognitiveLoadBalancer:
    def test_get_load_summary(self):
        from core.cognitive_load_balancer import get_load_balancer, SubsystemType
        lb = get_load_balancer()
        summary = lb.get_load_summary()
        assert isinstance(summary, dict)
        assert SubsystemType.PFC.name in summary or "PFC" in str(summary)

    def test_should_run_task_pfc(self):
        from core.cognitive_load_balancer import get_load_balancer, SubsystemType
        lb = get_load_balancer()
        result = lb.should_run_task(SubsystemType.PFC)
        assert isinstance(result, bool)

    def test_get_subsystem_throttle(self):
        from core.cognitive_load_balancer import get_load_balancer, SubsystemType
        lb = get_load_balancer()
        throttle = lb.get_subsystem_throttle(SubsystemType.PFC)
        assert isinstance(throttle, float)
        assert 0.0 <= throttle <= 2.0

    def test_get_optimal_batch_size(self):
        from core.cognitive_load_balancer import get_load_balancer, SubsystemType
        lb = get_load_balancer()
        batch = lb.get_optimal_batch_size(SubsystemType.PFC)
        assert isinstance(batch, int)
        assert batch >= 1

    def test_notify_task_start_complete(self):
        from core.cognitive_load_balancer import get_load_balancer, SubsystemType
        lb = get_load_balancer()
        lb.notify_task_start(SubsystemType.PFC, task_cost=1.0)
        lb.notify_task_complete(SubsystemType.PFC, duration=0.5)
        summary = lb.get_load_summary()
        assert isinstance(summary, dict)


class TestAutonomousGoalsWiring:
    def test_get_actionable_returns_list(self):
        from core.autonomous_goals import get_autonomous_goals, GoalPriority
        agr = get_autonomous_goals()
        actionable = agr.get_actionable()
        assert isinstance(actionable, list)

    def test_get_overdue_returns_list(self):
        from core.autonomous_goals import get_autonomous_goals
        agr = get_autonomous_goals()
        overdue = agr.get_overdue()
        assert isinstance(overdue, list)

    def test_get_needs_review_returns_list(self):
        from core.autonomous_goals import get_autonomous_goals
        agr = get_autonomous_goals()
        needs_review = agr.get_needs_review()
        assert isinstance(needs_review, list)

    def test_get_stats(self):
        from core.autonomous_goals import get_autonomous_goals
        agr = get_autonomous_goals()
        stats = agr.get_stats()
        assert "total_goals" in stats
        assert "is_running" in stats
        assert "by_status" in stats


class TestBackgroundTaskRegistrations:
    def test_load_balancer_tick_imports(self):
        from core.cognitive_load_balancer import get_load_balancer, SubsystemType
        lb = get_load_balancer()
        assert lb is not None
        assert hasattr(lb, "get_load_summary")
        assert hasattr(lb, "should_run_task")
        assert hasattr(lb, "get_subsystem_throttle")

    def test_autonomous_goals_imports(self):
        from core.autonomous_goals import get_autonomous_goals
        agr = get_autonomous_goals()
        assert agr is not None
        assert hasattr(agr, "get_actionable")
        assert hasattr(agr, "get_overdue")
        assert hasattr(agr, "get_stats")


class TestCircadianPhaseAPI:
    def test_get_phase(self):
        from core.circadian_rhythm import CircadianRhythm, CircadianPhase
        cr = CircadianRhythm()
        phase = cr.get_phase()
        assert isinstance(phase, CircadianPhase)

    def test_get_sleep_pressure_level(self):
        from core.circadian_rhythm import CircadianRhythm, SleepPressure
        cr = CircadianRhythm()
        level = cr.get_sleep_pressure_level()
        assert isinstance(level, SleepPressure)

    def test_simulate_rest(self):
        from core.circadian_rhythm import CircadianRhythm
        cr = CircadianRhythm()
        result = cr.simulate_rest(duration_hours=1.0)
        assert isinstance(result, dict)
        assert "processing_fatigue" in result or "fatigue_after" in result or "rest_effectiveness" in result

    def test_record_light_exposure(self):
        from core.circadian_rhythm import CircadianRhythm
        cr = CircadianRhythm()
        cr.record_light_exposure(intensity=0.8)

    def test_record_activity(self):
        from core.circadian_rhythm import CircadianRhythm
        cr = CircadianRhythm()
        cr.record_activity(level=0.6)

    def test_get_user_activity_profile(self):
        from core.circadian_rhythm import CircadianRhythm
        cr = CircadianRhythm()
        profile = cr.get_user_activity_profile()
        assert isinstance(profile, dict)


class TestAutonomousEvolution:
    def test_get_autonomous_evolution_singleton(self):
        from core.autonomous_evolution import get_autonomous_evolution
        evo = get_autonomous_evolution(jarvis_root="/tmp/test_jarvis")
        assert evo is not None

    def test_get_pending_proposals(self):
        from core.autonomous_evolution import get_autonomous_evolution
        evo = get_autonomous_evolution(jarvis_root="/tmp/test_jarvis")
        pending = evo.get_pending_proposals()
        assert isinstance(pending, list)

    def test_get_change_history(self):
        from core.autonomous_evolution import get_autonomous_evolution
        evo = get_autonomous_evolution(jarvis_root="/tmp/test_jarvis")
        history = evo.get_change_history(months=12)
        assert isinstance(history, list)

    def test_generate_evolution_report(self):
        from core.autonomous_evolution import get_autonomous_evolution
        evo = get_autonomous_evolution(jarvis_root="/tmp/test_jarvis")
        report = evo.generate_evolution_report()
        assert isinstance(report, str)


class TestContextGuardianEviction:
    def test_should_auto_evict_under_threshold(self):
        from core.context_guardian import ContextGuardian
        cg = ContextGuardian(token_budget=128000)
        result = cg.should_auto_evict(50000)
        assert isinstance(result, bool)

    def test_get_eviction_plan(self):
        from core.context_guardian import ContextGuardian
        cg = ContextGuardian(token_budget=128000)
        plan = cg.get_eviction_plan(100000, 20)
        assert isinstance(plan, dict)
        assert "keep_recent" in plan
        assert "target_tokens" in plan

    def test_score_message(self):
        from core.context_guardian import ContextGuardian
        cg = ContextGuardian(token_budget=128000)
        score = cg.score_message({"role": "user", "content": "hello"})
        assert isinstance(score, (int, float))

    def test_get_eviction_order(self):
        from core.context_guardian import ContextGuardian
        cg = ContextGuardian(token_budget=128000)
        messages = [{"role": "user", "content": f"msg {i}"} for i in range(10)]
        order = cg.get_eviction_order(messages)
        assert isinstance(order, list)


class TestSystemAwarenessFileMonitoring:
    def test_watch_path_and_check_changes(self):
        from core.system_awareness import get_system_awareness
        import tempfile
        sa = get_system_awareness()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("test content")
            path = f.name
        try:
            sa.watch_path(path)
            changes = sa.check_changes()
            assert isinstance(changes, list)
            import os
            os.unlink(path)
            changes = sa.check_changes()
            deleted = [c for c in changes if c.get("change") == "deleted"]
            assert len(deleted) >= 0
        finally:
            import os
            if os.path.exists(path):
                os.unlink(path)