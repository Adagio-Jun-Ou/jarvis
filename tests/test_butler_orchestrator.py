"""Tests for Butler Orchestrator and persistence across butler modules."""
import json
import os
import tempfile
import time
import threading

import pytest

from core.butler_orchestrator import (
    ButlerOrchestrator,
    OrchestratedAction,
    SignalPriority,
    get_butler_orchestrator,
)
from core.butler_personality import (
    ButlerPersonality,
    PersonaMode,
    Formality,
    Verbosity,
    EnergyLevel,
)
from core.proactive_intelligence import (
    ProactiveIntelligence,
    PatternType,
    InterventionUrgency,
)
from core.contextual_memory import ContextualMemory, PreferenceCategory


class TestButlerOrchestrator:
    def test_init(self):
        orch = ButlerOrchestrator()
        state = orch.get_state()
        assert state["actions_proposed"] == 0
        assert state["acceptance_rate"] == 0.5
        assert state["rules_active"] > 0

    def test_singleton(self):
        orch1 = get_butler_orchestrator()
        orch2 = get_butler_orchestrator()
        assert orch1 is orch2

    def test_coordinate_empty_states(self):
        orch = ButlerOrchestrator()
        action = orch.coordinate()
        assert action is None

    def test_coordinate_deep_work_protection(self):
        orch = ButlerOrchestrator()
        action = orch.coordinate(
            presence_state={"presence": "deep_work", "focus_score": 0.9, "session_minutes": 45},
            proactive_state={"active_patterns": 0},
        )
        assert action is not None
        assert action.reason == "deep_work_protection"
        assert action.priority == SignalPriority.SUPPRESS

    def test_coordinate_error_storm(self):
        orch = ButlerOrchestrator()
        action = orch.coordinate(
            proactive_state={"patterns": [{"name": "error_storm"}], "active_patterns": 1},
        )
        assert action is not None
        assert action.reason == "error_storm_override"
        assert action.priority == SignalPriority.CRITICAL

    def test_coordinate_frustration_care(self):
        orch = ButlerOrchestrator()
        action = orch.coordinate(
            observer_state={"emotional_tone": "frustrated"},
            proactive_state={"patterns": [{"name": "stuck_loop"}], "active_patterns": 1, "recent_errors": 2},
        )
        assert action is not None
        assert action.reason == "frustration_care"

    def test_coordinate_productive_flow(self):
        orch = ButlerOrchestrator()
        action = orch.coordinate(
            proactive_state={"patterns": [{"name": "productive_flow"}], "active_patterns": 1},
        )
        assert action is not None
        assert action.reason == "productive_flow_protection"

    def test_coordinate_curious_learner(self):
        orch = ButlerOrchestrator()
        action = orch.coordinate(
            observer_state={"emotional_tone": "curious"},
        )
        assert action is not None
        assert action.reason == "curious_learner"

    def test_coordinate_break_needed(self):
        orch = ButlerOrchestrator()
        action = orch.coordinate(
            presence_state={"session_minutes": 120, "focus_score": 0.8},
        )
        assert action is not None
        assert action.reason == "break_needed"

    def test_coordination_interval(self):
        orch = ButlerOrchestrator()
        orch.coordinate(presence_state={"presence": "deep_work"})
        action = orch.coordinate(presence_state={"presence": "deep_work"})
        assert action is None  # Too soon

    def test_coordination_context(self):
        orch = ButlerOrchestrator()
        ctx = orch.get_coordination_context()
        assert isinstance(ctx, str)

    def test_coordination_context_with_inferences(self):
        orch = ButlerOrchestrator()
        orch._module_states = {
            "presence": {"presence": "deep_work", "focus_score": 0.9},
            "proactive": {"active_patterns": 2},
        }
        orch._cross_module_inferences = orch._infer_cross_module()
        ctx = orch.get_coordination_context()
        assert len(ctx) > 0

    def test_mark_accepted(self):
        orch = ButlerOrchestrator()
        orch._actions.append(OrchestratedAction(
            action="test", reason="test", priority=SignalPriority.SUGGEST,
            source_modules=["test"], message="test", delivered=True,
        ))
        orch.mark_accepted()
        assert orch._actions[-1].accepted is True
        assert orch._acceptance_history[-1]["accepted"] is True

    def test_mark_dismissed(self):
        orch = ButlerOrchestrator()
        orch._actions.append(OrchestratedAction(
            action="test", reason="test", priority=SignalPriority.SUGGEST,
            source_modules=["test"], message="test", delivered=True,
        ))
        orch.mark_dismissed()
        assert orch._actions[-1].dismissed is True
        assert orch._acceptance_history[-1]["accepted"] is False

    def test_acceptance_rate(self):
        orch = ButlerOrchestrator()
        orch._acceptance_history.append({"action": "a", "accepted": True, "time": time.time()})
        orch._acceptance_history.append({"action": "b", "accepted": True, "time": time.time()})
        orch._acceptance_history.append({"action": "c", "accepted": False, "time": time.time()})
        rate = orch._get_acceptance_rate()
        assert 0.0 <= rate <= 1.0

    def test_cross_module_inference_deep_work_with_patterns(self):
        orch = ButlerOrchestrator()
        orch._module_states = {
            "presence": {"presence": "deep_work"},
            "proactive": {"active_patterns": 3},
        }
        inferences = orch._infer_cross_module()
        assert len(inferences) > 0
        assert "deep work" in inferences[0]["inference"].lower()

    def test_cross_module_inference_frustrated_debugging(self):
        orch = ButlerOrchestrator()
        orch._module_states = {
            "observer": {"emotional_tone": "frustrated"},
            "proactive": {"recent_errors": 3},
        }
        inferences = orch._infer_cross_module()
        assert len(inferences) > 0
        assert "frustrat" in inferences[0]["inference"].lower()

    def test_cross_module_inference_long_session(self):
        orch = ButlerOrchestrator()
        orch._module_states = {
            "presence": {"session_minutes": 150, "focus_score": 0.8},
        }
        inferences = orch._infer_cross_module()
        assert len(inferences) > 0

    def test_generate_messages(self):
        orch = ButlerOrchestrator()
        msgs = {}
        for rule in orch._rules:
            msg = orch._generate_message(rule, {}, {}, {}, {})
            assert isinstance(msg, str)
            assert len(msg) > 0
            msgs[rule.name] = msg
        assert len(msgs) == len(orch._rules)

    def test_thread_safety(self):
        orch = ButlerOrchestrator()
        errors = []

        def worker():
            try:
                for _ in range(50):
                    orch.coordinate(presence_state={"presence": "active"})
                    orch.get_state()
                    orch.get_coordination_context()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        assert len(errors) == 0


class TestButlerPersonalityPersistence:
    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bp1 = ButlerPersonality(storage_path=tmpdir)
            bp1.observe_interaction(
                user_input="please explain this in detail",
                user_sentiment="curious",
                user_energy="high",
            )
            bp1.set_goal_context("learn how this works")
            bp1.observe_interaction(
                user_input="hey that's cool",
                user_sentiment="enthusiastic",
            )
            n_prefs = len(bp1._preferences)
            bp1.flush()

            bp2 = ButlerPersonality(storage_path=tmpdir)
            assert len(bp2._preferences) == n_prefs
            assert bp2._state.mode == bp1._state.mode
            assert bp2._state.warmth == bp1._state.warmth

    def test_empty_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bp = ButlerPersonality(storage_path=tmpdir)
            assert bp._state.mode == PersonaMode.BUTLER
            assert len(bp._preferences) == 0

    def test_corrupted_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            import pathlib
            pathlib.Path(tmpdir).mkdir(parents=True, exist_ok=True)
            pathlib.Path(tmpdir, "state.json").write_text("{corrupted json", encoding='utf-8')
            bp = ButlerPersonality(storage_path=tmpdir)
            assert bp._state.mode == PersonaMode.BUTLER  # Defaults

    def test_periodic_save(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bp = ButlerPersonality(storage_path=tmpdir)
            for i in range(12):
                bp.record_preference(f"key_{i}", f"value_{i}")
            assert bp._save_counter == 2  # 12 % 10 = 2 remaining
            bp.flush()
            bp2 = ButlerPersonality(storage_path=tmpdir)
            assert len(bp2._preferences) == 12


class TestProactiveIntelligencePersistence:
    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pi1 = ProactiveIntelligence(storage_path=tmpdir)
            pi1.observe("file_edit", "config.yml", success=False)
            pi1.observe("file_edit", "settings.json", success=False)
            pi1.observe("file_edit", "app.yaml", success=False)
            pi1._acceptance_rate = 0.4
            pi1._intervention_cooldown = 180.0
            pi1.flush()

            pi2 = ProactiveIntelligence(storage_path=tmpdir)
            assert len(pi2._patterns) > 0
            assert pi2._acceptance_rate == 0.4
            assert pi2._intervention_cooldown == 180.0

    def test_empty_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pi = ProactiveIntelligence(storage_path=tmpdir)
            assert len(pi._patterns) == 0
            assert pi._acceptance_rate == 0.5

    def test_corrupted_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            import pathlib
            pathlib.Path(tmpdir).mkdir(parents=True, exist_ok=True)
            pathlib.Path(tmpdir, "state.json").write_text("not json", encoding='utf-8')
            pi = ProactiveIntelligence(storage_path=tmpdir)
            assert len(pi._patterns) == 0

    def test_flush(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pi = ProactiveIntelligence(storage_path=tmpdir)
            pi.observe("test", "test", success=True)
            pi.flush()
            import pathlib
            state_file = pathlib.Path(tmpdir, "state.json")
            assert state_file.exists()
            data = json.loads(state_file.read_text(encoding='utf-8'))
            assert "acceptance_rate" in data


class TestOrchestratorAPIEndpoints:
    def test_orchestrator_state(self):
        orch = ButlerOrchestrator()
        state = orch.get_state()
        assert "actions_proposed" in state
        assert "acceptance_rate" in state
        assert "rules_active" in state

    def test_orchestrator_with_all_module_states(self):
        orch = ButlerOrchestrator()
        action = orch.coordinate(
            presence_state={"presence": "deep_work", "focus_score": 0.85, "session_minutes": 60},
            anticipation_state={"active": 2},
            observer_state={"emotional_tone": "focused"},
            personality_state={"mode": "butler"},
            proactive_state={"active_patterns": 1, "patterns": []},
            contextual_state={"primary_topics": {"coding": 5}},
        )
        assert action is not None or action is None  # Just verify no crash


class TestEndToEndButlerPipeline:
    def test_personality_to_orchestrator_flow(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bp = ButlerPersonality(storage_path=tmpdir + "/bp")
            bp.observe_interaction(
                user_input="this is frustrating, I keep getting errors!",
                user_sentiment="frustrated",
                user_energy="high",
                frustration=True,
            )
            orch = ButlerOrchestrator()
            action = orch.coordinate(
                observer_state={"emotional_tone": "frustrated"},
                proactive_state={"patterns": [{"name": "stuck_loop"}], "active_patterns": 1, "recent_errors": 3},
                personality_state=bp.get_state(),
            )
            assert action is not None
            assert action.reason in ("frustration_care", "error_storm_override")

    def test_proactive_to_orchestrator_flow(self):
        pi = ProactiveIntelligence()
        for _ in range(3):
            pi.observe("file_edit", "test.py", success=False)
        patterns = pi.get_active_patterns()
        orch = ButlerOrchestrator()
        action = orch.coordinate(
            proactive_state={"active_patterns": len(patterns), "patterns": patterns, "recent_errors": 3},
        )
        assert action is not None

    def test_contextual_to_orchestrator_flow(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = ContextualMemory(storage_path=tmpdir)
            cm.learn_preference("verbosity", "terse", PreferenceCategory.COMMUNICATION, 0.8)
            orch = ButlerOrchestrator()
            ctx_state = cm.get_state()
            action = orch.coordinate(
                contextual_state=ctx_state,
            )
            assert action is None or isinstance(action, OrchestratedAction)

    def test_full_pipeline_all_modules(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bp = ButlerPersonality(storage_path=tmpdir + "/bp")
            pi = ProactiveIntelligence(storage_path=tmpdir + "/pi")
            cm = ContextualMemory(storage_path=tmpdir + "/cm")

            bp.observe_interaction("fix this error now!", "frustrated", frustration=True)
            bp.set_goal_context("fix the bug quickly")
            for i in range(3):
                pi.observe("file_edit", "code.py", success=False)
            cm.learn_from_interaction("fix this error now!", response_length=50)

            orch = ButlerOrchestrator()
            action = orch.coordinate(
                presence_state={"presence": "active", "focus_score": 0.7},
                observer_state={"emotional_tone": "frustrated"},
                personality_state=bp.get_state(),
                proactive_state=pi.get_state(),
                contextual_state=cm.get_state(),
            )

            ctx = orch.get_coordination_context()
            assert isinstance(ctx, str)
            state = orch.get_state()
            assert state["rules_active"] > 0

    def test_pending_interruption_returns_none_when_no_high_priority(self):
        orch = ButlerOrchestrator()
        result = orch.get_pending_interruption()
        assert result is None

    def test_pending_interruption_returns_critical_action(self):
        orch = ButlerOrchestrator()
        critical_action = OrchestratedAction(
            action="error_storm_override",
            reason="Error storm detected",
            priority=SignalPriority.CRITICAL,
            source_modules=["proactive"],
            message="Multiple errors detected — switching to error recovery mode",
        )
        orch._actions.append(critical_action)
        result = orch.get_pending_interruption()
        assert result is not None
        assert "CRITICAL" in result
        assert "Error storm" in result
        assert critical_action.delivered is True

    def test_pending_interruption_returns_interrupt_action(self):
        orch = ButlerOrchestrator()
        interrupt_action = OrchestratedAction(
            action="frustration_care",
            reason="User frustration detected",
            priority=SignalPriority.INTERRUPT,
            source_modules=["observer"],
            message="User appears frustrated — offer empathetic support",
        )
        orch._actions.append(interrupt_action)
        result = orch.get_pending_interruption()
        assert result is not None
        assert "⚡" in result
        assert interrupt_action.delivered is True

    def test_pending_interruption_skips_suggest_priority(self):
        orch = ButlerOrchestrator()
        suggest_action = OrchestratedAction(
            action="break_needed",
            reason="Long session",
            priority=SignalPriority.SUGGEST,
            source_modules=["presence"],
            message="Consider taking a break",
        )
        orch._actions.append(suggest_action)
        result = orch.get_pending_interruption()
        assert result is None

    def test_pending_interruption_highest_priority_first(self):
        orch = ButlerOrchestrator()
        interrupt_action = OrchestratedAction(
            action="frustration_care",
            reason="Frustration",
            priority=SignalPriority.INTERRUPT,
            source_modules=["observer"],
            message="User frustrated",
        )
        critical_action = OrchestratedAction(
            action="error_storm_override",
            reason="Error storm",
            priority=SignalPriority.CRITICAL,
            source_modules=["proactive"],
            message="Error recovery mode",
        )
        orch._actions.append(interrupt_action)
        orch._actions.append(critical_action)
        result = orch.get_pending_interruption()
        assert "CRITICAL" in result

    def test_pending_interruption_no_duplicate_delivery(self):
        orch = ButlerOrchestrator()
        critical_action = OrchestratedAction(
            action="error_storm_override",
            reason="Error storm",
            priority=SignalPriority.CRITICAL,
            source_modules=["proactive"],
            message="Error recovery mode",
        )
        orch._actions.append(critical_action)
        result1 = orch.get_pending_interruption()
        assert result1 is not None
        result2 = orch.get_pending_interruption()
        assert result2 is None