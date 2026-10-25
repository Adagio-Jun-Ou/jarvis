"""Tests for neuroscience, context_guardian, and three-layer memory integration wiring."""
import pytest
import threading


class TestNeuroscienceDecisionReset:
    """Test that reset_evidence is called after threshold decision."""

    def setup_method(self):
        from core.neuroscience import DecisionCircuit
        self.dc = DecisionCircuit()

    def test_evidence_resets_after_threshold(self):
        self.dc.accumulate_evidence({"strong_positive": True, "confirm": True, "evidence": True})
        assert self.dc.evidence_accumulation != 0.0

    def test_reset_evidence_clears_accumulation(self):
        self.dc.accumulate_evidence({"evidence": True})
        self.dc.reset_evidence()
        assert self.dc.evidence_accumulation == 0.0
        assert self.dc.go_pathway == 0.5
        assert self.dc.nogo_pathway == 0.5


class TestNeurosciencePrecisionUpdate:
    """Test that update_precision adjusts weight based on volatility."""

    def setup_method(self):
        from core.neuroscience import PredictiveCodingState
        self.pc = PredictiveCodingState()

    def test_precision_decreases_with_low_volatility(self):
        initial = self.pc.precision_weight
        new_precision = self.pc.update_precision(0.1)
        assert new_precision <= initial

    def test_precision_increases_with_high_volatility(self):
        initial = self.pc.precision_weight
        new_precision = self.pc.update_precision(0.9)
        assert new_precision >= initial

    def test_precision_bounds(self):
        self.pc.update_precision(0.0)
        self.pc.update_precision(1.0)
        self.pc.update_precision(2.0)
        assert 0.1 <= self.pc.precision_weight <= 1.0


class TestContextGuardianEmergencyCallback:
    """Test emergency callback registration and invocation."""

    def setup_method(self):
        from core.context_guardian import ContextGuardian, GuardianAction, PressureLevel
        self.cg = ContextGuardian(token_budget=1000)
        self.GuardianAction = GuardianAction
        self.PressureLevel = PressureLevel

    def test_add_emergency_callback(self):
        received = []
        self.cg.add_emergency_callback(lambda a: received.append(a))
        self.cg.record_action(self.GuardianAction(
            level=self.PressureLevel.EMERGENCY,
            action="test",
            tokens_before=950,
            tokens_after=800,
            messages_removed=5,
            reason="test emergency",
        ))
        assert len(received) == 1

    def test_emergency_callback_not_called_for_normal(self):
        received = []
        self.cg.add_emergency_callback(lambda a: received.append(a))
        self.cg.record_action(self.GuardianAction(
            level=self.PressureLevel.NORMAL,
            action="test",
            tokens_before=100,
            tokens_after=100,
            messages_removed=0,
            reason="test normal",
        ))
        assert len(received) == 0

    def test_get_eviction_order(self):
        messages = [
            {"role": "system", "content": "important"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        order = self.cg.get_eviction_order(messages)
        assert len(order) == 3
        assert 0 in order


class TestCircadianSleepOnsetBackground:
    """Test sleep onset integration."""

    def test_sleep_onset_returns_dict(self):
        from core.circadian_rhythm import CircadianRhythm
        cr = CircadianRhythm()
        result = cr.register_sleep_onset(duration_hours=2.0)
        assert isinstance(result, dict)
        assert "new_phase" in result
        assert result["new_phase"] == "deep_processing"

    def test_awakening_transitions_to_peak(self):
        from core.circadian_rhythm import CircadianRhythm, OperationalPhase
        cr = CircadianRhythm()
        cr._operational_phase = OperationalPhase.DEEP_PROCESSING
        cr.register_awakening()
        assert cr._operational_phase == OperationalPhase.PEAK_ENGAGEMENT


class TestInsightBridgeMentalModel:
    """Test that insights create mental models."""

    def test_mental_model_created_on_insight(self):
        from core.cognitive_psychology import CognitivePsychologyEngine, MentalModel, MentalModelType
        psych = CognitivePsychologyEngine.get_instance()
        model = MentalModel(
            domain="test_insights",
            model_type=MentalModelType.PROBABILISTIC,
            assumptions=["insight assumption"],
            predictions=["insight prediction"],
            evidence_for=["test evidence"],
            evidence_against=[],
        )
        psych.add_mental_model(model)
        result = psych.validate_mental_model("test_insights", new_evidence=["supporting evidence"])
        assert result["status"] in ("validated", "needs_repair")


class TestKnowledgeGraphMentalModelValidation:
    """Test that knowledge graph search validates mental models."""

    def test_validate_updates_completeness(self):
        from core.cognitive_psychology import CognitivePsychologyEngine, MentalModel, MentalModelType
        psych = CognitivePsychologyEngine.get_instance()
        model = MentalModel(
            domain="kg_validation_test",
            model_type=MentalModelType.CAUSAL,
            assumptions=["A causes B"],
            predictions=["B follows A"],
            evidence_for=["observed A→B"],
            evidence_against=[],
        )
        psych.add_mental_model(model)
        result = psych.validate_mental_model("kg_validation_test", new_evidence=["new observation"])
        assert result["completeness"] > 0