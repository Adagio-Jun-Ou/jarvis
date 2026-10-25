"""Tests for Emotion Regulation, Allostasis Predictor, and consciousness deepening features.

Tests:
- EmotionRegulationEngine: strategy selection, regulation effects, budget management
- AllostasisPredictor: allostatic load, anticipatory signals, recovery estimation
- GlobalWorkspace deepening: decay, reverberation, amplification, history
- StreamConsciousness deepening: fork, merge, topic drift, dream thoughts
- RecursiveSelfModel deepening: cognitive load, conflicts, introspection
- Integration: regulation loop (interoception → regulation → budget → bio)
"""

import pytest
import time

from core.emotion_regulation import (
    EmotionRegulationEngine, RegulationStrategy, RegulationAttempt,
    BodyBudget, BodyBudgetState,
    get_emotion_regulation, reset_emotion_regulation,
)
from core.bio_homeostasis import BioHomeostasis
from core.circadian_rhythm import CircadianRhythm
from core.interoception import (
    InteroceptionEngine, InteroceptiveAttentionModulator, FeelingQuality, BodyChannel,
    AllostasisPredictor, get_allostasis_predictor, reset_allostasis_predictor,
    get_interoception_engine, reset_interoception_engine,
)
from core.global_workspace import (
    GlobalWorkspace, CoalitionType, WorkspaceState,
    get_global_workspace, reset_global_workspace,
)
from core.recursive_self_model import (
    RecursiveSelfModel, SelfModelLevel, SelfModelMode,
    get_recursive_self_model, reset_recursive_self_model,
)
from core.stream_consciousness import (
    StreamOfConsciousness, ThoughtType, ChainRelation,
    get_stream_of_consciousness, reset_stream_of_consciousness,
)


class TestEmotionRegulation:
    def setup_method(self):
        reset_emotion_regulation()

    def test_regulate_high_arousal_negative(self):
        ere = EmotionRegulationEngine()
        feeling = {"valence": -0.5, "arousal": 0.85, "dominance": 0.2,
                   "quality": "tense", "awareness_score": 0.7,
                   "prediction_errors": {"cardiac": 0.5}}
        result = ere.regulate(feeling)
        assert result["strategy"] in (
            RegulationStrategy.COGNITIVE_REAPPRAISAL.value,
            RegulationStrategy.ATTENTIONAL_DEPLOYMENT.value,
        )
        assert "arousal_adjusted" in result
        assert "budget_before" in result

    def test_regulate_high_arousal_positive(self):
        ere = EmotionRegulationEngine()
        feeling = {"valence": 0.5, "arousal": 0.8, "dominance": 0.7,
                   "quality": "energized", "awareness_score": 0.5}
        result = ere.regulate(feeling)
        assert result["strategy"] == RegulationStrategy.AMPLIFICATION.value
        assert result["effects"]["arousal_adjustment"] > 0

    def test_regulate_calm_state(self):
        ere = EmotionRegulationEngine()
        feeling = {"valence": 0.1, "arousal": 0.35, "dominance": 0.55,
                   "quality": "grounded", "awareness_score": 0.5}
        result = ere.regulate(feeling)
        assert result["strategy"] == RegulationStrategy.ACCEPTANCE.value

    def test_budget_tracking(self):
        ere = EmotionRegulationEngine()
        ere.update_budget_from_stress(0.5, "test_stress")
        state = ere.get_state()
        assert state["budget"]["energy"] < 0.7
        assert state["budget"]["state"] in ("mild_deficit", "balanced", "deficit")

    def test_budget_replenishment(self):
        ere = EmotionRegulationEngine()
        ere.update_budget_from_stress(0.8, "big_stress")
        budget_before = ere.get_state()["budget"]["overall"]
        ere.replenish_budget(0.3, "reappraisal")
        budget_after = ere.get_state()["budget"]["overall"]
        assert budget_after > budget_before

    def test_reappraisal_generates_narrative(self):
        ere = EmotionRegulationEngine()
        feeling = {"valence": -0.4, "arousal": 0.8, "dominance": 0.3,
                   "quality": "tense", "awareness_score": 0.7,
                   "prediction_errors": {"cardiac": 0.3}}
        result = ere.regulate(feeling)
        if result["strategy"] == RegulationStrategy.COGNITIVE_REAPPRAISAL.value:
            assert len(result["reappraisal"]) > 10

    def test_regulation_history(self):
        ere = EmotionRegulationEngine()
        ere._regulation_cooldown = 0.0
        for i in range(5):
            feeling = {"valence": -0.1 * i, "arousal": 0.5 + 0.1 * i,
                       "dominance": 0.5, "quality": "grounded", "awareness_score": 0.5}
            ere.regulate(feeling)
        state = ere.get_state()
        assert state["regulation_count"] >= 5
        assert len(state["recent_regulations"]) <= 10

    def test_evaluate_effectiveness(self):
        ere = EmotionRegulationEngine()
        effectiveness = ere.evaluate_effectiveness(
            RegulationStrategy.COGNITIVE_REAPPRAISAL,
            before_arousal=0.8, before_valence=-0.3,
            after_arousal=0.6, after_valence=0.1,
        )
        assert 0.0 <= effectiveness <= 1.0
        state = ere.get_state()
        assert RegulationStrategy.COGNITIVE_REAPPRAISAL.value in state["average_effectiveness"]

    def test_strategy_effectiveness_tracking(self):
        ere = EmotionRegulationEngine()
        for _ in range(10):
            ere.evaluate_effectiveness(
                RegulationStrategy.ACCEPTANCE,
                before_arousal=0.5, before_valence=0.0,
                after_arousal=0.45, after_valence=0.05,
            )
        state = ere.get_state()
        avg = state["average_effectiveness"][RegulationStrategy.ACCEPTANCE.value]
        assert 0.0 <= avg <= 1.0

    def test_singleton(self):
        e1 = get_emotion_regulation()
        e2 = get_emotion_regulation()
        assert e1 is e2

    def test_crisis_budget_triggers_restoration(self):
        ere = EmotionRegulationEngine()
        for _ in range(15):
            ere.update_budget_from_stress(0.2, "crisis_test")
        feeling = {"valence": -0.5, "arousal": 0.5, "dominance": 0.2,
                   "quality": "overwhelmed", "awareness_score": 0.3}
        result = ere.regulate(feeling)
        assert result["strategy"] in (
            RegulationStrategy.BODY_BUDGET_RESTORATION.value,
            RegulationStrategy.SUPPRESSION.value,
            RegulationStrategy.ATTENTIONAL_DEPLOYMENT.value,
        )

    def test_dreaming_state_uses_acceptance(self):
        ere = EmotionRegulationEngine()
        feeling = {"valence": -0.2, "arousal": 0.4, "dominance": 0.3,
                   "quality": "scattered", "awareness_score": 0.4,
                   "prediction_errors": {"metabolic": 0.1}}
        cognitive = {"state": "dreaming"}
        result = ere.regulate(feeling, cognitive_state=cognitive)
        assert result["strategy"] == RegulationStrategy.ACCEPTANCE.value

    def test_contemplating_uses_reappraisal(self):
        ere = EmotionRegulationEngine()
        feeling = {"valence": -0.1, "arousal": 0.55, "dominance": 0.5,
                   "quality": "tense", "awareness_score": 0.65,
                   "prediction_errors": {"cardiac": 0.3}}
        cognitive = {"state": "contemplating"}
        result = ere.regulate(feeling, cognitive_state=cognitive)
        assert result["strategy"] == RegulationStrategy.COGNITIVE_REAPPRAISAL.value

    def test_get_state_structure(self):
        ere = EmotionRegulationEngine()
        state = ere.get_state()
        assert "budget" in state
        assert "regulation_count" in state
        assert "average_effectiveness" in state
        assert "recent_regulations" in state
        assert state["budget"]["overall"] >= 0.0


class TestAllostasisPredictor:
    def setup_method(self):
        reset_allostasis_predictor()

    def test_predict_needs_calm_state(self):
        ap = AllostasisPredictor()
        feeling = {"arousal": 0.35, "valence": 0.1, "prediction_errors": {}}
        result = ap.predict_needs(feeling)
        assert "allostatic_load" in result
        assert result["regulation_need"] in ("none", "low")

    def test_predict_needs_stressed_state(self):
        ap = AllostasisPredictor()
        ap.apply_stressor(0.5, "test")
        feeling = {"arousal": 0.9, "valence": -0.5, "prediction_errors": {"cardiac": 0.6}}
        bio = {"stress_intensity": 0.7, "cognitive_capacity": 0.3}
        result = ap.predict_needs(feeling, bio_state=bio)
        assert result["allostatic_load"] > 0.15
        assert result["regulation_need"] in ("moderate", "high", "urgent")

    def test_stressor_increases_load(self):
        ap = AllostasisPredictor()
        ap.apply_stressor(0.5, "test")
        state = ap.get_state()
        assert state["allostatic_load"] > 0.0
        assert state["stress_level"] > 0.0

    def test_recovery_decreases_load(self):
        ap = AllostasisPredictor()
        ap.apply_stressor(0.5, "test")
        load_before = ap.get_state()["allostatic_load"]
        ap.apply_recovery(0.3, "rest")
        load_after = ap.get_state()["allostatic_load"]
        assert load_after < load_before

    def test_tick_recovery(self):
        ap = AllostasisPredictor()
        ap.apply_stressor(0.5, "test")
        load_before = ap.get_state()["allostatic_load"]
        ap.tick(dt=10.0)
        load_after = ap.get_state()["allostatic_load"]
        assert load_after <= load_before

    def test_anticipatory_signals_high_arousal(self):
        ap = AllostasisPredictor()
        feeling = {"arousal": 0.85, "valence": -0.3, "prediction_errors": {"cardiac": 0.4}}
        result = ap.predict_needs(feeling)
        signals = result["anticipatory_signals"]
        assert any(s["signal"] == "energy_reserve_depletion" for s in signals)

    def test_anticipatory_signals_negative_valence(self):
        ap = AllostasisPredictor()
        feeling = {"arousal": 0.5, "valence": -0.5, "prediction_errors": {}}
        result = ap.predict_needs(feeling)
        signals = result["anticipatory_signals"]
        assert any(s["signal"] == "negative_valence_accumulation" for s in signals)

    def test_circadian_prediction(self):
        ap = AllostasisPredictor()
        feeling = {"arousal": 0.3, "valence": 0.1, "prediction_errors": {}}
        result = ap.predict_needs(feeling)
        assert "recommended_strategies" in result
        assert isinstance(result["recommended_strategies"], list)

    def test_load_state_classification(self):
        ap = AllostasisPredictor()
        result = ap.predict_needs({"arousal": 0.3, "valence": 0.1, "prediction_errors": {}})
        assert result["load_state"] in ("recovered", "mild_stress")

    def test_singleton(self):
        a1 = get_allostasis_predictor()
        a2 = get_allostasis_predictor()
        assert a1 is a2


class TestGlobalWorkspaceDeepening:
    def setup_method(self):
        reset_global_workspace()

    def test_get_ignition_history(self):
        gw = GlobalWorkspace()
        gw.submit_coalition(CoalitionType.CONCEPTUAL, "insight 1", 0.8)
        gw.compete()
        gw.submit_coalition(CoalitionType.MEMORY, "memory fragment", 0.7)
        gw.compete()
        history = gw.get_ignition_history(limit=5)
        assert len(history) >= 1
        assert history[0]["type"] == "conceptual"
        assert "strength" in history[0]

    def test_decay_ignitions(self):
        gw = GlobalWorkspace()
        gw.submit_coalition(CoalitionType.CONCEPTUAL, "fading thought", 0.9)
        gw.compete()
        gw.decay_ignitions(dt=10.0)
        assert gw.get_state()["total_ignitions"] >= 1

    def test_reverberate_returns_echoes(self):
        gw = GlobalWorkspace()
        gw.submit_coalition(CoalitionType.CONCEPTUAL, "bright idea", 0.9)
        gw.compete()
        echoes = gw.reverberate()
        assert isinstance(echoes, list)

    def test_amplify_context(self):
        gw = GlobalWorkspace()
        gw.submit_coalition(CoalitionType.EMOTIONAL, "feeling strong", 0.3)
        gw.amplify_context("emotion", strength_boost=0.3)
        state = gw.get_state()
        assert state["competing_coalitions"] >= 1

    def test_amplify_unknown_context(self):
        gw = GlobalWorkspace()
        gw.amplify_context("unknown_context", strength_boost=0.2)
        assert gw.get_state()["competing_coalitions"] == 0


class TestStreamConsciousnessDeepening:
    def setup_method(self):
        reset_stream_of_consciousness()

    def test_fork_creates_branches(self):
        soc = StreamOfConsciousness()
        soc.add_thought("original thought", ThoughtType.PERCEPTION, salience=0.7, topic="AI")
        child_a, child_b = soc.fork("branching idea", topic="AI_2")
        assert child_a is not None
        assert child_b is not None
        assert child_a.content != child_b.content

    def test_fork_on_empty_stream(self):
        soc = StreamOfConsciousness()
        a, b = soc.fork("branch")
        assert a is None
        assert b is None

    def test_merge_substreams(self):
        soc = StreamOfConsciousness()
        result = soc.merge_substreams(["thread one", "thread two", "thread three"],
                                       topic="merged")
        assert result is not None
        assert "thread one" in result.content
        assert "+" in result.content

    def test_merge_empty_list(self):
        soc = StreamOfConsciousness()
        result = soc.merge_substreams([], topic="none")
        assert result is None

    def test_detect_topic_drift_no_drift(self):
        soc = StreamOfConsciousness()
        for i in range(10):
            soc.add_thought(f"thought about topic_a", ThoughtType.REFLECTION,
                           salience=0.5, topic="topic_a", relation=ChainRelation.THEMATIC)
        result = soc.detect_topic_drift(window_size=10)
        assert result["drift"] <= 0.3
        assert not result["drifting"]

    def test_detect_topic_drift_with_drift(self):
        soc = StreamOfConsciousness()
        for i in range(5):
            soc.add_thought(f"thought about topic_a", ThoughtType.REFLECTION,
                           salience=0.5, topic="topic_a", relation=ChainRelation.THEMATIC)
        for i in range(5):
            soc.add_thought(f"thought about topic_b", ThoughtType.REFLECTION,
                           salience=0.5, topic="topic_b", relation=ChainRelation.ASSOCIATIVE)
        for i in range(5):
            soc.add_thought(f"thought about topic_c", ThoughtType.REFLECTION,
                           salience=0.5, topic="topic_c", relation=ChainRelation.CONTRASTIVE)
        result = soc.detect_topic_drift(window_size=10)
        assert result["drift"] > 0.0

    def test_generate_dream_thought(self):
        soc = StreamOfConsciousness()
        soc.add_thought("waking thought about patterns", ThoughtType.PERCEPTION, topic="patterns")
        dream = soc.generate_dream_thought(current_topic="patterns")
        assert dream is not None
        assert len(dream.content) > 5

    def test_dream_thought_uses_topic_history(self):
        soc = StreamOfConsciousness()
        soc.add_thought("conscious thought about AI", ThoughtType.PERCEPTION, topic="AI")
        soc.add_thought("another about consciousness", ThoughtType.PERCEPTION, topic="consciousness")
        dream = soc.generate_dream_thought(current_topic="AI")
        assert dream is not None


class TestRecursiveSelfModelDeepening:
    def setup_method(self):
        reset_recursive_self_model()

    def test_assess_cognitive_load_low(self):
        rsm = RecursiveSelfModel()
        result = rsm.assess_cognitive_load(external_load=0.2)
        assert result["total_load"] < 0.5
        assert result["available_depth"] >= 2
        assert not result["should_throttle"]

    def test_assess_cognitive_load_high(self):
        rsm = RecursiveSelfModel()
        result = rsm.assess_cognitive_load(external_load=0.9)
        assert result["total_load"] > 0.5
        assert result["should_throttle"] or result["total_load"] > 0.7

    def test_detect_conflicts_none(self):
        rsm = RecursiveSelfModel()
        rsm.observe(SelfModelLevel.FIRST_ORDER, "processing normally", confidence=0.7)
        conflicts = rsm.detect_conflicts()
        assert isinstance(conflicts, list)

    def test_introspect_basic(self):
        rsm = RecursiveSelfModel()
        rsm.observe(SelfModelLevel.OBJECT, "observing the world", confidence=0.8)
        rsm.observe(SelfModelLevel.FIRST_ORDER, "aware of processing", confidence=0.6)
        result = rsm.introspect(depth=2)
        assert result["current_mode"] in ("automatic", "monitored", "reflective", "meta_aware")
        assert "observations_by_level" in result
        assert len(result["observations_by_level"]) >= 2

    def test_introspect_with_depth(self):
        rsm = RecursiveSelfModel()
        rsm.observe(SelfModelLevel.META, "meta-awareness", confidence=0.5)
        result = rsm.introspect(depth=3)
        assert result["introspection_depth"] == 3

    def test_assess_cognitive_load_with_bio(self):
        rsm = RecursiveSelfModel()
        rsm.tick(bio_state={"stress_intensity": 0.8, "cognitive_capacity": 0.3})
        result = rsm.assess_cognitive_load(external_load=0.5)
        assert result["total_load"] > 0.5


class TestRegulationLoopIntegration:
    def setup_method(self):
        reset_emotion_regulation()
        reset_interoception_engine()
        reset_allostasis_predictor()

    def test_interoception_to_regulation_loop(self):
        ie = InteroceptionEngine()
        ere = EmotionRegulationEngine()
        nt_state = {"dopamine": 0.3, "serotonin": 0.3, "norepinephrine": 0.7,
                    "acetylcholine": 0.5, "cortisol": 0.6}
        bio_state = {"arousal": 0.8, "valence": -0.3, "stress_intensity": 0.5}

        feeling = ie.generate_feeling(nt_state, bio_state)
        feeling_dict = {
            "valence": feeling.valence,
            "arousal": feeling.arousal,
            "dominance": feeling.dominance,
            "quality": feeling.quality.value,
            "awareness_score": feeling.awareness_score,
            "prediction_errors": {k: round(v, 3) for k, v in feeling.prediction_errors.items()},
        }
        reg_result = ere.regulate(feeling_dict, bio_state)
        assert "strategy" in reg_result
        assert "budget_before" in reg_result
        assert "arousal_adjusted" in reg_result

    def test_allostasis_with_regulation(self):
        ap = AllostasisPredictor()
        ere = EmotionRegulationEngine()
        ap.apply_stressor(0.4, "test_stress")
        feeling = {"arousal": 0.7, "valence": -0.2, "prediction_errors": {"cardiac": 0.3}}
        bio = {"stress_intensity": 0.4, "cognitive_capacity": 0.6}

        needs = ap.predict_needs(feeling, bio_state=bio)
        reg = ere.regulate(feeling, bio_state=bio)
        assert reg["strategy"] != ""
        assert needs["allostatic_load"] > 0.0

    def test_full_pipeline_stress_recovery(self):
        ie = InteroceptionEngine()
        ere = EmotionRegulationEngine()
        ap = AllostasisPredictor()

        ap.apply_stressor(0.6, "deadline")

        nt = {"dopamine": 0.2, "serotonin": 0.3, "norepinephrine": 0.8,
              "acetylcholine": 0.6, "cortisol": 0.7}
        bio = {"arousal": 0.85, "valence": -0.4, "stress_intensity": 0.6}

        feeling = ie.generate_feeling(nt, bio)
        feeling_dict = {
            "valence": feeling.valence, "arousal": feeling.arousal,
            "dominance": feeling.dominance, "quality": feeling.quality.value,
            "awareness_score": feeling.awareness_score,
            "prediction_errors": {k: round(v, 3) for k, v in feeling.prediction_errors.items()},
        }

        needs = ap.predict_needs(feeling_dict, bio)
        reg = ere.regulate(feeling_dict, bio)
        assert reg["strategy"] in (
            RegulationStrategy.COGNITIVE_REAPPRAISAL.value,
            RegulationStrategy.ATTENTIONAL_DEPLOYMENT.value,
            RegulationStrategy.BODY_BUDGET_RESTORATION.value,
            RegulationStrategy.SUPPRESSION.value,
        )

        ere.replenish_budget(0.2, "reappraisal_recovery")
        ap.apply_recovery(0.3, "cognitive_rest")


class TestCrossModuleIntegration:
    def test_interoception_to_attention_queue(self):
        from core.interoception import InteroceptionEngine, InteroceptiveAttentionModulator
        from core.attention_priority import AttentionPriorityQueue, UrgencySignal
        ie = InteroceptionEngine()
        mod = InteroceptiveAttentionModulator()
        aq = AttentionPriorityQueue()
        nt = {"dopamine": 0.2, "serotonin": 0.3, "norepinephrine": 0.8,
              "acetylcholine": 0.6, "cortisol": 0.7}
        bio = {"arousal": 0.85, "valence": -0.4, "stress_intensity": 0.6}
        feeling = ie.generate_feeling(nt, bio)
        mod.update(feeling, bio.get("stress_intensity", 0.5))
        salience = mod.get_state().get("salience_bias", 0.0)
        aq.enqueue_with_interoception(
            content="body is stressed", urgency=UrgencySignal.NORMAL,
            topic="stress", interoceptive_salience=salience,
        )
        assert len(aq._queue) >= 1

    def test_attention_reprioritize_on_interoception(self):
        from core.attention_priority import AttentionPriorityQueue, UrgencySignal
        aq = AttentionPriorityQueue()
        aq.enqueue("task A", UrgencySignal.NORMAL, topic="coding")
        aq.enqueue("task B", UrgencySignal.NORMAL, topic="body stress relief")
        modified = aq.reprioritize_based_on_interoception(0.7)
        assert modified >= 1

    def test_emotion_regulation_bio_feedback(self):
        ere = EmotionRegulationEngine()
        ere._regulation_cooldown = 0.0
        h = BioHomeostasis()
        stress_before = h.get_stress_level()
        feeling = {"valence": -0.3, "arousal": 0.8, "dominance": 0.3,
                   "quality": "tense", "awareness_score": 0.5}
        result = ere.regulate(feeling, bio_state={"stress_intensity": stress_before})
        budget_effect = result.get("budget_effect", 0.0)
        if budget_effect > 0:
            h.apply_reward(budget_effect * 0.3, "serotonin")
        elif budget_effect < 0:
            h.apply_stressor(abs(budget_effect) * 0.2, "regulation_cost")

    def test_circadian_fatigue_affects_bio_capacity(self):
        c = CircadianRhythm()
        h = BioHomeostasis()
        for _ in range(100):
            c.register_user_interaction()
            c.tick(dt_minutes=1.0)
        state = c.get_state()
        fatigue = state["processing_fatigue"]
        bio_cap = h.get_cognitive_capacity()
        assert fatigue >= 0.0
        assert 0.0 < bio_cap <= 1.0

    def test_empty_topic_not_suppressed_in_attention(self):
        from core.attention_priority import AttentionPriorityQueue, UrgencySignal
        aq = AttentionPriorityQueue()
        aq.enqueue_with_interoception(
            content="some generic task", urgency=UrgencySignal.TIME_SENSITIVE,
            topic="", interoceptive_salience=0.5,
        )
        assert len(aq._queue) >= 1
        item = aq.peek()
        assert item is not None
        assert item.urgency == UrgencySignal.TIME_SENSITIVE

    def test_operational_rhythm_circadian_phase_compat(self):
        c = CircadianRhythm()
        state = c.get_state()
        assert "circadian_phase" in state
        assert "sleep_pressure" in state
        assert 0.0 <= state["sleep_pressure"] <= 1.0

    def test_regulate_returns_post_feeling(self):
        ere = EmotionRegulationEngine()
        feeling = {"valence": -0.5, "arousal": 0.8, "quality": "anxious", "dominance": 0.3,
                  "awareness_score": 0.6, "prediction_errors": {}}
        result = ere.regulate(feeling)
        assert "post_feeling" in result
        assert isinstance(result["post_feeling"], str)

    def test_regulate_returns_budget_effect(self):
        ere = EmotionRegulationEngine()
        feeling = {"valence": -0.3, "arousal": 0.7, "quality": "stressed", "dominance": 0.4,
                  "awareness_score": 0.5, "prediction_errors": {}}
        result = ere.regulate(feeling)
        assert "budget_effect" in result
        assert isinstance(result["budget_effect"], float)

    def test_infer_post_feeling_positive_shift(self):
        ere = EmotionRegulationEngine()
        assert ere._infer_post_feeling("anxious", {"valence_adjustment": 0.2}) == "grounded"
        assert ere._infer_post_feeling("stressed", {"valence_adjustment": 0.15}) == "grounded"

    def test_infer_post_feeling_negative_shift(self):
        ere = EmotionRegulationEngine()
        assert ere._infer_post_feeling("grounded", {"valence_adjustment": -0.2}) == "anxious"

    def test_infer_post_feeling_no_change(self):
        ere = EmotionRegulationEngine()
        assert ere._infer_post_feeling("calm", {"valence_adjustment": 0.0}) == "calm"


class TestAllostasisRecoveryCurve:
    def test_recovery_curve_tracked(self):
        from core.interoception import AllostasisPredictor
        ap = AllostasisPredictor()
        ap.apply_stressor(0.5)
        ap.apply_recovery(0.2)
        ap.apply_recovery(0.1)
        assert len(ap._recovery_curve) == 2

    def test_recovery_curve_bounded(self):
        from core.interoception import AllostasisPredictor
        ap = AllostasisPredictor()
        for i in range(150):
            ap.apply_recovery(0.01)
        assert len(ap._recovery_curve) <= 100

    def test_get_state_includes_recovery_trend(self):
        from core.interoception import AllostasisPredictor
        ap = AllostasisPredictor()
        ap.apply_stressor(0.5)
        ap.apply_recovery(0.1)
        ap.apply_recovery(0.1)
        state = ap.get_state()
        assert "recovery_trend" in state

    def test_get_state_includes_recent_anticipatory(self):
        from core.interoception import AllostasisPredictor
        ap = AllostasisPredictor()
        state = ap.get_state()
        assert "recent_anticipatory" in state

    def test_recovery_estimate_uses_curve(self):
        from core.interoception import AllostasisPredictor
        ap = AllostasisPredictor()
        ap._allostatic_load = 0.5
        base_estimate = ap._estimate_recovery_time()
        ap._recovery_curve = [0.5, 0.4, 0.3, 0.2, 0.1]
        curve_estimate = ap._estimate_recovery_time()
        assert curve_estimate > 0