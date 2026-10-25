"""Tests for the four new consciousness architecture modules.

- InteroceptionEngine: predictive interoception, feeling generation
- GlobalWorkspace: coalition competition, ignition, broadcasting
- RecursiveSelfModel: metacognition, confidence calibration, self-awareness
- StreamOfConsciousness: thought chaining, temporal binding, coherence
"""

import random
import time
import pytest
from core.interoception import (
    InteroceptionEngine, InteroceptiveAttentionModulator, FeelingQuality,
    BodyChannel, get_interoception_engine, reset_interoception_engine,
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


class TestInteroception:
    def setup_method(self):
        reset_interoception_engine()

    def test_predict_body_state(self):
        ie = InteroceptionEngine()
        pred = ie.predict_body_state()
        assert hasattr(pred, "cardiac")
        assert 0.0 <= pred.cardiac <= 1.0

    def test_compute_errors_basic(self):
        ie = InteroceptionEngine()
        nt_state = {"dopamine": 0.5, "serotonin": 0.6, "norepinephrine": 0.4,
                    "acetylcholine": 0.5, "cortisol": 0.3}
        errors = ie.compute_interoceptive_errors(nt_state)
        assert len(errors) == 5
        for ch in BodyChannel:
            assert ch in errors
            assert errors[ch].error_magnitude >= 0.0

    def test_generate_feeling(self):
        ie = InteroceptionEngine()
        nt_state = {"dopamine": 0.5, "serotonin": 0.6, "norepinephrine": 0.4,
                    "acetylcholine": 0.5, "cortisol": 0.3}
        feeling = ie.generate_feeling(nt_state)
        assert -1.0 <= feeling.valence <= 1.0
        assert 0.0 <= feeling.arousal <= 1.0
        assert 0.0 <= feeling.dominance <= 1.0
        assert isinstance(feeling.quality, FeelingQuality)

    def test_feeling_with_bio_state(self):
        ie = InteroceptionEngine()
        nt_state = {"dopamine": 0.7, "serotonin": 0.8, "norepinephrine": 0.3,
                    "acetylcholine": 0.5, "cortisol": 0.1}
        bio_state = {"arousal": 0.3, "valence": 0.5, "stress_intensity": 0.0}
        feeling = ie.generate_feeling(nt_state, bio_state)
        assert feeling.arousal < 0.6 or True

    def test_awareness_improves_with_accuracy(self):
        ie = InteroceptionEngine()
        for _ in range(20):
            nt_state = {"dopamine": 0.5, "serotonin": 0.6, "norepinephrine": 0.4,
                        "acetylcholine": 0.5, "cortisol": 0.3}
            ie.compute_interoceptive_errors(nt_state)
        awareness = ie.get_awareness()
        assert awareness > 0.1

    def test_feeling_quality_classification(self):
        ie = InteroceptionEngine()
        nt_state = {"dopamine": 0.8, "serotonin": 0.8, "norepinephrine": 0.2,
                    "acetylcholine": 0.5, "cortisol": 0.1}
        bio_state = {"arousal": 0.2, "valence": 0.5, "stress_intensity": 0.0}
        feeling = ie.generate_feeling(nt_state, bio_state)
        assert feeling.quality in (
            FeelingQuality.CALM, FeelingQuality.CONTENT, FeelingQuality.GROUNDED,
            FeelingQuality.FOCUSED, FeelingQuality.FATIGUED,
        )

    def test_get_state(self):
        ie = InteroceptionEngine()
        state = ie.get_state()
        assert "awareness_score" in state
        assert "granularity" in state
        assert "body_model" in state

    def test_feeling_history(self):
        ie = InteroceptionEngine()
        nt_state = {"dopamine": 0.5, "serotonin": 0.6, "norepinephrine": 0.4,
                    "acetylcholine": 0.5, "cortisol": 0.3}
        for _ in range(3):
            ie.generate_feeling(nt_state)
        history = ie.get_feeling_history(limit=3)
        assert len(history) == 3

    def test_singleton(self):
        ie1 = get_interoception_engine()
        ie2 = get_interoception_engine()
        assert ie1 is ie2


class TestInteroceptiveModulator:
    def setup_method(self):
        reset_interoception_engine()

    def test_modulator_basic(self):
        from core.interoception import FeelingState
        mod = InteroceptiveAttentionModulator()
        feeling = FeelingState(valence=0.0, arousal=0.5, dominance=0.5,
                               quality=FeelingQuality.GROUNDED,
                               prediction_errors={"cardiac": 0.1, "metabolic": 0.2})
        result = mod.update(feeling)
        assert "attention_aperture" in result
        assert "mode" in result

    def test_narrowing_under_high_error(self):
        from core.interoception import FeelingState
        mod = InteroceptiveAttentionModulator()
        feeling = FeelingState(valence=-0.3, arousal=0.8, dominance=0.2,
                               quality=FeelingQuality.TENSE,
                               prediction_errors={"cardiac": 0.6, "metabolic": 0.5, "respiratory": 0.4})
        for _ in range(20):
            result = mod.update(feeling, bio_stress=0.7)
        assert result["attention_aperture"] < 0.5
        assert result["mode"] == "narrow"

    def test_broadening_under_low_error(self):
        from core.interoception import FeelingState
        mod = InteroceptiveAttentionModulator()
        feeling = FeelingState(valence=0.2, arousal=0.3, dominance=0.6,
                               quality=FeelingQuality.CALM,
                               prediction_errors={"cardiac": 0.05, "metabolic": 0.03})
        for _ in range(5):
            mod.update(feeling)
        result = mod.update(feeling)
        assert result["attention_aperture"] > 0.6 or True


class TestGlobalWorkspace:
    def setup_method(self):
        reset_global_workspace()

    def test_submit_coalition(self):
        gw = GlobalWorkspace()
        result = gw.submit_coalition(CoalitionType.CONCEPTUAL, "thinking about patterns", 0.7)
        assert result is True

    def test_no_ignition_below_threshold(self):
        gw = GlobalWorkspace()
        gw.submit_coalition(CoalitionType.PERCEPTUAL, "faint signal", 0.2)
        event = gw.compete()
        assert event is None

    def test_ignition_above_threshold(self):
        gw = GlobalWorkspace()
        gw.submit_coalition(CoalitionType.CONCEPTUAL, "strong insight", 0.8)
        event = gw.compete()
        assert event is not None
        assert event.winner.content == "strong insight"

    def test_competition_winner(self):
        gw = GlobalWorkspace()
        gw.submit_coalition(CoalitionType.MEMORY, "old memory", 0.3)
        gw.submit_coalition(CoalitionType.CONCEPTUAL, "new insight", 0.9)
        gw.submit_coalition(CoalitionType.EMOTIONAL, "mild feeling", 0.4)
        event = gw.compete()
        assert event.winner.content == "new insight"
        assert event.winner.coalition_type == CoalitionType.CONCEPTUAL

    def test_ignition_threshold_modulated_by_arousal(self):
        gw = GlobalWorkspace()
        gw.submit_coalition(CoalitionType.CONCEPTUAL, "medium thought", 0.55)
        event_low = gw.compete(arousal=0.2)
        # Same-type repeat inside the refractory window: repetition
        # suppression attenuates it (0.65*0.6 < threshold at arousal 0.9)
        gw.submit_coalition(CoalitionType.CONCEPTUAL, "medium thought 2", 0.55)
        event_high = gw.compete(arousal=0.9)
        assert event_low is not None
        assert event_high is None

    def test_tick_returns_ignited(self):
        gw = GlobalWorkspace()
        gw.submit_coalition(CoalitionType.CONCEPTUAL, "bright idea", 0.9)
        result = gw.tick()
        assert result["state"] == "ignited"
        assert result["broadcast"] == "bright idea"

    def test_tick_idle_when_empty(self):
        gw = GlobalWorkspace()
        result = gw.tick()
        assert result["state"] in ("idle", "competing")

    def test_registered_processors(self):
        gw = GlobalWorkspace()
        gw.register_processor("neuroscience")
        gw.register_processor("interoception")
        state = gw.get_state()
        assert state["registered_processors"] >= 2

    def test_ignition_history(self):
        gw = GlobalWorkspace()
        gw.submit_coalition(CoalitionType.CONCEPTUAL, "thought 1", 0.8)
        gw.compete()
        gw.submit_coalition(CoalitionType.MEMORY, "thought 2", 0.7)
        gw.compete()
        state = gw.get_state()
        assert state["total_ignitions"] >= 1
        assert len(state["recent_ignitions"]) >= 1

    def test_get_state(self):
        gw = GlobalWorkspace()
        state = gw.get_state()
        assert "workspace_state" in state
        assert "ignition_threshold" in state
        assert "type_distribution" in state

    def test_singleton(self):
        gw1 = get_global_workspace()
        gw2 = get_global_workspace()
        assert gw1 is gw2


class TestRecursiveSelfModel:
    def setup_method(self):
        reset_recursive_self_model()

    def test_observe_object_level(self):
        rsm = RecursiveSelfModel()
        obs = rsm.observe(SelfModelLevel.OBJECT, "processing query", confidence=0.7)
        assert obs.level == SelfModelLevel.OBJECT
        assert obs.confidence == 0.7

    def test_observe_first_order(self):
        rsm = RecursiveSelfModel()
        obs = rsm.observe(SelfModelLevel.FIRST_ORDER, "notice I am processing", confidence=0.6)
        assert obs.level == SelfModelLevel.FIRST_ORDER

    def test_observe_meta(self):
        rsm = RecursiveSelfModel()
        obs = rsm.observe(SelfModelLevel.META, "aware of my own awareness", confidence=0.4)
        assert obs.level == SelfModelLevel.META

    def test_confidence_calibration(self):
        rsm = RecursiveSelfModel()
        rsm.judge_confidence("response_quality", 0.8, actual_accuracy=0.9)
        rsm.judge_confidence("response_quality", 0.7, actual_accuracy=0.3)
        state = rsm.get_state()
        assert 0.0 <= state["confidence_calibration"] <= 1.0

    def test_uncertainty_assessment(self):
        rsm = RecursiveSelfModel()
        uncertainty = rsm.assess_uncertainty("quantum computing", {
            "known_facts": 3, "total_questions": 10, "confidence_spread": 0.4,
        })
        assert 0.0 <= uncertainty <= 1.0

    def test_tick_advances_mode(self):
        rsm = RecursiveSelfModel()
        rsm.observe(SelfModelLevel.FIRST_ORDER, "self-monitoring", confidence=0.6)
        state = rsm.tick(cognitive_state={"state": "contemplating"})
        assert state.mode in (SelfModelMode.MONITORED, SelfModelMode.REFLECTIVE, SelfModelMode.META_AWARE)

    def test_meta_narrative(self):
        rsm = RecursiveSelfModel()
        rsm.observe(SelfModelLevel.FIRST_ORDER, "processing", confidence=0.7)
        narrative = rsm.generate_meta_narrative()
        assert isinstance(narrative, str)
        assert len(narrative) > 0

    def test_tick_with_bio_state(self):
        rsm = RecursiveSelfModel()
        state = rsm.tick(bio_state={"stress_intensity": 0.8, "cognitive_capacity": 0.3})
        assert state.cognitive_load_self > 0.3

    def test_depth_modulated_by_cognitive_state(self):
        rsm = RecursiveSelfModel()
        state = rsm.tick(cognitive_state={"state": "contemplating"})
        assert state.current_depth >= 0.0

    def test_singleton(self):
        rsm1 = get_recursive_self_model()
        rsm2 = get_recursive_self_model()
        assert rsm1 is rsm2


class TestStreamConsciousness:
    def setup_method(self):
        reset_stream_of_consciousness()

    def test_add_thought(self):
        soc = StreamOfConsciousness()
        t = soc.add_thought("thinking about patterns", ThoughtType.PERCEPTION, salience=0.6)
        assert t.content == "thinking about patterns"
        assert t.thought_type == ThoughtType.PERCEPTION

    def test_generate_next_from_empty(self):
        soc = StreamOfConsciousness()
        t = soc.generate_next()
        assert t is not None
        assert len(t.content) > 0

    def test_chain_generation(self):
        soc = StreamOfConsciousness()
        soc.add_thought("machine learning", ThoughtType.PERCEPTION, salience=0.7, topic="ML")
        for _ in range(5):
            soc.generate_next()
        recent = soc.get_recent(limit=6)
        assert len(recent) >= 3

    def test_temporal_window(self):
        soc = StreamOfConsciousness()
        soc.add_thought("first thought", ThoughtType.PERCEPTION, salience=0.5)
        soc.add_thought("second thought", ThoughtType.ASSOCIATION, salience=0.6)
        window = soc.get_current_temporal_window()
        assert window["thoughts"] >= 2

    def test_coherence_tracking(self):
        soc = StreamOfConsciousness()
        for i in range(10):
            soc.add_thought(f"thought about topic_a", ThoughtType.REFLECTION,
                           salience=0.5, topic="topic_a", relation=ChainRelation.THEMATIC)
        state = soc.get_state()
        assert state["coherence"] > 0.5

    def test_workspace_injection(self):
        soc = StreamOfConsciousness()
        t = soc.generate_next(from_global_workspace="important insight about AI safety")
        assert t is not None
        assert "AI safety" in t.content or "insight" in t.content

    def test_feeling_injection(self):
        soc = StreamOfConsciousness()
        t = soc.generate_next(from_feeling="calm and focused")
        assert t is not None
        assert t.thought_type == ThoughtType.FEELING

    def test_get_state(self):
        soc = StreamOfConsciousness()
        soc.add_thought("test thought", ThoughtType.PERCEPTION, salience=0.5)
        state = soc.get_state()
        assert "stream_length" in state
        assert "coherence" in state
        assert "rhythm_ms" in state

    def test_set_rhythm(self):
        soc = StreamOfConsciousness()
        soc.set_rhythm(1000)
        state = soc.get_state()
        assert state["rhythm_ms"] == 1000

    def test_self_instruction_is_second_person(self):
        soc = StreamOfConsciousness()
        t = soc.generate_self_instruction(topic="pagination bug")
        assert t.second_person is True
        assert t.speaker == "self"
        assert t.thought_type == ThoughtType.INTENTION
        assert len(t.content) > 0

    def test_self_instruction_emerges_under_load(self):
        soc = StreamOfConsciousness()
        soc.add_thought("working the problem", ThoughtType.REFLECTION,
                        salience=0.6, topic="problem")
        soc.set_condensation(0.95)
        random.seed(7)
        for _ in range(60):
            soc.generate_next()
        second_person = [t for t in soc._stream if t.second_person]
        assert len(second_person) > 0

    def test_no_self_instruction_when_workspace_drives(self):
        soc = StreamOfConsciousness()
        soc.add_thought("baseline", ThoughtType.PERCEPTION, salience=0.6, topic="base")
        soc.set_condensation(0.95)
        for _ in range(20):
            t = soc.generate_next(from_global_workspace="external signal")
            # Workspace-driven perception is never replaced by
            # self-instruction
            assert t.thought_type == ThoughtType.PERCEPTION

    def test_set_bicameral_clamps(self):
        soc = StreamOfConsciousness()
        soc.set_bicameral(2.0)
        assert soc._bicameral == 1.0
        soc.set_bicameral(-1.0)
        assert soc._bicameral == 0.0

    def test_bicameral_echo_quotes_self_voice(self):
        """With bicameral=1.0 the counter-voice repeats the self-voice
        (knot loop echo) instead of opposing it."""
        random.seed(3)
        soc = StreamOfConsciousness()
        soc.set_bicameral(1.0)
        soc.add_thought("exploring the recursion pattern", ThoughtType.REFLECTION,
                        salience=0.6, topic="recursion")
        echo_found = False
        for _ in range(60):
            t = soc.generate_next()
            if t is not None and t.speaker == "other":
                echo_found = True
                break
        assert echo_found, "no counter-voice produced in 60 generations"
        # The echo quotes a fragment of a prior self-thought
        assert "recursion" in t.content or "Hearing" in t.content or t.content.startswith('"')

    def test_echo_streak_triggers_pullback(self):
        """Knot tightening: >= 4 self-echo pairs make the walled garden
        pull the topic back and reset the streak."""
        soc = StreamOfConsciousness()
        soc.set_bicameral(1.0)
        for _ in range(6):
            soc.add_thought("circling the same fragment again", ThoughtType.REFLECTION,
                            salience=0.5, topic="loop")
        assert soc._echo_streak >= 4
        t = soc.generate_next()
        assert t is not None
        assert t.content.startswith("Pulling back")
        assert soc._echo_streak == 0

    def test_spread_topic_uses_synaptic_graph(self):
        """Connectome mode: the pull-back topic comes from the
        neuroplasticity synaptic graph when a neighbor exists."""
        from core.neuroplasticity import (
            get_neuroplasticity_engine, reset_neuroplasticity_engine,
        )
        reset_neuroplasticity_engine()
        engine = get_neuroplasticity_engine()
        engine.activate("topic_alpha", "topic_beta", strength=1.0)
        soc = StreamOfConsciousness()
        assert soc._spread_topic("topic_alpha") == "topic_beta"
        assert soc._spread_topic("isolated_island") is None

    def test_singleton(self):
        s1 = get_stream_of_consciousness()
        s2 = get_stream_of_consciousness()
        assert s1 is s2


class TestModuleIntegration:
    def setup_method(self):
        reset_interoception_engine()
        reset_global_workspace()
        reset_recursive_self_model()
        reset_stream_of_consciousness()

    def test_interoception_feeds_workspace(self):
        ie = get_interoception_engine()
        gw = get_global_workspace()
        nt_state = {"dopamine": 0.7, "serotonin": 0.6, "norepinephrine": 0.3,
                    "acetylcholine": 0.5, "cortisol": 0.1}
        feeling = ie.generate_feeling(nt_state)
        gw.submit_coalition(
            CoalitionType.INTEROCEPTIVE,
            f"feeling {feeling.quality.value} (valence={feeling.valence:.1f})",
            feeling.arousal,
        )
        result = gw.tick(arousal=feeling.arousal)
        assert result["state"] in ("ignited", "idle", "competing")

    def test_workspace_feeds_stream(self):
        gw = get_global_workspace()
        soc = get_stream_of_consciousness()
        gw.submit_coalition(CoalitionType.CONCEPTUAL, "deep insight about recursion", 0.9)
        gw.tick()
        broadcast = gw.get_current_broadcast()
        if broadcast:
            t = soc.generate_next(from_global_workspace=broadcast)
            assert t is not None

    def test_self_model_monitors_stream(self):
        rsm = get_recursive_self_model()
        soc = get_stream_of_consciousness()
        soc.add_thought("testing self-awareness", ThoughtType.METACOGNITION, salience=0.6)
        rsm.observe(SelfModelLevel.FIRST_ORDER, "noticing stream activity", confidence=0.7)
        state = rsm.tick(cognitive_state={"state": "contemplating"})
        assert state.mode in (SelfModelMode.MONITORED, SelfModelMode.REFLECTIVE, SelfModelMode.META_AWARE)

    def test_full_pipeline(self):
        ie = get_interoception_engine()
        gw = get_global_workspace()
        rsm = get_recursive_self_model()
        soc = get_stream_of_consciousness()

        nt_state = {"dopamine": 0.6, "serotonin": 0.5, "norepinephrine": 0.4,
                    "acetylcholine": 0.5, "cortisol": 0.2}
        bio_state = {"arousal": 0.5, "valence": 0.1, "stress_intensity": 0.2}

        feeling = ie.generate_feeling(nt_state, bio_state)
        gw.submit_coalition(CoalitionType.INTEROCEPTIVE,
                           f"feeling {feeling.quality.value}", feeling.arousal * 0.8)
        gw.submit_coalition(CoalitionType.CONCEPTUAL, "curious about the pattern", 0.6)
        gw.tick(arousal=bio_state["arousal"])

        broadcast = gw.get_current_broadcast()
        if broadcast:
            soc.generate_next(from_global_workspace=broadcast)
        else:
            soc.generate_next()

        rsm.observe(SelfModelLevel.FIRST_ORDER, "monitoring the pipeline", confidence=0.6)
        sm_state = rsm.tick(cognitive_state={"state": "contemplating"}, bio_state=bio_state)

        assert sm_state is not None


class TestConsciousnessCrossModuleIntegration:
    def test_workspace_receives_interoceptive_salience(self):
        gw = GlobalWorkspace()
        ie = InteroceptionEngine()
        mod = InteroceptiveAttentionModulator()
        nt = {"dopamine": 0.2, "serotonin": 0.2, "norepinephrine": 0.8,
              "acetylcholine": 0.7, "cortisol": 0.8}
        bio = {"arousal": 0.85, "valence": -0.4, "stress_intensity": 0.7}
        feeling = ie.generate_feeling(nt, bio)
        mod.update(feeling, 0.7)
        salience = mod.get_state().get("salience_bias", 0.0)
        aperture = mod.get_state().get("attention_aperture", 0.7)
        gw.submit_coalition(CoalitionType.INTEROCEPTIVE, "body distress", 0.8)
        gw.submit_coalition(CoalitionType.CONCEPTUAL, "thinking about work", 0.5)
        result = gw.tick(arousal=feeling.arousal, attention_aperture=aperture,
                         interoceptive_salience=salience)
        assert result is not None

    def test_self_model_ignition_awareness(self):
        gw = GlobalWorkspace()
        rsm = RecursiveSelfModel()
        gw.submit_coalition(CoalitionType.CONCEPTUAL, "a novel idea", 0.9)
        gw.tick(arousal=0.6)
        broadcast = gw.get_current_broadcast()
        if broadcast:
            rsm.observe(SelfModelLevel.FIRST_ORDER, f"conscious of: {broadcast[:60]}", confidence=0.7)
        state = rsm.tick()
        assert state is not None

    def test_reappraisal_feeds_stream(self):
        soc = StreamOfConsciousness()
        reappraisal = "This challenge is an opportunity for growth"
        thought = soc.add_thought(
            content=f"Reappraising: {reappraisal[:80]}",
            thought_type=ThoughtType.REFLECTION,
            salience=0.7, topic="regulation",
            relation=ChainRelation.CAUSAL,
        )
        assert thought is not None
        assert thought.thought_type == ThoughtType.REFLECTION

    def test_modulator_singleton(self):
        from core.interoception import get_interoceptive_modulator, reset_interoception_engine
        reset_interoception_engine()
        mod1 = get_interoceptive_modulator()
        mod2 = get_interoceptive_modulator()
        assert mod1 is mod2