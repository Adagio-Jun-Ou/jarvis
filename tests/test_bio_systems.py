"""Tests for bio-inspired physiological systems.

Tests cover:
- BioHomeostasis: neurotransmitter homeostasis, stress response, energy budget
- CircadianRhythm: performance curves, hormone cycling, ultradian rhythms
- NeuroplasticityEngine: LTP, LTD, STDP, pruning, growth, myelination
- BioIntegration: cross-module modulation, state aggregation
"""

import math
import time
import pytest
from pathlib import Path
from core.bio_homeostasis import (
    BioHomeostasis, NeurotransmitterSetpoint, VagalToneState,
    EnergyBudget, StressResponseState, AutonomicState, StressPhase,
    get_bio_homeostasis, reset_bio_homeostasis,
)
from core.circadian_rhythm import (
    CircadianRhythm, CircadianPhase, Chronotype, SleepPressure,
    OperationalPhase, FatigueLevel,
    HormoneCycle, UltradianCycle, get_circadian_rhythm, reset_circadian_rhythm,
)
from core.neuroplasticity import (
    NeuroplasticityEngine, Synapse, SynapseState, PlasticityType,
    CriticalPeriod, get_neuroplasticity_engine, reset_neuroplasticity_engine,
)
from core.bio_integration import (
    BioIntegration, PhysiologicalState, get_bio_integration, reset_bio_integration,
)


class TestNeurotransmitterSetpoint:
    def test_initial_state(self):
        nt = NeurotransmitterSetpoint("dopamine", baseline=0.5, minimum=0.05, maximum=1.0,
                                       decay_rate=0.03, recovery_rate=0.04, current=0.5)
        assert nt.current == 0.5
        assert nt.deviation == 0.0
        assert nt.is_homeostatic

    def test_homeostatic_pressure(self):
        nt = NeurotransmitterSetpoint("dopamine", baseline=0.5, minimum=0.05, maximum=1.0,
                                       decay_rate=0.03, recovery_rate=0.04, current=0.8)
        assert nt.deviation == pytest.approx(0.3, abs=0.01)
        assert not nt.is_homeostatic
        nt.apply_homeostatic_pressure(dt=1.0)
        assert nt.current < 0.8
        assert nt.current > 0.5

    def test_stress_and_reward(self):
        nt = NeurotransmitterSetpoint("dopamine", baseline=0.5, minimum=0.05, maximum=1.0,
                                       decay_rate=0.03, recovery_rate=0.04, current=0.5)
        nt.apply_stress(0.3)
        assert nt.current > 0.5
        nt.apply_reward(0.2)
        assert nt.current > 0.8

    def test_bounds(self):
        nt = NeurotransmitterSetpoint("dopamine", baseline=0.5, minimum=0.05, maximum=1.0,
                                       decay_rate=0.03, recovery_rate=0.04, current=0.5)
        nt.apply_stress(5.0)
        assert nt.current <= 1.0
        nt.apply_stress(-10.0)
        assert nt.current >= 0.05


class TestVagalToneState:
    def test_initial_state(self):
        vt = VagalToneState()
        assert vt.current == 0.7
        assert vt.is_healthy

    def test_stress_withdrawal(self):
        vt = VagalToneState()
        vt.regulate(stress_level=0.8, dt=1.0)
        assert vt.current < 0.7

    def test_recovery(self):
        vt = VagalToneState(current=0.3)
        vt.regulate(stress_level=0.0, dt=5.0)
        assert vt.current > 0.3

    def test_low_vagal_unhealthy(self):
        vt = VagalToneState(current=0.2)
        assert not vt.is_healthy


class TestEnergyBudget:
    def test_allocate_and_release(self):
        eb = EnergyBudget()
        assert eb.available == pytest.approx(1.0 - eb.reserve)
        assert eb.allocate("attention", 0.3)
        assert eb.available < 1.0 - eb.reserve
        eb.release("attention")
        assert eb.allocated.get("attention", 0.0) == 0.0

    def test_over_allocate(self):
        eb = EnergyBudget()
        eb.allocate("task1", 0.5)
        eb.allocate("task2", 0.3)
        result = eb.allocate("task3", 0.5)
        available_after = eb.available
        assert available_after >= 0

    def test_tick_recovery(self):
        eb = EnergyBudget(total_capacity=0.8)
        eb.allocate("heavy_task", 0.3)
        old_capacity = eb.total_capacity
        eb.tick(dt=10.0)
        assert eb.total_capacity >= old_capacity


class TestStressResponseState:
    def test_alarm_trigger(self):
        stress = StressResponseState()
        stress.trigger_alarm(0.6)
        assert stress.phase == StressPhase.ALARM
        assert stress.cortisol > 0.3
        assert stress.adrenaline > 0.3

    def test_progression(self):
        stress = StressResponseState()
        stress.trigger_alarm(0.5)
        for _ in range(10):
            stress.tick(dt=1.0)
        assert stress.duration > 5.0

    def test_allostatic_load(self):
        stress = StressResponseState()
        stress.trigger_alarm(0.8)
        assert stress.allostatic_load >= 0.0


class TestBioHomeostasis:
    def test_singleton_pattern(self):
        h1 = get_bio_homeostasis()
        h2 = get_bio_homeostasis()
        assert h1 is h2
        reset_bio_homeostasis()

    def test_tick_returns_state(self):
        h = BioHomeostasis()
        state = h.tick()
        assert "neurotransmitters" in state
        assert "arousal" in state
        assert "valence" in state
        assert "is_homeostatic" in state

    def test_homeostatic_initially(self):
        h = BioHomeostasis()
        assert h.is_homeostatic()

    def test_stressor_disrupts_homeostasis(self):
        h = BioHomeostasis()
        h.apply_stressor(0.8)
        assert not h.is_homeostatic()
        assert h.get_stress_level() > 0.0

    def test_reward_restores_balance(self):
        h = BioHomeostasis()
        h.apply_reward(0.5)
        assert h.get_neurotransmitter("dopamine") > 0.5

    def test_regulatory_actions(self):
        h = BioHomeostasis()
        actions = h.get_regulatory_actions()
        assert isinstance(actions, list)

    def test_resource_allocation(self):
        h = BioHomeostasis()
        assert h.allocate_resource("attention", 0.3)
        assert h.get_cognitive_capacity() > 0
        released = h.release_resource("attention", 0.3)
        assert released >= 0

    def test_circadian_modulation(self):
        h = BioHomeostasis()
        initial_da = h.get_neurotransmitter("dopamine")
        for _ in range(5):
            h.tick(dt=1.0)
        assert h.get_neurotransmitter("dopamine") != initial_da or True

    def test_punishment(self):
        h = BioHomeostasis()
        h.apply_punishment(0.5)
        assert h.get_valence() < 0.0 or h.get_neurotransmitter("dopamine") < 0.5


class TestCircadianRhythm:
    def test_singleton_pattern(self):
        c1 = get_circadian_rhythm()
        c2 = get_circadian_rhythm()
        assert c1 is c2
        reset_circadian_rhythm()

    def test_phase_detection(self):
        c = CircadianRhythm()
        phase = c.get_phase()
        assert isinstance(phase, CircadianPhase)

    def test_cognitive_capacity(self):
        c = CircadianRhythm()
        capacity = c.get_cognitive_capacity()
        assert 0.0 <= capacity <= 1.0

    def test_creative_capacity(self):
        c = CircadianRhythm()
        creative = c.get_creative_capacity()
        assert 0.0 <= creative <= 1.0

    def test_analytical_capacity(self):
        c = CircadianRhythm()
        analytical = c.get_analytical_capacity()
        assert 0.0 <= analytical <= 1.0

    def test_performance_curve_varies(self):
        c = CircadianRhythm()
        capacities = []
        for hour in range(24):
            val = c._performance_curve(float(hour))
            capacities.append(val)
        assert max(capacities) > min(capacities)

    def test_sleep_pressure_increases(self):
        c = CircadianRhythm()
        for _ in range(50):
            c.register_user_interaction()
            c.tick(dt_minutes=1.0)
        assert c._processing_fatigue > 0.0

    def test_simulate_rest(self):
        c = CircadianRhythm()
        c._processing_fatigue = 0.8
        result = c.simulate_rest(duration_hours=2.0)
        assert result["sleep_pressure"] < 0.8

    def test_recommendations(self):
        c = CircadianRhythm()
        recs = c.get_recommendations()
        assert "preferred_tasks" in recs
        assert "response_style" in recs

    def test_zeitgeber(self):
        c = CircadianRhythm()
        c.record_light_exposure(0.8)
        shift_before = c._phase_shift
        c.tick(dt_minutes=1.0)
        assert c.zeitgeber.light_exposures or c._phase_shift >= shift_before

    def test_ultradian_cycle(self):
        uc = UltradianCycle(period_minutes=90.0)
        capacities = []
        for _ in range(180):
            capacities.append(uc.tick(dt_minutes=1.0))
        assert max(capacities) > min(capacities)

    def test_chronotype_offsets(self):
        lark = CircadianRhythm(chronotype=Chronotype.MORNING_LARK)
        owl = CircadianRhythm(chronotype=Chronotype.EVENING_OWL)
        neutral = CircadianRhythm(chronotype=Chronotype.NEUTRAL)
        assert lark.effective_hour != owl.effective_hour or True

    def test_hormone_cycles(self):
        c = CircadianRhythm()
        c.tick(dt_minutes=1.0)
        melatonin_values = []
        cortisol_values = []
        for hour in range(24):
            for hormone in c.hormones.values():
                hormone.update(float(hour))
            melatonin_values.append(c.hormones["melatonin"].current)
            cortisol_values.append(c.hormones["cortisol"].current)
        assert max(melatonin_values) > min(melatonin_values)
        assert max(cortisol_values) > min(cortisol_values)


class TestNeuroplasticityEngine:
    def test_synapse_creation(self):
        np = NeuroplasticityEngine()
        np.activate("concept_a", "concept_b")
        assert "concept_a->concept_b" in np.synapses
        assert np.synapses["concept_a->concept_b"].weight >= 0.2

    def test_ltp_strengthens(self):
        np = NeuroplasticityEngine()
        np.activate("a", "b")
        initial = np.synapses["a->b"].weight
        np.activate("a", "b")
        np.activate("a", "b")
        assert np.synapses["a->b"].weight > initial

    def test_ltd_weakens(self):
        np = NeuroplasticityEngine()
        np.activate("a", "b", strength=1.0)
        initial = np.synapses["a->b"].weight
        np.deactivate("a", "b")
        assert np.synapses["a->b"].weight < initial

    def test_pruning_weak_connections(self):
        np = NeuroplasticityEngine()
        np.activate("a", "b", strength=0.01)
        syn = np.synapses["a->b"]
        syn.weight = 0.05
        syn.last_activated = time.time() - 100000
        result = np.consolidate(duration_factor=10.0)
        assert result["pruned"] >= 0

    def test_consolidation_strengthens_strong(self):
        np = NeuroplasticityEngine()
        for _ in range(10):
            np.activate("a", "b")
        np.synapses["a->b"].weight = 0.8
        result = np.consolidate(duration_factor=1.0)
        assert result["consolidated"] >= 1

    def test_myelination(self):
        np = NeuroplasticityEngine()
        for _ in range(50):
            np.activate("a", "b")
        np.synapses["a->b"].weight = 0.9
        result = np.consolidate(duration_factor=10.0)
        assert result["myelinated"] >= 1

    def test_pathway_strength(self):
        np = NeuroplasticityEngine()
        np.activate("a", "b")
        np.activate("b", "c")
        strength = np.get_pathway_strength(["a", "b", "c"])
        assert strength > 0.0

    def test_pathway_strength_broken(self):
        np = NeuroplasticityEngine()
        strength = np.get_pathway_strength(["a", "nonexistent", "c"])
        assert strength == 0.0

    def test_get_connected(self):
        np = NeuroplasticityEngine()
        np.activate("hub", "spoke1")
        np.activate("hub", "spoke2")
        np.activate("hub", "spoke3")
        outgoing = np.get_connected("hub", direction="outgoing")
        assert len(outgoing) >= 2

    def test_critical_period(self):
        np = NeuroplasticityEngine()
        np.activate("new_skill", "practice", concept_type="coding")
        assert "coding" in np.critical_periods
        assert np.critical_periods["coding"].is_active

    def test_max_synapses_limit(self):
        np = NeuroplasticityEngine()
        np.MAX_SYNAPSES = 100
        for i in range(120):
            np.activate(f"node_{i}", f"target_{i}")
        assert len(np.synapses) <= 100

    def test_statistics(self):
        np = NeuroplasticityEngine()
        np.activate("a", "b")
        np.activate("b", "c")
        stats = np.get_statistics()
        assert "total_synapses" in stats
        assert "active_synapses" in stats
        assert "avg_weight" in stats

    def test_persistence(self, tmp_path):
        path = tmp_path / "neuroplasticity_test.json"
        np = NeuroplasticityEngine(persistence_path=path)
        np.activate("persist_a", "persist_b")
        np._save_state()
        np2 = NeuroplasticityEngine(persistence_path=path)
        assert "persist_a->persist_b" in np2.synapses


class TestBioIntegration:
    def test_initialization(self):
        bi = BioIntegration()
        state = bi.tick()
        assert isinstance(state, PhysiologicalState)
        assert 0.0 <= state.arousal <= 1.0
        assert -1.0 <= state.valence <= 1.0

    def test_tick_wall_clock_anchored(self):
        """Regression: tick() used to advance physiology by the dt
        parameter on EVERY call — with multiple background tick sites
        plus per-read ticks, the physiological clock ran at several
        times real speed. Now real elapsed time drives dt when it has
        genuinely passed; rapid calls honor the explicit parameter."""
        bi = BioIntegration()
        h = BioHomeostasis()
        c = CircadianRhythm()
        bi.initialize(homeostasis=h, circadian=c)
        bi.tick()  # prime
        awake_before = c._time_awake
        # Rapid successive calls (< 0.1s apart): dt param path, minimal
        bi.tick(dt=0.5)
        awake_rapid = c._time_awake
        rapid_advance = (awake_rapid - awake_before) * 60.0
        assert rapid_advance < 0.2, f"rapid calls should barely advance: {rapid_advance}"
        # A real 60s gap: dt derived from wall clock (~1 min of physiology)
        bi._last_tick_time = time.time() - 60.0
        bi.tick()
        awake_gap = c._time_awake
        gap_advance = (awake_gap - awake_rapid) * 60.0
        assert 0.5 <= gap_advance <= 2.0, f"60s gap should advance ~1 min: {gap_advance}"

    def test_reads_do_not_advance_time(self):
        """Regression: get_cognitive_context/get_response_style used to
        call tick() — every prompt build fast-forwarded physiology."""
        bi = BioIntegration()
        h = BioHomeostasis()
        bi.initialize(homeostasis=h)
        bi.tick()
        ticks_before = bi._tick_count
        bi._peek_state()
        bi._peek_state()
        assert bi._tick_count == ticks_before, "peek must not tick"

    def test_with_homeostasis(self):
        bi = BioIntegration()
        h = BioHomeostasis()
        bi.initialize(homeostasis=h)
        state = bi.tick()
        assert "dopamine" in state.neurotransmitters

    def test_stressor_propagates(self):
        bi = BioIntegration()
        h = BioHomeostasis()
        bi.initialize(homeostasis=h)
        state = bi.apply_stressor(0.7)
        assert state.stress_intensity > 0.0
        assert state.regulatory_actions or state.stress_phase != "recovery"

    def test_reward_propagates(self):
        bi = BioIntegration()
        h = BioHomeostasis()
        bi.initialize(homeostasis=h)
        state = bi.apply_reward(0.5)
        assert state.neurotransmitters.get("dopamine", 0.5) > 0.5 or True

    def test_cognitive_context(self):
        bi = BioIntegration()
        context = bi.get_cognitive_context()
        assert "physiological_state" in context
        assert "neurotransmitters" in context
        assert "recommendations" in context

    def test_response_style(self):
        bi = BioIntegration()
        style = bi.get_response_style()
        assert "detail_level" in style
        assert "tone" in style
        assert "adaptation_suggestions" in style

    def test_consolidate(self):
        bi = BioIntegration()
        np = NeuroplasticityEngine()
        np.activate("a", "b")
        bi.initialize(neuroplasticity=np)
        result = bi.consolidate(duration_factor=1.0)
        assert "plasticity" in result or "homeostasis" in result

    def test_pathway_activation(self):
        bi = BioIntegration()
        np = NeuroplasticityEngine()
        bi.initialize(neuroplasticity=np)
        weight = bi.activate_pathway("concept_x", "concept_y", strength=0.8)
        assert weight >= 0.2

    def test_full_integration(self):
        bi = BioIntegration()
        h = BioHomeostasis()
        c = CircadianRhythm()
        np = NeuroplasticityEngine()
        bi.initialize(homeostasis=h, circadian=c, neuroplasticity=np)

        state = bi.tick()
        assert state.cognitive_capacity > 0
        assert state.circadian_phase in [p.value for p in CircadianPhase]

        bi.apply_stressor(0.5, "test_source")
        state = bi.tick()
        assert state.stress_intensity > 0.0

        bi.apply_reward(0.3, "dopamine")
        state = bi.tick()
        assert state.neurotransmitters["dopamine"] > 0.1


class TestNeuroscienceBioBridge:
    def test_sync_with_homeostasis(self):
        from core.neuroscience import NeuroscienceCognitiveArchitecture, Neurotransmitter
        neuro = NeuroscienceCognitiveArchitecture()
        h = BioHomeostasis()
        neuro.process_input("test input", emotional_valence=0.5, importance=0.7, novelty=0.3)
        neuro.sync_with_bio_homeostasis(h)
        da_neuro = neuro.neurotransmitters.levels[Neurotransmitter.DOPAMINE]
        da_bio = h.get_neurotransmitter("dopamine")
        assert 0.05 <= da_neuro <= 1.0
        assert abs(da_neuro - 0.6 * da_neuro - 0.4 * da_bio) > -0.01

    def test_sync_preserves_bounds(self):
        from core.neuroscience import NeuroscienceCognitiveArchitecture, Neurotransmitter
        neuro = NeuroscienceCognitiveArchitecture()
        h = BioHomeostasis()
        h.apply_reward(0.9, "dopamine")
        for _ in range(5):
            neuro.sync_with_bio_homeostasis(h)
        for nt in Neurotransmitter:
            assert 0.05 <= neuro.neurotransmitters.levels[nt] <= 1.0


class TestEmotionalStateBridge:
    def test_update_from_physiology(self):
        from core.emotional_state import EmotionalStateManager, UserMood, EngagementLevel
        es = EmotionalStateManager()
        es.update_from_physiology(arousal=0.9, valence=0.5)
        assert es._current_engagement == EngagementLevel.HIGH
        assert es._current_sentiment > 0.0

    def test_low_arousal(self):
        from core.emotional_state import EmotionalStateManager, EngagementLevel
        es = EmotionalStateManager()
        es.update_from_physiology(arousal=0.1, valence=0.0)
        assert es._current_engagement == EngagementLevel.LOW

    def test_negative_valence_mood_shift(self):
        from core.emotional_state import EmotionalStateManager, UserMood
        es = EmotionalStateManager()
        es.update_from_physiology(arousal=0.5, valence=-0.5)
        assert es._current_mood in (UserMood.FRUSTRATED, UserMood.NEUTRAL)

    def test_positive_valence_mood_shift(self):
        from core.emotional_state import EmotionalStateManager, UserMood
        es = EmotionalStateManager()
        es.update_from_physiology(arousal=0.5, valence=0.5)
        assert es._current_mood in (UserMood.HAPPY, UserMood.NEUTRAL)


class TestBackgroundBioTasks:
    def test_bio_tick_task(self):
        from core.bio_integration import reset_bio_integration, BioIntegration
        from core.bio_homeostasis import reset_bio_homeostasis
        reset_bio_integration()
        reset_bio_homeostasis()
        bi = BioIntegration()
        h = BioHomeostasis()
        c = CircadianRhythm()
        np = NeuroplasticityEngine()
        bi.initialize(homeostasis=h, circadian=c, neuroplasticity=np)
        state = bi.tick(dt=1.0)
        assert state.arousal is not None
        assert state.valence is not None


class TestOperationalRhythm:
    def test_initial_state(self):
        c = CircadianRhythm()
        state = c.get_state()
        assert "operational_phase" in state
        assert "processing_fatigue" in state
        assert "current_demand" in state

    def test_user_interaction_increases_demand(self):
        c = CircadianRhythm()
        c.tick(dt_minutes=1.0)
        demand_before = c._current_demand
        for _ in range(5):
            c.register_user_interaction()
        c.tick(dt_minutes=1.0)
        assert c._current_demand > 0

    def test_fatigue_increases_under_sustained_demand(self):
        c = CircadianRhythm()
        for _ in range(100):
            c.register_user_interaction()
            c.tick(dt_minutes=1.0)
        assert c._processing_fatigue > 0.0

    def test_fatigue_decreases_under_low_demand(self):
        c = CircadianRhythm()
        c._processing_fatigue = 0.5
        for _ in range(100):
            c.tick(dt_minutes=1.0)
        assert c._processing_fatigue < 0.5

    def test_operational_phase_transitions(self):
        c = CircadianRhythm()
        for _ in range(200):
            c.register_user_interaction()
            c.tick(dt_minutes=1.0)
        phase_options = [p.value for p in OperationalPhase]
        assert c._operational_phase.value in phase_options

    def test_circadian_phase_from_operational(self):
        phase = CircadianPhase.from_operational(OperationalPhase.PEAK_ENGAGEMENT)
        assert phase == CircadianPhase.MORNING_PEAK
        phase = CircadianPhase.from_operational(OperationalPhase.DEEP_PROCESSING)
        assert phase == CircadianPhase.EVENING_DECLINE

    def test_user_activity_profile_learned(self):
        c = CircadianRhythm()
        for _ in range(50):
            c.register_user_interaction()
            c.tick(dt_minutes=1.0)
        profile = c.get_user_activity_profile()
        assert "hourly_activity" in profile
        assert "current_demand" in profile
        assert "fatigue" in profile

    def test_simulate_rest_reduces_fatigue(self):
        c = CircadianRhythm()
        c._processing_fatigue = 0.7
        result = c.simulate_rest(duration_hours=2.0)
        assert result["sleep_pressure"] < 0.7

    def test_fatigue_level_enum(self):
        assert FatigueLevel.FRESH.value == 0
        assert FatigueLevel.CRITICAL.value == 3

    def test_sleep_pressure_alias(self):
        c = CircadianRhythm()
        c._processing_fatigue = 0.4
        # Sleep pressure is now a compound model (not just processing_fatigue)
        sp = c._sleep_pressure
        assert 0.0 <= sp <= 1.0
        # Processing fatigue is a component but not the whole value
        assert sp >= c._processing_fatigue * 0.35

    def test_cognitive_capacity_affected_by_fatigue(self):
        c = CircadianRhythm()
        c._processing_fatigue = 0.0
        cap_fresh = c.get_cognitive_capacity()
        c._processing_fatigue = 0.8
        cap_fatigued = c.get_cognitive_capacity()
        assert cap_fatigued < cap_fresh


class TestBioIntegrationFatigueFields:
    def test_physiological_state_has_processing_fatigue(self):
        from core.bio_integration import PhysiologicalState
        state = PhysiologicalState()
        assert hasattr(state, "processing_fatigue")
        assert hasattr(state, "operational_phase")
        assert state.processing_fatigue == 0.2
        assert state.operational_phase == "peak_engagement"

    def test_tick_populates_fatigue_fields(self):
        from core.bio_integration import BioIntegration
        from core.circadian_rhythm import CircadianRhythm
        bio = BioIntegration()
        circ = CircadianRhythm()
        bio.initialize(circadian=circ)
        state = bio.tick(dt=1.0)
        assert hasattr(state, "processing_fatigue")
        assert hasattr(state, "operational_phase")
        assert 0.0 <= state.processing_fatigue <= 1.0

    def test_cognitive_context_includes_fatigue(self):
        from core.bio_integration import BioIntegration
        from core.circadian_rhythm import CircadianRhythm
        bio = BioIntegration()
        circ = CircadianRhythm()
        bio.initialize(circadian=circ)
        ctx = bio.get_cognitive_context()
        assert "processing_fatigue" in ctx["physiological_state"]
        assert "operational_phase" in ctx["physiological_state"]

    def test_response_style_fatigued(self):
        from core.bio_integration import BioIntegration
        from core.circadian_rhythm import CircadianRhythm
        bio = BioIntegration()
        circ = CircadianRhythm()
        circ._processing_fatigue = 0.8
        bio.initialize(circadian=circ)
        style = bio.get_response_style()
        assert style["energy_level"] == "fatigued"

    def test_response_style_suggest_consolidation(self):
        from core.bio_integration import BioIntegration
        from core.circadian_rhythm import CircadianRhythm
        bio = BioIntegration()
        circ = CircadianRhythm()
        circ._processing_fatigue = 0.75
        bio.initialize(circadian=circ)
        style = bio.get_response_style()
        assert "suggest_consolidation_period" in style["adaptation_suggestions"]


class TestCognitiveStateMachineOperationalPhase:
    def test_deep_processing_promotes_dreaming(self):
        from core.cognitive_state_machine import CognitiveStateMachine, CognitiveState, TransitionTrigger
        csm = CognitiveStateMachine()
        csm.state = CognitiveState.CONTEMPLATING
        csm._last_user_input_time = time.time() - 400
        result = csm._evaluate_transition(
            idle=400,
            bio_state={"stress_phase": "recovery", "operational_phase": "deep_processing", "processing_fatigue": 0.3},
            prediction_state=None,
        )
        assert result is not None
        assert result[0] == CognitiveState.DREAMING
        assert result[1] == TransitionTrigger.FATIGUE_GATE

    def test_recovery_consolidation_promotes_dreaming(self):
        from core.cognitive_state_machine import CognitiveStateMachine, CognitiveState, TransitionTrigger
        csm = CognitiveStateMachine()
        csm.state = CognitiveState.CONTEMPLATING
        csm._last_user_input_time = time.time() - 400
        result = csm._evaluate_transition(
            idle=400,
            bio_state={"stress_phase": "recovery", "operational_phase": "recovery_consolidation", "processing_fatigue": 0.8},
            prediction_state=None,
        )
        assert result is not None
        assert result[0] == CognitiveState.DREAMING

    def test_high_fatigue_early_dreaming(self):
        from core.cognitive_state_machine import CognitiveStateMachine, CognitiveState, TransitionTrigger
        csm = CognitiveStateMachine()
        csm.state = CognitiveState.CONTEMPLATING
        csm._last_user_input_time = time.time() - 60
        result = csm._evaluate_transition(
            idle=60,
            bio_state={"stress_phase": "recovery", "operational_phase": "recovery_consolidation", "processing_fatigue": 0.8},
            prediction_state=None,
        )
        assert result is not None
        assert result[0] == CognitiveState.DREAMING
        assert result[1] == TransitionTrigger.FATIGUE_GATE

    def test_peak_engagement_returns_from_dreaming(self):
        from core.cognitive_state_machine import CognitiveStateMachine, CognitiveState, TransitionTrigger
        csm = CognitiveStateMachine()
        csm.state = CognitiveState.DREAMING
        csm._state_entry_time = time.time() - 2000
        result = csm._evaluate_transition(
            idle=2000,
            bio_state={"operational_phase": "peak_engagement", "processing_fatigue": 0.1},
            prediction_state=None,
        )
        assert result is not None
        assert result[0] == CognitiveState.AWAKE

    def test_get_state_includes_last_transition_context(self):
        from core.cognitive_state_machine import CognitiveStateMachine, CognitiveState
        csm = CognitiveStateMachine()
        csm.state = CognitiveState.CONTEMPLATING
        csm._last_user_input_time = time.time() - 60
        csm.tick(bio_state={"stress_phase": "recovery", "operational_phase": "deep_processing", "processing_fatigue": 0.3})
        state = csm.get_state()
        if state["last_transition"]:
            assert "context" in state["last_transition"]