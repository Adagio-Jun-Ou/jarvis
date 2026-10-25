"""Bio-Integration Layer for Jarvis.

Connects all physiological systems into a unified whole-organism model:
- BioHomeostasis (autonomic regulation, neurotransmitter setpoints)
- CircadianRhythm (time-of-day performance modulation)
- NeuroplasticityEngine (synaptic adaptation)
- NeuroscienceCognitiveArchitecture (brain region simulation)
- EmotionalStateManager (valence/arousal tracking)
- PrefrontalCortex (concept prediction)
- AttentionPriorityQueue (saliency-based selection)

The integration layer ensures cross-module communication:
- Circadian modulates homeostatic neurotransmitter baselines
- Homeostasis feeds arousal/valence to emotional state
- Neuroplasticity adjusts connection weights based on usage
- All systems feed context into the agent's cognitive alignment
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

_bio_integration: Optional["BioIntegration"] = None
_bio_lock = threading.Lock()


@dataclass
class PhysiologicalState:
    timestamp: float = field(default_factory=time.time)
    arousal: float = 0.5
    valence: float = 0.0
    cognitive_capacity: float = 0.8
    creative_capacity: float = 0.5
    analytical_capacity: float = 0.7
    homeostatic_deviation: float = 0.0
    circadian_phase: str = "morning_peak"
    sleep_pressure: float = 0.2
    processing_fatigue: float = 0.2
    operational_phase: str = "peak_engagement"
    stress_phase: str = "recovery"
    stress_intensity: float = 0.0
    allostatic_load: float = 0.0
    autonomic_state: str = "balanced"
    vagal_tone: float = 0.7
    energy_available: float = 0.7
    energy_utilization: float = 0.3
    is_homeostatic: bool = True
    regulatory_actions: list = field(default_factory=list)
    learning_plasticity: float = 0.5
    connection_count: int = 0
    strong_pathways: int = 0
    recommendations: dict = field(default_factory=dict)
    neurotransmitters: dict = field(default_factory=dict)
    hormones: dict = field(default_factory=dict)


class BioIntegration:
    """Unified integration layer connecting all physiological systems.

    Provides:
    - Single tick() to advance all systems in coordination
    - Unified PhysiologicalState for agent context injection
    - Cross-module feedback loops (circadian → homeostasis, stress → plasticity)
    - Adaptive response style selection based on physiological state
    - Resource allocation decisions based on energy budget
    """

    def __init__(self):
        self._homeostasis = None
        self._circadian = None
        self._neuroplasticity = None
        self._neuroscience = None
        self._emotional_state = None
        self._prefrontal_cortex = None
        self._attention_queue = None

        self._state_history: deque[PhysiologicalState] = deque(maxlen=1440)
        self._tick_count = 0
        self._last_tick_time = time.time()
        self._lock = threading.RLock()

    def initialize(self, homeostasis=None, circadian=None, neuroplasticity=None,
                    neuroscience=None, emotional_state=None, prefrontal_cortex=None,
                    attention_queue=None) -> None:
        self._homeostasis = homeostasis
        self._circadian = circadian
        self._neuroplasticity = neuroplasticity
        self._neuroscience = neuroscience
        self._emotional_state = emotional_state
        self._prefrontal_cortex = prefrontal_cortex
        self._attention_queue = attention_queue
        logger.info("BioIntegration initialized with all physiological subsystems")

    def tick(self, dt: float = 1.0) -> PhysiologicalState:
        """Advance physiological time.

        dt is WALL-CLOCK ANCHORED: when real time has genuinely passed
        since the last tick (>= 0.1s — the production case: background
        ticks, prompt-build reads), dt is derived from elapsed time so
        physiology runs at 1x real speed no matter how often callers
        tick. For rapid successive calls (< 0.1s — tests, event-driven
        bursts) the explicit dt parameter is honored instead.
        Previously the parameter advanced time on EVERY call, and with
        multiple background tick sites plus per-read ticks the
        physiological clock ran at several times real speed.
        """
        with self._lock:
            self._tick_count += 1
            now = time.time()
            real_dt = min(max(now - self._last_tick_time, 0.0), 300.0)
            self._last_tick_time = now
            effective_dt = real_dt if real_dt >= 0.1 else dt

            if self._homeostasis:
                homeo_state = self._homeostasis.tick(effective_dt)
            else:
                homeo_state = self._get_default_homeostasis()

            if self._circadian:
                circ_state = self._circadian.tick(effective_dt / 60.0)
            else:
                circ_state = self._get_default_circadian()

            self._apply_cross_modulation(homeo_state, circ_state)

            plasticity_stats = {}
            if self._neuroplasticity:
                plasticity_stats = self._neuroplasticity.get_statistics()

            nt_state = homeo_state.get("neurotransmitters", {})
            nt_deviation = max(
                (abs(nt_state.get(k, 0.5) - v) for k, v in {
                    "dopamine": 0.5, "serotonin": 0.6, "norepinephrine": 0.4,
                    "acetylcholine": 0.5, "gaba": 0.6
                }.items() if k in nt_state
                ),
                default=0.0
            )

            state = PhysiologicalState(
                timestamp=now,
                arousal=homeo_state.get("arousal", 0.5),
                valence=homeo_state.get("valence", 0.0),
                cognitive_capacity=circ_state.get("cognitive_capacity", 0.8),
                creative_capacity=circ_state.get("creative_capacity", 0.5),
                analytical_capacity=circ_state.get("analytical_capacity", 0.7),
                homeostatic_deviation=round(nt_deviation, 4),
                circadian_phase=circ_state.get("circadian_phase", "morning_peak"),
                sleep_pressure=circ_state.get("sleep_pressure", 0.2),
                processing_fatigue=circ_state.get("processing_fatigue", circ_state.get("sleep_pressure", 0.2)),
                operational_phase=circ_state.get("operational_phase", "peak_engagement"),
                stress_phase=homeo_state.get("stress_phase", "recovery"),
                stress_intensity=homeo_state.get("stress_intensity", 0.0),
                allostatic_load=homeo_state.get("allostatic_load", 0.0),
                autonomic_state=homeo_state.get("autonomic", "balanced"),
                vagal_tone=homeo_state.get("vagal_tone", 0.7),
                energy_available=homeo_state.get("energy_available", 0.7),
                energy_utilization=homeo_state.get("energy_utilization", 0.3),
                is_homeostatic=homeo_state.get("is_homeostatic", True),
                regulatory_actions=homeo_state.get("regulatory_actions", []),
                learning_plasticity=plasticity_stats.get("critical_periods_active", 0) * 0.1 + 0.5,
                connection_count=plasticity_stats.get("total_synapses", 0),
                strong_pathways=plasticity_stats.get("strong_pathways", 0),
                recommendations=circ_state.get("recommendations", {}),
                neurotransmitters=homeo_state.get("neurotransmitters", {}),
                hormones=circ_state.get("hormones", {}),
            )

            self._state_history.append(state)
            return state

    def apply_stressor(self, intensity: float, source: str = "") -> PhysiologicalState:
        with self._lock:
            if self._homeostasis:
                self._homeostasis.apply_stressor(intensity, source)
            try:
                if self._neuroscience:
                    ns = self._neuroscience
                    if hasattr(ns, 'neurotransmitters'):
                        nt = ns.neurotransmitters
                        if 'norepinephrine' in nt:
                            nt['norepinephrine'].current = min(1.0, nt['norepinephrine'].current + intensity * 0.3)
            except Exception as e:
                logger.debug(f"Neuroscience stress modulation failed: {e}")
            return self.tick()

    def apply_reward(self, magnitude: float, neurotransmitter: str = "dopamine") -> PhysiologicalState:
        with self._lock:
            if self._homeostasis:
                self._homeostasis.apply_reward(magnitude, neurotransmitter)
            return self.tick()

    def consolidate(self, duration_factor: float = 1.0) -> dict:
        with self._lock:
            result = {}
            if self._neuroplasticity:
                result["plasticity"] = self._neuroplasticity.consolidate(duration_factor)
            if self._homeostasis:
                result["homeostasis"] = self._homeostasis.tick(duration_factor)
            result["circadian"] = self._circadian.tick(duration_factor / 60.0) if self._circadian else {}
            return result

    def activate_pathway(self, pre: str, post: str, strength: float = 1.0,
                          concept_type: str = "") -> float:
        with self._lock:
            if self._neuroplasticity:
                return self._neuroplasticity.activate(pre, post, strength, concept_type)
            return 0.5

    def _peek_state(self) -> PhysiologicalState:
        """Read the latest state WITHOUT advancing physiological time.

        Returns the cached state if it is fresh (<60s old); otherwise
        ticks once (which advances real elapsed time only) and returns
        that. Reads must never fast-forward the physiological clock.
        """
        with self._lock:
            if self._state_history:
                last = self._state_history[-1]
                if time.time() - last.timestamp < 60.0:
                    return last
            return self.tick()

    def get_cognitive_context(self) -> dict:
        state = self._peek_state()
        return {
            "physiological_state": {
                "arousal": state.arousal,
                "valence": state.valence,
                "cognitive_capacity": state.cognitive_capacity,
                "creative_capacity": state.creative_capacity,
                "analytical_capacity": state.analytical_capacity,
                "stress_phase": state.stress_phase,
                "stress_intensity": state.stress_intensity,
                "allostatic_load": state.allostatic_load,
                "autonomic_state": state.autonomic_state,
                "vagal_tone": state.vagal_tone,
                "circadian_phase": state.circadian_phase,
                "sleep_pressure": state.sleep_pressure,
                "processing_fatigue": state.processing_fatigue,
                "operational_phase": state.operational_phase,
                "is_homeostatic": state.is_homeostatic,
                "learning_plasticity": state.learning_plasticity,
                "connection_count": state.connection_count,
                "strong_pathways": state.strong_pathways,
            },
            "neurotransmitters": state.neurotransmitters,
            "hormones": state.hormones,
            "regulatory_actions": state.regulatory_actions,
            "recommendations": state.recommendations,
            "energy": {
                "available": state.energy_available,
                "utilization": state.energy_utilization,
            },
        }

    def get_response_style(self) -> dict:
        state = self._peek_state()
        style = {
            "detail_level": "moderate",
            "tone": "balanced",
            "energy_level": "normal",
            "adaptation_suggestions": [],
        }

        if state.cognitive_capacity > 0.8:
            style["detail_level"] = "deep"
            style["adaptation_suggestions"].append("favor_complex_reasoning")
        elif state.cognitive_capacity < 0.4:
            style["detail_level"] = "concise"
            style["adaptation_suggestions"].append("simplify_responses")
            style["energy_level"] = "low"

        if state.creative_capacity > state.analytical_capacity + 0.15:
            style["tone"] = "creative_exploratory"
            style["adaptation_suggestions"].append("encourage_exploration")
        elif state.analytical_capacity > state.creative_capacity + 0.15:
            style["tone"] = "analytical_precise"
            style["adaptation_suggestions"].append("favor_structured_reasoning")

        if state.stress_phase in ("alarm", "exhaustion"):
            style["tone"] = "calm_supportive"
            style["adaptation_suggestions"].append("reduce_stimulation")
            style["energy_level"] = "depleted" if state.stress_phase == "exhaustion" else "stressed"

        if state.processing_fatigue > 0.7:
            style["adaptation_suggestions"].append("suggest_consolidation_period")
            style["energy_level"] = "fatigued"
        elif state.sleep_pressure > 0.7:
            style["adaptation_suggestions"].append("suggest_rest_period")
            style["energy_level"] = "fatigued"

        if state.vagal_tone > 0.8:
            style["adaptation_suggestions"].append("high_emotional_resilience")
        elif state.vagal_tone < 0.3:
            style["adaptation_suggestions"].append("low_emotional_resilience_careful_approach")

        if state.learning_plasticity > 0.7:
            style["adaptation_suggestions"].append("optimal_for_learning_new_concepts")

        return style

    def _apply_cross_modulation(self, homeo_state: dict, circ_state: dict) -> None:
        if self._homeostasis and self._circadian:
            for nt_name, nt in self._homeostasis.neurotransmitters.items():
                circ_mod = circ_state.get("hormones", {}).get(nt_name, 0.5)
                nt.baseline = nt.baseline * 0.7 + circ_mod * 0.3

            if self._homeostasis.stress.phase.value in ("alarm", "exhaustion"):
                self._homeostasis.allocate_resource("attention", 0.1)
                self._homeostasis.release_resource("creative", 0.05)

        if self._neuroscience and self._homeostasis:
            try:
                self._neuroscience.sync_with_bio_homeostasis(self._homeostasis)
            except Exception as e:
                logger.debug(f"Neuroscience-bio sync failed: {e}")

        if self._emotional_state and self._homeostasis:
            try:
                arousal = self._homeostasis.get_arousal()
                valence = self._homeostasis.get_valence()
                self._emotional_state.update_from_physiology(arousal=arousal, valence=valence)
            except Exception as e:
                logger.debug(f"Emotional-bio sync failed: {e}")

    def _get_default_homeostasis(self) -> dict:
        return {
            "neurotransmitters": {"dopamine": 0.5, "serotonin": 0.6, "norepinephrine": 0.4,
                                  "acetylcholine": 0.5, "gaba": 0.6, "glutamate": 0.5},
            "arousal": 0.5, "valence": 0.0, "cognitive_capacity": 0.8,
            "stress_phase": "recovery", "stress_intensity": 0.0,
            "allostatic_load": 0.0, "autonomic": "balanced", "vagal_tone": 0.7,
            "energy_available": 0.7, "energy_utilization": 0.3,
            "is_homeostatic": True, "regulatory_actions": [],
        }

    def _get_default_circadian(self) -> dict:
        return {
            "cognitive_capacity": 0.8, "creative_capacity": 0.5,
            "analytical_capacity": 0.7, "circadian_phase": "morning_peak",
            "sleep_pressure": 0.2, "processing_fatigue": 0.2,
            "operational_phase": "peak_engagement",
            "hormones": {"melatonin": 0.3, "cortisol": 0.5,
            "dopamine": 0.5, "serotonin": 0.55},
            "recommendations": {},
        }


def get_bio_integration() -> BioIntegration:
    global _bio_integration
    with _bio_lock:
        if _bio_integration is None:
            _bio_integration = BioIntegration()
    return _bio_integration


def reset_bio_integration() -> None:
    global _bio_integration
    with _bio_lock:
        _bio_integration = BioIntegration()