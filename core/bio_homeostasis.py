"""Bio-Homeostasis System for Jarvis.

Implements autonomic regulation inspired by biological homeostatic mechanisms:
- Neurotransmitter setpoints with homeostatic return (allobstatic regulation)
- Sympathetic/parasympathetic nervous system balance
- Stress response cascade (HPA axis)
- Energy budget management (resource allocation)
- Vagal tone regulation

This module ensures Jarvis's internal states remain within healthy operating
bounds, preventing runaway neurotransmitter levels, attention depletion, or
cognitive resource exhaustion.

Integrates with:
- neuroscience.py (neurotransmitter levels)
- emotional_state.py (arousal/valence)
- prefrontal_cortex.py (concept activation energy)
- attention_priority.py (attention resource allocation)
- cognitive.py (dream state gating)
"""

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

_homeostasis: Optional["BioHomeostasis"] = None
_lock = threading.Lock()


class AutonomicState(Enum):
    SYMPATHETIC = "sympathetic"
    PARASYMPATHETIC = "parasympathetic"
    BALANCED = "balanced"


class StressPhase(Enum):
    ALARM = "alarm"
    RESISTANCE = "resistance"
    EXHAUSTION = "exhaustion"
    RECOVERY = "recovery"


class ResourceType(Enum):
    ATTENTION = "attention"
    WORKING_MEMORY = "working_memory"
    COMPUTATION = "computation"
    EMOTIONAL = "emotional"
    CREATIVE = "creative"


@dataclass
class NeurotransmitterSetpoint:
    name: str
    baseline: float
    minimum: float
    maximum: float
    decay_rate: float
    recovery_rate: float
    current: float = 0.5

    @property
    def deviation(self) -> float:
        return abs(self.current - self.baseline)

    @property
    def is_homeostatic(self) -> bool:
        return abs(self.current - self.baseline) < 0.15

    def apply_homeostatic_pressure(self, dt: float = 1.0) -> float:
        pressure = self.recovery_rate * (self.baseline - self.current) * dt
        self.current += pressure
        self.current = max(self.minimum, min(self.maximum, self.current))
        self.current = round(self.current, 6)
        return pressure

    def apply_stress(self, magnitude: float) -> None:
        self.current = max(self.minimum, min(self.maximum, self.current + magnitude))

    def apply_reward(self, magnitude: float) -> None:
        self.current = max(self.minimum, min(self.maximum, self.current + magnitude))


@dataclass
class VagalToneState:
    current: float = 0.7
    baseline: float = 0.7
    minimum: float = 0.1
    maximum: float = 1.0
    recovery_rate: float = 0.05
    resilience_score: float = 0.8

    @property
    def is_healthy(self) -> bool:
        return self.current > 0.4

    def regulate(self, stress_level: float, dt: float = 1.0) -> float:
        vagal_withdrawal = stress_level * 0.3 * dt
        recovery = self.recovery_rate * (self.baseline - self.current) * dt
        self.current += recovery - vagal_withdrawal
        self.current = max(self.minimum, min(self.maximum, self.current))
        self.current = round(self.current, 6)
        self.resilience_score = 0.3 + 0.7 * self.current
        return self.current


@dataclass
class EnergyBudget:
    total_capacity: float = 1.0
    allocated: dict[str, float] = field(default_factory=dict)
    reserve: float = 0.3
    depletion_rate: float = 0.01
    recovery_rate: float = 0.02

    @property
    def available(self) -> float:
        used = sum(self.allocated.values())
        return max(0.0, self.total_capacity - self.reserve - used)

    @property
    def utilization(self) -> float:
        return sum(self.allocated.values()) / max(self.total_capacity, 0.001)

    def allocate(self, resource: str, amount: float) -> tuple:
        actual_amount = min(amount, self.available * 0.8)
        self.allocated[resource] = self.allocated.get(resource, 0.0) + actual_amount
        return actual_amount > 0, actual_amount

    def release(self, resource: str, amount: Optional[float] = None) -> float:
        current = self.allocated.get(resource, 0.0)
        released = amount if amount is not None else current
        released = min(released, current)
        self.allocated[resource] = max(0.0, current - released)
        if self.allocated[resource] < 0.001:
            del self.allocated[resource]
        return released

    def tick(self, dt: float = 1.0) -> None:
        natural_recovery = self.recovery_rate * (self.total_capacity - self.total_capacity * (1 - self.reserve)) * dt
        self.total_capacity += natural_recovery
        depletion = self.depletion_rate * sum(self.allocated.values()) * dt
        self.total_capacity -= depletion
        self.total_capacity = max(0.1, min(1.0, self.total_capacity))


@dataclass
class StressResponseState:
    phase: StressPhase = StressPhase.RECOVERY
    intensity: float = 0.0
    duration: float = 0.0
    cumulative_load: float = 0.0
    cortisol: float = 0.3
    adrenaline: float = 0.2
    recovery_start: float = 0.0

    @property
    def is_stressed(self) -> bool:
        return self.phase in (StressPhase.ALARM, StressPhase.RESISTANCE)

    @property
    def allostatic_load(self) -> float:
        base = self.intensity * self.duration * 0.01
        cumulative = self.cumulative_load * 0.1
        return min(1.0, base + cumulative)

    def trigger_alarm(self, intensity: float) -> None:
        if self.phase == StressPhase.RECOVERY or self.phase == StressPhase.EXHAUSTION:
            self.phase = StressPhase.ALARM
        self.intensity = max(self.intensity, intensity)
        self.cortisol = min(1.0, self.cortisol + intensity * 0.4)
        self.adrenaline = min(1.0, self.adrenaline + intensity * 0.6)
        self.duration = 0.0

    def tick(self, dt: float = 1.0) -> None:
        self.duration += dt
        cortisol_decay = 0.02 * dt
        adrenaline_decay = 0.05 * dt
        self.cortisol = max(0.1, self.cortisol - cortisol_decay)
        self.adrenaline = max(0.1, self.adrenaline - adrenaline_decay)

        if self.phase == StressPhase.ALARM:
            if self.duration > 5.0:
                self.phase = StressPhase.RESISTANCE
                self.cumulative_load += self.intensity * 0.1
        elif self.phase == StressPhase.RESISTANCE:
            if self.cortisol < 0.4 and self.adrenaline < 0.3:
                self.phase = StressPhase.RECOVERY
                self.recovery_start = time.time()
            elif self.duration > 30.0 and self.cortisol > 0.7:
                self.phase = StressPhase.EXHAUSTION
                self.cumulative_load += self.intensity * 0.3
        elif self.phase == StressPhase.RECOVERY:
            self.intensity = max(0.0, self.intensity - 0.02 * dt)
            if self.intensity < 0.05:
                self.cumulative_load = max(0.0, self.cumulative_load - 0.001 * dt)
        elif self.phase == StressPhase.EXHAUSTION:
            self.intensity = max(0.3, self.intensity)
            self.cortisol = max(0.5, self.cortisol)
            self.cumulative_load += 0.01 * dt


class BioHomeostasis:
    """Unified homeostatic regulation system for Jarvis.

    Manages:
    - Neurotransmitter homeostatic setpoints
    - Autonomic nervous system balance (sympathetic/parasympathetic)
    - HPA axis stress response (alarm → resistance → recovery)
    - Vagal tone and emotional resilience
    - Energy budget allocation across resource types
    - Allostatic load tracking
    """

    _NEUROTRANSMITTER_CONFIGS = {
        "dopamine": {"baseline": 0.50, "min": 0.05, "max": 1.0, "decay": 0.03, "recovery": 0.04},
        "serotonin": {"baseline": 0.60, "min": 0.10, "max": 1.0, "decay": 0.02, "recovery": 0.03},
        "norepinephrine": {"baseline": 0.40, "min": 0.05, "max": 1.0, "decay": 0.04, "recovery": 0.05},
        "acetylcholine": {"baseline": 0.50, "min": 0.10, "max": 1.0, "decay": 0.03, "recovery": 0.04},
        "gaba": {"baseline": 0.60, "min": 0.10, "max": 1.0, "decay": 0.02, "recovery": 0.03},
        "glutamate": {"baseline": 0.50, "min": 0.10, "max": 1.0, "decay": 0.03, "recovery": 0.04},
        "cortisol": {"baseline": 0.30, "min": 0.10, "max": 1.0, "decay": 0.02, "recovery": 0.01},
    }

    def __init__(self):
        self.neurotransmitters: dict[str, NeurotransmitterSetpoint] = {}
        for name, cfg in self._NEUROTRANSMITTER_CONFIGS.items():
            self.neurotransmitters[name] = NeurotransmitterSetpoint(
                name=name,
                baseline=cfg["baseline"],
                minimum=cfg["min"],
                maximum=cfg["max"],
                decay_rate=cfg["decay"],
                recovery_rate=cfg["recovery"],
                current=cfg["baseline"],
            )

        self.autonomic_state = AutonomicState.BALANCED
        self.sympathetic_tone = 0.4
        self.parasympathetic_tone = 0.6

        self.stress = StressResponseState()
        self.vagal_tone = VagalToneState()
        self.energy = EnergyBudget()


        self._history: deque[dict] = deque(maxlen=1000)
        self._tick_count = 0
        self._lock = threading.RLock()
        self._allostatic_drift_rate = 0.001

    def tick(self, dt: float = 1.0) -> dict:
        with self._lock:
            self._tick_count += 1

            for name, nt in self.neurotransmitters.items():
                nt.apply_homeostatic_pressure(dt)

            self._apply_allostatic_drift(dt)
            self._apply_stochastic_fluctuation(dt)

            self.stress.tick(dt)
            self.vagal_tone.regulate(self.stress.intensity, dt)
            self.energy.tick(dt)

            self._update_autonomic_state()
            self._apply_circadian_modulation()
            self._resource_rebalance()

            if self._tick_count % 10 == 0:
                self._history.append(self.get_state())

            return self.get_state()

    def apply_stressor(self, intensity: float, source: str = "") -> dict:
        with self._lock:
            self.stress.trigger_alarm(intensity)
            self.neurotransmitters["norepinephrine"].apply_stress(intensity * 0.5)
            self.neurotransmitters["dopamine"].apply_stress(-intensity * 0.2)
            self.neurotransmitters["cortisol"].apply_stress(intensity * 0.4)
            gaba_boost = intensity * 0.2
            self.neurotransmitters["gaba"].apply_stress(gaba_boost)
            self.vagal_tone.regulate(intensity)
            return self.get_state()

    def apply_reward(self, magnitude: float, neurotransmitter: str = "dopamine") -> dict:
        with self._lock:
            if neurotransmitter in self.neurotransmitters:
                self.neurotransmitters[neurotransmitter].apply_reward(magnitude)
            if magnitude > 0.3:
                self.neurotransmitters["serotonin"].apply_reward(magnitude * 0.3)
            self.energy.total_capacity = min(1.0, self.energy.total_capacity + magnitude * 0.01)
            return self.get_state()

    def apply_punishment(self, magnitude: float) -> dict:
        with self._lock:
            self.neurotransmitters["dopamine"].apply_stress(-magnitude * 0.4)
            self.neurotransmitters["serotonin"].apply_stress(-magnitude * 0.2)
            self.neurotransmitters["norepinephrine"].apply_stress(magnitude * 0.3)
            self.stress.trigger_alarm(magnitude * 0.5)
            return self.get_state()

    def allocate_resource(self, resource: str, amount: float) -> bool:
        with self._lock:
            success, _ = self.energy.allocate(resource, amount)
            return success

    def release_resource(self, resource: str, amount: Optional[float] = None) -> float:
        with self._lock:
            return self.energy.release(resource, amount)

    def get_neurotransmitter(self, name: str) -> float:
        with self._lock:
            nt = self.neurotransmitters.get(name)
            return nt.current if nt else 0.5

    def get_arousal(self) -> float:
        with self._lock:
            ne = self.neurotransmitters["norepinephrine"].current
            ach = self.neurotransmitters["acetylcholine"].current
            return 0.6 * ne + 0.4 * ach

    def get_valence(self) -> float:
        with self._lock:
            da = self.neurotransmitters["dopamine"].current
            ser = self.neurotransmitters["serotonin"].current
            return 0.6 * (da - 0.5) + 0.4 * (ser - 0.5)

    def get_cognitive_capacity(self) -> float:
        with self._lock:
            wm = self.energy.allocated.get(ResourceType.WORKING_MEMORY.value, 0.0)
            att = self.energy.allocated.get(ResourceType.ATTENTION.value, 0.0)
            capacity = self.energy.total_capacity - wm * 0.4 - att * 0.3
            stress_penalty = self.stress.allostatic_load * 0.3
            vagal_boost = self.vagal_tone.current * 0.1

            result = max(0.1, min(1.0, capacity - stress_penalty + vagal_boost))
            return result

    def get_stress_level(self) -> float:
        with self._lock:
            return self.stress.intensity

    def get_allostatic_load(self) -> float:
        with self._lock:
            return self.stress.allostatic_load

    def is_homeostatic(self) -> bool:
        with self._lock:
            deviations = [nt.deviation for nt in self.neurotransmitters.values()]
            max_dev = max(deviations) if deviations else 0.0
            return max_dev < 0.2 and self.stress.phase in (StressPhase.RECOVERY,)

    def get_regulatory_actions(self) -> list[str]:
        with self._lock:
            actions = []
            da = self.neurotransmitters["dopamine"]
            ser = self.neurotransmitters["serotonin"]
            ne = self.neurotransmitters["norepinephrine"]
            ach = self.neurotransmitters["acetylcholine"]

        if da.current < da.baseline - 0.2:
            actions.append("increase_novelty_seeking")
        if da.current > da.baseline + 0.3:
            actions.append("moderate_reward_seeking")
        if ser.current < ser.baseline - 0.2:
            actions.append("increase_positive_reframing")
        if ne.current > ne.baseline + 0.3:
            actions.append("reduce_stimulation")
        if ne.current < ne.baseline - 0.2:
            actions.append("increase_urgency_sensitivity")
        if ach.current > ach.baseline + 0.3:
            actions.append("focus_attention")
        if ach.current < ach.baseline - 0.2:
            actions.append("broaden_attention")
        if self.stress.phase == StressPhase.ALARM:
            actions.append("activate_fight_or_flight")
        if self.stress.phase == StressPhase.EXHAUSTION:
            actions.append("enter_recovery_mode")
        if self.energy.utilization > 0.8:
            actions.append("conserve_resources")
        return actions

    def get_state(self) -> dict:
        with self._lock:
            return {
                "neurotransmitters": {name: round(nt.current, 4) for name, nt in self.neurotransmitters.items()},
            "autonomic": self.autonomic_state.value,
            "sympathetic_tone": round(self.sympathetic_tone, 4),
            "parasympathetic_tone": round(self.parasympathetic_tone, 4),
            "arousal": round(self.get_arousal(), 4),
            "valence": round(self.get_valence(), 4),
            "cognitive_capacity": round(self.get_cognitive_capacity(), 4),
            "stress_phase": self.stress.phase.value,
            "stress_intensity": round(self.stress.intensity, 4),
            "allostatic_load": round(self.stress.allostatic_load, 4),
            "cortisol": round(self.stress.cortisol, 4),
            "adrenaline": round(self.stress.adrenaline, 4),
            "vagal_tone": round(self.vagal_tone.current, 4),
            "energy_available": round(self.energy.available, 4),
            "energy_utilization": round(self.energy.utilization, 4),
            "is_homeostatic": self.is_homeostatic(),
            "regulatory_actions": self.get_regulatory_actions(),
            "tick": self._tick_count,
        }

    def _update_autonomic_state(self) -> None:
        sympathetic_drive = (
            self.neurotransmitters["norepinephrine"].current * 0.4
            + self.stress.adrenaline * 0.3
            + (1.0 - self.vagal_tone.current) * 0.3
        )
        parasympathetic_drive = (
            self.neurotransmitters["gaba"].current * 0.4
            + self.vagal_tone.current * 0.3
            + self.neurotransmitters["serotonin"].current * 0.3
        )
        self.sympathetic_tone = max(0.0, min(1.0, 0.3 + sympathetic_drive))
        self.parasympathetic_tone = max(0.0, min(1.0, 0.3 + parasympathetic_drive))

        if self.sympathetic_tone > self.parasympathetic_tone + 0.2:
            self.autonomic_state = AutonomicState.SYMPATHETIC
        elif self.parasympathetic_tone > self.sympathetic_tone + 0.2:
            self.autonomic_state = AutonomicState.PARASYMPATHETIC
        else:
            self.autonomic_state = AutonomicState.BALANCED

    def _apply_circadian_modulation(self) -> None:
        hour = time.localtime().tm_hour
        if 6 <= hour < 12:
            offset = 0.02 * math.sin(math.pi * (hour - 6) / 6)
        elif 12 <= hour < 18:
            offset = 0.01
        elif 18 <= hour < 22:
            offset = -0.02
        else:
            offset = -0.05

        for name, nt in self.neurotransmitters.items():
            nudge = offset * nt.recovery_rate
            nt.current = max(nt.minimum, min(nt.maximum, nt.current + nudge))

        ach_offset = 0.05 if 6 <= hour < 12 else -0.03 if hour >= 22 or hour < 6 else 0.0
        ach = self.neurotransmitters["acetylcholine"]
        ach.current = max(ach.minimum, min(ach.maximum, ach.current + ach_offset * ach.recovery_rate))

        capacity_offset = 0.1 if 6 <= hour < 18 else -0.05
        self.energy.total_capacity = max(0.2, min(1.0, self.energy.total_capacity + capacity_offset * 0.01))

    def _resource_rebalance(self) -> None:
        if self.autonomic_state == AutonomicState.SYMPATHETIC:
            self.energy.allocate(ResourceType.ATTENTION.value, 0.05)
            self.energy.release(ResourceType.CREATIVE.value, 0.03)
        elif self.autonomic_state == AutonomicState.PARASYMPATHETIC:
            self.energy.allocate(ResourceType.CREATIVE.value, 0.03)
            self.energy.release(ResourceType.ATTENTION.value, 0.02)

        if self.stress.phase == StressPhase.EXHAUSTION:
            self.energy.release(ResourceType.ATTENTION.value, 0.1)
            self.energy.release(ResourceType.COMPUTATION.value, 0.1)

    def _apply_allostatic_drift(self, dt: float = 1.0) -> None:
        """Allostatic regulation: chronic stress shifts NT baselines over time.

        Unlike homeostasis (return to setpoint), allostasis adjusts the setpoint
        itself based on environmental demands. Chronic stress raises cortisol baseline,
        chronic reward raises dopamine baseline, etc.
        """
        drift = self._allostatic_drift_rate * dt
        if self.stress.phase in (StressPhase.ALARM, StressPhase.RESISTANCE):
            stress_intensity = self.stress.intensity
            if "cortisol" in self.neurotransmitters:
                self.neurotransmitters["cortisol"].baseline = min(
                    0.6, self.neurotransmitters["cortisol"].baseline + drift * stress_intensity * 0.5
                )
            self.neurotransmitters["norepinephrine"].baseline = min(
                0.6, self.neurotransmitters["norepinephrine"].baseline + drift * stress_intensity * 0.3
            )
            self.vagal_tone.baseline = max(
                0.4, self.vagal_tone.baseline - drift * stress_intensity * 0.2
            )

        elif self.stress.phase == StressPhase.RECOVERY:
            recovery_drift = drift * 0.5
            if "cortisol" in self.neurotransmitters:
                self.neurotransmitters["cortisol"].baseline = max(
                    0.1, self.neurotransmitters["cortisol"].baseline - recovery_drift
                )
            self.vagal_tone.baseline = min(
                0.8, self.vagal_tone.baseline + recovery_drift
            )

    def _apply_stochastic_fluctuation(self, dt: float = 1.0) -> None:
        """Small random fluctuations to prevent static equilibrium.

        Biological systems are never perfectly still. Small stochastic
        noise in neurotransmitter levels creates natural variation that
        makes the system feel alive rather than robotic.
        """
        import random
        noise_scale = 0.002 * dt
        for name, nt in self.neurotransmitters.items():
            if name == "cortisol":
                continue
            noise = random.gauss(0, noise_scale)
            nt.current = max(nt.minimum, min(nt.maximum, nt.current + noise))


def get_bio_homeostasis() -> BioHomeostasis:
    global _homeostasis
    with _lock:
        if _homeostasis is None:
            _homeostasis = BioHomeostasis()
        return _homeostasis


def reset_bio_homeostasis() -> None:
    global _homeostasis
    with _lock:
        _homeostasis = BioHomeostasis()