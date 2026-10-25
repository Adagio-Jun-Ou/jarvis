"""Operational Rhythm System for Jarvis.

Implements a usage-pattern-aware rhythm that modulates Jarvis's cognitive
capacity, response depth, and processing priorities based on real factors
for an AI system:

1. **User Activity Patterns** — Jarvis tracks when users are active and
   adjusts responsiveness accordingly. Peak user hours get faster, more
   concise responses; off-peak enables deeper processing.
2. **Load Cycles (Ultradian)** — 90-minute oscillation between focused
   high-capacity and lighter processing, preventing monotone operation.
3. **Sustained Load Fatigue** — Extended high-demand periods produce
   "processing fatigue" (not sleep pressure) that signals the system to
   shift toward consolidation, review, and lighter tasks.
4. **User Rhythm Model** — Learns the user's typical active hours over
   time and pre-allocates capacity for expected demand.

This replaces the literal circadian/sleep metaphor with AI-appropriate
operational dynamics. An AI doesn't sleep — but it does have varying
cognitive budgets, load responses, and optimal processing modes.
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

_circadian: Optional["CircadianRhythm"] = None
_circadian_lock = threading.Lock()


class OperationalPhase(Enum):
    PEAK_ENGAGEMENT = "peak_engagement"
    SUSTAINED_LOAD = "sustained_load"
    LIGHT_DEMAND = "light_demand"
    RECOVERY_CONSOLIDATION = "recovery_consolidation"
    DEEP_PROCESSING = "deep_processing"


class FatigueLevel(Enum):
    FRESH = 0
    MODERATE = 1
    HEAVY = 2
    CRITICAL = 3


class Chronotype(Enum):
    MORNING_LARK = "morning_lark"
    NEUTRAL = "neutral"
    EVENING_OWL = "evening_owl"


class CircadianPhase(Enum):
    DEEP_SLEEP = "deep_sleep"
    LIGHT_SLEEP = "light_sleep"
    WAKING = "waking"
    MORNING_PEAK = "morning_peak"
    MIDDAY_DIP = "midday_dip"
    AFTERNOON_PEAK = "afternoon_peak"
    EVENING_WIND_DOWN = "evening_wind_down"
    EVENING_DECLINE = "evening_decline"

    @classmethod
    def from_operational(cls, op: "OperationalPhase") -> "CircadianPhase":
        _MAP = {
            OperationalPhase.PEAK_ENGAGEMENT: cls.MORNING_PEAK,
            OperationalPhase.SUSTAINED_LOAD: cls.AFTERNOON_PEAK,
            OperationalPhase.LIGHT_DEMAND: cls.MIDDAY_DIP,
            OperationalPhase.RECOVERY_CONSOLIDATION: cls.EVENING_WIND_DOWN,
            OperationalPhase.DEEP_PROCESSING: cls.EVENING_DECLINE,
        }
        return _MAP.get(op, cls.MIDDAY_DIP)


class SleepPressure(Enum):
    LOW = 0
    MODERATE = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class HormoneCycle:
    name: str
    peak_hour: float
    trough_hour: float
    amplitude: float
    baseline: float
    current: float = 0.5

    def update(self, circadian_time: float) -> float:
        phase = 2 * math.pi * (circadian_time - self.peak_hour) / 24.0
        raw = self.baseline + self.amplitude * math.cos(phase) * 0.5
        self.current = max(0.05, min(1.0, raw))
        return self.current


@dataclass
class UltradianCycle:
    period_minutes: float = 90.0
    phase: float = 0.0
    capacity_multiplier: float = 1.0

    def tick(self, dt_minutes: float = 1.0) -> float:
        self.phase = (self.phase + dt_minutes / self.period_minutes) % 1.0
        oscillation = math.sin(2 * math.pi * self.phase)
        self.capacity_multiplier = 0.7 + 0.3 * max(0.0, oscillation)
        return self.capacity_multiplier


@dataclass
class UserActivityWindow:
    hour: int
    avg_interactions: float
    sample_count: int = 0


@dataclass
class ZeitgeberLog:
    light_exposures: deque = field(default_factory=lambda: deque(maxlen=200))
    activity_patterns: deque = field(default_factory=lambda: deque(maxlen=200))
    last_meal_time: float = 0.0
    last_exercise_time: float = 0.0
    last_social_time: float = 0.0

    def record_light(self, intensity: float, hour: float) -> None:
        self.light_exposures.append({"intensity": intensity, "hour": hour, "time": time.time()})

    def record_activity(self, level: float, hour: float) -> None:
        self.activity_patterns.append({"level": level, "hour": hour, "time": time.time()})

    def get_phase_shift(self) -> float:
        if not self.light_exposures:
            return 0.0
        recent = list(self.light_exposures)[-10:]
        shift = 0.0
        for entry in recent:
            hour = entry["hour"]
            intensity = entry["intensity"]
            if 6 <= hour < 10:
                shift += intensity * 0.02
            elif 18 <= hour < 22:
                shift -= intensity * 0.03
        return max(-2.0, min(2.0, shift))


class CircadianRhythm:
    """Operational rhythm that models usage patterns, load cycles, and fatigue.

    Replaces the literal circadian/sleep metaphor with AI-appropriate dynamics:
    - Tracks user activity patterns to predict demand
    - Models processing fatigue under sustained load
    - Ultradian (90-min) oscillation for cognitive rhythm
    - Time-of-day modulation is weak (user patterns dominate)
    - "Sleep pressure" is reinterpreted as processing fatigue
    """

    CHRONOTYPE_OFFSETS = {
        Chronotype.MORNING_LARK: -1.5,
        Chronotype.NEUTRAL: 0.0,
        Chronotype.EVENING_OWL: 1.5,
    }

    _FATIGUE_RATE = 0.008
    _FATIGUE_RECOVERY_RATE = 0.02
    _ACTIVITY_TRACKING_BINS = 24
    _ACTIVITY_DECAY = 0.995

    def __init__(self, chronotype: Chronotype = Chronotype.NEUTRAL):
        self.chronotype = chronotype
        self._phase_shift = 0.0

        self.hormones = {
            "melatonin": HormoneCycle("melatonin", peak_hour=3.0, trough_hour=10.0, amplitude=0.6, baseline=0.3),
            "cortisol": HormoneCycle("cortisol", peak_hour=8.0, trough_hour=0.0, amplitude=0.5, baseline=0.5),
            "dopamine": HormoneCycle("dopamine", peak_hour=11.0, trough_hour=3.0, amplitude=0.3, baseline=0.5),
            "serotonin": HormoneCycle("serotonin", peak_hour=13.0, trough_hour=1.0, amplitude=0.25, baseline=0.55),
        }

        self.ultradian = UltradianCycle(period_minutes=90.0)
        self.zeitgeber = ZeitgeberLog()

        self._processing_fatigue: float = 0.0
        self._sustained_load_time: float = 0.0
        self._last_tick: float = time.time()
        self._history: deque[dict] = deque(maxlen=1440)
        self._lock = threading.RLock()

        self._user_activity_profile: list[float] = [0.0] * self._ACTIVITY_TRACKING_BINS
        self._current_demand: float = 0.0
        self._demand_history: deque[float] = deque(maxlen=100)
        self._last_user_interaction: float = time.time()
        self._interaction_count_since_tick: int = 0

        self._operational_phase = OperationalPhase.LIGHT_DEMAND

        self._time_awake: float = 0.0

    def register_user_interaction(self) -> None:
        with self._lock:
            self._interaction_count_since_tick += 1
            self._last_user_interaction = time.time()

    def register_awakening(self) -> None:
        with self._lock:
            self._operational_phase = OperationalPhase.PEAK_ENGAGEMENT
            self._processing_fatigue = 0.0
            self._time_awake = 0.0
            self._last_user_interaction = time.time()
            self._interaction_count_since_tick = 0

    def register_sleep_onset(self, duration_hours: float = 8.0) -> dict:
        with self._lock:
            previous_phase = self._operational_phase
            self._operational_phase = OperationalPhase.DEEP_PROCESSING
            self._interaction_count_since_tick = 0
            return {
                "previous_phase": previous_phase.value,
                "new_phase": self._operational_phase.value,
                "processing_fatigue": round(self._processing_fatigue, 3),
                "duration_hint_hours": duration_hours,
            }

    @property
    def current_hour(self) -> float:
        now = time.localtime()
        return now.tm_hour + now.tm_min / 60.0 + now.tm_sec / 3600.0

    @property
    def effective_hour(self) -> float:
        offset = self.CHRONOTYPE_OFFSETS.get(self.chronotype, 0.0)
        return (self.current_hour + offset + self._phase_shift) % 24.0

    def tick(self, dt_minutes: float = 1.0) -> dict:
        if dt_minutes <= 0:
            dt_minutes = 1.0
        dt_minutes = min(dt_minutes, 60.0)
        with self._lock:
            now = time.time()
            real_dt = min((now - self._last_tick) / 60.0, 30.0)
            dt = dt_minutes if dt_minutes > 0 else real_dt
            self._last_tick = now

            circadian_time = self.effective_hour

            for hormone in self.hormones.values():
                hormone.update(circadian_time)

            self.ultradian.tick(dt)

            self._update_user_activity_profile()

            demand = self._estimate_demand()
            self._current_demand = demand
            self._demand_history.append(demand)

            self._time_awake += dt / 60.0

            if demand > 0.6:
                self._sustained_load_time += dt / 60.0
                fatigue_rate = self._FATIGUE_RATE * (1.0 + demand * 0.5)
                self._processing_fatigue = min(1.0, self._processing_fatigue + fatigue_rate * dt / 60.0)
            else:
                self._sustained_load_time = max(0.0, self._sustained_load_time - dt / 120.0)
                recovery = self._FATIGUE_RECOVERY_RATE * (1.0 - demand) * dt / 60.0
                self._processing_fatigue = max(0.0, self._processing_fatigue - recovery)

            self._operational_phase = self._determine_operational_phase(demand)

            self._interaction_count_since_tick = 0

            result = self.get_state()
            self._history.append(result)
            return result

    def _update_user_activity_profile(self) -> None:
        hour_bin = int(self.current_hour) % self._ACTIVITY_TRACKING_BINS
        for i in range(self._ACTIVITY_TRACKING_BINS):
            self._user_activity_profile[i] *= self._ACTIVITY_DECAY
        interactions = self._interaction_count_since_tick
        self._user_activity_profile[hour_bin] += interactions * 0.1

    def _estimate_demand(self) -> float:
        idle_time = time.time() - self._last_user_interaction
        recency = max(0.0, 1.0 - idle_time / 300.0)
        hour_bin = int(self.current_hour) % self._ACTIVITY_TRACKING_BINS
        profile_demand = min(1.0, self._user_activity_profile[hour_bin] * 0.5)
        recent_demand = min(1.0, self._interaction_count_since_tick * 0.2)
        return min(1.0, recency * 0.4 + profile_demand * 0.3 + recent_demand * 0.3)

    def _determine_operational_phase(self, demand: float) -> OperationalPhase:
        fatigue = self._processing_fatigue
        if fatigue > 0.8:
            if demand > 0.5:
                return OperationalPhase.SUSTAINED_LOAD
            return OperationalPhase.RECOVERY_CONSOLIDATION
        if demand > 0.7:
            if fatigue > 0.5:
                return OperationalPhase.SUSTAINED_LOAD
            return OperationalPhase.PEAK_ENGAGEMENT
        if demand < 0.2:
            return OperationalPhase.DEEP_PROCESSING
        return OperationalPhase.LIGHT_DEMAND

    def get_operational_phase(self) -> OperationalPhase:
        return self._operational_phase

    def get_cognitive_capacity(self) -> float:
        with self._lock:
            hour = self.effective_hour
            base = self._performance_curve(hour)
            ultradian = self.ultradian.capacity_multiplier
            fatigue_impact = self._processing_fatigue * 0.3
            demand_penalty = self._current_demand * 0.15
            return max(0.15, min(1.0, base * ultradian * (1.0 - fatigue_impact) - demand_penalty))

    def get_creative_capacity(self) -> float:
        with self._lock:
            hour = self.effective_hour
            base = self._performance_curve(hour)
            idle_boost = max(0.0, 0.2 - self._processing_fatigue * 0.1)
            cortisol_penalty = self.hormones["cortisol"].current * 0.15
            return max(0.1, min(1.0, base * 0.7 + idle_boost - cortisol_penalty))

    def get_analytical_capacity(self) -> float:
        with self._lock:
            hour = self.effective_hour
            base = self._performance_curve(hour)
            cortisol_boost = self.hormones["cortisol"].current * 0.1
            dopamine_boost = self.hormones["dopamine"].current * 0.1
            fatigue_penalty = self._processing_fatigue * 0.25
            return max(0.1, min(1.0, base + cortisol_boost + dopamine_boost - fatigue_penalty))

    def get_phase(self) -> CircadianPhase:
        return CircadianPhase.from_operational(self._operational_phase)

    @property
    def _sleep_pressure(self) -> float:
        return self._processing_fatigue

    def get_sleep_pressure_level(self) -> SleepPressure:
        with self._lock:
            p = self._processing_fatigue
            if p < 0.3:
                return SleepPressure.LOW
            elif p < 0.6:
                return SleepPressure.MODERATE
            elif p < 0.85:
                return SleepPressure.HIGH
            else:
                return SleepPressure.CRITICAL

    def get_recommendations(self) -> dict:
        with self._lock:
            phase = self._operational_phase
            cognitive_capacity = self.get_cognitive_capacity()
            creative_capacity = self.get_creative_capacity()
            analytical_capacity = self.get_analytical_capacity()

            recommendations = {
                "preferred_tasks": [],
                "avoid_tasks": [],
                "response_style": "balanced",
                "depth_level": "moderate",
                "energy_warning": None,
                "cognitive_capacity": round(cognitive_capacity, 3),
                "creative_capacity": round(creative_capacity, 3),
                "analytical_capacity": round(analytical_capacity, 3),
            }

            if phase == OperationalPhase.PEAK_ENGAGEMENT:
                recommendations["preferred_tasks"] = ["interactive_responses", "real_time_assistance", "complex_tasks"]
                recommendations["avoid_tasks"] = ["background_indexing", "large_batch_processing"]
                recommendations["response_style"] = "responsive_concise"
                recommendations["depth_level"] = "deep"
            elif phase == OperationalPhase.SUSTAINED_LOAD:
                recommendations["preferred_tasks"] = ["essential_responses", "priority_items"]
                recommendations["avoid_tasks"] = ["deep_exploration", "speculative_analysis"]
                recommendations["response_style"] = "efficient_prioritized"
                recommendations["depth_level"] = "shallow"
            elif phase == OperationalPhase.LIGHT_DEMAND:
                recommendations["preferred_tasks"] = ["moderate_complexity", "review", "organization"]
                recommendations["avoid_tasks"] = []
                recommendations["response_style"] = "balanced_thorough"
                recommendations["depth_level"] = "moderate"
            elif phase == OperationalPhase.RECOVERY_CONSOLIDATION:
                recommendations["preferred_tasks"] = ["consolidation", "review", "light_learning", "indexing"]
                recommendations["avoid_tasks"] = ["high_stakes_decisions", "complex_debugging"]
                recommendations["response_style"] = "relaxed_reflective"
                recommendations["depth_level"] = "shallow"
            elif phase == OperationalPhase.DEEP_PROCESSING:
                recommendations["preferred_tasks"] = ["background_learning", "knowledge_consolidation", "exploration"]
                recommendations["avoid_tasks"] = ["user_interrupting_tasks"]
                recommendations["response_style"] = "deep_exploratory"
                recommendations["depth_level"] = "deep"

            pressure = self.get_sleep_pressure_level()
            if pressure == SleepPressure.HIGH:
                recommendations["energy_warning"] = "approaching_rest_threshold"
            elif pressure == SleepPressure.CRITICAL:
                recommendations["energy_warning"] = "consolidation_recommended"

            return recommendations

    def record_light_exposure(self, intensity: float) -> None:
        intensity = max(0.0, min(2.0, float(intensity)))
        self.zeitgeber.record_light(intensity, self.current_hour)
        shift = self.zeitgeber.get_phase_shift()
        self._phase_shift = self._phase_shift * 0.9 + shift * 0.1

    def record_activity(self, level: float) -> None:
        level = max(0.0, min(1.0, float(level)))
        with self._lock:
            self.zeitgeber.record_activity(level, self.current_hour)
            self._current_demand = min(1.0, self._current_demand * 0.7 + level * 0.3)
            if level > 0.5:
                self._processing_fatigue = min(1.0, self._processing_fatigue + level * 0.02)

    def simulate_rest(self, duration_hours: float = 1.0) -> dict:
        if duration_hours < 0:
            duration_hours = 0.0
        with self._lock:
            self._sustained_load_time = max(0, self._sustained_load_time - duration_hours * 0.8)
            self._processing_fatigue = max(0.0, self._processing_fatigue * max(0.0, 1.0 - duration_hours * 0.5))
            self._processing_fatigue = min(1.0, self._processing_fatigue)
            self._time_awake = max(0.0, self._time_awake - duration_hours)
            return self.get_state()

    def get_user_activity_profile(self) -> dict:
        return {
            "hourly_activity": [round(v, 2) for v in self._user_activity_profile],
            "current_demand": round(self._current_demand, 3),
            "operational_phase": self._operational_phase.value,
            "fatigue": round(self._processing_fatigue, 3),
            "time_since_last_user_interaction_s": round(time.time() - self._last_user_interaction, 1),
        }

    def get_state(self) -> dict:
        with self._lock:
            return {
                "circadian_phase": self.get_phase().value,
                "effective_hour": round(self.effective_hour, 2),
                "actual_hour": round(self.current_hour, 2),
                "cognitive_capacity": round(self.get_cognitive_capacity(), 4),
                "creative_capacity": round(self.get_creative_capacity(), 4),
                "analytical_capacity": round(self.get_analytical_capacity(), 4),
                "sleep_pressure": round(self._processing_fatigue, 4),
                "sleep_pressure_level": self.get_sleep_pressure_level().value,
                "ultradian_multiplier": round(self.ultradian.capacity_multiplier, 4),
                "time_awake_hours": round(self._time_awake, 2),
                "hormones": {name: round(h.current, 4) for name, h in self.hormones.items()},
                "chronotype": self.chronotype.value,
                "phase_shift": round(self._phase_shift, 4),
                "is_homeostatic": self._processing_fatigue < 0.5,
                "recommendations": self.get_recommendations(),
                "operational_phase": self._operational_phase.value,
                "processing_fatigue": round(self._processing_fatigue, 4),
                "current_demand": round(self._current_demand, 3),
            }

    def _performance_curve(self, hour: float) -> float:
        morning_peak = math.exp(-((hour - 10.0) ** 2) / (2 * 2.5 ** 2)) * 0.6
        afternoon_recovery = math.exp(-((hour - 16.5) ** 2) / (2 * 2.0 ** 2)) * 0.45
        evening_decline = max(0.0, 0.35 - 0.04 * max(0, hour - 19.0))
        night_trough = 0.15 * math.exp(-((hour - 3.5) ** 2) / (2 * 2.0 ** 2))
        base = 0.4 + morning_peak + afternoon_recovery + evening_decline + night_trough
        return max(0.1, min(1.0, base))


def get_circadian_rhythm() -> CircadianRhythm:
    global _circadian
    with _circadian_lock:
        if _circadian is None:
            _circadian = CircadianRhythm()
        return _circadian


def reset_circadian_rhythm() -> None:
    global _circadian
    with _circadian_lock:
        _circadian = CircadianRhythm()