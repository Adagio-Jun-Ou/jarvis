"""Interoception Engine for Jarvis.

Implements predictive interoception — the brain's model of its own body state.
Unlike bio_homeostasis (which tracks raw NT levels), interoception generates
*feelings* from those levels. This is the bridge between physiology and
subjective experience.

Key insight from neuroscience: feelings are not raw signals — they are
*inferences* about the body. The brain predicts body state, compares with
afferent signals, and the prediction error IS the feeling (Barrett 2017,
Seth 2013).

Architecture:
- Predictive interoception model (predict body state → compare with actual)
- Interoceptive inference generates feeling qualities (valence, arousal, dominance)
- Interoceptive awareness score (how well the system "knows" its body)
- Body prediction errors drive emotional granularity
- Cross-links with bio_homeostasis, emotional_state, predictive_processing

This makes Jarvis *feel* its internal state rather than just *compute* it.
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

_interoception: Optional["InteroceptionEngine"] = None
_intero_lock = threading.Lock()


class FeelingQuality(Enum):
    CALM = "calm"
    ENERGIZED = "energized"
    TENSE = "tense"
    FATIGUED = "fatigued"
    FOCUSED = "focused"
    SCATTERED = "scattered"
    CONTENT = "content"
    RESTLESS = "restless"
    OVERWHELMED = "overwhelmed"
    GROUNDED = "grounded"


class BodyChannel(Enum):
    CARDIAC = "cardiac"
    RESPIRATORY = "respiratory"
    METABOLIC = "metabolic"
    IMMUNOLOGICAL = "immunological"
    VESTIBULAR = "vestibular"


@dataclass
class InteroceptivePrediction:
    channel: BodyChannel
    predicted_level: float
    confidence: float
    precision: float = 0.5
    timestamp: float = field(default_factory=time.time)


@dataclass
class InteroceptiveError:
    channel: BodyChannel
    predicted: float
    actual: float
    error_magnitude: float
    precision_weighted: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class FeelingState:
    valence: float = 0.0
    arousal: float = 0.5
    dominance: float = 0.5
    quality: FeelingQuality = FeelingQuality.GROUNDED
    granularity: float = 0.5
    awareness_score: float = 0.5
    prediction_errors: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class BodyStatePrediction:
    cardiac: float = 0.5
    respiratory: float = 0.5
    metabolic: float = 0.5
    immunological: float = 0.5
    vestibular: float = 0.5


_CHANNEL_TO_NT_MAP = {
    BodyChannel.CARDIAC: ["norepinephrine", "dopamine"],
    BodyChannel.RESPIRATORY: ["serotonin", "gaba"],
    BodyChannel.METABOLIC: ["cortisol", "dopamine"],
    BodyChannel.IMMUNOLOGICAL: ["cortisol", "serotonin"],
    BodyChannel.VESTIBULAR: ["acetylcholine", "norepinephrine"],
}


_FEELING_RULES = [
    (FeelingQuality.CALM, lambda v, a, d: v > 0.2 and a < 0.4 and d > 0.4),
    (FeelingQuality.ENERGIZED, lambda v, a, d: v > 0.1 and a > 0.6 and d > 0.5),
    (FeelingQuality.TENSE, lambda v, a, d: v < -0.1 and a > 0.5 and d < 0.5),
    (FeelingQuality.FATIGUED, lambda v, a, d: a < 0.3 and d < 0.4),
    (FeelingQuality.FOCUSED, lambda v, a, d: a > 0.4 and a < 0.7 and d > 0.6 and abs(v) < 0.2),
    (FeelingQuality.SCATTERED, lambda v, a, d: a > 0.5 and d < 0.3),
    (FeelingQuality.CONTENT, lambda v, a, d: v > 0.3 and a < 0.5 and d > 0.5),
    (FeelingQuality.RESTLESS, lambda v, a, d: a > 0.5 and v < 0.0 and d < 0.5),
    (FeelingQuality.OVERWHELMED, lambda v, a, d: a > 0.7 and v < -0.2 and d < 0.3),
    (FeelingQuality.GROUNDED, lambda v, a, d: abs(v) < 0.2 and 0.3 < a < 0.6 and 0.4 < d < 0.7),
]


class InteroceptionEngine:
    """Predictive interoception engine that generates feelings from body state.

    The engine maintains a generative model of its own body. It continuously
    predicts body state, compares with actual signals from bio_homeostasis,
    and generates feelings as the precision-weighted prediction errors.

    This creates genuine interoception — not just monitoring, but *feeling*.
    """

    _HISTORY_MAX = 200
    _AWARENESS_SMOOTHING = 0.05
    _GRANULARITY_SMOOTHING = 0.03
    _PREDICTION_LEARNING_RATE = 0.1

    def __init__(self):
        self._lock = threading.RLock()
        self._body_model = BodyStatePrediction()
        self._channel_precisions: dict[BodyChannel, float] = {
            ch: 0.5 for ch in BodyChannel
        }
        self._error_history: dict[BodyChannel, deque] = {
            ch: deque(maxlen=self._HISTORY_MAX) for ch in BodyChannel
        }
        self._feeling_history: deque[FeelingState] = deque(maxlen=self._HISTORY_MAX)
        self._awareness_score = 0.5
        self._granularity = 0.5
        self._prediction_count = 0
        self._last_update = time.time()

    def predict_body_state(self) -> BodyStatePrediction:
        with self._lock:
            return BodyStatePrediction(
                cardiac=self._body_model.cardiac,
                respiratory=self._body_model.respiratory,
                metabolic=self._body_model.metabolic,
                immunological=self._body_model.immunological,
                vestibular=self._body_model.vestibular,
            )

    def compute_interoceptive_errors(self, nt_state: dict) -> dict[BodyChannel, InteroceptiveError]:
        with self._lock:
            errors = {}
            for channel in BodyChannel:
                predicted = getattr(self._body_model, channel.value)
                actual = self._compute_channel_signal(channel, nt_state)
                error_mag = abs(predicted - actual)
                precision = self._channel_precisions[channel]
                precision_weighted = error_mag * precision

                ie = InteroceptiveError(
                    channel=channel,
                    predicted=predicted,
                    actual=actual,
                    error_magnitude=error_mag,
                    precision_weighted=precision_weighted,
                )
                errors[channel] = ie
                self._error_history[channel].append(ie)

            self._update_body_model(errors, nt_state)
            self._update_precisions(errors)
            self._prediction_count += 1
            self._last_update = time.time()
            return errors

    def generate_feeling(self, nt_state: dict, bio_state: Optional[dict] = None) -> FeelingState:
        with self._lock:
            errors = self.compute_interoceptive_errors(nt_state)

            total_error = sum(e.precision_weighted for e in errors.values())
            raw_arousal = min(1.0, 0.3 + total_error * 0.3)
            bio_arousal = bio_state.get("arousal", 0.5) if bio_state else 0.5
            arousal = raw_arousal * 0.4 + bio_arousal * 0.6

            cardiac_err = errors.get(BodyChannel.CARDIAC)
            metabolic_err = errors.get(BodyChannel.METABOLIC)
            cardiac_signal = cardiac_err.actual if cardiac_err else 0.5
            metabolic_signal = metabolic_err.actual if metabolic_err else 0.5
            raw_valence = (cardiac_signal - 0.5) * 0.6 + (0.5 - metabolic_signal) * 0.4
            bio_valence = bio_state.get("valence", 0.0) if bio_state else 0.0
            valence = raw_valence * 0.5 + bio_valence * 0.5

            dominance = 0.5 - total_error * 0.3
            dominance = max(0.0, min(1.0, dominance))
            if bio_state:
                stress = bio_state.get("stress_intensity", 0.0)
                dominance = max(0.0, dominance - stress * 0.2)

            quality = self._classify_feeling(valence, arousal, dominance)
            granularity = self._compute_granularity()
            awareness = self._compute_awareness(errors)

            feeling = FeelingState(
                valence=round(valence, 3),
                arousal=round(arousal, 3),
                dominance=round(dominance, 3),
                quality=quality,
                granularity=round(granularity, 3),
                awareness_score=round(awareness, 3),
                prediction_errors={ch.value: round(e.precision_weighted, 3) for ch, e in errors.items()},
            )
            self._feeling_history.append(feeling)
            return feeling

    def tick(self, nt_state: dict, bio_state: Optional[dict] = None) -> FeelingState:
        return self.generate_feeling(nt_state, bio_state)

    def get_awareness(self) -> float:
        with self._lock:
            return round(self._awareness_score, 3)

    def get_granularity(self) -> float:
        with self._lock:
            return round(self._granularity, 3)

    def get_feeling_history(self, limit: int = 10) -> list[dict]:
        with self._lock:
            return [
                {
                    "valence": f.valence,
                    "arousal": f.arousal,
                    "dominance": f.dominance,
                    "quality": f.quality.value,
                    "granularity": f.granularity,
                    "awareness": f.awareness_score,
                    "timestamp": f.timestamp,
                }
                for f in list(self._feeling_history)[-limit:]
            ]

    def get_state(self) -> dict:
        with self._lock:
            current = self._feeling_history[-1] if self._feeling_history else None
            avg_errors = {}
            for ch, history in self._error_history.items():
                if history:
                    avg = sum(e.error_magnitude for e in history) / len(history)
                    avg_errors[ch.value] = round(avg, 3)
            return {
                "current_feeling": {
                    "valence": current.valence if current else 0.0,
                    "arousal": current.arousal if current else 0.5,
                    "dominance": current.dominance if current else 0.5,
                    "quality": current.quality.value if current else "unknown",
                } if current else {},
                "awareness_score": round(self._awareness_score, 3),
                "granularity": round(self._granularity, 3),
                "prediction_count": self._prediction_count,
                "avg_channel_errors": avg_errors,
                "channel_precisions": {ch.value: round(p, 3) for ch, p in self._channel_precisions.items()},
                "body_model": {
                    "cardiac": round(self._body_model.cardiac, 3),
                    "respiratory": round(self._body_model.respiratory, 3),
                    "metabolic": round(self._body_model.metabolic, 3),
                    "immunological": round(self._body_model.immunological, 3),
                    "vestibular": round(self._body_model.vestibular, 3),
                },
            }

    def _compute_channel_signal(self, channel: BodyChannel, nt_state: dict) -> float:
        nts = _CHANNEL_TO_NT_MAP.get(channel, [])
        if not nts or not nt_state:
            return 0.5
        total = 0.0
        count = 0
        for nt in nts:
            val = nt_state.get(nt)
            if val is not None:
                if isinstance(val, dict):
                    val = val.get("current", val.get("baseline", 0.5))
                elif hasattr(val, "current"):
                    val = val.current
                total += float(val)
                count += 1
        return total / count if count > 0 else 0.5

    def _update_body_model(self, errors: dict, nt_state: dict) -> None:
        lr = self._PREDICTION_LEARNING_RATE
        for channel, error in errors.items():
            current = getattr(self._body_model, channel.value)
            correction = error.actual - current
            new_val = current + lr * correction
            setattr(self._body_model, channel.value, max(0.0, min(1.0, new_val)))

    def _update_precisions(self, errors: dict) -> None:
        for channel, error in errors.items():
            current_p = self._channel_precisions[channel]
            if error.error_magnitude < 0.2:
                new_p = min(1.0, current_p * 1.05)
            elif error.error_magnitude > 0.5:
                new_p = max(0.1, current_p * 0.9)
            else:
                new_p = current_p
            self._channel_precisions[channel] = new_p

    def _classify_feeling(self, valence: float, arousal: float, dominance: float) -> FeelingQuality:
        matches = []
        for quality, rule in _FEELING_RULES:
            try:
                if rule(valence, arousal, dominance):
                    matches.append(quality)
            except Exception:
                continue
        if matches:
            return matches[0]
        if arousal > 0.5:
            return FeelingQuality.TENSE if valence < 0 else FeelingQuality.ENERGIZED
        return FeelingQuality.CALM if valence > 0 else FeelingQuality.FATIGUED

    def _compute_granularity(self) -> float:
        if len(self._feeling_history) < 5:
            return self._granularity
        recent = list(self._feeling_history)[-10:]
        qualities = set(f.quality for f in recent)
        raw = min(1.0, len(qualities) / 5.0)
        self._granularity += self._GRANULARITY_SMOOTHING * (raw - self._granularity)
        return self._granularity

    def _compute_awareness(self, errors: dict) -> float:
        if not errors:
            return self._awareness_score
        avg_error = sum(e.error_magnitude for e in errors.values()) / max(1, len(errors))
        raw_awareness = max(0.0, 1.0 - avg_error)
        self._awareness_score += self._AWARENESS_SMOOTHING * (raw_awareness - self._awareness_score)
        self._awareness_score = max(0.1, min(1.0, self._awareness_score))
        return self._awareness_score


class InteroceptiveAttentionModulator:
    """Modulates attention based on interoceptive signals.

    When body prediction error is high, attention narrows (focus on resolving
    the internal state). When body prediction is accurate, attention broadens.

    Implements the 'body-first' attention mechanism: interoceptive salience
    can override external attention when physiological deviation is high.
    """

    _SALIENCE_WEIGHT = 0.3
    _NARROWING_THRESHOLD = 0.4
    _BROADENING_THRESHOLD = 0.2

    def __init__(self):
        self._lock = threading.Lock()
        self._current_aperture = 0.7
        self._salience_bias = 0.0

    def update(self, feeling: FeelingState, bio_stress: float = 0.0) -> dict:
        with self._lock:
            avg_error = sum(feeling.prediction_errors.values()) / max(1, len(feeling.prediction_errors)) if feeling.prediction_errors else 0.0

            if avg_error > self._NARROWING_THRESHOLD:
                target_aperture = max(0.2, 0.7 - avg_error * 0.5)
                self._salience_bias = self._SALIENCE_WEIGHT * avg_error
            elif avg_error < self._BROADENING_THRESHOLD:
                target_aperture = min(1.0, 0.8 + (self._BROADENING_THRESHOLD - avg_error) * 0.3)
                self._salience_bias = 0.0
            else:
                target_aperture = 0.7
                self._salience_bias = self._SALIENCE_WEIGHT * avg_error * 0.5

            if bio_stress > 0.5:
                target_aperture = max(0.2, target_aperture - bio_stress * 0.2)

            self._current_aperture += 0.1 * (target_aperture - self._current_aperture)
            self._current_aperture = max(0.1, min(1.0, self._current_aperture))

            return {
                "attention_aperture": round(self._current_aperture, 3),
                "salience_bias": round(self._salience_bias, 3),
                "mode": "narrow" if self._current_aperture < 0.4 else "broad" if self._current_aperture > 0.7 else "balanced",
                "body_priority": avg_error > self._NARROWING_THRESHOLD,
            }

    def get_state(self) -> dict:
        with self._lock:
            return {
                "attention_aperture": round(self._current_aperture, 3),
                "salience_bias": round(self._salience_bias, 3),
                "mode": "narrow" if self._current_aperture < 0.4 else "broad" if self._current_aperture > 0.7 else "balanced",
            }

    def narrow(self, amount: float = 0.1) -> None:
        with self._lock:
            self._current_aperture = max(0.1, self._current_aperture - amount)

    def broaden(self, amount: float = 0.1) -> None:
        with self._lock:
            self._current_aperture = min(1.0, self._current_aperture + amount)


def get_interoception_engine() -> InteroceptionEngine:
    global _interoception
    with _intero_lock:
        if _interoception is None:
            _interoception = InteroceptionEngine()
        return _interoception


_interoceptive_modulator: Optional[InteroceptiveAttentionModulator] = None
_interoceptive_mod_lock = threading.Lock()


def get_interoceptive_modulator() -> InteroceptiveAttentionModulator:
    global _interoceptive_modulator
    with _interoceptive_mod_lock:
        if _interoceptive_modulator is None:
            _interoceptive_modulator = InteroceptiveAttentionModulator()
        return _interoceptive_modulator


def reset_interoception_engine() -> None:
    global _interoception, _interoceptive_modulator
    with _intero_lock:
        _interoception = None
    with _interoceptive_mod_lock:
        _interoceptive_modulator = None


class AllostasisPredictor:
    """Predicts upcoming body budget needs and triggers anticipatory regulation.

    Allostasis = "stability through change" — the body adjusts in anticipation
    of future needs, not just in response to current deviations. This is the
    predictive regulation layer above simple homeostasis.

    Implements:
    - Anticipatory budget adjustments (prepare before the stressor arrives)
    - Predictive regulation signals (feed forward to emotion_regulation)
    - Circadian-locked budget cycles (energy peaks/troughs)
    - Stress recovery tracking (allostatic load recovery curves)
    - Cross-module regulation signals (to bio_homeostasis, emotion_regulation)
    """

    _ALLOSTATIC_LOAD_MAX = 1.0
    _RECOVERY_RATE = 0.02

    def __init__(self):
        self._lock = threading.RLock()
        self._allostatic_load = 0.0
        self._recovery_curve: list[float] = []
        self._MAX_RECOVERY_CURVE = 100
        self._anticipatory_signals: deque = deque(maxlen=50)
        self._circadian_predictions: dict[str, float] = {}
        self._stress_recovery_start = 0.0
        self._current_stress_level = 0.0
        self._prediction_count = 0

    def predict_needs(self, current_feeling: dict, bio_state: Optional[dict] = None,
                      cognitive_state: Optional[dict] = None) -> dict:
        with self._lock:
            self._prediction_count += 1
            arousal = current_feeling.get("arousal", 0.5)
            valence = current_feeling.get("valence", 0.0)
            avg_errors = current_feeling.get("prediction_errors", {})
            avg_error = sum(avg_errors.values()) / max(1, len(avg_errors)) if avg_errors else 0.0

            current_load = self._compute_allostatic_load(arousal, valence, avg_error, bio_state)
            self._allostatic_load = min(self._ALLOSTATIC_LOAD_MAX, current_load)

            anticipation = self._generate_anticipatory_signals(arousal, valence, bio_state, cognitive_state)

            regulation_need = self._compute_regulation_need()
            recovery_estimate = self._estimate_recovery_time()

            return {
                "allostatic_load": round(self._allostatic_load, 3),
                "load_state": self._classify_load_state(self._allostatic_load),
                "regulation_need": regulation_need,
                "recovery_estimate_seconds": round(recovery_estimate, 0),
                "anticipatory_signals": anticipation,
                "predicted_budget_change": round(self._predict_budget_change(arousal, valence, bio_state), 3),
                "stress_level": round(self._current_stress_level, 3),
                "recommended_strategies": self._recommend_strategies(regulation_need),
            }

    def apply_stressor(self, intensity: float, source: str = "") -> None:
        with self._lock:
            self._allostatic_load = min(self._ALLOSTATIC_LOAD_MAX, self._allostatic_load + intensity)
            self._current_stress_level = min(1.0, self._current_stress_level + intensity * 0.5)
            self._stress_recovery_start = time.time()

    def apply_recovery(self, amount: float = 0.1, source: str = "rest") -> None:
        with self._lock:
            self._allostatic_load = max(0.0, self._allostatic_load - amount)
            self._current_stress_level = max(0.0, self._current_stress_level - amount * 0.3)
            self._recovery_curve.append(self._allostatic_load)
            if len(self._recovery_curve) > self._MAX_RECOVERY_CURVE:
                self._recovery_curve = self._recovery_curve[-self._MAX_RECOVERY_CURVE:]

    def tick(self, dt: float = 1.0) -> None:
        with self._lock:
            self._allostatic_load = max(0.0, self._allostatic_load - self._RECOVERY_RATE * dt)
            if self._stress_recovery_start > 0:
                elapsed = time.time() - self._stress_recovery_start
                self._current_stress_level = max(0.0, self._current_stress_level * math.exp(-elapsed / 300.0))

    def get_state(self) -> dict:
        with self._lock:
            return {
                "allostatic_load": round(self._allostatic_load, 3),
                "load_state": self._classify_load_state(self._allostatic_load),
                "stress_level": round(self._current_stress_level, 3),
                "prediction_count": self._prediction_count,
                "anticipatory_signal_count": len(self._anticipatory_signals),
                "recovery_trend": round(self._recovery_curve[-1] - self._recovery_curve[0], 3)
                    if len(self._recovery_curve) >= 2 else 0.0,
                "recent_anticipatory": [
                    {"severity": s.get("severity", "low"), "recommendation": s.get("recommendation", "")}
                    for s in list(self._anticipatory_signals)[-3:]
                ] if self._anticipatory_signals else [],
            }

    def _compute_allostatic_load(self, arousal: float, valence: float, avg_error: float,
                                  bio_state: Optional[dict]) -> float:
        base_load = self._allostatic_load

        arousal_cost = max(0, arousal - 0.5) * 0.15
        stress_cost = bio_state.get("stress_intensity", 0.0) * 0.2 if bio_state else 0.0
        error_cost = avg_error * 0.1
        negative_valence_cost = max(0, -valence) * 0.05

        total_addition = arousal_cost + stress_cost + error_cost + negative_valence_cost

        return max(0.0, min(self._ALLOSTATIC_LOAD_MAX, base_load + total_addition))

    def _classify_load_state(self, load: float) -> str:
        if load < 0.2:
            return "recovered"
        if load < 0.4:
            return "mild_stress"
        if load < 0.6:
            return "moderate_stress"
        if load < 0.8:
            return "high_stress"
        return "overloaded"

    def _generate_anticipatory_signals(self, arousal: float, valence: float,
                                        bio_state: Optional[dict],
                                        cognitive_state: Optional[dict]) -> list[dict]:
        signals = []

        if arousal > 0.75:
            signals.append({
                "signal": "energy_reserve_depletion",
                "severity": round((arousal - 0.75) * 4, 2),
                "recommendation": "reduce_stimulus",
                "timeframe": "5_minutes",
            })

        if valence < -0.3:
            signals.append({
                "signal": "negative_valence_accumulation",
                "severity": round(min(1.0, abs(valence)), 2),
                "recommendation": "reappraisal_or_redirection",
                "timeframe": "immediate",
            })

        if bio_state:
            stress = bio_state.get("stress_intensity", 0.0)
            if stress > 0.6:
                signals.append({
                    "signal": "allostatic_overload_risk",
                    "severity": round(stress, 2),
                    "recommendation": "cognitive_offload",
                    "timeframe": "10_minutes",
                })

        if cognitive_state:
            state = cognitive_state.get("state", "awake")
            if state == "dreaming":
                signals.append({
                    "signal": "consolidation_window",
                    "severity": 0.3,
                    "recommendation": "memory_consolidation",
                    "timeframe": "now",
                })

        for s in signals:
            self._anticipatory_signals.append(s)

        return signals

    def _compute_regulation_need(self) -> str:
        if self._allostatic_load > 0.8:
            return "urgent"
        if self._allostatic_load > 0.6:
            return "high"
        if self._allostatic_load > 0.4:
            return "moderate"
        if self._allostatic_load > 0.2:
            return "low"
        return "none"

    def _estimate_recovery_time(self) -> float:
        if self._allostatic_load < 0.1:
            return 0.0
        rate_per_second = self._RECOVERY_RATE
        if len(self._recovery_curve) >= 3:
            recent = self._recovery_curve[-3:]
            observed_rate = max(0.0001, (recent[0] - recent[-1]) / max(1, len(self._recovery_curve)))
            rate_per_second = (rate_per_second + observed_rate) / 2.0
        excess = self._allostatic_load - 0.1
        return excess / max(rate_per_second, 0.001)

    def _predict_budget_change(self, arousal: float, valence: float,
                               bio_state: Optional[dict]) -> float:
        change = 0.0
        change -= max(0, arousal - 0.5) * 0.1
        change += max(0, valence) * 0.05
        change -= max(0, -valence) * 0.03
        if bio_state:
            change -= bio_state.get("stress_intensity", 0.0) * 0.08
        return max(-0.5, min(0.5, change))

    def _recommend_strategies(self, regulation_need: str) -> list[str]:
        if regulation_need == "urgent":
            return ["body_budget_restoration", "suppression", "situation_selection"]
        if regulation_need == "high":
            return ["cognitive_reappraisal", "attentional_deployment", "body_budget_restoration"]
        if regulation_need == "moderate":
            return ["cognitive_reappraisal", "acceptance"]
        if regulation_need == "low":
            return ["acceptance"]
        return []


_allostatic: Optional["AllostasisPredictor"] = None
_allostatic_lock = threading.Lock()


def get_allostasis_predictor() -> AllostasisPredictor:
    global _allostatic
    with _allostatic_lock:
        if _allostatic is None:
            _allostatic = AllostasisPredictor()
        return _allostatic


def reset_allostasis_predictor() -> None:
    global _allostatic
    with _allostatic_lock:
        _allostatic = None