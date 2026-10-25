"""Affective Context System for Jarvis.

Mood-congruent context enrichment and feeling prediction. When Jarvis is in
a particular emotional state, this system adjusts retrieved context, response
framing, and proactive suggestions to match:

- **Mood-Congruent Framing**: Positive state → optimistic framing, stressed → calming/supportive
- **Context Retrieval Bias**: Current mood biases which memories/prior context are weighted
- **Feeling Prediction**: Trajectory-based forecast of next feeling state
- **Emotional Resonance**: Score how well a proposed response matches current affect
- **Adaptive Empathy**: Adjust empathy level based on user's emotional trajectory
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

_affective_context: Optional["AffectiveContextSystem"] = None
_ac_lock = threading.Lock()


class MoodFraming(Enum):
    OPTIMISTIC = "optimistic"
    SUPPORTIVE = "supportive"
    CALMING = "calming"
    NEUTRAL = "neutral"
    CAUTIOUS = "cautious"
    ENERGETIC = "energetic"


class FeelingTrend(Enum):
    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"
    VOLATILE = "volatile"
    RECOVERING = "recovering"


@dataclass
class FeelingPrediction:
    predicted_valence: float = 0.0
    predicted_arousal: float = 0.5
    predicted_quality: str = "neutral"
    confidence: float = 0.3
    trend: FeelingTrend = FeelingTrend.STABLE
    time_horizon_minutes: float = 5.0
    method: str = "trajectory"


@dataclass
class ContextBias:
    framing: MoodFraming = MoodFraming.NEUTRAL
    recall_valence_bias: float = 0.0
    empathy_level: float = 0.5
    detail_bias: float = 0.5
    urgency_discount: float = 1.0
    positivity_bias: float = 0.0
    suggested_approach: str = ""


class AffectiveContextSystem:
    _MAX_HISTORY = 200
    _TRAJECTORY_WINDOW = 10
    _PREDICTION_WINDOW = 5.0

    _FRAMING_MAP = {
        "joyful": MoodFraming.OPTIMISTIC,
        "excited": MoodFraming.ENERGETIC,
        "grounded": MoodFraming.NEUTRAL,
        "calm": MoodFraming.CALMING,
        "anxious": MoodFraming.SUPPORTIVE,
        "stressed": MoodFraming.SUPPORTIVE,
        "frustrated": MoodFraming.CALMING,
        "overwhelmed": MoodFraming.CALMING,
        "curious": MoodFraming.ENERGETIC,
        "bored": MoodFraming.ENERGETIC,
        "sad": MoodFraming.SUPPORTIVE,
        "angry": MoodFraming.CAUTIOUS,
        "fearful": MoodFraming.SUPPORTIVE,
        "neutral": MoodFraming.NEUTRAL,
    }

    _EMPATHY_MULTIPLIERS = {
        "anxious": 1.4,
        "stressed": 1.3,
        "overwhelmed": 1.5,
        "sad": 1.4,
        "fearful": 1.3,
        "frustrated": 1.2,
        "angry": 1.1,
        "grounded": 0.8,
        "calm": 0.7,
        "joyful": 0.7,
        "excited": 0.6,
        "curious": 0.7,
        "neutral": 0.8,
        "bored": 0.6,
    }

    _APPROACH_SUGGESTIONS = {
        MoodFraming.OPTIMISTIC: "Lean into enthusiasm; share possibilities",
        MoodFraming.SUPPORTIVE: "Acknowledge feelings; offer gentle alternatives",
        MoodFraming.CALMING: "Reduce complexity; break into steps; reassure",
        MoodFraming.CAUTIOUS: "Present options neutrally; avoid pressure",
        MoodFraming.ENERGETIC: "Match energy; be concise and action-oriented",
        MoodFraming.NEUTRAL: "Balanced and informative response",
    }

    def __init__(self):
        self._lock = threading.RLock()
        self._valence_history: deque[tuple[float, float]] = deque(maxlen=self._MAX_HISTORY)
        self._arousal_history: deque[tuple[float, float]] = deque(maxlen=self._MAX_HISTORY)
        self._quality_history: deque[tuple[str, float]] = deque(maxlen=self._MAX_HISTORY)
        self._last_prediction: Optional[FeelingPrediction] = None
        self._last_bias: Optional[ContextBias] = None
        self._resonance_scores: deque[float] = deque(maxlen=50)

    def record_feeling_snapshot(self, valence: float, arousal: float,
                                 quality: str = "neutral") -> None:
        with self._lock:
            now = time.time()
            self._valence_history.append((valence, now))
            self._arousal_history.append((arousal, now))
            self._quality_history.append((quality, now))

    def predict_feeling(self, current_valence: float = 0.0,
                        current_arousal: float = 0.5,
                        current_quality: str = "neutral",
                        horizon_minutes: float = 5.0) -> FeelingPrediction:
        with self._lock:
            recent_val = list(self._valence_history)[-self._TRAJECTORY_WINDOW:]
            recent_ar = list(self._arousal_history)[-self._TRAJECTORY_WINDOW:]

            if len(recent_val) < 3:
                pred = FeelingPrediction(
                    predicted_valence=current_valence,
                    predicted_arousal=current_arousal,
                    predicted_quality=current_quality,
                    confidence=0.2,
                    trend=FeelingTrend.STABLE,
                    time_horizon_minutes=horizon_minutes,
                    method="insufficient_data",
                )
                self._last_prediction = pred
                return pred

            slope_data_val = recent_val + [(current_valence, time.time())]
            slope_data_ar = recent_ar + [(current_arousal, time.time())]

            val_slope = self._compute_slope(slope_data_val)
            ar_slope = self._compute_slope(slope_data_ar)

            predicted_val = current_valence + val_slope * horizon_minutes
            predicted_ar = current_arousal + ar_slope * horizon_minutes * 0.5
            predicted_val = max(-1.0, min(1.0, predicted_val))
            predicted_ar = max(0.0, min(1.0, predicted_ar))

            trend = self._classify_trend(val_slope, slope_data_val)
            predicted_quality = self._predict_quality(current_quality, trend, predicted_val)

            confidence = min(0.9, 0.3 + len(recent_val) * 0.05)
            volatility = self._compute_volatility(slope_data_val)
            if volatility > 0.3:
                confidence *= 0.7
                if trend != FeelingTrend.VOLATILE:
                    trend = FeelingTrend.VOLATILE

            pred = FeelingPrediction(
                predicted_valence=round(predicted_val, 3),
                predicted_arousal=round(predicted_ar, 3),
                predicted_quality=predicted_quality,
                confidence=round(confidence, 3),
                trend=trend,
                time_horizon_minutes=horizon_minutes,
                method="trajectory",
            )
            self._last_prediction = pred
            return pred

    def _compute_slope(self, data: list[tuple[float, float]]) -> float:
        if len(data) < 2:
            return 0.0
        values = [d[0] for d in data]
        n = len(values)
        half = n // 2
        avg_early = sum(values[:half]) / max(1, half)
        avg_late = sum(values[half:]) / max(1, n - half)
        time_span_minutes = (data[-1][1] - data[0][1]) / 60.0
        if time_span_minutes < 0.01:
            return (avg_late - avg_early) / max(1, n)
        return (avg_late - avg_early) / time_span_minutes

    def _compute_volatility(self, data: list[tuple[float, float]]) -> float:
        if len(data) < 3:
            return 0.0
        values = [d[0] for d in data]
        diffs = [abs(values[i+1] - values[i]) for i in range(len(values)-1)]
        return sum(diffs) / len(diffs)

    def _classify_trend(self, slope: float,
                         data: list[tuple[float, float]]) -> FeelingTrend:
        volatility = self._compute_volatility(data)
        if volatility > 0.3:
            return FeelingTrend.VOLATILE
        if slope > 0.01:
            return FeelingTrend.IMPROVING
        if slope < -0.01:
            values = [d[0] for d in data]
            if len(values) >= 5:
                tail_avg = sum(values[-3:]) / 3.0
                prev_avg = sum(values[-5:-3]) / 2.0 if len(values) >= 5 else values[-3]
                if tail_avg > prev_avg + 0.05:
                    return FeelingTrend.RECOVERING
            return FeelingTrend.DECLINING
        return FeelingTrend.STABLE

    def _predict_quality(self, current: str, trend: FeelingTrend,
                          predicted_val: float) -> str:
        if trend == FeelingTrend.IMPROVING:
            if predicted_val > 0.3:
                return "joyful" if current != "joyful" else "excited"
            return "grounded"
        if trend == FeelingTrend.DECLINING:
            if predicted_val < -0.3:
                return "stressed" if current != "stressed" else "overwhelmed"
            return "anxious"
        if trend == FeelingTrend.RECOVERING:
            return "grounded"
        if trend == FeelingTrend.VOLATILE:
            return "anxious"
        return current

    def compute_context_bias(self, current_valence: float = 0.0,
                              current_arousal: float = 0.5,
                              current_quality: str = "neutral",
                              fatigue: float = 0.0) -> ContextBias:
        with self._lock:
            framing = self._FRAMING_MAP.get(current_quality, MoodFraming.NEUTRAL)

            if current_arousal > 0.8:
                if framing == MoodFraming.OPTIMISTIC:
                    framing = MoodFraming.ENERGETIC
            if fatigue > 0.7:
                if framing == MoodFraming.ENERGETIC:
                    framing = MoodFraming.NEUTRAL

            recall_bias = current_valence * 0.3

            empathy = self._EMPATHY_MULTIPLIERS.get(current_quality, 0.8)
            if current_arousal > 0.7:
                empathy = min(1.5, empathy * 1.1)
            if abs(current_valence) > 0.5:
                empathy = min(1.5, empathy * 1.05)

            detail_bias = 0.5
            if current_arousal > 0.7:
                detail_bias = 0.3
            elif current_arousal < 0.3:
                detail_bias = 0.7
            if fatigue > 0.6:
                detail_bias = max(0.2, detail_bias - 0.2)

            urgency_discount = 1.0
            if current_quality in ("overwhelmed", "stressed"):
                urgency_discount = 0.6
            elif current_quality in ("anxious", "frustrated"):
                urgency_discount = 0.8

            positivity_bias = 0.0
            if current_valence < -0.3:
                positivity_bias = 0.3
            elif current_valence < 0.0:
                positivity_bias = 0.15

            approach = self._APPROACH_SUGGESTIONS.get(framing, "Balanced and informative response")

            bias = ContextBias(
                framing=framing,
                recall_valence_bias=round(recall_bias, 3),
                empathy_level=round(empathy, 3),
                detail_bias=round(detail_bias, 3),
                urgency_discount=round(urgency_discount, 3),
                positivity_bias=round(positivity_bias, 3),
                suggested_approach=approach,
            )
            self._last_bias = bias
            return bias

    def compute_resonance(self, current_valence: float, current_arousal: float,
                           response_valence: float, response_arousal: float) -> float:
        val_match = 1.0 - min(1.0, abs(current_valence - response_valence) * 1.5)
        ar_match = 1.0 - min(1.0, abs(current_arousal - response_arousal) * 2.0)
        resonance = val_match * 0.6 + ar_match * 0.4
        with self._lock:
            self._resonance_scores.append(round(resonance, 3))
        return round(resonance, 3)

    def get_feeling_context_summary(self, current_valence: float = 0.0,
                                     current_arousal: float = 0.5,
                                     current_quality: str = "neutral",
                                     fatigue: float = 0.0) -> str:
        prediction = self.predict_feeling(current_valence, current_arousal, current_quality)
        bias = self.compute_context_bias(current_valence, current_arousal, current_quality, fatigue)
        trend_str = prediction.trend.value
        parts = [
            f"Mood: {current_quality} (trend: {trend_str})",
            f"Approach: {bias.suggested_approach}",
            f"Empathy: {'high' if bias.empathy_level > 1.0 else 'normal' if bias.empathy_level > 0.7 else 'low'}",
        ]
        if prediction.confidence > 0.4:
            if prediction.trend in (FeelingTrend.DECLINING, FeelingTrend.VOLATILE):
                parts.append(f"Forecast: likely {prediction.predicted_quality} in ~{prediction.time_horizon_minutes:.0f}min")
        if bias.urgency_discount < 1.0:
            parts.append(f"Urgency: reduced ({bias.urgency_discount:.0%})")
        return " | ".join(parts)

    def get_state(self) -> dict:
        with self._lock:
            avg_resonance = 0.0
            if self._resonance_scores:
                avg_resonance = sum(self._resonance_scores) / len(self._resonance_scores)
            return {
                "snapshots": len(self._valence_history),
                "last_prediction": {
                    "predicted_valence": self._last_prediction.predicted_valence,
                    "predicted_arousal": self._last_prediction.predicted_arousal,
                    "predicted_quality": self._last_prediction.predicted_quality,
                    "confidence": self._last_prediction.confidence,
                    "trend": self._last_prediction.trend.value,
                } if self._last_prediction else None,
                "last_bias": {
                    "framing": self._last_bias.framing.value,
                    "empathy_level": self._last_bias.empathy_level,
                    "detail_bias": self._last_bias.detail_bias,
                    "suggested_approach": self._last_bias.suggested_approach,
                } if self._last_bias else None,
                "avg_resonance": round(avg_resonance, 3),
            }


def get_affective_context() -> AffectiveContextSystem:
    global _affective_context
    if _affective_context is None:
        with _ac_lock:
            if _affective_context is None:
                _affective_context = AffectiveContextSystem()
    return _affective_context


def reset_affective_context() -> None:
    global _affective_context
    with _ac_lock:
        _affective_context = None