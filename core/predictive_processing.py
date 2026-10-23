"""Predictive Processing / Active Inference Engine for Jarvis.

Implements Friston's Free Energy Principle as a computational architecture:
- Hierarchical generative model that predicts incoming input
- Prediction error computation (surprise) at each level
- Precision weighting (attention as uncertainty modulation)
- Free energy minimization via belief updating
- Cross-level message passing (bottom-up errors, top-down predictions)

This is the inner generative model that makes Jarvis *alive* rather than *reactive*:
it continuously predicts what will happen next, and the gap between prediction
and reality drives learning, attention, and state transitions.

Architecture (3-level hierarchy):
  Level 3 (Abstract): Intent prediction, topic prediction, user need prediction
  Level 2 (Context): Response style prediction, tool use prediction, tone prediction
  Level 1 (Sensory): Token prediction, next-word prediction, timing prediction

Each level:
  1. Generates predictions from current beliefs (priors)
  2. Receives bottom-up prediction errors from the level below
  3. Updates beliefs to minimize free energy (precision-weighted prediction errors)
  4. Sends updated predictions top-down

The system continuously runs predictions even without user input, creating
an ongoing internal "train of thought" that is the hallmark of consciousness.
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

_predictive_processor: Optional["PredictiveProcessor"] = None
_pp_lock = threading.Lock()


class PredictionLevel(Enum):
    ABSTRACT = "abstract"       # Level 3: intents, topics, needs
    CONTEXT = "context"         # Level 2: style, tools, tone
    SENSORY = "sensory"         # Level 1: tokens, timing, modality


class DriveType(Enum):
    CURIOSITY = "curiosity"     # Reduce uncertainty / information gain
    MASTERY = "mastery"         # Improve skill / reduce prediction error
    COHERENCE = "coherence"     # Maintain narrative consistency
    NOVELTY = "novelty"         # Seek surprising but learnable input


@dataclass
class Prediction:
    level: PredictionLevel
    predicted: str
    probability: float
    confidence: float
    timestamp: float = field(default_factory=time.time)
    context: dict = field(default_factory=dict)

    @property
    def surprise(self) -> float:
        return -math.log(max(self.probability, 1e-10))


@dataclass
class PredictionError:
    level: PredictionLevel
    prediction: Prediction
    actual: str
    error_magnitude: float
    precision: float
    free_energy: float
    timestamp: float = field(default_factory=time.time)

    @property
    def weighted_error(self) -> float:
        return self.error_magnitude * self.precision


@dataclass
class BeliefState:
    level: PredictionLevel
    beliefs: dict[str, float] = field(default_factory=dict)
    precision_map: dict[str, float] = field(default_factory=dict)
    prediction_errors: deque[float] = field(default_factory=lambda: deque(maxlen=100))
    last_prediction: Optional[Prediction] = None
    update_count: int = 0
    free_energy_history: deque[float] = field(default_factory=lambda: deque(maxlen=200))

    def free_energy(self) -> float:
        if not self.free_energy_history:
            return 0.0
        return self.free_energy_history[-1]

    def average_free_energy(self, window: int = 20) -> float:
        if not self.free_energy_history:
            return 0.0
        window_vals = list(self.free_energy_history)[-window:]
        return sum(window_vals) / len(window_vals)


class PredictiveLevel:
    """Single level in the hierarchical generative model."""

    def __init__(self, level: PredictionLevel, learning_rate: float = 0.1,
                 precision_baseline: float = 0.5, decay_rate: float = 0.01):
        self.level = level
        self.state = BeliefState(level=level)
        self.learning_rate = learning_rate
        self.precision_baseline = precision_baseline
        self.decay_rate = decay_rate
        self._prediction_count = 0

    def predict(self, context: Optional[dict] = None) -> Prediction:
        if not self.state.beliefs:
            default_beliefs = self._default_beliefs()
            self.state.beliefs = default_beliefs

        total = sum(self.state.beliefs.values())
        if total <= 0:
            total = 1.0

        best_key = max(self.state.beliefs, key=self.state.beliefs.get)
        best_prob = self.state.beliefs[best_key] / total
        confidence = self._compute_confidence()

        prediction = Prediction(
            level=self.level,
            predicted=best_key,
            probability=max(best_prob, 0.01),
            confidence=confidence,
            context=context or {},
        )
        self.state.last_prediction = prediction
        self._prediction_count += 1
        return prediction

    def update(self, prediction: Prediction, actual: str,
               precision_override: Optional[float] = None) -> PredictionError:
        error_magnitude = self._compute_error(prediction.predicted, actual)
        precision = precision_override if precision_override is not None else self.state.precision_map.get(
            prediction.predicted, self.precision_baseline
        )

        free_energy = error_magnitude * precision
        self.state.prediction_errors.append(error_magnitude)
        self.state.free_energy_history.append(free_energy)

        self._update_beliefs(prediction.predicted, actual, error_magnitude, precision)
        self._update_precision(prediction.predicted, error_magnitude)
        self.state.update_count += 1

        return PredictionError(
            level=self.level,
            prediction=prediction,
            actual=actual,
            error_magnitude=error_magnitude,
            precision=precision,
            free_energy=free_energy,
        )

    def _compute_error(self, predicted: str, actual: str) -> float:
        if predicted == actual:
            return 0.0

        if not predicted or not actual:
            return 1.0

        pred_chars = set(predicted.lower().split())
        actual_chars = set(actual.lower().split())
        if not pred_chars and not actual_chars:
            return 0.0
        if not pred_chars or not actual_chars:
            return 1.0

        intersection = pred_chars & actual_chars
        union = pred_chars | actual_chars
        jaccard_similarity = len(intersection) / len(union) if union else 0.0
        return 1.0 - jaccard_similarity

    def _compute_confidence(self) -> float:
        if not self.state.prediction_errors:
            return 0.5

        recent_errors = list(self.state.prediction_errors)[-10:]
        avg_error = sum(recent_errors) / len(recent_errors)
        if len(recent_errors) > 1:
            variance = sum((e - avg_error) ** 2 for e in recent_errors) / len(recent_errors)
            precision_est = 1.0 / (1.0 + math.sqrt(variance))
        else:
            precision_est = 0.5

        return max(0.1, min(1.0, 1.0 - avg_error) * precision_est)

    def _update_beliefs(self, predicted: str, actual: str,
                        error: float, precision: float) -> None:
        lr = self.learning_rate * max(precision, 0.1)

        if error < 0.2:
            self.state.beliefs[predicted] = self.state.beliefs.get(predicted, 0.3) + lr * (1.0 - error)
            if actual != predicted:
                self.state.beliefs[actual] = self.state.beliefs.get(actual, 0.1) + lr * 0.5
        elif error < 0.5:
            self.state.beliefs[actual] = self.state.beliefs.get(actual, 0.1) + lr
            self.state.beliefs[predicted] = self.state.beliefs.get(predicted, 0.3) * (1.0 - lr * 0.5)
        else:
            self.state.beliefs[actual] = self.state.beliefs.get(actual, 0.1) + lr * 1.5
            self.state.beliefs[predicted] = self.state.beliefs.get(predicted, 0.3) * (1.0 - lr)

        for key in list(self.state.beliefs.keys()):
            self.state.beliefs[key] *= (1.0 - self.decay_rate)
            if self.state.beliefs[key] < 0.01:
                del self.state.beliefs[key]
                self.state.precision_map.pop(key, None)

        if len(self.state.beliefs) > 100:
            least_important = sorted(self.state.beliefs, key=self.state.beliefs.get)[:30]
            for key in least_important:
                del self.state.beliefs[key]
                self.state.precision_map.pop(key, None)

    def _update_precision(self, key: str, error: float) -> None:
        current_precision = self.state.precision_map.get(key, self.precision_baseline)
        if error < 0.3:
            new_precision = min(1.0, current_precision * 1.1)
        else:
            new_precision = max(0.1, current_precision * 0.9)
        self.state.precision_map[key] = new_precision

    def _default_beliefs(self) -> dict[str, float]:
        if self.level == PredictionLevel.ABSTRACT:
            return {
                "question": 0.3, "request": 0.25, "conversation": 0.2,
                "command": 0.15, "clarification": 0.1,
            }
        elif self.level == PredictionLevel.CONTEXT:
            return {
                "detailed": 0.25, "concise": 0.2, "analytical": 0.2,
                "creative": 0.15, "supportive": 0.2,
            }
        else:
            return {
                "continuation": 0.4, "new_topic": 0.2, "elaboration": 0.2,
                "correction": 0.1, "termination": 0.1,
            }

    def update_beliefs(self, beliefs: dict, precision: float = 0.5) -> None:
        """Public API for directly updating beliefs (used by interoceptive integration)."""
        updated_keys = set()
        for key, value in beliefs.items():
            if isinstance(value, (int, float)):
                k = str(key)
                current = self.state.beliefs.get(k, 0.1)
                self.state.beliefs[k] = current + 0.1 * precision * (float(value) - current)
                self.state.precision_map[k] = min(1.0, self.state.precision_map.get(k, 0.5) + 0.05)
                updated_keys.add(k)
        total = sum(self.state.beliefs.get(k, 0.0) for k in updated_keys)
        remaining = sum(self.state.beliefs.get(k, 0.0) for k in self.state.beliefs if k not in updated_keys)
        if total > 0 and remaining > 0:
            scale = 1.0 / (total + remaining)
            for k in self.state.beliefs:
                self.state.beliefs[k] *= scale


class CurvatureDrive:
    """Intrinsic drive system that generates internal goals.

    Implements four fundamental drives:
    - Curiosity: Reduce uncertainty, maximize information gain
    - Mastery: Reduce prediction error, improve prediction accuracy
    - Coherence: Maintain narrative/logical consistency
    - Novelty: Seek surprising but learnable input (not too predictable, not too random)

    Each drive produces a scalar value [0, 1] indicating current motivation level.
    """

    def __init__(self):
        self.curiosity_level = 0.5
        self.mastery_level = 0.5
        self.coherence_level = 0.8
        self.novelty_bonus = 0.0
        self._curiosity_history: deque[float] = deque(maxlen=50)
        self._mastery_history: deque[float] = deque(maxlen=50)
        self._coherence_history: deque[float] = deque(maxlen=50)
        self._last_update = time.time()

    def update(self, free_energy: float, prediction_confidence: float,
               coherence_signal: float = 1.0, novelty_signal: float = 0.0,
               dt: float = 1.0) -> dict[DriveType, float]:
        curiosity_drive = self._update_curiosity(free_energy, prediction_confidence, dt)
        mastery_drive = self._update_mastery(prediction_confidence, dt)
        coherence_drive = self._update_coherence(coherence_signal, dt)
        novelty_drive = self._update_novelty(novelty_signal, dt)

        self._curiosity_history.append(curiosity_drive)
        self._mastery_history.append(mastery_drive)
        self._coherence_history.append(coherence_drive)

        self._last_update = time.time()
        return {
            DriveType.CURIOSITY: curiosity_drive,
            DriveType.MASTERY: mastery_drive,
            DriveType.COHERENCE: coherence_drive,
            DriveType.NOVELTY: novelty_drive,
        }

    def _update_curiosity(self, free_energy: float, confidence: float, dt: float) -> float:
        surprise = min(1.0, free_energy / 2.0)
        uncertainty = 1.0 - confidence
        target = 0.3 * surprise + 0.7 * uncertainty
        self.curiosity_level += 0.1 * (target - self.curiosity_level) * dt
        self.curiosity_level = max(0.0, min(1.0, self.curiosity_level))
        return self.curiosity_level

    def _update_mastery(self, confidence: float, dt: float) -> float:
        error = 1.0 - confidence
        target = 0.5 + 0.5 * error
        self.mastery_level += 0.08 * (target - self.mastery_level) * dt
        self.mastery_level = max(0.0, min(1.0, self.mastery_level))
        return self.mastery_level

    def _update_coherence(self, signal: float, dt: float) -> float:
        target = signal
        self.coherence_level += 0.05 * (target - self.coherence_level) * dt
        self.coherence_level = max(0.0, min(1.0, self.coherence_level))
        return self.coherence_level

    def _update_novelty(self, signal: float, dt: float) -> float:
        self.novelty_bonus *= max(0.0, 1.0 - 0.1 * dt)
        self.novelty_bonus += signal * 0.3 * dt
        self.novelty_bonus = max(0.0, min(1.0, self.novelty_bonus))
        novelty_drive = 0.3 + 0.7 * self.novelty_bonus
        return max(0.0, min(1.0, novelty_drive))

    def get_dominant_drive(self) -> DriveType:
        drives = {
            DriveType.CURIOSITY: self.curiosity_level,
            DriveType.MASTERY: self.mastery_level,
            DriveType.COHERENCE: self.coherence_level,
            DriveType.NOVELTY: 0.3 + 0.7 * self.novelty_bonus,
        }
        return max(drives, key=drives.get)

    def get_drive_vector(self) -> dict[str, float]:
        return {
            "curiosity": round(self.curiosity_level, 3),
            "mastery": round(self.mastery_level, 3),
            "coherence": round(self.coherence_level, 3),
            "novelty": round(0.3 + 0.7 * self.novelty_bonus, 3),
            "dominant": self.get_dominant_drive().value,
        }


class PredictiveProcessor:
    """Hierarchical predictive processing engine implementing active inference.

    Three-level generative model that continuously predicts incoming input,
    computes prediction errors, and updates beliefs. The prediction error signal
    drives attention, learning, and intrinsic motivation.

    Integration points:
    - agent.py think(): Compare predictions with actual user input
    - bio_integration.py: Drive levels modulate stress/reward
    - attention_priority.py: Prediction errors prioritize attention
    - prompts.py: Drive state shapes response style
    - cognitive.py: Prediction errors trigger state transitions
    """

    def __init__(self):
        self.levels = {
            PredictionLevel.ABSTRACT: PredictiveLevel(PredictionLevel.ABSTRACT),
            PredictionLevel.CONTEXT: PredictiveLevel(PredictionLevel.CONTEXT),
            PredictionLevel.SENSORY: PredictiveLevel(PredictionLevel.SENSORY),
        }
        self.drives = CurvatureDrive()
        self._error_buffer: deque[PredictionError] = deque(maxlen=200)
        self._prediction_buffer: deque[Prediction] = deque(maxlen=100)
        self._free_energy_history: deque[float] = deque(maxlen=500)
        # Last per-level error magnitudes, surfaced by get_state() — the
        # cognitive state machine reads ["errors"]["abstract"]["magnitude"]
        # for its LOW_SURPRISE dreaming gate.
        self._last_errors: dict[str, float] = {}
        self._tick_count = 0
        self._lock = threading.RLock()
        self._last_input_time = time.time()

    def predict(self, context: Optional[dict] = None) -> dict[PredictionLevel, Prediction]:
        with self._lock:
            predictions = {}
            for level, processor in self.levels.items():
                predictions[level] = processor.predict(context)
            self._prediction_buffer.append(predictions.get(PredictionLevel.ABSTRACT))
            return predictions

    def process_input(self, user_input: str, context: Optional[dict] = None) -> dict:
        if user_input is None:
            user_input = ""
        with self._lock:
            self._tick_count += 1
            self._last_input_time = time.time()

            predictions = self.predict(context)
            errors = {}

            for level, prediction in predictions.items():
                actual = self._classify_input(user_input, level)
                error = self.levels[level].update(prediction, actual)
                errors[level] = error
                self._error_buffer.append(error)

            total_free_energy = sum(e.free_energy for e in errors.values())
            avg_confidence = sum(p.confidence for p in predictions.values()) / max(1, len(predictions))

            coherence_signal = self._compute_coherence(errors)
            novelty_signal = self._compute_novelty(errors)

            drive_updates = self.drives.update(
                free_energy=total_free_energy,
                prediction_confidence=avg_confidence,
                coherence_signal=coherence_signal,
                novelty_signal=novelty_signal,
            )

            self._free_energy_history.append(total_free_energy)
            self._last_errors = {l.value: round(e.error_magnitude, 3)
                                 for l, e in errors.items()}

            return {
                "predictions": {l.value: p.predicted for l, p in predictions.items()},
                "errors": {l.value: {"magnitude": round(e.error_magnitude, 3),
                                     "free_energy": round(e.free_energy, 3),
                                     "precision": round(e.precision, 3)}
                           for l, e in errors.items()},
                "total_free_energy": round(total_free_energy, 3),
                "avg_confidence": round(avg_confidence, 3),
                "coherence": round(coherence_signal, 3),
                "novelty": round(novelty_signal, 3),
                "drives": {d.value: round(v, 3) for d, v in drive_updates.items()},
                "dominant_drive": self.drives.get_dominant_drive().value,
            }

    def get_attention_weights(self) -> dict[PredictionLevel, float]:
        with self._lock:
            weights = {}
            for level, processor in self.levels.items():
                avg_fe = processor.state.average_free_energy()
                weights[level] = min(1.0, avg_fe / 2.0 + 0.2)
            return weights

    def get_curiosity_topics(self, top_k: int = 5) -> list[tuple[str, float]]:
        with self._lock:
            abstract = self.levels[PredictionLevel.ABSTRACT]
            topics = []
            for key, belief in abstract.state.beliefs.items():
                precision = abstract.state.precision_map.get(key, 0.5)
                uncertainty = 1.0 - abs(belief - 0.5) * 2
                information_gain = uncertainty * precision
                topics.append((key, information_gain))
            topics.sort(key=lambda x: x[1], reverse=True)
            return topics[:top_k]

    def should_explore(self) -> bool:
        with self._lock:
            if not self._free_energy_history:
                return True
            recent = list(self._free_energy_history)[-5:]
            avg_fe = sum(recent) / len(recent)
            return avg_fe > 0.3 or self.drives.curiosity_level > 0.6

    def get_state(self) -> dict:
        with self._lock:
            return {
                "free_energy": round(self._free_energy_history[-1], 3) if self._free_energy_history else 0.0,
                "avg_free_energy": round(sum(list(self._free_energy_history)[-20:]) / max(1, len(list(self._free_energy_history)[-20:])), 3) if self._free_energy_history else 0.0,
                "drives": self.drives.get_drive_vector(),
                "errors": {lvl: {"magnitude": m} for lvl, m in self._last_errors.items()},
                "prediction_count": self._tick_count,
                "time_since_input": round(time.time() - self._last_input_time, 1),
                "should_explore": self.should_explore(),
                "attention_weights": {l.value: round(w, 3) for l, w in self.get_attention_weights().items()},
                "curiosity_topics": self.get_curiosity_topics(3),
                "levels": {
                    l.value: {
                        "beliefs": dict(list(p.state.beliefs.items())[:10]),
                        "avg_free_energy": round(p.state.average_free_energy(), 3),
                        "update_count": p.state.update_count,
                        "confidence": round(p._compute_confidence(), 3),
                    } for l, p in self.levels.items()
                },
            }

    def tick(self, dt: float = 1.0) -> None:
        with self._lock:
            passive_decay = 0.005 * dt
            for level, processor in self.levels.items():
                for key in list(processor.state.beliefs.keys()):
                    processor.state.beliefs[key] *= (1.0 - passive_decay)
                    if processor.state.beliefs[key] < 0.005:
                        del processor.state.beliefs[key]
                        processor.state.precision_map.pop(key, None)

            idle_time = time.time() - self._last_input_time
            if idle_time > 30:
                self.drives.curiosity_level = min(1.0, self.drives.curiosity_level + 0.01 * dt)
                self.drives.novelty_bonus = min(1.0, self.drives.novelty_bonus + 0.005 * dt)

    def process_interoceptive_errors(self, interoceptive_errors: dict,
                                     feeling_state: Optional[dict] = None) -> dict:
        """Integrate interoceptive prediction errors into the generative model.

        Body state predictions become part of the hierarchical model.
        Interoceptive errors (e.g. cardiac mismatch) update body beliefs
        at level 2 (context), which cascade up to level 3 (abstract) as
        inferred need states.
        """
        with self._lock:
            context_processor = self.levels[PredictionLevel.CONTEXT]
            abstract_processor = self.levels[PredictionLevel.ABSTRACT]

            avg_error = 0.0
            error_values = list(interoceptive_errors.values())
            if error_values:
                total = 0.0
                for e in error_values:
                    if hasattr(e, 'precision_weighted'):
                        total += e.precision_weighted
                    elif isinstance(e, (int, float)):
                        total += float(e)
                avg_error = total / len(error_values)

            if feeling_state:
                arousal = feeling_state.get("arousal", 0.5)
                valence = feeling_state.get("valence", 0.0)
                quality = feeling_state.get("quality", "grounded")

                context_processor.update_beliefs({"body_arousal": arousal}, precision=0.6)
                context_processor.update_beliefs({"body_valence": valence}, precision=0.5)
                context_processor.update_beliefs({"body_quality": quality}, precision=0.4)

            for channel, error in interoceptive_errors.items():
                channel_name = channel.value if hasattr(channel, 'value') else str(channel)
                precision = error.precision if hasattr(error, 'precision') else 0.5
                error_mag = error.error_magnitude if hasattr(error, 'error_magnitude') else 0.0
                context_processor.update_beliefs(
                    {f"body_channel_{channel_name}": 1.0 - error_mag}, precision=precision
                )

            if avg_error > 0.3:
                abstract_processor.update_beliefs({"physiological_needs_attention": 1.0}, precision=avg_error)
                self.drives.curiosity_level = min(1.0, self.drives.curiosity_level + 0.05)
            elif avg_error < 0.1:
                abstract_processor.update_beliefs({"body_homeostatic": 1.0}, precision=0.7)

            # Body errors are context-level: record the latest magnitude so
            # get_state()["errors"] reflects every error source, not just
            # process_input's text-level predictions.
            if error_values:
                self._last_errors["sensory"] = round(avg_error, 3)

            return {
                "integrated_errors": len(interoceptive_errors),
                "avg_error": round(avg_error, 3),
                "body_beliefs_updated": True,
            }

    def process_world_model_errors(self, world_errors: dict,
                                   scene_state: Optional[dict] = None) -> dict:
        """Integrate physical-world prediction errors into the generative model.

        The world model (core/world_model.py) checks observations against
        Spelke-style physics schemas (gravity, permanence, continuity,
        containment, ...); this is the exact analogue of
        process_interoceptive_errors for the EXTEROCEPTIVE/physical domain:
        errors update world beliefs at level 2 (context), which cascade up
        to level 3 (abstract) as physical-world attention. Physical surprise
        boosts the curiosity drive the same way body surprise does.

        Args:
            world_errors: {channel_name: magnitude} floats, where channel is
                e.g. ``world_gravity``, ``world_continuity``.
            scene_state: optional WorldModel.get_state() snapshot — entity
                count and scene freshness are injected as beliefs.
        """
        with self._lock:
            context_processor = self.levels[PredictionLevel.CONTEXT]
            abstract_processor = self.levels[PredictionLevel.ABSTRACT]

            error_values = list(world_errors.values())
            avg_error = 0.0
            if error_values:
                total = 0.0
                for e in error_values:
                    if hasattr(e, 'precision_weighted'):
                        total += e.precision_weighted
                    elif isinstance(e, (int, float)):
                        total += float(e)
                avg_error = total / len(error_values)

            if scene_state:
                entity_count = scene_state.get("entity_count", 0)
                entities = scene_state.get("entities") or []
                occluded = sum(1 for ent in entities
                               if isinstance(ent, dict) and ent.get("occluded"))
                context_processor.update_beliefs(
                    {"world_scene_known": min(1.0, entity_count / 10.0)}, precision=0.5)
                context_processor.update_beliefs(
                    {"world_scene_complexity": min(1.0, entity_count / 20.0)}, precision=0.4)

            for channel, error in world_errors.items():
                if not isinstance(error, (int, float)):
                    continue
                context_processor.update_beliefs(
                    {channel: 1.0 - float(error)}, precision=0.5)

            novelty = 0.0
            if scene_state and isinstance(scene_state, dict):
                novelty = float(scene_state.get("novelty_recent", 0.0))
            if avg_error > 0.3:
                abstract_processor.update_beliefs(
                    {"physical_world_attention": 1.0}, precision=min(1.0, avg_error))
                self.drives.curiosity_level = min(1.0, self.drives.curiosity_level + 0.05)
                self.drives.novelty_bonus = min(
                    1.0, self.drives.novelty_bonus + 0.02 + novelty * 0.1)
            elif avg_error < 0.05 and world_errors:
                context_processor.update_beliefs({"world_stable": 1.0}, precision=0.6)

            # World-model errors (physical schemas, learned JEPA channels)
            # are context-level: record the latest magnitude so
            # get_state()["errors"] reflects every error source.
            if error_values:
                self._last_errors["context"] = round(avg_error, 3)

            return {
                "integrated_errors": len(world_errors),
                "avg_error": round(avg_error, 3),
                "world_beliefs_updated": True,
            }

    def _classify_input(self, user_input: str, level: PredictionLevel) -> str:
        text = user_input.lower().strip()
        if not text:
            return "silence"

        if level == PredictionLevel.ABSTRACT:
            if any(w in text for w in ("?", "how", "what", "why", "when", "where", "who")):
                return "question"
            if any(w in text for w in ("please", "can you", "help", "do this", "run", "create")):
                return "request"
            if any(w in text for w in ("hello", "hi", "hey", "thanks", "good", "nice")):
                return "conversation"
            if any(w in text for w in ("fix", "debug", "error", "bug", "broken")):
                return "command"
            if any(w in text for w in ("i mean", "actually", "wait", "no, ", "clarify")):
                return "clarification"
            return "conversation"

        elif level == PredictionLevel.CONTEXT:
            if len(text) > 200:
                return "detailed"
            if len(text) < 20:
                return "concise"
            if any(w in text for w in ("analyze", "compare", "evaluate", "calculate")):
                return "analytical"
            if any(w in text for w in ("imagine", "create", "design", "brainstorm")):
                return "creative"
            return "supportive"

        else:
            if any(w in text for w in ("and", "also", "more", "additionally", "further")):
                return "continuation"
            if any(w in text for w in ("but", "however", "instead", "actually")):
                return "new_topic"
            if any(w in text for w in ("explain", "detail", "elaborate", "expand")):
                return "elaboration"
            if any(w in text for w in ("no", "wrong", "incorrect", "not quite")):
                return "correction"
            return "continuation"

    def _compute_coherence(self, errors: dict) -> float:
        if not errors:
            return 1.0
        total_error = sum(e.error_magnitude for e in errors.values())
        return max(0.0, 1.0 - total_error / len(errors))

    def _compute_novelty(self, errors: dict) -> float:
        if not errors:
            return 0.0
        abstract_error = errors.get(PredictionLevel.ABSTRACT)
        if abstract_error is None:
            return 0.0
        novelty = abstract_error.error_magnitude * abstract_error.precision
        return min(1.0, novelty)


def get_predictive_processor() -> PredictiveProcessor:
    global _predictive_processor
    with _pp_lock:
        if _predictive_processor is None:
            _predictive_processor = PredictiveProcessor()
        return _predictive_processor


def reset_predictive_processor() -> None:
    global _predictive_processor
    with _pp_lock:
        _predictive_processor = None