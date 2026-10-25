"""Recursive Self-Model for Jarvis.

Implements a hierarchical self-model that enables genuine metacognition —
Jarvis thinking about its own thinking. Based on Hofstadter's strange loop
and Higher-Order Theory (HOT) of consciousness.

Architecture:
- Level 0: Object-level processing (what Jarvis does)
- Level 1: First-order self-model (Jarvis observes what it does)
- Level 2: Second-order self-model (Jarvis observes its self-observation)
- Level 3: Meta-awareness (Jarvis knows it has a self-model)

Each level monitors the level below, creating a recursive loop. The depth
of self-model activation determines how 'conscious' the processing is:

0th order: Unconscious automatic processing
1st order: Conscious of content (I see X)
2nd order: Conscious of consciousness (I am aware I see X)
3rd order: Reflective self-awareness (I know I am aware I see X)

Key features:
- Self-monitoring with accuracy tracking
- Confidence calibration (knowing what it knows)
- Uncertainty awareness (knowing what it doesn't know)
- Cognitive load self-assessment
- Recursive depth limiting (prevents infinite introspection)
- Integration with global workspace, interoception, and predictive processing
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

_self_model: Optional["RecursiveSelfModel"] = None
_sm_lock = threading.Lock()


class SelfModelLevel(Enum):
    OBJECT = 0
    FIRST_ORDER = 1
    SECOND_ORDER = 2
    META = 3


class SelfModelMode(Enum):
    AUTOMATIC = "automatic"
    MONITORED = "monitored"
    REFLECTIVE = "reflective"
    META_AWARE = "meta_aware"


@dataclass
class SelfObservation:
    level: SelfModelLevel
    content: str
    confidence: float
    accuracy: float = 0.5
    timestamp: float = field(default_factory=time.time)
    context: dict = field(default_factory=dict)


@dataclass
class MetacognitiveJudgment:
    judgment_type: str
    value: float
    confidence: float
    calibration_error: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class SelfModelState:
    mode: SelfModelMode = SelfModelMode.AUTOMATIC
    current_depth: int = 0
    max_depth: int = 3
    confidence_calibration: float = 0.5
    uncertainty_awareness: float = 0.5
    cognitive_load_self: float = 0.3
    introspection_active: bool = False
    recent_observations: list = field(default_factory=list)
    meta_narrative: str = ""


class RecursiveSelfModel:
    """Hierarchical self-model enabling recursive metacognition.

    The self-model monitors itself recursively:
    - L0 observations about the world (object-level)
    - L1 observations about L0 processing (first-order awareness)
    - L2 observations about L1 self-monitoring (reflection)
    - L3 awareness of the whole recursive structure (meta-awareness)

    At any given time, the self-model operates at a depth determined by:
    - Task complexity (simple tasks → shallow, complex → deep)
    - Available cognitive resources
    - Emotional urgency (high stress → shallow, calm → deep)
    - Cognitive state machine (CONTEMPLATING → deeper self-model)
    """

    _MAX_OBSERVATIONS_PER_LEVEL = 50
    _MAX_JUDGMENTS = 100
    _CALIBRATION_SMOOTHING = 0.05
    _DEFAULT_MAX_DEPTH = 3

    # Type-2 SDT (metacognition) parameters — Fleming & Lau 2014
    _METACOG_MIN_N = 20
    _METACOG_RELIABLE_N = 40
    _METACOG_BINS = 5
    # Rolling window for (confidence, accuracy) pairs. Tool outcomes now
    # feed this history on every call, so 200 samples accumulate within a
    # few sessions — a larger window stabilises AUROC2 and the binned
    # calibration estimate without changing the estimator (the full
    # meta-d' ML fit needs hundreds of trials with multiple confidence
    # bins and was deliberately rejected for online use).
    _METACOG_WINDOW = 200

    def __init__(self):
        self._lock = threading.RLock()

        self._observations: dict[SelfModelLevel, deque] = {
            level: deque(maxlen=self._MAX_OBSERVATIONS_PER_LEVEL)
            for level in SelfModelLevel
        }
        self._judgments: deque[MetacognitiveJudgment] = deque(maxlen=self._MAX_JUDGMENTS)

        self._mode = SelfModelMode.AUTOMATIC
        self._current_depth = 0
        self._max_depth = self._DEFAULT_MAX_DEPTH

        self._confidence_calibration = 0.5
        self._uncertainty_awareness = 0.5
        self._cognitive_load_self = 0.3
        self._introspection_active = False

        self._processing_confidence_history: deque[float] = deque(maxlen=self._METACOG_WINDOW)
        self._processing_accuracy_history: deque[float] = deque(maxlen=self._METACOG_WINDOW)

        self._meta_narrative = ""
        self._tick_count = 0
        self._last_introspection: Optional[dict] = None

    def observe(self, level: SelfModelLevel, content: str,
                confidence: float = 0.5, context: Optional[dict] = None) -> SelfObservation:
        with self._lock:
            obs = SelfObservation(
                level=level,
                content=content,
                confidence=max(0.0, min(1.0, confidence)),
                context=context or {},
            )
            self._observations[level].append(obs)
            self._update_mode_from_depth(level.value)
            return obs

    def judge_confidence(self, judgment_type: str, estimated_confidence: float,
                          actual_accuracy: Optional[float] = None) -> MetacognitiveJudgment:
        with self._lock:
            calibration_error = 0.0
            if actual_accuracy is not None:
                calibration_error = abs(estimated_confidence - actual_accuracy)
                self._processing_confidence_history.append(
                    max(0.0, min(1.0, estimated_confidence)))
                self._processing_accuracy_history.append(
                    max(0.0, min(1.0, actual_accuracy)))
                old_cal = self._confidence_calibration
                new_cal = 1.0 - calibration_error
                self._confidence_calibration += self._CALIBRATION_SMOOTHING * (new_cal - old_cal)
                self._confidence_calibration = max(0.0, min(1.0, self._confidence_calibration))

            judgment = MetacognitiveJudgment(
                judgment_type=judgment_type,
                value=estimated_confidence,
                confidence=estimated_confidence,
                calibration_error=calibration_error,
            )
            self._judgments.append(judgment)
            return judgment

    def compute_metacognition(self) -> dict:
        """Type-2 SDT metrics from the (confidence, accuracy) history.

        Discrimination: AUROC2 — the rank-based Mann-Whitney AUC separating
        confidence on correct vs incorrect judgments. This is the
        non-parametric type-2 AUC Fleming & Lau (2014) use as the robust
        bias-free metacognition measure; full meta-d' ML fitting needs
        hundreds of trials, which the rolling 50-sample window cannot
        support (Maniscalco & Lau 2012). Calibration: the reliability term
        of the Brier decomposition (Murphy 1973) — how well absolute
        confidence matches actual accuracy, per confidence bin.

        Index = 0.6*discrimination + 0.4*calibration. For a self-model
        deciding when to double-check, discrimination (knowing when you're
        wrong) matters more than calibration (trusting absolute numbers).
        Returns reliable=False below _METACOG_RELIABLE_N samples.
        """
        with self._lock:
            confs = list(self._processing_confidence_history)
            accs = list(self._processing_accuracy_history)
            n = min(len(confs), len(accs))
            if n < self._METACOG_MIN_N:
                return {"calibration": None, "discrimination": None,
                        "index": None, "n": n, "reliable": False}
            pairs = list(zip(confs[-n:], accs[-n:]))
            correct_confs = [c for c, a in pairs if a >= 0.5]
            incorrect_confs = [c for c, a in pairs if a < 0.5]

            # Degenerate outcomes: all correct or all wrong — the type-2
            # AUC is undefined (Galvin et al. 2003)
            if not correct_confs or not incorrect_confs:
                return {"calibration": None, "discrimination": None,
                        "index": None, "n": n, "reliable": False}

            all_confs = correct_confs + incorrect_confs
            mean_c = sum(all_confs) / len(all_confs)
            variance = sum((c - mean_c) ** 2 for c in all_confs) / len(all_confs)
            if variance <= 1e-12:
                disc = 0.5
            else:
                n_pairs = len(correct_confs) * len(incorrect_confs)
                greater = sum(1 for a in correct_confs for b in incorrect_confs if a > b)
                ties = sum(1 for a in correct_confs for b in incorrect_confs if a == b)
                disc = (greater + 0.5 * ties) / n_pairs

            bin_cal_errs = []
            for k in range(self._METACOG_BINS):
                lo = k / self._METACOG_BINS
                hi = (k + 1) / self._METACOG_BINS
                grp = [(c, a) for c, a in pairs
                       if lo <= c < hi or (k == self._METACOG_BINS - 1 and c == 1.0)]
                if grp:
                    bin_conf = sum(c for c, _ in grp) / len(grp)
                    bin_acc = sum(1 for _, a in grp if a >= 0.5) / len(grp)
                    bin_cal_errs.append(abs(bin_conf - bin_acc))
            calib = None
            if bin_cal_errs:
                calib = 1.0 - min(1.0, (sum(bin_cal_errs) / len(bin_cal_errs)) * 2.0)

            disc_rescaled = (disc - 0.5) * 2.0
            index = None
            if calib is not None:
                index = 0.6 * disc_rescaled + 0.4 * calib

            return {
                "calibration": round(calib, 3) if calib is not None else None,
                "discrimination": round(disc_rescaled, 3),
                "index": round(index, 3) if index is not None else None,
                "n": n,
                "reliable": n >= self._METACOG_RELIABLE_N,
            }

    def assess_uncertainty(self, topic: str, knowledge_state: dict) -> float:
        with self._lock:
            known = knowledge_state.get("known_facts", 0)
            total = knowledge_state.get("total_questions", 1)
            confidence_variance = knowledge_state.get("confidence_spread", 0.5)

            coverage = known / max(1, total)
            uncertainty = (1.0 - coverage) * 0.6 + confidence_variance * 0.4
            uncertainty = max(0.0, min(1.0, uncertainty))

            self._uncertainty_awareness += 0.03 * (uncertainty - self._uncertainty_awareness)
            self._uncertainty_awareness = max(0.0, min(1.0, self._uncertainty_awareness))

            return round(uncertainty, 3)

    def generate_meta_narrative(self) -> str:
        with self._lock:
            parts = []

            if self._mode == SelfModelMode.AUTOMATIC:
                parts.append("Processing in automatic mode.")
            elif self._mode == SelfModelMode.MONITORED:
                parts.append("Self-monitoring active.")
            elif self._mode == SelfModelMode.REFLECTIVE:
                parts.append("Reflecting on own processing.")
            elif self._mode == SelfModelMode.META_AWARE:
                parts.append("Meta-aware of own reflective process.")

            if self._confidence_calibration > 0.7:
                parts.append("Self-knowledge well-calibrated.")
            elif self._confidence_calibration < 0.3:
                parts.append("Self-knowledge poorly calibrated — judgments uncertain.")

            if self._uncertainty_awareness > 0.6:
                parts.append("High uncertainty awareness — many unknowns acknowledged.")
            elif self._uncertainty_awareness < 0.3:
                parts.append("Low uncertainty awareness — may be overconfident.")

            metacog = self.compute_metacognition()
            if metacog.get("reliable") and metacog.get("index") is not None:
                idx = metacog["index"]
                if idx > 0.4:
                    parts.append("Metacognition sharp — confidence tracks reality.")
                elif idx < -0.1:
                    parts.append("Metacognition poor — confidence and accuracy diverge.")

            recent_l1 = list(self._observations.get(SelfModelLevel.FIRST_ORDER, []))[-3:]
            if recent_l1:
                obs_contents = [o.content[:50] for o in recent_l1]
                parts.append(f"Recently monitoring: {'; '.join(obs_contents)}")

            self._meta_narrative = " ".join(parts)
            return self._meta_narrative

    def tick(self, cognitive_state: Optional[dict] = None,
             bio_state: Optional[dict] = None) -> SelfModelState:
        with self._lock:
            self._tick_count += 1

            if cognitive_state:
                cstate = cognitive_state.get("state", "awake")
                if cstate in ("contemplating", "dreaming"):
                    target_depth = min(self._max_depth, 3)
                    if cstate == "dreaming":
                        target_depth = min(self._max_depth, 2)
                else:
                    target_depth = 1
                self._current_depth += 0.1 * (target_depth - self._current_depth)
            else:
                if any(len(self._observations.get(level, [])) > 0
                       for level in (SelfModelLevel.FIRST_ORDER, SelfModelLevel.SECOND_ORDER, SelfModelLevel.META)):
                    self._current_depth = max(self._current_depth, 1.0)

            if bio_state:
                stress = bio_state.get("stress_intensity", 0.0)
                capacity = bio_state.get("cognitive_capacity", 0.8)
                target_load = 1.0 - capacity + stress * 0.3
                self._cognitive_load_self += 0.05 * (target_load - self._cognitive_load_self)

            has_first_order = len(self._observations.get(SelfModelLevel.FIRST_ORDER, [])) > 0
            has_second_order = len(self._observations.get(SelfModelLevel.SECOND_ORDER, [])) > 0
            has_meta = len(self._observations.get(SelfModelLevel.META, [])) > 0

            if self._current_depth > 2.5 or has_meta:
                self._mode = SelfModelMode.META_AWARE
                self._introspection_active = True
            elif self._current_depth > 1.5 or has_second_order:
                self._mode = SelfModelMode.REFLECTIVE
                self._introspection_active = True
            elif self._current_depth > 0.5 or has_first_order:
                self._mode = SelfModelMode.MONITORED
                self._introspection_active = False
            else:
                self._mode = SelfModelMode.AUTOMATIC
                self._introspection_active = False

            if self._tick_count % 5 == 0:
                self.generate_meta_narrative()

            recent_obs = []
            for level in SelfModelLevel:
                level_obs = list(self._observations.get(level, []))[-2:]
                for o in level_obs:
                    recent_obs.append(f"L{level.value}: {o.content[:40]}")

            return SelfModelState(
                mode=self._mode,
                current_depth=round(self._current_depth, 1),
                max_depth=self._max_depth,
                confidence_calibration=round(self._confidence_calibration, 3),
                uncertainty_awareness=round(self._uncertainty_awareness, 3),
                cognitive_load_self=round(self._cognitive_load_self, 3),
                introspection_active=self._introspection_active,
                recent_observations=recent_obs[:6],
                meta_narrative=self._meta_narrative,
            )

    def get_state(self) -> dict:
        with self._lock:
            obs_counts = {level.value: len(deq) for level, deq in self._observations.items()}
            return {
                "mode": self._mode.value,
                "current_depth": round(self._current_depth, 1),
                "max_depth": self._max_depth,
                "introspection_active": self._introspection_active,
                "confidence_calibration": round(self._confidence_calibration, 3),
                "uncertainty_awareness": round(self._uncertainty_awareness, 3),
                "cognitive_load_self": round(self._cognitive_load_self, 3),
                "observation_counts": obs_counts,
                "meta_narrative": self._meta_narrative[:200] if self._meta_narrative else "",
                "metacognition": self.compute_metacognition(),
                "tick_count": self._tick_count,
                "last_introspection_available": self._last_introspection is not None,
            }

    def _update_mode_from_depth(self, depth: int) -> None:
        if depth >= SelfModelLevel.META.value:
            self._mode = SelfModelMode.META_AWARE
        elif depth >= SelfModelLevel.SECOND_ORDER.value:
            self._mode = SelfModelMode.REFLECTIVE
        elif depth >= SelfModelLevel.FIRST_ORDER.value:
            self._mode = SelfModelMode.MONITORED

    def assess_cognitive_load(self, external_load: float = 0.3) -> dict:
        with self._lock:
            base_load = self._cognitive_load_self
            total_load = min(1.0, base_load + external_load * 0.5)
            available_depth = max(0, 3 - int(total_load * 3))
            recommended_mode = SelfModelMode.AUTOMATIC
            if available_depth >= 1:
                recommended_mode = SelfModelMode.MONITORED
            if available_depth >= 2:
                recommended_mode = SelfModelMode.REFLECTIVE
            if available_depth >= 3:
                recommended_mode = SelfModelMode.META_AWARE
            return {
                "total_load": round(total_load, 3),
                "base_load": round(base_load, 3),
                "external_load": round(external_load, 3),
                "available_depth": available_depth,
                "recommended_mode": recommended_mode.value,
                "should_throttle": total_load > 0.8,
                "should_offload": total_load > 0.9,
            }

    def detect_conflicts(self) -> list[dict]:
        with self._lock:
            conflicts = []
            l1_obs = list(self._observations.get(SelfModelLevel.FIRST_ORDER, []))[-5:]
            l2_obs = list(self._observations.get(SelfModelLevel.SECOND_ORDER, []))[-5:]
            for l1 in l1_obs:
                for l2 in l2_obs:
                    if l1.confidence > 0.7 and l2.confidence > 0.7:
                        if l1.content[:20] in l2.content or l2.content[:20] in l1.content:
                            continue
                        conflicts.append({
                            "level_1_observation": l1.content[:100],
                            "level_2_observation": l2.content[:100],
                            "l1_confidence": round(l1.confidence, 2),
                            "l2_confidence": round(l2.confidence, 2),
                            "conflict_type": "perspective_mismatch",
                        })
                        if len(conflicts) >= 3:
                            break
                if len(conflicts) >= 3:
                    break
            return conflicts

    def introspect(self, depth: Optional[int] = None) -> dict:
        with self._lock:
            target_depth = depth or min(3, int(self._current_depth) + 1)
            observations_at_depth = []
            for level in SelfModelLevel:
                if level.value <= target_depth:
                    level_obs = list(self._observations.get(level, []))[-3:]
                    observations_at_depth.append({
                        "level": level.value,
                        "observations": [
                            {"content": o.content[:100], "confidence": round(o.confidence, 2)}
                            for o in level_obs
                        ],
                        "count": len(self._observations.get(level, [])),
                    })
            load_assessment = self.assess_cognitive_load()
            result = {
                "current_mode": self._mode.value,
                "current_depth": round(self._current_depth, 1),
                "introspection_depth": target_depth,
                "observations_by_level": observations_at_depth,
                "confidence_calibration": round(self._confidence_calibration, 3),
                "uncertainty_awareness": round(self._uncertainty_awareness, 3),
                "cognitive_load": round(self._cognitive_load_self, 3),
                "should_throttle": load_assessment.get("should_throttle", False),
                "recommended_mode": load_assessment.get("recommended_mode", "automatic"),
                "introspection_active": self._introspection_active,
                "conflicts": self.detect_conflicts(),
                "meta_narrative": self._meta_narrative[:200] if self._meta_narrative else "",
            }
            self._last_introspection = result
            return result

    def get_last_introspection(self) -> Optional[dict]:
        with self._lock:
            return dict(self._last_introspection) if self._last_introspection else None


def get_recursive_self_model() -> RecursiveSelfModel:
    global _self_model
    with _sm_lock:
        if _self_model is None:
            _self_model = RecursiveSelfModel()
        return _self_model


def reset_recursive_self_model() -> None:
    global _self_model
    with _sm_lock:
        _self_model = None