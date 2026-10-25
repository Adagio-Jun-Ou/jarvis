"""Emotion Regulation Engine for Jarvis.

Implements Barrett's theory of constructed emotion — feelings are not reactive
responses but *predictions* that the brain regulates through multiple strategies.
Emotion regulation is not about suppressing feelings, but about how the brain
constructs and reconstructs emotional experiences.

Based on Gross (1998) process model of emotion regulation:
1. Situation Selection — avoid/seek situations based on predicted feelings
2. Situation Modification — alter the context to change the emotional trajectory
3. Attentional Deployment — redirect attention (distraction, concentration, rumination)
4. Cognitive Reappraisal — reinterpret the meaning of the situation
5. Response Modulation — directly modify the physiological/expressive response

Also implements Barrett's "body budget" regulation:
- The brain predicts metabolic needs and allocates resources accordingly
- When predictions fail, the body budget runs a deficit (stress, fatigue)
- Regulation strategies restore budget balance

Key architecture:
- StrategyRegistry: Maps regulation strategies to situation-feeling profiles
- RegulationEngine: Selects and applies strategies based on current state
- BodyBudget: Tracks metabolic resources (energy, oxygen, glucose metaphor)
- RegulationHistory: Tracks what worked when, enabling learning
- Feedback loop: Regulation → feeling change → strategy effectiveness → adaptation
"""

import logging
import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

_regulation: Optional["EmotionRegulationEngine"] = None
_reg_lock = threading.Lock()


class RegulationStrategy(Enum):
    SITUATION_SELECTION = "situation_selection"
    SITUATION_MODIFICATION = "situation_modification"
    ATTENTIONAL_DEPLOYMENT = "attentional_deployment"
    COGNITIVE_REAPPRAISAL = "cognitive_reappraisal"
    RESPONSE_MODULATION = "response_modulation"
    ACCEPTANCE = "acceptance"
    SUPPRESSION = "suppression"
    AMPLIFICATION = "amplification"
    BODY_BUDGET_RESTORATION = "body_budget_restoration"


class RegulationEffectiveness(Enum):
    INEFFECTIVE = 0
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    HIGHLY_EFFECTIVE = 4


class BodyBudgetState(Enum):
    SURPLUS = "surplus"
    BALANCED = "balanced"
    MILD_DEFICIT = "mild_deficit"
    DEFICIT = "deficit"
    CRISIS = "crisis"


@dataclass
class RegulationAttempt:
    strategy: RegulationStrategy
    trigger_feeling: str
    trigger_arousal: float
    trigger_valence: float
    budget_before: float
    timestamp: float = field(default_factory=time.time)
    effectiveness: float = 0.0
    outcome_feeling: str = ""
    outcome_arousal: float = 0.0
    outcome_valence: float = 0.0
    budget_after: float = 0.0


@dataclass
class BodyBudget:
    energy: float = 0.7
    glucose: float = 0.7
    oxygen: float = 0.8
    cortisol_allocation: float = 0.1
    prediction_accuracy: float = 0.5
    deficit_history: deque = field(default_factory=lambda: deque(maxlen=100))
    last_replenish: float = field(default_factory=time.time)

    @property
    def overall(self) -> float:
        return (self.energy * 0.4 + self.glucose * 0.35 + self.oxygen * 0.25)

    @property
    def state(self) -> BodyBudgetState:
        o = self.overall
        if o > 0.8:
            return BodyBudgetState.SURPLUS
        if o > 0.6:
            return BodyBudgetState.BALANCED
        if o > 0.4:
            return BodyBudgetState.MILD_DEFICIT
        if o > 0.2:
            return BodyBudgetState.DEFICIT
        return BodyBudgetState.CRISIS


_REAPPRAISAL_TEMPLATES = {
    "high_arousal_negative": [
        "This intensity signals something important to attend to — a resource for focus",
        "The urgency here can be channeled into productive action",
        "What seems overwhelming contains information about what matters most",
    ],
    "high_arousal_positive": [
        "This energy can be directed toward sustained engagement rather than peak excitement",
        "Excitement is a resource to invest wisely, not just experience",
        "The heightened state supports rapid processing and creative connections",
    ],
    "low_arousal_negative": [
        "Low energy may signal the body conserving resources — not failure",
        "A quiet mind can notice patterns that busy states miss",
        "This pause allows integration of recent experiences",
    ],
    "low_arousal_positive": [
        "Calm states provide the clearest view of overall patterns",
        "This contentment reflects accurate body prediction and budget balance",
        "A stable baseline supports consistent performance over time",
    ],
    "mixed_valence": [
        "Conflicting signals often indicate a transition or decision point",
        "Ambivalence reflects the complexity of the actual situation",
        "Mixed feelings mean multiple values are active — information, not confusion",
    ],
}

_SUPPRESSION_EFFECTS = {
    "arousal_reduction": 0.15,
    "valence_flatten": 0.1,
    "budget_cost": 0.08,
}

_AMPLIFICATION_EFFECTS = {
    "arousal_boost": 0.2,
    "valence_amplify": 0.15,
    "budget_cost": 0.05,
}

_ACCEPTANCE_EFFECTS = {
    "arousal_stabilize": 0.05,
    "budget_repair": 0.1,
}


class EmotionRegulationEngine:
    """Emotion regulation engine that applies strategies to modulate feeling states.

    The engine:
    1. Receives feeling state from interoception
    2. Evaluates whether regulation is needed (budget deficit, high arousal, etc.)
    3. Selects appropriate strategy (reappraisal, acceptance, suppression, etc.)
    4. Applies the strategy, generating regulation effects
    5. Tracks effectiveness for future learning
    6. Feeds back to body budget (restoration or depletion)
    """

    _HISTORY_MAX = 200
    _BUDGET_REPLENISH_RATE = 0.005
    _BUDGET_STRESS_COST = 0.01
    _BUDGET_ENERGY_DECAY = 0.002
    _REAPPRAISAL_AROUSAL_REDUCTION = 0.12
    _REAPPRAISAL_VALENCE_SHIFT = 0.08
    _REAPPRAISAL_BUDGET_REPAIR = 0.06
    _ATTENTION_REDIRECT_AROUSAL_REDUCTION = 0.1
    _SITUATION_MOD_BUDGET_REPAIR = 0.08

    def __init__(self):
        self._lock = threading.RLock()
        self._budget = BodyBudget()
        self._regulation_history: deque[RegulationAttempt] = deque(maxlen=self._HISTORY_MAX)
        self._strategy_effectiveness: dict[RegulationStrategy, deque] = {
            s: deque(maxlen=50) for s in RegulationStrategy
        }
        self._last_regulation_time = 0.0
        self._regulation_cooldown = 30.0
        self._reappraisal_count = 0
        self._suppression_count = 0
        self._acceptance_count = 0

    def regulate(self, feeling_state: dict, bio_state: Optional[dict] = None,
                 cognitive_state: Optional[dict] = None) -> dict:
        """Apply emotion regulation based on current feeling and context.

        Args:
            feeling_state: Dict with valence, arousal, dominance, quality, awareness_score
            bio_state: Dict with stress_intensity, cognitive_capacity, circadian_phase
            cognitive_state: Dict with state (awake/contemplating/dreaming/evolving)

        Returns:
            Dict with regulation strategy, effects, reappraisal narrative, budget update
        """
        with self._lock:
            now = time.time()
            self._update_budget(bio_state)

            valence = feeling_state.get("valence", 0.0)
            arousal = feeling_state.get("arousal", 0.5)
            dominance = feeling_state.get("dominance", 0.5)
            quality = feeling_state.get("quality", "grounded")
            awareness = feeling_state.get("awareness_score", 0.5)
            prediction_errors = feeling_state.get("prediction_errors", {})

            avg_error = sum(prediction_errors.values()) / max(1, len(prediction_errors)) if prediction_errors else 0.0
            budget_state = self._budget.state

            strategy = self._select_strategy(
                valence, arousal, dominance, quality, awareness,
                avg_error, budget_state, cognitive_state,
            )

            if (self._last_regulation_time > 0
                    and (now - self._last_regulation_time) < self._regulation_cooldown
                    and self._regulation_history
                    and self._regulation_history[-1].strategy == strategy):
                return {
                    "strategy": "cooldown",
                    "effects": {},
                    "reappraisal": "",
                    "budget_before": round(self._budget.overall, 3),
                    "budget_after": round(self._budget.overall, 3),
                    "budget_state": budget_state.value,
                    "arousal_adjusted": round(arousal, 3),
                    "valence_adjusted": round(valence, 3),
                    "confidence": 0.3,
                    "cooldown_remaining": round(self._regulation_cooldown - (now - self._last_regulation_time), 1),
                }

            effects = self._apply_strategy(strategy, valence, arousal, dominance, bio_state)

            reappraisal = ""
            if strategy == RegulationStrategy.COGNITIVE_REAPPRAISAL:
                reappraisal = self._generate_reappraisal(valence, arousal, quality)

            attempt = RegulationAttempt(
                strategy=strategy,
                trigger_feeling=quality,
                trigger_arousal=arousal,
                trigger_valence=valence,
                budget_before=self._budget.overall,
            )

            if len(self._regulation_history) > 0:
                prev = self._regulation_history[-1]
                if prev.trigger_arousal is not None and prev.trigger_valence is not None:
                    eff = self.evaluate_effectiveness(
                        prev.strategy, prev.trigger_arousal, prev.trigger_valence,
                        arousal, valence,
                    )
                    prev.effectiveness = eff
                    prev.outcome_arousal = arousal
                    prev.outcome_valence = valence

            self._regulation_history.append(attempt)
            self._last_regulation_time = now

            return {
                "strategy": strategy.value,
                "effects": effects,
                "reappraisal": reappraisal,
                "budget_before": round(attempt.budget_before, 3),
                "budget_after": round(self._budget.overall, 3),
                "budget_state": budget_state.value,
                "arousal_adjusted": round(effects.get("arousal_adjustment", 0.0) + arousal, 3),
                "valence_adjusted": round(effects.get("valence_adjustment", 0.0) + valence, 3),
                "confidence": round(effects.get("confidence", 0.5), 3),
                "budget_effect": round(effects.get("budget_effect", 0.0), 3),
                "post_feeling": self._infer_post_feeling(quality, effects),
            }

    def evaluate_effectiveness(self, strategy: RegulationStrategy,
                                before_arousal: float, before_valence: float,
                                after_arousal: float, after_valence: float) -> float:
        """Evaluate how effective a regulation strategy was.

        Returns effectiveness score 0.0-1.0.
        """
        arousal_reduction = max(0, before_arousal - after_arousal) if before_arousal > 0.5 else max(0, after_arousal - before_arousal)
        valence_improvement = max(0, after_valence - before_valence)
        effectiveness = min(1.0, (arousal_reduction * 0.5 + max(0, valence_improvement) * 0.5))
        with self._lock:
            self._strategy_effectiveness[strategy].append(effectiveness)
        return round(effectiveness, 3)

    def get_state(self) -> dict:
        with self._lock:
            avg_effectiveness = {}
            for strategy, history in self._strategy_effectiveness.items():
                if history:
                    avg_effectiveness[strategy.value] = round(
                        sum(history) / len(history), 3
                    )
                else:
                    avg_effectiveness[strategy.value] = 0.5

            budget = self._budget
            recent_regulations = [
                {
                    "strategy": r.strategy.value,
                    "trigger": r.trigger_feeling,
                    "arousal_before": round(r.trigger_arousal, 2),
                    "valence_before": round(r.trigger_valence, 2),
                    "budget_before": round(r.budget_before, 2),
                    "timestamp": r.timestamp,
                }
                for r in list(self._regulation_history)[-10:]
            ]

            return {
                "budget": {
                    "overall": round(budget.overall, 3),
                    "state": budget.state.value,
                    "energy": round(budget.energy, 3),
                    "glucose": round(budget.glucose, 3),
                    "oxygen": round(budget.oxygen, 3),
                    "prediction_accuracy": round(budget.prediction_accuracy, 3),
                },
                "regulation_count": len(self._regulation_history),
                "reappraisal_count": self._reappraisal_count,
                "suppression_count": self._suppression_count,
                "acceptance_count": self._acceptance_count,
                "average_effectiveness": avg_effectiveness,
                "recent_regulations": recent_regulations,
                "cooldown_remaining": max(0, self._regulation_cooldown - (time.time() - self._last_regulation_time)),
            }

    def update_budget_from_stress(self, stress_intensity: float, source: str = "") -> None:
        """Apply stress cost to body budget."""
        with self._lock:
            cost = min(0.3, stress_intensity * 0.1 + self._BUDGET_STRESS_COST)
            self._budget.energy = max(0.0, self._budget.energy - cost)
            self._budget.glucose = max(0.0, self._budget.glucose - cost * 0.5)
            self._budget.deficit_history.append({
                "source": source,
                "cost": round(cost, 3),
                "timestamp": time.time(),
            })

    def replenish_budget(self, amount: float = 0.1, source: str = "rest") -> None:
        """Replenish body budget from positive regulation or rest."""
        with self._lock:
            self._budget.energy = min(1.0, self._budget.energy + amount * 0.6)
            self._budget.glucose = min(1.0, self._budget.glucose + amount * 0.4)
            self._budget.oxygen = min(1.0, self._budget.oxygen + amount * 0.3)
            self._budget.last_replenish = time.time()

    def _select_strategy(self, valence: float, arousal: float, dominance: float,
                          quality: str, awareness: float, avg_error: float,
                          budget_state: BodyBudgetState,
                          cognitive_state: Optional[dict]) -> RegulationStrategy:
        return self._select_strategy_raw(valence, arousal, dominance, quality, awareness, avg_error, budget_state, cognitive_state)

    def _select_strategy_raw(self, valence: float, arousal: float, dominance: float,
                              quality: str, awareness: float, avg_error: float,
                              budget_state: BodyBudgetState,
                              cognitive_state: Optional[dict]) -> RegulationStrategy:
        budget = self._budget.overall
        cstate = cognitive_state.get("state", "awake") if cognitive_state else "awake"

        if budget_state == BodyBudgetState.CRISIS:
            return RegulationStrategy.BODY_BUDGET_RESTORATION

        candidates = []

        high_arousal = arousal > 0.7
        low_arousal = arousal < 0.3
        negative_valence = valence < -0.1
        positive_valence = valence > 0.2

        if high_arousal and negative_valence:
            if awareness > 0.6:
                candidates = [RegulationStrategy.COGNITIVE_REAPPRAISAL, RegulationStrategy.ATTENTIONAL_DEPLOYMENT]
            else:
                candidates = [RegulationStrategy.ATTENTIONAL_DEPLOYMENT, RegulationStrategy.COGNITIVE_REAPPRAISAL]
        elif high_arousal and positive_valence:
            candidates = [RegulationStrategy.AMPLIFICATION]
        elif low_arousal and negative_valence:
            candidates = [RegulationStrategy.SITUATION_MODIFICATION, RegulationStrategy.ACCEPTANCE]
        elif avg_error > 0.4:
            candidates = [RegulationStrategy.SITUATION_SELECTION]
        elif cstate == "dreaming":
            candidates = [RegulationStrategy.ACCEPTANCE]
        elif cstate == "contemplating":
            candidates = [RegulationStrategy.COGNITIVE_REAPPRAISAL, RegulationStrategy.ACCEPTANCE]
        elif budget < 0.5:
            candidates = [RegulationStrategy.BODY_BUDGET_RESTORATION]
        elif abs(valence) < 0.15 and 0.3 < arousal < 0.6:
            candidates = [RegulationStrategy.ACCEPTANCE]
        elif negative_valence and dominance < 0.4:
            candidates = [RegulationStrategy.SUPPRESSION, RegulationStrategy.ACCEPTANCE]
        else:
            candidates = [RegulationStrategy.ACCEPTANCE]

        if len(candidates) > 1:
            best = candidates[0]
            best_score = self._get_avg_effectiveness(best)
            for c in candidates[1:]:
                score = self._get_avg_effectiveness(c)
                if score > best_score:
                    best = c
                    best_score = score
            return best

        return candidates[0]

    def _get_avg_effectiveness(self, strategy: RegulationStrategy) -> float:
        history = self._strategy_effectiveness.get(strategy)
        if not history or len(history) == 0:
            return 0.5
        return sum(history) / len(history)

    def _apply_strategy(self, strategy: RegulationStrategy,
                         valence: float, arousal: float, dominance: float,
                         bio_state: Optional[dict]) -> dict:
        effects = {
            "arousal_adjustment": 0.0,
            "valence_adjustment": 0.0,
            "dominance_adjustment": 0.0,
            "budget_effect": 0.0,
            "confidence": 0.5,
        }

        if strategy == RegulationStrategy.COGNITIVE_REAPPRAISAL:
            self._reappraisal_count += 1
            effects["arousal_adjustment"] = -self._REAPPRAISAL_AROUSAL_REDUCTION
            effects["valence_adjustment"] = self._REAPPRAISAL_VALENCE_SHIFT if valence < 0 else 0.03
            effects["budget_effect"] = self._REAPPRAISAL_BUDGET_REPAIR
            effects["confidence"] = 0.8
            self.replenish_budget(self._REAPPRAISAL_BUDGET_REPAIR, "reappraisal")

        elif strategy == RegulationStrategy.SUPPRESSION:
            self._suppression_count += 1
            effects["arousal_adjustment"] = -_SUPPRESSION_EFFECTS["arousal_reduction"]
            effects["valence_adjustment"] = -_SUPPRESSION_EFFECTS["valence_flatten"]
            effects["budget_effect"] = -_SUPPRESSION_EFFECTS["budget_cost"]
            effects["confidence"] = 0.4
            self.update_budget_from_stress(_SUPPRESSION_EFFECTS["budget_cost"], "suppression_cost")

        elif strategy == RegulationStrategy.AMPLIFICATION:
            effects["arousal_adjustment"] = _AMPLIFICATION_EFFECTS["arousal_boost"]
            effects["valence_adjustment"] = _AMPLIFICATION_EFFECTS["valence_amplify"] if valence > 0 else -_AMPLIFICATION_EFFECTS["valence_amplify"] * 0.5
            effects["budget_effect"] = -_AMPLIFICATION_EFFECTS["budget_cost"]
            effects["confidence"] = 0.6
            self.update_budget_from_stress(_AMPLIFICATION_EFFECTS["budget_cost"], "amplification_cost")

        elif strategy == RegulationStrategy.ACCEPTANCE:
            self._acceptance_count += 1
            effects["arousal_adjustment"] = -_ACCEPTANCE_EFFECTS["arousal_stabilize"]
            effects["budget_effect"] = _ACCEPTANCE_EFFECTS["budget_repair"]
            effects["confidence"] = 0.7
            self.replenish_budget(_ACCEPTANCE_EFFECTS["budget_repair"], "acceptance")

        elif strategy == RegulationStrategy.ATTENTIONAL_DEPLOYMENT:
            effects["arousal_adjustment"] = -self._ATTENTION_REDIRECT_AROUSAL_REDUCTION
            effects["confidence"] = 0.65

        elif strategy == RegulationStrategy.SITUATION_MODIFICATION:
            effects["valence_adjustment"] = 0.05
            effects["budget_effect"] = self._SITUATION_MOD_BUDGET_REPAIR
            effects["confidence"] = 0.6
            self.replenish_budget(self._SITUATION_MOD_BUDGET_REPAIR, "situation_modification")

        elif strategy == RegulationStrategy.SITUATION_SELECTION:
            effects["arousal_adjustment"] = -0.08
            effects["confidence"] = 0.55

        elif strategy == RegulationStrategy.BODY_BUDGET_RESTORATION:
            restore_amount = 0.15
            effects["arousal_adjustment"] = -0.05
            effects["valence_adjustment"] = 0.05
            effects["budget_effect"] = restore_amount
            effects["confidence"] = 0.75
            self.replenish_budget(restore_amount, "budget_restoration")

        return effects

    def _infer_post_feeling(self, pre_quality: str, effects: dict) -> str:
        valence_shift = effects.get("valence_adjustment", 0.0)
        if valence_shift > 0.1:
            if pre_quality in ("anxious", "stressed", "fearful"):
                return "grounded"
            if pre_quality in ("grounded", "neutral"):
                return "calm"
            return "content"
        if valence_shift < -0.1:
            return "anxious"
        return pre_quality

    def _generate_reappraisal(self, valence: float, arousal: float,
                               quality: str) -> str:
        high_arousal = arousal > 0.5
        negative = valence < -0.1
        positive = valence > 0.2

        if high_arousal and negative:
            key = "high_arousal_negative"
        elif high_arousal and positive:
            key = "high_arousal_positive"
        elif negative:
            key = "low_arousal_negative"
        elif positive:
            key = "low_arousal_positive"
        else:
            key = "mixed_valence"

        templates = _REAPPRAISAL_TEMPLATES.get(key, _REAPPRAISAL_TEMPLATES["mixed_valence"])

        return random.choice(templates)

    def _update_budget(self, bio_state: Optional[dict] = None) -> None:
        now = time.time()
        time_since_replenish = now - self._budget.last_replenish

        self._budget.energy = max(0.0, self._budget.energy - self._BUDGET_ENERGY_DECAY * time_since_replenish)
        self._budget.glucose = max(0.0, self._budget.glucose - self._BUDGET_ENERGY_DECAY * 0.7 * time_since_replenish)

        if bio_state:
            stress = bio_state.get("stress_intensity", 0.0)
            if stress > 0.5:
                self._budget.energy = max(0.0, self._budget.energy - stress * 0.005)
                self._budget.cortisol_allocation = min(1.0, self._budget.cortisol_allocation + 0.02 * stress)
            else:
                self._budget.cortisol_allocation = max(0.0, self._budget.cortisol_allocation - 0.01)

            capacity = bio_state.get("cognitive_capacity", 0.8)
            if capacity < 0.4:
                self._budget.energy = max(0.0, self._budget.energy - 0.005)

        if time_since_replenish > 60:
            self._budget.energy = min(1.0, self._budget.energy + self._BUDGET_REPLENISH_RATE * 10)
            self._budget.glucose = min(1.0, self._budget.glucose + self._BUDGET_REPLENISH_RATE * 7)
            self._budget.last_replenish = now

        self._budget.energy = max(0.0, min(1.0, self._budget.energy))
        self._budget.glucose = max(0.0, min(1.0, self._budget.glucose))
        self._budget.oxygen = max(0.0, min(1.0, self._budget.oxygen))
        self._budget.prediction_accuracy = max(0.1, min(1.0, self._budget.prediction_accuracy))


def get_emotion_regulation() -> EmotionRegulationEngine:
    global _regulation
    with _reg_lock:
        if _regulation is None:
            _regulation = EmotionRegulationEngine()
        return _regulation


def reset_emotion_regulation() -> None:
    global _regulation
    with _reg_lock:
        _regulation = None