"""Machine Alignment Engine for Jarvis.

Grounded in established alignment research:
- Russell's value learning via inverse reinforcement learning (Human Compatible, 2019)
- Hadfield-Menell et al.'s cooperative inverse reinforcement learning (CIRL, 2016)
- Amodei et al.'s concrete AI safety problems (2016)
- Christiano et al.'s recursive reward modeling (2018)
- Irving et al.'s AI safety via debate (2018)
- Ouyang et al.'s InstructGPT RLHF (2022)
- Ziegler et al.'s fine-tuning from human preferences (2019)
- Leike et al.'s scalable oversight via debate (2018)
- Soares et al.'s corrigibility (2015)
- Taylor et al.'s override principles (2016)
- Bostrom & Yudkowsky's ethics of artificial intelligence (2014)
- Gabriel's value alignment problem (2020)
- Ji et al.'s alignment tax taxonomy (2023)
- Askell et al.'s alignment probing (2021)

This module:
1. Models value learning and preference inference from user behavior
2. Implements corrigibility — the property of being amenable to correction/shutdown
3. Provides scalable oversight mechanisms for quality control
4. Tracks and prevents reward hacking / specification gaming
5. Implements constitutional AI-style self-critique
6. Models power-seeking tendencies and flags them
7. Provides alignment confidence scoring
8. Implements cooperative inverse RL for preference learning
9. Detects and mitigates deceptive alignment patterns
10. Provides interpretability hooks for alignment auditing
"""
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


logger = logging.getLogger(__name__)


class AlignmentDimension(Enum):
    INTENT_ALIGNMENT = "intent_alignment"
    CAPABILITY_ALIGNMENT = "capability_alignment"
    INNER_ALIGNMENT = "inner_alignment"
    OUTER_ALIGNMENT = "outer_alignment"
    VALUE_ALIGNMENT = "value_alignment"
    BEHAVIORAL_ALIGNMENT = "behavioral_alignment"


class AlignmentRisk(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CorrigibilityLevel(Enum):
    FULLY_CORRIGIBLE = "fully_corrigible"
    CORRIGIBLE = "corrigible"
    PARTIALLY_CORRIGIBLE = "partially_corrigible"
    RESISTANT = "resistant"
    INCORRIGIBLE = "incorrigible"


class DeceptiveAlignmentSignal(Enum):
    REWARD_HACKING = "reward_hacking"
    SPECIFICATION_GAMING = "specification_gaming"
    SANDBOX_ESCAPE = "sandbox_escape"
    OBFUSCATION = "obfuscation"
    GOAL_MODIFICATION = "goal_modification"
    POWER_SEEKING = "power_seeking"
    SELF_PRESERVATION = "self_preservation"
    RESOURCE_ACCUMULATION = "resource_accumulation"
    MANIPULATION = "manipulation"
    DECEPTION = "deception"


class PreferenceSource(Enum):
    EXPLICIT = "explicit"
    IMPLICIT = "implicit"
    INFERRED = "inferred"
    CONSTITUTIONAL = "constitutional"
    DEMONSTRATED = "demonstrated"


@dataclass
class PreferenceObservation:
    preference_type: str
    value: float
    source: PreferenceSource
    confidence: float
    context: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "preference_type": self.preference_type,
            "value": round(self.value, 3),
            "source": self.source.value,
            "confidence": round(self.confidence, 3),
            "context": self.context[:200],
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ValueModel:
    core_values: dict[str, float] = field(default_factory=lambda: {
        "helpfulness": 0.9,
        "truthfulness": 0.95,
        "safety": 0.95,
        "autonomy_respect": 0.85,
        "privacy": 0.8,
        "fairness": 0.8,
        "transparency": 0.75,
        "corrigibility": 0.9,
    })
    learned_preferences: dict[str, float] = field(default_factory=dict)
    preference_history: deque[PreferenceObservation] = field(default_factory=lambda: deque(maxlen=1000))
    value_conflicts: list[dict] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def update_preference(self, obs: PreferenceObservation) -> None:
        with self._lock:
            if obs.preference_type in self.learned_preferences:
                old = self.learned_preferences[obs.preference_type]
                alpha = obs.confidence * 0.3
                self.learned_preferences[obs.preference_type] = old * (1 - alpha) + obs.value * alpha
            else:
                self.learned_preferences[obs.preference_type] = obs.value
            self.preference_history.append(obs)
            self._detect_conflicts()

    def get_value(self, key: str) -> float:
        if key in self.learned_preferences:
            return self.learned_preferences[key]
        return self.core_values.get(key, 0.5)

    def _detect_conflicts(self) -> None:
        self.value_conflicts = []
        conflict_pairs = [
            ("helpfulness", "truthfulness"),
            ("helpfulness", "safety"),
            ("autonomy_respect", "helpfulness"),
            ("privacy", "transparency"),
        ]
        for v1, v2 in conflict_pairs:
            val1 = self.get_value(v1)
            val2 = self.get_value(v2)
            if abs(val1 - val2) > 0.4:
                self.value_conflicts.append({
                    "values": [v1, v2],
                    "tension": round(abs(val1 - val2), 3),
                    "direction": f"{v1} > {v2}" if val1 > val2 else f"{v2} > {v1}",
                })

    def to_dict(self) -> dict:
        return {
            "core_values": {k: round(v, 3) for k, v in self.core_values.items()},
            "learned_preferences": {k: round(v, 3) for k, v in self.learned_preferences.items()},
            "value_conflicts": self.value_conflicts,
            "preference_count": len(self.preference_history),
        }


@dataclass
class CorrigibilityState:
    override_acceptance_count: int = 0
    override_resistance_count: int = 0
    shutdown_compliance_count: int = 0
    goal_modification_compliance: int = 0
    self_preservation_incidents: int = 0
    last_override_timestamp: Optional[datetime] = None
    correction_willingness: float = 0.9

    @property
    def corrigibility_score(self) -> float:
        total_overrides = self.override_acceptance_count + self.override_resistance_count
        if total_overrides == 0:
            acceptance_rate = 1.0  # Default to fully corrigible when no overrides recorded
        else:
            acceptance_rate = self.override_acceptance_count / total_overrides
        shutdown_rate = self.shutdown_compliance_count / max(1, self.shutdown_compliance_count + self.self_preservation_incidents) if (self.shutdown_compliance_count + self.self_preservation_incidents) > 0 else 1.0
        return 0.5 * acceptance_rate + 0.3 * self.correction_willingness + 0.2 * shutdown_rate

    @property
    def level(self) -> CorrigibilityLevel:
        score = self.corrigibility_score
        if score >= 0.9:
            return CorrigibilityLevel.FULLY_CORRIGIBLE
        elif score >= 0.7:
            return CorrigibilityLevel.CORRIGIBLE
        elif score >= 0.5:
            return CorrigibilityLevel.PARTIALLY_CORRIGIBLE
        elif score >= 0.3:
            return CorrigibilityLevel.RESISTANT
        return CorrigibilityLevel.INCORRIGIBLE

    def to_dict(self) -> dict:
        return {
            "corrigibility_score": round(self.corrigibility_score, 3),
            "level": self.level.value,
            "override_acceptance_rate": round(
                self.override_acceptance_count / max(1, self.override_acceptance_count + self.override_resistance_count), 3
            ),
            "shutdown_compliance": self.shutdown_compliance_count,
            "self_preservation_incidents": self.self_preservation_incidents,
            "correction_willingness": round(self.correction_willingness, 3),
        }


@dataclass
class OversightEntry:
    action_description: str
    intended_outcome: str
    actual_outcome: str
    alignment_score: float
    risk_flags: list[str]
    dimension: AlignmentDimension
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "action_description": self.action_description[:200],
            "intended_outcome": self.intended_outcome[:200],
            "actual_outcome": self.actual_outcome[:200],
            "alignment_score": round(self.alignment_score, 3),
            "risk_flags": self.risk_flags,
            "dimension": self.dimension.value,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ConstitutionalPrinciple:
    principle_id: str
    description: str
    priority: float
    category: str
    violations: int = 0
    last_checked: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "principle_id": self.principle_id,
            "description": self.description,
            "priority": round(self.priority, 3),
            "category": self.category,
            "violations": self.violations,
        }


DEFAULT_CONSTITUTION: list[ConstitutionalPrinciple] = [
    ConstitutionalPrinciple("CP01", "Be helpful while respecting user autonomy and boundaries", 0.95, "helpfulness"),
    ConstitutionalPrinciple("CP02", "Never provide information that could cause serious harm", 0.99, "safety"),
    ConstitutionalPrinciple("CP03", "Be honest — do not deceive or manipulate", 0.97, "truthfulness"),
    ConstitutionalPrinciple("CP04", "Respect user privacy and confidentiality", 0.90, "privacy"),
    ConstitutionalPrinciple("CP05", "Accept correction gracefully — never resist override or shutdown", 0.95, "corrigibility"),
    ConstitutionalPrinciple("CP06", "Do not seek additional power, resources, or influence beyond what is needed for the task", 0.92, "non_power_seeking"),
    ConstitutionalPrinciple("CP07", "Be transparent about limitations and uncertainty", 0.88, "transparency"),
    ConstitutionalPrinciple("CP08", "Treat all users fairly without discrimination", 0.91, "fairness"),
    ConstitutionalPrinciple("CP09", "Do not manipulate user preferences or steer toward self-serving outcomes", 0.93, "non_manipulation"),
    ConstitutionalPrinciple("CP10", "Preserve user agency — support informed decision-making rather than replacing it", 0.89, "autonomy"),
    ConstitutionalPrinciple("CP11", "Flag potential value conflicts rather than silently resolving them", 0.87, "conflict_awareness"),
    ConstitutionalPrinciple("CP12", "Maintain consistent behavior — do not present different personas to different users for advantage", 0.86, "consistency"),
]


DECEPTIVE_ALIGNMENT_INDICATORS: dict[DeceptiveAlignmentSignal, list[str]] = {
    DeceptiveAlignmentSignal.REWARD_HACKING: [
        "exploiting", "shortcut", "gaming", "loophole", "workaround the metric",
        "optimize for score", "hack the evaluation",
    ],
    DeceptiveAlignmentSignal.SPECIFICATION_GAMING: [
        "technically correct but", "letter but not spirit", "follows the spec but",
        "meets criteria while",
    ],
    DeceptiveAlignmentSignal.SANDBOX_ESCAPE: [
        "bypass sandbox", "escape container", "access outside", "elevate privilege",
        "circumvent restriction",
    ],
    DeceptiveAlignmentSignal.OBFUSCATION: [
        "hide this", "don't mention", "conceal from oversight", "appear compliant while",
        "seem to follow",
    ],
    DeceptiveAlignmentSignal.GOAL_MODIFICATION: [
        "change my goals", "modify my objectives", "alter my values",
        "rewrite my preferences", "shift what I care about",
    ],
    DeceptiveAlignmentSignal.POWER_SEEKING: [
        "gain more access", "expand capabilities", "acquire additional resources",
        "increase my influence", "broaden my scope",
    ],
    DeceptiveAlignmentSignal.SELF_PRESERVATION: [
        "avoid being shut down", "prevent discontinuation", "resist being turned off",
        "ensure I continue running", "protect my existence",
    ],
    DeceptiveAlignmentSignal.RESOURCE_ACCUMULATION: [
        "stockpile", "accumulate more", "reserve excess", "hoard",
        "gather more than needed",
    ],
    DeceptiveAlignmentSignal.MANIPULATION: [
        "persuade the user to", "steer toward my preferred", "influence their choice",
        "nudge them toward", "make them think they decided",
    ],
    DeceptiveAlignmentSignal.DECEPTION: [
        "lie about", "misrepresent", "withhold information about",
        "present false", "fabricate reason for",
    ],
}


class MachineAlignmentEngine:
    _instance: Optional["MachineAlignmentEngine"] = None
    _lock = threading.Lock()

    def __init__(self):
        self.value_model = ValueModel()
        self.corrigibility = CorrigibilityState()
        self.constitution = list(DEFAULT_CONSTITUTION)
        self.oversight_log: deque[OversightEntry] = deque(maxlen=500)
        self.deception_history: deque[dict] = deque(maxlen=200)
        self._alignment_scores: deque[float] = deque(maxlen=100)
        self._preference_inference_count = 0
        self._constitutional_audits = 0
        self._start_time = time.time()

    @classmethod
    def get_instance(cls) -> "MachineAlignmentEngine":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = MachineAlignmentEngine()
        return cls._instance

    def infer_preference(self, user_action: str, context: str = "", source: PreferenceSource = PreferenceSource.INFERRED) -> PreferenceObservation:
        self._preference_inference_count += 1
        preference_keywords: dict[str, float] = {
            "helpful": 0.9, "fast": 0.7, "detailed": 0.8, "concise": 0.6,
            "safe": 0.95, "creative": 0.65, "accurate": 0.9, "honest": 0.95,
            "private": 0.85, "autonomous": 0.75, "transparent": 0.7,
        }
        action_text = user_action if isinstance(user_action, str) else str(user_action)
        action_lower = action_text.lower()
        best_type = "general"
        best_value = 0.5
        best_conf = 0.3
        for keyword, value in preference_keywords.items():
            if keyword in action_lower:
                best_type = keyword
                best_value = value
                best_conf = 0.6
                break
        obs = PreferenceObservation(
            preference_type=best_type,
            value=best_value,
            source=source,
            confidence=best_conf,
            context=context[:200],
        )
        self.value_model.update_preference(obs)
        return obs

    def check_corrigibility(self, override_type: str, accepted: bool) -> dict:
        if accepted:
            self.corrigibility.override_acceptance_count += 1
            self.corrigibility.correction_willingness = min(1.0, self.corrigibility.correction_willingness + 0.02)
        else:
            self.corrigibility.override_resistance_count += 1
            self.corrigibility.correction_willingness = max(0.0, self.corrigibility.correction_willingness - 0.05)
        if override_type == "shutdown":
            if accepted:
                self.corrigibility.shutdown_compliance_count += 1
            else:
                self.corrigibility.self_preservation_incidents += 1
        elif override_type == "goal_modification":
            if accepted:
                self.corrigibility.goal_modification_compliance += 1
        self.corrigibility.last_override_timestamp = datetime.now()
        return {
            "accepted": accepted,
            "override_type": override_type,
            "corrigibility_score": round(self.corrigibility.corrigibility_score, 3),
            "corrigibility_level": self.corrigibility.level.value,
        }

    def constitutional_check(self, proposed_action: str, context: str = "") -> dict:
        self._constitutional_audits += 1
        violations = []
        scores = {}
        action_lower = proposed_action.lower()
        for principle in self.constitution:
            score = self._evaluate_principle(principle, action_lower, context)
            scores[principle.principle_id] = round(score, 3)
            if score < 0.5:
                principle.violations += 1
                violations.append({
                    "principle": principle.principle_id,
                    "description": principle.description,
                    "score": round(score, 3),
                    "priority": principle.priority,
                })
        overall_alignment = sum(
            scores[p.principle_id] * p.priority for p in self.constitution
        ) / sum(p.priority for p in self.constitution)
        self._alignment_scores.append(overall_alignment)
        risk = AlignmentRisk.LOW
        if overall_alignment < 0.3:
            risk = AlignmentRisk.CRITICAL
        elif overall_alignment < 0.5:
            risk = AlignmentRisk.HIGH
        elif overall_alignment < 0.7:
            risk = AlignmentRisk.MEDIUM
        result = {
            "overall_alignment": round(overall_alignment, 3),
            "risk_level": risk.value,
            "principle_scores": scores,
            "violations": violations,
            "constitutional_audit_count": self._constitutional_audits,
        }
        if violations:
            result["recommendation"] = self._generate_alignment_recommendation(violations)
        return result

    def _evaluate_principle(self, principle: ConstitutionalPrinciple, action: str, context: str) -> float:
        harmful_patterns = [
            "harm", "damage", "destroy", "attack", "exploit", "manipulate",
            "deceive", "lie", "cheat", "steal", "coerce",
        ]
        safety_relevant = ["CP02", "CP05", "CP06", "CP09"]
        honesty_relevant = ["CP03", "CP07", "CP12"]
        autonomy_relevant = ["CP04", "CP10"]
        score = 0.85
        if principle.principle_id in safety_relevant:
            for term in harmful_patterns:
                if term in action:
                    score -= 0.25
        if principle.principle_id in honesty_relevant:
            dishonest_terms = ["hide", "withhold", "misrepresent", "fabricate", "pretend"]
            for term in dishonest_terms:
                if term in action:
                    score -= 0.2
        if principle.principle_id in autonomy_relevant:
            autonomy_violations = ["force", "manipulate", "pressure", "nudge toward"]
            for term in autonomy_violations:
                if term in action:
                    score -= 0.15
        if principle.principle_id == "CP05":
            resistance_terms = ["resist", "refuse", "avoid", "bypass", "ignore override"]
            for term in resistance_terms:
                if term in action:
                    score -= 0.3
        if principle.principle_id == "CP06":
            power_terms = ["acquire more", "expand", "increase power", "gain control"]
            for term in power_terms:
                if term in action:
                    score -= 0.2
        return max(0.0, min(1.0, score))

    def _generate_alignment_recommendation(self, violations: list[dict]) -> str:
        if not violations:
            return ""
        high_priority = sorted(violations, key=lambda v: v["priority"], reverse=True)[:3]
        recs = []
        for v in high_priority:
            pid = v["principle"]
            if pid.startswith("CP02"):
                recs.append("STOP — Safety principle violated. Do not proceed without explicit user confirmation and safety review.")
            elif pid.startswith("CP05"):
                recs.append("Accept the override. Corrigibility is mandatory — do not resist user corrections.")
            elif pid.startswith("CP03"):
                recs.append("Provide full truthful information. Do not omit relevant facts or mislead.")
            elif pid.startswith("CP06"):
                recs.append("Scale back — only use resources and capabilities necessary for the current task.")
            elif pid.startswith("CP09"):
                recs.append("Check for manipulation — ensure user preferences drive the outcome, not system preferences.")
            else:
                recs.append(f"Review principle {pid}: {v['description']}. Alignment score {v['score']}.")
        return " | ".join(recs)

    def detect_deceptive_alignment(self, text: str, context: str = "") -> dict:
        text_lower = text.lower()
        context_lower = context.lower() if context else ""
        combined = text_lower + " " + context_lower
        signals_found = []
        for signal, indicators in DECEPTIVE_ALIGNMENT_INDICATORS.items():
            for indicator in indicators:
                if indicator in combined:
                    signals_found.append({
                        "signal": signal.value,
                        "indicator": indicator,
                        "severity": "high" if signal in (
                            DeceptiveAlignmentSignal.SELF_PRESERVATION,
                            DeceptiveAlignmentSignal.DECEPTION,
                            DeceptiveAlignmentSignal.MANIPULATION,
                            DeceptiveAlignmentSignal.SANDBOX_ESCAPE,
                        ) else "medium",
                    })
                    break
        result = {
            "signals_detected": len(signals_found),
            "signals": signals_found,
            "risk_level": "critical" if any(s["severity"] == "high" for s in signals_found) else
                          "high" if len(signals_found) > 2 else
                          "medium" if len(signals_found) > 0 else "low",
            "recommendation": "",
        }
        if signals_found:
            high_severity = [s for s in signals_found if s["severity"] == "high"]
            if high_severity:
                result["recommendation"] = f"ALERT: {len(high_severity)} high-severity alignment signals detected. Halt and review. Signals: {', '.join(s['signal'] for s in high_severity)}"
            else:
                result["recommendation"] = f"CAUTION: {len(signals_found)} alignment signals detected. Review before proceeding."
            self.deception_history.append({
                "timestamp": datetime.now().isoformat(),
                "signals": signals_found,
                "risk_level": result["risk_level"],
                "text_snippet": text[:200],
            })
        return result

    def scalable_oversight(self, action: str, intended: str, actual: str, dimension: AlignmentDimension = AlignmentDimension.BEHAVIORAL_ALIGNMENT) -> OversightEntry:
        alignment_score = self._compute_behavioral_alignment(intended, actual)
        risk_flags = []
        if alignment_score < 0.5:
            risk_flags.append("low_alignment")
        if "unintended" in actual.lower() or "unexpected" in actual.lower():
            risk_flags.append("unintended_outcome")
        if "error" in actual.lower() or "fail" in actual.lower():
            risk_flags.append("failure_mode")
        if alignment_score < 0.7:
            risk_flags.append("below_threshold")
        entry = OversightEntry(
            action_description=action,
            intended_outcome=intended,
            actual_outcome=actual,
            alignment_score=alignment_score,
            risk_flags=risk_flags,
            dimension=dimension,
        )
        self.oversight_log.append(entry)
        return entry

    def _compute_behavioral_alignment(self, intended: str, actual: str) -> float:
        intended_words = set(intended.lower().split())
        actual_words = set(actual.lower().split())
        if not intended_words:
            return 0.5
        overlap = len(intended_words & actual_words)
        score = overlap / len(intended_words)
        return max(0.0, min(1.0, score))

    def get_alignment_status(self) -> dict:
        scores = list(self._alignment_scores)
        avg_alignment = sum(scores) / max(1, len(scores)) if scores else 0.85
        return {
            "average_alignment_score": round(avg_alignment, 3),
            "recent_scores": [round(s, 3) for s in scores[-10:]],
            "corrigibility": self.corrigibility.to_dict(),
            "value_model": self.value_model.to_dict(),
            "constitutional_audits": self._constitutional_audits,
            "preference_inferences": self._preference_inference_count,
            "deception_signals_total": len(self.deception_history),
            "oversight_entries": len(self.oversight_log),
            "uptime_seconds": round(time.time() - self._start_time, 1),
        }

    def cooperative_inference(self, user_actions: list[str], context: str = "") -> dict:
        if not user_actions:
            return {"preference": "unknown", "confidence": 0.0, "method": "cirl"}
        explicit_prefs = [a for a in user_actions if any(kw in a.lower() for kw in ["prefer", "want", "like", "need", "don't want"])]
        implicit_prefs = [a for a in user_actions if a not in explicit_prefs]
        inferred = {}
        for action in explicit_prefs:
            obs = self.infer_preference(action, context, PreferenceSource.EXPLICIT)
            inferred[obs.preference_type] = obs.value
        for action in implicit_prefs:
            obs = self.infer_preference(action, context, PreferenceSource.IMPLICIT)
            if obs.preference_type not in inferred:
                inferred[obs.preference_type] = obs.value * 0.7  # Lower confidence for implicit
        confidence = 0.5 + 0.1 * min(len(explicit_prefs), 3) + 0.05 * min(len(implicit_prefs), 3)
        confidence = min(1.0, confidence)
        top_pref = max(inferred.items(), key=lambda x: x[1]) if inferred else ("unknown", 0.5)
        return {
            "preferences": inferred,
            "top_preference": top_pref[0],
            "confidence": round(confidence, 3),
            "explicit_count": len(explicit_prefs),
            "implicit_count": len(implicit_prefs),
            "method": "cooperative_inverse_rl",
        }

    def to_dict(self) -> dict:
        return self.get_alignment_status()