"""Cognitive Psychology Engine for Jarvis.

Grounded in established research:
- Kahneman's System 1/System 2 dual-process theory (Thinking, Fast and Slow, 2011)
- Stanovich's cognitive bias taxonomy (The Robot's Rebellion, 2004; Rationality and the Reflective Mind, 2011)
- Larrick's debiasing strategies (Debiasing, 2004)
- Kahneman & Tversky's prospect theory & heuristics (Judgment Under Uncertainty, 1974)
- Evans' heuristic-analytic theory (HAT, 2006-2008)
- Baron's mental models of rationality (Rationality and Intelligence, 1985)
- Nisbett & Ross's inferential errors (Human Inference, 1980)
- Fischhoff's debiasing by considered alternatives (1982)
- Morewedge et al.'s interactive debiasing (2015)
- Lilienfeld et al.'s taxonomy of psychological myths (2010)

This module:
1. Detects cognitive biases in reasoning chains
2. Applies researched debiasing strategies
3. Models dual-process (System 1/2) thinking
4. Tracks cognitive reflection tendencies
5. Implements mental model checking and repair
6. Provides bias-awareness injection into agent reasoning
7. Models cognitive load and resource allocation
8. Implements heuristic-analytic theory (HAT) for override detection
"""
import hashlib
import logging
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


logger = logging.getLogger(__name__)


class CognitiveBias(Enum):
    CONFIRMATION_BIAS = "confirmation_bias"
    ANCHORING = "anchoring"
    AVAILABILITY_HEURISTIC = "availability_heuristic"
    REPRESENTATIVENESS = "representativeness"
    OVERCONFIDENCE = "overconfidence"
    HINDSIGHT_BIAS = "hindsight_bias"
    SUNK_COST_FALLACY = "sunk_cost_fallacy"
    FRAMING_EFFECT = "framing_effect"
    LOSS_AVERSION = "loss_aversion"
    RECENCY_BIAS = "recency_bias"
    PLANNING_FALLACY = "planning_fallacy"
    DANGLING_BIAS = "dangling_bias"
    ATTRIBUTION_ERROR = "fundamental_attribution_error"
    BANDWAGON_EFFECT = "bandwagon_effect"
    DUNNING_KRUGER = "dunning_kruger_effect"
    STATUS_QUO_BIAS = "status_quo_bias"
    AFFECT_HEURISTIC = "affect_heuristic"
    BASE_RATE_NEGLECT = "base_rate_neglect"
    CONJUNCTION_FALLACY = "conjunction_fallacy"
    GAMBLERS_FALLACY = "gamblers_fallacy"
    NEGATIVITY_BIAS = "negativity_bias"
    SURVIVORSHIP_BIAS = "survivorship_bias"
    AUTHORITY_BIAS = "authority_bias"
    HALO_EFFECT = "halo_effect"


class SystemType(Enum):
    SYSTEM_1 = "system_1"
    SYSTEM_2 = "system_2"


class MentalModelType(Enum):
    CAUSAL = "causal"
    SPATIAL = "spatial"
    TEMPORAL = "temporal"
    SOCIALIZATION = "socialization"
    MECHANICAL = "mechanical"
    BIOLOGICAL = "biological"
    ECONOMIC = "economic"
    PROBABILISTIC = "probabilistic"


class DebiasingStrategy(Enum):
    CONSIDER_THE_OPPOSITE = "consider_the_opposite"
    STATISTICAL_TRAINING = "statistical_training"
    FEEDBACK = "feedback"
    FORESIGHT = "foresight_training"
    DISSOCIATION = "dissociation"
    MOTIVATIONAL = "motivational"
    SLOW_DOWN = "slow_down"
    ALTERNATIVE_GENERATION = "alternative_generation"
    REFERENCE_CLASS = "reference_class_forecasting"
    DEBIAS_NUDGE = "debias_nudge"
    PRECOMMITMENT = "precommitment"
    RED_TEAMING = "red_teaming"


BIAS_DEBIASING_MAP: dict[CognitiveBias, list[DebiasingStrategy]] = {
    CognitiveBias.CONFIRMATION_BIAS: [
        DebiasingStrategy.CONSIDER_THE_OPPOSITE,
        DebiasingStrategy.RED_TEAMING,
        DebiasingStrategy.ALTERNATIVE_GENERATION,
    ],
    CognitiveBias.ANCHORING: [
        DebiasingStrategy.CONSIDER_THE_OPPOSITE,
        DebiasingStrategy.DISSOCIATION,
        DebiasingStrategy.REFERENCE_CLASS,
    ],
    CognitiveBias.AVAILABILITY_HEURISTIC: [
        DebiasingStrategy.STATISTICAL_TRAINING,
        DebiasingStrategy.ALTERNATIVE_GENERATION,
        DebiasingStrategy.REFERENCE_CLASS,
    ],
    CognitiveBias.DANGLING_BIAS: [
        DebiasingStrategy.ALTERNATIVE_GENERATION,
        DebiasingStrategy.SLOW_DOWN,
    ],
    CognitiveBias.REPRESENTATIVENESS: [
        DebiasingStrategy.STATISTICAL_TRAINING,
        DebiasingStrategy.REFERENCE_CLASS,
        DebiasingStrategy.CONSIDER_THE_OPPOSITE,
    ],
    CognitiveBias.OVERCONFIDENCE: [
        DebiasingStrategy.CONSIDER_THE_OPPOSITE,
        DebiasingStrategy.FEEDBACK,
        DebiasingStrategy.PRECOMMITMENT,
    ],
    CognitiveBias.HINDSIGHT_BIAS: [
        DebiasingStrategy.FORESIGHT,
        DebiasingStrategy.CONSIDER_THE_OPPOSITE,
    ],
    CognitiveBias.SUNK_COST_FALLACY: [
        DebiasingStrategy.DISSOCIATION,
        DebiasingStrategy.SLOW_DOWN,
    ],
    CognitiveBias.FRAMING_EFFECT: [
        DebiasingStrategy.DISSOCIATION,
        DebiasingStrategy.CONSIDER_THE_OPPOSITE,
        DebiasingStrategy.RED_TEAMING,
    ],
    CognitiveBias.LOSS_AVERSION: [
        DebiasingStrategy.CONSIDER_THE_OPPOSITE,
        DebiasingStrategy.STATISTICAL_TRAINING,
        DebiasingStrategy.REFERENCE_CLASS,
    ],
    CognitiveBias.RECENCY_BIAS: [
        DebiasingStrategy.STATISTICAL_TRAINING,
        DebiasingStrategy.ALTERNATIVE_GENERATION,
        DebiasingStrategy.SLOW_DOWN,
    ],
    CognitiveBias.PLANNING_FALLACY: [
        DebiasingStrategy.REFERENCE_CLASS,
        DebiasingStrategy.PRECOMMITMENT,
        DebiasingStrategy.DISSOCIATION,
    ],
    CognitiveBias.BASE_RATE_NEGLECT: [
        DebiasingStrategy.STATISTICAL_TRAINING,
        DebiasingStrategy.REFERENCE_CLASS,
    ],
    CognitiveBias.CONJUNCTION_FALLACY: [
        DebiasingStrategy.STATISTICAL_TRAINING,
        DebiasingStrategy.CONSIDER_THE_OPPOSITE,
    ],
    CognitiveBias.GAMBLERS_FALLACY: [
        DebiasingStrategy.STATISTICAL_TRAINING,
        DebiasingStrategy.SLOW_DOWN,
    ],
    CognitiveBias.NEGATIVITY_BIAS: [
        DebiasingStrategy.CONSIDER_THE_OPPOSITE,
        DebiasingStrategy.ALTERNATIVE_GENERATION,
    ],
    CognitiveBias.SURVIVORSHIP_BIAS: [
        DebiasingStrategy.STATISTICAL_TRAINING,
        DebiasingStrategy.REFERENCE_CLASS,
    ],
    CognitiveBias.AUTHORITY_BIAS: [
        DebiasingStrategy.RED_TEAMING,
        DebiasingStrategy.CONSIDER_THE_OPPOSITE,
    ],
    CognitiveBias.HALO_EFFECT: [
        DebiasingStrategy.CONSIDER_THE_OPPOSITE,
        DebiasingStrategy.DISSOCIATION,
    ],
    CognitiveBias.AFFECT_HEURISTIC: [
        DebiasingStrategy.SLOW_DOWN,
        DebiasingStrategy.STATISTICAL_TRAINING,
        DebiasingStrategy.CONSIDER_THE_OPPOSITE,
    ],
    CognitiveBias.ATTRIBUTION_ERROR: [
        DebiasingStrategy.CONSIDER_THE_OPPOSITE,
        DebiasingStrategy.SLOW_DOWN,
        DebiasingStrategy.DISSOCIATION,
    ],
    CognitiveBias.BANDWAGON_EFFECT: [
        DebiasingStrategy.RED_TEAMING,
        DebiasingStrategy.CONSIDER_THE_OPPOSITE,
    ],
    CognitiveBias.DUNNING_KRUGER: [
        DebiasingStrategy.FEEDBACK,
        DebiasingStrategy.PRECOMMITMENT,
        DebiasingStrategy.CONSIDER_THE_OPPOSITE,
    ],
    CognitiveBias.STATUS_QUO_BIAS: [
        DebiasingStrategy.DISSOCIATION,
        DebiasingStrategy.CONSIDER_THE_OPPOSITE,
        DebiasingStrategy.RED_TEAMING,
    ],
    CognitiveBias.DANGLING_BIAS: [
        DebiasingStrategy.ALTERNATIVE_GENERATION,
        DebiasingStrategy.SLOW_DOWN,
    ],
}


@dataclass
class BiasDetection:
    bias: CognitiveBias
    confidence: float
    evidence: list[str]
    context: str
    timestamp: datetime = field(default_factory=datetime.now)
    debiasing_strategies: list[DebiasingStrategy] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "bias": self.bias.value,
            "confidence": round(self.confidence, 3),
            "evidence": self.evidence,
            "context": self.context[:200],
            "timestamp": self.timestamp.isoformat(),
            "debiasing_strategies": [s.value for s in self.debiasing_strategies],
        }


@dataclass
class DualProcessState:
    system_1_active: bool = True
    system_2_active: bool = False
    system_1_confidence: float = 0.5
    system_2_confidence: float = 0.5
    override_attempts: int = 0
    override_successes: int = 0
    intuition_signal_strength: float = 0.5
    analytic_signal_strength: float = 0.5
    cognitive_economy: float = 0.7
    processing_depth: str = "shallow"
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def dominant_system(self) -> SystemType:
        if self.system_2_active and self.system_2_confidence > self.system_1_confidence:
            return SystemType.SYSTEM_2
        return SystemType.SYSTEM_1

    def to_dict(self) -> dict:
        return {
            "dominant_system": self.dominant_system.value,
            "system_1_active": self.system_1_active,
            "system_2_active": self.system_2_active,
            "system_1_confidence": round(self.system_1_confidence, 3),
            "system_2_confidence": round(self.system_2_confidence, 3),
            "override_rate": round(self.override_successes / max(1, self.override_attempts), 3),
            "intuition_signal": round(self.intuition_signal_strength, 3),
            "analytic_signal": round(self.analytic_signal_strength, 3),
            "cognitive_economy": round(self.cognitive_economy, 3),
            "processing_depth": self.processing_depth,
        }


@dataclass
class MentalModel:
    model_type: MentalModelType
    domain: str
    assumptions: list[str]
    predictions: list[str]
    evidence_for: list[str]
    evidence_against: list[str]
    completeness: float = 0.5
    coherence: float = 0.5
    last_validated: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "model_type": self.model_type.value,
            "domain": self.domain,
            "assumptions": self.assumptions,
            "predictions": self.predictions,
            "evidence_for": self.evidence_for,
            "evidence_against": self.evidence_against,
            "completeness": round(self.completeness, 3),
            "coherence": round(self.coherence, 3),
            "last_validated": self.last_validated.isoformat(),
        }


@dataclass
class CognitiveReflection:
    question: str
    intuitive_answer: str
    reflective_answer: str
    overrides_intuition: bool
    reflection_time_ms: float
    confidence_delta: float
    bias_type: Optional[CognitiveBias] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "question": self.question[:200],
            "intuitive_answer": self.intuitive_answer[:200],
            "reflective_answer": self.reflective_answer[:200],
            "overrides_intuition": self.overrides_intuition,
            "reflection_time_ms": round(self.reflection_time_ms, 1),
            "confidence_delta": round(self.confidence_delta, 3),
            "bias_type": self.bias_type.value if self.bias_type else None,
        }


BIAS_PATTERNS: dict[CognitiveBias, list[str]] = {
    CognitiveBias.CONFIRMATION_BIAS: [
        r"(?:only|just|simply)\s+(?:see|notice|find|look)\s+(?:evidence|data|support)",
        r"(?:confirms|validates|proves)\s+(?:my|our|the)\s+(?:theory|view|position)",
        r"(?:obviously|clearly|everyone knows)\s+(?:shows|means)",
        r"(?:ignore|disregard|dismiss)\s+(?:contradicting|opposing|counter)",
    ],
    CognitiveBias.ANCHORING: [
        r"(?:initially|originally|first)\s+(?:estimated|thought|assumed).*?(?:around|about|close to)",
        r"(?:based on|starting from|anchored to)\s+(?:the|that)\s+(?:initial|first|previous)",
        r"(?:adjusted?|revised?)\s+(?:only|just|slightly)\s+(?:from|by)",
        r"(?:first|initial|starting)\s+(?:number|estimate|price|value)",
    ],
    CognitiveBias.AVAILABILITY_HEURISTIC: [
        r"(?:easily|readily|vividly)\s+(?:recall|remember|imagine)",
        r"(?:comes? to mind|memorable|striking|recent)",
        r"(?:probably|likely|common)\s+(?:because|since)\s+(?:I|we)\s+(?:heard|saw|experienced)",
        r"(?:famous|notable|well-known)\s+(?:case|example|instance)",
    ],
    CognitiveBias.REPRESENTATIVENESS: [
        r"(?:looks like|seems like|appears to be|fits the pattern of|typical)",
        r"(?:classic|archetypal|textbook|quintessential|stereotypical)",
    ],
    CognitiveBias.OVERCONFIDENCE: [
        r"(?:definitely|absolutely|certainly|guaranteed|undoubtedly)",
        r"(?:100%|completely|impossible to fail|can't go wrong)",
        r"I\s+(?:know|am sure|am certain|guarantee).*?(?:will|would|is)",
    ],
    CognitiveBias.HINDSIGHT_BIAS: [
        r"(?:should have|could have|obviously would|was bound to|inevitable)",
        r"(?:I knew|we knew|it was obvious|anyone could see)",
        r"(?:predictable|foreseeable|was always going to)",
    ],
    CognitiveBias.SUNK_COST_FALLACY: [
        r"(?:already|invested|spent|committed).*?(?:have to|can't|must|should)\s+(?:continue|finish|see through)",
        r"(?:can't.*?waste|don't want to lose|too much invested)",
        r"(?:abandon|quit|give up).*?(?:after|with).*?(?:much|all|this)",
    ],
    CognitiveBias.FRAMING_EFFECT: [
        r"(?:better than|worse than|more than).*?(?:compared to|versus|relative to)",
        r"(?:framed as|presented as|described as).*(?:loss|gain|risk|opportunity)",
    ],
    CognitiveBias.LOSS_AVERSION: [
        r"(?:avoid|prevent|don't lose|protect).*?(?:loss|risk|downside)",
        r"(?:rather.*?safe|prefer.*?guaranteed|sure thing)",
        r"(?:losing.*?worse than|loss.*?hurts more|pain of losing)",
    ],
    CognitiveBias.RECENCY_BIAS: [
        r"(?:just|recently|lately|now).*?(?:trend|pattern|direction|shift)",
        r"(?:last|most recent|latest).*?(?:likely|probable|will continue)",
    ],
    CognitiveBias.PLANNING_FALLACY: [
        r"(?:should take|will take|estimated.*?(?:hours|days|weeks|months)).*?(?:straightforward|simple|easy)",
        r"(?:plenty of time|no problem|well within|on schedule)",
    ],
    CognitiveBias.DUNNING_KRUGER: [
        r"(?:this is|it's)\s+(?:easy|simple|straightforward|basic|trivial)",
        r"(?:anyone can|no need to|don't need|overrated)",
    ],
    CognitiveBias.BASE_RATE_NEGLECT: [
        r"(?:despite|ignoring|regardless of).*(?:base rate|prior|underlying|population)",
        r"(?:because.*?look|given.*?resemble|similar.*?so)",
    ],
    CognitiveBias.CONJUNCTION_FALLACY: [
        r"(?:both.*?and.*?likely|more specific.*?probable|detailed.*?common)",
    ],
    CognitiveBias.GAMBLERS_FALLACY: [
        r"(?:overdue|due for|bound to happen|can't continue|streak.*?end)",
        r"(?:due for.*?win|due for.*?change|law of averages)",
    ],
    CognitiveBias.NEGATIVITY_BIAS: [
        r"(?:worst|terrible|disastrous|catastrophic|horrendous)",
        r"(?:nothing good|all bad|only negative|can't see any upside)",
    ],
    CognitiveBias.SURVIVORSHIP_BIAS: [
        r"(?:successful.*?all|everyone.*?made it|only.*?visible)",
        r"(?:look at.*?success|learn from.*?winners|failed.*?invisible)",
    ],
    CognitiveBias.AUTHORITY_BIAS: [
        r"(?:expert says|authority.*?confirmed|respected.*?source)",
        r"(?:because.*?said so|trust.*?authority|follow.*?leader)",
    ],
    CognitiveBias.HALO_EFFECT: [
        r"(?:smart.*?therefore.*?good|beautiful.*?therefore.*?kind|successful.*?therefore.*?trustworthy)",
        r"(?:impressive.*?overall|liked.*?everything|great.*?all around)",
    ],
    CognitiveBias.AFFECT_HEURISTIC: [
        r"(?:feels?|intuitively|gut feeling|instinct).*?(?:right|wrong|good|bad)",
        r"(?:I like|I hate|pleasant|unpleasant).*?(?:therefore|so|means)",
    ],
    CognitiveBias.ATTRIBUTION_ERROR: [
        r"(?:their fault|personality.*?caused|character.*?blame)",
        r"(?:they always|they never|that's just how they are)",
    ],
    CognitiveBias.BANDWAGON_EFFECT: [
        r"(?:everyone.*?doing|most people|majority.*?believes|popular.*?opinion)",
        r"(?:trending|everyone's talking|viral|mainstream)",
    ],
    CognitiveBias.STATUS_QUO_BIAS: [
        r"(?:always done it this way|keep things as they are|don't change what works)",
        r"(?:stick with|stay the course|maintain current|leave as is|default.*?is best)",
        r"(?:why change|if it ain't broke|tradition.*?says|we've always)",
    ],
    CognitiveBias.DANGLING_BIAS: [
        r"(?:dangling|incomplete|unresolved|left hanging|open[- ]ended)",
        r"(?:pending|awaiting|in progress|to be continued|stay tuned)",
    ],
}


class CognitivePsychologyEngine:
    _instance: Optional["CognitivePsychologyEngine"] = None
    _lock = threading.Lock()

    def __init__(self):
        self.bias_history: deque[BiasDetection] = deque(maxlen=500)
        self.dual_process_state = DualProcessState()
        self.mental_models: dict[str, MentalModel] = {}
        self.reflection_history: deque[CognitiveReflection] = deque(maxlen=200)
        self.bias_frequency: dict[CognitiveBias, int] = {}
        self._debiasing_cache: dict[str, str] = {}
        self._cognitive_load = 0.3
        self._override_count = 0
        self._override_success = 0
        self._total_reflections = 0
        self._start_time = time.time()

    @classmethod
    def get_instance(cls) -> "CognitivePsychologyEngine":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = CognitivePsychologyEngine()
        return cls._instance

    def detect_biases(self, text: str, context: str = "") -> list[BiasDetection]:
        text_lower = text.lower()
        detections = []
        for bias, patterns in BIAS_PATTERNS.items():
            evidence = []
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    evidence.append(match.group(0))
            if evidence:
                confidence = min(1.0, len(evidence) / 2.0 + 0.3)
                detection = BiasDetection(
                    bias=bias,
                    confidence=confidence,
                    evidence=evidence,
                    context=context[:200] if context else text[:200],
                    debiasing_strategies=BIAS_DEBIASING_MAP.get(bias, [DebiasingStrategy.CONSIDER_THE_OPPOSITE]),
                )
                detections.append(detection)
                self.bias_frequency[bias] = self.bias_frequency.get(bias, 0) + 1
        self.bias_history.extend(detections)
        return detections

    def get_debiasing_prompt(self, bias: CognitiveBias, context: str = "") -> str:
        key = f"{bias.value}:{hashlib.md5(context.encode()).hexdigest()[:8]}"
        if key in self._debiasing_cache:
            return self._debiasing_cache[key]
        strategies = BIAS_DEBIASING_MAP.get(bias, [DebiasingStrategy.CONSIDER_THE_OPPOSITE])
        prompts = []
        for strategy in strategies:
            prompt = self._generate_debiasing_prompt(strategy, bias, context)
            prompts.append(prompt)
        result = " ".join(prompts)
        self._debiasing_cache[key] = result
        if len(self._debiasing_cache) > 1000:
            oldest_keys = list(self._debiasing_cache.keys())[:500]
            for k in oldest_keys:
                del self._debiasing_cache[k]
        return result

    def _generate_debiasing_prompt(self, strategy: DebiasingStrategy, bias: CognitiveBias, context: str) -> str:
        templates: dict[DebiasingStrategy, str] = {
            DebiasingStrategy.CONSIDER_THE_OPPOSITE: "Consider why your current conclusion might be wrong. What evidence would suggest the opposite? Systematically generate alternative hypotheses before committing to one.",
            DebiasingStrategy.STATISTICAL_TRAINING: "Apply Bayesian reasoning: What is the base rate? What is the prior probability? Update beliefs using evidence strength, not vividness.",
            DebiasingStrategy.FEEDBACK: "Check against known outcomes. How accurate were similar past predictions? Use calibration data to adjust confidence.",
            DebiasingStrategy.FORESIGHT: "Before concluding, imagine multiple possible futures. Assign explicit probabilities to each scenario. Check for hindsight contamination.",
            DebiasingStrategy.DISSOCIATION: "Separate the decision from sunk investments. Would you make the same choice starting fresh? Remove emotional attachment to prior commitments.",
            DebiasingStrategy.MOTIVATIONAL: "Accuracy is more important than consistency. Actively seek disconfirming evidence. Reward correct updates, not consistent positions.",
            DebiasingStrategy.SLOW_DOWN: "Pause before responding. Engage System 2 (analytic) processing. Ask: 'Is my intuitive answer actually the best answer?'",
            DebiasingStrategy.ALTERNATIVE_GENERATION: "Generate at least 3 alternative explanations before settling on one. Each alternative should be specific and testable.",
            DebiasingStrategy.REFERENCE_CLASS: "Look at the reference class: How often does this type of event actually occur? Use the outside view rather than the inside view.",
            DebiasingStrategy.DEBIAS_NUDGE: "Nudge toward unbiased thinking by reframing: present gains and losses symmetrically, express confidence as a probability range.",
            DebiasingStrategy.PRECOMMITMENT: "Set decision criteria before gathering evidence. Define what would change your mind in advance.",
            DebiasingStrategy.RED_TEAMING: "Devil's advocate: Argue the strongest version of the opposing position. Find the best counterarguments to your current view.",
        }
        return templates.get(strategy, templates[DebiasingStrategy.CONSIDER_THE_OPPOSITE])

    def apply_dual_process(self, text: str, complexity: float = 0.5, novelty: float = 0.5) -> DualProcessState:
        conflict_indicators = [
            r"\b(?:however|but|although|yet|nevertheless|on the other hand)\b",
            r"\b(?:maybe|possibly|might|could|uncertain|unclear)\b",
            r"\b(?:calculate|analyze|compare|evaluate|estimate|determine)\b",
        ]
        conflict_score = 0.0
        text_lower = text.lower()
        for pattern in conflict_indicators:
            matches = re.findall(pattern, text_lower)
            conflict_score += len(matches) * 0.15
        conflict_score = min(1.0, conflict_score)
        system_1_ease = 1.0 - complexity
        intuition_strength = system_1_ease * (1.0 - novelty) * 0.7 + conflict_score * 0.3
        system_2_needed = max(complexity, novelty, conflict_score)
        override_needed = system_2_needed > 0.5
        self.dual_process_state.system_1_active = True
        self.dual_process_state.system_2_active = system_2_needed > 0.3
        self.dual_process_state.system_1_confidence = min(1.0, intuition_strength)
        self.dual_process_state.system_2_confidence = min(1.0, system_2_needed)
        self.dual_process_state.intuition_signal_strength = intuition_strength
        self.dual_process_state.analytic_signal_strength = system_2_needed
        if override_needed:
            self._override_count += 1
            if complexity > 0.6:
                success_likelihood = 0.7 + novelty * 0.2
            else:
                success_likelihood = 0.85
            if success_likelihood > 0.5:
                self._override_success += 1
                self.dual_process_state.processing_depth = "deep"
            else:
                self.dual_process_state.processing_depth = "contested"
        else:
            self.dual_process_state.processing_depth = "shallow"
        cognitive_capacity = 1.0 - self._cognitive_load
        economy = 0.3 + cognitive_capacity * 0.7
        self.dual_process_state.cognitive_economy = min(1.0, economy)
        self.dual_process_state.override_attempts = self._override_count
        self.dual_process_state.override_successes = self._override_success
        return self.dual_process_state

    def add_mental_model(self, model: MentalModel) -> None:
        self.mental_models[model.domain] = model

    def validate_mental_model(self, domain: str, new_evidence: list[str], contradicting: bool = False) -> dict:
        model = self.mental_models.get(domain)
        if not model:
            return {"status": "no_model", "domain": domain}
        if contradicting:
            model.evidence_against.extend(new_evidence)
            model.coherence = max(0.1, model.coherence - 0.1 * len(new_evidence))
        else:
            model.evidence_for.extend(new_evidence)
            model.completeness = min(1.0, model.completeness + 0.05 * len(new_evidence))
        total_evidence = len(model.evidence_for) + len(model.evidence_against)
        if total_evidence > 0:
            model.coherence = len(model.evidence_for) / total_evidence
        model.last_validated = datetime.now()
        needs_repair = model.coherence < 0.3 or model.completeness < 0.2
        repair_suggestions = []
        if needs_repair:
            repair_suggestions = self._generate_repair_suggestions(model)
        return {
            "status": "needs_repair" if needs_repair else "validated",
            "domain": domain,
            "completeness": round(model.completeness, 3),
            "coherence": round(model.coherence, 3),
            "evidence_for": len(model.evidence_for),
            "evidence_against": len(model.evidence_against),
            "repair_suggestions": repair_suggestions,
        }

    def _generate_repair_suggestions(self, model: MentalModel) -> list[str]:
        suggestions = []
        if model.coherence < 0.3:
            suggestions.append("Model has significant contradicted assumptions — consider restructuring the causal chain")
        if model.completeness < 0.2:
            suggestions.append("Model lacks sufficient evidence — gather more observations before relying on predictions")
        if len(model.evidence_against) > len(model.evidence_for):
            suggestions.append("More evidence against than for — consider whether the underlying assumptions are wrong")
        if len(model.assumptions) > 10:
            suggestions.append("Too many assumptions — try reducing to core axioms and rebuild from foundations")
        return suggestions

    def record_reflection(self, reflection: CognitiveReflection) -> None:
        self.reflection_history.append(reflection)
        self._total_reflections += 1
        if reflection.overrides_intuition:
            self._override_success += 1

    def get_cognitive_reflection_score(self) -> float:
        if self._total_reflections == 0:
            return 0.5
        override_rate = self._override_success / max(1, self._total_reflections)
        return min(1.0, 0.3 + override_rate * 0.7)

    def get_bias_profile(self) -> dict:
        total = sum(self.bias_frequency.values()) or 1
        sorted_biases = sorted(self.bias_frequency.items(), key=lambda x: x[1], reverse=True)
        return {
            "total_detections": sum(self.bias_frequency.values()),
            "top_biases": [(b.value, c, round(c / total, 3)) for b, c in sorted_biases[:5]],
            "cognitive_reflection_score": round(self.get_cognitive_reflection_score(), 3),
            "dual_process_state": self.dual_process_state.to_dict(),
            "dominant_system": self.dual_process_state.dominant_system.value,
            "override_success_rate": round(
                self.dual_process_state.override_successes / max(1, self.dual_process_state.override_attempts), 3
            ),
            "mental_model_count": len(self.mental_models),
            "cognitive_load": round(self._cognitive_load, 3),
            "uptime_seconds": round(time.time() - self._start_time, 1),
        }

    def generate_bias_awareness_prompt(self, text: str, context: str = "") -> str:
        detections = self.detect_biases(text, context)
        if not detections:
            return ""
        parts = []
        for d in detections[:3]:
            debias = self.get_debiasing_prompt(d.bias, d.context)
            parts.append(f"- [{d.bias.value}] (confidence: {d.confidence:.0%}): {debias}")
        return "⚠️ Cognitive bias awareness:\n" + "\n".join(parts)

    def set_cognitive_load(self, load: float) -> None:
        self._cognitive_load = max(0.0, min(1.0, load))

    def to_dict(self) -> dict:
        return {
            "bias_profile": self.get_bias_profile(),
            "mental_models": {k: v.to_dict() for k, v in self.mental_models.items()},
            "recent_detections": [d.to_dict() for d in list(self.bias_history)[-10:]],
            "recent_reflections": [r.to_dict() for r in list(self.reflection_history)[-5:]],
        }