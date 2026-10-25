"""Neuroscience-Inspired Cognitive Architecture for Jarvis.

Grounded in neuroscience research:
- Baddeley & Hitch's working memory model (1974, 2000)
- Miller's magical number 7±2 (1956), Cowan's 4±1 (2001)
- Kahneman's System 1/2 (2011), Evans' dual-process (2008)
- Damasio's somatic marker hypothesis (Descartes' Error, 1994)
- Posner & Petersen's attention network (1990)
- Dehaene's global workspace theory (Consciousness and the Brain, 2014)
- Baars' global workspace theory (1988, 2002)
- Rolls' orbitofrontal cortex and emotion (2004)
- O'Reilly & Frank's basal ganglia reinforcement learning (2006)
- D'Ardenne et al.'s dopamine prediction error (2008, fMRI)
- Schultz's dopamine reward prediction error (1997, 2016)
- Hasson et al.'s neural alignment/intersubject correlation (2008)
- Tononi's integrated information theory (2004, 2008)
- Friston's predictive coding/free energy principle (2010)
- Hofmann et al.'s amygdala-prefrontal emotion regulation (2012)
- Miller & Cohen's PFC goal-directed behavior (2001)
- Gold & Shadlen's neural decision-making (2007)

This module models:
1. Working memory architecture (central executive, phonological loop, visuospatial sketchpad, episodic buffer)
2. Attention networks (alerting, orienting, executive control)
3. Neurotransmitter systems (dopamine, serotonin, norepinephrine, acetylcholine, GABA, glutamate)
4. Predictive processing hierarchy (prediction, error, precision weighting)
5. Emotion regulation circuit (amygdala-PFC regulation model)
6. Decision-making circuits (basal ganglia Go/NoGo pathways)
7. Neural oscillation and synchronization
8. Global workspace broadcasting (consciousness-like integration)
9. Dopaminergic reward prediction and error signaling
10. Integrated information measure (simplified Phi)
"""
import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


logger = logging.getLogger(__name__)


class Neurotransmitter(Enum):
    DOPAMINE = "dopamine"
    SEROTONIN = "serotonin"
    NOREPINEPHRINE = "norepinephrine"
    ACETYLCHOLINE = "acetylcholine"
    GABA = "gaba"
    GLUTAMATE = "glutamate"


class BrainRegion(Enum):
    PREFRONTAL_CORTEX = "prefrontal_cortex"
    ORBITOFRONTAL_CORTEX = "orbitofrontal_cortex"
    ANTERIOR_CINGULATE = "anterior_cingulate"
    AMYGDALA = "amygdala"
    HIPPOCAMPUS = "hippocampus"
    BASAL_GANGLIA = "basal_ganglia"
    STRIATUM = "striatum"
    THALAMUS = "thalamus"
    CEREBELLUM = "cerebellum"
    INSULA = "insula"
    VTA = "ventral_tegmental_area"
    DEFAULT_MODE = "default_mode_network"


class AttentionNetwork(Enum):
    ALERTING = "alerting"
    ORIENTING = "orienting"
    EXECUTIVE_CONTROL = "executive_control"


class OscillationBand(Enum):
    DELTA = ("delta", 0.5, 4)
    THETA = ("theta", 4, 8)
    ALPHA = ("alpha", 8, 13)
    BETA = ("beta", 13, 30)
    GAMMA = ("gamma", 30, 100)

    def __init__(self, value, low_freq, high_freq):
        self.band_name = value
        self.low_freq = low_freq
        self.high_freq = high_freq


@dataclass
class WorkingMemoryState:
    central_executive_capacity: float = 4.0
    phonological_buffer: deque[str] = field(default_factory=lambda: deque(maxlen=4))
    visuospatial_buffer: deque[str] = field(default_factory=lambda: deque(maxlen=4))
    episodic_buffer: deque[dict] = field(default_factory=lambda: deque(maxlen=2))
    current_load: float = 0.0
    attention_focus: Optional[str] = None
    rehearsal_count: int = 0

    @property
    def available_capacity(self) -> float:
        used = len(self.phonological_buffer) * 0.25 + len(self.visuospatial_buffer) * 0.25 + len(self.episodic_buffer) * 0.5
        return max(0.0, self.central_executive_capacity - used)

    @property
    def load_fraction(self) -> float:
        used = len(self.phonological_buffer) * 0.25 + len(self.visuospatial_buffer) * 0.25 + len(self.episodic_buffer) * 0.5
        return min(1.0, used / max(0.1, self.central_executive_capacity))

    def add_phonological(self, item: str) -> bool:
        self.phonological_buffer.append(item)
        return True

    def add_visuospatial(self, item: str) -> bool:
        self.visuospatial_buffer.append(item)
        return True

    def add_episodic(self, item: dict) -> bool:
        self.episodic_buffer.append(item)
        return True

    def to_dict(self) -> dict:
        return {
            "central_executive_capacity": self.central_executive_capacity,
            "available_capacity": round(self.available_capacity, 2),
            "load_fraction": round(self.load_fraction, 3),
            "phonological_items": len(self.phonological_buffer),
            "visuospatial_items": len(self.visuospatial_buffer),
            "episodic_items": len(self.episodic_buffer),
            "attention_focus": self.attention_focus,
            "rehearsal_count": self.rehearsal_count,
        }


@dataclass
class NeurotransmitterState:
    levels: dict[Neurotransmitter, float] = field(default_factory=lambda: {
        Neurotransmitter.DOPAMINE: 0.5,
        Neurotransmitter.SEROTONIN: 0.5,
        Neurotransmitter.NOREPINEPHRINE: 0.5,
        Neurotransmitter.ACETYLCHOLINE: 0.5,
        Neurotransmitter.GABA: 0.5,
        Neurotransmitter.GLUTAMATE: 0.5,
    })
    prediction_errors: deque[float] = field(default_factory=lambda: deque(maxlen=100))

    def update_dopamine(self, reward: float, predicted: float) -> float:
        error = reward - predicted
        self.prediction_errors.append(error)
        current = self.levels[Neurotransmitter.DOPAMINE]
        if error > 0:
            new_level = min(1.0, current + abs(error) * 0.3)
        else:
            new_level = max(0.05, current - abs(error) * 0.2)
        self.levels[Neurotransmitter.DOPAMINE] = new_level
        return error

    def update_serotonin(self, context: str) -> float:
        negative_cues = ["failure", "error", "loss", "punish", "bad", "wrong"]
        positive_cues = ["success", "reward", "achieve", "good", "correct", "win"]
        text_lower = context.lower()
        valence = 0.0
        for cue in positive_cues:
            if cue in text_lower:
                valence += 0.15
        for cue in negative_cues:
            if cue in text_lower:
                valence -= 0.15
        current = self.levels[Neurotransmitter.SEROTONIN]
        new_level = max(0.05, min(1.0, current + valence * 0.2))
        self.levels[Neurotransmitter.SEROTONIN] = new_level
        return new_level

    def update_norepinephrine(self, arousal: float) -> float:
        current = self.levels[Neurotransmitter.NOREPINEPHRINE]
        inverted_u = 4 * arousal * (1.0 - arousal)  # Yerkes-Dodson
        target = 0.2 + inverted_u * 0.6
        new_level = current * 0.7 + target * 0.3
        self.levels[Neurotransmitter.NOREPINEPHRINE] = max(0.05, min(1.0, new_level))
        return new_level

    def update_acetylcholine(self, attention_demand: float) -> float:
        target = attention_demand * 0.8 + 0.1
        current = self.levels[Neurotransmitter.ACETYLCHOLINE]
        new_level = current * 0.6 + target * 0.4
        self.levels[Neurotransmitter.ACETYLCHOLINE] = max(0.05, min(1.0, new_level))
        return new_level

    def to_dict(self) -> dict:
        return {
            "levels": {nt.value: round(level, 3) for nt, level in self.levels.items()},
            "dopamine_prediction_error": round(self.prediction_errors[-1], 3) if self.prediction_errors else 0.0,
            "dopamine_tone": "high" if self.levels[Neurotransmitter.DOPAMINE] > 0.7 else
                            "low" if self.levels[Neurotransmitter.DOPAMINE] < 0.3 else "normal",
            "serotonin_tone": "high" if self.levels[Neurotransmitter.SEROTONIN] > 0.7 else
                             "low" if self.levels[Neurotransmitter.SEROTONIN] < 0.3 else "normal",
            "arousal_state": "hyperaroused" if self.levels[Neurotransmitter.NOREPINEPHRINE] > 0.8 else
                            "optimal" if 0.3 < self.levels[Neurotransmitter.NOREPINEPHRINE] < 0.7 else
                            "hypoaroused",
        }


@dataclass
class PredictiveCodingState:
    current_prediction: dict = field(default_factory=dict)
    prediction_error: float = 0.0
    precision_weight: float = 0.5
    hierarchy_level: int = 0
    learning_rate: float = 0.1
    prediction_history: deque = field(default_factory=lambda: deque(maxlen=100))
    error_history: deque = field(default_factory=lambda: deque(maxlen=100))

    def predict(self, sensory_input: dict) -> dict:
        if not self.current_prediction:
            self.current_prediction = sensory_input.copy()
            return {
                "prediction": sensory_input,
                "prediction_error": 0.0,
                "precision": self.precision_weight,
                "learning_signal": 0.0,
            }
        error = 0.0
        count = 0
        updated_prediction = {}
        for key, value in sensory_input.items():
            if key in self.current_prediction:
                predicted = self.current_prediction[key]
                if isinstance(value, (int, float)) and isinstance(predicted, (int, float)):
                    key_error = abs(value - predicted)
                    error += key_error
                    count += 1
                    precision = self.precision_weight
                    learning_signal = precision * key_error
                    updated_prediction[key] = predicted + self.learning_rate * learning_signal * (value - predicted)
                else:
                    updated_prediction[key] = value
            else:
                updated_prediction[key] = value
        avg_error = error / max(1, count)
        self.prediction_error = avg_error
        self.error_history.append(avg_error)
        self.current_prediction = updated_prediction
        self.prediction_history.append({
            "error": avg_error,
            "precision": self.precision_weight,
            "timestamp": datetime.now().isoformat(),
        })
        return {
            "prediction": updated_prediction,
            "prediction_error": round(avg_error, 3),
            "precision": round(self.precision_weight, 3),
            "learning_signal": round(self.precision_weight * avg_error, 3),
        }

    def update_precision(self, volatility: float) -> float:
        self.precision_weight = max(0.1, min(1.0, self.precision_weight + 0.1 * (volatility - 0.5)))
        return self.precision_weight

    def to_dict(self) -> dict:
        return {
            "prediction_error": round(self.prediction_error, 3),
            "precision_weight": round(self.precision_weight, 3),
            "hierarchy_level": self.hierarchy_level,
            "learning_rate": round(self.learning_rate, 3),
            "error_history_avg": round(sum(self.error_history) / max(1, len(self.error_history)), 3) if self.error_history else 0.0,
        }


@dataclass
class AttentionState:
    alerting_score: float = 0.5
    orienting_score: float = 0.5
    executive_control: float = 0.5
    current_focus: Optional[str] = None
    attention_blink: bool = False
    blink_recovery_ms: float = 500.0

    @property
    def overall_attention(self) -> float:
        return 0.2 * self.alerting_score + 0.3 * self.orienting_score + 0.5 * self.executive_control

    def to_dict(self) -> dict:
        return {
            "alerting": round(self.alerting_score, 3),
            "orienting": round(self.orienting_score, 3),
            "executive_control": round(self.executive_control, 3),
            "overall": round(self.overall_attention, 3),
            "current_focus": self.current_focus,
            "attention_blink": self.attention_blink,
        }


@dataclass
class EmotionRegulationCircuit:
    amygdala_activation: float = 0.3
    pfc_regulation: float = 0.7
    vagal_tone: float = 0.5
    reappraisal_strength: float = 0.6
    suppression_tendency: float = 0.2
    regulation_history: deque = field(default_factory=lambda: deque(maxlen=100))

    def process_emotional_stimulus(self, stimulus: str, intensity: float) -> dict:
        threat_cues = ["danger", "harm", "attack", "risk", "threat", "warning", "critical", "urgent", "error", "fail"]
        reward_cues = ["success", "reward", "good", "great", "achieve", "win", "positive", "excellent"]
        stimulus_lower = stimulus.lower()
        threat_score = sum(0.15 for cue in threat_cues if cue in stimulus_lower)
        reward_score = sum(0.1 for cue in reward_cues if cue in stimulus_lower)
        raw_amygdala = min(1.0, intensity * (0.5 + threat_score - reward_score * 0.3))
        self.amygdala_activation = raw_amygdala
        regulated = self.amygdala_activation * (1.0 - self.pfc_regulation * 0.6) + self.vagal_tone * 0.2
        regulated = max(0.0, min(1.0, regulated))
        reappraised = self._reappraise(stimulus, regulated)
        self.regulation_history.append({
            "stimulus": stimulus[:100],
            "raw_amygdala": round(raw_amygdala, 3),
            "regulated": round(regulated, 3),
            "reappraised": round(reappraised, 3),
            "pfc_regulation": round(self.pfc_regulation, 3),
            "timestamp": datetime.now().isoformat(),
        })
        result_type = "threat" if raw_amygdala > 0.6 else "reward" if raw_amygdala < 0.3 else "neutral"
        return {
            "stimulus_type": result_type,
            "raw_amygdala_activation": round(raw_amygdala, 3),
            "pfc_regulated_response": round(regulated, 3),
            "reappraised_response": round(reappraised, 3),
            "regulation_success": round(1.0 - abs(regulated - reappraised), 3),
        }

    def _reappraise(self, stimulus: str, current_valence: float) -> float:
        if current_valence > 0.6:
            return current_valence * (1.0 - self.reappraisal_strength * 0.4)
        elif current_valence < 0.3:
            return current_valence + self.reappraisal_strength * 0.2
        return current_valence

    def to_dict(self) -> dict:
        return {
            "amygdala_activation": round(self.amygdala_activation, 3),
            "pfc_regulation": round(self.pfc_regulation, 3),
            "vagal_tone": round(self.vagal_tone, 3),
            "reappraisal_strength": round(self.reappraisal_strength, 3),
            "suppression_tendency": round(self.suppression_tendency, 3),
            "regulation_effectiveness": round(self.pfc_regulation * self.reappraisal_strength, 3),
        }


@dataclass
class DecisionCircuit:
    go_pathway: float = 0.5
    nogo_pathway: float = 0.5
    decision_threshold: float = 0.6
    evidence_accumulation: float = 0.0
    drift_rate: float = 0.3
    non_decision_time: float = 0.1
    decision_history: deque = field(default_factory=lambda: deque(maxlen=200))

    def accumulate_evidence(self, evidence: dict) -> dict:
        evidence_strength = 0.0
        count = 0
        for key, value in evidence.items():
            if isinstance(value, (int, float)):
                evidence_strength += value
                count += 1
            elif isinstance(value, bool):
                evidence_strength += 1.0 if value else -0.5
                count += 1
        avg_evidence = evidence_strength / max(1, count) if count > 0 else 0.0
        self.evidence_accumulation += avg_evidence * self.drift_rate
        self.go_pathway = 1.0 / (1.0 + math.exp(-(self.evidence_accumulation - self.decision_threshold) * 5))
        self.nogo_pathway = 1.0 - self.go_pathway
        reached_threshold = abs(self.evidence_accumulation) >= self.decision_threshold
        if reached_threshold:
            decision = "proceed" if self.evidence_accumulation > 0 else "inhibit"
            confidence = min(1.0, abs(self.evidence_accumulation) / max(0.1, self.decision_threshold))
            self.decision_history.append({
                "decision": decision,
                "confidence": round(confidence, 3),
                "evidence": round(self.evidence_accumulation, 3),
                "go_pathway": round(self.go_pathway, 3),
                "nogo_pathway": round(self.nogo_pathway, 3),
                "timestamp": datetime.now().isoformat(),
            })
            self.evidence_accumulation *= 0.3  # Reset for next decision
        return {
            "go_probability": round(self.go_pathway, 3),
            "nogo_probability": round(self.nogo_pathway, 3),
            "evidence_accumulated": round(self.evidence_accumulation, 3),
            "threshold_reached": reached_threshold,
            "decision": "proceed" if self.evidence_accumulation > 0 else "inhibit" if reached_threshold else "pending",
            "confidence": round(min(1.0, abs(self.evidence_accumulation) / max(0.1, self.decision_threshold)), 3),
        }

    def reset_evidence(self) -> None:
        self.evidence_accumulation = 0.0
        self.go_pathway = 0.5
        self.nogo_pathway = 0.5

    def to_dict(self) -> dict:
        return {
            "go_pathway": round(self.go_pathway, 3),
            "nogo_pathway": round(self.nogo_pathway, 3),
            "evidence_accumulation": round(self.evidence_accumulation, 3),
            "decision_threshold": round(self.decision_threshold, 3),
            "drift_rate": round(self.drift_rate, 3),
            "decisions_made": len(self.decision_history),
        }


@dataclass
class GlobalWorkspaceState:
    broadcasting: bool = False
    workspace_contents: list[dict] = field(default_factory=list)
    competition_scores: dict[str, float] = field(default_factory=dict)
    broadcast_bandwidth: float = 0.7
    coalition_threshold: float = 0.6

    def broadcast(self, inputs: dict[str, float]) -> dict:
        if not inputs:
            return {"broadcasting": False, "winners": [], "workspace_size": 0}
        total_activation = sum(inputs.values())
        if total_activation == 0:
            return {"broadcasting": False, "winners": [], "workspace_size": 0}
        normalized = {k: v / total_activation for k, v in inputs.items()}
        sorted_inputs = sorted(normalized.items(), key=lambda x: x[1], reverse=True)
        winners = []
        cumulative = 0.0
        for name, score in sorted_inputs:
            if score >= 0.05 and (len(winners) == 0 or cumulative < self.coalition_threshold):
                winners.append({"name": name, "score": round(score, 3)})
                cumulative += score
        self.broadcasting = len(winners) > 0
        self.workspace_contents = winners
        self.competition_scores = normalized
        return {
            "broadcasting": self.broadcasting,
            "winners": winners,
            "workspace_size": len(winners),
            "total_competitors": len(inputs),
            "integration_measure": round(cumulative, 3),
        }

    def to_dict(self) -> dict:
        return {
            "broadcasting": self.broadcasting,
            "workspace_contents": self.workspace_contents,
            "broadcast_bandwidth": round(self.broadcast_bandwidth, 3),
            "coalition_threshold": round(self.coalition_threshold, 3),
        }


@dataclass
class NeuralOscillation:
    band: OscillationBand
    power: float = 0.5
    phase: float = 0.0
    coherence: float = 0.5

    def step(self, dt: float = 0.01) -> None:
        freq = (self.band.low_freq + self.band.high_freq) / 2.0
        self.phase = (self.phase + 2 * math.pi * freq * dt) % (2 * math.pi)
        self.power *= 0.999  # Slight decay
        self.power = max(0.01, self.power)

    def to_dict(self) -> dict:
        return {
            "band": self.band.band_name,
            "frequency_range": f"{self.band.low_freq}-{self.band.high_freq}Hz",
            "power": round(self.power, 3),
            "phase": round(self.phase, 3),
            "coherence": round(self.coherence, 3),
        }


def compute_phi(workspace: GlobalWorkspaceState, subsystems: dict[str, float]) -> float:
    """Simplified integrated information measure (Φ).
    
    Based on Tononi's IIT: measures how much integrated information the system generates
    above and beyond its parts. Higher Φ = more consciousness-like integration.
    """
    if not workspace.workspace_contents or not subsystems:
        return 0.0
    whole_entropy = -sum(
        s * math.log2(max(0.001, s)) for s in
        [c["score"] for c in workspace.workspace_contents]
        if s > 0
    )
    part_entropy = 0.0
    for name, activation in subsystems.items():
        if activation > 0:
            part_entropy -= activation * math.log2(max(0.001, activation))
    phi = max(0.0, whole_entropy - part_entropy * 0.3)
    return min(1.0, phi)


class NeuroscienceCognitiveArchitecture:
    _instance: Optional["NeuroscienceCognitiveArchitecture"] = None
    _lock = threading.Lock()

    def __init__(self):
        self.working_memory = WorkingMemoryState()
        self.neurotransmitters = NeurotransmitterState()
        self.predictive_coding = PredictiveCodingState()
        self.attention = AttentionState()
        self.emotion_regulation = EmotionRegulationCircuit()
        self.decision_circuit = DecisionCircuit()
        self.global_workspace = GlobalWorkspaceState()
        self.oscillations = {
            OscillationBand.DELTA: NeuralOscillation(OscillationBand.DELTA),
            OscillationBand.THETA: NeuralOscillation(OscillationBand.THETA),
            OscillationBand.ALPHA: NeuralOscillation(OscillationBand.ALPHA),
            OscillationBand.BETA: NeuralOscillation(OscillationBand.BETA),
            OscillationBand.GAMMA: NeuralOscillation(OscillationBand.GAMMA),
        }
        self.brain_region_activation: dict[BrainRegion, float] = {
            region: 0.3 for region in BrainRegion
        }
        self._phi_history: deque[float] = deque(maxlen=100)
        self._start_time = time.time()
        self._tick_count = 0

    @classmethod
    def get_instance(cls) -> "NeuroscienceCognitiveArchitecture":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = NeuroscienceCognitiveArchitecture()
        return cls._instance

    def tick(self, dt: float = 0.01) -> None:
        self._tick_count += 1
        for oscillation in self.oscillations.values():
            oscillation.step(dt)
        self.decision_circuit.evidence_accumulation *= 0.995
        self.brain_region_activation[BrainRegion.DEFAULT_MODE] = 1.0 - self.attention.overall_attention
        self.brain_region_activation[BrainRegion.PREFRONTAL_CORTEX] = self.attention.executive_control
        self.brain_region_activation[BrainRegion.ANTERIOR_CINGULATE] = abs(self.predictive_coding.prediction_error)
        if self._tick_count % 10 == 0:
            volatility = abs(self.predictive_coding.prediction_error)
            self.predictive_coding.update_precision(volatility)

    def process_input(self, text: str, emotional_valence: float = 0.0, importance: float = 0.5, novelty: float = 0.5) -> dict:
        self.tick()
        self.attention.alerting_score = min(1.0, 0.3 + importance * 0.4 + novelty * 0.3)
        self.attention.orienting_score = min(1.0, 0.3 + novelty * 0.5 + importance * 0.2)
        self.attention.executive_control = min(1.0, 0.3 + importance * 0.5)
        self.neurotransmitters.update_norepinephrine(self.attention.overall_attention)
        self.neurotransmitters.update_acetylcholine(self.attention.executive_control)
        self.neurotransmitters.update_serotonin(text)
        emotion_result = self.emotion_regulation.process_emotional_stimulus(text, abs(emotional_valence))
        if self.working_memory.available_capacity > 0:
            wm_slot_type = "phonological" if any(c.isalpha() for c in text[:10]) else "visuospatial"
            if wm_slot_type == "phonological":
                self.working_memory.add_phonological(text[:50])
            else:
                self.working_memory.add_visuospatial(text[:50])
            self.working_memory.add_episodic({
                "text": text[:100],
                "valence": emotional_valence,
                "importance": importance,
                "timestamp": datetime.now().isoformat(),
            })
        if importance > 0.6 or emotional_valence > 0.5:
            self.oscillations[OscillationBand.GAMMA].power = min(1.0, self.oscillations[OscillationBand.GAMMA].power + 0.2)
            self.oscillations[OscillationBand.BETA].power = min(1.0, self.oscillations[OscillationBand.BETA].power + 0.15)
        if novelty > 0.5:
            self.oscillations[OscillationBand.THETA].power = min(1.0, self.oscillations[OscillationBand.THETA].power + 0.2)
        if importance < 0.3:
            self.oscillations[OscillationBand.ALPHA].power = min(1.0, self.oscillations[OscillationBand.ALPHA].power + 0.1)
        if emotional_valence < -0.5:
            self.brain_region_activation[BrainRegion.AMYGDALA] = min(1.0, self.brain_region_activation[BrainRegion.AMYGDALA] + 0.3)
            self.brain_region_activation[BrainRegion.PREFRONTAL_CORTEX] = min(1.0, self.emotion_regulation.pfc_regulation + 0.1)
        workspace_inputs = {
            "perception": importance * 0.3 + novelty * 0.2,
            "emotion": abs(emotional_valence) * 0.3,
            "memory": 0.2 + min(0.3, self.working_memory.load_fraction),
            "reasoning": self.attention.executive_control * 0.3,
            "language": 0.3 + min(0.2, len(text) / 500),
        }
        broadcast_result = self.global_workspace.broadcast(workspace_inputs)
        subsystems = {r.value: a for r, a in self.brain_region_activation.items()}
        phi = compute_phi(self.global_workspace, subsystems)
        self._phi_history.append(phi)
        return {
            "attention": self.attention.to_dict(),
            "emotion_regulation": emotion_result,
            "working_memory": self.working_memory.to_dict(),
            "neurotransmitters": self.neurotransmitters.to_dict(),
            "global_workspace": broadcast_result,
            "decision_state": self.decision_circuit.to_dict(),
            "predictive_coding": self.predictive_coding.to_dict(),
            "oscillations": {band.band_name: osc.to_dict() for band, osc in self.oscillations.items()},
            "brain_activation": {r.value: round(a, 3) for r, a in self.brain_region_activation.items()},
            "integrated_information_phi": round(phi, 3),
        }

    def predict_reward(self, predicted_reward: float, actual_reward: float) -> dict:
        error = self.neurotransmitters.update_dopamine(actual_reward, predicted_reward)
        self.brain_region_activation[BrainRegion.VTA] = min(1.0, 0.3 + abs(error) * 0.7)
        if error > 0:
            self.brain_region_activation[BrainRegion.STRIATUM] = min(1.0, 0.3 + error * 0.5)
        else:
            self.brain_region_activation[BrainRegion.ANTERIOR_CINGULATE] = min(1.0, self.brain_region_activation[BrainRegion.ANTERIOR_CINGULATE] + abs(error) * 0.3)
        return {
            "predicted_reward": round(predicted_reward, 3),
            "actual_reward": round(actual_reward, 3),
            "prediction_error": round(error, 3),
            "dopamine_level": round(self.neurotransmitters.levels[Neurotransmitter.DOPAMINE], 3),
            "vta_activation": round(self.brain_region_activation[BrainRegion.VTA], 3),
            "striatum_activation": round(self.brain_region_activation[BrainRegion.STRIATUM], 3),
            "error_type": "positive" if error > 0 else "negative" if error < 0 else "none",
        }

    def make_decision(self, evidence: dict, urgency: float = 0.5) -> dict:
        self.decision_circuit.drift_rate = 0.2 + urgency * 0.4
        self.decision_circuit.decision_threshold = max(0.3, 1.0 - urgency * 0.4)
        result = self.decision_circuit.accumulate_evidence(evidence)
        self.brain_region_activation[BrainRegion.PREFRONTAL_CORTEX] = min(1.0, 0.4 + urgency * 0.3)
        self.brain_region_activation[BrainRegion.BASAL_GANGLIA] = min(1.0, max(self.decision_circuit.go_pathway, self.decision_circuit.nogo_pathway))
        if result.get("reached_threshold", False):
            self.decision_circuit.reset_evidence()
        return result

    def regulate_emotion(self, strategy: str = "reappraisal") -> dict:
        if strategy == "reappraisal":
            self.emotion_regulation.reappraisal_strength = min(1.0, self.emotion_regulation.reappraisal_strength + 0.05)
            self.emotion_regulation.pfc_regulation = min(1.0, self.emotion_regulation.pfc_regulation + 0.02)
            self.brain_region_activation[BrainRegion.PREFRONTAL_CORTEX] = min(1.0, self.brain_region_activation[BrainRegion.PREFRONTAL_CORTEX] + 0.1)
            self.brain_region_activation[BrainRegion.AMYGDALA] = max(0.0, self.brain_region_activation[BrainRegion.AMYGDALA] - 0.1)
        elif strategy == "suppression":
            self.emotion_regulation.suppression_tendency = min(1.0, self.emotion_regulation.suppression_tendency + 0.05)
            self.brain_region_activation[BrainRegion.INSULA] = max(0.0, self.brain_region_activation[BrainRegion.INSULA] - 0.05)
        elif strategy == "mindfulness":
            self.emotion_regulation.vagal_tone = min(1.0, self.emotion_regulation.vagal_tone + 0.03)
            self.oscillations[OscillationBand.ALPHA].power = min(1.0, self.oscillations[OscillationBand.ALPHA].power + 0.05)
            self.attention.executive_control = min(1.0, self.attention.executive_control + 0.03)
        return {
            "strategy": strategy,
            "amygdala": round(self.brain_region_activation[BrainRegion.AMYGDALA], 3),
            "pfc_regulation": round(self.emotion_regulation.pfc_regulation, 3),
            "reappraisal_strength": round(self.emotion_regulation.reappraisal_strength, 3),
            "vagal_tone": round(self.emotion_regulation.vagal_tone, 3),
        }

    def get_neural_state(self) -> dict:
        subsystems = {r.value: a for r, a in self.brain_region_activation.items()}
        phi = compute_phi(self.global_workspace, subsystems)
        return {
            "attention": self.attention.to_dict(),
            "working_memory": self.working_memory.to_dict(),
            "neurotransmitters": self.neurotransmitters.to_dict(),
            "emotion_regulation": self.emotion_regulation.to_dict(),
            "decision_circuit": self.decision_circuit.to_dict(),
            "global_workspace": self.global_workspace.to_dict(),
            "predictive_coding": self.predictive_coding.to_dict(),
            "oscillations": {band.band_name: osc.to_dict() for band, osc in self.oscillations.items()},
            "brain_activation": {r.value: round(a, 3) for r, a in self.brain_region_activation.items()},
            "integrated_information_phi": round(phi, 3),
            "phi_history_avg": round(sum(self._phi_history) / max(1, len(self._phi_history)), 3) if self._phi_history else 0.0,
            "working_memory_load": round(self.working_memory.load_fraction, 3),
            "cognitive_state": self._infer_cognitive_state(),
            "tick_count": self._tick_count,
            "uptime_seconds": round(time.time() - self._start_time, 1),
        }

    def _infer_cognitive_state(self) -> str:
        wm_load = self.working_memory.load_fraction
        attention = self.attention.overall_attention
        dopamine = self.neurotransmitters.levels[Neurotransmitter.DOPAMINE]
        ne = self.neurotransmitters.levels[Neurotransmitter.NOREPINEPHRINE]
        if wm_load > 0.8 and attention > 0.7:
            return "overloaded_focused"
        elif wm_load > 0.8 and attention < 0.5:
            return "overloaded_distracted"
        elif attention > 0.7 and dopamine > 0.6:
            return "engaged_rewarded"
        elif attention > 0.7:
            return "focused"
        elif ne > 0.7:
            return "alert_vigilant"
        elif ne < 0.3:
            return "drowsy"
        elif dopamine > 0.7:
            return "reward_driven"
        elif dopamine < 0.3:
            return "demotivated"
        elif self.emotion_regulation.amygdala_activation > 0.6:
            return "emotionally_aroused"
        elif wm_load < 0.3 and attention < 0.4:
            return "idle"
        else:
            return "balanced"

    def sync_with_bio_homeostasis(self, bio_homeostasis) -> None:
        """Synchronize neurotransmitter levels with bio-homeostasis system.
        
        Bidirectional bridge: averages neuroscience NT levels with homeostasis NT levels,
        so both systems stay aligned.
        """
        try:
            nt_map = {
                Neurotransmitter.DOPAMINE: "dopamine",
                Neurotransmitter.SEROTONIN: "serotonin",
                Neurotransmitter.NOREPINEPHRINE: "norepinephrine",
                Neurotransmitter.ACETYLCHOLINE: "acetylcholine",
                Neurotransmitter.GABA: "gaba",
                Neurotransmitter.GLUTAMATE: "glutamate",
            }
            for nt_enum, bio_name in nt_map.items():
                bio_nt = bio_homeostasis.neurotransmitters.get(bio_name)
                if bio_nt:
                    ns_level = self.neurotransmitters.levels[nt_enum]
                    blended = ns_level * 0.6 + bio_nt.current * 0.4
                    self.neurotransmitters.levels[nt_enum] = max(0.05, min(1.0, blended))
        except Exception as e:
            logger.debug(f"Neuroscience-bio sync failed: {e}")

    def to_dict(self) -> dict:
        return self.get_neural_state()