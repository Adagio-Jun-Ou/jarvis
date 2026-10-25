"""Thought Sense — Predictive Intent Modeling System for Jarvis.

Operates at the concept level (like LCM) to anticipate user mental states,
emotional trajectories, and unspoken needs before they become explicit.

Architecture:
  IntentSignals — What the user likely wants next (curiosity? frustration? flow?)
  EmotionalTrajectory — Where the user's emotional state is heading
  CognitiveLoad — Is the user overwhelmed? Under-stimulated? In flow?
  UnspokenNeeds — What the user hasn't said but probably needs
  ThoughtCompletion — Can we complete the user's partial thought?

Thought Sense integrates:
  1. PFC concept sequences (what concepts appeared in what order)
  2. Conversation rhythm (timing, message lengths, correction patterns)
  3. Topic drift patterns (where attention is shifting)
  4. Confidence signals (how certain is the PFC about predictions)
  5. Historical patterns (what happened after similar sequences before)

The output is a ThoughtState that other subsystems can query to adapt behavior:
  - Agent can adjust response depth based on cognitive load
  - ThinkingEngine can time proactive suggestions to flow valleys
  - Context assembly can pre-fetch likely-needed context
  - LCM training can be weighted by prediction accuracy feedback
"""

import hashlib
import logging
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Optional


from core.helpers import atomic_write_json, read_json_with_integrity

logger = logging.getLogger(__name__)

_singleton: Optional["ThoughtSense"] = None
_lock = threading.Lock()


class IntentSignal(Enum):
    EXPLORATION = auto()
    CONFIRMATION = auto()
    CORRECTION = auto()
    DEEP_DIVE = auto()
    TASK_SWITCH = auto()
    COMPLETION = auto()
    FRUSTRATION = auto()
    CURIOSITY = auto()
    REVIEW = auto()
    CREATIVE = auto()


class CognitiveState(Enum):
    FLOW = auto()
    ENGAGED = auto()
    EXPLORING = auto()
    DEEP_DIVE = auto()
    OVERLOADED = auto()
    IDLE = auto()
    FRUSTRATED = auto()


class EmotionalValence(Enum):
    POSITIVE = auto()
    NEUTRAL = auto()
    NEGATIVE = auto()
    MIXED = auto()


@dataclass
class ThoughtState:
    intent_signals: dict[IntentSignal, float] = field(default_factory=dict)
    cognitive_state: CognitiveState = CognitiveState.IDLE
    emotional_valence: EmotionalValence = EmotionalValence.NEUTRAL
    emotional_trajectory: float = 0.0
    cognitive_load: float = 0.0
    predicted_next_topics: list[str] = field(default_factory=list)
    predicted_needs: list[str] = field(default_factory=list)
    thought_completions: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ConversationPulse:
    message_times: deque = field(default_factory=lambda: deque(maxlen=50))
    message_lengths: deque = field(default_factory=lambda: deque(maxlen=50))
    correction_count: int = 0
    question_count: int = 0
    exclamation_count: int = 0
    topic_switches: int = 0
    last_topic: Optional[str] = None
    last_intent: Optional[IntentSignal] = None


class PatternMemory:
    """Stores and matches behavioral patterns for predictive modeling."""

    MAX_PATTERNS = 5000
    MIN_PATTERN_LENGTH = 2
    MAX_PATTERN_LENGTH = 10

    def __init__(self):
        self._lock = threading.Lock()
        self._patterns: dict[str, dict] = {}
        self._sequence_counts: Counter = Counter()

    def record_pattern(
        self,
        sequence: list[str],
        outcome: str,
        reward: float = 1.0,
    ):
        """Record a behavioral pattern and its outcome."""
        if len(sequence) < self.MIN_PATTERN_LENGTH:
            return

        sequence = sequence[-self.MAX_PATTERN_LENGTH:]
        pattern_key = hashlib.sha256("|".join(sequence).encode()).hexdigest()[:16]

        with self._lock:
            if pattern_key not in self._patterns:
                self._patterns[pattern_key] = {
                    "sequence": sequence,
                    "outcomes": Counter(),
                    "total_reward": 0.0,
                    "count": 0,
                }

            pattern = self._patterns[pattern_key]
            pattern["outcomes"][outcome] += 1
            pattern["total_reward"] += reward
            pattern["count"] += 1

            self._sequence_counts[pattern_key] += 1

            if len(self._patterns) > self.MAX_PATTERNS:
                oldest_key = min(self._patterns, key=lambda k: self._patterns[k]["count"])
                del self._patterns[oldest_key]
                del self._sequence_counts[oldest_key]

    def predict_outcomes(
        self,
        current_sequence: list[str],
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Predict likely outcomes given the current sequence."""
        if not current_sequence:
            return []

        current = current_sequence[-self.MAX_PATTERN_LENGTH:]

        with self._lock:
            matches = []
            for pattern_key, pattern in self._patterns.items():
                pseq = pattern["sequence"]
                if len(pseq) < 2:
                    continue

                overlap = len(set(current[-3:]) & set(pseq[-3:])) if len(current) >= 2 else 0
                if overlap == 0:
                    continue

                recency = min(1.0, pattern["count"] / 10)
                quality = pattern["total_reward"] / max(pattern["count"], 1)
                score = (overlap / 3) * recency * quality

                for outcome, count in pattern["outcomes"].most_common(top_k):
                    prob = count / max(pattern["count"], 1)
                    matches.append((outcome, score * prob))

            matches.sort(key=lambda x: x[1], reverse=True)
            return matches[:top_k]

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "total_patterns": len(self._patterns),
                "total_sequences": sum(self._sequence_counts.values()),
            }


class ThoughtSense:
    """Predictive intent modeling system.

    Observes conversation patterns and PFC concept sequences to build
    a mental model of the user's likely intent, emotional state, and
    unspoken needs.
    """

    _MAX_HISTORY = 200
    _MAX_TOPIC_HISTORY = 100
    _MAX_TOPIC_COUNTS = 500
    _MAX_INTENT_TRANSITIONS = 100
    _SAVE_INTERVAL = 300  # Save every 5 minutes

    def __init__(self):
        self._lock = threading.RLock()
        self._pfc = None

        self._pattern_memory = PatternMemory()
        self._conversation_pulse = ConversationPulse()

        self._state = ThoughtState()
        self._state_history: deque[ThoughtState] = deque(maxlen=self._MAX_HISTORY)

        self._topic_history: deque[str] = deque(maxlen=self._MAX_TOPIC_HISTORY)
        self._topic_counts: Counter = Counter()

        self._intent_transition_counts: dict[IntentSignal, Counter] = {}
        for intent in IntentSignal:
            self._intent_transition_counts[intent] = Counter()

        self._last_save_time = time.time()
        self._observation_count = 0

        self._cognitive_features = {
            "avg_message_gap": 0.0,
            "avg_message_length": 0.0,
            "topic_switch_rate": 0.0,
            "question_ratio": 0.0,
            "correction_ratio": 0.0,
            "depth_trend": 0.0,
        }

    def observe(
        self,
        content: str,
        concept_type: Optional[str] = None,
        source: Optional[str] = None,
        pfc_prediction: Optional[dict] = None,
    ):
        """Observe a user interaction and update the mental model.

        Args:
            content: The user's message or interaction content
            concept_type: PFC concept type if available (defaults to "observation")
            source: Source of the observation
            pfc_prediction: PFC prediction data for integration
        """
        if not content or not content.strip():
            return

        concept_type = concept_type or "observation"

        with self._lock:
            now = time.time()
            content_len = len(content)

            self._observation_count += 1

            self._conversation_pulse.message_times.append(now)
            self._conversation_pulse.message_lengths.append(content_len)

            is_question = "?" in content
            is_correction = any(
                w in content.lower()
                for w in ["not what i meant", "actually", "sorry", "no,", "wrong", "i meant", "let me rephrase"]
            )
            is_exclamation = "!" in content

            if is_question:
                self._conversation_pulse.question_count += 1
            if is_correction:
                self._conversation_pulse.correction_count += 1
            if is_exclamation:
                self._conversation_pulse.exclamation_count += 1

            self._update_cognitive_features()

            detected_intent = self._detect_intent(
                content, is_question, is_correction, is_exclamation, pfc_prediction
            )

            topic = self._extract_topic(content)
            if topic:
                if self._conversation_pulse.last_topic and topic != self._conversation_pulse.last_topic:
                    self._conversation_pulse.topic_switches += 1
                self._topic_history.append(topic)
                self._topic_counts[topic] += 1
                # Cap topic counts
                if len(self._topic_counts) > self._MAX_TOPIC_COUNTS:
                    for t in list(self._topic_counts.keys())[:len(self._topic_counts) - self._MAX_TOPIC_COUNTS]:
                        del self._topic_counts[t]
                self._conversation_pulse.last_topic = topic

            if self._conversation_pulse.last_intent and detected_intent:
                transition = self._intent_transition_counts[self._conversation_pulse.last_intent]
                transition[detected_intent] += 1
                # Cap per-intent transitions
                if len(transition) > self._MAX_INTENT_TRANSITIONS:
                    for k in list(transition.keys())[:len(transition) - self._MAX_INTENT_TRANSITIONS]:
                        del transition[k]

            self._conversation_pulse.last_intent = detected_intent

            emotional_valence = self._detect_emotional_valence(content, is_correction, is_exclamation)

            cognitive_load = self._estimate_cognitive_load()
            flow_score = self._estimate_flow_score()
            cognitive_state = self._classify_cognitive_state(flow_score, cognitive_load)

            self._state = ThoughtState(
                intent_signals={detected_intent: self._state.intent_signals.get(detected_intent, 0.0) * 0.7 + 0.3}
                if detected_intent
                else self._state.intent_signals,
                cognitive_state=cognitive_state,
                emotional_valence=emotional_valence,
                emotional_trajectory=self._estimate_emotional_trajectory(emotional_valence),
                cognitive_load=cognitive_load,
                predicted_next_topics=self._predict_next_topics(),
                predicted_needs=self._predict_needs(cognitive_state, cognitive_load),
                thought_completions=self._generate_thought_completions(content),
                confidence=self._estimate_prediction_confidence(),
                timestamp=datetime.now(),
            )

            self._state_history.append(self._state)

            try:
                from core.cognitive_load_balancer import get_load_balancer
                lb = get_load_balancer()
                lb.adjust_for_cognitive_state(
                    cognitive_state.name, cognitive_load
                )
            except Exception as e:
                logger.debug(f"Failed to adjust load balancer for cognitive state: {e}")

            sequence = list(self._topic_history)[-10:]
            if len(sequence) >= 2 and detected_intent:
                self._pattern_memory.record_pattern(
                    sequence, detected_intent.name, reward=1.0
                )

            if now - self._last_save_time > self._SAVE_INTERVAL:
                self._save_state()
                self._last_save_time = now

    def get_state(self) -> ThoughtState:
        """Get the current thought state."""
        with self._lock:
            return self._state

    def get_intent_signals(self) -> dict[IntentSignal, float]:
        """Get current intent signal strengths."""
        with self._lock:
            return dict(self._state.intent_signals)

    def should_proactive_suggest(self) -> tuple[bool, str]:
        """Determine if a proactive suggestion is appropriate and what kind.

        Returns: (should_suggest, suggestion_type)
        """
        with self._lock:
            state = self._state

            if state.cognitive_state == CognitiveState.OVERLOADED:
                return False, "user_overloaded"

            if state.cognitive_state == CognitiveState.FLOW:
                return False, "in_flow"

            if state.cognitive_state == CognitiveState.IDLE:
                return True, "idle_suggestion"

            if state.emotional_valence == EmotionalValence.NEGATIVE:
                return True, "encouragement"

            if state.cognitive_load < 0.3 and len(state.predicted_next_topics) > 0:
                return True, "anticipatory"

            if state.cognitive_state == CognitiveState.EXPLORING:
                return True, "exploration_nudge"

            if state.cognitive_load > 0.7:
                return True, "simplification"

            return False, "not_appropriate"

    def get_adapted_response_depth(self) -> str:
        """Suggest response depth based on cognitive state."""
        with self._lock:
            load = self._state.cognitive_load
            if self._state.cognitive_state == CognitiveState.OVERLOADED:
                return "minimal"
            if load > 0.7:
                return "concise"
            if load > 0.4:
                return "standard"
            if self._state.cognitive_state == CognitiveState.DEEP_DIVE:
                return "detailed"
            return "standard"

    def get_adapted_tone(self) -> str:
        """Suggest response tone based on emotional state."""
        with self._lock:
            if self._state.emotional_valence == EmotionalValence.NEGATIVE:
                return "empathetic"
            if self._state.emotional_valence == EmotionalValence.POSITIVE:
                return "enthusiastic"
            if self._state.cognitive_state == CognitiveState.EXPLORING:
                return "curious"
            if self._state.cognitive_state == CognitiveState.FLOW:
                return "focused"
            return "balanced"

    def get_thought_completions(self, max_completions: int = 3) -> list[dict]:
        """Get thought completions for the current context.
        
        Returns completions sorted by confidence, suitable for
        suggesting to the user as autocomplete hints or pre-fill.
        
        Args:
            max_completions: Maximum number of completions to return
            
        Returns:
            List of dicts with 'text' and 'confidence' keys
        """
        with self._lock:
            completions = self._state.thought_completions
            if not completions:
                return []
            result = []
            for topic, score in sorted(completions.items(), key=lambda x: x[1], reverse=True)[:max_completions]:
                if score > 0.1:
                    result.append({"text": topic, "confidence": round(score, 3)})
            return result

    def get_proactive_suggestion_delay(self) -> float:
        """Get the recommended delay before making a proactive suggestion.
        
        Based on cognitive state and emotional trajectory:
        - FLOW: Long delay (60s+) — do not disturb
        - OVERLOADED: Long delay (45s) — wait for load to decrease  
        - ENGAGED: Medium delay (20s)
        - EXPLORING: Short delay (10s) — user is receptive
        - IDLE: Immediate (0s) — user is waiting
        
        Returns:
            Recommended delay in seconds
        """
        with self._lock:
            state = self._state
            
            delay_map = {
                CognitiveState.FLOW: 60.0,
                CognitiveState.OVERLOADED: 45.0,
                CognitiveState.FRUSTRATED: 30.0,
                CognitiveState.ENGAGED: 20.0,
                CognitiveState.DEEP_DIVE: 25.0,
                CognitiveState.EXPLORING: 10.0,
                CognitiveState.IDLE: 0.0,
            }
            
            base_delay = delay_map.get(state.cognitive_state, 15.0)
            
            # Adjust based on emotional trajectory
            if state.emotional_trajectory > 0.3:
                base_delay *= 0.7  # User improving — shorter delay
            elif state.emotional_trajectory < -0.3:
                base_delay *= 1.3  # User declining — longer delay
            
            # Adjust based on cognitive load
            if state.cognitive_load > 0.7:
                base_delay *= 1.5
            elif state.cognitive_load < 0.3:
                base_delay *= 0.5
            
            return min(base_delay, 90.0)  # Cap at 90 seconds

    def get_prediction_accuracy(self) -> float:
        """Estimate prediction accuracy based on pattern memory hit rate."""
        stats = self._pattern_memory.get_stats()
        if stats["total_patterns"] == 0:
            return 0.0
        return min(1.0, stats["total_sequences"] / max(stats["total_patterns"], 1))

    def _update_cognitive_features(self):
        """Update computed cognitive features from conversation pulse."""
        pulse = self._conversation_pulse

        if len(pulse.message_times) >= 2:
            gaps = [
                pulse.message_times[i] - pulse.message_times[i - 1]
                for i in range(1, len(pulse.message_times))
            ]
            self._cognitive_features["avg_message_gap"] = sum(gaps) / len(gaps)

        if len(pulse.message_lengths) > 0:
            self._cognitive_features["avg_message_length"] = sum(pulse.message_lengths) / len(pulse.message_lengths)

        total_messages = max(len(pulse.message_times), 1)
        self._cognitive_features["topic_switch_rate"] = pulse.topic_switches / total_messages
        self._cognitive_features["question_ratio"] = pulse.question_count / total_messages
        self._cognitive_features["correction_ratio"] = pulse.correction_count / total_messages

        if len(pulse.message_lengths) >= 2:
            recent = list(pulse.message_lengths)[-5:]
            older = list(pulse.message_lengths)[:-5] if len(pulse.message_lengths) > 5 else list(pulse.message_lengths)[:-1]
            if older:
                self._cognitive_features["depth_trend"] = (sum(recent) / len(recent)) / (sum(older) / len(older) + 0.01) - 1.0

    def _detect_intent(
        self,
        content: str,
        is_question: bool,
        is_correction: bool,
        is_exclamation: bool,
        pfc_prediction: Optional[dict],
    ) -> Optional[IntentSignal]:
        """Detect the user's primary intent from the current message."""
        content_lower = content.lower()

        if is_correction:
            return IntentSignal.CORRECTION

        curiosity_words = ["what if", "how does", "why", "what would happen", "curious", "wonder"]
        if any(w in content_lower for w in curiosity_words) or is_question:
            return IntentSignal.CURIOSITY if "?" in content else IntentSignal.EXPLORATION

        depth_words = ["explain", "elaborate", "dive deeper", "tell me more", "detail"]
        if any(w in content_lower for w in depth_words):
            return IntentSignal.DEEP_DIVE

        task_switch_words = ["switch", "different topic", "instead", "let's do", "move on", "actually"]
        if any(w in content_lower for w in task_switch_words):
            return IntentSignal.TASK_SWITCH

        completion_words = ["done", "thanks", "that works", "perfect", "got it", "great"]
        if any(w in content_lower for w in completion_words):
            return IntentSignal.COMPLETION

        review_words = ["review", "summarize", "recap", "overview", "status"]
        if any(w in content_lower for w in review_words):
            return IntentSignal.REVIEW

        creative_words = ["create", "design", "imagine", "build", "compose"]
        if any(w in content_lower for w in creative_words):
            return IntentSignal.CREATIVE

        frustration_words = ["not working", "still broken", "error again", "keeps failing", "frustrated"]
        if any(w in content_lower for w in frustration_words):
            return IntentSignal.FRUSTRATION

        if is_question:
            return IntentSignal.CURIOSITY

        if pfc_prediction and pfc_prediction.get("type_probs"):
            top_type = pfc_prediction["type_probs"][0][0] if pfc_prediction["type_probs"] else None
            type_map = {
                "intention": IntentSignal.EXPLORATION,
                "insight": IntentSignal.CURIOSITY,
                "tool_call": IntentSignal.TASK_SWITCH,
                "model_response": IntentSignal.CONFIRMATION,
            }
            if top_type in type_map:
                return type_map[top_type]

        return IntentSignal.EXPLORATION

    def _detect_emotional_valence(
        self, content: str, is_correction: bool, is_exclamation: bool
    ) -> EmotionalValence:
        """Detect emotional valence from content and conversation patterns."""
        content_lower = content.lower()

        positive_words = ["great", "awesome", "love", "perfect", "amazing", "excellent", "wonderful", "thanks"]
        negative_words = ["frustrated", "annoyed", "wrong", "broken", "fail", "error", "doesn't work", "hate"]
        _mixed_words = ["but", "however", "although", "except", "mixed"]

        pos_count = sum(1 for w in positive_words if w in content_lower)
        neg_count = sum(1 for w in negative_words if w in content_lower)

        if is_correction:
            neg_count += 1

        if pos_count > neg_count + 1:
            return EmotionalValence.POSITIVE
        if neg_count > pos_count + 1:
            return EmotionalValence.NEGATIVE
        if pos_count > 0 and neg_count > 0:
            return EmotionalValence.MIXED

        return EmotionalValence.NEUTRAL

    def _estimate_emotional_trajectory(self, current: EmotionalValence) -> float:
        """Estimate emotional trajectory (-1 declining to +1 improving)."""
        valence_map = {
            EmotionalValence.POSITIVE: 1.0,
            EmotionalValence.NEUTRAL: 0.0,
            EmotionalValence.NEGATIVE: -1.0,
            EmotionalValence.MIXED: 0.0,
        }

        if len(self._state_history) < 2:
            return valence_map.get(current, 0.0)

        current_valence = valence_map.get(current, 0.0)
        prev_valence = valence_map.get(self._state_history[-1].emotional_valence, 0.0)

        trajectory = current_valence - prev_valence

        if len(self._state_history) >= 3:
            older_valence = valence_map.get(self._state_history[-2].emotional_valence, 0.0)
            trajectory = 0.5 * trajectory + 0.5 * (current_valence - older_valence)

        max_t = 1.0
        return max(-max_t, min(max_t, trajectory))

    def _estimate_cognitive_load(self) -> float:
        """Estimate the user's cognitive load (0.0 = idle, 1.0 = overloaded)."""
        features = self._cognitive_features

        load = 0.3

        gap = features.get("avg_message_gap", 0)
        if gap < 2:
            load += 0.2
        elif gap < 5:
            load += 0.1
        elif gap > 30:
            load -= 0.1

        length = features.get("avg_message_length", 0)
        if length > 200:
            load += 0.15
        elif length > 100:
            load += 0.05

        switch_rate = features.get("topic_switch_rate", 0)
        if switch_rate > 0.5:
            load += 0.2
        elif switch_rate > 0.2:
            load += 0.1

        correction_rate = features.get("correction_ratio", 0)
        if correction_rate > 0.3:
            load += 0.15

        question_rate = features.get("question_ratio", 0)
        if question_rate > 0.5:
            load += 0.1

        return max(0.0, min(1.0, load))

    def _estimate_flow_score(self) -> float:
        """Estimate flow state score (0.0 = no flow, 1.0 = deep flow)."""
        features = self._cognitive_features

        flow = 0.5

        gap = features.get("avg_message_gap", 0)
        if 3 < gap < 15:
            flow += 0.2
        elif gap < 3:
            flow += 0.1
        elif gap > 60:
            flow -= 0.2

        depth_trend = features.get("depth_trend", 0)
        if depth_trend > 0.1:
            flow += 0.15
        elif depth_trend < -0.1:
            flow -= 0.1

        question_rate = features.get("question_ratio", 0)
        if question_rate < 0.2:
            flow += 0.1
        elif question_rate > 0.6:
            flow -= 0.1

        correction_rate = features.get("correction_ratio", 0)
        if correction_rate < 0.05:
            flow += 0.1
        elif correction_rate > 0.2:
            flow -= 0.2

        return max(0.0, min(1.0, flow))

    def _classify_cognitive_state(
        self, flow_score: float, cognitive_load: float
    ) -> CognitiveState:
        """Classify the cognitive state based on flow and load."""
        if flow_score > 0.7 and cognitive_load < 0.6:
            return CognitiveState.FLOW
        if cognitive_load > 0.8:
            return CognitiveState.OVERLOADED

        current_intent = self._conversation_pulse.last_intent

        if current_intent == IntentSignal.FRUSTRATION:
            return CognitiveState.FRUSTRATED
        if current_intent == IntentSignal.DEEP_DIVE:
            return CognitiveState.DEEP_DIVE
        if current_intent == IntentSignal.EXPLORATION or current_intent == IntentSignal.CURIOSITY:
            return CognitiveState.EXPLORING

        if flow_score > 0.4:
            return CognitiveState.ENGAGED
        if cognitive_load < 0.3:
            return CognitiveState.IDLE

        return CognitiveState.ENGAGED

    def _predict_next_topics(self, top_k: int = 3) -> list[str]:
        """Predict likely next topics based on patterns and intent transitions."""
        predicted = []

        if len(self._topic_history) < 2:
            return predicted

        recent = list(self._topic_history)[-5:]

        # Pattern memory prediction
        outcomes = self._pattern_memory.predict_outcomes(recent, top_k=top_k)
        for outcome, score in outcomes:
            if score > 0.05:
                predicted.append(outcome)

        # Intent transition prediction: use recent intent to boost topic predictions
        last_intent = self._conversation_pulse.last_intent
        if last_intent and last_intent in self._intent_transition_counts:
            transitions = self._intent_transition_counts[last_intent]
            if transitions:
                total = sum(transitions.values())
                for next_intent, count in transitions.most_common(3):
                    prob = count / max(total, 1)
                    # Map intent transitions to topic predictions
                    intent_topic_map = {
                        IntentSignal.CURIOSITY: "research",
                        IntentSignal.DEEP_DIVE: "detailed_analysis",
                        IntentSignal.EXPLORATION: "discovery",
                        IntentSignal.TASK_SWITCH: "new_topic",
                        IntentSignal.CORRECTION: "refinement",
                        IntentSignal.COMPLETION: "summary",
                        IntentSignal.FRUSTRATION: "troubleshooting",
                        IntentSignal.CREATIVE: "innovation",
                        IntentSignal.REVIEW: "recap",
                        IntentSignal.CONFIRMATION: "verification",
                    }
                    mapped_topic = intent_topic_map.get(next_intent)
                    if mapped_topic and mapped_topic not in predicted and prob > 0.15:
                        predicted.append(mapped_topic)

        return predicted[:top_k]

    def _predict_needs(
        self, cognitive_state: CognitiveState, cognitive_load: float
    ) -> list[str]:
        """Predict user needs based on their state."""
        needs = []

        if cognitive_load > 0.7:
            needs.append("simplification")
            needs.append("summary")

        if cognitive_state == CognitiveState.FRUSTRATED:
            needs.append("encouragement")
            needs.append("simpler_explanation")

        if cognitive_state == CognitiveState.EXPLORING:
            needs.append("related_topics")
            needs.append("context_expansion")

        if cognitive_state == CognitiveState.IDLE:
            needs.append("suggestion")
            needs.append("conversation_starter")

        if cognitive_state == CognitiveState.FLOW:
            needs.append("minimal_interruption")

        if cognitive_state == CognitiveState.DEEP_DIVE:
            needs.append("detailed_analysis")
            needs.append("technical_depth")

        if self._state.emotional_valence == EmotionalValence.NEGATIVE:
            needs.append("acknowledgment")

        return needs[:5]

    def _generate_thought_completions(self, content: str) -> dict[str, float]:
        """Generate likely thought completions for partial user input."""
        completions = {}

        content_lower = content.lower().strip()

        if not content_lower or len(content_lower) < 5:
            return completions

        for topic, count in self._topic_counts.most_common(10):
            recent_history = list(self._topic_history)[-5:]
            if topic in recent_history:
                completions[topic] = min(0.9, count / (self._observation_count + 1) * 10)

        pattern_outcomes = self._pattern_memory.predict_outcomes(list(self._topic_history)[-5:], top_k=3)
        for outcome, score in pattern_outcomes:
            completions[outcome] = max(completions.get(outcome, 0), score)

        return dict(sorted(completions.items(), key=lambda x: x[1], reverse=True)[:5])

    def _extract_topic(self, content: str) -> Optional[str]:
        """Extract a topic from user content."""
        import re

        content_lower = content.lower().strip()
        content_lower = re.sub(r'[^\w\s]', '', content_lower)
        words = content_lower.split()

        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "can", "shall", "to", "of", "in", "for",
            "on", "with", "at", "by", "from", "as", "into", "about", "like",
            "through", "over", "after", "between", "out", "against", "during",
            "without", "before", "under", "around", "among", "i", "me", "my",
            "we", "us", "our", "you", "your", "it", "its", "this", "that",
            "these", "those", "what", "which", "who", "whom", "when", "where",
            "why", "how", "all", "each", "every", "both", "few", "more", "most",
            "other", "some", "such", "no", "nor", "not", "only", "own", "same",
            "so", "than", "too", "very", "just", "because", "but", "and", "or",
            "if", "while", "then", "else", "also", "please", "help",
            "want", "need", "know", "think", "make", "tell", "give", "try",
        }

        meaningful = [w for w in words if w not in stop_words and len(w) > 2]

        if not meaningful:
            return None

        if len(meaningful) >= 2:
            bigrams = [f"{meaningful[i]}_{meaningful[i+1]}" for i in range(len(meaningful) - 1)]
            return bigrams[0]

        return meaningful[0]

    def _estimate_prediction_confidence(self) -> float:
        """Estimate confidence in predictions based on data available."""
        observation_factor = min(1.0, self._observation_count / 50)
        pattern_factor = min(1.0, self._pattern_memory.get_stats()["total_patterns"] / 100)
        history_factor = min(1.0, len(self._state_history) / 20)

        return 0.4 * observation_factor + 0.3 * pattern_factor + 0.3 * history_factor

    def _save_state(self):
        """Save thought sense state to disk."""
        storage_path = Path.home() / ".jarvis" / "thought_sense"
        storage_path.mkdir(parents=True, exist_ok=True)

        state_data = {
            "observation_count": self._observation_count,
            "topic_counts": dict(self._topic_counts.most_common(100)),
            "cognitive_features": self._cognitive_features,
            "pulse_question_count": self._conversation_pulse.question_count,
            "pulse_correction_count": self._conversation_pulse.correction_count,
            "pulse_topic_switches": self._conversation_pulse.topic_switches,
        }

        try:
            atomic_write_json(storage_path / "state.json", state_data)
        except Exception as e:
            logger.debug(f"Thought sense state save failed: {e}")

    def load_state(self):
        """Load thought sense state from disk."""
        storage_path = Path.home() / ".jarvis" / "thought_sense"
        state_file = storage_path / "state.json"

        if not state_file.exists():
            return

        try:

            data = read_json_with_integrity(state_file)
            self._observation_count = data.get("observation_count", 0)

            for topic, count in data.get("topic_counts", {}).items():
                self._topic_counts[topic] = count

            self._cognitive_features = data.get(
                "cognitive_features", self._cognitive_features
            )
            self._conversation_pulse.question_count = data.get("pulse_question_count", 0)
            self._conversation_pulse.correction_count = data.get("pulse_correction_count", 0)
            self._conversation_pulse.topic_switches = data.get("pulse_topic_switches", 0)

        except Exception as e:
            logger.debug(f"Thought sense state load failed: {e}")

    def get_stats(self) -> dict:
        """Get thought sense statistics."""
        with self._lock:
            return {
                "observation_count": self._observation_count,
                "current_state": {
                    "cognitive": self._state.cognitive_state.value,
                    "emotional": self._state.emotional_valence.value,
                    "load": round(self._state.cognitive_load, 2),
                    "trajectory": round(self._state.emotional_trajectory, 2),
                    "confidence": round(self._state.confidence, 2),
                },
                "pattern_memory": self._pattern_memory.get_stats(),
                "topic_diversity": len(self._topic_counts),
                "prediction_accuracy": round(self.get_prediction_accuracy(), 2),
                "adapted_depth": self.get_adapted_response_depth(),
                "adapted_tone": self.get_adapted_tone(),
            }

    def get_conversation_rhythm(self) -> dict:
        """Analyze the current conversation rhythm for adaptive behavior.

        Returns rhythm metrics that can inform response timing, depth, and tone:
        - cadence: 'rapid', 'moderate', 'slow', 'idle' — how fast messages are coming
        - tempo_stability: 0.0-1.0 — how consistent the rhythm is (high = predictable)
        - pause_pressure: 0.0-1.0 — how likely a natural pause is coming
        - engagement_depth: 0.0-1.0 — how deeply engaged the user is
        - optimal_response_delay: seconds to wait before responding
        """
        with self._lock:
            pulse = self._conversation_pulse
            
            if len(pulse.message_times) < 2:
                return {
                    "cadence": "idle",
                    "tempo_stability": 0.0,
                    "pause_pressure": 0.0,
                    "engagement_depth": 0.0,
                    "optimal_response_delay": 2.0,
                }

            # Calculate inter-message gaps
            times = list(pulse.message_times)
            gaps = [times[i] - times[i - 1] for i in range(1, len(times))]
            avg_gap = sum(gaps) / len(gaps) if gaps else 30.0

            # Cadence classification
            if avg_gap < 3:
                cadence = "rapid"
            elif avg_gap < 10:
                cadence = "moderate"
            elif avg_gap < 30:
                cadence = "slow"
            else:
                cadence = "idle"

            # Tempo stability — how consistent are the gaps?
            if len(gaps) >= 3:
                mean_gap = avg_gap
                variance = sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)
                std_dev = variance ** 0.5
                stability = max(0.0, 1.0 - (std_dev / (mean_gap + 0.01)))
            else:
                stability = 0.5

            # Pause pressure — is a natural conversation pause approaching?
            recent_gap = gaps[-1] if gaps else 30.0
            if cadence == "rapid":
                # Rapid exchanges naturally end with a pause
                pause_pressure = min(1.0, len(gaps) * 0.1) if recent_gap < avg_gap else 0.8
            elif cadence == "moderate":
                pause_pressure = 0.5
            else:
                pause_pressure = 0.2

            # Engagement depth — is the user going deeper or shallower?
            if len(pulse.message_lengths) >= 4:
                recent_lengths = list(pulse.message_lengths)[-4:]
                older_lengths = list(pulse.message_lengths)[:-4][-4:] if len(pulse.message_lengths) > 4 else recent_lengths
                avg_recent = sum(recent_lengths) / len(recent_lengths)
                avg_older = sum(older_lengths) / len(older_lengths) if older_lengths else avg_recent
                depth_ratio = avg_recent / (avg_older + 1)
                engagement_depth = min(1.0, max(0.0, (depth_ratio - 0.5) * 2))
            else:
                engagement_depth = 0.5

            # Optimal response delay
            if cadence == "rapid":
                optimal_delay = 0.5
            elif cadence == "moderate":
                optimal_delay = avg_gap * 0.3
            elif cadence == "slow":
                optimal_delay = avg_gap * 0.2
            else:
                optimal_delay = 2.0

            # Adjust based on cognitive state
            if self._state.cognitive_state == CognitiveState.FLOW:
                optimal_delay *= 1.5  # Don't interrupt flow
            elif self._state.cognitive_state == CognitiveState.OVERLOADED:
                optimal_delay *= 2.0  # Give more space

            optimal_delay = min(optimal_delay, 10.0)

            return {
                "cadence": cadence,
                "tempo_stability": round(stability, 2),
                "pause_pressure": round(pause_pressure, 2),
                "engagement_depth": round(engagement_depth, 2),
                "optimal_response_delay": round(optimal_delay, 1),
            }


def get_thought_sense() -> ThoughtSense:
    """Get singleton ThoughtSense instance."""
    global _singleton
    if _singleton is None:
        with _lock:
            if _singleton is None:
                _singleton = ThoughtSense()
                _singleton.load_state()
    return _singleton