"""Sentiment Analyzer — Goes beyond keyword matching to understand nuance.

The existing butler_personality does basic keyword matching for frustration/gratitude.
This module provides deeper analysis:

1. **Multi-dimensional sentiment** — not just positive/negative, but frustration,
   curiosity, urgency, confusion, satisfaction, and more.

2. **Intensity scoring** — "I'm furious" vs "I'm a bit annoyed" vs "meh"

3. **Context-awareness** — "this is bad" in debugging context ≠ negative sentiment

4. **Temporal tracking** — Sentiment trajectory over a conversation, not just per-turn

5. **Signal extraction** — What the user needs (help, explanation, confirmation,
   reassurance, challenge, agreement)
"""
import logging
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

_analyzer: Optional["SentimentAnalyzer"] = None
_analyzer_lock = threading.Lock()


class SentimentType(Enum):
    NEUTRAL = "neutral"
    FRUSTRATED = "frustrated"
    CURIOUS = "curious"
    EXCITED = "excited"
    ANXIOUS = "anxious"
    CONFUSED = "confused"
    SATISFIED = "satisfied"
    GRATEFUL = "grateful"
    DETERMINED = "determined"
    OVERWHELMED = "overwhelmed"
    PLAYFUL = "playful"
    URGENCY = "urgency"


class NeedType(Enum):
    HELP = "help"
    EXPLANATION = "explanation"
    CONFIRMATION = "confirmation"
    REASSURANCE = "reassurance"
    CHALLENGE = "challenge"
    AGREEMENT = "agreement"
    ACTION = "action"
    FEEDBACK = "feedback"
    EXPLORATION = "exploration"


@dataclass
class SentimentResult:
    sentiment: SentimentType = SentimentType.NEUTRAL
    intensity: float = 0.0
    confidence: float = 0.0
    needs: list[NeedType] = field(default_factory=list)
    signals: list[str] = field(default_factory=list)
    escalation_risk: float = 0.0
    requires_response_change: bool = False


@dataclass
class SentimentTrajectory:
    sentiments: list[SentimentType] = field(default_factory=list)
    intensities: list[float] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)
    trend: str = "stable"
    trend_confidence: float = 0.0

    def add(self, sentiment: SentimentType, intensity: float) -> None:
        self.sentiments.append(sentiment)
        self.intensities.append(intensity)
        self.timestamps.append(time.time())
        if len(self.sentiments) > 50:
            self.sentiments = self.sentiments[-50:]
            self.intensities = self.intensities[-50:]
            self.timestamps = self.timestamps[-50:]

    def compute_trend(self) -> None:
        if len(self.intensities) < 3:
            self.trend = "stable"
            self.trend_confidence = 0.3
            return
        recent = self.intensities[-5:]
        older = self.intensities[-10:-5] if len(self.intensities) > 5 else recent
        avg_recent = sum(recent) / len(recent)
        avg_older = sum(older) / len(older)
        diff = avg_recent - avg_older
        if diff > 0.15:
            self.trend = "escalating"
            self.trend_confidence = min(1.0, abs(diff) * 3)
        elif diff < -0.15:
            self.trend = "de_escalating"
            self.trend_confidence = min(1.0, abs(diff) * 3)
        else:
            self.trend = "stable"
            self.trend_confidence = 0.5

    def is_escalating(self) -> bool:
        return self.trend == "escalating" and self.trend_confidence > 0.4


_INTENSITY_MULTIPLIERS = {
    "very": 1.4, "really": 1.3, "extremely": 1.5, "super": 1.3, "so": 1.2,
    "absolutely": 1.4, "incredibly": 1.5, "quite": 1.1, "rather": 1.1,
    "somewhat": 0.7, "slightly": 0.6, "a bit": 0.7, "kind of": 0.6,
    "mildly": 0.6, "fairly": 0.8, "barely": 0.5,
}

_NEGATION_WORDS = {"not", "no", "don't", "dont", "isn't", "isnt", "wasn't", "wasnt",
                   "can't", "cant", "won't", "wont", "shouldn't", "shouldnt",
                   "wouldn't", "wouldnt", "never", "hardly", "barely"}

_SENTIMENT_PATTERNS: list[tuple[SentimentType, list[str], float]] = [
    (SentimentType.FRUSTRATED, [
        "frustrated", "frustrating", "frustration", "frustrates", "annoyed", "irritated", "angry", "mad", "furious",
        "pissed", "aggravated", "stuck", "broken", "not working", "doesn't work",
        "doesn't help", "useless", "waste of time", "this is terrible",
        "giving up", "fed up", "sick of", "can't stand",
    ], 0.7),
    (SentimentType.CURIOUS, [
        "how does", "why does", "what if", "can you explain", "tell me about",
        "curious", "interested", "wonder", "what happens when", "how come",
        "interesting", "fascinating", "I'd like to understand",
    ], 0.6),
    (SentimentType.EXCITED, [
        "awesome", "amazing", "great", "excellent", "love it", "fantastic",
        "wonderful", "brilliant", "perfect", "exciting", "cool",
        "this is great", "really helpful", "nailed it", "exactly what",
    ], 0.7),
    (SentimentType.ANXIOUS, [
        "worried", "concerned", "anxious", "nervous", "scared", "afraid",
        "urgent", "deadline", "need this now", "running out of time",
        "what if something goes wrong", "hope this works",
    ], 0.6),
    (SentimentType.CONFUSED, [
        "confused", "don't understand", "doesn't make sense", "unclear",
        "lost", "what do you mean", "I don't get", "over my head",
        "complicated", "can you clarify", "I'm not following",
    ], 0.6),
    (SentimentType.SATISFIED, [
        "that works", "thanks", "got it", "makes sense", "perfect",
        "exactly", "just what I needed", "that's it", "nailed it",
        "problem solved", "all good", "much better",
    ], 0.5),
    (SentimentType.GRATEFUL, [
        "thank you", "thanks", "appreciate", "helpful", "grateful",
        "you're the best", "lifesaver", "couldn't have done without",
        "this is exactly", "you rock",
    ], 0.6),
    (SentimentType.DETERMINED, [
        "I need to", "let's do this", "going to fix", "won't give up",
        "push through", "make it work", "figure this out", "determined",
        "focus", "concentrate",
    ], 0.5),
    (SentimentType.OVERWHELMED, [
        "overwhelmed", "too much", "can't handle", "so many", "where do I start",
        "lost track", "spinning my wheels", "behind", "catching up",
        "drowning", "snowed under",
    ], 0.6),
    (SentimentType.PLAYFUL, [
        "lol", "haha", "funny", "joke", "play", "experiment", "what if",
        "just for fun", "messing around", "try something wild",
    ], 0.4),
    (SentimentType.URGENCY, [
        "asap", "immediately", "urgent", "critical", "emergency",
        "right now", "need this done", "deadline", "hurry",
        "can't wait", "pressing",
    ], 0.8),
]

_NEED_PATTERNS: list[tuple[NeedType, list[str], float]] = [
    (NeedType.HELP, ["help", "assist", "guide", "support", "stuck", "can't figure"], 0.6),
    (NeedType.EXPLANATION, ["explain", "understand", "how does", "why", "what is", "teach me"], 0.6),
    (NeedType.CONFIRMATION, ["is this right", "correct", "does this look", "am I doing this right", "verify"], 0.5),
    (NeedType.REASSURANCE, ["worried", "concerned", "scared", "anxious", "what if", "hope"], 0.5),
    (NeedType.CHALLENGE, ["can you do better", "prove it", "show me", "convince me"], 0.4),
    (NeedType.AGREEMENT, ["right?", "agreed?", "don't you think", "makes sense?", "yeah?"], 0.4),
    (NeedType.ACTION, ["do this", "fix it", "implement", "change", "update", "add", "remove", "create"], 0.5),
    (NeedType.FEEDBACK, ["review", "check", "test", "evaluate", "assess", "look at this"], 0.4),
    (NeedType.EXPLORATION, ["try", "experiment", "what about", "explore", "what if we", "idea"], 0.4),
]

_MAX_HISTORY = 100
_ESCALATION_THRESHOLD = 0.7


class SentimentAnalyzer:
    def __init__(self):
        self._lock = threading.RLock()
        self._trajectory = SentimentTrajectory()
        self._recent_sentiments: deque[SentimentResult] = deque(maxlen=_MAX_HISTORY)
        self._escalation_count: int = 0
        self._de_escalation_count: int = 0

    def analyze(self, text: str, context: str = "") -> SentimentResult:
        with self._lock:
            if not text or len(text.strip()) < 2:
                return SentimentResult()
            text_lower = text.lower().strip()
            words = text_lower.split()
            has_negation = any(w in _NEGATION_WORDS for w in words[:5])
            intensity_multiplier = self._compute_intensity_multiplier(words)
            scores: dict[SentimentType, float] = {}
            matched_signals: list[str] = []
            for sentiment, patterns, base_score in _SENTIMENT_PATTERNS:
                for pattern in patterns:
                    if pattern in text_lower:
                        score = base_score * intensity_multiplier
                        if has_negation:
                            score *= 0.3
                        scores[sentiment] = max(scores.get(sentiment, 0), score)
                        matched_signals.append(pattern)
                        break
            if len(text) < 20 and not scores:
                scores[SentimentType.NEUTRAL] = 0.3
            if not scores:
                scores[SentimentType.NEUTRAL] = 0.4
            has_question = "?" in text or any(w in text_lower for w in ["how", "why", "what", "when", "where", "which"])
            has_exclamation = "!" in text
            caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
            if has_question and SentimentType.CURIOUS not in scores:
                scores[SentimentType.CURIOUS] = 0.3
            if has_exclamation and caps_ratio > 0.3:
                for s in scores:
                    if s in (SentimentType.EXCITED, SentimentType.FRUSTRATED, SentimentType.URGENCY):
                        scores[s] = min(1.0, scores[s] * 1.3)
            if caps_ratio > 0.5 and len(text) > 10:
                scores[SentimentType.FRUSTRATED] = scores.get(SentimentType.FRUSTRATED, 0) + 0.3
            primary = max(scores, key=scores.get)
            intensity = scores[primary]
            confidence = min(1.0, len(matched_signals) * 0.3 + intensity * 0.5)
            needs = self._detect_needs(text_lower)
            escalation_risk = self._compute_escalation_risk(primary, intensity)
            requires_response_change = (
                intensity > 0.6 or
                primary in (SentimentType.FRUSTRATED, SentimentType.ANXIOUS, SentimentType.OVERWHELMED) or
                escalation_risk > 0.5
            )
            result = SentimentResult(
                sentiment=primary,
                intensity=round(intensity, 2),
                confidence=round(confidence, 2),
                needs=needs,
                signals=matched_signals,
                escalation_risk=round(escalation_risk, 2),
                requires_response_change=requires_response_change,
            )
            self._trajectory.add(primary, intensity)
            self._trajectory.compute_trend()
            self._recent_sentiments.append(result)
            return result

    def get_trajectory(self) -> SentimentTrajectory:
        with self._lock:
            return self._trajectory

    def get_recent_sentiments(self, limit: int = 10) -> list[dict]:
        with self._lock:
            return [
                {
                    "sentiment": s.sentiment.value,
                    "intensity": s.intensity,
                    "confidence": s.confidence,
                    "needs": [n.value for n in s.needs],
                    "escalation_risk": s.escalation_risk,
                    "requires_response_change": s.requires_response_change,
                }
                for s in list(self._recent_sentiments)[-limit:]
            ]

    def get_state(self) -> dict:
        with self._lock:
            recent = list(self._recent_sentiments)
            return {
                "trajectory_trend": self._trajectory.trend,
                "trajectory_confidence": round(self._trajectory.trend_confidence, 2),
                "escalation_count": self._escalation_count,
                "de_escalation_count": self._de_escalation_count,
                "is_escalating": self._trajectory.is_escalating(),
                "recent_sentiment": recent[-1].sentiment.value if recent else "neutral",
                "recent_intensity": recent[-1].intensity if recent else 0.0,
                "recent_needs": [n.value for n in recent[-1].needs] if recent else [],
                "total_analyzed": len(self._recent_sentiments),
            }

    def reset(self) -> None:
        with self._lock:
            self._trajectory = SentimentTrajectory()
            self._recent_sentiments.clear()
            self._escalation_count = 0
            self._de_escalation_count = 0

    def _compute_intensity_multiplier(self, words: list[str]) -> float:
        multiplier = 1.0
        for word in words[:10]:
            if word in _INTENSITY_MULTIPLIERS:
                multiplier *= _INTENSITY_MULTIPLIERS[word]
        return min(multiplier, 2.0)

    def _detect_needs(self, text_lower: str) -> list[NeedType]:
        needs = []
        for need_type, patterns, threshold in _NEED_PATTERNS:
            for pattern in patterns:
                if pattern in text_lower:
                    needs.append(need_type)
                    break
        if not needs:
            needs.append(NeedType.ACTION if any(w in text_lower for w in ["do", "make", "fix", "change"]) else NeedType.HELP)
        return needs[:3]

    def _compute_escalation_risk(self, primary: SentimentType, intensity: float) -> float:
        risk = 0.0
        high_risk = {SentimentType.FRUSTRATED, SentimentType.ANXIOUS, SentimentType.OVERWHELMED, SentimentType.URGENCY}
        if primary in high_risk:
            risk += 0.4
        risk += intensity * 0.3
        if self._trajectory.is_escalating():
            risk += 0.2
            self._escalation_count += 1
        else:
            self._de_escalation_count += 1
        return min(1.0, risk)


def get_sentiment_analyzer() -> SentimentAnalyzer:
    global _analyzer
    if _analyzer is None:
        with _analyzer_lock:
            if _analyzer is None:
                _analyzer = SentimentAnalyzer()
    return _analyzer