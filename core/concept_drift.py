"""Concept Drift Detector - Detect conversation topic shifts and adjust context.

Detects when the conversation topic drifts away from the current context
window's focus, enabling proactive context reconfiguration:

1. Monitors consecutive message topics using TF-IDF-like keyword matching
2. Detects gradual drift (slow topic change) vs sudden shift (hard break)
3. Emits signals for the context engine to adjust the context window
4. Tracks drift history for pattern learning
5. Integrates with Thought Sense for cognitive state annotation

Drift types:
- GRADUAL: Topic slowly morphs (3+ messages transitioning)
- SUDDEN: Hard topic break (e.g., "by the way..." or "actually...")
- RETURN: Returning to a previous topic after a digression
- OSCILLATION: Alternating between topics rapidly (confusion or multitasking)
"""
import logging
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto

logger = logging.getLogger(__name__)


class DriftType(Enum):
    NONE = auto()
    GRADUAL = auto()
    SUDDEN = auto()
    RETURN = auto()
    OSCILLATION = auto()


@dataclass
class DriftEvent:
    drift_type: DriftType
    from_topic: str
    to_topic: str
    similarity: float
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)


@dataclass
class TopicWindow:
    keywords: Counter = field(default_factory=Counter)
    message_count: int = 0
    start_time: float = field(default_factory=time.time)

    def similarity(self, other: "TopicWindow") -> float:
        if not self.keywords or not other.keywords:
            return 0.0
        intersection = sum(
            (self.keywords[k] + other.keywords[k]) / 2.0
            for k in set(self.keywords) & set(other.keywords)
        )
        union = sum(
            max(self.keywords[k], other.keywords[k])
            for k in set(self.keywords) | set(other.keywords)
        )
        return intersection / union if union > 0 else 0.0


class ConceptDriftDetector:
    """Detect and track conversation topic drift."""

    _WINDOW_SIZE = 3
    _GRADUAL_THRESHOLD = 0.4
    _SUDDEN_THRESHOLD = 0.2
    _RETURN_THRESHOLD = 0.5
    _OSCILLATION_WINDOW = 6
    _OSCILLATION_THRESHOLD = 0.3
    _MAX_DRIFT_HISTORY = 200
    _MAX_TOPIC_STACK = 20
    _MAX_KEYWORDS = 50

    def __init__(self):
        self._lock = threading.RLock()
        self._current_window = TopicWindow()
        self._previous_window: TopicWindow | None = None
        self._message_buffer: deque[str] = deque(maxlen=self._WINDOW_SIZE)
        self._drift_history: deque[DriftEvent] = deque(maxlen=self._MAX_DRIFT_HISTORY)
        self._topic_stack: deque[str] = deque(maxlen=self._MAX_TOPIC_STACK)
        self._last_drift_time: float = 0
        self._detected_count: int = 0
        self._drift_cooldown: float = 5.0
        self._current_topic_str: str = ""
        self._topic_message_counts: dict[str, int] = {}
        self._recent_topics: deque[str] = deque(maxlen=10)
        self._stats = {
            "total_messages": 0,
            "gradual_drifts": 0,
            "sudden_drifts": 0,
            "returns": 0,
            "oscillations": 0,
        }

    def _extract_keywords(self, text: str) -> Counter:
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "shall", "can", "need", "dare", "ought",
            "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "through", "during", "before", "after", "above", "below",
            "between", "out", "off", "over", "under", "again", "further", "then",
            "once", "here", "there", "when", "where", "why", "how", "all", "each",
            "every", "both", "few", "more", "most", "other", "some", "such", "no",
            "not", "only", "own", "same", "so", "than", "too", "very", "just",
            "because", "but", "and", "or", "if", "while", "about", "up", "it",
            "its", "i", "me", "my", "we", "our", "you", "your", "he", "she",
            "they", "them", "this", "that", "these", "those", "what", "which",
            "who", "whom", "also", "like", "know", "think", "want",
            "really", "going", "make", "get", "go", "come", "take", "see",
            "look", "try", "use", "find", "give", "tell", "work", "call",
            "ask", "seem", "feel", "leave", "good", "thing", "lot",
            "much", "way", "well", "back", "even", "still", "new", "now",
        }
        words = text.lower().split()
        keywords = Counter()
        for word in words:
            clean = word.strip(".,!?;:'\"()-[]{}").strip()
            if len(clean) > 2 and clean not in stop_words:
                keywords[clean] += 1

        if len(keywords) > self._MAX_KEYWORDS:
            keywords = Counter(dict(keywords.most_common(self._MAX_KEYWORDS)))
        return keywords

    def observe(self, content: str, source: str = "user") -> DriftEvent | None:
        """Observe a new message and detect drift if it occurs."""
        if content is None:
            content = ""
        keywords = self._extract_keywords(content)
        if not keywords:
            return None

        with self._lock:
            self._stats["total_messages"] += 1
            self._message_buffer.append(content)

            new_window = TopicWindow(keywords=keywords, message_count=1)
            new_window.keywords.update(self._current_window.keywords)

            similarity = 0.0
            if self._previous_window is not None:
                similarity = self._current_window.similarity(new_window)

            drift = self._detect_drift(content, similarity, new_window)
            if drift:
                self._drift_history.append(drift)
                self._detected_count += 1
                self._update_topic_stack(drift)
                self._stats = self._update_drift_stats(self._stats, drift)

            new_topic = self._infer_topic(new_window)
            if new_topic:
                self._current_topic_str = new_topic
                self._recent_topics.append(new_topic)
                self._topic_message_counts[new_topic] = self._topic_message_counts.get(new_topic, 0) + 1
                if len(self._topic_message_counts) > 200:
                    least = sorted(self._topic_message_counts, key=self._topic_message_counts.get)[:50]
                    for k in least:
                        del self._topic_message_counts[k]

            self._previous_window = self._current_window
            self._current_window = new_window

            return drift

    def _detect_drift(self, content: str, similarity: float, new_window: TopicWindow) -> DriftEvent | None:
        now = time.time()
        if now - self._last_drift_time < self._drift_cooldown:
            return None

        content_lower = content.lower()
        shift_indicators = [
            "by the way", "actually", "anyway", "moving on",
            "let's talk about", "switching gears", "different topic",
            "on another note", "changing subject", "unrelated",
        ]
        is_sudden = any(indicator in content_lower for indicator in shift_indicators)

        current_topic = self._infer_topic(self._current_window)
        new_topic = self._infer_topic(new_window)

        if is_sudden and similarity < self._SUDDEN_THRESHOLD:
            self._last_drift_time = now
            return DriftEvent(
                drift_type=DriftType.SUDDEN,
                from_topic=current_topic,
                to_topic=new_topic,
                similarity=similarity,
                confidence=0.9,
                metadata={"indicator": "shift_cue"},
            )

        if similarity < self._GRADUAL_THRESHOLD:
            osc = self._check_oscillation()
            if osc:
                self._last_drift_time = now
                return DriftEvent(
                    drift_type=DriftType.OSCILLATION,
                    from_topic=current_topic,
                    to_topic=new_topic,
                    similarity=similarity,
                    confidence=0.7,
                )

            if self._is_return(new_topic):
                self._last_drift_time = now
                return DriftEvent(
                    drift_type=DriftType.RETURN,
                    from_topic=current_topic,
                    to_topic=new_topic,
                    similarity=similarity,
                    confidence=0.6,
                )

            self._last_drift_time = now
            return DriftEvent(
                drift_type=DriftType.GRADUAL,
                from_topic=current_topic,
                to_topic=new_topic,
                similarity=similarity,
                confidence=0.5 + (self._GRADUAL_THRESHOLD - similarity) / self._GRADUAL_THRESHOLD * 0.4,
            )

        return None

    def _infer_topic(self, window: TopicWindow) -> str:
        if not window.keywords:
            return ""
        top = window.keywords.most_common(3)
        return "_".join(k for k, _ in top)

    def _check_oscillation(self) -> bool:
        if len(self._drift_history) < self._OSCILLATION_WINDOW:
            return False
        recent = list(self._drift_history)[-self._OSCILLATION_WINDOW:]
        topics = set()
        for drift in recent:
            topics.add(drift.from_topic)
            topics.add(drift.to_topic)
        return len(topics) <= 3 and len(recent) >= 3

    def _is_return(self, new_topic: str) -> bool:
        if not new_topic or not self._topic_stack:
            return False
        recent = list(self._topic_stack)[-5:]
        return any(t == new_topic for t in recent[:-1] if len(recent) > 1)

    def _update_topic_stack(self, drift: DriftEvent):
        if drift.to_topic and drift.to_topic not in list(self._topic_stack)[-3:]:
            self._topic_stack.append(drift.to_topic)

    def _update_drift_stats(self, stats: dict, drift: DriftEvent) -> dict:
        type_key = {
            DriftType.GRADUAL: "gradual_drifts",
            DriftType.SUDDEN: "sudden_drifts",
            DriftType.RETURN: "returns",
            DriftType.OSCILLATION: "oscillations",
        }.get(drift.drift_type)
        if type_key:
            stats[type_key] += 1
        return stats

    def get_current_topic(self) -> str:
        with self._lock:
            return self._current_topic_str or self._infer_topic(self._current_window)

    def get_topic_stack(self) -> list[str]:
        with self._lock:
            return list(self._topic_stack)

    def get_drift_history(self, limit: int = 20) -> list[dict]:
        with self._lock:
            return [
                {
                    "type": d.drift_type.name,
                    "from": d.from_topic,
                    "to": d.to_topic,
                    "similarity": round(d.similarity, 3),
                    "confidence": round(d.confidence, 3),
                    "timestamp": d.timestamp.isoformat(),
                }
                for d in list(self._drift_history)[-limit:]
            ]

    def get_stats(self) -> dict:
        with self._lock:
            return {
                **self._stats,
                "current_topic": self._current_topic_str or self._infer_topic(self._current_window),
                "topic_stack_size": len(self._topic_stack),
                "drift_events_detected": self._detected_count,
                "recent_drifts": len(self._drift_history),
                "topic_message_counts": dict(self._topic_message_counts),
                "unique_topics_seen": len(self._topic_message_counts),
            }

    def reset(self):
        with self._lock:
            self._current_window = TopicWindow()
            self._previous_window = None
            self._message_buffer.clear()
            self._drift_history.clear()
            self._topic_stack.clear()
            self._recent_topics.clear()
            self._current_topic_str = ""
            self._topic_message_counts.clear()
            self._detected_count = 0
            self._last_drift_time = 0
            self._stats = {
                "total_messages": 0,
                "gradual_drifts": 0,
                "sudden_drifts": 0,
                "returns": 0,
                "oscillations": 0,
            }


_drift_detector: ConceptDriftDetector | None = None
_dd_lock = threading.Lock()


def get_drift_detector() -> ConceptDriftDetector:
    global _drift_detector
    with _dd_lock:
        if _drift_detector is None:
            _drift_detector = ConceptDriftDetector()
        return _drift_detector