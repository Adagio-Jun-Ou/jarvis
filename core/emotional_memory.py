"""Emotional Memory Bridge for Jarvis.

Connects the consciousness architecture (interoception, feelings, regulation)
with the memory system. Every significant interaction gets tagged with the
feeling state at that moment, creating affective memory traces.

Key insight from neuroscience: memories with emotional weight are recalled
more easily (amygdala-hippocampus interaction). This module implements that:

- **Affective Tagging**: Every memory gets valence/arousal/quality tags
- **Emotional Recall Boost**: Memories matching current feelings are boosted
- **Mood-Congruent Retrieval**: Sad mood → sad memories, happy → happy
- **Regulation Memory**: What regulation worked when, enabling learning
- **Feeling Trajectory**: Track how feelings change across conversations
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

_emotional_memory: Optional["EmotionalMemory"] = None
_em_lock = threading.Lock()


class EmotionalTagStrength(Enum):
    WEAK = 0.3
    MODERATE = 0.6
    STRONG = 0.9


@dataclass
class AffectiveTag:
    valence: float = 0.0
    arousal: float = 0.5
    dominance: float = 0.5
    quality: str = "neutral"
    awareness_score: float = 0.5
    regulation_strategy: str = ""
    timestamp: float = field(default_factory=time.time)
    context_topic: str = ""
    prediction_error_avg: float = 0.0

    @property
    def emotional_weight(self) -> float:
        intensity = abs(self.valence) * 0.4 + abs(self.arousal - 0.5) * 0.4 + abs(self.dominance - 0.5) * 0.2
        return min(1.0, intensity)


@dataclass
class FeelingMilestone:
    timestamp: float = field(default_factory=time.time)
    valence: float = 0.0
    arousal: float = 0.5
    quality: str = "neutral"
    trigger: str = ""
    regulation_applied: str = ""
    pre_reg_feeling: str = ""
    post_reg_feeling: str = ""


class EmotionalMemory:
    """Bridges consciousness feelings with memory system.

    Tracks:
    - Affective tags on conversation turns
    - Feeling trajectory across a session
    - What regulation strategies worked in what contexts
    - Mood-congruent recall boosting
    """

    _MAX_TAGS = 500
    _MAX_MILESTONES = 100
    _MAX_REGULATION_LEARNING = 200
    _CONGRUENCE_BOOST = 1.3
    _INCONGRUENCE_SUPPRESS = 0.85

    def __init__(self):
        self._lock = threading.RLock()
        self._affective_tags: deque[AffectiveTag] = deque(maxlen=self._MAX_TAGS)
        self._feeling_trajectory: deque[FeelingMilestone] = deque(maxlen=self._MAX_MILESTONES)
        self._regulation_learning: deque[dict] = deque(maxlen=self._MAX_REGULATION_LEARNING)
        self._current_tag: Optional[AffectiveTag] = None
        self._session_start = time.time()

    def tag_interaction(self, valence: float = 0.0, arousal: float = 0.5,
                       dominance: float = 0.5, quality: str = "neutral",
                       awareness: float = 0.5, topic: str = "",
                       prediction_error: float = 0.0,
                       regulation_strategy: str = "") -> AffectiveTag:
        with self._lock:
            tag = AffectiveTag(
                valence=valence,
                arousal=arousal,
                dominance=dominance,
                quality=quality,
                awareness_score=awareness,
                context_topic=topic,
                prediction_error_avg=prediction_error,
                regulation_strategy=regulation_strategy,
            )
            self._affective_tags.append(tag)
            self._current_tag = tag
            return tag

    def record_feeling_shift(self, pre_feeling: str, post_feeling: str,
                              regulation: str = "", trigger: str = "",
                              valence: float = 0.0, arousal: float = 0.5) -> None:
        with self._lock:
            milestone = FeelingMilestone(
                valence=valence,
                arousal=arousal,
                quality=post_feeling,
                trigger=trigger,
                regulation_applied=regulation,
                pre_reg_feeling=pre_feeling,
                post_reg_feeling=post_feeling,
            )
            self._feeling_trajectory.append(milestone)
            if regulation and pre_feeling != post_feeling:
                self._regulation_learning.append({
                    "strategy": regulation,
                    "from_feeling": pre_feeling,
                    "to_feeling": post_feeling,
                    "trigger": trigger,
                    "valence_shift": round(valence, 3),
                    "arousal_shift": round(arousal, 3),
                    "timestamp": time.time(),
                })

    def get_recall_boost(self, current_valence: float,
                         current_arousal: float,
                         memory_tags: list[AffectiveTag]) -> list[float]:
        with self._lock:
            boosts = []
            for tag in memory_tags:
                valence_match = 1.0 - min(1.0, abs(current_valence - tag.valence))
                arousal_match = 1.0 - min(1.0, abs(current_arousal - tag.arousal) * 2.0)
                congruence = valence_match * 0.6 + arousal_match * 0.4
                if congruence > 0.6:
                    boost = self._CONGRUENCE_BOOST + tag.emotional_weight * 0.3
                elif congruence < 0.3:
                    boost = self._INCONGRUENCE_SUPPRESS
                else:
                    boost = 1.0
                boosts.append(round(boost, 3))
            return boosts

    def get_regulation_history(self, feeling_quality: str = "",
                                limit: int = 10) -> list[dict]:
        with self._lock:
            entries = list(self._regulation_learning)
            if feeling_quality:
                entries = [e for e in entries
                           if feeling_quality in e.get("from_feeling", "")
                           or feeling_quality in e.get("to_feeling", "")]
            return entries[-limit:]

    def get_best_regulation_for(self, feeling: str) -> Optional[str]:
        with self._lock:
            relevant = [e for e in self._regulation_learning
                        if feeling in e.get("from_feeling", "")]
            if not relevant:
                return None
            strategy_counts: dict[str, int] = {}
            for e in relevant:
                s = e.get("strategy", "")
                if s:
                    strategy_counts[s] = strategy_counts.get(s, 0) + 1
            if not strategy_counts:
                return None
            return max(strategy_counts, key=strategy_counts.get)

    def get_feeling_trajectory(self, limit: int = 20) -> list[dict]:
        with self._lock:
            return [
                {
                    "timestamp": round(m.timestamp, 1),
                    "valence": round(m.valence, 3),
                    "arousal": round(m.arousal, 3),
                    "quality": m.quality,
                    "regulation": m.regulation_applied,
                    "pre": m.pre_reg_feeling,
                    "post": m.post_reg_feeling,
                }
                for m in list(self._feeling_trajectory)[-limit:]
            ]

    def get_recent_tags(self, limit: int = 10) -> list[dict]:
        with self._lock:
            return [
                {
                    "valence": round(t.valence, 3),
                    "arousal": round(t.arousal, 3),
                    "quality": t.quality,
                    "topic": t.context_topic,
                    "weight": round(t.emotional_weight, 3),
                    "regulation": t.regulation_strategy,
                }
                for t in list(self._affective_tags)[-limit:]
            ]

    def get_dominant_feeling(self, window: int = 10) -> dict:
        with self._lock:
            recent = list(self._affective_tags)[-window:]
            if not recent:
                return {"quality": "neutral", "avg_valence": 0.0, "avg_arousal": 0.5}
            avg_val = sum(t.valence for t in recent) / len(recent)
            avg_ar = sum(t.arousal for t in recent) / len(recent)
            quality_counts: dict[str, int] = {}
            for t in recent:
                quality_counts[t.quality] = quality_counts.get(t.quality, 0) + 1
            dominant = max(quality_counts, key=quality_counts.get) if quality_counts else "neutral"
            return {
                "quality": dominant,
                "avg_valence": round(avg_val, 3),
                "avg_arousal": round(avg_ar, 3),
                "sample_size": len(recent),
                "valence_trend": round(self._compute_valence_trend(recent), 3),
            }

    def _compute_valence_trend(self, tags: list[AffectiveTag]) -> float:
        if len(tags) < 2:
            return 0.0
        first_half = tags[:len(tags) // 2]
        second_half = tags[len(tags) // 2:]
        avg_first = sum(t.valence for t in first_half) / max(1, len(first_half))
        avg_second = sum(t.valence for t in second_half) / max(1, len(second_half))
        return avg_second - avg_first

    def get_session_duration_minutes(self) -> float:
        return (time.time() - self._session_start) / 60.0

    def prune_old_tags(self, max_age_hours: float = 48.0) -> int:
        with self._lock:
            cutoff = time.time() - max_age_hours * 3600
            before = len(self._affective_tags)
            self._affective_tags = deque(
                (t for t in self._affective_tags if t.timestamp >= cutoff),
                maxlen=self._MAX_TAGS,
            )
            cutoff_ms = time.time() - max_age_hours * 3600
            self._feeling_trajectory = deque(
                (m for m in self._feeling_trajectory if m.timestamp >= cutoff_ms),
                maxlen=self._MAX_MILESTONES,
            )
            pruned = before - len(self._affective_tags)
            if pruned > 0:
                logger.debug(f"EmotionalMemory: pruned {pruned} old tags")
            return pruned

    def get_state(self) -> dict:
        with self._lock:
            dominant = self.get_dominant_feeling()
            return {
                "total_tags": len(self._affective_tags),
                "total_milestones": len(self._feeling_trajectory),
                "regulation_lessons": len(self._regulation_learning),
                "dominant_feeling": dominant,
                "recent_trajectory": self.get_feeling_trajectory(limit=5),
                "recent_tags": self.get_recent_tags(limit=5),
                "session_duration_min": round(self.get_session_duration_minutes(), 1),
            }


def get_emotional_memory() -> EmotionalMemory:
    global _emotional_memory
    if _emotional_memory is None:
        with _em_lock:
            if _emotional_memory is None:
                _emotional_memory = EmotionalMemory()
    return _emotional_memory


def reset_emotional_memory() -> None:
    global _emotional_memory
    with _em_lock:
        _emotional_memory = None