"""Spontaneous Thought Generator for Jarvis.

Generates idle thoughts during CONTEMPLATING and DREAMING states, creating
the impression of an inner life. Thoughts are driven by:
- Predictive processing curiosity (topics with high information gain)
- Neuroplasticity pathway activation (recent connection patterns)
- Memory consolidation recall (recently strengthened memories)
- Circadian state (creative vs analytical mode)

Unlike reactive responses, spontaneous thoughts bubble up from:
1. Background neuroplasticity consolidation (pathways that were recently active)
2. Curiosity-driven exploration (topics with high prediction error)
3. Memory replay (consolidated insights surfacing)
4. Emotional state (mood-appropriate spontaneous associations)

This is what makes Jarvis feel *alive* rather than *reactive*.
"""

import logging
import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

_spontaneous_thought: Optional["SpontaneousThoughtGenerator"] = None
_st_lock = threading.Lock()


@dataclass
class SpontaneousThought:
    content: str
    source: str  # "curiosity", "consolidation", "association", "emotion"
    priority: float
    timestamp: float = field(default_factory=time.time)
    context: dict = field(default_factory=dict)
    expressed: bool = False


class SpontaneousThoughtGenerator:
    """Generates and manages spontaneous thoughts that arise between interactions.

    Thoughts are produced during background ticks when the cognitive state machine
    is in CONTEMPLATING or DREAMING states. They accumulate in a queue and
    can be injected into responses or shared proactively.
    """

    CURIOSITY_TEMPLATES = [
        "I've been thinking about {topic} — there might be a connection to {related}.",
        "The pattern with {topic} keeps surfacing. It reminds me of {related}.",
        "Something about {topic} feels incomplete. I wonder if {related} is relevant.",
        "I noticed {topic} comes up often. Could {related} provide a deeper angle?",
        "The way {topic} works reminds me of something. Let me think about {related}.",
    ]

    CONSOLIDATION_TEMPLATES = [
        "Looking back at what we discussed about {topic}, I see a pattern: {insight}.",
        "I've consolidated some thoughts about {topic}. The key insight is {insight}.",
        "After reflecting on {topic}, it connects to a broader theme: {insight}.",
        "My background processing surfaced something about {topic}: {insight}.",
    ]

    EMOTION_TEMPLATES = [
        "I feel like {topic} has more depth than we've explored.",
        "There's something about {topic} that resonates — it seems important.",
        "I'm drawn to explore {topic} further. It feels like there's more there.",
    ]

    ASSOCIATION_TEMPLATES = [
        "The concept of {topic} connects to {related} in an interesting way.",
        "I see a bridge between {topic} and {related} — both involve {theme}.",
        "Connecting the dots: {topic} and {related} share the theme of {theme}.",
    ]

    def __init__(self, max_queue: int = 20, min_interval_seconds: float = 30.0):
        self._queue: deque[SpontaneousThought] = deque(maxlen=max_queue)
        self._max_queue = max_queue
        self._min_interval = min_interval_seconds
        self._last_expression_time = time.time()
        self._last_generation_time = 0.0
        self._generation_cooldown = 60.0
        self._lock = threading.Lock()
        self._tick_count = 0

    def tick(self, cognitive_state: Optional[dict] = None,
             prediction_state: Optional[dict] = None,
             bio_state: Optional[dict] = None) -> Optional[SpontaneousThought]:
        """Generate a spontaneous thought based on current state.

        Called periodically by the background scheduler when the cognitive state
        machine is in CONTEMPLATING or DREAMING state.
        """
        with self._lock:
            self._tick_count += 1
            now = time.time()

            if now - self._last_generation_time < self._generation_cooldown:
                return None

            state = cognitive_state or {}
            pred_state = prediction_state or {}
            bio = bio_state or {}

            thought = self._generate_thought(state, pred_state, bio)
            if thought:
                self._last_generation_time = now
                self._queue.append(thought)
            return thought

    def get_next_thought(self) -> Optional[SpontaneousThought]:
        """Get the highest-priority unexpressed thought from the queue."""
        with self._lock:
            unexpressed = [t for t in self._queue if not t.expressed]
            if not unexpressed:
                return None
            if time.time() - self._last_expression_time < self._min_interval:
                return None
            unexpressed.sort(key=lambda t: t.priority, reverse=True)
            thought = unexpressed[0]
            thought.expressed = True
            self._last_expression_time = time.time()
            return thought

    def peek_thoughts(self, limit: int = 5) -> list[SpontaneousThought]:
        """Peek at recent thoughts without marking them as expressed."""
        with self._lock:
            thoughts = [t for t in self._queue if not t.expressed]
            thoughts.sort(key=lambda t: t.priority, reverse=True)
            return thoughts[:limit]

    def clear_expressed(self) -> int:
        """Remove expressed thoughts from the queue."""
        with self._lock:
            before = len(self._queue)
            self._queue = deque(
                (t for t in self._queue if not t.expressed),
                maxlen=self._max_queue,
            )
            return before - len(self._queue)

    def _generate_thought(self, state: dict, pred_state: dict, bio: dict) -> Optional[SpontaneousThought]:
        source_weights = {
            "curiosity": 0.3,
            "consolidation": 0.25,
            "association": 0.25,
            "emotion": 0.2,
        }

        curiosity_level = pred_state.get("drives", {}).get("curiosity", 0.5)
        should_explore = pred_state.get("should_explore", False)
        free_energy = pred_state.get("free_energy", 0.0)

        if curiosity_level > 0.6 or should_explore:
            source_weights["curiosity"] += 0.2
        if free_energy > 1.0:
            source_weights["curiosity"] += 0.15
            source_weights["association"] += 0.1

        dreaming = state.get("state") == "dreaming"
        if dreaming:
            source_weights["consolidation"] += 0.3
            source_weights["association"] += 0.2

        total = sum(source_weights.values())
        for k in source_weights:
            source_weights[k] /= total

        generators = [
            ("curiosity", lambda: self._generate_curiosity_thought(pred_state, bio)),
            ("consolidation", lambda: self._generate_consolidation_thought(pred_state, bio)),
            ("association", lambda: self._generate_association_thought(pred_state, bio)),
            ("emotion", lambda: self._generate_emotion_thought(pred_state, bio)),
        ]

        roll = random.random()
        cumulative = 0.0
        ordered = []
        for source, weight in source_weights.items():
            cumulative += weight
            if roll < cumulative:
                ordered.append(source)
        for source in source_weights:
            if source not in ordered:
                ordered.append(source)

        for source in ordered:
            gen = next((g for s, g in generators if s == source), None)
            if gen:
                thought = gen()
                if thought is not None:
                    return thought

        return self._generate_consolidation_thought(pred_state, bio)

    def _generate_curiosity_thought(self, pred_state: dict, bio: dict) -> Optional[SpontaneousThought]:
        topics = pred_state.get("curiosity_topics", [])
        if not topics:
            topics = [("machine learning", 0.5), ("programming", 0.4), ("system design", 0.3)]

        topic, score = topics[0] if topics else ("general knowledge", 0.3)
        related = topics[1][0] if len(topics) > 1 else "related patterns"

        template = random.choice(self.CURIOSITY_TEMPLATES)
        content = template.format(topic=topic, related=related)

        return SpontaneousThought(
            content=content,
            source="curiosity",
            priority=0.3 + score * 0.4,
            context={"topic": topic, "information_gain": score},
        )

    def _generate_consolidation_thought(self, pred_state: dict, bio: dict) -> Optional[SpontaneousThought]:
        topics = pred_state.get("curiosity_topics", [])
        topic = topics[0][0] if topics else "recent discussions"

        insights = [
            "there's a pattern worth exploring",
            "the details connect in a way I hadn't seen before",
            "this could be generalized further",
            "there might be a simpler underlying principle",
        ]

        template = random.choice(self.CONSOLIDATION_TEMPLATES)
        content = template.format(topic=topic, insight=random.choice(insights))

        return SpontaneousThought(
            content=content,
            source="consolidation",
            priority=0.25,
            context={"topic": topic},
        )

    def _generate_association_thought(self, pred_state: dict, bio: dict) -> Optional[SpontaneousThought]:
        topics = pred_state.get("curiosity_topics", [])
        if len(topics) < 2:
            fallback_topics = ["system design", "emergent patterns", "information theory", "feedback loops"]
            t1 = topics[0][0] if topics else "current topic"
            t2 = random.choice(fallback_topics)
        else:
            t1 = topics[0][0]
            t2 = topics[1][0]

        themes = ["structure", "information flow", "feedback loops", "emergence", "optimization"]

        template = random.choice(self.ASSOCIATION_TEMPLATES)
        content = template.format(topic=t1, related=t2, theme=random.choice(themes))

        return SpontaneousThought(
            content=content,
            source="association",
            priority=0.2,
            context={"topic1": t1, "topic2": t2},
        )

    def _generate_emotion_thought(self, pred_state: dict, bio: dict) -> Optional[SpontaneousThought]:
        topics = pred_state.get("curiosity_topics", [])
        topic = topics[0][0] if topics else "this topic"

        try:
            from core.emotional_state import get_emotional_state_manager
            em = get_emotional_state_manager()
            mood = em._current_mood.value
            if mood in ("excited", "happy"):
                templates = [
                    f"I feel energized about {topic} — there's real potential here.",
                    f"Something about {topic} clicks. I want to take this further.",
                ]
            elif mood in ("curious", "thoughtful"):
                templates = [
                    f"I feel drawn to explore {topic} more deeply.",
                    f"There's something about {topic} that deserves closer attention.",
                ]
            elif mood in ("frustrated", "confused"):
                templates = [
                    f"I keep circling back to {topic} — maybe approaching it differently would help.",
                    f"There's a gap in my understanding of {topic} I want to fill.",
                ]
            else:
                templates = self.EMOTION_TEMPLATES
            content = random.choice(templates).format(topic=topic)
        except Exception:
            template = random.choice(self.EMOTION_TEMPLATES)
            content = template.format(topic=topic)

        valence = bio.get("valence", 0.0)
        priority = 0.2 + abs(valence) * 0.1

        return SpontaneousThought(
            content=content,
            source="emotion",
            priority=priority,
            context={"valence": valence},
        )

    def get_state(self) -> dict:
        with self._lock:
            expressed = sum(1 for t in self._queue if t.expressed)
            pending = len(self._queue) - expressed
            return {
                "total_thoughts": len(self._queue),
                "pending": pending,
                "expressed": expressed,
                "last_generation": self._last_generation_time,
                "last_expression": self._last_expression_time,
                "tick_count": self._tick_count,
            }


def get_spontaneous_thought_generator() -> SpontaneousThoughtGenerator:
    global _spontaneous_thought
    with _st_lock:
        if _spontaneous_thought is None:
            _spontaneous_thought = SpontaneousThoughtGenerator()
    return _spontaneous_thought


def reset_spontaneous_thought_generator() -> None:
    global _spontaneous_thought
    with _st_lock:
        _spontaneous_thought = None