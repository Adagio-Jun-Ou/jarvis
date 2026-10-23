"""Attention Priority Queue - Urgency-based message and task processing.

Manages incoming messages/tasks with priority scoring based on:
1. Urgency signals (user corrections, time-sensitive requests)
2. Emotional valence (frustration = higher priority)
3. Cognitive state alignment (FLOW queries get fast-track)
4. Context relevance (continuation of current topic = higher priority)
5. Decay — old low-priority items slowly lose priority

This ensures Jarvis responds to what matters most, when it matters most.
"""
import heapq
import logging
import threading
import time
from typing import Optional
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto

logger = logging.getLogger(__name__)


class UrgencySignal(Enum):
    NORMAL = auto()
    TIME_SENSITIVE = auto()
    CORRECTION = auto()
    SAFETY_CRITICAL = auto()
    FOLLOW_UP = auto()
    SYSTEM_ALERT = auto()


_URGENCY_MULTIPLIERS = {
    UrgencySignal.NORMAL: 1.0,
    UrgencySignal.TIME_SENSITIVE: 2.0,
    UrgencySignal.CORRECTION: 2.5,
    UrgencySignal.SAFETY_CRITICAL: 5.0,
    UrgencySignal.FOLLOW_UP: 1.5,
    UrgencySignal.SYSTEM_ALERT: 3.0,
}

_COGNITIVE_STATE_PRIORITY = {
    "FRUSTRATED": 2.0,
    "OVERLOADED": 1.5,
    "FLOW": 0.8,
    "ENGAGED": 1.0,
    "EXPLORING": 0.9,
    "DEEP_DIVE": 0.9,
    "IDLE": 0.5,
}


@dataclass(order=False)
class PriorityItem:
    item_id: str
    content: str
    priority: float
    urgency: UrgencySignal = UrgencySignal.NORMAL
    source: str = "user"
    topic: str = ""
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)

    def __lt__(self, other):
        if self.priority != other.priority:
            return self.priority > other.priority
        return self.timestamp < other.timestamp


class AttentionPriorityQueue:
    """Priority queue for urgency-based message processing."""

    _DECAY_INTERVAL = 60
    _MAX_QUEUE_SIZE = 500
    _MAX_HISTORY = 200
    _DECAY_RATE = 0.95
    _MIN_PRIORITY = 0.01

    def __init__(self):
        self._lock = threading.RLock()
        self._queue: list[PriorityItem] = []
        self._item_ids: set[str] = set()
        self._counter = 0
        self._history: deque[dict] = deque(maxlen=self._MAX_HISTORY)
        self._current_topic: str = ""
        self._topic_continuity_bonus = 0.3
        self._processed_count: int = 0
        self._total_priority_processed: float = 0.0
        self._by_urgency: dict[str, int] = defaultdict(int)

    def enqueue(
        self,
        content: str,
        urgency: UrgencySignal = UrgencySignal.NORMAL,
        source: str = "user",
        topic: str = "",
        metadata: dict | None = None,
        cognitive_state: str = "",
    ) -> str | None:
        """Add an item to the priority queue."""
        with self._lock:
            self._counter += 1
            item_id = f"pq_{self._counter:06d}"

            if item_id in self._item_ids:
                return None

            if len(self._queue) >= self._MAX_QUEUE_SIZE:
                lowest_idx = min(range(len(self._queue)), key=lambda i: self._queue[i].priority)
                if self._queue[lowest_idx].priority < _URGENCY_MULTIPLIERS[urgency]:
                    removed = self._queue.pop(lowest_idx)
                    self._item_ids.discard(removed.item_id)
                else:
                    return None

            base_priority = _URGENCY_MULTIPLIERS.get(urgency, 1.0)

            cognitive_mod = _COGNITIVE_STATE_PRIORITY.get(cognitive_state, 1.0)
            priority = base_priority * cognitive_mod

            if topic and self._current_topic and topic == self._current_topic:
                priority += self._topic_continuity_bonus

            item = PriorityItem(
                item_id=item_id,
                content=content,
                priority=priority,
                urgency=urgency,
                source=source,
                topic=topic,
                metadata=metadata or {},
            )

            heapq.heappush(self._queue, item)
            self._item_ids.add(item_id)

            return item_id

    def dequeue(self) -> PriorityItem | None:
        """Get the highest priority item."""
        with self._lock:
            if not self._queue:
                return None

            item = heapq.heappop(self._queue)
            self._item_ids.discard(item.item_id)
            self._processed_count += 1
            self._total_priority_processed += item.priority
            self._by_urgency[item.urgency.name] += 1

            if item.topic:
                self._current_topic = item.topic

            self._history.append({
                "item_id": item.item_id,
                "urgency": item.urgency.name,
                "priority": round(item.priority, 3),
                "source": item.source,
                "topic": item.topic,
                "timestamp": item.timestamp,
            })

            return item

    def peek(self) -> PriorityItem | None:
        """Peek at the highest priority item without removing it."""
        with self._lock:
            if not self._queue:
                return None
            return self._queue[0]

    def size(self) -> int:
        with self._lock:
            return len(self._queue)

    def is_empty(self) -> bool:
        with self._lock:
            return len(self._queue) == 0

    def apply_decay(self) -> int:
        """Apply priority decay to all items in the queue. Returns count of items removed."""
        with self._lock:
            decayed_items = []
            removed = 0
            for item in self._queue:
                item.priority *= self._DECAY_RATE
                if item.priority < self._MIN_PRIORITY:
                    self._item_ids.discard(item.item_id)
                    removed += 1
                else:
                    decayed_items.append(item)

            self._queue = decayed_items
            heapq.heapify(self._queue)
            return removed

    def enqueue_with_interoception(
        self,
        content: str,
        urgency: UrgencySignal = UrgencySignal.NORMAL,
        source: str = "user",
        topic: str = "",
        metadata: dict | None = None,
        cognitive_state: str = "",
        interoceptive_salience: float = 0.0,
        body_priority_override: bool = False,
    ) -> str | None:
        """Add item to queue with interoceptive salience modulation.

        High interoceptive salience (body prediction error > 0.4) narrows
        attention: body-related items get boosted, non-body items get suppressed
        unless safety-critical. Implements the 'body-first' attention mechanism.
        """
        with self._lock:
            if interoceptive_salience > 0.6 and urgency != UrgencySignal.SAFETY_CRITICAL:
                urgency = UrgencySignal.SYSTEM_ALERT
            elif interoceptive_salience > 0.4:
                if body_priority_override:
                    urgency = UrgencySignal.SYSTEM_ALERT
                elif topic and not ("body" in topic.lower() or "stress" in topic.lower() or "feeling" in topic.lower()):
                    urgency = UrgencySignal.NORMAL

            return self.enqueue(
                content=content,
                urgency=urgency,
                source=source,
                topic=topic,
                metadata={**(metadata or {}), "interoceptive_salience": interoceptive_salience},
                cognitive_state=cognitive_state,
            )

    def reprioritize_based_on_interoception(self, interoceptive_salience: float,
                                           body_topic_keywords: Optional[list[str]] = None) -> int:
        """Re-prioritize existing queue items based on interoceptive signals.

        Returns count of items whose priority was modified.
        """
        with self._lock:
            modified = 0
            keywords = body_topic_keywords or ["body", "stress", "feeling", "health", "rest"]
            for item in self._queue:
                if interoceptive_salience > 0.4:
                    if any(kw in item.topic.lower() or kw in item.content.lower() for kw in keywords):
                        item.priority *= 1.2
                        modified += 1
                    elif item.urgency == UrgencySignal.NORMAL and interoceptive_salience > 0.6:
                        item.priority *= 0.8
                        modified += 1
            if modified > 0:
                heapq.heapify(self._queue)
            return modified

    def reprioritize(self, item_id: str, new_urgency: UrgencySignal) -> bool:
        """Change the urgency (and thus priority) of an existing item."""
        with self._lock:
            for item in self._queue:
                if item.item_id == item_id:
                    base = _URGENCY_MULTIPLIERS.get(new_urgency, 1.0)
                    item.priority = base
                    item.urgency = new_urgency
                    heapq.heapify(self._queue)
                    return True
            return False

    def set_current_topic(self, topic: str):
        """Update the current conversation topic for continuity scoring."""
        with self._lock:
            self._current_topic = topic

    def bump_priority(self, item_id: str, boost: float = 1.0) -> bool:
        """Increase priority of an item by a boost amount."""
        with self._lock:
            for item in self._queue:
                if item.item_id == item_id:
                    item.priority += boost
                    heapq.heapify(self._queue)
                    return True
            return False

    def get_queue_snapshot(self, limit: int = 20) -> list[dict]:
        """Get a snapshot of current queue items sorted by priority."""
        with self._lock:
            sorted_items = sorted(self._queue, key=lambda x: (-x.priority, x.timestamp))
            return [
                {
                    "item_id": item.item_id,
                    "content": item.content[:80],
                    "priority": round(item.priority, 3),
                    "urgency": item.urgency.name,
                    "source": item.source,
                    "topic": item.topic,
                    "age_seconds": round(time.time() - item.timestamp, 1),
                }
                for item in sorted_items[:limit]
            ]

    def get_stats(self) -> dict:
        with self._lock:
            avg_priority = (
                self._total_priority_processed / self._processed_count
                if self._processed_count > 0
                else 0
            )
            return {
                "queue_size": len(self._queue),
                "total_processed": self._processed_count,
                "avg_priority_processed": round(avg_priority, 3),
                "current_topic": self._current_topic,
                "by_urgency": dict(self._by_urgency),
                "history_size": len(self._history),
            }


_attention_queue: AttentionPriorityQueue | None = None
_aq_lock = threading.Lock()


def get_attention_queue() -> AttentionPriorityQueue:
    global _attention_queue
    if _attention_queue is None:
        with _aq_lock:
            if _attention_queue is None:
                _attention_queue = AttentionPriorityQueue()
    return _attention_queue


def reset_attention_queue() -> None:
    global _attention_queue
    with _aq_lock:
        _attention_queue = None