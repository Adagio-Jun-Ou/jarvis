"""Adaptive Context Window - Dynamically adjusts context window size and content.

Adjusts context strategy based on:
1. Token budget vs content demand
2. Cognitive state (FLOW gets more depth, OVERLOADED gets summaries)
3. Topic focus from Concept Drift Detector
4. Priority signals from Attention Priority Queue
5. Memory consolidation state (idle = more background context)
"""
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

logger = logging.getLogger(__name__)


class ContextStrategy(Enum):
    FULL = auto()
    FOCUSED = auto()
    SUMMARIZED = auto()
    MINIMAL = auto()


class ContentType(Enum):
    SYSTEM = auto()
    CONVERSATION = auto()
    BACKGROUND = auto()
    REFERENCE = auto()
    EMOTIONAL = auto()


@dataclass
class ContextBlock:
    content_type: ContentType
    content: str
    token_estimate: int
    priority: float
    topic: str = ""
    source: str = ""
    timestamp: float = field(default_factory=time.time)


class AdaptiveContextWindow:
    _MAX_TOKENS = 128000
    _MIN_TOKENS = 4000
    _RESERVE_TOKENS = 8000
    _MAX_BLOCKS = 200
    _MAX_HISTORY = 100
    _STRATEGY_MAP = {
        "FLOW": (ContextStrategy.FULL, 0.9),
        "ENGAGED": (ContextStrategy.FOCUSED, 0.75),
        "EXPLORING": (ContextStrategy.FOCUSED, 0.7),
        "DEEP_DIVE": (ContextStrategy.FOCUSED, 0.85),
        "IDLE": (ContextStrategy.SUMMARIZED, 0.5),
        "OVERLOADED": (ContextStrategy.MINIMAL, 0.4),
        "FRUSTRATED": (ContextStrategy.MINIMAL, 0.3),
    }

    def __init__(self, max_tokens: int = 96000):
        self._lock = threading.RLock()
        self._max_tokens = min(max_tokens, self._MAX_TOKENS)
        self._blocks: deque[ContextBlock] = deque(maxlen=self._MAX_BLOCKS)
        self._current_strategy = ContextStrategy.FOCUSED
        self._current_budget = self._max_tokens
        self._cognitive_state = "ENGAGED"
        self._focus_topic = ""
        self._adjustment_history: deque[dict] = deque(maxlen=self._MAX_HISTORY)
        self._attention_queue_ref: Any = None
        self._drift_detector_ref: Any = None
        self._load_balancer_ref: Any = None
        self._stats = {
            "total_adjustments": 0,
            "strategy_changes": 0,
            "blocks_evicted": 0,
            "tokens_saved": 0,
        }

    def _get_attention_queue(self):
        if self._attention_queue_ref is None:
            try:
                from core.attention_priority import get_attention_queue
                self._attention_queue_ref = get_attention_queue()
            except Exception as e:
                logger.debug(f"Failed to get attention queue: {e}")
        return self._attention_queue_ref

    def _get_drift_detector(self):
        if self._drift_detector_ref is None:
            try:
                from core.concept_drift import get_drift_detector
                self._drift_detector_ref = get_drift_detector()
            except Exception as e:
                logger.debug(f"Failed to get drift detector: {e}")
        return self._drift_detector_ref

    def _get_load_balancer(self):
        if self._load_balancer_ref is None:
            try:
                from core.cognitive_load_balancer import get_load_balancer
                self._load_balancer_ref = get_load_balancer()
            except Exception as e:
                logger.debug(f"Failed to get load balancer: {e}")
        return self._load_balancer_ref

    def add_block(
        self,
        content_type: ContentType,
        content: str,
        priority: float = 0.5,
        topic: str = "",
        source: str = "",
    ) -> bool:
        token_estimate = max(1, len(content) // 4)
        block = ContextBlock(
            content_type=content_type,
            content=content,
            token_estimate=token_estimate,
            priority=priority,
            topic=topic,
            source=source,
        )
        with self._lock:
            self._blocks.append(block)
        return True

    def adjust_for_cognitive_state(self, state_name: str) -> ContextStrategy:
        strategy, budget_ratio = self._STRATEGY_MAP.get(
            state_name, (ContextStrategy.FOCUSED, 0.7)
        )
        budget = int(self._max_tokens * budget_ratio)

        with self._lock:
            old_strategy = self._current_strategy
            self._current_strategy = strategy
            self._current_budget = budget
            self._cognitive_state = state_name
            self._stats["total_adjustments"] += 1
            if old_strategy != strategy:
                self._stats["strategy_changes"] += 1
                self._adjustment_history.append({
                    "from": old_strategy.name,
                    "to": strategy.name,
                    "state": state_name,
                    "budget": budget,
                    "timestamp": time.time(),
                })

        return strategy

    def assemble_context(self, max_tokens: int | None = None) -> str:
        budget = max_tokens or self._current_budget
        reserve = min(self._RESERVE_TOKENS, self._max_tokens // 4)
        budget = max(self._MIN_TOKENS, min(budget, self._max_tokens - reserve))

        with self._lock:
            blocks = list(self._blocks)

        priority_boost = self._compute_priority_boosts()

        scored_blocks = []
        for i, block in enumerate(blocks):
            score = block.priority
            if block.topic and self._focus_topic and block.topic == self._focus_topic:
                score += 0.3
            if block.content_type == ContentType.SYSTEM:
                score += 0.5
            elif block.content_type == ContentType.CONVERSATION:
                score += 0.2
            if block.content_type in priority_boost:
                score += priority_boost[block.content_type]
            recency = max(0, 1.0 - (time.time() - block.timestamp) / 3600)
            score += recency * 0.2

            if self._current_strategy == ContextStrategy.MINIMAL:
                if block.content_type not in (ContentType.SYSTEM, ContentType.CONVERSATION):
                    score *= 0.3
            elif self._current_strategy == ContextStrategy.SUMMARIZED:
                if block.content_type == ContentType.BACKGROUND:
                    score *= 0.5

            scored_blocks.append((score, block))

        scored_blocks.sort(key=lambda x: (-x[0], x[1].timestamp))

        selected = []
        total_tokens = 0
        evicted = 0

        for score, block in scored_blocks:
            content = block.content
            tokens = block.token_estimate

            if self._current_strategy == ContextStrategy.SUMMARIZED:
                if block.content_type == ContentType.BACKGROUND and tokens > 200:
                    content = content[:200] + "..."
                    tokens = max(1, len(content) // 4)
            elif self._current_strategy == ContextStrategy.MINIMAL:
                if block.content_type == ContentType.BACKGROUND:
                    content = content[:100] + "..."
                    tokens = max(1, len(content) // 4)

            if total_tokens + tokens <= budget:
                selected.append(content)
                total_tokens += tokens
            else:
                if block.content_type != ContentType.SYSTEM:
                    evicted += 1

        with self._lock:
            self._stats["blocks_evicted"] += evicted
            self._stats["tokens_saved"] += sum(
                b.token_estimate for _, b in scored_blocks if b.content not in selected
            )

        return "\n\n".join(selected)

    def _compute_priority_boosts(self) -> dict[ContentType, float]:
        boosts = {}
        aq = self._get_attention_queue()
        if aq is not None and aq.size() > 0:
            from core.attention_priority import UrgencySignal
            top = aq.peek()
            if top:
                if top.urgency == UrgencySignal.SAFETY_CRITICAL:
                    boosts[ContentType.SYSTEM] = 0.5
                elif top.urgency in (UrgencySignal.CORRECTION, UrgencySignal.TIME_SENSITIVE):
                    boosts[ContentType.CONVERSATION] = 0.3

        dd = self._get_drift_detector()
        if dd is not None:
            try:
                current = dd.get_current_topic()
                if current:
                    self._focus_topic = current
            except Exception as e:
                logger.debug(f"Failed to get current topic from drift detector: {e}")

        return boosts

    def get_stats(self) -> dict:
        with self._lock:
            return {
                **self._stats,
                "current_strategy": self._current_strategy.name,
                "current_budget": self._current_budget,
                "cognitive_state": self._cognitive_state,
                "focus_topic": self._focus_topic,
                "blocks_count": len(self._blocks),
                "total_block_tokens": sum(b.token_estimate for b in self._blocks),
                "adjustment_count": len(self._adjustment_history),
            }

    def set_focus_topic(self, topic: str):
        with self._lock:
            self._focus_topic = topic

    def clear(self):
        with self._lock:
            self._blocks.clear()
            self._adjustment_history.clear()


_adaptive_window: AdaptiveContextWindow | None = None
_window_lock = threading.Lock()


def get_adaptive_window() -> AdaptiveContextWindow:
    global _adaptive_window
    if _adaptive_window is None:
        with _window_lock:
            if _adaptive_window is None:
                _adaptive_window = AdaptiveContextWindow()
    return _adaptive_window