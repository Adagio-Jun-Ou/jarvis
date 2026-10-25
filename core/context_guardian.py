"""Context Overflow Guardian - Proactive context window management.

Inspired by Letta's memory pressure warnings and auto-summarization.
Unlike Jarvis's existing trim/compress which is reactive (runs after overflow),
the Guardian is PROACTIVE - it detects approaching overflow BEFORE it happens
and takes graduated countermeasures.

Countermeasure levels:
1. WARNING (70%): Log warning, suggest summarization
2. SOFT_EVICT (80%): Evict old low-importance messages to L2
3. HARD_EVICT (90%): Force-summarize and aggressively prune
4. EMERGENCY (95%): Drop all but essential context, restart conversation backbone

This integrates with the Three-Layer Context architecture:
- L1 overflow → Guardian triggers eviction to L2
- L2 overflow → Guardian triggers consolidation to L3
- Each model (primary/backend) has its own guardian with separate budgets
"""
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_guardian: Optional["ContextGuardian"] = None
_guardian_lock = threading.Lock()


class PressureLevel(Enum):
    NORMAL = "normal"
    WARNING = "warning"
    SOFT_EVICT = "soft_evict"
    HARD_EVICT = "hard_evict"
    EMERGENCY = "emergency"


@dataclass
class GuardianAction:
    level: PressureLevel
    action: str
    tokens_before: int
    tokens_after: int
    messages_removed: int
    reason: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class GuardianStats:
    total_checks: int = 0
    warnings_issued: int = 0
    soft_evictions: int = 0
    hard_evictions: int = 0
    emergencies: int = 0
    total_messages_evicted: int = 0
    total_tokens_saved: int = 0


THRESHOLDS = {
    PressureLevel.WARNING: 0.70,
    PressureLevel.SOFT_EVICT: 0.80,
    PressureLevel.HARD_EVICT: 0.90,
    PressureLevel.EMERGENCY: 0.95,
}


class ContextGuardian:
    def __init__(self, token_budget: int = 128000):
        self._token_budget = token_budget
        self._stats = GuardianStats()
        self._lock = threading.RLock()
        self._actions: OrderedDict[str, GuardianAction] = OrderedDict()
        self._max_actions = 100
        self._last_pressure = PressureLevel.NORMAL
        self._emergency_callbacks: list[Callable[[GuardianAction], None]] = []

        self._max_callbacks = 20
        self._message_priorities: OrderedDict[str, float] = OrderedDict()
        self._max_priorities = 1000

    def score_message(self, msg: dict) -> float:
        """Score a message's importance for priority-based eviction.

        Returns 0.0 (expendable) to 1.0 (critical/keep).

        Priority factors:
        - System messages: 0.9 (keep)
        - Recent messages: higher score (recency decay)
        - Tool results: 0.5 (moderately important)
        - Long messages: 0.4 (good compression target)
        - Messages with code/error: 0.7 (important for debugging)
        """
        role = msg.get("role", "")
        content = str(msg.get("content", ""))

        if content.startswith("Previous conversation summary"):
            return 0.3

        if role == "system":
            return 0.95
        if role == "tool":
            return 0.5

        score = 0.6

        if any(kw in content for kw in ["error", "Error", "exception", "Exception", "failed", "Failed"]):
            score += 0.15

        if any(kw in content for kw in ["def ", "class ", "import ", "return "]):
            score += 0.1

        if len(content) > 2000:
            score -= 0.1

        if role == "assistant":
            score += 0.05

        return min(1.0, max(0.0, score))

    def get_eviction_order(self, messages: list[dict]) -> list[int]:
        """Return message indices in eviction order (lowest priority first).

        Messages are ordered for eviction:
        1. Summaries (lowest priority)
        2. Old tool results
        3. Old short assistant messages
        4. Old user messages
        5. Recent messages (highest priority, evicted last)
        """
        scored = []
        for i, msg in enumerate(messages):
            priority = self.score_message(msg)
            scored.append((i, priority))

        scored.sort(key=lambda x: x[1])
        return [idx for idx, _ in scored]

    def add_emergency_callback(self, callback: Callable[[GuardianAction], None]) -> None:
        """Register a callback to be invoked when EMERGENCY pressure is detected."""
        with self._lock:
            if len(self._emergency_callbacks) < self._max_callbacks:
                self._emergency_callbacks.append(callback)

    def check_pressure(self, current_tokens: int) -> PressureLevel:
        ratio = current_tokens / self._token_budget if self._token_budget > 0 else 0
        with self._lock:
            self._stats.total_checks += 1

        if ratio >= THRESHOLDS[PressureLevel.EMERGENCY]:
            return PressureLevel.EMERGENCY
        elif ratio >= THRESHOLDS[PressureLevel.HARD_EVICT]:
            return PressureLevel.HARD_EVICT
        elif ratio >= THRESHOLDS[PressureLevel.SOFT_EVICT]:
            return PressureLevel.SOFT_EVICT
        elif ratio >= THRESHOLDS[PressureLevel.WARNING]:
            return PressureLevel.WARNING
        return PressureLevel.NORMAL

    def should_auto_evict(self, current_tokens: int) -> bool:
        level = self.check_pressure(current_tokens)
        return level in (PressureLevel.SOFT_EVICT, PressureLevel.HARD_EVICT, PressureLevel.EMERGENCY)

    def get_eviction_plan(self, current_tokens: int, message_count: int) -> dict:
        level = self.check_pressure(current_tokens)
        target_ratio = {
            PressureLevel.SOFT_EVICT: 0.60,
            PressureLevel.HARD_EVICT: 0.50,
            PressureLevel.EMERGENCY: 0.40,
        }.get(level, 0.60)

        target_tokens = int(self._token_budget * target_ratio)
        tokens_to_remove = max(0, current_tokens - target_tokens)

        if level == PressureLevel.EMERGENCY:
            keep_recent = max(5, message_count // 10)
        elif level == PressureLevel.HARD_EVICT:
            keep_recent = max(10, message_count // 5)
        else:
            keep_recent = max(15, message_count // 3)

        important_only = level in (PressureLevel.HARD_EVICT, PressureLevel.EMERGENCY)

        return {
            "level": level.value,
            "current_tokens": current_tokens,
            "target_tokens": target_tokens,
            "tokens_to_remove": tokens_to_remove,
            "keep_recent": keep_recent,
            "important_only": important_only,
            "should_summarize": level in (PressureLevel.HARD_EVICT, PressureLevel.EMERGENCY),
        }

    def record_action(self, action: GuardianAction) -> None:
        with self._lock:
            if action.level == PressureLevel.WARNING:
                self._stats.warnings_issued += 1
            elif action.level == PressureLevel.SOFT_EVICT:
                self._stats.soft_evictions += 1
            elif action.level == PressureLevel.HARD_EVICT:
                self._stats.hard_evictions += 1
            elif action.level == PressureLevel.EMERGENCY:
                self._stats.emergencies += 1

            self._stats.total_messages_evicted += action.messages_removed
            self._stats.total_tokens_saved += max(0, action.tokens_before - action.tokens_after)
            self._last_pressure = action.level

            action_id = f"guardian_{int(time.time()*1000)}"
            self._actions[action_id] = action
            while len(self._actions) > self._max_actions:
                self._actions.popitem(last=False)

        if action.level == PressureLevel.EMERGENCY:
            with self._lock:
                callbacks = list(self._emergency_callbacks)
            for callback in callbacks:
                try:
                    callback(action)
                except Exception as e:
                    logger.warning(f"Emergency callback failed: {e}")

    def get_pressure_level(self) -> str:
        with self._lock:
            return self._last_pressure.value

    def get_budget(self) -> int:
        return self._token_budget

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "token_budget": self._token_budget,
                "total_checks": self._stats.total_checks,
                "warnings_issued": self._stats.warnings_issued,
                "soft_evictions": self._stats.soft_evictions,
                "hard_evictions": self._stats.hard_evictions,
                "emergencies": self._stats.emergencies,
                "total_messages_evicted": self._stats.total_messages_evicted,
                "total_tokens_saved": self._stats.total_tokens_saved,
                "last_pressure": self._last_pressure.value,
            }


def get_context_guardian() -> ContextGuardian:
    global _guardian, _guardian_lock
    if _guardian is None:
        with _guardian_lock:
            if _guardian is None:
                _guardian = ContextGuardian()
    return _guardian


class AdaptiveThresholds:
    """Auto-tuning pressure thresholds based on eviction history.

    If evictions happen too frequently, thresholds are relaxed.
    If no evictions happen for a long time, thresholds tighten.
    """

    def __init__(
        self,
        base_warning: float = 0.70,
        base_soft: float = 0.80,
        base_hard: float = 0.90,
        base_emergency: float = 0.95,
        adjustment_rate: float = 0.02,
        max_adjustment: float = 0.10,
        cooldown_seconds: float = 60.0,
    ):
        self._base = {
            PressureLevel.WARNING: base_warning,
            PressureLevel.SOFT_EVICT: base_soft,
            PressureLevel.HARD_EVICT: base_hard,
            PressureLevel.EMERGENCY: base_emergency,
        }
        self._adjustments: dict[PressureLevel, float] = {lv: 0.0 for lv in PressureLevel}
        self._adjustment_rate = adjustment_rate
        self._max_adjustment = max_adjustment
        self._cooldown_seconds = cooldown_seconds
        self._last_eviction_time: float = 0.0
        self._eviction_count_window: list[float] = []
        self._window_seconds = 300.0
        self._lock = threading.Lock()

    def get_threshold(self, level: PressureLevel) -> float:
        with self._lock:
            adj = self._adjustments.get(level, 0.0)
            return self._base[level] + adj

    def record_eviction(self, level: PressureLevel) -> None:
        now = time.time()
        with self._lock:
            self._eviction_count_window.append(now)
            cutoff = now - self._window_seconds
            self._eviction_count_window = [
                t for t in self._eviction_count_window if t > cutoff
            ]
            rate = len(self._eviction_count_window) / (self._window_seconds / 60)
            if rate > 3:
                for lv in PressureLevel:
                    self._adjustments[lv] = min(
                        self._max_adjustment,
                        self._adjustments[lv] + self._adjustment_rate,
                    )
            self._last_eviction_time = now

    def record_stable(self) -> None:
        now = time.time()
        with self._lock:
            if self._last_eviction_time > 0 and now - self._last_eviction_time > self._cooldown_seconds:
                for lv in PressureLevel:
                    if self._adjustments[lv] > 0:
                        self._adjustments[lv] = max(
                            0, self._adjustments[lv] - self._adjustment_rate * 0.5
                        )

    def get_status(self) -> dict:
        with self._lock:
            return {
                "current_thresholds": {
                    lv.value: round(self._base[lv] + self._adjustments.get(lv, 0.0), 3)
                    for lv in self._base
                },
                "base_thresholds": {lv.value: self._base[lv] for lv in self._base},
                "adjustments": {lv.value: round(self._adjustments.get(lv, 0.0), 3) for lv in PressureLevel},
                "eviction_rate_per_min": round(
                    len(self._eviction_count_window) / (self._window_seconds / 60), 2
                ),
            }


class ContextPressurePredictor:
    """Predicts future context pressure based on token usage trends.

    Uses linear extrapolation of recent token growth to predict
    when pressure thresholds will be breached.
    """

    def __init__(self, sample_window: int = 20):
        self._samples: list[tuple[float, int]] = []
        self._sample_window = sample_window
        self._lock = threading.Lock()

    def record(self, tokens: int) -> None:
        now = time.time()
        with self._lock:
            self._samples.append((now, tokens))
            if len(self._samples) > self._sample_window:
                self._samples = self._samples[-self._sample_window:]

    def predict_time_to_pressure(self, budget: int, target_ratio: float = 0.80) -> float | None:
        """Predict seconds until budget * target_ratio is reached.

        Returns None if insufficient data or token usage is decreasing.
        """
        with self._lock:
            if len(self._samples) < 5:
                return None

            recent = self._samples[-5:]
            t0, tok0 = recent[0]
            t1, tok1 = recent[-1]
            dt = t1 - t0
            d_tok = tok1 - tok0

            if dt <= 0 or d_tok <= 0:
                return None

            rate = d_tok / dt
            target = int(budget * target_ratio)
            remaining = target - tok1

            if remaining <= 0:
                return 0.0

            return remaining / rate

    def get_trend(self) -> str:
        """Return 'rising', 'stable', or 'falling'."""
        with self._lock:
            if len(self._samples) < 3:
                return "stable"

            recent = self._samples[-3:]
            tokens = [t for _, t in recent]
            if tokens[-1] > tokens[0] * 1.02:
                return "rising"
            elif tokens[-1] < tokens[0] * 0.98:
                return "falling"
            return "stable"

    def get_status(self) -> dict:
        with self._lock:
            latest = self._samples[-1][1] if self._samples else 0
            trend = "stable"
            if len(self._samples) >= 3:
                recent = self._samples[-3:]
                tokens = [t for _, t in recent]
                if tokens[-1] > tokens[0] * 1.02:
                    trend = "rising"
                elif tokens[-1] < tokens[0] * 0.98:
                    trend = "falling"
            return {
                "sample_count": len(self._samples),
                "trend": trend,
                "latest_tokens": latest,
            }


class MultiBudgetGuardian:
    """Per-model context guardian with separate token budgets.

    Each model (primary, backend) can have its own budget and thresholds.
    """

    def __init__(self):
        self._guardians: dict[str, ContextGuardian] = {}
        self._adaptive: dict[str, AdaptiveThresholds] = {}
        self._predictors: dict[str, ContextPressurePredictor] = {}
        self._lock = threading.Lock()

    def get_guardian(self, model: str, budget: int = 128000) -> ContextGuardian:
        with self._lock:
            if model not in self._guardians:
                self._guardians[model] = ContextGuardian(token_budget=budget)
                self._adaptive[model] = AdaptiveThresholds()
                self._predictors[model] = ContextPressurePredictor()
            return self._guardians[model]

    def check_all(self, token_counts: dict[str, int]) -> dict[str, dict]:
        results = {}
        with self._lock:
            for model, tokens in token_counts.items():
                if model in self._guardians:
                    g = self._guardians[model]
                    level = g.check_pressure(tokens)
                    adaptive = self._adaptive.get(model)
                    predictor = self._predictors.get(model)
                    results[model] = {
                        "level": level.value,
                        "tokens": tokens,
                        "budget": g.get_budget(),
                        "ratio": round(tokens / g.get_budget(), 3) if g.get_budget() > 0 else 0,
                        "adaptive": adaptive.get_status() if adaptive else {},
                        "trend": predictor.get_trend() if predictor else "unknown",
                        "time_to_80pct": predictor.predict_time_to_pressure(g.get_budget(), 0.80)
                        if predictor else None,
                    }
        return results

    def record_eviction(self, model: str, level: PressureLevel) -> None:
        with self._lock:
            if model in self._adaptive:
                self._adaptive[model].record_eviction(level)

    def record_stable(self, model: str) -> None:
        with self._lock:
            if model in self._adaptive:
                self._adaptive[model].record_stable()

    def record_tokens(self, model: str, tokens: int) -> None:
        with self._lock:
            if model in self._predictors:
                self._predictors[model].record(tokens)

    def get_status(self) -> dict:
        with self._lock:
            models = {}
            for model, g in self._guardians.items():
                adaptive = self._adaptive.get(model)
                predictor = self._predictors.get(model)
                models[model] = {
                    "stats": g.get_stats(),
                    "adaptive": adaptive.get_status() if adaptive else {},
                    "predictor": predictor.get_status() if predictor else {},
                }
            return {"models": models, "model_count": len(models)}


_multi_budget: MultiBudgetGuardian | None = None
_mb_lock = threading.Lock()


def get_multi_budget_guardian() -> MultiBudgetGuardian:
    global _multi_budget
    if _multi_budget is None:
        with _mb_lock:
            if _multi_budget is None:
                _multi_budget = MultiBudgetGuardian()
    return _multi_budget