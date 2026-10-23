"""Goal Anticipation Engine - Predicting what the user needs before they ask.

A true AI butler doesn't wait to be told what to do — it anticipates.
This engine learns user patterns, infers goals from context, and prepares
suggestions, resources, and actions before the user explicitly requests them.

Three layers of anticipation:
1. **Pattern-Based** — "You usually check tests after editing code"
2. **Context-Based** — "You just modified 3 files in the same module, likely about to test"
3. **Goal-Based** — "Your stated goal is X, next logical step is Y"

Key design principles:
- Never annoying: suggestions are ranked by confidence and delivery timing
- Respects focus: suppresses low-confidence suggestions during deep work
- Learns from dismissal: tracks what the user ignores and reduces similar suggestions
- Explains reasoning: every suggestion includes WHY it was made
"""
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

_anticipation: Optional["GoalAnticipationEngine"] = None
_anticipation_lock = threading.Lock()


class AnticipationSource(Enum):
    PATTERN = "pattern"
    CONTEXT = "context"
    GOAL = "goal"
    SEQUENCE = "sequence"
    SCHEDULE = "schedule"


class AnticipationConfidence(Enum):
    SPECULATIVE = 0
    POSSIBLE = 1
    LIKELY = 2
    PROBABLE = 3
    CERTAIN = 4


@dataclass
class AnticipatedGoal:
    description: str
    source: AnticipationSource
    confidence: float
    reasoning: str
    suggested_action: str
    preparation: str = ""
    relevant_context: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    delivered: bool = False
    accepted: bool = False
    dismissed: bool = False


@dataclass
class UserPattern:
    trigger: str
    followed_by: str
    frequency: int = 1
    last_seen: float = field(default_factory=time.time)
    success_rate: float = 1.0
    dismissed_count: int = 0


@dataclass
class ActionSequence:
    actions: list[str]
    timestamp: float = field(default_factory=time.time)
    count: int = 1


@dataclass
class AnticipationState:
    active_anticipations: list[AnticipatedGoal] = field(default_factory=list)
    learned_patterns: dict[str, UserPattern] = field(default_factory=dict)
    recent_actions: deque = field(default_factory=lambda: deque(maxlen=50))
    sequence_history: deque = field(default_factory=lambda: deque(maxlen=30))
    dismissal_history: dict[str, int] = field(default_factory=dict)
    anticipation_count: int = 0
    accepted_count: int = 0
    dismissed_count: int = 0


_MAX_ACTIVE = 15
_PATTERN_MIN_FREQUENCY = 2
_PATTERN_WINDOW_SECONDS = 3600
_SEQUENCE_LENGTH = 3


class GoalAnticipationEngine:
    def __init__(self):
        self._lock = threading.RLock()
        self._state = AnticipationState()
        self._current_goals: list[str] = []
        self._last_action_time: float = 0.0
        self._action_times: deque[float] = deque(maxlen=100)

    def record_action(self, action_type: str, details: str = "") -> Optional[AnticipatedGoal]:
        with self._lock:
            now = time.time()
            self._state.recent_actions.append({"type": action_type, "details": details, "time": now})
            self._action_times.append(now)
            self._last_action_time = now

            self._learn_pattern(action_type)

            anticipation = self._anticipate_from_action(action_type, details)
            if anticipation:
                self._state.active_anticipations.append(anticipation)
                if len(self._state.active_anticipations) > _MAX_ACTIVE:
                    self._state.active_anticipations = self._state.active_anticipations[-_MAX_ACTIVE:]
                self._state.anticipation_count += 1

            self._update_sequences(action_type)

            if self._current_goals:
                goal_anticipation = self._anticipate_from_goals(action_type)
                if goal_anticipation and not any(
                    a.description == goal_anticipation.description for a in self._state.active_anticipations
                ):
                    self._state.active_anticipations.append(goal_anticipation)
                    if len(self._state.active_anticipations) > _MAX_ACTIVE:
                        self._state.active_anticipations = self._state.active_anticipations[-_MAX_ACTIVE:]
                    self._state.anticipation_count += 1

            return anticipation

    def set_active_goals(self, goals: list[str]) -> None:
        with self._lock:
            self._current_goals = goals

    def get_pending_anticipations(self, focus_score: float = 0.5) -> list[AnticipatedGoal]:
        with self._lock:
            now = time.time()
            min_confidence = 0.3 + focus_score * 0.4 if focus_score > 0.6 else 0.3
            pending = []
            for a in self._state.active_anticipations:
                if a.delivered or a.accepted or a.dismissed:
                    continue
                if now - a.timestamp > _PATTERN_WINDOW_SECONDS:
                    continue
                dismiss_key = a.description[:30]
                if self._state.dismissal_history.get(dismiss_key, 0) >= 3:
                    continue
                if a.confidence >= min_confidence:
                    pending.append(a)
            pending.sort(key=lambda x: x.confidence, reverse=True)
            return pending[:5]

    def deliver_anticipation(self, index: int) -> bool:
        with self._lock:
            pending = [a for a in self._state.active_anticipations if not a.delivered and not a.accepted and not a.dismissed]
            if 0 <= index < len(pending):
                pending[index].delivered = True
                return True
            return False

    def accept_anticipation(self, index: int) -> bool:
        with self._lock:
            delivered = [a for a in self._state.active_anticipations if a.delivered and not a.accepted and not a.dismissed]
            if 0 <= index < len(delivered):
                delivered[index].accepted = True
                self._state.accepted_count += 1
                key = delivered[index].description[:30]
                self._state.dismissal_history.pop(key, None)
                return True
            return False

    def dismiss_anticipation(self, index: int) -> bool:
        with self._lock:
            delivered_or_pending = [a for a in self._state.active_anticipations if not a.accepted and not a.dismissed]
            if 0 <= index < len(delivered_or_pending):
                delivered_or_pending[index].dismissed = True
                self._state.dismissed_count += 1
                key = delivered_or_pending[index].description[:30]
                self._state.dismissal_history[key] = self._state.dismissal_history.get(key, 0) + 1
                pattern_key = f"{delivered_or_pending[index].source.value}:{key}"
                if pattern_key in self._state.learned_patterns:
                    self._state.learned_patterns[pattern_key].dismissed_count += 1
                    p = self._state.learned_patterns[pattern_key]
                    if p.frequency > 0:
                        p.success_rate = max(0.0, 1.0 - (p.dismissed_count / p.frequency))
                return True
            return False

    def get_anticipation_context(self) -> str:
        with self._lock:
            parts = []
            if self._current_goals:
                parts.append(f"User goals: {'; '.join(self._current_goals[:3])}")
            pending = self.get_pending_anticipations()
            if pending:
                top = pending[0]
                parts.append(f"Anticipated: {top.description} ({top.source.value}, {top.confidence:.0%})")
                parts.append(f"Reasoning: {top.reasoning}")
                if top.suggested_action:
                    parts.append(f"Suggested: {top.suggested_action}")
                if top.preparation:
                    parts.append(f"Prepared: {top.preparation}")
            active_patterns = sum(1 for p in self._state.learned_patterns.values() if p.frequency >= _PATTERN_MIN_FREQUENCY)
            if active_patterns:
                parts.append(f"Learned patterns: {active_patterns}")
            return " | ".join(parts) if parts else ""

    def get_state(self) -> dict:
        with self._lock:
            active = [a for a in self._state.active_anticipations if not a.dismissed]
            patterns = {k: {"trigger": p.trigger, "followed_by": p.followed_by, "frequency": p.frequency, "success_rate": round(p.success_rate, 2)} for k, p in self._state.learned_patterns.items() if p.frequency >= _PATTERN_MIN_FREQUENCY}
            return {
                "active_anticipations": len(active),
                "total_anticipated": self._state.anticipation_count,
                "accepted": self._state.accepted_count,
                "dismissed": self._state.dismissed_count,
                "learned_patterns": patterns,
                "current_goals": self._current_goals[:5],
                "recent_actions": len(self._state.recent_actions),
            }

    def _learn_pattern(self, action: str) -> None:
        if len(self._state.recent_actions) < 2:
            return
        prev = None
        for a in reversed(list(self._state.recent_actions)[:-1]):
            if time.time() - a["time"] < _PATTERN_WINDOW_SECONDS:
                prev = a["type"]
                break
        if prev and prev != action:
            key = f"{prev}→{action}"
            now = time.time()
            if key in self._state.learned_patterns:
                self._state.learned_patterns[key].frequency += 1
                self._state.learned_patterns[key].last_seen = now
            else:
                self._state.learned_patterns[key] = UserPattern(trigger=prev, followed_by=action)

    def _anticipate_from_action(self, action: str, details: str = "") -> Optional[AnticipatedGoal]:
        for key, pattern in self._state.learned_patterns.items():
            if pattern.trigger == action and pattern.frequency >= _PATTERN_MIN_FREQUENCY and pattern.success_rate > 0.3:
                confidence = min(0.95, 0.3 + pattern.frequency * 0.1) * pattern.success_rate
                if confidence < 0.2:
                    continue
                return AnticipatedGoal(
                    description=f"Next: {pattern.followed_by}",
                    source=AnticipationSource.PATTERN,
                    confidence=confidence,
                    reasoning=f"After '{pattern.trigger}', you typically {pattern.followed_by} ({pattern.frequency}x observed, {pattern.success_rate:.0%} success)",
                    suggested_action=pattern.followed_by,
                )
        context_anticipation = self._anticipate_from_context(action, details)
        if context_anticipation:
            return context_anticipation
        return None

    _CODE_EDIT_PATTERNS = {
        "code_edit": {"next": "run_tests", "confidence": 0.6, "reason": "After editing code, running tests is standard practice"},
        "file_write": {"next": "verify_changes", "confidence": 0.5, "reason": "After writing a file, verifying the changes is wise"},
        "error_seen": {"next": "debug", "confidence": 0.7, "reason": "After encountering an error, debugging is the logical next step"},
        "test_fail": {"next": "fix_code", "confidence": 0.75, "reason": "After a test failure, fixing the code is the expected response"},
        "test_pass": {"next": "commit_changes", "confidence": 0.4, "reason": "Tests passing — good time to commit"},
        "deploy": {"next": "monitor", "confidence": 0.5, "reason": "After deployment, monitoring is essential"},
        "query_data": {"next": "export_results", "confidence": 0.35, "reason": "After querying, you may want to export"},
        "create_project": {"next": "setup_tests", "confidence": 0.45, "reason": "New projects need test setup"},
    }

    _DETAIL_PATTERNS = {
        "error": {"keywords": ["error", "exception", "traceback", "failed", "failure", "crash"], "next": "debug", "confidence": 0.7, "reason": "Error detected — debugging is the logical next step"},
        "warning": {"keywords": ["warning", "warn", "deprecated", "slow"], "next": "investigate", "confidence": 0.4, "reason": "Warning detected — investigation may be warranted"},
        "success": {"keywords": ["passed", "success", "completed", "done", "finished"], "next": "commit_or_deploy", "confidence": 0.45, "reason": "Operation succeeded — consider committing or deploying"},
        "test": {"keywords": ["test", "spec", "coverage"], "next": "run_tests", "confidence": 0.55, "reason": "Test-related activity — running the test suite is standard next step"},
        "config": {"keywords": ["config", "settings", "env", "secret", "credential"], "next": "verify_config", "confidence": 0.5, "reason": "Configuration change — verify the configuration is valid"},
        "security": {"keywords": ["auth", "permission", "vulnerability", "exploit", "inject"], "next": "security_review", "confidence": 0.6, "reason": "Security-related action — a review is prudent"},
        "database": {"keywords": ["migrate", "schema", "query", "sql", "database"], "next": "run_migrations", "confidence": 0.5, "reason": "Database changes — run migrations to apply"},
        "network": {"keywords": ["endpoint", "api", "request", "response", "http"], "next": "test_endpoint", "confidence": 0.45, "reason": "Network/API change — test the endpoint"},
        "install": {"keywords": ["install", "update", "upgrade", "depend"], "next": "verify_compatibility", "confidence": 0.5, "reason": "Dependency change — verify compatibility"},
        "documentation": {"keywords": ["docs", "readme", "comment", "changelog"], "next": "review_docs", "confidence": 0.3, "reason": "Documentation updated — review for accuracy"},
    }

    _TIME_OF_DAY_PATTERNS = {
        "morning": {"hours": range(6, 10), "anticipations": [("review_tasks", 0.4, "Morning — review daily tasks"), ("check_messages", 0.3, "Morning — check overnight messages")]},
        "midday": {"hours": range(11, 14), "anticipations": [("break_reminder", 0.3, "Midday — consider taking a break")]},
        "afternoon": {"hours": range(14, 18), "anticipations": [("deep_work", 0.45, "Afternoon — good for deep work"), ("commit_progress", 0.35, "Afternoon — commit progress")]},
        "evening": {"hours": range(18, 22), "anticipations": [("wrap_up", 0.5, "Evening — wrap up and summarize"), ("plan_tomorrow", 0.35, "Evening — plan tomorrow")]},
        "late_night": {"hours": range(22, 30), "anticipations": [("save_work", 0.6, "Late night — save work and rest"), ("simplify", 0.4, "Late night — reduce cognitive load")]},
    }

    def _anticipate_from_context(self, action: str, details: str = "") -> Optional[AnticipatedGoal]:
        if action in self._CODE_EDIT_PATTERNS:
            pattern = self._CODE_EDIT_PATTERNS[action]
            return AnticipatedGoal(
                description=f"Next: {pattern['next']}",
                source=AnticipationSource.CONTEXT,
                confidence=pattern["confidence"],
                reasoning=pattern["reason"],
                suggested_action=pattern["next"],
            )
        if details:
            details_lower = details.lower()
            for pattern_name, pattern in self._DETAIL_PATTERNS.items():
                for keyword in pattern["keywords"]:
                    if keyword in details_lower:
                        return AnticipatedGoal(
                            description=f"Next: {pattern['next']}",
                            source=AnticipationSource.CONTEXT,
                            confidence=pattern["confidence"],
                            reasoning=pattern["reason"],
                            suggested_action=pattern["next"],
                            relevant_context=[keyword],
                        )
        return None

    def _anticipate_from_schedule(self) -> Optional[AnticipatedGoal]:
        hour = time.localtime().tm_hour
        for period_name, period in self._TIME_OF_DAY_PATTERNS.items():
            normalized_hours = [h % 24 for h in period["hours"]]
            if hour in normalized_hours:
                for action, confidence, reasoning in period["anticipations"]:
                    dismiss_key = action[:30]
                    if self._state.dismissal_history.get(dismiss_key, 0) >= 3:
                        continue
                    return AnticipatedGoal(
                        description=f"Time-based: {action}",
                        source=AnticipationSource.SCHEDULE,
                        confidence=confidence,
                        reasoning=reasoning,
                        suggested_action=action,
                    )
        return None

    def _anticipate_from_sequence(self) -> Optional[AnticipatedGoal]:
        recent = list(self._state.recent_actions)
        if len(recent) < 2:
            return None
        last_two = tuple(a["type"] for a in recent[-2:])
        for seq in self._state.sequence_history:
            if seq.count >= 2 and tuple(seq.actions[-2:]) == last_two and len(seq.actions) > 2:
                next_action = seq.actions[-1]
                dismiss_key = f"seq:{next_action[:30]}"
                if self._state.dismissal_history.get(dismiss_key, 0) >= 3:
                    continue
                confidence = min(0.85, 0.3 + seq.count * 0.15)
                return AnticipatedGoal(
                    description=f"Sequence: {next_action}",
                    source=AnticipationSource.SEQUENCE,
                    confidence=confidence,
                    reasoning=f"After {' → '.join(seq.actions[:-1])}, next is typically {next_action} ({seq.count}x observed)",
                    suggested_action=next_action,
                )
        return None

    def _anticipate_from_goals(self, action: str) -> Optional[AnticipatedGoal]:
        if not self._current_goals:
            return None
        top_goal = self._current_goals[0]
        goal_action_map = {
            "build": [("write_tests", 0.6, "To build reliably, write tests first"), ("setup_ci", 0.4, "For a build goal, CI is important")],
            "fix": [("reproduce", 0.7, "To fix a bug, first reproduce it"), ("write_regression_test", 0.6, "Add a regression test after fixing")],
            "deploy": [("run_tests", 0.8, "Deploy only after tests pass"), ("check_dependencies", 0.5, "Verify dependencies before deploying")],
            "learn": [("read_docs", 0.5, "Documentation is a good starting point"), ("try_example", 0.6, "Hands-on examples accelerate learning")],
            "optimize": [("benchmark", 0.7, "Measure before optimizing"), ("profile", 0.65, "Profile to find the bottleneck")],
            "refactor": [("ensure_tests", 0.8, "Have tests before refactoring"), ("review_deps", 0.3, "Check if dependencies need updating")],
            "debug": [("isolate", 0.7, "Isolate the failure case"), ("add_logging", 0.5, "Add diagnostic logging")],
            "review": [("run_linter", 0.6, "Run linter first"), ("check_tests", 0.5, "Verify tests pass")],
            "document": [("review_api", 0.5, "Review API surface"), ("add_examples", 0.45, "Add usage examples")],
            "test": [("check_coverage", 0.6, "Check test coverage"), ("add_edge_cases", 0.5, "Add edge case tests")],
            "release": [("update_changelog", 0.7, "Update changelog"), ("tag_version", 0.5, "Tag the release version")],
        }
        for goal_key, anticipations in goal_action_map.items():
            if goal_key in top_goal.lower():
                for next_action, confidence, reasoning in anticipations:
                    if action != next_action:
                        return AnticipatedGoal(
                            description=f"Goal step: {next_action}",
                            source=AnticipationSource.GOAL,
                            confidence=confidence,
                            reasoning=f"For goal '{top_goal}': {reasoning}",
                            suggested_action=next_action,
                        )
        return None

    def _update_sequences(self, action: str) -> None:
        recent = list(self._state.recent_actions)
        if len(recent) >= _SEQUENCE_LENGTH:
            seq = [a["type"] for a in recent[-_SEQUENCE_LENGTH:]]
            for existing in self._state.sequence_history:
                if existing.actions == seq:
                    existing.count += 1
                    return
            self._state.sequence_history.append(ActionSequence(actions=seq))

    def tick(self) -> dict:
        with self._lock:
            stale_keys = [k for k, v in self._state.dismissal_history.items() if v <= 0]
            for k in stale_keys:
                del self._state.dismissal_history[k]
            if len(self._state.dismissal_history) > 50:
                sorted_items = sorted(self._state.dismissal_history.items(), key=lambda x: x[1], reverse=True)
                self._state.dismissal_history = dict(sorted_items[:50])
            schedule_anticipation = self._anticipate_from_schedule()
            if schedule_anticipation and not any(
                a.description == schedule_anticipation.description for a in self._state.active_anticipations
            ):
                self._state.active_anticipations.append(schedule_anticipation)
                if len(self._state.active_anticipations) > _MAX_ACTIVE:
                    self._state.active_anticipations = self._state.active_anticipations[-_MAX_ACTIVE:]
                self._state.anticipation_count += 1
            sequence_anticipation = self._anticipate_from_sequence()
            if sequence_anticipation and not any(
                a.description == sequence_anticipation.description for a in self._state.active_anticipations
            ):
                self._state.active_anticipations.append(sequence_anticipation)
                if len(self._state.active_anticipations) > _MAX_ACTIVE:
                    self._state.active_anticipations = self._state.active_anticipations[-_MAX_ACTIVE:]
                self._state.anticipation_count += 1
            return self.get_state()


def get_goal_anticipation() -> GoalAnticipationEngine:
    global _anticipation
    if _anticipation is None:
        with _anticipation_lock:
            if _anticipation is None:
                _anticipation = GoalAnticipationEngine()
    return _anticipation