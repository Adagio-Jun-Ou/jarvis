"""Proactive Intelligence Engine - The brain that makes Jarvis genuinely helpful.

A butler doesn't just observe — it UNDERSTANDS. This engine connects disparate
observations into meaningful patterns and decides what to DO about them.

The difference between a good butler and a great one:
- Good:  "You opened the terminal 5 times in 2 minutes"
- Great: "You seem to be struggling with a command. Here's what you need..."

This engine:
1. **Pattern Synthesis** — connects observations across time and type
   (3 file edits + 2 test fails = "debugging session, offer targeted help")

2. **Meaning Extraction** — derives intent from behavior
   (opening config files repeatedly = "configuring something, offer to help")

3. **Action Decision** — decides WHEN and HOW to intervene
   (don't interrupt deep work; DO offer help when stuck for 5+ minutes)

4. **Learning Loop** — tracks whether interventions helped or annoyed
   (offered help → accepted = +1 relevance; dismissed = -1 relevance)
"""
import logging
import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_proactive: Optional["ProactiveIntelligence"] = None
_proactive_lock = threading.Lock()


class InterventionTiming(Enum):
    IMMEDIATE = "immediate"
    NEXT_TURN = "next_turn"
    PASSIVE = "passive"
    SCHEDULED = "scheduled"


class InterventionUrgency(Enum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3


class PatternType(Enum):
    STUCK_LOOP = "stuck_loop"
    DEEP_WORK = "deep_work"
    LEARNING = "learning"
    DEBUGGING = "debugging"
    CONFIGURING = "configuring"
    EXPLORING = "exploring"
    REVIEWING = "reviewing"
    DEPLOYING = "deploying"
    BREAK_NEEDED = "break_needed"
    ERROR_STORM = "error_storm"
    PRODUCTIVE_FLOW = "productive_flow"
    CONTEXT_SWITCHING = "context_switching"
    END_OF_SESSION = "end_of_session"
    START_OF_SESSION = "start_of_session"


@dataclass
class ObservedPattern:
    pattern_type: PatternType
    confidence: float
    evidence: list[str]
    started_at: float = field(default_factory=time.time)
    last_observed: float = field(default_factory=time.time)
    occurrences: int = 1
    dismissed_count: int = 0


@dataclass
class ProposedIntervention:
    action: str
    reason: str
    pattern: PatternType
    timing: InterventionTiming
    urgency: InterventionUrgency
    message: str
    confidence: float
    created_at: float = field(default_factory=time.time)
    delivered: bool = False
    accepted: bool = False
    dismissed: bool = False


@dataclass
class InterventionOutcome:
    action: str
    accepted: bool
    user_response: str = ""
    timestamp: float = field(default_factory=time.time)


_MAX_PATTERNS = 30
_MAX_PROPOSED = 10
_MAX_OUTCOMES = 50
_STUCK_THRESHOLD = 3
_ERROR_STORM_THRESHOLD = 3
_CONTEXT_SWITCH_THRESHOLD = 4
_PRODUCTIVE_FLOW_THRESHOLD = 5


class ProactiveIntelligence:
    def __init__(self, storage_path: str | Path | None = None):
        self._lock = threading.RLock()
        self._patterns: dict[str, ObservedPattern] = {}
        self._proposed: deque[ProposedIntervention] = deque(maxlen=_MAX_PROPOSED)
        self._outcomes: deque[InterventionOutcome] = deque(maxlen=_MAX_OUTCOMES)
        self._recent_actions: deque[dict] = deque(maxlen=30)
        self._recent_errors: deque[float] = deque(maxlen=10)
        self._recent_files: deque[str] = deque(maxlen=20)
        self._session_start: float = time.time()
        self._last_intervention_time: float = 0.0
        self._intervention_cooldown: float = 120.0
        self._acceptance_rate: float = 0.5
        self._storage_path = Path(storage_path) if storage_path else Path.home() / ".jarvis" / "proactive_intelligence"
        self._load_state()

    def observe(self, action_type: str, details: str = "",
                success: bool = True, context: Optional[dict] = None) -> Optional[ProposedIntervention]:
        with self._lock:
            now = time.time()
            self._recent_actions.append({"type": action_type, "details": details,
                                          "success": success, "time": now,
                                          "context": context or {}})
            if not success:
                self._recent_errors.append(now)
            if action_type in ("file_edit", "file_read", "file_write") and details:
                self._recent_files.append(details[:100])
            self._detect_patterns()
            intervention = self._propose_intervention()
            return intervention

    def mark_delivered(self, intervention_id: Optional[int] = None) -> None:
        with self._lock:
            for p in self._proposed:
                if not p.delivered:
                    p.delivered = True
                    break

    def mark_accepted(self, action: str = "") -> None:
        with self._lock:
            for p in reversed(list(self._proposed)):
                if p.delivered and not p.accepted and not p.dismissed:
                    p.accepted = True
                    self._outcomes.append(InterventionOutcome(action=p.action, accepted=True))
                    self._acceptance_rate = min(1.0, self._acceptance_rate + 0.05)
                    break

    def mark_dismissed(self, action: str = "") -> None:
        with self._lock:
            for p in reversed(list(self._proposed)):
                if p.delivered and not p.accepted and not p.dismissed:
                    p.dismissed = True
                    self._outcomes.append(InterventionOutcome(action=p.action, accepted=False))
                    self._acceptance_rate = max(0.1, self._acceptance_rate - 0.1)
                    pattern_key = p.pattern.value
                    if pattern_key in self._patterns:
                        self._patterns[pattern_key].dismissed_count += 1
                    break
            self._save_state()

    def get_pending_interventions(self, presence_state: str = "active",
                                  focus_score: float = 0.5) -> list[ProposedIntervention]:
        with self._lock:
            now = time.time()
            if now - self._last_intervention_time < self._intervention_cooldown:
                return []
            pending = []
            for p in self._proposed:
                if p.delivered or p.accepted or p.dismissed:
                    continue
                if now - p.created_at > 300:
                    continue
                if p.pattern == PatternType.DEEP_WORK and focus_score > 0.7:
                    continue
                if p.urgency.value < InterventionUrgency.HIGH.value and presence_state in ("focused", "deep_work"):
                    continue
                if self._patterns.get(p.pattern.value, None) and self._patterns[p.pattern.value].dismissed_count >= 3:
                    continue
                pending.append(p)
            pending.sort(key=lambda x: x.urgency.value, reverse=True)
            return pending[:3]

    def get_active_patterns(self) -> list[dict]:
        with self._lock:
            return [
                {"type": p.pattern_type.value, "confidence": round(p.confidence, 2),
                 "occurrences": p.occurrences, "evidence": p.evidence[-3:],
                 "last_observed": round(time.time() - p.last_observed, 0)}
                for p in self._patterns.values()
                if p.confidence > 0.4
            ]

    def get_intelligence_context(self, presence: str = "active", focus: float = 0.5) -> str:
        with self._lock:
            parts = []
            active_patterns = [p for p in self._patterns.values() if p.confidence > 0.5
                               and time.time() - p.last_observed < 600]
            if active_patterns:
                top = max(active_patterns, key=lambda p: p.confidence)
                parts.append(f"Detected pattern: {top.pattern_type.value} (confidence: {top.confidence:.0%})")
                if top.evidence:
                    parts.append(f"Evidence: {'; '.join(top.evidence[-2:])}")
            pending = self.get_pending_interventions(presence, focus)
            if pending:
                top_pending = pending[0]
                urgency = "urgent" if top_pending.urgency.value >= 2 else "suggested"
                parts.append(f"{urgency.capitalize()}: {top_pending.message}")
            if self._acceptance_rate < 0.3:
                parts.append("User tends to dismiss suggestions — be less proactive")
            elif self._acceptance_rate > 0.7:
                parts.append("User values proactive suggestions — be more helpful")
            recent_errors = [t for t in self._recent_errors if time.time() - t < 300]
            if len(recent_errors) >= 2:
                parts.append(f"Recent errors: {len(recent_errors)} in last 5 min — user may need help")
            return " | ".join(parts) if parts else ""

    def get_state(self) -> dict:
        with self._lock:
            return {
                "active_patterns": len([p for p in self._patterns.values() if p.confidence > 0.5]),
                "pending_interventions": len([p for p in self._proposed if not p.delivered and not p.dismissed]),
                "acceptance_rate": round(self._acceptance_rate, 2),
                "recent_actions": len(self._recent_actions),
                "recent_errors": len([t for t in self._recent_errors if time.time() - t < 300]),
                "outcomes_tracked": len(self._outcomes),
            }

    def _detect_patterns(self) -> None:
        now = time.time()
        recent = list(self._recent_actions)
        if len(recent) < 2:
            return
        recent_types = [a["type"] for a in recent[-10:]]
        recent_successes = [a["success"] for a in recent[-10:]]
        self._detect_stuck_loop(recent, recent_types, recent_successes, now)
        self._detect_debugging(recent, recent_types, recent_successes, now)
        self._detect_error_storm(now)
        self._detect_configuring(recent_types, now)
        self._detect_exploring(recent_types, now)
        self._detect_context_switching(recent_types, now)
        self._detect_productive_flow(recent, recent_types, recent_successes, now)
        self._detect_error_cascade(now)
        self._detect_time_of_day_pattern()

    def _detect_stuck_loop(self, recent: list, recent_types: list,
                           recent_successes: list, now: float) -> None:
        if len(recent_types) < _STUCK_THRESHOLD:
            return
        last_n = recent_types[-_STUCK_THRESHOLD:]
        if len(set(last_n)) == 1 and last_n[0] in ("file_edit", "terminal_command", "code_edit"):
            failures = [not s for s in recent_successes[-_STUCK_THRESHOLD:]]
            if sum(failures) >= 2:
                self._update_pattern(PatternType.STUCK_LOOP, 0.85,
                                     [f"Repeated {last_n[0]} {len(last_n)} times with failures"])

    def _detect_debugging(self, recent: list, recent_types: list,
                         recent_successes: list, now: float) -> None:
        debug_actions = {"file_edit", "terminal_command", "test_run", "error_seen"}
        debug_count = sum(1 for t in recent_types[-8:] if t in debug_actions)
        failures = sum(1 for s in recent_successes[-8:] if not s)
        if debug_count >= 4 and failures >= 2:
            self._update_pattern(PatternType.DEBUGGING, 0.75,
                                 [f"Debugging session: {debug_count} debug actions, {failures} failures"])

    def _detect_error_storm(self, now: float) -> None:
        recent_errors = [t for t in self._recent_errors if now - t < 120]
        if len(recent_errors) >= _ERROR_STORM_THRESHOLD:
            self._update_pattern(PatternType.ERROR_STORM, 0.9,
                                 [f"{len(recent_errors)} errors in last 2 minutes"])

    def _detect_configuring(self, recent_types: list, now: float) -> None:
        config_actions = sum(1 for t in recent_types[-6:] if t in ("file_edit", "file_read", "terminal_command"))
        config_files = sum(1 for f in self._recent_files
                          if any(ext in f for ext in [".yml", ".yaml", ".json", ".toml", ".env", "config", "settings"]))
        if config_actions >= 3 and config_files >= 2:
            self._update_pattern(PatternType.CONFIGURING, 0.7,
                                 [f"Configuring: {config_files} config files accessed"])

    def _detect_exploring(self, recent_types: list, now: float) -> None:
        unique_types = len(set(recent_types[-8:]))
        reads = sum(1 for t in recent_types[-8:] if t == "file_read")
        if unique_types >= 4 and reads >= 3:
            self._update_pattern(PatternType.EXPLORING, 0.6,
                                 [f"Exploring: {unique_types} different action types, {reads} reads"])

    def _detect_context_switching(self, recent_types: list, now: float) -> None:
        if len(recent_types) < _CONTEXT_SWITCH_THRESHOLD:
            return
        unique_in_last_5 = len(set(recent_types[-_CONTEXT_SWITCH_THRESHOLD:]))
        if unique_in_last_5 >= 4:
            self._update_pattern(PatternType.CONTEXT_SWITCHING, 0.65,
                                 [f"Context switching: {unique_in_last_5} different actions in quick succession"])

    def _detect_productive_flow(self, recent: list, recent_types: list,
                                 recent_successes: list, now: float) -> None:
        if len(recent_successes) < _PRODUCTIVE_FLOW_THRESHOLD:
            return
        successes = sum(1 for s in recent_successes[-_PRODUCTIVE_FLOW_THRESHOLD:] if s)
        if successes >= _PRODUCTIVE_FLOW_THRESHOLD - 1:
            unique = len(set(recent_types[-_PRODUCTIVE_FLOW_THRESHOLD:]))
            if unique <= 3:
                self._update_pattern(PatternType.PRODUCTIVE_FLOW, 0.8,
                                     [f"Productive flow: {successes} successes in sequence, focused on {unique} action types"])

    def _detect_error_cascade(self, now: float) -> None:
        recent_errors = [t for t in self._recent_errors if now - t < 300]
        if len(recent_errors) >= 5:
            intervals = []
            for i in range(1, len(recent_errors)):
                intervals.append(recent_errors[i] - recent_errors[i-1])
            if intervals and all(iv < 30 for iv in intervals):
                self._update_pattern(PatternType.ERROR_STORM, 0.95,
                                     [f"Error cascade: {len(recent_errors)} errors in {intervals[-1]:.0f}s span, avg {sum(intervals)/len(intervals):.1f}s apart"])

    def _detect_time_of_day_pattern(self) -> None:
        import datetime
        hour = datetime.datetime.now().hour
        if 0 <= hour < 6:
            self._update_pattern(PatternType.BREAK_NEEDED, 0.4,
                                [f"Late night session (hour {hour}). Consider offering break."])

    def _update_pattern(self, pattern_type: PatternType, confidence: float,
                        evidence: list[str]) -> None:
        key = pattern_type.value
        now = time.time()
        if key in self._patterns:
            p = self._patterns[key]
            p.confidence = min(1.0, p.confidence + 0.1)
            p.occurrences += 1
            p.last_observed = now
            p.evidence.extend(evidence)
            if len(p.evidence) > 10:
                p.evidence = p.evidence[-10:]
        else:
            if len(self._patterns) >= _MAX_PATTERNS:
                oldest = min(self._patterns.items(), key=lambda x: x[1].last_observed)
                del self._patterns[oldest[0]]
            self._patterns[key] = ObservedPattern(
                pattern_type=pattern_type, confidence=confidence, evidence=evidence)

    def _propose_intervention(self) -> Optional[ProposedIntervention]:
        now = time.time()
        if now - self._last_intervention_time < self._intervention_cooldown:
            return None
        if self._acceptance_rate < 0.2:
            self._intervention_cooldown = min(600.0, self._intervention_cooldown * 1.5)
            return None
        for pattern in sorted(self._patterns.values(), key=lambda p: p.confidence, reverse=True):
            if pattern.confidence < 0.5:
                continue
            if pattern.dismissed_count >= 3:
                continue
            if now - pattern.last_observed > 600:
                continue
            if any(p.pattern == pattern.pattern_type and not p.dismissed for p in self._proposed):
                continue
            intervention_map = {
                PatternType.STUCK_LOOP: ProposedIntervention(
                    action="offer_debug_help", pattern=pattern.pattern_type,
                    timing=InterventionTiming.NEXT_TURN, urgency=InterventionUrgency.HIGH,
                    message="I notice you've been trying the same approach multiple times. Would you like me to suggest an alternative?",
                    reason=f"Stuck loop detected: {pattern.evidence[-1] if pattern.evidence else 'repeated actions'}",
                    confidence=pattern.confidence),
                PatternType.DEBUGGING: ProposedIntervention(
                    action="offer_debug_strategy", pattern=pattern.pattern_type,
                    timing=InterventionTiming.PASSIVE, urgency=InterventionUrgency.MEDIUM,
                    message="I see you're debugging. Want me to analyze the error patterns or suggest a fix?",
                    reason=f"Debugging session detected: {', '.join(pattern.evidence[-2:])}",
                    confidence=pattern.confidence),
                PatternType.ERROR_STORM: ProposedIntervention(
                    action="offer_error_analysis", pattern=pattern.pattern_type,
                    timing=InterventionTiming.IMMEDIATE, urgency=InterventionUrgency.CRITICAL,
                    message="Multiple errors detected. Let me analyze them for common causes.",
                    reason=f"Error storm: {pattern.evidence[-1] if pattern.evidence else 'multiple errors'}",
                    confidence=pattern.confidence),
                PatternType.CONFIGURING: ProposedIntervention(
                    action="offer_config_help", pattern=pattern.pattern_type,
                    timing=InterventionTiming.PASSIVE, urgency=InterventionUrgency.LOW,
                    message="Working with configuration files? I can help validate or find settings.",
                    reason="Configuration pattern detected",
                    confidence=pattern.confidence),
                PatternType.EXPLORING: ProposedIntervention(
                    action="offer_navigation", pattern=pattern.pattern_type,
                    timing=InterventionTiming.PASSIVE, urgency=InterventionUrgency.LOW,
                    message="Looks like you're exploring the codebase. Want me to point out relevant files?",
                    reason="Exploration pattern detected",
                    confidence=pattern.confidence),
                PatternType.CONTEXT_SWITCHING: ProposedIntervention(
                    action="suggest_focus", pattern=pattern.pattern_type,
                    timing=InterventionTiming.NEXT_TURN, urgency=InterventionUrgency.MEDIUM,
                    message="I notice you're switching between different tasks. Would it help to focus on one at a time?",
                    reason="Context switching detected",
                    confidence=pattern.confidence),
                PatternType.PRODUCTIVE_FLOW: ProposedIntervention(
                    action="protect_flow", pattern=pattern.pattern_type,
                    timing=InterventionTiming.PASSIVE, urgency=InterventionUrgency.LOW,
                    message="You're in a productive flow. I'll minimize interruptions.",
                    reason="Flow state detected",
                    confidence=pattern.confidence),
                PatternType.DEEP_WORK: ProposedIntervention(
                    action="protect_deep_work", pattern=pattern.pattern_type,
                    timing=InterventionTiming.PASSIVE, urgency=InterventionUrgency.LOW,
                    message="Deep work in progress. Only interrupting for critical items.",
                    reason="Deep work detected",
                    confidence=pattern.confidence),
                PatternType.LEARNING: ProposedIntervention(
                    action="offer_explanation", pattern=pattern.pattern_type,
                    timing=InterventionTiming.NEXT_TURN, urgency=InterventionUrgency.MEDIUM,
                    message="I can explain this in more detail if you'd like.",
                    reason="Learning pattern detected",
                    confidence=pattern.confidence),
            }
            intervention = intervention_map.get(pattern.pattern_type)
            if intervention:
                self._proposed.append(intervention)
                return intervention
        return None


    def _save_state(self) -> None:
        try:
            self._storage_path.mkdir(parents=True, exist_ok=True)
            data = {
                "acceptance_rate": self._acceptance_rate,
                "intervention_cooldown": self._intervention_cooldown,
                "patterns": {
                    k: {
                        "pattern_type": p.pattern_type.value,
                        "confidence": p.confidence,
                        "evidence": p.evidence[-5:],
                        "occurrences": p.occurrences,
                        "dismissed_count": p.dismissed_count,
                        "last_observed": p.last_observed,
                    }
                    for k, p in self._patterns.items()
                    if p.confidence > 0.3
                },
            }
            (self._storage_path / "state.json").write_text(json.dumps(data, indent=2), encoding='utf-8')
        except Exception as e:
            logger.debug(f"Proactive intelligence save failed: {e}")

    def _load_state(self) -> None:
        try:
            state_file = self._storage_path / "state.json"
            if not state_file.exists():
                return
            data = json.loads(state_file.read_text(encoding='utf-8'))
            self._acceptance_rate = data.get("acceptance_rate", 0.5)
            self._intervention_cooldown = data.get("intervention_cooldown", 120.0)
            if "patterns" in data:
                for k, v in data["patterns"].items():
                    self._patterns[k] = ObservedPattern(
                        pattern_type=PatternType(v["pattern_type"]),
                        confidence=v.get("confidence", 0.5),
                        evidence=v.get("evidence", []),
                        occurrences=v.get("occurrences", 1),
                        dismissed_count=v.get("dismissed_count", 0),
                        last_observed=v.get("last_observed", time.time()),
                    )
            logger.debug(f"Proactive intelligence loaded: {len(self._patterns)} patterns")
        except Exception as e:
            logger.debug(f"Proactive intelligence load failed: {e}")

    def flush(self) -> None:
        with self._lock:
            self._save_state()


def get_proactive_intelligence() -> ProactiveIntelligence:
    global _proactive
    if _proactive is None:
        with _proactive_lock:
            if _proactive is None:
                _proactive = ProactiveIntelligence()
    return _proactive