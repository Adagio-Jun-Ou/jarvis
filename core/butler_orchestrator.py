"""Butler Orchestrator — The conductor that makes all butler systems work as one.

Each butler module is smart on its own. But a real butler operates as a unified
intelligence, not as disconnected specialists. This orchestrator:

1. **Coordinates** — When presence detects "deep work", it tells anticipation to
   suppress low-confidence suggestions, personality to shift to TERSE mode, and
   the observer to log less disruptive observations.

2. **Synthesizes** — Combines signals from all modules into a single coherent
   picture: "User is frustrated (emotional trajectory) + debugging (proactive
   intelligence) + 3 error messages (observer) + terse preference (contextual
   memory) → shift to BUTLER mode, offer targeted help, suppress interruptions"

3. **Prioritizes** — When multiple signals conflict (deep work vs. error storm),
   resolves them: error storms override deep work protection, but only at CRITICAL
   urgency. A casual observation never interrupts focused work.

4. **Learns** — Tracks which orchestrations were accepted vs. dismissed, and adjusts
   the coordination strategy over time. If the user always dismisses break
   reminders, stop offering them.
"""
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

_orchestrator: Optional["ButlerOrchestrator"] = None
_orchestrator_lock = threading.Lock()


class SignalPriority(Enum):
    SUPPRESS = 0
    BACKGROUND = 1
    SUGGEST = 2
    RECOMMEND = 3
    INTERRUPT = 4
    CRITICAL = 5


@dataclass
class OrchestratedAction:
    action: str
    reason: str
    priority: SignalPriority
    source_modules: list[str]
    message: str
    context_modifiers: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    delivered: bool = False
    accepted: bool = False
    dismissed: bool = False


@dataclass
class CoordinationRule:
    name: str
    condition: str
    actions: list[str]
    priority_override: Optional[SignalPriority] = None
    context_overrides: dict = field(default_factory=dict)
    enabled: bool = True


_MAX_ORCHESTRATED = 20
_MAX_RULES = 50
_COORDINATION_INTERVAL = 30.0


class ButlerOrchestrator:
    def __init__(self):
        self._lock = threading.RLock()
        self._actions: deque[OrchestratedAction] = deque(maxlen=_MAX_ORCHESTRATED)
        self._rules: list[CoordinationRule] = []
        self._last_coordination: float = 0.0
        self._acceptance_history: deque[dict] = deque(maxlen=100)
        self._module_states: dict[str, dict] = {}
        self._cross_module_inferences: list[dict] = []
        self._setup_default_rules()

    def _setup_default_rules(self) -> None:
        self._rules = [
            CoordinationRule(
                name="deep_work_protection",
                condition="presence.deep_work AND proactive.priority < CRITICAL",
                actions=["suppress_low_priority", "shift_personality_terse"],
                priority_override=SignalPriority.SUPPRESS,
                context_overrides={"proactivity": 0.2, "verbosity": "terse"},
            ),
            CoordinationRule(
                name="error_storm_override",
                condition="proactive.pattern == error_storm",
                actions=["escalate_to_critical", "shift_personality_butler_concise"],
                priority_override=SignalPriority.CRITICAL,
                context_overrides={"proactivity": 1.0, "assertiveness": 0.9},
            ),
            CoordinationRule(
                name="frustration_care",
                condition="observer.emotional_tone == frustrated AND proactive.stuck_loop",
                actions=["shift_personality_companion", "increase_patience", "offer_debug_help"],
                context_overrides={"warmth": 0.9, "patience": 1.0, "proactivity": 0.8},
            ),
            CoordinationRule(
                name="productive_flow_protection",
                condition="proactive.pattern == productive_flow AND anticipation.confidence < 0.7",
                actions=["suppress_suggestions", "minimize interruptions"],
                priority_override=SignalPriority.SUPPRESS,
                context_overrides={"proactivity": 0.1},
            ),
            CoordinationRule(
                name="curious_learner",
                condition="observer.emotional_tone == curious AND personality.mode != mentor",
                actions=["shift_personality_mentor", "increase_verbosity"],
                context_overrides={"mode": "mentor", "verbosity": "detailed", "patience": 1.0},
            ),
            CoordinationRule(
                name="session_start_greeting",
                condition="presence.just_returned AND contextual.session_count > 3",
                actions=["personalized_greeting", "resume_context"],
                priority_override=SignalPriority.RECOMMEND,
                context_overrides={"warmth": 0.8},
            ),
            CoordinationRule(
                name="break_needed",
                condition="presence.session_minutes > 90 AND presence.focus_score > 0.6",
                actions=["suggest_break", "reduce_intensity"],
                priority_override=SignalPriority.SUGGEST,
            ),
            CoordinationRule(
                name="context_switch_fatigue",
                condition="proactive.pattern == context_switching AND presence.distraction_count > 2",
                actions=["suggest_focus", "reduce_multitasking"],
                context_overrides={"assertiveness": 0.7},
            ),
        ]

    def coordinate(self, presence_state: dict = None,
                   anticipation_state: dict = None,
                   observer_state: dict = None,
                   personality_state: dict = None,
                   proactive_state: dict = None,
                   contextual_state: dict = None) -> Optional[OrchestratedAction]:
        with self._lock:
            now = time.time()
            if now - self._last_coordination < _COORDINATION_INTERVAL:
                return None
            self._last_coordination = now
            self._module_states = {
                "presence": presence_state or {},
                "anticipation": anticipation_state or {},
                "observer": observer_state or {},
                "personality": personality_state or {},
                "proactive": proactive_state or {},
                "contextual": contextual_state or {},
            }
            self._cross_module_inferences = self._infer_cross_module()
            action = self._evaluate_rules()
            return action

    def get_coordination_context(self) -> str:
        with self._lock:
            parts = []
            if self._cross_module_inferences:
                for inf in self._cross_module_inferences[:3]:
                    parts.append(f"{inf['inference']}: {inf['evidence']}")
            pending = [a for a in self._actions if not a.accepted and not a.dismissed and not a.delivered]
            if pending:
                top = pending[0]
                parts.append(f"Recommended: {top.message}")
            acceptance_rate = self._get_acceptance_rate()
            if acceptance_rate < 0.3:
                parts.append("User tends to dismiss suggestions — be restrained")
            elif acceptance_rate > 0.7:
                parts.append("User values proactive suggestions — be helpful")
            return " | ".join(parts) if parts else ""

    def get_pending_interruption(self) -> str | None:
        """Return an active interruption message for INTERRUPT/CRITICAL priority actions.

        Returns the message and marks the action as delivered.
        Returns None if no high-priority undelivered actions exist.
        """
        with self._lock:
            high_priority = [
                a for a in self._actions
                if not a.delivered
                and a.priority.value >= SignalPriority.INTERRUPT.value
            ]
            if not high_priority:
                return None
            high_priority.sort(key=lambda a: a.priority.value, reverse=True)
            action = high_priority[0]
            action.delivered = True
            prefix = "🚨 CRITICAL" if action.priority == SignalPriority.CRITICAL else "⚡"
            return f"{prefix} {action.message} (reason: {action.reason})"

    def mark_accepted(self, action: str = "") -> None:
        with self._lock:
            for a in reversed(list(self._actions)):
                if a.delivered and not a.accepted and not a.dismissed:
                    a.accepted = True
                    self._acceptance_history.append({"action": a.action, "accepted": True, "time": time.time(), "rule": a.reason})
                    self._adjust_rule_on_acceptance(a.reason, True)
                    break

    def mark_dismissed(self, action: str = "") -> None:
        with self._lock:
            for a in reversed(list(self._actions)):
                if a.delivered and not a.accepted and not a.dismissed:
                    a.dismissed = True
                    self._acceptance_history.append({"action": a.action, "accepted": False, "time": time.time(), "rule": a.reason})
                    self._adjust_rule_on_acceptance(a.reason, False)
                    break

    def get_state(self) -> dict:
        with self._lock:
            rule_stats = {}
            for rule in self._rules:
                acceptances = sum(1 for h in self._acceptance_history if h.get("rule") == rule.name and h.get("accepted"))
                dismissals = sum(1 for h in self._acceptance_history if h.get("rule") == rule.name and not h.get("accepted"))
                rule_stats[rule.name] = {"enabled": rule.enabled, "acceptances": acceptances, "dismissals": dismissals}
            return {
                "actions_proposed": len(self._actions),
                "cross_module_inferences": len(self._cross_module_inferences),
                "acceptance_rate": round(self._get_acceptance_rate(), 2),
                "rules_active": len([r for r in self._rules if r.enabled]),
                "rules_total": len(self._rules),
                "rule_stats": rule_stats,
                "last_coordination": round(time.time() - self._last_coordination, 1),
                "pending_actions": len([a for a in self._actions if not a.accepted and not a.dismissed]),
            }

    def _adjust_rule_on_acceptance(self, rule_name: str, accepted: bool) -> None:
        for rule in self._rules:
            if rule.name == rule_name:
                if accepted:
                    if rule.priority_override and rule.priority_override.value < SignalPriority.RECOMMEND.value:
                        rule.priority_override = SignalPriority(
                            min(rule.priority_override.value + 1, SignalPriority.INTERRUPT.value)
                        )
                else:
                    dismissals = sum(
                        1 for h in self._acceptance_history
                        if h.get("rule") == rule_name and not h.get("accepted")
                        and time.time() - h.get("time", 0) < 3600
                    )
                    if dismissals >= 3:
                        rule.enabled = False
                    elif rule.priority_override and rule.priority_override.value > SignalPriority.SUPPRESS.value:
                        rule.priority_override = SignalPriority(
                            max(rule.priority_override.value - 1, SignalPriority.SUPPRESS.value)
                        )
                break

    def _infer_cross_module(self) -> list[dict]:
        inferences = []
        pres = self._module_states.get("presence", {})
        obs = self._module_states.get("observer", {})
        pro = self._module_states.get("proactive", {})
        ctx = self._module_states.get("contextual", {})
        if pres.get("presence") == "deep_work" and pro.get("active_patterns", 0) > 0:
            inferences.append({
                "inference": "Deep work with active patterns",
                "evidence": f"Presence: deep work, {pro.get('active_patterns', 0)} patterns active",
                "action": "suppress_non_critical",
            })
        if obs.get("emotional_tone") in ("frustrated", "angry") and pro.get("recent_errors", 0) >= 2:
            inferences.append({
                "inference": "Frustrated debugging",
                "evidence": f"Emotional tone: {obs.get('emotional_tone')}, {pro.get('recent_errors', 0)} recent errors",
                "action": "offer_targeted_help",
            })
        if ctx.get("primary_topics"):
            topics = ctx.get("primary_topics", {})
            if isinstance(topics, dict):
                top_topic = max(topics, key=topics.get) if topics else None
                if top_topic:
                    inferences.append({
                        "inference": f"User focuses on {top_topic}",
                        "evidence": f"Primary topic: {top_topic} ({topics[top_topic]} interactions)",
                        "action": "tailor_to_topic",
                    })
        if pres.get("session_minutes", 0) > 120 and pres.get("focus_score", 0) > 0.7:
            inferences.append({
                "inference": "Long focused session — may need break",
                "evidence": f"Session: {pres.get('session_minutes', 0):.0f}min, focus: {pres.get('focus_score', 0):.2f}",
                "action": "suggest_break_gently",
            })
        return inferences

    def _evaluate_rules(self) -> Optional[OrchestratedAction]:
        pres = self._module_states.get("presence", {})
        obs = self._module_states.get("observer", {})
        pro = self._module_states.get("proactive", {})
        ctx = self._module_states.get("contextual", {})
        for rule in self._rules:
            if not rule.enabled:
                continue
            cond = rule.condition.lower()
            match = False
            if "deep_work" in cond and pres.get("presence") == "deep_work":
                match = True
            elif "error_storm" in cond and pro.get("active_patterns", 0) > 0:
                patterns = pro.get("patterns", [])
                if any("error_storm" in str(p).lower() for p in patterns):
                    match = True
            elif "frustrated" in cond and obs.get("emotional_tone") in ("frustrated", "angry"):
                match = True
            elif "productive_flow" in cond and pro.get("active_patterns", 0) > 0:
                patterns = pro.get("patterns", [])
                if any("productive_flow" in str(p).lower() for p in patterns):
                    match = True
            elif "curious" in cond and obs.get("emotional_tone") in ("curious", "inquiring"):
                match = True
            elif "just_returned" in cond and pres.get("last_transition") == "returned":
                match = True
            elif "session_minutes > 90" in cond and pres.get("session_minutes", 0) > 90:
                match = True
            elif "context_switch" in cond and pro.get("active_patterns", 0) > 0:
                patterns = pro.get("patterns", [])
                if any("context_switch" in str(p).lower() for p in patterns):
                    match = True
            if match:
                action = OrchestratedAction(
                    action=rule.actions[0],
                    reason=rule.name,
                    priority=rule.priority_override or SignalPriority.SUGGEST,
                    source_modules=["orchestrator"],
                    message=self._generate_message(rule, pres, obs, pro, ctx),
                    context_modifiers=rule.context_overrides,
                )
                self._actions.append(action)
                return action
        return None

    def _generate_message(self, rule: CoordinationRule, pres: dict, obs: dict,
                         pro: dict, ctx: dict) -> str:
        if rule.name == "deep_work_protection":
            return "You're in deep work mode. I'll keep interruptions minimal."
        elif rule.name == "error_storm_override":
            return f"I notice multiple errors. Let me analyze the common cause — {', '.join(str(p) for p in pro.get('patterns', [])[:2])}."
        elif rule.name == "frustration_care":
            return "I can see this is frustrating. Let me focus on solving the core issue — no preamble needed."
        elif rule.name == "productive_flow_protection":
            return "Productive flow detected. Only critical items will surface."
        elif rule.name == "curious_learner":
            return "Great question! Let me explain this thoroughly."
        elif rule.name == "session_start_greeting":
            return f"Welcome back. Picking up where we left off."
        elif rule.name == "break_needed":
            mins = pres.get("session_minutes", 0)
            return f"You've been working for {mins:.0f} minutes with high focus. Consider a short break."
        elif rule.name == "context_switch_fatigue":
            return "I notice you're switching between many tasks. Want help focusing on one?"
        return rule.name.replace("_", " ").title()

    def _get_acceptance_rate(self) -> float:
        if not self._acceptance_history:
            return 0.5
        recent = list(self._acceptance_history)[-20:]
        accepted = sum(1 for h in recent if h.get("accepted"))
        return accepted / max(len(recent), 1)


def get_butler_orchestrator() -> ButlerOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        with _orchestrator_lock:
            if _orchestrator is None:
                _orchestrator = ButlerOrchestrator()
    return _orchestrator