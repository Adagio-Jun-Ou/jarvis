"""Butler Session — Lifecycle management for butler intelligence.

A butler doesn't just react — it prepares. When a session starts, it greets
appropriately. When a session ends, it saves state. When the user returns,
it remembers what happened last time.

This module handles:
1. **Session start** — Time-aware greeting, context resumption suggestions
2. **Session end** — Graceful persistence, session summary, state decay
3. **Idle detection** — Suggest break after long focus, detect departure
4. **Return greeting** — Personalized "welcome back" with context
5. **Memory decay** — Older interactions fade, recent ones stay vivid
"""
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from core.butler_personality import ButlerPersonality, PersonaMode, get_butler_personality
from core.proactive_intelligence import ProactiveIntelligence, get_proactive_intelligence
from core.contextual_memory import ContextualMemory, get_contextual_memory

logger = logging.getLogger(__name__)

_session: Optional["ButlerSession"] = None
_session_lock = threading.Lock()

_IDLE_WARNING_THRESHOLD = 2700.0  # 45 minutes
_BREAK_SUGGEST_THRESHOLD = 5400.0  # 90 minutes
_DECAY_HALF_LIFE = 86400.0  # 24 hours


@dataclass
class SessionEvent:
    timestamp: float
    event_type: str
    details: str = ""
    sentiment: str = "neutral"


@dataclass
class SessionStats:
    start_time: float = 0.0
    end_time: float = 0.0
    turn_count: int = 0
    errors_encountered: int = 0
    tools_used: int = 0
    topics: list[str] = field(default_factory=list)
    mode_changes: int = 0
    peak_mode: str = "butler"
    peak_mode_duration: float = 0.0
    was_productive: bool = False


_MAX_EVENTS = 200
_MAX_STATS_HISTORY = 50


class ButlerSession:
    def __init__(self):
        self._lock = threading.RLock()
        self._current_session: Optional[SessionStats] = None
        self._session_history: deque[SessionStats] = deque(maxlen=_MAX_STATS_HISTORY)
        self._events: deque[SessionEvent] = deque(maxlen=_MAX_EVENTS)
        self._last_activity: float = time.time()
        self._last_idle_warning: float = 0.0
        self._last_break_suggestion: float = 0.0
        self._mode_durations: dict[str, float] = {}
        self._session_count: int = 0
        self._initialized: bool = False

    def start_session(self, user_message: str = "") -> str:
        with self._lock:
            if self._current_session is not None:
                self.end_session(reason="new_session")
            self._current_session = SessionStats(start_time=time.time())
            self._session_count += 1
            self._last_activity = time.time()
            self._initialized = True
            greeting = self._generate_greeting(user_message)
            self._events.append(SessionEvent(
                timestamp=time.time(), event_type="session_start",
                details=greeting, sentiment="neutral"))
            return greeting

    def end_session(self, reason: str = "user_departed") -> None:
        with self._lock:
            if self._current_session is None:
                return
            self._current_session.end_time = time.time()
            self._current_session.was_productive = (
                self._current_session.turn_count > 5
                and self._current_session.errors_encountered < 3
            )
            self._session_history.append(self._current_session)
            self._events.append(SessionEvent(
                timestamp=time.time(), event_type="session_end",
                details=reason))
            try:
                bp = get_butler_personality()
                bp.flush()
            except Exception:
                pass
            try:
                pi = get_proactive_intelligence()
                pi.flush()
            except Exception:
                pass
            self._current_session = None

    def record_turn(self, user_message: str = "", sentiment: str = "neutral") -> None:
        with self._lock:
            if self._current_session is None:
                self._current_session = SessionStats(start_time=time.time())
                self._session_count += 1
            self._current_session.turn_count += 1
            self._last_activity = time.time()
            if user_message:
                words = user_message.lower().split()[:5]
                for w in words:
                    if len(w) > 3 and w not in self._current_session.topics[:10]:
                        self._current_session.topics.append(w)
                        if len(self._current_session.topics) > 20:
                            self._current_session.topics = self._current_session.topics[-20:]
            self._events.append(SessionEvent(
                timestamp=time.time(), event_type="turn",
                details=user_message[:100] if user_message else "",
                sentiment=sentiment))

    def record_error(self) -> None:
        with self._lock:
            if self._current_session:
                self._current_session.errors_encountered += 1

    def record_tool_use(self) -> None:
        with self._lock:
            if self._current_session:
                self._current_session.tools_used += 1

    def record_mode_change(self, mode: str) -> None:
        with self._lock:
            if self._current_session:
                self._current_session.mode_changes += 1
                self._mode_durations[mode] = self._mode_durations.get(mode, 0) + 1
                self._current_session.peak_mode = max(
                    self._mode_durations, key=self._mode_durations.get, default="butler")

    def check_idle(self, current_time: float = None) -> Optional[str]:
        with self._lock:
            if self._current_session is None:
                return None
            now = current_time or time.time()
            idle = now - self._last_activity
            session_duration = now - self._current_session.start_time
            if idle > _IDLE_WARNING_THRESHOLD and now - self._last_idle_warning > 600:
                self._last_idle_warning = now
                idle_min = int(idle / 60)
                return f"You've been idle for {idle_min} minutes. Still here?"
            if session_duration > _BREAK_SUGGEST_THRESHOLD and now - self._last_break_suggestion > 1800:
                self._last_break_suggestion = now
                session_min = int(session_duration / 60)
                return f"You've been working for {session_min} minutes. Consider a short break."
            return None

    def get_resume_context(self) -> str:
        with self._lock:
            if not self._session_history:
                return ""
            last = self._session_history[-1]
            duration_min = int((last.end_time - last.start_time) / 60) if last.end_time else 0
            parts = []
            if duration_min > 0:
                parts.append(f"Last session was {duration_min} minutes")
            if last.topics:
                parts.append(f"Topics: {', '.join(last.topics[:5])}")
            if last.errors_encountered > 0:
                parts.append(f"Hit {last.errors_encountered} errors")
            if last.was_productive:
                parts.append("productive session")
            if last.mode_changes > 3:
                parts.append("switched between modes")
            return " | ".join(parts) if parts else ""

    def get_state(self) -> dict:
        with self._lock:
            session = self._current_session
            return {
                "in_session": session is not None,
                "session_count": self._session_count,
                "turn_count": session.turn_count if session else 0,
                "errors": session.errors_encountered if session else 0,
                "tools_used": session.tools_used if session else 0,
                "session_duration": round(time.time() - session.start_time, 0) if session else 0,
                "idle_seconds": round(time.time() - self._last_activity, 0),
                "topics": session.topics[:5] if session else [],
                "peak_mode": session.peak_mode if session else "butler",
                "events_tracked": len(self._events),
                "session_history_count": len(self._session_history),
            }

    def apply_memory_decay(self) -> None:
        with self._lock:
            now = time.time()
            events_to_keep = []
            for event in self._events:
                age = now - event.timestamp
                if age < _DECAY_HALF_LIFE * 3:
                    events_to_keep.append(event)
            self._events = deque(events_to_keep, maxlen=_MAX_EVENTS)

    def _generate_greeting(self, user_message: str = "") -> str:
        import datetime
        hour = datetime.datetime.now().hour
        bp = get_butler_personality()
        state = bp.get_state()
        formality = state.get("formality", "professional")
        if self._session_count <= 1:
            from core.butler_voice import get_butler_voice, Situation
            voice = get_butler_voice()
            return voice.get_greeting(bp, Situation.GREETING_FIRST)
        if hour < 5 or hour >= 22:
            from core.butler_voice import get_butler_voice, Situation
            voice = get_butler_voice()
            return voice.get_greeting(bp, Situation.GREETING_EVENING)
        resume = self.get_resume_context()
        if resume and formality in ("formal", "professional"):
            return f"Welcome back. {resume}. How shall we proceed?"
        elif resume:
            return f"Hey, welcome back! {resume}. What's up?"
        from core.butler_voice import get_butler_voice, Situation
        voice = get_butler_voice()
        return voice.get_greeting(bp, Situation.GREETING_RETURN)


def get_butler_session() -> ButlerSession:
    global _session
    if _session is None:
        with _session_lock:
            if _session is None:
                _session = ButlerSession()
    return _session