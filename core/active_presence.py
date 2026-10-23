"""Active Presence Engine - Jarvis's ambient awareness and autonomous presence.

An AI butler doesn't just respond when spoken to — it maintains an active presence,
noticing when you arrive, when you're focused, when you step away, and when you
might need something before you ask.

Key capabilities:
1. User Presence Detection — tracks active/away/focused/distracted patterns
2. Ambient Observations — periodic scan for things worth noting
3. Proactive Check-ins — gentle interruption when something needs attention
4. Session Rhythm — learns user's work patterns for better timing
5. Environmental Awareness — monitors system state for user-relevant changes
"""
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

_presence: Optional["ActivePresenceEngine"] = None
_presence_lock = threading.Lock()


class UserPresence(Enum):
    ACTIVE = "active"
    TYPING = "typing"
    IDLE = "idle"
    AWAY = "away"
    FOCUSED = "focused"
    DISTRACTED = "distracted"
    RETURNED = "returned"
    DEEP_WORK = "deep_work"
    BREAK = "break"


class PresenceTransition(Enum):
    ARRIVED = "arrived"
    LEFT = "left"
    FOCUSED = "focused"
    UNFOCUSED = "unfocused"
    RETURNED = "returned"
    TYPING_STARTED = "typing_started"
    TYPING_STOPPED = "typing_stopped"
    DEEP_WORK_STARTED = "deep_work_started"
    DEEP_WORK_ENDED = "deep_work_ended"
    BREAK_STARTED = "break_started"
    BREAK_ENDED = "break_ended"
    DISTRACTED = "distracted"


_DEEP_WORK_THRESHOLD = 600.0
_DEEP_WORK_MESSAGE_GAP = 30.0
_BREAK_REMINDER_INTERVAL = 2700.0
_CHECKIN_COOLDOWN = 120.0


@dataclass
class AmbientObservation:
    category: str
    content: str
    relevance: float
    timestamp: float = field(default_factory=time.time)
    acknowledged: bool = False


@dataclass
class CheckIn:
    reason: str
    message: str
    urgency: str
    timestamp: float = field(default_factory=time.time)
    delivered: bool = False
    dismissed: bool = False


@dataclass
class SessionRhythm:
    peak_hours: list[int] = field(default_factory=list)
    typical_session_minutes: float = 0.0
    average_gap_minutes: float = 0.0
    total_sessions: int = 0
    focus_periods: int = 0
    deep_work_periods: int = 0
    breaks_taken: int = 0
    longest_focus_minutes: float = 0.0


@dataclass
class PresenceState:
    presence: UserPresence = UserPresence.IDLE
    idle_seconds: float = 0.0
    last_activity: float = 0.0
    last_message_time: float = 0.0
    consecutive_messages: int = 0
    focus_score: float = 0.0
    session_start: float = 0.0
    presence_transitions: list[PresenceTransition] = field(default_factory=list)
    observations: list[AmbientObservation] = field(default_factory=list)
    pending_checkins: list[CheckIn] = field(default_factory=list)
    rhythm: SessionRhythm = field(default_factory=SessionRhythm)
    return_count: int = 0
    total_idle_periods: int = 0
    deep_work_start: float = 0.0
    last_break_time: float = 0.0
    last_checkin_time: float = 0.0
    distraction_count: int = 0
    rapid_context_switches: int = 0


_IDLE_AWAY_THRESHOLD = 300.0
_IDLE_FOCUS_THRESHOLD = 60.0
_TYPING_TIMEOUT = 15.0
_RAPID_MESSAGES_FOR_FOCUS = 3
_RAPID_MESSAGES_WINDOW = 120.0
_MAX_OBSERVATIONS = 20
_MAX_CHECKINS = 10
_RHYTHM_WINDOW = 50


class ActivePresenceEngine:
    def __init__(self):
        self._lock = threading.RLock()
        self._state = PresenceState(last_activity=time.time(), session_start=time.time())
        self._message_times: deque[float] = deque(maxlen=100)
        self._session_starts: deque[float] = deque(maxlen=_RHYTHM_WINDOW)
        self._gap_times: deque[float] = deque(maxlen=_RHYTHM_WINDOW)
        self._focus_durations: deque[float] = deque(maxlen=_RHYTHM_WINDOW)
        self._last_transition_time: float = time.time()
        self._typical_session_duration: float = 0.0
        self._activity_types: deque[str] = deque(maxlen=30)
        self._last_activity_type: str = ""

    def record_user_activity(self, activity_type: str = "message") -> Optional[PresenceTransition]:
        with self._lock:
            now = time.time()
            old_presence = self._state.presence
            self._state.last_activity = now
            self._state.idle_seconds = 0.0
            self._message_times.append(now)
            self._state.consecutive_messages += 1

            if activity_type != self._last_activity_type:
                if self._last_activity_type:
                    self._activity_types.append(self._last_activity_type)
                self._last_activity_type = activity_type

            transition = None

            if old_presence in (UserPresence.AWAY, UserPresence.IDLE):
                self._state.return_count += 1
                was_away = old_presence == UserPresence.AWAY
                transition = PresenceTransition.RETURNED if was_away else PresenceTransition.ARRIVED
                self._session_starts.append(now)
                self._state.session_start = now

            if activity_type == "typing":
                self._state.presence = UserPresence.TYPING
                self._state.last_message_time = now
                if old_presence != UserPresence.TYPING:
                    transition = PresenceTransition.TYPING_STARTED

            if self._is_focused():
                if old_presence != UserPresence.FOCUSED and old_presence != UserPresence.DEEP_WORK:
                    transition = PresenceTransition.FOCUSED
                self._state.presence = UserPresence.FOCUSED
                self._state.focus_score = min(1.0, self._state.focus_score + 0.1)
                if self._state.deep_work_start == 0.0:
                    self._state.deep_work_start = now
            elif self._state.presence == UserPresence.TYPING:
                pass
            else:
                self._state.presence = UserPresence.ACTIVE
                if old_presence == UserPresence.FOCUSED:
                    transition = PresenceTransition.UNFOCUSED
                    if self._state.deep_work_start > 0:
                        duration = now - self._state.deep_work_start
                        self._focus_durations.append(duration / 60.0)
                        self._state.rhythm.focus_periods += 1
                        if duration / 60.0 > self._state.rhythm.longest_focus_minutes:
                            self._state.rhythm.longest_focus_minutes = round(duration / 60.0, 1)
                        self._state.deep_work_start = 0.0
                self._state.focus_score = max(0.0, self._state.focus_score - 0.05)

            if transition:
                self._state.presence_transitions.append(transition)
                if len(self._state.presence_transitions) > 20:
                    self._state.presence_transitions = self._state.presence_transitions[-20:]

            return transition

    def tick(self) -> dict:
        with self._lock:
            now = time.time()
            idle = now - self._state.last_activity
            self._state.idle_seconds = idle
            transition = None

            if idle > _IDLE_AWAY_THRESHOLD and self._state.presence != UserPresence.AWAY:
                transition = PresenceTransition.LEFT
                self._state.presence = UserPresence.AWAY
                self._state.total_idle_periods += 1
                gap = now - self._state.session_start
                if gap > 60:
                    self._gap_times.append(gap / 60.0)
                self._state.focus_score = 0.0
            elif idle > _IDLE_FOCUS_THRESHOLD and self._state.presence == UserPresence.FOCUSED:
                transition = PresenceTransition.UNFOCUSED
                self._state.presence = UserPresence.IDLE
                self._state.focus_score = max(0.0, self._state.focus_score - 0.2)
                if self._state.deep_work_start > 0:
                    duration = now - self._state.deep_work_start
                    self._focus_durations.append(duration / 60.0)
                    self._state.rhythm.focus_periods += 1
                    if duration / 60.0 > self._state.rhythm.longest_focus_minutes:
                        self._state.rhythm.longest_focus_minutes = round(duration / 60.0, 1)
                    self._state.deep_work_start = 0.0
            elif idle > _TYPING_TIMEOUT and self._state.presence == UserPresence.TYPING:
                transition = PresenceTransition.TYPING_STOPPED
                self._state.presence = UserPresence.ACTIVE
            elif idle > _IDLE_FOCUS_THRESHOLD and self._state.presence == UserPresence.ACTIVE:
                self._state.presence = UserPresence.IDLE
                self._state.focus_score = max(0.0, self._state.focus_score - 0.1)

            if self._state.deep_work_start > 0 and now - self._state.deep_work_start > _DEEP_WORK_THRESHOLD:
                if self._state.presence != UserPresence.DEEP_WORK:
                    transition = PresenceTransition.DEEP_WORK_STARTED
                    self._state.presence = UserPresence.DEEP_WORK
                    self._state.rhythm.deep_work_periods += 1

            if now - self._state.last_break_time > _BREAK_REMINDER_INTERVAL and self._state.presence not in (UserPresence.AWAY, UserPresence.BREAK):
                if self._state.focus_score > 0.5 and self._state.last_break_time > 0:
                    checkin = CheckIn(
                        reason="break_reminder",
                        message="You've been working for a while. Consider taking a short break.",
                        urgency="low",
                    )
                    self._state.pending_checkins.append(checkin)

            if len(self._activity_types) >= 5:
                recent_types = list(self._activity_types)[-5:]
                unique_types = len(set(recent_types))
                if unique_types >= 4 and self._state.presence == UserPresence.ACTIVE:
                    self._state.distraction_count += 1
                    if self._state.distraction_count >= 3:
                        transition = PresenceTransition.DISTRACTED
                        self._state.presence = UserPresence.DISTRACTED
                        self._state.focus_score = max(0.0, self._state.focus_score - 0.3)

            if transition:
                self._state.presence_transitions.append(transition)
                if len(self._state.presence_transitions) > 20:
                    self._state.presence_transitions = self._state.presence_transitions[-20:]

            self._update_rhythm()

            return {
                "presence": self._state.presence.value,
                "idle_seconds": round(idle, 1),
                "focus_score": round(self._state.focus_score, 2),
                "session_minutes": round((now - self._state.session_start) / 60, 1),
                "transition": transition.value if transition else None,
                "observations": len(self._state.observations),
                "pending_checkins": len(self._state.pending_checkins),
            }

    def add_observation(self, category: str, content: str, relevance: float = 0.5) -> AmbientObservation:
        with self._lock:
            obs = AmbientObservation(category=category, content=content, relevance=max(0.0, min(1.0, relevance)))
            self._state.observations.append(obs)
            if len(self._state.observations) > _MAX_OBSERVATIONS:
                self._state.observations = self._state.observations[-_MAX_OBSERVATIONS:]
            return obs

    def get_unacknowledged_observations(self) -> list[dict]:
        with self._lock:
            return [
                {"category": o.category, "content": o.content, "relevance": o.relevance, "age_seconds": round(time.time() - o.timestamp, 0)}
                for o in self._state.observations if not o.acknowledged
            ]

    def acknowledge_observation(self, index: int) -> bool:
        with self._lock:
            unack = [o for o in self._state.observations if not o.acknowledged]
            if 0 <= index < len(unack):
                unack[index].acknowledged = True
                return True
            return False

    def should_check_in(self) -> Optional[CheckIn]:
        with self._lock:
            now = time.time()
            if self._state.presence == UserPresence.AWAY:
                return None
            if now - self._state.last_checkin_time < _CHECKIN_COOLDOWN:
                return None
            for ci in self._state.pending_checkins:
                if not ci.delivered and not ci.dismissed:
                    self._state.last_checkin_time = now
                    return ci
            if self._state.presence == UserPresence.RETURNED:
                idle_duration = now - self._state.last_activity
                if idle_duration < 10:
                    ci = CheckIn(
                        reason="user_returned",
                        message="Welcome back! Anything I can help with?",
                        urgency="low",
                    )
                    self._state.pending_checkins.append(ci)
                    if len(self._state.pending_checkins) > _MAX_CHECKINS:
                        self._state.pending_checkins = self._state.pending_checkins[-_MAX_CHECKINS:]
                    return ci
            high_relevance = [o for o in self._state.observations if not o.acknowledged and o.relevance > 0.7]
            if high_relevance and self._state.presence in (UserPresence.ACTIVE, UserPresence.IDLE):
                obs = high_relevance[0]
                ci = CheckIn(
                    reason="observation",
                    message=f"FYI: {obs.content}",
                    urgency="medium" if obs.relevance > 0.85 else "low",
                )
                self._state.pending_checkins.append(ci)
                if len(self._state.pending_checkins) > _MAX_CHECKINS:
                    self._state.pending_checkins = self._state.pending_checkins[-_MAX_CHECKINS:]
                return ci
            return None

    def dismiss_checkin(self, index: int = -1) -> bool:
        with self._lock:
            active = [ci for ci in self._state.pending_checkins if not ci.dismissed]
            if active:
                active[index if -1 <= index < len(active) else 0].dismissed = True
                return True
            return False

    def get_presence_context(self) -> str:
        with self._lock:
            parts = []
            p = self._state.presence.value
            parts.append(f"User presence: {p}")
            if self._state.presence == UserPresence.DEEP_WORK:
                deep_work_duration = time.time() - self._state.deep_work_start
                parts.append(f"In deep work for {deep_work_duration / 60:.0f} minutes — DO NOT INTERRUPT unless urgent")
            elif self._state.focus_score > 0.6:
                parts.append("User appears focused — minimize interruptions")
            elif self._state.presence == UserPresence.AWAY:
                parts.append("User is away — defer non-urgent notifications")
            elif self._state.presence == UserPresence.RETURNED:
                parts.append("User just returned — brief welcome appropriate")
            elif self._state.presence == UserPresence.IDLE:
                parts.append("User is idle — good time for background tasks")
            elif self._state.presence == UserPresence.DISTRACTED:
                parts.append(f"User is distracted ({self._state.distraction_count} switches) — gentle guidance helpful")
            elif self._state.presence == UserPresence.BREAK:
                parts.append("User is on break — hold notifications")
            if self._state.consecutive_messages >= 3:
                parts.append("Active conversation flow — maintain engagement")
            if self._state.idle_seconds > 60:
                parts.append(f"Idle for {self._state.idle_seconds:.0f}s")
            if self._state.rhythm.deep_work_periods > 0:
                parts.append(f"Deep work periods: {self._state.rhythm.deep_work_periods}")
            if self._state.rhythm.longest_focus_minutes > 0:
                parts.append(f"Longest focus: {self._state.rhythm.longest_focus_minutes:.0f}min")
            unack = [o for o in self._state.observations if not o.acknowledged]
            if unack:
                parts.append(f"{len(unack)} unacknowledged observation(s)")
            return " | ".join(parts)

    def get_state(self) -> dict:
        with self._lock:
            return {
                "presence": self._state.presence.value,
                "idle_seconds": round(self._state.idle_seconds, 1),
                "focus_score": round(self._state.focus_score, 2),
                "session_minutes": round((time.time() - self._state.session_start) / 60, 1),
                "return_count": self._state.return_count,
                "consecutive_messages": self._state.consecutive_messages,
                "deep_work_minutes": round((time.time() - self._state.deep_work_start) / 60, 1) if self._state.deep_work_start > 0 else 0,
                "distraction_count": self._state.distraction_count,
                "observations": len(self._state.observations),
                "pending_checkins": len(self._state.pending_checkins),
                "rhythm": {
                    "peak_hours": self._state.rhythm.peak_hours,
                    "typical_session_minutes": round(self._state.rhythm.typical_session_minutes, 1),
                    "total_sessions": self._state.rhythm.total_sessions,
                    "deep_work_periods": self._state.rhythm.deep_work_periods,
                    "breaks_taken": self._state.rhythm.breaks_taken,
                    "longest_focus_minutes": self._state.rhythm.longest_focus_minutes,
                },
                "last_transition": self._state.presence_transitions[-1].value if self._state.presence_transitions else None,
            }

    def _is_focused(self) -> bool:
        now = time.time()
        recent = [t for t in self._message_times if now - t < _RAPID_MESSAGES_WINDOW]
        if len(recent) >= _RAPID_MESSAGES_FOR_FOCUS:
            return True
        return False

    def _update_rhythm(self):
        if self._session_starts:
            self._state.rhythm.total_sessions = len(self._session_starts)
            hours = [time.localtime(t).tm_hour for t in self._session_starts]
            from collections import Counter
            hour_counts = Counter(hours)
            self._state.rhythm.peak_hours = [h for h, c in hour_counts.most_common(3)]
        if self._gap_times:
            avg = sum(self._gap_times) / len(self._gap_times)
            self._state.rhythm.average_gap_minutes = round(avg, 1)
            self._state.rhythm.typical_session_minutes = round(avg, 1)


def get_active_presence() -> ActivePresenceEngine:
    global _presence
    if _presence is None:
        with _presence_lock:
            if _presence is None:
                _presence = ActivePresenceEngine()
    return _presence