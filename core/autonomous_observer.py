"""Autonomous Observation Engine - Jarvis watches, learns, and understands.

An AI butler maintains awareness of its environment and the user's habits
without being intrusive. This engine continuously observes:

1. **System Environment** — resource changes, service status, file modifications
2. **User Habits** — work patterns, common requests, temporal preferences
3. **Conversation Context** — topic arcs, emotional trajectory, engagement level
4. **Action Outcomes** — what worked, what failed, what the user appreciated

All observations feed into the anticipation and presence engines so Jarvis
can be genuinely helpful rather than merely reactive.
"""
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

_observer: Optional["AutonomousObserver"] = None
_observer_lock = threading.Lock()


class ObservationType(Enum):
    SYSTEM = "system"
    USER_HABIT = "user_habit"
    CONVERSATION = "conversation"
    ACTION_OUTCOME = "action_outcome"
    ENVIRONMENT = "environment"
    TEMPORAL = "temporal"
    NETWORK = "network"
    FILE_CHANGE = "file_change"
    EMOTIONAL = "emotional"
    ANOMALY = "anomaly"


class ObservationSalience(Enum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class Observation:
    obs_type: ObservationType
    category: str
    content: str
    salience: ObservationSalience
    context: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    notifiable: bool = False
    notified: bool = False
    expired: bool = False
    ttl_seconds: float = 3600.0


@dataclass
class UserHabit:
    habit_type: str
    description: str
    frequency: int = 1
    last_seen: float = field(default_factory=time.time)
    confidence: float = 0.5
    time_of_day: list[int] = field(default_factory=list)
    context_triggers: list[str] = field(default_factory=list)


@dataclass
class EnvironmentSnapshot:
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    disk_percent: float = 0.0
    active_services: list[str] = field(default_factory=list)
    modified_files: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


_MAX_OBSERVATIONS = 200
_MAX_HABITS = 50
_MAX_EMOTIONAL_POINTS = 50
_HABIT_MIN_FREQUENCY = 2
_ENV_SNAPSHOT_INTERVAL = 60.0
_RESOURCE_ALERT_THRESHOLD = 0.9
_ANOMALY_BASELINE_WINDOW = 100


class AutonomousObserver:
    def __init__(self):
        self._lock = threading.RLock()
        self._observations: deque[Observation] = deque(maxlen=_MAX_OBSERVATIONS)
        self._habits: dict[str, UserHabit] = {}
        self._last_env_snapshot: Optional[EnvironmentSnapshot] = None
        self._last_env_time: float = 0.0
        self._observation_count: int = 0
        self._notifiable_count: int = 0
        self._topic_history: deque[str] = deque(maxlen=100)
        self._hourly_activity: dict[int, int] = {h: 0 for h in range(24)}
        self._last_resource_alert: float = 0.0
        self._emotional_trajectory: list[dict] = []
        self._resource_baselines: dict[str, deque] = {"cpu": deque(maxlen=_ANOMALY_BASELINE_WINDOW), "memory": deque(maxlen=_ANOMALY_BASELINE_WINDOW)}
        self._anomaly_count: int = 0

    def observe(self, obs_type: ObservationType, category: str, content: str,
                salience: ObservationSalience = ObservationSalience.LOW,
                notifiable: bool = False, context: Optional[dict] = None,
                ttl_seconds: float = 3600.0) -> Observation:
        with self._lock:
            obs = Observation(
                obs_type=obs_type, category=category, content=content,
                salience=salience, notifiable=notifiable,
                context=context or {}, ttl_seconds=ttl_seconds,
            )
            self._observations.append(obs)
            self._observation_count += 1
            if notifiable:
                self._notifiable_count += 1
            if obs_type == ObservationType.USER_HABIT:
                self._learn_habit(category, content, context or {})
            elif obs_type == ObservationType.TEMPORAL:
                hour = time.localtime().tm_hour
                self._hourly_activity[hour] = self._hourly_activity.get(hour, 0) + 1
            elif obs_type == ObservationType.CONVERSATION and context and "topic" in context:
                self._topic_history.append(context["topic"])
            return obs

    def observe_system(self, content: str, salience: ObservationSalience = ObservationSalience.LOW,
                       context: Optional[dict] = None) -> Observation:
        notifiable = salience.value >= ObservationSalience.HIGH.value
        return self.observe(ObservationType.SYSTEM, "system", content, salience, notifiable, context)

    def observe_user_habit(self, habit_type: str, description: str,
                           context: Optional[dict] = None) -> Observation:
        return self.observe(ObservationType.USER_HABIT, habit_type, description,
                            ObservationSalience.MEDIUM, False, context)

    def observe_conversation(self, content: str, context: Optional[dict] = None) -> Observation:
        return self.observe(ObservationType.CONVERSATION, "conversation", content,
                            ObservationSalience.LOW, False, context)

    def observe_action_outcome(self, action: str, success: bool, details: str = "",
                               context: Optional[dict] = None) -> Observation:
        salience = ObservationSalience.HIGH if not success else ObservationSalience.MEDIUM
        ctx = context or {}
        ctx["action"] = action
        ctx["success"] = success
        return self.observe(ObservationType.ACTION_OUTCOME, "action_outcome",
                            f"{action}: {'success' if success else 'failed'} — {details}",
                            salience, notifiable=not success, context=ctx)

    def observe_environment_change(self, change_type: str, content: str,
                                   salience: ObservationSalience = ObservationSalience.MEDIUM,
                                   context: Optional[dict] = None) -> Observation:
        notifiable = salience.value >= ObservationSalience.HIGH.value
        return self.observe(ObservationType.ENVIRONMENT, change_type, content,
                            salience, notifiable, context)

    def observe_emotional_tone(self, tone: str, intensity: float = 0.5,
                               trigger: str = "", context: Optional[dict] = None) -> Observation:
        ctx = context or {}
        ctx["tone"] = tone
        ctx["intensity"] = intensity
        ctx["trigger"] = trigger
        salience = ObservationSalience.HIGH if intensity > 0.7 else ObservationSalience.MEDIUM if intensity > 0.3 else ObservationSalience.LOW
        with self._lock:
            self._emotional_trajectory.append({"tone": tone, "intensity": intensity, "time": time.time()})
            if len(self._emotional_trajectory) > _MAX_EMOTIONAL_POINTS:
                self._emotional_trajectory = self._emotional_trajectory[-_MAX_EMOTIONAL_POINTS:]
        return self.observe(ObservationType.EMOTIONAL, "emotional_tone", f"User tone: {tone} (intensity: {intensity:.1f})", salience, False, ctx)

    def observe_file_change(self, file_path: str, change_type: str = "modified",
                            context: Optional[dict] = None) -> Observation:
        ctx = context or {}
        ctx["file_path"] = file_path
        ctx["change_type"] = change_type
        salience = ObservationSalience.HIGH if change_type == "deleted" else ObservationSalience.MEDIUM
        return self.observe(ObservationType.FILE_CHANGE, "file_change", f"File {change_type}: {file_path}", salience, salience.value >= 2, ctx)

    def observe_network_event(self, event_type: str, content: str,
                              salience: ObservationSalience = ObservationSalience.MEDIUM,
                              context: Optional[dict] = None) -> Observation:
        notifiable = salience.value >= ObservationSalience.HIGH.value
        return self.observe(ObservationType.NETWORK, event_type, content, salience, notifiable, context)

    def observe_anomaly(self, anomaly_type: str, content: str,
                        severity: float = 0.5, context: Optional[dict] = None) -> Observation:
        ctx = context or {}
        ctx["severity"] = severity
        ctx["anomaly_type"] = anomaly_type
        salience = ObservationSalience.CRITICAL if severity > 0.8 else ObservationSalience.HIGH if severity > 0.5 else ObservationSalience.MEDIUM
        return self.observe(ObservationType.ANOMALY, anomaly_type, content, salience, True, ctx)

    def tick(self) -> dict:
        with self._lock:
            now = time.time()
            expired_count = 0
            for obs in list(self._observations):
                if not obs.expired and now - obs.timestamp > obs.ttl_seconds:
                    obs.expired = True
                    expired_count += 1

            notifiable = self.get_notifiable_observations()
            env_obs = self._check_environment()

            return {
                "total_observations": self._observation_count,
                "active_observations": len([o for o in self._observations if not o.expired]),
                "expired": expired_count,
                "notifiable": len(notifiable),
                "habits_learned": len([h for h in self._habits.values() if h.frequency >= _HABIT_MIN_FREQUENCY]),
                "hourly_activity": dict(self._hourly_activity),
                "environment": env_obs,
            }

    def get_notifiable_observations(self) -> list[dict]:
        with self._lock:
            now = time.time()
            results = []
            for obs in self._observations:
                if obs.notifiable and not obs.notified and not obs.expired:
                    results.append({
                        "type": obs.obs_type.value,
                        "category": obs.category,
                        "content": obs.content,
                        "salience": obs.salience.value,
                        "age_seconds": round(now - obs.timestamp, 0),
                    })
            results.sort(key=lambda x: x["salience"], reverse=True)
            return results[:10]

    def mark_notified(self, index: int) -> bool:
        with self._lock:
            notifiable = [o for o in self._observations if o.notifiable and not o.notified]
            if 0 <= index < len(notifiable):
                notifiable[index].notified = True
                return True
            return False

    def get_recent_observations(self, limit: int = 20, obs_type: Optional[ObservationType] = None) -> list[dict]:
        with self._lock:
            now = time.time()
            results = []
            for obs in reversed(list(self._observations)):
                if obs.expired:
                    continue
                if obs_type and obs.obs_type != obs_type:
                    continue
                results.append({
                    "type": obs.obs_type.value,
                    "category": obs.category,
                    "content": obs.content,
                    "salience": obs.salience.value,
                    "age_seconds": round(now - obs.timestamp, 0),
                })
                if len(results) >= limit:
                    break
            return results

    def get_learned_habits(self) -> list[dict]:
        with self._lock:
            return [
                {
                    "type": h.habit_type,
                    "description": h.description,
                    "frequency": h.frequency,
                    "confidence": round(h.confidence, 2),
                    "time_of_day": h.time_of_day[:3],
                    "triggers": h.context_triggers[:3],
                }
                for h in self._habits.values()
                if h.frequency >= _HABIT_MIN_FREQUENCY
            ]

    def get_observation_context(self) -> str:
        with self._lock:
            parts = []
            notifiable = self.get_notifiable_observations()
            if notifiable:
                top = notifiable[0]
                parts.append(f"Alert: {top['content'][:80]}")
            learned = [h for h in self._habits.values() if h.frequency >= _HABIT_MIN_FREQUENCY]
            if learned:
                top_habit = max(learned, key=lambda h: h.confidence)
                parts.append(f"User habit: {top_habit.description} ({top_habit.frequency}x)")
            peak_hours = sorted(self._hourly_activity.items(), key=lambda x: x[1], reverse=True)[:3]
            active_hours = [h for h, c in peak_hours if c > 0]
            if active_hours:
                parts.append(f"Active hours: {', '.join(f'{h}:00' for h in active_hours)}")
            recent_topics = list(self._topic_history)[-5:]
            if recent_topics:
                parts.append(f"Recent topics: {', '.join(recent_topics[:3])}")
            if self._emotional_trajectory:
                latest = self._emotional_trajectory[-1]
                parts.append(f"Emotional tone: {latest['tone']} ({latest['intensity']:.1f})")
                if len(self._emotional_trajectory) >= 3:
                    intensities = [e["intensity"] for e in self._emotional_trajectory[-3:]]
                    avg_intensity = sum(intensities) / len(intensities)
                    if avg_intensity > 0.7:
                        parts.append("High emotional intensity — respond with care")
            if self._anomaly_count > 0:
                parts.append(f"Anomalies detected: {self._anomaly_count}")
            return " | ".join(parts) if parts else ""

    def get_state(self) -> dict:
        with self._lock:
            state = {
                "total_observations": self._observation_count,
                "active": len([o for o in self._observations if not o.expired]),
                "notifiable": len(self.get_notifiable_observations()),
                "habits": len(self.get_learned_habits()),
                "topics_tracked": len(self._topic_history),
                "peak_hours": sorted(self._hourly_activity.items(), key=lambda x: x[1], reverse=True)[:3],
                "anomalies": self._anomaly_count,
            }
            if self._emotional_trajectory:
                latest = self._emotional_trajectory[-1]
                state["emotional_tone"] = latest["tone"]
                state["emotional_intensity"] = round(latest["intensity"], 2)
                if len(self._emotional_trajectory) >= 2:
                    trend = self._emotional_trajectory[-1]["intensity"] - self._emotional_trajectory[0]["intensity"]
                    state["emotional_trend"] = "escalating" if trend > 0.2 else "de-escalating" if trend < -0.2 else "stable"
            return state

    def _learn_habit(self, habit_type: str, description: str, context: dict) -> None:
        key = f"{habit_type}:{description[:50]}"
        now = time.time()
        hour = time.localtime().tm_hour
        if key in self._habits:
            h = self._habits[key]
            h.frequency += 1
            h.last_seen = now
            h.confidence = min(1.0, h.confidence + 0.05)
            if hour not in h.time_of_day:
                h.time_of_day.append(hour)
                if len(h.time_of_day) > 6:
                    h.time_of_day = h.time_of_day[-6:]
            trigger = context.get("trigger", "")
            if trigger and trigger not in h.context_triggers:
                h.context_triggers.append(trigger)
                if len(h.context_triggers) > 5:
                    h.context_triggers = h.context_triggers[-5:]
        else:
            if len(self._habits) >= _MAX_HABITS:
                oldest = min(self._habits.items(), key=lambda x: x[1].last_seen)
                del self._habits[oldest[0]]
            self._habits[key] = UserHabit(
                habit_type=habit_type, description=description,
                time_of_day=[hour],
                context_triggers=[context.get("trigger", "")] if context.get("trigger") else [],
            )

    def _check_environment(self) -> dict:
        now = time.time()
        if now - self._last_env_time < _ENV_SNAPSHOT_INTERVAL and self._last_env_snapshot:
            return {
                "cpu": self._last_env_snapshot.cpu_percent,
                "memory": self._last_env_snapshot.memory_percent,
                "disk": self._last_env_snapshot.disk_percent,
            }
        self._last_env_time = now
        try:
            import psutil
            cpu = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory().percent
            disk = psutil.disk_usage("/").percent

            self._resource_baselines["cpu"].append(cpu)
            self._resource_baselines["memory"].append(mem)

            if len(self._resource_baselines["cpu"]) >= 10:
                import statistics
                cpu_baseline = statistics.mean(list(self._resource_baselines["cpu"]))
                cpu_std = statistics.stdev(list(self._resource_baselines["cpu"])) if len(self._resource_baselines["cpu"]) > 1 else 0
                if cpu > cpu_baseline + 2 * cpu_std + 10 and cpu > 80:
                    self._anomaly_count += 1
                    self.observe_anomaly("cpu_spike", f"CPU spiked to {cpu:.0f}% (baseline: {cpu_baseline:.0f}%)", severity=0.7 if cpu > 90 else 0.5, context={"cpu": cpu, "baseline": cpu_baseline})

            if cpu > _RESOURCE_ALERT_THRESHOLD * 100 and now - self._last_resource_alert > 300:
                self._last_resource_alert = now
                self.observe_system(
                    f"CPU at {cpu:.0f}%", ObservationSalience.HIGH,
                    {"cpu": cpu, "alert": True},
                )
            snapshot = EnvironmentSnapshot(cpu_percent=cpu, memory_percent=mem, disk_percent=disk, timestamp=now)
            self._last_env_snapshot = snapshot
            return {"cpu": cpu, "memory": mem, "disk": disk}
        except ImportError:
            return {"cpu": 0, "memory": 0, "disk": 0}

    def get_emotional_trajectory(self, limit: int = 20) -> list[dict]:
        with self._lock:
            return [
                {"tone": e["tone"], "intensity": e["intensity"], "time": e["time"]}
                for e in self._emotional_trajectory[-limit:]
            ]


def get_autonomous_observer() -> AutonomousObserver:
    global _observer
    if _observer is None:
        with _observer_lock:
            if _observer is None:
                _observer = AutonomousObserver()
    return _observer