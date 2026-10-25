"""Contextual Memory - Jarvis remembers who you are, what you like, and how you work.

This isn't just key-value storage — it's understanding. The system learns:
1. Communication preferences (brief vs. detailed, formal vs. casual)
2. Technical patterns (favorite tools, languages, workflows)
3. Emotional patterns (what frustrates you, what motivates you)
4. Project preferences (naming conventions, test frameworks, architectures)
5. Temporal patterns (when you're productive, how long your sessions are)

All stored locally, all under your control, all used to make Jarvis feel like
it actually knows you rather than just serving anyone.
"""
import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_context_memory: Optional["ContextualMemory"] = None
_context_lock = threading.Lock()


class PreferenceCategory(Enum):
    COMMUNICATION = "communication"
    TECHNICAL = "technical"
    EMOTIONAL = "emotional"
    PROJECT = "project"
    TEMPORAL = "temporal"
    WORKFLOW = "workflow"


@dataclass
class UserPreference:
    key: str
    category: PreferenceCategory
    value: str
    confidence: float = 0.3
    times_observed: int = 1
    first_observed: float = field(default_factory=time.time)
    last_observed: float = field(default_factory=time.time)
    source: str = "observed"


@dataclass
class SessionSummary:
    duration_minutes: float = 0.0
    messages: int = 0
    topics: list[str] = field(default_factory=list)
    sentiment: str = "neutral"
    productivity: float = 0.5
    start_time: float = field(default_factory=time.time)


class ContextualMemory:
    MAX_PREFERENCES = 200
    MAX_SESSION_HISTORY = 50

    def __init__(self, storage_path: Optional[Path] = None):
        self._lock = threading.RLock()
        self._storage_path = storage_path or (Path.home() / ".jarvis" / "contextual_memory")
        self._preferences: dict[str, UserPreference] = {}
        self._session_history: deque[SessionSummary] = deque(maxlen=self.MAX_SESSION_HISTORY)
        self._current_session: Optional[SessionSummary] = None
        self._productivity_scores: deque[float] = deque(maxlen=20)
        self._topic_counts: dict[str, int] = {}
        self._load_preferences()

    def learn_preference(self, key: str, value: str, category: PreferenceCategory = PreferenceCategory.COMMUNICATION,
                         confidence: float = 0.3, source: str = "observed") -> None:
        with self._lock:
            if key in self._preferences:
                p = self._preferences[key]
                if p.value == value:
                    p.times_observed += 1
                    p.confidence = min(1.0, p.confidence + 0.15)
                    p.last_observed = time.time()
                else:
                    p.value = value
                    p.confidence = max(0.2, p.confidence * 0.7)
                    p.times_observed += 1
                    p.last_observed = time.time()
                    p.source = source
            else:
                if len(self._preferences) >= self.MAX_PREFERENCES:
                    weakest = min(self._preferences.items(), key=lambda x: x[1].confidence * x[1].times_observed)
                    del self._preferences[weakest[0]]
                self._preferences[key] = UserPreference(
                    key=key, category=category, value=value,
                    confidence=confidence, source=source)
            self._save_preferences()

    def learn_from_interaction(self, user_message: str, response_length: int = 0,
                               user_sentiment: str = "neutral",
                               tools_used: list[str] = None,
                               topic: str = "") -> None:
        with self._lock:
            words = user_message.lower().split()
            if len(words) < 5:
                self.learn_preference("preferred_response_length", "brief",
                                     PreferenceCategory.COMMUNICATION, 0.3)
            elif len(words) > 50:
                self.learn_preference("preferred_response_length", "detailed",
                                     PreferenceCategory.COMMUNICATION, 0.3)
            informal = sum(1 for w in words if w in ("hey", "yo", "cool", "nah", "yeah", "gonna", "wanna"))
            if informal > 2:
                self.learn_preference("formality", "casual", PreferenceCategory.COMMUNICATION, 0.5)
            has_code = any("`" in user_message or w in words
                          for w in ["function", "class", "def ", "import", "var ", "const "])
            if has_code:
                self.learn_preference("technical_level", "advanced", PreferenceCategory.TECHNICAL, 0.4)
            if user_sentiment in ("frustrated", "angry"):
                existing = self._preferences.get("frustration_frequency")
                if existing:
                    existing.times_observed += 1
                    existing.value = str(int(existing.value) + 1)
                else:
                    self.learn_preference("frustration_frequency", "1",
                                         PreferenceCategory.EMOTIONAL, 0.5, "observed_emotion")
            if tools_used:
                for tool in tools_used:
                    tool_key = f"tool_preference_{tool}"
                    self.learn_preference(tool_key, "frequently_used",
                                         PreferenceCategory.WORKFLOW, 0.4)
            if topic:
                self._topic_counts[topic] = self._topic_counts.get(topic, 0) + 1
                if self._topic_counts[topic] >= 3:
                    self.learn_preference(f"primary_topic_{topic}", "active",
                                         PreferenceCategory.PROJECT, 0.5, "topic_tracking")
            file_extensions = set()
            for pattern in [r'\.(\w{1,4})\b', r'/(\w+)/']:
                import re
                found = re.findall(pattern, user_message)
                file_extensions.update(found)
            if file_extensions:
                for ext in file_extensions:
                    lang_map = {
                        "py": "python", "js": "javascript", "ts": "typescript",
                        "rs": "rust", "go": "go", "java": "java", "rb": "ruby",
                        "cpp": "cpp", "c": "c", "h": "c_header", "tsx": "react",
                        "jsx": "react", "css": "css", "html": "html",
                        "sql": "database", "sh": "shell", "yml": "config",
                        "yaml": "config", "toml": "config", "json": "config",
                        "md": "documentation", "txt": "text",
                    }
                    language = lang_map.get(ext, ext)
                    self.learn_preference(f"programming_language_{language}", "active",
                                         PreferenceCategory.TECHNICAL, 0.3, "file_pattern")
            productivity = 0.5
            if user_sentiment in ("happy", "excited"):
                productivity += 0.2
            elif user_sentiment in ("frustrated", "confused"):
                productivity -= 0.1
            if tools_used:
                productivity += 0.1
            self._productivity_scores.append(max(0.0, min(1.0, productivity)))

    def start_session(self) -> None:
        with self._lock:
            self._current_session = SessionSummary()

    def end_session(self) -> None:
        with self._lock:
            if self._current_session:
                self._current_session.duration_minutes = (time.time() - self._current_session.start_time) / 60
                self._session_history.append(self._current_session)
                self._current_session = None
                self._save_preferences()

    def get_communication_style(self) -> dict:
        with self._lock:
            style = {"verbosity": "moderate", "formality": "professional",
                     "warmth": 0.6, "humor": 0.3, "proactivity": 0.7}
            length_pref = self._preferences.get("preferred_response_length")
            if length_pref:
                style["verbosity"] = length_pref.value
            formality = self._preferences.get("formality")
            if formality:
                style["formality"] = formality.value
                style["warmth"] = 0.8 if formality.value == "casual" else 0.5
                style["humor"] = 0.5 if formality.value == "casual" else 0.2
            prod_scores = list(self._productivity_scores)
            if prod_scores:
                avg_prod = sum(prod_scores) / len(prod_scores)
                style["proactivity"] = min(1.0, 0.5 + avg_prod * 0.5)
            return style

    def get_temporal_patterns(self) -> dict:
        with self._lock:
            if not self._session_history:
                return {"peak_hours": [], "avg_session_minutes": 0, "total_sessions": 0}
            hours = []
            durations = []
            for s in self._session_history:
                hour = time.localtime(s.start_time).tm_hour
                hours.append(hour)
                durations.append(s.duration_minutes)
            from collections import Counter
            hour_counts = Counter(hours)
            peak_hours = [h for h, c in hour_counts.most_common(3)]
            return {
                "peak_hours": sorted(peak_hours),
                "avg_session_minutes": round(sum(durations) / max(len(durations), 1), 1),
                "total_sessions": len(self._session_history),
                "avg_productivity": round(sum(self._productivity_scores) / max(len(self._productivity_scores), 1), 2),
            }

    def get_project_preferences(self) -> dict:
        with self._lock:
            prefs = {}
            for key, p in self._preferences.items():
                if p.category == PreferenceCategory.PROJECT and p.times_observed >= 2:
                    prefs[key] = {"value": p.value, "confidence": round(p.confidence, 2)}
            return prefs

    def get_summary(self) -> str:
        with self._lock:
            parts = []
            comm_style = self.get_communication_style()
            prefs = [(k, p) for k, p in self._preferences.items() if p.confidence > 0.5]
            if prefs:
                top_prefs = sorted(prefs, key=lambda x: x[1].confidence, reverse=True)[:5]
                pref_str = ", ".join(f"{k}: {p.value}" for k, p in top_prefs)
                parts.append(f"Known preferences: {pref_str}")
            if comm_style.get("formality") != "professional":
                parts.append(f"Communication style: {comm_style['formality']}, {comm_style['verbosity']}")
            temporal = self.get_temporal_patterns()
            if temporal["peak_hours"]:
                hours_str = ", ".join(f"{h}:00" for h in temporal["peak_hours"])
                parts.append(f"Usually active: {hours_str}")
            frustration = self._preferences.get("frustration_frequency")
            if frustration and int(frustration.value) > 3:
                parts.append("Has experienced frustration recently — be patient and clear")
            return " | ".join(parts) if parts else ""

    def get_state(self) -> dict:
        with self._lock:
            return {
                "preferences_learned": len(self._preferences),
                "high_confidence_prefs": len([p for p in self._preferences.values() if p.confidence > 0.5]),
                "session_count": len(self._session_history),
                "avg_productivity": round(sum(self._productivity_scores) / max(len(self._productivity_scores), 1), 2),
                "primary_topics": dict(sorted(self._topic_counts.items(), key=lambda x: x[1], reverse=True)[:5]),
                "communication_style": self.get_communication_style(),
            }

    def _save_preferences(self) -> None:
        try:
            self._storage_path.mkdir(parents=True, exist_ok=True)
            data = {}
            for key, p in self._preferences.items():
                data[key] = {
                    "category": p.category.value, "value": p.value,
                    "confidence": p.confidence, "times_observed": p.times_observed,
                    "first_observed": p.first_observed, "last_observed": p.last_observed,
                    "source": p.source,
                }
            prefs_file = self._storage_path / "preferences.json"
            prefs_file.write_text(json.dumps(data, indent=2), encoding='utf-8')
        except Exception as e:
            logger.debug(f"Failed to save contextual preferences: {e}")

    def _load_preferences(self) -> None:
        try:
            prefs_file = self._storage_path / "preferences.json"
            if prefs_file.exists():
                data = json.loads(prefs_file.read_text(encoding='utf-8'))
                for key, d in data.items():
                    self._preferences[key] = UserPreference(
                        key=key, category=PreferenceCategory(d["category"]),
                        value=d["value"], confidence=d.get("confidence", 0.3),
                        times_observed=d.get("times_observed", 1),
                        first_observed=d.get("first_observed", time.time()),
                        last_observed=d.get("last_observed", time.time()),
                        source=d.get("source", "observed"))
        except Exception as e:
            logger.debug(f"Failed to load contextual preferences: {e}")


def get_contextual_memory() -> ContextualMemory:
    global _context_memory
    if _context_memory is None:
        with _context_lock:
            if _context_memory is None:
                _context_memory = ContextualMemory()
    return _context_memory