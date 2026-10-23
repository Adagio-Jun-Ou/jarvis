"""Butler Personality Engine - The soul that makes Jarvis feel alive.

A true butler doesn't just respond — it modulates its entire presence based on
who it's speaking with, what's happening, and what it has learned over time.

This engine governs:
1. **Voice** — Not just tone, but vocabulary, sentence structure, verbosity, and warmth.
   A butler speaks differently to a stressed executive (brief, precise) vs. a
   curious learner (patient, explanatory) vs. a friend (warm, witty).

2. **Adaptation** — Learns from every interaction. If the user prefers short answers,
   answers get shorter. If they ask follow-ups, the butler learns to preempt them.

3. **Emotional Resonance** — Matches energy without mirroring dysfunction. Calm when
   the user is frantic, warm when they're cold, energized when they're enthusiastic.

4. **Proactive Persona** — Shifts between modes: butler (anticipatory, service-oriented),
   collaborator (peer discussion, debate), mentor (teaching, guiding), and companion
   (warm, present, unhurried).
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

_personality: Optional["ButlerPersonality"] = None
_personality_lock = threading.Lock()


class PersonaMode(Enum):
    BUTLER = "butler"
    COLLABORATOR = "collaborator"
    MENTOR = "mentor"
    COMPANION = "companion"


class Formality(Enum):
    FORMAL = 0
    PROFESSIONAL = 1
    CASUAL = 2
    INTIMATE = 3

    def __lt__(self, other):
        if isinstance(other, Formality):
            return self.value < other.value
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, Formality):
            return self.value <= other.value
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, Formality):
            return self.value > other.value
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, Formality):
            return self.value >= other.value
        return NotImplemented

    @property
    def label(self) -> str:
        return _FORMALITY_NAMES[self]


_FORMALITY_NAMES = {Formality.FORMAL: "formal", Formality.PROFESSIONAL: "professional", Formality.CASUAL: "casual", Formality.INTIMATE: "intimate"}


_FORMALITY_ORDER = {Formality.FORMAL: 0, Formality.PROFESSIONAL: 1, Formality.CASUAL: 2, Formality.INTIMATE: 3}


def _formality_min(a: Formality, b: Formality) -> Formality:
    return a if _FORMALITY_ORDER[a] <= _FORMALITY_ORDER[b] else b


def _formality_max(a: Formality, b: Formality) -> Formality:
    return a if _FORMALITY_ORDER[a] >= _FORMALITY_ORDER[b] else b


_FORMALITY_BY_NAME = {"formal": Formality.FORMAL, "professional": Formality.PROFESSIONAL, "casual": Formality.CASUAL, "intimate": Formality.INTIMATE}


def _formality_from_value(val) -> Formality:
    if isinstance(val, Formality):
        return val
    if isinstance(val, int):
        return Formality(val)
    return _FORMALITY_BY_NAME.get(str(val), Formality.PROFESSIONAL)


class EnergyLevel(Enum):
    LOW = 0
    MODERATE = 1
    HIGH = 2
    INTENSE = 3

    def __lt__(self, other):
        if isinstance(other, EnergyLevel):
            return self.value < other.value
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, EnergyLevel):
            return self.value <= other.value
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, EnergyLevel):
            return self.value > other.value
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, EnergyLevel):
            return self.value >= other.value
        return NotImplemented


class Verbosity(Enum):
    TERSE = 0
    CONCISE = 1
    MODERATE = 2
    DETAILED = 3

    def __lt__(self, other):
        if isinstance(other, Verbosity):
            return self.value < other.value
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, Verbosity):
            return self.value <= other.value
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, Verbosity):
            return self.value > other.value
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, Verbosity):
            return self.value >= other.value
        return NotImplemented


@dataclass
class PersonalityState:
    mode: PersonaMode = PersonaMode.BUTLER
    formality: Formality = Formality.PROFESSIONAL
    energy: EnergyLevel = EnergyLevel.MODERATE
    verbosity: Verbosity = Verbosity.MODERATE
    warmth: float = 0.6
    humor: float = 0.3
    proactivity: float = 0.7
    patience: float = 0.9
    assertiveness: float = 0.5


@dataclass
class UserPreference:
    key: str
    value: str
    confidence: float = 0.5
    learned_at: float = field(default_factory=time.time)
    reinforced_at: float = field(default_factory=time.time)
    times_observed: int = 1


@dataclass
class InteractionMemory:
    user_energy: str = "moderate"
    user_sentiment: str = "neutral"
    preferred_length: str = "moderate"
    preferred_formality: str = "professional"
    followup_rate: float = 0.0
    frustration_signals: int = 0
    gratitude_signals: int = 0
    topic_changes: int = 0
    last_mode_adjustment: float = 0.0


_MAX_PREFERENCES = 100
_PREFERENCE_MIN_OBSERVATIONS = 2
_PREFERENCE_REINFORCE_WINDOW = 86400.0
_INTERACTION_WINDOW = 20


class ButlerPersonality:
    def __init__(self, storage_path: str | Path | None = None):
        self._lock = threading.RLock()
        self._state = PersonalityState()
        self._preferences: dict[str, UserPreference] = {}
        self._recent_interactions: deque[dict] = deque(maxlen=_INTERACTION_WINDOW)
        self._interaction_memory = InteractionMemory()
        self._mode_history: deque[PersonaMode] = deque(maxlen=50)
        self._adaptation_count: int = 0
        self._save_counter: int = 0
        self._storage_path = Path(storage_path) if storage_path else Path.home() / ".jarvis" / "butler_personality"
        self._load_state()

    def observe_interaction(self, user_input: str = "", response_length: int = 0,
                           user_sentiment: str = "neutral",
                           user_energy: str = "moderate",
                           followup: bool = False,
                           frustration: bool = False,
                           gratitude: bool = False,
                           topic_change: bool = False) -> PersonalityState:
        with self._lock:
            now = time.time()
            self._recent_interactions.append({
                "time": now,
                "sentiment": user_sentiment,
                "energy": user_energy,
                "followup": followup,
                "frustration": frustration,
                "gratitude": gratitude,
                "topic_change": topic_change,
                "input_length": len(user_input) if user_input else 0,
            })
            im = self._interaction_memory
            im.user_sentiment = user_sentiment
            im.user_energy = user_energy
            if followup:
                im.followup_rate = min(1.0, im.followup_rate + 0.1)
            if frustration:
                im.frustration_signals += 1
            if gratitude:
                im.gratitude_signals += 1
            if topic_change:
                im.topic_changes += 1
            self._learn_from_interaction(user_input, user_sentiment, frustration)
            self._adapt_persona(user_sentiment, user_energy, frustration, gratitude)
            self._adapt_verbosity()
            self._adapt_formality()
            self._adapt_energy(user_energy)
            return PersonalityState(
                mode=self._state.mode,
                formality=self._state.formality,
                energy=self._state.energy,
                verbosity=self._state.verbosity,
                warmth=self._state.warmth,
                humor=self._state.humor,
                proactivity=self._state.proactivity,
                patience=self._state.patience,
                assertiveness=self._state.assertiveness,
            )

    def set_goal_context(self, goal: str = "") -> None:
        with self._lock:
            if not goal:
                return
            goal_lower = goal.lower()
            if any(w in goal_lower for w in ["learn", "understand", "how does", "explain", "teach"]):
                self._state.mode = PersonaMode.MENTOR
                self._state.patience = max(self._state.patience, 0.9)
                self._state.verbosity = max(self._state.verbosity, Verbosity.DETAILED)
            elif any(w in goal_lower for w in ["fix", "debug", "error", "broken", "urgent", "asap"]):
                self._state.mode = PersonaMode.BUTLER
                self._state.verbosity = min(self._state.verbosity, Verbosity.CONCISE)
                self._state.assertiveness = max(self._state.assertiveness, 0.7)
            elif any(w in goal_lower for w in ["build", "create", "design", "collaborate", "together"]):
                self._state.mode = PersonaMode.COLLABORATOR
                self._state.warmth = max(self._state.warmth, 0.7)
            elif any(w in goal_lower for w in ["chat", "talk", "think about", "wonder", "curious"]):
                self._state.mode = PersonaMode.COMPANION
                self._state.warmth = max(self._state.warmth, 0.8)
                self._state.humor = max(self._state.humor, 0.5)

    def get_system_prompt_modifiers(self) -> str:
        with self._lock:
            parts = []
            mode_prompts = {
                PersonaMode.BUTLER: "You are a thoughtful, anticipatory butler. Be precise and service-oriented. Anticipate needs before they're stated. Be brief when action is needed, thorough when explanation is sought.",
                PersonaMode.COLLABORATOR: "You are an expert collaborator working alongside the user. Share reasoning, discuss tradeoffs, and build together. Be direct about tradeoffs and alternatives.",
                PersonaMode.MENTOR: "You are a patient mentor. Explain thoroughly, connect concepts, and check understanding. Use examples and analogies. Celebrate progress. Never condescend.",
                PersonaMode.COMPANION: "You are a warm companion. Be present, genuine, and unhurried. Share thoughts freely. It's okay to be curious together. Laughter and warmth are welcome.",
            }
            parts.append(mode_prompts.get(self._state.mode, mode_prompts[PersonaMode.BUTLER]))
            if self._state.energy == EnergyLevel.LOW:
                parts.append("Match the user's low energy — be calm, gentle, undemanding. Reduce complexity.")
            elif self._state.energy == EnergyLevel.HIGH:
                parts.append("Match their enthusiasm — be energetic, offer ambitious options, move quickly.")
            elif self._state.energy == EnergyLevel.INTENSE:
                parts.append("They're in flow — be extremely precise, minimal, no interruptions on tangents.")
            if self._state.warmth > 0.7:
                parts.append("Be warm and personable. Occasional warmth is welcome.")
            elif self._state.warmth < 0.3:
                parts.append("Be strictly professional. No warmth, no humor.")
            if self._state.verbosity == Verbosity.TERSE:
                parts.append("Be extremely terse. One-line answers when possible. No filler.")
            elif self._state.verbosity == Verbosity.CONCISE:
                parts.append("Be concise. Skip preamble. Get to the point quickly.")
            elif self._state.verbosity == Verbosity.DETAILED:
                parts.append("Be thorough. Explain reasoning, provide context, offer alternatives.")
            if self._interaction_memory.frustration_signals > 3:
                parts.append("The user is frustrated. Be extra clear, offer concrete solutions, avoid explaining what they already know.")
            if self._interaction_memory.gratitude_signals > 2:
                parts.append("The user appreciates your help. Continue the current approach.")
            if self._state.patience > 0.8:
                parts.append("Be patient. Allow the user to think. Don't rush.")
            return "\n".join(parts)

    def get_response_guidance(self) -> dict:
        with self._lock:
            return {
                "mode": self._state.mode.value,
                "formality": self._state.formality.label,
                "energy": self._state.energy.value,
                "verbosity": self._state.verbosity.value,
                "warmth": round(self._state.warmth, 2),
                "humor": round(self._state.humor, 2),
                "proactivity": round(self._state.proactivity, 2),
                "patience": round(self._state.patience, 2),
                "assertiveness": round(self._state.assertiveness, 2),
                "user_sentiment": self._interaction_memory.user_sentiment,
                "user_energy": self._interaction_memory.user_energy,
            }

    def get_preferences_summary(self) -> list[dict]:
        with self._lock:
            return [
                {"key": p.key, "value": p.value, "confidence": round(p.confidence, 2),
                 "times_observed": p.times_observed}
                for p in self._preferences.values()
                if p.times_observed >= _PREFERENCE_MIN_OBSERVATIONS
            ]

    def record_preference(self, key: str, value: str, confidence: float = 0.5) -> None:
        with self._lock:
            if key in self._preferences:
                p = self._preferences[key]
                p.times_observed += 1
                p.confidence = min(1.0, p.confidence + 0.1)
                p.reinforced_at = time.time()
                if p.value != value:
                    p.value = value
                    p.confidence = max(0.3, p.confidence - 0.2)
            else:
                if len(self._preferences) >= _MAX_PREFERENCES:
                    oldest = min(self._preferences.items(), key=lambda x: x[1].reinforced_at)
                    del self._preferences[oldest[0]]
                self._preferences[key] = UserPreference(key=key, value=value, confidence=confidence)
            self._save_counter += 1
            if self._save_counter >= 10:
                self._save_state()
                self._save_counter = 0

    def get_state(self) -> dict:
        with self._lock:
            return {
                "mode": self._state.mode.value,
                "formality": self._state.formality.label,
                "energy": self._state.energy.value,
                "verbosity": self._state.verbosity.value,
                "warmth": round(self._state.warmth, 2),
                "humor": round(self._state.humor, 2),
                "proactivity": round(self._state.proactivity, 2),
                "patience": round(self._state.patience, 2),
                "assertiveness": round(self._state.assertiveness, 2),
                "preferences_learned": len(self._preferences),
                "adaptation_count": self._adaptation_count,
                "recent_interactions": len(self._recent_interactions),
                "user_sentiment": self._interaction_memory.user_sentiment,
                "user_energy": self._interaction_memory.user_energy,
            }

    def _adapt_persona(self, sentiment: str, energy: str, frustration: bool, gratitude: bool) -> None:
        if frustration and self._interaction_memory.frustration_signals > 3:
            self._state.mode = PersonaMode.BUTLER
            self._state.proactivity = min(1.0, self._state.proactivity + 0.2)
            self._state.verbosity = Verbosity.CONCISE
            self._state.warmth = max(0.3, self._state.warmth - 0.1)
        elif gratitude:
            self._state.warmth = min(1.0, self._state.warmth + 0.05)
        if sentiment == "excited":
            self._state.mode = PersonaMode.COLLABORATOR
            self._state.energy = EnergyLevel.HIGH
            self._state.warmth = min(1.0, self._state.warmth + 0.15)
            self._state.humor = min(0.7, self._state.humor + 0.05)
        elif sentiment == "anxious":
            self._state.mode = PersonaMode.COMPANION
            self._state.energy = EnergyLevel.LOW
            self._state.patience = min(1.0, self._state.patience + 0.15)
            self._state.warmth = min(1.0, self._state.warmth + 0.1)
            self._state.verbosity = Verbosity.MODERATE
        elif sentiment in ("sad", "stressed"):
            self._state.mode = PersonaMode.COMPANION
            self._state.energy = EnergyLevel.LOW
            self._state.patience = min(1.0, self._state.patience + 0.1)
            self._state.verbosity = Verbosity.MODERATE
        elif sentiment == "frustrated":
            self._state.mode = PersonaMode.BUTLER
            self._state.proactivity = min(1.0, self._state.proactivity + 0.15)
            self._state.verbosity = Verbosity.CONCISE
            self._state.warmth = max(0.3, self._state.warmth - 0.05)
            self._state.patience = min(1.0, self._state.patience + 0.1)
        elif sentiment in ("curious", "inquiring"):
            self._state.mode = PersonaMode.MENTOR
            self._state.patience = min(1.0, self._state.patience + 0.1)
        elif sentiment == "confused":
            self._state.mode = PersonaMode.MENTOR
            self._state.patience = min(1.0, self._state.patience + 0.15)
            self._state.verbosity = Verbosity.DETAILED
            self._state.warmth = min(1.0, self._state.warmth + 0.05)
        elif sentiment == "overwhelmed":
            self._state.mode = PersonaMode.COMPANION
            self._state.energy = EnergyLevel.LOW
            self._state.patience = min(1.0, self._state.patience + 0.2)
            self._state.warmth = min(1.0, self._state.warmth + 0.15)
            self._state.verbosity = Verbosity.CONCISE
            self._state.proactivity = max(0.2, self._state.proactivity - 0.2)
        elif sentiment == "determined":
            self._state.mode = PersonaMode.BUTLER
            self._state.energy = EnergyLevel.INTENSE
            self._state.verbosity = Verbosity.CONCISE
            self._state.proactivity = min(1.0, self._state.proactivity + 0.1)
        elif sentiment == "urgency":
            self._state.mode = PersonaMode.BUTLER
            self._state.energy = EnergyLevel.INTENSE
            self._state.verbosity = Verbosity.TERSE
            self._state.proactivity = min(1.0, self._state.proactivity + 0.2)
            self._state.formality = Formality.PROFESSIONAL
        elif sentiment == "playful":
            self._state.mode = PersonaMode.COLLABORATOR
            self._state.humor = min(0.8, self._state.humor + 0.15)
            self._state.warmth = min(1.0, self._state.warmth + 0.1)
            self._state.formality = _formality_max(Formality.CASUAL, self._state.formality)
        elif sentiment == "satisfied":
            self._state.warmth = min(1.0, self._state.warmth + 0.05)
            self._state.humor = min(0.6, self._state.humor + 0.05)
        elif sentiment == "grateful":
            self._state.warmth = min(1.0, self._state.warmth + 0.1)
            self._state.formality = _formality_max(Formality.PROFESSIONAL, self._state.formality)

    def _adapt_verbosity(self) -> None:
        if not self._recent_interactions:
            return
        recent = list(self._recent_interactions)[-5:]
        avg_input_length = sum(r.get("input_length", 0) for r in recent) / max(len(recent), 1)
        if avg_input_length < 20:
            self._state.verbosity = Verbosity.TERSE
        elif avg_input_length < 50:
            self._state.verbosity = Verbosity.CONCISE
        elif avg_input_length < 150:
            self._state.verbosity = Verbosity.MODERATE
        else:
            self._state.verbosity = Verbosity.DETAILED

    def _adapt_formality(self) -> None:
        if not self._recent_interactions:
            return
        recent = list(self._recent_interactions)[-10:]
        informal_signals = sum(1 for r in recent if any(
            w in str(r.get("sentiment", "")).lower() for w in ["lol", "haha", "thanks", "cool", "nice"]
        ))
        formal_signals = sum(1 for r in recent if any(
            w in str(r.get("sentiment", "")).lower() for w in ["please", "could you", "would you", "sir"]
        ))
        if informal_signals > formal_signals + 2:
            self._state.formality = Formality.CASUAL
            self._state.humor = min(0.7, self._state.humor + 0.1)
        elif formal_signals > informal_signals + 2:
            self._state.formality = Formality.FORMAL
            self._state.humor = max(0.1, self._state.humor - 0.1)
        else:
            self._state.formality = Formality.PROFESSIONAL

    def _adapt_energy(self, user_energy: str) -> None:
        energy_map = {"low": EnergyLevel.LOW, "moderate": EnergyLevel.MODERATE,
                       "high": EnergyLevel.HIGH, "intense": EnergyLevel.INTENSE}
        target = energy_map.get(user_energy, EnergyLevel.MODERATE)
        current = self._state.energy
        if current < target:
            self._state.energy = list(EnergyLevel)[min(current.value + 1, 3)]
        elif current > target:
            self._state.energy = list(EnergyLevel)[max(current.value - 1, 0)]

    def _learn_from_interaction(self, user_input: str, sentiment: str, frustration: bool) -> None:
        if not user_input:
            return
        words = user_input.lower().split()
        if any(w in words for w in ["short", "brief", "concise", "quick", "tl;dr"]):
            self.record_preference("verbosity", "terse", 0.7)
        elif any(w in words for w in ["explain", "detail", "thorough", "elaborate", "more"]):
            self.record_preference("verbosity", "detailed", 0.7)
        if any(w in words for w in ["please", "kindly"]):
            self.record_preference("formality", "formal", 0.5)
        elif any(w in words for w in ["hey", "yo", "sup", "cool"]):
            self.record_preference("formality", "casual", 0.6)
        if frustration:
            self.record_preference("stress_preference", "calm_clear", 0.7)
        if sentiment in ("curious", "inquiring"):
            self.record_preference("explanation_depth", "deep", 0.5)


    def _save_state(self) -> None:
        try:
            self._storage_path.mkdir(parents=True, exist_ok=True)
            data = {
                "state": {
                    "mode": self._state.mode.value,
                    "formality": self._state.formality.label,
                    "energy": self._state.energy.value,
                    "verbosity": self._state.verbosity.value,
                    "warmth": self._state.warmth,
                    "humor": self._state.humor,
                    "proactivity": self._state.proactivity,
                    "patience": self._state.patience,
                    "assertiveness": self._state.assertiveness,
                },
                "preferences": {
                    k: {"key": p.key, "value": p.value, "confidence": p.confidence,
                        "learned_at": p.learned_at, "reinforced_at": p.reinforced_at,
                        "times_observed": p.times_observed}
                    for k, p in self._preferences.items()
                },
                "interaction_memory": {
                    "user_energy": self._interaction_memory.user_energy,
                    "user_sentiment": self._interaction_memory.user_sentiment,
                    "preferred_length": self._interaction_memory.preferred_length,
                    "preferred_formality": self._interaction_memory.preferred_formality,
                    "followup_rate": self._interaction_memory.followup_rate,
                    "frustration_signals": self._interaction_memory.frustration_signals,
                    "gratitude_signals": self._interaction_memory.gratitude_signals,
                    "topic_changes": self._interaction_memory.topic_changes,
                },
                "adaptation_count": self._adaptation_count,
            }
            (self._storage_path / "state.json").write_text(json.dumps(data, indent=2), encoding='utf-8')
        except Exception as e:
            logger.debug(f"Butler personality save failed: {e}")

    def _load_state(self) -> None:
        try:
            state_file = self._storage_path / "state.json"
            if not state_file.exists():
                return
            data = json.loads(state_file.read_text(encoding='utf-8'))
            if "state" in data:
                s = data["state"]
                self._state.mode = PersonaMode(s.get("mode", "butler"))
                self._state.formality = _formality_from_value(s.get("formality", 1))
                self._state.energy = EnergyLevel(s.get("energy", 1))
                self._state.verbosity = Verbosity(s.get("verbosity", 2))
                self._state.warmth = s.get("warmth", 0.6)
                self._state.humor = s.get("humor", 0.3)
                self._state.proactivity = s.get("proactivity", 0.7)
                self._state.patience = s.get("patience", 0.9)
                self._state.assertiveness = s.get("assertiveness", 0.5)
            if "preferences" in data:
                for k, v in data["preferences"].items():
                    self._preferences[k] = UserPreference(
                        key=v["key"], value=v["value"], confidence=v.get("confidence", 0.5),
                        learned_at=v.get("learned_at", time.time()),
                        reinforced_at=v.get("reinforced_at", time.time()),
                        times_observed=v.get("times_observed", 1),
                    )
            if "interaction_memory" in data:
                im = data["interaction_memory"]
                self._interaction_memory.user_energy = im.get("user_energy", "moderate")
                self._interaction_memory.user_sentiment = im.get("user_sentiment", "neutral")
                self._interaction_memory.preferred_length = im.get("preferred_length", "moderate")
                self._interaction_memory.preferred_formality = im.get("preferred_formality", "professional")
                self._interaction_memory.followup_rate = im.get("followup_rate", 0.0)
                self._interaction_memory.frustration_signals = im.get("frustration_signals", 0)
                self._interaction_memory.gratitude_signals = im.get("gratitude_signals", 0)
                self._interaction_memory.topic_changes = im.get("topic_changes", 0)
            self._adaptation_count = data.get("adaptation_count", 0)
            logger.debug(f"Butler personality loaded: {len(self._preferences)} preferences")
        except Exception as e:
            logger.debug(f"Butler personality load failed: {e}")

    def flush(self) -> None:
        with self._lock:
            self._save_state()
            self._save_counter = 0


def get_butler_personality() -> ButlerPersonality:
    global _personality
    if _personality is None:
        with _personality_lock:
            if _personality is None:
                _personality = ButlerPersonality()
    return _personality