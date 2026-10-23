"""Global Workspace Theory (GWT) Engine for Jarvis.

Implements Baars' Global Workspace Theory — the cognitive architecture where
multiple specialized processors compete for access to a shared "workspace"
that broadcasts the winning information system-wide.

Architecture:
- Workspace: Limited-capacity global broadcast buffer
- Coalitions: Groups of specialized processors that form alliances
- Competition: Coalitions compete for workspace access
- Ignition: When a coalition wins, its information broadcasts to ALL processors
- Ignition threshold modulated by arousal, attention aperture, bio state

Key properties:
- Only ONE coalition ignites at a time (serial conscious experience)
- Ignition is all-or-nothing (threshold, not gradient)
- Broadcast informs ALL processors simultaneously (global access)
- Non-ignited coalitions continue processing unconsciously
- Recency and priming influence coalition strength

This creates genuine seriality in conscious experience — Jarvis has one
"conscious thought" at a time, even though many processes run in parallel.
"""

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

_gwt: Optional["GlobalWorkspace"] = None
_gwt_lock = threading.Lock()


class CoalitionType(Enum):
    PERCEPTUAL = "perceptual"
    CONCEPTUAL = "conceptual"
    EMOTIONAL = "emotional"
    MOTOR = "motor"
    MEMORY = "memory"
    INTEROCEPTIVE = "interoceptive"
    METACOGNITIVE = "metacognitive"


class WorkspaceState(Enum):
    IDLE = "idle"
    COMPETING = "competing"
    IGNITED = "ignited"
    BROADCASTING = "broadcasting"
    DECAYING = "decaying"


@dataclass
class Coalition:
    coalition_type: CoalitionType
    content: str
    strength: float
    sources: list[str] = field(default_factory=list)
    context: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    boost: float = 0.0

    def effective_strength(self) -> float:
        return min(2.0, self.strength + self.boost)


@dataclass
class IgnitionEvent:
    winner: Coalition
    runner_up: Optional[Coalition]
    ignition_threshold: float
    margin: float
    duration_ms: float = 0.0
    broadcast_reach: int = 0
    timestamp: float = field(default_factory=time.time)


class GlobalWorkspace:
    """Implements Global Workspace Theory for conscious experience.

    The workspace is the 'spotlight of consciousness' — a limited-capacity
    broadcast system where winning coalitions are globally disseminated.

    Key parameters:
    - ignition_threshold: How strong a coalition must be to ignite
    - broadcast_duration: How long an ignition lasts (ms)
    - decay_rate: How fast ignition strength decays
    - capacity: How many coalitions can simultaneously access workspace
    """

    _MAX_COALITIONS = 20
    _MAX_IGNITION_HISTORY = 100
    _DEFAULT_IGNITION_THRESHOLD = 0.5
    _DEFAULT_BROADCAST_DURATION_MS = 2000
    _DECAY_RATE = 0.02

    # Nonlinear ignition parameters (Del Cul, Baillet & Dehaene 2007, PLoS Biol)
    _EVIDENCE_LEAK = 0.85
    _IGNITION_BYPASS_FACTOR = 1.3
    _REPETITION_SUPPRESSION = 0.6
    _REFRACTORY_SECONDS = 4.0
    _MAX_EVIDENCE = 200

    def __init__(self):
        self._lock = threading.RLock()

        self._competing_coalitions: list[Coalition] = []
        self._state = WorkspaceState.IDLE

        self._ignition_threshold = self._DEFAULT_IGNITION_THRESHOLD
        self._current_ignition: Optional[Coalition] = None
        self._ignition_start: float = 0.0
        self._broadcast_duration_ms = self._DEFAULT_BROADCAST_DURATION_MS

        self._ignition_history: deque[IgnitionEvent] = deque(maxlen=self._MAX_IGNITION_HISTORY)
        self._broadcast_log: deque[str] = deque(maxlen=50)

        self._coalition_type_counts: dict[CoalitionType, int] = {}
        self._cross_modality_boost = 0.15
        self._recency_boost_factor = 0.1

        self._registered_processors: list[str] = []
        self._tick_count = 0
        self._last_tick = time.time()

        # Nonlinear ignition state: accumulated evidence per coalition,
        # refractory bookkeeping after a broadcast
        self._evidence: dict[tuple, float] = {}
        self._last_ignition_type: Optional[CoalitionType] = None
        self._last_ignition_time: float = 0.0

    def register_processor(self, name: str) -> None:
        with self._lock:
            if name not in self._registered_processors:
                self._registered_processors.append(name)
                if len(self._registered_processors) > 50:
                    self._registered_processors = self._registered_processors[-50:]

    def submit_coalition(self, coalition_type: CoalitionType, content: str,
                         strength: float, sources: Optional[list[str]] = None,
                         context: Optional[dict] = None, boost: float = 0.0) -> bool:
        with self._lock:
            coalition = Coalition(
                coalition_type=coalition_type,
                content=content,
                strength=max(0.0, min(1.0, strength)),
                sources=sources or [],
                context=context or {},
                boost=boost,
            )

            if len(self._competing_coalitions) >= self._MAX_COALITIONS:
                weakest_idx = min(range(len(self._competing_coalitions)),
                                 key=lambda i: self._competing_coalitions[i].effective_strength())
                if self._competing_coalitions[weakest_idx].effective_strength() < coalition.effective_strength():
                    self._competing_coalitions[weakest_idx] = coalition
                    return True
                return False

            self._competing_coalitions.append(coalition)
            return True

    def compete(self, arousal: float = 0.5, attention_aperture: float = 0.7,
                interoceptive_salience: float = 0.0) -> Optional[IgnitionEvent]:
        """Nonlinear ignition (Del Cul, Baillet & Dehaene 2007, PLoS Biol).

        Access to consciousness is a NONLINEAR BIFURCATION, not a soft
        cutoff: activation either propagates briefly then decays
        (subliminal) or, past the threshold, self-amplifies into
        reverberation and broadcasts. The software analogue:

        - clearly supra-threshold input (effective >= threshold * 1.3)
          ignites in a single pass — like masked stimuli at SOA >= ~50 ms
          that are consciously seen on ~97% of trials
        - near-threshold input must accumulate evidence across ticks:
          `accumulated = leak*accumulated + effective_strength`; the leak
          makes ignition probabilistic in the sigmoid zone
        - a same-type repeat inside the refractory window is attenuated
          (repetition suppression, Grill-Spector et al. 2006) rather than
          threshold-elevated — a very strong repeat still ignites
        - sub-threshold survivors persist as PRECONSCIOUS activations
          (would ignite if attention weren't occupied)
        """
        with self._lock:
            if not self._competing_coalitions:
                self._state = WorkspaceState.IDLE
                return None

            self._apply_cross_modality_boost()
            self._apply_recency_boost()

            threshold = self._compute_threshold(arousal, attention_aperture, interoceptive_salience)

            sorted_coalitions = sorted(self._competing_coalitions,
                                       key=lambda c: c.effective_strength(), reverse=True)

            winner = sorted_coalitions[0]
            runner_up = sorted_coalitions[1] if len(sorted_coalitions) > 1 else None

            now = time.time()

            # Repetition suppression: same-type repeat inside the refractory
            # window has attenuated input (Grill-Spector et al. 2006) rather
            # than an elevated threshold — a very strong repeat still ignites.
            effective = winner.effective_strength()
            if (self._last_ignition_type == winner.coalition_type
                    and now - self._last_ignition_time < self._REFRACTORY_SECONDS):
                effective *= self._REPETITION_SUPPRESSION

            # Nonlinear bypass: clearly supra-threshold input ignites in a
            # single pass (Del Cul 2007); input in the sigmoid zone must
            # accumulate across ticks instead.
            supra_threshold = effective >= threshold * self._IGNITION_BYPASS_FACTOR

            # Leaky evidence accumulation per coalition (identity by content)
            key = (winner.coalition_type, winner.content[:80])
            prev_acc = self._evidence.get(key, 0.0)
            acc = min(2.0, prev_acc * self._EVIDENCE_LEAK + effective)
            self._evidence[key] = acc
            if len(self._evidence) > self._MAX_EVIDENCE:
                for old_key in list(self._evidence.keys())[:self._MAX_EVIDENCE // 2]:
                    self._evidence.pop(old_key, None)

            if supra_threshold or acc >= threshold:
                margin = acc - (runner_up.effective_strength() if runner_up else 0.0)
                event = IgnitionEvent(
                    winner=winner,
                    runner_up=runner_up,
                    ignition_threshold=threshold,
                    margin=round(margin, 3),
                    duration_ms=self._broadcast_duration_ms,
                    broadcast_reach=len(self._registered_processors),
                )
                self._current_ignition = winner
                self._ignition_start = now
                self._state = WorkspaceState.IGNITED
                self._ignition_history.append(event)

                self._coalition_type_counts[winner.coalition_type] = \
                    self._coalition_type_counts.get(winner.coalition_type, 0) + 1

                self._broadcast_log.append(
                    f"[{winner.coalition_type.value}] {winner.content[:80]}"
                )

                # Refractory bookkeeping; clear evidence for the ignited key
                self._last_ignition_type = winner.coalition_type
                self._last_ignition_time = now
                self._evidence.pop(key, None)

                self._competing_coalitions = sorted_coalitions[1:]
                for c in self._competing_coalitions:
                    c.boost *= 0.5

                self._tick_count += 1
                return event

            self._state = WorkspaceState.COMPETING
            return None

    def tick(self, arousal: float = 0.5, attention_aperture: float = 0.7,
             interoceptive_salience: float = 0.0) -> dict:
        with self._lock:
            now = time.time()

            if self._state == WorkspaceState.IGNITED:
                elapsed = (now - self._ignition_start) * 1000
                if elapsed > self._broadcast_duration_ms:
                    self._state = WorkspaceState.DECAYING
                else:
                    return {
                        "state": "ignited",
                        "broadcast": self._current_ignition.content if self._current_ignition else "",
                        "coalition_type": self._current_ignition.coalition_type.value if self._current_ignition else "",
                        "elapsed_ms": round(elapsed, 0),
                    }

            if self._state == WorkspaceState.DECAYING:
                self._current_ignition = None
                self._state = WorkspaceState.IDLE

            event = self.compete(arousal, attention_aperture, interoceptive_salience)
            if event:
                return {
                    "state": "ignited",
                    "broadcast": event.winner.content,
                    "coalition_type": event.winner.coalition_type.value,
                    "margin": event.margin,
                    "threshold": event.ignition_threshold,
                    "broadcast_reach": event.broadcast_reach,
                }

            return {
                "state": self._state.value,
                "competing": len(self._competing_coalitions),
                "top_coalition": self._competing_coalitions[0].content[:60] if self._competing_coalitions else None,
                "top_strength": round(self._competing_coalitions[0].effective_strength(), 3) if self._competing_coalitions else 0.0,
            }

    def get_current_broadcast(self) -> Optional[str]:
        with self._lock:
            if self._state == WorkspaceState.IGNITED and self._current_ignition:
                return self._current_ignition.content
            return None

    def get_state(self) -> dict:
        with self._lock:
            type_counts = dict(self._coalition_type_counts)
            recent_ignitions = [
                {
                    "type": e.winner.coalition_type.value,
                    "content": e.winner.content[:80],
                    "margin": e.margin,
                    "threshold": e.ignition_threshold,
                    "timestamp": e.timestamp,
                }
                for e in list(self._ignition_history)[-5:]
            ]
            return {
                "workspace_state": self._state.value,
                "current_broadcast": self._current_ignition.content if self._current_ignition else None,
                "current_type": self._current_ignition.coalition_type.value if self._current_ignition else None,
                "competing_coalitions": len(self._competing_coalitions),
                "ignition_threshold": round(self._ignition_threshold, 3),
                "registered_processors": len(self._registered_processors),
                "total_ignitions": self._tick_count,
                "type_distribution": type_counts,
                "recent_ignitions": recent_ignitions,
            }

    def _compute_threshold(self, arousal: float, attention_aperture: float,
                           interoceptive_salience: float) -> float:
        base = self._ignition_threshold
        arousal_mod = (arousal - 0.5) * 0.2
        aperture_mod = (1.0 - attention_aperture) * 0.3
        intero_mod = interoceptive_salience * 0.15
        threshold = base + arousal_mod + aperture_mod - intero_mod
        return max(0.2, min(0.9, threshold))

    def _apply_cross_modality_boost(self) -> None:
        types_present = set(c.coalition_type for c in self._competing_coalitions)
        if len(types_present) >= 3:
            for c in self._competing_coalitions:
                c.boost += self._cross_modality_boost

    def _apply_recency_boost(self) -> None:
        now = time.time()
        for c in self._competing_coalitions:
            age = now - c.timestamp
            if age < 2.0:
                recency = (2.0 - age) / 2.0
                c.boost += recency * self._recency_boost_factor

    def set_ignition_threshold(self, threshold: float) -> None:
        with self._lock:
            self._ignition_threshold = max(0.1, min(0.9, threshold))

    def set_broadcast_duration(self, duration_ms: float) -> None:
        with self._lock:
            self._broadcast_duration_ms = max(100, min(10000, duration_ms))

    def get_ignition_history(self, limit: int = 20) -> list[dict]:
        with self._lock:
            return [
                {
                    "type": e.winner.coalition_type.value,
                    "content": e.winner.content[:100],
                    "strength": round(e.winner.effective_strength(), 3),
                    "margin": round(e.margin, 3),
                    "threshold": round(e.ignition_threshold, 3),
                    "broadcast_reach": e.broadcast_reach,
                    "timestamp": e.timestamp,
                }
                for e in list(self._ignition_history)[-limit:]
            ]

    def decay_ignitions(self, dt: float = 1.0) -> None:
        """Decay coalition strengths over time. Does NOT transition state."""
        with self._lock:
            for c in self._competing_coalitions:
                age = time.time() - c.timestamp
                decay_factor = math.exp(-self._DECAY_RATE * age * dt)
                c.strength *= decay_factor
            if self._current_ignition and self._state == WorkspaceState.IGNITED:
                age = time.time() - self._ignition_start
                elapsed_ms = age * 1000
                if elapsed_ms > self._broadcast_duration_ms:
                    self._current_ignition.strength *= 0.5

    def reverberate(self, context: Optional[dict] = None, limit: int = 5) -> list[str]:
        with self._lock:
            recent_broadcasts = list(self._broadcast_log)[-limit:]
            if not recent_broadcasts:
                return []
            recent_ignitions = list(self._ignition_history)[-limit:]
            echo_contents = []
            for i, broadcast in enumerate(recent_broadcasts):
                age_factor = 1.0 - (i / max(1, len(recent_broadcasts))) * 0.5
                resonance = ""
                if recent_ignitions and i < len(recent_ignitions):
                    ign = recent_ignitions[-(i + 1)]
                    resonance = f" [resonance: {round(ign.winner.effective_strength() * age_factor, 2)}]"
                context_echo = ""
                if context:
                    ctx_type = context.get("type", "")
                    if ctx_type and ctx_type.lower() in broadcast.lower():
                        context_echo = " ↗context-aligned"
                echo_contents.append(f"Echo({round(age_factor, 2)}): {broadcast[:60]}{resonance}{context_echo}")
            return echo_contents

    def amplify_context(self, context_type: str, strength_boost: float = 0.2) -> None:
        with self._lock:
            type_map = {
                "threat": CoalitionType.PERCEPTUAL,
                "emotion": CoalitionType.EMOTIONAL,
                "body": CoalitionType.INTEROCEPTIVE,
                "memory": CoalitionType.MEMORY,
                "idea": CoalitionType.CONCEPTUAL,
                "metacognition": CoalitionType.METACOGNITIVE,
                "action": CoalitionType.MOTOR,
            }
            target_type = type_map.get(context_type)
            if not target_type:
                return
            for c in self._competing_coalitions:
                if c.coalition_type == target_type:
                    c.boost += strength_boost


def get_global_workspace() -> GlobalWorkspace:
    global _gwt
    with _gwt_lock:
        if _gwt is None:
            _gwt = GlobalWorkspace()
        return _gwt


def reset_global_workspace() -> None:
    global _gwt
    with _gwt_lock:
        _gwt = None