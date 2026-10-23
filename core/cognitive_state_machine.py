"""Cognitive State Machine for Jarvis.

Implements autonomous state transitions that make Jarvis feel *alive*:
- AWAKE: Normal interaction, high attention, fast responses
- CONTEMPLATING: Idle processing, background consolidation, creative exploration
- DREAMING: Deep memory consolidation, insight generation, pathway strengthening
- EVOLVING: Applying consolidated insights, self-improvement

State transitions are driven by:
- Idle time (no user input → contemplating → dreaming)
- Circadian phase (evening promotes dreaming, morning promotes evolution)
- Bio-integration signals (stress suppresses dreaming, high coherence promotes evolution)
- Prediction error (high surprise → more contemplation, low surprise → can dream)
- Intrinsic drives (curiosity → contemplating, mastery → evolving)

This is the "autonomous nervous system" that keeps Jarvis alive between interactions,
processing memories, consolidating learning, and generating spontaneous insights.
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

_cognitive_sm: Optional["CognitiveStateMachine"] = None
_csm_lock = threading.Lock()


class CognitiveState(Enum):
    AWAKE = "awake"
    CONTEMPLATING = "contemplating"
    DREAMING = "dreaming"
    EVOLVING = "evolving"


class TransitionTrigger(Enum):
    USER_INPUT = "user_input"
    IDLE_TIMEOUT = "idle_timeout"
    LOW_DEMAND = "low_demand"
    HIGH_SURPRISE = "high_surprise"
    LOW_SURPRISE = "low_surprise"
    HIGH_CURIOSITY = "high_curiosity"
    HIGH_MASTERY = "high_mastery"
    STRESS_RESPONSE = "stress_response"
    FATIGUE_GATE = "fatigue_gate"
    MANUAL = "manual"


@dataclass
class StateTransition:
    from_state: CognitiveState
    to_state: CognitiveState
    trigger: TransitionTrigger
    timestamp: float = field(default_factory=time.time)
    context: dict = field(default_factory=dict)


@dataclass
class CognitiveStateConfig:
    idle_to_contemplating_seconds: float = 30.0
    contemplating_to_dreaming_seconds: float = 300.0
    dreaming_to_evolving_seconds: float = 600.0
    max_dreaming_duration_seconds: float = 1800.0
    max_evolution_duration_seconds: float = 300.0
    stress_suppresses_dreaming: bool = True
    surprise_threshold_dream: float = 0.4
    curiosity_threshold_explore: float = 0.5
    mastery_threshold_evolve: float = 0.6


class CognitiveStateMachine:
    """Autonomous cognitive state machine that transitions between operational modes.

    The state machine drives the "inner life" of Jarvis:
    - When user is talking → AWAKE (full attention, fast processing)
    - When idle 30s → CONTEMPLATING (background processing, consolidation)
    - When idle 5min → DREAMING (deep consolidation, insight generation)
    - After productive dream → EVOLVING (apply insights)

    Transition conditions incorporate:
    - Time since last interaction (primary driver)
    - Circadian phase (gate dreaming intensity)
    - Bio state (stress suppresses dreaming, coherence promotes evolution)
    - Prediction error (surprise promotes contemplation, calm promotes dreaming)
    - Intrinsic drives (curiosity → contemplate, mastery → evolve)
    """

    TRANSITIONS = {
        (CognitiveState.AWAKE, CognitiveState.CONTEMPLATING): {
            TransitionTrigger.IDLE_TIMEOUT, TransitionTrigger.HIGH_CURIOSITY,
        },
        (CognitiveState.CONTEMPLATING, CognitiveState.AWAKE): {
            TransitionTrigger.USER_INPUT, TransitionTrigger.STRESS_RESPONSE,
        },
        (CognitiveState.CONTEMPLATING, CognitiveState.DREAMING): {
            TransitionTrigger.IDLE_TIMEOUT, TransitionTrigger.LOW_SURPRISE,
            TransitionTrigger.FATIGUE_GATE, TransitionTrigger.LOW_DEMAND,
        },
        (CognitiveState.DREAMING, CognitiveState.CONTEMPLATING): {
            TransitionTrigger.USER_INPUT, TransitionTrigger.HIGH_SURPRISE,
            TransitionTrigger.STRESS_RESPONSE,
        },
        # Direct waking edge: notify_user_input() requests DREAMING->AWAKE,
        # but this edge was missing, so _transition() rejected it and the
        # machine STUCK in DREAMING while the user actively chatted until
        # the timeout path fired.
        (CognitiveState.DREAMING, CognitiveState.AWAKE): {
            TransitionTrigger.USER_INPUT, TransitionTrigger.STRESS_RESPONSE,
        },
        (CognitiveState.DREAMING, CognitiveState.EVOLVING): {
            TransitionTrigger.HIGH_MASTERY, TransitionTrigger.FATIGUE_GATE,
            TransitionTrigger.IDLE_TIMEOUT,
        },
        (CognitiveState.EVOLVING, CognitiveState.AWAKE): {
            TransitionTrigger.USER_INPUT, TransitionTrigger.IDLE_TIMEOUT,
        },
        (CognitiveState.EVOLVING, CognitiveState.CONTEMPLATING): {
            TransitionTrigger.IDLE_TIMEOUT,
        },
    }

    def __init__(self, config: Optional[CognitiveStateConfig] = None):
        self.config = config or CognitiveStateConfig()
        self.state = CognitiveState.AWAKE
        self._last_user_input_time = time.time()
        self._state_entry_time = time.time()
        self._transition_history: deque[StateTransition] = deque(maxlen=100)
        self._lock = threading.RLock()

        self._idle_check_interval = 5.0
        self._last_idle_check = time.time()

        self._dreaming_accumulated_insights = 0
        self._exploration_count = 0
        self._evolution_applications = 0

    @property
    def current_state(self) -> CognitiveState:
        return self.state

    @property
    def idle_seconds(self) -> float:
        return time.time() - self._last_user_input_time

    @property
    def state_duration_seconds(self) -> float:
        return time.time() - self._state_entry_time

    def notify_user_input(self) -> CognitiveState:
        with self._lock:
            if self.state != CognitiveState.AWAKE:
                self._transition(CognitiveState.AWAKE, TransitionTrigger.USER_INPUT)
            self._last_user_input_time = time.time()
            return self.state

    def tick(self, bio_state: Optional[dict] = None,
             prediction_state: Optional[dict] = None) -> dict:
        with self._lock:
            now = time.time()
            idle = now - self._last_user_input_time

            transition = self._evaluate_transition(idle, bio_state, prediction_state)
            if transition:
                self._transition(transition[0], transition[1], transition[2])

            return self._get_state(bio_state, prediction_state)

    def _evaluate_transition(self, idle: float, bio_state: Optional[dict],
                            prediction_state: Optional[dict]) -> Optional[tuple]:
        operational_phase = None
        processing_fatigue = 0.0
        if bio_state:
            operational_phase = bio_state.get("operational_phase")
            processing_fatigue = bio_state.get("processing_fatigue", 0.0)

        if self.state == CognitiveState.AWAKE:
            if idle > self.config.idle_to_contemplating_seconds:
                curiosity = 0.5
                if prediction_state:
                    curiosity = prediction_state.get("drives", {}).get("curiosity", 0.5)
                if curiosity > self.config.curiosity_threshold_explore:
                    return (CognitiveState.CONTEMPLATING, TransitionTrigger.HIGH_CURIOSITY, {"curiosity": curiosity})
                return (CognitiveState.CONTEMPLATING, TransitionTrigger.IDLE_TIMEOUT, {"idle": idle})

        elif self.state == CognitiveState.CONTEMPLATING:
            if idle > self.config.contemplating_to_dreaming_seconds:
                stress_phase = "recovery"
                if bio_state:
                    stress_phase = bio_state.get("stress_phase", "recovery")
                if self.config.stress_suppresses_dreaming and stress_phase in ("alarm", "exhaustion"):
                    return None

                if operational_phase in ("deep_processing", "recovery_consolidation"):
                    return (CognitiveState.DREAMING, TransitionTrigger.FATIGUE_GATE,
                            {"operational_phase": operational_phase})

                surprise = 0.5
                if prediction_state:
                    surprise = prediction_state.get("errors", {}).get("abstract", {}).get("magnitude", 0.5)
                if surprise < self.config.surprise_threshold_dream:
                    return (CognitiveState.DREAMING, TransitionTrigger.LOW_SURPRISE, {"surprise": surprise})
                return (CognitiveState.DREAMING, TransitionTrigger.LOW_DEMAND, {"idle": idle})

            if processing_fatigue > 0.7 and operational_phase in ("recovery_consolidation",):
                return (CognitiveState.DREAMING, TransitionTrigger.FATIGUE_GATE,
                        {"fatigue": processing_fatigue, "operational_phase": operational_phase})

        elif self.state == CognitiveState.DREAMING:
            if self.state_duration_seconds > self.config.max_dreaming_duration_seconds:
                if operational_phase in ("peak_engagement",):
                    return (CognitiveState.AWAKE, TransitionTrigger.FATIGUE_GATE,
                            {"operational_phase": operational_phase})
                return (CognitiveState.EVOLVING, TransitionTrigger.IDLE_TIMEOUT, {})
            mastery = 0.5
            if prediction_state:
                mastery = prediction_state.get("drives", {}).get("mastery", 0.5)
            if mastery > self.config.mastery_threshold_evolve and self._dreaming_accumulated_insights > 0:
                return (CognitiveState.EVOLVING, TransitionTrigger.HIGH_MASTERY, {"mastery": mastery})

        elif self.state == CognitiveState.EVOLVING:
            if self.state_duration_seconds > self.config.max_evolution_duration_seconds:
                if operational_phase in ("deep_processing", "recovery_consolidation"):
                    return (CognitiveState.CONTEMPLATING, TransitionTrigger.FATIGUE_GATE,
                            {"operational_phase": operational_phase})
                return (CognitiveState.CONTEMPLATING, TransitionTrigger.IDLE_TIMEOUT, {})

        return None

    def _transition(self, new_state: CognitiveState, trigger: TransitionTrigger,
                    context: Optional[dict] = None) -> None:
        old_state = self.state
        valid_triggers = self.TRANSITIONS.get((old_state, new_state), set())
        if trigger not in valid_triggers and trigger != TransitionTrigger.MANUAL:
            logger.debug(f"Invalid transition: {old_state.value} → {new_state.value} via {trigger.value}")
            return

        self._transition_history.append(StateTransition(
            from_state=old_state, to_state=new_state, trigger=trigger, context=context or {},
        ))
        self.state = new_state
        self._state_entry_time = time.time()

        # Leaving DREAMING consumes the accumulated insights — without this
        # the HIGH_MASTERY gate fires instantly on every dream re-entry.
        if old_state == CognitiveState.DREAMING and new_state != CognitiveState.DREAMING:
            self._dreaming_accumulated_insights = 0

        if new_state == CognitiveState.CONTEMPLATING:
            self._exploration_count += 1
        elif new_state == CognitiveState.EVOLVING:
            self._evolution_applications += 1

        logger.info(f"Cognitive state: {old_state.value} → {new_state.value} (trigger: {trigger.value})")

    def force_state(self, state: CognitiveState, reason: str = "") -> None:
        with self._lock:
            self._transition(state, TransitionTrigger.MANUAL, {"reason": reason})

    def record_insight(self, count: int = 1) -> None:
        with self._lock:
            self._dreaming_accumulated_insights += count

    def should_daydream(self) -> bool:
        with self._lock:
            return self.state in (CognitiveState.CONTEMPLATING, CognitiveState.DREAMING)

    def should_consolidate(self) -> bool:
        with self._lock:
            return self.state == CognitiveState.DREAMING

    def should_evolve(self) -> bool:
        with self._lock:
            return self.state == CognitiveState.EVOLVING

    def get_daydream_depth(self) -> float:
        with self._lock:
            if self.state == CognitiveState.CONTEMPLATING:
                return min(1.0, self.state_duration_seconds / 60.0) * 0.3
            elif self.state == CognitiveState.DREAMING:
                return min(1.0, self.state_duration_seconds / 300.0) * 0.7 + 0.3
            elif self.state == CognitiveState.EVOLVING:
                return 0.9
            return 0.0

    def get_processing_depth(self) -> str:
        with self._lock:
            if self.state == CognitiveState.AWAKE:
                return "full"
            elif self.state == CognitiveState.CONTEMPLATING:
                return "moderate"
            elif self.state == CognitiveState.DREAMING:
                return "deep"
            elif self.state == CognitiveState.EVOLVING:
                return "focused"
            return "moderate"

    def _get_state(self, bio_state: Optional[dict] = None,
                   prediction_state: Optional[dict] = None) -> dict:
        return self.get_state()

    def get_state(self) -> dict:
        return {
            "state": self.state.value,
            "idle_seconds": round(self.idle_seconds, 1),
            "state_duration": round(self.state_duration_seconds, 1),
            "daydream_depth": round(self.get_daydream_depth(), 2),
            "processing_depth": self.get_processing_depth(),
            "should_daydream": self.should_daydream(),
            "should_consolidate": self.should_consolidate(),
            "should_evolve": self.should_evolve(),
            "exploration_count": self._exploration_count,
            "evolution_count": self._evolution_applications,
            "dreaming_insights": self._dreaming_accumulated_insights,
            "transition_count": len(self._transition_history),
            "last_transition": {
                "from": self._transition_history[-1].from_state.value,
                "to": self._transition_history[-1].to_state.value,
                "trigger": self._transition_history[-1].trigger.value,
                "context": self._transition_history[-1].context,
            } if self._transition_history else None,
        }


def get_cognitive_state_machine() -> CognitiveStateMachine:
    global _cognitive_sm
    with _csm_lock:
        if _cognitive_sm is None:
            _cognitive_sm = CognitiveStateMachine()
    return _cognitive_sm


def reset_cognitive_state_machine() -> None:
    global _cognitive_sm
    with _csm_lock:
        _cognitive_sm = None