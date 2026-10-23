"""Cognitive Load Balancer - Distribute cognitive load across subsystems.

Monitors resource usage across all cognitive subsystems and rebalances
work to prevent overload, starvation, or resource contention.

Key features:
1. Per-subsystem load tracking with weighted priority
2. Dynamic throttling when a subsystem approaches capacity
3. Work stealing from overloaded subsystems
4. Cooperative scheduling between Erlős, Daydream, Evolution, and PFC
5. Adaptive batch sizing based on system throughput
"""
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any

logger = logging.getLogger(__name__)


class SubsystemType(Enum):
    ERDOS = auto()
    DAYDREAM = auto()
    EVOLUTION = auto()
    PFC = auto()
    LEARNING = auto()
    MEMORY = auto()
    DEFENSE = auto()
    CONTEXT = auto()


class LoadLevel(Enum):
    IDLE = 0
    LIGHT = 1
    MODERATE = 2
    HEAVY = 3
    CRITICAL = 4


@dataclass
class SubsystemLoad:
    subsystem: SubsystemType
    cpu_estimate: float = 0.0
    memory_estimate_mb: float = 0.0
    pending_tasks: int = 0
    active_tasks: int = 0
    completed_tasks: int = 0
    avg_task_duration: float = 0.0
    last_activity: datetime = field(default_factory=datetime.now)
    throttled: bool = False
    throttle_level: float = 0.0

    @property
    def load_score(self) -> float:
        base = self.active_tasks * 0.3 + self.pending_tasks * 0.2
        cpu_factor = min(1.0, self.cpu_estimate / 100.0)
        mem_factor = min(1.0, self.memory_estimate_mb / 500.0)
        throttle_penalty = self.throttle_level * 0.5
        return min(4.0, base + cpu_factor + mem_factor + throttle_penalty)

    @property
    def load_level(self) -> LoadLevel:
        score = self.load_score
        if score < 0.5:
            return LoadLevel.IDLE
        elif score < 1.5:
            return LoadLevel.LIGHT
        elif score < 2.5:
            return LoadLevel.MODERATE
        elif score < 3.5:
            return LoadLevel.HEAVY
        else:
            return LoadLevel.CRITICAL


@dataclass
class RebalanceAction:
    action_type: str
    source: SubsystemType | None = None
    target: SubsystemType | None = None
    description: str = ""
    applied_at: datetime = field(default_factory=datetime.now)


_SUBSYSTEM_WEIGHTS = {
    SubsystemType.ERDOS: 0.8,
    SubsystemType.DAYDREAM: 0.6,
    SubsystemType.EVOLUTION: 0.7,
    SubsystemType.PFC: 1.0,
    SubsystemType.LEARNING: 0.7,
    SubsystemType.MEMORY: 0.5,
    SubsystemType.DEFENSE: 0.9,
    SubsystemType.CONTEXT: 0.6,
}

_SUBSYSTEM_CAPS = {
    SubsystemType.ERDOS: 3.0,
    SubsystemType.DAYDREAM: 2.5,
    SubsystemType.EVOLUTION: 2.5,
    SubsystemType.PFC: 3.5,
    SubsystemType.LEARNING: 2.5,
    SubsystemType.MEMORY: 2.0,
    SubsystemType.DEFENSE: 3.0,
    SubsystemType.CONTEXT: 2.0,
}

_THROTTLE_THRESHOLDS = {
    LoadLevel.MODERATE: 0.7,
    LoadLevel.HEAVY: 0.4,
    LoadLevel.CRITICAL: 0.1,
}


class CognitiveLoadBalancer:
    """Monitors and balances load across cognitive subsystems."""

    _CHECK_INTERVAL = 10
    _HISTORY_LENGTH = 100
    _MAX_REBALANCE_ACTIONS = 200
    _REBALANCE_COOLDOWN = 30

    def __init__(self):
        self._lock = threading.RLock()
        self._running = False
        self._thread: threading.Thread | None = None

        self._subsystem_loads: dict[SubsystemType, SubsystemLoad] = {
            st: SubsystemLoad(subsystem=st) for st in SubsystemType
        }
        self._load_history: deque[dict] = deque(maxlen=self._HISTORY_LENGTH)
        self._rebalance_actions: deque[RebalanceAction] = deque(maxlen=self._MAX_REBALANCE_ACTIONS)
        self._last_rebalance: datetime | None = None
        self._global_throttle: float = 1.0

        self._subsystem_refs: dict[str, Any] = {}

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._monitoring_loop,
            name="cognitive-load-balancer",
            daemon=True,
        )
        self._thread.start()
        logger.info("Cognitive load balancer started")

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        logger.info("Cognitive load balancer stopped")

    def _monitoring_loop(self):
        while self._running:
            try:
                time.sleep(self._CHECK_INTERVAL)
                if not self._running:
                    break
                self._collect_subsystem_metrics()
                self._record_snapshot()
                self._check_and_rebalance()
            except Exception as e:
                logger.warning(f"Load balancer error: {e}")
                time.sleep(30)

    def _collect_subsystem_metrics(self):
        for st in SubsystemType:
            try:
                self._update_subsystem_metrics(st)
            except Exception as e:
                logger.debug(f"Failed to collect metrics for {st.name}: {e}")

    def _update_subsystem_metrics(self, st: SubsystemType):
        with self._lock:
            load = self._subsystem_loads[st]
            load.last_activity = datetime.now()

        if st == SubsystemType.ERDOS:
            try:
                from core.erdos_engine import get_erdos_engine
                engine = get_erdos_engine()
                status = engine.get_status()
                with self._lock:
                    load.active_tasks = status.get("current_problems", 0)
                    load.pending_tasks = status.get("insights_count", 0)
                    load.completed_tasks = status.get("solved_count", 0)
                    load.throttled = status.get("throttled", False)
            except Exception as e:
                logger.debug(f"Failed to collect ERDOS metrics: {e}")

        elif st == SubsystemType.DAYDREAM:
            try:
                from core.daydream_engine import get_daydream_engine
                engine = get_daydream_engine()
                status = engine.get_status()
                with self._lock:
                    load.active_tasks = status.get("active_dreams", 0)
                    load.pending_tasks = status.get("pending_tasks", 0)
                    load.completed_tasks = status.get("completed_actions", 0)
            except Exception as e:
                logger.debug(f"Failed to collect DAYDREAM metrics: {e}")

        elif st == SubsystemType.EVOLUTION:
            try:
                from core.evolution_orchestrator import get_evolution_orchestrator
                orch = get_evolution_orchestrator()
                status = orch.get_status()
                with self._lock:
                    load.active_tasks = status.get("active_genes", 0)
                    load.pending_tasks = status.get("pending_proposals", 0)
                    load.completed_tasks = status.get("total_evolutions", 0)
            except Exception as e:
                logger.debug(f"Failed to collect EVOLUTION metrics: {e}")

        elif st == SubsystemType.PFC:
            try:
                from core.prefrontal_cortex import get_prefrontal_cortex
                pfc = get_prefrontal_cortex()
                stats = pfc.get_stats()
                with self._lock:
                    load.active_tasks = stats.get("active_concepts", 0)
                    load.pending_tasks = stats.get("pending_observations", 0)
                    load.completed_tasks = stats.get("total_observations", 0)
            except Exception as e:
                logger.debug(f"Failed to collect PFC metrics: {e}")

    def _record_snapshot(self):
        with self._lock:
            snapshot = {
                "timestamp": datetime.now().isoformat(),
                "global_throttle": self._global_throttle,
                "subsystems": {},
            }
            for st, load in self._subsystem_loads.items():
                snapshot["subsystems"][st.name] = {
                    "load_score": round(load.load_score, 2),
                    "load_level": load.load_level.name,
                    "active_tasks": load.active_tasks,
                    "pending_tasks": load.pending_tasks,
                    "throttled": load.throttled,
                }
            self._load_history.append(snapshot)

    def _check_and_rebalance(self):
        if self._last_rebalance:
            elapsed = (datetime.now() - self._last_rebalance).total_seconds()
            if elapsed < self._REBALANCE_COOLDOWN:
                return

        overloaded = []
        underloaded = []

        with self._lock:
            for st, load in self._subsystem_loads.items():
                level = load.load_level
                if level in (LoadLevel.HEAVY, LoadLevel.CRITICAL):
                    overloaded.append((st, load))
                elif level in (LoadLevel.IDLE, LoadLevel.LIGHT):
                    underloaded.append((st, load))

        any_critical = any(
            load.load_level == LoadLevel.CRITICAL for _, load in overloaded
        )

        if any_critical:
            self._global_throttle = max(0.1, self._global_throttle * 0.8)
            action = RebalanceAction(
                action_type="global_throttle",
                description=f"Reduced global throttle to {self._global_throttle:.2f} due to critical load",
            )
            with self._lock:
                self._rebalance_actions.append(action)
            logger.warning(f"Critical load detected, global throttle: {self._global_throttle:.2f}")
        else:
            self._global_throttle = min(1.0, self._global_throttle * 1.05)

            for st, load in overloaded:
                cap = _SUBSYSTEM_CAPS.get(st, 3.0)
                if load.load_score > cap:
                    throttle = _THROTTLE_THRESHOLDS.get(load.load_level, 1.0)
                    with self._lock:
                        load.throttled = True
                        load.throttle_level = 1.0 - throttle
                    action = RebalanceAction(
                        action_type="throttle",
                        target=st,
                        description=f"Throttled {st.name} to {throttle:.0%} capacity",
                    )
                    with self._lock:
                        self._rebalance_actions.append(action)

        self._last_rebalance = datetime.now()

    def get_subsystem_throttle(self, subsystem: SubsystemType) -> float:
        with self._lock:
            load = self._subsystem_loads.get(subsystem)
            if load is None:
                return self._global_throttle
            if load.throttled:
                base_throttle = 1.0 - load.throttle_level
                return base_throttle * self._global_throttle
            return self._global_throttle

    def should_run_task(self, subsystem: SubsystemType, estimate_cost: float = 1.0) -> bool:
        throttle = self.get_subsystem_throttle(subsystem)
        with self._lock:
            load = self._subsystem_loads.get(subsystem)
            if load is None:
                return True
            cap = _SUBSYSTEM_CAPS.get(subsystem, 3.0)
            if load.load_score + estimate_cost > cap:
                return False
        return throttle >= 0.3

    def get_optimal_batch_size(self, subsystem: SubsystemType, max_batch: int = 10) -> int:
        throttle = self.get_subsystem_throttle(subsystem)
        with self._lock:
            load = self._subsystem_loads.get(subsystem)
            if load is None:
                return max_batch
            score = load.load_score
        available_capacity = max(0, _SUBSYSTEM_CAPS.get(subsystem, 3.0) - score)
        batch = int(max(1, available_capacity * throttle * max_batch / 3.0))
        return min(batch, max_batch)

    def notify_task_start(self, subsystem: SubsystemType, task_cost: float = 1.0):
        with self._lock:
            load = self._subsystem_loads.get(subsystem)
            if load:
                load.active_tasks += 1
                load.cpu_estimate += task_cost * 10
                load.last_activity = datetime.now()

    def notify_task_complete(self, subsystem: SubsystemType, duration: float = 0.0):
        with self._lock:
            load = self._subsystem_loads.get(subsystem)
            if load:
                load.active_tasks = max(0, load.active_tasks - 1)
                load.completed_tasks += 1
                load.cpu_estimate = max(0, load.cpu_estimate - 5)
                if duration > 0:
                    if load.avg_task_duration == 0:
                        load.avg_task_duration = duration
                    else:
                        load.avg_task_duration = 0.9 * load.avg_task_duration + 0.1 * duration
                load.last_activity = datetime.now()

    def notify_task_fail(self, subsystem: SubsystemType):
        with self._lock:
            load = self._subsystem_loads.get(subsystem)
            if load:
                load.active_tasks = max(0, load.active_tasks - 1)
                load.cpu_estimate = max(0, load.cpu_estimate - 5)

    def get_load_summary(self) -> dict:
        with self._lock:
            subsystems = {}
            for st, load in self._subsystem_loads.items():
                subsystems[st.name] = {
                    "load_score": round(load.load_score, 2),
                    "load_level": load.load_level.name,
                    "active_tasks": load.active_tasks,
                    "pending_tasks": load.pending_tasks,
                    "completed_tasks": load.completed_tasks,
                    "throttled": load.throttled,
                    "throttle_level": round(load.throttle_level, 2),
                }

            return {
                "running": self._running,
                "global_throttle": round(self._global_throttle, 3),
                "subsystems": subsystems,
                "history_size": len(self._load_history),
                "rebalance_actions": len(self._rebalance_actions),
                "last_rebalance": (
                    self._last_rebalance.isoformat()
                    if self._last_rebalance
                    else None
                ),
                "overloaded": [
                    st.name
                    for st, load in self._subsystem_loads.items()
                    if load.load_level in (LoadLevel.HEAVY, LoadLevel.CRITICAL)
                ],
            }

    def get_rebalance_history(self, limit: int = 20) -> list[dict]:
        with self._lock:
            return [
                {
                    "action": a.action_type,
                    "source": a.source.name if a.source else None,
                    "target": a.target.name if a.target else None,
                    "description": a.description,
                    "applied_at": a.applied_at.isoformat(),
                }
                for a in list(self._rebalance_actions)[-limit:]
            ]

    def force_rebalance(self) -> dict:
        self._collect_subsystem_metrics()
        self._check_and_rebalance()
        return self.get_load_summary()

    def adjust_for_cognitive_state(self, cognitive_state_name: str, cognitive_load: float = 0.5) -> None:
        """Adjust global throttle based on Thought Sense cognitive state.

        Thought Sense detects the user's cognitive state (FLOW, ENGAGED, 
        OVERLOADED, FRUSTRATED, IDLE). The load balancer uses this to 
        decide how aggressively background tasks should run:
        
        - FLOW/ENGAGED: Full background processing (user is productive)
        - EXPLORING: Moderate background (user is curious)
        - IDLE: Maximum background processing (user is away)
        - OVERLOADED: Minimal background (user needs system resources)
        - FRUSTRATED: Almost no background (user needs all resources)
        """
        throttle_map = {
            "FLOW": 1.0,
            "ENGAGED": 0.9,
            "EXPLORING": 0.7,
            "DEEP_DIVE": 0.8,
            "IDLE": 1.0,
            "OVERLOADED": 0.3,
            "FRUSTRATED": 0.1,
        }

        target = throttle_map.get(cognitive_state_name, 0.7)

        load_modifier = max(0.5, 1.0 - cognitive_load * 0.5)
        target *= load_modifier

        with self._lock:
            self._global_throttle = self._global_throttle * 0.7 + target * 0.3
            self._global_throttle = max(0.1, min(1.0, self._global_throttle))

        action = RebalanceAction(
            action_type="cognitive_state_adjustment",
            description=f"Adjusted throttle to {self._global_throttle:.2f} for cognitive state {cognitive_state_name}",
        )
        with self._lock:
            self._rebalance_actions.append(action)


_load_balancer: CognitiveLoadBalancer | None = None
_lb_lock = threading.Lock()


def get_load_balancer() -> CognitiveLoadBalancer:
    global _load_balancer
    if _load_balancer is None:
        with _lb_lock:
            if _load_balancer is None:
                _load_balancer = CognitiveLoadBalancer()
    return _load_balancer