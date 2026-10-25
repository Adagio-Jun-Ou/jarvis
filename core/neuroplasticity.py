"""Neuroplasticity Engine for Jarvis.

Implements biologically-inspired structural and functional adaptation:
- Long-Term Potentiation (LTP): Strengthening frequently-used neural pathways
- Long-Term Depression (LTD): Weakening rarely-used pathways
- Spike-Timing-Dependent Plasticity (STDP): Temporal correlation learning
- Synaptic pruning: Removing weak connections during rest cycles
- Structural plasticity: Growing new connections based on demand
- Myelination: Speed optimization for high-traffic pathways
- Critical periods: Heightened plasticity during early learning phases
- Consolidation during rest: Memory replay and pathway reinforcement

This module enables Jarvis to genuinely learn and adapt its internal
connection topology over time, not just update weights.
"""

import json
import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_neuroplasticity: Optional["NeuroplasticityEngine"] = None
_plasticity_lock = threading.Lock()


class PlasticityType(Enum):
    LTP = "ltp"
    LTD = "ltd"
    STDP = "stdp"
    PRUNING = "pruning"
    GROWTH = "growth"
    MYELINATION = "myelination"


class SynapseState(Enum):
    ACTIVE = "active"
    WEAKENING = "weakening"
    DORMANT = "dormant"
    PRUNED = "pruned"


@dataclass
class Synapse:
    pre: str
    post: str
    weight: float = 0.5
    baseline_weight: float = 0.5
    strength: float = 0.5
    last_activated: float = 0.0
    activation_count: int = 0
    ltp_count: int = 0
    ltd_count: int = 0
    myelination: float = 0.0
    state: SynapseState = SynapseState.ACTIVE
    created_at: float = field(default_factory=time.time)
    critical_period: bool = True

    @property
    def key(self) -> str:
        return f"{self.pre}->{self.post}"

    @property
    def is_strong(self) -> bool:
        return self.weight > 0.7 and self.state == SynapseState.ACTIVE

    @property
    def is_weak(self) -> bool:
        return self.weight < 0.3 and self.state != SynapseState.PRUNED

    @property
    def conduction_speed(self) -> float:
        base = 1.0 + self.myelination * 2.0
        return base


@dataclass
class PlasticityEvent:
    timestamp: float
    event_type: PlasticityType
    synapse_key: str
    old_weight: float
    new_weight: float
    trigger: str = ""


@dataclass
class CriticalPeriod:
    concept_type: str
    start_time: float
    duration_hours: float = 72.0
    initial_plasticity: float = 1.0
    decay_rate: float = 0.01
    current_plasticity: float = 1.0

    @property
    def is_active(self) -> bool:
        elapsed = (time.time() - self.start_time) / 3600.0
        return elapsed < self.duration_hours

    def update(self) -> float:
        if not self.is_active:
            self.current_plasticity = max(0.01, self.initial_plasticity * 0.05)
            return self.current_plasticity
        elapsed_hours = (time.time() - self.start_time) / 3600.0
        progress = elapsed_hours / self.duration_hours
        self.current_plasticity = self.initial_plasticity * math.exp(-self.decay_rate * progress * 100)
        return self.current_plasticity


class NeuroplasticityEngine:
    """Biologically-inspired structural and functional adaptation engine.

    Manages synaptic connections between conceptual nodes (concepts, tools,
    pathways) and applies neuroplasticity rules to strengthen, weaken, grow,
    or prune connections based on usage patterns, temporal correlations, and
    demand.

    The engine operates in two modes:
    - Online: During active use, applies LTP/LTD/STDP rules
    - Offline: During rest periods, applies consolidation, pruning, and myelination
    """

    MAX_SYNAPSES = 50000
    MAX_EVENTS = 5000
    PRUNING_THRESHOLD = 0.15
    LTP_BOOST = 0.05
    LTD_DECAY = 0.02
    STDP_WINDOW_MS = 50.0
    MYELINATION_RATE = 0.001
    GROWTH_THRESHOLD = 0.6
    CONSOLIDATION_BOOST = 0.03
    PRUNING_AGGRESSION = 0.1

    def __init__(self, persistence_path: Optional[Path] = None):
        self.synapses: dict[str, Synapse] = {}
        self.activation_trace: dict[str, float] = {}
        self.critical_periods: dict[str, CriticalPeriod] = {}
        self.events: deque[PlasticityEvent] = deque(maxlen=self.MAX_EVENTS)

        self._total_ltp = 0
        self._total_ltd = 0
        self._total_pruned = 0
        self._total_grown = 0
        self._total_myelinated = 0
        self._last_consolidation = time.time()
        self._lock = threading.Lock()
        self._persistence_path = persistence_path

        if persistence_path:
            self._load_state()

    def activate(self, pre: str, post: str, strength: float = 1.0, concept_type: str = "") -> float:
        with self._lock:
            key = f"{pre}->{post}"
            now = time.time()

            if key not in self.synapses:
                self._grow_synapse(pre, post, concept_type)

            synapse = self.synapses.get(key)
            if synapse is None or synapse.state == SynapseState.PRUNED:
                return 0.0

            synapse.last_activated = now
            synapse.activation_count += 1
            synapse.strength = min(1.0, synapse.strength + strength * 0.1)

            self.activation_trace[pre] = now
            self.activation_trace[post] = now

            self._apply_ltp(synapse, concept_type)
            self._check_std_window(pre, post, concept_type)

            return synapse.weight

    def deactivate(self, pre: str, post: str, strength: float = 1.0) -> float:
        with self._lock:
            key = f"{pre}->{post}"
            synapse = self.synapses.get(key)
            if synapse is None or synapse.state == SynapseState.PRUNED:
                return 0.0

            self._apply_ltd(synapse)
            synapse.strength = max(0.0, synapse.strength - strength * 0.05)
            return synapse.weight

    def get_weight(self, pre: str, post: str) -> float:
        synapse = self.synapses.get(f"{pre}->{post}")
        return synapse.weight if synapse and synapse.state != SynapseState.PRUNED else 0.0

    def get_connected(self, node: str, direction: str = "outgoing", min_weight: float = 0.1) -> list[tuple[str, float]]:
        results = []
        for key, syn in self.synapses.items():
            if syn.state == SynapseState.PRUNED or syn.weight < min_weight:
                continue
            if direction == "outgoing" and syn.pre == node:
                results.append((syn.post, syn.weight * syn.conduction_speed))
            elif direction == "incoming" and syn.post == node:
                results.append((syn.pre, syn.weight * syn.conduction_speed))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:50]

    def consolidate(self, duration_factor: float = 1.0) -> dict:
        with self._lock:
            now = time.time()
            self._last_consolidation = now

            consolidated = 0
            pruned = 0
            myelinated = 0

            for key, syn in list(self.synapses.items()):
                if syn.state == SynapseState.PRUNED:
                    continue

                time_since_activation = now - syn.last_activated

                if syn.is_strong and syn.activation_count > 5:
                    boost = self.CONSOLIDATION_BOOST * duration_factor
                    old_weight = syn.weight
                    syn.weight = min(1.0, syn.weight + boost)
                    syn.baseline_weight = (syn.baseline_weight + syn.weight) / 2
                    consolidated += 1
                    self._record_event(PlasticityType.LTP, key, old_weight, syn.weight, "consolidation")

                elif syn.is_weak and time_since_activation > 7200:
                    old_weight = syn.weight
                    syn.state = SynapseState.PRUNED
                    pruned += 1
                    self._total_pruned += 1
                    self._record_event(PlasticityType.PRUNING, key, old_weight, 0.0, "consolidation_cleanup")

                elif time_since_activation > 86400 and syn.weight < 0.4:
                    decay = self.PRUNING_AGGRESSION * duration_factor
                    old_weight = syn.weight
                    syn.weight = max(0.0, syn.weight - decay)
                    if syn.weight < self.PRUNING_THRESHOLD:
                        syn.state = SynapseState.PRUNED
                        pruned += 1
                        self._total_pruned += 1
                        self._record_event(PlasticityType.PRUNING, key, old_weight, 0.0, "consolidation_decay")

                if syn.weight > 0.7 and syn.activation_count > 20 and syn.state == SynapseState.ACTIVE:
                    myelination_boost = self.MYELINATION_RATE * duration_factor * min(1.0, syn.activation_count / 100.0)
                    syn.myelination = min(1.0, syn.myelination + myelination_boost)
                    myelinated += 1
                    self._total_myelinated += 1

            self._prune_dormant()

            for cp in self.critical_periods.values():
                cp.update()

            if self._persistence_path:
                self._save_state()

            return {
                "consolidated": consolidated,
                "pruned": pruned,
                "myelinated": myelinated,
                "total_synapses": len(self.synapses),
                "active_synapses": sum(1 for s in self.synapses.values() if s.state == SynapseState.ACTIVE),
            }

    def get_pathway_strength(self, path: list[str]) -> float:
        if len(path) < 2:
            return 0.0
        strengths = []
        for i in range(len(path) - 1):
            syn = self.synapses.get(f"{path[i]}->{path[i+1]}")
            if syn is None or syn.state == SynapseState.PRUNED:
                return 0.0
            strengths.append(syn.weight * syn.conduction_speed)
        return min(strengths) if strengths else 0.0

    def get_statistics(self) -> dict:
        active = [s for s in self.synapses.values() if s.state == SynapseState.ACTIVE]
        pruned = [s for s in self.synapses.values() if s.state == SynapseState.PRUNED]
        weights = [s.weight for s in active] if active else [0.0]

        return {
            "total_synapses": len(self.synapses),
            "active_synapses": len(active),
            "pruned_synapses": len(pruned),
            "avg_weight": sum(weights) / len(weights) if weights else 0.0,
            "max_weight": max(weights) if weights else 0.0,
            "min_weight": min(weights) if weights else 0.0,
            "strong_pathways": sum(1 for w in weights if w > 0.7),
            "weak_pathways": sum(1 for w in weights if w < 0.3),
            "myelinated_pathways": sum(1 for s in active if s.myelination > 0.1),
            "total_ltp": self._total_ltp,
            "total_ltd": self._total_ltd,
            "total_pruned": self._total_pruned,
            "total_grown": self._total_grown,
            "total_myelinated": self._total_myelinated,
            "critical_periods_active": sum(1 for cp in self.critical_periods.values() if cp.is_active),
        }

    def _apply_ltp(self, synapse: Synapse, concept_type: str = "") -> None:
        plasticity = self._get_plasticity(concept_type)
        boost = self.LTP_BOOST * plasticity * synapse.strength
        old_weight = synapse.weight
        synapse.weight = min(1.0, synapse.weight + boost)
        synapse.ltp_count += 1
        self._total_ltp += 1
        self._record_event(PlasticityType.LTP, synapse.key, old_weight, synapse.weight, "activation")

    def _apply_ltd(self, synapse: Synapse) -> None:
        decay = self.LTD_DECAY
        old_weight = synapse.weight
        synapse.weight = max(0.0, synapse.weight - decay)
        synapse.ltd_count += 1
        self._total_ltd += 1
        if synapse.weight < self.PRUNING_THRESHOLD:
            synapse.state = SynapseState.WEAKENING
        self._record_event(PlasticityType.LTD, synapse.key, old_weight, synapse.weight, "deactivation")

    def _check_std_window(self, pre: str, post: str, concept_type: str = "") -> None:
        pre_time = self.activation_trace.get(pre, 0)
        post_time = self.activation_trace.get(post, 0)
        time_diff_ms = abs(post_time - pre_time) * 1000

        if time_diff_ms < self.STDP_WINDOW_MS and pre_time > 0 and post_time > 0:
            key = f"{pre}->{post}"
            synapse = self.synapses.get(key)
            if synapse and synapse.state == SynapseState.ACTIVE:
                plasticity = self._get_plasticity(concept_type)
                if post_time > pre_time:
                    boost = 0.03 * plasticity * (1.0 - time_diff_ms / self.STDP_WINDOW_MS)
                    synapse.weight = min(1.0, synapse.weight + boost)
                    self._record_event(PlasticityType.STDP, key, synapse.weight - boost, synapse.weight, "pre_before_post")
                else:
                    decay = 0.02 * plasticity * (1.0 - time_diff_ms / self.STDP_WINDOW_MS)
                    synapse.weight = max(0.0, synapse.weight - decay)
                    self._record_event(PlasticityType.STDP, key, synapse.weight + decay, synapse.weight, "post_before_pre")

    def _grow_synapse(self, pre: str, post: str, concept_type: str = "") -> None:
        if len(self.synapses) >= self.MAX_SYNAPSES:
            self._prune_weakest(1)
            if len(self.synapses) >= self.MAX_SYNAPSES:
                return

        initial_weight = 0.25
        synapse = Synapse(
            pre=pre,
            post=post,
            weight=initial_weight,
            baseline_weight=initial_weight,
            strength=0.3,
            last_activated=time.time(),
            created_at=time.time(),
            critical_period=bool(concept_type),
        )
        self.synapses[synapse.key] = synapse
        self._total_grown += 1

        if concept_type and concept_type not in self.critical_periods:
            self.critical_periods[concept_type] = CriticalPeriod(
                concept_type=concept_type,
                start_time=time.time(),
                initial_plasticity=1.0,
            )

        self._record_event(PlasticityType.GROWTH, synapse.key, 0.0, initial_weight, "demand_driven")

    def _get_plasticity(self, concept_type: str = "") -> float:
        base = 0.5
        if concept_type in self.critical_periods:
            cp = self.critical_periods[concept_type]
            if cp.is_active:
                return base + 0.5 * cp.current_plasticity
        return base

    def _prune_weakest(self, count: int) -> int:
        candidates = [
            (k, s) for k, s in self.synapses.items()
            if s.state == SynapseState.ACTIVE and s.weight <= 0.35
        ]
        candidates.sort(key=lambda x: x[1].weight)
        pruned = 0
        for key, synapse in candidates[:count]:
            self._total_pruned += 1
            pruned += 1
            del self.synapses[key]
        return pruned

    def _prune_dormant(self) -> None:
        now = time.time()
        keys_to_remove = []
        for key, synapse in self.synapses.items():
            if synapse.state == SynapseState.PRUNED:
                if now - synapse.last_activated > 86400 * 30:
                    keys_to_remove.append(key)
        for key in keys_to_remove:
            del self.synapses[key]

    def _record_event(self, event_type: PlasticityType, synapse_key: str,
                      old_weight: float, new_weight: float, trigger: str) -> None:
        self.events.append(PlasticityEvent(
            timestamp=time.time(),
            event_type=event_type,
            synapse_key=synapse_key,
            old_weight=old_weight,
            new_weight=new_weight,
            trigger=trigger,
        ))

    def _save_state(self) -> None:
        if not self._persistence_path:
            return
        try:
            data = {
                "synapses": {k: {"pre": s.pre, "post": s.post, "weight": s.weight,
                                  "baseline_weight": s.baseline_weight, "strength": s.strength,
                                  "activation_count": s.activation_count,
                                  "myelination": s.myelination, "state": s.state.value,
                                  "ltp_count": s.ltp_count, "ltd_count": s.ltd_count}
                             for k, s in self.synapses.items()
                             if s.state != SynapseState.PRUNED},
                "stats": {
                    "total_ltp": self._total_ltp, "total_ltd": self._total_ltd,
                    "total_pruned": self._total_pruned, "total_grown": self._total_grown,
                    "total_myelinated": self._total_myelinated,
                }
            }
            self._persistence_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._persistence_path, 'w', encoding='utf-8') as f:
                json.dump(data, f)
        except Exception as e:
            logger.debug(f"Failed to save neuroplasticity state: {e}")

    def _load_state(self) -> None:
        if not self._persistence_path or not self._persistence_path.exists():
            return
        try:
            with open(self._persistence_path, encoding='utf-8') as f:
                data = json.load(f)
            for k, sd in data.get("synapses", {}).items():
                syn = Synapse(
                    pre=sd["pre"], post=sd["post"], weight=sd["weight"],
                    baseline_weight=sd.get("baseline_weight", sd["weight"]),
                    strength=sd.get("strength", sd["weight"]),
                    activation_count=sd.get("activation_count", 0),
                    myelination=sd.get("myelination", 0.0),
                    state=SynapseState(sd.get("state", "active")),
                    ltp_count=sd.get("ltp_count", 0),
                    ltd_count=sd.get("ltd_count", 0),
                )
                self.synapses[k] = syn
            stats = data.get("stats", {})
            self._total_ltp = stats.get("total_ltp", 0)
            self._total_ltd = stats.get("total_ltd", 0)
            self._total_pruned = stats.get("total_pruned", 0)
            self._total_grown = stats.get("total_grown", 0)
            self._total_myelinated = stats.get("total_myelinated", 0)
            logger.info(f"Loaded {len(self.synapses)} synapses from neuroplasticity state")
        except Exception as e:
            logger.debug(f"Failed to load neuroplasticity state: {e}")


def get_neuroplasticity_engine(persistence_path: Optional[Path] = None) -> NeuroplasticityEngine:
    global _neuroplasticity
    with _plasticity_lock:
        if _neuroplasticity is None:
            path = persistence_path or Path.home() / ".jarvis" / "neuroplasticity_state.json"
            _neuroplasticity = NeuroplasticityEngine(persistence_path=path)
    return _neuroplasticity


def reset_neuroplasticity_engine() -> None:
    global _neuroplasticity
    with _plasticity_lock:
        _neuroplasticity = None