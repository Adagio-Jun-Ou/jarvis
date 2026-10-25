"""Context-Aware Memory Router - Intelligently route queries to optimal memory tier.

Routes incoming queries to the most appropriate memory subsystem based on:
1. Query type (factual recall, procedural, conversational, exploratory)
2. Recency and access frequency
3. Current cognitive state and attention priority
4. Concept drift signals (topic changes trigger fresh tier selection)
5. Consolidation state (idle periods favor L3 deep retrieval)

Memory tiers:
- L1 Hot: Active conversation, last N turns, high attention
- L2 Warm: Recent context, accessed in last session, medium attention
- L3 Cold: Long-term knowledge, compressed facts, bulk retrieval
- Episodic: Time-stamped memories, chronological search
- Semantic: Knowledge graph, concept relationships, cross-references
"""
import logging
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

logger = logging.getLogger(__name__)


class MemoryTier(Enum):
    L1_HOT = auto()
    L2_WARM = auto()
    L3_COLD = auto()
    EPISODIC = auto()
    SEMANTIC = auto()


class QueryType(Enum):
    FACTUAL = auto()
    PROCEDURAL = auto()
    CONVERSATIONAL = auto()
    EXPLORATORY = auto()
    CORRECTION = auto()
    CREATIVE = auto()


@dataclass
class RoutingDecision:
    tier: MemoryTier
    query_type: QueryType
    confidence: float
    reason: str
    alternative_tiers: list[MemoryTier] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class MemoryRouter:
    _MAX_HISTORY = 500
    _MAX_ROUTE_CACHE = 200
    _RECENCY_WINDOW = 3600
    _QUERY_KEYWORDS = {
        QueryType.FACTUAL: {"what is", "who is", "when did", "where is", "how many", "define", "explain"},
        QueryType.PROCEDURAL: {"how to", "how do", "steps for", "process", "method", "algorithm", "recipe"},
        QueryType.CONVERSATIONAL: {"remember", "recall", "earlier", "before", "you said", "we discussed"},
        QueryType.EXPLORATORY: {"what if", "imagine", "suppose", "explore", "consider", "maybe", "could"},
        QueryType.CORRECTION: {"no,", "wrong", "actually", "i meant", "correction", "not that"},
        QueryType.CREATIVE: {"create", "generate", "write", "compose", "design", "invent", "build"},
    }

    def __init__(self):
        self._lock = threading.RLock()
        self._route_history: deque[RoutingDecision] = deque(maxlen=self._MAX_HISTORY)
        self._route_cache: dict[str, RoutingDecision] = {}
        self._tier_access_counts: dict[MemoryTier, int] = Counter()
        self._query_type_counts: dict[QueryType, int] = Counter()
        self._cognitive_state = "ENGAGED"
        self._last_drift_time: float = 0
        self._attention_source = ""
        self._stats = {
            "total_routes": 0,
            "l1_routes": 0,
            "l2_routes": 0,
            "l3_routes": 0,
            "episodic_routes": 0,
            "semantic_routes": 0,
            "cache_hits": 0,
        }
        self._drift_detector_ref: Any = None
        self._attention_queue_ref: Any = None

    def _get_drift_detector(self):
        if self._drift_detector_ref is None:
            try:
                from core.concept_drift import get_drift_detector
                self._drift_detector_ref = get_drift_detector()
            except Exception as e:
                logger.debug(f"Failed to get drift detector: {e}")
        return self._drift_detector_ref

    def _get_attention_queue(self):
        if self._attention_queue_ref is None:
            try:
                from core.attention_priority import get_attention_queue
                self._attention_queue_ref = get_attention_queue()
            except Exception as e:
                logger.debug(f"Failed to get attention queue: {e}")
        return self._attention_queue_ref

    def classify_query(self, query: str) -> QueryType:
        query_lower = query.lower().strip()
        best_type = QueryType.FACTUAL
        best_score = 0

        for qtype, keywords in self._QUERY_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            if score > best_score:
                best_score = score
                best_type = qtype

        if best_score == 0:
            if any(c in query_lower for c in "?!"):
                return QueryType.EXPLORATORY
            return QueryType.CONVERSATIONAL

        return best_type

    def route(self, query: str, context: dict[str, Any] | None = None) -> RoutingDecision:
        context = context or {}
        query_type = self.classify_query(query)

        cache_key = f"{query_type.value}:{hash(query[:100])}"
        with self._lock:
            cached = self._route_cache.get(cache_key)
            if cached and time.time() - cached.metadata.get("cache_time", 0) < 300:
                self._stats["cache_hits"] += 1
                return cached

        tier, confidence, reason = self._select_tier(query_type, query, context)
        alternatives = self._compute_alternatives(tier, query_type)

        decision = RoutingDecision(
            tier=tier,
            query_type=query_type,
            confidence=confidence,
            reason=reason,
            alternative_tiers=alternatives,
            metadata={"cache_time": time.time(), "query_length": len(query)},
        )

        with self._lock:
            self._route_history.append(decision)
            self._tier_access_counts[tier] += 1
            self._query_type_counts[query_type] += 1
            self._stats["total_routes"] += 1
            tier_key = f"{tier.name.lower()}_routes"
            self._stats[tier_key] = self._stats.get(tier_key, 0) + 1
            if len(self._route_cache) >= self._MAX_ROUTE_CACHE:
                oldest_key = next(iter(self._route_cache))
                del self._route_cache[oldest_key]
            self._route_cache[cache_key] = decision

        return decision

    def _select_tier(
        self, query_type: QueryType, query: str, context: dict[str, Any]
    ) -> tuple[MemoryTier, float, str]:
        state = context.get("cognitive_state", self._cognitive_state)

        drift = self._get_drift_detector()
        if drift is not None:
            try:
                current_topic = drift.get_current_topic()
                if current_topic and time.time() - self._last_drift_time < 300:
                    state = "EXPLORING"
            except Exception as e:
                logger.debug(f"Failed to get current topic from drift detector: {e}")

        aq = self._get_attention_queue()
        urgency = None
        if aq is not None:
            try:
                top = aq.peek()
                if top:
                    urgency = top.urgency
            except Exception as e:
                logger.debug(f"Failed to peek attention queue: {e}")

        if state == "FRUSTRATED":
            return MemoryTier.L1_HOT, 0.85, "Frustrated user → hot cache for speed"
        elif state == "OVERLOADED":
            return MemoryTier.L1_HOT, 0.80, "Overloaded → minimal retrieval, hot cache only"

        if urgency is not None:
            from core.attention_priority import UrgencySignal
            if urgency == UrgencySignal.CORRECTION:
                return MemoryTier.L2_WARM, 0.75, "Correction → recent context needed"
            if urgency == UrgencySignal.SAFETY_CRITICAL:
                return MemoryTier.L1_HOT, 0.95, "Safety critical → immediate hot retrieval"

        if query_type == QueryType.CORRECTION:
            return MemoryTier.L2_WARM, 0.78, "Correction → recent context needed"
        elif query_type == QueryType.CONVERSATIONAL:
            return MemoryTier.L1_HOT, 0.80, "Conversational → hot memory for continuity"
        elif query_type == QueryType.FACTUAL:
            l3_count = self._tier_access_counts.get(MemoryTier.L3_COLD, 0)
            total = max(1, sum(self._tier_access_counts.values()))
            if l3_count / total > 0.5:
                return MemoryTier.L2_WARM, 0.65, "Frequent deep queries → warm cache"
            return MemoryTier.L3_COLD, 0.75, "Factual → deep retrieval"
        elif query_type == QueryType.PROCEDURAL:
            return MemoryTier.SEMANTIC, 0.70, "Procedural → knowledge graph for methods"
        elif query_type == QueryType.EXPLORATORY:
            return MemoryTier.L3_COLD, 0.60, "Exploratory → broad retrieval"
        elif query_type == QueryType.CREATIVE:
            return MemoryTier.SEMANTIC, 0.65, "Creative → cross-references and connections"

        return MemoryTier.L2_WARM, 0.50, "Default → warm cache"

    def _compute_alternatives(self, primary: MemoryTier, query_type: QueryType) -> list[MemoryTier]:
        tier_order = [MemoryTier.L1_HOT, MemoryTier.L2_WARM, MemoryTier.L3_COLD, MemoryTier.EPISODIC, MemoryTier.SEMANTIC]
        alternatives = [t for t in tier_order if t != primary]
        if query_type == QueryType.FACTUAL:
            alternatives = [MemoryTier.L2_WARM, MemoryTier.EPISODIC, MemoryTier.L1_HOT, MemoryTier.SEMANTIC]
        elif query_type == QueryType.PROCEDURAL:
            alternatives = [MemoryTier.L3_COLD, MemoryTier.L2_WARM, MemoryTier.EPISODIC, MemoryTier.L1_HOT]
        return alternatives[:3]

    def update_cognitive_state(self, state_name: str):
        with self._lock:
            self._cognitive_state = state_name

    def notify_drift(self, drift_time: float):
        with self._lock:
            self._last_drift_time = drift_time

    def get_stats(self) -> dict:
        with self._lock:
            return {
                **self._stats,
                "cognitive_state": self._cognitive_state,
                "tier_distribution": dict(self._tier_access_counts),
                "query_type_distribution": dict(self._query_type_counts),
                "route_history_size": len(self._route_history),
                "cache_size": len(self._route_cache),
            }

    def clear_cache(self):
        with self._lock:
            self._route_cache.clear()


_memory_router: MemoryRouter | None = None
_router_lock = threading.Lock()


def get_memory_router() -> MemoryRouter:
    global _memory_router
    if _memory_router is None:
        with _router_lock:
            if _memory_router is None:
                _memory_router = MemoryRouter()
    return _memory_router