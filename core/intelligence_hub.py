"""Intelligence Integration Hub - Cross-subsystem knowledge sharing.

Enables intelligent coordination between:
- Prefrontal Cortex (reasoning)
- Context Cascade (context awareness)
- Erdős Engine (mathematical exploration)
- Daydream Engine (autonomous discovery)
- Knowledge Graph (insights)
- Memory Systems (context)

Provides:
- Cross-subsystem insight sharing
- Intelligent query routing
- Collaborative problem solving
- Emergent intelligence coordination
"""
import asyncio
import logging
import threading
import time
from core.helpers import safe_error_message
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class IntelligenceSource(Enum):
    """Sources of intelligence in Jarvis."""
    PREFRONTAL_CORTEX = "prefrontal_cortex"
    CONTEXT_CASCADE = "context_cascade"
    ERDOS_ENGINE = "erdos_engine"
    DAYDREAM_ENGINE = "daydream_engine"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    MEMORY_SYSTEM = "memory_system"
    INSIGHT_BRIDGE = "insight_bridge"
    USER_MODEL = "user_model"
    EMOTIONAL_STATE = "emotional_state"


class QueryType(Enum):
    """Types of queries routed through the hub."""
    FACTUAL = "factual"
    REASONING = "reasoning"
    CREATIVE = "creative"
    MATHEMATICAL = "mathematical"
    ANALYTICAL = "analytical"
    EMOTIONAL = "emotional"
    CONTEXTUAL = "contextual"
    META_COGNITIVE = "meta_cognitive"


@dataclass
class IntelligenceSignal:
    """A signal from one intelligence subsystem to others."""
    source: IntelligenceSource
    signal_type: str
    content: Any
    confidence: float = 0.5
    urgency: float = 0.5  # 0=low, 1=critical
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)


@dataclass
class Query:
    """A query to be processed by the intelligence hub."""
    id: str
    query_text: str
    query_type: QueryType
    context: dict = field(default_factory=dict)
    target_subsystems: list[IntelligenceSource] = field(default_factory=list)
    responses: dict[str, Any] = field(default_factory=dict)
    integration_result: Optional[str] = None
    confidence: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


@dataclass
class CrossSubsystemInsight:
    """An insight that spans multiple subsystems."""
    id: str
    insight_text: str
    contributing_sources: list[IntelligenceSource]
    evidence: dict[str, list[str]] = field(default_factory=dict)
    confidence: float = 0.0
    novelty_score: float = 0.0
    applicability: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


class IntelligenceHub:
    """Central hub for coordinating Jarvis's intelligence subsystems.
    
    Features:
    - Intelligent query routing
    - Cross-subsystem insight synthesis
    - Collaborative problem solving
    - Signal broadcasting
    - Emergent behavior detection
    
    Usage:
        hub = get_intelligence_hub()
        
        # Route query
        response = await hub.route_query("What patterns exist in this data?")
        
        # Broadcast signal
        hub.broadcast_signal(IntelligenceSignal(
            source=IntelligenceSource.ERDOS_ENGINE,
            signal_type="insight",
            content="New mathematical pattern discovered"
        ))
        
        # Get integrated insights
        insights = hub.synthesize_insights()
    """
    
    MAX_INSIGHT_HISTORY = 200
    MAX_QUERY_HISTORY = 100
    MAX_SIGNAL_BUFFER = 500
    
    def __init__(self):
        self._subsystem_status: dict[IntelligenceSource, bool] = {
            source: False for source in IntelligenceSource
        }
        self._signal_buffer: deque = deque(maxlen=self.MAX_SIGNAL_BUFFER)
        self._insight_history: deque = deque(maxlen=self.MAX_INSIGHT_HISTORY)
        self._query_history: deque = deque(maxlen=self.MAX_QUERY_HISTORY)
        self._subsystem_callbacks: dict[IntelligenceSource, Callable] = {}
        self._integration_patterns: dict[str, int] = {}
        self._lock = threading.Lock()
        self._running = False
        
        # Query routing rules
        self._routing_rules = self._init_routing_rules()
    
    def _init_routing_rules(self) -> dict[QueryType, list[IntelligenceSource]]:
        """Initialize query routing rules."""
        return {
            QueryType.FACTUAL: [
                IntelligenceSource.MEMORY_SYSTEM,
                IntelligenceSource.KNOWLEDGE_GRAPH,
            ],
            QueryType.REASONING: [
                IntelligenceSource.PREFRONTAL_CORTEX,
                IntelligenceSource.KNOWLEDGE_GRAPH,
            ],
            QueryType.CREATIVE: [
                IntelligenceSource.PREFRONTAL_CORTEX,
                IntelligenceSource.DAYDREAM_ENGINE,
            ],
            QueryType.MATHEMATICAL: [
                IntelligenceSource.ERDOS_ENGINE,
                IntelligenceSource.KNOWLEDGE_GRAPH,
            ],
            QueryType.ANALYTICAL: [
                IntelligenceSource.PREFRONTAL_CORTEX,
                IntelligenceSource.MEMORY_SYSTEM,
            ],
            QueryType.EMOTIONAL: [
                IntelligenceSource.EMOTIONAL_STATE,
                IntelligenceSource.USER_MODEL,
            ],
            QueryType.CONTEXTUAL: [
                IntelligenceSource.CONTEXT_CASCADE,
                IntelligenceSource.MEMORY_SYSTEM,
            ],
            QueryType.META_COGNITIVE: [
                IntelligenceSource.PREFRONTAL_CORTEX,
                IntelligenceSource.INSIGHT_BRIDGE,
            ],
        }
    
    def register_subsystem(
        self,
        source: IntelligenceSource,
        callback: Callable,
    ) -> None:
        """Register a subsystem with its callback.
        
        Args:
            source: The subsystem source
            callback: Async function to call for queries
        """
        with self._lock:
            self._subsystem_callbacks[source] = callback
            self._subsystem_status[source] = True
        logger.info(f"Registered subsystem: {source.value}")
    
    def set_subsystem_status(self, source: IntelligenceSource, active: bool) -> None:
        """Update subsystem availability status."""
        with self._lock:
            self._subsystem_status[source] = active
        logger.debug(f"Subsystem {source.value} status: {active}")
    
    async def route_query(
        self,
        query_text: str,
        query_type: QueryType = None,
        context: dict = None,
        target_subsystems: list[IntelligenceSource] = None,
    ) -> Query:
        """Route a query to appropriate subsystems.
        
        Args:
            query_text: The query to process
            query_type: Type of query (auto-detected if None)
            context: Additional context
            target_subsystems: Specific subsystems to query (auto-selected if None)
            
        Returns:
            Query with responses from relevant subsystems
        """
        import hashlib
        query_id = hashlib.sha256(f"{query_text}:{time.time()}".encode()).hexdigest()[:12]
        
        # Auto-detect query type if not provided
        if query_type is None:
            query_type = self._detect_query_type(query_text)
        
        # Select target subsystems
        if target_subsystems is None:
            target_subsystems = self._routing_rules.get(query_type, [])
        
        query = Query(
            id=query_id,
            query_text=query_text,
            query_type=query_type,
            context=context or {},
            target_subsystems=target_subsystems,
        )
        
        # Query each subsystem
        for source in target_subsystems:
            if not self._subsystem_status.get(source, False):
                query.responses[source.value] = {"status": "unavailable"}
                continue
            
            callback = self._subsystem_callbacks.get(source)
            if callback is None:
                query.responses[source.value] = {"status": "not_registered"}
                continue
            
            try:
                if asyncio.iscoroutinefunction(callback):
                    response = await callback(query_text, context)
                else:
                    response = callback(query_text, context)
                
                query.responses[source.value] = response
                
            except Exception as e:
                logger.error(f"Error querying {source.value}: {e}")
                query.responses[source.value] = {"status": "error", "error": safe_error_message(e)}
        
        # Integrate responses
        query.integration_result = self._integrate_responses(query)
        query.confidence = self._calculate_confidence(query)
        query.completed_at = datetime.now()
        
        # Store in history
        with self._lock:
            self._query_history.append(query)
        
        return query
    
    def _detect_query_type(self, query_text: str) -> QueryType:
        """Auto-detect query type from text."""
        query_lower = query_text.lower()
        
        # Mathematical keywords
        if any(kw in query_lower for kw in ["calculate", "solve", "prove", "equation", "math"]):
            return QueryType.MATHEMATICAL
        
        # Reasoning keywords
        if any(kw in query_lower for kw in ["why", "because", "reason", "explain", "how"]):
            return QueryType.REASONING
        
        # Creative keywords
        if any(kw in query_lower for kw in ["create", "design", "imagine", "generate", "invent"]):
            return QueryType.CREATIVE
        
        # Emotional keywords
        if any(kw in query_lower for kw in ["feel", "emotion", "mood", "happy", "sad"]):
            return QueryType.EMOTIONAL
        
        # Contextual keywords
        if any(kw in query_lower for kw in ["context", "background", "history", "remember"]):
            return QueryType.CONTEXTUAL
        
        # Meta-cognitive keywords
        if any(kw in query_lower for kw in ["think about", "reflect", "meta", "aware"]):
            return QueryType.META_COGNITIVE
        
        # Default to factual
        return QueryType.FACTUAL
    
    def broadcast_signal(self, signal: IntelligenceSignal) -> None:
        """Broadcast a signal to all subsystems.
        
        Args:
            signal: The signal to broadcast
        """
        with self._lock:
            self._signal_buffer.append(signal)
        
        logger.debug(
            f"Broadcast signal from {signal.source.value}: {signal.signal_type}"
        )
        
        # Subsystems should poll for signals or have event handlers
        # This is a simple pub/sub mechanism
    
    def get_signals(
        self,
        source: Optional[IntelligenceSource] = None,
        signal_type: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> list[IntelligenceSignal]:
        """Get signals matching criteria.
        
        Args:
            source: Filter by source
            signal_type: Filter by signal type
            since: Filter by timestamp
            
        Returns:
            List of matching signals
        """
        with self._lock:
            signals = list(self._signal_buffer)
        
        # Filter
        if source is not None:
            signals = [s for s in signals if s.source == source]
        
        if signal_type is not None:
            signals = [s for s in signals if s.signal_type == signal_type]
        
        if since is not None:
            signals = [s for s in signals if s.timestamp > since]
        
        return signals
    
    def create_cross_subsystem_insight(
        self,
        insight_text: str,
        sources: list[IntelligenceSource],
        evidence: dict[str, list[str]] = None,
    ) -> CrossSubsystemInsight:
        """Create an insight that spans multiple subsystems.
        
        Args:
            insight_text: The insight description
            sources: Contributing subsystems
            evidence: Evidence from each source
            
        Returns:
            Cross-subsystem insight
        """
        import hashlib
        insight_id = hashlib.sha256(f"{insight_text}:{time.time()}".encode()).hexdigest()[:12]
        
        # Calculate confidence based on number of sources
        confidence = min(0.95, 0.5 + len(sources) * 0.1)
        
        # Calculate novelty score
        novelty_score = self._calculate_novelty(insight_text)
        
        insight = CrossSubsystemInsight(
            id=insight_id,
            insight_text=insight_text,
            contributing_sources=sources,
            evidence=evidence or {},
            confidence=confidence,
            novelty_score=novelty_score,
        )
        
        with self._lock:
            self._insight_history.append(insight)
        
        logger.info(f"Created cross-subsystem insight: {insight_text[:60]}...")
        return insight
    
    def _calculate_novelty(self, insight_text: str) -> float:
        """Calculate novelty score for an insight."""
        # Simple novelty calculation based on pattern matching
        pattern_key = insight_text[:50].lower()
        
        with self._lock:
            count = self._integration_patterns.get(pattern_key, 0)
            self._integration_patterns[pattern_key] = count + 1
        
        # Higher count = lower novelty
        return max(0.0, 1.0 - count * 0.1)
    
    def _integrate_responses(self, query: Query) -> str:
        """Integrate responses from multiple subsystems."""
        responses = query.responses
        
        if not responses:
            return "No responses available"
        
        # Combine successful responses
        successful = []
        for source, response in responses.items():
            if isinstance(response, dict) and response.get("status") in ["unavailable", "not_registered", "error"]:
                continue
            successful.append(f"[{source}]: {response}")
        
        if not successful:
            return "No successful responses"
        
        return "\n".join(successful)
    
    def _calculate_confidence(self, query: Query) -> float:
        """Calculate overall confidence in the response."""
        responses = query.responses
        
        if not responses:
            return 0.0
        
        # Count successful responses
        successful = 0
        for response in responses.values():
            if isinstance(response, dict):
                if response.get("status") not in ["unavailable", "not_registered", "error"]:
                    successful += 1
            else:
                successful += 1
        
        return successful / len(responses) if responses else 0.0
    
    def synthesize_insights(self, limit: int = 10) -> list[CrossSubsystemInsight]:
        """Synthesize insights from recent signals and queries.
        
        Args:
            limit: Maximum insights to return
            
        Returns:
            List of synthesized insights
        """
        # Get recent signals
        recent_signals = self.get_signals(since=datetime.now().replace(
            hour=0, minute=0, second=0, microsecond=0
        ))
        
        # Group by source
        source_signals: dict[IntelligenceSource, list] = {}
        for signal in recent_signals:
            if signal.source not in source_signals:
                source_signals[signal.source] = []
            source_signals[signal.source].append(signal)
        
        # Find patterns across sources
        insights = []
        
        # Check for multi-source signals
        if len(source_signals) >= 2:
            # Create synthesis insight
            sources = list(source_signals.keys())
            signal_types = [s.signal_type for signals in source_signals.values() for s in signals]
            
            insight = self.create_cross_subsystem_insight(
                insight_text=f"Multiple subsystems active: {len(sources)} sources, {len(signal_types)} signals",
                sources=sources,
                evidence={s.value: [sig.signal_type for sig in signals[:3]] for s, signals in source_signals.items()},
            )
            insights.append(insight)
        
        # Return top insights by confidence and novelty
        insights.sort(key=lambda i: (i.confidence + i.novelty_score), reverse=True)
        return insights[:limit]
    
    def get_hub_stats(self) -> dict:
        """Get hub statistics."""
        with self._lock:
            return {
                "active_subsystems": sum(1 for status in self._subsystem_status.values() if status),
                "total_subsystems": len(self._subsystem_status),
                "signal_buffer_size": len(self._signal_buffer),
                "insight_count": len(self._insight_history),
                "query_count": len(self._query_history),
                "integration_patterns": len(self._integration_patterns),
                "subsystem_status": {s.value: status for s, status in self._subsystem_status.items()},
            }
    
    def clear(self) -> None:
        """Clear all hub state."""
        with self._lock:
            self._signal_buffer.clear()
            self._insight_history.clear()
            self._query_history.clear()
            self._integration_patterns.clear()
        logger.info("Intelligence hub cleared")


# Singleton
_intelligence_hub: Optional[IntelligenceHub] = None
_hub_lock = threading.Lock()


def get_intelligence_hub() -> IntelligenceHub:
    """Get the singleton intelligence hub."""
    global _intelligence_hub
    if _intelligence_hub is None:
        with _hub_lock:
            if _intelligence_hub is None:
                _intelligence_hub = IntelligenceHub()
    return _intelligence_hub