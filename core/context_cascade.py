"""Context Cascade Integration - Flows from LCM through Context to Active Models.

This module integrates the concept cascade:
    LCM (Prefrontal Cortex) → Concept Model → Unified Context → Dual Models

Key Integration Points:
1. LCM predictions inform context assembly
2. Concept embeddings bridge all memory systems
3. Active model context includes concept guidance
4. Workflow stages emit concept observations
5. Cross-model learning via shared embeddings

Architecture:
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CONTEXT CASCADE FLOW                                  │
├───────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐    │
│   │                     ACTIVITY SOURCES                                 │    │
│   │                                                                      │    │
│   │  User Query     Model Response    Tool Calls     Subsystem Events   │    │
│   │       │              │               │                  │             │    │
│   │       └──────────────┴───────────────┴──────────────────┘             │    │
│   │                                 │                                    │    │
│   └─────────────────────────────────┼─────────────────────────────────────┘    │
│                                     │                                        │
│                          ┌──────────▼──────────┐                              │
│                          │   CONCEPT ENCODER    │                              │
│                          │   (Jina v5, 768-dim)  │                              │
│                          └──────────┬──────────┘                              │
│                                     │                                        │
│   ┌─────────────────────────────────┼─────────────────────────────────────┐   │
│   │                     LCM CORE (Prefrontal Cortex)                    │   │
│   │                                                                      │   │
│   │   ┌─────────────────────────────────────────────────────────┐       │   │
│   │   │  Concept Sequence → Transformer → Next Concept Pred     │       │   │
│   │   │                                                           │       │   │
│   │   │  Input: [User_Query, Model_Response, Tool_Call, ...]    │       │   │
│   │   │  Output: Predicted_Next_Concept, Type_Probs, Confidence │       │   │
│   │   └─────────────────────────────────────────────────────────┘       │   │
│   │                                 │                                    │   │
│   │                    ┌────────────┼────────────┐                      │   │
│   │                    ▼            ▼             ▼                      │   │
│   │              Meta-Guidance  Anomaly     Next Concept               │   │
│   │                                Detection    Prediction              │   │
│   │                    │            │             │                      │   │
│   └────────────────────┼────────────┼─────────────┼──────────────────────┘   │
│                        │            │             │                          │
│   ┌────────────────────┼────────────┼─────────────┼──────────────────────┐   │
│   │                     CONCEPT MODEL (Meta-Reasoning)                │   │
│   │                                                                     │   │
│   │  Pattern Extraction    Concept Bridges    Meta-Reasoning           │   │
│   │        │                     │                  │                  │   │
│   │        └─────────────────────┴──────────────────┘                  │   │
│   │                              │                                      │   │
│   │                    ┌─────────▼──────────┐                           │   │
│   │                    │  CONCEPT EMBEDDINGS │                           │   │
│   │                    │  (Pattern Library)   │                           │   │
│   │                    └─────────┬──────────┘                           │   │
│   │                              │                                      │   │
│   └──────────────────────────────┼──────────────────────────────────────┘   │
│                                  │                                          │
│   ┌──────────────────────────────┼──────────────────────────────────────┐   │
│   │                    MEMORY SYSTEMS                                    │   │
│   │                                                                       │   │
│   │  ┌───────────────┐ ┌────────────────┐ ┌────────────────────────┐   │   │
│   │  │ Vector Memory │ │ Knowledge Graph│ │ Persistent Memory      │   │   │
│   │  │ (Semantic)    │ │ (PageRank)     │ │ (Facts, Preference)   │   │   │
│   │  │               │ │                │ │                        │   │   │
│   │  │ Embedding →   │ │ Embedding →    │ │ Embedding →            │   │   │
│   │  │ Similarity    │ │ Node Similarity│ │ Semantic Search        │   │   │
│   │  └───────┬───────┘ └────────┬───────┘ └───────────┬────────────┘   │   │
│   │          │                  │                     │                │   │
│   │          └──────────────────┴─────────────────────┘                │   │
│   │                             │                                      │   │
│   └─────────────────────────────┼──────────────────────────────────────┘   │
│                                 │                                          │
│   ┌─────────────────────────────┼──────────────────────────────────────┐   │
│   │                   UNIFIED CONTEXT ASSEMBLY                          │   │
│   │                                                                      │   │
│   │  Working │ Project │ Knowledge │ Episodic │ Session │ Semantic    │   │
│   │     │        │          │          │          │          │        │   │
│   │     └────────┴──────────┴──────────┴──────────┴──────────┘        │   │
│   │                              │                                      │   │
│   │                    ┌─────────▼──────────┐                           │   │
│   │                    │ LCM CONTEXT LAYER  │ ◄─── Meta-Guidance,       │   │
│   │                    │ (Predictions,      │      Anomaly Detection    │   │
│   │                    │  Pattern Context)  │      Next Concepts       │   │
│   │                    └─────────┬──────────┘                           │   │
│   │                              │                                      │   │
│   │                    ┌─────────▼──────────┐                           │   │
│   │                    │ TOKEN OPTIMIZER   │ ◄─── Budget Management    │   │
│   │                    │ (16000 tokens)    │      Priority Ordering    │   │
│   │                    └─────────┬──────────┘                           │   │
│   │                              │                                      │   │
│   └──────────────────────────────┼──────────────────────────────────────┘   │
│                                  │                                          │
│   ┌──────────────────────────────┼──────────────────────────────────────┐   │
│   │                     DUAL MODEL ORCHESTRATOR                          │   │
│   │                                                                       │   │
│   │  ┌─────────────────┐            ┌─────────────────────────────┐     │   │
│   │  │   PRIMARY LLM   │            │        BACKEND LLM          │     │   │
│   │  │   (Kimi K2.5)   │            │        (GLM-5)              │     │   │
│   │  │                 │            │                             │     │   │
│   │  │  • Chat         │            │  • Code                      │     │   │
│   │  │  • Creative     │            │  • Math                     │     │   │
│   │  │  • Vision       │            │  • Reasoning                │     │   │
│   │  │  • Abstractions ◄────────────┼─────────┬───────────────────┤     │   │
│   │  │                 │            │         │                    │     │   │
│   │  │                 │            │  ◄───────┘ Concept Guidance  │     │   │
│   │  │                 │            │         (Patterns, Bridges)  │     │   │
│   │  └─────────────────┘            └─────────────────────────────┘     │   │
│   │                                                                       │   │
│   │                    ┌────────────────────────────────┐               │   │
│   │                    │     EMBEDDING INTERCONNECT     │               │   │
│   │                    │   (Cross-model shared space)   │               │   │
│   │                    │                                │               │   │
│   │                    │  Primary ⟄──────────────────→ Backend         │   │
│   │                    │            Shared Embeddings                  │   │
│   │                    └────────────────────────────────┘               │   │
│   │                                                                       │   │
│   └───────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└───────────────────────────────────────────────────────────────────────────────┘
"""

import hashlib
import logging
import threading
import time
from collections import OrderedDict, deque
from core.helpers import safe_error_message
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ConceptCascade:
    """Represents a concept cascade from LCM to models."""
    trigger_type: str  # "user_query", "tool_call", "subsystem_event", etc.
    trigger_content: str
    concepts_observed: list[dict] = field(default_factory=list)
    predicted_next: dict | None = None
    meta_guidance: str | None = None
    anomaly_detected: bool = False
    anomaly_reason: str | None = None
    concept_patterns: list[str] = field(default_factory=list)
    embedding_cross_refs: list[dict] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    timing: dict = field(default_factory=dict)


class ContextCascadeIntegration:
    """Integrates LCM predictions into context assembly for dual models.
    
    This is the SINGLE integration point where:
    1. All activity flows through concept encoding
    2. LCM predicts next concepts and detects anomalies
    3. Concept model provides meta-guidance
    4. Unified context includes concept-derived context
    5. Dual models receive concept-enhanced context
    
    Usage:
        cascade = ContextCascadeIntegration()
        
        # Observe activity (triggers cascade)
        cascade.observe("user_query", "How do I improve cell viability?")
        
        # Get context with LCM guidance
        context = await cascade.assemble_context_with_guidance(
            query="improve cell viability simulation",
            max_tokens=16000,
        )
        
        # Context includes:
        # - All memory layers
        # - LCM predicted next concepts
        # - Concept meta-guidance
        # - Anomaly warnings
        # - Pattern matches
    """
    
    def __init__(self):
        self._pfc = None
        self._concept_model = None
        self._embedding = None
        self._unified_context = None
        self._lock = threading.RLock()  # Use RLock for recursive safety
        
        self._cascade_history: deque[ConceptCascade] = deque(maxlen=100)
        
        self._concept_embeddings: OrderedDict[str, list[float]] = OrderedDict()
        self._pattern_embeddings: OrderedDict[str, list[float]] = OrderedDict()
        self._max_embeddings = 1000
        self._max_pattern_embeddings = 500  # Add bound for pattern embeddings
        
        self._initialized = False
    
    def _ensure_initialized(self):
        """Lazy initialization of dependencies."""
        if self._initialized:
            return
        
        with self._lock:
            if self._initialized:
                return
            
            try:
                from core.prefrontal_cortex import get_prefrontal_cortex
                self._pfc = get_prefrontal_cortex()
            except Exception:
                logger.warning("PFC not available")
            
            try:
                from core.concept_model import get_concept_orchestrator
                self._concept_model = get_concept_orchestrator()
            except Exception:
                logger.warning("Concept model not available")
            
            try:
                from core.embedding_interconnect import get_embedding_interconnect
                self._embedding = get_embedding_interconnect()
            except Exception:
                logger.warning("Embedding interconnect not available")
            
            try:
                from core.unified_context import get_unified_context
                self._unified_context = get_unified_context()
            except Exception:
                logger.warning("Unified context not available")
            
            self._initialized = True
    
    def observe(self, trigger_type: str, content: str, source: str = "unknown") -> ConceptCascade:
        """Observe activity and cascade through all systems.
        
        This is the MAIN entry point for the concept cascade:
        1. Encode content to concept embedding
        2. Observe in PFC for LCM training
        3. Store in concept embeddings
        4. Check for anomalies
        5. Predict next concepts
        
        Args:
            trigger_type: Type of trigger (user_query, tool_call, etc.)
            content: The content to observe
            source: Where this came from
        
        Returns:
            ConceptCascade with all cascade results
        """
        self._ensure_initialized()
        
        cascade = ConceptCascade(
            trigger_type=trigger_type,
            trigger_content=content[:500],  # Limit content size
        )
        
        # Timing metrics
        timing = {}
        
        try:
            # Time embedding generation
            t0 = time.perf_counter()
            embedding = self._get_embedding(content)
            timing["embedding_ms"] = (time.perf_counter() - t0) * 1000
            if embedding is not None:
                cascade.concepts_observed.append({
                    "type": trigger_type,
                    "source": source,
                    "embedding_dim": len(embedding),
                })
                
                self._store_concept_embedding(content, embedding)
                
                if self._pfc and hasattr(self._pfc, '_running') and self._pfc._running:
                    try:
                        from core.prefrontal_cortex import ConceptType
                        type_map = {
                            "user_query": ConceptType.USER_QUERY,
                            "model_response": ConceptType.MODEL_RESPONSE,
                            "tool_call": ConceptType.TOOL_CALL,
                            "tool_result": ConceptType.TOOL_RESULT,
                            "insight": ConceptType.INSIGHT,
                            "memory": ConceptType.MEMORY,
                            "subsystem_event": ConceptType.SUBSYSTEM_EVENT,
                            "observation": ConceptType.OBSERVATION,
                            "music_token": ConceptType.MUSIC_TOKEN,
                        }
                        concept_type = type_map.get(trigger_type, ConceptType.OBSERVATION)
                        
                        # Schedule async observation with error tracking
                        import asyncio
                        try:
                            loop = asyncio.get_running_loop()
                            task = loop.create_task(
                                self._pfc.observe(
                                    content=content,
                                    concept_type=concept_type,
                                    source=source,
                                )
                            )
                            task.add_done_callback(lambda t: t.exception() if not t.cancelled() and t.exception() else None)
                        except RuntimeError:
                            logger.debug("No running event loop for PFC observation")
                            # Fallback: synchronous observation in background thread
                            try:
                                import concurrent.futures
                                with concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="pfc_obs") as executor:
                                    future = executor.submit(
                                        asyncio.run,
                                        self._pfc.observe(content=content, concept_type=concept_type, source=source)
                                    )
                                    future.result(timeout=5.0)
                            except Exception as e2:
                                logger.debug(f"PFC observation fallback also failed: {e2}")
                    except Exception as e:
                        logger.debug(f"PFC observation failed: {e}")
                
                if self._pfc:
                    try:
                        recent_concepts = list(self._pfc._state.active_concepts)[-10:]
                        predicted_emb, type_probs, confidence = self._pfc.lcm.predict_next(recent_concepts)
                        
                        cascade.predicted_next = {
                            "embedding_dim": len(predicted_emb),
                            "type_probs": [(t.value, float(p)) for t, p in type_probs[:3]],
                            "confidence": float(confidence),
                        }
                        
                        anomaly_score = self._check_anomalies(embedding, predicted_emb, content)
                        cascade.anomaly_detected = anomaly_score > 0.7
                        if cascade.anomaly_detected:
                            cascade.anomaly_reason = f"Anomaly score: {anomaly_score:.2f}"
                    except Exception as e:
                        logger.debug(f"LCM prediction failed: {e}")
                
                # Time pattern matching
                t1 = time.perf_counter()
                patterns = self._find_patterns(embedding)
                timing["pattern_ms"] = (time.perf_counter() - t1) * 1000
                cascade.concept_patterns = patterns
                
                # Time cross-ref lookup
                t2 = time.perf_counter()
                cross_refs = self._find_cross_refs(embedding, source)
                timing["cross_ref_ms"] = (time.perf_counter() - t2) * 1000
                cascade.embedding_cross_refs = cross_refs
            
        except Exception as e:
            logger.error(f"Cascade observation failed: {e}", exc_info=True)
            timing["error"] = str(e)
            cascade.anomaly_detected = True
            cascade.anomaly_reason = f"observation_failed: {e}"
        
        # Store timing in cascade for observability
        cascade.timing = timing
        
        # Log timing for debugging
        if timing:
            logger.debug(f"[CASCADE] Timing: embedding={timing.get('embedding_ms', 0):.1f}ms, "
                        f"pattern={timing.get('pattern_ms', 0):.1f}ms, "
                        f"cross_ref={timing.get('cross_ref_ms', 0):.1f}ms")
        
        with self._lock:
            self._cascade_history.append(cascade)
        
        return cascade
    
    def _get_embedding(self, content: str) -> list[float] | None:
        """Get embedding from shared interconnect."""
        if self._embedding:
            try:
                emb = self._embedding.encode(content)
                return emb.tolist() if hasattr(emb, 'tolist') else list(emb)
            except Exception as e:
                logger.debug(f"Embedding failed: {e}")
        return None
    
    def _store_concept_embedding(self, content: str, embedding: list[float]):
        """Store concept embedding with LRU eviction."""
        key = hashlib.md5(content.encode()).hexdigest()[:8]
        with self._lock:
            if key in self._concept_embeddings:
                self._concept_embeddings.move_to_end(key)
            self._concept_embeddings[key] = embedding
            if len(self._concept_embeddings) > self._max_embeddings:
                self._concept_embeddings.popitem(last=False)

    def _store_pattern_embedding(self, pattern_id: str, embedding: list[float]):
        """Store pattern embedding with bounded eviction."""
        with self._lock:
            if pattern_id in self._pattern_embeddings:
                self._pattern_embeddings.move_to_end(pattern_id)
            self._pattern_embeddings[pattern_id] = embedding
            if len(self._pattern_embeddings) > self._max_pattern_embeddings:
                self._pattern_embeddings.popitem(last=False)
    
    def _check_anomalies(self, current_emb: list, predicted_emb: list, content: str) -> float:
        """Check if current concept is anomalous.
        
        Returns anomaly score (0-1, higher = more anomalous).
        """
        import numpy as np
        
        try:
            current = np.array(current_emb)
            predicted = np.array(predicted_emb)
            
            similarity = np.dot(current, predicted) / (np.linalg.norm(current) * np.linalg.norm(predicted) + 1e-8)
            
            anomaly_score = 1.0 - similarity
            
            unusual_patterns = [
                "error", "fail", "crash", "exception", "unknown",
                "impossible", "invalid", "denied", "forbidden",
            ]
            content_lower = content.lower()
            for pattern in unusual_patterns:
                if pattern in content_lower:
                    anomaly_score += 0.1
            
            return min(1.0, anomaly_score)
        except Exception as e:
            logger.debug(f"Anomaly check failed: {e}")
            return 0.0
    
    def _find_patterns(self, embedding: list, source: str = "") -> list[str]:
        """Find similar concept patterns and store new ones."""
        patterns = []
        
        try:
            import numpy as np
            emb = np.array(embedding)
            
            with self._lock:
                for key, stored_emb in self._pattern_embeddings.items():
                    stored = np.array(stored_emb)
                    similarity = np.dot(emb, stored) / (np.linalg.norm(emb) * np.linalg.norm(stored) + 1e-8)
                    if similarity > 0.85:
                        patterns.append(key)
            
            # Store new pattern if no close match found
            if not patterns and source:
                pattern_id = f"pat_{hashlib.md5(source.encode()).hexdigest()[:8]}"
                self._store_pattern_embedding(pattern_id, embedding)
        
        except Exception as e:
            logger.debug(f"Pattern detection failed: {e}")
        
        return patterns[:5]
    
    def _find_cross_refs(self, embedding: list, source: str) -> list[dict]:
        """Find cross-references in different memory systems."""
        cross_refs = []
        
        try:
            kg = self._get_knowledge_graph()
            if kg:
                similar = kg.find_similar_nodes(embedding, limit=3)
                for node_id, score in similar:
                    node = kg.nodes.get(node_id)
                    if node:
                        cross_refs.append({
                            "type": "knowledge_graph",
                            "node_id": node_id,
                            "node_type": node.node_type.value,
                            "score": float(score),
                        })
        except Exception as e:
            logger.debug(f"Knowledge graph lookup failed: {e}")
        
        try:
            vm = self._get_vector_memory()
            if vm:
                results = vm.search(embedding, k=3)
                for result in results[:3]:
                    if len(result) >= 3:
                        entry_id, score, metadata = result
                        cross_refs.append({
                            "type": "vector_memory",
                            "id": entry_id,
                            "score": score,
                        })
        except Exception as e:
            logger.debug(f"Vector memory lookup failed: {e}")
        
        return cross_refs
    
    def _get_knowledge_graph(self):
        """Lazy load knowledge graph."""
        try:
            from core.knowledge_graph import get_knowledge_graph
            return get_knowledge_graph()
        except Exception as e:
            logger.warning(f"Failed to get knowledge graph: {e}")
            return None
    
    def _get_vector_memory(self):
        """Lazy load vector memory."""
        try:
            from core.vector_memory import get_vector_memory
            return get_vector_memory()
        except Exception as e:
            logger.warning(f"Failed to get vector memory: {e}")
            return None
    
    async def assemble_context_with_guidance(
        self,
        query: str,
        max_tokens: int = 16000,
        project_path: str | None = None,
        include_concept_guidance: bool = True,
    ) -> dict:
        """Assemble context with LCM-derived guidance.
        
        This is the MAIN method for getting context for dual models.
        
        Args:
            query: Current query
            max_tokens: Maximum tokens
            project_path: Optional project path
            include_concept_guidance: Whether to include LCM guidance
        
        Returns:
            Dict with messages, guidance, and metadata
        """
        self._ensure_initialized()
        
        result = {
            "messages": [],
            "concept_guidance": None,
            "meta_guidance": None,
            "anomaly_warning": None,
            "pattern_context": [],
            "layers_used": {},
        }
        
        try:
            if self._unified_context:
                from core.unified_context import assemble_context
                assembled = await assemble_context(
                    query=query,
                    max_tokens=max_tokens,
                    project_path=project_path,
                )
                result["messages"] = assembled.messages
                result["layers_used"] = assembled.layers_used
            else:
                result["messages"] = [{"role": "system", "content": f"Query: {query}"}]
            
            if include_concept_guidance and self._concept_model:
                try:
                    meta_guidance = await self._concept_model.guide_primary(
                        user_query=query,
                        conversation_context="",
                    )
                    result["meta_guidance"] = meta_guidance
                    
                    if len(result["messages"]) > 0 and meta_guidance:
                        guidance_msg = {
                            "role": "system",
                            "content": f"[Concept Guidance]\n{meta_guidance[:500]}"
                        }
                        result["messages"].insert(0, guidance_msg)
                        
                except Exception as e:
                    logger.debug(f"Concept guidance failed: {e}")
            
            # Include Thought Sense state for adaptive behavior
            try:
                from core.thought_sense import get_thought_sense, CognitiveState
                thought_sense = get_thought_sense()
                ts_state = thought_sense.get_state()
                
                if ts_state.cognitive_state != CognitiveState.IDLE:
                    depth = thought_sense.get_adapted_response_depth()
                    tone = thought_sense.get_adapted_tone()
                    rhythm = thought_sense.get_conversation_rhythm()
                    
                    thought_msg = {
                        "role": "system",
                        "content": f"[Thought Sense] Cognitive: {ts_state.cognitive_state.value}, "
                                    f"Load: {ts_state.cognitive_load:.1f}, "
                                    f"Tone: {tone}, Depth: {depth}, "
                                    f"Rhythm: {rhythm.get('cadence', 'unknown')}"
                    }
                    result["thought_sense"] = {
                        "cognitive_state": ts_state.cognitive_state.name,
                        "emotional_valence": ts_state.emotional_valence.name,
                        "cognitive_load": round(ts_state.cognitive_load, 2),
                        "emotional_trajectory": round(ts_state.emotional_trajectory, 2),
                        "predicted_needs": ts_state.predicted_needs,
                        "adapted_depth": depth,
                        "adapted_tone": tone,
                        "confidence": round(ts_state.confidence, 2),
                        "rhythm": rhythm,
                    }
                    result["messages"].append(thought_msg)
                    
            except Exception as e:
                logger.debug(f"Thought Sense integration failed: {e}")
            
            with self._lock:
                recent_cascades = list(self._cascade_history)[-5:]
            
            if recent_cascades:
                patterns = []
                for cascade in recent_cascades:
                    if cascade.concept_patterns:
                        patterns.extend(cascade.concept_patterns)
                
                if patterns:
                    unique_patterns = list(set(patterns))[:3]
                    result["pattern_context"] = unique_patterns
                    
                    pattern_msg = {
                        "role": "system",
                        "content": "[Recent Patterns]\n" + "\n".join(f"- {p}" for p in unique_patterns)
                    }
                    result["messages"].insert(1, pattern_msg)
            
            anomalies = [c for c in recent_cascades if c.anomaly_detected]
            if anomalies:
                result["anomaly_warning"] = anomalies[-1].anomaly_reason
                
        except Exception as e:
            logger.error(f"Context cascade assembly failed: {e}")
            result["messages"] = [{"role": "user", "content": query}]
        
        return result
    
    def get_cascade_stats(self) -> dict:
        """Get statistics about the cascade."""
        with self._lock:
            recent = list(self._cascade_history)[-20:]
            
            return {
                "total_cascades": len(self._cascade_history),
                "stored_embeddings": len(self._concept_embeddings),
                "pfc_available": self._pfc is not None,
                "concept_model_available": self._concept_model is not None,
                "embedding_available": self._embedding is not None,
                "unified_context_available": self._unified_context is not None,
                "recent_anomalies": sum(1 for c in recent if c.anomaly_detected),
                "recent_patterns_found": sum(len(c.concept_patterns) for c in recent),
            }
    
    def health_check(self) -> tuple[bool, dict]:
        """Return health status of the context cascade.
        
        Returns:
            Tuple of (is_healthy, status_dict) where status_dict contains:
            - healthy: True if at least one subsystem is available
            - components: dict of component availability
            - total_observations: count of cascade observations
            - error: error message if unhealthy
        """
        try:
            stats = self.get_cascade_stats()
            
            components = {
                "pfc": stats.get("pfc_available", False),
                "concept_model": stats.get("concept_model_available", False),
                "embedding": stats.get("embedding_available", False),
                "unified_context": stats.get("unified_context_available", False),
            }
            
            is_healthy = any(components.values())
            
            status = {
                "healthy": is_healthy,
                "components": components,
                "total_observations": stats.get("total_cascades", 0),
            }
            
            if not is_healthy:
                status["error"] = "No cascade subsystems available"
            
            return (is_healthy, status)
            
        except Exception as e:
            return (False, {"healthy": False, "error": safe_error_message(e)})


# Singleton
_cascade_instance: ContextCascadeIntegration | None = None
_cascade_lock = threading.Lock()


def get_context_cascade() -> ContextCascadeIntegration:
    """Get or create the context cascade integration."""
    global _cascade_instance
    
    with _cascade_lock:
        if _cascade_instance is None:
            _cascade_instance = ContextCascadeIntegration()
    return _cascade_instance


def observe_activity(trigger_type: str, content: str, source: str = "unknown") -> ConceptCascade:
    """Convenience function to observe activity."""
    cascade = get_context_cascade()
    return cascade.observe(trigger_type, content, source)


async def get_guided_context(
    query: str,
    max_tokens: int = 16000,
    project_path: str | None = None,
) -> dict:
    """Convenience function to get context with concept guidance."""
    cascade = get_context_cascade()
    return await cascade.assemble_context_with_guidance(
        query=query,
        max_tokens=max_tokens,
        project_path=project_path,
    )