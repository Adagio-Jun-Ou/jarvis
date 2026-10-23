"""
Memory Integration Hub - Unified interface for all memory systems.

This module provides intelligent routing between:
1. Conversation Memory (short-term context)
2. Vector Memory (semantic search via HNSW)
3. Knowledge Graph (relationships + PageRank)
4. Unified Memory (consolidated storage)

Key improvements:
- Uses EmbeddingInterconnect for ALL semantic operations
- Intelligent retrieval combining graph + vector + recency
- Matryoshka embedding support for dimension reduction
- Forgetting curves for automatic memory decay
- Cross-system knowledge sharing
"""
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

# Singleton
_memory_hub: Optional["MemoryHub"] = None
_hub_lock = threading.Lock()


@dataclass
class RetrievalResult:
    """Result from memory retrieval."""
    content: str
    source: str  # 'conversation', 'vector', 'knowledge', 'persistent'
    score: float
    timestamp: datetime
    importance: float
    metadata: dict = field(default_factory=dict)


@dataclass
class MemoryStats:
    """Memory system statistics."""
    conversation_turns: int = 0
    vector_entries: int = 0
    knowledge_nodes: int = 0
    knowledge_edges: int = 0
    persistent_memories: int = 0
    embedding_cache_hits: int = 0
    embedding_cache_misses: int = 0
    last_retrieval_ms: float = 0.0


class MemoryHub:
    """
    Central hub for all memory operations in Jarvis.
    
    Architecture:
    ┌──────────────────────────────────────────────────────────────────────────┐
    │                           MEMORY HUB                                      │
    ├──────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
    │   │  EMBEDDING    │    │   VECTOR     │    │  KNOWLEDGE   │              │
    │   │  INTERCONNECT │◄──►│   MEMORY     │◄──►│    GRAPH     │              │
    │   │  (Jina v5)    │    │   (HNSW)     │    │  (PageRank)  │              │
    │   └──────┬───────┘    └──────┬───────┘    └──────┬───────┘              │
    │          │                   │                    │                       │
    │          └───────────────────┼────────────────────┘                       │
    │                              │                                            │
    │   ┌──────────────────────────▼────────────────────────────────────────┐  │
    │   │                    UNIFIED MEMORY                                  │  │
    │   │  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐            │  │
    │   │  │ CONVERSATION  │ │  PERSISTENT   │ │   INSIGHT     │            │  │
    │   │  │   (context)   │ │   (facts)     │ │  (patterns)   │            │  │
    │   │  └───────────────┘ └───────────────┘ └───────────────┘            │  │
    │   └────────────────────────────────────────────────────────────────────┘  │
    │                                                                          │
    │   RETRIEVAL STRATEGIES:                                                  │
    │   1. Semantic search → Vector Memory (HNSW)                            │
    │   2. Relationship traversal → Knowledge Graph                          │
    │   3. Recency + importance → Conversation Memory                        │
    │   4. All combined → Weighted fusion                                     │
    │                                                                          │
    └──────────────────────────────────────────────────────────────────────────┘
    
    Matryoshka Embeddings:
    - Full: 768-dim for high-precision tasks
    - 512-dim for general semantic search
    - 256-dim for fast approximate search
    - 128-dim for ultra-fast clustering
    
    Forgetting Curve:
    - Memory importance decays: importance * e^(-λt)
    - λ configurable per memory type
    - Consolidation threshold: 0.7 for persistent storage
    """
    
    # Memory decay parameters (lambda for exponential decay)
    DECAY_RATES = {
        'conversation': 0.1,    # Fast decay - conversations age quickly
        'working': 0.5,         # Very fast decay - task context
        'semantic': 0.02,       # Slow decay - semantic knowledge
        'knowledge': 0.01,      # Very slow decay - graph nodes
        'insight': 0.005,      # Minimal decay - learned patterns
        'persistent': 0.001,   # Near-permanent - user preferences
    }
    
    # Matryoshka dimensions
    MATROSHKA_DIMS = [768, 512, 256, 128]
    
    # Retrieval weights
    RETRIEVAL_WEIGHTS = {
        'semantic': 0.4,       # Vector similarity
        'recency': 0.2,       # Time-based
        'importance': 0.2,    # User-marked importance
        'pagerank': 0.2,     # Knowledge graph authority
    }
    
    MAX_CONVERSATION_TURNS = 100
    MAX_WORKING_MEMORY = 50
    CONSOLIDATION_THRESHOLD = 0.7

    # How often (seconds) to warn that the embedder is missing, instead of
    # once per retrieval call.
    EMBEDDER_WARNING_COOLDOWN = 300.0
    
    def __init__(self):
        # Subsystems (lazy loaded)
        self._embedding_interconnect = None
        self._vector_memory = None
        self._knowledge_graph = None
        self._conversation_memory = None
        self._unified_memory = None
        
        # Embedding cache for frequently accessed content
        self._embedding_cache: dict[str, tuple[np.ndarray, float]] = {}
        self._cache_max_size = 5000  # Generous for 32GB RAM
        self._cache_ttl = 3600  # 1 hour - embeddings are stable
        
        # Statistics
        self._stats = MemoryStats()
        self._lock = threading.RLock()
        
        # Working memory (task context)
        self._working_memory: dict[str, Any] = {}
        self._working_timestamps: dict[str, float] = {}
        
    def _get_embedding_interconnect(self):
        """Lazy load embedding interconnect."""
        if self._embedding_interconnect is None:
            try:
                from core.embedding_interconnect import get_embedding_interconnect
                self._embedding_interconnect = get_embedding_interconnect()
            except Exception as e:
                logger.warning(f"Embedding interconnect not available: {e}")
        return self._embedding_interconnect
    
    def _get_vector_memory(self):
        """Lazy load vector memory."""
        if self._vector_memory is None:
            try:
                from core.vector_memory import get_vector_memory
                self._vector_memory = get_vector_memory()
            except Exception as e:
                logger.warning(f"Vector memory not available: {e}")
        return self._vector_memory
    
    def _get_knowledge_graph(self):
        """Lazy load knowledge graph."""
        if self._knowledge_graph is None:
            try:
                from core.knowledge_graph import get_knowledge_graph
                self._knowledge_graph = get_knowledge_graph()
            except Exception as e:
                logger.warning(f"Knowledge graph not available: {e}")
        return self._knowledge_graph
    
    def _get_conversation_memory(self):
        """Lazy load conversation memory."""
        if self._conversation_memory is None:
            try:
                from core.memory import get_conversation_memory
                self._conversation_memory = get_conversation_memory()
            except Exception as e:
                logger.warning(f"Conversation memory not available: {e}")
        return self._conversation_memory
    
    def encode(self, text: str, dim: int = 768) -> np.ndarray | None:
        """
        Encode text to embedding with caching and Matryoshka support.

        Returns None when no embedder is available. A missing embedder used to
        fall back to a *random* vector, cached like a real one — every
        retrieval score became noise that looked plausible. Honest failure lets
        callers degrade to lexical scoring instead.

        Args:
            text: Text to encode
            dim: Target dimension (768, 512, 256, or 128 for Matryoshka)

        Returns:
            Embedding vector of specified dimension, or None if unavailable
        """
        # Create cache key from text hash + dimension
        import hashlib
        cache_key = f"{hashlib.sha256(text.encode()).hexdigest()[:16]}_{dim}"

        with self._lock:
            # Check cache
            if cache_key in self._embedding_cache:
                emb, timestamp = self._embedding_cache[cache_key]
                if time.time() - timestamp < self._cache_ttl:
                    self._stats.embedding_cache_hits += 1
                    return emb

            self._stats.embedding_cache_misses += 1

        # Get full embedding
        interconnect = self._get_embedding_interconnect()
        if not interconnect:
            self._warn_no_embedder()
            return None
        full_emb = interconnect.encode(text)
        if full_emb is None:
            self._warn_no_embedder()
            return None

        # Matryoshka truncation
        if dim < len(full_emb):
            emb = full_emb[:dim].copy()
            # Re-normalize after truncation (important for Matryoshka)
            emb /= np.linalg.norm(emb) + 1e-8
        else:
            emb = full_emb

        # Cache result
        with self._lock:
            self._embedding_cache[cache_key] = (emb, time.time())
            if len(self._embedding_cache) > self._cache_max_size:
                # Remove oldest entries
                keys_by_age = sorted(
                    self._embedding_cache.keys(),
                    key=lambda k: self._embedding_cache[k][1]
                )
                for key in keys_by_age[:len(self._embedding_cache) - self._cache_max_size // 2]:
                    del self._embedding_cache[key]

        return emb

    def _warn_no_embedder(self) -> None:
        """Log once per cooldown so a dead embedder is visible without spam."""
        now = time.time()
        if now - getattr(self, "_last_embedder_warning", 0.0) > self.EMBEDDER_WARNING_COOLDOWN:
            self._last_embedder_warning = now
            logger.warning(
                "MemoryHub: embedding interconnect unavailable — semantic "
                "retrieval falls back to lexical (word-overlap) scoring. "
                "Install/repair the embedding model for full semantic search."
            )
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        sources: list[str] | None = None,
        min_score: float = 0.3,
    ) -> list[RetrievalResult]:
        """
        Unified retrieval across all memory systems.
        
        Args:
            query: Search query
            top_k: Maximum results to return
            sources: Optional list of sources to search ['conversation', 'vector', 'knowledge', 'persistent']
            min_score: Minimum relevance score
        
        Returns:
            List of RetrievalResult sorted by combined score
        """
        start_time = time.time()
        sources = sources or ['conversation', 'vector', 'knowledge', 'persistent']
        results: list[RetrievalResult] = []
        
        # 1. Semantic search via vector memory
        if 'vector' in sources:
            vector_results = self._retrieve_from_vector(query, top_k)
            results.extend(vector_results)
        
        # 2. Knowledge graph traversal
        if 'knowledge' in sources:
            graph_results = self._retrieve_from_knowledge(query, top_k)
            results.extend(graph_results)
        
        # 3. Recent conversation
        if 'conversation' in sources:
            conv_results = self._retrieve_from_conversation(query, top_k)
            results.extend(conv_results)
        
        # 4. Persistent memories
        if 'persistent' in sources:
            persistent_results = self._retrieve_from_persistent(query, top_k)
            results.extend(persistent_results)
        
        # Combine and deduplicate
        combined = self._combine_results(results, query)
        
        # Filter by minimum score
        combined = [r for r in combined if r.score >= min_score]
        
        # Update stats
        self._stats.last_retrieval_ms = (time.time() - start_time) * 1000
        
        return combined[:top_k]
    
    def _retrieve_from_vector(self, query: str, top_k: int) -> list[RetrievalResult]:
        """Retrieve from vector memory using semantic search."""
        results = []
        vector_memory = self._get_vector_memory()
        
        if not vector_memory:
            return results
        
        try:
            # VectorMemory.search() encodes the query with its own embedder;
            # encoding here too would just burn a model call (and, in the old
            # random-fallback days, cache a poison vector for it).
            search_results = vector_memory.search(
                query=query,
                k=top_k,
            )
            
            # VectorMemory.search() yields (entry_id, score, metadata) tuples.
            # This used to probe them with hasattr(result, 'metadata'/'text'/'score'),
            # which is always False on a tuple — so every hit produced empty
            # content and a score of exactly decay_factor (1.0 for undated
            # entries), out-ranking every correctly-scored source.
            for entry_id, score, metadata in search_results:
                metadata = metadata or {}
                timestamp = metadata.get('timestamp') or metadata.get('stored_at')
                if timestamp:
                    age_hours = (datetime.now() - datetime.fromisoformat(timestamp)).total_seconds() / 3600
                    decay_factor = np.exp(-self.DECAY_RATES['semantic'] * age_hours)
                else:
                    decay_factor = 1.0

                content = metadata.get('content') or metadata.get('text') or ''

                results.append(RetrievalResult(
                    content=content[:500],
                    source='vector',
                    score=score * decay_factor,
                    timestamp=datetime.fromisoformat(timestamp) if timestamp else datetime.now(),
                    importance=metadata.get('importance', 0.5),
                    metadata={**metadata, 'entry_id': entry_id},
                ))
                
                self._stats.vector_entries = vector_memory.count() if hasattr(vector_memory, 'count') else len(vector_memory.index.nodes)
        except Exception as e:
            logger.debug(f"Vector retrieval error: {e}")
        
        return results
    
    def _retrieve_from_knowledge(self, query: str, top_k: int) -> list[RetrievalResult]:
        """Retrieve from knowledge graph using semantic + PageRank."""
        results = []
        knowledge_graph = self._get_knowledge_graph()
        
        if not knowledge_graph:
            return results
        
        try:
            # Use embedding interconnect for semantic search
            interconnect = self._get_embedding_interconnect()
            if interconnect:
                # Semantic search over knowledge nodes
                node_results = interconnect.search_knowledge(query, top_k=top_k * 2)
                
                for node_id, sim_score in node_results:
                    # Get node details
                    if node_id in knowledge_graph.nodes:
                        node = knowledge_graph.nodes[node_id]
                        
                        # Combine semantic similarity with PageRank
                        pagerank_score = node.pagerank
                        combined_score = (
                            self.RETRIEVAL_WEIGHTS['semantic'] * sim_score +
                            self.RETRIEVAL_WEIGHTS['pagerank'] * pagerank_score
                        )
                        
                        results.append(RetrievalResult(
                            content=node.content,
                            source='knowledge',
                            score=combined_score,
                            timestamp=node.created_at,
                            importance=node.pagerank,
                            metadata={'node_id': node_id, 'node_type': node.node_type.value},
                        ))
                
                self._stats.knowledge_nodes = len(knowledge_graph.nodes)
                self._stats.knowledge_edges = len(knowledge_graph.edge_details)
        except Exception as e:
            logger.debug(f"Knowledge retrieval error: {e}")
        
        return results
    
    def _retrieve_from_conversation(self, query: str, top_k: int) -> list[RetrievalResult]:
        """Retrieve from recent conversation."""
        results = []
        conv_memory = self._get_conversation_memory()
        
        if not conv_memory or not hasattr(conv_memory, 'messages'):
            return results
        
        try:
            # Get recent messages
            messages = conv_memory.messages[-50:] if conv_memory.messages else []

            if not messages:
                return results

            # Encode query for semantic matching; when the embedder is
            # unavailable (None), degrade to lexical scoring — the previous
            # random-vector fallback made every similarity noise.
            query_emb = self.encode(query, dim=256)  # Fast dimension for re-ranking
            query_words = self._lexical_tokens(query)

            for msg in messages:
                if isinstance(msg, dict):
                    content = msg.get('content', '')
                    role = msg.get('role', 'unknown')
                    timestamp = msg.get('timestamp', datetime.now())
                else:
                    content = str(msg)
                    role = 'unknown'
                    timestamp = datetime.now()

                if query_emb is not None:
                    # Semantic similarity
                    msg_emb = self.encode(content[:200], dim=256)
                    if msg_emb is None:
                        sim = 0.0
                    else:
                        sim = float(np.dot(query_emb, msg_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(msg_emb) + 1e-8))
                else:
                    # Lexical fallback: Jaccard word overlap, rescaled to the
                    # ~[0, 1] range cosine occupies for related texts.
                    overlap = self._lexical_similarity(query_words, content)
                    sim = overlap

                # Recency score (newer = higher)
                if isinstance(timestamp, datetime):
                    age_hours = (datetime.now() - timestamp).total_seconds() / 3600
                else:
                    age_hours = 0
                recency_score = np.exp(-self.DECAY_RATES['conversation'] * age_hours)
                
                # Combined score
                combined_score = (
                    self.RETRIEVAL_WEIGHTS['semantic'] * sim +
                    self.RETRIEVAL_WEIGHTS['recency'] * recency_score
                )
                
                if combined_score > 0.1:  # Threshold
                    results.append(RetrievalResult(
                        content=content[:500],
                        source='conversation',
                        score=combined_score,
                        timestamp=timestamp if isinstance(timestamp, datetime) else datetime.now(),
                        importance=0.5,
                        metadata={'role': role},
                    ))
            
            self._stats.conversation_turns = len(messages)
        except Exception as e:
            logger.debug(f"Conversation retrieval error: {e}")

        return results

    # ------------------------------------------------------------------
    # Lexical fallback (used when the embedding model is unavailable)
    # ------------------------------------------------------------------

    _LEXICAL_STOP_WORDS = frozenset({
        "a", "an", "the", "to", "for", "and", "or", "in", "on", "is", "are",
        "was", "were", "be", "been", "being", "have", "has", "had", "do",
        "does", "did", "will", "would", "could", "should", "may", "might",
        "must", "shall", "can", "of", "at", "by", "with", "about", "into",
        "from", "that", "this", "these", "those", "it", "its", "as", "i",
        "you", "he", "she", "we", "they", "me", "my", "your", "what", "how",
        "when", "where", "why", "which", "who", "not", "no", "so", "than",
        "too", "very", "just", "there", "here", "am",
    })

    @classmethod
    def _lexical_tokens(cls, text: str) -> set[str]:
        """Lowercased word set minus stop words — the basis of lexical scoring."""
        words = set(text.lower().split())
        # Trim punctuation so "end." matches "end".
        return {w.strip(".,!?;:()[]{}\"'`") for w in words} - cls._LEXICAL_STOP_WORDS

    @classmethod
    def _lexical_similarity(cls, query_words: set[str], text: str) -> float:
        """Jaccard word overlap between the query and *text*, on [0, 1].

        A deliberately simple stand-in for cosine similarity when embeddings
        are unavailable: honest (0 for unrelated text) and deterministic,
        where the retired random-vector fallback produced noise that looked
        like a score.
        """
        if not query_words:
            return 0.0
        text_words = cls._lexical_tokens(text)
        if not text_words:
            return 0.0
        intersection = len(query_words & text_words)
        union = len(query_words | text_words)
        return intersection / union if union else 0.0

    def _retrieve_from_persistent(self, query: str, top_k: int) -> list[RetrievalResult]:
        """Retrieve from persistent memory (user facts, preferences)."""
        results = []
        
        try:
            interconnect = self._get_embedding_interconnect()
            if interconnect:
                memory_results = interconnect.search_memories(query, top_k=top_k)
                
                for memory_id, score in memory_results:
                    results.append(RetrievalResult(
                        content=memory_id,  # Would need to look up actual content
                        source='persistent',
                        score=score,
                        timestamp=datetime.now(),
                        importance=0.8,  # Persistent memories are important
                        metadata={'memory_id': memory_id},
                    ))
                
                ic_stats = interconnect.get_stats()
                self._stats.persistent_memories = ic_stats["memory_embeddings"]
        except Exception as e:
            logger.debug(f"Persistent retrieval error: {e}")
        
        return results
    
    def _combine_results(self, results: list[RetrievalResult], query: str) -> list[RetrievalResult]:
        """Combine and deduplicate results from multiple sources."""
        # Group by content hash
        seen = {}
        
        for result in results:
            # Create content hash for deduplication
            import hashlib
            content_hash = hashlib.sha256(result.content.encode()).hexdigest()[:16]
            
            if content_hash in seen:
                # Combine scores from different sources
                existing = seen[content_hash]
                # Take max score and merge metadata
                existing.score = max(existing.score, result.score)
                existing.metadata.update(result.metadata)
            else:
                seen[content_hash] = result
        
        # Sort by combined score
        combined = list(seen.values())
        combined.sort(key=lambda r: r.score, reverse=True)
        
        return combined
    
    def store(
        self,
        content: str,
        memory_type: str = 'semantic',
        importance: float = 0.5,
        metadata: dict | None = None,
    ) -> str:
        """
        Store content in appropriate memory subsystem.
        
        Args:
            content: Content to store
            memory_type: 'conversation', 'semantic', 'knowledge', 'insight', 'persistent'
            importance: 0-1 importance score
            metadata: Additional metadata
        
        Returns:
            Memory ID
        """
        import uuid
        memory_id = str(uuid.uuid4())[:8]
        metadata = metadata or {}
        metadata['timestamp'] = datetime.now().isoformat()
        metadata['importance'] = importance
        metadata['content'] = content
        
        # Store in embedding interconnect (all types)
        interconnect = self._get_embedding_interconnect()
        if interconnect:
            if memory_type == 'knowledge':
                interconnect.store_knowledge_node(memory_id, content)
            elif memory_type == 'insight':
                interconnect.store_insight(memory_id, content)
            elif memory_type in ('persistent', 'semantic'):
                interconnect.store_memory(memory_id, content)
        
        # Store in vector memory (semantic types)
        if memory_type in ('semantic', 'knowledge', 'insight'):
            vector_memory = self._get_vector_memory()
            if vector_memory:
                vector_memory.store(
                    text=content,
                    metadata=metadata,
                    entry_id=memory_id,
                )
        
        # Store in knowledge graph (knowledge type)
        if memory_type == 'knowledge':
            knowledge_graph = self._get_knowledge_graph()
            if knowledge_graph:
                from core.knowledge_graph import NodeType
                knowledge_graph.add_node(
                    content=content,
                    node_type=NodeType.INSIGHT,
                    metadata=metadata,
                )
        
        return memory_id
    
    def get_stats(self) -> MemoryStats:
        """Get memory system statistics."""
        return self._stats
    
    def clear_working_memory(self):
        """Clear working memory (call between tasks)."""
        with self._lock:
            self._working_memory.clear()
            self._working_timestamps.clear()
    
    def set_working_memory(self, key: str, value: Any):
        """Store item in working memory with eviction."""
        with self._lock:
            if len(self._working_memory) >= self.MAX_WORKING_MEMORY:
                oldest = min(self._working_timestamps, key=self._working_timestamps.get, default=None)
                if oldest:
                    self._working_memory.pop(oldest, None)
                    self._working_timestamps.pop(oldest, None)
            self._working_memory[key] = value
            self._working_timestamps[key] = time.time()
    
    def get_working_memory(self, key: str, default: Any = None) -> Any:
        """Get item from working memory."""
        with self._lock:
            return self._working_memory.get(key, default)


def get_memory_hub() -> MemoryHub:
    """Get singleton memory hub instance."""
    global _memory_hub
    if _memory_hub is None:
        with _hub_lock:
            if _memory_hub is None:
                _memory_hub = MemoryHub()
                logger.info("Memory Hub initialized")
    return _memory_hub