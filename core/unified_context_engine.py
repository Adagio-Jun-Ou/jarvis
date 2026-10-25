"""Optimized Unified Context Engine with performance improvements.

Key optimizations:
1. Lazy subsystem loading with caching
2. Parallel fetching with asyncio.gather
3. Embedding caching to avoid recomputation
4. Token estimation without loops
5. Cache key with collision-free hashing
6. Batched similarity computation
7. Early termination when budget reached
8. Pre-computed priority scores
"""

import asyncio
import hashlib
from core.helpers import atomic_write_json, read_json_with_integrity
import logging
import threading
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

from core.unified_memory import MemoryType

logger = logging.getLogger(__name__)

# Singleton
_unified_context: Optional["UnifiedContextEngine"] = None
_context_lock = threading.Lock()


class ContextSource(Enum):
    SYSTEM = "system"
    CONVERSATION_RECENT = "conversation_recent"
    CONVERSATION_IMPORTANT = "conversation_important"
    INSIGHT_ACTIVE = "insight_active"
    INSIGHT_RELEVANT = "insight_relevant"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    USER_FACTS = "user_facts"
    CROSS_MODEL = "cross_model"
    WORKING_MEMORY = "working_memory"
    SUBSYSTEM_STATUS = "subsystem_status"


class ContextPriority(Enum):
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3


# Pre-computed priority scores (avoid division on each call)
PRIORITY_SCORES = {
    ContextPriority.CRITICAL: 1.0,
    ContextPriority.HIGH: 0.75,
    ContextPriority.MEDIUM: 0.5,
    ContextPriority.LOW: 0.25,
}

# Scoring weights
SCORE_WEIGHTS = {
    "relevance": 0.35,
    "importance": 0.25,
    "recency": 0.20,
    "priority": 0.20,
}


@dataclass
class ContextEntry:
    """Optimized context entry with pre-computed scoring."""
    id: str
    content: str
    source: ContextSource
    priority: ContextPriority
    relevance_score: float = 0.0
    importance: float = 0.5
    recency: float = 1.0
    tokens: int = 0
    model: str = "primary"
    embedding: Optional[np.ndarray] = None
    connections: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    _score: float = field(default=0.0, repr=False)

    def compute_score(self) -> float:
        """Compute overall score for ranking - optimized version."""
        if self._score > 0:
            return self._score
        
        self._score = (
            SCORE_WEIGHTS["relevance"] * self.relevance_score +
            SCORE_WEIGHTS["importance"] * self.importance +
            SCORE_WEIGHTS["recency"] * self.recency +
            SCORE_WEIGHTS["priority"] * PRIORITY_SCORES[self.priority]
        )
        return self._score

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "source": self.source.value,
            "priority": self.priority.value,
            "relevance_score": self.relevance_score,
            "importance": self.importance,
            "recency": self.recency,
            "tokens": self.tokens,
            "model": self.model,
        }


@dataclass
class ContextPack:
    """Assembled context package ready for LLM."""
    entries: list[ContextEntry]
    total_tokens: int
    query: str
    sources_included: list[ContextSource]
    sources_excluded: list[ContextSource]
    score_cutoff: float = 0.0
    assembly_time_ms: float = 0.0

    def to_prompt(self, format: str = "default") -> str:
        """Convert to prompt format - optimized."""
        if not self.entries:
            return ""

        if format == "compact":
            # Fastest: just join contents
            return "\n".join(e.content for e in self.entries)

        if format == "ranked":
            # Sorted by score (pre-computed)
            return "\n".join(
                f"[{e.source.value}] {e.content}" 
                for e in sorted(self.entries, key=lambda x: x.compute_score(), reverse=True)
            )

        # Default: group by source
        by_source: dict[ContextSource, list[ContextEntry]] = defaultdict(list)
        for entry in self.entries:
            by_source[entry.source].append(entry)

        sections = []
        for source in ContextSource:
            if source in by_source:
                entries = by_source[source]
                source_name = source.value.replace("_", " ").title()
                sections.append(f"## {source_name}\n" + "\n".join(e.content for e in entries))

        return "\n\n".join(sections)

    def to_dict(self) -> dict:
        return {
            "entries": [e.to_dict() for e in self.entries],
            "total_tokens": self.total_tokens,
            "query": self.query,
            "sources_included": [s.value for s in self.sources_included],
            "sources_excluded": [s.value for s in self.sources_excluded],
            "score_cutoff": self.score_cutoff,
            "assembly_time_ms": self.assembly_time_ms,
        }


@dataclass
class SearchResult:
    """A search result from knowledge query."""
    id: str
    content: str
    source: str
    score: float
    metadata: dict = field(default_factory=dict)


class UnifiedContextEngine:
    """Optimized unified context management with parallel fetching and caching."""

    # Configurable constants - generous limits for systems with ample RAM
    MAX_EMBEDDING_CACHE_SIZE = 10000  # ~60MB max, trivial for 32GB RAM
    MAX_CONTEXT_CACHE_SIZE = 500  # Bound for context cache to prevent unbounded growth
    CONTEXT_CACHE_TTL = 60.0  # 1 minute - context doesn't change rapidly
    EMBEDDING_CACHE_TTL = 300.0  # 5 minutes - embeddings are stable

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path.home() / ".jarvis" / "context"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()

        # Lazy-loaded subsystems
        self._memory = None
        self._persistent_memory = None
        self._unified_memory = None
        self._embedding_interconnect = None
        self._insight_bridge = None
        self._intuition_bridge = None
        self._knowledge_graph = None

        # Context cache with size limit (OrderedDict for O(1) LRU)
        self._context_cache: OrderedDict[str, tuple[ContextPack, float]] = OrderedDict()
        self._cache_ttl = self.CONTEXT_CACHE_TTL
        self._last_assembly_time = 0.0

        # Embedding cache (avoid recomputation) — OrderedDict for LRU eviction
        self._embedding_cache: OrderedDict[str, tuple[np.ndarray, float]] = OrderedDict()
        self._embedding_cache_ttl = self.EMBEDDING_CACHE_TTL

        # Statistics
        self._stats = {
            "assemblies": 0,
            "searches": 0,
            "stores": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "embedding_cache_hits": 0,
        }

        self._initialized = False

    def _initialize(self):
        """Lazy initialization of subsystems with double-checked locking."""
        if self._initialized:
            return
        
        with self._lock:
            if self._initialized:
                return
            
            # Pre-load frequently used subsystems
            try:
                from .persistent_memory import get_persistent_memory
                self._persistent_memory = get_persistent_memory()
            except Exception as e:
                logger.warning(f"Failed to initialize persistent_memory: {e}")

            try:
                from .insight_bridge import get_insight_bridge
                self._insight_bridge = get_insight_bridge()
            except Exception as e:
                logger.warning(f"Failed to initialize insight_bridge: {e}")

            try:
                from .knowledge_graph import get_knowledge_graph
                self._knowledge_graph = get_knowledge_graph()
            except Exception as e:
                logger.warning(f"Failed to initialize knowledge_graph: {e}")

            self._initialized = True

    def _get_memory(self):
        """Get conversation memory with lazy loading and None check."""
        if self._memory is None:
            with self._lock:
                if self._memory is None:
                    try:
                        from .memory import get_conversation_memory
                        self._memory = get_conversation_memory()
                    except Exception as e:
                        logger.warning(f"Failed to get conversation memory: {e}")
                        return None
        return self._memory

    def _get_persistent_memory(self):
        """Get persistent memory with lazy loading and None check."""
        if self._persistent_memory is None:
            with self._lock:
                if self._persistent_memory is None:
                    try:
                        from .persistent_memory import get_persistent_memory
                        self._persistent_memory = get_persistent_memory()
                    except Exception as e:
                        logger.warning(f"Failed to get persistent memory: {e}")
                        return None
        return self._persistent_memory

    def _get_unified_memory(self):
        """Get unified memory with lazy loading and None check."""
        if self._unified_memory is None:
            with self._lock:
                if self._unified_memory is None:
                    try:
                        from .unified_memory import get_unified_memory
                        self._unified_memory = get_unified_memory()
                    except Exception as e:
                        logger.warning(f"Failed to get unified memory: {e}")
                        return None
        return self._unified_memory

    def _get_embedding_interconnect(self):
        """Get embedding interconnect with lazy loading and None check."""
        if self._embedding_interconnect is None:
            with self._lock:
                if self._embedding_interconnect is None:
                    try:
                        from .embedding_interconnect import get_embedding_interconnect
                        self._embedding_interconnect = get_embedding_interconnect()
                    except Exception as e:
                        logger.warning(f"Failed to get embedding interconnect: {e}")
                        return None
        return self._embedding_interconnect

    def _get_insight_bridge(self):
        """Get insight bridge with lazy loading and None check."""
        if self._insight_bridge is None:
            with self._lock:
                if self._insight_bridge is None:
                    try:
                        from .insight_bridge import get_insight_bridge
                        self._insight_bridge = get_insight_bridge()
                    except Exception as e:
                        logger.warning(f"Failed to get insight bridge: {e}")
                        return None
        return self._insight_bridge

    def _get_intuition_bridge(self):
        """Get intuition bridge with lazy loading and None check."""
        if self._intuition_bridge is None:
            with self._lock:
                if self._intuition_bridge is None:
                    try:
                        from .intuition_bridge import get_intuition_bridge
                        self._intuition_bridge = get_intuition_bridge()
                    except Exception as e:
                        logger.warning(f"Failed to get intuition bridge: {e}")
                        return None
        return self._intuition_bridge

    def _get_knowledge_graph(self):
        """Get knowledge graph with lazy loading and None check."""
        if self._knowledge_graph is None:
            with self._lock:
                if self._knowledge_graph is None:
                    try:
                        from .knowledge_graph import get_knowledge_graph
                        self._knowledge_graph = get_knowledge_graph()
                    except Exception as e:
                        logger.warning(f"Failed to get knowledge graph: {e}")
                        return None
        return self._knowledge_graph

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count - optimized with internal cache."""
        return max(1, len(text) // 4)

    def _encode(self, text: str) -> Optional[np.ndarray]:
        """Encode text to embedding with caching.
        
        Uses full text hash as cache key to avoid collisions.
        Implements LRU eviction when cache exceeds max size.
        """
        if not text:
            return None

        # Use hash of full text as cache key to avoid collisions
        cache_key = hashlib.sha256(text.encode()).hexdigest()
        now = time.time()
        
        with self._lock:
            if cache_key in self._embedding_cache:
                emb, cache_time = self._embedding_cache[cache_key]
                if (now - cache_time) < self._embedding_cache_ttl:
                    self._stats["embedding_cache_hits"] += 1
                    self._embedding_cache.move_to_end(cache_key)  # LRU promotion
                    return emb
                else:
                    del self._embedding_cache[cache_key]

        try:
            interconnect = self._get_embedding_interconnect()
            if interconnect is None:
                return None
                
            result = interconnect.encode(text)
            
            # Convert to numpy array for faster operations
            if result is not None:
                emb = np.array(result, dtype=np.float32)
                
                with self._lock:
                    # Evict oldest if over limit
                    while len(self._embedding_cache) >= self.MAX_EMBEDDING_CACHE_SIZE:
                        # Remove oldest entry
                        oldest_key = next(iter(self._embedding_cache))
                        del self._embedding_cache[oldest_key]
                    
                    self._embedding_cache[cache_key] = (emb, now)
                return emb
            return None
        except Exception as e:
            logger.warning(f"Failed to encode text: {e}")
            return None

    def _compute_similarity_fast(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Fast cosine similarity using numpy."""
        if emb1 is None or emb2 is None:
            return 0.0
        
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(emb1, emb2) / (norm1 * norm2))

    def _compute_batch_similarity(self, query_emb: np.ndarray, embeddings: list[np.ndarray]) -> list[float]:
        """Compute similarities in batch for speed."""
        if not embeddings or query_emb is None:
            return []
        
        # Stack embeddings into matrix
        emb_matrix = np.vstack(embeddings)
        
        # Compute norms once
        query_norm = np.linalg.norm(query_emb)
        if query_norm == 0:
            return [0.0] * len(embeddings)
        
        emb_norms = np.linalg.norm(emb_matrix, axis=1)
        
        # Avoid division by zero - replace zero norms with 1.0 (will give 0 similarity)
        valid = emb_norms > 0
        similarities = np.zeros(len(embeddings))
        similarities[valid] = np.dot(emb_matrix[valid], query_emb) / (query_norm * emb_norms[valid])
        
        return similarities.tolist()

    async def assemble_context(
        self,
        query: str,
        max_tokens: int = 4000,
        include_sources: Optional[list[ContextSource]] = None,
        exclude_sources: Optional[list[ContextSource]] = None,
        include_insights: bool = True,
        include_cross_model: bool = True,
        model: str = "primary",
        min_relevance: float = 0.3,
    ) -> ContextPack:
        """Assemble context from all sources in parallel.
        
        Args:
            query: User query for context relevance
            max_tokens: Maximum tokens for context budget (default 4000)
            include_sources: Specific sources to include (None = all)
            exclude_sources: Sources to exclude
            include_insights: Include insight-based context
            include_cross_model: Include cross-model context
            model: Target model for context ("primary" or "backend")
            min_relevance: Minimum relevance score threshold (0.0-1.0)
        
        Returns:
            ContextPack with assembled context entries
        
        Raises:
            ValueError: If query is empty or invalid parameters
        """
        start_time = time.time()
        
        # Input validation
        if not query or not query.strip():
            logger.warning("Empty query provided to assemble_context")
            return ContextPack(
                entries=[],
                total_tokens=0,
                query="",
                sources_included=[],
                sources_excluded=[],
                score_cutoff=0.0,
                assembly_time_ms=0.0,
            )
        
        # Validate and clamp parameters
        max_tokens = max(100, min(50000, max_tokens))  # Clamp to reasonable range
        min_relevance = max(0.0, min(1.0, min_relevance))  # Clamp to 0-1 range
        
        self._initialize()
        with self._lock:
            self._stats["assemblies"] += 1

        # Generate cache key including all parameters
        cache_key = hashlib.sha256(
            f"{query}:{max_tokens}:{include_insights}:{include_cross_model}:{include_sources}:{min_relevance}".encode()
        ).hexdigest()[:16]
        
        # Check cache
        with self._lock:
            if cache_key in self._context_cache:
                cached_pack, cached_time = self._context_cache[cache_key]
                if (time.time() - cached_time) < self._cache_ttl:
                    self._stats["cache_hits"] += 1
                    self._context_cache.move_to_end(cache_key)
                    return cached_pack
        
        with self._lock:
            self._stats["cache_misses"] += 1

        # Determine sources
        sources = include_sources or list(ContextSource)
        if exclude_sources:
            sources = [s for s in sources if s not in exclude_sources]

        # Encode query once for all semantic operations
        query_embedding = self._encode(query)

        # Fetch all sources in parallel
        fetch_tasks = []
        sources_to_fetch = []

        if ContextSource.CONVERSATION_RECENT in sources:
            fetch_tasks.append(self._fetch_conversation_recent(query_embedding))
            sources_to_fetch.append(ContextSource.CONVERSATION_RECENT)

        if ContextSource.CONVERSATION_IMPORTANT in sources:
            fetch_tasks.append(self._fetch_conversation_important(query_embedding))
            sources_to_fetch.append(ContextSource.CONVERSATION_IMPORTANT)

        if include_insights and ContextSource.INSIGHT_ACTIVE in sources:
            fetch_tasks.append(self._fetch_active_insights())
            sources_to_fetch.append(ContextSource.INSIGHT_ACTIVE)

        if include_insights and ContextSource.INSIGHT_RELEVANT in sources and query_embedding is not None:
            fetch_tasks.append(self._fetch_relevant_insights(query, query_embedding, min_relevance))
            sources_to_fetch.append(ContextSource.INSIGHT_RELEVANT)

        if ContextSource.KNOWLEDGE_GRAPH in sources and query_embedding is not None:
            fetch_tasks.append(self._fetch_knowledge_graph(query_embedding, min_relevance))
            sources_to_fetch.append(ContextSource.KNOWLEDGE_GRAPH)

        if ContextSource.USER_FACTS in sources:
            fetch_tasks.append(self._fetch_user_facts(query_embedding))
            sources_to_fetch.append(ContextSource.USER_FACTS)

        if include_cross_model and ContextSource.CROSS_MODEL in sources:
            fetch_tasks.append(self._fetch_cross_model_context(query, query_embedding, model))
            sources_to_fetch.append(ContextSource.CROSS_MODEL)

        if ContextSource.WORKING_MEMORY in sources:
            fetch_tasks.append(self._fetch_working_memory())
            sources_to_fetch.append(ContextSource.WORKING_MEMORY)

        if ContextSource.SUBSYSTEM_STATUS in sources:
            fetch_tasks.append(self._fetch_subsystem_status(query))
            sources_to_fetch.append(ContextSource.SUBSYSTEM_STATUS)

        # Execute all fetches in parallel
        if fetch_tasks:
            results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
        else:
            results = []

        # Collect entries
        entries: list[ContextEntry] = []
        sources_included: list[ContextSource] = []
        sources_excluded = [s for s in sources if s not in sources_to_fetch]

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.debug(f"Fetch failed for {sources_to_fetch[i]}: {result}")
                continue
            
            if result:
                entries.extend(result)
                sources_included.append(sources_to_fetch[i])

        # Compute scores in batch
        if query_embedding is not None:
            embeddings_to_score = []
            entries_with_embeddings = []
            
            for entry in entries:
                if entry.embedding is not None and entry.relevance_score == 0.0:
                    embeddings_to_score.append(entry.embedding)
                    entries_with_embeddings.append(entry)
            
            if embeddings_to_score:
                similarities = self._compute_batch_similarity(query_embedding, embeddings_to_score)
                for i, entry in enumerate(entries_with_embeddings):
                    entry.relevance_score = similarities[i]

        # Estimate tokens
        for entry in entries:
            entry.tokens = self._estimate_tokens(entry.content)

        # Sort by score
        entries.sort(key=lambda e: e.compute_score(), reverse=True)

        # Select within budget (early termination)
        selected: list[ContextEntry] = []
        total_tokens = 0

        for entry in entries:
            if entry.relevance_score < min_relevance and entry.priority != ContextPriority.CRITICAL:
                continue
            
            if total_tokens + entry.tokens <= max_tokens:
                selected.append(entry)
                total_tokens += entry.tokens
            elif total_tokens >= max_tokens * 0.8:
                # Stop early if we've filled 80% of budget
                break

        assembly_time_ms = (time.time() - start_time) * 1000

        pack = ContextPack(
            entries=selected,
            total_tokens=total_tokens,
            query=query,
            sources_included=sources_included,
            sources_excluded=sources_excluded,
            score_cutoff=selected[-1].compute_score() if selected else 0.0,
            assembly_time_ms=assembly_time_ms,
        )

        # Cache result with LRU eviction
        with self._lock:
            self._context_cache[cache_key] = (pack, time.time())
            self._context_cache.move_to_end(cache_key)
            
            # Evict if over limit (FIFO from oldest)
            while len(self._context_cache) > self.MAX_CONTEXT_CACHE_SIZE:
                self._context_cache.popitem(last=False)

            self._last_assembly_time = assembly_time_ms
        return pack

    async def _fetch_conversation_recent(self, query_embedding: Optional[np.ndarray]) -> list[ContextEntry]:
        """Fetch recent conversation messages."""
        entries = []
        try:
            memory = self._get_memory()
            if memory is None:
                return entries
            
            messages = memory.get_messages_for_llm(max_messages=10)
            if not messages:
                return entries

            for i, msg in enumerate(messages[-10:]):
                content = msg.get("content", "")
                if not content:
                    continue

                entry = ContextEntry(
                    id=f"conv_recent_{i}",
                    content=content[:500],
                    source=ContextSource.CONVERSATION_RECENT,
                    priority=ContextPriority.HIGH,
                    relevance_score=1.0 if i >= len(messages) - 3 else 0.8,
                    importance=msg.get("importance", 0.5),
                    recency=1.0 - (i * 0.05),
                    model="primary",
                )
                entries.append(entry)
        except Exception as e:
            logger.warning(f"Failed to fetch recent conversation: {e}")

        return entries

    async def _fetch_conversation_important(self, query_embedding: Optional[np.ndarray]) -> list[ContextEntry]:
        """Fetch important conversation messages."""
        entries = []
        try:
            memory = self._get_memory()
            if memory is None:
                return entries
            
            if hasattr(memory, "important_messages") and hasattr(memory, "history"):
                history_len = len(memory.history) if memory.history else 0
                
                for idx, importance in list(memory.important_messages.items())[:5]:
                    # Ensure idx is within bounds
                    if not isinstance(idx, int) or idx < 0 or idx >= history_len:
                        continue
                        
                    if importance > 0.7:
                        msg = memory.history[idx]
                        content = msg.get("content", "") if isinstance(msg, dict) else ""

                        entry = ContextEntry(
                            id=f"conv_important_{idx}",
                            content=content[:500],
                            source=ContextSource.CONVERSATION_IMPORTANT,
                            priority=ContextPriority.MEDIUM,
                            relevance_score=0.7,
                            importance=importance,
                            recency=0.5,
                        )
                        entries.append(entry)
        except Exception as e:
            logger.warning(f"Failed to fetch important messages: {e}")

        return entries

    async def _fetch_active_insights(self) -> list[ContextEntry]:
        """Fetch recent learned insights."""
        entries = []
        try:
            bridge = self._get_insight_bridge()
            if bridge is None:
                return entries
            
            recent = list(bridge.processed_insights)[-5:] if bridge.processed_insights else []

            for insight in recent:
                if getattr(insight, 'evolution_relevance', None) and insight.evolution_relevance.value >= 2:
                    entry = ContextEntry(
                        id=f"insight_{insight.id}",
                        content=f"[{getattr(insight, 'source_category', 'unknown')}] {getattr(insight, 'raw_insight', '')[:200]}",
                        source=ContextSource.INSIGHT_ACTIVE,
                        priority=ContextPriority.MEDIUM,
                        relevance_score=0.8,
                        importance=float(insight.evolution_relevance.value) / 3.0,
                        recency=0.9,
                    )
                    entries.append(entry)
        except Exception as e:
            logger.warning(f"Failed to fetch active insights: {e}")

        return entries[:5]

    async def _fetch_relevant_insights(
        self,
        query: str,
        query_embedding: np.ndarray,
        min_relevance: float,
    ) -> list[ContextEntry]:
        """Fetch insights semantically relevant to query."""
        entries = []
        try:
            interconnect = self._get_embedding_interconnect()
            if interconnect is None:
                return entries

            if hasattr(interconnect, "search_insights"):
                results = interconnect.search_insights(query, top_k=5)

                for insight_id, similarity in results:
                    if similarity >= min_relevance:
                        bridge = self._get_insight_bridge()
                        if bridge is None:
                            continue
                            
                        content = ""
                        for i in getattr(bridge, 'processed_insights', []):
                            if getattr(i, 'id', None) == insight_id:
                                content = getattr(i, 'raw_insight', '')[:500]
                                break
                        
                        if content:
                            entry = ContextEntry(
                                id=f"insight_rel_{insight_id}",
                                content=content,
                                source=ContextSource.INSIGHT_RELEVANT,
                                priority=ContextPriority.MEDIUM,
                                relevance_score=similarity,
                                importance=0.7,
                                recency=0.8,
                            )
                            entries.append(entry)
        except Exception as e:
            logger.warning(f"Failed to fetch relevant insights: {e}")

        return entries

    async def _fetch_knowledge_graph(
        self,
        query_embedding: np.ndarray,
        min_relevance: float,
    ) -> list[ContextEntry]:
        """Fetch connected concepts from knowledge graph."""
        entries = []
        try:
            kg = self._get_knowledge_graph()
            if kg is None:
                return entries
            
            important = kg.get_important_insights(limit=10) if hasattr(kg, 'get_important_insights') else []

            contents = [n.get("content", "") for n in important if isinstance(n, dict)]
            embeddings = []
            valid_nodes = []

            for i, node in enumerate(important):
                content = contents[i] if i < len(contents) else ""
                if len(content) > 10:
                    emb = self._encode(content)
                    if emb is not None:
                        embeddings.append(emb)
                        valid_nodes.append((node, content))

            if embeddings:
                similarities = self._compute_batch_similarity(query_embedding, embeddings)

                for i, (node, content) in enumerate(valid_nodes):
                    pagerank = node.get("pagerank", 0)
                    similarity = similarities[i]

                    if similarity >= min_relevance * 0.5 or pagerank > 0.1:
                        entry = ContextEntry(
                            id=f"kg_{node.get('id', '')}",
                            content=f"[discovered] {content[:300]}",
                            source=ContextSource.KNOWLEDGE_GRAPH,
                            priority=ContextPriority.LOW,
                            relevance_score=similarity,
                            importance=min(pagerank, 1.0),
                            recency=0.7,
                        )
                        entries.append(entry)
        except Exception as e:
            logger.warning(f"Failed to fetch knowledge graph: {e}")

        return entries[:5]

    async def _fetch_user_facts(self, query_embedding: Optional[np.ndarray]) -> list[ContextEntry]:
        """Fetch persistent user facts."""
        entries = []
        try:
            pm = self._get_persistent_memory()
            if pm is None:
                return entries
            
            facts = pm.get_context_for_prompt() if hasattr(pm, 'get_context_for_prompt') else None

            if facts:
                for line in facts.split("\n")[:3]:
                    if line.strip():
                        entry = ContextEntry(
                            id=f"fact_{hashlib.sha256(line.encode()).hexdigest()[:8]}",
                            content=line[:200],
                            source=ContextSource.USER_FACTS,
                            priority=ContextPriority.HIGH,
                            relevance_score=0.9,
                            importance=0.9,
                            recency=1.0,
                        )
                        entries.append(entry)
        except Exception as e:
            logger.warning(f"Failed to fetch user facts: {e}")

        return entries

    async def _fetch_cross_model_context(
        self,
        query: str,
        query_embedding: np.ndarray,
        model: str,
    ) -> list[ContextEntry]:
        """Fetch relevant context from the other model."""
        entries = []
        try:
            interconnect = self._get_embedding_interconnect()
            if interconnect is None:
                return entries
            
            other_model = "backend" if model == "primary" else "primary"

            if hasattr(interconnect, "get_cross_model_context"):
                context = interconnect.get_cross_model_context(query, model)

                if context and len(context) > 10:
                    entry = ContextEntry(
                        id=f"cross_model_{other_model}",
                        content=f"[{other_model.capitalize()} Context] {context[:300]}",
                        source=ContextSource.CROSS_MODEL,
                        priority=ContextPriority.MEDIUM,
                        relevance_score=0.7,
                        importance=0.7,
                        recency=0.8,
                        model=other_model,
                    )
                    entries.append(entry)
        except Exception as e:
            logger.warning(f"Failed to fetch cross-model context: {e}")

        return entries

    async def _fetch_working_memory(self) -> list[ContextEntry]:
        """Fetch working memory (active task context)."""
        entries = []
        try:
            um = self._get_unified_memory()
            if um is None:
                return entries

            # Use search to get recent working memories
            working_entries = []
            if hasattr(um, 'search'):
                from .unified_memory import MemoryType
                working_entries = um.search(
                    query="",
                    memory_types=[MemoryType.WORKING],
                    limit=5,
                )
            elif hasattr(um, '_by_type'):
                # Fallback: get from internal index
                with um._lock:
                    working_entries = list(um._by_type.get(MemoryType.WORKING, []))[:5]
            
            for item in working_entries:
                content = item.content if hasattr(item, 'content') else (item.get("content", "") if isinstance(item, dict) else "")
                if content:
                    entry = ContextEntry(
                        id=f"working_{item.id if hasattr(item, 'id') else hashlib.sha256(content.encode()).hexdigest()[:8]}",
                        content=content[:300],
                        source=ContextSource.WORKING_MEMORY,
                        priority=ContextPriority.HIGH,
                        relevance_score=0.85,
                        importance=item.importance if hasattr(item, 'importance') else (item.get("importance", 0.7) if isinstance(item, dict) else 0.7),
                        recency=0.95,
                    )
                    entries.append(entry)
        except Exception as e:
            logger.warning(f"Failed to fetch working memory: {e}")

        return entries[:3]

    async def _fetch_subsystem_status(self, query: str = "") -> list[ContextEntry]:
        """Fetch live status of active subsystems (music, erdos, daydream, etc.).
        
        Args:
            query: Optional query to detect subsystem-relevant topics
            
        Returns:
            List of ContextEntry for active subsystem states
        """
        entries = []
        
        # Keywords that indicate subsystem status queries - boost relevance when matched
        subsystem_keywords = {
            'music': ['music', 'composition', 'compose', 'melody', 'harmony', 'playing', 
                     'listening', 'song', 'piece', 'creative'],
            'lcm': ['lcm', 'prefrontal', 'cortex', 'concept', 'cascade', 'pfc', 
                   'learning', 'model', 'neural'],
            'erdos': ['erdos', 'math', 'problem', 'theorem', 'proof', 'conjecture'],
            'daydream': ['daydream', 'imagining', 'exploring', 'idle', 'background'],
        }
        
        # Calculate relevance boost based on query
        query_lower = query.lower() if query else ""
        for subsystem, keywords in subsystem_keywords.items():
            if any(kw in query_lower for kw in keywords):
                break
        
        try:
            # Music Creation Engine status
            try:
                from core.music_creation_engine import get_music_creation_engine
                engine = get_music_creation_engine()
                if engine:
                    stats = engine.get_stats()
                    compositions_count = len(engine.compositions) if hasattr(engine, 'compositions') else 0
                    
                    # Get latest composition info
                    latest = None
                    if compositions_count > 0 and hasattr(engine, 'compositions'):
                        try:
                            last_comp = engine.compositions[-1]
                            latest = {
                                'title': getattr(last_comp, 'title', 'Unknown'),
                                'style': getattr(last_comp, 'style', {}).value if hasattr(getattr(last_comp, 'style', None), 'value', None) else 'classical',
                                'form': getattr(last_comp, 'form', {}).value if hasattr(getattr(last_comp, 'form', None), 'value', None) else 'ternary',
                                'created_at': str(getattr(last_comp, 'created_at', 'recently')),
                            }
                        except Exception as e:
                            logger.debug(f"Music subsystem status failed: {e}")
                    
                    status_parts = [
                        f"[Music Creation] Active: {stats.get('running', False)}",
                        f"Compositions: {stats.get('compositions_created', 0)} created",
                        f"Style: {stats.get('current_style', 'classical')}",
                        f"Tokens observed: {stats.get('tokens_observed', 0)}",
                    ]
                    if latest:
                        status_parts.append(f"Latest: {latest['title']} ({latest['style']} {latest['form']})")
                    
                    entry = ContextEntry(
                        id="subsystem_music",
                        content=" | ".join(status_parts),
                        source=ContextSource.SUBSYSTEM_STATUS,
                        priority=ContextPriority.HIGH,
                        relevance_score=1.0,
                        importance=0.9,
                        recency=1.0,
                        metadata={"subsystem": "music_creation", "stats": stats},
                    )
                    entries.append(entry)
            except Exception as e:
                logger.debug(f"Music status fetch failed: {e}")
            
            # Erdos Engine status
            try:
                from core.erdos_engine import get_erdos_engine
                erdos = get_erdos_engine()
                if erdos:
                    stats = erdos.get_status() if hasattr(erdos, 'get_status') else {}
                    proven = stats.get('problems_proven', 0)
                    attempted = stats.get('problems_attempted', 0)
                    
                    entry = ContextEntry(
                        id="subsystem_erdos",
                        content=f"[Erdos Engine] {attempted} problems attempted, {proven} proven",
                        source=ContextSource.SUBSYSTEM_STATUS,
                        priority=ContextPriority.MEDIUM,
                        relevance_score=0.85,
                        importance=0.7,
                        recency=0.9,
                        metadata={"subsystem": "erdos_engine", "proven": proven, "attempted": attempted},
                    )
                    entries.append(entry)
            except Exception as e:
                logger.debug(f"Daydream subsystem status failed: {e}")
            
            # Daydream Engine status
            try:
                from core.daydream_engine import get_daydream_engine
                daydream = get_daydream_engine()
                if daydream:
                    status = daydream.get_status() if hasattr(daydream, 'get_status') else {}
                    running = status.get('running', False)
                    topics = status.get('exploration_topics', [])
                    current = topics[-1] if topics else None
                    
                    status_text = f"[Daydream] Running: {running}"
                    if current:
                        status_text += f" | Exploring: {current[:50]}"
                    
                    entry = ContextEntry(
                        id="subsystem_daydream",
                        content=status_text,
                        source=ContextSource.SUBSYSTEM_STATUS,
                        priority=ContextPriority.MEDIUM,
                        relevance_score=0.8,
                        importance=0.6,
                        recency=0.95,
                        metadata={"subsystem": "daydream_engine"},
                    )
                    entries.append(entry)
            except Exception as e:
                logger.debug(f"Daydream integration status failed: {e}")
            
            # LCM/Prefrontal Cortex status
            try:
                from core.prefrontal_cortex import get_prefrontal_cortex
                pfc = get_prefrontal_cortex()
                if pfc:
                    concepts = len(pfc._concepts) if hasattr(pfc, '_concepts') else 0
                    training_steps = pfc.state.training_steps if hasattr(pfc, 'state') else 0
                    
                    entry = ContextEntry(
                        id="subsystem_pfc",
                        content=f"[LCM/PFC] {concepts} concepts, {training_steps} training steps",
                        source=ContextSource.SUBSYSTEM_STATUS,
                        priority=ContextPriority.HIGH,
                        relevance_score=1.0,
                        importance=0.95,
                        recency=1.0,
                        metadata={"subsystem": "prefrontal_cortex", "concepts": concepts},
                    )
                    entries.append(entry)
            except Exception as e:
                logger.debug(f"PFC subsystem status failed: {e}")
            
        except Exception as e:
            logger.warning(f"Failed to fetch subsystem status: {e}")
        
        return entries

    async def search_all_knowledge(
        self,
        query: str,
        sources: Optional[list[str]] = None,
        top_k: int = 10,
        min_similarity: float = 0.5,
    ) -> list[SearchResult]:
        """Search across ALL knowledge sources semantically."""
        self._initialize()
        with self._lock:
            self._stats["searches"] += 1
        
        query_embedding = self._encode(query)
        if query_embedding is None:
            return []

        all_sources = sources or ["insights", "memory", "knowledge_graph", "model_outputs"]
        results: list[SearchResult] = []

        # Search in parallel
        search_tasks = []

        if "insights" in all_sources:
            search_tasks.append(self._search_insights(query_embedding, min_similarity))
        if "memory" in all_sources:
            search_tasks.append(self._search_memory(query, query_embedding, min_similarity))
        if "knowledge_graph" in all_sources:
            search_tasks.append(self._search_knowledge_graph(query_embedding, min_similarity))
        if "model_outputs" in all_sources:
            search_tasks.append(self._search_model_outputs(query_embedding, min_similarity))

        if search_tasks:
            search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
            for result in search_results:
                if isinstance(result, list):
                    results.extend(result)

        # Sort and return top_k
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    async def _search_insights(self, query_embedding: np.ndarray, min_similarity: float) -> list[SearchResult]:
        """Search insights."""
        results = []
        try:
            bridge = self._get_insight_bridge()
            if bridge is None:
                return results
            
            candidates = []
            embeddings = []
            
            for insight in getattr(bridge, 'processed_insights', [])[-20:]:
                content = getattr(insight, 'raw_insight', '')
                emb = self._encode(content)
                if emb is not None:
                    candidates.append((insight, content))
                    embeddings.append(emb)
            
            if embeddings:
                similarities = self._compute_batch_similarity(query_embedding, embeddings)
                
                for i, (insight, content) in enumerate(candidates):
                    sim = similarities[i]
                    if sim >= min_similarity:
                        results.append(SearchResult(
                            id=f"insight_{getattr(insight, 'id', i)}",
                            content=content[:500],
                            source="insights",
                            score=sim,
                            metadata={
                                "category": getattr(insight, 'source_category', 'unknown'),
                                "relevance": getattr(getattr(insight, 'evolution_relevance', None), 'value', 0),
                            },
                        ))
        except Exception as e:
            logger.warning(f"Failed to search insights: {e}")
        
        return results

    async def _search_memory(self, query: str, query_embedding: np.ndarray, min_similarity: float) -> list[SearchResult]:
        """Search unified memory."""
        results = []
        try:
            um = self._get_unified_memory()
            if um is None:
                return results
            
            if hasattr(um, "search_semantic"):
                memory_results = um.search_semantic(query, top_k=10) or []
                for item in memory_results:
                    if not isinstance(item, dict):
                        continue
                    results.append(SearchResult(
                        id=f"memory_{item.get('id', '')}",
                        content=item.get("content", "")[:500],
                        source="memory",
                        score=item.get("score", 0.5),
                        metadata=item.get("metadata", {}),
                    ))
        except Exception as e:
            logger.warning(f"Failed to search memory: {e}")
        
        return results

    async def _search_knowledge_graph(self, query_embedding: np.ndarray, min_similarity: float) -> list[SearchResult]:
        """Search knowledge graph."""
        results = []
        try:
            kg = self._get_knowledge_graph()
            if kg is None:
                return results
            
            if hasattr(kg, "search_nodes"):
                kg_results = kg.search_nodes("", top_k=10) or []
                for node in kg_results:
                    if not isinstance(node, dict):
                        continue
                    results.append(SearchResult(
                        id=f"kg_{node.get('id', '')}",
                        content=node.get("content", "")[:500],
                        source="knowledge_graph",
                        score=node.get("score", 0.5),
                        metadata={"pagerank": node.get("pagerank", 0)},
                    ))
        except Exception as e:
            logger.warning(f"Failed to search knowledge graph: {e}")
        
        return results

    async def _search_model_outputs(self, query_embedding: np.ndarray, min_similarity: float) -> list[SearchResult]:
        """Search cross-model outputs."""
        results = []
        try:
            interconnect = self._get_embedding_interconnect()
            if interconnect is None:
                return results

            candidates = []
            embeddings = []

            for model in ["primary", "backend"]:
                model_outputs = interconnect.get_model_outputs(model) if hasattr(interconnect, 'get_model_outputs') else []
                for prompt, quality, stored_emb in model_outputs[-20:]:
                    candidates.append((model, prompt, quality))
                    embeddings.append(stored_emb)

            if embeddings:
                similarities = self._compute_batch_similarity(query_embedding, embeddings)

                for i, (model, prompt, quality) in enumerate(candidates):
                    sim = similarities[i]
                    if sim >= min_similarity:
                        results.append(SearchResult(
                            id=f"model_{model}_{hashlib.sha256(prompt.encode()).hexdigest()[:8]}",
                            content=f"[{model}] {prompt[:200]}",
                            source="model_outputs",
                            score=sim * quality,
                            metadata={"model": model, "quality": quality},
                        ))
        except Exception as e:
            logger.warning(f"Failed to search model outputs: {e}")

        return results

    def store_knowledge(
        self,
        content: str,
        source: str,
        memory_type: str = "semantic",
        importance: float = 0.5,
        model: str = "primary",
        connections: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """Store knowledge in the appropriate subsystem."""
        self._initialize()
        with self._lock:
            self._stats["stores"] += 1
        
        # Validate input
        if not content or not content.strip():
            logger.warning("Empty content provided to store_knowledge")
            return ""

        importance = max(0.0, min(1.0, importance))

        try:
            memory_type_enum = MemoryType(memory_type)
        except ValueError:
            memory_type_enum = MemoryType.SEMANTIC

        entry_id = f"{memory_type}_{hashlib.sha256(content.encode()).hexdigest()[:12]}"
        self._encode(content)

        try:
            um = self._get_unified_memory()
            if um is None:
                return entry_id

            um.store(
                content=content,
                memory_type=memory_type_enum,
                importance=importance,
                metadata={
                    "source": source,
                    "model": model,
                    **(metadata or {}),
                },
            )

            # Clear embedding cache for related queries
            with self._lock:
                self._embedding_cache.clear()

        except Exception as e:
            logger.warning(f"Failed to store knowledge: {e}")

        return entry_id

    def get_stats(self) -> dict:
        """Get context engine statistics."""
        return {
            **self._stats,
            "cache_size": len(self._context_cache),
            "embedding_cache_size": len(self._embedding_cache),
            "last_assembly_time_ms": self._last_assembly_time,
        }

    def clear_cache(self):
        """Clear all caches."""
        with self._lock:
            self._context_cache.clear()
            self._embedding_cache.clear()

    def save_session(self) -> bool:
        """Save session state for continuity."""
        try:
            session_file = self.storage_path / "session.json"
            data = {
                "stats": self._stats,
                "timestamp": datetime.now().isoformat(),
            }
            atomic_write_json(session_file, data)
            return True
        except Exception as e:
            logger.warning(f"Failed to save session: {e}")
            return False

    def load_session(self) -> bool:
        """Load session state for continuity."""
        try:
            session_file = self.storage_path / "session.json"
            if session_file.exists():
                data = read_json_with_integrity(session_file)
                self._stats.update(data.get("stats", {}))
                return True
        except Exception as e:
            logger.debug(f"Failed to load session: {e}")
        return False


def get_context_engine() -> UnifiedContextEngine:
    """Get singleton context engine instance."""
    global _unified_context
    if _unified_context is None:
        with _context_lock:
            if _unified_context is None:
                _unified_context = UnifiedContextEngine()
                _unified_context.load_session()
    return _unified_context