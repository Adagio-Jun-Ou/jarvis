"""Unified Memory System for Jarvis.

Consolidates all memory subsystems into a single coherent interface:
- Conversation memory (short-term context)
- Semantic memory (vector embeddings)
- Persistent memory (long-term storage)
- Episodic memory (event sequences)
- Working memory (active state)
- Insight memory (learned patterns, hash-searchable)

Provides a single source of truth for all memory operations.
"""
import asyncio
import json
import logging
import threading
import time
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)

# Singleton
_unified_memory: Optional["UnifiedMemory"] = None
_memory_lock = threading.Lock()


class MemoryType(Enum):
    CONVERSATION = "conversation"    # Short-term chat context
    SEMANTIC = "semantic"           # Vector-embedded knowledge
    PERSISTENT = "persistent"       # Long-term facts
    EPISODIC = "episodic"           # Event sequences
    WORKING = "working"             # Active task state
    INSIGHT = "insight"             # Learned patterns/insights


@dataclass
class MemoryEntry:
    """A single memory entry."""
    id: str
    content: str
    memory_type: MemoryType
    timestamp: datetime = field(default_factory=datetime.now)
    importance: float = 0.5          # 0-1, affects retention
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)
    embedding: Optional[list[float]] = None
    connections: list[str] = field(default_factory=list)  # IDs of related memories

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "timestamp": self.timestamp.isoformat(),
            "importance": self.importance,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat(),
            "metadata": self.metadata,
            "connections": self.connections,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryEntry":
        return cls(
            id=data["id"],
            content=data["content"],
            memory_type=MemoryType(data["memory_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            importance=data.get("importance", 0.5),
            access_count=data.get("access_count", 0),
            last_accessed=datetime.fromisoformat(data["last_accessed"]) if data.get("last_accessed") else datetime.now(),
            metadata=data.get("metadata", {}),
            connections=data.get("connections", []),
        )


@dataclass
class MemoryStats:
    """Statistics about memory usage."""
    total_entries: int = 0
    by_type: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    total_size_bytes: int = 0
    oldest_entry: Optional[datetime] = None
    newest_entry: Optional[datetime] = None
    avg_importance: float = 0.0
    total_connections: int = 0


class UnifiedMemory:
    """Unified memory system consolidating all memory types.
    
    Architecture:
    - Working Memory: Active task context, cleared between tasks
    - Conversation Memory: Recent chat history, limited size
    - Semantic Memory: Vector-embedded knowledge for retrieval
    - Persistent Memory: Long-term facts and user preferences
    - Episodic Memory: Event sequences and experiences
    - Insight Memory: Learned patterns and discoveries
    
    Features:
    - Automatic memory consolidation (working → persistent)
    - Importance-based retention
    - Cross-memory connections
    - Unified search across all types
    - Memory decay and cleanup
    - Uses shared EmbeddingInterconnect for semantic search
    """

    MAX_CONVERSATION = 100
    MAX_WORKING = 50
    MAX_EPISODIC = 500
    MAX_INSIGHT = 1000
    CONSOLIDATION_THRESHOLD = 0.7  # Importance threshold for long-term storage

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path.home() / ".jarvis" / "unified_memory"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Memory stores by type
        self._memories: dict[str, MemoryEntry] = {}
        self._by_type: dict[MemoryType, list[str]] = defaultdict(list)
        self._index: dict[str, list[str]] = defaultdict(list)  # Word -> memory IDs
        self._MAX_INDEX_WORDS = 10000  # Bound on word index size

        # Working memory (cleared frequently)
        self._working: dict[str, Any] = {}

        # Connection graph for memory associations
        self._connections: dict[str, set[str]] = defaultdict(set)

        # Stats
        self._stats = MemoryStats()

        # Lock for thread safety
        self._lock = threading.RLock()
        
        # Shared embedding interconnect
        self._interconnect = None

        # Load existing memories
        self._load()
        
        # Try to get interconnect
        try:
            from core.embedding_interconnect import get_embedding_interconnect
            self._interconnect = get_embedding_interconnect()
        except Exception as e:
            logger.debug(f"Embedding interconnect unavailable: {e}")

        logger.info(f"UnifiedMemory initialized: {len(self._memories)} entries")

    def _index_entry(self, mem_id: str, content: str) -> None:
        """Add a memory's words to the keyword index."""
        for word in set((content or "").lower().split()):
            if len(word) > 2:  # Skip short words
                self._index.setdefault(word, []).append(mem_id)

    def _load(self):
        """Load memories from storage."""
        state_file = self.storage_path / "unified_state.json"
        if state_file.exists():
            try:
                data = json.loads(state_file.read_text(encoding='utf-8'))
                for mem_data in data.get("memories", []):
                    try:
                        entry = MemoryEntry.from_dict(mem_data)
                        self._memories[entry.id] = entry
                        self._by_type[entry.memory_type].append(entry.id)
                        self._connections[entry.id].update(entry.connections)
                        # The word index is not persisted and was only ever
                        # built inside store(), so after a restart it was empty
                        # and search() — which draws its entire candidate set
                        # from it — returned [] for everything.
                        self._index_entry(entry.id, entry.content)
                    except Exception as e:
                        logger.debug(f"Failed to load memory: {e}")
                self._enforce_index_bounds()
                logger.info(f"Loaded {len(self._memories)} memories from storage")
            except Exception as e:
                logger.error(f"Failed to load unified memory: {e}")

    def _save(self):
        """Save memories to storage (snapshot under lock, I/O outside)."""
        state_file = self.storage_path / "unified_state.json"
        try:
            # Self-heal the storage directory: if it was removed after
            # init (e.g. pytest tmp-dir retention pruning during a long
            # embedding-model load), write_text would fail with ENOENT.
            self.storage_path.mkdir(parents=True, exist_ok=True)
            with self._lock:
                data = {
                    "memories": [m.to_dict() for m in self._memories.values()],
                    "stats": {
                        "total": len(self._memories),
                        "by_type": dict(self._stats.by_type),
                    },
                    "saved_at": datetime.now().isoformat(),
                }
            state_file.write_text(json.dumps(data, indent=2, default=str), encoding='utf-8')
        except Exception as e:
            logger.error(f"Failed to save unified memory: {e}")

    def store(
        self,
        content: str,
        memory_type: MemoryType | str,
        importance: float = 0.5,
        metadata: Optional[dict] = None,
        connections: Optional[list[str]] = None,
    ) -> str:
        """Store a new memory entry."""
        if isinstance(memory_type, str):
            try:
                memory_type = MemoryType(memory_type)
            except ValueError:
                memory_type = MemoryType.WORKING

        with self._lock:
            # Generate ID
            mem_id = f"{memory_type.value}_{int(time.time() * 1000)}_{len(self._memories)}"

            # Create entry
            entry = MemoryEntry(
                id=mem_id,
                content=content,
                memory_type=memory_type,
                importance=max(0.0, min(1.0, importance)),
                metadata=metadata or {},
                connections=connections or [],
            )

            # Store
            self._memories[mem_id] = entry
            self._by_type[memory_type].append(mem_id)

            # Update connections
            for conn_id in (connections or []):
                self._connections[mem_id].add(conn_id)
                self._connections[conn_id].add(mem_id)

            # Build index
            self._index_entry(mem_id, content)

            # Enforce index bounds
            self._enforce_index_bounds()

            # Store in shared embedding interconnect
            if self._interconnect:
                try:
                    self._interconnect.store_memory(mem_id, content)
                except Exception as e:
                    logger.debug(f"Failed to store memory embedding: {e}")

            # Check capacity
            self._enforce_capacity(memory_type)

            # Auto-consolidate important memories
            if importance >= self.CONSOLIDATION_THRESHOLD and memory_type == MemoryType.WORKING:
                self._consolidate(mem_id)

            # Update stats
            self._stats.total_entries = len(self._memories)
            self._stats.by_type[memory_type.value] = len(self._by_type[memory_type])

            return mem_id

    def retrieve(self, mem_id: str) -> Optional[MemoryEntry]:
        """Retrieve a memory by ID."""
        with self._lock:
            entry = self._memories.get(mem_id)
            if entry:
                entry.access_count += 1
                entry.last_accessed = datetime.now()
            return entry

    def search(
        self,
        query: str,
        memory_types: Optional[list[MemoryType]] = None,
        limit: int = 10,
    ) -> list[MemoryEntry]:
        """Search memories by content.
        
        Args:
            query: Search query
            memory_types: Filter by memory types (None = all)
            limit: Maximum results
            
        Returns:
            List of matching memories, sorted by relevance
        """
        with self._lock:
            query_words = set(query.lower().split())

            # Find candidate memories by word index
            candidates = defaultdict(int)
            for word in query_words:
                if len(word) > 2:
                    for mem_id in self._index.get(word, []):
                        candidates[mem_id] += 1

            # Score and filter
            results = []
            for mem_id, score in candidates.items():
                entry = self._memories.get(mem_id)
                if entry:
                    # Filter by type
                    if memory_types and entry.memory_type not in memory_types:
                        continue

                    # Score = word matches * importance * recency
                    recency = (datetime.now() - entry.timestamp).total_seconds() / 86400  # Days
                    recency_factor = 1.0 / (1.0 + recency * 0.1)
                    final_score = score * entry.importance * recency_factor

                    results.append((final_score, entry))

            # Sort by score and return top results
            results.sort(key=lambda x: x[0], reverse=True)
            return [entry for _, entry in results[:limit]]

    def search_semantic(
        self,
        query: str,
        limit: int = 10,
    ) -> list[MemoryEntry]:
        """Semantic search using shared EmbeddingInterconnect."""
        # Try to use shared embedding interconnect first
        if self._interconnect:
            try:
                results = self._interconnect.search_memories(query, top_k=limit)
                with self._lock:
                    entries = []
                    for mem_id, score in results:
                        if mem_id in self._memories:
                            entries.append(self._memories[mem_id])
                if entries:
                    return entries
            except Exception as e:
                logger.debug(f"Interconnect semantic search failed: {e}")
            
            # Fallback to vector memory
            try:
                from .vector_memory import get_vector_memory
                vm = get_vector_memory()
                results = vm.search(query, k=limit)

                # Convert to MemoryEntry objects. search() yields
                # (entry_id, score, metadata) — the first element was being
                # bound to `text` and stored as the memory content, so every
                # result's content was an opaque id string while the real text
                # sat unused in metadata['text'].
                entries = []
                for entry_id, score, metadata in results:
                    metadata = metadata or {}
                    entries.append(MemoryEntry(
                        id=metadata.get("id", entry_id),
                        content=metadata.get("content") or metadata.get("text", ""),
                        memory_type=MemoryType.SEMANTIC,
                        importance=score,
                        metadata=metadata,
                    ))
                return entries
            except Exception as e:
                logger.debug(f"Vector search failed, falling back to keyword: {e}")
                return self.search(query, [MemoryType.SEMANTIC], limit)

    def get_recent(self, memory_type: MemoryType, limit: int = 10) -> list[MemoryEntry]:
        """Get recent memories of a specific type."""
        with self._lock:
            ids = self._by_type.get(memory_type, [])
            entries = [self._memories[mid] for mid in ids[-limit:] if mid in self._memories]
            return list(reversed(entries))

    def get_connected(self, mem_id: str, depth: int = 2) -> list[MemoryEntry]:
        """Get memories connected to a given memory."""
        with self._lock:
            visited = set()
            result = []

            def traverse(current_id: str, current_depth: int):
                if current_depth > depth or current_id in visited:
                    return
                visited.add(current_id)

                if current_id != mem_id:  # Don't include the starting memory
                    entry = self._memories.get(current_id)
                    if entry:
                        result.append(entry)

                for conn_id in self._connections.get(current_id, []):
                    traverse(conn_id, current_depth + 1)

            traverse(mem_id, 0)
            return result

    def consolidate(self, mem_id: str, target_type: MemoryType = MemoryType.PERSISTENT):
        """Promote a memory to a more permanent type."""
        with self._lock:
            self._consolidate(mem_id, target_type)

    def _consolidate(self, mem_id: str, target_type: MemoryType = MemoryType.PERSISTENT):
        """Internal consolidation."""
        entry = self._memories.get(mem_id)
        if not entry:
            return

        # Remove from old type list
        if entry.memory_type in self._by_type:
            try:
                self._by_type[entry.memory_type].remove(mem_id)
            except ValueError:
                pass

        # Update type
        old_type = entry.memory_type
        entry.memory_type = target_type
        self._by_type[target_type].append(mem_id)

        logger.debug(f"Consolidated memory {mem_id}: {old_type.value} → {target_type.value}")

    def consolidate_all(self) -> dict:
        """Batch consolidate high-importance memories to persistent storage.
        
        Returns:
            Statistics about consolidation performed.
        """
        with self._lock:
            stats = {
                "consolidated": 0,
                "working_to_persistent": 0,
                "conversation_to_episodic": 0,
                "total_before": len(self._memories),
            }
            
            for mem_id, entry in list(self._memories.items()):
                if entry.importance >= self.CONSOLIDATION_THRESHOLD:
                    if entry.memory_type == MemoryType.WORKING:
                        self._consolidate(mem_id, MemoryType.PERSISTENT)
                        stats["working_to_persistent"] += 1
                        stats["consolidated"] += 1
                    elif entry.memory_type == MemoryType.CONVERSATION:
                        self._consolidate(mem_id, MemoryType.EPISODIC)
                        stats["conversation_to_episodic"] += 1
                        stats["consolidated"] += 1
            
            stats["total_after"] = len(self._memories)
            
        self._save()
        return stats

    def _enforce_capacity(self, memory_type: MemoryType):
        """Enforce capacity limits for memory types."""
        max_sizes = {
            MemoryType.CONVERSATION: self.MAX_CONVERSATION,
            MemoryType.WORKING: self.MAX_WORKING,
            MemoryType.EPISODIC: self.MAX_EPISODIC,
            MemoryType.INSIGHT: self.MAX_INSIGHT,
        }

        max_size = max_sizes.get(memory_type)
        if max_size is None:
            return

        ids = self._by_type.get(memory_type, [])
        if len(ids) > max_size:
            # Remove oldest, least important
            to_remove = len(ids) - max_size
            ids_to_remove = ids[:to_remove]

            for mem_id in ids_to_remove:
                self._remove(mem_id)


    def _enforce_index_bounds(self) -> None:
        """Enforce bounds on word index to prevent unbounded growth."""
        if len(self._index) > self._MAX_INDEX_WORDS:
            # Remove least-used words
            word_counts = {w: len(ids) for w, ids in self._index.items()}
            sorted_words = sorted(word_counts.items(), key=lambda x: x[1])
            words_to_remove = len(self._index) - self._MAX_INDEX_WORDS
            for word, _ in sorted_words[:words_to_remove]:
                del self._index[word]

    def _remove(self, mem_id: str):
        """Remove a memory."""
        if mem_id not in self._memories:
            return

        entry = self._memories[mem_id]

        # Remove from type list
        if entry.memory_type in self._by_type:
            try:
                self._by_type[entry.memory_type].remove(mem_id)
            except ValueError:
                pass

        # Remove from connections
        if mem_id in self._connections:
            del self._connections[mem_id]
        for conn_set in self._connections.values():
            conn_set.discard(mem_id)

        # Remove from index
        words = set(entry.content.lower().split())
        for word in words:
            if word in self._index and mem_id in self._index[word]:
                self._index[word].remove(mem_id)
                if not self._index[word]:
                    del self._index[word]

        # Remove entry
        del self._memories[mem_id]

    def decay(self, threshold_days: int = 30, min_importance: float = 0.3):
        """Decay old, unimportant memories."""
        with self._lock:
            now = datetime.now()
            to_remove = []

            for mem_id, entry in self._memories.items():
                age_days = (now - entry.timestamp).total_seconds() / 86400
                decay_rate = 1.0 - (entry.importance * 0.5)  # Higher importance = slower decay
                effective_age = age_days * decay_rate

                if effective_age > threshold_days and entry.importance < min_importance:
                    to_remove.append(mem_id)

            for mem_id in to_remove:
                self._remove(mem_id)

            if to_remove:
                logger.info(f"Decayed {len(to_remove)} old memories")

            return len(to_remove)

    def set_working(self, key: str, value: Any):
        """Set a working memory value (cleared between tasks)."""
        with self._lock:
            self._working[key] = {
                "value": value,
                "timestamp": datetime.now().isoformat(),
            }

    def get_working(self, key: str, default: Any = None) -> Any:
        """Get a working memory value."""
        with self._lock:
            entry = self._working.get(key)
            return entry["value"] if entry else default

    def clear_working(self):
        """Clear all working memory."""
        with self._lock:
            self._working.clear()

    def get_stats(self) -> dict:
        """Get memory statistics."""
        with self._lock:
            self._stats.total_entries = len(self._memories)
            self._stats.by_type = {t.value: len(ids) for t, ids in self._by_type.items()}
            self._stats.total_connections = sum(len(conns) for conns in self._connections.values()) // 2

            if self._memories:
                importances = [e.importance for e in self._memories.values()]
                self._stats.avg_importance = sum(importances) / len(importances)

                timestamps = [e.timestamp for e in self._memories.values()]
                self._stats.oldest_entry = min(timestamps)
                self._stats.newest_entry = max(timestamps)

            return {
                "total_entries": self._stats.total_entries,
                "by_type": dict(self._stats.by_type),
                "total_connections": self._stats.total_connections,
                "avg_importance": self._stats.avg_importance,
                "oldest_entry": self._stats.oldest_entry.isoformat() if self._stats.oldest_entry else None,
                "newest_entry": self._stats.newest_entry.isoformat() if self._stats.newest_entry else None,
            }

    def get_context_for_prompt(self, max_entries: int = 10) -> str:
        """Get relevant context for LLM prompt."""
        with self._lock:
            # Get recent insights and high-importance memories
            insights = self.get_recent(MemoryType.INSIGHT, 5)
            persistent = sorted(
                [self._memories[mid] for mid in self._by_type.get(MemoryType.PERSISTENT, []) if mid in self._memories],
                key=lambda e: e.importance,
                reverse=True
            )[:5]

            context_parts = []

            if insights:
                context_parts.append("Recent insights:")
                for entry in insights:
                    context_parts.append(f"  - {entry.content[:200]}")

            if persistent:
                context_parts.append("\nImportant facts:")
                for entry in persistent:
                    context_parts.append(f"  - {entry.content[:200]}")

            return "\n".join(context_parts) if context_parts else ""

    def save(self):
        """Save memory state."""
        self._save()

    async def async_save(self):
        """Async save."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.save)

    def store_insight(
        self,
        content: str,
        source: str = "unified",
        insight_type: str = "pattern",
        importance: float = 0.7,
        metadata: Optional[dict] = None,
    ) -> str:
        """Store an insight and index it for hash-based search.
        
        Args:
            content: The insight content
            source: Source of the insight (e.g., 'insight_bridge', 'erdos', 'daydream')
            insight_type: Type of insight
            importance: Importance score (0-1)
            metadata: Additional metadata
            
        Returns:
            Memory ID
        """
        # Generate content hash for deduplication
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        
        # Check if already stored
        with self._lock:
            existing = self._find_by_hash(content_hash)
            if existing:
                # Update importance if higher
                if importance > existing.importance:
                    existing.importance = importance
                    existing.access_count += 1
                    existing.last_accessed = datetime.now()
                return existing.id
        
        # Store with INSIGHT type
        metadata = metadata or {}
        metadata["source"] = source
        metadata["insight_type"] = insight_type
        metadata["content_hash"] = content_hash
        
        mem_id = self.store(
            content=content,
            memory_type=MemoryType.INSIGHT,
            importance=importance,
            metadata=metadata,
        )
        
        # Also index in insight_review for hash-based search
        try:
            from core.insight_review import get_insight_review
            review = get_insight_review()
            review.index_insight(
                insight_id=mem_id,
                content=content,
                source=source,
                insight_type=insight_type,
                timestamp=datetime.now().isoformat(),
                category=metadata.get("category", ""),
                problem=metadata.get("problem", ""),
            )
        except Exception as e:
            logger.debug(f"Failed to index insight in review: {e}")
        
        return mem_id
    
    def _find_by_hash(self, content_hash: str) -> Optional["MemoryEntry"]:
        """Find a memory by content hash."""
        for entry in self._memories.values():
            if entry.memory_type == MemoryType.INSIGHT:
                if entry.metadata.get("content_hash") == content_hash:
                    return entry
        return None
    
    def search_insights(
        self,
        query: str = "",
        source: Optional[str] = None,
        insight_type: Optional[str] = None,
        limit: int = 20,
    ) -> list[MemoryEntry]:
        """Search insights with optional filtering.
        
        Args:
            query: Search query (empty = all)
            source: Filter by source
            insight_type: Filter by type
            limit: Maximum results
            
        Returns:
            List of matching insights
        """
        with self._lock:
            results = []
            
            for mem_id in self._by_type.get(MemoryType.INSIGHT, []):
                entry = self._memories.get(mem_id)
                if not entry:
                    continue
                
                # Filter by source
                if source and entry.metadata.get("source") != source:
                    continue
                
                # Filter by type
                if insight_type and entry.metadata.get("insight_type") != insight_type:
                    continue
                
                # Filter by query
                if query:
                    query_lower = query.lower()
                    if query_lower not in entry.content.lower():
                        continue
                
                results.append(entry)
            
            # Sort by timestamp descending
            results.sort(key=lambda e: e.timestamp, reverse=True)
            return results[:limit]
    
    def get_insight_by_hash(self, content_hash: str) -> Optional[MemoryEntry]:
        """Get insight by content hash.
        
        Args:
            content_hash: SHA256 hash (first 16 chars)
            
        Returns:
            MemoryEntry or None
        """
        with self._lock:
            return self._find_by_hash(content_hash)
    
    def find_similar_insights(
        self,
        content: str,
        threshold: float = 0.3,
        limit: int = 10,
    ) -> list[tuple[MemoryEntry, float]]:
        """Find similar insights by content overlap.
        
        Args:
            content: Content to match
            threshold: Minimum similarity (0-1)
            limit: Maximum results
            
        Returns:
            List of (entry, similarity) tuples
        """
        try:
            from core.insight_review import get_insight_review
            review = get_insight_review()
            
            similar = review.find_by_content(content, threshold=threshold)[:limit]
            
            with self._lock:
                results = []
                for item in similar:
                    mem_id = item.get("id")
                    if mem_id and mem_id in self._memories:
                        entry = self._memories[mem_id]
                        similarity = item.get("similarity", 0)
                        results.append((entry, similarity))
            
            return results
        except Exception as e:
            logger.debug(f"Failed to use insight_review for similarity: {e}")
            return []
    
    def get_insight_heatmap(self, limit: int = 50) -> list[tuple[str, int]]:
        """Get word frequency heatmap from insights.
        
        Args:
            limit: Maximum words to return
            
        Returns:
            List of (word, count) tuples
        """
        try:
            from core.insight_review import get_insight_review
            review = get_insight_review()
            return review.get_word_heatmap(limit)
        except Exception as e:
            logger.debug(f"Failed to get heatmap from insight_review: {e}")
            return []
    
    def reindex_insights(self) -> int:
        """Reindex all insight_bridge insights.
        
        Returns:
            Number of insights indexed
        """
        try:
            from core.insight_review import get_insight_review
            review = get_insight_review()
            return review.index_from_bridge()
        except Exception as e:
            logger.error(f"Failed to reindex insights: {e}")
            return 0
    
    def get_insight_stats(self) -> dict:
        """Get statistics about insights."""
        with self._lock:
            insight_ids = self._by_type.get(MemoryType.INSIGHT, [])
            insights = [self._memories[mid] for mid in insight_ids if mid in self._memories]
            
            sources = defaultdict(int)
            types = defaultdict(int)
            for entry in insights:
                sources[entry.metadata.get("source", "unknown")] += 1
                types[entry.metadata.get("insight_type", "unknown")] += 1
            
            return {
                "total": len(insights),
                "by_source": dict(sources),
                "by_type": dict(types),
                "avg_importance": sum(e.importance for e in insights) / len(insights) if insights else 0,
            }


def get_unified_memory() -> UnifiedMemory:
    """Get the unified memory singleton."""
    global _unified_memory
    if _unified_memory is None:
        with _memory_lock:
            if _unified_memory is None:
                _unified_memory = UnifiedMemory()
    return _unified_memory
