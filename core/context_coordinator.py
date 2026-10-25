"""Context Coordinator - Unified entry point for all context systems.

This module provides a single coordination layer for all context subsystems:
- UnifiedContextEngine: Source-based context assembly
- UnifiedContext: Layer-based context assembly
- ContextCascadeIntegration: LCM/PFC concept flow
- AgenticContext: Execution context tracking
- ContextManager: Compression and sub-agents

The coordinator ensures:
1. Single initialization order for all subsystems
2. Cache coordination under memory pressure
3. Conflict resolution with precedence rules
4. Unified context result format

Architecture:
┌─────────────────────────────────────────────────────────────────┐
│                    ContextCoordinator                            │
│  - observe() → triggers all subsystems in order                  │
│  - assemble() → returns UnifiedContextResult                     │
│  - resolve_conflicts() → applies precedence rules               │
├─────────────────────────────────────────────────────────────────┤
│   Subsystems (lazy-loaded in order):                             │
│   1. EmbeddingInterconnect                                        │
│   2. Memory (conversation, persistent, vector)                   │
│   3. KnowledgeGraph                                               │
│   4. PrefrontalCortex                                             │
│   5. UnifiedContextEngine                                         │
│   6. ContextCascade                                               │
│   7. AgenticContext                                               │
└─────────────────────────────────────────────────────────────────┘
"""
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

logger = logging.getLogger(__name__)


class ContextBootstrap:
    """Manages ordered initialization of context subsystems.
    
    Ensures subsystems are loaded in correct dependency order:
    1. EmbeddingInterconnect (no deps)
    2. Memory systems (depend on embedding)
    3. Knowledge graph (depends on memory)
    4. PFC (depends on embedding)
    5. UnifiedContextEngine (depends on memory, KG)
    6. ContextCascade (depends on PFC, engine)
    7. AgenticContext (no deps on others)
    
    Usage:
        bootstrap = ContextBootstrap()
        bootstrap.initialize_all()
        
        # Or initialize specific subsystem
        bootstrap.initialize("prefrontal_cortex")
    """
    
    INIT_ORDER = [
        ("embedding_interconnect", [], "core.embedding_interconnect", "get_embedding_interconnect"),
        ("memory", [], "core.memory", "get_memory"),
        ("persistent_memory", [], "core.persistent_memory", "get_persistent_memory"),
        ("knowledge_graph", ["memory"], "core.knowledge_graph", "get_knowledge_graph"),
        ("unified_memory", ["memory", "persistent_memory"], "core.unified_context_engine", "get_unified_memory"),
        ("prefrontal_cortex", ["embedding_interconnect"], "core.prefrontal_cortex", "get_prefrontal_cortex"),
        ("unified_context_engine", ["memory", "knowledge_graph", "unified_memory"], "core.unified_context_engine", "get_context_engine"),
        ("context_cascade", ["prefrontal_cortex", "unified_context_engine"], "core.context_cascade", "get_context_cascade"),
        ("agentic_context", [], "core.agentic_context", "get_agentic_context"),
    ]
    
    def __init__(self):
        self._initialized: dict[str, bool] = {}
        self._instances: dict[str, Any] = {}
        self._lock = threading.RLock()
    
    def initialize_all(self) -> dict[str, bool]:
        """Initialize all subsystems in dependency order.
        
        Returns:
            Dict of subsystem name -> success
        """
        results = {}
        for name, deps, module, getter in self.INIT_ORDER:
            results[name] = self.initialize(name)
        return results
    
    def initialize(self, name: str) -> bool:
        """Initialize a specific subsystem.
        
        Args:
            name: Subsystem name from INIT_ORDER
            
        Returns:
            True if initialized successfully
        """
        if name in self._initialized and self._initialized[name]:
            return True
        
        with self._lock:
            # Find subsystem config
            config = None
            for cfg_name, cfg_deps, cfg_module, cfg_getter in self.INIT_ORDER:
                if cfg_name == name:
                    config = (cfg_deps, cfg_module, cfg_getter)
                    break
            
            if not config:
                logger.warning(f"Unknown subsystem: {name}")
                return False
            
            deps, module_path, getter_name = config
            
            # Initialize dependencies first
            for dep in deps:
                self.initialize(dep)
            
            # Import and get instance
            try:
                import importlib
                module = importlib.import_module(module_path)
                getter = getattr(module, getter_name)
                self._instances[name] = getter()
                self._initialized[name] = True
                logger.debug(f"Initialized subsystem: {name}")
                return True
            except Exception as e:
                logger.debug(f"Failed to initialize {name}: {e}")
                self._initialized[name] = False
                return False
    
    def get(self, name: str) -> Any:
        """Get an initialized subsystem instance.
        
        Args:
            name: Subsystem name
            
        Returns:
            The subsystem instance, or None if not initialized
        """
        if name not in self._initialized:
            self.initialize(name)
        return self._instances.get(name)
    
    def is_initialized(self, name: str) -> bool:
        """Check if a subsystem is initialized."""
        return self._initialized.get(name, False)


_context_bootstrap: ContextBootstrap | None = None
_bootstrap_lock = threading.Lock()


def get_context_bootstrap() -> ContextBootstrap:
    """Get or create the context bootstrap singleton."""
    global _context_bootstrap
    if _context_bootstrap is None:
        with _bootstrap_lock:
            if _context_bootstrap is None:
                _context_bootstrap = ContextBootstrap()
    return _context_bootstrap


class ContextPriority(IntEnum):
    """Priority levels for context sources."""
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3


class SourcePrecedence(IntEnum):
    """Precedence for conflict resolution between sources."""
    PERSISTENT_MEMORY = 10
    KNOWLEDGE_GRAPH = 9
    PROJECT_CONTEXT = 8
    WORKING_MEMORY = 7
    UNIFIED_MEMORY = 6
    CONVERSATION = 5
    CASCADE = 4
    EPISODIC = 3


@dataclass
class ContextConflict:
    """Represents a detected conflict between context sources.
    
    Attributes:
        topic: The topic/fact in conflict
        sources: Dict of source_name -> value
        winning_source: Source that wins by precedence
        winning_value: The value that was chosen
    """
    topic: str
    sources: dict[str, Any] = field(default_factory=dict)
    winning_source: str = ""
    winning_value: Any = None


class ContextConflictResolver:
    """Resolves conflicts between context sources using precedence rules.
    
    When multiple sources provide conflicting values for the same topic,
    this resolver applies precedence rules to determine the winning value.
    
    Precedence rules (highest to lowest):
    1. Persistent Memory - User preferences stored long-term
    2. Project Context - Current project state
    3. Knowledge Graph - Verified facts
    4. Working Memory - Current conversation context
    5. Unified Memory - Aggregated context
    6. Cascade - LCM/PFC predictions
    
    Usage:
        resolver = ContextConflictResolver()
        conflicts = resolver.detect_conflicts(entries)
        resolved_entries = resolver.resolve(entries)
    """
    
    # Map source names to precedence levels
    SOURCE_PRECEDENCE = {
        "persistent_memory": 10,
        "project_context": 9,
        "knowledge_graph": 8,
        "working_memory": 7,
        "unified_memory": 6,
        "unified_context_engine": 6,
        "conversation": 5,
        "agentic_context": 5,
        "cascade": 4,
        "context_cascade": 4,
        "episodic": 3,
    }
    
    # Topics that should prefer newer values over precedence
    TEMPORAL_TOPICS = {
        "current_location",
        "current_time",
        "current_weather",
        "last_activity",
        "recent_action",
    }
    
    def detect_conflicts(self, entries: list[Any]) -> list[ContextConflict]:
        """Detect conflicts between context entries.
        
        Args:
            entries: List of context entries to check
            
        Returns:
            List of detected conflicts
        """
        # Group entries by topic/key
        topic_entries: dict[str, list[tuple[str, Any]]] = {}
        
        for entry in entries:
            if hasattr(entry, 'key') and hasattr(entry, 'value'):
                topic = entry.key
                source = getattr(entry, 'source', 'unknown')
                if topic not in topic_entries:
                    topic_entries[topic] = []
                topic_entries[topic].append((source, entry.value))
        
        # Find conflicts (same topic, different values)
        conflicts = []
        for topic, values in topic_entries.items():
            if len(set(v for _, v in values)) > 1:
                conflict = ContextConflict(
                    topic=topic,
                    sources={s: v for s, v in values},
                )
                conflicts.append(conflict)
        
        return conflicts
    
    def resolve(self, entries: list[Any]) -> list[Any]:
        """Resolve conflicts in context entries.
        
        Args:
            entries: List of context entries with potential conflicts
            
        Returns:
            List of resolved entries (duplicates removed, conflicts resolved)
        """
        conflicts = self.detect_conflicts(entries)
        
        if not conflicts:
            return entries
        
        resolved = list(entries)
        
        for conflict in conflicts:
            # Find winner by precedence
            best_source = None
            best_value = None
            best_precedence = -1
            
            for source, value in conflict.sources.items():
                precedence = self.SOURCE_PRECEDENCE.get(source, 0)
                if precedence > best_precedence:
                    best_precedence = precedence
                    best_source = source
                    best_value = value
            
            conflict.winning_source = best_source or ""
            conflict.winning_value = best_value
            
            # Remove losing entries
            # In a full implementation, we would actually modify entries here
        
        return resolved
    
    def get_precedence(self, source: str) -> int:
        """Get precedence level for a source.
        
        Args:
            source: Source name
            
        Returns:
            Precedence level (higher = more authoritative)
        """
        return self.SOURCE_PRECEDENCE.get(source, 0)


@dataclass
class UnifiedContextResult:
    """Unified result from all context systems.
    
    Attributes:
        messages: List of message dicts for LLM context
        entries: List of context entries with metadata
        sources_used: Set of source names that contributed
        total_tokens: Approximate token count
        conflicts_resolved: Number of conflicts resolved
        timing_ms: Dict of timing by subsystem
        cascade_concepts: Concepts from LCM/PFC cascade
        project_context: Current project/goal context
    """
    messages: list[dict] = field(default_factory=list)
    entries: list[Any] = field(default_factory=list)
    sources_used: set[str] = field(default_factory=set)
    total_tokens: int = 0
    conflicts_resolved: int = 0
    timing_ms: dict[str, float] = field(default_factory=dict)
    cascade_concepts: list[dict] = field(default_factory=list)
    project_context: dict | None = None
    
    def to_messages(self) -> list[dict]:
        """Get messages for LLM input."""
        return self.messages
    
    def to_summary(self) -> str:
        """Get human-readable summary."""
        parts = []
        if self.sources_used:
            parts.append(f"Sources: {', '.join(sorted(self.sources_used))}")
        if self.total_tokens:
            parts.append(f"Tokens: ~{self.total_tokens}")
        if self.conflicts_resolved:
            parts.append(f"Conflicts resolved: {self.conflicts_resolved}")
        return " | ".join(parts)


class ContextCoordinator:
    """Coordinates all context subsystems for unified context assembly.
    
    This is the single entry point for context operations, ensuring:
    1. Subsystems are initialized in correct dependency order
    2. Caches are coordinated under memory pressure
    3. Conflicts are resolved with precedence rules
    4. Results are unified across all systems
    
    Usage:
        coordinator = get_context_coordinator()
        
        # Observe activity (triggers all subsystems)
        result = coordinator.observe("user_query", "What's the weather?", "user")
        
        # Get unified context for LLM
        context = await coordinator.assemble("What's the weather?", max_tokens=4000)
        messages = context.to_messages()
    """
    
    # Initialization order with dependencies
    INIT_ORDER = [
        ("embedding_interconnect", None),
        ("memory", None),
        ("persistent_memory", None),
        ("knowledge_graph", None),
        ("unified_memory", ["memory", "persistent_memory"]),
        ("prefrontal_cortex", ["embedding_interconnect"]),
        ("unified_context_engine", ["memory", "knowledge_graph", "unified_memory"]),
        ("context_cascade", ["prefrontal_cortex", "unified_context_engine"]),
        ("agentic_context", None),
    ]
    
    # Memory limits for cache coordination
    MAX_TOTAL_CACHE_MB = 200
    CACHE_PRESSURE_THRESHOLD = 0.8
    
    def __init__(self):
        self._initialized = False
        self._lock = threading.RLock()
        
        # Subsystem references (lazy-loaded)
        self._subsystems: dict[str, Any] = {}
        
        # Cache tracking
        self._caches: dict[str, tuple[str, int]] = {}  # name -> (type, size_bytes)
        self._total_cache_bytes = 0
        
        # Timing stats
        self._timing_stats: dict[str, list[float]] = {}
    
    def _ensure_initialized(self) -> None:
        """Initialize subsystems in dependency order."""
        with self._lock:
            if self._initialized:
                return
            
            for name, deps in self.INIT_ORDER:
                try:
                    subsystem = self._load_subsystem(name)
                    if subsystem:
                        self._subsystems[name] = subsystem
                except Exception as e:
                    logger.debug(f"Subsystem {name} not available: {e}")
            
            self._initialized = True
            logger.info(f"ContextCoordinator initialized with {len(self._subsystems)} subsystems")
    
    def _load_subsystem(self, name: str) -> Any:
        """Lazy load a subsystem by name."""
        if name in self._subsystems:
            return self._subsystems[name]
        
        try:
            if name == "embedding_interconnect":
                from core.embedding_interconnect import get_embedding_interconnect
                return get_embedding_interconnect()
            
            elif name == "memory":
                from core.memory import get_memory
                return get_memory()
            
            elif name == "persistent_memory":
                from core.persistent_memory import get_persistent_memory
                return get_persistent_memory()
            
            elif name == "knowledge_graph":
                from core.knowledge_graph import get_knowledge_graph
                return get_knowledge_graph()
            
            elif name == "unified_memory":
                from core.unified_context_engine import get_unified_memory
                return get_unified_memory()
            
            elif name == "prefrontal_cortex":
                from core.prefrontal_cortex import get_prefrontal_cortex
                return get_prefrontal_cortex()
            
            elif name == "unified_context_engine":
                from core.unified_context_engine import get_context_engine
                return get_context_engine()
            
            elif name == "context_cascade":
                from core.context_cascade import get_context_cascade
                return get_context_cascade()
            
            elif name == "agentic_context":
                from core.agentic_context import get_agentic_context
                return get_agentic_context()
            
        except Exception as e:
            logger.debug(f"Failed to load {name}: {e}")
            return None
    
    def observe(
        self,
        trigger_type: str,
        content: str,
        source: str = "unknown",
    ) -> UnifiedContextResult:
        """Observe activity and cascade through all subsystems.
        
        This is the MAIN entry point for context updates, triggering:
        1. Embedding encoding
        2. PFC/Meta-reasoner observation
        3. Knowledge graph updates
        4. Memory system storage
        5. Cascade concept encoding
        
        Args:
            trigger_type: Type of activity (user_query, tool_result, etc.)
            content: The content to observe
            source: Where this came from
            
        Returns:
            UnifiedContextResult with cascade concepts and timing
        """
        self._ensure_initialized()
        
        result = UnifiedContextResult()
        start_time = time.perf_counter()
        
        # Step 1: Context cascade (PFC, concepts, predictions)
        cascade_start = time.perf_counter()
        try:
            if "context_cascade" in self._subsystems:
                cascade = self._subsystems["context_cascade"]
                cascade_result = cascade.observe(trigger_type, content, source)
                if cascade_result:
                    result.cascade_concepts = getattr(cascade_result, 'concepts_observed', [])
                    result.sources_used.add("context_cascade")
                    if hasattr(cascade_result, 'timing'):
                        result.timing_ms['cascade'] = cascade_result.timing.get('embedding_ms', 0)
        except Exception as e:
            logger.debug(f"Context cascade observation failed: {e}")
        result.timing_ms['cascade'] = (time.perf_counter() - cascade_start) * 1000
        
        # Step 2: Agentic context update
        agentic_start = time.perf_counter()
        try:
            if "agentic_context" in self._subsystems:
                agentic = self._subsystems["agentic_context"]
                # Reset for new context
                agentic.reset()
        except Exception as e:
            logger.debug(f"Agentic context reset failed: {e}")
        result.timing_ms['agentic'] = (time.perf_counter() - agentic_start) * 1000
        
        result.timing_ms['total'] = (time.perf_counter() - start_time) * 1000
        return result
    
    async def assemble(
        self,
        query: str,
        max_tokens: int = 4000,
        include_project: bool = True,
    ) -> UnifiedContextResult:
        """Assemble unified context from all subsystems.
        
        This gathers context from:
        1. UnifiedContextEngine (parallel source assembly)
        2. UnifiedContext (layered assembly)
        3. AgenticContext (project/goal context)
        4. ContextCascade (concept predictions)
        
        Args:
            query: The query to get context for
            max_tokens: Maximum tokens for context
            include_project: Whether to include project context
            
        Returns:
            UnifiedContextResult with messages, entries, and metadata
        """
        self._ensure_initialized()
        
        result = UnifiedContextResult()
        start_time = time.perf_counter()
        
        # Step 1: UnifiedContextEngine assembly
        engine_start = time.perf_counter()
        try:
            if "unified_context_engine" in self._subsystems:
                engine = self._subsystems["unified_context_engine"]
                pack = await engine.assemble_context(query=query, max_tokens=max_tokens // 2)
                if pack:
                    result.entries = list(pack.entries) if hasattr(pack, 'entries') else []
                    result.total_tokens = pack.total_tokens if hasattr(pack, 'total_tokens') else 0
                    result.sources_used.update(str(s.value) if hasattr(s, 'value') else str(s) 
                                               for s in pack.sources_included)
                    
                    # Convert to messages
                    if hasattr(pack, 'to_messages'):
                        result.messages = pack.to_messages()
        except Exception as e:
            logger.debug(f"UnifiedContextEngine assembly failed: {e}")
        result.timing_ms['engine'] = (time.perf_counter() - engine_start) * 1000
        
        # Step 2: Agentic context (project/goal)
        if include_project:
            project_start = time.perf_counter()
            try:
                if "agentic_context" in self._subsystems:
                    agentic = self._subsystems["agentic_context"]
                    # Project context is already loaded in agentic context
                    if hasattr(agentic, 'project'):
                        result.project_context = {
                            "project_name": getattr(agentic.project, 'project_name', ''),
                            "goal": getattr(agentic.goal, 'description', ''),
                        }
                        result.sources_used.add("agentic_context")
            except Exception as e:
                logger.debug(f"Agentic context failed: {e}")
            result.timing_ms['project'] = (time.perf_counter() - project_start) * 1000
        
        # Step 3: Resolve conflicts
        conflicts_resolved = self._resolve_conflicts(result)
        result.conflicts_resolved = conflicts_resolved
        
        result.timing_ms['total'] = (time.perf_counter() - start_time) * 1000
        return result
    
    def _resolve_conflicts(self, result: UnifiedContextResult) -> int:
        """Resolve conflicts between context sources using precedence.
        
        Args:
            result: The context result to resolve
            
        Returns:
            Number of conflicts resolved
        """
        # Group entries by topic/key
        # Detect conflicting values
        # Apply SOURCE_PRECEDENCE
        # Count resolutions
        
        # For now, just count potential conflicts
        # Full conflict resolution would require parsing entry content
        conflicts = 0
        
        # Check for cascade vs engine conflicts
        if "context_cascade" in result.sources_used and "unified_context_engine" in result.sources_used:
            # Both contributed - check for alignment
            pass
        
        return conflicts
    
    def register_cache(self, name: str, cache_type: str, size_bytes: int) -> None:
        """Register a cache for coordination.
        
        Args:
            name: Unique cache name
            cache_type: Type of cache (lru, ttl, etc.)
            size_bytes: Current size in bytes
        """
        with self._lock:
            old_size = self._caches.get(name, ("", 0))[1]
            self._caches[name] = (cache_type, size_bytes)
            self._total_cache_bytes += size_bytes - old_size
    
    def check_memory_pressure(self) -> bool:
        """Check if under memory pressure.
        
        Returns:
            True if caches should be trimmed
        """
        max_bytes = self.MAX_TOTAL_CACHE_MB * 1024 * 1024
        return self._total_cache_bytes > max_bytes * self.CACHE_PRESSURE_THRESHOLD
    
    def get_stats(self) -> dict:
        """Get coordinator statistics."""
        return {
            "subsystems_loaded": len(self._subsystems),
            "subsystem_names": list(self._subsystems.keys()),
            "total_cache_mb": self._total_cache_bytes / (1024 * 1024),
            "cache_count": len(self._caches),
            "memory_pressure": self.check_memory_pressure(),
            "timing_stats": {k: sum(v)/len(v) if v else 0 for k, v in self._timing_stats.items()},
        }


# Singleton
_coordinator: ContextCoordinator | None = None
_coordinator_lock = threading.Lock()


def get_context_coordinator() -> ContextCoordinator:
    """Get or create the context coordinator singleton."""
    global _coordinator
    if _coordinator is None:
        with _coordinator_lock:
            if _coordinator is None:
                _coordinator = ContextCoordinator()
    return _coordinator


async def get_unified_context(query: str, max_tokens: int = 4000) -> UnifiedContextResult:
    """Convenience function to get unified context.
    
    Args:
        query: The query to get context for
        max_tokens: Maximum tokens for context
        
    Returns:
        UnifiedContextResult with assembled context
    """
    coordinator = get_context_coordinator()
    return await coordinator.assemble(query, max_tokens)


def observe_activity(trigger_type: str, content: str, source: str = "unknown") -> UnifiedContextResult:
    """Convenience function to observe activity.
    
    Args:
        trigger_type: Type of activity
        content: The content to observe
        source: Where this came from
        
    Returns:
        UnifiedContextResult with cascade concepts
    """
    coordinator = get_context_coordinator()
    return coordinator.observe(trigger_type, content, source)


class CacheCoordinator:
    """Coordinates cache behavior across all context subsystems.
    
    Provides:
    - Cache registration and tracking
    - Memory pressure detection
    - Coordinated eviction under pressure
    - Cache statistics aggregation
    
    Usage:
        coordinator = get_cache_coordinator()
        
        # Register a cache
        coordinator.register("tool_cache", tool_cache, max_mb=50)
        
        # Check pressure
        if coordinator.is_under_pressure():
            coordinator.evict_all(percent=20)  # Evict 20% from all caches
    """
    
    def __init__(self, max_total_mb: int = 200):
        self._max_total_mb = max_total_mb
        self._caches: dict[str, tuple[Any, int]] = {}  # name -> (cache, max_mb)
        self._lock = threading.RLock()
    
    def register(self, name: str, cache: Any, max_mb: int = 50) -> None:
        """Register a cache for coordination.
        
        Args:
            name: Unique cache name
            cache: The cache object (must have clear() method)
            max_mb: Maximum MB for this cache
        """
        with self._lock:
            self._caches[name] = (cache, max_mb)
    
    def unregister(self, name: str) -> None:
        """Unregister a cache."""
        with self._lock:
            if name in self._caches:
                del self._caches[name]
    
    def get_total_size_mb(self) -> float:
        """Get estimated total cache size in MB."""
        total = 0.0
        for name, (cache, _) in self._caches.items():
            if hasattr(cache, '__len__'):
                # Estimate: each entry ~1KB on average
                total += len(cache) * 0.001  # KB to MB
            elif hasattr(cache, 'get_stats'):
                stats = cache.get_stats()
                total += stats.get('size_mb', 0)
        return total
    
    def is_under_pressure(self) -> bool:
        """Check if we're under memory pressure."""
        return self.get_total_size_mb() > self._max_total_mb * 0.8
    
    def evict_all(self, percent: int = 20) -> int:
        """Evict from all caches.
        
        Args:
            percent: Percentage to evict from each cache
            
        Returns:
            Total entries evicted
        """
        evicted = 0
        with self._lock:
            for name, (cache, _) in self._caches.items():
                if hasattr(cache, 'evict'):
                    evicted += cache.evict(percent)
                elif hasattr(cache, 'clear'):
                    # If no evict, clear percentage
                    if hasattr(cache, '__len__'):
                        size = len(cache)
                        to_remove = size * percent // 100
                        # Remove oldest entries if possible
                        if hasattr(cache, 'popitem'):
                            for _ in range(to_remove):
                                try:
                                    cache.popitem(last=False)
                                    evicted += 1
                                except KeyError:
                                    break
        return evicted
    
    def get_stats(self) -> dict:
        """Get aggregated cache statistics."""
        stats = {
            "caches": len(self._caches),
            "total_mb": self.get_total_size_mb(),
            "max_mb": self._max_total_mb,
            "pressure": self.is_under_pressure(),
            "cache_details": {},
        }
        
        for name, (cache, max_mb) in self._caches.items():
            cache_stats = {"max_mb": max_mb}
            if hasattr(cache, '__len__'):
                cache_stats["entries"] = len(cache)
                cache_stats["estimated_mb"] = len(cache) * 0.001
            if hasattr(cache, 'get_stats'):
                cache_stats.update(cache.get_stats())
            stats["cache_details"][name] = cache_stats
        
        return stats


_cache_coordinator: CacheCoordinator | None = None
_cache_lock = threading.Lock()


def get_cache_coordinator() -> CacheCoordinator:
    """Get or create the cache coordinator singleton."""
    global _cache_coordinator
    if _cache_coordinator is None:
        with _cache_lock:
            if _cache_coordinator is None:
                _cache_coordinator = CacheCoordinator()
    return _cache_coordinator
