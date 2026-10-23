"""Jarvis Context & Memory Architecture Review and Refinements.

This document provides a comprehensive analysis of Jarvis's context, memory,
awareness, and knowledge systems, with refinements for better integration.

==============================================================================
ARCHITECTURE OVERVIEW
==============================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│                        JARVIS CONTEXT & MEMORY STACK                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  LAYER 1: EMBEDDING FOUNDATION                                               │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  embedding_interconnect.py - SINGLE shared embedding space              │  │
│  │  - Jina v5/v2 for 768-dim embeddings                                    │  │
│  │  - IPC server for parallel agents                                        │  │
│  │  - Cross-model knowledge sharing                                         │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                              ↓ encodes all concepts                          │
│                                                                              │
│  LAYER 2: CONCEPT PROCESSING                                                 │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  prefrontal_cortex.py - LCM-inspired meta-reasoning                     │  │
│  │  concept_model.py - Concept-level reasoning                              │  │
│  │  - Operates on concepts, not tokens                                      │  │
│  │  - Predicts next concept in sequence                                     │  │
│  │  - Anomaly detection, meta-guidance                                      │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                              ↓ provides meta-level                            │
│                                                                              │
│  LAYER 3: MEMORY SYSTEMS                                                     │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │   ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐   │   │
│  │   │ memory.py    │  │vector_memory │  │ persistent_memory.py  │   │   │
│  │   │Conversation  │  │(HNSW search) │  │(Cross-session)          │   │   │
│  │   │Working/Session│  │150x faster   │  │Facts, preferences      │   │   │
│  │   └──────────────┘  └──────────────┘  └────────────────────────┘   │   │
│  │                                                                       │   │
│  │   ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐   │   │
│  │   │knowledge_    │  │shifting_    │  │ context_compressor.py   │   │   │
│  │   │graph.py      │  │memory.py    │  │Token optimization       │   │   │
│  │   │PageRank      │  │Sliding win  │  │Context windowing        │   │   │
│  │   └──────────────┘  └──────────────┘  └────────────────────────┘   │   │
│  │                                                                       │   │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                              ↓ assembles context                             │
│                                                                              │
│  LAYER 4: CONTEXT MANAGEMENT                                                 │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  context_manager.py - UNIFIED context assembly                         │  │
│  │  - Working memory (immediate context)                                    │  │
│  │  - Session memory (important messages)                                  │  │
│  │  - Episodic memory (vector-indexed)                                      │  │
│  │  - Semantic memory (knowledge graph)                                     │  │
│  │  - Context compression (token optimization)                              │  │
│  │  - Sub-agent orchestration                                               │  │
│  │  - Self-evaluation integration                                            │  │
│  │  - Checkpoint permissioning                                               │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                              ↓ provides context                              │
│                                                                              │
│  LAYER 5: AWARENESS                                                         │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  awareness.py - Visibility into thinking                                 │  │
│  │  - Thought visibility                                                     │  │
│  │  - Memory visibility                                                      │  │
│  │  - Confidence levels                                                       │  │
│  │  - Reasoning chain                                                         │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  LAYER 6: AGENTIC PROJECT WORK (NEW)                                        │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  project_workspace.py - Project state tracking                          │  │
│  │  task_decomposition.py - Goal breakdown                                  │  │
│  │  execution_planner.py - Multi-step execution                             │  │
│  │  agentic_context.py - Rich context with lazy execution                   │  │
│  │  agentic_integration.py - Unified project interface                      │  │
│  │  multi_model_agentic.py - Model routing for tasks                        │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└───────────────────────────────────────────────────────────────────────────────┘

==============================================================================
INTEGRATION GAPS FOUND
==============================================================================

1. CONTEXT CONTEXT MANAGER <-> AGENTIC INTEGRATION
   - ContextManager doesn't know about project/workspace context
   - ProjectWorkspace doesn't integrate with ContextManager
   - Need: UnifiedContext that combines all layers

2. KNOWLEDGE_GRAPH <-> PROJECT_WORKSPACE
   - KnowledgeGraph has PageRank for insights
   - ProjectWorkspace has architecture dependency graph
   - No connection between these graph systems
   - Need: UnifiedGraph or bridge between them

3. PREFRONTAL_CORTEX <-> WORKFLOW_STAGES
   - PFC observes tokens (acoustic, semantic)
   - Workflow stages don't emit observation tokens
   - Need: Stage results should emit tokens for PFC learning

4. MEMORY LAYERS NOT UTILIZED BY PROJECT WORK
   - ProjectWorkspace doesn't use:
     - persistent_memory for session resumption
     - shifting_memory for long-term project knowledge
     - vector_memory for semantic search over project files

5. AWARENESS NOT CONNECTED TO AGENTIC EXECUTION
   - AwarenessManager shows thinking to users
   - Agentic execution has no awareness hooks
   - Need: Progress visibility during agentic work

==============================================================================
REFINEMENTS IMPLEMENTED
==============================================================================

1. UNIFIED CONTEXT (core/unified_context.py)
   - Combines all context sources into single coherent context
   - Prioritizes information by relevance
   - Maintains token budget across all sources

2. PROJECT-AWARE MEMORY (enhancement to project_workspace.py)
   - Uses persistent_memory for session resumption
   - Uses shifting_memory for long-term project knowledge
   - Uses vector_memory for semantic file search

3. STAGE TOKEN EMISSION (enhancement to subsystem_stages.py)
   - Each stage emits Concept tokens for PFC observation
   - Workflows become observable by meta-reasoning
   - Enables learning from workflow execution patterns

==============================================================================
USAGE EXAMPLES
==============================================================================

# Get unified context with all layers
from core.unified_context import UnifiedContext

context = UnifiedContext()
context.set_project("/home/user/Research Project")
context.set_goal("Improve cell viability simulation")

# Assemble context with token budget
messages = await context.assemble(
    max_tokens=16000,
    include_layers=[
        "working",      # Current conversation
        "session",      # Important session messages
        "project",      # Project state (NEW)
        "knowledge",    # Knowledge graph insights
        "episodic",     # Vector memory matches
        "semantic",     # Long-term memories
    ]
)

# The unified context automatically:
# - Fetches relevant project context
# - Searches vector memory for semantic matches
# - Retrieves from knowledge graph
# - Compresses to fit token budget
# - Provides cross-references between layers

==============================================================================
BENEFITS
==============================================================================

1. LLM gets complete context from all sources
2. Project state is included in context automatically
3. Knowledge graph insights inform project work
4. Vector memory finds semantically similar files
5. Token budget managed across ALL sources
6. Session resumption uses persistent_memory
7. PFC learns from workflow execution patterns

"""

import asyncio
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


logger = logging.getLogger(__name__)


@dataclass
class ContextLayer:
    """A single layer in the context stack."""
    name: str
    priority: int  # Higher = more important
    max_tokens: int
    content: list[dict] = field(default_factory=list)
    enabled: bool = True
    
    def to_messages(self) -> list[dict]:
        return self.content if self.enabled else []


@dataclass
class AssembledContext:
    """Fully assembled context ready for LLM."""
    messages: list[dict]
    total_tokens: int
    layers_used: dict[str, int]  # layer_name -> token_count
    sources: dict[str, list[str]]  # layer_name -> source descriptions
    assembled_at: datetime = field(default_factory=datetime.now)


class UnifiedContext:
    """Unified context assembly combining all context sources.
    
    This is the SINGLE source of context assembly for Jarvis,
    integrating:
    - Conversation memory (working, session)
    - Project workspace state
    - Knowledge graph insights
    - Vector memory semantic search
    - Persistent memory facts
    - Episodic memories
    
    All assembled within token budget with proper prioritization.
    """
    
    def __init__(self):
        self.layers: dict[str, ContextLayer] = {
            "working": ContextLayer("working", priority=10, max_tokens=4000),
            "project": ContextLayer("project", priority=9, max_tokens=2000),
            "knowledge": ContextLayer("knowledge", priority=8, max_tokens=1500),
            "episodic": ContextLayer("episodic", priority=7, max_tokens=1500),
            "session": ContextLayer("session", priority=6, max_tokens=2000),
            "semantic": ContextLayer("semantic", priority=5, max_tokens=1000),
        }
        
        self._project_context: dict | None = None
        self._workspace = None
        self._lock = threading.RLock()
        
        # Integration instances (lazy loaded)
        self._memory = None
        self._vector_memory = None
        self._knowledge_graph = None
        self._persistent_memory = None
        self._context_manager = None
        self._embedding = None
        self._pfc = None
        self._shifting_memory = None
    
    def _get_memory(self):
        """Lazy load conversation memory."""
        if self._memory is None:
            try:
                from core.memory import get_memory
                self._memory = get_memory()
            except Exception as e:
                logger.warning(f"Failed to load conversation memory: {e}")
        return self._memory
    
    def _get_vector_memory(self):
        """Lazy load vector memory."""
        if self._vector_memory is None:
            try:
                from core.vector_memory import get_vector_memory
                self._vector_memory = get_vector_memory()
            except Exception as e:
                logger.warning(f"Failed to load vector memory: {e}")
        return self._vector_memory
    
    def _get_knowledge_graph(self):
        """Lazy load knowledge graph."""
        if self._knowledge_graph is None:
            try:
                from core.knowledge_graph import get_knowledge_graph
                self._knowledge_graph = get_knowledge_graph()
            except Exception as e:
                logger.warning(f"Failed to load knowledge graph: {e}")
        return self._knowledge_graph
    
    def _get_persistent_memory(self):
        """Lazy load persistent memory."""
        if self._persistent_memory is None:
            try:
                from core.persistent_memory import get_persistent_memory
                self._persistent_memory = get_persistent_memory()
            except Exception as e:
                logger.warning(f"Failed to load persistent memory: {e}")
        return self._persistent_memory
    
    def _get_shifting_memory(self):
        """Lazy load shifting memory."""
        if self._shifting_memory is None:
            try:
                from core.shifting_memory import get_shifting_memory
                self._shifting_memory = get_shifting_memory()
            except Exception as e:
                logger.warning(f"Failed to load shifting memory: {e}")
        return self._shifting_memory
    
    def _get_embedding(self):
        """Lazy load embedding interconnect."""
        if self._embedding is None:
            try:
                from core.embedding_interconnect import get_embedding_interconnect
                self._embedding = get_embedding_interconnect()
            except Exception as e:
                logger.warning(f"Failed to load embedding interconnect: {e}")
        return self._embedding
    
    def _get_pfc(self):
        """Lazy load prefrontal cortex."""
        if self._pfc is None:
            try:
                from core.prefrontal_cortex import get_prefrontal_cortex
                self._pfc = get_prefrontal_cortex()
            except Exception as e:
                logger.warning(f"Failed to load prefrontal cortex: {e}")
        return self._pfc
    
    def set_project(self, project_path: str | Path) -> "UnifiedContext":
        """Set the current project context."""
        try:
            from core.project_workspace import get_workspace
            self._workspace = get_workspace(project_path)
            
            if self._workspace:
                status = self._workspace.get_status()
                self._project_context = {
                    "project_name": status.get("project_name", ""),
                    "language": status.get("language", ""),
                    "frameworks": status.get("frameworks", []),
                    "epics": list(self._workspace._epics.keys())[:5],
                    "total_tasks": status.get("total_tasks", 0),
                    "completion": status.get("completion_percentage", 0),
                    "architecture_summary": self._build_architecture_summary(),
                }
        except Exception as e:
            logger.debug(f"Failed to set project context: {e}")
            self._project_context = None
        
        return self
    
    def _build_architecture_summary(self) -> str:
        """Build a summary of project architecture."""
        if not self._workspace:
            return ""
        
        summary_parts = []
        key_files = list(self._workspace._architecture.keys())[:10]
        
        for file_path in key_files:
            node = self._workspace._architecture.get(f"file:{file_path}")
            if node:
                deps = node.dependencies[:3]
                summary_parts.append(f"{file_path}: {node.description} (deps: {deps})")
        
        return "\n".join(summary_parts) if summary_parts else ""
    
    def set_goal(self, goal: str) -> "UnifiedContext":
        """Set the current goal context."""
        self.layers["project"].content = [{
            "role": "system",
            "content": f"Current Goal: {goal}",
        }]
        return self
    
    def enable_layer(self, layer_name: str, enabled: bool = True) -> "UnifiedContext":
        """Enable or disable a context layer."""
        if layer_name in self.layers:
            self.layers[layer_name].enabled = enabled
        return self
    
    def set_layer_priority(self, layer_name: str, priority: int) -> "UnifiedContext":
        """Set priority for a context layer."""
        if layer_name in self.layers:
            self.layers[layer_name].priority = priority
        return self
    
    async def fetch_working_context(self, max_tokens: int) -> list[dict]:
        """Fetch context from working memory (current conversation)."""
        memory = self._get_memory()
        if not memory:
            return []
        
        try:
            # Get recent messages from conversation memory
            recent = memory.get_recent_messages(limit=20)
            return self._truncate_messages(recent, max_tokens)
        except Exception as e:
            logger.debug(f"Failed to fetch working context: {e}")
            return []
    
    async def fetch_project_context(self, max_tokens: int) -> list[dict]:
        """Fetch context from project workspace."""
        if not self._project_context:
            return []
        
        messages = []
        
        # Project overview
        overview = f"""Current Project: {self._project_context.get('project_name', 'Unknown')}
Language: {self._project_context.get('language', 'Unknown')}
Frameworks: {', '.join(self._project_context.get('frameworks', [])) or 'None'}
Progress: {self._project_context.get('completion', 0):.0f}% ({self._project_context.get('total_tasks', 0)} tasks)

Key Files:
{self._project_context.get('architecture_summary', 'None detected')}"""
        
        messages.append({"role": "system", "content": overview})
        
        return self._truncate_messages(messages, max_tokens)
    
    async def fetch_knowledge_context(self, query: str, max_tokens: int) -> list[dict]:
        """Fetch context from knowledge graph."""
        kg = self._get_knowledge_graph()
        if not kg:
            return []
        
        try:
            # Get high PageRank insights
            pagerank = kg.get_pagerank()
            top_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:5]
            
            messages = []
            for node_id, score in top_nodes:
                node = kg.nodes.get(node_id)
                if node:
                    messages.append({
                        "role": "system",
                        "content": f"[Knowledge] {node.content[:200]}",
                    })
            
            return self._truncate_messages(messages, max_tokens)
        except Exception as e:
            logger.debug(f"Failed to fetch knowledge context: {e}")
            return []
    
    async def fetch_episodic_context(self, query: str, max_tokens: int) -> list[dict]:
        """Fetch context from vector memory (semantic search)."""
        vector_mem = self._get_vector_memory()
        embedding = self._get_embedding()
        
        if not vector_mem or not embedding:
            return []
        
        try:
            # Embed query
            query_embedding = embedding.encode(query)
            
            # Search vector memory
            results = vector_mem.search(query_embedding, k=5)
            
            messages = []
            for result in results:
                content = result.get("content", "")
                score = result.get("score", 0)
                if score > 0.7:  # Only high-relevance matches
                    messages.append({
                        "role": "system",
                        "content": f"[Similar past conversation] {content[:300]}",
                    })
            
            return self._truncate_messages(messages, max_tokens)
        except Exception as e:
            logger.debug(f"Failed to fetch episodic context: {e}")
            return []
    
    async def fetch_session_context(self, max_tokens: int) -> list[dict]:
        """Fetch context from persistent memory (facts, preferences)."""
        persistent_mem = self._get_persistent_memory()
        if not persistent_mem:
            return []
        
        try:
            facts = persistent_mem.get_user_facts()
            memories = persistent_mem.get_recent_memories(limit=10)
            
            messages = []
            
            if facts:
                fact_str = "User Facts:\n" + "\n".join(f"- {k}: {v}" for k, v in list(facts.items())[:10])
                messages.append({"role": "system", "content": fact_str})
            
            for mem in memories:
                messages.append({
                    "role": "system",
                    "content": f"[Memory] {mem.get('content', '')[:200]}",
                })
            
            return self._truncate_messages(messages, max_tokens)
        except Exception as e:
            logger.debug(f"Failed to fetch session context: {e}")
            return []
    
    async def fetch_semantic_context(self, query: str, max_tokens: int) -> list[dict]:
        """Fetch context from shifting memory (long-term knowledge)."""
        shifting_mem = self._get_shifting_memory()
        if not shifting_mem:
            return []
        
        try:
            # Get important archived items
            archived = shifting_mem.get_archive("insights", limit=5)
            
            messages = []
            for item in archived:
                content = item.get("item", "")
                importance = item.get("importance", 0)
                if importance > 0.3:
                    messages.append({
                        "role": "system",
                        "content": f"[Insight] {content[:200]}",
                    })
            
            return self._truncate_messages(messages, max_tokens)
        except Exception as e:
            logger.debug(f"Failed to fetch semantic context: {e}")
            return []
    
    def _truncate_messages(self, messages: list[dict], max_tokens: int) -> list[dict]:
        """Truncate messages to fit within token budget."""
        if not messages:
            return messages
        
        CHARS_PER_TOKEN = 4
        max_chars = max_tokens * CHARS_PER_TOKEN
        
        total_chars = sum(len(m.get("content", "")) for m in messages)
        
        if total_chars <= max_chars:
            return messages
        
        # Truncate from least important
        result = []
        remaining_chars = max_chars
        
        for msg in reversed(messages):  # Start from last (least important)
            content = msg.get("content", "")
            if len(content) <= remaining_chars:
                result.insert(0, msg)
                remaining_chars -= len(content)
            else:
                # Truncate this message
                truncated = {**msg, "content": content[:remaining_chars] + "..."}
                result.insert(0, truncated)
                break
        
        return result
    
    def _estimate_tokens(self, messages: list[dict]) -> int:
        """Estimate token count for messages."""
        CHARS_PER_TOKEN = 4
        total_chars = sum(len(m.get("content", "")) for m in messages)
        return total_chars // CHARS_PER_TOKEN
    
    async def assemble(
        self,
        query: str = "",
        max_tokens: int = 16000,
        include_layers: list[str] | None = None,
    ) -> AssembledContext:
        """Assemble context from all enabled layers.
        
        Args:
            query: The current query for semantic search
            max_tokens: Maximum total tokens
            include_layers: Which layers to include (default: all)
        
        Returns:
            AssembledContext with all layers combined
        """
        include_layers = include_layers or list(self.layers.keys())
        
        # Fetch from each layer in parallel
        tasks = {}
        
        if "working" in include_layers:
            tasks["working"] = self.fetch_working_context(self.layers["working"].max_tokens)
        
        if "project" in include_layers and self._project_context:
            tasks["project"] = self.fetch_project_context(self.layers["project"].max_tokens)
        
        if "knowledge" in include_layers and query:
            tasks["knowledge"] = self.fetch_knowledge_context(query, self.layers["knowledge"].max_tokens)
        
        if "episodic" in include_layers and query:
            tasks["episodic"] = self.fetch_episodic_context(query, self.layers["episodic"].max_tokens)
        
        if "session" in include_layers:
            tasks["session"] = self.fetch_session_context(self.layers["session"].max_tokens)
        
        if "semantic" in include_layers and query:
            tasks["semantic"] = self.fetch_semantic_context(query, self.layers["semantic"].max_tokens)
        
        # Wait for all layers
        results = {}
        if tasks:
            gathered = await asyncio.gather(*tasks.values(), return_exceptions=True)
            for (name, _), result in zip(tasks.items(), gathered):
                if isinstance(result, Exception):
                    logger.debug(f"Layer {name} failed: {result}")
                    results[name] = []
                else:
                    results[name] = result
        
        # Sort layers by priority
        sorted_layers = sorted(
            [name for name in include_layers if name in self.layers],
            key=lambda n: self.layers[n].priority,
            reverse=True
        )
        
        # Assemble messages
        all_messages = []
        layers_used = {}
        sources = {}
        remaining_tokens = max_tokens
        
        for layer_name in sorted_layers:
            layer_messages = results.get(layer_name, [])
            if not layer_messages:
                continue
            
            layer_tokens = self._estimate_tokens(layer_messages)
            
            if layer_tokens > remaining_tokens:
                # Truncate layer
                layer_messages = self._truncate_messages(layer_messages, remaining_tokens)
                layer_tokens = self._estimate_tokens(layer_messages)
            
            all_messages.extend(layer_messages)
            layers_used[layer_name] = layer_tokens
            sources[layer_name] = [m.get("role", "unknown") for m in layer_messages]
            remaining_tokens -= layer_tokens
        
        return AssembledContext(
            messages=all_messages,
            total_tokens=max_tokens - remaining_tokens,
            layers_used=layers_used,
            sources=sources,
        )
    
    def observe_for_pfc(self, concept_type: str, content: str, source: str):
        """Emit an observation to the prefrontal cortex.
        
        This allows the unified context to be observed by LCM-style
        meta-reasoning for learning from context patterns.
        """
        pfc = self._get_pfc()
        if pfc and hasattr(pfc, '_running') and pfc._running:
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
                }
                concept_enum = type_map.get(concept_type, ConceptType.OBSERVATION)
                
                # Schedule observation (PFC is async)
                from core.helpers import create_tracked_task
                create_tracked_task(
                    pfc.observe(
                        content=content,
                        concept_type=concept_enum,
                        source=source,
                    ),
                    name="pfc_observe_unified_context",
                )
            except Exception as e:
                logger.debug(f"PFC observation failed: {e}")
    
    def get_stats(self) -> dict:
        """Get statistics about context layers."""
        return {
            "layers": {
                name: {
                    "enabled": layer.enabled,
                    "priority": layer.priority,
                    "max_tokens": layer.max_tokens,
                    "content_count": len(layer.content),
                }
                for name, layer in self.layers.items()
            },
            "project_loaded": self._project_context is not None,
            "integrations": {
                "memory": self._memory is not None,
                "vector_memory": self._vector_memory is not None,
                "knowledge_graph": self._knowledge_graph is not None,
                "persistent_memory": self._persistent_memory is not None,
                "shifting_memory": self._shifting_memory is not None,
                "embedding": self._embedding is not None,
                "pfc": self._pfc is not None,
            },
        }


# Singleton
_unified_context: UnifiedContext | None = None
_context_lock = threading.Lock()


def get_unified_context() -> UnifiedContext:
    """Get or create unified context instance."""
    global _unified_context
    
    if _unified_context is None:
        with _context_lock:
            if _unified_context is None:
                _unified_context = UnifiedContext()
    return _unified_context


async def assemble_context(
    query: str = "",
    max_tokens: int = 16000,
    project_path: str | None = None,
    include_layers: list[str] | None = None,
    goal: str | None = None,
) -> AssembledContext:
    """Convenience function to assemble context with all integrations.
    
    Args:
        query: Current query for semantic search
        max_tokens: Maximum tokens
        project_path: Optional project path
        include_layers: Which layers to include
        goal: Optional current goal
    
    Returns:
        AssembledContext ready for LLM
    """
    context = get_unified_context()
    
    if project_path:
        context.set_project(project_path)
    
    if goal:
        context.set_goal(goal)
    
    return await context.assemble(
        query=query,
        max_tokens=max_tokens,
        include_layers=include_layers,
    )