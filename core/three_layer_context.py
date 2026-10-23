"""Three-Layer Context Architecture for Jarvis.

Implements a stratified context system inspired by Letta's memory blocks,
LangGraph's durable execution, and Deep Agents' auto-summarization.

Architecture:
┌──────────────────────────────────────────────────────────────────┐
│                 THREE-LAYER CONTEXT                               │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│   L1 - HOT (Active Token Window)                                   │
│   ┌──────────────────────────────────────────────────────────┐    │
│   │  • Current conversation tokens (~8-32K)                  │    │
│   │  • System prompt + tool schemas + active context          │    │
│   │  • Per-model: Kimi K2.5 has its own L1, GLM-5 its own    │    │
│   │  • LCM meta-guidance injected here                        │    │
│   │  • AUTO-EVICTION: When approaching context limit           │    │
│   └──────────────────────────────────────────────────────────┘    │
│          │ overflow          ▲ recall                               │
│          ▼                    │                                      │
│   L2 - WARM (Self-Editing Memory Blocks)                          │
│   ┌──────────────────────────────────────────────────────────┐    │
│   │  • Per-model memory blocks (primary_block, backend_block) │   │
│   │  • LCM meta-memory block (routing patterns, quality)      │    │
│   │  • Self-editable: models can append/replace/delete        │    │
│   │  • Session-scoped: persists within session, not across    │    │
│   │  • AUTO-SUMMARIZE: When block exceeds threshold            │    │
│   │  • Key innovation: models EDIT their own memory            │    │
│   └──────────────────────────────────────────────────────────┘    │
│          │ consolidation  ▲ retrieval                              │
│          ▼                  │                                       │
│   L3 - COLD (Persistent Vector Storage)                            │
│   ┌──────────────────────────────────────────────────────────┐    │
│   │  • Vector memory (HNSW) for semantic search                │    │
│   │  • Knowledge graph (PageRank) for relationship queries     │    │
│   │  • Persistent memory (user facts, preferences)             │    │
│   │  • Cross-session: survives restarts                        │    │
│   │  • EMBEDDING-BASED: Retrieved by semantic similarity        │    │
│   │  • LCM concept embeddings for pattern matching              │    │
│   └──────────────────────────────────────────────────────────┘    │
│                                                                    │
│   LCM CONTROLLER (manages flow between layers)                    │
│   ┌──────────────────────────────────────────────────────────┐    │
│   │  • Observes all L1 traffic                                 │    │
│   │  • Decides when to promote L2→L1 (context injection)      │    │
│   │  • Decides when to demote L1→L2 (summarize old context)   │    │
│   │  • Decides when to consolidate L2→L3 (long-term storage)   │    │
│   │  • Decides when to retrieve L3→L2 (relevant context)       │    │
│   │  • Tracks what each model "knows" vs "needs to know"      │    │
│   └──────────────────────────────────────────────────────────┘    │
│                                                                    │
└──────────────────────────────────────────────────────────────────┘
"""
import logging
import threading
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

_three_layer: Optional["ThreeLayerContext"] = None
_three_layer_lock = threading.Lock()


class ContextLayer(Enum):
    L1_HOT = "l1_hot"
    L2_WARM = "l2_warm"
    L3_COLD = "l3_cold"


class MemoryBlockOp(Enum):
    APPEND = "append"
    REPLACE = "replace"
    DELETE = "delete"
    CLEAR = "clear"


@dataclass
class MemoryBlock:
    """A self-editing memory block owned by a specific model.
    
    Inspired by Letta's memory blocks - models can modify their own
    memory without affecting other models' memories.
    
    Blocks can be marked read_only to prevent modification by models.
    The LCM can always modify any block regardless of read_only status.
    """
    name: str
    owner: str
    content: str = ""
    max_chars: int = 4000
    last_modified: datetime = field(default_factory=datetime.now)
    modification_count: int = 0
    read_only: bool = False
    
    def append(self, text: str) -> None:
        """Append content to the block."""
        if len(self.content) + len(text) > self.max_chars:
            self._summarize_old()
        self.content = f"{self.content}\n{text}".strip()
        self.last_modified = datetime.now()
        self.modification_count += 1
    
    def replace(self, old: str, new: str) -> bool:
        """Replace content in the block."""
        if old in self.content:
            self.content = self.content.replace(old, new)
            self.last_modified = datetime.now()
            self.modification_count += 1
            return True
        return False
    
    def delete(self, text: str) -> bool:
        """Delete content from the block."""
        if text in self.content:
            self.content = self.content.replace(text, "").strip()
            self.last_modified = datetime.now()
            self.modification_count += 1
            return True
        return False
    
    def clear(self) -> None:
        """Clear the block."""
        self.content = ""
        self.last_modified = datetime.now()
        self.modification_count += 1
    
    def _summarize_old(self) -> None:
        """Summarize older content when block exceeds max size."""
        lines = self.content.split("\n")
        if len(lines) > 4:
            old_lines = lines[:len(lines) // 2]
            new_lines = lines[len(lines) // 2:]
            
            old_summary = f"[Summary of {len(old_lines)} earlier items]"
            self.content = f"{old_summary}\n" + "\n".join(new_lines)
    
    def get_context_string(self) -> str:
        """Get formatted context for injection into L1."""
        if not self.content:
            return ""
        return f"<{self.name}>\n{self.content}\n</{self.name}>"
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "owner": self.owner,
            "content_length": len(self.content),
            "max_chars": self.max_chars,
            "modifications": self.modification_count,
            "last_modified": self.last_modified.isoformat(),
        }


@dataclass
class LayerTransition:
    """A transition of context between layers."""
    source_layer: ContextLayer
    target_layer: ContextLayer
    content: str
    reason: str
    timestamp: datetime = field(default_factory=datetime.now)
    token_count: int = 0


@dataclass
class ThreeLayerStats:
    l1_evictions: int = 0
    l1_to_l2_demotions: int = 0
    l2_to_l1_promotions: int = 0
    l2_to_l3_consolidations: int = 0
    l3_to_l2_retrievals: int = 0
    memory_block_edits: int = 0
    auto_summarizations: int = 0


class ThreeLayerContext:
    """Three-layer context architecture for Jarvis.
    
    Manages the flow of context between L1 (hot), L2 (warm), and L3 (cold)
    layers, with the LCM controller deciding when to move context between
    layers based on relevance, importance, and token budget.
    """
    
    def __init__(
        self,
        l1_token_limit: int = 32000,
        l2_block_limit: int = 4000,
        l1_eviction_threshold: float = 0.85,
    ):
        self._l1_token_limit = l1_token_limit
        self._l2_block_limit = l2_block_limit
        self._l1_eviction_threshold = l1_eviction_threshold
        
        self._memory_blocks: OrderedDict[str, MemoryBlock] = OrderedDict()
        self._blocks_lock = threading.RLock()
        self._max_blocks = 20
        
        self._stats = ThreeLayerStats()
        self._stats_lock = threading.Lock()
        self._max_transitions = 200
        
        self._transitions: deque[LayerTransition] = deque(maxlen=self._max_transitions)
        
        self._version = 0
        self._init_default_blocks()
    
    def _init_default_blocks(self) -> None:
        """Initialize default memory blocks for each model."""
        self._memory_blocks["primary_block"] = MemoryBlock(
            name="primary_block",
            owner="kimi-k2.5",
            max_chars=self._l2_block_limit,
        )
        self._memory_blocks["backend_block"] = MemoryBlock(
            name="backend_block",
            owner="glm-5",
            max_chars=self._l2_block_limit,
        )
        self._memory_blocks["lcm_meta"] = MemoryBlock(
            name="lcm_meta",
            owner="prefrontal_cortex",
            max_chars=self._l2_block_limit,
            read_only=True,
        )
        self._memory_blocks["conversation_summary"] = MemoryBlock(
            name="conversation_summary",
            owner="system",
            max_chars=self._l2_block_limit,
        )
        self._memory_blocks["user_context"] = MemoryBlock(
            name="user_context",
            owner="system",
            max_chars=self._l2_block_limit,
        )
    
    def get_block(self, block_name: str) -> MemoryBlock | None:
        """Get a memory block by name."""
        with self._blocks_lock:
            return self._memory_blocks.get(block_name)
    
    def edit_block(self, block_name: str, operation: MemoryBlockOp, caller: str = "", **kwargs) -> bool:
        """Edit a memory block (self-editing memory pattern).
        
        Models can edit their own blocks but not others'.
        Read-only blocks cannot be edited (except by owner="prefrontal_cortex").
        The LCM can edit any block.
        """
        with self._blocks_lock:
            block = self._memory_blocks.get(block_name)
            if not block:
                return False
            
            if block.read_only and caller != "prefrontal_cortex" and caller != block.owner:
                logger.warning(f"Block {block_name} is read-only (caller: {caller})")
                return False
            
            result = False
            if operation == MemoryBlockOp.APPEND:
                text = kwargs.get("text", "")
                if text:
                    block.append(text)
                    result = True
            elif operation == MemoryBlockOp.REPLACE:
                old = kwargs.get("old", "")
                new = kwargs.get("new", "")
                if old and new:
                    result = block.replace(old, new)
            elif operation == MemoryBlockOp.DELETE:
                text = kwargs.get("text", "")
                if text:
                    result = block.delete(text)
            elif operation == MemoryBlockOp.CLEAR:
                block.clear()
                result = True
            
            if result:
                with self._stats_lock:
                    self._stats.memory_block_edits += 1
                self._version += 1
            
            return result
    
    _PROTECTED_BLOCKS = frozenset(["primary_block", "backend_block", "lcm_meta",
                                     "conversation_summary", "user_context"])

    def create_block(self, name: str, owner: str, max_chars: int | None = None) -> MemoryBlock:
        """Create a new memory block."""
        with self._blocks_lock:
            block = MemoryBlock(
                name=name,
                owner=owner,
                max_chars=max_chars or self._l2_block_limit,
            )
            self._memory_blocks[name] = block
            while len(self._memory_blocks) > self._max_blocks:
                oldest = next(iter(self._memory_blocks))
                if oldest not in self._PROTECTED_BLOCKS:
                    del self._memory_blocks[oldest]
                else:
                    for key in list(self._memory_blocks.keys()):
                        if key not in self._PROTECTED_BLOCKS:
                            del self._memory_blocks[key]
                            break
                    else:
                        break
            return block
    
    def list_blocks(self) -> list[dict]:
        """List all memory blocks."""
        with self._blocks_lock:
            return [b.to_dict() for b in self._memory_blocks.values()]
    
    def get_l2_context_for_model(self, model: str = "primary") -> str:
        """Get L2 context relevant to a specific model.
        
        Returns the concatenation of:
        1. The model's own memory block
        2. The conversation summary
        3. The user context
        4. LCM meta-guidance (always included)
        """
        parts = []
        
        with self._blocks_lock:
            if model == "primary":
                block = self._memory_blocks.get("primary_block")
            elif model == "backend":
                block = self._memory_blocks.get("backend_block")
            else:
                block = None
            
            if block and block.content:
                parts.append(block.get_context_string())
            
            conv_summary = self._memory_blocks.get("conversation_summary")
            if conv_summary and conv_summary.content:
                parts.append(conv_summary.get_context_string())
            
            user_ctx = self._memory_blocks.get("user_context")
            if user_ctx and user_ctx.content:
                parts.append(user_ctx.get_context_string())
            
            lcm_meta = self._memory_blocks.get("lcm_meta")
            if lcm_meta and lcm_meta.content:
                parts.append(lcm_meta.get_context_string())
        
        return "\n\n".join(parts)
    
    def should_evict_l1(self, estimated_tokens: int) -> bool:
        """Check if L1 needs eviction to L2."""
        threshold = self._l1_token_limit * self._l1_eviction_threshold
        return estimated_tokens > threshold
    
    def evict_to_l2(self, conversation_messages: list[dict], max_messages_to_keep: int = 10) -> str:
        """Evict old L1 context to L2 (summarize and store in memory blocks).
        
        Returns a summary of what was evicted.
        """
        if len(conversation_messages) <= max_messages_to_keep:
            return ""
        
        evicted = conversation_messages[:-max_messages_to_keep]
        
        # Use the Conversation Summarizer if available for better compression
        summary_text = ""
        try:
            from core.conversation_summarizer import get_conversation_summarizer
            summarizer = get_conversation_summarizer()
            summary_obj = summarizer.summarize_messages(evicted)
            summary_text = summary_obj.summary_text
        except Exception as e:
            logger.debug(f"Falling back to simple summary: {e}")
            summary_parts = []
            for msg in evicted:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")[:200]
                if content:
                    summary_parts.append(f"{role}: {content}")
            summary_text = "\n".join(summary_parts)
        
        with self._blocks_lock:
            conv_block = self._memory_blocks.get("conversation_summary")
            if conv_block:
                conv_block.append(f"[{datetime.now().strftime('%H:%M')}] {summary_text[:conv_block.max_chars // 2]}")
        
        with self._stats_lock:
            self._stats.l1_evictions += 1
            self._stats.l1_to_l2_demotions += 1
            self._stats.auto_summarizations += 1
        
        self._transitions.append(LayerTransition(
            source_layer=ContextLayer.L1_HOT,
            target_layer=ContextLayer.L2_WARM,
            content=summary_text[:500],
            reason=f"Evicted {len(evicted)} messages (L1 overflow)",
            token_count=sum(len(m.get("content", "")) // 4 for m in evicted),
        ))
        
        return summary_text
    
    def retrieve_from_l3(self, query: str, max_results: int = 3) -> list[dict]:
        """Retrieve relevant context from L3 (persistent storage).
        
        Queries VectorMemory, KnowledgeGraph, and PersistentMemory.
        Results are promoted to L2 for use in current context.
        """
        results = []
        
        try:
            from core.vector_memory import get_vector_memory
            vm = get_vector_memory()
            if vm:
                # search() yields (entry_id, score, metadata) tuples. Calling
                # .get() on them raised AttributeError on the first iteration,
                # which the except below swallowed at debug level — so L3 recall
                # silently returned [] whenever vector memory actually had hits.
                similar = vm.search(query, k=max_results)
                for entry_id, score, metadata in similar:
                    metadata = metadata or {}
                    results.append({
                        "source": "vector_memory",
                        "content": metadata.get("content") or metadata.get("text", ""),
                        "score": score,
                        "entry_id": entry_id,
                        "layer": "L3",
                    })
        except Exception as e:
            logger.warning(f"L3 vector memory retrieval failed: {e}")
        
        try:
            from core.persistent_memory import get_persistent_memory
            pm = get_persistent_memory()
            if pm:
                facts = pm.get_user_facts()
                if facts:
                    with self._blocks_lock:
                        user_block = self._memory_blocks.get("user_context")
                        if user_block:
                            for key, value in facts.items():
                                if isinstance(value, str) and len(value) < 200:
                                    user_block.append(f"{key}: {value}")
        except Exception as e:
            logger.debug(f"L3 persistent memory retrieval failed: {e}")
        
        with self._stats_lock:
            self._stats.l3_to_l2_retrievals += 1
        
        return results
    
    def consolidate_l2_to_l3(self, force: bool = False) -> int:
        """Consolidate L2 memory blocks into L3 persistent storage.
        
        Called periodically to persist important context across sessions.
        """
        consolidated = 0
        
        blocks_to_consolidate = []
        with self._blocks_lock:
            for block_name, block in self._memory_blocks.items():
                if block.modification_count > 5 or force:
                    blocks_to_consolidate.append((block_name, block.content[:500]))
        
        for block_name, content in blocks_to_consolidate:
            try:
                from core.persistent_memory import get_persistent_memory
                pm = get_persistent_memory()
                if pm:
                    pm.add_memory(
                        content=f"[{block_name}] {content}",
                        memory_type="insight",
                        source="three_layer_context",
                        metadata={"tags": ["auto_consolidation", block_name]},
                    )
                    consolidated += 1
            except Exception as e:
                logger.debug(f"L2→L3 consolidation failed for {block_name}: {e}")
        
        with self._stats_lock:
            self._stats.l2_to_l3_consolidations += consolidated
        
        return consolidated
    
    def inject_l2_into_l1(self, messages: list[dict], model: str = "primary") -> list[dict]:
        messages = list(messages)
        l2_context = self.get_l2_context_for_model(model)
        if not l2_context:
            return messages
        
        context_msg = {
            "role": "system",
            "content": f"Relevant memory context:\n{l2_context}",
        }
        
        has_system = any(m.get("role") == "system" for m in messages)
        
        if has_system:
            for i, msg in enumerate(messages):
                if msg.get("role") == "system":
                    messages[i] = {
                        **msg,
                        "content": f"{msg.get('content', '')}\n\n{l2_context}",
                    }
                    break
        else:
            messages.insert(0, context_msg)
        
        with self._stats_lock:
            self._stats.l2_to_l1_promotions += 1
        
        return messages
    
    def get_stats(self) -> dict:
        """Get three-layer context statistics.
        
        Lock ordering: always acquire _blocks_lock before _stats_lock
        to prevent deadlock with other methods that acquire _blocks_lock.
        """
        with self._blocks_lock:
            blocks_snapshot = [b.to_dict() for b in self._memory_blocks.values()]
            l1_token_limit = self._l1_token_limit
            block_count = len(self._memory_blocks)
        with self._stats_lock:
            return {
                "l1_token_limit": l1_token_limit,
                "l2_block_count": block_count,
                "l1_evictions": self._stats.l1_evictions,
                "l1_to_l2_demotions": self._stats.l1_to_l2_demotions,
                "l2_to_l1_promotions": self._stats.l2_to_l1_promotions,
                "l2_to_l3_consolidations": self._stats.l2_to_l3_consolidations,
                "l3_to_l2_retrievals": self._stats.l3_to_l2_retrievals,
                "memory_block_edits": self._stats.memory_block_edits,
                "auto_summarizations": self._stats.auto_summarizations,
                "blocks": blocks_snapshot,
            }
    
    def get_transition_log(self, limit: int = 20) -> list[dict]:
        """Get recent layer transitions."""
        return [
            {
                "from": t.source_layer.value,
                "to": t.target_layer.value,
                "reason": t.reason,
                "tokens": t.token_count,
                "time": t.timestamp.isoformat(),
            }
            for t in list(self._transitions)[-limit:]
        ]
    
    def get_version(self) -> int:
        """Get current version counter (increments on each block edit)."""
        return self._version


def get_three_layer_context() -> ThreeLayerContext:
    """Get the global ThreeLayerContext instance."""
    global _three_layer, _three_layer_lock
    if _three_layer is None:
        with _three_layer_lock:
            if _three_layer is None:
                _three_layer = ThreeLayerContext()
    return _three_layer