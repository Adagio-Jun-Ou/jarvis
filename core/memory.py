"""Conversation memory management for Jarvis.

Enhanced with multi-layer context assembly:
- Working memory (immediate context)
- Session memory (important messages)
- Episodic memory (vector-indexed)
- Semantic memory (knowledge graph)

Integrates with vector_memory and knowledge_graph for context-aware retrieval.
"""
import asyncio
import json
import logging
import threading
import time
from collections import OrderedDict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN = 4
DEFAULT_MAX_TOKENS = 128000  # Match typical model context window


class MemoryLayer(Enum):
    WORKING = "working"
    SESSION = "session"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


class ContextPack:
    """Assembled context from multiple memory layers."""
    
    def __init__(self):
        self.working: list[dict] = []
        self.session: list[dict] = []
        self.episodic: list[dict] = []
        self.semantic: list[dict] = []
        self.total_tokens: int = 0
        self.assembled_at: datetime = datetime.now()
    
    def to_messages(self) -> list[dict]:
        messages = []
        messages.extend(self.semantic)
        messages.extend(self.episodic)
        messages.extend(self.session)
        messages.extend(self.working)
        return messages


class ContextCache:
    """Cache for assembled context packs."""
    
    def __init__(self, max_size: int = 50, ttl_seconds: float = 300):
        self._cache: OrderedDict[str, tuple[ContextPack, float]] = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> ContextPack | None:
        with self._lock:
            if key in self._cache:
                pack, timestamp = self._cache[key]
                if time.time() - timestamp < self._ttl:
                    self._cache.move_to_end(key)
                    self._hits += 1
                    return pack
                del self._cache[key]
            self._misses += 1
            return None
    
    def set(self, key: str, pack: ContextPack) -> None:
        with self._lock:
            self._cache[key] = (pack, time.time())
            self._cache.move_to_end(key)
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)
    
    def invalidate(self) -> None:
        with self._lock:
            self._cache.clear()
    
    def stats(self) -> dict:
        with self._lock:
            total = self._hits + self._misses
            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total if total > 0 else 0,
            }


class ConversationMemory:
    """Manages conversation history and context with smart optimization.

    Features:
    - Token-aware context management using full model context window
    - Efficient message trimming only when approaching limits
    - Importance tracking for messages
    - Long-term and short-term memory separation
    - Thread-safe operations
    - Cached token counts for O(1) total token calculation
    """

    MAX_IMPORTANT_MESSAGES = 100
    MAX_TOPIC_IMPORTANCE = 200
    MAX_LONG_TERM_MEMORY = 100

    def __init__(
        self,
        max_turns: int = 100,
        max_tokens: int | None = None,
        storage_path: Path | None = None,
        strategy: str = "smart",
        use_vector_memory: bool = True,
        context_window: int | None = None,
    ):
        self.max_turns = max_turns
        self.max_tokens = max_tokens if max_tokens is not None else DEFAULT_MAX_TOKENS
        self.storage_path = storage_path
        self.strategy = strategy
        self.use_vector_memory = use_vector_memory
        self.history: list[dict[str, Any]] = []
        self.context: dict[str, Any] = {}
        self._lock = threading.RLock()

        self.important_messages: dict[int, float] = {}
        self.topic_importance: dict[str, float] = {}
        self.long_term_memory: list[dict[str, Any]] = []

        # Cached token count for O(1) total token calculation
        self._total_tokens = 0
        self._pending_eviction = None
        self._version = 0  # Generation counter for optimistic eviction locking

        self._vector_memory = None
        if use_vector_memory:
            try:
                from core.vector_memory import get_vector_memory
                self._vector_memory = get_vector_memory()
            except Exception as e:
                logger.debug(f"Vector memory not available: {e}")

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (4 chars per token average)."""
        return max(1, len(text) // CHARS_PER_TOKEN) if text else 0

    def _count_history_tokens(self) -> int:
        """Get total tokens - O(1) using cached count."""
        with self._lock:
            return self._total_tokens

    def _recalculate_total_tokens(self) -> None:
        """Recalculate total tokens from scratch (for recovery/correction)."""
        with self._lock:
            self._total_tokens = sum(
                self._estimate_tokens(msg.get("content", ""))
                for msg in self.history
            )

    def add_message(self, role: str, content: str, importance: float | None = None) -> None:
        """Add a message to conversation history."""
        evicted_messages = None
        with self._lock:
            # Deduplicate: check if ANY recent message (last 3) has same content
            if self.history:
                for recent_msg in self.history[-3:]:
                    if recent_msg.get("role") == role and recent_msg.get("content") == content:
                        logger.debug(f"[MEMORY] Skipping duplicate message: {role}: {content[:50]}...")
                        return
            
            token_count = self._estimate_tokens(content)
            self.history.append({
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat(),
                "token_count": token_count,
            })
            self._total_tokens += token_count

            if importance is not None:
                importance = max(0.0, min(1.0, importance))
                self.important_messages[len(self.history) - 1] = importance

            evicted_messages = self._collect_eviction_candidates()
            self._prune_importance_dicts()
        
        # Summarize evicted messages outside the lock (slow LLM call)
        if evicted_messages:
            self._summarize_and_store_evicted(evicted_messages)

    def add_user_message(self, content: str, importance: float | None = None) -> None:
        """Add a user message with auto-detected importance."""
        if importance is None:
            importance = 0.7
            content_lower = content.lower()
            if any(w in content_lower for w in ["remember", "important", "don't forget", "save"]):
                if len(content) > 200:
                    importance = 0.85
                else:
                    importance = 0.9
            elif any(w in content_lower for w in ["my name", "i am", "i'm", "my favorite", "i like"]):
                importance = 0.85
        self.add_message("user", content, importance)

    def add_assistant_message(self, content: str, importance: float | None = None) -> None:
        """Add an assistant message."""
        self.add_message("assistant", content, importance if importance is not None else 0.6)

    def add_tool_result(self, tool_name: str, result: str | dict) -> None:
        """Add a tool result message (truncated if too long)."""
        safe_name = tool_name.replace("\n", " ").replace("\r", " ")[:100]
        if isinstance(result, dict):
            result = str(result)
        evicted_messages = None
        with self._lock:
            # Truncate very long tool results
            max_result_len = 2000
            truncated_result = result[:max_result_len] if len(result) > max_result_len else result
            msg = {
                "role": "tool",
                "name": safe_name,
                "content": truncated_result,
                "timestamp": datetime.now().isoformat(),
            }
            self.history.append(msg)
            self.important_messages[len(self.history) - 1] = 0.5
            # Update token cache (include the [Tool: name] prefix overhead)
            tool_prefix = f"[Tool: {safe_name}]\n"
            self._total_tokens += self._estimate_tokens(tool_prefix + truncated_result)
            evicted_messages = self._collect_eviction_candidates()
            self._prune_importance_dicts()
        
        # Summarize evicted messages outside the lock (slow LLM call)
        if evicted_messages:
            self._summarize_and_store_evicted(evicted_messages)

    def mark_important(self, index: int, importance: float = 1.0) -> None:
        """Mark a message as important."""
        with self._lock:
            if 0 <= index < len(self.history):
                self.important_messages[index] = importance

    def get_history(self) -> list[dict]:
        """Get conversation history."""
        with self._lock:
            return self.history.copy()

    @property
    def message_count(self) -> int:
        """Thread-safe count of user/assistant messages (avoids building full list)."""
        with self._lock:
            return sum(1 for m in self.history if m["role"] in ("user", "assistant"))

    def get_recent_messages(self, n: int = 10) -> list[dict]:
        """Get last n messages formatted for LLM API (avoids building full list)."""
        with self._lock:
            messages = []
            for msg in reversed(self.history):
                if msg["role"] == "user":
                    messages.append({"role": msg["role"], "content": msg["content"]})
                elif msg["role"] == "assistant":
                    messages.append({"role": msg["role"], "content": msg["content"]})
                elif msg["role"] == "tool":
                    content = msg.get("content", "")
                    tool_name = msg.get("name", "unknown_tool")
                    safe_name = tool_name.replace("\n", " ").replace("\r", " ")[:100]
                    messages.append({"role": "assistant", "content": f"[Tool: {safe_name}]\n{content}"})
                if len(messages) >= n:
                    break
            messages.reverse()
            return messages

    def get_messages_for_llm(self, max_messages: int | None = None) -> list[dict]:
        """Get messages formatted for LLM API within token budget."""
        with self._lock:
            # First check by message count
            if max_messages is None:
                max_messages = self.max_turns * 2

            # Check token budget
            total_tokens = self._count_history_tokens()

            if total_tokens <= self.max_tokens and len(self.history) <= max_messages:
                # Everything fits, return all
                messages = []
                for msg in self.history:
                    if msg["role"] in ["user", "assistant"]:
                        messages.append({
                            "role": msg["role"],
                            "content": msg["content"],
                        })
                    elif msg["role"] == "tool":
                        content = msg.get("content", "")
                        tool_name = msg.get("name", "unknown_tool")
                        safe_name = tool_name.replace("\n", " ").replace("\r", " ")[:100]
                        messages.append({
                            "role": "assistant",
                            "content": f"[Tool: {safe_name}]\n{content}",
                        })
                return messages

            # Need to trim - use smart selection
            if self.strategy == "smart":
                return self._smart_select_messages(max_messages)
            else:
                return self._recent_select_messages(max_messages)

    def _smart_select_messages(self, max_messages: int) -> list[dict]:
        """Select messages prioritizing important ones within token budget.
        
        Tool messages are included when they are adjacent to selected user/assistant
        messages, preserving conversational context.
        """
        # Score all messages
        scores = {}
        for i, item_history in enumerate(self.history):
            recency = i / max(len(self.history), 1)
            importance = self.important_messages.get(i, 0.5)
            scores[i] = 0.3 * recency + 0.7 * importance

        # Sort by score
        sorted_indices = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        # Add messages until budget
        selected_indices = []
        total_tokens = 0

        for idx in sorted_indices:
            msg = self.history[idx]
            if msg["role"] == "tool":
                continue
            if msg["role"] not in ["user", "assistant"]:
                continue
            msg_tokens = self._estimate_tokens(msg.get("content", ""))
            if total_tokens + msg_tokens <= self.max_tokens and len(selected_indices) < max_messages:
                selected_indices.append(idx)
                total_tokens += msg_tokens

        # Sort by position for proper conversation order
        selected_indices.sort()
        selected_set = set(selected_indices)

        # Include tool messages adjacent to selected messages (before and after)
        adjacency_set = set(selected_set)
        if selected_set:
            min_idx = min(selected_set)
            max_idx = max(selected_set)
            for i in range(max(0, min_idx - 1), min(len(self.history), max_idx + 2)):
                if self.history[i]["role"] == "tool":
                    adjacency_set.add(i)

        messages = []
        for i in range(len(self.history)):
            msg = self.history[i]
            if i in selected_set:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                })
            elif i in adjacency_set and msg["role"] == "tool":
                tool_name = msg.get("name", "unknown_tool").replace("\n", " ").replace("\r", " ")[:100]
                content = msg.get("content", "")
                messages.append({
                    "role": "assistant",
                    "content": f"[Tool: {tool_name}]\n{content}",
                })

        return messages

    def _recent_select_messages(self, max_messages: int) -> list[dict]:
        messages = []
        total_tokens = 0
        recent = list(reversed(self.history))
        selected_indices = set()
        pending_tools = []

        for i, msg in enumerate(recent):
            if len(selected_indices) >= max_messages:
                break

            if msg["role"] == "tool":
                pending_tools.append(msg)
                selected_indices.add(i)
                continue

            if msg["role"] not in ("user", "assistant", "system"):
                continue

            msg_tokens = self._estimate_tokens(msg.get("content", ""))
            if total_tokens + msg_tokens > self.max_tokens * 0.9:
                break

            selected_indices.add(i)
            messages.append({"role": msg["role"], "content": msg["content"]})
            total_tokens += msg_tokens

            if pending_tools:
                for tool_msg in reversed(pending_tools):
                    tool_name = tool_msg.get("name", "unknown_tool")
                    tool_content = tool_msg.get("content", "")
                    tool_tokens = self._estimate_tokens(tool_content)
                    if total_tokens + tool_tokens > self.max_tokens * 0.95:
                        break
                    tool_entry = {
                        "role": "tool",
                        "content": tool_content,
                        "name": tool_name,
                    }
                    if tool_msg.get("tool_call_id"):
                        tool_entry["tool_call_id"] = tool_msg["tool_call_id"]
                    messages.append(tool_entry)
                    total_tokens += tool_tokens
                pending_tools = []

        messages.reverse()
        return messages

    def _collect_eviction_candidates(self) -> list[dict] | None:
        """Determine if eviction is needed and collect candidates (fast, called under lock).
        
        Returns:
            List of evicted messages if eviction needed, None otherwise.
            The caller should apply eviction via _apply_eviction() AFTER summarizing
            outside the lock.
        """
        total_tokens = self._total_tokens

        if total_tokens <= self.max_tokens and len(self.history) <= self.max_turns * 2:
            return None

        if len(self.history) < 10:
            return None

        # Determine keep_recent count
        keep_recent = min(len(self.history) // 2, 30)
        try:
            from core.context_guardian import get_context_guardian
            guardian = get_context_guardian()
            if guardian.should_auto_evict(total_tokens):
                plan = guardian.get_eviction_plan(total_tokens, len(self.history))
                keep_recent = max(5, plan["keep_recent"])
        except Exception as e:
            logger.debug(f"Context guardian not available for eviction planning: {e}")

        recent_count = keep_recent or min(len(self.history) // 2, 30)
        to_keep_indices = set(range(len(self.history) - recent_count, len(self.history)))

        # Add important old messages
        for i, msg in enumerate(self.history[:-recent_count] if recent_count < len(self.history) else []):
            if self.important_messages.get(i, 0) > 0.8:
                to_keep_indices.add(i)
                if len(to_keep_indices) > len(self.history) * 0.8:
                    break

        evicted_indices = sorted(set(range(len(self.history))) - to_keep_indices)
        if not evicted_indices:
            return None

        # Collect evicted messages and store eviction plan on self for later application
        evicted_messages = [self.history[i] for i in evicted_indices]
        self._version += 1
        self._pending_eviction = {
            "to_keep_indices": to_keep_indices,
            "evicted_indices": evicted_indices,
            "tokens_before": total_tokens,
            "version": self._version,
        }
        return evicted_messages

    def _summarize_and_store_evicted(self, evicted_messages: list[dict]) -> None:
        """Summarize evicted messages outside the lock, then apply eviction under lock."""
        # Step 1: Summarize outside lock (slow LLM call)
        summary_text = None
        summary_topics = []
        try:
            from core.conversation_summarizer import get_conversation_summarizer
            summarizer = get_conversation_summarizer()
            summary = summarizer.summarize_messages(evicted_messages)
            if summary.summary_text:
                summary_text = summary.summary_text[:500]
                summary_topics = summary.topics[:5]
        except Exception as e:
            logger.warning(f"Summarization not available: {e}")

        # Step 2: Apply eviction under lock (fast), with version check
        with self._lock:
            pending = getattr(self, '_pending_eviction', None)
            if pending is None:
                return

            # Optimistic locking: if version changed, history was modified during summarization
            # Re-collect eviction candidates instead of applying stale indices
            if pending.get("version") != self._version:
                effective_max = self.max_turns * 2
                try:
                    from core.evolution_orchestrator import EvolutionGene, get_evolution_orchestrator
                    orchestrator = get_evolution_orchestrator()
                    if orchestrator:
                        retention = orchestrator.genes.get("memory_retention", EvolutionGene("memory_retention", 0.5, 0, 1)).value
                        effective_max = int(self.max_turns * (1 + retention))
                except Exception:
                    pass
                if self._total_tokens <= self.max_tokens and len(self.history) <= effective_max:
                    self._pending_eviction = None
                    return
                # Re-collect eviction candidates with current history
                new_evicted = self._collect_eviction_candidates()
                if new_evicted is None:
                    self._pending_eviction = None
                    return
                # Re-evict recursively (but only once to avoid infinite loops)
                pending = self._pending_eviction
                if pending is None:
                    return

            to_keep_indices = pending["to_keep_indices"]

            # Validate indices are still in bounds
            history_len = len(self.history)
            valid_keep = {i for i in to_keep_indices if i < history_len}
            if len(valid_keep) < len(to_keep_indices) * 0.5:
                # Too many indices became invalid; skip eviction
                self._pending_eviction = None
                return

            new_history = [self.history[i] for i in sorted(valid_keep) if i < history_len]

            # Recalculate total tokens
            self._total_tokens = sum(
                int(msg.get("token_count", self._estimate_tokens(msg.get("content", ""))))
                for msg in new_history
            )

            # Update importance indices
            old_to_new = {old: new for new, old in enumerate(sorted(valid_keep))}
            new_important = {}
            for old_idx, score in self.important_messages.items():
                if old_idx in old_to_new:
                    new_important[old_to_new[old_idx]] = score

            self.history = new_history
            self.important_messages = new_important
            self._pending_eviction = None

            # Also evict to three-layer context if available
            try:
                from core.three_layer_context import get_three_layer_context
                tlc = get_three_layer_context()
                tlc.evict_to_l2(self.history)
            except Exception as e:
                logger.debug(f"Three-layer context not available: {e}")
            logger.info(f"Compressed history: {len(self.history)} messages, ~{self._total_tokens} tokens")

        # Step 3: Store summary in long-term memory (no lock needed, add_to_long_term has its own locking)
        if summary_text:
            self.add_to_long_term(
                summary=summary_text,
                topics=summary_topics,
                importance=0.5,
            )

    def _compress_history(self, keep_recent: int | None = None) -> None:
        """Compress history by evicting old messages, keeping recent ones.

        This is a synchronous fallback that works under the lock.
        It trims old messages and optionally stores a summary in long-term memory.
        Called from _trim_if_needed when the context guardian path fails.

        Args:
            keep_recent: Number of recent messages to keep. If None, keeps half.
        """
        with self._lock:
            if len(self.history) < 10:
                return

            if keep_recent is None:
                keep_recent = min(len(self.history) // 2, 30)
            keep_recent = max(5, min(keep_recent, len(self.history) - 5))

            to_keep = set(range(len(self.history) - keep_recent, len(self.history)))
            for i, msg in enumerate(self.history[:-keep_recent] if keep_recent < len(self.history) else []):
                if self.important_messages.get(i, 0) > 0.8:
                    to_keep.add(i)
                    if len(to_keep) > len(self.history) * 0.8:
                        break

            evicted = [self.history[i] for i in sorted(set(range(len(self.history))) - to_keep)]
            new_history = [self.history[i] for i in sorted(to_keep) if i < len(self.history)]

            if len(new_history) >= len(self.history):
                return

            old_to_new = {}
            for new_idx, old_idx in enumerate(sorted(i for i in to_keep if i < len(self.history))):
                old_to_new[old_idx] = new_idx

            new_important = {}
            for old_idx, score in self.important_messages.items():
                if old_idx in old_to_new:
                    new_important[old_to_new[old_idx]] = score

            self.history = new_history
            self.important_messages = new_important
            self._total_tokens = sum(
                int(msg.get("token_count", self._estimate_tokens(msg.get("content", ""))))
                for msg in self.history
            )

        if evicted:
            try:
                summary_parts = [m.get("content", "")[:200] for m in evicted[:5]]
                summary = " | ".join(summary_parts)
                self.add_to_long_term(summary=summary[:500], topics=[], importance=0.3)
            except Exception as e:
                logger.warning(f"Failed to store evicted summary: {e}")

        logger.info(f"Compressed history: {len(evicted)} evicted, {len(self.history)} kept, ~{self._total_tokens} tokens")

    def _trim_if_needed(self) -> None:
        """Trim history if exceeding limits. Called from contexts that already hold lock.
        
        For callers that don't hold the lock, use _collect_eviction_candidates() +
        _summarize_and_store_evicted() instead.
        """
        total_tokens = self._total_tokens

        try:
            from core.evolution_orchestrator import get_evolution_orchestrator
            orchestrator = get_evolution_orchestrator()
            if orchestrator and hasattr(orchestrator, 'get_behavior_params'):
                params = orchestrator.get_behavior_params()
                retention = params.get("memory_retention", 0.7)
                effective_max = int(self.max_turns * (0.5 + retention))
            else:
                effective_max = self.max_turns * 2
        except Exception as e:
            logger.debug(f"Evolution orchestrator unavailable for trim: {e}")
            effective_max = self.max_turns * 2

        if total_tokens > self.max_tokens or len(self.history) > effective_max:
            try:
                from core.context_guardian import get_context_guardian
                guardian = get_context_guardian()
                if guardian.should_auto_evict(total_tokens):
                    plan = guardian.get_eviction_plan(total_tokens, len(self.history))
                    keep_recent = max(5, plan["keep_recent"])
                    if keep_recent < len(self.history):
                        from core.context_guardian import GuardianAction, PressureLevel
                        tokens_before = total_tokens
                        self._compress_history(keep_recent=keep_recent)
                        from core.three_layer_context import get_three_layer_context
                        tlc = get_three_layer_context()
                        tlc.evict_to_l2(self.history, max_messages_to_keep=keep_recent)
                        guardian.record_action(GuardianAction(
                            level=PressureLevel(plan["level"]),
                            action="guardian_evict",
                            tokens_before=tokens_before,
                            tokens_after=self._total_tokens,
                            messages_removed=max(0, len(self.history) - keep_recent),
                            reason=f"Guardian {plan['level']} pressure trigger",
                        ))
                        return
            except Exception as e:
                logger.debug(f"Context guardian evict failed: {e}")
            self._compress_history()

    def _prune_importance_dicts(self) -> None:
        """Prune importance dictionaries to prevent unbounded growth."""
        if len(self.important_messages) > self.MAX_IMPORTANT_MESSAGES:
            valid_indices = set(range(len(self.history)))
            self.important_messages = {
                k: v for k, v in self.important_messages.items()
                if k in valid_indices
            }
            if len(self.important_messages) > self.MAX_IMPORTANT_MESSAGES:
                sorted_items = sorted(
                    self.important_messages.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:self.MAX_IMPORTANT_MESSAGES]
                self.important_messages = dict(sorted_items)

        if len(self.topic_importance) > self.MAX_TOPIC_IMPORTANCE:
            sorted_topics = sorted(
                self.topic_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:self.MAX_TOPIC_IMPORTANCE]
            self.topic_importance = dict(sorted_topics)

        if len(self.long_term_memory) > self.MAX_LONG_TERM_MEMORY:
            self.long_term_memory.sort(key=lambda x: x.get("importance", 0.5), reverse=True)
            self.long_term_memory = self.long_term_memory[:self.MAX_LONG_TERM_MEMORY]

    def clear(self) -> None:
        """Clear conversation history."""
        with self._lock:
            self.history.clear()
            self.important_messages.clear()
            self.topic_importance.clear()
            self.long_term_memory.clear()

    def compact(self, keep_recent: int = 10) -> dict[str, Any]:
        """Model-triggered compaction (Claude Code /compact semantics).

        Replaces older messages with a structured summary block while
        keeping the most recent ``keep_recent`` messages verbatim. The
        summary is also stored in long-term memory so the compressed
        content remains retrievable.

        Returns a stats dict (empty ``compacted: False`` when there was
        nothing worth compacting).
        """
        with self._lock:
            total = len(self.history)
            if total <= keep_recent + 4:
                return {"compacted": False, "reason": "history already small",
                        "messages": total}
            older = self.history[:-keep_recent]
            recent = self.history[-keep_recent:]
            tokens_before = self._total_tokens

        topics: list[str] = []
        try:
            from core.conversation_summarizer import get_conversation_summarizer
            summary = get_conversation_summarizer().summarize_messages(older)
            summary_text = summary.summary_text
            topics = list(getattr(summary, "topics", []) or [])
        except Exception as e:
            logger.warning(f"Compaction summarizer failed, using fallback: {e}")
            summary_text = self._fallback_compact_text(older)

        summary_message = {
            "role": "assistant",
            "content": f"[Conversation compacted — {len(older)} earlier messages]\n{summary_text}",
            "timestamp": datetime.now().isoformat(),
        }
        summary_tokens = self._estimate_tokens(summary_message["content"])
        summary_message["token_count"] = summary_tokens

        with self._lock:
            self.history = [summary_message] + recent
            self.important_messages = {}
            self._recalculate_total_tokens()
            tokens_after = self._total_tokens

        try:
            self.add_to_long_term(summary_text, topics, importance=0.8)
        except Exception as e:
            logger.debug(f"Compaction long-term store failed: {e}")

        logger.info(
            f"Compacted conversation: {total} -> {len(recent) + 1} messages "
            f"({tokens_before} -> {tokens_after} tokens)"
        )
        return {
            "compacted": True,
            "messages_removed": len(older),
            "messages_kept": keep_recent,
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
            "summary": summary_text[:2000],
        }

    def _fallback_compact_text(self, messages: list[dict]) -> str:
        """Heuristic summary when the summarizer is unavailable."""
        lines = []
        for msg in messages[-20:]:
            role = msg.get("role", "unknown")
            content = (msg.get("content") or "")[:200]
            if role == "user":
                lines.append(f"User: {content}")
            elif role == "assistant":
                lines.append(f"Assistant: {content}")
            elif role == "tool":
                lines.append(f"Tool {msg.get('name', '?')}: {content[:100]}")
        return "\n".join(lines) if lines else "Earlier conversation omitted."

    MAX_CONTEXT_KEYS = 200

    def set_context(self, key: str, value: Any) -> None:
        """Set a context value (bounded to prevent unbounded growth)."""
        with self._lock:
            if len(self.context) >= self.MAX_CONTEXT_KEYS and key not in self.context:
                oldest_key = next(iter(self.context))
                del self.context[oldest_key]
            self.context[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        """Get a context value."""
        with self._lock:
            return self.context.get(key, default)

    def add_to_long_term(self, summary: str, topics: list[str], importance: float = 0.8) -> None:
        """Add a summary to long-term memory.

        Args:
            summary: Summary of the conversation/topic
            topics: List of related topics
            importance: Importance score
        """
        with self._lock:
            self.long_term_memory.append({
                "summary": summary,
                "topics": topics,
                "importance": importance,
                "timestamp": datetime.now().isoformat(),
            })

            if len(self.long_term_memory) > self.MAX_LONG_TERM_MEMORY:
                self.long_term_memory.sort(key=lambda x: x["importance"], reverse=True)
                self.long_term_memory = self.long_term_memory[:self.MAX_LONG_TERM_MEMORY]

    def get_relevant_long_term(self, topics: list[str], limit: int = 5) -> list[dict]:
        """Get relevant long-term memories for given topics.

        Args:
            topics: Topics to search for
            limit: Maximum number of memories to return

        Returns:
            List of relevant long-term memories
        """
        with self._lock:
            relevant = []
            for memory in self.long_term_memory:
                overlap = len(set(memory["topics"]) & set(topics))
                if overlap > 0:
                    relevant.append({
                        **memory,
                        "relevance": overlap / max(len(topics), 1),
                    })

            relevant.sort(key=lambda x: x["relevance"], reverse=True)
            return relevant[:limit]

    async def save(self, name: str = "conversation") -> bool:
        """Save conversation to disk asynchronously.
        
        Args:
            name: Name for the save file (sanitized to prevent path traversal)
            
        Returns:
            True if saved successfully
        """
        if self.storage_path is None:
            return False

        try:
            import re
            safe_name = re.sub(r'[^\w\-.]', '_', name)
            if safe_name != name:
                logger.error(f"Path traversal blocked in save: suspicious characters in name {name!r}")
                try:
                    from core.security_audit import get_security_audit_logger, SecurityEvent
                    get_security_audit_logger().log_event(SecurityEvent.PATH_TRAVERSAL_BLOCKED, {"source": "save", "name": name}, severity="critical")
                except Exception as e:
                    logger.warning(f"Security audit logging failed: {e}")
                return False

            self.storage_path.mkdir(parents=True, exist_ok=True)
            file_path = (self.storage_path / name).with_suffix(".json")
            file_path = file_path.resolve()
            if not str(file_path).startswith(str(self.storage_path.resolve())):
                logger.error(f"Path traversal blocked in save: {name!r}")
                return False

            with self._lock:
                data = {
                    "history": self.history.copy(),
                    "context": self.context.copy(),
                    "important_messages": dict(self.important_messages),
                    "long_term_memory": self.long_term_memory.copy(),
                    "version": 1,
                }

            loop = asyncio.get_running_loop()
            try:
                await loop.run_in_executor(
                    None,
                    self._write_file,
                    file_path,
                    data
                )
            except (asyncio.CancelledError, RuntimeError):
                self._write_file(file_path, data)
            return True
        except Exception as e:
            logger.error(f"Failed to save conversation: {e}")
            return False

    def _write_file(self, file_path: Path, data: dict) -> None:
        """Synchronous file write (called from executor)."""
        import os

        # Write to temp file first, then atomic rename
        temp_path = file_path.with_suffix('.tmp')
        with open(temp_path, "w", encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        os.replace(temp_path, file_path)

    async def load(self, name: str = "conversation") -> bool:
        """Load conversation from disk asynchronously.
        
        Args:
            name: Name of the save file (sanitized to prevent path traversal)
            
        Returns:
            True if loaded successfully
        """
        if self.storage_path is None:
            return False

        import re
        safe_name = re.sub(r'[^\w\-.]', '_', name)
        if safe_name != name:
            logger.error(f"Path traversal blocked in load: suspicious characters in name {name!r}")
            return False

        file_path = (self.storage_path / name).with_suffix(".json")
        file_path = file_path.resolve()
        if not str(file_path).startswith(str(self.storage_path.resolve())):
            logger.error(f"Path traversal blocked in load: {name!r}")
            return False

        if not file_path.exists():
            return False

        try:
            loop = asyncio.get_running_loop()
            data = await loop.run_in_executor(
                None,
                self._read_file,
                file_path
            )

            with self._lock:
                raw_history = data.get("history", [])
                self.history = []
                for msg in raw_history:
                    if not isinstance(msg, dict):
                        continue
                    if "role" not in msg or "content" not in msg:
                        continue
                    if msg["role"] not in ("user", "assistant", "tool", "system"):
                        continue
                    if not isinstance(msg["content"], str):
                        msg["content"] = str(msg["content"])
                    self.history.append(msg)
                self.context = data.get("context", {})
                if len(self.context) > self.MAX_CONTEXT_KEYS:
                    keys = list(self.context.keys())
                    for k in keys[:-self.MAX_CONTEXT_KEYS]:
                        del self.context[k]
                # Safe conversion with error handling for corrupted data
                try:
                    self.important_messages = {
                        int(k): v for k, v in data.get("important_messages", {}).items()
                    }
                except (ValueError, TypeError) as e:
                    logger.warning(f"Corrupted important_messages data, resetting: {e}")
                    self.important_messages = {}
                self.long_term_memory = [
                    m for m in data.get("long_term_memory", [])
                    if isinstance(m, dict) and "summary" in m
                ]
                self._recalculate_total_tokens()
            return True
        except Exception as e:
            logger.error(f"Failed to load conversation: {e}")
            return False

    def _read_file(self, file_path: Path) -> dict:
        """Synchronous file read (called from executor)."""
        with open(file_path, encoding='utf-8') as f:
            return json.load(f)

    def get_summary(self) -> str:
        """Get a summary of the conversation."""
        with self._lock:
            turns = len([m for m in self.history if m["role"] == "user"])
            important = len(self.important_messages)
            long_term = len(self.long_term_memory)
            return f"Conversation with {turns} turns, {important} marked important, {long_term} long-term memories"

    MEMORY_PRESSURE_WARNING = 0.8
    MEMORY_PRESSURE_CRITICAL = 0.92

    def get_context_stats(self) -> dict[str, Any]:
        """Get statistics about the context."""
        with self._lock:
            return {
                "total_messages": len(self.history),
                "max_turns": self.max_turns,
                "important_messages": len(self.important_messages),
                "long_term_memories": len(self.long_term_memory),
                "strategy": self.strategy,
                "estimated_tokens": self._total_tokens,
                "token_budget": self.max_tokens,
                "pressure": round(self._total_tokens / self.max_tokens, 3) if self.max_tokens > 0 else 0,
            }

    def get_memory_pressure(self) -> str:
        """Get current memory pressure level: 'normal', 'warning', or 'critical'."""
        with self._lock:
            ratio = self._total_tokens / self.max_tokens if self.max_tokens > 0 else 0
        if ratio >= self.MEMORY_PRESSURE_CRITICAL:
            return "critical"
        elif ratio >= self.MEMORY_PRESSURE_WARNING:
            return "warning"
        return "normal"
    
    def _build_context_pack(
        self,
        query: str = "",
        max_tokens: int | None = None,
        include_episodic: bool = True,
        include_semantic: bool = True,
        working_ratio: float = 0.4,
        session_ratio: float = 0.3,
        episodic_ratio: float = 0.2,
        semantic_ratio: float = 0.1,
        max_important: int | None = None,
    ) -> ContextPack:
        """Core context pack builder used by both sync and async methods.

        Args:
            query: The current query for context retrieval
            max_tokens: Maximum tokens for the context pack
            include_episodic: Whether to include vector memory results
            include_semantic: Whether to include knowledge graph results
            working_ratio: Fraction of budget for working memory
            session_ratio: Fraction of budget for session/important memory
            episodic_ratio: Fraction of budget for episodic memory
            semantic_ratio: Fraction of budget for semantic memory
            max_important: Max important messages to include (None = all)

        Returns:
            ContextPack with layered context
        """
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        pack = ContextPack()

        working_budget = int(max_tokens * working_ratio)
        session_budget = int(max_tokens * session_ratio)
        episodic_budget = int(max_tokens * episodic_ratio)
        semantic_budget = int(max_tokens * semantic_ratio)

        # Iterate in REVERSE (most recent first) so recent context takes priority
        for msg in reversed(self.history):
            if msg["role"] in ("user", "assistant"):
                tokens = self._estimate_tokens(msg.get("content", ""))
                if pack.total_tokens + tokens <= working_budget:
                    pack.working.append({
                        "role": msg["role"],
                        "content": msg["content"],
                    })
                    pack.total_tokens += tokens
        pack.working.reverse()  # Restore chronological order

        important_indices = sorted(
            self.important_messages.keys(),
            key=lambda x: self.important_messages[x],
            reverse=True,
        )
        if max_important is not None:
            important_indices = important_indices[:max_important]

        session_tokens = 0
        for idx in important_indices:
            if idx < len(self.history):
                msg = self.history[idx]
                tokens = self._estimate_tokens(msg.get("content", ""))
                if session_tokens + tokens <= session_budget:
                    pack.session.append({
                        "role": msg["role"],
                        "content": msg["content"],
                        "importance": self.important_messages[idx],
                    })
                    session_tokens += tokens

        pack.total_tokens += session_tokens

        if include_episodic and self._vector_memory and pack.total_tokens < max_tokens:
            try:
                from core.vector_memory import get_vector_memory_context
                vm_ctx = get_vector_memory_context()
                current_valence = 0.0
                current_arousal = 0.5
                try:
                    from core.interoception import get_interoception_engine
                    ie = get_interoception_engine()
                    feeling = ie.get_state().get("current_feeling", {})
                    current_valence = feeling.get("valence", 0.0)
                    current_arousal = feeling.get("arousal", 0.5)
                except Exception:
                    pass
                episodic_context = vm_ctx.get_context_for_query(
                    query,
                    max_items=5,
                    max_tokens=episodic_budget,
                    current_valence=current_valence,
                    current_arousal=current_arousal,
                )
                pack.episodic.extend(episodic_context)
                pack.total_tokens += sum(
                    len(m.get("content", "").split()) * 2
                    for m in episodic_context
                )
            except Exception as e:
                logger.debug(f"Episodic context failed: {e}")

        if include_semantic and pack.total_tokens < max_tokens:
            try:
                from core.knowledge_graph import get_knowledge_graph_context
                kg_ctx = get_knowledge_graph_context()
                semantic_context = kg_ctx.get_context_for_query(
                    query,
                    max_items=5,
                    max_tokens=semantic_budget,
                )
                pack.semantic.extend(semantic_context)
                pack.total_tokens += sum(
                    len(m.get("content", "").split()) * 2
                    for m in semantic_context
                )
            except Exception as e:
                logger.debug(f"Semantic context failed: {e}")

        return pack

    def _build_context_pack_from_snapshot(
        self,
        history_snapshot: list[dict],
        important_snapshot: dict[int, float],
        query: str,
        max_tokens: int,
        include_episodic: bool = True,
        include_semantic: bool = True,
        working_ratio: float = 0.4,
        session_ratio: float = 0.3,
        episodic_ratio: float = 0.2,
        semantic_ratio: float = 0.1,
        max_important: int | None = None,
    ) -> ContextPack:
        """Build context pack from a snapshot of history (no lock needed).

        This is the lock-free version that operates on snapshot data,
        allowing I/O operations (vector memory, knowledge graph) without
        holding the conversation memory lock.
        """
        pack = ContextPack()

        working_budget = int(max_tokens * working_ratio)
        session_budget = int(max_tokens * session_ratio)
        episodic_budget = int(max_tokens * episodic_ratio)
        semantic_budget = int(max_tokens * semantic_ratio)

        for msg in reversed(history_snapshot):
            if msg["role"] in ("user", "assistant"):
                tokens = len(msg.get("content", "")) // CHARS_PER_TOKEN
                if pack.total_tokens + tokens <= working_budget:
                    pack.working.append({
                        "role": msg["role"],
                        "content": msg["content"],
                    })
                    pack.total_tokens += tokens
        pack.working.reverse()

        important_indices = sorted(
            important_snapshot.keys(),
            key=lambda x: important_snapshot[x],
            reverse=True,
        )
        if max_important is not None:
            important_indices = important_indices[:max_important]

        session_tokens = 0
        for idx in important_indices:
            if idx < len(history_snapshot):
                msg = history_snapshot[idx]
                tokens = len(msg.get("content", "")) // CHARS_PER_TOKEN
                if session_tokens + tokens <= session_budget:
                    pack.session.append({
                        "role": msg["role"],
                        "content": msg["content"],
                        "importance": important_snapshot[idx],
                    })
                    session_tokens += tokens

        pack.total_tokens += session_tokens

        if include_episodic and self._vector_memory and pack.total_tokens < max_tokens:
            try:
                from core.vector_memory import get_vector_memory_context
                vm_ctx = get_vector_memory_context()
                current_valence = 0.0
                current_arousal = 0.5
                try:
                    from core.interoception import get_interoception_engine
                    ie = get_interoception_engine()
                    feeling = ie.get_state().get("current_feeling", {})
                    current_valence = feeling.get("valence", 0.0)
                    current_arousal = feeling.get("arousal", 0.5)
                except Exception:
                    pass
                episodic_context = vm_ctx.get_context_for_query(
                    query,
                    max_items=5,
                    max_tokens=episodic_budget,
                    current_valence=current_valence,
                    current_arousal=current_arousal,
                )
                pack.episodic.extend(episodic_context)
                pack.total_tokens += sum(
                    len(m.get("content", "").split()) * 2
                    for m in episodic_context
                )
            except Exception as e:
                logger.debug(f"Episodic context failed: {e}")

        if include_semantic and pack.total_tokens < max_tokens:
            try:
                from core.knowledge_graph import get_knowledge_graph_context
                kg_ctx = get_knowledge_graph_context()
                semantic_context = kg_ctx.get_context_for_query(
                    query,
                    max_items=5,
                    max_tokens=semantic_budget,
                )
                pack.semantic.extend(semantic_context)
                pack.total_tokens += sum(
                    len(m.get("content", "").split()) * 2
                    for m in semantic_context
                )
            except Exception as e:
                logger.debug(f"Semantic context failed: {e}")

        return pack

    async def assemble_context_pack(
        self,
        query: str,
        max_tokens: int | None = None,
        include_episodic: bool = True,
        include_semantic: bool = True,
    ) -> ContextPack:
        """Assemble context from all memory layers (async).

        Uses full budget with episodic and semantic memory.
        Thread-safe: snapshots data under lock, then builds pack outside lock
        to avoid holding the lock across I/O (vector memory, knowledge graph).
        """
        with self._lock:
            history_snapshot = [
                {"role": m["role"], "content": m.get("content", ""), "importance": m.get("importance", 0.5)}
                for m in self.history
            ]
            important_snapshot = dict(self.important_messages)
            max_tokens_val = max_tokens if max_tokens is not None else self.max_tokens

        pack = self._build_context_pack_from_snapshot(
            history_snapshot=history_snapshot,
            important_snapshot=important_snapshot,
            query=query,
            max_tokens=max_tokens_val,
            include_episodic=include_episodic,
            include_semantic=include_semantic,
            working_ratio=0.4,
            session_ratio=0.3,
            episodic_ratio=0.2,
            semantic_ratio=0.1,
        )
        return pack
    
    def get_context_pack_for_llm(
        self,
        query: str = "",
        max_tokens: int | None = None,
    ) -> list[dict]:
        """Get context formatted for LLM API.

        Synchronous wrapper using _build_context_pack.
        Uses simpler budget (60/40) without episodic/semantic layers.
        Thread-safe with lock protection.
        """
        with self._lock:
            pack = self._build_context_pack(
                query=query,
                max_tokens=max_tokens,
                include_episodic=False,
                include_semantic=False,
                working_ratio=0.6,
                session_ratio=0.4,
                episodic_ratio=0.0,
                semantic_ratio=0.0,
                max_important=10,
            )
            return pack.to_messages()
    
    async def store_insight(
        self,
        content: str,
        importance: float = 0.85,
        concepts: list[str] | None = None,
    ) -> str | None:
        """Store an insight in both long-term and episodic memory.
        
        Args:
            content: The insight content
            importance: Importance score (0-1)
            concepts: Related concepts
            
        Returns:
            Entry ID if successful, None otherwise
        """
        self.add_to_long_term(content, concepts or [], importance)
        
        if self._vector_memory:
            try:
                from core.vector_memory import get_vector_memory_context
                vm_ctx = get_vector_memory_context()
                entry_id = vm_ctx.store_with_context(
                    content,
                    importance=importance,
                    concepts=concepts,
                    metadata={"type": "insight"},
                )
                return entry_id
            except Exception as e:
                logger.debug(f"Failed to store insight in vector memory: {e}")
        
        return None
    
    def get_enhanced_stats(self) -> dict[str, Any]:
        """Get enhanced memory statistics."""
        stats = self.get_context_stats()
        stats["cache_stats"] = getattr(self, "_context_cache", ContextCache()).stats() if hasattr(self, "_context_cache") else {}
        stats["vector_memory_available"] = self._vector_memory is not None
        
        if self._vector_memory:
            try:
                from core.vector_memory import get_vector_memory
                vm = get_vector_memory()
                stats["vector_memory_stats"] = vm.stats()
            except Exception as e:
                logger.warning(f"Failed to get vector memory stats: {e}")
        
        return stats


_memory: ConversationMemory | None = None
_memory_lock = threading.Lock()


def get_conversation_memory(
    max_turns: int = 100,
    max_tokens: int | None = None,
    storage_path: Path | None = None,
    use_vector_memory: bool = True,
) -> ConversationMemory:
    """Get global conversation memory instance (thread-safe)."""
    global _memory
    if _memory is None:
        with _memory_lock:
            if _memory is None:
                _memory = ConversationMemory(
                    max_turns=max_turns,
                    max_tokens=max_tokens,
                    storage_path=storage_path,
                    use_vector_memory=use_vector_memory,
                )
    return _memory
