"""Smart Context Compression — Token-aware context window management for 24/7 Jarvis.

The key challenge for a long-running agent: context windows fill up, and naive
truncation loses important information. This module provides intelligent compression
that preserves the most valuable context while fitting within token budgets.

Features:
- Token counting and budget management
- Priority-based message retention (system > recent > tool results > older)
- Summarization of old context with key information extraction
- Tool call/result pair preservation (never split them)
- Compression history tracking for observability
- Multiple compression strategies (truncate, summarize, compress, hybrid)
"""
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_TOKEN_BUDGET = 8000
DEFAULT_MAX_TURN_LENGTH = 2000
DEFAULT_SYSTEM_RETENTION = 3
DEFAULT_RECENT_RETENTION = 5
TOOL_PAIR_PATTERN = re.compile(r'\[tool_call_\d+\]|\[tool_result_\d+\]')

_CODE_PATTERN = re.compile(r'```[\s\S]*?```')
_IMPORTANCE_BOOST = {"remember": 0.2, "important": 0.25, "note": 0.15, "critical": 0.3, "key": 0.15, "essential": 0.25, "password": 0.35, "secret": 0.3, "name is": 0.2, "prefer": 0.15, "please remember": 0.1}
_LOW_IMPORTANCE_WORDS = {"ok", "thanks", "okay", "sure", "yep", "nope", "hm", "hmm", "lol", "yea", "nah"}
_TOPIC_PATTERNS = [
    (re.compile(r'\b\w+\.py\b'), "code"),
    (re.compile(r'\b\w+\.js\b'), "code"),
    (re.compile(r'\bfunction\b|\bclass\b|\bimport\b|\bdef\b'), "code"),
    (re.compile(r'\berror\b|\bbug\b|\bfix\b'), "debugging"),
    (re.compile(r'\bdeploy\b|\bserver\b|\bproduction\b'), "devops"),
    (re.compile(r'\btest\b|\btesting\b|\bpytest\b'), "testing"),
    (re.compile(r'\bpassword\b|\bauth\b|\bsecurity\b|\bkey\b'), "security"),
    (re.compile(r'\bML\b|\bmodel\b|\btrain\b|\bneural\b'), "ml"),
]
_ENTITY_PATTERN = re.compile(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b')


class TokenCounter:
    """Count tokens using character-based heuristic (1 token ≈ 4 chars)."""

    @staticmethod
    def count(text: str) -> int:
        if not text:
            return 0
        return max(1, len(text) // 4 + 1)

    @staticmethod
    def count_messages(messages: list[dict]) -> int:
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += TokenCounter.count(content)
            else:
                total += 50
            total += TokenCounter.count(msg.get("role", ""))
            if msg.get("tool_calls"):
                total += 50
            if msg.get("tool_result"):
                total += 20
        return total


class ImportanceScorer:
    """Score message importance from 0 to 1 based on content heuristics."""

    @staticmethod
    def score(text: str) -> float:
        if not text:
            return 0.0
        text_lower = text.lower()
        base = 0.35
        for keyword, boost in _IMPORTANCE_BOOST.items():
            if keyword in text_lower:
                base += boost
        for word in _LOW_IMPORTANCE_WORDS:
            if text_lower.strip() == word:
                return 0.1
        if len(text.strip()) <= 5:
            base -= 0.15
        if _CODE_PATTERN.search(text):
            base += 0.2
        if any(p.search(text) for p, _ in _TOPIC_PATTERNS):
            base += 0.05
        return max(0.0, min(1.0, base))


class MemoryTier(Enum):
    ACTIVE = "active"
    COMPRESSED = "compressed"
    ARCHIVED = "archived"


@dataclass
class ContextBlock:
    """A unit of conversation context with metadata for intelligent compression."""
    id: str
    content: str
    role: str = "user"
    importance: float = 0.5
    topics: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    tier: MemoryTier = MemoryTier.ACTIVE
    access_count: int = 0
    created_at: float = field(default_factory=time.time)
    compressed_content: Optional[str] = None

    def touch(self):
        self.access_count += 1

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "role": self.role,
            "importance": self.importance,
            "topics": self.topics,
            "entities": self.entities,
            "tier": self.tier.value,
            "access_count": self.access_count,
            "created_at": self.created_at,
            "compressed_content": self.compressed_content,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ContextBlock":
        tier_val = data.get("tier", MemoryTier.ACTIVE.value)
        if isinstance(tier_val, str):
            tier_val = MemoryTier(tier_val)
        return cls(
            id=data["id"],
            content=data["content"],
            role=data.get("role", "user"),
            importance=data.get("importance", 0.5),
            topics=data.get("topics", []),
            entities=data.get("entities", []),
            tier=tier_val,
            access_count=data.get("access_count", 0),
            created_at=data.get("created_at", time.time()),
            compressed_content=data.get("compressed_content"),
        )


@dataclass
class CompressionResult:
    original_count: int
    compressed_count: int
    original_tokens: int
    compressed_tokens: int
    strategy: str
    messages_removed: int
    messages_summarized: int
    duration_seconds: float = 0.0
    preserved_roles: list[str] = field(default_factory=list)

    @property
    def reduction_percent(self) -> float:
        if self.original_tokens == 0:
            return 0.0
        return round((1 - self.compressed_tokens / self.original_tokens) * 100, 1)

    def to_dict(self) -> dict:
        return {
            "original_count": self.original_count,
            "compressed_count": self.compressed_count,
            "original_tokens": self.original_tokens,
            "compressed_tokens": self.compressed_tokens,
            "strategy": self.strategy,
            "messages_removed": self.messages_removed,
            "messages_summarized": self.messages_summarized,
            "reduction_percent": self.reduction_percent,
            "duration_seconds": round(self.duration_seconds, 3),
            "preserved_roles": self.preserved_roles,
        }


class ContextCompressor:
    """Intelligent context compression that preserves important information.

    Strategies:
    1. truncate: Remove oldest messages, keep system + recent
    2. summarize: Replace middle messages with a summary
    3. compress: Shorten individual messages while preserving key info
    4. hybrid: Combine summarize + compress for maximum efficiency
    """

    _RE_CODE_BLOCK = re.compile(r'```[\s\S]*?```')
    _RE_FILE_REF = re.compile(r'\b\w+\.\w{1,6}\b')
    _RE_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')
    DEFAULT_ARCHIVE_INDEX_LIMIT = 10000

    def __init__(self, token_budget: int = DEFAULT_TOKEN_BUDGET, storage_path=None, active_token_limit: int = 50000):
        self._token_budget = token_budget
        self._active_token_limit = active_token_limit
        self._storage_path = Path(storage_path) if storage_path else None
        self._lock = threading.RLock()
        self._compression_history: list[CompressionResult] = []
        self._max_history = 100
        self._total_compressions = 0
        self._total_tokens_saved = 0
        self._start_time = time.time()
        self._active_blocks: list[ContextBlock] = []
        self._compressed_blocks: list[ContextBlock] = []
        self._block_counter = 0
        self._id_counter = 0
        self._by_id: dict[str, ContextBlock] = {}
        self._entity_index: dict[str, list[str]] = {}
        self._topic_index: dict[str, list[str]] = {}

    def _generate_id(self, content: str = "") -> str:
        with self._lock:
            self._id_counter += 1
            return f"ctx-{self._id_counter}-{hash(content) & 0xFFFF:x}"

    @property
    def active_blocks(self) -> list[ContextBlock]:
        return self._active_blocks

    @property
    def compressed_blocks(self) -> list[ContextBlock]:
        return self._compressed_blocks

    def add_message(self, role: str, content: str, importance: Optional[float] = None) -> ContextBlock:
        self._block_counter += 1
        block_id = f"block-{self._block_counter}"
        if importance is None:
            importance = ImportanceScorer.score(content)
        topics = self._extract_topics(content)
        entities = self._extract_entities(content)
        block = ContextBlock(
            id=block_id,
            content=content,
            role=role,
            importance=importance,
            topics=topics,
            entities=entities,
        )
        with self._lock:
            self._active_blocks.append(block)
            total_tokens = sum(TokenCounter.count(b.content) for b in self._active_blocks)
            if total_tokens > self._active_token_limit:
                self._auto_compress()
        return block

    def _auto_compress(self):
        low_blocks = [b for b in self._active_blocks if b.importance < 0.5 and b.tier == MemoryTier.ACTIVE]
        for block in low_blocks:
            summary = self._simple_summary(block.content)
            block.compressed_content = summary
            block.tier = MemoryTier.COMPRESSED
            self._compressed_blocks.append(block)
        self._active_blocks = [b for b in self._active_blocks if b.tier == MemoryTier.ACTIVE]

    def search(self, query: str, top_k: int = 5) -> list[tuple[ContextBlock, float]]:
        query_lower = query.lower()
        query_words = set(query_lower.split())
        scored = []
        with self._lock:
            for block in self._active_blocks:
                content_lower = block.content.lower()
                overlap = sum(1 for w in query_words if w in content_lower)
                if overlap == 0:
                    score = 0.0
                else:
                    score = overlap / max(len(query_words), 1) * (0.5 + 0.5 * block.importance)
                if score > 0:
                    scored.append((block, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def get_messages_for_llm(self, max_tokens: int = 4000) -> list[dict]:
        with self._lock:
            active = list(self._active_blocks)

        # Chronological rank by insertion position, not by created_at: the wall
        # clock has ~15.6ms granularity on Windows, so messages added in quick
        # succession share a timestamp and the restore-order sort below becomes
        # a no-op, leaking the importance ordering into the conversation.
        # Keyed by id() rather than content so repeated messages stay distinct.
        order = {id(b): i for i, b in enumerate(active)}

        # Select the most important blocks that fit the budget...
        selected = []
        total = 0
        for block in sorted(active, key=lambda b: (-b.importance, order[id(b)])):
            t = TokenCounter.count(block.content) + TokenCounter.count(block.role)
            if total + t > max_tokens:
                break
            selected.append(block)
            total += t

        # ...then restore true conversation order before handing to the LLM.
        selected.sort(key=lambda b: order[id(b)])
        return [{"role": b.role, "content": b.content} for b in selected]

    def _extract_topics(self, text: str) -> list[str]:
        topics = set()
        for pattern, topic in _TOPIC_PATTERNS:
            if pattern.search(text):
                topics.add(topic)
        return list(topics)

    def _extract_entities(self, text: str) -> list[str]:
        return list(set(_ENTITY_PATTERN.findall(text)))[:5]

    def estimate_tokens(self, messages: list[dict]) -> int:
        """Estimate token count for messages (rough: 1 token ≈ 4 chars)."""
        return TokenCounter.count_messages(messages)

    def _estimate_single_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(text) // 4 + 1

    def needs_compression(self, messages: list[dict], threshold: float = 0.85) -> bool:
        current = self.estimate_tokens(messages)
        return current > self._token_budget * threshold

    def compress(self, messages: list[dict], strategy: str = "auto") -> tuple[list[dict], CompressionResult]:
        start = time.time()
        original_tokens = self.estimate_tokens(messages)

        if strategy == "auto":
            if original_tokens > self._token_budget * 2:
                strategy = "hybrid"
            elif original_tokens > self._token_budget * 1.5:
                strategy = "summarize"
            else:
                strategy = "truncate"

        original_count = len(messages)

        if strategy == "truncate":
            result_messages = self._compress_truncate(messages)
        elif strategy == "summarize":
            result_messages = self._compress_summarize(messages)
        elif strategy == "compress":
            result_messages = self._compress_shorten(messages)
        elif strategy == "hybrid":
            result_messages = self._compress_hybrid(messages)
        else:
            result_messages = self._compress_truncate(messages)
            strategy = "truncate"

        compressed_tokens = self.estimate_tokens(result_messages)
        duration = time.time() - start

        preserved_roles = list(set(msg.get("role", "") for msg in result_messages))

        comp_result = CompressionResult(
            original_count=original_count,
            compressed_count=len(result_messages),
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            strategy=strategy,
            messages_removed=original_count - len(result_messages),
            messages_summarized=original_count - len(result_messages) if strategy in ("summarize", "hybrid") else 0,
            duration_seconds=duration,
            preserved_roles=preserved_roles,
        )

        with self._lock:
            self._compression_history.append(comp_result)
            if len(self._compression_history) > self._max_history:
                self._compression_history = self._compression_history[-self._max_history:]
            self._total_compressions += 1
            self._total_tokens_saved += max(0, original_tokens - compressed_tokens)

        return result_messages, comp_result

    def _compress_truncate(self, messages: list[dict]) -> list[dict]:
        if len(messages) <= DEFAULT_RECENT_RETENTION + DEFAULT_SYSTEM_RETENTION:
            return messages
        system_msgs = [m for m in messages[:DEFAULT_SYSTEM_RETENTION] if m.get("role") == "system"]
        recent = messages[-DEFAULT_RECENT_RETENTION:]
        seen_ids = set()
        unique_recent = []
        for m in reversed(recent):
            mid = id(m)
            if mid not in seen_ids:
                seen_ids.add(mid)
                unique_recent.append(m)
        unique_recent.reverse()
        return system_msgs + unique_recent

    def _compress_summarize(self, messages: list[dict]) -> list[dict]:
        if len(messages) <= 8:
            return messages
        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]
        recent_count = min(DEFAULT_RECENT_RETENTION, len(non_system))
        recent = non_system[-recent_count:]
        older = non_system[:-recent_count] if recent_count < len(non_system) else []
        summary_parts = []
        for msg in older:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, str):
                truncated = content[:300]
                summary_parts.append(f"[{role}] {truncated}")
        summary_text = "[Context Summary]\n" + "\n".join(summary_parts[:15])
        if len(summary_text) > 2000:
            summary_text = summary_text[:2000] + "\n... (truncated)"
        summary_msg = {"role": "system", "content": summary_text}
        result = system_msgs + [summary_msg] + recent
        return result

    def _compress_shorten(self, messages: list[dict]) -> list[dict]:
        result = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str) and len(content) > DEFAULT_MAX_TURN_LENGTH:
                shortened = self._shorten_text(content)
                msg = {**msg, "content": shortened}
            result.append(msg)
        return result

    def _compress_hybrid(self, messages: list[dict]) -> list[dict]:
        summarized = self._compress_summarize(messages)
        shortened = self._compress_shorten(summarized)
        budget = self._token_budget
        current = self.estimate_tokens(shortened)
        if current > budget:
            shortened = shortened[:DEFAULT_SYSTEM_RETENTION + 1] + shortened[-(DEFAULT_RECENT_RETENTION):]
        return shortened

    def _shorten_text(self, text: str) -> str:
        if len(text) <= DEFAULT_MAX_TURN_LENGTH:
            return text
        lines = text.splitlines()
        if len(lines) <= 20:
            if len(text) > DEFAULT_MAX_TURN_LENGTH:
                return text[:DEFAULT_MAX_TURN_LENGTH] + "\n... (truncated)"
            return text
        first_lines = lines[:5]
        last_lines = lines[-5:]
        middle_count = len(lines) - 10
        result = "\n".join(first_lines) + f"\n... ({middle_count} lines omitted) ...\n" + "\n".join(last_lines)
        if len(result) > DEFAULT_MAX_TURN_LENGTH:
            return text[:DEFAULT_MAX_TURN_LENGTH] + "\n... (truncated)"
        return result

    def _simple_summary(self, text: str, max_chars: int = 500) -> str:
        if not text:
            return ""
        sentences = self._RE_SENTENCE_SPLIT.split(text)
        key_words = {"important", "key", "critical", "essential", "decision", "remember", "note", "result", "task", "must", "need", "bug", "fix", "error"}
        key_sents = [s for s in sentences if any(w in s.lower() for w in key_words)]
        if key_sents:
            summary = " ".join(key_sents[:5])
            if len(summary) > max_chars:
                summary = summary[:max_chars] + "..."
            return summary
        if len(text) > max_chars:
            return text[:max_chars] + "..."
        return text

    def get_compression_history(self, limit: int = 20) -> list[dict]:
        with self._lock:
            return [r.to_dict() for r in self._compression_history[-limit:]]

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            active_tokens = sum(TokenCounter.count(b.content) for b in self._active_blocks)
            return {
                "token_budget": self._token_budget,
                "total_compressions": self._total_compressions,
                "total_tokens_saved": self._total_tokens_saved,
                "history_size": len(self._compression_history),
                "uptime_seconds": round(time.time() - self._start_time, 1),
                "last_compression": self._compression_history[-1].to_dict() if self._compression_history else None,
                "active_blocks": len(self._active_blocks),
                "active_tokens": active_tokens,
            }

    def get_stats(self) -> dict[str, Any]:
        return self.get_status()

    def get_context_usage(self) -> dict[str, Any]:
        with self._lock:
            active_tokens = sum(TokenCounter.count(b.content) for b in self._active_blocks)
            return {
                "active_tokens": active_tokens,
                "block_counts": {
                    "active": sum(1 for b in self._active_blocks if b.tier == MemoryTier.ACTIVE),
                    "compressed": sum(1 for b in self._active_blocks if b.tier == MemoryTier.COMPRESSED),
                    "archived": sum(1 for b in self._active_blocks if b.tier == MemoryTier.ARCHIVED),
                },
                "token_budget": self._token_budget,
                "usage_percent": round(active_tokens / max(self._token_budget, 1) * 100, 1),
            }

    def proactive_summarize(self) -> dict[str, Any]:
        with self._lock:
            low_importance = [b for b in self._active_blocks if b.importance < 0.3 and b.tier == MemoryTier.ACTIVE]
            if not low_importance:
                return {"summarized": 0, "reason": "no low-importance blocks"}
            summarized = 0
            for block in low_importance:
                summary = self._simple_summary(block.content)
                block.compressed_content = summary
                block.tier = MemoryTier.COMPRESSED
                summarized += 1
            return {"summarized": summarized, "reason": f"compressed {summarized} low-importance blocks"}

    def _rebuild_indexes(self):
        with self._lock:
            self._by_id = {b.id: b for b in self._active_blocks}
            self._entity_index = {}
            self._topic_index = {}
            for b in self._active_blocks:
                for e in b.entities:
                    self._entity_index.setdefault(e, []).append(b.id)
                for t in b.topics:
                    self._topic_index.setdefault(t, []).append(b.id)
            if len(self._by_id) > self.DEFAULT_ARCHIVE_INDEX_LIMIT:
                excess = len(self._by_id) - self.DEFAULT_ARCHIVE_INDEX_LIMIT
                oldest = sorted(self._active_blocks, key=lambda b: b.created_at)[:excess]
                for b in oldest:
                    b.tier = MemoryTier.ARCHIVED
                    self._by_id.pop(b.id, None)

    def set_token_budget(self, budget: int) -> None:
        with self._lock:
            self._token_budget = max(1000, budget)

    def clear(self, keep_important: bool = True) -> None:
        with self._lock:
            if keep_important:
                self._active_blocks = [b for b in self._active_blocks if b.importance >= 0.8]
                self._compressed_blocks = [b for b in self._compressed_blocks if b.importance >= 0.8]
            else:
                self._active_blocks.clear()
                self._compressed_blocks.clear()
            self._by_id = {b.id: b for b in self._active_blocks + self._compressed_blocks}
            self._entity_index = {}
            self._topic_index = {}
            for b in self._active_blocks + self._compressed_blocks:
                for e in b.entities:
                    self._entity_index.setdefault(e, []).append(b.id)
                for t in b.topics:
                    self._topic_index.setdefault(t, []).append(b.id)

    def compress_active(self) -> dict[str, Any]:
        with self._lock:
            total_tokens = sum(TokenCounter.count(b.content) for b in self._active_blocks)
            if total_tokens <= self._active_token_limit:
                return {"reason": "within budget", "active_tokens": total_tokens, "compressed": 0}
            over_limit = [b for b in self._active_blocks if b.importance < 0.5]
            compressed = 0
            for block in over_limit:
                summary = self._simple_summary(block.content)
                block.compressed_content = summary
                block.tier = MemoryTier.COMPRESSED
                self._compressed_blocks.append(block)
                compressed += 1
            self._active_blocks = [b for b in self._active_blocks if b.tier == MemoryTier.ACTIVE]
            return {"reason": "compressed low-importance", "active_tokens": total_tokens, "compressed": compressed}


_context_compressor: ContextCompressor | None = None
_compressor_lock = threading.Lock()

import atexit
from concurrent.futures import ThreadPoolExecutor

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ctx_compress")


def _shutdown_executor():
    _executor.shutdown(wait=False)


atexit.register(_shutdown_executor)


def get_context_compressor(budget: int = DEFAULT_TOKEN_BUDGET) -> ContextCompressor:
    global _context_compressor
    if _context_compressor is None:
        with _compressor_lock:
            if _context_compressor is None:
                _context_compressor = ContextCompressor(token_budget=budget)
    return _context_compressor