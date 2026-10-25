"""Conversation Summarizer - Auto-summarization for context management.

When the Context Guardian triggers eviction, the summarizer compresses
old conversation turns into concise summaries before storing in L2,
preserving key information while freeing L1 token budget.

Inspired by Letta's summarize_messages pattern and LangGraph's
auto-summarization checkpointing.
"""
import hashlib
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

_summarizer: Optional["ConversationSummarizer"] = None
_summarizer_lock = threading.Lock()


class SummaryStyle(Enum):
    CONCISE = "concise"
    DETAILED = "detailed"
    ACTION_ITEMS = "action_items"


@dataclass
class Summary:
    id: str
    source_turns: int
    source_messages: int
    original_tokens: int
    summary_text: str
    key_decisions: list[str] = field(default_factory=list)
    key_facts: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class SummarizerStats:
    total_summaries: int = 0
    total_source_messages: int = 0
    total_source_tokens: int = 0
    total_summary_tokens: int = 0
    avg_compression_ratio: float = 0.0


class ConversationSummarizer:
    def __init__(self, max_summaries: int = 200):
        self._summaries: OrderedDict[str, Summary] = OrderedDict()
        self._max_summaries = max_summaries
        self._stats = SummarizerStats()
        self._lock = threading.RLock()

    def summarize_messages(self, messages: list[dict], style: SummaryStyle = SummaryStyle.CONCISE) -> Summary:
        if not messages:
            return self._empty_summary()

        source_tokens = sum(len(m.get("content", "")) // 4 for m in messages)
        
        topics = self._extract_topics(messages)
        key_facts = self._extract_key_facts(messages)
        key_decisions = self._extract_key_decisions(messages)
        tools_used = self._extract_tools_used(messages)
        
        summary_text = self._build_summary_text(messages, topics, key_facts, key_decisions, tools_used, style)
        
        summary_id = f"sum_{hashlib.sha256(summary_text.encode()).hexdigest()[:8]}_{int(time.time()*1000)}"
        
        summary = Summary(
            id=summary_id,
            source_turns=self._count_turns(messages),
            source_messages=len(messages),
            original_tokens=source_tokens,
            summary_text=summary_text,
            key_decisions=key_decisions[:10],
            key_facts=key_facts[:10],
            tools_used=tools_used[:10],
            topics=topics[:5],
        )

        summary_tokens = len(summary_text) // 4
        with self._lock:
            self._stats.total_summaries += 1
            self._stats.total_source_messages += len(messages)
            self._stats.total_source_tokens += source_tokens
            self._stats.total_summary_tokens += summary_tokens
            if source_tokens > 0:
                ratio = summary_tokens / source_tokens
                self._stats.avg_compression_ratio = self._stats.avg_compression_ratio * 0.9 + ratio * 0.1

            self._summaries[summary_id] = summary
            while len(self._summaries) > self._max_summaries:
                self._summaries.popitem(last=False)

        return summary

    def _build_summary_text(self, messages, topics, facts, decisions, tools, style) -> str:
        parts = []
        
        if topics:
            parts.append(f"Topics: {', '.join(topics[:5])}")
        
        if style == SummaryStyle.ACTION_ITEMS:
            if decisions:
                parts.append("Decisions: " + "; ".join(decisions[:5]))
            if tools:
                parts.append(f"Tools used: {', '.join(tools[:8])}")
            if facts:
                parts.append("Key facts: " + "; ".join(facts[:5]))
        elif style == SummaryStyle.DETAILED:
            for msg in messages:
                role = msg.get("role", "?")
                content = msg.get("content", "")[:200]
                if content:
                    parts.append(f"[{role}] {content}")
        else:
            user_msgs = [m.get("content", "")[:150] for m in messages if m.get("role") == "user" and m.get("content")]
            assistant_msgs = [m.get("content", "")[:150] for m in messages if m.get("role") == "assistant" and m.get("content")]
            
            if user_msgs:
                parts.append(f"User asked: {'; '.join(user_msgs[:5])}")
            if assistant_msgs:
                parts.append(f"Assistant covered: {'; '.join(assistant_msgs[:3])}")
            if facts:
                parts.append("Facts: " + "; ".join(facts[:5]))
            if tools:
                parts.append(f"Tools: {', '.join(tools[:8])}")
        
        return "\n".join(parts)

    def _extract_topics(self, messages: list[dict]) -> list[str]:
        keywords = {}
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                      "being", "have", "has", "had", "do", "does", "did", "will",
                      "would", "could", "should", "may", "might", "can", "shall",
                      "i", "you", "he", "she", "it", "we", "they", "me", "him",
                      "her", "us", "them", "my", "your", "his", "its", "our",
                      "their", "this", "that", "these", "those", "what", "which",
                      "who", "whom", "how", "when", "where", "why", "not", "no",
                      "but", "or", "and", "if", "then", "so", "than", "too", "very",
                      "just", "about", "also", "with", "for", "from", "into"}
        
        for msg in messages:
            text = msg.get("content", "").lower()
            for word in text.split():
                w = word.strip(".,!?;:()[]{}\"'-").lower()
                if len(w) > 4 and w not in stop_words:
                    keywords[w] = keywords.get(w, 0) + 1
        
        sorted_kw = sorted(keywords.items(), key=lambda x: x[1], reverse=True)
        return [kw for kw, count in sorted_kw[:10] if count >= 2]

    def _extract_key_facts(self, messages: list[dict]) -> list[str]:
        facts = []
        fact_keywords = ["my name", "i am", "i like", "i prefer", "i want", "i need",
                         "remember", "important", "the answer", "the result", "the solution"]
        for msg in messages:
            content = msg.get("content", "")
            content_lower = content.lower()
            if msg.get("role") == "user" and any(kw in content_lower for kw in fact_keywords):
                facts.append(content[:200])
        return facts[:10]

    def _extract_key_decisions(self, messages: list[dict]) -> list[str]:
        decisions = []
        decision_keywords = ["decided", "chose", "will use", "going to", "let's go",
                             "confirmed", "agreed", "finalize", "selected"]
        for msg in messages:
            content = msg.get("content", "")
            content_lower = content.lower()
            if any(kw in content_lower for kw in decision_keywords):
                decisions.append(content[:200])
        return decisions[:10]

    def _extract_tools_used(self, messages: list[dict]) -> list[str]:
        tools = []
        for msg in messages:
            if msg.get("role") == "tool":
                name = msg.get("name", "")
                if name and name not in tools:
                    tools.append(name)
            elif msg.get("role") == "assistant":
                content = msg.get("content", "")
                if "using tool:" in content.lower():
                    tools.append(content[:80])
        return tools[:10]

    def _count_turns(self, messages: list[dict]) -> int:
        return sum(1 for m in messages if m.get("role") == "user")

    def _empty_summary(self) -> Summary:
        return Summary(
            id="sum_empty",
            source_turns=0,
            source_messages=0,
            original_tokens=0,
            summary_text="",
        )

    def get_summary(self, summary_id: str) -> Summary | None:
        with self._lock:
            return self._summaries.get(summary_id)

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "total_summaries": self._stats.total_summaries,
                "total_source_messages": self._stats.total_source_messages,
                "avg_compression_ratio": round(self._stats.avg_compression_ratio, 3),
                "stored_summaries": len(self._summaries),
            }


def get_conversation_summarizer() -> ConversationSummarizer:
    global _summarizer, _summarizer_lock
    if _summarizer is None:
        with _summarizer_lock:
            if _summarizer is None:
                _summarizer = ConversationSummarizer()
    return _summarizer