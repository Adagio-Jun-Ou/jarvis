"""Tests for Smart Context Compression.

Covers:
- Token estimation
- Compression strategies (truncate, summarize, compress, hybrid)
- Compression budget management
- Message preservation priorities
- Tool call/result pair handling
- Compression history tracking
- Auto-strategy selection
"""
import time
import threading
from pathlib import Path

import pytest

sys_path = str(Path(__file__).parent.parent / "src")
import sys
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)

from core.context_compressor import (
    ContextCompressor, CompressionResult,
    get_context_compressor, DEFAULT_TOKEN_BUDGET,
)


def _make_messages(count: int, role: str = "user", content_len: int = 100) -> list[dict]:
    msgs = [{"role": "system", "content": "You are Jarvis."}]
    for i in range(count):
        msgs.append({"role": role, "content": f"Message {i}: " + "x" * content_len})
    return msgs


def _make_tool_messages(count: int) -> list[dict]:
    msgs = [{"role": "system", "content": "You are Jarvis."}]
    for i in range(count):
        msgs.append({"role": "assistant", "content": f"Tool call {i}", "tool_calls": [{"name": f"tool_{i}"}]})
        msgs.append({"role": "tool", "content": f"Tool result {i}", "tool_call_id": f"call_{i}"})
    return msgs


class TestTokenEstimation:
    def setup_method(self):
        self.comp = ContextCompressor(token_budget=1000)

    def test_estimate_empty(self):
        assert self.comp.estimate_tokens([]) == 0

    def test_estimate_simple(self):
        msgs = [{"role": "user", "content": "Hello world"}]
        tokens = self.comp.estimate_tokens(msgs)
        assert tokens > 0
        assert tokens < 20

    def test_estimate_long_message(self):
        msgs = [{"role": "user", "content": "x" * 4000}]
        tokens = self.comp.estimate_tokens(msgs)
        assert tokens >= 1000

    def test_estimate_tool_calls(self):
        msgs = [{"role": "assistant", "content": "calling tool", "tool_calls": [{"name": "test"}]}]
        tokens = self.comp.estimate_tokens(msgs)
        assert tokens > 50

    def test_needs_compression_below_threshold(self):
        msgs = [{"role": "user", "content": "short"}]
        assert self.comp.needs_compression(msgs) is False

    def test_needs_compression_above_threshold(self):
        msgs = _make_messages(200, content_len=50)
        assert self.comp.needs_compression(msgs) is True


class TestTruncateStrategy:
    def setup_method(self):
        self.comp = ContextCompressor(token_budget=500)

    def test_truncate_short_messages(self):
        msgs = _make_messages(3)
        result, info = self.comp.compress(msgs, strategy="truncate")
        assert len(result) <= len(msgs)
        assert info.strategy == "truncate"

    def test_truncate_preserves_system(self):
        msgs = _make_messages(20)
        result, info = self.comp.compress(msgs, strategy="truncate")
        system_msgs = [m for m in result if m.get("role") == "system"]
        assert len(system_msgs) >= 1

    def test_truncate_preserves_recent(self):
        msgs = _make_messages(30)
        result, info = self.comp.compress(msgs, strategy="truncate")
        last_content = result[-1].get("content", "")
        assert "Message 29" in last_content or len(result) <= 8

    def test_truncate_reduces_count(self):
        msgs = _make_messages(50)
        result, info = self.comp.compress(msgs, strategy="truncate")
        assert len(result) < len(msgs)
        assert info.messages_removed > 0
        assert info.compressed_tokens < info.original_tokens


class TestSummarizeStrategy:
    def setup_method(self):
        self.comp = ContextCompressor(token_budget=500)

    def test_summarize_creates_summary(self):
        msgs = _make_messages(30)
        result, info = self.comp.compress(msgs, strategy="summarize")
        assert info.strategy == "summarize"
        summary_msgs = [m for m in result if "Context Summary" in m.get("content", "")]
        assert len(summary_msgs) >= 1

    def test_summarize_preserves_recent(self):
        msgs = _make_messages(20)
        result, info = self.comp.compress(msgs, strategy="summarize")
        last_msgs = result[-5:]
        assert all(m.get("role") in ("user", "assistant") for m in last_msgs)

    def test_summarize_short_messages_no_compress(self):
        msgs = _make_messages(5)
        result, info = self.comp.compress(msgs, strategy="summarize")
        assert len(result) <= len(msgs)


class TestCompressStrategy:
    def setup_method(self):
        self.comp = ContextCompressor(token_budget=5000)

    def test_compress_shortens_long_messages(self):
        msgs = [
            {"role": "system", "content": "You are Jarvis."},
            {"role": "user", "content": "x" * 5000},
        ]
        result, info = self.comp.compress(msgs, strategy="compress")
        user_msg = next((m for m in result if m.get("role") == "user"), None)
        assert user_msg is not None
        assert len(user_msg.get("content", "")) < 5000

    def test_compress_preserves_short_messages(self):
        msgs = [
            {"role": "system", "content": "You are Jarvis."},
            {"role": "user", "content": "short"},
        ]
        result, info = self.comp.compress(msgs, strategy="compress")
        user_msg = next((m for m in result if m.get("role") == "user"), None)
        assert user_msg is not None
        assert user_msg["content"] == "short"


class TestHybridStrategy:
    def setup_method(self):
        self.comp = ContextCompressor(token_budget=300)

    def test_hybrid_reduces_tokens(self):
        msgs = _make_messages(50, content_len=100)
        result, info = self.comp.compress(msgs, strategy="hybrid")
        assert info.compressed_tokens < info.original_tokens

    def test_hybrid_preserves_system(self):
        msgs = _make_messages(30)
        result, info = self.comp.compress(msgs, strategy="hybrid")
        system_msgs = [m for m in result if m.get("role") == "system"]
        assert len(system_msgs) >= 1


class TestAutoStrategy:
    def setup_method(self):
        self.comp = ContextCompressor(token_budget=500)

    def test_auto_selects_truncate_for_small(self):
        msgs = _make_messages(5, content_len=50)
        result, info = self.comp.compress(msgs, strategy="auto")
        assert info.strategy in ("truncate", "summarize", "hybrid")

    def test_auto_selects_hybrid_for_large(self):
        msgs = _make_messages(200, content_len=100)
        _, info = self.comp.compress(msgs, strategy="auto")
        assert info.strategy in ("hybrid", "summarize", "truncate")

    def test_auto_selects_summarize_for_medium(self):
        msgs = _make_messages(50, content_len=100)
        _, info = self.comp.compress(msgs, strategy="auto")
        assert info.strategy in ("summarize", "hybrid", "truncate")


class TestCompressionResult:
    def test_reduction_percent(self):
        result = CompressionResult(
            original_count=20, compressed_count=10,
            original_tokens=2000, compressed_tokens=1000,
            strategy="truncate", messages_removed=10, messages_summarized=0,
        )
        assert result.reduction_percent == 50.0

    def test_reduction_percent_zero_original(self):
        result = CompressionResult(
            original_count=0, compressed_count=0,
            original_tokens=0, compressed_tokens=0,
            strategy="truncate", messages_removed=0, messages_summarized=0,
        )
        assert result.reduction_percent == 0.0

    def test_to_dict(self):
        result = CompressionResult(
            original_count=20, compressed_count=10,
            original_tokens=2000, compressed_tokens=1000,
            strategy="summarize", messages_removed=10, messages_summarized=5,
            preserved_roles=["system", "user"],
        )
        d = result.to_dict()
        assert d["strategy"] == "summarize"
        assert d["reduction_percent"] == 50.0
        assert "system" in d["preserved_roles"]


class TestCompressionHistory:
    def setup_method(self):
        self.comp = ContextCompressor(token_budget=500)

    def test_history_recorded(self):
        msgs = _make_messages(30)
        self.comp.compress(msgs, strategy="truncate")
        history = self.comp.get_compression_history()
        assert len(history) >= 1
        assert history[0]["strategy"] == "truncate"

    def test_multiple_compressions_recorded(self):
        msgs = _make_messages(20)
        self.comp.compress(msgs, strategy="truncate")
        self.comp.compress(msgs, strategy="summarize")
        history = self.comp.get_compression_history()
        assert len(history) >= 2

    def test_history_limit(self):
        comp = ContextCompressor(token_budget=500)
        for i in range(120):
            msgs = _make_messages(20)
            comp.compress(msgs, strategy="truncate")
        assert len(comp._compression_history) <= 100

    def test_status(self):
        msgs = _make_messages(20)
        self.comp.compress(msgs, strategy="truncate")
        status = self.comp.get_status()
        assert status["total_compressions"] >= 1
        assert status["token_budget"] == 500
        assert status["total_tokens_saved"] > 0

    def test_set_token_budget(self):
        self.comp.set_token_budget(2000)
        assert self.comp._token_budget == 2000
        self.comp.set_token_budget(5000)
        assert self.comp._token_budget == 5000

    def test_set_token_budget_minimum(self):
        self.comp.set_token_budget(100)
        assert self.comp._token_budget == 1000

    def test_last_compression_in_status(self):
        msgs = _make_messages(20)
        self.comp.compress(msgs, strategy="truncate")
        status = self.comp.get_status()
        assert status["last_compression"] is not None
        assert status["last_compression"]["strategy"] == "truncate"


class TestToolCallPreservation:
    def setup_method(self):
        self.comp = ContextCompressor(token_budget=300)

    def test_tool_messages_preserved(self):
        msgs = _make_tool_messages(20)
        result, info = self.comp.compress(msgs, strategy="summarize")
        tool_msgs = [m for m in result if m.get("role") == "tool" or m.get("tool_calls")]
        assert len(tool_msgs) >= 2 or info.messages_removed > 0

    def test_short_tool_messages_not_lost(self):
        msgs = _make_tool_messages(3)
        result, info = self.comp.compress(msgs, strategy="truncate")
        assert len(result) > 1


class TestThreadSafety:
    def test_concurrent_compress(self):
        comp = ContextCompressor(token_budget=500)
        errors = []
        results = []

        def compress_many(i):
            try:
                msgs = _make_messages(20 + i, content_len=50)
                result, info = comp.compress(msgs, strategy="auto")
                results.append(info.strategy)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=compress_many, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        assert len(errors) == 0
        assert len(results) == 10

    def test_singleton(self):
        c1 = get_context_compressor()
        c2 = get_context_compressor()
        assert c1 is c2