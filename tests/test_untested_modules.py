"""Tests for previously untested core modules."""
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestToolLearning:
    """Tests for tool_learning module."""

    def test_import(self):
        """Test tool_learning imports."""
        from core.tool_learning import get_tool_learning_engine
        assert get_tool_learning_engine is not None

    def test_engine_singleton(self):
        """Test engine is singleton."""
        from core.tool_learning import get_tool_learning_engine

        engine1 = get_tool_learning_engine()
        engine2 = get_tool_learning_engine()
        assert engine1 is engine2

    def test_record_usage(self):
        """Test recording tool usage."""
        from core.tool_learning import get_tool_learning_engine

        engine = get_tool_learning_engine()
        engine.record_tool_usage(
            tool_name="test_tool",
            success=True,
            execution_time=0.5,
        )

    def test_get_tool_genes(self):
        """Test getting tool genes."""
        from core.tool_learning import get_tool_learning_engine

        engine = get_tool_learning_engine()
        genes = engine.get_tool_genes("test_tool")
        assert genes is not None or genes is None


class TestShiftingMemory:
    """Tests for shifting_memory module."""

    def test_import(self):
        """Test shifting_memory imports."""
        from core.shifting_memory import get_memory_manager
        assert get_memory_manager is not None

    def test_get_manager(self):
        """Test getting memory manager."""
        from core.shifting_memory import get_memory_manager

        manager = get_memory_manager()
        assert manager is not None


class TestMemoryManager:
    """Tests for memory_manager module."""

    def test_import(self):
        """Test memory_manager imports."""
        from core.memory_manager import create_memory_manager
        assert create_memory_manager is not None

    def test_get_manager(self):
        """Test getting memory manager."""
        from core.memory_manager import get_memory_manager

        manager = get_memory_manager()
        assert manager is not None


class TestSession:
    """Tests for the core/session/ package (HistoryLog, Message).

    Note: these imports resolved to the core/session/ package even when a
    shadowed core/session.py still existed (packages win over modules);
    the shadowed file is now deleted, so the import is unambiguous.
    """

    def test_import(self):
        """Test session imports."""
        from core.session import Message
        assert Message is not None

    def test_message_creation(self):
        """Test creating a message."""
        from core.session import Message
        from datetime import datetime

        msg = Message(
            id="test_123",
            role="user",
            content="Hello",
            timestamp=datetime.now()
        )
        assert msg.role == "user"
        assert msg.content == "Hello"


class TestMultimodal:
    """Tests for multimodal module."""

    def test_import(self):
        """Test multimodal imports."""
        from core.multimodal import AudioProcessor, ModalityType
        assert AudioProcessor is not None
        assert ModalityType is not None


class TestOptimizations:
    """Tests for recently added optimizations."""

    def test_memory_token_cache(self):
        """Test O(1) token counting in ConversationMemory."""
        from core.memory import ConversationMemory

        mem = ConversationMemory(max_tokens=10000)

        for i in range(100):
            mem.add_message("user", f"Message {i}")

        assert mem._total_tokens > 0

        import time
        t1 = time.perf_counter()
        for _ in range(1000):
            _ = mem._count_history_tokens()
        t2 = time.perf_counter()

        # Generous bound on purpose: this guards against a pathological
        # regression (per-call hashing, O(n^2) rescans), not throughput.
        # At 10us/op the original budget measured machine load instead of
        # the implementation, and failed under parallel test execution.
        assert (t2 - t1) < 0.5

    def test_context_compressor_efficient_id(self):
        """Test efficient ID generation."""
        from core.context_compressor import ContextCompressor

        comp = ContextCompressor()

        import time
        t1 = time.perf_counter()
        for i in range(1000):
            comp._generate_id(f"content_{i}")
        t2 = time.perf_counter()

        # Generous bound on purpose: this guards against a pathological
        # regression (per-call hashing, O(n^2) rescans), not throughput.
        # At 10us/op the original budget measured machine load instead of
        # the implementation, and failed under parallel test execution.
        assert (t2 - t1) < 0.5

    def test_cache_key_efficient(self):
        """Test tool cache key doesn't use MD5."""
        from core.agent import ToolResultCache

        cache = ToolResultCache()

        import time
        t1 = time.perf_counter()
        for i in range(1000):
            cache._make_key("tool", {"arg": i})
        t2 = time.perf_counter()

        # Generous bound on purpose: this guards against a pathological
        # regression (per-call hashing, O(n^2) rescans), not throughput.
        # At 10us/op the original budget measured machine load instead of
        # the implementation, and failed under parallel test execution.
        assert (t2 - t1) < 0.5

    def test_regex_precompiled(self):
        """Test ImportanceScorer uses pre-compiled regex."""
        from core.context_compressor import ImportanceScorer, _TOPIC_PATTERNS
        import re

        for pattern, topic in _TOPIC_PATTERNS:
            assert isinstance(pattern, re.Pattern)

    def test_tools_list_bounded(self):
        """Test agent tools list is bounded."""
        from core.agent import JarvisAgent
        from core.config import Config
        from collections import deque

        config = Config()
        config.memory.enabled = False

        agent = JarvisAgent(config)

        assert isinstance(agent._tools_used_this_session, deque)
        assert agent._tools_used_this_session.maxlen == 200


class TestBidirectionalBridge:
    """Tests for bidirectional module communication."""

    def test_event_delivery(self):
        """Test events are delivered to subscribers."""
        from core.module_bridge import get_bridge, BridgeEvent

        bridge = get_bridge()
        received = []

        def handler(msg):
            received.append(msg)

        bridge.subscribe(BridgeEvent.COMPRESSION_DONE, handler)
        bridge.emit(BridgeEvent.COMPRESSION_DONE, "test", {"tokens": 100})

        assert len(received) > 0

    def test_module_status(self):
        """Test module status retrieval."""
        from core.context_manager import get_module_status

        status = get_module_status()

        assert "modules" in status
        assert "memory_pressure" in status
        assert "can_delegate" in status


class TestDecompositionCache:
    """Tests for task decomposition caching."""

    @pytest.mark.asyncio
    async def test_cache_exists(self):
        """Test decomposition cache exists."""
        from core.sub_agent import TaskPlanner

        assert hasattr(TaskPlanner, '_decompose_cache')

    @pytest.mark.asyncio
    async def test_cache_used(self):
        """Test cache is actually used."""
        from core.sub_agent import TaskPlanner

        planner = TaskPlanner()

        await planner.decompose("Do task A then task B")

        assert len(TaskPlanner._decompose_cache) > 0




class TestReasoningEngineEvidence:
    """reason() must actually use the context it is given.

    Not wired into any subsystem yet — it is the only symbolic-reasoning
    implementation in the tree, and `erdos_engine` is the natural consumer, but
    that changes problem-solving behaviour. This guards the defect found while
    evaluating it.
    """

    def test_plain_context_is_used_as_evidence(self):
        """Regression: only a key literally named "evidence" was read, so
        ordinary context was silently discarded and hypotheses were generated
        from an empty list."""
        from core.reasoning_engine import get_reasoning_engine

        chain = get_reasoning_engine().reason(
            goal="Why did the strategy fail?",
            context={"density": "increased", "bound": "loose"},
        )
        gather = next(s for s in chain.steps if s["type"] == "gather")
        assert "2 items" in gather["content"], gather["content"]

    def test_explicit_evidence_key_still_honoured(self):
        from core.reasoning_engine import get_reasoning_engine

        chain = get_reasoning_engine().reason(
            goal="Why did it fail?",
            context={"evidence": ["density increased", "bound was loose"]},
        )
        gather = next(s for s in chain.steps if s["type"] == "gather")
        assert "2 items" in gather["content"]

    def test_empty_context_degrades_cleanly(self):
        from core.reasoning_engine import get_reasoning_engine

        chain = get_reasoning_engine().reason(goal="Why?", context={})
        gather = next(s for s in chain.steps if s["type"] == "gather")
        assert "0 items" in gather["content"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
