"""Tests for context compression and sub-agent orchestration."""
import pytest
from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestTokenCounter:
    """Tests for token counting."""

    def test_count_basic(self):
        """Test basic token counting."""
        from core.context_compressor import TokenCounter

        text = "Hello world"
        count = TokenCounter.count(text)
        assert count > 0
        assert count <= len(text)  # At most 1 token per char

    def test_count_messages(self):
        """Test counting messages."""
        from core.context_compressor import TokenCounter

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        count = TokenCounter.count_messages(messages)
        assert count > 0


class TestImportanceScorer:
    """Tests for importance scoring."""

    def test_high_importance_patterns(self):
        """Test detection of high importance."""
        from core.context_compressor import ImportanceScorer

        # High importance phrases
        score1 = ImportanceScorer.score("Please remember that my name is Alice")
        score2 = ImportanceScorer.score("This is important: the password is secret123")
        score3 = ImportanceScorer.score("Note: I prefer Python over JavaScript")

        assert score1 > 0.7
        assert score2 > 0.7
        assert score3 > 0.6

    def test_low_importance_patterns(self):
        """Test detection of low importance."""
        from core.context_compressor import ImportanceScorer

        score1 = ImportanceScorer.score("ok")
        score2 = ImportanceScorer.score("thanks")

        assert score1 < 0.5
        assert score2 < 0.5

    def test_code_importance(self):
        """Test code detection."""
        from core.context_compressor import ImportanceScorer

        score1 = ImportanceScorer.score("Here is some code:\n```python\nprint('hello')\n```")
        score2 = ImportanceScorer.score("Just some regular text")

        assert score1 > score2


class TestContextBlock:
    """Tests for ContextBlock."""

    def test_create_block(self):
        """Test creating a context block."""
        from core.context_compressor import ContextBlock, MemoryTier

        block = ContextBlock(
            id="test-1",
            content="Hello world",
            role="user",
            importance=0.8,
        )

        assert block.id == "test-1"
        assert block.content == "Hello world"
        assert block.importance == 0.8
        assert block.tier == MemoryTier.ACTIVE
        assert block.access_count == 0

    def test_touch(self):
        """Test access tracking."""
        from core.context_compressor import ContextBlock

        block = ContextBlock(id="test-1", content="Test")

        block.touch()
        assert block.access_count == 1

    def test_serialization(self):
        """Test to_dict and from_dict."""
        from core.context_compressor import ContextBlock, MemoryTier

        block = ContextBlock(
            id="test-1",
            content="Test content",
            role="assistant",
            importance=0.9,
            topics=["code", "python"],
            entities=["Jarvis"],
            tier=MemoryTier.COMPRESSED,
        )

        data = block.to_dict()
        restored = ContextBlock.from_dict(data)

        assert restored.id == block.id
        assert restored.content == block.content
        assert restored.importance == block.importance
        assert restored.topics == block.topics
        assert restored.tier == MemoryTier.COMPRESSED


class TestContextCompressor:
    """Tests for ContextCompressor."""

    def test_create_compressor(self, tmp_path):
        """Test creating compressor."""
        from core.context_compressor import ContextCompressor

        compressor = ContextCompressor(storage_path=tmp_path)

        assert compressor is not None
        assert compressor.active_blocks == []

    def test_add_message(self, tmp_path):
        """Test adding messages."""
        from core.context_compressor import ContextCompressor

        compressor = ContextCompressor(storage_path=tmp_path)

        block = compressor.add_message(
            role="user",
            content="Hello, how are you?",
            importance=0.8,
        )

        assert len(compressor.active_blocks) == 1
        assert block.role == "user"
        assert block.importance == 0.8

    def test_topic_extraction(self, tmp_path):
        """Test topic extraction from content."""
        from core.context_compressor import ContextCompressor

        compressor = ContextCompressor(storage_path=tmp_path)

        block = compressor.add_message(
            role="user",
            content="Please implement the error fix in main.py",
        )

        # Should detect 'code' topic for file reference
        assert len(block.topics) > 0 or len(block.entities) > 0

    def test_get_messages_for_llm(self, tmp_path):
        """Test getting optimized context."""
        from core.context_compressor import ContextCompressor

        compressor = ContextCompressor(storage_path=tmp_path)

        compressor.add_message("user", "Hello")
        compressor.add_message("assistant", "Hi there!")
        compressor.add_message("user", "How are you?")

        messages = compressor.get_messages_for_llm(max_tokens=1000)

        assert len(messages) == 3
        assert messages[0]["role"] == "user"

    def test_search_memory(self, tmp_path):
        """Test memory search."""
        from core.context_compressor import ContextCompressor

        compressor = ContextCompressor(storage_path=tmp_path)

        compressor.add_message("user", "My favorite color is blue", importance=0.9)
        compressor.add_message("assistant", "I'll remember that!")
        compressor.add_message("user", "What's the weather?")

        results = compressor.search("favorite color")

        assert len(results) > 0
        # results is list of (block, score) tuples
        block = results[0][0] if isinstance(results[0], tuple) else results[0]
        assert "blue" in block.content.lower()

    def test_get_stats(self, tmp_path):
        """Test getting statistics."""
        from core.context_compressor import ContextCompressor

        compressor = ContextCompressor(storage_path=tmp_path)

        compressor.add_message("user", "Hello")
        compressor.add_message("assistant", "Hi!")

        stats = compressor.get_stats()

        assert "active_blocks" in stats
        assert "active_tokens" in stats
        assert stats["active_blocks"] == 2


class TestTaskPlanner:
    """Tests for TaskPlanner."""

    def test_heuristic_decompose_single(self):
        """Test decomposition of single task."""
        from core.sub_agent import TaskPlanner

        planner = TaskPlanner()

        tasks = planner._heuristic_decompose(
            "Write a hello world program",
            None,
        )

        assert len(tasks) >= 1
        assert tasks[0].title != ""

    def test_heuristic_decompose_multiple(self):
        """Test decomposition of multiple tasks."""
        from core.sub_agent import TaskPlanner

        planner = TaskPlanner()

        tasks = planner._heuristic_decompose(
            "Search for the file and then fix the bug",
            None,
        )

        # Should detect "then" pattern
        assert len(tasks) >= 2

    def test_build_dag(self):
        """Test DAG building."""
        from core.sub_agent import TaskPlanner, SubTask

        planner = TaskPlanner()

        tasks = [
            SubTask(id="t1", title="A", description="Task A"),
            SubTask(id="t2", title="B", description="Task B", dependencies=["t1"]),
            SubTask(id="t3", title="C", description="Task C", dependencies=["t1"]),
            SubTask(id="t4", title="D", description="Task D", dependencies=["t2", "t3"]),
        ]

        dag = planner.build_dag(tasks)

        # First level should have t1 only
        assert len(dag[0]) == 1
        assert dag[0][0].id == "t1"

        # Second level should have t2 and t3
        assert len(dag[1]) == 2

        # Third level should have t4
        assert len(dag[2]) == 1
        assert dag[2][0].id == "t4"

    def test_build_dag_parallel(self):
        """Test DAG building with parallel tasks."""
        from core.sub_agent import TaskPlanner, SubTask

        planner = TaskPlanner()

        tasks = [
            SubTask(id="t1", title="A", description="Task A"),
            SubTask(id="t2", title="B", description="Task B"),  # No dependency
            SubTask(id="t3", title="C", description="Task C"),  # No dependency
        ]

        dag = planner.build_dag(tasks)

        # All 3 should be in first level (parallel)
        assert len(dag[0]) == 3


class TestAgentPool:
    """Tests for AgentPool."""

    def test_create_pool(self):
        """Test creating agent pool."""
        from core.sub_agent import AgentPool

        pool = AgentPool()

        assert len(pool.agents) > 0
        assert "researcher" in pool.agents
        assert "coder" in pool.agents

    def test_select_agent(self):
        """Test agent selection."""
        from core.sub_agent import AgentPool, SubTask

        pool = AgentPool()

        task = SubTask(
            id="test-1",
            title="Code Task",
            description="Please implement a function in Python",
        )

        agent_id = pool.select_agent(task)

        # Should return an available agent
        assert agent_id is not None
        assert agent_id in pool.agents

    def test_agent_availability(self):
        """Test marking agents busy/available."""
        from core.sub_agent import AgentPool

        pool = AgentPool()

        # Mark agent busy
        pool.mark_busy("coder")

        # Should not be in available list
        available = pool.get_available_agents()
        assert "coder" not in available

        # Mark available
        pool.mark_available("coder")
        available = pool.get_available_agents()
        assert "coder" in available


class TestSubAgentOrchestrator:
    """Tests for SubAgentOrchestrator."""

    def test_create_orchestrator(self):
        """Test creating orchestrator."""
        from core.sub_agent import SubAgentOrchestrator

        orchestrator = SubAgentOrchestrator()

        assert orchestrator is not None
        assert orchestrator.planner is not None
        assert orchestrator.pool is not None

    def test_get_stats(self):
        """Test getting statistics."""
        from core.sub_agent import SubAgentOrchestrator

        orchestrator = SubAgentOrchestrator()

        stats = orchestrator.get_stats()

        assert "total_tasks" in stats
        assert "available_agents" in stats


class TestContextManager:
    """Tests for ContextManager integration."""

    def test_analyze_simple_request(self):
        """Test analyzing a simple request."""
        from core.context_manager import analyze_request

        decision = analyze_request(
            "What is the capital of France?",
            messages=None,
        )

        assert decision.detected_tasks >= 1
        assert decision.use_subagents == False  # Simple request

    def test_analyze_complex_request(self):
        """Test analyzing a complex request."""
        from core.context_manager import analyze_request

        # Complex request with multiple task indicators
        decision = analyze_request(
            "First, search for all Python files, then analyze each one for bugs, "
            "and then fix the bugs, and finally run the tests to verify",
            messages=None,
        )

        # Should detect multiple tasks based on "first", "then", "and then"
        assert decision.detected_tasks >= 2

    def test_compression_decision(self):
        """Test compression decision."""
        from core.context_manager import analyze_request

        large_context = [
            {"role": "user", "content": "Hello" * 500},
            {"role": "assistant", "content": "Response" * 500},
        ]

        decision = analyze_request(
            "New question",
            messages=large_context,
            max_tokens=1000,
        )

        assert decision.estimated_tokens > 1000
        assert decision.needs_compression


class TestModuleBridge:
    """Tests for bidirectional module bridge."""

    def test_bridge_singleton(self):
        """Test bridge is a singleton."""
        from core.module_bridge import get_bridge

        bridge1 = get_bridge()
        bridge2 = get_bridge()

        assert bridge1 is bridge2

    def test_module_registration(self):
        """Test registering a module."""
        from core.module_bridge import get_bridge, ModuleState

        bridge = get_bridge()
        handle = bridge.register_module("test_module_1")

        assert handle is not None
        assert handle._name == "test_module_1"

        # Can update state
        handle.update_state(ModuleState.ACTIVE, {"foo": "bar"})

        status = handle.get_status()
        assert status is not None
        assert status["state"] == ModuleState.ACTIVE
        assert status["metrics"]["foo"] == "bar"

    def test_event_subscription(self):
        """Test event pub/sub."""
        from core.module_bridge import get_bridge, BridgeEvent

        bridge = get_bridge()
        received = []

        def handler(msg):
            received.append(msg)

        bridge.subscribe(BridgeEvent.COMPRESSION_DONE, handler)
        bridge.emit(BridgeEvent.COMPRESSION_DONE, "test_source", {"tokens_freed": 1000})

        assert len(received) == 1
        assert received[0].event == BridgeEvent.COMPRESSION_DONE
        assert received[0].data["tokens_freed"] == 1000

        # Cleanup
        bridge.unsubscribe(BridgeEvent.COMPRESSION_DONE, handler)

    def test_pending_operations(self):
        """Test tracking pending operations."""
        from core.module_bridge import get_bridge, BridgeEvent

        bridge = get_bridge()

        # Emit memory pressure
        bridge.emit(BridgeEvent.MEMORY_PRESSURE, "test", {"level": 0.9})

        pending = bridge.get_pending_operations()
        assert "pending_compressions" in pending
        assert pending["pending_compressions"] >= 1

    def test_event_history(self):
        """Test event history tracking."""
        from core.module_bridge import get_bridge, BridgeEvent

        bridge = get_bridge()

        # Emit some events
        bridge.emit(BridgeEvent.TASK_DELEGATED, "test", {"task": "task1"})
        bridge.emit(BridgeEvent.TASK_COMPLETED, "test", {"task": "task1"})

        history = bridge.get_event_history(limit=5)
        assert len(history) >= 2

    def test_compression_request(self):
        """Test requesting compression via bridge."""
        from core.module_bridge import get_bridge, BridgeEvent

        bridge = get_bridge()
        received = []

        def handler(msg):
            received.append(msg)

        bridge.subscribe(BridgeEvent.COMPRESSION_NEEDED, handler)

        # Request compression
        result = bridge.request_compression("test_module", "test reason", 0.5)

        # Event should be emitted
        assert len(received) == 1
        assert received[0].data["reason"] == "test reason"

        bridge.unsubscribe(BridgeEvent.COMPRESSION_NEEDED, handler)

    def test_memory_pressure_check(self):
        """Test memory pressure detection."""
        from core.module_bridge import get_bridge, BridgeEvent

        bridge = get_bridge()

        # Clear previous state by waiting out the 30s timeout
        # or checking the timestamp directly
        with bridge._state_lock:
            bridge._last_memory_pressure = 0.0
            bridge._pending_compressions = 0

        # Now should not have pressure
        assert not bridge.is_memory_pressure()

        # Emit pressure event
        bridge.emit(BridgeEvent.MEMORY_PRESSURE, "test", {"level": 0.9})

        # Should detect pressure
        assert bridge.is_memory_pressure()

    def test_delegate_capability(self):
        """Test checking if delegation is possible."""
        from core.module_bridge import get_bridge, BridgeEvent

        bridge = get_bridge()

        # Clear previous state
        with bridge._state_lock:
            bridge._pending_compressions = 0
            bridge._last_memory_pressure = 0.0

        # Should be able to delegate
        assert bridge.can_delegate_tasks()

        # Emit memory pressure
        bridge.emit(BridgeEvent.MEMORY_PRESSURE, "test", {"level": 0.95})

        # Should not be able to delegate under pressure
        assert not bridge.can_delegate_tasks()

    def test_context_manager_bridge_integration(self):
        """Test context manager exposes bridge status."""
        from core.context_manager import get_module_status

        status = get_module_status()

        assert "modules" in status
        assert "pending" in status
        assert "memory_pressure" in status
        assert "can_delegate" in status


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
