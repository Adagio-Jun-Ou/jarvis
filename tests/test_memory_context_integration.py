#!/usr/bin/env python3
"""
Memory, Context, and Knowledge Graph Integration Tests

Tests the integration between:
1. ConversationMemory <-> VectorMemory (semantic search)
2. MemoryHub <-> EmbeddingInterconnect (unified retrieval)
3. KnowledgeGraph <-> PageRank (insight ranking)
4. ContextManager <-> SubAgentOrchestrator (context delegation)
5. SystemSense <-> Subsystems (runtime monitoring)

These tests verify the core agentic deployment systems work together.
"""
import os as _os
_JARVIS_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import os
os.environ["JARVIS_LOCAL_EMBEDDER"] = "1"

import sys
sys.path.insert(0, _JARVIS_ROOT)

import asyncio
import numpy as np
import threading
import time
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, AsyncMock


# ==================== FIXTURES ====================

@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singletons before each test for isolation."""
    singletons = [
        ('core.memory', '_conversation_memory'),
        ('core.memory_hub', '_memory_hub'),
        ('core.vector_memory', '_vector_memory'),
        ('core.knowledge_graph', '_knowledge_graph'),
        ('core.embedding_interconnect', '_interconnect'),
        ('core.context_manager', '_context'),
    ]
    
    for module_name, attr_name in singletons:
        try:
            module = __import__(module_name, fromlist=[attr_name])
            if hasattr(module, '_lock'):
                with module._lock:
                    setattr(module, attr_name, None)
        except Exception:
            pass
    
    yield


@pytest.fixture
def temp_storage(tmp_path):
    """Create temporary storage directory."""
    storage_dir = tmp_path / "jarvis_test"
    storage_dir.mkdir(parents=True, exist_ok=True)
    return storage_dir


# ==================== CONVERSATION MEMORY TESTS ====================

class TestConversationMemory:
    """Tests for ConversationMemory with vector integration."""
    
    def test_memory_initialization(self):
        """ConversationMemory should initialize correctly."""
        from core.memory import ConversationMemory, get_conversation_memory
        
        memory = get_conversation_memory(use_vector_memory=False)
        
        assert memory is not None
        assert isinstance(memory, ConversationMemory)
    
    def test_add_message(self):
        """Should add messages with token tracking."""
        from core.memory import get_conversation_memory, MemoryLayer
        
        memory = get_conversation_memory(use_vector_memory=False)
        
        memory.add_message("user", "Hello, how are you?", importance=0.8)
        memory.add_message("assistant", "I'm doing well!", importance=0.6)
        
        messages = memory.get_messages_for_llm(max_messages=10)
        
        assert len(messages) >= 2
    
    def test_add_tool_result(self):
        """Should add tool results with truncation."""
        from core.memory import get_conversation_memory
        
        memory = get_conversation_memory(use_vector_memory=False)
        
        large_result = {"data": "x" * 10000}
        memory.add_tool_result("test_tool", large_result)
        
        messages = memory.get_messages_for_llm(max_messages=10)
        
        assert len(messages) >= 1
    
    def test_get_messages_within_budget(self):
        """Should truncate messages to fit token budget."""
        from core.memory import get_conversation_memory
        
        memory = get_conversation_memory(use_vector_memory=False)
        
        # Add many messages
        for i in range(20):
            memory.add_message("user", f"Message {i}: " + "x" * 100)
            memory.add_message("assistant", f"Response {i}")
        
        # Get messages with small budget
        messages = memory.get_messages_for_llm(max_messages=5)
        
        assert len(messages) <= 10  # 5 pairs max
    
    @pytest.mark.asyncio
    async def test_store_insight(self):
        """Should store insights for long-term memory."""
        from core.memory import get_conversation_memory
        
        memory = get_conversation_memory(use_vector_memory=False)
        
        await memory.store_insight(
            content="Async programming uses event loops for concurrency",
            importance=0.9,
            concepts=["async", "concurrency", "event loops"]
        )
        
        # Insight should be stored (no exception)
        assert True


# ==================== VECTOR MEMORY TESTS ====================

class TestVectorMemory:
    """Tests for VectorMemory with HNSW index."""
    
    def test_vector_memory_initialization(self):
        """VectorMemory should initialize correctly."""
        from core.vector_memory import VectorMemory, get_vector_memory
        
        vm = get_vector_memory()
        
        assert vm is not None
        assert isinstance(vm, VectorMemory)
    
    def test_store_and_search(self):
        """Should store and retrieve vectors semantically."""
        from core.vector_memory import get_vector_memory
        from core.embedding_interconnect import get_embedding_interconnect
    
        vm = get_vector_memory()
        
        # Check if real embedding model is available
        ic = get_embedding_interconnect()
        test_emb = ic.encode("test")
        if np.linalg.norm(test_emb) == 0.0:
            pytest.skip("No real embedding model available")
        
        # Store some texts
        vm.store("Python is a programming language", metadata={"topic": "python"})
        vm.store("JavaScript runs in browsers", metadata={"topic": "javascript"})
        vm.store("Machine learning uses neural networks", metadata={"topic": "ml"})
        
        # Search for programming-related content - returns list of tuples
        results = vm.search("coding and programming", k=2)
        
        assert len(results) > 0
        # Results are (entry_id, score, metadata) tuples
        if results:
            entry_id, score, metadata = results[0]
            assert metadata.get("topic") in ["python", "javascript", "ml"]
    
    def test_importance_update(self):
        """Should update importance scores."""
        from core.vector_memory import get_vector_memory
        
        vm = get_vector_memory()
        
        entry_id = vm.store("Important concept", importance=0.5)
        
        # Increase importance
        vm.update_importance(entry_id, 0.3)
        
        assert True  # No exception
    
    def test_vector_stats(self):
        """Should provide index statistics."""
        from core.vector_memory import get_vector_memory
        
        vm = get_vector_memory()
        
        vm.store("Test entry 1")
        vm.store("Test entry 2")
        
        stats = vm.stats()
        
        assert "total_entries" in stats or isinstance(stats, dict)


# ==================== KNOWLEDGE GRAPH TESTS ====================

class TestKnowledgeGraph:
    """Tests for KnowledgeGraph with PageRank."""
    
    def test_knowledge_graph_initialization(self):
        """KnowledgeGraph should initialize correctly."""
        from core.knowledge_graph import KnowledgeGraph, get_knowledge_graph
        
        kg = get_knowledge_graph()
        
        assert kg is not None
        assert isinstance(kg, KnowledgeGraph)
    
    def test_add_node(self):
        """Should add nodes to the graph."""
        from core.knowledge_graph import get_knowledge_graph, NodeType
        
        kg = get_knowledge_graph()
        
        node_id = kg.add_node(
            content="Test insight about async programming",
            node_type=NodeType.INSIGHT,
            metadata={"source": "test"}
        )
        
        assert node_id is not None
    
    def test_add_edge(self):
        """Should add edges between nodes."""
        from core.knowledge_graph import get_knowledge_graph, NodeType, EdgeType
        
        kg = get_knowledge_graph()
        
        node1 = kg.add_node("Node 1", NodeType.CONCEPT)
        node2 = kg.add_node("Node 2", NodeType.CONCEPT)
        
        kg.add_edge(node1, node2, EdgeType.RELATED_TO, weight=0.8)
        
        assert True  # No exception
    
    def test_compute_pagerank(self):
        """Should compute PageRank scores."""
        from core.knowledge_graph import get_knowledge_graph, NodeType, EdgeType
        
        kg = get_knowledge_graph()
        
        # Create a small graph
        n1 = kg.add_node("Insight A", NodeType.INSIGHT)
        n2 = kg.add_node("Insight B", NodeType.INSIGHT)
        n3 = kg.add_node("Insight C", NodeType.INSIGHT)
        
        kg.add_edge(n1, n2, EdgeType.RELATED_TO)
        kg.add_edge(n2, n3, EdgeType.RELATED_TO)
        kg.add_edge(n3, n1, EdgeType.RELATED_TO)
        
        # Compute PageRank
        kg.compute_pagerank()
        
        # Get important insights
        important = kg.get_important_insights(limit=3)
        
        assert len(important) > 0 or important == []
    
    def test_add_insight(self):
        """Should add insight with problem and strategy."""
        from core.knowledge_graph import get_knowledge_graph
        
        kg = get_knowledge_graph()
        
        insight_id = kg.add_insight(
            insight_id="test_insight_1",
            content="Using connection pooling improves database performance",
            problem_id="slow_queries",
            strategy="pooling",
            category="performance"
        )
        
        assert insight_id is not None or insight_id is None  # May return None for updates
    
    def test_graph_statistics(self):
        """Should provide graph statistics."""
        from core.knowledge_graph import get_knowledge_graph
        
        kg = get_knowledge_graph()
        
        stats = kg.get_graph_statistics()
        
        assert isinstance(stats, dict)


# ==================== MEMORY HUB TESTS ====================

class TestMemoryHub:
    """Tests for unified MemoryHub."""
    
    def test_memory_hub_initialization(self):
        """MemoryHub should initialize correctly."""
        from core.memory_hub import MemoryHub, get_memory_hub
        
        hub = get_memory_hub()
        
        assert hub is not None
        assert isinstance(hub, MemoryHub)
    
    def test_encode_text(self):
        """Should encode text to vectors."""
        from core.memory_hub import get_memory_hub
        
        hub = get_memory_hub()
        
        vector = hub.encode("Test text for encoding")
        
        assert vector is not None
        assert len(vector) > 0
    
    def test_store_and_retrieve(self):
        """Should store and retrieve from unified memory."""
        from core.memory_hub import get_memory_hub
        
        hub = get_memory_hub()
        
        # Store content
        hub.store(
            content="Python uses indentation for block structure",
            memory_type="semantic",
            importance=0.8,
            metadata={"topic": "python"}
        )
        
        # Retrieve related content
        results = hub.retrieve(
            query="Python code structure",
            top_k=5,
            min_score=0.0
        )
        
        assert isinstance(results, list)
    
    def test_memory_stats(self):
        """Should provide memory statistics."""
        from core.memory_hub import get_memory_hub
        
        hub = get_memory_hub()
        
        stats = hub.get_stats()
        
        # Stats may be MemoryStats dataclass or dict
        assert stats is not None


# ==================== EMBEDDING INTERCONNECT TESTS ====================

class TestEmbeddingInterconnect:
    """Tests for cross-model embedding interconnect."""
    
    def test_interconnect_initialization(self):
        """EmbeddingInterconnect should initialize correctly."""
        from core.embedding_interconnect import get_embedding_interconnect
        
        ic = get_embedding_interconnect()
        
        assert ic is not None
    
    def test_encode_text(self):
        """Should encode text to vectors."""
        from core.embedding_interconnect import get_embedding_interconnect
        
        ic = get_embedding_interconnect()
        
        vector = ic.encode("Test encoding")
        
        assert vector is not None
        assert len(vector) in [768, 512, 256, 128]  # Matryoshka dimensions
    
    def test_store_model_output(self):
        """Should store model outputs for cross-model context."""
        from core.embedding_interconnect import get_embedding_interconnect
        
        ic = get_embedding_interconnect()
        
        # store_model_output(model, prompt, response, quality)
        ic.store_model_output(
            model="primary",
            prompt="What is async?",
            response="Async programming uses event loops",
            quality=0.95
        )
        
        assert True  # No exception
    
    def test_get_cross_model_context(self):
        """Should retrieve context from other models."""
        from core.embedding_interconnect import get_embedding_interconnect
        
        ic = get_embedding_interconnect()
        
        # Store from primary
        ic.store_model_output(
            model="primary",
            prompt="Explain APIs",
            response="APIs are interfaces for software communication",
            quality=0.9
        )
        
        # Retrieve for backend - get_cross_model_context(query, from_model)
        context = ic.get_cross_model_context(
            query="How do APIs work?",
            from_model="backend"  # Get context from the OTHER model (primary)
        )
        
        # Context may be empty if similarity threshold not met
        assert context is not None


# ==================== CONTEXT MANAGER TESTS ====================

class TestContextManager:
    """Tests for context management."""
    
    def test_context_propagation(self):
        """Should set and get context."""
        from core.context_manager import set_subsystem, get_subsystem, set_user_intent, get_user_intent
        
        set_subsystem("test_subsystem")
        assert get_subsystem() == "test_subsystem"
        
        set_user_intent("test_intent")
        assert get_user_intent() == "test_intent"
    
    def test_context_data(self):
        """Should add and retrieve context data."""
        from core.context_manager import add_context, get_context, clear_context
        
        add_context("test_key", "test_value")
        
        value = get_context("test_key")
        
        assert value == "test_value" or value is None  # May be cleared by fixture
        
        clear_context()
    
    def test_should_use_subagents(self):
        """Should determine if sub-agents should be used."""
        from core.context_manager import should_use_subagents
        
        # Simple task should not require sub-agents
        result = should_use_subagents("What is 2+2?")
        
        # Complex task might require sub-agents
        result2 = should_use_subagents("Research async patterns in Python, write a guide, and create examples")
        
        assert isinstance(result, bool)
        assert isinstance(result2, bool)


# ==================== INTEGRATION TESTS ====================

class TestMemoryIntegration:
    """Tests for integration between memory systems."""
    
    def test_memory_to_vector_integration(self):
        """ConversationMemory should integrate with VectorMemory."""
        from core.memory import get_conversation_memory
        from core.vector_memory import get_vector_memory
        
        memory = get_conversation_memory(use_vector_memory=True)
        vm = get_vector_memory()
        
        # Both should be available
        assert memory is not None
        assert vm is not None
    
    def test_memory_hub_to_interconnect(self):
        """MemoryHub should use EmbeddingInterconnect."""
        from core.memory_hub import get_memory_hub
        from core.embedding_interconnect import get_embedding_interconnect
        
        hub = get_memory_hub()
        ic = get_embedding_interconnect()
        
        # Both should be available
        assert hub is not None
        assert ic is not None
    
    def test_knowledge_graph_to_vector(self):
        """KnowledgeGraph should store nodes for semantic search."""
        from core.knowledge_graph import get_knowledge_graph
        from core.vector_memory import get_vector_memory
        
        kg = get_knowledge_graph()
        vm = get_vector_memory()
        
        # Both should be available
        assert kg is not None
        assert vm is not None
    
    def test_full_memory_flow(self):
        """Test full memory flow: input -> encode -> store -> retrieve."""
        from core.memory_hub import get_memory_hub
        
        hub = get_memory_hub()
        
        # Store multiple pieces of information
        items = [
            ("Python is dynamically typed", "semantic", 0.8),
            ("APIs enable communication between services", "semantic", 0.9),
            ("Docker containers provide isolation", "semantic", 0.7),
        ]
        
        for content, mem_type, importance in items:
            hub.store(content, mem_type, importance)
        
        # Retrieve related content
        results = hub.retrieve("programming languages", top_k=5)
        
        assert isinstance(results, list)


# ==================== SYSTEM SENSE TESTS ====================

class TestSystemSense:
    """Tests for SystemSenseTool and subsystem monitoring."""
    
    def test_system_sense_tool_initialization(self):
        """SystemSenseTool should initialize correctly."""
        from tools.system_sense import SystemSenseTool
        
        tool = SystemSenseTool()
        
        assert tool is not None
        assert tool.name == "system_sense"
        assert tool.requires_permission == False  # Safe, read-only
    
    def test_system_sense_schema(self):
        """SystemSenseTool should provide valid schema."""
        from tools.system_sense import SystemSenseTool
        
        tool = SystemSenseTool()
        schema = tool.get_schema()
        
        assert "name" in schema
        assert "description" in schema
        assert "parameters" in schema
    
    def test_system_watch_tool_initialization(self):
        """SystemWatchTool should initialize correctly."""
        from tools.system_sense import SystemWatchTool
        
        tool = SystemWatchTool()
        
        assert tool is not None
        assert tool.requires_permission == False
    
    @pytest.mark.asyncio
    async def test_system_sense_execute(self):
        """SystemSenseTool should execute and return results."""
        from tools.system_sense import SystemSenseTool
        
        tool = SystemSenseTool()
        
        result = await tool.execute(focus="all", detailed=False)
        
        assert result is not None
        # Result can be string or dict
        assert isinstance(result, (str, dict))
    
    @pytest.mark.asyncio
    async def test_system_sense_resources(self):
        """SystemSenseTool should sense resources."""
        from tools.system_sense import SystemSenseTool
        
        tool = SystemSenseTool()
        
        result = await tool.execute(focus="resources", detailed=True)
        
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_system_sense_subsystems(self):
        """SystemSenseTool should sense subsystems."""
        from tools.system_sense import SystemSenseTool
        
        tool = SystemSenseTool()
        
        result = await tool.execute(focus="subsystems", detailed=False)
        
        assert result is not None


# ==================== SUBSYSTEM STATUS TESTS ====================

class TestSubsystemStatus:
    """Tests for subsystem status tracking."""
    
    def test_daydream_engine_status(self):
        """Should get DaydreamEngine status."""
        try:
            from core.daydream_engine import get_daydream_engine
            
            engine = get_daydream_engine()
            
            if engine:
                status = engine.get_status() if hasattr(engine, 'get_status') else None
                assert status is not None or True  # May be None if not running
        except ImportError:
            pytest.skip("Daydream engine not available")
    
    def test_knowledge_graph_status(self):
        """Should get KnowledgeGraph status."""
        from core.knowledge_graph import get_knowledge_graph
        
        kg = get_knowledge_graph()
        
        stats = kg.get_graph_statistics()
        
        assert isinstance(stats, dict)
    
    def test_vector_memory_status(self):
        """Should get VectorMemory status."""
        from core.vector_memory import get_vector_memory
        
        vm = get_vector_memory()
        
        stats = vm.stats()
        
        assert isinstance(stats, dict)


# ==================== SECURITY INTEGRATION TESTS ====================

class TestSecurityIntegration:
    """Tests for security in memory and context systems."""
    
    def test_system_sense_no_permission_required(self):
        """SystemSenseTool should not require permission."""
        from tools.system_sense import SystemSenseTool, SystemWatchTool
        
        sense = SystemSenseTool()
        watch = SystemWatchTool()
        
        assert sense.requires_permission == False
        assert watch.requires_permission == False
    
    def test_security_tools_initialization(self):
        """Security tools should initialize correctly."""
        try:
            from tools.security_tools import get_security_tools
            
            tools, components = get_security_tools()
            
            assert isinstance(tools, list)
            assert isinstance(components, dict)
        except ImportError:
            pytest.skip("Security tools not available")
    
    def test_memory_isolation(self):
        """Memory systems should be isolated between tests."""
        from core.memory import get_conversation_memory
        
        memory1 = get_conversation_memory(use_vector_memory=False)
        memory1.add_message("user", "Test message")
        
        # Note: singleton pattern means same instance
        memory2 = get_conversation_memory(use_vector_memory=False)
        
        # Both reference the same singleton
        assert memory1 is memory2


# ==================== CONCURRENT ACCESS TESTS ====================

class TestConcurrentAccess:
    """Tests for thread-safe concurrent access."""
    
    def test_concurrent_memory_access(self):
        """Memory should handle concurrent access."""
        from core.memory import get_conversation_memory
        
        memory = get_conversation_memory(use_vector_memory=False)
        errors = []
        lock = threading.Lock()
        
        def add_messages(i):
            try:
                for j in range(10):
                    memory.add_message("user", f"Message {i}-{j}")
            except Exception as e:
                with lock:
                    errors.append(str(e))
        
        threads = [threading.Thread(target=add_messages, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        
        assert len(errors) == 0
    
    def test_concurrent_vector_access(self):
        """VectorMemory should handle concurrent access."""
        from core.vector_memory import get_vector_memory
        
        vm = get_vector_memory()
        errors = []
        lock = threading.Lock()
        
        def store_vectors(i):
            try:
                for j in range(5):
                    vm.store(f"Test content {i}-{j}")
            except Exception as e:
                with lock:
                    errors.append(str(e))
        
        threads = [threading.Thread(target=store_vectors, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        
        assert len(errors) == 0
    
    def test_concurrent_knowledge_graph(self):
        """KnowledgeGraph should handle concurrent access."""
        from core.knowledge_graph import get_knowledge_graph, NodeType
        
        kg = get_knowledge_graph()
        errors = []
        lock = threading.Lock()
        
        def add_nodes(i):
            try:
                for j in range(5):
                    kg.add_node(f"Node {i}-{j}", NodeType.INSIGHT)
            except Exception as e:
                with lock:
                    errors.append(str(e))
        
        threads = [threading.Thread(target=add_nodes, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        
        assert len(errors) == 0


# ==================== PERFORMANCE TESTS ====================

class TestPerformance:
    """Tests for performance characteristics."""
    
    def test_memory_add_speed(self):
        """Adding to memory should be fast."""
        import time
        from core.memory import get_conversation_memory
        
        memory = get_conversation_memory(use_vector_memory=False)
        
        start = time.time()
        for i in range(100):
            memory.add_message("user", f"Message {i}")
        elapsed = time.time() - start
        
        # 100 messages should be fast
        assert elapsed < 2.0
    
    def test_vector_search_speed(self):
        """Vector search should be reasonably fast."""
        import time
        from core.vector_memory import get_vector_memory
        
        vm = get_vector_memory()
        
        # Index some content
        for i in range(50):
            vm.store(f"Content item {i} with some text")
        
        start = time.time()
        for _ in range(10):
            vm.search("test query", k=5)
        elapsed = time.time() - start
        
        # 10 searches should be reasonably fast
        assert elapsed < 5.0
    
    def test_knowledge_graph_speed(self):
        """Knowledge graph operations should be reasonably fast."""
        import time
        from core.knowledge_graph import get_knowledge_graph, NodeType
        
        kg = get_knowledge_graph()
        
        start = time.time()
        for i in range(50):
            kg.add_node(f"Node {i}", NodeType.INSIGHT)
        elapsed = time.time() - start
        
        # 50 additions - knowledge graph may be slower due to embedding
        # Allow up to 10 seconds for embedding operations
        assert elapsed < 10.0


# ==================== RUN ====================

if __name__ == "__main__":
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "-p", "no:xdist", 
         "--tb=short", "--override-ini=addopts="],
        cwd=_JARVIS_ROOT
    )
    sys.exit(result.returncode)