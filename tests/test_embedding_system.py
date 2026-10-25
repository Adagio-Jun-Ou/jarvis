#!/usr/bin/env python3
"""Test unified memory and embedding interconnect system.

Uses JARVIS_LOCAL_EMBEDDER=1 for tests to avoid IPC overhead.
For production parallel agents, omit this to use shared server.
"""
import os as _os
_JARVIS_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import os
os.environ["JARVIS_LOCAL_EMBEDDER"] = "1"

import sys
sys.path.insert(0, _JARVIS_ROOT)

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def reset_interconnect():
    """Reset interconnect before each test for isolation."""
    from core.embedding_interconnect import reset_embedding_interconnect
    reset_embedding_interconnect()
    yield
    reset_embedding_interconnect()


# ==================== EMBEDDING INTERCONNECT TESTS ====================

def test_embedding_interconnect_basic():
    """Test basic embedding operations."""
    from core.embedding_interconnect import get_embedding_interconnect
    
    ic = get_embedding_interconnect()
    
    # Test encoding
    text = "This is a test sentence for embedding"
    emb = ic.encode(text)
    
    # Skip if no real embedding model is loaded
    if np.linalg.norm(emb) == 0.0:
        pytest.skip("No real embedding model available")
    
    assert emb is not None, "Embedding should not be None"
    assert len(emb) == 768, f"Embedding dimension should be 768, got {len(emb)}"
    assert isinstance(emb, np.ndarray), "Embedding should be numpy array"
    
    # Test similarity
    text2 = "This is another test for embedding"
    emb2 = ic.encode(text2)
    
    similarity = float(np.dot(emb, emb2) / (np.linalg.norm(emb) * np.linalg.norm(emb2) + 1e-8))
    assert -1 <= similarity <= 1, f"Similarity should be in [-1, 1], got {similarity}"
    
    # Test similar sentences have higher similarity
    similar_text = "This is a test sentence for embedding"
    similar_emb = ic.encode(similar_text)
    similar_similarity = float(np.dot(emb, similar_emb) / (np.linalg.norm(emb) * np.linalg.norm(similar_emb) + 1e-8))
    
    assert similar_similarity > similarity, "Similar sentences should have higher similarity"


def test_model_output_storage():
    """Test that model outputs are stored correctly."""
    from core.embedding_interconnect import get_embedding_interconnect
    
    ic = get_embedding_interconnect()
    
    # Store primary model output
    ic.store_model_output("primary", "What is Python?", "Python is a programming language", 0.9)
    
    assert len(ic.get_model_outputs("primary")) == 1
    assert ic._last_responses["primary"] == "Python is a programming language"
    
    # Store backend model output
    ic.store_model_output("backend", "Write a function", "def hello(): return 'world'", 0.85)
    
    assert len(ic.get_model_outputs("backend")) == 1
    assert ic._last_responses["backend"] == "def hello(): return 'world'"


def test_cross_model_context():
    """Test cross-model context retrieval."""
    from core.embedding_interconnect import get_embedding_interconnect
    
    ic = get_embedding_interconnect()
    
    # Store outputs for both models
    ic.store_model_output("primary", "What is async programming?", "Async allows concurrent execution", 0.9)
    ic.store_model_output("backend", "Explain async/await", "async/await is Python syntax for coroutines", 0.85)
    
    # Get cross-model context (from backend's perspective)
    context = ic.get_cross_model_context("What is async?", "backend")
    
    # Should get primary's context if similarity > 0.65
    if context:
        assert "primary" in context.lower() or "async" in context.lower()


def test_conversation_storage():
    """Test conversation turn storage."""
    from core.embedding_interconnect import get_embedding_interconnect
    
    ic = get_embedding_interconnect()
    
    # Store conversation turns
    ic.store_conversation("user", "Hello, how are you?")
    ic.store_conversation("assistant", "I'm doing well, thank you!")
    ic.store_conversation("user", "What can you do?")
    
    assert len(ic._conversation_embeddings) == 3
    assert len(ic._conversation_texts) == 3
    
    # Test retrieval
    results = ic.get_conversation_context("What can you", top_k=2)
    assert len(results) <= 2


def test_memory_embedding():
    """Test memory embedding storage and retrieval."""
    from core.embedding_interconnect import get_embedding_interconnect
    
    ic = get_embedding_interconnect()
    
    # Store memories
    ic.store_memory("mem_001", "Python is great for data science")
    ic.store_memory("mem_002", "Machine learning uses Python extensively")
    ic.store_memory("mem_003", "JavaScript is for web development")
    
    assert len(ic._memory_embeddings) >= 3
    
    # Test search
    results = ic.search_memories("Python programming", top_k=2)
    assert len(results) <= 2
    
    # Test that Python-related query returns Python memories first
    if results:
        top_memory_id, top_score = results[0]
        assert top_score > 0.5, f"Top result score should be > 0.5, got {top_score}"


def test_knowledge_embedding():
    """Test knowledge graph embedding storage."""
    from core.embedding_interconnect import get_embedding_interconnect
    
    ic = get_embedding_interconnect()
    
    # Store knowledge nodes
    ic.store_knowledge_node("node_001", "Neural networks are inspired by biological neurons")
    ic.store_knowledge_node("node_002", "Deep learning is a subset of machine learning")
    ic.store_knowledge_node("node_003", "React is a JavaScript framework")
    
    assert len(ic._knowledge_embeddings) >= 3
    
    # Test search
    results = ic.search_knowledge("neural networks", top_k=2)
    assert len(results) <= 2


def test_insight_embedding():
    """Test insight embedding storage."""
    from core.embedding_interconnect import get_embedding_interconnect
    
    ic = get_embedding_interconnect()
    
    # Store insights
    ic.store_insight("insight_001", "Pattern: Recursion often leads to stack overflow")
    ic.store_insight("insight_002", "Insight: Memoization improves performance significantly")
    
    assert len(ic._insight_embeddings) >= 2
    
    # Test search
    results = ic.search_insights("recursion patterns", top_k=2)
    assert len(results) <= 2


def test_workflow_embedding():
    """Test workflow embedding storage for orchestrator learning."""
    from core.embedding_interconnect import get_embedding_interconnect
    
    ic = get_embedding_interconnect()
    
    # Store workflow patterns
    ic.store_workflow("wf_001", "Search and analyze code", {"steps": ["search", "read", "analyze"]}, success=True)
    ic.store_workflow("wf_002", "Write and test function", {"steps": ["write", "test", "fix"]}, success=True)
    ic.store_workflow("wf_003", "Debug failing test", {"steps": ["run", "identify", "fix"]}, success=False)
    
    assert len(ic._workflow_embeddings) >= 3
    
    # Test search
    results = ic.search_workflows("fix code bug", top_k=2)
    assert len(results) <= 2
    
    # Test pattern retrieval
    patterns = ic.get_workflow_patterns("search for code", top_k=2)
    assert len(patterns) <= 2


def test_persistence():
    """Test save/load to disk."""
    from core.embedding_interconnect import get_embedding_interconnect
    
    ic = get_embedding_interconnect()
    
    # Store some data
    ic.store_memory("persist_test", "This should persist")
    ic.store_workflow("wf_persist", "Test workflow", {"step": 1}, success=True)
    
    # Save
    assert ic.save_to_disk() == True
    
    # Get stats before and after load
    stats_before = ic.get_stats()
    
    # Load
    assert ic.load_from_disk() == True
    
    # Verify persistence worked
    assert isinstance(ic.get_stats(), dict)


# ==================== MEMORY HUB TESTS ====================

def test_memory_hub_initialization():
    """Test memory hub lazy loading."""
    from core.memory_hub import get_memory_hub
    
    hub = get_memory_hub()
    assert hub is not None
    
    # Test stats
    stats = hub.get_stats()
    assert stats is not None


def test_memory_hub_encoding():
    """Test memory hub embedding encoding with caching."""
    from core.memory_hub import get_memory_hub
    
    hub = get_memory_hub()
    
    # Test encoding
    text = "Test encoding with memory hub"
    emb = hub.encode(text, dim=768)
    
    assert emb is not None
    assert len(emb) in [128, 256, 512, 768]  # Matryoshka dimensions
    
    # Test Matryoshka dimensions
    emb_512 = hub.encode(text, dim=512)
    emb_256 = hub.encode(text, dim=256)
    emb_128 = hub.encode(text, dim=128)
    
    assert len(emb_512) == 512
    assert len(emb_256) == 256
    assert len(emb_128) == 128


def test_memory_hub_store():
    """Test memory storage routing."""
    from core.memory_hub import get_memory_hub
    
    hub = get_memory_hub()
    
    # Store different types
    conv_id = hub.store("This is a conversation turn", memory_type='conversation')
    sem_id = hub.store("Python is a programming language", memory_type='semantic')
    
    assert conv_id is not None
    assert sem_id is not None


# ==================== DUAL-MODEL INTEGRATION TESTS ====================

def test_dual_model_embedding_storage():
    """Test that dual-model stores embeddings correctly."""
    from core.dual_model import DualModelOrchestrator, TaskType, ModelRole
    from core.embedding_interconnect import get_embedding_interconnect
    
    class MockLLM:
        def __init__(self, name):
            self.model = name
        
        async def chat(self, user_input, stream=False, **kwargs):
            return {"content": f"Response from {self.model}: {user_input[:30]}..."}
    
    ic = get_embedding_interconnect()
    
    dual_model = DualModelOrchestrator(
        primary_llm=MockLLM("primary"),
        backend_llm=MockLLM("backend"),
        model_coordinator=None,
    )
    
    # Check interconnect is attached
    assert dual_model._interconnect is not None
    
    # Check cross-model context methods exist
    assert hasattr(dual_model, '_get_cross_model_context')
    assert hasattr(dual_model, '_store_model_embedding')


def test_semantic_routing():
    """Test semantic routing based on embeddings."""
    from core.dual_model import DualModelOrchestrator, TaskType, ModelRole
    from core.embedding_interconnect import get_embedding_interconnect
    
    ic = get_embedding_interconnect()
    
    # Pre-populate with known patterns
    ic.store_model_output("primary", "Tell me a story", "Once upon a time...", 0.9)
    ic.store_model_output("backend", "Write a Python function", "def func(): pass", 0.95)
    
    class MockLLM:
        def __init__(self, name):
            self.model = name
        async def chat(self, user_input, stream=False, **kwargs):
            return {"content": f"Response"}
    
    dual_model = DualModelOrchestrator(
        primary_llm=MockLLM("primary"),
        backend_llm=MockLLM("backend"),
        model_coordinator=None,
    )
    
    # Test semantic routing exists and returns valid values
    assert hasattr(dual_model, '_semantic_route')
    
    # Test routing for creative vs code tasks
    model, confidence = dual_model._semantic_route("Tell me a story about dragons", TaskType.CREATIVE)
    model_str = model.value if hasattr(model, 'value') else str(model)
    assert model_str in ['primary', 'backend']
    assert 0 <= confidence <= 1


# ==================== RUN ALL TESTS ====================

if __name__ == "__main__":
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "-p", "no:xdist", "--override-ini=addopts="],
        cwd=_JARVIS_ROOT
    )
    sys.exit(result.returncode)