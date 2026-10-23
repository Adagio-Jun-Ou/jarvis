#!/usr/bin/env python3
"""Test embedding IPC server/client."""
import os as _os
_JARVIS_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import os
os.environ["JARVIS_LOCAL_EMBEDDER"] = "1"  # Use local for initial tests

import sys
sys.path.insert(0, _JARVIS_ROOT)

import time
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def reset_interconnect():
    """Reset interconnect before each test."""
    from core.embedding_interconnect import reset_embedding_interconnect
    reset_embedding_interconnect()
    yield
    reset_embedding_interconnect()


def test_local_embedder():
    """Test that local embedder works (fallback mode)."""
    from core.embedding_interconnect import get_embedding_interconnect
    
    ic = get_embedding_interconnect()
    
    # Test encoding
    emb = ic.encode("Test text for local embedder")
    
    assert emb is not None
    assert len(emb) == 768
    assert isinstance(emb, np.ndarray)
    assert not ic._using_ipc  # Should be local mode


def test_embedding_client_fallback():
    """Test embedding client falls back to local model when server unavailable."""
    from core.embedding_client import EmbeddingClient
    
    client = EmbeddingClient(prefer_server=True, timeout=1.0)
    
    # Server not running, should fall back to local
    emb = client.encode("Test fallback encoding")
    
    assert emb is not None
    assert len(emb) == 768
    assert isinstance(emb, np.ndarray)


def test_embedding_batch():
    """Test batch encoding."""
    from core.embedding_client import EmbeddingClient
    
    client = EmbeddingClient(prefer_server=False)  # Force local
    
    texts = [
        "First test text",
        "Second test text",
        "Third test text"
    ]
    
    embeddings = client.encode_batch(texts)
    
    assert len(embeddings) == 3
    assert all(len(e) == 768 for e in embeddings)
    assert all(isinstance(e, np.ndarray) for e in embeddings)


def test_embedding_consistency():
    """Test that same text produces same embedding."""
    from core.embedding_client import EmbeddingClient
    
    client = EmbeddingClient(prefer_server=False)
    
    text = "Consistency test text"
    emb1 = client.encode(text)
    emb2 = client.encode(text)
    
    # Should be identical
    assert np.allclose(emb1, emb2)


def test_embedding_similarity():
    """Test that similar texts have higher similarity."""
    from core.embedding_client import EmbeddingClient
    
    client = EmbeddingClient(prefer_server=False)
    
    text1 = "Python is a programming language"
    text2 = "Python is used for programming"
    text3 = "The weather is sunny today"
    
    emb1 = client.encode(text1)
    emb2 = client.encode(text2)
    emb3 = client.encode(text3)
    
    # Skip if no real embedding model is loaded (zero vectors)
    if np.linalg.norm(emb1) == 0.0:
        pytest.skip("No real embedding model available")
    
    # Compute cosine similarity
    sim_1_2 = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    sim_1_3 = np.dot(emb1, emb3) / (np.linalg.norm(emb1) * np.linalg.norm(emb3))
    
    # Similar texts should have higher similarity
    assert sim_1_2 > sim_1_3


def test_interconnect_with_local():
    """Test embedding interconnect with local mode."""
    from core.embedding_interconnect import get_embedding_interconnect
    
    ic = get_embedding_interconnect()
    
    # Test all embedding operations
    ic.store_model_output("primary", "What is AI?", "AI stands for artificial intelligence", 0.9)
    ic.store_memory("mem_001", "Test memory")
    ic.store_knowledge_node("node_001", "Test knowledge")
    ic.store_insight("insight_001", "Test insight")
    ic.store_workflow("wf_001", "Test workflow", {"step": 1}, success=True)
    
    # Verify storage
    assert len(ic.get_model_outputs("primary")) == 1
    assert len(ic._memory_embeddings) == 1
    assert len(ic._knowledge_embeddings) == 1
    assert len(ic._insight_embeddings) == 1
    assert len(ic._workflow_embeddings) == 1


if __name__ == "__main__":
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "-p", "no:xdist", "--override-ini=addopts="],
        cwd=_JARVIS_ROOT
    )
    sys.exit(result.returncode)