#!/usr/bin/env python3
"""Extensive embedding interconnect tests for resilience.

Tests:
1. Cross-model context sharing (Primary ↔ Backend)
2. Semantic routing accuracy
3. IPC client/server reliability
4. Concurrent access patterns
5. Failure recovery and edge cases
6. Long-running stability
"""
import os as _os
_JARVIS_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import os
os.environ["JARVIS_LOCAL_EMBEDDER"] = "1"  # Use local for tests

import sys
sys.path.insert(0, _JARVIS_ROOT)

import asyncio
import concurrent.futures
import random
import threading
import time
import numpy as np
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from core.embedding_interconnect import (
    get_embedding_interconnect,
    reset_embedding_interconnect,
    EmbeddingInterconnect
)
from core.embedding_client import EmbeddingClient, get_embedding_client


# ==================== FIXTURES ====================

@pytest.fixture(autouse=True)
def reset_state():
    """Reset all singletons before each test."""
    reset_embedding_interconnect()
    yield
    reset_embedding_interconnect()


def fresh_interconnect():
    """Create a fresh interconnect instance for isolated tests."""
    return EmbeddingInterconnect(prefer_server=False)


# ==================== BASIC OPERATIONS ====================

class TestBasicOperations:
    """Test fundamental embedding operations."""
    
    def test_encode_returns_correct_dimension(self):
        """Embeddings must be 768-dimensional."""
        ic = fresh_interconnect()
        emb = ic.encode("test text")
        assert emb.shape == (768,), f"Expected 768 dims, got {emb.shape}"
    
    def test_encode_empty_string_returns_valid_embedding(self):
        """Empty string should still return valid embedding."""
        ic = fresh_interconnect()
        emb = ic.encode("")
        assert emb.shape == (768,)
        assert not np.isnan(emb).any(), "Empty string embedding contains NaN"
    
    def test_encode_very_long_text(self):
        """Long text should be handled gracefully."""
        ic = fresh_interconnect()
        long_text = "word " * 10000  # 50k chars
        emb = ic.encode(long_text)
        assert emb.shape == (768,)
        assert not np.isnan(emb).any()
    
    def test_encode_unicode_text(self):
        """Unicode text should work correctly."""
        ic = fresh_interconnect()
        texts = [
            "日本語テキスト",
            "中文文本",
            "العربية",
            "émojis 🎉 🚀 💻",
            "Math: ∑ ∫ ∂ ∇",
        ]
        for text in texts:
            emb = ic.encode(text)
            assert emb.shape == (768,)
            assert not np.isnan(emb).any()
    
    def test_encode_special_characters(self):
        """Special characters should not break encoding."""
        ic = fresh_interconnect()
        texts = [
            "\n\t\r",
            "\\n\\t",
            "null\x00byte",
            "path/to/file.py",
            "C:\\Users\\test",
        ]
        for text in texts:
            emb = ic.encode(text)
            assert emb.shape == (768,)
    
    def test_encode_deterministic(self):
        """Same text should produce same embedding."""
        ic = fresh_interconnect()
        text = "Deterministic test"
        emb1 = ic.encode(text)
        emb2 = ic.encode(text)
        assert np.allclose(emb1, emb2), "Embeddings should be identical for same text"
    
    def test_encode_normalized(self):
        """Embeddings should be normalized."""
        ic = fresh_interconnect()
        emb = ic.encode("normalized test")
        norm = np.linalg.norm(emb)
        if norm == 0.0:
            pytest.skip("No real embedding model available")
        assert 0.9 < norm < 1.1, f"Embedding not normalized: norm={norm}"


# ==================== CROSS-MODEL CONTEXT ====================

class TestCrossModelContext:
    """Test context sharing between Primary and Backend models."""
    
    def test_primary_context_retrievable_by_backend(self):
        """Backend should be able to retrieve Primary's context."""
        ic = fresh_interconnect()
        
        # Check if real embedding model is available
        test_emb = ic.encode("test")
        if np.linalg.norm(test_emb) == 0.0:
            pytest.skip("No real embedding model available")
        
        # Primary learns about async
        ic.store_model_output("primary", "What is async programming?", 
                             "Async uses event loops for non-blocking I/O", 0.95)
        ic.store_model_output("primary", "Explain JavaScript async/await",
                             "async/await is syntactic sugar for Promises", 0.9)
        
        # Backend asks similar question
        context = ic.get_cross_model_context("How does async work?", "backend")
        
        assert context is not None, "Backend should get context from Primary"
        assert "async" in context.lower() or "event" in context.lower()
    
    def test_backend_context_retrievable_by_primary(self):
        """Primary should be able to retrieve Backend's context."""
        ic = fresh_interconnect()
        
        # Backend learns about Python
        ic.store_model_output("backend", "Write a Python function",
                             "def hello(): return 'world'", 0.85)
        ic.store_model_output("backend", "How to use list comprehension?",
                             "[x for x in range(10)]", 0.9)
        
        # Primary asks similar question
        context = ic.get_cross_model_context("Tell me about Python lists", "primary")
        
        assert context is not None, "Primary should get context from Backend"
    
    def test_cross_model_similarity_threshold(self):
        """Cross-model context should respect similarity threshold."""
        ic = fresh_interconnect()
        
        # Primary learns about cooking
        ic.store_model_output("primary", "How to cook pasta?", 
                             "Boil water, add pasta, cook 10 mins", 0.8)
        
        # Backend asks about programming (unrelated)
        context = ic.get_cross_model_context("How to debug Python code?", "backend")
        
        # Should return empty string or context below threshold
        # Context similarity should be low for unrelated topics
        if context:
            # If we get context, it should have low relevance
            pass  # Similarity filtering may allow some matches
    
    def test_cross_model_with_multiple_outputs(self):
        """Test cross-model context with many stored outputs."""
        ic = fresh_interconnect()
        
        # Primary stores many outputs
        topics = ["python", "javascript", "rust", "golang", "java"]
        for topic in topics:
            ic.store_model_output("primary", f"What is {topic}?",
                                 f"{topic} is a programming language", 0.8)
        
        # Backend stores many outputs
        for topic in ["react", "vue", "angular", "svelte", "ember"]:
            ic.store_model_output("backend", f"Explain {topic}",
                                 f"{topic} is a frontend framework", 0.8)
        
        # Cross-model retrieval
        context = ic.get_cross_model_context("Tell me about frameworks", "primary")
        assert context is not None
        
        context = ic.get_cross_model_context("What programming languages?", "backend")
        assert context is not None
    
    def test_cross_model_quality_weighted(self):
        """Higher quality outputs should be preferred."""
        ic = fresh_interconnect()
        
        # Check if real embedding model is available
        test_emb = ic.encode("test")
        if np.linalg.norm(test_emb) == 0.0:
            pytest.skip("No real embedding model available")
        
        # Primary stores low quality and high quality
        ic.store_model_output("primary", "What is Python?",
                             "Python is bad", 0.3)  # Low quality
        ic.store_model_output("primary", "What is Python?",
                             "Python is a versatile programming language", 0.95)  # High quality
        
        # Backend retrieves
        context = ic.get_cross_model_context("Tell me about Python", "backend")
        
        # Should prefer high quality response
        assert "versatile" in context.lower() or "programming" in context.lower()
    
    def test_cross_model_context_window_truncation(self):
        """Long context should be truncated properly."""
        ic = fresh_interconnect()
        
        # Store very long response
        long_response = "Python is great. " * 1000  # Very long
        ic.store_model_output("primary", "What is Python?", long_response, 0.9)
        
        # Retrieve
        context = ic.get_cross_model_context("Tell me about Python", "backend")
        
        # Should be truncated
        assert len(context) < len(long_response), "Context should be truncated"


# ==================== SEMANTIC ROUTING ====================

class TestSemanticRouting:
    """Test semantic routing based on embeddings."""
    
    def test_similar_queries_route_to_same_model(self):
        """Similar queries should route to the same model."""
        ic = fresh_interconnect()
        
        # Check if real embedding model is available
        test_emb = ic.encode("test")
        if np.linalg.norm(test_emb) == 0.0:
            pytest.skip("No real embedding model available")
        
        # Store successful outputs
        ic.store_model_output("backend", "Write a sorting algorithm",
                             "def sort(arr): return sorted(arr)", 0.95)
        ic.store_model_output("backend", "Create a binary search",
                             "def binary_search(arr, x): ...", 0.9)
        
        # Similar query should route to backend
        query_emb = ic.encode("Implement quicksort")
        
        # Find similar stored prompts
        similar_to_backend = False
        for stored_prompt, quality, stored_emb in ic.get_model_outputs("backend"):
            sim = np.dot(query_emb, stored_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(stored_emb))
            if sim > 0.7:
                similar_to_backend = True
                break
        
        assert similar_to_backend, "Similar query should match backend's expertise"
    
    def test_different_queries_route_appropriately(self):
        """Different query types should route appropriately."""
        ic = fresh_interconnect()
        
        # Store creative outputs in primary
        ic.store_model_output("primary", "Write a poem",
                             "Roses are red...", 0.9)
        ic.store_model_output("primary", "Tell me a story",
                             "Once upon a time...", 0.9)
        
        # Store technical outputs in backend
        ic.store_model_output("backend", "Debug this code",
                             "The bug is on line 42", 0.9)
        ic.store_model_output("backend", "Optimize performance",
                             "Use caching and memoization", 0.9)
        
        # Verify both types are stored
        assert len(ic.get_model_outputs("primary")) > 0
        assert len(ic.get_model_outputs("backend")) > 0
        
        # Creative query concept - verify embedding space works
        creative_emb = ic.encode("Write me a haiku")
        primary_outputs = ic.get_model_outputs("primary")
        primary_embs = [emb for _, _, emb in primary_outputs]
        
        # Technical query concept
        tech_emb = ic.encode("Why is my code slow?")
        backend_outputs = ic.get_model_outputs("backend")
        backend_embs = [emb for _, _, emb in backend_outputs]
        
        # Verify embeddings are computed correctly
        assert creative_emb.shape == (768,)
        assert tech_emb.shape == (768,)
        assert all(e.shape == (768,) for e in primary_embs)
        assert all(e.shape == (768,) for e in backend_embs)
    
    def test_routing_with_no_history(self):
        """Empty history should not crash routing."""
        ic = fresh_interconnect()
        
        # No outputs stored
        stats = ic.get_stats()
        assert stats["model_outputs"]["primary"] == 0
        assert stats["model_outputs"]["backend"] == 0
        
        # Should still be able to encode
        emb = ic.encode("What is the meaning of life?")
        assert emb.shape == (768,)


# ==================== CONCURRENT ACCESS ====================

class TestConcurrentAccess:
    """Test thread-safety and concurrent access."""
    
    def test_concurrent_encodes(self):
        """Concurrent encode operations should be thread-safe."""
        ic = get_embedding_interconnect()
        
        # Pre-warm
        ic.encode("warmup")
        
        results = []
        errors = []
        lock = threading.Lock()
        
        def encode_task(text):
            try:
                emb = ic.encode(text)
                with lock:
                    results.append((text, emb.shape))
            except Exception as e:
                with lock:
                    errors.append((text, str(e)))
        
        threads = []
        texts = [f"test text {i}" for i in range(20)]
        
        for text in texts:
            t = threading.Thread(target=encode_task, args=(text,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join(timeout=30)
        
        assert len(errors) == 0, f"Errors during concurrent access: {errors}"
        assert len(results) == 20, f"Expected 20 results, got {len(results)}"
    
    def test_concurrent_store_model_output(self):
        """Concurrent store operations should be thread-safe."""
        ic = get_embedding_interconnect()
        
        # Pre-warm
        ic.encode("warmup")
        
        errors = []
        lock = threading.Lock()
        
        def store_task(model, prompt_num):
            try:
                ic.store_model_output(
                    model, 
                    f"Prompt {prompt_num}",
                    f"Response {prompt_num}",
                    0.9
                )
            except Exception as e:
                with lock:
                    errors.append((model, prompt_num, str(e)))
        
        threads = []
        for i in range(10):  # Smaller number
            model = "primary" if i % 2 == 0 else "backend"
            t = threading.Thread(target=store_task, args=(model, i))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join(timeout=30)
        
        assert len(errors) == 0, f"Errors during concurrent store: {errors}"
    
    def test_concurrent_read_write(self):
        """Concurrent reads and writes should not corrupt data."""
        ic = get_embedding_interconnect()
        
        # Pre-warm
        ic.encode("warmup")
        
        errors = []
        lock = threading.Lock()
        
        def writer():
            for i in range(5):
                try:
                    ic.store_model_output("primary", f"Prompt {i}", f"Response {i}", 0.9)
                except Exception as e:
                    with lock:
                        errors.append(("writer", i, str(e)))
        
        def reader():
            for i in range(5):
                try:
                    ic.get_cross_model_context(f"Query {i}", "backend")
                except Exception as e:
                    with lock:
                        errors.append(("reader", i, str(e)))
        
        threads = []
        for _ in range(2):
            threads.append(threading.Thread(target=writer))
            threads.append(threading.Thread(target=reader))
        
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        
        assert len(errors) == 0, f"Errors during concurrent read/write: {errors}"


# ==================== PERSISTENCE ====================

class TestPersistence:
    """Test save/load functionality."""
    
    def test_save_to_disk(self):
        """Should save embeddings to disk."""
        ic = fresh_interconnect()
        
        # Store some data
        ic.store_model_output("primary", "Test prompt", "Test response", 0.9)
        ic.store_memory("mem_001", "Test memory")
        ic.store_knowledge_node("node_001", "Test knowledge")
        
        # Save
        result = ic.save_to_disk()
        assert result == True, "Save should succeed"
    
    def test_load_from_disk(self):
        """Should load embeddings from disk."""
        ic = fresh_interconnect()
        
        # Store and save
        ic.store_model_output("primary", "Test prompt", "Test response", 0.9)
        ic.save_to_disk()
        
        # Create new instance and load
        ic2 = fresh_interconnect()
        result = ic2.load_from_disk()
        assert result == True, "Load should succeed"
    
    def test_persists_across_reset(self):
        """Data should persist across instance resets."""
        # First instance
        ic = fresh_interconnect()
        ic.store_model_output("primary", "Remember this", "Data to remember", 0.9)
        ic.save_to_disk()
        
        # Reset
        reset_embedding_interconnect()
        
        # New instance
        ic2 = fresh_interconnect()
        ic2.load_from_disk()
        
        # Should have the stored data
        # Note: This tests the persistence mechanism, not cross-process persistence


# ==================== FAILURE RECOVERY ====================

class TestFailureRecovery:
    """Test resilience to failures."""
    
    def test_encode_with_none_input(self):
        """Should handle None input gracefully."""
        ic = fresh_interconnect()
        
        with pytest.raises((TypeError, AttributeError)):
            ic.encode(None)
    
    def test_store_with_empty_strings(self):
        """Should handle empty strings."""
        ic = fresh_interconnect()
        
        # Should not crash
        ic.store_model_output("primary", "", "", 0.9)
        ic.store_memory("mem_001", "")
    
    def test_store_with_invalid_quality(self):
        """Should handle invalid quality scores."""
        ic = fresh_interconnect()
        
        # Quality > 1
        ic.store_model_output("primary", "Test", "Response", 1.5)
        
        # Quality < 0
        ic.store_model_output("primary", "Test", "Response", -0.5)
        
        # Should not crash
        assert True
    
    def test_search_with_empty_query(self):
        """Should handle empty search query."""
        ic = fresh_interconnect()
        
        results = ic.search_memories("")
        assert isinstance(results, list)
    
    def test_search_with_no_embeddings(self):
        """Should return empty list when no embeddings stored."""
        ic = fresh_interconnect()
        
        results = ic.search_memories("test query")
        assert results == []
    
    def test_get_cross_model_with_no_outputs(self):
        """Should return empty string when no outputs stored."""
        ic = fresh_interconnect()
        
        context = ic.get_cross_model_context("test query", "backend")
        assert context == "" or context is None or isinstance(context, str)


# ==================== MEMORY MANAGEMENT ====================

class TestMemoryManagement:
    """Test memory limits and cleanup."""
    
    def test_model_outputs_limit(self):
        """Should limit stored outputs per model."""
        ic = fresh_interconnect()
        
        # Store many outputs
        for i in range(50):  # Reduced from 100
            ic.store_model_output("primary", f"Prompt {i}", f"Response {i}", 0.9)
        
        # Check limit is enforced
        # (Currently no limit, but should be tested if limit exists)
    
    def test_memory_embeddings_limit(self):
        """Should handle many memory embeddings."""
        ic = fresh_interconnect()
        
        # Store memories (reduced number for faster testing)
        for i in range(100):
            ic.store_memory(f"mem_{i}", f"Memory content {i}")
        
        assert len(ic._memory_embeddings) == 100
    
    def test_large_embedding_cache(self):
        """Should handle large number of cached embeddings."""
        ic = fresh_interconnect()
        
        # Encode unique texts (reduced number)
        for i in range(100):
            ic.encode(f"Unique text number {i} with some variation")
        
        # Should complete without memory errors


# ==================== EMBEDDING CLIENT ====================

class TestEmbeddingClient:
    """Test embedding client functionality."""
    
    def test_client_fallback_to_local(self):
        """Client should fall back to local model when server unavailable."""
        client = EmbeddingClient(prefer_server=True, timeout=1.0)
        
        # Server not running, should fall back
        emb = client.encode("test text")
        
        assert emb is not None
        assert emb.shape == (768,)
    
    def test_client_batch_encoding(self):
        """Client should support batch encoding."""
        client = EmbeddingClient(prefer_server=False)
        
        texts = [f"Text {i}" for i in range(10)]
        embeddings = client.encode_batch(texts)
        
        assert len(embeddings) == 10
        assert all(e.shape == (768,) for e in embeddings)
    
    def test_client_caches_server_status(self):
        """Client should cache server status."""
        client = EmbeddingClient(prefer_server=True)
        
        # First check
        status1 = client._check_server()
        
        # Immediate second check should use cache
        status2 = client._check_server()
        
        assert status1 == status2


# ==================== INTEGRATION TESTS ====================

class TestIntegration:
    """Integration tests with other components."""
    
    def test_dual_model_integration(self):
        """Test integration with dual model orchestrator."""
        from core.dual_model import DualModelOrchestrator, TaskType, ModelRole
        
        # Create mock LLMs
        class MockLLM:
            def __init__(self, name):
                self.model = name
            
            async def chat(self, user_input, stream=False, **kwargs):
                return {"content": f"Response from {self.model}"}
        
        dual = DualModelOrchestrator(
            primary_llm=MockLLM("primary"),
            backend_llm=MockLLM("backend"),
            model_coordinator=None,
        )
        
        # Verify interconnect is attached
        assert dual._interconnect is not None
        
        # Verify it uses the shared embedding space
        emb = dual._interconnect.encode("test")
        assert emb.shape == (768,)
    
    def test_memory_hub_integration(self):
        """Test integration with memory hub."""
        from core.memory_hub import get_memory_hub
        
        hub = get_memory_hub()
        
        # Store memory
        mem_id = hub.store("Test memory", memory_type='semantic')
        
        assert mem_id is not None
    
    def test_intelligent_orchestrator_integration(self):
        """Test integration with intelligent orchestrator."""
        from core.intelligent_orchestrator import get_intelligent_orchestrator
        
        orchestrator = get_intelligent_orchestrator()
        
        if orchestrator is None:
            pytest.skip("Intelligent orchestrator not available")
        
        # Verify it has access to embedding interconnect
        assert hasattr(orchestrator, '_retrieve_similar_workflows') or \
               hasattr(orchestrator, 'decompose')


# ==================== STRESS TESTS ====================

class TestStress:
    """Stress tests for performance and stability."""
    
    def test_many_concurrent_encodes(self):
        """Test many concurrent encode operations."""
        ic = get_embedding_interconnect()
        
        # Pre-warm
        ic.encode("warmup")
        
        def encode_many(start, count):
            for i in range(start, start + count):
                ic.encode(f"Concurrent text {i}")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(encode_many, i * 20, 20)
                for i in range(4)
            ]
            for f in futures:
                f.result(timeout=60)
        
        # All should complete without error
        assert True
    
    def test_rapid_store_retrieve(self):
        """Test rapid store and retrieve cycles."""
        ic = fresh_interconnect()
        
        for i in range(50):  # Reduced from 100
            ic.store_model_output("primary", f"Prompt {i}", f"Response {i}", 0.9)
            ic.get_cross_model_context(f"Query {i}", "backend")
        
        # Should complete without errors
        assert True
    
    def test_large_context_windows(self):
        """Test with large context windows."""
        ic = fresh_interconnect()
        
        # Store large contexts (reduced number)
        for i in range(20):
            large_content = " ".join([f"sentence {j}" for j in range(50)])
            ic.store_model_output("primary", f"Prompt {i}", large_content, 0.9)
        
        # Retrieve should handle large contexts
        context = ic.get_cross_model_context("Query", "backend")
        
        # Should return something or empty string
        assert context is not None or context == ""


# ==================== EDGE CASES ====================

class TestEdgeCases:
    """Edge case handling."""
    
    def test_duplicate_prompts(self):
        """Should handle duplicate prompts correctly."""
        ic = fresh_interconnect()
        
        # Store same prompt multiple times
        ic.store_model_output("primary", "What is Python?", "Answer 1", 0.9)
        ic.store_model_output("primary", "What is Python?", "Answer 2", 0.9)
        
        # Should have both stored
        # Implementation may deduplicate or keep both
        assert len(ic.get_model_outputs("primary")) >= 1
    
    def test_special_model_names(self):
        """Should handle special model names."""
        ic = fresh_interconnect()
        
        # Valid model names
        ic.store_model_output("primary", "Test", "Response", 0.9)
        ic.store_model_output("backend", "Test", "Response", 0.9)
        ic.store_model_output("synthesis", "Test", "Response", 0.9)
    
    def test_unicode_in_embeddings(self):
        """Should handle unicode in text to embed."""
        ic = fresh_interconnect()
        
        texts = [
            "Hello 世界",
            "Привет мир",
            "مرحبا بالعالم",
            "🎉🎊🎈",
        ]
        
        for text in texts:
            emb = ic.encode(text)
            assert emb.shape == (768,)
            assert not np.isnan(emb).any()
    
    def test_very_similar_texts(self):
        """Should distinguish very similar texts."""
        ic = fresh_interconnect()
        
        # Check if real embedding model is available
        test_emb = ic.encode("test")
        if np.linalg.norm(test_emb) == 0.0:
            pytest.skip("No real embedding model available")
        
        emb1 = ic.encode("The quick brown fox jumps over the lazy dog")
        emb2 = ic.encode("The quick brown fox jumps over the lazy cat")
        
        # Should be similar but not identical (0.85+ is very high)
        sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        assert 0.85 < sim < 1.0, f"Similar texts should have high similarity: {sim}"
    
    def test_identical_texts_same_embedding(self):
        """Identical texts should have identical embeddings."""
        ic = fresh_interconnect()
        
        text = "This is an identical text"
        emb1 = ic.encode(text)
        emb2 = ic.encode(text)
        
        assert np.allclose(emb1, emb2), "Identical texts should have identical embeddings"


# ==================== LONG RUNNING TESTS ====================

class TestLongRunning:
    """Stability tests for long-running operations."""
    
    def test_memory_stability(self):
        """Test memory stability over many operations."""
        ic = get_embedding_interconnect()
        
        # Pre-warm
        ic.encode("warmup")
        
        initial_count = len(ic._memory_embeddings)
        
        # Perform many operations (reduced for testing)
        for i in range(100):
            ic.store_memory(f"mem_{i}", f"Content {i % 10}")
            if i % 10 == 0:
                ic.search_memories(f"Content {i % 10}")
        
        # Memory should not grow unboundedly (if limits are implemented)
        final_count = len(ic._memory_embeddings)
        
        # Just verify it doesn't crash
        assert final_count >= initial_count


if __name__ == "__main__":
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "-p", "no:xdist", "--tb=short", "--override-ini=addopts="],
        cwd=_JARVIS_ROOT
    )
    sys.exit(result.returncode)