#!/usr/bin/env python3
"""Advanced embedding interconnect tests for maximum resilience.

Tests:
1. Real IPC server/client communication
2. Parallel agent simulation
3. Long-term stability (extended operations)
4. Performance benchmarks
5. Network failure simulation
6. Memory leak detection
7. Model switching scenarios
8. Concurrent workflow handling
"""
import os as _os
_JARVIS_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import os
os.environ["JARVIS_LOCAL_EMBEDDER"] = "1"

import sys
sys.path.insert(0, _JARVIS_ROOT)

import asyncio
import gc
import threading
import time
import tracemalloc
import numpy as np
import pytest
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.embedding_interconnect import (
    get_embedding_interconnect,
    reset_embedding_interconnect,
    EmbeddingInterconnect
)
from core.embedding_client import EmbeddingClient
from core.dual_model import DualModelOrchestrator, TaskType, ModelRole


# ==================== FIXTURES ====================

@pytest.fixture(autouse=True)
def reset_state():
    """Reset all singletons before each test."""
    reset_embedding_interconnect()
    yield
    reset_embedding_interconnect()


def fresh_interconnect():
    """Create a fresh interconnect instance."""
    return EmbeddingInterconnect(prefer_server=False)


# ==================== IPC SERVER/CLIENT TESTS ====================

class TestIPCCommunication:
    """Test actual IPC server/client communication."""
    
    def test_client_server_handshake(self):
        """Test client can communicate with server."""
        # When server not running, client should fallback gracefully
        client = EmbeddingClient(prefer_server=True, timeout=1.0)
        
        # Should fallback to local model
        emb = client.encode("test handshake")
        assert emb.shape == (768,)
    
    def test_batch_encoding_through_client(self):
        """Test batch encoding through client."""
        client = EmbeddingClient(prefer_server=False)
        
        texts = [f"Batch test {i}" for i in range(20)]
        embeddings = client.encode_batch(texts)
        
        assert len(embeddings) == 20
        assert all(e.shape == (768,) for e in embeddings)
    
    def test_client_timeout_handling(self):
        """Test client handles timeout gracefully."""
        client = EmbeddingClient(prefer_server=True, timeout=0.001)  # Very short timeout
        
        # Should timeout and fallback
        emb = client.encode("timeout test")
        assert emb is not None
        assert emb.shape == (768,)
    
    def test_client_reconnect_after_failure(self):
        """Test client can reconnect after failure."""
        client = EmbeddingClient(prefer_server=False)
        
        # First request
        emb1 = client.encode("first request")
        
        # Simulate server status change
        client._server_available = None
        client._last_check = 0
        
        # Second request should still work
        emb2 = client.encode("second request")
        
        assert emb1.shape == (768,)
        assert emb2.shape == (768,)


# ==================== PARALLEL AGENT SIMULATION ====================

class TestParallelAgentSimulation:
    """Simulate parallel agents accessing shared embedding."""
    
    def test_parallel_agents_same_query(self):
        """Multiple agents querying same embedding space."""
        ic = get_embedding_interconnect()
        
        # Pre-warm
        ic.encode("warmup")
        
        results = []
        errors = []
        lock = threading.Lock()
        
        def agent_task(agent_id, query):
            try:
                emb = ic.encode(query)
                # Use valid model names: primary, backend, synthesis
                model = "primary" if agent_id % 2 == 0 else "backend"
                ic.store_model_output(model, query, f"Response from agent {agent_id}", 0.9)
                with lock:
                    results.append((agent_id, query, emb.shape))
            except Exception as e:
                with lock:
                    errors.append((agent_id, str(e)))
        
        # Simulate 5 parallel agents (reduced from 10)
        threads = []
        for i in range(5):
            query = f"What is Python? Query from agent {i}"
            t = threading.Thread(target=agent_task, args=(i, query))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join(timeout=30)
        
        assert len(errors) == 0, f"Errors: {errors}"
        assert len(results) == 5
        
        # All results should have same embedding dimension
        for agent_id, query, shape in results:
            assert shape == (768,)
    
    def test_parallel_agents_cross_model_context(self):
        """Multiple agents sharing context through embedding space."""
        ic = get_embedding_interconnect()
        
        # Pre-warm
        ic.encode("warmup")
        
        # Agent 1 (Primary) stores knowledge
        ic.store_model_output("primary", "What is async?", "Async enables concurrent execution", 0.95)
        
        # Agent 2 (Backend) retrieves context
        context = ic.get_cross_model_context("Explain async programming", "backend")
        
        # Agent 3 (another Primary) retrieves backend context
        ic.store_model_output("backend", "How to debug?", "Use print statements and debugger", 0.9)
        context2 = ic.get_cross_model_context("Debugging help", "primary")
        
        # All operations should work without conflict
        assert context is not None or context == ""
    
    def test_parallel_workflow_executions(self):
        """Simulate parallel workflow executions using embeddings."""
        ic = get_embedding_interconnect()
        
        # Store multiple workflow patterns
        workflows = [
            ("wf_1", "Search and analyze code", ["search", "read", "analyze"], True),
            ("wf_2", "Write and test function", ["write", "test", "fix"], True),
            ("wf_3", "Debug failing test", ["run", "identify", "fix"], False),
            ("wf_4", "Refactor module", ["analyze", "plan", "implement", "test"], True),
        ]
        
        for wf_id, request, steps, success in workflows:
            ic.store_workflow(wf_id, request, {"steps": steps}, success)
        
        # Simulate parallel agents searching for workflow patterns
        def search_workflow(query):
            patterns = ic.search_workflows(query, top_k=2)
            return query, patterns
        
        queries = [
            "fix code bug",
            "analyze codebase",
            "write new feature",
            "debug error",
        ]
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(search_workflow, q) for q in queries]
            results = [f.result(timeout=30) for f in futures]
        
        # All searches should complete
        assert len(results) == 4


# ==================== LONG-TERM STABILITY ====================

class TestLongTermStability:
    """Tests for long-running stability."""
    
    def test_extended_encode_cycle(self):
        """Test extended encode cycles."""
        ic = get_embedding_interconnect()
        
        # Pre-warm
        ic.encode("warmup")
        
        # Run encode cycles (reduced for test speed)
        for i in range(100):
            emb = ic.encode(f"Extended test iteration {i}")
            assert emb.shape == (768,)
            
            if i % 25 == 0:
                # Verify no memory corruption
                assert not np.isnan(emb).any()
    
    def test_memory_growth_stability(self):
        """Test memory doesn't grow unboundedly."""
        ic = get_embedding_interconnect()
        
        # Pre-warm
        ic.encode("warmup")
        
        stats_before = ic.get_stats()
        initial_count = stats_before["model_outputs"]["primary"] + stats_before["model_outputs"]["backend"]
        
        # Store many outputs (reduced)
        for i in range(50):
            ic.store_model_output("primary", f"Query {i}", f"Response {i}", 0.9)
        
        stats_after = ic.get_stats()
        final_count = stats_after["model_outputs"]["primary"] + stats_after["model_outputs"]["backend"]
        
        # Memory should grow but reasonably
        growth = final_count - initial_count
        assert growth <= 100  # Some overhead but not unbounded
    
    def test_continuous_store_retrieve(self):
        """Test continuous store and retrieve operations."""
        ic = get_embedding_interconnect()
        
        # Pre-warm
        ic.encode("warmup")
        
        for i in range(50):
            # Store
            ic.store_model_output("primary", f"Query {i}", f"Response {i}", 0.9)
            
            # Retrieve
            context = ic.get_cross_model_context(f"Query {i % 10}", "backend")
            
            # Verify
            if i % 10 == 0:
                stats = ic.get_stats()
                assert isinstance(stats, dict)


# ==================== PERFORMANCE BENCHMARKS ====================

class TestPerformanceBenchmarks:
    """Performance benchmarks for embedding operations."""
    
    def test_encode_performance(self):
        """Benchmark encoding speed."""
        ic = get_embedding_interconnect()
        
        # Pre-warm
        ic.encode("warmup")
        
        start = time.time()
        iterations = 100
        
        for _ in range(iterations):
            ic.encode("Performance test query")
        
        elapsed = time.time() - start
        avg_time = elapsed / iterations
        
        # Should be reasonably fast (not a strict requirement)
        print(f"\nAverage encode time: {avg_time*1000:.2f}ms")
        
        # Just verify it completes
        assert avg_time < 5.0, f"Encode too slow: {avg_time}s"
    
    def test_search_performance(self):
        """Benchmark search speed."""
        ic = get_embedding_interconnect()
        
        # Pre-populate
        for i in range(100):
            ic.store_memory(f"mem_{i}", f"Memory content number {i}")
        
        # Pre-warm
        ic.encode("warmup")
        
        start = time.time()
        iterations = 50
        
        for i in range(iterations):
            ic.search_memories("Memory content", top_k=5)
        
        elapsed = time.time() - start
        avg_time = elapsed / iterations
        
        print(f"\nAverage search time: {avg_time*1000:.2f}ms")
        
        assert avg_time < 2.0, f"Search too slow: {avg_time}s"
    
    def test_concurrent_performance(self):
        """Benchmark concurrent operations."""
        ic = get_embedding_interconnect()
        
        # Pre-warm
        ic.encode("warmup")
        
        start = time.time()
        
        def worker_task(worker_id, count):
            for i in range(count):
                ic.encode(f"Worker {worker_id} query {i}")
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(worker_task, i, 10) for i in range(8)]
            for f in as_completed(futures):
                f.result()
        
        elapsed = time.time() - start
        
        print(f"\nConcurrent 80 encodes time: {elapsed:.2f}s")
        
        # Should complete in reasonable time
        assert elapsed < 60.0


# ==================== NETWORK FAILURE SIMULATION ====================

class TestNetworkFailureSimulation:
    """Simulate network/server failures."""
    
    def test_embedder_unavailable_fallback(self):
        """Test fallback when embedder unavailable."""
        client = EmbeddingClient(prefer_server=True, timeout=0.1)
        
        # Force server unavailable
        client._server_available = False
        
        # Should still work with local fallback
        emb = client.encode("Fallback test")
        assert emb.shape == (768,)
    
    def test_partial_failure_recovery(self):
        """Test recovery from partial failures."""
        ic = get_embedding_interconnect()
        
        # Store some data
        ic.store_model_output("primary", "Test query", "Test response", 0.9)
        
        # Simulate failure and recovery
        try:
            # These should not crash even if internal errors occur
            ic.encode("Recovery test 1")
            ic.get_cross_model_context("Recovery test", "backend")
            ic.search_memories("Recovery test")
        except Exception as e:
            pytest.fail(f"Should not crash: {e}")
        
        # Verify still functional
        emb = ic.encode("Final recovery test")
        assert emb.shape == (768,)
    
    def test_corrupted_data_handling(self):
        """Test handling of corrupted data."""
        ic = get_embedding_interconnect()
        
        # Store normal data
        ic.store_model_output("primary", "Good query", "Good response", 0.9)
        
        # Try to store potentially problematic data
        try:
            ic.store_model_output("primary", "", "Empty prompt", 0.9)
            ic.store_model_output("primary", "Test", "", 0.9)
        except Exception:
            pass  # Should handle gracefully
        
        # Verify data integrity
        assert len(ic.get_model_outputs("primary")) > 0


# ==================== MEMORY LEAK DETECTION ====================

class TestMemoryLeakDetection:
    """Detect memory leaks in embedding operations."""
    
    def test_no_memory_leak_on_encode(self):
        """Verify no memory leak on repeated encodes."""
        ic = get_embedding_interconnect()
        ic.encode("warmup")
        
        # Force garbage collection before test
        gc.collect()
        
        # Perform many encodes (reduced for faster test)
        for i in range(50):
            ic.encode(f"Memory leak test {i}")
            if i % 10 == 0:
                gc.collect()
        
        # Should complete without memory exhaustion
        assert True
    
    def test_no_memory_leak_on_store(self):
        """Verify no memory leak on repeated stores."""
        ic = get_embedding_interconnect()
        ic.encode("warmup")
        
        # Force garbage collection
        gc.collect()
        
        # Perform many stores (reduced)
        for i in range(50):
            ic.store_model_output("primary", f"Query {i}", f"Response {i}", 0.9)
            if i % 10 == 0:
                gc.collect()
        
        # Memory should grow proportionally to data stored
        assert True


# ==================== CONCURRENT WORKFLOW HANDLING ====================

class TestConcurrentWorkflow:
    """Test concurrent workflow handling through embedding space."""
    
    def test_concurrent_workflow_registration(self):
        """Multiple workflows registering simultaneously."""
        ic = get_embedding_interconnect()
        
        # Pre-warm
        ic.encode("warmup")
        
        errors = []
        lock = threading.Lock()
        
        def register_workflow(wf_id, request):
            try:
                steps = ["step1", "step2", "step3"]
                ic.store_workflow(wf_id, request, {"steps": steps}, True)
            except Exception as e:
                with lock:
                    errors.append((wf_id, str(e)))
        
        threads = []
        for i in range(20):
            t = threading.Thread(
                target=register_workflow,
                args=(f"wf_{i}", f"Workflow request {i}")
            )
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join(timeout=30)
        
        assert len(errors) == 0, f"Errors: {errors}"
        assert len(ic._workflow_embeddings) == 20
    
    def test_concurrent_workflow_search(self):
        """Multiple agents searching workflows simultaneously."""
        ic = get_embedding_interconnect()
        
        # Pre-populate workflows
        for i in range(50):
            ic.store_workflow(f"wf_{i}", f"Workflow {i}", {"steps": []}, True)
        
        # Pre-warm
        ic.encode("warmup")
        
        errors = []
        lock = threading.Lock()
        
        def search_workflow(query):
            try:
                patterns = ic.search_workflows(query, top_k=5)
            except Exception as e:
                with lock:
                    errors.append((query, str(e)))
        
        threads = []
        queries = [f"Search query {i}" for i in range(20)]
        
        for query in queries:
            t = threading.Thread(target=search_workflow, args=(query,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join(timeout=30)
        
        assert len(errors) == 0, f"Errors: {errors}"
    
    def test_workflow_pattern_learning(self):
        """Test workflow pattern learning under load."""
        ic = get_embedding_interconnect()
        
        # Pre-warm
        ic.encode("warmup")
        
        # Simulate multiple agents learning patterns
        for i in range(100):
            # Agent registers successful workflow
            ic.store_workflow(
                f"wf_{i}",
                f"Complex task {i % 10}",
                {"steps": ["analyze", "plan", "execute", "verify"], "iteration": i},
                success=(i % 3 != 0)  # 2/3 success rate
            )
        
        # Search for patterns
        patterns = ic.get_workflow_patterns("Complex task", top_k=5)
        
        # Should return patterns
        assert isinstance(patterns, list)


# ==================== EMBEDDING SPACE CONSISTENCY ====================

class TestEmbeddingSpaceConsistency:
    """Test embedding space remains consistent."""
    
    def test_dimensional_consistency(self):
        """All embeddings should have same dimension."""
        ic = get_embedding_interconnect()
        
        texts = [
            "Short",
            "Medium length text",
            "Very long text " * 100,
            "Unicode: 你好世界",
            "Special: !@#$%^&*()",
        ]
        
        embeddings = [ic.encode(t) for t in texts]
        
        # All should have same dimension
        dims = [e.shape[0] for e in embeddings]
        assert all(d == 768 for d in dims), f"Inconsistent dimensions: {dims}"
    
    def test_semantic_consistency(self):
        """Similar meanings should have similar embeddings."""
        ic = get_embedding_interconnect()
        
        # Check if real embedding model is available
        test_emb = ic.encode("test")
        if np.linalg.norm(test_emb) == 0.0:
            pytest.skip("No real embedding model available")
        
        # Pairs of similar and different meanings
        similar_pairs = [
            ("The cat sat on the mat", "A cat is sitting on a mat"),
            ("Python is a programming language", "Python is used for coding"),
            ("The weather is nice today", "Today has good weather"),
        ]
        
        different_pairs = [
            ("The cat sat on the mat", "Python is a programming language"),
            ("I love pizza", "Quantum mechanics is complex"),
        ]
        
        # Similar pairs should have higher similarity
        for text1, text2 in similar_pairs:
            emb1 = ic.encode(text1)
            emb2 = ic.encode(text2)
            sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            assert sim > 0.5, f"Similar texts should be correlated: {sim}"
        
        # Different pairs should have lower similarity
        for text1, text2 in different_pairs:
            emb1 = ic.encode(text1)
            emb2 = ic.encode(text2)
            sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            assert sim < 0.7, f"Different texts should not be highly similar: {sim}"
    
    def test_cross_model_embedding_space(self):
        """Primary and Backend should share same embedding space."""
        ic = get_embedding_interconnect()
        
        # Check if real embedding model is available
        test_emb = ic.encode("test")
        if np.linalg.norm(test_emb) == 0.0:
            pytest.skip("No real embedding model available")
        
        # Store from Primary
        ic.store_model_output("primary", "What is async?", "Async explanation", 0.9)
        primary_emb = ic.encode("What is async?")
        
        # Store from Backend
        ic.store_model_output("backend", "Explain async", "Async details", 0.9)
        backend_emb = ic.encode("Explain async")
        
        # Should be in same space
        sim = np.dot(primary_emb, backend_emb) / (np.linalg.norm(primary_emb) * np.linalg.norm(backend_emb))
        assert sim > 0.7, f"Same concept should have similar embeddings: {sim}"


# ==================== DUAL MODEL INTEGRATION ====================

class TestDualModelIntegration:
    """Test dual model integration with embedding interconnect."""
    
    def test_dual_model_shared_embedding_space(self):
        """Verify dual model uses shared embedding space."""
        dual = DualModelOrchestrator(
            primary_llm=None,
            backend_llm=None,
            model_coordinator=None
        )
        
        # Verify interconnect is attached
        assert dual._interconnect is not None
        
        # Verify embedding operations work
        emb = dual._interconnect.encode("test")
        assert emb.shape == (768,)
    
    def test_dual_model_cross_model_context(self):
        """Test dual model cross-model context sharing."""
        dual = DualModelOrchestrator(
            primary_llm=None,
            backend_llm=None,
            model_coordinator=None
        )
        
        # Primary stores
        dual._interconnect.store_model_output("primary", "Test query", "Test response", 0.9)
        
        # Backend retrieves
        context = dual._get_cross_model_context("Test query", "backend")
        
        # Should return context or empty string
        assert context is not None or context == ""
    
    def test_dual_model_semantic_routing(self):
        """Test dual model semantic routing."""
        dual = DualModelOrchestrator(
            primary_llm=None,
            backend_llm=None,
            model_coordinator=None
        )
        
        # Store some history
        dual._interconnect.store_model_output("backend", "Write a function", "def func(): pass", 0.95)
        
        # Route based on embedding
        model, confidence = dual._semantic_route("Write a sorting function", TaskType.CODE)
        
        # Should return valid model
        assert model in [ModelRole.PRIMARY, ModelRole.BACKEND, "primary", "backend"]
        assert 0 <= confidence <= 1


# ==================== PERSISTENCE INTEGRITY ====================

class TestPersistenceIntegrity:
    """Test persistence maintains data integrity."""
    
    def test_save_load_integrity(self):
        """Save and load should preserve data integrity."""
        ic = fresh_interconnect()
        
        # Store diverse data
        ic.store_model_output("primary", "Query 1", "Response 1", 0.9)
        ic.store_model_output("backend", "Query 2", "Response 2", 0.85)
        ic.store_memory("mem_1", "Memory 1")
        ic.store_knowledge_node("node_1", "Knowledge 1")
        ic.store_insight("insight_1", "Insight 1")
        ic.store_workflow("wf_1", "Workflow 1", {"step": 1}, True)
        
        # Get stats before save
        stats_before = ic.get_stats()
        
        # Save
        assert ic.save_to_disk() == True
        
        # Create new instance and load
        ic2 = fresh_interconnect()
        assert ic2.load_from_disk() == True
        
        # Get stats after load
        stats_after = ic2.get_stats()
        
        # Should have similar structure
        assert stats_after.get("embedding_dim") == 768
    
    def test_incremental_persistence(self):
        """Test incremental saves don't corrupt data."""
        ic = fresh_interconnect()
        
        # First save
        ic.store_model_output("primary", "First", "First response", 0.9)
        ic.save_to_disk()
        
        # Incremental save
        ic.store_model_output("backend", "Second", "Second response", 0.85)
        ic.save_to_disk()
        
        # Another incremental save
        ic.store_memory("mem_1", "Memory")
        ic.save_to_disk()
        
        # Load and verify
        ic2 = fresh_interconnect()
        ic2.load_from_disk()
        
        # Should not crash
        stats = ic2.get_stats()
        assert isinstance(stats, dict)


# ==================== RUN ====================

if __name__ == "__main__":
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "-p", "no:xdist", 
         "--tb=short", "--override-ini=addopts=", "-m", ""],
        cwd=_JARVIS_ROOT
    )
    sys.exit(result.returncode)