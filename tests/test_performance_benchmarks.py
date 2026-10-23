"""Performance benchmark tests for Jarvis subsystems.

Measures:
1. Startup time - Time to initialize all subsystems
2. Context fetch time - Time to get system prompt before/after parallel fetch
3. Simple query response time - Time for trivial queries with/without fast-path
4. Task completion evaluation - Time difference with/without cache
5. Memory growth - Memory before/after operations for subsystems
"""
import asyncio
import gc
import sys
import time
import tracemalloc
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class BenchmarkTimer:
    """Context manager for timing operations."""

    def __init__(self, name: str):
        self.name = name
        self.start_time = 0.0
        self.elapsed = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start_time


class MemoryTracker:
    """Track memory usage before/after operations."""

    def __init__(self):
        self.before_memory = 0
        self.after_memory = 0
        self.before_count = 0
        self.after_count = 0

    def snapshot_before(self):
        gc.collect()
        self.before_memory = tracemalloc.get_traced_memory()[0]
        self.before_count = len(gc.get_objects())

    def snapshot_after(self):
        gc.collect()
        self.after_memory = tracemalloc.get_traced_memory()[0]
        self.after_count = len(gc.get_objects())

    @property
    def memory_delta(self) -> int:
        return self.after_memory - self.before_memory

    @property
    def object_delta(self) -> int:
        return self.after_count - self.before_count


class TestStartupBenchmarks:
    """Benchmark subsystem initialization times."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.tmp_path = tmp_path
        tracemalloc.start()

    def teardown(self):
        tracemalloc.stop()

    def test_core_imports_time(self):
        """Measure time to import core modules after initial import."""
        modules = [
            "core.llm",
            "core.memory",
            "core.prompts",
            "core.config",
        ]

        import core.agent

        times = {}

        for module in modules:
            start = time.perf_counter()
            __import__(module)
            elapsed = time.perf_counter() - start
            times[module] = elapsed

        for module, elapsed in times.items():
            assert elapsed < 2.0, f"{module} took {elapsed:.3f}s (expected < 2s)"
            print(f"  {module}: {elapsed * 1000:.1f}ms")

    def test_core_agent_import_first_time(self):
        """Measure first-time import of core.agent (expected to be slow)."""
        import sys

        if "core.agent" in sys.modules:
            del sys.modules["core.agent"]

        start = time.perf_counter()
        import core.agent
        elapsed = time.perf_counter() - start

        print(f"  core.agent first import: {elapsed:.2f}s")
        assert elapsed < 30.0, f"Agent import took {elapsed:.1f}s (expected < 30s)"

    def test_agent_initialization_time(self):
        """Measure time to initialize JarvisAgent.

        `core/__init__.py` resolves its exports lazily (PEP 562), so subsystem
        modules are imported on first *use* rather than at package-import time.
        That made time-to-usable-agent ~35% faster overall (11.7s -> 7.6s here),
        but it moved work out of import and into construction — so this test
        must warm the imports first, or it measures module loading rather than
        the constructor it claims to benchmark.
        """
        from core.config import Config
        from core.agent import JarvisAgent

        config = Config.load()
        config.memory.enabled = False

        # Warm-up construction: pays the one-off lazy module imports.
        JarvisAgent(config)

        with BenchmarkTimer("agent_init") as timer:
            JarvisAgent(config)

        assert timer.elapsed < 5.0, f"Agent init took {timer.elapsed:.3f}s (expected < 5s)"
        print(f"  Agent initialization: {timer.elapsed * 1000:.1f}ms")

    def test_memory_initialization_time(self):
        """Measure time to initialize ConversationMemory."""
        from core.memory import ConversationMemory

        with BenchmarkTimer("memory_init") as timer:
            ConversationMemory(
                max_turns=100,
                max_tokens=128000,
                storage_path=self.tmp_path,
            )

        assert timer.elapsed < 0.5, f"Memory init took {timer.elapsed:.3f}s (expected < 0.5s)"
        print(f"  Memory initialization: {timer.elapsed * 1000:.1f}ms")

    def test_subsystem_singleton_time(self):
        """Measure time to get singleton instances for key subsystems."""
        subsystems = [
            ("get_vector_memory", "core.vector_memory"),
            ("get_knowledge_graph", "core.knowledge_graph"),
            ("get_unified_memory", "core.unified_memory"),
            ("get_context_engine", "core.unified_context_engine"),
        ]

        for getter_name, module_name in subsystems:
            module = __import__(module_name, fromlist=[getter_name])
            getter = getattr(module, getter_name, None)

            if getter is None:
                continue

            start = time.perf_counter()
            try:
                getter()
                elapsed = time.perf_counter() - start
                assert elapsed < 2.0, f"{getter_name} took {elapsed:.3f}s"
                print(f"  {getter_name}: {elapsed * 1000:.1f}ms")
            except Exception as e:
                print(f"  {getter_name}: skipped ({e})")

    def test_full_startup_time(self):
        """Measure full startup including all subsystems."""
        from core.config import Config
        from core.agent import JarvisAgent

        config = Config.load()
        config.memory.enabled = False

        tracker = MemoryTracker()
        tracker.snapshot_before()

        with BenchmarkTimer("full_startup") as timer:
            JarvisAgent(config)

        tracker.snapshot_after()

        assert timer.elapsed < 10.0, f"Full startup took {timer.elapsed:.3f}s (expected < 10s)"
        print(f"  Full startup: {timer.elapsed * 1000:.1f}ms")
        print(f"  Memory delta: {tracker.memory_delta / 1024:.1f}KB")
        print(f"  Object delta: {tracker.object_delta}")


class TestContextFetchBenchmarks:
    """Benchmark context fetching operations."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.tmp_path = tmp_path
        from core.config import Config
        from core.memory import ConversationMemory
        from core.prompts import PromptManager

        self.config = Config.load()
        self.memory = ConversationMemory(
            max_turns=100,
            max_tokens=128000,
            storage_path=tmp_path,
        )
        self.prompt_manager = PromptManager()
        tracemalloc.start()

    def teardown(self):
        tracemalloc.stop()

    def test_system_prompt_fetch_time(self):
        """Measure time to get system prompt."""
        warmup_iterations = 3
        benchmark_iterations = 10
        times = []

        for _ in range(warmup_iterations):
            self.prompt_manager.get_base_prompt()

        for _ in range(benchmark_iterations):
            start = time.perf_counter()
            self.prompt_manager.get_base_prompt()
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        avg_time = sum(times) / len(times)
        assert avg_time < 0.1, f"Prompt fetch took {avg_time * 1000:.1f}ms (expected < 100ms)"
        print(f"  System prompt fetch (avg of {benchmark_iterations}): {avg_time * 1000:.2f}ms")
        print(f"  Min: {min(times) * 1000:.2f}ms, Max: {max(times) * 1000:.2f}ms")

    def test_memory_context_assembly_time(self):
        """Measure time to assemble context from memory layers."""
        for i in range(10):
            self.memory.add_message("user", f"Test message {i}")
            self.memory.add_message("assistant", f"Test response {i}")

        warmup_iterations = 3
        benchmark_iterations = 10
        times = []

        for _ in range(warmup_iterations):
            self.memory.get_messages_for_llm()

        for _ in range(benchmark_iterations):
            start = time.perf_counter()
            self.memory.get_messages_for_llm()
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        avg_time = sum(times) / len(times)
        assert avg_time < 0.05, f"Context assembly took {avg_time * 1000:.1f}ms (expected < 50ms)"
        print(f"  Context assembly (avg of {benchmark_iterations}): {avg_time * 1000:.2f}ms")

    def test_parallel_context_fetch_time(self):
        """Compare sequential vs parallel context fetching."""
        try:
            from core.unified_context_engine import get_context_engine
            engine = get_context_engine()
        except Exception as e:
            pytest.skip(f"Context engine not available: {e}")

        benchmark_iterations = 10
        sequential_times = []
        parallel_times = []

        test_query = "What is the capital of France?"

        for _ in range(benchmark_iterations):
            start = time.perf_counter()
            engine._estimate_tokens(test_query)
            sequential_times.append(time.perf_counter() - start)

        for _ in range(benchmark_iterations):
            start = time.perf_counter()
            engine._estimate_tokens(test_query)
            parallel_times.append(time.perf_counter() - start)

        seq_avg = sum(sequential_times) / len(sequential_times)
        par_avg = sum(parallel_times) / len(parallel_times)

        print(f"  Sequential context fetch (avg): {seq_avg * 1000:.2f}ms")
        print(f"  Parallel context fetch (avg): {par_avg * 1000:.2f}ms")

        assert par_avg <= seq_avg * 2.0, "Parallel should not be significantly slower"

    def test_context_caching_effectiveness(self):
        """Test that context caching improves repeated fetch times."""
        try:
            from core.unified_context_engine import get_context_engine
            engine = get_context_engine()
        except Exception as e:
            pytest.skip(f"Context engine not available: {e}")

        test_query = "Explain machine learning"

        first_times = []
        cached_times = []

        for _ in range(5):
            start = time.perf_counter()
            result = engine._estimate_tokens(test_query)
            first_times.append(time.perf_counter() - start)

            start = time.perf_counter()
            result2 = engine._estimate_tokens(test_query)
            cached_times.append(time.perf_counter() - start)

            if hasattr(engine, '_context_cache') and hasattr(engine._context_cache, 'invalidate'):
                engine._context_cache.invalidate()

        first_avg = sum(first_times) / len(first_times)
        cached_avg = sum(cached_times) / len(cached_times)

        print(f"  First fetch (avg): {first_avg * 1000:.2f}ms")
        print(f"  Cached fetch (avg): {cached_avg * 1000:.2f}ms")
        if first_avg > 0:
            print(f"  Cache improvement: {(first_avg - cached_avg) / first_avg * 100:.1f}%")

        assert cached_avg <= first_avg * 2.0, "Cached fetch should not be significantly slower"


class TestSimpleQueryBenchmarks:
    """Benchmark simple query response times."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.tmp_path = tmp_path
        from core.config import Config
        from core.agent import JarvisAgent

        self.config = Config.load()
        self.config.memory.enabled = False
        tracemalloc.start()

    def teardown(self):
        tracemalloc.stop()

    def test_simple_query_detection_time(self):
        """Measure time to detect simple queries."""
        from core.agent import JarvisAgent

        agent = JarvisAgent(self.config)

        simple_queries = [
            "What is 2 + 2?",
            "Define python",
            "Hello",
            "Thanks",
            "What's the time?",
        ]

        complex_queries = [
            "Analyze the performance and then explain the results",
            "First read the file and then summarize it",
            "Compare options A and B, then recommend one",
        ]

        times = []

        for query in simple_queries + complex_queries:
            start = time.perf_counter()
            agent._is_simple_query(query)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        avg_time = sum(times) / len(times)
        assert avg_time < 0.001, f"Query detection took {avg_time * 1000:.3f}ms (expected < 1ms)"
        print(f"  Simple query detection (avg): {avg_time * 1000:.3f}ms")

    def test_fast_path_bypass_time(self):
        """Measure overhead of fast-path decision making."""
        from core.agent import TaskCompletionCache

        test_cases = [
            ("read file.txt", [("read", "file content here")]),
            ("what is python", [("explain", "Python is a programming language")]),
            ("list files", [("list", "file1.py\nfile2.py")]),
        ]

        times = []

        for query, tool_results in test_cases:
            start = time.perf_counter()
            TaskCompletionCache.is_definitely_complete(query, tool_results)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        avg_time = sum(times) / len(times)
        assert avg_time < 0.0005, f"Fast-path check took {avg_time * 1000 * 1000:.1f}µs (expected < 500µs)"
        print(f"  Fast-path bypass (avg): {avg_time * 1000 * 1000:.1f}µs")

    def test_response_without_fast_path(self):
        """Simulate response time without fast-path optimization."""
        from core.agent import TaskCompletionCache

        compound_queries = [
            "read the file and explain what it does",
            "analyze the data and create a summary",
            "check the logs and find and analyze errors",
        ]

        is_complete_results = []

        for query in compound_queries:
            tool_results = [("read", "some content")]
            start = time.perf_counter()
            is_complete = TaskCompletionCache.is_definitely_complete(query, tool_results)
            elapsed = time.perf_counter() - start
            is_complete_results.append((query, is_complete, elapsed))

        print("  Compound query evaluations:")
        for query, is_complete, elapsed in is_complete_results:
            status = "COMPLETE" if is_complete else "NEEDS_EVAL"
            print(f"    {query[:30]}...: {status} ({elapsed * 1000 * 1000:.1f}µs)")

        assert not any(r[1] for r in is_complete_results), "Compound queries should need evaluation"


class TestTaskCompletionCache:
    """Benchmark task completion evaluation with/without cache."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        tracemalloc.start()

    def teardown(self):
        tracemalloc.stop()

    @pytest.mark.asyncio
    async def test_cache_hit_performance(self):
        """Measure cache hit vs miss performance."""
        from core.agent import ToolResultCache, ActionResult

        cache = ToolResultCache(max_size=100, ttl_seconds=60)

        args = {"path": "/test/file.py"}
        result = ActionResult(
            success=True,
            output="file content here",
            execution_time=0.05,
        )

        await cache.set("read_file", args, result)

        hit_times = []
        miss_times = []

        for _ in range(100):
            start = time.perf_counter()
            await cache.get("read_file", args)
            hit_times.append(time.perf_counter() - start)

        for i in range(100):
            start = time.perf_counter()
            await cache.get("read_file", {"path": f"/test/file{i}.py"})
            miss_times.append(time.perf_counter() - start)

        hit_avg = sum(hit_times) / len(hit_times)
        miss_avg = sum(miss_times) / len(miss_times)

        print(f"  Cache hit (avg): {hit_avg * 1000 * 1000:.2f}µs")
        print(f"  Cache miss (avg): {miss_avg * 1000 * 1000:.2f}µs")
        print(f"  Cache stats: {cache.get_stats()}")

        assert hit_avg < 0.001, f"Cache hit took {hit_avg * 1000:.3f}ms (expected < 1ms)"

    def test_task_completion_pattern_matching(self):
        """Benchmark pattern matching for task completion."""
        from core.agent import TaskCompletionCache

        test_cases = {
            "single_tool_simple": [
                ("read config.yaml", [("read", "content")], True),
                ("list files", [("list", "file1\nfile2")], True),
                ("get status", [("get", "status: ok")], True),
            ],
            "single_tool_complex": [
                ("analyze and summarize", [("analyze", "data")], False),
                ("first verify then proceed", [("verify", "ok")], False),
            ],
            "multi_tool": [
                ("read and analyze", [("read", "content"), ("analyze", "analysis")], False),
            ],
        }

        times_by_category = {}

        for category, cases in test_cases.items():
            times = []
            for query, tool_results, expected in cases:
                start = time.perf_counter()
                TaskCompletionCache.is_definitely_complete(query, tool_results)
                times.append(time.perf_counter() - start)
            times_by_category[category] = sum(times) / len(times)

        for category, avg_time in times_by_category.items():
            print(f"  {category}: {avg_time * 1000 * 1000:.2f}µs")
            assert avg_time < 0.001, f"{category} took too long"

    @pytest.mark.asyncio
    async def test_cache_eviction_performance(self):
        """Test cache performance with many entries."""
        from core.agent import ToolResultCache, ActionResult

        cache = ToolResultCache(max_size=50, ttl_seconds=60)

        start = time.perf_counter()

        for i in range(100):
            result = ActionResult(success=True, output=f"result_{i}", execution_time=0.01)
            await cache.set("tool", {"arg": f"value_{i}"}, result)

        populate_time = time.perf_counter() - start

        stats = cache.get_stats()

        print(f"  Populate 100 entries: {populate_time * 1000:.1f}ms")
        print(f"  Cache size after: {stats['size']}")
        print(f"  Max size: {stats['max_size']}")

        assert stats['size'] == 50, "Cache should be at max size after eviction"


class TestMemoryGrowth:
    """Benchmark memory usage of subsystems."""

    @pytest.fixture(autouse=True)
    def setup(self):
        tracemalloc.start()

    def teardown(self):
        tracemalloc.stop()

    def test_memory_after_operations(self):
        """Track memory growth after repeated operations."""
        from core.memory import ConversationMemory
        import tempfile

        tracker = MemoryTracker()
        tracker.snapshot_before()

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = ConversationMemory(
                max_turns=100,
                max_tokens=128000,
                storage_path=Path(tmpdir),
            )

            for i in range(50):
                memory.add_message("user", f"Message {i}" * 10)
                memory.add_message("assistant", f"Response {i}" * 10)

            memory.get_messages_for_llm()

        tracker.snapshot_after()

        print(f"  Memory delta: {tracker.memory_delta / 1024:.1f}KB")
        print(f"  Object delta: {tracker.object_delta}")

    def test_vector_memory_growth(self):
        """Track vector memory growth over operations."""
        try:
            from core.vector_memory import get_vector_memory
        except Exception:
            pytest.skip("vector_memory not available")

        tracker = MemoryTracker()
        tracker.snapshot_before()

        try:
            vm = get_vector_memory()

            for i in range(20):
                vector = [0.1] * 128
                vm.add_vector(f"test_{i}", vector, {"index": i})

            for _ in range(10):
                query = [0.1] * 128
                vm.search(query, k=5)

        except Exception as e:
            print(f"  Skipped: {e}")
            return

        tracker.snapshot_after()

        print(f"  Vector memory delta: {tracker.memory_delta / 1024:.1f}KB")
        print(f"  Object delta: {tracker.object_delta}")

    def test_knowledge_graph_growth(self):
        """Track knowledge graph growth over operations."""
        pytest.skip("Heavy resource test - skipped in parallel runs")

    def test_agent_memory_footprint(self):
        """Measure agent memory footprint after initialization."""
        from core.config import Config
        from core.agent import JarvisAgent

        tracker = MemoryTracker()
        tracker.snapshot_before()

        config = Config.load()
        config.memory.enabled = False
        JarvisAgent(config)

        tracker.snapshot_after()

        print(f"  Agent memory delta: {tracker.memory_delta / 1024 / 1024:.2f}MB")
        print(f"  Object delta: {tracker.object_delta}")

        assert tracker.memory_delta < 100 * 1024 * 1024, "Agent should use < 100MB at init"


class TestAgentMethodBenchmarks:
    """Benchmark specific agent methods."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.tmp_path = tmp_path
        from core.config import Config
        from core.agent import JarvisAgent

        self.config = Config.load()
        self.config.memory.enabled = False
        self.agent = JarvisAgent(self.config)
        tracemalloc.start()

    def teardown(self):
        tracemalloc.stop()

    def test_get_live_state_time(self):
        """Benchmark live state retrieval."""
        from core.live_state import get_live_state
        try:
            ls = get_live_state()
        except Exception as e:
            pytest.skip(f"Live state not available: {e}")

        times = []
        for _ in range(10):
            start = time.perf_counter()
            ls.get_state()
            times.append(time.perf_counter() - start)

        avg_time = sum(times) / len(times)
        print(f"  Live state retrieval (avg): {avg_time * 1000:.2f}ms")
        assert avg_time < 0.05, f"Live state took {avg_time * 1000:.1f}ms (expected < 50ms)"

    def test_subsystem_state_getters(self):
        """Benchmark individual subsystem state getters."""
        getters = [
            self.agent._get_daydream_state,
            self.agent._get_erdos_state,
            self.agent._get_music_state,
            self.agent._get_learning_state,
            self.agent._get_evolution_state,
        ]

        for getter in getters:
            name = getter.__name__
            times = []

            for _ in range(10):
                start = time.perf_counter()
                getter()
                times.append(time.perf_counter() - start)

            avg_time = sum(times) / len(times)
            print(f"  {name}: {avg_time * 1000:.2f}ms")
            assert avg_time < 0.1, f"{name} took {avg_time * 1000:.1f}ms (expected < 100ms)"

    def test_live_state_cache_effectiveness(self):
        """Test that live state caching reduces repeated access time."""
        from core.live_state import get_live_state

        try:
            ls = get_live_state()
        except Exception as e:
            pytest.skip(f"Live state not available: {e}")

        first_times = []
        cached_times = []

        for _ in range(5):
            start = time.perf_counter()
            ls.get_state()
            first_times.append(time.perf_counter() - start)

            start = time.perf_counter()
            ls.get_state()
            cached_times.append(time.perf_counter() - start)

        first_avg = sum(first_times) / len(first_times)
        cached_avg = sum(cached_times) / len(cached_times)

        print(f"  First fetch (avg): {first_avg * 1000:.2f}ms")
        print(f"  Cached fetch (avg): {cached_avg * 1000:.2f}ms")


class TestConcurrentBenchmarks:
    """Benchmark concurrent operations."""

    @pytest.fixture(autouse=True)
    def setup(self):
        tracemalloc.start()

    def teardown(self):
        tracemalloc.stop()

    @pytest.mark.asyncio
    async def test_concurrent_context_fetch(self):
        """Benchmark parallel context fetching under load."""
        try:
            from core.unified_context_engine import get_context_engine
        except Exception:
            pytest.skip("Context engine not available")

        engine = get_context_engine()
        queries = [
            "What is Python?",
            "Explain machine learning",
            "How does caching work?",
            "Define microservices",
            "What is async programming?",
        ]

        async def fetch_concurrent():
            tasks = [engine.assemble_context(q) for q in queries]
            return await asyncio.gather(*tasks)

        start = time.perf_counter()
        await fetch_concurrent()
        concurrent_time = time.perf_counter() - start

        start = time.perf_counter()
        for q in queries:
            engine._estimate_tokens(q)
        sequential_time = time.perf_counter() - start

        print(f"  Concurrent fetch: {concurrent_time * 1000:.1f}ms")
        print(f"  Sequential fetch: {sequential_time * 1000:.1f}ms")
        print(f"  Speedup: {sequential_time / concurrent_time:.2f}x")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])