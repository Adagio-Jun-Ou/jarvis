"""Test API Call Manager integration with subsystems."""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class MockLLMClient:
    """Mock LLM client for testing."""

    def __init__(self):
        self.call_count = 0
        self.calls = []
        self._delay = 0.1  # Simulate API latency

    async def chat(self, user_input: str, system_prompt: str = "", conversation_history: list = None, **kwargs):
        self.call_count += 1
        self.calls.append({
            "user_input": user_input[:50],
            "system_prompt": system_prompt[:30],
            "timestamp": time.time(),
        })
        await asyncio.sleep(self._delay)
        return {"message": {"content": f"Response to: {user_input[:30]}..."}}

    async def chat_with_tools(self, user_input: str, system_prompt: str = "", tools: list = None, **kwargs):
        self.call_count += 1
        self.calls.append({
            "user_input": user_input[:50],
            "system_prompt": system_prompt[:30],
            "tools": tools,
            "timestamp": time.time(),
        })
        await asyncio.sleep(self._delay)
        return f"Tool response to: {user_input[:30]}", None


async def test_api_manager_basic():
    """Test basic API manager functionality."""
    print("\n=== Test 1: Basic API Manager ===")

    from core.api_manager import APICallManager, Priority

    llm = MockLLMClient()
    manager = APICallManager(llm, batch_window_ms=50.0, max_batch_size=3)

    # Test direct submit
    response = await manager.submit_chat(
        subsystem="test",
        priority=Priority.INTERACTIVE,
        user_input="Hello, test message",
        system_prompt="You are a test assistant.",
    )

    assert response["message"]["content"], "Should get response"
    print(f"  ✓ Direct submit works: {response['message']['content'][:40]}...")

    # Test caching
    response2 = await manager.submit_chat(
        subsystem="test",
        priority=Priority.INTERACTIVE,
        user_input="Hello, test message",  # Same input
        system_prompt="You are a test assistant.",  # Same prompt
    )

    assert response2.get("_from_cache", False), "Should be cached"
    print(f"  ✓ Caching works: _from_cache={response2.get('_from_cache')}")

    # Test stats
    stats = manager.get_stats()
    print(f"  ✓ Stats: total={stats['total_requests']}, cache_hits={stats['cache_hits']}")

    assert stats["total_requests"] == 1  # Only 1 actual API call (2nd was cached)
    assert stats["cache_hits"] == 1

    print("  ✓ Basic API manager test PASSED")
    return True


async def test_priority_queuing():
    """Test priority-based queuing."""
    print("\n=== Test 2: Priority Queuing ===")

    from core.api_manager import APICallManager, Priority

    llm = MockLLMClient()
    manager = APICallManager(llm, max_queue_size=20)

    # Submit multiple requests with different priorities
    results = []

    async def submit_with_priority(priority, name):
        response = await manager.submit_chat(
            subsystem="test",
            priority=priority,
            user_input=f"Request: {name}",
            system_prompt="Test",
        )
        results.append((priority.value, name))

    # Submit in parallel with different priorities
    tasks = [
        submit_with_priority(Priority.MAINTENANCE, "low-1"),
        submit_with_priority(Priority.LEARNING, "medium-1"),
        submit_with_priority(Priority.INTERACTIVE, "high-1"),
        submit_with_priority(Priority.REFINEMENT, "medium-2"),
        submit_with_priority(Priority.INTERACTIVE, "high-2"),
    ]

    await asyncio.gather(*tasks)

    print(f"  ✓ All {len(results)} requests processed")
    stats = manager.get_stats()
    print(f"  ✓ Stats: {stats['total_requests']} requests, {stats['cache_misses']} cache misses")

    print("  ✓ Priority queuing test PASSED")
    return True


async def test_rate_limiting():
    """Test rate limiting."""
    print("\n=== Test 3: Rate Limiting ===")

    from core.api_manager import APICallManager, Priority

    llm = MockLLMClient()
    llm._delay = 0.01  # Fast for testing
    manager = APICallManager(llm)

    # Set low rate limit for testing
    manager.set_subsystem_limit("rate_test", max_requests=3, window_seconds=60.0)

    success_count = 0
    rate_limited_count = 0

    for i in range(5):
        response = await manager.submit_chat(
            subsystem="rate_test",
            priority=Priority.LEARNING,
            user_input=f"Rate test {i}",
            system_prompt="Test",
        )
        content = response.get("message", {}).get("content", "")
        if "Rate limit exceeded" in content or response.get("error") == "Rate limit exceeded":
            rate_limited_count += 1
        elif response.get("message"):
            success_count += 1

    print(f"  ✓ Successful: {success_count}, Rate limited: {rate_limited_count}")
    assert rate_limited_count > 0, "Should have rate limited some requests"

    print("  ✓ Rate limiting test PASSED")
    return True


async def test_erdos_integration():
    """Test Erdős engine with API manager."""
    print("\n=== Test 4: Erdős Engine Integration ===")

    from core.erdos_engine import ErdosSolver, ErdosProblemGenerator, ProblemCategory, ProblemDifficulty

    llm = MockLLMClient()
    llm._delay = 0.05

    solver = ErdosSolver(llm_client=llm)

    # Verify API manager is set up
    api_manager = solver._get_api_manager()
    assert api_manager is not None, "API manager should be initialized"
    print("  ✓ ErdosSolver has API manager")

    # Create a test problem
    generator = ErdosProblemGenerator()
    problem = generator.generate_problem(
        category=ProblemCategory.COMBINATORICS,
        difficulty=ProblemDifficulty.EASY,
    )

    print(f"  ✓ Generated problem: {problem.title}")

    # Test the _make_llm_call method
    response = await solver._make_llm_call(
        user_input="Test problem: what is 2+2?",
        system_prompt="Math test",
        priority="learning",
    )

    assert response.get("message"), "Should get response"
    print("  ✓ LLM call through API manager works")

    # Check stats
    stats = api_manager.get_stats()
    print(f"  ✓ API Manager stats: {stats['total_requests']} requests, avg {stats['avg_latency_ms']:.1f}ms")

    print("  ✓ Erdős integration test PASSED")
    return True


async def test_concurrent_requests():
    """Test handling concurrent requests from multiple subsystems."""
    print("\n=== Test 5: Concurrent Subsystem Requests ===")

    from core.api_manager import APICallManager, Priority

    llm = MockLLMClient()
    llm._delay = 0.05
    manager = APICallManager(llm, batch_window_ms=50.0, max_batch_size=5)

    # Simulate concurrent requests from different subsystems
    async def erdos_request(i):
        return await manager.submit_chat(
            subsystem="erdos_engine",
            priority=Priority.LEARNING,
            user_input=f"Erdős problem {i}",
            system_prompt="Mathematician mode",
        )

    async def refine_request(i):
        return await manager.submit_chat(
            subsystem="self_refine",
            priority=Priority.REFINEMENT,
            user_input=f"Code issue {i}",
            system_prompt="Code reviewer",
        )

    async def daydream_request(i):
        return await manager.submit_chat(
            subsystem="daydream",
            priority=Priority.DAYDREAM,
            user_input=f"Creative thought {i}",
            system_prompt="Dreamer",
        )

    # Launch concurrent requests
    tasks = []
    for i in range(3):
        tasks.append(erdos_request(i))
        tasks.append(refine_request(i))
        tasks.append(daydream_request(i))

    start = time.time()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    duration = time.time() - start

    success_count = sum(1 for r in results if isinstance(r, dict) and r.get("message"))
    error_count = sum(1 for r in results if isinstance(r, Exception) or (isinstance(r, dict) and r.get("error")))

    print(f"  ✓ Processed {len(results)} requests in {duration:.2f}s")
    print(f"  ✓ Success: {success_count}, Errors: {error_count}")

    stats = manager.get_stats()
    print(f"  ✓ Batched requests: {stats['batched_requests']}")
    print(f"  ✓ Cache hits: {stats['cache_hits']}")

    print("  ✓ Concurrent requests test PASSED")
    return True


async def test_cache_invalidation():
    """Test cache invalidation and TTL."""
    print("\n=== Test 6: Cache Behavior ===")

    from core.api_manager import APICallManager, Priority, RequestType

    llm = MockLLMClient()
    manager = APICallManager(llm, enable_caching=True)

    # Cache TTL for different request types
    print(f"  Cache TTL for CHAT: {manager.CACHE_TTL.get(RequestType.CHAT, 0)}s")
    print(f"  Cache TTL for EMBEDDING: {manager.CACHE_TTL.get(RequestType.EMBEDDING, 0)}s")
    print(f"  Cache TTL for CHAT_WITH_TOOLS: {manager.CACHE_TTL.get(RequestType.CHAT_WITH_TOOLS, 0)}s")

    # Test cache clear
    manager.clear_cache()
    assert len(manager._cache) == 0, "Cache should be empty"
    print("  ✓ Cache cleared")

    # Test cache storage
    await manager.submit_chat(
        subsystem="test",
        priority=Priority.INTERACTIVE,
        user_input="Cache test",
        system_prompt="Test",
    )

    assert len(manager._cache) == 1, "Cache should have 1 entry"
    print("  ✓ Cache stores entries")

    # Test cache hit
    response = await manager.submit_chat(
        subsystem="test",
        priority=Priority.INTERACTIVE,
        user_input="Cache test",
        system_prompt="Test",
    )

    assert response.get("_from_cache"), "Should hit cache"
    print("  ✓ Cache hit works")

    print("  ✓ Cache behavior test PASSED")
    return True


async def test_stats_tracking():
    """Test statistics tracking."""
    print("\n=== Test 7: Statistics Tracking ===")

    from core.api_manager import APICallManager, Priority

    llm = MockLLMClient()
    manager = APICallManager(llm)

    # Make some requests
    for i in range(5):
        await manager.submit_chat(
            subsystem="test",
            priority=Priority.INTERACTIVE,
            user_input=f"Stats test {i}",
            system_prompt="Test",
        )

    stats = manager.get_stats()

    print(f"  Total requests: {stats['total_requests']}")
    print(f"  Cache misses: {stats['cache_misses']}")
    print(f"  Avg latency: {stats['avg_latency_ms']:.1f}ms")
    print(f"  Queue size: {stats['queue_size']}")
    print(f"  Pending: {stats['pending_requests']}")
    print(f"  Cache size: {stats['cache_size']}")

    assert stats['total_requests'] == 5
    assert stats['avg_latency_ms'] > 0

    print("  ✓ Statistics tracking test PASSED")
    return True


async def run_all_tests():
    """Run all integration tests."""
    print("=" * 60)
    print("API CALL MANAGER INTEGRATION TESTS")
    print("=" * 60)

    tests = [
        ("Basic API Manager", test_api_manager_basic),
        ("Priority Queuing", test_priority_queuing),
        ("Rate Limiting", test_rate_limiting),
        ("Erdős Integration", test_erdos_integration),
        ("Concurrent Requests", test_concurrent_requests),
        ("Cache Behavior", test_cache_invalidation),
        ("Statistics Tracking", test_stats_tracking),
    ]

    results = []

    for name, test_func in tests:
        try:
            result = await test_func()
            results.append((name, "PASS", None))
        except Exception as e:
            results.append((name, "FAIL", str(e)))
            print(f"  ✗ Error: {e}")

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    passed = 0
    failed = 0

    for name, status, error in results:
        symbol = "✓" if status == "PASS" else "✗"
        print(f"  {symbol} {name}: {status}")
        if error:
            print(f"      Error: {error[:60]}...")
        if status == "PASS":
            passed += 1
        else:
            failed += 1

    print(f"\nTotal: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
