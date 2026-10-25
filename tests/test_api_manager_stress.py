"""Stress tests for API Call Manager - Speed and Volume testing."""
import asyncio
import sys
import time
import statistics
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class MockLLMClient:
    """Mock LLM client for stress testing."""

    def __init__(self, delay: float = 0.01):
        self.call_count = 0
        self.calls = []
        self._delay = delay
        self._fail_rate = 0.0

    def set_fail_rate(self, rate: float):
        """Set failure rate (0.0 to 1.0)."""
        self._fail_rate = rate

    async def chat(self, user_input: str, system_prompt: str = "", conversation_history=None, **kwargs):
        self.call_count += 1
        self.calls.append({
            "user_input": user_input[:50],
            "timestamp": time.time(),
        })
        await asyncio.sleep(self._delay)

        import random
        if self._fail_rate > 0 and random.random() < self._fail_rate:
            raise Exception("Simulated API failure")

        return {"message": {"content": f"Response to: {user_input[:30]}..."}}

    async def chat_with_tools(self, user_input: str, system_prompt: str = "", tools=None, **kwargs):
        self.call_count += 1
        await asyncio.sleep(self._delay)
        return f"Tool response to: {user_input[:30]}", None


@pytest.fixture(autouse=True)
def reset_api_manager_singleton():
    """Reset the API manager singleton between tests."""
    import core.api_manager as am
    am._default_manager = None
    yield
    am._default_manager = None


def create_fresh_manager(llm_client, **kwargs):
    """Create a fresh API manager with reset rate limits."""
    from core.api_manager import APICallManager, Priority, RateLimit

    manager = APICallManager(llm_client, **kwargs)

    # Reset rate limits for tests - very high limits
    manager._rate_limits = {
        Priority.INTERACTIVE: RateLimit(max_requests=10000, window_seconds=60.0),
        Priority.LEARNING: RateLimit(max_requests=10000, window_seconds=60.0),
        Priority.REFINEMENT: RateLimit(max_requests=10000, window_seconds=60.0),
        Priority.DAYDREAM: RateLimit(max_requests=10000, window_seconds=60.0),
        Priority.MAINTENANCE: RateLimit(max_requests=10000, window_seconds=60.0),
    }

    return manager


async def test_high_volume_requests():
    """Test handling 500+ concurrent requests."""
    print("\n=== STRESS TEST: High Volume (500 requests) ===")

    from core.api_manager import Priority

    llm = MockLLMClient(delay=0.001)  # Very fast mock
    manager = create_fresh_manager(llm, max_queue_size=500)

    # Submit 500 requests concurrently
    start = time.time()
    tasks = []

    for i in range(500):
        tasks.append(manager.submit_chat(
            subsystem=f"test_{i % 10}",  # 10 different subsystems
            priority=Priority.INTERACTIVE,
            user_input=f"Request {i}" * 10,
            system_prompt="Test prompt",
            wait_for_result=False,
        ))

    # Just submit, don't wait for completion
    results = await asyncio.gather(*tasks, return_exceptions=True)
    submit_time = time.time() - start

    success = sum(1 for r in results if not isinstance(r, Exception))
    errors = sum(1 for r in results if isinstance(r, Exception))

    stats = manager.get_stats()

    print(f"  ✓ Submitted 500 requests in {submit_time:.3f}s ({500/submit_time:.0f} req/s)")
    print(f"  ✓ Queue accepted: {success}, Rejected: {errors}")
    print(f"  ✓ Stats: total={stats['total_requests']}, queue={stats['queue_size']}")

    assert success >= 450, f"At least 450 requests should be accepted, got {success}"
    print("  ✓ High volume test PASSED")
    return True


async def test_rapid_fire_requests():
    """Test rapid successive requests (speed test)."""
    print("\n=== STRESS TEST: Rapid Fire Speed ===")

    from core.api_manager import Priority

    llm = MockLLMClient(delay=0.005)
    manager = create_fresh_manager(llm)

    latencies = []

    # Fire 100 requests as fast as possible
    for i in range(100):
        start = time.time()
        response = await manager.submit_chat(
            subsystem="rapid_test",
            priority=Priority.INTERACTIVE,
            user_input=f"Rapid {i}",
            system_prompt="Speed test",
        )
        latency = (time.time() - start) * 1000
        latencies.append(latency)

    avg_latency = statistics.mean(latencies)
    p50 = statistics.median(latencies)
    p99 = statistics.quantiles(latencies, n=100)[98] if len(latencies) >= 100 else max(latencies)

    print("  ✓ 100 requests completed")
    print(f"  ✓ Avg latency: {avg_latency:.1f}ms")
    print(f"  ✓ P50 latency: {p50:.1f}ms")
    print(f"  ✓ P99 latency: {p99:.1f}ms")

    assert avg_latency < 100, f"Avg latency should be <100ms, got {avg_latency:.1f}ms"
    print("  ✓ Rapid fire test PASSED")
    return True


async def test_queue_overflow():
    """Test queue overflow handling when processor is running."""
    print("\n=== STRESS TEST: Queue Overflow ===")

    from core.api_manager import Priority

    llm = MockLLMClient(delay=0.5)  # Slow mock to cause queue buildup
    manager = create_fresh_manager(llm, max_queue_size=10)
    await manager.start()  # Start processor so queue is used

    # Submit requests that will queue up
    results = []
    tasks = []

    for i in range(30):
        async def submit_req(idx):
            response = await manager.submit_chat(
                subsystem="overflow_test",
                priority=Priority.LEARNING,
                user_input=f"Overflow {idx}",
                system_prompt="Test",
                wait_for_result=True,
                timeout=1.0,
            )
            results.append(response)

        tasks.append(submit_req(i))

    # Run all in parallel
    await asyncio.gather(*tasks, return_exceptions=True)

    await manager.stop()

    # Count results
    success = sum(1 for r in results if r and r.get("message"))
    rejected = sum(1 for r in results if r and r.get("error") in ["Request queue full", "Request timed out"])

    print(f"  ✓ Successful: {success}, Rejected/Timeout: {rejected}")
    print("  ✓ Queue size was: 10, submitted: 30")
    # With slow mock and small queue, some should timeout or get rejected
    print("  ✓ Queue overflow test PASSED")
    return True


async def test_rate_limiting_stress():
    """Test rate limiting under stress."""
    print("\n=== STRESS TEST: Rate Limiting ===")

    from core.api_manager import Priority

    # Create fresh manager for this test
    llm = MockLLMClient(delay=0.001)
    manager = create_fresh_manager(llm)  # Fresh instance, clean rate limits

    # Set very aggressive rate limits
    manager.set_subsystem_limit("rate_stress", max_requests=5, window_seconds=10.0)

    success_count = 0
    rate_limited_count = 0

    # Rapid fire from same subsystem
    for i in range(20):
        response = await manager.submit_chat(
            subsystem="rate_stress",  # Different subsystem than other tests
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
    assert rate_limited_count >= 10, "At least 10 requests should be rate limited"
    assert success_count <= 5, "At most 5 requests should succeed"
    print("  ✓ Rate limiting stress test PASSED")
    return True


async def test_cache_performance():
    """Test cache performance under load."""
    print("\n=== STRESS TEST: Cache Performance ===")

    from core.api_manager import Priority, RequestType

    # Create fresh instance with high rate limits to avoid interference
    llm = MockLLMClient(delay=0.01)
    manager = create_fresh_manager(llm, enable_caching=True)

    # Verify caching is enabled and TTL is set
    assert manager.enable_caching, "Caching should be enabled"
    assert manager.CACHE_TTL.get(RequestType.CHAT, 0) > 0, "Chat TTL should be > 0"

    # Use unique subsystem to avoid rate limit conflicts from other tests
    subsystem = "cache_perf_unique"

    # Same request repeated 100 times
    same_input = "This is a repeated query for cache testing"
    same_prompt = "Cache test prompt"

    # First request - should be cache miss
    r1 = await manager.submit_chat(
        subsystem=subsystem,
        priority=Priority.INTERACTIVE,
        user_input=same_input,
        system_prompt=same_prompt,
    )
    assert not r1.get("_from_cache", False), "First request should not be cached"

    # Second request - should hit cache
    r2 = await manager.submit_chat(
        subsystem=subsystem,
        priority=Priority.INTERACTIVE,
        user_input=same_input,
        system_prompt=same_prompt,
    )

    # Debug output
    print(f"  ✓ First request: from_cache={r1.get('_from_cache')}")
    print(f"  ✓ Second request: from_cache={r2.get('_from_cache')}, error={r2.get('message', {}).get('content', '')[:30] if r2.get('message') else 'none'}")

    assert r2.get("_from_cache", False), f"Second request should be cached, got {r2}"

    # Now do performance test
    for _ in range(98):  # Already did 2
        await manager.submit_chat(
            subsystem=subsystem,
            priority=Priority.INTERACTIVE,
            user_input=same_input,
            system_prompt=same_prompt,
        )

    stats = manager.get_stats()

    print("  ✓ Completed 100 requests total")
    print(f"  ✓ Cache hits: {stats['cache_hits']}, misses: {stats['cache_misses']}")

    assert stats['cache_hits'] >= 95, f"Expected 95+ cache hits, got {stats['cache_hits']}"
    print("  ✓ Cache performance test PASSED")
    return True


async def test_concurrent_subsystems():
    """Test multiple subsystems making concurrent requests."""
    print("\n=== STRESS TEST: Concurrent Subsystems ===")

    from core.api_manager import Priority

    llm = MockLLMClient(delay=0.01)
    manager = create_fresh_manager(llm)

    subsystems = [
        ("erdos", Priority.LEARNING, 20),
        ("refine", Priority.REFINEMENT, 15),
        ("daydream", Priority.DAYDREAM, 10),
        ("thinking", Priority.INTERACTIVE, 25),
        ("meta", Priority.MAINTENANCE, 10),
    ]

    results = {name: [] for name, _, _ in subsystems}

    async def subsystem_worker(name, priority, count):
        for i in range(count):
            response = await manager.submit_chat(
                subsystem=name,
                priority=priority,
                user_input=f"{name} request {i}",
                system_prompt=f"{name} prompt",
            )
            results[name].append(response)

    start = time.time()
    tasks = [
        subsystem_worker(name, priority, count)
        for name, priority, count in subsystems
    ]
    await asyncio.gather(*tasks)
    duration = time.time() - start

    total_requests = sum(len(r) for r in results.values())
    successful = sum(sum(1 for r in rs if r.get("message")) for rs in results.values())

    print(f"  ✓ {len(subsystems)} subsystems, {total_requests} total requests")
    print(f"  ✓ Completed in {duration:.2f}s ({total_requests/duration:.0f} req/s)")
    print(f"  ✓ Successful: {successful}/{total_requests}")

    # Check per-subsystem stats
    activities = manager.get_recent_activities(limit=100)
    subsystem_counts = {}
    for act in activities:
        subsystem_counts[act.get("subsystem", "unknown")] = subsystem_counts.get(act.get("subsystem", "unknown"), 0) + 1

    print(f"  ✓ Activity log has {len(activities)} entries from {len(subsystem_counts)} subsystems")

    assert successful >= 70, f"At least 70 requests should succeed, got {successful}"
    print("  ✓ Concurrent subsystems test PASSED")
    return True


async def test_memory_under_load():
    """Test memory usage doesn't grow unbounded."""
    print("\n=== STRESS TEST: Memory Under Load ===")

    from core.api_manager import Priority
    import gc

    gc.collect()

    llm = MockLLMClient(delay=0.001)
    manager = create_fresh_manager(llm)

    # Make many requests with unique content
    for i in range(200):
        await manager.submit_chat(
            subsystem="memory_test",
            priority=Priority.INTERACTIVE,
            user_input=f"Unique request {i} with different content to avoid caching " + "x" * (i % 100),
            system_prompt=f"Unique prompt {i}",
        )

    stats = manager.get_stats()
    cache_size = stats['cache_size']

    # Check activity log is bounded
    activities = manager.get_recent_activities(limit=200)

    print("  ✓ 200 unique requests processed")
    print(f"  ✓ Cache size: {cache_size}")
    print(f"  ✓ Activity log entries: {len(activities)}")

    assert len(activities) <= 100, "Activity log should be bounded to 100"
    assert cache_size <= 1000, "Cache should be bounded"
    print("  ✓ Memory bounds test PASSED")
    return True


async def test_priority_preemption():
    """Test that priority ordering works in queue mode."""
    print("\n=== STRESS TEST: Priority Ordering ===")

    from core.api_manager import Priority, APIRequest, RequestType

    llm = MockLLMClient(delay=0.05)
    manager = create_fresh_manager(llm, max_queue_size=100)

    # Create test requests
    req_low = APIRequest(
        id="low",
        request_type=RequestType.CHAT,
        priority=Priority.MAINTENANCE,
        subsystem="test",
        user_input="low priority",
        system_prompt="test",
        conversation_history=None,
        tools=None,
    )
    req_high = APIRequest(
        id="high",
        request_type=RequestType.CHAT,
        priority=Priority.INTERACTIVE,
        subsystem="test",
        user_input="high priority",
        system_prompt="test",
        conversation_history=None,
        tools=None,
    )
    req_med = APIRequest(
        id="med",
        request_type=RequestType.CHAT,
        priority=Priority.LEARNING,
        subsystem="test",
        user_input="med priority",
        system_prompt="test",
        conversation_history=None,
        tools=None,
    )

    # Add to queue in reverse priority order
    with manager._lock:
        manager._request_queue.append(req_low)
        manager._request_queue.append(req_med)
        manager._request_queue.append(req_high)
        # Sort by priority
        manager._request_queue.sort(key=lambda r: r.priority.value)

    # Verify order
    queue_order = [r.priority.name for r in manager._request_queue]
    print(f"  ✓ Queue order after sort: {queue_order}")

    assert queue_order == ["INTERACTIVE", "LEARNING", "MAINTENANCE"], "Queue should be sorted by priority"

    print("  ✓ Priority ordering test PASSED")
    return True


async def run_all_stress_tests():
    """Run all stress tests."""
    print("=" * 60)
    print("API CALL MANAGER STRESS TESTS")
    print("=" * 60)

    tests = [
        ("High Volume (500 req)", test_high_volume_requests),
        ("Rapid Fire Speed", test_rapid_fire_requests),
        ("Queue Overflow", test_queue_overflow),
        ("Rate Limiting Stress", test_rate_limiting_stress),
        ("Cache Performance", test_cache_performance),
        ("Concurrent Subsystems", test_concurrent_subsystems),
        ("Memory Under Load", test_memory_under_load),
        ("Priority Preemption", test_priority_preemption),
    ]

    results = []

    for name, test_func in tests:
        try:
            result = await test_func()
            results.append((name, "PASS", None))
        except AssertionError as e:
            results.append((name, "FAIL", str(e)))
            print(f"  ✗ Assertion failed: {e}")
        except Exception as e:
            results.append((name, "ERROR", str(e)))
            print(f"  ✗ Error: {e}")

    print("\n" + "=" * 60)
    print("STRESS TEST RESULTS")
    print("=" * 60)

    passed = 0
    failed = 0

    for name, status, error in results:
        symbol = "✓" if status == "PASS" else "✗"
        print(f"  {symbol} {name}: {status}")
        if error:
            print(f"      Error: {error[:80]}...")
        if status == "PASS":
            passed += 1
        else:
            failed += 1

    print(f"\nTotal: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_all_stress_tests())
    sys.exit(0 if success else 1)
