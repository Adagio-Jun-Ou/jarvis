"""Tests for the shared AgenticHarness singleton (Stage H).

Before this, every /api/harness/* request constructed a fresh
AgenticHarness, so a plan created by one request was invisible to the
next and checkpoints never accumulated.
"""
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.agentic_harness import (  # noqa: E402
    AgenticHarness,
    get_agentic_harness,
    reset_agentic_harness,
)


@pytest.fixture(autouse=True)
def _fresh_singleton():
    reset_agentic_harness()
    yield
    reset_agentic_harness()


class TestSingleton:
    def test_same_instance_returned(self):
        first = get_agentic_harness()
        second = get_agentic_harness()
        assert first is second

    def test_reset_creates_new_instance(self):
        first = get_agentic_harness()
        reset_agentic_harness()
        second = get_agentic_harness()
        assert first is not second

    def test_agent_late_binds(self):
        harness = get_agentic_harness()
        assert harness.agent is None

        fake_agent = object()
        same = get_agentic_harness(agent=fake_agent)
        assert same is harness
        assert same.agent is fake_agent

    def test_agent_not_overwritten_by_later_call(self):
        first_agent = object()
        second_agent = object()
        harness = get_agentic_harness(agent=first_agent)
        same = get_agentic_harness(agent=second_agent)
        assert same.agent is first_agent

    def test_thread_safe_single_instance(self):
        instances = []
        barrier = threading.Barrier(4)

        def worker():
            barrier.wait()
            instances.append(get_agentic_harness())

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(instances) == 4
        assert all(inst is instances[0] for inst in instances)


class TestSharedStateAcrossCalls:
    def test_plan_visible_across_callers(self):
        """Simulates two REST requests: plan created by one, found by the next."""
        planner = get_agentic_harness()
        plan = planner.create_plan("build a widget", max_iterations=5)
        planner.save_checkpoint(plan)

        executor = get_agentic_harness()
        assert executor.plans.get(plan.id) is plan
        assert len(executor._checkpoints) == 1

    def test_checkpoint_restore_across_callers(self):
        first = get_agentic_harness()
        plan = first.create_plan("fix the bug", max_iterations=5)
        first.save_checkpoint(plan)

        second = get_agentic_harness()
        restored = second.restore_checkpoint()
        assert restored is not None
        assert restored.id == plan.id

    def test_agent_harness_shares_singleton(self):
        from core.agent import JarvisAgent

        fake = JarvisAgent.__new__(JarvisAgent)
        fake._agentic_harness = None
        harness = fake.get_agentic_harness()
        assert harness is get_agentic_harness()
        assert harness.agent is fake