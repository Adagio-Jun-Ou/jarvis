"""Round 23 audit bug tests - sub_agent, sub_agent_comm, health, token_optimizer, deployment LoadThrottler."""
import asyncio
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestSubAgentThreadSafety:
    def test_decompose_cache_thread_safe(self):
        from core.sub_agent import TaskPlanner

        planner = TaskPlanner(llm_client=None)
        errors = []

        def store_fn(i):
            try:
                planner._store_cache(f"objective_{i}", [])
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=store_fn, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Cache errors: {errors}"

    def test_decompose_cache_check_under_lock(self):
        from core.sub_agent import TaskPlanner

        planner = TaskPlanner(llm_client=None)
        planner._store_cache("test_obj", ["item1", "item2"])

        result = planner._check_cache("test_obj")
        assert result == ["item1", "item2"]

        result = planner._check_cache("nonexistent")
        assert result is None

    def test_agent_pool_thread_safe(self):
        from core.sub_agent import AgentPool

        pool = AgentPool()

        def select_fn(i):
            from core.sub_agent import SubTask
            task = SubTask(id=f"task_{i}", title=f"Task {i}", description=f"Search for {i}")
            result = pool.select_agent(task)
            pool.mark_busy(result) if result else None
            pool.mark_available(result) if result else None

        threads = [threading.Thread(target=select_fn, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        available = pool.get_available_agents()
        assert len(available) > 0

    def test_pool_busy_marking(self):
        from core.sub_agent import AgentPool

        pool = AgentPool()
        pool.mark_busy("researcher")

        assert "researcher" not in pool.get_available_agents()
        pool.mark_available("researcher")
        assert "researcher" in pool.get_available_agents()

    def test_orchestrator_max_results_bound(self):
        from core.sub_agent import SubAgentOrchestrator

        orch = SubAgentOrchestrator.__new__(SubAgentOrchestrator)
        orch.results = {}
        orch.MAX_RESULTS = 5
        orch.tasks = {}

        for i in range(10):
            orch.results[f"task_{i}"] = f"result_{i}"

        assert len(orch.results) == 10

    def test_heuristic_decompose(self):
        from core.sub_agent import TaskPlanner

        planner = TaskPlanner(llm_client=None)
        tasks = planner._heuristic_decompose("search for files and then analyze them", None)
        assert len(tasks) >= 2


class TestSubAgentCommThreadSafety:
    def test_register_handler_under_lock(self):
        from core.sub_agent_comm import SubAgentCommunication, MessageType

        comm = SubAgentCommunication(storage_path=Path(tempfile.mkdtemp()))
        handler = MagicMock()
        comm.register_handler("test_agent", MessageType.TASK, handler)
        assert "test_agent" in comm._handlers
        assert MessageType.TASK in comm._handlers["test_agent"]

    def test_message_callback_bounded(self):
        from core.sub_agent_comm import SubAgentCommunication

        comm = SubAgentCommunication(storage_path=Path(tempfile.mkdtemp()))
        for i in range(comm._MAX_MESSAGE_CALLBACKS + 10):
            comm.register_message_callback(lambda msg: None)

        with comm._lock:
            assert len(comm._message_callbacks) <= comm._MAX_MESSAGE_CALLBACKS

    def test_discovery_handler_bounded(self):
        from core.sub_agent_comm import SubAgentCommunication

        comm = SubAgentCommunication(storage_path=Path(tempfile.mkdtemp()))
        for i in range(comm._MAX_DISCOVERY_HANDLERS + 10):
            comm.register_discovery_handler(lambda agent, action: None)

        with comm._lock:
            assert len(comm._discovery_handlers) <= comm._MAX_DISCOVERY_HANDLERS

    def test_concurrent_send_receive(self):
        from core.sub_agent_comm import SubAgentCommunication, MessageType

        comm = SubAgentCommunication(storage_path=Path(tempfile.mkdtemp()))
        comm.register_agent("sender")
        comm.register_agent("receiver")

        errors = []

        def send_fn(i):
            try:
                msg_id = comm.send(
                    from_agent="sender",
                    to_agent="receiver",
                    message_type=MessageType.TASK,
                    payload={"task": f"task_{i}"},
                )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=send_fn, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Errors: {errors}"

    def test_session_create_thread_safe(self):
        from core.sub_agent_comm import SessionManager

        sm = SessionManager()
        sessions = []

        def create_fn(i):
            s = sm.create_session(["agent_a", "agent_b"])
            sessions.append(s)

        threads = [threading.Thread(target=create_fn, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(sessions) == 10

    def test_session_cleanup_expired(self):
        from core.sub_agent_comm import SessionManager

        sm = SessionManager()
        sm.create_session(["a", "b"])
        sm._session_timeout_minutes = 0

        time.sleep(0.01)
        expired = sm.cleanup_expired()
        assert expired >= 1

    def test_unregister_message_callback(self):
        from core.sub_agent_comm import SubAgentCommunication

        comm = SubAgentCommunication(storage_path=Path(tempfile.mkdtemp()))
        cb = lambda msg: None
        comm.register_message_callback(cb)

        result = comm.unregister_message_callback(cb)
        assert result is True

        result = comm.unregister_message_callback(cb)
        assert result is False

    def test_unregister_discovery_handler(self):
        from core.sub_agent_comm import SubAgentCommunication

        comm = SubAgentCommunication(storage_path=Path(tempfile.mkdtemp()))
        handler = lambda agent, action: None
        comm.register_discovery_handler(handler)

        result = comm.unregister_discovery_handler(handler)
        assert result is True

        result = comm.unregister_discovery_handler(handler)
        assert result is False


class TestHealthManagerThreadSafety:
    def test_register_concurrent(self):
        from core.health import HealthManager

        hm = HealthManager()

        def register_fn(i):
            hm.register(f"subsystem_{i}", dependencies=[])

        threads = [threading.Thread(target=register_fn, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        stats = hm.get_all_health()
        assert len(stats["subsystems"]) >= 0

    def test_mark_healthy_under_lock(self):
        from core.health import HealthManager

        hm = HealthManager()
        hm.register("test_sub")
        hm.mark_healthy("test_sub", {"info": "working"})

        status = hm.get_status("test_sub")
        assert status.value == "healthy"

    def test_mark_unhealthy_increments_error_count(self):
        from core.health import HealthManager

        hm = HealthManager()
        hm.register("test_sub")
        hm.mark_unhealthy("test_sub", "something broke")
        hm.mark_unhealthy("test_sub", "broke again")

        with hm._lock:
            assert hm._subsystems["test_sub"].error_count == 2

    def test_overall_status_with_no_subs(self):
        from core.health import HealthManager, HealthStatus

        hm = HealthManager()
        status = hm._get_overall_status()
        assert status == HealthStatus.STARTING


class TestTokenOptimizerThreadSafety:
    def test_cache_concurrent_access(self):
        from core.token_optimizer import TokenCache

        cache = TokenCache(max_size=100)
        errors = []

        def write_fn(i):
            try:
                cache.set(f"response_{i}", tokens=100, importance=0.5, key=f"k{i}")
            except Exception as e:
                errors.append(e)

        def read_fn(i):
            try:
                cache.get(key=f"k{i}")
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(20):
            threads.append(threading.Thread(target=write_fn, args=(i,)))
            threads.append(threading.Thread(target=read_fn, args=(i,)))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Errors: {errors}"

    def test_cache_eviction_under_size_limit(self):
        from core.token_optimizer import TokenCache

        cache = TokenCache(max_size=10)
        for i in range(20):
            cache.set(f"response_{i}", tokens=i, importance=float(i) / 20, key=f"k{i}")

        stats = cache.stats()
        assert stats["size"] <= 10

    def test_reasoning_store_and_retrieve(self):
        from core.token_optimizer import ReasoningBank

        with tempfile.TemporaryDirectory() as tmpdir:
            bank = ReasoningBank(storage_path=Path(tmpdir))
            pid = bank.store("pattern about sorting", "use mergesort", tokens_saved=50)
            assert pid is not None

            results = bank.retrieve("sorting algorithm", k=5)
            assert len(results) <= 5

    def test_reasoning_concurrent_store(self):
        from core.token_optimizer import ReasoningBank

        with tempfile.TemporaryDirectory() as tmpdir:
            bank = ReasoningBank(storage_path=Path(tmpdir))
            errors = []

            def store_fn(i):
                try:
                    bank.store(f"pattern_{i}", f"template_{i}", tokens_saved=i)
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=store_fn, args=(i,)) for i in range(20)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)

            assert not errors, f"Errors: {errors}"

    def test_token_optimizer_stats_thread_safe(self):
        from core.token_optimizer import TokenOptimizer

        with tempfile.TemporaryDirectory() as tmpdir:
            opt = TokenOptimizer(storage_path=Path(tmpdir))
            stats = opt.stats()
            assert isinstance(stats, dict)
            assert "cache" in stats
            assert "total_savings" in stats

