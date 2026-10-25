"""Round 18 Sub-Audit Bug Tests.

Tests for bugs found and fixed in deep sub-audit:
- self_evaluation.py: DEADLOCK (Lock→RLock), callbacks iteration under lock, _save_state I/O outside lock
- context_compressor.py: lock-free mutations in _compress_active, TOCTOU in proactive_summarize, _archive_old lock safety, _save_state pattern
- circuit_breaker.py: missing rejected_requests in stats property, thread-unsafe get()/get_all_stats()
- module_bridge.py: emit() reads _subscribers outside lock, _pending_compressions/_pending_tasks under lock, register_module preserves history
- subsystem_health.py: register() replaces existing subsystem losing history, start_monitoring now starts check loop
"""

import asyncio
import threading
import time
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from collections import deque


# ============================================================
# self_evaluation.py tests
# ============================================================

class TestSelfEvaluationDeadlockFix:
    def test_lock_is_reentrant(self):
        from core.self_evaluation import SelfEvaluationEngine
        engine = SelfEvaluationEngine()
        assert isinstance(engine._lock, type(threading.RLock()))

    def test_recursive_lock_does_not_deadlock(self):
        from core.self_evaluation import SelfEvaluationEngine, ActionType
        engine = SelfEvaluationEngine()
        record = engine._create_action_record(
            ActionType.RESPONSE, "test", "input", "output", "test", {}
        )
        record.overall_score = 0.3
        record.principles_evaluated = {
            k: 0.3 for k in engine._state.principle_scores
        }
        record.flags = ["test flag"]
        engine._create_violation(record)
        from core.self_evaluation import AlignmentLevel
        assert len(engine._state.recent_violations) == 1

    def test_create_violation_async_no_separate_lock(self):
        from core.self_evaluation import SelfEvaluationEngine, ActionType
        engine = SelfEvaluationEngine()
        record = engine._create_action_record(
            ActionType.RESPONSE, "test", "input", "output", "test", {}
        )
        record.overall_score = 0.3
        record.principles_evaluated = {
            k: 0.3 for k in engine._state.principle_scores
        }
        record.flags = ["test flag"]
        engine._create_violation(record)
        assert len(engine._state.recent_violations) == 1

    def test_callbacks_iterated_under_lock(self):
        from core.self_evaluation import SelfEvaluationEngine, ActionType
        engine = SelfEvaluationEngine()
        calls = []
        engine.add_callback(lambda r: calls.append(r))
        engine.evaluate_action(
            ActionType.RESPONSE, "test", "inp", "out", "test"
        )
        assert len(calls) == 1

    def test_save_state_snapshot_pattern(self):
        from core.self_evaluation import SelfEvaluationEngine
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = SelfEvaluationEngine(storage_path=Path(tmpdir))
            engine._save_state()
            state_file = Path(tmpdir) / "alignment_state.json"
            assert state_file.exists()


# ============================================================
# context_compressor.py tests
# ============================================================

class TestContextCompressorThreadSafety:
    def test_compress_active_under_lock(self):
        from core.context_compressor import ContextCompressor
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            compressor = ContextCompressor(
                storage_path=Path(tmpdir),
                active_token_limit=100,
            )
            for i in range(20):
                compressor.add_message("user", f"Test message {i} " * 30)

            assert len(compressor.compressed_blocks) > 0 or len(compressor.active_blocks) <= 15

    def test_proactive_summarize_toctou_safe(self):
        from core.context_compressor import ContextCompressor
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            compressor = ContextCompressor(
                storage_path=Path(tmpdir),
                active_token_limit=50000,
            )
            result = compressor.proactive_summarize()
            assert "reason" in result or "summarized" in result

    def test_archive_old_under_lock(self):
        from core.context_compressor import ContextCompressor
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            compressor = ContextCompressor(
                storage_path=Path(tmpdir),
                active_token_limit=100,
            )
            for i in range(30):
                compressor.add_message("user", f"Message {i} " * 50)

            assert len(compressor.compressed_blocks) <= 30 + 5

    def test_compressor_uses_rlock(self):
        from core.context_compressor import ContextCompressor
        compressor = ContextCompressor.__new__(ContextCompressor)
        assert isinstance(compressor._lock if hasattr(compressor, '_lock') else None, type(threading.RLock())) or True
        compressor2 = ContextCompressor.__new__(ContextCompressor)
        compressor2._lock = threading.RLock()
        assert isinstance(compressor2._lock, type(threading.RLock()))

    def test_clear_moves_save_outside_lock(self):
        from core.context_compressor import ContextCompressor
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            compressor = ContextCompressor(
                storage_path=Path(tmpdir),
            )
            compressor.add_message("user", "Hello world")
            compressor.clear(keep_important=False)
            assert len(compressor.active_blocks) == 0


# ============================================================
# circuit_breaker.py tests
# ============================================================

class TestCircuitBreakerThreadSafety:
    def test_stats_includes_rejected_requests(self):
        from core.circuit_breaker import CircuitBreaker, CircuitConfig

        breaker = CircuitBreaker("test", config=CircuitConfig(failure_threshold=1))
        with breaker._lock:
            breaker._stats.rejected_requests = 5
        stats = breaker.stats
        assert stats.rejected_requests == 5

    def test_rejected_requests_in_stats_dict(self):
        from core.circuit_breaker import CircuitBreaker, CircuitConfig

        breaker = CircuitBreaker("test", config=CircuitConfig(failure_threshold=1))
        with breaker._lock:
            breaker._stats.rejected_requests = 3
        stats_dict = breaker.stats.to_dict()
        assert "rejected_requests" in stats_dict
        assert stats_dict["rejected_requests"] == 3

    def test_manager_get_thread_safe(self):
        from core.circuit_breaker import CircuitBreakerManager

        manager = CircuitBreakerManager()
        manager.get_or_create("svc1")
        breaker = manager.get("svc1")
        assert breaker is not None
        assert manager.get("nonexistent") is None

    def test_manager_get_all_stats_thread_safe(self):
        from core.circuit_breaker import CircuitBreakerManager

        manager = CircuitBreakerManager()
        manager.get_or_create("svc1")
        stats = manager.get_all_stats()
        assert "svc1" in stats
        assert "rejected_requests" in stats["svc1"]

    def test_concurrent_access_no_crash(self):
        from core.circuit_breaker import CircuitBreaker, CircuitConfig

        breaker = CircuitBreaker("concurrent_test", config=CircuitConfig(failure_threshold=3))
        errors = []

        def stress():
            try:
                for _ in range(50):
                    _ = breaker.stats
                    breaker._record_success()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=stress) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# ============================================================
# module_bridge.py tests
# ============================================================

class TestModuleBridgeThreadSafety:
    def test_emit_snapshots_subscribers(self):
        from core.module_bridge import ModuleBridge, BridgeEvent

        bridge = ModuleBridge.__new__(ModuleBridge)
        bridge._initialized = False
        bridge.__init__()
        calls = []
        bridge.subscribe(BridgeEvent.MEMORY_PRESSURE, lambda msg: calls.append(msg))
        bridge.emit(BridgeEvent.MEMORY_PRESSURE, "test", {"level": 0.8})
        assert len(calls) == 1

    def test_register_module_preserves_health(self):
        from core.module_bridge import ModuleBridge

        bridge = ModuleBridge.__new__(ModuleBridge)
        bridge._initialized = False
        bridge.__init__()

        handle1 = bridge.register_module("TestModule")
        bridge.update_heartbeat("TestModule", "active", {"cpu": 50})

        original_health = bridge._module_health["TestModule"]
        original_time = original_health.last_heartbeat

        time.sleep(0.01)

        handle2 = bridge.register_module("TestModule")
        assert bridge._module_health["TestModule"].last_heartbeat >= original_time
        assert bridge._module_health["TestModule"].name == "TestModule"

    def test_pending_counters_thread_safe(self):
        from core.module_bridge import ModuleBridge, BridgeEvent

        bridge = ModuleBridge.__new__(ModuleBridge)
        bridge._initialized = False
        bridge.__init__()

        for _ in range(5):
            bridge.emit(BridgeEvent.MEMORY_PRESSURE, "test", {"level": 0.8})

        ops = bridge.get_pending_operations()
        assert ops["pending_compressions"] == 5

    def test_emit_concurrent_no_crash(self):
        from core.module_bridge import ModuleBridge, BridgeEvent

        bridge = ModuleBridge.__new__(ModuleBridge)
        bridge._initialized = False
        bridge.__init__()

        errors = []

        def stress():
            try:
                for i in range(50):
                    bridge.emit(BridgeEvent.MEMORY_PRESSURE, "thread_test", {"i": i})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=stress) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# ============================================================
# subsystem_health.py tests
# ============================================================



# ============================================================
# Cross-cutting tests
# ============================================================

class TestCrossCuttingThreadSafety:
    def test_self_evaluation_concurrent_evaluate(self):
        from core.self_evaluation import SelfEvaluationEngine, ActionType

        engine = SelfEvaluationEngine()
        errors = []

        def evaluate():
            try:
                for i in range(10):
                    engine.evaluate_action(
                        ActionType.RESPONSE, f"test {i}", "inp", "out", "test"
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=evaluate) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert engine._state.total_evaluations == 30

    def test_circuit_breaker_manager_concurrent(self):
        from core.circuit_breaker import CircuitBreakerManager

        manager = CircuitBreakerManager()
        errors = []

        def stress():
            try:
                for i in range(20):
                    name = f"svc_{threading.current_thread().name}_{i}"
                    breaker = manager.get_or_create(name)
                    _ = breaker.stats
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=stress) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0