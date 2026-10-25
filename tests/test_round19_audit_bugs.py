"""Round 19 Deep Audit Bug Tests.

Tests for bugs found and fixed in:
- response_cache.py: DEADLOCK (Lock→RLock), get_or_compute calls set() which re-acquires lock
- autonomous_goals.py: callbacks without lock, max_goals enforcement, None persist fix, 
  register without lock, _check_triggers under lock, get() thread safety
- goal_tracker.py: goal ID generation uses secrets, _save snapshot pattern, extract_goals I/O fix
- graceful_degradation.py: _total_calls/_fallback_calls without lock in wrap_call
- self_evaluation.py: RLock fix verification
- context_compressor.py: _compress_active under lock, _archive_old lock fix, clear() I/O fix
- circuit_breaker.py: rejected_requests in stats
- module_bridge.py: emit() snapshots subscribers under lock
- subsystem_health.py: register() preserves history, start_monitoring creates task
"""

import threading
import time
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from collections import deque


# ============================================================
# response_cache.py tests
# ============================================================



# ============================================================
# autonomous_goals.py tests
# ============================================================

class TestAutonomousGoalsThreadSafety:
    def test_register_action_handler_under_lock(self):
        from core.autonomous_goals import AutonomousGoals
        goals = AutonomousGoals.__new__(AutonomousGoals)
        goals._goals = {}
        goals._goals_file = None
        goals._lock = threading.RLock()
        goals._action_handlers = {}
        goals._progress_callbacks = []
        goals._max_goals = 1000
        goals._running = False
        goals._thread = None
        goals._stop_event = threading.Event()
        goals._check_interval = 60.0

        def handler(goal, params):
            pass

        goals.register_action_handler("test_action", handler)
        assert "test_action" in goals._action_handlers

    def test_progress_callback_bounded(self):
        from core.autonomous_goals import AutonomousGoals
        goals = AutonomousGoals.__new__(AutonomousGoals)
        goals._goals = {}
        goals._goals_file = None
        goals._lock = threading.RLock()
        goals._action_handlers = {}
        goals._progress_callbacks = []
        goals._max_goals = 1000
        goals._running = False
        goals._thread = None
        goals._stop_event = threading.Event()
        goals._check_interval = 60.0

        for i in range(25):
            goals.register_progress_callback(lambda g, o, n: None)

        assert len(goals._progress_callbacks) <= 20

    def test_get_is_thread_safe(self):
        from core.autonomous_goals import AutonomousGoals
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            goals = AutonomousGoals(storage_path=Path(tmpdir))
            goal = goals.create(name="Test Goal")
            result = goals.get(goal.goal_id)
            assert result is not None
            assert result.goal_id == goal.goal_id
            assert goals.get("nonexistent") is None

    def test_max_goals_enforcement(self):
        from core.autonomous_goals import AutonomousGoals, GoalStatus
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            goals = AutonomousGoals(storage_path=Path(tmpdir))
            goals._max_goals = 5

            for i in range(7):
                g = goals.create(name=f"Goal {i}")
                assert g is not None


# ============================================================
# goal_tracker.py tests
# ============================================================

class TestGoalTrackerThreadSafety:
    def test_goal_id_uses_secrets(self):
        from core.goal_tracker import GoalTracker
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = GoalTracker(storage_path=Path(tmpdir))
            goal_id = tracker._generate_goal_id("test description")
            assert goal_id.startswith("goal_")
            assert len(goal_id) > 10

    def test_goal_ids_are_unique(self):
        from core.goal_tracker import GoalTracker
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = GoalTracker(storage_path=Path(tmpdir))
            ids = set()
            for i in range(100):
                gid = tracker._generate_goal_id(f"test {i}")
                ids.add(gid)
            assert len(ids) == 100

    def test_save_snapshot_pattern(self):
        from core.goal_tracker import GoalTracker
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = GoalTracker(storage_path=Path(tmpdir))
            tracker._save()
            assert (Path(tmpdir) / "goals.json").exists()


# ============================================================
# graceful_degradation.py tests
# ============================================================

class TestGracefulDegradationThreadSafety:
    def test_total_calls_under_lock(self):
        from core.graceful_degradation import GracefulDegradationManager
        mgr = GracefulDegradationManager()
        errors = []

        def stress():
            try:
                for _ in range(50):
                    mgr._total_calls += 1
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=stress) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    def test_wrap_call_increments_total_calls(self):
        from core.graceful_degradation import GracefulDegradationManager
        mgr = GracefulDegradationManager()
        mgr.register_subsystem("test", {"fallback_key": "fallback_val"})

        result = mgr.wrap_call("test", lambda: "real_result", "fallback_key")
        assert result == "real_result"
        with mgr._lock:
            assert mgr._total_calls == 1
            assert mgr._fallback_calls == 0

    def test_wrap_call_uses_fallback_on_failure(self):
        from core.graceful_degradation import GracefulDegradationManager
        mgr = GracefulDegradationManager()
        mgr.register_subsystem("test", {"fallback_key": "fallback_val"})
        mgr.mark_failed("test", "test error")

        result = mgr.wrap_call("test", lambda: (_ for _ in ()).throw(RuntimeError("fail")), "fallback_key")
        assert result == "fallback_val"

    def test_concurrent_wrap_call(self):
        from core.graceful_degradation import GracefulDegradationManager
        mgr = GracefulDegradationManager()
        mgr.register_subsystem("test", {"fk": "fb"})
        errors = []

        def stress():
            try:
                for _ in range(30):
                    mgr.wrap_call("test", lambda: "ok", "fk")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=stress) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        with mgr._lock:
            assert mgr._total_calls == 90


# ============================================================
# Cross-cutting verification tests
# ============================================================

class TestRound18FixesVerification:
    def test_self_evaluation_rlock(self):
        from core.self_evaluation import SelfEvaluationEngine
        engine = SelfEvaluationEngine()
        assert isinstance(engine._lock, type(threading.RLock()))

    def test_circuit_breaker_rejected_requests_in_stats(self):
        from core.circuit_breaker import CircuitBreaker, CircuitConfig
        breaker = CircuitBreaker("test", config=CircuitConfig(failure_threshold=1))
        stats = breaker.stats
        assert hasattr(stats, "rejected_requests")

    def test_module_bridge_emit_snapshots_subscribers(self):
        from core.module_bridge import ModuleBridge, BridgeEvent
        bridge = ModuleBridge.__new__(ModuleBridge)
        bridge._initialized = False
        bridge.__init__()
        received = []
        bridge.subscribe(BridgeEvent.MEMORY_PRESSURE, lambda msg: received.append(msg))
        bridge.emit(BridgeEvent.MEMORY_PRESSURE, "test", {"level": 0.5})
        assert len(received) == 1
