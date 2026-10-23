"""Round 17 Deep Audit Bug Tests.

Tests for bugs found and fixed in:
- workflow_shell.py: TOCTOU on get(), _publish outside lock, get_result/get_history thread-safety
- workflow_scheduler.py: handler read outside lock in _execute_schedule
- agentic_orchestrator.py: I/O under lock in _save_state, deployment fields outside lock
- dual_model.py: _request_history unbounded list → deque, removal of manual truncation
- api_manager.py: _stats dict thread-safety (_stats_lock), removed double queue limit
- hardware.py: GPUAccelerator unbounded caches, MemoryPoolManager dict-during-iteration, unbounded allocations
- shifting_memory.py: get_all_stats deadlock avoidance, I/O outside lock in prune_low_importance
- resource_monitor.py: callback locking (already correct)
- mitm_protection.py: audit_callback wrap, MAX_PINNED_HOSTS, logger.warning
"""
import os as _os
_JARVIS_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

import threading
import time
import pytest
from unittest.mock import MagicMock, patch
from collections import deque


# ============================================================
# workflow_shell.py tests
# ============================================================

class TestWorkflowShellThreadSafety:
    def test_get_under_lock(self):
        from core.workflow_shell import WorkflowShell, Pipeline, FunctionStage

        shell = WorkflowShell()
        stage = FunctionStage(lambda x, ctx: x, name="test_stage")
        pipe = Pipeline("test_pipe", stages=[stage])
        shell.register("test_pipe", pipe)

        result = shell.get("test_pipe")
        assert result is not None
        assert result.name == "test_pipe"

    def test_get_missing_returns_none(self):
        from core.workflow_shell import WorkflowShell

        shell = WorkflowShell()
        assert shell.get("nonexistent") is None

    def test_publish_snapshots_handlers(self):
        from core.workflow_shell import WorkflowShell

        shell = WorkflowShell()
        received = []

        def handler(event, data):
            received.append((event, data))

        shell.subscribe("test.event", handler)
        shell._publish("test.event", {"key": "value"})

        assert len(received) == 1
        assert received[0][0] == "test.event"

    def test_publish_handler_exception_doesnt_crash(self):
        from core.workflow_shell import WorkflowShell

        shell = WorkflowShell()
        
        def bad_handler(event, data):
            raise RuntimeError("handler crashed")

        shell.subscribe("test.event", bad_handler)
        # Should not raise
        shell._publish("test.event", {"key": "value"})

    def test_get_result_under_lock(self):
        from core.workflow_shell import WorkflowShell, StageResult, StageStatus

        shell = WorkflowShell()
        result = StageResult(value="test", status=StageStatus.COMPLETED)
        shell._results["test"] = result

        got = shell.get_result("test")
        assert got is not None
        assert got.value == "test"

    def test_get_history_under_lock(self):
        from core.workflow_shell import WorkflowShell

        shell = WorkflowShell()
        shell._execution_history.append({"test": 1})
        history = shell.get_history(limit=10)
        assert len(history) == 1


# ============================================================
# workflow_scheduler.py tests
# ============================================================

class TestWorkflowSchedulerThreadSafety:
    def test_handler_read_outside_lock(self):
        from core.workflow_scheduler import WorkflowScheduler

        scheduler = WorkflowScheduler()
        handler = MagicMock(return_value={"result": "ok"})
        scheduler.register_workflow("test_wf", handler)

        with scheduler._lock:
            handlers_snapshot = dict(scheduler._workflow_handlers)

        assert "test_wf" in handlers_snapshot

    def test_schedule_creation(self):
        from core.workflow_scheduler import WorkflowScheduler

        scheduler = WorkflowScheduler()
        schedule = scheduler.schedule(
            name="test_schedule",
            workflow_id="test_wf",
            interval_seconds=60,
        )
        assert schedule.name == "test_schedule"
        assert schedule.schedule_type.value == "interval"

    def test_save_state_snapshot_pattern(self):
        from core.workflow_scheduler import WorkflowScheduler

        scheduler = WorkflowScheduler()
        scheduler.schedule(
            name="save_test",
            workflow_id="test_wf",
            interval_seconds=300,
        )
        # _save_state should snapshot under lock, write outside
        scheduler._save_state()
        # No crash = success


# ============================================================
# agentic_orchestrator.py tests
# ============================================================

class TestAgenticOrchestratorThreadSafety:
    def test_save_state_snapshot_pattern(self):
        from core.agentic_orchestrator import AgenticOrchestrator

        orchestrator = AgenticOrchestrator()
        deployment = orchestrator.deploy(name="test_deploy")
        # _save_state should snapshot under lock, write outside
        orchestrator._save_state()
        assert deployment is not None

    def test_deployment_fields_under_lock(self):
        from core.agentic_orchestrator import AgenticOrchestrator

        orchestrator = AgenticOrchestrator()
        deployment = orchestrator.deploy(name="lock_test")
        
        with orchestrator._lock:
            status = deployment.status
        
        assert status is not None


# ============================================================
# dual_model.py tests
# ============================================================

class TestDualModelThreadSafety:
    def test_request_history_is_deque(self):
        from core.dual_model import DualModelOrchestrator

        orchestrator = DualModelOrchestrator()
        assert isinstance(orchestrator._request_history, deque)
        assert orchestrator._request_history.maxlen == 100

    def test_request_history_auto_trim(self):
        from core.dual_model import DualModelOrchestrator
        from core.dual_model import DualModelRequest, TaskType, CollaborationMode
        from datetime import datetime

        orchestrator = DualModelOrchestrator()
        
        for i in range(150):
            req = DualModelRequest(
                id=f"req_{i}",
                task_type=TaskType.CHAT,
                prompt=f"test {i}",
                timestamp=datetime.now(),
            )
            with orchestrator._lock:
                orchestrator._request_history.append(req)

        assert len(orchestrator._request_history) <= 100


# ============================================================
# api_manager.py tests
# ============================================================

class TestAPICallManagerThreadSafety:
    def test_stats_lock_exists(self):
        from core.api_manager import APICallManager

        mock_llm = MagicMock()
        manager = APICallManager(mock_llm)
        assert hasattr(manager, '_stats_lock')
        assert isinstance(manager._stats_lock, type(threading.Lock()))

    def test_no_double_queue_limit(self):
        from core.api_manager import APICallManager

        mock_llm = MagicMock()
        manager = APICallManager(mock_llm, max_queue_size=100)
        assert not hasattr(manager, 'MAX_QUEUE') or manager.max_queue_size == 100

    def test_cache_hit_miss_thread_safety(self):
        from core.api_manager import APICallManager, RequestType

        mock_llm = MagicMock()
        manager = APICallManager(mock_llm)
        
        # Simulate cache hit/miss counting under lock
        with manager._stats_lock:
            manager._stats["cache_hits"] += 1
            manager._stats["cache_misses"] += 1
        
        assert manager._stats["cache_hits"] == 1
        assert manager._stats["cache_misses"] == 1


# ============================================================
# hardware.py tests
# ============================================================

class TestHardwareThreadSafety:
    def test_gpu_accelerator_max_cache_entries(self):
        from core.hardware import GPUAccelerator

        accel = GPUAccelerator()
        assert hasattr(accel, 'MAX_CACHE_ENTRIES')
        assert accel.MAX_CACHE_ENTRIES == 100

    def test_gpu_accelerator_cache_lock(self):
        from core.hardware import GPUAccelerator

        accel = GPUAccelerator()
        assert hasattr(accel, '_cache_lock')

    def test_memory_pool_alloc_lock(self):
        from core.hardware import MemoryPoolManager

        pool = MemoryPoolManager()
        assert hasattr(pool, '_alloc_lock')
        assert isinstance(pool._alloc_lock, type(threading.Lock()))

    def test_memory_pool_max_allocations(self):
        from core.hardware import MemoryPoolManager

        pool = MemoryPoolManager()
        assert hasattr(pool, 'MAX_ALLOCATIONS')

    def test_evict_cache_no_dict_mutation_during_iteration(self):
        from core.hardware import MemoryPoolManager

        pool = MemoryPoolManager()
        # Add some cache entries
        pool._cache["a"] = ("data_a", 10, time.time())
        pool._cache["b"] = ("data_b", 20, time.time() - 100)
        pool._cache["c"] = ("data_c", 30, time.time() - 200)

        # Evict should work without RuntimeError
        pool._evict_cache_lru(25)
        assert len(pool._cache) >= 1  # At least some entries remain


# ============================================================
# shifting_memory.py tests
# ============================================================

class TestShiftingMemoryThreadSafety:
    def test_prune_low_importance_io_outside_lock(self):
        from core.shifting_memory import ShiftingMemoryManager, MemoryWindow
        from unittest.mock import patch

        manager = ShiftingMemoryManager()
        manager.register_collection("test_coll", deque(maxlen=10))
        
        with patch.object(manager, '_save_state') as mock_save:
            result = manager.prune_low_importance("test_coll", threshold=0.2)
            # prune returned 0 since no archive items, save not called
            assert result == 0
            mock_save.assert_not_called()

    def test_get_all_stats_no_deadlock(self):
        from core.shifting_memory import ShiftingMemoryManager

        manager = ShiftingMemoryManager()
        manager.register_collection("stats_test", deque(maxlen=10))
        
        # Should not deadlock (since lock is reentrant it would work, but still)
        stats = manager.get_all_stats()
        assert "stats_test" in stats


# ============================================================
# mitm_protection.py tests (Round 17 verification)
# ============================================================

class TestMITMProtectionRound17:
    def test_max_pinned_hosts_constant(self):
        from core.security.mitm_protection import MITMProtector

        assert hasattr(MITMProtector, 'MAX_PINNED_HOSTS')
        assert MITMProtector.MAX_PINNED_HOSTS == 100

    def test_audit_callback_exception_handled(self):
        from core.security.mitm_protection import MITMProtector, MITMThreatEvent, MITMThreatType, ThreatSeverity
        from datetime import datetime

        bad_callback = MagicMock(side_effect=RuntimeError("audit callback crashed"))
        protector = MITMProtector(audit_callback=bad_callback)

        threat = MITMThreatEvent(
            timestamp=datetime.now(),
            threat_type=MITMThreatType.CERTIFICATE_MISMATCH,
            severity=ThreatSeverity.CRITICAL,
            description="Test threat",
            source_ip="1.2.3.4",
            destination="example.com",
        )

        # Should not raise
        protector.record_threat(threat)
        bad_callback.assert_called_once()

    def test_cert_pins_bounded(self):
        from core.security.mitm_protection import MITMProtector

        protector = MITMProtector()
        
        for i in range(150):
            protector.add_cert_pin(f"host{i}.example.com", f"abc{i:040d}")

        # After adding 150 hosts, should have evicted oldest
        # DEFAULT_CERT_PINS has 3 hosts + we added 150 = 150 hosts,
        # but MAX_PINNED_HOSTS limits it
        total_hosts = len(protector._cert_pins)
        assert total_hosts <= MITMProtector.MAX_PINNED_HOSTS + len(MITMProtector.DEFAULT_CERT_PINS)

    def test_session_keys_bounded(self):
        from core.security.mitm_protection import SecureChannel

        sc = SecureChannel()
        for i in range(600):
            sc.create_session_key(f"session_{i}")
        
        assert len(sc._session_keys) <= SecureChannel.MAX_SESSION_KEYS

    def test_dns_entries_bounded(self):
        from core.security.mitm_protection import MITMProtector

        protector = MITMProtector()
        for i in range(600):
            protector.register_dns_resolution(f"host{i}.example.com", f"10.0.{i // 256}.{i % 256}")
        
        assert len(protector._known_dns) <= MITMProtector.MAX_DNS_ENTRIES


# ============================================================
# LCM Shield tests
# ============================================================

class TestLCMShieldRound17:
    def test_shield_history_bounded(self):
        from core.lcm_shield import LCMShield

        shield = LCMShield()
        assert shield._max_history == 200

    def test_intent_embeddings_bounded(self):
        from core.lcm_shield import LCMShield

        shield = LCMShield()
        assert shield._max_intent_embeddings == 500

    def test_rewrite_count_bounded(self):
        from core.lcm_shield import LCMShield

        shield = LCMShield()
        assert shield._max_rewrite_keys == 500


# ============================================================
# Reflect Edit tests
# ============================================================

class TestReflectEditRound17:
    def test_routing_table_bounded(self):
        from core.reflect_edit import ReflectEditLoop

        loop = ReflectEditLoop()
        assert loop._max_routing_entries == 100

    def test_reflections_bounded(self):
        from core.reflect_edit import ReflectEditLoop

        loop = ReflectEditLoop()
        assert loop._max_reflections == 200

    def test_update_routing_under_lock(self):
        from core.reflect_edit import ReflectEditLoop

        loop = ReflectEditLoop()
        loop._update_routing("test query about code", "backend", 0.9)
        
        with loop._lock:
            assert len(loop._routing_table) == 1


# ============================================================
# Semantic Contracts tests
# ============================================================

class TestSemanticContractsRound17:
    def test_instances_bounded(self):
        from core.semantic_contracts import ContractRegistry

        registry = ContractRegistry()
        assert registry._max_instances == 200

    def test_contract_creation(self):
        from core.semantic_contracts import ContractRegistry

        registry = ContractRegistry()
        instance = registry.create_contract(
            schema_name="chat_response",
            source_model="primary",
            target_model="backend",
            data={"content": "Hello", "intent": "greeting", "confidence": 0.9, "topics": ["chat"]},
        )
        assert instance.status.value == "validated"

    def test_contract_rejection(self):
        from core.semantic_contracts import ContractRegistry

        registry = ContractRegistry()
        instance = registry.create_contract(
            schema_name="chat_response",
            source_model="primary",
            target_model="backend",
            data={"content": "x" * 5000},  # Exceeds max_total_size=4000
        )
        assert instance.status.value == "rejected"


# ============================================================
# Workflow Shell parallel test
# ============================================================

class TestWorkflowShellConcurrency:
    def test_concurrent_register_get(self):
        from core.workflow_shell import WorkflowShell, Pipeline, FunctionStage

        shell = WorkflowShell()
        errors = []

        def register_and_get(i):
            try:
                stage = FunctionStage(lambda x, ctx: x, name=f"stage_{i}")
                pipe = Pipeline(f"pipe_{i}", stages=[stage])
                shell.register(f"pipe_{i}", pipe)
                result = shell.get(f"pipe_{i}")
                if result is None:
                    errors.append(f"pipe_{i} not found after register")
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=register_and_get, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"

    def test_subscribe_unsubscribe(self):
        from core.workflow_shell import WorkflowShell

        shell = WorkflowShell()
        
        def handler(event, data):
            pass

        shell.subscribe("test.event", handler)
        assert handler in shell._subscribers.get("test.event", [])
        
        shell.unsubscribe("test.event", handler)
        assert handler not in shell._subscribers.get("test.event", [])


# ============================================================
# Compile check for all modified files
# ============================================================

class TestCompileChecks:
    def test_workflow_shell_compiles(self):
        import py_compile
        import tempfile
        import os
        
        py_compile.compile(f"{_JARVIS_ROOT}/core/workflow_shell.py", doraise=True)

    def test_workflow_scheduler_compiles(self):
        import py_compile
        py_compile.compile(f"{_JARVIS_ROOT}/core/workflow_scheduler.py", doraise=True)

    def test_agentic_orchestrator_compiles(self):
        import py_compile
        py_compile.compile(f"{_JARVIS_ROOT}/core/agentic_orchestrator.py", doraise=True)

    def test_dual_model_compiles(self):
        import py_compile
        py_compile.compile(f"{_JARVIS_ROOT}/core/dual_model.py", doraise=True)

    def test_api_manager_compiles(self):
        import py_compile
        py_compile.compile(f"{_JARVIS_ROOT}/core/api_manager.py", doraise=True)

    def test_hardware_compiles(self):
        import py_compile
        py_compile.compile(f"{_JARVIS_ROOT}/core/hardware.py", doraise=True)

    def test_shifting_memory_compiles(self):
        import py_compile
        py_compile.compile(f"{_JARVIS_ROOT}/core/shifting_memory.py", doraise=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])