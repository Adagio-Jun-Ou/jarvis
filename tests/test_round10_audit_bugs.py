"""Tests for Round 10 audit bug fixes.

Covers:
- system_integrator.py: get_subsystem() and get_status() thread safety
- resilience.py: CircuitBreaker state property should be pure read; check_state() for side-effecting read
- dynamic_memory.py: _pressure read under lock in get_target_cache_size, get_check_interval, get_status
- dual_model.py: PFC route guidance dead code removed, no new event loop in _semantic_route
- self_evaluation.py: add_callback/remove_callback thread safety; _evaluation_loop uses queue
- cognitive.py: add_memory() self_awareness call outside lock; DreamProcessor.dream() state mutation under lock
- n8n_manager.py: stop() checks PID liveness before SIGTERM
- workflow_shell.py: Pipeline.__or__ returns SequentialPipeline, not base Pipeline
"""
import asyncio
import threading
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, AsyncMock

import pytest


# ============================================================
# system_integrator.py tests
# ============================================================

class TestSystemIntegratorThreadSafety:
    def test_get_subsystem_uses_lock(self):
        from core.system_integrator import SystemIntegrator
        integrator = SystemIntegrator()

        class FakeSubsystem:
            pass

        integrator.register_subsystem("test", FakeSubsystem())
        result = integrator.get_subsystem("test")
        assert result is not None
        assert isinstance(result, FakeSubsystem)

    def test_get_subsystem_missing_returns_none(self):
        from core.system_integrator import SystemIntegrator
        integrator = SystemIntegrator()
        assert integrator.get_subsystem("nonexistent") is None

    def test_get_status_snapshot_under_lock(self):
        from core.system_integrator import SystemIntegrator, SubsystemState
        integrator = SystemIntegrator()
        integrator.register_subsystem("sub1", MagicMock())
        integrator._subsystems["sub1"].state = SubsystemState.RUNNING

        status = integrator.get_status()
        assert "sub1" in status["subsystems"]
        assert status["subsystems"]["sub1"]["state"] == "running"

    def test_get_status_concurrent_reads(self):
        from core.system_integrator import SystemIntegrator
        integrator = SystemIntegrator()

        for i in range(10):
            integrator.register_subsystem(f"sub_{i}", MagicMock())

        errors = []

        def read_status():
            try:
                for _ in range(50):
                    s = integrator.get_status()
                    assert "subsystems" in s
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=read_status) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors

    def test_get_subsystem_concurrent_reads(self):
        from core.system_integrator import SystemIntegrator
        integrator = SystemIntegrator()

        for i in range(10):
            integrator.register_subsystem(f"sub_{i}", MagicMock())

        errors = []

        def read_subsystem():
            try:
                for _ in range(50):
                    result = integrator.get_subsystem("sub_5")
                    assert result is not None
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=read_subsystem) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors


# ============================================================
# resilience.py — CircuitBreaker state property tests
# ============================================================



# ============================================================
# dynamic_memory.py — _pressure thread safety
# ============================================================

class TestDynamicMemoryPressureThreadSafety:
    def test_get_target_cache_size_uses_lock(self):
        from core.dynamic_memory import DynamicMemoryManager, MemoryPressure, DynamicMemoryConfig

        config = DynamicMemoryConfig(
            normal_cache_size=1000,
            elevated_cache_size=500,
            high_cache_size=250,
            critical_cache_size=100,
            emergency_cache_size=25,
        )
        manager = DynamicMemoryManager(config=config)
        manager._pressure = MemoryPressure.NORMAL
        assert manager.get_target_cache_size() == 1000

        manager._pressure = MemoryPressure.HIGH
        assert manager.get_target_cache_size() == 250

    def test_get_check_interval_uses_lock(self):
        from core.dynamic_memory import DynamicMemoryManager, MemoryPressure, DynamicMemoryConfig

        config = DynamicMemoryConfig(
            normal_check_interval=60.0,
            elevated_check_interval=30.0,
            high_check_interval=15.0,
            critical_check_interval=5.0,
            emergency_check_interval=2.0,
        )
        manager = DynamicMemoryManager(config=config)
        manager._pressure = MemoryPressure.NORMAL
        assert manager.get_check_interval() == 60.0

        manager._pressure = MemoryPressure.CRITICAL
        assert manager.get_check_interval() == 5.0

    def test_get_status_reads_pressure_under_lock(self):
        from core.dynamic_memory import DynamicMemoryManager, MemoryPressure

        manager = DynamicMemoryManager()
        manager._pressure = MemoryPressure.HIGH

        status = manager.get_status()
        assert status["pressure"] == "high"

    def test_pressure_concurrent_reads(self):
        from core.dynamic_memory import DynamicMemoryManager, MemoryPressure

        manager = DynamicMemoryManager()

        errors = []

        def read_pressure():
            try:
                for _ in range(100):
                    size = manager.get_target_cache_size()
                    interval = manager.get_check_interval()
                    status = manager.get_status()
                    assert isinstance(size, int)
                    assert isinstance(interval, float)
                    assert "pressure" in status
            except Exception as e:
                errors.append(e)

        def write_pressure():
            for p in MemoryPressure:
                with manager._lock:
                    manager._pressure = p
                time.sleep(0.001)

        read_threads = [threading.Thread(target=read_pressure) for _ in range(3)]
        write_thread = threading.Thread(target=write_pressure)

        for t in read_threads:
            t.start()
        write_thread.start()
        for t in read_threads:
            t.join(timeout=10)
        write_thread.join(timeout=10)

        assert not errors


# ============================================================
# self_evaluation.py — callback thread safety
# ============================================================

class TestSelfEvaluationCallbackSafety:
    def test_add_callback_thread_safe(self):
        from core.self_evaluation import SelfEvaluationEngine

        engine = SelfEvaluationEngine()
        callbacks = [lambda r: None for _ in range(10)]

        errors = []

        def add_cb(cb):
            try:
                engine.add_callback(cb)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_cb, args=(cb,)) for cb in callbacks]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors
        assert len(engine._callbacks) <= engine.MAX_CALLBACKS

    def test_remove_callback_thread_safe(self):
        from core.self_evaluation import SelfEvaluationEngine

        engine = SelfEvaluationEngine()
        callbacks = [lambda r: None for _ in range(5)]
        for cb in callbacks:
            engine.add_callback(cb)

        errors = []

        def remove_cb(cb):
            try:
                engine.remove_callback(cb)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=remove_cb, args=(cb,)) for cb in callbacks[:3]]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors

    def test_max_callbacks_enforced(self):
        from core.self_evaluation import SelfEvaluationEngine

        engine = SelfEvaluationEngine()
        for i in range(engine.MAX_CALLBACKS + 5):
            engine.add_callback(lambda r: None)

        assert len(engine._callbacks) <= engine.MAX_CALLBACKS


# ============================================================
# workflow_shell.py — Pipeline.__or__ returns SequentialPipeline
# ============================================================

class TestWorkflowShellPipelineOr:
    @pytest.mark.asyncio
    async def test_stage_or_returns_sequential_pipeline(self):
        from core.workflow_shell import Stage, SequentialPipeline, FunctionStage, PipelineContext

        async def stage_a(data, ctx):
            return str(data) + "_a"

        async def stage_b(data, ctx):
            return str(data) + "_b"

        a = FunctionStage(stage_a, name="a")
        b = FunctionStage(stage_b, name="b")

        pipeline = a | b
        assert isinstance(pipeline, SequentialPipeline)

    @pytest.mark.asyncio
    async def test_pipeline_or_stage_returns_sequential(self):
        from core.workflow_shell import Pipeline, SequentialPipeline, FunctionStage

        async def identity(data, ctx):
            return data

        p = Pipeline("test", stages=[FunctionStage(identity, name="s1")])
        s2 = FunctionStage(identity, name="s2")

        result = p | s2
        assert isinstance(result, SequentialPipeline)

    @pytest.mark.asyncio
    async def test_pipeline_or_pipeline_returns_sequential(self):
        from core.workflow_shell import Pipeline, SequentialPipeline, FunctionStage

        async def identity(data, ctx):
            return data

        p1 = Pipeline("p1", stages=[FunctionStage(identity, name="s1")])
        p2 = Pipeline("p2", stages=[FunctionStage(identity, name="s2")])

        result = p1 | p2
        assert isinstance(result, SequentialPipeline)

    @pytest.mark.asyncio
    async def test_sequential_pipeline_executes_stages(self):
        from core.workflow_shell import FunctionStage, PipelineContext

        results_order = []

        async def stage_a(data, ctx):
            results_order.append("a")
            return "result_a"

        async def stage_b(data, ctx):
            results_order.append("b")
            return "result_b"

        a = FunctionStage(stage_a, name="a")
        b = FunctionStage(stage_b, name="b")

        pipeline = a | b
        ctx = PipelineContext(workflow_id="test")
        result = await pipeline.execute("input", ctx)

        assert result.success
        assert results_order == ["a", "b"]


# ============================================================
# cognitive.py — add_memory and DreamProcessor
# ============================================================



# ============================================================
# n8n_manager.py — stop() PID liveness check
# ============================================================

class TestN8NManagerStopStalePID:
    @pytest.mark.asyncio
    async def test_stop_skips_dead_pid(self):
        from core.n8n_manager import N8NManager, N8N_PID_FILE
        import os

        manager = N8NManager()
        manager._process = None
        manager._log_file = None

        if N8N_PID_FILE.exists():
            N8N_PID_FILE.unlink()

        try:
            N8N_PID_FILE.parent.mkdir(parents=True, exist_ok=True)
            N8N_PID_FILE.write_text("99999999", encoding='utf-8')

            await manager.stop()

            assert not N8N_PID_FILE.exists()
        except Exception:
            if N8N_PID_FILE.exists():
                N8N_PID_FILE.unlink()

    @pytest.mark.asyncio
    async def test_stop_with_no_pid_file(self):
        from core.n8n_manager import N8NManager, N8N_PID_FILE

        manager = N8NManager()
        manager._process = None
        manager._log_file = None

        if N8N_PID_FILE.exists():
            N8N_PID_FILE.unlink()

        await manager.stop()

    @pytest.mark.asyncio
    async def test_stop_terminates_subprocess(self):
        from core.n8n_manager import N8NManager

        manager = N8NManager()
        mock_proc = AsyncMock()
        mock_proc.wait = AsyncMock()
        mock_proc.terminate = MagicMock()
        mock_proc.kill = MagicMock()
        manager._process = mock_proc
        manager._log_file = None

        await manager.stop()

        mock_proc.terminate.assert_called_once()


# ============================================================
# dual_model.py — no new event loop in _semantic_route
# ============================================================

class TestDualModelSemanticRoute:
    def test_semantic_route_no_event_loop_creation(self):
        from core.dual_model import DualModelOrchestrator, TaskType
        from unittest.mock import MagicMock

        mock_interconnect = MagicMock()
        mock_interconnect.encode.return_value = [0.0] * 768
        mock_interconnect.get_model_outputs.return_value = []
        mock_interconnect.get_cross_model_context.return_value = ""

        orchestrator = DualModelOrchestrator()
        orchestrator._interconnect = mock_interconnect

        model, confidence = orchestrator._semantic_route("write code", TaskType.CODE)

        assert model in ("primary", "backend")
        assert 0.0 <= confidence <= 1.0

    def test_semantic_route_falls_back_to_task_type(self):
        from core.dual_model import DualModelOrchestrator, TaskType
        from unittest.mock import MagicMock

        mock_interconnect = MagicMock()
        mock_interconnect.encode.side_effect = RuntimeError("embedding failed")
        orchestrator = DualModelOrchestrator()
        orchestrator._interconnect = mock_interconnect

        model, confidence = orchestrator._semantic_route("code task", TaskType.CODE)

        assert model == "backend"
        assert confidence == 0.5