"""Round 24 audit bug tests - llm_queue, workflow_shell, workflow_scheduler, sub_agent cache."""
import threading
import time
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest




class TestWorkflowShellThreadSafety:
    def test_subscribe_bounded(self):
        from core.workflow_shell import WorkflowShell

        shell = WorkflowShell()
        for i in range(shell._max_subscribers_per_event + 5):
            shell.subscribe("test_event", lambda event, data: None)

        with shell._lock:
            assert len(shell._subscribers.get("test_event", [])) <= shell._max_subscribers_per_event

    def test_register_and_unregister(self):
        from core.workflow_shell import WorkflowShell, Pipeline

        shell = WorkflowShell()
        pipeline = Pipeline("test")
        shell.register("test_pipe", pipeline)

        result = shell.get("test_pipe")
        assert result is not None

        shell.unregister("test_pipe")
        result = shell.get("test_pipe")
        assert result is None

    def test_list_pipelines_under_lock(self):
        from core.workflow_shell import WorkflowShell, Pipeline

        shell = WorkflowShell()
        shell.register("pipe1", Pipeline("p1"))
        shell.register("pipe2", Pipeline("p2"))

        names = shell.list_pipelines()
        assert len(names) == 2
        assert "pipe1" in names

    def test_concurrent_register(self):
        from core.workflow_shell import WorkflowShell, Pipeline

        shell = WorkflowShell()
        errors = []

        def register_fn(i):
            try:
                shell.register(f"pipe_{i}", Pipeline(f"p{i}"))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=register_fn, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Errors: {errors}"
        assert len(shell.list_pipelines()) == 20


class TestWorkflowSchedulerThreadSafety:
    def test_schedule_and_get(self):
        from core.workflow_scheduler import WorkflowScheduler

        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = WorkflowScheduler(storage_path=Path(tmpdir))
            schedule = scheduler.schedule(
                name="test_schedule",
                workflow_id="test_workflow",
                interval_seconds=300,
            )

            assert schedule.schedule_id is not None
            assert schedule.next_run_at is not None

            retrieved = scheduler.get_schedule(schedule.schedule_id)
            assert retrieved is not None
            assert retrieved.name == "test_schedule"

    def test_pause_resume(self):
        from core.workflow_scheduler import WorkflowScheduler, ScheduleStatus

        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = WorkflowScheduler(storage_path=Path(tmpdir))
            schedule = scheduler.schedule(
                name="test_pause",
                workflow_id="test_workflow",
                interval_seconds=300,
            )

            scheduler.pause(schedule.schedule_id)
            retrieved = scheduler.get_schedule(schedule.schedule_id)
            assert retrieved.status == ScheduleStatus.PAUSED

            scheduler.resume(schedule.schedule_id)
            retrieved = scheduler.get_schedule(schedule.schedule_id)
            assert retrieved.status == ScheduleStatus.ACTIVE

    def test_template_registration(self):
        from core.workflow_scheduler import WorkflowScheduler

        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = WorkflowScheduler(storage_path=Path(tmpdir))
            template = scheduler.register_template(
                name="test_template",
                workflow_id="test_workflow",
                description="A test template",
                default_params={"key": "value"},
            )

            assert template.template_id is not None
            assert template.name == "test_template"

    def test_concurrent_schedule_creation(self):
        from core.workflow_scheduler import WorkflowScheduler

        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = WorkflowScheduler(storage_path=Path(tmpdir))
            errors = []

            def schedule_fn(i):
                try:
                    scheduler.schedule(
                        name=f"schedule_{i}",
                        workflow_id="test_workflow",
                        interval_seconds=300 + i,
                    )
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=schedule_fn, args=(i,)) for i in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)

            assert not errors, f"Errors: {errors}"

    def test_execution_callback_bounded(self):
        from core.workflow_scheduler import WorkflowScheduler

        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = WorkflowScheduler(storage_path=Path(tmpdir))
            for i in range(scheduler.MAX_EXECUTION_CALLBACKS + 5):
                scheduler.register_execution_callback(lambda rec: None)

            with scheduler._lock:
                assert len(scheduler._execution_callbacks) <= scheduler.MAX_EXECUTION_CALLBACKS

    def test_stats_under_lock(self):
        from core.workflow_scheduler import WorkflowScheduler

        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler = WorkflowScheduler(storage_path=Path(tmpdir))
            stats = scheduler.get_stats()
            assert isinstance(stats, dict)
            assert "total_schedules" in stats


class TestSubAgentCacheThreadSafety:
    def test_cache_concurrent_read_write(self):
        from core.sub_agent import TaskPlanner

        planner = TaskPlanner(llm_client=None)
        errors = []

        def writer(i):
            try:
                planner._store_cache(f"objective_{i}", [f"task_{i}"])
            except Exception as e:
                errors.append(e)

        def reader(i):
            try:
                result = planner._check_cache(f"objective_{i}")
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(10):
            threads.append(threading.Thread(target=writer, args=(i,)))
            threads.append(threading.Thread(target=reader, args=(i,)))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Errors: {errors}"

    def test_cache_eviction(self):
        from core.sub_agent import TaskPlanner

        planner = TaskPlanner(llm_client=None)

        for i in range(250):
            planner._store_cache(f"obj_{i}", [f"task_{i}"])

        assert len(planner._decompose_cache) <= planner._cache_max_size