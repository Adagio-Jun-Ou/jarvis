"""Tests for Agentic Capabilities - Task Persistence, Scheduling, Self-Correction, Goals, and Orchestrator."""
import asyncio
import pytest
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestTaskPersistence:
    def test_import_task_persistence(self):
        from core.task_persistence import (
            TaskPersistence, TaskStatus, TaskPriority, PersistentTask,
            get_task_persistence, task_create, task_enqueue,
        )
        assert TaskPersistence is not None
        assert TaskStatus.PENDING.value == "pending"
        assert TaskPriority.HIGH.value == 75

    def test_create_task(self):
        from core.task_persistence import get_task_persistence, TaskPriority
        
        persistence = get_task_persistence()
        task = persistence.create_task(
            name="test_task",
            task_type="test",
            payload={"data": "test"},
            priority=TaskPriority.HIGH,
        )
        
        assert task is not None
        assert task.name == "test_task"
        assert task.task_type == "test"
        assert task.priority == TaskPriority.HIGH
        assert task.status.value == "pending"

    def test_enqueue_and_claim(self):
        from core.task_persistence import get_task_persistence, TaskStatus
        
        persistence = get_task_persistence()
        task = persistence.create_task(
            name="enqueue_test_unique",
            task_type="test_claim",
            payload={"action": "test"},
        )
        
        assert persistence.enqueue(task.task_id)
        
        queued_task = persistence.get_task(task.task_id)
        assert queued_task.status == TaskStatus.QUEUED
        
        claimed = persistence.claim("worker_1")
        assert claimed is not None
        assert claimed.status.value == "running"

    def test_checkpoint(self):
        from core.task_persistence import get_task_persistence
        
        persistence = get_task_persistence()
        task = persistence.create_task(name="checkpoint_test", task_type="test", payload={})
        persistence.enqueue(task.task_id)
        claimed = persistence.claim("worker_1")
        
        assert persistence.checkpoint(
            claimed.task_id,
            {"step": 1, "data": "progress"},
            progress=50.0,
            message="Halfway done",
        )
        
        checkpoint = persistence.get_latest_checkpoint(claimed.task_id)
        assert checkpoint is not None
        assert checkpoint.progress == 50.0
        assert checkpoint.state["step"] == 1

    def test_complete_task(self):
        from core.task_persistence import get_task_persistence
        
        persistence = get_task_persistence()
        task = persistence.create_task(name="complete_test", task_type="test", payload={})
        persistence.enqueue(task.task_id)
        claimed = persistence.claim("worker_1")
        
        assert persistence.complete(claimed.task_id, {"output": "done"})
        
        completed = persistence.get_task(claimed.task_id)
        assert completed.status.value == "completed"
        assert completed.result["output"] == "done"

    def test_fail_and_retry(self):
        from core.task_persistence import get_task_persistence, TaskStatus
        
        persistence = get_task_persistence()
        task = persistence.create_task(
            name="fail_test",
            task_type="test",
            payload={},
            max_retries=2,
        )
        persistence.enqueue(task.task_id)
        claimed = persistence.claim("worker_1")
        
        assert persistence.fail(claimed.task_id, "Test error", retry=True)
        
        failed = persistence.get_task(claimed.task_id)
        assert failed.retry_count == 1
        assert failed.status == TaskStatus.PENDING

    def test_recover_interrupted(self):
        from core.task_persistence import get_task_persistence, TaskStatus
        
        persistence = get_task_persistence()
        task = persistence.create_task(name="interrupt_test", task_type="test", payload={})
        persistence.enqueue(task.task_id)
        claimed = persistence.claim("worker_1")
        
        recovered = persistence.recover_interrupted()
        assert len(recovered) > 0
        
        recovered_task = persistence.get_task(claimed.task_id)
        assert recovered_task.status in (TaskStatus.QUEUED, TaskStatus.PENDING)


class TestWorkflowScheduler:
    def test_import_scheduler(self):
        from core.workflow_scheduler import (
            WorkflowScheduler, ScheduledWorkflow, ScheduleType,
            get_workflow_scheduler, schedule_workflow,
        )
        assert WorkflowScheduler is not None
        assert ScheduleType.CRON.value == "cron"

    def test_cron_expression(self):
        from core.workflow_scheduler import CronExpression
        
        cron = CronExpression.parse("0 9 * * *")
        dt = datetime(2024, 1, 15, 9, 0)
        assert cron.matches(dt)
        
        dt_off = datetime(2024, 1, 15, 10, 0)
        assert not cron.matches(dt_off)

    def test_schedule_cron(self):
        from core.workflow_scheduler import get_workflow_scheduler
        
        scheduler = get_workflow_scheduler()
        schedule = scheduler.schedule(
            name="daily_report",
            workflow_id="report_gen",
            cron="0 9 * * *",
        )
        
        assert schedule is not None
        assert schedule.schedule_type.value == "cron"
        assert schedule.status.value == "active"
        assert schedule.next_run_at is not None

    def test_schedule_interval(self):
        from core.workflow_scheduler import get_workflow_scheduler
        
        scheduler = get_workflow_scheduler()
        schedule = scheduler.schedule(
            name="health_check",
            workflow_id="check_health",
            interval_seconds=300,
        )
        
        assert schedule is not None
        assert schedule.schedule_type.value == "interval"
        assert schedule.interval_seconds == 300

    def test_template(self):
        from core.workflow_scheduler import get_workflow_scheduler
        
        scheduler = get_workflow_scheduler()
        template = scheduler.register_template(
            name="backup_template",
            workflow_id="backup",
            description="Backup workflow template",
            default_params={"compression": True},
        )
        
        assert template is not None
        assert template.name == "backup_template"
        
        schedule = scheduler.schedule_from_template(
            template_name="backup_template",
            cron="0 2 * * *",
        )
        assert schedule is not None

    def test_pause_resume(self):
        from core.workflow_scheduler import get_workflow_scheduler, ScheduleStatus
        
        scheduler = get_workflow_scheduler()
        schedule = scheduler.schedule(
            name="pause_test",
            workflow_id="test",
            cron="*/5 * * * *",
        )
        
        assert scheduler.pause(schedule.schedule_id)
        
        paused = scheduler.get_schedule(schedule.schedule_id)
        assert paused.status == ScheduleStatus.PAUSED
        
        assert scheduler.resume(schedule.schedule_id)
        
        resumed = scheduler.get_schedule(schedule.schedule_id)
        assert resumed.status == ScheduleStatus.ACTIVE


class TestSelfCorrection:
    def test_import_self_correction(self):
        from core.self_correction import (
            SelfCorrectionEngine, ExecutionPlan, ExecutionStep,
            RecoveryStrategy, ErrorType,
        )
        assert SelfCorrectionEngine is not None
        assert RecoveryStrategy.RETRY.value == "retry"
        assert ErrorType.TRANSIENT.value == "transient"

    def test_decompose_goal(self):
        from core.self_correction import get_self_correction_engine
        
        engine = get_self_correction_engine()
        plan = engine.decompose_goal("Analyze the code and create a report")
        
        assert plan is not None
        assert len(plan.steps) > 0
        assert plan.goal == "Analyze the code and create a report"

    def test_classify_error(self):
        from core.self_correction import get_self_correction_engine, ErrorType
        
        engine = get_self_correction_engine()
        
        assert engine.classify_error("connection timeout") == ErrorType.TIMEOUT
        assert engine.classify_error("permission denied") == ErrorType.PERMISSION
        assert engine.classify_error("out of memory") == ErrorType.RESOURCE

    def test_get_recovery_strategies(self):
        from core.self_correction import get_self_correction_engine, RecoveryStrategy
        
        engine = get_self_correction_engine()
        
        strategies = engine.get_recovery_strategies("connection failed")
        assert RecoveryStrategy.RETRY in strategies or RecoveryStrategy.ROLLBACK in strategies

    def test_execution_step(self):
        from core.self_correction import ExecutionStep, StepStatus
        
        step = ExecutionStep(
            step_id="step_1",
            name="Test Step",
            description="Test step description",
            action="test_action",
            params={"key": "value"},
        )
        
        assert step.step_id == "step_1"
        assert step.status == StepStatus.PENDING
        assert step.retry_count == 0

    def test_record_outcome(self):
        from core.self_correction import get_self_correction_engine, RecoveryStrategy
        
        engine = get_self_correction_engine()
        plan = engine.decompose_goal("Test goal")
        
        engine.record_outcome(
            goal_id=plan.goal_id,
            success=True,
            error=None,
            strategy_used=RecoveryStrategy.RETRY,
        )
        
        stats = engine.get_stats()
        assert stats["total_outcomes"] >= 1


class TestAutonomousGoals:
    def test_import_autonomous_goals(self):
        from core.autonomous_goals import (
            AutonomousGoals, AutonomousGoal, GoalStatus, GoalPriority,
            get_autonomous_goals, create_goal,
        )
        assert AutonomousGoals is not None
        assert GoalStatus.ACTIVE.value == "active"
        assert GoalPriority.HIGH.value == 75

    def test_create_goal(self):
        from core.autonomous_goals import get_autonomous_goals, GoalPriority
        
        goals = get_autonomous_goals()
        goal = goals.create(
            name="Improve code quality",
            description="Reduce technical debt",
            priority=GoalPriority.HIGH,
        )
        
        assert goal is not None
        assert goal.name == "Improve code quality"
        assert goal.priority == GoalPriority.HIGH
        assert goal.status.value == "draft"

    def test_activate_goal(self):
        from core.autonomous_goals import get_autonomous_goals, GoalStatus
        
        goals = get_autonomous_goals()
        goal = goals.create(name="Test activation")
        
        assert goals.activate(goal.goal_id)
        
        active_goal = goals.get(goal.goal_id)
        assert active_goal.status == GoalStatus.ACTIVE

    def test_update_progress(self):
        from core.autonomous_goals import get_autonomous_goals
        
        goals = get_autonomous_goals()
        goal = goals.create(name="Test progress")
        goals.activate(goal.goal_id)
        
        assert goals.update_progress(goal.goal_id, 50.0, "Halfway done")
        
        updated = goals.get(goal.goal_id)
        assert updated.progress.progress_percent == 50.0
        assert len(updated.progress.notes) > 0

    def test_milestones(self):
        from core.autonomous_goals import get_autonomous_goals
        
        goals = get_autonomous_goals()
        goal = goals.create(
            name="Test milestones",
            milestones=[
                {"name": "First step", "target_percent": 25},
                {"name": "Second step", "target_percent": 50},
                {"name": "Final", "target_percent": 100},
            ],
        )
        
        assert len(goal.milestones) == 3
        
        goals.activate(goal.goal_id)
        goals.update_progress(goal.goal_id, 60)
        
        updated = goals.get(goal.goal_id)
        assert updated.milestones[0].achieved
        assert updated.milestones[1].achieved

    def test_block_unblock(self):
        from core.autonomous_goals import get_autonomous_goals, GoalStatus
        
        goals = get_autonomous_goals()
        goal = goals.create(name="Test block")
        goals.activate(goal.goal_id)
        
        assert goals.block(goal.goal_id, "External dependency")
        
        blocked = goals.get(goal.goal_id)
        assert blocked.status == GoalStatus.BLOCKED
        assert "External dependency" in blocked.blocked_by
        
        assert goals.unblock(goal.goal_id)
        
        unblocked = goals.get(goal.goal_id)
        assert unblocked.status == GoalStatus.ACTIVE

    def test_get_actionable(self):
        from core.autonomous_goals import get_autonomous_goals
        
        goals = get_autonomous_goals()
        
        goal1 = goals.create(name="Actionable 1", priority=75)
        goal2 = goals.create(name="Actionable 2", priority=50)
        
        goals.activate(goal1.goal_id)
        goals.activate(goal2.goal_id)
        
        actionable = goals.get_actionable()
        assert len(actionable) >= 2
        
        assert actionable[0].priority.value >= actionable[1].priority.value

    def test_triggers(self):
        from core.autonomous_goals import get_autonomous_goals, TriggerType
        
        goals = get_autonomous_goals()
        goal = goals.create(name="Test triggers")
        
        trigger = goals.add_trigger(
            goal_id=goal.goal_id,
            trigger_type=TriggerType.PROGRESS_BASED,
            condition={"threshold": 50},
            action="notify",
        )
        
        assert trigger is not None
        assert trigger.trigger_type == TriggerType.PROGRESS_BASED


class TestAgenticOrchestrator:
    def test_import_orchestrator(self):
        from core.agentic_orchestrator import (
            AgenticOrchestrator, Deployment, DeploymentStatus, DeploymentType,
            get_agentic_orchestrator, deploy,
        )
        assert AgenticOrchestrator is not None
        assert DeploymentStatus.RUNNING.value == "running"
        assert DeploymentType.WORKFLOW.value == "workflow"

    def test_create_deployment(self):
        from core.agentic_orchestrator import get_agentic_orchestrator, DeploymentType
        
        orchestrator = get_agentic_orchestrator()
        deployment = orchestrator.deploy(
            name="test_deployment",
            deployment_type=DeploymentType.CONTINUOUS,
            config={"test": True},
        )
        
        assert deployment is not None
        assert deployment.name == "test_deployment"
        assert deployment.deployment_type == DeploymentType.CONTINUOUS

    def test_start_stop_deployment(self):
        from core.agentic_orchestrator import get_agentic_orchestrator, DeploymentStatus
        
        orchestrator = get_agentic_orchestrator()
        deployment = orchestrator.deploy(name="start_stop_test")
        
        assert orchestrator.start_deployment(deployment.deployment_id)
        
        started = orchestrator.get_deployment(deployment.deployment_id)
        assert started.status == DeploymentStatus.RUNNING
        
        assert orchestrator.stop_deployment(deployment.deployment_id)
        
        stopped = orchestrator.get_deployment(deployment.deployment_id)
        assert stopped.status == DeploymentStatus.STOPPED

    def test_get_deployments(self):
        from core.agentic_orchestrator import get_agentic_orchestrator, DeploymentType
        
        orchestrator = get_agentic_orchestrator()
        
        orchestrator.deploy(name="list_test_1", deployment_type=DeploymentType.WORKFLOW)
        orchestrator.deploy(name="list_test_2", deployment_type=DeploymentType.SCHEDULED)
        
        all_deployments = orchestrator.get_deployments()
        assert len(all_deployments) >= 2
        
        workflows = orchestrator.get_deployments(deployment_type=DeploymentType.WORKFLOW)
        assert all(d.deployment_type == DeploymentType.WORKFLOW for d in workflows)

    def test_health_check(self):
        from core.agentic_orchestrator import get_agentic_orchestrator
        
        orchestrator = get_agentic_orchestrator()
        health = orchestrator.check_health()
        
        assert health is not None
        assert "components" in health.to_dict()
        assert "last_check" in health.to_dict()

    def test_get_stats(self):
        from core.agentic_orchestrator import get_agentic_orchestrator
        
        orchestrator = get_agentic_orchestrator()
        stats = orchestrator.get_stats()
        
        assert "total_deployments" in stats
        assert "is_running" in stats


class TestIntegration:
    def test_full_deployment_flow(self):
        from core.agentic_orchestrator import get_agentic_orchestrator, DeploymentType
        from core.autonomous_goals import get_autonomous_goals
        from core.task_persistence import get_task_persistence
        
        orchestrator = get_agentic_orchestrator()
        
        deployment = orchestrator.deploy(
            name="integration_test",
            deployment_type=DeploymentType.CONTINUOUS,
            goals=["Complete integration test"],
            config={"test_mode": True},
        )
        
        assert len(deployment.goal_ids) > 0
        assert len(deployment.task_ids) > 0
        
        goals = get_autonomous_goals()
        goal = goals.get(deployment.goal_ids[0])
        
        if goal:
            goals.activate(goal.goal_id)
            goals.update_progress(goal.goal_id, 100, "Test complete")
            
            assert goal.progress.progress_percent == 100.0

    @pytest.mark.asyncio
    async def test_self_correction_integration(self):
        from core.self_correction import get_self_correction_engine
        
        engine = get_self_correction_engine()
        
        plan = engine.decompose_goal("Test task")
        
        assert plan is not None
        assert len(plan.steps) >= 1
        assert plan.goal == "Test task"
        
        success = engine.classify_error("connection timeout")
        assert success is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])