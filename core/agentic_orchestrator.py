"""Agentic Deployment Orchestrator for Long-Running Autonomous Operations.

Integrates task persistence, workflow scheduling, self-correction, and
autonomous goals into a unified system for long-running agentic deployments.

Key features:
- Unified interface for all agentic capabilities
- Coordinated task and goal management
- Integrated error recovery and retry logic
- Progress tracking across all subsystems
- Health monitoring and diagnostics
- Graceful startup/shutdown with state recovery

Example:
    orchestrator = get_agentic_orchestrator()
    
    # Deploy a long-running workflow
    deployment = orchestrator.deploy(
        name="daily_analysis",
        workflow="analyze_repository",
        schedule="0 6 * * *",  # 6 AM daily
        goals=["improve code quality", "reduce technical debt"],
        auto_recover=True,
    )
    
    # Monitor deployment
    status = orchestrator.get_deployment_status(deployment.deployment_id)
    
    # Start the orchestrator
    orchestrator.start()
"""
import logging
from core.helpers import atomic_write_json, safe_error_message, read_json_with_integrity
import secrets
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


class DeploymentStatus(Enum):
    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class DeploymentType(Enum):
    WORKFLOW = "workflow"
    GOAL = "goal"
    SCHEDULED = "scheduled"
    CONTINUOUS = "continuous"


@dataclass
class Deployment:
    deployment_id: str
    name: str
    deployment_type: DeploymentType
    status: DeploymentStatus = DeploymentStatus.CREATED
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    stopped_at: datetime | None = None
    
    workflow_id: str | None = None
    schedule_id: str | None = None
    goal_ids: list[str] = field(default_factory=list)
    task_ids: list[str] = field(default_factory=list)
    
    config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    auto_recover: bool = True
    max_retries: int = 3
    retry_count: int = 0
    
    last_execution_at: datetime | None = None
    last_success_at: datetime | None = None
    last_error: str | None = None
    execution_count: int = 0
    success_count: int = 0
    error_count: int = 0

    def to_dict(self) -> dict:
        return {
            "deployment_id": self.deployment_id,
            "name": self.name,
            "deployment_type": self.deployment_type.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "stopped_at": self.stopped_at.isoformat() if self.stopped_at else None,
            "workflow_id": self.workflow_id,
            "schedule_id": self.schedule_id,
            "goal_ids": self.goal_ids,
            "task_ids": self.task_ids,
            "config": self.config,
            "metadata": self.metadata,
            "auto_recover": self.auto_recover,
            "max_retries": self.max_retries,
            "retry_count": self.retry_count,
            "last_execution_at": self.last_execution_at.isoformat() if self.last_execution_at else None,
            "last_success_at": self.last_success_at.isoformat() if self.last_success_at else None,
            "last_error": self.last_error,
            "execution_count": self.execution_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Deployment":
        return cls(
            deployment_id=data["deployment_id"],
            name=data["name"],
            deployment_type=DeploymentType(data["deployment_type"]),
            status=DeploymentStatus(data.get("status", "created")),
            created_at=datetime.fromisoformat(data["created_at"]),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            stopped_at=datetime.fromisoformat(data["stopped_at"]) if data.get("stopped_at") else None,
            workflow_id=data.get("workflow_id"),
            schedule_id=data.get("schedule_id"),
            goal_ids=data.get("goal_ids", []),
            task_ids=data.get("task_ids", []),
            config=data.get("config", {}),
            metadata=data.get("metadata", {}),
            auto_recover=data.get("auto_recover", True),
            max_retries=data.get("max_retries", 3),
            retry_count=data.get("retry_count", 0),
            last_execution_at=datetime.fromisoformat(data["last_execution_at"]) if data.get("last_execution_at") else None,
            last_success_at=datetime.fromisoformat(data["last_success_at"]) if data.get("last_success_at") else None,
            last_error=data.get("last_error"),
            execution_count=data.get("execution_count", 0),
            success_count=data.get("success_count", 0),
            error_count=data.get("error_count", 0),
        )


@dataclass
class HealthStatus:
    healthy: bool
    components: dict[str, dict[str, Any]]
    last_check: datetime
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "healthy": self.healthy,
            "components": self.components,
            "last_check": self.last_check.isoformat(),
            "issues": self.issues,
        }


class AgenticOrchestrator:
    def __init__(self, storage_path: Path | None = None, llm_client=None):
        self.storage_path = storage_path or Path.home() / ".jarvis" / "agentic"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.llm = llm_client
        
        self._deployments_file = self.storage_path / "deployments.json"
        self._deployments: dict[str, Deployment] = {}
        
        self._lock = threading.RLock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        
        self._task_persistence = None
        self._workflow_scheduler = None
        self._self_correction = None
        self._autonomous_goals = None
        
        self._health_check_interval = 60.0
        self._recovery_check_interval = 30.0
        
        self._execution_handlers: dict[str, Callable] = {}
        
        self._load_state()
        self._init_subsystems()

    def _init_subsystems(self) -> None:
        try:
            from core.task_persistence import get_task_persistence
            self._task_persistence = get_task_persistence()
        except Exception as e:
            logger.warning(f"Task persistence not available: {e}")
        
        try:
            from core.workflow_scheduler import get_workflow_scheduler
            self._workflow_scheduler = get_workflow_scheduler()
        except Exception as e:
            logger.warning(f"Workflow scheduler not available: {e}")
        
        try:
            from core.self_correction import get_self_correction_engine
            self._self_correction = get_self_correction_engine(llm_client=self.llm)
        except Exception as e:
            logger.warning(f"Self-correction engine not available: {e}")
        
        try:
            from core.autonomous_goals import get_autonomous_goals
            self._autonomous_goals = get_autonomous_goals()
        except Exception as e:
            logger.warning(f"Autonomous goals not available: {e}")

    def _load_state(self) -> None:
        if self._deployments_file.exists():
            try:
                data = read_json_with_integrity(self._deployments_file)
                for dep_data in data.get("deployments", []):
                    deployment = Deployment.from_dict(dep_data)
                    self._deployments[deployment.deployment_id] = deployment
                logger.info(f"Loaded {len(self._deployments)} deployments")
            except Exception as e:
                logger.error(f"Failed to load deployments: {e}")

    def _save_state(self) -> None:
        with self._lock:
            deployments_data = [d.to_dict() for d in self._deployments.values()]
        try:
            atomic_write_json(self._deployments_file, {
                    "deployments": deployments_data,
                    "updated_at": datetime.now().isoformat(),
                })
        except Exception as e:
            logger.error(f"Failed to save deployments: {e}")

    def deploy(
        self,
        name: str,
        deployment_type: DeploymentType | str = DeploymentType.CONTINUOUS,
        workflow: str | None = None,
        schedule: str | None = None,
        goals: list[str] | None = None,
        config: dict[str, Any] | None = None,
        auto_recover: bool = True,
        max_retries: int = 3,
    ) -> Deployment:
        with self._lock:
            deployment_id = f"deploy_{secrets.token_hex(8)}"
            
            if isinstance(deployment_type, str):
                deployment_type = DeploymentType(deployment_type)
            
            deployment = Deployment(
                deployment_id=deployment_id,
                name=name,
                deployment_type=deployment_type,
                workflow_id=workflow,
                config=config or {},
                auto_recover=auto_recover,
                max_retries=max_retries,
            )
            
            if schedule and self._workflow_scheduler:
                from core.workflow_scheduler import schedule_workflow
                schedule_obj = schedule_workflow(
                    name=f"{name}_schedule",
                    workflow_id=workflow or "default",
                    cron=schedule if " " in schedule else None,
                    interval_seconds=float(schedule) if schedule.isdigit() else None,
                    params=config,
                )
                deployment.schedule_id = schedule_obj.schedule_id
            
            if goals:
                for goal_name in goals:
                    if self._autonomous_goals:
                        from core.autonomous_goals import create_goal
                        goal = create_goal(
                            name=goal_name,
                            description=f"Goal for deployment: {name}",
                        )
                        deployment.goal_ids.append(goal.goal_id)
            
            if self._task_persistence:
                from core.task_persistence import task_create
                task = task_create(
                    name=f"{name}_task",
                    task_type="deployment",
                    payload=config or {},
                )
                deployment.task_ids.append(task.task_id)
            
            self._deployments[deployment_id] = deployment
            self._save_state()
            
            logger.info(f"Created deployment: {name} ({deployment_id})")
            return deployment

    def start_deployment(self, deployment_id: str) -> bool:
        with self._lock:
            deployment = self._deployments.get(deployment_id)
            if not deployment:
                return False
            
            deployment.status = DeploymentStatus.STARTING
            self._save_state()
            
            try:
                if deployment.schedule_id and self._workflow_scheduler:
                    self._workflow_scheduler.resume(deployment.schedule_id)
                
                for goal_id in deployment.goal_ids:
                    if self._autonomous_goals:
                        self._autonomous_goals.activate(goal_id)
                
                for task_id in deployment.task_ids:
                    if self._task_persistence:
                        self._task_persistence.enqueue(task_id)
                
                deployment.status = DeploymentStatus.RUNNING
                deployment.started_at = datetime.now()
                self._save_state()
                
                logger.info(f"Started deployment: {deployment_id}")
                return True
                
            except Exception as e:
                deployment.status = DeploymentStatus.ERROR
                deployment.last_error = str(e)
                self._save_state()
                logger.error(f"Failed to start deployment {deployment_id}: {e}")
                return False

    def stop_deployment(self, deployment_id: str, reason: str = "") -> bool:
        with self._lock:
            deployment = self._deployments.get(deployment_id)
            if not deployment:
                return False
            
            deployment.status = DeploymentStatus.STOPPING
            self._save_state()
            
            try:
                if deployment.schedule_id and self._workflow_scheduler:
                    self._workflow_scheduler.pause(deployment.schedule_id)
                
                for goal_id in deployment.goal_ids:
                    if self._autonomous_goals:
                        self._autonomous_goals.pause(goal_id, reason)
                
                for task_id in deployment.task_ids:
                    if self._task_persistence:
                        self._task_persistence.cancel(task_id)
                
                deployment.status = DeploymentStatus.STOPPED
                deployment.stopped_at = datetime.now()
                self._save_state()
                
                logger.info(f"Stopped deployment: {deployment_id}")
                return True
                
            except Exception as e:
                deployment.status = DeploymentStatus.ERROR
                deployment.last_error = str(e)
                self._save_state()
                logger.error(f"Failed to stop deployment {deployment_id}: {e}")
                return False

    def get_deployment(self, deployment_id: str) -> Deployment | None:
        with self._lock:
            return self._deployments.get(deployment_id)

    def get_deployments(
        self,
        status: DeploymentStatus | None = None,
        deployment_type: DeploymentType | None = None,
    ) -> list[Deployment]:
        with self._lock:
            deployments = list(self._deployments.values())
            
            if status:
                deployments = [d for d in deployments if d.status == status]
            if deployment_type:
                deployments = [d for d in deployments if d.deployment_type == deployment_type]
            
            return sorted(deployments, key=lambda d: d.created_at, reverse=True)

    async def execute_deployment(self, deployment_id: str) -> dict[str, Any]:
        with self._lock:
            deployment = self._deployments.get(deployment_id)
            if not deployment:
                return {"success": False, "error": "Deployment not found"}
            deployment.last_execution_at = datetime.now()
            deployment.execution_count += 1
            deployment.status = DeploymentStatus.RUNNING
        
        try:
            result = {"success": True, "steps": []}
            
            if self._self_correction:
                success, correction_result = await self._self_correction.execute_with_recovery(
                    goal=deployment.name,
                    context=deployment.config,
                )
                result["correction"] = correction_result
                if not success:
                    raise Exception(correction_result.get("error", "Execution failed"))
            
            with self._lock:
                deployment.last_success_at = datetime.now()
                deployment.success_count += 1
                deployment.last_error = None
            
            for goal_id in deployment.goal_ids:
                if self._autonomous_goals:
                    goal = self._autonomous_goals.get(goal_id)
                    if goal and goal.is_actionable():
                        self._autonomous_goals.update_progress(
                            goal_id,
                            min(100, goal.progress.progress_percent + 10),
                            f"Executed deployment {deployment_id}",
                        )
            
            self._save_state()
            return result
            
        except Exception as e:
            with self._lock:
                deployment.error_count += 1
                deployment.last_error = str(e)
                deployment.retry_count += 1
                
                if deployment.auto_recover and deployment.retry_count < deployment.max_retries:
                    deployment.status = DeploymentStatus.RUNNING
                    logger.warning(f"Deployment {deployment_id} failed, will retry: {e}")
                else:
                    deployment.status = DeploymentStatus.ERROR
                    logger.error(f"Deployment {deployment_id} failed permanently: {e}")
            
            self._save_state()
            return {"success": False, "error": safe_error_message(e)}

    def check_health(self) -> HealthStatus:
        components = {}
        issues = []
        
        if self._task_persistence:
            try:
                stats = self._task_persistence.get_queue_stats()
                components["task_persistence"] = {
                    "healthy": True,
                    "stats": stats,
                }
            except Exception as e:
                components["task_persistence"] = {"healthy": False, "error": safe_error_message(e)}
                issues.append(f"Task persistence error: {e}")
        
        if self._workflow_scheduler:
            try:
                stats = self._workflow_scheduler.get_stats()
                components["workflow_scheduler"] = {
                    "healthy": True,
                    "stats": stats,
                }
            except Exception as e:
                components["workflow_scheduler"] = {"healthy": False, "error": safe_error_message(e)}
                issues.append(f"Workflow scheduler error: {e}")
        
        if self._self_correction:
            try:
                stats = self._self_correction.get_stats()
                components["self_correction"] = {
                    "healthy": True,
                    "stats": stats,
                }
            except Exception as e:
                components["self_correction"] = {"healthy": False, "error": safe_error_message(e)}
                issues.append(f"Self-correction error: {e}")
        
        if self._autonomous_goals:
            try:
                stats = self._autonomous_goals.get_stats()
                components["autonomous_goals"] = {
                    "healthy": True,
                    "stats": stats,
                }
            except Exception as e:
                components["autonomous_goals"] = {"healthy": False, "error": safe_error_message(e)}
                issues.append(f"Autonomous goals error: {e}")
        
        healthy = all(c.get("healthy", False) for c in components.values())
        
        return HealthStatus(
            healthy=healthy,
            components=components,
            last_check=datetime.now(),
            issues=issues,
        )

    def recover_interrupted(self) -> dict[str, Any]:
        recovered = {
            "tasks": [],
            "deployments": [],
            "goals": [],
        }
        
        if self._task_persistence:
            tasks = self._task_persistence.recover_interrupted()
            recovered["tasks"] = [t.task_id for t in tasks]
        
        with self._lock:
            for deployment in self._deployments.values():
                if deployment.status == DeploymentStatus.RUNNING:
                    deployment.status = DeploymentStatus.STOPPED
                    deployment.last_error = "Interrupted during execution"
                    recovered["deployments"].append(deployment.deployment_id)
        
        if self._autonomous_goals:
            for goal in self._autonomous_goals.get_actionable():
                if goal.status.value == "in_progress":
                    recovered["goals"].append(goal.goal_id)
        
        self._save_state()
        
        logger.info(f"Recovered: {len(recovered['tasks'])} tasks, {len(recovered['deployments'])} deployments, {len(recovered['goals'])} goals")
        return recovered

    def _run_health_check(self) -> None:
        health = self.check_health()
        if not health.healthy:
            logger.warning(f"Health check issues: {health.issues}")

    def _run_recovery_check(self) -> None:
        for deployment in self._deployments.values():
            if deployment.status == DeploymentStatus.ERROR and deployment.auto_recover:
                if deployment.retry_count < deployment.max_retries:
                    self.start_deployment(deployment.deployment_id)

    def start(self) -> None:
        if self._running:
            return
        
        self._running = True
        self._stop_event.clear()
        
        if self._workflow_scheduler:
            self._workflow_scheduler.start()
        
        if self._autonomous_goals:
            self._autonomous_goals.start()
        
        self.recover_interrupted()
        
        def run():
            last_health_check = time.time()
            last_recovery_check = time.time()
            
            while not self._stop_event.is_set():
                now = time.time()
                
                if now - last_health_check >= self._health_check_interval:
                    self._run_health_check()
                    last_health_check = now
                
                if now - last_recovery_check >= self._recovery_check_interval:
                    self._run_recovery_check()
                    last_recovery_check = now
                
                self._stop_event.wait(1.0)
        
        self._thread = threading.Thread(target=run, daemon=True, name="AgenticOrchestrator")
        self._thread.start()
        logger.info("Agentic orchestrator started")

    def stop(self) -> None:
        if not self._running:
            return
        
        self._running = False
        self._stop_event.set()
        
        for deployment_id in list(self._deployments.keys()):
            self.stop_deployment(deployment_id, "Orchestrator shutting down")
        
        if self._workflow_scheduler:
            self._workflow_scheduler.stop()
        
        if self._autonomous_goals:
            self._autonomous_goals.stop()
        
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        
        self._save_state()
        logger.info("Agentic orchestrator stopped")

    def register_execution_handler(self, deployment_type: str, handler: Callable) -> None:
        self._execution_handlers[deployment_type] = handler

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            status_counts = {}
            type_counts = {}
            
            for deployment in self._deployments.values():
                status_counts[deployment.status.value] = status_counts.get(deployment.status.value, 0) + 1
                type_counts[deployment.deployment_type.value] = type_counts.get(deployment.deployment_type.value, 0) + 1
            
            total_executions = sum(d.execution_count for d in self._deployments.values())
            total_successes = sum(d.success_count for d in self._deployments.values())
            total_errors = sum(d.error_count for d in self._deployments.values())
            
            return {
                "total_deployments": len(self._deployments),
                "by_status": status_counts,
                "by_type": type_counts,
                "total_executions": total_executions,
                "total_successes": total_successes,
                "total_errors": total_errors,
                "success_rate": total_successes / total_executions if total_executions > 0 else 0,
                "is_running": self._running,
            }


_orchestrator: AgenticOrchestrator | None = None
_orchestrator_lock = threading.Lock()


def get_agentic_orchestrator(llm_client=None) -> AgenticOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        with _orchestrator_lock:
            if _orchestrator is None:
                _orchestrator = AgenticOrchestrator(llm_client=llm_client)
    return _orchestrator


def deploy(**kwargs) -> Deployment:
    return get_agentic_orchestrator().deploy(**kwargs)


def get_deployment(deployment_id: str) -> Deployment | None:
    return get_agentic_orchestrator().get_deployment(deployment_id)


def start_orchestrator() -> None:
    get_agentic_orchestrator().start()


def stop_orchestrator() -> None:
    get_agentic_orchestrator().stop()