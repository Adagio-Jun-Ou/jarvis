"""Workflow Scheduler for Long-Running Agentic Deployments.

Provides cron-like scheduling, workflow templates, and execution management
for autonomous agent operations.

Key features:
- Cron pattern scheduling (e.g., "0 9 * * *" for 9 AM daily)
- One-time and recurring schedules
- Workflow templates with parameterization
- Schedule management (pause, resume, modify)
- Execution history and logging
- Timezone support
- Missed schedule handling (catch-up or skip)

Example:
    scheduler = get_workflow_scheduler()
    
    # Schedule a daily report
    schedule_id = scheduler.schedule(
        name="morning_report",
        workflow="daily_report",
        cron="0 9 * * *",  # Every day at 9 AM
        params={"report_type": "summary"},
    )
    
    # Schedule with interval
    scheduler.schedule(
        name="health_check",
        workflow="system_check",
        interval_seconds=300,  # Every 5 minutes
    )
    
    # Use a template
    scheduler.schedule_from_template(
        template_name="weekly_backup",
        params={"target": "/backup"},
    )
"""
import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable
import secrets

logger = logging.getLogger(__name__)


class ScheduleStatus(Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    DISABLED = "disabled"
    ERROR = "error"


class ScheduleType(Enum):
    CRON = "cron"
    INTERVAL = "interval"
    ONCE = "once"
    DELAYED = "delayed"


class MissedRunPolicy(Enum):
    CATCH_UP = "catch_up"
    SKIP = "skip"
    RUN_ONCE = "run_once"


@dataclass
class CronExpression:
    minute: str = "*"
    hour: str = "*"
    day_of_month: str = "*"
    month: str = "*"
    day_of_week: str = "*"

    @classmethod
    def parse(cls, cron_str: str) -> "CronExpression":
        parts = cron_str.strip().split()
        if len(parts) != 5:
            raise ValueError(f"Invalid cron expression: {cron_str}")
        return cls(
            minute=parts[0],
            hour=parts[1],
            day_of_month=parts[2],
            month=parts[3],
            day_of_week=parts[4],
        )

    def matches(self, dt: datetime) -> bool:
        return (
            self._field_matches(self.minute, dt.minute, 0, 59) and
            self._field_matches(self.hour, dt.hour, 0, 23) and
            self._field_matches(self.day_of_month, dt.day, 1, 31) and
            self._field_matches(self.month, dt.month, 1, 12) and
            self._field_matches(self.day_of_week, dt.weekday(), 0, 6)
        )

    def _field_matches(self, field: str, value: int, min_val: int, max_val: int) -> bool:
        if field == "*":
            return True

        if "/" in field:
            base, step = field.split("/")
            if base == "*":
                base_val = min_val
            else:
                base_val = int(base)
            step = int(step)
            return (value - base_val) % step == 0

        if "," in field:
            return any(self._field_matches(f.strip(), value, min_val, max_val) for f in field.split(","))

        if "-" in field:
            start, end = field.split("-")
            return int(start) <= value <= int(end)

        return int(field) == value

    def next_run(self, after: datetime) -> datetime:
        next_dt = after.replace(second=0, microsecond=0) + timedelta(minutes=1)
        for _ in range(366 * 24 * 60):
            if self.matches(next_dt):
                return next_dt
            next_dt += timedelta(minutes=1)
        return after + timedelta(days=365)


@dataclass
class ScheduledWorkflow:
    schedule_id: str
    name: str
    workflow_id: str
    schedule_type: ScheduleType
    status: ScheduleStatus = ScheduleStatus.ACTIVE
    
    cron_expression: str | None = None
    interval_seconds: float | None = None
    run_at: datetime | None = None
    
    params: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    created_at: datetime = field(default_factory=datetime.now)
    next_run_at: datetime | None = None
    last_run_at: datetime | None = None
    last_run_status: str | None = None
    last_run_result: dict[str, Any] = field(default_factory=dict)
    
    run_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    
    max_runs: int | None = None
    end_at: datetime | None = None
    timezone: str = "UTC"
    
    missed_run_policy: MissedRunPolicy = MissedRunPolicy.SKIP
    retry_on_failure: bool = True
    max_retries: int = 3
    retry_delay_seconds: float = 60.0
    
    tags: list[str] = field(default_factory=list)
    priority: int = 50

    def to_dict(self) -> dict:
        return {
            "schedule_id": self.schedule_id,
            "name": self.name,
            "workflow_id": self.workflow_id,
            "schedule_type": self.schedule_type.value,
            "status": self.status.value,
            "cron_expression": self.cron_expression,
            "interval_seconds": self.interval_seconds,
            "run_at": self.run_at.isoformat() if self.run_at else None,
            "params": self.params,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "next_run_at": self.next_run_at.isoformat() if self.next_run_at else None,
            "last_run_at": self.last_run_at.isoformat() if self.last_run_at else None,
            "last_run_status": self.last_run_status,
            "last_run_result": self.last_run_result,
            "run_count": self.run_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "max_runs": self.max_runs,
            "end_at": self.end_at.isoformat() if self.end_at else None,
            "timezone": self.timezone,
            "missed_run_policy": self.missed_run_policy.value,
            "retry_on_failure": self.retry_on_failure,
            "max_retries": self.max_retries,
            "retry_delay_seconds": self.retry_delay_seconds,
            "tags": self.tags,
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ScheduledWorkflow":
        return cls(
            schedule_id=data["schedule_id"],
            name=data["name"],
            workflow_id=data["workflow_id"],
            schedule_type=ScheduleType(data["schedule_type"]),
            status=ScheduleStatus(data.get("status", "active")),
            cron_expression=data.get("cron_expression"),
            interval_seconds=data.get("interval_seconds"),
            run_at=datetime.fromisoformat(data["run_at"]) if data.get("run_at") else None,
            params=data.get("params", {}),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            next_run_at=datetime.fromisoformat(data["next_run_at"]) if data.get("next_run_at") else None,
            last_run_at=datetime.fromisoformat(data["last_run_at"]) if data.get("last_run_at") else None,
            last_run_status=data.get("last_run_status"),
            last_run_result=data.get("last_run_result", {}),
            run_count=data.get("run_count", 0),
            success_count=data.get("success_count", 0),
            failure_count=data.get("failure_count", 0),
            max_runs=data.get("max_runs"),
            end_at=datetime.fromisoformat(data["end_at"]) if data.get("end_at") else None,
            timezone=data.get("timezone", "UTC"),
            missed_run_policy=MissedRunPolicy(data.get("missed_run_policy", "skip")),
            retry_on_failure=data.get("retry_on_failure", True),
            max_retries=data.get("max_retries", 3),
            retry_delay_seconds=data.get("retry_delay_seconds", 60.0),
            tags=data.get("tags", []),
            priority=data.get("priority", 50),
        )


@dataclass
class WorkflowTemplate:
    template_id: str
    name: str
    description: str
    workflow_id: str
    parameters: dict[str, Any] = field(default_factory=dict)
    default_params: dict[str, Any] = field(default_factory=dict)
    param_schema: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    use_count: int = 0

    def to_dict(self) -> dict:
        return {
            "template_id": self.template_id,
            "name": self.name,
            "description": self.description,
            "workflow_id": self.workflow_id,
            "parameters": self.parameters,
            "default_params": self.default_params,
            "param_schema": self.param_schema,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "use_count": self.use_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WorkflowTemplate":
        return cls(
            template_id=data["template_id"],
            name=data["name"],
            description=data.get("description", ""),
            workflow_id=data["workflow_id"],
            parameters=data.get("parameters", {}),
            default_params=data.get("default_params", {}),
            param_schema=data.get("param_schema", {}),
            tags=data.get("tags", []),
            created_at=datetime.fromisoformat(data["created_at"]),
            use_count=data.get("use_count", 0),
        )


@dataclass
class ExecutionRecord:
    execution_id: str
    schedule_id: str
    workflow_id: str
    started_at: datetime
    completed_at: datetime | None = None
    status: str = "running"
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    duration_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "execution_id": self.execution_id,
            "schedule_id": self.schedule_id,
            "workflow_id": self.workflow_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "duration_seconds": self.duration_seconds,
        }


class WorkflowScheduler:
    MAX_EXECUTION_CALLBACKS = 20

    def __init__(self, storage_path: Path | None = None):
        self.storage_path = storage_path or Path.home() / ".jarvis" / "scheduler"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self._schedules_file = self.storage_path / "schedules.json"
        self._templates_file = self.storage_path / "templates.json"
        self._history_file = self.storage_path / "history.json"
        
        self._schedules: dict[str, ScheduledWorkflow] = {}
        self._templates: dict[str, WorkflowTemplate] = {}
        self._history: list[ExecutionRecord] = []
        
        self._lock = threading.RLock()
        self._dirty = False
        self._running = False
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        
        self._workflow_handlers: dict[str, Callable] = {}
        self._execution_callbacks: list[Callable] = []
        
        self._max_history = 10000
        self._tick_interval = 1.0
        
        self._load_state()

    def _load_state(self) -> None:
        try:
            if self._schedules_file.exists():
                with open(self._schedules_file, encoding='utf-8') as f:
                    data = json.load(f)
                    for sched_data in data.get("schedules", []):
                        schedule = ScheduledWorkflow.from_dict(sched_data)
                        self._schedules[schedule.schedule_id] = schedule
                logger.info(f"Loaded {len(self._schedules)} schedules")

            if self._templates_file.exists():
                with open(self._templates_file, encoding='utf-8') as f:
                    data = json.load(f)
                    for tmpl_data in data.get("templates", []):
                        template = WorkflowTemplate.from_dict(tmpl_data)
                        self._templates[template.template_id] = template
                logger.info(f"Loaded {len(self._templates)} templates")

            if self._history_file.exists():
                with open(self._history_file, encoding='utf-8') as f:
                    data = json.load(f)
                    for rec_data in data.get("history", []):
                        record = ExecutionRecord(
                            execution_id=rec_data["execution_id"],
                            schedule_id=rec_data["schedule_id"],
                            workflow_id=rec_data["workflow_id"],
                            started_at=datetime.fromisoformat(rec_data["started_at"]),
                            completed_at=datetime.fromisoformat(rec_data["completed_at"]) if rec_data.get("completed_at") else None,
                            status=rec_data.get("status", "completed"),
                            result=rec_data.get("result", {}),
                            error=rec_data.get("error"),
                            duration_seconds=rec_data.get("duration_seconds", 0),
                        )
                        self._history.append(record)
                logger.info(f"Loaded {len(self._history)} execution records")
                
        except Exception as e:
            logger.error(f"Failed to load scheduler state: {e}")

    def _save_state(self) -> None:
        """Save state to disk atomically. Acquires lock for snapshot, writes outside lock."""
        with self._lock:
            schedules_data = [s.to_dict() for s in self._schedules.values()]
            templates_data = [t.to_dict() for t in self._templates.values()]
            history_data = [h.to_dict() for h in self._history[-self._max_history:]]
            updated_at = datetime.now().isoformat()
        try:
            tmp = self._schedules_file.with_suffix(".tmp")
            with open(tmp, "w", encoding='utf-8') as f:
                json.dump({"schedules": schedules_data, "updated_at": updated_at}, f)
            tmp.replace(self._schedules_file)

            tmp = self._templates_file.with_suffix(".tmp")
            with open(tmp, "w", encoding='utf-8') as f:
                json.dump({"templates": templates_data}, f)
            tmp.replace(self._templates_file)

            tmp = self._history_file.with_suffix(".tmp")
            with open(tmp, "w", encoding='utf-8') as f:
                json.dump({"history": history_data}, f)
            tmp.replace(self._history_file)
        except Exception as e:
            logger.error(f"Failed to save scheduler state: {e}")

    def _mark_dirty(self) -> None:
        """Mark state as needing save. Call inside lock, flush outside."""
        self._dirty = True

    def _flush_if_dirty(self) -> None:
        """Flush dirty state to disk if needed. Call outside lock."""
        with self._lock:
            if not self._dirty:
                return
            self._dirty = False
        self._save_state()

    def schedule(
        self,
        name: str,
        workflow_id: str,
        cron: str | None = None,
        interval_seconds: float | None = None,
        run_at: datetime | None = None,
        params: dict[str, Any] | None = None,
        max_runs: int | None = None,
        end_at: datetime | None = None,
        missed_run_policy: MissedRunPolicy = MissedRunPolicy.SKIP,
        tags: list[str] | None = None,
        priority: int = 50,
    ) -> ScheduledWorkflow:
        with self._lock:
            schedule_id = f"sched_{secrets.token_hex(8)}"
            
            if cron:
                schedule_type = ScheduleType.CRON
                CronExpression.parse(cron)
            elif interval_seconds:
                schedule_type = ScheduleType.INTERVAL
            elif run_at:
                schedule_type = ScheduleType.ONCE
            else:
                raise ValueError("Must specify cron, interval_seconds, or run_at")
            
            schedule = ScheduledWorkflow(
                schedule_id=schedule_id,
                name=name,
                workflow_id=workflow_id,
                schedule_type=schedule_type,
                cron_expression=cron,
                interval_seconds=interval_seconds,
                run_at=run_at,
                params=params or {},
                max_runs=max_runs,
                end_at=end_at,
                missed_run_policy=missed_run_policy,
                tags=tags or [],
                priority=priority,
            )
            
            schedule.next_run_at = self._calculate_next_run(schedule)
            
            self._schedules[schedule_id] = schedule
            self._mark_dirty()
        self._flush_if_dirty()
        
        logger.info(f"Scheduled {name} ({schedule_id}): next run at {schedule.next_run_at}")
        return schedule

    def schedule_from_template(
        self,
        template_name: str,
        params: dict[str, Any] | None = None,
        cron: str | None = None,
        interval_seconds: float | None = None,
        run_at: datetime | None = None,
        **kwargs,
    ) -> ScheduledWorkflow:
        with self._lock:
            template = self._templates.get(template_name)
            if not template:
                template = next((t for t in self._templates.values() if t.name == template_name), None)
            
            if not template:
                raise ValueError(f"Template not found: {template_name}")
            
            merged_params = {**template.default_params, **(params or {})}
            
            template.use_count += 1
        
        self._save_state()
        
        return self.schedule(
            name=f"{template.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            workflow_id=template.workflow_id,
            cron=cron or template.parameters.get("cron"),
            interval_seconds=interval_seconds or template.parameters.get("interval_seconds"),
            run_at=run_at,
            params=merged_params,
            **kwargs,
        )

    def _calculate_next_run(self, schedule: ScheduledWorkflow) -> datetime | None:
        now = datetime.now()
        
        if schedule.status != ScheduleStatus.ACTIVE:
            return None
        
        if schedule.max_runs and schedule.run_count >= schedule.max_runs:
            return None
        
        if schedule.end_at and now > schedule.end_at:
            return None
        
        if schedule.schedule_type == ScheduleType.CRON:
            cron = CronExpression.parse(schedule.cron_expression)
            
            if schedule.next_run_at and schedule.next_run_at < now:
                if schedule.missed_run_policy == MissedRunPolicy.CATCH_UP:
                    return schedule.next_run_at
                else:
                    return cron.next_run(now)
            
            return cron.next_run(now)
        
        elif schedule.schedule_type == ScheduleType.INTERVAL:
            if schedule.last_run_at:
                return schedule.last_run_at + timedelta(seconds=schedule.interval_seconds)
            return now + timedelta(seconds=schedule.interval_seconds)
        
        elif schedule.schedule_type == ScheduleType.ONCE:
            if not schedule.run_at:
                return None
            if schedule.run_count > 0:
                return None
            return schedule.run_at
        
        elif schedule.schedule_type == ScheduleType.DELAYED:
            if schedule.run_at and schedule.run_at > now:
                return schedule.run_at
            return None
        
        return None

    def register_workflow(self, workflow_id: str, handler: Callable) -> None:
        with self._lock:
            self._workflow_handlers[workflow_id] = handler
        logger.info(f"Registered workflow handler: {workflow_id}")

    def register_template(
        self,
        name: str,
        workflow_id: str,
        description: str = "",
        default_params: dict[str, Any] | None = None,
        param_schema: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> WorkflowTemplate:
        with self._lock:
            template_id = f"tmpl_{secrets.token_hex(8)}"
            
            template = WorkflowTemplate(
                template_id=template_id,
                name=name,
                description=description,
                workflow_id=workflow_id,
                default_params=default_params or {},
                param_schema=param_schema or {},
                tags=tags or [],
            )
            
            self._templates[template_id] = template
            self._mark_dirty()
        self._flush_if_dirty()
        
        logger.info(f"Registered template: {name}")
        return template

    def get_schedule(self, schedule_id: str) -> ScheduledWorkflow | None:
        with self._lock:
            return self._schedules.get(schedule_id)

    def get_schedules(
        self,
        status: ScheduleStatus | None = None,
        workflow_id: str | None = None,
        tags: list[str] | None = None,
    ) -> list[ScheduledWorkflow]:
        with self._lock:
            schedules = list(self._schedules.values())
            
            if status:
                schedules = [s for s in schedules if s.status == status]
            if workflow_id:
                schedules = [s for s in schedules if s.workflow_id == workflow_id]
            if tags:
                schedules = [s for s in schedules if any(t in s.tags for t in tags)]
            
            return sorted(schedules, key=lambda s: s.priority, reverse=True)

    def pause(self, schedule_id: str) -> bool:
        with self._lock:
            schedule = self._schedules.get(schedule_id)
            if not schedule:
                return False
            
            schedule.status = ScheduleStatus.PAUSED
            self._mark_dirty()
            logger.info(f"Paused schedule: {schedule_id}")
            result = True
        self._flush_if_dirty()
        return result

    def resume(self, schedule_id: str) -> bool:
        with self._lock:
            schedule = self._schedules.get(schedule_id)
            if not schedule:
                return False
            
            schedule.status = ScheduleStatus.ACTIVE
            schedule.next_run_at = self._calculate_next_run(schedule)
            self._mark_dirty()
            result = True
            logger.info(f"Resumed schedule: {schedule_id}")
        self._flush_if_dirty()
        return result

    def disable(self, schedule_id: str) -> bool:
        with self._lock:
            schedule = self._schedules.get(schedule_id)
            if not schedule:
                return False
            
            schedule.status = ScheduleStatus.DISABLED
            self._mark_dirty()
            result = True
            logger.info(f"Disabled schedule: {schedule_id}")
        self._flush_if_dirty()
        return result

    def update(self, schedule_id: str, **kwargs) -> ScheduledWorkflow | None:
        with self._lock:
            schedule = self._schedules.get(schedule_id)
            if not schedule:
                return None
            
            if "cron" in kwargs:
                schedule.cron_expression = kwargs.pop("cron")
                CronExpression.parse(schedule.cron_expression)
            if "interval_seconds" in kwargs:
                schedule.interval_seconds = kwargs.pop("interval_seconds")
            if "params" in kwargs:
                schedule.params = {**schedule.params, **kwargs.pop("params")}
            
            for key, value in kwargs.items():
                if hasattr(schedule, key):
                    setattr(schedule, key, value)
            
            schedule.next_run_at = self._calculate_next_run(schedule)
            self._mark_dirty()
        self._flush_if_dirty()
        return schedule

    def delete(self, schedule_id: str) -> bool:
        with self._lock:
            if schedule_id not in self._schedules:
                return False
            del self._schedules[schedule_id]
            self._mark_dirty()
            logger.info(f"Deleted schedule: {schedule_id}")
            result = True
        self._flush_if_dirty()
        return result

    async def execute_now(self, schedule_id: str) -> ExecutionRecord:
        with self._lock:
            schedule = self._schedules.get(schedule_id)
        if not schedule:
            raise ValueError(f"Schedule not found: {schedule_id}")
        
        return await self._execute_schedule(schedule)

    async def _execute_schedule(self, schedule: ScheduledWorkflow) -> ExecutionRecord:
        execution_id = f"exec_{secrets.token_hex(8)}"
        started_at = datetime.now()
        
        record = ExecutionRecord(
            execution_id=execution_id,
            schedule_id=schedule.schedule_id,
            workflow_id=schedule.workflow_id,
            started_at=started_at,
        )
        
        with self._lock:
            self._history.append(record)
            if len(self._history) > self._max_history:
                del self._history[:len(self._history) - self._max_history]
            handler = self._workflow_handlers.get(schedule.workflow_id)
        
        try:
            if handler:
                import asyncio
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(schedule.params)
                else:
                    result = handler(schedule.params)
            else:
                from core.task_persistence import get_task_persistence
                persistence = get_task_persistence()
                task = persistence.create_task(
                    name=schedule.name,
                    task_type="scheduled_workflow",
                    payload=schedule.params,
                )
                result = {"task_id": task.task_id, "status": "queued"}
            
            record.status = "completed"
            record.result = result if isinstance(result, dict) else {"output": str(result)}
            schedule.last_run_status = "success"
            schedule.success_count += 1
            
            logger.info(f"Executed {schedule.name}: {schedule.schedule_id}")
            
        except Exception as e:
            record.status = "failed"
            record.error = str(e)
            schedule.last_run_status = "failed"
            schedule.failure_count += 1
            logger.error(f"Failed to execute {schedule.name}: {e}")
        
        finally:
            record.completed_at = datetime.now()
            record.duration_seconds = (record.completed_at - started_at).total_seconds()
            
            schedule.last_run_at = started_at
            schedule.run_count += 1
            schedule.next_run_at = self._calculate_next_run(schedule)
            
            self._save_state()
            
            with self._lock:
                callbacks = list(self._execution_callbacks)
            for callback in callbacks:
                try:
                    callback(record)
                except Exception as e:
                    logger.error(f"Execution callback error: {e}")
        
        return record

    def _tick(self) -> None:
        now = datetime.now()
        
        with self._lock:
            for schedule in list(self._schedules.values()):
                if schedule.status != ScheduleStatus.ACTIVE:
                    continue
                
                if schedule.next_run_at and schedule.next_run_at <= now:
                    import asyncio
                    try:
                        loop = asyncio.get_running_loop()
                        asyncio.run_coroutine_threadsafe(self._execute_schedule(schedule), loop)
                    except RuntimeError:
                        asyncio.run(self._execute_schedule(schedule))
        
        self._cleanup_history()

    def _cleanup_history(self) -> None:
        with self._lock:
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]

    def start(self) -> None:
        if self._running:
            return
        
        self._running = True
        self._stop_event.clear()
        
        def run():
            while not self._stop_event.is_set():
                try:
                    self._tick()
                except Exception as e:
                    logger.error(f"Scheduler tick error: {e}")
                self._stop_event.wait(self._tick_interval)
        
        self._thread = threading.Thread(target=run, daemon=True, name="WorkflowScheduler")
        self._thread.start()
        logger.info("Workflow scheduler started")

    def stop(self) -> None:
        if not self._running:
            return
        
        self._running = False
        self._stop_event.set()
        
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        
        self._save_state()
        logger.info("Workflow scheduler stopped")

    def get_pending_executions(self) -> list[tuple[ScheduledWorkflow, datetime]]:
        now = datetime.now()
        pending = []
        
        with self._lock:
            for schedule in self._schedules.values():
                if schedule.status == ScheduleStatus.ACTIVE and schedule.next_run_at:
                    if schedule.next_run_at <= now:
                        pending.append((schedule, schedule.next_run_at))
        
        return sorted(pending, key=lambda x: x[1])

    def get_history(
        self,
        schedule_id: str | None = None,
        workflow_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[ExecutionRecord]:
        with self._lock:
            records = list(self._history)
        
        if schedule_id:
            records = [r for r in records if r.schedule_id == schedule_id]
        if workflow_id:
            records = [r for r in records if r.workflow_id == workflow_id]
        if status:
            records = [r for r in records if r.status == status]
        
        return records[-limit:]

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            status_counts = {}
            for schedule in self._schedules.values():
                status_counts[schedule.status.value] = status_counts.get(schedule.status.value, 0) + 1
            
            return {
                "total_schedules": len(self._schedules),
                "total_templates": len(self._templates),
                "history_size": len(self._history),
                "by_status": status_counts,
                "is_running": self._running,
            }

    def register_execution_callback(self, callback: Callable) -> None:
        with self._lock:
            if len(self._execution_callbacks) >= self.MAX_EXECUTION_CALLBACKS:
                return
            self._execution_callbacks.append(callback)


_scheduler: WorkflowScheduler | None = None
_scheduler_lock = threading.Lock()


def get_workflow_scheduler() -> WorkflowScheduler:
    global _scheduler
    if _scheduler is None:
        with _scheduler_lock:
            if _scheduler is None:
                _scheduler = WorkflowScheduler()
    return _scheduler


def schedule_workflow(**kwargs) -> ScheduledWorkflow:
    return get_workflow_scheduler().schedule(**kwargs)


def schedule_from_template(template_name: str, **kwargs) -> ScheduledWorkflow:
    return get_workflow_scheduler().schedule_from_template(template_name, **kwargs)


def register_template(**kwargs) -> WorkflowTemplate:
    return get_workflow_scheduler().register_template(**kwargs)