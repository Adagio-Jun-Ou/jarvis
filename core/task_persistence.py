"""Task Persistence System for Long-Running Agentic Deployments.

Provides durable task storage with state recovery, enabling Jarvis to
resume tasks after restarts, crashes, or planned downtime.

Key features:
- Task queue persistence to disk
- State snapshots for running tasks
- Recovery on startup
- Task prioritization and scheduling
- Progress tracking and checkpointing
- Retry with exponential backoff
- Task dependencies and DAG execution
- Heartbeat monitoring for hung tasks

Example:
    persistence = get_task_persistence()
    
    # Create a persistent task
    task = persistence.create_task(
        name="daily_report",
        task_type="workflow",
        payload={"report_type": "daily"},
        schedule="0 9 * * *",  # Cron pattern
    )
    
    # Enqueue for execution
    persistence.enqueue(task.task_id)
    
    # Worker picks up task
    task = persistence.claim("worker_1")
    if task:
        # Do work...
        persistence.checkpoint(task.task_id, {"progress": 50})
        persistence.complete(task.task_id, result)
    
    # On restart, recover:
    persistence.recover_interrupted()
"""
from core.helpers import atomic_write_json, read_json_with_integrity
import logging
import secrets
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class TaskPriority(Enum):
    CRITICAL = 100
    HIGH = 75
    NORMAL = 50
    LOW = 25
    BACKGROUND = 10


@dataclass
class TaskCheckpoint:
    checkpoint_id: str
    task_id: str
    timestamp: datetime
    progress: float
    state: dict[str, Any]
    message: str = ""

    def to_dict(self) -> dict:
        return {
            "checkpoint_id": self.checkpoint_id,
            "task_id": self.task_id,
            "timestamp": self.timestamp.isoformat(),
            "progress": self.progress,
            "state": self.state,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TaskCheckpoint":
        return cls(
            checkpoint_id=data["checkpoint_id"],
            task_id=data["task_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            progress=data["progress"],
            state=data["state"],
            message=data.get("message", ""),
        )


@dataclass
class PersistentTask:
    task_id: str
    name: str
    task_type: str
    payload: dict[str, Any]
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.NORMAL
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    scheduled_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    
    worker_id: str | None = None
    parent_task_id: str | None = None
    dependencies: list[str] = field(default_factory=list)
    
    schedule: str | None = None
    max_retries: int = 3
    retry_count: int = 0
    retry_delay_seconds: float = 60.0
    timeout_seconds: float = 3600.0
    
    checkpoints: deque = field(default_factory=lambda: deque(maxlen=10))
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    heartbeat_at: datetime | None = None
    heartbeat_timeout_seconds: float = 300.0

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "name": self.name,
            "task_type": self.task_type,
            "payload": self.payload,
            "status": self.status.value,
            "priority": self.priority.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "scheduled_at": self.scheduled_at.isoformat() if self.scheduled_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "worker_id": self.worker_id,
            "parent_task_id": self.parent_task_id,
            "dependencies": self.dependencies,
            "schedule": self.schedule,
            "max_retries": self.max_retries,
            "retry_count": self.retry_count,
            "retry_delay_seconds": self.retry_delay_seconds,
            "timeout_seconds": self.timeout_seconds,
            "checkpoints": [c.to_dict() for c in self.checkpoints],
            "result": self.result,
            "error": self.error,
            "metadata": self.metadata,
            "heartbeat_at": self.heartbeat_at.isoformat() if self.heartbeat_at else None,
            "heartbeat_timeout_seconds": self.heartbeat_timeout_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PersistentTask":
        return cls(
            task_id=data["task_id"],
            name=data["name"],
            task_type=data["task_type"],
            payload=data.get("payload", {}),
            status=TaskStatus(data.get("status", "pending")),
            priority=TaskPriority(data.get("priority", 50)),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            scheduled_at=datetime.fromisoformat(data["scheduled_at"]) if data.get("scheduled_at") else None,
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            worker_id=data.get("worker_id"),
            parent_task_id=data.get("parent_task_id"),
            dependencies=data.get("dependencies", []),
            schedule=data.get("schedule"),
            max_retries=data.get("max_retries", 3),
            retry_count=data.get("retry_count", 0),
            retry_delay_seconds=data.get("retry_delay_seconds", 60.0),
            timeout_seconds=data.get("timeout_seconds", 3600.0),
            checkpoints=deque((TaskCheckpoint.from_dict(c) for c in data.get("checkpoints", [])), maxlen=10),
            result=data.get("result", {}),
            error=data.get("error"),
            metadata=data.get("metadata", {}),
            heartbeat_at=datetime.fromisoformat(data["heartbeat_at"]) if data.get("heartbeat_at") else None,
            heartbeat_timeout_seconds=data.get("heartbeat_timeout_seconds", 300.0),
        )

    def is_ready(self, completed_tasks: set[str]) -> bool:
        if self.status not in (TaskStatus.PENDING, TaskStatus.QUEUED):
            return False
        return all(dep in completed_tasks for dep in self.dependencies)

    def can_retry(self) -> bool:
        return self.retry_count < self.max_retries

    def is_timed_out(self) -> bool:
        if self.status != TaskStatus.RUNNING or not self.started_at:
            return False
        elapsed = (datetime.now() - self.started_at).total_seconds()
        return elapsed > self.timeout_seconds

    def is_heartbeat_timeout(self) -> bool:
        if self.status != TaskStatus.RUNNING or not self.heartbeat_at:
            return False
        elapsed = (datetime.now() - self.heartbeat_at).total_seconds()
        return elapsed > self.heartbeat_timeout_seconds

    def get_progress(self) -> float:
        if not self.checkpoints:
            return 0.0
        return self.checkpoints[-1].progress


@dataclass
class TaskResult:
    task_id: str
    success: bool
    result: dict[str, Any]
    error: str | None = None
    duration_seconds: float = 0.0
    retries: int = 0

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "duration_seconds": self.duration_seconds,
            "retries": self.retries,
        }


class TaskPersistence:
    def __init__(self, storage_path: Path | None = None):
        self.storage_path = storage_path or Path.home() / ".jarvis" / "tasks"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self._tasks_file = self.storage_path / "tasks.json"
        self._queue_file = self.storage_path / "queue.json"
        self._lock = threading.RLock()
        self._file_lock = threading.Lock()
        
        self._tasks: dict[str, PersistentTask] = {}
        self._queue: list[str] = []
        self._completed_ids: set[str] = set()
        self._handlers: dict[str, Callable] = {}
        self._running_tasks: dict[str, str] = {}
        
        self._max_tasks = 10000
        self._max_checkpoints = 10
        self._cleanup_days = 30
        
        self._load_state()

    def _load_state(self) -> None:
        try:
            if self._tasks_file.exists():
                data = read_json_with_integrity(self._tasks_file)
                if data is not None:
                    for task_data in data.get("tasks", []):
                        task = PersistentTask.from_dict(task_data)
                        self._tasks[task.task_id] = task
                        if task.status == TaskStatus.COMPLETED:
                            self._completed_ids.add(task.task_id)
                    logger.info(f"Loaded {len(self._tasks)} tasks from persistence")

            if self._queue_file.exists():
                queue_data = read_json_with_integrity(self._queue_file)
                if isinstance(queue_data, list):
                    self._queue = queue_data
                elif queue_data is not None:
                    logger.warning(f"queue.json was not a list, resetting: {type(queue_data)}")
                    
        except Exception as e:
            logger.error(f"Failed to load task state: {e}")

    def _snapshot_tasks(self) -> dict:
        return {
            "tasks": [t.to_dict() for t in self._tasks.values()],
            "updated_at": datetime.now().isoformat(),
        }

    def _snapshot_queue(self) -> list:
        return list(self._queue)

    def _persist_state(self, tasks_data: dict, queue_data: list) -> None:
        try:
            with self._file_lock:
                atomic_write_json(self._tasks_file, tasks_data)
                atomic_write_json(self._queue_file, queue_data)
        except Exception as e:
            logger.error(f"Failed to save task state: {e}")

    def _save_state(self) -> None:
        with self._lock:
            tasks_data = self._snapshot_tasks()
            queue_data = self._snapshot_queue()
        self._persist_state(tasks_data, queue_data)

    def create_task(
        self,
        name: str,
        task_type: str,
        payload: dict[str, Any],
        priority: TaskPriority | int = TaskPriority.NORMAL,
        schedule: str | None = None,
        dependencies: list[str] | None = None,
        max_retries: int = 3,
        timeout_seconds: float = 3600.0,
        parent_task_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PersistentTask:
        with self._lock:
            task_id = f"task_{secrets.token_hex(8)}"
            
            if isinstance(priority, int):
                priority = TaskPriority(priority) if priority in [p.value for p in TaskPriority] else TaskPriority.NORMAL
            
            task = PersistentTask(
                task_id=task_id,
                name=name,
                task_type=task_type,
                payload=payload,
                priority=priority,
                schedule=schedule,
                dependencies=dependencies or [],
                max_retries=max_retries,
                timeout_seconds=timeout_seconds,
                parent_task_id=parent_task_id,
                metadata=metadata or {},
            )
            
            self._tasks[task_id] = task
            
            tasks_data = self._snapshot_tasks()
            queue_data = self._snapshot_queue()
            
        self._persist_state(tasks_data, queue_data)
            
        logger.info(f"Created task {task_id}: {name}")
        return task

    def enqueue(self, task_id: str, scheduled_at: datetime | None = None) -> bool:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return False
            
            if task.status not in (TaskStatus.PENDING, TaskStatus.FAILED):
                return False
            
            task.status = TaskStatus.QUEUED
            task.scheduled_at = scheduled_at or datetime.now()
            task.updated_at = datetime.now()
            
            if task_id not in self._queue:
                self._queue.append(task_id)
                self._queue.sort(key=lambda tid: -self._tasks[tid].priority.value)
            
            tasks_data = self._snapshot_tasks()
            queue_data = self._snapshot_queue()
            
        self._persist_state(tasks_data, queue_data)
        return True

    def claim(self, worker_id: str, task_types: list[str] | None = None) -> PersistentTask | None:
        with self._lock:
            now = datetime.now()
            
            for i, task_id in enumerate(self._queue):
                task = self._tasks.get(task_id)
                if not task:
                    continue
                
                if task.status != TaskStatus.QUEUED:
                    continue
                
                if task.scheduled_at and task.scheduled_at > now:
                    continue
                
                if not task.is_ready(self._completed_ids):
                    continue
                
                if task_types and task.task_type not in task_types:
                    continue
                
                task.status = TaskStatus.RUNNING
                task.worker_id = worker_id
                task.started_at = now
                task.heartbeat_at = now
                task.updated_at = now
                
                self._queue.pop(i)
                self._running_tasks[task_id] = worker_id
                tasks_data = self._snapshot_tasks()
                queue_data = self._snapshot_queue()
                claimed_task = task
                break
            else:
                tasks_data = None
                queue_data = None
                claimed_task = None
        
        if tasks_data is not None:
            self._persist_state(tasks_data, queue_data)
                
        if claimed_task:
            logger.info(f"Task {claimed_task.task_id} claimed by {worker_id}")
        return claimed_task

    def checkpoint(
        self,
        task_id: str,
        state: dict[str, Any],
        progress: float | None = None,
        message: str = "",
    ) -> bool:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task or task.status != TaskStatus.RUNNING:
                return False
            
            if progress is None:
                progress = task.get_progress()
            
            checkpoint = TaskCheckpoint(
                checkpoint_id=f"ckpt_{secrets.token_hex(4)}",
                task_id=task_id,
                timestamp=datetime.now(),
                progress=min(100.0, max(0.0, progress)),
                state=state,
                message=message,
            )
            
            task.checkpoints.append(checkpoint)
            
            task.heartbeat_at = datetime.now()
            task.updated_at = datetime.now()
            tasks_data = self._snapshot_tasks()
            queue_data = self._snapshot_queue()
            
        self._persist_state(tasks_data, queue_data)
        return True

    def heartbeat(self, task_id: str) -> bool:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return False
            
            task.heartbeat_at = datetime.now()
            task.updated_at = datetime.now()
            return True

    def complete(self, task_id: str, result: dict[str, Any]) -> bool:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return False
            
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.updated_at = datetime.now()
            task.result = result
            
            self._completed_ids.add(task_id)
            self._running_tasks.pop(task_id, None)
            tasks_data = self._snapshot_tasks()
            queue_data = self._snapshot_queue()
            
        self._persist_state(tasks_data, queue_data)
            
        logger.info(f"Task {task_id} completed")
        return True

    def fail(self, task_id: str, error: str, retry: bool = True) -> bool:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return False
            
            task.error = error
            task.updated_at = datetime.now()
            self._running_tasks.pop(task_id, None)
            
            if retry and task.can_retry():
                task.retry_count += 1
                task.status = TaskStatus.PENDING
                delay = task.retry_delay_seconds * (2 ** (task.retry_count - 1))
                task.scheduled_at = datetime.now() + timedelta(seconds=delay)
                logger.warning(f"Task {task_id} failed, retry {task.retry_count}/{task.max_retries} in {delay}s")
            else:
                task.status = TaskStatus.FAILED
                task.completed_at = datetime.now()
                logger.error(f"Task {task_id} failed permanently: {error}")
            
            tasks_data = self._snapshot_tasks()
            queue_data = self._snapshot_queue()
            
        self._persist_state(tasks_data, queue_data)
        return True

    def cancel(self, task_id: str) -> bool:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return False
            
            if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                return False
            
            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.now()
            task.updated_at = datetime.now()
            
            if task_id in self._queue:
                self._queue.remove(task_id)
            
            self._running_tasks.pop(task_id, None)
            tasks_data = self._snapshot_tasks()
            queue_data = self._snapshot_queue()
            
        self._persist_state(tasks_data, queue_data)
            
        logger.info(f"Task {task_id} cancelled")
        return True

    def pause(self, task_id: str) -> bool:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task or task.status != TaskStatus.RUNNING:
                return False
            
            task.status = TaskStatus.PAUSED
            task.updated_at = datetime.now()
            self._running_tasks.pop(task_id, None)
            tasks_data = self._snapshot_tasks()
            queue_data = self._snapshot_queue()
            
        self._persist_state(tasks_data, queue_data)
        return True

    def resume(self, task_id: str) -> bool:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task or task.status != TaskStatus.PAUSED:
                return False
            
            task.status = TaskStatus.QUEUED
            task.updated_at = datetime.now()
            
            if task_id not in self._queue:
                self._queue.append(task_id)
                self._queue.sort(key=lambda tid: -self._tasks[tid].priority.value)
            
            tasks_data = self._snapshot_tasks()
            queue_data = self._snapshot_queue()
            
        self._persist_state(tasks_data, queue_data)
        return True

    def recover_interrupted(self) -> list[PersistentTask]:
        with self._lock:
            recovered = []
            
            for task_id, worker_id in list(self._running_tasks.items()):
                task = self._tasks.get(task_id)
                if not task:
                    continue
                
                if task.is_timed_out() or task.is_heartbeat_timeout():
                    if task.can_retry():
                        task.status = TaskStatus.PENDING
                        task.retry_count += 1
                        task.scheduled_at = datetime.now()
                        if task_id not in self._queue:
                            self._queue.append(task_id)
                        recovered.append(task)
                        logger.info(f"Recovered timed-out task {task_id} for retry")
                    else:
                        task.status = TaskStatus.TIMEOUT
                        task.error = "Task timed out during execution"
                        task.completed_at = datetime.now()
                        logger.warning(f"Task {task_id} timed out permanently")
                else:
                    task.status = TaskStatus.QUEUED
                    task.worker_id = None
                    if task_id not in self._queue:
                        self._queue.append(task_id)
                    recovered.append(task)
                    logger.info(f"Recovered running task {task_id} to queue")
            
            self._running_tasks.clear()
            self._queue.sort(key=lambda tid: -self._tasks[tid].priority.value)
            tasks_data = self._snapshot_tasks()
            queue_data = self._snapshot_queue()
        
        self._persist_state(tasks_data, queue_data)

        return recovered

    def get_task(self, task_id: str) -> PersistentTask | None:
        return self._tasks.get(task_id)

    def get_latest_checkpoint(self, task_id: str) -> TaskCheckpoint | None:
        task = self._tasks.get(task_id)
        if not task or not task.checkpoints:
            return None
        return task.checkpoints[-1]

    def get_queue_stats(self) -> dict[str, Any]:
        with self._lock:
            status_counts = {}
            priority_counts = {}
            
            for task in self._tasks.values():
                status_counts[task.status.value] = status_counts.get(task.status.value, 0) + 1
                priority_counts[task.priority.name] = priority_counts.get(task.priority.name, 0) + 1
            
            return {
                "total_tasks": len(self._tasks),
                "queue_size": len(self._queue),
                "running": len(self._running_tasks),
                "completed": len(self._completed_ids),
                "by_status": status_counts,
                "by_priority": priority_counts,
            }

    def get_pending_tasks(self, limit: int = 100) -> list[PersistentTask]:
        with self._lock:
            return [
                self._tasks[tid] for tid in self._queue[:limit]
                if tid in self._tasks
            ]

    def get_running_tasks(self) -> list[PersistentTask]:
        with self._lock:
            return [
                self._tasks[tid] for tid in self._running_tasks
                if tid in self._tasks
            ]

    def get_scheduled_tasks(self) -> list[PersistentTask]:
        with self._lock:
            datetime.now()
            return [
                task for task in self._tasks.values()
                if task.schedule and task.status == TaskStatus.PENDING
            ]

    def register_handler(self, task_type: str, handler: Callable) -> None:
        self._handlers[task_type] = handler

    def cleanup_old_tasks(self, days: int = 30) -> int:
        with self._lock:
            cutoff = datetime.now() - timedelta(days=days)
            to_remove = []
            
            for task_id, task in self._tasks.items():
                if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.TIMEOUT):
                    if task.completed_at and task.completed_at < cutoff:
                        to_remove.append(task_id)
            
            for task_id in to_remove:
                del self._tasks[task_id]
                self._completed_ids.discard(task_id)
            
            tasks_data = self._snapshot_tasks()
            queue_data = self._snapshot_queue()
        
        self._persist_state(tasks_data, queue_data)
        logger.info(f"Cleaned up {len(to_remove)} old tasks")
        return len(to_remove)


_task_persistence: TaskPersistence | None = None
_persistence_lock = threading.Lock()


def get_task_persistence() -> TaskPersistence:
    global _task_persistence
    if _task_persistence is None:
        with _persistence_lock:
            if _task_persistence is None:
                _task_persistence = TaskPersistence()
    return _task_persistence


def task_create(**kwargs) -> PersistentTask:
    return get_task_persistence().create_task(**kwargs)


def task_enqueue(task_id: str, **kwargs) -> bool:
    return get_task_persistence().enqueue(task_id, **kwargs)


def task_claim(worker_id: str, **kwargs) -> PersistentTask | None:
    return get_task_persistence().claim(worker_id, **kwargs)


def task_checkpoint(task_id: str, state: dict, **kwargs) -> bool:
    return get_task_persistence().checkpoint(task_id, state, **kwargs)


def task_complete(task_id: str, result: dict) -> bool:
    return get_task_persistence().complete(task_id, result)


def task_fail(task_id: str, error: str, **kwargs) -> bool:
    return get_task_persistence().fail(task_id, error, **kwargs)


def task_get(task_id: str) -> PersistentTask | None:
    return get_task_persistence().get_task(task_id)