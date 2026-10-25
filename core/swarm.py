"""Swarm Coordinator - Multi-agent team coordination with task distribution.

Enables multiple Jarvis agents to work together as a coordinated swarm,
distributing tasks, sharing findings, and achieving consensus on results.
"""
import hashlib
import logging
import threading
import time
from collections import defaultdict, deque
from core.helpers import atomic_write_json, read_json_with_integrity
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    COORDINATOR = "coordinator"
    WORKER = "worker"
    SPECIALIST = "specialist"
    VALIDATOR = "validator"
    OBSERVER = "observer"


class AgentStatus(Enum):
    IDLE = "idle"
    WORKING = "working"
    WAITING = "waiting"
    OFFLINE = "offline"


class TaskStatus(Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class AgentProfile:
    agent_id: str
    role: AgentRole = AgentRole.WORKER
    status: AgentStatus = AgentStatus.IDLE
    capabilities: list[str] = field(default_factory=list)
    current_task: str | None = None
    completed_tasks: int = 0
    reliability_score: float = 1.0
    last_heartbeat: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "role": self.role.value,
            "status": self.status.value,
            "capabilities": self.capabilities,
            "current_task": self.current_task,
            "completed_tasks": self.completed_tasks,
            "reliability": round(self.reliability_score, 3),
        }


@dataclass
class SwarmTask:
    task_id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.NORMAL
    assigned_to: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    result: Any = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "description": self.description,
            "status": self.status.value,
            "priority": self.priority.value,
            "assigned_to": self.assigned_to,
            "dependencies": self.dependencies,
            "created": self.created_at.isoformat(),
            "started": self.started_at.isoformat() if self.started_at else None,
            "completed": self.completed_at.isoformat() if self.completed_at else None,
        }


@dataclass
class SwarmMessage:
    message_id: str
    sender_id: str
    recipient_id: str | None
    message_type: str
    payload: dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "id": self.message_id,
            "sender": self.sender_id,
            "recipient": self.recipient_id,
            "type": self.message_type,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
        }


class TaskDistributor:
    """Distributes tasks among agents based on capabilities and load."""

    def __init__(self):
        self._capability_map: dict[str, set[str]] = {}  # Not defaultdict, thread-safe
        self._load_map: dict[str, int] = {}  # Not defaultdict, thread-safe
        self._lock = threading.RLock()

    def register_capabilities(self, agent_id: str, capabilities: list[str]) -> None:
        with self._lock:
            self._capability_map[agent_id] = set(capabilities)

    def update_load(self, agent_id: str, current_tasks: int) -> None:
        with self._lock:
            self._load_map[agent_id] = current_tasks

    def find_best_agent(
        self,
        required_capabilities: list[str] | None = None,
        exclude_agents: list[str] | None = None,
        agents: dict[str, AgentProfile] | None = None,
    ) -> str | None:
        with self._lock:
            exclude = set(exclude_agents or [])
            required = set(required_capabilities or [])

            candidates = []

            for agent_id, caps in self._capability_map.items():
                if agent_id in exclude:
                    continue

                if required and not required.issubset(caps):
                    continue

                load = self._load_map.get(agent_id, 0)
                candidates.append((agent_id, load))

            if not candidates:
                return None

            candidates.sort(key=lambda x: x[1])
            return candidates[0][0]

    def find_agents_for_task(
        self,
        required_capabilities: list[str] | None = None,
        min_agents: int = 1,
        exclude_agents: list[str] | None = None,
        agents: dict[str, AgentProfile] | None = None,
    ) -> list[str]:
        with self._lock:
            exclude = set(exclude_agents or [])
            required = set(required_capabilities or [])

            candidates = []

            for agent_id, caps in self._capability_map.items():
                if agent_id in exclude:
                    continue

                if required and not required.issubset(caps):
                    continue

                load = self._load_map.get(agent_id, 0)
                candidates.append((agent_id, load))

            candidates.sort(key=lambda x: x[1])
            return [c[0] for c in candidates[:min_agents]]

    def get_load_statistics(self) -> dict[str, int]:
        with self._lock:
            return dict(self._load_map)


class ConsensusEngine:
    """Achieves consensus among swarm members for distributed decisions."""

    CONSENSUS_THRESHOLD = 0.6
    MINIMUM_VOTES = 2
    VOTE_TIMEOUT_SECONDS = 30
    MAX_PROPOSALS = 200

    def __init__(self):
        self._proposals: dict[str, dict] = {}
        self._votes: dict[str, dict[str, bool]] = {}
        self._lock = threading.RLock()

    def create_proposal(
        self,
        proposal_type: str,
        data: dict[str, Any],
        quorum: int = 3,
    ) -> str:
        with self._lock:
            # Auto-cleanup every 10 proposals or if over max
            if len(self._proposals) > self.MAX_PROPOSALS:
                self._cleanup_expired()
            if len(self._proposals) % 10 == 0 and len(self._proposals) > 0:
                self._cleanup_expired()
            
            proposal_id = hashlib.sha256(
                f"{proposal_type}:{data}:{time.time()}".encode()
            ).hexdigest()[:16]

            self._proposals[proposal_id] = {
                "type": proposal_type,
                "data": data,
                "quorum": quorum,
                "created_at": datetime.now(),
                "decision": None,
            }

            return proposal_id

    def vote(self, proposal_id: str, agent_id: str, approve: bool) -> bool:
        with self._lock:
            if proposal_id not in self._proposals:
                return False

            if proposal_id not in self._votes:
                self._votes[proposal_id] = {}
            
            self._votes[proposal_id][agent_id] = approve

            proposal = self._proposals[proposal_id]
            votes = self._votes[proposal_id]

            if len(votes) >= proposal["quorum"]:
                approvals = sum(1 for v in votes.values() if v)
                ratio = approvals / len(votes)

                if ratio >= self.CONSENSUS_THRESHOLD:
                    proposal["decision"] = "approved"
                else:
                    proposal["decision"] = "rejected"

            return True
    
    def _cleanup_expired(self) -> None:
        """Remove expired proposals and their votes."""
        cutoff = datetime.now() - timedelta(seconds=self.VOTE_TIMEOUT_SECONDS)
        to_remove = [
            pid for pid, pdata in self._proposals.items()
            if pdata["created_at"] < cutoff and pdata["decision"] is None
        ]
        for pid in to_remove:
            self._proposals.pop(pid, None)
            self._votes.pop(pid, None)
        
        if len(self._proposals) > self.MAX_PROPOSALS:
            sorted_proposals = sorted(
                self._proposals.items(),
                key=lambda x: x[1]["created_at"],
            )
            excess = len(self._proposals) - self.MAX_PROPOSALS
            for pid, _ in sorted_proposals[:excess]:
                self._proposals.pop(pid, None)
                self._votes.pop(pid, None)

    def get_decision(self, proposal_id: str) -> str | None:
        with self._lock:
            if proposal_id not in self._proposals:
                return None
            return self._proposals[proposal_id]["decision"]

    def get_proposal(self, proposal_id: str) -> dict | None:
        with self._lock:
            if proposal_id not in self._proposals:
                return None

            proposal = self._proposals[proposal_id].copy()
            proposal["votes"] = dict(self._votes.get(proposal_id, {}))
            return proposal

    def cleanup_expired(self, max_age_minutes: int = 30) -> int:
        with self._lock:
            cutoff = datetime.now() - timedelta(minutes=max_age_minutes)
            expired = [
                pid for pid, p in self._proposals.items()
                if p["created_at"] < cutoff
            ]
            for pid in expired:
                del self._proposals[pid]
                if pid in self._votes:
                    del self._votes[pid]
            return len(expired)


class SwarmCoordinator:
    """Coordinates a swarm of Jarvis agents for collaborative work."""

    MAX_AGENTS = 20
    MAX_TASKS = 1000
    MAX_MESSAGES = 5000
    HEARTBEAT_TIMEOUT_SECONDS = 120
    MAX_PENDING_TASKS = 100
    MAX_QUEUE = 1000

    def __init__(
        self,
        storage_path: Path | None = None,
        agent_id: str = "jarvis_main",
        role: AgentRole = AgentRole.COORDINATOR,
    ):
        self.storage_path = storage_path or Path.home() / ".jarvis" / "swarm"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.agent_id = agent_id
        self.role = role

        self._agents: dict[str, AgentProfile] = {}
        self._tasks: dict[str, SwarmTask] = {}
        self._task_queue: list[str] = []
        self._messages: deque[SwarmMessage] = deque(maxlen=self.MAX_MESSAGES)

        self._task_counter = 0
        self._message_counter = 0
        self._lock = threading.RLock()

        self._distributor = TaskDistributor()
        self._consensus = ConsensusEngine()

        self._handlers: dict[str, Callable] = {}

        self._load_state()

        self._register_self()

    def _load_state(self) -> None:
        state_file = self.storage_path / "swarm_state.json"
        if state_file.exists():
            try:
                data = read_json_with_integrity(state_file)
                self._task_counter = data.get("task_counter", 0)
                self._message_counter = data.get("message_counter", 0)
                logger.info(f"Loaded swarm state: tasks={self._task_counter}, messages={self._message_counter}")
            except Exception as e:
                logger.warning(f"Failed to load swarm state: {e}")

    def _save_state(self) -> None:
        state_file = self.storage_path / "swarm_state.json"
        try:
            atomic_write_json(state_file, {
                    "task_counter": self._task_counter,
                    "message_counter": self._message_counter,
                    "agents": len(self._agents),
                    "tasks": len(self._tasks),
                    "last_updated": datetime.now().isoformat(),
                })
        except Exception as e:
            logger.warning(f"Failed to save swarm state: {e}")

    def _register_self(self) -> None:
        with self._lock:
            self._agents[self.agent_id] = AgentProfile(
                agent_id=self.agent_id,
                role=self.role,
                status=AgentStatus.IDLE,
                capabilities=["coordinator", "task_distribution", "consensus"],
            )
            self._distributor.register_capabilities(
                self.agent_id,
                ["coordinator", "task_distribution", "consensus"]
            )

    def register_agent(
        self,
        agent_id: str,
        role: AgentRole = AgentRole.WORKER,
        capabilities: list[str] | None = None,
    ) -> bool:
        with self._lock:
            if len(self._agents) >= self.MAX_AGENTS:
                return False

            profile = AgentProfile(
                agent_id=agent_id,
                role=role,
                status=AgentStatus.IDLE,
                capabilities=capabilities or [],
            )

            self._agents[agent_id] = profile
            self._distributor.register_capabilities(agent_id, capabilities or [])

            self._broadcast_message(
                "agent_joined",
                {"agent_id": agent_id, "role": role.value, "capabilities": capabilities}
            )

            logger.info(f"Agent {agent_id} joined swarm as {role.value}")
            return True

    def unregister_agent(self, agent_id: str) -> bool:
        with self._lock:
            if agent_id not in self._agents:
                return False

            profile = self._agents[agent_id]

            if profile.current_task:
                task = self._tasks.get(profile.current_task)
                if task:
                    task.status = TaskStatus.PENDING
                    if agent_id in task.assigned_to:
                        task.assigned_to.remove(agent_id)

            del self._agents[agent_id]

            self._broadcast_message(
                "agent_left",
                {"agent_id": agent_id}
            )

            logger.info(f"Agent {agent_id} left swarm")
            return True

    def heartbeat(self, agent_id: str, status: AgentStatus = AgentStatus.IDLE) -> bool:
        with self._lock:
            if agent_id not in self._agents:
                return False

            self._agents[agent_id].last_heartbeat = datetime.now()
            self._agents[agent_id].status = status

            return True

    def create_task(
        self,
        description: str,
        priority: TaskPriority = TaskPriority.NORMAL,
        dependencies: list[str] | None = None,
        required_capabilities: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        with self._lock:
            if len(self._tasks) >= self.MAX_TASKS:
                self._cleanup_completed_tasks()

            self._task_counter += 1
            task_id = f"task_{self._task_counter:06d}"

            task = SwarmTask(
                task_id=task_id,
                description=description,
                priority=priority,
                dependencies=dependencies or [],
                metadata={
                    **(metadata or {}),
                    "required_capabilities": required_capabilities or [],
                },
            )

            self._tasks[task_id] = task

            self._insert_task_by_priority(task_id)

            self._save_state()

            logger.info(f"Created task {task_id}: {description[:50]}")

            return task_id

    def _insert_task_by_priority(self, task_id: str) -> None:
        task = self._tasks.get(task_id)
        if not task:
            return

        inserted = False
        for i, queued_id in enumerate(self._task_queue):
            queued_task = self._tasks.get(queued_id)
            if queued_task and task.priority.value > queued_task.priority.value:
                self._task_queue.insert(i, task_id)
                inserted = True
                break

        if not inserted:
            self._task_queue.append(task_id)
        if len(self._task_queue) > self.MAX_QUEUE:
            self._task_queue = self._task_queue[-self.MAX_QUEUE:]

    def get_next_task(self, agent_id: str) -> SwarmTask | None:
        with self._lock:
            if agent_id not in self._agents:
                return None

            profile = self._agents[agent_id]

            for task_id in self._task_queue:
                task = self._tasks.get(task_id)
                if not task:
                    continue

                if task.status != TaskStatus.PENDING:
                    continue

                if not self._dependencies_met(task):
                    continue

                required = task.metadata.get("required_capabilities", [])
                if required and not set(required).issubset(set(profile.capabilities)):
                    continue

                task.status = TaskStatus.ASSIGNED
                task.assigned_to.append(agent_id)
                task.started_at = datetime.now()

                profile.current_task = task_id
                profile.status = AgentStatus.WORKING

                self._task_queue.remove(task_id)
                task_count = len([t for t in self._tasks.values() if t.assigned_to and agent_id in t.assigned_to and t.status in (TaskStatus.PENDING, TaskStatus.ASSIGNED)])
                self._distributor.update_load(agent_id, task_count)

                self._broadcast_message(
                    "task_assigned",
                    {"task_id": task_id, "agent_id": agent_id}
                )

                return task

            return None

    def _dependencies_met(self, task: SwarmTask) -> bool:
        for dep_id in task.dependencies:
            dep = self._tasks.get(dep_id)
            if not dep or dep.status != TaskStatus.COMPLETED:
                return False
        return True

    def complete_task(
        self,
        task_id: str,
        agent_id: str,
        result: Any = None,
        success: bool = True,
    ) -> bool:
        with self._lock:
            if task_id not in self._tasks:
                return False

            task = self._tasks[task_id]

            if not task.assigned_to:
                return False
            
            if agent_id not in task.assigned_to:
                return False

            task.status = TaskStatus.COMPLETED if success else TaskStatus.FAILED
            task.result = result
            task.completed_at = datetime.now()

            if agent_id in self._agents:
                profile = self._agents[agent_id]
                profile.current_task = None
                profile.status = AgentStatus.IDLE
                profile.completed_tasks += 1

                if success:
                    profile.reliability_score = min(1.0, profile.reliability_score + 0.01)
                else:
                    profile.reliability_score = max(0.1, profile.reliability_score - 0.05)

            self._broadcast_message(
                "task_completed",
                {"task_id": task_id, "agent_id": agent_id, "success": success}
            )

            self._save_state()

            logger.info(f"Task {task_id} completed by {agent_id}: success={success}")

            return True

    def cancel_task(self, task_id: str) -> bool:
        with self._lock:
            if task_id not in self._tasks:
                return False

            task = self._tasks[task_id]
            task.status = TaskStatus.CANCELLED

            for agent_id in task.assigned_to:
                if agent_id in self._agents:
                    self._agents[agent_id].current_task = None
                    self._agents[agent_id].status = AgentStatus.IDLE

            if task_id in self._task_queue:
                self._task_queue.remove(task_id)

            self._broadcast_message(
                "task_cancelled",
                {"task_id": task_id}
            )

            return True

    def propose_consensus(self, proposal_type: str, data: dict[str, Any]) -> str:
        return self._consensus.create_proposal(proposal_type, data)

    def vote(self, proposal_id: str, approve: bool) -> bool:
        return self._consensus.vote(proposal_id, self.agent_id, approve)

    def vote_on_behalf(self, proposal_id: str, agent_id: str, approve: bool) -> bool:
        return self._consensus.vote(proposal_id, agent_id, approve)

    def get_consensus_decision(self, proposal_id: str) -> str | None:
        return self._consensus.get_decision(proposal_id)

    def send_message(
        self,
        recipient_id: str | None,
        message_type: str,
        payload: dict[str, Any],
    ) -> str:
        with self._lock:
            self._message_counter += 1
            message_id = f"msg_{self._message_counter:06d}"

            message = SwarmMessage(
                message_id=message_id,
                sender_id=self.agent_id,
                recipient_id=recipient_id,
                message_type=message_type,
                payload=payload,
            )

            self._messages.append(message)

            return message_id

    def _broadcast_message(self, message_type: str, payload: dict[str, Any]) -> str:
        return self.send_message(None, message_type, payload)

    def get_messages(
        self,
        agent_id: str | None = None,
        since: datetime | None = None,
    ) -> list[SwarmMessage]:
        with self._lock:
            messages = []
            for msg in self._messages:
                if agent_id and msg.recipient_id and msg.recipient_id != agent_id:
                    continue
                if since and msg.timestamp < since:
                    continue
                messages.append(msg)
            return messages

    def register_handler(self, message_type: str, handler: Callable) -> None:
        with self._lock:
            self._handlers[message_type] = handler

    def process_messages(self) -> int:
        with self._lock:
            to_process = list(self._messages)
            self._messages.clear()
        processed = 0
        for msg in to_process:
            handler = self._handlers.get(msg.message_type)
            if handler:
                try:
                    handler(msg)
                    processed += 1
                except Exception as e:
                    logger.error(f"Handler error for {msg.message_type}: {e}")
        return processed

    def cleanup_stale_agents(self, timeout_seconds: int | None = None) -> int:
        with self._lock:
            timeout = timeout_seconds or self.HEARTBEAT_TIMEOUT_SECONDS
            cutoff = datetime.now() - timedelta(seconds=timeout)

            stale = [
                aid for aid, profile in self._agents.items()
                if profile.last_heartbeat < cutoff and aid != self.agent_id
            ]

            for agent_id in stale:
                self.unregister_agent(agent_id)

            return len(stale)

    def _cleanup_completed_tasks(self, keep_recent: int = 100) -> int:
        with self._lock:
            completed = [
                tid for tid, task in self._tasks.items()
                if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)
            ]

            completed.sort(key=lambda tid: self._tasks[tid].completed_at or datetime.min, reverse=True)

            to_remove = completed[keep_recent:]
            for tid in to_remove:
                del self._tasks[tid]

            # Also remove stale task IDs from _task_queue
            valid_ids = set(self._tasks.keys())
            self._task_queue = [tid for tid in self._task_queue if tid in valid_ids]

            return len(to_remove)

    def get_swarm_status(self) -> dict[str, Any]:
        with self._lock:
            agents_by_status: dict[str, int] = defaultdict(int)
            for profile in self._agents.values():
                agents_by_status[profile.status.value] += 1

            tasks_by_status: dict[str, int] = defaultdict(int)
            for task in self._tasks.values():
                tasks_by_status[task.status.value] += 1

            return {
                "agent_id": self.agent_id,
                "role": self.role.value,
                "total_agents": len(self._agents),
                "agents_by_status": dict(agents_by_status),
                "total_tasks": len(self._tasks),
                "tasks_by_status": dict(tasks_by_status),
                "pending_tasks": len(self._task_queue),
                "messages": len(self._messages),
            }

    def get_agent_status(self, agent_id: str | None = None) -> dict | None:
        with self._lock:
            aid = agent_id or self.agent_id
            profile = self._agents.get(aid)
            return profile.to_dict() if profile else None

    def get_task_status(self, task_id: str) -> dict | None:
        with self._lock:
            task = self._tasks.get(task_id)
            return task.to_dict() if task else None


_swarm_coordinator: SwarmCoordinator | None = None
_swarm_lock = threading.Lock()


def get_swarm_coordinator() -> SwarmCoordinator:
    global _swarm_coordinator
    if _swarm_coordinator is None:
        with _swarm_lock:
            if _swarm_coordinator is None:
                _swarm_coordinator = SwarmCoordinator()
    return _swarm_coordinator
