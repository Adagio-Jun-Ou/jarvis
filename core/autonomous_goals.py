"""Autonomous Goals System for Long-Running Agentic Deployments.

Provides goal management, progress tracking, and proactive behavior for
autonomous agent operations over extended periods.

Key features:
- Long-term goal tracking with milestones
- Goal decomposition and progress metrics
- Proactive behavior triggers
- Goal priority management
- Goal dependencies and blocking
- Periodic review and adjustment
- Goal achievement recognition

Example:
    goals = get_autonomous_goals()
    
    # Create a long-term goal
    goal = goals.create(
        name="Improve code quality",
        description="Reduce technical debt and improve test coverage",
        target_date=datetime.now() + timedelta(days=30),
        milestones=[
            {"name": "Add tests for core modules", "target_percent": 25},
            {"name": "Refactor legacy code", "target_percent": 50},
            {"name": "Improve documentation", "target_percent": 75},
            {"name": "Final review", "target_percent": 100},
        ],
    )
    
    # Update progress
    goals.update_progress(goal.goal_id, 35, "Completed test coverage for agent.py")
    
    # Get actionable goals
    actionable = goals.get_actionable()
"""
import logging
from core.helpers import atomic_write_json, read_json_with_integrity
import secrets
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class GoalStatus(Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class GoalPriority(Enum):
    CRITICAL = 100
    HIGH = 75
    MEDIUM = 50
    LOW = 25
    BACKGROUND = 10


class TriggerType(Enum):
    TIME_BASED = "time_based"
    PROGRESS_BASED = "progress_based"
    DEPENDENCY_BASED = "dependency_based"
    CONDITION_BASED = "condition_based"
    EVENT_BASED = "event_based"


@dataclass
class Milestone:
    milestone_id: str
    name: str
    description: str = ""
    target_percent: float = 0.0
    achieved: bool = False
    achieved_at: datetime | None = None
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "milestone_id": self.milestone_id,
            "name": self.name,
            "description": self.description,
            "target_percent": self.target_percent,
            "achieved": self.achieved,
            "achieved_at": self.achieved_at.isoformat() if self.achieved_at else None,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Milestone":
        return cls(
            milestone_id=data["milestone_id"],
            name=data["name"],
            description=data.get("description", ""),
            target_percent=data.get("target_percent", 0.0),
            achieved=data.get("achieved", False),
            achieved_at=datetime.fromisoformat(data["achieved_at"]) if data.get("achieved_at") else None,
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
        )


@dataclass
class ProactiveTrigger:
    trigger_id: str
    trigger_type: TriggerType
    condition: dict[str, Any]
    action: str
    action_params: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    last_triggered: datetime | None = None
    trigger_count: int = 0

    def to_dict(self) -> dict:
        return {
            "trigger_id": self.trigger_id,
            "trigger_type": self.trigger_type.value,
            "condition": self.condition,
            "action": self.action,
            "action_params": self.action_params,
            "enabled": self.enabled,
            "last_triggered": self.last_triggered.isoformat() if self.last_triggered else None,
            "trigger_count": self.trigger_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProactiveTrigger":
        return cls(
            trigger_id=data["trigger_id"],
            trigger_type=TriggerType(data["trigger_type"]),
            condition=data.get("condition", {}),
            action=data["action"],
            action_params=data.get("action_params", {}),
            enabled=data.get("enabled", True),
            last_triggered=datetime.fromisoformat(data["last_triggered"]) if data.get("last_triggered") else None,
            trigger_count=data.get("trigger_count", 0),
        )


@dataclass
class GoalProgress:
    progress_percent: float = 0.0
    current_milestone: str | None = None
    blocker: str | None = None
    notes: deque = field(default_factory=lambda: deque(maxlen=100))
    metrics: dict[str, Any] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)

    def add_note(self, note: str, progress_delta: float = 0.0) -> None:
        self.notes.append({
            "timestamp": datetime.now().isoformat(),
            "note": note,
            "progress_delta": progress_delta,
        })

    def to_dict(self) -> dict:
        return {
            "progress_percent": self.progress_percent,
            "current_milestone": self.current_milestone,
            "blocker": self.blocker,
            "notes": list(self.notes),
            "metrics": self.metrics,
            "last_updated": self.last_updated.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GoalProgress":
        return cls(
            progress_percent=data.get("progress_percent", 0.0),
            current_milestone=data.get("current_milestone"),
            blocker=data.get("blocker"),
            notes=deque(data.get("notes", []), maxlen=100),
            metrics=data.get("metrics", {}),
            last_updated=datetime.fromisoformat(data["last_updated"]) if data.get("last_updated") else datetime.now(),
        )


@dataclass
class AutonomousGoal:
    goal_id: str
    name: str
    description: str
    status: GoalStatus = GoalStatus.DRAFT
    priority: GoalPriority = GoalPriority.MEDIUM
    
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    target_date: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    
    progress: GoalProgress = field(default_factory=GoalProgress)
    milestones: list[Milestone] = field(default_factory=list)
    
    dependencies: list[str] = field(default_factory=list)
    blocked_by: list[str] = field(default_factory=list)
    sub_goals: list[str] = field(default_factory=list)
    parent_goal: str | None = None
    
    triggers: list[ProactiveTrigger] = field(default_factory=list)
    
    tags: list[str] = field(default_factory=list)
    category: str = "general"
    owner: str = "jarvis"
    
    metadata: dict[str, Any] = field(default_factory=dict)
    
    review_frequency_days: int = 7
    last_reviewed: datetime | None = None
    next_review: datetime | None = None
    
    auto_decompose: bool = True
    auto_retry_on_failure: bool = True
    max_auto_retries: int = 3
    retry_count: int = 0

    def to_dict(self) -> dict:
        return {
            "goal_id": self.goal_id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "priority": self.priority.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "target_date": self.target_date.isoformat() if self.target_date else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "progress": self.progress.to_dict(),
            "milestones": [m.to_dict() for m in self.milestones],
            "dependencies": self.dependencies,
            "blocked_by": self.blocked_by,
            "sub_goals": self.sub_goals,
            "parent_goal": self.parent_goal,
            "triggers": [t.to_dict() for t in self.triggers],
            "tags": self.tags,
            "category": self.category,
            "owner": self.owner,
            "metadata": self.metadata,
            "review_frequency_days": self.review_frequency_days,
            "last_reviewed": self.last_reviewed.isoformat() if self.last_reviewed else None,
            "next_review": self.next_review.isoformat() if self.next_review else None,
            "auto_decompose": self.auto_decompose,
            "auto_retry_on_failure": self.auto_retry_on_failure,
            "max_auto_retries": self.max_auto_retries,
            "retry_count": self.retry_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AutonomousGoal":
        return cls(
            goal_id=data["goal_id"],
            name=data["name"],
            description=data.get("description", ""),
            status=GoalStatus(data.get("status", "draft")),
            priority=GoalPriority(data.get("priority", 50)),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            target_date=datetime.fromisoformat(data["target_date"]) if data.get("target_date") else None,
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            progress=GoalProgress.from_dict(data.get("progress", {})),
            milestones=[Milestone.from_dict(m) for m in data.get("milestones", [])],
            dependencies=data.get("dependencies", []),
            blocked_by=data.get("blocked_by", []),
            sub_goals=data.get("sub_goals", []),
            parent_goal=data.get("parent_goal"),
            triggers=[ProactiveTrigger.from_dict(t) for t in data.get("triggers", [])],
            tags=data.get("tags", []),
            category=data.get("category", "general"),
            owner=data.get("owner", "jarvis"),
            metadata=data.get("metadata", {}),
            review_frequency_days=data.get("review_frequency_days", 7),
            last_reviewed=datetime.fromisoformat(data["last_reviewed"]) if data.get("last_reviewed") else None,
            next_review=datetime.fromisoformat(data["next_review"]) if data.get("next_review") else None,
            auto_decompose=data.get("auto_decompose", True),
            auto_retry_on_failure=data.get("auto_retry_on_failure", True),
            max_auto_retries=data.get("max_auto_retries", 3),
            retry_count=data.get("retry_count", 0),
        )

    def is_overdue(self) -> bool:
        if not self.target_date:
            return False
        return datetime.now() > self.target_date and self.status not in (GoalStatus.COMPLETED, GoalStatus.CANCELLED)

    def needs_review(self) -> bool:
        if not self.next_review:
            return True
        return datetime.now() >= self.next_review

    def is_actionable(self) -> bool:
        if self.status not in (GoalStatus.ACTIVE, GoalStatus.IN_PROGRESS):
            return False
        if self.blocked_by:
            return False
        return True

    def get_next_milestone(self) -> Milestone | None:
        for milestone in sorted(self.milestones, key=lambda m: m.target_percent):
            if not milestone.achieved:
                return milestone
        return None


class AutonomousGoals:
    def __init__(self, storage_path: Path | None = None):
        self.storage_path = storage_path or Path.home() / ".jarvis" / "goals"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self._goals_file = self.storage_path / "goals.json"
        self._goals: dict[str, AutonomousGoal] = {}
        
        self._lock = threading.RLock()
        self._action_handlers: dict[str, Callable] = {}
        self._progress_callbacks: list[Callable] = []
        
        self._running = False
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        
        self._max_goals = 1000
        self._check_interval = 60.0
        
        self._load_state()

    def _load_state(self) -> None:
        if self._goals_file.exists():
            try:
                data = read_json_with_integrity(self._goals_file)
                for goal_data in data.get("goals", []):
                    goal = AutonomousGoal.from_dict(goal_data)
                    self._goals[goal.goal_id] = goal
                logger.info(f"Loaded {len(self._goals)} goals")
            except Exception as e:
                logger.error(f"Failed to load goals: {e}")

    def _snapshot_state(self) -> dict:
        return {
            "goals": [g.to_dict() for g in self._goals.values()],
            "updated_at": datetime.now().isoformat(),
        }

    def _persist_state(self, data: dict) -> None:
        try:
            atomic_write_json(self._goals_file, data)
        except Exception as e:
            logger.error(f"Failed to save goals: {e}")

    def _save_state(self) -> None:
        with self._lock:
            data = self._snapshot_state()
        self._persist_state(data)

    def create(
        self,
        name: str,
        description: str = "",
        priority: GoalPriority | int = GoalPriority.MEDIUM,
        target_date: datetime | None = None,
        milestones: list[dict] | None = None,
        dependencies: list[str] | None = None,
        triggers: list[dict] | None = None,
        tags: list[str] | None = None,
        category: str = "general",
        metadata: dict[str, Any] | None = None,
        auto_decompose: bool = True,
    ) -> AutonomousGoal:
        with self._lock:
            if len(self._goals) >= self._max_goals:
                completed = [gid for gid, g in self._goals.items() if g.status == GoalStatus.COMPLETED]
                for gid in completed[:len(completed) // 2 + 1]:
                    del self._goals[gid]
                if len(self._goals) >= self._max_goals:
                    oldest = min(self._goals.values(), key=lambda g: g.updated_at)
                    del self._goals[oldest.goal_id]

            goal_id = f"goal_{secrets.token_hex(8)}"
            
            if isinstance(priority, int):
                priority = GoalPriority(priority) if priority in [p.value for p in GoalPriority] else GoalPriority.MEDIUM
            
            milestone_objs = []
            if milestones:
                for i, m in enumerate(milestones):
                    milestone_objs.append(Milestone(
                        milestone_id=f"{goal_id}_m{i+1}",
                        name=m.get("name", f"Milestone {i+1}"),
                        description=m.get("description", ""),
                        target_percent=m.get("target_percent", (i + 1) * 100 / len(milestones)),
                    ))
            
            trigger_objs = []
            if triggers:
                for i, t in enumerate(triggers):
                    trigger_objs.append(ProactiveTrigger(
                        trigger_id=f"{goal_id}_t{i+1}",
                        trigger_type=TriggerType(t.get("type", "condition_based")),
                        condition=t.get("condition", {}),
                        action=t.get("action", ""),
                        action_params=t.get("params", {}),
                    ))
            
            next_review = datetime.now() + timedelta(days=7)
            
            goal = AutonomousGoal(
                goal_id=goal_id,
                name=name,
                description=description,
                priority=priority,
                target_date=target_date,
                milestones=milestone_objs,
                dependencies=dependencies or [],
                triggers=trigger_objs,
                tags=tags or [],
                category=category,
                metadata=metadata or {},
                auto_decompose=auto_decompose,
                next_review=next_review,
            )
            
            self._goals[goal_id] = goal
            data = self._snapshot_state()
        
        self._persist_state(data)
        
        logger.info(f"Created goal: {name} ({goal_id})")
        return goal

    def activate(self, goal_id: str) -> bool:
        with self._lock:
            goal = self._goals.get(goal_id)
            if not goal:
                return False
            
            if goal.dependencies:
                unmet = [d for d in goal.dependencies if d in self._goals and self._goals[d].status != GoalStatus.COMPLETED]
                if unmet:
                    goal.blocked_by = unmet
                    goal.status = GoalStatus.BLOCKED
                    data = self._snapshot_state()
                    blocked = True
                    blocked_by = unmet
                else:
                    data = None
                    blocked = False
                    blocked_by = None
            else:
                data = None
                blocked = False
                blocked_by = None
            
            if not blocked:
                goal.status = GoalStatus.ACTIVE
                goal.started_at = datetime.now() if not goal.started_at else goal.started_at
                goal.updated_at = datetime.now()
                data = self._snapshot_state()
        
        if data is not None:
            self._persist_state(data)
        
        if blocked:
            logger.info(f"Goal {goal_id} blocked by: {blocked_by}")
            return False
            
        logger.info(f"Activated goal: {goal_id}")
        return True

    def update_progress(
        self,
        goal_id: str,
        progress_percent: float,
        note: str = "",
        milestone_id: str | None = None,
    ) -> bool:
        with self._lock:
            goal = self._goals.get(goal_id)
            if not goal:
                return False
            
            old_progress = goal.progress.progress_percent
            goal.progress.progress_percent = min(100.0, max(0.0, progress_percent))
            goal.progress.last_updated = datetime.now()
            
            if note:
                goal.progress.add_note(note, progress_percent - old_progress)
            
            for milestone in sorted(goal.milestones, key=lambda m: m.target_percent):
                if milestone.target_percent <= goal.progress.progress_percent and not milestone.achieved:
                    milestone.achieved = True
                    milestone.achieved_at = datetime.now()
                    goal.progress.add_note(f"Achieved milestone: {milestone.name}")
                    logger.info(f"Goal {goal_id} achieved milestone: {milestone.name}")
            
            if goal.progress.progress_percent >= 100.0:
                goal.status = GoalStatus.COMPLETED
                goal.completed_at = datetime.now()
                goal.progress.add_note("Goal completed!")
                self._check_dependent_goals(goal_id)
            
            goal.updated_at = datetime.now()
            data = self._snapshot_state()
        
        self._persist_state(data)

        with self._lock:
            callbacks = list(self._progress_callbacks)
        for callback in callbacks:
            try:
                callback(goal, old_progress, goal.progress.progress_percent)
            except Exception as e:
                logger.error(f"Progress callback error: {e}")
        
        return True

    def _check_dependent_goals(self, completed_goal_id: str) -> None:
        for goal_id, goal in self._goals.items():
            if completed_goal_id in goal.dependencies:
                goal.dependencies.remove(completed_goal_id)
            
            if completed_goal_id in goal.blocked_by:
                goal.blocked_by.remove(completed_goal_id)
                if not goal.blocked_by and goal.status == GoalStatus.BLOCKED:
                    goal.status = GoalStatus.ACTIVE
                    logger.info(f"Unblocked goal: {goal_id}")

    def block(self, goal_id: str, blocker: str) -> bool:
        with self._lock:
            goal = self._goals.get(goal_id)
            if not goal:
                return False
            
            if blocker not in goal.blocked_by:
                goal.blocked_by.append(blocker)
            
            if goal.status == GoalStatus.ACTIVE:
                goal.status = GoalStatus.BLOCKED
            
            goal.updated_at = datetime.now()
            data = self._snapshot_state()
        
        self._persist_state(data)
            
        logger.info(f"Blocked goal {goal_id}: {blocker}")
        return True

    def unblock(self, goal_id: str) -> bool:
        with self._lock:
            goal = self._goals.get(goal_id)
            if not goal:
                return False
            
            goal.blocked_by = []
            if goal.status == GoalStatus.BLOCKED:
                goal.status = GoalStatus.ACTIVE
            
            goal.updated_at = datetime.now()
            data = self._snapshot_state()
        
        self._persist_state(data)
        
        logger.info(f"Unblocked goal: {goal_id}")
        return True

    def pause(self, goal_id: str, reason: str = "") -> bool:
        with self._lock:
            goal = self._goals.get(goal_id)
            if not goal:
                return False
            
            goal.status = GoalStatus.PAUSED
            goal.updated_at = datetime.now()
            if reason:
                goal.progress.add_note(f"Paused: {reason}")
            
            data = self._snapshot_state()
        
        self._persist_state(data)
        return True

    def resume(self, goal_id: str) -> bool:
        with self._lock:
            goal = self._goals.get(goal_id)
            if not goal:
                return False
            
            goal.status = GoalStatus.ACTIVE
            goal.updated_at = datetime.now()
            goal.progress.add_note("Resumed")
            
            data = self._snapshot_state()
        
        self._persist_state(data)
        return True

    def fail(self, goal_id: str, reason: str = "") -> bool:
        with self._lock:
            goal = self._goals.get(goal_id)
            if not goal:
                return False
            
            goal.status = GoalStatus.FAILED
            goal.completed_at = datetime.now()
            goal.updated_at = datetime.now()
            if reason:
                goal.progress.add_note(f"Failed: {reason}")
            
            data = self._snapshot_state()
        
        self._persist_state(data)
        return True

    def cancel(self, goal_id: str, reason: str = "") -> bool:
        with self._lock:
            goal = self._goals.get(goal_id)
            if not goal:
                return False
            
            goal.status = GoalStatus.CANCELLED
            goal.completed_at = datetime.now()
            goal.updated_at = datetime.now()
            if reason:
                goal.progress.add_note(f"Cancelled: {reason}")
            
            data = self._snapshot_state()
        
        self._persist_state(data)
        return True

    def add_sub_goal(self, parent_id: str, sub_goal_id: str) -> bool:
        with self._lock:
            parent = self._goals.get(parent_id)
            sub = self._goals.get(sub_goal_id)
            
            if not parent or not sub:
                return False
            
            if sub_goal_id not in parent.sub_goals:
                parent.sub_goals.append(sub_goal_id)
            
            sub.parent_goal = parent_id
            sub.dependencies.append(parent_id)
            
            data = self._snapshot_state()
        
        self._persist_state(data)
        return True

    def add_milestone(
        self,
        goal_id: str,
        name: str,
        target_percent: float,
        description: str = "",
    ) -> Milestone | None:
        with self._lock:
            goal = self._goals.get(goal_id)
            if not goal:
                return None
            
            milestone = Milestone(
                milestone_id=f"{goal_id}_m{len(goal.milestones) + 1}",
                name=name,
                description=description,
                target_percent=target_percent,
            )
            
            goal.milestones.append(milestone)
            goal.updated_at = datetime.now()
            data = self._snapshot_state()
        
        self._persist_state(data)
        return milestone

    def add_trigger(
        self,
        goal_id: str,
        trigger_type: TriggerType,
        condition: dict[str, Any],
        action: str,
        action_params: dict[str, Any] | None = None,
    ) -> ProactiveTrigger | None:
        with self._lock:
            goal = self._goals.get(goal_id)
            if not goal:
                return None
            
            trigger = ProactiveTrigger(
                trigger_id=f"{goal_id}_t{len(goal.triggers) + 1}",
                trigger_type=trigger_type,
                condition=condition,
                action=action,
                action_params=action_params or {},
            )
            
            goal.triggers.append(trigger)
            goal.updated_at = datetime.now()
            data = self._snapshot_state()
        
        self._persist_state(data)
        return trigger

    def get(self, goal_id: str) -> AutonomousGoal | None:
        with self._lock:
            return self._goals.get(goal_id)

    def get_actionable(self) -> list[AutonomousGoal]:
        with self._lock:
            actionable = [
                g for g in self._goals.values()
                if g.is_actionable()
            ]
            return sorted(actionable, key=lambda g: -g.priority.value)

    def get_overdue(self) -> list[AutonomousGoal]:
        with self._lock:
            return [
                g for g in self._goals.values()
                if g.is_overdue()
            ]

    def get_needs_review(self) -> list[AutonomousGoal]:
        with self._lock:
            return [
                g for g in self._goals.values()
                if g.needs_review() and g.status in (GoalStatus.ACTIVE, GoalStatus.IN_PROGRESS)
            ]

    def get_by_status(self, status: GoalStatus) -> list[AutonomousGoal]:
        with self._lock:
            return [g for g in self._goals.values() if g.status == status]

    def get_by_category(self, category: str) -> list[AutonomousGoal]:
        with self._lock:
            return [g for g in self._goals.values() if g.category == category]

    def get_by_tags(self, tags: list[str]) -> list[AutonomousGoal]:
        with self._lock:
            return [
                g for g in self._goals.values()
                if any(t in g.tags for t in tags)
            ]

    def search(self, query: str) -> list[AutonomousGoal]:
        with self._lock:
            query_lower = query.lower()
            return [
                g for g in self._goals.values()
                if query_lower in g.name.lower() or query_lower in g.description.lower()
            ]

    def review(self, goal_id: str, notes: str = "", adjust_priority: GoalPriority | None = None) -> bool:
        with self._lock:
            goal = self._goals.get(goal_id)
            if not goal:
                return False
            
            goal.last_reviewed = datetime.now()
            goal.next_review = datetime.now() + timedelta(days=goal.review_frequency_days)
            
            if notes:
                goal.progress.add_note(f"Review: {notes}")
            
            if adjust_priority:
                goal.priority = adjust_priority
            
            goal.updated_at = datetime.now()
            data = self._snapshot_state()
        
        self._persist_state(data)
        
        return True

    def register_action_handler(self, action: str, handler: Callable) -> None:
        with self._lock:
            self._action_handlers[action] = handler

    def register_progress_callback(self, callback: Callable) -> None:
        with self._lock:
            if len(self._progress_callbacks) >= 20:
                return
            self._progress_callbacks.append(callback)

    def _check_triggers(self) -> None:
        now = datetime.now()
        
        with self._lock:
            trigger_queue = []
            for goal in self._goals.values():
                if goal.status not in (GoalStatus.ACTIVE, GoalStatus.IN_PROGRESS):
                    continue
                
                for trigger in goal.triggers:
                    if not trigger.enabled:
                        continue
                    
                    should_fire = False
                    
                    if trigger.trigger_type == TriggerType.TIME_BASED:
                        schedule = trigger.condition.get("schedule")
                        if schedule:
                            pass
                    
                    elif trigger.trigger_type == TriggerType.PROGRESS_BASED:
                        threshold = trigger.condition.get("threshold", 0)
                        if goal.progress.progress_percent >= threshold:
                            should_fire = True
                    
                    elif trigger.trigger_type == TriggerType.CONDITION_BASED:
                        condition = trigger.condition.get("expression")
                        if condition:
                            should_fire = True
                    
                    if should_fire:
                        handler = self._action_handlers.get(trigger.action)
                        if handler:
                            trigger_queue.append((handler, goal.goal_id, trigger))

        for handler, goal_id, trigger in trigger_queue:
            try:
                handler(self._goals.get(goal_id), trigger.action_params)
            except Exception as e:
                logger.error(f"Trigger handler error: {e}")

        with self._lock:
            for _, goal_id, trigger in trigger_queue:
                trigger.last_triggered = now
                trigger.trigger_count += 1
                logger.info(f"Triggered action for goal {goal_id}: {trigger.action}")

    def start(self) -> None:
        if self._running:
            return
        
        self._running = True
        self._stop_event.clear()
        
        def run():
            while not self._stop_event.is_set():
                try:
                    self._check_triggers()
                    self._check_overdue()
                except Exception as e:
                    logger.error(f"Goals check error: {e}")
                self._stop_event.wait(self._check_interval)
        
        self._thread = threading.Thread(target=run, daemon=True, name="AutonomousGoals")
        self._thread.start()
        logger.info("Autonomous goals system started")

    def stop(self) -> None:
        if not self._running:
            return
        
        self._running = False
        self._stop_event.set()
        
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        
        self._save_state()
        logger.info("Autonomous goals system stopped")

    def _check_overdue(self) -> None:
        with self._lock:
            for goal in self.get_overdue():
                goal.progress.add_note(f"Goal is overdue (target: {goal.target_date})")
                logger.warning(f"Goal {goal.goal_id} is overdue")

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            status_counts = defaultdict(int)
            priority_counts = defaultdict(int)
            category_counts = defaultdict(int)
            
            total_progress = 0
            completed_count = 0
            
            for goal in self._goals.values():
                status_counts[goal.status.value] += 1
                priority_counts[goal.priority.name] += 1
                category_counts[goal.category] += 1
                total_progress += goal.progress.progress_percent
                
                if goal.status == GoalStatus.COMPLETED:
                    completed_count += 1
            
            return {
                "total_goals": len(self._goals),
                "by_status": dict(status_counts),
                "by_priority": dict(priority_counts),
                "by_category": dict(category_counts),
                "average_progress": total_progress / len(self._goals) if self._goals else 0,
                "completed_count": completed_count,
                "overdue_count": len(self.get_overdue()),
                "needs_review_count": len(self.get_needs_review()),
                "is_running": self._running,
            }


_autonomous_goals: AutonomousGoals | None = None
_goals_lock = threading.Lock()


def get_autonomous_goals() -> AutonomousGoals:
    global _autonomous_goals
    if _autonomous_goals is None:
        with _goals_lock:
            if _autonomous_goals is None:
                _autonomous_goals = AutonomousGoals()
    return _autonomous_goals


def create_goal(**kwargs) -> AutonomousGoal:
    return get_autonomous_goals().create(**kwargs)


def update_progress(goal_id: str, progress: float, note: str = "") -> bool:
    return get_autonomous_goals().update_progress(goal_id, progress, note)