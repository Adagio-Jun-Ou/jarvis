"""Goal Tracker - Track and manage user's goals across sessions.

This module provides sophisticated goal tracking with:
- Goal extraction from conversations
- Progress tracking with tasks and blockers
- Next step suggestions
- Cross-session persistence
"""

import logging
import threading
import time
import re
from dataclasses import dataclass, field
from pathlib import Path

from core.helpers import atomic_write_json, read_json_with_integrity

logger = logging.getLogger(__name__)

DEFAULT_STORAGE_PATH = Path.home() / ".jarvis" / "goals"


@dataclass
class Task:
    """A task within a goal."""
    description: str
    status: str = "pending"  # pending, in_progress, completed, blocked
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class Goal:
    """A user's goal that Jarvis is helping with."""
    
    goal_id: str
    description: str
    status: str = "active"  # active, paused, completed, abandoned
    priority: int = 3  # 1-5, 5 is highest
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    # Progress tracking
    tasks: list[Task] = field(default_factory=list)
    current_task: str | None = None
    blockers: list[str] = field(default_factory=list)
    
    # Context
    related_files: list[str] = field(default_factory=list)
    related_concepts: list[str] = field(default_factory=list)
    deadlines: list[dict] = field(default_factory=list)
    
    # Success criteria
    success_conditions: list[str] = field(default_factory=list)
    estimated_completion: float | None = None
    
    # Metadata
    mention_count: int = 1
    last_mentioned: float = field(default_factory=time.time)
    progress_notes: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "goal_id": self.goal_id,
            "description": self.description,
            "status": self.status,
            "priority": self.priority,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "tasks": [
                {
                    "description": t.description,
                    "status": t.status,
                    "created_at": t.created_at,
                    "completed_at": t.completed_at,
                    "notes": t.notes
                }
                for t in self.tasks
            ],
            "current_task": self.current_task,
            "blockers": self.blockers,
            "related_files": self.related_files,
            "related_concepts": self.related_concepts,
            "deadlines": self.deadlines,
            "success_conditions": self.success_conditions,
            "estimated_completion": self.estimated_completion,
            "mention_count": self.mention_count,
            "last_mentioned": self.last_mentioned,
            "progress_notes": self.progress_notes
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Goal":
        """Create from dictionary."""
        tasks = [
            Task(
                description=t["description"],
                status=t.get("status", "pending"),
                created_at=t.get("created_at", time.time()),
                completed_at=t.get("completed_at"),
                notes=t.get("notes", [])
            )
            for t in data.get("tasks", [])
        ]
        return cls(
            goal_id=data["goal_id"],
            description=data["description"],
            status=data.get("status", "active"),
            priority=data.get("priority", 3),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            tasks=tasks,
            current_task=data.get("current_task"),
            blockers=data.get("blockers", []),
            related_files=data.get("related_files", []),
            related_concepts=data.get("related_concepts", []),
            deadlines=data.get("deadlines", []),
            success_conditions=data.get("success_conditions", []),
            estimated_completion=data.get("estimated_completion"),
            mention_count=data.get("mention_count", 1),
            last_mentioned=data.get("last_mentioned", time.time()),
            progress_notes=data.get("progress_notes", [])
        )


class GoalTracker:
    """Track and manage user's goals across sessions."""
    
    # Patterns for goal extraction
    GOAL_PATTERNS = [
        (r"i want to (build|create|make|develop|write|learn|finish|complete) (.+?)(?:\.|$)", 2),
        (r"i need to (build|create|make|develop|write|learn|finish|complete) (.+?)(?:\.|$)", 2),
        (r"my goal is to (.+?)(?:\.|$)", 1),
        (r"i'm working on (.+?)(?:\.|$)", 1),
        (r"i'm trying to (build|create|make|develop|write|learn) (.+?)(?:\.|$)", 2),
        (r"i plan to (.+?)(?:\.|$)", 1),
        (r"working towards (.+?)(?:\.|$)", 1),
    ]
    
    BLOCKER_PATTERNS = [
        r"stuck on (.+?)(?:\.|$)",
        r"blocked by (.+?)(?:\.|$)",
        r"can't (?:get|figure out|understand) (.+?)(?:\.|$)",
        r"having trouble with (.+?)(?:\.|$)",
        r"problem is (.+?)(?:\.|$)",
    ]
    
    PROGRESS_PATTERNS = [
        (r"(?:finished|completed|done with) (.+?)(?:\.|$)", "completed"),
        (r"(?:started|began) (.+?)(?:\.|$)", "started"),
        (r"(?:made progress on|progress on) (.+?)(?:\.|$)", "progress"),
    ]
    
    FILE_EXTENSIONS = [".py", ".js", ".ts", ".go", ".rs", ".java", ".cpp", ".c", ".h", ".md", ".txt"]
    
    def __init__(self, storage_path: Path = None):
        self.storage_path = storage_path or DEFAULT_STORAGE_PATH
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self._goals: dict[str, Goal] = {}
        self._max_goals = 200
        self._lock = threading.RLock()
        self._load()
    
    def _load(self) -> None:
        """Load goals from disk."""
        goals_file = self.storage_path / "goals.json"
        if goals_file.exists():
            try:
                data = read_json_with_integrity(goals_file)
                for goal_data in data.get("goals", []):
                    goal = Goal.from_dict(goal_data)
                    self._goals[goal.goal_id] = goal
            except Exception as e:
                logger.warning(f"Failed to load goals: {e}")
    
    def _snapshot_state(self) -> dict:
        """Snapshot current state under lock."""
        return {"goals": [g.to_dict() for g in self._goals.values()]}

    def _persist_state(self, data: dict) -> None:
        """Persist state to disk (I/O, no lock needed)."""
        goals_file = self.storage_path / "goals.json"
        try:
            atomic_write_json(goals_file, data)
        except Exception as e:
            logger.warning(f"Failed to save goals: {e}")

    def _save(self) -> None:
        """Save goals to disk."""
        try:
            with self._lock:
                data = self._snapshot_state()
            atomic_write_json(self.storage_path / "goals.json", data)
        except Exception as e:
            logger.warning(f"Failed to save goals: {e}")
    
    def _generate_goal_id(self, description: str) -> str:
        """Generate unique goal ID."""
        import secrets
        return f"goal_{secrets.token_hex(8)}"
    
    def extract_goals(self, text: str) -> list[Goal]:
        """Extract potential goals from user input.
        
        Returns list of newly detected goals (not already tracked).
        """
        text_lower = text.lower()
        new_goals = []
        data_to_save = None

        with self._lock:
            for pattern, group in self.GOAL_PATTERNS:
                match = re.search(pattern, text_lower)
                if match:
                    description = match.group(group).strip()
                    if len(description) < 3:
                        continue
                    
                    existing = self._find_similar_goal(description)
                    if existing:
                        existing.last_mentioned = time.time()
                        existing.mention_count += 1
                        existing.updated_at = time.time()
                        data_to_save = self._snapshot_state()
                    else:
                        goal = Goal(
                            goal_id=self._generate_goal_id(description),
                            description=description,
                            status="active"
                        )
                        self._goals[goal.goal_id] = goal
                        new_goals.append(goal)
        
        if data_to_save:
            self._persist_state(data_to_save)
        
        if new_goals:
            self._save()
        
        return new_goals
    
    def _find_similar_goal(self, description: str) -> Goal | None:
        """Find a similar existing goal using word-level similarity.
        
        Uses Jaccard similarity to find goals with overlapping keywords
        rather than loose substring matching.
        
        Note: Caller must hold self._lock.
        """
        desc_lower = description.lower()
        desc_words = set(desc_lower.split())
        
        # Remove common stop words for better matching
        stop_words = {"a", "an", "the", "to", "for", "and", "or", "in", "on", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "must", "shall", "can", "need", "dare", "ought", "used", "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "here", "there", "when", "where", "why", "how", "all", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "just"}
        desc_words = desc_words - stop_words
        
        best_match = None
        best_similarity = 0.0
        similarity_threshold = 0.4  # 40% word overlap required
        
        for goal in self._goals.values():
            if goal.status != "active":
                continue
            
            # Calculate Jaccard similarity
            goal_words = set(goal.description.lower().split()) - stop_words
            
            if not desc_words or not goal_words:
                continue
            
            intersection = desc_words & goal_words
            union = desc_words | goal_words
            
            similarity = len(intersection) / len(union) if union else 0.0
            
            # Also check if one is subset of other (for short descriptions)
            if len(desc_words) <= 3 or len(goal_words) <= 3:
                if desc_words <= goal_words or goal_words <= desc_words:
                    similarity = max(similarity, 0.5)
            
            if similarity > best_similarity and similarity >= similarity_threshold:
                best_similarity = similarity
                best_match = goal
        
        return best_match
    
    def track_goal(self, goal: Goal) -> None:
        """Start tracking a goal."""
        with self._lock:
            if goal.goal_id in self._goals:
                return
            if len(self._goals) >= self._max_goals:
                self._evict_completed_goals()
            self._goals[goal.goal_id] = goal
            self._save()

    def _evict_completed_goals(self) -> None:
        """Remove completed goals to make room. Must hold self._lock."""
        completed = [gid for gid, g in self._goals.items() if g.status in ("completed", "abandoned")]
        for gid in completed:
            del self._goals[gid]
        if len(self._goals) >= self._max_goals:
            oldest = sorted(self._goals.items(), key=lambda x: x[1].updated_at)[:len(self._goals) - self._max_goals + 10]
            for gid, _ in oldest:
                del self._goals[gid]
    
    def get_active_goals(self) -> list[Goal]:
        """Get currently active goals."""
        with self._lock:
            return [g for g in self._goals.values() if g.status == "active"]
    
    def get_all_goals(self) -> list[Goal]:
        """Get all goals."""
        with self._lock:
            return list(self._goals.values())
    
    def update_goal(self, goal_id: str, **kwargs) -> Goal | None:
        """Update goal properties."""
        with self._lock:
            goal = self._goals.get(goal_id)
            if not goal:
                return None
            
            for key, value in kwargs.items():
                if hasattr(goal, key):
                    setattr(goal, key, value)
            
            goal.updated_at = time.time()
            self._save()
            return goal
    
    def complete_goal(self, goal_id: str) -> Goal | None:
        """Mark a goal as completed."""
        return self.update_goal(goal_id, status="completed")
    
    def pause_goal(self, goal_id: str) -> Goal | None:
        """Pause a goal."""
        return self.update_goal(goal_id, status="paused")
    
    def add_task(self, goal_id: str, task_description: str) -> Goal | None:
        """Add a task to a goal."""
        with self._lock:
            goal = self._goals.get(goal_id)
            if not goal:
                return None
            
            task = Task(description=task_description)
            goal.tasks.append(task)
            goal.updated_at = time.time()
            self._save()
            return goal
    
    def complete_task(self, goal_id: str, task_description: str) -> Goal | None:
        """Mark a task as completed."""
        with self._lock:
            goal = self._goals.get(goal_id)
            if not goal:
                return None
            
            for task in goal.tasks:
                if task_description.lower() in task.description.lower():
                    task.status = "completed"
                    task.completed_at = time.time()
            
            goal.updated_at = time.time()
            self._save()
            return goal
    
    def add_blocker(self, goal_id: str, blocker: str) -> Goal | None:
        """Add a blocker to a goal."""
        with self._lock:
            goal = self._goals.get(goal_id)
            if not goal:
                return None
            
            if blocker not in goal.blockers:
                goal.blockers.append(blocker)
                goal.updated_at = time.time()
                self._save()
            return goal
    
    def remove_blocker(self, goal_id: str, blocker: str) -> Goal | None:
        """Remove a blocker from a goal."""
        with self._lock:
            goal = self._goals.get(goal_id)
            if not goal:
                return None
            
            goal.blockers = [b for b in goal.blockers if blocker.lower() not in b.lower()]
            goal.updated_at = time.time()
            self._save()
            return goal
    
    def detect_updates(self, text: str) -> dict:
        """Detect goal updates from text (blockers, progress, files)."""
        text_lower = text.lower()
        updates = {
            "new_goals": [],
            "blockers": [],
            "progress": [],
            "files": []
        }
        
        updates["new_goals"] = self.extract_goals(text)
        
        for pattern in self.BLOCKER_PATTERNS:
            match = re.search(pattern, text_lower)
            if match:
                updates["blockers"].append(match.group(1).strip())
        
        # Detect progress
        for pattern, status in self.PROGRESS_PATTERNS:
            match = re.search(pattern, text_lower)
            if match:
                updates["progress"].append({
                    "description": match.group(1).strip(),
                    "status": status
                })
        
        # Extract file mentions
        for word in text.split():
            for ext in self.FILE_EXTENSIONS:
                if word.endswith(ext):
                    updates["files"].append(word)
            if "/" in word or "\\" in word:
                if "." in word:
                    updates["files"].append(word)
        
        return updates
    
    def suggest_next_step(self, goal: Goal) -> str:
        """Suggest next step for a goal."""
        if goal.status != "active":
            return f"Goal '{goal.description}' is {goal.status}."
        
        # Check for blockers first
        if goal.blockers:
            blocker = goal.blockers[0]
            return f"You're currently blocked on '{blocker}'. Would you like help resolving this?"
        
        # Check for incomplete tasks
        pending_tasks = [t for t in goal.tasks if t.status == "pending"]
        if pending_tasks:
            return f"Next task: '{pending_tasks[0].description}'. Need help with this?"
        
        # Check for success conditions
        if goal.success_conditions:
            unmet = [c for c in goal.success_conditions if not self._check_condition_met(goal, c)]
            if unmet:
                return f"Still need to achieve: '{unmet[0]}'. What's the next step?"
        
        # Default suggestions based on goal type
        goal_lower = goal.description.lower()
        
        if any(word in goal_lower for word in ["build", "create", "make", "develop"]):
            return "Consider starting with a basic structure or prototype."
        elif any(word in goal_lower for word in ["learn", "study"]):
            return "Try starting with fundamentals and practice exercises."
        elif any(word in goal_lower for word in ["finish", "complete"]):
            return "What's the remaining work? Let's break it down."
        else:
            return "What aspect of this goal would you like to work on?"
    
    def _check_condition_met(self, goal: Goal, condition: str) -> bool:
        """Check if a success condition is met."""
        condition_lower = condition.lower()
        
        # Check tasks
        for task in goal.tasks:
            if task.status == "completed" and task.description.lower() in condition_lower:
                return True
        
        # Check progress notes
        for note in goal.progress_notes:
            if condition_lower in note.lower():
                return True
        
        return False
    
    def get_goal_context(self) -> str:
        """Get goal context for prompts."""
        active_goals = self.get_active_goals()
        if not active_goals:
            return ""
        
        lines = ["<goals>"]
        
        for goal in active_goals[:3]:  # Limit to 3 most active
            status = f" ({goal.status})" if goal.status != "active" else ""
            lines.append(f"  - {goal.description}{status}")
            
            if goal.current_task:
                lines.append(f"    Currently working on: {goal.current_task}")
            
            if goal.blockers:
                lines.append(f"    Blocked by: {', '.join(goal.blockers)}")
            
            if goal.tasks:
                completed = len([t for t in goal.tasks if t.status == "completed"])
                total = len(goal.tasks)
                lines.append(f"    Progress: {completed}/{total} tasks")
        
        lines.append("</goals>")
        return "\n".join(lines)


# Singleton
_tracker: GoalTracker | None = None
_test_storage_path: Path | None = None
_lock = threading.Lock()
_UNSET = object()


def get_goal_tracker() -> GoalTracker:
    """Get singleton GoalTracker instance."""
    global _tracker
    if _tracker is None:
        with _lock:
            if _tracker is None:
                _tracker = GoalTracker(storage_path=_test_storage_path)
    return _tracker


def reset_goal_tracker(storage_path: Path | None = _UNSET) -> None:
    """Reset the singleton (for testing).

    If *storage_path* is given, the next ``get_goal_tracker()`` will construct
    against an isolated store instead of the default ``~/.jarvis/goals``. This
    keeps singleton-based tests from reading or mutating the real on-disk
    state — which previously made assertions depend on prior runs.

    Calling without an argument preserves any previously configured test
    storage path, so an autouse fixture can set it once and individual tests
    can still call ``reset_goal_tracker()`` to drop the cached singleton.
    """
    global _tracker, _test_storage_path
    with _lock:
        _tracker = None
        if storage_path is not _UNSET:
            _test_storage_path = storage_path