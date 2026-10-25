"""Agentic Project Workspace for autonomous software development.

Enables Jarvis to work on complex multi-file projects agentically:
- Project context awareness (goals, state, architecture)
- Hierarchical task management (epics -> stories -> tasks)
- Progress tracking with checkpoints and resumption
- Architecture understanding and code relationships
- Multi-step execution planning and estimation
- Long-term memory across sessions

Key Concepts:
1. Project: A workspace with goals, files, and tasks
2. Epic: High-level feature or milestone
3. Story: User-facing capability within an epic
4. Task: Atomic unit of work with acceptance criteria
5. Checkpoint: Saved state for resumption

Example:
    workspace = ProjectWorkspace("/home/user/Research Project")
    workspace.load()
    
    # Add a goal
    workspace.add_epic("Improve cell viability simulation")
    
    # Create tasks
    epic = workspace.get_epic("cell-viability")
    epic.add_story("Add new cell type parameters")
    story = epic.get_story("cell-type-params")
    story.add_task("Update cell_types_library.py with new parameters")
    
    # Work on tasks
    task = workspace.get_next_task()
    result = workspace.execute_task(task)
    
    # Save progress
    workspace.checkpoint("Added new cell type parameters")
"""
from core.helpers import atomic_write_json, read_json_with_integrity
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
import hashlib
import re

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    REVIEW = "review"
    DONE = "done"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


class TaskType(Enum):
    IMPLEMENTATION = "implementation"
    REFACTORING = "refactoring"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    RESEARCH = "research"
    FIX = "fix"
    REVIEW = "review"
    INFRASTRUCTURE = "infrastructure"


@dataclass
class Artifact:
    """A file or resource produced/consumed by a task."""
    path: str
    artifact_type: str  # "source", "test", "config", "doc", "data"
    checksum: str | None = None
    last_modified: datetime | None = None
    lines_added: int = 0
    lines_removed: int = 0
    
    def compute_checksum(self, base_path: Path) -> str:
        file_path = base_path / self.path
        if file_path.exists():
            self.last_modified = datetime.fromtimestamp(file_path.stat().st_mtime)
            return hashlib.sha256(file_path.read_bytes()).hexdigest()[:16]
        return ""


@dataclass
class AcceptanceCriterion:
    """A specific acceptance criterion for a task."""
    description: str
    verification_method: str  # "test", "review", "automated", "manual"
    status: str = "pending"  # pending, passed, failed
    
    def to_dict(self) -> dict:
        return {
            "description": self.description,
            "verification_method": self.verification_method,
            "status": self.status,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "AcceptanceCriterion":
        return cls(
            description=data["description"],
            verification_method=data.get("verification_method", "manual"),
            status=data.get("status", "pending"),
        )


@dataclass
class Task:
    """Atomic unit of work with acceptance criteria."""
    id: str
    title: str
    description: str
    task_type: TaskType = TaskType.IMPLEMENTATION
    priority: TaskPriority = TaskPriority.MEDIUM
    status: TaskStatus = TaskStatus.TODO
    epic_id: str | None = None
    story_id: str | None = None
    
    acceptance_criteria: list[AcceptanceCriterion] = field(default_factory=list)
    
    artifacts_created: list[str] = field(default_factory=list)
    artifacts_modified: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    
    estimated_minutes: int = 30
    actual_minutes: int = 0
    
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    
    assignee: str = "jarvis"
    reviewer: str | None = None
    review_notes: str | None = None
    
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "task_type": self.task_type.value,
            "priority": self.priority.value,
            "status": self.status.value,
            "epic_id": self.epic_id,
            "story_id": self.story_id,
            "acceptance_criteria": [ac.to_dict() for ac in self.acceptance_criteria],
            "artifacts_created": self.artifacts_created,
            "artifacts_modified": self.artifacts_modified,
            "dependencies": self.dependencies,
            "estimated_minutes": self.estimated_minutes,
            "actual_minutes": self.actual_minutes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "assignee": self.assignee,
            "reviewer": self.reviewer,
            "review_notes": self.review_notes,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Task":
        return cls(
            id=data["id"],
            title=data["title"],
            description=data["description"],
            task_type=TaskType(data.get("task_type", "implementation")),
            priority=TaskPriority(data.get("priority", 3)),
            status=TaskStatus(data.get("status", "todo")),
            epic_id=data.get("epic_id"),
            story_id=data.get("story_id"),
            acceptance_criteria=[AcceptanceCriterion.from_dict(ac) for ac in data.get("acceptance_criteria", [])],
            artifacts_created=data.get("artifacts_created", []),
            artifacts_modified=data.get("artifacts_modified", []),
            dependencies=data.get("dependencies", []),
            estimated_minutes=data.get("estimated_minutes", 30),
            actual_minutes=data.get("actual_minutes", 0),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            assignee=data.get("assignee", "jarvis"),
            reviewer=data.get("reviewer"),
            review_notes=data.get("review_notes"),
            metadata=data.get("metadata", {}),
        )
    
    def start(self) -> None:
        self.status = TaskStatus.IN_PROGRESS
        self.started_at = datetime.now()
    
    def complete(self, notes: str = "") -> None:
        self.status = TaskStatus.DONE
        self.completed_at = datetime.now()
        if notes:
            self.review_notes = notes
    
    def block(self, reason: str) -> None:
        self.status = TaskStatus.BLOCKED
        self.review_notes = reason
    
    def estimate_completion(self) -> float:
        if self.status == TaskStatus.DONE:
            return 1.0
        if self.status == TaskStatus.TODO:
            return 0.0
        return (self.actual_minutes / max(1, self.estimated_minutes)) * 0.5


@dataclass
class Story:
    """User-facing capability within an epic."""
    id: str
    title: str
    description: str
    epic_id: str
    
    tasks: list[Task] = field(default_factory=list)
    
    acceptance_criteria: list[AcceptanceCriterion] = field(default_factory=list)
    
    status: TaskStatus = TaskStatus.TODO
    priority: TaskPriority = TaskPriority.MEDIUM
    
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "epic_id": self.epic_id,
            "tasks": [t.to_dict() for t in self.tasks],
            "acceptance_criteria": [ac.to_dict() for ac in self.acceptance_criteria],
            "status": self.status.value,
            "priority": self.priority.value,
            "created_at": self.created_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Story":
        return cls(
            id=data["id"],
            title=data["title"],
            description=data["description"],
            epic_id=data["epic_id"],
            tasks=[Task.from_dict(t) for t in data.get("tasks", [])],
            acceptance_criteria=[AcceptanceCriterion.from_dict(ac) for ac in data.get("acceptance_criteria", [])],
            status=TaskStatus(data.get("status", "todo")),
            priority=TaskPriority(data.get("priority", 3)),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
        )
    
    def add_task(self, task: Task) -> None:
        task.epic_id = self.epic_id
        task.story_id = self.id
        self.tasks.append(task)
    
    def get_task(self, task_id: str) -> Task | None:
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None
    
    def get_next_task(self) -> Task | None:
        todo_tasks = [t for t in self.tasks if t.status == TaskStatus.TODO]
        if todo_tasks:
            sorted_tasks = sorted(todo_tasks, key=lambda t: t.priority.value)
            for task in sorted_tasks:
                if all(self._is_task_done(dep) for dep in task.dependencies):
                    return task
        return None
    
    def _is_task_done(self, task_id: str) -> bool:
        for task in self.tasks:
            if task.id == task_id and task.status == TaskStatus.DONE:
                return True
        return False
    
    def completion_percentage(self) -> float:
        if not self.tasks:
            return 0.0
        done = sum(1 for t in self.tasks if t.status == TaskStatus.DONE)
        return done / len(self.tasks) * 100
    
    def total_estimated_minutes(self) -> int:
        return sum(t.estimated_minutes for t in self.tasks)


@dataclass
class Epic:
    """High-level feature or milestone."""
    id: str
    title: str
    description: str
    milestone: str | None = None
    
    stories: list[Story] = field(default_factory=list)
    
    acceptance_criteria: list[AcceptanceCriterion] = field(default_factory=list)
    
    status: TaskStatus = TaskStatus.TODO
    priority: TaskPriority = TaskPriority.MEDIUM
    
    created_at: datetime = field(default_factory=datetime.now)
    deadline: datetime | None = None
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "milestone": self.milestone,
            "stories": [s.to_dict() for s in self.stories],
            "acceptance_criteria": [ac.to_dict() for ac in self.acceptance_criteria],
            "status": self.status.value,
            "priority": self.priority.value,
            "created_at": self.created_at.isoformat(),
            "deadline": self.deadline.isoformat() if self.deadline else None,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Epic":
        return cls(
            id=data["id"],
            title=data["title"],
            description=data["description"],
            milestone=data.get("milestone"),
            stories=[Story.from_dict(s) for s in data.get("stories", [])],
            acceptance_criteria=[AcceptanceCriterion.from_dict(ac) for ac in data.get("acceptance_criteria", [])],
            status=TaskStatus(data.get("status", "todo")),
            priority=TaskPriority(data.get("priority", 3)),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            deadline=datetime.fromisoformat(data["deadline"]) if data.get("deadline") else None,
        )
    
    def add_story(self, story: Story) -> None:
        story.epic_id = self.id
        self.stories.append(story)
    
    def get_story(self, story_id: str) -> Story | None:
        for story in self.stories:
            if story.id == story_id:
                return story
        return None
    
    def get_all_tasks(self) -> list[Task]:
        tasks = []
        for story in self.stories:
            tasks.extend(story.tasks)
        return tasks
    
    def get_next_task(self) -> Task | None:
        for story in self.stories:
            task = story.get_next_task()
            if task:
                return task
        return None
    
    def completion_percentage(self) -> float:
        all_tasks = self.get_all_tasks()
        if not all_tasks:
            return 0.0
        done = sum(1 for t in all_tasks if t.status == TaskStatus.DONE)
        return done / len(all_tasks) * 100
    
    def total_estimated_minutes(self) -> int:
        return sum(s.total_estimated_minutes() for s in self.stories)


@dataclass  
class Checkpoint:
    """A saved state of the project for resumption."""
    id: str
    message: str
    created_at: datetime = field(default_factory=datetime.now)
    
    current_tasks: list[str] = field(default_factory=list)
    completed_tasks: list[str] = field(default_factory=list)
    
    artifacts_state: dict[str, str] = field(default_factory=dict)
    
    context: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "message": self.message,
            "created_at": self.created_at.isoformat(),
            "current_tasks": self.current_tasks,
            "completed_tasks": self.completed_tasks,
            "artifacts_state": self.artifacts_state,
            "context": self.context,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Checkpoint":
        return cls(
            id=data["id"],
            message=data["message"],
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            current_tasks=data.get("current_tasks", []),
            completed_tasks=data.get("completed_tasks", []),
            artifacts_state=data.get("artifacts_state", {}),
            context=data.get("context", {}),
        )


@dataclass
class ProjectGoal:
    """A high-level project goal with success metrics."""
    id: str
    title: str
    description: str
    success_criteria: list[str] = field(default_factory=list)
    target_date: datetime | None = None
    progress: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "success_criteria": self.success_criteria,
            "target_date": self.target_date.isoformat() if self.target_date else None,
            "progress": self.progress,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ProjectGoal":
        return cls(
            id=data["id"],
            title=data["title"],
            description=data["description"],
            success_criteria=data.get("success_criteria", []),
            target_date=datetime.fromisoformat(data["target_date"]) if data.get("target_date") else None,
            progress=data.get("progress", 0.0),
        )


@dataclass
class ArchitectureNode:
    """A node in the project architecture graph."""
    id: str
    name: str
    node_type: str  # "module", "class", "function", "file", "directory"
    path: str
    dependencies: list[str] = field(default_factory=list)
    dependents: list[str] = field(default_factory=list)
    description: str = ""
    metrics: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "node_type": self.node_type,
            "path": self.path,
            "dependencies": self.dependencies,
            "dependents": self.dependents,
            "description": self.description,
            "metrics": self.metrics,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ArchitectureNode":
        return cls(
            id=data["id"],
            name=data["name"],
            node_type=data["node_type"],
            path=data["path"],
            dependencies=data.get("dependencies", []),
            dependents=data.get("dependents", []),
            description=data.get("description", ""),
            metrics=data.get("metrics", {}),
        )


class ProjectWorkspace:
    """Manages a project workspace for agentic development.
    
    Features:
    - Project context with goals, epics, stories, tasks
    - Architecture understanding and code relationships
    - Checkpoint/save state for session resumption
    - Task decomposition and execution planning
    - Progress tracking and reporting
    
    Example:
        workspace = ProjectWorkspace("/home/user/Research Project")
        workspace.initialize()
        
        # Understand project
        arch = workspace.get_architecture()
        
        # Plan work
        workspace.add_epic("Improve simulation accuracy")
        epic = workspace.get_epic("simulation-accuracy")
        
        # Execute
        task = workspace.get_next_task()
        while task:
            result = workspace.execute_task(task)
            if result.success:
                workspace.complete_task(task.id, result.notes)
            task = workspace.get_next_task()
        
        # Save progress
        workspace.checkpoint("Completed simulation improvements")
    """
    
    def __init__(self, project_path: str | Path):
        self.project_path = Path(project_path)
        self.project_name = self.project_path.name
        
        self._epics: dict[str, Epic] = {}
        self._goals: dict[str, ProjectGoal] = {}
        self._checkpoints: list[Checkpoint] = []
        self._MAX_CHECKPOINTS = 50
        self._architecture: dict[str, ArchitectureNode] = {}
        self._file_cache: dict[str, str] = {}
        self._MAX_FILE_CACHE = 500
        
        self._config_path = self.project_path / ".jarvis_workspace"
        self._config_file = self._config_path / "workspace.json"
        
        self._lock = threading.RLock()
        
        self._created_at: datetime | None = None
        self._last_activity: datetime | None = None
        
        self._metadata: dict = {
            "language": None,
            "frameworks": [],
            "entry_points": [],
            "test_framework": None,
        }
    
    def initialize(self) -> None:
        self._config_path.mkdir(parents=True, exist_ok=True)
        
        if self._config_file.exists():
            self.load()
        else:
            self._analyze_project()
            self.save()
    
    def load(self) -> None:
        if not self._config_file.exists():
            return
        
        try:
            data = read_json_with_integrity(self._config_file)
            
            self._epics = {e["id"]: Epic.from_dict(e) for e in data.get("epics", [])}
            self._goals = {g["id"]: ProjectGoal.from_dict(g) for g in data.get("goals", [])}
            self._checkpoints = [Checkpoint.from_dict(c) for c in data.get("checkpoints", [])]
            self._architecture = {n["id"]: ArchitectureNode.from_dict(n) for n in data.get("architecture", [])}
            self._metadata = data.get("metadata", {})
            self._created_at = datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None
            self._last_activity = datetime.fromisoformat(data["last_activity"]) if data.get("last_activity") else None
            
            logger.info(f"Loaded workspace: {self.project_name} with {len(self._epics)} epics, {len(self._checkpoints)} checkpoints")
            
        except Exception as e:
            logger.error(f"Failed to load workspace: {e}")
    
    def save(self) -> None:
        self._last_activity = datetime.now()
        if not self._created_at:
            self._created_at = datetime.now()
        
        data = {
            "project_name": self.project_name,
            "project_path": str(self.project_path),
            "created_at": self._created_at.isoformat() if self._created_at else None,
            "last_activity": self._last_activity.isoformat() if self._last_activity else None,
            "epics": [e.to_dict() for e in self._epics.values()],
            "goals": [g.to_dict() for g in self._goals.values()],
            "checkpoints": [c.to_dict() for c in self._checkpoints],
            "architecture": [n.to_dict() for n in self._architecture.values()],
            "metadata": self._metadata,
        }
        
        atomic_write_json(self._config_file, data)
        
        logger.debug(f"Saved workspace: {self.project_name}")
    
    def _analyze_project(self) -> None:
        self._detect_language()
        self._detect_frameworks()
        self._find_entry_points()
        self._build_architecture_graph()
    
    def _detect_language(self) -> None:
        extensions = {}
        for path in self.project_path.rglob("*"):
            if path.is_file() and not path.name.startswith("."):
                ext = path.suffix.lower()
                extensions[ext] = extensions.get(ext, 0) + 1
        
        lang_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".go": "go",
            ".rs": "rust",
            ".java": "java",
            ".cpp": "cpp",
            ".c": "c",
            ".md": "markdown",
        }
        
        for ext, count in sorted(extensions.items(), key=lambda x: x[1], reverse=True):
            if ext in lang_map:
                self._metadata["language"] = lang_map[ext]
                break
    
    def _detect_frameworks(self) -> None:
        frameworks = []
        
        req_file = self.project_path / "requirements.txt"
        if req_file.exists():
            content = req_file.read_text(encoding='utf-8').lower()
            if "numpy" in content:
                frameworks.append("numpy")
            if "scipy" in content:
                frameworks.append("scipy")
            if "flask" in content:
                frameworks.append("flask")
            if "django" in content:
                frameworks.append("django")
            if "pytest" in content:
                frameworks.append("pytest")
                self._metadata["test_framework"] = "pytest"
        
        pkg_file = self.project_path / "package.json"
        if pkg_file.exists():
            content = pkg_file.read_text(encoding='utf-8').lower()
            if "react" in content:
                frameworks.append("react")
            if "vue" in content:
                frameworks.append("vue")
            if "express" in content:
                frameworks.append("express")
            if "jest" in content:
                frameworks.append("jest")
                self._metadata["test_framework"] = "jest"
        
        self._metadata["frameworks"] = frameworks
    
    def _find_entry_points(self) -> None:
        entry_points = []
        
        common_entries = [
            "main.py",
            "app.py",
            "run.py",
            "server.py",
            "cli.py",
            "index.js",
            "index.ts",
            "app.js",
            "server.js",
        ]
        
        for entry in common_entries:
            path = self.project_path / entry
            if path.exists():
                entry_points.append(str(path.relative_to(self.project_path)))
        
        for path in self.project_path.rglob("run_*.py"):
            entry_points.append(str(path.relative_to(self.project_path)))
        
        self._metadata["entry_points"] = entry_points[:10]
    
    def _build_architecture_graph(self) -> None:
        if self._metadata["language"] == "python":
            self._build_python_architecture()
    
    def _build_python_architecture(self) -> None:
        py_files = list(self.project_path.rglob("*.py"))
        
        for py_file in py_files:
            try:
                rel_path = str(py_file.relative_to(self.project_path))
                node_id = f"file:{rel_path}"
                
                content = py_file.read_text(errors="ignore")
                imports = self._extract_python_imports(content, rel_path)
                
                node = ArchitectureNode(
                    id=node_id,
                    name=py_file.stem,
                    node_type="file",
                    path=rel_path,
                    dependencies=imports,
                    description=f"Python module: {py_file.stem}",
                    metrics={
                        "lines": len(content.splitlines()),
                        "size_bytes": py_file.stat().st_size,
                    },
                )
                
                self._architecture[node_id] = node
                
            except Exception as e:
                logger.debug(f"Failed to analyze {py_file}: {e}")
        
        for node_id, node in self._architecture.items():
            for dep in node.dependencies:
                dep_node_id = f"file:{dep}" if not dep.startswith("file:") else dep
                if dep_node_id in self._architecture:
                    self._architecture[dep_node_id].dependents.append(node_id)
    
    def _extract_python_imports(self, content: str, file_path: str) -> list[str]:
        imports = []
        
        import_pattern = r'^(?:from\s+(\S+)\s+import|import\s+(\S+))'
        for match in re.finditer(import_pattern, content, re.MULTILINE):
            module = match.group(1) or match.group(2)
            if module:
                module = module.split(".")[0]
                
                if not module.startswith(("sys", "os", "json", "re", "time", "datetime", "logging", "threading", "asyncio", "pathlib", "typing", "dataclasses", "collections", "functools", "itertools", "copy", "hashlib", "math")):
                    potential_path = self.project_path / f"{module}.py"
                    if potential_path.exists():
                        imports.append(f"{module}.py")
                    elif (self.project_path / module).is_dir():
                        imports.append(module)
        
        return list(set(imports))
    
    def get_status(self) -> dict:
        total_tasks = 0
        completed_tasks = 0
        in_progress_tasks = 0
        blocked_tasks = 0
        
        for epic in self._epics.values():
            for story in epic.stories:
                for task in story.tasks:
                    total_tasks += 1
                    if task.status == TaskStatus.DONE:
                        completed_tasks += 1
                    elif task.status == TaskStatus.IN_PROGRESS:
                        in_progress_tasks += 1
                    elif task.status == TaskStatus.BLOCKED:
                        blocked_tasks += 1
        
        return {
            "project_name": self.project_name,
            "project_path": str(self.project_path),
            "language": self._metadata.get("language"),
            "frameworks": self._metadata.get("frameworks", []),
            "entry_points": self._metadata.get("entry_points", []),
            "total_epics": len(self._epics),
            "total_goals": len(self._goals),
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "in_progress_tasks": in_progress_tasks,
            "blocked_tasks": blocked_tasks,
            "completion_percentage": (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0,
            "checkpoints": len(self._checkpoints),
            "architecture_nodes": len(self._architecture),
            "created_at": self._created_at.isoformat() if self._created_at else None,
            "last_activity": self._last_activity.isoformat() if self._last_activity else None,
        }
    
    def add_epic(self, title: str, description: str = "", milestone: str | None = None) -> Epic:
        epic_id = title.lower().replace(" ", "-").replace("_", "-")[:50]
        epic_id = re.sub(r"[^a-z0-9-]", "", epic_id)
        
        if epic_id in self._epics:
            epic_id = f"{epic_id}-{len(self._epics)}"
        
        epic = Epic(
            id=epic_id,
            title=title,
            description=description,
            milestone=milestone,
        )
        
        self._epics[epic_id] = epic
        self.save()
        
        logger.info(f"Added epic: {epic_id}")
        return epic
    
    def get_epic(self, epic_id: str) -> Epic | None:
        return self._epics.get(epic_id)
    
    def add_story(self, epic_id: str, title: str, description: str = "") -> Story | None:
        epic = self._epics.get(epic_id)
        if not epic:
            return None
        
        story_id = f"{epic_id}-{title.lower().replace(' ', '-')[:30]}"
        story_id = re.sub(r"[^a-z0-9-]", "", story_id)
        
        story = Story(
            id=story_id,
            title=title,
            description=description,
            epic_id=epic_id,
        )
        
        epic.add_story(story)
        self.save()
        
        return story
    
    def add_task(
        self,
        story_id: str,
        title: str,
        description: str = "",
        task_type: TaskType = TaskType.IMPLEMENTATION,
        priority: TaskPriority = TaskPriority.MEDIUM,
        estimated_minutes: int = 30,
    ) -> Task | None:
        for epic in self._epics.values():
            for story in epic.stories:
                if story.id == story_id:
                    task_id = f"{story_id}-{len(story.tasks)}"
                    
                    task = Task(
                        id=task_id,
                        title=title,
                        description=description,
                        task_type=task_type,
                        priority=priority,
                        epic_id=epic.id,
                        story_id=story_id,
                        estimated_minutes=estimated_minutes,
                    )
                    
                    story.add_task(task)
                    self.save()
                    
                    return task
        
        return None
    
    def get_task(self, task_id: str) -> Task | None:
        for epic in self._epics.values():
            for story in epic.stories:
                task = story.get_task(task_id)
                if task:
                    return task
        return None
    
    def get_next_task(self, epic_id: str | None = None) -> Task | None:
        epics_to_check = [self._epics[epic_id]] if epic_id and epic_id in self._epics else list(self._epics.values())
        
        highest_priority_task = None
        highest_priority_value = float('inf')
        
        for epic in epics_to_check:
            task = epic.get_next_task()
            if task and task.priority.value < highest_priority_value:
                if all(self._is_dependency_met(dep) for dep in task.dependencies):
                    highest_priority_task = task
                    highest_priority_value = task.priority.value
        
        return highest_priority_task
    
    def _is_dependency_met(self, task_id: str) -> bool:
        task = self.get_task(task_id)
        return task is not None and task.status == TaskStatus.DONE
    
    def start_task(self, task_id: str) -> bool:
        task = self.get_task(task_id)
        if task and task.status == TaskStatus.TODO:
            task.start()
            self.save()
            return True
        return False
    
    def complete_task(self, task_id: str, notes: str = "") -> bool:
        task = self.get_task(task_id)
        if task:
            task.complete(notes)
            self.save()
            return True
        return False
    
    def block_task(self, task_id: str, reason: str) -> bool:
        task = self.get_task(task_id)
        if task:
            task.block(reason)
            self.save()
            return True
        return False
    
    def checkpoint(self, message: str) -> Checkpoint:
        current_tasks = []
        completed_tasks = []
        
        for epic in self._epics.values():
            for story in epic.stories:
                for task in story.tasks:
                    if task.status == TaskStatus.IN_PROGRESS:
                        current_tasks.append(task.id)
                    elif task.status == TaskStatus.DONE:
                        completed_tasks.append(task.id)
        
        artifacts_state = {}
        for node_id, node in self._architecture.items():
            if node.node_type == "file":
                file_path = self.project_path / node.path
                if file_path.exists():
                    checksum = hashlib.sha256(file_path.read_bytes()).hexdigest()[:16]
                    artifacts_state[node.path] = checksum
        
        checkpoint = Checkpoint(
            id=f"checkpoint-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            message=message,
            current_tasks=current_tasks,
            completed_tasks=completed_tasks,
            artifacts_state=artifacts_state,
        )
        
        self._checkpoints.append(checkpoint)
        if len(self._checkpoints) > self._MAX_CHECKPOINTS:
            self._checkpoints = self._checkpoints[-self._MAX_CHECKPOINTS:]
        self.save()
        
        logger.info(f"Created checkpoint: {checkpoint.id}")
        return checkpoint
    
    def get_last_checkpoint(self) -> Checkpoint | None:
        if self._checkpoints:
            return self._checkpoints[-1]
        return None
    
    def get_architecture(self) -> dict[str, ArchitectureNode]:
        return self._architecture
    
    def get_dependencies(self, file_path: str) -> list[str]:
        node_id = f"file:{file_path}"
        node = self._architecture.get(node_id)
        return node.dependencies if node else []
    
    def get_dependents(self, file_path: str) -> list[str]:
        node_id = f"file:{file_path}"
        node = self._architecture.get(node_id)
        return node.dependents if node else []
    
    def find_files(self, pattern: str) -> list[str]:
        matches = []
        for node_id, node in self._architecture.items():
            if node.node_type == "file":
                if pattern.lower() in node.name.lower() or pattern.lower() in node.path.lower():
                    matches.append(node.path)
        return matches
    
    def add_goal(self, title: str, description: str, success_criteria: list[str] | None = None) -> ProjectGoal:
        goal_id = title.lower().replace(" ", "-")[:50]
        goal_id = re.sub(r"[^a-z0-9-]", "", goal_id)
        
        goal = ProjectGoal(
            id=goal_id,
            title=title,
            description=description,
            success_criteria=success_criteria or [],
        )
        
        self._goals[goal_id] = goal
        self.save()
        
        return goal
    
    def get_goal(self, goal_id: str) -> ProjectGoal | None:
        return self._goals.get(goal_id)
    
    def update_goal_progress(self, goal_id: str, progress: float) -> bool:
        goal = self._goals.get(goal_id)
        if goal:
            goal.progress = progress
            self.save()
            return True
        return False
    
    def suggest_next_steps(self, limit: int = 5) -> list[dict]:
        suggestions = []
        
        next_task = self.get_next_task()
        if next_task:
            suggestions.append({
                "type": "task",
                "priority": "high",
                "description": f"Continue working on: {next_task.title}",
                "task_id": next_task.id,
                "epic": next_task.epic_id,
                "story": next_task.story_id,
            })
        
        blocked = []
        for epic in self._epics.values():
            for story in epic.stories:
                for task in story.tasks:
                    if task.status == TaskStatus.BLOCKED:
                        blocked.append({
                            "type": "blocked",
                            "priority": "medium",
                            "description": f"Unblock: {task.title} - {task.review_notes}",
                            "task_id": task.id,
                        })
        suggestions.extend(blocked[:2])
        
        in_progress = []
        for epic in self._epics.values():
            for story in epic.stories:
                for task in story.tasks:
                    if task.status == TaskStatus.IN_PROGRESS:
                        in_progress.append({
                            "type": "in_progress",
                            "priority": "high",
                            "description": f"Resume: {task.title}",
                            "task_id": task.id,
                        })
        suggestions.extend(in_progress[:2])
        
        return suggestions[:limit]
    
    def to_summary(self) -> str:
        status = self.get_status()
        lines = [
            f"Project: {status['project_name']}",
            f"Path: {status['project_path']}",
            f"Language: {status['language'] or 'Unknown'}",
            f"Frameworks: {', '.join(status['frameworks']) or 'None detected'}",
            "",
            f"Progress: {status['completed_tasks']}/{status['total_tasks']} tasks ({status['completion_percentage']:.1f}%)",
            f"Epics: {status['total_epics']}, Goals: {status['total_goals']}",
            f"In Progress: {status['in_progress_tasks']}, Blocked: {status['blocked_tasks']}",
            "",
            "Epics:",
        ]
        
        for epic_id, epic in self._epics.items():
            completion = epic.completion_percentage()
            tasks = len(epic.get_all_tasks())
            lines.append(f"  - {epic.title}: {completion:.0f}% ({tasks} tasks)")
        
        return "\n".join(lines)


_workspace_instances: dict[str, ProjectWorkspace] = {}
_workspace_lock = threading.Lock()


def get_workspace(project_path: str | Path) -> ProjectWorkspace:
    """Get or create a workspace for a project path."""
    path_str = str(Path(project_path).absolute())
    
    with _workspace_lock:
        if path_str not in _workspace_instances:
            workspace = ProjectWorkspace(path_str)
            workspace.initialize()
            _workspace_instances[path_str] = workspace
        return _workspace_instances[path_str]


def close_workspace(project_path: str | Path) -> bool:
    """Close and save a workspace."""
    path_str = str(Path(project_path).absolute())
    
    with _workspace_lock:
        if path_str in _workspace_instances:
            workspace = _workspace_instances.pop(path_str)
            workspace.save()
            return True
    return False