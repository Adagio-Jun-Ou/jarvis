"""Agent task list — Claude Code TodoWrite semantics for Jarvis.

The agent maintains an explicit decomposition of its current work as a
short task list (content + status). Research basis (Anthropic agent
guidance): explicit todo tracking (1) forces task decomposition before
execution, (2) keeps the plan visible to the user (transparency), and
(3) survives context compaction as *state* rather than conversation.

Semantics mirrors Claude Code's TodoWrite:
- the list is REPLACED wholesale on each update (the model sends the
  full desired state, the store normalizes it)
- at most ONE task is in_progress at a time (extras are demoted to
  pending, never silently dropped)
- statuses: pending | in_progress | completed | cancelled
- bounded: at most MAX_TASKS entries (oldest completed entries are
  evicted first by the tool layer, not here)

The completion evaluator consults this store: while unfinished tasks
remain, the agentic loop continues instead of accepting a COMPLETE
verdict (bounded by the loop's max iterations).
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

_task_list: Optional["AgentTaskList"] = None
_task_list_lock = threading.Lock()

VALID_STATUSES = ("pending", "in_progress", "completed", "cancelled")


@dataclass
class AgentTask:
    """One task in the agent's working decomposition."""

    id: int
    content: str
    status: str = "pending"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    @property
    def is_terminal(self) -> bool:
        return self.status in ("completed", "cancelled")

    def marker(self) -> str:
        return {
            "completed": "[✓]",
            "in_progress": "[►]",
            "cancelled": "[✗]",
        }.get(self.status, "[ ]")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "status": self.status,
        }


class AgentTaskList:
    """Thread-safe, bounded task list for the current work stream."""

    MAX_TASKS = 20
    MAX_CONTENT_CHARS = 300

    def __init__(self):
        self._lock = threading.RLock()
        self._tasks: list[AgentTask] = []
        self._next_id = 1

    # ------------------------------------------------------------------ #
    # Mutation
    # ------------------------------------------------------------------ #
    def update(self, tasks: list[dict]) -> list[AgentTask]:
        """Replace the list with *tasks* (TodoWrite full-state update).

        Normalizes: validates statuses (unknown -> pending), clamps
        content length, enforces a single in_progress (later ones are
        demoted to pending), splits an over-long list by evicting
        terminal entries first, then the oldest pending ones.
        Returns the normalized list.
        """
        with self._lock:
            normalized: list[AgentTask] = []
            seen_in_progress = False
            for raw in tasks[: self.MAX_TASKS * 2]:
                if not isinstance(raw, dict):
                    continue
                content = str(raw.get("content", "")).strip()
                if not content:
                    continue
                status = str(raw.get("status", "pending")).strip().lower()
                if status not in VALID_STATUSES:
                    status = "pending"
                if status == "in_progress":
                    if seen_in_progress:
                        status = "pending"  # only one active task
                    else:
                        seen_in_progress = True
                task = AgentTask(
                    id=self._next_id,
                    content=content[: self.MAX_CONTENT_CHARS],
                    status=status,
                )
                self._next_id += 1
                normalized.append(task)

            # Bound the list: drop cancelled first, then oldest completed,
            # then oldest pending — never the in_progress task.
            if len(normalized) > self.MAX_TASKS:
                active = [t for t in normalized if not t.is_terminal]
                terminal = [t for t in normalized if t.is_terminal]
                keep_active = active[: self.MAX_TASKS]
                room = self.MAX_TASKS - len(keep_active)
                normalized = keep_active + terminal[: max(0, room)]

            self._tasks = normalized
            logger.debug(f"Task list updated: {len(normalized)} tasks")
            return list(self._tasks)

    def clear(self) -> None:
        with self._lock:
            self._tasks = []

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #
    def get_tasks(self) -> list[AgentTask]:
        with self._lock:
            return list(self._tasks)

    def unfinished_count(self) -> int:
        with self._lock:
            return sum(1 for t in self._tasks if not t.is_terminal)

    def has_unfinished(self) -> bool:
        return self.unfinished_count() > 0

    def next_action(self) -> Optional[str]:
        """Content of the in_progress task, else the first pending one."""
        with self._lock:
            for t in self._tasks:
                if t.status == "in_progress":
                    return t.content
            for t in self._tasks:
                if t.status == "pending":
                    return t.content
        return None

    def to_prompt(self) -> str:
        """Compact rendering for system-prompt injection.

        Empty string when there is nothing to show (no list, or every
        task terminal AND already seen — a finished list is kept for
        the user's benefit in the UI but is not worth prompt tokens
        beyond the session that completed it).
        """
        with self._lock:
            if not self._tasks:
                return ""
            lines = [f"{t.marker()} {t.content}" for t in self._tasks]
            remaining = self.unfinished_count()
            if remaining:
                header = f"## Task List ({remaining} remaining)"
                footer = ("Work through these in order; mark each "
                          "completed immediately with update_tasks.")
                return header + "\n" + "\n".join(lines) + "\n" + footer
            return "## Task List (all complete)\n" + "\n".join(lines)

    def get_state(self) -> dict:
        with self._lock:
            return {
                "tasks": [t.to_dict() for t in self._tasks],
                "unfinished": self.unfinished_count(),
                "total": len(self._tasks),
            }


def get_task_list() -> AgentTaskList:
    global _task_list
    with _task_list_lock:
        if _task_list is None:
            _task_list = AgentTaskList()
        return _task_list


def reset_task_list() -> None:
    global _task_list
    with _task_list_lock:
        _task_list = None
