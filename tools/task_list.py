"""Task-list tool — the agent's explicit decomposition primitive.

Mirrors Claude Code's TodoWrite: the model sends the COMPLETE desired
task list each time (not a diff), with exactly one task in_progress.
The list is injected into the system prompt and consulted by the
completion evaluator, so an unfinished list keeps the agentic loop
working (bounded by the loop's max iterations).
"""
from typing import Any

from .base import Tool


class UpdateTasksTool(Tool):
    """Create or update the agent's task list (full-state replace)."""

    name = "update_tasks"
    description = (
        "Maintain your working task list for multi-step work. Send the "
        "COMPLETE list every call (it replaces the previous one). Use it "
        "to decompose work before starting, keep exactly one task "
        "in_progress, and mark tasks completed IMMEDIATELY as soon as "
        "they are done — never batch completions. Skip the list for "
        "trivial single-step requests. Send tasks=[] to clear the list "
        "when the work stream ends."
    )
    parameters = {
        "type": "object",
        "properties": {
            "tasks": {
                "type": "array",
                "description": "The complete task list (full desired state).",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Imperative description of the task.",
                        },
                        "status": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "completed", "cancelled"],
                            "description": "Task status; at most one in_progress.",
                        },
                    },
                    "required": ["content", "status"],
                },
            },
        },
        "required": ["tasks"],
    }
    requires_permission = False

    async def execute(self, tasks: list | None = None) -> str:
        from core.task_list import get_task_list

        task_list = get_task_list()
        if not tasks:
            task_list.clear()
            return "Task list cleared."

        normalized = task_list.update(list(tasks))
        if not normalized:
            return "Task list cleared (no valid entries)."

        lines = [f"{t.marker()} {t.content}" for t in normalized]
        remaining = task_list.unfinished_count()
        summary = f"Task list updated ({remaining} remaining):"
        return summary + "\n" + "\n".join(lines)
