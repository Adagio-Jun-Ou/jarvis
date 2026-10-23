"""Task dependencies — enforced order between pieces of work.

``task_id`` is blocked until every ``blocked_by`` task reaches ``done``.
The relation is enforced, not decorative:

* a run cannot START on a task with open blockers
  (:func:`agent.taskboard.start_run` raises :class:`TaskBlocked`) —
  an agent working a task whose input does not exist yet burns a run
  producing something the blocker will invalidate;
* finishing a blocker NOTIFIES the tasks it was holding, in their own
  threads, the moment the last blocker clears — waiting work announces
  itself instead of being rediscovered;
* cycles are refused at insert. A dependency cycle is a deadlock nobody
  can work, and the polite time to find out is when the second edge is
  proposed, not on the board weeks later.

A blocker in ``done`` (or ``cancelled``, which the board maps to done)
no longer blocks. Everything else — including a blocker sitting in
``review`` — does: work awaiting a reviewer's verdict is not yet a fact
its dependents may build on.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("agent.task_deps")

DEPS_TABLE = "media.control.task_deps"
#: Statuses that release a blocker.
_RELEASED = ("done",)
#: A sane bound; a task blocked by more than this is a project, not a task.
MAX_BLOCKERS = 10


class DependencyCycle(ValueError):
    """Adding this edge would make a loop nobody could ever work."""


def blockers_of(sql, task_id: str) -> list[str]:
    """Task ids this task waits on (raw edges, any status)."""
    try:
        rows = sql(
            f"SELECT blocked_by FROM {DEPS_TABLE} WHERE task_id = :t",
            {"t": task_id},
        )
    except Exception:  # noqa: BLE001
        log.debug("blocker read failed for %s", task_id, exc_info=True)
        return []
    return [str(r.get("blocked_by")) for r in rows or [] if r.get("blocked_by")]


def dependents_of(sql, task_id: str) -> list[str]:
    """Task ids waiting on this task."""
    try:
        rows = sql(
            f"SELECT task_id FROM {DEPS_TABLE} WHERE blocked_by = :t",
            {"t": task_id},
        )
    except Exception:  # noqa: BLE001
        return []
    return [str(r.get("task_id")) for r in rows or [] if r.get("task_id")]


def add(sql, task_id: str, blocked_by: str, *, added_by: str) -> None:
    """Declare that ``task_id`` waits on ``blocked_by``. Idempotent.

    Refuses self-dependency and any edge that would close a cycle —
    checked by walking the new blocker's own (transitive) blockers and
    refusing if the walk reaches ``task_id``.
    """
    task_id = (task_id or "").strip()
    blocked_by = (blocked_by or "").strip()
    if not task_id or not blocked_by:
        raise ValueError("both task ids are required")
    if task_id == blocked_by:
        raise DependencyCycle("a task cannot block itself")
    existing = blockers_of(sql, task_id)
    if blocked_by in existing:
        return
    if len(existing) >= MAX_BLOCKERS:
        raise ValueError(
            f"this task already waits on {len(existing)} tasks — "
            "past that it is a project, not a task"
        )
    # Walk blocked_by's transitive blockers; reaching task_id means the
    # proposed edge closes a loop.
    seen: set[str] = set()
    frontier = [blocked_by]
    while frontier:
        current = frontier.pop()
        if current in seen:
            continue
        seen.add(current)
        for upstream in blockers_of(sql, current):
            if upstream == task_id:
                raise DependencyCycle(
                    f"{blocked_by} already waits on {task_id} "
                    "(directly or through other tasks) — this edge would "
                    "make a loop nobody could work"
                )
            frontier.append(upstream)
    sql(
        f"INSERT INTO {DEPS_TABLE} (task_id, blocked_by, added_by, added_at) "
        "VALUES (:t, :b, :u, current_timestamp())",
        {"t": task_id, "b": blocked_by, "u": added_by},
    )


def remove(sql, task_id: str, blocked_by: str) -> None:
    sql(
        f"DELETE FROM {DEPS_TABLE} WHERE task_id = :t AND blocked_by = :b",
        {"t": task_id, "b": blocked_by},
    )


def _status_of(sql, task_id: str) -> str:
    from agent.taskboard import TASKS_TABLE, normalise_status

    try:
        rows = sql(
            f"SELECT status FROM {TASKS_TABLE} WHERE id = :t", {"t": task_id}
        )
    except Exception:  # noqa: BLE001
        return ""
    if not rows:
        # A blocker that no longer exists cannot be waited on; treat it
        # as released rather than blocking a task for ever on a ghost.
        return "done"
    return normalise_status(str(rows[0].get("status") or ""))


def open_blockers(sql, task_id: str) -> list[dict[str, Any]]:
    """The blockers still holding this task, with title and status."""
    from agent.taskboard import TASKS_TABLE

    out: list[dict[str, Any]] = []
    for blocker_id in blockers_of(sql, task_id):
        status = _status_of(sql, blocker_id)
        if status in _RELEASED:
            continue
        title = ""
        try:
            rows = sql(
                f"SELECT title FROM {TASKS_TABLE} WHERE id = :t",
                {"t": blocker_id},
            )
            title = str(rows[0].get("title") or "") if rows else ""
        except Exception:  # noqa: BLE001
            pass
        out.append({"task_id": blocker_id, "title": title, "status": status})
    return out


def notify_released(sql, finished_task_id: str) -> list[str]:
    """Tell each dependent whose LAST blocker just cleared. Returns the
    task ids that were released.

    Posted into the dependent's own thread — the place its crew already
    reads — rather than a notification surface nobody checks. Dependents
    that still wait on something else are left quiet: "one of your
    blockers finished but you are still blocked" is noise.
    """
    from agent import taskboard

    released: list[str] = []
    for dependent in dependents_of(sql, finished_task_id):
        if open_blockers(sql, dependent):
            continue
        try:
            finished_title = ""
            rows = sql(
                f"SELECT title FROM {taskboard.TASKS_TABLE} WHERE id = :t",
                {"t": finished_task_id},
            )
            finished_title = str(rows[0].get("title") or "") if rows else ""
            taskboard.post_message(
                sql,
                dependent,
                author="system",
                body=(
                    f"Unblocked: '{finished_title or finished_task_id}' is "
                    "done — nothing is holding this task any more."
                ),
                kind="system",
            )
            released.append(dependent)
        except Exception:  # noqa: BLE001
            log.debug("release note failed for %s", dependent, exc_info=True)
    return released
