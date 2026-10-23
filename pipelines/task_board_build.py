"""
Task-board build: materialize ``media.alex.task_board_view`` from
``media.alex.tasks``.

The board view is the canonical read surface for /api/tasks and the
agent's ``list_my_tasks`` tool, but (like inbox_view before
pipelines/inbox_build.py) no job populated it. Only live tasks are
surfaced; rows whose task moved to a terminal status are deleted so the
board doesn't show ghosts. MERGE + DELETE keep re-runs idempotent.
Column lists match governance/app_schema.sql exactly.
"""
from __future__ import annotations

# spark_python_task executes this file with sys.path[0] = pipelines/;
# the bundle root (which holds pipelines/ and alex/) must be importable.
import os as _os
import sys as _sys

_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import logging

from pyspark.sql import SparkSession

log = logging.getLogger("pipelines.task_board_build")

# Live = not terminal. The tasks table carries two status vocabularies
# (board lanes + legacy tool statuses — see chk_tasks_status in
# governance/permissions.sql); enumerating the live ones drifts every
# time a lane is added, and the drift is silent: the view just stops
# carrying those tasks. Excluding the terminal pair cannot drift.
_SOURCE_SQL = """
    SELECT owner_email,
           status,
           id       AS task_id,
           title,
           priority,
           due_at
      FROM media.alex.tasks
     WHERE status NOT IN ('done', 'cancelled')
"""


def run_build() -> None:
    spark = SparkSession.builder.getOrCreate()
    log.info("task_board_build: start")
    spark.sql(_SOURCE_SQL).createOrReplaceTempView("board_src")
    spark.sql("""
        MERGE INTO media.alex.task_board_view t USING board_src s
          ON t.owner_email = s.owner_email AND t.task_id = s.task_id
        WHEN MATCHED THEN UPDATE SET
          t.status = s.status, t.title = s.title,
          t.priority = s.priority, t.due_at = s.due_at
        WHEN NOT MATCHED THEN INSERT
          (owner_email, status, task_id, title, priority, due_at)
          VALUES (s.owner_email, s.status, s.task_id, s.title,
                  s.priority, s.due_at)
    """)
    # Remove board rows whose task is done/cancelled (or deleted).
    spark.sql("""
        DELETE FROM media.alex.task_board_view v
         WHERE NOT EXISTS (
           SELECT 1 FROM media.alex.tasks t
            WHERE t.id = v.task_id AND t.status NOT IN ('done', 'cancelled')
         )
    """)
    log.info("task_board_build: done")


if __name__ == "__main__":
    run_build()
