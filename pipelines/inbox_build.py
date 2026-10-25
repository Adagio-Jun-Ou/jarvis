"""
Inbox build: materialize the Email-tab read models from the email pipelines.

The app reads two denormalized tables that no job populated before:
  * ``media.alex.inbox``      — one row per email with body + summary + priority
                                (the detail view).
  * ``media.alex.inbox_view`` — the lighter list projection.

Both are derived by joining the raw email (``media.alex.emails``) with its
intelligence (``media.alex.email_intel`` → priority) and its summary
(``media.alex.email_summaries`` → summary). We MERGE so re-runs are
idempotent. Column lists match governance/app_schema.sql exactly.
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

log = logging.getLogger("pipelines.inbox_build")

# The join every read model derives from. Non-archived emails only.
_SOURCE_SQL = """
    SELECT e.owner_email,
           e.id            AS email_id,
           i.priority      AS priority,
           e.subject       AS subject,
           e.from_email    AS from_email,
           e.received_at   AS received_at,
           e.body          AS body,
           s.summary       AS summary,
           CAST(NULL AS STRING) AS thread_id
      FROM media.alex.emails e
      LEFT JOIN media.alex.email_intel i ON i.email_id = e.id
      LEFT JOIN media.alex.email_summaries s ON s.email_id = e.id
     WHERE e.is_archived = false
"""


def run_build() -> None:
    spark = SparkSession.builder.getOrCreate()
    log.info("inbox_build: start")
    spark.sql(_SOURCE_SQL).createOrReplaceTempView("inbox_src")

    # Detail table: full row incl. body + thread_id.
    spark.sql("""
        MERGE INTO media.alex.inbox t USING inbox_src s
          ON t.owner_email = s.owner_email AND t.email_id = s.email_id
        WHEN MATCHED THEN UPDATE SET
          t.priority = s.priority, t.subject = s.subject,
          t.from_email = s.from_email, t.received_at = s.received_at,
          t.body = s.body, t.summary = s.summary, t.thread_id = s.thread_id
        WHEN NOT MATCHED THEN INSERT
          (owner_email, email_id, priority, subject, from_email,
           received_at, body, summary, thread_id)
          VALUES (s.owner_email, s.email_id, s.priority, s.subject,
                  s.from_email, s.received_at, s.body, s.summary, s.thread_id)
    """)

    # List projection: the lighter view the inbox list reads.
    spark.sql("""
        MERGE INTO media.alex.inbox_view t USING inbox_src s
          ON t.owner_email = s.owner_email AND t.email_id = s.email_id
        WHEN MATCHED THEN UPDATE SET
          t.priority = s.priority, t.subject = s.subject,
          t.from_email = s.from_email, t.received_at = s.received_at,
          t.summary = s.summary
        WHEN NOT MATCHED THEN INSERT
          (owner_email, priority, email_id, subject, from_email,
           received_at, summary)
          VALUES (s.owner_email, s.priority, s.email_id, s.subject,
                  s.from_email, s.received_at, s.summary)
    """)
    log.info("inbox_build: done")


if __name__ == "__main__":
    run_build()
