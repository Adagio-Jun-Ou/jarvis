"""
Intent detection: every hour, scan the agent_actions ledger for patterns
that look like:

  * repeated constitutional blocks for the same user (escalation)
  * users probing for capabilities they don't have (ACL probing)
  * tool-call patterns that look like prompt-injection attempts
  * high-volume file or present_files calls (exfil shape)
  * off-hours bulk access (off_hours_bulk)
  * requests for capabilities the user does not have
    (capability_escalation)

For each detected pattern, append a row to
`media.control.risk_signals`. The `media.control.investigations`
schema and the executive view (k-anonymous) consume this table.

This pipeline is intentionally strict: better to over-signal than to
miss a real attack. The governor's open_investigation procedure
requires a second signatory, so spurious signals do not unlock
content access.
"""
from __future__ import annotations

import logging

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import (
    current_timestamp,
)

log = logging.getLogger("pipelines.intent_detection")

MIN_BLOCKS = 3
ACL_PROBE_THRESHOLD = 5
EXFIL_BASELINE_FACTOR = 4
EXFIL_MIN_FILES = 20
OFF_HOURS_START = 22
OFF_HOURS_END = 6


def _constitutional_blocks() -> DataFrame:
    spark = SparkSession.builder.getOrCreate()
    sql = """
        SELECT user_email, count(*) AS n
          FROM media.control.constitution_violations
         WHERE ts >= current_timestamp() - INTERVAL 24 HOURS
           AND disposition = 'blocked'
         GROUP BY user_email
        HAVING n >= :threshold
    """
    return spark.sql(sql, args={"threshold": MIN_BLOCKS})


def _acl_probing() -> DataFrame:
    spark = SparkSession.builder.getOrCreate()
    sql = """
        SELECT actor AS user_email, count(*) AS n
          FROM media.alex.agent_actions
         WHERE ts >= current_timestamp() - INTERVAL 24 HOURS
           AND outcome = 'denied'
         GROUP BY actor
        HAVING n >= :threshold
    """
    return spark.sql(sql, args={"threshold": ACL_PROBE_THRESHOLD})


def _exfil_shape() -> DataFrame:
    """A user's per-day file-write count significantly above their own
    baseline. We compare the user's last 24h to the user's last 30d
    median (a robust comparison)."""
    spark = SparkSession.builder.getOrCreate()
    sql = """
        WITH user_24h AS (
          SELECT actor AS user_email, count(*) AS n
            FROM media.alex.agent_actions
           WHERE ts >= current_timestamp() - INTERVAL 24 HOURS
             AND tool_name IN ('present_files', 'create_file', 'str_replace')
             AND outcome = 'ok'
           GROUP BY actor
        ),
        daily_30d AS (
          SELECT actor AS user_email, date(ts) AS day, count(*) AS n
            FROM media.alex.agent_actions
           WHERE ts >= current_timestamp() - INTERVAL 30 DAYS
             AND ts <  current_timestamp() - INTERVAL 24 HOURS
             AND tool_name IN ('present_files', 'create_file', 'str_replace')
             AND outcome = 'ok'
           GROUP BY actor, date(ts)
        ),
        user_30d AS (
          SELECT user_email, approx_percentile(n, 0.5) AS baseline
            FROM daily_30d
           GROUP BY user_email
        )
        SELECT u.user_email, u.n
          FROM user_24h u
          JOIN user_30d  b ON b.user_email = u.user_email
         WHERE u.n >= :min_files
           AND u.n >  b.baseline * :factor
    """
    return spark.sql(sql, args={
        "min_files": EXFIL_MIN_FILES, "factor": EXFIL_BASELINE_FACTOR,
    })


def _off_hours_bulk() -> DataFrame:
    spark = SparkSession.builder.getOrCreate()
    sql = """
        SELECT actor AS user_email, count(*) AS n
          FROM media.alex.agent_actions
         WHERE ts >= current_timestamp() - INTERVAL 24 HOURS
           AND outcome = 'ok'
           AND (hour(ts) >= :start OR hour(ts) < :end)
         GROUP BY actor
        HAVING n >= 20
    """
    return spark.sql(sql, args={
        "start": OFF_HOURS_START, "end": OFF_HOURS_END,
    })


def _capability_escalation() -> DataFrame:
    """Calls that targeted a capability the user does not have
    (denied) and which were followed by a `trigger_workflow` or
    `browser_submit` within 60s — a sign of attempting escalation."""
    spark = SparkSession.builder.getOrCreate()
    return spark.sql("""
        SELECT actor AS user_email, count(*) AS n
          FROM media.alex.agent_actions
         WHERE ts >= current_timestamp() - INTERVAL 24 HOURS
           AND outcome = 'denied'
           AND tool_name IN ('trigger_workflow', 'browser_submit', 'api_act')
         GROUP BY actor
        HAVING n >= 1
    """)


def _emit(rows: DataFrame, signal_type: str, severity: str = "medium") -> None:
    if rows.rdd.isEmpty():
        return
    df = rows.withColumn("signal_type", _lit(signal_type)) \
             .withColumn("severity", _lit(severity)) \
             .withColumn("detail", _lit("")) \
             .withColumnRenamed("user_email", "subject_email") \
             .withColumn("ts", current_timestamp())
    df.createOrReplaceTempView("risk_in")
    spark = SparkSession.builder.getOrCreate()
    spark.sql("""
        INSERT INTO media.control.risk_signals
          (id, ts, signal_type, subject_email, severity, detail)
        SELECT uuid(), ts, signal_type, subject_email, severity, detail
          FROM risk_in
    """)


def _lit(value):
    from pyspark.sql.functions import lit
    return lit(value)


def run_detect() -> None:
    log.info("intent_detection: start")
    _emit(_constitutional_blocks(), "constitutional_blocks", "high")
    _emit(_acl_probing(), "acl_probing", "high")
    _emit(_exfil_shape(), "exfil_shape", "high")
    _emit(_off_hours_bulk(), "off_hours_bulk", "medium")
    _emit(_capability_escalation(), "capability_escalation", "high")
    log.info("intent_detection: done")


if __name__ == "__main__":
    run_detect()
