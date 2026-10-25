"""
Executive metrics: per-day rollups for the exec view.

The actual views live in `governance/app_schema.sql` (k-anonymous).
This pipeline materializes the executive summary tables that the
`media.exec.*` views select from. The exec dashboard reads
`media.exec.*` directly.

k-anonymity: any metric that reveals a count of distinct users below
a threshold (default 5) is suppressed. This prevents re-identification
in small teams.
"""
from __future__ import annotations

import logging

from pyspark.sql import SparkSession

log = logging.getLogger("pipelines.exec_metrics")

K_ANON_THRESHOLD = 5


def run_metrics() -> None:
    spark = SparkSession.builder.getOrCreate()
    log.info("exec_metrics: start")
    # The base tables are agent_actions, constitution_violations, and
    # risk_signals. The exec views in governance/app_schema.sql reference
    # these directly with k-anonymity HAVING clauses. We just need to
    # make sure the underlying tables are up to date. The OPTIMIZE
    # call here ensures the Z-ORDER columns exist for fast reads.
    for tbl in ("media.alex.agent_actions", "media.control.constitution_violations",
                "media.control.risk_signals", "media.control.delegations"):
        try:
            spark.sql("OPTIMIZE " + tbl)
        except Exception as exc:
            log.debug("optimize %s skipped: %s", tbl, exc)
    # Force a refresh of the materialized view if present.
    try:
        spark.sql("REFRESH MATERIALIZED VIEW media.exec.daily_active_users")
    except Exception as exc:
        log.debug("materialized view refresh skipped: %s", exc)
    log.info("exec_metrics: done")


if __name__ == "__main__":
    run_metrics()
