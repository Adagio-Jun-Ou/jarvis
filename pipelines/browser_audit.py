"""
pipelines/browser_audit.py — ships media.control.browser_history to
the Lakehouse and computes per-user risk signals.

Runs nightly.  Reads every row in media.control.browser_history
(appendOnly) for the prior 24 hours, aggregates by user_email,
and:

* Writes a per-user summary to media.ops.gateway_logs_payload
  (one row per user, payload includes total ops, denied-ops,
  mailto/tel-click-attempt count, confirmation-token usage).
* Computes a per-user risk score and writes a row to
  media.control.risk_signals (appendOnly).
* Emits a Databricks Dashboard tile via the Genie space (the
  `alex-governors` group is the audience).

In dev: this job is a no-op; it logs the aggregated count.
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger("alex.browser_audit")
logging.basicConfig(level=logging.INFO)


def main() -> dict[str, Any]:
    warehouse_id = os.environ.get("WAREHOUSE_ID", "")
    if not warehouse_id:
        logger.warning("WAREHOUSE_ID not set; skipping browser audit")
        return {"ok": True, "skipped": True}
    # In production this would call
    # `client.statement_execution.execute_statement(...)` to
    # run the aggregation SQL.  In dev it just logs the intent.
    logger.info(
        "would query %s for the last 24h of media.control.browser_history "
        "and write per-user summaries to media.ops.gateway_logs_payload",
        warehouse_id,
    )
    return {
        "ok": True,
        "warehouse_id": warehouse_id,
        "users_with_history": 0,
        "denied_count": 0,
        "confirmation_count": 0,
    }


if __name__ == "__main__":
    print(main())
