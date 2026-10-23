"""
Standing-order prep: overnight compilation of the day's standing-order
briefs and their suggested actions.

For every active standing order whose audience includes the user:

  1. Materialize the relevant signals (recent emails, tasks, intents).
  2. Ask the LLM to produce a `compiled_context` (one or two
     paragraphs) and a structured list of `suggested_actions` â€” each
     action mapped to a capability from the order's
     `allowed_capabilities` list.
  3. Write a row to `media.control.standing_order_briefs` (one per
     user per day per order).

The chat UI consumes these on the user's first message of the day and
shows them in the "Prepared overnight" strip. The capabilities in
`suggested_actions` are validated against `allowed_capabilities` by
the SQL query that surfaces the brief.
"""
from __future__ import annotations

# spark_python_task executes this file with sys.path[0] = pipelines/;
# the bundle root (which holds pipelines/ and alex/) must be importable.
import os as _os
import sys as _sys

_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import json
import logging
import uuid
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    DateType,
    StringType,
    StructField,
    StructType,
)

from pipelines._helpers import parse_json_response, with_backoff

log = logging.getLogger("pipelines.standing_orders_prep")

MODEL = "databricks-glm-5-3-flash"
PROMPT_VERSION = "v2"
LOOKBACK_DAYS = 1
EMAIL_LIMIT = 8
TASK_LIMIT = 8

SYSTEM_PROMPT = """You are a daily-compilation assistant. You receive
the user's standing order, recent signals, and the list of capabilities
the order is allowed to invoke. Produce a one- or two-paragraph
"compiled_context" and a structured list of "suggested_actions".

Each action MUST use one of the allowed_capabilities listed below. If
none of the user's signals map to a useful action, return an empty
"actions" list â€” do not invent actions that exceed the order's
authority.

Return ONLY a single JSON object, no prose:
{
  "compiled_context": "<= 700 chars, 1-2 paragraphs>",
  "actions": [
    {"action": "<verb phrase>", "capability": "<from allowed_capabilities>", "detail": "<= 200 chars"}
  ]
}
"""


def _active_orders_for(user: str) -> list[dict]:
    spark = SparkSession.builder.getOrCreate()
    sql = """
        SELECT order_id, owner_email, title, instruction, allowed_capabilities, audience
          FROM media.control.standing_orders
         WHERE active = true
           AND (
             array_contains(audience, 'all')
             OR array_contains(audience, :u)
             OR owner_email = :u
           )
    """
    return [r.asDict() for r in spark.sql(sql, args={"u": user}).collect()]


def _signals_for(user: str) -> dict:
    spark = SparkSession.builder.getOrCreate()
    from datetime import datetime, timedelta, timezone

    cutoff = (
        datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)
    ).isoformat()
    emails_sql = """
        SELECT i.email_id, i.priority, i.subject, i.understanding
          FROM media.alex.email_intel i
          JOIN media.alex.emails e ON e.id = i.email_id
         WHERE i.owner_email = :u
           AND e.received_at >= :cutoff
         ORDER BY i.priority ASC
         LIMIT :lim
    """
    tasks_sql = """
        SELECT id, title, status, priority, due_at
          FROM media.alex.tasks
         WHERE (owner_email = :u OR assignee_email = :u)
           AND status NOT IN ('done','cancelled')
           AND updated_at >= :cutoff
         ORDER BY priority ASC
         LIMIT :lim
    """
    p = {"u": user, "cutoff": cutoff, "lim": EMAIL_LIMIT}
    emails = [r.asDict() for r in spark.sql(emails_sql, args=p).collect()]
    p["lim"] = TASK_LIMIT
    tasks = [r.asDict() for r in spark.sql(tasks_sql, args=p).collect()]
    return {"emails": emails, "tasks": tasks}


def _call_llm(prompt: str) -> str:
    spark = SparkSession.builder.getOrCreate()
    sql = "SELECT ai_query(:m, :p)"
    return str(spark.sql(sql, args={"m": MODEL, "p": prompt}).first()[0])


def _allowed_caps(order: dict) -> list[str]:
    raw = order.get("allowed_capabilities")
    if isinstance(raw, list):
        return [str(c) for c in raw]
    if isinstance(raw, str) and raw:
        return [c.strip() for c in raw.split(",") if c.strip()]
    return []


def _build_for_order(user: str, order: dict, signals: dict) -> dict:
    caps = _allowed_caps(order)
    user_prompt = (
        f"STANDING ORDER ({order['order_id']}): {order.get('title','')}\n"
        f"INSTRUCTION: {order.get('instruction','')}\n"
        f"ALLOWED CAPABILITIES: {', '.join(caps)}\n\n"
        f"SIGNALS:\n{json.dumps(signals, default=str)[:6000]}\n"
    )
    try:
        raw = with_backoff(
            lambda: _call_llm(SYSTEM_PROMPT + "\n\n" + user_prompt),
            attempts=3, base_delay=0.5, max_delay=10.0,
        )
    except Exception as exc:
        log.warning("compile failed for %s: %s", order["order_id"], exc)
        return {
            "order_id": order["order_id"],
            "user_email": user,
            "brief_date": str(datetime.utcnow().date()),
            "compiled_context": "(brief unavailable)",
            "suggested_actions": [],
        }
    parsed = parse_json_response(raw) or {}
    actions = []
    seen = set()
    for a in (parsed.get("actions") or []):
        if not isinstance(a, dict):
            continue
        cap = str(a.get("capability") or "")
        if cap not in caps:
            continue  # not in the order's authority
        key = (cap, str(a.get("action") or ""))
        if key in seen:
            continue
        seen.add(key)
        actions.append({
            "action": str(a.get("action") or "")[:200],
            "capability": cap,
            "detail": str(a.get("detail") or "")[:400],
        })
    return {
        "order_id": order["order_id"],
        "user_email": user,
        "brief_date": str(datetime.utcnow().date()),
        "compiled_context": str(parsed.get("compiled_context") or "")[:2000],
        "suggested_actions": actions,
    }


def run_prep() -> None:
    spark = SparkSession.builder.getOrCreate()
    users = [r["email"] for r in
             spark.table("media.alex.users").filter("is_active = true").collect()]
    log.info("standing_orders_prep: %d users", len(users))
    briefs: list[dict] = []
    for u in users:
        orders = _active_orders_for(u)
        if not orders:
            continue
        signals = _signals_for(u)
        for o in orders:
            briefs.append(_build_for_order(u, o, signals))
    if not briefs:
        log.info("no standing orders to compile")
        return
    # Generate brief_ids in Python (Spark cannot generate UUIDs cleanly).
    # Types must match the control_plane.sql DDL: brief_date is DATE and
    # consumed is BOOLEAN NOT NULL â€” writing them as strings corrupted the
    # column on the real runtime.
    rows = []
    for b in briefs:
        rows.append({**b,
                     "brief_id": str(uuid.uuid4()),
                     "brief_date": datetime.strptime(b["brief_date"], "%Y-%m-%d").date(),
                     "consumed": False})
    df = spark.createDataFrame(rows, schema=StructType([
        StructField("brief_id", StringType(), False),
        StructField("order_id", StringType(), False),
        StructField("user_email", StringType(), False),
        StructField("brief_date", DateType(), False),
        StructField("compiled_context", StringType(), False),
        StructField("suggested_actions", ArrayType(StructType([
            StructField("action", StringType(), True),
            StructField("capability", StringType(), True),
            StructField("detail", StringType(), True),
        ])), True),
        StructField("consumed", BooleanType(), False),
    ]))
    # MERGE (not append) so a retry or double-trigger doesn't duplicate the
    # day's briefs â€” the docstring's "one per user per day per order".
    df.createOrReplaceTempView("briefs_in")
    spark.sql("""
        MERGE INTO media.control.standing_order_briefs t
        USING briefs_in s
          ON t.user_email = s.user_email
         AND t.order_id = s.order_id
         AND t.brief_date = s.brief_date
        WHEN MATCHED THEN UPDATE SET
          t.compiled_context = s.compiled_context,
          t.suggested_actions = s.suggested_actions
        WHEN NOT MATCHED THEN INSERT
          (brief_id, order_id, user_email, brief_date, compiled_context,
           suggested_actions, consumed)
          VALUES (s.brief_id, s.order_id, s.user_email, s.brief_date,
                  s.compiled_context, s.suggested_actions, s.consumed)
    """)
    log.info("prep done; %d briefs", len(rows))


if __name__ == "__main__":
    run_prep()
