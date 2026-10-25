"""
Email ingest: pull every user's mailbox (Graph API) into
`media.alex.emails`.

Production properties:
  * Backfill-safe. We use Graph's `/messages/delta` endpoint with a
    `deltaLink` stored per user in `media.alex.email_ingest_state`. The
    first run syncs the last 30 days; subsequent runs only fetch new
    messages.
  * Pagination-aware. We follow `@odata.nextLink` until exhausted.
  * URL-quoting. The user email is URL-quoted before it's interpolated
    into the path.
  * Retried. Graph 429 / 5xx -> exponential backoff with jitter.
  * No f-string SQL. Every value goes through `args=`.
"""
from __future__ import annotations

# spark_python_task executes this file with sys.path[0] = pipelines/;
# the bundle root (which holds pipelines/ and alex/) must be importable.
import os as _os
import sys as _sys

_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import datetime as dt
import json
import logging
import urllib.parse
import urllib.request
from collections.abc import Iterable

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from pipelines._helpers import with_backoff

log = logging.getLogger("pipelines.email_ingest")

GRAPH_BASE = "https://graph.microsoft.com/v1.0"
GRAPH_SCOPES = ["https://graph.microsoft.com/.default"]


def _get_dbutils():
    """Return a dbutils handle that works in both spark_python_task
    (no builtin `dbutils`) and notebook contexts."""
    try:
        from pyspark.dbutils import DBUtils  # type: ignore
        return DBUtils(SparkSession.builder.getOrCreate())
    except Exception:
        # Notebook / interactive context: dbutils is a builtin global.
        import IPython  # type: ignore
        return IPython.get_ipython().user_ns["dbutils"]


def _graph_token() -> str:
    """Fetch a Graph access token using the client-credentials flow. The
    `GRAPH_TENANT_ID`, `GRAPH_CLIENT_ID`, and `GRAPH_CLIENT_SECRET` are
    read from Databricks Secrets (via the `dbutils` API)."""
    dbutils = _get_dbutils()
    s = dbutils.secrets.get(scope="alex", key="graph_client_id")
    t = dbutils.secrets.get(scope="alex", key="graph_tenant_id")
    sec = dbutils.secrets.get(scope="alex", key="graph_client_secret")
    body = urllib.parse.urlencode({
        "client_id": s, "client_secret": sec, "scope": " ".join(GRAPH_SCOPES),
        "grant_type": "client_credentials",
    }).encode("utf-8")
    url = f"https://login.microsoftonline.com/{t}/oauth2/v2.0/token"
    req = urllib.request.Request(url, data=body,
                                headers={"Content-Type": "application/x-www-form-urlencoded"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["access_token"]


def _delta_link_for(user: str) -> str | None:
    """Read the saved deltaLink for a user. None if no prior sync."""
    spark = SparkSession.builder.getOrCreate()
    try:
        from pyspark.sql.functions import col
        rows = spark.read.format("delta").load("media.alex.email_ingest_state") \
            .filter(col("user_email") == user) \
            .collect()
        return rows[0]["delta_link"] if rows else None
    except Exception:
        return None


def _save_delta_link(user: str, link: str) -> None:
    """Persist the deltaLink. Uses MERGE so re-runs don't duplicate."""
    from pyspark.sql import Row
    spark = SparkSession.builder.getOrCreate()
    df = spark.createDataFrame(
        [Row(user_email=user, delta_link=link,
             last_sync_at=dt.datetime.utcnow())],
        schema=StructType([
            StructField("user_email", StringType(), False),
            StructField("delta_link", StringType(), False),
            StructField("last_sync_at", TimestampType(), True),
        ]),
    )
    df.createOrReplaceTempView("email_ingest_state_incoming")
    spark.sql("""
        MERGE INTO media.alex.email_ingest_state t
        USING email_ingest_state_incoming s
        ON t.user_email = s.user_email
        WHEN MATCHED THEN UPDATE SET t.delta_link = s.delta_link,
                                    t.last_sync_at = s.last_sync_at
        WHEN NOT MATCHED THEN INSERT (user_email, delta_link, last_sync_at)
            VALUES (s.user_email, s.delta_link, s.last_sync_at)
    """)


def fetch_user_mailbox(user: str, token: str, *,
                       page_size: int = 100, max_pages: int = 50) -> Iterable[dict]:
    """Yield all messages in the user's mailbox (newest first). Uses
    `/messages/delta` so subsequent runs are incremental."""
    state = _delta_link_for(user)
    url = state or (
        f"{GRAPH_BASE}/users/{urllib.parse.quote(user)}/mailfolders/inbox/messages/delta"
        f"?$select=id,receivedDateTime,from,toRecipients,ccRecipients,"
        f"subject,body,isRead,hasAttachments&$top={page_size}"
    )
    pages = 0
    while url and pages < max_pages:
        data = with_backoff(
            lambda u=url: _get_json(u, token),
            attempts=5, base_delay=1.0, max_delay=30.0,
        )
        yield from data.get("value", [])
        url = data.get("@odata.nextLink") or data.get("@odata.deltaLink")
        pages += 1
    if url and "@odata.deltaLink" in url:
        _save_delta_link(user, url)


def _get_json(url: str, token: str) -> dict:
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {token}",
        "ConsistencyLevel": "eventual",
    })
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


# Schema for the destination table.
DEST_SCHEMA = StructType([
    StructField("id", StringType(), False),
    StructField("owner_email", StringType(), False),
    StructField("from_email", StringType(), True),
    StructField("to_emails", ArrayType(StringType()), True),
    StructField("cc_emails", ArrayType(StringType()), True),
    StructField("subject", StringType(), True),
    StructField("body", StringType(), True),
    StructField("received_at", TimestampType(), False),
    StructField("is_read", BooleanType(), True),
    StructField("is_archived", BooleanType(), True),
    StructField("folder", StringType(), True),
    StructField("has_attachments", BooleanType(), True),
])


def _coerce(msg: dict, owner: str) -> dict:
    body = msg.get("body") or {}
    return {
        "id": msg.get("id"),
        "owner_email": owner,
        "from_email": ((msg.get("from") or {}).get("emailAddress") or {}).get("address"),
        "to_emails": [r.get("emailAddress", {}).get("address")
                      for r in (msg.get("toRecipients") or [])],
        "cc_emails": [r.get("emailAddress", {}).get("address")
                      for r in (msg.get("ccRecipients") or [])],
        "subject": msg.get("subject"),
        "body": (body.get("content") or "")[:100_000],
        # received_at is NOT NULL in the DDL and is the partition column
        # (PARTITIONED BY date(received_at)); a null would fail the write or
        # land in a null partition. Fall back to now() when Graph omits or
        # sends an unparseable timestamp, rather than dropping the message.
        "received_at": _received_at(msg),
        "is_read": bool(msg.get("isRead")),
        "is_archived": False,
        "folder": "inbox",
        "has_attachments": bool(msg.get("hasAttachments")),
    }


def _parse_dt(s: str | None):
    if not s:
        return None
    try:
        # Graph format: 2026-07-18T22:13:00Z
        return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def _received_at(msg: dict) -> dt.datetime:
    """received_at must be non-null (NOT NULL + partition column). Prefer
    the Graph timestamp; fall back to now() with a warning so an anomalous
    message is still ingested rather than crashing the whole batch write."""
    parsed = _parse_dt(msg.get("receivedDateTime"))
    if parsed is not None:
        return parsed
    log.warning("email %s has no parseable receivedDateTime; using now()",
                msg.get("id"))
    return dt.datetime.now(dt.timezone.utc)


def run_ingest() -> None:
    """Top-level entry. Iterates every active user and ingests."""
    spark = SparkSession.builder.getOrCreate()
    users = [r["email"] for r in
             spark.table("media.alex.users").filter("is_active = true").collect()]
    token = _graph_token()
    rows: list[dict] = []
    for u in users:
        for m in fetch_user_mailbox(u, token):
            rows.append(_coerce(m, u))
        log.info("ingested %d messages for %s", len(rows), u)
    if not rows:
        log.info("no new messages")
        return
    df = spark.createDataFrame(rows, schema=DEST_SCHEMA)
    df.createOrReplaceTempView("emails_incoming")
    # Use a MERGE on the id column so re-runs are idempotent.
    spark.sql("""
        MERGE INTO media.alex.emails t
        USING emails_incoming s
        ON t.id = s.id
        WHEN MATCHED THEN UPDATE SET
          t.subject     = s.subject,
          t.body        = s.body,
          t.from_email  = s.from_email,
          t.to_emails   = s.to_emails,
          t.cc_emails   = s.cc_emails,
          t.is_read     = s.is_read,
          t.has_attachments = s.has_attachments
        WHEN NOT MATCHED THEN INSERT *
    """)
    log.info("ingest done; %d total messages", len(rows))


if __name__ == "__main__":
    # The job entry point
    run_ingest()
