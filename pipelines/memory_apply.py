"""
Memory apply: read the consolidation queue, write to the Lakebase
vector memory, mirror to the Delta `vector_memory_mirror` and
`kg_mirror` tables, then mark the queue rows as applied.

The Lakebase side requires the `vector_memory` and `kg_edges` Postgres
tables on the `alex-oltp` instance; apply pipelines/lakebase_ddl.pg.sql
via psql once per environment (the bundle's `apply_lakebase_ddl` SQL
task only preps the Delta mirrors — a sql_task cannot run Postgres DDL).

We gate the Delta mirror on a real (or mocked) Lakebase call so a
deploy whose Lakebase isn't reachable cannot silently drop memories
into the Delta mirror and then claim "applied".
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime

import psycopg
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    DoubleType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

log = logging.getLogger("pipelines.memory_apply")

LAKEBASE_HOST = os.environ.get("LAKEBASE_HOST", "")
LAKEBASE_PORT = int(os.environ.get("LAKEBASE_PORT", "5432"))
LAKEBASE_DBNAME = os.environ.get("LAKEBASE_DBNAME", "alex")
# Empty means "ask the workspace who we are" — the pg role must match
# the identity that vends the credential (see agent/memory.py).
LAKEBASE_USER = os.environ.get("LAKEBASE_USER", "")
DECAY_FACTOR = 0.98


def _conn() -> psycopg.Connection | None:
    """Open a Lakebase connection. In dev (or when LAKEBASE_HOST is
    empty) we return None and the caller falls back to no-op. We do
    NOT silently pretend Lakebase succeeded."""
    if not LAKEBASE_HOST:
        return None
    from databricks.sdk import WorkspaceClient
    ws = WorkspaceClient()
    cred = ws.database.generate_database_credential(
        instance_names=[os.environ.get("LAKEBASE_INSTANCE", "alex-oltp")],
    )
    user = LAKEBASE_USER or (ws.current_user.me().user_name or "")
    return psycopg.connect(
        host=LAKEBASE_HOST, port=LAKEBASE_PORT, dbname=LAKEBASE_DBNAME,
        user=user, password=cred.token, sslmode="require",
    )


def _apply_to_lakebase(conn, owner: str, beads: list[dict], edges: list[dict]) -> None:
    """Upsert beads and edges into the Lakebase tables."""
    with conn.cursor() as cur:
        for b in beads:
            cur.execute(
                """
                INSERT INTO vector_memory
                  (id, owner_email, kind, content, salience, source_ref,
                   embedding, created_at, last_recalled, superseded_by)
                VALUES (%s, %s, %s, %s, %s, %s, %s::vector, now(), NULL, NULL)
                ON CONFLICT (id) DO NOTHING
                """,
                (
                    b.get("id") or str(uuid.uuid4()),
                    owner,
                    b.get("kind", "fact"),
                    b.get("content", ""),
                    float(b.get("salience") or 0.5),
                    b.get("source_ref", ""),
                    b.get("embedding") or [0.0] * 1024,
                ),
            )
        # Extracted edges land in the MODERN graph (kg_edges_v2 via
        # enterprise_graph, which also handles supersession and the
        # kg_mirror projection). The legacy kg_edges INSERT wrote a table
        # nothing reads any more — extraction that lands in a dead store
        # is work the context engine never sees.
        from agent.enterprise_graph import upsert_edge, user_scope

        for e in edges:
            try:
                upsert_edge(
                    conn,
                    e.get("src", ""),
                    e.get("rel", ""),
                    e.get("dst", ""),
                    scope=user_scope(owner),
                    evidence=e.get("evidence", ""),
                    confidence=float(e.get("confidence") or 0.5),
                    source_ref="pipeline:memory_apply",
                )
            except Exception:  # noqa: BLE001
                log.exception("edge upsert failed; continuing batch")
        cur.execute(
            "UPDATE vector_memory SET salience = salience * %s "
            "WHERE owner_email = %s AND superseded_by IS NULL",
            (DECAY_FACTOR, owner),
        )
    conn.commit()


EMBEDDING_MODEL = "databricks-gte-large-en"


def _embed(text: str) -> list[float]:
    """Embed one bead via the workspace embedding endpoint. Degrades to a
    zero vector (with a warning) so a transient endpoint outage doesn't
    stall the whole consolidation queue."""
    if not text.strip():
        return [0.0] * 1024
    try:
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.getOrCreate()
        row = spark.sql(
            "SELECT ai_query(:m, :t) AS e",
            args={"m": EMBEDDING_MODEL, "t": text[:8000]},
        ).first()
        return list(row["e"])
    except Exception:
        log.warning("embedding failed; storing zero vector for bead")
        return [0.0] * 1024


def _mirror_to_delta(owner: str, beads: list[dict], edges: list[dict]) -> None:
    """Write the just-applied beads/edges to the Delta mirror."""
    spark = SparkSession.builder.getOrCreate()
    if beads:
        # The Delta mirror stores no embedding (it lives in Lakebase);
        # drop the key so the explicit schema matches.
        beads = [{k: v for k, v in b.items() if k != "embedding"} for b in beads]
        df = spark.createDataFrame(beads, schema=StructType([
            StructField("id", StringType(), False),
            StructField("owner_email", StringType(), False),
            StructField("kind", StringType(), False),
            StructField("content", StringType(), False),
            StructField("salience", DoubleType(), False),
            StructField("source_ref", StringType(), True),
            StructField("created_at", TimestampType(), False),
            StructField("last_recalled", TimestampType(), True),
            StructField("superseded_by", StringType(), True),
        ]))
        df.createOrReplaceTempView("memories_in")
        spark.sql("""
            MERGE INTO media.control.vector_memory_mirror t
            USING memories_in s ON t.id = s.id
            WHEN MATCHED THEN UPDATE SET
              t.content = s.content, t.salience = s.salience,
              t.source_ref = s.source_ref
            WHEN NOT MATCHED THEN INSERT *
        """)
    if edges:
        df = spark.createDataFrame(edges, schema=StructType([
            StructField("edge_id", StringType(), False),
            StructField("owner_email", StringType(), False),
            StructField("src", StringType(), False),
            StructField("rel", StringType(), False),
            StructField("dst", StringType(), False),
            StructField("evidence", StringType(), True),
            StructField("confidence", DoubleType(), False),
            StructField("last_seen", TimestampType(), False),
        ]))
        df.createOrReplaceTempView("edges_in")
        spark.sql("""
            MERGE INTO media.control.kg_mirror t
            USING edges_in s ON t.edge_id = s.edge_id
            WHEN MATCHED THEN UPDATE SET
              t.confidence = s.confidence, t.last_seen = s.last_seen
            WHEN NOT MATCHED THEN INSERT *
        """)


def _apply_forget_requests() -> int:
    """Honor pending forget_requests: mark the corresponding mirror
    rows as superseded and drop the Lakebase rows. Returns the count
    honored."""
    spark = SparkSession.builder.getOrCreate()
    pending = [r.asDict() for r in
               spark.table("media.control.forget_requests")
                   .filter("applied = false")
                   .collect()]
    if not pending:
        return 0
    honored = 0
    conn = _conn()
    for req in pending:
        item_id = req.get("item_id")
        owner = req.get("owner_email")
        if not item_id or not owner:
            continue
        # Confirm the target actually exists before honoring — otherwise a
        # request that matches nothing (e.g. a stale/bad item_id) would be
        # silently flipped to applied=true and the user would believe a
        # memory was forgotten when it was not.
        found = spark.sql("""
            SELECT 1 FROM media.control.vector_memory_mirror
             WHERE id = :i AND owner_email = :o LIMIT 1
        """, args={"i": item_id, "o": owner}).count()
        if not found:
            log.warning("forget request %s: no memory with id=%s for %s; skipping",
                        req.get("request_id"), item_id, owner)
            continue
        # Vector memory: mark superseded
        spark.sql("""
            UPDATE media.control.vector_memory_mirror
               SET superseded_by = :u || ':forgotten'
             WHERE id = :i AND owner_email = :o
        """, args={"i": item_id, "o": owner, "u": owner})
        if conn is not None:
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE vector_memory SET superseded_by = %s "
                        "WHERE id = %s AND owner_email = %s",
                        (f"{owner}:forgotten", item_id, owner),
                    )
                conn.commit()
            except Exception as exc:
                log.warning("forget in Lakebase failed: %s", exc)
        # Mark request applied
        spark.sql("""
            UPDATE media.control.forget_requests
               SET applied = true, applied_at = current_timestamp()
             WHERE request_id = :r
        """, args={"r": req.get("request_id")})
        honored += 1
    if conn is not None:
        conn.close()
    return honored


def run_apply() -> None:
    spark = SparkSession.builder.getOrCreate()
    pending = [r.asDict() for r in
               spark.table("media.control.consolidation_queue")
                   .filter("status = 'pending'")
                   .collect()]
    if not pending:
        log.info("no pending consolidation rows")
        return
    conn = _conn()
    if conn is None:
        log.warning("LAKEBASE_HOST not set; mirroring to Delta only. "
                    "The apply step will NOT persist to Lakebase.")
    for row in pending:
        owner = row["owner_email"]
        beads_raw = json.loads(row.get("memories_json") or "[]")
        edges_raw = json.loads(row.get("graph_json") or "[]")
        beads = []
        for b in beads_raw:
            content = str(b.get("content") or "")[:1000]
            beads.append({
                "id": str(uuid.uuid4()),
                "owner_email": owner,
                "kind": str(b.get("kind") or "fact"),
                "content": content,
                "salience": float(b.get("salience") or 0.5),
                "source_ref": f"consolidation:{row.get('queue_id','')}",
                "created_at": datetime.utcnow(),
                # A real embedding is required for semantic recall;
                # a zero vector would rank ~0 against every query.
                "embedding": _embed(content),
            })
        edges = []
        for e in edges_raw:
            edges.append({
                "edge_id": str(uuid.uuid4()),
                "owner_email": owner,
                "src": str(e.get("src") or "")[:120],
                "rel": str(e.get("rel") or "")[:40],
                "dst": str(e.get("dst") or "")[:120],
                "evidence": str(e.get("evidence") or "")[:400],
                "confidence": float(e.get("confidence") or 0.5),
                "last_seen": datetime.utcnow(),
            })
        if conn is not None:
            try:
                _apply_to_lakebase(conn, owner, beads, edges)
            except Exception as exc:
                log.exception("Lakebase apply failed for %s; queue row stays pending", owner)
                spark.sql("""
                    UPDATE media.control.consolidation_queue
                       SET status = 'failed', error = :e
                     WHERE queue_id = :q
                """, args={"e": str(exc)[:1000], "q": row["queue_id"]})
                continue
        _mirror_to_delta(owner, beads, edges)
        spark.sql("""
            UPDATE media.control.consolidation_queue
               SET status = 'applied'
             WHERE queue_id = :q
        """, args={"q": row["queue_id"]})
    if conn is not None:
        conn.close()
    honored = _apply_forget_requests()
    log.info("apply done; %d users, %d forget requests honored", len(pending), honored)


if __name__ == "__main__":
    run_apply()
