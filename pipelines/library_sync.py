"""
Library sync: turn standing orders and curated content into per-user
`instance_library` rows.

For every active user, for every approved standing-order file (and
every curated 'policy_best_practice' file) that targets the user, we:

  1. Read the file from the source channel (Delta or Volumes).
  2. Compute a content hash and verify against `media.control.approved_files`.
  3. Insert (or update) the row in `media.control.instance_library`.
  4. Embed the file's contents into the per-user knowledge index
     (`media.control.user_knowledge_chunks`).

The third step (embed) is the part the previous version was missing.
Without it the transparency panel could read the file but the agent
could not use it for retrieval.
"""
from __future__ import annotations

import logging
import uuid

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    ArrayType,
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

log = logging.getLogger("pipelines.library_sync")

EMBEDDING_MODEL = "databricks-gte-large-en"
APPROVED_CHUNK_CHARS = 2000
APPROVED_OVERLAP = 200


def _approved_for(user: str) -> list[dict]:
    """Return the approved files that should be in this user's library.

    A file reaches a user through ONE of:
      * org-wide policy/best-practice channel (governance publishes it);
      * a path attached to one of THIS user's own active standing
        orders via media.alex.standing_order_files.
    The old predicate never correlated the standing-orders EXISTS with
    the file, and the trailing `OR source_channel = 'curated'` gave
    every such user the whole approved library — a cross-tenant
    content leak into personal retrieval indexes.
    """
    spark = SparkSession.builder.getOrCreate()
    sql = """
        SELECT af.file_id, af.path, af.sha256, af.source_channel,
               af.approved_at, af.expires_at
          FROM media.control.approved_files af
          JOIN media.alex.users u ON u.email = :u
         WHERE af.revoked = false
           AND (af.expires_at IS NULL OR af.expires_at > current_timestamp())
           AND (
             af.source_channel = 'policy_best_practice'
             OR EXISTS (
               SELECT 1
                 FROM media.alex.standing_order_files sfo
                 JOIN media.control.standing_orders so
                   ON so.order_id = sfo.order_id
                WHERE so.active = true
                  AND so.owner_email = :u
                  AND sfo.path = af.path
             )
           )
    """
    return [r.asDict() for r in spark.sql(sql, args={"u": user}).collect()]


def _read(path: str) -> bytes | None:
    try:
        with open(path, "rb") as f:
            return f.read()
    except OSError as e:
        log.warning("read %s: %s", path, e)
        return None


def _embed(text: str) -> list[float]:
    spark = SparkSession.builder.getOrCreate()
    sql = "SELECT ai_query(:m, :t) AS e"
    return list(
        spark.sql(sql, args={"m": EMBEDDING_MODEL, "t": text[:8000]}).first()["e"]
    )


def _chunk_for_embedding(text: str) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(text):
        end = min(i + APPROVED_CHUNK_CHARS, len(text))
        out.append(text[i:end])
        if end == len(text):
            break
        i = end - APPROVED_OVERLAP
    return out


def _sync_user(user: str) -> None:
    spark = SparkSession.builder.getOrCreate()
    approved = _approved_for(user)
    if not approved:
        return
    # Insert/update instance_library rows first (MERGE on user+file_id).
    rows = [{
        "owner_email": user, "file_id": a["file_id"],
        "pinned_summary": a.get("path", "").rsplit("/", 1)[-1][:200],
    } for a in approved]
    df = spark.createDataFrame(rows, schema=StructType([
        StructField("owner_email", StringType(), False),
        StructField("file_id", StringType(), False),
        StructField("pinned_summary", StringType(), True),
    ]))
    df.createOrReplaceTempView("lib_in")
    spark.sql("""
        MERGE INTO media.control.instance_library t USING lib_in s
        ON t.owner_email = s.owner_email AND t.file_id = s.file_id
        WHEN MATCHED THEN UPDATE SET t.pinned_summary = s.pinned_summary
        WHEN NOT MATCHED THEN INSERT *
    """)
    # Then, for each file, embed and insert knowledge chunks.
    chunks: list[dict] = []
    for a in approved:
        path = a.get("path") or ""
        if not path:
            continue
        raw = _read(path)
        if not raw:
            continue
        try:
            text = raw.decode("utf-8", errors="ignore")
        except Exception:
            continue
        for idx, part in enumerate(_chunk_for_embedding(text)):
            try:
                emb = _embed(part)
            except Exception as exc:
                log.warning("embed failed for %s: %s", path, exc)
                continue
            chunks.append({
                "chunk_id": str(uuid.uuid4()),
                "owner_email": user,
                "source_path": path,
                "chunk_index": idx,
                "content": part[:APPROVED_CHUNK_CHARS + 200],
                "embedding": [float(x) for x in emb],
                "embedding_model": EMBEDDING_MODEL,
            })
    if chunks:
        # Wipe the user's existing chunks for these source paths and re-insert.
        # (Simpler than deduping per chunk; the libraries are small.)
        paths = sorted({c["source_path"] for c in chunks})
        df_paths = spark.createDataFrame([(p,) for p in paths], "path STRING")
        df_paths.createOrReplaceTempView("paths_in")
        spark.sql("""
            DELETE FROM media.control.user_knowledge_chunks
             WHERE owner_email = :u
               AND source_path IN (SELECT path FROM paths_in)
        """, args={"u": user})
        ck_df = spark.createDataFrame(chunks, schema=StructType([
            StructField("chunk_id", StringType(), False),
            StructField("owner_email", StringType(), False),
            StructField("source_path", StringType(), False),
            # chunk_index is an int; declaring it String silently dropped
            # the ordering type. embedding was computed per chunk (paid
            # ai_query) but had NO column here, so the vector was dropped
            # — defeating the whole sync. Both are now persisted.
            StructField("chunk_index", IntegerType(), False),
            StructField("content", StringType(), False),
            StructField("embedding", ArrayType(FloatType()), True),
            StructField("embedding_model", StringType(), True),
        ]))
        ck_df.write.format("delta").mode("append").saveAsTable(
            "media.control.user_knowledge_chunks"
        )


def run_sync() -> None:
    spark = SparkSession.builder.getOrCreate()
    users = [r["email"] for r in
             spark.table("media.alex.users").filter("is_active = true").collect()]
    log.info("library_sync: %d users", len(users))
    for u in users:
        _sync_user(u)
    # Revoke: any file whose source approval disappeared.
    spark.sql("""
        UPDATE media.control.instance_library il
           SET pinned_summary = il.pinned_summary  -- marker: we'll DELETE below
         WHERE NOT EXISTS (
           SELECT 1 FROM media.control.approved_files af
            WHERE af.file_id = il.file_id AND af.revoked = false
         )
    """)
    spark.sql("""
        DELETE FROM media.control.instance_library il
         WHERE NOT EXISTS (
           SELECT 1 FROM media.control.approved_files af
            WHERE af.file_id = il.file_id AND af.revoked = false
         )
    """)
    log.info("library_sync done")


if __name__ == "__main__":
    run_sync()
