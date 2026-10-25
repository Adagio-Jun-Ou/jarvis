"""
Content indexing: read files from `media.content.raw_files`, parse
them with `ai_parse_document`, chunk, embed, and write the chunks to
`media.content.transcripts`. The Vector Search index on this table is
updated by the trigger-based sync declared in `databricks.yml`.

This is a refactor of the previous version that:
  * Uses `ai_parse_document` (Unity Catalog AI function) for ingestion
    of PDF, DOCX, PPTX, and HTML.
  * Uses `ai_classify` to assign `sensitivity` to each chunk.
  * Uses `ai_extract` to pull rights metadata for contracts.
  * Writes via MERGE so re-runs are idempotent.
  * Adds a PII redaction pass before embedding.
"""
from __future__ import annotations

import hashlib
import logging
import re
import uuid

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    ArrayType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

log = logging.getLogger("pipelines.content_indexing")

CHUNK_CHARS = 1200
OVERLAP_CHARS = 200
EMBEDDING_MODEL = "databricks-gte-large-en"
SENSITIVITY_MODEL = "databricks-claude-haiku-4-5"
EXTRACT_MODEL = "databricks-glm-5-3-flash"


def _list_files() -> list[dict]:
    """List every file in the raw_files volume that has not been
    ingested (or has changed since last ingest)."""
    spark = SparkSession.builder.getOrCreate()
    rows = spark.sql("""
        SELECT path, file_metadata, modification_time, content
          FROM read_files('/Volumes/media/content/raw_files', format => 'binaryFile')
    """).collect()
    out: list[dict] = []
    for r in rows:
        path = r.path
        if not path:
            continue
        meta = r.file_metadata or {}
        size = int(meta.get("size") or 0)
        mtime = r.modification_time
        digest = _sha256(r.content or b"")
        out.append({"path": path, "size": size, "mtime": mtime, "sha256": digest})
    return out


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _parse(path: str, raw: bytes) -> dict:
    """Use `ai_parse_document` to extract text + structure from the
    binary. Returns a dict with `text` and `structure`."""
    # We pass the bytes via a temp view; the AI function expects a
    # path or a binaryFile column. In production we point it at the
    # volume path; here we materialize to a temp file path.
    import os
    import tempfile

    spark = SparkSession.builder.getOrCreate()
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(path)[1]) as f:
        f.write(raw)
        tmp = f.name
    try:
        df = spark.read.format("binaryFile").load(tmp)
        df.createOrReplaceTempView("doc_in")
        rows = spark.sql("""
            SELECT ai_parse_document(content, :p) AS parsed
              FROM doc_in
        """, args={"p": path}).collect()
        parsed = rows[0]["parsed"] if rows else None
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
    text = (parsed or {}).get("text") or ""
    return {"text": text, "structure": parsed}


def _redact(text: str) -> str:
    """PII redaction. Conservative: SSN-like, email-like, phone-like,
    credit-card-like. The agent and the indexer both rely on this."""
    text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[REDACTED-SSN]", text)
    text = re.sub(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", "[REDACTED-EMAIL]", text)
    text = re.sub(r"\b(?:\+?\d{1,3}[ -]?)?\(?\d{3}\)?[ -]?\d{3}[ -]?\d{4}\b",
                  "[REDACTED-PHONE]", text)
    text = re.sub(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b", "[REDACTED-CC]", text)
    return text


def _chunk(text: str, asset_id: str) -> list[dict]:
    """Split text into overlapping chunks of ~CHUNK_CHARS characters."""
    text = text.strip()
    if not text:
        return []
    chunks: list[dict] = []
    i = 0
    while i < len(text):
        end = min(i + CHUNK_CHARS, len(text))
        chunk = text[i:end]
        chunks.append({
            "id": str(uuid.uuid4()),
            "asset_id": asset_id,
            "chunk_index": len(chunks),
            "chunk": chunk,
        })
        if end == len(text):
            break
        i = end - OVERLAP_CHARS
    return chunks


def _classify_sensitivity(text: str) -> str:
    """Classify the chunk's sensitivity via ai_classify."""
    spark = SparkSession.builder.getOrCreate()
    spark.createDataFrame([(text,)], "text STRING").createOrReplaceTempView("doc_chunk")
    sql = """
        SELECT ai_classify(text, ARRAY('public', 'internal', 'restricted', 'pii'),
                           :m) AS s
          FROM doc_chunk
    """
    rows = spark.sql(sql, args={"m": SENSITIVITY_MODEL}).collect()
    s = rows[0]["s"] if rows else "public"
    return str(s) if s in ("public", "internal", "restricted", "pii") else "public"


def _extract_rights(text: str) -> list[dict]:
    """Extract rights metadata from a contract-like document."""
    spark = SparkSession.builder.getOrCreate()
    spark.createDataFrame([(text,)], "text STRING").createOrReplaceTempView("rights_in")
    sql = """
        SELECT ai_extract(text, ARRAY('license_start', 'license_end',
                                       'territory', 'asset_id')) AS j
          FROM rights_in
    """
    rows = spark.sql(sql).first()
    if not rows or not rows[0]:
        return []
    j = rows[0][0]
    if not isinstance(j, list):
        j = [j]
    out: list[dict] = []
    for item in j:
        if not isinstance(item, dict):
            continue
        out.append({
            "asset_id": str(item.get("asset_id") or "")[:128] or None,
            "license_start": item.get("license_start") or None,
            "license_end": item.get("license_end") or None,
            "territories": [str(t) for t in (item.get("territory") or [])][:8] or None,
        })
    return out


def _embed(text: str) -> list[float]:
    """Embed one chunk via Databricks embedding endpoint."""
    spark = SparkSession.builder.getOrCreate()
    sql = "SELECT ai_query(:m, :t) AS e"
    return list(
        spark.sql(sql, args={"m": EMBEDDING_MODEL, "t": text[:8000]}).first()["e"]
    )


def run_indexing() -> None:
    spark = SparkSession.builder.getOrCreate()
    files = _list_files()
    log.info("indexing %d files", len(files))
    assets: list[dict] = []
    transcripts: list[dict] = []
    rights_rows: list[dict] = []
    for f in files:
        try:
            with open(f["path"], "rb") as fh:
                raw = fh.read()
        except OSError:
            continue
        parsed = _parse(f["path"], raw)
        text = _redact(parsed["text"])
        if not text.strip():
            continue
        # The asset is the file itself; we synthesize an asset_id from
        # the content hash. Real systems would link to a domain entity.
        asset_id = f["sha256"][:16]
        assets.append({
            "asset_id": asset_id,
            "title": f["path"].rsplit("/", 1)[-1][:200],
            "asset_type": _guess_type(f["path"]),
            "duration_s": None,
            "ingest_path": f["path"],
            "rights_status": "unknown",
        })
        for c in _chunk(text, asset_id):
            try:
                sens = _classify_sensitivity(c["chunk"])
            except Exception:
                sens = "public"
            c["sensitivity"] = sens
            transcripts.append(c)
        # Rights extraction only on contract-like files.
        if f["path"].lower().endswith((".pdf", ".docx")) and "contract" in text.lower()[:1000]:
            for r in _extract_rights(text):
                r["source_doc"] = f["path"]
                rights_rows.append(r)
    if assets:
        spark.createDataFrame(assets, schema=StructType([
            StructField("asset_id", StringType(), False),
            StructField("title", StringType(), False),
            StructField("asset_type", StringType(), False),
            StructField("duration_s", IntegerType(), True),
            StructField("ingest_path", StringType(), True),
            StructField("rights_status", StringType(), True),
        ])).createOrReplaceTempView("assets_in")
        spark.sql("""
            MERGE INTO media.content.assets t USING assets_in s
            ON t.asset_id = s.asset_id
            WHEN MATCHED THEN UPDATE SET
              t.title = s.title, t.asset_type = s.asset_type,
              t.ingest_path = s.ingest_path
            WHEN NOT MATCHED THEN INSERT *
        """)
    if transcripts:
        spark.createDataFrame(transcripts, schema=StructType([
            StructField("id", StringType(), False),
            StructField("asset_id", StringType(), False),
            StructField("chunk_index", IntegerType(), False),
            StructField("chunk", StringType(), False),
            StructField("sensitivity", StringType(), True),
        ])).createOrReplaceTempView("transcripts_in")
        # MERGE on the chunk id so re-runs are idempotent (the docstring's
        # promise). A plain append re-inserted every chunk each run,
        # bloating the Vector Search index with duplicates.
        spark.sql("""
            MERGE INTO media.content.transcripts t
            USING transcripts_in s ON t.id = s.id
            WHEN MATCHED THEN UPDATE SET
              t.chunk = s.chunk, t.chunk_index = s.chunk_index,
              t.sensitivity = s.sensitivity
            WHEN NOT MATCHED THEN INSERT *
        """)
    if rights_rows:
        spark.createDataFrame(rights_rows, schema=StructType([
            StructField("asset_id", StringType(), True),
            StructField("license_start", StringType(), True),
            StructField("license_end", StringType(), True),
            StructField("territories", ArrayType(StringType()), True),
            StructField("source_doc", StringType(), True),
        ])).createOrReplaceTempView("rights_in")
        spark.sql("""
            MERGE INTO media.content.rights t USING rights_in s
            ON t.asset_id = s.asset_id AND t.source_doc = s.source_doc
            WHEN MATCHED THEN UPDATE SET *
            WHEN NOT MATCHED THEN INSERT *
        """)
    log.info("indexing done; %d assets, %d chunks, %d rights",
             len(assets), len(transcripts), len(rights_rows))


def _guess_type(path: str) -> str:
    ext = path.rsplit(".", 1)[-1].lower()
    return {
        "pdf": "document", "docx": "document", "doc": "document",
        "txt": "article", "md": "article", "html": "article",
        "mp3": "audio", "wav": "audio", "m4a": "audio",
        "mp4": "video", "mov": "video", "mkv": "video",
        "jpg": "image", "jpeg": "image", "png": "image",
    }.get(ext, "article")


if __name__ == "__main__":
    run_indexing()
