"""Invariants for pipelines/email_intel_job.py write path.

The media.alex.email_intel DDL (app_schema.sql) types the nested
commitments/asks_of_user `when_due` field as TIMESTAMP and created_at/
updated_at as NOT NULL TIMESTAMP. The pipeline previously wrote when_due as
a raw string and never set created_at/updated_at — the latter crashed
createDataFrame on every run with emails. Read statically (imports pyspark).
"""
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent.parent / "pipelines" / "email_intel_job.py"


def _src() -> str:
    return _SRC.read_text(encoding="utf-8")


def test_when_due_is_timestamp_type():
    src = _src()
    assert 'StructField("when_due", TimestampType()' in src
    assert 'StructField("when_due", StringType()' not in src


def test_when_due_is_parsed_not_raw_string():
    src = _src()
    assert '"when_due": _parse_due(item.get("when_due"))' in src
    assert 'def _parse_due(' in src


def test_created_and_updated_at_are_set_on_records():
    src = _src()
    assert '"created_at": now' in src
    assert '"updated_at": now' in src
