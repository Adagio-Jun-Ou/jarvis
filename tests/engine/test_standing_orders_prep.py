"""Invariants for the standing_orders_prep write path.

The brief writer must match the control_plane.sql DDL:
  brief_date DATE, consumed BOOLEAN NOT NULL.
An earlier version wrote both as strings (StringType / "false"), which
corrupts those columns on the real Databricks runtime. It also had a dead,
schema-invalid first createDataFrame and used mode("append"), duplicating
briefs on any retry. These static checks lock the fixes in place.

The module imports pyspark at top level, so we read the source statically.
"""
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent.parent / "pipelines" / "standing_orders_prep.py"


def _src() -> str:
    return _SRC.read_text(encoding="utf-8")


def test_consumed_is_boolean_not_string():
    src = _src()
    assert '"consumed": False' in src, "consumed must be a real bool, not a string"
    assert '"consumed": "false"' not in src
    assert 'StructField("consumed", BooleanType()' in src


def test_brief_date_is_date_type():
    src = _src()
    assert 'StructField("brief_date", DateType()' in src
    assert 'StructField("brief_date", StringType()' not in src


def test_uses_merge_not_blind_append():
    src = _src()
    assert "MERGE INTO media.control.standing_order_briefs" in src
    assert 'mode("append").saveAsTable("media.control.standing_order_briefs")' not in src


def test_no_duplicate_dead_createdataframe():
    # There must be exactly one createDataFrame in the write path now.
    assert _src().count("spark.createDataFrame(") == 1
