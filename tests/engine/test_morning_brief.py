"""Invariant for the morning_brief write path.

`brief_date` is the MERGE join key AND the partition column, and the DDL
(app_schema.sql media.alex.morning_briefs) declares it DATE. Writing it as
a string mis-partitions and breaks the merge on the real runtime. Lock it
to DateType. The module imports pyspark at top level, so read statically.
"""
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent.parent / "pipelines" / "morning_brief.py"


def _src() -> str:
    return _SRC.read_text(encoding="utf-8")


def test_brief_date_is_date_type():
    src = _src()
    assert 'StructField("brief_date", DateType()' in src
    assert 'StructField("brief_date", StringType()' not in src


def test_brief_date_value_is_date_not_string():
    src = _src()
    assert '"brief_date": datetime.utcnow().date()' in src
    assert '"brief_date": str(datetime.utcnow().date())' not in src


def test_still_uses_merge():
    assert "MERGE INTO media.alex.morning_briefs" in _src()
