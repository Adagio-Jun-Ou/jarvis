"""Invariant: consolidation_queue.for_date is a DATE column (memory_graph.sql),
not a string. Writing it as StringType corrupts the column / partition-style
clustering on the real runtime. Read statically (module imports pyspark).
"""
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent.parent / "pipelines" / "memory_consolidation.py"


def _src() -> str:
    return _SRC.read_text(encoding="utf-8")


def test_for_date_is_date_type():
    src = _src()
    assert 'StructField("for_date", DateType()' in src
    assert 'StructField("for_date", StringType()' not in src


def test_for_date_value_is_date_not_string():
    src = _src()
    assert '"for_date": datetime.utcnow().date()' in src
    assert '"for_date": str(datetime.utcnow().date())' not in src
