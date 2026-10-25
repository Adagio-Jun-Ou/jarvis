"""Invariant: agent_review_job writes the `ts` column it later reads.

media.ops.agent_reviews.ts is NOT NULL, the CLUSTER BY column, and the
field _check_regression() filters on (WHERE ts >= ...). Omitting it from the
written DataFrame relied on a fragile column-default backfill. Read the
source statically (the module imports pyspark).
"""
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent.parent / "pipelines" / "agent_review_job.py"


def _src() -> str:
    return _SRC.read_text(encoding="utf-8")


def test_ts_is_written_and_typed():
    src = _src()
    assert '"ts": datetime.utcnow()' in src
    assert 'StructField("ts", TimestampType(), False)' in src
