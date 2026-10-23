"""Invariant: pipelines/email_ingest.py received_at is never null.

media.alex.emails.received_at is TIMESTAMP NOT NULL and the partition
column (PARTITIONED BY date(received_at)). A null crashes the write / lands
in a null partition, so the ingester must supply a fallback and declare the
field non-nullable. Read statically (module imports pyspark).
"""
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent.parent / "pipelines" / "email_ingest.py"


def _src() -> str:
    return _SRC.read_text(encoding="utf-8")


def test_received_at_field_is_non_nullable():
    src = _src()
    assert 'StructField("received_at", TimestampType(), False)' in src
    assert 'StructField("received_at", TimestampType(), True)' not in src


def test_received_at_has_non_null_fallback():
    src = _src()
    assert "def _received_at(" in src
    assert '"received_at": _received_at(msg)' in src
    # The fallback must produce a datetime, not None.
    assert "dt.datetime.now(dt.timezone.utc)" in src
