"""skill_reflection's MERGE must use an explicit INSERT column list.

The source view supplies only 8 of media.control.skills' 13 columns. With
`INSERT *`, Delta had to backfill the NOT NULL defaults (success_count,
created_at) for the missing columns — version-dependent and fragile. An
explicit column list lets the DDL DEFAULTs apply reliably. Read statically
(the module imports pyspark).
"""
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent.parent / "pipelines" / "skill_reflection.py"


def _src() -> str:
    return _SRC.read_text(encoding="utf-8")


def test_merge_uses_explicit_insert_columns():
    src = _src()
    assert "WHEN NOT MATCHED THEN INSERT\n" in src or "INSERT\n" in src
    assert "INSERT *" not in src
    # The defaulted / nullable columns must NOT be in the insert list.
    insert_region = src[src.index("WHEN NOT MATCHED"):]
    for col in ("success_count", "created_at", "last_used", "ratified_by"):
        assert col not in insert_region, f"{col} should be left to the DDL default/NULL"
    # The provided columns are present.
    for col in ("skill_id", "owner_email", "name", "status"):
        assert col in insert_region
