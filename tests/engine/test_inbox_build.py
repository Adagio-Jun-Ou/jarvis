"""inbox_build must write columns that exist in the inbox / inbox_view DDLs,
and be wired into the email job so the read models actually get populated.
"""
import re
from pathlib import Path

from tests.engine.test_schema_consistency import _ddl_columns

_ROOT = Path(__file__).resolve().parent.parent.parent
_SRC = _ROOT / "pipelines" / "inbox_build.py"
_BUNDLE = _ROOT / "databricks.yml"


def _insert_columns(src: str, table: str) -> set[str]:
    # Grab the INSERT (col, col, ...) list that follows the MERGE INTO <table>.
    idx = src.index(f"MERGE INTO {table} ")
    region = src[idx:]
    m = re.search(r"INSERT\s*\(([^)]+)\)", region)
    assert m, f"no INSERT column list for {table}"
    return {c.strip() for c in m.group(1).split(",")}


def test_inbox_build_columns_match_ddl():
    src = _SRC.read_text(encoding="utf-8")
    ddl = _ddl_columns()
    for table in ("media.alex.inbox", "media.alex.inbox_view"):
        cols = _insert_columns(src, table)
        missing = cols - ddl[table]
        assert not missing, f"{table}: inbox_build writes columns not in DDL: {sorted(missing)}"



def test_task_board_build_columns_match_ddl():
    src = (_ROOT / "pipelines" / "task_board_build.py").read_text(encoding="utf-8")
    ddl = _ddl_columns()
    cols = _insert_columns(src, "media.alex.task_board_view")
    missing = cols - ddl["media.alex.task_board_view"]
    assert not missing, f"task_board_build writes columns not in DDL: {sorted(missing)}"



