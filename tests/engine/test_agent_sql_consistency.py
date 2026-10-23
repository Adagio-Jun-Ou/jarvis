"""Guard: agent-layer SQL must match the governance DDL.

The schema-consistency guard covered app.py SELECTs but NOT agent/*.py, so
the tasks column-drift (create_task writing task_id/assignee — columns absent
from the media.alex.tasks DDL) slipped through and would have failed in
production. This guard extracts INSERT/UPDATE/simple-SELECT column sets from
every agent/*.py module (via the AST, which folds Python's adjacent
string-literal concatenation into one constant) and asserts each column
exists in the target table's governance DDL.
"""
import ast
import re
from pathlib import Path

from tests.engine.test_schema_consistency import _ddl_columns, _ddl_required_columns

_ROOT = Path(__file__).resolve().parent.parent.parent
_AGENT = _ROOT / "agent"

_INSERT_RE = re.compile(
    r"INSERT\s+INTO\s+(media\.[a-z_]+\.[a-z_]+)\s*\((?P<cols>[^)]+)\)",
    re.IGNORECASE,
)
_UPDATE_RE = re.compile(
    r"UPDATE\s+(media\.[a-z_]+\.[a-z_]+)\s+SET\s+(?P<body>.+)",
    re.IGNORECASE | re.DOTALL,
)
_SELECT_RE = re.compile(
    r"SELECT\s+(?P<cols>.+?)\s+FROM\s+(media\.[a-z_]+\.[a-z_]+)(?P<after>\s+WHERE\b|\s*$|\s+ORDER\b|\s+LIMIT\b)",
    re.IGNORECASE | re.DOTALL,
)


def _string_constants(src: str) -> list[str]:
    out: list[str] = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            out.append(node.value)
    return out


def _simple_cols(cols_raw: str) -> list[str] | None:
    cols = [c.strip() for c in cols_raw.split(",")]
    clean = []
    for c in cols:
        if "(" in c or "." in c or " " in c or c == "*":
            return None  # function / alias / join — out of scope
        if not re.fullmatch(r"[a-z_][a-z0-9_]*", c, re.IGNORECASE):
            return None
        clean.append(c)
    return clean


def _assignment_cols(body: str) -> list[str]:
    # Column names on the left of '=' in a SET or WHERE clause.
    return re.findall(r"([a-z_][a-z0-9_]*)\s*=", body, re.IGNORECASE)


def test_agent_sql_columns_match_ddl():
    ddl = _ddl_columns()
    required = _ddl_required_columns()
    problems: list[str] = []
    for py in sorted(_AGENT.rglob("*.py")):
        src = py.read_text(encoding="utf-8")
        for s in _string_constants(src):
            if "media." not in s:
                continue
            # INSERT column lists.
            for m in _INSERT_RE.finditer(s):
                table = m.group(1).lower()
                cols = _simple_cols(m.group("cols"))
                if cols is None or table not in ddl:
                    continue
                missing = set(cols) - ddl[table]
                if missing:
                    problems.append(f"{py.name} INSERT {table}: unknown cols {sorted(missing)}")
                # NOT-NULL-without-default columns the INSERT drops would
                # fail at write time (e.g. the vector_memory_mirror PK `id`).
                dropped = required.get(table, set()) - set(cols)
                if dropped:
                    problems.append(f"{py.name} INSERT {table}: missing NOT-NULL cols {sorted(dropped)}")
            # UPDATE SET/WHERE column names.
            for m in _UPDATE_RE.finditer(s):
                table = m.group(1).lower()
                if table not in ddl:
                    continue
                cols = set(_assignment_cols(m.group("body")))
                missing = cols - ddl[table]
                if missing:
                    problems.append(f"{py.name} UPDATE {table}: {sorted(missing)}")
            # Simple single-table SELECTs.
            for m in _SELECT_RE.finditer(s):
                table = m.group(2).lower()
                cols = _simple_cols(m.group("cols"))
                if cols is None or table not in ddl:
                    continue
                missing = set(cols) - ddl[table]
                if missing:
                    problems.append(f"{py.name} SELECT {table}: {sorted(missing)}")
    assert not problems, "agent SQL vs governance DDL drift:\n" + "\n".join(sorted(set(problems)))
