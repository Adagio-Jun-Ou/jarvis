"""
Smoke test: every SQL passed to `spark.sql(...)` or `_spark().sql(...)`
or a similar executor must be a plain string (not an f-string) so user
data is never interpolated into the query.

The check is targeted: we look for SQL-execution call sites and verify
their string argument is a plain literal (or a parameterized call), not
an f-string. Comments and docstrings are ignored.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOTS = [
    Path(__file__).resolve().parents[2] / "pipelines",
    Path(__file__).resolve().parents[2] / "workflows",
]

# Functions that execute SQL.
SQL_EXECUTORS = frozenset({
    "spark.sql", "cursor.execute",
    "_spark",
    # Allow test helpers and the warehouse wrapper itself.
})


def _has_fstring_argument(call: ast.Call) -> bool:
    """True if the first positional argument to the call is a JoinedStr
    (an f-string)."""
    if not call.args:
        return False
    first = call.args[0]
    return isinstance(first, ast.JoinedStr)


def _check_file(path: Path) -> list[str]:
    """Return a list of `f"{path}:{lineno}: f-string in {call}"` issues."""
    text = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError:
        return [f"{path.name}: syntax error"]
    issues: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # Resolve the callee name
        callee = None
        if isinstance(node.func, ast.Attribute):
            callee = node.func.attr
        elif isinstance(node.func, ast.Name):
            callee = node.func.id
        if callee is None:
            continue
        if callee not in ("sql", "execute", "filter", "where"):
            continue
        if not _has_fstring_argument(node):
            continue
        # The call is `xxx.sql(f"...")`, `xxx.execute(f"...")`, or a
        # DataFrame `.filter(f"...")` / `.where(f"...")`. That
        # means the SQL string has a `{...}` placeholder. The
        # convention is to use `:name` parameters and pass them via
        # `parameters=...`. An f-string bypasses the engine's
        # parameter binding.
        issues.append(
            f"{path.name}:{node.lineno}: f-string in {callee}() — "
            f"use :name parameters / column expressions, never interpolation"
        )
    return issues


@pytest.mark.parametrize("path", sorted(p for r in ROOTS for p in r.glob("*.py")))
def test_sql_callers_use_parameter_binding(path):
    issues = _check_file(path)
    assert not issues, "\n".join(issues)
