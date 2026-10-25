"""Guard against schema drift across DDL ↔ dev-seed ↔ app SELECT.

Recurring bug class: an app.py endpoint SELECTs a column that isn't in the
table's dev-seed registration (which mirrors the governance DDL), so the
query returns wrong/missing data in dev — e.g. /api/transparency/actions
selecting `outcome` that the seed didn't register, or /api/briefs selecting
a `title` column that doesn't exist in the DDL.

This test parses the *simple* single-table SELECTs in app.py (no JOIN, no
table alias, bare columns only — the shape where this class actually bites)
and asserts every selected column exists in the dev-seed's registered
columns for that table.
"""
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
_APP = _ROOT / "app" / "app.py"
_GOV = _ROOT / "governance"


def _ddl_columns() -> dict[str, set[str]]:
    """Parse CREATE TABLE column names per media.<schema>.<table> from the
    governance DDL. Only the column definitions are taken (lines starting
    with an identifier + type), not CONSTRAINT/PRIMARY KEY lines."""
    out: dict[str, set[str]] = {}
    ddl = "\n".join(p.read_text(encoding="utf-8") for p in _GOV.glob("*.sql"))
    for m in re.finditer(
        r"CREATE TABLE IF NOT EXISTS\s+(media\.[a-z_]+\.[a-z_]+)\s*\((?P<body>.*?)\n\)",
        ddl, re.IGNORECASE | re.DOTALL,
    ):
        cols: set[str] = set()
        for line in m.group("body").splitlines():
            line = line.strip().rstrip(",")
            if not line or line.upper().startswith("CONSTRAINT"):
                continue
            cm = re.match(r"`?([a-z_][a-z0-9_]*)`?\s+[A-Za-z]", line)
            if cm:
                cols.add(cm.group(1))
        out[m.group(1).lower()] = cols
    return out


def _ddl_required_columns() -> dict[str, set[str]]:
    """Columns that are NOT NULL and have NO DEFAULT — i.e. an INSERT that
    omits them fails. Used to catch INSERTs that drop a required PK/column."""
    out: dict[str, set[str]] = {}
    ddl = "\n".join(p.read_text(encoding="utf-8") for p in _GOV.glob("*.sql"))
    for m in re.finditer(
        r"CREATE TABLE IF NOT EXISTS\s+(media\.[a-z_]+\.[a-z_]+)\s*\((?P<body>.*?)\n\)",
        ddl, re.IGNORECASE | re.DOTALL,
    ):
        req: set[str] = set()
        for line in m.group("body").splitlines():
            line = line.strip().rstrip(",")
            if not line or line.upper().startswith("CONSTRAINT"):
                continue
            cm = re.match(r"`?([a-z_][a-z0-9_]*)`?\s+[A-Za-z]", line)
            if not cm:
                continue
            up = line.upper()
            if "NOT NULL" in up and "DEFAULT" not in up:
                req.add(cm.group(1))
        out[m.group(1).lower()] = req
    return out

# SELECT <cols> FROM [`]media.schema.table[`] — capture cols + table, stop at
# the first FROM. DOTALL so multi-line column lists match.
_SELECT_RE = re.compile(
    r"SELECT\s+(?P<cols>.+?)\s+FROM\s+`?(?P<table>media\.[a-z_]+\.[a-z_]+)`?",
    re.IGNORECASE | re.DOTALL,
)


def _seeded_columns() -> dict[str, set[str]]:
    from alex.testing import reset_fakes
    from app._dev_seed import seed_dev_warehouse

    fakes = reset_fakes()
    wh = fakes.warehouse
    seed_dev_warehouse(wh)
    return {name: set(t.columns) for name, t in wh._tables.items()}


def _is_simple_column(col: str) -> bool:
    col = col.strip()
    if not col or col == "*":
        return False
    if "(" in col or "." in col or " " in col:  # functions, aliases, alias.col
        return False
    return bool(re.fullmatch(r"[a-z_][a-z0-9_]*", col, re.IGNORECASE))


def test_app_selects_match_dev_seed_columns():
    src = _APP.read_text(encoding="utf-8")
    seeded = _seeded_columns()
    problems: list[str] = []
    for m in _SELECT_RE.finditer(src):
        cols_raw = m.group("cols")
        table = m.group("table").lower()
        # Skip joined/aliased statements: a following token like "t" / "il"
        # alias or a JOIN means the columns may be qualified — out of scope.
        # Detect by presence of an alias right after the table name.
        cols = [c for c in cols_raw.split(",")]
        if not all(_is_simple_column(c) for c in cols):
            continue  # complex SELECT (functions/aliases/joins) — skip
        if table not in seeded:
            continue  # table not registered in dev — separate concern
        missing = {c.strip() for c in cols} - seeded[table]
        if missing:
            problems.append(f"{table}: SELECT columns not in dev-seed: {sorted(missing)}")
    assert not problems, "schema drift (app SELECT vs dev-seed):\n" + "\n".join(problems)


def test_dev_seed_mirrors_the_governance_ddl_exactly():
    """The direct comparison: for every table registered in the dev seed
    that also has production DDL, the column SETS must be identical.

    Either direction of drift is a dev/prod split-brain: a seed-only
    column is something dev code can write that production will reject;
    a DDL-only column is something production has that dev queries can
    never exercise. 19 tables had drifted before this test existed.
    """
    ddl = _ddl_columns()
    seeded = _seeded_columns()
    problems: list[str] = []
    for table in sorted(set(ddl) & set(seeded)):
        seed_only = seeded[table] - ddl[table]
        ddl_only = ddl[table] - seeded[table]
        if seed_only:
            problems.append(f"{table}: seed-only columns {sorted(seed_only)}")
        if ddl_only:
            problems.append(f"{table}: DDL-only columns {sorted(ddl_only)}")
    assert not problems, (
        "dev seed and governance DDL disagree:\n" + "\n".join(problems)
    )


def test_every_seeded_table_has_production_ddl():
    """A registered dev table with no CREATE anywhere is a table the
    system can only ever have in dev — the exact shape the task board
    originally shipped in. The one allowed exception is
    effective_employee_acl, which is a VIEW in production."""
    ddl = _ddl_columns()
    seeded = _seeded_columns()
    allowed_views = {"media.control.effective_employee_acl"}
    orphans = set(seeded) - set(ddl) - allowed_views
    assert not orphans, (
        f"dev-only tables with no production DDL: {sorted(orphans)}"
    )


def test_no_sql_join_on_the_dev_warehouse_path():
    """The in-memory warehouse does not implement JOIN — it answers one
    with rows of Nones, which is silent wrongness rather than an error.
    Every app-route query therefore joins in Python (two reads, dict
    lookup), identical on both backends. A JOIN that sneaks back in
    ships a panel that quietly shows nothing in dev."""
    src = _APP.read_text(encoding="utf-8")
    problems = []
    for m in re.finditer(r'"""(.*?)"""', src, re.DOTALL):
        body = m.group(1)
        if re.search(r"\bJOIN\b", body) and re.search(r"\bFROM\s+media\.", body):
            line = src[: m.start()].count("\n") + 1
            problems.append(f"app.py:{line}: SQL JOIN on the dev path")
    assert not problems, "\n".join(problems)


def test_app_selects_match_governance_ddl():
    """Same simple-SELECT check, but against the governance DDL — catches the
    case where the dev-seed and DDL disagree (e.g. inbox_view lacked
    from_email in the DDL while the seed had it)."""
    src = _APP.read_text(encoding="utf-8")
    ddl = _ddl_columns()
    problems: list[str] = []
    for m in _SELECT_RE.finditer(src):
        cols_raw = m.group("cols")
        table = m.group("table").lower()
        cols = [c for c in cols_raw.split(",")]
        if not all(_is_simple_column(c) for c in cols):
            continue
        if table not in ddl:
            continue  # table has no DDL (separate governance gap)
        missing = {c.strip() for c in cols} - ddl[table]
        if missing:
            problems.append(f"{table}: SELECT columns not in DDL: {sorted(missing)}")
    assert not problems, "schema drift (app SELECT vs governance DDL):\n" + "\n".join(problems)
