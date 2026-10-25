"""Every table a pipeline writes to (saveAsTable / MERGE INTO / insertInto)
must have a CREATE TABLE in governance/*.sql. Implicitly-created tables get
an inferred schema with no PK/constraints/grants — the class of gap that
left email_intel_quarantine and email_ingest_state ungoverned.
"""
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
_PIPE = _ROOT / "pipelines"
_GOV = _ROOT / "governance"

# Views/temp views and staging names are not persisted tables.
_WRITE_RE = re.compile(
    r'(?:saveAsTable\(\s*["\']|insertInto\(\s*["\']|MERGE\s+INTO\s+)'
    r'(media\.[a-z_]+\.[a-z_]+)',
    re.IGNORECASE,
)


def _governed_tables() -> set[str]:
    ddl = "\n".join(p.read_text(encoding="utf-8") for p in _GOV.glob("*.sql"))
    return set(re.findall(r"CREATE TABLE IF NOT EXISTS\s+(media\.[a-z_]+\.[a-z_]+)",
                          ddl, re.IGNORECASE))


def _pipeline_targets() -> dict[str, str]:
    targets: dict[str, str] = {}
    for py in _PIPE.glob("*.py"):
        for m in _WRITE_RE.finditer(py.read_text(encoding="utf-8")):
            targets[m.group(1)] = py.name
    return targets


def test_all_pipeline_write_targets_have_ddl():
    governed = _governed_tables()
    missing = {t: f for t, f in _pipeline_targets().items() if t not in governed}
    assert not missing, (
        "pipeline write targets with no CREATE TABLE in governance/*.sql: "
        + ", ".join(f"{t} ({f})" for t, f in sorted(missing.items()))
    )
