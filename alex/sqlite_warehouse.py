"""SqliteWarehouse — the durable local store.

The dev warehouse was an ~800-line hand-rolled SQL parser over dicts:
no persistence (every task, session, run, and memory row vanished on
restart), no JOIN, one process-global lock serializing every read. On a
personal machine the right answer is the stdlib: SQLite in WAL mode,
one file, real SQL, durable across restarts, concurrent readers.

Same surface as :class:`alex.warehouse.InMemoryWarehouse` — ``register``
(CREATE TABLE IF NOT EXISTS + seed rows only when the table is empty),
``execute(statement, params) -> StatementResult``, ``drop`` — so every
seam that reaches ``get_fakes().warehouse`` works unchanged. The tiny
Spark dialect the code writes is translated here in one place:

* ``media.control.task_runs`` → ``media_control_task_runs``
* ``current_timestamp()`` / ``now()`` → ``CURRENT_TIMESTAMP``
* ``current_timestamp() ± INTERVAL n UNIT`` → ``datetime('now', '±n units')``
* backtick identifiers → plain
* list/dict params → JSON text; datetimes → ISO text

A missing table surfaces as :class:`alex.errors.NotFound`, which callers
already treat as "apply the fallback" rather than "crash".
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import threading
from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import Any

from alex.errors import BadRequest, NotFound
from alex.warehouse import StatementResult

log = logging.getLogger("alex.sqlite_warehouse")

_IDENT_RE = re.compile(r"\bmedia\.(\w+)\.(\w+)\b")
_INTERVAL_RE = re.compile(
    r"(?:current_timestamp\(\)|CURRENT_TIMESTAMP|now\(\))\s*([+-])\s*"
    r"INTERVAL\s+(\d+)\s+(\w+)", re.IGNORECASE,
)
_NOW_RE = re.compile(r"\b(?:current_timestamp\(\)|now\(\))", re.IGNORECASE)
_UNITS = {
    "second": "seconds", "seconds": "seconds", "minute": "minutes",
    "minutes": "minutes", "hour": "hours", "hours": "hours",
    "day": "days", "days": "days",
}


def ident(name: str) -> str:
    """`media.control.task_runs` → `media_control_task_runs`."""
    return _IDENT_RE.sub(lambda m: f"media_{m.group(1)}_{m.group(2)}", name)


def translate(statement: str) -> str:
    s = " ".join(statement.split()).replace("`", "")
    s = _IDENT_RE.sub(lambda m: f"media_{m.group(1)}_{m.group(2)}", s)

    def _iv(m: re.Match) -> str:
        unit = _UNITS.get(m.group(3).lower())
        if unit is None:
            raise BadRequest(f"unsupported INTERVAL unit: {m.group(3)}")
        return f"datetime('now', '{m.group(1)}{m.group(2)} {unit}')"

    s = _INTERVAL_RE.sub(_iv, s)
    s = _NOW_RE.sub("CURRENT_TIMESTAMP", s)
    return s


def _coerce(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, default=str)
    if isinstance(value, datetime):
        return value.isoformat(sep=" ")
    return value


class SqliteWarehouse:
    """Durable, file-backed, same surface as the in-memory engine."""

    def __init__(self, path: str | None = None) -> None:
        self.path = path or os.environ.get("ALEX_DB_PATH") or ":memory:"
        if self.path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._registered: set[str] = set()

    # ---- registry (the seed's surface) -----------------------------------
    def register(
        self,
        name: str,
        columns: Sequence[str],
        *,
        pk: Sequence[str] | None = None,
        row_filter: Any | None = None,
        rows: Iterable[tuple] | None = None,
    ) -> None:
        tbl = ident(name)
        cols = ", ".join(f'"{c}"' for c in columns)
        pk_sql = f', PRIMARY KEY ({", ".join(chr(34) + c + chr(34) for c in pk)})' if pk else ""
        with self._lock:
            self._conn.execute(f'CREATE TABLE IF NOT EXISTS "{tbl}" ({cols}{pk_sql})')
            self._registered.add(tbl)
            seed = list(rows or [])
            if seed:
                (n,) = self._conn.execute(f'SELECT count(*) FROM "{tbl}"').fetchone()
                if n == 0:
                    marks = ", ".join("?" for _ in columns)
                    self._conn.executemany(
                        f'INSERT OR IGNORE INTO "{tbl}" VALUES ({marks})',
                        [tuple(_coerce(v) for v in r) for r in seed],
                    )
            self._conn.commit()

    def drop(self, name: str) -> None:
        with self._lock:
            self._conn.execute(f'DROP TABLE IF EXISTS "{ident(name)}"')
            self._conn.commit()

    # ---- the contract ----------------------------------------------------
    def execute(self, statement: str, parameters: dict | None = None) -> StatementResult:
        sql = translate(statement)
        params = {k: _coerce(v) for k, v in (parameters or {}).items()}
        with self._lock:
            try:
                cur = self._conn.execute(sql, params)
            except sqlite3.OperationalError as e:
                msg = str(e)
                if "no such table" in msg:
                    raise NotFound(msg) from e
                raise BadRequest(f"{msg} :: {sql[:160]}") from e
            except sqlite3.IntegrityError as e:
                self._conn.rollback()
                raise BadRequest(str(e)) from e
            if cur.description:
                cols = [d[0] for d in cur.description]
                rows = [tuple(r) for r in cur.fetchall()]
                return StatementResult(columns=cols, rows=rows, row_count=len(rows))
            self._conn.commit()
            return StatementResult(columns=[], rows=[], row_count=cur.rowcount)

    def close(self) -> None:
        with self._lock:
            self._conn.close()
