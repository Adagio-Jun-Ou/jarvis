"""
SQL helpers shared by agent, app, pipelines, and tests.

We separate "how do I talk to a warehouse" from "what query do I run" so
that every other module gets a single, well-tested surface. The surface has
two implementations:

  * `WarehouseClient` — the production implementation, thin wrapper around
    `databricks.sdk.WorkspaceClient.statement_execution.execute_statement`.
    All values pass through the SDK's `parameters=` channel; no f-string
    SQL anywhere in the codebase.

  * `InMemoryWarehouse` — the dev/test implementation. Stores tables as
    dicts of lists, executes a small subset of SQL (parameterized SELECT,
    INSERT, UPDATE, MERGE-ON-MATCH, DELETE), and supports a row-level
    policy hook so tests can simulate ACL behaviour.

The split is intentional: there is no "mode" flag inside the production
code. Tests and dev get a different binary code path; production gets
the real one. The shared `execute` signature is the contract.
"""

from __future__ import annotations

import logging
import re
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from alex.config import get_settings
from alex.errors import BadRequest, NotFound, UpstreamError

log = logging.getLogger("alex.warehouse")

# Public, well-known OBO scope values (Databricks Apps).
SCOPE_SERVING = "serving.serving-endpoints"
SCOPE_SQL = "sql"
SCOPE_FILES = "files.files"
SCOPE_GENIE = "dashboards.genie"


@dataclass
class StatementResult:
    columns: list[str]
    rows: list[tuple]
    row_count: int = 0
    statement_id: str = ""

    def as_dicts(self) -> list[dict]:
        return [dict(zip(self.columns, r, strict=False)) for r in self.rows]


class WarehouseClient:
    """Thin wrapper over the Databricks SQL warehouse with parameter binding."""

    def __init__(
        self,
        sdk_client=None,
        warehouse_id: str | None = None,
        *,
        wait_timeout_seconds: int = 60,
    ) -> None:
        settings = get_settings()
        if sdk_client is None:
            from databricks.sdk import WorkspaceClient

            sdk_client = WorkspaceClient()
        self._sdk = sdk_client
        self._warehouse_id = warehouse_id or settings.warehouse_id
        self._wait = wait_timeout_seconds

    def execute(
        self,
        statement: str,
        parameters: dict | None = None,
        *,
        catalog: str | None = None,
        schema: str | None = None,
        row_limit: int = 10000,
    ) -> StatementResult:
        if not self._warehouse_id:
            raise UpstreamError("WAREHOUSE_ID is not configured")
        params = self._normalize_params(parameters or {})
        kwargs = {
            "warehouse_id": self._warehouse_id,
            "statement": statement,
            "parameters": params,
            "wait_timeout": f"{self._wait}s",
        }
        if catalog:
            kwargs["catalog"] = catalog
        if schema:
            kwargs["schema"] = schema
        try:
            resp = self._sdk.statement_execution.execute_statement(**kwargs)
        except Exception as e:  # pragma: no cover (network)
            raise UpstreamError(f"warehouse execute failed: {e}") from e
        status = getattr(resp.status, "state", None) if resp.status else None
        if status and str(status).endswith("FAILED"):
            msg = getattr(resp.status, "message", "unknown")
            raise UpstreamError(f"warehouse FAILED: {msg}")
        cols = [
            c.name
            for c in (
                resp.manifest.schema.columns
                if resp.manifest and resp.manifest.schema
                else []
            )
        ]
        rows = (
            list(resp.result.data_array)
            if resp.result and resp.result.data_array
            else []
        )
        return StatementResult(
            columns=cols,
            rows=[tuple(r) for r in rows[:row_limit]],
            row_count=len(rows),
            statement_id=resp.statement_id or "",
        )

    @staticmethod
    def _normalize_params(params: dict) -> dict:
        """Coerce Python values to the Spark-style parameter dict that the SDK accepts.

        Spark SQL parameter binding supports: STRING, INT, BIGINT, DOUBLE, BOOLEAN,
        DATE, TIMESTAMP, DECIMAL, NULL. We use the `type` field in the dict."""
        out: dict = {}
        for k, v in params.items():
            if v is None:
                out[k] = {"value": None, "type": "NULL"}
            elif isinstance(v, bool):
                out[k] = {"value": "true" if v else "false", "type": "BOOLEAN"}
            elif isinstance(v, int):
                out[k] = {"value": v, "type": "BIGINT"}
            elif isinstance(v, float):
                out[k] = {"value": v, "type": "DOUBLE"}
            elif isinstance(v, (list, tuple)):
                # Use the array form: {value: [...], type: "ARRAY<T>"} for strings, else per-element
                elements = [WarehouseClient._normalize_params({"": e})[""] for e in v]
                out[k] = {"value": [e["value"] for e in elements], "type": "ARRAY"}
            else:
                out[k] = {"value": str(v), "type": "STRING"}
        return out


# ---------------------------------------------------------------------------
# Dev / test implementation
# ---------------------------------------------------------------------------


# Tables and policies registered by the dev harness. Tests use these
# directly; the agent code never reaches into them.
@dataclass
class _Table:
    name: str
    columns: list[str]
    rows: list[tuple] = field(default_factory=list)
    pk: list[str] | None = None
    # Optional row filter: callable(row_dict) -> bool
    row_filter: Any | None = None


class InMemoryWarehouse:
    """A tiny SQL-ish engine for dev and tests.

    Supports a deliberately small subset of SQL:
      * SELECT col, col FROM table [WHERE col = :p] [ORDER BY col] [LIMIT n]
      * INSERT INTO table [(cols)] VALUES (...), (...)
      * UPDATE table SET col = :p [, col = :p] [WHERE col = :p]
      * DELETE FROM table [WHERE col = :p]
      * MERGE INTO target USING src ON cond WHEN MATCHED THEN UPDATE ...
      * CREATE TABLE IF NOT EXISTS name (col type, ...)

    This is **not** a database. It exists to make the agent code
    testable without a real Databricks warehouse. The shape of the
    public methods matches `WarehouseClient` so the agent can be
    parameterised on either.
    """

    _lock = threading.RLock()

    def __init__(self) -> None:
        self._tables: dict[str, _Table] = {}

    # ---- registry --------------------------------------------------------
    def register(
        self,
        name: str,
        columns: Sequence[str],
        *,
        pk: Sequence[str] | None = None,
        row_filter: Any | None = None,
        rows: Iterable[tuple] | None = None,
    ) -> None:
        with self._lock:
            self._tables[name.lower()] = _Table(
                name=name,
                columns=list(columns),
                pk=list(pk) if pk else None,
                row_filter=row_filter,
                rows=list(rows or []),
            )

    def drop(self, name: str) -> None:
        with self._lock:
            self._tables.pop(name.lower(), None)

    def table(self, name: str) -> _Table:
        with self._lock:
            t = self._tables.get(name.lower())
            if t is None:
                raise NotFound(f"no such table: {name}")
            return t

    # ---- public ----------------------------------------------------------
    def execute(
        self, statement: str, parameters: dict | None = None
    ) -> StatementResult:
        """Serialize ALL statements behind the table lock.

        Mutations mutate ``self._tables`` in place, and an UPDATE
        iterating one row-list while a DELETE reassigns it is an
        interleaving the threaded dev FastAPI app actually produces —
        SELECTs were safe to run concurrently but the compound
        read-decide-write in UPDATE/DELETE/MERGE was not.
        """
        with self._lock:
            return self._execute_unlocked(statement, parameters)

    def _execute_unlocked(
        self, statement: str, parameters: dict | None = None
    ) -> StatementResult:
        params = parameters or {}
        s = " ".join(statement.split())  # normalize whitespace
        try:
            if re.match(r"(?i)^SELECT\b", s):
                cols, rows = self._do_select(s, params)
                return StatementResult(columns=cols, rows=rows, row_count=len(rows))
            if re.match(r"(?i)^INSERT\s+INTO\b", s):
                self._do_insert(s, params)
                return StatementResult(columns=[], rows=[], row_count=0)
            if re.match(r"(?i)^UPDATE\b", s):
                n = self._do_update(s, params)
                return StatementResult(columns=[], rows=[], row_count=n)
            if re.match(r"(?i)^DELETE\s+FROM\b", s):
                n = self._do_delete(s, params)
                return StatementResult(columns=[], rows=[], row_count=n)
            if re.match(r"(?i)^MERGE\s+INTO\b", s):
                self._do_merge(s, params)
                return StatementResult(columns=[], rows=[], row_count=0)
            if re.match(r"(?i)^CREATE\s+TABLE\b", s):
                self._do_create(s)
                return StatementResult(columns=[], rows=[], row_count=0)
            if re.match(r"(?i)^DROP\s+TABLE\b", s):
                self._do_drop(s)
                return StatementResult(columns=[], rows=[], row_count=0)
        except Exception as e:
            log.exception("InMemoryWarehouse error: %s", e)
            raise
        raise BadRequest(f"unsupported statement: {s[:60]}...")

    # ---- impl ------------------------------------------------------------
    def _resolve_param(self, params: dict, token: str) -> Any:
        # Tokens like :foo, :1, ?
        if token == "?":
            # Position-based — we use a single ordered list under the key "_p".
            seq = params.get("_p")
            if not seq:
                return None
            v = seq.pop(0)
            return v
        name = token.lstrip(":")
        if name in params:
            return params[name]
        return None

    def _parse_where(
        self, where: str, params: dict, columns: list[str]
    ) -> Any | None:
        """Compile a WHERE clause into ``predicate(row) -> bool``.

        Returns ``None`` only when there is no clause at all. A clause
        this grammar cannot express raises :class:`BadRequest` rather
        than returning ``None``, because callers that treat "no
        predicate" as "match everything" would hand back exactly the
        rows the query asked to exclude — on ``user_roles`` or
        ``user_preferences`` that is one user reading another's.

        The grammar deliberately mirrors the subset the production
        statements actually use, so dev exercises the same SQL::

            expr    := term (OR term)*
            term    := factor (AND factor)*
            factor  := '(' expr ')' | comparison
            comparison := col op operand | col IS [NOT] NULL
            op      := = | != | <> | > | < | >= | <=
            operand := :param | ? | 'literal' | number
                     | current_date() | current_timestamp()
        """
        if not where or not where.strip():
            return None
        cols = list(columns)
        tokens = self._tokenize_where(where)
        pos = 0

        def peek() -> str | None:
            return tokens[pos] if pos < len(tokens) else None

        def take() -> str:
            nonlocal pos
            tok = tokens[pos]
            pos += 1
            return tok

        def parse_expr():
            node = parse_term()
            while peek() and peek().upper() == "OR":
                take()
                right = parse_term()
                left = node
                node = lambda r, a=left, b=right: a(r) or b(r)  # noqa: E731
            return node

        def parse_term():
            node = parse_factor()
            while peek() and peek().upper() == "AND":
                take()
                right = parse_factor()
                left = node
                node = lambda r, a=left, b=right: a(r) and b(r)  # noqa: E731
            return node

        def parse_factor():
            tok = peek()
            if tok is None:
                raise BadRequest(f"unexpected end of WHERE clause: {where!r}")
            if tok == "(":
                take()
                node = parse_expr()
                if peek() != ")":
                    raise BadRequest(f"unbalanced parentheses in WHERE: {where!r}")
                take()
                return node
            return parse_comparison()

        def parse_comparison():
            col = take()
            if not re.fullmatch(r"[\w\.]+", col):
                raise BadRequest(f"expected a column name in WHERE, got {col!r}")
            col = col.split(".")[-1]
            op = peek()
            if op is None:
                raise BadRequest(f"expected an operator after {col!r} in WHERE")
            if op.upper() in ("IN", "NOT"):
                negated = False
                if op.upper() == "NOT":
                    take()
                    negated = True
                    if not peek() or peek().upper() != "IN":
                        raise BadRequest(f"expected IN after NOT in WHERE: {where!r}")
                take()  # IN
                if peek() != "(":
                    raise BadRequest(f"expected ( after IN in WHERE: {where!r}")
                take()
                members: list[Any] = []
                while peek() is not None and peek() != ")":
                    tok = take()
                    if tok == ",":
                        continue
                    members.append(self._where_operand(tok, params))
                if peek() != ")":
                    raise BadRequest(f"unterminated IN list in WHERE: {where!r}")
                take()
                wanted = {str(m) for m in members}
                return lambda r, c=col, w=wanted, n=negated: (
                    (str(r.get(c)) not in w) if n else (str(r.get(c)) in w)
                )
            if op.upper() == "IS":
                take()
                negated = False
                if peek() and peek().upper() == "NOT":
                    take()
                    negated = True
                nxt = take()
                if nxt.upper() != "NULL":
                    raise BadRequest(f"expected NULL after IS in WHERE, got {nxt!r}")
                return lambda r, c=col, n=negated: (
                    (r.get(c) is not None) if n else (r.get(c) is None)
                )
            take()
            if op not in ("=", "!=", "<>", ">", "<", ">=", "<="):
                raise BadRequest(f"unsupported operator {op!r} in WHERE")
            operand = take()
            value = self._where_operand(operand, params)
            return lambda r, c=col, o=op, v=value: self._compare(r.get(c), o, v)

        node = parse_expr()
        if pos != len(tokens):
            raise BadRequest(
                f"unsupported WHERE clause: {where.strip()[:120]!r} "
                f"(stopped at {tokens[pos]!r})"
            )

        def pred(row):
            return bool(node(dict(zip(cols, row, strict=False))))

        return pred

    @staticmethod
    def _tokenize_where(where: str) -> list[str]:
        """Split a WHERE clause into tokens, keeping quoted literals whole."""
        pattern = re.compile(
            r"""
            \s*(
                '(?:[^']|'')*'                 # single-quoted literal
              | (?:current_date|current_timestamp|now)\s*\(\s*\)
                (?:\s*[-+]\s*INTERVAL\s+\d+\s+\w+)?   # optional offset
              | :\w+ | \?                      # bound parameter
              | >= | <= | != | <>              # two-char operators
              | [=<>(),]                       # single-char operators / list separator
              | [\w\.]+                        # identifier / number / keyword
            )
            """,
            re.VERBOSE | re.IGNORECASE,
        )
        tokens: list[str] = []
        idx = 0
        while idx < len(where):
            m = pattern.match(where, idx)
            if not m:
                if where[idx].isspace():
                    idx += 1
                    continue
                raise BadRequest(
                    f"unparseable token in WHERE at {where[idx:idx + 20]!r}"
                )
            tokens.append(m.group(1))
            idx = m.end()
        return tokens

    def _where_operand(self, token: str, params: dict) -> Any:
        """Resolve one right-hand operand to a comparable value."""
        low = " ".join(token.lower().split())
        if low.startswith(("current_date", "current_timestamp", "now")):
            from datetime import datetime, timedelta, timezone

            now = datetime.now(timezone.utc)
            offset = re.search(
                r"([-+])\s*interval\s+(\d+)\s+(\w+)", low
            )
            if offset:
                sign = -1 if offset.group(1) == "-" else 1
                amount = int(offset.group(2))
                unit = offset.group(3).rstrip("s")
                per = {
                    "day": timedelta(days=1),
                    "hour": timedelta(hours=1),
                    "minute": timedelta(minutes=1),
                    "second": timedelta(seconds=1),
                    "week": timedelta(weeks=1),
                    "month": timedelta(days=30),
                    "year": timedelta(days=365),
                }.get(unit)
                if per is None:
                    raise BadRequest(f"unsupported INTERVAL unit {unit!r}")
                now = now + sign * amount * per
            return (
                now.date().isoformat()
                if low.startswith("current_date")
                else now.isoformat(sep=" ", timespec="seconds")
            )
        if token.startswith(":") or token == "?":
            return self._resolve_param(params, token)
        if token.startswith("'") and token.endswith("'"):
            return token[1:-1].replace("''", "'")
        try:
            return float(token) if "." in token else int(token)
        except ValueError:
            if token.upper() in ("TRUE", "FALSE"):
                return token.upper() == "TRUE"
            if token.upper() == "NULL":
                return None
            return token

    @staticmethod
    def _compare(left: Any, op: str, right: Any) -> bool:
        """Compare two cell values the way the production engine would.

        NULL never satisfies a comparison — matching SQL, where
        ``col = :p`` is unknown (and therefore false) for a NULL column.
        """
        if left is None or right is None:
            return False
        if isinstance(left, bool) or isinstance(right, bool):
            def as_bool(v):
                if isinstance(v, bool):
                    return v
                return str(v).strip().lower() in ("true", "1", "yes")

            left, right = as_bool(left), as_bool(right)
        else:
            try:
                left, right = float(left), float(right)
            except (TypeError, ValueError):
                left, right = str(left), str(right)
        if op == "=":
            return left == right
        if op in ("!=", "<>"):
            return left != right
        if op == ">":
            return left > right
        if op == "<":
            return left < right
        if op == ">=":
            return left >= right
        if op == "<=":
            return left <= right
        return False

    def _table_columns(self, qual: str) -> tuple[str, list[str]]:
        # `qual` may be `schema.table` or just `table`.
        if "." in qual:
            schema, name = qual.split(".", 1)
        else:
            schema, name = "", qual
        full = f"{schema}.{name}".lower() if schema else name.lower()
        t = self._tables.get(full) or self._tables.get(name.lower())
        if t is None:
            raise NotFound(f"no such table: {qual}")
        return t.name, t.columns

    def _do_select(self, s: str, params: dict) -> tuple[list[str], list[tuple]]:
        m = re.match(
            r"(?is)^SELECT\s+(?P<cols>[\w\.,\s\*]+?)\s+FROM\s+`?(?P<table>[\w\.]+)`?",
            s,
        )
        if not m:
            raise BadRequest(f"cannot parse SELECT: {s[:80]}")
        cols_raw = m.group("cols").strip()
        table = m.group("table")
        full, all_cols = self._table_columns(table)
        if cols_raw == "*":
            select_cols = list(all_cols)
        else:
            select_cols = [c.strip() for c in cols_raw.split(",")]
        # WHERE
        where_m = re.search(r"(?i)\bWHERE\b\s+(?P<w>.+?)(?=\bORDER\b|\bLIMIT\b|\Z)", s)
        where = where_m.group("w").strip() if where_m else None
        # ORDER BY
        order_m = re.search(
            r"(?i)\bORDER\s+BY\s+`?(?P<c>[\w\.]+)`?(?:\s+(?P<d>ASC|DESC))?", s
        )
        # LIMIT
        lim_m = re.search(r"(?i)\bLIMIT\s+(:?\w+|\d+)", s)
        limit = None
        if lim_m:
            tok = lim_m.group(1)
            if tok.startswith(":"):
                limit = int(self._resolve_param(params, tok) or 0)
            else:
                limit = int(tok)
        t = self.table(full)
        rows = list(t.rows)
        if where:
            pred = self._parse_where(where, params, t.columns)
            if pred is None:
                # A WHERE this engine cannot honour must NOT degrade to
                # "return everything" — that hands back precisely the rows
                # the query asked to exclude. On tables like user_roles or
                # user_preferences that is one user reading another's, so
                # it fails loudly here instead of leaking quietly.
                raise BadRequest(
                    f"unsupported WHERE clause: {where.strip()[:100]!r}. "
                    "This engine supports `col = :param` joined by AND; "
                    "apply functions like lower() before binding."
                )
            rows = [r for r in rows if pred(r)]
        if t.row_filter:
            rows = [r for r in rows if t.row_filter(dict(zip(t.columns, r, strict=False)))]
        if order_m:
            oc = order_m.group("c")
            od = (order_m.group("d") or "ASC").upper()
            idx = t.columns.index(oc) if oc in t.columns else None
            if idx is not None:
                rows = sorted(
                    rows, key=lambda r: (r[idx] is None, r[idx]), reverse=(od == "DESC")
                )
        if limit is not None:
            rows = rows[:limit]
        # Project
        out_rows = []
        for r in rows:
            d = dict(zip(t.columns, r, strict=False))
            out_rows.append(tuple(d.get(c) for c in select_cols))
        return select_cols, out_rows

    def _parse_cols_values_lists(
        self, s: str
    ) -> tuple[str, list[str], list[list[str]]]:
        """Parse `INSERT INTO t (a, b) VALUES (:p, :p), (:p, :p)` shape."""
        m = re.match(
            r"(?is)^INSERT\s+INTO\s+`?(?P<t>[\w\.]+)`?\s*\((?P<cols>[^)]+)\)\s*VALUES\s*(?P<rest>.+)$",
            s,
        )
        if not m:
            raise BadRequest(f"cannot parse INSERT: {s[:80]}")
        cols = [c.strip().strip("`") for c in m.group("cols").split(",")]
        rest = m.group("rest").strip().rstrip(";")
        # Split rows on '),('  — but not on '(', ')' inside string params.
        depth = 0
        rows: list[list[str]] = []
        buf = ""
        for ch in rest:
            if ch == "(" and depth == 0:
                depth = 1
                buf = ""
                continue
            if ch == "(":
                depth += 1
                buf += ch
                continue
            if ch == ")" and depth == 1:
                rows.append([v.strip() for v in buf.split(",")])
                depth = 0
                buf = ""
                continue
            if ch == ")":
                depth -= 1
                buf += ch
                continue
            if ch == "," and depth == 1 and not buf.strip():
                # separator between row tuples
                continue
            buf += ch
        return m.group("t"), cols, rows

    _NOW_FNS = ("current_timestamp()", "current_timestamp", "now()")

    def _value_token(self, params: dict, tok: str) -> Any:
        """One value in an INSERT or a SET, however it was written.

        A bound parameter, a quoted string, a number, NULL, or the clock.
        Anything unrecognised raises rather than becoming NULL, which is
        what this used to do: `INSERT ... VALUES (..., 'running', ...)`
        stored NULL for the literal, so a run's status was NULL in dev and
        'running' in production, and no test noticed because nothing read
        the column back. Silently discarding a value the caller wrote is
        the worst available behaviour — worse than refusing it.
        """
        tok = (tok or "").strip()
        if tok.startswith(":") or tok == "?":
            return self._coerce_literal(self._resolve_param(params, tok))
        lowered = tok.lower().rstrip(";").strip()
        if lowered in self._NOW_FNS:
            return self._stamp(datetime.now(timezone.utc))
        if lowered == "current_date()":
            return datetime.now(timezone.utc).date().isoformat()
        if lowered in ("null", ""):
            return None
        if (tok.startswith("'") and tok.endswith("'")) or re.match(
            r"^-?\d+(\.\d+)?$", tok
        ):
            return self._coerce_literal(tok)
        if lowered in ("true", "false"):
            return lowered == "true"
        shifted = self._interval_expr(params, tok)
        if shifted is not None:
            return shifted
        raise BadRequest(f"unsupported value expression: {tok}")

    #: Dev stores timestamps as ISO strings (see app/_dev_seed._ts). The
    #: clock has to produce the same shape as everything else in the
    #: table: a mix of str and datetime in one column sorts by raising
    #: TypeError, which is a strange way to learn about a type mismatch.
    @staticmethod
    def _stamp(when: datetime) -> str:
        return when.isoformat(sep=" ")

    _INTERVAL_UNITS = {
        "second": 1, "seconds": 1, "minute": 60, "minutes": 60,
        "hour": 3600, "hours": 3600, "day": 86400, "days": 86400,
        "week": 604800, "weeks": 604800,
    }

    def _interval_expr(self, params: dict, tok: str) -> str | None:
        """``current_timestamp() ± INTERVAL <n> <unit>``, or None.

        Used for expiry columns — a delegation that lasts N days is
        written this way rather than computed in Python so the *server's*
        clock decides, which is the whole point of doing it in SQL.
        """
        m = re.match(
            r"(?i)^(current_timestamp\(\)|now\(\))\s*(?P<sign>[+-])\s*"
            r"INTERVAL\s+(?P<n>:\w+|\?|-?\d+)\s+(?P<unit>[a-z]+)\s*$",
            tok.strip(),
        )
        if not m:
            return None
        unit = self._INTERVAL_UNITS.get(m.group("unit").lower())
        if unit is None:
            return None
        raw = m.group("n")
        amount = (
            self._resolve_param(params, raw)
            if raw.startswith(":") or raw == "?"
            else raw
        )
        try:
            seconds = float(amount) * unit
        except (TypeError, ValueError):
            return None
        if m.group("sign") == "-":
            seconds = -seconds
        return self._stamp(
            datetime.now(timezone.utc) + timedelta(seconds=seconds)
        )

    def _do_insert(self, s: str, params: dict) -> None:
        # Detect the column-list form first
        if re.match(r"(?is)^INSERT\s+INTO\s+`?[\w\.]+`?\s*\(", s):
            table, cols, rows = self._parse_cols_values_lists(s)
            t = self.table(table)
            for r in rows:
                vals = [self._value_token(params, tok) for tok in r]
                # Map to full column order; missing columns -> None
                full_row = [None] * len(t.columns)
                for i, c in enumerate(cols):
                    if c in t.columns:
                        full_row[t.columns.index(c)] = (
                            vals[i] if i < len(vals) else None
                        )
                t.rows.append(tuple(full_row))
            return
        # INSERT INTO t VALUES (a, b, c)  — no column list
        m = re.match(
            r"(?is)^INSERT\s+INTO\s+`?(?P<t>[\w\.]+)`?\s+VALUES\s*(?P<rest>.+)$",
            s,
        )
        if not m:
            raise BadRequest(f"cannot parse INSERT: {s[:80]}")
        t = self.table(m.group("t"))
        rest = m.group("rest").strip().rstrip(";")
        depth = 0
        rows: list[list[str]] = []
        cur: list[str] = []
        buf = ""
        for ch in rest:
            if ch == "(" and depth == 0:
                depth = 1
                buf = ""
                continue
            if ch == "(":
                depth += 1
                buf += ch
                continue
            if ch == ")" and depth == 1:
                cur.append(buf.strip())
                rows.append(list(cur))
                cur = []
                depth = 0
                buf = ""
                continue
            if ch == ")":
                depth -= 1
                buf += ch
                continue
            if ch == "," and depth == 1 and not buf.strip():
                continue
            if ch == "," and depth == 1:
                cur.append(buf.strip())
                buf = ""
                continue
            buf += ch
        for r in rows:
            vals = [
                self._coerce_literal(
                    self._resolve_param(params, tok)
                    if tok.startswith(":") or tok == "?"
                    else tok
                )
                for tok in r
            ]
            t.rows.append(tuple(vals))

    def _coerce_literal(self, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, (int, float, bool)):
            return v
        s = str(v).strip()
        if s.lower() == "null":
            return None
        if s.lower() == "true":
            return True
        if s.lower() == "false":
            return False
        if re.match(r"^-?\d+$", s):
            return int(s)
        if re.match(r"^-?\d+\.\d+$", s):
            return float(s)
        if s.startswith("'") and s.endswith("'"):
            return s[1:-1]
        return s

    def _do_update(self, s: str, params: dict) -> int:
        m = re.match(
            r"(?is)^UPDATE\s+`?(?P<t>[\w\.]+)`?\s+SET\s+(?P<set>.+?)(?=\bWHERE\b|\Z)(?P<rest>.*)$",
            s,
        )
        if not m:
            raise BadRequest(f"cannot parse UPDATE: {s[:80]}")
        t = self.table(m.group("t"))
        set_part = m.group("set").strip()
        where_part = m.group("rest").strip()
        if where_part.upper().startswith("WHERE"):
            where_part = where_part[5:].strip()
        # split SET on commas at top level
        sets = self._split_top_commas(set_part)
        assigns: list[tuple[str, Any]] = []
        for piece in sets:
            mm = re.match(r"`?(\w+)`?\s*=\s*(.+)$", piece.strip(), re.S)
            if not mm:
                raise BadRequest(f"bad SET clause: {piece}")
            col, expr = mm.group(1), mm.group(2).strip()
            # SET used to accept a bound parameter and nothing else, so
            # `SET ended_at = current_timestamp()` — ordinary Databricks
            # SQL — failed here for a reason that does not exist there.
            # Same value grammar as INSERT, so a statement cannot be
            # valid in one and rejected in the other.
            try:
                assigns.append((col, self._value_token(params, expr)))
            except BadRequest as exc:
                raise BadRequest(f"bad SET clause: {piece}") from exc
        pred = self._parse_where(where_part, params, t.columns) if where_part else None
        if where_part and pred is None:
            # An unparseable WHERE must NOT silently degrade to "match all"
            # for a mutation — that would rewrite every row in the table.
            raise BadRequest(f"unsupported UPDATE WHERE clause: {where_part!r}")
        n = 0
        for i, r in enumerate(list(t.rows)):
            if pred is None or pred(r):
                new = list(r)
                for col, v in assigns:
                    new[t.columns.index(col)] = v
                t.rows[i] = tuple(new)
                n += 1
        return n

    def _do_delete(self, s: str, params: dict) -> int:
        m = re.match(
            r"(?is)^DELETE\s+FROM\s+`?(?P<t>[\w\.]+)`?(?:\s+WHERE\s+(?P<w>.+))?$",
            s,
        )
        if not m:
            raise BadRequest(f"cannot parse DELETE: {s[:80]}")
        t = self.table(m.group("t"))
        where = m.group("w")
        pred = self._parse_where(where.strip(), params, t.columns) if where else None
        if where and where.strip() and pred is None:
            # Fail closed: an unparseable WHERE must not delete the whole table.
            raise BadRequest(f"unsupported DELETE WHERE clause: {where.strip()!r}")
        keep = []
        n = 0
        for r in t.rows:
            if pred is None or pred(r):
                n += 1
            else:
                keep.append(r)
        t.rows = keep
        return n

    def _do_merge(self, s: str, params: dict) -> None:
        m = re.match(
            r"(?is)^MERGE\s+INTO\s+`?(?P<tgt>[\w\.]+)`?\s+"
            r"(?:AS\s+`?(?P<tgt_a>\w+)`?\s+)?"
            r"USING\s+`?(?P<src>[\w\.]+)`?\s+"
            r"(?:AS\s+`?(?P<src_a>\w+)`?\s+)?"
            r"ON\s+(?P<on>.+?)\s+"
            r"WHEN\s+MATCHED\s+THEN\s+UPDATE\s+SET\s+(?P<set>.+?)$",
            s,
        )
        if not m:
            raise BadRequest(f"cannot parse MERGE: {s[:80]}")
        tgt = self.table(m.group("tgt"))
        src_name = m.group("src")
        on = m.group("on")
        set_part = m.group("set").strip()
        # ON tgt.x = src.x
        on_m = re.match(r"`?(?P<tc>\w+)`?\s*=\s*`?(?P<sc>\w+)`?", on.strip())
        if not on_m:
            raise BadRequest("MERGE ON must be a single equality")
        tc, _sc = on_m.group("tc"), on_m.group("sc")
        # SET tgt.a = src.b, ...
        sets = self._split_top_commas(set_part)
        assigns: list[tuple[str, tuple[str, Any]]] = []
        for piece in sets:
            mm = re.match(r"`?(?P<tc>\w+)`?\s*=\s*`?(?P<sc>\w+)`?", piece.strip())
            if not mm:
                raise BadRequest(f"bad MERGE SET: {piece}")
            sc = mm.group("sc")
            if sc in tgt.columns:
                # Source column name (SET t.a = t.b): keep the name; the
                # value is read per-row below. Coerce only literals.
                assigns.append((mm.group("tc"), ("col", sc)))
            else:
                assigns.append((mm.group("tc"), ("lit", self._coerce_literal(sc))))
        if src_name.lower() == tgt.name.lower():
            # self-merge: copy fields from one row to another
            rows = list(tgt.rows)
            i = 0
            while i < len(rows):
                j = i + 1
                while j < len(rows):
                    if rows[i][tgt.columns.index(tc)] == rows[j][tgt.columns.index(tc)]:
                        new = list(rows[i])
                        for col, (kind, val) in assigns:
                            if kind == "col":
                                new[tgt.columns.index(col)] = rows[j][
                                    tgt.columns.index(val)
                                ]
                            else:
                                new[tgt.columns.index(col)] = val
                        rows[i] = tuple(new)
                        del rows[j]
                        break
                    j += 1
                i += 1
            tgt.rows = rows
            return
        raise BadRequest(
            "MERGE with a separate source table is not implemented in the dev engine"
        )

    def _do_create(self, s: str) -> None:
        m = re.match(
            r"(?is)^CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?`?(?P<t>[\w\.]+)`?\s*\((?P<defs>.+)\)\s*$",
            s,
        )
        if not m:
            raise BadRequest(f"cannot parse CREATE TABLE: {s[:80]}")
        full = m.group("t")
        if full in self._tables:
            return
        defs = m.group("defs")
        cols: list[str] = []
        pk: list[str] = []
        for d in self._split_top_commas(defs):
            d = d.strip()
            mm = re.match(r"`?(?P<c>\w+)`?\s+[\w<>,\s]+", d)
            if mm:
                cols.append(mm.group("c"))
            if re.search(r"(?i)PRIMARY\s+KEY", d):
                pk_m = re.search(r"(?i)PRIMARY\s+KEY\s*\(([^)]+)\)", d)
                if pk_m:
                    pk = [c.strip().strip("`") for c in pk_m.group(1).split(",")]
        self.register(full, cols, pk=pk or None)

    def _do_drop(self, s: str) -> None:
        m = re.match(r"(?is)^DROP\s+TABLE\s+(?:IF\s+EXISTS\s+)?`?(?P<t>[\w\.]+)`?$", s)
        if not m:
            raise BadRequest(f"cannot parse DROP TABLE: {s[:80]}")
        self.drop(m.group("t"))

    @staticmethod
    def _split_top_commas(s: str) -> list[str]:
        out: list[str] = []
        depth = 0
        cur = ""
        for ch in s:
            if ch == "(":
                depth += 1
                cur += ch
                continue
            if ch == ")":
                depth -= 1
                cur += ch
                continue
            if ch == "," and depth == 0:
                out.append(cur)
                cur = ""
                continue
            cur += ch
        if cur.strip():
            out.append(cur)
        return out
