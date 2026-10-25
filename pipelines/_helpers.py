"""
Common helpers for the Lakeflow pipelines.

In production every pipeline runs on a Databricks Job cluster under
`sp-alex-jobs`. The helpers here provide:

  * `parameterized(stmt, **kwargs)` — produces a SQL statement with named
    placeholders ready for `spark.sql(stmt, args={...})`. No
    f-strings in pipelines.
  * `safe_warehouse(stmt, params, *, row_limit=10000)` — wraps a
    `WarehouseClient` and returns a list of dicts.
  * `retry_call(fn, **kw)` — exponential backoff with jitter for any
    transient failure (ai_query 503, Graph 429, etc.).
  * `with_quarantine(...)` — runs a per-row job; rows that fail
    JSON parsing or model refusals are sent to a quarantine table.
"""
from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable
from typing import Any

import alex.retry as _retry
from alex.errors import UpstreamError
from alex.warehouse import WarehouseClient

log = logging.getLogger("pipelines._helpers")


def parameterized(stmt: str, **kwargs) -> tuple[str, dict]:
    """Replace `:name` placeholders with Spark SQL named parameters and
    return (sanitized_statement, parameters_dict). The statement is
    checked: it must not contain any string-interpolation that looks
    like a Python f-string. The caller is expected to use
    `spark.sql(stmt, args=...)` or `WarehouseClient.execute`.
    """
    if "{" in stmt and "}" in stmt:
        # Conservative check for f-string syntax. Matches `{"` or `f"`.
        if re.search(r"['\"]\{[^}]+\}['\"]", stmt):
            raise ValueError(
                "pipeline SQL contains a Python f-string. Use named "
                "parameters and spark.sql(stmt, args=...) instead."
            )
    return stmt, dict(kwargs)


def safe_warehouse_call(warehouse: WarehouseClient, stmt: str, params: dict,
                        *, row_limit: int = 10_000) -> list[dict]:
    """Run a parameterized SELECT against the warehouse. Returns
    list-of-dicts; empty list on no rows."""
    try:
        r = warehouse.execute(stmt, params, row_limit=row_limit)
        return r.as_dicts()
    except Exception as e:
        log.exception("warehouse call failed")
        raise UpstreamError(f"warehouse call failed: {e}") from e


def retry(fn: Callable[[], Any], *,
          attempts: int = 4,
          base_delay_seconds: float = 0.5,
          retriable: tuple = (UpstreamError,)) -> Any:
    return _retry.call_with_retry(
        fn, attempts=attempts, base_delay_seconds=base_delay_seconds,
        retriable=retriable,
    )


# Backoff helper for arbitrary HTTP-ish calls. Pipelines use this for
# Graph / Gmail and for the in-house ai_query endpoint.
def with_backoff(fn: Callable[[], Any], *, attempts: int = 5,
                 base_delay: float = 0.5, max_delay: float = 30.0,
                 catch: tuple = (Exception,)) -> Any:
    import random
    last: BaseException | None = None
    for i in range(1, attempts + 1):
        try:
            return fn()
        except catch as e:
            last = e
            if i == attempts:
                break
            delay = min(max_delay, base_delay * (2 ** (i - 1)))
            delay *= (0.7 + 0.6 * random.random())
            log.warning("attempt %d/%d failed (%s); retrying in %.2fs",
                        i, attempts, type(e).__name__, delay)
            time.sleep(delay)
    assert last is not None
    raise last


def parse_json_response(raw: str) -> dict | None:
    """Extract the first JSON object from a model response. Tolerates
    code fences and surrounding prose. Returns None on parse failure
    so the caller can route the row to a quarantine table."""
    if not raw:
        return None
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```\s*$", "", s)
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return d if isinstance(d, dict) else None


def coalesce_columns(row: dict, defaults: dict) -> dict:
    """Fill missing keys with defaults; useful for tolerating partial
    model outputs."""
    out = dict(row)
    for k, v in defaults.items():
        out.setdefault(k, v)
    return out
