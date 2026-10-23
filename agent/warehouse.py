"""
Thin adapter from `agent.*` to `alex.warehouse.WarehouseClient` /
`alex.testing.get_fakes().warehouse`.

The agent doesn't care whether it's in dev or production; the adapter
picks the right one.

NotFound errors from the warehouse are passed through unchanged so that
the agent's callers can distinguish 'table missing' (NotFound) from
'no rows' (empty result) and apply their own fallbacks.
"""

from __future__ import annotations

from collections.abc import Callable

from alex.config import get_settings
from alex.errors import NotFound
from alex.testing import get_fakes
from alex.warehouse import InMemoryWarehouse, WarehouseClient


def make_sql() -> Callable:
    """Return a callable `sql(statement, parameters) -> list[dict]`.

    In dev: backed by the in-memory warehouse registered in fakes.
    In production: backed by the Databricks SDK statement-execution
    endpoint.
    """
    settings = get_settings()
    if settings.mode == "dev":
        wh: InMemoryWarehouse = get_fakes().warehouse
        return _wrap_in_memory(wh)
    client = WarehouseClient()
    return _wrap_databricks(client)


def _wrap_in_memory(wh: InMemoryWarehouse) -> Callable:
    def sql(statement: str, parameters: dict | None = None) -> list[dict]:
        try:
            r = wh.execute(statement, parameters or {})
        except NotFound:
            # Re-raise with the table name so callers can apply fallbacks.
            raise
        return r.as_dicts()

    return sql


def _wrap_databricks(client: WarehouseClient) -> Callable:
    def sql(statement: str, parameters: dict | None = None) -> list[dict]:
        r = client.execute(statement, parameters or {})
        return r.as_dicts()

    return sql
