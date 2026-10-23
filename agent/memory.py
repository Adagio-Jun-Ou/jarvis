"""
Lakebase (Postgres) connection helpers for the agent's hot-state tables.

In dev/test this module uses `alex.testing.get_fakes().lakebase`. In
production it uses the Databricks SDK's database credential vending.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager

from alex.config import get_settings
from alex.testing import get_fakes

log = logging.getLogger("agent.memory")


@contextmanager
def connection() -> Iterator:
    """Yield a connection. In dev mode this is a FakeLakebase; in
    production it's a real psycopg connection to Lakebase.

    The connection is short-lived; the caller MUST NOT cache it across
    requests. We use minted-downscoped credentials per call."""
    settings = get_settings()
    if settings.mode == "dev":
        fk = get_fakes().lakebase
        try:
            yield fk
        finally:
            pass
        return
    # Production: real psycopg via Databricks SDK credential vending.
    import psycopg
    from databricks.sdk import WorkspaceClient

    ws = WorkspaceClient()
    host = os.environ.get("LAKEBASE_HOST", "")
    if not host:
        raise RuntimeError("LAKEBASE_HOST is not set")
    cred = ws.database.generate_database_credential(
        instance_names=[settings.lakebase_instance],
    )
    # The Postgres role must be the SAME identity that vended the
    # credential — the app SP in the App, the agent SP in serving, a
    # developer at a laptop. A hard-coded role works for exactly one of
    # those and fails authentication for the rest, so unless LAKEBASE_USER
    # pins it, ask the workspace who we are.
    user = os.environ.get("LAKEBASE_USER", "").strip()
    if not user:
        user = ws.current_user.me().user_name or ""
    conn = psycopg.connect(
        host=host,
        port=int(os.environ.get("LAKEBASE_PORT", "5432")),
        dbname=os.environ.get("LAKEBASE_DBNAME", "alex"),
        user=user,
        password=cred.token,
        sslmode="require",
    )
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
