"""
Testing harness.

This module gives the rest of the system a single place to wire dev/test
fakes: a Warehouse, a Vector Search retriever, a Genie answer, an LLM
endpoint, and a Lakebase connection. Tests populate this; production code
reaches for `alex.testing.fakes` only when the runtime mode is "dev".
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from alex.config import get_settings
from alex.warehouse import InMemoryWarehouse


@dataclass
class FakeLLM:
    """A scripted LLM.

    Tests register a sequence of (predicate, response) pairs. The first
    matching predicate wins. If none match, `default` is returned. This is
    good enough to drive the agent end-to-end without a real model serving
    endpoint.
    """

    responses: list = field(default_factory=list)
    default: Any = None
    calls: list = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def register(self, predicate: Callable[[Any], bool], response: Any) -> None:
        self.responses.append((predicate, response))

    def invoke(self, messages: list, **kwargs) -> Any:
        with self._lock:
            self.calls.append({"messages": list(messages), **kwargs})
            for pred, resp in self.responses:
                try:
                    if pred(messages):
                        return resp
                except Exception:
                    continue
            return self.default


@dataclass
class FakeRetriever:
    docs: list = field(default_factory=list)

    def invoke(self, query: str) -> list:
        # Return all docs whose `text` contains the query as a substring.
        return [d for d in self.docs if query.lower() in d.get("text", "").lower()]


@dataclass
class FakeGenie:
    answers: dict = field(default_factory=dict)
    default: str = "I'm afraid I don't have data for that question."

    def invoke(self, payload: dict) -> dict:
        q = payload["messages"][-1]["content"]
        return {
            "messages": [
                {"role": "assistant", "content": self.answers.get(q, self.default)}
            ]
        }


@dataclass
class FakeLakebase:
    """An in-memory Postgres-shaped fake for `agent/memory.py` connections."""

    tables: dict = field(default_factory=dict)

    def cursor(self):
        return _FakeCursor(self)

    def commit(self):
        pass

    def rollback(self):
        pass

    def close(self):
        pass

    def execute(self, sql: str, params: tuple = ()):
        # Return a result-like object that supports fetchall / fetchone
        return _FakeResult(self, sql, params)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeCursor:
    def __init__(self, lb: FakeLakebase):
        self.lb = lb
        self._sql = ""
        self._params = ()

    def execute(self, sql, params=()):
        self._sql = sql
        self._params = tuple(params)

    def fetchall(self):
        return self.lb.execute(self._sql, self._params).fetchall()

    def fetchone(self):
        return self.lb.execute(self._sql, self._params).fetchone()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeResult:
    def __init__(self, lb: FakeLakebase, sql: str, params: tuple):
        self.lb = lb
        self.sql = sql
        self.params = params

    def fetchall(self):
        return list(self.lb.tables.get("default", []))

    def fetchone(self):
        rows = self.lb.tables.get("default", [])
        return rows[0] if rows else None


@dataclass
class Fakes:
    """Container for all dev fakes. The agent reads from this in dev mode."""

    warehouse: Any = field(default_factory=lambda: _default_warehouse())
    llm: FakeLLM = field(default_factory=FakeLLM)
    retriever: FakeRetriever = field(default_factory=FakeRetriever)
    genie: FakeGenie = field(default_factory=FakeGenie)
    lakebase: FakeLakebase = field(default_factory=FakeLakebase)
    secrets: dict = field(default_factory=dict)
    # Optional: the agentic browser OS backend. Tests that exercise
    # the browser tool surface register an InMemoryBrowserOS here.  In
    # production this attribute is never set and the agent uses the
    # Playwright-backed singleton from `agent.browser_os`.
    browser_os: Any = None
    # Optional: a callable ``(Event) -> None`` that the agentic
    # browser operations push SSE events through.  Tests register a
    # recording list-append function; production code subscribes
    # through the FastAPI route.
    sse_sink: Any = None
    # Per-session SSE subscriber registry: ``{session_id: set(callbacks)}``.
    # The browser stream route registers its own queue-push here so each
    # subscriber receives ONLY its session's events (no cross-user leak).
    browser_subscribers: dict = field(default_factory=dict)


# Process-singleton. Tests can replace the whole fakes object with
# `set_fakes(...)` and reset with `reset_fakes()`.
_fakes: Fakes | None = None
_lock = threading.Lock()


def _default_warehouse():
    """The local store: durable SQLite when ALEX_DB_PATH names a file
    (the running app), the in-memory engine otherwise (tests)."""
    import os

    if os.environ.get("ALEX_DB_PATH", "").strip():
        from alex.sqlite_warehouse import SqliteWarehouse

        return SqliteWarehouse(os.environ["ALEX_DB_PATH"].strip())
    return InMemoryWarehouse()


def get_fakes() -> Fakes:
    global _fakes
    if _fakes is None:
        _fakes = Fakes()
    return _fakes


def set_fakes(f: Fakes) -> None:
    global _fakes
    with _lock:
        _fakes = f


def reset_fakes() -> Fakes:
    global _fakes
    with _lock:
        _fakes = Fakes()
    return _fakes


def is_dev() -> bool:
    return get_settings().mode == "dev"
