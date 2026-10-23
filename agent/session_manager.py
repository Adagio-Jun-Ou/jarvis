"""
Session manager: list, create, append-turn, and branch per-conversation state.

**One storage model, two backends.** Dev and production both persist to
the `sessions` / `session_turns` tables — the in-memory warehouse in
dev, Lakebase in production. It was not always so: dev used to keep a
parallel in-process dict, which meant sessions had *two writers* (the
`/api/sessions` route inserted into the warehouse while this module
wrote its dict), and the ownership check consulted only the dict — so
`GET /api/sessions/{id}/messages` returned `[]` for the owner's own
UI-created session. Resume was broken for exactly the sessions the UI
makes. Unifying the store is what removes that class of bug, not
patching the check: two stores for one concept will always find a new
way to disagree.

The app imports this; the agent does not (the agent is stateless
between turns — the app feeds it the full conversation).
"""

from __future__ import annotations

import contextlib
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from alex.config import get_settings

log = logging.getLogger("agent.session_manager")


@contextlib.contextmanager
def _conn_ctx(conn):
    """The production Postgres connection, provided or self-acquired.

    Every caller in the app passes ``conn=None`` — the routes have no
    Lakebase handle of their own — and the prod branches used to walk
    straight into ``None.cursor()``. When no connection is supplied,
    borrow one from the memory layer (same instance, same credential
    vending); a supplied one is used as-is and not closed here.
    """
    if conn is not None:
        yield conn
        return
    from agent.memory import connection

    with connection() as owned:
        yield owned

SESSIONS_TABLE = "media.control.sessions"
TURNS_TABLE = "media.control.session_turns"


@dataclass
class Session:
    session_id: str
    owner_email: str
    kind: str  # 'personal' | 'group' | 'branch'
    title: str = ""
    parent_id: str | None = None
    participants: list[str] = None  # type: ignore[assignment]
    model: str = ""
    created_at: float = 0.0
    last_active_at: float = 0.0

    def __post_init__(self):
        if self.participants is None:
            self.participants = []


# ---------------------------------------------------------------------------
# Dev backend — the shared in-memory warehouse, same tables as production
# ---------------------------------------------------------------------------


def _dev_sql(stmt: str, params: dict | None = None) -> list[dict]:
    from alex.testing import get_fakes

    wh = get_fakes().warehouse
    _ensure_dev_tables(wh)
    res = wh.execute(stmt, params or {})
    as_dicts = getattr(res, "as_dicts", None)
    return as_dicts() if callable(as_dicts) else list(res or [])


def _ensure_dev_tables(wh) -> None:
    """Register the two tables when the app seed has not run (bare unit
    tests). Registration is skipped when they exist, so seeded rows are
    never wiped."""
    existing = getattr(wh, "_tables", {}) or {}
    if SESSIONS_TABLE.lower() not in existing:
        wh.register(
            SESSIONS_TABLE,
            ["session_id", "owner_email", "kind", "title", "parent_id",
             "participants", "model", "created_at", "last_active_at"],
            pk=["session_id"],
        )
    if TURNS_TABLE.lower() not in existing:
        wh.register(
            TURNS_TABLE,
            ["session_id", "turn_no", "role", "content", "ts"],
            pk=["session_id", "turn_no"],
        )


def _epoch(value) -> float:
    """Timestamp column -> epoch seconds, whatever shape the backend used."""
    if value is None:
        return 0.0
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, datetime):
        v = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return v.timestamp()
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _participants(value) -> list[str]:
    """ARRAY<STRING> column -> list, tolerating the dev warehouse's str()."""
    if value is None or value == "":
        return []
    if isinstance(value, list | tuple):
        return [str(v) for v in value if str(v)]
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        import ast

        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return [text]
        if isinstance(parsed, list | tuple):
            return [str(v) for v in parsed if str(v)]
    return [text]


def _dict_to_session(r: dict) -> Session:
    return Session(
        session_id=str(r.get("session_id") or ""),
        owner_email=str(r.get("owner_email") or ""),
        kind=str(r.get("kind") or "personal"),
        title=str(r.get("title") or ""),
        parent_id=r.get("parent_id"),
        participants=_participants(r.get("participants")),
        model=str(r.get("model") or ""),
        created_at=_epoch(r.get("created_at")),
        last_active_at=_epoch(r.get("last_active_at")),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_sessions(conn, owner: str) -> list[Session]:
    settings = get_settings()
    if settings.mode == "dev":
        rows = _dev_sql(
            f"SELECT * FROM {SESSIONS_TABLE} WHERE owner_email = :o",
            {"o": owner},
        )
        out = [_dict_to_session(r) for r in rows]
        out.sort(key=lambda s: -s.last_active_at)
        return out[:200]
    with _conn_ctx(conn) as c, c.cursor() as cur:
        cur.execute(
            "SELECT session_id, owner_email, kind, title, parent_id, "
            "       participants, model, created_at, last_active_at "
            "FROM sessions WHERE owner_email = %s "
            "ORDER BY last_active_at DESC LIMIT 200",
            (owner,),
        )
        rows = cur.fetchall()
    return [_row_to_session(r) for r in rows]


def open_session(
    conn,
    owner: str,
    title: str = "New chat",
    kind: str = "personal",
    parent_id: str | None = None,
    participants: list[str] | None = None,
) -> str:
    """Create a new session and return its id. Optimistic: returns immediately
    with the new id. The caller can race two `open_session` calls for the
    same `parent_id`; the second INSERT will succeed (different primary key)
    but the UI can de-dup by title."""
    settings = get_settings()
    sid = str(uuid.uuid4())
    if settings.mode == "dev":
        _dev_sql(
            f"INSERT INTO {SESSIONS_TABLE} "
            "(session_id, owner_email, kind, title, parent_id, participants, "
            " model, created_at, last_active_at) "
            "VALUES (:sid, :o, :k, :t, :p, :pp, '', current_timestamp(), "
            "        current_timestamp())",
            {"sid": sid, "o": owner, "k": kind, "t": title,
             "p": parent_id, "pp": list(participants or [owner])},
        )
        return sid
    with _conn_ctx(conn) as c, c.cursor() as cur:
        cur.execute(
            "INSERT INTO sessions (session_id, owner_email, kind, title, "
            "                    parent_id, participants, model, created_at, last_active_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, now(), now())",
            (sid, owner, kind, title, parent_id, participants or [owner], ""),
        )
    return sid


def append_turn(conn, session_id: str, turn_no: int, role: str, content: str) -> None:
    """Persist a turn. The caller supplies the turn number from its
    optimistic counter. On conflict the row is left as-is (the loser
    retries with the fresh tail)."""
    settings = get_settings()
    if settings.mode == "dev":
        existing = _dev_sql(
            f"SELECT turn_no FROM {TURNS_TABLE} "
            "WHERE session_id = :s AND turn_no = :n",
            {"s": session_id, "n": turn_no},
        )
        if existing:
            return
        _dev_sql(
            f"INSERT INTO {TURNS_TABLE} (session_id, turn_no, role, content, ts) "
            "VALUES (:s, :n, :r, :c, current_timestamp())",
            {"s": session_id, "n": turn_no, "r": role, "c": content},
        )
        return
    with _conn_ctx(conn) as c, c.cursor() as cur:
        cur.execute(
            "INSERT INTO session_turns (session_id, turn_no, role, content, ts) "
            "VALUES (%s, %s, %s, %s, now()) "
            "ON CONFLICT (session_id, turn_no) DO NOTHING",
            (session_id, turn_no, role, content),
        )


def get_session(conn, session_id: str) -> Session | None:
    settings = get_settings()
    if settings.mode == "dev":
        rows = _dev_sql(
            f"SELECT * FROM {SESSIONS_TABLE} WHERE session_id = :s",
            {"s": session_id},
        )
        return _dict_to_session(rows[0]) if rows else None
    with _conn_ctx(conn) as c, c.cursor() as cur:
        cur.execute(
            "SELECT session_id, owner_email, kind, title, parent_id, "
            "       participants, model, created_at, last_active_at "
            "FROM sessions WHERE session_id = %s",
            (session_id,),
        )
        r = cur.fetchone()
    return _row_to_session(r) if r else None


def branch_session(conn, parent_id: str, owner: str, title: str) -> str:
    """Create a branch session off a parent. The branch inherits the
    parent's `participants` list. Raises ValueError if the parent does
    not exist."""
    settings = get_settings()
    if not parent_id:
        raise ValueError("parent_id required")
    parent = get_session(conn, parent_id)
    if parent is None:
        raise ValueError(f"parent session not found: {parent_id}")
    sid = str(uuid.uuid4())
    if settings.mode == "dev":
        _dev_sql(
            f"INSERT INTO {SESSIONS_TABLE} "
            "(session_id, owner_email, kind, title, parent_id, participants, "
            " model, created_at, last_active_at) "
            "VALUES (:sid, :o, 'branch', :t, :p, :pp, '', "
            "        current_timestamp(), current_timestamp())",
            {"sid": sid, "o": owner, "t": title, "p": parent_id,
             "pp": list(parent.participants or [owner])},
        )
        return sid
    with _conn_ctx(conn) as c, c.cursor() as cur:
        cur.execute(
            "INSERT INTO sessions (session_id, owner_email, kind, title, "
            "                    parent_id, participants, model, created_at, last_active_at) "
            "SELECT %s, %s, 'branch', %s, session_id, participants, model, "
            "       now(), now() "
            "  FROM sessions WHERE session_id = %s",
            (sid, owner, title, parent_id),
        )
    return sid


def get_turns(
    conn, session_id: str, *, limit: int = 200, owner_email: str | None = None
) -> list[dict]:
    """Read a session's persisted turns in order. Enables the SDK / a
    fresh client to resume a conversation server-side.

    ``owner_email`` scopes the read: without it a session id is the
    only access control on a user's conversation history, and session
    ids travel in URLs, logs and group shares. Callers that have
    *already* verified ownership (the app routes do it via the
    sessions-table predicate) may omit it; every route that has not
    must pass it.
    """
    settings = get_settings()
    if settings.mode == "dev":
        if owner_email is not None:
            s = get_session(conn, session_id)
            if s is None or s.owner_email != owner_email:
                return []
        rows = _dev_sql(
            f"SELECT turn_no, role, content FROM {TURNS_TABLE} "
            "WHERE session_id = :s",
            {"s": session_id},
        )
        rows.sort(key=lambda r: int(r.get("turn_no") or 0))
        return rows[-max(1, int(limit)):]
    sql = (
        "SELECT turn_no, role, content FROM session_turns "
        "WHERE session_id = %s"
    )
    params: tuple = (session_id,)
    if owner_email is not None:
        sql += " AND session_id IN (SELECT session_id FROM sessions " \
               "WHERE session_id = %s AND owner_email = %s)"
        params = (session_id, session_id, owner_email)
    sql += " ORDER BY turn_no DESC LIMIT %s"
    params = params + (limit,)
    with _conn_ctx(conn) as c, c.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    out = [{"turn_no": r[0], "role": r[1], "content": r[2]} for r in rows]
    return list(reversed(out))


def delete_turns(
    conn, session_id: str, *, owner_email: str | None = None
) -> int:
    """Remove a session's persisted turns (called when the session is
    deleted, so orphaned turns don't linger).

    Returns the number of sessions whose turns were removed — a route
    that deletes ``0`` sessions has no business removing anyone's
    turns, and that is exactly the check the caller should make
    before trusting the count.
    """
    settings = get_settings()
    if settings.mode == "dev":
        if owner_email is not None:
            s = get_session(conn, session_id)
            if s is None or s.owner_email != owner_email:
                return 0
        _dev_sql(
            f"DELETE FROM {TURNS_TABLE} WHERE session_id = :s",
            {"s": session_id},
        )
        return 1
    sql = "DELETE FROM session_turns WHERE session_id = %s"
    params: tuple = (session_id,)
    if owner_email is not None:
        sql = (
            "DELETE FROM session_turns WHERE session_id = %s AND session_id IN "
            "(SELECT session_id FROM sessions WHERE session_id = %s "
            "AND owner_email = %s)"
        )
        params = (session_id, session_id, owner_email)
    with _conn_ctx(conn) as c, c.cursor() as cur:
        cur.execute(sql, params)
        return cur.rowcount


def rename_session(conn, session_id: str, owner: str, title: str) -> bool:
    """Rename a session, owner-scoped. Returns whether a row changed."""
    title = (title or "").strip()[:200]
    if not title or not session_id:
        return False
    settings = get_settings()
    if settings.mode == "dev":
        s = get_session(conn, session_id)
        if s is None or s.owner_email != owner:
            return False
        _dev_sql(
            f"UPDATE {SESSIONS_TABLE} SET title = :t "
            "WHERE session_id = :s AND owner_email = :o",
            {"t": title, "s": session_id, "o": owner},
        )
        return True
    with _conn_ctx(conn) as c, c.cursor() as cur:
        cur.execute(
            "UPDATE sessions SET title = %s "
            "WHERE session_id = %s AND owner_email = %s",
            (title, session_id, owner),
        )
        return cur.rowcount > 0


def delete_session(conn, session_id: str, owner: str) -> bool:
    """Delete a session AND its turns, owner-scoped in one place.

    Returns whether the session was actually the caller's to delete —
    turns are only removed on a true ownership match, so a guessed id
    can never reach into anyone's history.
    """
    settings = get_settings()
    if settings.mode == "dev":
        s = get_session(conn, session_id)
        if s is None or s.owner_email != owner:
            return False
        _dev_sql(
            f"DELETE FROM {SESSIONS_TABLE} "
            "WHERE session_id = :s AND owner_email = :o",
            {"s": session_id, "o": owner},
        )
        delete_turns(conn, session_id, owner_email=owner)
        return True
    with _conn_ctx(conn) as c:
        with c.cursor() as cur:
            cur.execute(
                "DELETE FROM sessions "
                "WHERE session_id = %s AND owner_email = %s",
                (session_id, owner),
            )
            if cur.rowcount == 0:
                return False
            cur.execute(
                "DELETE FROM session_turns WHERE session_id = %s",
                (session_id,),
            )
        return True


def _row_to_session(r) -> Session:
    return Session(
        session_id=r[0],
        owner_email=r[1],
        kind=r[2],
        title=r[3] or "",
        parent_id=r[4],
        participants=list(r[5] or []),
        model=r[6] or "",
        created_at=r[7].timestamp() if r[7] else 0.0,
        last_active_at=r[8].timestamp() if r[8] else 0.0,
    )
