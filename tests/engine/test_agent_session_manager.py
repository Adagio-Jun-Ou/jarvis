"""Tests for agent/session_manager.py — the per-session state."""
from __future__ import annotations

import pytest

from agent import session_manager as sm
from agent.memory import connection
from alex.testing import reset_fakes


@pytest.fixture(autouse=True)
def _reset():
    reset_fakes()
    yield
    reset_fakes()


def test_open_and_list_sessions():
    with connection() as lb:
        sid = sm.open_session(lb, "alice@x.com", "Test chat")
        assert sid
        sessions = sm.list_sessions(lb, "alice@x.com")
        assert len(sessions) == 1
        assert sessions[0].session_id == sid
        assert sessions[0].owner_email == "alice@x.com"
        assert sessions[0].title == "Test chat"


def test_list_excludes_other_users():
    with connection() as lb:
        sm.open_session(lb, "alice@x.com", "Alice")
        sm.open_session(lb, "bob@x.com", "Bob")
        a = sm.list_sessions(lb, "alice@x.com")
        b = sm.list_sessions(lb, "bob@x.com")
        assert {s.owner_email for s in a} == {"alice@x.com"}
        assert {s.owner_email for s in b} == {"bob@x.com"}


def test_branch_inherits_parent():
    with connection() as lb:
        parent = sm.open_session(lb, "alice@x.com", "parent")
        child = sm.branch_session(lb, parent, "alice@x.com", "branch")
        assert child != parent
        s = sm.get_session(lb, child)
        assert s is not None
        assert s.parent_id == parent
        assert s.kind == "branch"


def test_append_and_get_turns_dev_roundtrip():
    from alex.testing import reset_fakes

    reset_fakes()
    from agent import session_manager as sm

    sm.append_turn(None, "s-abc", 0, "user", "hi")
    sm.append_turn(None, "s-abc", 1, "assistant", "hello there")
    sm.append_turn(None, "s-abc", 0, "user", "dup")  # same turn_no ignored
    got = sm.get_turns(None, "s-abc")
    assert [t["content"] for t in got] == ["hi", "hello there"]
    assert [t["role"] for t in got] == ["user", "assistant"]
    # Isolation: a different session is empty.
    assert sm.get_turns(None, "s-other") == []


# ---- owner scoping ----------------------------------------------------------
# A session id is not a capability: it travels in URLs, logs and group
# shares. Reads and deletes of a session's turns must be scoped to the
# caller, or anyone holding the id reads (or destroys) the history.


def test_get_turns_is_owner_scoped():
    with connection() as lb:
        sid = sm.open_session(lb, "alice@x.com", "private")
        sm.append_turn(lb, sid, 0, "user", "the payroll numbers are…")

        mine = sm.get_turns(lb, sid, owner_email="alice@x.com")
        theirs = sm.get_turns(lb, sid, owner_email="bob@x.com")

        assert [t["content"] for t in mine], "the owner lost their own history"
        assert theirs == [], "another user read a session they do not own"


def test_get_turns_of_an_unknown_session_is_empty_not_an_error():
    with connection() as lb:
        assert sm.get_turns(lb, "no-such-sid", owner_email="bob@x.com") == []


def test_delete_turns_is_owner_scoped():
    with connection() as lb:
        sid = sm.open_session(lb, "alice@x.com", "private")
        sm.append_turn(lb, sid, 0, "user", "secret")

        removed = sm.delete_turns(lb, sid, owner_email="bob@x.com")
        assert removed == 0, "a non-owner's delete removed the owner's turns"
        assert sm.get_turns(lb, sid, owner_email="alice@x.com"), (
            "the non-owner's delete destroyed the history"
        )

        removed = sm.delete_turns(lb, sid, owner_email="alice@x.com")
        assert removed == 1
        assert sm.get_turns(lb, sid, owner_email="alice@x.com") == []
