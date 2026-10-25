"""Tests for the /api/security/* routes.

The access ledger is the oversight surface that *replaces* letting
administrators read colleagues' content, so its own gate matters as much
as the gates it reports on.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("ALEX_ENV", "dev")
    from alex import config, testing

    config._settings = None
    testing._fakes = None
    from agent import access

    access.invalidate()
    yield
    config._settings = None
    testing._fakes = None
    access.invalidate()


def _client_and_token(email: str):
    from app.app import app

    client = TestClient(app)
    with client:
        tok = client.get(f"/api/dev/token?email={email}").json()["token"]
    return TestClient(app), {"X-Forwarded-Access-Token": tok}


def _set_role(email: str, role: str, team: str | None = None):
    from alex.testing import get_fakes

    get_fakes().warehouse.execute(
        "INSERT INTO media.control.user_roles "
        "(user_email, role, team, granted_scope, clearance) "
        "VALUES (:u, :r, :t, NULL, NULL)",
        {"u": email, "r": role, "t": team},
    )
    from agent import access

    access.invalidate(email)


def test_security_me_reports_the_callers_own_context(env):
    client, headers = _client_and_token("demo@alex.dev")
    with client:
        r = client.get("/api/security/me", headers=headers)
    assert r.status_code == 200
    body = r.json()
    # The dev seed makes the demo user a manager on the platform team.
    assert body["role"] == "manager"
    assert body["clearance"] == "confidential"
    assert "team:platform" in body["teams"]
    assert "user:demo@alex.dev" in body["visible_scopes"]


def test_security_me_requires_authentication(env):
    from app.app import app

    with TestClient(app) as c:
        assert c.get("/api/security/me").status_code == 401


def test_access_log_is_refused_to_non_admins(env):
    client, headers = _client_and_token("demo@alex.dev")
    with client:
        r = client.get("/api/security/access-log", headers=headers)
    assert r.status_code == 403
    assert r.json()["your_role"] == "manager"


def test_access_log_is_served_to_admins(env):
    client, headers = _client_and_token("root@alex.dev")
    with client:
        # Seed the tables, then promote.
        client.get("/api/security/me", headers=headers)  # triggers the dev seed
        _set_role("root@alex.dev", "admin")
        r = client.get("/api/security/access-log", headers=headers)
    assert r.status_code == 200
    assert "entries" in r.json() and "denied" in r.json()


def test_access_log_records_a_denied_read(env):
    from agent import access, rag

    client, headers = _client_and_token("root@alex.dev")
    with client:
        client.get("/api/security/me", headers=headers)  # triggers the dev seed
        _set_role("root@alex.dev", "admin")
        # A member attempts a restricted document.
        rag.ingest(
            None, title="Board pack", text="restricted board material",
            source_ref="w/board", classification="restricted",
        )
        member = access.Principal(
            email="junior@alex.dev", role="member",
            capabilities=frozenset({rag.DEFAULT_CAPABILITY}),
        )
        assert rag.retrieve(None, "board material", k=3, principal=member) == []
        r = client.get("/api/security/access-log", headers=headers)
    body = r.json()
    assert body["denied"] >= 1
    denial = next(e for e in body["entries"] if not e["allowed"])
    assert denial["user_email"] == "junior@alex.dev"
    assert denial["classification"] == "restricted"
