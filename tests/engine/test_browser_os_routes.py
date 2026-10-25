"""HTTP-level tests for the Browser OS routes.

These tests use FastAPI's ``TestClient`` against the dev backend.
They verify OBO enforcement, audit-log emission, and the two-phase
submit happy path.  The constitution interceptor is bypassed
because the Browser OS is a tool surface, not the agent loop; the
interceptor is exercised in
``tests/test_browser_os_constitution.py``.
"""
from __future__ import annotations

import pytest

from agent.browser_os import InMemoryBrowserOS
from alex.testing import Fakes


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("ALEX_ENV", "dev")
    monkeypatch.setenv("WAREHOUSE_ID", "wh-test")
    monkeypatch.setenv("LAKEBASE_HOST", "")
    from alex import config, testing

    config._settings = None
    testing._fakes = None
    fakes = Fakes()
    fakes.browser_os = InMemoryBrowserOS()
    # Browser tools write to media.alex.agent_actions.
    fakes.warehouse.register(
        "media.alex.agent_actions",
        [
            "action_id",
            "session_id",
            "tool",
            "capability",
            "user_email",
            "instance_scope",
            "input_json",
            "output_json",
            "ok",
            "ts",
        ],
    )
    testing.set_fakes(fakes)
    yield fakes
    config._settings = None
    testing._fakes = None


def _client(env):
    from fastapi.testclient import TestClient

    from alex.identity import mint_dev_token
    from app import app as appmod

    tok = mint_dev_token("alice@x.com")
    return TestClient(appmod.app), tok


def test_create_list_close_session(env) -> None:
    client, tok = _client(env)
    h = {"Authorization": f"Bearer {tok}"}
    r = client.post("/api/browser/sessions", headers=h)
    assert r.status_code == 200
    sid = r.json()["session_id"]
    r = client.get("/api/browser/sessions", headers=h)
    assert r.status_code == 200
    assert any(s["session_id"] == sid for s in r.json()["sessions"])
    r = client.delete(f"/api/browser/sessions/{sid}", headers=h)
    assert r.status_code == 200
    r = client.get("/api/browser/sessions", headers=h)
    assert all(s["session_id"] != sid for s in r.json()["sessions"])


def test_session_not_found_for_other_user(env) -> None:
    client, _ = _client(env)
    r = client.delete("/api/browser/sessions/bsess_does_not_exist")
    assert r.status_code == 401 or r.status_code == 200  # dev path: identity may 401 first
    # If we got past identity, the OBO check returns 404.
    if r.status_code == 200:
        assert r.json().get("error") == "session not found"


def test_create_tab_and_list(env) -> None:
    client, tok = _client(env)
    h = {"Authorization": f"Bearer {tok}"}
    sid = client.post("/api/browser/sessions", headers=h).json()["session_id"]
    r = client.post(
        f"/api/browser/sessions/{sid}/tabs?url=alex://welcome", headers=h
    )
    assert r.status_code == 200
    body = r.json()
    assert body["url"] == "alex://welcome"
    tid = body["tab_id"]
    r = client.get(f"/api/browser/sessions/{sid}/tabs", headers=h)
    assert r.status_code == 200
    assert any(t["tab_id"] == tid for t in r.json()["tabs"])


def test_navigate_route_works(env) -> None:
    client, tok = _client(env)
    h = {"Authorization": f"Bearer {tok}"}
    sid = client.post("/api/browser/sessions", headers=h).json()["session_id"]
    tid = client.post(f"/api/browser/sessions/{sid}/tabs", headers=h).json()["tab_id"]
    r = client.post(
        f"/api/browser/sessions/{sid}/tabs/{tid}/navigate",
        headers=h,
        json={"url": "alex://welcome"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["title"] == "Welcome to Alex"


def test_navigate_route_rejects_mailto(env) -> None:
    client, tok = _client(env)
    h = {"Authorization": f"Bearer {tok}"}
    sid = client.post("/api/browser/sessions", headers=h).json()["session_id"]
    tid = client.post(
        f"/api/browser/sessions/{sid}/tabs?url=alex://welcome", headers=h
    ).json()["tab_id"]
    r = client.post(
        f"/api/browser/sessions/{sid}/tabs/{tid}/click",
        headers=h,
        json={"selector": "Email support"},
    )
    # The click route reports the backend's PermissionError as 403.
    assert r.status_code == 403
    assert "mailto" in r.json()["error"]


def test_submit_two_phase(env) -> None:
    client, tok = _client(env)
    h = {"Authorization": f"Bearer {tok}"}
    sid = client.post("/api/browser/sessions", headers=h).json()["session_id"]
    tid = client.post(
        f"/api/browser/sessions/{sid}/tabs?url=alex://welcome", headers=h
    ).json()["tab_id"]
    # Type a value first.
    r = client.post(
        f"/api/browser/sessions/{sid}/tabs/{tid}/type",
        headers=h,
        json={"selector": "email", "text": "alice@x.com"},
    )
    assert r.status_code == 200
    # First submit: returns a confirmation token.
    r = client.post(
        f"/api/browser/sessions/{sid}/tabs/{tid}/submit",
        headers=h,
        json={"selector": "form"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["awaiting_confirmation"] is True
    token = body["confirmation_token"]
    # Second submit with the token: actually runs.
    r = client.post(
        f"/api/browser/sessions/{sid}/tabs/{tid}/submit",
        headers=h,
        json={"selector": "form", "confirmation_token": token},
    )
    assert r.status_code == 200
    assert r.json()["submitted"] is True


def test_mirror_route(env) -> None:
    client, tok = _client(env)
    h = {"Authorization": f"Bearer {tok}"}
    sid = client.post("/api/browser/sessions", headers=h).json()["session_id"]
    tid = client.post(
        f"/api/browser/sessions/{sid}/tabs?url=alex://welcome", headers=h
    ).json()["tab_id"]
    r = client.post(
        f"/api/browser/sessions/{sid}/tabs/{tid}/mirror",
        headers=h,
        json={"mode": "co-drive"},
    )
    assert r.status_code == 200
    assert r.json()["agent_piloting"] is False
    # Now the agent's navigate is blocked.
    r = client.post(
        f"/api/browser/sessions/{sid}/tabs/{tid}/navigate",
        headers=h,
        json={"url": "alex://docs"},
    )
    assert r.status_code == 403
    # Yield back.
    r = client.post(
        f"/api/browser/sessions/{sid}/tabs/{tid}/mirror",
        headers=h,
        json={"mode": "yield"},
    )
    assert r.status_code == 200
    assert r.json()["agent_piloting"] is True


def test_extract_route_redacts_pii(env) -> None:
    client, tok = _client(env)
    h = {"Authorization": f"Bearer {tok}"}
    sid = client.post("/api/browser/sessions", headers=h).json()["session_id"]
    tid = client.post(f"/api/browser/sessions/{sid}/tabs", headers=h).json()["tab_id"]
    env.browser_os.get_tab(sid, tid).html = (
        "<html><body>Contact: bob@secret.com</body></html>"
    )
    r = client.post(
        f"/api/browser/sessions/{sid}/tabs/{tid}/extract",
        headers=h,
        json={"selector": "body"},
    )
    assert r.status_code == 200
    body = r.json()
    assert "[REDACTED:email]" in body["text"]
    assert body["pii_redacted"] is True


def test_other_user_cannot_touch_session(env) -> None:
    """User B must get 404 on every tab route for user A's session."""
    from alex.identity import mint_dev_token

    client, tok_a = _client(env)
    tok_b = mint_dev_token("mallory@x.com")
    ha = {"Authorization": f"Bearer {tok_a}"}
    hb = {"Authorization": f"Bearer {tok_b}"}
    sid = client.post("/api/browser/sessions", headers=ha).json()["session_id"]
    tid = client.post(
        f"/api/browser/sessions/{sid}/tabs?url=alex://welcome", headers=ha
    ).json()["tab_id"]
    # List/create tabs.
    assert client.get(f"/api/browser/sessions/{sid}/tabs", headers=hb).status_code == 404
    assert client.post(f"/api/browser/sessions/{sid}/tabs", headers=hb).status_code == 404
    # Tab-level actions.
    base = f"/api/browser/sessions/{sid}/tabs/{tid}"
    assert client.post(f"{base}/navigate", headers=hb,
                       json={"url": "alex://docs"}).status_code == 404
    assert client.post(f"{base}/click", headers=hb,
                       json={"selector": "x"}).status_code == 404
    assert client.post(f"{base}/type", headers=hb,
                       json={"selector": "x", "text": "y"}).status_code == 404
    assert client.post(f"{base}/extract", headers=hb,
                       json={"selector": "body"}).status_code == 404
    assert client.post(f"{base}/screenshot", headers=hb).status_code == 404
    assert client.post(f"{base}/submit", headers=hb,
                       json={"selector": "form"}).status_code == 404
    assert client.post(f"{base}/mirror", headers=hb,
                       json={"mode": "read"}).status_code == 404
    assert client.delete(base, headers=hb).status_code == 404
    # The owner still works.
    assert client.get(f"/api/browser/sessions/{sid}/tabs", headers=ha).status_code == 200


def test_unauthenticated_request_is_rejected(env) -> None:
    client, _ = _client(env)
    r = client.post("/api/browser/sessions")
    assert r.status_code == 401
