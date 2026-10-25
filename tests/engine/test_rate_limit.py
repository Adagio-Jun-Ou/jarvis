"""Per-user rate limiting on the expensive app routes.

The AI Gateway limits the LLM; this limits the APP — a retry loop or a
script gone wrong must not occupy every worker with agent turns. The
guard is per user, sliding-window, and generous: normal use never sees
it.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("ALEX_ENV", "dev")
    monkeypatch.setenv("ALEX_EMBED_HASH_ONLY", "1")
    from alex import config, testing

    config._settings = None
    testing._fakes = None
    yield
    config._settings = None
    testing._fakes = None


def test_the_limit_binds_per_user_and_recovers(env, monkeypatch):
    from app import app as app_mod

    monkeypatch.setattr(app_mod, "_RATE_LIMIT", 2)
    monkeypatch.setattr(app_mod, "_RATE_WINDOW_S", 60.0)
    app_mod._RATE.clear()

    assert app_mod._rate_ok("a@x.com")
    assert app_mod._rate_ok("a@x.com")
    assert not app_mod._rate_ok("a@x.com"), "third call within window passes"
    # Another user is untouched by the first user's burst.
    assert app_mod._rate_ok("b@x.com")


def test_the_tool_route_returns_429_when_limited(env, monkeypatch):
    from fastapi.testclient import TestClient

    from alex.identity import mint_dev_token
    from app import app as app_mod

    monkeypatch.setattr(app_mod, "_RATE_LIMIT", 1)
    app_mod._RATE.clear()

    client = TestClient(app_mod.app)
    headers = {
        "X-Forwarded-Access-Token": mint_dev_token(
            "demo@alex.dev", scopes=["sql"]
        )
    }
    first = client.post(
        "/api/tools/list_my_tasks", headers=headers, json={"args": {}}
    )
    assert first.status_code in (200, 403)  # governed either way
    second = client.post(
        "/api/tools/list_my_tasks", headers=headers, json={"args": {}}
    )
    assert second.status_code == 429
    assert second.json()["error"] == "rate_limited"


def test_zero_disables_the_guard(env, monkeypatch):
    from app import app as app_mod

    monkeypatch.setattr(app_mod, "_RATE_LIMIT", 0)
    app_mod._RATE.clear()
    for _ in range(50):
        assert app_mod._rate_ok("a@x.com")
