"""The constitutional record, and who may read it.

The interceptor now denies and records; this is the other half. A
violation that nothing surfaces is a log nobody reads, and the whole
reason the constitutional layers exist is that the people accountable
for the agent can see when it was stopped.

Distinct from `/api/oversight/findings`: supervision reviews a run after
it finished, this is the ledger of refusals made while the agent was
attempting an action.
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


def _set_role(email: str, role: str):
    from alex.testing import get_fakes

    get_fakes().warehouse.execute(
        "INSERT INTO media.control.user_roles "
        "(user_email, role, team, granted_scope, clearance) "
        "VALUES (:u, :r, NULL, NULL, NULL)",
        {"u": email, "r": role},
    )
    from agent import access

    access.invalidate(email)


def _violation(article="A9", disposition="blocked", user="dana@alex.dev"):
    from agent.constitution import make_violation_logger
    from alex.testing import get_fakes

    wh = get_fakes().warehouse

    def sql(stmt, parameters=None):
        res = wh.execute(stmt, parameters or {})
        as_dicts = getattr(res, "as_dicts", None)
        return as_dicts() if callable(as_dicts) else list(res or [])

    make_violation_logger(sql, user, "s1")(
        article, "vm_exec", disposition, "prohibited capability: compute.shell"
    )


def test_violations_are_visible_to_an_administrator(env):
    client, headers = _client_and_token("boss@alex.dev")
    with client:
        client.get("/api/security/me", headers=headers)  # triggers the dev seed
        _set_role("boss@alex.dev", "admin")
        _violation()
        r = client.get("/api/oversight/violations", headers=headers)

    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 1
    row = body["violations"][0]
    assert row["article_id"] == "A9"
    assert row["disposition"] == "blocked"
    assert row["action"] == "vm_exec"


def test_a_non_admin_is_refused(env):
    client, headers = _client_and_token("demo@alex.dev")  # manager, not admin
    with client:
        client.get("/api/security/me", headers=headers)
        _violation()
        r = client.get("/api/oversight/violations", headers=headers)

    assert r.status_code == 403
    assert "violations" not in r.json()


def test_it_requires_authentication(env):
    from app.app import app

    with TestClient(app) as c:
        assert c.get("/api/oversight/violations").status_code == 401


def test_the_summary_separates_blocked_from_merely_noted(env):
    """The difference between the two is whether the action happened, so
    a summary that lumps them together is worse than none."""
    client, headers = _client_and_token("boss@alex.dev")
    with client:
        client.get("/api/security/me", headers=headers)
        _set_role("boss@alex.dev", "admin")
        _violation(article="A9", disposition="blocked")
        _violation(article="A9", disposition="warned")
        _violation(article="A2", disposition="warned", user="sam@alex.dev")
        body = client.get("/api/oversight/violations", headers=headers).json()

    summary = body["summary"]
    assert summary["total"] == 3
    assert summary["by_disposition"]["blocked"] == 1
    assert summary["by_disposition"]["warned"] == 2
    assert summary["by_article"]["A9"] == 2
    assert summary["distinct_users"] == 2


def test_it_can_be_filtered_to_blocks_only(env):
    client, headers = _client_and_token("boss@alex.dev")
    with client:
        client.get("/api/security/me", headers=headers)
        _set_role("boss@alex.dev", "admin")
        _violation(disposition="blocked")
        _violation(disposition="warned")
        body = client.get(
            "/api/oversight/violations?disposition=blocked", headers=headers
        ).json()

    assert [v["disposition"] for v in body["violations"]] == ["blocked"]


def test_an_empty_ledger_is_not_an_error(env):
    client, headers = _client_and_token("boss@alex.dev")
    with client:
        client.get("/api/security/me", headers=headers)
        _set_role("boss@alex.dev", "admin")
        r = client.get("/api/oversight/violations", headers=headers)

    assert r.status_code == 200
    assert r.json()["violations"] == []


def test_recent_violations_survives_a_missing_table(env):
    """Oversight degrading to empty beats oversight 500ing — but only
    because the read is the thing failing, not the enforcement."""
    from agent import constitution

    def broken(_stmt, _params=None):
        raise RuntimeError("no such table")

    assert constitution.recent_violations(broken) == []


# ---- interrupted runs must reach the console -----------------------------
# The runs endpoint used to list only the in-process registry, so a restart
# erased its history — including, precisely, the runs that restart had
# interrupted. An administrator could not learn that agent work had been
# dropped, which is the thing they most need to know after one.


def _interrupted_run(run_id="run-int", task_id="t-int"):
    from alex.testing import get_fakes

    get_fakes().warehouse.execute(
        "INSERT INTO media.control.task_runs "
        "(run_id, task_id, session_id, started_by, status, started_at, summary) "
        "VALUES (:r, :t, :s, :u, 'interrupted', :ts, :m)",
        {
            "r": run_id, "t": task_id, "s": f"task-{task_id}",
            "u": "dana@alex.dev", "ts": "2026-08-01 09:00:00+00:00",
            "m": "the process working it went away",
        },
    )


def test_dropped_work_is_visible_after_a_restart(env):
    client, headers = _client_and_token("boss@alex.dev")
    with client:
        client.get("/api/security/me", headers=headers)
        _set_role("boss@alex.dev", "admin")
        _interrupted_run()
        body = client.get("/api/oversight/runs", headers=headers).json()

    assert body["interrupted_count"] == 1
    assert body["interrupted"][0]["task_id"] == "t-int"
    # And it is in the durable history, not only the callout.
    assert any(r["run_id"] == "run-int" for r in body["recent"])


def test_an_interrupted_run_is_not_reported_as_live(env):
    """It is the opposite of live: nothing is working on it."""
    client, headers = _client_and_token("boss@alex.dev")
    with client:
        client.get("/api/security/me", headers=headers)
        _set_role("boss@alex.dev", "admin")
        _interrupted_run()
        body = client.get("/api/oversight/runs", headers=headers).json()

    assert body["live"] == []
    assert body["live_count"] == 0


def test_the_runs_console_is_admin_only(env):
    client, headers = _client_and_token("demo@alex.dev")  # manager, not admin
    with client:
        client.get("/api/security/me", headers=headers)
        assert client.get("/api/oversight/runs", headers=headers).status_code == 403
