"""Tests for the app/ FastAPI backend.

These tests run against the FastAPI TestClient. We mock the warehouse
and the agent endpoint so the test runs without Databricks.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("ALEX_ENV", "dev")
    monkeypatch.setenv("WAREHOUSE_ID", "wh-test")
    monkeypatch.setenv("LAKEBASE_HOST", "")  # force dev warehouse
    # Reset the cached settings so the env changes take effect.
    from alex import config
    config._settings = None
    from alex import testing
    testing._fakes = None
    yield
    config._settings = None
    testing._fakes = None


def test_healthz(env):
    from fastapi.testclient import TestClient

    from app import app as appmod
    with TestClient(appmod.app) as client:
        r = client.get("/healthz")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


def test_alex_error_maps_to_its_http_status(env):
    from fastapi.testclient import TestClient

    from alex.errors import Denied
    from app import app as appmod

    # Register a throwaway route that raises a cross-boundary error; the
    # global middleware must surface it as 403 (not a blanket 500).
    async def _boom(request):
        raise Denied("nope", details={"cap": "x"})

    appmod.app.add_route("/_test_denied", _boom)
    # Move it ahead of any catch-all SPA route so it actually matches.
    appmod.app.router.routes.insert(0, appmod.app.router.routes.pop())
    try:
        with TestClient(appmod.app, raise_server_exceptions=False) as client:
            r = client.get("/_test_denied")
        assert r.status_code == 403
        assert r.json()["error"] == "denied"
        assert r.json()["details"] == {"cap": "x"}
    finally:
        appmod.app.router.routes = [
            rt for rt in appmod.app.router.routes
            if getattr(rt, "path", None) != "/_test_denied"
        ]


def test_security_headers_present(env):
    from fastapi.testclient import TestClient

    from app import app as appmod
    with TestClient(appmod.app) as client:
        r = client.get("/healthz")
        # HSTS
        assert "strict-transport-security" in {k.lower() for k in r.headers}
        # CSP
        csp = r.headers.get("content-security-policy", "")
        assert "default-src 'self'" in csp
        assert "frame-ancestors 'none'" in csp
        # X-Frame-Options
        assert r.headers.get("x-frame-options") == "DENY"


def test_me_requires_token(env):
    from fastapi.testclient import TestClient

    from app import app as appmod
    with TestClient(appmod.app) as client:
        r = client.get("/api/me")
        assert r.status_code == 401  # identity rejected -> 401


def test_me_with_dev_token(env):
    from fastapi.testclient import TestClient

    from alex.identity import mint_dev_token
    from app import app as appmod
    tok = mint_dev_token("alice@x.com", scopes=("serving.serving-endpoints", "sql"))
    with TestClient(appmod.app) as client:
        r = client.get("/api/me", headers={"Authorization": f"Bearer {tok}"})
        assert r.status_code == 200
        body = r.json()
        assert body["email"] == "alice@x.com"


def test_dev_token_endpoint_in_dev_mode(env):
    from fastapi.testclient import TestClient

    from app import app as appmod
    with TestClient(appmod.app) as client:
        r = client.get("/api/dev/token?email=alice@x.com")
        assert r.status_code == 200
        assert "token" in r.json()


def test_chat_without_obo_token_rejected(env):
    from fastapi.testclient import TestClient

    from alex.identity import mint_dev_token
    from app import app as appmod
    tok = mint_dev_token("alice@x.com")
    with TestClient(appmod.app) as client:
        r = client.post("/api/chat",
                        headers={"Authorization": f"Bearer {tok}"},
                        json={"session_id": "s1", "text": "hi"})
        assert r.status_code == 200
        # Body is an SSE stream; we read the first events.
        body = r.text
        assert "UNAUTHORIZED" in body


def test_consume_brief_flips_flag(env):
    from fastapi.testclient import TestClient

    from alex.identity import mint_dev_token
    from alex.testing import get_fakes
    from app import app as appmod

    tok = mint_dev_token("alice@x.com")
    with TestClient(appmod.app) as client:
        # Run the dev seed first (sets wh._seeded) so the endpoint's _sql
        # won't re-seed and wipe the row we register below.
        from app._dev_seed import seed_dev_warehouse
        wh = get_fakes().warehouse
        seed_dev_warehouse(wh)
        wh.register(
            "media.control.standing_order_briefs",
            ["brief_id", "order_id", "user_email", "brief_date",
             "compiled_context", "suggested_actions", "consumed"],
            pk=["brief_id"],
            rows=[("b-1", "so-1", "alice@x.com", "2026-07-21", "ctx", [], False)],
        )
        r = client.post("/api/briefs/consume",
                        headers={"Authorization": f"Bearer {tok}"},
                        json={"brief_id": "b-1"})
        assert r.status_code == 200
        assert r.json()["consumed"] == "b-1"
        # The row is now consumed.
        row = wh.table("media.control.standing_order_briefs").rows[0]
        assert row[-1] is True


def test_consume_brief_requires_id(env):
    from fastapi.testclient import TestClient

    from alex.identity import mint_dev_token
    from app import app as appmod
    tok = mint_dev_token("alice@x.com")
    with TestClient(appmod.app) as client:
        r = client.post("/api/briefs/consume",
                        headers={"Authorization": f"Bearer {tok}"}, json={})
        assert r.status_code == 400


def test_session_rename_and_delete(env):
    from fastapi.testclient import TestClient

    from alex.identity import mint_dev_token
    from app import app as appmod
    tok = {"Authorization": f"Bearer {mint_dev_token('demo@alex.dev')}"}
    with TestClient(appmod.app) as client:
        sid = client.post("/api/sessions", headers=tok,
                          json={"title": "Draft"}).json()["session_id"]
        # Rename.
        r = client.patch(f"/api/sessions/{sid}", headers=tok, json={"title": "Q1 plan"})
        assert r.status_code == 200 and r.json()["title"] == "Q1 plan"
        sessions = client.get("/api/sessions", headers=tok).json()["sessions"]
        assert any(s["session_id"] == sid and s["title"] == "Q1 plan" for s in sessions)
        # Delete.
        d = client.delete(f"/api/sessions/{sid}", headers=tok)
        assert d.status_code == 200
        sessions2 = client.get("/api/sessions", headers=tok).json()["sessions"]
        assert not any(s["session_id"] == sid for s in sessions2)


def test_session_rename_requires_title(env):
    from fastapi.testclient import TestClient

    from alex.identity import mint_dev_token
    from app import app as appmod
    tok = {"Authorization": f"Bearer {mint_dev_token('demo@alex.dev')}"}
    with TestClient(appmod.app) as client:
        sid = client.post("/api/sessions", headers=tok, json={}).json()["session_id"]
        r = client.patch(f"/api/sessions/{sid}", headers=tok, json={"title": "  "})
        assert r.status_code == 400


def test_transparency_memory_returns_primary_key(env):
    # The forget flow matches on the mirror PK `id`; the memory endpoint MUST
    # return it or the UI cannot request a real forget.
    from fastapi.testclient import TestClient

    from alex.identity import mint_dev_token
    from app import app as appmod
    tok = mint_dev_token("demo@alex.dev")
    with TestClient(appmod.app) as client:
        r = client.get("/api/transparency/memory",
                       headers={"Authorization": f"Bearer {tok}"})
        assert r.status_code == 200
        mems = r.json()["memories"]
        assert mems and all("id" in m and m["id"] for m in mems)


def test_transparency_actions_surface_outcome_and_tool(env):
    from fastapi.testclient import TestClient

    from alex.identity import mint_dev_token
    from alex.testing import get_fakes
    from app import app as appmod

    tok = mint_dev_token("alice@x.com")
    with TestClient(appmod.app) as client:
        from app._dev_seed import seed_dev_warehouse
        wh = get_fakes().warehouse
        seed_dev_warehouse(wh)
        wh.register(
            "media.alex.agent_actions",
            ["action_id", "actor", "ts", "action_type", "target_id",
             "detail", "outcome", "tool_name"],
            pk=["action_id"],
            rows=[("a-x", "alice@x.com", "2026-07-21 09:00:00", "tool_call",
                   None, "Blocked", "denied", "send_email")],
        )
        r = client.get("/api/transparency/actions",
                       headers={"Authorization": f"Bearer {tok}"})
        assert r.status_code == 200
        acts = r.json()["actions"]
        assert acts and acts[0]["outcome"] == "denied"
        assert acts[0]["tool_name"] == "send_email"


def test_path_traversal_blocked(env):
    from fastapi.testclient import TestClient

    from alex.identity import mint_dev_token
    from app import app as appmod
    tok = mint_dev_token("alice@x.com")
    with TestClient(appmod.app) as client:
        r = client.get("/api/files/download?path=../../../etc/passwd",
                       headers={"Authorization": f"Bearer {tok}"})
        assert r.status_code == 400
        assert "error" in r.json()


def test_path_download_only_outputs(env):
    from fastapi.testclient import TestClient

    from alex.identity import mint_dev_token
    from app import app as appmod
    tok = mint_dev_token("alice@x.com")
    with TestClient(appmod.app) as client:
        r = client.get("/api/files/download?path=uploads/foo.txt",
                       headers={"Authorization": f"Bearer {tok}"})
        assert r.status_code == 400


def test_sessions_create(env):
    from fastapi.testclient import TestClient

    from alex import testing
    from alex.identity import mint_dev_token
    from alex.warehouse import InMemoryWarehouse
    from app import app as appmod
    fk = testing.Fakes()
    fk.warehouse = InMemoryWarehouse()
    fk.warehouse.register("media.control.sessions",
                          ["session_id", "owner_email", "kind", "title",
                           "parent_id", "participants", "model",
                           "created_at", "last_active_at"],
                          pk=["session_id"])
    testing.set_fakes(fk)
    tok = mint_dev_token("alice@x.com")
    with TestClient(appmod.app) as client:
        r = client.post("/api/sessions",
                        headers={"Authorization": f"Bearer {tok}",
                                 "Content-Type": "application/json"},
                        json={"title": "Test"})
        assert r.status_code == 200
        assert "session_id" in r.json()


def test_sse_cache_headers(env):
    from fastapi.testclient import TestClient

    from alex.identity import mint_dev_token
    from app import app as appmod
    tok = mint_dev_token("alice@x.com")
    with TestClient(appmod.app) as client:
        r = client.post("/api/chat",
                        headers={"Authorization": f"Bearer {tok}"},
                        json={"session_id": "s1", "text": "hi"})
        assert r.headers.get("cache-control") == "no-cache"
        assert r.headers.get("x-accel-buffering") == "no"


def test_inbox_empty_in_dev(env):
    from fastapi.testclient import TestClient

    from alex.identity import mint_dev_token
    from app import app as appmod
    tok = mint_dev_token("alice@x.com")
    with TestClient(appmod.app) as client:
        r = client.get("/api/inbox", headers={"Authorization": f"Bearer {tok}"})
        assert r.status_code == 200
        body = r.json()
        assert "inbox" in body
        assert body["inbox"] == []


def test_tasks_empty_in_dev(env):
    from fastapi.testclient import TestClient

    from alex.identity import mint_dev_token
    from app import app as appmod
    tok = mint_dev_token("alice@x.com")
    with TestClient(appmod.app) as client:
        r = client.get("/api/tasks", headers={"Authorization": f"Bearer {tok}"})
        assert r.status_code == 200
        body = r.json()
        assert "tasks" in body
        assert body["tasks"] == []


def test_briefs_empty_in_dev(env):
    from fastapi.testclient import TestClient

    from alex.identity import mint_dev_token
    from app import app as appmod
    tok = mint_dev_token("alice@x.com")
    with TestClient(appmod.app) as client:
        r = client.get("/api/briefs", headers={"Authorization": f"Bearer {tok}"})
        assert r.status_code == 200
        body = r.json()
        # Endpoint shape: {"brief": <row or None>, "standing_orders": [...]}
        assert "brief" in body
        assert body["brief"] is None
        assert body["standing_orders"] == []


def test_safe_path_join_sanitizes_email(env, tmp_path, monkeypatch):
    from app import app as appmod
    monkeypatch.setattr(appmod, "WORKSPACE_ROOT", str(tmp_path))
    p = appmod._safe_path_join("..Weird/User@X.com", "outputs/a.txt")
    # Strict allowlist: lowercased, '/' replaced, leading dots stripped.
    assert "weird_user@x.com" in p.replace("\\", "/")
    assert ".." not in p


def test_download_content_disposition_is_escaped(env, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from alex.identity import mint_dev_token
    from app import app as appmod
    monkeypatch.setattr(appmod, "WORKSPACE_ROOT", str(tmp_path))
    outdir = tmp_path / "alice@x.com" / "outputs"
    outdir.mkdir(parents=True)
    name = "naïve report.pdf"
    (outdir / name).write_bytes(b"pdf")
    tok = mint_dev_token("alice@x.com")
    with TestClient(appmod.app) as client:
        r = client.get("/api/files/download",
                       params={"path": f"outputs/{name}"},
                       headers={"Authorization": f"Bearer {tok}"})
        assert r.status_code == 200
        disp = r.headers["content-disposition"]
        # ASCII-only fallback plus RFC 5987 filename*.
        assert disp.encode("ascii")
        assert "filename*=UTF-8''na%C3%AFve%20report.pdf" in disp
        assert 'filename="na_ve report.pdf"' in disp


def _delegation_fakes():
    from alex import testing
    from alex.warehouse import InMemoryWarehouse
    from app._dev_seed import seed_dev_warehouse
    fk = testing.Fakes()
    fk.warehouse = InMemoryWarehouse()
    seed_dev_warehouse(fk.warehouse)
    # alice supervises bob (and only bob).
    fk.warehouse.table("media.control.supervision").rows.append(
        ("alice@x.com", "bob@x.com")
    )
    testing.set_fakes(fk)
    return fk


def test_delegate_requires_supervision_link(env):
    from fastapi.testclient import TestClient

    from alex.identity import mint_dev_token
    from app import app as appmod
    _delegation_fakes()
    tok = mint_dev_token("alice@x.com")
    h = {"Authorization": f"Bearer {tok}"}
    with TestClient(appmod.app) as client:
        # Not a report of alice -> denied.
        r = client.post("/api/supervisor/delegate", headers=h,
                        json={"report_email": "carol@x.com",
                              "capability": "email.draft", "days": 5})
        assert r.status_code == 403
        # A real report -> allowed.
        r = client.post("/api/supervisor/delegate", headers=h,
                        json={"report_email": "bob@x.com",
                              "capability": "email.draft", "days": 5})
        assert r.status_code == 200


def test_dev_seed_is_idempotent():
    from alex.warehouse import InMemoryWarehouse
    from app._dev_seed import seed_dev_warehouse
    wh = InMemoryWarehouse()
    seed_dev_warehouse(wh)
    assert wh._seeded is True
    counts = {
        t: len(wh.execute(f"SELECT * FROM {t}").as_dicts())
        for t in (
            "media.alex.inbox_view",
            "media.alex.task_board_view",
            "media.alex.morning_briefs",
            "media.control.sessions",
            "media.alex.agent_actions",
            "media.control.constitution",
            "media.control.employee_acl",
        )
    }
    # A second call is a no-op: rows written after the first seed survive.
    wh.execute(
        "INSERT INTO media.control.employee_acl (user_email, capability) "
        "VALUES (:u, :c)",
        {"u": "alice@x.com", "c": "email.read"},
    )
    seed_dev_warehouse(wh)
    assert len(wh.execute("SELECT * FROM media.control.employee_acl").as_dicts()) == (
        counts["media.control.employee_acl"] + 1
    )
    # Explicit re-seed allowed if the flag is cleared; counts are restored.
    wh._seeded = False
    seed_dev_warehouse(wh)
    for t, n in counts.items():
        assert len(wh.execute(f"SELECT * FROM {t}").as_dicts()) == n
