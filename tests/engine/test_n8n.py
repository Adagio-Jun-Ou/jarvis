"""The n8n workflow arm: one seam, every security property in it.

* the instance is operator-configured (no tool or route takes a URL);
* the API key never appears in errors;
* TLS is required outside dev;
* created workflows carry provenance and arrive INACTIVE;
* the routes are capability-gated;
* results are marked untrusted (they can carry anything n8n touched).
"""

from __future__ import annotations

import pytest


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("ALEX_ENV", "dev")
    monkeypatch.setenv("ALEX_EMBED_HASH_ONLY", "1")
    monkeypatch.delenv("N8N_BASE_URL", raising=False)
    monkeypatch.delenv("N8N_API_KEY", raising=False)
    from alex import config, testing

    config._settings = None
    testing._fakes = None
    yield
    config._settings = None
    testing._fakes = None


# ---- the client (dev fake, same surface) -----------------------------------


def test_create_run_and_status_round_trip(env):
    from agent import n8n_client as n8n

    wf = n8n.create_workflow(
        name="Invoice sync",
        nodes=[{"name": "Trigger", "type": "n8n-nodes-base.webhook",
                "position": [0, 0]},
               {"name": "Post", "type": "n8n-nodes-base.httpRequest",
                "position": [200, 0]}],
        connections={},
        created_by="dana@alex.dev",
    )
    assert wf["id"]
    assert wf["active"] is False, "created workflows must arrive inactive"
    assert "[alex:dana@alex.dev]" in wf["name"], "provenance tag missing"

    listed = n8n.list_workflows()
    assert any(w["id"] == wf["id"] for w in listed)
    assert listed[0]["node_count"] == 2

    run = n8n.run_workflow(wf["id"], {"invoice": 42})
    status = n8n.execution(run["execution_id"])
    assert status["status"] == "success"


def test_activation_is_a_separate_deliberate_step(env):
    from agent import n8n_client as n8n

    wf = n8n.create_workflow(
        name="x", nodes=[{"name": "a"}], connections={}, created_by="d@x.co"
    )
    out = n8n.set_active(wf["id"], True)
    assert out["active"] is True
    assert n8n.get_workflow(wf["id"])["active"] is True


def test_a_workflow_needs_a_name_and_nodes(env):
    from agent import n8n_client as n8n

    with pytest.raises(n8n.N8NError):
        n8n.create_workflow(name="", nodes=[{}], connections={},
                            created_by="d@x.co")
    with pytest.raises(n8n.N8NError):
        n8n.create_workflow(name="y", nodes=[], connections={},
                            created_by="d@x.co")


# ---- security properties ---------------------------------------------------


def test_plain_http_is_refused_outside_dev(monkeypatch):
    monkeypatch.setenv("ALEX_ENV", "production")
    monkeypatch.setenv("N8N_BASE_URL", "http://n8n.internal:5678")
    monkeypatch.setenv("N8N_API_KEY", "k")
    from alex import config

    config._settings = None
    from agent import n8n_client as n8n

    with pytest.raises(n8n.N8NError, match="https"):
        n8n._base_url()
    config._settings = None


def test_the_api_key_never_appears_in_errors(env, monkeypatch):
    monkeypatch.setenv("N8N_BASE_URL", "https://n8n.example.invalid")
    monkeypatch.setenv("N8N_API_KEY", "sup3r-secret-key")
    from agent import n8n_client as n8n

    with pytest.raises(n8n.N8NError) as err:
        n8n.list_workflows()
    assert "sup3r-secret-key" not in str(err.value)


def test_no_tool_or_route_accepts_a_url(env):
    """The instance is operator configuration: a model that could name
    the host could exfiltrate through its own server."""
    import inspect

    from agent.tools import impls

    for fn in (impls.n8n_list_workflows, impls.n8n_create_workflow,
               impls.n8n_run_workflow, impls.n8n_execution_status):
        params = set(inspect.signature(fn).parameters)
        assert "url" not in params and "base_url" not in params


def test_the_n8n_tools_are_untrusted_and_capability_gated(env):
    from agent.guardrails import TOOL_CAPABILITY
    from agent.tools.registry import TOOL_REGISTRY

    assert TOOL_REGISTRY["n8n_list_workflows"].untrusted_output
    assert TOOL_REGISTRY["n8n_run_workflow"].untrusted_output
    assert TOOL_CAPABILITY["n8n_create_workflow"] == "workflow.create"
    assert TOOL_CAPABILITY["n8n_run_workflow"] == "workflow.trigger"
    from agent.execution_governor import CAPABILITY_POWER

    assert CAPABILITY_POWER["workflow.trigger"] >= 3, (
        "running enterprise automations is an external act"
    )


# ---- the routes ------------------------------------------------------------


def test_the_routes_serve_the_ui_end_to_end(env):
    from fastapi.testclient import TestClient

    from alex.identity import mint_dev_token
    from app.app import app

    client = TestClient(app)
    headers = {
        "X-Forwarded-Access-Token": mint_dev_token(
            "demo@alex.dev", scopes=["sql"]
        ),
        "Content-Type": "application/json",
    }
    created = client.post("/api/n8n/workflows", headers=headers, json={
        "name": "Morning digest",
        "nodes": [{"name": "Cron", "position": [0, 0]},
                  {"name": "Email", "position": [220, 0]}],
        "connections": {},
    })
    assert created.status_code == 200, created.text
    wf_id = created.json()["id"]

    listed = client.get("/api/n8n/workflows", headers=headers)
    assert listed.status_code == 200
    assert any(w["id"] == wf_id for w in listed.json()["workflows"])

    ran = client.post(f"/api/n8n/workflows/{wf_id}/run", headers=headers,
                      json={"payload": {}})
    assert ran.status_code == 200
    ex_id = ran.json()["execution_id"]

    status = client.get(f"/api/n8n/executions/{ex_id}", headers=headers)
    assert status.status_code == 200
    assert status.json()["status"] == "success"
