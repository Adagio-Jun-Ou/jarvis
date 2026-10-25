"""n8n — the enterprise workflow arm.

The organisation runs its own n8n Enterprise instance (separate AWS
Kubernetes cluster). This module is the ONLY seam between Alex and that
instance: the agent and the app both reach n8n through here, so every
security property lives in one place.

Security properties (each enforced in code, not prose):

* **The instance is operator-configured, never model-named.** The base
  URL comes from ``N8N_BASE_URL``; no tool or route accepts a URL. A
  model cannot point this client at a different host.
* **The API key is invisible.** ``N8N_API_KEY`` (injected from a
  Databricks secret in production) is read at call time, sent only as
  the ``X-N8N-API-KEY`` header, and never logged, returned, or included
  in errors.
* **TLS is required outside dev.** A plain-http enterprise credential
  on a real network is a credential shared with the network.
* **n8n responses are untrusted content.** Workflow definitions and
  execution payloads can embed data from anywhere n8n touched; the
  tools that expose them are marked ``untrusted_output`` so results are
  fenced before re-entering the model's context.
* **Provenance tagging.** Workflows Alex creates are tagged
  ``alex:<user>`` in their name, so the n8n side can always answer
  "where did this workflow come from?".
* **Bounded everything.** Timeouts on every call; response bodies
  size-capped; list endpoints truncated.

Dev runs against an in-memory fake (on ``alex.testing`` fakes) with the
same surface, so the whole arm is exercised offline.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
import uuid
from typing import Any

log = logging.getLogger("agent.n8n")

TIMEOUT_S = float(os.environ.get("N8N_TIMEOUT_S", "30"))
MAX_RESPONSE_BYTES = 2_000_000
MAX_LIST = 50


class N8NError(RuntimeError):
    """A clean, key-free error from the n8n seam."""


def _base_url() -> str:
    url = (os.environ.get("N8N_BASE_URL") or "").strip().rstrip("/")
    if not url:
        raise N8NError(
            "N8N_BASE_URL is not configured; the n8n arm is not connected"
        )
    from alex.testing import is_dev

    if not url.startswith("https://") and not is_dev():
        raise N8NError(
            "N8N_BASE_URL must be https in production — an enterprise "
            "credential over plain http is shared with the network"
        )
    return url


def _api_key() -> str:
    key = (os.environ.get("N8N_API_KEY") or "").strip()
    if not key:
        raise N8NError("N8N_API_KEY is not configured")
    return key


def _is_dev() -> bool:
    from alex.testing import is_dev

    return is_dev()


def _request(
    method: str, path: str, body: dict | None = None
) -> dict[str, Any]:
    """One call to the instance. The key never appears in errors."""
    url = _base_url() + path
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "X-N8N-API-KEY": _api_key(),
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            raw = resp.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read(400).decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            pass
        raise N8NError(f"n8n returned {exc.code} for {path}: {detail}") from None
    except Exception as exc:  # noqa: BLE001
        raise N8NError(f"n8n unreachable at {path}: {type(exc).__name__}") from None
    if len(raw) > MAX_RESPONSE_BYTES:
        raise N8NError(f"n8n response for {path} exceeded the size cap")
    try:
        parsed = json.loads(raw or b"{}")
    except ValueError:
        raise N8NError(f"n8n returned non-JSON for {path}") from None
    return parsed if isinstance(parsed, dict) else {"data": parsed}


# ---------------------------------------------------------------------------
# The dev fake — same surface, in memory, so the arm tests offline
# ---------------------------------------------------------------------------


def _dev_store() -> dict:
    from alex.testing import get_fakes

    fk = get_fakes()
    store = getattr(fk, "_n8n_store", None)
    if store is None:
        store = {"workflows": {}, "executions": {}}
        fk._n8n_store = store
    return store


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def _summ(wf: dict[str, Any]) -> dict[str, Any]:
    nodes = wf.get("nodes") or []
    return {
        "id": str(wf.get("id") or ""),
        "name": str(wf.get("name") or ""),
        "active": bool(wf.get("active")),
        "node_count": len(nodes),
        "nodes": [
            {
                "name": str(n.get("name") or ""),
                "type": str(n.get("type") or "").rsplit(".", 1)[-1],
                "position": n.get("position") or [0, 0],
            }
            for n in nodes[:24]
        ],
        "updated_at": wf.get("updatedAt") or wf.get("updated_at") or "",
    }


def list_workflows() -> list[dict[str, Any]]:
    if _is_dev() and not os.environ.get("N8N_BASE_URL"):
        return [_summ(w) for w in _dev_store()["workflows"].values()][:MAX_LIST]
    out = _request("GET", "/api/v1/workflows?limit=50")
    return [_summ(w) for w in (out.get("data") or [])][:MAX_LIST]


def get_workflow(workflow_id: str) -> dict[str, Any]:
    if _is_dev() and not os.environ.get("N8N_BASE_URL"):
        wf = _dev_store()["workflows"].get(str(workflow_id))
        if wf is None:
            raise N8NError(f"no such workflow: {workflow_id}")
        return _summ(wf)
    return _summ(_request("GET", f"/api/v1/workflows/{workflow_id}"))


def create_workflow(
    *, name: str, nodes: list[dict], connections: dict,
    created_by: str,
) -> dict[str, Any]:
    """Create a workflow on the enterprise instance.

    Provenance is structural: the name carries an ``alex:<user>`` tag so
    the n8n side can always attribute the workflow. Created INACTIVE —
    activation is a separate, deliberate step.
    """
    name = (name or "").strip()[:120]
    if not name:
        raise N8NError("a workflow needs a name")
    if not isinstance(nodes, list) or not nodes:
        raise N8NError("a workflow needs at least one node")
    if len(nodes) > 40:
        raise N8NError("more than 40 nodes — split the workflow")
    tagged = f"{name} [alex:{created_by}]"
    payload = {
        "name": tagged,
        "nodes": nodes,
        "connections": connections or {},
        "settings": {},
    }
    if _is_dev() and not os.environ.get("N8N_BASE_URL"):
        wf_id = f"wf-{uuid.uuid4().hex[:8]}"
        _dev_store()["workflows"][wf_id] = {
            "id": wf_id, "active": False, **payload,
        }
        return {"id": wf_id, "name": tagged, "active": False}
    out = _request("POST", "/api/v1/workflows", payload)
    return {
        "id": str(out.get("id") or ""),
        "name": str(out.get("name") or tagged),
        "active": bool(out.get("active")),
    }


def set_active(workflow_id: str, active: bool) -> dict[str, Any]:
    if _is_dev() and not os.environ.get("N8N_BASE_URL"):
        wf = _dev_store()["workflows"].get(str(workflow_id))
        if wf is None:
            raise N8NError(f"no such workflow: {workflow_id}")
        wf["active"] = bool(active)
        return {"id": workflow_id, "active": bool(active)}
    verb = "activate" if active else "deactivate"
    out = _request("POST", f"/api/v1/workflows/{workflow_id}/{verb}")
    return {"id": workflow_id, "active": bool(out.get("active", active))}


def run_workflow(
    workflow_id: str, payload: dict | None = None
) -> dict[str, Any]:
    """Execute a workflow now and return the execution handle.

    Self-hosted n8n builds differ on the execution route; try the newer
    ``/execute`` then the older ``/run``, and say plainly when neither
    exists (webhook-triggered workflows are run by calling their
    webhook, which is a different, operator-published URL).
    """
    if _is_dev() and not os.environ.get("N8N_BASE_URL"):
        store = _dev_store()
        wf = store["workflows"].get(str(workflow_id))
        if wf is None:
            raise N8NError(f"no such workflow: {workflow_id}")
        ex_id = f"ex-{uuid.uuid4().hex[:8]}"
        store["executions"][ex_id] = {
            "id": ex_id, "workflow_id": str(workflow_id),
            "status": "success", "started_at": time.time(),
            "finished": True,
            "data": {"echo": payload or {}},
        }
        return {"execution_id": ex_id, "status": "success"}
    last: N8NError | None = None
    for route in (f"/api/v1/workflows/{workflow_id}/execute",
                  f"/api/v1/workflows/{workflow_id}/run"):
        try:
            out = _request("POST", route, {"data": payload or {}})
            ex = out.get("data") or out
            return {
                "execution_id": str(ex.get("executionId") or ex.get("id") or ""),
                "status": str(ex.get("status") or "started"),
            }
        except N8NError as exc:
            last = exc
            if "404" not in str(exc):
                raise
    raise N8NError(
        "this n8n build exposes no execute route for API-triggered runs; "
        f"use the workflow's webhook trigger instead ({last})"
    )


def execution(execution_id: str) -> dict[str, Any]:
    if _is_dev() and not os.environ.get("N8N_BASE_URL"):
        ex = _dev_store()["executions"].get(str(execution_id))
        if ex is None:
            raise N8NError(f"no such execution: {execution_id}")
        return {
            "id": ex["id"], "workflow_id": ex["workflow_id"],
            "status": ex["status"], "finished": ex["finished"],
        }
    out = _request("GET", f"/api/v1/executions/{execution_id}")
    return {
        "id": str(out.get("id") or execution_id),
        "workflow_id": str(out.get("workflowId") or ""),
        "status": str(out.get("status") or ("success" if out.get("finished") else "running")),
        "finished": bool(out.get("finished")),
    }


def health() -> dict[str, Any]:
    """Whether the arm is connected, without touching credentials."""
    configured = bool(os.environ.get("N8N_BASE_URL")) and bool(
        os.environ.get("N8N_API_KEY")
    )
    return {
        "configured": configured or _is_dev(),
        "mode": "dev-fake" if (_is_dev() and not os.environ.get("N8N_BASE_URL"))
        else "live",
    }
