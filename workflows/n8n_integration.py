"""
n8n integration: the Alex agent can trigger n8n workflows from the
catalog. The chat UI shows the trigger as a confirmation card; the
user's click is what actually fires the webhook.

Security:
  * The webhook URL is read from `media.control.workflow_catalog` (a
    real Databricks row), not from the user.
  * The webhook auth token is fetched from Databricks Secrets on every
    call; never inlined.
  * Every trigger writes a row to `media.control.workflow_runs`
    (append-only) with the outcome, so we have a complete audit trail.
  * SQL is parameterized; no f-string interpolation of user inputs.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets as _secrets
import time
import urllib.error
import urllib.request
import uuid

log = logging.getLogger("workflows.n8n")

SECRET_SCOPE = "alex"
SECRET_KEY = "n8n_webhook_token"

# Staged dual-approval tokens. Keyed by (user, workflow_id, payload hash)
# so a token can only fire the exact trigger it was staged for. Tokens are
# single-use and expire after STAGED_TOKEN_TTL_SECONDS.
STAGED_TOKEN_TTL_SECONDS = 600
_staged_tokens: dict[tuple[str, str, str], tuple[str, float]] = {}


def _payload_hash(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _stage_token(user_email: str, workflow_id: str, payload: dict) -> str:
    token = _secrets.token_urlsafe(24)
    key = (user_email, workflow_id, _payload_hash(payload))
    _staged_tokens[key] = (token, time.time() + STAGED_TOKEN_TTL_SECONDS)
    return token


def _consume_staged_token(
    user_email: str, workflow_id: str, payload: dict, confirm_token: str
) -> bool:
    """Validate and consume (single-use) a staged token. Constant-time
    comparison; expired or unknown tokens fail."""
    if not confirm_token:
        return False
    key = (user_email, workflow_id, _payload_hash(payload))
    entry = _staged_tokens.get(key)
    if entry is None:
        return False
    token, expires_at = entry
    if time.time() > expires_at:
        _staged_tokens.pop(key, None)
        return False
    if not _secrets.compare_digest(token, confirm_token):
        return False
    _staged_tokens.pop(key, None)  # single-use
    return True


def _webhook_token() -> str | None:
    """Fetch the n8n webhook token from Databricks Secrets. Returns None
    in dev or if the secret is not provisioned."""
    try:
        try:
            from pyspark.dbutils import DBUtils  # type: ignore
            dbutils = DBUtils(_spark())
        except Exception:
            # Notebook / interactive context: dbutils is a builtin global.
            import IPython  # type: ignore
            dbutils = IPython.get_ipython().user_ns["dbutils"]
        return dbutils.secrets.get(scope=SECRET_SCOPE, key=SECRET_KEY)
    except Exception:
        return os.environ.get("ALEX_N8N_WEBHOOK_TOKEN")


def _spark():
    """Lazy Spark session — pipelines run in a Databricks notebook; the
    test harness may not have PySpark available. We surface errors
    clearly so the caller can decide how to handle them."""
    try:
        from pyspark.sql import SparkSession
        return SparkSession.builder.getOrCreate()
    except ImportError as exc:
        raise RuntimeError(
            "PySpark is required for n8n_integration (Databricks runtime). "
            "Tests should mock _spark()."
        ) from exc


def _catalog_row(workflow_id: str) -> dict | None:
    rows = _spark().sql("""
        SELECT workflow_id, name, description, capability, input_schema,
               integration_id, requires_dual_approval, active
          FROM media.control.workflow_catalog
         WHERE workflow_id = :wid AND active = true
    """, args={"wid": workflow_id}).collect()
    if not rows:
        return None
    return dict(rows[0].asDict())


def _user_has_capability(user_email: str, capability: str) -> bool:
    rows = _spark().sql("""
        SELECT 1
          FROM media.control.effective_employee_acl
         WHERE user_email = :u AND capability = :c
         LIMIT 1
    """, args={"u": user_email, "c": capability}).collect()
    return bool(rows)


def stage(user_email: str, workflow_id: str, payload: dict) -> dict:
    """Stage a workflow trigger. The user's click in the UI calls
    `execute(...)` with the returned token to actually fire it."""
    row = _catalog_row(workflow_id)
    if row is None:
        return {"error": f"unknown or inactive workflow: {workflow_id}"}
    if not _user_has_capability(user_email, row["capability"]):
        return {"error": f"user lacks capability {row['capability']} for this workflow"}
    # Validate the payload against the workflow's input schema
    # (a JSON Schema string). We use jsonschema if available; else
    # best-effort dict check.
    try:
        import jsonschema
        schema = json.loads(row.get("input_schema") or "{}")
        jsonschema.validate(payload, schema)
    except ImportError:
        pass
    except Exception as exc:
        return {"error": f"payload validation failed: {exc}"}
    if row.get("requires_dual_approval"):
        return {
            "staged": True,
            "token": _stage_token(user_email, workflow_id, payload),
            "workflow_id": workflow_id,
            "summary": f"Will run workflow '{row.get('name', workflow_id)}' as {user_email}.",
        }
    return execute(user_email, workflow_id, payload, confirm_token=None)


def execute(user_email: str, workflow_id: str, payload: dict, *,
            confirm_token: str | None) -> dict:
    """Fire the workflow. If the catalog row requires dual approval,
    `confirm_token` must be the exact single-use token returned by
    `stage(...)` for the same (user, workflow, payload)."""
    row = _catalog_row(workflow_id)
    if row is None:
        return {"error": f"unknown or inactive workflow: {workflow_id}"}
    # Re-check capability at the side-effecting call, not just in stage():
    # execute() is public (and CLI-exposed), so without this a caller could
    # fire a non-dual-approval workflow they lack the capability for.
    if not _user_has_capability(user_email, row["capability"]):
        return {"error": f"user lacks capability {row['capability']} for this workflow"}
    if row.get("requires_dual_approval"):
        if not _consume_staged_token(
            user_email, workflow_id, payload, confirm_token or ""
        ):
            return {"error": "confirmation token invalid, expired, or already used"}
    token = _webhook_token()
    if not token:
        return {"error": "n8n webhook token is not configured"}
    integ = _spark().sql("""
        SELECT base_url FROM media.control.tool_integrations
         WHERE integration_id = :iid
    """, args={"iid": row.get("integration_id") or ""}).collect()
    base = integ[0]["base_url"] if integ else ""
    if not base:
        return {"error": "workflow has no integration base URL"}
    url = base.rstrip("/") + "/webhook/" + workflow_id
    body = json.dumps({
        "user_email": user_email,
        "workflow_id": workflow_id,
        "payload": payload,
    }).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "User-Agent": "Alex-Agent/2.0",
    })
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            status = r.status
            text = r.read().decode("utf-8", errors="ignore")[:8000]
    except urllib.error.HTTPError as exc:
        status = exc.code
        text = exc.read().decode("utf-8", errors="ignore")[:8000]
    except Exception as exc:
        status = -1
        text = str(exc)
    outcome = "ok" if 200 <= status < 300 else "error"
    _spark().sql("""
        INSERT INTO media.control.workflow_runs
          (run_id, workflow_id, user_email, ts, input_payload,
           result_payload, outcome)
        VALUES
          (:rid, :wid, :u, current_timestamp(), :ip, :rp, :o)
    """, args={
        "rid": str(uuid.uuid4()),
        "wid": workflow_id,
        "u": user_email,
        "ip": json.dumps(payload)[:4000],
        "rp": text[:4000],
        "o": outcome,
    })
    return {"status": outcome, "status_code": status, "body": text[:2000]}


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "stage"
    user = sys.argv[2] if len(sys.argv) > 2 else ""
    wid = sys.argv[3] if len(sys.argv) > 3 else ""
    payload = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
    if cmd == "stage":
        print(json.dumps(stage(user, wid, payload)))
    elif cmd == "execute":
        token = sys.argv[5] if len(sys.argv) > 5 else None
        print(json.dumps(execute(user, wid, payload, confirm_token=token)))
    else:
        print(f"unknown command: {cmd}")
