"""
Internal APIs: governed read/act on third-party REST APIs (PlanHub,
Ledger Pro, etc.). The agent calls these through the `api_read` and
`api_act` tools.

Security:
  * The agent never holds a long-lived token. Every call is OBO: the
    agent receives a per-user, per-integration access token via the
    `user_token_provider` callable supplied by the app backend.
  * The `api_act` tool is two-phase. The first call stages the act and
    returns a confirmation token; the user's click on the confirmation
    card in the UI is what actually triggers the POST.
  * All URLs are validated against the integration's `base_url` so a
    model that produces `../admin` cannot reach a sibling path.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable
from urllib.parse import unquote, urljoin, urlparse

import httpx

# Pending two-phase api_act confirmations. Tokens are random, single-use,
# expiring, and bound to the exact (integration, path, body) so a staged
# benign act cannot be swapped for a different one at execute time.
_PENDING_ACTS: dict[str, dict] = {}
_ACT_CONFIRM_TTL_S = 600.0


def _act_binding(integration_id: str, path: str, body: dict) -> str:
    payload = json.dumps(body, sort_keys=True, default=str)
    raw = f"{integration_id}\x00{path}\x00{payload}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

log = logging.getLogger("agent.internal_apis")


def _is_under(url: str, base: str) -> bool:
    """True iff `url` is `base` or a deeper path under it. Both scheme and
    host must match exactly, AND the URL path must be `base.path` or a
    path under it."""
    try:
        u = urlparse(url)
        b = urlparse(base)
    except ValueError:
        return False
    if u.scheme != b.scheme or u.netloc != b.netloc:
        return False
    # Decode before the prefix check, and reject dot segments: the
    # server sees `%2e%2e` as `..`, so comparing the encoded form would
    # approve a path that resolves outside the integration's base.
    base_path = unquote(b.path or "").rstrip("/")
    url_path = unquote(u.path or "")
    if any(seg in ("..", ".") for seg in url_path.split("/")):
        return False
    if not base_path:
        return True
    if url_path == base_path or url_path.startswith(base_path + "/"):
        return True
    return False


def api_read(
    integration: dict,
    path: str,
    *,
    user_token_provider: Callable,
    timeout: float = 15.0,
) -> str:
    """Read from an internal API as the user. Returns the response body
    truncated to 12k characters.

    `integration` is a row from `media.control.tool_integrations`. The
    `read_capability` and `act_capability` columns tell the gate which
    ACL entry to check.
    """
    base = integration.get("base_url") or ""
    url = urljoin(base.rstrip("/") + "/", path.lstrip("/"))
    if not _is_under(url, base):
        return f"[error] path {path!r} escapes the integration's base URL"
    tok = user_token_provider(integration.get("integration_id") or "")
    headers = {
        "Authorization": f"Bearer {tok}",
        "Accept": "application/json",
        "User-Agent": "Alex-Agent/2.0",
    }
    try:
        with httpx.Client(timeout=timeout, follow_redirects=False) as client:
            r = client.get(url, headers=headers)
    except httpx.HTTPError as e:
        return f"[error] GET {url} failed: {e}"
    if r.status_code >= 400:
        return f"[error] HTTP {r.status_code} from {url}"
    return r.text[:12_000]


def api_act(
    integration: dict,
    path: str,
    body: dict,
    *,
    user_token_provider: Callable,
    user_email: str = "",
    confirm_token: str | None = None,
    timeout: float = 15.0,
) -> dict:
    """Act on an internal API as the user.

    Two-phase: if `confirm_token` is None, the act is STAGED and a
    token is returned. The user's click on the UI's confirmation card
    calls this function again with the same `body` and the token; only
    then is the POST actually performed.

    The token is bound to `user_email` when supplied: a token minted
    for one principal is not redeemable by another. The staging result
    carries only an opaque card id plus the token — the card id is
    safe to hand to the model (it can redeem nothing), while the token
    itself is delivered by the caller to the *authenticated UI route*
    and never back into the model's tool result. A token in the tool
    result is a token the model can hand itself in the next call.
    """
    base = integration.get("base_url") or ""
    url = urljoin(base.rstrip("/") + "/", path.lstrip("/"))
    if not _is_under(url, base):
        return {"error": f"path {path!r} escapes the integration's base URL"}
    requires_confirm = bool(integration.get("act_requires_confirm", True))
    integ_id = str(integration.get("integration_id") or "")
    if requires_confirm and not confirm_token:
        # Phase 1 — STAGE. Mint a random, single-use, expiring token bound
        # to THIS exact (integration, path, body, user) so the model cannot
        # stage a benign act and execute a different one with the token,
        # and one user's approval cannot redeem another's act.
        import secrets

        tok = secrets.token_urlsafe(24)
        _PENDING_ACTS[tok] = {
            "expires": time.time() + _ACT_CONFIRM_TTL_S,
            "binding": _act_binding(integ_id, path, body),
            "user_email": user_email,
        }
        return {
            "staged": True,
            "awaiting_confirmation": True,
            # The card id is what the model and the thread see. It is
            # not a credential: phase 2 accepts only the token.
            "card_id": tok[:8],
            "confirmation_token": tok,
            "summary": f"Will POST to {integ_id}:{path}",
            "note": (
                "This action is staged and requires the user's "
                "confirmation in the UI. You cannot confirm it yourself."
            ),
        }
    if requires_confirm:
        # Phase 2 — verify the token: must exist, be unexpired, match THIS
        # act's binding AND this caller's identity, and be consumed exactly
        # once.
        import secrets as _secrets

        pending = _PENDING_ACTS.get(confirm_token or "")
        valid = (
            pending is not None
            and pending["expires"] > time.time()
            and (not pending.get("user_email") or not user_email or
                 pending["user_email"] == user_email)
            and _secrets.compare_digest(
                pending["binding"], _act_binding(integ_id, path, body)
            )
        )
        if not valid:
            # Do NOT consume the token on a failed attempt — a mismatched
            # body must not let anyone burn a legitimate pending approval.
            return {"error": "invalid or expired confirmation_token"}
        # Single use: consume only on a valid match.
        _PENDING_ACTS.pop(confirm_token, None)
    tok = user_token_provider(integration.get("integration_id") or "")
    headers = {
        "Authorization": f"Bearer {tok}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "Alex-Agent/2.0",
    }
    try:
        with httpx.Client(timeout=timeout, follow_redirects=False) as client:
            r = client.post(url, json=body, headers=headers)
    except httpx.HTTPError as e:
        return {"error": f"POST {url} failed: {e}"}
    if r.status_code >= 400:
        return {
            "error": f"HTTP {r.status_code} from {url}",
            "status_code": r.status_code,
        }
    return {"ok": True, "status_code": r.status_code, "body": r.text[:12_000]}

# The LangChain tool factory that used to live here has been removed.
# It wrapped these functions directly, so a LangChain caller got them
# with no ACL check, no governor, no constitution and no audit row —
# and one of them wrote to an internal API. agent/langchain_tools.py
# exports the same capability through VM._invoke_tool instead, which
# is where all of that lives. An ungoverned export sitting beside a
# governed one is a trap, not a convenience.
