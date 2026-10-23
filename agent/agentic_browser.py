"""
Agentic browser â€” the high-level operations Alex performs on the web.

Each public function is a thin wrapper that:

* Looks up the right :class:`agent.browser_os.BrowserOS` backend.
* Looks up the right :class:`agent.session_manager.Session` (for
  identity + audit).
* Calls the backend operation.
* Emits a :class:`alex.events.Event` so the SSE stream shows what the
  agent did.
* Audit-logs to ``media.alex.agent_actions`` (constitution A3).

The public functions are wrapped as :class:`agent.omnigent.FunctionTool`
in :mod:`agent.browser_tools`.  The :mod:`agent.guardrails` gate
sees them and resolves the capability through the dual-ACL
view/table fallback just like every other tool.

A destructive operation (``submit_form``) emits a
``confirmation_required`` SSE event instead of navigating, and the
function returns a ``{"awaiting_confirmation": True, ...}`` payload.
The agent's caller is expected to re-invoke with a
``confirmation_token`` once the user has approved (constitution A8).
"""

from __future__ import annotations

import logging
import secrets
import time
from typing import Any

from alex.events import (
    artifact_event,
    confirmation_event,
    error_event,
    tool_event,
)
from alex.testing import get_fakes

logger = logging.getLogger(__name__)

# Pending two-phase submits: (session_id, tab_id) -> staged confirmation.
# Tokens are random, single-use, expiring, and bound to the staging user;
# the agent can never predict or reuse one (constitution A8).
_PENDING_CONFIRMATIONS: dict[tuple[str, str], dict[str, Any]] = {}
_CONFIRMATION_TTL_S = 600.0


def _get_browser_os(user_email: str | None = None):
    """Return the configured :class:`BrowserOS` backend.

    The VM registers its browser OS on the fakes container under
    ``browser_os_per_user`` (a ``dict[email, BrowserOS]``); this
    helper resolves the right instance for the calling user.  In
    production the same code path returns a Playwright-backed
    instance.  In tests the singleton is the per-test fakes'
    ``browser_os`` (or a per-user entry the VM registered).
    """

    fakes = get_fakes()
    if user_email:
        per_user = getattr(fakes, "browser_os_per_user", None) or {}
        bos = per_user.get(user_email)
        if bos is not None:
            return bos
    bos = getattr(fakes, "browser_os", None)
    if bos is None:
        # Lazy-initialise a default in-memory backend.
        from agent.browser_os import browser_os_for_mode

        bos = browser_os_for_mode()
        fakes.browser_os = bos
    return bos


def _get_audit_log():
    """Return the configured audit log callable.

    The VM registers its audit log on the fakes container under
    ``audit_log`` so the browser tools here can write to the same
    store as the rest of the VM.  In production the audit log is
    the Databricks SDK's statement-execution client; in dev it is
    the in-memory warehouse-backed writer.
    """

    return getattr(get_fakes(), "audit_log", None) or _default_audit


def _default_audit(**kwargs: Any) -> None:
    wh = get_fakes().warehouse
    try:
        import json as _json
        import uuid as _uuid

        aid = str(_uuid.uuid4())
        outcome = str(kwargs.get("outcome", ""))
        # Populate both column families of media.alex.agent_actions
        # (legacy id/actor/action_type/... and harness action_id/tool/...)
        # so the governance DDL's NOT NULLs and skill_reflection's
        # coalesce() reads are both satisfied.
        wh.execute(
            "INSERT INTO media.alex.agent_actions "
            "(id, actor, action_type, tool_name, tool_args, outcome, "
            " action_id, session_id, tool, capability, user_email, instance_scope, "
            " input_json, output_json, ok, ts) "
            "VALUES (:aid, :u, :atype, :tool, :in, :outcome, "
            " :aid, :sid, :tool, :cap, :u, :scope, :in, :out, :ok, current_timestamp())",
            {
                "aid": aid,
                "atype": "tool_call",
                "sid": kwargs.get("session_id", ""),
                "tool": kwargs.get("tool", ""),
                "cap": f"alex:tool:{kwargs.get('tool', '')}",
                "u": kwargs.get("user_email", ""),
                "scope": "",
                "in": _json.dumps(kwargs.get("args") or {})[:2000],
                "out": outcome,
                # CHECK constraint allows ok/denied/error/revoked/started;
                # a staged (awaiting_confirmation) submit is 'started'.
                "outcome": (
                    "ok"
                    if outcome.startswith("ok")
                    else "started"
                    if outcome == "awaiting_confirmation"
                    else "denied"
                ),
                "ok": 1 if outcome.startswith("ok") else 0,
            },
        )
    except Exception:  # noqa: BLE001
        logger.exception(
            "audit log write failed; agent_actions row missing for %s",
            kwargs.get("tool", "?"),
        )


def _audit(
    *, session_id: str, user_email: str, tool: str, args: dict, outcome: str
) -> None:
    """Append a row to ``media.alex.agent_actions`` (constitution A3)."""
    _get_audit_log()(
        session_id=session_id,
        user_email=user_email,
        tool=tool,
        args=args,
        outcome=outcome,
    )


def _emit(event, session_id: str | None = None) -> None:
    """Best-effort SSE emit.

    ``session_id`` stamps the event so the SSE route can route/filter it to
    the owning session only â€” without it, a process-global sink fanned every
    browser event to every open stream, leaking one user's URLs, extracted
    text, and confirmation tokens to another. The per-session subscriber
    registry lives on the fakes container under ``browser_subscribers``
    (``{session_id: set(callbacks)}``); the legacy single ``sse_sink`` is
    kept for the test recorder and the VM pump.
    """
    if session_id and getattr(event, "session_id", None) in (None, ""):
        event.session_id = session_id
    fakes = get_fakes()
    sink = getattr(fakes, "sse_sink", None)
    if sink is not None:
        try:
            sink(event)
        except Exception:  # noqa: BLE001
            logger.exception("sse sink raised; swallowing")
    # Fan out to the subscribers of THIS event's session only.
    subs = getattr(fakes, "browser_subscribers", None)
    if subs is not None and event.session_id:
        for cb in list(subs.get(event.session_id, ()) or ()):
            try:
                cb(event)
            except Exception:  # noqa: BLE001
                logger.exception("browser subscriber raised; swallowing")
    # Also fan out to the VM's pump (if one is registered).
    vm_pump = getattr(fakes, "current_vm_pump", None)
    if vm_pump is not None:
        try:
            vm_pump.emit(event)
        except Exception:  # noqa: BLE001
            logger.exception("vm pump raised; swallowing")


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------


def browser_navigate(
    *, session_id: str, tab_id: str, url: str, user_email: str, instance_scope: str = ""
) -> dict[str, Any]:
    """Navigate the agent-controlled tab to *url*."""
    bos = _get_browser_os()
    try:
        tab = bos.navigate(session_id, tab_id, url=url, user_email=user_email)
    except PermissionError as exc:
        _emit(error_event("denied", str(exc)), session_id)
        _audit(
            session_id=session_id,
            user_email=user_email,
            tool="browser_navigate",
            args={"url": url},
            outcome="denied",
        )
        return {"ok": False, "error": str(exc)}
    _emit(
        tool_event(
            "browser_navigate",
            {"url": url},
            {"title": tab.title, "status": tab.last_status},
        ),
        session_id,
    )
    _audit(
        session_id=session_id,
        user_email=user_email,
        tool="browser_navigate",
        args={"url": url},
        outcome="ok",
    )
    return {
        "ok": True,
        "tab_id": tab_id,
        "url": tab.url,
        "title": tab.title,
        "status": tab.last_status,
    }


def browser_click(
    *,
    session_id: str,
    tab_id: str,
    selector: str,
    user_email: str,
    instance_scope: str = "",
) -> dict[str, Any]:
    """Click an element by selector (link text or ``href``)."""
    bos = _get_browser_os()
    try:
        tab = bos.click(session_id, tab_id, selector=selector, user_email=user_email)
    except PermissionError as exc:
        _emit(error_event("denied", str(exc)), session_id)
        _audit(
            session_id=session_id,
            user_email=user_email,
            tool="browser_click",
            args={"selector": selector},
            outcome="denied",
        )
        return {"ok": False, "error": str(exc)}
    except LookupError as exc:
        _audit(
            session_id=session_id,
            user_email=user_email,
            tool="browser_click",
            args={"selector": selector},
            outcome="no_match",
        )
        return {"ok": False, "error": str(exc)}
    _emit(
        tool_event(
            "browser_click",
            {"selector": selector},
            {"url": tab.url, "title": tab.title},
        ),
        session_id,
    )
    _audit(
        session_id=session_id,
        user_email=user_email,
        tool="browser_click",
        args={"selector": selector},
        outcome="ok",
    )
    return {"ok": True, "url": tab.url, "title": tab.title}


def browser_type(
    *,
    session_id: str,
    tab_id: str,
    selector: str,
    text: str,
    user_email: str,
    instance_scope: str = "",
) -> dict[str, Any]:
    """Type *text* into a form field identified by *selector*."""
    bos = _get_browser_os()
    try:
        tab = bos.type_text(
            session_id,
            tab_id,
            selector=selector,
            text=text,
            user_email=user_email,
        )
    except PermissionError as exc:
        _emit(error_event("denied", str(exc)), session_id)
        _audit(
            session_id=session_id,
            user_email=user_email,
            tool="browser_type",
            args={"selector": selector, "text_len": len(text)},
            outcome="denied",
        )
        return {"ok": False, "error": str(exc)}
    except LookupError as exc:
        _audit(
            session_id=session_id,
            user_email=user_email,
            tool="browser_type",
            args={"selector": selector},
            outcome="no_match",
        )
        return {"ok": False, "error": str(exc)}
    _emit(
        tool_event(
            "browser_type",
            {"selector": selector, "text_len": len(text)},
            {"form_keys": list(tab.form)},
        ),
        session_id,
    )
    _audit(
        session_id=session_id,
        user_email=user_email,
        tool="browser_type",
        args={"selector": selector, "text_len": len(text)},
        outcome="ok",
    )
    return {"ok": True, "form_keys": list(tab.form)}


def browser_extract(
    *,
    session_id: str,
    tab_id: str,
    selector: str,
    user_email: str,
    instance_scope: str = "",
) -> dict[str, Any]:
    """Extract the page text and form state, with PII redaction."""
    bos = _get_browser_os()
    payload = bos.extract(session_id, tab_id, selector=selector, user_email=user_email)
    _emit(artifact_event(f"browser://{tab_id}/extract", kind="page_text"), session_id)
    _audit(
        session_id=session_id,
        user_email=user_email,
        tool="browser_extract",
        args={"selector": selector},
        outcome="ok",
    )
    return payload


def browser_screenshot(
    *,
    session_id: str,
    tab_id: str,
    user_email: str,
    instance_scope: str = "",
) -> dict[str, Any]:
    """Capture a screenshot of the current tab.

    In dev, returns a placeholder.  In production, the
    :class:`PlaywrightBrowserOS` returns ``{"png": <bytes>}`` which
    the FastAPI route serialises into a ``data:image/png;base64,...``
    URL inside an ``artifact`` SSE event.
    """
    bos = _get_browser_os()
    payload = bos.screenshot(session_id, tab_id, user_email=user_email)
    _emit(artifact_event(f"browser://{tab_id}/screenshot", kind="png"), session_id)
    _audit(
        session_id=session_id,
        user_email=user_email,
        tool="browser_screenshot",
        args={},
        outcome="ok",
    )
    return payload


def browser_submit(
    *,
    session_id: str,
    tab_id: str,
    selector: str,
    confirmation_token: str | None,
    user_email: str,
    instance_scope: str = "",
) -> dict[str, Any]:
    """Submit a form.  Two-phase: emit a ``confirmation_required`` event first.

    The first invocation (no ``confirmation_token``) emits the
    confirmation request and returns ``{"awaiting_confirmation": True}``.
    A subsequent invocation with a valid ``confirmation_token`` actually
    submits.  This is the implementation of constitution A8.
    """
    bos = _get_browser_os()
    if not confirmation_token:
        # Stage and ask.
        try:
            bos.submit_form(
                session_id, tab_id, selector=selector, user_email=user_email
            )
        except (PermissionError, LookupError) as exc:
            _emit(
                error_event(
                    "denied" if isinstance(exc, PermissionError) else "not_found",
                    str(exc),
                ),
                session_id,
            )
            return {"ok": False, "error": str(exc)}
        token = secrets.token_urlsafe(24)
        _PENDING_CONFIRMATIONS[(session_id, tab_id)] = {
            "token": token,
            "selector": selector,
            "user_email": user_email,
            "expires": time.time() + _CONFIRMATION_TTL_S,
        }
        _emit(
            confirmation_event(
                token=token,
                summary=f"Submit form on tab {tab_id}",
                body={"selector": selector, "session_id": session_id, "tab_id": tab_id},
            ),
            session_id,
        )
        _audit(
            session_id=session_id,
            user_email=user_email,
            tool="browser_submit",
            args={"selector": selector, "phase": "staged"},
            outcome="awaiting_confirmation",
        )
        # The token goes to the UI through the SSE confirmation event â€”
        # NOT through this tool result. A token the model can read is a
        # token the model can hand back on the next call, which turns
        # two-phase approval into the model approving itself.
        return {"ok": True, "awaiting_confirmation": True}
    # Second phase: the user approved via the UI, which redeemed the
    # token through the authenticated HTTP route. Here the model can
    # only report that phase 1 happened; the actual submit is executed
    # now, for real, on the backend.
    pending = _PENDING_CONFIRMATIONS.get((session_id, tab_id))
    valid = (
        pending is not None
        and pending["expires"] > time.time()
        and pending["user_email"] == user_email
        and secrets.compare_digest(pending["token"], confirmation_token)
    )
    if not valid:
        _emit(
            error_event(
                "invalid_confirmation_token", "confirmation_token did not match"
            ),
            session_id,
        )
        return {"ok": False, "error": "invalid_confirmation_token"}
    # Single use: a token can never confirm twice.
    del _PENDING_CONFIRMATIONS[(session_id, tab_id)]
    try:
        bos.execute_submit(
            session_id, tab_id, selector=pending["selector"], user_email=user_email
        )
    except (PermissionError, LookupError) as exc:
        _emit(
            error_event(
                "denied" if isinstance(exc, PermissionError) else "not_found",
                str(exc),
            ),
            session_id,
        )
        return {"ok": False, "error": str(exc)}
    _audit(
        session_id=session_id,
        user_email=user_email,
        tool="browser_submit",
        args={"selector": pending["selector"], "phase": "executed"},
        outcome="ok",
    )
    return {"ok": True, "submitted": True, "selector": pending["selector"]}


def browser_list_tabs(
    *,
    session_id: str,
    user_email: str,
    instance_scope: str = "",
) -> dict[str, Any]:
    """List the tabs in the current browser session."""
    bos = _get_browser_os()
    tabs = bos.list_tabs(session_id, user_email=user_email)
    return {
        "ok": True,
        "tabs": [
            {
                "tab_id": t.tab_id,
                "url": t.url,
                "title": t.title,
                "agent_piloting": t.agent_piloting,
            }
            for t in tabs
        ],
    }


def browser_close_tab(
    *,
    session_id: str,
    tab_id: str,
    user_email: str,
    instance_scope: str = "",
) -> dict[str, Any]:
    """Close a single tab in the current browser session."""
    bos = _get_browser_os()
    bos.close_tab(session_id, tab_id, user_email=user_email)
    _audit(
        session_id=session_id,
        user_email=user_email,
        tool="browser_close_tab",
        args={"tab_id": tab_id},
        outcome="ok",
    )
    return {"ok": True, "tab_id": tab_id}


def browser_open_in_user_session(
    *,
    session_id: str,
    tab_id: str,
    mode: str,
    user_email: str,
    instance_scope: str = "",
) -> dict[str, Any]:
    """Mirror the agent's tab to the user's browser.

    ``mode`` is one of ``read`` (user watches), ``co-drive`` (user
    takes the wheel; agent pauses), ``yield`` (user gives the wheel
    back to the agent).
    """
    bos = _get_browser_os()
    tab = bos.set_user_control(session_id, tab_id, mode=mode, user_email=user_email)
    _audit(
        session_id=session_id,
        user_email=user_email,
        tool="browser_open_in_user_session",
        args={"mode": mode, "tab_id": tab_id},
        outcome="ok",
    )
    return {
        "ok": True,
        "tab_id": tab.tab_id,
        "agent_piloting": tab.agent_piloting,
        "mode": mode,
    }
