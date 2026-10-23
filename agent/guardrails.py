"""
Dual-ACL gate: every consequential tool the agent can call goes through
`guarded(...)` which:

  1. Checks the user is in `media.control.effective_employee_acl` for the
     capability the tool requires.
  2. Checks the user's current `instance_acl` ceiling (this is per-row;
     the gate runs at tool-bind time, not in SQL).
  3. (Optionally) runs the constitutional interceptor for the tool, which
     enforces prohibitions and procedure rules from the active articles.

The agent endpoint wraps every UC toolkit tool, every browser tool,
every internal API tool, and every workspace tool with `guarded(...)`.
The result is a function that looks identical to the original but
refuses to run if the gate fails.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from functools import wraps
from typing import Any

from alex.errors import Denied

log = logging.getLogger("agent.guardrails")


# Tools that the agent SP is *always* allowed to call (it owns the
# gateway; the per-user dual-ACL does not apply). `guarded` asks about
# CAPABILITIES, not tool names, so both spellings are listed: the old
# name-only set never matched (`"agent.log"` is not
# `"log_agent_action"`), making the bypass dead code.
AGENT_INTERNAL_TOOLS = frozenset(
    {
        "log_agent_action",
    }
)
AGENT_INTERNAL_CAPABILITIES = frozenset(
    {
        "agent.log",
    }
)


# Map tool name -> required capability. The agent uses this to ask the
# warehouse "is this user allowed to call this tool?" It must be kept in
# sync with the agent's `TOOL_DESCRIPTIONS` and the UC function surface.
TOOL_CAPABILITY: dict[str, str] = {
    "create_task": "tasks.create",
    "update_task_status": "tasks.update",
    "list_my_tasks": "tasks.read",
    "summarize_email": "email.read",
    "draft_email_reply": "email.draft",
    "send_email": "email.send",
    "publish_post": "publish.post",
    "check_rights": "content.read",
    "log_agent_action": "agent.log",
    "search_media_library": "content.read",
    "audience_analytics": "audience.read",
    "run_python": "compute.python",
    "pip_install": "compute.install",
    "system_install": "system.install",
    "vm_exec": "compute.shell",
    "bash": "compute.bash",
    "view": "files.read",
    "create_file": "files.write",
    "download_file": "compute.download",
    "str_replace": "files.write",
    "present_files": "files.read",
    # Desktop control (agent.desktop).
    "desktop_screenshot": "desktop.read",
    "desktop_find_text": "desktop.read",
    "desktop_click": "desktop.act",
    "desktop_type": "desktop.act",
    "desktop_key": "desktop.act",
    "desktop_hotkey": "desktop.act",
    "desktop_scroll": "desktop.act",
    # Home automation (agent.home).
    "home_status": "home.read",
    "home_control": "home.act",
    "find_tools": "knowledge.read",
    "create_spreadsheet": "office.write",
    "read_spreadsheet": "office.read",
    "create_presentation": "office.write",
    "create_document": "office.write",
    "browser_open": "browser.use",
    "browser_read": "browser.read",
    "browser_fill": "browser.act",
    "browser_click": "browser.act",
    "browser_submit": "browser.act",
    # New Browser OS tool names (see agent.browser_tools).
    "browser_navigate": "browser.use",
    "browser_type": "browser.act",
    "browser_extract": "browser.read",
    "browser_screenshot": "browser.read",
    "browser_list_tabs": "browser.read",
    "browser_close_tab": "browser.use",
    "browser_open_in_user_session": "browser.use",
    "api_read": "internal_api.read",
    "api_act": "internal_api.act",
    "trigger_workflow": "workflow.trigger",
    "n8n_list_workflows": "workflow.read",
    "n8n_create_workflow": "workflow.create",
    "n8n_run_workflow": "workflow.trigger",
    "n8n_execution_status": "workflow.read",
    "stage_action": "agent.stage",
    # Enterprise RAG + vector knowledge graph (agent.rag /
    # agent.enterprise_graph).
    "search_knowledge": "knowledge.read",
    "list_knowledge": "knowledge.read",
    "ingest_document": "knowledge.write",
    "ingest_workspace_file": "knowledge.write",
    "graph_entity": "memory.read",
    "graph_neighbors": "memory.read",
    "graph_link": "memory.write",
    "graph_about_me": "memory.read",
    # Standing orders + the best-practice library.
    "list_standing_orders": "orders.read",
    "run_standing_order": "orders.run",
    "search_practices": "practices.read",
    "propose_practice": "practices.propose",
    "graph_query": "memory.read",
    # Memory + KG write/read tools (agent.tools.impls).
    "remember": "memory.write",
    "recall": "memory.read",
    "add_edge": "memory.write",
    # Taught routines (agent.tools.impls / agent.learn_with_you).
    "run_routine": "learning.read",
    "delegate_subagents": "agents.delegate",
    "undo_file_change": "files.write",
    "plan_task": "tasks.update",
    "check_plan_step": "tasks.update",
    "check_acceptance": "tasks.update",
    "ask_user": "tasks.update",
    # Inbox / email tools (agent.tools.impls).
    "read_inbox": "email.read",
    "search_email": "email.read",
    # LSP code intelligence (agent.lsp_tool).
    "code_outline": "code.read",
    "code_definition": "code.read",
    "code_references": "code.read",
    "code_hover": "code.read",
    "code_diagnostics": "code.read",
    "code_search_symbols": "code.read",
}


_EFFECTIVE_EMPLOYEE_ACL_SQL = """
    SELECT capability, expires_at FROM media.control.effective_employee_acl
     WHERE user_email = :u AND capability = :c
"""

_EMPLOYEE_ACL_SQL = """
    SELECT capability, expires_at FROM media.control.employee_acl
     WHERE user_email = :u AND capability = :c
"""


def _grant_is_live(row) -> bool:
    """Whether a base-table grant row is still in force.

    The effective view excludes expired grants in SQL; the base-table
    fallback used to ignore expires_at entirely, so an expired grant
    kept working whenever the view was empty or unavailable — the
    fallback quietly out-granted the source of truth. Unparseable
    expiry fails CLOSED: an expiry someone bothered to set, in a shape
    we cannot read, is not permission."""
    raw = None
    if isinstance(row, dict):
        raw = row.get("expires_at")
    if raw in (None, ""):
        return True
    from datetime import datetime, timezone

    if isinstance(raw, datetime):
        stamp = raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
        return stamp > datetime.now(timezone.utc)
    try:
        stamp = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return False
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp > datetime.now(timezone.utc)

_INSTANCE_ACL_SQL = """
    SELECT capability FROM media.control.instance_acl
     WHERE instance_scope = :s AND capability = :c
"""


def _user_has_capability(
    sql, user_email: str, instance_scope: str, capability: str
) -> bool:
    """Check both halves of the dual-ACL.

    The instance ACL is keyed by an `instance_scope` string (a tenant
    id or an org unit). The agent receives this in the request context.

    In production we query the `effective_employee_acl` view (which
    UNIONs `employee_acl` and active `delegations`). For environments
    that don't have that view (e.g. dev), we fall back to the base
    `employee_acl` table; the dual-ACL property is preserved either way.
    """
    if not user_email:
        return False
    if capability in AGENT_INTERNAL_TOOLS or capability in AGENT_INTERNAL_CAPABILITIES:
        return True
    # The dual-ACL reads from `media.control.effective_employee_acl` in
    # production (a view that UNIONs employee_acl + active delegations).
    # In dev / tests the view may not exist; we try the view first, then
    # fall back to the base `employee_acl` table. Either way, the user
    # must have a row in the resolved source.
    employee_rows: list = []
    try:
        employee_rows = _try_select(
            sql,
            _EFFECTIVE_EMPLOYEE_ACL_SQL,
            {"u": user_email, "c": capability},
        )
    except Exception:
        employee_rows = []
    if not employee_rows:
        try:
            employee_rows = [
                r
                for r in _try_select(
                    sql,
                    _EMPLOYEE_ACL_SQL,
                    {"u": user_email, "c": capability},
                )
                if _grant_is_live(r)
            ]
        except Exception:
            employee_rows = []
    if not employee_rows:
        return False
    if instance_scope:
        try:
            instance_rows = _try_select(
                sql,
                _INSTANCE_ACL_SQL,
                {"s": instance_scope, "c": capability},
            )
        except Exception:
            instance_rows = []
        if not instance_rows:
            return False
    return True


def _try_select(sql, stmt: str, params: dict) -> list:
    try:
        return list(sql(stmt, params) or [])
    except Exception:
        return []


def guarded(
    sql: Callable,
    user_email: str,
    instance_scope: str,
    log_action: Callable,
    interceptor: Any | None = None,
) -> Callable:
    """Return a decorator that wraps a tool function with the dual-ACL gate
    and (optionally) the constitutional interceptor.

    Usage:
        gate = guarded(sql=sql, user_email=ident.email,
                       instance_scope="media-corp",
                       log_action=ledger.log, interceptor=ic)

        @gate
        def create_task(title, ...): ...

    Or, when the function's __name__ doesn't match the tool's canonical
    name (e.g. a lambda), pass the name explicitly:

        @gate
        def _create(...): ...   # canonical name is 'create_task'
        # the agent calls: gate(_create, tool_name='create_task')
    """

    def decorator(fn: Callable, tool_name: str | None = None) -> Callable:
        # Resolve the canonical tool name. The agent supplies it
        # explicitly via `gate(tool, tool_name="...")`. The fallback
        # chain covers ad-hoc tests with plain functions or lambdas.
        name = tool_name or getattr(fn, "name", None) or fn.__name__
        capability = TOOL_CAPABILITY.get(name, name)

        @wraps(fn)
        def wrapper(*args, **kwargs):
            # The interceptor's procedure rules inspect the tool's ACTUAL
            # argument dict (e.g. must_have_field 'asset_id'). Tool calls
            # arrive as kwargs, so pass them flat — the old
            # {"args":..,"kwargs":..} wrapper hid every field from those
            # checks, silently disabling field-level enforcement.
            if kwargs:
                tool_arg_dict = dict(kwargs)
            elif len(args) == 1 and isinstance(args[0], dict):
                tool_arg_dict = dict(args[0])
            else:
                tool_arg_dict = {}
            # 1) Dual-ACL (uses capability)
            if not _user_has_capability(sql, user_email, instance_scope, capability):
                try:
                    log_action(
                        action_type="denied",
                        target_id=capability,
                        detail=f"missing capability {capability}",
                        tool_name=capability,
                        tool_args=_stringify({"args": list(args), "kwargs": kwargs})[
                            :4000
                        ],
                    )
                except Exception:
                    log.exception("denial log failed")
                raise Denied(
                    f"user {user_email} lacks capability {capability} for tool {capability}",
                    details={"tool": capability, "capability": capability},
                )
            # 2) Constitutional interceptor (L2) — uses the tool name
            # (the model's canonical identifier), not the capability.
            if interceptor is not None:
                try:
                    interceptor.check(name, tool_arg_dict)
                except Exception as e:
                    try:
                        log_action(
                            action_type="law_block",
                            target_id=capability,
                            detail=str(e),
                            tool_name=capability,
                            tool_args=_stringify(
                                {"args": list(args), "kwargs": kwargs}
                            )[:4000],
                        )
                    except Exception:
                        log.exception("law block log failed")
                    raise
            # 3) Run the tool
            result = fn(*args, **kwargs)
            # 4) Record history for procedure rules on future calls
            if interceptor is not None:
                try:
                    interceptor.record(name, tool_arg_dict, result)
                except Exception:
                    log.exception("interceptor.record failed")
            return result

        wrapper.__name__ = capability
        wrapper._alex_capability = capability
        wrapper._alex_is_guarded = True
        return wrapper

    return decorator


def _stringify(x: Any) -> str:
    if isinstance(x, str):
        return x
    try:
        return json.dumps(x, default=str, ensure_ascii=False)
    except Exception:
        return str(x)
