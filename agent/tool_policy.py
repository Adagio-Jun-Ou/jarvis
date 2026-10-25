"""The risk-graded permission layer: allow / ask / deny per tool.

What the strong open harnesses converged on (OpenCode's permission
engine, OpenHands' ConfirmRisky, goose's approval modes): a declarative
policy that lets the cheap 95% of actions run at full speed and puts a
HUMAN gate — not a prompt suggestion — in front of the consequential
few. Here that gate is real in the place that needs it most: an
unattended board run parks on a one-click approve/deny in the task
thread (the elicitation machinery), and only an explicit approve lets
the action through.

Resolution order, most specific wins:

1. an operator rule for the TOOL (``tool_policy.<tool_name>``);
2. an operator rule for the CAPABILITY (``tool_policy.<capability>``);
3. the default: capabilities the delegation catalog already marks as
   dual-approval (install/shell/download) plus the outward-facing acts
   (email.send, publish.post) default to ASK; everything else ALLOWS.

Operator rules live in ``media.control.workspace_policy`` — an
operator-writable table, never something the model or a request can
set. Values other than allow/ask/deny are ignored (a typo must not
widen anything, and must not brick a tool either — the default stands).

``ask`` semantics depend on where the run is:

* a **board-task session** (unattended) parks on
  :func:`agent.task_questions.request_approval` — fail closed;
* any other session (a supervised chat, a swarm worker) proceeds but
  the act is SAID: a governor-style warn on the stream and the audit
  row record that policy flagged it. The person is present and
  watching in chat; the boardroom-grade gate belongs on the runs
  nobody is watching. Operators who want the hard gate everywhere set
  the rule to ``deny``.

One exception is pinned by test rather than listed here: pip_install
(capability compute.install) is governed by its own approved-library
allowlist, so its default is ALLOW — two human gates on one curated
install would teach people to click through both.
"""

from __future__ import annotations

import logging
import time

log = logging.getLogger("agent.tool_policy")

ALLOW = "allow"
ASK = "ask"
DENY = "deny"

_VALID = frozenset({ALLOW, ASK, DENY})

#: Capabilities whose default is ASK. Grounded in existing governance:
#: the first four are exactly the delegable_capabilities rows marked
#: requires_second_approver in governance/delegation.sql; the last two
#: are the acts that leave the building.
DEFAULT_ASK_CAPABILITIES = frozenset({
    "compute.install",
    "system.install",
    "compute.shell",
    "compute.download",
    "email.send",
    "publish.post",
    # Acting on the person's own desktop.
    "desktop.act",
    # Touching the house.
    "home.act",
})

#: Tool-level default overrides (checked before the capability default,
#: after operator rules). pip_install carries compute.install but has
#: its own curated allowlist as the control.
DEFAULT_TOOL_RULES = {
    "pip_install": ALLOW,
}

_PREFIX = "tool_policy."
_TTL_S = 60.0
_cache: tuple[float, dict[str, str]] | None = None


def _sql(stmt: str, params: dict | None = None) -> list[dict]:
    from agent.tools.impls import _sql as impl_sql

    return impl_sql(stmt, params)


def _operator_rules() -> dict[str, str]:
    """Rules from workspace_policy, cached briefly. Fail STICKY: on a
    store outage, serve the last successful read — an operator's deny
    rules evaporating for a TTL because the control plane hiccuped is
    exactly "widening policy", which the module promises not to do.
    Only when there has never been a successful read do the built-in
    defaults stand (a fresh deploy with no table yet)."""
    global _cache
    now = time.time()
    if _cache is not None and now - _cache[0] < _TTL_S:
        return _cache[1]
    rules: dict[str, str] = {}
    try:
        rows = _sql(
            "SELECT name, value FROM media.control.workspace_policy", {}
        )
        for r in rows or []:
            name = str(r.get("name") or "")
            if not name.startswith(_PREFIX):
                continue
            value = str(r.get("value") or "").strip().lower()
            if value in _VALID:
                rules[name[len(_PREFIX):]] = value
            else:
                log.warning(
                    "workspace_policy %s has invalid value %r (ignored)",
                    name, value,
                )
        _cache = (now, rules)
    except Exception:  # noqa: BLE001
        # Sticky: keep serving the last good rules. A never-read store
        # falls back to defaults.
        if _cache is not None:
            log.warning(
                "workspace_policy read failed; serving the last good "
                "rules rather than widening policy", exc_info=True,
            )
            return _cache[1]
        log.warning(
            "workspace_policy read failed on first use; built-in "
            "defaults stand", exc_info=True,
        )
        _cache = (now, rules)
    return rules


def reset_cache() -> None:
    global _cache
    _cache = None


def decision(tool_name: str, capability: str) -> str:
    """allow | ask | deny for one tool call. Most specific rule wins."""
    rules = _operator_rules()
    for key in (tool_name, capability):
        if key in rules:
            return rules[key]
    if tool_name in DEFAULT_TOOL_RULES:
        return DEFAULT_TOOL_RULES[tool_name]
    if capability in DEFAULT_ASK_CAPABILITIES:
        return ASK
    return ALLOW


def approval_question(tool_name: str, args: dict | None) -> str:
    """The one-line question the crew sees on the board."""
    preview = ""
    for key in ("to", "url", "packages", "path", "command", "argv", "name"):
        if args and args.get(key):
            preview = f" ({key}: {str(args[key])[:120]})"
            break
    return (
        f"May I run {tool_name}{preview}? This action is policy-gated — "
        "approve to let it through, deny to skip it."
    )
