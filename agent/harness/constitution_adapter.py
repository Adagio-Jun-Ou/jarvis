"""
ConstitutionAdapter — the bridge between Alex 2.0's
:class:`agent.constitution.ConstitutionalInterceptor` and the
Omnigent :class:`FunctionPolicy` shape.

The constitution interceptor is the canonical L2 layer; it owns
per-session state (article list, prior results, user identity,
session id, log fn).  The adapter exposes that interceptor as an
Omnigent :class:`FunctionPolicy` whose callable receives an
``event`` dict shaped by
:func:`agent.omnigent.policies._build_inner_event` and returns a
:class:`PolicyResult` (or dict) the policy dispatcher understands.

On :class:`ConstitutionalBlock` the adapter:

1. Emits a ``law_block`` :class:`alex.events.Event` through the
   VM's event pump (so the UI shows which rule fired).
2. Writes a row to ``media.control.constitution_violations``
   (appendOnly) via the audit log the VM owns.
3. Returns ``PolicyResult(action=DENY, reason=str(exc))`` so the
   executor stops the turn cleanly.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any

from agent.omnigent import FunctionPolicy, PolicyAction, PolicyResult

logger = logging.getLogger("alex.harness.constitution_adapter")


class ConstitutionAdapter(FunctionPolicy):
    """A :class:`FunctionPolicy` that delegates to the Alex constitution.

    The session is fetched via a callable so the adapter can be
    used from any thread without holding a reference to the VM.

    Parameters
    ----------
    session_getter:
        Callable that returns the current session object.  The
        session must expose ``user_email``, ``instance_scope``,
        ``session_id``, ``constitution`` (a
        :class:`ConstitutionalInterceptor`), and ``audit_log``
        (callable that appends a row to
        ``media.alex.agent_actions`` /
        ``media.control.constitution_violations``).
    on_block:
        Optional callable invoked on every :class:`ConstitutionalBlock`
        *before* the policy returns.  The VM passes its event pump
        here so the law_block event is fanned out to SSE subscribers.
    """

    def __init__(
        self,
        *,
        session_getter: Any,
        on_block: Any | None = None,
        name: str = "alex-constitution",
    ) -> None:
        super().__init__(
            name=name,
            on=["request", "response", "tool_call", "tool_result"],
            callable=self._evaluate,
        )
        self._session_getter = session_getter
        self._on_block = on_block

    def _evaluate(self, event: dict[str, Any]) -> PolicyResult:
        session = self._session_getter()
        phase = event.get("type", "")
        data = event.get("data") or {}
        if phase != "tool_call":
            return PolicyResult(action=PolicyAction.ALLOW)

        tool_name: str | None = None
        tool_args: dict[str, Any] = {}
        if isinstance(data, dict):
            tool_name = data.get("tool")
            raw_args = data.get("args", {})
            tool_args = raw_args if isinstance(raw_args, dict) else {}
        if not tool_name:
            return PolicyResult(action=PolicyAction.ALLOW)

        from agent.constitution import ConstitutionalBlock
        from agent.guardrails import TOOL_CAPABILITY

        capability = TOOL_CAPABILITY.get(tool_name)
        if capability is None and tool_name.startswith("mcp__"):
            # Remote MCP tools are not in the static map — their gate
            # capability is declared per server in the control plane and
            # already enforced at exposure and at call binding. Here the
            # adapter needs A capability for the interceptor's rules;
            # the family umbrella is the honest one.
            capability = "mcp.use"
        if capability is None:
            return PolicyResult(
                action=PolicyAction.DENY,
                reason=f"no capability mapping for tool {tool_name!r}",
            )

        interceptor = getattr(session, "constitution", None)
        if interceptor is None:
            return PolicyResult(action=PolicyAction.ALLOW)

        try:
            # The canonical ConstitutionalInterceptor.check signature is
            # (tool, args); richer interceptors (tests, future L2
            # layers) may accept identity keywords too.
            try:
                params = inspect.signature(interceptor.check).parameters
            except (TypeError, ValueError):
                params = {}
            if "user_email" in params:
                interceptor.check(
                    user_email=getattr(session, "user_email", ""),
                    instance_scope=getattr(session, "instance_scope", ""),
                    tool=tool_name,
                    capability=capability,
                    args=tool_args,
                )
            else:
                interceptor.check(tool_name, tool_args)
        except ConstitutionalBlock as exc:
            if self._on_block is not None:
                try:
                    self._on_block(
                        {
                            "tool": tool_name,
                            "capability": capability,
                            "args": tool_args,
                        },
                        exc,
                    )
                except Exception:  # noqa: BLE001
                    logger.exception("constitution on_block hook failed")
            return PolicyResult(action=PolicyAction.DENY, reason=str(exc))
        return PolicyResult(action=PolicyAction.ALLOW)


__all__ = ["ConstitutionAdapter"]
