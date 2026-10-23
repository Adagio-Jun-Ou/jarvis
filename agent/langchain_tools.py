"""Alex's governed tools, as LangChain tools.

This is the interop that matters for an organisation already building on
LangChain or LangGraph: their agent can call Alex's tools and **keep
every control**. It is not a convenience wrapper around the tool
functions — that would be a hole straight through the control stack.

Every exported tool routes through :meth:`agent.vm.VM._invoke_tool`, so
a LangGraph node calling ``send_email`` gets exactly what an Alex turn
gets:

* **L0 ACL** — only tools the principal may use are exported at all, so
  an unauthorised tool is not offered to the model rather than refused
  after it is chosen.
* **L2 governor** — budgets, repetition, oscillation, denial escalation.
  A LangGraph loop that thrashes is halted by the same counter.
* **L3 constitution** — the interceptor runs before the callable.
* **Audit** — every call writes its row.
* **Identity** — ``user_email`` and ``instance_scope`` are overwritten
  by the VM from the session, so a model-supplied value cannot
  impersonate. Calling ``spec.callable`` directly would hand that
  decision to whatever produced the arguments.

The rule to keep: **never export ``spec.callable``**. The registry entry
is the unguarded function; the VM is the guard.

LangChain is an optional dependency. This module imports without it; the
functions raise a clear error instead of failing at import time, so the
rest of the system does not acquire a hard dependency on a framework it
does not otherwise use.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any

from agent.tools import TOOL_REGISTRY, ToolSpec

log = logging.getLogger("agent.langchain_tools")

#: Parameters the VM fills in from the session. They must never appear in
#: an exported schema: a model that can see them is a model that will try
#: to set them.
INJECTED = frozenset({"user_email", "instance_scope"})

_MISSING = (
    "langchain-core is not installed. It is an optional dependency: "
    "`pip install langchain-core` to expose Alex's tools to a LangChain "
    "or LangGraph agent."
)


def langchain_available() -> bool:
    """Whether the optional dependency is importable."""
    try:
        import langchain_core.tools  # noqa: F401
    except ImportError:
        return False
    return True


def _require_langchain() -> None:
    if not langchain_available():
        raise ImportError(_MISSING)


def _python_type(annotation: Any) -> Any:
    """A concrete type for the args schema.

    Unannotated parameters become ``str`` rather than ``Any``: a schema
    with no type lets a model send anything, and the tool's own
    validation is not a substitute for telling the model what to send.
    """
    if annotation is inspect.Parameter.empty:
        return str
    return annotation


def arg_model_for(spec: ToolSpec):
    """A pydantic model describing what the model may pass to this tool.

    Built from the callable's signature with the injected identity
    parameters removed, so the exported schema is exactly the surface the
    VM will accept.
    """
    _require_langchain()
    from pydantic import Field, create_model

    try:
        sig = inspect.signature(spec.callable)
    except (TypeError, ValueError):  # pragma: no cover - builtins
        sig = None

    fields: dict[str, Any] = {}
    if sig is not None:
        for name, param in sig.parameters.items():
            if name in INJECTED or name == "kwargs":
                continue
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            annotation = _python_type(param.annotation)
            if param.default is inspect.Parameter.empty:
                fields[name] = (annotation, Field(...))
            else:
                fields[name] = (annotation, Field(default=param.default))
    return create_model(f"{spec.name}_args", **fields)


def governed_tool(vm, spec: ToolSpec):
    """One LangChain tool that runs through the VM's governed path."""
    _require_langchain()
    from langchain_core.tools import StructuredTool

    from agent.omnigent import ToolCallRequest

    def _run(**kwargs: Any) -> Any:
        # Identity is dropped rather than forwarded. The VM overwrites it
        # anyway; stripping it here means a caller cannot even express
        # the attempt, and the audit trail never shows one.
        args = {k: v for k, v in kwargs.items() if k not in INJECTED}
        return vm._invoke_tool(ToolCallRequest(name=spec.name, args=args))

    return StructuredTool.from_function(
        func=_run,
        name=spec.name,
        description=spec.description or f"Alex tool: {spec.name}",
        args_schema=arg_model_for(spec),
        # A tool error is information the agent should reason about, not
        # a reason to tear down the graph — the VM already turned
        # denials into structured results rather than exceptions.
        handle_tool_error=True,
    )


def governed_tools(vm, *, names: list[str] | None = None) -> list:
    """Every tool this VM's principal may use, as LangChain tools.

    ``names`` narrows the export. It can only narrow: a name the
    principal's ACL does not allow stays excluded, because a caller
    asking for a tool is not the same as being entitled to it.
    """
    _require_langchain()
    allowed = vm._allowed_capabilities()
    wanted = set(names) if names else None
    out = []
    for spec in TOOL_REGISTRY.values():
        if allowed is not None and spec.capability not in allowed:
            continue
        if wanted is not None and spec.name not in wanted:
            continue
        try:
            out.append(governed_tool(vm, spec))
        except Exception:  # noqa: BLE001
            # One tool with an unrepresentable signature must not cost
            # the caller every other tool.
            log.debug("could not export %s to langchain", spec.name, exc_info=True)
    return out
