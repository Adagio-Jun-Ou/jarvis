"""
Tool registry — the single source of truth for every tool the agent
can call.

The :data:`TOOL_REGISTRY` maps the tool name (as the LLM sees it)
to a :class:`ToolSpec` with the four fields every tool needs:

* ``name`` — the LLM-facing name.
* ``capability`` — the ACL token :mod:`agent.guardrails` resolves.
* ``description`` — one-line summary; first line of the docstring
  is preferred.
* ``return_schema`` — a JSON-Schema fragment that describes the
  tool's return shape (best-effort; tools that return arbitrary
  dicts get a generic ``{"type": "object"}``).
* ``constitution_articles`` — the article IDs the tool touches
  (``A1``-``A9`` from ``media.control.constitution``).  Used by
  the harness to pre-warm the agent's prompt with the relevant
  rules.

The :class:`agent.harness.VM` and the SDK both read this registry
when building :class:`agent.omnigent.AgentDef`s.  The FastAPI
``/api/tools`` route serves the same data as a public manifest so
the React UI can render the dock's tool catalogue from a single
fetch.

Tool authoring rules:

1. The function's name is the LLM-facing name.
2. The function lives in :mod:`agent.agentic_browser` (for browser
   tools) or in a domain-specific module under :mod:`agent` (for
   task, mail, file, etc.).
3. The function's docstring's first line is the description.
4. The function's first parameter is a :class:`ToolContext` (or a
   set of explicit kwargs: ``session_id``, ``user_email``, etc.).
5. The function's return value is a JSON-serialisable dict (or a
   dataclass with ``to_dict()``).
6. Every tool has an entry here AND in
   :data:`agent.guardrails.TOOL_CAPABILITY`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolSpec:
    """The canonical tool metadata."""

    name: str
    capability: str
    callable: Callable[..., Any]
    description: str = ""
    return_schema: dict[str, Any] = field(default_factory=lambda: {"type": "object"})
    constitution_articles: tuple[str, ...] = ()
    # Whether this tool's RESULT carries content an outside party could
    # have authored (a web page, an email body, an external API
    # response, an uploaded file). Untrusted results are fenced before
    # they re-enter the model's context — indirect prompt injection
    # arrives through exactly these tools — and the governor treats an
    # irreversible action taken after consuming one as a signal.
    untrusted_output: bool = False
    # Optional explicit JSON schema for the model (richer than what the
    # signature yields: descriptions, enums). Used by adapted tools.
    input_schema: dict[str, Any] | None = None
    # Deferred tools are registered and governed but NOT in the default
    # working set: they enter a turn by relevance to the request, by a
    # find_tools search, or by having been used before. This is how a
    # library of hundreds of tools stays out of every model pass.
    deferred: bool = False


def _browser_tool(name: str, capability: str, **kwargs: Any) -> ToolSpec:
    """Build a :class:`ToolSpec` for one of the agentic browser ops.

    Browser tools live in :mod:`agent.agentic_browser`; this
    helper centralises the resolution + the constitution-article
    list so the registry reads cleanly.
    """
    import agent.agentic_browser as ab

    fn = getattr(ab, name)
    return ToolSpec(
        name=name,
        capability=capability,
        callable=fn,
        description=(fn.__doc__ or "").strip().splitlines()[0] if fn.__doc__ else name,
        return_schema={"type": "object", "properties": {"ok": {"type": "boolean"}}},
        constitution_articles=kwargs.get("constitution_articles", ("A3", "A8")),
        untrusted_output=bool(kwargs.get("untrusted_output", False)),
    )


def _plan_task(steps, *, user_email: str = "", instance_scope: str = "default"):
    from agent.task_plan import plan_task

    return plan_task(steps, user_email=user_email, instance_scope=instance_scope)


def _check_plan_step(step_no, status, note="", *, user_email: str = "",
                     instance_scope: str = "default"):
    from agent.task_plan import check_plan_step

    return check_plan_step(step_no, status, note,
                           user_email=user_email, instance_scope=instance_scope)


def _check_acceptance(crit_no, status, note="", *, user_email: str = "",
                      instance_scope: str = "default"):
    from agent.task_acceptance import check_acceptance

    return check_acceptance(crit_no, status, note,
                            user_email=user_email, instance_scope=instance_scope)


def _ask_user(question, options=None, *, user_email: str = "",
              instance_scope: str = "default"):
    from agent.task_questions import ask_user

    return ask_user(question, options,
                    user_email=user_email, instance_scope=instance_scope)


def _delegate_subagents(tasks, *, user_email: str = "", instance_scope: str = "default"):
    from agent.swarm import delegate_subagents

    return delegate_subagents(
        tasks, user_email=user_email, instance_scope=instance_scope
    )


def _impl_tool(
    name: str,
    capability: str,
    *,
    description: str = "",
    articles: tuple[str, ...] = ("A3",),
    untrusted_output: bool = False,
) -> ToolSpec:
    """Build a :class:`ToolSpec` for a non-browser tool.

    The implementations live in :mod:`agent.tools.impls`; each takes
    ``user_email``/``instance_scope`` keyword args (injected by the
    VM) plus its own LLM-facing parameters.
    """
    from agent.tools import impls

    fn = getattr(impls, name)
    desc = description or (
        (fn.__doc__ or "").strip().splitlines()[0] if fn.__doc__ else name
    )
    return ToolSpec(
        name=name,
        capability=capability,
        callable=fn,
        description=desc,
        return_schema={"type": "object"},
        constitution_articles=articles,
        untrusted_output=untrusted_output,
    )


def input_schema_for(spec: ToolSpec) -> dict[str, Any]:
    """Derive an MCP-compatible JSON-schema ``inputSchema`` from the
    tool callable's signature.  Identity kwargs the VM injects
    (``user_email``, ``instance_scope``) are excluded — the LLM (and
    an MCP client) never supplies them.

    A spec that carries an explicit ``input_schema`` (the adapted
    Jarvis tools do) returns it verbatim: the signature only knows
    types, while the explicit schema carries parameter descriptions
    and enums the model needs to call the tool correctly.
    """
    explicit = getattr(spec, "input_schema", None)
    if explicit:
        return explicit
    import inspect
    import typing

    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }
    # Param names that carry tabular/collection payloads even when their
    # annotation is ``Any`` (the office tools accept forgiving shapes).
    array_hints = {
        "sheets", "headers", "rows", "slides", "sections", "bullets",
        "packages", "argv", "items", "columns", "data",
    }

    def _json_type(ann: Any) -> str | None:
        """Best JSON type for an annotation, or None for 'any'."""
        if ann is inspect.Parameter.empty or ann is typing.Any:
            return None
        origin = typing.get_origin(ann)
        if origin is typing.Union:  # e.g. str | None
            non_none = [a for a in typing.get_args(ann) if a is not type(None)]
            return _json_type(non_none[0]) if non_none else None
        if origin in (list, tuple, set):
            return "array"
        if origin is dict:
            return "object"
        base = origin or ann
        if isinstance(base, type):
            for pytype, jt in type_map.items():
                if issubclass(base, pytype):
                    return jt
        # String annotations (`from __future__ import annotations` makes
        # every annotation a string, e.g. "list[str]", "str | None").
        # Parse the OUTER type precisely so "list[str]" isn't mis-matched
        # to "str".
        text = str(ann).strip()
        if "|" in text:  # union like "str | None"
            parts = [p.strip() for p in text.split("|") if p.strip().lower() != "none"]
            return _json_type(parts[0]) if parts else None
        head = text.split("[", 1)[0].strip().lower()
        outer = {
            "list": "array", "tuple": "array", "set": "array", "sequence": "array",
            "dict": "object", "mapping": "object",
            "str": "string", "int": "integer", "float": "number",
            "number": "number", "bool": "boolean",
        }
        return outer.get(head)

    props: dict[str, Any] = {}
    required: list[str] = []
    try:
        sig = inspect.signature(spec.callable)
    except (TypeError, ValueError):
        return {"type": "object", "properties": {}}
    for pname, param in sig.parameters.items():
        if pname in ("user_email", "instance_scope"):
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        jt = _json_type(param.annotation)
        if jt is None and pname in array_hints:
            jt = "array"
        entry: dict[str, Any] = {}
        if jt is not None:
            entry["type"] = jt
            if jt == "array":
                entry["items"] = {}
        props[pname] = entry
        if param.default is inspect.Parameter.empty:
            required.append(pname)
    schema: dict[str, Any] = {"type": "object", "properties": props}
    if required:
        schema["required"] = required
    return schema


# Build the registry in declaration order.  Each entry is built
# once; tests assert every entry has the four required fields.
TOOL_REGISTRY: dict[str, ToolSpec] = {
    # ---- agentic browser ---------------------------------------------
    "browser_navigate": _browser_tool("browser_navigate", "browser.use"),
    "browser_click": _browser_tool("browser_click", "browser.act"),
    "browser_type": _browser_tool("browser_type", "browser.act"),
    "browser_extract": _browser_tool(
        "browser_extract", "browser.read", untrusted_output=True
    ),
    "browser_screenshot": _browser_tool("browser_screenshot", "browser.read"),
    "browser_submit": _browser_tool(
        "browser_submit", "browser.act", **{"constitution_articles": ("A3", "A8")}
    ),
    "browser_list_tabs": _browser_tool("browser_list_tabs", "browser.read"),
    "browser_close_tab": _browser_tool("browser_close_tab", "browser.use"),
    "browser_open_in_user_session": _browser_tool(
        "browser_open_in_user_session", "browser.use"
    ),
    # ---- memory ------------------------------------------------------
    "remember": _impl_tool("remember", "memory.write"),
    "recall": _impl_tool("recall", "memory.read"),
    # ---- taught routines ----------------------------------------------
    "run_routine": _impl_tool("run_routine", "learning.read"),
    # Swarm delegation. Deliberately its own capability so an operator
    # grants fan-out explicitly; a sub-agent never receives it (depth-one
    # by construction — see agent/swarm.py).
    "plan_task": ToolSpec(
        name="plan_task",
        capability="tasks.update",
        callable=_plan_task,
        description=(
            "Declare or replace the step-by-step plan for the board task "
            "you are working. Call FIRST with 2-7 concrete steps before "
            "acting; call again only if the situation genuinely changes. "
            "The crew sees the plan live."
        ),
        constitution_articles=("A3",),
    ),
    "check_plan_step": ToolSpec(
        name="check_plan_step",
        capability="tasks.update",
        callable=_check_plan_step,
        description=(
            "Update one step of your current task plan: 'doing' when you "
            "start it, 'done' when finished, 'skipped' with a reason, or "
            "'blocked' when a human decision is needed."
        ),
        constitution_articles=("A3",),
    ),
    "check_acceptance": ToolSpec(
        name="check_acceptance",
        capability="tasks.update",
        callable=_check_acceptance,
        description=(
            "Record whether one acceptance criterion of your current task "
            "holds: 'met' with the evidence, 'unmet' with what stands in "
            "the way. The crew writes the criteria; you only verify them."
        ),
        constitution_articles=("A3",),
    ),
    "ask_user": ToolSpec(
        name="ask_user",
        capability="tasks.update",
        callable=_ask_user,
        description=(
            "Ask the human crew ONE question you cannot decide for them "
            "and wait for the answer — a preference, an authorization, a "
            "genuine fork in the work. Give 2-4 short options when the "
            "choice is enumerable. Never use it for facts your tools can "
            "find. If nobody answers in time you'll be told to proceed on "
            "your own stated assumption."
        ),
        constitution_articles=("A3",),
    ),
    "delegate_subagents": ToolSpec(
        name="delegate_subagents",
        capability="agents.delegate",
        callable=_delegate_subagents,
        description=(
            "Split the current task across up to 10 named sub-agents "
            "running in parallel, each in its own governed VM. tasks is "
            "a list of {name, brief, capabilities?}; briefs must be "
            "self-contained (workers start fresh, cannot see each other "
            "or ask questions). Returns every worker's output keyed by "
            "name, including failures."
        ),
        constitution_articles=("A3",),
    ),
    # ---- knowledge graph ---------------------------------------------
    "graph_query": _impl_tool("graph_query", "memory.read"),
    "add_edge": _impl_tool("add_edge", "memory.write"),
    # ---- tasks --------------------------------------------------------
    "create_task": _impl_tool("create_task", "tasks.create"),
    "update_task_status": _impl_tool("update_task_status", "tasks.update"),
    "list_my_tasks": _impl_tool("list_my_tasks", "tasks.read"),
    # ---- inbox / email ------------------------------------------------
    # Email bodies, subjects and summaries are authored by outside
    # senders — the canonical indirect-injection channel.
    "read_inbox": _impl_tool("read_inbox", "email.read", untrusted_output=True),
    "summarize_email": _impl_tool(
        "summarize_email", "email.read", untrusted_output=True
    ),
    "search_email": _impl_tool(
        "search_email", "email.read", untrusted_output=True
    ),
    # ---- workspace files ---------------------------------------------
    # Workspace files include user UPLOADS — documents whose original
    # author is unknown to the system.
    "view": _impl_tool("view", "files.read", untrusted_output=True),
    "create_file": _impl_tool("create_file", "files.write"),
    "download_file": _impl_tool("download_file", "compute.download"),
    "str_replace": _impl_tool("str_replace", "files.write"),
    "undo_file_change": _impl_tool("undo_file_change", "files.write"),
    # ---- n8n workflows (the enterprise automation arm) ----------------
    # Results are untrusted: a workflow definition or execution payload
    # can carry data from anywhere the n8n instance touched.
    "n8n_list_workflows": _impl_tool(
        "n8n_list_workflows", "workflow.read", untrusted_output=True
    ),
    "n8n_create_workflow": _impl_tool(
        "n8n_create_workflow", "workflow.create"
    ),
    "n8n_run_workflow": _impl_tool(
        "n8n_run_workflow", "workflow.trigger", untrusted_output=True
    ),
    "n8n_execution_status": _impl_tool(
        "n8n_execution_status", "workflow.read", untrusted_output=True
    ),
    "present_files": _impl_tool("present_files", "files.read"),
    # ---- desktop control (the machine Jarvis lives on) ----------------
    # Reads are power 0; acts are power 2 and policy-ASK by default.
    "desktop_screenshot": _impl_tool(
        "desktop_screenshot", "desktop.read", untrusted_output=True
    ),
    "desktop_find_text": _impl_tool(
        "desktop_find_text", "desktop.read", untrusted_output=True
    ),
    "desktop_click": _impl_tool("desktop_click", "desktop.act"),
    "desktop_type": _impl_tool("desktop_type", "desktop.act"),
    "desktop_key": _impl_tool("desktop_key", "desktop.act"),
    "desktop_hotkey": _impl_tool("desktop_hotkey", "desktop.act"),
    "desktop_scroll": _impl_tool("desktop_scroll", "desktop.act"),
    # ---- home automation ----------------------------------------------
    "home_status": _impl_tool("home_status", "home.read", untrusted_output=True),
    "home_control": _impl_tool("home_control", "home.act"),
    # ---- the deferred library ------------------------------------------
    "find_tools": _impl_tool("find_tools", "knowledge.read"),
    "bash": _impl_tool("bash", "compute.bash"),
    # ---- Office documents (Excel / PowerPoint / Word) ----------------
    "create_spreadsheet": _impl_tool("create_spreadsheet", "office.write"),
    "read_spreadsheet": _impl_tool("read_spreadsheet", "office.read"),
    "create_presentation": _impl_tool("create_presentation", "office.write"),
    "create_document": _impl_tool("create_document", "office.write"),
    # ---- python sandbox + VM package install / CLI -------------------
    "run_python": _impl_tool("run_python", "compute.python"),
    "pip_install": _impl_tool("pip_install", "compute.install"),
    "system_install": _impl_tool("system_install", "system.install"),
    "vm_exec": _impl_tool("vm_exec", "compute.shell"),
    # ---- enterprise RAG + vector knowledge graph ---------------------
    "search_knowledge": _impl_tool("search_knowledge", "knowledge.read"),
    "list_knowledge": _impl_tool("list_knowledge", "knowledge.read"),
    "ingest_document": _impl_tool("ingest_document", "knowledge.write"),
    "ingest_workspace_file": _impl_tool("ingest_workspace_file", "knowledge.write"),
    "graph_entity": _impl_tool("graph_entity", "memory.read"),
    "graph_neighbors": _impl_tool("graph_neighbors", "memory.read"),
    "graph_link": _impl_tool("graph_link", "memory.write"),
    "graph_about_me": _impl_tool("graph_about_me", "memory.read"),
    # ---- standing orders + best-practice library ---------------------
    "list_standing_orders": _impl_tool("list_standing_orders", "orders.read"),
    "run_standing_order": _impl_tool(
        "run_standing_order", "orders.run", articles=("A3", "A8")
    ),
    "search_practices": _impl_tool("search_practices", "practices.read"),
    "propose_practice": _impl_tool("propose_practice", "practices.propose"),
    # ---- LSP code intelligence (agent.lsp_tool) ----------------------
    "code_outline": _impl_tool("code_outline", "code.read"),
    "code_definition": _impl_tool("code_definition", "code.read"),
    "code_references": _impl_tool("code_references", "code.read"),
    "code_hover": _impl_tool("code_hover", "code.read"),
    "code_diagnostics": _impl_tool("code_diagnostics", "code.read"),
    "code_search_symbols": _impl_tool("code_search_symbols", "code.read"),
    # ---- internal APIs (api_act stays two-phase, A8) ------------------
    "api_read": _impl_tool(
        "api_read", "internal_api.read", untrusted_output=True
    ),
    "api_act": _impl_tool("api_act", "internal_api.act", articles=("A3", "A8")),
}


# A separate "capability lookup" so the guardrails gate can find
# the tool by capability too.  Mirrors :data:`TOOL_REGISTRY`; the
# harness asserts they stay in sync.
def tools_by_capability() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for spec in TOOL_REGISTRY.values():
        out.setdefault(spec.capability, []).append(spec.name)
    return out


__all__ = ["ToolSpec", "TOOL_REGISTRY", "tools_by_capability", "input_schema_for"]
