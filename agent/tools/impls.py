"""
Non-browser tool implementations for the agent's tool registry.

Every function here follows the registry's calling convention: the VM
injects ``user_email`` and ``instance_scope`` keyword arguments, the
LLM supplies the rest.  Each function returns a JSON-serialisable
value.  All of them degrade gracefully offline: in dev they run
against the in-memory fakes (:mod:`alex.testing`), in production the
same SQL runs against Unity Catalog / Lakebase.

Capability scoping and the constitution both apply *outside* this
module: the VM's ``_invoke_tool`` runs the constitutional interceptor
and the guardrails gate resolves each tool's capability from
:data:`agent.guardrails.TOOL_CAPABILITY`.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

log = logging.getLogger("agent.tools.impls")


# ---------------------------------------------------------------------------
# Warehouse helper (dev: fakes' InMemoryWarehouse; prod: injected client)
# ---------------------------------------------------------------------------

# Mirrors the media.alex.tasks DDL (governance/permissions.sql). The old
# list used task_id/assignee/shared_with/last_updated_at — columns that do
# not exist in the DDL, so create_task failed outright in production.
_TASKS_COLS = [
    "id",
    "owner_email",
    "assignee_email",
    "title",
    "description",
    "status",
    "priority",
    "due_at",
    "source",
    "source_ref",
    "created_at",
    "updated_at",
    "closed_at",
]
_KG_MIRROR_COLS = ["edge_id", "owner_email", "src", "rel", "dst", "evidence", "confidence"]
_INBOX_VIEW_COLS = [
    "email_id",
    "owner_email",
    "priority",
    "subject",
    "received_at",
    "summary",
    "from_email",
]
_VECTOR_MIRROR_COLS = [
    "id",
    "owner_email",
    "kind",
    "content",
    "salience",
    "source_ref",
    "created_at",
    "superseded_by",
]
_TOOL_INTEGRATIONS_COLS = [
    "integration_id",
    "base_url",
    "read_capability",
    "act_capability",
    "act_requires_confirm",
    "display_name",
    "mode",
    "auth_kind",
    "api_spec_path",
    "obo_audience",
    "obo_scopes",
    "vault_key_id",
    "created_at",
    "updated_at",
]


_prod_warehouse = None


def _warehouse():
    # Dev/test: the in-memory fakes. Production: the real Databricks SQL
    # warehouse. Returning the fake unconditionally (the previous behavior)
    # meant every tool that writes via _sql — create_task, remember's mirror
    # upsert, etc. — silently hit an ephemeral in-process table in prod and
    # never persisted to Unity Catalog.
    from alex import testing as _testing

    if _testing.is_dev():
        return _testing.get_fakes().warehouse
    global _prod_warehouse
    if _prod_warehouse is None:
        from alex.warehouse import WarehouseClient

        _prod_warehouse = WarehouseClient()
    return _prod_warehouse


def _ensure(wh, table: str, cols: list[str]) -> None:
    """Register the table in the dev warehouse only if it truly doesn't
    exist yet.  InMemoryWarehouse keys tables lowercase in `_tables`;
    re-registering an existing table would WIPE its seeded rows."""
    register = getattr(wh, "register", None)
    if not callable(register):
        return
    existing = getattr(wh, "_tables", None) or getattr(wh, "tables", None) or {}
    if table.lower() in existing or table in existing:
        return
    try:
        register(table, cols)
    except Exception:  # noqa: BLE001
        pass


def _sql(stmt: str, params: dict | None = None) -> list[dict]:
    wh = _warehouse()
    res = wh.execute(stmt, params or {})
    as_dicts = getattr(res, "as_dicts", None)
    return as_dicts() if callable(as_dicts) else list(res or [])


def _llm_invoke():
    """Return an ``llm_invoke(messages) -> str`` callable for the email
    extraction. Dev/test: the scripted FakeLLM (returns a string). Prod: the
    real serving LLM, unwrapping its LLMResponse to text — the previous code
    used the fake unconditionally, so the prod fallback ran a null model."""
    from alex import testing as _testing

    if _testing.is_dev():
        return _testing.get_fakes().llm.invoke
    from alex.llm import get_llm

    llm = get_llm()

    def _inv(messages):
        resp = llm.invoke(messages)
        return getattr(resp, "content", resp) or ""

    return _inv


# ---------------------------------------------------------------------------
# Memory tools (agent.memory + agent.vector_memory)
# ---------------------------------------------------------------------------


def remember(
    content: str,
    kind: str = "fact",
    source_ref: str = "chat",
    *,
    user_email: str,
    instance_scope: str = "default",
) -> dict:
    """Store a durable memory about the user (a preference, fact, or project note)."""
    if not content or not content.strip():
        return {"ok": False, "error": "content is required"}
    from agent import vector_memory
    from agent.memory import connection

    with connection() as conn:
        bead = vector_memory.store(
            conn, user_email, kind[:64], content.strip()[:4000], source_ref[:200]
        )
    # Reflect the new memory into the transparency read model immediately.
    # store() writes the vector store (Lakebase / dev _DevStore); the "What
    # my AI knows" tab reads media.control.vector_memory_mirror, which was
    # otherwise only populated later by the consolidation job — so a just-
    # remembered fact was invisible to the user (and never appeared in dev).
    wh = _warehouse()
    _ensure(wh, "media.control.vector_memory_mirror", _VECTOR_MIRROR_COLS)
    _sql(
        "INSERT INTO media.control.vector_memory_mirror "
        "(id, owner_email, kind, content, salience, source_ref) "
        "VALUES (:id, :u, :k, :c, :s, :sr)",
        {"id": bead.id, "u": user_email, "k": bead.kind, "c": bead.content,
         "s": bead.salience, "sr": bead.source_ref},
    )
    out = bead.to_dict()
    out["ok"] = True
    return out


def recall(
    query: str = "",
    k: int = 5,
    *,
    user_email: str,
    instance_scope: str = "default",
) -> dict:
    """Recall the user's stored memories most relevant to a query."""
    from agent import vector_memory
    from agent.memory import connection

    with connection() as conn:
        beads = vector_memory.recall(
            conn, user_email, query_text=query, k=max(1, min(int(k), 25))
        )
    return {"ok": True, "memories": [b.to_dict() for b in beads]}


# ---------------------------------------------------------------------------
# Taught routines (agent.learn_with_you)
# ---------------------------------------------------------------------------


def run_routine(
    name_or_query: str,
    *,
    user_email: str,
    instance_scope: str = "default",
) -> dict:
    """Fetch the steps of a routine the user taught (matched by name or query) so you can follow them."""
    if not name_or_query or not name_or_query.strip():
        return {"ok": False, "error": "name_or_query is required"}
    wh = _warehouse()
    _ensure(wh, "media.control.vector_memory_mirror", _VECTOR_MIRROR_COLS)
    from agent.learn_with_you import match_routine

    routine = match_routine(
        _sql, user_email, name_or_query.strip(), threshold=0.15
    )
    if routine is None:
        return {
            "ok": False,
            "error": f"no taught routine matches {name_or_query!r}",
        }
    return {"ok": True, **routine}


# ---------------------------------------------------------------------------
# Knowledge-graph tools (agent.knowledge_graph / kg_mirror)
# ---------------------------------------------------------------------------


def graph_query(
    text: str,
    limit: int = 8,
    *,
    user_email: str,
    instance_scope: str = "default",
) -> dict:
    """Query the knowledge graph for facts related to the text.

    Backed by :mod:`agent.enterprise_graph` — embedded entity
    resolution, temporal edges, enterprise ∪ user scope — not the old
    token-match over the mirror table. The mirror is a read model for
    the Transparency surface, not a query engine: substring matching on
    it missed every synonym the embedded graph resolves.
    """
    if not text:
        return {"ok": True, "edges": []}
    try:
        from agent import enterprise_graph as _eg
        from agent.memory import connection

        with connection() as conn:
            edges = _eg.subgraph(
                conn, text, user_email=user_email
            ).get("edges") or []
    except Exception:  # noqa: BLE001
        log.debug("graph_query unavailable", exc_info=True)
        return {"ok": True, "edges": []}
    return {"ok": True, "edges": edges[: max(1, int(limit))]}


def add_edge(
    src: str,
    rel: str,
    dst: str,
    evidence: str = "",
    confidence: float = 0.7,
    *,
    user_email: str,
    instance_scope: str = "default",
) -> dict:
    """Add a knowledge-graph edge (src --rel--> dst) to the user's graph."""
    if not (src and rel and dst):
        return {"ok": False, "error": "src, rel, dst are required"}
    # The MODERN graph is the store: embedded entities, temporal
    # supersession, scope model. Writing only the mirror — the previous
    # behaviour — meant the tool's edges were invisible to graph_query,
    # retrieval expansion and the context engine, all of which read the
    # real graph.
    from agent.enterprise_graph import upsert_edge, upsert_entity, user_scope
    from agent.memory import connection

    # An edge whose endpoints are not entity nodes is unresolvable: the
    # graph answers queries by resolving entities first, so a dangling
    # edge is write-only data. Asserting the endpoints is part of what
    # "add an edge" means. (conn=None raised in prod — the in-memory
    # dev backend ignored it, so tests never saw the failure.)
    with connection() as conn:
        upsert_entity(conn, src[:200], scope=user_scope(user_email))
        upsert_entity(conn, dst[:200], scope=user_scope(user_email))
        edge = upsert_edge(
            conn,
            src[:200],
            rel[:100],
            dst[:200],
            scope=user_scope(user_email),
            evidence=evidence[:300],
            confidence=min(max(float(confidence), 0.0), 1.0),
            source_ref="tool:add_edge",
        )
    # The mirror stays as the Transparency surface's read model — an
    # explicit projection, not a second source of truth.
    wh = _warehouse()
    _ensure(wh, "media.control.kg_mirror", _KG_MIRROR_COLS)
    _sql(
        "INSERT INTO media.control.kg_mirror "
        "(edge_id, owner_email, src, rel, dst, evidence, confidence) "
        "VALUES (:eid, :u, :s, :r, :d, :e, :c)",
        {
            "eid": edge.edge_id,
            "u": user_email,
            "s": src[:200],
            "r": rel[:100],
            "d": dst[:200],
            "e": evidence[:300],
            "c": min(max(float(confidence), 0.0), 1.0),
        },
    )
    return {"ok": True, "src": src, "rel": rel, "dst": dst}


# ---------------------------------------------------------------------------
# Task tools (media.alex.tasks)
# ---------------------------------------------------------------------------


def create_task(
    title: str,
    description: str = "",
    priority: int = 3,
    due_at: str | None = None,
    *,
    user_email: str,
    instance_scope: str = "default",
) -> dict:
    """Create a task owned by the signed-in user."""
    if not title or not title.strip():
        return {"ok": False, "error": "title is required"}
    wh = _warehouse()
    _ensure(wh, "media.alex.tasks", _TASKS_COLS)
    task_id = f"t-{uuid.uuid4()}"
    _sql(
        "INSERT INTO media.alex.tasks "
        "(id, owner_email, assignee_email, title, description, status, "
        " priority, due_at, source, source_ref, created_at, updated_at) "
        "VALUES (:tid, :u, :u, :t, :d, 'open', :p, :due, 'agent', 'agent', "
        "        current_timestamp(), current_timestamp())",
        {
            "tid": task_id,
            "u": user_email,
            "t": title.strip()[:300],
            "d": description[:2000],
            "p": max(1, min(int(priority), 5)),
            "due": due_at,
        },
    )
    return {"ok": True, "task_id": task_id, "title": title.strip()[:300]}


def update_task_status(
    task_id: str,
    status: str,
    *,
    user_email: str,
    instance_scope: str = "default",
) -> dict:
    """Update the status of one of the user's tasks (todo, in_progress,
    review, done, cancelled)."""
    # The union of the board lanes and the legacy statuses — the same set
    # chk_tasks_status accepts (governance/permissions.sql). The board
    # normalises on read (open->todo, cancelled->done).
    allowed = {"backlog", "todo", "open", "in_progress", "review",
               "done", "cancelled", "blocked"}
    if status not in allowed:
        return {"ok": False, "error": f"status must be one of {sorted(allowed)}"}
    wh = _warehouse()
    _ensure(wh, "media.alex.tasks", _TASKS_COLS)

    # This table is the task board's source of truth, and the board's own
    # run flow deliberately lands finished agent work in `review` so a
    # person decides whether it is done. That gate is worth nothing if
    # Alex can reach the same row through this tool and write `done`
    # itself, which is exactly what it could do before. A task Alex never
    # worked on is still the user's to close.
    if status == "done" and _agent_worked_on(task_id):
        _sql(
            "UPDATE media.alex.tasks SET status = :s "
            "WHERE id = :tid AND owner_email = :u",
            {"s": "review", "tid": task_id, "u": user_email},
        )
        return {
            "ok": True,
            "task_id": task_id,
            "status": "review",
            "note": (
                "Alex worked this task, so it moves to review rather than "
                "done — a person confirms the work before it is closed."
            ),
        }

    _sql(
        "UPDATE media.alex.tasks SET status = :s "
        "WHERE id = :tid AND owner_email = :u",
        {"s": status, "tid": task_id, "u": user_email},
    )
    return {"ok": True, "task_id": task_id, "status": status}


def _agent_worked_on(task_id: str) -> bool:
    """Whether Alex has left any fingerprint of work on this task.

    Three independent signals, any of which routes done -> review:

    * a **run row** — the board opened a VM session on the task;
    * a **plan row** — only the agent's plan tools write task_plans, so
      a plan means Alex worked the task even if it was through a plain
      chat that never opened a board run (the gap the run-only check
      left: chat-worked tasks could be closed unreviewed);
    * an **acceptance verdict** — a criterion marked met/unmet was
      verified by the agent (humans write the criteria; verdicts reset
      to 'open' whenever the criteria are rewritten).

    The RUN check fails closed (an unreadable runs table treats the
    task as agent-worked — wrongly routing to review costs a click,
    wrongly closing hides unreviewed work). The plan and acceptance
    checks are additive evidence only: unreadable means no evidence,
    never a verdict, so an environment without those tables keeps the
    original run-only semantics.
    """
    from agent import taskboard

    try:
        if _sql(
            f"SELECT run_id FROM {taskboard.RUNS_TABLE} WHERE task_id = :t",
            {"t": task_id},
        ):
            return True
    except Exception:  # noqa: BLE001
        log.debug("could not check agent runs for %s", task_id, exc_info=True)
        return True
    try:
        from agent.task_plan import PLANS_TABLE

        if _sql(
            f"SELECT step_no FROM {PLANS_TABLE} WHERE task_id = :t",
            {"t": task_id},
        ):
            return True
    except Exception:  # noqa: BLE001
        log.debug("plan check unavailable for %s", task_id, exc_info=True)
    try:
        from agent.task_acceptance import ACCEPTANCE_TABLE

        verdicts = _sql(
            f"SELECT status FROM {ACCEPTANCE_TABLE} WHERE task_id = :t",
            {"t": task_id},
        )
        if any(str(v.get("status")) in ("met", "unmet") for v in verdicts):
            return True
    except Exception:  # noqa: BLE001
        log.debug("acceptance check unavailable for %s", task_id, exc_info=True)
    return False


def list_my_tasks(
    status: str = "",
    *,
    user_email: str,
    instance_scope: str = "default",
) -> dict:
    """List the signed-in user's tasks, optionally filtered by status."""
    wh = _warehouse()
    # Read the base `tasks` table directly, NOT task_board_view: the board
    # view is a materialized read model refreshed by pipelines/task_board_
    # build.py on a schedule, so a task the agent just created wouldn't show
    # until the next run. The base table is always current. (The dashboard's
    # /api/tasks path keeps the board view for its cheap pre-aggregated read.)
    _ensure(wh, "media.alex.tasks", _TASKS_COLS)
    rows = _sql(
        "SELECT id, title, status, priority, due_at "
        "FROM media.alex.tasks WHERE owner_email = :u",
        {"u": user_email},
    )
    # Expose `task_id` for a stable tool output shape (base table PK is `id`).
    rows = [{**r, "task_id": r.get("id")} for r in rows]
    if status:
        rows = [r for r in rows if r.get("status") == status]
    return {"ok": True, "tasks": rows[:100]}


# ---------------------------------------------------------------------------
# Inbox / email tools (agent.email_intel + inbox_view)
# ---------------------------------------------------------------------------


def read_inbox(
    limit: int = 20,
    *,
    user_email: str,
    instance_scope: str = "default",
) -> dict:
    """Read the user's prioritized inbox (id, priority, subject, summary)."""
    wh = _warehouse()
    _ensure(wh, "media.alex.inbox_view", _INBOX_VIEW_COLS)
    rows = _sql(
        "SELECT email_id, priority, subject, received_at, summary "
        "FROM media.alex.inbox_view WHERE owner_email = :u",
        {"u": user_email},
    )
    return {"ok": True, "inbox": rows[: max(1, min(int(limit), 100))]}


def summarize_email(
    email_id: str,
    *,
    user_email: str,
    instance_scope: str = "default",
) -> dict:
    """Summarize a single email from the user's inbox by id."""
    if not email_id:
        return {"ok": False, "error": "email_id is required"}
    wh = _warehouse()
    _ensure(wh, "media.alex.inbox_view", _INBOX_VIEW_COLS)
    rows = _sql(
        "SELECT email_id, subject, summary, priority "
        "FROM media.alex.inbox_view WHERE owner_email = :u AND email_id = :e",
        {"u": user_email, "e": email_id},
    )
    if not rows:
        return {"ok": False, "error": f"email {email_id!r} not found"}
    r = rows[0]
    if r.get("summary"):
        return {"ok": True, **r}
    # No cached summary: run the injection-proof extraction over the body.
    try:
        from agent.email_intel import analyze

        body_rows = _sql(
            "SELECT body FROM media.alex.inbox "
            "WHERE owner_email = :u AND email_id = :e",
            {"u": user_email, "e": email_id},
        )
        body = str(body_rows[0].get("body") or "") if body_rows else ""
        intel = analyze(email_id, user_email, body, llm_invoke=_llm_invoke())
        return {"ok": True, "email_id": email_id, "summary": intel.understanding}
    except Exception:  # noqa: BLE001
        log.debug("summarize_email extraction unavailable", exc_info=True)
        return {"ok": True, **r}


def search_email(
    query: str,
    limit: int = 10,
    *,
    user_email: str,
    instance_scope: str = "default",
) -> dict:
    """Search the user's inbox by keywords in the subject or summary."""
    if not query:
        return {"ok": True, "results": []}
    wh = _warehouse()
    _ensure(wh, "media.alex.inbox_view", _INBOX_VIEW_COLS)
    rows = _sql(
        "SELECT email_id, priority, subject, received_at, summary "
        "FROM media.alex.inbox_view WHERE owner_email = :u",
        {"u": user_email},
    )
    terms = [t.lower() for t in query.split() if t]
    scored = []
    for r in rows:
        hay = (str(r.get("subject", "")) + " " + str(r.get("summary", ""))).lower()
        score = sum(t in hay for t in terms)
        if score:
            scored.append((score, r))
    scored.sort(key=lambda x: (-x[0], x[1].get("priority") or 9))
    return {"ok": True, "results": [r for _, r in scored[: max(1, min(int(limit), 50))]]}


# ---------------------------------------------------------------------------
# Workspace file tools (agent.workspace_tools)
# ---------------------------------------------------------------------------


def view(path: str, *, user_email: str, instance_scope: str = "default") -> str:
    """Read a file from the user's workspace (path relative to the workspace root)."""
    from agent import workspace_tools

    return workspace_tools.view(user_email, path)


def create_file(
    path: str, content: str, *, user_email: str, instance_scope: str = "default"
) -> str:
    """Create a new file in the user's workspace. Fails if the file already exists."""
    from agent import workspace_tools

    return workspace_tools.create_file(user_email, path, content)


def str_replace(
    path: str, old: str, new: str, *, user_email: str, instance_scope: str = "default"
) -> str:
    """Replace the first occurrence of a substring in a workspace file."""
    from agent import workspace_tools

    return workspace_tools.str_replace(user_email, path, old, new)


# ---------------------------------------------------------------------------
# n8n — the enterprise workflow arm (agent.n8n_client)
# ---------------------------------------------------------------------------


def n8n_list_workflows(*, user_email: str, instance_scope: str = "default") -> dict:
    """List the workflows on the organisation's n8n instance (name,
    active state, node count)."""
    from agent import n8n_client

    try:
        return {"ok": True, "workflows": n8n_client.list_workflows()}
    except n8n_client.N8NError as exc:
        return {"ok": False, "error": str(exc)}


def n8n_create_workflow(
    name: str,
    nodes: list,
    connections: dict | None = None,
    *,
    user_email: str,
    instance_scope: str = "default",
) -> dict:
    """Create a workflow on the organisation's n8n instance. Created
    INACTIVE and tagged with your identity; a person activates it from
    the Workflows surface after review."""
    from agent import n8n_client

    try:
        return {
            "ok": True,
            **n8n_client.create_workflow(
                name=name, nodes=list(nodes or []),
                connections=dict(connections or {}), created_by=user_email,
            ),
        }
    except n8n_client.N8NError as exc:
        return {"ok": False, "error": str(exc)}


def n8n_run_workflow(
    workflow_id: str,
    payload: dict | None = None,
    *,
    user_email: str,
    instance_scope: str = "default",
) -> dict:
    """Execute an n8n workflow now and return the execution handle."""
    from agent import n8n_client

    try:
        return {"ok": True, **n8n_client.run_workflow(workflow_id, payload)}
    except n8n_client.N8NError as exc:
        return {"ok": False, "error": str(exc)}


def n8n_execution_status(
    execution_id: str, *, user_email: str, instance_scope: str = "default"
) -> dict:
    """The status of one n8n execution (running / success / error)."""
    from agent import n8n_client

    try:
        return {"ok": True, **n8n_client.execution(execution_id)}
    except n8n_client.N8NError as exc:
        return {"ok": False, "error": str(exc)}


def undo_file_change(*, user_email: str, instance_scope: str = "default") -> str:
    """Revert your most recent file change in the workspace (a replaced
    file gets its previous content back; a created file is removed).
    Repeatable: each call undoes one more change."""
    from agent import workspace_tools

    return workspace_tools.undo_file_change(user_email)


def present_files(*, user_email: str, instance_scope: str = "default") -> list[dict]:
    """List the files in the user's outputs/ folder that can be shown to them."""
    from agent import workspace_tools

    return workspace_tools.list_presentable(user_email)


def bash(
    argv: list[str], *, user_email: str, instance_scope: str = "default"
) -> str:
    """Run one allowlisted command as an argv list (e.g. ['ls','-la']); no shell."""
    from agent import workspace_tools

    return workspace_tools.bash(user_email, argv)


# ---------------------------------------------------------------------------
# Python execution (agent.python_tool — subprocess sandbox with timeout)
# ---------------------------------------------------------------------------


def run_python(
    code: str, *, user_email: str, instance_scope: str = "default"
) -> str:
    """Run a Python snippet in the VM. It runs in your workspace, so it
    can read files you uploaded and write results (e.g. save to
    'outputs/…'). Returns the combined stdout/stderr."""
    from agent import python_tool, workspace_tools

    try:
        cwd = workspace_tools.work_dir(user_email)
    except Exception:  # noqa: BLE001
        cwd = None
    return python_tool.run_python(code, cwd=cwd)


def download_file(
    url: str, filename: str = "", *, user_email: str, instance_scope: str = "default"
) -> dict:
    """Download a file (software installer, dataset, asset) from an
    http/https URL into the user's workspace storage so it can be used
    for the task."""
    from agent import workspace_tools

    return workspace_tools.download(user_email, url, filename)


def pip_install(
    packages: list[str],
    upgrade: bool = False,
    *,
    user_email: str,
    instance_scope: str = "default",
) -> str:
    """Install Python packages into the VM so run_python can import them.
    Pass a list of names, e.g. ["pandas", "requests==2.32.0"]."""
    from agent import python_tool

    if isinstance(packages, str):
        packages = [p for p in packages.replace(",", " ").split() if p]
    return python_tool.pip_install(packages, upgrade=bool(upgrade))


def system_install(
    packages: list[str],
    manager: str = "",
    *,
    user_email: str,
    instance_scope: str = "default",
) -> str:
    """Install OS-level packages via the VM's package manager
    (winget/apt-get/brew). Pass a list of package ids."""
    from agent import python_tool

    if isinstance(packages, str):
        packages = [p for p in packages.replace(",", " ").split() if p]
    return python_tool.system_install(packages, manager=manager or None)


def vm_exec(
    argv: list[str], *, user_email: str, instance_scope: str = "default"
) -> str:
    """Run one allowlisted command in the agent's VM (its CLI). Pass an
    argv list, e.g. ["pip", "list"] or ["winget", "--version"]. No shell."""
    from agent import python_tool

    if isinstance(argv, str):
        import shlex

        argv = shlex.split(argv)
    return python_tool.vm_exec(argv)


# ---------------------------------------------------------------------------
# Internal APIs (agent.internal_apis — api_act stays two-phase)
# ---------------------------------------------------------------------------


def _integration_lookup(integration_id: str) -> dict | None:
    wh = _warehouse()
    _ensure(wh, "media.control.tool_integrations", _TOOL_INTEGRATIONS_COLS)
    try:
        rows = _sql(
            "SELECT integration_id, base_url, read_capability, act_capability, "
            "act_requires_confirm FROM media.control.tool_integrations "
            "WHERE integration_id = :i",
            {"i": integration_id},
        )
    except Exception:  # noqa: BLE001
        return None
    return rows[0] if rows else None


def _user_token_provider(integration_id: str) -> str:
    """Resolve the bearer token for an internal-API integration.

    Dev/test: the scripted fakes' secrets. Production: the Databricks secret
    scope. Returning the fake `dev-token` unconditionally (the previous
    behavior) meant every internal-API call in prod authenticated with a
    bogus token; a missing secret now fails loudly instead of silently using
    a fake one."""
    from alex import testing as _testing

    if _testing.is_dev():
        secrets = getattr(_testing.get_fakes(), "secrets", None) or {}
        return str(secrets.get(f"integration:{integration_id}", "dev-token"))
    import base64
    import re as _re

    from alex.config import get_settings

    settings = get_settings()
    key = "integration_" + _re.sub(r"[^A-Za-z0-9_.-]", "_", integration_id)
    try:
        from databricks.sdk import WorkspaceClient

        resp = WorkspaceClient().secrets.get_secret(
            scope=settings.secrets_scope, key=key
        )
        raw = getattr(resp, "value", None)
        if raw:
            return base64.b64decode(raw).decode("utf-8")
    except Exception:  # noqa: BLE001
        log.exception("secret lookup failed for integration %s", integration_id)
    raise RuntimeError(f"no secret configured for integration {integration_id!r}")


def api_read(
    integration_id: str,
    path: str,
    *,
    user_email: str,
    instance_scope: str = "default",
) -> str:
    """Read from a registered internal API as the signed-in user (GET only)."""
    from agent import internal_apis

    integration = _integration_lookup(integration_id)
    if not integration:
        return f"[error] unknown integration: {integration_id}"
    return internal_apis.api_read(
        integration, path, user_token_provider=_user_token_provider
    )


def api_act(
    integration_id: str,
    path: str,
    body_json: str = "{}",
    confirm_token: str | None = None,
    *,
    user_email: str,
    instance_scope: str = "default",
) -> dict:
    """Act on a registered internal API (two-phase: the first call stages
    the act and returns a confirmation card for the user; the POST runs
    only after the user confirms it in the UI).

    The token in the staged result is removed before the model sees
    it: a token the model can read is a token the model can hand back
    on the next call, which makes two-phase a one-phase with extra
    steps. The UI reads the token from the authenticated confirmation
    route, not from the tool result."""
    from agent import internal_apis

    try:
        body: Any = json.loads(body_json or "{}")
    except json.JSONDecodeError:
        return {"error": "body_json is not valid JSON"}
    if not isinstance(body, dict):
        return {"error": "body_json must be a JSON object"}
    integration = _integration_lookup(integration_id)
    if not integration:
        return {"error": f"unknown integration: {integration_id}"}
    out = internal_apis.api_act(
        integration,
        path,
        body,
        user_token_provider=_user_token_provider,
        user_email=user_email,
        confirm_token=confirm_token,
    )
    if isinstance(out, dict) and out.get("staged"):
        # Strip the credential: the model keeps the card (it can talk
        # about what is pending) but cannot redeem it.
        out.pop("confirmation_token", None)
    return out


# ---------------------------------------------------------------------------
# Office documents (agent.tools.office — Excel / PowerPoint / Word)
# Re-exported here so the registry's `_impl_tool` resolver finds them.
# ---------------------------------------------------------------------------

from agent.tools.office import (  # noqa: E402,F401
    create_document,
    create_presentation,
    create_spreadsheet,
    read_spreadsheet,
)

__all_office__ = [
    "create_document",
    "create_presentation",
    "create_spreadsheet",
    "read_spreadsheet",
]


# ---------------------------------------------------------------------------
# LSP code intelligence (agent.lsp_tool — jedi-backed, ast fallback)
# ---------------------------------------------------------------------------


def code_outline(
    path: str, *, user_email: str, instance_scope: str = "default"
) -> dict:
    """List the classes/functions/methods defined in a workspace source file."""
    from agent import lsp_tool

    return lsp_tool.outline(user_email, path)


def code_definition(
    path: str, line: int, column: int, *, user_email: str,
    instance_scope: str = "default",
) -> dict:
    """Go to the definition of the name at (line, column) in a workspace file."""
    from agent import lsp_tool

    return lsp_tool.definition(user_email, path, line, column)


def code_references(
    path: str, line: int, column: int, *, user_email: str,
    instance_scope: str = "default",
) -> dict:
    """Find all references to the name at (line, column) in a workspace file."""
    from agent import lsp_tool

    return lsp_tool.references(user_email, path, line, column)


def code_hover(
    path: str, line: int, column: int, *, user_email: str,
    instance_scope: str = "default",
) -> dict:
    """Show the signature and docstring of the name at (line, column)."""
    from agent import lsp_tool

    return lsp_tool.hover(user_email, path, line, column)


def code_diagnostics(
    path: str, *, user_email: str, instance_scope: str = "default"
) -> dict:
    """Report syntax/static errors in a workspace source file."""
    from agent import lsp_tool

    return lsp_tool.diagnostics(user_email, path)


def code_search_symbols(
    query: str, *, user_email: str, instance_scope: str = "default"
) -> dict:
    """Search for a symbol by name across all workspace source files."""
    from agent import lsp_tool

    return lsp_tool.workspace_symbols(user_email, query)


# ---------------------------------------------------------------------------
# Enterprise RAG + vector knowledge graph
# (agent.rag / agent.enterprise_graph)
# ---------------------------------------------------------------------------


def _principal(user_email: str):
    """Resolve the acting principal for an authorization decision."""
    from agent import access

    return access.load_principal(_sql, user_email)


def search_knowledge(
    query: str, k: int = 6, *, user_email: str, instance_scope: str = "default"
) -> dict:
    """Search the enterprise knowledge base; returns passages with citations."""
    from agent import rag
    from agent.memory import connection

    with connection() as conn:
        return rag.search(
            conn, query, user_email=user_email, k=max(1, min(k, 12)),
            principal=_principal(user_email),
        )


def list_knowledge(*, user_email: str, instance_scope: str = "default") -> dict:
    """List the documents in the enterprise knowledge base."""
    from agent import access, rag
    from agent.memory import connection

    principal = _principal(user_email)
    with connection() as conn:
        docs = rag.list_documents(conn, scopes=access.visible_scopes(principal))
    # A document is listed only if its classification is also readable.
    docs = [
        d for d in docs
        if access.can_read(
            principal,
            scope=d.get("scope", ""),
            classification=d.get("classification", "internal"),
        ).allowed
    ]
    return {"documents": docs, "count": len(docs)}


def ingest_document(
    title: str,
    text: str,
    source_ref: str = "",
    scope: str = "",
    *,
    user_email: str,
    instance_scope: str = "default",
    replace: bool = False,
) -> dict:
    """Add a document to the enterprise knowledge base so it can be retrieved.

    The scope the model names is *authorized*, not trusted: reads are
    principal-filtered everywhere, and a write path that accepted any
    scope string let a prompt-injected document land inside a
    restricted corpus (team:, user:<victim>) for its members to
    retrieve. Re-ingest also no longer silently replaces an existing
    governed document — the caller must ask for that explicitly.
    """
    from agent import access, rag

    target_scope = scope or rag.ENTERPRISE_SCOPE
    principal = _principal(user_email)
    decision = access.scope_allowed(principal, target_scope)
    if not decision.allowed:
        return {
            "ok": False,
            "error": (
                f"you may not publish into scope {target_scope!r}: "
                f"{decision.reason}"
            ),
        }
    from agent.memory import connection

    with connection() as conn:
        return rag.ingest(
            conn,
            title=title,
            text=text,
            source_ref=source_ref,
            scope=target_scope,
            metadata={"ingested_by": user_email},
            replace=replace,
        )


def ingest_workspace_file(
    path: str, *, user_email: str, instance_scope: str = "default"
) -> dict:
    """Index a file from your workspace into the knowledge base."""
    from agent import rag
    from agent.memory import connection

    with connection() as conn:
        return rag.ingest_workspace_file(conn, user_email, path)


def graph_entity(
    name: str, *, user_email: str, instance_scope: str = "default"
) -> dict:
    """Look up an entity (person, team, project, system) in the knowledge graph."""
    from agent import enterprise_graph as eg
    from agent.memory import connection

    with connection() as conn:
        matches = eg.resolve_entity(
            conn, name, scopes=eg.visible_scopes(user_email), k=3
        )
        facts = (
            eg.neighbors(
                conn, matches[0]["name"],
                scopes=eg.visible_scopes(user_email), hops=1, limit=20,
            )
            if matches
            else []
        )
    return {"query": name, "matches": matches, "facts": facts}


def graph_neighbors(
    entity: str, hops: int = 2, *, user_email: str, instance_scope: str = "default"
) -> dict:
    """Traverse the knowledge graph out from an entity (multi-hop)."""
    from agent import enterprise_graph as eg
    from agent.memory import connection

    with connection() as conn:
        edges = eg.neighbors(
            conn, entity, scopes=eg.visible_scopes(user_email),
            hops=max(1, min(int(hops), 4)), limit=40,
        )
    return {"entity": entity, "edges": edges, "count": len(edges)}


def graph_link(
    src: str,
    rel: str,
    dst: str,
    evidence: str = "",
    *,
    user_email: str,
    instance_scope: str = "default",
) -> dict:
    """Assert a relationship in the enterprise knowledge graph.

    Authorized, not trusted: the edge defaults to the enterprise scope,
    which every user's T3 and retrieval expansion reads — an
    unauthorized write here was an integrity hole and a cross-user
    injection channel. Model assertions carry the agent's source_ref
    and a capped confidence so the graph records WHO claimed it."""
    from agent import access as _access
    from agent import enterprise_graph as eg
    from agent.memory import connection

    principal = _principal(user_email)
    decision = _access.scope_allowed(principal, eg.ENTERPRISE_SCOPE)
    if not decision.allowed:
        return {
            "ok": False,
            "error": (
                "writing to the enterprise graph requires organisation "
                f"scope ({decision.reason}); your edges land in your "
                "own graph via add_edge"
            ),
        }
    with connection() as conn:
        edge = eg.upsert_edge(
            conn,
            src,
            rel,
            dst,
            scope=eg.ENTERPRISE_SCOPE,
            evidence=evidence,
            confidence=0.6,
            source_ref=f"agent:{user_email}",
        )
    return edge.to_dict()


def graph_about_me(*, user_email: str, instance_scope: str = "default") -> dict:
    """Show what the knowledge graph records about the signed-in user."""
    from agent import enterprise_graph as eg
    from agent.memory import connection

    with connection() as conn:
        return {"user": user_email, "facts": eg.user_context(conn, user_email)}


# ---------------------------------------------------------------------------
# Standing orders + the best-practice library
# (agent.standing_orders / agent.practices)
# ---------------------------------------------------------------------------


def list_standing_orders(
    *, user_email: str, instance_scope: str = "default"
) -> dict:
    """List the standing orders assigned to you, with which are runnable."""
    from agent import standing_orders

    orders = standing_orders.for_user(_sql, user_email)
    return {
        "orders": [
            {
                "order_id": o.order_id,
                "title": o.title,
                "instruction": o.instruction,
                "trigger": o.trigger,
                "runnable": o.trigger == standing_orders.TRIGGER_MANUAL,
            }
            for o in orders
        ],
        "count": len(orders),
    }


def run_standing_order(
    order_id: str,
    note: str = "",
    *,
    user_email: str,
    instance_scope: str = "default",
) -> dict:
    """Execute one of your standing orders now and return its result."""
    from agent import standing_orders

    principal = _principal(user_email)
    try:
        run = standing_orders.run(
            _sql, principal, order_id, note=note, background=False
        )
    except (LookupError, PermissionError) as exc:
        return {"ok": False, "error": str(exc)}
    return {
        "ok": run.status == "done",
        "run_id": run.run_id,
        "status": run.status,
        "summary": run.summary,
        "capabilities_used": run.capabilities,
    }


def search_practices(
    query: str = "",
    *,
    user_email: str,
    instance_scope: str = "default",
) -> dict:
    """Find the best practices that govern a situation, with their authority."""
    from agent import practices

    principal = _principal(user_email)
    matches = practices.match(_sql, principal, query=query, limit=8)
    return {
        "query": query,
        "practices": [m.to_dict() for m in matches],
        "mandatory": sum(1 for m in matches if m.practice.is_mandatory),
    }


def propose_practice(
    title: str,
    guidance: str,
    rationale: str = "",
    applies_to: str = "",
    *,
    user_email: str,
    instance_scope: str = "default",
) -> dict:
    """Propose a new best practice for a governor to ratify (advisory only)."""
    from agent import practices

    try:
        p = practices.upsert(
            _sql,
            title=title,
            guidance=guidance,
            rationale=rationale,
            # A practice the agent proposes is never mandatory and never
            # in force until a governor raises it — learning cannot
            # legislate.
            authority=practices.ADVISORY,
            applies_to=[s.strip() for s in applies_to.split(",") if s.strip()],
            owner=user_email,
            source_ref=f"proposed-by-agent:{user_email}",
        )
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    return {
        "ok": True,
        "practice_id": p.practice_id,
        "authority": p.authority,
        "note": "Recorded as advisory; a governor must raise it to take effect.",
    }


# ---------------------------------------------------------------------------
# Desktop control (agent.desktop) — the machine Jarvis lives on
# ---------------------------------------------------------------------------


def desktop_screenshot(
    monitor: int = 1, *, user_email: str, instance_scope: str = "default"
) -> dict:
    """Capture the screen to outputs/ and read its text (OCR when available)."""
    from agent import desktop

    return desktop.screenshot(user_email, int(monitor))


def desktop_find_text(
    text: str, monitor: int = 1, *, user_email: str, instance_scope: str = "default"
) -> dict:
    """Find text on the screen; returns the center coordinates to click."""
    from agent import desktop

    return desktop.find_text(user_email, text, int(monitor))


def desktop_click(
    x: int, y: int, button: str = "left", clicks: int = 1,
    *, user_email: str, instance_scope: str = "default",
) -> dict:
    """Click the mouse at screen coordinates (left|right|middle, 1-3 clicks)."""
    from agent import desktop

    return desktop.click(x, y, button, clicks)


def desktop_type(
    text: str, *, user_email: str, instance_scope: str = "default"
) -> dict:
    """Type text into the focused window (max 2000 characters)."""
    from agent import desktop

    return desktop.type_text(text)


def desktop_key(
    key: str, modifiers: list[str] | None = None,
    *, user_email: str, instance_scope: str = "default",
) -> dict:
    """Press one key with optional modifiers, e.g. key='s', modifiers=['ctrl']."""
    from agent import desktop

    if isinstance(modifiers, str):
        modifiers = [m for m in modifiers.replace(",", " ").split() if m]
    return desktop.press_key(key, modifiers)


def desktop_hotkey(
    keys: list[str], *, user_email: str, instance_scope: str = "default"
) -> dict:
    """Press a key combination, e.g. ['ctrl', 'shift', 'p']."""
    from agent import desktop

    if isinstance(keys, str):
        keys = [k for k in keys.replace("+", " ").replace(",", " ").split() if k]
    return desktop.hotkey(keys)


def desktop_scroll(
    direction: str = "down", amount: int = 3,
    *, user_email: str, instance_scope: str = "default",
) -> dict:
    """Scroll the mouse wheel up or down by N clicks."""
    from agent import desktop

    return desktop.scroll(direction, amount)


# ---------------------------------------------------------------------------
# Home automation (agent.home)
# ---------------------------------------------------------------------------


def home_control(
    device: str, action: str, value: str = "",
    *, user_email: str, instance_scope: str = "default",
) -> dict:
    """Control a smart-home device: Home Assistant entity id (light.desk) with
    on|off|toggle|brightness|color|temperature, or a configured command."""
    from agent import home

    return home.control(device, action, value or None)


def home_status(
    device: str = "", *, user_email: str, instance_scope: str = "default"
) -> dict:
    """Current state of one Home Assistant device, or all of them."""
    from agent import home

    return home.status(device or None)


# ---------------------------------------------------------------------------
# The deferred tool library (agent.vm deferred working set)
# ---------------------------------------------------------------------------


def find_tools(
    query: str = "", *, user_email: str, instance_scope: str = "default"
) -> dict:
    """Search the extended tool library by keywords (e.g. 'music compose',
    'docker', 'calendar'); matching tools become callable right away. Call
    with no keywords to list the library's categories and examples."""
    from agent.swarm import current_vm
    from agent.tools.registry import TOOL_REGISTRY, input_schema_for
    from agent.vm import score_tool

    words = {w for w in str(query or "").lower().replace("_", " ").split() if len(w) >= 3}
    if not words:
        # Discovery, not search: a model probing the library usually
        # calls this with no arguments. The old error said "give a few
        # keywords" and taught it nothing about what exists.
        by_category: dict[str, list[str]] = {}
        for spec in TOOL_REGISTRY.values():
            if not getattr(spec, "deferred", False):
                continue
            parts = str(spec.capability).split(".")
            category = (
                parts[1] if parts and parts[0] == "jarvis" and len(parts) > 2
                else (parts[0] if parts else "misc")
            )
            by_category.setdefault(category, []).append(spec.name)
        return {
            "ok": True,
            "tools": [],
            "pinned": [],
            "library": [
                {"category": cat, "count": len(names),
                 "examples": sorted(names)[:3]}
                for cat, names in sorted(by_category.items())
            ],
            "note": (
                "call find_tools with a few keywords (e.g. 'music compose') "
                "to make matching tools callable in this turn"
            ),
        }
    ranked = sorted(
        ((score_tool(spec, words), spec) for spec in TOOL_REGISTRY.values()
         if getattr(spec, "deferred", False)),
        key=lambda t: -t[0],
    )
    hits = [spec for score, spec in ranked if score > 0][:10]
    vm = current_vm()
    pinned: list[str] = []
    if vm is not None and hits:
        try:
            pinned = list(vm.pin_tools([spec.name for spec in hits]) or [])
        except Exception:  # noqa: BLE001
            log.exception("find_tools: pinning failed")
    if not hits:
        note = "no matching tool in the library"
    elif len(pinned) == len(hits):
        note = "these tools are now callable in this turn"
    elif pinned:
        note = (
            f"{len(pinned)} of {len(hits)} matches are callable in this "
            "turn; the rest need permissions this user does not have"
        )
    else:
        note = ("matches found but none are usable in this turn "
                "(permissions or plan mode)")
    return {
        "ok": True,
        "tools": [
            {"name": spec.name, "description": spec.description,
             "parameters": input_schema_for(spec)}
            for spec in hits
        ],
        "pinned": pinned,
        "note": note,
    }
