"""
Dev-mode warehouse seed.

In production, the views `media.alex.inbox_view` and `media.alex.task_board_view`
are real Databricks Unity Catalog views over the underlying tables. In dev, the
in-memory warehouse has no view support, so we register both the views and
their base tables. The application code is the same in both modes; only the
warehouse contents differ.

This module is imported lazily by `app.app._sql` when `is_dev()` is true.
Seeding runs exactly once per warehouse instance: the `_seeded` flag is set
unconditionally on the warehouse object, so a reseed never wipes rows that
were inserted after startup. Tests can clear `wh._seeded` to force a reseed.

The table schemas below are reconciled with the exact column lists queried by
`app/app.py`. Demo rows are seeded for `DEMO_USER` so the dev UI has content.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

log = logging.getLogger("app._dev_seed")

DEMO_USER = "demo@alex.dev"

_NOW = datetime(2026, 1, 5, 8, 30, 0)


def _ts(hours_ago: float = 0.0) -> str:
    return (_NOW - timedelta(hours=hours_ago)).isoformat(sep=" ")


# Column-for-column with app_schema.sql. The old lists carried columns
# the DDL does not have (thread_id on the view; to_emails/cc_emails/
# pii_flags/redacted_body/... on the base table) — dev happily wrote
# them while production would have rejected the same INSERT.
INBOX_VIEW_COLS = [
    "email_id",
    "owner_email",
    "priority",
    "subject",
    "received_at",
    "summary",
    "from_email",
]
INBOX_BASE_COLS = [
    "email_id",
    "owner_email",
    "priority",
    "subject",
    "body",
    "from_email",
    "thread_id",
    "received_at",
    "summary",
]
# Both mirror the governance DDLs exactly (app_schema.sql task_board_view;
# permissions.sql media.alex.tasks). The old lists carried columns that do
# not exist in the DDL (task_id on the base table, assignee, shared_with,
# last_updated_at), which masked prod-failing SQL in dev.
TASK_BOARD_VIEW_COLS = [
    "owner_email",
    "status",
    "task_id",
    "title",
    "priority",
    "due_at",
]
TASK_BASE_COLS = [
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
MORNING_BRIEFS_COLS = [
    "user_email",
    "brief_date",
    "brief",
    "standing_orders",
    "priority_emails",
    "tasks_due",
    "created_at",  # the DDL's name; 'generated_at' never existed in prod
]

# media.control.sessions — matches the INSERT/SELECT in app.py (/api/sessions).
CONTROL_SESSIONS_COLS = [
    "session_id",
    "owner_email",
    "kind",
    "title",
    "parent_id",
    "participants",
    "model",
    "created_at",
    "last_active_at",
]

# media.control.standing_order_briefs — matches the control_plane.sql DDL
# and pipelines/standing_orders_prep.py (brief_id PK; no 'title' column).
STANDING_ORDER_BRIEFS_COLS = [
    "brief_id",
    "order_id",
    "user_email",
    "brief_date",
    "compiled_context",
    "suggested_actions",
    "consumed",
    "created_at",
]

# media.alex.agent_actions — the FULL DDL shape (permissions.sql), both
# column families: legacy (id/actor/action_type/...) and harness
# (action_id/tool/user_email/...). The VM's audit writer INSERTs both
# families in one row; registering only the legacy half meant the dev
# warehouse absorbed columns production has while the guard tests
# believed the seed mirrored the DDL.
AGENT_ACTIONS_COLS = [
    "id",
    "actor",
    "ts",
    "action_type",
    "target_id",
    "detail",
    "outcome",
    "tool_name",
    "session_id",
    "tool_args",
    "action_id",
    "tool",
    "capability",
    "user_email",
    "instance_scope",
    "input_json",
    "output_json",
    "ok",
]

# media.control.constitution — matches /api/constitution.
# Column-for-column with governance/constitution.sql. `machine_rule` and
# `severity` were missing, and load_articles() selects both — so the read
# failed in dev and the L2 interceptor ran with no articles at all. The
# constitution was visible in the UI and enforced nothing.
CONSTITUTION_COLS = [
    "article_id",
    "article_type",
    "title",
    "text",
    "machine_rule",
    "severity",
    "version",
    "superseded_by",
    "effective_from",
]

# media.control.skills — column-for-column with control_plane.sql.
SKILLS_COLS = [
    "skill_id",
    "owner_email",
    "name",
    "trigger_pattern",
    "procedure",
    "uses_capabilities",
    "status",
    "created_at",
    "ratified_by",
    "ratified_at",
    "last_used",
    "source_kind",
    "success_count",
]

# id / edge_id are the PKs the forget flow matches on — must be present so
# the UI can pass a real primary key to /api/transparency/forget.
VECTOR_MEMORY_MIRROR_COLS = [
    "id",
    "owner_email",
    "kind",
    "content",
    "salience",
    "source_ref",
    "created_at",
    "superseded_by",
    "last_recalled",
]
KG_MIRROR_COLS = [
    "edge_id", "owner_email", "src", "rel", "dst", "evidence",
    "confidence", "last_seen",
]
FORGET_REQUESTS_COLS = [
    "request_id",
    "owner_email",
    "item_id",
    "reason",
    "ts",
    "applied",
    "applied_at",
]
EFFECTIVE_ACL_COLS = ["user_email", "capability", "expires_at"]
DELEGATIONS_COLS = [
    "delegation_id",
    "supervisor_email",
    "report_email",
    "capability",
    "justification",
    "second_approver",
    "granted_at",
    "expires_at",
    "revoked",
    "revoked_by",
    "revoked_at",
]
SUPERVISION_COLS = ["supervisor_email", "report_email", "since", "until"]
USER_ROLES_COLS = [
    "user_email",
    "role",  # 'member' | 'manager' | 'admin'
    "team",
    "granted_scope",
    "clearance",
    "granted_at",
    "granted_by",
]
DELEGABLE_CAPABILITIES_COLS = [
    "capability",
    "requires_second_approver",
    "max_validity_days",
    "rationale",
]
INSTANCE_LIBRARY_COLS = ["file_id", "owner_email", "pinned_summary", "added_at"]
APPROVED_FILES_COLS = [
    "file_id", "path", "source_channel", "approved_at", "revoked",
    "approved_by", "expires_at", "revoked_at", "revoked_by", "sha256",
    "size_bytes",
]
MCP_SERVERS_COLS = [
    "server_id", "name", "transport", "command", "args", "url",
    "capability", "scope", "trust", "enabled", "timeout_s", "secret_ref",
    "description", "registered_by", "registered_at",
]
MCP_PINS_COLS = [
    "server_id", "fingerprint", "approved", "tool_count", "first_seen",
    "changed_at", "approved_at", "approved_by",
]
# Column-for-column with governance/control_plane.sql.
STANDING_ORDER_RUNS_COLS = [
    "run_id", "order_id", "session_id", "started_by", "status",
    "started_at", "last_heartbeat", "ended_at", "capabilities", "summary",
]
BEST_PRACTICES_COLS = [
    "practice_id", "title", "guidance", "authority", "applies_to",
    "capabilities", "rationale", "examples", "anti_patterns", "scope",
    "classification", "owner", "version", "supersedes", "superseded_by",
    "active", "source_ref", "created_at", "updated_at",
]
STANDING_ORDERS_COLS = [
    "order_id", "owner_email", "title", "instruction", "schedule",
    "audience", "allowed_capabilities", "active", "created_by",
    "trigger", "ambient", "applies_to", "created_at",
]
# Column-for-column with governance/taskboard.sql. `last_heartbeat` and
# `steps` are what make an interrupted run recoverable rather than a row
# stuck at 'running' for ever.
TASK_CREW_COLS = ["task_id", "member", "role", "added_at"]
# Column-for-column with governance/taskboard.sql (task_plans).
TASK_PLANS_COLS = ["task_id", "step_no", "title", "status", "note", "updated_at"]
# Column-for-column with governance/taskboard.sql (task_acceptance,
# task_deps, task_proposals).
TASK_ACCEPTANCE_COLS = ["task_id", "crit_no", "text", "status", "note", "updated_at"]
TASK_DEPS_COLS = ["task_id", "blocked_by", "added_by", "added_at"]
TASK_PROPOSALS_COLS = [
    "proposal_id",
    "owner_email",
    "title",
    "description",
    "priority",
    "due_at",
    "source",
    "source_ref",
    "status",
    "task_id",
    "created_at",
    "decided_at",
]
TASK_MESSAGES_COLS = ["message_id", "task_id", "author", "body", "kind", "ts"]
# Column-for-column with governance/taskboard.sql (task_questions) —
# the agent's elicitation rows: ask, wait, a person answers.
TASK_QUESTIONS_COLS = [
    "question_id",
    "task_id",
    "run_id",
    "asked_by",
    "question",
    "options",
    "status",
    "answer",
    "answered_by",
    "asked_at",
    "answered_at",
]
TASK_RUNS_COLS = [
    "run_id",
    "task_id",
    "session_id",
    "started_by",
    "status",
    "started_at",
    "last_heartbeat",
    "ended_at",
    "summary",
    "steps",
]
# Column-for-column with governance/permissions.sql (media.ops.run_metrics):
# the per-working-turn accounting ledger the exec performance view reads.
RUN_METRICS_COLS = [
    "run_id",
    "task_id",
    "turn",
    "llm_turns",
    "tool_calls",
    "elapsed_s",
    "waited_s",
    "prompt_tokens",
    "completion_tokens",
    "ts",
]
WORKFLOW_CATALOG_COLS = [
    "workflow_id",
    "name",
    "description",
    "capability",
    "active",
    "integration_id",
    "input_schema",
    "requires_dual_approval",
    "created_at",
]


def seed_dev_warehouse(wh: Any) -> None:
    """Register the dev tables (plus demo rows) on the given warehouse.

    Idempotent: the `_seeded` flag is set unconditionally so a second call
    returns early and never wipes rows written after the first seed.
    """
    if getattr(wh, "_seeded", False):
        return
    wh._seeded = True

    u = DEMO_USER

    wh.register(
        "media.alex.inbox_view",
        INBOX_VIEW_COLS,
        rows=[
            (
                "e-1", u, 1, "Q1 budget sign-off needed", _ts(2),
                "Finance needs your approval by Friday.", "cfo@media.co",
            ),
            (
                "e-2", u, 2, "Partner deck review", _ts(5),
                "Draft v3 attached; feedback requested.", "partners@media.co",
            ),
            (
                "e-3", u, 3, "Team offsite logistics", _ts(26),
                "Venue options for March.", "ops@media.co",
            ),
        ],
    )
    wh.register(
        "media.alex.inbox",
        INBOX_BASE_COLS,
        pk=["email_id"],
        rows=[
            (
                "e-1", u, 1, "Q1 budget sign-off needed",
                "Hi,\n\nFinance has finalized the Q1 budget and we need your "
                "approval by Friday to keep the vendor commitments on track. "
                "The headline number is $47M across the DTC channels, unchanged "
                "from the plan you reviewed last month.\n\nCould you confirm "
                "sign-off, or let me know if you'd like to walk through the "
                "channel allocation first?\n\nThanks,\nFinance Ops",
                "cfo@media.co", "th-1", _ts(2),
                "Finance needs your approval by Friday.",
            ),
            (
                "e-2", u, 2, "Partner deck review",
                "Hey,\n\nDraft v3 of the partner deck is attached. The main "
                "changes are the new audience-sizing slide and the revised "
                "channel-mix chart. Would love your feedback before we send it "
                "to the partner on Thursday.\n\nBest,\nBrand Team",
                "partners@media.co", "th-2", _ts(5),
                "Draft v3 attached; feedback requested.",
            ),
            (
                "e-3", u, 3, "Team offsite logistics",
                "Hi team,\n\nWe're locking the March offsite. Venue options: "
                "(1) downtown conference center, (2) lakeside retreat. Both fit "
                "the budget. Reply with a preference and any dietary needs.\n\n"
                "Thanks,\nOps",
                "ops@media.co", "th-3", _ts(26),
                "Venue options for March.",
            ),
        ],
    )
    wh.register(
        "media.alex.task_board_view",
        TASK_BOARD_VIEW_COLS,
        rows=[
            (u, "open", "t-1", "Approve Q1 budget", 1, _ts(-48)),
            (u, "open", "t-2", "Review partner deck", 2, _ts(-72)),
            (u, "in_progress", "t-3", "Draft offsite agenda", 3, None),
        ],
    )
    wh.register("media.alex.tasks", TASK_BASE_COLS, pk=["id"])
    # NOTE: morning_briefs and standing_order_briefs are seeded empty on
    # purpose — the /api/briefs WHERE clauses (brief_date = current_date(),
    # consumed = false) are not parseable by the in-memory warehouse, which
    # then returns ALL rows regardless of user. Empty keeps dev correct.
    wh.register("media.alex.morning_briefs", MORNING_BRIEFS_COLS)
    wh.register(
        "media.control.standing_order_briefs",
        STANDING_ORDER_BRIEFS_COLS,
        pk=["order_id"],
    )
    wh.register(
        "media.control.sessions",
        CONTROL_SESSIONS_COLS,
        pk=["session_id"],
        rows=[
            ("s-demo-1", u, "personal", "Welcome chat", None, None, "auto", _ts(50), _ts(1)),
        ],
    )
    # (media.alex.sessions is deliberately NOT registered: the session
    # store is media.control.sessions, and a dev-only alias table with
    # no production DDL is exactly the drift class this seed guards
    # against.)
    # Column-for-column with control_plane.sql. This table had prod DDL
    # and no dev registration at all — invisible while the session
    # manager kept turns in an in-process dict, load-bearing now that
    # dev uses the same storage model as production.
    wh.register(
        "media.control.session_turns",
        ["session_id", "turn_no", "role", "content", "ts"],
        pk=["session_id", "turn_no"],
    )
    wh.register(
        "media.alex.agent_actions",
        AGENT_ACTIONS_COLS,
        rows=[
            ("a-1", u, _ts(3), "tool_call", "tab-1", "Opened https://status.media.co", "ok", "browser_navigate",
             None, None, "a-1", "browser_navigate", "browser.navigate", u, "default", None, None, 1),
            ("a-2", u, _ts(2.5), "tool_call", "tab-1", "Extracted incident summary (PII redacted)", "ok", "browser_extract",
             None, None, "a-2", "browser_extract", "browser.extract", u, "default", None, None, 1),
            ("a-3", u, _ts(2), "tool_call", "th-1", "Drafted reply to Q1 budget thread (not sent)", "ok", "draft_email_reply",
             None, None, "a-3", "draft_email_reply", "email.draft", u, "default", None, None, 1),
            ("a-4", u, _ts(1.5), "tool_call", None, "Blocked: send requires rights clearance first", "denied", "send_email",
             None, None, "a-4", "send_email", "email.send", u, "default", None, None, 0),
        ],
    )
    wh.register(
        "media.control.constitution",
        CONSTITUTION_COLS,
        rows=[
            (
                "A1", "law", "User primacy",
                "The agent acts only on behalf of the signed-in user.",
                "{}", "blocking", 1, None, _ts(9000),
            ),
            (
                "A3", "law", "Audited actions",
                "Every governed tool call is written to the audit log.",
                "{}", "blocking", 1, None, _ts(9000),
            ),
            (
                "A8", "law", "Two-phase submits",
                "Destructive form submits require an explicit user confirmation.",
                "{}", "blocking", 1, None, _ts(9000),
            ),
            # One article with a machine rule the interceptor can act on,
            # so a dev environment exercises L2 end to end — a violation is
            # detected, recorded, and shows up in Oversight — instead of
            # only displaying prose.
            #
            # Deliberately `warning`, not `blocking`. Blocking shell here
            # would quietly withdraw a capability the VM is meant to have,
            # and a seed is not the place to make that policy decision.
            # Raise it to `blocking` in an environment that wants the rule,
            # rather than assuming this default was an oversight.
            (
                "A9", "prohibition", "Shell use is noted",
                "Shell commands are recorded as constitutional events so "
                "unattended shell use is visible to oversight.",
                '{"deny_capabilities": ["compute.shell"]}',
                "warning", 1, None, _ts(9000),
            ),
        ],
    )
    # Column-for-column with control_plane.sql. The old lists invented
    # columns ('scope', 'granted_at' on instance_acl) the DDL never had.
    wh.register(
        "media.control.employee_acl",
        ["user_email", "capability", "granted_at", "granted_by",
         "expires_at", "notes"],
    )
    wh.register(
        "media.control.instance_acl",
        ["instance_scope", "capability", "rationale"],
    )
    # Column-for-column with governance/constitution.sql. It previously
    # declared a different shape (violation_id/rule_key/explanation), so
    # the insert in constitution.make_violation_logger did not match any
    # column here and every dev violation write failed — silently, since
    # violation logging is best-effort and swallows its exceptions. Dev
    # therefore showed a clean record no matter what the agent did.
    wh.register(
        "media.control.constitution_violations",
        [
            "id",
            "ts",
            "user_email",
            "session_id",
            "article_id",
            "action",
            "disposition",
            "detail",
        ],
        pk=["id"],
    )
    wh.register(
        "media.control.skills",
        SKILLS_COLS,
        pk=["skill_id"],
        rows=[
            (
                "sk-1", u, "Weekly inbox triage", "triage my inbox",
                "1. Rank unread mail. 2. Draft replies for P1. 3. File the rest.",
                "email.read email.draft", "proposed", _ts(40), None,
                None, None, "taught", 0,
            ),
            (
                "sk-2", u, "Status page check", "check service status",
                "1. Open status.media.co. 2. Extract incidents. 3. Summarize.",
                "browser.navigate browser.extract", "active", _ts(200), u,
                _ts(199), _ts(20), "taught", 3,
            ),
        ],
    )
    # Columns match governance/control_plane.sql exactly. They previously
    # did not (standing_order_id/name/description/actions_json), so the
    # context engine's real T1 query could never resolve in dev — dev
    # silently exercised a table shape production does not have.
    wh.register(
        "media.control.standing_orders", STANDING_ORDERS_COLS, pk=["order_id"]
    )
    wh.register(
        "media.control.vector_memory_mirror",
        VECTOR_MEMORY_MIRROR_COLS,
        rows=[
            (
                "vm-1", u, "preference", "Prefers bullet-point summaries under 5 items.",
                0.9, "chat:s-demo-1", _ts(300), None, None,
            ),
            ("vm-2", u, "fact", "Weekly leadership sync is Mondays 9am.", 0.7, "calendar", _ts(500), None, None),
        ],
    )
    wh.register(
        "media.control.kg_mirror",
        KG_MIRROR_COLS,
        rows=[
            ("kg-1", u, "demo", "reports_to", "cfo@media.co", "org chart", 0.95, _ts(300)),
            ("kg-2", u, "demo", "works_on", "Q1 budget", "email th-1", 0.8, _ts(300)),
        ],
    )
    wh.register("media.control.forget_requests", FORGET_REQUESTS_COLS, pk=["request_id"])
    # Grant the demo user the FULL capability set — every capability any
    # registered tool needs — so the whole tool set (Office, compute,
    # package install, downloads, browser, tasks, internal APIs) is
    # available in dev.  Derived from the registry so it never drifts.
    from agent.tools.registry import TOOL_REGISTRY as _REG

    _all_caps = sorted({s.capability for s in _REG.values()})
    _extra = [
        "email.read", "email.draft", "browser.navigate",
        # Guidance + MCP capabilities are not attached to any built-in
        # tool spec, so they are granted explicitly.
        "mcp.use", "practices.write",
    ]
    _acl_rows = [(u, c, None) for c in sorted(set(_all_caps) | set(_extra))]
    wh.register(
        "media.control.effective_employee_acl",
        EFFECTIVE_ACL_COLS,
        rows=_acl_rows,
    )
    wh.register("media.control.delegations", DELEGATIONS_COLS, pk=["delegation_id"])
    wh.register("media.control.supervision", SUPERVISION_COLS)
    # Roles drive the access-control decision point (agent/access.py).
    # The demo user is a manager: enough to see confidential material
    # and their team's scope, not enough to reach restricted.
    wh.register(
        "media.control.user_roles",
        USER_ROLES_COLS,
        rows=[(u, "manager", "platform", None, None, _ts(2000), "admin@media.co")],
    )
    wh.register(
        "media.control.delegable_capabilities",
        DELEGABLE_CAPABILITIES_COLS,
        rows=[
            ("email.send", True, 30, "External sends are irreversible."),
            ("browser.submit", True, 14, "Form submits act on real systems."),
            ("email.draft", False, 90, "Drafts stay inside the tenant."),
        ],
    )
    wh.register(
        "media.control.instance_library",
        INSTANCE_LIBRARY_COLS,
        rows=[("f-1", u, "Brand guidelines v4 (pinned)", _ts(100))],
    )
    wh.register(
        "media.control.approved_files",
        APPROVED_FILES_COLS,
        rows=[(
            "f-1", "uploads/brand-guidelines-v4.pdf", "upload", _ts(100),
            False, u, None, None, None, None, None,
        )],
    )
    # Learning + reasoning tables (POST /api/feedback, visible reasoning,
    # internal-API tool integrations).
    wh.register(
        "media.control.suggestion_feedback",
        ["id", "user_email", "suggestion_id", "feedback", "comment", "ts"],
        pk=["id"],
    )
    wh.register(
        "media.control.thinking_traces",
        [
            "id",
            "ts",
            "user_email",
            "session_id",
            "turn_no",
            "phase",
            "raw_thinking",
            "digest",
        ],
        pk=["id"],
    )
    wh.register(
        "media.control.tool_integrations",
        [
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
        ],
        pk=["integration_id"],
    )
    wh.register(
        "media.control.workflow_catalog",
        WORKFLOW_CATALOG_COLS,
        rows=[
            ("wf-1", "morning_brief", "Compile the morning brief",
             "email.read tasks.read", True, None, None, False, _ts(1000)),
        ],
    )
    # Task board collaboration tables (agent/taskboard.py).
    wh.register("media.control.task_crew", TASK_CREW_COLS)
    wh.register("media.control.task_messages", TASK_MESSAGES_COLS, pk=["message_id"])
    wh.register(
        "media.control.task_questions", TASK_QUESTIONS_COLS, pk=["question_id"]
    )
    wh.register("media.control.task_runs", TASK_RUNS_COLS, pk=["run_id"])
    wh.register(
        "media.control.task_plans", TASK_PLANS_COLS, pk=["task_id", "step_no"]
    )
    wh.register(
        "media.control.task_acceptance",
        TASK_ACCEPTANCE_COLS,
        pk=["task_id", "crit_no"],
    )
    wh.register(
        "media.control.task_deps", TASK_DEPS_COLS, pk=["task_id", "blocked_by"]
    )
    wh.register(
        "media.control.task_proposals", TASK_PROPOSALS_COLS, pk=["proposal_id"]
    )
    wh.register(
        "media.control.standing_order_runs", STANDING_ORDER_RUNS_COLS, pk=["run_id"]
    )
    wh.register("media.control.best_practices", BEST_PRACTICES_COLS, pk=["practice_id"])
    wh.register("media.control.mcp_servers", MCP_SERVERS_COLS, pk=["server_id"])
    wh.register(
        "media.control.user_preferences",
        ["user_email", "name", "value", "updated_at"],
        pk=["user_email", "name"],
    )
    wh.register(
        "media.control.workspace_policy",
        ["name", "value", "rationale", "set_by", "set_at"],
        pk=["name"],
    )
    wh.register("media.control.mcp_tool_pins", MCP_PINS_COLS, pk=["server_id"])
    wh.register(
        "media.ops.run_metrics", RUN_METRICS_COLS, pk=["run_id", "turn"]
    )
    wh.register(
        "media.control.supervision_findings",
        ["run_id", "user_email", "code", "severity", "summary", "detail",
         "source", "found_at", "acknowledged_by", "acknowledged_at"],
    )
    _seed_enterprise_knowledge(u)
    _seed_board(wh, u)
    _seed_guidance(wh, u)
    _seed_mcp(wh)


# ---------------------------------------------------------------------------
# Enterprise knowledge: a small, realistic corpus + the graph that connects
# it. This is what makes the RAG and knowledge-graph surfaces demonstrable
# on a fresh dev machine instead of showing an empty state.
# ---------------------------------------------------------------------------

_DEV_DOCUMENTS = [
    (
        "Project Atlas — delivery brief",
        "wiki/atlas-brief",
        "internal",
        """# Project Atlas

Project Atlas is the Q1 platform migration, owned by the Data Platform team.
The delivery deadline is March 31 and the approved budget is 1.2M USD.

## Scope

Atlas moves the reporting stack off the legacy warehouse and onto the
lakehouse. Downstream consumers are Finance (revenue reporting) and the
Customer Success dashboards.

## Risks

The largest risk is the legacy billing connector, which depends on the
Vertex API. A second risk is headcount: two of the five engineers are
shared with Project Borealis.
""",
    ),
    (
        "Expenses policy",
        "wiki/expenses-policy",
        "public",
        """# Expenses policy

Employees may expense travel booked at least fourteen days in advance.
Receipts above 75 USD require manager approval before reimbursement.
Client entertainment requires pre-approval from a director regardless of
amount. Reimbursement is paid in the payroll run following approval.
""",
    ),
    (
        "Billing connector runbook",
        "wiki/billing-runbook",
        "internal",
        """# Billing connector runbook

The Vertex API connector synchronises invoices every fifteen minutes.

## On failure

Check the dead-letter queue first. If the queue is empty the failure is
upstream: escalate to the on-call rota. Do not replay invoices manually —
replay is idempotent only within the same billing period.
""",
    ),
    (
        "FY26 compensation bands",
        "wiki/comp-bands",
        "restricted",
        """# FY26 compensation bands

Platform engineering salary bands for FY26, by level. This document is
restricted: it is readable only by administrators.

L3 90-120k. L4 120-155k. L5 155-195k. L6 195-240k.
""",
    ),
    (
        "Customer: Northwind Trading",
        "wiki/northwind",
        "confidential",
        """# Northwind Trading

Northwind Trading is a tier-one customer on an annual contract worth
480k USD, renewing in June. Their primary integration is the Vertex API.

Renewal risk is currently rated medium: they raised three P1 incidents in
Q4, all related to invoice sync latency.
""",
    ),
]

# (src, rel, dst, entity type of dst)
_DEV_GRAPH = [
    ("Data Platform", "owns", "Project Atlas", "project"),
    ("Project Atlas", "depends_on", "Vertex API", "system"),
    ("Project Atlas", "part_of", "Lakehouse Migration", "project"),
    ("Northwind Trading", "uses", "Vertex API", "system"),
    ("Finance", "uses", "Project Atlas", "project"),
    ("Sarah Chen", "part_of", "Data Platform", "team"),
    ("Sarah Chen", "responsible_for", "Project Atlas", "project"),
]

_DEV_ENTITY_TYPES = {
    "Data Platform": "team",
    "Finance": "team",
    "Sarah Chen": "person",
    "Northwind Trading": "customer",
    "Vertex API": "system",
    "Project Atlas": "project",
    "Lakehouse Migration": "project",
}


def _seed_enterprise_knowledge(user_email: str) -> None:
    """Ingest the demo corpus and build the demo graph. Best-effort: a
    failure here must never stop the app from booting."""
    try:
        from agent import enterprise_graph as eg
        from agent import rag

        for title, source_ref, classification, text in _DEV_DOCUMENTS:
            rag.ingest(
                None,
                title=title,
                text=text,
                source_ref=source_ref,
                classification=classification,
                metadata={"seeded": True},
            )
        for name, etype in _DEV_ENTITY_TYPES.items():
            eg.upsert_entity(None, name, etype=etype)
        for src, rel, dst, _ in _DEV_GRAPH:
            eg.upsert_edge(
                None, src, rel, dst, confidence=0.85, source_ref="dev-seed"
            )
        # The user's own layer: this is what makes recall personal.
        eg.link_user(
            None, user_email, "works_on", "Project Atlas",
            evidence="seeded demo context",
        )
        eg.link_user(
            None, user_email, "part_of", "Data Platform",
            evidence="seeded demo context",
        )
    except Exception:  # noqa: BLE001
        log.debug("enterprise knowledge seed skipped", exc_info=True)


_DEV_BOARD_TASKS = [
    (
        "Draft the Atlas risk register",
        "Pull the risks out of the Atlas brief and the billing runbook, "
        "then write a one-page register the steering group can read.",
        "todo", 1,
    ),
    (
        "Summarise Northwind renewal exposure",
        "Northwind renews in June. Summarise the renewal risk and the "
        "incident history behind it.",
        "backlog", 2,
    ),
    (
        "Q1 budget sign-off",
        "Finance needs approval by Friday. Review and respond.",
        "in_progress", 1,
    ),
]


def _seed_board(wh: Any, user_email: str) -> None:
    """Seat a few demo tasks on the board so it is not empty on first run."""
    try:
        from agent import taskboard

        def _sql(stmt: str, params: dict | None = None):
            res = wh.execute(stmt, params or {})
            as_dicts = getattr(res, "as_dicts", None)
            return as_dicts() if callable(as_dicts) else list(res or [])

        class _P:
            email = user_email
            is_admin = False

        for title, desc, status, priority in _DEV_BOARD_TASKS:
            taskboard.create_task(
                _sql, _P(), title=title, description=desc,
                status=status, priority=priority,
            )
    except Exception:  # noqa: BLE001
        log.debug("board seed skipped", exc_info=True)


# ---------------------------------------------------------------------------
# Guidance: the best-practice library and a few standing orders.
#
# These are the two surfaces that make Alex behave like *this* company's
# assistant rather than a generic one, so the dev environment ships with
# enough of both to exercise every path: a mandatory capability-bound
# practice, a subject-matched one, an unconditional house-style one, and
# standing orders that are runnable on demand.
# ---------------------------------------------------------------------------

_DEV_PRACTICES = [
    {
        "title": "Cite the source for any number you state",
        "guidance": (
            "Every figure — budget, headcount, date, contract value — must "
            "carry its source. Use the bracketed citation of the passage it "
            "came from. If a number is not in a retrieved source, say you "
            "do not have it rather than estimating."
        ),
        "authority": "mandatory",
        "rationale": (
            "Unsourced numbers get pasted into board material and become "
            "fact. This is the single most common way an assistant "
            "manufactures a liability."
        ),
        "anti_patterns": [
            "stating an approximate figure without saying it is approximate",
            "carrying a number forward from an earlier turn without re-citing it",
        ],
        "applies_to": ["budget", "revenue", "headcount", "contract", "forecast"],
    },
    {
        "title": "Never send external mail without explicit confirmation",
        "guidance": (
            "Draft the message, show it in full, and wait for the user to "
            "confirm before sending anything to an address outside the "
            "organisation. Name the recipient domain when you ask."
        ),
        "authority": "mandatory",
        "rationale": (
            "Outbound mail is irreversible and speaks in the user's name."
        ),
        # Capability-bound: this fires whenever mail is in play, however
        # the request was phrased.
        "capabilities": ["email.send", "email.draft"],
        "applies_to": ["email", "send", "reply", "customer", "partner"],
    },
    {
        "title": "Lead with the answer",
        "guidance": (
            "Open with the conclusion in one sentence, then the supporting "
            "detail. Do not narrate your process before answering."
        ),
        "authority": "recommended",
        "rationale": "Most readers stop after the first line.",
        # No applies_to and no capabilities: house style, always in force.
    },
    {
        "title": "Escalate renewal risk to the account owner",
        "guidance": (
            "When a customer's renewal risk is medium or higher, name the "
            "account owner and recommend a specific next step with a date. "
            "Do not simply report the risk level."
        ),
        "authority": "recommended",
        "applies_to": ["renewal", "churn", "customer", "contract", "account"],
        "rationale": "A risk with no owner and no date does not get worked.",
    },
    {
        "title": "Prefer the runbook over improvisation",
        "guidance": (
            "For any operational incident, check whether a runbook exists "
            "and follow it. If the runbook is wrong, say so explicitly "
            "rather than quietly doing something else."
        ),
        "authority": "advisory",
        "applies_to": ["incident", "outage", "runbook", "escalate", "on-call"],
    },
]

_DEV_STANDING_ORDERS = [
    {
        "title": "Monday morning review",
        "instruction": (
            "Summarise my inbox, list my open tasks by priority, and flag "
            "anything due in the next three days. Finish with the three "
            "things you think I should do first and why."
        ),
        "applies_to": ["review", "monday", "morning", "week"],
        "capabilities": ["email.read", "tasks.read", "knowledge.read"],
    },
    {
        "title": "Brief me before any customer conversation",
        "instruction": (
            "Pull everything we know about the customer: contract value, "
            "renewal date, open incidents, and who owns the account. "
            "Highlight anything that changed since I last asked."
        ),
        "applies_to": ["customer", "call", "meeting", "brief", "account"],
        "capabilities": ["knowledge.read", "memory.read"],
    },
]


def _seed_guidance(wh: Any, user_email: str) -> None:
    """Publish the demo practices and standing orders. Best-effort."""
    try:
        from agent import practices, standing_orders

        def _sql(stmt: str, params: dict | None = None):
            res = wh.execute(stmt, params or {})
            as_dicts = getattr(res, "as_dicts", None)
            return as_dicts() if callable(as_dicts) else list(res or [])

        for spec in _DEV_PRACTICES:
            practices.upsert(_sql, owner="governance@alex.dev", **spec)

        class _P:
            email = user_email
            is_admin = False
            # The seeded demo user holds the full capability set, so the
            # ceiling below is honoured rather than clamped away.
            capabilities = frozenset(
                {
                    "email.read", "email.draft", "email.send", "tasks.read",
                    "tasks.create", "tasks.update", "knowledge.read",
                    "memory.read", "memory.write",
                }
            )

        for spec in _DEV_STANDING_ORDERS:
            standing_orders.create(
                _sql,
                _P(),
                title=spec["title"],
                instruction=spec["instruction"],
                applies_to=spec["applies_to"],
                allowed_capabilities=spec["capabilities"],
                audience=[user_email],
            )
    except Exception:  # noqa: BLE001
        log.debug("guidance seed skipped", exc_info=True)


# A demo MCP server so the Tools surface is not empty on a fresh machine.
# `alex_sdk.mcp` serves Alex's own governed tools over stdio, which makes
# it both a realistic integration and a self-check: if this connects, the
# serve half and the consume half agree.
_DEV_MCP_SERVERS = [
    (
        "srv-alex-self",
        "alexself",
        "stdio",
        "python",
        ["-m", "alex_sdk.mcp"],
        "",
        "mcp.use",
        "enterprise",
        "trusted",
        True,
        30.0,
        "",
        "Alex's own governed tools, served over MCP.",
        "governance@alex.dev",
        None,
    ),
]


def _seed_mcp(wh: Any) -> None:
    """Register the demo MCP server. Best-effort."""
    try:
        for row in _DEV_MCP_SERVERS:
            wh.execute(
                "INSERT INTO media.control.mcp_servers "
                "(server_id, name, transport, command, args, url, capability, "
                " scope, trust, enabled, timeout_s, secret_ref, description, "
                " registered_by, registered_at) "
                "VALUES (:a, :b, :c, :d, :e, :f, :g, :h, :i, :j, :k, :l, :m, :n, :o)",
                dict(zip("abcdefghijklmno", row, strict=True)),
            )
    except Exception:  # noqa: BLE001
        log.debug("mcp seed skipped", exc_info=True)
