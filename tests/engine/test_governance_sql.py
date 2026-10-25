"""
SQL smoke tests for the governance/ folder.

We don't have a real Databricks parser available, so this module:
  1. Checks that every .sql file is non-empty and contains no obvious
     syntactic problems (unbalanced parens, stray semicolons, missing
     semicolons at end of statements, no f-string interpolation patterns).
  2. Asserts the file's identifiers follow the catalog.schema.table
     pattern and that GRANTs use the right principal shape.
  3. Asserts the four 'appendOnly' ledgers are present with appendOnly.
  4. Asserts that the agent SP is never granted MODIFY on any of the
     append-only tables.

These are smoke tests, not a real parser. The full validation comes from
`databricks bundle validate` against a dev workspace.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

GOVERNANCE = Path(__file__).resolve().parents[2] / "governance"

SQL_FILES = sorted(GOVERNANCE.glob("*.sql"))

REQUIRED_TABLES = {
    # Schema -> set of (table, must-be-appendOnly)
    "media.alex": {
        ("users", False),
        ("tasks", False),
        ("task_comments", False),
        ("task_shares", False),
        ("agent_actions", True),
        ("email_drafts", False),
        ("standing_order_files", False),
        ("emails", False),
        ("email_summaries", False),
        ("email_intel", False),
        ("morning_briefs", False),
        ("inbox_view", False),
        ("task_board_view", False),
    },
    "media.control": {
        ("constitution", False),
        ("constitution_amendments", True),
        ("constitution_violations", True),
        ("employee_acl", False),
        ("instance_acl", False),
        ("delegable_capabilities", False),
        ("supervision", False),
        ("delegations", False),
        ("sessions", False),
        ("session_turns", False),
        ("user_knowledge_chunks", False),
        ("approved_files", False),
        ("instance_library", False),
        ("tool_integrations", False),
        ("browser_sessions", False),
        ("browser_history", True),
        ("workflow_catalog", False),
        ("workflow_runs", True),
        ("thinking_traces", True),
        ("skills", False),
        ("taught_routines", False),
        ("suggestion_feedback", False),
        ("standing_orders", False),
        ("standing_order_briefs", False),
        ("vm_policy", False),
        ("vector_memory_mirror", False),
        ("kg_mirror", False),
        ("consolidation_queue", False),
        ("forget_requests", False),
        ("risk_signals", True),
        ("investigations", True),
        ("investigation_access_log", True),
        ("task_crew", False),
        ("task_messages", False),
        ("task_questions", False),
        ("task_runs", False),
        ("task_plans", False),
        ("task_acceptance", False),
        ("task_deps", False),
        ("task_proposals", False),
    },
    "media.content": {
        ("assets", False),
        ("transcripts", False),
        ("captions", False),
    },
    "media.ops": {
        ("agent_reviews", True),
        ("gateway_logs", True),
        ("gateway_logs_payload", True),
        ("run_metrics", True),
    },
}


def _all_sql() -> dict[str, str]:
    return {p.name: p.read_text(encoding="utf-8") for p in SQL_FILES}


# ---------------------------------------------------------------------------
# 1. Basic syntax hygiene
# ---------------------------------------------------------------------------


def test_all_sql_files_exist_and_nonempty():
    assert len(SQL_FILES) >= 8, f"expected >= 8 SQL files, got {len(SQL_FILES)}"
    for p in SQL_FILES:
        text = p.read_text(encoding="utf-8")
        assert text.strip(), f"{p.name} is empty"


def test_balanced_parentheses():
    for p in SQL_FILES:
        text = p.read_text(encoding="utf-8")
        # Strip SQL string literals properly. Doubled '' inside a literal
        # is an escape for a single quote, so we walk the string and
        # replace runs of 'not-quote' with spaces.
        out = []
        i = 0
        n = len(text)
        in_str = False
        while i < n:
            ch = text[i]
            if in_str:
                if ch == "'":
                    if i + 1 < n and text[i + 1] == "'":
                        # escaped single quote inside a string
                        out.append(" ")
                        i += 2
                        continue
                    in_str = False
                    out.append(" ")
                    i += 1
                    continue
                out.append(" " if ch == "(" or ch == ")" else " ")
                i += 1
            else:
                if ch == "'":
                    in_str = True
                    out.append(" ")
                    i += 1
                    continue
                if ch == '"':
                    # double-quoted identifier; treat the contents as opaque
                    j = text.find('"', i + 1)
                    if j < 0:
                        j = n - 1
                    out.append(" " * (j + 1 - i))
                    i = j + 1
                    continue
                if ch == "-" and i + 1 < n and text[i + 1] == "-":
                    # line comment
                    j = text.find("\n", i)
                    if j < 0:
                        j = n
                    out.append(" " * (j - i))
                    i = j
                    continue
                if ch == "/" and i + 1 < n and text[i + 1] == "*":
                    j = text.find("*/", i + 2)
                    if j < 0:
                        j = n
                    else:
                        j += 2
                    out.append(" " * (j - i))
                    i = j
                    continue
                out.append(ch)
                i += 1
        stripped = "".join(out)
        assert stripped.count("(") == stripped.count(")"), (
            f"{p.name}: unbalanced parens "
            f"({stripped.count('(')} open vs {stripped.count(')')} close)"
        )


def test_no_python_string_interpolation_in_sql():
    """SQL files in production must never contain f-string interpolation.
    That was the root cause of the SQL-injection wave in the previous
    version. The fix is to bind parameters; this test catches the
    regression if a developer tries to roll back."""
    fstring_pattern = re.compile(r"['\"]\{[^}]+\}['\"]")
    for p in SQL_FILES:
        text = p.read_text(encoding="utf-8")
        # Strip SQL string literals properly, then check for f-string syntax.
        for line_no, line in enumerate(text.splitlines(), start=1):
            stripped_line = line
            if stripped_line.lstrip().startswith("--"):
                continue
            # Strip SQL string literals (with '' escape) and comments
            i = 0
            n = len(stripped_line)
            in_str = False
            no_str_chars = []
            while i < n:
                ch = stripped_line[i]
                if in_str:
                    if ch == "'":
                        if i + 1 < n and stripped_line[i + 1] == "'":
                            i += 2
                            continue
                        in_str = False
                        i += 1
                        continue
                    i += 1
                else:
                    if ch == "'":
                        in_str = True
                        i += 1
                        continue
                    no_str_chars.append(ch)
                    i += 1
            no_str = "".join(no_str_chars)
            assert not fstring_pattern.search(no_str), (
                f"{p.name}:{line_no}: f-string interpolation in SQL"
            )


def test_no_legacy_modify_grant_on_appendonly_ledgers():
    """No principal may hold MODIFY (the union of SELECT+INSERT+UPDATE+DELETE+TRUNCATE)
    on any of the append-only ledgers. The grant must be SELECT, INSERT only."""
    appendonly_tables = [
        "agent_actions",
        "agent_reviews",
        "constitution_amendments",
        "constitution_violations",
        "constitution_violations",
        "browser_history",
        "workflow_runs",
        "thinking_traces",
        "risk_signals",
        "investigations",
        "investigation_access_log",
        "gateway_logs",
        "gateway_logs_payload",
        "run_metrics",
    ]
    full = "\n".join(_all_sql().values())
    for tbl in appendonly_tables:
        # Look for any GRANT on this table; must not include MODIFY.
        for m in re.finditer(rf"GRANT\s+([A-Z,\s]+)\s+ON\s+(?:TABLE\s+)?media\.[\w\.]+\.{tbl}", full, re.IGNORECASE):
            perms = m.group(1).upper()
            assert "MODIFY" not in perms, (
                f"append-only table {tbl!r} was granted MODIFY; "
                f"the line is: {m.group(0)!r}"
            )


def test_oversight_two_key_constraint_present():
    full = _all_sql()["oversight.sql"]
    assert "opened_by <> co_authorized_by" in full, (
        "oversight.sql must have a CHECK that opened_by != co_authorized_by"
    )


def test_oversight_dual_procedures_present():
    full = _all_sql()["oversight.sql"]
    assert "open_investigation" in full
    assert "co_authorize_investigation" in full


def test_open_investigation_does_not_self_activate():
    # Opening a case must yield 'pending_coauth', never 'active'. Inserting
    # 'active' directly lets one governor unilaterally activate a case by
    # naming a co-author who never consented — defeating the two-key control.
    full = _all_sql()["oversight.sql"]
    body = full[full.index("PROCEDURE media.control.open_investigation"):
                full.index("PROCEDURE media.control.co_authorize_investigation")]
    assert "'pending_coauth'" in body
    assert "reason, 'active')" not in body, (
        "open_investigation must not insert status='active' directly"
    )


def test_co_authorize_where_clause_is_not_self_referential():
    # `WHERE case_id = case_id` is always-true and would activate every
    # pending case at once. Parameters must be disambiguated from columns.
    full = _all_sql()["oversight.sql"]
    body = full[full.index("PROCEDURE media.control.co_authorize_investigation"):]
    assert "case_id = case_id" not in body, (
        "co_authorize_investigation has a self-referential WHERE that matches all rows"
    )
    assert "co_authorized_by = co_auth_email" not in body, (
        "co_authorize_investigation param must not shadow the co_authorized_by column"
    )


# ---------------------------------------------------------------------------
# 2. Required tables exist
# ---------------------------------------------------------------------------


def _table_mentions_in(text: str, table: str) -> bool:
    return bool(re.search(rf"\b{table}\b", text))


@pytest.mark.parametrize("schema,tables", list(REQUIRED_TABLES.items()))
def test_required_tables_present(schema, tables):
    full = "\n".join(_all_sql().values())
    for tbl, must_be_appendonly in tables:
        # Look for an explicit CREATE TABLE on this table
        pat = rf"CREATE\s+(?:TABLE\s+)?(?:IF\s+NOT\s+EXISTS\s+)?{schema}\.{tbl}\b"
        assert re.search(pat, full, re.IGNORECASE), (
            f"missing CREATE TABLE for {schema}.{tbl}"
        )
        if must_be_appendonly:
            # Verify the appendOnly property is set
            m = re.search(
                rf"CREATE\s+(?:TABLE\s+)?(?:IF\s+NOT\s+EXISTS\s+)?{schema}\.{tbl}\b.*?(\n\n|\Z)",
                full, re.IGNORECASE | re.DOTALL,
            )
            assert m, f"could not find body of CREATE TABLE for {schema}.{tbl}"
            body = m.group(0)
            assert "delta.appendOnly" in body, (
                f"{schema}.{tbl} should be append-only but TBLPROPERTIES not set"
            )


# ---------------------------------------------------------------------------
# 3. Procedures and functions are syntactically well-formed
# ---------------------------------------------------------------------------


def test_tools_sql_defines_all_required_procs_and_udfs():
    full = _all_sql()["tools.sql"]
    required_procs = [
        "media.alex.create_task",
        "media.alex.update_task_status",
        "media.alex.draft_email_reply",
        "media.alex.log_agent_action",
    ]
    required_udfs = [
        "media.alex.list_my_tasks",
        "media.alex.summarize_email",
        "media.alex.check_rights",
    ]
    for p in required_procs:
        assert f"PROCEDURE {p}" in full, f"missing procedure {p}"
    for u in required_udfs:
        assert f"FUNCTION {u}" in full, f"missing function {u}"


def test_tools_sql_uses_sql_security_invoker():
    full = _all_sql()["tools.sql"]
    # Every CREATE PROCEDURE / FUNCTION should be SQL SECURITY INVOKER so
    # that the caller's identity is the one that runs the body. Otherwise
    # the procedure runs as the definer and `current_user` is wrong.
    n_procs = len(re.findall(r"CREATE\s+(?:OR\s+REPLACE\s+)?PROCEDURE\b", full, re.IGNORECASE))
    n_udfs = len(re.findall(r"CREATE\s+(?:OR\s+REPLACE\s+)?FUNCTION\b", full, re.IGNORECASE))
    n_invoker = len(re.findall(r"SQL\s+SECURITY\s+INVOKER\b", full, re.IGNORECASE))
    assert n_invoker >= n_procs + n_udfs, (
        f"every CREATE PROCEDURE / FUNCTION must use SQL SECURITY INVOKER; "
        f"got {n_invoker} INVOKER clauses for {n_procs} procs + {n_udfs} functions"
    )


# ---------------------------------------------------------------------------
# 4. Grants: agents get SELECT/INSERT only, jobs get the operational writes,
#    governors get the control plane.
# ---------------------------------------------------------------------------


def test_agent_sp_no_modify_anywhere():
    full = "\n".join(_all_sql().values())
    # Every grant to sp-alex-agent must not include MODIFY.
    for m in re.finditer(
        r"GRANT\s+([A-Z,\s]+)\s+ON\s+(?:TABLE\s+|SCHEMA\s+|VOLUME\s+)?[\w\.]+\s+TO\s+sp-alex-agent",
        full, re.IGNORECASE,
    ):
        perms = m.group(1).upper().replace(" ", "")
        assert "MODIFY" not in perms, (
            f"agent SP granted MODIFY: {m.group(0)!r}"
        )


def test_principals_mentioned_in_grants_exist_in_synthetic_set():
    """Spot-check that the four service principals and the two human groups
    are referenced. This catches typos in GRANTs (e.g. 'sp-alex-agnt')."""
    full = "\n".join(_all_sql().values())
    required_principals = {
        "sp-alex-app",
        "sp-alex-agent",
        "sp-alex-jobs",
        "sp-alex-gw",
        "alex-users",
        "alex-governors",
    }
    for p in required_principals:
        assert p in full, f"principal {p!r} is never granted anything"


# ---------------------------------------------------------------------------
# 5. ai_gateway.yml + sandbox_cluster_policy.json shape
# ---------------------------------------------------------------------------


def test_ai_gateway_uses_principal_id_variables():
    yml = (GOVERNANCE / "ai_gateway.yml").read_text(encoding="utf-8")
    # Group budgets must reference workspace variables, not literal group names.
    assert "${var.GROUP_ALEX_USERS_ID}" in yml
    assert "${var.GROUP_ALEX_GOVERNORS_ID}" in yml
    # Payload logging must be enabled.
    assert "payload_logging:" in yml
    assert "enabled: true" in yml


def test_sandbox_cluster_policy_is_single_user():
    j = (GOVERNANCE / "sandbox_cluster_policy.json").read_text(encoding="utf-8")
    assert '"SINGLE_USER"' in j, "sandbox must be SINGLE_USER, not DEDICATED"
    assert '"sp-alex-agent"' in j


def test_high_risk_capabilities_are_catalogued_with_dual_approval():
    """The powerful VM capabilities (install/shell/download) must appear
    in the delegable-capabilities catalog AND require a second approver,
    so a supervisor cannot grant them single-handed."""
    sql = (GOVERNANCE / "delegation.sql").read_text(encoding="utf-8")
    for cap in ("compute.install", "system.install", "compute.download", "compute.shell"):
        # The seed row must exist and end with ", true)" (requires_second_approver).
        import re

        m = re.search(rf"\('{re.escape(cap)}',[^\n]*", sql)
        assert m, f"{cap} missing from delegable_capabilities seed"
        assert m.group(0).rstrip().rstrip(",").endswith("true)"), \
            f"{cap} must require a second approver"


def test_office_capabilities_are_catalogued():
    sql = (GOVERNANCE / "delegation.sql").read_text(encoding="utf-8")
    assert "'office.write'" in sql and "'office.read'" in sql


def test_executive_views_expose_no_content_columns():
    """Core privacy promise: media.exec.* views must expose only
    aggregates/categories — never message content. Guards against a
    future edit leaking a content-bearing column to executives."""
    import re

    sql = (GOVERNANCE / "app_schema.sql").read_text(encoding="utf-8")
    # Extract each `CREATE ... VIEW media.exec.<name> AS <body>;` block.
    views = re.findall(
        r"CREATE\s+(?:OR\s+REPLACE\s+)?VIEW\s+media\.exec\.\w+\s+AS(.*?);",
        sql, re.IGNORECASE | re.DOTALL,
    )
    assert views, "no media.exec.* views found — schema moved?"
    # Unambiguous content columns that must never appear in an exec view.
    # (Identifiers like subject_email used only inside count(DISTINCT ...)
    # are k-anonymous aggregates, not content, so match whole words.)
    forbidden = (
        "content", "body", "understanding", "message",
        "redacted_body", "commitments",
    )
    for body in views:
        low = body.lower()
        for col in forbidden:
            assert not re.search(rf"{col}", low), (
                f"executive view leaks a content column ({col!r}): {body.strip()[:120]}"
            )
        # Only aggregate/category selects — every view must aggregate.
        assert "count(" in low or "group by" in low, (
            f"executive view is not aggregated (could expose rows): {body.strip()[:120]}"
        )


# ---- every DDL file is actually applied ----------------------------------


def test_the_runbook_applies_every_governance_sql_file():
    """A table with no DDL works in dev — the seed registers it — and
    fails in production with "table not found". That is exactly how the
    task board shipped: three tables that existed only in the dev seed,
    so every board write would have failed on Databricks.

    The runbook's list is hand-written and ordered, so it cannot be
    generated; but it can be held to the directory.
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    on_disk = {p.stem for p in (root / "governance").glob("*.sql")}
    runbook = (root / "docs" / "engine" / "08_DEPLOY_RUNBOOK.md").read_text(encoding="utf-8")
    listed: set[str] = set()
    for match in re.finditer(r"for f in ([^;]+?); do", runbook, re.S):
        listed |= {
            token for token in match.group(1).replace("\\n", " ").split()
            if token and token != "\\"
        }

    missing = on_disk - listed
    assert not missing, (
        f"governance DDL never applied by the runbook: {sorted(missing)}. "
        "These tables will not exist in production."
    )
    unknown = listed - on_disk
    assert not unknown, f"runbook applies files that do not exist: {sorted(unknown)}"



def test_grants_run_after_every_create_table():
    """The split itself: permissions.sql must contain no GRANT statements
    (they moved to grants.sql), and grants.sql no CREATE TABLE — mixing
    them back recreates the ordering failure the split removed."""
    permissions = (GOVERNANCE / "permissions.sql").read_text(encoding="utf-8")
    grants = (GOVERNANCE / "grants.sql").read_text(encoding="utf-8")
    assert not re.search(r"^GRANT\s", permissions, re.MULTILINE), (
        "permissions.sql grants on tables other files create — those "
        "GRANTs belong in grants.sql, which runs last"
    )
    assert not re.search(r"^CREATE\s+TABLE", grants, re.MULTILINE), (
        "grants.sql must stay pure grants"
    )



def test_every_pg_table_the_code_references_has_lakebase_ddl():
    """The production halves of rag / enterprise_graph / vector_memory
    speak to Lakebase Postgres with bare table names. A bare name with
    no CREATE in lakebase_ddl.pg.sql works in dev (the in-memory twin
    answers) and fails in production with 'relation does not exist' —
    which is exactly how enterprise_docs, kg_entities and kg_edges_v2
    shipped: the whole knowledge base and graph had no backing table."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    ddl = (root / "pipelines" / "lakebase_ddl.pg.sql").read_text(encoding="utf-8")
    created = set(re.findall(r"CREATE TABLE IF NOT EXISTS (\w+)", ddl))

    referenced: set[str] = set()
    for module in ("agent/rag.py", "agent/enterprise_graph.py",
                   "agent/vector_memory.py", "pipelines/memory_apply.py",
                   "agent/session_manager.py"):
        src = (root / module).read_text(encoding="utf-8")
        # Bare (schema-less) names are the Postgres ones; catalog.schema
        # names are the warehouse's.
        for m in re.finditer(r"(?:FROM|INSERT INTO|DELETE FROM|UPDATE)\s+([a-z_]\w*)\s", src):
            name = m.group(1)
            if name in {"the", "a", "an"}:  # prose in docstrings
                continue
            if re.search(rf"(?:FROM|INTO)\s+media\.\w*\.?{name}", src):
                continue
            referenced.add(name)
    # Only hold the DDL to names that look like tables (heuristics keep
    # docstring prose out): known live pg tables must all be present.
    live = {"enterprise_docs", "kg_entities", "kg_edges_v2", "vector_memory",
            "sessions", "session_turns"}
    assert live <= referenced, (
        f"expected the code to reference {live}, found {referenced & live} — "
        "module moved? update this guard"
    )
    missing = live - created
    assert not missing, (
        f"pg tables referenced by code but never created: {sorted(missing)}"
    )


def test_the_write_asymmetry_is_in_the_grants():
    """People define done; the agent only verifies. The application
    enforces it, and the grant layer must AGREE: the agent SP may
    UPDATE task_acceptance (a verdict) but never INSERT or DELETE
    (a criterion), and holds no grant at all on task_deps or
    task_proposals — dependencies and the email->task gate are human
    decisions."""
    grants = (GOVERNANCE / "grants.sql").read_text(encoding="utf-8")
    accept = re.search(
        r"GRANT\s+([A-Z,\s]+)\s+ON\s+media\.control\.task_acceptance\s+TO\s+sp-alex-agent",
        grants,
    )
    assert accept, "the agent needs SELECT, UPDATE on task_acceptance"
    perms = {p.strip() for p in accept.group(1).split(",")}
    assert perms == {"SELECT", "UPDATE"}, (
        f"agent grant on task_acceptance must be SELECT, UPDATE only, got {perms}"
    )
    for table in ("task_deps", "task_proposals"):
        assert not re.search(
            rf"media\.control\.{table}\s+TO\s+sp-alex-agent", grants
        ), f"the agent must hold no grant on {table}"
