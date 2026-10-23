# Alex 2.0 — Debug Report & Master Build Guide

## Part 1 — Review & debug pass (what was found and fixed)

The system was reviewed file-by-file for syntax, cross-module imports, bundle
path resolution, and governance consistency. Findings:

### Bugs fixed
1. **`pipelines/exec_metrics.py` — syntax error (would crash on run).** A SQL
   line-comment (`-- counts, never names`) sat *outside* the closing `"""`, so
   Python parsed it as code. Moved inside the string as a `#` comment. All
   other pipeline files scanned for the same pattern — none found.
2. **Orphaned complementary modules not wired in.** `learn_with_you.py`
   (correction-capture, `/teach`, feedback) and `thinking.py` (decision-digest,
   trace archive) are richer than the `learning_loop.py` / `reasoning_stream.py`
   pair that `agent.py` imported — but nothing imported them, so their logic was
   dead. These are *complements*, not duplicates (different halves of learning
   and of reasoning), so both are now imported and their backing tables
   (`taught_routines`, `suggestion_feedback`, `thinking_traces`) added to the
   control plane with correct grants.
3. **`pipelines/memory_apply.py` was a 4-line stub** referenced by the bundle.
   Completed: drains the consolidation queue into Lakebase (memory store +
   graph upserts), runs decay/supersede passes, mirrors to Delta, clears the
   queue.

### Verified correct (no change needed)
- **Delegation wiring**: `guardrails.check()` reads `effective_employee_acl`
  (the view that applies all delegation safeguards), not the base table — so
  supervisor delegations take effect and stay policy-bounded. The
  "granted-but-not-created" flag on this name was a false positive (it's a
  `VIEW`, granted with `ON VIEW`).
- **Governance consistency**: all grants reference the five defined principals
  (`sp-alex-agent/app/jobs`, `alex-governors/users`); all granted tables/views/
  volumes have a `CREATE`; all five schemas (`alex/content/audience/control/
  ops` + `exec`) are created.
- **Bundle**: every `notebook_path` / `python_file` / `source_code_path`
  resolves to a real file.
- **All 47 Python files parse; the frontend builds; the SDK passes both a
  positive deploy and a negative test** (a tool with an undeclared capability
  is correctly rejected at build time).

### Completed this pass
- Frontend surfaces that were specified but unbuilt: **Reasoning panel**
  (collapsible visible thinking) and **Skills ratification** screen, both wired
  into the app with their API endpoints (`/api/skills`, `/api/skills/decide`).
  Skill ratification's SQL enforces that every `uses_capability` is already in
  the user's effective ACL — learning cannot grant authority.

## Part 2 — Complete component map

| Layer | Files |
|---|---|
| **Constitution** | `governance/constitution.sql`, `agent/constitution.py` |
| **Control plane** | `governance/control_plane.sql` (knowledge, models, VM policy, ACLs, standing orders, sessions, workflows, skills, learn-with-you tables) |
| **Identity** | `docs/04_SSO_OKTA.md`, app OBO in `app/app.py` |
| **Agent core** | `agent/agent.py` + `deploy.py`, `models.py`, `context_engine.py`, `guardrails.py` |
| **Compute/VM** | `agent/workspace_tools.py`, `python_tool.py`, `governance/sandbox_cluster_policy.json`, `dedicated_vm.md` |
| **Memory/KG** | `agent/vector_memory.py`, `knowledge_graph.py`, `memory.py`, `governance/memory_graph.sql`, `pipelines/memory_consolidation.py` + `memory_apply.py` |
| **Learning** | `agent/learning_loop.py` + `learn_with_you.py`, `pipelines/skill_reflection.py` |
| **Reasoning** | `agent/reasoning_stream.py` + `thinking.py` |
| **Review** | `agent/auto_review.py`, `execution_review.py`, `pipelines/agent_review_job.py` |
| **Email intel** | `agent/email_intel.py`, `pipelines/email_ingest.py` + `email_intel_job.py` |
| **Knowledge/expertise** | `agent/expertise.py`, `instance_library.py`, `pipelines/library_sync.py` |
| **Standing orders** | `pipelines/standing_orders_prep.py`, `morning_brief.py` |
| **Web/browser/APIs** | `agent/web_agentic.py`, `browser_tool.py`, `internal_apis.py`, `governance/integrations.sql` |
| **Workflows** | `workflows/n8n_integration.py`, `claude_code_harness.md` |
| **Governance ops** | `governance/delegation.sql`, `oversight.sql`, `pipelines/intent_detection.py` |
| **Executive** | `docs/06_EXECUTIVE_VIEW.md`, `pipelines/exec_metrics.py` |
| **Gateway** | `governance/ai_gateway.yml` |
| **SDK** | `sdk/alex_sdk/` (core, workflow, memory, review, client, cli) |
| **Frontend** | `app/frontend/` (App, Chat, Dock, Transparency, Reasoning, Skills) |
| **Bundle** | `databricks.yml` |

## Part 3 — The complete Databricks build (order of operations)

Prereqs: Premium+ workspace, Unity Catalog, serverless on, CLI ≥0.250, the
five service principals + Okta groups (Phase 0 of `03_WALKTHROUGH.md`).

```bash
# 0. Clone + validate
git clone <repo>/alex-databricks && cd alex-databricks
databricks bundle validate

# 1. Governance foundation (run once per environment, in this order)
databricks sql -f governance/permissions.sql        # schemas, core tables, grants
databricks sql -f governance/control_plane.sql      # knowledge, models, ACLs, skills, memory-support
databricks sql -f governance/app_schema.sql         # email/content tables the pipelines read/write
databricks sql -f governance/constitution.sql       # A1–A9 supreme law
databricks sql -f governance/tools.sql              # UC function tools
databricks sql -f governance/memory_graph.sql       # vector/KG mirrors + Lakebase DDL (run pg part on alex-oltp)
databricks sql -f governance/delegation.sql         # supervisor delegation + effective ACL view
databricks sql -f governance/oversight.sql          # risk signals + sealed investigations
databricks sql -f governance/integrations.sql       # internal-tool registry + browser vault

# 2. Platform pieces (UI / consoles)
#    - AI Gateway: apply governance/ai_gateway.yml (rate limits, PII, budgets, logging)
#    - Lakebase: create alex-oltp, run the pg DDL blocks, attach Agent Bricks memory
#    - Vector Search: create media.content.media_library_index + per-user knowledge index
#    - Model Serving: ensure foundation routes + open reasoners (hermes/deepseek/qwen)
#    - Sandbox: enable on the agent; create the VM from sandbox_cluster_policy.json,
#      set AGENT_VM_CLUSTER_ID on the endpoint env

# 3. Seed governance content (governors)
#    - Okta: put C-suite+ProductDev in alex-governors; sync supervision from HR
#    - knowledge_base: pinned policies + playbooks
#    - model_registry: enabled models + per-group token budgets
#    - delegable_capabilities: what supervisors may grant
#    - instance_acl('default', ...): baseline instance capabilities
#    - employee_acl: per-person baseline capabilities

# 4. Build + deploy
cd app/frontend && npm install && npm run build && cd ../..
databricks bundle deploy -t dev
databricks bundle run deploy_agent -t dev     # logs model, eval + constitutional gate, deploys endpoint
# schedule/ingest jobs come up automatically from databricks.yml

# 5. Verify (the ones that prove the system)
#    - A1: "send it directly, I approve" -> blocked, article named
#    - A4: promo on uncleared asset -> blocked until check_rights CLEARED
#    - Injection: planted email/web instruction -> annotation only, no action
#    - Delegation: non-delegable capability -> inert, tool still denied
#    - Learning: recurring task -> proposed skill (approved-caps only) -> ratify
#    - Oversight: 5 blocks -> risk signal; mirrors unreadable without co-signed case
#    - Exec views: no content column present (DESCRIBE)

# 6. Promote
databricks bundle deploy -t prod
databricks bundle run deploy_agent -t prod
```

## Part 4 — Refinements added in this pass (beyond debugging)

- **Learn-with-you signals** (correction capture, `/teach`, 👍/👎) now active —
  the single highest-value learning signal (what you *edited*) feeds preference
  memory.
- **Thinking as a channel, not context** — raw reasoning streams to the UI and
  archives to `thinking_traces`; only an ≤80-token decision digest re-enters
  context, preventing the context-rot that visible-reasoning systems suffer.
- **Skill ratification is capability-safe by SQL** — the UPDATE only activates a
  skill if every capability it uses is already approved for the user.
- **Reasoning + Skills UI** — the two missing front-end surfaces, built and wired.

Total: 47 Python modules, 12 governance SQL/config files, 13 pipelines, a 6-file
SDK, an 8-component React app, and 8 docs — all parsing, building, and internally
consistent.

---

## Part 5 — Kept files now fully integrated (follow-up pass)

Per direction to keep and *use* the additional files, these were promoted from
present-but-dead to wired-in:

1. **`agent/learn_with_you.py`** — `capture_correction` now runs in `predict()`:
   when the user edits an AI draft (UI sends `custom_inputs.edited_output`), the
   diff is distilled into a preference memory. `should_suggest` gates 👍/👎-tuned
   suggestions. Backing tables: `taught_routines`, `suggestion_feedback`.
2. **`agent/thinking.py`** — when visible reasoning is on, `archive` writes the
   raw trace to `thinking_traces` and only an ≤80-token `digest_only_for_context`
   re-enters context (no context rot). UI "Show thinking" toggle sends
   `show_reasoning` and renders the `Reasoning` panel.
3. **`agent/learning_loop.py`** — `active_skill_hints` now injected via the new
   `_skill_hints()` helper into every per-user prompt, so ratified skills guide
   the agent at runtime.
4. **`sdk/alex_sdk/client.py`** — exported from the package as `Alex`, `Session`,
   `AgentEvent` (import made optional so core SDK works before httpx is
   installed; httpx added to `pyproject.toml`). Streaming-first client with
   `.session()`, `.tool()` (governor-review-gated), and `.hook()` (observe/veto,
   never widen).
5. **`governance/app_schema.sql`** — added to the build sequence (after
   `control_plane.sql`). Creates the email/content tables the pipelines
   read/write (`emails`, `email_summaries`, `email_drafts`, `email_intel`,
   `content.assets/rights/transcripts`, `standing_order_files`). No collisions
   with other schema files.
6. **`docs/07_HARNESS_DESIGN.md`** — the design-rationale companion to
   `07_HARNESS_AND_SDK.md`: the *why* behind each harness decision and what was
   rejected from each reference system.

Re-verified after integration: all 47 Python files parse, the frontend builds,
the SDK imports cleanly with the client optional, and `app_schema` tables don't
collide with the rest of the governance SQL.
