# Deploy runbook — verified state as of 2026-07-19

## Local (fully offline) — verified working

```powershell
# prerequisites: local Ollama running with at least one chat model pulled
cd <repo>\alex
$env:ALEX_ENV = "dev"; $env:OLLAMA_MODEL = "hcp-hermes:latest"
.venv\Scripts\python.exe -m uvicorn app.app:app --host 127.0.0.1 --port 8765
# open http://127.0.0.1:8765 — the UI self-bootstraps a dev token
```

- Dev LLM: local Ollama (`http://127.0.0.1:11434`); if `OLLAMA_MODEL` isn't
  pulled, the client falls back to the first pulled chat model (log warning).
- Verify: `.venv\Scripts\python.exe -m pytest -q` (329 pass), `ruff check .`
  clean, `cd app/frontend && npm test && npm run build` (23 pass).

## Databricks

The bundle validates clean against the workspace (`databricks bundle validate`,
CLI authed as you@your-org.com; `warehouse_id` resolves by lookup to
"Serverless Starter Warehouse").

```bash
cd alex-databricks
# 1. one-time governance DDL. Preferred: the bundle job applies every
# file, ordered, after `databricks bundle deploy`:
#   databricks bundle run apply_governance_ddl -t dev
# Manual equivalent (order matters: schemas first, table DDL in FK
# order, procedures once their tables exist, and grants LAST — a GRANT
# on a table a later file creates would fail). Every file in
# governance/ must appear here — a table with no DDL works in dev,
# where the seed registers it, and fails in production with "table not
# found". tests/test_governance_sql.py holds this list to the directory
# so an omission fails the build rather than the deploy.
for f in 00_preamble permissions app_schema control_plane taskboard \
         oversight delegation integrations memory_graph constitution \
         tools grants; do
  databricks sql -f governance/$f.sql; done
# 2. frontend build ships with the app (dist/ is sync-included)
cd app/frontend && npm ci && npm run build && cd ../..
# 3. deploy, then DDL (both sides), then the agent
databricks bundle deploy -t dev
databricks bundle run apply_governance_ddl -t dev   # all governance SQL, ordered, grants last
databricks bundle run apply_lakebase_ddl -t dev     # Delta memory mirrors
# Postgres side of Lakebase (pgvector): enterprise_docs, kg_entities,
# kg_edges_v2, vector_memory — a warehouse sql_task cannot speak
# Postgres DDL, so this one is psql (runbook header in the file):
#   psql ... -f pipelines/lakebase_ddl.pg.sql
databricks bundle run deploy_agent -t dev
```

The app container installs the ROOT `requirements.txt` (Databricks Apps
convention); the serving endpoint uses `agent/requirements.txt`; jobs get
theirs from the bundle's per-job `environments` spec. All three are pinned
to the same tested versions.

Deploy-time work the bundle cannot do for you (must exist in the workspace
before prod):
- Service principals `sp-alex-agent/app/jobs/bundle` and the Okta groups.
- Lakebase instance `alex-oltp` (then the psql DDL above); AI Gateway
  config (`governance/ai_gateway.yml`); sandbox cluster from
  `governance/sandbox_cluster_policy.json`. (There is no Vector Search
  index: retrieval runs on the `enterprise_docs` pgvector table.)
- Secrets scope `alex` (Graph API creds for email ingest; plus
  `n8n-api-key` — the enterprise n8n instance's scoped API key:
  `databricks secrets put-secret alex n8n-api-key`). Set the app's
  `N8N_BASE_URL` in app.yaml to the real internal n8n host (the
  separate AWS Kubernetes cluster); https is enforced in production.
- Production JWT: the app fails closed if `ALEX_JWT_DEV_SECRET` is the dev
  default; the platform-injected `X-Forwarded-Access-Token` is the only
  trusted identity source in production (client Bearer headers are ignored).

## Standing automation (the scheduler)

Standing orders with `trigger = schedule` now actually fire: the app's
dispatcher sweep (`ALEX_DISPATCH_INTERVAL_S`, default 120s) evaluates
every active scheduled order and fires the due ones as the order's
owner, through the same single-flight and permission checks as a
manual run. Schedule grammar (UTC): `hourly [@:MM]`, `daily HH:MM`,
`weekly DAY HH:MM`, `every N minutes|hours` (15-minute floor), or the
numeric cron subset `M H * * D`. An unparseable schedule never fires
(logged); missed windows collapse into one firing; at most
`ALEX_SCHEDULE_MAX_PER_SWEEP` (3) fire per sweep.

## Telemetry export

`ALEX_TELEMETRY` unset = off (default). `log` = one structured JSON
line per event on the `alex.telemetry` logger (Databricks Apps captures
app logs; any log pipeline can scrape it). `otlp` = OpenTelemetry spans
to `ALEX_OTEL_ENDPOINT` — requires `opentelemetry-sdk` +
`opentelemetry-exporter-otlp`, deliberately NOT in the pinned
requirements; missing packages degrade to log lines with one warning.
Events are attributes only (ids, counts, durations, token counts) —
never content. Emitted today: `task_run_turn`, `task_run_finished`.

## Tool policy (allow / ask / deny)

Operators grade tool risk with rows in `media.control.workspace_policy`:
`name = tool_policy.<tool-or-capability>`, `value = allow|ask|deny`
(most specific wins; invalid values are ignored). Defaults: the
dual-approval capabilities (`compute.install`, `system.install`,
`compute.shell`, `compute.download`) plus `email.send` and
`publish.post` are **ask**; `pip_install` is allow (its curated
library allowlist is the control); everything else allows. `ask` on an
unattended board run parks the run on a one-click approve/deny in the
task thread and FAILS CLOSED (deny, timeout `ALEX_APPROVE_WAIT_S`=180s,
and stop all refuse); in a supervised chat the action proceeds and is
announced on the stream.

## Cost engineering (server CPU + LLM)

App compute — Databricks Apps bills a fixed rate per instance size
while the app is deployed: **Medium = up to 2 vCPU / 6 GB at 0.5
DBU/hr (the default)**, **Large = up to 4 vCPU / 12 GB at 1 DBU/hr**.
Stay on Medium: this app is I/O-shaped (SSE pumps, warehouse calls,
model streams), and the things that could genuinely eat its 2 vCPUs
are bounded in code —

- `ALEX_MAX_CONCURRENT_RUNS` (default 3) caps simultaneous board runs
  per app instance; the route answers 429 and the task stays queued.
- Swarm children spend a slice of the parent's remaining budget, never
  fresh full budgets.
- `pip_install` only accepts the approved data/ML library set
  (`agent/python_tool.APPROVED_PACKAGES`; operators extend via
  `ALEX_EXTRA_PACKAGES`). Heavy frameworks (torch, transformers) are
  approved but belong on the sandbox cluster, not the app container.
- Instance size is set in the app's **Settings → Compute** (UI/API),
  not in app.yaml.

LLM — the default serving endpoint is **`databricks-glm-5-3-flash`**
(Databricks-hosted GLM 5.3 Flash, pay-per-token, ~1M context, agentic
tool use); the catalog's frontier pick is `databricks-glm-5-3`
(documented function calling + parallel tool calls), with Claude
routes available by explicit selection. AI Gateway carries the rate
limits and per-group token budgets (`governance/ai_gateway.yml`,
route now glm-5-3-flash); per-turn spend lands in
`media.ops.run_metrics`, and `media.exec.run_performance` reports
daily volume, avg/P95 turn latency, and token spend. Job pipelines
(email intel, content indexing, standing-order prep, skill
reflection) also run on glm-5-3-flash.

## HIPAA note

PII redaction (`pipelines/content_indexing._redact`, `agent/browser_os`) is
regex-based — adequate for demos, NOT a defensible de-identification method
for PHI. Before ingesting regulated data, swap in a real detector (e.g.
Presidio / Comprehend Medical) and fail closed on detector errors.
