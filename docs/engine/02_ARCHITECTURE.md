# Alex 2.0 — Databricks-Native AI Workspace for a Media Enterprise

## 1. Overview

Alex becomes a **Databricks App** (the React/Flask workspace UI) talking to a **governed agent** deployed on **Mosaic AI Model Serving** via the **Agent Framework**. All data lives in **Unity Catalog**; every capability the agent has is a **Unity Catalog function tool** with explicit permissions; every agent turn is traced with **MLflow** and logged to inference tables.

```
┌─────────────────────────────── Databricks Workspace ───────────────────────────────┐
│                                                                                    │
│  Databricks App (serverless compute, SSO)                                          │
│  ┌──────────────┐   ┌──────────────────┐                                           │
│  │ React UI     │──▶│ FastAPI backend  │── on-behalf-of user token ──┐             │
│  └──────────────┘   └──────────────────┘                             ▼             │
│                                                     ┌──────────────────────────┐   │
│   AI Gateway (rate limits, PII guardrail,           │  Agent Serving Endpoint  │   │
│   usage & payload logging)  ◀───────────────────────│  "alex-agent"            │   │
│                                                     │  isolated container/env  │   │
│                                                     │  scale-to-zero, GPU-free │   │
│                                                     └───────────┬──────────────┘   │
│                                                                 │ UC tool calls    │
│              ┌───────────────┬──────────────────┬───────────────┼──────────────┐   │
│              ▼               ▼                  ▼               ▼              ▼   │
│   media.alex.create_task  media.alex.search_ SQL warehouse   Vector Search  Genie  │
│   media.alex.send_email   media_library      (read-scoped)   media_library  spaces │
│   (UC functions, each individually GRANTed to the agent's service principal)       │
│                                                                                    │
│   Unity Catalog: catalog `media` ─ schemas: alex (app), content, audience, ads     │
│   Delta tables · UC Volumes (files) · lineage · audit logs · row/column masks      │
│                                                                                    │
│   Lakeflow jobs: email ingest, content indexing, embeddings refresh, evals         │
└────────────────────────────────────────────────────────────────────────────────────┘
```

## 2. The agent harness (the core of your ask)

**"Harnessed with Databricks, with approved tooling and permissioning, its own VM if possible":**

- **Its own compute/VM.** Deploying via `databricks.agents.deploy()` gives the agent a dedicated Model Serving endpoint: an isolated, containerized serving environment (its own provisioned compute, not shared with the app), autoscaling with scale-to-zero. The app never runs model code; it calls the endpoint over HTTPS. If you need heavier isolation for code-execution tools, those run as **serverless jobs** the agent can only *submit*, never shell into.
- **Approved tooling.** The agent's tools are **Unity Catalog functions** (`media.alex.*`). A tool that isn't registered and granted simply doesn't exist for the agent. Adding a capability = a code-reviewed SQL/Python function + a GRANT — an auditable change, not a code push.
- **Permissioning.** The endpoint runs as a **service principal** with least privilege: `EXECUTE` on its tool functions, `SELECT` on the specific schemas it needs, `READ VOLUME` on the media library, nothing else. User-scoped actions (send email as me, see my tasks) use **on-behalf-of-user auth** from the Databricks App, so the agent can never see data the signed-in user can't.
- **Guardrails.** The endpoint sits behind **AI Gateway**: per-user rate limits, PII detection on requests/responses, invalid-keyword blocking, and full payload logging to an inference table.
- **Audit.** MLflow tracing records every tool call with inputs/outputs; UC audit logs record every data access. "What did the agent do on March 3rd" is one query.

## 3. Data model (Unity Catalog)

Catalog `media`:

- **`alex`** (app schema): `users` (mirror of SSO identities, no passwords), `tasks`, `task_comments`, `task_shares`, `emails`, `email_summaries`, `files`, `agent_actions` (append-only action ledger), `agent_memory` (per-user long-term memory, Delta not RAM).
- **`content`**: `assets` (video/audio/article metadata), `transcripts`, `rights` (licensing windows, territories), `captions`. Vector Search index `content.media_library_index` over transcripts + descriptions.
- **`audience`**: engagement/analytics tables — exposed to the agent **only** through a Genie space and read-only SQL tool, with column masks on anything user-identifiable.
- **`ads`**: campaigns, flights, pacing — for the sales-side features.

OLTP-ish app writes (task updates, comments) go through Delta with liquid clustering; if sub-100ms transactional writes matter, front with Lakebase (managed Postgres) synced to Delta.

## 4. Media-enterprise features (new)

1. **Content librarian.** "Find every clip where the CEO discusses the streaming deal" → Vector Search over transcripts, returns assets with timecodes, rights status inline (blocks suggesting expired-license footage).
2. **Rights & clearance checks as a tool.** `check_rights(asset_id, territory, window)` — the agent must call it before recommending reuse; the function encodes legal rules, not the prompt.
3. **Editorial packaging.** From a task like "package the earnings story", the agent drafts headline variants, pulls related assets, generates social cuts list — each step a traced tool call, output lands as task attachments.
4. **Audience insights via Genie.** Natural-language analytics ("how did last night's episode perform vs. season average in 18–34") delegated to a Genie space — governed SQL, no hallucinated numbers.
5. **Priority email, done properly.** Lakeflow job ingests mail (Graph/Gmail API) into `alex.emails`; batch `ai_query()` scores urgency and writes persistent summaries; UI reads precomputed results — cheap, instant, consistent.
6. **Broadcast-day briefing.** Scheduled agent run compiles overnight performance, breaking tasks, and flagged emails into a morning brief per user.
7. **Compliance mode.** All generated public-facing copy is watermarked in `agent_actions` with model, prompt version, and reviewer; nothing ships without a human approval task.

## 5. What was fixed from the prototype

- One app factory, secrets from Databricks Secrets, migrations not `create_all()`, seeds as a job.
- Auth deleted in favor of platform SSO; `optional_auth.py` gone — every request has an identity.
- AI cache/memory/rate-limits moved from process RAM to Delta/Gateway.
- CSP tightened (no `unsafe-eval`, explicit connect-src), security headers kept.
- Document parsers repurposed as ingestion UDFs feeding Vector Search — parse once, retrieve forever.
- Evals: an MLflow evaluation job runs a golden set of media Q&A + tool-use scenarios on every agent version before `deploy()` promotes it.

## 6. Deployment

Everything ships as a **Databricks Asset Bundle** (`databricks.yml`): the app, the agent model + endpoint, UC objects, pipelines, and the eval job — `databricks bundle deploy -t prod`. Dev/stage/prod targets map to separate schemas and service principals.
