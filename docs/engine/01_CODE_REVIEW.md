# Alex Workspace — Code Review & Gap Analysis

Review of the existing project ("Alex — AI-Powered Workspace", Flask + React + SQLite + local LLMs) against what an enterprise media deployment on Databricks actually requires.

## What's good (keep it)

- **Clear product concept.** Three-panel minimalist UX (priority email / task board / AI panel) with AI woven into workflows rather than bolted on. This survives the migration intact.
- **Provider abstraction** (`base.py`, `ollama.py`, `lmstudio.py`, `openai_provider.py`) — the right instinct. We replace the implementations with a single Databricks Model Serving provider, but the interface pattern stays.
- **Blueprint-per-domain backend layout** (tasks, emails, files, team, ai) — clean, portable.
- **Document parsers** (pdf/word/excel/pptx) — reusable as ingestion UDFs in Databricks pipelines.
- **Config class with env overrides, error module, prompt templates** — decent hygiene for a prototype.

## Critical problems

1. **Hardcoded secret key.** `main.py` line 16: `SECRET_KEY = 'asdf#FGSgvasgf$5$WGT'` committed to source. Also `os.urandom(24)` fallback in `config.py` means sessions/JWTs invalidate on every restart and break under multiple workers. → Databricks Secrets / SP OAuth.
2. **SQLite as the system of record.** Single-writer, file-on-disk, no concurrency, no governance, no lineage, no backup story. Unusable for an enterprise and unusable for AI: the agent can't safely query it and analytics can't run on it. → Unity Catalog + Delta (Lakebase/Postgres for OLTP if needed).
3. **Two competing entrypoints.** `main.py` (hardcoded config, inline seeding, `db.create_all()` at import) vs `run.py`/`config.py` (proper factory). Seed data lives in application code. → one factory, migrations (Alembic), seeds in a job.
4. **No real authentication model.** `auth.py` rolls its own JWT users; `optional_auth.py` lets endpoints silently run unauthenticated. Demo scripts contain plaintext passwords. For an enterprise: SSO (Entra/Okta) via Databricks Apps, identity passed through via on-behalf-of tokens — no password table at all.
5. **The "agent" isn't an agent.** `ai.py`/`ai_service.py` are single-shot prompt calls with an in-memory MD5 cache. No tools, no permission boundary, no audit trail, no memory beyond a naive `memory.py`. It cannot take actions, and if it could, nothing would stop it from taking any action.
6. **In-memory everything**: AI cache, rate-limit storage (`memory://`), agent "memory". All lost on restart, all broken with >1 replica.
7. **CSP allows `unsafe-inline` + `unsafe-eval` and `connect-src https:`** (any origin). XSS protection is decorative.
8. **No observability.** File logging only, no tracing of AI calls, no cost tracking, no eval loop. You can't answer "what did the agent do and why" — which is the first question legal/compliance asks at a media company.
9. **Email integration is fictional** (local IMAP stub) and file storage is local disk — both non-starters for a multi-user enterprise.
10. **Test files, demos, reviews, and zips all live flat in one directory** including nested copies of the whole backend (`alex_backend.zip`, `alex-workspace-v2-final.zip`...). No package structure, no CI.

## What "doesn't make sense" for the target use case

- Local LLMs (phi3 on Ollama) for enterprise media work — quality, throughput, and governance all fail. Model Serving with hosted foundation models (Claude, Llama) + AI Gateway solves all three.
- Client-side polling for "team online" presence rather than events.
- AI summarization results not persisted — recomputed per view, cost with no memory.
- `DocumentAnalyzer` re-uploads and re-parses files per analysis instead of indexing once (→ Vector Search).

## What the Databricks rebuild gives you

| Prototype | Rebuilt |
|---|---|
| SQLite file | Unity Catalog + Delta tables, ACLs, lineage, time travel |
| Ollama/phi3 | Model Serving endpoints (Claude/Llama) behind AI Gateway w/ rate limits, PII guardrails, usage tracking |
| Prompt-call "agent" | Mosaic AI Agent Framework agent with **Unity Catalog function tools** — every tool is a governed object with explicit GRANTs |
| No identity | Databricks Apps SSO + service principal with least-privilege scopes; on-behalf-of-user auth for user-scoped data |
| No audit | MLflow 3 tracing on every agent turn + UC audit logs + inference tables |
| Local files | UC Volumes + Vector Search index over the media library |
| No compute story | Dedicated serverless compute for the app; the agent runs in Model Serving's isolated harness (its own container/VM per endpoint) with an egress-controlled environment |

See `02_ARCHITECTURE.md` for the target design and the `agent/`, `governance/`, `pipelines/` directories for the implementation.
