-- ===========================================================================
-- permissions.sql â€” identity, schema, and least-privilege grants.
--
-- The four service principals:
--   sp-alex-app    â€” runs the Databricks App (FastAPI + React)
--   sp-alex-agent  â€” runs the agent endpoint (Model Serving)
--   sp-alex-jobs   â€” runs Lakeflow jobs (pipelines/*)
--   sp-alex-gw     â€” writes to media.ops.gateway_logs
--
-- The two human groups:
--   alex-users     â€” employees; can read their own rows in media.alex
--   alex-governors â€” control-plane governors (constitution, ACLs, etc.)
--
-- Properties enforced here:
--   * Append-only ledgers (`agent_actions`, `agent_reviews`,
--     `constitution_violations`, `risk_signals`, `investigation_access_log`)
--     hold delta.appendOnly=true AND are granted SELECT, INSERT only â€”
--     UPDATE/DELETE are not possible from any principal we authorize.
--   * Cross-tenant isolation: there is no per-volume grant on
--     media.alex.workspaces. The app performs the user check; the volume
--     is granted only to sp-alex-app.
--   * The "control plane" (`media.control.*`) is writable by
--     sp-alex-jobs and alex-governors, never by the agent or the app
--     except through stored procedures in tools.sql.
-- ===========================================================================

-- Service principals exist in the workspace; we reference them by name.
-- Their actual provisioning is a deployment prerequisite.

-- ---------------------------------------------------------------------------
-- media.alex: per-employee operational state
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS media.alex.users (
  email          STRING        NOT NULL,
  display_name   STRING,
  sso_subject    STRING        NOT NULL,        -- the JWT 'sub' claim
  is_active      BOOLEAN       NOT NULL DEFAULT true,
  last_login_at  TIMESTAMP,
  created_at     TIMESTAMP     NOT NULL DEFAULT current_timestamp(),
  CONSTRAINT pk_users PRIMARY KEY (email)
)
CLUSTER BY (email);

CREATE TABLE IF NOT EXISTS media.alex.tasks (
  id              STRING        NOT NULL,
  owner_email     STRING        NOT NULL,
  assignee_email  STRING,
  title           STRING        NOT NULL,
  description     STRING,
  -- Status carries TWO vocabularies on purpose: the board lanes
  -- (backlog|todo|in_progress|review|done â€” agent/taskboard.py COLUMNS)
  -- are canonical, and the legacy tool statuses (open|blocked|cancelled)
  -- are still accepted because governance/tools.sql procedures and older
  -- rows write them. Reads normalise via taskboard.normalise_status
  -- (open->todo, blocked->in_progress, cancelled->done). The CHECK must
  -- accept the union â€” a lane the constraint rejects would make every
  -- board write fail in production while passing in dev, where CHECKs
  -- are not enforced.
  status          STRING        NOT NULL DEFAULT 'open',
  priority        INT           NOT NULL DEFAULT 3,        -- 1..5
  due_at          TIMESTAMP,
  source          STRING,                                 -- 'human'|'agent'|'standing_order'|'workflow'
  source_ref      STRING,                                 -- e.g. standing_order_id, workflow_run_id
  created_at      TIMESTAMP     NOT NULL DEFAULT current_timestamp(),
  updated_at      TIMESTAMP     NOT NULL DEFAULT current_timestamp(),
  closed_at       TIMESTAMP,
  CONSTRAINT pk_tasks PRIMARY KEY (id),
  CONSTRAINT fk_tasks_owner FOREIGN KEY (owner_email) REFERENCES media.alex.users(email),
  CONSTRAINT chk_tasks_status CHECK (status IN
    ('backlog','todo','open','in_progress','review','done','cancelled','blocked')),
  CONSTRAINT chk_tasks_priority CHECK (priority BETWEEN 1 AND 5)
)
CLUSTER BY (owner_email, status);

CREATE TABLE IF NOT EXISTS media.alex.task_comments (
  id              STRING        NOT NULL,
  task_id         STRING        NOT NULL,
  author_email    STRING        NOT NULL,
  body            STRING        NOT NULL,
  created_at      TIMESTAMP     NOT NULL DEFAULT current_timestamp(),
  CONSTRAINT pk_task_comments PRIMARY KEY (id),
  CONSTRAINT fk_task_comments_task FOREIGN KEY (task_id) REFERENCES media.alex.tasks(id),
  CONSTRAINT fk_task_comments_author FOREIGN KEY (author_email) REFERENCES media.alex.users(email)
)
CLUSTER BY (task_id);

CREATE TABLE IF NOT EXISTS media.alex.task_shares (
  task_id         STRING        NOT NULL,
  shared_with     STRING        NOT NULL,        -- email or 'GROUP:<name>'
  permission      STRING        NOT NULL DEFAULT 'read',  -- 'read'|'comment'|'edit'
  shared_by       STRING        NOT NULL,
  shared_at       TIMESTAMP     NOT NULL DEFAULT current_timestamp(),
  CONSTRAINT pk_task_shares PRIMARY KEY (task_id, shared_with),
  CONSTRAINT fk_task_shares_task FOREIGN KEY (task_id) REFERENCES media.alex.tasks(id)
);

-- Append-only action ledger. The agent writes every consequential action
-- here; nobody â€” not even the agent â€” can UPDATE/DELETE.
CREATE TABLE IF NOT EXISTS media.alex.agent_actions (
  id            STRING        NOT NULL,
  ts            TIMESTAMP     NOT NULL DEFAULT current_timestamp(),
  actor         STRING        NOT NULL,            -- who initiated the action (user or 'agent:<id>')
  session_id    STRING,
  action_type   STRING        NOT NULL,
  target_id     STRING,
  detail        STRING,                              -- free-form, bounded
  outcome       STRING        NOT NULL DEFAULT 'ok', -- 'ok'|'denied'|'error'|'revoked'|'started'
  tool_name     STRING,
  tool_args     STRING,
  -- Harness (agent/vm.py) columns. The VM's audit writer populates
  -- both column families in one row; older writers may leave these NULL.
  action_id     STRING,                              -- mirrors id
  tool          STRING,                              -- mirrors tool_name
  capability    STRING,
  user_email    STRING,                              -- mirrors actor
  instance_scope STRING,
  input_json    STRING,                              -- mirrors tool_args
  output_json   STRING,
  ok            INT,                                 -- 1 iff outcome = 'ok'
  CONSTRAINT pk_agent_actions PRIMARY KEY (id),
  CONSTRAINT chk_agent_actions_outcome CHECK (outcome IN ('ok','denied','error','revoked','started'))
)
CLUSTER BY (actor, ts)
TBLPROPERTIES (
  'delta.appendOnly' = 'true',
  'delta.enableChangeDataFeed' = 'true',
  'delta.deletedFileRetentionDuration' = 'interval 30 days'
);

-- ---------------------------------------------------------------------------
-- Append-only ops tables (auto-review, gateway payload log)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS media.ops.agent_reviews (
  id            STRING        NOT NULL,
  trace_id      STRING        NOT NULL,
  ts            TIMESTAMP     NOT NULL DEFAULT current_timestamp(),
  scorer        STRING        NOT NULL,
  score         DOUBLE,
  rationale     STRING,
  CONSTRAINT pk_agent_reviews PRIMARY KEY (id)
)
CLUSTER BY (trace_id, ts)
TBLPROPERTIES (
  'delta.appendOnly' = 'true',
  'delta.enableChangeDataFeed' = 'true'
);

CREATE TABLE IF NOT EXISTS media.ops.gateway_logs (
  id            STRING        NOT NULL,
  ts            TIMESTAMP     NOT NULL DEFAULT current_timestamp(),
  principal     STRING        NOT NULL,
  endpoint      STRING        NOT NULL,
  status_code   INT,
  latency_ms    INT,
  prompt_tokens INT,
  completion_tokens INT,
  CONSTRAINT pk_gateway_logs PRIMARY KEY (id)
)
CLUSTER BY (ts)
TBLPROPERTIES ('delta.appendOnly' = 'true');

-- Per-working-turn run accounting, written by the app as each task-run
-- turn ends: passes, tool calls, wall seconds (with human-wait credited
-- out separately), and token spend when the provider reports it. The
-- ledger behind media.exec.run_performance (avg / P95 latency, spend) —
-- before it, budget accounting died with the SSE stream.
CREATE TABLE IF NOT EXISTS media.ops.run_metrics (
  run_id            STRING      NOT NULL,
  task_id           STRING      NOT NULL,
  turn              INT         NOT NULL,
  llm_turns         INT         NOT NULL DEFAULT 0,
  tool_calls        INT         NOT NULL DEFAULT 0,
  elapsed_s         DOUBLE      NOT NULL DEFAULT 0,
  waited_s          DOUBLE      NOT NULL DEFAULT 0,   -- parked on a human (ask_user)
  prompt_tokens     BIGINT      NOT NULL DEFAULT 0,
  completion_tokens BIGINT      NOT NULL DEFAULT 0,
  ts                TIMESTAMP   NOT NULL DEFAULT current_timestamp(),
  CONSTRAINT pk_run_metrics PRIMARY KEY (run_id, turn)
)
CLUSTER BY (ts)
TBLPROPERTIES ('delta.appendOnly' = 'true');

-- Payload-level log (for compliance reviewers). Distinct from gateway_logs
-- (which is a volume histogram) â€” this one captures a sample of bodies.
CREATE TABLE IF NOT EXISTS media.ops.gateway_logs_payload (
  id            STRING        NOT NULL,
  trace_id      STRING        NOT NULL,
  ts            TIMESTAMP     NOT NULL DEFAULT current_timestamp(),
  principal     STRING        NOT NULL,
  endpoint      STRING        NOT NULL,
  request_body  STRING,
  response_body STRING,
  flagged       BOOLEAN       NOT NULL DEFAULT false,
  CONSTRAINT pk_gateway_logs_payload PRIMARY KEY (id)
)
CLUSTER BY (trace_id, ts)
TBLPROPERTIES (
  'delta.appendOnly' = 'true',
  'delta.enableChangeDataFeed' = 'true',
  'delta.deletedFileRetentionDuration' = 'interval 90 days'
);

-- ---------------------------------------------------------------------------
-- Stage / briefing tables
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS media.alex.email_drafts (
  id            STRING        NOT NULL,
  email_id      STRING        NOT NULL,
  author_email  STRING        NOT NULL,
  body          STRING        NOT NULL,
  state         STRING        NOT NULL DEFAULT 'pending',  -- 'pending'|'approved'|'sent'|'rejected'
  created_at    TIMESTAMP     NOT NULL DEFAULT current_timestamp(),
  decided_at    TIMESTAMP,
  decided_by    STRING,
  CONSTRAINT pk_email_drafts PRIMARY KEY (id),
  CONSTRAINT chk_email_drafts_state CHECK (state IN ('pending','approved','sent','rejected'))
)
CLUSTER BY (author_email, state);

CREATE TABLE IF NOT EXISTS media.alex.standing_order_files (
  order_id      STRING        NOT NULL,
  path          STRING        NOT NULL,        -- relative path within the approved library
  uploaded_by   STRING        NOT NULL,
  uploaded_at   TIMESTAMP     NOT NULL DEFAULT current_timestamp(),
  CONSTRAINT pk_standing_order_files PRIMARY KEY (order_id, path),
  CONSTRAINT fk_standing_order_files_order FOREIGN KEY (order_id)
    REFERENCES media.control.standing_orders(order_id)
)
CLUSTER BY (order_id);

-- ---------------------------------------------------------------------------
-- Media catalog
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS media.content.assets (
  asset_id      STRING        NOT NULL,
  title         STRING        NOT NULL,
  asset_type    STRING        NOT NULL,         -- 'video'|'audio'|'article'|'image'|'document'
  duration_s    INT,
  created_at    TIMESTAMP     NOT NULL DEFAULT current_timestamp(),
  ingest_path   STRING,
  rights_status STRING        NOT NULL DEFAULT 'unknown',  -- 'cleared'|'unknown'|'expired'|'restricted'
  rights_refresh_at TIMESTAMP,
  CONSTRAINT pk_assets PRIMARY KEY (asset_id),
  CONSTRAINT chk_assets_rights CHECK (rights_status IN ('cleared','unknown','expired','restricted'))
)
CLUSTER BY (asset_type);

CREATE TABLE IF NOT EXISTS media.content.transcripts (
  id            STRING        NOT NULL,
  asset_id      STRING        NOT NULL,
  chunk_index   INT           NOT NULL,
  ts_start      DOUBLE,
  ts_end        DOUBLE,
  chunk         STRING        NOT NULL,
  rights_status STRING        NOT NULL DEFAULT 'unknown',
  sensitivity   STRING        NOT NULL DEFAULT 'public',  -- 'public'|'internal'|'restricted'|'pii'
  CONSTRAINT pk_transcripts PRIMARY KEY (id),
  CONSTRAINT fk_transcripts_asset FOREIGN KEY (asset_id) REFERENCES media.content.assets(asset_id)
)
CLUSTER BY (asset_id);

CREATE TABLE IF NOT EXISTS media.content.captions (
  id            STRING        NOT NULL,
  asset_id      STRING        NOT NULL,
  language      STRING        NOT NULL,
  text          STRING        NOT NULL,
  ts_start      DOUBLE,
  ts_end        DOUBLE,
  CONSTRAINT pk_captions PRIMARY KEY (id),
  CONSTRAINT fk_captions_asset FOREIGN KEY (asset_id) REFERENCES media.content.assets(asset_id)
)
CLUSTER BY (asset_id, language);

-- Rights metadata extracted from contract documents by the content-indexing
-- job (ai_extract) and JOINed by the rights-check UDF in tools.sql. Columns
-- must match pipelines/content_indexing.py rights_rows schema; the job MERGEs
-- on (asset_id, source_doc).
CREATE TABLE IF NOT EXISTS media.content.rights (
  asset_id      STRING,
  license_start STRING,
  license_end   STRING,
  territories   ARRAY<STRING>,
  source_doc    STRING,
  CONSTRAINT pk_rights PRIMARY KEY (asset_id, source_doc)
)
CLUSTER BY (asset_id);

-- Grants for every principal live in grants.sql, applied LAST by the
-- DDL job — they reference tables created across all the DDL files.
