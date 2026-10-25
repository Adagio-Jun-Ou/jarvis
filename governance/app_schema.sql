-- ===========================================================================
-- app_schema.sql — application / email schema.
-- Tables in this file are written to by the ingest and summarization
-- pipelines. The agent endpoint READS them but never writes directly.
-- ===========================================================================

CREATE TABLE IF NOT EXISTS media.alex.emails (
  id              STRING        NOT NULL,
  owner_email     STRING        NOT NULL,
  graph_id        STRING,                             -- Graph message id
  from_email      STRING,
  to_emails       ARRAY<STRING>,
  cc_emails       ARRAY<STRING>,
  subject         STRING,
  body            STRING,
  received_at     TIMESTAMP     NOT NULL,
  is_read         BOOLEAN       NOT NULL DEFAULT false,
  is_archived     BOOLEAN       NOT NULL DEFAULT false,
  folder          STRING        NOT NULL DEFAULT 'inbox',
  has_attachments BOOLEAN       NOT NULL DEFAULT false,
  created_at      TIMESTAMP     NOT NULL DEFAULT current_timestamp(),
  CONSTRAINT pk_emails PRIMARY KEY (id),
  CONSTRAINT fk_emails_owner FOREIGN KEY (owner_email) REFERENCES media.alex.users(email)
)
CLUSTER BY (owner_email, received_at)
PARTITIONED BY (date(received_at));

-- 1:1 per email. Backfill-safe: rerunning the summarization job replaces.
CREATE TABLE IF NOT EXISTS media.alex.email_summaries (
  email_id        STRING        NOT NULL,
  owner_email     STRING        NOT NULL,
  summary         STRING        NOT NULL,
  urgency         INT           NOT NULL DEFAULT 3,
  topics          ARRAY<STRING>,
  model           STRING        NOT NULL,
  prompt_version  STRING        NOT NULL,
  created_at      TIMESTAMP     NOT NULL DEFAULT current_timestamp(),
  updated_at      TIMESTAMP     NOT NULL DEFAULT current_timestamp(),
  CONSTRAINT pk_email_summaries PRIMARY KEY (email_id),
  CONSTRAINT fk_email_summaries_email FOREIGN KEY (email_id) REFERENCES media.alex.emails(id),
  CONSTRAINT chk_email_summaries_urgency CHECK (urgency BETWEEN 1 AND 5)
)
CLUSTER BY (owner_email, urgency);

-- Structured extraction. One row per email.
CREATE TABLE IF NOT EXISTS media.alex.email_intel (
  email_id        STRING        NOT NULL,
  owner_email     STRING        NOT NULL,
  priority        INT,
  why_priority    STRING,
  commitments     ARRAY<STRUCT<who STRING, what STRING, when_due TIMESTAMP>>,
  asks_of_user    ARRAY<STRUCT<who STRING, what STRING, when_due TIMESTAMP>>,
  understanding   STRING,                             -- free-form, sanitized
  sentiment       STRING,                             -- positive|neutral|negative|mixed
  model           STRING        NOT NULL,
  prompt_version  STRING        NOT NULL,
  created_at      TIMESTAMP     NOT NULL DEFAULT current_timestamp(),
  updated_at      TIMESTAMP     NOT NULL DEFAULT current_timestamp(),
  CONSTRAINT pk_email_intel PRIMARY KEY (email_id),
  CONSTRAINT fk_email_intel_email FOREIGN KEY (email_id) REFERENCES media.alex.emails(id),
  CONSTRAINT chk_email_intel_priority CHECK (priority IS NULL OR priority BETWEEN 1 AND 5)
)
CLUSTER BY (owner_email, priority);

-- Intel quarantine: emails whose intel extraction failed to parse or was
-- refused by the model (or tripped injection defenses). Append-only; the
-- intel job writes here and an operator/re-prompt job drains it. Columns
-- must match pipelines/email_intel_job.py Q_SCHEMA exactly.
CREATE TABLE IF NOT EXISTS media.alex.email_intel_quarantine (
  email_id        STRING        NOT NULL,
  reason          STRING        NOT NULL,
  raw             STRING,
  created_at      TIMESTAMP     NOT NULL DEFAULT current_timestamp(),
  CONSTRAINT pk_email_intel_quarantine PRIMARY KEY (email_id, created_at)
)
CLUSTER BY (email_id)
TBLPROPERTIES ('delta.appendOnly' = 'true');

-- Incremental-sync cursor per user (Graph deltaLink). The ingest job
-- MERGEs on user_email so re-runs resume rather than re-pull. Columns
-- must match pipelines/email_ingest.py _save_delta_link.
CREATE TABLE IF NOT EXISTS media.alex.email_ingest_state (
  user_email      STRING        NOT NULL,
  delta_link      STRING        NOT NULL,
  last_sync_at    TIMESTAMP,
  CONSTRAINT pk_email_ingest_state PRIMARY KEY (user_email),
  CONSTRAINT fk_email_ingest_state_user FOREIGN KEY (user_email)
    REFERENCES media.alex.users(email)
)
CLUSTER BY (user_email);

-- Quarantine: parse failures or model refusals. The summarization job
-- reads from here to drive a re-prompt with a different model.
CREATE TABLE IF NOT EXISTS media.alex.email_summaries_quarantine (
  id              STRING        NOT NULL,
  email_id        STRING        NOT NULL,
  reason          STRING        NOT NULL,
  raw_response    STRING,
  created_at      TIMESTAMP     NOT NULL DEFAULT current_timestamp(),
  CONSTRAINT pk_quarantine PRIMARY KEY (id)
)
CLUSTER BY (email_id);

-- Morning brief, per user per day. Job writes; agent reads.
CREATE TABLE IF NOT EXISTS media.alex.morning_briefs (
  user_email      STRING        NOT NULL,
  brief_date      DATE          NOT NULL,
  brief           STRING        NOT NULL,
  standing_orders ARRAY<STRUCT<order_id STRING, title STRING, summary STRING, suggested_actions ARRAY<STRUCT<action STRING, capability STRING, detail STRING>>>>,
  priority_emails INT,
  tasks_due       INT,
  created_at      TIMESTAMP     NOT NULL DEFAULT current_timestamp(),
  CONSTRAINT pk_morning_briefs PRIMARY KEY (user_email, brief_date),
  CONSTRAINT fk_morning_briefs_user FOREIGN KEY (user_email) REFERENCES media.alex.users(email)
)
CLUSTER BY (user_email, brief_date)
PARTITIONED BY (brief_date);

-- The "inbox" view the dock calls /api/inbox with. Materialized daily by
-- the executive-metrics job; the dock reads cheap.
-- Denormalized read model for the Email tab: one row per email joining the
-- raw email (body/thread), its intel (priority) and its summary. Populated
-- by the executive-metrics job; the app reads it for the detail view (body).
CREATE TABLE IF NOT EXISTS media.alex.inbox (
  owner_email     STRING        NOT NULL,
  email_id        STRING        NOT NULL,
  priority        INT,
  subject         STRING,
  from_email      STRING,
  received_at     TIMESTAMP,
  body            STRING,
  summary         STRING,
  thread_id       STRING,
  CONSTRAINT pk_inbox PRIMARY KEY (owner_email, email_id)
)
CLUSTER BY (owner_email, received_at);

CREATE TABLE IF NOT EXISTS media.alex.inbox_view (
  owner_email     STRING        NOT NULL,
  priority        INT,
  email_id        STRING        NOT NULL,
  subject         STRING,
  from_email      STRING,                              -- surfaced by /api/inbox + the Email UI
  received_at     TIMESTAMP,
  summary         STRING,
  CONSTRAINT pk_inbox_view PRIMARY KEY (owner_email, email_id)
)
CLUSTER BY (owner_email, priority, received_at)
PARTITIONED BY (date(received_at));

-- The "task board" view.
CREATE TABLE IF NOT EXISTS media.alex.task_board_view (
  owner_email     STRING        NOT NULL,
  status          STRING        NOT NULL,
  task_id         STRING        NOT NULL,
  title           STRING,
  priority        INT,
  due_at          TIMESTAMP,
  CONSTRAINT pk_task_board_view PRIMARY KEY (owner_email, status, task_id)
)
CLUSTER BY (owner_email, status)
PARTITIONED BY (status);

-- Executive metrics schema + views. No content columns anywhere.
CREATE SCHEMA IF NOT EXISTS media.exec;

CREATE OR REPLACE VIEW media.exec.daily_active_users AS
SELECT date(ts) AS day, count(DISTINCT actor) AS dau
FROM media.alex.agent_actions
GROUP BY date(ts);

CREATE OR REPLACE VIEW media.exec.constitutional_violations_by_article AS
SELECT article_id, date(ts) AS day, count(*) AS n
FROM media.control.constitution_violations
GROUP BY article_id, date(ts);

-- k-anonymous: only emit counts where the subject group is >= 5 to
-- prevent re-identification in small teams.
CREATE OR REPLACE VIEW media.exec.risk_summary AS
SELECT date(ts) AS day, signal_type,
       count(*) AS n,
       count(DISTINCT subject_email) AS distinct_subjects
FROM media.control.risk_signals
GROUP BY date(ts), signal_type
HAVING count(DISTINCT subject_email) >= 5;

CREATE OR REPLACE VIEW media.exec.delegation_activity AS
SELECT date(granted_at) AS day, capability, count(*) AS n
FROM media.control.delegations
GROUP BY date(granted_at), capability;

-- How the agent's working turns perform: volume, latency (avg + P95),
-- and token spend, by day. Aggregates only — no task titles, no run
-- text, nothing a person wrote.
CREATE OR REPLACE VIEW media.exec.run_performance AS
SELECT date(ts) AS day,
       count(*) AS turns,
       count(DISTINCT run_id) AS runs,
       avg(elapsed_s) AS avg_turn_seconds,
       percentile_approx(elapsed_s, 0.95) AS p95_turn_seconds,
       sum(waited_s) AS human_wait_seconds,
       sum(tool_calls) AS tool_calls,
       sum(prompt_tokens) AS prompt_tokens,
       sum(completion_tokens) AS completion_tokens
FROM media.ops.run_metrics
GROUP BY date(ts);

-- ---------------------------------------------------------------------------
-- Grants for the views
-- ---------------------------------------------------------------------------
GRANT USAGE ON SCHEMA media.exec TO `alex-executives`;
GRANT SELECT ON media.exec.daily_active_users             TO `alex-executives`;
GRANT SELECT ON media.exec.constitutional_violations_by_article TO `alex-executives`;
GRANT SELECT ON media.exec.risk_summary                   TO `alex-executives`;
GRANT SELECT ON media.exec.delegation_activity            TO `alex-executives`;
GRANT SELECT ON media.exec.run_performance                TO `alex-executives`;
