-- ===========================================================================
-- control_plane.sql — governor-owned control plane.
-- This is the schema that defines what the system IS allowed to do.
-- All writes are by alex-governors or sp-alex-jobs. The agent and the app
-- have SELECT only.
-- ===========================================================================

-- ---------------------------------------------------------------------------
-- Dual ACL: every employee has a base ACL; every "instance" (think: an
-- organization's deployment of the Alex template) has a cap on what it may
-- grant. The effective ACL is the INTERSECTION of these two; we never
-- compute that in SQL because the second axis (instance) is per-row, not
-- per-user. The agent enforces the intersection at tool-bind time.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS media.control.employee_acl (
  user_email      STRING        NOT NULL,
  capability      STRING        NOT NULL,         -- e.g. 'tasks.create', 'email.draft', 'compute.bash'
  granted_by      STRING        NOT NULL,
  granted_at      TIMESTAMP     NOT NULL DEFAULT current_timestamp(),
  expires_at      TIMESTAMP,                       -- null = until revoked
  notes           STRING,
  CONSTRAINT pk_employee_acl PRIMARY KEY (user_email, capability)
)
CLUSTER BY (user_email);

CREATE TABLE IF NOT EXISTS media.control.instance_acl (
  instance_scope  STRING        NOT NULL,         -- e.g. 'media-corp', 'media-corp:legal'
  capability      STRING        NOT NULL,
  rationale       STRING        NOT NULL,
  CONSTRAINT pk_instance_acl PRIMARY KEY (instance_scope, capability)
)
CLUSTER BY (instance_scope);

-- ---------------------------------------------------------------------------
-- Sessions (per-user, per-conversation, per-group, per-branch).
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS media.control.sessions (
  session_id      STRING        NOT NULL,
  owner_email     STRING        NOT NULL,
  kind            STRING        NOT NULL DEFAULT 'personal',  -- 'personal'|'group'|'branch'
  title           STRING,
  parent_id       STRING,                              -- for branches
  participants    ARRAY<STRING>,                       -- for group sessions
  model           STRING,
  created_at      TIMESTAMP     NOT NULL DEFAULT current_timestamp(),
  last_active_at  TIMESTAMP     NOT NULL DEFAULT current_timestamp(),
  CONSTRAINT pk_sessions PRIMARY KEY (session_id),
  CONSTRAINT fk_sessions_owner FOREIGN KEY (owner_email) REFERENCES media.alex.users(email),
  CONSTRAINT chk_sessions_kind CHECK (kind IN ('personal','group','branch'))
)
CLUSTER BY (owner_email, last_active_at);

-- Per-turn bookkeeping. The session manager writes here on every turn.
-- Used for "did the optimistic update land" and turn ordering.
CREATE TABLE IF NOT EXISTS media.control.session_turns (
  session_id      STRING        NOT NULL,
  turn_no         INT           NOT NULL,
  role            STRING        NOT NULL,            -- 'user'|'assistant'|'tool'|'system'
  content         STRING        NOT NULL,
  ts              TIMESTAMP     NOT NULL DEFAULT current_timestamp(),
  CONSTRAINT pk_session_turns PRIMARY KEY (session_id, turn_no),
  CONSTRAINT fk_session_turns_session FOREIGN KEY (session_id)
    REFERENCES media.control.sessions(session_id)
)
CLUSTER BY (session_id, turn_no);

-- ---------------------------------------------------------------------------
-- Knowledge: per-user chunks, instance library, approved files.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS media.control.user_knowledge_chunks (
  chunk_id        STRING        NOT NULL,
  owner_email     STRING        NOT NULL,
  source_path     STRING        NOT NULL,
  chunk_index     INT           NOT NULL,
  content         STRING        NOT NULL,
  embedding       ARRAY<FLOAT>,
  embedding_model STRING,
  created_at      TIMESTAMP     NOT NULL DEFAULT current_timestamp(),
  CONSTRAINT pk_user_knowledge_chunks PRIMARY KEY (chunk_id),
  CONSTRAINT fk_user_knowledge_owner FOREIGN KEY (owner_email) REFERENCES media.alex.users(email)
)
CLUSTER BY (owner_email);

-- The "approved files" table is the single source of truth for what may be
-- served to a user. instance_library references these.
CREATE TABLE IF NOT EXISTS media.control.approved_files (
  file_id         STRING        NOT NULL,
  path            STRING        NOT NULL,
  sha256          STRING        NOT NULL,
  size_bytes      BIGINT        NOT NULL,
  source_channel  STRING        NOT NULL,         -- 'standing_order'|'policy_best_practice'|'curated'
  approved_by     STRING        NOT NULL,
  approved_at     TIMESTAMP     NOT NULL DEFAULT current_timestamp(),
  expires_at      TIMESTAMP,
  revoked         BOOLEAN       NOT NULL DEFAULT false,
  revoked_at      TIMESTAMP,
  revoked_by      STRING,
  CONSTRAINT pk_approved_files PRIMARY KEY (file_id),
  CONSTRAINT uq_approved_files_path UNIQUE (path),
  CONSTRAINT chk_approved_files_source CHECK (source_channel IN ('standing_order','policy_best_practice','curated'))
)
CLUSTER BY (source_channel, revoked);

-- The per-user curated library. Only references approved_files; deletion
-- of an approved_file cascades to instance_library rows.
CREATE TABLE IF NOT EXISTS media.control.instance_library (
  owner_email     STRING        NOT NULL,
  file_id         STRING        NOT NULL,
  pinned_summary  STRING,
  added_at        TIMESTAMP     NOT NULL DEFAULT current_timestamp(),
  CONSTRAINT pk_instance_library PRIMARY KEY (owner_email, file_id),
  CONSTRAINT fk_instance_library_owner FOREIGN KEY (owner_email)
    REFERENCES media.alex.users(email),
  CONSTRAINT fk_instance_library_file FOREIGN KEY (file_id)
    REFERENCES media.control.approved_files(file_id)
)
CLUSTER BY (owner_email);

-- ---------------------------------------------------------------------------
-- Tools & workflows
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS media.control.tool_integrations (
  integration_id  STRING        NOT NULL,
  display_name    STRING        NOT NULL,
  mode            STRING        NOT NULL,             -- 'api'|'browser'|'hybrid'
  base_url        STRING,
  read_capability STRING,                             -- capability needed to use this tool
  act_capability  STRING,                             -- capability needed to act (write) via this tool
  act_requires_confirm BOOLEAN  NOT NULL DEFAULT true,
  auth_kind       STRING,                             -- 'okta_sso'|'oauth'|'service_account'
  obo_audience    STRING,                             -- OAuth audience for OBO token exchange
  obo_scopes      ARRAY<STRING>,
  vault_key_id    STRING,                             -- KMS DEK id for cookie storage
  api_spec_path   STRING,                             -- path in /Volumes/media/alex/api_specs/
  created_at      TIMESTAMP     NOT NULL DEFAULT current_timestamp(),
  updated_at      TIMESTAMP     NOT NULL DEFAULT current_timestamp(),
  CONSTRAINT pk_tool_integrations PRIMARY KEY (integration_id),
  CONSTRAINT chk_tool_integrations_mode CHECK (mode IN ('api','browser','hybrid'))
)
CLUSTER BY (mode);

-- Browser sessions: ciphertext cookies only, never plaintext.
CREATE TABLE IF NOT EXISTS media.control.browser_sessions (
  session_id        STRING        NOT NULL,
  user_email        STRING        NOT NULL,
  integration_id    STRING        NOT NULL,
  cookie_ciphertext STRING        NOT NULL,             -- envelope-encrypted
  vault_key_id      STRING        NOT NULL,             -- KMS DEK id
  cipher_suite      STRING        NOT NULL DEFAULT 'AES-256-GCM',
  created_at        TIMESTAMP     NOT NULL DEFAULT current_timestamp(),
  last_used_at      TIMESTAMP,
  expires_at        TIMESTAMP     NOT NULL,
  revoked           BOOLEAN       NOT NULL DEFAULT false,
  revoked_at        TIMESTAMP,
  CONSTRAINT pk_browser_sessions PRIMARY KEY (session_id),
  CONSTRAINT fk_browser_sessions_user FOREIGN KEY (user_email) REFERENCES media.alex.users(email),
  CONSTRAINT fk_browser_sessions_integration FOREIGN KEY (integration_id)
    REFERENCES media.control.tool_integrations(integration_id)
)
CLUSTER BY (user_email, revoked);

-- Append-only audit of every browser step the agent took.
CREATE TABLE IF NOT EXISTS media.control.browser_history (
  id              STRING        NOT NULL,
  ts              TIMESTAMP     NOT NULL DEFAULT current_timestamp(),
  user_email      STRING        NOT NULL,
  session_id      STRING,
  integration_id  STRING        NOT NULL,
  action          STRING        NOT NULL,             -- 'open'|'read'|'fill'|'click'|'submit'|'navigate'
  url             STRING,
  detail          STRING,                              -- bounded
  CONSTRAINT pk_browser_history PRIMARY KEY (id)
)
CLUSTER BY (user_email, ts)
TBLPROPERTIES (
  'delta.appendOnly' = 'true',
  'delta.enableChangeDataFeed' = 'true'
);

-- Workflow catalog: governed triggers (e.g., n8n webhook flows).
CREATE TABLE IF NOT EXISTS media.control.workflow_catalog (
  workflow_id     STRING        NOT NULL,
  name            STRING        NOT NULL,
  description     STRING        NOT NULL,
  capability      STRING        NOT NULL,             -- capability needed to invoke
  input_schema    STRING        NOT NULL,             -- JSON schema
  integration_id  STRING,
  requires_dual_approval BOOLEAN NOT NULL DEFAULT false,
  active          BOOLEAN       NOT NULL DEFAULT true,
  created_at      TIMESTAMP     NOT NULL DEFAULT current_timestamp(),
  CONSTRAINT pk_workflow_catalog PRIMARY KEY (workflow_id),
  CONSTRAINT fk_workflow_integration FOREIGN KEY (integration_id)
    REFERENCES media.control.tool_integrations(integration_id)
)
CLUSTER BY (active);

-- Workflow run audit.
CREATE TABLE IF NOT EXISTS media.control.workflow_runs (
  run_id          STRING        NOT NULL,
  workflow_id     STRING        NOT NULL,
  user_email      STRING        NOT NULL,
  ts              TIMESTAMP     NOT NULL DEFAULT current_timestamp(),
  input_payload   STRING,
  result_payload  STRING,
  outcome         STRING        NOT NULL DEFAULT 'ok',  -- 'ok'|'denied'|'error'
  CONSTRAINT pk_workflow_runs PRIMARY KEY (run_id),
  CONSTRAINT fk_workflow_runs_workflow FOREIGN KEY (workflow_id)
    REFERENCES media.control.workflow_catalog(workflow_id)
)
CLUSTER BY (user_email, ts)
TBLPROPERTIES (
  'delta.appendOnly' = 'true',
  'delta.enableChangeDataFeed' = 'true'
);

-- ---------------------------------------------------------------------------
-- Thinking traces — append-only. One definition. A single archive of every
-- step the model took, with retention for compliance review.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS media.control.thinking_traces (
  id              STRING        NOT NULL,
  ts              TIMESTAMP     NOT NULL DEFAULT current_timestamp(),
  user_email      STRING        NOT NULL,
  session_id      STRING        NOT NULL,
  turn_no         INT,
  phase           STRING        NOT NULL,             -- 'plan'|'tool_call'|'critique'|'final'|'error'
  raw_thinking    STRING        NOT NULL,             -- bounded to 60k chars by the agent
  digest          STRING,                              -- decision summary, <= 80 tokens
  CONSTRAINT pk_thinking_traces PRIMARY KEY (id)
)
CLUSTER BY (user_email, session_id, ts)
TBLPROPERTIES (
  'delta.appendOnly' = 'true',
  'delta.enableChangeDataFeed' = 'true',
  'delta.deletedFileRetentionDuration' = 'interval 30 days'
);

-- ---------------------------------------------------------------------------
-- Skills & learning (proposed by agent, ratified by governor, never
-- self-granted).
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS media.control.skills (
  skill_id          STRING        NOT NULL,
  owner_email       STRING        NOT NULL,
  name              STRING        NOT NULL,
  trigger_pattern   STRING        NOT NULL,
  procedure         ARRAY<STRING> NOT NULL,
  uses_capabilities ARRAY<STRING> NOT NULL,
  source_kind       STRING        NOT NULL DEFAULT 'observed',  -- 'observed'|'taught'
  status            STRING        NOT NULL DEFAULT 'proposed',  -- 'proposed'|'active'|'rejected'
  success_count     INT           NOT NULL DEFAULT 0,
  last_used         TIMESTAMP,
  ratified_by       STRING,
  ratified_at       TIMESTAMP,
  created_at        TIMESTAMP     NOT NULL DEFAULT current_timestamp(),
  CONSTRAINT pk_skills PRIMARY KEY (skill_id),
  CONSTRAINT fk_skills_owner FOREIGN KEY (owner_email) REFERENCES media.alex.users(email),
  CONSTRAINT chk_skills_status CHECK (status IN ('proposed','active','rejected')),
  CONSTRAINT chk_skills_source CHECK (source_kind IN ('observed','taught'))
)
CLUSTER BY (owner_email, status);

-- A skill can only become 'active' if every capability it uses is in the
-- user's effective ACL AND a governor has ratified it.
CREATE OR REPLACE FUNCTION media.control.can_ratify_skill(
  ratifier_email STRING, skill_id STRING
)
RETURNS BOOLEAN
LANGUAGE SQL
SQL SECURITY INVOKER
RETURN
  NOT EXISTS (
    SELECT 1
    FROM media.control.skills s
    CROSS JOIN LATERAL explode(s.uses_capabilities) AS cap
    LEFT JOIN media.control.effective_employee_acl e
      ON e.user_email = s.owner_email AND e.capability = cap
    WHERE s.skill_id = skill_id AND e.capability IS NULL
  );

CREATE TABLE IF NOT EXISTS media.control.taught_routines (
  routine_id      STRING        NOT NULL,
  owner_email     STRING        NOT NULL,
  prompt          STRING        NOT NULL,
  context         STRING,
  captured_at     TIMESTAMP     NOT NULL DEFAULT current_timestamp(),
  CONSTRAINT pk_taught_routines PRIMARY KEY (routine_id),
  CONSTRAINT fk_taught_routines_owner FOREIGN KEY (owner_email) REFERENCES media.alex.users(email)
)
CLUSTER BY (owner_email);

CREATE TABLE IF NOT EXISTS media.control.suggestion_feedback (
  id              STRING        NOT NULL,
  user_email      STRING        NOT NULL,
  suggestion_id   STRING        NOT NULL,
  feedback        STRING        NOT NULL,             -- 'helpful'|'unhelpful'|'inaccurate'|'other'
  comment         STRING,
  ts              TIMESTAMP     NOT NULL DEFAULT current_timestamp(),
  CONSTRAINT pk_suggestion_feedback PRIMARY KEY (id)
)
CLUSTER BY (user_email, ts);

-- ---------------------------------------------------------------------------
-- Standing orders
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS media.control.standing_orders (
  order_id        STRING        NOT NULL,
  owner_email     STRING        NOT NULL,
  title           STRING        NOT NULL,
  instruction     STRING        NOT NULL,             -- what to do when triggered
  schedule        STRING,                              -- cron or 'on_event:foo'
  audience        ARRAY<STRING> NOT NULL,             -- group ids
  allowed_capabilities ARRAY<STRING> NOT NULL,        -- the capabilities this order may invoke
  active          BOOLEAN       NOT NULL DEFAULT true,
  created_by      STRING        NOT NULL,
  -- How the order fires and where it applies (agent/standing_orders.py
  -- round-trips all three; they existed only in the dev seed before,
  -- so every order write would have failed in production).
  trigger         STRING        NOT NULL DEFAULT 'manual',  -- 'manual'|'schedule'|'event'
  ambient         BOOLEAN       NOT NULL DEFAULT true,
  applies_to      ARRAY<STRING>,
  created_at      TIMESTAMP     NOT NULL DEFAULT current_timestamp(),
  CONSTRAINT chk_standing_orders_trigger CHECK (
    trigger IN ('manual','schedule','event')
  ),
  CONSTRAINT pk_standing_orders PRIMARY KEY (order_id),
  CONSTRAINT fk_standing_orders_owner FOREIGN KEY (owner_email) REFERENCES media.alex.users(email)
)
CLUSTER BY (owner_email, active);

-- Where a standing order runs on demand. A person presses "Run now",
-- the order executes in its own VM session, and this row is the record.
-- `capabilities` is the EFFECTIVE set for that run: the intersection of
-- the order's declared ceiling with the runner's own ACL. An order can
-- narrow authority, never widen it.
CREATE TABLE IF NOT EXISTS media.control.standing_order_runs (
  run_id          STRING        NOT NULL,
  order_id        STRING        NOT NULL,
  session_id      STRING        NOT NULL,
  started_by      STRING        NOT NULL,
  status          STRING        NOT NULL DEFAULT 'running',
  started_at      TIMESTAMP     NOT NULL DEFAULT current_timestamp(),
  -- Cleared when the run ends. A silent heartbeat is how a run abandoned
  -- by a dead process is told apart from one still working; without it an
  -- order run sits at 'running' for ever. See agent/taskboard for the
  -- same mechanism on task runs.
  last_heartbeat  TIMESTAMP,
  ended_at        TIMESTAMP,
  capabilities    ARRAY<STRING>,
  summary         STRING,
  CONSTRAINT pk_standing_order_runs PRIMARY KEY (run_id),
  CONSTRAINT fk_standing_order_runs_order FOREIGN KEY (order_id)
    REFERENCES media.control.standing_orders(order_id),
  CONSTRAINT ck_standing_order_runs_status
    CHECK (status IN ('running', 'done', 'failed', 'stopped', 'interrupted'))
)
CLUSTER BY (order_id, started_at);

-- The supervision ledger. An independent review of what agent runs
-- actually did, written append-only and read by administrators through
-- /api/oversight/*. Supervision that is not visible is not supervision.
--
-- `source` distinguishes a deterministic rule from a model judgment.
-- Rules cannot be overturned by the model pass: a supervisor that could
-- clear its own findings would be one prompt injection away from being
-- the hole in the fence.
CREATE TABLE IF NOT EXISTS media.control.supervision_findings (
  run_id          STRING,
  user_email      STRING,
  code            STRING        NOT NULL,
  severity        STRING        NOT NULL DEFAULT 'info',
  summary         STRING        NOT NULL,
  detail          STRING,
  source          STRING        NOT NULL DEFAULT 'rule',
  found_at        TIMESTAMP     NOT NULL DEFAULT current_timestamp(),
  acknowledged_by STRING,
  acknowledged_at TIMESTAMP,
  CONSTRAINT ck_supervision_severity
    CHECK (severity IN ('info', 'concern', 'violation')),
  CONSTRAINT ck_supervision_source CHECK (source IN ('rule', 'model'))
)
TBLPROPERTIES (delta.appendOnly = true)
CLUSTER BY (severity, found_at);

-- Per-user workspace preferences. The server is the source of truth so
-- a person's workspace follows them across devices; the browser keeps a
-- write-through cache only so first paint is not a flash of the wrong
-- theme. Device-local things (panel widths, last open tab) stay local.
CREATE TABLE IF NOT EXISTS media.control.user_preferences (
  user_email      STRING        NOT NULL,
  name            STRING        NOT NULL,
  value           STRING        NOT NULL,         -- JSON-encoded
  updated_at      TIMESTAMP     NOT NULL DEFAULT current_timestamp(),
  CONSTRAINT pk_user_preferences PRIMARY KEY (user_email, name)
)
CLUSTER BY (user_email);

-- Workspace policy pins. A value here OVERRIDES the user's own
-- preference and is reported to the UI as governed, so an
-- administrator's decision is visible rather than a setting that
-- silently refuses to save.
CREATE TABLE IF NOT EXISTS media.control.workspace_policy (
  name            STRING        NOT NULL,
  value           STRING        NOT NULL,         -- JSON-encoded
  rationale       STRING,
  set_by          STRING        NOT NULL,
  set_at          TIMESTAMP     NOT NULL DEFAULT current_timestamp(),
  CONSTRAINT pk_workspace_policy PRIMARY KEY (name)
);

-- MCP servers this organisation has connected. Scope decides who can
-- SEE a server; `capability` decides who may USE it. A server's tool
-- descriptions are attacker-controlled text that lands in the model's
-- prompt, so `trust` governs whether they are sanitised.
CREATE TABLE IF NOT EXISTS media.control.mcp_servers (
  server_id       STRING        NOT NULL,
  name            STRING        NOT NULL,          -- namespace: mcp__<name>__<tool>
  transport       STRING        NOT NULL DEFAULT 'stdio',
  command         STRING,
  args            ARRAY<STRING>,
  url             STRING,
  capability      STRING        NOT NULL DEFAULT 'mcp.use',
  scope           STRING        NOT NULL DEFAULT 'enterprise',
  trust           STRING        NOT NULL DEFAULT 'untrusted',
  enabled         BOOLEAN       NOT NULL DEFAULT true,
  timeout_s       DOUBLE        NOT NULL DEFAULT 30.0,
  secret_ref      STRING,
  description     STRING,
  registered_by   STRING,
  registered_at   TIMESTAMP     NOT NULL DEFAULT current_timestamp(),
  CONSTRAINT pk_mcp_servers PRIMARY KEY (server_id),
  CONSTRAINT uq_mcp_servers_name UNIQUE (name),
  CONSTRAINT ck_mcp_servers_transport CHECK (transport IN ('stdio', 'http')),
  CONSTRAINT ck_mcp_servers_trust CHECK (trust IN ('trusted', 'untrusted'))
)
CLUSTER BY (scope, enabled);

-- The rug-pull defence. A server's advertised tool surface is
-- fingerprinted on first sight; if it later changes, `approved` flips
-- to false and its tools are withheld until a governor re-approves.
CREATE TABLE IF NOT EXISTS media.control.mcp_tool_pins (
  server_id       STRING        NOT NULL,
  fingerprint     STRING        NOT NULL,
  approved        BOOLEAN       NOT NULL DEFAULT true,
  tool_count      INT,
  first_seen      TIMESTAMP     NOT NULL DEFAULT current_timestamp(),
  changed_at      TIMESTAMP,
  approved_at     TIMESTAMP,
  approved_by     STRING,
  CONSTRAINT pk_mcp_tool_pins PRIMARY KEY (server_id),
  CONSTRAINT fk_mcp_tool_pins_server FOREIGN KEY (server_id)
    REFERENCES media.control.mcp_servers(server_id)
);

-- The best-practice library: how this organisation works.
--
-- This is NOT a document store. A practice is normative: it enters the
-- prompt as a directive above retrieved passages, and a `mandatory` one
-- must be followed and cited. `capabilities` is what makes it reliable
-- — a practice bound to a capability fires whenever that tool is in
-- play, however the request was phrased.
--
-- Revisions SUPERSEDE rather than edit, so a past decision can be
-- judged against the policy that was actually in force at the time.
CREATE TABLE IF NOT EXISTS media.control.best_practices (
  practice_id     STRING        NOT NULL,
  title           STRING        NOT NULL,
  guidance        STRING        NOT NULL,
  authority       STRING        NOT NULL DEFAULT 'recommended',
  applies_to      ARRAY<STRING>,
  capabilities    ARRAY<STRING>,
  rationale       STRING,
  examples        ARRAY<STRING>,
  anti_patterns   ARRAY<STRING>,
  scope           STRING        NOT NULL DEFAULT 'enterprise',
  classification  STRING        NOT NULL DEFAULT 'internal',
  owner           STRING,
  version         INT           NOT NULL DEFAULT 1,
  supersedes      STRING,
  superseded_by   STRING,
  active          BOOLEAN       NOT NULL DEFAULT true,
  source_ref      STRING,
  created_at      TIMESTAMP     NOT NULL DEFAULT current_timestamp(),
  updated_at      TIMESTAMP     NOT NULL DEFAULT current_timestamp(),
  CONSTRAINT pk_best_practices PRIMARY KEY (practice_id),
  CONSTRAINT ck_best_practices_authority
    CHECK (authority IN ('mandatory', 'recommended', 'advisory')),
  CONSTRAINT ck_best_practices_classification
    CHECK (classification IN ('public','internal','confidential','restricted'))
)
CLUSTER BY (scope, authority, active);

CREATE TABLE IF NOT EXISTS media.control.standing_order_briefs (
  brief_id        STRING        NOT NULL,
  order_id        STRING        NOT NULL,
  user_email      STRING        NOT NULL,
  brief_date      DATE          NOT NULL,
  compiled_context STRING        NOT NULL,             -- full LLM response
  suggested_actions ARRAY<STRUCT<action STRING, capability STRING, detail STRING>>,
  consumed        BOOLEAN       NOT NULL DEFAULT false,
  created_at      TIMESTAMP     NOT NULL DEFAULT current_timestamp(),
  CONSTRAINT pk_standing_order_briefs PRIMARY KEY (brief_id),
  CONSTRAINT fk_standing_order_briefs_order FOREIGN KEY (order_id)
    REFERENCES media.control.standing_orders(order_id)
)
CLUSTER BY (user_email, brief_date, consumed);

-- ---------------------------------------------------------------------------
-- VM policy (governor-owned; consumed by the agent when spawning work VMs)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS media.control.vm_policy (
  policy_id       STRING        NOT NULL,
  name            STRING        NOT NULL,
  egress          STRING        NOT NULL DEFAULT 'deny',  -- 'deny'|'allowlist'
  egress_allowlist ARRAY<STRING>,
  languages       ARRAY<STRING> NOT NULL,             -- languages allowed
  max_runtime_seconds INT       NOT NULL DEFAULT 600,
  max_memory_mb   INT           NOT NULL DEFAULT 4096,
  network_isolated BOOLEAN      NOT NULL DEFAULT true,
  active          BOOLEAN       NOT NULL DEFAULT true,
  CONSTRAINT pk_vm_policy PRIMARY KEY (policy_id)
)
CLUSTER BY (active);

-- Seed: the default policy for any agent-spawned VM.
MERGE INTO media.control.vm_policy AS t
USING (SELECT 'default-vm' AS policy_id, 'Default agent VM' AS name,
              'deny' AS egress,
              array() AS egress_allowlist,
              array('python') AS languages,
              600 AS max_runtime_seconds,
              4096 AS max_memory_mb,
              true AS network_isolated,
              true AS active) AS src
ON t.policy_id = src.policy_id
WHEN NOT MATCHED THEN INSERT
  (policy_id, name, egress, egress_allowlist, languages,
   max_runtime_seconds, max_memory_mb, network_isolated, active)
  VALUES (src.policy_id, src.name, src.egress, src.egress_allowlist, src.languages,
          src.max_runtime_seconds, src.max_memory_mb, src.network_isolated, src.active);
