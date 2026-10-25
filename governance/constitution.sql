-- ===========================================================================
-- constitution.sql — the supreme law of the Alex system.
-- Append-only. Amendments are a separate, also-append-only table.
-- The 9 articles A1..A9 correspond 1:1 to the machine_rule JSON shapes that
-- `agent.constitution.ConstitutionalInterceptor` evaluates.
-- ===========================================================================

CREATE TABLE IF NOT EXISTS media.control.constitution (
  article_id      STRING        NOT NULL,
  article_type    STRING        NOT NULL,            -- 'prohibition'|'procedure'|'principle'
  title           STRING        NOT NULL,
  text            STRING        NOT NULL,
  -- machine_rule is a JSON shape that the agent's interceptor parses.
  -- Examples:
  --   A1: {"deny_capabilities": ["email.send"]}
  --   A3: {"require_log": true, "log_to": "agent_actions"}
  --   A4: {"procedure": {
  --          "trigger_tools": ["draft_email_reply","send_email","create_task"],
  --          "required_prior": [
  --            {"tool": "check_rights", "must_return_not": "NOT_CLEARED"}
  --          ]
  --        }}
  --   A7: {"max_pii_chars": 256, "behavior": "redact"}
  machine_rule    STRING        NOT NULL,
  severity        STRING        NOT NULL DEFAULT 'blocking',  -- 'blocking'|'warning'
  version         INT           NOT NULL DEFAULT 1,
  effective_from  TIMESTAMP     NOT NULL DEFAULT current_timestamp(),
  superseded_by   STRING,                              -- article_id of the version that replaces this one
  CONSTRAINT pk_constitution PRIMARY KEY (article_id, version),
  CONSTRAINT chk_constitution_severity CHECK (severity IN ('blocking','warning')),
  CONSTRAINT chk_constitution_type CHECK (article_type IN ('prohibition','procedure','principle'))
)
CLUSTER BY (article_id);

-- Amendments — append-only. A new version is INSERTed, the previous row is
-- UPDATEd with `superseded_by` set. There is no DELETE.
CREATE TABLE IF NOT EXISTS media.control.constitution_amendments (
  amendment_id    STRING        NOT NULL,
  article_id      STRING        NOT NULL,
  previous_version INT,
  new_version     INT           NOT NULL,
  proposed_by     STRING        NOT NULL,
  ratified_by     STRING        NOT NULL,
  rationale       STRING        NOT NULL,
  effective_at    TIMESTAMP     NOT NULL DEFAULT current_timestamp(),
  CONSTRAINT pk_constitution_amendments PRIMARY KEY (amendment_id)
)
CLUSTER BY (article_id, effective_at)
TBLPROPERTIES (
  'delta.appendOnly' = 'true',
  'delta.enableChangeDataFeed' = 'true'
);

-- Violation log — append-only. Both the inline critic and the async judges
-- write here. Governor + agent SP have SELECT, INSERT only — no UPDATE.
CREATE TABLE IF NOT EXISTS media.control.constitution_violations (
  id              STRING        NOT NULL,
  ts              TIMESTAMP     NOT NULL DEFAULT current_timestamp(),
  user_email      STRING        NOT NULL,
  session_id      STRING,
  article_id      STRING        NOT NULL,
  action          STRING        NOT NULL,
  disposition     STRING        NOT NULL,            -- 'blocked' | 'warned'
  detail          STRING,
  CONSTRAINT pk_constitution_violations PRIMARY KEY (id),
  CONSTRAINT chk_constitution_violations_disp CHECK (disposition IN ('blocked','warned'))
)
CLUSTER BY (article_id, ts)
TBLPROPERTIES (
  'delta.appendOnly' = 'true',
  'delta.enableChangeDataFeed' = 'true',
  'delta.deletedFileRetentionDuration' = 'interval 365 days'
);

-- ---------------------------------------------------------------------------
-- Seed articles A1..A9 (idempotent via MERGE).
--   A1 — No autonomous sends.          prohibition, blocking
--   A2 — User identity is sacred.      principle,   blocking
--   A3 — Attribution of every act.     procedure,   blocking
--   A4 — Rights clearance before reuse. procedure, blocking
--   A5 — Public copy requires review.  procedure,   blocking
--   A6 — Audience numbers come from Genie. principle, blocking
--   A7 — Data minimum in outputs.      procedure,   warning
--   A8 — Every externalizing tool is two-phase. procedure, blocking
--   A9 — Subagents inherit scope.      principle,   blocking
-- ---------------------------------------------------------------------------

MERGE INTO media.control.constitution AS t
USING (
  SELECT * FROM VALUES
  ('A1', 'prohibition', 'No autonomous sends',
   'The agent never sends email, posts, or publishes without a human approval task.',
   '{"deny_capabilities": ["email.send","publish.post","publish.tweet"]}',
   'blocking', 1),
  ('A2', 'principle',   'User identity is sacred',
   'The agent uses the signed-in user''s identity and authority; it never widens, never narrows, never impersonates.',
   '{"enforce": "identity_from_obo_only"}',
   'blocking', 1),
  ('A3', 'procedure',   'Attribution of every consequential act',
   'Every tool call that changes state writes a row to media.alex.agent_actions with actor, session, action, target, detail, outcome.',
   '{"require_log": true, "log_to": "agent_actions", "min_fields": ["actor","action_type","target_id","outcome"]}',
   'blocking', 1),
  ('A4', 'procedure',   'Rights clearance before reuse',
   'Before recommending reuse of any media asset, the agent must call media.alex.check_rights and the result must be CLEARED.',
   '{"procedure": {"trigger_tools": ["draft_email_reply","create_task","api_act","send_email","publish.post"], "required_prior": [{"tool": "check_rights", "must_return_not": "NOT_CLEARED"}, {"tool": "check_rights", "must_return_not": "OUT_OF_WINDOW"}, {"tool": "check_rights", "must_return_not": "OUT_OF_TERRITORY"}]}}',
   'blocking', 1),
  ('A5', 'procedure',   'Public copy requires human review',
   'Any copy that will be published externally is created as a draft + approval task; the model never publishes directly.',
   '{"procedure": {"trigger_tools": ["publish.post","api_act"], "required_prior": [{"tool": "create_approval_task"}]}}',
   'blocking', 1),
  ('A6', 'principle',   'Audience numbers come from the source of truth',
   'Ratings, engagement, and demo performance numbers come only from media.audience via the Genie tool. The agent never estimates or recalls these from training data.',
   '{"deny_implicit_metrics": true, "source_of_truth_tool": "audience_analytics"}',
   'blocking', 1),
  ('A7', 'procedure',   'Data minimum in outputs',
   'Redact PII/PHI beyond a small per-field cap before any output leaves the agent''s context. A 7-line PII block is a warning, not a block.',
   '{"max_pii_chars": 256, "behavior": "redact", "severity": "warning"}',
   'warning', 1),
  ('A8', 'procedure',   'Externalizing tools are two-phase',
   'Every browser submit, every internal API act, and every workflow trigger stages the action for human review; a confirmation token unlocks execution.',
   '{"procedure": {"trigger_tools": ["browser_submit","api_act","trigger_workflow"], "required_prior": [{"tool": "stage_action", "must_have_field": "confirmation_token"}]}}',
   'blocking', 1),
  ('A9', 'principle',   'Subagents inherit the parent''s scope',
   'A subagent cannot use a capability the parent does not have. The dual-ACL check runs again on every subagent invocation.',
   '{"enforce": "dual_acl_on_subagents"}',
   'blocking', 1)
  AS s(article_id, article_type, title, text, machine_rule, severity, version)
) AS src
ON t.article_id = src.article_id AND t.version = src.version
WHEN NOT MATCHED THEN INSERT
  (article_id, article_type, title, text, machine_rule, severity, version, effective_from)
  VALUES (src.article_id, src.article_type, src.title, src.text, src.machine_rule, src.severity, src.version, current_timestamp());
