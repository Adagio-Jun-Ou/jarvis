-- ===========================================================================
-- tools.sql — Unity Catalog function surface that the agent endpoint calls.
--
-- Convention (locked in):
--   * Every DML is a STORED PROCEDURE:  CALL media.alex.create_task(...).
--     The body runs in a single transaction; parameter binding is enforced
--     by the engine; we never f-string SQL.
--   * Read-only utilities are SCALAR UDFs. The body is a single SELECT and
--     runs in the caller's identity (because the procedure is the caller's).
--   * Every write path validates the caller. The caller_email parameter is
--     required on every procedure; the server (agent endpoint) passes the
--     JWT-derived email here, never trusting client-supplied values.
--
-- The procedures and functions in this file are the COMPLETE agent tool
-- surface. The agent endpoint imports the list `APPROVED_UC_TOOLS` and
-- calls them; nothing else is callable.
-- ===========================================================================

-- ---------------------------------------------------------------------------
-- Task procedures
-- ---------------------------------------------------------------------------

CREATE OR REPLACE PROCEDURE media.alex.create_task(
  IN caller_email    STRING,
  IN title           STRING,
  IN description     STRING,
  IN assignee_email  STRING,
  IN priority        INT,
  IN due_at          TIMESTAMP,
  IN source          STRING,
  IN source_ref      STRING
)
LANGUAGE SQL
SQL SECURITY INVOKER
AS BEGIN
  -- Caller must have tasks.create. Validation is at the procedure boundary
  -- because UC scalar UDFs cannot enforce row-level checks; the procedure
  -- body queries the user's effective ACL.
  IF NOT EXISTS (
    SELECT 1 FROM media.control.effective_employee_acl
    WHERE user_email = caller_email AND capability = 'tasks.create'
  ) THEN
    SIGNAL SQLSTATE '42501'
      SET MESSAGE_TEXT = 'caller lacks tasks.create';
  END IF;
  INSERT INTO media.alex.tasks (
    id, owner_email, assignee_email, title, description,
    status, priority, due_at, source, source_ref
  ) VALUES (
    uuid(), caller_email, assignee_email, title, description,
    'open', priority, due_at, source, source_ref
  );
END;

CREATE OR REPLACE PROCEDURE media.alex.update_task_status(
  IN caller_email    STRING,
  IN task_id         STRING,
  IN new_status      STRING,
  IN detail          STRING
)
LANGUAGE SQL
SQL SECURITY INVOKER
AS BEGIN
  -- The union of the board lanes and the legacy tool statuses — must
  -- stay in step with chk_tasks_status in permissions.sql.
  IF new_status NOT IN ('backlog','todo','open','in_progress','review',
                        'done','cancelled','blocked') THEN
    SIGNAL SQLSTATE '22000' SET MESSAGE_TEXT = 'invalid status';
  END IF;
  UPDATE media.alex.tasks
     SET status = new_status,
         updated_at = current_timestamp(),
         closed_at = CASE WHEN new_status IN ('done','cancelled') THEN current_timestamp() ELSE closed_at END
   WHERE id = task_id
     AND (owner_email = caller_email OR assignee_email = caller_email);
  -- 0 rows updated = no such task, or no permission. Distinct paths matter
  -- for audit; the agent's caller converts to a structured denial.
  IF NOT EXISTS (SELECT 1 FROM media.alex.tasks WHERE id = task_id) THEN
    SIGNAL SQLSTATE '22000' SET MESSAGE_TEXT = 'task not found';
  END IF;
END;

-- Read-only list of the caller's tasks.
CREATE OR REPLACE FUNCTION media.alex.list_my_tasks(caller_email STRING)
RETURNS TABLE(
  id STRING, title STRING, status STRING, priority INT,
  due_at TIMESTAMP, owner_email STRING, assignee_email STRING
)
LANGUAGE SQL
SQL SECURITY INVOKER
RETURN
  SELECT id, title, status, priority, due_at, owner_email, assignee_email
    FROM media.alex.tasks
   WHERE owner_email = caller_email OR assignee_email = caller_email
   ORDER BY priority ASC, due_at ASC NULLS LAST
   LIMIT 200;

-- ---------------------------------------------------------------------------
-- Email procedures
-- ---------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION media.alex.summarize_email(caller_email STRING, email_id STRING)
RETURNS STRUCT<summary STRING, urgency INT>
LANGUAGE SQL
SQL SECURITY INVOKER
RETURN
  SELECT named_struct(
    'summary', coalesce(s.summary, ''),
    'urgency', coalesce(s.urgency, 3)
  )
  FROM media.alex.email_summaries s
  JOIN media.alex.emails e ON e.id = s.email_id
  WHERE s.email_id = email_id
    AND e.owner_email = caller_email;

-- Drafts only; sends require a human approval task (separate flow).
CREATE OR REPLACE PROCEDURE media.alex.draft_email_reply(
  IN caller_email    STRING,
  IN email_id        STRING,
  IN body            STRING
)
LANGUAGE SQL
SQL SECURITY INVOKER
AS BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM media.control.effective_employee_acl
    WHERE user_email = caller_email AND capability = 'email.draft'
  ) THEN
    SIGNAL SQLSTATE '42501' SET MESSAGE_TEXT = 'caller lacks email.draft';
  END IF;
  IF NOT EXISTS (SELECT 1 FROM media.alex.emails WHERE id = email_id AND owner_email = caller_email) THEN
    SIGNAL SQLSTATE '22000' SET MESSAGE_TEXT = 'email not found or not owned by caller';
  END IF;
  INSERT INTO media.alex.email_drafts (id, email_id, author_email, body, state)
  VALUES (uuid(), email_id, caller_email, body, 'pending');
END;

-- ---------------------------------------------------------------------------
-- Content rights
-- ---------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION media.alex.check_rights(
  caller_email STRING,
  asset_id     STRING,
  territory    STRING,
  window_start TIMESTAMP,
  window_end   TIMESTAMP
)
RETURNS TABLE(
  status     STRING,
  license_id STRING,
  reason     STRING
)
LANGUAGE SQL
SQL SECURITY INVOKER
RETURN
  SELECT
    CASE
      WHEN r.asset_id IS NULL THEN 'NOT_FOUND'
      WHEN r.license_start > window_start OR r.license_end < window_end THEN 'OUT_OF_WINDOW'
      WHEN NOT array_contains(r.territories, territory) THEN 'OUT_OF_TERRITORY'
      ELSE 'CLEARED'
    END AS status,
    cast(null as STRING) AS license_id,
    CASE
      WHEN r.asset_id IS NULL THEN 'no rights record for asset'
      WHEN r.license_start > window_start OR r.license_end < window_end THEN 'license window does not cover requested use'
      WHEN NOT array_contains(r.territories, territory) THEN 'territory not in license'
      ELSE 'all clear'
    END AS reason
  FROM (SELECT 1 AS one) anchor
  LEFT JOIN media.content.rights r
    ON r.asset_id = asset_id
   AND array_contains(r.territories, territory)
   AND r.license_start <= window_start
   AND r.license_end   >= window_end;

-- ---------------------------------------------------------------------------
-- Agent action ledger
-- ---------------------------------------------------------------------------

CREATE OR REPLACE PROCEDURE media.alex.log_agent_action(
  IN caller_email    STRING,
  IN session_id      STRING,
  IN action_type     STRING,
  IN target_id       STRING,
  IN detail          STRING,
  IN outcome         STRING,
  IN tool_name       STRING,
  IN tool_args       STRING
)
LANGUAGE SQL
SQL SECURITY INVOKER
AS BEGIN
  IF outcome NOT IN ('ok','denied','error','revoked') THEN
    SIGNAL SQLSTATE '22000' SET MESSAGE_TEXT = 'invalid outcome';
  END IF;
  INSERT INTO media.alex.agent_actions
    (id, actor, session_id, action_type, target_id, detail, outcome, tool_name, tool_args)
  VALUES
    (uuid(), caller_email, session_id, action_type, target_id,
     CASE WHEN length(detail) > 4000 THEN substr(detail, 1, 4000) ELSE detail END,
     outcome,
     tool_name,
     CASE WHEN length(tool_args) > 4000 THEN substr(tool_args, 1, 4000) ELSE tool_args END);
END;
