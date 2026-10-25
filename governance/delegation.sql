-- ===========================================================================
-- delegation.sql — supervisor delegation, second-approver rules, effective
-- employee ACL.
-- ===========================================================================

-- Delegable capabilities list. Anything not in this list cannot be
-- delegated, period — even by a supervisor.
CREATE TABLE IF NOT EXISTS media.control.delegable_capabilities (
  capability      STRING        NOT NULL,
  rationale       STRING        NOT NULL,
  max_validity_days INT         NOT NULL DEFAULT 30,
  requires_second_approver BOOLEAN NOT NULL DEFAULT false,
  CONSTRAINT pk_delegable_capabilities PRIMARY KEY (capability)
);

-- Seed: the default set. Adding a capability = governor + governance change.
MERGE INTO media.control.delegable_capabilities AS t
USING (SELECT * FROM VALUES
  ('tasks.create',          'Routine task management',                  30, false),
  ('email.draft',           'Email drafting (still requires approval)',  30, false),
  ('files.read',            'Read files in user workspace',              30, false),
  ('files.write',           'Write files in user workspace',             7,  true),
  ('compute.python',        'Run sandboxed Python',                     14, true),
  ('workflow.trigger',      'Trigger a workflow in the catalog',         7,  true),
  ('internal_api.read',     'Read from an internal API',                14, false),
  ('internal_api.act',      'Act on an internal API',                    3,  true),
  -- Office document generation (moderate risk: writes to the workspace).
  ('office.read',           'Read Office documents (xlsx/pptx/docx)',   30, false),
  ('office.write',          'Generate Office documents',                14, false),
  -- High-risk VM capabilities: software install, shell, and network
  -- egress into storage. The riskiest ones require a second approver
  -- and get short validity so a standing grant cannot linger.
  ('compute.bash',          'Run allowlisted shell commands',            7, true),
  ('compute.shell',         'Run allowlisted VM commands (CLI)',         7, true),
  ('compute.install',       'Install Python packages into the VM',       3, true),
  ('system.install',        'Install OS packages (winget/apt/brew)',     3, true),
  ('compute.download',      'Download files from the web into storage',  3, true)
  AS s(capability, rationale, max_validity_days, requires_second_approver)
) AS src
ON t.capability = src.capability
WHEN NOT MATCHED THEN INSERT
  (capability, rationale, max_validity_days, requires_second_approver)
  VALUES (src.capability, src.rationale, src.max_validity_days, src.requires_second_approver);

-- Supervisor -> report edges. A report is a direct report of the supervisor
-- for the purposes of AI delegation.
CREATE TABLE IF NOT EXISTS media.control.supervision (
  supervisor_email STRING        NOT NULL,
  report_email    STRING        NOT NULL,
  since           TIMESTAMP     NOT NULL DEFAULT current_timestamp(),
  until           TIMESTAMP,
  CONSTRAINT pk_supervision PRIMARY KEY (supervisor_email, report_email),
  CONSTRAINT fk_supervision_supervisor FOREIGN KEY (supervisor_email) REFERENCES media.alex.users(email),
  CONSTRAINT fk_supervision_report FOREIGN KEY (report_email) REFERENCES media.alex.users(email)
)
CLUSTER BY (supervisor_email);

-- Roles drive the knowledge access-control decision point
-- (agent/access.py). A user may hold several rows: one per team, and
-- optional per-scope grants. The effective role is the highest held.
--
--   member    -> may read up to 'internal'
--   manager   -> may read up to 'confidential', plus their team scope
--   admin     -> may read up to 'restricted', plus every team scope
--
-- No role grants access to another user's PRIVATE scope
-- ('user:<email>'). Administrators get oversight through
-- media.control.knowledge_access_log, not by reading colleagues'
-- material. `granted_scope` is the explicit, audited exception.
CREATE TABLE IF NOT EXISTS media.control.user_roles (
  user_email    STRING     NOT NULL,
  role          STRING     NOT NULL DEFAULT 'member',
  team          STRING,
  granted_scope STRING,
  clearance     STRING,
  granted_by    STRING,
  granted_at    TIMESTAMP  NOT NULL DEFAULT current_timestamp(),
  CONSTRAINT pk_user_roles PRIMARY KEY (user_email, role, team, granted_scope),
  CONSTRAINT ck_user_roles_role
    CHECK (role IN ('member', 'manager', 'admin')),
  CONSTRAINT ck_user_roles_clearance
    CHECK (clearance IS NULL OR clearance IN
           ('public', 'internal', 'confidential', 'restricted')),
  CONSTRAINT fk_user_roles_user FOREIGN KEY (user_email) REFERENCES media.alex.users(email)
)
CLUSTER BY (user_email);

-- Append-only ledger of every knowledge access decision, allow AND
-- deny. A denied read is precisely the signal a security review needs,
-- so denials are never dropped.
CREATE TABLE IF NOT EXISTS media.control.knowledge_access_log (
  ts             TIMESTAMP  NOT NULL DEFAULT current_timestamp(),
  user_email     STRING     NOT NULL,
  role           STRING,
  resource       STRING     NOT NULL,
  scope          STRING,
  classification STRING,
  allowed        BOOLEAN    NOT NULL,
  reason         STRING,
  query          STRING
)
TBLPROPERTIES (delta.appendOnly = true)
CLUSTER BY (user_email);

-- The delegation grant. Only delegable_capabilities can be inserted;
-- the row-level CHECK enforces it.
CREATE TABLE IF NOT EXISTS media.control.delegations (
  delegation_id     STRING        NOT NULL,
  supervisor_email  STRING        NOT NULL,
  report_email      STRING        NOT NULL,
  capability        STRING        NOT NULL,
  justification     STRING        NOT NULL,
  second_approver   STRING,                            -- required if delegable_capabilities.requires_second_approver
  granted_at        TIMESTAMP     NOT NULL DEFAULT current_timestamp(),
  expires_at        TIMESTAMP     NOT NULL,
  revoked           BOOLEAN       NOT NULL DEFAULT false,
  revoked_by        STRING,
  revoked_at        TIMESTAMP,
  CONSTRAINT pk_delegations PRIMARY KEY (delegation_id),
  CONSTRAINT fk_delegations_supervisor FOREIGN KEY (supervisor_email) REFERENCES media.alex.users(email),
  CONSTRAINT fk_delegations_report FOREIGN KEY (report_email) REFERENCES media.alex.users(email),
  CONSTRAINT chk_delegations_window CHECK (expires_at > granted_at)
)
CLUSTER BY (report_email, revoked);

-- A row-insert CHECK: capability must be in delegable_capabilities; if
-- requires_second_approver, second_approver must be non-null and distinct
-- from the supervisor. Implemented as a procedure (see below) because
-- CHECK constraints cannot reference other tables.
CREATE OR REPLACE PROCEDURE media.control.grant_delegation(
  IN supervisor_email STRING,
  IN report_email     STRING,
  IN capability       STRING,
  IN justification    STRING,
  IN second_approver  STRING,
  IN validity_days    INT
)
LANGUAGE SQL
SQL SECURITY INVOKER
AS BEGIN
  DECLARE requires_second BOOLEAN;
  DECLARE max_days INT;
  SELECT requires_second_approver, max_validity_days
    INTO requires_second, max_days
    FROM media.control.delegable_capabilities
   WHERE capability = capability;
  IF max_days IS NULL THEN
    SIGNAL SQLSTATE '42501' SET MESSAGE_TEXT = 'capability is not delegable';
  END IF;
  IF validity_days > max_days THEN
    SIGNAL SQLSTATE '22000' SET MESSAGE_TEXT = 'validity exceeds the cap for this capability';
  END IF;
  IF requires_second AND (second_approver IS NULL OR second_approver = supervisor_email) THEN
    SIGNAL SQLSTATE '22000' SET MESSAGE_TEXT = 'this capability requires a distinct second approver';
  END IF;
  INSERT INTO media.control.delegations
    (delegation_id, supervisor_email, report_email, capability,
     justification, second_approver, expires_at)
  VALUES
    (uuid(), supervisor_email, report_email, capability,
     justification, second_approver, current_timestamp() + INTERVAL validity_days DAYS);
END;

-- The effective employee ACL view. Active rows of `employee_acl` UNION
-- active rows of `delegations`. The agent enforces the intersection with
-- the instance_acl at tool-bind time.
CREATE OR REPLACE VIEW media.control.effective_employee_acl AS
SELECT user_email, capability, expires_at
FROM media.control.employee_acl
WHERE (expires_at IS NULL OR expires_at > current_timestamp())
UNION ALL
SELECT report_email AS user_email, capability, expires_at
FROM media.control.delegations
WHERE revoked = false
  AND expires_at > current_timestamp();

GRANT SELECT ON media.control.effective_employee_acl TO `alex-users`;
GRANT SELECT ON media.control.effective_employee_acl TO sp-alex-agent;
GRANT SELECT ON media.control.effective_employee_acl TO sp-alex-app;
GRANT SELECT ON media.control.effective_employee_acl TO sp-alex-jobs;
GRANT SELECT ON media.control.effective_employee_acl TO `alex-governors`;
GRANT EXECUTE ON PROCEDURE media.control.grant_delegation TO `alex-users`;
GRANT EXECUTE ON PROCEDURE media.control.grant_delegation TO sp-alex-app;
