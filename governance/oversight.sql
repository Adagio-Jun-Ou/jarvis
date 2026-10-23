-- ===========================================================================
-- oversight.sql — risk signals and the dual-key investigation workflow.
-- The detection job writes risk_signals; the governor console reads them
-- and opens an investigation. An investigation is "active" only when
-- opened_by != co_authorized_by; the sealed-view job (a Lakeflow task,
-- not in this file) creates per-case views from active rows.
-- ===========================================================================

CREATE TABLE IF NOT EXISTS media.control.risk_signals (
  id              STRING        NOT NULL,
  ts              TIMESTAMP     NOT NULL DEFAULT current_timestamp(),
  signal_type     STRING        NOT NULL,             -- 'constitutional_blocks'|'acl_probing'|'injection_attempt'|'exfil_shape'|'off_hours_bulk'|'capability_escalation'
  subject_email   STRING        NOT NULL,             -- the user the signal is about
  severity        STRING        NOT NULL DEFAULT 'low',  -- 'low'|'medium'|'high'
  detail          STRING,
  CONSTRAINT pk_risk_signals PRIMARY KEY (id),
  CONSTRAINT chk_risk_signals_severity CHECK (severity IN ('low','medium','high'))
)
CLUSTER BY (signal_type, ts)
TBLPROPERTIES (
  'delta.appendOnly' = 'true',
  'delta.enableChangeDataFeed' = 'true',
  'delta.deletedFileRetentionDuration' = 'interval 365 days'
)
PARTITIONED BY (date(ts));

CREATE TABLE IF NOT EXISTS media.control.investigations (
  case_id         STRING        NOT NULL,
  ts              TIMESTAMP     NOT NULL DEFAULT current_timestamp(),
  opened_by       STRING        NOT NULL,
  co_authorized_by STRING       NOT NULL,             -- structural: a governor alone cannot read memory
  subject_email   STRING        NOT NULL,
  reason          STRING        NOT NULL,
  status          STRING        NOT NULL DEFAULT 'pending_coauth',  -- 'pending_coauth'|'active'|'closed'
  closed_at       TIMESTAMP,
  CONSTRAINT pk_investigations PRIMARY KEY (case_id),
  CONSTRAINT chk_investigations_status CHECK (status IN ('pending_coauth','active','closed')),
  CONSTRAINT chk_investigations_two_key CHECK (opened_by <> co_authorized_by)
)
CLUSTER BY (status, ts)
TBLPROPERTIES ('delta.appendOnly' = 'true');

CREATE TABLE IF NOT EXISTS media.control.investigation_access_log (
  id              STRING        NOT NULL,
  case_id         STRING        NOT NULL,
  ts              TIMESTAMP     NOT NULL DEFAULT current_timestamp(),
  governor_email  STRING        NOT NULL,
  action          STRING        NOT NULL,             -- 'view'|'export'|'close'
  target          STRING,                              -- memory_id, kg_edge_id, session_id, etc.
  CONSTRAINT pk_investigation_access_log PRIMARY KEY (id)
)
CLUSTER BY (case_id, ts)
TBLPROPERTIES ('delta.appendOnly' = 'true');

-- A procedure that opens an investigation. The DB-level CHECK enforces
-- opened_by != co_authorized_by, so a single governor cannot activate
-- the case alone.
CREATE OR REPLACE PROCEDURE media.control.open_investigation(
  IN opener_email  STRING,
  IN co_auth_email STRING,
  IN subject_email STRING,
  IN reason        STRING
)
LANGUAGE SQL
SQL SECURITY INVOKER
AS BEGIN
  IF opener_email = co_auth_email THEN
    SIGNAL SQLSTATE '22000' SET MESSAGE_TEXT = 'two-key required: opener and co-author must differ';
  END IF;
  -- Opening a case does NOT activate it. The named co-author must
  -- separately call co_authorize_investigation. Inserting 'active' here
  -- would let a single governor unilaterally activate a case by naming a
  -- co-author who never consented — defeating the two-key control.
  INSERT INTO media.control.investigations
    (case_id, opened_by, co_authorized_by, subject_email, reason, status)
  VALUES (uuid(), opener_email, co_auth_email, subject_email, reason, 'pending_coauth');
END;

CREATE OR REPLACE PROCEDURE media.control.co_authorize_investigation(
  IN in_co_auth_email STRING,
  IN in_case_id       STRING
)
LANGUAGE SQL
SQL SECURITY INVOKER
AS BEGIN
  -- Parameters are prefixed in_ so they cannot shadow the column names in
  -- the WHERE clause. Matching a column against an identically-named
  -- parameter is always-true and would activate (and log) EVERY pending
  -- case at once.
  UPDATE media.control.investigations
     SET status = 'active'
   WHERE case_id = in_case_id
     AND co_authorized_by = in_co_auth_email
     AND opened_by <> in_co_auth_email
     AND status = 'pending_coauth';
  IF NOT FOUND THEN
    SIGNAL SQLSTATE '22000' SET MESSAGE_TEXT = 'case not found or not yours to authorize';
  END IF;
  INSERT INTO media.control.investigation_access_log
    (id, case_id, governor_email, action) VALUES (uuid(), in_case_id, in_co_auth_email, 'co_authorize');
END;

GRANT EXECUTE ON PROCEDURE media.control.open_investigation     TO `alex-governors`;
GRANT EXECUTE ON PROCEDURE media.control.co_authorize_investigation TO `alex-governors`;
