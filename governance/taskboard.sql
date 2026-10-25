-- ===========================================================================
-- taskboard.sql — the shared board where Alex is a crew member.
--
-- These three tables had no DDL at all. They existed only in the dev
-- seed, so the board worked locally and every write would have failed in
-- production with "table not found" — the board, the group thread, and
-- every agent run on a task. Column-for-column with app/_dev_seed.py;
-- when one changes the other has to.
--
-- `media.alex.tasks` is not here: it is the task list itself and lives in
-- the application schema (app_schema.sql). `status` on that table stays
-- the source of truth for which lane a task is in; these tables add the
-- crew, the conversation, and the record of Alex's runs.
-- ===========================================================================

-- Who is on a task. Alex appears here as the member 'alex' with role
-- 'assignee' when a run starts, which is what makes the agent a crew
-- member rather than a feature of the board.
CREATE TABLE IF NOT EXISTS media.control.task_crew (
  task_id         STRING        NOT NULL,
  member          STRING        NOT NULL,             -- user email, or 'alex'
  role            STRING        NOT NULL DEFAULT 'member',  -- 'assignee'|'member'|'watcher'
  added_at        TIMESTAMP     NOT NULL DEFAULT current_timestamp(),
  CONSTRAINT pk_task_crew PRIMARY KEY (task_id, member)
)
CLUSTER BY (task_id)
TBLPROPERTIES (
  'delta.enableChangeDataFeed' = 'true'
);

-- The thread. Humans and Alex post to the same one on purpose: a separate
-- agent log would let the conversation about a task and the work on it
-- drift apart.
CREATE TABLE IF NOT EXISTS media.control.task_messages (
  message_id      STRING        NOT NULL,
  task_id         STRING        NOT NULL,
  author          STRING        NOT NULL,             -- user email, or 'alex'
  body            STRING,
  kind            STRING        NOT NULL DEFAULT 'human',  -- 'human'|'agent'|'system'
  ts              TIMESTAMP     NOT NULL DEFAULT current_timestamp(),
  CONSTRAINT pk_task_messages PRIMARY KEY (message_id),
  CONSTRAINT chk_task_messages_kind CHECK (kind IN ('human','agent','system'))
)
CLUSTER BY (task_id, ts)
TBLPROPERTIES (
  'delta.appendOnly' = 'true',
  'delta.enableChangeDataFeed' = 'true'
);

-- One execution of a task by Alex.
--
-- `last_heartbeat` and `steps` are what make a run recoverable. A run
-- used to live only in a process-local dictionary: if the server died
-- mid-run the work stopped, and the row sat at 'running' for ever —
-- showing in the oversight console as in-flight work that would never
-- finish, and leaving the task parked in `in_progress` with nobody
-- aware it needed doing again.
--
-- The heartbeat is what distinguishes "still working" from "the process
-- that was working is gone". Recovery cannot rely on a clean shutdown
-- hook, because the case that matters is the one where there wasn't one.
CREATE TABLE IF NOT EXISTS media.control.task_runs (
  run_id          STRING        NOT NULL,
  task_id         STRING        NOT NULL,
  session_id      STRING        NOT NULL,
  started_by      STRING        NOT NULL,
  status          STRING        NOT NULL DEFAULT 'running',  -- 'running'|'done'|'failed'|'stopped'|'interrupted'
  started_at      TIMESTAMP     NOT NULL DEFAULT current_timestamp(),
  last_heartbeat  TIMESTAMP,                          -- NULL once the run ends
  ended_at        TIMESTAMP,
  summary         STRING,
  steps           STRING,                             -- JSON array of the run's events
  CONSTRAINT pk_task_runs PRIMARY KEY (run_id),
  CONSTRAINT chk_task_runs_status CHECK (
    status IN ('running','done','failed','stopped','interrupted')
  )
)
CLUSTER BY (task_id, started_at)
TBLPROPERTIES (
  'delta.enableChangeDataFeed' = 'true'
);

-- The plan — the structured spine of a run, and the thing that makes
-- "helping an employee complete a task" inspectable rather than a prose
-- stream. Plan-then-execute with interleaved replanning: the agent
-- declares steps before acting, checks them off as it works, and may
-- replace the plan when the situation changes (a replacement is a new
-- set of rows; the thread records that it happened). The crew reads the
-- same rows the agent writes, which is what lets a person steer against
-- the plan instead of against a guess.
CREATE TABLE IF NOT EXISTS media.control.task_plans (
  task_id         STRING        NOT NULL,
  step_no         INT           NOT NULL,
  title           STRING        NOT NULL,
  status          STRING        NOT NULL DEFAULT 'pending',  -- 'pending'|'doing'|'done'|'skipped'|'blocked'
  note            STRING,
  updated_at      TIMESTAMP     NOT NULL DEFAULT current_timestamp(),
  CONSTRAINT pk_task_plans PRIMARY KEY (task_id, step_no),
  CONSTRAINT chk_task_plans_status CHECK (
    status IN ('pending','doing','done','skipped','blocked')
  )
)
CLUSTER BY (task_id)
TBLPROPERTIES (
  'delta.enableChangeDataFeed' = 'true'
);

-- Acceptance criteria — the human's definition of done, distinct from the
-- agent's plan. The asymmetry is deliberate: people WRITE the criteria
-- (board API), the agent may only MARK them met or unmet with evidence.
-- An agent that could edit its own definition of done would grade its
-- own homework. The finish gate reports any criterion not marked met in
-- the summary the reviewer reads at review->done.
CREATE TABLE IF NOT EXISTS media.control.task_acceptance (
  task_id         STRING        NOT NULL,
  crit_no         INT           NOT NULL,
  text            STRING        NOT NULL,
  status          STRING        NOT NULL DEFAULT 'open',  -- 'open'|'met'|'unmet'
  note            STRING,                                 -- the evidence, or what stands in the way
  updated_at      TIMESTAMP     NOT NULL DEFAULT current_timestamp(),
  CONSTRAINT pk_task_acceptance PRIMARY KEY (task_id, crit_no),
  CONSTRAINT chk_task_acceptance_status CHECK (status IN ('open','met','unmet'))
)
CLUSTER BY (task_id)
TBLPROPERTIES (
  'delta.enableChangeDataFeed' = 'true'
);

-- Questions — the agent asking a human and waiting for the answer
-- (elicitation). Durable on purpose: an answer given after the asking
-- run died still lands in the thread and frames the next attempt.
-- Write asymmetry, enforced at the grant layer like task_acceptance:
-- the agent may ASK (INSERT); only the app — a person acting through
-- it — may resolve (UPDATE). An agent that could write `answer` could
-- approve itself.
CREATE TABLE IF NOT EXISTS media.control.task_questions (
  question_id     STRING        NOT NULL,
  task_id         STRING        NOT NULL,
  run_id          STRING,                              -- the run that asked, if known
  asked_by        STRING        NOT NULL,              -- 'alex'
  question        STRING        NOT NULL,
  options         STRING,                              -- JSON array of short choices
  status          STRING        NOT NULL DEFAULT 'open',  -- 'open'|'answered'|'cancelled'
  answer          STRING,
  answered_by     STRING,                              -- user email
  asked_at        TIMESTAMP     NOT NULL DEFAULT current_timestamp(),
  answered_at     TIMESTAMP,
  CONSTRAINT pk_task_questions PRIMARY KEY (question_id),
  CONSTRAINT chk_task_questions_status CHECK (
    status IN ('open','answered','cancelled')
  )
)
CLUSTER BY (task_id, asked_at)
TBLPROPERTIES (
  'delta.enableChangeDataFeed' = 'true'
);

-- Dependencies: task_id is blocked until blocked_by reaches 'done'.
-- Enforced, not decorative — a run cannot start on a task with open
-- blockers, and finishing a blocker notifies the tasks it was holding.
-- Cycles are refused at insert (a cycle is a deadlock nobody can work).
CREATE TABLE IF NOT EXISTS media.control.task_deps (
  task_id         STRING        NOT NULL,
  blocked_by      STRING        NOT NULL,
  added_by        STRING        NOT NULL,
  added_at        TIMESTAMP     NOT NULL DEFAULT current_timestamp(),
  CONSTRAINT pk_task_deps PRIMARY KEY (task_id, blocked_by)
)
CLUSTER BY (task_id)
TBLPROPERTIES (
  'delta.enableChangeDataFeed' = 'true'
);

-- Task proposals — work the system NOTICED, waiting for a person to
-- adopt it. email_intel extracts asks_of_user from every email; this is
-- where those asks become candidate tasks. The two-step shape is the
-- control property: email bodies are UNTRUSTED content, so nothing from
-- an email becomes agent-runnable work until the employee reads the
-- proposal and accepts it. A dismissed proposal stays dismissed — the
-- harvest is idempotent on source_ref regardless of status.
CREATE TABLE IF NOT EXISTS media.control.task_proposals (
  proposal_id     STRING        NOT NULL,
  owner_email     STRING        NOT NULL,
  title           STRING        NOT NULL,
  description     STRING,
  priority        INT           NOT NULL DEFAULT 3,
  due_at          TIMESTAMP,
  source          STRING        NOT NULL DEFAULT 'email',
  source_ref      STRING        NOT NULL,               -- e.g. 'email:<id>#ask0'
  status          STRING        NOT NULL DEFAULT 'proposed',  -- 'proposed'|'accepted'|'dismissed'
  task_id         STRING,                               -- set when accepted
  created_at      TIMESTAMP     NOT NULL DEFAULT current_timestamp(),
  decided_at      TIMESTAMP,
  CONSTRAINT pk_task_proposals PRIMARY KEY (proposal_id),
  CONSTRAINT chk_task_proposals_status CHECK (
    status IN ('proposed','accepted','dismissed')
  )
)
CLUSTER BY (owner_email, status)
TBLPROPERTIES (
  'delta.enableChangeDataFeed' = 'true'
);
