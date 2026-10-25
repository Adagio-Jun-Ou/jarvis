-- ===========================================================================
-- memory_graph.sql — Lakebase is the hot path (OLTP); Delta mirrors are the
-- cold store. We declare the Delta side here; the Lakebase side is created
-- by a one-time Lakeflow job that the bundle runs first.
-- ===========================================================================

-- The Delta mirror of the Lakebase vector memory. Written by the
-- memory_apply job; read by the agent endpoint for the "what does my AI
-- know about me?" transparency view.
CREATE TABLE IF NOT EXISTS media.control.vector_memory_mirror (
  id              STRING        NOT NULL,
  owner_email     STRING        NOT NULL,
  kind            STRING        NOT NULL,             -- 'preference'|'fact'|'skill_cue'|'project'
  content         STRING        NOT NULL,
  salience        DOUBLE        NOT NULL DEFAULT 0.5,
  source_ref      STRING,                              -- session_id, doc_id, etc.
  last_recalled   TIMESTAMP,
  created_at      TIMESTAMP     NOT NULL DEFAULT current_timestamp(),
  superseded_by   STRING,                              -- forget/supersede flow
  CONSTRAINT pk_vector_memory_mirror PRIMARY KEY (id)
)
CLUSTER BY (owner_email, superseded_by)
PARTITIONED BY (date(created_at));

-- The Delta mirror of the Lakebase knowledge graph.
CREATE TABLE IF NOT EXISTS media.control.kg_mirror (
  edge_id         STRING        NOT NULL,
  owner_email     STRING        NOT NULL,
  src             STRING        NOT NULL,
  rel             STRING        NOT NULL,
  dst             STRING        NOT NULL,
  evidence        STRING,                              -- quote or doc id
  confidence      DOUBLE        NOT NULL DEFAULT 0.5,
  last_seen       TIMESTAMP     NOT NULL DEFAULT current_timestamp(),
  CONSTRAINT pk_kg_mirror PRIMARY KEY (edge_id)
)
CLUSTER BY (owner_email);

-- The queue the consolidation job reads. Each row is one user's worth of
-- LLM-extracted memories + graph edges to apply.
CREATE TABLE IF NOT EXISTS media.control.consolidation_queue (
  queue_id        STRING        NOT NULL,
  owner_email     STRING        NOT NULL,
  for_date        DATE          NOT NULL,
  memories_json   STRING        NOT NULL,
  graph_json      STRING        NOT NULL,
  status          STRING        NOT NULL DEFAULT 'pending',  -- 'pending'|'applied'|'failed'
  error           STRING,
  created_at      TIMESTAMP     NOT NULL DEFAULT current_timestamp(),
  CONSTRAINT pk_consolidation_queue PRIMARY KEY (queue_id)
)
CLUSTER BY (status, for_date);

-- Forget request queue. Users INSERT (with SELECT, INSERT only); jobs
-- UPDATE the applied flag after they've superseded the memory in Lakebase
-- and updated the mirror.
CREATE TABLE IF NOT EXISTS media.control.forget_requests (
  request_id      STRING        NOT NULL,
  owner_email     STRING        NOT NULL,
  item_id         STRING        NOT NULL,             -- vector_memory_mirror.id or kg_mirror.edge_id
  reason          STRING,
  ts              TIMESTAMP     NOT NULL DEFAULT current_timestamp(),
  applied         BOOLEAN       NOT NULL DEFAULT false,
  applied_at      TIMESTAMP,
  CONSTRAINT pk_forget_requests PRIMARY KEY (request_id)
)
CLUSTER BY (owner_email, applied);
