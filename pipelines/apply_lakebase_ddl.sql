-- ===========================================================================
-- apply_lakebase_ddl.sql — Delta-side prep for the memory system.
--
-- This file runs as a Databricks sql_task (bundle job `apply_lakebase_ddl`)
-- against a SQL warehouse, so it can ONLY contain Spark SQL. The
-- Postgres/pgvector side (`vector_memory`, `kg_edges` on the `alex-oltp`
-- Lakebase instance) lives in pipelines/lakebase_ddl.pg.sql and must be
-- applied via psql — see the runbook header in that file.
--
-- Here we make sure the Delta mirrors and queues the memory jobs write to
-- exist (same definitions as governance/memory_graph.sql; idempotent).
-- ===========================================================================

CREATE TABLE IF NOT EXISTS media.control.vector_memory_mirror (
  id              STRING        NOT NULL,
  owner_email     STRING        NOT NULL,
  kind            STRING        NOT NULL,
  content         STRING        NOT NULL,
  salience        DOUBLE        NOT NULL DEFAULT 0.5,
  source_ref      STRING,
  last_recalled   TIMESTAMP,
  created_at      TIMESTAMP     NOT NULL DEFAULT current_timestamp(),
  superseded_by   STRING,
  CONSTRAINT pk_vector_memory_mirror PRIMARY KEY (id)
)
CLUSTER BY (owner_email, superseded_by);

CREATE TABLE IF NOT EXISTS media.control.kg_mirror (
  edge_id         STRING        NOT NULL,
  owner_email     STRING        NOT NULL,
  src             STRING        NOT NULL,
  rel             STRING        NOT NULL,
  dst             STRING        NOT NULL,
  evidence        STRING,
  confidence      DOUBLE        NOT NULL DEFAULT 0.5,
  last_seen       TIMESTAMP     NOT NULL DEFAULT current_timestamp(),
  CONSTRAINT pk_kg_mirror PRIMARY KEY (edge_id)
)
CLUSTER BY (owner_email);
