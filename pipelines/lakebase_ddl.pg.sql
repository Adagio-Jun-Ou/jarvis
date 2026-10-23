-- ===========================================================================
-- lakebase_ddl.pg.sql — Pure Postgres / pgvector DDL for the Lakebase
-- (OLTP) side of the memory system.
--
-- RUNBOOK: This file CANNOT be run as a Databricks sql_task (that runs
-- against a SQL warehouse, which does not speak Postgres DDL). Run it
-- against the `alex-oltp` Lakebase instance, database `alex`, via psql:
--
--   PGPASSWORD=$(databricks database generate-database-credential \
--       --instance-names alex-oltp --output json | jq -r .token) \
--   psql "host=<alex-oltp host> port=5432 dbname=alex user=<sp> sslmode=require" \
--       -f pipelines/lakebase_ddl.pg.sql
--
-- Written by pipelines/memory_apply.py, agent/vector_memory.py,
-- agent/enterprise_graph.py (kg_entities / kg_edges_v2), and
-- agent/rag.py (enterprise_docs). The Delta mirrors live in
-- governance/memory_graph.sql; the Delta-side prep the bundle CAN run is
-- pipelines/apply_lakebase_ddl.sql.
--
-- Idempotent: every CREATE uses IF NOT EXISTS.
-- ===========================================================================

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS vector_memory (
    id            UUID PRIMARY KEY,
    owner_email   TEXT NOT NULL,
    kind          TEXT NOT NULL,
    content       TEXT NOT NULL,
    salience      DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    source_ref    TEXT,
    embedding     VECTOR(1024),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_recalled TIMESTAMPTZ,
    superseded_by TEXT
);

CREATE INDEX IF NOT EXISTS vector_memory_owner_idx
    ON vector_memory (owner_email, salience DESC NULLS LAST);

CREATE INDEX IF NOT EXISTS vector_memory_embedding_idx
    ON vector_memory USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- LEGACY: the per-user graph before agent/enterprise_graph.py unified
-- the knowledge graph. No code reads or writes it any more; kept so
-- re-running this file against an existing instance never drops data.
CREATE TABLE IF NOT EXISTS kg_edges (
    edge_id     UUID PRIMARY KEY,
    owner_email TEXT NOT NULL,
    src         TEXT NOT NULL,
    rel         TEXT NOT NULL,
    dst         TEXT NOT NULL,
    evidence    TEXT,
    confidence  DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    last_seen   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS kg_edges_owner_idx ON kg_edges (owner_email);

CREATE INDEX IF NOT EXISTS kg_edges_src_idx ON kg_edges (owner_email, src);

CREATE INDEX IF NOT EXISTS kg_edges_dst_idx ON kg_edges (owner_email, dst);

-- ---------------------------------------------------------------------------
-- The chat session store (agent/session_manager.py production path).
-- One storage model, two backends: dev keeps these rows on the fake
-- warehouse under media.control.sessions/session_turns; production
-- keeps them HERE, on the OLTP side, because a chat turn is a
-- row-at-a-time write the warehouse is the wrong engine for. Column
-- lists must match session_manager's INSERT/SELECTs exactly.
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS sessions (
    session_id     TEXT PRIMARY KEY,
    owner_email    TEXT NOT NULL,
    kind           TEXT NOT NULL DEFAULT 'personal',
    title          TEXT NOT NULL DEFAULT '',
    parent_id      TEXT,
    participants   TEXT[],
    model          TEXT NOT NULL DEFAULT '',
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_active_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS sessions_owner_idx
    ON sessions (owner_email, last_active_at DESC);

CREATE TABLE IF NOT EXISTS session_turns (
    session_id TEXT NOT NULL,
    turn_no    INTEGER NOT NULL,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    ts         TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (session_id, turn_no)
);

-- ---------------------------------------------------------------------------
-- The enterprise knowledge base (agent/rag.py production path). Column
-- list must match rag._put exactly. The embedding column is untyped
-- VECTOR on purpose: the embedding model (and so the width) is an
-- endpoint choice, and a fixed width here would reject a model change
-- at the INSERT instead of at the config review. Retrieval loads the
-- scope's chunks and ranks in process (no ANN index yet — the known
-- gap; add ivfflat with a fixed width when the model is pinned).
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS enterprise_docs (
    chunk_id       TEXT PRIMARY KEY,
    doc_id         TEXT NOT NULL,
    title          TEXT NOT NULL DEFAULT '',
    text           TEXT NOT NULL,
    chunk_index    INTEGER NOT NULL DEFAULT 0,
    source_ref     TEXT,
    scope          TEXT NOT NULL,
    acl_capability TEXT,
    embedding      VECTOR,
    metadata       TEXT,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS enterprise_docs_scope_idx
    ON enterprise_docs (scope);

CREATE INDEX IF NOT EXISTS enterprise_docs_doc_idx
    ON enterprise_docs (doc_id);

-- ---------------------------------------------------------------------------
-- The ONE enterprise knowledge graph (agent/enterprise_graph.py).
-- kg_entities upserts ON CONFLICT (scope, lower(name)) — that arbiter
-- requires the unique expression index below; without it every upsert
-- fails with "no unique or exclusion constraint matching".
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS kg_entities (
    entity_id  TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    etype      TEXT NOT NULL DEFAULT 'other',
    summary    TEXT NOT NULL DEFAULT '',
    scope      TEXT NOT NULL,
    aliases    TEXT,
    embedding  VECTOR,
    source_ref TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS kg_entities_scope_name_idx
    ON kg_entities (scope, lower(name));

CREATE TABLE IF NOT EXISTS kg_edges_v2 (
    edge_id       TEXT PRIMARY KEY,
    src           TEXT NOT NULL,
    rel           TEXT NOT NULL,
    dst           TEXT NOT NULL,
    scope         TEXT NOT NULL,
    evidence      TEXT,
    confidence    DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    source_ref    TEXT,
    valid_from    TIMESTAMPTZ,
    valid_to      TIMESTAMPTZ,
    superseded_by TEXT
);

CREATE INDEX IF NOT EXISTS kg_edges_v2_scope_idx ON kg_edges_v2 (scope);

CREATE INDEX IF NOT EXISTS kg_edges_v2_src_idx ON kg_edges_v2 (scope, src);

CREATE INDEX IF NOT EXISTS kg_edges_v2_dst_idx ON kg_edges_v2 (scope, dst);
