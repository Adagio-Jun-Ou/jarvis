-- ===========================================================================
-- Alex — DDL preamble (idempotent, parameter-binding friendly).
-- Every statement below uses `CREATE ... IF NOT EXISTS` and a deterministic
-- column list. Inserts use `MERGE` so re-running is a no-op.
-- Tables follow the Unity Catalog pattern:
--   <catalog>.<schema>.<table>
-- Default catalog/schema are bound by the bundle target (dev/staging/prod).
-- ===========================================================================

-- Catalog creation is intentionally NOT in this file; the bundle creates it
-- once and grants ownership to the platform group. SQL files in this folder
-- assume the catalog exists.

-- All schemas are created up-front. Order matters only when objects in one
-- schema reference objects in another; that never happens here.
CREATE SCHEMA IF NOT EXISTS media.alex;
CREATE SCHEMA IF NOT EXISTS media.control;
CREATE SCHEMA IF NOT EXISTS media.content;
CREATE SCHEMA IF NOT EXISTS media.audience;
CREATE SCHEMA IF NOT EXISTS media.ops;

-- The volumes we write into. Names match the rest of the system; failures to
-- pre-create a volume surface immediately on bundle deploy rather than at
-- first read.
CREATE VOLUME IF NOT EXISTS media.alex.workspaces;
CREATE VOLUME IF NOT EXISTS media.alex.libraries;
CREATE VOLUME IF NOT EXISTS media.alex.browser_evidence;
CREATE VOLUME IF NOT EXISTS media.alex.api_specs;
CREATE VOLUME IF NOT EXISTS media.content.raw_files;
CREATE VOLUME IF NOT EXISTS media.content.transcripts_blob;
