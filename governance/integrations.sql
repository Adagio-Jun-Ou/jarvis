-- ===========================================================================
-- integrations.sql — external tool registry + browser session store +
-- browser audit. The agent reads tool_integrations at startup to know what
-- is callable; the app writes to browser_sessions and browser_history.
-- ===========================================================================

-- Seed two real integrations so the registry is non-empty after a clean
-- deploy. The seeds reference /Volumes/media/alex/api_specs/<name>.yaml
-- which the bundle provisions (see 00_preamble.sql + databricks.yml).
MERGE INTO media.control.tool_integrations AS t
USING (SELECT * FROM VALUES
  ('planhub', 'PlanHub', 'api', 'https://api.planhub.example/v1',
   'internal_api.read', 'internal_api.act', true,
   'oauth', 'https://oauth.planhub.example', array('read:plans','write:plans'),
   'vault-key-planhub-2026',
   '/Volumes/media/alex/api_specs/planhub.yaml',
   cast(null as TIMESTAMP), cast(null as TIMESTAMP)) AS s(integration_id, display_name, mode, base_url,
       read_capability, act_capability, act_requires_confirm,
       auth_kind, obo_audience, obo_scopes, vault_key_id, api_spec_path, created_at, updated_at)
) AS src
ON t.integration_id = src.integration_id
WHEN NOT MATCHED THEN INSERT
  (integration_id, display_name, mode, base_url, read_capability, act_capability,
   act_requires_confirm, auth_kind, obo_audience, obo_scopes, vault_key_id, api_spec_path)
  VALUES (src.integration_id, src.display_name, src.mode, src.base_url,
          src.read_capability, src.act_capability, src.act_requires_confirm,
          src.auth_kind, src.obo_audience, src.obo_scopes, src.vault_key_id, src.api_spec_path);

MERGE INTO media.control.tool_integrations AS t
USING (SELECT * FROM VALUES
  ('ledgerpro', 'Ledger Pro', 'api', 'https://api.ledgerpro.example/v1',
   'internal_api.read', 'internal_api.act', true,
   'oauth', 'https://oauth.ledgerpro.example', array('read:invoices','write:invoices'),
   'vault-key-ledgerpro-2026',
   '/Volumes/media/alex/api_specs/ledgerpro.yaml',
   cast(null as TIMESTAMP), cast(null as TIMESTAMP)) AS s(integration_id, display_name, mode, base_url,
       read_capability, act_capability, act_requires_confirm,
       auth_kind, obo_audience, obo_scopes, vault_key_id, api_spec_path, created_at, updated_at)
) AS src
ON t.integration_id = src.integration_id
WHEN NOT MATCHED THEN INSERT
  (integration_id, display_name, mode, base_url, read_capability, act_capability,
   act_requires_confirm, auth_kind, obo_audience, obo_scopes, vault_key_id, api_spec_path)
  VALUES (src.integration_id, src.display_name, src.mode, src.base_url,
          src.read_capability, src.act_capability, src.act_requires_confirm,
          src.auth_kind, src.obo_audience, src.obo_scopes, src.vault_key_id, src.api_spec_path);
