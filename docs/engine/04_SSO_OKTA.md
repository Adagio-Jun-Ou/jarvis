# Alex 2.0 — Okta SSO + Databricks Apps Identity (verified against current docs)

This wires your company's Okta into every layer: workspace login, user/group sync, the Alex app, and the agent's per-user permissions. No password ever exists in the system.

## A. Account-level SSO with Okta (OIDC)

Okta supports both OIDC and SAML 2.0 for Databricks SSO; use OIDC (simpler, modern).

1. **Okta side**: Admin console → Applications → Create App Integration → OIDC / Web Application. Sign-in redirect URI: `https://accounts.cloud.databricks.com/oidc/consume` (the account console shows the exact value). Assign the app to your Databricks Okta groups. Note Client ID/Secret and the Okta issuer URL.
2. **Databricks side**: As account admin → account console → Settings → **Authentication** tab → Manage → "Single sign-on with my identity provider" → Continue. Enter the Okta issuer, client ID, and secret. Optionally set a Username claim if you don't want email as the Databricks username.
3. Click **Test SSO** first, then **Enable SSO**. Keep the account console open in a second browser window during testing to avoid locking yourself out.
4. **Emergency access**: once account-level SSO is on, all users including admins must sign in via SSO — so configure emergency access (up to 20 users with username+password+security key) before enforcing.
5. **Unified login** is enabled by default for accounts created after June 2023 — one SSO config covers the account console and all workspaces, and it can't be disabled. Nothing extra to do; MFA, device trust, and session policies are then enforced by Okta.

**Verify**: incognito window → workspace URL → redirected to Okta → MFA → landed in workspace.

## B. SCIM provisioning from Okta (users + groups)

Databricks is in the Okta Integration Network as a provisioning app, so Okta manages the full lifecycle: invite users, group membership, deactivation.

1. **Databricks**: account console → Security → **User provisioning** → Set up user provisioning → copy the **SCIM token** and **Account SCIM URL**. (This token only works against the SCIM API — it can't call other Databricks APIs, which limits blast radius.)
2. **Okta**: open the Databricks OIN app → Provisioning → enter the SCIM URL + token → Test API Credentials → enable "To App": Create Users, Update Attributes, Deactivate Users.
3. **Push groups**: push `alex-users`, `alex-admins`, `audience-admins` from Okta. These become the exact groups referenced by `governance/permissions.sql` — Okta group membership *is* the data permission model. Leaving the company in Okta = deprovisioned from Databricks = all Alex access gone, automatically.
4. Alternative if you prefer zero SCIM config: JIT provisioning auto-creates accounts on first SSO login — but you lose group sync and deactivation, so for an enterprise use SCIM.

**Verify**: assign a test user in Okta → they appear in account console User Management within minutes; remove them → deactivated.

## C. The Alex app: SSO + on-behalf-of-user (OBO)

Databricks Apps sit behind workspace auth automatically — an unauthenticated visit to the app URL bounces through Okta before a single byte of the app is served. Zero auth code in the frontend.

1. **Enable the preview**: workspace → your username → **Previews** → turn on **On-Behalf-Of User Authorization** (workspace-admin action; existing apps must be restarted after enabling).
2. **Declare scopes** in `databricks.yml` (already done): `serving.serving-endpoints` (query the agent as the user) and `sql` (user-scoped queries). On first visit each user sees a consent screen for exactly these scopes; admins can pre-consent org-wide.
3. **Headers the app receives per request** (see `app/app.py`):
   - `X-Forwarded-Email` / `X-Forwarded-User` — verified identity from Okta.
   - `X-Forwarded-Access-Token` — the user's **downscoped** OAuth token.
4. **Critical implementation rules** (from Databricks' own guidance):
   - **Never cache the token or connections built from it** — tokens are short-lived (~1h); read the header fresh on every request and build the client per-request. Caching causes the classic "works then 401s until redeploy" failure.
   - Use the app's own service-principal identity (`sp-alex-app`) for background/long-running work; OBO is for interactive requests only.
   - Locally the header won't exist — the app falls back to dev identity with a loud log warning.
5. **Why OBO matters here**: the agent's data tools run with the user's token, so Unity Catalog row filters and column masks apply automatically — the exec sees ad revenue, the intern doesn't, and neither the app nor the agent contains a single line of permission logic.

## D. Identity flow, end to end

```
Okta (MFA, device trust, lifecycle)
  │  OIDC                    │ SCIM: users/groups
  ▼                          ▼
Databricks unified login   Account groups (alex-users…)
  │                          │ UC GRANTs reference groups
  ▼                          ▼
Alex App (auto-SSO) ── X-Forwarded-Access-Token (per request, downscoped)
  ▼
alex-agent endpoint ── tools execute as the user → UC masks/filters apply
  ▼
media.alex.agent_actions ledger — actor = real Okta identity, always
```

Result: one place to onboard/offboard (Okta), one place to define data access (UC grants on Okta-synced groups), and every agent action attributable to a human identity.
