"""
Identity and on-behalf-of-user (OBO) claim handling.

Production flow (Databricks Apps):
  The platform reverse-proxy injects two headers on every request:
    * `X-Forwarded-Email`        — human-readable email, for display only.
    * `X-Forwarded-Access-Token` — a signed JWT, scoped to the app's
                                   `user_api_scopes` (serving-endpoints, sql,
                                   files). The token's `sub` claim is the
                                   workspace user id; the `email` claim (or
                                   `preferred_username` / `upn`) is the user's
                                   email.

  The Databricks App MUST verify the token is signed by the platform and
  intended for this app's audience. We do that here, in a single function.
  Every other module reads `Identity` objects — never headers, never the
  raw JWT.

Dev flow:
  No platform. We accept a JWT signed with the dev secret. Tests mint a
  dev token with `mint_dev_token(email, scopes=...)` and pass it as a
  bearer token. This lets us exercise the entire OBO code path without a
  real Databricks deployment.

Production hardening (see `identity_from_headers`):
  * The client-supplied `Authorization: Bearer` header is honoured in
    dev only. In production/staging the ONLY accepted token source is
    the platform-injected `X-Forwarded-Access-Token` header.
  * Production/staging verify **RS256/ES256 against the issuer's JWKS**
    (standard OIDC discovery, `ALEX_JWKS_URL` to override). HS256 is a
    dev-only primitive and is rejected outside dev before any key is
    resolved — the shared-secret guard is gone because the secret no
    longer participates in production verification at all.

Security properties:
  * The email is derived from JWT claims, not from the unsigned
    `X-Forwarded-Email` header. A misconfigured proxy cannot impersonate.
  * The audience is checked. A token issued for another app on the same
    workspace is rejected.
  * The issuer is checked. Tokens minted by anyone other than the
    configured OIDC issuer are rejected.
  * Expiry is checked. A short-TTL cached token is not.
  * The same identity object is what rows are filtered by; SQL never sees
    a different "user_email" than the JWT authorized.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from alex.config import get_settings

# The default signing secret shipped in alex.config. Production must
# override it; identity resolution refuses to run against this value
# outside dev.
DEFAULT_DEV_SECRET = "dev-secret-do-not-use-in-prod"


class IdentityError(Exception):
    """Raised when a request's identity is missing, expired, or invalid."""


@dataclass(frozen=True)
class Identity:
    user_id: str  # The platform's stable user id (JWT `sub`)
    email: str  # The user's email (from `email` / `preferred_username`)
    scopes: tuple[str, ...] = ()
    raw_claims: dict = field(default_factory=dict, compare=False, repr=False)

    def has_scope(self, scope: str) -> bool:
        return scope in self.scopes

    def __str__(self) -> str:  # pragma: no cover (cosmetic)
        return self.email or self.user_id


def _b64url_decode(s: str | bytes) -> bytes:
    if isinstance(s, str):
        s = s.encode("ascii")
    pad = b"=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def _b64url_encode(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")


def _b64url_decode_strict(s: str) -> bytes:
    s_bytes = s.encode("ascii") if isinstance(s, str) else s
    pad = b"=" * (-len(s_bytes) % 4)
    return base64.urlsafe_b64decode(s_bytes + pad)


def _parse_unverified(token: str) -> tuple[dict, dict, bytes]:
    """Parse a JWT into (header, claims, signing_input) without verifying.
    Use only for inspection; always run verify_jwt() before trusting claims."""
    try:
        h_b64, p_b64, sig_b64 = token.split(".")
    except ValueError as e:
        raise IdentityError("Malformed token: expected three segments") from e
    try:
        header = json.loads(_b64url_decode_strict(h_b64))
        claims = json.loads(_b64url_decode_strict(p_b64))
        _b64url_decode_strict(sig_b64)
    except Exception as e:
        raise IdentityError(f"Malformed token: {e}") from e
    return header, claims, f"{h_b64}.{p_b64}".encode("ascii")


def _verify_hs256(signing_input: bytes, signature: bytes, secret: str) -> None:
    expected = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    if not hmac.compare_digest(expected, signature):
        raise IdentityError("Invalid token signature")


def verify_jwt(
    token: str, *, audience: str, issuer: str, secret: str, now: float | None = None
) -> dict:
    """Verify a JWT (HS256, dev) and return the claims.

    Production uses RS256 with the platform's JWKS. The shape is the same;
    only the verification primitive changes. We expose a single seam here.
    """
    if not token:
        raise IdentityError("Empty token")
    header, claims, signing_input = _parse_unverified(token)
    alg = header.get("alg")
    if alg != "HS256":
        # In production we'd also accept RS256/ES256. Dev is HS256 only.
        raise IdentityError(f"Unsupported alg: {alg!r}")
    _verify_hs256(signing_input, _b64url_decode_strict(token.split(".")[2]), secret)

    now = now if now is not None else time.time()
    # Clock-skew tolerance: 60s.
    skew = 60
    # The verifier is the enforcement point, not the minter: a token
    # without an exp claim is valid forever, and one leaked no-exp
    # token is permanent access. Dev's minter always sets exp; anything
    # that arrives without one is malformed or hostile.
    if "exp" not in claims:
        raise IdentityError("Token missing exp claim; refusing indefinite lifetime")
    try:
        expires = float(claims["exp"])
    except (TypeError, ValueError) as e:
        # A non-numeric exp must be a clean 401, not a TypeError that
        # surfaces as a 500 and reads as a server bug.
        raise IdentityError("Token exp claim is not a number") from e
    if expires + skew < now:
        raise IdentityError("Token expired")
    if "nbf" in claims and claims["nbf"] - skew > now:
        raise IdentityError("Token not yet valid")
    if "aud" not in claims or audience not in (
        claims["aud"] if isinstance(claims["aud"], list) else [claims["aud"]]
    ):
        raise IdentityError("Audience mismatch")
    if "iss" not in claims or claims["iss"] != issuer:
        raise IdentityError("Issuer mismatch")
    return claims


# ---------------------------------------------------------------------------
# Production verification — RS256/ES256 against the issuer's JWKS
# ---------------------------------------------------------------------------
#
# Databricks Apps inject an OBO token signed by the platform's OIDC
# issuer. Verification is standard OIDC: discover `jwks_uri` from
# `<issuer>/.well-known/openid-configuration` (ALEX_JWKS_URL overrides
# discovery for air-gapped deployments), fetch the key set, select by
# `kid`, verify RS256/ES256. HS256 is a dev-only primitive and is never
# accepted outside dev — sharing one symmetric secret with the platform
# is not a thing that exists, and accepting HS256 in production is the
# classic key-confusion attack surface.

_ASYMMETRIC_ALGS = ("RS256", "ES256")
_JWKS_CLIENTS: dict[str, Any] = {}
_OIDC_META_CACHE: dict[str, str] = {}  # issuer -> jwks_uri


def _discover_jwks_url(issuer: str) -> str:
    override = os.environ.get("ALEX_JWKS_URL", "").strip()
    if override:
        return override
    cached = _OIDC_META_CACHE.get(issuer)
    if cached:
        return cached
    import json as _json
    import urllib.request

    meta_url = issuer.rstrip("/") + "/.well-known/openid-configuration"
    try:
        with urllib.request.urlopen(meta_url, timeout=10) as resp:
            meta = _json.loads(resp.read())
    except Exception as e:  # noqa: BLE001
        raise IdentityError(
            f"Cannot discover JWKS from {meta_url}: {e}. Set ALEX_JWKS_URL "
            "to the key-set URL directly if discovery is blocked."
        ) from e
    jwks_uri = str(meta.get("jwks_uri") or "")
    if not jwks_uri:
        raise IdentityError(f"{meta_url} has no jwks_uri")
    _OIDC_META_CACHE[issuer] = jwks_uri
    return jwks_uri


def _signing_key_for(token: str, issuer: str):
    """Resolve the verification key for this token's `kid`.

    A seam on purpose: tests patch this to a local keypair so the
    verification logic is exercised without a network. The production
    implementation uses PyJWKClient, which caches the key set and
    re-fetches on an unknown `kid` — that is what makes key rotation
    work without a deploy.
    """
    from jwt import PyJWKClient

    url = _discover_jwks_url(issuer)
    client = _JWKS_CLIENTS.get(url)
    if client is None:
        client = _JWKS_CLIENTS[url] = PyJWKClient(url, cache_keys=True)
    return client.get_signing_key_from_jwt(token).key


def verify_asymmetric_jwt(
    token: str, *, audience: str, issuer: str
) -> dict:
    """Verify a platform-issued RS256/ES256 token and return its claims.

    Same required-claims posture as the dev verifier: `exp`, `aud` and
    `iss` are mandatory, with 60s leeway — a missing expiry is refused,
    not tolerated.
    """
    import jwt as pyjwt

    if not token:
        raise IdentityError("Empty token")
    try:
        header = pyjwt.get_unverified_header(token)
    except pyjwt.PyJWTError as e:
        raise IdentityError(f"Malformed token: {e}") from e
    alg = header.get("alg")
    if alg not in _ASYMMETRIC_ALGS:
        raise IdentityError(
            f"Unsupported alg {alg!r}: production accepts {_ASYMMETRIC_ALGS} only"
        )
    try:
        key = _signing_key_for(token, issuer)
        return pyjwt.decode(
            token,
            key,
            algorithms=list(_ASYMMETRIC_ALGS),
            audience=audience,
            issuer=issuer,
            leeway=60,
            options={"require": ["exp", "aud", "iss"]},
        )
    except IdentityError:
        raise
    except pyjwt.PyJWTError as e:
        # One exception type out, whatever PyJWT raised: expired,
        # bad audience, bad signature, unknown kid. The route layer
        # turns IdentityError into a clean 401.
        raise IdentityError(f"Token rejected: {e}") from e


def claims_to_identity(claims: dict) -> Identity:
    sub = str(claims.get("sub") or claims.get("user_id") or "")
    email = (
        claims.get("email")
        or claims.get("preferred_username")
        or claims.get("upn")
        or ""
    )
    if not sub or not email:
        raise IdentityError("Token missing sub/email")
    scope = claims.get("scope") or claims.get("scp") or ""
    if isinstance(scope, str):
        scopes = tuple(s for s in scope.split() if s)
    elif isinstance(scope, list):
        scopes = tuple(str(s) for s in scope)
    else:
        scopes = ()
    return Identity(user_id=sub, email=email, scopes=scopes, raw_claims=dict(claims))


def identity_from_headers(headers: dict) -> Identity:
    """Resolve an Identity from the request headers.

    In production: parse and verify the JWT in `X-Forwarded-Access-Token`
    (platform-injected). The client-supplied `Authorization: Bearer`
    header is IGNORED outside dev — it is attacker-controlled.
    In dev:        also accept the JWT in `Authorization: Bearer ...`.
    Either way the email comes from the JWT claims. The `X-Forwarded-Email`
    header is logged at info for debugging but is never used as identity.
    """
    settings = get_settings()
    token: str | None = None
    if settings.mode == "dev":
        auth = headers.get("authorization") or headers.get("Authorization") or ""
        if auth.lower().startswith("bearer "):
            token = auth[7:].strip()
    if not token:
        token = (
            headers.get("x-forwarded-access-token")
            or headers.get("X-Forwarded-Access-Token")
            or ""
        ).strip()
    if not token:
        raise IdentityError("No bearer token and no X-Forwarded-Access-Token")
    if settings.mode == "dev":
        claims = verify_jwt(
            token,
            audience=settings.jwt_audience,
            issuer=settings.jwt_issuer,
            secret=settings.jwt_dev_secret,
        )
    else:
        # Production and staging verify against the issuer's JWKS,
        # RS256/ES256 only. There is no HS256 outside dev and no shared
        # secret to configure — the old "refuse the default dev secret"
        # guard is gone because the secret no longer participates at all,
        # which is the stronger property.
        claims = verify_asymmetric_jwt(
            token,
            audience=settings.jwt_audience,
            issuer=settings.jwt_issuer,
        )
    return claims_to_identity(claims)


def mint_dev_token(
    email: str,
    *,
    scopes: Iterable[str] = (),
    user_id: str | None = None,
    ttl_seconds: int = 3600,
    secret: str | None = None,
    audience: str | None = None,
    issuer: str | None = None,
) -> str:
    """Mint an HS256 JWT for local dev/tests. NEVER use in production."""
    settings = get_settings()
    secret = secret or settings.jwt_dev_secret
    audience = audience or settings.jwt_audience
    issuer = issuer or settings.jwt_issuer
    now = int(time.time())
    payload = {
        "sub": user_id or f"local|{email}",
        "email": email,
        "aud": audience,
        "iss": issuer,
        "iat": now,
        "nbf": now,
        "exp": now + ttl_seconds,
        "scope": " ".join(scopes)
        if scopes
        else "serving.serving-endpoints sql files.files",
    }
    header = {"alg": "HS256", "typ": "JWT"}
    h = _b64url_encode(json.dumps(header, separators=(",", ":")).encode())
    p = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode())
    sig = hmac.new(secret.encode("utf-8"), f"{h}.{p}".encode(), hashlib.sha256).digest()
    return f"{h}.{p}.{_b64url_encode(sig)}"
