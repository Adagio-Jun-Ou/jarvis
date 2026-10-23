"""Tests for identity, JWT, and OBO claim resolution."""
from __future__ import annotations

import pytest

from alex.identity import (
    IdentityError,
    claims_to_identity,
    identity_from_headers,
    mint_dev_token,
    verify_jwt,
)


def test_mint_and_verify_roundtrip():
    tok = mint_dev_token("alice@corp.com", scopes=("serving.serving-endpoints", "sql"))
    claims = verify_jwt(tok, audience="alex-workspace", issuer="https://databricks.local/oidc",
                        secret="dev-secret-do-not-use-in-prod")
    assert claims["email"] == "alice@corp.com"
    assert claims["aud"] == "alex-workspace"


def test_audience_mismatch_rejected():
    tok = mint_dev_token("alice@corp.com")
    with pytest.raises(IdentityError, match="Audience"):
        verify_jwt(tok, audience="other-app", issuer="https://databricks.local/oidc",
                   secret="dev-secret-do-not-use-in-prod")


def test_issuer_mismatch_rejected():
    tok = mint_dev_token("alice@corp.com")
    with pytest.raises(IdentityError, match="Issuer"):
        verify_jwt(tok, audience="alex-workspace", issuer="https://attacker.example/oidc",
                   secret="dev-secret-do-not-use-in-prod")


def test_expired_rejected():
    tok = mint_dev_token("alice@corp.com", ttl_seconds=-120)
    with pytest.raises(IdentityError, match="expired"):
        verify_jwt(tok, audience="alex-workspace", issuer="https://databricks.local/oidc",
                   secret="dev-secret-do-not-use-in-prod")


def test_tampered_signature_rejected():
    tok = mint_dev_token("alice@corp.com")
    h, p, sig = tok.split(".")
    bad = ".".join([h, p, sig[:-4] + "AAAA"])
    with pytest.raises(IdentityError, match="signature"):
        verify_jwt(bad, audience="alex-workspace", issuer="https://databricks.local/oidc",
                   secret="dev-secret-do-not-use-in-prod")


def test_claims_to_identity():
    idn = claims_to_identity({
        "sub": "u-1", "email": "b@x.com", "scope": "serving.serving-endpoints sql",
    })
    assert idn.user_id == "u-1"
    assert idn.email == "b@x.com"
    assert idn.has_scope("serving.serving-endpoints")
    assert idn.has_scope("sql")
    assert not idn.has_scope("files.files")


def test_identity_from_headers_via_bearer():
    tok = mint_dev_token("alice@corp.com", scopes=("serving.serving-endpoints", "sql"))
    idn = identity_from_headers({"Authorization": f"Bearer {tok}"})
    assert idn.email == "alice@corp.com"


def test_identity_from_headers_via_x_forwarded():
    tok = mint_dev_token("alice@corp.com")
    idn = identity_from_headers({"X-Forwarded-Access-Token": tok})
    assert idn.email == "alice@corp.com"


def test_identity_rejects_unverified_email_header():
    """The X-Forwarded-Email header must NOT be used as identity."""
    tok = mint_dev_token("alice@corp.com")
    # The email header claims CEO; the JWT says alice. Identity must be alice.
    idn = identity_from_headers({
        "X-Forwarded-Access-Token": tok,
        "X-Forwarded-Email": "ceo@corp.com",
    })
    assert idn.email == "alice@corp.com", "must derive identity from JWT, not unsigned header"


def test_identity_missing_token_raises():
    with pytest.raises(IdentityError):
        identity_from_headers({})


# --- production-mode hardening -------------------------------------------


@pytest.fixture
def prod_settings():
    """Switch settings to production with a non-default secret; restore after."""
    from alex import config

    config.reset_settings(mode="production", jwt_dev_secret="a-real-prod-secret")
    yield
    config._settings = None


def test_production_refuses_hs256_outright():
    """The symmetric primitive does not exist outside dev — regardless of
    the secret. Accepting HS256 in production is the classic
    key-confusion surface, so the alg gate comes before everything."""
    from alex import config

    config.reset_settings(mode="production")
    try:
        tok = mint_dev_token("alice@corp.com", secret="a-real-prod-secret")
        with pytest.raises(IdentityError, match="RS256"):
            identity_from_headers({"X-Forwarded-Access-Token": tok})
    finally:
        config._settings = None


def test_production_ignores_client_bearer_header(prod_settings):
    tok = mint_dev_token("alice@corp.com", secret="a-real-prod-secret")
    # A valid token in Authorization: Bearer is NOT accepted in production.
    with pytest.raises(IdentityError, match="No bearer token"):
        identity_from_headers({"Authorization": f"Bearer {tok}"})


def test_production_accepts_a_platform_signed_rs256_token(
    prod_settings, monkeypatch
):
    """The real production path: an RS256 token signed by the issuer's
    key, resolved through the JWKS seam, verified, and turned into the
    identity every downstream row filter uses."""
    from alex import identity as idmod

    private, public, token = _rs256_token(
        email="alice@corp.com",
        audience="alex-workspace",
        issuer=idmod.get_settings().jwt_issuer,
    )
    monkeypatch.setattr(idmod, "_signing_key_for", lambda t, i: public)

    idn = identity_from_headers({"X-Forwarded-Access-Token": token})
    assert idn.email == "alice@corp.com"


# ---- expiry is mandatory --------------------------------------------------
# A token without exp used to pass every check and never expire — the
# exact credential an attacker wants to hold. Both real issuers (the dev
# minter and the platform's OIDC) always set exp, so requiring it costs
# nothing legitimate.


def _mint_raw(payload: dict, secret: str) -> str:
    import base64
    import hashlib
    import hmac
    import json

    def b64(b: bytes) -> str:
        return base64.urlsafe_b64encode(b).rstrip(b"=").decode()

    h = b64(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    p = b64(json.dumps(payload).encode())
    sig = hmac.new(secret.encode(), f"{h}.{p}".encode(), hashlib.sha256).digest()
    return f"{h}.{p}.{b64(sig)}"


def test_a_token_without_exp_is_rejected():
    from alex.config import get_settings
    from alex.identity import IdentityError, verify_jwt

    s = get_settings()
    token = _mint_raw(
        {"sub": "u1", "email": "a@b.c", "aud": s.jwt_audience, "iss": s.jwt_issuer},
        s.jwt_dev_secret,
    )
    with pytest.raises(IdentityError, match="exp"):
        verify_jwt(token, audience=s.jwt_audience, issuer=s.jwt_issuer,
                   secret=s.jwt_dev_secret)


def test_a_non_numeric_exp_is_a_clean_401_not_a_typeerror():
    from alex.config import get_settings
    from alex.identity import IdentityError, verify_jwt

    s = get_settings()
    token = _mint_raw(
        {"sub": "u1", "email": "a@b.c", "aud": s.jwt_audience,
         "iss": s.jwt_issuer, "exp": "never"},
        s.jwt_dev_secret,
    )
    with pytest.raises(IdentityError, match="not a number"):
        verify_jwt(token, audience=s.jwt_audience, issuer=s.jwt_issuer,
                   secret=s.jwt_dev_secret)


# ---- RS256 / JWKS: the real production primitive --------------------------
# HS256 never leaves dev. Production verifies against the issuer's JWKS
# with RS256/ES256 — standard OIDC, which is what the Databricks platform
# actually signs OBO tokens with. These tests exercise the verification
# logic through the key-resolution seam with a real local keypair, so
# everything except the network fetch is the production code path.


def _rs256_token(*, email: str, audience: str, issuer: str,
                 exp_delta: int = 3600, drop_exp: bool = False,
                 alg: str = "RS256"):
    import time as _time

    import jwt as pyjwt
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    payload = {
        "sub": f"platform|{email}", "email": email,
        "aud": audience, "iss": issuer,
        "exp": int(_time.time()) + exp_delta,
    }
    if drop_exp:
        payload.pop("exp")
    token = pyjwt.encode(payload, key, algorithm=alg,
                         headers={"kid": "test-key-1"})
    return key, key.public_key(), token


@pytest.fixture
def rs_env(prod_settings, monkeypatch):
    """Production settings plus a helper that pins the JWKS seam to a
    given public key."""
    from alex import identity as idmod

    def pin(public_key):
        monkeypatch.setattr(idmod, "_signing_key_for", lambda t, i: public_key)

    return pin


def test_rs256_wrong_audience_is_rejected(rs_env):
    from alex.identity import get_settings, verify_asymmetric_jwt

    _, pub, tok = _rs256_token(email="a@b.c", audience="some-other-app",
                               issuer=get_settings().jwt_issuer)
    rs_env(pub)
    with pytest.raises(IdentityError, match="[Aa]udience"):
        verify_asymmetric_jwt(tok, audience="alex-workspace",
                              issuer=get_settings().jwt_issuer)


def test_rs256_wrong_issuer_is_rejected(rs_env):
    from alex.identity import verify_asymmetric_jwt

    _, pub, tok = _rs256_token(email="a@b.c", audience="alex-workspace",
                               issuer="https://evil.example/oidc")
    rs_env(pub)
    with pytest.raises(IdentityError, match="[Ii]ssuer"):
        verify_asymmetric_jwt(tok, audience="alex-workspace",
                              issuer="https://databricks.local/oidc")


def test_rs256_expired_token_is_rejected(rs_env):
    from alex.identity import get_settings, verify_asymmetric_jwt

    _, pub, tok = _rs256_token(email="a@b.c", audience="alex-workspace",
                               issuer=get_settings().jwt_issuer,
                               exp_delta=-3600)
    rs_env(pub)
    with pytest.raises(IdentityError, match="[Ee]xpire|[Ss]ignature|reject"):
        verify_asymmetric_jwt(tok, audience="alex-workspace",
                              issuer=get_settings().jwt_issuer)


def test_rs256_token_without_exp_is_rejected(rs_env):
    """Same posture as dev: expiry is mandatory, not merely checked when
    present."""
    from alex.identity import get_settings, verify_asymmetric_jwt

    _, pub, tok = _rs256_token(email="a@b.c", audience="alex-workspace",
                               issuer=get_settings().jwt_issuer, drop_exp=True)
    rs_env(pub)
    with pytest.raises(IdentityError):
        verify_asymmetric_jwt(tok, audience="alex-workspace",
                              issuer=get_settings().jwt_issuer)


def test_rs256_wrong_keys_signature_is_rejected(rs_env):
    """A token signed by anyone else's key fails, however well-formed."""
    from cryptography.hazmat.primitives.asymmetric import rsa

    from alex.identity import get_settings, verify_asymmetric_jwt

    _, _theirs, tok = _rs256_token(email="a@b.c", audience="alex-workspace",
                                   issuer=get_settings().jwt_issuer)
    unrelated = rsa.generate_private_key(
        public_exponent=65537, key_size=2048
    ).public_key()
    rs_env(unrelated)
    with pytest.raises(IdentityError, match="[Ss]ignature|reject"):
        verify_asymmetric_jwt(tok, audience="alex-workspace",
                              issuer=get_settings().jwt_issuer)


def test_alg_confusion_hs256_token_never_reaches_key_resolution(rs_env, monkeypatch):
    """The classic attack: an HS256 token 'signed' with the public key as
    the HMAC secret. The alg gate rejects it before any key is resolved,
    so the confusion has nothing to confuse."""
    from alex import identity as idmod
    from alex.identity import get_settings, verify_asymmetric_jwt

    resolved = {"n": 0}

    def counting(t, i):
        resolved["n"] += 1
        raise AssertionError("key resolution must not be reached")

    monkeypatch.setattr(idmod, "_signing_key_for", counting)
    hs = mint_dev_token("a@b.c", secret="pretend-public-key-pem")
    with pytest.raises(IdentityError, match="RS256"):
        verify_asymmetric_jwt(hs, audience="alex-workspace",
                              issuer=get_settings().jwt_issuer)
    assert resolved["n"] == 0


def test_jwks_url_env_override_wins_over_discovery(prod_settings, monkeypatch):
    from alex import identity as idmod

    monkeypatch.setenv("ALEX_JWKS_URL", "https://keys.internal/jwks.json")
    assert idmod._discover_jwks_url("https://any-issuer") == (
        "https://keys.internal/jwks.json"
    )


def test_unreachable_discovery_is_a_clean_identity_error(prod_settings, monkeypatch):
    """A network failure during discovery must be a 401-shaped refusal
    that names the fix, not a raw URLError surfacing as a 500."""
    from alex import identity as idmod

    monkeypatch.delenv("ALEX_JWKS_URL", raising=False)
    idmod._OIDC_META_CACHE.clear()
    with pytest.raises(IdentityError, match="ALEX_JWKS_URL"):
        idmod._discover_jwks_url("https://unreachable.invalid")
