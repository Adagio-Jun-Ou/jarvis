"""_user_token_provider must resolve real secrets in prod, not the fake token."""
import base64

import pytest


def _reset(monkeypatch, mode):
    monkeypatch.setenv("ALEX_ENV", mode)
    from alex import config, testing
    config._settings = None
    testing._fakes = None
    from agent.tools import impls
    return impls


def test_dev_uses_fake_secret(monkeypatch):
    impls = _reset(monkeypatch, "dev")
    from alex import testing
    testing.get_fakes().secrets["integration:crm"] = "real-crm-token"
    assert impls._user_token_provider("crm") == "real-crm-token"


def test_dev_defaults_when_unset(monkeypatch):
    impls = _reset(monkeypatch, "dev")
    assert impls._user_token_provider("unknown") == "dev-token"


def test_prod_resolves_from_secret_scope(monkeypatch):
    impls = _reset(monkeypatch, "production")

    class _Resp:
        value = base64.b64encode(b"prod-secret-token").decode()

    class _Secrets:
        def get_secret(self, scope, key):
            assert key == "integration_crm"
            return _Resp()

    class _WC:
        secrets = _Secrets()

    import databricks.sdk as sdk
    monkeypatch.setattr(sdk, "WorkspaceClient", lambda *a, **k: _WC())
    assert impls._user_token_provider("crm") == "prod-secret-token"


def test_prod_fails_loud_when_secret_missing(monkeypatch):
    impls = _reset(monkeypatch, "production")

    class _Secrets:
        def get_secret(self, scope, key):
            raise RuntimeError("not found")

    class _WC:
        secrets = _Secrets()

    import databricks.sdk as sdk
    monkeypatch.setattr(sdk, "WorkspaceClient", lambda *a, **k: _WC())
    with pytest.raises(RuntimeError, match="no secret configured"):
        impls._user_token_provider("crm")
