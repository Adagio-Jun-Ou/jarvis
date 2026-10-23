"""Tests for agent/internal_apis.py — governed read/act with two-phase confirm."""
from __future__ import annotations

import pytest

from agent import internal_apis as ia


class FakeResp:
    def __init__(self, status_code=200, text="{}", json_payload=None):
        self.status_code = status_code
        self.text = text
        self._json = json_payload or {}
    def json(self):
        return self._json


class FakeClient:
    def __init__(self):
        self.calls = []
        self.responses = []
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False
    def get(self, url, **kw):
        self.calls.append(("GET", url, kw))
        return self.responses.pop(0) if self.responses else FakeResp()
    def post(self, url, **kw):
        self.calls.append(("POST", url, kw))
        return self.responses.pop(0) if self.responses else FakeResp()


@pytest.fixture
def integration():
    return {
        "integration_id": "planhub",
        "base_url": "https://api.planhub.example/v1",
        "act_requires_confirm": True,
    }


def test_read_calls_under_base_url(integration, monkeypatch):
    fc = FakeClient()
    fc.responses = [FakeResp(text='{"plans":[]}')]
    monkeypatch.setattr(ia.httpx, "Client", lambda **kw: fc)
    tokens = []
    out = ia.api_read(integration, "/plans", user_token_provider=lambda i: tokens.append(i) or "tok")
    assert tokens == ["planhub"]
    assert "plans" in out


def test_read_rejects_path_traversal(integration, monkeypatch):
    fc = FakeClient()
    monkeypatch.setattr(ia.httpx, "Client", lambda **kw: fc)
    out = ia.api_read(integration, "../admin", user_token_provider=lambda i: "tok")
    assert "escapes" in out
    assert fc.calls == []  # never made the request


def test_read_rejects_encoded_path_traversal(integration, monkeypatch):
    """%2e%2e reaches the server as .., so the check must decode first."""
    fc = FakeClient()
    monkeypatch.setattr(ia.httpx, "Client", lambda **kw: fc)
    out = ia.api_read(
        integration, "%2e%2e/admin", user_token_provider=lambda i: "tok"
    )
    assert "escapes" in out
    assert fc.calls == []


def test_read_allows_a_deeper_path(integration, monkeypatch):
    fc = FakeClient()
    fc.responses = [FakeResp(text="{}")]
    monkeypatch.setattr(ia.httpx, "Client", lambda **kw: fc)
    ia.api_read(integration, "/plans/7", user_token_provider=lambda i: "tok")
    assert fc.calls and fc.calls[0][1].endswith("/v1/plans/7")


def test_act_stages_with_token_when_requires_confirm(integration, monkeypatch):
    fc = FakeClient()
    monkeypatch.setattr(ia.httpx, "Client", lambda **kw: fc)
    out = ia.api_act(integration, "/plans", {"name": "x"},
                     user_token_provider=lambda i: "tok")
    assert out.get("staged") is True
    assert out.get("confirmation_token")
    assert fc.calls == []  # the POST was NOT made yet


def test_act_executes_with_staged_token(integration, monkeypatch):
    fc = FakeClient()
    fc.responses = [FakeResp(status_code=200, text='{"id":1}')]
    monkeypatch.setattr(ia.httpx, "Client", lambda **kw: fc)
    body = {"name": "x"}
    staged = ia.api_act(integration, "/plans", body, user_token_provider=lambda i: "tok")
    tok = staged["confirmation_token"]
    out = ia.api_act(integration, "/plans", body,
                     user_token_provider=lambda i: "tok", confirm_token=tok)
    assert out.get("ok") is True
    assert fc.calls and fc.calls[0][0] == "POST"


def test_act_rejects_unstaged_or_arbitrary_token(integration, monkeypatch):
    fc = FakeClient()
    monkeypatch.setattr(ia.httpx, "Client", lambda **kw: fc)
    # An arbitrary long token that was never staged must NOT execute the POST
    # (this was the vulnerability: any >=16-char string was accepted).
    out = ia.api_act(integration, "/plans", {"name": "x"},
                     user_token_provider=lambda i: "tok",
                     confirm_token="this-is-a-long-enough-token-yes")
    assert "invalid or expired" in out.get("error", "")
    assert fc.calls == []


def test_act_token_is_single_use_and_body_bound(integration, monkeypatch):
    fc = FakeClient()
    fc.responses = [FakeResp(status_code=200, text='{}'), FakeResp(status_code=200, text='{}')]
    monkeypatch.setattr(ia.httpx, "Client", lambda **kw: fc)
    body = {"amount": 5}
    tok = ia.api_act(integration, "/pay", body, user_token_provider=lambda i: "t")["confirmation_token"]
    # A DIFFERENT body with the same token is rejected (binding).
    swap = ia.api_act(integration, "/pay", {"amount": 5000},
                      user_token_provider=lambda i: "t", confirm_token=tok)
    assert "invalid or expired" in swap.get("error", "")
    # The correct body executes once...
    ok = ia.api_act(integration, "/pay", body, user_token_provider=lambda i: "t", confirm_token=tok)
    assert ok.get("ok") is True
    # ...and the token cannot be reused.
    reuse = ia.api_act(integration, "/pay", body, user_token_provider=lambda i: "t", confirm_token=tok)
    assert "invalid or expired" in reuse.get("error", "")


# ---- the token is a user credential, not a model artifact -------------------
# A token returned to the model is a token the model can hand back on
# the next call — two-phase becomes one-phase with extra steps. The
# token is user-bound; the tool layer strips it from the staged result.


def test_a_token_is_bound_to_the_user_who_staged_it(integration, monkeypatch):
    fc = FakeClient()
    fc.responses = [FakeResp(status_code=200, text="{}")]
    monkeypatch.setattr(ia.httpx, "Client", lambda **kw: fc)
    body = {"amount": 5}
    tok = ia.api_act(integration, "/pay", body, user_token_provider=lambda i: "t",
                     user_email="alice@x.com")["confirmation_token"]
    # A different user presenting the same token, same body: refused.
    out = ia.api_act(integration, "/pay", body, user_token_provider=lambda i: "t",
                     user_email="bob@x.com", confirm_token=tok)
    assert "invalid or expired" in out.get("error", "")
    assert fc.calls == [], "another user's token executed the POST"


def test_a_token_without_user_binding_stays_backwards_compatible(integration, monkeypatch):
    """The direct-call path (tests, the authenticated UI route) may omit
    user_email; the binding must not then reject a valid redeem."""
    fc = FakeClient()
    fc.responses = [FakeResp(status_code=200, text="{}")]
    monkeypatch.setattr(ia.httpx, "Client", lambda **kw: fc)
    body = {"amount": 5}
    tok = ia.api_act(integration, "/pay", body, user_token_provider=lambda i: "t")[
        "confirmation_token"
    ]
    out = ia.api_act(integration, "/pay", body, user_token_provider=lambda i: "t",
                     confirm_token=tok)
    assert out.get("ok") is True


def test_the_tool_layer_never_hands_the_model_the_token(monkeypatch):
    """The staged result the model sees carries a card id, not the
    redeemable token."""
    from agent.tools import impls

    monkeypatch.setattr(
        impls, "_integration_lookup",
        lambda iid: {
            "integration_id": iid, "base_url": "https://api.example.com/v1",
            "act_requires_confirm": True,
        },
    )
    out = impls.api_act(
        "planhub", "/plans", '{"name": "x"}', user_email="alice@x.com"
    )
    assert out.get("staged") is True
    assert "confirmation_token" not in out, "the model received the credential"
    assert out.get("card_id"), "the card id is missing"
    assert "cannot confirm it yourself" in out.get("note", "")
