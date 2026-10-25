"""End-to-end tests for the Browser OS backend.

Exercises the :class:`InMemoryBrowserOS` against the constitution's
two-phase submit, mailto/tel refusal, and the user-control hand-back
loop.  The full Playwright backend is not loaded; the dev backend
covers the same protocol surface.
"""
from __future__ import annotations

import pytest

from agent.browser_os import InMemoryBrowserOS


@pytest.fixture
def bos() -> InMemoryBrowserOS:
    return InMemoryBrowserOS()


def test_create_session_and_tabs(bos: InMemoryBrowserOS) -> None:
    s = bos.create_session(user_email="alice@x.com", instance_scope="default")
    assert s.session_id.startswith("bsess_")
    assert bos.list_sessions("alice@x.com") == [s]
    tab = bos.create_tab(s.session_id, url="alex://welcome")
    assert tab.url == "alex://welcome"
    assert tab.title == "Welcome to Alex"
    assert bos.list_tabs(s.session_id) == [tab]


def test_navigate_known_fixture_loads_html(bos: InMemoryBrowserOS) -> None:
    s = bos.create_session(user_email="alice@x.com", instance_scope="default")
    t = bos.create_tab(s.session_id)
    bos.navigate(s.session_id, t.tab_id, url="alex://welcome", user_email="alice@x.com")
    tab = bos.get_tab(s.session_id, t.tab_id)
    assert "Welcome to Alex" in tab.html
    assert tab.last_status == 200


def test_navigate_unknown_url_404s(bos: InMemoryBrowserOS) -> None:
    s = bos.create_session(user_email="alice@x.com", instance_scope="default")
    t = bos.create_tab(s.session_id)
    bos.navigate(s.session_id, t.tab_id, url="http://nope.invalid", user_email="alice@x.com")
    tab = bos.get_tab(s.session_id, t.tab_id)
    assert tab.last_status == 404
    assert "no fixture" in (tab.last_error or "")


def test_click_follows_link(bos: InMemoryBrowserOS) -> None:
    s = bos.create_session(user_email="alice@x.com", instance_scope="default")
    t = bos.create_tab(s.session_id, url="alex://welcome")
    bos.click(s.session_id, t.tab_id, selector="Docs", user_email="alice@x.com")
    tab = bos.get_tab(s.session_id, t.tab_id)
    assert tab.url == "alex://docs"
    assert tab.title == "Alex Docs"


def test_click_mailto_is_refused(bos: InMemoryBrowserOS) -> None:
    s = bos.create_session(user_email="alice@x.com", instance_scope="default")
    t = bos.create_tab(s.session_id, url="alex://welcome")
    with pytest.raises(PermissionError) as exc:
        bos.click(
            s.session_id,
            t.tab_id,
            selector="Email support",
            user_email="alice@x.com",
        )
    assert "mailto" in str(exc.value)
    # Audit row recorded.
    assert any(h["outcome"].startswith("blocked:") for h in t.history)


def test_navigate_mailto_is_refused(bos: InMemoryBrowserOS) -> None:
    s = bos.create_session(user_email="alice@x.com", instance_scope="default")
    t = bos.create_tab(s.session_id, url="alex://welcome")
    with pytest.raises(PermissionError) as exc:
        bos.navigate(
            s.session_id, t.tab_id, url="mailto:x@y.com", user_email="alice@x.com"
        )
    assert "mailto" in str(exc.value)
    assert any(h["outcome"] == "blocked:mailto" for h in t.history)


def test_navigate_tel_is_refused(bos: InMemoryBrowserOS) -> None:
    s = bos.create_session(user_email="alice@x.com", instance_scope="default")
    t = bos.create_tab(s.session_id, url="alex://welcome")
    with pytest.raises(PermissionError):
        bos.navigate(
            s.session_id, t.tab_id, url="tel:+15551234", user_email="alice@x.com"
        )


def test_type_into_input_stages_form_value(bos: InMemoryBrowserOS) -> None:
    s = bos.create_session(user_email="alice@x.com", instance_scope="default")
    t = bos.create_tab(s.session_id, url="alex://welcome")
    bos.type_text(
        s.session_id,
        t.tab_id,
        selector="email",
        text="alice@x.com",
        user_email="alice@x.com",
    )
    tab = bos.get_tab(s.session_id, t.tab_id)
    assert tab.form["email"] == "alice@x.com"


def test_extract_redacts_pii(bos: InMemoryBrowserOS) -> None:
    s = bos.create_session(user_email="alice@x.com", instance_scope="default")
    t = bos.create_tab(s.session_id, url="alex://welcome")
    # Mutate the page to include a PII string.
    t.html = t.html + "<p>Email: bob@secret.com, Phone: 555-123-4567, SSN: 123-45-6789</p>"
    payload = bos.extract(s.session_id, t.tab_id, selector="body", user_email="alice@x.com")
    assert "[REDACTED:email]" in payload["text"]
    assert "[REDACTED:phone]" in payload["text"]
    assert "[REDACTED:ssn]" in payload["text"]
    assert payload["pii_redacted"] is True


def test_user_control_take_wheel_blocks_agent_ops(bos: InMemoryBrowserOS) -> None:
    s = bos.create_session(user_email="alice@x.com", instance_scope="default")
    t = bos.create_tab(s.session_id, url="alex://welcome")
    bos.set_user_control(s.session_id, t.tab_id, mode="co-drive", user_email="alice@x.com")
    tab = bos.get_tab(s.session_id, t.tab_id)
    assert tab.agent_piloting is False
    # Now the agent's navigate is blocked.
    with pytest.raises(PermissionError):
        bos.navigate(
            s.session_id,
            t.tab_id,
            url="alex://docs",
            user_email="alice@x.com",
        )
    # User yields back.
    bos.set_user_control(s.session_id, t.tab_id, mode="yield", user_email="alice@x.com")
    tab = bos.get_tab(s.session_id, t.tab_id)
    assert tab.agent_piloting is True


def test_submit_form_is_two_phase(bos: InMemoryBrowserOS) -> None:
    s = bos.create_session(user_email="alice@x.com", instance_scope="default")
    t = bos.create_tab(s.session_id, url="alex://welcome")
    bos.type_text(
        s.session_id,
        t.tab_id,
        selector="email",
        text="alice@x.com",
        user_email="alice@x.com",
    )
    bos.submit_form(s.session_id, t.tab_id, selector="form", user_email="alice@x.com")
    tab = bos.get_tab(s.session_id, t.tab_id)
    assert tab.form["_pending_submit_selector"] == "form"
    # The last audit row should be `awaiting_confirmation`.
    assert tab.history[-1]["outcome"] == "awaiting_confirmation"


def test_export_history_is_flat_record_per_op(bos: InMemoryBrowserOS) -> None:
    s = bos.create_session(user_email="alice@x.com", instance_scope="default")
    t = bos.create_tab(s.session_id, url="alex://welcome")
    # Navigate to a tab that has a form, then type.
    bos.navigate(s.session_id, t.tab_id, url="alex://welcome", user_email="alice@x.com")
    bos.type_text(
        s.session_id,
        t.tab_id,
        selector="email",
        text="alice@x.com",
        user_email="alice@x.com",
    )
    flat = bos.export_history(s.session_id)
    ops = [h["op"] for h in flat]
    assert "navigate" in ops
    assert "type" in ops


def test_close_session_removes_it(bos: InMemoryBrowserOS) -> None:
    s = bos.create_session(user_email="alice@x.com", instance_scope="default")
    bos.close_session(s.session_id)
    assert bos.list_sessions("alice@x.com") == []


# ---- backend selection ----------------------------------------------------
# The Playwright backend existed, was documented as "the only backend in
# production", and nothing ever constructed it: three call sites each
# hardcoded InMemoryBrowserOS, so production would have browsed an
# HTML-string simulator while reporting success. browser_os_for_mode is
# now the only place that decides.


def test_dev_mode_gets_the_simulator(monkeypatch):
    monkeypatch.setenv("ALEX_ENV", "dev")
    from alex import config

    config._settings = None
    from agent.browser_os import InMemoryBrowserOS, browser_os_for_mode

    try:
        assert isinstance(browser_os_for_mode(), InMemoryBrowserOS)
    finally:
        config._settings = None


def test_production_mode_refuses_to_simulate(monkeypatch):
    """Production constructs the Playwright backend — and if Playwright
    is unavailable it must raise, not quietly hand back the simulator.
    A browser that pretends is worse than one that fails."""
    monkeypatch.setenv("ALEX_ENV", "production")
    from alex import config

    config._settings = None
    import agent.browser_os as bos_mod

    called = {}

    def fake_prod():
        called["prod"] = True
        raise ImportError("playwright not installed (expected in this test)")

    monkeypatch.setattr(bos_mod, "make_production_browser_os", fake_prod)
    try:
        with pytest.raises(ImportError):
            bos_mod.browser_os_for_mode()
        assert called.get("prod"), "production did not even try Playwright"
    finally:
        config._settings = None
