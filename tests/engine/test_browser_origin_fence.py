"""Origin fencing for the Browser OS.

Ported from the retired `agent/browser_tool.py`. The control existed
there and was never adopted by the Browser OS that replaced it, so the
live browser had no origin fence at all — the tests came back with the
behaviour.
"""

from __future__ import annotations

import pytest

from agent.browser_os import InMemoryBrowserOS, origin_ok


def test_a_lookalike_subdomain_is_not_inside_the_origin():
    """The attack this exists for: `intranet.corp.attacker.example` ends
    with `intranet.corp` as a string, so any suffix or prefix match hands
    an attacker an internal origin for the price of a domain."""
    assert not origin_ok(
        "https://intranet.corp.attacker.example/x", "https://intranet.corp/"
    )


def test_an_exact_origin_passes():
    assert origin_ok("https://intranet.corp/x", "https://intranet.corp/")


def test_a_different_scheme_is_a_different_origin():
    assert not origin_ok("http://intranet.corp/", "https://intranet.corp/")


def test_a_different_host_fails():
    assert not origin_ok("https://other.example/", "https://intranet.corp/")


def test_a_different_port_is_a_different_origin():
    assert not origin_ok("https://intranet.corp:8443/", "https://intranet.corp/")


def test_the_default_port_matches_an_implicit_one():
    assert origin_ok("https://intranet.corp:443/x", "https://intranet.corp/")


def test_no_fence_means_unfenced():
    """Most sessions legitimately browse the open web; the fence is
    opt-in per session."""
    assert origin_ok("https://anything.example/", "")


def test_an_empty_url_never_passes_a_fence():
    assert not origin_ok("", "https://intranet.corp/")


# ---- enforcement on the live navigate path -------------------------------


@pytest.fixture
def bos():
    return InMemoryBrowserOS()


def test_navigation_outside_the_fence_is_refused(bos):
    session = bos.create_session(user_email="a@x.com", instance_scope="default")
    session.allowed_origin = "alex://"
    tab = bos.create_tab(session.session_id, url="alex://welcome")
    with pytest.raises(PermissionError) as exc:
        bos.navigate(
            session.session_id, tab.tab_id,
            url="https://attacker.example/", user_email="a@x.com",
        )
    assert "fenced" in str(exc.value)
    assert any(h["outcome"] == "blocked:origin" for h in tab.history)


def test_navigation_inside_the_fence_is_allowed(bos):
    session = bos.create_session(user_email="a@x.com", instance_scope="default")
    session.allowed_origin = "alex://"
    tab = bos.create_tab(session.session_id, url="alex://welcome")
    bos.navigate(
        session.session_id, tab.tab_id, url="alex://docs", user_email="a@x.com"
    )
    assert bos.get_tab(session.session_id, tab.tab_id).url == "alex://docs"


def test_an_unfenced_session_is_unaffected(bos):
    session = bos.create_session(user_email="a@x.com", instance_scope="default")
    tab = bos.create_tab(session.session_id, url="alex://welcome")
    bos.navigate(
        session.session_id, tab.tab_id, url="alex://docs", user_email="a@x.com"
    )
    assert bos.get_tab(session.session_id, tab.tab_id).url == "alex://docs"
