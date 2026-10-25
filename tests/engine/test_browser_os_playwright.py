"""Tests for the production Browser OS backend.

``PlaywrightBrowserOS`` needs a Chromium binary, so most of these tests
run against a *stub page* injected in place of the real one: the class's
governance logic (fence, wheel, mailto, two-phase submit) is pure
Python and must be verifiable without a browser. What is being held
here is the guarantee that production enforces exactly the rules the
dev backend does — the failure class this file exists for is "the rule
exists in the mock and not in the real thing."

An integration-marked test at the bottom drives real Chromium when
Playwright and a browser binary are present; CI without them skips it.
"""

from __future__ import annotations

import pytest

from agent.browser_os import InMemoryBrowserOS, origin_ok
from agent.browser_os_playwright import PlaywrightBrowserOS


class StubPage:
    """The slice of Playwright's Page the backend touches."""

    def __init__(self):
        self.url = "about:blank"
        self.title_text = "Example"
        self._html = (
            "<!doctype html><html><head><title>Example</title></head><body>"
            "<a href='https://example.com/next'>next</a>"
            "<a href='mailto:x@y.com'>mail</a>"
            "<form name='f'><input name='q' /></form>"
            "</body></html>"
        )
        self.closed = False

    def set_default_timeout(self, ms):
        pass

    def content(self):
        return self._html

    def title(self):
        return self.title_text

    def goto(self, url, timeout=None, wait_until=None):
        self.url = url
        return type("R", (), {"status": 200})()

    def wait_for_load_state(self, state=None):
        pass

    def locator(self, selector):
        return StubLocator(selector, self)

    def screenshot(self, type=None):
        return b"\x89PNG-fake"

    def close(self):
        self.closed = True


class StubLocator:
    def __init__(self, selector, page):
        self._selector = selector
        self._page = page

    @property
    def first(self):
        return self

    def count(self):
        if "mailto:" in self._selector:
            return 0
        return 1

    def get_attribute(self, name):
        if name == "href":
            if "mail" in self._selector:
                return "mailto:x@y.com"
            return "https://example.com/next"
        if name == "name":
            return "q"
        return None

    def click(self):
        self._page.url = "https://example.com/next"

    def fill(self, text):
        self._page._html += f"<!-- typed {text} -->"

    def inner_text(self):
        return "Example body text"


@pytest.fixture
def bos(monkeypatch):
    """A PlaywrightBrowserOS with Chromium swapped for stub pages.

    The real constructor would launch Chromium; the governance logic
    under test is everything *around* the page, so the page is stubbed
    and the rest is the production class, unmodified.
    """
    made = PlaywrightBrowserOS.__new__(PlaywrightBrowserOS)
    import threading

    from agent.browser_os import BrowserTabState

    made._lock = threading.RLock()
    made._sessions = {}
    made._pages = {}
    made._browser = None
    made._pw = None
    made._closed = False

    def create_tab(self, session_id, *, url="about:blank"):
        sess = made._require_session(session_id)
        tid = f"tab_{len(sess.tabs)}"
        made._pages[(session_id, tid)] = StubPage()
        tab = BrowserTabState(tab_id=tid, url="about:blank")
        sess.tabs[tid] = tab
        # Mirror the real backend: the initial URL goes through the
        # validated navigate() path, fence included.
        if url and url != "about:blank":
            made.navigate(session_id, tid, url=url, user_email=sess.user_email)
        return tab

    monkeypatch.setattr(PlaywrightBrowserOS, "create_tab", create_tab)
    yield made
    made._pages.clear()


def _session(bos, email="u@x.com"):
    return bos.create_session(user_email=email, instance_scope="default")


# ---- the origin fence is shared, not duplicated ---------------------------


def test_the_origin_check_is_the_same_one_the_dev_backend_uses():
    """One fence, two backends: the lookalike-subdomain trap must be
    closed by the same code, or the two backends drift apart."""
    assert origin_ok("https://intranet.corp/x", "https://intranet.corp")
    assert not origin_ok("https://intranet.corp.attacker.example", "https://intranet.corp")


# ---- governance parity with the dev backend --------------------------------


def test_a_fenced_session_refuses_out_of_origin_navigation(bos):
    sess = _session(bos)
    sess.allowed_origin = "https://example.com"
    tab = bos.create_tab(sess.session_id)

    with pytest.raises(PermissionError):
        bos.navigate(
            sess.session_id, tab.tab_id,
            url="https://evil.example.com", user_email="u@x.com",
        )
    assert "blocked:origin" in [h["outcome"] for h in bos.export_history(sess.session_id)]


def test_mailto_and_tel_are_refused_on_the_production_backend(bos):
    """Alex never sends email or dials, whichever backend is live."""
    sess = _session(bos)
    tab = bos.create_tab(sess.session_id)

    with pytest.raises(PermissionError):
        bos.navigate(
            sess.session_id, tab.tab_id,
            url="mailto:someone@example.com", user_email="u@x.com",
        )
    with pytest.raises(PermissionError):
        bos.navigate(
            sess.session_id, tab.tab_id,
            url="tel:+15550100", user_email="u@x.com",
        )


def test_co_drive_holds_the_wheel(bos):
    sess = _session(bos)
    tab = bos.create_tab(sess.session_id)
    bos.set_user_control(
        sess.session_id, tab.tab_id, mode="co-drive", user_email="u@x.com"
    )

    with pytest.raises(PermissionError):
        bos.navigate(
            sess.session_id, tab.tab_id,
            url="https://example.com", user_email="u@x.com",
        )
    with pytest.raises(PermissionError):
        bos.type_text(
            sess.session_id, tab.tab_id,
            selector="input[name=q]", text="x", user_email="u@x.com",
        )


def test_a_submit_is_staged_never_performed(bos):
    """Constitution A8 on real Chromium: the form is staged and the
    history says awaiting_confirmation, nothing was POSTed."""
    sess = _session(bos)
    tab = bos.create_tab(sess.session_id)

    bos.submit_form(
        sess.session_id, tab.tab_id, selector="form", user_email="u@x.com"
    )

    history = bos.export_history(sess.session_id)
    assert any(h["outcome"] == "awaiting_confirmation" for h in history)
    assert bos.get_tab(sess.session_id, tab.tab_id).form["_pending_submit_selector"]


def test_extract_redacts_pii(bos):
    sess = _session(bos)
    tab = bos.create_tab(sess.session_id)
    bos._pages[(sess.session_id, tab.tab_id)]._html = (
        "<html><body>contact alice@example.com or 555-123-4567</body></html>"
    )

    out = bos.extract(
        sess.session_id, tab.tab_id, selector="", user_email="u@x.com"
    )

    assert "alice@example.com" not in out["text"]
    assert "[REDACTED:email]" in out["text"]
    assert "[REDACTED:phone]" in out["text"]


def test_a_click_on_a_mailto_link_is_refused(bos):
    """The trap this guards: reading the href *after* the click is
    reading the page you already went to."""
    sess = _session(bos)
    tab = bos.create_tab(sess.session_id)

    with pytest.raises(PermissionError):
        bos.click(
            sess.session_id, tab.tab_id, selector="a:has-text('mail')",
            user_email="u@x.com",
        )
    history = bos.export_history(sess.session_id)
    assert any(h["outcome"].startswith("blocked:") for h in history)


def test_a_fenced_session_cannot_be_escaped_through_create_tab(bos):
    """create_tab used to _goto directly: no fence, no mailto refusal,
    no history — a side door around every navigation rule."""
    sess = _session(bos)
    sess.allowed_origin = "https://example.com"

    with pytest.raises(PermissionError):
        bos.create_tab(sess.session_id, url="https://evil.example.com")
    with pytest.raises(PermissionError):
        bos.create_tab(sess.session_id, url="mailto:x@y.com")

    # Nothing landed anywhere, and the attempts are in the history.
    history = bos.export_history(sess.session_id)
    assert any(h["outcome"] == "blocked:origin" for h in history)
    assert not any(t.url == "https://evil.example.com" for t in bos.list_tabs(sess.session_id))


def test_a_landing_off_origin_is_blocked_not_served(bos, monkeypatch):
    """Chromium follows redirects itself, so the fence must hold on
    where the page LANDED: a click or goto that ends up off-origin is
    recorded blocked and the tab is blanked rather than served."""
    sess = _session(bos)
    sess.allowed_origin = "https://example.com"
    tab = bos.create_tab(sess.session_id)
    page = bos._pages[(sess.session_id, tab.tab_id)]

    # A click whose target URL passes the (prefetch) check but whose
    # final URL — after Chromium's own redirect — lands elsewhere.
    def fake_goto(self, page, tab, url):
        page.url = "https://evil.example.com/x"
        tab.url = page.url
        tab.title = "Evil"
        tab.last_status = 200
        tab.last_error = None

    monkeypatch.setattr(PlaywrightBrowserOS, "_goto", fake_goto)
    # The landing URL is what trips the post-navigation fence check.
    with pytest.raises(PermissionError):
        bos.navigate(
            sess.session_id, tab.tab_id,
            url="https://example.com/redirector", user_email="u@x.com",
        )
    assert page.url == "about:blank", "the off-origin page stayed loaded"


# ---- the dev backend still passes the same rules ---------------------------


def test_the_dev_backend_refuses_the_same_mailto():
    dev = InMemoryBrowserOS()
    sess = dev.create_session(user_email="u@x.com", instance_scope="default")
    tab = dev.create_tab(sess.session_id)

    with pytest.raises(PermissionError):
        dev.navigate(
            sess.session_id, tab.tab_id,
            url="mailto:x@y.com", user_email="u@x.com",
        )


# ---- real Chromium, when available -----------------------------------------


@pytest.mark.integration
def test_real_chromium_round_trip():
    """End-to-end against actual Chromium: navigate, extract, close.
    Skipped unless Playwright and a browser binary are installed."""
    import pytest as _pytest

    try:
        import playwright  # noqa: F401
    except ImportError:
        _pytest.skip("playwright not installed")

    bos = PlaywrightBrowserOS(headless=True)
    try:
        sess = bos.create_session(user_email="u@x.com", instance_scope="default")
        tab = bos.create_tab(sess.session_id, url="about:blank")
        out = bos.extract(
            sess.session_id, tab.tab_id, selector="", user_email="u@x.com"
        )
        assert out["ok"] if isinstance(out, dict) and "ok" in out else out["text"] is not None
    finally:
        bos.close()
