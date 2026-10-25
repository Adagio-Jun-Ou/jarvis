"""PlaywrightBrowserOS â€” the production Browser OS backend.

The dev backend (``InMemoryBrowserOS`` in ``agent/browser_os.py``) is
an HTML string store; this is the same contract against a real
Chromium. Every governance rule the dev backend enforces is enforced
here too, because a rule that only exists in the mock is a rule the
tests validate and production does not run:

* **Origin fence** â€” a fenced session may only navigate within its
  exact origin (``origin_ok``, shared with the dev backend so the
  lookalike-subdomain trap is closed once, not twice).
* **The piloting wheel** â€” a tab in ``co-drive`` refuses agent
  navigation, clicks and submits; the user holds the wheel until they
  yield it back.
* **mailto:/tel: refusal** â€” Alex never sends email or dials numbers
  on the user's behalf, including from a click-followed link.
* **PII redaction** â€” ``extract`` returns redacted text with the same
  regex pass the dev backend applies.
* **Two-phase submits** â€” ``submit_form`` stages and records; it never
  performs the POST. The caller surfaces the confirmation and re-calls
  with a token.
* **History** â€” every operation records to the tab's history exactly as
  the dev backend does, so ``export_history`` (and the oversight
  console that reads it) cannot tell which backend produced it.

Playwright is an optional dependency of this module only: importing
``agent.browser_os`` never imports Playwright, so dev machines and CI
without a Chromium binary are unaffected. The class is constructed
through :func:`agent.browser_os.make_production_browser_os`, never at
import time.

Synchronization: Playwright's sync API is used from a single owner
thread per operation. The FastAPI routes and the agent tools call the
Browser OS synchronously, matching the ``BrowserOS`` Protocol, so each
public method acquires the Chromium lock, performs the operation, and
releases. Sessions are kept in ``self._pages`` keyed by (session, tab);
the Playwright objects are never exposed through the protocol.

Lifespan: ``close()`` must be called when the backend is discarded
(app shutdown), otherwise the Chromium process leaks. ``__del__`` is a
backstop, not the mechanism.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from agent.browser_os import (
    BrowserSessionState,
    BrowserTabState,
    _redact_pii,
    _strip_tags,
    origin_ok,
)

logger = logging.getLogger("agent.browser_os_playwright")

#: Hard ceiling on one navigate waiting for the page. A page that never
#: finishes loading must not hold the browser lock forever â€” every other
#: session shares this process.
NAV_TIMEOUT_MS = 30_000


class PlaywrightBrowserOS:
    """The production :class:`agent.browser_os.BrowserOS` Protocol.

    Constructed only through
    :func:`agent.browser_os.make_production_browser_os`, never at import
    time â€” see the module docstring.
    """

    def __init__(self, *, headless: bool = True) -> None:
        from playwright.sync_api import sync_playwright

        self._lock = threading.RLock()
        self._sessions: dict[str, BrowserSessionState] = {}
        self._pages: dict[tuple[str, str], Any] = {}  # (sid, tid) -> Page
        self._closed = False
        self._pw = sync_playwright().start()
        try:
            self._browser = self._pw.chromium.launch(
                headless=headless,
                # The Chromium sandbox: no shared filesystem with the host
                # process beyond the download dir, and dev-shm disabled for
                # container quota determinism.
                args=[
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-blink-features=AutomationControlled",
                ],
            )
        except Exception:
            # A launch failure (no binary, no /dev/shm) must not leak the
            # driver subprocess it already started.
            self._pw.stop()
            raise

    # -- session management ------------------------------------------------

    def create_session(
        self, *, user_email: str, instance_scope: str
    ) -> BrowserSessionState:
        sid = f"bsess_{int(time.time() * 1000)}_{len(self._sessions)}"
        sess = BrowserSessionState(
            session_id=sid,
            user_email=user_email,
            instance_scope=instance_scope,
        )
        self._sessions[sid] = sess
        return sess

    def close_session(self, session_id: str) -> None:
        sess = self._sessions.pop(session_id, None)
        if sess is None:
            return
        with self._lock:
            for tid in list(sess.tabs):
                page = self._pages.pop((session_id, tid), None)
                if page is not None:
                    try:
                        page.close()
                    except Exception:  # noqa: BLE001
                        logger.debug("page close failed", exc_info=True)

    def list_sessions(self, user_email: str) -> list[BrowserSessionState]:
        return [s for s in self._sessions.values() if s.user_email == user_email]

    def _require_session(self, session_id: str) -> BrowserSessionState:
        sess = self._sessions.get(session_id)
        if sess is None:
            raise KeyError(f"no such browser session: {session_id}")
        sess.last_active_at = time.time()
        return sess

    def _require_owned(self, session_id: str, user_email: str) -> BrowserSessionState:
        """Ownership gate â€” see InMemoryBrowserOS._require_owned. Session
        ids are enumerable; without this the model-supplied session_id
        was the only control on another user's browser."""
        sess = self._require_session(session_id)
        if user_email and sess.user_email != user_email:
            raise PermissionError("browser session belongs to another user")
        return sess

    # -- tab management -------------------------------------------------------

    def create_tab(
        self, session_id: str, *, url: str = "about:blank"
    ) -> BrowserTabState:
        sess = self._require_session(session_id)
        # A URL here bypasses nothing: the initial page goes through the
        # same validated navigation every other page load uses â€” fence,
        # wheel, mailto/tel, history â€” instead of being a side door
        # around all of them.
        with self._lock:
            tid = f"tab_{len(sess.tabs)}"
            page = self._browser.new_page()
            page.set_default_timeout(NAV_TIMEOUT_MS)
            tab = BrowserTabState(tab_id=tid, url="about:blank")
            self._pages[(session_id, tid)] = page
            sess.tabs[tid] = tab
            if url and url != "about:blank":
                page.goto("about:blank")  # deterministic blank origin
                self.navigate(
                    session_id, tid, url=url, user_email=sess.user_email
                )
            return tab

    def close_tab(
        self, session_id: str, tab_id: str, *, user_email: str = ""
    ) -> None:
        if user_email:
            self._require_owned(session_id, user_email)
        sess = self._require_session(session_id)
        sess.tabs.pop(tab_id, None)
        with self._lock:
            page = self._pages.pop((session_id, tab_id), None)
            if page is not None:
                try:
                    page.close()
                except Exception:  # noqa: BLE001
                    logger.debug("tab close failed", exc_info=True)

    def list_tabs(
        self, session_id: str, *, user_email: str = ""
    ) -> list[BrowserTabState]:
        if user_email:
            self._require_owned(session_id, user_email)
        return list(self._require_session(session_id).tabs.values())

    def get_tab(self, session_id: str, tab_id: str) -> BrowserTabState:
        sess = self._require_session(session_id)
        tab = sess.tabs.get(tab_id)
        if tab is None:
            raise KeyError(f"no such tab: {tab_id}")
        return tab

    def _page_for(self, session_id: str, tab_id: str):
        page = self._pages.get((session_id, tab_id))
        if page is None:
            raise KeyError(f"no such tab: {tab_id}")
        return page

    # -- core operations ------------------------------------------------------

    def navigate(
        self, session_id: str, tab_id: str, *, url: str, user_email: str
    ) -> BrowserTabState:
        self._require_owned(session_id, user_email)
        tab = self.get_tab(session_id, tab_id)
        if not tab.agent_piloting:
            tab.last_error = (
                "tab is user-controlled; agent must yield before navigating"
            )
            tab.record(
                op="navigate",
                user_email=user_email,
                args={"url": url},
                outcome="blocked:user_holds_wheel",
            )
            raise PermissionError(tab.last_error)
        fence = tab.allowed_origin or self._require_session(
            session_id
        ).allowed_origin
        if not origin_ok(url, fence):
            tab.last_error = (
                f"refusing to navigate outside {fence}; this session is "
                "fenced to a single origin"
            )
            tab.record(
                op="navigate", user_email=user_email,
                args={"url": url}, outcome="blocked:origin",
            )
            raise PermissionError(tab.last_error)
        from urllib.parse import urlparse

        parsed = urlparse(url)
        if parsed.scheme in {"mailto", "tel"}:
            tab.last_error = (
                f"refusing to navigate to a {parsed.scheme}: URL; "
                "Alex never sends email or makes calls on the user's behalf"
            )
            tab.record(
                op="navigate",
                user_email=user_email,
                args={"url": url},
                outcome=f"blocked:{parsed.scheme}",
            )
            raise PermissionError(tab.last_error)
        page = self._page_for(session_id, tab_id)
        with self._lock:
            self._goto(page, tab, url)
            # goto follows redirects inside Chromium: the fence must
            # hold on where the page landed, not just the URL checked
            # above. Same recovery as the click path â€” blank the tab,
            # record the block, return nothing from the internal site.
            fence = tab.allowed_origin or self._require_session(
                session_id
            ).allowed_origin
            if fence and tab.last_error is None and not origin_ok(
                tab.url, fence
            ):
                tab.last_error = (
                    f"a redirect took the page outside {fence}; the "
                    "session is fenced to a single origin"
                )
                tab.record(
                    op="navigate",
                    user_email=user_email,
                    args={"url": url, "landed": tab.url},
                    outcome="blocked:origin",
                )
                page.goto("about:blank")
                raise PermissionError(tab.last_error)
        # _goto swallows the failure into tab.last_error (a dead URL is
        # an outcome, not an exception, for a browsing session) â€” so the
        # history row must tell the truth about it, or oversight reads a
        # navigation that never happened as a success.
        failed = tab.last_error is not None
        tab.record(
            op="navigate",
            user_email=user_email,
            args={"url": url},
            outcome="failed" if failed else "ok",
        )
        return tab

    def click(
        self, session_id: str, tab_id: str, *, selector: str, user_email: str
    ) -> BrowserTabState:
        self._require_owned(session_id, user_email)
        tab = self.get_tab(session_id, tab_id)
        if not tab.agent_piloting:
            tab.last_error = (
                "tab is user-controlled; agent must yield before clicking"
            )
            tab.record(
                op="click",
                user_email=user_email,
                args={"selector": selector},
                outcome="blocked:user_holds_wheel",
            )
            raise PermissionError(tab.last_error)
        page = self._page_for(session_id, tab_id)
        with self._lock:
            try:
                # Capture the href BEFORE the click mutates the page: a
                # mailto: link must be refused, and reading it after the
                # click is reading the page you already went to.
                link = page.locator(selector).first
                href = link.get_attribute("href") or ""
                from urllib.parse import urlparse

                if urlparse(href).scheme in {"mailto", "tel"}:
                    tab.last_error = (
                        f"refusing to follow {href[:12]} link from a click; "
                        "Alex never sends email or makes calls on the "
                        "user's behalf"
                    )
                    tab.record(
                        op="click",
                        user_email=user_email,
                        args={"selector": selector, "href": href},
                        outcome="blocked:email_or_tel",
                    )
                    raise PermissionError(tab.last_error)
                link.click()
                page.wait_for_load_state("domcontentloaded")
            except PermissionError:
                raise
            except Exception as exc:  # noqa: BLE001
                tab.last_error = f"no link matched selector {selector!r}: {exc}"
                tab.record(
                    op="click",
                    user_email=user_email,
                    args={"selector": selector},
                    outcome="no_match",
                )
                raise LookupError(tab.last_error) from exc
            self._sync_tab(page, tab)
            # Chromium follows HTTP redirects and JS navigation on its
            # own, so the fence must be checked on where the page
            # actually LANDED, not only on the URL we asked for. An
            # off-origin landing in a fenced session is recorded as a
            # blocked navigation and the tab is sent back to a blank
            # page â€” the internal resource was fetched, but nothing
            # from it is returned to the model, and the history says
            # what happened.
            fence = tab.allowed_origin or self._require_session(
                session_id
            ).allowed_origin
            if fence and not origin_ok(tab.url, fence):
                tab.last_error = (
                    f"a click took the page outside {fence}; the session "
                    "is fenced to a single origin"
                )
                tab.record(
                    op="click",
                    user_email=user_email,
                    args={"selector": selector, "landed": tab.url},
                    outcome="blocked:origin",
                )
                page.goto("about:blank")
                raise PermissionError(tab.last_error)
        tab.record(
            op="click",
            user_email=user_email,
            args={"selector": selector},
            outcome="ok",
        )
        return tab

    def type_text(
        self,
        session_id: str,
        tab_id: str,
        *,
        selector: str,
        text: str,
        user_email: str,
    ) -> BrowserTabState:
        self._require_owned(session_id, user_email)
        tab = self.get_tab(session_id, tab_id)
        if not tab.agent_piloting:
            tab.last_error = (
                "tab is user-controlled; agent must yield before typing"
            )
            tab.record(
                op="type",
                user_email=user_email,
                args={"selector": selector},
                outcome="blocked:user_holds_wheel",
            )
            raise PermissionError(tab.last_error)
        page = self._page_for(session_id, tab_id)
        with self._lock:
            try:
                field = page.locator(selector).first
                field.fill(text)
                name = field.get_attribute("name") or selector
            except Exception as exc:  # noqa: BLE001
                tab.last_error = (
                    f"no input/textarea matched selector {selector!r}: {exc}"
                )
                tab.record(
                    op="type",
                    user_email=user_email,
                    args={"selector": selector},
                    outcome="no_match",
                )
                raise LookupError(tab.last_error) from exc
            tab.form[name] = text
        tab.record(
            op="type",
            user_email=user_email,
            args={"selector": selector, "name": name, "text_len": len(text)},
            outcome="ok",
        )
        return tab

    def extract(
        self, session_id: str, tab_id: str, *, selector: str, user_email: str
    ) -> dict[str, Any]:
        self._require_owned(session_id, user_email)
        tab = self.get_tab(session_id, tab_id)
        page = self._page_for(session_id, tab_id)
        with self._lock:
            try:
                html = page.content()
                if selector:
                    target = page.locator(selector).first
                    text = target.inner_text() if target.count() else ""
                    html = (
                        f"<html><head><title>{tab.title}</title></head>"
                        f"<body>{text}</body></html>"
                    )
            except Exception:  # noqa: BLE001
                html = page.content()
            raw = _strip_tags(html)
            redacted = _redact_pii(raw)
            form = dict(tab.form)
        tab.record(
            op="extract", user_email=user_email,
            args={"selector": selector}, outcome="ok",
        )
        return {
            "tab_id": tab_id,
            "url": tab.url,
            "title": tab.title,
            "text": redacted,
            "form": form,
            "selector": selector,
            # Same truthfulness as the dev backend: did redaction
            # actually change anything, not "we ran the regex".
            "pii_redacted": redacted != raw,
        }

    def screenshot(
        self, session_id: str, tab_id: str, *, user_email: str
    ) -> dict[str, Any]:
        self._require_owned(session_id, user_email)
        tab = self.get_tab(session_id, tab_id)
        page = self._page_for(session_id, tab_id)
        with self._lock:
            png = page.screenshot(type="png")
        tab.record(op="screenshot", user_email=user_email, args={}, outcome="ok")
        return {
            "tab_id": tab_id,
            "url": tab.url,
            "mime": "image/png",
            "data": png.hex(),
        }

    def submit_form(
        self, session_id: str, tab_id: str, *, selector: str, user_email: str
    ) -> BrowserTabState:
        """Stage a form submission for human approval (constitution A8).

        Two-phase, same as the dev backend: this records and stages; it
        never performs the submit. The caller surfaces the confirmation
        and re-calls with a token.
        """
        self._require_owned(session_id, user_email)
        tab = self.get_tab(session_id, tab_id)
        if not tab.agent_piloting:
            tab.last_error = (
                "tab is user-controlled; agent must yield before submitting"
            )
            tab.record(
                op="submit",
                user_email=user_email,
                args={"selector": selector},
                outcome="blocked:user_holds_wheel",
            )
            raise PermissionError(tab.last_error)
        page = self._page_for(session_id, tab_id)
        with self._lock:
            try:
                form = page.locator(selector).first
                found = form.count() > 0
            except Exception:  # noqa: BLE001
                found = False
        if not found:
            tab.last_error = f"no form matched selector {selector!r}"
            tab.record(
                op="submit",
                user_email=user_email,
                args={"selector": selector},
                outcome="no_match",
            )
            raise LookupError(tab.last_error)
        tab.form["_pending_submit_selector"] = selector
        tab.record(
            op="submit_staged",
            user_email=user_email,
            args={"selector": selector, "fields": list(tab.form)},
            outcome="awaiting_confirmation",
        )
        return tab

    def execute_submit(
        self, session_id: str, tab_id: str, *, selector: str, user_email: str
    ) -> BrowserTabState:
        """Perform the submit the user approved (see the dev backend for
        the contract). Here it is a real click: Chromium fills nothing
        new — the staged values are already in the page — it presses
        the button the user said yes to."""
        self._require_owned(session_id, user_email)
        tab = self.get_tab(session_id, tab_id)
        if not tab.agent_piloting:
            tab.last_error = (
                "tab is user-controlled; agent must yield before submitting"
            )
            tab.record(
                op="submit_execute",
                user_email=user_email,
                args={"selector": selector},
                outcome="blocked:user_holds_wheel",
            )
            raise PermissionError(tab.last_error)
        staged = tab.form.pop("_pending_submit_selector", None)
        if staged is None:
            tab.last_error = "no staged form submission to execute"
            tab.record(
                op="submit_execute",
                user_email=user_email,
                args={"selector": selector},
                outcome="no_staged_submit",
            )
            raise LookupError(tab.last_error)
        page = self._page_for(session_id, tab_id)
        with self._lock:
            try:
                page.locator(staged).first.click()
                page.wait_for_load_state("domcontentloaded")
            except Exception as exc:  # noqa: BLE001
                tab.last_error = f"executing the approved submit failed: {exc}"
                tab.record(
                    op="submit_execute",
                    user_email=user_email,
                    args={"selector": staged},
                    outcome="failed",
                )
                raise LookupError(tab.last_error) from exc
            self._sync_tab(page, tab)
        tab.record(
            op="submit_execute",
            user_email=user_email,
            args={"selector": staged},
            outcome="ok",
        )
        return tab

    def set_user_control(
        self, session_id: str, tab_id: str, *, mode: str, user_email: str
    ) -> BrowserTabState:
        self._require_owned(session_id, user_email)
        tab = self.get_tab(session_id, tab_id)
        if mode == "yield":
            tab.agent_piloting = True
            tab.user_yielded_at = None
        elif mode == "co-drive":
            tab.agent_piloting = False
            tab.user_yielded_at = time.time()
        elif mode == "read":
            tab.agent_piloting = True
            tab.user_yielded_at = time.time()
        else:
            raise ValueError(f"unknown mode {mode!r}; expected yield|co-drive|read")
        tab.record(
            op="set_user_control",
            user_email=user_email,
            args={"mode": mode},
            outcome="ok",
        )
        return tab

    def export_history(self, session_id: str) -> list[dict[str, Any]]:
        sess = self._require_session(session_id)
        out: list[dict[str, Any]] = []
        for tab in sess.tabs.values():
            for h in tab.history:
                out.append({"tab_id": tab.tab_id, **h})
        return out

    # -- internals ---------------------------------------------------------

    def _goto(self, page, tab: BrowserTabState, url: str) -> None:
        """Navigate and mirror the outcome into the tab state.

        A refused or errored navigation is recorded on the tab exactly
        as the dev backend records it, so the oversight console reads
        the same history either backend.
        """
        try:
            resp = page.goto(url, timeout=NAV_TIMEOUT_MS, wait_until="domcontentloaded")
            tab.url = url
            tab.title = page.title() or ""
            tab.last_status = resp.status if resp else 200
            tab.last_error = None
            tab.form = {}
        except Exception as exc:  # noqa: BLE001
            tab.url = url
            tab.title = ""
            tab.last_status = 0
            tab.last_error = f"navigation failed: {exc}"

    def _sync_tab(self, page, tab: BrowserTabState) -> None:
        try:
            tab.url = page.url
            tab.title = page.title() or ""
        except Exception:  # noqa: BLE001
            logger.debug("tab sync failed", exc_info=True)

    # -- lifespan ----------------------------------------------------------

    def close(self) -> None:
        """Stop Chromium. Call at app shutdown; ``__del__`` is a backstop."""
        if self._closed:
            return
        self._closed = True
        with self._lock:
            for page in list(self._pages.values()):
                try:
                    page.close()
                except Exception:  # noqa: BLE001
                    pass
            self._pages.clear()
            try:
                self._browser.close()
            except Exception:  # noqa: BLE001
                pass
            try:
                self._pw.stop()
            except Exception:  # noqa: BLE001
                pass

    def __del__(self) -> None:  # pragma: no cover - interpreter teardown
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass
