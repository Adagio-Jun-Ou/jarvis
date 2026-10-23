"""
Browser OS â€” the agent's sandboxed browser + the user-facing web view.

The Browser OS gives Alex a real Chromium sandbox in production
(``playwright.async_api``) and a small in-process HTML store in
development (``InMemoryBrowserOS``).  Both implement the
:class:`BrowserOS` interface, so the constitution layer, the
guardrails gate, the FastAPI routes, and the agent's tool factories
do not care which backend is in use.

The dev backend is **not** a rendering engine â€” it stores HTML as
strings and answers ``extract`` / ``click`` / ``type`` operations with
straightforward structural queries.  This is enough to:

* Exercise the constitution L2 interceptor (A8 fires for ``POST``
  ``/api_act`` triggers; PII redaction fires for ``extract``).
* Drive end-to-end agent loops in unit tests.
* Power the React Browser OS shell's iframe-backed ``BrowserTab`` view
  in dev (the same HTML is served as the iframe ``srcdoc``).

A production deploy uses the ``PlaywrightBrowserOS`` (see
``agent/browser_os.py::PlaywrightBrowserOS``) which is intentionally
not imported at module load time so dev environments don't need a
working Chromium.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import urljoin, urlparse

logger = logging.getLogger(__name__)


# A small set of HTML patterns we recognize in dev. Production
# extraction uses Playwright's ``page.locator(...)`` API.
_HTML_LINK_RE = re.compile(
    r'<a[^>]+href=["\'](?P<href>[^"\']+)["\'][^>]*>(?P<text>.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)
_HTML_FORM_RE = re.compile(
    r"<form[^>]*>(?P<body>.*?)</form>", re.IGNORECASE | re.DOTALL
)
_HTML_INPUT_RE = re.compile(
    r"<input[^>]+name=[\"\'](?P<name>[^\"\']+)[\"\'][^>]*>", re.IGNORECASE
)
_HTML_TEXTAREA_RE = re.compile(
    r"<textarea[^>]+name=[\"\'](?P<name>[^\"\']+)[\"\'][^>]*>(?P<text>.*?)</textarea>",
    re.IGNORECASE | re.DOTALL,
)
_HTML_TITLE_RE = re.compile(
    r"<title[^>]*>(?P<title>.*?)</title>", re.IGNORECASE | re.DOTALL
)
_HTML_SCRIPT_RE = re.compile(r"<script[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL)


def _strip_tags(html: str) -> str:
    """Strip script + style blocks, then HTML tags, then collapse
    whitespace, then entity-decode.

    Entity-decoding is not cosmetic: a page that renders
    ``joe&#64;corp.com`` shows the browser (and Playwright's
    inner_text) a real address, while the raw-source path used by
    extract sees only the encoded form â€” which none of the PII
    regexes match. Decoding here closes that gap for every caller."""
    html = _HTML_SCRIPT_RE.sub(" ", html)
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text).strip()
    import html as _html

    return _html.unescape(text)


def _redact_pii(text: str, *, max_pii_chars: int = 256) -> str:
    """Constitution A7: redact PII beyond a small per-field cap.

    The dev implementation looks for the obvious patterns (emails,
    phone numbers, SSN-shaped 9-digit numbers).  Production redaction
    uses the same regexes plus a Presidio / Comprehend PII detector.
    """
    out = text
    out = re.sub(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[REDACTED:email]", out
    )
    out = re.sub(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", "[REDACTED:phone]", out)
    out = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[REDACTED:ssn]", out)
    # If redaction grew the text past the cap, drop a sentinel so the
    # caller can choose to escalate to a warning event.
    if len(out) > max_pii_chars:
        out = out[: max_pii_chars - 16] + " [TRUNCATED]"
    return out


@dataclass
class BrowserTabState:
    """Per-tab runtime state.

    Holds the URL, the HTML body, the in-memory form state, and the
    last error string.  The state is updated by the operations
    (navigate, click, type, extract).  The history list is append-only
    and written to ``media.control.browser_history`` on every change.
    """

    tab_id: str
    url: str = "about:blank"
    title: str = ""
    html: str = ""
    form: dict[str, str] = field(default_factory=dict)
    last_error: str | None = None
    last_status: int = 200
    # Per-tab override of the session fence; empty means inherit.
    allowed_origin: str = ""
    agent_piloting: bool = True
    user_yielded_at: float | None = None
    history: list[dict[str, Any]] = field(default_factory=list)

    def record(
        self, *, op: str, user_email: str, args: dict[str, Any], outcome: str
    ) -> None:
        self.history.append(
            {
                "ts": time.time(),
                "op": op,
                "user_email": user_email,
                "args": {k: str(v)[:200] for k, v in args.items()},
                "outcome": outcome,
                "url": self.url,
            }
        )


@dataclass
class BrowserSessionState:
    """A single browser session scoped to one user/instance."""

    session_id: str
    user_email: str
    instance_scope: str
    tabs: dict[str, BrowserTabState] = field(default_factory=dict)
    # When set, the agent may only navigate within this exact origin.
    # Used for sessions driving an internal system, where wandering off
    # to another host is never legitimate.
    allowed_origin: str = ""
    created_at: float = field(default_factory=time.time)
    last_active_at: float = field(default_factory=time.time)

    def active_tabs(self) -> Iterable[BrowserTabState]:
        return list(self.tabs.values())


class BrowserOS(Protocol):
    """The interface every Browser OS backend must implement."""

    def create_session(
        self, *, user_email: str, instance_scope: str
    ) -> BrowserSessionState: ...
    def close_session(self, session_id: str) -> None: ...
    def list_sessions(self, user_email: str) -> list[BrowserSessionState]: ...
    def create_tab(
        self, session_id: str, *, url: str = "about:blank"
    ) -> BrowserTabState: ...
    def close_tab(
        self, session_id: str, tab_id: str, *, user_email: str = ""
    ) -> None: ...
    def list_tabs(
        self, session_id: str, *, user_email: str = ""
    ) -> list[BrowserTabState]: ...
    def navigate(
        self, session_id: str, tab_id: str, *, url: str, user_email: str
    ) -> BrowserTabState: ...
    def click(
        self, session_id: str, tab_id: str, *, selector: str, user_email: str
    ) -> BrowserTabState: ...
    def type_text(
        self,
        session_id: str,
        tab_id: str,
        *,
        selector: str,
        text: str,
        user_email: str,
    ) -> BrowserTabState: ...
    def extract(
        self, session_id: str, tab_id: str, *, selector: str, user_email: str
    ) -> dict[str, Any]: ...
    def screenshot(
        self, session_id: str, tab_id: str, *, user_email: str
    ) -> dict[str, Any]: ...
    def submit_form(
        self, session_id: str, tab_id: str, *, selector: str, user_email: str
    ) -> BrowserTabState: ...
    def set_user_control(
        self, session_id: str, tab_id: str, *, mode: str, user_email: str
    ) -> BrowserTabState: ...
    def export_history(self, session_id: str) -> list[dict[str, Any]]: ...


# ---------------------------------------------------------------------------
# In-memory dev backend
# ---------------------------------------------------------------------------


def origin_ok(url: str, allowed_origin: str) -> bool:
    """Is ``url`` within ``allowed_origin``?

    Compares scheme and host **exactly**. The trap this exists for is
    suffix matching: ``intranet.corp.attacker.example`` ends with
    ``intranet.corp`` as a string, and a naive ``startswith`` or
    ``endswith`` check hands an attacker any internal origin they can
    register a lookalike domain for. Port differences are also a
    different origin, as they are everywhere else on the web.

    An empty ``allowed_origin`` means unfenced, which is the default â€”
    the fence is opt-in per session, because most sessions legitimately
    browse the open web.
    """
    if not allowed_origin:
        return True
    if not url:
        return False
    target, allowed = urlparse(url), urlparse(allowed_origin)
    if not target.scheme or not allowed.scheme:
        return False
    if not allowed.hostname:
        # A scheme-only fence ("alex://") means "anything served by this
        # scheme" â€” used for the internal pseudo-origin, which has no
        # host to compare.
        return target.scheme == allowed.scheme
    return (
        target.scheme == allowed.scheme
        and (target.hostname or "").lower() == (allowed.hostname or "").lower()
        and (target.port or _default_port(target.scheme))
        == (allowed.port or _default_port(allowed.scheme))
    )


def _default_port(scheme: str) -> int:
    return {"https": 443, "http": 80}.get(scheme, 0)


class InMemoryBrowserOS:
    """Process-local Browser OS for dev / unit tests.

    The :class:`BrowserOS` Protocol above documents the contract;
    this class is the implementation.  Pages are stored as raw HTML
    strings.  ``navigate`` accepts ``about:blank`` and ``file://`` URLs
    pointing into the registered fixture set, plus ``http://localhost:NNNN``
    for the dev FastAPI server when running end-to-end.

    The :class:`PlaywrightBrowserOS` (not loaded by default) implements
    the same Protocol against a real Chromium.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, BrowserSessionState] = {}
        # Pre-seeded pages so unit tests can navigate by URL.
        self._fixtures: dict[str, str] = {
            "about:blank": "<!doctype html><html><head><title>blank</title></head><body></body></html>",
            "alex://welcome": (
                "<!doctype html><html><head><title>Welcome to Alex</title></head>"
                "<body><h1>Welcome to Alex</h1>"
                "<p>Sign in with your work email to get started.</p>"
                "<form action='/submit' method='post'>"
                "<input name='email' type='email' required />"
                "<input name='message' type='text' />"
                "<button type='submit'>Send</button>"
                "</form>"
                "<a href='alex://docs'>Docs</a>"
                "<a href='mailto:support@example.com'>Email support</a>"
                "</body></html>"
            ),
            "alex://docs": (
                "<!doctype html><html><head><title>Alex Docs</title></head>"
                "<body><h1>Docs</h1><p>Read the docs.</p></body></html>"
            ),
        }

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
        self._sessions.pop(session_id, None)

    def list_sessions(self, user_email: str) -> list[BrowserSessionState]:
        return [s for s in self._sessions.values() if s.user_email == user_email]

    def _require_session(self, session_id: str) -> BrowserSessionState:
        sess = self._sessions.get(session_id)
        if sess is None:
            raise KeyError(f"no such browser session: {session_id}")
        sess.last_active_at = time.time()
        return sess

    def _require_owned(self, session_id: str, user_email: str) -> BrowserSessionState:
        """The session, but only for its owner.

        Session ids are enumerable (``bsess_<ms>_<n>``): without this
        check, a session id seen in a URL or log was the only access
        control on another user's tabs â€” page text, typed form values
        and all. The HTTP routes checked ownership; the agent tool path
        did not, and it is the model that supplies session_id.
        """
        sess = self._require_session(session_id)
        if user_email and sess.user_email != user_email:
            raise PermissionError(
                "browser session belongs to another user"
            )
        return sess

    # -- tab management ----------------------------------------------------
    def create_tab(
        self, session_id: str, *, url: str = "about:blank"
    ) -> BrowserTabState:
        sess = self._require_session(session_id)
        tid = f"tab_{len(sess.tabs)}"
        tab = BrowserTabState(tab_id=tid, url="about:blank")
        sess.tabs[tid] = tab
        # The initial URL goes through the same validated navigate()
        # every other page load uses â€” fence, mailto/tel, history â€”
        # rather than being a side door around all of them.
        if url and url != "about:blank":
            self.navigate(session_id, tid, url=url, user_email=sess.user_email)
        return tab

    def close_tab(
        self, session_id: str, tab_id: str, *, user_email: str = ""
    ) -> None:
        self._require_owned(session_id, user_email) if user_email else None
        sess = self._require_session(session_id)
        sess.tabs.pop(tab_id, None)

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

    # -- core operations ---------------------------------------------------
    def navigate(
        self, session_id: str, tab_id: str, *, url: str, user_email: str
    ) -> BrowserTabState:
        tab = self.get_tab(session_id, tab_id)
        self._require_owned(session_id, user_email)
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
        # mailto: and tel: are not navigation in Alex â€” same guarantee as
        # click-following, enforced on the direct-navigate path too.
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
        self._load_html(tab, url)
        tab.record(
            op="navigate", user_email=user_email, args={"url": url}, outcome="ok"
        )
        return tab

    def click(
        self, session_id: str, tab_id: str, *, selector: str, user_email: str
    ) -> BrowserTabState:
        tab = self.get_tab(session_id, tab_id)
        self._require_owned(session_id, user_email)
        if not tab.agent_piloting:
            tab.last_error = "tab is user-controlled; agent must yield before clicking"
            tab.record(
                op="click",
                user_email=user_email,
                args={"selector": selector},
                outcome="blocked:user_holds_wheel",
            )
            raise PermissionError(tab.last_error)
        # Find the link matching the selector (a name= attr or visible text).
        target_url: str | None = None
        for m in _HTML_LINK_RE.finditer(tab.html):
            text = _strip_tags(m.group("text"))
            if selector in text or selector == m.group("href"):
                target_url = m.group("href")
                break
        if not target_url:
            tab.last_error = f"no link matched selector {selector!r}"
            tab.record(
                op="click",
                user_email=user_email,
                args={"selector": selector},
                outcome="no_match",
            )
            raise LookupError(tab.last_error)
        # mailto: and tel: are not navigation in Alex.
        parsed = urlparse(target_url)
        if parsed.scheme in {"mailto", "tel"}:
            tab.last_error = f"refusing to follow {parsed.scheme}:// link from a click; Alex never sends email or makes calls on the user's behalf"
            tab.record(
                op="click",
                user_email=user_email,
                args={"selector": selector, "href": target_url},
                outcome=f"blocked:{parsed.scheme}",
            )
            raise PermissionError(tab.last_error)
        # The fence applies to click-followed navigation too: a link on
        # the allowed origin pointing elsewhere (or a relative href the
        # loader resolves off-origin) is exactly how a fenced session
        # escapes. Relative hrefs resolve against the tab's URL, so
        # the check runs on the resolved absolute target.
        fence = tab.allowed_origin or self._require_session(
            session_id
        ).allowed_origin
        resolved = urljoin(tab.url or "about:blank", target_url)
        if not origin_ok(resolved, fence):
            tab.last_error = (
                f"refusing to follow a link outside {fence or 'the origin fence'}"
            )
            tab.record(
                op="click",
                user_email=user_email,
                args={"selector": selector, "href": target_url},
                outcome="blocked:origin",
            )
            raise PermissionError(tab.last_error)
        self._load_html(tab, target_url)
        tab.record(
            op="click",
            user_email=user_email,
            args={"selector": selector, "href": target_url},
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
        tab = self.get_tab(session_id, tab_id)
        self._require_owned(session_id, user_email)
        if not tab.agent_piloting:
            tab.last_error = "tab is user-controlled; agent must yield before typing"
            tab.record(
                op="type",
                user_email=user_email,
                args={"selector": selector, "text": text},
                outcome="blocked:user_holds_wheel",
            )
            raise PermissionError(tab.last_error)
        # Find an input or textarea by name= and stage the value.
        match_input = _HTML_INPUT_RE.search(tab.html)
        match_textarea = _HTML_TEXTAREA_RE.search(tab.html)
        target_name: str | None = None
        for m in (match_input, match_textarea):
            if m and (
                selector == m.group("name")
                or selector in (m.groupdict().get("text") or "")
            ):
                target_name = m.group("name")
                break
        if not target_name:
            tab.last_error = f"no input/textarea matched selector {selector!r}"
            tab.record(
                op="type",
                user_email=user_email,
                args={"selector": selector},
                outcome="no_match",
            )
            raise LookupError(tab.last_error)
        tab.form[target_name] = text
        tab.record(
            op="type",
            user_email=user_email,
            args={"selector": selector, "name": target_name, "text_len": len(text)},
            outcome="ok",
        )
        return tab

    def extract(
        self, session_id: str, tab_id: str, *, selector: str, user_email: str
    ) -> dict[str, Any]:
        self._require_owned(session_id, user_email)
        tab = self.get_tab(session_id, tab_id)
        # extract() does not require the agent to be piloting; the user can
        # call it from a co-drive session too.  Returns redacted text +
        # form state.
        title_m = _HTML_TITLE_RE.search(tab.html)
        title = title_m.group("title").strip() if title_m else tab.title
        text = _strip_tags(tab.html)
        redacted = _redact_pii(text)
        form = dict(tab.form)
        tab.record(
            op="extract",
            user_email=user_email,
            args={"selector": selector},
            outcome="ok",
        )
        return {
            "tab_id": tab_id,
            "url": tab.url,
            "title": title,
            "text": redacted,
            "form": form,
            "selector": selector,
            "pii_redacted": redacted != text,
        }

    def screenshot(
        self, session_id: str, tab_id: str, *, user_email: str
    ) -> dict[str, Any]:
        self._require_owned(session_id, user_email)
        tab = self.get_tab(session_id, tab_id)
        tab.record(op="screenshot", user_email=user_email, args={}, outcome="ok")
        # Dev returns a tiny placeholder; prod returns a PNG bytes payload.
        return {
            "tab_id": tab_id,
            "url": tab.url,
            "placeholder": "dev-mode-screenshot",
        }

    def submit_form(
        self, session_id: str, tab_id: str, *, selector: str, user_email: str
    ) -> BrowserTabState:
        """Stage a form submission for human approval (constitution A8).

        The actual navigation to the form's action URL is *not* performed
        here; the agent's caller is expected to surface a
        ``confirmation_required`` SSE event to the user and re-call
        ``submit_form`` with a ``confirmation_token`` once the user has
        approved.  Two-phase externalization, by design.
        """
        tab = self.get_tab(session_id, tab_id)
        self._require_owned(session_id, user_email)
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
        form_m = _HTML_FORM_RE.search(tab.html)
        if not form_m:
            tab.last_error = f"no form matched selector {selector!r}"
            tab.record(
                op="submit",
                user_email=user_email,
                args={"selector": selector},
                outcome="no_match",
            )
            raise LookupError(tab.last_error)
        # Stage â€” do not navigate.
        tab.form["_pending_submit_selector"] = selector
        tab.record(
            op="submit_staged",
            user_email=user_email,
            args={"selector": selector, "fields": list(tab.form)},
            outcome="awaiting_confirmation",
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

    def execute_submit(
        self, session_id: str, tab_id: str, *, selector: str, user_email: str
    ) -> BrowserTabState:
        """Perform the submit the user approved (phase 2 of A8).

        The dev backend is an HTML store, so 'performing' the submit is
        consuming the staged selector and recording the execution — the
        honest equivalent for a page that does not exist. The Playwright
        backend really clicks. What phase 2 must never do, on either
        backend, is return ``submitted: True`` while consuming the token
        and changing nothing: the user approved an action, and the
        record must say it happened.
        """
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
        tab.record(
            op="submit_execute",
            user_email=user_email,
            args={"selector": staged},
            outcome="ok",
        )
        return tab

    # -- internals ---------------------------------------------------------
    def _load_html(self, tab: BrowserTabState, url: str) -> None:
        html = self._fixtures.get(url)
        if html is None:
            # The URL is unknown to the dev backend; remember it but
            # leave the body empty so the caller sees a 404-like state.
            tab.url = url
            tab.html = f"<!doctype html><html><body><h1>404</h1><p>{url} not in dev fixtures</p></body></html>"
            tab.title = "Not Found"
            tab.last_status = 404
            tab.last_error = f"dev backend has no fixture for {url}"
            return
        tab.url = url
        tab.html = html
        title_m = _HTML_TITLE_RE.search(html)
        tab.title = title_m.group("title").strip() if title_m else ""
        tab.last_status = 200
        tab.last_error = None
        tab.form = {}


# ---------------------------------------------------------------------------
# Production backend (loaded lazily; requires Playwright in prod)
# ---------------------------------------------------------------------------


def make_production_browser_os() -> BrowserOS:
    """Construct the production :class:`PlaywrightBrowserOS`.

    Imported lazily so the dev path doesn't require Playwright or a
    Chromium binary on the test machine.  In production, this is the
    only backend in use; the dev backend is only ever in tests.
    """
    from agent.browser_os_playwright import PlaywrightBrowserOS  # type: ignore[import-not-found]

    return PlaywrightBrowserOS()


def browser_os_for_mode() -> BrowserOS:
    """The backend the current mode calls for.

    This is the ONLY place that decides, because the failure it replaces
    was three call sites each hardcoding ``InMemoryBrowserOS()`` — the
    Playwright backend existed, was documented as "the only backend in
    production", and nothing ever constructed it. Production browsed an
    HTML-string simulator: every navigate "succeeded" against nothing,
    which is worse than an error because it looks like work.

    Production does NOT fall back to the simulator when Playwright is
    missing. A browser capability that silently simulates is a lie the
    agent then repeats to the user; an ImportError at startup names the
    missing dependency and is fixed in one deploy.
    """
    import os

    from alex.config import get_settings

    # The local machine runs in dev mode (the exercised offline path)
    # but wants a REAL browser: ALEX_BROWSER=playwright opts in; the
    # no-silent-fallback rule above still holds.
    choice = os.environ.get("ALEX_BROWSER", "").strip().lower()
    if choice == "playwright":
        return make_production_browser_os()
    if choice == "memory" or get_settings().mode == "dev":
        return InMemoryBrowserOS()
    return make_production_browser_os()
