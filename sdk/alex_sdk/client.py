"""
The HTTP + streaming client for the Alex Databricks App.

Every method maps 1:1 onto a real route in ``app/app.py``. The client:

  * sends the OBO token as BOTH ``Authorization: Bearer ...`` and
    ``X-Forwarded-Access-Token`` (the app's identity check reads the
    forwarded header in production; dev mode also accepts the bearer),
  * applies connect/read timeouts to every call,
  * retries idempotent GETs (and explicitly-opted calls) on 429/5xx
    and connection errors with exponential backoff + jitter,
  * maps HTTP failures to typed exceptions (:class:`AuthError`,
    :class:`NotFoundError`, :class:`ClientError`, :class:`ServerError`)
    instead of leaking raw ``httpx`` exceptions,
  * streams chat turns as typed :class:`AgentEvent` objects with SSE
    keepalive/partial-line/done handling. The stream can emit
    ``text``, ``reasoning``, ``tool``, ``artifact``, ``law_block``,
    ``memory``, ``confirmation_required``, ``error``, and ``done``
    events (see :class:`EventType`),
  * mirrors the product's learning surface: :meth:`Alex.tools`
    (MCP-style manifest), :meth:`Alex.teach`, :meth:`Alex.feedback`,
    and :meth:`Session.correct` (teach-by-editing).

Example (dev, against a local app)::

    from alex_sdk import Alex

    with Alex.for_dev("http://localhost:8000", "alice@example.com") as alex:
        session = alex.session("hello")
        for ev in session.run("Summarize my inbox"):
            if ev.type == "text":
                print(ev.text, end="", flush=True)
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from collections.abc import Generator, Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import httpx

log = logging.getLogger("alex_sdk.client")

DEFAULT_TIMEOUT_S = 30.0
STREAM_READ_TIMEOUT_S = 300.0
DEFAULT_RETRIES = 3
RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


# ---------------------------------------------------------------------------
# Typed exceptions
# ---------------------------------------------------------------------------


class AlexError(Exception):
    """Base class for every error the SDK raises on purpose."""

    def __init__(self, message: str, *, status_code: int = 0, body: str = "") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class AuthError(AlexError):
    """401/403 — the token is missing, expired, or not authorized."""


class NotFoundError(AlexError):
    """404 — the resource does not exist (or is not yours)."""


class ClientError(AlexError):
    """Any other 4xx — the request itself is malformed."""


class ServerError(AlexError):
    """5xx — the app or an upstream dependency failed."""


class TransportError(AlexError):
    """The request never got an HTTP response (DNS, connect, timeout)."""


# Backwards-friendly alias requested by consumers who spell it NotFound.
NotFound = NotFoundError


def _raise_for_status(resp: httpx.Response) -> None:
    """Map an error response onto a typed SDK exception."""
    if resp.status_code < 400:
        return
    try:
        body = resp.text
    except Exception:  # noqa: BLE001 - body decoding must never mask the error
        body = ""
    detail = body[:500]
    msg = f"{resp.request.method} {resp.request.url.path} -> {resp.status_code}: {detail}"
    if resp.status_code in (401, 403):
        raise AuthError(msg, status_code=resp.status_code, body=body)
    if resp.status_code == 404:
        raise NotFoundError(msg, status_code=resp.status_code, body=body)
    if resp.status_code >= 500:
        raise ServerError(msg, status_code=resp.status_code, body=body)
    raise ClientError(msg, status_code=resp.status_code, body=body)


def auth_headers(token: str) -> dict[str, str]:
    """The single place auth headers are built.

    Both headers are required: the app's production identity check
    reads ``X-Forwarded-Access-Token`` (the OBO header), while dev
    mode reads ``Authorization: Bearer``.
    """
    if not token:
        return {}
    return {
        "Authorization": f"Bearer {token}",
        "X-Forwarded-Access-Token": token,
    }


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


class EventType(str, Enum):
    """The SSE event types the chat stream can emit.

    ``AgentEvent.type`` is a plain string on the wire; compare against
    these members (``ev.type == EventType.REASONING`` works because
    the enum subclasses ``str``). Unknown types pass through untouched.
    """

    TEXT = "text"
    REASONING = "reasoning"
    TOOL = "tool"
    ARTIFACT = "artifact"
    LAW_BLOCK = "law_block"
    MEMORY = "memory"
    CONFIRMATION_REQUIRED = "confirmation_required"
    ERROR = "error"
    DONE = "done"


@dataclass(frozen=True)
class ToolInfo:
    """One entry from the ``GET /api/tools`` MCP-style manifest."""

    name: str
    description: str = ""
    input_schema: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> ToolInfo:
        return cls(
            name=str(d.get("name") or ""),
            description=str(d.get("description") or ""),
            input_schema=dict(d.get("inputSchema") or d.get("input_schema") or {}),
        )


@dataclass
class AgentEvent:
    """Typed wrapper around one SSE event.

    ``payload`` is the raw JSON dict the server sent; the convenience
    fields (`text`, `tool`, `args`, `phase`, `article`, `detail`,
    `summary`, `body`, `path`, `code`, `message`, `token`) mirror the
    well-known keys from ``alex.events`` so consumers can branch on
    ``event.type`` without dict lookups.
    """

    type: str
    payload: dict = field(default_factory=dict)
    # Convenience fields
    text: str = ""
    tool: str = ""
    args: dict = field(default_factory=dict)
    phase: str = ""
    article: str = ""
    detail: str = ""
    summary: str = ""
    body: dict = field(default_factory=dict)
    path: str = ""
    code: str = ""
    message: str = ""
    token: str = ""
    # Private: the Alex client that produced this event, used by confirm().
    _alex: Alex | None = field(default=None, repr=False)
    _confirm_url: str = field(default="", repr=False)

    @classmethod
    def from_dict(
        cls,
        d: dict,
        *,
        client: Alex | None = None,
        confirm_url: str = "",
    ) -> AgentEvent:
        ev = cls(type=d.get("type", "text"), payload=d, _alex=client, _confirm_url=confirm_url)
        for k in (
            "text",
            "tool",
            "phase",
            "article",
            "detail",
            "summary",
            "body",
            "path",
            "code",
            "message",
            "token",
        ):
            v = d.get(k)
            if v is not None and isinstance(v, str | dict | int | float | bool):
                setattr(ev, k, v)
        if d.get("args") is not None and isinstance(d["args"], dict):
            ev.args = d["args"]
        return ev

    @property
    def data(self) -> dict:
        """The event's payload minus the ``type`` discriminator.

        Handy for the structured event types: for ``reasoning`` events
        ``ev.data["text"]`` is the reasoning line, for ``tool`` events
        ``ev.data`` carries ``tool``/``args``/``phase``, for
        ``law_block`` it carries ``article``/``detail``, and for
        ``memory`` it carries whatever the server recorded.
        """
        return {k: v for k, v in self.payload.items() if k != "type"}

    def confirm(self) -> dict | None:
        """Send the user's click for a ``confirmation_required`` event.

        Posts the real two-phase submit endpoint
        ``POST /api/browser/sessions/{sid}/tabs/{tid}/submit`` with the
        staged ``confirmation_token``. The session/tab ids are read
        from the event payload (top-level or inside ``body``).

        Returns the decoded JSON response, or None if the event
        cannot be confirmed (wrong type, no client, no token).
        """
        if self.type != "confirmation_required" or self._alex is None:
            return None
        if not self.token:
            log.warning("confirmation_required event has no token; cannot confirm")
            return None
        body = self.body if isinstance(self.body, dict) else {}
        if self._confirm_url:
            url = self._confirm_url
        else:
            sid = self.payload.get("session_id") or body.get("session_id")
            tid = self.payload.get("tab_id") or body.get("tab_id")
            if not sid or not tid:
                log.warning(
                    "confirmation_required event has no session_id/tab_id; cannot confirm"
                )
                return None
            url = f"/api/browser/sessions/{sid}/tabs/{tid}/submit"
        selector = body.get("selector") or "form"
        return self._alex._request(
            "POST",
            url,
            json={"selector": selector, "confirmation_token": self.token},
            ok_statuses=(403,),  # constitution block carries a JSON body
        )

    def cancel(self) -> None:
        """Cancel a staged action. The server expires unconfirmed stages
        automatically; this is a no-op for the client."""
        return None


# ---------------------------------------------------------------------------
# SSE parsing
# ---------------------------------------------------------------------------


def _iter_sse(resp: httpx.Response) -> Iterator[str]:
    """Yield decoded ``data:`` payloads from an SSE response.

    Robust against: multi-byte UTF-8 split across read boundaries,
    CRLF line endings, keepalive comment lines (``: keepalive``),
    ``data:`` with or without the space, and a trailing partial line.
    """
    buf = ""
    for chunk in resp.iter_bytes():
        buf += chunk.decode("utf-8", errors="replace")
        lines = buf.split("\n")
        # The last "line" may be incomplete; stash it for the next chunk.
        buf = lines[-1]
        for line in lines[:-1]:
            line = line.rstrip("\r")
            if not line or line.startswith(":"):
                continue  # blank separator or keepalive comment
            if line.startswith("data:"):
                yield line[5:].lstrip(" ").strip()
    # Flush a final complete-looking data line the server sent without
    # a trailing newline (some proxies truncate the last separator).
    tail = buf.rstrip("\r")
    if tail.startswith("data:"):
        yield tail[5:].lstrip(" ").strip()


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class Session:
    """A single chat session against ``POST /api/chat``.

    Example::

        session = alex.session("triage")
        for ev in session.run("What's urgent today?"):
            if ev.type == "text":
                print(ev.text, end="")
            elif ev.type == "confirmation_required":
                ev.confirm()   # user said yes
    """

    def __init__(self, alex: Alex, session_id: str) -> None:
        self.alex = alex
        self.id = session_id

    def run(
        self,
        prompt: str,
        model: str = "auto",
        show_reasoning: bool = False,
        *,
        custom_inputs: dict | None = None,
    ) -> Generator[AgentEvent, None, None]:
        """Stream one turn; yields :class:`AgentEvent` objects.

        With ``show_reasoning=True`` the server interleaves
        ``reasoning`` events with the ``text``/``tool``/``artifact``/
        ``law_block``/``memory`` stream; branch on ``ev.type`` (see
        :class:`EventType`) and use ``ev.data`` for the structured
        payload. ``custom_inputs`` is passed through verbatim (e.g.
        ``{"edited_output": ...}`` — see :meth:`correct`).

        The stream ends after the server's ``done`` event (which is
        always yielded last) or when the generator is closed. Errors
        before the stream starts raise typed SDK exceptions.
        """
        body: dict[str, Any] = {
            "session_id": self.id,
            "model": model,
            "show_reasoning": show_reasoning,
            "text": prompt,
        }
        if custom_inputs:
            body["custom_inputs"] = custom_inputs
        try:
            with self.alex._http.stream(
                "POST",
                "/api/chat",
                json=body,
                timeout=httpx.Timeout(
                    connect=10.0,
                    read=STREAM_READ_TIMEOUT_S,
                    write=10.0,
                    pool=10.0,
                ),
            ) as resp:
                if resp.status_code >= 400:
                    resp.read()
                    _raise_for_status(resp)
                for raw in _iter_sse(resp):
                    ev = self.alex._parse(raw)
                    if ev is None:
                        continue
                    yield ev
                    if ev.type == "done":
                        return  # graceful close; server is finished
        except httpx.HTTPError as e:
            raise TransportError(f"chat stream failed: {e}") from e

    def correct(
        self,
        edited_output: str,
        original_output: str = "",
        *,
        prompt: str = "I edited your previous output; please learn from the correction.",
        model: str = "auto",
    ) -> Generator[AgentEvent, None, None]:
        """Teach-by-editing: post a follow-up chat turn carrying the
        user's edited version of a prior AI draft.

        The server's learn-with-you loop captures the diff between
        ``original_output`` and ``edited_output`` as a preference
        (never a self-granted skill). Streams the follow-up turn like
        :meth:`run`; iterate it (or drain it) to send the request::

            for ev in session.correct(fixed_text, original_output=draft):
                ...
        """
        custom_inputs: dict[str, Any] = {"edited_output": edited_output}
        if original_output:
            custom_inputs["original_output"] = original_output
        yield from self.run(prompt, model=model, custom_inputs=custom_inputs)

    def messages(self, *, limit: int = 200) -> list[dict]:
        """``GET /api/sessions/{id}/messages`` — the persisted turns of
        this conversation (role + content, in order), so a fresh client
        can resume a session without the browser's local state.

        Example::

            for m in alex.session("s-42").messages():
                print(m["role"], m["content"])
        """
        d = self.alex._request(
            "GET", f"/api/sessions/{self.id}/messages", params={"limit": limit}
        )
        return d.get("messages", []) if isinstance(d, dict) else []

    def subagent(
        self, name: str, prompt: str, inherit: tuple = ("workspace",), model: str = "auto"
    ) -> str:
        """NOT IMPLEMENTED: the app backend exposes no ``POST
        /api/subagent`` route. This raises rather than failing over
        HTTP; run subagents through the agent runtime instead."""
        raise NotImplementedError(
            "Session.subagent requires the app backend route "
            "'POST /api/subagent', which does not exist. "
            "Run subagents through the agent runtime instead."
        )


# ---------------------------------------------------------------------------
# Alex — the main client
# ---------------------------------------------------------------------------


class Alex:
    """The main SDK client. One method per real app route.

    The base URL comes from ``ALEX_APP_URL`` or the ``base_url``
    argument. The OBO token comes from the ``token`` argument or the
    ``DATABRICKS_TOKEN`` / ``ALEX_OBO_TOKEN`` env vars, and is sent as
    both ``Authorization`` and ``X-Forwarded-Access-Token``.

    Dev quickstart::

        alex = Alex.for_dev("http://localhost:8000", "alice@example.com")
        print(alex.me())
    """

    def __init__(
        self,
        base_url: str = "",
        token: str = "",
        timeout_seconds: float = DEFAULT_TIMEOUT_S,
        *,
        retries: int = DEFAULT_RETRIES,
        backoff_base_s: float = 0.25,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        base = base_url or os.environ.get("ALEX_APP_URL", "http://localhost:8000")
        self._base_url = base
        token = (
            token or os.environ.get("DATABRICKS_TOKEN") or os.environ.get("ALEX_OBO_TOKEN") or ""
        )
        self._token = token
        self._retries = max(0, retries)
        self._backoff_base_s = backoff_base_s
        self._http = httpx.Client(
            base_url=base,
            headers=auth_headers(token),
            timeout=httpx.Timeout(connect=10.0, read=timeout_seconds, write=10.0, pool=10.0),
            transport=transport,
        )
        self._hooks: dict[str, list] = {}

    # ---- constructors -------------------------------------------------------

    @classmethod
    def for_dev(
        cls,
        base_url: str = "http://localhost:8000",
        email: str = "dev@example.com",
        *,
        scopes: str = "",
        **kwargs: Any,
    ) -> Alex:
        """Build a dev client by fetching ``GET /api/dev/token`` from a
        locally running app (dev mode only; the route 404s in prod).

        Example::

            alex = Alex.for_dev("http://localhost:8000", "alice@example.com")
        """
        transport = kwargs.get("transport")
        params: dict[str, str] = {"email": email}
        if scopes:
            params["scopes"] = scopes
        with httpx.Client(
            base_url=base_url,
            timeout=httpx.Timeout(connect=10.0, read=15.0, write=10.0, pool=10.0),
            transport=transport,
        ) as c:
            try:
                r = c.get("/api/dev/token", params=params)
            except httpx.HTTPError as e:
                raise TransportError(f"GET /api/dev/token failed: {e}") from e
            _raise_for_status(r)
            token = str(r.json().get("token") or "")
        if not token:
            raise AuthError("/api/dev/token returned no token (is the app in dev mode?)")
        return cls(base_url=base_url, token=token, **kwargs)

    @staticmethod
    def dev_token(email: str, *, ttl_seconds: int = 3600) -> str:
        """Mint a dev JWT locally (requires the ``alex`` package on the
        path). Prefer :meth:`for_dev`, which asks the running app."""
        from alex.identity import mint_dev_token

        return mint_dev_token(email, ttl_seconds=ttl_seconds)

    # ---- transport core -----------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: Any | None = None,
        params: dict | None = None,
        files: dict | None = None,
        retry: bool | None = None,
        ok_statuses: tuple[int, ...] = (),
        raw: bool = False,
    ) -> Any:
        """One HTTP call with bounded retries and typed error mapping.

        Retries (backoff + jitter) fire on 429/5xx and connect errors,
        for GETs by default; pass ``retry=True`` to opt a call in or
        ``retry=False`` to opt out. ``ok_statuses`` lists error codes
        that carry a meaningful JSON body (e.g. a constitution block)
        and should be returned instead of raised. ``raw=True`` returns
        the :class:`httpx.Response` instead of decoded JSON.
        """
        do_retry = (method.upper() == "GET") if retry is None else retry
        attempts = (self._retries + 1) if do_retry else 1
        last_exc: Exception | None = None
        for attempt in range(attempts):
            try:
                resp = self._http.request(method, path, json=json, params=params, files=files)
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as e:
                last_exc = TransportError(f"{method} {path} failed: {e}")
                last_exc.__cause__ = e
                if attempt + 1 < attempts:
                    self._sleep_backoff(attempt)
                    continue
                raise last_exc from e
            except httpx.HTTPError as e:
                raise TransportError(f"{method} {path} failed: {e}") from e
            if resp.status_code in RETRY_STATUS and attempt + 1 < attempts:
                self._sleep_backoff(attempt)
                continue
            if resp.status_code in ok_statuses:
                return resp if raw else resp.json()
            _raise_for_status(resp)
            return resp if raw else (resp.json() if resp.content else {})
        raise last_exc if last_exc else ServerError(f"{method} {path}: retries exhausted")

    def _sleep_backoff(self, attempt: int) -> None:
        delay = self._backoff_base_s * (2**attempt) + random.uniform(0, self._backoff_base_s)
        if delay > 0:
            time.sleep(delay)

    # ---- identity & health ----------------------------------------------

    def healthz(self) -> dict:
        """``GET /healthz`` — liveness probe."""
        return self._request("GET", "/healthz")

    def me(self) -> dict:
        """``GET /api/me`` — resolved identity: email, user_id, scopes."""
        return self._request("GET", "/api/me")

    # ---- sessions & chat --------------------------------------------------

    def session(self, title: str = "sdk session") -> Session:
        """``POST /api/sessions`` — create a NEW chat session and return a
        handle bound to its id. Use :meth:`resume` to reattach to an
        existing session."""
        d = self._request("POST", "/api/sessions", json={"title": title})
        return Session(self, d["session_id"])

    def resume(self, session_id: str) -> Session:
        """Return a :class:`Session` bound to an EXISTING session id
        (no new session is created). Combine with
        :meth:`Session.messages` to reload the conversation server-side::

            s = alex.resume("s-42")
            history = s.messages()
            for ev in s.run("and then?"):
                ...
        """
        return Session(self, session_id)

    def sessions(self) -> list[dict]:
        """``GET /api/sessions`` — the caller's sessions."""
        return list(self._request("GET", "/api/sessions").get("sessions", []))

    # ---- workspace surface -------------------------------------------------

    def inbox(self, limit: int = 50) -> list[dict]:
        """``GET /api/inbox`` — prioritized inbox view."""
        return list(self._request("GET", "/api/inbox", params={"limit": limit}).get("inbox", []))

    def tasks(self, status: str | None = None) -> list[dict]:
        """``GET /api/tasks`` — task board, optionally filtered by status."""
        params = {"status": status} if status else None
        return list(self._request("GET", "/api/tasks", params=params).get("tasks", []))

    def briefs(self) -> dict:
        """``GET /api/briefs`` — today's morning brief + standing orders."""
        return self._request("GET", "/api/briefs")

    def files(self) -> dict:
        """``GET /api/files/list`` — uploads/work/outputs listings."""
        return self._request("GET", "/api/files/list")

    def upload(self, filename: str, content: bytes) -> dict:
        """``POST /api/files/upload`` — store a file in the workspace."""
        return self._request(
            "POST",
            "/api/files/upload",
            files={"file": (filename, content)},
        )

    def download(self, path: str) -> bytes:
        """``GET /api/files/download`` — fetch an ``outputs/`` artifact."""
        resp = self._request("GET", "/api/files/download", params={"path": path}, raw=True)
        return bytes(resp.content)

    # ---- skills -------------------------------------------------------------

    def skills(self) -> list[dict]:
        """``GET /api/skills`` — the caller's proposed + active skills."""
        return list(self._request("GET", "/api/skills").get("skills", []))

    def decide_skill(self, skill_id: str, action: str = "ratify") -> dict:
        """``POST /api/skills/decide`` — ratify or reject a proposed
        skill. ``action`` is ``"ratify"`` or ``"reject"``."""
        return self._request(
            "POST", "/api/skills/decide", json={"skill_id": skill_id, "action": action}
        )

    # ---- transparency ---------------------------------------------------

    def transparency_memory(self) -> list[dict]:
        """``GET /api/transparency/memory`` — the memory mirror."""
        return list(self._request("GET", "/api/transparency/memory").get("memories", []))

    def transparency_graph(self) -> list[dict]:
        """``GET /api/transparency/graph`` — the knowledge-graph mirror."""
        return list(self._request("GET", "/api/transparency/graph").get("edges", []))

    def forget(self, item_id: str, reason: str = "user_requested") -> dict:
        """``POST /api/transparency/forget`` — queue a forget request."""
        return self._request(
            "POST", "/api/transparency/forget", json={"item_id": item_id, "reason": reason}
        )

    def capabilities(self) -> dict:
        """``GET /api/transparency/capabilities`` — effective ACL +
        active delegations."""
        return self._request("GET", "/api/transparency/capabilities")

    def actions(self) -> list[dict]:
        """``GET /api/transparency/actions`` — the audit trail."""
        return list(self._request("GET", "/api/transparency/actions").get("actions", []))

    def library(self) -> list[dict]:
        """``GET /api/transparency/library`` — approved instance files."""
        return list(self._request("GET", "/api/transparency/library").get("library", []))

    def constitution(self) -> list[dict]:
        """``GET /api/constitution`` — active constitutional articles."""
        return list(self._request("GET", "/api/constitution").get("articles", []))

    # ---- tools & learning ---------------------------------------------------

    def tools(self) -> list[ToolInfo]:
        """``GET /api/tools`` — the MCP-style tool manifest, ACL-filtered
        to what the signed-in user may actually call. Each entry is a
        :class:`ToolInfo` with ``name``, ``description``, and
        ``input_schema`` (the manifest's ``inputSchema``)."""
        d = self._request("GET", "/api/tools")
        return [ToolInfo.from_dict(t) for t in d.get("tools", []) if isinstance(t, dict)]

    def call_tool(self, name: str, **args: Any) -> Any:
        """``POST /api/tools/{name}`` — invoke a single governed tool
        directly (the MCP call-tool seam), bypassing the chat loop.

        The call is ACL-checked and constitution-gated server-side just
        like the agent's own tool use; identity is injected from the
        session. Returns the tool's ``result`` payload.

        Example::

            alex.call_tool("create_spreadsheet", filename="q1",
                           headers=["Month", "Rev"], rows=[["Jan", 100]])
        """
        d = self._request("POST", f"/api/tools/{name}", json={"args": args})
        return d.get("result", d)

    def teach(self, prompt: str, *, context: str = "") -> dict:
        """``POST /api/teach`` — explicitly walk the agent through a
        routine. Stored as a taught-routine memory; the nightly
        reflection job may propose it as a skill (never self-ratified).
        Returns ``{"routine_id": ..., "status": "stored"}``."""
        body: dict[str, Any] = {"prompt": prompt}
        if context:
            body["context"] = context
        return self._request("POST", "/api/teach", json=body)

    FEEDBACK_VERDICTS = ("helpful", "unhelpful", "inaccurate", "other")

    def feedback(self, suggestion_id: str, verdict: str, *, comment: str = "") -> dict:
        """``POST /api/feedback`` — thumbs on a suggestion. ``verdict``
        must be one of ``helpful | unhelpful | inaccurate | other``
        (validated client-side to fail fast)."""
        if verdict not in self.FEEDBACK_VERDICTS:
            raise ValueError(
                f"verdict must be one of {'|'.join(self.FEEDBACK_VERDICTS)}, got {verdict!r}"
            )
        body: dict[str, Any] = {"suggestion_id": suggestion_id, "feedback": verdict}
        if comment:
            body["comment"] = comment
        return self._request("POST", "/api/feedback", json=body)

    # ---- SDK memory/graph/workflow routes ------------------------------------

    def remember(
        self, content: str, *, kind: str = "fact", salience: float = 0.5, source_ref: str = "sdk"
    ) -> dict:
        """``POST /api/sdk/remember`` — store a memory bead."""
        return self._request(
            "POST",
            "/api/sdk/remember",
            json={"content": content, "kind": kind, "salience": salience, "source_ref": source_ref},
        )

    def recall(self, query: str, *, k: int = 10) -> list[dict]:
        """``POST /api/sdk/recall`` — top-k memories for ``query``."""
        d = self._request("POST", "/api/sdk/recall", json={"query": query, "k": k}, retry=True)
        return list(d.get("memories", []))

    def add_edge(
        self, src: str, rel: str, dst: str, *, evidence: str = "sdk", confidence: float = 0.7
    ) -> dict:
        """``POST /api/sdk/add_edge`` — add a knowledge-graph edge."""
        return self._request(
            "POST",
            "/api/sdk/add_edge",
            json={"src": src, "rel": rel, "dst": dst, "evidence": evidence, "confidence": confidence},
        )

    def graph_query(self, entity: str = "") -> list[dict]:
        """``POST /api/sdk/graph_query`` — edges touching ``entity``."""
        d = self._request("POST", "/api/sdk/graph_query", json={"entity": entity}, retry=True)
        return list(d.get("edges", []))

    def list_workflows(self) -> list[dict]:
        """``GET /api/sdk/list_workflows`` — the active workflow catalog."""
        return list(self._request("GET", "/api/sdk/list_workflows").get("workflows", []))

    def memory(self, owner_email: str = "") -> Any:
        """Return a bound :class:`alex_sdk.memory.Memory` helper. If
        ``owner_email`` is omitted it is resolved via ``GET /api/me``."""
        from alex_sdk.memory import Memory

        email = owner_email or str(self.me().get("email") or "")
        return Memory(self, email)

    # ---- supervisor -----------------------------------------------------

    def supervisor_reports(self) -> list[dict]:
        """``GET /api/supervisor/reports`` — the caller's reports."""
        return list(self._request("GET", "/api/supervisor/reports").get("reports", []))

    def delegate(
        self,
        report_email: str,
        capability: str,
        *,
        justification: str = "",
        days: int = 30,
        second_approver: str | None = None,
    ) -> dict:
        """``POST /api/supervisor/delegate`` — grant a capability to a
        report (dual-approval rules enforced server-side)."""
        return self._request(
            "POST",
            "/api/supervisor/delegate",
            json={
                "report_email": report_email,
                "capability": capability,
                "justification": justification,
                "days": days,
                "second_approver": second_approver,
            },
        )

    def revoke(self, delegation_id: str) -> dict:
        """``POST /api/supervisor/revoke`` — revoke a delegation."""
        return self._request(
            "POST", "/api/supervisor/revoke", json={"delegation_id": delegation_id}
        )

    # ---- browser OS -----------------------------------------------------

    def browser_sessions(self) -> list[dict]:
        """``GET /api/browser/sessions`` — the caller's browser sessions."""
        return list(self._request("GET", "/api/browser/sessions").get("sessions", []))

    def browser_create_session(self) -> str:
        """``POST /api/browser/sessions`` — returns the new session_id."""
        return str(self._request("POST", "/api/browser/sessions")["session_id"])

    def browser_close_session(self, session_id: str) -> dict:
        """``DELETE /api/browser/sessions/{sid}``."""
        return self._request("DELETE", f"/api/browser/sessions/{session_id}")

    def browser_tabs(self, session_id: str) -> list[dict]:
        """``GET /api/browser/sessions/{sid}/tabs``."""
        return list(
            self._request("GET", f"/api/browser/sessions/{session_id}/tabs").get("tabs", [])
        )

    def browser_create_tab(self, session_id: str, url: str = "about:blank") -> dict:
        """``POST /api/browser/sessions/{sid}/tabs`` (``url`` is a query
        parameter on this route)."""
        return self._request(
            "POST", f"/api/browser/sessions/{session_id}/tabs", params={"url": url}
        )

    def browser_close_tab(self, session_id: str, tab_id: str) -> dict:
        """``DELETE /api/browser/sessions/{sid}/tabs/{tid}``."""
        return self._request("DELETE", f"/api/browser/sessions/{session_id}/tabs/{tab_id}")

    def _browser_op(self, session_id: str, tab_id: str, op: str, body: dict) -> dict:
        """POST a tab operation. Constitution blocks come back as 403
        with a JSON body; we return that body instead of raising so
        the caller can inspect ``ok`` / ``law_block`` details."""
        return self._request(
            "POST",
            f"/api/browser/sessions/{session_id}/tabs/{tab_id}/{op}",
            json=body,
            ok_statuses=(403,),
        )

    def browser_navigate(self, session_id: str, tab_id: str, url: str) -> dict:
        """``POST .../tabs/{tid}/navigate``."""
        return self._browser_op(session_id, tab_id, "navigate", {"url": url})

    def browser_click(self, session_id: str, tab_id: str, selector: str) -> dict:
        """``POST .../tabs/{tid}/click``."""
        return self._browser_op(session_id, tab_id, "click", {"selector": selector})

    def browser_type(self, session_id: str, tab_id: str, selector: str, text: str) -> dict:
        """``POST .../tabs/{tid}/type``."""
        return self._browser_op(session_id, tab_id, "type", {"selector": selector, "text": text})

    def browser_extract(self, session_id: str, tab_id: str, selector: str = "body") -> dict:
        """``POST .../tabs/{tid}/extract``."""
        return self._browser_op(session_id, tab_id, "extract", {"selector": selector})

    def browser_screenshot(self, session_id: str, tab_id: str) -> dict:
        """``POST .../tabs/{tid}/screenshot``."""
        return self._browser_op(session_id, tab_id, "screenshot", {})

    def browser_submit(
        self,
        session_id: str,
        tab_id: str,
        selector: str = "form",
        confirmation_token: str | None = None,
    ) -> dict:
        """``POST .../tabs/{tid}/submit`` — two-phase form submit.

        First call (no token): stages the submit and returns
        ``{"awaiting_confirmation": true, "confirmation_token": ...}``.
        Second call (with the token): executes it. Example::

            r1 = alex.browser_submit(sid, tid, "form#send")
            tok = r1["confirmation_token"]
            r2 = alex.browser_submit(sid, tid, "form#send", confirmation_token=tok)
        """
        body: dict = {"selector": selector}
        if confirmation_token is not None:
            body["confirmation_token"] = confirmation_token
        return self._browser_op(session_id, tab_id, "submit", body)

    def browser_mirror(self, session_id: str, tab_id: str, mode: str) -> dict:
        """``POST .../tabs/{tid}/mirror`` — set the user-control mode:
        ``read`` | ``co-drive`` | ``yield``."""
        return self._browser_op(session_id, tab_id, "mirror", {"mode": mode})

    def browser_stream(self, session_id: str) -> Generator[AgentEvent, None, None]:
        """``GET /api/browser/sessions/{sid}/stream`` — the session's
        SSE event stream as :class:`AgentEvent` objects. Keepalive
        comments are consumed silently; close the generator to stop."""
        try:
            with self._http.stream(
                "GET",
                f"/api/browser/sessions/{session_id}/stream",
                timeout=httpx.Timeout(
                    connect=10.0, read=STREAM_READ_TIMEOUT_S, write=10.0, pool=10.0
                ),
            ) as resp:
                if resp.status_code >= 400:
                    resp.read()
                    _raise_for_status(resp)
                for raw in _iter_sse(resp):
                    ev = self._parse(raw)
                    if ev is not None:
                        yield ev
        except httpx.HTTPError as e:
            raise TransportError(f"browser stream failed: {e}") from e

    # ---- extension points --------------------------------------------------

    def tool(self, capability: str):
        """NOT IMPLEMENTED: the app backend exposes no
        ``POST /api/sdk/register_tool`` route, so this raises at
        decoration time. Use ``alex_sdk.core.tool`` for local
        declarations."""

        def deco(fn):
            raise NotImplementedError(
                f"Alex.tool cannot register '{fn.__name__}': the app backend "
                "route 'POST /api/sdk/register_tool' does not exist. "
                "Use alex_sdk.core.tool for local declarations."
            )

        return deco

    def hook(self, event: str):
        """Subscribe to a client-side event. ``pre_event`` runs before
        any streamed event is delivered; ``pre_tool``/``post_tool``
        bracket tool calls. Hooks may observe or veto (raise) but never
        widen the agent's authority."""
        if event not in {"pre_event", "pre_tool", "post_tool"}:
            raise ValueError(f"unknown hook: {event}")

        def deco(fn):
            self._hooks.setdefault(event, []).append(fn)
            return fn

        return deco

    # ---- internal -----------------------------------------------------------

    def _parse(self, raw: str) -> AgentEvent | None:
        """Parse one SSE ``data:`` payload into an AgentEvent."""
        if not raw:
            return None
        if raw == "[DONE]":
            return AgentEvent(type="done")
        try:
            d = json.loads(raw)
        except json.JSONDecodeError:
            return AgentEvent(type="text", text=raw, payload={"text": raw})
        if not isinstance(d, dict):
            return AgentEvent(type="text", text=raw, payload={"text": raw})
        ev = AgentEvent.from_dict(d, client=self, confirm_url="")
        for h in self._hooks.get("pre_event", []):
            h(ev)
        return ev

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> Alex:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
