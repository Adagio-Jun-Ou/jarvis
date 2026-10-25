"""
Alex Databricks App backend — FastAPI.

The app serves the React SPA, proxies chat to the agent endpoint with
on-behalf-of-user auth, streams tokens back as JSON events
(per `alex.events`), and exposes a thin set of transparency + supervisor
APIs.

Security properties (every one enforced here):
  * Identity is derived from the OBO JWT in `X-Forwarded-Access-Token`.
    The unsigned `X-Forwarded-Email` header is logged for debugging
    but NEVER used for SQL row filtering or ACL checks.
  * Every SQL call uses parameter binding (`parameters=...`). No
    f-string interpolation of user data into SQL anywhere in this file.
  * SSE responses set `Cache-Control: no-cache` and
    `X-Accel-Buffering: no` so proxies don't buffer the stream.
  * Streaming generators wrap the upstream iterator in try/except so
    the client always receives a final `done` or `error` event.
  * All write operations that could be destructive go through
    `safe_path_join` which uses `os.path.realpath` to detect escapes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import time
import urllib.parse
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import httpx
from fastapi import FastAPI, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from alex.errors import AlexError
from alex.events import (
    done_event,
    error_event,
    text_event,
)
from alex.identity import (
    Identity,
    IdentityError,
    identity_from_headers,
    mint_dev_token,
)
from alex.testing import is_dev

log = logging.getLogger("app.app")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(app: FastAPI):
    if is_dev():
        log.warning(
            "Alex is running in DEV mode. Use ALEX_ENV=production "
            "for real deployments."
        )
    # Jarvis's native tools join the governed registry before any VM is
    # built. Best-effort: a missing optional dependency costs one family.
    try:
        from agent.jarvis_tools import register_jarvis_tools

        registered = register_jarvis_tools()
        log.info("jarvis tools registered: %d", len(registered))
    except Exception:  # noqa: BLE001
        log.warning("jarvis tool registration failed", exc_info=True)
    _recover_interrupted_runs()
    # The dispatcher loop: every interval, sweep for runs whose process
    # died (recover) and re-start the interrupted ones (dispatch, with
    # its attempt circuit-breaker). This is what makes a task on the
    # board durable WORK rather than durable data — a crash costs one
    # sweep interval, not a person noticing. 0 disables.
    try:
        interval = float(os.environ.get("ALEX_DISPATCH_INTERVAL_S", "120"))
    except ValueError:
        interval = 120.0
    stopping = asyncio.Event()
    sweeper = None
    if interval > 0:
        async def _sweep_loop() -> None:
            from agent import taskboard

            while not stopping.is_set():
                try:
                    await asyncio.wait_for(stopping.wait(), timeout=interval)
                    break
                except (TimeoutError, asyncio.TimeoutError):
                    pass
                try:
                    _recover_interrupted_runs()
                    restarted = await run_in_threadpool(
                        taskboard.dispatch_once, _board_sql
                    )
                    if restarted:
                        log.warning(
                            "dispatcher restarted %d interrupted task(s): %s",
                            len(restarted), ", ".join(restarted),
                        )
                except Exception:  # noqa: BLE001
                    log.debug("dispatch sweep failed", exc_info=True)
                try:
                    # Standing automation: fire schedule-triggered orders
                    # that have come due (collapse-on-catch-up; capped
                    # per sweep).
                    from agent import standing_orders

                    fired = await run_in_threadpool(
                        standing_orders.run_due, _board_sql
                    )
                    if fired:
                        log.warning(
                            "scheduler fired %d standing order(s): %s",
                            len(fired), ", ".join(fired),
                        )
                except Exception:  # noqa: BLE001
                    log.debug("schedule sweep failed", exc_info=True)

        sweeper = asyncio.create_task(_sweep_loop())
    try:
        yield
    finally:
        stopping.set()
        if sweeper is not None:
            sweeper.cancel()


def _recover_interrupted_runs() -> None:
    """Close out agent runs abandoned by a process that went away.

    A run is driven by a thread in one process. If that process dies, the
    work stops but the row stays at 'running' — the oversight console
    shows in-flight work that will never finish and the task sits in
    `in_progress` with nobody aware it needs doing again. Recovery is
    keyed on a heartbeat rather than a shutdown hook, because the case
    that matters is the one where there was no clean shutdown.

    Safe on every boot and with several replicas: a run whose heartbeat
    is still fresh belongs to a live process and is left alone.
    """
    from agent import standing_orders, taskboard

    recovered: list = []
    for name, recover in (
        ("task", taskboard.recover_interrupted_runs),
        ("standing order", standing_orders.recover_interrupted_runs),
    ):
        try:
            found = recover(_board_sql)
        except Exception:  # noqa: BLE001
            # Never block startup on this. An uncollected run is a visible
            # nuisance; an app that will not boot is an outage.
            log.debug("%s run recovery failed", name, exc_info=True)
            continue
        recovered.extend(found)
        if found:
            log.warning("recovered %d interrupted %s run(s)", len(found), name)
    return recovered


app = FastAPI(title="Alex Workspace", lifespan=_lifespan)
AGENT_ENDPOINT = os.environ.get("AGENT_ENDPOINT", "alex-agent")


# ---------------------------------------------------------------------------
# Security headers (apply to every response, no exceptions)
# ---------------------------------------------------------------------------


@app.middleware("http")
async def security_headers(request: Request, call_next):
    try:
        resp = await call_next(request)
    except AlexError as e:
        # Known cross-boundary errors carry their own HTTP status (Denied ->
        # 403, NotFound -> 404, BadRequest -> 400, UpstreamError -> 502)
        # rather than collapsing to an opaque 500.
        if e.http_status >= 500:
            log.exception("alex error (%s)", e.code)
        resp = JSONResponse(e.to_dict(), status_code=e.http_status)
    except Exception:  # pragma: no cover
        log.exception("unhandled error")
        resp = JSONResponse({"error": "internal_error"}, status_code=500)
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault(
        "Strict-Transport-Security", "max-age=31536000; includeSubDomains; preload"
    )
    # The CSP is restrictive: no inline scripts, no external resources.
    # The frontend is built with hashed asset references; we don't need
    # 'unsafe-inline' for the React app because the production build
    # uses a single bundled JS file. We allow `self` only.
    resp.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; script-src 'self'; "
        "style-src 'self'; img-src 'self' data:; "
        "connect-src 'self'; frame-ancestors 'none'; "
        "object-src 'none'; base-uri 'self'",
    )
    resp.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    resp.headers.setdefault(
        "Permissions-Policy", "geolocation=(), camera=(), microphone=()"
    )
    return resp


# ---------------------------------------------------------------------------
# Identity helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Per-user rate limiting for the expensive routes. The AI Gateway
# limits the LLM; this limits the APP — a runaway client (a retry loop,
# a script gone wrong) must not be able to occupy every worker with
# agent turns. Sliding window, in-process (per replica, which is the
# honest scope of an in-process guard), generous by default.
# ---------------------------------------------------------------------------

_RATE_WINDOW_S = float(os.environ.get("ALEX_RATE_WINDOW_S", "300"))
_RATE_LIMIT = int(os.environ.get("ALEX_RATE_LIMIT", "60"))
_RATE: dict[str, list[float]] = {}
_RATE_LOCK = threading.Lock()


def _rate_ok(email: str) -> bool:
    if _RATE_LIMIT <= 0:
        return True
    now = time.monotonic()
    with _RATE_LOCK:
        recent = [t for t in _RATE.get(email, []) if now - t < _RATE_WINDOW_S]
        if len(recent) >= _RATE_LIMIT:
            _RATE[email] = recent
            return False
        recent.append(now)
        _RATE[email] = recent
        # Evict idle users so the dict cannot grow one entry per user
        # for the replica's lifetime.
        if len(_RATE) > 512:
            for k in [
                k for k, ts in _RATE.items()
                if not ts or now - ts[-1] >= _RATE_WINDOW_S
            ]:
                _RATE.pop(k, None)
        return True


def _identity(request: Request) -> Identity:
    """Resolve the request's identity from the OBO JWT. Raises 401 on failure."""
    try:
        return identity_from_headers(dict(request.headers))
    except IdentityError as e:
        from fastapi import HTTPException

        raise HTTPException(status_code=401, detail=f"identity: {e}") from e


# ---------------------------------------------------------------------------
# Warehouse — production wrapper or dev in-memory engine.
# ---------------------------------------------------------------------------


def _sql(stmt: str, params: dict) -> list[dict]:
    """Run a parameterized query.

    In dev: routes to the in-memory warehouse registered in fakes.
    In production: routes to the Databricks SDK's statement-execution
    endpoint with parameter binding.
    """
    if is_dev():
        wh = _get_dev_warehouse()
        from app._dev_seed import seed_dev_warehouse

        seed_dev_warehouse(wh)
        r = wh.execute(stmt, params or {})
        return r.as_dicts()
    return _real_warehouse().execute(stmt, params or {})


def _get_dev_warehouse():
    from alex.testing import get_fakes

    return get_fakes().warehouse


def _real_warehouse():
    global _warehouse
    if _warehouse is None:
        _warehouse = _Warehouse()
    return _warehouse


class _Warehouse:
    def __init__(self) -> None:
        self._client = None

    def execute(self, stmt: str, params: dict) -> list[dict]:
        from databricks.sdk import WorkspaceClient

        if self._client is None:
            self._client = WorkspaceClient()
        warehouse_id = os.environ.get("WAREHOUSE_ID", "")
        if not warehouse_id:
            raise RuntimeError("WAREHOUSE_ID is not set")
        try:
            r = self._client.statement_execution.execute_statement(
                warehouse_id=warehouse_id,
                statement=stmt,
                parameters=_normalize_params(params),
                wait_timeout="60s",
            )
        except Exception as e:
            raise RuntimeError(f"warehouse failed: {e}") from e
        state = getattr(getattr(r, "status", None), "state", None)
        if state and str(state).endswith("FAILED"):
            msg = getattr(r.status, "message", "unknown")
            raise RuntimeError(f"warehouse FAILED: {msg}")
        cols = [
            c.name
            for c in (
                r.manifest.schema.columns if r.manifest and r.manifest.schema else []
            )
        ]
        rows = list(r.result.data_array) if r.result and r.result.data_array else []
        return [dict(zip(cols, row, strict=False)) for row in rows]


def _normalize_params(params: dict) -> dict:
    """Spark SQL parameter format: {name: {value, type}}."""
    out: dict = {}
    for k, v in params.items():
        if v is None:
            out[k] = {"value": None, "type": "NULL"}
        elif isinstance(v, bool):
            out[k] = {"value": "true" if v else "false", "type": "BOOLEAN"}
        elif isinstance(v, int):
            out[k] = {"value": v, "type": "BIGINT"}
        elif isinstance(v, float):
            out[k] = {"value": v, "type": "DOUBLE"}
        elif isinstance(v, (list, tuple)):
            out[k] = {"value": list(v), "type": "ARRAY"}
        else:
            out[k] = {"value": str(v), "type": "STRING"}
    return out


_warehouse = None


def _sql_as_user(request: Request, stmt: str, params: dict) -> list[dict]:
    """Run a parameterized query. The caller MUST pass `user_email` in
    `params`; the query is responsible for row-filtering. No f-string
    SQL anywhere on this path."""
    return _sql(stmt, params)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/healthz")
async def healthz():
    """Liveness probe. The bundle's `health.path` points here."""
    return {"status": "ok"}


@app.get("/api/me")
async def me(request: Request):
    """Return the resolved identity. The frontend uses this to render
    the user's name and to gate UI elements."""
    ident = _identity(request)
    from alex.llm import DEFAULT_CLOUD_MODEL

    return {
        "email": ident.email,
        "user_id": ident.user_id,
        "scopes": list(ident.scopes),
        "model": DEFAULT_CLOUD_MODEL,
    }


# Dev helper: mint a JWT for local testing. NEVER exposed in production.
@app.get("/api/dev/token")
async def dev_token(email: str, scopes: str = ""):
    if not is_dev():
        return JSONResponse({"error": "dev-only endpoint"}, status_code=404)
    s = scopes or "serving.serving-endpoints sql files.files"
    return {"token": mint_dev_token(email, scopes=s.split())}


# ---------------------------------------------------------------------------
# Chat — proxy to the agent endpoint with OBO token, stream JSON events
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    session_id: str
    messages: list[dict] = Field(default_factory=list)
    model: str = "auto"
    show_reasoning: bool = False
    # Plan mode: the agent reads and plans, and can change nothing —
    # mutating tools are not exposed to the model at all and are denied
    # at call time as well (agent/vm.py plan_mode). The OpenCode-style
    # plan/build split, enforced structurally.
    plan_mode: bool = False
    # Convenience: the app also accepts `text` (last user message) so the
    # frontend can post a single turn without building a list.
    text: str | None = None
    # Optional extras: `edited_output` (+ `original_output`) feeds the
    # learn-with-you correction capture; `show_reasoning` may also be
    # passed here for parity with the serving-endpoint contract.
    custom_inputs: dict = Field(default_factory=dict)


def _store_preference(user_email: str, kind: str, content: str, source_ref: str):
    """`store_fn` for agent.learn_with_you — writes to the owner's
    vector-memory mirror (same table the transparency panel reads)."""
    _sql(
        """
        INSERT INTO media.control.vector_memory_mirror
          (id, owner_email, kind, content, salience, source_ref, created_at, superseded_by)
        VALUES (:id, :u, :k, :c, 0.6, :ref, current_timestamp(), NULL)
    """,
        {
            "id": str(uuid.uuid4()),
            "u": user_email,
            "k": kind[:64],
            "c": content[:4000],
            "ref": (source_ref or "chat")[:200],
        },
    )


@app.post("/api/chat")
async def chat(req: ChatRequest, request: Request):
    ident = _identity(request)
    if not _rate_ok(ident.email):
        async def limited():
            yield error_event(
                "RATE_LIMITED",
                f"more than {_RATE_LIMIT} turns in "
                f"{int(_RATE_WINDOW_S / 60)} minutes — pause a moment and "
                "try again",
            ).to_sse()
            yield done_event(session_id=req.session_id).to_sse()

        return StreamingResponse(
            limited(),
            media_type="text/event-stream",
            status_code=429,
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    # `X-Forwarded-Access-Token` is the OBO token the platform injected
    # for this request. The agent endpoint validates it independently.
    obo_token = (
        request.headers.get("X-Forwarded-Access-Token")
        or request.headers.get("x-forwarded-access-token")
        or ""
    )
    if not obo_token:
        # We refuse to chat without a verified OBO token. This is the
        # core of the "no on-behalf-of-user impersonation" guarantee.
        async def err():
            yield error_event(
                "UNAUTHORIZED",
                "no on-behalf-of-user token; " "the request is not authenticated",
            ).to_sse()
            yield done_event(session_id=req.session_id).to_sse()

        return StreamingResponse(
            err(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Learn-with-you: the frontend passes the user's edit of a prior
    # AI draft in `custom_inputs.edited_output`; capture it as a
    # preference (never as a self-granted skill).
    edited = (req.custom_inputs or {}).get("edited_output")
    if edited:
        from agent.learn_with_you import capture_correction

        try:
            capture_correction(
                lambda o, k, c, ref: _store_preference(o, k, c, ref),
                ident.email,
                str((req.custom_inputs or {}).get("original_output") or ""),
                str(edited),
                ref=f"session:{req.session_id}",
            )
        except Exception:  # noqa: BLE001
            log.exception("capture_correction failed")

    # In dev mode, run the chat through the in-process VM so the
    # full harness (executor + constitution + tools + audit log)
    # is exercised without a real Databricks workspace.  In
    # production we still proxy to the upstream agent endpoint.
    if is_dev():
        return await _chat_via_vm(req, ident)

    messages = list(req.messages or [])
    if req.text and (not messages or messages[-1].get("role") != "user"):
        messages.append({"role": "user", "content": req.text})

    async def stream() -> AsyncIterator[bytes]:
        upstream_url = _agent_url(
            "/serving-endpoints/" + AGENT_ENDPOINT + "/invocations"
        )
        body = {
            "input": messages,
            "custom_inputs": {
                "user_email": ident.email,
                "user_id": ident.user_id,
                "session_id": req.session_id,
                "model": req.model,
                "show_reasoning": req.show_reasoning,
                "plan_mode": bool(req.plan_mode),
            },
        }
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(connect=10.0, read=300.0)
            ) as client:
                async with client.stream(
                    "POST",
                    upstream_url,
                    headers={
                        "Authorization": f"Bearer {obo_token}",
                        "Content-Type": "application/json",
                    },
                    json=body,
                ) as r:
                    if r.status_code >= 400:
                        yield error_event(
                            "UPSTREAM",
                            f"agent endpoint returned {r.status_code}",
                        ).to_sse()
                        return
                    async for raw_line in r.aiter_lines():
                        if not raw_line or not raw_line.startswith("data: "):
                            continue
                        chunk = raw_line[6:].strip()
                        if chunk == "[DONE]":
                            yield done_event(session_id=req.session_id).to_sse()
                            return
                        # The agent endpoint may emit raw text or JSON
                        # events. Normalize through alex.events.parse.
                        from alex.events import parse as _parse

                        ev = _parse("data: " + chunk)
                        if ev is None:
                            # Plain text from the upstream; wrap as a
                            # text event so the client never sees raw
                            # deltas.
                            yield text_event(chunk).to_sse()
                        else:
                            yield ev.to_sse()
                    yield done_event(session_id=req.session_id).to_sse()
        except Exception as e:
            log.exception("chat stream failed")
            yield error_event("UPSTREAM", str(e)).to_sse()
            yield done_event(session_id=req.session_id).to_sse()

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Dev VM cache — reuse one VM per (session_id, user) so multi-turn chats
# keep the executor, history, ACL, and AgentDef instead of rebuilding
# them on every request.  TTL + LRU bounded; evicted VMs are close()d.
# ---------------------------------------------------------------------------

_VM_CACHE_LOCK = threading.Lock()
#: Serializes turn-index assignment + persistence (see _chat_via_vm).
_TURN_PERSIST_LOCK = threading.Lock()
_VM_CACHE: OrderedDict[tuple[str, str, bool], tuple[object, float]] = OrderedDict()
_VM_TTL_S = float(os.environ.get("ALEX_VM_TTL_S", "600"))
_VM_CACHE_MAX = int(os.environ.get("ALEX_VM_CACHE_MAX", "32"))


def _vm_cache_get(
    session_id: str, user_email: str, instance_scope: str,
    *, plan_mode: bool = False,
):
    """Return the cached VM for (session_id, user), creating one if
    needed.  Expired and over-capacity entries are evicted and closed.

    ``plan_mode`` is part of the cache key: a cached build-mode VM
    serving a plan-mode request would expose every mutating tool the
    mode exists to withhold."""
    from agent.vm import make_vm

    key = (session_id, user_email, plan_mode)
    now = time.monotonic()
    evicted: list[object] = []
    with _VM_CACHE_LOCK:
        # Evict expired entries.
        for k in [k for k, (_, ts) in _VM_CACHE.items() if now - ts > _VM_TTL_S]:
            evicted.append(_VM_CACHE.pop(k)[0])
        entry = _VM_CACHE.get(key)
        if entry is not None:
            _VM_CACHE.move_to_end(key)
            vm = entry[0]
            _VM_CACHE[key] = (vm, now)
        else:
            vm = make_vm(
                session_id=session_id,
                user_email=user_email,
                instance_scope=instance_scope,
                plan_mode=plan_mode,
            )
            _VM_CACHE[key] = (vm, now)
            while len(_VM_CACHE) > _VM_CACHE_MAX:
                evicted.append(_VM_CACHE.popitem(last=False)[1][0])
    for old in evicted:
        try:
            old.close()
        except Exception:  # noqa: BLE001
            log.exception("error closing evicted VM")
    return vm


def _vm_cache_clear() -> None:
    """Close and drop every cached VM (tests + dev reload)."""
    with _VM_CACHE_LOCK:
        vms = [vm for vm, _ in _VM_CACHE.values()]
        _VM_CACHE.clear()
    for vm in vms:
        try:
            vm.close()
        except Exception:  # noqa: BLE001
            log.exception("error closing cached VM")


def _chat_refused(reason: str) -> StreamingResponse:
    """A short SSE stream carrying one refusal. Chat routes stream by
    contract, so a refusal has to arrive the same way a turn does."""
    async def _gen() -> AsyncIterator[bytes]:
        yield error_event("forbidden", reason).to_sse()
        yield done_event(session_id=None).to_sse()

    return StreamingResponse(_gen(), media_type="text/event-stream")


async def _chat_via_vm(req: ChatRequest, ident) -> StreamingResponse:
    """Dev-only chat path: drive the in-process :class:`agent.vm.VM`.

    The VM owns the executor, the browser OS, the constitution
    interceptor, the audit log, and the SSE event pump.  This
    function is the canonical end-to-end harness path in dev.
    """
    import asyncio

    # A session id is not a capability: it travels in URLs, logs and
    # group shares. If the caller names a session that exists but
    # belongs to someone else, refuse — otherwise this route both
    # chats under the victim's conversation id and (dev) persists
    # turns into their history.
    if req.session_id:
        # One session store, one place that answers ownership. The route
        # used to query the Delta table directly, which in production
        # was a DIFFERENT store than the one turn persistence writes —
        # the same split-brain the dev unification removed, reborn
        # between backends.
        try:
            from agent import session_manager as _sm

            existing = _sm.get_session(None, req.session_id)
        except Exception:  # noqa: BLE001
            existing = None
        if existing is not None and existing.owner_email != ident.email:
            return _chat_refused("session belongs to another user")

    # Ensure the dev warehouse is seeded before the VM reads the ACL
    # (the VM filters its tool exposure by effective_employee_acl).
    _sql("SELECT 1 FROM media.control.effective_employee_acl LIMIT 1", {})

    def _events() -> asyncio.Queue[object]:
        return asyncio.Queue()

    queue: asyncio.Queue = _events()
    loop = asyncio.get_running_loop()

    def _put(ev) -> None:
        # The event pump runs on a worker thread (the executor's
        # streaming helper); marshal back to the event loop.
        loop.call_soon_threadsafe(queue.put_nowait, ev)

    vm = _vm_cache_get(
        req.session_id or f"chat-{ident.user_id}",
        ident.email,
        ident.user_id,
        plan_mode=bool(req.plan_mode),
    )
    unsubscribe = vm.subscribe_sse(_put)

    async def _drive() -> None:
        try:
            text = req.text or ""
            if not text and req.messages:
                # Use the last user message if no fresh text was sent.
                for m in reversed(req.messages):
                    if m.get("role") == "user":
                        text = m.get("content", "")
                        break
            show_reasoning = bool(
                req.show_reasoning
                or (req.custom_inputs or {}).get("show_reasoning")
            )
            if text:
                last, _ = await asyncio.to_thread(
                    lambda: vm.run_turn(text=text, show_reasoning=show_reasoning)
                )
                # Persist the turn server-side so the SDK / a fresh client
                # can resume the conversation (dev + prod). Owner-scoped:
                # an un-owned sid must never gain turns from this caller.
                try:
                    from agent import session_manager as _sm

                    sid = req.session_id or f"chat-{ident.user_id}"
                    # Turn indices are caller-assigned, and the append is
                    # insert-or-drop: two chats finishing together both
                    # read the same base and one pair was silently lost.
                    # Serialize the read-modify-write and derive the base
                    # from the max existing index (gaps stay gaps).
                    with _TURN_PERSIST_LOCK:
                        existing = _sm.get_turns(None, sid, owner_email=ident.email)
                        base = (
                            max(int(t.get("turn_no") or 0) for t in existing) + 1
                            if existing else 0
                        )
                        _sm.append_turn(None, sid, base, "user", text)
                        _sm.append_turn(None, sid, base + 1, "assistant", last or "")
                except Exception:  # noqa: BLE001
                    log.debug("turn persistence skipped", exc_info=True)
            else:
                queue.put_nowait(error_event("EMPTY", "no user text in request"))
        except Exception as exc:  # noqa: BLE001
            queue.put_nowait(error_event("VM", str(exc)))
        finally:
            queue.put_nowait(_DoneSentinel())
            unsubscribe()

    # Hold a reference: asyncio only weakly references tasks, and an
    # unreferenced driver task can be garbage-collected mid-turn — the
    # documented create_task footgun, presenting as a chat that stops
    # for no reason.
    drive_task = asyncio.create_task(_drive())
    _BACKGROUND_TASKS.add(drive_task)
    drive_task.add_done_callback(_BACKGROUND_TASKS.discard)

    async def _gen() -> AsyncIterator[bytes]:
        while True:
            ev = await queue.get()
            if isinstance(ev, _DoneSentinel):
                yield done_event(session_id=req.session_id).to_sse()
                return
            try:
                yield ev.to_sse()
            except Exception:  # noqa: BLE001
                # The agent didn't return a chat reply; surface a final
                # done event so the client can clean up.
                yield done_event(session_id=req.session_id).to_sse()
                return

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class _DoneSentinel:
    """Sentinel inserted at the end of the VM event stream."""


#: Strong references to in-flight driver tasks (see chat): asyncio's
#: task registry is weak, and a collected task is a silently dead chat.
_BACKGROUND_TASKS: set = set()


def _agent_url(path: str) -> str:
    """Build the agent endpoint URL. In production, the app's serving
    endpoint resource is resolved by the platform. In dev, we use
    a localhost URL the test harness provides."""
    base = os.environ.get("ALEX_AGENT_BASE_URL", "")
    if base:
        return base.rstrip("/") + path
    # Production: the Databricks App platform injects the host.
    host = os.environ.get("DATABRICKS_HOST", "https://dummy.cloud.databricks.com")
    return host.rstrip("/") + path


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


# The session routes go through agent.session_manager — the ONE session
# store abstraction — not through _sql. The routes used to INSERT into
# the Delta table directly while the manager's production path lives on
# Lakebase Postgres: two stores for one concept, in production only,
# which is precisely the split-brain the dev unification removed. Every
# read and write below hits the same backend turn persistence uses.


@app.get("/api/sessions")
async def list_sessions(request: Request):
    ident = _identity(request)
    from agent import session_manager

    sessions = await run_in_threadpool(
        session_manager.list_sessions, None, ident.email
    )
    return {
        "sessions": [
            {
                "session_id": s.session_id,
                "owner_email": s.owner_email,
                "kind": s.kind,
                "title": s.title,
                "parent_id": s.parent_id,
                "model": s.model,
                "created_at": s.created_at,
                "last_active_at": s.last_active_at,
            }
            for s in sessions
        ]
    }


@app.post("/api/sessions")
async def new_session(request: Request):
    ident = _identity(request)
    body = await request.json()
    from agent import session_manager

    sid = await run_in_threadpool(
        session_manager.open_session,
        None,
        ident.email,
        (body.get("title") or "New chat")[:200],
    )
    return {"session_id": sid}


@app.get("/api/sessions/{session_id}/messages")
async def session_messages(session_id: str, request: Request, limit: int = 200):
    """The persisted turns of a chat session, so the SDK or a fresh
    client (not just the browser's localStorage) can resume it.

    Owner-scoped: a session id is not a capability, and ids travel in
    URLs, logs and group shares — the read is filtered to the caller's
    own sessions before any turn is returned."""
    ident = _identity(request)
    from agent import session_manager

    turns = session_manager.get_turns(
        None, session_id, limit=limit, owner_email=ident.email
    )
    return {"session_id": session_id, "messages": turns}


@app.patch("/api/sessions/{session_id}")
async def rename_session(session_id: str, request: Request):
    """Rename a chat session, owner-scoped inside the manager."""
    ident = _identity(request)
    body = await request.json()
    title = (body.get("title") or "").strip()[:200]
    if not title:
        return JSONResponse({"error": "title required"}, status_code=400)
    from agent import session_manager

    await run_in_threadpool(
        session_manager.rename_session, None, session_id, ident.email, title
    )
    return {"ok": True, "session_id": session_id, "title": title}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str, request: Request):
    """Delete a chat session and its persisted turns.

    Ownership, the session row and the turns are handled in ONE place
    (session_manager.delete_session): turns are only removed on a true
    ownership match, and a guessed id never reaches anyone's history.
    The response does not reveal whether the session exists for
    someone else."""
    ident = _identity(request)
    from agent import session_manager

    deleted = await run_in_threadpool(
        session_manager.delete_session, None, session_id, ident.email
    )
    return {"ok": True, "deleted": bool(deleted), "session_id": session_id}


# ---------------------------------------------------------------------------
# Inbox, tasks, briefs — these were the "phantom" endpoints the UI was
# calling without a backend. We implement them as cheap reads from the
# already-materialized `inbox_view` and `task_board_view` tables, plus
# the user's morning brief.
# ---------------------------------------------------------------------------


@app.get("/api/inbox")
async def inbox(request: Request, limit: int = 50):
    ident = _identity(request)
    rows = _sql_as_user(
        request,
        """
        SELECT email_id, priority, subject, from_email, received_at, summary
          FROM media.alex.inbox_view
         WHERE owner_email = :u
         ORDER BY priority ASC NULLS LAST, received_at DESC
         LIMIT :lim
    """,
        {"u": ident.email, "lim": max(1, min(limit, 200))},
    )
    return {"inbox": rows}


@app.get("/api/email/{email_id}")
async def email_detail(email_id: str, request: Request):
    """Full email (subject, from, body, AI summary) for the AI-email tab."""
    ident = _identity(request)
    rows = _sql_as_user(
        request,
        """
        SELECT email_id, priority, subject, from_email, received_at,
               body, summary, thread_id
          FROM media.alex.inbox
         WHERE owner_email = :u AND email_id = :id
         LIMIT 1
    """,
        {"u": ident.email, "id": email_id},
    )
    if not rows:
        # Fall back to the view (no body) so the tab still works.
        rows = _sql_as_user(
            request,
            """
            SELECT email_id, priority, subject, from_email, received_at, summary
              FROM media.alex.inbox_view
             WHERE owner_email = :u AND email_id = :id
             LIMIT 1
        """,
            {"u": ident.email, "id": email_id},
        )
        if not rows:
            return JSONResponse({"error": "not_found"}, status_code=404)
    return {"email": rows[0]}


class EmailDraftRequest(BaseModel):
    email_id: str
    instruction: str = ""
    tone: str = "professional"


@app.post("/api/email/draft")
async def email_draft(req: EmailDraftRequest, request: Request):
    """AI-assisted reply draft for an email. Runs the dev LLM directly
    (a fast, single-shot call — no tools) so the email tab feels instant."""
    ident = _identity(request)
    rows = _sql_as_user(
        request,
        """
        SELECT subject, from_email, body, summary
          FROM media.alex.inbox
         WHERE owner_email = :u AND email_id = :id
         LIMIT 1
    """,
        {"u": ident.email, "id": req.email_id},
    )
    if not rows:
        return JSONResponse({"error": "not_found"}, status_code=404)
    e = rows[0]
    body = (e.get("body") or e.get("summary") or "")[:6000]
    sys_prompt = (
        "You are an executive assistant drafting a concise, "
        f"{req.tone} email reply on behalf of the user. Return only the "
        "reply body — no subject line, no preamble. Keep it under 150 words."
    )
    user_prompt = (
        f"Original email from {e.get('from_email', '')}:\n"
        f"Subject: {e.get('subject', '')}\n\n{body}\n\n"
        f"Draft a reply. {('Extra instruction: ' + req.instruction) if req.instruction else ''}"
    )
    try:
        from alex.llm import get_llm

        resp = get_llm().invoke(
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
            max_tokens=600,
        )
        draft = (getattr(resp, "content", "") or "").strip()
    except Exception as exc:  # noqa: BLE001
        log.warning("email draft failed: %s", exc)
        return JSONResponse({"error": "draft_failed"}, status_code=502)
    return {"draft": draft, "email_id": req.email_id}


@app.get("/api/tasks")
async def tasks(request: Request, status: str | None = None):
    ident = _identity(request)
    if status:
        rows = _sql_as_user(
            request,
            """
            SELECT task_id, title, priority, due_at
              FROM media.alex.task_board_view
             WHERE owner_email = :u AND status = :s
             ORDER BY priority ASC, due_at ASC NULLS LAST
             LIMIT 200
        """,
            {"u": ident.email, "s": status},
        )
    else:
        rows = _sql_as_user(
            request,
            """
            SELECT task_id, title, status, priority, due_at
              FROM media.alex.task_board_view
             WHERE owner_email = :u
             ORDER BY priority ASC, due_at ASC NULLS LAST
             LIMIT 200
        """,
            {"u": ident.email},
        )
    return {"tasks": rows}


@app.get("/api/briefs")
async def briefs(request: Request, session: str = ""):
    """The 'Prepared overnight' strip. We return today's brief for
    the user, plus any unconsumed standing-order briefs."""
    ident = _identity(request)
    today = _sql_as_user(
        request,
        """
        SELECT user_email, brief_date, brief, standing_orders,
               priority_emails, tasks_due
          FROM media.alex.morning_briefs
         WHERE user_email = :u AND brief_date = current_date()
         LIMIT 1
    """,
        {"u": ident.email},
    )
    sob = _sql_as_user(
        request,
        """
        SELECT brief_id, order_id, compiled_context, suggested_actions
          FROM media.control.standing_order_briefs
         WHERE user_email = :u AND consumed = false
         ORDER BY brief_date
         LIMIT 5
    """,
        {"u": ident.email},
    )
    out = {"brief": today[0] if today else None, "standing_orders": sob}
    return out


@app.post("/api/briefs/consume")
async def consume_brief(request: Request):
    """Mark a standing-order brief consumed so the 'Prepared overnight'
    strip stops surfacing it. Without this the strip showed the same
    briefs forever — the lifecycle had no terminal 'consume' step."""
    ident = _identity(request)
    body = await request.json()
    brief_id = str(body.get("brief_id") or "").strip()
    if not brief_id:
        return JSONResponse({"error": "brief_id required"}, status_code=400)
    # RLS + the user_email predicate ensure a user can only consume their
    # own brief. consumed is bound as a param (the dev warehouse's UPDATE
    # grammar only accepts parameterized SET values).
    _sql_as_user(
        request,
        """
        UPDATE media.control.standing_order_briefs
           SET consumed = :c
         WHERE brief_id = :id AND user_email = :u
        """,
        {"c": True, "id": brief_id, "u": ident.email},
    )
    return {"ok": True, "consumed": brief_id}


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


WORKSPACE_ROOT = os.environ.get("ALEX_WORKSPACE_ROOT", "/Volumes/media/alex/workspaces")


def _safe_path_join(user_email: str, rel: str) -> str:
    """Resolve a user-relative path against the user's workspace root
    and verify the result does not escape. Uses realpath."""
    if not rel or "\x00" in rel:
        raise ValueError("invalid path")
    # Strict allowlist for the email -> directory mapping: lowercase
    # alphanumerics plus @ . _ - ; everything else becomes '_' and
    # leading dots are stripped (no hidden/relative dirs).
    safe_user = re.sub(r"[^a-z0-9@._-]", "_", user_email.lower()).lstrip(".")
    if not safe_user:
        raise ValueError("invalid user")
    user_root = os.path.realpath(os.path.join(WORKSPACE_ROOT, safe_user))
    candidate = os.path.realpath(os.path.join(user_root, rel))
    if not (candidate == user_root or candidate.startswith(user_root + os.sep)):
        raise ValueError("path escapes the workspace")
    return candidate


@app.post("/api/files/upload")
async def upload(request: Request, file: UploadFile):
    ident = _identity(request)
    rel = (file.filename or "").replace("\\", "/").lstrip("/")
    if not rel:
        return JSONResponse({"error": "filename required"}, status_code=400)
    try:
        dest = _safe_path_join(ident.email, rel)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    max_bytes = int(
        os.environ.get("ALEX_UPLOAD_MAX_BYTES", str(25 * 1024 * 1024))
    )
    # Stream to disk in chunks with the cap enforced as the bytes
    # arrive: reading the whole body first meant a single large
    # multipart upload was fully resident in memory (or OOM-killed the
    # replica) before the size check ever ran.
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    written = 0
    tmp = dest + ".part"
    try:
        with open(tmp, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    f.close()
                    os.remove(tmp)
                    return JSONResponse(
                        {"error": "file too large"}, status_code=413
                    )
                f.write(chunk)
        os.replace(tmp, dest)
    except Exception:  # noqa: BLE001
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    return {"path": rel, "bytes": written}


@app.get("/api/files/download")
async def download(request: Request, path: str):
    ident = _identity(request)
    if not path.startswith("outputs/"):
        return JSONResponse(
            {"error": "only outputs/ artifacts are downloadable"}, status_code=400
        )
    try:
        full = _safe_path_join(ident.email, path)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    if not os.path.isfile(full):
        return JSONResponse({"error": "not found"}, status_code=404)
    with open(full, "rb") as f:
        data = f.read()
    name = os.path.basename(full)
    # RFC 6266/5987: ASCII fallback (quotes and control chars stripped)
    # plus a UTF-8 percent-encoded filename*.
    ascii_name = re.sub(r'[^\x20-\x7e]', "_", name).replace('"', "_").replace("\\", "_")
    utf8_name = urllib.parse.quote(name, safe="")
    return Response(
        data,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": (
                f'attachment; filename="{ascii_name}"; '
                f"filename*=UTF-8''{utf8_name}"
            )
        },
    )


@app.get("/api/files/list")
async def list_files(request: Request):
    ident = _identity(request)
    out: dict[str, list[str]] = {}
    for folder in ("uploads", "work", "outputs"):
        try:
            root = _safe_path_join(ident.email, folder)
            if os.path.isdir(root):
                out[folder] = sorted(os.listdir(root))
            else:
                out[folder] = []
        except ValueError:
            out[folder] = []
    return out


# ---------------------------------------------------------------------------
# Transparency — what the user's AI knows, may do, and has done.
# Every row is filtered by owner_email in the query.
# ---------------------------------------------------------------------------


@app.get("/api/transparency/memory")
async def t_memory(request: Request):
    ident = _identity(request)
    rows = _sql_as_user(
        request,
        """
        SELECT id, kind, content, salience, source_ref, created_at
          FROM media.control.vector_memory_mirror
         WHERE owner_email = :u AND superseded_by IS NULL
         ORDER BY salience DESC LIMIT 200
    """,
        {"u": ident.email},
    )
    return {"memories": rows}


@app.get("/api/transparency/graph")
async def t_graph(request: Request):
    ident = _identity(request)
    rows = _sql_as_user(
        request,
        """
        SELECT edge_id, src, rel, dst, evidence, confidence
          FROM media.control.kg_mirror
         WHERE owner_email = :u
         ORDER BY confidence DESC LIMIT 500
    """,
        {"u": ident.email},
    )
    return {"edges": rows}


@app.post("/api/transparency/forget")
async def t_forget(request: Request):
    ident = _identity(request)
    body = await request.json()
    item_id = (body.get("item_id") or "")[:128]
    reason = (body.get("reason") or "")[:300]
    if not item_id:
        return JSONResponse({"error": "item_id required"}, status_code=400)
    # INSERT only — the user is the requester; the jobs SP flips
    # `applied = true` after honoring the request.
    _sql_as_user(
        request,
        """
        INSERT INTO media.control.forget_requests
          (request_id, owner_email, item_id, reason, ts, applied)
        VALUES (:rid, :u, :i, :r, current_timestamp(), false)
    """,
        {"rid": str(uuid.uuid4()), "u": ident.email, "i": item_id, "r": reason},
    )
    return {"status": "queued"}


# ---------------------------------------------------------------------------
# SDK — programmatic memory/graph/workflow surface (sdk/alex_sdk).
# Same owner-filtered tables as the transparency panel; the OBO
# identity is always the owner, so an SDK caller can only touch
# their own rows.
# ---------------------------------------------------------------------------


@app.post("/api/sdk/remember")
async def sdk_remember(request: Request):
    ident = _identity(request)
    body = await request.json()
    content = (body.get("content") or "").strip()[:4000]
    if not content:
        return JSONResponse({"error": "content required"}, status_code=400)
    kind = (body.get("kind") or "fact")[:64]
    salience = min(max(float(body.get("salience") or 0.5), 0.0), 1.0)
    _sql_as_user(
        request,
        """
        INSERT INTO media.control.vector_memory_mirror
          (id, owner_email, kind, content, salience, source_ref, created_at, superseded_by)
        VALUES (:id, :u, :k, :c, :s, :ref, current_timestamp(), NULL)
    """,
        {
            "id": str(uuid.uuid4()),
            "u": ident.email,
            "k": kind,
            "c": content,
            "s": salience,
            "ref": (body.get("source_ref") or "sdk")[:200],
        },
    )
    return {"status": "stored"}


@app.post("/api/sdk/recall")
async def sdk_recall(request: Request):
    ident = _identity(request)
    body = await request.json()
    query = (body.get("query") or "").strip().lower()
    k = min(int(body.get("k") or 10), 50)
    rows = _sql_as_user(
        request,
        """
        SELECT id, kind, content, salience, source_ref, created_at
          FROM media.control.vector_memory_mirror
         WHERE owner_email = :u AND superseded_by IS NULL
         ORDER BY salience DESC LIMIT 200
    """,
        {"u": ident.email},
    )
    # Lightweight lexical ranking over the owner's beads; semantic
    # ranking happens in the Lakebase-backed recall the agent uses.
    if query:
        terms = [t for t in query.split() if t]
        rows = sorted(
            rows,
            key=lambda r: sum(t in str(r.get("content", "")).lower() for t in terms),
            reverse=True,
        )
    return {"memories": rows[:k]}


@app.post("/api/sdk/add_edge")
async def sdk_add_edge(request: Request):
    ident = _identity(request)
    body = await request.json()
    src = (body.get("src") or "").strip()[:200]
    rel = (body.get("rel") or "").strip()[:100]
    dst = (body.get("dst") or "").strip()[:200]
    if not (src and rel and dst):
        return JSONResponse({"error": "src, rel, dst required"}, status_code=400)
    _sql_as_user(
        request,
        """
        INSERT INTO media.control.kg_mirror
          (edge_id, owner_email, src, rel, dst, evidence, confidence)
        VALUES (:eid, :u, :s, :r, :d, :e, :c)
    """,
        {
            "eid": str(uuid.uuid4()),
            "u": ident.email,
            "s": src,
            "r": rel,
            "d": dst,
            "e": (body.get("evidence") or "sdk")[:300],
            "c": min(max(float(body.get("confidence") or 0.7), 0.0), 1.0),
        },
    )
    return {"status": "stored"}


@app.post("/api/sdk/graph_query")
async def sdk_graph_query(request: Request):
    ident = _identity(request)
    body = await request.json()
    entity = (body.get("entity") or "").strip()
    rows = _sql_as_user(
        request,
        """
        SELECT edge_id, src, rel, dst, evidence, confidence
          FROM media.control.kg_mirror
         WHERE owner_email = :u
         ORDER BY confidence DESC LIMIT 500
    """,
        {"u": ident.email},
    )
    if entity:
        low = entity.lower()
        rows = [
            r
            for r in rows
            if low in str(r.get("src", "")).lower()
            or low in str(r.get("dst", "")).lower()
        ]
    return {"edges": rows[:100]}


@app.get("/api/sdk/list_workflows")
async def sdk_list_workflows(request: Request):
    _identity(request)
    rows = _sql(
        """
        SELECT workflow_id, name, description, capability, active
          FROM media.control.workflow_catalog
         WHERE active = true
    """,
        {},
    )
    return {"workflows": rows}


@app.get("/api/transparency/capabilities")
async def t_capabilities(request: Request):
    ident = _identity(request)
    effective = _sql_as_user(
        request,
        """
        SELECT capability, expires_at FROM media.control.effective_employee_acl
         WHERE user_email = :u
    """,
        {"u": ident.email},
    )
    delegations = _sql_as_user(
        request,
        """
        SELECT capability, supervisor_email, justification, expires_at
          FROM media.control.delegations
         WHERE report_email = :u AND revoked = false
    """,
        {"u": ident.email},
    )
    return {"effective": effective, "delegations": delegations}


@app.get("/api/transparency/actions")
async def t_actions(request: Request):
    ident = _identity(request)
    rows = _sql_as_user(
        request,
        """
        SELECT ts, action_type, target_id, detail, outcome, tool_name
          FROM media.alex.agent_actions
         WHERE actor = :u
         ORDER BY ts DESC LIMIT 500
    """,
        {"u": ident.email},
    )
    return {"actions": rows}


@app.get("/api/transparency/library")
async def t_library(request: Request):
    ident = _identity(request)
    # Two reads joined in Python, not SQL JOIN: the dev warehouse
    # "answers" a JOIN with rows of Nones — silent wrongness — so a
    # joined query here showed an empty library in dev however many
    # files were pinned. Python joining is identical on both backends.
    pinned = _sql_as_user(
        request,
        """
        SELECT file_id, pinned_summary, added_at
          FROM media.control.instance_library
         WHERE owner_email = :u
        """,
        {"u": ident.email},
    )
    files = _sql(
        "SELECT file_id, path, source_channel, approved_at, revoked "
        "FROM media.control.approved_files",
        {},
    )
    by_id = {
        str(f.get("file_id")): f
        for f in files
        if str(f.get("revoked")).lower() not in ("true", "1")
    }
    rows = [
        {
            "file_id": p.get("file_id"),
            "pinned_summary": p.get("pinned_summary"),
            "added_at": p.get("added_at"),
            "path": by_id[str(p.get("file_id"))].get("path"),
            "source_channel": by_id[str(p.get("file_id"))].get("source_channel"),
            "approved_at": by_id[str(p.get("file_id"))].get("approved_at"),
        }
        for p in pinned
        if str(p.get("file_id")) in by_id
    ]
    rows.sort(key=lambda r: str(r.get("added_at") or ""), reverse=True)
    return {"library": rows[:100]}


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------


def _principal_for(request: Request):
    """Resolve the acting principal for authorization decisions."""
    from agent import access

    ident = _identity(request)
    return access.load_principal(
        lambda stmt, params=None: _sql(stmt, params or {}), ident.email
    )


@app.get("/api/security/me")
async def security_me(request: Request):
    """The caller's own resolved role, clearance and scopes.

    Everyone may see their *own* authorization context — knowing what you
    are allowed to reach is not itself sensitive, and hiding it only makes
    denials confusing.
    """
    from agent import access

    principal = _principal_for(request)
    return {
        **principal.to_dict(),
        "visible_scopes": access.visible_scopes(principal),
        "classifications": list(access.CLASSIFICATIONS),
    }


@app.get("/api/security/access-log")
async def security_access_log(request: Request, limit: int = 100):
    """The knowledge access ledger — administrators only.

    This is the oversight surface that replaces reading colleagues'
    content: an admin can see *that* a document was read and by whom,
    without the role granting them the document itself.
    """
    from agent import access

    principal = _principal_for(request)
    if not principal.is_admin:
        return JSONResponse(
            {
                "error": "administrator role required",
                "your_role": principal.role,
            },
            status_code=403,
        )
    rows = access.read_ledger(limit=max(1, min(int(limit), 1000)))
    return {
        "entries": rows,
        "count": len(rows),
        "denied": sum(1 for r in rows if not r.get("allowed")),
    }


def _board_sql(stmt: str, params: dict | None = None):
    return _sql(stmt, params or {})


# ---------------------------------------------------------------------------
# Oversight — the administrator's view of whether the agents are behaving.
#
# Deliberately separate from the audit log. The audit log answers "what
# happened"; this answers "did anything go wrong", which is the question
# leadership actually has and the one nobody has time to derive from a
# raw event stream.
# ---------------------------------------------------------------------------

_ADMIN_ONLY = {"error": "administrator role required"}


@app.post("/api/dev/role")
async def dev_grant_role(request: Request):
    """Dev helper: set a role so oversight can be exercised locally.

    404s outside dev, exactly like ``/api/dev/token``. Roles in
    production are provisioned through the control plane by a governor,
    never by an API the app itself exposes.
    """
    if not is_dev():
        return JSONResponse({"error": "dev-only endpoint"}, status_code=404)
    from agent import access, preferences  # noqa: F401

    body = await request.json()
    email = str((body or {}).get("email") or "").strip().lower()
    role = str((body or {}).get("role") or "member").strip().lower()
    if not email or role not in access.ROLES:
        return JSONResponse(
            {"error": f"email and a role in {list(access.ROLES)} are required"},
            status_code=400,
        )
    _sql(
        "INSERT INTO media.control.user_roles "
        "(user_email, role, team, granted_scope, clearance) "
        "VALUES (:u, :r, :t, :g, :c)",
        {"u": email, "r": role, "t": None, "g": None, "c": None},
    )
    access.invalidate(email)
    return {"email": email, "role": role}


def _require_admin(request: Request):
    principal = _principal_for(request)
    return principal if principal.is_admin else None


@app.get("/api/oversight/summary")
async def oversight_summary(request: Request, limit: int = 500):
    """Counts an executive can read at a glance."""
    from agent import supervisor

    principal = _require_admin(request)
    if principal is None:
        return JSONResponse(
            {**_ADMIN_ONLY, "your_role": _principal_for(request).role},
            status_code=403,
        )
    rows = supervisor.recent(_board_sql, limit=limit)
    return {
        "summary": supervisor.summarise(rows),
        "severities": list(supervisor.SEVERITIES),
    }


@app.get("/api/oversight/findings")
async def oversight_findings(request: Request, severity: str = "", limit: int = 100):
    """The supervision ledger — every finding, newest first."""
    from agent import supervisor

    if _require_admin(request) is None:
        return JSONResponse(_ADMIN_ONLY, status_code=403)
    rows = supervisor.recent(
        _board_sql, limit=max(1, min(int(limit), 500)), severity=severity
    )
    return {"findings": rows, "count": len(rows)}


@app.get("/api/oversight/violations")
async def oversight_violations(
    request: Request, disposition: str = "", limit: int = 100
):
    """The constitutional ledger — what the law stopped, and what it only
    noted. Distinct from `/findings`: supervision reviews a run after the
    fact, this is the record of the interceptor refusing an action while
    it was being attempted."""
    from agent import constitution

    if _require_admin(request) is None:
        return JSONResponse(_ADMIN_ONLY, status_code=403)
    rows = constitution.recent_violations(
        _board_sql, limit=max(1, min(int(limit), 500)), disposition=disposition
    )
    return {
        "violations": rows,
        "count": len(rows),
        "summary": constitution.summarise_violations(rows),
    }


@app.get("/api/oversight/runs")
async def oversight_runs(request: Request):
    """Every agent run currently in flight, across all users.

    A halted or long-running agent should be visible to an administrator
    while it is happening, not only in the record afterwards.
    """
    from agent import standing_orders as so
    from agent import taskboard

    if _require_admin(request) is None:
        return JSONResponse(_ADMIN_ONLY, status_code=403)

    # Live comes from this process, because that is what "in flight"
    # means — a run with a thread behind it. History comes from the
    # durable record, because the in-process registry is erased by the
    # very restarts whose consequences an administrator most needs to
    # see: the runs that restart interrupted.
    live = [
        r.to_dict()
        for r in [*taskboard._RUNS.values(), *so._RUNS.values()]
        if r.status == "running"
    ]
    recent = taskboard.recent_runs(_board_sql, limit=40)
    interrupted = [r for r in recent if r.get("status") == "interrupted"]
    return {
        "live": live,
        "recent": recent,
        "live_count": len(live),
        "interrupted": interrupted,
        "interrupted_count": len(interrupted),
    }


@app.get("/api/oversight/limits")
async def oversight_limits(request: Request):
    """The execution ceilings in force, by autonomy level.

    Published so the bounds are inspectable rather than folklore.
    """
    from agent.execution_governor import CAPABILITY_POWER, Budget

    if _require_admin(request) is None:
        return JSONResponse(_ADMIN_ONLY, status_code=403)
    return {
        "budgets": {
            level: Budget.for_autonomy(level).to_dict()
            for level in ("suggest", "co-pilot", "autopilot")
        },
        "capability_power": CAPABILITY_POWER,
        "pathologies": [
            "budget_time", "budget_tools", "budget_turns", "budget_tokens",
            "repetition", "oscillation", "stagnation", "denial_persistence",
            "denial_escalation", "scope_drift", "untrusted_content",
        ],
    }


@app.post("/api/oversight/findings/acknowledge")
async def oversight_acknowledge(request: Request):
    """Mark findings as reviewed. The ledger is append-only, so this
    records who looked rather than removing anything."""
    principal = _require_admin(request)
    if principal is None:
        return JSONResponse(_ADMIN_ONLY, status_code=403)
    body = await request.json()
    run_id = str((body or {}).get("run_id") or "")
    if not run_id:
        return JSONResponse({"error": "run_id is required"}, status_code=400)
    try:
        _sql(
            "UPDATE media.control.supervision_findings "
            "SET acknowledged_by = :who, acknowledged_at = :ts "
            "WHERE run_id = :r",
            {"who": principal.email, "ts": _now_iso(), "r": run_id},
        )
    except Exception:  # noqa: BLE001
        log.debug("acknowledge failed", exc_info=True)
        return JSONResponse({"error": "could not acknowledge"}, status_code=500)
    return {"acknowledged": run_id, "by": principal.email}


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(sep=" ", timespec="seconds")


# ---------------------------------------------------------------------------
# Preferences — the workspace shaped to the person using it.
# ---------------------------------------------------------------------------


@app.get("/api/preferences")
async def preferences_get(request: Request):
    from agent import preferences

    ident = _identity(request)
    return preferences.get(_board_sql, ident.email)


@app.patch("/api/preferences")
async def preferences_set(request: Request):
    """Apply preference changes. Keys pinned by workspace policy are
    reported as refused rather than silently dropped."""
    from agent import preferences

    ident = _identity(request)
    body = await request.json()
    return preferences.set_many(_board_sql, ident.email, body or {})


@app.post("/api/preferences/reset")
async def preferences_reset(request: Request):
    from agent import preferences

    ident = _identity(request)
    preferences.reset(_board_sql, ident.email)
    return preferences.get(_board_sql, ident.email)


# ---------------------------------------------------------------------------
# MCP servers — what this user's VM can reach.
# ---------------------------------------------------------------------------


@app.get("/api/mcp/servers")
async def mcp_servers(request: Request):
    """Per-user MCP server health: reachable, tool counts, quarantine."""
    from agent import mcp_client

    principal = _principal_for(request)
    return {
        "servers": await run_in_threadpool(
            mcp_client.health, _board_sql, principal
        ),
        "breaker": mcp_client.breaker_state(),
    }


@app.get("/api/mcp/tools")
async def mcp_tools(request: Request):
    """The remote tools this user's VM exposes, with their origin."""
    from agent import mcp_client

    principal = _principal_for(request)
    tools = await run_in_threadpool(mcp_client.tools_for, _board_sql, principal)
    return {"tools": [t.to_dict() for t in tools], "count": len(tools)}


@app.post("/api/mcp/servers/{server_id}/approve")
async def mcp_approve(request: Request, server_id: str):
    """Re-approve a server whose tool definitions changed. Admin only —
    this is the human step the rug-pull defence exists to force."""
    from agent import mcp_client

    principal = _principal_for(request)
    if not principal.is_admin:
        return JSONResponse(
            {"error": "administrator role required"}, status_code=403
        )
    ok = mcp_client.approve(_board_sql, server_id)
    mcp_client.reset()
    return {"approved": ok}


# ---------------------------------------------------------------------------
# Standing orders — durable instructions a person can configure, assign,
# and execute against the enterprise system.
# ---------------------------------------------------------------------------

_ORDER_NOT_FOUND = {"error": "no such standing order"}


@app.get("/api/orders")
async def orders_list(request: Request):
    """Orders assigned to the caller, plus the ones they own."""
    from agent import standing_orders as so

    principal = _principal_for(request)
    mine = so.for_user(_board_sql, principal.email, active_only=False)
    return {
        "orders": [
            {
                **o.to_dict(),
                "is_owner": o.owner_email.lower() == principal.email.lower(),
                "runnable": o.trigger == so.TRIGGER_MANUAL and o.active,
                "effective_capabilities": so.effective_capabilities(o, principal),
                "last_run": (
                    so.latest_run_for(o.order_id).to_dict()
                    if so.latest_run_for(o.order_id)
                    else None
                ),
            }
            for o in mine
        ],
        "triggers": list(so.TRIGGERS),
        "my_capabilities": sorted(principal.capabilities),
    }


@app.post("/api/orders")
async def orders_create(request: Request):
    from agent import standing_orders as so

    principal = _principal_for(request)
    body = await request.json()
    try:
        order = so.create(
            _board_sql,
            principal,
            title=body.get("title") or "",
            instruction=body.get("instruction") or "",
            audience=body.get("audience") or [principal.email],
            allowed_capabilities=body.get("allowed_capabilities") or [],
            trigger=body.get("trigger") or so.TRIGGER_MANUAL,
            schedule=body.get("schedule") or "",
            applies_to=body.get("applies_to") or [],
            ambient=bool(body.get("ambient", True)),
        )
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    return order.to_dict()


@app.patch("/api/orders/{order_id}")
async def orders_update(request: Request, order_id: str):
    from agent import standing_orders as so

    principal = _principal_for(request)
    body = await request.json()
    try:
        order = so.update(_board_sql, principal, order_id, **body)
    except PermissionError as exc:
        return JSONResponse({"error": str(exc)}, status_code=403)
    if order is None:
        return JSONResponse(_ORDER_NOT_FOUND, status_code=404)
    return order.to_dict()


@app.post("/api/orders/{order_id}/assign")
async def orders_assign(request: Request, order_id: str):
    """Assign an order to more people (or to ``all``)."""
    from agent import standing_orders as so

    principal = _principal_for(request)
    body = await request.json()
    try:
        order = so.assign(
            _board_sql, principal, order_id, body.get("members") or []
        )
    except PermissionError as exc:
        return JSONResponse({"error": str(exc)}, status_code=403)
    if order is None:
        return JSONResponse(_ORDER_NOT_FOUND, status_code=404)
    return order.to_dict()


@app.post("/api/orders/{order_id}/run")
async def orders_run(request: Request, order_id: str):
    """Execute a standing order now, in its own VM session."""
    from agent import standing_orders as so

    principal = _principal_for(request)
    body = await request.json() if await request.body() else {}
    try:
        run = so.run(
            _board_sql, principal, order_id, note=(body or {}).get("note", "")
        )
    except LookupError:
        return JSONResponse(_ORDER_NOT_FOUND, status_code=404)
    except PermissionError as exc:
        return JSONResponse({"error": str(exc)}, status_code=403)
    return run.to_dict()


@app.get("/api/orders/runs/{run_id}/stream")
async def orders_run_stream(request: Request, run_id: str):
    """Live feed of a standing order executing."""
    from agent import standing_orders as so

    principal = _principal_for(request)
    run = so.get_run(run_id)
    if run is None:
        return JSONResponse({"error": "no such run"}, status_code=404)
    if run.started_by.lower() != principal.email.lower() and not principal.is_admin:
        return JSONResponse({"error": "not your run"}, status_code=403)

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    unsubscribe = run.subscribe(
        lambda event: loop.call_soon_threadsafe(queue.put_nowait, event)
    )

    async def _gen():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                except TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("type") == "run_finished":
                    break
        finally:
            unsubscribe()

    return StreamingResponse(_gen(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# The best-practice library — primary guidance, not retrieval.
# ---------------------------------------------------------------------------


@app.get("/api/practices")
async def practices_list(request: Request, include_retired: bool = False):
    from agent import practices

    principal = _principal_for(request)
    items = practices.visible(
        _board_sql, principal, include_retired=include_retired
    )
    return {
        "practices": [p.to_dict() for p in items],
        "authorities": list(practices.AUTHORITIES),
        "can_publish": principal.role in ("manager", "admin"),
    }


@app.post("/api/practices")
async def practices_create(request: Request):
    """Publish a practice. Only managers and administrators may — a
    practice is policy, and policy is not authored by everyone."""
    from agent import practices

    principal = _principal_for(request)
    if principal.role not in ("manager", "admin"):
        return JSONResponse(
            {"error": "manager or administrator role required",
             "your_role": principal.role},
            status_code=403,
        )
    body = await request.json()
    # Only an administrator may make a practice mandatory.
    authority = body.get("authority") or practices.RECOMMENDED
    if authority == practices.MANDATORY and not principal.is_admin:
        return JSONResponse(
            {"error": "only an administrator may publish a mandatory practice"},
            status_code=403,
        )
    try:
        p = practices.upsert(
            _board_sql,
            title=body.get("title") or "",
            guidance=body.get("guidance") or "",
            authority=authority,
            applies_to=body.get("applies_to") or [],
            capabilities=body.get("capabilities") or [],
            rationale=body.get("rationale") or "",
            examples=body.get("examples") or [],
            anti_patterns=body.get("anti_patterns") or [],
            classification=body.get("classification") or "internal",
            owner=principal.email,
            supersedes=body.get("supersedes") or "",
        )
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    return p.to_dict()


@app.post("/api/practices/{practice_id}/retire")
async def practices_retire(request: Request, practice_id: str):
    from agent import practices

    principal = _principal_for(request)
    if principal.role not in ("manager", "admin"):
        return JSONResponse({"error": "manager or administrator role required"},
                            status_code=403)
    return {"retired": practices.retire(_board_sql, practice_id)}


@app.get("/api/practices/match")
async def practices_match(request: Request, q: str = "", capabilities: str = ""):
    """Which practices govern this situation, and why they matched.

    This is the introspection surface for "why did Alex do that?" — the
    same matcher the prompt uses, so what you see here is what the model
    was actually told.
    """
    from agent import practices

    principal = _principal_for(request)
    caps = [c for c in capabilities.split(",") if c.strip()]
    matches = practices.match(
        _board_sql, principal, query=q, capabilities_in_play=caps
    )
    return {
        "query": q,
        "matches": [m.to_dict() for m in matches],
        "guidance_block": practices.guidance_block(matches),
    }


# ---------------------------------------------------------------------------
# Task board — the shared queue people and Alex work together.
# ---------------------------------------------------------------------------

_TASK_FORBIDDEN = {"error": "no such task, or you may not act on it"}


@app.get("/api/board/tasks")
async def board_tasks(request: Request):
    """The board, filtered to what the caller may see."""
    from agent import taskboard

    principal = _principal_for(request)
    tasks = taskboard.visible_tasks(_board_sql, principal)
    return {
        "columns": list(taskboard.COLUMNS),
        "tasks": tasks,
        "me": principal.email,
    }


@app.post("/api/board/tasks")
async def board_create(request: Request):
    from agent import taskboard

    principal = _principal_for(request)
    body = await request.json()
    try:
        created = taskboard.create_task(
            _board_sql,
            principal,
            title=body.get("title") or "",
            description=body.get("description") or "",
            status=body.get("status") or "todo",
            priority=int(body.get("priority") or 3),
            due_at=body.get("due_at"),
            crew=body.get("crew") or [],
        )
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    return created


@app.patch("/api/board/tasks/{task_id}")
async def board_update(request: Request, task_id: str):
    from agent import taskboard

    principal = _principal_for(request)
    if not taskboard.can_act_on(_board_sql, principal, task_id):
        return JSONResponse(_TASK_FORBIDDEN, status_code=403)
    body = await request.json()
    out: dict = {"task_id": task_id}
    if body.get("status"):
        out["status"] = taskboard.set_status(_board_sql, task_id, body["status"])
    if body.get("assignee"):
        taskboard.set_assignee(_board_sql, task_id, body["assignee"])
        out["assignee"] = body["assignee"]
    for member in body.get("add_crew") or []:
        taskboard.add_crew(_board_sql, task_id, member)
    # Acceptance criteria: the HUMAN write path (the agent may only mark
    # them met/unmet — see agent/task_acceptance.py).
    if "acceptance" in body:
        from agent import task_acceptance

        try:
            out["acceptance"] = task_acceptance.set_criteria(
                task_id, list(body.get("acceptance") or [])
            )
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
    # Dependencies. Both ends must be visible to the caller: a dep edge
    # to someone else's task would leak its title onto this board.
    if body.get("add_blocker") or body.get("remove_blocker"):
        from agent import task_deps

        for key, action in (("add_blocker", "add"), ("remove_blocker", "remove")):
            other = str(body.get(key) or "").strip()
            if not other:
                continue
            if not taskboard.can_act_on(_board_sql, principal, other):
                return JSONResponse(_TASK_FORBIDDEN, status_code=403)
            try:
                if action == "add":
                    task_deps.add(
                        _board_sql, task_id, other, added_by=principal.email
                    )
                else:
                    task_deps.remove(_board_sql, task_id, other)
            except ValueError as exc:  # includes DependencyCycle
                return JSONResponse({"error": str(exc)}, status_code=400)
        out["blocked_by"] = task_deps.open_blockers(_board_sql, task_id)
    return out


def _task_plan_of(task_id: str) -> list:
    try:
        from agent.task_plan import get_plan

        return get_plan(task_id)
    except Exception:  # noqa: BLE001
        return []


def _task_acceptance_of(task_id: str) -> list:
    try:
        from agent.task_acceptance import get_criteria

        return get_criteria(task_id)
    except Exception:  # noqa: BLE001
        return []


@app.get("/api/board/tasks/{task_id}")
async def board_task(request: Request, task_id: str):
    from agent import taskboard

    principal = _principal_for(request)
    task = taskboard.get_task(_board_sql, principal, task_id)
    if task is None:
        return JSONResponse(_TASK_FORBIDDEN, status_code=404)
    run = taskboard.run_for_task(task_id)
    return {
        **task,
        "messages": taskboard.messages(_board_sql, task_id),
        "run": run.to_dict(with_events=True) if run else None,
        # The plan — the run's structured spine, same rows the agent
        # writes, so the crew steers against the plan, not a guess.
        "plan": _task_plan_of(task_id),
        # The human's bar for the work, with the agent's verdicts.
        "acceptance": _task_acceptance_of(task_id),
        # What past runs on work like this discovered, so the crew
        # decides with the history in front of them.
        "past_work": taskboard.past_work(_board_sql, principal, task),
        # The agent's questions to the crew — open ones are the board's
        # "Alex is waiting on YOU" state.
        "questions": _task_questions_of(task_id),
    }


def _task_questions_of(task_id: str) -> list:
    try:
        from agent import task_questions

        return task_questions.for_task(_board_sql, task_id)
    except Exception:  # noqa: BLE001
        return []


@app.get("/api/board/tasks/{task_id}/messages")
async def board_messages(request: Request, task_id: str):
    """The thread plus the question state — one call, so the board's
    detail pane polls a single cheap route while it is open."""
    from agent import taskboard

    principal = _principal_for(request)
    if not taskboard.can_act_on(_board_sql, principal, task_id):
        return JSONResponse(_TASK_FORBIDDEN, status_code=403)
    return {
        "messages": taskboard.messages(_board_sql, task_id),
        "questions": _task_questions_of(task_id),
    }


@app.post("/api/board/tasks/{task_id}/questions/{question_id}/answer")
async def board_answer_question(
    request: Request, task_id: str, question_id: str
):
    """A crew member answers an open agent question. The answer lands in
    the thread, wakes the parked run (or steers a live one that already
    moved on), and closes the question."""
    from agent import task_questions, taskboard

    principal = _principal_for(request)
    if not taskboard.can_act_on(_board_sql, principal, task_id):
        return JSONResponse(_TASK_FORBIDDEN, status_code=403)
    body = await request.json()
    try:
        answered = await run_in_threadpool(
            task_questions.answer,
            _board_sql,
            principal,
            task_id,
            question_id,
            str((body or {}).get("answer") or ""),
        )
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    return answered


@app.post("/api/board/tasks/{task_id}/messages")
async def board_post_message(request: Request, task_id: str):
    """Post to the task thread. Humans and Alex share one record.

    When a run is live on this task the post is *also* steered into
    the running loop — a crew member can course-correct Alex without
    stopping the work. The message lands in the thread either way,
    because the thread is the record of the task regardless of who
    is working at the moment.
    """
    from agent import taskboard

    principal = _principal_for(request)
    if not taskboard.can_act_on(_board_sql, principal, task_id):
        return JSONResponse(_TASK_FORBIDDEN, status_code=403)
    body = await request.json()
    try:
        posted = taskboard.post_message(
            _board_sql,
            task_id,
            author=principal.email,
            body=body.get("body") or "",
            kind="human",
        )
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    posted["steered"] = taskboard.steer_task(
        task_id, principal.email, posted.get("body") or ""
    )
    return posted


@app.post("/api/board/tasks/{task_id}/assign-ai")
async def board_assign_ai(request: Request, task_id: str):
    """Hand the task to Alex. It opens its own VM session and works it."""
    from agent import taskboard

    principal = _principal_for(request)
    try:
        run = taskboard.start_run(_board_sql, principal, task_id)
    except PermissionError as exc:
        return JSONResponse({"error": str(exc)}, status_code=403)
    except taskboard.RunCapacity as exc:
        # 429: the app container is at its concurrent-run cap — a cost
        # and responsiveness bound, not a permission problem.
        return JSONResponse({"error": str(exc)}, status_code=429)
    except taskboard.TaskBlocked as exc:
        # 409: the task exists and the caller may act on it, but its
        # blockers are not done. The blockers ride along so the UI can
        # say WHICH tasks are in the way, not just "no".
        return JSONResponse(
            {"error": str(exc), "blocked_by": exc.blockers}, status_code=409
        )
    return run.to_dict()


@app.get("/api/board/proposals")
async def board_proposals(request: Request):
    """Work the system noticed in this employee's email, waiting to be
    adopted. Harvest-on-read: the extraction (email_intel) already
    happened; this only turns un-proposed asks into proposal rows, so the
    read is cheap and always current."""
    from agent import task_origination

    principal = _principal_for(request)
    harvested = task_origination.harvest(_board_sql, principal.email)
    return {
        "proposals": task_origination.pending(_board_sql, principal.email),
        "harvested": harvested,
    }


@app.post("/api/board/proposals/{proposal_id}")
async def board_decide_proposal(request: Request, proposal_id: str):
    """THE control gate on email-originated work: only this route — a
    person, reading the proposal — turns an email ask into a task."""
    from agent import task_origination

    principal = _principal_for(request)
    body = await request.json()
    action = str((body or {}).get("action") or "").strip().lower()
    try:
        if action == "accept":
            return task_origination.accept(_board_sql, principal, proposal_id)
        if action == "dismiss":
            return task_origination.dismiss(_board_sql, principal, proposal_id)
    except PermissionError as exc:
        return JSONResponse({"error": str(exc)}, status_code=403)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=409)
    return JSONResponse(
        {"error": "action must be 'accept' or 'dismiss'"}, status_code=400
    )


@app.post("/api/board/tasks/{task_id}/stop")
async def board_stop(request: Request, task_id: str):
    from agent import taskboard

    principal = _principal_for(request)
    if not taskboard.can_act_on(_board_sql, principal, task_id):
        return JSONResponse(_TASK_FORBIDDEN, status_code=403)
    return {"stopped": taskboard.stop_run(_board_sql, task_id)}


@app.get("/api/board/tasks/{task_id}/stream")
async def board_stream(request: Request, task_id: str):
    """Live feed of Alex working this task — the whole crew sees the
    same execution log, replayed from the start for late joiners."""
    from agent import taskboard

    principal = _principal_for(request)
    if not taskboard.can_act_on(_board_sql, principal, task_id):
        return JSONResponse(_TASK_FORBIDDEN, status_code=403)
    run = taskboard.run_for_task(task_id)
    if run is None:
        return JSONResponse({"error": "no run for this task"}, status_code=404)

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def _sink(event: dict) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, event)

    unsubscribe = run.subscribe(_sink)

    async def _gen():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                except TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("type") == "run_finished":
                    break
        finally:
            unsubscribe()

    return StreamingResponse(_gen(), media_type="text/event-stream")


@app.get("/api/supervisor/reports")
async def sup_reports(request: Request):
    ident = _identity(request)
    # Python-joined (see /api/transparency/library): the dev warehouse
    # answers a JOIN + GROUP BY with garbage rows, so this panel showed
    # every report with no capabilities however many they held.
    reports = _sql_as_user(
        request,
        "SELECT report_email FROM media.control.supervision "
        "WHERE supervisor_email = :u",
        {"u": ident.email},
    )
    out = []
    for r in reports:
        report = str(r.get("report_email") or "")
        if not report:
            continue
        caps = _sql(
            "SELECT capability FROM media.control.effective_employee_acl "
            "WHERE user_email = :e",
            {"e": report},
        )
        out.append({
            "report_email": report,
            "active_capabilities": len({
                c.get("capability") for c in caps if c.get("capability")
            }),
        })
    return {"reports": out}


@app.post("/api/supervisor/delegate")
async def sup_delegate(request: Request):
    ident = _identity(request)
    body = await request.json()
    report_email = (body.get("report_email") or "")[:200]
    # The caller must actually supervise the report. Without this check
    # any authenticated user could delegate capabilities to anyone.
    sup = _sql_as_user(
        request,
        """
        SELECT 1
          FROM media.control.supervision
         WHERE supervisor_email = :u AND report_email = :r
         LIMIT 1
    """,
        {"u": ident.email, "r": report_email},
    )
    if not sup:
        return JSONResponse(
            {"error": "you do not supervise this report"}, status_code=403
        )
    # The capability must be in the delegable list, and the
    # requires_second_approver rule is enforced in the procedure.
    rows = _sql_as_user(
        request,
        """
        SELECT capability, requires_second_approver, max_validity_days
          FROM media.control.delegable_capabilities
         WHERE capability = :c
    """,
        {"c": body.get("capability") or ""},
    )
    if not rows:
        return JSONResponse({"error": "capability is not delegable"}, status_code=400)
    requires_second = bool(rows[0].get("requires_second_approver"))
    max_days = int(rows[0].get("max_validity_days") or 30)
    if int(body.get("days") or 30) > max_days:
        return JSONResponse(
            {"error": f"validity exceeds the cap ({max_days} days)"}, status_code=400
        )
    second = body.get("second_approver")
    if requires_second and (not second or second == ident.email):
        return JSONResponse({"error": "second approver is required"}, status_code=400)
    days = max(1, min(int(body.get("days") or 30), max_days))
    delegation_id = str(uuid.uuid4())
    if requires_second:
        # Two-person control that is actually two-person: the named
        # second approver must confirm under their OWN identity before
        # the delegation becomes effective. The pending row is stored
        # revoked (the effective-ACL view activates only
        # revoked = false rows, and this also satisfies the
        # expires_at > granted_at CHECK with a nominal window).
        # A free-text approver name used to activate the capability on
        # the spot — the check existed only as a string.
        _sql_as_user(
            request,
            """
            INSERT INTO media.control.delegations
              (delegation_id, supervisor_email, report_email, capability,
               justification, second_approver, granted_at, expires_at,
               revoked)
            VALUES
              (:did, :u, :r, :c, :j, :s, current_timestamp(),
               current_timestamp() + INTERVAL 1 DAY, true)
        """,
            {
                "did": delegation_id,
                "u": ident.email,
                "r": report_email,
                "c": (body.get("capability") or "")[:60],
                "j": (body.get("justification") or "")[:500],
                "s": (second or "")[:200] or None,
            },
        )
        return {
            "status": "pending_second_approval",
            "delegation_id": delegation_id,
            "second_approver": second,
        }
    _sql_as_user(
        request,
        """
        INSERT INTO media.control.delegations
          (delegation_id, supervisor_email, report_email, capability,
           justification, second_approver, granted_at, expires_at, revoked)
        VALUES
          (:did, :u, :r, :c, :j, NULL, current_timestamp(),
           :expires, false)
    """,
        {
            "did": delegation_id,
            "u": ident.email,
            "r": report_email,
            "c": (body.get("capability") or "")[:60],
            "j": (body.get("justification") or "")[:500],
            # The window is computed here and bound as a value: a
            # parameter marker is not valid inside INTERVAL literal
            # syntax, and a fixed 30-day default would ignore the
            # capability's own max_validity_days cap.
            "expires": (
                datetime.now(timezone.utc) + timedelta(days=days)
            ).isoformat(),
        },
    )
    return {"status": "granted", "delegation_id": delegation_id}


@app.post("/api/supervisor/delegate/confirm")
async def sup_delegate_confirm(request: Request):
    """The second approver ratifies a pending delegation under their own
    identity. Only the named approver can confirm; the confirmation is
    what sets the validity window and makes the delegation effective."""
    ident = _identity(request)
    body = await request.json()
    delegation_id = (body.get("delegation_id") or "")[:64]
    rows = _sql_as_user(
        request,
        """
        SELECT second_approver, expires_at
          FROM media.control.delegations
         WHERE delegation_id = :d AND revoked = false
    """,
        {"d": delegation_id},
    )
    if not rows:
        return JSONResponse({"error": "no such delegation"}, status_code=404)
    named = rows[0].get("second_approver") or ""
    if named != ident.email:
        # Do not reveal who the named approver is to anyone else.
        return JSONResponse({"error": "not for you"}, status_code=403)
    cap_rows = _sql_as_user(
        request,
        """
        SELECT max_validity_days
          FROM media.control.delegable_capabilities
         WHERE capability = (SELECT capability FROM media.control.delegations
                              WHERE delegation_id = :d)
    """,
        {"d": delegation_id},
    )
    max_days = int(cap_rows[0].get("max_validity_days") or 30) if cap_rows else 30
    days = max(1, min(int(body.get("days") or 30), max_days))
    # Confirmation activates the delegation: revoked flips false and the
    # real window starts now, sized by the capability's cap. The expiry
    # is bound as a value (parameter markers are invalid inside
    # INTERVAL literal syntax).
    _sql_as_user(
        request,
        """
        UPDATE media.control.delegations
           SET revoked = false,
               granted_at = current_timestamp(),
               expires_at = :expires
         WHERE delegation_id = :did AND second_approver = :u
           AND revoked = true
    """,
        {
            "did": delegation_id,
            "u": ident.email,
            "expires": (
                datetime.now(timezone.utc) + timedelta(days=days)
            ).isoformat(),
        },
    )
    return {"status": "confirmed", "delegation_id": delegation_id}


@app.post("/api/supervisor/revoke")
async def sup_revoke(request: Request):
    ident = _identity(request)
    body = await request.json()
    _sql_as_user(
        request,
        """
        UPDATE media.control.delegations
           SET revoked = true, revoked_by = :u, revoked_at = current_timestamp()
         WHERE delegation_id = :d AND supervisor_email = :u
    """,
        {"u": ident.email, "d": (body.get("delegation_id") or "")[:64]},
    )
    return {"status": "revoked"}


# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------


@app.get("/api/skills")
async def skills_list(request: Request):
    ident = _identity(request)
    rows = _sql_as_user(
        request,
        """
        SELECT skill_id, name, trigger_pattern, procedure,
               uses_capabilities, status
          FROM media.control.skills
         WHERE owner_email = :u AND status IN ('proposed','active')
         ORDER BY status, created_at DESC
    """,
        {"u": ident.email},
    )
    return {"skills": rows}


@app.post("/api/skills/decide")
async def skills_decide(request: Request):
    ident = _identity(request)
    body = await request.json()
    new_status = "active" if body.get("action") == "ratify" else "rejected"
    # The DB-level check is: every used_capability must be in the
    # user's effective_employee_acl. If ratifying, the WHERE clause
    # excludes rows with missing capabilities. We rely on the procedure
    # `media.control.can_ratify_skill` for the assertion; the UPDATE
    # only sets status='active' if the procedure returns true.
    if new_status == "active":
        rows = _sql_as_user(
            request,
            """
            SELECT media.control.can_ratify_skill(:u, :s) AS ok
        """,
            {"u": ident.email, "s": (body.get("skill_id") or "")[:64]},
        )
        if not rows or not rows[0].get("ok"):
            return JSONResponse(
                {"error": "skill uses capabilities not in your effective ACL"},
                status_code=400,
            )
    _sql_as_user(
        request,
        """
        UPDATE media.control.skills
           SET status = :st, ratified_by = :u
         WHERE skill_id = :s AND owner_email = :u AND status = 'proposed'
    """,
        {"st": new_status, "u": ident.email, "s": (body.get("skill_id") or "")[:64]},
    )
    return {"status": new_status}


# ---------------------------------------------------------------------------
# Learning: /api/teach (explicit routine teaching), /api/feedback
# (thumbs on suggestions). Both write through agent.learn_with_you.
# ---------------------------------------------------------------------------


@app.post("/api/teach")
async def teach(request: Request):
    """Capture a /teach command: the user explicitly walks the agent
    through a procedure. Stored as a 'taught_routine' memory; the
    nightly reflection job may propose it as a skill (never
    self-ratified)."""
    ident = _identity(request)
    body = await request.json()
    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        return JSONResponse({"error": "prompt required"}, status_code=400)
    from agent.learn_with_you import teach_routine

    rid = teach_routine(
        _store_preference,
        ident.email,
        prompt[:4000],
        (body.get("context") or "")[:2000],
    )
    if not rid:
        return JSONResponse({"error": "could not store routine"}, status_code=500)
    return {"routine_id": rid, "status": "stored"}


@app.post("/api/feedback")
async def feedback(request: Request):
    """Thumbs up/down on a suggestion. Recorded via
    agent.learn_with_you.suggestion_feedback."""
    ident = _identity(request)
    body = await request.json()
    suggestion_id = (body.get("suggestion_id") or "")[:128]
    fb = body.get("feedback") or ""
    if not suggestion_id:
        return JSONResponse({"error": "suggestion_id required"}, status_code=400)
    if fb not in ("helpful", "unhelpful", "inaccurate", "other"):
        return JSONResponse(
            {"error": "feedback must be helpful|unhelpful|inaccurate|other"},
            status_code=400,
        )
    from agent.learn_with_you import suggestion_feedback

    suggestion_feedback(
        lambda stmt, parameters=None: _sql(stmt, parameters or {}),
        ident.email,
        suggestion_id,
        fb,
        (body.get("comment") or "")[:1000],
    )
    return {"status": "recorded"}


# ---------------------------------------------------------------------------
# Tools manifest — MCP-compatible list of the agent's registered tools,
# filtered to the caller's effective capabilities.
# ---------------------------------------------------------------------------


@app.get("/api/tools")
async def tools_manifest(request: Request):
    """Return the tool registry as an MCP-compatible tool list
    (name, description, inputSchema), filtered to the tools the
    signed-in user's effective ACL actually allows."""
    ident = _identity(request)
    from agent.tools.registry import TOOL_REGISTRY, input_schema_for

    allowed: set[str] = set()
    for stmt in (
        "SELECT capability FROM media.control.effective_employee_acl "
        "WHERE user_email = :u",
        "SELECT capability FROM media.control.employee_acl WHERE user_email = :u",
    ):
        try:
            allowed |= {
                str(r.get("capability"))
                for r in _sql(stmt, {"u": ident.email})
                if r.get("capability")
            }
        except Exception:  # noqa: BLE001
            log.debug("tools manifest ACL query failed", exc_info=True)
    tools = [
        {
            "name": spec.name,
            "description": spec.description,
            "inputSchema": input_schema_for(spec),
        }
        for spec in TOOL_REGISTRY.values()
        if spec.capability in allowed
    ]
    return {"tools": tools}


def _allowed_capabilities(email: str) -> set[str]:
    allowed: set[str] = set()
    try:
        allowed |= {
            str(r.get("capability"))
            for r in _sql(
                "SELECT capability FROM media.control.effective_employee_acl "
                "WHERE user_email = :u",
                {"u": email},
            )
            if r.get("capability")
        }
    except Exception:  # noqa: BLE001
        log.debug("ACL view query failed", exc_info=True)
    # Base-table fallback, expiry-filtered the same way guardrails does:
    # the view excludes expired grants in SQL, and a fallback that does
    # not would advertise (and pass) capabilities the source of truth
    # has already withdrawn.
    try:
        from agent.guardrails import _grant_is_live

        allowed |= {
            str(r.get("capability"))
            for r in _sql(
                "SELECT capability, expires_at FROM media.control.employee_acl "
                "WHERE user_email = :u",
                {"u": email},
            )
            if r.get("capability") and _grant_is_live(r)
        }
    except Exception:  # noqa: BLE001
        log.debug("ACL base query failed", exc_info=True)
    return allowed


@app.post("/api/tools/{name}")
async def invoke_tool(name: str, request: Request):
    """Invoke a single governed tool directly (the MCP call-tool seam).

    Same governance as the chat path: the tool must be in the registry,
    the user's effective ACL must allow its capability, and the call
    goes through the VM's ``_invoke_tool`` so the constitution,
    argument validation, and audit log all apply.  ``args`` is the
    JSON body (the LLM-facing parameters; identity is injected)."""
    ident = _identity(request)
    from agent.omnigent import ToolCallRequest
    from agent.tools.registry import TOOL_REGISTRY

    if not _rate_ok(ident.email):
        return JSONResponse(
            {"error": "rate_limited", "retry_after_s": int(_RATE_WINDOW_S)},
            status_code=429,
        )
    spec = TOOL_REGISTRY.get(name)
    if spec is None:
        return JSONResponse({"error": "unknown_tool"}, status_code=404)
    if spec.capability not in _allowed_capabilities(ident.email):
        return JSONResponse(
            {"error": "forbidden", "capability": spec.capability}, status_code=403
        )
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        body = {}
    args = body.get("args") if isinstance(body, dict) and "args" in body else body
    if not isinstance(args, dict):
        args = {}
    try:
        vm = _vm_cache_get(f"tool:{ident.email}", ident.email, "default")
        result = await run_in_threadpool(
            lambda: vm._invoke_tool(ToolCallRequest(name=name, args=args))
        )
    except Exception as exc:  # noqa: BLE001
        # The VM's _invoke_tool is designed never to raise, but the cache
        # or thread boundary might; return a clean error, never a 500.
        log.exception("tool invocation failed")
        return JSONResponse({"tool": name, "error": str(exc)}, status_code=500)
    # Guard against a tool returning an enormous payload over the wire.
    import json as _json

    try:
        size = len(_json.dumps(result, default=str))
    except Exception:  # noqa: BLE001
        size = 0
    if size > 2_000_000:
        return {
            "tool": name,
            "result": {
                "ok": True,
                "truncated": True,
                "note": "result too large to return inline; it was produced "
                "in your workspace — fetch it via /api/files if applicable",
            },
        }
    return {"tool": name, "result": result}


# ---------------------------------------------------------------------------
# n8n — the enterprise workflow arm. Same capability gates as the agent
# tools (workflow.read / workflow.create / workflow.trigger); the base
# URL and API key are operator configuration the routes never accept.
# ---------------------------------------------------------------------------


def _has_cap(request: Request, capability: str):
    principal = _principal_for(request)
    if capability in principal.capabilities:
        return principal, None
    return principal, JSONResponse(
        {"error": "forbidden", "capability": capability}, status_code=403
    )


@app.get("/api/n8n/workflows")
async def n8n_workflows(request: Request):
    from agent import n8n_client

    _p, denied = _has_cap(request, "workflow.read")
    if denied is not None:
        return denied
    try:
        rows = await run_in_threadpool(n8n_client.list_workflows)
    except n8n_client.N8NError as exc:
        return JSONResponse({"error": str(exc)}, status_code=502)
    return {"workflows": rows, "health": n8n_client.health()}


@app.post("/api/n8n/workflows")
async def n8n_create(request: Request):
    from agent import n8n_client

    principal, denied = _has_cap(request, "workflow.create")
    if denied is not None:
        return denied
    body = await request.json()
    try:
        out = await run_in_threadpool(
            lambda: n8n_client.create_workflow(
                name=str(body.get("name") or ""),
                nodes=list(body.get("nodes") or []),
                connections=dict(body.get("connections") or {}),
                created_by=principal.email,
            )
        )
    except n8n_client.N8NError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    return out


@app.post("/api/n8n/workflows/{workflow_id}/run")
async def n8n_run(request: Request, workflow_id: str):
    from agent import n8n_client

    _p, denied = _has_cap(request, "workflow.trigger")
    if denied is not None:
        return denied
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        body = {}
    try:
        out = await run_in_threadpool(
            lambda: n8n_client.run_workflow(
                workflow_id, dict((body or {}).get("payload") or {})
            )
        )
    except n8n_client.N8NError as exc:
        return JSONResponse({"error": str(exc)}, status_code=502)
    return out


@app.post("/api/n8n/workflows/{workflow_id}/activate")
async def n8n_activate(request: Request, workflow_id: str):
    from agent import n8n_client

    _p, denied = _has_cap(request, "workflow.create")
    if denied is not None:
        return denied
    body = await request.json()
    try:
        out = await run_in_threadpool(
            lambda: n8n_client.set_active(
                workflow_id, bool((body or {}).get("active", True))
            )
        )
    except n8n_client.N8NError as exc:
        return JSONResponse({"error": str(exc)}, status_code=502)
    return out


@app.get("/api/n8n/executions/{execution_id}")
async def n8n_execution(request: Request, execution_id: str):
    from agent import n8n_client

    _p, denied = _has_cap(request, "workflow.read")
    if denied is not None:
        return denied
    try:
        out = await run_in_threadpool(
            lambda: n8n_client.execution(execution_id)
        )
    except n8n_client.N8NError as exc:
        return JSONResponse({"error": str(exc)}, status_code=502)
    return out


# ---------------------------------------------------------------------------
# Browser OS — sandboxed browser sessions, tabs, and agent operations.
#
# The agent uses these routes to drive the user-facing Browser OS shell.
# Every operation is OBO-scoped (the user's identity), every change is
# audit-logged to ``media.alex.agent_actions`` by the browser tool
# itself (constitution A3), and destructive submits are two-phase per
# constitution A8.
# ---------------------------------------------------------------------------


def _get_browser_os():
    """Return the configured :class:`agent.browser_os.BrowserOS`.

    In dev the singleton lives on the fakes; in production this
    resolves to a real Playwright-backed instance through the same
    fakes container or a Databricks Volume-backed profile store.
    """
    from agent.browser_os import browser_os_for_mode
    from alex.testing import get_fakes

    fakes = get_fakes()
    bos = getattr(fakes, "browser_os", None)
    if bos is None:
        bos = browser_os_for_mode()
        fakes.browser_os = bos
    return bos


def _owned_browser_session(bos, ident: Identity, session_id: str) -> bool:
    """OBO guard: True iff the session exists AND belongs to the caller.
    Every /api/browser/sessions/{sid}/... route must call this before
    touching the session."""
    return any(s.session_id == session_id for s in bos.list_sessions(ident.email))


_SESSION_NOT_FOUND = {"error": "session not found"}


def _sse_sink():
    """Return a function the browser tools push events through.

    In dev, ``alex.testing.Fakes.sse_sink`` is a recording callable
    the test harness inspects.  In production, the FastAPI route
    subscribes its per-request stream generator through a
    module-level registry keyed by ``(session_id, tab_id)``.
    """
    from alex.testing import get_fakes

    fakes = get_fakes()
    return getattr(fakes, "sse_sink", None)


@app.get("/api/browser/sessions")
async def browser_list_sessions(request: Request):
    """List the current user's browser sessions."""
    ident = _identity(request)
    bos = _get_browser_os()
    return {
        "sessions": [
            {
                "session_id": s.session_id,
                "tab_count": len(s.tabs),
                "created_at": s.created_at,
                "last_active_at": s.last_active_at,
            }
            for s in bos.list_sessions(ident.email)
        ]
    }


@app.post("/api/browser/sessions")
async def browser_create_session(request: Request):
    """Create a new browser session for the signed-in user."""
    ident = _identity(request)
    bos = _get_browser_os()
    s = bos.create_session(user_email=ident.email, instance_scope=ident.user_id)
    return {"session_id": s.session_id}


@app.delete("/api/browser/sessions/{session_id}")
async def browser_close_session(request: Request, session_id: str):
    """Close a browser session. Refuses to close sessions owned by
    a different user."""
    ident = _identity(request)
    bos = _get_browser_os()
    # OBO guard: only the owner can close.
    owned = [s for s in bos.list_sessions(ident.email) if s.session_id == session_id]
    if not owned:
        return JSONResponse({"error": "session not found"}, status_code=404)
    bos.close_session(session_id)
    return {"ok": True}


@app.get("/api/browser/sessions/{session_id}/tabs")
async def browser_list_tabs(request: Request, session_id: str):
    """List the tabs in a browser session."""
    ident = _identity(request)
    bos = _get_browser_os()
    if not _owned_browser_session(bos, ident, session_id):
        return JSONResponse(_SESSION_NOT_FOUND, status_code=404)
    try:
        tabs = bos.list_tabs(session_id)
    except KeyError:
        return JSONResponse({"error": "session not found"}, status_code=404)
    return {
        "tabs": [
            {
                "tab_id": t.tab_id,
                "url": t.url,
                "title": t.title,
                "agent_piloting": t.agent_piloting,
                "last_status": t.last_status,
            }
            for t in tabs
        ]
    }


@app.post("/api/browser/sessions/{session_id}/tabs")
async def browser_create_tab(
    request: Request, session_id: str, url: str = "about:blank"
):
    """Open a new tab in the given session."""
    ident = _identity(request)
    bos = _get_browser_os()
    if not _owned_browser_session(bos, ident, session_id):
        return JSONResponse(_SESSION_NOT_FOUND, status_code=404)
    try:
        tab = bos.create_tab(session_id, url=url)
    except KeyError:
        return JSONResponse({"error": "session not found"}, status_code=404)
    return {
        "tab_id": tab.tab_id,
        "url": tab.url,
        "title": tab.title,
        "last_status": tab.last_status,
    }


@app.post("/api/browser/sessions/{session_id}/tabs/{tab_id}/navigate")
async def browser_navigate(request: Request, session_id: str, tab_id: str):
    """Drive a tab to a new URL via the agentic browser tool."""
    ident = _identity(request)
    if not _owned_browser_session(_get_browser_os(), ident, session_id):
        return JSONResponse(_SESSION_NOT_FOUND, status_code=404)
    body = await request.json()
    url = (body.get("url") or "").strip()
    if not url:
        return JSONResponse({"error": "url is required"}, status_code=400)
    from agent import agentic_browser as ab

    res = ab.browser_navigate(
        session_id=session_id,
        tab_id=tab_id,
        url=url,
        user_email=ident.email,
        instance_scope=ident.user_id,
    )
    status = 200 if res.get("ok") else 403
    return JSONResponse(res, status_code=status)


@app.post("/api/browser/sessions/{session_id}/tabs/{tab_id}/click")
async def browser_click(request: Request, session_id: str, tab_id: str):
    ident = _identity(request)
    if not _owned_browser_session(_get_browser_os(), ident, session_id):
        return JSONResponse(_SESSION_NOT_FOUND, status_code=404)
    body = await request.json()
    selector = (body.get("selector") or "").strip()
    if not selector:
        return JSONResponse({"error": "selector is required"}, status_code=400)
    from agent import agentic_browser as ab

    res = ab.browser_click(
        session_id=session_id,
        tab_id=tab_id,
        selector=selector,
        user_email=ident.email,
        instance_scope=ident.user_id,
    )
    status = 200 if res.get("ok") else 403
    return JSONResponse(res, status_code=status)


@app.post("/api/browser/sessions/{session_id}/tabs/{tab_id}/type")
async def browser_type(request: Request, session_id: str, tab_id: str):
    ident = _identity(request)
    if not _owned_browser_session(_get_browser_os(), ident, session_id):
        return JSONResponse(_SESSION_NOT_FOUND, status_code=404)
    body = await request.json()
    selector = (body.get("selector") or "").strip()
    text = body.get("text") or ""
    if not selector:
        return JSONResponse({"error": "selector is required"}, status_code=400)
    from agent import agentic_browser as ab

    res = ab.browser_type(
        session_id=session_id,
        tab_id=tab_id,
        selector=selector,
        text=text,
        user_email=ident.email,
        instance_scope=ident.user_id,
    )
    status = 200 if res.get("ok") else 403
    return JSONResponse(res, status_code=status)


@app.post("/api/browser/sessions/{session_id}/tabs/{tab_id}/extract")
async def browser_extract(request: Request, session_id: str, tab_id: str):
    ident = _identity(request)
    if not _owned_browser_session(_get_browser_os(), ident, session_id):
        return JSONResponse(_SESSION_NOT_FOUND, status_code=404)
    body = await request.json()
    selector = (body.get("selector") or "body").strip()
    from agent import agentic_browser as ab

    res = ab.browser_extract(
        session_id=session_id,
        tab_id=tab_id,
        selector=selector,
        user_email=ident.email,
        instance_scope=ident.user_id,
    )
    return res


@app.post("/api/browser/sessions/{session_id}/tabs/{tab_id}/screenshot")
async def browser_screenshot(request: Request, session_id: str, tab_id: str):
    ident = _identity(request)
    if not _owned_browser_session(_get_browser_os(), ident, session_id):
        return JSONResponse(_SESSION_NOT_FOUND, status_code=404)
    from agent import agentic_browser as ab

    res = ab.browser_screenshot(
        session_id=session_id,
        tab_id=tab_id,
        user_email=ident.email,
        instance_scope=ident.user_id,
    )
    return res


@app.post("/api/browser/sessions/{session_id}/tabs/{tab_id}/submit")
async def browser_submit(request: Request, session_id: str, tab_id: str):
    """Two-phase form submit. Pass ``confirmation_token`` only on the
    second call to actually submit (constitution A8).

    This is the *authenticated UI* path, so the staged token is
    attached here for the frontend's Approve call — the agent tool
    path returns no token, because that result reaches the model."""
    ident = _identity(request)
    if not _owned_browser_session(_get_browser_os(), ident, session_id):
        return JSONResponse(_SESSION_NOT_FOUND, status_code=404)
    body = await request.json()
    selector = (body.get("selector") or "form").strip()
    confirmation_token = body.get("confirmation_token")
    from agent import agentic_browser as ab

    res = ab.browser_submit(
        session_id=session_id,
        tab_id=tab_id,
        selector=selector,
        confirmation_token=confirmation_token,
        user_email=ident.email,
        instance_scope=ident.user_id,
    )
    if res.get("awaiting_confirmation"):
        pending = ab._PENDING_CONFIRMATIONS.get((session_id, tab_id))
        if pending is not None and pending["user_email"] == ident.email:
            res["confirmation_token"] = pending["token"]
    status = 200 if res.get("ok") or res.get("awaiting_confirmation") else 403
    return JSONResponse(res, status_code=status)


@app.post("/api/browser/sessions/{session_id}/tabs/{tab_id}/mirror")
async def browser_mirror(request: Request, session_id: str, tab_id: str):
    """Set the user-control mode for a tab: ``read``, ``co-drive``,
    or ``yield``."""
    ident = _identity(request)
    if not _owned_browser_session(_get_browser_os(), ident, session_id):
        return JSONResponse(_SESSION_NOT_FOUND, status_code=404)
    body = await request.json()
    mode = (body.get("mode") or "").strip()
    if mode not in {"read", "co-drive", "yield"}:
        return JSONResponse(
            {"error": "mode must be one of read|co-drive|yield"}, status_code=400
        )
    from agent import agentic_browser as ab

    res = ab.browser_open_in_user_session(
        session_id=session_id,
        tab_id=tab_id,
        mode=mode,
        user_email=ident.email,
        instance_scope=ident.user_id,
    )
    return res


@app.delete("/api/browser/sessions/{session_id}/tabs/{tab_id}")
async def browser_close_tab(request: Request, session_id: str, tab_id: str):
    ident = _identity(request)
    if not _owned_browser_session(_get_browser_os(), ident, session_id):
        return JSONResponse(_SESSION_NOT_FOUND, status_code=404)
    from agent import agentic_browser as ab

    res = ab.browser_close_tab(
        session_id=session_id,
        tab_id=tab_id,
        user_email=ident.email,
        instance_scope=ident.user_id,
    )
    return res


@app.get("/api/browser/sessions/{session_id}/stream")
async def browser_stream(request: Request, session_id: str):
    """SSE stream of the browser session's events.

    The browser tool surface emits alex.events.Event instances through
    the fakes' ``sse_sink``; in dev the fakes' sink is a recording list.
    In production the route subscribes a per-request queue to that sink
    and re-emits each event as a ``data: <json>\\n\\n`` SSE chunk.
    """
    import asyncio

    ident = _identity(request)
    bos = _get_browser_os()
    # OBO guard
    if not any(s.session_id == session_id for s in bos.list_sessions(ident.email)):
        return JSONResponse({"error": "session not found"}, status_code=404)

    queue: asyncio.Queue = asyncio.Queue()

    def _push(event) -> None:
        try:
            queue.put_nowait(event)
        except Exception:  # noqa: BLE001
            pass

    # Register THIS request's push under the session id — additive, so a
    # concurrent stream for another session never sees these events (the old
    # code overwrote a single process-global sink -> cross-user event leak).
    from alex import testing

    subs = testing.get_fakes().browser_subscribers
    subs.setdefault(session_id, set()).add(_push)

    async def _gen():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    ev = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                # Defense in depth: only emit events for THIS session.
                if getattr(ev, "session_id", None) not in (None, "", session_id):
                    continue
                yield f"data: {ev.to_json()}\n\n"
        finally:
            # Deregister only our own callback.
            bucket = subs.get(session_id)
            if bucket is not None:
                bucket.discard(_push)
                if not bucket:
                    subs.pop(session_id, None)

    from fastapi.responses import StreamingResponse

    return StreamingResponse(_gen(), media_type="text/event-stream")


def _chain_sinks(sinks: list) -> object:
    """Build a sink callable that forwards to every supplied sink in order.

    Used by the browser SSE stream route so a single ``Event`` is
    fanned out to both the test recorder and the per-request
    subscriber queue.
    """

    def _call(ev) -> None:
        for s in sinks:
            try:
                s(ev)
            except Exception:  # noqa: BLE001
                pass

    return _call


@app.get("/api/constitution")
async def constitution(request: Request):
    """The constitution, as visible to the signed-in user. We
    return a snapshot of the active articles."""
    _identity(request)
    rows = _sql_as_user(
        request,
        """
        SELECT article_id, article_type, title, text
          FROM media.control.constitution
         WHERE superseded_by IS NULL
           AND effective_from <= current_timestamp()
         ORDER BY article_id
    """,
        {},
    )
    return {"articles": rows}


# ---------------------------------------------------------------------------
# Frontend (SPA) — served from the dist directory. The build step
# is part of the bundle's `build` block in app.yaml.
# ---------------------------------------------------------------------------


_DIST = os.path.join(os.path.dirname(__file__), "frontend", "dist")
if os.path.isdir(os.path.join(_DIST, "assets")):
    app.mount(
        "/assets", StaticFiles(directory=os.path.join(_DIST, "assets")), name="assets"
    )


# ---------------------------------------------------------------------------
# Voice — hear and speak (agent.voice: whisper on the GPU, edge-tts)
# ---------------------------------------------------------------------------


@app.get("/api/voice/status")
async def voice_status(request: Request):
    _identity(request)
    from agent import voice

    return voice.status()


@app.post("/api/voice/transcribe")
async def voice_transcribe(request: Request, audio: UploadFile):
    """Audio from the browser (MediaRecorder webm/ogg/wav) -> text."""
    _identity(request)
    from agent import voice

    data = await audio.read()
    if len(data) > 25 * 1024 * 1024:
        return JSONResponse({"error": "audio too large"}, status_code=413)
    name = audio.filename or "clip.webm"
    suffix = "." + name.rsplit(".", 1)[-1] if "." in name else ".webm"
    out = await run_in_threadpool(voice.transcribe, data, suffix=suffix)
    if not out.get("ok"):
        return JSONResponse(out, status_code=503)
    return out


@app.post("/api/voice/speak")
async def voice_speak(request: Request):
    """Text -> audio (MP3). 204 when no TTS backend can speak, so the
    client degrades to text without inventing audio."""
    _identity(request)
    from agent import voice

    body = await request.json()
    text = str((body or {}).get("text") or "")
    audio, mime = await voice.synthesize(text, (body or {}).get("voice"))
    if not audio:
        return Response(status_code=204)
    return Response(content=audio, media_type=mime)


@app.get("/{path:path}")
async def spa(path: str):
    """SPA fallback. Any non-API route returns index.html so the
    client-side router can take over."""
    index = os.path.join(_DIST, "index.html")
    if not os.path.isfile(index):
        return JSONResponse(
            {"note": "frontend not built yet; run npm run build"}, status_code=200
        )
    return FileResponse(index)
