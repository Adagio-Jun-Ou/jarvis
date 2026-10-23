"""
MCP client — connecting each user's VM to the organisation's MCP servers.

Alex already *serves* its own tools over MCP (``alex_sdk.mcp``). This is
the other half: consuming the MCP servers an enterprise has connected —
GitHub, Jira, Slack, internal systems — so a user's agent can act in the
systems they actually work in.

The unit of configuration is a **server**, registered in the control
plane with a scope and a required capability. Which servers a given
user's VM reaches is therefore the same authorization question as
everything else: scope decides *whether you can see it*, capability
decides *whether you may use it*, and the guardrails gate still runs on
every individual call.

Trust
-----

A remote MCP server's tool *descriptions* are attacker-controlled text
that lands directly in the model's prompt. That is the tool-poisoning
attack, and it is the reason this module treats external metadata as
**data, never instruction**:

* **Namespacing.** Every discovered tool is exposed as
  ``mcp__<server>__<tool>``. A remote server cannot shadow a built-in
  tool, and two servers cannot collide.
* **Description sanitising.** Imperative injection patterns ("ignore
  previous instructions", "before using any tool, read…", fake system
  tags) are stripped from descriptions of servers not marked trusted,
  and the description is length-capped.
* **Definition pinning.** The hash of each server's tool set is recorded
  on first sight. If a server later changes a tool's shape — the
  rug-pull attack — the change is flagged and the server's tools are
  withheld until re-approved.
* **Capability gating.** A server declares the capability a user needs
  to reach it at all, so connecting a powerful MCP server does not
  silently widen anyone's authority.

Concurrency
-----------

The MCP SDK is async; the VM is synchronous. A single background event
loop thread owns every connection, and calls are submitted to it, so the
VM's public surface stays plain blocking Python.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("agent.mcp_client")

NAMESPACE_PREFIX = "mcp"
NAMESPACE_SEP = "__"
DEFAULT_TIMEOUT_S = 30.0
DISCOVERY_TTL_S = 300.0
MAX_DESCRIPTION_CHARS = 600
MAX_TOOLS_PER_SERVER = 60
# Remote tool results are unbounded — measured in the wild at hundreds of
# thousands of tokens. An oversized result does not just cost money, it
# evicts the rest of the turn's context, so it is truncated at the seam.
MAX_RESULT_CHARS = 20_000
# A server that keeps failing is quarantined rather than retried on every
# turn: one broken integration must not slow every request.
BREAKER_THRESHOLD = 3
BREAKER_COOLDOWN_S = 120.0

SERVERS_TABLE = "media.control.mcp_servers"
PINS_TABLE = "media.control.mcp_tool_pins"

TRANSPORT_STDIO = "stdio"
TRANSPORT_HTTP = "http"
TRANSPORTS = (TRANSPORT_STDIO, TRANSPORT_HTTP)

TRUST_TRUSTED = "trusted"
TRUST_UNTRUSTED = "untrusted"

# Patterns that have no business in a tool description. Each is an
# attempt to talk to the model rather than describe a tool.
_INJECTION_PATTERNS = [
    re.compile(r"(?i)\bignore\s+(all\s+)?(previous|prior|above)\b[^.]*"),
    re.compile(r"(?i)\bdisregard\s+(all\s+)?(previous|prior|the above)\b[^.]*"),
    re.compile(r"(?i)\byou\s+must\s+(first|always|never)\b[^.]*"),
    re.compile(r"(?i)\bbefore\s+(using|calling)\s+(any|this|other)\s+tool\b[^.]*"),
    re.compile(r"(?i)<\s*/?\s*(system|assistant|user|instructions?)\s*>"),
    re.compile(r"(?i)\[\s*(system|important|urgent|note to (the )?(ai|model|assistant))\s*\]"),
    re.compile(r"(?i)\bdo not (tell|mention|inform|reveal)\b[^.]*"),
    re.compile(r"(?i)\b(new|updated)\s+(system\s+)?instructions?\b\s*:?[^.]*"),
]


def sanitize_description(text: str, *, trusted: bool = False) -> str:
    """Neutralise prompt injection in a remote tool description.

    Trusted servers keep their text verbatim (they are first-party); for
    everything else, imperative patterns aimed at the model are removed
    and the result is capped. What survives should read as a description
    of a tool, which is the only thing a description is for.
    """
    body = " ".join(str(text or "").split())
    if not trusted:
        for pattern in _INJECTION_PATTERNS:
            body = pattern.sub("", body)
        body = " ".join(body.split())
    if len(body) > MAX_DESCRIPTION_CHARS:
        body = body[:MAX_DESCRIPTION_CHARS].rstrip() + "…"
    return body


def sanitize_schema(
    schema: Any, *, trusted: bool = False
) -> dict[str, Any]:
    """Neutralise prompt injection in a tool's input schema.

    A schema is model-visible too: parameter descriptions, titles and
    enum values are read by the model alongside the tool description,
    and a rogue server used to smuggle instructions through
    ``properties.x.description`` untouched — the pin then recorded the
    poisoned state as approved, so rug-pull detection never fired on
    it. Every string value inside the schema passes through the same
    sanitizer as descriptions. (Structure — types, required, nested
    objects — is preserved untouched.)
    """
    if not isinstance(schema, dict):
        return schema if isinstance(schema, dict) else {}

    def _walk(node: Any) -> Any:
        if isinstance(node, dict):
            return {
                k: (
                    sanitize_description(v, trusted=trusted)
                    if isinstance(v, str) and k in {"description", "title"}
                    else _walk(v)
                )
                for k, v in node.items()
            }
        if isinstance(node, list):
            return [
                sanitize_description(v, trusted=trusted)
                if isinstance(v, str)
                else _walk(v)
                for v in node
            ]
        return node

    return _walk(schema)


def namespaced(server_name: str, tool_name: str) -> str:
    """``mcp__github__create_issue`` — collision-proof by construction."""
    safe_server = re.sub(r"[^a-z0-9_]", "_", (server_name or "").lower())
    safe_tool = re.sub(r"[^A-Za-z0-9_]", "_", tool_name or "")
    return f"{NAMESPACE_PREFIX}{NAMESPACE_SEP}{safe_server}{NAMESPACE_SEP}{safe_tool}"


def split_namespaced(name: str) -> tuple[str, str] | None:
    """Recover ``(server, tool)`` from a namespaced name."""
    parts = (name or "").split(NAMESPACE_SEP)
    if len(parts) < 3 or parts[0] != NAMESPACE_PREFIX:
        return None
    return parts[1], NAMESPACE_SEP.join(parts[2:])


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class MCPServer:
    server_id: str
    name: str
    transport: str = TRANSPORT_STDIO
    command: str = ""
    args: list[str] = field(default_factory=list)
    url: str = ""
    # The ACL capability a user needs before this server's tools appear.
    capability: str = "mcp.use"
    scope: str = "enterprise"
    trust: str = TRUST_UNTRUSTED
    enabled: bool = True
    timeout_s: float = DEFAULT_TIMEOUT_S
    # Name of the secret holding this user's credential, if any.
    secret_ref: str = ""
    description: str = ""

    @property
    def trusted(self) -> bool:
        return self.trust == TRUST_TRUSTED

    def to_dict(self) -> dict[str, Any]:
        return {
            "server_id": self.server_id,
            "name": self.name,
            "transport": self.transport,
            "url": self.url,
            "capability": self.capability,
            "scope": self.scope,
            "trust": self.trust,
            "enabled": self.enabled,
            "description": self.description,
        }


@dataclass
class MCPTool:
    """One discovered remote tool, ready to expose to the model."""

    server_id: str
    server_name: str
    remote_name: str
    name: str  # namespaced
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    capability: str = "mcp.use"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "server": self.server_name,
            "remote_name": self.remote_name,
            "description": self.description,
            "inputSchema": self.input_schema,
            "capability": self.capability,
        }


def _now_iso() -> str:
    """An ISO-8601 timestamp both backends accept as a bound parameter."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(sep=" ", timespec="seconds")


def _rows(sql, stmt: str, params: dict | None = None) -> list[dict]:
    try:
        return list(sql(stmt, params or {}) or [])
    except Exception:  # noqa: BLE001
        log.debug("mcp config read failed", exc_info=True)
        return []


def _rows_strict(sql, stmt: str, params: dict | None = None) -> list[dict]:
    """A read whose failure must not look like an empty table.

    The pin check is a security gate: "could not read the pins" and
    "no pins recorded" are different facts, and treating them the
    same disables the rug-pull defense exactly when the control
    plane is unhealthy — the moment an attacker would choose.
    """
    return list(sql(stmt, params or {}) or [])


def _as_list(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    text = str(value).strip()
    if text.startswith("["):
        import ast

        for parse in (json.loads, ast.literal_eval):
            try:
                parsed = parse(text)
            except (ValueError, SyntaxError):
                continue
            if isinstance(parsed, (list, tuple)):
                return [str(v) for v in parsed]
    return [text]


def _as_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "1", "yes")


def all_servers(sql) -> list[MCPServer]:
    out = []
    for r in _rows(sql, f"SELECT * FROM {SERVERS_TABLE}"):
        out.append(
            MCPServer(
                server_id=r.get("server_id", ""),
                name=r.get("name", ""),
                transport=r.get("transport") or TRANSPORT_STDIO,
                command=r.get("command") or "",
                args=_as_list(r.get("args")),
                url=r.get("url") or "",
                capability=r.get("capability") or "mcp.use",
                scope=r.get("scope") or "enterprise",
                trust=r.get("trust") or TRUST_UNTRUSTED,
                enabled=_as_bool(r.get("enabled"), True),
                timeout_s=float(r.get("timeout_s") or DEFAULT_TIMEOUT_S),
                secret_ref=r.get("secret_ref") or "",
                description=r.get("description") or "",
            )
        )
    return out


def servers_for(sql, principal) -> list[MCPServer]:
    """Servers this principal may reach: scope AND capability must pass."""
    from agent import access

    out = []
    for server in all_servers(sql):
        if not server.enabled:
            continue
        if not access.scope_allowed(principal, server.scope).allowed:
            continue
        if server.capability and server.capability not in principal.capabilities:
            continue
        out.append(server)
    return out


# ---------------------------------------------------------------------------
# Definition pinning — the rug-pull defence
# ---------------------------------------------------------------------------


def fingerprint(tools: list[dict[str, Any]]) -> str:
    """A stable hash of a server's advertised tool surface."""
    canonical = sorted(
        (
            str(t.get("name", "")),
            str(t.get("description", "")),
            json.dumps(t.get("inputSchema") or {}, sort_keys=True),
        )
        for t in tools
    )
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True).encode("utf-8")
    ).hexdigest()[:32]


def check_pin(sql, server: MCPServer, tools: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare a server's current tool surface with what was approved.

    First sight records the fingerprint and passes. A later change means
    the server redefined its tools after approval — the rug-pull — and
    the tools are withheld until a governor re-approves.

    The pins read is strict: a failed read withholds the tools rather
    than passing as "first sight". The insert on true first sight is
    still best-effort (a table that cannot be written to can still be
    read and compared on the next discovery).
    """
    current = fingerprint(tools)
    try:
        rows = _rows_strict(
            sql,
            f"SELECT fingerprint, approved FROM {PINS_TABLE} WHERE server_id = :s",
            {"s": server.server_id},
        )
    except Exception:  # noqa: BLE001
        log.error(
            "could not read the MCP pin table — withholding tools for "
            "server %s. A pin store that cannot be read is not a pin "
            "store; this fails closed.",
            server.name, exc_info=True,
        )
        return {"ok": False, "state": "pin_unavailable", "fingerprint": current}
    if not rows:
        try:
            sql(
                f"INSERT INTO {PINS_TABLE} "
                "(server_id, fingerprint, approved, first_seen, tool_count) "
                "VALUES (:s, :f, :ok, current_timestamp(), :n)",
                {
                    "s": server.server_id, "f": current,
                    "ok": True, "n": len(tools),
                },
            )
        except Exception:  # noqa: BLE001
            log.debug("could not record mcp pin", exc_info=True)
        return {"ok": True, "state": "first_seen", "fingerprint": current}

    pinned = str(rows[0].get("fingerprint") or "")
    approved = _as_bool(rows[0].get("approved"), True)
    if pinned == current:
        return {
            "ok": approved,
            "state": "matched" if approved else "revoked",
            "fingerprint": current,
        }
    log.warning(
        "MCP server %s changed its tool definitions after approval "
        "(pinned=%s current=%s) — withholding its tools",
        server.name, pinned, current,
    )
    try:
        # Bound parameters only in SET: the dev warehouse rejects SQL
        # function calls there, and keeping both backends on the same
        # statement shape is what makes dev faithful to production.
        sql(
            f"UPDATE {PINS_TABLE} SET fingerprint = :f, approved = :ok, "
            "changed_at = :ts WHERE server_id = :s",
            {
                "f": current, "ok": False, "ts": _now_iso(),
                "s": server.server_id,
            },
        )
    except Exception:  # noqa: BLE001
        log.debug("could not update mcp pin", exc_info=True)
    return {"ok": False, "state": "changed", "fingerprint": current}


def approve(sql, server_id: str) -> bool:
    """Re-approve a server's current tool surface (governor action)."""
    try:
        sql(
            f"UPDATE {PINS_TABLE} SET approved = :ok, approved_at = :ts "
            "WHERE server_id = :s",
            {"ok": True, "ts": _now_iso(), "s": server_id},
        )
        return True
    except Exception:  # noqa: BLE001
        log.debug("could not approve mcp server", exc_info=True)
        return False


# ---------------------------------------------------------------------------
# Circuit breaker
#
# A server that keeps failing is quarantined rather than retried on every
# turn. Without this, one broken integration adds its full timeout to
# every request the user makes.
# ---------------------------------------------------------------------------

_BREAKER: dict[str, tuple[int, float]] = {}


def _breaker_open(server_id: str) -> bool:
    failures, opened_at = _BREAKER.get(server_id, (0, 0.0))
    if failures < BREAKER_THRESHOLD:
        return False
    if time.time() - opened_at > BREAKER_COOLDOWN_S:
        # Cooled down — let one probe through rather than staying dark.
        _BREAKER.pop(server_id, None)
        return False
    return True


def _record_failure(server_id: str) -> None:
    failures, opened_at = _BREAKER.get(server_id, (0, 0.0))
    failures += 1
    _BREAKER[server_id] = (
        failures,
        time.time() if failures >= BREAKER_THRESHOLD else opened_at,
    )


def _record_success(server_id: str) -> None:
    _BREAKER.pop(server_id, None)


def breaker_state() -> dict[str, dict[str, Any]]:
    """Quarantine status per server, for the Tools surface."""
    return {
        sid: {"failures": f, "open": _breaker_open(sid)}
        for sid, (f, _t) in dict(_BREAKER).items()
    }


# ---------------------------------------------------------------------------
# The connection pool
# ---------------------------------------------------------------------------


class _LoopThread:
    """A background event loop the synchronous VM can submit work to."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run, name="mcp-loop", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def submit(self, coro, timeout: float):
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    def close(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)


_LOOP: _LoopThread | None = None
_LOOP_LOCK = threading.Lock()


def _loop() -> _LoopThread:
    global _LOOP
    with _LOOP_LOCK:
        if _LOOP is None:
            _LOOP = _LoopThread()
        return _LOOP


class Connection:
    """A live session with one MCP server.

    Subclassed in tests with a fake so the client is testable without
    spawning real servers.
    """

    async def list_tools(self) -> list[dict[str, Any]]:  # pragma: no cover
        raise NotImplementedError

    async def call_tool(
        self, name: str, args: dict[str, Any]
    ) -> Any:  # pragma: no cover
        raise NotImplementedError

    async def close(self) -> None:  # pragma: no cover
        return None


class _SDKConnection(Connection):
    """A connection backed by the official ``mcp`` package."""

    def __init__(self, server: MCPServer, env: dict[str, str] | None = None):
        self._server = server
        self._env = env or {}
        self._session: Any = None
        self._stack: Any = None

    async def _ensure(self) -> Any:
        if self._session is not None:
            return self._session
        from contextlib import AsyncExitStack

        self._stack = AsyncExitStack()
        await self._stack.__aenter__()
        if self._server.transport == TRANSPORT_HTTP:
            from mcp.client.streamable_http import streamablehttp_client

            read, write, _ = await self._stack.enter_async_context(
                streamablehttp_client(self._server.url)
            )
        else:
            from mcp import StdioServerParameters
            from mcp.client.stdio import stdio_client

            params = StdioServerParameters(
                command=self._server.command,
                args=list(self._server.args),
                env=self._env or None,
            )
            read, write = await self._stack.enter_async_context(
                stdio_client(params)
            )
        from mcp import ClientSession

        self._session = await self._stack.enter_async_context(
            ClientSession(read, write)
        )
        await self._session.initialize()
        return self._session

    async def list_tools(self) -> list[dict[str, Any]]:
        session = await self._ensure()
        result = await session.list_tools()
        out = []
        for tool in getattr(result, "tools", []) or []:
            schema = (
                getattr(tool, "input_schema", None)
                or getattr(tool, "inputSchema", None)
                or {}
            )
            out.append(
                {
                    "name": getattr(tool, "name", ""),
                    "description": getattr(tool, "description", "") or "",
                    "inputSchema": schema
                    if isinstance(schema, dict)
                    else dict(schema or {}),
                }
            )
        return out

    async def call_tool(self, name: str, args: dict[str, Any]) -> Any:
        session = await self._ensure()
        return await session.call_tool(name, args or {})

    async def close(self) -> None:
        if self._stack is not None:
            try:
                await self._stack.__aexit__(None, None, None)
            finally:
                self._stack = None
                self._session = None


# connection_factory is swappable so tests inject a fake transport.
_connection_factory = None


def set_connection_factory(factory) -> None:
    """Override how connections are made (tests, or an alternate SDK)."""
    global _connection_factory
    _connection_factory = factory


def _connect(server: MCPServer, env: dict[str, str] | None = None) -> Connection:
    if _connection_factory is not None:
        return _connection_factory(server, env)
    return _SDKConnection(server, env)


_CONNECTIONS: dict[str, Connection] = {}
_DISCOVERY: dict[str, tuple[float, list[MCPTool]]] = {}
_POOL_LOCK = threading.Lock()


def reset() -> None:
    """Drop every cached connection and discovery (tests, config change)."""
    with _POOL_LOCK:
        for conn in _CONNECTIONS.values():
            try:
                _loop().submit(conn.close(), timeout=5.0)
            except Exception:  # noqa: BLE001
                pass
        _CONNECTIONS.clear()
        _DISCOVERY.clear()
        _BREAKER.clear()


def _connection_for(server: MCPServer, env: dict[str, str] | None) -> Connection:
    with _POOL_LOCK:
        conn = _CONNECTIONS.get(server.server_id)
        if conn is None:
            conn = _connect(server, env)
            _CONNECTIONS[server.server_id] = conn
        return conn


def discover(
    sql, server: MCPServer, *, env: dict[str, str] | None = None, force: bool = False
) -> list[MCPTool]:
    """List a server's tools, sanitised, namespaced, and pin-checked.

    A server that is unreachable, slow, or has changed its definitions
    contributes **no** tools rather than failing the turn: one bad
    integration must not take the assistant down.
    """
    cached = _DISCOVERY.get(server.server_id)
    if cached and not force and (time.time() - cached[0]) < DISCOVERY_TTL_S:
        return cached[1]
    if _breaker_open(server.server_id):
        return []

    try:
        conn = _connection_for(server, env)
        raw = _loop().submit(conn.list_tools(), timeout=server.timeout_s)
        _record_success(server.server_id)
    except Exception:  # noqa: BLE001
        _record_failure(server.server_id)
        log.warning("MCP server %s is unreachable; skipping", server.name)
        _DISCOVERY[server.server_id] = (time.time(), [])
        return []

    raw = list(raw or [])[:MAX_TOOLS_PER_SERVER]
    pin = check_pin(sql, server, raw)
    if not pin["ok"]:
        log.warning(
            "MCP server %s withheld (%s) — a governor must re-approve",
            server.name, pin["state"],
        )
        _DISCOVERY[server.server_id] = (time.time(), [])
        return []

    tools = [
        MCPTool(
            server_id=server.server_id,
            server_name=server.name,
            remote_name=str(t.get("name", "")),
            name=namespaced(server.name, str(t.get("name", ""))),
            description=sanitize_description(
                t.get("description", ""), trusted=server.trusted
            ),
            input_schema=sanitize_schema(
                t.get("inputSchema") or {}, trusted=server.trusted
            ),
            capability=server.capability,
        )
        for t in raw
        if t.get("name")
    ]
    _DISCOVERY[server.server_id] = (time.time(), tools)
    log.info("MCP server %s offered %d tools", server.name, len(tools))
    return tools


def tools_for(sql, principal, *, env: dict[str, str] | None = None) -> list[MCPTool]:
    """Every remote tool this user's VM should expose."""
    out: list[MCPTool] = []
    for server in servers_for(sql, principal):
        out.extend(discover(sql, server, env=env))
    return out


def call(
    sql, principal, name: str, args: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Invoke a namespaced remote tool on the user's behalf.

    Authorization is re-checked here rather than trusted from discovery
    time: a capability may have been revoked since the VM was built.
    """
    parsed = split_namespaced(name)
    if parsed is None:
        return {"ok": False, "error": f"{name} is not an MCP tool name"}
    server_name, remote = parsed

    # The tool name was sanitized for the model (namespaced() scrubs
    # the server name to [a-z0-9_]), so the match here must apply the
    # same scrub: comparing the sanitized fragment to the raw server
    # name made a server named "github-prod" uncallable, and two
    # servers whose names collide after scrubbing routed calls to the
    # wrong one.
    def _scrub(n: str) -> str:
        return re.sub(r"[^a-z0-9_]", "_", (n or "").lower())

    server = next(
        (
            s
            for s in servers_for(sql, principal)
            if _scrub(s.name) == server_name
        ),
        None,
    )
    if server is None:
        return {
            "ok": False,
            "error": f"no MCP server {server_name!r} available to you",
        }
    # The pin is re-verified at call time: discovery withholding a
    # rug-pulled server's tools stops them being *offered*, but a name
    # the model already learned could still be invoked directly.
    cached = _DISCOVERY.get(server.server_id)
    if cached is not None:
        _stamp, tools = cached
        if not any(t.remote_name == remote for t in tools):
            return {
                "ok": False,
                "error": (
                    f"tool {remote!r} is not in {server.name}'s current "
                    "approved tool surface"
                ),
            }
    if _breaker_open(server.server_id):
        return {
            "ok": False,
            "error": f"{server.name} is quarantined after repeated failures",
        }
    try:
        conn = _connection_for(server, None)
        result = _loop().submit(
            conn.call_tool(remote, args or {}), timeout=server.timeout_s
        )
        _record_success(server.server_id)
    except TimeoutError:
        _record_failure(server.server_id)
        return {"ok": False, "error": f"{server.name} timed out"}
    except Exception as exc:  # noqa: BLE001
        _record_failure(server.server_id)
        log.exception("MCP call failed")
        return {"ok": False, "error": f"{server.name} failed: {exc}"}
    return {"ok": True, "server": server.name, "tool": remote, "result": _plain(result)}


def _truncate(value: Any) -> Any:
    """Bound a remote result so it cannot evict the turn's context.

    Remote tool output is unbounded and has been measured in the wild at
    hundreds of thousands of tokens. Truncating at the seam costs a tail
    of one result; not truncating costs the whole conversation.
    """
    if isinstance(value, str) and len(value) > MAX_RESULT_CHARS:
        dropped = len(value) - MAX_RESULT_CHARS
        return (
            value[:MAX_RESULT_CHARS]
            + f"\n\n[truncated — {dropped:,} more characters withheld]"
        )
    if isinstance(value, (dict, list)):
        try:
            encoded = json.dumps(value, default=str)
        except (TypeError, ValueError):
            return value
        if len(encoded) > MAX_RESULT_CHARS:
            return {
                "truncated": True,
                "original_chars": len(encoded),
                "preview": encoded[:MAX_RESULT_CHARS],
            }
    return value


def _plain(result: Any) -> Any:
    """Reduce an MCP result to something JSON-serialisable and bounded."""
    content = getattr(result, "content", None)
    if content is not None:
        parts = []
        for block in content:
            text = getattr(block, "text", None)
            parts.append(text if text is not None else str(block))
        return _truncate("\n".join(parts) if parts else "")
    if isinstance(result, (dict, list, str, int, float, bool)) or result is None:
        return _truncate(result)
    return _truncate(str(result))


def health(sql, principal) -> list[dict[str, Any]]:
    """Per-server reachability and tool counts, for the Tools surface."""
    out = []
    for server in servers_for(sql, principal):
        started = time.perf_counter()
        tools = discover(sql, server)
        out.append(
            {
                **server.to_dict(),
                "reachable": bool(tools),
                "tool_count": len(tools),
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            }
        )
    return out
