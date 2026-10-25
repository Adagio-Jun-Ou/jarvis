"""Tests for alex_sdk.mcp — the real MCP server bridge.

Uses an httpx.MockTransport-backed Alex client (no network) and the
optional ``mcp`` package (skipped when absent).
"""

from __future__ import annotations

import json

import httpx
import pytest
from alex_sdk import Alex
from alex_sdk.mcp import _proxy_for, build_mcp_server, to_mcp_tools

import tests.engine.sdk_path  # noqa: F401

MANIFEST = {
    "tools": [
        {
            "name": "create_task",
            "description": "Create a task.",
            "inputSchema": {
                "type": "object",
                "properties": {"title": {"type": "string"}, "due": {"type": "string"}},
                "required": ["title"],
            },
        },
        {
            "name": "read_inbox",
            "description": "Read the inbox.",
            "inputSchema": {"type": "object", "properties": {}},
        },
    ]
}


def _client() -> Alex:
    calls: list[tuple[str, str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}") if request.content else {}
        calls.append((request.method, request.url.path, body))
        if request.url.path == "/api/tools":
            return httpx.Response(200, json=MANIFEST)
        if request.url.path.startswith("/api/tools/"):
            return httpx.Response(
                200, json={"result": {"echo": body.get("args", {})}}
            )
        return httpx.Response(404, json={})

    a = Alex(base_url="http://testserver", token="t")
    a._http = httpx.Client(
        base_url="http://testserver", transport=httpx.MockTransport(handler)
    )
    a._calls = calls  # type: ignore[attr-defined]
    return a


def test_proxy_signature_from_schema():
    import inspect

    a = _client()
    proxy = _proxy_for(a, to_mcp_tools(MANIFEST)[0])
    sig = inspect.signature(proxy)
    assert set(sig.parameters) == {"title", "due"}
    assert sig.parameters["title"].default is inspect.Parameter.empty
    assert sig.parameters["due"].default is None
    assert proxy.__name__ == "create_task"


def test_proxy_forwards_to_call_tool():
    a = _client()
    proxy = _proxy_for(a, to_mcp_tools(MANIFEST)[0])
    res = proxy(title="ship it")
    assert res == {"echo": {"title": "ship it"}}
    assert any(p == "/api/tools/create_task" for _, p, _ in a._calls)


mcp = pytest.importorskip("mcp")


def test_build_mcp_server_registers_all_tools():
    import anyio

    server = build_mcp_server(_client(), name="alex-test")
    tools = anyio.run(server.list_tools)
    names = {t.name for t in tools}
    assert names == {"create_task", "read_inbox"}
    create = next(t for t in tools if t.name == "create_task")
    assert "title" in (create.input_schema.get("properties") or {})


def test_mcp_call_tool_roundtrip():
    import anyio

    server = build_mcp_server(_client(), name="alex-test")

    async def run():
        return await server.call_tool("create_task", {"title": "hello"})

    result = anyio.run(run)
    # FastMCP-style servers return content blocks (and possibly a
    # structured result); assert the echo made it through somewhere.
    assert "hello" in str(result)
