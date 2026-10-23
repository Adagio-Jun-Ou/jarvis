"""Tests for the MCP stdio transport.

The protocol core (handlers, tool registry) existed but had no
transport, so the advertised MCP compatibility was unusable. These
tests cover the newline-delimited JSON-RPC stdio loop: request/
response round-trips, notification silence (per spec), parse and
invalid-request errors, and the write-capable tool gate.
"""
import asyncio
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.mcp_server import MCPServer, get_mcp_server  # noqa: E402


def _run(lines: list[str]):
    server = MCPServer()

    @server.tool("echo", "Echo text back")
    async def echo(text: str) -> dict:
        return {"echo": text}

    stdin = io.StringIO("\n".join(lines) + "\n")
    stdout = io.StringIO()
    processed = asyncio.run(server.serve_stdio(stdin=stdin, stdout=stdout))
    responses = [
        json.loads(l) for l in stdout.getvalue().splitlines() if l.strip()
    ]
    return processed, responses


class TestMCPStdioTransport:
    def test_request_response_roundtrip(self):
        processed, responses = _run([
            json.dumps({"jsonrpc": "2.0", "id": 1,
                        "method": "initialize", "params": {}}),
            json.dumps({"jsonrpc": "2.0", "id": 2,
                        "method": "tools/list", "params": {}}),
        ])
        assert processed == 2
        assert len(responses) == 2
        by_id = {r["id"]: r for r in responses}
        assert by_id[1]["result"]["serverInfo"]["name"] == "jarvis"
        tool_names = [t["name"] for t in by_id[2]["result"]["tools"]]
        assert "echo" in tool_names

    def test_tools_call_executes(self):
        processed, responses = _run([
            json.dumps({"jsonrpc": "2.0", "id": 7, "method": "tools/call",
                        "params": {"name": "echo",
                                   "arguments": {"text": "hello"}}}),
        ])
        assert processed == 1
        assert "hello" in json.dumps(responses[0]["result"])

    def test_notification_gets_no_response(self):
        processed, responses = _run([
            json.dumps({"jsonrpc": "2.0", "id": 1,
                        "method": "initialize", "params": {}}),
            json.dumps({"jsonrpc": "2.0",
                        "method": "notifications/initialized"}),
            json.dumps({"jsonrpc": "2.0", "id": 2,
                        "method": "tools/list", "params": {}}),
        ])
        assert processed == 3  # notification counted as processed
        assert len(responses) == 2  # but never answered
        assert [r["id"] for r in responses] == [1, 2]

    def test_parse_error_and_invalid_request(self):
        processed, responses = _run([
            "this is not json",
            json.dumps([1, 2, 3]),
        ])
        # Neither is a valid JSON-RPC message: not counted as processed,
        # but both produce error frames.
        assert processed == 0
        kinds = [r.get("error", {}).get("code") for r in responses]
        assert -32700 in kinds, "parse error expected"
        assert -32600 in kinds, "invalid request expected"

    def test_unknown_method_returns_error(self):
        processed, responses = _run([
            json.dumps({"jsonrpc": "2.0", "id": 9,
                        "method": "does/not/exist", "params": {}}),
        ])
        assert responses[0]["error"]["code"] == -32601

    def test_blank_lines_skipped_and_eof_ends(self):
        processed, responses = _run(["", "   "])
        assert processed == 0
        assert responses == []


class TestMCPWriteGate:
    def test_memory_store_gated_by_default(self, monkeypatch):
        monkeypatch.delenv("JARVIS_MCP_ALLOW_WRITES", raising=False)
        server = MCPServer()
        server.register_jarvis_tools(agent=None)
        assert "jarvis_memory_store" not in server.tools
        assert "jarvis_chat" in server.tools

    def test_memory_store_enabled_with_env(self, monkeypatch):
        monkeypatch.setenv("JARVIS_MCP_ALLOW_WRITES", "1")
        server = MCPServer()
        server.register_jarvis_tools(agent=None)
        assert "jarvis_memory_store" in server.tools
