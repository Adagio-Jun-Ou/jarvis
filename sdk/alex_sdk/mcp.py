"""
MCP interop — expose Alex's tool manifest to Model Context Protocol hosts.

The app's ``GET /api/tools`` route already speaks MCP's tool shape
(``name`` / ``description`` / ``inputSchema``, ACL-filtered per user).
This module converts that manifest into MCP tool-definition dicts
(:func:`to_mcp_tools` — pure, no I/O) and offers :func:`serve_mcp`,
which requires the optional ``mcp`` package. The SDK keeps ``httpx``
as its only hard dependency.

Example::

    from alex_sdk import Alex
    from alex_sdk.mcp import to_mcp_tools

    with Alex.for_dev("http://127.0.0.1:8765", "alice@example.com") as alex:
        defs = to_mcp_tools([t.__dict__ for t in alex.tools()])
"""

from __future__ import annotations

from typing import Any

__all__ = ["to_mcp_tools", "build_mcp_server", "serve_mcp", "main"]


def to_mcp_tools(manifest: Any) -> list[dict]:
    """Convert an ``/api/tools`` manifest into MCP tool definitions.

    Pure function: accepts either the raw response dict
    (``{"tools": [...]}``) or a bare list of tool entries. Each entry
    may spell its schema ``inputSchema`` or ``input_schema``; entries
    without a name are dropped. Returns a list of
    ``{"name", "description", "inputSchema"}`` dicts, with a default
    empty object schema when none is provided.
    """
    if isinstance(manifest, dict):
        entries = manifest.get("tools", [])
    else:
        entries = manifest or []
    out: list[dict] = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        name = str(e.get("name") or "").strip()
        if not name:
            continue
        schema = e.get("inputSchema") or e.get("input_schema")
        if not isinstance(schema, dict) or not schema:
            schema = {"type": "object", "properties": {}}
        out.append(
            {
                "name": name,
                "description": str(e.get("description") or ""),
                "inputSchema": schema,
            }
        )
    return out


_SCHEMA_TYPES: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _proxy_for(alex: Any, tool: dict) -> Any:
    """Build a callable proxy for one MCP tool definition.

    The proxy forwards to ``alex.call_tool(name, **kwargs)`` (the
    governed ``POST /api/tools/{name}`` seam — ACL-checked and
    constitution-gated server-side). Its ``__signature__`` is derived
    from the tool's ``inputSchema`` so an MCP host sees real parameter
    names, types, and required-ness instead of ``**kwargs``.
    """
    import inspect

    tool_name = tool["name"]
    schema = tool.get("inputSchema") or {}
    props: dict = schema.get("properties") or {}
    required = set(schema.get("required") or [])

    def proxy(**kwargs: Any) -> Any:
        return alex.call_tool(tool_name, **kwargs)

    params = []
    for pname, entry in props.items():
        ann = _SCHEMA_TYPES.get((entry or {}).get("type", ""), Any)
        default = inspect.Parameter.empty if pname in required else None
        params.append(
            inspect.Parameter(
                pname,
                inspect.Parameter.KEYWORD_ONLY,
                default=default,
                annotation=ann,
            )
        )
    proxy.__name__ = tool_name
    proxy.__doc__ = tool.get("description") or tool_name
    proxy.__signature__ = inspect.Signature(params)  # type: ignore[attr-defined]
    return proxy


def build_mcp_server(alex: Any, *, name: str = "alex") -> Any:
    """Build (without running) an MCP server exposing Alex's tools.

    Requires the optional ``mcp`` package (>= 2.0). Every tool in the
    user's ACL-filtered manifest is registered; each call is forwarded
    through the governed ``call_tool`` seam, so guardrails and the
    constitution apply exactly as they do for the agent's own tool use.
    """
    try:
        from mcp.server.mcpserver import MCPServer
    except ImportError as e:  # pragma: no cover - exercised without mcp
        raise NotImplementedError(
            "build_mcp_server requires the optional 'mcp' package "
            "(pip install 'mcp>=2')."
        ) from e

    server = MCPServer(
        name,
        instructions=(
            "Governed tools of the Alex enterprise agent. Every call is "
            "ACL-checked and constitution-gated server-side."
        ),
    )
    manifest = to_mcp_tools([t.__dict__ for t in alex.tools()])
    for tool in manifest:
        server.add_tool(
            _proxy_for(alex, tool),
            name=tool["name"],
            description=tool["description"],
        )
    return server


def serve_mcp(alex: Any, *, name: str = "alex", transport: str = "stdio") -> None:
    """Run an MCP server over ``stdio`` (or ``sse``/``streamable-http``)
    exposing the signed-in user's Alex tools to any MCP host."""
    build_mcp_server(alex, name=name).run(transport=transport)


def main() -> None:
    """``python -m alex_sdk.mcp`` — stdio MCP server for an Alex app.

    Configuration via env vars: ``ALEX_URL`` (default
    ``http://127.0.0.1:8765``) and ``ALEX_EMAIL`` (dev-token identity,
    default ``demo@alex.dev``).
    """
    import os

    from alex_sdk import Alex

    url = os.environ.get("ALEX_URL", "http://127.0.0.1:8765")
    email = os.environ.get("ALEX_EMAIL", "demo@alex.dev")
    with Alex.for_dev(url, email) as alex:
        serve_mcp(alex)


if __name__ == "__main__":  # pragma: no cover
    main()
