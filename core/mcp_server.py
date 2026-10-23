"""Model Context Protocol (MCP) Server for Jarvis.

Implements the MCP protocol for standardized tool/context interface with LLMs.
Compatible with Anthropic Claude Code, OpenAI, and other MCP-supporting platforms.

Based on the Model Context Protocol specification from Anthropic.
"""
import asyncio
import json
import logging
import os
from core.helpers import safe_error_message
import sys
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class MCPMessageType(Enum):
    """MCP message types."""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    ERROR = "error"


@dataclass
class MCPTool:
    """Definition of an MCP tool."""
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable | None = None

    def to_mcp(self) -> dict:
        """Convert to MCP format."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


@dataclass
class MCPResource:
    """Definition of an MCP resource."""
    uri: str
    name: str
    description: str
    mime_type: str = "application/json"
    handler: Callable | None = None

    def to_mcp(self) -> dict:
        """Convert to MCP format."""
        return {
            "uri": self.uri,
            "name": self.name,
            "description": self.description,
            "mimeType": self.mime_type,
        }


@dataclass
class MCPPrompt:
    """Definition of an MCP prompt template."""
    name: str
    description: str
    arguments: list[dict[str, Any]] = field(default_factory=list)
    handler: Callable | None = None

    def to_mcp(self) -> dict:
        """Convert to MCP format."""
        return {
            "name": self.name,
            "description": self.description,
            "arguments": self.arguments,
        }


@dataclass
class MCPServerInfo:
    """Server information for MCP handshake."""
    name: str = "jarvis"
    version: str = "3.0.0"
    protocol_version: str = "2024-11-05"

    def to_mcp(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "protocolVersion": self.protocol_version,
        }


@dataclass
class MCPCapabilities:
    """Server capabilities for MCP handshake."""
    tools: bool = True
    resources: bool = True
    prompts: bool = True
    logging: bool = True
    streaming: bool = True

    def to_mcp(self) -> dict:
        return {
            "tools": {"listChanged": True} if self.tools else None,
            "resources": {"subscribe": True, "listChanged": True} if self.resources else None,
            "prompts": {"listChanged": True} if self.prompts else None,
            "logging": {} if self.logging else None,
        }


class MCPServer:
    """MCP Server implementation for Jarvis.
    
    Provides a standardized interface for:
    - Tool discovery and execution
    - Resource access
    - Prompt templates
    - Logging
    
    Example usage:
        server = MCPServer()
        
        @server.tool("search_web")
        async def search_web(query: str) -> dict:
            return {"results": [...]}
        
        # Start server
        await server.start()
    """

    def __init__(
        self,
        name: str = "jarvis",
        version: str = "3.0.0",
    ):
        self.info = MCPServerInfo(name=name, version=version)
        self.capabilities = MCPCapabilities()

        self._lock = threading.RLock()
        self.tools: dict[str, MCPTool] = {}
        self.resources: dict[str, MCPResource] = {}
        self.prompts: dict[str, MCPPrompt] = {}

        self._running = False
        self._request_handlers: dict[str, Callable] = {}
        self._setup_handlers()

    def _setup_handlers(self) -> None:
        """Setup built-in request handlers."""
        self._request_handlers = {
            "initialize": self._handle_initialize,
            "tools/list": self._handle_tools_list,
            "tools/call": self._handle_tools_call,
            "resources/list": self._handle_resources_list,
            "resources/read": self._handle_resources_read,
            "prompts/list": self._handle_prompts_list,
            "prompts/get": self._handle_prompts_get,
            "logging/setLevel": self._handle_logging_set_level,
        }

    def tool(
        self,
        name: str | None = None,
        description: str = "",
        input_schema: dict | None = None,
    ) -> Callable:
        """Decorator to register a tool."""
        def decorator(func: Callable) -> Callable:
            tool_name = name or func.__name__
            tool_desc = description or func.__doc__ or f"Execute {tool_name}"
            tool_schema = input_schema or self._infer_schema(func)

            with self._lock:
                self.tools[tool_name] = MCPTool(
                    name=tool_name,
                    description=tool_desc,
                    input_schema=tool_schema,
                    handler=func,
                )
            return func
        return decorator

    def resource(
        self,
        uri: str,
        name: str | None = None,
        description: str = "",
        mime_type: str = "application/json",
    ) -> Callable:
        """Decorator to register a resource."""
        def decorator(func: Callable) -> Callable:
            resource_name = name or func.__name__
            resource_desc = description or func.__doc__ or f"Access {resource_name}"

            with self._lock:
                self.resources[uri] = MCPResource(
                    uri=uri,
                    name=resource_name,
                    description=resource_desc,
                    mime_type=mime_type,
                    handler=func,
                )
            return func
        return decorator

    def prompt(
        self,
        name: str | None = None,
        description: str = "",
        arguments: list[dict] | None = None,
    ) -> Callable:
        """Decorator to register a prompt template."""
        def decorator(func: Callable) -> Callable:
            prompt_name = name or func.__name__
            prompt_desc = description or func.__doc__ or f"Generate {prompt_name}"
            prompt_args = arguments or []

            with self._lock:
                self.prompts[prompt_name] = MCPPrompt(
                    name=prompt_name,
                    description=prompt_desc,
                    arguments=prompt_args,
                    handler=func,
                )
            return func
        return decorator

    def _infer_schema(self, func: Callable) -> dict:
        """Infer JSON schema from function signature."""
        import inspect

        schema = {
            "type": "object",
            "properties": {},
            "required": [],
        }

        sig = inspect.signature(func)
        for param_name, param in sig.parameters.items():
            if param_name in ('self', 'cls'):
                continue

            param_schema = {"type": "string"}  # Default to string

            if param.annotation != inspect.Parameter.empty:
                if param.annotation is int:
                    param_schema = {"type": "integer"}
                elif param.annotation is float:
                    param_schema = {"type": "number"}
                elif param.annotation is bool:
                    param_schema = {"type": "boolean"}
                elif param.annotation is list:
                    param_schema = {"type": "array"}
                elif param.annotation is dict:
                    param_schema = {"type": "object"}

            schema["properties"][param_name] = param_schema

            if param.default == inspect.Parameter.empty:
                schema["required"].append(param_name)

        return schema

    async def _handle_initialize(self, params: dict) -> dict:
        """Handle initialize request."""
        return {
            "protocolVersion": self.info.protocol_version,
            "capabilities": self.capabilities.to_mcp(),
            "serverInfo": self.info.to_mcp(),
        }

    async def _handle_tools_list(self, params: dict) -> dict:
        """Handle tools/list request."""
        with self._lock:
            tools = [tool.to_mcp() for tool in self.tools.values()]
        return {
            "tools": tools,
        }

    async def _handle_tools_call(self, params: dict) -> dict:
        """Handle tools/call request."""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        with self._lock:
            if tool_name not in self.tools:
                return {
                    "isError": True,
                    "content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}],
                }

            tool = self.tools[tool_name]

        try:
            if tool.handler is None:
                return {
                    "isError": True,
                    "content": [{"type": "text", "text": f"Tool {tool_name} has no handler"}],
                }

            if asyncio.iscoroutinefunction(tool.handler):
                result = await tool.handler(**arguments)
            else:
                result = tool.handler(**arguments)

            if isinstance(result, dict) and "content" in result:
                return result

            if isinstance(result, str):
                return {
                    "isError": False,
                    "content": [{"type": "text", "text": result}],
                }

            return {
                "isError": False,
                "content": [{"type": "text", "text": json.dumps(result, indent=2)}],
            }

        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}")
            return {
                "isError": True,
                "content": [{"type": "text", "text": f"Error: {safe_error_message(e)}"}],
            }

    async def execute_tool(self, name: str, arguments: dict | None = None) -> Any:
        """Execute a tool directly.

        Args:
            name: Tool name to execute
            arguments: Tool arguments

        Returns:
            Tool result

        Raises:
            ValueError: If tool not found
        """
        arguments = arguments or {}

        with self._lock:
            if name not in self.tools:
                raise ValueError(f"Unknown tool: {name}")

            tool = self.tools[name]

        if asyncio.iscoroutinefunction(tool.handler):
            result = await tool.handler(**arguments)
        else:
            result = tool.handler(**arguments)

        return result

    async def _handle_resources_list(self, params: dict) -> dict:
        """Handle resources/list request."""
        with self._lock:
            resources = [res.to_mcp() for res in self.resources.values()]
        return {
            "resources": resources,
        }

    async def _handle_resources_read(self, params: dict) -> dict:
        """Handle resources/read request."""
        uri = params.get("uri")

        with self._lock:
            if uri not in self.resources:
                return {
                    "contents": [],
                }

            resource = self.resources[uri]

        try:
            if resource.handler is None:
                return {
                    "contents": [{
                        "uri": uri,
                        "mimeType": resource.mime_type,
                        "text": "Resource has no handler",
                    }],
                }

            if asyncio.iscoroutinefunction(resource.handler):
                result = await resource.handler()
            else:
                result = resource.handler()

            if isinstance(result, str):
                text = result
            else:
                text = json.dumps(result, indent=2)

            return {
                "contents": [{
                    "uri": uri,
                    "mimeType": resource.mime_type,
                    "text": text,
                }],
            }

        except Exception as e:
            logger.error(f"Resource {uri} failed: {e}")
            return {
                "contents": [{
                    "uri": uri,
                    "mimeType": "text/plain",
                    "text": f"Error: {safe_error_message(e)}",
                }],
            }

    async def _handle_prompts_list(self, params: dict) -> dict:
        """Handle prompts/list request."""
        with self._lock:
            prompts = [prompt.to_mcp() for prompt in self.prompts.values()]
        return {
            "prompts": prompts,
        }

    async def _handle_prompts_get(self, params: dict) -> dict:
        """Handle prompts/get request."""
        prompt_name = params.get("name")
        arguments = params.get("arguments", {})

        with self._lock:
            if prompt_name not in self.prompts:
                return {
                    "messages": [{"role": "user", "content": {"type": "text", "text": f"Unknown prompt: {prompt_name}"} }],
                }

            prompt = self.prompts[prompt_name]

        try:
            if prompt.handler is None:
                return {
                    "messages": [{"role": "user", "content": {"type": "text", "text": prompt.description}}],
                }

            if asyncio.iscoroutinefunction(prompt.handler):
                result = await prompt.handler(**arguments)
            else:
                result = prompt.handler(**arguments)

            if isinstance(result, str):
                return {
                    "messages": [{"role": "user", "content": {"type": "text", "text": result}}],
                }

            if isinstance(result, list):
                return {"messages": result}

            return {
                "messages": [{"role": "user", "content": {"type": "text", "text": json.dumps(result)}}],
            }

        except Exception as e:
            logger.error(f"Prompt {prompt_name} failed: {e}")
            return {
                "messages": [{"role": "user", "content": {"type": "text", "text": f"Error: {safe_error_message(e)}"}}],
            }

    async def _handle_logging_set_level(self, params: dict) -> dict:
        """Handle logging/setLevel request."""
        level = params.get("level", "info")
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        return {}

    async def handle_request(self, request: dict) -> dict:
        """Handle an incoming MCP request."""
        method = request.get("method", "")
        params = request.get("params", {})
        request_id = request.get("id")

        with self._lock:
            if method not in self._request_handlers:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {method}",
                    },
                }

            handler = self._request_handlers[method]

        try:
            result = await handler(params)
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": result,
            }
        except Exception as e:
            logger.error(f"Request {method} failed: {e}")
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32603,
                    "message": safe_error_message(e, "Internal server error"),
                },
            }

    # ------------------------------------------------------------------ #
    # stdio transport (newline-delimited JSON-RPC 2.0, MCP spec)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _write_message(out_stream, message: dict) -> None:
        try:
            out_stream.write(json.dumps(message, default=str) + "\n")
            out_stream.flush()
        except (BrokenPipeError, ValueError, OSError) as e:
            # Client closed the pipe — the read loop will hit EOF next.
            logger.debug(f"MCP stdout write failed: {e}")

    async def serve_stdio(self, stdin=None, stdout=None) -> int:
        """Serve newline-delimited JSON-RPC over stdio until EOF.

        ``stdout`` must carry protocol frames ONLY — all logging and
        prints must already be redirected to stderr (run_stdio_server
        handles that). Notifications (no "id") never get a response,
        per the MCP spec. Returns the number of messages processed.
        """
        in_stream = stdin if stdin is not None else sys.stdin
        out_stream = stdout if stdout is not None else sys.__stdout__
        processed = 0
        while True:
            try:
                line = await asyncio.to_thread(in_stream.readline)
            except Exception as e:
                logger.debug(f"MCP stdin read failed: {e}")
                break
            if not line:
                break  # EOF — client closed the connection
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                self._write_message(out_stream, {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": "Parse error"},
                })
                continue
            if not isinstance(request, dict):
                self._write_message(out_stream, {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32600, "message": "Invalid Request"},
                })
                continue

            is_notification = "id" not in request
            response = await self.handle_request(request)
            if not is_notification:
                self._write_message(out_stream, response)
            processed += 1
        logger.info(f"MCP stdio session ended ({processed} messages)")
        return processed

    def register_jarvis_tools(self, agent) -> None:
        """Register all Jarvis tools as MCP tools.

        Write-capable tools are gated behind JARVIS_MCP_ALLOW_WRITES=1:
        an external MCP client (Claude Code, etc.) starts read-only.
        """
        from core.vector_memory import get_vector_memory

        @self.tool("jarvis_chat", "Chat with Jarvis AI assistant")
        async def chat(message: str, context: dict | None = None) -> dict:
            result = await agent.chat(message)
            return {"response": result}

        if os.environ.get("JARVIS_MCP_ALLOW_WRITES", "") == "1":
            @self.tool("jarvis_memory_store", "Store information in vector memory")
            async def memory_store(text: str, metadata: dict | None = None) -> dict:
                vm = get_vector_memory()
                entry_id = vm.store(text, metadata)
                return {"entry_id": entry_id}

        @self.tool("jarvis_memory_search", "Search vector memory for similar content")
        async def memory_search(query: str, k: int = 5) -> dict:
            vm = get_vector_memory()
            results = vm.search(query, k=k)
            return {
                "results": [
                    {"id": rid, "score": score, "metadata": meta}
                    for rid, score, meta in results
                ]
            }

        @self.tool("jarvis_self_evolve", "Get evolution status and trigger evolution planning")
        async def self_evolve(domain: str = "general") -> dict:
            from core.evolution_orchestrator import get_evolution_orchestrator
            orchestrator = get_evolution_orchestrator()
            return {
                "status": "available",
                "pathway": orchestrator.get_evolutionary_pathway(),
                "pending": orchestrator.get_pending_permissions(),
            }

        @self.tool("jarvis_daydream", "Trigger daydream exploration")
        async def daydream(topic: str | None = None, duration: int = 30) -> dict:
            from core.erdos_engine import get_erdos_engine
            erdos = get_erdos_engine()
            problems = list(erdos.problems.values())[:3]
            return {
                "problems": [
                    {"id": p.id, "title": p.title, "difficulty": p.difficulty.value}
                    for p in problems
                ]
            }

        @self.tool("jarvis_get_status", "Get Jarvis system status")
        async def get_status() -> dict:
            from core.observability import get_observability
            from core.resource_monitor import get_resource_monitor

            obs = get_observability()
            res = get_resource_monitor()

            return {
                "dashboard": obs.get_dashboard_data(),
                "resources": res.get_statistics(),
                "tools_registered": len(agent.tools),
            }

    def register_jarvis_resources(self, agent) -> None:
        """Register Jarvis resources."""

        @self.resource("jarvis://tools", "Jarvis Tools", "List of all registered tools")
        async def get_tools_resource() -> dict:
            return {
                "tools": [
                    {"name": name, "description": getattr(t, "description", "")}
                    for name, t in agent.tools.items()
                ]
            }

        @self.resource("jarvis://memory", "Jarvis Memory", "Current conversation memory")
        async def get_memory_resource() -> dict:
            return {
                "history": agent.memory.get_history(),
                "context": agent.memory.context,
            }

        @self.resource("jarvis://status", "Jarvis Status", "System status and metrics")
        async def get_status_resource() -> dict:
            from core.observability import get_observability
            obs = get_observability()
            return obs.get_dashboard_data()

        @self.resource("jarvis://evolution", "Evolution Status", "Gene evolution status")
        async def get_evolution_resource() -> dict:
            from core.evolution_orchestrator import get_evolution_orchestrator
            orch = get_evolution_orchestrator()
            return orch.get_status()

    def register_jarvis_prompts(self, agent) -> None:
        """Register Jarvis prompt templates."""

        @self.prompt(
            "jarvis_assistant",
            "Start a conversation with Jarvis",
            arguments=[{"name": "task", "description": "What you want Jarvis to help with", "required": True}]
        )
        async def jarvis_assistant_prompt(task: str) -> str:
            return f"""You are working with Jarvis, an AI assistant with self-evolution capabilities.

Your task: {task}

Jarvis has access to:
- Vector memory for semantic search
- Self-evolution system for improvement
- Daydream engine for creative exploration
- Multiple tools for web, code, data, and system operations

How can Jarvis help you with this task?"""

        @self.prompt(
            "jarvis_code_review",
            "Request a code review from Jarvis",
            arguments=[{"name": "code", "description": "Code to review", "required": True}]
        )
        async def jarvis_code_review_prompt(code: str) -> str:
            return f"""Review the following code for:
1. Bugs and potential issues
2. Performance improvements
3. Security vulnerabilities
4. Code style and best practices

Code to review:
```
{code}
```

Provide specific, actionable feedback."""

        @self.prompt(
            "jarvis_evolve",
            "Guide Jarvis self-evolution",
            arguments=[
                {"name": "domain", "description": "Evolution domain (general, creativity, analysis)", "required": False},
                {"name": "goal", "description": "Evolution goal", "required": False},
            ]
        )
        async def jarvis_evolve_prompt(domain: str = "general", goal: str = "") -> str:
            goal_text = f"\nGoal: {goal}" if goal else ""
            return f"""Guide Jarvis's self-evolution in the {domain} domain.{goal_text}

Consider:
1. What behaviors should be reinforced?
2. What behaviors should be discouraged?
3. What new capabilities should be considered?
4. What parameters might need adjustment?

Provide feedback for Jarvis's evolution system."""


_server: MCPServer | None = None
_server_lock = threading.Lock()


def get_mcp_server(agent: Any = None) -> MCPServer:
    """Get global MCP server instance (thread-safe)."""
    global _server
    if _server is None:
        with _server_lock:
            if _server is None:
                _server = MCPServer()
                if agent is not None:
                    _server.register_jarvis_tools(agent)
                    _server.register_jarvis_resources(agent)
                    _server.register_jarvis_prompts(agent)
    return _server


def run_stdio_server(agent: Any = None) -> int:
    """Run the MCP stdio server (blocking) — the process entry point.

    Protocol purity: the original stdout is reserved for JSON-RPC
    frames and sys.stdout is redirected to stderr FIRST, so every
    banner, print and log line during (or after) startup lands on
    stderr instead of corrupting the protocol stream. Write-capable
    tools stay unregistered unless JARVIS_MCP_ALLOW_WRITES=1.
    """
    protocol_stdout = sys.__stdout__
    sys.stdout = sys.stderr

    server = get_mcp_server()
    if agent is not None and not server.tools:
        try:
            server.register_jarvis_tools(agent)
            server.register_jarvis_resources(agent)
            server.register_jarvis_prompts(agent)
        except Exception as e:
            logger.error(f"MCP tool registration failed: {e}")

    writes = os.environ.get("JARVIS_MCP_ALLOW_WRITES", "") == "1"
    logger.info(
        f"MCP stdio server ready ({len(server.tools)} tools, "
        f"writes {'enabled' if writes else 'disabled'})"
    )
    try:
        return asyncio.run(server.serve_stdio(stdout=protocol_stdout))
    except KeyboardInterrupt:
        logger.info("MCP stdio server interrupted")
        return 0


if __name__ == "__main__":  # pragma: no cover - process entry
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    run_stdio_server()
