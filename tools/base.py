"""Base tool class and registry for Jarvis.

Enhanced with parameter validation, timeout support, and structured results.
"""
import asyncio
import functools
import logging
import time
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


async def async_read_file(path: Path | str, encoding: str = "utf-8", errors: str = "replace") -> str:
    """Read a file asynchronously using run_in_executor."""
    p = Path(path) if not isinstance(path, Path) else path
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: p.read_text(encoding=encoding, errors=errors))


async def async_read_file_bytes(path: Path | str) -> bytes:
    """Read a file as bytes asynchronously."""
    p = Path(path) if not isinstance(path, Path) else path
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, p.read_bytes)


async def async_write_file(path: Path | str, content: str, encoding: str = "utf-8") -> None:
    """Write a file asynchronously using run_in_executor."""
    p = Path(path) if not isinstance(path, Path) else path
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, lambda: p.write_text(content, encoding=encoding))


async def async_mkdir(path: Path | str, parents: bool = True, exist_ok: bool = True) -> None:
    """Create directory asynchronously."""
    p = Path(path) if not isinstance(path, Path) else path
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, lambda: p.mkdir(parents=parents, exist_ok=exist_ok))


@dataclass
class ToolResult:
    """Structured result of tool execution.

    All tools should return ToolResult for consistent error handling,
    timeout detection, and output validation by the agentic harness.
    """
    success: bool = True
    result: dict[str, Any] = field(default_factory=dict)
    output: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "result": self.result,
            "output": self.output,
            "error": self.error,
        }


def validate_params(schema: dict) -> callable:
    """Decorator that validates tool parameters against a JSON schema.

    Usage:
        @validate_params({"properties": {"path": {"type": "string"}}, "required": ["path"]})
        async def execute(self, **params):
            ...
    """
    required = set(schema.get("required", []))
    properties = schema.get("properties", {})

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(self, **params):
            for req in required:
                if req not in params or params[req] is None:
                    return ToolResult(
                        success=False,
                        error=f"Missing required parameter: {req}",
                    )
            for key, value in params.items():
                if key in properties:
                    prop_schema = properties[key]
                    expected_type = prop_schema.get("type")
                    if expected_type == "string" and not isinstance(value, str):
                        return ToolResult(success=False, error=f"Parameter '{key}' must be a string")
                    elif expected_type == "number" and not isinstance(value, (int, float)):
                        return ToolResult(success=False, error=f"Parameter '{key}' must be a number")
                    elif expected_type == "integer" and not isinstance(value, int):
                        return ToolResult(success=False, error=f"Parameter '{key}' must be an integer")
                    elif expected_type == "boolean" and not isinstance(value, bool):
                        return ToolResult(success=False, error=f"Parameter '{key}' must be a boolean")
                    elif expected_type == "array" and not isinstance(value, list):
                        return ToolResult(success=False, error=f"Parameter '{key}' must be an array")
                    elif expected_type == "object" and not isinstance(value, dict):
                        return ToolResult(success=False, error=f"Parameter '{key}' must be an object")
                    max_length = prop_schema.get("maxLength")
                    if max_length and isinstance(value, str) and len(value) > max_length:
                        return ToolResult(success=False, error=f"Parameter '{key}' exceeds maxLength {max_length}")
                    min_value = prop_schema.get("minimum")
                    if min_value is not None and isinstance(value, (int, float)) and value < min_value:
                        return ToolResult(success=False, error=f"Parameter '{key}' below minimum {min_value}")
                    max_value = prop_schema.get("maximum")
                    if max_value is not None and isinstance(value, (int, float)) and value > max_value:
                        return ToolResult(success=False, error=f"Parameter '{key}' above maximum {max_value}")
                    enum_values = prop_schema.get("enum")
                    if enum_values and value not in enum_values:
                        return ToolResult(success=False, error=f"Parameter '{key}' must be one of: {enum_values}")
            return await func(self, **params)
        return wrapper
    return decorator


def tool_timeout(seconds: float = 120.0):
    """Decorator that adds timeout to tool execution.

    Usage:
        @tool_timeout(30.0)
        async def execute(self, **params):
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(self, **params):
            try:
                return await asyncio.wait_for(func(self, **params), timeout=seconds)
            except asyncio.TimeoutError:
                logger.warning(f"Tool {getattr(self, 'name', 'unknown')} timed out after {seconds}s")
                return ToolResult(
                    success=False,
                    error=f"Tool execution timed out after {seconds}s",
                )
        return wrapper
    return decorator


class Tool(ABC):
    """Base class for all Jarvis tools.

    Enhanced with:
    - Parameter validation via validate_params decorator
    - Timeout support via tool_timeout decorator
    - Structured ToolResult return type
    - Execution timing metadata
    """

    name: str
    description: str
    parameters: dict[str, Any]
    requires_permission: bool = False
    timeout: float = 120.0

    @abstractmethod
    async def execute(self, **params) -> str | dict[str, Any] | ToolResult:
        """Execute the tool with given parameters."""
        pass

    def get_schema(self) -> dict[str, Any]:
        """Get the JSON schema for this tool."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    async def safe_execute(self, **params) -> ToolResult:
        """Execute with timeout, error wrapping, and timing.

        This method should be called by the agentic harness and agent loop
        instead of calling execute() directly, to ensure consistent error
        handling and timeout management.
        """
        start = time.time()
        try:
            result = await asyncio.wait_for(self.execute(**params), timeout=self.timeout)
            elapsed = time.time() - start
            if isinstance(result, ToolResult):
                return result
            elif isinstance(result, dict):
                if result.get("error") or result.get("success") is False:
                    return ToolResult(success=False, result=result, error=result.get("error", "Tool returned error"))
                return ToolResult(success=True, result=result)
            elif isinstance(result, str):
                return ToolResult(success=True, output=result)
            else:
                return ToolResult(success=True, result={"data": result})
        except asyncio.TimeoutError:
            elapsed = time.time() - start
            logger.warning(f"Tool {self.name} timed out after {self.timeout}s")
            return ToolResult(success=False, error=f"Timeout after {self.timeout}s")
        except Exception as e:
            elapsed = time.time() - start
            error_msg = str(e)
            logger.error(f"Tool {self.name} error: {error_msg}")
            return ToolResult(success=False, error=error_msg)


class ToolRegistry:
    """Registry for managing available tools."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        """Unregister a tool by name."""
        if name in self._tools:
            del self._tools[name]

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def get_schemas(self) -> list[dict[str, Any]]:
        """Get schemas for all tools."""
        return [tool.get_schema() for tool in self._tools.values()]

    def get_tools(self) -> list[Tool]:
        """Get all registered tools."""
        return list(self._tools.values())

    async def safe_execute(self, name: str, **params) -> ToolResult:
        """Execute a tool by name with safe_execute wrapper."""
        tool = self.get(name)
        if tool is None:
            return ToolResult(success=False, error=f"Tool '{name}' not found")
        return await tool.safe_execute(**params)
