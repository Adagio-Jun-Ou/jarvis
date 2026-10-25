"""Tool type hierarchy for Omnigent (vendored, adapted).

This is an adapted subset of ``omnigent/inner/tools.py`` from
https://github.com/omnigent-ai/omnigent (Apache-2.0).  The
adaptations:

* **Kept** as-is: the full :class:`Tool` dataclass family —
  :class:`FunctionTool`, :class:`MCPTool`, :class:`AgentTool`,
  :class:`SkillTool`, :class:`HandoffTool`, :class:`InheritedTool`,
  :class:`SelfAgentTool`, :class:`CancellableFunctionTool` — and the
  ``_schema_from_callable`` auto-JSON-Schema generator.
* **Trimmed**: removed the upstream CWD-scan / seccomp / bwrap
  runners (``CancellableRunner`` protocol is kept as a typing-only
  Protocol because :class:`CancellableFunctionTool` references it).

The tool ``scopes`` field is how Alex 2.0 wires its
:mod:`agent.guardrails` capability check: every :class:`FunctionTool`
in Alex is created with ``scopes={"alex:tool:<name>"}`` and the
:mod:`agent.constitution` L2 interceptor consumes the request.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeAlias, runtime_checkable

if TYPE_CHECKING:
    from .datamodel import AgentDef, ExecutorSpec

ToolSchema: TypeAlias = dict[str, Any]
ToolArgs: TypeAlias = dict[str, Any]
ToolResult: TypeAlias = dict[str, Any]
DynamicCallable: TypeAlias = Callable[..., object]


@runtime_checkable
class CancellableRun(Protocol):
    def cancel(self, reason: str) -> bool | Awaitable[bool]: ...


@runtime_checkable
class CancellableRunner(Protocol):
    def start(
        self,
        args: ToolArgs,
        on_complete: Callable[[ToolResult], None],
    ) -> CancellableRun | None: ...


_PY_TYPE_TO_JSON: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _schema_from_callable(fn: DynamicCallable) -> ToolSchema:
    sig = inspect.signature(fn)
    properties: ToolSchema = {}
    required: list[str] = []

    for pname, param in sig.parameters.items():
        if pname in ("self", "cls"):
            continue
        annotation = param.annotation
        if annotation is inspect.Parameter.empty:
            json_type = "string"
        else:
            json_type = _PY_TYPE_TO_JSON.get(annotation, "string")
        properties[pname] = {"type": json_type}
        if param.default is inspect.Parameter.empty:
            required.append(pname)

    schema: ToolSchema = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required
    return schema


@dataclass
class Tool:
    name: str | None = None
    description: str | None = None
    input_schema: ToolSchema | None = None
    output_schema: ToolSchema | None = None
    scopes: set[str] | None = None
    cancellable: bool = False

    def tool_schema(self) -> ToolSchema:
        schema: ToolSchema = {
            "name": self.name if self.name is not None else "",
            "description": self.description if self.description is not None else "",
        }
        if self.input_schema:
            schema["parameters"] = self.input_schema
        elif isinstance(self, FunctionTool) and self.callable:
            schema["parameters"] = _schema_from_callable(self.callable)
        elif isinstance(self, AgentTool):
            limit_text = (
                f"Max concurrent sessions: {self.max_sessions}"
                if self.max_sessions is not None
                else None
            )
            if limit_text:
                schema["description"] = (
                    f"{schema['description']}\n\n{limit_text}"
                    if schema["description"]
                    else limit_text
                )
            schema["parameters"] = {
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "The input text or request to send to this agent.",
                    },
                },
                "required": ["input"],
            }
        return schema


@dataclass
class FunctionTool(Tool):
    callable: DynamicCallable | None = None
    catalog_path: str | None = None
    runtime: Literal["server", "client"] = "server"
    warehouse_id: str | None = None


@dataclass
class CancellableFunctionTool(Tool):
    runner: CancellableRunner | DynamicCallable | None = None


@dataclass
class MCPTool(Tool):
    url: str | None = None
    command: str | None = None
    args: list[str] | None = None
    env: dict[str, str] | None = None
    tools: list[str] | None = None
    tool_name: str | None = None
    profile: str | None = None
    databricks_server: str | None = None
    headers: dict[str, str] | None = None


@dataclass
class AgentTool(Tool):
    prompt: str | None = None
    tools: dict[str, Tool] = field(default_factory=dict)
    executor: ExecutorSpec | None = None
    os_env: object | None = None
    pass_history: bool = False
    pass_histories: list[str] | None = None
    max_sessions: int | None = None


@dataclass
class SelfAgentTool(Tool):
    pass


@dataclass
class InheritedTool(Tool):
    pass


@dataclass
class SkillTool(Tool):
    path: str | None = None


@dataclass
class HandoffTool(Tool):
    target_agent: str | AgentDef | None = None
    pass_history: bool = True
    history_filter: DynamicCallable | None = None
    input_type: type | None = None
    on_handoff: DynamicCallable | None = None
    bidirectional: bool = True
