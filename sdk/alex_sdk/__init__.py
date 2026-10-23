"""
Alex SDK — the governed Python SDK for the AI-employee platform.

The SDK is the client-side counterpart to the agent endpoint. It
exposes:

  * `Alex` and `Session` — streaming chat clients, plus the product's
    learning surface (`Alex.tools`, `Alex.teach`, `Alex.feedback`,
    `Session.correct`).
  * `AgentEvent` / `EventType` — typed event objects matching the SSE
    wire format (text, reasoning, tool, artifact, law_block, memory,
    confirmation_required, error, done).
  * `ToolInfo` + `alex_sdk.mcp.to_mcp_tools` — the MCP-style tool
    manifest and a pure converter to MCP tool definitions.
  * `Agent`, `tool`, `capability`, `skill` — decorators for building
    custom agents that the bundle can deploy to the same endpoint.
  * `workflow` / `Workflow` — typed workflow builder for two-phase
    confirms.
  * `Memory` — the per-user vector + graph memory API.
  * `ReviewPolicy` — the strict-async-judge preset.

The SDK depends on `httpx` only. Everything else (MLflow, agent
deploy, Databricks SDK) is optional.
"""

from __future__ import annotations

from typing import Any

__version__ = "2.0.0"

# Eager imports. The SDK is small enough that we can pay the import
# cost upfront; the few functions that touch the network are
# documented as such.
from .cli import main as _main
from .client import AgentEvent, Alex, EventType, Session, ToolInfo
from .core import Agent, capability, skill, tool
from .memory import Memory
from .review import ReviewPolicy
from .workflow import Workflow, workflow

__all__ = [
    "Agent",
    "tool",
    "capability",
    "skill",
    "workflow",
    "Workflow",
    "Memory",
    "ReviewPolicy",
    "Alex",
    "Session",
    "AgentEvent",
    "EventType",
    "ToolInfo",
    "main",
]


def main() -> int:
    """Entry point for the `alex` console script."""
    return _main()


def __getattr__(name: str) -> Any:
    # Keep `from alex_sdk import Alex, Session` working even if a
    # future refactor lazy-loads the client.
    if name in ("Alex", "Session", "AgentEvent", "EventType", "ToolInfo"):
        from . import client as _client

        return getattr(_client, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
