"""
agent.tools — the tool registry for the Alex 2.0 agent.

The :mod:`agent.tools.registry` module is the single source of
truth for every tool the agent can call.  The harness
(:mod:`agent.harness`) reads this registry to build
:class:`agent.omnigent.AgentDef` instances; the SDK and the
FastAPI ``/api/tools`` route serve the same data as a public
manifest.
"""

from .registry import TOOL_REGISTRY, ToolSpec, tools_by_capability

__all__ = ["TOOL_REGISTRY", "ToolSpec", "tools_by_capability"]
