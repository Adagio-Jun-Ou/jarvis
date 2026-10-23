"""
agent.harness — the Alex 2.0 agentic harness.

The harness is the single seam every Alex surface (FastAPI route,
SDK, tests) calls into to drive one turn.  The components:

* :class:`EventPump` — thread-safe SSE event subscriber registry
  (:mod:`agent.harness.event_pump`).  Every :class:`alex.events.Event`
  the agent produces is fanned out to every subscriber (SSE
  generators, the audit log, the in-memory test recorder).
* :func:`executor_for` — selects the right Omnigent executor for
  the current mode (dev → Ollama Cloud, production → Databricks
  Model Serving) (:mod:`agent.harness.executor_factory`).
* :class:`ConstitutionAdapter` — the bridge between
  :class:`agent.constitution.ConstitutionalInterceptor` and the
  Omnigent :class:`FunctionPolicy` shape
  (:mod:`agent.harness.constitution_adapter`).
* :class:`ReactLoop` — the single-turn orchestrator
  (:mod:`agent.harness.react_loop`).  Reads
  :data:`agent.tools.registry.TOOL_REGISTRY`, builds an
  :class:`AgentDef`, and drives the executor through one turn.
* :class:`agent.vm.VM` — the top-level VM that owns the executor,
  the browser OS, the session, the event pump, the constitution,
  and the audit log.  Most callers construct a VM and call
  ``vm.run_turn(...)``.

The :mod:`agent.tools.registry` module is the single source of
truth for tool metadata; both the harness and the SDK read it.
"""

from .constitution_adapter import ConstitutionAdapter
from .event_pump import EventPump, Subscriber
from .executor_factory import executor_for
from .react_loop import ReactLoop

__all__ = [
    "ConstitutionAdapter",
    "EventPump",
    "ReactLoop",
    "Subscriber",
    "executor_for",
]
