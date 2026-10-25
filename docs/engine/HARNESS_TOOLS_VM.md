# Harness, Tools, VM — the Alex 2.0 agentic loop

Alex 2.0 is a single-process agentic system.  Every user-facing
operation — a chat message, a browser action, a skill invoke — goes
through the same seam: the **VM**.  This document walks through the
three layers that compose the loop, the canonical public surface
of each, and how a request flows from the React frontend through
the FastAPI route through the VM to the LLM and back.

## TL;DR

| Layer | What | Where |
|---|---|---|
| **Harness** | The wiring between LLM, constitution, tools, and the SSE pump | `agent/harness/` |
| **Tools** | The single source of truth for every tool the agent can call | `agent/tools/registry.py` |
| **VM** | The canonical process model — owns everything one Alex session needs | `agent/vm.py` |

The VM is the only seam.  The FastAPI route, the SDK, and the
test harness all construct a VM and call its methods.

## The VM (`agent/vm.py`)

The :class:`agent.vm.VM` is the canonical Alex 2.0 process model.
A VM is created once per Alex session (one per signed-in user per
chat session).  The class owns:

1. an **LLM executor** — `OllamaCloudLLM` in dev,
   `DatabricksServingLLM` in production.  The selection is made
   by :func:`agent.harness.executor_for` reading
   `alex.config.Settings.mode`.
2. a **browser OS backend** — `InMemoryBrowserOS` in dev,
   `PlaywrightBrowserOS` in production (loaded lazily).
3. a **session identity** — the OBO user email and instance
   scope.
4. a **constitution interceptor** — the canonical L2 layer
   (`agent.constitution.ConstitutionalInterceptor`).
5. an **event pump** — the SSE event subscriber registry
   (`agent.harness.EventPump`).
6. an **audit log** — writes to `media.alex.agent_actions`
   (A3) and `media.control.constitution_violations`.

Public surface (one method per user-facing operation):

```python
from agent.vm import make_vm

vm = make_vm(session_id="s1", user_email="alice@x.com")
vm.subscribe_sse(lambda ev: print(ev.type))

# Chat
last, history = vm.run_turn(text="summarize my inbox")

# Browser OS
session = vm.open_browser_session()
tab = vm.open_browser_tab(session.session_id, url="alex://welcome")
vm.navigate_browser(session.session_id, tab.tab_id, url="alex://docs")
res = vm.submit_browser(session.session_id, tab.tab_id, selector="form")
# res == {"awaiting_confirmation": True, "confirmation_token": "..."}
res = vm.submit_browser(
    session.session_id, tab.tab_id,
    selector="form", confirmation_token=res["confirmation_token"],
)
vm.mirror_browser(session.session_id, tab.tab_id, mode="co-drive")

# Lifecycle
vm.close()
```

The VM exposes a context manager so callers can use
`with make_vm(...) as vm: ...` and have the executor closed
automatically.

## The harness (`agent/harness/`)

The harness is the small, focused wiring layer that connects the
LLM, the constitution, the tool registry, and the event pump.
The public symbols are:

| Symbol | Purpose |
|---|---|
| `EventPump` | Thread-safe fan-out of `alex.events.Event` instances |
| `Subscriber` | The shape every subscriber callable conforms to |
| `executor_for(mode)` | Returns the right Omnigent `Executor` for the mode |
| `ConstitutionAdapter` | A `FunctionPolicy` that delegates to the constitution |
| `ReactLoop` | The single-turn orchestrator (called by `VM.run_turn`) |

The ReactLoop is the only piece that drives the executor.  It
builds an `AgentDef` from the tool registry, attaches the
ConstitutionAdapter, runs the executor, and translates every
`ExecutorEvent` to an `alex.events.Event` on the event pump.

```python
from agent.harness import executor_for, ConstitutionAdapter, ReactLoop

executor = executor_for("dev")
adapter = ConstitutionAdapter(
    session_getter=lambda: vm._session,
    on_block=lambda payload, exc: vm.pump.emit(
        law_block_event(article=exc.article_id, detail=str(exc))
    ),
)
loop = ReactLoop(
    executor=executor,
    agent_def=vm.agent_def,
    history=vm.session.history,
    system_prompt=None,
    pump=vm.pump,
    constitution_adapter=adapter,
)
loop._vm = vm  # back-reference for _invoke_tool
last, _ = loop.run()
```

The `ConstitutionAdapter` is the only path by which tool calls
go through the four constitution layers (L1 prompt + L2
interceptor + L3 critic + L4 deploy gate).  When a `ConstitutionalBlock`
fires, the adapter emits a `law_block` SSE event through the pump
*and* returns `PolicyResult(action=DENY)`.

## The tools (`agent/tools/registry.py`)

The :data:`agent.tools.registry.TOOL_REGISTRY` is the single source
of truth for every tool the agent can call.  Each entry is a
:class:`ToolSpec` with five fields:

| Field | Purpose |
|---|---|
| `name` | The LLM-facing name |
| `capability` | The ACL token `agent.guardrails.TOOL_CAPABILITY` resolves |
| `callable` | The Python function the LLM invokes |
| `description` | One-line summary (first line of the docstring) |
| `return_schema` | A JSON-Schema fragment for the tool's return shape |
| `constitution_articles` | The article IDs the tool touches |

The harness reads the registry to build `AgentDef`s; the SDK
and the FastAPI `/api/tools` route serve the same data as a
public manifest.

Adding a new tool is a single registration:

```python
TOOL_REGISTRY["my_new_tool"] = ToolSpec(
    name="my_new_tool",
    capability="domain.capability",
    callable=my_function,
    description="One-line summary",
    return_schema={"type": "object", "properties": {"ok": {"type": "boolean"}}},
    constitution_articles=("A3",),
)
```

The harness's `tests/test_tools_registry.py` asserts every entry
has all four required fields and the `capability` matches the
guardrails gate.

## How a request flows

The React frontend → FastAPI `/api/chat` route → VM → ReactLoop →
executor → SSE event pump → frontend.

1. **Frontend** opens an `EventSource` on
   `POST /api/chat` with the user's text.
2. **FastAPI route** authenticates the OBO token, then (in dev)
   constructs a :class:`VM` and a subscriber that puts every
   `alex.events.Event` onto an `asyncio.Queue`.
3. **VM**'s `run_turn(text=...)` appends the user message to the
   session history, builds a :class:`ReactLoop`, and runs it.
4. **ReactLoop** drives the executor's `run_turn(...)`, which
   yields `TextChunk` / `ToolCallRequest` / `TurnComplete` events.
5. The loop **translates every event** to an
   `alex.events.Event` and emits it through the **event pump**.
6. For every `ToolCallRequest`, the loop calls `VM._invoke_tool`,
   which:
   * Resolves the tool from the registry.
   * Audit-logs the call (A3).
   * Emits a `tool` event.
   * Calls the function (which may itself emit tool-internal events
     like `confirmation_required` for A8 two-phase submits).
   * Emits an `artifact` event with the result.
7. The **constitution interceptor** runs as a `FunctionPolicy` on
   every `tool_call` phase.  When it raises `ConstitutionalBlock`,
   the bridge emits a `law_block` event and returns DENY.
8. The **SSE stream** delivers the events to the frontend; the
   `AgentInspector` component renders them in real time.

## How a browser action flows

The React frontend → FastAPI `/api/browser/sessions/{sid}/tabs/{tid}/navigate`
route → VM → `InMemoryBrowserOS` → audit log → SSE event pump.

For destructive operations (`browser_submit`):

1. The frontend hits `/api/browser/.../submit` without a
   `confirmation_token`.
2. The VM calls `browser_submit(...)` which returns
   `{"awaiting_confirmation": True, "confirmation_token": "..."}`
   and emits a `confirmation_required` SSE event.
3. The frontend shows a "Run it?" card in the Agent Inspector.
4. The user clicks "Run it"; the frontend hits
   `/api/browser/.../submit` with the `confirmation_token`.
5. The VM re-validates the token and calls the actual submit.

This is the implementation of constitution A8.  Every
externalizing tool is two-phase.

## Audit log (A3)

Every consequential act is logged to `media.alex.agent_actions` by
the VM's `_DefaultAuditLog` (which uses the `InMemoryWarehouse` in
dev and the Databricks SDK's statement-execution client in
production).  Constitution violations are also logged to
`media.control.constitution_violations` by the
`ConstitutionalInterceptor` itself.

The audit log is best-effort: failures are logged but never block
the user-facing operation.  In production the
`browser_audit` job (`pipelines/browser_audit.py`) ships the prior
24h of activity to the Lakehouse nightly for governor review.

## Public imports

```python
# The canonical VM seam
from agent.vm import VM, VMSession, HarnessConfig, make_vm
from agent.agent import VM, VMSession, HarnessConfig, make_vm  # re-export

# The harness wiring
from agent.harness import (
    EventPump, Subscriber, ReactLoop,
    ConstitutionAdapter, executor_for,
)

# The tool registry
from agent.tools import TOOL_REGISTRY, ToolSpec, tools_by_capability
```

## Tests

* `tests/test_harness.py` — `EventPump` fan-out, `executor_for`
  selection, `ConstitutionAdapter` allow/deny/on_block hook.
* `tests/test_tools_registry.py` — every tool has the five
  required fields, every tool's capability is in the guardrails
  gate, every browser tool marks A3, `browser_submit` marks A8.
* `tests/test_vm_lifecycle.py` — construction, audit log,
  `_invoke_tool`, list/close, error path, context manager.
* `tests/test_vm_sse_protocol.py` — every VM op emits a
  canonical SSE event; sweep of all ops confirms the protocol
  contract.
* `tests/test_vm_browser_two_phase.py` — the full A8
  first-phase-emits-confirmation / second-phase-executes flow
  through the VM seam.

Combined, the harness + tools + VM layer is exercised by 51 tests
in addition to the existing 250, for a 301-test suite.  `ruff
check` and `ruff format --check` are clean.  `npm test` and
`npm run build` are clean.
