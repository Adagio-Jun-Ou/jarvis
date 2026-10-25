# Omnigent integration — the agent-loop backbone

Alex 2.0's agent loop is built on the public Omnigent project
([github.com/omnigent-ai/omnigent](https://github.com/omnigent-ai/omnigent),
Apache-2.0).  We vendor a focused, audit-friendly subset of Omnigent
under `agent/omnigent/` and adapt it so it works against our
existing LLM, identity, and tracing layers without pulling in the
upstream Claude/Codex/Cursor subprocess executors, the
`databricks-sdk` / `~/.databrickscfg` credential chain, the
`bwrap`/`seatbelt` sandbox backends, or the `omni` CLI / web UI.

## What we vendored

| Module | Purpose |
|---|---|
| `agent/omnigent/__init__.py` | Public re-exports; md5 FIPS patch; legacy env-var shim |
| `agent/omnigent/_env_compat.py` | Legacy env-var mirror (no-op in Alex 2.0) |
| `agent/omnigent/datamodel.py` | `AgentDef`, `History`, `Message`, `Memory`, `Credentials`, etc. |
| `agent/omnigent/executor.py` | `Executor`, `ExecutorConfig`, all `ExecutorEvent` dataclasses, `MockExecutor`, `iterate_blocking_stream` |
| `agent/omnigent/tools.py` | `Tool`, `FunctionTool`, `MCPTool`, `AgentTool`, `SkillTool`, `HandoffTool`, etc. |
| `agent/omnigent/policies.py` | `Policy`, `FunctionPolicy`, `PromptPolicy` (stub), `PolicyResult`, `PolicyAction` |
| `agent/omnigent/async_utils.py` | `run_sync_on_thread` helper for blocking SDK calls |
| `agent/omnigent/tracing.py` | MLflow-aware shim (upstream uses OpenTelemetry) |
| `agent/omnigent/databricks_executor.py` | Production executor backed by `alex.llm.DatabricksServingLLM` |
| `agent/omnigent/local_ollama_executor.py` | Dev executor backed by `alex.llm.OllamaCloudLLM` |
| `agent/omnigent/loader.py` | YAML loader for `AgentDef.from_yaml` |
| `agent/omnigent/_alex_bridge.py` | `ConstitutionPolicy` (L2 adapter), `build_agent_def`, `run_turn` that streams `ExecutorEvent` → `alex.events` SSE |

## What we removed

The following upstream surfaces are intentionally **not** vendored;
they would only confuse Alex's deployment shape:

* `claude_native_*`, `codex_native_*`, `cursor_native_*`,
  `antigravity_native_*`, `goose_*`, `kimi_*`, `kiro_*`,
  `hermes_*`, `copilot_*`, `pi_*`, `opencode_*` subprocess
  executors and harnesses.
* `acp_*` and the `omn` CLI / web UI / Discord-bot server.
* `bwrap_sandbox.py`, `_seccomp.py`, `seatbelt` SBPL emitter.
* `credential_proxy.py`, the `oa_cred_*` placeholder mechanism,
  and the `~/.databrickscfg` parser.
* `egress/` L7 MITM proxy and the cloud-trap CIDR list.
* `claude_native_bridge`, `claude_launcher`, `codex_*` IPC
  forwarders.
* The `omnigent.runtime.telemetry` runtime (we use MLflow
  instead).

## The seam: `agent.vm.VM` (canonical) and `agent.harness._alex_bridge` (vendored)

The canonical Alex 2.0 process model is :class:`agent.vm.VM`.  It
owns the LLM executor, the browser OS, the constitution, the
audit log, the event pump, and the tool registry.  The FastAPI
routes, the SDK, and the test harness all construct a VM and call
its methods; the ReactLoop delegates to the executor through the
bridge in `agent.omnigent._alex_bridge`.  See
`docs/HARNESS_TOOLS_VM.md` for the full walkthrough.

`build_agent_def(name, prompt, tools, capabilities=...)` (in
`agent.omnigent._alex_bridge`) turns a flat Python-callables dict
into an Omnigent `AgentDef` whose every `FunctionTool` carries the
right `scopes` for the :mod:`agent.guardrails` dual-ACL gate.  The
same call attaches a `ConstitutionPolicy` (a `FunctionPolicy`
whose callable is a thin wrapper around
:func:`agent.constitution.ConstitutionalInterceptor.check` /
`.record`).  The bridge emits a `law_block` SSE event on
`ConstitutionalBlock` and returns `PolicyResult(action=DENY)`.

The harness's `ConstitutionAdapter` is the new canonical
seam (in `agent.harness.constitution_adapter`); it implements
the same behavior as the bridge's `ConstitutionPolicy` and is
the one the VM uses.

`run_turn(executor, agent_def, history, ...)` drives the executor
through one ReAct turn.  Every `ExecutorEvent` is translated to an
`alex.events.Event` so the FastAPI SSE route (`app/app.py`)
sees the same event stream it always has.

## Executors

Two executors ship with the vendored copy:

* `DatabricksExecutor` — production, backed by
  :class:`alex.llm.DatabricksServingLLM`.  Streams
  :class:`alex.llm.LLMDelta` chunks and converts them to Omnigent
  `TextChunk` / `ToolCallRequest` / `TurnComplete` events.
* `LocalOllamaExecutor` — dev/test, backed by
  :class:`alex.llm.OllamaCloudLLM`.  Same `run_turn` interface.

`MockExecutor` is also available for unit tests.

## The browser tool is a `FunctionTool`

The agent's nine browser operations
(`browser_navigate`, `browser_click`, `browser_type`,
`browser_extract`, `browser_screenshot`, `browser_submit`,
`browser_list_tabs`, `browser_close_tab`,
`browser_open_in_user_session`) live in
:mod:`agent.agentic_browser` and are registered as
`FunctionTool` entries in :mod:`agent.browser_tools`.  They go
through the same constitution L2 interceptor as every other tool,
so the two-phase submit (A8) and the no-mailto/no-tel rule apply
uniformly.  See `docs/BROWSER_OS.md` for the full surface.

## Tracing

`enable_tracing()` starts an MLflow trace; every agent turn
emits a `agent:<name>` AGENT span, every tool call a `tool:<name>`
TOOL span, and every policy evaluation a `policy:<name>`
GUARDRAIL span.  When MLflow is not importable (e.g. a unit
test) the helpers degrade to logging-only mode.

## What the upstream contract is preserved

* `AgentDef.from_yaml(path)` still works against the YAML
  keys we support (`name`, `prompt`, `instructions`, `tools`,
  `policies`, `executor`, `params`, `metadata`).
* `Executor.run_turn(messages, tools, system_prompt, config)`
  yields the same event dataclasses upstream.
* `FunctionPolicy` accepts a callable shaped
  `(event, config?) -> dict | PolicyResult`.
* `MockExecutor` has the same `enqueue_response`,
  `enqueue_tool_call`, `enqueue_events` API.

## License

The vendored copy is Apache-2.0.  The original Omnigent copyright
notice is in `agent/omnigent/NOTICE`.  The adaptations are
copyright (c) the Alex 2.0 authors and licensed under the same
Apache-2.0 terms.

## Tests

* `tests/test_omnigent_imports.py` — public surface imports
  and re-exports (4 tests).
* `tests/test_omnigent_executor_runs.py` —
  `MockExecutor` event sequences, `FunctionTool` schema
  generation, `build_agent_def`, `run_turn`, event mapping,
  `ConstitutionPolicy` allow/deny, `load_agent_def`
  (14 tests).
