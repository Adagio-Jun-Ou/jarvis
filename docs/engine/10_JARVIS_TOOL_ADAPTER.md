# Jarvis's native tools under the engine

`agent/jarvis_tools.py` adapts every Jarvis `tools.base.Tool` subclass it can
build into the engine's governed registry. Jarvis's tools keep running exactly
as they do under Jarvis's own agent — the adapter is the bridge, not a fork.

## What discovery covers

`discover()` builds tool instances the same way `run.py` does, one guarded
family per import, so a missing optional dependency costs one family rather
than the boot. Nineteen families currently load (~112 tools): system, system
sense, web, productivity, spatial, art/philosophy, physics, data, free-time,
screen, browser, terminal, native files, insight search, tool discovery,
evolution, music, project, oracle, cognitive AI.

Deliberately absent:

* `tools.meta_tools` — its tools need a Jarvis LLM client
  (`SelfReflectionEngine.llm.chat`), which the engine does not provide. The
  engine has its own review surface (`agent/execution_review.py`).
* `tools.tool_creation`, `tools.web_ui_refine` — they need Jarvis's agent/web
  interface and self-modification pipeline; the engine has its own governance
  for those concerns.

## Mapping

* **Capability**: `jarvis.<category>.<read|act>`.
  * `read` (power 0) for plain tools.
  * `act` (power 2, policy-ASK) when the tool declares
    `requires_permission`, **or** when its name is in `_ALWAYS_ACT` — Jarvis
    understates some tools (`shell` declares False but runs arbitrary
    commands; the project tools write/run/commit; approve/deny resolve
    someone else's sandbox request). Those are promoted regardless.
  * Category comes from `core.tool_registry` when it catalogues the tool, else
    from the family that built it (without that fallback, tools collapsed into
    a single `jarvis.misc.*` capability — a least-privilege problem).
* **Untrusted output**: web, browser, screen, data and multimodal results are
  fenced (`<untrusted_content>`), as are local file readers (`read_file`,
  `native_read`, `project_file_read`, …) — local files are an injection path
  too. The engine's own `view` / `read_inbox` carry the same flag.
* **Schema**: the synthesized typed signature feeds argument validation; the
  original JSON schema rides along as `ToolSpec.input_schema` so the model
  sees Jarvis's descriptions/enums. `input_schema_for()` returns that schema
  verbatim (`find_tools`, `/api/tools`, the generated doc all use it).
* **Execution**: Jarvis coroutines run on a worker thread with a private loop
  (`safe_execute`, bounded by `JARVIS_TOOL_TIMEOUT_S`, default 600s). Results
  map to `{"ok": ...}`.

## The deferred library

~112 adapted tools must not ride every model pass, so they register with
`deferred=True`: governed and audited, but outside the default working set.

A turn admits:

1. **Pinned tools** — from `find_tools` (search by keywords, up to 10 hits)
   or from **successful prior use** (a tool the agent proved it needs stays).
2. **Best lexical matches** to the user's message (`score_tool`: name words
   count double), capped by `ALEX_MAX_DEFERRED_TOOLS` (default 24).

`find_tools` pins into the live `AgentDef`; the bridge re-reads
`agent_def.tools` every pass, so a match is callable in the *same* turn. The
pinned set is a bounded LRU (oldest evicted) so a long session cannot inflate
every prompt. Pinning applies the same exposure filters as the working set
(capability ACL, plan mode) and reports honestly when matches cannot be used.

MCP discovery is cached per VM for `ALEX_MCP_DISCOVERY_TTL_S` (default 300s)
because the working set is rebuilt per turn; call-time authorization is still
re-resolved, so a revoked remote capability bites on the next call.

## Governance guarantees

Every adapted call passes the VM's single gate: capability ACL (checked at
*invocation* time — adapted tools have no `guarded()` wrapper of their own),
plan mode, run-stop, governor, constitution (L2), workspace policy
(allow/ask/deny), argument validation, audit row, and result fencing when the
spec says untrusted.

## Knobs

| Variable | Default | Effect |
|---|---|---|
| `ALEX_MAX_DEFERRED_TOOLS` | 24 | deferred tools admitted to one turn |
| `ALEX_MCP_DISCOVERY_TTL_S` | 300 | per-VM MCP discovery cache |
| `JARVIS_TOOL_TIMEOUT_S` | 600 | hard cap on one adapted tool call |

## Known limits

* Hot reload: `unregister_jarvis_tools()` cleans the registry, capability
  power and ask-policy maps, but a live VM's `AgentDef` keeps entries until
  its next per-turn rebuild. No production path unregisters today.
* Background jobs (`terminal_execute(background=true)`) run on their own
  worker thread and outlive the tool call; `check_jobs` reads them.
