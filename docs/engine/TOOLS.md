# Alex — Tool & capability reference

_Auto-generated from `agent/tools/registry.py` by `scripts/gen_tools_doc.py`.
74 tools across 18 capability families._

Every tool runs through the VM's governed path: the caller's **effective ACL** must grant
the tool's capability, the **constitution** is evaluated before execution, arguments are
validated, and the call is **audited**. High-risk capabilities (`compute.install`,
`system.install`, `compute.download`, `compute.shell`) require a **second approver** to
delegate (see `governance/delegation.sql`).

The **extended library** (Jarvis's adapted native tools) is registered and governed the
same way but deferred out of the default working set: a turn admits the best matches to
the request, a successful use keeps them for later turns, and the model can search the
rest with `find_tools` — matches become callable in the same turn.

## How to invoke

- **Agent (chat):** the LLM calls tools during `/api/chat`; results stream as `tool`/`artifact` events.
- **MCP / direct:** `GET /api/tools` returns the ACL-filtered manifest (`{name, description, inputSchema}`);
  `POST /api/tools/{name}` with `{"args": {...}}` invokes one tool (same governance).
  `python -m alex_sdk.mcp` serves the same manifest to any MCP host over stdio.
- **SDK:** `alex.tools()` lists them; `alex.call_tool(name, **args)` invokes one.

## Tools by capability family


### Browser (VM)

| Tool | Capability | Inputs |
|------|------------|--------|
| `browser_click` | `browser.act` | `session_id`*, `tab_id`*, `selector`* |
| `browser_close_tab` | `browser.use` | `session_id`*, `tab_id`* |
| `browser_extract` | `browser.read` | `session_id`*, `tab_id`*, `selector`* |
| `browser_list_tabs` | `browser.read` | `session_id`* |
| `browser_navigate` | `browser.use` | `session_id`*, `tab_id`*, `url`* |
| `browser_open_in_user_session` | `browser.use` | `session_id`*, `tab_id`*, `mode`* |
| `browser_screenshot` | `browser.read` | `session_id`*, `tab_id`* |
| `browser_submit` | `browser.act` | `session_id`*, `tab_id`*, `selector`*, `confirmation_token`* |
| `browser_type` | `browser.act` | `session_id`*, `tab_id`*, `selector`*, `text`* |


### Code intelligence (LSP)

| Tool | Capability | Inputs |
|------|------------|--------|
| `code_definition` | `code.read` | `path`*, `line`*, `column`* |
| `code_diagnostics` | `code.read` | `path`* |
| `code_hover` | `code.read` | `path`*, `line`*, `column`* |
| `code_outline` | `code.read` | `path`* |
| `code_references` | `code.read` | `path`*, `line`*, `column`* |
| `code_search_symbols` | `code.read` | `query`* |


### Compute & VM CLI

| Tool | Capability | Inputs |
|------|------------|--------|
| `bash` | `compute.bash` | `argv`* |
| `download_file` | `compute.download` | `url`*, `filename` |
| `pip_install` | `compute.install` | `packages`*, `upgrade` |
| `run_python` | `compute.python` | `code`* |
| `vm_exec` | `compute.shell` | `argv`* |


### Email

| Tool | Capability | Inputs |
|------|------------|--------|
| `read_inbox` | `email.read` | `limit` |
| `search_email` | `email.read` | `query`*, `limit` |
| `summarize_email` | `email.read` | `email_id`* |


### Files & workspace

| Tool | Capability | Inputs |
|------|------------|--------|
| `create_file` | `files.write` | `path`*, `content`* |
| `present_files` | `files.read` | — |
| `str_replace` | `files.write` | `path`*, `old`*, `new`* |
| `undo_file_change` | `files.write` | — |
| `view` | `files.read` | `path`* |


### Internal APIs

| Tool | Capability | Inputs |
|------|------------|--------|
| `api_act` | `internal_api.act` | `integration_id`*, `path`*, `body_json`, `confirm_token` |
| `api_read` | `internal_api.read` | `integration_id`*, `path`* |


### Enterprise knowledge (RAG)

| Tool | Capability | Inputs |
|------|------------|--------|
| `find_tools` | `knowledge.read` | `query`* |
| `ingest_document` | `knowledge.write` | `title`*, `text`*, `source_ref`, `scope`, `replace` |
| `ingest_workspace_file` | `knowledge.write` | `path`* |
| `list_knowledge` | `knowledge.read` | — |
| `search_knowledge` | `knowledge.read` | `query`*, `k` |


### Agent delegation — swarm fan-out to bounded sub-agents

| Tool | Capability | Inputs |
|------|------------|--------|
| `delegate_subagents` | `agents.delegate` | `tasks`* |


### Learning & routines

| Tool | Capability | Inputs |
|------|------------|--------|
| `run_routine` | `learning.read` | `name_or_query`* |


### Memory & knowledge graph

| Tool | Capability | Inputs |
|------|------------|--------|
| `add_edge` | `memory.write` | `src`*, `rel`*, `dst`*, `evidence`, `confidence` |
| `graph_about_me` | `memory.read` | — |
| `graph_entity` | `memory.read` | `name`* |
| `graph_link` | `memory.write` | `src`*, `rel`*, `dst`*, `evidence` |
| `graph_neighbors` | `memory.read` | `entity`*, `hops` |
| `graph_query` | `memory.read` | `text`*, `limit` |
| `recall` | `memory.read` | `query`, `k` |
| `remember` | `memory.write` | `content`*, `kind`, `source_ref` |


### Office documents

| Tool | Capability | Inputs |
|------|------------|--------|
| `create_document` | `office.write` | `filename`*, `title`, `body`, `sections` |
| `create_presentation` | `office.write` | `filename`*, `title`, `slides` |
| `create_spreadsheet` | `office.write` | `filename`*, `sheets`, `headers`, `rows` |
| `read_spreadsheet` | `office.read` | `path`*, `max_rows` |


### Standing orders

| Tool | Capability | Inputs |
|------|------------|--------|
| `list_standing_orders` | `orders.read` | — |
| `run_standing_order` | `orders.run` | `order_id`*, `note` |


### Best-practice library

| Tool | Capability | Inputs |
|------|------------|--------|
| `propose_practice` | `practices.propose` | `title`*, `guidance`*, `rationale`, `applies_to` |
| `search_practices` | `practices.read` | `query` |


### System packages

| Tool | Capability | Inputs |
|------|------------|--------|
| `system_install` | `system.install` | `packages`*, `manager` |


### Tasks

| Tool | Capability | Inputs |
|------|------------|--------|
| `ask_user` | `tasks.update` | `question`*, `options` |
| `check_acceptance` | `tasks.update` | `crit_no`*, `status`*, `note` |
| `check_plan_step` | `tasks.update` | `step_no`*, `status`*, `note` |
| `create_task` | `tasks.create` | `title`*, `description`, `priority`, `due_at` |
| `list_my_tasks` | `tasks.read` | `status` |
| `plan_task` | `tasks.update` | `steps`* |
| `update_task_status` | `tasks.update` | `task_id`*, `status`* |


### Workflows (the enterprise n8n instance)

| Tool | Capability | Inputs |
|------|------------|--------|
| `n8n_create_workflow` | `workflow.create` | `name`*, `nodes`*, `connections` |
| `n8n_execution_status` | `workflow.read` | `execution_id`* |
| `n8n_list_workflows` | `workflow.read` | — |
| `n8n_run_workflow` | `workflow.trigger` | `workflow_id`*, `payload` |


### Desktop control (screen, mouse, keyboard)

| Tool | Capability | Inputs |
|------|------------|--------|
| `desktop_click` | `desktop.act` | `x`*, `y`*, `button`, `clicks` |
| `desktop_find_text` | `desktop.read` | `text`*, `monitor` |
| `desktop_hotkey` | `desktop.act` | `keys`* |
| `desktop_key` | `desktop.act` | `key`*, `modifiers` |
| `desktop_screenshot` | `desktop.read` | `monitor` |
| `desktop_scroll` | `desktop.act` | `direction`, `amount` |
| `desktop_type` | `desktop.act` | `text`* |


### Home automation (Home Assistant, local commands)

| Tool | Capability | Inputs |
|------|------------|--------|
| `home_control` | `home.act` | `device`*, `action`*, `value` |
| `home_status` | `home.read` | `device` |

_`*` marks a required input. Identity (`user_email`, `instance_scope`) is injected by the VM and never supplied by the caller._
