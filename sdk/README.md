# Alex SDK

The governed Python SDK for the Alex AI-employee platform — the
programmatic twin of the product UI. Hard dependency: `httpx` only.

## QUICKSTART

Run the app locally in dev mode (defaults below assume it listens on
`http://127.0.0.1:8765`), then:

```python
from alex_sdk import Alex, EventType

# Dev auth: mints a token from GET /api/dev/token (dev mode only).
with Alex.for_dev("http://127.0.0.1:8765", "alice@example.com") as alex:
    print(alex.me())          # {'email': 'alice@example.com', ...}

    # --- streamed chat, with reasoning ---------------------------------
    session = alex.session("quickstart")
    for ev in session.run("What's urgent today?", show_reasoning=True):
        if ev.type == EventType.TEXT:
            print(ev.text, end="", flush=True)
        elif ev.type == EventType.REASONING:
            print(f"\n[reasoning] {ev.text}")
        elif ev.type == EventType.TOOL:
            print(f"\n[tool] {ev.tool}({ev.args})")
        elif ev.type == EventType.LAW_BLOCK:
            print(f"\n[law_block] {ev.article}: {ev.detail}")
        elif ev.type == EventType.CONFIRMATION_REQUIRED:
            ev.confirm()  # the user said yes

    # --- teach-by-editing ----------------------------------------------
    draft = "Dear sir, ..."
    for _ in session.correct("Hi team, ...", original_output=draft):
        pass  # drain the follow-up turn; the server captures the diff

    # --- tools, teaching, feedback -------------------------------------
    for t in alex.tools():                       # GET /api/tools (ACL-filtered)
        print(t.name, "-", t.description)
    alex.teach("To file an expense: open Workday, ...")   # POST /api/teach
    alex.feedback("sugg-123", "helpful")                  # POST /api/feedback

    # --- memory & graph ------------------------------------------------
    alex.remember("Prefers bullet-point summaries")
    print(alex.recall("summaries"))
    alex.add_edge("alice", "works_with", "bob")
    print(alex.graph_query("alice"))
    print(alex.list_workflows())
```

### MCP interop

```python
from alex_sdk.mcp import to_mcp_tools

with Alex.for_dev("http://127.0.0.1:8765", "alice@example.com") as alex:
    manifest = alex._request("GET", "/api/tools")   # or alex.tools()
    defs = to_mcp_tools(manifest)   # pure: [{'name', 'description', 'inputSchema'}]
```

`alex_sdk.mcp.serve_mcp()` is a stub; it raises `NotImplementedError`
with instructions unless the optional `mcp` package is installed.

### CLI

```
alex --app-url http://127.0.0.1:8765 tools          # name + description table
alex teach "To triage my inbox, start with flagged mail"
alex chat "Summarize my inbox" --reasoning          # reasoning lines dimmed
alex skills
alex cap list
```

Auth for the CLI comes from `--token` or `ALEX_OBO_TOKEN`; the base URL
from `--app-url` or `ALEX_APP_URL`.

## Event types

The chat SSE stream emits: `text`, `reasoning`, `tool`, `artifact`,
`law_block`, `memory`, `confirmation_required`, `error`, `done`.
`AgentEvent.type` is the wire string (compare with `EventType`);
`AgentEvent.data` is the payload minus the discriminator.

## Layout

- `alex_sdk/client.py` — `Alex`, `Session`, `AgentEvent`, `EventType`,
  `ToolInfo`, typed errors, retries, SSE parsing.
- `alex_sdk/mcp.py` — MCP tool-definition conversion (`to_mcp_tools`).
- `alex_sdk/memory.py` — per-user vector + graph memory helper.
- `alex_sdk/workflow.py` — typed workflow builder.
- `alex_sdk/review.py` — async-judge review policy preset.
- `alex_sdk/core.py` — `Agent`, `@tool`, `@capability`, `@skill`.
- `alex_sdk/cli.py` — the `alex` console script.
