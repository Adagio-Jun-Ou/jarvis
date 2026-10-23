#!/usr/bin/env python
"""Regenerate ``docs/TOOLS.md`` from the tool registry.

The doc is the human-facing contract for what Alex can do; the registry
is the machine-facing one. Keeping the doc hand-written meant it drifted
every time a tool was added, so it is generated instead:

    python scripts/gen_tools_doc.py          # rewrite the doc
    python scripts/gen_tools_doc.py --check  # fail if it would change

``tests/test_tools_doc.py`` asserts the result stays in sync.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("ALEX_ENV", "dev")

DOC = ROOT / "docs" / "engine" / "TOOLS.md"

# Capability family -> section heading, in the order they should appear.
FAMILIES: dict[str, str] = {
    "browser": "Browser (VM)",
    "code": "Code intelligence (LSP)",
    "compute": "Compute & VM CLI",
    "email": "Email",
    "files": "Files & workspace",
    "internal_api": "Internal APIs",
    "knowledge": "Enterprise knowledge (RAG)",
    "agents": "Agent delegation — swarm fan-out to bounded sub-agents",
    "learning": "Learning & routines",
    "memory": "Memory & knowledge graph",
    "office": "Office documents",
    "orders": "Standing orders",
    "practices": "Best-practice library",
    "system": "System packages",
    "tasks": "Tasks",
    "workflow": "Workflows (the enterprise n8n instance)",
    "desktop": "Desktop control (screen, mouse, keyboard)",
    "home": "Home automation (Home Assistant, local commands)",
    "jarvis": "Jarvis native tools (adapted)",
}

HEADER = """# Alex — Tool & capability reference

_Auto-generated from `agent/tools/registry.py` by `scripts/gen_tools_doc.py`.
{n_tools} tools across {n_families} capability families._

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
- **MCP / direct:** `GET /api/tools` returns the ACL-filtered manifest (`{{name, description, inputSchema}}`);
  `POST /api/tools/{{name}}` with `{{"args": {{...}}}}` invokes one tool (same governance).
  `python -m alex_sdk.mcp` serves the same manifest to any MCP host over stdio.
- **SDK:** `alex.tools()` lists them; `alex.call_tool(name, **args)` invokes one.

## Tools by capability family
"""

FOOTER = (
    "\n_`*` marks a required input. Identity (`user_email`, `instance_scope`) "
    "is injected by the VM and never supplied by the caller._\n"
)


def build() -> str:
    from agent.tools.registry import TOOL_REGISTRY, input_schema_for

    by_family: dict[str, list] = {}
    for name, spec in sorted(TOOL_REGISTRY.items()):
        family = spec.capability.split(".", 1)[0]
        by_family.setdefault(family, []).append((name, spec))

    unknown = set(by_family) - set(FAMILIES)
    if unknown:
        raise SystemExit(
            f"unknown capability families {sorted(unknown)}; "
            f"add them to FAMILIES in {Path(__file__).name}"
        )

    out = [
        HEADER.format(n_tools=len(TOOL_REGISTRY), n_families=len(by_family))
    ]
    for family, heading in FAMILIES.items():
        entries = by_family.get(family)
        if not entries:
            continue
        out.append(f"\n### {heading}\n")
        out.append("| Tool | Capability | Inputs |")
        out.append("|------|------------|--------|")
        for name, spec in entries:
            schema = input_schema_for(spec)
            props = schema.get("properties") or {}
            required = set(schema.get("required") or [])
            inputs = ", ".join(
                f"`{p}`*" if p in required else f"`{p}`" for p in props
            ) or "—"
            out.append(f"| `{name}` | `{spec.capability}` | {inputs} |")
        out.append("")
    return "\n".join(out).rstrip("\n") + "\n" + FOOTER


def main() -> int:
    text = build()
    if "--check" in sys.argv:
        current = DOC.read_text(encoding="utf-8") if DOC.exists() else ""
        if current != text:
            print("docs/TOOLS.md is out of date; run scripts/gen_tools_doc.py")
            return 1
        print("docs/TOOLS.md is up to date")
        return 0
    DOC.write_text(text, encoding="utf-8")
    print(f"wrote {DOC.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
