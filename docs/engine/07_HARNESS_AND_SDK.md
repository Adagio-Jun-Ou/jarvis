# Harness design & the Alex SDK — what we borrowed and how it's governed

Four ideas from the best current harnesses, adapted so each stays inside the
constitution and dual-ACL model.

## 1. Learning loop (from Hermes — "grows with you")
`agent/learning_loop.py` + `pipelines/skill_reflection.py` + `control.skills`.
The instance derives named SKILLS from your repeated successful work and
applies them as compact procedure hints at runtime, getting visibly better at
your recurring tasks over weeks. **Governed twist:** a skill is a proposal
authored by the reflection job, can only chain ALREADY-APPROVED capabilities,
and goes active only after a human ratifies it in the transparency/governor
panel. Broadly useful skills get promoted org-wide and feed the playbook
library. Learning changes how well, never what's allowed.

## 2. Triggered strict review (fixing Codex's always-on harshness)
`agent/execution_review.py`. Per-step review is OFF for plain VM compute
(bash/python in scratch — the sandbox is the guardrail there). It FIRES only
on consequence: irreversible/external capabilities, in-flight constitutional
procedures, rights/money/external output, or elevated risk signals. When it
fires it uses a small model with a STRICT fixed preset (temperature 0, literal
rules + constitution) — fast, cheap, deterministic; it judges, it doesn't chat.
You keep the "check during execution" you liked without the tax on every step.

## 3. Visible reasoning (from OpenCode) via open weights
`agent/reasoning_stream.py`. Toggle "Show reasoning" and execution-planning
routes to an OPEN-WEIGHT reasoner (Hermes/DeepSeek/Qwen thinking models) served
on Databricks — so the chain-of-thought is real and legitimately inspectable,
not a post-hoc summary. Reasoning streams as collapsible `reasoning` events on
the same channel as the answer, and is stored per turn: a user (or an
investigation) can see WHY, not just what. Reasoning is still under the
constitution — the strict reviewer sees the plan; planned tools still pass
every gate.

## 4. A proper SDK (from Hermes's agent SDK + workflow ergonomics)
`sdk/alex_sdk/`. Governance is the type system:
- `@capability(...)`, `@tool(capability=..., reviewed=, confirm=)`,
  `@skill(...)`, `workflow(...).step(...).confirm(...)`, `Agent(...).deploy()`.
- **Build-time validation**: a tool or workflow step referencing an undeclared
  capability fails at `deploy()`/`dev()`, not at 2am. (Smoke-tested.)
- Workflows compile to n8n definitions (confirm steps -> requires_confirmation)
  or native agent plans.
- `alex dev` runs the whole loop locally against a dev catalog with a mocked
  identity; `alex deploy -t prod` validates, runs the eval + constitutional
  gate, and ships the endpoint.

```python
from alex_sdk import Agent, tool, capability, workflow

capability("reports.compile"); capability("email.draft")

@tool(capability="reports.compile", reviewed=True)
def compile_report(quarter: str) -> str:
    "Compile the quarterly report."
    ...

weekly = (workflow("weekly-brief")
          .step("gather", capability="reports.compile")
          .confirm("send", capability="email.draft"))     # human-gated

Agent(name="analyst",
      models=["hermes-4-70b", "claude-sonnet-5"],          # visible reasoning first
      tools=[compile_report], workflows=[weekly],
      memory="vector+graph", learning=True, review="triggered"
).deploy(target="prod")
```
