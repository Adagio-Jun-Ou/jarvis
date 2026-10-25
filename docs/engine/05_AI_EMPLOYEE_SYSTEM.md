# Alex — The AI Employee System (design + how the pieces fit)

Every employee gets a persistent AI instance — same brain (governed models), personal context, own workspace on the agent computer — governed centrally by a control plane that C-suite & Product Dev own.

## 1. The Control Plane (`media.control` schema)

The single place governors (`alex-governors` group, Okta-synced) manage the entire AI workforce:

| Table | Governs |
|---|---|
| `knowledge_base` | What every instance knows: policies, best practices, org facts, playbooks. Pinned entries ride natively in every session's context; the rest is embedded for RAG. Versioned, audience-targeted by group. |
| `model_registry` | Which models exist, who may use each, per-user daily token budgets per route. The UI model picker and the agent's router both read this — flipping `enabled=false` removes a model everywhere instantly. |
| `vm_policy` | The agent computer's setup: node type, auto-terminate, pip allowlist, command timeout, egress. A sync job renders rows → cluster policy JSON. Changing the fleet's VM = a row update. |
| `employee_acl` / `instance_acl` | The **dual approved-access lists** (below). |
| `standing_orders` / `standing_order_briefs` | Active recurring mandates + their overnight-compiled prep. |
| `workflow_catalog` | The only n8n workflows the agent may ever touch. |
| `sessions` | Registry + compacted summaries of every persistent/group/branch session. |

The agent has SELECT on all of it and MODIFY on none of it (except its own sessions). **The AI is governed by the control plane; it can never write to its own rules.**

## 2. Dual-ACL execution guardrail

Every consequential tool is wrapped (`agent/guardrails.py`): before the tool body runs, the capability must exist in **both** the employee's ACL (what this human may have done on their behalf) **and** the instance's ACL (what this AI instance may do at all). The check happens at execution time in code — prompt injection can talk the model into *wanting* anything, but the intersection gate means it executes nothing outside the approved lists. Denials tell the user which capability to request from a governor, and land in the ledger.

## 3. Context & knowledge: native + RAG, token-budgeted

`agent/context_engine.py` assembles context in tiers with hard budgets: static system + guardrails (~1.2k, prompt-cached), pinned org policy for the user's groups (~2k, prompt-cached), user layer — memory highlights + today's standing-order headlines (~1k), rolling session summary (~0.8k). Everything else — non-pinned knowledge, the user's own uploaded documents (per-user-filtered vector index, provenance-tagged) — arrives by retrieval only when a turn needs it. A three-week-old session resumes for ~5k context tokens, not the whole history. Compaction runs on the fast tier every N turns.

So the agent is *always* policy-aware (pinned layer) and *always* current (overnight jobs + RAG), while the marginal token cost per message stays flat.

## 4. Standing orders

Users (or their managers) register standing orders: "track competitor streaming launches," "keep the Q3 slate rights posture current." At 04:30 the prep job gathers each order's fresh signals — new high-priority email understanding, task movement, audience deltas, library additions — and pre-compiles a brief plus concrete suggested actions, each already mapped to an approved capability. The first message of the day surfaces them: one click and the agent executes from the pre-compiled context. Expensive thinking happens once, off-peak, on the fast tier.

## 5. Email intelligence: support-only, injection-proof

The email system is deliberately **not** an acting agent:

- The analysis models have **no tools** — they can only emit structured JSON. Email content is wrapped as data with an explicit ignore-embedded-instructions frame, and even a total prompt-injection success can produce nothing but a mis-ranked annotation, never an action. The execution layer never receives raw email text.
- Scope is exclusively the employee's own mailbox; understanding never crosses employees.
- Output per mailbox: priority ranking with a one-line why; per-thread state (who said what, commitments made, who owes whom, what's outstanding); and the **understanding blocks** rendered between emails in the UI — "Maria agreed to the March window; you still owe her the rights confirmation — related: task T-1042." Click one and the main agent (with its guardrails) helps you act.

Email → priority inbox → understanding → suggested action → *human* clicks → guarded execution. The AI reads and connects; the human decides.

## 6. Taskboard, group chat, private branches

- **AI-assisted taskboard**: the agent sees your tasks via `list_my_tasks` (OBO — only yours), links email commitments to task cards, and drafts subtask breakdowns; writes go through the dual-ACL `tasks.*` capabilities.
- **Group sessions**: a project sandbox where the team + Alex work together, sharing a project workspace folder. The agent's capability set in a group session is the **intersection of every participant's ACL** — it can never do for the group what it couldn't do for the most-restricted member.
- **Private branches**: any member forks the group session — summary + recent context + a copy-on-write snapshot of project files — into a personal branch under their own ACLs. Nothing merges back automatically; you *present* chosen outputs to the group.

## 7. Workflows: n8n + Claude Code

- The agent triggers only catalog-listed n8n workflows; each carries a required capability and, for consequential ones, a `requires_confirmation` flag — the agent returns a confirm card and the **user's click** fires the webhook, not the model.
- For complex automation building, Alex escalates to a **Claude Code session on its own VM** (Agent Bricks supports the Claude Code SDK harness): workplan approved by the user first, bounded token/turn budget, no egress, output is a workflow definition + review report in `outputs/` — and only a governor can promote it into the catalog. The AI builds workflows; humans approve them into existence.

## 8. Performance & token economics summary

Precompute (email intel, standing orders, summaries) on the fast tier off-peak; prompt-cache the stable context tiers; RAG instead of resident context; per-route daily token budgets in the model registry enforced alongside Gateway budgets; Lakebase for all hot session state (ms reads); Delta for everything governed and analytical. The result: morning-ready intelligence, flat per-message cost, sub-second time-to-first-token.
