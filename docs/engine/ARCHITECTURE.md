# Alex — what it can do, and how it is built

**Scope: the system as it exists in this repository today.** Every claim
here was checked against the code, and each section names the module that
owns the behaviour so it can be checked again. Where this document and
any other in `docs/` disagree about *what the code currently does*, this
one is correct.

It deliberately does not describe the Databricks deployment target —
Unity Catalog function tools, Model Serving, AI Gateway, on-behalf-of
auth. That is [`02_ARCHITECTURE.md`](02_ARCHITECTURE.md), and it
describes where this is going rather than where it is. The two are
complementary: read this one for capability and structure, that one for
deployment.

Two sections: **what the system can do**, then **how it is structured**.
A closing section names the decisions that were deliberate and the gaps
that are known, because an architecture document that only lists
strengths is a brochure.

---

## Part 1 — Abilities

Alex is a governed enterprise agent with its own VM. Ten capabilities,
each with the module that owns it.

### 1. Answer from the organisation's knowledge
Two-stage retrieval over the enterprise corpus, with citations.
`agent/rag.py` · `agent/rerank.py`

Stage one fuses dense vectors, BM25 and graph expansion into a wide
candidate pool; stage two reranks that pool on signals fusion cannot
see. Every passage carries a citation and the ranks that produced it, so
the Transparency surface can show *why* something was retrieved.

Measured on real enterprise documents — EnterpriseRAG-Bench, 20,000
documents across Confluence, Slack, Jira and Gmail, 142,520 chunks, with
human-written questions and ground-truth document ids:

| | all 147 questions | the 117 with a gold document |
|---|---|---|
| recall@5 | 0.769 | **0.710** |
| MRR | 0.554 | 0.696 |
| nDCG@5 | 0.539 | — |

**Quote the second column.** Thirty of those questions are
`info_not_found` or `high_level` and have no expected document at all,
so recall is trivially 1.000 for them and they inflate the headline by
six points. Latency p50 88 ms, p95 119 ms over 142k chunks.

Switching the dense channel from the hash placeholder to a real
embedding model moved **ranking, not recall**: MRR 0.501 → 0.554 and
semantic-class MRR 0.289 → 0.389 (+35%), while recall@5 stayed flat at
0.765 → 0.769. The honest reading is that the dense channel is now
ordering the candidate pool much better but still is not *finding*
documents BM25 misses — which is where the next work is, not in the
reranker.

Loading is a separate shape from querying, in two ways.

**Throughput.** `ingest()` handles documents arriving one at a time;
`ingest_many()` batches embedding *across* document boundaries, because
the embedding backend charges per request rather than per text — at a
quarter of a million documents, per-document loading spends days on
round-trips alone. It streams in bounded waves, so peak memory is set by
`INGEST_WAVE` and not by the corpus, and `embed_many()` keeps several
requests in flight at once (ordered, so vectors cannot drift onto the
wrong document) because the backend spends most of a request waiting
rather than computing.

**Failure.** The two callers want opposite things when the embedding
backend is unreachable, so `embed_many(..., strict=)` lets them differ.
A **query** should degrade to the lexical fallback: it affects one
request, and a lexical answer beats an error. An **ingest** must refuse,
because the fallback is a *different embedding space* — 96 dimensions
against the model's 768 — and one substituted batch leaves the corpus
holding two kinds of vector that cannot be compared. The index detects
the mismatch and abandons vector search **for the whole corpus**, so a
single blip during a bulk load silently costs semantic retrieval
everywhere, not just for the affected documents. This is not
hypothetical: it happened during a 20,000-document benchmark load here,
and the only symptom was retrieval quietly getting worse. Ingestion now
retries, then fails loudly; the index logs the mismatch at `error` and
names the likely cause rather than degrading in silence.

### 2. Understand what the organisation *is*
A vector knowledge graph of entities and their relationships.
`agent/enterprise_graph.py`

Entities are embedded, so "the finance team" and "FinOps" resolve to one
node. Edges are **temporal**: asserting a new fact closes the old one
with `valid_to` rather than overwriting it, so "who owned this in March"
has an answer. Scopes compose as `enterprise ∪ user:<email>`.

### 3. Remember the person it works with
Per-user semantic memory, decayed and consolidated.
`agent/vector_memory.py` · `agent/learn_with_you.py`

### 4. Follow the organisation's rules
The best-practice library — **directives, not retrieval**.
`agent/practices.py`

A practice enters the prompt in the T1 directive slice, above retrieved
passages, and a `mandatory` one must be followed and cited. It matches on
**capability** as well as subject, so a practice bound to `email.send`
fires whenever mail is in play however the request was phrased. This is
the difference between "here is a relevant document" and "this is how we
do it here."

### 5. Carry out standing instructions
Durable orders a person writes once, then runs.
`agent/standing_orders.py`

Available three ways: ambient (in every relevant turn), on demand (its
own VM session, streaming), and agent-initiated (as a tool). Bounded by
a **capability ceiling** — the intersection of the order's declared list
with the runner's own ACL. An order can narrow authority, never widen it.
The ceiling is now *enforced* by the VM the order runs in
(`capability_ceiling`), not merely computed and displayed — reading the
run path showed the intersection was calculated, recorded in the run
event, and then a VM was built with the user's full ACL. The same
mechanism bounds swarm workers, so there is one ceiling implementation
to audit, not two.

### 6. Work tasks alongside people
A shared board where Alex is a crew member.
`agent/taskboard.py`

Hand Alex a task and it opens its own VM session, works, streams every
step to the whole crew, and posts to the same thread the humans use.
It lands in `review`, never `done` — Alex does not close its own task.

**Work is plan-first.** Before acting, the run declares 2–7 concrete
steps (`plan_task`), checks them off as it works (`check_plan_step`:
doing / done / skipped-with-reason / blocked-needs-a-human), and may
replace the plan when the situation genuinely changes — the interleaved
plan-act-reflect shape the pattern literature converges on. The plan is
a first-class durable artifact (`media.control.task_plans`,
`agent/task_plan.py`), not prose in a stream:

- the **crew reads the same rows the agent writes**, live, as a
  checklist on the task — so a person steers against the plan, not
  against a guess, and a `blocked` step is a visible request for a
  human decision;
- the plan **frames the next run** after an interruption — continuity
  across process death, alongside the fact-carrying resume note;
- the finish gate is honest about **plan staleness** (the pattern's
  known silent failure): a run that declares victory with steps neither
  done nor deliberately skipped gets the unmet count appended to the
  summary the reviewer reads.

The plan tools bind to the invoking VM's session task and take **no
task-id argument** — a model that could name a task could rewrite
another crew's plan, so the spoof surface does not exist.

**Done is defined by people, verified by the agent.** Acceptance
criteria (`media.control.task_acceptance`, `agent/task_acceptance.py`)
are the crew's bar for the work, distinct from the agent's plan, and the
write asymmetry is the control property: the board API writes and
rewords criteria; the agent's only tool is `check_acceptance`, which
marks one criterion `met` (with the evidence) or `unmet` (with what
stands in the way) — an agent that could edit its own definition of done
would grade its own homework. Criteria render into the run prompt,
verdicts stream live (`acceptance` events), and the finish gate names
every criterion not verified met in the summary the reviewer reads.
Rewording the criteria resets their verdicts, because evidence gathered
against the old wording proves nothing about the new.

**Dependencies are enforced, not drawn.** `media.control.task_deps`
(`agent/task_deps.py`) records that a task waits on others; a run cannot
start while a blocker is unfinished (`TaskBlocked`, surfaced as a 409
naming the blockers), cycles are refused when the closing edge is
proposed, and a blocker still in `review` still blocks — work awaiting a
verdict is not yet a fact its dependents may build on. When the *last*
blocker reaches `done`, the dependent's own thread gets an "Unblocked"
message; while other blockers remain it stays quiet, because "one down,
still blocked" is noise.

**Tasks originate from work signals, gated by a person.**
`agent/email_intel.py` extracts `asks_of_user` from every email through
an injection-proof prompt; `agent/task_origination.py` harvests those
asks into `media.control.task_proposals` — visible, attributed to their
source email, and inert. Only the employee accepting a proposal (a board
action) creates a task (`source='email'`, `source_ref` preserved), so
nothing that arrived in an untrusted email becomes agent-runnable work
without a person reading it first: an attacker who fully controls an
inbound email controls, at most, the text of a suggestion. The harvest
is idempotent on `source_ref`, and a dismissal is durable — a decision,
not a snooze.

**A run survives the process that was driving it.** It is still a thread
in one process, so if that process dies the work stops — but it no
longer stops *silently*, which was the real damage. A live run writes a
heartbeat and checkpoints its steps; on startup, `recover_interrupted_runs`
finds any run whose heartbeat has gone quiet, marks it `interrupted`,
says so in the task thread, and puts the task back on the board. The
crew was told Alex picked the task up, so they have to be told it
stopped; and `in_progress` with nothing progressing is the state that
hides the problem, because it looks exactly like work under way.

Three properties follow from the heartbeat rather than a shutdown hook,
which matters because the case worth handling is the one where there was
no clean shutdown:

- A run with a **fresh** heartbeat is left alone, so recovery never
  disturbs a run another replica is driving.
- A task with a live run **cannot be started twice** — without a check
  against the record, two replicas would each start one and both would
  act.
- A **stale** run does not lock the task out of ever being worked again.

When the work is picked up again, the new attempt is told which tools the
interrupted one already ran, so it checks the current state instead of
blindly repeating side effects. Text the previous attempt generated is
deliberately not quoted back: it was never delivered to anyone, and
replaying it invites the model to treat an undelivered draft as done.

Standing orders get the same treatment (`standing_orders.recover_interrupted_runs`),
sharing the task board's heartbeat interval and staleness window rather
than restating them — two definitions of "stale" that can drift apart is
how one of these paths quietly stops recovering. An order matters more
here, not less: it runs unattended with real capabilities against real
systems, so there is nobody watching a stream to notice it stopped.

Administrators see this. `/api/oversight/runs` reports *live* runs from
this process — that is what in-flight means — but its history comes from
the durable record, because the in-process registry is erased by exactly
the restarts whose consequences most need reviewing. Dropped work gets
its own panel: somebody asked for it and has not been told it failed.

### 7. Work as a swarm
Bounded, named, fully visible delegation. `agent/swarm.py`

An agent holding the `agents.delegate` capability can split a task
across up to **ten** named sub-agents (ten per delegation, ten
concurrent — a hard policy ceiling the environment override cannot
raise). The shape is orchestrator–worker, which published multi-agent
work converges on: each worker gets a **self-contained brief** and a
fresh context, workers never talk to each other, and every decision
about what happens next belongs to the lead agent.

Each worker is a full VM — the same tool registry, the same control
stack — bounded by a **capability ceiling** enforced *inside* the child
VM (`HarnessConfig.capability_ceiling`): parent ∩ requested, minus
delegation always, so the swarm is depth-one by construction and the
concurrency cap cannot be multiplied away by recursion.

Visibility is a first-class output: every worker's thinking, tool
calls, denials and streamed text are forwarded onto the parent's event
stream as named `subagent` events, and the Chat surface renders one
lane per worker — status at a glance, thinking behind progressive
disclosure, failures open by default because a lane that quietly
vanishes reads as "nothing to report" when it means "nobody looked".
Failed and timed-out workers appear in the result set with their
status; a silent hole is never returned.

### 8. Act in the world
A tool VM: 54 built-in tools across 14 capability families — files,
shell, Python, browser, code intelligence, remote systems.
`agent/vm.py` · `agent/workspace_tools.py` · `agent/python_tool.py` ·
`agent/browser_os.py` · `agent/lsp_tool.py` · `agent/mcp_client.py`
— itemised in [`TOOLS.md`](TOOLS.md), generated from the registry.

Those tools are also exported to LangChain and LangGraph
(`agent/langchain_tools.py`), routed through the same governed path, so
an agent built on someone else's framework is governed by ours. See
"Why the agent loop is ours" below.

The browser is a real sandbox with two-phase submit, PII redaction and
origin fencing. MCP connects each user's VM to the organisation's own
MCP servers, with the tool-poisoning and rug-pull defences described
below. The working set is capped at 70 tools because selection accuracy
degrades past roughly that.

Code the agent runs is bounded in **four** dimensions, not just time: a
runtime cap, a memory ceiling, a process-count ceiling, and a whole-tree
kill so a grandchild cannot orphan and keep consuming. Memory is
enforced by the kernel — `setrlimit` on POSIX, a job object on Windows —
so an over-allocation fails at the moment it is requested and comes back
as an ordinary `MemoryError` the agent can read and react to. That
matters more than it sounds: a sampling watchdog was the first attempt,
and a loop allocating 16 MB blocks passed two gigabytes between two
samples. A timeout alone bounds nothing here, because a process can take
the host into swap — and every other user's workspace with it — long
before its sixty seconds are up.

Code the agent runs also gets an **allowlisted environment**, not the
server's. Every other control here — the workspace jail, the runtime
cap, the binary allowlist, the constitution — assumes misbehaving code
cannot take the process's credentials and act somewhere the sandbox does
not reach. Inheriting `os.environ` made that assumption false, because
`print(os.environ)` was enough to read the Databricks token and every
model API key. `python_tool.child_env()` is the allowlist, and it is an
allowlist rather than a denylist because a denylist has to be right
about every secret anyone ever adds. Installing a package runs its build
code, so `pip` goes through the same gate, widened only by its own index
and proxy settings.

### 9. Be governed
Roles, data classification, capability ACLs, an audit ledger.
`agent/access.py` · `agent/guardrails.py`

Identity is verified before authorization ever runs: production and
staging validate the platform's OBO token as **RS256/ES256 against the
issuer's JWKS** (standard OIDC discovery, `ALEX_JWKS_URL` override, key
rotation handled by re-fetch on unknown `kid`). HS256 exists only in
dev and is rejected before any key is resolved, so the classic
alg-confusion attack has nothing to confuse.

One authorization decision point. Scope decides whether you can see a
thing; classification decides whether your clearance reaches it;
capability decides whether you may use the tool. All three fail closed.
**A user's private scope is readable only by that user** — an admin role
does not confer the right to read a colleague's material; administrators
get oversight through the audit ledger instead.

### 10. Be supervised
Bounded execution, an enforced constitution, and an independent review
of what happened.
`agent/execution_governor.py` · `agent/constitution.py` ·
`agent/supervisor.py`

Seven control layers, covered in full in Part 2. What an administrator
actually sees: supervision findings, runs in flight, the execution
budgets in force, and the constitutional record — every action the law
blocked and every one it only noted, separated, because the difference
between those two is whether the action happened.

---

## Part 2 — Architecture

### The control stack

Seven layers. They answer genuinely different questions at different
moments, which is why there are seven and not one policy engine.

| Layer | Question | Module | When |
|---|---|---|---|
| **L0 ACL** | Does this tool exist for you at all? | `vm._allowed_capabilities` | Once, when the VM is built |
| **L1 Prompt** | Does the model *know* the law? | `constitution.build_constitution_prompt` | Every turn, in T0 |
| **L2 Governor** | Is this *run* still behaving like work? | `execution_governor` | Before each tool call |
| **L3 Interceptor** | Is this action permitted? | `constitution.ConstitutionalInterceptor` | Before each tool call |
| **L4 Review** | Is this answer safe to send? | `execution_review` | After a consequential turn |
| **L5 Deploy gate** | Does this build still obey? | `deploy.generate_eval_scenarios` | At promotion |
| **L6 Supervision** | Did anything go wrong? | `supervisor` | After board & standing-order runs |

Two of these are easy to get wrong when reading the code, so they are
worth stating plainly.

**Capability enforcement is L0, not a per-call check.** The VM resolves
the user's ACL once at construction and builds the tool list from it, so
a tool the caller may not use is not merely refused — it is not offered
to the model at all. That is the stronger property, but it means
grep-ping `_invoke_tool` for an ACL check finds nothing and invites the
conclusion that there isn't one.

It also means **this is the only place that decides**, so it has to fail
closed — and it did not. `agent/guardrails.guarded()` is named by four
separate docstrings as the call-time enforcement point and has **no
production call sites at all**, so there was no second line. Meanwhile
the capability lookup returned "do not filter" in two cases that are not
safe: a user with *no* grants, and *any* exception reading the ACL store.
The first offered every tool to the least entitled user; the second meant
a control-plane outage escalated everyone to `run_python` and `vm_exec`.
Both now deny, and the permissive answer survives only for dev with no
ACL table at all, where the alternative is a local agent that can do
nothing. If the call-time gate is ever wanted back, it has to be *called*
— a decorator nobody applies is not defence in depth.

**L6 does not run on chat turns.** Supervision covers task-board runs
and standing orders, where the agent works unattended, plus a sampled
hourly review job. A live chat turn is supervised by the person reading
it.

**The governor and the interceptor are different controls**, which is
why both run before every tool call. The interceptor asks whether an
action is allowed — and every step of a runaway may be individually
allowed. The governor asks about the *shape* of the run, and goes first
so a halted run stops before anything else is evaluated. It detects six
pathologies deterministically, with no model in the loop:

| Pathology | What it catches |
|---|---|
| `budget_*` | Wall-clock, tool-call and turn ceilings |
| `repetition` | The same call repeated with the same arguments |
| `oscillation` | Two states alternating — undo/redo, A→B→A |
| `stagnation` | Turns passing with nothing produced |
| `denial_persistence` | Retrying the thing it was just refused |
| `denial_escalation` | Refused, then reaching for a *stronger* capability |

The last two are the ones that matter most, and they are why the
governor exists separately from the interceptor: an agent that is
refused and then
tries a more powerful route is pursuing its goal past its authority,
and every individual attempt may be permitted. `Budget.for_autonomy()`
tightens the ceilings for `suggest` and loosens them for `autopilot`, so
autonomy preference is a real bound rather than a label.

**Supervision cannot overrule the layers below it.** It runs
deterministic rules first and a model second, and the model can only
*add* findings. A supervisor able to clear a deterministic violation
would be one prompt injection away from being the hole in the fence.

**A control that is wired but empty is not a control.** Both
constitutional layers were in this state, and the suite passed anyway.

*L3, the interceptor*, was in the call path from the beginning — checked
before every tool, with its own `law_block` SSE event and an audit row
for denials — but `make_vm` never passed a constitution, so every VM the
application built fell back to an interceptor with **no articles and a
no-op violation logger**. Two dev/prod schema divergences kept it there:
the dev constitution table omitted `machine_rule` and `severity`, so the
article query failed outright, and the violations table declared
different columns than the insert, so a write would have failed silently
even if one had been attempted.

*L1, the prompt*, was worse, because it is the layer that makes the
others rarely necessary. `context_engine.load_context()` has always
accepted a `constitution_block` and **nothing ever passed one**, so
`build_constitution_prompt()` was reachable only from its own unit test.
The model was being asked to obey a constitution it had never been
shown, and the first it could learn of any article was a tool call
coming back refused — a slow, expensive way to discover a rule it could
simply have been told.

Both now load from the articles in force. They read the *same* loaded
list (`vm._live_constitution` populates it, `vm._constitution_block()`
renders it), so the system cannot tell the model one thing and enforce
another. The tests assert the layers *block*, *record* and *state the
law*, rather than merely that they are called — which is the property
that was missing when all of this passed.

### Knowledge: four stores, deliberately separate

| Store | Holds | Enters the prompt as |
|---|---|---|
| `rag` | Documents the organisation has | Cited passages (T3) |
| `enterprise_graph` | Entities and relationships | Facts (T3) |
| `vector_memory` | What Alex learned about *you* | Recollections (T3) |
| `practices` | How the organisation works | **Directives (T1)** |

They are separate because they answer different questions and carry
different authority. Collapsing them into "one vector store" would put
policy and hearsay at the same rank, which is exactly the failure the
practice library exists to prevent.

### Context assembly

`context_engine.py` builds four slices; `context_fusion.py` assembles T3.

```
T0  persona · constitution · workspace · custom instructions · skills
T1  DIRECTIVES — best practices, then standing orders     ← instructs
T2  session state — open tasks
T3  the turn — question, retrieved passages, facts, memory ← informs
```

T1 above T3 is the load-bearing decision: a practice instructs, a
retrieved document merely informs. T1 is also trimmed **last** when the
budget binds.

T3 is *fused*, not concatenated. Entities resolved from the message
become **anchors** that widen every channel's query, and evidence is
grouped by anchor — so the model sees one subject with several kinds of
evidence rather than three parallel inventories. When the session is
bound to a board task (`task-<id>`), that task is the strongest anchor
and frames the whole turn.

### What actually happens in one turn

The layers above are easier to hold onto in sequence. This is a real
chat turn in which the agent calls one tool and answers.

1. **Identity first.** `app.py::_identity()` verifies the JWT before any
   work; a missing on-behalf-of token ends the request there.
2. **The VM is fetched or built** for `(session, user)`. Building it
   reads the user's ACL and filters the tool registry (**L0**), and
   loads the constitution in force (**L3**).
3. **A fresh governor** is constructed for this turn, with budgets
   scaled to the user's autonomy preference (**L2**).
4. **Context is assembled before the first model call** — retrieval,
   graph lookup and memory recall all run eagerly, not lazily as tools,
   and each emits a `context` event as it returns. This is why the user
   can see what grounded an answer rather than inferring it.
5. **The ReAct loop calls the model.** Text streams to the browser as it
   is generated. Loop-level properties held to the published state of
   the art (OpenCode / Hermes-agent class):
   - the iteration cap comes from the **governor budget**, so autonomy
     is a real control on how long a run may think;
   - a **transient executor failure** that produced nothing is retried
     with backoff (a partial pass never is — replaying can double side
     effects and contradict text already streamed);
   - long histories go through **budgeted compaction**
     (`agent/harness/compaction.py`): the middle is summarized by an
     auxiliary model under a ~20% budget, the task and the working set
     survive verbatim, stale tool payloads are pruned first, and
     repeated compactions chain with generation stamps (lineage), with
     a deterministic digest as the no-model fallback;
   - **plan mode** (the plan/build split) is structural: a plan-mode VM
     exposes no mutating tool to the model at all, withholds every
     remote MCP tool, and denies at call time on the paths that bypass
     exposure. The `plan_task`/`check_plan_step` tools are the one
     exception — writing the plan is the point of the mode;
   - the workspace file loop feeds **diagnostics back on the tool
     result** (Python/JSON/YAML parse errors named with line numbers)
     and every file change is **snapshotted with a repeatable
     `undo_file_change`** — a wrong edit is one call from gone;
   - **budget awareness**: with two passes left after real tool use,
     one harness note tells the model to land the work — the
     context-anxiety failure mode (rushing early, overreaching late)
     addressed by letting the model see where the turn stands;
   - **request observability**: before the first model call, one event
     announces the assembled request's shape — per-slice context sizes,
     total, exposed tool count, plan-mode flag. Sizes, never content.
     "What did the model actually see?" is answerable per turn, in the
     UI.
6. **For each tool call:** governor, then interceptor, then an audit row
   and a `tool` event, then argument validation, then the tool runs.
   On success: governor records it, an audit row with the result, an
   `artifact` event. A denial at either gate emits `law_block` or
   `error` and the tool never runs.
7. **L4 inline review** runs after the loop, if a tool succeeded and the
   tool set was consequential. On a block it replaces the answer.
8. **The turn is persisted**, a durable fact may be captured, and `done`
   closes the stream.

One honest consequence of step 5 preceding step 7: text has already
streamed to the browser by the time inline review runs, so a block
replaces the stored and returned answer but cannot unsay what the user
already watched appear. Review is a stop on the *record and the reply*,
not on the stream.

### Execution

`agent/vm.py` is the single seam. Every caller — the FastAPI routes, the
SDK, the task board, standing orders, tests — goes through it. The VM
owns the executor, the browser OS, the session identity, the
constitution interceptor, the event pump, the audit log, the tool
registry, and the governor.

Tools reach the model from three sources, in this precedence:

1. **Built-in** (`agent/tools/registry.py`), ACL-filtered per user.
2. **Remote MCP** (`agent/mcp_client.py`), namespaced `mcp__server__tool`,
   filling only the headroom under the 70-tool cap.
3. Never both: a remote tool duplicating a built-in name is dropped.

### Why the agent loop is ours — and how to use LangGraph anyway

LangChain and LangGraph are the obvious alternative to `agent/vm.py`,
and the repository used to *claim* we used them: `agent/requirements.txt`
pinned `langgraph` and `databricks-langchain`, the README described "an
MLflow ChatAgent + LangGraph agent", and the walkthrough pointed at a
LangGraph `agent/agent.py`. None of it was true — `agent/agent.py` had
been reduced to a nineteen-line re-export when the VM replaced it, and
nothing had imported either package since.

Two separate questions were tangled up in that, and they have different
answers.

**Should LangGraph drive Alex's loop? No.** Databricks Model Serving
asks for one interface, MLflow's `ResponsesAgent`, and is explicit that
any authoring framework is acceptable behind it — LangGraph is
*supported*, never *required*, so adopting it would not by itself have
deployed anything. Meanwhile six of the seven control layers live at one
seam, `VM._invoke_tool`, and one of them is stronger than a framework
hook could express: capability enforcement happens at VM construction by
*building the tool list from the user's ACL*, so an unauthorised tool is
never offered rather than refused after the model chooses it.
Re-expressing that on someone else's graph runtime is real risk against
no capability we lack.

**Should a LangGraph agent be able to use Alex's tools? Yes** — and it
can, which is what `agent/langchain_tools.py` and
`agent/langgraph_agent.py` are for. A team already building on LangGraph
gets Alex's tools inside their own orchestration:

- `governed_tools(vm)` exports the tools **this principal may use** as
  LangChain `StructuredTool`s.
- `build_agent(vm)` is a `create_react_agent` graph over them, running on
  whatever `alex.llm` is routed to, so the graph does not acquire its own
  model configuration to drift from the rest of the system.

The property that makes this safe is that every exported tool routes
through `VM._invoke_tool`, so a LangGraph node calling `send_email` gets
the ACL, the governor, the constitution and the audit row — the same
things an Alex turn gets. **`spec.callable` is never exported.** The
registry entry is the unguarded function; the VM is the guard. Identity
arguments are stripped before the call, so a caller cannot even express
an impersonation attempt.

Two ungoverned exports used to exist beside this — LangChain factories in
`internal_apis.py` and `web_agentic.py` that wrapped the raw functions,
one of which *wrote* to an internal API. They had no callers and no
tests, and they are gone: an ungoverned export sitting next to a governed
one is a trap, not a convenience.

Building this surfaced a real hole. The governor was only ever created by
`run_turn`, so **every other route into a tool ran unbounded** — the
`/api/tools/{name}` endpoint, the SDK, and a LangGraph node. Those are
runs too, and a caller looping on the invoke endpoint is exactly the
shape the governor exists to stop. `_invoke_tool` now builds one on first
use, because absence of a governor must never mean absence of governance.

**What LangGraph still has that we do not** is durable execution: its
checkpointer persists state at every step, so a run can resume mid-turn.
Alex's runs survive a process death — they are detected, closed out and
restartable — but a restart begins a fresh turn. That remains the honest
limit, and it is in the gaps below.

### MCP trust

A remote server's tool *descriptions* are attacker-controlled text that
lands in the prompt. Four defences, each with an adversarial test:
namespacing (a server cannot shadow `send_email`), description
sanitising (tool poisoning), **definition pinning** (a server that
changes its tools after approval is withheld until a governor
re-approves — the rug-pull), and result truncation (remote output is
unbounded and would otherwise evict the turn's context).

### Surfaces

`app/app.py` + `app/frontend/`. Chat, Board, Guidance, Email, Sessions,
Files, Browser, Tools, Transparency, and **Oversight** (administrators
only, and *hidden* from everyone else — a tab that always 403s teaches
people to ignore permission errors; the hiding is also enforced against
a hand-typed `?view=`, not just the nav).

Two of these carry the transparency requirement, and they answer
different questions:

- **Transparency** is what *you* may see about your own agent: what it
  remembers about you, what it knows, what you are allowed to do, and
  every action it took on your behalf, with denials marked as denials.
- **Oversight** is what an administrator may see across everyone:
  supervision findings, runs in flight, the execution limits in force,
  and the **constitutional record** — every action the law blocked, and
  every one it only noted. The two dispositions are shown separately
  because the difference between them is whether the action happened.

Neither surface exposes another person's content. Administrators get
accountability through these ledgers precisely so they do not need to
read colleagues' material to exercise it.

---

## Part 3 — Decisions, and what is still open

### Deliberate

- **Seven control layers, not one policy engine.** Each asks a different
  question at a different moment. Collapsing them would mean the
  run-shape question — is this still behaving like work? — never gets
  asked at all, because every individual step of a runaway is permitted.
- **Deterministic where the subject is a model.** The governor has no LLM
  in it, and supervision's rules run before its model.
- **A graceful fallback is a decision, not a default.** The same
  substitute that is correct for a query is corrupting for an ingest.
  Any fallback that changes the *kind* of data written — not just its
  quality — has to be refused on the write path, however convenient it
  is on the read path.
- **A schema is defined once and imported.** The task board's three
  tables had no production DDL at all — they existed only in the dev
  seed, so the board worked locally and every write would have failed on
  Databricks with "table not found". The column lists now live in
  `governance/*.sql`, the dev seed mirrors them, and the tests *import*
  the seed's lists rather than restating them: a third copy is how a
  schema change passes its own tests while breaking the application.
  `tests/test_governance_sql.py` also holds the deploy runbook's
  hand-written apply list to the contents of `governance/`, because a
  missing DDL file is invisible until production.
- **A control's test asserts that it acts, not that it runs.** Both
  constitutional layers were fully wired, individually unit-tested, and
  enforcing nothing, and the suite was green throughout. A test that
  proves the interceptor was consulted proves nothing about whether it
  had anything to say. When auditing a layer here, find who supplies its
  input, not who calls it.
- **The dev warehouse fails loudly.** An unparseable `WHERE` raises rather
  than matching every row — silence there was a cross-user data leak — and
  a value expression it cannot evaluate is refused rather than written as
  `NULL`. That second one had been quietly discarding literals: an
  `INSERT ... VALUES (..., 'running', ...)` stored NULL in dev and
  `'running'` in production, and no test noticed because nothing read the
  column back. `INSERT` and `SET` now share one value grammar, so a
  statement cannot be valid in one and rejected in the other.
- **Alex never closes its own task**, and never sends mail without
  confirmation. Both are enforced in code, not prompt. Closing is
  guarded in two places, because one was not enough: the board's run
  flow lands finished work in `review`, and the `update_task_status`
  tool — which writes the same row — refuses `done` on any task Alex has
  a recorded run against, failing closed if it cannot tell.
- **Four knowledge stores, not one vector database.** They answer
  different questions and carry different authority; collapsing them
  would rank policy and hearsay together, which is the exact failure the
  practice library exists to prevent.

### Known gaps

1. **Semantic retrieval is still the weakest class, and a real embedding
   model did not fix it.** On real documents: `constrained` queries
   (identifiers, dates, names) reach recall 0.925, while `semantic`
   queries reach **0.486** and `completeness` 0.439. Moving the dense
   channel off the hash placeholder onto `nomic-embed-text` lifted
   semantic *ranking* sharply (MRR 0.289 → 0.389) and semantic *recall*
   barely at all (0.457 → 0.486). So the dense channel orders well and
   retrieves poorly, and the ceiling is in stage one, not the reranker.
   25 of 147 questions still return nothing at all — mostly policy and
   incident questions whose answers are spread across a thread rather
   than stated in one chunk. Query expansion or a multi-vector
   representation is the honest next step; more reranking is not.
2. **Network egress is still unbounded locally.** Agent-run code is now
   confined to the workspace, capped by a timeout, bounded in memory and
   process count, killed as a whole process tree, and given an
   allowlisted environment — but nothing stops it opening a socket to
   anywhere. That one needs the platform (a firewall, a network
   namespace, the cluster's egress policy) and cannot be done honestly
   from inside the process it is meant to contain. Production has it;
   local does not.
3. **No ANN index anywhere.** Dense search is exact NumPy locally, and
   in production an exact scan of the scope's chunks from the
   `enterprise_docs` pgvector table on Lakebase — fine to roughly 500k
   chunks. There is deliberately no Databricks Vector Search index in
   the architecture; when scale demands ANN, add an ivfflat index to
   `enterprise_docs` with the embedding width pinned to the chosen
   model.
4. **A run still cannot resume mid-turn.** An interrupted run is now
   detected, closed out, and restartable with a note about what the last
   attempt did (below), but the restart begins a fresh turn rather than
   continuing the one that was in flight. Genuine mid-turn resumption
   needs a checkpointing runtime; this is the honest limit of what was
   built.
5. **The serving path is now validated against real MLflow, but not a
   real workspace.** `tests/test_serving_mlflow_contract.py` (gated on
   MLflow being installed) holds every emitted shape to
   `mlflow.types.responses` — the schema the loader wrapper actually
   model-validates — and runs a full `predict` through a typed
   `ResponsesAgentRequest` with `ChatContext` identity. What is still
   unexercised is the workspace itself: `log_model`, the eval gate, and
   the endpoint. The deploy path has since been hardened for exactly
   those first-contact failures — the logged model declares its runtime
   resources (without which serving has no credentials and 401s on the
   first question), the endpoint container is pinned out of dev mode
   (`ALEX_ENV=production` in `environment_vars`), the governance DDL is
   a strictly-ordered bundle job with grants applied last, and the AI
   Gateway config is applied by the deploy rather than described by a
   comment. First deploy is a smaller test than it was, not zero.
6. **The Playwright browser backend is untested against real pages.**
   `agent/browser_os_playwright.py` exists and `browser_os_for_mode()`
   now selects it in production — previously three call sites each
   hardcoded the in-memory simulator, so production would have browsed
   an HTML-string store while reporting success. Production deliberately
   does **not** fall back to the simulator when Playwright is missing:
   a browser that pretends is worse than one that fails at startup with
   the dependency named. What remains untested is the backend against a
   real Chromium and real sites.
7. **`update_task_status` guards `done` by checking for a recorded agent
   run.** That is the right signal today, but it means a task Alex works
   on *without* the board opening a run — a plain chat request to do the
   work — is not covered. The invariant deserves a more direct notion of
   "Alex did this work" than the presence of a run row.

### Verifying any of this

```bash
python -m pytest -q                                   # 1142 tests
python -m ruff check .
python scripts/gen_tools_doc.py                       # regenerate docs/TOOLS.md
python scripts/bench_enterprise.py                    # real documents + qrels
python scripts/bench_retrieval.py --docs 250000       # scale
cd app/frontend && npx vitest run && npm run build    # 79 tests
```
