"""Golden-thread data-pass tests — one tracer through every node.

Each test injects a distinctive tracer at one node's entry boundary and
asserts it emerges, intact and attributed, at the node's exit boundary.
Unit tests prove a node works; these prove the *wiring* works — that a
datum written at one end is the same datum read at the other, with
nothing translated away, defaulted to NULL, or attributed to the wrong
user in between. Every wired-but-empty defect this project has found
(the constitution with no articles, the ceiling that was displayed but
not enforced, the INSERT that wrote NULL for literals) would have failed
a test of exactly this shape.

The tracer is deliberately rare and token-splittable so lexical
retrieval finds it and accidental matches are impossible.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

TRACER = "xilophase-7741-vermeil"
USER = "demo@alex.dev"


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("ALEX_ENV", "dev")
    from alex import config, testing

    config._settings = None
    testing._fakes = None
    from agent import access, taskboard

    access.invalidate()
    taskboard.reset_runs()
    yield
    config._settings = None
    testing._fakes = None
    access.invalidate()
    taskboard.reset_runs()


@pytest.fixture
def client(env):
    """A booted app: the lifespan seed registers every table."""
    from app.app import app

    c = TestClient(app)
    with c:
        tok = c.get(f"/api/dev/token?email={USER}").json()["token"]
        c.headers.update({"X-Forwarded-Access-Token": tok})
        c.get("/api/security/me")  # force the dev seed
        yield c


@pytest.fixture
def sql(client):
    from alex.testing import get_fakes

    wh = get_fakes().warehouse

    def _sql(stmt, params=None):
        res = wh.execute(stmt, params or {})
        as_dicts = getattr(res, "as_dicts", None)
        return as_dicts() if callable(as_dicts) else list(res or [])

    return _sql


def principal():
    from agent.access import Principal

    return Principal(email=USER, role="admin", capabilities=frozenset({
        "knowledge.read", "tasks.read", "tasks.create",
    }))


# ---- node: identity -> access ---------------------------------------------


def test_identity_pass_jwt_claims_to_principal(client):
    """A minted token's email passes through verification into the
    principal the access layer resolves — same casing rules end to end."""
    body = client.get("/api/security/me").json()

    assert body["role"] == "manager"  # the seeded role for the demo user
    assert f"user:{USER}" in body["visible_scopes"]


# ---- node: warehouse ------------------------------------------------------


def test_warehouse_pass_literal_and_param_round_trip(client, sql):
    sql(
        "INSERT INTO media.alex.tasks (id, owner_email, title, status, "
        "priority) VALUES (:i, :o, :t, 'open', 3)",
        {"i": f"t-{TRACER}", "o": USER, "t": f"Ship {TRACER}"},
    )
    row = sql("SELECT * FROM media.alex.tasks WHERE id = :i",
              {"i": f"t-{TRACER}"})[0]

    assert row["title"] == f"Ship {TRACER}"
    assert row["status"] == "open", "the literal was not stored verbatim"
    assert row["priority"] == 3


# ---- node: RAG ingest -> retrieve -> prompt block -------------------------


def test_rag_pass_document_to_cited_prompt_text(client):
    from agent import rag

    rag.ingest(
        None,
        title="Vermeil rollout brief",
        text=f"The rollout codename is {TRACER}. Launch is gated on legal.",
        source_ref="wiki/vermeil",
    )
    hits = rag.retrieve(None, TRACER, k=3, use_graph=False)

    assert hits, "the ingested document did not come back"
    assert TRACER in hits[0].chunk.text
    assert hits[0].chunk.citation()["source_ref"] == "wiki/vermeil"
    block = rag.context_block(hits)
    assert TRACER in block, "retrieval succeeded but the prompt block lost it"


# ---- node: vector memory --------------------------------------------------


def test_memory_pass_remember_to_recall(client):
    from agent.tools import impls

    out = impls.remember(f"Prefers the {TRACER} format", user_email=USER)
    assert out.get("ok", True)

    recalled = impls.recall(TRACER, user_email=USER)
    flat = json.dumps(recalled)
    assert TRACER in flat, "a stored memory did not come back out"


# ---- node: enterprise graph -----------------------------------------------


def test_graph_pass_edge_to_expansion(client):
    from agent import enterprise_graph as eg
    from agent import rag

    rag.ingest(None, title="Vermeil brief",
               text=f"Project {TRACER} is the Q3 launch.", source_ref="a")
    rag.ingest(None, title="Billing runbook",
               text="The billing connector escalation path lives here.",
               source_ref="b")
    eg.upsert_entity(None, f"Project {TRACER}", etype="project")
    eg.upsert_entity(None, "billing connector", etype="system")
    eg.upsert_edge(None, f"Project {TRACER}", "depends_on", "billing connector")

    expanded = rag.retrieve(None, f"Project {TRACER}", k=4, use_graph=True,
                            user_email=USER)

    titles = {h.chunk.title for h in expanded}
    assert "Billing runbook" in titles, (
        "the graph edge did not carry retrieval to the connected document"
    )
    runbook = next(h for h in expanded if h.chunk.title == "Billing runbook")
    assert runbook.via_entities, "the expansion lost its provenance"


# ---- node: context engine (all channels into one prompt) ------------------


def test_context_pass_every_channel_reaches_the_prompt(client, sql):
    from agent import rag
    from agent.context_engine import load_context

    rag.ingest(None, title="Brief",
               text=f"The codename is {TRACER}.", source_ref="wiki/x")

    ctx = load_context(
        system_prompt="You are Alex.",
        user_email=USER,
        session_id="s-pass",
        last_user_msg=f"what is {TRACER}?",
        sql=sql,
        memory_recall=lambda u, limit=8: [("fact", f"memory-{TRACER}")],
        kg_query=lambda u, text: [
            {"src": TRACER, "rel": "owned_by", "dst": "platform"}
        ],
        retrieve=lambda u, text: [
            {"text": h.chunk.text, "citation": h.chunk.citation()}
            for h in rag.retrieve(None, text, k=3, use_graph=False)
        ],
        principal=principal(),
    )
    whole = "\n".join(str(v) for v in ctx.values())

    assert f"memory-{TRACER}" in whole, "the memory channel was dropped"
    assert TRACER in whole, "the retrieval channel was dropped"


# ---- node: practices (directives, not retrieval) --------------------------


def test_practices_pass_published_guidance_reaches_a_match(client, sql):
    from agent import practices

    practices.upsert(
        sql,
        title="Vermeil handling",
        guidance=f"Always route {TRACER} requests through legal first.",
        applies_to=["vermeil", TRACER],
        owner=USER,
    )
    matches = practices.match(sql, principal(), query=f"handle a {TRACER} request")

    flat = json.dumps([m.practice.to_dict() for m in matches], default=str)
    assert TRACER in flat, "a published practice did not match its own subject"


# ---- node: VM turn (message -> tool -> audit -> event stream) -------------


def test_vm_pass_tool_call_to_audit_row_and_event_stream(client, sql):
    from agent.omnigent import MockExecutor
    from agent.vm import make_vm
    from alex.testing import get_fakes

    ex = MockExecutor()
    ex.enqueue_tool_call(
        "create_task",
        {"title": f"Follow up on {TRACER}"},
        follow_up_response=f"Created the {TRACER} task.",
    )
    vm = make_vm(session_id="pass-1", user_email=USER, executor=ex,
                 warehouse=get_fakes().warehouse)
    seen = []
    vm.subscribe_sse(seen.append)

    text, _ = vm.run_turn(text=f"track {TRACER}")

    # 1. The tool's side effect landed in the warehouse.
    rows = sql("SELECT title, owner_email FROM media.alex.tasks", {})
    mine = [r for r in rows if TRACER in str(r.get("title"))]
    assert mine and mine[0]["owner_email"] == USER, (
        "the tool ran but its write is missing or misattributed"
    )
    # 2. The audit ledger recorded the same datum, attributed to the user.
    audits = sql(
        "SELECT tool, input_json, user_email FROM media.alex.agent_actions "
        "WHERE user_email = :u", {"u": USER},
    )
    hit = [a for a in audits if TRACER in str(a.get("input_json"))]
    assert hit and hit[0]["tool"] == "create_task", (
        "the action happened but the audit ledger does not show its data"
    )
    # 3. The person watching saw the same call live.
    tool_events = [e for e in seen if getattr(e, "type", "") == "tool"]
    assert any(TRACER in json.dumps(getattr(e, "args", None) or {})
               for e in tool_events), "the SSE stream lost the tool's data"
    # 4. And the model's answer carried it back to the user.
    assert TRACER in text


# ---- node: task board (task -> run -> thread -> durable steps) ------------


def test_board_pass_task_through_run_to_durable_record(client, sql):
    from agent import taskboard as tb

    class P:
        email = USER
        is_admin = False

    out = tb.create_task(sql, P(), title=f"Reconcile {TRACER}",
                         description="pass test")
    task_id = out["task_id"]

    class EchoVM:
        def __init__(self, **kw):
            self._subs = []

        def subscribe_sse(self, fn):
            self._subs.append(fn)
            return lambda: None

        def run_turn(self, *, text, **_kw):
            assert TRACER in text, "the task brief lost the tracer en route"
            for fn in self._subs:
                fn({"type": "tool", "tool": "search_enterprise",
                    "args": {"q": TRACER}})
            return f"Reconciled {TRACER}: all clear.", []

        def close(self):
            pass

    run = tb.start_run(sql, P(), task_id, background=False,
                       vm_factory=lambda **kw: EchoVM(**kw))

    assert run.status == "done"
    # Thread: the crew reads the same datum the run produced.
    bodies = [m["body"] for m in tb.messages(sql, task_id)]
    assert any(f"Reconciled {TRACER}" in b for b in bodies)
    # Durable steps: the record survives the process.
    row = sql("SELECT steps, summary FROM media.control.task_runs "
              "WHERE run_id = :r", {"r": run.run_id})[0]
    assert TRACER in str(row["steps"]), "the durable step record lost the data"
    assert TRACER in str(row["summary"])
    # Lane: the task went to review, never done.
    lane = sql("SELECT status FROM media.alex.tasks WHERE id = :i",
               {"i": task_id})[0]["status"]
    assert lane == "review"


# ---- node: standing orders (instruction -> prompt -> run record) ----------


def test_orders_pass_instruction_to_run_prompt_and_record(client, sql):
    from agent import standing_orders as so

    order = so.create(
        sql, principal(),
        title="Vermeil sweep",
        instruction=f"Each run, verify the {TRACER} ledger balances.",
        allowed_capabilities=["knowledge.read"],
    )
    prompt = so.build_prompt(order)
    assert TRACER in prompt, "the order's instruction did not reach its prompt"

    class EchoVM:
        def subscribe_sse(self, fn):
            return lambda: None

        def run_turn(self, *, text, **_kw):
            assert TRACER in text
            return f"{TRACER} balanced.", []

        def close(self):
            pass

    run = so.run(sql, principal(), order.order_id, background=False,
                 vm_factory=lambda **kw: EchoVM())
    row = sql("SELECT summary, status FROM media.control.standing_order_runs "
              "WHERE run_id = :r", {"r": run.run_id})[0]
    assert row["status"] == "done"
    assert TRACER in str(row["summary"])


# ---- node: swarm (brief -> worker -> forwarded events -> results) ---------


def test_swarm_pass_brief_through_worker_to_named_result(client):
    from agent import swarm
    from agent.omnigent import MockExecutor
    from agent.vm import make_vm

    parent = make_vm(session_id="lead-pass", user_email=USER,
                     executor=MockExecutor())
    seen = []
    parent.pump.subscribe(seen.append)

    class EchoChild:
        def __init__(self, **kw):
            self._subs = []

        def subscribe_sse(self, fn):
            self._subs.append(fn)
            return lambda: None

        def run_turn(self, *, text, **_kw):
            assert TRACER in text, "the worker's brief lost the tracer"
            for fn in self._subs:
                fn({"type": "text", "text": f"found {TRACER}"})
            return f"report: {TRACER} located", []

        def close(self):
            pass

    out = swarm.run_swarm(
        parent,
        [{"name": "scout", "brief": f"locate every mention of {TRACER}"}],
        vm_factory=lambda **kw: EchoChild(**kw),
    )

    assert out["succeeded"] == 1
    assert TRACER in out["results"][0]["output"]
    lanes = [e for e in seen if getattr(e, "type", "") == "subagent"]
    assert any(e.source == "scout" and e.phase == "text"
               and TRACER in json.dumps(e.body or {}) for e in lanes), (
        "the worker's streamed data never reached the parent's stream"
    )


# ---- node: serving contract (typed request -> items) ----------------------


def test_serving_pass_request_to_final_item(client, monkeypatch):
    from agent import serving

    class FakeVM:
        def subscribe_sse(self, fn):
            self._sub = fn
            return lambda: None

        def run_turn(self, *, text, history=None, **_kw):
            assert TRACER in text
            self._sub({"type": "text", "text": f"the answer is {TRACER}"})
            return f"the answer is {TRACER}", []

        def close(self):
            pass

    monkeypatch.setattr(serving, "make_vm_for_request",
                        lambda *a, **k: FakeVM())
    out = serving.AlexResponsesAgent().predict(
        {"input": [{"role": "user", "content": f"what is {TRACER}?"}]}
    )

    final = out["output"][-1]
    assert TRACER in final["content"][0]["text"]


# ---- node: LangChain bridge (framework arg -> governed VM) ----------------


def test_bridge_pass_langchain_arg_into_the_vm(client, monkeypatch):
    lct = pytest.importorskip("agent.langchain_tools")
    if not lct.langchain_available():
        pytest.skip("langchain-core not installed")
    from agent.omnigent import MockExecutor
    from agent.vm import make_vm

    vm = make_vm(session_id="lc-pass", user_email=USER, executor=MockExecutor())
    seen = {}
    monkeypatch.setattr(
        vm, "_invoke_tool",
        lambda r: seen.update(name=r.name, args=dict(r.args or {})) or {"ok": True},
    )
    tool = next(t for t in lct.governed_tools(vm) if t.name == "create_task")
    tool.invoke({"title": f"Bridge {TRACER}"})

    assert seen["name"] == "create_task"
    assert seen["args"].get("title") == f"Bridge {TRACER}"


# ---- node: supervisor (run events -> findings ledger) ---------------------


def test_supervisor_pass_halt_event_to_findings_row(client, sql):
    from agent import supervisor

    review = supervisor.review(
        [{"type": "governor", "action": "halt",
          "reason": f"looping on {TRACER}"}],
        goal=f"reconcile {TRACER}",
        run_id="run-pass",
        user_email=USER,
        sql=sql,
    )

    assert review.findings, "a governor halt produced no finding"
    rows = sql(
        "SELECT * FROM media.control.supervision_findings", {}
    )
    flat = json.dumps(rows, default=str)
    assert TRACER in flat, "the finding was not written to the ledger"


# ---- node: HTTP surface (API in -> API out) -------------------------------


def test_http_pass_board_create_to_board_read(client):
    r = client.post("/api/board/tasks",
                    json={"title": f"HTTP {TRACER}", "description": "d"})
    assert r.status_code == 200, r.text

    board = client.get("/api/board/tasks").json()
    flat = json.dumps(board)
    assert TRACER in flat, "a task created over HTTP is invisible over HTTP"


def test_http_pass_preference_set_to_get(client):
    """Preferences is a schema'd store, so the pass uses a real key —
    and the invented tracer key must be REFUSED LOUDLY, because a
    setting that appears to save and then reverts is worse than a no."""
    r = client.patch("/api/preferences",
                     json={"theme": "light", TRACER: "anything"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("applied", {}).get("theme") == "light"
    assert TRACER in body.get("refused", {}), (
        "an unknown key was silently dropped instead of refused"
    )

    prefs = client.get("/api/preferences").json()["preferences"]
    assert prefs["theme"] == "light", "an applied preference did not round-trip"


def test_http_pass_audited_action_to_transparency(client, sql):
    from agent.omnigent import MockExecutor
    from agent.vm import make_vm
    from alex.testing import get_fakes

    ex = MockExecutor()
    ex.enqueue_tool_call("list_my_tasks", {}, follow_up_response="none")
    vm = make_vm(session_id="tx-pass", user_email=USER, executor=ex,
                 warehouse=get_fakes().warehouse)
    vm.run_turn(text="my tasks?")

    actions = client.get("/api/transparency/actions").json()["actions"]
    assert any(a.get("tool_name") == "list_my_tasks" for a in actions), (
        "an audited tool call is invisible on the user's own surface"
    )


# ---- node: constitution (article -> block -> violation -> oversight) ------


def test_constitution_pass_article_to_oversight_surface(client, sql):
    """The full loop of the law: an article written to the control plane
    blocks a live tool call, the denial is recorded, and an administrator
    reads that record over HTTP. Four nodes, one datum."""
    from datetime import datetime, timedelta, timezone

    from agent.omnigent import MockExecutor
    from agent.vm import make_vm
    from alex.testing import get_fakes

    sql(
        "INSERT INTO media.control.constitution (article_id, article_type, "
        "title, text, machine_rule, severity, effective_from) "
        "VALUES ('A77', 'prohibition', 'No shell', :tx, :r, 'blocking', :ef)",
        {
            "tx": f"Shell is prohibited during the {TRACER} freeze.",
            "r": '{"deny_capabilities": ["compute.shell"]}',
            "ef": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(sep=" "),
        },
    )
    ex = MockExecutor()
    ex.enqueue_tool_call("vm_exec", {"argv": ["ls"]},
                         follow_up_response="understood")
    vm = make_vm(session_id="law-pass", user_email=USER, executor=ex,
                 warehouse=get_fakes().warehouse)
    seen = []
    vm.subscribe_sse(seen.append)
    vm.run_turn(text="run ls")

    # The person watching saw the block, naming the article.
    blocks = [e for e in seen if getattr(e, "type", "") == "law_block"]
    assert blocks and blocks[0].article == "A77"
    # The record exists...
    rows = sql("SELECT article_id, disposition FROM "
               "media.control.constitution_violations", {})
    assert any(r["article_id"] == "A77" and r["disposition"] == "blocked"
               for r in rows)
    # ...and an administrator can read it over HTTP.
    sql("INSERT INTO media.control.user_roles (user_email, role, team, "
        "granted_scope, clearance) VALUES (:u, 'admin', NULL, NULL, NULL)",
        {"u": USER})
    from agent import access

    access.invalidate(USER)
    body = client.get("/api/oversight/violations").json()
    assert any(v["article_id"] == "A77" for v in body["violations"])


# ---- node: governor (halt reason -> the person watching) ------------------


def test_governor_pass_halt_reason_reaches_the_stream(client):
    from agent.omnigent import MockExecutor, ToolCallRequest
    from agent.vm import make_vm

    vm = make_vm(session_id="gov-pass", user_email=USER,
                 executor=MockExecutor())
    seen = []
    vm.subscribe_sse(seen.append)

    outcomes = [
        vm._invoke_tool(ToolCallRequest(name="list_my_tasks", args={}))
        for _ in range(60)
    ]

    halted = [o for o in outcomes if isinstance(o, dict) and not o.get("ok", True)]
    assert halted, "sixty identical calls were never halted"
    gov = [e for e in seen
           if getattr(e, "type", "") in ("governor", "error")
           and (getattr(e, "reason", None) or getattr(e, "message", None))]
    assert gov, "the run was halted but the person watching was never told why"


# ---- node: sessions (persisted turn -> HTTP resume) -----------------------


def test_sessions_pass_turn_to_http_resume(client, sql):
    sid = client.post("/api/sessions", json={"title": f"About {TRACER}"}).json()[
        "session_id"
    ]
    from agent import session_manager

    session_manager.append_turn(None, sid, 1, "user", f"what is {TRACER}?")
    session_manager.append_turn(None, sid, 2, "assistant", f"{TRACER} is the codename.")

    msgs = client.get(f"/api/sessions/{sid}/messages").json()
    flat = json.dumps(msgs)
    assert f"what is {TRACER}?" in flat
    assert f"{TRACER} is the codename." in flat


# ---- node: tools over HTTP (manifest -> governed invoke) ------------------


def test_tools_pass_manifest_to_governed_http_invoke(client, sql):
    manifest = client.get("/api/tools").json()
    names = {t["name"] for t in manifest.get("tools", manifest.get("manifest", []))}
    assert "delegate_subagents" in names, "the swarm tool is not published"
    assert "create_task" in names

    r = client.post("/api/tools/create_task",
                    json={"args": {"title": f"Via HTTP {TRACER}"}})
    assert r.status_code == 200, r.text

    rows = sql("SELECT title, owner_email FROM media.alex.tasks", {})
    mine = [x for x in rows if TRACER in str(x.get("title"))]
    assert mine and mine[0]["owner_email"] == USER, (
        "the HTTP invoke ran but the write is missing or misattributed"
    )


# ---- NO-pass: boundaries the tracer must NOT cross ------------------------


def test_no_pass_private_document_to_another_user(client):
    """The inverse tests are the ones that matter most: proving the datum
    does NOT emerge where the boundary says it must not."""
    from agent import rag

    rag.ingest(
        None, title="Dana's notes",
        text=f"Private: the {TRACER} salary bands.",
        source_ref="p/1", scope=rag.user_scope("dana@alex.dev"),
    )

    theirs = rag.retrieve(None, TRACER, user_email="intruder@alex.dev", k=5)
    assert all(TRACER not in h.chunk.text for h in theirs), (
        "a private document crossed a user boundary"
    )
    mine = rag.retrieve(None, TRACER, user_email="dana@alex.dev", k=5)
    assert any(TRACER in h.chunk.text for h in mine), (
        "the isolation test is vacuous — the owner cannot see it either"
    )


def test_no_pass_memory_to_another_user(client):
    from agent.tools import impls

    impls.remember(f"Secret plan {TRACER}", user_email="dana@alex.dev")

    theirs = impls.recall(TRACER, user_email="intruder@alex.dev")
    assert TRACER not in json.dumps(theirs), (
        "one user's memory surfaced in another user's recall"
    )
    mine = impls.recall(TRACER, user_email="dana@alex.dev")
    assert TRACER in json.dumps(mine), "vacuous: the owner cannot recall it"


def test_no_pass_board_task_to_a_stranger_over_http(client, env):
    from app.app import app

    client.post("/api/board/tasks", json={"title": f"Mine {TRACER}"})

    with TestClient(app) as other:
        tok = other.get("/api/dev/token?email=stranger@alex.dev").json()["token"]
        board = other.get(
            "/api/board/tasks",
            headers={"X-Forwarded-Access-Token": tok},
        ).json()

    assert TRACER not in json.dumps(board), (
        "a task crossed the visibility boundary over HTTP"
    )


def test_no_pass_session_history_to_a_stranger_over_http(client, sql):
    sid = client.post("/api/sessions", json={"title": "mine"}).json()["session_id"]
    from agent import session_manager

    session_manager.append_turn(None, sid, 1, "user", f"secret {TRACER}")

    from app.app import app

    with TestClient(app) as other:
        tok = other.get("/api/dev/token?email=stranger@alex.dev").json()["token"]
        r = other.get(
            f"/api/sessions/{sid}/messages",
            headers={"X-Forwarded-Access-Token": tok},
        )

    assert TRACER not in r.text, (
        "a session id alone was enough to read another user's conversation"
    )
