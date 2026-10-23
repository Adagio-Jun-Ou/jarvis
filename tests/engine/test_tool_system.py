"""
Tests for the completed tool/skills/learning system:

* registry <-> TOOL_CAPABILITY invariant (both directions)
* a memory tool call through the VM (guardrails+constitution path)
* /api/teach, /api/feedback
* /api/tools MCP manifest ACL filtering
* visible-reasoning digest archival + re-injection
* skill-hint wiring into the VM prompt assembly
"""
from __future__ import annotations

import pytest

from agent.omnigent import MockExecutor, ToolCallRequest
from agent.vm import make_vm


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("ALEX_ENV", "dev")
    from alex import config, testing
    config._settings = None
    testing._fakes = None
    yield
    config._settings = None
    testing._fakes = None


# ---------------------------------------------------------------------------
# 1. Registry / capability invariant
# ---------------------------------------------------------------------------

# Tool names that are intentionally in TOOL_CAPABILITY but NOT in the
# registry: legacy UC-function tool names, aliases for the pre-Browser-OS
# tool surface, and internal/not-yet-implemented tools. Each stays in the
# capability map so the guardrails gate keeps resolving old audit rows
# and UC function wrappers, but none is exposed to the LLM.
CAPABILITY_ONLY_ALLOWLIST = {
    "draft_email_reply",   # UC function tool (media.alex.*), not a local impl
    "send_email",          # consequential; requires the send pipeline
    "publish_post",        # consequential; requires the publish pipeline
    "check_rights",        # UC function tool
    "log_agent_action",    # agent-internal (AGENT_INTERNAL_TOOLS)
    "search_media_library",  # UC function tool
    "audience_analytics",  # UC function tool
    "browser_open",        # legacy pre-Browser-OS alias
    "browser_read",        # legacy pre-Browser-OS alias
    "browser_fill",        # legacy pre-Browser-OS alias
    "trigger_workflow",    # workflow engine not yet exposed as a tool
    "stage_action",        # staging surface is app-internal
}


def test_registry_and_capability_map_are_mutually_complete(env):
    from agent.guardrails import TOOL_CAPABILITY
    from agent.tools.registry import TOOL_REGISTRY

    # Direction 1: every registered tool has a capability entry and the
    # registry's capability matches the guardrails map exactly.
    for name, spec in TOOL_REGISTRY.items():
        assert name in TOOL_CAPABILITY, f"{name} missing from TOOL_CAPABILITY"
        assert TOOL_CAPABILITY[name] == spec.capability, (
            f"{name}: registry says {spec.capability!r}, "
            f"guardrails say {TOOL_CAPABILITY[name]!r}"
        )
        assert spec.description, f"{name} has no LLM-facing description"

    # Direction 2: every capability-mapped tool is registered or
    # explicitly allowlisted above.
    missing = set(TOOL_CAPABILITY) - set(TOOL_REGISTRY) - CAPABILITY_ONLY_ALLOWLIST
    assert not missing, f"tools in TOOL_CAPABILITY but not registered: {missing}"

    # The allowlist itself must not rot: no entry may be both
    # allowlisted and registered.
    overlap = CAPABILITY_ONLY_ALLOWLIST & set(TOOL_REGISTRY)
    assert not overlap, f"allowlisted tools are now registered: {overlap}"


# ---------------------------------------------------------------------------
# 2. Memory tool call through the VM
# ---------------------------------------------------------------------------


def test_memory_tools_roundtrip_through_vm(env):
    vm = make_vm(
        session_id="s-mem",
        user_email="alice@x.com",
        executor=MockExecutor(),
    )
    stored = vm._invoke_tool(
        ToolCallRequest(
            name="remember",
            args={"content": "Prefers Friday demos", "kind": "preference"},
        )
    )
    assert stored["ok"] is True
    assert stored["owner_email"] == "alice@x.com"

    recalled = vm._invoke_tool(
        ToolCallRequest(name="recall", args={"query": "demos", "k": 3})
    )
    assert recalled["ok"] is True
    contents = [m["content"] for m in recalled["memories"]]
    assert "Prefers Friday demos" in contents

    # The call was audit-logged through the same path as browser tools.
    rows = vm._session.audit_log._wh.execute(
        "SELECT tool, capability, outcome FROM media.alex.agent_actions"
    ).as_dicts()
    remembered = [r for r in rows if r["tool"] == "remember"]
    assert remembered and remembered[-1]["outcome"] == "ok"
    assert remembered[-1]["capability"] == "memory.write"


def test_constitution_denies_memory_tool_through_vm(env):
    from agent.constitution import ConstitutionalInterceptor
    from agent.vm import VM, HarnessConfig

    interceptor = ConstitutionalInterceptor(
        articles=[
            {
                "article_id": "A77",
                "title": "No memory writes",
                "severity": "blocking",
                "rule": {"deny_capabilities": ["remember"]},
            }
        ],
        log_violation=lambda *a, **k: None,
        session_id="s-mem2",
        user_email="alice@x.com",
    )
    vm = VM(
        session_id="s-mem2",
        user_email="alice@x.com",
        config=HarnessConfig(executor=MockExecutor(), constitution=interceptor),
    )
    result = vm._invoke_tool(
        ToolCallRequest(name="remember", args={"content": "blocked"})
    )
    assert result["ok"] is False


# ---------------------------------------------------------------------------
# 3/4. /api/teach and /api/feedback
# ---------------------------------------------------------------------------


def _client_and_token(email="demo@alex.dev"):
    from fastapi.testclient import TestClient

    from alex.identity import mint_dev_token
    from app import app as appmod

    tok = mint_dev_token(email, scopes=("serving.serving-endpoints", "sql"))
    return TestClient(appmod.app), {"Authorization": f"Bearer {tok}"}


def test_api_teach_stores_taught_routine(env):
    client, headers = _client_and_token()
    with client:
        r = client.post(
            "/api/teach",
            headers=headers,
            json={"prompt": "To file expenses: export, tag, submit by Friday."},
        )
        assert r.status_code == 200
        assert r.json()["routine_id"]

        # Empty prompt is rejected.
        r2 = client.post("/api/teach", headers=headers, json={"prompt": ""})
        assert r2.status_code == 400

    from alex.testing import get_fakes

    rows = get_fakes().warehouse.execute(
        "SELECT owner_email, kind, content FROM media.control.vector_memory_mirror"
    ).as_dicts()
    taught = [r for r in rows if r["kind"] == "taught_routine"]
    assert taught and "file expenses" in taught[-1]["content"]
    assert taught[-1]["owner_email"] == "demo@alex.dev"


def test_api_feedback_records_thumbs(env):
    client, headers = _client_and_token()
    with client:
        r = client.post(
            "/api/feedback",
            headers=headers,
            json={"suggestion_id": "sug-1", "feedback": "helpful", "comment": "nice"},
        )
        assert r.status_code == 200
        assert r.json() == {"status": "recorded"}

        # Invalid enum rejected.
        r2 = client.post(
            "/api/feedback",
            headers=headers,
            json={"suggestion_id": "sug-1", "feedback": "meh"},
        )
        assert r2.status_code == 400

    from alex.testing import get_fakes

    rows = get_fakes().warehouse.execute(
        "SELECT user_email, suggestion_id, feedback FROM media.control.suggestion_feedback"
    ).as_dicts()
    assert rows and rows[-1]["suggestion_id"] == "sug-1"
    assert rows[-1]["feedback"] == "helpful"
    assert rows[-1]["user_email"] == "demo@alex.dev"


def test_chat_edited_output_captures_correction(env):
    client, headers = _client_and_token("alice@x.com")
    from alex.identity import mint_dev_token

    obo = mint_dev_token("alice@x.com")
    with client:
        r = client.post(
            "/api/chat",
            headers={**headers, "X-Forwarded-Access-Token": obo},
            json={
                "session_id": "s-edit",
                "text": "thanks",
                "custom_inputs": {
                    "original_output": "Dear team,",
                    "edited_output": "Hi all — quick update:\nDear team,",
                },
            },
        )
        assert r.status_code == 200

    from alex.testing import get_fakes

    rows = get_fakes().warehouse.execute(
        "SELECT owner_email, kind, content FROM media.control.vector_memory_mirror"
    ).as_dicts()
    prefs = [
        r
        for r in rows
        if r["kind"] == "preference" and r["owner_email"] == "alice@x.com"
    ]
    assert prefs, "edited_output should be captured as a preference memory"


# ---------------------------------------------------------------------------
# 5. /api/tools — MCP manifest, ACL-filtered
# ---------------------------------------------------------------------------


def test_api_tools_filters_by_effective_acl(env):
    client, headers = _client_and_token("bob@x.com")
    with client:
        # bob has no ACL rows: the manifest is empty.
        r = client.get("/api/tools", headers=headers)
        assert r.status_code == 200
        assert r.json()["tools"] == []

        # Grant bob memory.read; every memory.read tool appears, and
        # remember / graph_link (memory.write) do not.
        from alex.testing import get_fakes

        get_fakes().warehouse.execute(
            "INSERT INTO media.control.effective_employee_acl "
            "(user_email, capability, expires_at) VALUES (:u, :c, NULL)",
            {"u": "bob@x.com", "c": "memory.read"},
        )
        r2 = client.get("/api/tools", headers=headers)
        names = {t["name"] for t in r2.json()["tools"]}
        assert names == {
            "recall",
            "graph_query",
            "graph_entity",
            "graph_neighbors",
            "graph_about_me",
        }
        assert "remember" not in names and "graph_link" not in names

        # MCP shape: name/description/inputSchema; identity kwargs are
        # never exposed in the schema.
        recall_tool = next(t for t in r2.json()["tools"] if t["name"] == "recall")
        assert recall_tool["description"]
        schema = recall_tool["inputSchema"]
        assert schema["type"] == "object"
        assert "query" in schema["properties"]
        assert "user_email" not in schema["properties"]
        assert "instance_scope" not in schema["properties"]

        # Unauthenticated access is rejected.
        r3 = client.get("/api/tools")
        assert r3.status_code == 401


def test_input_schema_derivation_types(env):
    from agent.tools.registry import TOOL_REGISTRY, input_schema_for

    schema = input_schema_for(TOOL_REGISTRY["add_edge"])
    props = schema["properties"]
    assert props["src"]["type"] == "string"
    assert props["confidence"]["type"] == "number"
    assert set(schema["required"]) == {"src", "rel", "dst"}

    bash_schema = input_schema_for(TOOL_REGISTRY["bash"])
    assert bash_schema["properties"]["argv"]["type"] == "array"


# ---------------------------------------------------------------------------
# 6. Visible reasoning: SSE events, archive, digest re-injection
# ---------------------------------------------------------------------------


def test_reasoning_digest_archived_and_reinjected(env):
    from alex.testing import get_fakes

    ex = MockExecutor()
    ex.enqueue_response("Decision: approve the Q1 budget after CFO sign-off.")
    vm = make_vm(session_id="s-think", user_email="alice@x.com", executor=ex)
    seen: list = []
    vm.subscribe_sse(seen.append)

    final, _ = vm.run_turn(text="What should I do about the budget?",
                           show_reasoning=True)
    assert "Q1 budget" in final

    # Reasoning SSE events were emitted (plan + conclude).
    phases = [e.phase for e in seen if e.type == "reasoning"]
    assert "plan" in phases and "conclude" in phases

    # The full trace was archived to media.control.thinking_traces.
    rows = get_fakes().warehouse.execute(
        "SELECT user_email, session_id, phase, raw_thinking, digest "
        "FROM media.control.thinking_traces"
    ).as_dicts()
    assert rows and rows[-1]["user_email"] == "alice@x.com"
    assert "Q1 budget" in rows[-1]["raw_thinking"]

    # Only the digest (<=320 chars ~= 80 tokens) is kept for context.
    digest = vm._session.last_reasoning_digest
    assert digest and len(digest) <= 320

    # The next turn's prompt re-injects the digest, not the raw trace.
    prompt = vm._build_system_prompt("follow up")
    assert "PRIOR TURN DECISION DIGEST" in prompt
    assert digest[:50] in prompt
    assert "history_len" not in prompt  # raw trace never re-enters context


def test_reasoning_off_by_default(env):
    ex = MockExecutor()
    ex.enqueue_response("plain answer")
    vm = make_vm(session_id="s-plain", user_email="alice@x.com", executor=ex)
    seen: list = []
    vm.subscribe_sse(seen.append)
    vm.run_turn(text="hello")
    assert not [e for e in seen if e.type == "reasoning"]
    assert vm._session.last_reasoning_digest == ""


# ---------------------------------------------------------------------------
# Skill hints wired into prompt assembly
# ---------------------------------------------------------------------------


def test_skill_hints_flow_into_system_prompt(env, monkeypatch):
    import agent.learning_loop as ll

    def fake_hints(sql, owner_email, last_user_msg, **kw):
        assert owner_email == "alice@x.com"
        return "LEARNED PROCEDURES (follow these; you ratified them):\n- Triage: rank, draft, file"

    monkeypatch.setattr(ll, "active_skill_hints", fake_hints)
    vm = make_vm(session_id="s-skill", user_email="alice@x.com",
                 executor=MockExecutor())
    prompt = vm._build_system_prompt("triage my inbox")
    assert "LEARNED PROCEDURES" in prompt
    assert "Triage: rank, draft, file" in prompt


def test_invoke_tool_endpoint_runs_governed_tool(env):
    """POST /api/tools/{name} executes a governed tool directly (the MCP
    call-tool seam) with ACL + audit, and rejects unknown/forbidden."""
    client, headers = _client_and_token()
    hdr = {**headers, "X-Forwarded-Access-Token": headers["Authorization"].split()[1]}

    # A permitted tool runs and returns a result.
    r = client.post("/api/tools/remember", headers=hdr,
                    json={"args": {"content": "loop test note"}})
    assert r.status_code == 200
    assert r.json()["tool"] == "remember"
    assert r.json()["result"].get("ok") is True

    # Unknown tool -> 404.
    assert client.post("/api/tools/no_such_tool", headers=hdr, json={}).status_code == 404


def test_invoke_tool_endpoint_forbids_uncapable_user(env):
    from fastapi.testclient import TestClient

    from alex.identity import mint_dev_token
    from app import app as appmod

    # A user with no seeded ACL cannot invoke a capability-gated tool.
    tok = mint_dev_token("stranger@x.com", scopes=("sql",))
    hdr = {"Authorization": f"Bearer {tok}", "X-Forwarded-Access-Token": tok}
    client = TestClient(appmod.app)
    r = client.post("/api/tools/remember", headers=hdr, json={"args": {"content": "x"}})
    assert r.status_code == 403


def test_remember_reflects_into_transparency_mirror(env):
    # A just-remembered fact must appear in the read model that the
    # "What my AI knows" tab reads (media.control.vector_memory_mirror),
    # not only in the vector store — otherwise it is invisible to the user.
    from alex.testing import get_fakes
    from app._dev_seed import seed_dev_warehouse

    wh = get_fakes().warehouse
    seed_dev_warehouse(wh)
    from agent.tools import impls
    r = impls.remember("User prefers dark mode", kind="preference",
                        user_email="demo@alex.dev")
    assert r["ok"]
    rows = wh.execute(
        "SELECT id, content FROM media.control.vector_memory_mirror "
        "WHERE owner_email = :u AND superseded_by IS NULL",
        {"u": "demo@alex.dev"},
    ).as_dicts()
    assert any(m["id"] == r["id"] and m["content"] == "User prefers dark mode"
               for m in rows)
