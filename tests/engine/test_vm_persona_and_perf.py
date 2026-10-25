"""
Tests for the persona/guardrail-UX/knowledge-loop and the VM/context
performance work:

* the crafted persona is the AgentDef prompt and lands in the system
  prompt the context engine assembles;
* tool denials come back to the model as the standardized denial
  payload (constitution and ACL paths);
* the learned-fact loop captures durable facts, emits a ``memory``
  SSE event, stores through vector memory, and respects forget
  requests;
* the app's dev VM cache reuses one VM per (session, user), evicts
  by LRU, and closes evicted VMs;
* the context engine's total budget is env-configurable and trims
  lowest-priority slices first (T0 is never trimmed).
"""
from __future__ import annotations

import pytest

from agent.omnigent import MockExecutor, ToolCallRequest
from agent.persona import PERSONA, extract_durable_fact
from agent.vm import VM, make_vm


@pytest.fixture
def vm() -> VM:
    return make_vm(
        session_id="s-persona",
        user_email="alice@x.com",
        instance_scope="default",
        executor=MockExecutor(),
    )


# -- persona ----------------------------------------------------------------


def test_agent_def_prompt_is_the_persona(vm: VM) -> None:
    assert vm.agent_def.prompt == PERSONA
    assert "proactive enterprise copilot" in vm.agent_def.prompt
    assert "Never fabricate data" in vm.agent_def.prompt


def test_persona_lands_in_built_system_prompt(vm: VM) -> None:
    prompt = vm._build_system_prompt("hello")
    assert "proactive enterprise copilot" in prompt


def test_persona_stays_tight() -> None:
    # Local models pay latency per token; the persona must stay small.
    assert len(PERSONA.split()) < 250


# -- denial payload ---------------------------------------------------------


class _DenyPolicy:
    def callable(self, event):  # noqa: A003
        return {"action": "deny", "reason": "A7 forbids it", "article_id": "A7"}


def test_constitution_denial_returns_standard_payload(vm: VM) -> None:
    vm.agent_def.policies["constitution"] = _DenyPolicy()
    res = vm._invoke_tool(ToolCallRequest(name="recall", args={"query": "x"}))
    assert res["ok"] is False
    assert res["denied"] is True
    assert res["article"] == "A7"
    assert res["reason"] == "A7 forbids it"
    assert "how_to_proceed" in res and res["how_to_proceed"]


def test_acl_denied_returns_standard_payload(vm: VM, monkeypatch) -> None:
    import dataclasses

    from agent.tools import TOOL_REGISTRY
    from alex.errors import Denied

    spec = TOOL_REGISTRY["recall"]

    def _deny(**kwargs):
        raise Denied("alice lacks memory.read")

    monkeypatch.setitem(
        TOOL_REGISTRY, "recall", dataclasses.replace(spec, callable=_deny)
    )
    res = vm._invoke_tool(ToolCallRequest(name="recall", args={"query": "x"}))
    assert res["ok"] is False and res["denied"] is True
    assert "memory.read" in res["how_to_proceed"]
    assert "retry" in res["how_to_proceed"].lower() or "Do not retry" in res[
        "how_to_proceed"
    ]


# -- learned-fact capture ---------------------------------------------------


def test_extract_durable_fact_heuristics() -> None:
    assert extract_durable_fact("Please remember that the launch is May 5") == (
        "fact",
        "the launch is May 5",
    )
    assert extract_durable_fact("I prefer bullet points over prose.") == (
        "preference",
        "bullet points over prose",
    )
    kind, content = extract_durable_fact("my manager is Dana Chen")
    assert kind == "fact" and "manager" in content and "Dana Chen" in content
    assert extract_durable_fact("what's the weather?") is None
    # Forget requests suppress capture.
    assert extract_durable_fact("forget that my manager is Dana Chen") is None


def test_run_turn_captures_fact_with_memory_event(vm: VM) -> None:
    seen: list = []
    vm.subscribe_sse(seen.append)
    vm.run_turn(text="Remember that our launch date is May 5")
    mem = [e for e in seen if e.type == "memory"]
    assert len(mem) == 1
    assert "launch date is May 5" in mem[0].text
    assert mem[0].extras.get("learned") is True
    # Stored through vector memory so future recalls surface it.
    from agent import vector_memory
    from agent.memory import connection

    with connection() as conn:
        beads = vector_memory.recall(conn, "alice@x.com", k=5)
    assert any("launch date is May 5" in b.content for b in beads)
    # Audit-visible: a memory_capture row exists.
    from alex.testing import get_fakes

    rows = (
        get_fakes()
        .warehouse.execute(
            "SELECT tool, outcome FROM media.alex.agent_actions", {}
        )
        .as_dicts()
    )
    assert any(r.get("tool") == "memory_capture" for r in rows)


def test_run_turn_respects_forget_requests(vm: VM) -> None:
    seen: list = []
    vm.subscribe_sse(seen.append)
    vm.run_turn(text="forget my manager is Dana Chen")
    assert not [e for e in seen if e.type == "memory"]


# -- VM cache (app dev chat path) ------------------------------------------


def test_vm_cache_reuses_same_vm_per_session(monkeypatch) -> None:
    from app import app as app_mod

    app_mod._vm_cache_clear()
    a = app_mod._vm_cache_get("sess-1", "alice@x.com", "default")
    b = app_mod._vm_cache_get("sess-1", "alice@x.com", "default")
    assert a is b
    c = app_mod._vm_cache_get("sess-2", "alice@x.com", "default")
    assert c is not a
    app_mod._vm_cache_clear()


def test_vm_cache_lru_eviction_closes_vm(monkeypatch) -> None:
    from app import app as app_mod

    app_mod._vm_cache_clear()
    monkeypatch.setattr(app_mod, "_VM_CACHE_MAX", 1)
    a = app_mod._vm_cache_get("sess-1", "alice@x.com", "default")
    b = app_mod._vm_cache_get("sess-2", "alice@x.com", "default")
    assert b is not a
    assert a._closed is True  # evicted VMs are close()d
    assert b._closed is False
    app_mod._vm_cache_clear()


def test_vm_cache_ttl_expiry(monkeypatch) -> None:
    from app import app as app_mod

    app_mod._vm_cache_clear()
    a = app_mod._vm_cache_get("sess-ttl", "alice@x.com", "default")
    # Age the entry past the TTL by hand.
    key = ("sess-ttl", "alice@x.com", False)
    vm, _ts = app_mod._VM_CACHE[key]
    app_mod._VM_CACHE[key] = (vm, -1e9)
    b = app_mod._vm_cache_get("sess-ttl", "alice@x.com", "default")
    assert b is not a
    assert a._closed is True
    app_mod._vm_cache_clear()


# -- context budget ---------------------------------------------------------


def _noisy_sql(query: str, parameters=None):
    if "standing_orders" in query:
        return [
            {
                "order_id": f"SO-{i}",
                "title": "t" * 50,
                "instruction": "x" * 300,
            }
            for i in range(5)
        ]
    if "tasks" in query:
        return [
            {"id": i, "title": "y" * 120, "status": "open", "priority": 1}
            for i in range(8)
        ]
    return []


def test_context_budget_env_trims_low_priority_slices(monkeypatch) -> None:
    from agent import context_engine

    monkeypatch.setenv("ALEX_CONTEXT_BUDGET_CHARS", "1500")
    persona = "P" * 900
    ctx = context_engine.load_context(
        system_prompt=persona,
        user_email="alice@x.com",
        session_id="s",
        last_user_msg="m" * 500,
        sql=_noisy_sql,
        memory_recall=lambda u, limit=5: [("fact", "z" * 190)] * 5,
        kg_query=lambda u, t: [],
    )
    # T0 (persona) is never trimmed by the total budget.
    assert ctx["T0"].startswith(persona)
    # Total respects the configured budget.
    assert ctx["total"] <= 1500
    # T3 (lowest priority) was trimmed/dropped before T1.
    assert len(ctx["T3"]) < 1000


def test_context_budget_default_when_env_unset(monkeypatch) -> None:
    from agent import context_engine

    monkeypatch.delenv("ALEX_CONTEXT_BUDGET_CHARS", raising=False)
    assert context_engine._total_budget() == 6000


# -- history pruning --------------------------------------------------------


def test_history_pruned_to_env_cap_with_digest(monkeypatch) -> None:
    from agent.vm import _prune_history

    monkeypatch.setenv("ALEX_HISTORY_MAX_MSGS", "4")
    history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
        for i in range(10)
    ]
    pruned = _prune_history(history)
    assert len(pruned) == 5  # digest + last 4
    assert pruned[0]["role"] == "system"
    assert "DIGEST" in pruned[0]["content"]
    assert "msg 0" in pruned[0]["content"]
    assert pruned[-1]["content"] == "msg 9"


def test_history_untouched_under_cap(monkeypatch) -> None:
    from agent.vm import _prune_history

    monkeypatch.setenv("ALEX_HISTORY_MAX_MSGS", "20")
    history = [{"role": "user", "content": "hi"}]
    assert _prune_history(history) is history


def test_repeated_prunes_carry_the_old_digest_forward(monkeypatch) -> None:
    """The digest head promises 'decisions below still stand'. Re-pruning
    used to keep only the banner's first line, so every bullet from the
    previous digest vanished one turn later."""
    from agent.vm import _prune_history

    monkeypatch.setenv("ALEX_HISTORY_MAX_MSGS", "4")
    history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
        for i in range(10)
    ]
    first = _prune_history(history)
    assert "msg 0" in first[0]["content"]

    # Keep chatting past the cap again: the old digest is now inside
    # the pruned region of the second call.
    grown = first + [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"more {i}"}
        for i in range(6)
    ]
    second = _prune_history(grown)
    digest = second[0]["content"]
    assert "msg 0" in digest, (
        "the previous digest's decisions did not survive the second prune"
    )


# -- ACL cache --------------------------------------------------------------


def test_acl_lookup_is_cached_per_user(monkeypatch) -> None:
    from agent import vm as vm_mod
    from alex.testing import get_fakes

    vm_mod._acl_cache_clear()
    wh = get_fakes().warehouse
    wh.register(
        "media.control.effective_employee_acl", ["user_email", "capability"]
    )
    wh.execute(
        "INSERT INTO media.control.effective_employee_acl "
        "(user_email, capability) VALUES (:u, :c)",
        {"u": "bob@x.com", "c": "memory.read"},
    )
    v = make_vm(
        session_id="s-acl",
        user_email="bob@x.com",
        executor=MockExecutor(),
    )
    # Construction warmed the cache; clear it so the first lookup
    # below hits the warehouse and the second is served from cache.
    vm_mod._acl_cache_clear()
    calls = {"n": 0}
    orig = wh.execute

    def counting(query, parameters=None, **kw):
        if "effective_employee_acl" in query:
            calls["n"] += 1
        return orig(query, parameters, **kw)

    monkeypatch.setattr(wh, "execute", counting)
    assert v._allowed_capabilities() == {"memory.read"}
    assert v._allowed_capabilities() == {"memory.read"}
    assert calls["n"] == 1  # second lookup served from the cache
    vm_mod._acl_cache_clear()
