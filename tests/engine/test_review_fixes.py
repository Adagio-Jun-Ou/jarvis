"""
Tests for the code-review fixes: VM constitution enforcement, the
bridge's multi-tool / turn-cap / history behaviour, Databricks
streaming tool-call accumulation, session branching SQL, the
context-engine wiring in VM.run_turn, and the audit-log columns.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest

from agent.constitution import ConstitutionalInterceptor
from agent.omnigent import MockExecutor, ToolCallRequest
from agent.omnigent._alex_bridge import run_turn as bridge_run_turn
from agent.omnigent.executor import (
    Executor,
    TurnComplete,
)
from agent.vm import VM, HarnessConfig, make_vm
from alex.errors import ConstitutionalBlock


def _make_vm(**kwargs) -> VM:
    return make_vm(
        session_id="s1",
        user_email="alice@x.com",
        instance_scope="default",
        executor=kwargs.pop("executor", MockExecutor()),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Finding 1: constitution enforcement in the VM tool path
# ---------------------------------------------------------------------------


def test_blocking_article_denies_tool_call_through_vm():
    violations: list = []
    interceptor = ConstitutionalInterceptor(
        articles=[
            {
                "article_id": "A99",
                "title": "No navigation",
                "severity": "blocking",
                "rule": {"deny_capabilities": ["browser_navigate"]},
            }
        ],
        log_violation=lambda *a, **k: violations.append((a, k)),
        session_id="s1",
        user_email="alice@x.com",
    )
    vm = VM(
        session_id="s1",
        user_email="alice@x.com",
        config=HarnessConfig(executor=MockExecutor(), constitution=interceptor),
    )
    seen: list = []
    vm.subscribe_sse(seen.append)
    s = vm.open_browser_session()
    t = vm.open_browser_tab(s.session_id, url="alex://welcome")
    result = vm._invoke_tool(
        ToolCallRequest(
            name="browser_navigate",
            args={"session_id": s.session_id, "tab_id": t.tab_id, "url": "alex://x"},
        )
    )
    assert result["ok"] is False
    # An error event was emitted for the denial.
    assert any(e.type == "error" for e in seen)
    # An audit row records the denial.
    rows = vm._session.audit_log._wh.execute(
        "SELECT tool, outcome, ok FROM media.alex.agent_actions"
    ).as_dicts()
    denied = [r for r in rows if r["tool"] == "browser_navigate"]
    assert denied and denied[-1]["outcome"] == "denied"
    assert denied[-1]["ok"] == 0
    # The tool never ran: no 'started'/'ok' rows for it.
    assert all(r["outcome"] == "denied" for r in denied)


def test_allowed_tool_still_runs_through_vm():
    vm = _make_vm()
    s = vm.open_browser_session()
    t = vm.open_browser_tab(s.session_id, url="alex://welcome")
    result = vm._invoke_tool(
        ToolCallRequest(
            name="browser_navigate",
            args={
                "session_id": s.session_id,
                "tab_id": t.tab_id,
                "url": "alex://welcome",
            },
        )
    )
    assert result["ok"] is True


# ---------------------------------------------------------------------------
# Findings 2/4/5/6: the bridge's ReAct loop
# ---------------------------------------------------------------------------


def _agent_def():
    from agent.omnigent._alex_bridge import build_agent_def

    return build_agent_def(
        name="t",
        prompt="You are a test agent.",
        tools={"echo": lambda **kw: kw},
        capabilities={"echo": "alex:tool:echo"},
    )


def test_bridge_returns_tuple_on_constitutional_block():
    ex = MockExecutor()
    ex.enqueue_tool_call("echo", {"x": 1})

    def on_tool_call(req):
        raise ConstitutionalBlock("blocked by A1")

    result = asyncio.run(
        bridge_run_turn(
            executor=ex,
            agent_def=_agent_def(),
            history=[{"role": "user", "content": "hi"}],
            on_tool_call=on_tool_call,
        )
    )
    assert isinstance(result, tuple)
    last, events = result
    assert last == ""
    assert isinstance(events, list)


def test_bridge_executes_all_tool_calls_in_order():
    ex = MockExecutor()
    ex.enqueue_events(
        [
            ToolCallRequest(name="echo", args={"n": 1}),
            ToolCallRequest(name="echo", args={"n": 2}),
        ]
    )
    ex.enqueue_response("done")
    calls: list = []
    history = [{"role": "user", "content": "hi"}]
    last, _ = asyncio.run(
        bridge_run_turn(
            executor=ex,
            agent_def=_agent_def(),
            history=history,
            on_tool_call=lambda req: calls.append(req.args) or "ok",
        )
    )
    assert calls == [{"n": 1}, {"n": 2}]
    assert last == "done"
    # Both tool rounds recorded in the history, in order.
    tool_calls = [h for h in history if h["role"] == "tool_call"]
    assert [h["content"]["args"] for h in tool_calls] == [{"n": 1}, {"n": 2}]


def test_bridge_appends_assistant_response_to_history():
    ex = MockExecutor()
    ex.enqueue_response("final answer")
    history = [{"role": "user", "content": "hi"}]
    last, _ = asyncio.run(
        bridge_run_turn(executor=ex, agent_def=_agent_def(), history=history)
    )
    assert last == "final answer"
    assert history[-1]["role"] == "assistant"
    assert history[-1]["content"] == "final answer"


class _AlwaysToolExecutor(Executor):
    """Yields a tool call on every round, forever."""

    def __init__(self) -> None:
        self.rounds = 0

    async def run_turn(self, messages, tools, system_prompt, config=None):
        self.rounds += 1
        yield ToolCallRequest(name="echo", args={"round": self.rounds})


def test_bridge_caps_iterations():
    ex = _AlwaysToolExecutor()
    history = [{"role": "user", "content": "hi"}]
    last, events = asyncio.run(
        bridge_run_turn(
            executor=ex,
            agent_def=_agent_def(),
            history=history,
            on_tool_call=lambda req: "ok",
            max_iterations=3,
        )
    )
    assert ex.rounds == 3
    assert any(
        e.type == "error" and "max_iterations" in (e.message or "") for e in events
    )
    assert last == ""


def test_vm_run_turn_does_not_rerun_completed_turn():
    class _CountingMock(MockExecutor):
        def __init__(self) -> None:
            super().__init__()
            self.invocations = 0

        async def run_turn(self, messages, tools, system_prompt, config=None):
            self.invocations += 1
            async for ev in super().run_turn(
                messages, tools, system_prompt, config=config
            ):
                yield ev

    ex = _CountingMock()
    ex.enqueue_response("hello there")
    vm = _make_vm(executor=ex)
    last, history = vm.run_turn(text="hi")
    assert last == "hello there"
    # One LLM round only; the turn is not re-run after completion.
    assert ex.invocations == 1
    assert history[-1]["role"] == "assistant"


# ---------------------------------------------------------------------------
# Finding 3: Databricks "streaming" cannot stream with the pinned SDK.
#
# databricks-sdk 0.50.0's serving_endpoints.query(stream=True) returns a
# SINGLE QueryEndpointResponse (Accept: application/json, from_dict) —
# verified against the installed SDK source; there is no
# ChatCompletionChunk type. The old code iterated the "chunk" iterator
# and raised TypeError on the first production turn. stream() now
# surfaces one-shot invoke() as a single delta, so the ReAct loop, tool
# calls and audit trail work; only token-by-token display is lost.
# ---------------------------------------------------------------------------


def test_databricks_stream_surfaces_tool_calls_and_finish_reason():
    from alex.llm import DatabricksServingLLM

    calls_seen = []

    def fake_invoke(messages, *, tools=None, temperature=None, max_tokens=None):
        calls_seen.append({"tools": tools, "max_tokens": max_tokens})
        # Shaped like the REAL LLMResponse (content, no full_text): the
        # old fake carried a phantom full_text attribute, so the test
        # passed while production streams died on AttributeError.
        return SimpleNamespace(
            content="thinking",
            tool_calls=[
                SimpleNamespace(
                    name="echo", args={"a": 1, "b": "x"}, id="c1"
                ),
                SimpleNamespace(name="other", args={}, id="c2"),
            ],
        )

    llm = DatabricksServingLLM(sdk_client=SimpleNamespace())
    llm.invoke = fake_invoke  # type: ignore[method-assign]
    deltas = list(llm.stream([{"role": "user", "content": "hi"}]))
    tool_deltas = [d for d in deltas if d.tool_call is not None]
    assert len(tool_deltas) == 2
    assert tool_deltas[0].tool_call.name == "echo"
    assert tool_deltas[0].tool_call.args == {"a": 1, "b": "x"}
    assert tool_deltas[0].tool_call.id == "c1"
    assert tool_deltas[1].tool_call.name == "other"
    # Complete tool calls arrive before the final done delta.
    assert deltas[-1].done is True
    assert deltas[-1].finish_reason == "tool_calls"
    assert deltas[-1].full_text == "thinking"
    assert calls_seen and calls_seen[0]["tools"] is None


def test_databricks_stream_no_tools_finishes_with_stop():
    from alex.llm import DatabricksServingLLM

    def fake_invoke(messages, **kw):
        return SimpleNamespace(content="hi there", tool_calls=[])

    llm = DatabricksServingLLM(sdk_client=SimpleNamespace())
    llm.invoke = fake_invoke  # type: ignore[method-assign]
    deltas = list(llm.stream([{"role": "user", "content": "hi"}]))
    assert deltas[-1].finish_reason == "stop"
    assert deltas[-1].full_text == "hi there"


# ---------------------------------------------------------------------------
# Finding 7: branch_session copies parent_id itself
# ---------------------------------------------------------------------------


class _FakeCursor:
    def __init__(self, fetchone_row=None):
        self.executed: list[tuple[str, tuple]] = []
        self._row = fetchone_row

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchone(self):
        return self._row

    def fetchall(self):
        return []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_branch_session_prod_sql_sets_parent_id_to_parent(monkeypatch):
    from agent import session_manager

    parent_row = (
        "parent-1",
        "alice@x.com",
        "personal",
        "root",
        None,
        ["alice@x.com"],
        "",
        None,
        None,
    )
    cursor = _FakeCursor(fetchone_row=parent_row)
    conn = SimpleNamespace(cursor=lambda: cursor)
    monkeypatch.setattr(
        session_manager,
        "get_settings",
        lambda: SimpleNamespace(mode="production"),
    )
    session_manager.branch_session(conn, "parent-1", "alice@x.com", "branch title")
    insert_sql, params = cursor.executed[-1]
    # The new row's parent_id column is the parent's session_id, not the
    # parent's own parent_id.
    assert "session_id, participants, model" in insert_sql
    assert "parent_id, participants, model," not in insert_sql.split("SELECT")[1]
    assert params[-1] == "parent-1"


# ---------------------------------------------------------------------------
# Finding 8: context engine wired into VM.run_turn
# ---------------------------------------------------------------------------


class _RecordingExecutor(Executor):
    def __init__(self) -> None:
        self.system_prompts: list[str] = []
        self.messages: list[list[dict[str, Any]]] = []

    async def run_turn(self, messages, tools, system_prompt, config=None):
        self.system_prompts.append(system_prompt)
        self.messages.append(list(messages))
        yield TurnComplete(response="ok")


def test_vm_run_turn_includes_memory_and_standing_orders_in_prompt():
    from agent import vector_memory
    from alex.testing import get_fakes

    fakes = get_fakes()
    fakes.warehouse.register(
        "media.control.standing_orders",
        ["order_id", "title", "instruction", "active", "audience"],
        rows=[
            (
                "SO-1",
                "Metric units",
                "Always report distances in kilometres.",
                True,
                ["all"],
            )
        ],
    )
    vector_memory.store(
        None, "alice@x.com", "preference", "Alice prefers summaries in bullet points."
    )
    ex = _RecordingExecutor()
    vm = _make_vm(executor=ex, warehouse=fakes.warehouse)
    # The message shares terms with the stored bead: relevance-aware
    # recall (0.6*lexical + 0.4*salience) keeps it in the prompt.
    last, _ = vm.run_turn(text="Plan my trip with bullet summaries")
    assert last == "ok"
    prompt = ex.system_prompts[0]
    assert "You are Alex" in prompt
    assert "Always report distances in kilometres." in prompt
    assert "Alice prefers summaries in bullet points." in prompt
    assert "Plan my trip" in prompt


def test_vm_run_turn_degrades_to_base_prompt_offline():
    ex = _RecordingExecutor()
    vm = _make_vm(executor=ex)
    vm.run_turn(text="hello")
    assert "You are Alex" in ex.system_prompts[0]


# ---------------------------------------------------------------------------
# Finding 9: audit rows carry real args, unique ids, governance columns
# ---------------------------------------------------------------------------


def test_audit_rows_have_unique_ids_and_real_args():
    vm = _make_vm()
    s = vm.open_browser_session()
    t = vm.open_browser_tab(s.session_id, url="alex://welcome")
    for url in ("alex://welcome", "alex://docs"):
        vm._invoke_tool(
            ToolCallRequest(
                name="browser_navigate",
                args={"session_id": s.session_id, "tab_id": t.tab_id, "url": url},
            )
        )
    rows = vm._session.audit_log._wh.execute(
        "SELECT id, action_id, actor, tool_name, tool_args, outcome, "
        "input_json, output_json FROM media.alex.agent_actions"
    ).as_dicts()
    ids = [r["action_id"] for r in rows]
    assert len(ids) == len(set(ids)), "action_id must not collide"
    ok_rows = [r for r in rows if r["outcome"] == "ok"]
    assert ok_rows
    for r in ok_rows:
        args = json.loads(r["input_json"])
        assert args["url"].startswith("alex://")
        # Governance columns mirror the harness columns.
        assert r["actor"] == "alice@x.com"
        assert r["tool_name"] == "browser_navigate"
        assert r["tool_args"] == r["input_json"]
        assert r["id"] == r["action_id"]
        assert r["output_json"]


# ---------------------------------------------------------------------------
# Finding 10: ollama executor preserves tool-call linkage
# ---------------------------------------------------------------------------


def test_ollama_message_conversion_replays_tools_as_plain_context():
    # Prior tool interactions are replayed to the model as plain
    # conversational context (assistant note + user results block), NOT
    # the strict OpenAI tool_calls/tool-role protocol — the protocol form
    # makes some served models (Kimi via Ollama) return empty synthesis.
    from agent.omnigent.local_ollama_executor import _to_oai_messages

    history = [
        {"role": "user", "content": "hi"},
        {"role": "tool_call", "content": {"tool": "echo", "args": {"n": 1}}},
        {"role": "tool_result", "content": '{"ok": true}'},
    ]
    out = _to_oai_messages(history, "sys")
    assert out[0] == {"role": "system", "content": "sys"}
    # No tool-role messages and no tool_calls arrays are sent back.
    assert all(m["role"] != "tool" for m in out)
    assert all("tool_calls" not in m for m in out)
    # The tool name and its result both appear as text the model can read.
    joined = " ".join(m.get("content") or "" for m in out)
    assert "echo" in joined
    assert '{"ok": true}' in joined


# ---------------------------------------------------------------------------
# Finding 11: dead code removed
# ---------------------------------------------------------------------------


def test_vm_module_has_no_browser_articles_constant():
    import agent.vm as vm_mod

    assert not hasattr(vm_mod, "_BROWSER_ARTICLES")


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
