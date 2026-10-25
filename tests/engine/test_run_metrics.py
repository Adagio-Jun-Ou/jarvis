"""Run accounting + the stream watchdog — the VM-usage-control pass.

What these pin:

* **Token/latency usage flows end to end.** The provider reports it on
  the final delta; the executor emits PassUsage (on TOOL-CALL passes
  too — most of a working turn); the bridge hands it, with the pass's
  measured latency, to the governor hook; the governor accumulates it
  into state().
* **The token ceiling is a real bound** when configured, and OFF by
  default (providers that report nothing must not halt runs).
* **A stalled stream cannot hang the loop.** Silence past
  ALEX_STREAM_IDLE_S abandons the pass as an error instead of blocking
  the worker thread forever — the one bound nothing used to enforce.
* **The ledger is durable.** Each working turn writes a
  media.ops.run_metrics row the exec view can aggregate (P95, spend).
* **Compaction can carry fresh state.** The live plan snapshot rides
  the digest head, so mid-turn plan changes survive the compression
  that used to bury them.
* **The production stream regression.** DatabricksServingLLM.stream
  read `.full_text` off LLMResponse, which has `.content` — every
  production turn died as an executor error before the first token.
"""

from __future__ import annotations

import asyncio

import pytest

from agent.execution_governor import Budget, ExecutionGovernor


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("ALEX_ENV", "dev")
    from alex import config, testing

    config._settings = None
    testing._fakes = None
    yield
    config._settings = None
    testing._fakes = None


def _msg(role, content):
    return {"role": role, "content": content}


# ---- usage: executor -> bridge -> governor ---------------------------------


def test_pass_usage_reaches_the_hook_with_latency(env):
    from agent.omnigent import AgentDef
    from agent.omnigent._alex_bridge import run_turn
    from agent.omnigent.executor import PassUsage, TurnComplete

    class CountingExecutor:
        async def run_turn(self, history, tools, sys_prompt, config=None):
            yield PassUsage(usage={"prompt_tokens": 900, "completion_tokens": 55})
            yield TurnComplete(response="done")

    seen = []
    asyncio.run(run_turn(
        executor=CountingExecutor(),
        agent_def=AgentDef(name="t", prompt="p"),
        history=[_msg("user", "hello")],
        on_llm_pass=lambda usage=None: seen.append(usage),
    ))
    assert len(seen) == 1
    assert seen[0]["prompt_tokens"] == 900
    assert seen[0]["completion_tokens"] == 55
    assert seen[0]["latency_ms"] >= 0, "the bridge measures the pass"


def test_zero_arg_hooks_still_count_passes(env):
    """Backward compatibility: a hook written before usage existed."""
    from agent.omnigent import AgentDef
    from agent.omnigent._alex_bridge import run_turn
    from agent.omnigent.executor import TurnComplete

    class OneShot:
        async def run_turn(self, history, tools, sys_prompt, config=None):
            yield TurnComplete(response="ok")

    hits = []
    asyncio.run(run_turn(
        executor=OneShot(),
        agent_def=AgentDef(name="t", prompt="p"),
        history=[_msg("user", "hi")],
        on_llm_pass=lambda: hits.append(1),
    ))
    assert hits == [1]


def test_the_governor_accumulates_usage_and_reports_it(env):
    gov = ExecutionGovernor(goal="report")
    gov.record_llm_pass({"prompt_tokens": 1000, "completion_tokens": 100,
                         "latency_ms": 850.0})
    gov.record_llm_pass({"prompt_tokens": 1200, "completion_tokens": 80,
                         "latency_ms": 640.0})
    state = gov.state()
    assert state["llm_turns"] == 2, "counting and cost move in lockstep"
    assert state["usage"]["prompt_tokens"] == 2200
    assert state["usage"]["completion_tokens"] == 180
    assert state["usage"]["pass_ms"] == [850.0, 640.0]
    assert gov.total_tokens == 2380


def test_the_token_ceiling_halts_and_is_off_by_default(env):
    gov = ExecutionGovernor(goal="spend")
    gov.record_llm_pass({"prompt_tokens": 10_000_000, "completion_tokens": 0})
    assert gov.before_tool("view", {}, "files.read").action == "continue", (
        "no ceiling configured -> tokens never halt"
    )
    bounded = ExecutionGovernor(goal="spend", budget=Budget(max_total_tokens=500))
    bounded.record_llm_pass({"prompt_tokens": 400, "completion_tokens": 200})
    verdict = bounded.before_tool("view", {}, "files.read")
    assert verdict.action == "halt"
    assert verdict.pathology == "budget_tokens"


# ---- the stream watchdog ---------------------------------------------------


def test_a_stalled_stream_is_abandoned_not_awaited_forever(env, monkeypatch):
    monkeypatch.setenv("ALEX_STREAM_IDLE_S", "0.1")
    from agent.omnigent import AgentDef
    from agent.omnigent._alex_bridge import run_turn
    from agent.omnigent.executor import TextChunk

    class HangingExecutor:
        async def run_turn(self, history, tools, sys_prompt, config=None):
            yield TextChunk(text="starting…")
            await asyncio.sleep(30)  # never yields again

    async def _bounded():
        return await asyncio.wait_for(run_turn(
            executor=HangingExecutor(),
            agent_def=AgentDef(name="t", prompt="p"),
            history=[_msg("user", "hello")],
        ), timeout=5)

    final, events = asyncio.run(_bounded())
    stalls = [e for e in events
              if getattr(e, "type", "") == "error"
              and "stalled" in (getattr(e, "message", "") or "")]
    assert stalls, "the stall must be SAID, not silent"


def test_a_slow_but_streaming_model_is_never_cut(env, monkeypatch):
    monkeypatch.setenv("ALEX_STREAM_IDLE_S", "0.3")
    from agent.omnigent import AgentDef
    from agent.omnigent._alex_bridge import run_turn
    from agent.omnigent.executor import TextChunk, TurnComplete

    class SlowDrip:
        async def run_turn(self, history, tools, sys_prompt, config=None):
            for _ in range(4):
                await asyncio.sleep(0.1)  # under the idle cap each time
                yield TextChunk(text="…")
            yield TurnComplete(response="made it")

    final, _events = asyncio.run(run_turn(
        executor=SlowDrip(),
        agent_def=AgentDef(name="t", prompt="p"),
        history=[_msg("user", "hello")],
    ))
    assert final == "made it", "per-event silence, not per-pass total"


# ---- the production stream regression --------------------------------------


def test_databricks_stream_yields_the_response_content(env):
    from alex.llm import DatabricksServingLLM, LLMResponse

    llm = object.__new__(DatabricksServingLLM)
    llm.invoke = lambda *a, **kw: LLMResponse(  # type: ignore[method-assign]
        content="the actual answer",
        usage={"prompt_tokens": 10, "completion_tokens": 3},
    )
    deltas = list(llm.stream([{"role": "user", "content": "q"}]))
    assert any(d.text == "the actual answer" for d in deltas)
    final = deltas[-1]
    assert final.done and final.finish_reason == "stop"
    assert final.usage == {"prompt_tokens": 10, "completion_tokens": 3}


# ---- the durable ledger ----------------------------------------------------


def test_each_working_turn_writes_a_run_metrics_row(env):
    from agent import taskboard as tb
    from alex.testing import get_fakes
    from app._dev_seed import (
        RUN_METRICS_COLS,
        TASK_CREW_COLS,
        TASK_MESSAGES_COLS,
        TASK_RUNS_COLS,
    )

    tb.reset_runs()
    wh = get_fakes().warehouse
    wh.register(
        "media.alex.tasks",
        ["id", "owner_email", "assignee_email", "title", "description",
         "status", "priority", "due_at", "source", "source_ref",
         "created_at", "updated_at", "closed_at"],
        pk=["id"],
    )
    wh.register("media.control.task_crew", TASK_CREW_COLS)
    wh.register("media.control.task_messages", TASK_MESSAGES_COLS,
                pk=["message_id"])
    wh.register("media.control.task_runs", TASK_RUNS_COLS, pk=["run_id"])
    wh.register("media.ops.run_metrics", RUN_METRICS_COLS,
                pk=["run_id", "turn"])

    def sql(stmt, params=None):
        res = wh.execute(stmt, params or {})
        as_dicts = getattr(res, "as_dicts", None)
        return as_dicts() if callable(as_dicts) else list(res or [])

    sql("INSERT INTO media.alex.tasks (id, owner_email, title, status) "
        "VALUES ('t-m', 'dana@alex.dev', 'Metered task', 'todo')")

    class MeteredVM:
        session_id = "task-t-m"

        def subscribe_sse(self, fn):
            return lambda: None

        def last_governor_state(self):
            return {
                "elapsed_s": 7.5, "waited_s": 2.0, "tool_calls": 4,
                "llm_turns": 6, "denials": 0,
                "usage": {"prompt_tokens": 3200, "completion_tokens": 410,
                          "pass_ms": [500.0]},
                "budget": {"max_llm_turns": 12}, "halted": None,
            }

        def run_turn(self, *, text, **_kw):
            return "worked", []

        def close(self):
            pass

    class P:
        email = "dana@alex.dev"
        is_admin = False

    run = tb.start_run(sql, P(), "t-m", background=False,
                       vm_factory=lambda **kw: MeteredVM())
    rows = sql("SELECT run_id, turn, llm_turns, tool_calls, elapsed_s, "
               "waited_s, prompt_tokens, completion_tokens "
               "FROM media.ops.run_metrics")
    assert len(rows) == 1
    row = rows[0]
    assert row["run_id"] == run.run_id
    assert int(row["prompt_tokens"]) == 3200
    assert int(row["completion_tokens"]) == 410
    assert float(row["waited_s"]) == 2.0
    ev = next(e for e in run.events if e.get("type") == "run_usage")
    assert ev["prompt_tokens"] == 3200 and ev["waited_s"] == 2.0
    tb.reset_runs()


# ---- swarm children spend the parent's meter -------------------------------


def test_child_budget_is_a_slice_of_the_parents_remainder(env):
    from types import SimpleNamespace

    from agent.swarm import _child_budget

    gov = ExecutionGovernor(
        goal="fan out",
        budget=Budget(max_tool_calls=40, max_llm_turns=12, max_seconds=600),
    )
    for _ in range(20):
        gov.record(kind="tool", name="view")
    for _ in range(4):
        gov.record(kind="llm")
    parent = SimpleNamespace(governor=gov, _governor=gov)
    child = _child_budget(parent, 4)
    assert child.max_tool_calls == 5, "remaining 20 tools split across 4"
    assert child.max_llm_turns == 3, "remaining 8 passes split, +1"
    assert child.max_seconds <= 600

    assert _child_budget(SimpleNamespace(governor=None, _governor=None), 4) is None


def test_the_vm_narrows_its_budget_to_the_ceiling(env):
    from agent.omnigent import MockExecutor
    from agent.vm import make_vm

    vm = make_vm(
        session_id="s-ceiling", user_email="dana@alex.dev",
        executor=MockExecutor(),
        budget_ceiling=Budget(max_tool_calls=5, max_llm_turns=2,
                              max_seconds=60.0),
    )
    vm.run_turn(text="hello")
    b = vm.governor.budget
    assert b.max_tool_calls == 5
    assert b.max_llm_turns == 2
    assert b.max_seconds == 60.0
    vm.close()


def test_run_swarm_hands_every_child_the_shared_slice(env):
    from types import SimpleNamespace

    from agent import swarm

    gov = ExecutionGovernor(goal="fan out", budget=Budget(
        max_tool_calls=40, max_llm_turns=12, max_seconds=600))

    class Pump:
        def emit(self, ev):
            pass

    parent = SimpleNamespace(
        governor=gov, _governor=gov, pump=Pump(),
        session_id="s-par", user_email="dana@alex.dev",
        _allowed_capabilities=lambda: {"files.read"},
    )
    seen = []

    class Child:
        def __init__(self, **kw):
            seen.append(kw)

        def subscribe_sse(self, fn):
            return lambda: None

        def run_turn(self, *, text, **_kw):
            return "ok", []

        def close(self):
            pass

    out = swarm.run_swarm(
        parent,
        [{"name": "a", "brief": "x"}, {"name": "b", "brief": "y"}],
        vm_factory=lambda **kw: Child(**kw),
    )
    assert out["ok"]
    assert len(seen) == 2
    for kw in seen:
        assert kw["budget_ceiling"] is not None
        assert kw["budget_ceiling"].max_tool_calls == 20, "40 across 2 workers"


# ---- compaction carries fresh state ----------------------------------------


def test_compaction_head_carries_the_live_note(env):
    from agent.harness.compaction import compact_history

    history = [{"role": "user", "content": "the task"}]
    history += [{"role": "assistant", "content": f"step {i}"} for i in range(30)]
    out, ran = compact_history(
        history, trigger=10, keep_tail=4,
        live_note="EXISTING PLAN:\n  [x] 1. gather\n  [~] 2. draft",
    )
    assert ran
    head = out[0]["content"]
    assert "CURRENT STATE (re-read live just now" in head
    assert "[~] 2. draft" in head
    assert out[1]["content"] == "the task", "the anchor still survives verbatim"


def test_no_live_note_means_no_extra_banner(env):
    from agent.harness.compaction import compact_history

    history = [{"role": "user", "content": "the task"}]
    history += [{"role": "assistant", "content": f"step {i}"} for i in range(30)]
    out, ran = compact_history(history, trigger=10, keep_tail=4)
    assert ran
    assert "CURRENT STATE" not in out[0]["content"]
