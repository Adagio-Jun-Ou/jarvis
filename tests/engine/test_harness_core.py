"""The agentic-harness core, held to the published state of the art.

The strong open harnesses (OpenCode, Hermes) converge on four loop-level
properties this suite pins:

* **Budgeted compaction, not a haircut** — the middle of a long
  conversation is summarized under a budget, the task (head) and the
  working set (tail) survive verbatim, stale tool payloads are pruned
  first, and repeated compactions chain (lineage) instead of rewriting
  rewritten history.
* **Plan mode is structural** — a plan-mode session's model never even
  sees a mutating tool, remote tools are withheld wholesale, and the
  call-time gate holds for paths that bypass exposure.
* **Transient executor failures retry, partial passes never do** — a
  pass that produced nothing is retried with backoff; a pass that
  already streamed text or announced tool calls is not replayed.
* **The governor budget is the loop's authority** — autopilot genuinely
  gets more model passes than suggest.
"""

from __future__ import annotations

import pytest

from agent.harness.compaction import compact_history


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


def _long_history(n_rounds=20):
    history = [_msg("user", "Reconcile the Q3 ledger against SAP totals")]
    for i in range(n_rounds):
        history.append(_msg("tool_call", {"tool": f"tool_{i}", "args": {}}))
        history.append(_msg("tool_result", f"result payload {i} " + "x" * 500))
        history.append(_msg("assistant", f"noted result {i}"))
    return history


# ---- compaction ------------------------------------------------------------


def test_short_histories_are_untouched():
    history = _long_history(2)
    out, ran = compact_history(history, llm_invoke=None, trigger=24)
    assert not ran and out is history


def test_the_task_and_the_working_set_survive_verbatim():
    history = _long_history(20)
    out, ran = compact_history(history, llm_invoke=None, trigger=24,
                               keep_tail=6)
    assert ran
    # Head protection: the FIRST user message (the task) is present,
    # verbatim, right after the digest.
    assert out[1]["content"] == "Reconcile the Q3 ledger against SAP totals"
    # Tail protection: the last 6 messages are the originals.
    assert out[-6:] == history[-6:]
    assert len(out) < len(history)


def test_stale_tool_payloads_are_pruned_before_summarizing():
    """The excerpt handed to the summarizer must carry stubs, not the
    500-char payloads — old tool output is the cheapest thing to drop."""
    seen = {}

    def fake_llm(messages):
        seen["excerpt"] = messages[0]["content"]
        return "- reconciled 14 of 20 entries; SAP delta 402.10 remains"

    history = _long_history(20)
    out, ran = compact_history(history, llm_invoke=fake_llm, trigger=24,
                               keep_tail=6)
    assert ran
    assert "stale tool result pruned" in seen["excerpt"]
    assert seen["excerpt"].count("x" * 400) == 0, (
        "raw payloads reached the summarizer — pruning did not happen"
    )
    assert "SAP delta 402.10" in out[0]["content"]


def test_summarizer_failure_degrades_to_the_deterministic_digest():
    def broken_llm(messages):
        raise RuntimeError("model unavailable")

    history = _long_history(20)
    out, ran = compact_history(history, llm_invoke=broken_llm, trigger=24,
                               keep_tail=6)
    assert ran
    assert "EARLIER CONVERSATION DIGEST" in out[0]["content"]
    # Deterministic digest carries first-line bullets.
    assert "- assistant: noted result" in out[0]["content"]


def test_repeated_compactions_update_the_summary_with_lineage():
    """Lineage without bloat: the second compaction hands the FIRST
    summary back to the summarizer to be UPDATED (the Hermes
    incremental design — done items collapse, obsolete lines drop),
    bumps the generation counter, and never concatenates digests of
    digests. The summarizer sees the prior content, so nothing is
    silently rewritten away — it is deliberately revised."""
    history = _long_history(20)
    out1, _ = compact_history(
        history, llm_invoke=lambda m: "- decision A stands", trigger=24,
        keep_tail=6,
    )
    assert "(compaction #1)" in out1[0]["content"]
    # Grow it again past the trigger.
    for i in range(20, 40):
        out1.append(_msg("tool_call", {"tool": f"tool_{i}", "args": {}}))
        out1.append(_msg("tool_result", f"result payload {i} " + "y" * 500))
        out1.append(_msg("assistant", f"noted result {i}"))
    prompts: list[str] = []

    def _update(m):
        prompts.append(m[0]["content"])
        return "- decision A stands; decision B stands"

    out2, ran = compact_history(out1, llm_invoke=_update, trigger=24,
                                keep_tail=6)
    assert ran
    assert "(compaction #2)" in out2[0]["content"]
    assert "decision A stands" in prompts[0], (
        "the prior summary must be handed back for revision"
    )
    assert "WORKING SUMMARY" in prompts[0]
    head = out2[0]["content"]
    assert "decision A stands; decision B stands" in head
    assert head.count("decision A stands") == 1, (
        "updated, not concatenated — no digests of digests"
    )


# ---- plan mode -------------------------------------------------------------


def _plan_vm(env):
    from agent.omnigent import MockExecutor
    from agent.vm import VM, HarnessConfig

    return VM(
        session_id="s-plan",
        user_email="demo@alex.dev",
        config=HarnessConfig(executor=MockExecutor(), plan_mode=True),
    )


def test_plan_mode_never_shows_the_model_a_mutating_tool(env):
    vm = _plan_vm(env)
    from agent.tools.registry import TOOL_REGISTRY
    from agent.vm import _PLAN_MODE_TOOL_EXCEPTIONS, _capability_is_readonly

    exposed = set(vm.agent_def.tools)
    assert exposed, "plan mode exposed nothing at all"
    for name in exposed:
        spec = TOOL_REGISTRY.get(name)
        assert spec is not None, f"plan mode exposed a non-builtin: {name}"
        assert (
            _capability_is_readonly(spec.capability)
            or name in _PLAN_MODE_TOOL_EXCEPTIONS
        ), f"plan mode exposed mutating tool {name} ({spec.capability})"
    # The point of the mode: reading and planning both work.
    assert "search_knowledge" in exposed
    assert "plan_task" in exposed
    # And the obvious mutators are gone.
    for tool in ("send_email", "create_file", "browser_submit",
                 "delegate_subagents", "run_python"):
        assert tool not in exposed, f"{tool} visible in plan mode"


def test_plan_mode_denies_at_call_time_too(env):
    """Exposure filtering is what the model sees; the /api/tools route
    and the SDK reach _invoke_tool directly. The gate must hold there."""
    from agent.omnigent import ToolCallRequest

    vm = _plan_vm(env)
    out = vm._invoke_tool(
        ToolCallRequest(name="create_task", args={"title": "x"})
    )
    assert out["ok"] is False and "plan mode" in out["error"]


def test_build_mode_is_unchanged(env):
    from agent.omnigent import MockExecutor
    from agent.vm import VM, HarnessConfig

    vm = VM(
        session_id="s-build",
        user_email="demo@alex.dev",
        config=HarnessConfig(executor=MockExecutor()),
    )
    assert "create_file" in vm.agent_def.tools


def test_every_turn_announces_its_model_request(env):
    """Request observability (the production-agent consensus): before
    the first model call, one event says how much context was assembled
    and how many tools were on the table — sizes, never content, since
    slices can carry confidential passages."""
    from agent.omnigent import MockExecutor
    from agent.vm import VM, HarnessConfig

    ex = MockExecutor()
    ex.enqueue_response("hello")
    vm = VM(
        session_id="s-observed",
        user_email="demo@alex.dev",
        config=HarnessConfig(executor=ex),
    )
    seen = []
    vm.subscribe_sse(seen.append)
    vm.run_turn(text="what is on my plate today?")

    requests = [
        e for e in seen
        if getattr(e, "type", "") == "context"
        and getattr(e, "source", "") == "request"
    ]
    assert requests, "no model-request event reached the stream"
    ev = requests[0]
    assert "tools exposed" in (ev.summary or "")
    items = (ev.extras or {}).get("items") or []
    assert items and all("chars" in item for item in items), (
        "slice items must carry sizes"
    )
    assert "confidential" not in str(items).lower(), "sizes only, never content"


def test_llm_turns_are_actually_counted(env):
    """The governor's llm_turns was a permanent zero — nothing ever
    recorded an llm step, so the max_llm_turns budget clause could never
    fire and oversight reported every run as having thought 0 times."""
    from agent.omnigent import MockExecutor
    from agent.vm import VM, HarnessConfig

    ex = MockExecutor()
    ex.enqueue_response("done thinking")
    vm = VM(
        session_id="s-llmcount",
        user_email="demo@alex.dev",
        config=HarnessConfig(executor=ex),
    )
    vm.run_turn(text="hello")
    assert vm.governor.llm_turns >= 1, (
        "no llm step was recorded — the turn budget is wired to nothing"
    )


def test_the_model_is_told_when_the_budget_nears(env):
    """Context anxiety, addressed at the harness level: with two passes
    left after real tool use, ONE harness note lands in the history and
    a budget event reaches the stream — the model gets to land the work
    instead of being cut off mid-flight."""
    import asyncio

    from agent.omnigent import AgentDef, MockExecutor
    from agent.omnigent._alex_bridge import run_turn

    ex = MockExecutor()
    ex.enqueue_tool_call("list_my_tasks")
    ex.enqueue_tool_call("read_inbox")
    ex.enqueue_tool_call("search_email")
    ex.enqueue_response("Here is what I found.")

    history = [{"role": "user", "content": "triage my morning"}]
    events = []
    asyncio.run(run_turn(
        executor=ex,
        agent_def=AgentDef(name="t", prompt="p"),
        history=history,
        on_tool_call=lambda req: {"ok": True},
        emit=events.append,
        max_iterations=4,
    ))

    notes = [
        m for m in history
        if m.get("role") == "user"
        and "[harness note]" in str(m.get("content"))
    ]
    assert len(notes) == 1, "exactly one budget note, never a nag"
    assert "Finish now" in notes[0]["content"]
    budget_events = [
        e for e in events
        if getattr(e, "type", "") == "context"
        and getattr(e, "source", "") == "budget"
    ]
    assert budget_events, "the budget wrap-up never reached the stream"


def test_short_turns_are_never_nagged(env):
    """A quick answer with one tool pass must not hear about budgets."""
    import asyncio

    from agent.omnigent import AgentDef, MockExecutor
    from agent.omnigent._alex_bridge import run_turn

    ex = MockExecutor()
    ex.enqueue_tool_call("read_inbox", follow_up_response="All quiet.")

    history = [{"role": "user", "content": "anything urgent?"}]
    asyncio.run(run_turn(
        executor=ex,
        agent_def=AgentDef(name="t", prompt="p"),
        history=history,
        on_tool_call=lambda req: {"ok": True},
        max_iterations=12,
    ))
    assert not any(
        "[harness note]" in str(m.get("content")) for m in history
    )


# ---- undo + diagnostics (the file loop) ------------------------------------


@pytest.fixture
def workspace(env, tmp_path, monkeypatch):
    monkeypatch.setenv("ALEX_WORKSPACE_ROOT", str(tmp_path))
    return "dana@alex.dev"


def test_a_wrong_edit_is_one_call_from_gone(workspace):
    from agent import workspace_tools as ws

    ws.create_file(workspace, "work/report.txt", "the good version")
    ws.str_replace(workspace, "work/report.txt", "good", "ruined")
    assert "ruined" in ws.view(workspace, "work/report.txt")

    out = ws.undo_file_change(workspace)
    assert "restored" in out
    assert ws.view(workspace, "work/report.txt") == "the good version"


def test_undo_of_a_create_removes_the_file(workspace):
    from agent import workspace_tools as ws

    ws.create_file(workspace, "work/oops.txt", "should not exist")
    out = ws.undo_file_change(workspace)
    assert "removed" in out
    assert "[error]" in ws.view(workspace, "work/oops.txt")
    # And with the journal empty, undo says so instead of guessing.
    assert "[error] nothing to undo" in ws.undo_file_change(workspace)


def test_undo_is_repeatable_change_by_change(workspace):
    from agent import workspace_tools as ws

    ws.create_file(workspace, "work/a.txt", "v1")
    ws.str_replace(workspace, "work/a.txt", "v1", "v2")
    ws.str_replace(workspace, "work/a.txt", "v2", "v3")
    ws.undo_file_change(workspace)
    assert ws.view(workspace, "work/a.txt") == "v2"
    ws.undo_file_change(workspace)
    assert ws.view(workspace, "work/a.txt") == "v1"


def test_a_python_syntax_error_rides_back_on_the_tool_result(workspace):
    """The diagnostics loop: the model hears about its own syntax error
    in the same tool result, not from a failed run three steps later."""
    from agent import workspace_tools as ws

    out = ws.create_file(workspace, "work/broken.py", "def f(:\n    pass\n")
    assert "[diagnostic] Python syntax error" in out
    assert "line 1" in out

    ok = ws.create_file(workspace, "work/fine.py", "def f():\n    return 1\n")
    assert "[diagnostic]" not in ok


def test_invalid_json_is_flagged_too(workspace):
    from agent import workspace_tools as ws

    out = ws.create_file(workspace, "work/cfg.json", '{"a": 1,}')
    assert "[diagnostic] invalid JSON" in out


# ---- executor retry --------------------------------------------------------


def test_a_transient_executor_failure_is_retried(env):
    """First pass dies with nothing produced; the retry answers. The
    user sees the answer, not the blip."""
    import asyncio

    from agent.omnigent import AgentDef
    from agent.omnigent._alex_bridge import run_turn
    from agent.omnigent.executor import ExecutorError, TurnComplete

    class FlakyExecutor:
        def __init__(self):
            self.calls = 0

        async def run_turn(self, history, tools, sys_prompt, config=None):
            self.calls += 1
            if self.calls == 1:
                yield ExecutorError(message="connection reset")
                return
            yield TurnComplete(response="All good on the second try.")

    ex = FlakyExecutor()
    final, _events = asyncio.run(run_turn(
        executor=ex,
        agent_def=AgentDef(name="t", prompt="p"),
        history=[_msg("user", "hello")],
    ))
    assert ex.calls == 2
    assert final == "All good on the second try."


def test_a_pass_that_streamed_text_is_never_replayed(env):
    """Replaying a partial pass can double side effects and contradict
    text the user already saw — it must fail forward, not retry."""
    import asyncio

    from agent.omnigent import AgentDef
    from agent.omnigent._alex_bridge import run_turn
    from agent.omnigent.executor import ExecutorError, TextChunk

    class PartialExecutor:
        def __init__(self):
            self.calls = 0

        async def run_turn(self, history, tools, sys_prompt, config=None):
            self.calls += 1
            yield TextChunk(text="Here is the first half…")
            yield ExecutorError(message="stream dropped")

    ex = PartialExecutor()
    asyncio.run(run_turn(
        executor=ex,
        agent_def=AgentDef(name="t", prompt="p"),
        history=[_msg("user", "hello")],
    ))
    assert ex.calls == 1, "a partial pass was replayed"


# ---- governor-scaled iteration budget --------------------------------------


def test_the_loop_budget_comes_from_the_governor(env, monkeypatch):
    """Autonomy is a real control on the LOOP, not only on tool calls:
    the max_turns handed to ReactLoop is the governor's max_llm_turns."""
    from agent.omnigent import MockExecutor
    from agent.vm import VM, HarnessConfig

    seen = {}

    class SpyLoop:
        def __init__(self, **kw):
            pass

        def run(self, max_turns=8):
            seen["max_turns"] = max_turns
            return "done", []

    import agent.vm as vm_mod

    monkeypatch.setattr(vm_mod, "ReactLoop", SpyLoop)
    vm = VM(
        session_id="s-budget",
        user_email="demo@alex.dev",
        config=HarnessConfig(executor=MockExecutor()),
    )
    vm.run_turn(text="hello")
    assert seen["max_turns"] == vm.governor.budget.max_llm_turns
    assert seen["max_turns"] >= 8, (
        "the governor default must not be tighter than the old hard-coded 8"
    )
