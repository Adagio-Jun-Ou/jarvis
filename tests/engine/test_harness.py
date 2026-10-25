"""
Tests for the agent.harness package.

The harness is the wiring layer between the agent's tools, the
LLM executor, the constitution interceptor, and the SSE event
pump.  These tests assert the small, specific invariants every
caller (the VM, the FastAPI route, the SDK, the test recorder)
relies on.
"""
from __future__ import annotations

import pytest

from agent.harness import (
    ConstitutionAdapter,
    EventPump,
    executor_for,
)
from agent.omnigent import PolicyAction

# ---------------------------------------------------------------------------
# EventPump
# ---------------------------------------------------------------------------


def test_event_pump_fans_out_to_every_subscriber():
    pump = EventPump(name="t1")
    a, b = [], []
    pump.subscribe(a.append, tag="a")
    pump.subscribe(b.append, tag="b")
    from alex.events import text_event

    pump.emit(text_event("hi"))
    assert len(a) == 1
    assert len(b) == 1
    assert a[0].text == "hi"


def test_event_pump_unsubscribe_works():
    pump = EventPump()
    seen = []
    unsub = pump.subscribe(seen.append, tag="s")
    from alex.events import text_event

    pump.emit(text_event("first"))
    assert len(seen) == 1
    unsub()
    pump.emit(text_event("second"))
    assert len(seen) == 1


def test_event_pump_drops_failing_subscriber_but_keeps_others():
    pump = EventPump()

    def bad(_ev):
        raise RuntimeError("boom")

    good: list = []
    pump.subscribe(bad, tag="bad")
    pump.subscribe(good.append, tag="good")
    from alex.events import text_event

    pump.emit(text_event("hi"))
    assert len(good) == 1
    # The bad subscriber is removed automatically.
    assert len(pump) == 1


def test_event_pump_buffer_is_bounded():
    pump = EventPump()
    from alex.events import text_event

    for i in range(2000):
        pump.emit(text_event(f"e{i}"))
    assert len(pump.recent(10000)) <= 1024


def test_event_pump_recent_returns_tail_in_order():
    pump = EventPump()
    from alex.events import text_event

    for i in range(5):
        pump.emit(text_event(f"e{i}"))
    tail = [e.text for e in pump.recent(3)]
    assert tail == ["e2", "e3", "e4"]


def test_event_pump_emit_dict_builds_event():
    pump = EventPump()
    seen: list = []
    pump.subscribe(seen.append)
    pump.emit_dict("text", text="hello")
    assert seen[0].type == "text"
    assert seen[0].text == "hello"


# ---------------------------------------------------------------------------
# executor_for
# ---------------------------------------------------------------------------


def test_executor_for_dev_returns_local_ollama_executor(monkeypatch):
    from alex import config
    monkeypatch.setenv("ALEX_ENV", "dev")
    config._settings = None
    ex = executor_for("dev")
    from agent.omnigent import LocalOllamaExecutor
    assert isinstance(ex, LocalOllamaExecutor)


def test_executor_for_production_returns_databricks_executor(monkeypatch):
    from alex import config
    monkeypatch.setenv("ALEX_ENV", "production")
    monkeypatch.setenv("ALEX_LLM_BACKEND", "databricks")
    config._settings = None
    ex = executor_for("production")
    from agent.omnigent import DatabricksExecutor
    assert isinstance(ex, DatabricksExecutor)


def test_executor_for_unknown_mode_raises():
    with pytest.raises(ValueError):
        executor_for("nope")


# ---------------------------------------------------------------------------
# ConstitutionAdapter
# ---------------------------------------------------------------------------


def test_constitution_adapter_allows_when_no_session():
    pol = ConstitutionAdapter(session_getter=lambda: object())
    res = asyncio_run(pol.evaluate(content={"tool": "x", "args": {}}, phase="tool_call"))
    # Unknown tool: fail-closed DENY (the adapter's no-capability rule).
    assert res.action == PolicyAction.DENY


def test_constitution_adapter_allows_when_session_has_no_interceptor():
    from agent.guardrails import TOOL_CAPABILITY

    class _S:
        user_email = "alice@x.com"
        instance_scope = "default"
        # Deliberately no `constitution` attribute.

    pol = ConstitutionAdapter(session_getter=lambda: _S())
    known = next(iter(TOOL_CAPABILITY))
    res = asyncio_run(pol.evaluate(content={"tool": known, "args": {}}, phase="tool_call"))
    assert res.action == PolicyAction.ALLOW


def test_constitution_adapter_invokes_interceptor_and_emits_law_block():
    from agent.constitution import ConstitutionalBlock
    from agent.guardrails import TOOL_CAPABILITY

    blocks: list = []

    def _make_interceptor(*_a, **_k):
        class _Stub:
            def check(self, *args, **kwargs):
                raise ConstitutionalBlock(
                    article_id="A4",
                    title="rights",
                    detail="check_rights missing",
                )

            def record(self, *args, **kwargs):
                return

        return _Stub()

    import agent.constitution as cm

    orig = cm.ConstitutionalInterceptor.__init__
    cm.ConstitutionalInterceptor.__init__ = _make_interceptor  # type: ignore[assignment]
    try:
        class _S:
            user_email = "alice@x.com"
            instance_scope = "default"
            constitution = _make_interceptor()

        pol = ConstitutionAdapter(
            session_getter=lambda: _S(),
            on_block=lambda payload, exc: blocks.append((payload, exc)),
        )
        known = next(iter(TOOL_CAPABILITY))
        res = asyncio_run(pol.evaluate(content={"tool": known, "args": {}}, phase="tool_call"))
        assert res.action == PolicyAction.DENY
        assert len(blocks) == 1
        assert blocks[0][0]["tool"] == known
    finally:
        cm.ConstitutionalInterceptor.__init__ = orig  # type: ignore[assignment]


def test_constitution_adapter_ignores_non_tool_call_phases():
    pol = ConstitutionAdapter(session_getter=lambda: object())
    res = asyncio_run(pol.evaluate(content="just a user message", phase="request"))
    assert res.action == PolicyAction.ALLOW


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def asyncio_run(awaitable):
    # asyncio.run works regardless of whether an earlier test left the
    # thread without a current event loop (asyncio.get_event_loop no
    # longer implicitly creates one after another test ran asyncio.run).
    import asyncio

    return asyncio.run(awaitable)
