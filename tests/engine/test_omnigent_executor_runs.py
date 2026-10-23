"""End-to-end tests for the Omnigent executor + bridge.

These tests exercise the smallest possible agent loop against the
:mod:`agent.omnigent` vendored copy.  We use the upstream
:class:`MockExecutor` so the test is fully offline and deterministic;
a separate integration test (gated behind
``pytest.mark.integration``) drives a real Ollama call.
"""
from __future__ import annotations

import asyncio

from agent.harness.constitution_adapter import ConstitutionAdapter
from agent.omnigent import (
    FunctionTool,
    MockExecutor,
    TextChunk,
    ToolCallRequest,
    TurnComplete,
)
from agent.omnigent._alex_bridge import (
    build_agent_def,
    executor_event_to_agent_event,
    run_turn,
)


def _add(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y


def _echo(text: str) -> str:
    """Echo the input back."""
    return text


def test_mock_executor_emits_scripted_events():
    async def _go() -> None:
        ex = MockExecutor()
        ex.enqueue_response("hello world")
        events = []
        async for ev in ex.run_turn(
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
            system_prompt="",
        ):
            events.append(ev)
        assert len(events) == 1
        assert isinstance(events[0], TurnComplete)
        assert events[0].response == "hello world"

    asyncio.run(_go())


def test_mock_executor_tool_call_then_response():
    async def _go() -> None:
        ex = MockExecutor()
        ex.enqueue_tool_call("add", {"x": 2, "y": 3}, follow_up_response="5")
        ex.enqueue_response("5 it is")
        out = []
        async for ev in ex.run_turn(
            messages=[{"role": "user", "content": "2+3?"}],
            tools=[],
            system_prompt="",
        ):
            out.append(ev)
        # First turn: ToolCallRequest
        assert isinstance(out[0], ToolCallRequest)
        assert out[0].name == "add"
        assert out[0].args == {"x": 2, "y": 3}

    asyncio.run(_go())


def test_function_tool_schema_auto_generated():
    tool = FunctionTool(name="add", description="Add two numbers", callable=_add)
    schema = tool.tool_schema()
    assert schema["name"] == "add"
    assert schema["description"] == "Add two numbers"
    assert schema["parameters"]["type"] == "object"
    assert "x" in schema["parameters"]["properties"]
    assert "y" in schema["parameters"]["properties"]
    assert set(schema["parameters"]["required"]) == {"x", "y"}


def test_build_agent_def_creates_function_tools_with_scopes():
    agent = build_agent_def(
        name="alex",
        prompt="you are alex",
        tools={"add": _add, "echo": _echo},
    )
    assert agent.name == "alex"
    assert "add" in agent.tools
    assert "echo" in agent.tools
    for tool in agent.tools.values():
        # The default scope is `alex:tool:<name>`
        assert tool.scopes == {f"alex:tool:{tool.name}"}


def test_run_turn_emits_sse_events_for_text_response():
    async def _go() -> None:
        ex = MockExecutor()
        # Use raw events so we get both a text chunk and a turn complete.
        from agent.omnigent import TextChunk, TurnComplete

        ex.enqueue_events([TextChunk(text="hi from alex"), TurnComplete(response="hi from alex")])
        agent = build_agent_def(
            name="alex",
            prompt="you are alex",
            tools={"add": _add},
        )
        history = [{"role": "user", "content": "hi"}]
        last, events = await run_turn(
            executor=ex,
            agent_def=agent,
            history=history,
        )
        assert last == "hi from alex"
        # The bridge streams `text`; `done` is emitted once by the app
        # layer after run_turn returns (never by the bridge, so
        # post-turn events like `memory` aren't cut off).
        types = [e.type for e in events]
        assert "text" in types
        assert "done" not in types

    asyncio.run(_go())


def test_run_turn_invokes_tool_then_continues():
    async def _go() -> None:
        ex = MockExecutor()
        ex.enqueue_tool_call("add", {"x": 1, "y": 2})
        ex.enqueue_response("1+2=3")
        agent = build_agent_def(
            name="alex",
            prompt="you are alex",
            tools={"add": _add},
        )
        tool_calls: list[str] = []

        def _on(req: ToolCallRequest) -> dict[str, Any]:
            tool_calls.append(req.name)
            # Dispatch: invoke the actual Python callable
            return _add(**req.args)

        history = [{"role": "user", "content": "what is 1+2?"}]
        last, events = await run_turn(
            executor=ex,
            agent_def=agent,
            history=history,
            on_tool_call=_on,
        )
        assert tool_calls == ["add"]
        assert last == "1+2=3"
        # The history was extended with the tool_call + tool_result pair
        assert any(m["role"] == "tool_call" for m in history)
        assert any(m["role"] == "tool_result" for m in history)

    asyncio.run(_go())


def test_executor_event_to_agent_event_handles_text_chunk():

    ev = TextChunk(text="hello")
    out = executor_event_to_agent_event(ev)
    assert out is not None
    assert out.type == "text"
    assert out.text == "hello"


def test_executor_event_to_agent_event_ignores_empty_chunk():
    assert executor_event_to_agent_event(TextChunk(text="")) is None


# ---------------------------------------------------------------------------
# Constitution adapter
# ---------------------------------------------------------------------------


def test_constitution_policy_allows_when_no_session_getter():
    pol = ConstitutionAdapter(session_getter=lambda: object())
    res = asyncio.run(
        pol.evaluate(
            content={"tool": "nuke_it", "args": {}},
            phase="tool_call",
        )
    )
    # No session, but unknown tool → fail-closed DENY.
    from agent.omnigent import PolicyAction

    assert res.action == PolicyAction.DENY


def test_constitution_policy_allows_when_session_has_no_interceptor():
    """A session that didn't wire a constitution lets the guardrails gate decide."""
    from agent.guardrails import TOOL_CAPABILITY

    class _S:
        user_email = "alice@example.com"
        instance_scope = "default"
        # Deliberately no `constitution` attribute.

    pol = ConstitutionAdapter(session_getter=lambda: _S())
    known_tool = next(iter(TOOL_CAPABILITY))
    res = asyncio.run(
        pol.evaluate(
            content={"tool": known_tool, "args": {}},
            phase="tool_call",
        )
    )
    from agent.omnigent import PolicyAction

    assert res.action == PolicyAction.ALLOW


def test_constitution_policy_denies_unknown_tool():
    """A tool with no capability mapping must fail closed, even with no session."""
    pol = ConstitutionAdapter(session_getter=lambda: object())
    res = asyncio.run(
        pol.evaluate(
            content={"tool": "nuke_it_from_orbit", "args": {}},
            phase="tool_call",
        )
    )
    from agent.omnigent import PolicyAction

    assert res.action == PolicyAction.DENY
    assert "no capability mapping" in (res.reason or "")


def test_constitution_policy_invokes_interceptor_and_allows():
    """When the session wires a constitution, the policy delegates to it."""
    from agent.constitution import ConstitutionalInterceptor

    calls: list[tuple[str, dict]] = []

    def _make_interceptor(*_args, **_kwargs) -> ConstitutionalInterceptor:
        class _StubInterceptor:
            def check(self, *args, **kwargs) -> None:
                calls.append((args, kwargs))

            def record(self, *args, **kwargs) -> None:  # noqa: ARG002
                return

        return _StubInterceptor()  # type: ignore[return-value]

    # Monkey-patch ConstitutionalInterceptor.__init__ for the test scope.
    import agent.constitution as constitution_mod

    orig = constitution_mod.ConstitutionalInterceptor.__init__
    constitution_mod.ConstitutionalInterceptor.__init__ = _make_interceptor  # type: ignore[assignment]

    try:
        from agent.guardrails import TOOL_CAPABILITY

        class _S:
            user_email = "alice@example.com"
            instance_scope = "default"
            constitution = _make_interceptor()

        pol = ConstitutionAdapter(session_getter=lambda: _S())
        known_tool = next(iter(TOOL_CAPABILITY))
        res = asyncio.run(
            pol.evaluate(
                content={"tool": known_tool, "args": {"k": "v"}},
                phase="tool_call",
            )
        )
        from agent.omnigent import PolicyAction

        assert res.action == PolicyAction.ALLOW
        assert len(calls) == 1
        # The adapter matches the interceptor's actual signature: it
        # inspects check() and, because the stub takes no identity
        # keywords, calls it positionally as (tool, args).
        args, kwargs = calls[0]
        assert args == (known_tool, {"k": "v"})
        assert kwargs == {}
    finally:
        constitution_mod.ConstitutionalInterceptor.__init__ = orig  # type: ignore[assignment]


def test_constitution_policy_invokes_on_block_hook():
    """A ConstitutionBlock from the interceptor triggers the on_block hook + DENY."""
    from agent.constitution import ConstitutionalBlock

    def _make_interceptor(*_args, **_kwargs):
        class _StubInterceptor:
            def check(self, *args, **kwargs) -> None:  # noqa: ARG002
                raise ConstitutionalBlock(
                    article_id="A4",
                    title="rights before reuse",
                    detail="check_rights was not performed",
                )

            def record(self, *args, **kwargs) -> None:
                return

        return _StubInterceptor()  # type: ignore[return-value]

    import agent.constitution as constitution_mod

    orig = constitution_mod.ConstitutionalInterceptor.__init__
    constitution_mod.ConstitutionalInterceptor.__init__ = _make_interceptor  # type: ignore[assignment]

    try:
        from agent.guardrails import TOOL_CAPABILITY

        class _S:
            user_email = "alice@example.com"
            instance_scope = "default"
            constitution = _make_interceptor()

        blocks: list[tuple[dict, ConstitutionalBlock]] = []

        def _on(payload: dict, exc: ConstitutionalBlock) -> None:
            blocks.append((payload, exc))

        pol = ConstitutionAdapter(session_getter=lambda: _S(), on_block=_on)
        known_tool = next(iter(TOOL_CAPABILITY))
        res = asyncio.run(
            pol.evaluate(
                content={"tool": known_tool, "args": {}},
                phase="tool_call",
            )
        )
        from agent.omnigent import PolicyAction

        assert res.action == PolicyAction.DENY
        assert len(blocks) == 1
        assert blocks[0][0]["tool"] == known_tool
        assert "A4" in str(blocks[0][1]) or "rights" in str(blocks[0][1])
    finally:
        constitution_mod.ConstitutionalInterceptor.__init__ = orig  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def test_load_agent_def_resolves_dotted_callables(tmp_path):
    import yaml

    spec = tmp_path / "alex.yaml"
    spec.write_text(
        yaml.safe_dump(
            {
                "name": "alex",
                "prompt": "you are alex",
                "tools": {
                    "add": {"type": "function", "callable": "agent.omnigent._alex_bridge:run_turn"},
                },
                "policies": {
                    "constitution": {
                        "type": "function",
                        "callable": "agent.omnigent._alex_bridge:run_turn",
                        "on": ["tool_call"],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    from agent.omnigent import load_agent_def

    agent = load_agent_def(str(spec))
    assert agent.name == "alex"
    assert "add" in agent.tools
    assert "constitution" in agent.policies


# ---------------------------------------------------------------------------
# Local import to keep `Any` happy in the closure above
# ---------------------------------------------------------------------------
from typing import Any  # noqa: E402
