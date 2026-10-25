"""Tests for the vendored Omnigent public surface.

These tests verify the package imports cleanly and re-exports the
upstream-shaped symbols :mod:`agent.omnigent` documents in its
``__all__``.
"""
from __future__ import annotations


def test_omnigent_package_imports():
    import agent.omnigent

    assert agent.omnigent.__doc__ is not None
    # md5 patch is wired at import time
    import hashlib

    assert hashlib.md5(b"abc", usedforsecurity=False).hexdigest() == "900150983cd24fb0d6963f7d28e17f72"


def test_omnigent_reexports_required_symbols():
    from agent import omnigent

    required = {
        "AgentDef",
        "AgentTool",
        "AskRequest",
        "CancellableFunctionTool",
        "Connection",
        "Credentials",
        "DatabricksExecutor",
        "Executor",
        "ExecutorConfig",
        "ExecutorError",
        "ExecutorEvent",
        "FunctionPolicy",
        "FunctionTool",
        "History",
        "HandoffTool",
        "InheritedTool",
        "LocalOllamaExecutor",
        "MCPTool",
        "Memory",
        "MemoryConfig",
        "Message",
        "MessageSplit",
        "MockExecutor",
        "ParamDef",
        "Policy",
        "PolicyAction",
        "PolicyResult",
        "PolicyRuntimeContext",
        "PromptPolicy",
        "ReasoningChunk",
        "SelfAgentTool",
        "SessionState",
        "SkillTool",
        "TextChunk",
        "Tool",
        "ToolCallComplete",
        "ToolCallRequest",
        "ToolCallStatus",
        "TracingContext",
        "TurnCancelled",
        "TurnComplete",
        "classify_tool_result",
        "disable_tracing",
        "enable_tracing",
        "is_tracing_enabled",
        "iterate_blocking_stream",
        "load_agent_def",
        "split_transient_tail",
    }
    exported = set(omnigent.__all__)
    missing = required - exported
    assert not missing, f"missing re-exports: {sorted(missing)}"
    # No stale `Classification` alias
    assert "Classification" not in exported


def test_omnigent_env_compat_noop():
    from agent.omnigent._env_compat import mirror_legacy_env

    # Should not raise and should be idempotent.
    mirror_legacy_env()
    mirror_legacy_env()


def test_tracing_module_noop_when_disabled():
    from agent.omnigent import TracingContext

    ctx = TracingContext()
    span = ctx.start_agent_span("alex", "hi", model="gpt-x")
    ctx.end_agent_span(span, response="hello")
    tool = ctx.start_tool_span("browser_navigate", {"url": "https://example.com"})
    ctx.end_tool_span(tool, result={"ok": True}, parent_span=span)
    pol = ctx.start_policy_span("constitution", "tool_call", content={"tool": "x"})
    ctx.end_policy_span(pol, action="allow")
    # `active` is False because tracing was never enabled; just ensure
    # the calls are no-ops and don't raise.
    assert ctx.active is False
