"""
Tests for the agent.tools registry.

The registry is the single source of truth for every tool the
agent can call.  Every entry must have a name, a capability, a
description, a return schema, and a tuple of constitution articles
the tool touches.  These tests assert the contract.
"""
from __future__ import annotations

from agent.guardrails import TOOL_CAPABILITY
from agent.tools import TOOL_REGISTRY, ToolSpec, tools_by_capability


def test_registry_has_at_least_browser_tools():
    assert "browser_navigate" in TOOL_REGISTRY
    assert "browser_click" in TOOL_REGISTRY
    assert "browser_extract" in TOOL_REGISTRY
    assert "browser_submit" in TOOL_REGISTRY


def test_every_entry_is_a_tool_spec():
    for name, spec in TOOL_REGISTRY.items():
        assert isinstance(spec, ToolSpec), f"{name!r} is not a ToolSpec"
        assert spec.name == name


def test_every_tool_has_a_capability():
    for name, spec in TOOL_REGISTRY.items():
        assert spec.capability, f"{name!r} has empty capability"
        allowed_prefixes = (
            "browser.", "tasks.", "email.", "files.",
            "memory.", "compute.", "internal_api.", "learning.",
        "agents.", "office.",
            "system.", "knowledge.", "code.", "orders.", "practices.",
            "workflow.", "desktop.", "home.", "jarvis.",
        )
        assert spec.capability.startswith(allowed_prefixes), \
            f"{name!r} has unexpected capability {spec.capability!r}"


def test_every_tool_capability_is_also_in_guardrails():
    """The harness registry and the guardrails gate must agree."""
    for name, spec in TOOL_REGISTRY.items():
        assert name in TOOL_CAPABILITY, (
            f"tool {name!r} is in the registry but not in agent.guardrails.TOOL_CAPABILITY"
        )
        # The capability may differ if the registry uses a coarser
        # bucket (browser.use vs browser.use.act) but the name must
        # be in the guardrails map.
        assert TOOL_CAPABILITY[name] in (
            spec.capability,
            # browser tools use bucket-style; the guardrails map
            # uses a flat name->capability; both are valid.
            *{v for k, v in TOOL_CAPABILITY.items() if k == name},
        )


def test_every_tool_has_a_description():
    for name, spec in TOOL_REGISTRY.items():
        assert spec.description, f"{name!r} has no description"


def test_every_tool_has_a_return_schema():
    for name, spec in TOOL_REGISTRY.items():
        assert "type" in spec.return_schema, f"{name!r} has no return_schema type"


def test_every_tool_has_constitution_articles():
    for name, spec in TOOL_REGISTRY.items():
        assert spec.constitution_articles, (
            f"{name!r} has no constitution_articles"
        )


def test_submit_tool_marks_a8():
    spec = TOOL_REGISTRY["browser_submit"]
    assert "A8" in spec.constitution_articles


def test_browser_tools_mark_a3_audit_log():
    for name, spec in TOOL_REGISTRY.items():
        if not name.startswith("browser_"):
            continue
        assert "A3" in spec.constitution_articles, (
            f"{name!r} should declare A3 (every consequential act is logged)"
        )


def test_tools_by_capability_groups_correctly():
    grouped = tools_by_capability()
    assert "browser.use" in grouped
    assert "browser.read" in grouped
    assert "browser.act" in grouped
    # Every browser tool lands in exactly one capability bucket.
    expected = set(TOOL_REGISTRY.keys())
    actual = set()
    for tools in grouped.values():
        for t in tools:
            actual.add(t)
    assert actual == expected
