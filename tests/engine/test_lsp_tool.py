"""Tests for agent/lsp_tool.py — LSP-style code intelligence.

Covers the ast baseline (outline, workspace symbols, diagnostics),
the jedi-backed operations (definition, references, hover), path
traversal refusal, and non-Python graceful degradation.
"""

from __future__ import annotations

import pytest

from agent import lsp_tool
from agent import workspace_tools as w

SAMPLE = '''\
"""A sample module."""


def greet(name: str) -> str:
    """Say hello."""
    return f"hello {name}"


class Greeter:
    def run(self) -> str:
        return greet("world")


RESULT = greet("bob")
'''


@pytest.fixture
def ws(monkeypatch, tmp_path):
    user = "alice@x.com"
    user_root = tmp_path / user
    (user_root / "work").mkdir(parents=True)
    (user_root / "work" / "sample.py").write_text(SAMPLE, encoding="utf-8")
    (user_root / "work" / "notes.txt").write_text("plain text", encoding="utf-8")
    (user_root / "work" / "broken.py").write_text("def x(:\n", encoding="utf-8")
    monkeypatch.setattr(w, "WORKSPACE_ROOT", str(tmp_path))
    return user


def test_outline_lists_symbols(ws):
    out = lsp_tool.outline(ws, "work/sample.py")
    assert out["supported"] is True
    names = {(s["name"], s["kind"]) for s in out["symbols"]}
    assert ("greet", "function") in names
    assert ("Greeter", "class") in names
    assert ("run", "method") in names


def test_outline_non_python_unsupported(ws):
    out = lsp_tool.outline(ws, "work/notes.txt")
    assert out["supported"] is False
    assert out["symbols"] == []


def test_outline_syntax_error_is_reported_not_raised(ws):
    out = lsp_tool.outline(ws, "work/broken.py")
    assert out["supported"] is True
    assert "syntax error" in out.get("error", "")


def test_diagnostics_flags_broken_file(ws):
    out = lsp_tool.diagnostics(ws, "work/broken.py")
    assert out["supported"] is True
    assert out["diagnostics"], "expected at least one diagnostic"
    assert out["diagnostics"][0]["severity"] == "error"


def test_diagnostics_clean_file_is_empty(ws):
    out = lsp_tool.diagnostics(ws, "work/sample.py")
    assert out["diagnostics"] == []


def test_workspace_symbols_search(ws):
    out = lsp_tool.workspace_symbols(ws, "greet")
    names = {s["name"] for s in out["symbols"]}
    assert "greet" in names and "Greeter" in names
    assert all(s["path"] == "work/sample.py" for s in out["symbols"])


def test_workspace_symbols_empty_query(ws):
    assert lsp_tool.workspace_symbols(ws, "  ")["symbols"] == []


def test_path_traversal_refused(ws):
    with pytest.raises(ValueError):
        lsp_tool.outline(ws, "../../etc/passwd.py")


def test_missing_file_raises(ws):
    with pytest.raises(FileNotFoundError):
        lsp_tool.outline(ws, "work/nope.py")


# ---- jedi-backed operations (skipped when jedi absent) --------------------

jedi = pytest.importorskip("jedi")


def test_definition_of_call_site(ws):
    # RESULT = greet("bob") — find `greet` on that line.
    line = SAMPLE.splitlines().index('RESULT = greet("bob")') + 1
    out = lsp_tool.definition(ws, "work/sample.py", line, 10)
    assert out["available"] is True
    defs = out["definitions"]
    assert defs and defs[0]["name"] == "greet"
    assert defs[0]["path"] == "work/sample.py"


def test_references_finds_all_uses(ws):
    out = lsp_tool.references(ws, "work/sample.py", 4, 5)  # def greet
    assert out["available"] is True
    assert len(out["references"]) >= 3  # def + 2 call sites


def test_hover_returns_signature_and_doc(ws):
    out = lsp_tool.hover(ws, "work/sample.py", 4, 5)
    assert out["available"] is True
    h = out["hover"]
    assert h["name"] == "greet"
    assert "Say hello." in h["docstring"]


def test_registry_has_lsp_tools():
    from agent.tools.registry import TOOL_REGISTRY

    for name in (
        "code_outline",
        "code_definition",
        "code_references",
        "code_hover",
        "code_diagnostics",
        "code_search_symbols",
    ):
        assert name in TOOL_REGISTRY
        assert TOOL_REGISTRY[name].capability == "code.read"
