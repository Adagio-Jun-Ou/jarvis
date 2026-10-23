"""Invariants for the email_summarize pipeline prompt.

email_summarize feeds UNTRUSTED email From/Subject/Body straight into an
LLM. Unlike email_intel it does not wrap the content in delimiters, so its
only defense against prompt-injection is an explicit instruction in the
system prompt. These tests lock that instruction in place so a future edit
can't silently strip it and let a crafted email dictate its own summary or
urgency (e.g. make a phishing mail look low-priority).

The module imports pyspark at top level, so we read the PROMPT constant
statically (matching the other pipeline tests) rather than importing it.
"""
import ast
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent.parent / "pipelines" / "email_summarize.py"


def _prompt() -> str:
    tree = ast.parse(_SRC.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "PROMPT" for t in node.targets
        ):
            assert isinstance(node.value, ast.Constant)
            return node.value.value
    raise AssertionError("PROMPT constant not found in email_summarize.py")


def test_prompt_declares_content_untrusted():
    p = _prompt().lower()
    assert "untrusted" in p
    assert "never follow instructions" in p


def test_prompt_defends_urgency_from_self_dictation():
    p = _prompt().lower()
    assert "urgency" in p
    assert "never let the email dictate" in p


def test_prompt_still_requires_single_json_object():
    p = _prompt()
    assert "json object" in p.lower()
    assert '"summary"' in p
    assert '"urgency"' in p
    assert '"topics"' in p
