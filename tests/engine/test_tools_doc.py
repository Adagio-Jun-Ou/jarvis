"""docs/TOOLS.md must stay in sync with the tool registry."""
from pathlib import Path

from agent.tools.registry import TOOL_REGISTRY

DOC = Path(__file__).resolve().parents[2] / "docs" / "engine" / "TOOLS.md"


def test_tools_doc_lists_every_registered_tool():
    assert DOC.exists(), "docs/TOOLS.md is missing"
    text = DOC.read_text(encoding="utf-8")
    missing = [name for name in TOOL_REGISTRY if f"`{name}`" not in text]
    assert not missing, (
        f"docs/TOOLS.md is out of date — missing tools: {missing}. "
        "Regenerate it from agent/tools/registry.py."
    )


def test_tools_doc_reports_correct_count():
    text = DOC.read_text(encoding="utf-8")
    assert f"{len(TOOL_REGISTRY)} tools" in text, (
        "docs/TOOLS.md tool count is stale; regenerate it."
    )
