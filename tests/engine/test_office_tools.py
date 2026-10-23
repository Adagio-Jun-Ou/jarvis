"""Office-document generation tools (Excel / PowerPoint / Word)."""
from __future__ import annotations

import os
import zipfile

import pytest

from agent.tools import office


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("ALEX_WORKSPACE_ROOT", str(tmp_path))
    return str(tmp_path)


U = "demo@alex.dev"


def _abs(ws, rel):
    return os.path.join(ws, U, rel)


def test_create_spreadsheet_writes_valid_xlsx(workspace):
    r = office.create_spreadsheet(
        "budget",
        headers=["Channel", "Spend"],
        rows=[["Search", 1200000], ["Social", "800000"]],
        user_email=U,
    )
    assert r["ok"] and r["path"] == "outputs/budget.xlsx"
    assert r["download"] == "/api/files/download?path=outputs/budget.xlsx"
    p = _abs(workspace, "outputs/budget.xlsx")
    assert os.path.getsize(p) > 0
    assert zipfile.is_zipfile(p)  # xlsx is a zip container


def test_read_spreadsheet_roundtrip_coerces_numbers(workspace):
    office.create_spreadsheet(
        "data", headers=["A", "B"], rows=[["x", "42"], ["y", 7]], user_email=U
    )
    rr = office.read_spreadsheet("outputs/data.xlsx", user_email=U)
    assert rr["ok"]
    assert rr["rows"][0] == ["A", "B"]
    # "42" was coerced to a number on write.
    assert rr["rows"][1] == ["x", 42]
    assert rr["rows"][2] == ["y", 7]


def test_create_presentation_writes_valid_pptx(workspace):
    r = office.create_presentation(
        "deck",
        title="Q1 Review",
        slides=[{"title": "Highlights", "bullets": ["Approved", "On track"]}],
        user_email=U,
    )
    assert r["ok"]
    p = _abs(workspace, "outputs/deck.pptx")
    assert zipfile.is_zipfile(p)


def test_create_document_writes_valid_docx(workspace):
    r = office.create_document(
        "memo",
        title="Memo",
        body="First.\n\nSecond.",
        sections=[{"heading": "Detail", "text": "More."}],
        user_email=U,
    )
    assert r["ok"]
    p = _abs(workspace, "outputs/memo.docx")
    assert zipfile.is_zipfile(p)


def test_filename_cannot_escape_outputs(workspace):
    # A traversal filename is basenamed into outputs/.
    r = office.create_document("../../etc/passwd", body="x", user_email=U)
    assert r["ok"]
    # Sanitized to a flat name inside outputs/ — no traversal, no separators.
    assert r["path"].startswith("outputs/")
    assert ".." not in r["path"]
    assert r["path"].count("/") == 1
    # The real file lives under the user's outputs dir, nowhere else.
    p = _abs(workspace, r["path"])
    assert os.path.isfile(p)


def test_office_tools_registered_and_capable():
    from agent.guardrails import TOOL_CAPABILITY
    from agent.tools.registry import TOOL_REGISTRY

    for name in (
        "create_spreadsheet",
        "read_spreadsheet",
        "create_presentation",
        "create_document",
    ):
        assert name in TOOL_REGISTRY
        assert TOOL_CAPABILITY[name] == TOOL_REGISTRY[name].capability


@pytest.mark.parametrize("rows", [
    [["Jan", 100], ["Feb", 150]],            # proper list of row-lists
    ["Jan", 100, "Feb", 150],                # flat list, chunked by headers
    '[["Jan",100],["Feb",150]]',             # JSON string
    ["Jan,100", "Feb,150"],                  # CSV row strings
    [{"Month": "Jan", "Revenue": 100}, {"Month": "Feb", "Revenue": 150}],  # dicts
])
def test_spreadsheet_row_shapes_normalized(workspace, rows):
    r = office.create_spreadsheet("shp", headers=["Month", "Revenue"], rows=rows, user_email=U)
    assert r["ok"]
    got = office.read_spreadsheet("outputs/shp.xlsx", user_email=U)["rows"]
    assert got[0] == ["Month", "Revenue"]
    assert ["Jan", 100] in got and ["Feb", 150] in got


def test_slug_preserves_and_dedupes_extension():
    assert office._slug("tax_note", ".docx") == "tax_note.docx"
    assert office._slug("tax_note.docx", ".docx") == "tax_note.docx"
    assert office._slug("report.xlsx.docx", ".docx") == "report.docx"
    assert office._slug("q1/budget", ".xlsx") == "budget.xlsx"
