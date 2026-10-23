"""Constitution-level tests for the browser operations.

These exercise the seam between :mod:`agent.agentic_browser` and
the constitution's two-phase submit (A8) and the no-mailto/no-tel
rule (operational guard on top of A8).  The constitution interceptor
itself is bypassed — these tests verify that the *tool surface*
emits the correct SSE event types and audit-log rows, which is the
unit the constitution can actually reason about.
"""
from __future__ import annotations

import pytest

from agent import agentic_browser as ab
from agent.browser_os import InMemoryBrowserOS
from alex.events import (
    EVENT_CONFIRMATION,
    EVENT_ERROR,
    EVENT_TOOL,
)
from alex.testing import Fakes, set_fakes


@pytest.fixture
def fakes_with_sink():
    recorded: list = []
    fakes = Fakes()
    fakes.browser_os = InMemoryBrowserOS()
    fakes.sse_sink = recorded.append
    # The browser tools write to media.alex.agent_actions (A3 audit).
    fakes.warehouse.register(
        "media.alex.agent_actions",
        [
            "action_id",
            "session_id",
            "tool",
            "capability",
            "user_email",
            "instance_scope",
            "input_json",
            "output_json",
            "ok",
            "ts",
        ],
    )
    set_fakes(fakes)
    yield fakes, recorded
    from alex import testing

    testing._fakes = None


def _seed(fakes, url: str = "alex://welcome") -> tuple[str, str]:
    s = fakes.browser_os.create_session(user_email="alice@x.com", instance_scope="default")
    t = fakes.browser_os.create_tab(s.session_id, url=url)
    return s.session_id, t.tab_id


def test_browser_navigate_emits_tool_event_and_audits(fakes_with_sink) -> None:
    fakes, sink = fakes_with_sink
    sid, tid = _seed(fakes)
    res = ab.browser_navigate(
        session_id=sid,
        tab_id=tid,
        url="alex://docs",
        user_email="alice@x.com",
    )
    assert res["ok"] is True
    types = [e.type for e in sink]
    assert EVENT_TOOL in types
    # Audit log row written.
    rows = fakes.warehouse.execute("SELECT tool, ok FROM media.alex.agent_actions").as_dicts()
    tools = {r["tool"] for r in rows}
    assert "browser_navigate" in tools


def test_browser_click_mailto_emits_error_event(fakes_with_sink) -> None:
    fakes, sink = fakes_with_sink
    sid, tid = _seed(fakes)
    res = ab.browser_click(
        session_id=sid,
        tab_id=tid,
        selector="Email support",
        user_email="alice@x.com",
    )
    assert res["ok"] is False
    # The error event was emitted via the sink.
    assert any(e.type == EVENT_ERROR for e in sink)
    # Audit row marked denied.
    rows = fakes.warehouse.execute("SELECT tool, ok FROM media.alex.agent_actions").as_dicts()
    assert any(r["tool"] == "browser_click" and r["ok"] == 0 for r in rows)


def test_browser_submit_first_phase_emits_confirmation(fakes_with_sink) -> None:
    fakes, sink = fakes_with_sink
    sid, tid = _seed(fakes)
    res = ab.browser_submit(
        session_id=sid,
        tab_id=tid,
        selector="form",
        confirmation_token=None,
        user_email="alice@x.com",
    )
    assert res["ok"] is True
    assert res["awaiting_confirmation"] is True
    # The token travels ONLY on the SSE confirmation event: the tool
    # result goes back to the model, and a model-readable token is a
    # self-approving one.
    assert "confirmation_token" not in res
    confirmations = [e for e in sink if e.type == EVENT_CONFIRMATION]
    assert confirmations
    assert confirmations[0].token
    assert confirmations[0].token != f"confirm_{sid}_{tid}"


def test_browser_submit_second_phase_requires_correct_token(fakes_with_sink) -> None:
    fakes, sink = fakes_with_sink
    sid, tid = _seed(fakes)
    res = ab.browser_submit(
        session_id=sid,
        tab_id=tid,
        selector="form",
        confirmation_token="not-a-valid-token",
        user_email="alice@x.com",
    )
    assert res["ok"] is False
    assert res["error"] == "invalid_confirmation_token"
    assert any(e.type == EVENT_ERROR for e in sink)


def test_browser_submit_second_phase_succeeds(fakes_with_sink) -> None:
    fakes, sink = fakes_with_sink
    sid, tid = _seed(fakes)
    ab.browser_submit(
        session_id=sid,
        tab_id=tid,
        selector="form",
        confirmation_token=None,
        user_email="alice@x.com",
    )
    confirmations = [e for e in sink if e.type == EVENT_CONFIRMATION]
    res = ab.browser_submit(
        session_id=sid,
        tab_id=tid,
        selector="form",
        confirmation_token=confirmations[-1].token,
        user_email="alice@x.com",
    )
    assert res["ok"] is True
    assert res["submitted"] is True
    # The submit actually executed: the staged selector was consumed
    # and the history records submit_execute, not just the approval.
    tab = fakes.browser_os.get_tab(sid, tid)
    assert "_pending_submit_selector" not in tab.form
    history = fakes.browser_os.export_history(sid)
    assert any(h.get("op") == "submit_execute" and h["outcome"] == "ok"
               for h in history)
    # Audit row recorded for both phases.
    rows = fakes.warehouse.execute(
        "SELECT tool, output_json FROM media.alex.agent_actions"
    ).as_dicts()
    submit_rows = [r for r in rows if r["tool"] == "browser_submit"]
    assert len(submit_rows) == 2
    assert "awaiting_confirmation" in submit_rows[0]["output_json"]
    assert submit_rows[1]["output_json"] == "ok"


def test_user_yield_blocks_navigate(fakes_with_sink) -> None:
    fakes, _sink = fakes_with_sink
    sid, tid = _seed(fakes)
    ab.browser_open_in_user_session(
        session_id=sid,
        tab_id=tid,
        mode="co-drive",
        user_email="alice@x.com",
    )
    res = ab.browser_navigate(
        session_id=sid,
        tab_id=tid,
        url="alex://docs",
        user_email="alice@x.com",
    )
    assert res["ok"] is False
    assert "user-controlled" in res["error"]


def test_extract_emits_artifact_event(fakes_with_sink) -> None:
    fakes, sink = fakes_with_sink
    sid, tid = _seed(fakes)
    res = ab.browser_extract(
        session_id=sid,
        tab_id=tid,
        selector="body",
        user_email="alice@x.com",
    )
    assert res["pii_redacted"] is False  # alex://welcome has no PII
    # artifact event emitted
    assert any(e.type == "artifact" and e.kind == "page_text" for e in sink)


def test_extract_redacts_pii_in_text(fakes_with_sink) -> None:
    from alex.events import artifact_event  # noqa: F401

    fakes, _sink = fakes_with_sink
    sid, tid = _seed(fakes)
    # Mutate the page HTML to embed a PII string.
    tab = fakes.browser_os.get_tab(sid, tid)
    tab.html = "<html><body>Contact: bob@secret.com</body></html>"
    res = ab.browser_extract(
        session_id=sid,
        tab_id=tid,
        selector="body",
        user_email="alice@x.com",
    )
    assert "[REDACTED:email]" in res["text"]
    assert res["pii_redacted"] is True
