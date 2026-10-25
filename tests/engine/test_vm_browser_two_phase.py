"""
Tests for the VM two-phase submit happy path.

Constitution A8 requires every externalising tool to stage the
action for human review.  ``browser_submit`` emits a
``confirmation_required`` event on the first call and only
navigates on the second call with a matching token.  These tests
verify the full flow through the VM seam.
"""
from __future__ import annotations

import pytest

from agent.omnigent import MockExecutor
from agent.vm import make_vm


@pytest.fixture
def vm():
    return make_vm(
        session_id="s1",
        user_email="alice@x.com",
        instance_scope="default",
        executor=MockExecutor(),
    )


def test_two_phase_submit_first_call_emits_confirmation_and_audits(vm) -> None:
    seen: list = []
    vm.subscribe_sse(seen.append)
    s = vm.open_browser_session()
    t = vm.open_browser_tab(s.session_id, url="alex://welcome")
    vm.type_browser(s.session_id, t.tab_id, selector="email", text="alice@x.com")
    seen.clear()

    res = vm.submit_browser(s.session_id, t.tab_id, selector="form")
    assert res["awaiting_confirmation"] is True
    # The tool result no longer carries the token: a token the model
    # can read is a token the model can hand back itself. The UI gets
    # it from the confirmation event instead.
    assert "confirmation_token" not in res
    confirmations = [
        e for e in seen if e.type == "confirmation_required"
    ]
    assert confirmations, "no confirmation event emitted"
    token = confirmations[-1].token
    # The token is random and single-use — never derivable from ids.
    assert token and token != f"confirm_{s.session_id}_{t.tab_id}"
    # The audit log shows the staged outcome.
    rows = vm._session.audit_log._wh.execute(
        "SELECT tool, output_json FROM media.alex.agent_actions"
    ).as_dicts()
    submit_rows = [r for r in rows if r["tool"] == "browser_submit"]
    assert len(submit_rows) == 1
    assert "awaiting_confirmation" in submit_rows[0]["output_json"]


def test_two_phase_submit_invalid_token_returns_error(vm) -> None:
    s = vm.open_browser_session()
    t = vm.open_browser_tab(s.session_id, url="alex://welcome")
    res = vm.submit_browser(
        s.session_id,
        t.tab_id,
        selector="form",
        confirmation_token="not-the-right-token",
    )
    assert res["ok"] is False
    assert res["error"] == "invalid_confirmation_token"


def test_two_phase_submit_valid_token_executes_and_audits(vm) -> None:
    seen: list = []
    vm.subscribe_sse(seen.append)
    s = vm.open_browser_session()
    t = vm.open_browser_tab(s.session_id, url="alex://welcome")
    vm.submit_browser(s.session_id, t.tab_id, selector="form")
    confirmations = [e for e in seen if e.type == "confirmation_required"]
    token = confirmations[-1].token
    res = vm.submit_browser(
        s.session_id,
        t.tab_id,
        selector="form",
        confirmation_token=token,
    )
    assert res["ok"] is True
    assert res["submitted"] is True
    rows = vm._session.audit_log._wh.execute(
        "SELECT tool, output_json FROM media.alex.agent_actions"
    ).as_dicts()
    submit_rows = [r for r in rows if r["tool"] == "browser_submit"]
    assert len(submit_rows) == 2
    assert submit_rows[1]["output_json"] == "ok"
    # The submit really executed: the staged selector was consumed and
    # the history records the execution, not just the approval.
    tab = vm._browser_os.get_tab(s.session_id, t.tab_id)
    assert "_pending_submit_selector" not in tab.form
    history = vm._browser_os.export_history(s.session_id)
    assert any(h.get("op") == "submit_execute" and h["outcome"] == "ok"
               for h in history)


def test_two_phase_submit_resume_after_user_yield(vm) -> None:
    """A user who has the wheel cannot submit; yielding back re-enables submit."""
    s = vm.open_browser_session()
    t = vm.open_browser_tab(s.session_id, url="alex://welcome")
    vm.mirror_browser(s.session_id, t.tab_id, mode="co-drive")
    res = vm.submit_browser(s.session_id, t.tab_id, selector="form")
    assert res["ok"] is False
    # User yields.
    vm.mirror_browser(s.session_id, t.tab_id, mode="yield")
    res2 = vm.submit_browser(s.session_id, t.tab_id, selector="form")
    assert res2["awaiting_confirmation"] is True
