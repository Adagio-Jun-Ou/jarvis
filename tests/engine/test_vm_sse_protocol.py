"""
Tests for the VM SSE protocol.

Every public VM operation must produce at least one
:class:`alex.events.Event` through the event pump, and the event
type must be one of the eight canonical SSE types.  These tests
walk the surface to confirm the protocol contract.
"""
from __future__ import annotations

import pytest

from agent.omnigent import MockExecutor
from agent.vm import make_vm
from alex.events import ALL_TYPES


@pytest.fixture
def vm():
    return make_vm(
        session_id="s1",
        user_email="alice@x.com",
        instance_scope="default",
        executor=MockExecutor(),
    )


def test_navigate_emits_tool_and_artifact_events(vm) -> None:
    seen: list = []
    vm.subscribe_sse(seen.append)
    s = vm.open_browser_session()
    t = vm.open_browser_tab(s.session_id, url="alex://welcome")
    seen.clear()  # drop the open artifacts
    res = vm.navigate_browser(s.session_id, t.tab_id, url="alex://docs")
    assert res["ok"] is True
    types = {e.type for e in seen}
    assert "tool" in types
    assert "artifact" in types
    assert all(e.type in ALL_TYPES for e in seen)


def test_click_mailto_emits_error_event(vm) -> None:
    seen: list = []
    vm.subscribe_sse(seen.append)
    s = vm.open_browser_session()
    t = vm.open_browser_tab(s.session_id, url="alex://welcome")
    seen.clear()
    res = vm.click_browser(s.session_id, t.tab_id, selector="Email support")
    assert res["ok"] is False
    assert any(e.type == "error" for e in seen)


def test_extract_emits_artifact(vm) -> None:
    seen: list = []
    vm.subscribe_sse(seen.append)
    s = vm.open_browser_session()
    t = vm.open_browser_tab(s.session_id, url="alex://welcome")
    seen.clear()
    res = vm.extract_browser(s.session_id, t.tab_id, selector="body")
    assert "text" in res
    assert any(e.type == "artifact" for e in seen)


def test_screenshot_emits_artifact(vm) -> None:
    seen: list = []
    vm.subscribe_sse(seen.append)
    s = vm.open_browser_session()
    t = vm.open_browser_tab(s.session_id, url="alex://welcome")
    seen.clear()
    res = vm.screenshot_browser(s.session_id, t.tab_id)
    assert "placeholder" in res or "png" in res
    assert any(e.type == "artifact" for e in seen)


def test_submit_first_phase_emits_confirmation(vm) -> None:
    seen: list = []
    vm.subscribe_sse(seen.append)
    s = vm.open_browser_session()
    t = vm.open_browser_tab(s.session_id, url="alex://welcome")
    vm.type_browser(s.session_id, t.tab_id, selector="email", text="alice@x.com")
    seen.clear()
    res = vm.submit_browser(s.session_id, t.tab_id, selector="form")
    assert res["awaiting_confirmation"] is True
    assert any(e.type == "confirmation_required" for e in seen)


def test_submit_second_phase_emits_no_confirmation(vm) -> None:
    seen: list = []
    vm.subscribe_sse(seen.append)
    s = vm.open_browser_session()
    t = vm.open_browser_tab(s.session_id, url="alex://welcome")
    vm.submit_browser(s.session_id, t.tab_id, selector="form")
    # Phase 1's confirmation event is the token's ONLY channel to the
    # UI — the tool result carries no token the model could redeem.
    confirmations = [e for e in seen if e.type == "confirmation_required"]
    assert confirmations
    token = confirmations[-1].token
    seen.clear()
    second = vm.submit_browser(
        s.session_id,
        t.tab_id,
        selector="form",
        confirmation_token=token,
    )
    assert second["submitted"] is True
    # No new confirmation event was emitted on the second phase.
    assert not any(e.type == "confirmation_required" for e in seen)


def test_mirror_to_co_drive_emits_artifact(vm) -> None:
    seen: list = []
    vm.subscribe_sse(seen.append)
    s = vm.open_browser_session()
    t = vm.open_browser_tab(s.session_id, url="alex://welcome")
    seen.clear()
    res = vm.mirror_browser(s.session_id, t.tab_id, mode="co-drive")
    assert res["agent_piloting"] is False
    assert any(e.type == "artifact" for e in seen)


def test_run_turn_streams_text_and_done_with_mock(vm) -> None:
    """A mock executor that returns text + a turn complete drives
    the full SSE pipeline through the VM."""
    from agent.omnigent import TextChunk, TurnComplete

    mock = MockExecutor()
    mock.enqueue_events([TextChunk(text="hello"), TurnComplete(response="hello")])
    vm._executor = mock
    seen: list = []
    vm.subscribe_sse(seen.append)
    last, _ = vm.run_turn(text="hi")
    assert last == "hello"
    types = [e.type for e in seen]
    assert "text" in types
    # `done` is the app layer's; the VM/bridge never emit it so
    # post-turn events (e.g. `memory`) aren't cut off by clients.
    assert "done" not in types


def test_every_event_has_a_canonical_type(vm) -> None:
    """Sweep every public op and assert the event types are all
    in the canonical :data:`alex.events.ALL_TYPES` set."""
    s = vm.open_browser_session()
    t = vm.open_browser_tab(s.session_id, url="alex://welcome")
    seen: list = []
    vm.subscribe_sse(seen.append)
    vm.navigate_browser(s.session_id, t.tab_id, url="alex://docs")
    vm.type_browser(s.session_id, t.tab_id, selector="email", text="alice@x.com")
    vm.extract_browser(s.session_id, t.tab_id)
    vm.screenshot_browser(s.session_id, t.tab_id)
    vm.submit_browser(s.session_id, t.tab_id, selector="form")
    vm.mirror_browser(s.session_id, t.tab_id, mode="co-drive")
    for e in seen:
        assert e.type in ALL_TYPES, f"unknown event type {e.type!r}"


def test_close_tab_emits_artifact(vm) -> None:
    seen: list = []
    vm.subscribe_sse(seen.append)
    s = vm.open_browser_session()
    t = vm.open_browser_tab(s.session_id, url="alex://welcome")
    seen.clear()
    vm.close_browser_tab(s.session_id, t.tab_id)
    assert any(e.type == "artifact" for e in seen)
