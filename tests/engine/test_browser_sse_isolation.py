"""Browser SSE events must reach ONLY the owning session's subscribers.

Regression for the cross-user leak where a process-global sink fanned every
browser event to every open stream (leaking URLs, extracted text, and
confirmation tokens across users).
"""
import pytest


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("ALEX_ENV", "dev")
    from alex import config, testing
    config._settings = None
    testing._fakes = None
    yield
    testing._fakes = None


def test_emit_routes_only_to_own_session(env):
    from agent import agentic_browser as ab
    from alex import testing
    from alex.events import tool_event

    fakes = testing.get_fakes()
    got_a, got_b = [], []
    fakes.browser_subscribers["sess-A"] = {got_a.append}
    fakes.browser_subscribers["sess-B"] = {got_b.append}

    ab._emit(tool_event("browser_navigate", {"url": "https://secret-A.example"}), "sess-A")

    assert len(got_a) == 1
    assert got_a[0].session_id == "sess-A"
    assert got_a[0].args["url"] == "https://secret-A.example"
    # The other user's stream must NOT receive it.
    assert got_b == []


def test_emit_stamps_session_id(env):
    from agent import agentic_browser as ab
    from alex import testing
    from alex.events import artifact_event

    testing.get_fakes()
    ev = artifact_event("browser://t1/screenshot", kind="png")
    assert ev.session_id is None
    ab._emit(ev, "sess-X")
    assert ev.session_id == "sess-X"
