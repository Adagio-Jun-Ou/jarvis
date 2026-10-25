"""Mid-run steering — a crew member's note reaching the running loop.

The delivery contract is the thing under test: `steer()` returns True
ONLY when the note will actually be drained by the loop. Every False is
a promise too — the caller then knows the message survives as a thread
post that the *next* run's prompt reads, not as a delivered correction.
A steer that lies (accepted but never read) is worse than no steering,
because a person believes the agent heard them.
"""

from __future__ import annotations

import pytest

from agent.omnigent import MockExecutor
from agent.vm import make_vm


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("ALEX_ENV", "dev")
    from alex import config, testing

    config._settings = None
    testing._fakes = None
    yield
    config._settings = None
    testing._fakes = None


@pytest.fixture
def vm(env):
    return make_vm(
        session_id="steer-1", user_email="demo@alex.dev", executor=MockExecutor()
    )


def test_a_note_with_no_turn_running_is_refused(vm):
    """Nothing is listening, so claiming delivery would be a lie."""
    assert vm.steer("focus on billing") is False


def test_a_note_during_a_turn_is_accepted_and_drained(vm):
    vm._turn_active = True
    try:
        assert vm.steer("focus on billing", author="alice@x.com") is True
        assert vm.drain_steering() == [("alice@x.com", "focus on billing")]
        # Drained means drained: a second drain must not replay it.
        assert vm.drain_steering() == []
    finally:
        vm._turn_active = False


def test_notes_are_delivered_in_order(vm):
    vm._turn_active = True
    try:
        vm.steer("first", author="a")
        vm.steer("second", author="b")
        assert [n for _, n in vm.drain_steering()] == ["first", "second"]
    finally:
        vm._turn_active = False


def test_an_empty_note_is_refused(vm):
    vm._turn_active = True
    try:
        assert vm.steer("   ") is False
        assert vm.drain_steering() == []
    finally:
        vm._turn_active = False


def test_ending_the_turn_closes_the_mailbox_atomically(vm):
    """A note accepted after the final drain would be queued forever and
    its author told it was delivered. The turn flag and the mailbox are
    closed as one step so that window does not exist."""
    vm._turn_active = True
    vm.steer("late but in time", author="a")

    leftover = vm._end_turn_steering()

    assert [n for _, n in leftover] == ["late but in time"]
    assert vm.steer("too late", author="a") is False
    assert vm.drain_steering() == []


def test_a_closed_vm_refuses_notes(vm):
    vm._turn_active = True
    vm.close()
    assert vm.steer("anyone there?") is False


def test_the_task_board_forwards_a_post_into_the_live_run(env):
    """The wiring: a crew member's thread post reaches the running VM,
    and the caller learns whether it did."""
    from agent import taskboard as tb

    class FakeVM:
        def __init__(self):
            self.notes = []

        def steer(self, text, *, author="human"):
            self.notes.append((author, text))
            return True

    run = tb.TaskRun(
        run_id="r1", task_id="t1", session_id="s1", started_by="dana@alex.dev"
    )
    fake = FakeVM()
    run._vm = fake

    assert run.steer("alice@x.com", "check the Q2 numbers too") is True
    assert fake.notes == [("alice@x.com", "check the Q2 numbers too")]


def test_a_finished_run_does_not_pretend_to_forward(env):
    from agent import taskboard as tb

    run = tb.TaskRun(
        run_id="r2", task_id="t2", session_id="s2", started_by="dana@alex.dev"
    )
    run.status = "done"
    run._vm = object()  # even with a stale VM reference

    assert run.steer("alice@x.com", "one more thing") is False


def test_steering_an_unknown_task_is_false_not_an_error(env):
    from agent import taskboard as tb

    tb.reset_runs()
    assert tb.steer_task("no-such-task", "a", "hello") is False
