"""Tests for agent/taskboard.py — the shared queue people and Alex work.

Covers board visibility, the crew, the thread, and a full AI run driven
by a stub VM (no model call): assignment, streaming, late-joiner replay,
completion bookkeeping, and the rule that Alex never closes its own task.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timedelta, timezone

import pytest

from agent import taskboard as tb


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("ALEX_ENV", "dev")
    from alex import config, testing

    config._settings = None
    testing._fakes = None
    tb.reset_runs()
    yield
    config._settings = None
    testing._fakes = None
    tb.reset_runs()


class P:
    """A minimal principal."""

    def __init__(self, email: str, is_admin: bool = False):
        self.email = email
        self.is_admin = is_admin


@pytest.fixture
def sql(env):
    from alex.testing import get_fakes

    wh = get_fakes().warehouse
    wh.register(
        "media.alex.tasks",
        ["id", "owner_email", "assignee_email", "title", "description",
         "status", "priority", "due_at", "source", "source_ref",
         "created_at", "updated_at", "closed_at"],
        pk=["id"],
    )
    # Imported rather than restated. These columns already existed in two
    # places — governance/taskboard.sql and the dev seed — and a third
    # copy here is how a schema change passes its own tests while
    # breaking the application.
    from app._dev_seed import (
        TASK_CREW_COLS,
        TASK_MESSAGES_COLS,
        TASK_RUNS_COLS,
    )

    wh.register("media.control.task_crew", TASK_CREW_COLS)
    wh.register(
        "media.control.task_messages", TASK_MESSAGES_COLS, pk=["message_id"]
    )
    wh.register("media.control.task_runs", TASK_RUNS_COLS, pk=["run_id"])
    # The memory + graph mirrors the run-manifest writes go to. Same
    # column sets the dev seed uses, imported for the same reason as
    # the task tables above.
    from app._dev_seed import KG_MIRROR_COLS, VECTOR_MEMORY_MIRROR_COLS

    wh.register("media.control.kg_mirror", KG_MIRROR_COLS)
    wh.register(
        "media.control.vector_memory_mirror", VECTOR_MEMORY_MIRROR_COLS
    )

    def _sql(stmt: str, params: dict | None = None):
        res = wh.execute(stmt, params or {})
        as_dicts = getattr(res, "as_dicts", None)
        return as_dicts() if callable(as_dicts) else list(res or [])

    return _sql


# ---- status lanes --------------------------------------------------------


def test_legacy_statuses_map_onto_board_lanes():
    assert tb.normalise_status("open") == "todo"
    assert tb.normalise_status("In Progress") == "in_progress"
    assert tb.normalise_status("cancelled") == "done"
    assert tb.normalise_status("nonsense") == "todo"


# ---- creation and visibility --------------------------------------------


def test_create_task_seats_the_owner_and_opens_the_thread(sql):
    alice = P("alice@x.com")
    out = tb.create_task(sql, alice, title="Write the register", description="d")
    assert out["task_id"].startswith("t-")
    crew = tb.crew_for(sql, out["task_id"])
    assert crew == [{"member": "alice@x.com", "role": "owner"}]
    thread = tb.messages(sql, out["task_id"])
    assert thread[0]["kind"] == "system"
    assert "created this task" in thread[0]["body"]


def test_create_task_requires_a_title(sql):
    with pytest.raises(ValueError):
        tb.create_task(sql, P("a@x.com"), title="   ")


def test_a_task_is_invisible_to_someone_not_on_it(sql):
    tb.create_task(sql, P("alice@x.com"), title="Private work")
    assert tb.visible_tasks(sql, P("alice@x.com"))
    assert tb.visible_tasks(sql, P("bob@x.com")) == []


def test_a_collaborator_sees_the_task(sql):
    out = tb.create_task(
        sql, P("alice@x.com"), title="Shared work", crew=["bob@x.com"]
    )
    seen = tb.visible_tasks(sql, P("bob@x.com"))
    assert [t["task_id"] for t in seen] == [out["task_id"]]
    assert {c["member"] for c in seen[0]["crew"]} == {"alice@x.com", "bob@x.com"}


def test_admins_see_the_whole_board(sql):
    tb.create_task(sql, P("alice@x.com"), title="Alice's work")
    assert len(tb.visible_tasks(sql, P("root@x.com", is_admin=True))) == 1


def test_board_is_ordered_by_lane_then_priority(sql):
    a = P("alice@x.com")
    tb.create_task(sql, a, title="done thing", status="done", priority=1)
    tb.create_task(sql, a, title="low todo", status="todo", priority=5)
    tb.create_task(sql, a, title="urgent todo", status="todo", priority=1)
    titles = [t["title"] for t in tb.visible_tasks(sql, a)]
    assert titles == ["urgent todo", "low todo", "done thing"]


def test_can_act_on_matches_visibility(sql):
    out = tb.create_task(sql, P("alice@x.com"), title="t")
    assert tb.can_act_on(sql, P("alice@x.com"), out["task_id"])
    assert not tb.can_act_on(sql, P("bob@x.com"), out["task_id"])


# ---- crew and thread -----------------------------------------------------


def test_add_crew_is_idempotent(sql):
    out = tb.create_task(sql, P("alice@x.com"), title="t")
    tb.add_crew(sql, out["task_id"], "bob@x.com")
    tb.add_crew(sql, out["task_id"], "bob@x.com")
    members = [c["member"] for c in tb.crew_for(sql, out["task_id"])]
    assert members.count("bob@x.com") == 1


def test_thread_is_shared_by_humans_and_the_agent(sql):
    out = tb.create_task(sql, P("alice@x.com"), title="t")
    tid = out["task_id"]
    tb.post_message(sql, tid, author="alice@x.com", body="what's the deadline?")
    tb.post_message(sql, tid, author=tb.AGENT_ACTOR, body="March 31.", kind="agent")
    thread = tb.messages(sql, tid)
    assert [m["kind"] for m in thread] == ["system", "human", "agent"]


def test_empty_message_is_rejected(sql):
    out = tb.create_task(sql, P("alice@x.com"), title="t")
    with pytest.raises(ValueError):
        tb.post_message(sql, out["task_id"], author="a@x.com", body="  ")


# ---- the working prompt --------------------------------------------------


def test_task_prompt_carries_details_crew_and_discussion():
    task = {
        "title": "Draft the risk register",
        "description": "Pull risks from the brief.",
        "crew": [{"member": "alice@x.com"}, {"member": tb.AGENT_ACTOR}],
    }
    thread = [
        {"author": "system", "body": "created", "kind": "system"},
        {"author": "alice@x.com", "body": "focus on billing", "kind": "human"},
    ]
    prompt = tb.build_task_prompt(task, thread)
    assert "Draft the risk register" in prompt
    assert "Pull risks from the brief." in prompt
    assert "alice@x.com" in prompt
    assert "focus on billing" in prompt
    assert tb.AGENT_ACTOR not in prompt.split("TEAM ON THIS TASK:")[1].split("\n")[0]
    assert "created" not in prompt  # system chatter is not instruction


# ---- running a task with Alex -------------------------------------------


class StubVM:
    """A VM that emits a couple of steps and returns a summary."""

    def __init__(self, session_id: str, user_email: str, fail: bool = False):
        self.session_id = session_id
        self.user_email = user_email
        self.closed = False
        self._subs: list = []
        self._fail = fail
        self.steered: list = []
        self.turn_active = False

    def subscribe_sse(self, fn):
        self._subs.append(fn)
        return lambda: None

    def steer(self, text, author="human"):
        """Steerable stub: records the note and reports whether a turn
        was active. Tests assert on this to prove forwarding."""
        if not getattr(self, "turn_active", False):
            return False
        self.steered.append((author, text))
        return True

    def run_turn(self, text: str):
        for sub in self._subs:
            sub({"type": "tool", "tool": "search_knowledge", "args": {"q": "risk"}})
            sub({"type": "text", "text": "working"})
        if self._fail:
            raise RuntimeError("model exploded")
        return "Drafted the register; two risks need a human decision.", []

    def close(self):
        self.closed = True


def _factory(**kw):
    made = {}

    def make(session_id: str, user_email: str):
        vm = StubVM(session_id, user_email, **kw)
        made["vm"] = vm
        return vm

    make.made = made  # type: ignore[attr-defined]
    return make


def test_assigning_to_alex_runs_it_and_streams_every_step(sql):
    alice = P("alice@x.com")
    tid = tb.create_task(sql, alice, title="Draft the register")["task_id"]
    factory = _factory()
    run = tb.start_run(sql, alice, tid, vm_factory=factory, background=False)

    assert run.status == "done"
    assert run.session_id == f"task-{tid}"
    types = [e["type"] for e in run.events]
    assert types[0] == "run_started" and types[-1] == "run_finished"
    assert "tool" in types
    assert factory.made["vm"].closed, "the run must release its VM"


def test_a_finished_run_lands_in_review_not_done(sql):
    """Alex never closes its own task — a human signs off."""
    alice = P("alice@x.com")
    tid = tb.create_task(sql, alice, title="t")["task_id"]
    tb.start_run(sql, alice, tid, vm_factory=_factory(), background=False)
    task = tb.get_task(sql, alice, tid)
    assert task["status"] == "review"


def test_the_run_posts_its_summary_to_the_thread(sql):
    alice = P("alice@x.com")
    tid = tb.create_task(sql, alice, title="t")["task_id"]
    tb.start_run(sql, alice, tid, vm_factory=_factory(), background=False)
    bodies = [m["body"] for m in tb.messages(sql, tid) if m["kind"] == "agent"]
    assert any("Picked this up" in b for b in bodies)
    assert any("Drafted the register" in b for b in bodies)


def test_alex_joins_the_crew_when_assigned(sql):
    alice = P("alice@x.com")
    tid = tb.create_task(sql, alice, title="t")["task_id"]
    tb.start_run(sql, alice, tid, vm_factory=_factory(), background=False)
    task = tb.get_task(sql, alice, tid)
    assert task["ai_assigned"] is True
    assert tb.AGENT_ACTOR in {c["member"] for c in task["crew"]}


def test_a_failing_run_is_recorded_and_returns_the_task(sql):
    alice = P("alice@x.com")
    tid = tb.create_task(sql, alice, title="t")["task_id"]
    run = tb.start_run(sql, alice, tid, vm_factory=_factory(fail=True), background=False)
    assert run.status == "failed"
    assert "model exploded" in run.summary
    # A failed run must not strand the task in progress.
    assert tb.get_task(sql, alice, tid)["status"] == "todo"


def test_someone_not_on_the_task_cannot_run_it(sql):
    tid = tb.create_task(sql, P("alice@x.com"), title="t")["task_id"]
    with pytest.raises(PermissionError):
        tb.start_run(sql, P("mallory@x.com"), tid, vm_factory=_factory(),
                     background=False)


def test_a_late_watcher_sees_the_whole_run(sql):
    alice = P("alice@x.com")
    tid = tb.create_task(sql, alice, title="t")["task_id"]
    run = tb.start_run(sql, alice, tid, vm_factory=_factory(), background=False)
    seen: list = []
    unsubscribe = run.subscribe(seen.append)
    assert [e["type"] for e in seen] == [e["type"] for e in run.events]
    unsubscribe()
    run.emit({"type": "later"})
    assert not any(e["type"] == "later" for e in seen)


def test_starting_a_second_run_returns_the_live_one(sql):
    alice = P("alice@x.com")
    tid = tb.create_task(sql, alice, title="t")["task_id"]
    run = tb.TaskRun(
        run_id="run-1", task_id=tid, session_id="s", started_by=alice.email
    )
    tb._register(run)
    again = tb.start_run(sql, alice, tid, vm_factory=_factory(), background=False)
    assert again is run, "a task must not be worked twice at once"


def test_stop_run_marks_it_stopped(sql):
    alice = P("alice@x.com")
    tid = tb.create_task(sql, alice, title="t")["task_id"]
    run = tb.TaskRun(
        run_id="run-2", task_id=tid, session_id="s", started_by=alice.email
    )
    tb._register(run)
    assert tb.stop_run(sql, tid) is True
    assert run.status == "stopped"
    assert tb.stop_run(sql, tid) is False


def test_a_stopped_run_cannot_be_refinished_as_done(sql):
    """stop_run closes the run from the route thread while the worker
    is still in vm.run_turn; the worker's later _finish used to
    overwrite 'stopped' with 'done' and remember work the crew
    halted as completed. The first verdict stands."""
    alice = P("alice@x.com")
    tid = tb.create_task(sql, alice, title="t")["task_id"]
    run = tb.TaskRun(
        run_id="run-3", task_id=tid, session_id="s", started_by=alice.email
    )
    tb._register(run)
    tb._finish(sql, run, "stopped", "Stopped by a member of the crew.")

    # The worker thread returns later and tries to finish again.
    tb._finish(sql, run, "done", "Drafted the register after all.")

    assert run.status == "stopped", "the crew's stop was overwritten"
    assert "Stopped by a member" in run.summary

    # And the manifest does not record a run the crew halted as
    # completed work.
    from agent import vector_memory
    from agent.memory import connection

    with connection() as conn:
        beads = vector_memory.recall(conn, alice.email, query_text="t", k=10)
    assert not [b for b in beads if b.kind == "work"]


# ---- mid-run steering ------------------------------------------------------
# A crew member's thread post on a task with a live run is forwarded
# into the running VM at the next safe point, so course correction
# does not wait for the next run. When nothing is live the post stands
# as an ordinary message and nothing pretends otherwise.


class LiveStubVM(StubVM):
    """A stub whose run_turn spins long enough to be steered."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.gate = threading.Event()

    def run_turn(self, text: str):
        self.turn_active = True
        try:
            for sub in self._subs:
                sub({"type": "tool", "tool": "search_knowledge", "args": {}})
            self.gate.wait(timeout=5)  # "working" while the crew talks
            for sub in self._subs:
                sub({"type": "text", "text": "done"})
            return "Steered and finished.", []
        finally:
            self.turn_active = False


def test_a_crew_post_reaches_the_live_run(sql):
    alice = P("alice@x.com")
    tid = tb.create_task(sql, alice, title="t")["task_id"]
    factory = _factory()
    factory.vm_cls = LiveStubVM  # type: ignore[attr-defined]

    def make(session_id, user_email):
        vm = LiveStubVM(session_id, user_email)
        factory.made["vm"] = vm
        return vm

    tb.start_run(sql, alice, tid, vm_factory=make, background=True)
    vm = factory.made["vm"]
    try:
        # Wait for the run to actually be working before steering: the
        # worker thread sets _vm and enters its turn asynchronously.
        deadline = time.monotonic() + 5
        while not vm.turn_active and time.monotonic() < deadline:
            time.sleep(0.02)
        assert vm.turn_active, "stub never started its turn"
        assert tb.steer_task(tid, "bob@x.com", "focus on billing")
        assert vm.steered == [("bob@x.com", "focus on billing")]
    finally:
        vm.gate.set()
        run_thread = [t for t in threading.enumerate() if t.name.startswith("taskrun")]
        for t in run_thread:
            t.join(timeout=5)


def test_a_post_with_no_live_run_is_not_steered(sql):
    alice = P("alice@x.com")
    tid = tb.create_task(sql, alice, title="t")["task_id"]
    assert tb.steer_task(tid, alice.email, "whenever you get to it") is False


def test_steering_a_finished_run_returns_false(sql):
    alice = P("alice@x.com")
    tid = tb.create_task(sql, alice, title="t")["task_id"]
    tb.start_run(sql, alice, tid, vm_factory=_factory(), background=False)
    # The run is done; a late note is a thread post, not a correction.
    assert tb.steer_task(tid, alice.email, "one more thing") is False


# ---- the execution manifest -------------------------------------------------
# A completed run is not just a summary in a thread: it is remembered.
# The manifest makes "how did we handle this last time?" a recall
# rather than an archaeology dig through task_runs.steps.


def test_a_completed_run_is_remembered(sql):
    from agent import vector_memory
    from agent.memory import connection

    alice = P("alice@x.com")
    tid = tb.create_task(sql, alice, title="Draft the register")["task_id"]
    tb.start_run(sql, alice, tid, vm_factory=_factory(), background=False)

    with connection() as conn:
        beads = vector_memory.recall(conn, alice.email, query_text="register", k=5)
    work = [b for b in beads if b.kind == "work"]
    assert work, "the finished run left no memory"
    bead = work[0]
    assert "Drafted the register" in bead.content, "the result is in the bead"
    assert bead.source_ref.startswith("task-run:"), "the bead cites its run"

    # And the mirror — the surface humans and /api/sdk/recall read.
    rows = sql(
        "SELECT kind, content FROM media.control.vector_memory_mirror "
        "WHERE owner_email = :u",
        {"u": alice.email},
    )
    assert any(r["kind"] == "work" for r in rows), "the manifest never reached the mirror"


def test_a_completed_run_links_the_graph(sql):
    alice = P("alice@x.com")
    tid = tb.create_task(sql, alice, title="Draft the register")["task_id"]
    tb.start_run(sql, alice, tid, vm_factory=_factory(), background=False)

    rows = sql(
        "SELECT src, rel, dst FROM media.control.kg_mirror WHERE owner_email = :u",
        {"u": alice.email},
    )
    rels = {(r["src"], r["rel"]) for r in rows}
    assert (f"task:{tid}", "executed_by") in rels, "the graph does not link the run"
    assert (alice.email, "worked_on") in rels, "the crew is not linked to the task"


def test_a_failed_run_is_not_remembered_as_completed_work(sql):
    """Only successful work goes into the manifest — a failed run in
    memory would be recalled as though it had been done."""
    from agent import vector_memory

    alice = P("alice@x.com")
    tid = tb.create_task(sql, alice, title="t")["task_id"]
    tb.start_run(sql, alice, tid, vm_factory=_factory(fail=True), background=False)

    from agent.memory import connection

    with connection() as conn:
        beads = vector_memory.recall(conn, alice.email, query_text="t", k=10)
    assert not [b for b in beads if b.kind == "work"], (
        "a failed run was remembered as completed work"
    )


def test_the_next_run_prompt_includes_past_work(sql):
    """Resuming or repeating a task starts from what the last run
    learned — the recall note in the prompt is what carries it."""

    alice = P("alice@x.com")
    tid = tb.create_task(sql, alice, title="Draft the register")["task_id"]
    tb.start_run(sql, alice, tid, vm_factory=_factory(), background=False)

    task = tb.get_task(sql, alice, tid)
    note = tb._past_work_note(sql, alice.email, task)
    assert "Drafted the register" in note, "past work did not reach the next prompt"
    assert "check it still applies" in note, "the note lost its caveat"


def test_one_broken_manifest_table_does_not_cost_the_others(sql):
    """The manifest writes are four separate tables. A missing mirror
    used to abort the whole block — bead stored, graph silently empty,
    only a debug log to say so. Each write now fails alone."""
    from alex.testing import get_fakes

    alice = P("alice@x.com")
    tid = tb.create_task(sql, alice, title="t")["task_id"]

    get_fakes().warehouse.drop("media.control.vector_memory_mirror")

    tb.start_run(sql, alice, tid, vm_factory=_factory(), background=False)

    # The vector bead still landed (memory is a different store)…
    from agent import vector_memory
    from agent.memory import connection

    with connection() as conn:
        beads = vector_memory.recall(conn, alice.email, query_text="t", k=5)
    assert [b for b in beads if b.kind == "work"], "the bead was lost too"

    # …and the graph edges survived the broken mirror.
    rows = sql(
        "SELECT rel FROM media.control.kg_mirror WHERE owner_email = :u",
        {"u": alice.email},
    )
    rels = {r["rel"] for r in rows}
    assert "executed_by" in rels, "the mirror failure cost the graph links"


# ---- the review gate cannot be walked around -----------------------------
# taskboard._finish() lands finished agent work in `review` so a person
# decides whether it is done. `media.alex.tasks` is the board's source of
# truth, and the older `update_task_status` tool writes to the same row —
# so the gate is only as strong as that tool. These tests hold it shut.


def _task(sql, task_id: str, *, owner: str = "dana@alex.dev") -> None:
    sql(
        "INSERT INTO media.alex.tasks (id, owner_email, title, status) "
        "VALUES (:i, :o, :t, :s)",
        {"i": task_id, "o": owner, "t": "Ship the migration", "s": "in_progress"},
    )


def _status(sql, task_id: str) -> str:
    rows = sql("SELECT status FROM media.alex.tasks WHERE id = :i", {"i": task_id})
    return rows[0]["status"]


def test_agent_cannot_close_a_task_it_worked_on(sql):
    from agent.tools import impls

    _task(sql, "t-agent")
    sql(
        "INSERT INTO media.control.task_runs (run_id, task_id, status) "
        "VALUES (:r, :t, :s)",
        {"r": "run-1", "t": "t-agent", "s": "done"},
    )

    out = impls.update_task_status(
        "t-agent", "done", user_email="dana@alex.dev"
    )

    assert out["status"] == "review", "Alex closed a task it had worked on"
    assert _status(sql, "t-agent") == "review"
    assert "review" in out.get("note", "").lower()


def test_user_can_still_close_a_task_alex_never_touched(sql):
    """The gate protects unreviewed agent work, not the user's own todos."""
    from agent.tools import impls

    _task(sql, "t-mine")

    out = impls.update_task_status("t-mine", "done", user_email="dana@alex.dev")

    assert out["status"] == "done"
    assert _status(sql, "t-mine") == "done"


def test_non_done_transitions_are_unaffected_on_agent_worked_tasks(sql):
    from agent.tools import impls

    _task(sql, "t-agent2")
    sql(
        "INSERT INTO media.control.task_runs (run_id, task_id, status) "
        "VALUES (:r, :t, :s)",
        {"r": "run-2", "t": "t-agent2", "s": "done"},
    )

    assert impls.update_task_status(
        "t-agent2", "cancelled", user_email="dana@alex.dev"
    )["status"] == "cancelled"
    assert _status(sql, "t-agent2") == "cancelled"


def test_run_check_fails_closed_when_the_runs_table_is_unreadable(sql, monkeypatch):
    """Wrongly routing to review costs a click; wrongly closing hides work."""
    from agent.tools import impls

    _task(sql, "t-blind")
    monkeypatch.setattr(
        impls, "_sql",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("warehouse down")),
    )

    assert impls._agent_worked_on("t-blind") is True


# ---- durability: surviving the process that was doing the work ----------
# A run is a thread in one process. When that process goes away the work
# stops, but the row used to stay at 'running' for ever: in-flight work
# in the oversight console that would never finish, and a task parked in
# in_progress with nobody aware it needed doing again. Silence is the
# worst outcome, because the crew has been told Alex picked it up.

DANA = "dana@alex.dev"


def _recording_factory(summary: str = "Done.", seen: dict | None = None):
    """A VM factory that records the prompt it was handed."""

    def make(session_id: str, user_email: str):
        vm = StubVM(session_id, user_email)

        def run_turn(text: str):
            if seen is not None:
                seen["prompt"] = text
            for sub in vm._subs:
                sub({"type": "tool", "tool": "read_file", "args": {"p": "x"}})
            return summary, []

        vm.run_turn = run_turn
        return vm

    return make


def _orphan_run(sql, task_id="t-orphan", run_id="run-orphan", *, heartbeat=None):
    """A row exactly as a killed process would leave it."""
    sql(
        "INSERT INTO media.alex.tasks (id, owner_email, title, status) "
        "VALUES (:i, :o, :t, 'in_progress')",
        {"i": task_id, "o": DANA, "t": "Reconcile the ledger"},
    )
    sql(
        "INSERT INTO media.control.task_runs "
        "(run_id, task_id, session_id, started_by, status, started_at, "
        " last_heartbeat) "
        "VALUES (:r, :t, :s, :u, 'running', :started, :beat)",
        {
            "r": run_id, "t": task_id, "s": f"task-{task_id}", "u": DANA,
            "started": datetime.now(timezone.utc) - timedelta(hours=1),
            "beat": heartbeat,
        },
    )


def _run_row(sql, run_id="run-orphan"):
    rows = sql(
        "SELECT * FROM media.control.task_runs WHERE run_id = :r", {"r": run_id}
    )
    return rows[0]


def test_an_abandoned_run_is_not_left_running_for_ever(sql):
    _orphan_run(sql, heartbeat=datetime.now(timezone.utc) - timedelta(hours=1))

    recovered = tb.recover_interrupted_runs(sql)

    assert [r["task_id"] for r in recovered] == ["t-orphan"]
    assert _run_row(sql)["status"] == "interrupted"


def test_recovery_frees_the_task_so_a_human_can_see_it(sql):
    """A task left in_progress with nothing progressing is the state that
    hides the problem — it looks exactly like work under way."""
    _orphan_run(sql, heartbeat=datetime.now(timezone.utc) - timedelta(hours=1))

    tb.recover_interrupted_runs(sql)

    rows = sql("SELECT status FROM media.alex.tasks WHERE id = :i",
               {"i": "t-orphan"})
    assert rows[0]["status"] == "todo"


def test_recovery_says_so_in_the_thread(sql):
    """The crew was told Alex picked this up. They have to be told it
    stopped, or the board lies to them."""
    _orphan_run(sql, heartbeat=datetime.now(timezone.utc) - timedelta(hours=1))

    tb.recover_interrupted_runs(sql)

    bodies = [m["body"] for m in tb.messages(sql, "t-orphan")]
    assert any("went away" in b for b in bodies)
    assert any("not finished" in b for b in bodies)


def test_a_run_with_a_fresh_heartbeat_is_left_alone(sql):
    """Another replica may be driving it. Declaring a live run dead would
    put two processes on the same task."""
    _orphan_run(sql, heartbeat=datetime.now(timezone.utc))

    assert tb.recover_interrupted_runs(sql) == []
    assert _run_row(sql)["status"] == "running"


def test_a_run_this_process_is_driving_is_never_recovered(sql):
    """Its heartbeat may be momentarily stale — a long tool call, a GC
    pause — but we can see the thread is ours and alive."""
    _orphan_run(sql, heartbeat=datetime.now(timezone.utc) - timedelta(hours=1))
    tb._register(tb.TaskRun(
        run_id="run-orphan", task_id="t-orphan",
        session_id="task-t-orphan", started_by=DANA,
    ))

    assert tb.recover_interrupted_runs(sql) == []
    assert _run_row(sql)["status"] == "running"


def test_recovery_is_safe_to_run_twice(sql):
    """It runs on every boot, and a crash loop must not spam the thread."""
    _orphan_run(sql, heartbeat=datetime.now(timezone.utc) - timedelta(hours=1))

    first = tb.recover_interrupted_runs(sql)
    second = tb.recover_interrupted_runs(sql)

    assert len(first) == 1 and second == []
    notes = [
        m["body"] for m in tb.messages(sql, "t-orphan") if "went away" in m["body"]
    ]
    assert len(notes) == 1


def test_a_finished_run_never_looks_alive(sql):
    """last_heartbeat must be cleared, or a completed run would be
    indistinguishable from a stalled one."""
    _task(sql, "t-fin")
    run = tb.start_run(
        sql, P(DANA), "t-fin", background=False,
        vm_factory=_recording_factory("Done."),
    )

    row = _run_row(sql, run.run_id)
    assert row["status"] == "done"
    assert row["last_heartbeat"] is None
    assert row["ended_at"] is not None


def test_the_steps_survive_the_run(sql):
    """The record is the point: what the agent did has to outlive the
    process that did it."""
    _task(sql, "t-steps")
    run = tb.start_run(
        sql, P(DANA), "t-steps", background=False,
        vm_factory=_recording_factory("Reconciled."),
    )

    steps = json.loads(_run_row(sql, run.run_id)["steps"])
    assert any(s.get("type") == "tool" for s in steps)


def test_a_second_process_will_not_start_a_live_task(sql):
    """run_for_task only knows this process. Without a check against the
    table, two replicas would each start a run and both would act."""
    _orphan_run(sql, task_id="t-live", run_id="run-live",
                heartbeat=datetime.now(timezone.utc))

    with pytest.raises(tb.RunAlreadyLive):
        tb.start_run(sql, P(DANA), "t-live", background=False,
                     vm_factory=_recording_factory())


def test_a_stale_run_does_not_block_a_restart(sql):
    """The opposite failure: an abandoned run must not lock the task out
    of ever being worked again."""
    _orphan_run(sql, task_id="t-stale", run_id="run-stale",
                heartbeat=datetime.now(timezone.utc) - timedelta(hours=1))

    run = tb.start_run(
        sql, P(DANA), "t-stale", background=False,
        vm_factory=_recording_factory("Picked it up again."),
    )
    assert run.status == "done"


def test_an_unparseable_heartbeat_reads_as_live_not_dead(sql):
    """The ambiguous case: a heartbeat string the parser cannot read
    used to become 0.0, which every freshness check read as 'long
    dead' — a second process would start a duplicate run. Unknown
    liveness must fail closed."""
    _orphan_run(sql, task_id="t-weird", run_id="run-weird",
                heartbeat="not-a-timestamp")

    assert tb._run_is_live(sql, "t-weird") is True
    with pytest.raises(tb.RunAlreadyLive):
        tb.start_run(sql, P(DANA), "t-weird", background=False,
                     vm_factory=_recording_factory())


def test_recovery_leaves_an_unparseable_heartbeat_alone(sql):
    """Same ambiguity on the recovery side: marking a run interrupted
    while its process may still be working is the duplicate-work
    failure the heartbeat exists to prevent."""
    _orphan_run(sql, task_id="t-weird2", run_id="run-weird2",
                heartbeat="garbage")

    recovered = tb.recover_interrupted_runs(sql)
    assert not any(r["run_id"] == "run-weird2" for r in recovered), (
        "an unknown-liveness run was declared interrupted"
    )


def test_the_next_attempt_is_told_what_the_last_one_did(sql):
    """Resuming means not blindly repeating side effects the previous
    attempt already caused."""
    _orphan_run(sql, task_id="t-resume", run_id="run-resume",
                heartbeat=datetime.now(timezone.utc) - timedelta(hours=1))
    sql(
        "UPDATE media.control.task_runs SET status = 'interrupted', "
        "steps = :s WHERE run_id = :r",
        {
            "s": json.dumps([{"type": "tool", "tool": "send_email"}]),
            "r": "run-resume",
        },
    )

    seen: dict = {}
    tb.start_run(
        sql, P(DANA), "t-resume", background=False,
        vm_factory=_recording_factory("Checked first.", seen),
    )

    assert "send_email" in seen["prompt"]
    assert "do not repeat writes" in seen["prompt"]


def test_no_interrupted_history_means_no_resume_note(sql):
    """A first attempt must not be told to check for effects of a run
    that never happened."""
    _task(sql, "t-first")
    seen: dict = {}
    tb.start_run(
        sql, P(DANA), "t-first", background=False,
        vm_factory=_recording_factory("Fresh start.", seen),
    )

    assert "previous attempt" not in seen["prompt"]


def test_the_resume_note_carries_results_not_just_names(sql):
    """The honest half of mid-turn resumption: the knowledge survives the
    process. A note that lists only tool names sends the next attempt
    back to re-run every read — or worse, repeat a write to be sure."""
    _orphan_run(sql, task_id="t-facts", run_id="run-facts",
                heartbeat=datetime.now(timezone.utc) - timedelta(hours=1))
    steps = [
        {"type": "tool", "tool": "search_enterprise", "args": {"q": "rates"}},
        {"type": "artifact", "kind": "tool_result",
         "preview": "3 hits: the Q3 rate card is doc-4451"},
        {"type": "tool", "tool": "create_task", "args": {"title": "follow-up"},
         "result": {"ok": True, "task_id": "t-9\u00981"}},
        {"type": "text", "text": "an undelivered draft that must NOT appear"},
    ]
    sql("UPDATE media.control.task_runs SET status = 'interrupted', "
        "steps = :s WHERE run_id = :r",
        {"s": json.dumps(steps), "r": "run-facts"})

    seen: dict = {}
    tb.start_run(sql, P(DANA), "t-facts", background=False,
                 vm_factory=_recording_factory("resumed", seen))

    prompt = seen["prompt"]
    assert "the Q3 rate card is doc-4451" in prompt, (
        "the read's result was lost — the next attempt has to re-run it"
    )
    assert "create_task" in prompt and "t-9" in prompt
    assert "undelivered draft" not in prompt, (
        "generated-but-never-delivered text leaked into the resume prompt"
    )


# ---- stop actually stops ---------------------------------------------------
# "Stop" used to be bookkeeping only: the in-flight turn ran to its
# budget, executing tools under a run the crew had already stopped.


def test_stop_aborts_the_live_loop(sql):
    """stop_run sets the VM's abort flag: the loop unwinds at its next
    safe point and no further tool executes."""
    alice = P("alice@x.com")
    tid = tb.create_task(sql, alice, title="t")["task_id"]

    class AbortableVM(StubVM):
        gate = None

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.stopped = False
            self.gate = threading.Event()

        def request_stop(self, reason=""):
            self.stopped = True
            self.gate.set()
            return True

        def run_turn(self, text):
            self.turn_active = True
            try:
                for sub in self._subs:
                    sub({"type": "tool", "tool": "search_knowledge", "args": {}})
                self.gate.wait(timeout=5)
                self.gate.clear()
                # The loop's steering poll raises TurnAborted when the
                # VM's abort flag is up; this stub models the unwind by
                # re-raising what the real drain would.
                if self.stopped:
                    from agent.vm import TurnAborted

                    raise TurnAborted()
                for sub in self._subs:
                    sub({"type": "text", "text": "done"})
                return "Finished.", []
            finally:
                self.turn_active = False

    def make(session_id, user_email):
        vm = AbortableVM(session_id, user_email)
        factory.made["vm"] = vm
        return vm

    factory = _factory()
    run = tb.start_run(sql, alice, tid, vm_factory=make, background=True)
    vm = factory.made["vm"]
    try:
        deadline = time.monotonic() + 5
        while not vm.turn_active and time.monotonic() < deadline:
            time.sleep(0.02)
        assert tb.stop_run(sql, tid) is True
        # The worker catches TurnAborted and records stopped — the
        # idempotent _finish keeps the crew's verdict over the run's
        # own "done".
        for _ in range(250):
            if run.status == "stopped":
                break
            time.sleep(0.02)
        assert run.status == "stopped", f"status: {run.status}"
        assert vm.stopped, "stop never reached the VM"
    finally:
        vm.gate.set()
        run_thread = [t for t in threading.enumerate() if t.name.startswith("taskrun")]
        for t in run_thread:
            t.join(timeout=5)


def test_stopped_vm_refuses_further_tools(sql, vm_cls=None):
    """The tool gate honours the abort flag: no write can slip in
    between 'the crew stopped it' and the loop's unwind."""

    from agent.omnigent import MockExecutor, ToolCallRequest
    from agent.vm import make_vm

    v = make_vm(session_id="s-stop", user_email="alice@x.com",
                executor=MockExecutor())
    v._turn_active = True
    try:
        v.request_stop("test")
        out = v._invoke_tool(ToolCallRequest(name="list_my_tasks", args={}))
        assert out.get("ok") is False
        assert out.get("stopped") is True
        # A fresh turn clears the stop: the VM is reusable.
        v._turn_active = False
        v.run_turn(text="hello")
        v._turn_active = True
        assert v.abort_requested is False
    finally:
        v._turn_active = False
