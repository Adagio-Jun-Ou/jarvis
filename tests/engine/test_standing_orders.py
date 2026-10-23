"""Tests for agent/standing_orders.py.

The load-bearing property is the capability ceiling: a standing order
must be able to narrow authority and must never widen it — not for the
author, not for the runner, not through a later edit.
"""

from __future__ import annotations

import pytest

from agent import access
from agent import standing_orders as so


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("ALEX_ENV", "dev")
    from alex import config, testing

    config._settings = None
    testing._fakes = None
    so.reset_runs()
    yield
    config._settings = None
    testing._fakes = None
    so.reset_runs()


def principal(email="alice@x.com", caps=(), role="member"):
    return access.Principal(
        email=email, role=role, capabilities=frozenset(caps)
    )


@pytest.fixture
def sql(env):
    from alex.testing import get_fakes

    wh = get_fakes().warehouse
    wh.register(
        "media.control.standing_orders",
        ["order_id", "owner_email", "title", "instruction", "schedule",
         "audience", "allowed_capabilities", "active", "created_by",
         "trigger", "ambient", "applies_to"],
        pk=["order_id"],
    )
    # Imported, not restated: a third copy of this schema is how a change
    # passes its own tests while breaking the application.
    from app._dev_seed import STANDING_ORDER_RUNS_COLS

    wh.register(
        "media.control.standing_order_runs",
        STANDING_ORDER_RUNS_COLS,
        pk=["run_id"],
    )

    def _sql(stmt: str, params: dict | None = None):
        res = wh.execute(stmt, params or {})
        as_dicts = getattr(res, "as_dicts", None)
        return as_dicts() if callable(as_dicts) else list(res or [])

    return _sql


# ---- authoring -----------------------------------------------------------


def test_create_requires_title_and_instruction(sql):
    with pytest.raises(ValueError):
        so.create(sql, principal(), title="", instruction="do a thing")
    with pytest.raises(ValueError):
        so.create(sql, principal(), title="t", instruction="   ")


def test_author_cannot_grant_capabilities_they_lack(sql):
    """The ceiling is clamped to the AUTHOR's own ACL at write time."""
    author = principal(caps={"email.read"})
    order = so.create(
        sql, author, title="Review", instruction="do it",
        allowed_capabilities=["email.read", "email.send", "compute.shell"],
    )
    assert order.allowed_capabilities == ["email.read"]


def test_an_order_is_assigned_to_its_author_by_default(sql):
    order = so.create(sql, principal(), title="t", instruction="i")
    assert order.assigned_to("alice@x.com")
    assert not order.assigned_to("bob@x.com")


def test_audience_all_reaches_everyone(sql):
    order = so.create(sql, principal(), title="t", instruction="i", audience=["all"])
    assert order.assigned_to("anyone@x.com")


def test_for_user_lists_only_assigned_orders(sql):
    so.create(sql, principal("alice@x.com"), title="Alice's", instruction="i")
    so.create(sql, principal("bob@x.com"), title="Bob's", instruction="i")
    titles = [o.title for o in so.for_user(sql, "alice@x.com")]
    assert titles == ["Alice's"]


def test_unknown_trigger_falls_back_to_manual(sql):
    order = so.create(sql, principal(), title="t", instruction="i", trigger="whenever")
    assert order.trigger == so.TRIGGER_MANUAL


# ---- editing and assignment ---------------------------------------------


def test_only_the_owner_or_an_admin_may_edit(sql):
    order = so.create(sql, principal("alice@x.com"), title="t", instruction="i")
    with pytest.raises(PermissionError):
        so.update(sql, principal("mallory@x.com"), order.order_id, title="hijacked")
    admin = principal("root@x.com", role="admin")
    assert so.update(sql, admin, order.order_id, title="fixed").title == "fixed"


def test_editing_cannot_widen_the_ceiling(sql):
    """The escalation attempt: edit an order to grant more than the
    editor holds."""
    author = principal(caps={"email.read"})
    order = so.create(
        sql, author, title="t", instruction="i",
        allowed_capabilities=["email.read"],
    )
    edited = so.update(
        sql, author, order.order_id,
        allowed_capabilities=["email.read", "email.send", "compute.shell"],
    )
    assert edited.allowed_capabilities == ["email.read"]


def test_editing_may_narrow_the_ceiling(sql):
    author = principal(caps={"email.read", "tasks.read"})
    order = so.create(
        sql, author, title="t", instruction="i",
        allowed_capabilities=["email.read", "tasks.read"],
    )
    edited = so.update(
        sql, author, order.order_id, allowed_capabilities=["email.read"]
    )
    assert edited.allowed_capabilities == ["email.read"]


def test_assign_adds_members_without_dropping_existing(sql):
    author = principal("alice@x.com")
    order = so.create(sql, author, title="t", instruction="i")
    updated = so.assign(sql, author, order.order_id, ["bob@x.com", "bob@x.com"])
    assert updated.audience.count("bob@x.com") == 1
    assert "alice@x.com" in updated.audience


def test_assign_unknown_order_returns_none(sql):
    assert so.assign(sql, principal(), "so-nope", ["bob@x.com"]) is None


def test_deactivating_an_order(sql):
    author = principal()
    order = so.create(sql, author, title="t", instruction="i")
    so.update(sql, author, order.order_id, active=False)
    assert so.for_user(sql, author.email) == []
    assert len(so.for_user(sql, author.email, active_only=False)) == 1


# ---- the capability ceiling at run time ---------------------------------


def test_effective_capabilities_intersect_order_and_runner(sql):
    order = so.StandingOrder(
        order_id="so-1", title="t", instruction="i", owner_email="a@x.com",
        allowed_capabilities=["email.read", "email.send"],
    )
    runner = principal(caps={"email.read", "tasks.read"})
    assert so.effective_capabilities(order, runner) == ["email.read"]


def test_an_order_never_widens_the_runners_authority(sql):
    """Even if the order permits it, a capability the runner lacks is
    not granted."""
    order = so.StandingOrder(
        order_id="so-1", title="t", instruction="i", owner_email="a@x.com",
        allowed_capabilities=["compute.shell", "email.send"],
    )
    assert so.effective_capabilities(order, principal(caps={"email.read"})) == []


def test_an_order_with_no_ceiling_inherits_the_runner(sql):
    order = so.StandingOrder(
        order_id="so-1", title="t", instruction="i", owner_email="a@x.com",
    )
    assert so.effective_capabilities(order, principal(caps={"email.read"})) == ["email.read"]


# ---- execution -----------------------------------------------------------


class StubVM:
    def __init__(self, session_id, user_email, fail=False):
        self.session_id = session_id
        self.user_email = user_email
        self.closed = False
        self._subs = []
        self._fail = fail

    def subscribe_sse(self, fn):
        self._subs.append(fn)
        return lambda: None

    def run_turn(self, text):
        self.prompt = text
        for s in self._subs:
            s({"type": "tool", "tool": "read_inbox", "args": {}})
        if self._fail:
            raise RuntimeError("model exploded")
        return "Reviewed the inbox; three items need you.", []

    def close(self):
        self.closed = True


def _factory(**kw):
    made = {}

    def make(session_id, user_email):
        made["vm"] = StubVM(session_id, user_email, **kw)
        return made["vm"]

    make.made = made
    return make


def test_running_an_order_executes_it_and_records_the_result(sql):
    author = principal(caps={"email.read"})
    order = so.create(
        sql, author, title="Morning review", instruction="Summarise my inbox.",
        allowed_capabilities=["email.read"],
    )
    factory = _factory()
    run = so.run(sql, author, order.order_id, vm_factory=factory, background=False)

    assert run.status == "done"
    assert "Reviewed the inbox" in run.summary
    assert run.capabilities == ["email.read"]
    assert factory.made["vm"].closed
    types = [e["type"] for e in run.events]
    assert types[0] == "run_started" and types[-1] == "run_finished"


def test_the_run_gets_its_own_session(sql):
    author = principal()
    order = so.create(sql, author, title="t", instruction="i")
    a = so.run(sql, author, order.order_id, vm_factory=_factory(), background=False)
    b = so.run(sql, author, order.order_id, vm_factory=_factory(), background=False)
    assert a.session_id != b.session_id, "runs must not share a session"


def test_an_order_cannot_run_twice_at_once(sql):
    """An unattended order runs with real capabilities: two simultaneous
    executions send the email twice and no one is watching either.
    Sequential runs stay fine (the first finishes before the second
    starts); only an overlapping run is refused."""
    author = principal()
    order = so.create(sql, author, title="t", instruction="i")
    so.run(sql, author, order.order_id, vm_factory=_factory(), background=False)

    class LiveRun:
        run_id = "orun-live"
        order_id = order.order_id
        started_at = 0.0
        status = "running"

        def running(self):  # pragma: no cover - shape only
            return True

    import agent.standing_orders as so_mod

    live = so_mod.OrderRun(
        run_id="orun-live", order_id=order.order_id, session_id="s",
        started_by=author.email,
    )
    with so_mod._RUNS_LOCK:
        so_mod._RUNS["orun-live"] = live
    try:
        with pytest.raises(so_mod.RunAlreadyLive):
            so.run(sql, author, order.order_id, vm_factory=_factory(),
                   background=False)
    finally:
        with so_mod._RUNS_LOCK:
            so_mod._RUNS.pop("orun-live", None)


def test_the_prompt_carries_the_instruction_and_the_note(sql):
    author = principal()
    order = so.create(sql, author, title="Review", instruction="Summarise my inbox.")
    factory = _factory()
    so.run(
        sql, author, order.order_id, note="focus on finance",
        vm_factory=factory, background=False,
    )
    prompt = factory.made["vm"].prompt
    assert "Summarise my inbox." in prompt
    assert "focus on finance" in prompt
    assert order.order_id in prompt


def test_someone_the_order_is_not_assigned_to_cannot_run_it(sql):
    order = so.create(sql, principal("alice@x.com"), title="t", instruction="i")
    with pytest.raises(PermissionError):
        so.run(sql, principal("mallory@x.com"), order.order_id,
               vm_factory=_factory(), background=False)


def test_an_admin_may_run_any_order(sql):
    order = so.create(sql, principal("alice@x.com"), title="t", instruction="i")
    run = so.run(sql, principal("root@x.com", role="admin"), order.order_id,
                 vm_factory=_factory(), background=False)
    assert run.status == "done"


def test_an_inactive_order_cannot_be_run(sql):
    author = principal()
    order = so.create(sql, author, title="t", instruction="i")
    so.update(sql, author, order.order_id, active=False)
    with pytest.raises(PermissionError):
        so.run(sql, author, order.order_id, vm_factory=_factory(), background=False)


def test_running_an_unknown_order_raises(sql):
    with pytest.raises(LookupError):
        so.run(sql, principal(), "so-nope", vm_factory=_factory(), background=False)


def test_a_failing_run_is_recorded(sql):
    author = principal()
    order = so.create(sql, author, title="t", instruction="i")
    run = so.run(sql, author, order.order_id,
                 vm_factory=_factory(fail=True), background=False)
    assert run.status == "failed"
    assert "model exploded" in run.summary


def test_a_late_watcher_sees_the_whole_run(sql):
    author = principal()
    order = so.create(sql, author, title="t", instruction="i")
    run = so.run(sql, author, order.order_id, vm_factory=_factory(), background=False)
    seen = []
    run.subscribe(seen.append)
    assert [e["type"] for e in seen] == [e["type"] for e in run.events]


# ---- the ambient prompt slice -------------------------------------------


def test_an_unscoped_order_is_always_ambient(sql):
    author = principal()
    so.create(sql, author, title="House rule", instruction="Always be brief.")
    assert [o.title for o in so.matching(sql, author.email, "anything")] == [
        "House rule"
    ]


def test_a_scoped_order_only_applies_to_its_subjects(sql):
    author = principal()
    so.create(
        sql, author, title="Customer brief", instruction="Pull the account.",
        applies_to=["customer", "account"],
    )
    assert so.matching(sql, author.email, "tell me about this customer")
    assert so.matching(sql, author.email, "what is the weather") == []


def test_an_order_can_opt_out_of_the_ambient_slice(sql):
    author = principal()
    so.create(sql, author, title="Manual only", instruction="i", ambient=False)
    assert so.matching(sql, author.email, "anything") == []


def test_directive_block_marks_runnable_orders(sql):
    author = principal()
    so.create(sql, author, title="Morning review", instruction="Summarise.")
    block = so.directive_block(so.matching(sql, author.email, "x"))
    assert "STANDING ORDERS" in block
    assert "runnable on demand" in block
    assert "Morning review" in block


def test_directive_block_empty_is_empty_string():
    assert so.directive_block([]) == ""


# ---- an unattended run that was abandoned --------------------------------
# An order runs with real capabilities against real systems and nobody is
# watching a stream. If the process driving it dies, the row says
# 'running' for ever and the oversight console reports work in flight
# that nothing is doing — the one place a phantom is most expensive.


def _stale_order_run(sql, run_id="or-1", order_id="o-1", *, heartbeat=None):
    from datetime import datetime, timedelta, timezone

    sql(
        "INSERT INTO media.control.standing_order_runs "
        "(run_id, order_id, session_id, started_by, status, started_at, "
        " last_heartbeat, capabilities, summary) "
        "VALUES (:r, :o, :s, :u, 'running', :started, :beat, :c, '')",
        {
            "r": run_id, "o": order_id, "s": "sess", "u": "dana@alex.dev",
            "started": (
                datetime.now(timezone.utc) - timedelta(hours=2)
            ).isoformat(sep=" "),
            "beat": heartbeat,
            "c": "['email.send']",
        },
    )


def test_an_abandoned_order_run_is_closed_out(sql):
    _stale_order_run(sql)

    recovered = so.recover_interrupted_runs(sql)

    assert [r["order_id"] for r in recovered] == ["o-1"]
    row = sql("SELECT status, summary FROM media.control.standing_order_runs "
              "WHERE run_id = :r", {"r": "or-1"})[0]
    assert row["status"] == "interrupted"
    assert "did not finish" in row["summary"]


def test_a_live_order_run_is_left_alone(sql):
    from datetime import datetime, timezone

    _stale_order_run(sql, heartbeat=datetime.now(timezone.utc).isoformat(sep=" "))

    assert so.recover_interrupted_runs(sql) == []


def test_order_recovery_uses_the_same_staleness_as_task_runs(sql):
    """Two definitions of 'stale' that drift apart is how one of these
    paths quietly stops recovering."""
    from agent import taskboard

    _stale_order_run(sql)
    assert so.recover_interrupted_runs(sql, stale_after=10**9) == []
    assert taskboard.STALE_AFTER_SECONDS > 0
    assert so.recover_interrupted_runs(sql)
