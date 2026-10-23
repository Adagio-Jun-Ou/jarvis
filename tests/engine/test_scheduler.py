"""The scheduler — the missing half of TRIGGER_SCHEDULE.

The trigger existed with a comment claiming a pipeline executed it;
nothing did. Properties pinned:

* the grammar is small, readable, and refuses sub-15-minute intervals
  (each firing is a governed run and a model bill);
* an unparseable schedule NEVER fires — it cannot fire at a surprising
  time instead;
* due-ness anchors on the last run, so missed windows collapse into
  one firing (a brief wants the latest edition, not three stale ones);
* a firing goes through run() as the order's OWNER — single-flight,
  assignment and activity checks exactly as for a human click;
* firings per sweep are capped.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import pytest

from agent import access
from agent import schedule as sched
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


@pytest.fixture
def sql(env):
    from alex.testing import get_fakes
    from app._dev_seed import STANDING_ORDER_RUNS_COLS, STANDING_ORDERS_COLS

    wh = get_fakes().warehouse
    wh.register("media.control.standing_orders", STANDING_ORDERS_COLS,
                pk=["order_id"])
    wh.register("media.control.standing_order_runs",
                STANDING_ORDER_RUNS_COLS, pk=["run_id"])

    def _sql(stmt: str, params: dict | None = None):
        res = wh.execute(stmt, params or {})
        as_dicts = getattr(res, "as_dicts", None)
        return as_dicts() if callable(as_dicts) else list(res or [])

    return _sql


def principal(email="alice@x.com"):
    return access.Principal(email=email, role="member",
                            capabilities=frozenset())


class QuickVM:
    def subscribe_sse(self, fn):
        return lambda: None

    def run_turn(self, *, text, **_kw):
        return "the brief went out", []

    def close(self):
        pass


def _utc(y, mo, d, h, mi):
    return datetime(y, mo, d, h, mi, tzinfo=timezone.utc)


# ---- the grammar -----------------------------------------------------------


def test_the_grammar_covers_the_enterprise_shapes():
    after = _utc(2026, 9, 1, 9, 0)  # a Tuesday
    assert sched.next_due("daily 07:30", after) == _utc(2026, 9, 2, 7, 30)
    assert sched.next_due("hourly @:15", after) == _utc(2026, 9, 1, 9, 15)
    assert sched.next_due("weekly mon 08:00", after) == _utc(2026, 9, 7, 8, 0)
    assert sched.next_due("every 30 minutes", after) == _utc(2026, 9, 1, 9, 30)
    # Cron subset: 30 7 * * 1 = Mondays 07:30 (cron 1 = Monday).
    assert sched.next_due("30 7 * * 1", after) == _utc(2026, 9, 7, 7, 30)
    # Cron Sunday is 0.
    assert sched.next_due("0 9 * * 0", after) == _utc(2026, 9, 6, 9, 0)


def test_the_cost_floor_and_junk_are_refused():
    assert "floor" in sched.parse_error("every 5 minutes")
    assert sched.parse_error("whenever the mood strikes") is not None
    assert sched.next_due("every 5 minutes", _utc(2026, 9, 1, 9, 0)) is None
    assert sched.next_due("* * * * *", _utc(2026, 9, 1, 9, 0)) is None, (
        "full-cron every-minute must not slip past the subset"
    )
    assert sched.parse_error("daily 07:30") is None


# ---- firing ----------------------------------------------------------------


def _seed_run(sql, order_id, started_at):
    sql(
        "INSERT INTO media.control.standing_order_runs "
        "(run_id, order_id, session_id, started_by, status, started_at, "
        " capabilities, summary) "
        "VALUES (:r, :o, 's', 'alice@x.com', 'done', :at, '', '')",
        {"r": f"orun-{order_id}-old", "o": order_id, "at": started_at},
    )


def test_a_due_order_fires_once_and_missed_windows_collapse(sql):
    order = so.create(
        sql, principal(), title="Morning brief",
        instruction="compile and post the brief",
        trigger="schedule", schedule="daily 07:30",
    )
    # Last ran three days ago: three windows were missed.
    _seed_run(sql, order.order_id, "2026-08-29 07:30:00")
    now = time.time()
    fired = so.run_due(sql, now=now, vm_factory=lambda **kw: QuickVM())
    assert fired == [order.order_id], "three missed windows -> ONE firing"
    # The firing recorded a run; the very next sweep sees it as the
    # anchor and finds nothing due.
    assert so.run_due(sql, now=now + 1,
                      vm_factory=lambda **kw: QuickVM()) == []


def test_inactive_manual_and_unparseable_orders_never_fire(sql):
    o1 = so.create(sql, principal(), title="manual", instruction="i",
                   trigger="manual", schedule="daily 07:30")
    o2 = so.create(sql, principal(), title="junk", instruction="i",
                   trigger="schedule", schedule="whenever")
    o3 = so.create(sql, principal(), title="off", instruction="i",
                   trigger="schedule", schedule="daily 07:30")
    so.update(sql, principal(), o3.order_id, active=False)
    for oid in (o1.order_id, o2.order_id, o3.order_id):
        _seed_run(sql, oid, "2026-08-01 07:30:00")
    assert so.run_due(sql, now=time.time(),
                      vm_factory=lambda **kw: QuickVM()) == []


def test_firings_per_sweep_are_capped(sql, monkeypatch):
    monkeypatch.setattr(so, "_MAX_FIRINGS_PER_SWEEP", 2)
    for i in range(4):
        order = so.create(sql, principal(), title=f"o{i}", instruction="i",
                          trigger="schedule", schedule="daily 07:30")
        _seed_run(sql, order.order_id, "2026-08-01 07:30:00")
    fired = so.run_due(sql, now=time.time(),
                       vm_factory=lambda **kw: QuickVM())
    assert len(fired) == 2, "a 07:30 pile-up staggers across sweeps"


# ---- first fire -------------------------------------------------------------
# The due-ness anchor for an order with zero runs is the row's
# created_at; _from_row used to leave the dataclass default ("now"),
# so a scheduled order never fired until someone clicked Run once.


def test_a_never_run_scheduled_order_fires_on_its_first_window(sql):
    order = so.create(
        sql, principal(), title="Morning brief",
        instruction="compile and post the brief",
        trigger="schedule", schedule="daily 07:30",
    )
    # Materialized yesterday; never run. The seed's created_at default
    # is the seed moment, so rewrite it to yesterday to model an order
    # that has been sitting due since before the sweep woke up.
    sql(
        "UPDATE media.control.standing_orders SET created_at = :c "
        "WHERE order_id = :o",
        {"c": "2026-08-31 07:30:00", "o": order.order_id},
    )
    now = time.time()
    fired = so.run_due(sql, now=now, vm_factory=lambda **kw: QuickVM())
    assert fired == [order.order_id], (
        "a scheduled order with no runs must fire from its created_at anchor"
    )


# ---- out-of-range values are refused, not clamped ---------------------------


def test_out_of_range_clock_values_never_fire():
    assert sched.parse_error("daily 25:00") is not None, (
        "'daily 25:00' must not silently fire at 23:00"
    )
    assert sched.parse_error("hourly @:75") is not None
    assert sched.parse_error("70 7 * * 1") is not None
    assert sched.next_due("daily 25:00", _utc(2026, 9, 1, 9, 0)) is None
    # The valid forms still pass.
    assert sched.parse_error("daily 23:59") is None
    assert sched.parse_error("hourly @:59") is None
