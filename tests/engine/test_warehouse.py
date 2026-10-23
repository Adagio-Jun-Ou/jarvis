"""Tests for the in-memory warehouse used in dev and tests."""
from __future__ import annotations

import pytest

from alex.errors import BadRequest, NotFound
from alex.warehouse import InMemoryWarehouse


@pytest.fixture
def wh():
    w = InMemoryWarehouse()
    w.register("alex.users", ["email", "name"], pk=["email"], rows=[
        ("a@x.com", "Alice"),
        ("b@x.com", "Bob"),
    ])
    w.register("alex.tasks", ["id", "assignee", "title", "status"], pk=["id"], rows=[
        (1, "a@x.com", "Write spec", "open"),
        (2, "b@x.com", "Review", "open"),
    ])
    return w


def test_select_star(wh):
    r = wh.execute("SELECT * FROM alex.users")
    assert r.columns == ["email", "name"]
    assert len(r.rows) == 2


def test_select_projection(wh):
    r = wh.execute("SELECT email FROM alex.users")
    assert r.columns == ["email"]
    assert r.rows == [("a@x.com",), ("b@x.com",)]


def test_select_where_param(wh):
    r = wh.execute("SELECT name FROM alex.users WHERE email = :e", {"e": "a@x.com"})
    assert r.rows == [("Alice",)]


def test_select_limit(wh):
    r = wh.execute("SELECT * FROM alex.tasks ORDER BY id LIMIT 1")
    assert len(r.rows) == 1


def test_insert_explicit_columns(wh):
    wh.execute(
        "INSERT INTO alex.tasks (id, assignee, title, status) VALUES (:i, :a, :t, :s)",
        {"i": 3, "a": "a@x.com", "t": "Test", "s": "open"},
    )
    r = wh.execute("SELECT * FROM alex.tasks WHERE id = :i", {"i": 3})
    assert r.rows == [(3, "a@x.com", "Test", "open")]


def test_update_with_where(wh):
    r = wh.execute(
        "UPDATE alex.tasks SET status = :s WHERE id = :i",
        {"s": "closed", "i": 1},
    )
    assert r.row_count == 1
    r = wh.execute("SELECT status FROM alex.tasks WHERE id = :i", {"i": 1})
    assert r.rows == [("closed",)]


def test_delete(wh):
    r = wh.execute("DELETE FROM alex.tasks WHERE id = :i", {"i": 2})
    assert r.row_count == 1
    r = wh.execute("SELECT id FROM alex.tasks")
    assert r.rows == [(1,)]


def test_create_table(wh):
    wh.execute("CREATE TABLE IF NOT EXISTS alex.events (id INT, msg STRING)")
    wh.execute(
        "INSERT INTO alex.events (id, msg) VALUES (:i, :m)",
        {"i": 1, "m": "hello"},
    )
    r = wh.execute("SELECT * FROM alex.events")
    assert r.rows == [(1, "hello")]


def test_row_filter(wh):
    """Row filters simulate UC row-level security."""
    wh.register(
        "alex.private",
        ["id", "owner"],
        pk=["id"],
        rows=[(1, "a@x.com"), (2, "b@x.com")],
        row_filter=lambda r: r["owner"] == "a@x.com",
    )
    r = wh.execute("SELECT * FROM alex.private")
    assert r.rows == [(1, "a@x.com")]


def test_delete_with_unparseable_where_fails_closed(wh):
    # `LIKE` is outside the grammar. It must raise, NOT silently delete
    # every row in the table.
    with pytest.raises(BadRequest):
        wh.execute("DELETE FROM alex.tasks WHERE title LIKE 'W%'", {})
    # Table is untouched.
    r = wh.execute("SELECT id FROM alex.tasks")
    assert len(r.rows) == 2


def test_update_with_unparseable_where_fails_closed(wh):
    # A construct the grammar genuinely cannot express must raise, NOT
    # silently degrade to "match all" and rewrite the whole table.
    with pytest.raises(BadRequest):
        wh.execute(
            "UPDATE alex.tasks SET status = :s WHERE title LIKE 'W%'",
            {"s": "closed"},
        )
    r = wh.execute("SELECT status FROM alex.tasks WHERE id = :i", {"i": 1})
    assert r.rows == [("open",)]


def test_update_honours_an_or_predicate(wh):
    """`OR` is supported, so it must actually be evaluated — an engine
    that ignored it would rewrite rows the caller excluded."""
    wh.execute(
        "UPDATE alex.tasks SET status = :s WHERE id = :a OR id = :b",
        {"s": "closed", "a": 1, "b": 2},
    )
    rows = wh.execute("SELECT id, status FROM alex.tasks ORDER BY id").rows
    assert {r[1] for r in rows} == {"closed"}


def test_select_with_an_unsupported_where_raises(wh):
    """A SELECT whose WHERE cannot be honoured must fail rather than
    return the rows it was asked to exclude — on a per-user table that
    silent widening is one person reading another's data."""
    with pytest.raises(BadRequest):
        wh.execute("SELECT id FROM alex.tasks WHERE title LIKE 'W%'", {})


def test_select_honours_in_and_not_in(wh):
    """`IN` is how the production statements filter by status; silently
    ignoring it returned every row, including other users'."""
    rows = wh.execute(
        "SELECT id FROM alex.tasks WHERE status IN ('open', 'in_progress')"
    ).rows
    assert [r[0] for r in rows] == [1, 2]
    none = wh.execute(
        "SELECT id FROM alex.tasks WHERE status NOT IN ('open')"
    ).rows
    assert none == []


def test_in_composes_with_a_bound_parameter(wh):
    rows = wh.execute(
        "SELECT id FROM alex.tasks "
        "WHERE assignee = :a AND status IN ('open', 'done')",
        {"a": "a@x.com"},
    ).rows
    assert [r[0] for r in rows] == [1]


def test_select_honours_is_null_and_comparison(wh):
    wh.execute(
        "INSERT INTO alex.tasks (id, assignee, title, status) "
        "VALUES (:i, :a, :t, :s)",
        {"i": 3, "a": None, "t": "unassigned", "s": "open"},
    )
    null_rows = wh.execute("SELECT id FROM alex.tasks WHERE assignee IS NULL").rows
    assert [r[0] for r in null_rows] == [3]
    not_null = wh.execute(
        "SELECT id FROM alex.tasks WHERE assignee IS NOT NULL"
    ).rows
    assert 3 not in [r[0] for r in not_null]
    gt = wh.execute("SELECT id FROM alex.tasks WHERE id > :n", {"n": 1}).rows
    assert [r[0] for r in gt] == [2, 3]


def test_select_honours_a_string_literal(wh):
    rows = wh.execute("SELECT id FROM alex.tasks WHERE status = 'open'").rows
    assert len(rows) == 2


def test_unknown_table_raises(wh):
    with pytest.raises(NotFound):
        wh.execute("SELECT * FROM alex.nope")


def test_unsupported_statement_raises(wh):
    with pytest.raises(BadRequest):
        wh.execute("TRUNCATE TABLE alex.users")


# ---- value expressions: what the dev engine writes -----------------------
# The dev engine is only useful if a statement that runs against Databricks
# behaves the same here. Two ways it did not:
#
#   * INSERT turned any non-parameter token into NULL. A literal the caller
#     wrote — 'running', a number — was silently discarded, so a run's
#     status was NULL in dev and 'running' in production. Nothing noticed
#     because no test read the column back.
#   * SET accepted a bound parameter and nothing else, so ordinary SQL like
#     `SET ended_at = current_timestamp()` failed here for a reason that
#     does not exist in Databricks.
#
# Silently writing NULL is the worst of the options. Refusing is honest;
# storing what was asked for is correct.


def _wh_with_run_table():
    from alex.warehouse import InMemoryWarehouse

    wh = InMemoryWarehouse()
    wh.register(
        "media.control.runs",
        ["run_id", "status", "started_at", "ended_at", "note", "n", "flag"],
        pk=["run_id"],
    )
    return wh


def test_insert_stores_a_string_literal_rather_than_null():
    wh = _wh_with_run_table()
    wh.execute(
        "INSERT INTO media.control.runs (run_id, status) VALUES (:r, 'running')",
        {"r": "r1"},
    )
    row = wh.execute("SELECT * FROM media.control.runs", {}).as_dicts()[0]
    assert row["status"] == "running", "a written literal was silently dropped"


def test_insert_stores_numbers_and_booleans():
    wh = _wh_with_run_table()
    wh.execute(
        "INSERT INTO media.control.runs (run_id, n, flag) VALUES (:r, 42, true)",
        {"r": "r1"},
    )
    row = wh.execute("SELECT * FROM media.control.runs", {}).as_dicts()[0]
    assert row["n"] == 42
    assert row["flag"] is True


def test_insert_stores_the_clock():
    wh = _wh_with_run_table()
    wh.execute(
        "INSERT INTO media.control.runs (run_id, started_at) "
        "VALUES (:r, current_timestamp())",
        {"r": "r1"},
    )
    row = wh.execute("SELECT * FROM media.control.runs", {}).as_dicts()[0]
    assert row["started_at"], "current_timestamp() wrote nothing"


def test_update_accepts_the_clock_and_null():
    wh = _wh_with_run_table()
    wh.execute(
        "INSERT INTO media.control.runs (run_id, status) VALUES (:r, 'running')",
        {"r": "r1"},
    )
    wh.execute(
        "UPDATE media.control.runs SET status = :s, ended_at = current_timestamp(), "
        "started_at = NULL WHERE run_id = :r",
        {"s": "done", "r": "r1"},
    )
    row = wh.execute("SELECT * FROM media.control.runs", {}).as_dicts()[0]
    assert row["status"] == "done"
    assert row["ended_at"]
    assert row["started_at"] is None


def test_timestamps_are_one_type_so_they_can_be_sorted():
    """A column holding both str and datetime sorts by raising TypeError,
    which is a strange way to find out about a type mismatch. The clock
    has to produce the same shape the rest of dev writes."""
    wh = _wh_with_run_table()
    wh.execute(
        "INSERT INTO media.control.runs (run_id, started_at) VALUES (:r, :t)",
        {"r": "seeded", "t": "2020-01-01 00:00:00+00:00"},
    )
    wh.execute(
        "INSERT INTO media.control.runs (run_id, started_at) "
        "VALUES (:r, current_timestamp())",
        {"r": "fresh"},
    )
    rows = wh.execute("SELECT * FROM media.control.runs", {}).as_dicts()

    ordered = sorted(rows, key=lambda r: r["started_at"])  # must not raise
    assert [r["run_id"] for r in ordered] == ["seeded", "fresh"]


def test_interval_arithmetic_is_supported():
    """Expiry columns are written server-side on purpose: the server's
    clock decides, not the caller's."""
    from datetime import datetime, timezone

    wh = _wh_with_run_table()
    wh.execute(
        "INSERT INTO media.control.runs (run_id, ended_at) "
        "VALUES (:r, current_timestamp() + INTERVAL :d DAYS)",
        {"r": "r1", "d": 7},
    )
    row = wh.execute("SELECT * FROM media.control.runs", {}).as_dicts()[0]

    when = datetime.fromisoformat(str(row["ended_at"]))
    ahead = (when - datetime.now(timezone.utc)).total_seconds()
    assert 6 * 86400 < ahead < 8 * 86400


def test_an_unsupported_expression_is_refused_not_nulled():
    """The point of the change: an expression the engine cannot evaluate
    must fail loudly rather than write NULL and look like it worked."""
    import pytest

    from alex.errors import BadRequest

    wh = _wh_with_run_table()
    with pytest.raises(BadRequest):
        wh.execute(
            "INSERT INTO media.control.runs (run_id, n) VALUES (:r, sqrt(9))",
            {"r": "r1"},
        )
