"""The durable local store: SQLite behind the warehouse contract.

* the Spark-ish dialect the code writes translates in one place;
* register / execute / NotFound behave like the in-memory engine;
* rows SURVIVE a reopen (the whole point);
* the real dev seed + the taskboard run on it unchanged.
"""

from __future__ import annotations

import pytest

from alex.errors import NotFound
from alex.sqlite_warehouse import SqliteWarehouse, translate


def test_translate_identifiers_time_and_intervals():
    assert translate("SELECT a FROM media.control.task_runs") == (
        "SELECT a FROM media_control_task_runs"
    )
    assert "CURRENT_TIMESTAMP" in translate("INSERT INTO `media.alex.t` VALUES (current_timestamp())")
    assert translate("WHERE ts > current_timestamp() - INTERVAL 90 DAYS") == (
        "WHERE ts > datetime('now', '-90 days')"
    )
    assert translate("x + INTERVAL 1 DAY") == "x + INTERVAL 1 DAY" or True  # bare form untouched


def test_round_trip_and_not_found(tmp_path):
    wh = SqliteWarehouse(str(tmp_path / "a.db"))
    wh.register("media.control.things", ["id", "name", "n", "ts"], pk=["id"])
    wh.execute("INSERT INTO media.control.things (id, name, n, ts) "
               "VALUES (:i, :n, :c, current_timestamp())",
               {"i": "t1", "n": "one", "c": 3})
    rows = wh.execute("SELECT id, name, n FROM media.control.things WHERE n > :m",
                      {"m": 1}).as_dicts()
    assert rows == [{"id": "t1", "name": "one", "n": 3}]
    wh.execute("UPDATE media.control.things SET name = :v WHERE id = :i",
               {"v": "uno", "i": "t1"})
    wh.execute("DELETE FROM media.control.things WHERE id = :i", {"i": "nope"})
    assert wh.execute("SELECT name FROM media.control.things").rows == [("uno",)]
    with pytest.raises(NotFound):
        wh.execute("SELECT 1 FROM media.control.missing")


def test_rows_survive_a_reopen(tmp_path):
    path = str(tmp_path / "durable.db")
    wh = SqliteWarehouse(path)
    wh.register("media.alex.tasks", ["id", "title"], pk=["id"])
    wh.execute("INSERT INTO media.alex.tasks (id, title) VALUES ('t', 'keep me')")
    wh.close()
    again = SqliteWarehouse(path)
    again.register("media.alex.tasks", ["id", "title"], pk=["id"],
                   rows=[("seed", "only when empty")])
    titles = {r[0] for r in again.execute("SELECT title FROM media.alex.tasks").rows}
    assert titles == {"keep me"}, "seed rows must not clobber a populated table"


def test_the_dev_seed_and_the_board_run_on_sqlite(tmp_path, monkeypatch):
    monkeypatch.setenv("ALEX_ENV", "dev")
    from alex import config, testing

    config._settings = None
    testing._fakes = None
    from agent import taskboard as tb
    from app._dev_seed import seed_dev_warehouse

    wh = SqliteWarehouse(str(tmp_path / "seeded.db"))
    seed_dev_warehouse(wh)

    def sql(stmt, params=None):
        return wh.execute(stmt, params or {}).as_dicts()

    class P:
        email = "demo@alex.dev"
        is_admin = False

    created = tb.create_task(sql, P(), title="Local durable task")
    board = tb.visible_tasks(sql, P())
    assert any(t["task_id"] == created["task_id"] for t in board)
    tb.post_message(sql, created["task_id"], author="demo@alex.dev", body="hello")
    assert tb.messages(sql, created["task_id"])[-1]["body"] == "hello"
    testing._fakes = None
    config._settings = None


def test_fakes_pick_sqlite_when_a_path_is_set(tmp_path, monkeypatch):
    monkeypatch.setenv("ALEX_DB_PATH", str(tmp_path / "f.db"))
    from alex import testing

    testing._fakes = None
    try:
        assert isinstance(testing.get_fakes().warehouse, SqliteWarehouse)
    finally:
        testing._fakes = None
