"""Tests for agent/preferences.py — the personalised workspace.

The property that matters is precedence: workspace policy beats user
preference, and a refused change says so rather than appearing to save
and then reverting.
"""

from __future__ import annotations

import json

import pytest

from agent import preferences as prefs


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
def sql(env):
    from alex.testing import get_fakes

    wh = get_fakes().warehouse
    wh.register(
        "media.control.user_preferences",
        ["user_email", "name", "value", "updated_at"],
        pk=["user_email", "name"],
    )
    wh.register(
        "media.control.workspace_policy",
        ["name", "value", "rationale", "set_by", "set_at"],
        pk=["name"],
    )

    def _sql(stmt, params=None):
        res = wh.execute(stmt, params or {})
        as_dicts = getattr(res, "as_dicts", None)
        return as_dicts() if callable(as_dicts) else list(res or [])

    return _sql


def pin(sql, name, value):
    sql(
        "INSERT INTO media.control.workspace_policy "
        "(name, value, rationale, set_by, set_at) "
        "VALUES (:n, :v, :r, :b, :t)",
        {"n": name, "v": json.dumps(value), "r": "policy", "b": "root@x.com",
         "t": None},
    )


# ---- defaults and reads --------------------------------------------------


def test_a_new_user_gets_the_defaults(sql):
    out = prefs.get(sql, "alice@x.com")
    assert out["preferences"] == prefs.DEFAULTS
    assert out["pinned"] == []


def test_options_are_reported_for_the_ui(sql):
    options = prefs.get(sql, "alice@x.com")["options"]
    assert "dark" in options["theme"]
    assert "compact" in options["density"]
    assert "autopilot" in options["autonomy"]


# ---- writing -------------------------------------------------------------


def test_a_valid_change_is_applied_and_persists(sql):
    out = prefs.set_many(sql, "alice@x.com", {"density": "compact"})
    assert out["applied"] == {"density": "compact"}
    assert prefs.get(sql, "alice@x.com")["preferences"]["density"] == "compact"


def test_updating_the_same_key_twice_does_not_duplicate(sql):
    prefs.set_many(sql, "alice@x.com", {"theme": "light"})
    prefs.set_many(sql, "alice@x.com", {"theme": "dark"})
    assert prefs.get(sql, "alice@x.com")["preferences"]["theme"] == "dark"


def test_an_unknown_preference_is_refused(sql):
    out = prefs.set_many(sql, "alice@x.com", {"font_size": 40})
    assert out["applied"] == {}
    assert out["refused"]["font_size"] == "unknown preference"


def test_an_invalid_value_is_refused(sql):
    out = prefs.set_many(sql, "alice@x.com", {"density": "enormous"})
    assert out["refused"]["density"] == "invalid value"
    assert prefs.get(sql, "alice@x.com")["preferences"]["density"] == "comfortable"


def test_oversized_custom_instructions_are_refused(sql):
    out = prefs.set_many(sql, "alice@x.com", {"custom_instructions": "x" * 5000})
    assert "custom_instructions" in out["refused"]


def test_list_preferences_are_validated(sql):
    assert prefs.set_many(sql, "a@x.com", {"pinned_tabs": "chat"})["refused"]
    assert prefs.set_many(sql, "a@x.com", {"pinned_tabs": ["chat", "board"]})["applied"]


def test_one_bad_key_does_not_block_the_good_ones(sql):
    out = prefs.set_many(
        sql, "alice@x.com", {"theme": "light", "density": "nope"}
    )
    assert out["applied"] == {"theme": "light"}
    assert "density" in out["refused"]


def test_reset_returns_a_user_to_the_defaults(sql):
    prefs.set_many(sql, "alice@x.com", {"theme": "light", "density": "compact"})
    prefs.reset(sql, "alice@x.com")
    assert prefs.get(sql, "alice@x.com")["preferences"] == prefs.DEFAULTS


def test_preferences_are_per_user(sql):
    prefs.set_many(sql, "alice@x.com", {"theme": "light"})
    assert prefs.get(sql, "bob@x.com")["preferences"]["theme"] == "dark"


# ---- policy precedence ---------------------------------------------------


def test_policy_overrides_a_stored_preference(sql):
    prefs.set_many(sql, "alice@x.com", {"autonomy": "autopilot"})
    pin(sql, "autonomy", "suggest")
    out = prefs.get(sql, "alice@x.com")
    assert out["preferences"]["autonomy"] == "suggest"
    assert "autonomy" in out["pinned"]


def test_a_pinned_key_reports_refusal_rather_than_silently_dropping(sql):
    """A setting that appears to save and then reverts is worse than one
    that says no."""
    pin(sql, "autonomy", "suggest")
    out = prefs.set_many(sql, "alice@x.com", {"autonomy": "autopilot"})
    assert out["applied"] == {}
    assert out["refused"]["autonomy"] == "pinned by workspace policy"


def test_an_invalid_policy_value_is_ignored(sql):
    pin(sql, "density", "enormous")
    assert prefs.get(sql, "alice@x.com")["preferences"]["density"] == "comfortable"


def test_policy_on_an_unknown_key_is_ignored(sql):
    pin(sql, "not_a_setting", True)
    assert prefs.get(sql, "alice@x.com")["pinned"] == []


# ---- custom instructions -------------------------------------------------


def test_custom_instructions_render_as_preference_not_policy(sql):
    prefs.set_many(
        sql, "alice@x.com", {"custom_instructions": "Always answer in bullets."}
    )
    block = prefs.custom_instruction_block(sql, "alice@x.com")
    assert "Always answer in bullets." in block
    # Framed so it cannot be used to talk over the constitution.
    assert "mandatory" in block.lower()
    assert "constitution" in block.lower()


def test_no_custom_instructions_means_no_block(sql):
    assert prefs.custom_instruction_block(sql, "alice@x.com") == ""


def test_autonomy_of_reads_the_effective_value(sql):
    assert prefs.autonomy_of(sql, "alice@x.com") == "co-pilot"
    prefs.set_many(sql, "alice@x.com", {"autonomy": "autopilot"})
    assert prefs.autonomy_of(sql, "alice@x.com") == "autopilot"


# ---- resilience ----------------------------------------------------------


def test_a_broken_store_degrades_to_defaults(env):
    def broken(_stmt, _params=None):
        raise RuntimeError("warehouse down")

    assert prefs.get(broken, "alice@x.com")["preferences"] == prefs.DEFAULTS
