"""Tests for the configuration module."""
from __future__ import annotations


def test_default_mode_is_dev_when_unset():
    from alex.config import get_settings, reset_settings
    reset_settings()
    s = get_settings()
    assert s.mode == "dev"
    assert s.catalog == "media"


def test_explicit_prod_mode(monkeypatch):
    monkeypatch.setenv("ALEX_ENV", "production")
    monkeypatch.setenv("WAREHOUSE_ID", "wh-12345")
    from alex.config import get_settings, reset_settings
    reset_settings()
    s = get_settings()
    assert s.mode == "production"
    assert s.warehouse_id == "wh-12345"


def test_unknown_mode_raises(monkeypatch):
    import pytest
    monkeypatch.setenv("ALEX_ENV", "garbage")
    from alex.config import load
    with pytest.raises(ValueError, match="ALEX_ENV"):
        load()


def test_unset_mode_defaults_to_dev(monkeypatch):
    monkeypatch.delenv("ALEX_ENV", raising=False)
    from alex.config import load
    assert load().mode == "dev"


def test_reset_overrides():
    from alex.config import get_settings, reset_settings
    s = reset_settings(catalog="mycat", warehouse_id="wh-xyz")
    assert s.catalog == "mycat"
    assert get_settings() is s


def test_behaviour_knobs_read_from_env(monkeypatch):
    monkeypatch.setenv("ALEX_ENV", "dev")
    monkeypatch.setenv("ALEX_AUTO_REVIEW_SAMPLE_RATE", "0.25")
    monkeypatch.setenv("ALEX_UPLOAD_MAX_BYTES", "1048576")
    monkeypatch.setenv("ALEX_EVAL_PASS_THRESHOLD", "0.9")
    from alex.config import load
    s = load()
    assert s.auto_review_sample_rate == 0.25
    assert s.upload_max_bytes == 1048576
    assert s.eval_pass_threshold == 0.9


def test_out_of_range_knob_raises_at_load(monkeypatch):
    import pytest

    from alex.config import ConfigError, load
    monkeypatch.setenv("ALEX_ENV", "dev")
    monkeypatch.setenv("ALEX_AUTO_REVIEW_SAMPLE_RATE", "2.0")
    with pytest.raises(ConfigError, match="AUTO_REVIEW_SAMPLE_RATE"):
        load()


def test_non_numeric_knob_raises_at_load(monkeypatch):
    import pytest

    from alex.config import ConfigError, load
    monkeypatch.setenv("ALEX_ENV", "dev")
    monkeypatch.setenv("ALEX_UPLOAD_MAX_BYTES", "not-a-number")
    with pytest.raises(ConfigError, match="UPLOAD_MAX_BYTES"):
        load()


def test_blank_knob_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("ALEX_ENV", "dev")
    monkeypatch.setenv("ALEX_EVAL_PASS_THRESHOLD", "")
    from alex.config import load
    assert load().eval_pass_threshold == 0.8


def test_settings_is_frozen():
    import dataclasses

    import pytest

    from alex.config import reset_settings
    s = reset_settings()
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.catalog = "other"  # type: ignore[misc]
