"""Cost controls: the approved terminal library, the concurrent-run
cap, and the GLM cost spine.

A Databricks App runs on a small fixed container (Medium: 2 vCPU /
6 GB, 0.5 DBU/hr) and the LLM bills per token — so cost control is
three bounds, each pinned here:

* the agent's pip universe is the CURATED data/ML set (supply-chain
  and CPU control; operators extend via env, the model cannot);
* one app instance works at most N tasks at once (429 past the cap);
* the default model route is GLM 5.3 Flash, in the approved catalog,
  with GLM 5.3 as the frontier pick.
"""

from __future__ import annotations

import pytest

from agent import python_tool


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("ALEX_ENV", "dev")
    monkeypatch.delenv("ALEX_EXTRA_PACKAGES", raising=False)
    from alex import config, testing

    config._settings = None
    testing._fakes = None
    yield
    config._settings = None
    testing._fakes = None


# ---- the approved library --------------------------------------------------


def test_pip_refuses_packages_outside_the_approved_library(env):
    out = python_tool.pip_install(["leftpad-utils"])
    assert out.startswith("[error]")
    assert "approved library" in out
    assert "ALEX_EXTRA_PACKAGES" in out, "the message must say the way out"


def test_the_check_sees_through_pins_extras_and_underscores(env):
    # Version pins and extras don't dodge the check; nor does the
    # underscore/dash spelling difference PyPI treats as identical.
    assert python_tool._package_base_name("pandas[excel]==2.2.0") == "pandas"
    assert python_tool._package_base_name("python_docx") == "python-docx"
    out = python_tool.pip_install(["totally_not_real[extra]>=1.0"])
    assert out.startswith("[error]") and "approved" in out


def test_operators_extend_the_set_by_env_never_the_model(env, monkeypatch):
    monkeypatch.setenv("ALEX_EXTRA_PACKAGES", "acme-internal, other_pkg")
    names = python_tool._approved_names()
    assert "acme-internal" in names
    assert "other-pkg" in names, "extensions normalize like PyPI names"
    import inspect

    params = inspect.signature(python_tool.pip_install).parameters
    assert "extra" not in params and "approved" not in params, (
        "the approved set must not be a tool argument"
    )


def test_the_ml_stack_is_genuinely_covered(env):
    for pkg in ("numpy", "pandas", "scikit-learn", "torch", "matplotlib",
                "polars", "duckdb", "mlflow", "databricks-sdk"):
        assert pkg in python_tool.APPROVED_PACKAGES, pkg


# ---- the concurrent-run cap ------------------------------------------------


def test_the_app_refuses_runs_past_its_concurrency_cap(env, monkeypatch):
    monkeypatch.setenv("ALEX_MAX_CONCURRENT_RUNS", "0")
    from agent import taskboard as tb
    from alex.testing import get_fakes
    from app._dev_seed import TASK_CREW_COLS, TASK_MESSAGES_COLS, TASK_RUNS_COLS

    tb.reset_runs()
    wh = get_fakes().warehouse
    wh.register(
        "media.alex.tasks",
        ["id", "owner_email", "assignee_email", "title", "description",
         "status", "priority", "due_at", "source", "source_ref",
         "created_at", "updated_at", "closed_at"],
        pk=["id"],
    )
    wh.register("media.control.task_crew", TASK_CREW_COLS)
    wh.register("media.control.task_messages", TASK_MESSAGES_COLS,
                pk=["message_id"])
    wh.register("media.control.task_runs", TASK_RUNS_COLS, pk=["run_id"])

    def sql(stmt, params=None):
        res = wh.execute(stmt, params or {})
        as_dicts = getattr(res, "as_dicts", None)
        return as_dicts() if callable(as_dicts) else list(res or [])

    sql("INSERT INTO media.alex.tasks (id, owner_email, title, status) "
        "VALUES ('t-cap', 'dana@alex.dev', 'Capped', 'todo')")

    class P:
        email = "dana@alex.dev"
        is_admin = False

    with pytest.raises(tb.RunCapacity):
        tb.start_run(sql, P(), "t-cap", background=False,
                     vm_factory=lambda **kw: None)
    tb.reset_runs()


# ---- the GLM cost spine ----------------------------------------------------


def test_the_default_endpoint_is_glm_flash_and_catalogued(env):
    from agent.models import CHEAP_MODEL, DEFAULT_MODEL, MODEL_CATALOG
    from alex.config import get_settings

    endpoints = {r.endpoint for r in MODEL_CATALOG.values()}
    assert get_settings().default_llm_endpoint == "databricks-glm-5-3-flash"
    assert "databricks-glm-5-3-flash" in endpoints, (
        "the configured default must pass the approved-endpoint gate"
    )
    assert MODEL_CATALOG[CHEAP_MODEL].endpoint == "databricks-glm-5-3-flash"
    assert MODEL_CATALOG[DEFAULT_MODEL].endpoint == "databricks-glm-5-3"
    assert MODEL_CATALOG[DEFAULT_MODEL].supports_tools


def test_auto_routes_cheap_to_flash_and_frontier_to_glm(env):
    from agent.models import resolve_model

    assert resolve_model("auto", "what time is the standup?").endpoint == (
        "databricks-glm-5-3-flash"
    )
    assert resolve_model(
        "auto", "analyze the contract rights for the Northwind license"
    ).endpoint == "databricks-glm-5-3"


def test_the_gateway_route_matches_the_default(env):
    from pathlib import Path

    yml = (Path(__file__).resolve().parents[2] / "governance"
           / "ai_gateway.yml").read_text(encoding="utf-8")
    assert "route: databricks-glm-5-3-flash" in yml
