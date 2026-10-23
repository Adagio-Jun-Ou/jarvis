"""L2 — the constitution the VM actually enforces.

The interceptor was always in the call path: `_invoke_tool` checked it
before running any tool, the design documented it as one of the
constitutional layers, and the SSE protocol had a `law_block` event for
its denials. But nothing ever handed it any articles. `make_vm` did not
pass `constitution=`, so every VM the application built fell back to
`ConstitutionalInterceptor(articles=[], log_violation=no-op)` — a check
that could not deny anything and could not record that it had not.

The whole suite passed in that state, which is the point of these tests:
they assert the layer *does something*, not merely that it is called.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from agent.omnigent import MockExecutor
from agent.vm import make_vm

VIOLATIONS = "media.control.constitution_violations"
CONSTITUTION_COLS = [
    "article_id", "article_type", "title", "text",
    "machine_rule", "severity", "version", "superseded_by", "effective_from",
]
VIOLATION_COLS = [
    "id", "ts", "user_email", "session_id",
    "article_id", "action", "disposition", "detail",
]


@pytest.fixture
def warehouse(monkeypatch):
    monkeypatch.setenv("ALEX_ENV", "dev")
    from alex import config, testing

    config._settings = None
    testing._fakes = None
    wh = testing.get_fakes().warehouse
    wh.register("media.control.constitution", CONSTITUTION_COLS, pk=["article_id"])
    wh.register(VIOLATIONS, VIOLATION_COLS, pk=["id"])
    yield wh
    config._settings = None
    testing._fakes = None


def _article(wh, article_id: str, rule: str, *, severity: str = "blocking") -> None:
    # effective_from must be set and in the past: load_articles filters on
    # `effective_from <= current_timestamp()`, so a NULL here silently
    # excludes the article and the test would pass for the wrong reason.
    wh.execute(
        "INSERT INTO media.control.constitution "
        "(article_id, article_type, title, text, machine_rule, severity, "
        " effective_from) "
        "VALUES (:a, :ty, :ti, :tx, :r, :s, :ef)",
        {
            "a": article_id, "ty": "prohibition", "ti": "No shell",
            "tx": "The agent may not run shell commands.",
            "r": rule, "s": severity,
            "ef": datetime.now(timezone.utc) - timedelta(days=1),
        },
    )


def _vm(wh):
    return make_vm(
        session_id="s-law",
        user_email="dana@alex.dev",
        executor=MockExecutor(),
        warehouse=wh,
    )


def test_vm_loads_the_articles_actually_in_force(warehouse):
    _article(warehouse, "A9", '{"deny_capabilities": ["compute.shell"]}')

    interceptor = _vm(warehouse)._session.constitution

    ids = [a["article_id"] for a in interceptor.articles]
    assert ids == ["A9"], "the VM built an interceptor with no constitution"


def test_a_prohibited_capability_is_blocked(warehouse):
    from agent.constitution import ConstitutionalBlock

    _article(warehouse, "A9", '{"deny_capabilities": ["compute.shell"]}')
    interceptor = _vm(warehouse)._session.constitution

    with pytest.raises(ConstitutionalBlock):
        interceptor.check("vm_exec", {"argv": ["rm", "-rf", "/"]})


def test_a_block_is_recorded_where_oversight_can_see_it(warehouse):
    from agent.constitution import ConstitutionalBlock

    _article(warehouse, "A9", '{"deny_capabilities": ["compute.shell"]}')
    interceptor = _vm(warehouse)._session.constitution

    with pytest.raises(ConstitutionalBlock):
        interceptor.check("vm_exec", {})

    rows = warehouse.execute(f"SELECT * FROM {VIOLATIONS}", {}).as_dicts()
    assert len(rows) == 1, "the denial left no record"
    assert rows[0]["article_id"] == "A9"
    assert rows[0]["disposition"] == "blocked"
    assert rows[0]["user_email"] == "dana@alex.dev"
    assert rows[0]["session_id"] == "s-law"


def test_a_warning_article_records_without_blocking(warehouse):
    _article(
        warehouse, "A10", '{"deny_capabilities": ["compute.shell"]}',
        severity="warning",
    )
    interceptor = _vm(warehouse)._session.constitution

    interceptor.check("vm_exec", {})  # must not raise

    rows = warehouse.execute(f"SELECT * FROM {VIOLATIONS}", {}).as_dicts()
    assert [r["disposition"] for r in rows] == ["warned"]


def test_unrelated_tools_are_untouched(warehouse):
    _article(warehouse, "A9", '{"deny_capabilities": ["compute.shell"]}')
    interceptor = _vm(warehouse)._session.constitution

    interceptor.check("list_my_tasks", {})

    assert warehouse.execute(f"SELECT * FROM {VIOLATIONS}", {}).as_dicts() == []


def test_a_missing_constitution_table_does_not_break_the_session(monkeypatch):
    """Degrading to no articles beats refusing to start a chat — the
    governor and L3 still apply — but it must be a warning, not silence."""
    monkeypatch.setenv("ALEX_ENV", "dev")
    from alex import config, testing

    config._settings = None
    testing._fakes = None
    wh = testing.get_fakes().warehouse  # no constitution table registered

    vm = make_vm(session_id="s2", user_email="dana@alex.dev",
                 executor=MockExecutor(), warehouse=wh)

    assert vm._session.constitution.articles == []
    config._settings = None
    testing._fakes = None


def test_an_explicitly_injected_constitution_still_wins(warehouse):
    """Tests and the SDK inject their own; loading must not override it."""
    from agent.constitution import ConstitutionalInterceptor
    from agent.vm import VM, HarnessConfig

    _article(warehouse, "A9", '{"deny_capabilities": ["compute.shell"]}')
    mine = ConstitutionalInterceptor(
        articles=[], log_violation=lambda *a, **k: None,
        session_id="s3", user_email="dana@alex.dev",
    )
    vm = VM(
        session_id="s3", user_email="dana@alex.dev",
        config=HarnessConfig(
            mode="dev", executor=MockExecutor(),
            warehouse=warehouse, constitution=mine,
        ),
    )

    assert vm._session.constitution is mine


# ---- L1: the model is told what the law says ------------------------------
# The interceptor refusing a call is the last resort, not the mechanism.
# An agent that has never been shown the constitution learns each article
# by having a tool call bounce, which is a slow, expensive and confusing
# way to discover a rule it could simply have been told.


def test_the_constitution_reaches_the_system_prompt(warehouse):
    _article(warehouse, "A9", '{"deny_capabilities": ["compute.shell"]}')

    block = _vm(warehouse)._constitution_block()

    assert block, "the model is never shown the constitution"
    assert "A9" in block
    assert "may not run shell commands" in block


def test_the_prompt_and_the_interceptor_agree(warehouse):
    """Both read the same loaded articles, so they cannot drift into
    telling the model one thing and enforcing another."""
    _article(warehouse, "A9", '{"deny_capabilities": ["compute.shell"]}')
    _article(warehouse, "A11", '{"deny_capabilities": ["email.send"]}')
    vm = _vm(warehouse)

    block = vm._constitution_block()
    enforced = {a["article_id"] for a in vm._session.constitution.articles}

    assert enforced == {"A9", "A11"}
    for article_id in enforced:
        assert article_id in block, f"{article_id} is enforced but never stated"


def test_no_articles_means_no_empty_heading_in_the_prompt(monkeypatch):
    """An empty constitution section is worse than none: it reads as
    'there are no rules' rather than 'the rules could not be loaded'."""
    monkeypatch.setenv("ALEX_ENV", "dev")
    from alex import config, testing

    config._settings = None
    testing._fakes = None
    vm = make_vm(
        session_id="s4", user_email="dana@alex.dev",
        executor=MockExecutor(), warehouse=testing.get_fakes().warehouse,
    )

    assert vm._constitution_block() == ""
    config._settings = None
    testing._fakes = None
