"""Tests for agent/guardrails.py — the dual-ACL gate."""
from __future__ import annotations

import pytest

from agent import guardrails as g
from agent.constitution import ConstitutionalInterceptor
from alex.errors import Denied
from alex.warehouse import InMemoryWarehouse


@pytest.fixture
def wh():
    w = InMemoryWarehouse()
    w.register("media.control.employee_acl", ["user_email", "capability"],
               pk=["user_email", "capability"], rows=[
        ("alice@x.com", "tasks.create"),
        ("alice@x.com", "email.draft"),
    ])
    w.register("media.control.instance_acl", ["instance_scope", "capability"],
               pk=["instance_scope", "capability"], rows=[
        ("x.com", "tasks.create"),
        ("x.com", "email.draft"),
        ("x.com", "files.write"),
    ])
    return w


def _sql(wh):
    def f(statement, parameters=None):
        return wh.execute(statement, parameters or {}).as_dicts()
    return f


def test_user_with_capability_passes(wh):
    log = []
    gate = g.guarded(sql=_sql(wh), user_email="alice@x.com", instance_scope="x.com",
                     log_action=lambda **kw: log.append(kw))
    @gate
    def create_task(title: str) -> str:
        return f"created {title}"
    # We registered 'create_task' in the test fixture's ACL.
    # The decorator infers capability from the function's __name__ which
    # is 'create_task', and that maps to 'tasks.create' in TOOL_CAPABILITY.
    assert create_task("hi") == "created hi"
    assert log == []  # no denials


def test_user_without_capability_is_denied(wh):
    log = []
    gate = g.guarded(sql=_sql(wh), user_email="bob@x.com", instance_scope="x.com",
                     log_action=lambda **kw: log.append(kw))
    @gate
    def create_task(title: str) -> str:
        return "ok"
    with pytest.raises(Denied):
        create_task("hi")
    assert log and log[0]["action_type"] == "denied"


def test_instance_acl_blocks_capability_not_in_ceiling(wh):
    # user has the employee half, but instance_acl denies the capability
    # (no row for 'x.com' / 'tasks.create'). This is intentional: the
    # INTERSECTION gate says no.
    wh.drop("media.control.instance_acl")
    wh.register("media.control.instance_acl", ["instance_scope", "capability"],
               pk=["instance_scope", "capability"], rows=[
        # 'tasks.create' is not in the instance ceiling
        ("x.com", "files.read"),
    ])
    log = []
    gate = g.guarded(sql=_sql(wh), user_email="alice@x.com", instance_scope="x.com",
                     log_action=lambda **kw: log.append(kw))
    @gate
    def create_task(title: str) -> str:
        return "ok"
    with pytest.raises(Denied):
        create_task("hi")


def test_constitutional_interceptor_blocks_denied(wh):
    log = []
    g.guarded(sql=_sql(wh), user_email="alice@x.com", instance_scope="x.com",
              log_action=lambda **kw: log.append(kw), interceptor=None)
    articles = [{
        "article_id": "A1", "title": "no sends", "text": "x",
        "severity": "blocking",
        "rule": {"deny_capabilities": ["email.draft"]},
    }]
    ic = ConstitutionalInterceptor(articles, lambda *a, **kw: log.append(("v", a, kw)),
                                   "s1", "alice@x.com")
    gate2 = g.guarded(sql=_sql(wh), user_email="alice@x.com", instance_scope="x.com",
                      log_action=lambda **kw: log.append(("deny", kw)),
                      interceptor=ic)
    @gate2
    def draft_email_reply(body: str) -> str:
        return "drafted"
    from agent.constitution import ConstitutionalBlock
    with pytest.raises(ConstitutionalBlock):
        draft_email_reply("hi")
    assert any(e[0] == "deny" for e in log)


def test_interceptor_must_have_field_sees_real_kwargs(wh):
    # A procedure article requiring the 'asset_id' field must see it in the
    # tool's actual kwargs. The old {"args":..,"kwargs":..} wrapper hid it,
    # so a compliant call was wrongly blocked and a missing field slipped by.
    articles = [{
        "article_id": "A4", "title": "asset required", "text": "x",
        "severity": "blocking",
        "rule": {"procedure": {
            "trigger_tools": ["draft_email_reply"],
            "required_prior": [{"must_have_field": "asset_id"}],
        }},
    }]
    ic = ConstitutionalInterceptor(articles, lambda *a, **k: None, "s1", "alice@x.com")
    gate = g.guarded(sql=_sql(wh), user_email="alice@x.com", instance_scope="x.com",
                     log_action=lambda **kw: None, interceptor=ic)

    @gate
    def draft_email_reply(**kwargs) -> str:
        return "drafted"

    from agent.constitution import ConstitutionalBlock
    # Missing asset_id -> blocked.
    with pytest.raises(ConstitutionalBlock):
        draft_email_reply(body="hi")
    # Present asset_id -> allowed (proves the field is actually seen).
    assert draft_email_reply(body="hi", asset_id="A1") == "drafted"
