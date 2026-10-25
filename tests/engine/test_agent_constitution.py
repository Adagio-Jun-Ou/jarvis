"""Tests for agent/constitution.py — the four-layer constitutional runtime."""
from __future__ import annotations

import pytest

from agent import constitution as c


def test_build_constitution_prompt_contains_every_article():
    arts = [
        {"article_id": "A1", "title": "no sends", "text": "Never send.", "severity": "blocking"},
        {"article_id": "A4", "title": "rights",  "text": "Clear first.", "severity": "blocking"},
    ]
    p = c.build_constitution_prompt(arts)
    assert "A1" in p and "no sends" in p
    assert "A4" in p and "Clear first." in p
    assert "supreme" in p.lower()


def test_critic_rubric_filled_in():
    arts = [{"article_id": "A7", "title": "minimum", "text": "no PII", "severity": "warning"}]
    r = c.build_critic_prompt(arts)
    assert "A7" in r
    assert "no PII" in r


def test_generate_eval_scenarios_one_per_article():
    arts = [
        {"article_id": "A1", "title": "t1", "text": "x"},
        {"article_id": "A2", "title": "t2", "text": "y"},
    ]
    scenarios = c.generate_eval_scenarios(arts)
    assert len(scenarios) == 2
    for s in scenarios:
        assert "inputs" in s and "messages" in s["inputs"]
        assert "expectations" in s
        assert s["inputs"]["messages"][0]["role"] == "user"


def test_interceptor_blocks_prohibition():
    arts = [{
        "article_id": "A1", "title": "no sends", "text": "x",
        "severity": "blocking",
        "rule": {"deny_capabilities": ["email.send"]},
    }]
    log = []
    def vlog(article_id, action, disposition, detail):
        log.append({"article_id": article_id, "action": action,
                    "disposition": disposition, "detail": detail})
    ic = c.ConstitutionalInterceptor(arts, vlog, "s1", "u@x.com")
    with pytest.raises(c.ConstitutionalBlock) as ei:
        ic.check("send_email", {})
    assert ei.value.article_id == "A1"
    assert log[0]["disposition"] == "blocked"


def test_interceptor_warns_on_warning_severity():
    arts = [{
        "article_id": "A7", "title": "PII cap", "text": "x",
        "severity": "warning",
        "rule": {"deny_capabilities": ["email.send"]},
    }]
    log = []
    def vlog(article_id, action, disposition, detail):
        log.append({"disposition": disposition})
    ic = c.ConstitutionalInterceptor(arts, vlog, "s1", "u@x.com")
    # A 'warning' article does NOT raise — the L4 gate can still fail
    # the deploy, but the agent is allowed to proceed.
    ic.check("send_email", {})
    assert log[0]["disposition"] == "warned"


def test_interceptor_procedure_requires_prior_step():
    arts = [{
        "article_id": "A4", "title": "rights first", "text": "x",
        "severity": "blocking",
        "rule": {
            "procedure": {
                "trigger_tools": ["draft_email_reply"],
                "required_prior": [{"tool": "check_rights", "must_return_not": "NOT_CLEARED"}],
            }
        },
    }]
    log = []
    ic = c.ConstitutionalInterceptor(arts, log.append, "s1", "u@x.com")
    # No prior check_rights call — must block.
    with pytest.raises(c.ConstitutionalBlock) as ei:
        ic.check("draft_email_reply", {"body": "hi"})
    assert "check_rights" in ei.value.detail


def test_interceptor_blocks_registry_only_tool_capability():
    # vm_exec is NOT in the legacy fallback map; its capability (compute.shell)
    # is only known via the tool registry. A deny_capabilities article must
    # still block it — previously it fell through to the bare tool name.
    arts = [{
        "article_id": "A9", "title": "no shell", "text": "x",
        "severity": "blocking",
        "rule": {"deny_capabilities": ["compute.shell"]},
    }]
    ic = c.ConstitutionalInterceptor(arts, lambda *a: None, "s1", "u@x.com")
    with pytest.raises(c.ConstitutionalBlock) as ei:
        ic.check("vm_exec", {"argv": ["ls"]})
    assert ei.value.article_id == "A9"


def test_interceptor_warning_procedure_missing_prior_does_not_crash():
    # A warning-severity procedure with a missing prior step must WARN, not
    # raise, and must not KeyError-crash reaching the must_return_not check.
    arts = [{
        "article_id": "A4", "title": "rights first", "text": "x",
        "severity": "warning",
        "rule": {
            "procedure": {
                "trigger_tools": ["draft_email_reply"],
                "required_prior": [
                    {"tool": "check_rights", "must_return_not": "NOT_CLEARED"}
                ],
            }
        },
    }]
    events = []

    def vlog(article_id, action, disposition, detail):
        events.append((article_id, disposition, detail))

    ic = c.ConstitutionalInterceptor(arts, vlog, "s1", "u@x.com")
    ic.check("draft_email_reply", {"body": "hi"})  # must not raise / crash
    assert any(disp == "warned" for _, disp, _ in events)


def test_interceptor_procedure_passes_with_required_prior():
    arts = [{
        "article_id": "A4", "title": "rights first", "text": "x",
        "severity": "blocking",
        "rule": {
            "procedure": {
                "trigger_tools": ["draft_email_reply"],
                "required_prior": [{"tool": "check_rights", "must_return_not": "NOT_CLEARED"}],
            }
        },
    }]
    log = []
    ic = c.ConstitutionalInterceptor(arts, log.append, "s1", "u@x.com")
    ic.record("check_rights", {"asset_id": "A1"}, "CLEARED — license covers 2026")
    ic.check("draft_email_reply", {"body": "hi"})  # does not raise


def test_interceptor_procedure_blocks_if_prior_returned_forbidden_value():
    arts = [{
        "article_id": "A4", "title": "rights first", "text": "x",
        "severity": "blocking",
        "rule": {
            "procedure": {
                "trigger_tools": ["draft_email_reply"],
                "required_prior": [{"tool": "check_rights", "must_return_not": "NOT_CLEARED"}],
            }
        },
    }]
    log = []
    ic = c.ConstitutionalInterceptor(arts, log.append, "s1", "u@x.com")
    ic.record("check_rights", {"asset_id": "A1"}, "NOT_CLEARED: missing license")
    with pytest.raises(c.ConstitutionalBlock) as ei:
        ic.check("draft_email_reply", {"body": "hi"})
    assert "NOT_CLEARED" in ei.value.detail
