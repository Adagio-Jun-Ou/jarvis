"""Tests for agent/auto_review.py — the inline L3 critic."""
from __future__ import annotations

from agent import auto_review as ar


def test_needs_review_flags_risky_tools():
    assert ar.needs_review(["draft_email_reply", "list_my_tasks"])
    assert not ar.needs_review(["list_my_tasks"])
    assert not ar.needs_review([])


def test_parse_handles_fenced_json():
    out = ar._parse_critic_json('```json\n{"verdict": "block", "reason": "x"}\n```')
    assert out["verdict"] == "block"


def test_parse_handles_inline_json():
    out = ar._parse_critic_json('verdict: ok. Some prose. {"verdict": "ok"}')
    assert out["verdict"] == "ok"


def test_parse_defaults_to_revise_on_garbage():
    out = ar._parse_critic_json("not json")
    assert out["verdict"] == "revise"


def test_inline_review_invokes_critic():
    called = []
    def critic(msgs):
        called.append(msgs)
        return '{"verdict": "ok", "reason": "", "article": null}'
    out = ar.inline_review(
        "send a draft", ["draft_email_reply"], "draft text",
        articles=[{"article_id": "A1", "title": "x", "text": "y", "severity": "blocking", "rule": {}}],
        critic_invoke=critic,
    )
    assert out["verdict"] == "ok"
    assert called  # the critic was actually called
    # The system prompt should include the rubric
    sys_msg = called[0][0]["content"]
    assert "ACTIVE ARTICLES" in sys_msg
