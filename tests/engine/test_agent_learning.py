"""Tests for agent/learning_loop.py and agent/learn_with_you.py."""
from __future__ import annotations

import pytest

from agent import learn_with_you as lwy
from agent import learning_loop as ll
from alex.warehouse import InMemoryWarehouse


@pytest.fixture
def wh():
    w = InMemoryWarehouse()
    w.register("media.control.skills",
               ["skill_id", "owner_email", "name", "trigger_pattern",
                "procedure", "uses_capabilities", "source_kind",
                "status", "success_count", "last_used", "ratified_by",
                "ratified_at", "created_at"],
               pk=["skill_id"], rows=[
        ("s1", "alice@x.com", "Rights clearance",
         "rights clearance for asset",
         array_of_str(["check_rights", "draft_email_reply"]),
         array_of_str(["content.read", "email.draft"]),
         "observed", "active", 5, None, "alice@x.com", None, None),
    ])
    return w


def array_of_str(xs):
    return list(xs)  # InMemoryWarehouse stores as a list


def test_active_skill_hints_returns_empty_when_no_match(wh):
    def sql(stmt, params=None, **kw):
        # The real seam is ``sql(statement, parameters=...)``; accept both
        # spellings so a caller using the keyword form is not silently
        # executed with no parameters at all.
        return wh.execute(stmt, params or kw.get("parameters") or {}).as_dicts()
    out = ll.active_skill_hints(sql, "alice@x.com", "unrelated message")
    assert out == ""


def test_tokens_filter_stopwords():
    # Common function words (>2 chars) must not become match tokens.
    toks = set(ll._tokens("Please send the report about that new thing"))
    assert "the" not in toks
    assert "about" not in toks
    assert "please" not in toks
    assert "that" not in toks
    assert "new" not in toks
    # Content words survive.
    assert "send" in toks
    assert "report" in toks


def test_active_skill_hints_no_false_match_on_stopwords(wh):
    def sql(stmt, params=None, **kw):
        # The real seam is ``sql(statement, parameters=...)``; accept both
        # spellings so a caller using the keyword form is not silently
        # executed with no parameters at all.
        return wh.execute(stmt, params or kw.get("parameters") or {}).as_dicts()
    # Shares only the stopword "for" with the trigger "rights clearance for
    # asset" — must NOT surface the skill.
    out = ll.active_skill_hints(sql, "alice@x.com", "what should I cook for dinner")
    assert out == ""


def test_active_skill_hints_returns_match(wh):
    def sql(stmt, params=None, **kw):
        # The real seam is ``sql(statement, parameters=...)``; accept both
        # spellings so a caller using the keyword form is not silently
        # executed with no parameters at all.
        return wh.execute(stmt, params or kw.get("parameters") or {}).as_dicts()
    out = ll.active_skill_hints(sql, "alice@x.com",
                                "Please do rights clearance for asset A1")
    assert "LEARNED PROCEDURES" in out
    assert "Rights clearance" in out


def test_capture_correction_writes_memory():
    stored = []
    def store(owner, kind, content, source_ref):
        stored.append((owner, kind, content, source_ref))
    corr = lwy.capture_correction(store, "alice@x.com", "old", "new body")
    assert stored
    assert corr.prompt == "Saved as a preference."


def test_capture_correction_empty_diff_is_noop():
    stored = []
    lwy.capture_correction(lambda *a: stored.append(a), "alice@x.com", "same", "same")
    assert stored == []


def test_teach_routine_returns_id():
    def store(owner, kind, content, source_ref):
        pass
    rid = lwy.teach_routine(store, "alice@x.com", "do this: step 1, step 2")
    assert rid


def test_should_suggest_heuristic():
    assert lwy.should_suggest("alice@x.com", "Can you show me how to do this?")
    assert not lwy.should_suggest("alice@x.com", "do the thing")
