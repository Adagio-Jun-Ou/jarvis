"""Tests for agent/practices.py — the best-practice library.

The library's job is to be *guidance*, not search results. These tests
pin the properties that make that true: capability-bound practices fire
regardless of phrasing, mandatory practices are never crowded out,
revisions supersede rather than overwrite, and the rendered block reads
as instruction.
"""

from __future__ import annotations

import pytest

from agent import access, practices


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("ALEX_ENV", "dev")
    from alex import config, testing

    config._settings = None
    testing._fakes = None
    access.invalidate()
    yield
    config._settings = None
    testing._fakes = None
    access.invalidate()


def principal(role="manager", **kw):
    return access.Principal(
        email=kw.pop("email", "m@x.com"),
        role=role,
        capabilities=frozenset(kw.pop("capabilities", set())),
        **kw,
    )


SQL = None  # dev mode stores in memory; sql is unused


@pytest.fixture
def library(env):
    practices.upsert(
        SQL,
        title="Cite every number",
        guidance="Every figure must carry its source.",
        authority=practices.MANDATORY,
        applies_to=["budget", "revenue", "forecast"],
    )
    practices.upsert(
        SQL,
        title="Confirm before sending mail",
        guidance="Draft, show, and wait for confirmation.",
        authority=practices.MANDATORY,
        capabilities=["email.send"],
    )
    practices.upsert(
        SQL,
        title="Lead with the answer",
        guidance="Conclusion first, detail after.",
        authority=practices.RECOMMENDED,
    )
    practices.upsert(
        SQL,
        title="Escalate renewal risk",
        guidance="Name the account owner and a date.",
        authority=practices.ADVISORY,
        applies_to=["renewal", "churn", "account"],
    )
    return SQL


# ---- publishing ----------------------------------------------------------


def test_a_practice_needs_a_title_and_guidance(env):
    with pytest.raises(ValueError):
        practices.upsert(SQL, title="", guidance="x")
    with pytest.raises(ValueError):
        practices.upsert(SQL, title="x", guidance="  ")


def test_unknown_authority_falls_back_to_recommended(env):
    p = practices.upsert(SQL, title="t", guidance="g", authority="critical")
    assert p.authority == practices.RECOMMENDED


def test_revision_supersedes_rather_than_overwrites(env):
    v1 = practices.upsert(
        SQL, title="Expenses", guidance="Book 14 days ahead.",
        authority=practices.RECOMMENDED,
    )
    v2 = practices.upsert(
        SQL, title="Expenses", guidance="Book 21 days ahead.",
        authority=practices.RECOMMENDED, supersedes=v1.practice_id,
    )
    assert v2.version == 2
    old = practices.get(SQL, v1.practice_id)
    assert old is not None, "the superseded version must remain on record"
    assert old.superseded_by == v2.practice_id
    assert not old.current
    # Only the current version is in force.
    titles = [p.practice_id for p in practices.visible(SQL, principal(role="admin"))]
    assert v2.practice_id in titles and v1.practice_id not in titles


def test_retire_takes_a_practice_out_of_force_without_deleting(env):
    p = practices.upsert(SQL, title="t", guidance="g")
    assert practices.retire(SQL, p.practice_id) is True
    assert practices.get(SQL, p.practice_id) is not None
    assert p.practice_id not in [
        x.practice_id for x in practices.visible(SQL, principal(role="admin"))
    ]


def test_retire_unknown_practice_is_false(env):
    assert practices.retire(SQL, "bp-nope") is False


# ---- visibility ----------------------------------------------------------


def test_a_practice_above_your_clearance_is_invisible(env):
    practices.upsert(
        SQL, title="Board protocol", guidance="secret",
        classification="restricted",
    )
    assert practices.visible(SQL, principal(role="member")) == []
    assert len(practices.visible(SQL, principal(role="admin"))) == 1


# ---- matching: the part that makes this guidance ------------------------


def test_capability_match_fires_regardless_of_phrasing(library):
    """The whole point: a mandatory practice bound to a capability cannot
    be phrased around."""
    p = principal(capabilities={"email.send"})
    # The query says nothing about mail.
    matches = practices.match(
        library, p, query="handle this for me", capabilities_in_play=["email.send"]
    )
    titles = [m.practice.title for m in matches]
    assert "Confirm before sending mail" in titles
    hit = next(m for m in matches if m.practice.title == "Confirm before sending mail")
    assert hit.reason == "capability"


def test_capability_bound_practice_stays_out_when_the_tool_is_not_in_play(library):
    matches = practices.match(library, principal(), query="what is the weather")
    assert "Confirm before sending mail" not in [m.practice.title for m in matches]


def test_subject_match_on_topic_overlap(library):
    matches = practices.match(library, principal(), query="what is the Q1 budget forecast?")
    hit = next(m for m in matches if m.practice.title == "Cite every number")
    assert hit.reason in ("subject", "semantic")


def test_unconditional_practice_always_applies(library):
    matches = practices.match(library, principal(), query="anything at all")
    hit = next(m for m in matches if m.practice.title == "Lead with the answer")
    assert hit.reason == "always"


def test_mandatory_practices_are_never_crowded_out(library):
    """A strongly-matching advisory practice must not displace a
    weakly-matching mandatory one. Here the query is squarely about
    renewals (advisory, strong subject hit) while the mandatory mail
    practice matches only because the tool is in play — and it still
    takes the single available slot."""
    matches = practices.match(
        library, principal(capabilities={"email.send"}),
        query="renewal churn account risk", capabilities_in_play=["email.send"],
        limit=1,
    )
    assert len(matches) == 1
    assert matches[0].practice.title == "Confirm before sending mail"
    assert matches[0].practice.is_mandatory


def test_a_subject_practice_matches_on_a_single_declared_subject(library):
    """Declaring several subjects must not make a practice harder to
    match — one declared subject appearing in the request is enough."""
    matches = practices.match(library, principal(), query="what is the budget?")
    hit = next(m for m in matches if m.practice.title == "Cite every number")
    assert hit.reason == "subject"


def test_match_respects_the_limit(library):
    assert len(practices.match(library, principal(), query="budget", limit=1)) == 1


def test_empty_library_matches_nothing(env):
    assert practices.match(SQL, principal(), query="anything") == []


# ---- the prompt block ----------------------------------------------------


def test_guidance_block_states_authority_explicitly(library):
    matches = practices.match(library, principal(), query="Q1 budget forecast")
    block = practices.guidance_block(matches)
    assert "MANDATORY" in block
    assert "must follow" in block
    # Rationale and anti-patterns give the model something to act on.
    assert "[bp-" in block


def test_guidance_block_puts_mandatory_first(library):
    matches = practices.match(library, principal(), query="budget renewal")
    block = practices.guidance_block(matches)
    assert block.index("MANDATORY") < block.index("RECOMMENDED / ADVISORY")


def test_guidance_block_respects_its_budget(library):
    matches = practices.match(library, principal(), query="budget renewal account")
    assert len(practices.guidance_block(matches, max_chars=300)) < 1200


def test_guidance_block_empty_is_empty_string():
    assert practices.guidance_block([]) == ""


# ---- compliance ----------------------------------------------------------


def test_compliance_flags_an_uncited_mandatory_practice(library):
    matches = practices.match(library, principal(), query="Q1 budget forecast")
    report = practices.compliance_report("The budget is about 1.2M.", matches)
    assert report["mandatory_in_play"] >= 1
    assert report["compliant"] is False
    assert report["uncited"]


def test_compliance_passes_when_the_practice_is_cited(library):
    matches = practices.match(library, principal(), query="Q1 budget forecast")
    mandatory = next(m for m in matches if m.practice.is_mandatory)
    answer = f"Following {mandatory.practice.title}, the budget is 1.2M [1]."
    report = practices.compliance_report(answer, matches)
    assert report["compliant"] is True
    assert report["cited"][0]["practice_id"] == mandatory.practice.practice_id


def test_compliance_with_no_mandatory_practices_is_compliant(env):
    practices.upsert(SQL, title="t", guidance="g", authority=practices.ADVISORY)
    matches = practices.match(SQL, principal(), query="anything")
    assert practices.compliance_report("whatever", matches)["compliant"] is True
