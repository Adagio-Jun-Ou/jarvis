"""Tests for agent/access.py — the knowledge authorization decision point.

Security tests are written adversarially: each one names the escalation
it is trying to achieve. A failure here is a data-leak, not a bug.
"""

from __future__ import annotations

import pytest

from agent import access, rag


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


def sql_for(rows: dict):
    """A fake control-plane reader driven by a table -> rows mapping."""

    def _sql(stmt: str, params: dict):
        email = (params or {}).get("e", "")
        if "user_roles" in stmt:
            return [r for r in rows.get("user_roles", []) if r["user_email"] == email]
        if "effective_employee_acl" in stmt:
            return [r for r in rows.get("acl", []) if r["user_email"] == email]
        if "supervision" in stmt:
            return [
                r for r in rows.get("supervision", [])
                if r["supervisor_email"] == email
            ]
        return []

    return _sql


P = access.Principal


# ---- resolution ----------------------------------------------------------


def test_load_principal_reads_role_team_and_reports(env):
    sql = sql_for({
        "user_roles": [
            {"user_email": "m@x.com", "role": "manager", "team": "Platform",
             "granted_scope": None, "clearance": None}
        ],
        "acl": [{"user_email": "m@x.com", "capability": "knowledge.read"}],
        "supervision": [{"supervisor_email": "m@x.com", "report_email": "r@x.com"}],
    })
    p = access.load_principal(sql, "m@x.com", use_cache=False)
    assert p.role == "manager"
    assert p.teams == frozenset({"team:platform"})
    assert p.reports == frozenset({"r@x.com"})
    assert p.clearance == "confidential"


def test_highest_role_wins_across_rows(env):
    sql = sql_for({"user_roles": [
        {"user_email": "a@x.com", "role": "member", "team": "A",
         "granted_scope": None, "clearance": None},
        {"user_email": "a@x.com", "role": "admin", "team": "B",
         "granted_scope": None, "clearance": None},
    ]})
    assert access.load_principal(sql, "a@x.com", use_cache=False).role == "admin"


def test_unknown_user_gets_least_privilege(env):
    p = access.load_principal(sql_for({}), "ghost@x.com", use_cache=False)
    assert p.role == "member" and p.capabilities == frozenset()
    assert p.clearance == "internal"


def test_control_plane_failure_fails_closed(env):
    def broken(_stmt, _params):
        raise RuntimeError("warehouse down")

    p = access.load_principal(broken, "m@x.com", use_cache=False)
    assert p.role == "member", "a failed role lookup must not grant privilege"
    assert p.capabilities == frozenset()


def test_empty_email_is_anonymous(env):
    assert access.load_principal(sql_for({}), "", use_cache=False).email == ""


def test_invalid_role_value_is_ignored(env):
    sql = sql_for({"user_roles": [
        {"user_email": "a@x.com", "role": "superuser", "team": None,
         "granted_scope": None, "clearance": None}
    ]})
    assert access.load_principal(sql, "a@x.com", use_cache=False).role == "member"


# ---- classification ------------------------------------------------------


def test_unknown_classification_is_treated_as_restricted(env):
    # A typo in a classification must narrow access, never widen it.
    assert access.normalise_classification("confidentail") == "restricted"
    assert access.normalise_classification("") == "internal"


def test_member_cannot_read_confidential(env):
    p = P(email="a@x.com", role="member")
    assert not access.clearance_allowed(p, "confidential")
    assert access.clearance_allowed(p, "internal")


def test_manager_cannot_read_restricted(env):
    p = P(email="m@x.com", role="manager")
    assert access.clearance_allowed(p, "confidential")
    assert not access.clearance_allowed(p, "restricted")


def test_admin_reads_every_classification(env):
    p = P(email="admin@x.com", role="admin")
    assert all(access.clearance_allowed(p, c) for c in access.CLASSIFICATIONS)


def test_explicit_clearance_override_raises_but_never_lowers(env):
    raised = P(email="a@x.com", role="member", clearance_override="confidential")
    assert raised.clearance == "confidential"
    lowered = P(email="b@x.com", role="admin", clearance_override="public")
    assert lowered.clearance == "restricted", "override must not reduce clearance"


# ---- scope isolation (the escalation tests) ------------------------------


def test_a_user_reads_their_own_private_scope(env):
    p = P(email="alice@x.com")
    assert access.scope_allowed(p, "user:alice@x.com")


def test_admin_cannot_read_another_users_private_scope(env):
    """Escalation attempt: admin role used to read a colleague's private
    material. Oversight comes from the audit ledger, not from content."""
    admin = P(email="root@x.com", role="admin")
    d = access.scope_allowed(admin, "user:alice@x.com")
    assert not d.allowed
    assert "private" in d.reason


def test_manager_cannot_read_a_reports_private_scope(env):
    """Escalation attempt: supervising someone does not grant access to
    their private notes."""
    boss = P(email="boss@x.com", role="manager", reports=frozenset({"alice@x.com"}))
    assert not access.scope_allowed(boss, "user:alice@x.com")


def test_member_cannot_read_a_team_they_are_not_in(env):
    p = P(email="a@x.com", role="member", teams=frozenset({"team:platform"}))
    assert access.scope_allowed(p, "team:platform")
    assert not access.scope_allowed(p, "team:finance")


def test_admin_reads_team_scopes(env):
    admin = P(email="root@x.com", role="admin")
    assert access.scope_allowed(admin, "team:finance")


def test_explicit_grant_opens_one_named_scope(env):
    p = P(email="a@x.com", granted_scopes=frozenset({"team:finance"}))
    assert access.scope_allowed(p, "team:finance")
    assert not access.scope_allowed(p, "team:legal")


def test_unrecognised_scope_is_denied(env):
    assert not access.scope_allowed(P(email="a@x.com"), "wat:something")
    assert not access.scope_allowed(P(email="a@x.com"), "")


# ---- the combined decision ----------------------------------------------


def test_can_read_requires_all_three_gates(env):
    p = P(
        email="m@x.com", role="manager",
        capabilities=frozenset({"knowledge.read"}),
        teams=frozenset({"team:platform"}),
    )
    ok = access.can_read(
        p, scope="team:platform", classification="confidential",
        capability="knowledge.read",
    )
    assert ok.allowed
    # Missing capability.
    assert not access.can_read(
        p, scope="team:platform", classification="internal",
        capability="knowledge.admin",
    )
    # Wrong scope.
    assert not access.can_read(p, scope="team:legal", classification="public")
    # Above clearance.
    assert not access.can_read(p, scope="team:platform", classification="restricted")


def test_visible_scopes_never_includes_another_users_scope(env):
    p = P(
        email="a@x.com", role="admin",
        teams=frozenset({"team:platform"}),
    )
    scopes = access.visible_scopes(p)
    assert "user:a@x.com" in scopes
    assert not any(s.startswith("user:") and s != "user:a@x.com" for s in scopes)


# ---- audit ---------------------------------------------------------------


def test_every_decision_is_audited_including_denials(env):
    p = P(email="a@x.com", role="member")
    access.audit_read(
        p, resource="doc-1:0",
        decision=access.can_read(p, scope="enterprise", classification="restricted"),
        query="salaries",
    )
    ledger = access.read_ledger()
    assert len(ledger) == 1
    row = ledger[0]
    assert row["allowed"] is False
    assert row["user_email"] == "a@x.com"
    assert row["resource"] == "doc-1:0"
    assert row["classification"] == "restricted"
    assert row["query"] == "salaries"


def test_audit_sink_failure_never_breaks_the_read(env):
    def boom(_row):
        raise RuntimeError("ledger unavailable")

    p = P(email="a@x.com")
    row = access.audit_read(
        p, resource="d", decision=access.Decision(True, "ok"), sink=boom
    )
    assert row["allowed"] is True


# ---- end-to-end through retrieval ---------------------------------------


def test_retrieval_hides_documents_above_the_users_clearance(env):
    rag.ingest(
        None, title="Org chart", text="The platform team has nine engineers.",
        source_ref="w/1", classification="internal",
    )
    rag.ingest(
        None, title="Comp review", text="Platform engineer salary bands for FY26.",
        source_ref="w/2", classification="restricted",
    )
    member = P(
        email="a@x.com", role="member",
        capabilities=frozenset({rag.DEFAULT_CAPABILITY}),
    )
    admin = P(
        email="root@x.com", role="admin",
        capabilities=frozenset({rag.DEFAULT_CAPABILITY}),
    )
    seen_by_member = {
        h.chunk.title
        for h in rag.retrieve(None, "platform engineer", k=5, principal=member)
    }
    seen_by_admin = {
        h.chunk.title
        for h in rag.retrieve(None, "platform engineer", k=5, principal=admin)
    }
    assert seen_by_member == {"Org chart"}
    assert seen_by_admin == {"Org chart", "Comp review"}


def test_retrieval_denials_are_recorded(env):
    rag.ingest(
        None, title="Secret", text="restricted board material",
        source_ref="w/3", classification="restricted",
    )
    member = P(
        email="a@x.com", role="member",
        capabilities=frozenset({rag.DEFAULT_CAPABILITY}),
    )
    assert rag.retrieve(None, "board material", k=3, principal=member) == []
    denials = [r for r in access.read_ledger() if not r["allowed"]]
    assert denials and denials[0]["classification"] == "restricted"


def test_retrieval_without_capability_returns_nothing(env):
    rag.ingest(None, title="Doc", text="anything at all", source_ref="w/4")
    p = P(email="a@x.com", role="admin", capabilities=frozenset())
    assert rag.retrieve(None, "anything", k=3, principal=p) == []


def test_denied_passage_is_replaced_not_just_dropped(env):
    """k results must still be returned when a higher-ranked passage is
    filtered out — otherwise classification leaks through result counts."""
    for i in range(5):
        rag.ingest(
            None, title=f"Public {i}", text="quarterly revenue growth summary",
            source_ref=f"pub/{i}", classification="internal",
        )
    rag.ingest(
        None, title="Restricted", text="quarterly revenue growth summary",
        source_ref="res/1", classification="restricted",
    )
    member = P(
        email="a@x.com", role="member",
        capabilities=frozenset({rag.DEFAULT_CAPABILITY}),
    )
    hits = rag.retrieve(None, "quarterly revenue growth", k=5, principal=member)
    assert len(hits) == 5
    assert "Restricted" not in {h.chunk.title for h in hits}


# ---- a control-plane blip must not be cached -----------------------------
# Failing closed for the decision in hand is right. Caching that failure
# turned a one-second store blip into a thirty-second privilege outage:
# every decision in the window silently agreed with the failure, and an
# admin mid-blip became a member with no capabilities until the TTL ran
# out — long after the store was back.


def test_a_degraded_principal_is_not_cached(env):
    from agent import access

    access.invalidate()
    calls = {"n": 0}

    def flaky(stmt, params):
        calls["n"] += 1
        if calls["n"] <= 3:  # the three reads of the first resolution
            raise RuntimeError("store blip")
        if "user_roles" in stmt:
            return [{"role": "admin", "team": None, "granted_scope": None,
                     "clearance": None}]
        if "effective_employee_acl" in stmt:
            return [{"capability": "email.send"}]
        return []

    during = access.load_principal(flaky, "dana@alex.dev")
    after = access.load_principal(flaky, "dana@alex.dev")

    assert during.role == "member" and not during.capabilities, (
        "the blip itself must still fail closed"
    )
    assert after.role == "admin", "the store recovered but the failure was cached"
    assert "email.send" in after.capabilities


def test_a_genuinely_empty_grant_is_still_cached(env):
    """'You have nothing' is a real answer and caching it is correct —
    otherwise every unentitled user costs three store reads per turn."""
    from agent import access

    access.invalidate()
    calls = {"n": 0}

    def empty(stmt, params):
        calls["n"] += 1
        return []

    access.load_principal(empty, "nobody@alex.dev")
    first = calls["n"]
    access.load_principal(empty, "nobody@alex.dev")

    assert calls["n"] == first, "an empty (valid) resolution was not cached"
