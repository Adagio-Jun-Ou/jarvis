"""
Access control for enterprise knowledge — one authorization decision
point for every read of a document, passage, or graph fact.

The rest of the system asks exactly one question here: *may this
principal see this thing?* Centralising it means a new surface (an API
route, an MCP tool, a pipeline) cannot accidentally invent a weaker
rule, and every decision — allow **and** deny — is auditable.

Model
-----

A :class:`Principal` is the acting human, resolved from the control
plane: their **role**, their **effective capabilities**, the **teams**
they belong to, and the **reports** they supervise.

Every document carries a **classification**:

    public < internal < confidential < restricted

and a **scope** — ``enterprise`` (organisation-wide), ``team:<name>``,
or ``user:<email>`` (one person's private material).

Two independent gates must both pass:

1. **Scope** — is this the principal's own material, their team's, a
   report's team, or organisation-wide? A user's private scope is
   readable *only by that user*: an admin role does not confer the
   right to read a colleague's private notes. Administrators get
   oversight through the audit ledger, not by reading content.
2. **Clearance** — does the principal's role reach the document's
   classification? ``member`` reaches internal, ``manager`` reaches
   confidential, ``admin`` reaches restricted. An explicit per-user
   grant can raise clearance for a named scope without changing role.

Both gates **fail closed**: an unresolvable principal, an unknown
classification, or a control-plane error yields the least privilege,
never the most.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("agent.access")

# Classification lattice, least to most sensitive.
CLASSIFICATIONS = ("public", "internal", "confidential", "restricted")
CLASSIFICATION_RANK = {name: i for i, name in enumerate(CLASSIFICATIONS)}
DEFAULT_CLASSIFICATION = "internal"

ROLE_MEMBER = "member"
ROLE_MANAGER = "manager"
ROLE_ADMIN = "admin"
ROLES = (ROLE_MEMBER, ROLE_MANAGER, ROLE_ADMIN)

# The highest classification each role may read by default.
ROLE_CLEARANCE = {
    ROLE_MEMBER: "internal",
    ROLE_MANAGER: "confidential",
    ROLE_ADMIN: "restricted",
}

ENTERPRISE_SCOPE = "enterprise"
_USER_PREFIX = "user:"
_TEAM_PREFIX = "team:"

# Principals are resolved once per turn, not once per candidate document.
_PRINCIPAL_CACHE: dict[str, tuple[float, Principal]] = {}
_PRINCIPAL_TTL_S = 30.0


@dataclass(frozen=True)
class Decision:
    """The outcome of one authorization check."""

    allowed: bool
    reason: str
    classification: str = ""
    scope: str = ""

    def __bool__(self) -> bool:
        return self.allowed


@dataclass
class Principal:
    """The acting user and everything authorization needs to know."""

    email: str
    role: str = ROLE_MEMBER
    capabilities: frozenset[str] = field(default_factory=frozenset)
    teams: frozenset[str] = field(default_factory=frozenset)
    # Emails this principal supervises (from media.control.supervision).
    reports: frozenset[str] = field(default_factory=frozenset)
    # Extra scopes explicitly granted, and any raised clearance.
    granted_scopes: frozenset[str] = field(default_factory=frozenset)
    clearance_override: str = ""

    @property
    def clearance(self) -> str:
        base = ROLE_CLEARANCE.get(self.role, "internal")
        if self.clearance_override in CLASSIFICATION_RANK:
            if CLASSIFICATION_RANK[self.clearance_override] > (
                CLASSIFICATION_RANK[base]
            ):
                return self.clearance_override
        return base

    @property
    def own_scope(self) -> str:
        return f"{_USER_PREFIX}{self.email.strip().lower()}"

    @property
    def is_admin(self) -> bool:
        return self.role == ROLE_ADMIN

    def to_dict(self) -> dict[str, Any]:
        return {
            "email": self.email,
            "role": self.role,
            "clearance": self.clearance,
            "teams": sorted(self.teams),
            "reports": sorted(self.reports),
            "granted_scopes": sorted(self.granted_scopes),
            "capabilities": sorted(self.capabilities),
        }


def anonymous() -> Principal:
    """The least-privileged principal: used whenever identity is unknown."""
    return Principal(email="", role=ROLE_MEMBER, capabilities=frozenset())


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def _rows(sql, statement: str, params: dict) -> list[dict] | None:
    """Read from the control plane.

    Note the statements below compare ``user_email`` directly rather than
    wrapping it in ``lower()``. Identity is lowercased before it reaches
    here, and a ``lower(col) = :param`` predicate is not portable across
    both warehouse backends — where it is unsupported it degrades to
    matching every row, which on these tables means inheriting everyone
    else's roles and capabilities.
    """
    try:
        return list(sql(statement, params) or [])
    except Exception:  # noqa: BLE001
        log.debug("access: control-plane read failed", exc_info=True)
        # None, not []: the caller must be able to tell "you have no
        # grants" (a real answer, cacheable) from "the store did not
        # answer" (fail closed for this decision, but do NOT cache it —
        # a 30-second cache of a one-second blip strips a legitimate
        # user's capabilities for thirty seconds after the store is back).
        return None


def load_principal(sql, user_email: str, *, use_cache: bool = True) -> Principal:
    """Resolve a principal from the control plane.

    Fails closed: any control-plane error yields a ``member`` with no
    capabilities rather than an over-privileged principal.
    """
    email = (user_email or "").strip().lower()
    if not email:
        return anonymous()
    if use_cache:
        hit = _PRINCIPAL_CACHE.get(email)
        if hit and (time.time() - hit[0]) < _PRINCIPAL_TTL_S:
            return hit[1]

    role = ROLE_MEMBER
    teams: set[str] = set()
    granted: set[str] = set()
    override = ""
    degraded = False
    role_rows = _rows(
        sql,
        "SELECT role, team, granted_scope, clearance "
        "FROM media.control.user_roles WHERE user_email = :e",
        {"e": email},
    )
    if role_rows is None:
        degraded, role_rows = True, []
    for r in role_rows:
        candidate = str(r.get("role") or "").strip().lower()
        if candidate in ROLES and ROLES.index(candidate) > ROLES.index(role):
            role = candidate
        if r.get("team"):
            teams.add(f"{_TEAM_PREFIX}{str(r['team']).strip().lower()}")
        if r.get("granted_scope"):
            granted.add(str(r["granted_scope"]).strip().lower())
        c = str(r.get("clearance") or "").strip().lower()
        if c in CLASSIFICATION_RANK:
            if not override or (
                CLASSIFICATION_RANK[c] > CLASSIFICATION_RANK[override]
            ):
                override = c

    cap_rows = _rows(
        sql,
        "SELECT capability FROM media.control.effective_employee_acl "
        "WHERE user_email = :e",
        {"e": email},
    )
    if cap_rows is None:
        degraded, cap_rows = True, []
    capabilities = {
        str(r["capability"]) for r in cap_rows if r.get("capability")
    }

    report_rows = _rows(
        sql,
        "SELECT report_email FROM media.control.supervision "
        "WHERE supervisor_email = :e",
        {"e": email},
    )
    if report_rows is None:
        degraded, report_rows = True, []
    reports = {
        str(r["report_email"]).strip().lower()
        for r in report_rows
        if r.get("report_email")
    }

    principal = Principal(
        email=email,
        role=role,
        capabilities=frozenset(capabilities),
        teams=frozenset(teams),
        reports=frozenset(reports),
        granted_scopes=frozenset(granted),
        clearance_override=override,
    )
    if use_cache and not degraded:
        # A degraded principal is the right answer for THIS decision —
        # fail closed — but caching it turns a transient store blip into
        # a 30-second privilege outage for a legitimate user, and every
        # decision in that window silently agrees with the failure.
        now = time.time()
        # Evict on write: without it the cache only ever grows — one
        # stale entry per user ever seen, for the process's lifetime.
        for k in [
            k for k, (ts, _p) in _PRINCIPAL_CACHE.items()
            if now - ts >= _PRINCIPAL_TTL_S
        ]:
            _PRINCIPAL_CACHE.pop(k, None)
        _PRINCIPAL_CACHE[email] = (now, principal)
    return principal


def invalidate(user_email: str = "") -> None:
    """Drop cached principals after a role or ACL change."""
    if user_email:
        _PRINCIPAL_CACHE.pop((user_email or "").strip().lower(), None)
    else:
        _PRINCIPAL_CACHE.clear()


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------


def normalise_classification(value: str) -> str:
    """Map a stored value onto the lattice; unknown values are treated as
    the most sensitive, so a typo cannot widen access."""
    v = (value or "").strip().lower()
    if not v:
        return DEFAULT_CLASSIFICATION
    return v if v in CLASSIFICATION_RANK else "restricted"


def scope_allowed(principal: Principal, scope: str) -> Decision:
    """Is this scope within the principal's reach?"""
    s = (scope or "").strip().lower()
    if not s:
        return Decision(False, "no scope on the resource", scope=scope)
    if s in (ENTERPRISE_SCOPE, "default"):
        return Decision(True, "organisation-wide scope", scope=s)
    if s in principal.granted_scopes:
        return Decision(True, "explicitly granted scope", scope=s)
    if s.startswith(_USER_PREFIX):
        # A private scope belongs to exactly one person. No role — not
        # manager, not admin — reads someone else's private material.
        if s == principal.own_scope:
            return Decision(True, "the principal's own scope", scope=s)
        return Decision(False, "private to another user", scope=s)
    if s.startswith(_TEAM_PREFIX):
        if s in principal.teams:
            return Decision(True, "member of the team", scope=s)
        if principal.is_admin:
            return Decision(True, "administrator", scope=s)
        return Decision(False, "not a member of this team", scope=s)
    return Decision(False, "unrecognised scope", scope=s)


def clearance_allowed(principal: Principal, classification: str) -> Decision:
    """Does the principal's clearance reach this classification?"""
    level = normalise_classification(classification)
    have = CLASSIFICATION_RANK[principal.clearance]
    need = CLASSIFICATION_RANK[level]
    if have >= need:
        return Decision(
            True, f"clearance {principal.clearance} covers {level}",
            classification=level,
        )
    return Decision(
        False,
        f"clearance {principal.clearance} is below {level}",
        classification=level,
    )


def can_read(
    principal: Principal,
    *,
    scope: str,
    classification: str = DEFAULT_CLASSIFICATION,
    capability: str = "",
) -> Decision:
    """The single read-authorization decision. All three gates must pass."""
    if capability and capability not in principal.capabilities:
        return Decision(
            False, f"missing capability {capability}",
            classification=normalise_classification(classification), scope=scope,
        )
    scope_d = scope_allowed(principal, scope)
    if not scope_d.allowed:
        return Decision(
            False, scope_d.reason,
            classification=normalise_classification(classification),
            scope=scope_d.scope,
        )
    clear_d = clearance_allowed(principal, classification)
    if not clear_d.allowed:
        return Decision(
            False, clear_d.reason,
            classification=clear_d.classification, scope=scope_d.scope,
        )
    return Decision(
        True, f"{scope_d.reason}; {clear_d.reason}",
        classification=clear_d.classification, scope=scope_d.scope,
    )


def visible_scopes(principal: Principal) -> list[str]:
    """Every scope this principal may read, for query pre-filtering.

    This is an optimisation, not the authorization boundary —
    :func:`can_read` is still applied per document.
    """
    scopes = [ENTERPRISE_SCOPE, "default"]
    if principal.email:
        scopes.append(principal.own_scope)
    scopes.extend(sorted(principal.teams))
    scopes.extend(sorted(principal.granted_scopes))
    seen: list[str] = []
    for s in scopes:
        if s and s not in seen:
            seen.append(s)
    return seen


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


def audit_read(
    principal: Principal,
    *,
    resource: str,
    decision: Decision,
    query: str = "",
    sink=None,
) -> dict[str, Any]:
    """Record one access decision. Denials are recorded too — an
    attempted read of material a user may not see is exactly the signal
    a security review needs."""
    row = {
        "ts": time.time(),
        "user_email": principal.email,
        "role": principal.role,
        "resource": resource,
        "scope": decision.scope,
        "classification": decision.classification,
        "allowed": bool(decision.allowed),
        "reason": decision.reason,
        "query": (query or "")[:200],
    }
    try:
        if sink is not None:
            sink(row)
        else:
            _default_sink(row)
    except Exception:  # noqa: BLE001
        log.debug("access audit sink failed", exc_info=True)
    return row


def _default_sink(row: dict[str, Any]) -> None:
    """Append to the in-process ledger (dev) and the app log."""
    from alex.testing import get_fakes

    try:
        fk = get_fakes()
        ledger = getattr(fk, "_access_ledger", None)
        if ledger is None:
            ledger = []
            fk._access_ledger = ledger
        ledger.append(row)
    except Exception:  # noqa: BLE001
        pass
    if not row["allowed"]:
        log.warning(
            "access DENIED user=%s resource=%s reason=%s",
            row["user_email"], row["resource"], row["reason"],
        )


def read_ledger(limit: int = 100) -> list[dict[str, Any]]:
    """The recent access decisions (dev/test introspection)."""
    from alex.testing import get_fakes

    try:
        return list(getattr(get_fakes(), "_access_ledger", []))[-limit:]
    except Exception:  # noqa: BLE001
        return []
