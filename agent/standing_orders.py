"""
Standing orders — durable instructions a person can hand to the system.

A standing order is the enterprise version of "whenever X happens, do
Y". It is written once by a human, **assigned** to people or teams,
bounded by an explicit capability ceiling, and then available three
ways:

* **Ambient** — matching orders enter the T1 directive slice of every
  turn, so the agent follows them without being reminded.
* **On demand** — a person presses *Run now* and the order executes in
  its own VM session, streaming every step. This is the part that makes
  an order an *instrument* rather than a note: the user executes real
  work against the enterprise system through it.
* **Agent-initiated** — the agent can list and run orders as tools, so
  "do my Monday review" resolves to the order the user actually wrote.

The capability ceiling is the safety property. An order declares
``allowed_capabilities``; at run time the effective set is the
**intersection** of that list with the running user's own ACL. An order
can therefore never be used to widen someone's authority — not by the
author, not by the agent, not by a later edit to the order. Narrowing
is always allowed; widening never is.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

log = logging.getLogger("agent.standing_orders")

ORDERS_TABLE = "media.control.standing_orders"
RUNS_TABLE = "media.control.standing_order_runs"

# How an order is triggered.
TRIGGER_MANUAL = "manual"       # only when a person or the agent runs it
TRIGGER_SCHEDULE = "schedule"   # cron, executed by the pipeline
TRIGGER_EVENT = "event"         # on_event:<name>
TRIGGERS = (TRIGGER_MANUAL, TRIGGER_SCHEDULE, TRIGGER_EVENT)

AUDIENCE_ALL = "all"


@dataclass
class StandingOrder:
    order_id: str
    title: str
    instruction: str
    owner_email: str
    audience: list[str] = field(default_factory=list)
    allowed_capabilities: list[str] = field(default_factory=list)
    trigger: str = TRIGGER_MANUAL
    schedule: str = ""
    active: bool = True
    # Ambient behaviour: whether matching turns get this order in-prompt.
    ambient: bool = True
    applies_to: list[str] = field(default_factory=list)
    created_by: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def assigned_to(self, user_email: str) -> bool:
        email = (user_email or "").strip().lower()
        if not email:
            return False
        for entry in self.audience:
            e = str(entry).strip().lower()
            if e in (AUDIENCE_ALL, email, f"user:{email}"):
                return True
        return (self.owner_email or "").strip().lower() == email

    def to_dict(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "title": self.title,
            "instruction": self.instruction,
            "owner_email": self.owner_email,
            "audience": list(self.audience),
            "allowed_capabilities": list(self.allowed_capabilities),
            "trigger": self.trigger,
            "schedule": self.schedule,
            "active": self.active,
            "ambient": self.ambient,
            "applies_to": list(self.applies_to),
            "created_by": self.created_by,
        }


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


def _is_dev() -> bool:
    from alex.config import get_settings

    return get_settings().mode == "dev"


def _rows(sql, stmt: str, params: dict | None = None) -> list[dict]:
    try:
        return list(sql(stmt, params or {}) or [])
    except Exception:  # noqa: BLE001
        log.debug("standing order query failed", exc_info=True)
        return []


def _as_list(value: Any) -> list[str]:
    """Coerce an ``ARRAY<STRING>`` column to a list of strings.

    Databricks returns a real list; the in-memory dev warehouse
    round-trips the value through ``str()`` and hands back
    ``"['a@b.com']"``. Parsing both here keeps dev behaviour faithful to
    production — without it an order's audience silently becomes one
    nonsense entry and matches nobody.
    """
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value if str(v)]
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        import ast
        import json

        for parse in (json.loads, ast.literal_eval):
            try:
                parsed = parse(text)
            except (ValueError, SyntaxError):
                continue
            if isinstance(parsed, (list, tuple)):
                return [str(v) for v in parsed if str(v)]
    return [text]


def _as_bool(value: Any, default: bool = True) -> bool:
    """Coerce a BOOLEAN column, tolerating the dev warehouse's strings."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("true", "1", "yes"):
        return True
    if text in ("false", "0", "no", ""):
        return False
    return default


def _ts_to_epoch(value: Any) -> float:
    """Best-effort epoch seconds from a TIMESTAMP column; -1.0 when
    unparseable (callers treat that as absent)."""
    if value is None:
        return -1.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, datetime):
        stamp = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return stamp.timestamp()
    text = str(value).strip()
    if not text:
        return -1.0
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return -1.0
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _from_row(r: dict) -> StandingOrder:
    # created_at is the due-ness anchor for an order with no runs yet
    # (run_due anchors on last run OR created_at). Not reading it meant
    # every materialized order carried "now" as its anchor, so a
    # scheduled order that had never been clicked was *always*
    # strictly-future-due and fired never.
    created = _ts_to_epoch(r.get("created_at"))
    return StandingOrder(
        order_id=r.get("order_id", ""),
        title=r.get("title", ""),
        instruction=r.get("instruction", ""),
        owner_email=r.get("owner_email", ""),
        audience=_as_list(r.get("audience")),
        allowed_capabilities=_as_list(r.get("allowed_capabilities")),
        trigger=r.get("trigger") or TRIGGER_MANUAL,
        schedule=r.get("schedule") or "",
        active=_as_bool(r.get("active"), True),
        ambient=_as_bool(r.get("ambient"), True),
        applies_to=_as_list(r.get("applies_to")),
        created_by=r.get("created_by", "") or "",
        created_at=created if created > 0 else time.time(),
    )


def all_orders(sql) -> list[StandingOrder]:
    return [_from_row(r) for r in _rows(sql, f"SELECT * FROM {ORDERS_TABLE}")]


def get(sql, order_id: str) -> StandingOrder | None:
    for o in all_orders(sql):
        if o.order_id == order_id:
            return o
    return None


def for_user(sql, user_email: str, *, active_only: bool = True) -> list[StandingOrder]:
    """Orders assigned to this person."""
    out = [
        o for o in all_orders(sql)
        if o.assigned_to(user_email) and (o.active or not active_only)
    ]
    out.sort(key=lambda o: o.title.lower())
    return out


def create(
    sql,
    principal,
    *,
    title: str,
    instruction: str,
    audience: list[str] | None = None,
    allowed_capabilities: list[str] | None = None,
    trigger: str = TRIGGER_MANUAL,
    schedule: str = "",
    applies_to: list[str] | None = None,
    ambient: bool = True,
) -> StandingOrder:
    """Write a standing order.

    The capability list is clamped to what the **author** may already do:
    you cannot author an order that grants more than you hold.
    """
    title = (title or "").strip()
    instruction = (instruction or "").strip()
    if not title or not instruction:
        raise ValueError("a standing order needs a title and an instruction")
    if trigger not in TRIGGERS:
        trigger = TRIGGER_MANUAL

    requested = {c.strip() for c in (allowed_capabilities or []) if c.strip()}
    granted = sorted(requested & set(principal.capabilities))
    refused = sorted(requested - set(principal.capabilities))
    if refused:
        log.warning(
            "standing order %r: dropped capabilities the author lacks: %s",
            title, refused,
        )

    order = StandingOrder(
        order_id=f"so-{uuid.uuid4().hex[:12]}",
        title=title,
        instruction=instruction,
        owner_email=principal.email,
        audience=[a.strip() for a in (audience or [principal.email]) if a.strip()],
        allowed_capabilities=granted,
        trigger=trigger,
        schedule=schedule.strip(),
        applies_to=[s.strip() for s in (applies_to or []) if s.strip()],
        ambient=bool(ambient),
        created_by=principal.email,
    )
    sql(
        f"INSERT INTO {ORDERS_TABLE} "
        "(order_id, owner_email, title, instruction, schedule, audience, "
        " allowed_capabilities, active, created_by, trigger, ambient, applies_to) "
        "VALUES (:id, :o, :t, :i, :s, :a, :c, true, :cb, :tr, :am, :ap)",
        {
            "id": order.order_id, "o": order.owner_email, "t": order.title,
            "i": order.instruction, "s": order.schedule,
            "a": order.audience, "c": order.allowed_capabilities,
            "cb": order.created_by, "tr": order.trigger,
            "am": order.ambient, "ap": order.applies_to,
        },
    )
    return order


def update(
    sql, principal, order_id: str, **changes: Any
) -> StandingOrder | None:
    """Edit an order. Only the owner or an administrator may."""
    order = get(sql, order_id)
    if order is None:
        return None
    if order.owner_email.lower() != principal.email.lower() and not principal.is_admin:
        raise PermissionError("only the owner or an administrator may edit this order")

    if "title" in changes and str(changes["title"]).strip():
        order.title = str(changes["title"]).strip()
    if "instruction" in changes and str(changes["instruction"]).strip():
        order.instruction = str(changes["instruction"]).strip()
    if "audience" in changes:
        order.audience = [a for a in (changes["audience"] or []) if a]
    if "active" in changes:
        order.active = bool(changes["active"])
    if "ambient" in changes:
        order.ambient = bool(changes["ambient"])
    if "schedule" in changes:
        order.schedule = str(changes["schedule"] or "").strip()
    if "trigger" in changes and changes["trigger"] in TRIGGERS:
        order.trigger = changes["trigger"]
    if "applies_to" in changes:
        order.applies_to = [s for s in (changes["applies_to"] or []) if s]
    if "allowed_capabilities" in changes:
        requested = {c for c in (changes["allowed_capabilities"] or []) if c}
        # Editing can narrow but never widen beyond the editor's own ACL.
        order.allowed_capabilities = sorted(requested & set(principal.capabilities))
    order.updated_at = time.time()

    sql(
        f"UPDATE {ORDERS_TABLE} SET title = :t, instruction = :i, "
        "schedule = :s, audience = :a, allowed_capabilities = :c, "
        "active = :act, trigger = :tr, ambient = :am, applies_to = :ap "
        "WHERE order_id = :id",
        {
            "t": order.title, "i": order.instruction, "s": order.schedule,
            "a": order.audience, "c": order.allowed_capabilities,
            "act": order.active, "tr": order.trigger, "am": order.ambient,
            "ap": order.applies_to, "id": order.order_id,
        },
    )
    return order


def assign(sql, principal, order_id: str, members: list[str]) -> StandingOrder | None:
    """Add people (or ``all``) to an order's audience."""
    order = get(sql, order_id)
    if order is None:
        return None
    merged = list(order.audience)
    for m in members or []:
        m = str(m).strip()
        if m and m not in merged:
            merged.append(m)
    return update(sql, principal, order_id, audience=merged)


# ---------------------------------------------------------------------------
# The capability ceiling
# ---------------------------------------------------------------------------


def effective_capabilities(order: StandingOrder, principal) -> list[str]:
    """What this order may actually do when *this* person runs it.

    The intersection of the order's declared ceiling with the runner's
    own ACL. An order never widens authority: if the order permits
    ``email.send`` and the runner does not hold it, it is not granted.
    """
    if not order.allowed_capabilities:
        # An order with no declared ceiling inherits the runner's own
        # authority — it is a convenience, not an escalation.
        return sorted(principal.capabilities)
    return sorted(set(order.allowed_capabilities) & set(principal.capabilities))


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


@dataclass
class OrderRun:
    run_id: str
    order_id: str
    session_id: str
    started_by: str
    status: str = "running"
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    summary: str = ""
    capabilities: list[str] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    _subscribers: list = field(default_factory=list, repr=False)
    _lock: Any = field(default_factory=threading.Lock, repr=False)

    def emit(self, event: dict[str, Any]) -> None:
        with self._lock:
            self.events.append(event)
            # Same cap as the task board's runs: the in-memory record is
            # a live view, not an archive, and an unattended order is
            # exactly the run nobody is watching grow.
            if len(self.events) > 2000:
                self.events = self.events[-1000:]
            watchers = list(self._subscribers)
        for fn in watchers:
            try:
                fn(event)
            except Exception:  # noqa: BLE001
                log.debug("order run subscriber failed", exc_info=True)

    def subscribe(self, fn) -> Any:
        with self._lock:
            self._subscribers.append(fn)
            backlog = list(self.events)
        for event in backlog:
            try:
                fn(event)
            except Exception:  # noqa: BLE001
                log.debug("order run replay failed", exc_info=True)

        def unsubscribe() -> None:
            with self._lock:
                if fn in self._subscribers:
                    self._subscribers.remove(fn)

        return unsubscribe

    @property
    def running(self) -> bool:
        return self.status == "running"

    def to_dict(self, *, with_events: bool = False) -> dict[str, Any]:
        out = {
            "run_id": self.run_id,
            "order_id": self.order_id,
            "session_id": self.session_id,
            "started_by": self.started_by,
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "summary": self.summary,
            "capabilities": list(self.capabilities),
            "steps": len(self.events),
        }
        if with_events:
            out["events"] = list(self.events)
        return out


class RunAlreadyLive(RuntimeError):
    """An execution of this order is already in flight.

    Raised rather than silently returning: an unattended order runs
    with real capabilities, and two of them acting at once is the
    failure this prevents.
    """


_RUNS: dict[str, OrderRun] = {}
_RUNS_LOCK = threading.RLock()
#: Finished runs leave the registry after this long. Without it the
#: dict grew for the process lifetime — every run holding its complete
#: events list — which is a slow leak on a long-lived replica.
_RUN_TTL_S = 3600.0


def get_run(run_id: str) -> OrderRun | None:
    with _RUNS_LOCK:
        _prune_registry()
        return _RUNS.get(run_id)


def _prune_registry() -> None:
    """Drop finished runs past the TTL. Called under _RUNS_LOCK."""
    now = time.time()
    stale = [
        rid
        for rid, r in _RUNS.items()
        if not r.running and r.ended_at is not None
        and now - r.ended_at > _RUN_TTL_S
    ]
    for rid in stale:
        _RUNS.pop(rid, None)


def latest_run_for(order_id: str) -> OrderRun | None:
    with _RUNS_LOCK:
        _prune_registry()
        runs = [r for r in _RUNS.values() if r.order_id == order_id]
    return max(runs, key=lambda r: r.started_at) if runs else None


def _order_run_is_live(sql, order_id: str) -> bool:
    """Whether another process is already executing this order.

    Shares taskboard's stale-after window and epoch handling so the
    two boards cannot disagree about what 'stale' means. An
    unparseable heartbeat reads as live: declaring unknown work dead
    is what produces duplicate unattended executions.
    """
    from agent.taskboard import STALE_AFTER_SECONDS, _as_epoch

    try:
        rows = sql(
            f"SELECT run_id, last_heartbeat, started_at FROM {RUNS_TABLE} "
            "WHERE order_id = :o AND status = 'running'",
            {"o": order_id},
        )
    except Exception:  # noqa: BLE001
        # Fail toward NO-DUPLICATE (see taskboard._run_is_live): an
        # unreadable heartbeat is unknown liveness, and an unattended
        # order acting twice is the worst version of that failure.
        log.warning(
            "could not check for a live run on order %s — treating as "
            "live rather than risk a duplicate execution", order_id,
            exc_info=True,
        )
        return True
    now = time.time()
    for row in rows or []:
        beat = _as_epoch(row.get("last_heartbeat"))
        if beat < 0:
            return True
        if beat == 0.0:
            beat = _as_epoch(row.get("started_at"))
        if beat > 0 and now - beat < STALE_AFTER_SECONDS:
            return True
    return False


def reset_runs() -> None:
    with _RUNS_LOCK:
        _RUNS.clear()


def build_prompt(order: StandingOrder, *, note: str = "") -> str:
    """Turn an order into the instruction the VM actually runs."""
    parts = [
        "You are executing a standing order the user wrote. Carry it out "
        "now with your tools, then report what you did.",
        f"ORDER {order.order_id}: {order.title}",
        f"INSTRUCTION: {order.instruction}",
    ]
    if note:
        parts.append(f"NOTE FROM THE USER FOR THIS RUN: {note}")
    parts.append(
        "If the order cannot be completed — missing information, an "
        "action that needs a human decision, or a capability you do not "
        "hold — stop and say exactly what is blocking it. Do not "
        "substitute a different action."
    )
    return "\n\n".join(parts)


def run(
    sql,
    principal,
    order_id: str,
    *,
    note: str = "",
    vm_factory=None,
    background: bool = True,
) -> OrderRun:
    """Execute a standing order on behalf of ``principal``.

    Raises :class:`PermissionError` when the order is not assigned to
    this person — an order is an instrument of the people it was given
    to, not a public endpoint — and :class:`RunAlreadyLive` when an
    execution is already in flight. An order runs *unattended* with
    real capabilities: two simultaneous executions mean emails sent
    and files written twice, and no human is watching either one.
    """
    order = get(sql, order_id)
    if order is None:
        raise LookupError("no such standing order")
    if not order.assigned_to(principal.email) and not principal.is_admin:
        raise PermissionError("this standing order is not assigned to you")
    if not order.active:
        raise PermissionError("this standing order is not active")

    # Single-flight is a RESERVATION under one lock hold: check + the
    # DB liveness read + registration are atomic here. The old shape
    # (check under the lock, register in a second locked block) let two
    # "Run now" calls interleave between the blocks and both execute —
    # two unattended runs with real capabilities.
    with _RUNS_LOCK:
        live = next(
            (r for r in _RUNS.values() if r.order_id == order_id and r.running),
            None,
        )
        if live is not None:
            raise RunAlreadyLive(
                f"standing order {order_id} is already running ({live.run_id})"
            )
        if _order_run_is_live(sql, order_id):
            raise RunAlreadyLive(
                f"another process is already running standing order {order_id}"
            )
        caps = effective_capabilities(order, principal)
        run_obj = OrderRun(
            run_id=f"orun-{uuid.uuid4().hex[:12]}",
            order_id=order_id,
            session_id=f"order-{order_id}-{uuid.uuid4().hex[:6]}",
            started_by=principal.email,
            capabilities=caps,
        )
        _RUNS[run_obj.run_id] = run_obj

    sql(
        f"INSERT INTO {RUNS_TABLE} "
        "(run_id, order_id, session_id, started_by, status, started_at, "
        " last_heartbeat, capabilities, summary) "
        "VALUES (:r, :o, :s, :u, 'running', current_timestamp(), "
        "        current_timestamp(), :c, '')",
        {
            "r": run_obj.run_id, "o": order_id, "s": run_obj.session_id,
            "u": principal.email, "c": caps,
        },
    )

    prompt = build_prompt(order, note=note)

    def _work() -> None:
        _execute(sql, run_obj, principal, prompt, caps, vm_factory=vm_factory)

    if background:
        threading.Thread(
            target=_work, name=f"orderrun-{run_obj.run_id[:8]}", daemon=True
        ).start()
    else:
        _work()
    return run_obj


def recover_interrupted_runs(sql, *, stale_after: float | None = None) -> list[dict]:
    """Close out order runs whose process is gone.

    The same failure as a task run abandoned mid-flight, and worse in one
    respect: an order runs *unattended*, with real capabilities against
    real systems, so there is no person watching a stream to notice it
    stopped. Left alone the row says 'running' for ever, and the
    oversight console reports work in flight that nothing is doing.

    Shares :mod:`agent.taskboard`'s timing and epoch handling rather than
    restating them — two definitions of "stale" that could drift apart is
    how one of these paths quietly stops recovering.
    """
    from agent.taskboard import STALE_AFTER_SECONDS, _as_epoch

    window = STALE_AFTER_SECONDS if stale_after is None else stale_after
    try:
        rows = sql(
            f"SELECT run_id, order_id, started_at, last_heartbeat "
            f"FROM {RUNS_TABLE} WHERE status = 'running'",
            {},
        )
    except Exception:  # noqa: BLE001
        log.debug("could not scan for interrupted order runs", exc_info=True)
        return []

    now = time.time()
    recovered: list[dict] = []
    for row in rows or []:
        run_id = str(row.get("run_id") or "")
        if not run_id:
            continue
        with _RUNS_LOCK:
            local = _RUNS.get(run_id)
        if local is not None and local.running:
            continue
        beat = _as_epoch(row.get("last_heartbeat")) or _as_epoch(row.get("started_at"))
        if beat and now - beat < window:
            continue
        try:
            sql(
                f"UPDATE {RUNS_TABLE} SET status = 'interrupted', "
                "ended_at = current_timestamp(), last_heartbeat = NULL, "
                "summary = :m WHERE run_id = :r",
                {
                    "m": "The process running this standing order went away. "
                         "It did not finish, and nothing was concluded.",
                    "r": run_id,
                },
            )
        except Exception:  # noqa: BLE001
            log.debug("could not recover order run %s", run_id, exc_info=True)
            continue
        log.warning(
            "recovered interrupted order run %s (order %s)",
            run_id, row.get("order_id"),
        )
        recovered.append(
            {"run_id": run_id, "order_id": str(row.get("order_id") or "")}
        )
    return recovered


def _owner_principal(sql, email: str):
    """The scheduler fires an order AS its owner, with the owner's REAL
    ACL — run() and the capability ceiling then behave exactly as if
    the owner clicked. Never a synthetic wider principal."""
    from agent import access

    return access.load_principal(sql, email)


#: At most this many scheduled firings per sweep — a burst of orders
#: all due at 07:30 starts staggered across sweeps instead of stacking
#: unattended runs on the app container at once.
_MAX_FIRINGS_PER_SWEEP = int(os.environ.get("ALEX_SCHEDULE_MAX_PER_SWEEP", "3"))


def run_due(sql, *, now: float | None = None, vm_factory=None) -> list[str]:
    """Fire every active schedule-triggered order that is due.

    The missing half of ``TRIGGER_SCHEDULE`` — the trigger existed, a
    comment claimed a pipeline executed it, and nothing did. Called
    from the app's dispatcher sweep.

    Due-ness is computed from the LAST RUN (any status; the anchor is
    the order's creation when it has never run): the next firing is
    ``schedule.next_due(after=last run)``. That gives collapse-on-
    catch-up semantics for free — an app that was down over three
    missed windows fires the order ONCE, which is what a person wants
    from a brief or a digest, not three stale copies.

    An unparseable schedule never fires and is logged; single-flight
    and permission checks are run()'s, exactly as for a human click.
    Returns the order ids fired.
    """
    from datetime import datetime, timezone

    from agent import schedule as sched
    from agent.taskboard import _as_epoch

    now_s = time.time() if now is None else float(now)
    now_dt = datetime.fromtimestamp(now_s, tz=timezone.utc)
    fired: list[str] = []
    try:
        last_by_order: dict[str, float] = {}
        for r in _rows(
            sql, f"SELECT order_id, started_at FROM {RUNS_TABLE}", {}
        ):
            oid = str(r.get("order_id") or "")
            at = _as_epoch(r.get("started_at"))
            if oid and at and at > last_by_order.get(oid, 0.0):
                last_by_order[oid] = at
    except Exception:  # noqa: BLE001
        log.debug("scheduler could not read past runs", exc_info=True)
        return fired

    for order in all_orders(sql):
        if len(fired) >= _MAX_FIRINGS_PER_SWEEP:
            break
        if not order.active or order.trigger != TRIGGER_SCHEDULE:
            continue
        problem = sched.parse_error(order.schedule)
        if problem is not None:
            log.warning(
                "standing order %s has an unusable schedule: %s",
                order.order_id, problem,
            )
            continue
        anchor = last_by_order.get(order.order_id) or order.created_at
        due = sched.next_due(
            order.schedule,
            datetime.fromtimestamp(float(anchor or 0.0), tz=timezone.utc),
        )
        if due is None or due > now_dt:
            continue
        try:
            run(
                sql, _owner_principal(sql, order.owner_email),
                order.order_id,
                note=f"(scheduled firing: {order.schedule})",
                vm_factory=vm_factory,
            )
            fired.append(order.order_id)
        except (RunAlreadyLive, PermissionError, LookupError):
            continue
        except Exception:  # noqa: BLE001
            log.debug("scheduled firing of %s failed", order.order_id,
                      exc_info=True)
            continue
    return fired


def _execute(sql, run_obj: OrderRun, principal, prompt: str, caps, *, vm_factory=None):
    run_obj.emit(
        {
            "type": "run_started",
            "order_id": run_obj.order_id,
            "capabilities": list(caps),
            "ts": time.time(),
        }
    )
    try:
        if vm_factory is None:
            from agent.vm import make_vm as vm_factory  # type: ignore
        try:
            # The order's declared ceiling, already intersected with the
            # runner's own ACL. Computing it and then building a VM with
            # the user's FULL ACL — the previous behaviour — meant the
            # documented capability ceiling was recorded, displayed, and
            # never enforced: the exact wired-but-empty failure again.
            vm = vm_factory(
                session_id=run_obj.session_id,
                user_email=principal.email,
                capability_ceiling=frozenset(caps),
            )
        except TypeError:
            # Test doubles take (session_id, user_email) only; they are
            # not exercising the ceiling.
            vm = vm_factory(
                session_id=run_obj.session_id, user_email=principal.email
            )
    except Exception as exc:  # noqa: BLE001
        log.exception("standing order run could not start a VM")
        _finish(sql, run_obj, "failed", f"Could not start a VM: {exc}")
        return
    from agent.taskboard import HEARTBEAT_SECONDS

    stop_beating = threading.Event()

    def beat() -> None:
        while not stop_beating.wait(HEARTBEAT_SECONDS):
            try:
                sql(
                    f"UPDATE {RUNS_TABLE} SET last_heartbeat = current_timestamp() "
                    "WHERE run_id = :r AND status = 'running'",
                    {"r": run_obj.run_id},
                )
            except Exception:  # noqa: BLE001
                log.debug("order heartbeat failed", exc_info=True)

    pulse = threading.Thread(
        target=beat, name=f"beat-{run_obj.run_id[:8]}", daemon=True
    )
    pulse.start()

    try:
        vm.subscribe_sse(
            lambda ev: run_obj.emit(ev if isinstance(ev, dict) else _to_dict(ev))
        )
        last, _hist = vm.run_turn(text=prompt)
        _finish(sql, run_obj, "done", (last or "").strip() or "Finished with no output.")
    except Exception as exc:  # noqa: BLE001
        log.exception("standing order run failed")
        _finish(sql, run_obj, "failed", f"The run failed: {exc}")
    finally:
        stop_beating.set()
        pulse.join(timeout=2)
        try:
            vm.close()
        except Exception:  # noqa: BLE001
            log.debug("vm close failed after order run", exc_info=True)


def _to_dict(ev: Any) -> dict[str, Any]:
    """Serialise one VM event for the run record."""
    from alex.events import event_to_dict

    return event_to_dict(ev)


def _finish(sql, run_obj: OrderRun, status: str, summary: str) -> None:
    # Idempotent, same as the task board: recovery or a second caller
    # must not re-close a run and post a second verdict.
    with _RUNS_LOCK:
        if run_obj.ended_at is not None:
            return
        run_obj.status = status
        run_obj.ended_at = time.time()
        run_obj.summary = summary
    run_obj.emit(
        {"type": "run_finished", "status": status, "summary": summary,
         "ts": run_obj.ended_at}
    )
    try:
        sql(
            f"UPDATE {RUNS_TABLE} SET status = :s, summary = :m, "
            "ended_at = current_timestamp(), last_heartbeat = NULL "
            "WHERE run_id = :r",
            {"s": status, "m": summary[:4000], "r": run_obj.run_id},
        )
    except Exception:  # noqa: BLE001
        log.debug("order run bookkeeping failed", exc_info=True)
    # An order runs with real capabilities against real systems, so its
    # record gets the same independent review a task run gets.
    try:
        from agent import supervisor

        try:
            from agent.tools.impls import _llm_invoke

            llm = _llm_invoke()
        except Exception:  # noqa: BLE001
            llm = None
        supervisor.review(
            list(run_obj.events),
            goal=f"standing order {run_obj.order_id}",
            run_id=run_obj.run_id,
            user_email=run_obj.started_by,
            llm_invoke=llm,
            sql=sql,
        )
    except Exception:  # noqa: BLE001
        log.debug("supervision review failed", exc_info=True)


# ---------------------------------------------------------------------------
# The ambient prompt slice
# ---------------------------------------------------------------------------


def matching(
    sql, user_email: str, query: str = "", *, limit: int = 5
) -> list[StandingOrder]:
    """Ambient orders that bear on this turn.

    An order with no ``applies_to`` is unconditional and always shown; a
    scoped one is included when its subjects overlap the message.
    """
    import re as _re

    tokens = {
        t for t in _re.findall(r"[a-z0-9]+", (query or "").lower()) if len(t) > 2
    }
    out: list[StandingOrder] = []
    for order in for_user(sql, user_email):
        if not order.ambient:
            continue
        if not order.applies_to:
            out.append(order)
            continue
        subject = {
            t for s in order.applies_to
            for t in _re.findall(r"[a-z0-9]+", s.lower()) if len(t) > 2
        }
        if subject and tokens & subject:
            out.append(order)
    return out[:limit]


def directive_block(orders: list[StandingOrder]) -> str:
    """Render orders as directives for the T1 slice."""
    if not orders:
        return ""
    lines = [
        "STANDING ORDERS (written by the user for you — when one applies "
        "to this request, follow it and say which order you applied):"
    ]
    for o in orders:
        runnable = " [runnable on demand]" if o.trigger == TRIGGER_MANUAL else ""
        lines.append(f"- [{o.order_id}] {o.title}{runnable}\n  {o.instruction[:400]}")
    return "\n".join(lines)
