"""Task origination — work the system noticed, waiting to be adopted.

The founding sentence of this system is "help employees complete tasks",
and most real tasks are not typed onto a board: they arrive as an ask in
an email. :mod:`agent.email_intel` already extracts those asks
(``asks_of_user``, with who asked and when it is due) through an
injection-proof prompt — and until this module, nothing consumed them.
This is the node that closes the loop from *signal* to *board*.

The shape is a two-step handoff, and the pause in the middle is the
control property:

1. **Harvest** turns each extracted ask into a row in
   ``media.control.task_proposals`` — visible, attributed to its source
   email, and inert. Idempotent on ``source_ref``: re-harvesting never
   duplicates, and a dismissed proposal stays dismissed.
2. **A person accepts or dismisses.** Only acceptance creates a real
   task (``source='email'``, ``source_ref`` preserved) that a run can be
   started on.

Email bodies are UNTRUSTED content. The extractor is told to ignore
instructions inside them, but the deeper defence is structural: nothing
that arrived in an email becomes agent-runnable work without an employee
reading it first. An attacker who fully controls an inbound email
controls, at most, the text of a suggestion.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

log = logging.getLogger("agent.task_origination")

PROPOSALS_TABLE = "media.control.task_proposals"
INTEL_TABLE = "media.alex.email_intel"
STATUSES = ("proposed", "accepted", "dismissed")
#: New proposals per harvest call — a bound, said out loud in the return,
#: not a silent truncation.
MAX_PER_HARVEST = 20
MAX_TITLE = 200


def _asks(raw: Any) -> list[dict]:
    """The asks_of_user cell as a list of dicts, whatever the backend
    stored: production returns an array of structs or JSON text; the dev
    warehouse stringifies the inserted Python value with repr (single
    quotes), which json.loads rejects — hence the literal_eval fallback.
    Both parsers only ever produce data, never execute anything."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError):
            import ast

            try:
                raw = ast.literal_eval(raw)
            except (SyntaxError, ValueError):
                return []
    if not isinstance(raw, list):
        return []
    return [a for a in raw if isinstance(a, dict) and str(a.get("what") or "").strip()]


def harvest(sql, owner_email: str) -> dict[str, Any]:
    """Turn this employee's un-proposed email asks into proposals.

    Reads ``email_intel`` (already extracted, injection-proof) — never
    raw email bodies. Returns counts so the caller can say what happened.
    """
    owner_email = (owner_email or "").strip().lower()
    if not owner_email:
        return {"created": 0, "skipped": 0}
    try:
        intel_rows = sql(
            f"SELECT email_id, priority, asks_of_user FROM {INTEL_TABLE} "
            "WHERE owner_email = :u",
            {"u": owner_email},
        )
    except Exception:  # noqa: BLE001
        log.debug("email_intel unavailable for harvest", exc_info=True)
        return {"created": 0, "skipped": 0}
    try:
        seen_refs = {
            str(r.get("source_ref"))
            for r in sql(
                f"SELECT source_ref FROM {PROPOSALS_TABLE} "
                "WHERE owner_email = :u",
                {"u": owner_email},
            )
            or []
        }
    except Exception:  # noqa: BLE001
        log.debug("proposal read failed; refusing to harvest blind", exc_info=True)
        # Without the dedupe set every harvest would duplicate everything
        # — including resurrecting dismissed proposals. Fail closed.
        return {"created": 0, "skipped": 0}

    created = 0
    skipped = 0
    for row in intel_rows or []:
        email_id = str(row.get("email_id") or "")
        if not email_id:
            continue
        priority = row.get("priority")
        priority = int(priority) if isinstance(priority, int) else 3
        for i, ask in enumerate(_asks(row.get("asks_of_user"))):
            ref = f"email:{email_id}#ask{i}"
            if ref in seen_refs:
                skipped += 1
                continue
            if created >= MAX_PER_HARVEST:
                skipped += 1
                continue
            what = str(ask.get("what") or "").strip()
            who = str(ask.get("who") or "").strip()
            due = ask.get("when_due")
            description = (
                f"From an email ({email_id})"
                + (f" — {who} asked" if who else "")
                + f": {what}"
            )
            sql(
                f"INSERT INTO {PROPOSALS_TABLE} "
                "(proposal_id, owner_email, title, description, priority, "
                " due_at, source, source_ref, status, task_id, created_at, "
                " decided_at) "
                "VALUES (:p, :u, :t, :d, :pr, :due, 'email', :ref, "
                "        'proposed', NULL, current_timestamp(), NULL)",
                {
                    "p": f"prop-{uuid.uuid4()}",
                    "u": owner_email,
                    "t": what[:MAX_TITLE],
                    "d": description[:2000],
                    "pr": max(1, min(priority, 5)),
                    "due": (str(due) if due else None),
                    "ref": ref,
                },
            )
            seen_refs.add(ref)
            created += 1
    return {"created": created, "skipped": skipped}


def pending(sql, owner_email: str) -> list[dict[str, Any]]:
    """This employee's open proposals, highest priority first."""
    try:
        rows = sql(
            f"SELECT proposal_id, title, description, priority, due_at, "
            f"source, source_ref, status, created_at FROM {PROPOSALS_TABLE} "
            "WHERE owner_email = :u",
            {"u": (owner_email or "").strip().lower()},
        )
    except Exception:  # noqa: BLE001
        return []
    out = [dict(r) for r in rows or [] if str(r.get("status")) == "proposed"]
    out.sort(key=lambda r: (int(r.get("priority") or 3), str(r.get("created_at"))))
    return out


def _own_proposal(sql, principal, proposal_id: str) -> dict[str, Any] | None:
    try:
        rows = sql(
            f"SELECT proposal_id, owner_email, title, description, priority, "
            f"due_at, source_ref, status FROM {PROPOSALS_TABLE} "
            "WHERE proposal_id = :p",
            {"p": proposal_id},
        )
    except Exception:  # noqa: BLE001
        return None
    if not rows:
        return None
    row = dict(rows[0])
    owner = str(row.get("owner_email") or "").lower()
    if owner != (principal.email or "").lower() and not principal.is_admin:
        # Someone else's proposal is someone else's email content.
        return None
    return row


def accept(sql, principal, proposal_id: str) -> dict[str, Any]:
    """Adopt a proposal onto the board. THE control gate: this call is
    only reachable from the board UI, by the employee the ask was
    extracted for."""
    row = _own_proposal(sql, principal, proposal_id)
    if row is None:
        raise PermissionError("no such proposal, or it is not yours")
    if row.get("status") != "proposed":
        raise ValueError(f"proposal already {row.get('status')}")
    from agent import taskboard

    task = taskboard.create_task(
        sql,
        principal,
        title=str(row.get("title") or ""),
        description=str(row.get("description") or ""),
        status="todo",
        priority=int(row.get("priority") or 3),
        due_at=row.get("due_at"),
        source="email",
        source_ref=str(row.get("source_ref") or ""),
    )
    sql(
        f"UPDATE {PROPOSALS_TABLE} SET status = 'accepted', task_id = :t, "
        "decided_at = current_timestamp() WHERE proposal_id = :p",
        {"t": task["task_id"], "p": proposal_id},
    )
    return {"proposal_id": proposal_id, "task_id": task["task_id"]}


def dismiss(sql, principal, proposal_id: str) -> dict[str, Any]:
    """Decline a proposal. Durable: the harvest will not re-raise it."""
    row = _own_proposal(sql, principal, proposal_id)
    if row is None:
        raise PermissionError("no such proposal, or it is not yours")
    if row.get("status") != "proposed":
        raise ValueError(f"proposal already {row.get('status')}")
    sql(
        f"UPDATE {PROPOSALS_TABLE} SET status = 'dismissed', "
        "decided_at = current_timestamp() WHERE proposal_id = :p",
        {"p": proposal_id},
    )
    return {"proposal_id": proposal_id, "dismissed": True}
