"""Schedule grammar for standing orders — small, human, deterministic.

``standing_orders.trigger == "schedule"`` existed with a comment
claiming "executed by the pipeline" — and nothing anywhere executed
it: the canonical wired-but-dead. This module is the missing half: a
tiny grammar a person can type into the order form, and a pure
``next_due`` the scheduler sweep evaluates.

Grammar (case-insensitive; times are 24h **UTC**):

* ``hourly``  /  ``hourly @:MM``
* ``daily HH:MM``
* ``weekly DAY HH:MM``          (mon…sun)
* ``every N minutes`` / ``every N hours``   (floor: 15 minutes)
* ``M H * * D`` — the numeric cron subset (minute, hour, any day of
  month/month, day-of-week ``*`` or 0-6 list, 0=Sunday)

Deliberately NOT full cron: every accepted form is one a reviewer can
read aloud, and an unparseable schedule is simply never due (the sweep
logs it; it cannot fire at a surprising time instead). The 15-minute
floor is a cost control — each firing is a governed agent run on the
app container and the LLM meter.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

MIN_INTERVAL_MINUTES = 15

_DOW = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}

_HOURLY_RE = re.compile(r"^hourly(?:\s*@?:(\d{1,2}))?$")
_DAILY_RE = re.compile(r"^daily\s+(\d{1,2}):(\d{2})$")
_WEEKLY_RE = re.compile(r"^weekly\s+([a-z]{3})[a-z]*\s+(\d{1,2}):(\d{2})$")
_EVERY_RE = re.compile(r"^every\s+(\d+)\s*(minutes?|mins?|hours?|hrs?)$")
_CRON_RE = re.compile(
    r"^(\d{1,2})\s+(\d{1,2})\s+\*\s+\*\s+(\*|[0-6](?:,[0-6])*)$"
)


def _bad_clock(hour: int, minute: int) -> str | None:
    """Range validation for HH:MM components. Clamping in next_due is
    defense-in-depth only: silently firing 'daily 25:00' at 23:00 (or
    'hourly @:75' at :59) contradicts the contract that an invalid
    schedule never fires at a surprising time — it must not fire."""
    if not 0 <= hour <= 23:
        return f"hour {hour} out of range (0-23)"
    if not 0 <= minute <= 59:
        return f"minute {minute} out of range (0-59)"
    return None


def parse_error(schedule: str) -> str | None:
    """Why this schedule text cannot fire, or None when it can."""
    s = (schedule or "").strip().lower()
    if not s:
        return "empty schedule"
    m = _EVERY_RE.match(s)
    if m:
        n = int(m.group(1))
        minutes = n if m.group(2).startswith(("min",)) else n * 60
        if minutes < MIN_INTERVAL_MINUTES:
            return (
                f"every {minutes} minutes is under the "
                f"{MIN_INTERVAL_MINUTES}-minute floor — each firing is a "
                "governed agent run and a model bill"
            )
        return None
    m = _HOURLY_RE.match(s)
    if m:
        return _bad_clock(0, int(m.group(1) or 0))
    m = _DAILY_RE.match(s)
    if m:
        return _bad_clock(int(m.group(1)), int(m.group(2)))
    m = _WEEKLY_RE.match(s)
    if m:
        if m.group(1) not in _DOW:
            return f"unknown weekday {m.group(1)!r} (mon…sun)"
        return _bad_clock(int(m.group(2)), int(m.group(3)))
    m = _CRON_RE.match(s)
    if m:
        return _bad_clock(int(m.group(2)), int(m.group(1)))
    return (
        f"unrecognized schedule {schedule!r} — use 'hourly', 'daily HH:MM', "
        "'weekly DAY HH:MM', 'every N minutes/hours', or 'M H * * D'"
    )


def _utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def next_due(schedule: str, after: datetime) -> datetime | None:
    """The first UTC instant strictly after ``after`` when this
    schedule fires, or None when the text is invalid (never due)."""
    s = (schedule or "").strip().lower()
    if parse_error(s) is not None:
        return None
    after = _utc(after)

    m = _EVERY_RE.match(s)
    if m:
        n = int(m.group(1))
        minutes = n if m.group(2).startswith("min") else n * 60
        return after + timedelta(minutes=minutes)

    m = _HOURLY_RE.match(s)
    if m:
        minute = min(int(m.group(1) or 0), 59)
        candidate = after.replace(minute=minute, second=0, microsecond=0)
        while candidate <= after:
            candidate += timedelta(hours=1)
        return candidate

    m = _DAILY_RE.match(s)
    if m:
        hour, minute = min(int(m.group(1)), 23), min(int(m.group(2)), 59)
        candidate = after.replace(hour=hour, minute=minute, second=0,
                                  microsecond=0)
        while candidate <= after:
            candidate += timedelta(days=1)
        return candidate

    m = _WEEKLY_RE.match(s)
    if m:
        dow = _DOW[m.group(1)]
        hour, minute = min(int(m.group(2)), 23), min(int(m.group(3)), 59)
        candidate = after.replace(hour=hour, minute=minute, second=0,
                                  microsecond=0)
        while candidate <= after or candidate.weekday() != dow:
            candidate += timedelta(days=1)
        return candidate

    m = _CRON_RE.match(s)
    if m:
        minute, hour = min(int(m.group(1)), 59), min(int(m.group(2)), 23)
        days = (
            set(range(7)) if m.group(3) == "*"
            # cron counts 0=Sunday; Python counts 0=Monday.
            else {(int(d) - 1) % 7 for d in m.group(3).split(",")}
        )
        candidate = after.replace(hour=hour, minute=minute, second=0,
                                  microsecond=0)
        while candidate <= after or candidate.weekday() not in days:
            candidate += timedelta(days=1)
        return candidate
    return None
