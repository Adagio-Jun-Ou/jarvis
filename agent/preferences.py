"""
Per-user preferences — the workspace shaped to the person using it.

Three rules keep this from turning into a settings swamp:

* **A short, opinionated list.** Every knob here changes how the
  workspace behaves for someone who uses it daily. Theme builders,
  drag-anywhere dashboards and font pickers are deliberately absent —
  most people never open settings, so the *default* is the real product.

* **Policy beats preference.** Precedence is
  ``workspace policy > user preference > session override``. An
  administrator can pin a value (autonomy level, say) and a user cannot
  quietly opt out of it; the API reports which keys are pinned so the UI
  can show them as governed rather than silently ignoring the change.

* **The server is the source of truth.** Preferences follow a person
  across devices. The client keeps a write-through cache so the first
  paint has the right density and theme, but it is a cache — genuinely
  device-local things (panel pixel widths, which tab was last open) stay
  in the browser and never come here.
"""

from __future__ import annotations

import json
import logging
from typing import Any

log = logging.getLogger("agent.preferences")

TABLE = "media.control.user_preferences"
POLICY_TABLE = "media.control.workspace_policy"

# name -> (default, validator). A value that fails validation is
# discarded rather than stored: a preference store is not a place to
# find out you have arbitrary strings in your CSS.
_DENSITIES = ("comfortable", "compact")
_THEMES = ("dark", "light")
_AUTONOMY = ("suggest", "co-pilot", "autopilot")

SCHEMA: dict[str, tuple[Any, Any]] = {
    "theme": ("dark", lambda v: v in _THEMES),
    "density": ("comfortable", lambda v: v in _DENSITIES),
    # How much the agent may do before asking. The research consensus is
    # that this should default conservative and be per-user, not global.
    "autonomy": ("co-pilot", lambda v: v in _AUTONOMY),
    "default_model": ("", lambda v: isinstance(v, str) and len(v) < 120),
    # Tabs pinned to the top of the nav, in order.
    "pinned_tabs": (
        ["chat", "board"],
        lambda v: isinstance(v, list) and all(isinstance(x, str) for x in v)
        and len(v) <= 10,
    ),
    # Tools surfaced first in the palette and the Tools tab.
    "pinned_tools": (
        [],
        lambda v: isinstance(v, list) and all(isinstance(x, str) for x in v)
        and len(v) <= 20,
    ),
    # Free text prepended to this user's system prompt. Bounded, because
    # it is prompt budget every single turn.
    "custom_instructions": (
        "", lambda v: isinstance(v, str) and len(v) <= 2000
    ),
    "show_reasoning": (False, lambda v: isinstance(v, bool)),
    "notify_on": (
        ["run_finished", "needs_approval"],
        lambda v: isinstance(v, list) and len(v) <= 8,
    ),
    "reduce_motion": (False, lambda v: isinstance(v, bool)),
}

DEFAULTS = {name: default for name, (default, _v) in SCHEMA.items()}


def _rows(sql, stmt: str, params: dict | None = None) -> list[dict]:
    try:
        return list(sql(stmt, params or {}) or [])
    except Exception:  # noqa: BLE001
        log.debug("preferences read failed", exc_info=True)
        return []


def _decode(raw: Any) -> Any:
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            return raw
    return raw


def validate(name: str, value: Any) -> bool:
    entry = SCHEMA.get(name)
    if entry is None:
        return False
    try:
        return bool(entry[1](value))
    except Exception:  # noqa: BLE001
        return False


def policy(sql) -> dict[str, Any]:
    """Workspace-level pins an administrator has set."""
    out: dict[str, Any] = {}
    for r in _rows(sql, f"SELECT name, value FROM {POLICY_TABLE}"):
        name = str(r.get("name") or "")
        if name in SCHEMA:
            value = _decode(r.get("value"))
            if validate(name, value):
                out[name] = value
    return out


def get(sql, user_email: str) -> dict[str, Any]:
    """This user's effective preferences, with provenance.

    Returns the resolved values plus ``pinned``, the keys an
    administrator has fixed — so the UI can render them as governed
    instead of accepting an edit that will not stick.
    """
    resolved = dict(DEFAULTS)
    stored: dict[str, Any] = {}
    for r in _rows(
        sql,
        f"SELECT name, value FROM {TABLE} WHERE user_email = :u",
        {"u": (user_email or "").strip().lower()},
    ):
        name = str(r.get("name") or "")
        if name not in SCHEMA:
            continue
        value = _decode(r.get("value"))
        if validate(name, value):
            stored[name] = value
    resolved.update(stored)

    pins = policy(sql)
    resolved.update(pins)  # policy wins
    return {
        "preferences": resolved,
        "pinned": sorted(pins),
        "defaults": DEFAULTS,
        "options": {
            "theme": list(_THEMES),
            "density": list(_DENSITIES),
            "autonomy": list(_AUTONOMY),
        },
    }


def set_many(sql, user_email: str, changes: dict[str, Any]) -> dict[str, Any]:
    """Store preference changes. Returns what was applied and what was not.

    A key an administrator has pinned is *reported as refused* rather
    than silently dropped — a setting that appears to save and then
    reverts is worse than one that says no.
    """
    email = (user_email or "").strip().lower()
    pins = policy(sql)
    applied: dict[str, Any] = {}
    refused: dict[str, str] = {}

    for name, value in (changes or {}).items():
        if name not in SCHEMA:
            refused[name] = "unknown preference"
            continue
        if name in pins:
            refused[name] = "pinned by workspace policy"
            continue
        if not validate(name, value):
            refused[name] = "invalid value"
            continue
        encoded = json.dumps(value)
        existing = _rows(
            sql,
            f"SELECT name FROM {TABLE} WHERE user_email = :u AND name = :n",
            {"u": email, "n": name},
        )
        try:
            if existing:
                sql(
                    f"UPDATE {TABLE} SET value = :v, updated_at = :ts "
                    "WHERE user_email = :u AND name = :n",
                    {"v": encoded, "ts": _now_iso(), "u": email, "n": name},
                )
            else:
                sql(
                    f"INSERT INTO {TABLE} (user_email, name, value, updated_at) "
                    "VALUES (:u, :n, :v, current_timestamp())",
                    {"u": email, "n": name, "v": encoded},
                )
            applied[name] = value
        except Exception:  # noqa: BLE001
            log.debug("preference write failed for %s", name, exc_info=True)
            refused[name] = "could not be stored"

    return {"applied": applied, "refused": refused}


def reset(sql, user_email: str) -> int:
    """Drop this user's overrides, returning them to the defaults."""
    try:
        sql(
            f"DELETE FROM {TABLE} WHERE user_email = :u",
            {"u": (user_email or "").strip().lower()},
        )
        return 1
    except Exception:  # noqa: BLE001
        log.debug("preference reset failed", exc_info=True)
        return 0


def custom_instruction_block(sql, user_email: str) -> str:
    """This user's custom instructions, rendered for the system prompt.

    Framed explicitly as preference rather than policy so it cannot be
    used to talk over the constitution or a mandatory practice.
    """
    prefs = get(sql, user_email)["preferences"]
    text = str(prefs.get("custom_instructions") or "").strip()
    if not text:
        return ""
    return (
        "HOW THIS USER LIKES TO WORK (their own preference — follow it "
        "unless it conflicts with the constitution or a mandatory "
        f"practice, which always win):\n{text}"
    )


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(sep=" ", timespec="seconds")


def autonomy_of(sql, user_email: str) -> str:
    """How much this user lets the agent do before asking."""
    return str(get(sql, user_email)["preferences"].get("autonomy") or "co-pilot")


__all__ = [
    "SCHEMA",
    "DEFAULTS",
    "get",
    "set_many",
    "reset",
    "policy",
    "validate",
    "autonomy_of",
    "custom_instruction_block",
]
