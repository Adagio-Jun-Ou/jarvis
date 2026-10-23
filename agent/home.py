"""Home automation — Home Assistant first, shell commands as the escape
hatch. Ported from Jarvis's home tools; same honesty rule: no
placeholder success, ever. If nothing is configured the tool says so
and how to configure it.

Configuration is OPERATOR configuration (env / a local file), never a
tool argument — a model that could name the Home Assistant host could
point the token at its own server:

* ``HOME_ASSISTANT_URL`` + ``HOME_ASSISTANT_TOKEN`` (long-lived token)
* ``ALEX_HOME_COMMANDS`` — path to a JSON map
  ``{"living_light": {"on": ["cmd", "arg"], "off": [...]}}`` of argv
  lists (no shell); for gear without an API.

Capabilities: ``home.read`` (status, power 0) and ``home.act`` (control,
power 2, policy-ASK by default — an unattended run parks on approval
before it touches the house).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import urllib.error
import urllib.request
from typing import Any

log = logging.getLogger("agent.home")

TIMEOUT_S = 10
_ACTIONS_TO_SERVICE = {
    "on": "turn_on", "off": "turn_off", "toggle": "toggle",
    "open": "open_cover", "close": "close_cover", "lock": "lock", "unlock": "unlock",
}


def _ha() -> tuple[str, str]:
    return (
        os.environ.get("HOME_ASSISTANT_URL", "").strip().rstrip("/"),
        os.environ.get("HOME_ASSISTANT_TOKEN", "").strip(),
    )


def _commands() -> dict[str, dict[str, list[str]]]:
    path = os.environ.get("ALEX_HOME_COMMANDS", "").strip()
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        log.warning("ALEX_HOME_COMMANDS unreadable: %s", path)
        return {}


def configured() -> dict:
    url, token = _ha()
    return {"home_assistant": bool(url and token), "commands": sorted(_commands())}


def _ha_request(method: str, path: str, body: dict | None = None) -> Any:
    url, token = _ha()
    req = urllib.request.Request(
        f"{url}/api/{path.lstrip('/')}",
        data=json.dumps(body).encode("utf-8") if body is not None else None,
        method=method,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
            raw = r.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Home Assistant HTTP {e.code}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Home Assistant unreachable: {e.reason}") from e


_NOT_CONFIGURED = (
    "home automation is not configured. Set HOME_ASSISTANT_URL and "
    "HOME_ASSISTANT_TOKEN (a long-lived access token), or point "
    "ALEX_HOME_COMMANDS at a JSON map of device -> action -> argv."
)


def control(device: str, action: str, value: str | None = None) -> dict:
    """Act on a device. Home Assistant by entity id; else the command map."""
    device = (device or "").strip()
    action = (action or "").strip().lower()
    if not device or not action:
        return {"ok": False, "error": "device and action are required"}
    url, token = _ha()
    cmds = _commands()
    if not (url and token) and not cmds:
        return {"ok": False, "error": _NOT_CONFIGURED}

    if url and token and "." in device:
        domain = device.split(".", 1)[0]
        payload: dict[str, Any] = {"entity_id": device}
        service = _ACTIONS_TO_SERVICE.get(action, action)
        if action == "brightness" and value is not None:
            service, payload["brightness_pct"] = "turn_on", max(0, min(100, int(float(value))))
        elif action == "color" and value:
            service, payload["color_name"] = "turn_on", str(value)
        elif action == "temperature" and value is not None:
            service, payload["temperature"] = "set_temperature", float(value)
            domain = "climate"
        try:
            _ha_request("POST", f"services/{domain}/{service}", payload)
            return {"ok": True, "platform": "home_assistant", "device": device,
                    "action": action, "value": value}
        except (RuntimeError, ValueError) as exc:
            return {"ok": False, "error": str(exc)}

    argv = (cmds.get(device) or {}).get(action)
    if not argv or not isinstance(argv, list):
        return {"ok": False,
                "error": f"no command configured for {device!r} -> {action!r}"}
    try:
        proc = subprocess.run(
            [str(a) for a in argv] + ([str(value)] if value is not None else []),
            capture_output=True, text=True, timeout=TIMEOUT_S, shell=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": str(exc)}
    out = ((proc.stdout or "") + (proc.stderr or ""))[-800:]
    return {"ok": proc.returncode == 0, "platform": "command", "device": device,
            "action": action, "exit": proc.returncode, "output": out}


def status(device: str | None = None) -> dict:
    """Device state(s) from Home Assistant."""
    url, token = _ha()
    if not (url and token):
        return {"ok": False, "error": _NOT_CONFIGURED}
    try:
        if device:
            d = _ha_request("GET", f"states/{device.strip()}")
            attrs = d.get("attributes") or {}
            return {"ok": True, "device": device, "state": d.get("state"),
                    "name": attrs.get("friendly_name", device)}
        states = _ha_request("GET", "states") or []
        rows = [
            {"device": s.get("entity_id"), "state": s.get("state"),
             "name": (s.get("attributes") or {}).get("friendly_name", s.get("entity_id"))}
            for s in states[:60]
        ]
        return {"ok": True, "devices": rows, "count": len(states)}
    except RuntimeError as exc:
        return {"ok": False, "error": str(exc)}
