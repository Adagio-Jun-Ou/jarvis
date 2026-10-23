#!/usr/bin/env python
"""Local diagnostic for the engine + Jarvis tool adapter.

Answers the questions a person actually has when their local agent will
not start or cannot do what they expect, without printing any secret
values and without needing a network:

* which LLM backend is configured, and is its key present?
* where is the durable store, and can this process write there?
* which optional "arms" (browser, voice, screen) are installed?
* did the boot-time registration load the adapted Jarvis library?
* does the dev ACL actually grant the capabilities the tools need?
* did the constitution load?

Run from the repo root:

    python scripts/engine/doctor.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("ALEX_ENV", "dev")

_OK = "ok  "
_WARN = "warn"
_FAIL = "fail"
_failed = False


def _say(level: str, label: str, detail: str = "") -> None:
    global _failed
    if level == _FAIL:
        _failed = True
    print(f"[{level}] {label}" + (f": {detail}" if detail else ""))


def _arm(name: str, module: str, why: str) -> None:
    present = importlib.util.find_spec(module) is not None
    if present:
        _say(_OK, name, "installed")
    else:
        _say(_WARN, name, f"not installed — {why}")


def main() -> int:
    from alex.config import get_settings

    settings = get_settings()
    print(f"Alex local doctor — repo {ROOT}")
    print(f"mode: {settings.mode}\n")

    # -- LLM -----------------------------------------------------------
    from alex.llm import llm_backend

    backend = llm_backend()
    key = bool(os.environ.get("OPENROUTER_API_KEY", "").strip())
    # Never print the key itself, only whether one is present.
    _say(_OK, "llm backend", backend)
    if backend == "openrouter":
        _say(_OK if key else _FAIL, "OPENROUTER_API_KEY",
             "present" if key else "missing (set it in "
             "~/.jarvis/secrets/api_keys.env)")
    _say(_OK, "default model", settings.default_llm_endpoint)
    secrets = Path.home() / ".jarvis" / "secrets" / "api_keys.env"
    _say(_OK if secrets.exists() else _WARN, "secrets file",
         str(secrets) + ("" if secrets.exists() else " (absent)"))
    if backend == "ollama":
        _say(_WARN, "ollama", "expected at http://127.0.0.1:11434 "
             "(not contacted by this check)")

    # -- durable store --------------------------------------------------
    db = Path(os.environ.get(
        "ALEX_DB_PATH", str(Path.home() / ".jarvis" / "alex.db")))
    parent = db.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
        writable = os.access(parent, os.W_OK)
    except Exception:
        writable = False
    _say(_OK if writable else _FAIL, "store", str(db))
    if not writable:
        _say(_FAIL, "store parent", f"not writable: {parent}")

    # -- optional arms --------------------------------------------------
    _arm("browser", "playwright", "browser tools fall back to the memory "
         "backend")
    _arm("voice (stt)", "faster_whisper", "voice input unavailable")
    _arm("voice (tts)", "edge_tts", "voice output unavailable")
    _arm("screen", "mss", "desktop screenshots unavailable")
    _arm("input", "pynput", "desktop control unavailable")
    _arm("ocr", "pytesseract", "screen text finding unavailable")

    # -- registry + the adapted library ---------------------------------
    from agent.tools.registry import TOOL_REGISTRY

    builtin = len(TOOL_REGISTRY)
    deferred = sum(1 for s in TOOL_REGISTRY.values()
                   if getattr(s, "deferred", False))
    _say(_OK if builtin else _FAIL, "tool registry",
         f"{builtin} tools ({deferred} deferred)")

    try:
        from agent.jarvis_tools import register_jarvis_tools

        adapted = register_jarvis_tools()
        _say(_OK if adapted else _WARN, "jarvis library",
             f"{len(adapted)} adapted tools")
        acts = sum(1 for c in adapted.values() if c.endswith(".act"))
        _say(_OK, "jarvis acts",
             f"{acts} tools require the .act capability (policy-ASK)")
    except Exception as exc:  # noqa: BLE001
        _say(_FAIL, "jarvis library", f"registration failed: {exc}")

    # -- dev ACL + constitution -----------------------------------------
    if settings.mode == "dev":
        try:
            from alex.testing import get_fakes
            from app._dev_seed import seed_dev_warehouse

            wh = get_fakes().warehouse
            seed_dev_warehouse(wh)

            def _sql(stmt, params=None, **kwargs):
                bound = params if params is not None else kwargs.get("parameters")
                return wh.execute(stmt, bound or {}).as_dicts()

            caps = {
                str(r.get("capability"))
                for r in _sql(
                    "SELECT capability FROM "
                    "media.control.effective_employee_acl",
                    {},
                )
                if r.get("capability")
            }
            jarvis_caps = {c for c in caps if c.startswith("jarvis.")}
            _say(_OK if jarvis_caps else _FAIL, "dev ACL",
                 f"{len(caps)} capabilities ({len(jarvis_caps)} jarvis)")
            from agent.constitution import load_articles

            articles = load_articles(_sql)
            _say(_OK if articles else _WARN, "constitution",
                 f"{len(articles)} articles")
        except Exception as exc:  # noqa: BLE001
            _say(_WARN, "dev seed", f"could not verify: {exc}")

    print()
    print("FAILED" if _failed else "PASSED — core checks are satisfied")
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
