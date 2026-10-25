"""Run the whole system on this machine.

    python run_local.py            # http://127.0.0.1:8765

Local mode is the exercised dev path plus three things a personal
machine wants: a DURABLE store (SQLite at ~/.jarvis/alex.db), the LLM
via OpenRouter (OPENROUTER_API_KEY), and a real browser (Playwright).
Secrets load from ~/.jarvis/secrets/api_keys.env (KEY=value lines) the
same way Jarvis does, so one file feeds both.

Environment knobs (all optional):
  OPENROUTER_API_KEY   the LLM (falls back to a local Ollama daemon)
  ALEX_LLM_BACKEND     openrouter | ollama | mock
  OPENROUTER_MODEL     default z-ai/glm-5.3-flash
  ALEX_DB_PATH         default ~/.jarvis/alex.db
  ALEX_BROWSER         playwright | memory (default playwright if installed)
  ALEX_PORT            default 8765
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _load_secrets() -> None:
    f = Path.home() / ".jarvis" / "secrets" / "api_keys.env"
    if not f.exists():
        return
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main() -> None:
    _load_secrets()
    os.environ.setdefault("ALEX_ENV", "dev")
    os.environ.setdefault("ALEX_DB_PATH", str(Path.home() / ".jarvis" / "alex.db"))
    if "ALEX_BROWSER" not in os.environ:
        try:
            import playwright  # noqa: F401

            os.environ["ALEX_BROWSER"] = "playwright"
        except ImportError:
            os.environ["ALEX_BROWSER"] = "memory"
    if not os.environ.get("OPENROUTER_API_KEY") and not os.environ.get("ALEX_LLM_BACKEND"):
        print("no OPENROUTER_API_KEY: using the local Ollama daemon", file=sys.stderr)

    # Jarvis skills (SKILL.md playbooks) become practices on every boot:
    # unchanged ones are skipped, edited ones get a new version.
    from agent.skills_import import import_default_skills

    rep = import_default_skills(owner=os.environ.get("ALEX_USER_EMAIL", ""))
    if rep["imported"] or rep["updated"]:
        print(f"skills: +{len(rep['imported'])} new, {len(rep['updated'])} updated",
              file=sys.stderr)

    import uvicorn

    uvicorn.run(
        "app.app:app",
        host=os.environ.get("ALEX_HOST", "127.0.0.1"),
        port=int(os.environ.get("ALEX_PORT", "8765")),
        log_level="info",
    )


if __name__ == "__main__":
    main()
