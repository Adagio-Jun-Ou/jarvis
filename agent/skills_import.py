"""SKILL.md → practices. Jarvis's skills are Markdown playbooks with
YAML frontmatter (the OpenClaw / Claude Code shape); this engine's
guidance layer is the practices library — durable, versioned, matched
into the prompt by relevance. So a skill becomes a practice: the
frontmatter's name/description/tags are the title and applies_to, the
body is the guidance, and ``source_ref = skill:<name>`` makes re-import
idempotent (unchanged = skipped; changed = a superseding version, so
the old text stays on record).

Directories (``ALEX_SKILL_DIRS``, os.pathsep-separated; defaults:
``~/.jarvis/skills``, ``<repo>/skills``). No YAML dependency — the
frontmatter subset used is flat keys and ``[a, b]`` lists.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

log = logging.getLogger("agent.skills_import")

MAX_GUIDANCE = 6000
_FM_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.DOTALL)


def parse_skill_md(text: str) -> dict[str, Any]:
    """Frontmatter + body. Tolerant: no frontmatter -> body only."""
    m = _FM_RE.match(text.strip())
    meta: dict[str, Any] = {}
    body = text.strip()
    if m:
        body = m.group(2).strip()
        for line in m.group(1).splitlines():
            if not line.strip() or line.startswith((" ", "\t")):
                continue  # nested metadata (jarvis.requires...) is ignored
            if ":" not in line:
                continue
            key, _, val = line.partition(":")
            val = val.strip().strip('"').strip("'")
            if val.startswith("[") and val.endswith("]"):
                meta[key.strip()] = [
                    v.strip().strip('"').strip("'") for v in val[1:-1].split(",") if v.strip()
                ]
            else:
                meta[key.strip()] = val
    name = str(meta.get("name") or "").strip()
    if not name:
        first = next((ln for ln in body.splitlines() if ln.strip()), "skill")
        name = first.lstrip("# ").strip()[:80]
    return {
        "name": name,
        "description": str(meta.get("description") or "").strip(),
        "tags": [str(t) for t in (meta.get("tags") or []) if str(t).strip()],
        "body": body,
    }


def default_dirs(repo_root: str | None = None) -> list[Path]:
    env = os.environ.get("ALEX_SKILL_DIRS", "").strip()
    if env:
        return [Path(p) for p in env.split(os.pathsep) if p.strip()]
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[1]
    return [Path.home() / ".jarvis" / "skills", root / "skills"]


def import_skills(sql, dirs: list[Path] | None = None, *, owner: str = "") -> dict:
    """Publish every SKILL.md under ``dirs`` as a practice. Returns
    {imported: [...], updated: [...], unchanged: [...], errors: [...]}."""
    from agent import practices

    dirs = dirs if dirs is not None else default_dirs()
    report: dict[str, list] = {"imported": [], "updated": [], "unchanged": [], "errors": []}
    existing = {p.source_ref: p for p in practices._all(sql) if p.active and p.source_ref}
    for d in dirs:
        if not d.is_dir():
            continue
        for md in sorted(d.glob("*/SKILL.md")) + sorted(d.glob("SKILL.md")):
            try:
                skill = parse_skill_md(md.read_text(encoding="utf-8"))
                guidance = skill["body"][:MAX_GUIDANCE]
                if skill["description"] and not guidance.startswith(skill["description"]):
                    guidance = f"{skill['description']}\n\n{guidance}"
                ref = f"skill:{skill['name']}"
                old = existing.get(ref)
                if old is not None and old.guidance.strip() == guidance.strip():
                    report["unchanged"].append(skill["name"])
                    continue
                practices.upsert(
                    sql,
                    title=skill["name"],
                    guidance=guidance,
                    applies_to=skill["tags"],
                    rationale=f"imported from {md}",
                    owner=owner,
                    source_ref=ref,
                    supersedes=old.practice_id if old is not None else "",
                )
                report["updated" if old is not None else "imported"].append(skill["name"])
            except Exception as exc:  # noqa: BLE001
                log.warning("skill import failed for %s: %s", md, exc)
                report["errors"].append(f"{md}: {exc}")
    return report


def import_default_skills(owner: str = "") -> dict:
    """Boot-time import against the live store; never raises."""
    try:
        from agent.warehouse import make_sql

        return import_skills(make_sql(), owner=owner)
    except Exception as exc:  # noqa: BLE001
        log.warning("skill import skipped: %s", exc)
        return {"imported": [], "updated": [], "unchanged": [], "errors": [str(exc)]}
