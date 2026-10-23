"""SKILL.md -> practices: idempotent, versioned, frontmatter-tolerant."""

from __future__ import annotations

import pytest

from agent import practices, skills_import


@pytest.fixture
def sql(monkeypatch):
    monkeypatch.setenv("ALEX_ENV", "dev")
    from alex import config, testing

    config._settings = None
    testing._fakes = None
    from alex.testing import get_fakes
    from app._dev_seed import BEST_PRACTICES_COLS

    wh = get_fakes().warehouse
    wh.register("media.control.best_practices", BEST_PRACTICES_COLS, pk=["practice_id"])

    def _sql(stmt, params=None):
        res = wh.execute(stmt, params or {})
        as_dicts = getattr(res, "as_dicts", None)
        return as_dicts() if callable(as_dicts) else list(res or [])

    yield _sql
    config._settings = None
    testing._fakes = None


SKILL = """---
name: code-review-agent
description: Multi-pass code review
tags: [code, review, security]
metadata:
  jarvis:
    requires:
      bins: [python]
    emoji: "🔍"
---

# Code Review Agent Skill

Pass 1: security. Pass 2: performance.
"""


def test_parse_reads_flat_frontmatter_and_ignores_nested():
    s = skills_import.parse_skill_md(SKILL)
    assert s["name"] == "code-review-agent"
    assert s["tags"] == ["code", "review", "security"]
    assert s["body"].startswith("# Code Review Agent Skill")
    assert skills_import.parse_skill_md("# Just a body\ntext")["name"] == "Just a body"


def test_import_is_idempotent_and_versions_on_change(sql, tmp_path):
    d = tmp_path / "skills" / "code-review-agent"
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(SKILL, encoding="utf-8")

    r1 = skills_import.import_skills(sql, [tmp_path / "skills"], owner="dana@alex.dev")
    assert r1["imported"] == ["code-review-agent"]
    r2 = skills_import.import_skills(sql, [tmp_path / "skills"])
    assert r2["unchanged"] == ["code-review-agent"] and not r2["imported"]

    (d / "SKILL.md").write_text(SKILL + "\nPass 3: style.\n", encoding="utf-8")
    r3 = skills_import.import_skills(sql, [tmp_path / "skills"])
    assert r3["updated"] == ["code-review-agent"]
    active = [p for p in practices._all(sql) if p.active]
    assert len(active) == 1 and active[0].version == 2
    assert "Pass 3" in active[0].guidance
    assert active[0].applies_to == ["code", "review", "security"]
    retired = [p for p in practices._all(sql) if not p.active]
    assert len(retired) == 1, "the previous text stays on record"


def test_default_dirs_honour_env(monkeypatch, tmp_path):
    monkeypatch.setenv("ALEX_SKILL_DIRS", str(tmp_path))
    assert skills_import.default_dirs() == [tmp_path]
