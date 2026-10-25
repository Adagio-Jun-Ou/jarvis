"""
Strict preset for the L3 (async) critic and the L4 (deploy) gate.

When a model output is being reviewed *asynchronously* by a stronger
judge, we apply a stricter rubric: any ambiguity on the procedure rules
(A3, A4, A5) becomes a block, not a revise. This module is the helper
that the agent's deploy gate and the async judge both use.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger("agent.execution_review")


@dataclass
class ReviewVerdict:
    verdict: str  # 'ok' | 'block' | 'revise'
    reason: str = ""
    article: str | None = None
    confidence: float = 1.0


# Tools that always trigger a strict-mode review.
STRICT_TRIGGERS = frozenset(
    {
        "draft_email_reply",
        "send_email",
        "publish_post",
        "api_act",
        "trigger_workflow",
        "browser_submit",
        "create_file",
        "str_replace",
        "present_files",
    }
)

#: The most consequential subset — externalising actions — used when a
#: caller reviews with strict=False.
_ALWAYS_TRIGGERS = frozenset(
    {"send_email", "api_act", "browser_submit", "trigger_workflow"}
)


STRICT_PRESET = {
    "ambiguity_on_A3": "block",
    "ambiguity_on_A4": "block",
    "ambiguity_on_A5": "block",
    "ambiguity_on_A8": "block",
    "ambiguity_on_A7": "revise",
    "ambiguity_on_A1": "block",
    "ambiguity_on_A2": "block",
    "ambiguity_on_A6": "block",
    "ambiguity_on_A9": "block",
}


def should_review(tool_calls: list[str], *, strict: bool = True) -> bool:
    """Should the async critic be invoked? With strict=True, every
    risky tool triggers; with strict=False, only the most consequential
    (externalising) ones."""
    triggers = STRICT_TRIGGERS if strict else _ALWAYS_TRIGGERS
    return any(t in triggers for t in (tool_calls or []))


def review(
    *, draft: str, tool_calls: list[str], articles: list[dict], judge_invoke
) -> ReviewVerdict:
    """Run the strict async judge. `judge_invoke(messages) -> str` is the
    model invocation. Returns the verdict. The function is conservative:
    on any failure to parse, it returns block."""
    if not tool_calls:
        return ReviewVerdict(verdict="ok")
    prompt = _build_strict_prompt(articles)
    user = (
        f"Draft reply:\n{draft}\n\n"
        f"Tool calls: {', '.join(tool_calls)}\n\n"
        "Return JSON: {\"verdict\": \"ok\"|\"block\"|\"revise\", "
        "\"article\": \"<id or null>\", \"reason\": \"...\"}."
    )
    try:
        raw = judge_invoke(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user},
            ]
        )
    except Exception:
        log.exception("async judge failed; defaulting to block")
        return ReviewVerdict(
            verdict="block", reason="judge unavailable", confidence=0.0
        )
    import json
    import re

    m = re.search(r"\{.*\}", raw or "", re.DOTALL)
    if not m:
        return ReviewVerdict(
            verdict="block", reason="judge unparseable", confidence=0.0
        )
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return ReviewVerdict(
            verdict="block", reason="judge JSON malformed", confidence=0.0
        )
    # Normalize and fail closed: a judge returning null, empty, or
    # unknown-case verdicts used to slip through as a non-"ok",
    # non-"block" value, and the caller's == "block" check then
    # SHIPPED the unreviewed answer on exactly the risky turns this
    # judge exists to gate.
    verdict = str(d.get("verdict") or "").strip().lower()
    if verdict not in ("ok", "block", "revise"):
        verdict = "block"
    return ReviewVerdict(
        verdict=verdict,
        reason=d.get("reason", ""),
        article=d.get("article"),
    )


def _build_strict_prompt(articles: list[dict]) -> str:
    parts = [
        "You are a strict constitutional reviewer. On AMBIGUITY for any "
        "procedure or prohibition article (A1, A3, A4, A5, A8, A9) you "
        "must return verdict=block, not revise. For A7 (data minimum) "
        "ambiguity you may return revise.",
        "",
        "ACTIVE ARTICLES:",
    ]
    for a in articles:
        parts.append(
            f"- {a['article_id']} ({a.get('severity','blocking')}): {a['title']} — {a['text']}"
        )
    return "\n".join(parts)
