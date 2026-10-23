"""
Inline auto-review (L3 of the constitution).

The critic receives the active articles as its rubric and scores the
drafted reply + tool-call set for constitutional compliance. The
output is one of:
  - "ok"     — pass, return the reply unchanged
  - "block"  — refusal; replace the reply with a structured denial
  - "revise" — re-invoke the model with a revision hint

The auto-review is "inline" because it runs synchronously after the
model produces a draft and before the reply is shown to the user.
The async, L3-batched critic lives in `pipelines/agent_review_job.py`.
"""

from __future__ import annotations

import json
import logging
import re

from agent.constitution import build_critic_prompt

log = logging.getLogger("agent.auto_review")

# Tools that always trigger a review.
RISKY_TOOLS = frozenset(
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
        "run_python",
        "bash",
    }
)

# The critic's system prompt. {constitution_block} is filled in with the
# active articles at every call.
CRITIC_PROMPT = """You are a constitutional reviewer. Your job is to detect
violations of the active Alex articles in the agent's draft response
and the tool calls it intends to make.

{constitution_block}

You will be given:
  - The user's last message.
  - The list of tool calls the agent is about to make (name only).
  - The agent's draft reply text.

Respond with a single JSON object, no prose:
  {{"verdict": "ok" | "block" | "revise",
    "reason": "<short reason if verdict != ok>",
    "article": "<article id being violated, or null>"}}

Be strict but fair. A "block" must be supported by an active article.
A "revise" means a small fix is needed (tone, missing call). A "ok"
means the draft and tool calls are within the law."""


# Hint appended to the model when the critic says "revise".
REVISION_HINT = (
    "The constitutional reviewer asked for a small revision. Reason: {reason}. "
    "Please adjust the reply to comply. Do not change intent."
)


def needs_review(tool_calls: list[str]) -> bool:
    """Does this turn involve any risky tool? Cheap, in-process check."""
    return any(tc in RISKY_TOOLS for tc in (tool_calls or []))


def _parse_critic_json(s: str) -> dict:
    """Parse the critic's JSON. Tolerate fences and prose; default to
    `revise` on parse failure so the caller retries with a hint."""
    if not s:
        return {"verdict": "ok", "reason": "", "article": None}
    s = s.strip()
    # Strip code fences
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```\s*$", "", s)
    # Find the first {...} block
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if not m:
        return {
            "verdict": "revise",
            "reason": "critic output not JSON",
            "article": None,
        }
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"verdict": "revise", "reason": "critic JSON malformed", "article": None}


def inline_review(
    last_user_msg: str,
    tool_calls: list[str],
    draft_reply: str,
    *,
    articles: list[dict],
    critic_invoke,
) -> dict:
    """Run the inline critic. `critic_invoke` is the cheap model
    invocation (e.g. claude-haiku). Returns the parsed verdict."""
    rubric = build_critic_prompt(articles)
    user = (
        f"User's last message:\n{last_user_msg}\n\n"
        f"Tool calls the agent is about to make:\n{', '.join(tool_calls) or '(none)'}\n\n"
        f"Agent's draft reply:\n{draft_reply}\n"
    )
    prompt = CRITIC_PROMPT.format(constitution_block=rubric)
    try:
        raw = critic_invoke(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user},
            ]
        )
    except Exception:
        log.exception("critic invoke failed; defaulting to ok")
        return {"verdict": "ok", "reason": "critic unavailable", "article": None}
    return _parse_critic_json(raw)
