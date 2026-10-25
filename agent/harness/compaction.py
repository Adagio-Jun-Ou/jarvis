"""Context compaction — budgeted summarization, not a haircut.

What the strong open harnesses converged on (Hermes is the clearest
published design): when the conversation outgrows its window, the middle
is SUMMARIZED by an auxiliary model under an explicit token budget while
the head and tail are protected verbatim — and stale tool outputs are
pruned *before* summarization, because a six-turn-old inbox dump is the
cheapest thing in the window to give up and the most expensive to keep.

The shape here:

* **Head protection.** The first user message survives verbatim. It is
  the task; a summary of the task drifts, and everything downstream
  anchors on it.
* **Tail protection.** The most recent ``keep_tail`` messages survive
  verbatim — the model's working set.
* **Stale tool-result pruning.** Tool results older than the tail are
  reduced to a one-line stub (tool name + size) before anything is
  summarized. The *fact that the call happened* stays; the payload goes.
* **Budgeted aux-model summary.** The middle is compressed to roughly
  20% of its size, clamped to [~500, ~3000] tokens' worth of characters.
  The summarizer is told to keep decisions, identifiers, and open
  questions — the things a later turn actually reaches back for.
* **Deterministic fallback.** When no summarizer is available (or it
  fails), the middle degrades to the first-line digest this system used
  before. Compaction must never make a turn fail.
* **Lineage.** Each compaction folds the previous compaction summary in
  verbatim and stamps a generation counter, so repeated compactions
  form a chain instead of silently rewriting rewritten history.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Callable
from typing import Any

log = logging.getLogger("alex.harness.compaction")

#: Messages kept verbatim at the tail (the working set).
KEEP_TAIL = int(os.environ.get("ALEX_COMPACT_KEEP_TAIL", "14"))
#: Compaction triggers when history exceeds this many messages.
TRIGGER_MSGS = int(os.environ.get("ALEX_COMPACT_TRIGGER", "24"))
#: Summary budget: fraction of the middle's characters...
BUDGET_FRACTION = 0.2
#: ...clamped to this range (chars; ~4 chars per token).
BUDGET_MIN_CHARS = 2_000
BUDGET_MAX_CHARS = 12_000
#: Stale tool results are stubbed down to this preview length.
STUB_PREVIEW_CHARS = 120

_DIGEST_BANNER = "EARLIER CONVERSATION DIGEST"
_GEN_RE = re.compile(r"\(compaction #(\d+)\)")


def _text_of(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, dict):
        try:
            import json

            return json.dumps(content, default=str, ensure_ascii=False)
        except Exception:  # noqa: BLE001
            return str(content)
    return str(content or "")


def _stub_stale_tool_results(middle: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replace old tool-result payloads with one-line stubs, in place of
    the originals (the originals are not mutated)."""
    out: list[dict[str, Any]] = []
    for message in middle:
        if message.get("role") != "tool_result":
            out.append(message)
            continue
        text = _text_of(message)
        if len(text) <= STUB_PREVIEW_CHARS:
            out.append(message)
            continue
        stub = dict(message)
        stub["content"] = (
            f"[stale tool result pruned: {len(text)} chars; began: "
            f"{' '.join(text[:STUB_PREVIEW_CHARS].split())}…]"
        )
        out.append(stub)
    return out


def _deterministic_digest(middle: list[dict[str, Any]]) -> str:
    """The no-model fallback: one first-line bullet per message."""
    lines: list[str] = []
    for message in middle:
        role = str(message.get("role", "?"))
        text = _text_of(message).strip()
        first = text.splitlines()[0] if text else ""
        if first:
            lines.append(f"- {role}: {first[:120]}")
    return "\n".join(lines)[-2400:]


def _budget_for(middle_chars: int) -> int:
    return max(BUDGET_MIN_CHARS, min(int(middle_chars * BUDGET_FRACTION),
                                     BUDGET_MAX_CHARS))


_SUMMARY_PROMPT = (
    "Compress this conversation excerpt into a compact working summary "
    "another assistant will rely on to CONTINUE the work. Keep, verbatim "
    "where short: decisions made and why; concrete identifiers (ids, "
    "paths, emails, ticket/task numbers, amounts, dates); what each tool "
    "call established; unresolved questions and anything the user asked "
    "for that has not been delivered. Drop pleasantries and phrasing. "
    "Plain bullets, at most {budget} characters. Excerpt:\n\n{excerpt}"
)

# Incremental update (the Hermes compression design): the previous
# compaction's summary is handed BACK to the summarizer to be revised —
# finished items collapse, obsolete lines drop, new work merges in —
# instead of being concatenated ahead of a fresh summary. Concatenation
# was unbounded: by compaction #5 the digest was mostly digests of
# digests, growing while telling the model less.
_UPDATE_PROMPT = (
    "Below is the WORKING SUMMARY of a conversation so far, followed by "
    "the excerpt of what happened since. UPDATE the summary into one "
    "merged summary another assistant will rely on to CONTINUE the "
    "work: mark previously in-progress items done when the excerpt "
    "shows it, drop lines the excerpt makes obsolete, and fold in new "
    "decisions, identifiers (ids, paths, emails, task numbers, amounts, "
    "dates), tool findings, and unresolved questions. Plain bullets, at "
    "most {budget} characters — the merged summary REPLACES the old "
    "one.\n\nWORKING SUMMARY:\n{prior}\n\nSINCE THEN:\n{excerpt}"
)

#: The deterministic-fallback lineage must still be bounded: keep the
#: newest tail of the chain rather than growing without limit.
_FALLBACK_CHAIN_CHARS = 8_000


def compact_history(
    history: list[dict[str, Any]],
    *,
    llm_invoke: Callable[[list[dict[str, str]]], str] | None = None,
    trigger: int | None = None,
    keep_tail: int | None = None,
    live_note: str = "",
) -> tuple[list[dict[str, Any]], bool]:
    """Return ``(possibly-compacted history, whether compaction ran)``.

    Never raises, never returns an empty history, and never touches the
    input list.
    """
    trigger = TRIGGER_MSGS if trigger is None else trigger
    keep_tail = KEEP_TAIL if keep_tail is None else keep_tail
    if trigger <= 0 or len(history) <= trigger:
        return history, False

    # --- carve the protected regions ------------------------------------
    prior_summary = ""
    generation = 0
    body = list(history)
    if body and str(body[0].get("content", "")).startswith(_DIGEST_BANNER):
        head_text = str(body[0].get("content") or "")
        prior_summary = head_text.split("\n", 1)[-1].strip()
        m = _GEN_RE.search(head_text)
        generation = int(m.group(1)) if m else 1
        body = body[1:]

    anchor: dict[str, Any] | None = None
    for message in body:
        if message.get("role") == "user":
            anchor = message
            break
    tail = body[-keep_tail:] if keep_tail > 0 else []
    middle = body[: len(body) - len(tail)]
    middle = [m for m in middle if m is not anchor]
    if not middle:
        return history, False

    # --- prune stale payloads, then summarize ---------------------------
    middle = _stub_stale_tool_results(middle)
    excerpt = "\n".join(
        f"{m.get('role', '?')}: {_text_of(m)}" for m in middle
    )
    budget = _budget_for(len(excerpt))

    summary = ""
    updated_prior = False
    if llm_invoke is not None:
        try:
            if prior_summary:
                prompt = _UPDATE_PROMPT.format(
                    budget=budget,
                    prior=prior_summary[: budget * 3],
                    excerpt=excerpt[: budget * 5],
                )
            else:
                prompt = _SUMMARY_PROMPT.format(
                    budget=budget, excerpt=excerpt[: budget * 6]
                )
            raw = llm_invoke([{"role": "user", "content": prompt}])
            summary = str(raw or "").strip()[:budget]
            updated_prior = bool(summary) and bool(prior_summary)
        except Exception:  # noqa: BLE001
            log.warning("compaction summarizer failed; using the "
                        "deterministic digest", exc_info=True)
    if not summary:
        summary = _deterministic_digest(middle)
    if prior_summary and not updated_prior:
        # No summarizer to merge with: chain, but bounded — an unbounded
        # chain of digests grows while informing less.
        summary = (prior_summary + "\n" + summary)[-_FALLBACK_CHAIN_CHARS:]

    content = (
        f"{_DIGEST_BANNER} (compaction #{generation + 1}) — older turns "
        "were compressed; decisions and identifiers below still "
        "stand:\n" + summary
    )
    if live_note.strip():
        # Fresh, authoritative state the caller re-reads at compaction
        # time (e.g. the CURRENT task plan). The anchor keeps the plan
        # as it stood at turn START; mid-turn step changes lived only in
        # the middle — exactly the region compaction just compressed.
        content += (
            "\n\nCURRENT STATE (re-read live just now; where the digest "
            "disagrees, THIS is the truth):\n" + live_note.strip()
        )
    head = {"role": "system", "content": content}
    out: list[dict[str, Any]] = [head]
    if anchor is not None:
        out.append(anchor)
    out.extend(tail)
    return out, True
