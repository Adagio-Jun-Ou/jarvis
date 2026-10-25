"""
Streaming event protocol between the agent endpoint, the app backend, and
the SDK.

Wire format (Server-Sent Events over `text/event-stream`):
  data: {"type": "text", "text": "Hello"}\n\n
  data: {"type": "reasoning", "phase": "plan", "text": "User asked about..."}\n\n
  data: {"type": "tool", "tool": "check_rights", "args": {"asset_id": "A1"}, "result": "CLEARED"}\n\n
  data: {"type": "law_block", "article": "A4", "detail": "check_rights not performed"}\n\n
  data: {"type": "confirmation_required", "token": "...", "summary": "...", "body": {...}}\n\n
  data: {"type": "artifact", "path": "outputs/report.md", "kind": "file"}\n\n
  data: {"type": "done", "session_id": "..."}\n\n
  data: [DONE]\n\n   (sentinel for older clients; new clients may also rely on `done` event)

Event types are deliberately small, ordered, and append-only. A consumer
that does not recognize a type renders nothing (it does not error).

Each event is a single JSON object on a single line. We do not interleave
fragments of multiple events on the same line. Newline inside a string is
explicitly escaped by the JSON encoder.

This module is pure-Python, no IO, no Databricks SDK — so it can be
imported by the agent, the app, the SDK, and tests.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

# Event types — keep this list stable. New types are additive.
EVENT_TEXT = "text"
EVENT_REASONING = "reasoning"
EVENT_TOOL = "tool"
EVENT_LAW_BLOCK = "law_block"
EVENT_CONFIRMATION = "confirmation_required"
EVENT_ARTIFACT = "artifact"
# What the agent consulted before answering: retrieved passages, recalled
# memories, graph facts. Emitted so grounding is visible, not implied.
EVENT_CONTEXT = "context"
# A decision by the execution governor: this run was warned or stopped
# for a reason the user is entitled to see. Safety information that is
# invisible is indistinguishable from a hang.
EVENT_GOVERNOR = "governor"
# A human steered the agent mid-run: their note has been queued and will
# be read at the next safe point in the loop. Steering is not interrupt
# or cancel — it is course correction while the work continues.
EVENT_STEERING = "steering"
EVENT_DONE = "done"
EVENT_ERROR = "error"
# One named sub-agent's lifecycle and forwarded work inside a swarm run.
# `source` is the sub-agent's name, `phase` is either a lifecycle stage
# (spawned/done/failed) or the type of the child event being forwarded,
# and `body` carries that child event verbatim. One event type rather
# than N mirrored ones, so the UI can lane by `source` without a schema
# per child kind.
EVENT_SUBAGENT = "subagent"
# The task plan changed: declared, replanned, or a step's status moved.
# Carries the WHOLE current plan so a late-joining watcher renders the
# checklist from any single event.
EVENT_PLAN = "plan"
# An acceptance criterion's verdict changed. Same whole-state contract
# as EVENT_PLAN: carries all criteria so any single event renders.
EVENT_ACCEPTANCE = "acceptance"
# The agent asked the crew a question and the run is (or was) parked on
# it — elicitation. Carries the WHOLE question row (id, text, options,
# status, answer) so any single event renders the panel; emitted again
# when the question is answered or cancelled.
EVENT_QUESTION = "question"

ALL_TYPES = frozenset(
    {
        EVENT_TEXT,
        EVENT_REASONING,
        EVENT_TOOL,
        EVENT_LAW_BLOCK,
        EVENT_CONFIRMATION,
        EVENT_ARTIFACT,
        EVENT_CONTEXT,
        EVENT_GOVERNOR,
        EVENT_STEERING,
        EVENT_SUBAGENT,
        EVENT_PLAN,
        EVENT_ACCEPTANCE,
        EVENT_QUESTION,
        EVENT_DONE,
        EVENT_ERROR,
    }
)


@dataclass
class Event:
    type: str
    # Type-specific payload — only the keys relevant to `type` should be set.
    text: str | None = None
    phase: str | None = None
    tool: str | None = None
    args: dict | None = None
    result: Any | None = None
    article: str | None = None
    detail: str | None = None
    token: str | None = None
    summary: str | None = None
    body: dict | None = None
    path: str | None = None
    source: str | None = None
    # Governor decisions.
    action: str | None = None
    pathology: str | None = None
    reason: str | None = None
    state: dict | None = None
    kind: str | None = None
    session_id: str | None = None
    code: str | None = None
    message: str | None = None
    extras: dict = field(default_factory=dict)

    def to_json(self) -> str:
        d = {k: v for k, v in asdict(self).items() if v is not None and k != "extras"}
        d["type"] = self.type
        for k, v in self.extras.items():
            d.setdefault(k, v)
        # Compact, deterministic key order, no extra whitespace.
        return json.dumps(d, separators=(",", ":"), ensure_ascii=False)

    def to_sse(self) -> str:
        return f"data: {self.to_json()}\n\n"


def parse(line: str) -> Event | None:
    """Parse a single `data: ...` line. Returns None for sentinel/empty."""
    line = line.strip()
    if not line.startswith("data: "):
        return None
    body = line[6:].strip()
    if not body or body == "[DONE]":
        return None
    try:
        d = json.loads(body)
    except json.JSONDecodeError:
        # Legacy clients that sent raw text deltas — wrap as a text event.
        return Event(type=EVENT_TEXT, text=body)
    if not isinstance(d, dict) or "type" not in d:
        return Event(type=EVENT_TEXT, text=body)
    t = d["type"]
    if t not in ALL_TYPES:
        # Unknown type — keep it as a passthrough in `extras` so consumers
        # can decide what to do.
        e = Event(type=t)
        for k, v in d.items():
            if k != "type":
                e.extras[k] = v
        return e
    known = {f for f in Event.__dataclass_fields__ if f not in ("type", "extras")}
    e = Event(type=t)
    for k, v in d.items():
        if k == "type":
            continue
        if k in known:
            setattr(e, k, v)
        else:
            e.extras[k] = v
    return e


def text_event(text: str) -> Event:
    return Event(type=EVENT_TEXT, text=text)


def done_event(session_id: str | None = None) -> Event:
    return Event(type=EVENT_DONE, session_id=session_id)


def error_event(code: str, message: str) -> Event:
    return Event(type=EVENT_ERROR, code=code, message=message)


def tool_event(tool: str, args: dict | None = None, result: Any = None) -> Event:
    return Event(type=EVENT_TOOL, tool=tool, args=args, result=result)


def reasoning_event(phase: str, text: str) -> Event:
    return Event(type=EVENT_REASONING, phase=phase, text=text)


def law_block_event(article: str, detail: str) -> Event:
    return Event(type=EVENT_LAW_BLOCK, article=article, detail=detail)


def confirmation_event(
    *, token: str, summary: str, body: dict | None = None
) -> Event:
    return Event(type=EVENT_CONFIRMATION, token=token, summary=summary, body=body)


def context_event(
    source: str, items: list | None = None, summary: str = ""
) -> Event:
    """What the agent consulted before answering.

    ``source`` names the retrieval channel (``"knowledge"``, ``"memory"``,
    ``"graph"``), ``items`` carries the citations/facts themselves so the
    UI can show exactly which passage grounded which claim, and
    ``summary`` is the one-line version for the execution log.
    """
    ev = Event(type=EVENT_CONTEXT, source=source, summary=summary)
    ev.extras = {"items": list(items or [])}
    return ev


def event_to_dict(ev: Any) -> dict:
    """Convert an Event (or anything event-shaped) to a plain dict.

    Prefers the Event's own serialisation, which carries every populated
    field plus extras. Hand-written key lists were dropping fields
    silently — a governor decision arrived with its action and reason
    stripped, which is precisely the information the user needs.
    """
    to_json = getattr(ev, "to_json", None)
    if callable(to_json):
        try:
            return json.loads(to_json())
        except (TypeError, ValueError):
            pass
    if isinstance(ev, dict):
        return dict(ev)
    out = {"type": getattr(ev, "type", "event")}
    for key, value in vars(ev).items() if hasattr(ev, "__dict__") else ():
        if value is not None and key != "extras":
            out[key] = value
    extras = getattr(ev, "extras", None)
    if isinstance(extras, dict):
        out.update(extras)
    return out


def governor_event(
    *,
    action: str,
    reason: str,
    pathology: str = "",
    detail: dict | None = None,
    state: dict | None = None,
) -> Event:
    """A bound was reached, or a runaway was detected.

    Emitted into the same stream as everything else so a halted run
    explains itself where the user is already looking, rather than
    appearing to stall.
    """
    ev = Event(
        type=EVENT_GOVERNOR,
        action=action,
        reason=reason,
        pathology=pathology or None,
        state=state,
    )
    ev.extras = {"detail": dict(detail or {})}
    return ev


def plan_event(*, task_id: str, steps: list) -> Event:
    """The current plan, whole, after any change."""
    return Event(type=EVENT_PLAN, body={"task_id": task_id, "steps": steps})


def acceptance_event(*, task_id: str, criteria: list) -> Event:
    """All acceptance criteria, whole, after any verdict change."""
    return Event(
        type=EVENT_ACCEPTANCE, body={"task_id": task_id, "criteria": criteria}
    )


def question_event(*, question: dict) -> Event:
    """One agent question to the crew, whole (id, text, options, status,
    answer) — emitted when asked and again on every status change."""
    return Event(type=EVENT_QUESTION, body=dict(question or {}))


def subagent_event(
    *,
    name: str,
    phase: str,
    text: str = "",
    body: dict | None = None,
) -> Event:
    """One sub-agent's lifecycle stage or forwarded child event."""
    return Event(
        type=EVENT_SUBAGENT,
        source=name,
        phase=phase,
        text=text or None,
        body=body,
    )


def steering_event(*, author: str, text: str, delivered: bool = True) -> Event:
    """A human steered the agent mid-run.

    ``delivered`` is False on the *acknowledgement* (the note was
    queued and will be read at the next safe point) and True when the
    note is actually injected into the loop, so the crew watching the
    stream can tell "sent" from "read".
    """
    ev = Event(type=EVENT_STEERING, text=text, source=author)
    ev.extras = {"delivered": delivered}
    return ev


def artifact_event(path: str, kind: str = "file", preview: str = "") -> Event:
    ev = Event(type=EVENT_ARTIFACT, path=path, kind=kind)
    if preview:
        ev.extras = {"preview": preview}
    return ev
