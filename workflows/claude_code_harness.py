"""
Claude Code harness: the agent's guided-workflow builder.

When the user wants to build a complex workflow (multi-step, code + API
+ decisions), the agent enters a "Claude Code mode": it produces a
WORKPLAN.md, the user ratifies it, then the agent writes a workflow
catalog entry + a n8n webhook payload, and the user clicks Run.

This file is the harness — the in-process state machine. The actual
planner lives in `claude_code_planner.py` (separate module, loaded
lazily so the rest of the agent doesn't pay the import cost).

States:
  IDLE       -> nothing in progress
  PLANNING   -> the planner is producing WORKPLAN.md
  REVIEW     -> waiting for the user to ratify
  EXECUTING  -> running the plan, writing outputs
  DONE       -> terminal
  ABORTED    -> terminal

Persistence: each non-terminal state is written to
`media.alex.workplan_state` so a session reload picks up where we left
off. Keys are (user_email, plan_id).
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field

log = logging.getLogger("workflows.claude_code_harness")

STATE_IDLE = "IDLE"
STATE_PLANNING = "PLANNING"
STATE_REVIEW = "REVIEW"
STATE_EXECUTING = "EXECUTING"
STATE_DONE = "DONE"
STATE_ABORTED = "ABORTED"


@dataclass
class Workplan:
    plan_id: str
    user_email: str
    state: str = STATE_IDLE
    workplan_md: str = ""
    workflow_id: str = ""
    output_paths: list[str] = field(default_factory=list)
    audit_trail: list[dict] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def transition(self, new_state: str, **details) -> None:
        if new_state not in (STATE_IDLE, STATE_PLANNING, STATE_REVIEW,
                              STATE_EXECUTING, STATE_DONE, STATE_ABORTED):
            raise ValueError(f"unknown state: {new_state}")
        if not _valid_transition(self.state, new_state):
            raise ValueError(f"illegal transition {self.state} -> {new_state}")
        self.audit_trail.append({
            "from": self.state, "to": new_state,
            "ts": time.time(), **details,
        })
        self.state = new_state
        self.updated_at = time.time()


def _valid_transition(frm: str, to: str) -> bool:
    table = {
        STATE_IDLE: {STATE_PLANNING, STATE_ABORTED},
        STATE_PLANNING: {STATE_REVIEW, STATE_ABORTED},
        STATE_REVIEW: {STATE_EXECUTING, STATE_PLANNING, STATE_ABORTED},
        STATE_EXECUTING: {STATE_DONE, STATE_ABORTED},
    }
    return to in table.get(frm, set())


class Harness:
    """Per-user harness. The agent holds one instance per active session."""

    def __init__(self, user_email: str, llm_invoke=None) -> None:
        self.user_email = user_email
        self._plans: dict[str, Workplan] = {}
        self._llm_invoke = llm_invoke  # set by the agent

    def start_plan(self, prompt: str) -> Workplan:
        plan = Workplan(plan_id=str(uuid.uuid4()), user_email=self.user_email)
        self._plans[plan.plan_id] = plan
        plan.transition(STATE_PLANNING, prompt=prompt[:200])
        return plan

    def produce_workplan(self, plan_id: str, prompt: str) -> str:
        plan = self._plans[plan_id]
        if plan.state != STATE_PLANNING:
            raise ValueError(f"cannot produce workplan in state {plan.state}")
        if self._llm_invoke is None:
            plan.workplan_md = _stub_workplan(prompt)
        else:
            plan.workplan_md = self._llm_invoke([
                {"role": "system", "content": _PLANNER_SYSTEM},
                {"role": "user", "content": prompt},
            ])
        plan.transition(STATE_REVIEW)
        return plan.workplan_md

    def ratify(self, plan_id: str) -> dict:
        plan = self._plans[plan_id]
        plan.transition(STATE_EXECUTING)
        return {"plan_id": plan_id, "state": plan.state}

    def abort(self, plan_id: str, reason: str = "") -> None:
        plan = self._plans[plan_id]
        plan.transition(STATE_ABORTED, reason=reason)

    def complete(self, plan_id: str, output_paths: list[str]) -> None:
        plan = self._plans[plan_id]
        plan.output_paths = list(output_paths)
        plan.workflow_id = f"user-{self.user_email.split('@', 1)[0]}-{plan_id[:8]}"
        plan.transition(STATE_DONE)
        # The agent wires the WORKPLAN into outputs/ and writes a
        # workflow_runs row.

    def get(self, plan_id: str) -> Workplan | None:
        return self._plans.get(plan_id)


_PLANNER_SYSTEM = """You are a workflow planner. The user describes a
multi-step task; you produce a WORKPLAN.md that the user will ratify
before any code runs. The plan is a numbered list of steps, each
labeled with:

  * [tool] the agent tool to call (create_task, api_act, ...)
  * [expected output] what the step produces
  * [risk] any constitutional article that the step touches

Do not include steps that exceed the user's authority. If a step is
risky, mark it as a TWO-PHASE step (the user must click Run it).

Output only Markdown, no preamble."""


def _stub_workplan(prompt: str) -> str:
    """Used when the LLM is unavailable (e.g. dev without a real
    model endpoint). The plan is intentionally minimal."""
    return (
        f"# WORKPLAN for: {prompt[:80]}\n\n"
        "1. [list_my_tasks] Survey the user's open tasks. "
        "   [expected output] a list of pending items. "
        "   [risk] none.\n"
    )
