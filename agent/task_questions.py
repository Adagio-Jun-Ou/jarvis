"""Questions — the agent asking a human, and waiting for the answer.

Every existing human-in-the-loop path in this system is fire-and-forget:
steering is a mailbox drained between passes, ``confirmation_required``
returns to the model immediately, ``check_plan_step(status='blocked')``
is a marker the run walks past. What none of them provide is the thing
task work actually needs mid-run: *"I cannot choose this for you —
which is it?"* — asked once, visibly, with the run parked until a crew
member answers.

The shape (the one the agent-platform literature converges on —
LangGraph's ``interrupt()``/resume, Linear's agent elicitation
activity):

* the question is a DURABLE row (`media.control.task_questions`), not a
  process-local flag — an answer given after the run died still lands
  in the thread and frames the next attempt;
* the question also lands in the task THREAD, where the crew already
  reads, and on the live event stream, so the board can badge
  "needs your answer" the moment it is asked;
* the run **waits** — a bounded, polling wait, so the answer is picked
  up even when it was written by another replica — and the governor is
  credited the waiting time: a person taking ten minutes to decide must
  not push the agent into its wall-clock ceiling;
* an unanswered question times out into *guidance*, not silence: the
  model is told to proceed on its own stated assumption and to treat
  the silence as nobody's approval. The row stays ``open`` for the
  crew.

Write asymmetry, enforced at the grant layer like acceptance criteria:
the agent may ASK (INSERT) and read; only the app — a person — may
resolve (UPDATE). An agent that could write ``answer`` could approve
itself.

Binding: like the plan tools, ``ask_user`` resolves its task from the
VM's session (``task-<id>``), never from a model-supplied id.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("agent.task_questions")

QUESTIONS_TABLE = "media.control.task_questions"
STATUSES = ("open", "answered", "cancelled")

MAX_QUESTION = 600
MAX_OPTIONS = 4
MAX_OPTION = 120
MAX_ANSWER = 2000
MAX_OPEN_PER_TASK = 3


def _wait_timeout_s() -> float:
    try:
        return float(os.environ.get("ALEX_ASK_WAIT_S", "300"))
    except ValueError:
        return 300.0


def _poll_s() -> float:
    try:
        return float(os.environ.get("ALEX_ASK_POLL_S", "5"))
    except ValueError:
        return 5.0


def _sql(stmt: str, params: dict | None = None) -> list[dict]:
    from agent.tools.impls import _sql as impl_sql

    return impl_sql(stmt, params)


# ---------------------------------------------------------------------------
# In-process waiters — the fast path. The DB poll is the truth; the
# waiter just wakes the parked thread without waiting out a poll cycle.
# ---------------------------------------------------------------------------


@dataclass
class _Waiter:
    event: threading.Event = field(default_factory=threading.Event)


_WAITERS: dict[str, _Waiter] = {}
_WAITERS_LOCK = threading.Lock()


def _register_waiter(question_id: str) -> _Waiter:
    with _WAITERS_LOCK:
        w = _WAITERS.setdefault(question_id, _Waiter())
    return w


def _drop_waiter(question_id: str) -> None:
    with _WAITERS_LOCK:
        _WAITERS.pop(question_id, None)


def _wake(question_id: str) -> None:
    with _WAITERS_LOCK:
        w = _WAITERS.get(question_id)
    if w is not None:
        w.event.set()


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


def _row_to_question(r: dict) -> dict[str, Any]:
    try:
        options = json.loads(r.get("options") or "[]")
    except (TypeError, ValueError):
        options = []
    return {
        "question_id": str(r.get("question_id") or ""),
        "task_id": str(r.get("task_id") or ""),
        "run_id": str(r.get("run_id") or ""),
        "asked_by": str(r.get("asked_by") or ""),
        "question": str(r.get("question") or ""),
        "options": options if isinstance(options, list) else [],
        "status": str(r.get("status") or "open"),
        "answer": str(r.get("answer") or ""),
        "answered_by": str(r.get("answered_by") or ""),
        "asked_at": r.get("asked_at"),
        "answered_at": r.get("answered_at"),
    }


def for_task(sql, task_id: str) -> list[dict[str, Any]]:
    """Every question ever asked on a task, oldest first."""
    rows = sql(
        f"SELECT question_id, task_id, run_id, asked_by, question, options, "
        f"status, answer, answered_by, asked_at, answered_at "
        f"FROM {QUESTIONS_TABLE} WHERE task_id = :t",
        {"t": task_id},
    )
    out = [_row_to_question(r) for r in rows]
    out.sort(key=lambda q: str(q.get("asked_at") or ""))
    return out


def open_for_task(sql, task_id: str) -> list[dict[str, Any]]:
    return [q for q in for_task(sql, task_id) if q["status"] == "open"]


def open_counts(sql) -> dict[str, int]:
    """task_id -> number of open questions, one scan for the whole board."""
    counts: dict[str, int] = {}
    try:
        rows = sql(
            f"SELECT task_id, status FROM {QUESTIONS_TABLE}", {}
        )
    except Exception:  # noqa: BLE001
        return counts
    for r in rows or []:
        if str(r.get("status") or "") == "open":
            tid = str(r.get("task_id") or "")
            if tid:
                counts[tid] = counts.get(tid, 0) + 1
    return counts


def _get(sql, question_id: str) -> dict[str, Any] | None:
    rows = sql(
        f"SELECT question_id, task_id, run_id, asked_by, question, options, "
        f"status, answer, answered_by, asked_at, answered_at "
        f"FROM {QUESTIONS_TABLE} WHERE question_id = :q",
        {"q": question_id},
    )
    return _row_to_question(rows[0]) if rows else None


def prompt_block(task_id: str) -> str:
    """Unanswered questions from earlier runs, rendered for a run prompt.

    An answered question needs no block — the answer was posted into the
    thread, which the prompt already carries as DISCUSSION. Open ones
    must be surfaced so the next attempt neither re-asks nor quietly
    assumes the silence resolved them.
    """
    try:
        open_qs = [
            q for q in for_task(_sql, task_id) if q["status"] == "open"
        ]
    except Exception:  # noqa: BLE001
        return ""
    if not open_qs:
        return ""
    lines = [f"  - {q['question']}" for q in open_qs[-MAX_OPEN_PER_TASK:]]
    return (
        "QUESTIONS YOU ASKED THE CREW EARLIER, STILL UNANSWERED:\n"
        + "\n".join(lines)
        + "\nDo not re-ask these. Proceed with your best judgment where "
        "you can, state each assumption explicitly, and leave anything "
        "truly blocked for the crew."
    )


# ---------------------------------------------------------------------------
# The human side — answer and cancel. `sql` is the caller's (the app's),
# because resolving a question is a PERSON's act.
# ---------------------------------------------------------------------------


def answer(
    sql, principal, task_id: str, question_id: str, answer_text: str
) -> dict[str, Any]:
    """A crew member answers an open question.

    Marks the row, posts the answer into the task thread (the record),
    wakes any parked run in this process, and — when the run is live but
    already past its wait — steers the answer into the loop so it still
    arrives mid-flight rather than next run.
    """
    answer_text = (answer_text or "").strip()[:MAX_ANSWER]
    if not answer_text:
        raise ValueError("an answer is required")
    q = _get(sql, question_id)
    if q is None or q["task_id"] != task_id:
        raise ValueError("no such question on this task")
    if q["status"] != "open":
        raise ValueError(f"this question is already {q['status']}")
    sql(
        f"UPDATE {QUESTIONS_TABLE} SET status = 'answered', answer = :a, "
        "answered_by = :by, answered_at = current_timestamp() "
        "WHERE question_id = :q AND status = 'open'",
        {"a": answer_text, "by": principal.email, "q": question_id},
    )
    from agent import taskboard

    try:
        taskboard.post_message(
            sql, task_id, author=principal.email, body=answer_text,
            kind="human",
        )
    except Exception:  # noqa: BLE001
        log.debug("answer thread post failed", exc_info=True)
    _wake(question_id)
    q = _get(sql, question_id) or {**q, "status": "answered",
                                   "answer": answer_text,
                                   "answered_by": principal.email}
    # If the run outlived its wait (timeout guidance already delivered),
    # the answer still reaches it as steering at the next safe point.
    try:
        run = taskboard.run_for_task(task_id)
        waiting = False
        with _WAITERS_LOCK:
            waiting = question_id in _WAITERS
        if run is not None and run.running and not waiting:
            run.steer(
                principal.email,
                f"[answer to your earlier question '{q['question'][:120]}']: "
                f"{answer_text}",
            )
    except Exception:  # noqa: BLE001
        log.debug("answer steering failed", exc_info=True)
    _emit_question(q)
    return q


def cancel_open(sql, task_id: str) -> int:
    """Cancel every open question on a task (the crew stopped the run).

    Parked waits wake immediately and report the cancellation to the
    model — which the governor's halt then makes moot.
    """
    open_qs = open_for_task(sql, task_id)
    for q in open_qs:
        sql(
            f"UPDATE {QUESTIONS_TABLE} SET status = 'cancelled', "
            "answered_at = current_timestamp() "
            "WHERE question_id = :q AND status = 'open'",
            {"q": q["question_id"]},
        )
        _wake(q["question_id"])
        _emit_question({**q, "status": "cancelled"})
    return len(open_qs)


def _emit_question(q: dict[str, Any]) -> None:
    """Whole-state question event onto the task's live stream, if any."""
    try:
        from agent import taskboard
        from alex.events import question_event

        run = taskboard.run_for_task(q.get("task_id") or "")
        if run is not None:
            run.emit(
                json.loads(question_event(question=q).to_json())
            )
    except Exception:  # noqa: BLE001
        log.debug("question event emit failed", exc_info=True)


# ---------------------------------------------------------------------------
# The agent-facing tool — session-bound, blocking, budget-credited
# ---------------------------------------------------------------------------


def _bound_task() -> str:
    from agent.swarm import current_vm
    from agent.task_plan import task_for_session

    vm = current_vm()
    if vm is None:
        return ""
    return task_for_session(getattr(vm, "session_id", ""))


def ask_user(
    question: str,
    options: list | None = None,
    *,
    user_email: str = "",
    instance_scope: str = "default",
) -> dict[str, Any]:
    """Ask the human crew a question you cannot answer for them and WAIT
    for the reply. Use for genuine decisions only — a preference, an
    authorization, a fork in the work — never for facts your tools can
    find. One clear question; 2-4 short options when the choice is
    enumerable. If nobody answers in time you are told to proceed on
    your own stated assumption."""
    from agent.swarm import current_vm

    task_id = _bound_task()
    if not task_id:
        return {
            "ok": False,
            "error": "ask_user works on board-task runs; in a chat the "
                     "person is already here — ask in your reply instead",
        }
    q_text = (question or "").strip()[:MAX_QUESTION]
    if not q_text:
        return {"ok": False, "error": "a question is required"}
    opts = [str(o).strip()[:MAX_OPTION] for o in (options or []) if str(o).strip()]
    if len(opts) > MAX_OPTIONS:
        return {
            "ok": False,
            "error": f"at most {MAX_OPTIONS} options — more is a survey, "
                     "not a decision",
        }
    already_open = open_for_task(_sql, task_id)
    if len(already_open) >= MAX_OPEN_PER_TASK:
        return {
            "ok": False,
            "error": f"{len(already_open)} questions are already waiting on "
                     "the crew; work what you can and check back",
        }

    question_id = f"q-{uuid.uuid4()}"
    vm = current_vm()
    run_id = ""
    try:
        from agent import taskboard

        run = taskboard.run_for_task(task_id)
        run_id = run.run_id if run is not None else ""
    except Exception:  # noqa: BLE001
        pass
    _sql(
        f"INSERT INTO {QUESTIONS_TABLE} "
        "(question_id, task_id, run_id, asked_by, question, options, "
        " status, asked_at) "
        "VALUES (:q, :t, :r, :by, :text, :opts, 'open', "
        "current_timestamp())",
        {
            "q": question_id,
            "t": task_id,
            "r": run_id,
            "by": "alex",
            "text": q_text,
            "opts": json.dumps(opts),
        },
    )
    # The thread is the record: the question lives where the crew reads.
    try:
        from agent import taskboard

        rendered = q_text if not opts else (
            q_text + "\n" + "\n".join(f"  {i}. {o}" for i, o in
                                      enumerate(opts, start=1))
        )
        taskboard.post_message(
            _sql, task_id, author="alex", body=f"❓ {rendered}", kind="agent"
        )
    except Exception:  # noqa: BLE001
        log.debug("question thread post failed", exc_info=True)
    _emit_question(_get(_sql, question_id) or {
        "question_id": question_id, "task_id": task_id,
        "question": q_text, "options": opts, "status": "open",
    })

    # Park. The event is the in-process fast path; the DB poll catches an
    # answer written by another replica. Waiting time is credited back to
    # the governor — a person deliberating must not eat the run's clock.
    outcome = _wait_for_answer(question_id, vm=vm, wait_s=_wait_timeout_s())
    if outcome is not None:
        if outcome["status"] == "answered":
            return {
                "ok": True,
                "answered": True,
                "question_id": question_id,
                "answer": outcome["answer"],
                "answered_by": outcome["answered_by"],
            }
        return {
            "ok": False,
            "cancelled": True,
            "question_id": question_id,
            "error": "the crew stopped this run; do not act further on "
                     "this question",
        }
    return {
        "ok": True,
        "answered": False,
        "question_id": question_id,
        "guidance": (
            f"No answer after {int(_wait_timeout_s())}s. Proceed with your "
            "best judgment: state the assumption you are making explicitly "
            "in your work and your summary, prefer the reversible path, and "
            "do not treat the silence as approval for any consequential "
            "action. The question stays open for the crew."
        ),
    }


def _wait_for_answer(
    question_id: str, *, vm=None, wait_s: float
) -> dict[str, Any] | None:
    """Park until the question resolves or the window closes.

    Returns ``{"status": "answered", "answer", "answered_by"}`` or
    ``{"status": "cancelled"}`` — or ``None`` on timeout. The wall time
    spent parked is credited back to the governor via ``vm``.
    """
    waiter = _register_waiter(question_id)
    deadline = time.time() + wait_s
    waited_from = time.time()
    outcome: dict[str, Any] | None = None
    try:
        while time.time() < deadline:
            # A stopped run must not keep a worker parked on a question
            # nobody will answer: refuse immediately (fail closed).
            if vm is not None and getattr(vm, "abort_requested", False):
                outcome = {"status": "cancelled", "reason": "run stopped"}
                break
            step = min(_poll_s(), max(0.0, deadline - time.time()))
            woke = waiter.event.wait(step or 0.01)
            q = _get(_sql, question_id)
            if q is None:
                break
            if q["status"] == "answered":
                outcome = {
                    "status": "answered",
                    "answer": q["answer"],
                    "answered_by": q["answered_by"],
                }
                break
            if q["status"] == "cancelled":
                outcome = {"status": "cancelled"}
                break
            if woke:
                waiter.event.clear()
    finally:
        _drop_waiter(question_id)
        waited = time.time() - waited_from
        if vm is not None:
            try:
                vm.credit_wait(waited)
            except Exception:  # noqa: BLE001
                log.debug("wait credit failed", exc_info=True)
    return outcome


def _approve_wait_s() -> float:
    try:
        return float(os.environ.get("ALEX_APPROVE_WAIT_S", "180"))
    except ValueError:
        return 180.0


def request_approval(
    task_id: str, question: str, *, vm=None, wait_s: float | None = None
) -> dict[str, Any]:
    """Park the run on a crew yes/no for one specific action.

    Same machinery as ask_user — durable row, thread post, live event,
    the board's answer panel — but the OPPOSITE default: an approval
    FAILS CLOSED. Only an explicit approve returns approved=True; a
    deny, a timeout, and a stopped run all refuse, and a timed-out
    approval is cancelled rather than left open, because a later
    "approve" landing on an action the run long since gave up on is an
    authorization nobody meant to give.
    """
    q_text = (question or "").strip()[:MAX_QUESTION]
    question_id = f"q-{uuid.uuid4()}"
    run_id = ""
    try:
        from agent import taskboard

        run = taskboard.run_for_task(task_id)
        run_id = run.run_id if run is not None else ""
    except Exception:  # noqa: BLE001
        pass
    _sql(
        f"INSERT INTO {QUESTIONS_TABLE} "
        "(question_id, task_id, run_id, asked_by, question, options, "
        " status, asked_at) "
        "VALUES (:q, :t, :r, 'alex', :text, :opts, 'open', "
        "current_timestamp())",
        {"q": question_id, "t": task_id, "r": run_id, "text": q_text,
         "opts": json.dumps(["approve", "deny"])},
    )
    try:
        from agent import taskboard

        taskboard.post_message(
            _sql, task_id, author="alex", kind="agent",
            body=f"❓ {q_text}\n  1. approve\n  2. deny",
        )
    except Exception:  # noqa: BLE001
        log.debug("approval thread post failed", exc_info=True)
    _emit_question(_get(_sql, question_id) or {
        "question_id": question_id, "task_id": task_id, "question": q_text,
        "options": ["approve", "deny"], "status": "open",
    })

    outcome = _wait_for_answer(
        question_id, vm=vm, wait_s=_approve_wait_s() if wait_s is None else wait_s
    )
    if outcome is not None and outcome["status"] == "answered":
        answer = str(outcome.get("answer") or "").strip().lower()
        approved = answer.startswith(("approve", "yes", "ok", "go ahead"))
        return {
            "approved": approved,
            "answer": outcome.get("answer"),
            "answered_by": outcome.get("answered_by"),
        }
    if outcome is None:
        # Timed out: close the window so a stale approve can't land later.
        try:
            _sql(
                f"UPDATE {QUESTIONS_TABLE} SET status = 'cancelled', "
                "answered_at = current_timestamp() "
                "WHERE question_id = :q AND status = 'open'",
                {"q": question_id},
            )
            _emit_question(_get(_sql, question_id) or {})
        except Exception:  # noqa: BLE001
            log.debug("approval timeout close failed", exc_info=True)
        return {"approved": False,
                "reason": f"no approval within {int(_approve_wait_s())}s"}
    return {"approved": False, "reason": "the crew stopped the run"}
