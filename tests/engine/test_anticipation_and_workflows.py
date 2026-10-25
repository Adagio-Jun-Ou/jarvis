"""
Tests for the anticipation / context-relevance / tool-workflow refinements:

1. GOAL ANTICIPATION — the "WHAT LIKELY MATTERS NOW" slice appears in
   the VM's system prompt when a P1 task is due soon, and is absent
   otherwise.  Unit coverage for the inbox-match and brief signals.
2. CONTEXT RELEVANCE — _memory_recall ranks beads against the current
   message (0.6*lexical + 0.4*salience); irrelevant high-salience
   beads stay out of the prompt.
3. TOOL-USE WORKFLOWS — a taught routine that lexically matches the
   message is injected as a TAUGHT ROUTINE block, and the
   `run_routine` tool fetches its steps explicitly.
4. WORKFLOW SURFACING — a turn that used >=2 tools emits exactly one
   `workflow_summary` artifact with the ordered tool chain.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from agent.omnigent import MockExecutor, ToolCallRequest
from agent.vm import make_vm


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("ALEX_ENV", "dev")
    from alex import config, testing
    config._settings = None
    testing._fakes = None
    yield
    config._settings = None
    testing._fakes = None


USER = "alice@x.com"

TASK_COLS = ["task_id", "owner_email", "title", "status", "priority", "due_at"]
MIRROR_COLS = [
    "owner_email", "kind", "content", "salience", "source_ref",
    "created_at", "superseded_by",
]


def _iso(hours_from_now: float) -> str:
    # UTC — these represent DB rows, which anticipation compares against a
    # UTC reference clock (datetime.utcnow()).
    return (datetime.utcnow() + timedelta(hours=hours_from_now)).isoformat(sep=" ")


# ---------------------------------------------------------------------------
# 1. Goal anticipation
# ---------------------------------------------------------------------------


def test_anticipation_slice_appears_for_due_soon_p1_task(env):
    from alex.testing import get_fakes

    wh = get_fakes().warehouse
    wh.register(
        "media.alex.tasks",
        TASK_COLS,
        rows=[("t-1", USER, "Approve Q1 budget", "open", 1, _iso(6))],
    )
    vm = make_vm(session_id="s-ant", user_email=USER, executor=MockExecutor())
    prompt = vm._build_system_prompt("good morning")
    assert "WHAT LIKELY MATTERS NOW" in prompt
    assert "P1 task due soon: Approve Q1 budget" in prompt
    assert "otherwise ignore silently" in prompt


def test_anticipation_slice_absent_without_urgent_signals(env):
    from alex.testing import get_fakes

    wh = get_fakes().warehouse
    # Only a low-priority task, far in the future: no signal.
    wh.register(
        "media.alex.tasks",
        TASK_COLS,
        rows=[("t-9", USER, "Tidy the wiki", "open", 3, _iso(24 * 30))],
    )
    vm = make_vm(session_id="s-quiet", user_email=USER, executor=MockExecutor())
    prompt = vm._build_system_prompt("good morning")
    assert "WHAT LIKELY MATTERS NOW" not in prompt


def test_anticipation_overdue_inbox_and_brief_signals(env):
    from agent.anticipation import anticipation_block

    tables = {
        "media.alex.tasks": [
            {"task_id": "t-1", "owner_email": USER, "title": "Approve Q1 budget",
             "status": "open", "priority": 1, "due_at": _iso(-3)},
        ],
        "media.alex.inbox_view": [
            {"email_id": "e-1", "owner_email": USER, "priority": 1,
             "subject": "Q1 budget sign-off needed"},
            {"email_id": "e-2", "owner_email": USER, "priority": 3,
             "subject": "Team lunch"},
        ],
        "media.control.standing_order_briefs": [
            {"order_id": "o-1", "user_email": USER,
             "compiled_context": "Monday revenue brief", "consumed": False},
        ],
    }

    def sql(stmt, params=None):
        for name, rows in tables.items():
            if name in stmt:
                return rows
        return []

    block = anticipation_block(sql, USER)
    assert "OVERDUE P1 task: Approve Q1 budget" in block
    assert "P1 email matches an open task: Q1 budget sign-off needed" in block
    assert "Team lunch" not in block
    assert "Unconsumed standing-order brief: Monday revenue brief" in block
    # Compact: header + <=400 chars of signals + instruction.
    body = block.splitlines()[1:-1]
    assert sum(len(line) + 1 for line in body) <= 401


def test_anticipation_fails_soft(env):
    from agent.anticipation import anticipation_block

    def broken_sql(stmt, params=None):
        raise RuntimeError("warehouse down")

    assert anticipation_block(broken_sql, USER) == ""
    assert anticipation_block(lambda s, p=None: [], "") == ""


# ---------------------------------------------------------------------------
# 2. Context relevance: message-aware memory recall
# ---------------------------------------------------------------------------


def test_memory_recall_ranks_by_relevance_not_just_salience(env):
    from agent import vector_memory
    from agent.memory import connection
    from agent.vm import _memory_recall

    with connection() as conn:
        relevant = vector_memory.store(
            conn, USER, "preference",
            "Run the Databricks warehouse migration on weekends only.",
        )
        irrelevant = vector_memory.store(
            conn, USER, "fact", "Loves pineapple pizza on Fridays.",
        )
    irrelevant.salience = 0.95  # high salience, zero relevance
    relevant.salience = 0.5

    out = _memory_recall(USER, limit=5, query="plan the databricks warehouse migration")
    contents = [c for _, c in out]
    assert any("Databricks warehouse migration" in c for c in contents)
    assert not any("pineapple" in c for c in contents)


def test_memory_recall_without_query_falls_back_to_salience(env):
    from agent import vector_memory
    from agent.memory import connection
    from agent.vm import _memory_recall

    with connection() as conn:
        bead = vector_memory.store(conn, USER, "fact", "Weekly sync is Mondays 9am.")
    bead.salience = 0.9
    out = _memory_recall(USER, limit=5)
    assert ("fact", "Weekly sync is Mondays 9am.") in out


def test_context_engine_passes_message_to_memory_recall(env):
    from agent.context_engine import load_context

    seen: dict = {}

    def recall(user_email, limit=5, query=""):
        seen["query"] = query
        return [("preference", "relevant bead")]

    ctx = load_context(
        system_prompt="base",
        user_email=USER,
        session_id="s1",
        last_user_msg="what about the budget?",
        sql=lambda *a, **k: [],
        memory_recall=recall,
        kg_query=lambda *a, **k: [],
    )
    assert seen["query"] == "what about the budget?"
    assert "relevant bead" in ctx["T3"]


# ---------------------------------------------------------------------------
# 3. Taught routines guide execution
# ---------------------------------------------------------------------------

ROUTINE = (
    "Expense report routine: 1. open the finance portal with the browser. "
    "2. upload the receipts. 3. submit for approval."
)


def _seed_routine(wh):
    wh.register(
        "media.control.vector_memory_mirror",
        MIRROR_COLS,
        rows=[(USER, "taught_routine", ROUTINE, 0.6, "taught-r1", _iso(-100), None)],
    )


def test_matching_message_injects_taught_routine_block(env):
    from alex.testing import get_fakes

    _seed_routine(get_fakes().warehouse)
    vm = make_vm(session_id="s-rt", user_email=USER, executor=MockExecutor())
    prompt = vm._build_system_prompt("how do I file my expense report?")
    assert "TAUGHT ROUTINE" in prompt
    assert "open the finance portal" in prompt


def test_non_matching_message_has_no_routine_block(env):
    from alex.testing import get_fakes

    _seed_routine(get_fakes().warehouse)
    vm = make_vm(session_id="s-rt2", user_email=USER, executor=MockExecutor())
    prompt = vm._build_system_prompt("what's the weather like today?")
    assert "TAUGHT ROUTINE" not in prompt


def test_run_routine_tool_returns_matching_steps(env):
    from alex.testing import get_fakes

    _seed_routine(get_fakes().warehouse)
    vm = make_vm(session_id="s-rt3", user_email=USER, executor=MockExecutor())
    res = vm._invoke_tool(
        ToolCallRequest(name="run_routine", args={"name_or_query": "expense report"})
    )
    assert res["ok"] is True
    assert res["routine_id"] == "r1"
    assert "upload the receipts" in res["steps"]


def test_run_routine_tool_registered_with_learning_capability(env):
    from agent.guardrails import TOOL_CAPABILITY
    from agent.tools.registry import TOOL_REGISTRY

    assert TOOL_REGISTRY["run_routine"].capability == "learning.read"
    assert TOOL_CAPABILITY["run_routine"] == "learning.read"


def test_run_routine_no_match(env):
    from agent.tools.impls import run_routine

    res = run_routine("completely unrelated request", user_email=USER)
    assert res["ok"] is False


# ---------------------------------------------------------------------------
# 4. Workflow surfacing: one summary artifact per multi-tool turn
# ---------------------------------------------------------------------------


def test_two_tool_turn_emits_one_workflow_summary(env):
    ex = MockExecutor()
    ex.enqueue_tool_call("recall", {"query": "budget"})
    ex.enqueue_tool_call("list_my_tasks", {})
    ex.enqueue_response("Here's your morning summary.")
    vm = make_vm(session_id="s-wf", user_email=USER, executor=ex)
    seen: list = []
    vm.subscribe_sse(seen.append)
    last, _ = vm.run_turn(text="give me my morning summary")
    assert "morning summary" in last
    summaries = [
        e for e in seen if e.type == "artifact" and e.kind == "workflow_summary"
    ]
    assert len(summaries) == 1
    wf = summaries[0].extras["workflow"]
    assert wf["tools"] == ["recall", "list_my_tasks"]
    assert [s["outcome"] for s in wf["steps"]] == ["ok", "ok"]
    assert "recall:ok -> list_my_tasks:ok" == wf["chain"]


def test_single_tool_turn_emits_no_workflow_summary(env):
    ex = MockExecutor()
    ex.enqueue_tool_call("list_my_tasks", {}, follow_up_response="done")
    vm = make_vm(session_id="s-wf1", user_email=USER, executor=ex)
    seen: list = []
    vm.subscribe_sse(seen.append)
    vm.run_turn(text="list my tasks")
    assert not [
        e for e in seen if e.type == "artifact" and e.kind == "workflow_summary"
    ]


def test_context_trim_preserves_directives_and_relevant_memory(monkeypatch):
    """Under a tight budget, session state (T2) is trimmed before
    message-relevant memories (T3) and standing-order directives (T1)."""
    from agent import context_engine as ce

    slices = {
        "T0": "x" * 500,          # persona/constitution — never trimmed
        "T1": "DIRECTIVE " * 200,  # standing orders — trim last
        "T2": "session " * 400,    # session/tasks — trim first
        "T3": "MEMORY " * 200,     # message-relevant recall — trim after T2
    }
    # Simulate the global trim pass with a tight budget.
    monkeypatch.setenv("ALEX_CONTEXT_BUDGET_CHARS", "1600")
    budget = ce._total_budget()
    for key in ("T2", "T3", "T1"):
        total = sum(len(s) for s in slices.values())
        if total <= budget:
            break
        overflow = total - budget
        keep = max(0, len(slices[key]) - overflow)
        slices[key] = ce._trim(slices[key], keep) if keep >= 40 else ""
    # T2 was hit first; T1 (directives) survived larger than T2.
    assert len(slices["T1"]) >= len(slices["T2"])
    assert len(slices["T0"]) == 500


def test_anticipation_windowing_is_deterministic_with_explicit_now():
    # Pass an explicit reference clock so the overdue/due-soon windows are
    # independent of the machine's timezone. Proves the UTC-based math.
    from agent.anticipation import anticipation_block

    now = datetime(2026, 7, 20, 12, 0, 0)

    def _sql(stmt, params=None):
        if "media.alex.tasks" in stmt:
            return [
                {"task_id": "t-over", "owner_email": USER, "title": "Overdue thing",
                 "status": "open", "priority": 1,
                 "due_at": (now - timedelta(hours=5)).isoformat(sep=" ")},
                {"task_id": "t-soon", "owner_email": USER, "title": "Soon thing",
                 "status": "open", "priority": 2,
                 "due_at": (now + timedelta(hours=10)).isoformat(sep=" ")},
                {"task_id": "t-far", "owner_email": USER, "title": "Far thing",
                 "status": "open", "priority": 1,
                 "due_at": (now + timedelta(hours=200)).isoformat(sep=" ")},
            ]
        return []

    block = anticipation_block(_sql, USER, now=now)
    assert "OVERDUE P1 task: Overdue thing" in block
    assert "P2 task due soon: Soon thing" in block
    assert "Far thing" not in block  # outside the 48h window
