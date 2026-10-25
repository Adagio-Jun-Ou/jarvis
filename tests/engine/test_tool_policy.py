"""The risk-graded permission layer: allow / ask / deny per tool.

Properties pinned:

* Defaults are grounded in existing governance: the dual-approval
  capabilities (install/shell/download) and the outward acts
  (email.send, publish.post) default to ASK; pip_install stays ALLOW
  because its curated allowlist is already the control.
* Operator rules in workspace_policy override defaults; an invalid
  value is ignored (a typo must neither widen nor brick).
* DENY refuses everywhere, with no route-around guidance.
* ASK on an unattended board run parks on a crew approve/deny and
  FAILS CLOSED: deny and timeout both refuse, and a timed-out approval
  window is cancelled so a stale approve cannot land later.
* ASK on a supervised session proceeds but is said out loud (one
  governor warn per tool).
"""

from __future__ import annotations

import threading
import time

import pytest

from agent import task_questions as tq
from agent import tool_policy
from agent.omnigent import MockExecutor
from agent.omnigent.executor import ToolCallRequest
from agent.vm import make_vm


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("ALEX_ENV", "dev")
    monkeypatch.setenv("ALEX_ASK_POLL_S", "0.05")
    from alex import config, testing

    config._settings = None
    testing._fakes = None
    tool_policy.reset_cache()
    from alex.testing import get_fakes

    get_fakes().warehouse.register(
        "media.control.workspace_policy",
        ["name", "value", "rationale", "set_by", "set_at"],
        pk=["name"],
    )
    yield
    config._settings = None
    testing._fakes = None
    tool_policy.reset_cache()


def _set_rule(name: str, value: str) -> None:
    from agent.tools.impls import _sql

    _sql(
        "INSERT INTO media.control.workspace_policy "
        "(name, value, rationale, set_by, set_at) "
        "VALUES (:n, :v, 'test', 'op@alex.dev', current_timestamp())",
        {"n": f"tool_policy.{name}", "v": value},
    )
    tool_policy.reset_cache()


# ---- decisions -------------------------------------------------------------


def test_defaults_ground_in_the_dual_approval_catalog(env):
    assert tool_policy.decision("system_install", "system.install") == "ask"
    assert tool_policy.decision("download_file", "compute.download") == "ask"
    assert tool_policy.decision("view", "files.read") == "allow"
    assert tool_policy.decision("pip_install", "compute.install") == "allow", (
        "the curated library allowlist is already pip's human gate"
    )


def test_operator_rules_override_and_typos_are_ignored(env):
    _set_rule("system_install", "allow")
    assert tool_policy.decision("system_install", "system.install") == "allow"
    _set_rule("view", "deny")
    assert tool_policy.decision("view", "files.read") == "deny"
    _set_rule("download_file", "yolo")  # invalid → default stands
    assert tool_policy.decision("download_file", "compute.download") == "ask"


# ---- enforcement in the VM -------------------------------------------------


def _events_of(vm):
    seen = []
    vm.subscribe_sse(lambda ev: seen.append(ev))
    return seen


def test_deny_refuses_with_no_route_around(env):
    _set_rule("view", "deny")
    vm = make_vm(session_id="chat-pol", user_email="dana@alex.dev",
                 executor=MockExecutor())
    res = vm._invoke_tool(ToolCallRequest(name="view", args={"path": "x"}))
    assert res["ok"] is False
    assert "workspace policy" in res["error"]
    assert "another route" in res["error"]
    vm.close()


def test_board_ask_parks_and_an_approve_lets_it_through(env):
    from alex.testing import get_fakes
    from app._dev_seed import TASK_MESSAGES_COLS, TASK_QUESTIONS_COLS

    wh = get_fakes().warehouse
    wh.register("media.control.task_messages", TASK_MESSAGES_COLS,
                pk=["message_id"])
    wh.register("media.control.task_questions", TASK_QUESTIONS_COLS,
                pk=["question_id"])
    _set_rule("view", "ask")

    def sql(stmt, params=None):
        res = wh.execute(stmt, params or {})
        as_dicts = getattr(res, "as_dicts", None)
        return as_dicts() if callable(as_dicts) else list(res or [])

    class P:
        email = "dana@alex.dev"
        is_admin = False

    def _approve_soon():
        deadline = time.time() + 3
        while time.time() < deadline:
            open_qs = tq.open_for_task(sql, "t-pol")
            if open_qs:
                tq.answer(sql, P(), "t-pol", open_qs[0]["question_id"],
                          "approve")
                return
            time.sleep(0.02)

    vm = make_vm(session_id="task-t-pol", user_email="dana@alex.dev",
                 executor=MockExecutor())
    t = threading.Thread(target=_approve_soon, daemon=True)
    t.start()
    res = vm._invoke_tool(ToolCallRequest(name="view", args={"path": "x"}))
    t.join(timeout=3)
    # Approved: whatever view returned, it is NOT the policy refusal.
    text = str(res)
    assert "not approved" not in text and "workspace policy" not in text
    qs = tq.for_task(sql, "t-pol")
    assert qs and qs[0]["status"] == "answered"
    vm.close()


def test_board_ask_fails_closed_on_deny_and_on_timeout(env, monkeypatch):
    from alex.testing import get_fakes
    from app._dev_seed import TASK_MESSAGES_COLS, TASK_QUESTIONS_COLS

    wh = get_fakes().warehouse
    wh.register("media.control.task_messages", TASK_MESSAGES_COLS,
                pk=["message_id"])
    wh.register("media.control.task_questions", TASK_QUESTIONS_COLS,
                pk=["question_id"])
    _set_rule("view", "ask")

    def sql(stmt, params=None):
        res = wh.execute(stmt, params or {})
        as_dicts = getattr(res, "as_dicts", None)
        return as_dicts() if callable(as_dicts) else list(res or [])

    class P:
        email = "dana@alex.dev"
        is_admin = False

    # Denied by a person.
    def _deny_soon():
        deadline = time.time() + 3
        while time.time() < deadline:
            open_qs = tq.open_for_task(sql, "t-no")
            if open_qs:
                tq.answer(sql, P(), "t-no", open_qs[0]["question_id"], "deny")
                return
            time.sleep(0.02)

    vm = make_vm(session_id="task-t-no", user_email="dana@alex.dev",
                 executor=MockExecutor())
    t = threading.Thread(target=_deny_soon, daemon=True)
    t.start()
    res = vm._invoke_tool(ToolCallRequest(name="view", args={"path": "x"}))
    t.join(timeout=3)
    assert res["ok"] is False and "said no" in res["error"]
    vm.close()

    # Nobody answered: fail closed AND close the window.
    monkeypatch.setenv("ALEX_APPROVE_WAIT_S", "0.2")
    vm2 = make_vm(session_id="task-t-quiet", user_email="dana@alex.dev",
                  executor=MockExecutor())
    res2 = vm2._invoke_tool(ToolCallRequest(name="view", args={"path": "x"}))
    assert res2["ok"] is False and "no approval" in res2["error"]
    qs = tq.for_task(sql, "t-quiet")
    assert qs and qs[-1]["status"] == "cancelled", (
        "a timed-out approval window must close — a stale approve later "
        "would be an authorization nobody meant to give"
    )
    vm2.close()


def test_supervised_ask_proceeds_but_is_said_once(env):
    _set_rule("view", "ask")
    vm = make_vm(session_id="chat-say", user_email="dana@alex.dev",
                 executor=MockExecutor())
    seen = _events_of(vm)
    r1 = vm._invoke_tool(ToolCallRequest(name="view", args={"path": "x"}))
    vm._invoke_tool(ToolCallRequest(name="view", args={"path": "y"}))
    assert "workspace policy" not in str(r1)
    warns = [
        e for e in seen
        if getattr(e, "type", "") == "governor"
        and getattr(e, "pathology", "") == "policy"
    ]
    assert len(warns) == 1, "one announcement per tool, not per call"
    vm.close()
