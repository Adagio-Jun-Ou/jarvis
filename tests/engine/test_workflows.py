"""Smoke tests for the workflows/ modules."""
from __future__ import annotations

import pytest


def test_claude_code_harness_state_machine():
    from workflows.claude_code_harness import (
        STATE_DONE,
        STATE_EXECUTING,
        STATE_PLANNING,
        STATE_REVIEW,
        Harness,
    )
    h = Harness("alice@x.com", llm_invoke=lambda msgs: "# Plan\n\n1. step one")
    plan = h.start_plan("draft the brief")
    assert plan.state == STATE_PLANNING
    md = h.produce_workplan(plan.plan_id, "draft the brief")
    assert "# Plan" in md
    assert plan.state == STATE_REVIEW
    h.ratify(plan.plan_id)
    assert plan.state == STATE_EXECUTING
    h.complete(plan.plan_id, ["outputs/brief.md"])
    assert plan.state == STATE_DONE
    assert plan.workflow_id.startswith("user-alice")
    assert plan.output_paths == ["outputs/brief.md"]


def test_claude_code_harness_rejects_illegal_transition():
    from workflows.claude_code_harness import Harness
    h = Harness("alice@x.com", llm_invoke=lambda msgs: "x")
    plan = h.start_plan("x")
    # We are in PLANNING. Cannot ratify() yet.
    with pytest.raises(ValueError, match="illegal transition"):
        h.ratify(plan.plan_id)


def test_claude_code_harness_abort_from_any_state():
    from workflows.claude_code_harness import (
        STATE_ABORTED,
        STATE_REVIEW,
        Harness,
    )
    h = Harness("alice@x.com", llm_invoke=lambda msgs: "x")
    plan = h.start_plan("x")
    h.produce_workplan(plan.plan_id, "x")
    assert plan.state == STATE_REVIEW
    h.abort(plan.plan_id, reason="user changed mind")
    assert plan.state == STATE_ABORTED


class _FakeRow:
    def __init__(self, d):
        self._d = dict(d)

    def asDict(self):  # noqa: N802 - mirrors pyspark Row.asDict
        return dict(self._d)

    def __getitem__(self, k):
        return self._d[k]


class _FakeRows:
    def __init__(self, data):
        self._data = data

    def collect(self):
        return self._data


class _FakeSparkFull:
    """Answers the catalog/ACL/integration/insert queries n8n makes."""

    def __init__(self, requires_dual_approval=True):
        self.requires_dual = requires_dual_approval
        self.inserted = []

    def sql(self, stmt, args=None, parameters=None):
        parameters = args if args is not None else parameters
        if "workflow_catalog" in stmt:
            return _FakeRows([_FakeRow({
                "workflow_id": "wf-1", "name": "Test WF", "description": "",
                "capability": "cap.run", "input_schema": "{}",
                "integration_id": "n8n-main",
                "requires_dual_approval": self.requires_dual, "active": True,
            })])
        if "effective_employee_acl" in stmt:
            return _FakeRows([_FakeRow({"1": 1})])
        if "tool_integrations" in stmt:
            return _FakeRows([_FakeRow({"base_url": "https://n8n.local"})])
        if "INSERT" in stmt:
            self.inserted.append(parameters)
            return _FakeRows([])
        return _FakeRows([])


@pytest.fixture
def n8n(monkeypatch):
    from workflows import n8n_integration as n

    spark = _FakeSparkFull()
    monkeypatch.setattr(n, "_spark", lambda: spark)
    monkeypatch.setattr(n, "_webhook_token", lambda: "tok-secret")
    n._staged_tokens.clear()

    class _Resp:
        status = 200

        def read(self):
            return b"{}"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(n.urllib.request, "urlopen", lambda req, timeout=0: _Resp())
    return n, spark


def test_n8n_stage_persists_token_and_execute_accepts_it(n8n):
    n, spark = n8n
    r = n.stage("alice@x.com", "wf-1", {"a": 1})
    assert r.get("staged") is True
    token = r["token"]
    out = n.execute("alice@x.com", "wf-1", {"a": 1}, confirm_token=token)
    assert out["status"] == "ok"
    assert spark.inserted, "audit row must be written"


def test_n8n_execute_enforces_capability(n8n, monkeypatch):
    # execute() must re-check capability (not only stage()): a direct call by
    # a user lacking the capability must be denied and must NOT fire.
    n, spark = n8n
    monkeypatch.setattr(n, "_user_has_capability", lambda u, c: False)
    out = n.execute("mallory@x.com", "wf-1", {"a": 1}, confirm_token=None)
    assert "error" in out and "capability" in out["error"]
    assert not spark.inserted, "denied execute must not write an audit/fire row"


def test_n8n_execute_rejects_auto_approved_sentinel(n8n):
    n, _ = n8n
    n.stage("alice@x.com", "wf-1", {"a": 1})
    out = n.execute("alice@x.com", "wf-1", {"a": 1}, confirm_token="auto-approved")
    assert "error" in out


def test_n8n_execute_rejects_arbitrary_long_token(n8n):
    n, _ = n8n
    n.stage("alice@x.com", "wf-1", {"a": 1})
    out = n.execute("alice@x.com", "wf-1", {"a": 1},
                    confirm_token="x" * 40)
    assert "error" in out


def test_n8n_token_is_single_use(n8n):
    n, _ = n8n
    token = n.stage("alice@x.com", "wf-1", {"a": 1})["token"]
    assert n.execute("alice@x.com", "wf-1", {"a": 1},
                     confirm_token=token)["status"] == "ok"
    out = n.execute("alice@x.com", "wf-1", {"a": 1}, confirm_token=token)
    assert "error" in out


def test_n8n_token_bound_to_payload_and_user(n8n):
    n, _ = n8n
    token = n.stage("alice@x.com", "wf-1", {"a": 1})["token"]
    # Different payload -> reject.
    assert "error" in n.execute("alice@x.com", "wf-1", {"a": 2},
                                confirm_token=token)
    # Different user -> reject.
    assert "error" in n.execute("mallory@x.com", "wf-1", {"a": 1},
                                confirm_token=token)


def test_n8n_token_expires(n8n):
    n, _ = n8n
    token = n.stage("alice@x.com", "wf-1", {"a": 1})["token"]
    key = ("alice@x.com", "wf-1", n._payload_hash({"a": 1}))
    tok, _exp = n._staged_tokens[key]
    n._staged_tokens[key] = (tok, 1.0)  # long past
    assert "error" in n.execute("alice@x.com", "wf-1", {"a": 1},
                                confirm_token=token)


def test_n8n_non_dual_approval_executes_without_token(n8n):
    n, spark = n8n
    spark.requires_dual = False
    out = n.stage("alice@x.com", "wf-1", {"a": 1})
    assert out["status"] == "ok"


def test_n8n_stage_unknown_workflow():
    """A stage() call for a workflow that doesn't exist returns an
    error, not a token. We patch the Spark session to a stub so the
    test runs without PySpark (the production runtime has it)."""
    from workflows import n8n_integration as n
    class _FakeRows:
        def __init__(self, data): self._data = data
        def collect(self): return self._data
    class _FakeSpark:
        def sql(self, stmt, args=None, parameters=None):
            return _FakeRows([])
    n._spark = lambda: _FakeSpark()
    r = n.stage("alice@x.com", "no-such-workflow", {})
    assert "error" in r
    assert "staged" not in r
