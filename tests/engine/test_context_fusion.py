"""Tests for agent/context_fusion.py.

Fusion's job is to turn three independent lookups into one
subject-organised context. These tests pin that: anchors widen every
channel, evidence groups under the subject it concerns, the active task
frames the turn, and any single channel failing degrades the context
instead of failing the turn.
"""

from __future__ import annotations

import pytest

from agent import context_fusion as cf


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("ALEX_ENV", "dev")
    from alex import config, testing

    config._settings = None
    testing._fakes = None
    yield
    config._settings = None
    testing._fakes = None


def retrieve_of(*docs):
    def _retrieve(_user, _query):
        return [
            {
                "text": text,
                "score": score,
                "citation": {"title": title, "source_ref": f"wiki/{title.lower()}"},
            }
            for title, text, score in docs
        ]

    return _retrieve


# ---- anchoring -----------------------------------------------------------


def test_anchors_resolve_from_the_graph(env):
    from agent import enterprise_graph as eg

    eg.upsert_entity(None, "Project Atlas", etype="project")
    anchors = cf.resolve_anchors(None, "how is the atlas project going?")
    assert "Project Atlas" in anchors


def test_caller_supplied_anchors_come_first_and_are_kept(env):
    from agent import enterprise_graph as eg

    eg.upsert_entity(None, "Project Atlas", etype="project")
    anchors = cf.resolve_anchors(
        None, "atlas", extra=["Draft the risk register"]
    )
    assert anchors[0] == "Draft the risk register"


def test_anchor_resolution_survives_an_empty_graph(env):
    assert cf.resolve_anchors(None, "anything at all") == []


# ---- fusion --------------------------------------------------------------


def test_evidence_groups_under_its_subject(env):
    from agent import enterprise_graph as eg

    eg.upsert_entity(None, "Project Atlas", etype="project")

    ctx = cf.fuse(
        query="what is happening with Project Atlas?",
        user_email="alice@x.com",
        retrieve=retrieve_of(("Atlas Brief", "Project Atlas is the Q1 migration.", 0.9)),
        kg_query=lambda _u, _q: [
            {"src": "Project Atlas", "rel": "owned_by", "dst": "Data Platform",
             "confidence": 0.9}
        ],
        memory_recall=lambda _u, limit=5, query="": [
            ("pref", "Alice prefers Project Atlas updates on Mondays")
        ],
    )
    assert ctx.anchors == ["Project Atlas"]
    # All three channels anchored on the same subject.
    assert {e.channel for e in ctx.evidence} == {"knowledge", "graph", "memory"}
    assert all(e.anchor == "Project Atlas" for e in ctx.evidence)

    block = cf.render(ctx)
    assert "WHAT WE KNOW ABOUT PROJECT ATLAS:" in block
    assert "[1] Atlas Brief" in block
    assert "fact: Project Atlas owned_by Data Platform" in block
    assert "about you: Alice prefers" in block


def test_unanchored_evidence_is_separated(env):
    ctx = cf.fuse(
        query="unrelated question",
        user_email="a@x.com",
        retrieve=retrieve_of(("Expenses", "Book travel 14 days ahead.", 0.4)),
    )
    assert ctx.evidence[0].anchor == ""
    assert "ALSO POSSIBLY RELEVANT:" in cf.render(ctx)


def test_anchored_evidence_outranks_a_better_scoring_stranger(env):
    from agent import enterprise_graph as eg

    eg.upsert_entity(None, "Northwind", etype="customer")
    ctx = cf.fuse(
        query="Northwind renewal",
        user_email="a@x.com",
        retrieve=retrieve_of(
            ("Unrelated", "Something with a high score.", 0.99),
            ("Northwind Account", "Northwind renews in June.", 0.10),
        ),
    )
    assert ctx.evidence[0].citation["title"] == "Northwind Account"


def test_anchors_widen_the_query_given_to_each_channel(env):
    """A channel must receive the entity name as the graph knows it, not
    only the words the user typed."""
    from agent import enterprise_graph as eg

    eg.upsert_entity(None, "Project Atlas", etype="project")
    seen = {}

    def retrieve(_user, query):
        seen["query"] = query
        return []

    cf.fuse(
        query="the atlas project",
        user_email="a@x.com",
        retrieve=retrieve,
    )
    assert "Project Atlas" in seen["query"]


def test_a_failing_channel_does_not_fail_the_turn(env):
    def boom(*_a, **_k):
        raise RuntimeError("store down")

    ctx = cf.fuse(
        query="anything",
        user_email="a@x.com",
        retrieve=boom,
        kg_query=boom,
        memory_recall=boom,
    )
    assert ctx.evidence == []
    assert cf.render(ctx) == ""


def test_malformed_evidence_is_dropped(env):
    ctx = cf.fuse(
        query="x",
        user_email="a@x.com",
        retrieve=lambda _u, _q: ["not-a-dict", {"text": "  "}],
        kg_query=lambda _u, _q: [42, {"src": "", "rel": "", "dst": ""}],
    )
    assert ctx.evidence == []


def test_render_respects_its_budget(env):
    ctx = cf.fuse(
        query="x",
        user_email="a@x.com",
        retrieve=retrieve_of(*[(f"Doc{i}", "body " * 200, 0.5) for i in range(6)]),
    )
    assert len(cf.render(ctx, max_chars=400)) < 1600


def test_render_of_empty_context_is_empty():
    assert cf.render(cf.FusedContext()) == ""


def test_to_dict_reports_counts(env):
    ctx = cf.fuse(
        query="x",
        user_email="a@x.com",
        retrieve=retrieve_of(("D", "text", 0.5)),
        kg_query=lambda _u, _q: [{"src": "A", "rel": "r", "dst": "B"}],
    )
    counts = ctx.to_dict()["counts"]
    assert counts["knowledge"] == 1 and counts["graph"] == 1


# ---- the active task -----------------------------------------------------


class P:
    def __init__(self, email="alice@x.com", is_admin=False):
        self.email = email
        self.is_admin = is_admin


@pytest.fixture
def board(env):
    from alex.testing import get_fakes

    wh = get_fakes().warehouse
    wh.register(
        "media.alex.tasks",
        ["id", "owner_email", "assignee_email", "title", "description",
         "status", "priority", "due_at", "source", "source_ref",
         "created_at", "updated_at", "closed_at"],
        pk=["id"],
    )
    wh.register("media.control.task_crew", ["task_id", "member", "role", "added_at"])
    wh.register(
        "media.control.task_messages",
        ["message_id", "task_id", "author", "body", "kind", "ts"],
        pk=["message_id"],
    )
    wh.register(
        "media.control.task_runs",
        ["run_id", "task_id", "session_id", "started_by", "status",
         "started_at", "summary"],
        pk=["run_id"],
    )

    def _sql(stmt, params=None):
        res = wh.execute(stmt, params or {})
        as_dicts = getattr(res, "as_dicts", None)
        return as_dicts() if callable(as_dicts) else list(res or [])

    return _sql


def test_a_chat_session_has_no_active_task(board):
    assert cf.active_task_for(board, "default", P()) is None


def test_a_task_session_resolves_its_task(board):
    from agent import taskboard

    tid = taskboard.create_task(
        board, P(), title="Draft the risk register", description="Pull the risks."
    )["task_id"]
    taskboard.post_message(board, tid, author="alice@x.com", body="focus on billing")

    task = cf.active_task_for(board, f"task-{tid}", P())
    assert task is not None
    assert task["title"] == "Draft the risk register"
    # System chatter is excluded; human discussion is carried.
    assert [m["body"] for m in task["recent_thread"]] == ["focus on billing"]


def test_someone_not_on_the_task_gets_nothing(board):
    from agent import taskboard

    tid = taskboard.create_task(board, P("alice@x.com"), title="Private")["task_id"]
    assert cf.active_task_for(board, f"task-{tid}", P("mallory@x.com")) is None


def test_the_active_task_frames_the_turn(board):
    from agent import taskboard

    tid = taskboard.create_task(
        board, P(), title="Draft the risk register", description="Pull the risks."
    )["task_id"]

    ctx = cf.fuse(
        query="where should I start?",
        user_email="alice@x.com",
        session_id=f"task-{tid}",
        sql=board,
        principal=P(),
    )
    assert ctx.active_task is not None
    # The task's own title seeds the anchors — known context, not a guess.
    assert "Draft the risk register" in ctx.anchors

    block = cf.render(ctx)
    assert block.startswith("THE TASK YOU ARE WORKING")
    assert "Draft the risk register" in block
    assert "Pull the risks." in block


def test_task_block_of_nothing_is_empty():
    assert cf.task_block({}) == ""
