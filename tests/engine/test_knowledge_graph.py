"""The knowledge-graph read path — one graph, everywhere.

`agent/knowledge_graph.py` (the legacy per-user Lakebase reader) is
gone. It was the older of two graphs: raw substring matching over a
table the dev environment could not even reach, which meant the context
engine's KG channel was silently empty in dev, and edges written by the
`add_edge` tool went only to the Transparency mirror — invisible to
graph_query, retrieval expansion and context assembly, all of which
read the modern graph. These tests hold every consumer to
:mod:`agent.enterprise_graph`.
"""

from __future__ import annotations

import pytest

from agent import enterprise_graph as eg


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("ALEX_ENV", "dev")
    from alex import config, testing

    config._settings = None
    testing._fakes = None
    yield
    config._settings = None
    testing._fakes = None


USER = "demo@alex.dev"


def test_the_legacy_module_is_gone():
    with pytest.raises(ImportError):
        import agent.knowledge_graph  # noqa: F401


def test_the_context_engine_adapter_reads_the_modern_graph(env):
    from agent.vm import _kg_query

    eg.upsert_entity(None, "Project Vermeil", etype="project")
    eg.upsert_entity(None, "legal team", etype="team")
    eg.upsert_edge(None, "Project Vermeil", "reviewed_by", "legal team")

    edges = _kg_query(USER, "what is the status of Project Vermeil?")

    assert any(
        e.get("src") == "Project Vermeil" and e.get("dst") == "legal team"
        for e in edges
    ), "the context engine cannot see the modern graph"


def test_the_graph_query_tool_reads_the_modern_graph(env):
    from agent.tools import impls

    eg.upsert_entity(None, "Atlas rollout", etype="project")
    eg.upsert_entity(None, "billing connector", etype="system")
    eg.upsert_edge(None, "Atlas rollout", "depends_on", "billing connector")

    out = impls.graph_query("tell me about the Atlas rollout",
                            user_email=USER)

    assert out["ok"]
    assert any(e.get("dst") == "billing connector" for e in out["edges"]), (
        "the tool cannot see edges the graph holds"
    )


def test_add_edge_lands_in_the_graph_other_readers_query(env):
    """The write tool and the read tool must share one store — the old
    add_edge wrote only the Transparency mirror, so an agent could write
    an edge and then fail to find it one tool call later."""
    from agent.tools import impls

    impls.add_edge("Orion launch", "owned_by", "platform team",
                   user_email=USER)

    out = impls.graph_query("who owns the Orion launch?", user_email=USER)
    assert any(e.get("dst") == "platform team" for e in out["edges"])


def test_add_edge_still_projects_to_the_transparency_mirror(env):
    """The mirror survives as the Transparency read model — a projection,
    not a second source of truth."""
    from agent.tools import impls
    from alex.testing import get_fakes

    impls.add_edge("Orion launch", "owned_by", "platform team",
                   user_email=USER)

    rows = get_fakes().warehouse.execute(
        "SELECT src, dst FROM media.control.kg_mirror WHERE owner_email = :u",
        {"u": USER},
    ).as_dicts()
    assert any(r["src"] == "Orion launch" for r in rows)


def test_user_scoped_edges_stay_private(env):
    from agent.tools import impls

    impls.add_edge("secret initiative", "led_by", "dana",
                   user_email="dana@alex.dev")

    theirs = impls.graph_query("the secret initiative",
                               user_email="intruder@alex.dev")
    assert not any(
        e.get("src") == "secret initiative" for e in theirs["edges"]
    ), "a user-scoped edge crossed the user boundary"
