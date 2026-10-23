"""Tests for agent/enterprise_graph.py — the vector knowledge graph.

Covers semantic entity resolution, temporal edge supersession,
multi-hop traversal, scope isolation between users, the user↔enterprise
memory graph, and LLM-driven extraction.
"""

from __future__ import annotations

import json

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


@pytest.fixture
def graph(env):
    eg.upsert_entity(
        None, "Project Atlas", etype="project", summary="Q1 platform migration"
    )
    eg.upsert_entity(None, "Data Platform", etype="team")
    eg.upsert_entity(None, "Sarah Chen", etype="person")
    eg.upsert_entity(None, "Vertex API", etype="system")
    eg.upsert_edge(None, "Data Platform", "owns", "Project Atlas", confidence=0.9)
    eg.upsert_edge(None, "Sarah Chen", "part_of", "Data Platform", confidence=0.8)
    eg.upsert_edge(None, "Project Atlas", "depends_on", "Vertex API", confidence=0.7)
    return None


# ---- entities ------------------------------------------------------------


def test_upsert_entity_is_idempotent(env):
    a = eg.upsert_entity(None, "Acme Corp", etype="customer")
    b = eg.upsert_entity(None, "Acme Corp", etype="customer", summary="Key account")
    assert a.entity_id == b.entity_id
    assert b.summary == "Key account"


def test_upsert_entity_requires_a_name(env):
    with pytest.raises(ValueError):
        eg.upsert_entity(None, "   ")


def test_unknown_entity_type_falls_back_to_other(env):
    assert eg.upsert_entity(None, "Thing", etype="banana").etype == "other"


def test_get_entity_matches_case_insensitively_and_by_alias(env):
    eg.upsert_entity(None, "Finance", etype="team", aliases=["FinOps"])
    assert eg.get_entity(None, "finance") is not None
    assert eg.get_entity(None, "FINOPS") is not None
    assert eg.get_entity(None, "nope") is None


def test_resolve_entity_handles_paraphrase(graph):
    hits = eg.resolve_entity(None, "the atlas project", k=2)
    assert hits and hits[0]["name"] == "Project Atlas"


def test_resolve_entity_exact_match_wins(graph):
    hits = eg.resolve_entity(None, "Project Atlas", k=3)
    assert hits[0]["match"] == "exact" and hits[0]["similarity"] == 1.0


def test_resolve_entity_empty_query(env):
    assert eg.resolve_entity(None, "  ") == []


# ---- temporal edges ------------------------------------------------------


def test_reasserting_the_same_fact_does_not_duplicate(env):
    a = eg.upsert_edge(None, "A", "owns", "B", confidence=0.5)
    b = eg.upsert_edge(None, "A", "owns", "B", confidence=0.9)
    assert a.edge_id == b.edge_id
    assert b.confidence == 0.9


def test_conflicting_fact_supersedes_rather_than_overwrites(env):
    old = eg.upsert_edge(None, "Project Atlas", "owns", "Alice")
    new = eg.upsert_edge(None, "Project Atlas", "owns", "Bob")
    current = eg.neighbors(None, "Project Atlas", hops=1)
    assert [e["dst"] for e in current] == ["Bob"]
    # The old fact survives as history, pointing at its replacement.
    history = eg.neighbors(None, "Project Atlas", hops=1, include_history=True)
    old_row = next(e for e in history if e["edge_id"] == old.edge_id)
    assert old_row["current"] is False
    assert old_row["valid_to"] is not None
    assert eg._dev_graph().edges[old.edge_id].superseded_by == new.edge_id


def test_supersession_can_be_disabled_for_multi_valued_relations(env):
    eg.upsert_edge(None, "Alice", "works_on", "Atlas", supersede_conflicting=False)
    eg.upsert_edge(None, "Alice", "works_on", "Borealis", supersede_conflicting=False)
    dsts = {e["dst"] for e in eg.neighbors(None, "Alice", hops=1)}
    assert dsts == {"Atlas", "Borealis"}


def test_edge_requires_all_three_parts(env):
    with pytest.raises(ValueError):
        eg.upsert_edge(None, "A", "owns", "")


# ---- traversal -----------------------------------------------------------


def test_neighbors_one_hop(graph):
    edges = eg.neighbors(None, "Project Atlas", hops=1)
    assert {e["hop"] for e in edges} == {1}
    assert {(e["src"], e["dst"]) for e in edges} == {
        ("Data Platform", "Project Atlas"),
        ("Project Atlas", "Vertex API"),
    }


def test_neighbors_multi_hop_reaches_indirect_facts(graph):
    edges = eg.neighbors(None, "Project Atlas", hops=2)
    reached = {(e["src"], e["rel"], e["dst"]) for e in edges}
    # Sarah is two hops away: Atlas <- Data Platform <- Sarah.
    assert ("Sarah Chen", "part_of", "Data Platform") in reached
    hop2 = [e for e in edges if e["hop"] == 2]
    assert hop2, "no second-hop edges returned"


def test_neighbors_can_filter_by_relation(graph):
    edges = eg.neighbors(None, "Project Atlas", hops=2, rel="depends_on")
    assert {e["rel"] for e in edges} == {"depends_on"}


def test_neighbors_respects_the_limit(graph):
    assert len(eg.neighbors(None, "Project Atlas", hops=3, limit=1)) == 1


def test_neighbors_of_unknown_entity_is_empty(graph):
    assert eg.neighbors(None, "Nonexistent") == []


def test_subgraph_resolves_then_traverses(graph):
    out = eg.subgraph(None, "tell me about the atlas project")
    assert out["entities"] and out["entities"][0]["name"] == "Project Atlas"
    assert out["edges"]


# ---- scopes: user vs enterprise -----------------------------------------


def test_user_scope_is_isolated_from_other_users(env):
    eg.upsert_edge(
        None, "alice@x.com", "prefers", "morning briefings",
        scope=eg.user_scope("alice@x.com"),
    )
    assert eg.user_context(None, "alice@x.com")
    assert eg.user_context(None, "bob@x.com") == []


def test_visible_scopes_layer_user_over_enterprise(env):
    scopes = eg.visible_scopes("alice@x.com")
    assert scopes == ["enterprise", "user:alice@x.com"]


def test_link_user_creates_the_enterprise_node_and_a_private_edge(env):
    out = eg.link_user(
        None, "alice@x.com", "responsible_for", "Project Atlas", etype="project"
    )
    assert out["scope"] == "user:alice@x.com"
    # The target is created in the shared enterprise scope, so colleagues
    # see the entity without seeing Alice's relationship to it.
    assert eg.get_entity(None, "Project Atlas", scopes=["enterprise"]) is not None
    assert eg.user_context(None, "bob@x.com") == []


def test_context_block_puts_the_user_layer_first(graph):
    eg.link_user(None, "alice@x.com", "responsible_for", "Project Atlas")
    block = eg.context_block(None, "what is happening with Atlas?", user_email="alice@x.com")
    assert block.startswith("ABOUT THIS USER")
    assert "responsible_for" in block
    assert "RELATED ENTERPRISE FACTS" in block


def test_context_block_respects_char_budget(graph):
    block = eg.context_block(None, "Project Atlas", user_email="a@x.com", max_chars=50)
    assert len(block) <= 50


# ---- extraction ----------------------------------------------------------


def test_extract_and_store_reads_a_json_response(env):
    payload = {
        "entities": [{"name": "Acme", "type": "customer", "summary": "Key account"}],
        "edges": [
            {"src": "Acme", "rel": "uses", "dst": "Atlas",
             "evidence": "Acme uses Atlas", "confidence": 0.8}
        ],
    }
    out = eg.extract_and_store(
        None, "Acme uses Atlas.", llm_invoke=lambda _m: json.dumps(payload)
    )
    assert out == {"entities": 1, "edges": 1}
    assert eg.get_entity(None, "Acme").etype == "customer"
    assert eg.neighbors(None, "Acme", hops=1)[0]["rel"] == "uses"


def test_extract_tolerates_fenced_json(env):
    raw = '```json\n{"entities": [{"name": "Beta"}], "edges": []}\n```'
    assert eg.extract_and_store(None, "x", llm_invoke=lambda _m: raw)["entities"] == 1


def test_extract_survives_a_garbage_response(env):
    out = eg.extract_and_store(None, "x", llm_invoke=lambda _m: "not json at all")
    assert out == {"entities": 0, "edges": 0}


def test_extract_survives_an_llm_failure(env):
    def boom(_m):
        raise RuntimeError("model down")

    assert eg.extract_and_store(None, "x", llm_invoke=boom) == {
        "entities": 0,
        "edges": 0,
    }


def test_extract_empty_text_is_a_no_op(env):
    calls = []
    eg.extract_and_store(None, "  ", llm_invoke=lambda m: calls.append(m) or "{}")
    assert calls == [], "the model must not be called for empty text"


# ---- stats ---------------------------------------------------------------


def test_stats_counts_current_and_historical(graph):
    eg.upsert_edge(None, "Data Platform", "owns", "Something Else")
    s = eg.stats(None)
    assert s["entities"] >= 4
    assert s["historical_edges"] == 1
    assert s["entity_types"]["project"] >= 1


# ---- cache identity ---------------------------------------------------------


def test_a_reset_graph_never_serves_the_previous_graphs_adjacency():
    """reset_fakes() restarts the version counter at 0; a version-only
    cache key used to hand test B the previous graph's edges. The cache
    is stamped with a per-instance token, so a fresh graph is a fresh
    key — and its adjacency is empty, not inherited."""
    from alex.testing import reset_fakes

    eg.upsert_edge(None, "Alpha", "owns", "Beta")
    before = eg.neighbors(None, "alpha")

    reset_fakes()  # fresh graph, version restarts at 0
    after = eg.neighbors(None, "alpha")

    assert before, "precondition: the first graph was traversable"
    assert after == [], "the new graph served the old graph's edges"
