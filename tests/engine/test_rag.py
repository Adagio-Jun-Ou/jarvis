"""Tests for agent/rag.py — the enterprise RAG engine.

Covers structure-aware chunking, ingestion/replacement, hybrid
retrieval (dense + BM25 + graph expansion), ACL/scope enforcement, and
citation fidelity.
"""

from __future__ import annotations

import pytest

from agent import enterprise_graph as eg
from agent import rag

BRIEF = """# Project Atlas

Project Atlas is the Q1 platform migration owned by the Data Platform team.
The delivery deadline is March 31 and the approved budget is 1.2M USD.

## Risks

The main risk is the legacy billing connector, which depends on the Vertex API.
"""

HANDBOOK = """# Expenses policy

Employees may expense travel booked at least fourteen days in advance.
Receipts above 75 USD require manager approval before reimbursement.
"""


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
def corpus(env):
    """A small enterprise corpus. ``env`` (conftest) puts us in dev mode
    with a fresh fakes object per test."""
    rag.ingest(None, title="Project Atlas Brief", text=BRIEF, source_ref="wiki/atlas")
    rag.ingest(None, title="Expenses Policy", text=HANDBOOK, source_ref="wiki/expenses")
    return None


# ---- chunking ------------------------------------------------------------


def test_chunk_text_keeps_short_docs_whole():
    chunks = rag.chunk_text(BRIEF)
    assert len(chunks) == 1
    assert "Project Atlas" in chunks[0]


def test_chunk_text_splits_long_docs_with_headings():
    body = "\n\n".join(
        f"## Section {i}\n\n" + ("lorem ipsum dolor sit amet " * 40) for i in range(4)
    )
    chunks = rag.chunk_text(body)
    assert len(chunks) > 3
    # Every chunk carries a heading so a retrieved fragment stays legible.
    assert all(c.lstrip().startswith("## Section") for c in chunks)
    # Headings appear in document order, never ahead of their content.
    seen = [int(c.split("## Section ")[1][0]) for c in chunks]
    assert seen == sorted(seen)


def test_chunk_text_empty_is_empty():
    assert rag.chunk_text("") == []
    assert rag.chunk_text("   \n  ") == []


# ---- ingestion -----------------------------------------------------------


def test_ingest_returns_summary_and_stable_doc_id(env):
    a = rag.ingest(None, title="T", text=BRIEF, source_ref="wiki/x")
    b = rag.ingest(None, title="T", text=BRIEF, source_ref="wiki/x")
    assert a["doc_id"] == b["doc_id"], "same source must map to the same doc"
    assert a["chunks"] >= 1


def test_reingest_replaces_rather_than_duplicates(env):
    rag.ingest(None, title="T", text=BRIEF, source_ref="wiki/x")
    before = len(rag._dev_corpus().all())
    rag.ingest(None, title="T", text=BRIEF, source_ref="wiki/x")
    assert len(rag._dev_corpus().all()) == before


def test_ingest_empty_text_is_a_no_op(env):
    out = rag.ingest(None, title="T", text="   ", source_ref="wiki/e")
    assert out["chunks"] == 0 and out["skipped"] == "empty"


def test_delete_document_removes_all_chunks(env):
    out = rag.ingest(None, title="T", text=BRIEF, source_ref="wiki/x")
    removed = rag.delete_document(None, out["doc_id"])
    assert removed == out["chunks"]
    assert rag._dev_corpus().all() == []


def test_list_documents_summarises_corpus(corpus):
    docs = rag.list_documents(None)
    titles = {d["title"] for d in docs}
    assert titles == {"Project Atlas Brief", "Expenses Policy"}
    assert all(d["chunks"] >= 1 and d["chars"] > 0 for d in docs)


# ---- retrieval -----------------------------------------------------------


def test_retrieve_finds_the_relevant_document(corpus):
    hits = rag.retrieve(None, "what is the Atlas budget?", k=2, use_graph=False)
    assert hits
    assert hits[0].chunk.title == "Project Atlas Brief"


def test_retrieve_discriminates_between_documents(corpus):
    hits = rag.retrieve(None, "can I expense a flight?", k=1, use_graph=False)
    assert hits and hits[0].chunk.title == "Expenses Policy"


def test_retrieve_carries_citations(corpus):
    hits = rag.retrieve(None, "Atlas deadline", k=1, use_graph=False)
    cite = hits[0].to_dict()["citation"]
    assert cite["title"] == "Project Atlas Brief"
    assert cite["source_ref"] == "wiki/atlas"
    assert cite["chunk_index"] == 0


def test_retrieve_records_which_ranker_found_it(corpus):
    hits = rag.retrieve(None, "Project Atlas budget", k=2, use_graph=False)
    why = hits[0].to_dict()["why"]
    assert why["vector_rank"] or why["lexical_rank"], "no ranker claimed the hit"


def test_retrieve_empty_query_returns_nothing(corpus):
    assert rag.retrieve(None, "   ", k=3) == []


def test_retrieve_on_empty_corpus_returns_nothing(env):
    assert rag.retrieve(None, "anything", k=3) == []


# ---- scope / ACL ---------------------------------------------------------


def test_private_scope_is_invisible_to_another_user(env):
    rag.ingest(
        None, title="Alice notes", text="Alice's private salary review notes.",
        source_ref="p/1", scope=rag.user_scope("alice@x.com"),
    )
    mine = rag.retrieve(None, "salary review", user_email="alice@x.com", k=3)
    theirs = rag.retrieve(None, "salary review", user_email="bob@x.com", k=3)
    assert mine and mine[0].chunk.title == "Alice notes"
    assert theirs == [], "another user must not retrieve a private document"


def test_capability_filter_excludes_unreadable_documents(corpus):
    # The caller holds no capabilities at all.
    assert rag.retrieve(None, "Atlas budget", k=3, capabilities=set()) == []
    ok = rag.retrieve(
        None, "Atlas budget", k=3, capabilities={rag.DEFAULT_CAPABILITY}
    )
    assert ok


# ---- graph expansion -----------------------------------------------------


def test_graph_expansion_reaches_documents_that_never_mention_the_query(env):
    rag.ingest(
        None, title="Atlas Brief", text="Project Atlas is the Q1 migration.",
        source_ref="wiki/atlas",
    )
    rag.ingest(
        None, title="Billing Runbook",
        text="The Vertex API connector is maintained here; escalate to on-call.",
        source_ref="wiki/billing",
    )
    eg.upsert_entity(None, "Project Atlas", etype="project")
    eg.upsert_entity(None, "Vertex API", etype="system")
    eg.upsert_edge(None, "Project Atlas", "depends_on", "Vertex API")

    plain = rag.retrieve(None, "Project Atlas", k=4, use_graph=False)
    expanded = rag.retrieve(
        None, "Project Atlas", k=4, use_graph=True, user_email="demo@alex.dev"
    )
    assert {h.chunk.title for h in plain} == {"Atlas Brief"}
    assert "Billing Runbook" in {h.chunk.title for h in expanded}
    runbook = next(h for h in expanded if h.chunk.title == "Billing Runbook")
    assert runbook.graph_rank is not None
    assert "Vertex API" in runbook.via_entities


# ---- prompt assembly -----------------------------------------------------


def test_context_block_numbers_passages_for_citation(corpus):
    hits = rag.retrieve(None, "Atlas budget", k=2, use_graph=False)
    block = rag.context_block(hits)
    assert block.startswith("ENTERPRISE KNOWLEDGE")
    assert "[1]" in block


def test_context_block_respects_its_char_budget(corpus):
    hits = rag.retrieve(None, "Atlas", k=2, use_graph=False)
    assert len(rag.context_block(hits, max_chars=200)) <= 220


def test_context_block_empty_results_is_empty_string():
    assert rag.context_block([]) == ""


def test_search_returns_json_serialisable_results(corpus):
    out = rag.search(None, "Atlas budget", k=2)
    assert out["count"] >= 1
    first = out["results"][0]
    assert set(first) == {"text", "score", "citation", "why"}


# ---------------------------------------------------------------- bulk ingest
# ingest_many exists because the embedding backend charges per *request*,
# not per text: a corpus loaded one document at a time pays that cost
# hundreds of thousands of times. These tests hold it to the same
# semantics as ingest(), because a faster path that stores different
# things is not a faster path.


def _docs(n: int, *, prefix: str = "bulk") -> list[dict]:
    return [
        {
            "title": f"{prefix} document {i}",
            "text": f"# Heading {i}\n\nBody of document {i} mentions widget{i}.",
            "source_ref": f"{prefix}/{i}",
        }
        for i in range(n)
    ]


def test_ingest_many_stores_same_chunks_as_ingest(env):
    """Parity: the bulk path must produce byte-identical chunks."""
    one_at_a_time = [
        rag.ingest(None, title=d["title"], text=d["text"], source_ref=d["source_ref"])
        for d in _docs(5)
    ]
    singles = {c.chunk_id: c for c in rag._dev_corpus().all()}

    from alex import testing
    testing.reset_fakes()
    bulk = rag.ingest_many(None, _docs(5))
    batched = {c.chunk_id: c for c in rag._dev_corpus().all()}

    assert bulk["documents"] == len(one_at_a_time) == 5
    assert bulk["chunks"] == sum(r["chunks"] for r in one_at_a_time)
    assert set(batched) == set(singles)
    for chunk_id, chunk in batched.items():
        other = singles[chunk_id]
        assert (chunk.text, chunk.title, chunk.doc_id) == (
            other.text, other.title, other.doc_id
        )
        assert chunk.embedding == other.embedding
        assert chunk.scope == other.scope
        assert chunk.classification == other.classification


@pytest.mark.parametrize("replace", [True, False])
def test_ingest_many_crosses_document_boundaries_in_one_call(
    env, monkeypatch, replace
):
    """The whole point: chunks from different documents share a request.

    Parametrised over ``replace`` because the replacing path is the
    default, and an earlier version flushed the buffer before every
    single document to make deletion safe — which quietly reduced the
    bulk path to one embedding request per document, exactly what it
    exists to avoid. Testing only ``replace=False`` hid that completely.
    """
    calls: list[int] = []
    real = rag._embed_many

    def counting(texts, **kw):
        calls.append(len(texts))
        return real(texts, **kw)

    monkeypatch.setattr(rag, "_embed_many", counting)
    rag.ingest_many(None, _docs(30), replace=replace)

    assert len(calls) == 1, f"expected one batched call, got {len(calls)}: {calls}"
    assert calls[0] == 30, "every document's chunks should share the call"


def test_ingest_many_flushes_at_wave_boundary(env, monkeypatch):
    """Peak memory is set by the wave size, not the corpus size."""
    monkeypatch.setattr(rag, "INGEST_WAVE", 10)
    sizes: list[int] = []
    real = rag._embed_many
    monkeypatch.setattr(
        rag, "_embed_many", lambda t, **kw: (sizes.append(len(t)), real(t, **kw))[1]
    )
    result = rag.ingest_many(None, _docs(25), replace=False)

    assert result["chunks"] == 25
    assert max(sizes) <= 10, f"a wave exceeded the cap: {sizes}"
    assert sum(sizes) == 25


def test_ingest_many_replace_does_not_delete_queued_chunks(env):
    """Regression: replace=True deletes a document's old chunks. If the
    pending buffer were not flushed first, deleting document N would take
    already-queued chunks of an *earlier* document back out again when
    they shared a doc_id — and more subtly, a document appearing twice in
    one batch must end with only its second version stored."""
    rag.ingest_many(None, _docs(3), replace=True)
    assert len({c.doc_id for c in rag._dev_corpus().all()}) == 3

    revised = _docs(3)
    revised[1]["text"] = "# Revised\n\nThe second document now says gadget instead."
    rag.ingest_many(None, revised, replace=True)

    chunks = rag._dev_corpus().all()
    assert len({c.doc_id for c in chunks}) == 3, "replace must not duplicate documents"
    texts = " ".join(c.text for c in chunks)
    assert "gadget" in texts
    assert "widget1" not in texts, "the superseded version survived"


def test_ingest_many_same_document_twice_in_one_batch(env):
    """A batch containing the same source twice keeps only the last."""
    dup = [
        {"title": "Policy", "text": "Original text about parking.",
         "source_ref": "hr/policy"},
        {"title": "Policy", "text": "Updated text about parking permits.",
         "source_ref": "hr/policy"},
    ]
    result = rag.ingest_many(None, dup, replace=True)

    chunks = rag._dev_corpus().all()
    assert len({c.doc_id for c in chunks}) == 1
    assert result["documents"] == 2
    joined = " ".join(c.text for c in chunks)
    assert "permits" in joined and "Original" not in joined


def test_ingest_many_honours_per_document_overrides(env):
    docs = [
        {"title": "Public note", "text": "Anyone may read this.",
         "source_ref": "a", "classification": "public"},
        {"title": "Secret note", "text": "Restricted material here.",
         "source_ref": "b", "classification": "restricted",
         "scope": rag.user_scope("dana@alex.dev")},
    ]
    rag.ingest_many(None, docs, classification="internal")

    by_title = {c.title: c for c in rag._dev_corpus().all()}
    assert by_title["Public note"].classification == "public"
    assert by_title["Secret note"].classification == "restricted"
    assert by_title["Secret note"].scope == rag.user_scope("dana@alex.dev")
    assert by_title["Public note"].scope == rag.ENTERPRISE_SCOPE


def test_ingest_many_skips_empty_and_reports_it(env):
    docs = _docs(2) + [{"title": "Blank", "text": "   ", "source_ref": "z"}]
    result = rag.ingest_many(None, docs, replace=False)

    assert result["documents"] == 2
    assert result["skipped"] == 1
    assert all(c.title != "Blank" for c in rag._dev_corpus().all())


def test_ingest_many_accepts_a_generator_and_reports_progress(env, monkeypatch):
    """Bulk loads stream; the caller must not have to materialise the corpus."""
    monkeypatch.setattr(rag, "INGEST_WAVE", 5)
    seen: list[tuple[int, int]] = []
    result = rag.ingest_many(
        None, (d for d in _docs(12)), replace=False,
        progress=lambda docs, chunks: seen.append((docs, chunks)),
    )

    assert result["chunks"] == 12
    assert seen, "progress was never reported"
    assert seen[-1] == (12, 12)
    assert [c for _d, c in seen] == sorted(c for _d, c in seen)


def test_ingest_many_result_is_retrievable(env):
    """End to end: bulk-loaded documents come back out of retrieval."""
    rag.ingest_many(None, [
        {"title": "Atlas Brief", "text": BRIEF, "source_ref": "wiki/atlas"},
        {"title": "Expenses policy", "text": HANDBOOK, "source_ref": "wiki/expenses"},
    ])
    hits = rag.retrieve(None, "what is the approved budget for Atlas", k=3)

    assert hits, "bulk-ingested documents were not retrievable"
    assert any("1.2M" in h.chunk.text for h in hits)


def test_index_cache_does_not_serve_a_replaced_corpus(env):
    """Regression: the index cache outlives the corpus it indexed.

    A fresh corpus restarts its version counter at zero, so validating
    the cache on the version alone made every replacement corpus collide
    with its predecessor at the same version — and the cache would hand
    back an index built from documents that no longer exist. In dev that
    is how a query answers out of a corpus that was deliberately cleared.
    """
    from alex import testing

    rag.ingest(None, title="Old", text="The retired system was called Vertex.",
               source_ref="a")
    before = rag.retrieve(None, "Vertex", k=3)
    assert any("Vertex" in h.chunk.text for h in before)

    testing.reset_fakes()  # same scopes, same capabilities, brand new corpus
    rag.ingest(None, title="New", text="The replacement is called Meridian.",
               source_ref="a")
    after = rag.retrieve(None, "Vertex", k=3)

    assert all("Vertex" not in h.chunk.text for h in after), (
        "retrieval served documents from the replaced corpus"
    )
    assert rag.retrieve(None, "Meridian", k=3), "the new corpus is not visible"


def test_a_mixed_embedding_corpus_is_reported_not_silently_degraded(env, caplog):
    """Two embedding widths in one corpus means two embedding *spaces*,
    and cosine between them is noise. Abandoning the dense matrix is
    right; doing it quietly is how a corpus stays broken — the symptom
    is retrieval getting worse for every document, with nothing to say
    that a backend blip during ingest was the cause.
    """
    import logging

    rag.ingest(None, title="Real", text=BRIEF, source_ref="a")
    rag.ingest(None, title="Also real", text=HANDBOOK, source_ref="b")
    # Simulate what a failed batch used to leave behind: one chunk
    # carrying a vector from the other embedding space.
    victim = rag._dev_corpus().all()[0]
    victim.embedding = [0.1] * (len(victim.embedding) + 7)
    rag._dev_corpus().version += 1

    with caplog.at_level(logging.ERROR, logger="agent.rag"):
        rag.retrieve(None, "Atlas budget", k=3, use_graph=False)

    reported = [r.getMessage() for r in caplog.records]
    assert any("embedding widths" in m for m in reported), (
        "a corpus with mixed embedding spaces was degraded silently"
    )
    # The message has to name the cause, or the next person reads
    # "vector search disabled" and goes looking in the wrong place.
    assert any("failed during ingest" in m for m in reported)


def test_retrieval_still_works_on_a_mixed_corpus(env):
    """Degraded, but not broken: BM25 and the scalar path still answer."""
    rag.ingest(None, title="Atlas Brief", text=BRIEF, source_ref="a")
    rag.ingest(None, title="Expenses", text=HANDBOOK, source_ref="b")
    victim = rag._dev_corpus().all()[0]
    victim.embedding = [0.1] * (len(victim.embedding) + 7)
    rag._dev_corpus().version += 1

    hits = rag.retrieve(None, "approved budget for Atlas", k=3, use_graph=False)

    assert hits, "a mixed corpus returned nothing at all"
    assert any("1.2M" in h.chunk.text for h in hits)


# ---- neighbour stitching -------------------------------------------------
# Document-level metrics cannot see stitching by construction (a
# neighbour belongs to a document already retrieved), so these tests
# hold the properties that matter instead: the right neighbour is
# admitted, provenance stays honest, and authorization is not skippable
# by arriving as a neighbour.


def _long_doc_ingest(env_title="Incident review"):
    sections = [
        f"## Part {i}\n\n" + ("routine filler content sentence here. " * 40)
        for i in range(4)
    ]
    sections[2] = (
        "## Part 2\n\nThe zephyr-9912 incident began at 09:14. "
        + ("more narrative " * 60)
    )
    sections[3] = (
        "## Part 3\n\nRoot cause: the mitigation for zephyr-crisis was "
        "a config rollback completed at 10:02. " + ("aftermath notes " * 50)
    )
    rag.ingest(None, title=env_title, text="\n\n".join(sections),
               source_ref="wiki/incident")


def test_stitching_admits_the_adjacent_chunk(env):
    _long_doc_ingest()
    hits = rag.retrieve(None, "zephyr-9912 incident", k=6, use_graph=False)

    assert hits, "the anchor chunk was not retrieved"
    indices = {h.chunk.chunk_index for h in hits}
    anchor = next(h.chunk.chunk_index for h in hits
                  if "zephyr-9912" in h.chunk.text)
    assert (anchor - 1 in indices) or (anchor + 1 in indices), (
        "no neighbour of the anchor chunk was admitted"
    )


def test_stitched_neighbours_carry_no_fake_provenance(env):
    """A neighbour was not ranked by any retriever; claiming a vector or
    lexical rank for it would make the Transparency 'why' a lie."""
    _long_doc_ingest()
    hits = rag.retrieve(None, "zephyr-9912 incident", k=6, use_graph=False)

    stitched = [h for h in hits if "zephyr-9912" not in h.chunk.text
                and h.vector_rank is None and h.lexical_rank is None
                and h.graph_rank is None]
    ranked = [h for h in hits if h.vector_rank or h.lexical_rank]
    assert ranked, "nothing was ranked at all"
    # At least the anchor has provenance; stitched entries (if admitted
    # to the top-k) declare none.
    for h in stitched:
        assert h.score < max(x.score for x in ranked)


def test_stitching_cannot_bypass_authorization(env):
    """Arriving as a neighbour is not a way around the decision point."""
    from agent.access import Principal

    rag.ingest(None, title="Mixed doc",
               text="## Open\n\nzephyr-9912 details here. " + "pad " * 80,
               source_ref="a")
    # A private chunk positioned as the neighbour of a public hit.
    corpus = rag._dev_corpus()
    anchor = next(c for c in corpus.all() if "zephyr-9912" in c.text)
    private = rag.Chunk(
        chunk_id=f"{anchor.doc_id}:{anchor.chunk_index + 1}",
        doc_id=anchor.doc_id, title=anchor.title,
        text="SECRET severance list follows.",
        chunk_index=anchor.chunk_index + 1,
        scope=rag.user_scope("ceo@alex.dev"),
        embedding=[0.0],
    )
    corpus.add(private)

    viewer = Principal(email="intruder@alex.dev", role="member",
                       capabilities=frozenset({rag.DEFAULT_CAPABILITY}))
    hits = rag.retrieve(None, "zephyr-9912", k=6, principal=viewer,
                        use_graph=False)

    assert all("SECRET severance" not in h.chunk.text for h in hits), (
        "a private chunk crossed the boundary by arriving as a neighbour"
    )


def test_stitching_can_be_disabled(env, monkeypatch):
    monkeypatch.setattr(rag, "STITCH_MODE", "off")
    _long_doc_ingest()
    hits = rag.retrieve(None, "zephyr-9912 incident", k=6, use_graph=False)
    # Every hit must have earned its place through a ranker.
    assert all(h.vector_rank or h.lexical_rank or h.graph_rank for h in hits)
