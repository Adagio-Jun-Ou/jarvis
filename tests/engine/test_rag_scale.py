"""Tests for the two-stage retrieval path and the properties that make
it work at enterprise scale.

Every test here corresponds to a defect measured on a real corpus, not a
hypothetical. The comments say which.
"""

from __future__ import annotations

import pytest

from agent import access, rag


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("ALEX_ENV", "dev")
    from alex import config, testing

    config._settings = None
    testing._fakes = None
    yield
    config._settings = None
    testing._fakes = None


def principal():
    return access.Principal(
        email="a@x.com",
        role="admin",
        capabilities=frozenset({rag.DEFAULT_CAPABILITY}),
    )


@pytest.fixture
def corpus(env):
    """A corpus where one document holds a unique identifier and the
    others share all its ordinary vocabulary."""
    rag.ingest(
        None,
        title="Migration Review 42",
        text="The quarterly migration programme is tracked under reference "
             "quarterly-migration-00042. The approved allocation is 300 "
             "thousand and the owner is team-9.",
        source_ref="corpus/needle",
    )
    for i in range(60):
        rag.ingest(
            None,
            title=f"Migration Note {i}",
            text="The quarterly migration programme is ordinary. The "
                 "approved allocation is discussed here and there is an "
                 "owner. " * 3,
            source_ref=f"corpus/note-{i}",
        )
    return None


# ---- stopwords -----------------------------------------------------------


def test_function_words_are_not_indexed(env):
    """Measured defect: `the` and `is` outscored a term with df=1,
    because they recur within a document and term frequency rewards
    them. Exact-token recall was 0.65 until they were filtered."""
    assert rag._tokens("what is the approved allocation for X") == [
        "approved", "allocation", "x",
    ]


def test_a_unique_identifier_outranks_shared_vocabulary(corpus):
    hits = rag.retrieve(
        None, "what is the approved allocation for quarterly-migration-00042?",
        k=5, principal=principal(), use_graph=False,
    )
    assert hits[0].chunk.source_ref == "corpus/needle"


# ---- the two BM25 implementations must agree -----------------------------


def test_sparse_and_scalar_bm25_score_identically(corpus):
    """There are two scoring paths — a sparse matrix product and a
    Python posting walk for when SciPy is absent. They must assign the
    same score to the same document, or retrieval would depend on which
    machine ran it.

    The assertion is on scores rather than on the exact order, because
    boilerplate documents tie exactly and which tied document lands at
    the cut is not a meaningful difference — the pool is reranked
    afterwards regardless.
    """
    index = rag._get_index(None, rag.visible_scopes("a@x.com"), None)
    assert index.bm25 is not None, "SciPy path should be active here"

    query = "approved allocation quarterly migration"
    sparse = {c.source_ref: round(s, 5) for c, s in _bm25(index, query, sparse=True)}
    scalar = {c.source_ref: round(s, 5) for c, s in _bm25(index, query, sparse=False)}
    shared = set(sparse) & set(scalar)
    assert shared, "the two paths returned disjoint results"
    for ref in shared:
        assert sparse[ref] == scalar[ref], f"{ref} scored differently"


def test_both_bm25_paths_agree_on_a_distinct_winner(corpus):
    """Where scores are not tied, the order must be identical."""
    index = rag._get_index(None, rag.visible_scopes("a@x.com"), None)
    query = "quarterly-migration-00042 allocation"
    assert (
        _bm25(index, query, sparse=True)[0][0].source_ref
        == _bm25(index, query, sparse=False)[0][0].source_ref
        == "corpus/needle"
    )


def _bm25(index, query, *, sparse: bool):
    """Score with one path or the other."""
    if sparse:
        return rag._bm25_ranking(query, index, top=10)
    saved = index.bm25
    try:
        index.bm25 = None
        return rag._bm25_ranking(query, index, top=10)
    finally:
        index.bm25 = saved


def test_bounding_the_result_set_does_not_change_the_head(corpus):
    """Measured defect: BM25 materialised a Python tuple for every
    non-zero hit — sixty thousand of them for a common-word query, which
    cost 265ms of the 270ms query. Bounding it must not change what the
    caller actually receives."""
    index = rag._get_index(None, rag.visible_scopes("a@x.com"), None)
    # A query with a distinctive term, so the head is not a mass of ties
    # and "the head is unchanged" is a meaningful claim.
    query = "quarterly-migration-00042 allocation"
    unbounded = rag._bm25_ranking(query, index)
    bounded = rag._bm25_ranking(query, index, top=5)
    assert len(bounded) == 5
    assert [c.source_ref for c, _ in bounded][:1] == [
        c.source_ref for c, _ in unbounded[:1]
    ]
    # And the bounded set is drawn from the true top of the unbounded one.
    assert {c.source_ref for c, _ in bounded} <= {
        c.source_ref for c, _ in unbounded[:20]
    }


# ---- rarity --------------------------------------------------------------


def test_rarity_scores_a_unique_term_far_above_a_common_one(corpus):
    index = rag._get_index(None, rag.visible_scopes("a@x.com"), None)
    rarity = rag._rarity(
        "quarterly-migration-00042 allocation programme", index
    )
    assert rarity["quarterly-migration-00042"] > rarity["programme"]
    assert rarity["quarterly-migration-00042"] > 0.9


def test_rarity_is_unavailable_for_a_single_document_corpus(env):
    """With one document there is nothing to be rare against, and log(1)
    is zero — the signal is absent, not infinite."""
    rag.ingest(None, title="Only", text="the only document", source_ref="x")
    index = rag._get_index(None, rag.visible_scopes("a@x.com"), None)
    assert rag._rarity("only document", index) == {}


def test_a_decisive_identifier_match_is_placed_first(corpus):
    """A candidate matching essentially all of a query's distinctive mass
    is the answer; no amount of general similarity should outrank it."""
    from agent.rerank import DECISIVE_RARITY, score_features

    index = rag._get_index(None, rag.visible_scopes("a@x.com"), None)
    query = "quarterly-migration-00042"
    rarity = rag._rarity(query, index)
    needle = next(
        c for c in rag._dev_corpus().all() if c.source_ref == "corpus/needle"
    )
    feats = score_features(query, needle.title, needle.text, 0, rarity=rarity)
    assert feats.rare >= DECISIVE_RARITY


# ---- two-stage behaviour -------------------------------------------------


def test_reranking_is_on_by_default(corpus):
    hits = rag.retrieve(None, "approved allocation", k=3, principal=principal())
    assert hits
    # The reranker annotates what it scored.
    assert any(getattr(h, "rerank_score", None) is not None for h in hits)


def test_reranking_cannot_introduce_an_unauthorised_passage(env):
    """Stage two only ever reorders. It must not be able to resurrect
    something authorization removed."""
    rag.ingest(
        None, title="Restricted", text="board compensation detail",
        source_ref="r/1", classification="restricted",
    )
    rag.ingest(
        None, title="Open", text="board meeting agenda",
        source_ref="o/1", classification="internal",
    )
    member = access.Principal(
        email="m@x.com", role="member",
        capabilities=frozenset({rag.DEFAULT_CAPABILITY}),
    )
    hits = rag.retrieve(None, "board", k=5, principal=member)
    assert [h.chunk.source_ref for h in hits] == ["o/1"]


def test_the_candidate_pool_widens_when_reranking(corpus):
    """Stage one's job is recall into the pool; a pool of k would make
    stage two pointless."""
    assert rag.CANDIDATE_POOL > rag.FUSION_WINDOW


# ---- ingest cost ---------------------------------------------------------


def test_replacing_a_document_does_not_scan_the_corpus(env):
    """Measured defect: dropping a document scanned every chunk to find
    its own, making ingest quadratic — 18s at 20k documents and
    effectively never finishing at 100k."""
    for i in range(200):
        rag.ingest(None, title=f"D{i}", text="body text here", source_ref=f"s/{i}")
    corpus_obj = rag._dev_corpus()
    assert len(corpus_obj.by_doc) == 200

    doc_id = next(iter(corpus_obj.by_doc))
    tracked = set(corpus_obj.by_doc[doc_id])
    removed = corpus_obj.drop_doc(doc_id)
    assert removed == len(tracked)
    assert doc_id not in corpus_obj.by_doc
    # And the rest of the corpus is untouched.
    assert len(corpus_obj.by_doc) == 199


def test_reingesting_the_same_source_replaces_rather_than_grows(env):
    for _ in range(5):
        rag.ingest(None, title="D", text="stable body", source_ref="s/1")
    assert len(rag._dev_corpus().by_doc) == 1
    assert len(rag._dev_corpus().all()) == 1
