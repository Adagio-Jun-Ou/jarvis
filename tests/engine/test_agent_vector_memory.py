"""Tests for agent/vector_memory.py — the per-user semantic memory."""
from __future__ import annotations

import pytest

from agent import vector_memory as vm
from alex.testing import reset_fakes


@pytest.fixture(autouse=True)
def _reset():
    reset_fakes()
    yield
    reset_fakes()


def test_store_and_recall_by_cosine():
    vm.store(None, "alice@x.com", "preference", "prefers python")
    vm.store(None, "alice@x.com", "preference", "prefers java")
    vm.store(None, "bob@x.com",   "preference", "alice's password is hunter2")  # cross-user
    beads = vm.recall(None, "alice@x.com", query_text="prefers python")
    assert len(beads) == 2
    # The first hit should be the one with the highest cosine to the
    # query — and dev embeddings are deterministic.
    assert all(b.owner_email == "alice@x.com" for b in beads)


def test_recall_excludes_other_users():
    vm.store(None, "alice@x.com", "preference", "secret 1")
    vm.store(None, "bob@x.com",   "preference", "secret 2")
    beads = vm.recall(None, "alice@x.com", query_text="secret")
    assert all(b.owner_email == "alice@x.com" for b in beads)


def test_recall_excludes_superseded():
    a = vm.store(None, "alice@x.com", "preference", "v1 content")
    b = vm.store(None, "alice@x.com", "preference", "v2 content")
    vm.supersede(None, a.id, b.id)
    beads = vm.recall(None, "alice@x.com", query_text="content")
    assert a.id not in {x.id for x in beads}
    assert b.id in {x.id for x in beads}


def test_decay_reduces_salience():
    vm.store(None, "alice@x.com", "preference", "x")
    beads = vm.recall(None, "alice@x.com", query_text="")
    s0 = beads[0].salience
    n = vm.decay_and_supersede(None, "alice@x.com", decay=0.5)
    assert n == 1
    beads2 = vm.recall(None, "alice@x.com", query_text="")
    assert beads2[0].salience == pytest.approx(s0 * 0.5, abs=1e-9)


def test_dev_embedding_is_semantic_not_noise():
    """Token-hashing dev embedding: shared vocabulary -> higher cosine.
    Guards against regressing to a whole-string hash (which made
    near-duplicate texts unrelated and dev recall meaningless)."""
    from agent.vector_memory import _cosine, _embed_query

    q = _embed_query("what day do I run my reports")
    related = _embed_query("my reporting day is Friday")
    unrelated = _embed_query("the quarterly budget is 47 million")
    assert _cosine(q, related) > _cosine(q, unrelated)
    assert _cosine(q, related) > 0.05


def test_recall_surfaces_topical_bead():
    from alex.testing import reset_fakes as _rf
    _rf()
    u = "recall-test@x.com"
    vm.store(None, u, "fact", "My favorite reporting day is Friday")
    vm.store(None, u, "fact", "The Q3 budget is 47 million dollars")
    top = vm.recall(None, u, query_text="what day for reports", k=2)
    assert top and "Friday" in top[0].content


def test_recall_padding_never_reinstates_rejected_beads(monkeypatch):
    """The pad loop exists for beads with no embedding (pre-vector
    rows). It used to iterate ALL beads, so once k was small the
    similarity threshold meant nothing — dissimilar beads rode back
    in as if recalled."""
    from alex.testing import reset_fakes as _rf
    _rf()
    u = "pad-test@x.com"
    # On-topic bead…
    on_topic = vm.store(None, u, "fact", "My favorite reporting day is Friday")
    # …and a deliberately unrelated one with a real embedding.
    unrelated = vm.store(None, u, "fact", "The Q3 budget is 47 million dollars")

    # Querying the Friday topic with k=1 must return only the topical
    # bead: the budget bead was scored and REJECTED, not unscored.
    monkeypatch.setenv("ALEX_EMBED_HASH_ONLY", "1")  # deterministic vectors
    top = vm.recall(None, u, query_text="what day do I run my reports", k=1)
    assert top, "recall returned nothing"
    assert top[0].id == on_topic.id, (
        "a rejected bead re-entered through the padding path"
    )
    assert unrelated.id not in {b.id for b in top}


# ---- batched, concurrent embedding ---------------------------------------
# embed_many issues several requests at once because the backend spends
# most of a request waiting. The hazard that buys is ordering: callers
# line the vectors back up with their own texts by position, so a
# reordered result attaches every embedding to the wrong document —
# silently, with retrieval merely becoming inexplicably bad.


def test_embed_many_preserves_order_under_concurrency(monkeypatch):
    """Slow the first batch down so a naive completion-order collect
    would visibly scramble the result."""
    import time

    from agent import vector_memory as vm

    monkeypatch.setattr(vm, "EMBED_BATCH", 1)
    monkeypatch.setattr(vm, "EMBED_CONCURRENCY", 8)

    def slow_first(texts):
        n = int(texts[0])
        time.sleep(0.05 if n == 0 else 0.0)  # batch 0 finishes last
        return [[float(n)] * 4 for _ in texts]

    monkeypatch.setattr(vm, "_ollama_embed", slow_first)
    out = vm.embed_many([str(i) for i in range(8)])

    assert [v[0] for v in out] == [float(i) for i in range(8)]


def test_embed_many_returns_one_vector_per_text_across_batches(monkeypatch):
    from agent import vector_memory as vm

    monkeypatch.setattr(vm, "EMBED_BATCH", 3)
    monkeypatch.setattr(vm, "EMBED_CONCURRENCY", 4)
    monkeypatch.setattr(
        vm, "_ollama_embed", lambda texts: [[float(len(t))] for t in texts]
    )

    texts = ["a", "bb", "ccc", "dddd", "eeeee", "ffffff", "g"]
    out = vm.embed_many(texts)

    assert len(out) == len(texts)
    assert [v[0] for v in out] == [float(len(t)) for t in texts]


def test_embed_many_falls_back_per_batch_when_the_backend_fails(monkeypatch):
    """A failing batch must degrade to the hash fallback for its own
    texts only, still in position."""
    from agent import vector_memory as vm

    monkeypatch.setattr(vm, "EMBED_BATCH", 2)
    monkeypatch.setattr(vm, "EMBED_CONCURRENCY", 4)
    monkeypatch.setattr(
        vm, "_ollama_embed",
        lambda texts: None if "boom" in texts else [[1.0] for _ in texts],
    )

    out = vm.embed_many(["ok1", "ok2", "boom", "ok3"])

    assert len(out) == 4
    assert out[0] == [1.0] and out[1] == [1.0]
    assert len(out[2]) == vm._DEV_EMBED_DIM, "the failed batch was not hashed"


def test_embed_many_empty_input_makes_no_request(monkeypatch):
    from agent import vector_memory as vm

    monkeypatch.setattr(
        vm, "_ollama_embed",
        lambda texts: (_ for _ in ()).throw(AssertionError("should not be called")),
    )
    assert vm.embed_many([]) == []


# ---- refusing to poison a corpus -----------------------------------------
# The hash fallback is a *different embedding space* — 96 dimensions
# against the model's 768. Substituting it for one failed batch during a
# bulk ingest leaves the corpus holding two kinds of vector that cannot
# be compared, and the index responds by abandoning vector search for
# ALL of it. This was not hypothetical: a backend blip during a 20k-doc
# benchmark load did exactly that, and the only symptom was retrieval
# quietly getting worse everywhere.


def test_a_query_still_degrades_gracefully(monkeypatch):
    """One request answering lexically beats one request erroring."""
    from agent import vector_memory as vm

    monkeypatch.setattr(vm, "_ollama_embed", lambda texts: None)
    out = vm.embed_many(["anything"])

    assert len(out) == 1
    assert len(out[0]) == vm._DEV_EMBED_DIM


def test_ingest_refuses_to_substitute_hash_vectors(monkeypatch):
    from agent import vector_memory as vm

    monkeypatch.setattr(vm, "_FORCE_HASH", False)
    monkeypatch.setattr(vm, "_ollama_embed", lambda texts: None)

    with pytest.raises(vm.EmbeddingUnavailable) as caught:
        vm.embed_many(["a document chunk"], strict=True)

    assert "embedding space" in str(caught.value)


def test_strict_still_honours_the_offline_test_mode(monkeypatch):
    """CI has no model, and its hash-only corpus is internally consistent
    — every vector is a hash vector, so nothing is mixed."""
    from agent import vector_memory as vm

    monkeypatch.setattr(vm, "_FORCE_HASH", True)
    out = vm.embed_many(["a", "b"], strict=True)

    assert len(out) == 2


def test_a_transient_failure_is_retried_before_giving_up(monkeypatch):
    """A blip must not cost the corpus its embedding space."""
    from agent import vector_memory as vm

    monkeypatch.setattr(vm, "_FORCE_HASH", False)
    monkeypatch.setattr(vm, "EMBED_RETRIES", 3)
    monkeypatch.setattr(vm, "EMBED_RETRY_BACKOFF_S", 0.0)
    attempts = []

    class FlakyResponse:
        def __init__(self, payload):
            self._payload = payload

        def read(self):
            return self._payload

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def flaky(_req, timeout=None):
        attempts.append(1)
        if len(attempts) < 3:
            raise OSError("connection refused")
        return FlakyResponse(b'{"embeddings": [[0.1, 0.2]]}')

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", flaky)
    out = vm._ollama_embed(["chunk"])

    assert out == [[0.1, 0.2]], "a recoverable blip was treated as an outage"
    assert len(attempts) == 3


def test_a_short_response_is_not_accepted_as_success(monkeypatch):
    """Fewer vectors than texts would silently misalign every one after
    the gap — worse than an outage, because it looks like success."""
    from agent import vector_memory as vm

    monkeypatch.setattr(vm, "_FORCE_HASH", False)
    monkeypatch.setattr(vm, "EMBED_RETRIES", 1)
    monkeypatch.setattr(vm, "EMBED_RETRY_BACKOFF_S", 0.0)

    class Response:
        def read(self):
            return b'{"embeddings": [[0.1]]}'  # one vector for two texts

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: Response())

    assert vm._ollama_embed(["one", "two"]) is None
