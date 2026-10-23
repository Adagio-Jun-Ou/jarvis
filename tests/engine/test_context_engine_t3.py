"""T3 assembly must not emit a section header without any rendered items,
and must tolerate beads/edges of unexpected shapes."""
from agent.context_engine import _build_t3


def test_memory_header_only_when_items_render():
    # beads is truthy but every entry is an unrenderable shape -> no header.
    def recall(user, limit=5, query=None):
        return [object(), 42]  # not (k, v) tuples

    out = _build_t3("hi", recall, None, "alice@x.com")
    assert "RECALLED MEMORIES" not in out
    assert "USER MESSAGE: hi" in out


def test_memory_renders_valid_beads():
    """Fusion renders a memory as something known about the user, and
    keeps its kind. The legacy per-channel header is gone by design —
    evidence is grouped by subject now, not by which store it came
    from."""
    def recall(user, limit=5, query=None):
        return [("pref", "likes python"), ("loc", "NYC")]

    out = _build_t3("hi", recall, None, "alice@x.com")
    assert "about you: likes python" in out
    assert "NYC" in out


def test_malformed_edges_are_dropped_not_rendered():
    def kg(user, msg):
        return ["not-a-dict", 5]

    out = _build_t3("hi", None, kg, "alice@x.com")
    assert "fact:" not in out
    assert "not-a-dict" not in out


def test_kg_renders_valid_edges():
    def kg(user, msg):
        return [{"src": "Acme", "rel": "employs", "dst": "Bob", "confidence": 0.9}]

    out = _build_t3("hi", None, kg, "alice@x.com")
    assert "fact: Acme employs Bob" in out


def test_evidence_is_grouped_under_the_subject_it_concerns():
    """The point of fusion: a document and a fact about the same subject
    arrive together under that subject, not in separate inventories."""
    def retrieve(user, msg):
        return [{
            "text": "Project Atlas is the Q1 migration.",
            "score": 0.9,
            "citation": {"title": "Atlas Brief", "source_ref": "wiki/atlas"},
        }]

    def kg(user, msg):
        return [{"src": "Project Atlas", "rel": "owned_by", "dst": "Data Platform",
                 "confidence": 0.9}]

    out = _build_t3("tell me about Project Atlas", None, kg, "alice@x.com", retrieve)
    # Both land under one subject heading, and the document is citable.
    assert "[1] Atlas Brief" in out
    assert "fact: Project Atlas owned_by Data Platform" in out
    assert "Cite a document by its bracketed number" in out
