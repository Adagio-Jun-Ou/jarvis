"""The compression eval — compaction changes must be measurable.

Hermes ships a separate compression-eval harness (fixtures through
compress(), probe questions, judged answers) precisely because
compaction is where long-horizon agents quietly lose their minds: a
"small" change to the summarizer prompt or the carve boundaries shows
up weeks later as an agent that forgot the invoice number. This is the
deterministic CI version: a realistic working-session fixture and
probes that must hold STRUCTURALLY, so any compaction regression fails
a named assertion instead of a customer.

Probes:
* the anchor (the task) survives verbatim through any number of
  compactions;
* the working tail survives verbatim;
* stale tool payloads are stubbed but the fact each call happened
  survives;
* identifiers placed where the deterministic digest reads (first
  lines) survive with no model at all;
* compaction actually compresses (ratio bound);
* five successive compactions do not grow the head (non-degrading
  lineage) and the generation counter counts them;
* the live task-state note rides the digest head.
"""

from __future__ import annotations

import pytest

from agent.harness.compaction import compact_history


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("ALEX_ENV", "dev")
    from alex import config, testing

    config._settings = None
    testing._fakes = None
    yield
    config._settings = None
    testing._fakes = None


# A realistic working session: a task with identifiers, tool calls with
# fat payloads, decisions, and a mid-run steering note.
TASK = ("Reconcile invoice INV-4471 against the Northwind ledger and "
        "email finance the variance")


def _fixture_history(rounds: int = 14) -> list[dict]:
    hist: list[dict] = [{"role": "user", "content": TASK}]
    for i in range(rounds):
        hist.append({
            "role": "assistant",
            "content": f"Decision: variance line {i} traced to PO-88{i:02d}",
        })
        hist.append({
            "role": "tool_call",
            "content": {"tool": "read_spreadsheet", "args": {"row": i}},
        })
        hist.append({
            "role": "tool_result",
            "content": f"ledger row {i}: amount 402.1{i} EUR " + "cell " * 400,
        })
    hist.append({
        "role": "user",
        "content": "[steering from dana@alex.dev, mid-task]: use the FY26 "
                   "ledger, not FY25",
    })
    hist += [
        {"role": "assistant", "content": f"working the tail {i}"}
        for i in range(6)
    ]
    return hist


def _chars(history: list[dict]) -> int:
    return sum(len(str(m.get("content") or "")) for m in history)


def test_anchor_and_tail_survive_verbatim(env):
    hist = _fixture_history()
    out, ran = compact_history(hist, trigger=10, keep_tail=6)
    assert ran
    assert any(m.get("content") == TASK for m in out[:2]), (
        "the task itself must never be summarized"
    )
    assert [m["content"] for m in out[-6:]] == [m["content"] for m in hist[-6:]]


def test_stale_payloads_are_stubbed_but_the_calls_survive(env):
    hist = _fixture_history()
    out, _ = compact_history(hist, trigger=10, keep_tail=6)
    digest = out[0]["content"]
    # Stubs keep a short identifying preview by design; what must NOT
    # survive is any unstubbed payload (a long verbatim run).
    assert "cell " * 50 not in digest, "an unstubbed payload survived"
    assert "read_spreadsheet" in digest, (
        "the FACT each call happened is the cheap thing worth keeping"
    )


def test_identifiers_survive_with_no_model_at_all(env):
    """The deterministic fallback digests first lines — decisions and
    identifiers written there survive an aux-model outage."""
    hist = _fixture_history()
    out, _ = compact_history(hist, trigger=10, keep_tail=6)
    digest = out[0]["content"]
    assert "PO-88" in digest
    assert "402.1" in digest


def test_compaction_actually_compresses(env):
    hist = _fixture_history(rounds=20)
    out, _ = compact_history(hist, trigger=10, keep_tail=6)
    assert _chars(out) < _chars(hist) * 0.5, (
        f"compaction kept {_chars(out)}/{_chars(hist)} chars — that is "
        "not compression"
    )


def test_five_compactions_do_not_grow_the_head(env):
    """Non-degrading lineage: repeated compaction must converge, not
    accrete digests of digests."""
    hist = _fixture_history()
    sizes = []
    for gen in range(5):
        hist, ran = compact_history(hist, trigger=10, keep_tail=6)
        assert ran, f"compaction #{gen + 1} did not run"
        sizes.append(len(hist[0]["content"]))
        # Grow the tail again so the next round has a middle to eat.
        hist = list(hist) + [
            {"role": "tool_result",
             "content": f"gen {gen} payload " + "z" * 600},
            {"role": "assistant", "content": f"gen {gen} noted"},
        ] * 6
    assert "(compaction #5)" in [m for m in hist if "DIGEST" in str(m.get("content"))][0]["content"]
    assert max(sizes) <= sizes[0] * 3, (
        f"the digest head grew without bound across compactions: {sizes}"
    )


def test_the_live_note_rides_every_compaction(env):
    hist = _fixture_history()
    out, _ = compact_history(
        hist, trigger=10, keep_tail=6,
        live_note="EXISTING PLAN:\n  [x] 1. trace\n  [~] 2. email finance",
    )
    head = out[0]["content"]
    assert "CURRENT STATE" in head and "[~] 2. email finance" in head
