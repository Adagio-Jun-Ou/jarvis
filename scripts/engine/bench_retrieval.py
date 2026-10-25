#!/usr/bin/env python
"""Benchmark retrieval at realistic enterprise scale.

    python scripts/bench_retrieval.py --docs 20000
    python scripts/bench_retrieval.py --docs 100000 --rerank

Generates a synthetic corpus with known ground truth — "needle"
documents that uniquely answer a query, buried in distractors that share
vocabulary with them. Sharing vocabulary is the point: a haystack of
unrelated text makes any retriever look good, and the failure mode at
scale is *near misses*, not random noise.

Reports ingest cost, query latency, and accuracy, because at a hundred
thousand documents those trade against each other and a change that
improves one silently ruins another.
"""

from __future__ import annotations

import argparse
import os
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("ALEX_ENV", "dev")

# Vocabulary that reads like enterprise material, so distractors are
# genuinely confusable rather than lorem ipsum.
SUBJECTS = [
    "migration", "renewal", "incident", "forecast", "onboarding", "audit",
    "contract", "pipeline", "rollout", "outage", "budget", "headcount",
    "integration", "compliance", "escalation", "retention", "latency",
    "throughput", "governance", "procurement", "settlement", "reconciliation",
]
QUALIFIERS = [
    "quarterly", "annual", "regional", "enterprise", "customer", "internal",
    "vendor", "platform", "billing", "reporting", "security", "finance",
]
VERBS = [
    "requires", "blocks", "affects", "depends on", "supersedes", "delays",
    "accelerates", "constrains", "informs", "escalates to",
]


def make_corpus(n_docs: int, n_needles: int, seed: int = 7):
    """Build the haystack and the needles.

    Each needle carries a unique token pair that appears nowhere else, so
    ground truth is exact. Distractors are drawn from the same vocabulary
    and often share one of the needle's two tokens — the near-miss case
    that separates a real retriever from a lucky one.
    """
    rng = random.Random(seed)
    docs: list[tuple[str, str, str]] = []  # (source_ref, title, body)
    needles: list[dict] = []

    for i in range(n_needles):
        subject = rng.choice(SUBJECTS)
        qualifier = rng.choice(QUALIFIERS)
        marker = f"{qualifier}-{subject}-{i:05d}"
        body = (
            f"# {qualifier.title()} {subject.title()} Review {i}\n\n"
            f"The {qualifier} {subject} programme is tracked under reference "
            f"{marker}. Its owner is team-{i % 37} and the committed date is "
            f"day {i % 90} of the cycle. The approved allocation is "
            f"{100 + i % 900} thousand.\n\n## Detail\n\n"
            + " ".join(
                f"{rng.choice(QUALIFIERS)} {rng.choice(SUBJECTS)} "
                f"{rng.choice(VERBS)} {rng.choice(SUBJECTS)}."
                for _ in range(25)
            )
        )
        ref = f"corpus/needle-{i:05d}"
        docs.append((ref, f"{qualifier.title()} {subject.title()} Review {i}", body))
        needles.append(
            {
                "query": f"what is the approved allocation for {marker}?",
                "expected": [ref],
                "kind": "exact-token",
            }
        )
        # A paraphrased question with no unique marker — the hard case.
        needles.append(
            {
                "query": f"who owns the {qualifier} {subject} programme "
                         f"for team-{i % 37}?",
                "expected": [ref],
                "kind": "paraphrase",
            }
        )

    for j in range(n_docs - n_needles):
        qualifier = rng.choice(QUALIFIERS)
        subject = rng.choice(SUBJECTS)
        body = (
            f"# {qualifier.title()} {subject.title()} Note {j}\n\n"
            + " ".join(
                f"{rng.choice(QUALIFIERS)} {rng.choice(SUBJECTS)} "
                f"{rng.choice(VERBS)} {rng.choice(SUBJECTS)}."
                for _ in range(30)
            )
        )
        docs.append(
            (f"corpus/note-{j:06d}", f"{qualifier.title()} {subject.title()} Note {j}", body)
        )

    rng.shuffle(docs)
    return docs, needles


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=20000)
    ap.add_argument("--needles", type=int, default=60)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--queries", type=int, default=60)
    args = ap.parse_args()

    from agent import access, rag
    from agent import retrieval_eval as ev

    docs, needles = make_corpus(args.docs, args.needles)
    print(f"corpus: {len(docs):,} documents, {len(needles)} golden queries")

    started = time.perf_counter()
    for ref, title, body in docs:
        rag.ingest(None, title=title, text=body, source_ref=ref)
    ingest_s = time.perf_counter() - started
    chunks = len(rag._dev_corpus().all())
    print(
        f"ingest: {ingest_s:.1f}s for {chunks:,} chunks "
        f"({chunks / max(ingest_s, 1e-9):,.0f} chunks/s)"
    )

    principal = access.Principal(
        email="bench@alex.dev",
        role="admin",
        capabilities=frozenset({rag.DEFAULT_CAPABILITY}),
    )
    cases = [ev.Case(**n) for n in needles][: args.queries]

    def retrieve(query: str, k: int):
        return rag.retrieve(
            None, query, user_email=principal.email, k=k,
            principal=principal, rerank=args.rerank, use_graph=False,
        )

    # Warm the index so the first query does not pay for everyone.
    warm = time.perf_counter()
    retrieve("warmup query", args.k)
    print(f"first query (index build): {(time.perf_counter() - warm) * 1000:,.0f} ms")

    report = ev.evaluate(
        cases, retrieve=retrieve, k=args.k,
        label=f"{len(docs):,} docs" + (" + rerank" if args.rerank else ""),
    )
    print()
    print(report.summary())

    lat = [r.latency_ms for r in report.results]
    lat.sort()
    print(
        f"\nlatency: p50={statistics.median(lat):.1f}ms  "
        f"p95={lat[int(len(lat) * 0.95)]:.1f}ms  max={lat[-1]:.1f}ms"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
