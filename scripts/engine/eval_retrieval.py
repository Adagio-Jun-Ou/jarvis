#!/usr/bin/env python
"""Measure the enterprise retrieval path against the golden set.

    python scripts/eval_retrieval.py                 # baseline
    python scripts/eval_retrieval.py --rerank        # with reranking
    python scripts/eval_retrieval.py --compare       # both, with deltas
    python scripts/eval_retrieval.py --json out.json # machine-readable

Retrieval is measured on its own, before generation: if the right
passage never reaches the prompt, no amount of prompting recovers it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("ALEX_ENV", "dev")

GOLDEN = ROOT / "tests" / "golden" / "retrieval.json"


def build_corpus():
    """Seed the same corpus the dev app serves, then return a principal.

    The evaluation runs as a *manager*, not an administrator, so the
    access-control cases measure a real boundary rather than a
    privileged view that would never fail.
    """
    from agent import access
    from alex.testing import get_fakes

    get_fakes()  # fresh fakes
    from app import _dev_seed

    _dev_seed._seed_enterprise_knowledge("demo@alex.dev")

    from agent import rag

    return access.Principal(
        email="demo@alex.dev",
        role="manager",
        capabilities=frozenset({rag.DEFAULT_CAPABILITY}),
        teams=frozenset({"team:platform"}),
    )


def make_retriever(principal, *, rerank: bool):
    from agent import rag

    def retrieve(query: str, k: int):
        return rag.retrieve(
            None,
            query,
            user_email=principal.email,
            k=k,
            principal=principal,
            rerank=rerank,
        )

    return retrieve


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--json", dest="json_out", default="")
    parser.add_argument("--fail-under", type=float, default=0.0,
                        help="exit non-zero if recall@k falls below this")
    args = parser.parse_args()

    from agent import retrieval_eval as ev

    cases = ev.load_cases(str(GOLDEN))
    principal = build_corpus()

    baseline = ev.evaluate(
        cases,
        retrieve=make_retriever(principal, rerank=False),
        k=args.k,
        label="baseline (hybrid RRF)",
    )

    payload: dict = {"baseline": baseline.to_dict()}
    print(baseline.summary())

    if args.compare or args.rerank:
        candidate = ev.evaluate(
            cases,
            retrieve=make_retriever(principal, rerank=True),
            k=args.k,
            label="with reranking",
        )
        print()
        print(candidate.summary())
        payload["candidate"] = candidate.to_dict()
        if args.compare:
            delta = ev.compare(baseline, candidate)
            payload["comparison"] = delta
            print(f"\nverdict: {delta['verdict']}")
            print(f"  deltas: {delta['delta']}")
            for r in delta["regressed"]:
                print(f"  REGRESSED [{r['kind']}] {r['query']!r} "
                      f"ndcg {r['ndcg'][0]} -> {r['ndcg'][1]}")
            for r in delta["improved"][:6]:
                print(f"  improved  [{r['kind']}] {r['query']!r} "
                      f"ndcg {r['ndcg'][0]} -> {r['ndcg'][1]}")

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        print(f"\nwrote {args.json_out}")

    headline = payload.get("candidate", payload["baseline"])
    if args.fail_under and headline["recall_at_k"] < args.fail_under:
        print(
            f"\nFAIL: recall@{args.k}={headline['recall_at_k']} "
            f"is below the {args.fail_under} gate"
        )
        return 1
    if headline["leaks"]:
        print(f"\nFAIL: {headline['leaks']} case(s) retrieved forbidden material")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
