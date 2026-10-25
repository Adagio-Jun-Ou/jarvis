#!/usr/bin/env python
"""Benchmark retrieval against real enterprise documents.

    python scripts/bench_enterprise.py                 # whatever slices are present
    python scripts/bench_enterprise.py --no-rerank     # stage one alone
    python scripts/bench_enterprise.py --download      # fetch the corpus first

Uses **EnterpriseRAG-Bench** (Onyx, MIT): a fictional company's documents
across nine real enterprise systems — Confluence pages, Slack threads,
Jira tickets, Gmail, Drive, GitHub, HubSpot, Linear, meeting notes — with
500 human-written questions and ground-truth document ids.

This exists because the synthetic benchmark saturated. Generated needles
in generated haystacks measure whether retrieval can find a token; they
cannot measure what actually breaks in production, which is documents
that disagree with each other, questions whose answer is spread across a
thread, and questions with no answer at all. This corpus is deliberately
seeded with misfiled documents, near-duplicates and conflicting versions,
and its question set includes `conflicting_info`, `completeness` and
`info_not_found` categories for exactly that reason.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("ALEX_ENV", "dev")

CORPUS = ROOT / "tests" / "golden" / "erb"
RELEASE = (
    "https://github.com/onyx-dot-app/EnterpriseRAG-Bench/"
    "releases/download/v1.0.0"
)
SLICES = [
    "confluence_slice_0001", "slack_slice_0001",
    "gmail_slice_0001", "jira_slice_0001",
]


def download() -> None:
    import urllib.request

    CORPUS.mkdir(parents=True, exist_ok=True)
    for name in ["questions.jsonl", *(f"{s}.zip" for s in SLICES)]:
        target = CORPUS / name
        if target.exists():
            continue
        print(f"fetching {name}…")
        urllib.request.urlretrieve(f"{RELEASE}/{name}", target)


def load_documents() -> dict[str, tuple[str, str, str]]:
    """doc_id -> (source system, title, body), from whatever slices exist."""
    docs: dict[str, tuple[str, str, str]] = {}
    for archive in sorted(CORPUS.glob("*.zip")):
        with zipfile.ZipFile(archive) as zf:
            for name in zf.namelist():
                if not name.endswith(".txt"):
                    continue
                # confluence/dsid_<hex>__<slug>.txt
                stem = Path(name).stem
                doc_id, _, slug = stem.partition("__")
                source = name.split("/", 1)[0]
                body = zf.read(name).decode("utf-8", "replace")
                title = body.split("\n", 1)[0].strip() or slug.replace("-", " ")
                docs[doc_id] = (source, title[:200], body)
    return docs


def load_questions(docs: dict) -> list:
    """Questions answerable from the slices we actually have.

    A question whose gold document is in a slice we did not download is
    not a failure of retrieval, so it is excluded rather than scored as
    a miss. `info_not_found` questions are kept: they have no gold
    document by design, and getting those right matters.
    """
    from agent.retrieval_eval import Case

    cases, skipped = [], 0
    for line in (CORPUS / "questions.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        q = json.loads(line)
        expected = list(q.get("expected_doc_ids") or [])
        if expected and not all(d in docs for d in expected):
            skipped += 1
            continue
        cases.append(
            Case(
                query=q["question"],
                expected=expected,
                kind=q.get("question_type", "basic"),
                note=q.get("question_id", ""),
            )
        )
    print(f"questions: {len(cases)} answerable, {skipped} skipped (docs not loaded)")
    return cases


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="cap questions")
    args = ap.parse_args()

    if args.download:
        download()
    if not (CORPUS / "questions.jsonl").exists():
        print("corpus missing — run with --download")
        return 1

    from agent import access, rag
    from agent import retrieval_eval as ev

    docs = load_documents()
    print(f"corpus: {len(docs):,} documents from "
          f"{len(Counter(s for s, _t, _b in docs.values()))} systems "
          f"({', '.join(sorted(Counter(s for s, _t, _b in docs.values())))})")

    import time

    started = time.perf_counter()
    rag.ingest_many(
        None,
        (
            {
                "doc_id": doc_id, "title": title, "text": body,
                "source_ref": doc_id, "metadata": {"system": source},
            }
            for doc_id, (source, title, body) in docs.items()
        ),
        replace=False,  # fresh in-memory corpus; nothing to replace
        progress=lambda d, c: print(
            f"  ingested {d:,}/{len(docs):,} docs · {c:,} chunks "
            f"({c / max(time.perf_counter() - started, 1e-9):,.0f}/s)",
            end="\r", flush=True,
        ),
    )
    ingest_s = time.perf_counter() - started
    print()
    chunks = len(rag._dev_corpus().all())
    print(f"ingest: {ingest_s:.1f}s for {chunks:,} chunks "
          f"({chunks / max(ingest_s, 1e-9):,.0f}/s)")

    cases = load_questions(docs)
    if args.limit:
        cases = cases[: args.limit]

    principal = access.Principal(
        email="bench@alex.dev", role="admin",
        capabilities=frozenset({rag.DEFAULT_CAPABILITY}),
    )
    rerank = not args.no_rerank

    def retrieve(query: str, k: int):
        return rag.retrieve(
            None, query, user_email=principal.email, k=k,
            principal=principal, rerank=rerank, use_graph=False,
        )

    warm = time.perf_counter()
    retrieve("warmup", args.k)
    print(f"first query (index build): {(time.perf_counter() - warm) * 1000:,.0f} ms\n")

    report = ev.evaluate(
        cases, retrieve=retrieve, k=args.k,
        label="EnterpriseRAG-Bench" + ("" if rerank else " (stage 1 only)"),
    )
    print(report.summary())

    lat = sorted(r.latency_ms for r in report.results)
    print(f"\nlatency: p50={statistics.median(lat):.1f}ms  "
          f"p95={lat[int(len(lat) * 0.95)]:.1f}ms")

    worst = [r for r in report.results if r.recall == 0 and r.case.expected]
    if worst:
        print(f"\n{len(worst)} question(s) found nothing. First few:")
        for r in worst[:5]:
            print(f"  [{r.case.kind}] {r.case.query[:88]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
