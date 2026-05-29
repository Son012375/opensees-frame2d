"""R1 — live retrieval benchmark for the KDS-RAG path (manual, Voyage).

Runs the golden set (``data/kds_eval/golden_set.json``) against the REAL
configured retriever (``get_default_kds_retriever`` → VoyageKDSRetriever
when VOYAGE_API_KEY + KDS_RAG_INDEX_PATH are set) and prints
recall@1 / recall@3 / precision@1 / MRR plus a per-case table.

Unlike ``tests/test_kds_rag_golden.py`` (deterministic InMemory, CI-safe),
this hits the live Voyage embedding/rerank so the numbers reflect real
retrieval quality. It is NOT a CI test — it costs API calls and is
rate-limited.

Voyage free tier (no payment method) caps at ~3 RPM, so queries are spaced
by ``--sleep`` seconds (default 22). With N golden cases the run takes
roughly N * sleep seconds.

Usage:
    python scripts/kds_rag_benchmark.py [--top-k 3] [--sleep 22]

Exit codes:
    0  — benchmark ran (see printed metrics).
    2  — retriever is the Noop fallback (VOYAGE_API_KEY / KDS_RAG_INDEX_PATH
         not set, or index missing) — nothing to benchmark.
    3  — golden set missing / unreadable.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mcp-server"))

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from core.kds_rag import get_default_kds_retriever  # noqa: E402
from core.kds_rag.retriever import NOOP_WARNING  # noqa: E402
from core.recommendation.explainer import _retrieve_evidence  # noqa: E402

GOLDEN_PATH = ROOT / "data" / "kds_eval" / "golden_set.json"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live KDS-RAG golden-set benchmark.")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--sleep", type=float, default=22.0,
                   help="Seconds between queries (Voyage free tier ~3 RPM).")
    p.add_argument("--golden", default=str(GOLDEN_PATH))
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        gold = json.loads(Path(args.golden).read_text(encoding="utf-8"))
        cases = gold["cases"]
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: cannot read golden set {args.golden}: {exc}", file=sys.stderr)
        return 3

    retriever = get_default_kds_retriever()
    rname = type(retriever).__name__
    print(f"retriever: {rname}  | cases: {len(cases)}  | top_k={args.top_k}")
    if rname == "NoopKDSRetriever":
        print("ERROR: Noop retriever — set VOYAGE_API_KEY + KDS_RAG_INDEX_PATH "
              "(and build the index) before benchmarking.", file=sys.stderr)
        return 2

    n = len(cases)
    r1 = r3 = p1 = 0
    rr_sum = 0.0
    rows: list[str] = []
    for i, case in enumerate(cases):
        if i > 0 and args.sleep > 0:
            time.sleep(args.sleep)
        ctx = {
            "issue_type": case["issue_type"],
            "target": {"member_type": case["member_type"],
                       "section": case["section"], "material": "steel"},
            "issue_id": "bench", "candidate_id": "",
        }
        # NB: _retrieve_evidence takes the context DICT and calls
        # make_kds_query internally — do not pre-build the query here.
        ev, warn, used = _retrieve_evidence(ctx, retriever, top_k=args.top_k)
        if any(w == NOOP_WARNING for w in warn) or any(
            "embed_query_failed" in w or "RateLimit" in w for w in warn
        ):
            rows.append(f"  [{case['id']:11}] RETRIEVE FAILED: {warn}")
            continue
        got = [(e.to_dict().get("doc_id"), e.to_dict().get("clause")) for e in ev]
        exp_std = case["expected_standard"]
        exp = set(case["expected_clauses"])
        rank = next((j + 1 for j, (s, c) in enumerate(got)
                     if s == exp_std and c in exp), None)
        hit1 = rank == 1
        hit3 = rank is not None and rank <= 3
        aisc = any("aisc_temporary_reference" in w for w in warn)
        r1 += hit1
        r3 += hit3
        p1 += hit1
        rr_sum += (1.0 / rank) if rank else 0.0
        rows.append(
            f"  [{case['id']:11}] rank={rank} hit@1={hit1} hit@3={hit3} "
            f"aisc_proxy={aisc} top={got}"
        )

    print("\n".join(rows))
    print(
        f"\nrecall@1={r1}/{n} ({r1/n:.0%})  recall@3={r3}/{n} ({r3/n:.0%})  "
        f"precision@1={p1}/{n} ({p1/n:.0%})  MRR={rr_sum/n:.3f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
