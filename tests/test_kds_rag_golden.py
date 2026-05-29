"""R1 — deterministic golden-set regression for the KDS-RAG retrieval path.

Locks the just-ingested KDS 14 31 10 corpus + the issue_type->topic query
routing against a golden set of realistic member design-check scenarios
(``data/kds_eval/golden_set.json``).

This test is CI-safe: it builds an :class:`InMemoryKDSRetriever` over the
REAL ``data/kds_sample`` corpus (field-equality + keyword-overlap scoring),
so it never calls Voyage. The live Voyage quality numbers (which differ —
e.g. embedding can let the interaction chunk bleed above compression for an
axial query) are produced separately by ``scripts/kds_rag_benchmark.py``.

What a failure means:
    * a corpus chunk lost/changed its ``topic`` / ``limit_state`` tag, or
    * ``make_kds_query`` / ISSUE_TYPE_TO_TOPIC routing changed, or
    * a golden case's expected clause is no longer retrievable.
Any of these is a real regression in the citation path — investigate
before updating the golden set.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mcp-server"))

from core.kds_rag import make_kds_query  # noqa: E402
from core.kds_rag.ingest import chunk_kds_document, load_kds_documents  # noqa: E402
from core.kds_rag.retriever import InMemoryKDSRetriever  # noqa: E402
from core.kds_rag.schemas import KDSChunk  # noqa: E402

CORPUS_DIR = ROOT / "data" / "kds_sample"
GOLDEN_PATH = ROOT / "data" / "kds_eval" / "golden_set.json"
TOP_K = 3


def _build_inmemory_from_corpus() -> InMemoryKDSRetriever:
    """Load the real sample corpus and wrap it in the deterministic retriever."""
    docs = load_kds_documents(str(CORPUS_DIR))
    chunks: list[KDSChunk] = []
    for d in docs:
        for c in chunk_kds_document(d):
            chunks.append(KDSChunk(
                chunk_id=c["chunk_id"],
                standard_id=c.get("standard_id") or "",
                text=c.get("text") or "",
                clause_id=c.get("clause_id"),
                title=c.get("title"),
                topic=c.get("topic"),
                limit_state=c.get("limit_state"),
                material=c.get("material"),
                jurisdiction=c.get("jurisdiction") or "KDS",
            ))
    return InMemoryKDSRetriever(chunks)


def _load_cases() -> list[dict]:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))["cases"]


def _retrieve(retriever: InMemoryKDSRetriever, case: dict) -> list[tuple[str, str]]:
    ctx = {
        "issue_type": case["issue_type"],
        "target": {
            "member_type": case["member_type"],
            "section": case["section"],
            "material": "steel",
        },
        "issue_id": "golden",
        "candidate_id": "",
    }
    res = retriever.retrieve(make_kds_query(ctx), top_k=TOP_K)
    return [(c.standard_id, c.clause_id) for c in res.chunks]


_RETRIEVER = _build_inmemory_from_corpus()
_CASES = _load_cases()


def test_corpus_loaded_and_all_kds():
    """Sanity: corpus is non-empty and carries no AISC proxy chunk anymore."""
    assert _RETRIEVER.num_chunks >= 6
    standards = {c.standard_id for c in _RETRIEVER._chunks}
    assert not any(s.upper().startswith("AISC") for s in standards), (
        f"AISC proxy chunk still in corpus: {standards}"
    )


@pytest.mark.parametrize("case", _CASES, ids=[c["id"] for c in _CASES])
def test_golden_case_expected_clause_in_topk(case):
    """recall@3: the expected clause for each scenario must appear in top-3."""
    got = _retrieve(_RETRIEVER, case)
    exp_std = case["expected_standard"]
    exp_clauses = set(case["expected_clauses"])
    hits = [(s, c) for s, c in got if s == exp_std and c in exp_clauses]
    assert hits, (
        f"[{case['id']}] expected {exp_std} {sorted(exp_clauses)} not in top-{TOP_K}: {got}"
    )


def test_golden_precision_at_1_full():
    """precision@1: every scenario's top-1 is an acceptable clause.

    Deterministic on InMemory today (5/5). If corpus growth pushes a
    distractor above an expected clause, this flags it so we tune tags /
    keywords or update the golden set deliberately.
    """
    misses = []
    for case in _CASES:
        got = _retrieve(_RETRIEVER, case)
        ok = bool(got) and got[0][0] == case["expected_standard"] \
            and got[0][1] in set(case["expected_clauses"])
        if not ok:
            misses.append((case["id"], got[:1]))
    assert not misses, f"precision@1 misses: {misses}"
