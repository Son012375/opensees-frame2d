"""Deterministic retrieval-quality regression for the Voyage KDS path.

Phase 3B smoke deliberately keeps three issue_type query paths covered:
``drift_exceeded`` / ``shear_exceeded`` / ``strength_exceeded``. This test
locks in the *retrieval ranking* for those three so a future change to
the chunker, query builder, or cosine fast-path can't silently flip the
top-1 chunk and drag KDS evidence in the explainer with it.

What we test
------------
* The full Voyage code path (``VoyageKDSRetriever.retrieve``) — embed
  query → cosine → (optional) rerank → top-k — but with an in-process
  topic-aware fake embedder so no network is touched. The same path runs
  in production behind ``get_default_kds_retriever()``.
* ``make_kds_query`` is invoked end-to-end so the contextual mapping
  (issue_type → keywords) is exercised, not just the bare retriever.

What we deliberately do NOT test here
-------------------------------------
* Voyage SDK error paths — covered by ``test_kds_voyage_rag.py``.
* The deterministic explainer prose — covered by
  ``test_recommendation_explainer.py``.
* PDF parsing — Phase 4 work; not in scope for this smoke.

The fake embedder
-----------------
We map each chunk to a fixed unit basis vector keyed off ``topic``:

    story_drift     → e_drift   = [1, 0, 0, 0]
    member_shear    → e_shear   = [0, 1, 0, 0]
    member_strength → e_strength= [0, 0, 1, 0]
    (anything else) → e_noise   = [0, 0, 0, 1]

The query vector is derived the same way from the *expected* topic for
each issue_type, with a small amount of noise sprinkled in so the test
also asserts that the top-1 wins by a margin rather than by tie-break.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mcp-server"))

from core.kds_rag import (  # noqa: E402
    KDSRetrievalQuery,
    make_kds_query,
)
from core.kds_rag.voyage_retriever import VoyageKDSRetriever  # noqa: E402


# ---------------------------------------------------------------------------
# Topic-aware fake embedder
# ---------------------------------------------------------------------------

# Basis vectors. The order [drift, shear, strength, noise] is fixed across
# index and query embedders so cosine actually discriminates topic.
_TOPIC_BASIS: dict[str, list[float]] = {
    "story_drift":     [1.0, 0.0, 0.0, 0.05],
    "member_shear":    [0.0, 1.0, 0.0, 0.05],
    "member_strength": [0.0, 0.0, 1.0, 0.05],
}

# Issue_type → topic the query should be routed to. Mirrors
# ``core.kds_rag.pipeline.ISSUE_TYPE_TO_TOPIC`` but kept local so a future
# rename of that map fails this test loudly (and you remember to update
# the synthetic basis at the same time).
_EXPECTED_TOPIC_FOR_ISSUE: dict[str, str] = {
    "drift_exceeded":    "story_drift",
    "shear_exceeded":    "member_shear",
    "strength_exceeded": "member_strength",
}


class _TopicAwareEmbedder:
    """Deterministic fake embedder that separates by KDSChunk.topic.

    The chunk index is built directly off ``topic``; the query vector is
    inferred from ``query.topic`` (which ``make_kds_query`` already
    populates via the issue_type → topic mapping). When ``query.topic``
    is unset, we fall back to a keyword scan over ``query_text``. The
    fall-back is what guarantees the test still pings the right basis
    even if ``ISSUE_TYPE_TO_TOPIC`` drops a key.
    """

    model = "fake-topic-aware"

    def __init__(self):
        self.embed_query_calls = 0
        self.embed_doc_calls = 0
        # Last-seen query for assertion convenience.
        self.last_query_text: str = ""

    @staticmethod
    def _topic_from_text(text: str) -> str:
        t = (text or "").lower()
        if "층간변위" in t or "drift" in t:
            return "story_drift"
        if "전단" in t or "shear" in t:
            return "member_shear"
        if "강도" in t or "조합응력" in t or "interaction" in t or "단면" in t:
            return "member_strength"
        return "noise"

    def _vec_for_topic(self, topic: str) -> list[float]:
        return list(_TOPIC_BASIS.get(topic, [0.05, 0.05, 0.05, 1.0]))

    def embed_documents(self, texts):
        self.embed_doc_calls += 1
        # Documents are never embedded at runtime in this test path (the
        # index is written directly), but the contract is preserved so a
        # future build_kds_index call can reuse this class unchanged.
        return [self._vec_for_topic(self._topic_from_text(t)) for t in texts]

    def embed_query(self, text):
        self.embed_query_calls += 1
        self.last_query_text = text
        return self._vec_for_topic(self._topic_from_text(text))


# ---------------------------------------------------------------------------
# Synthetic chunks — minimal, topic-tagged, one chunk per issue path
# ---------------------------------------------------------------------------

def _synthetic_chunks() -> list[dict]:
    """Three orthogonal chunks, each clearly tagged with topic.

    Vectors are aligned with ``_TOPIC_BASIS`` so a query vector pointing
    at one topic strictly outranks the others by cosine alone — no
    reranker needed.
    """
    return [
        {
            "chunk_id": "syn_drift_0",
            "standard_id": "KDS 41 17 00",
            "title": "허용 층간변위비",
            "clause_id": "8.2.3",
            "topic": "story_drift",
            "limit_state": "drift_limit",
            "jurisdiction": "KDS",
            "text": (
                "비탄성 층간변위 Δ = Cd · δ_xe / IE 는 표 8.2-1 의 허용 "
                "층간변위비를 초과해서는 안 된다. 특/I/II 등급별로 0.010·hsx, "
                "0.015·hsx, 0.020·hsx 가 적용된다."
            ),
            "embedding": list(_TOPIC_BASIS["story_drift"]),
        },
        {
            "chunk_id": "syn_shear_0",
            "standard_id": "AISC 360-22",
            "title": "Shear Strength of I-Shaped Members",
            "clause_id": "G2",
            "topic": "member_shear",
            "limit_state": "shear_strength",
            "material": "steel",
            "jurisdiction": "AISC",
            "text": (
                "공칭 전단강도 Vn = 0.6 · Fy · Aw · Cv1. 전단 DCR Vu/(φv·Vn) 가 "
                "1.0 을 초과하면 단면 교체, 횡보강재, 더블러 플레이트 등으로 보강한다."
            ),
            "embedding": list(_TOPIC_BASIS["member_shear"]),
        },
        {
            "chunk_id": "syn_strength_0",
            "standard_id": "AISC 360-22",
            "title": "Flexural Strength of Compact I-Shaped Members",
            "clause_id": "F2",
            "topic": "member_strength",
            "limit_state": "strength",
            "material": "steel",
            "jurisdiction": "AISC",
            "text": (
                "양축대칭 콤팩트 I형 부재의 휨강도 Mn 은 항복 (Mp = Fy·Zx) 과 "
                "횡-비틀림좌굴 한계상태의 작은 값. 조합응력 H1-1a/H1-1b 의 "
                "DCR 이 1.0 을 초과하면 단면 교체 또는 등급 상향 검토."
            ),
            "embedding": list(_TOPIC_BASIS["member_strength"]),
        },
    ]


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")


@pytest.fixture
def synthetic_retriever(tmp_path):
    """A fully-wired VoyageKDSRetriever over the three synthetic chunks.

    Embedder/reranker are injected — no Voyage SDK or network needed.
    """
    idx_path = tmp_path / "kds_synth_index.jsonl"
    _write_jsonl(idx_path, _synthetic_chunks())
    embedder = _TopicAwareEmbedder()
    retriever = VoyageKDSRetriever(
        index_path=str(idx_path),
        embedding_client=embedder,
        reranker=None,        # No rerank — pure cosine quality check
        rerank_model=None,
    )
    return retriever, embedder


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRetrievalQualityTop1:
    """Each issue_type returns its topic-matched chunk as top-1."""

    def test_drift_query_returns_drift_chunk_top1(self, synthetic_retriever):
        retriever, embedder = synthetic_retriever
        ctx = {
            "candidate_id": "syn_c_drift",
            "issue_id": "iss_drift_exceeded_story_1",
            "issue_type": "drift_exceeded",
            "action_type": "add_lateral_resistance",
            "target": {"member_type": "column"},
            "proposed_change": {"applicable": False},
        }
        q = make_kds_query(ctx)
        assert q.topic == "story_drift"

        result = retriever.retrieve(q, top_k=3)
        assert not result.is_empty, (
            f"retrieval returned 0 chunks for drift; warnings={result.warnings}"
        )
        assert result.chunks[0].chunk_id == "syn_drift_0", (
            f"expected syn_drift_0 top-1, got "
            f"{[c.chunk_id for c in result.chunks]}"
        )
        # Top-1 score should clearly beat the runner-up (no tie).
        s0 = result.scores[result.chunks[0].chunk_id]
        s_others = [result.scores[c.chunk_id] for c in result.chunks[1:]]
        assert all(s0 > s for s in s_others), (
            f"drift top-1 ({s0}) did not strictly outscore runners-up "
            f"({s_others}); scores={result.scores}"
        )
        # Embedder was actually called (no accidental cache short-circuit).
        assert embedder.embed_query_calls == 1

    def test_shear_query_returns_shear_chunk_top1(self, synthetic_retriever):
        retriever, embedder = synthetic_retriever
        ctx = {
            "candidate_id": "syn_c_shear",
            "issue_id": "iss_shear_exceeded_b1",
            "issue_type": "shear_exceeded",
            "action_type": "replace_section",
            "target": {"member_type": "beam"},
            "proposed_change": {
                "operation": "replace_section",
                "from": "H-300x150",
                "to": "H-400x200",
                "applicable": True,
            },
        }
        q = make_kds_query(ctx)
        # No ISSUE_TYPE_TO_TOPIC entry for shear_exceeded historically →
        # fall back path. We still expect the retriever to rank by text.
        # (make_kds_query may set topic=None — that's fine; embedder
        # falls back to keyword scan over query_text.)
        result = retriever.retrieve(q, top_k=3)
        assert not result.is_empty, (
            f"retrieval returned 0 chunks for shear; warnings={result.warnings}"
        )
        assert result.chunks[0].chunk_id == "syn_shear_0", (
            f"expected syn_shear_0 top-1, got "
            f"{[c.chunk_id for c in result.chunks]}; "
            f"query_text={q.query_text!r}"
        )
        s0 = result.scores[result.chunks[0].chunk_id]
        s_others = [result.scores[c.chunk_id] for c in result.chunks[1:]]
        assert all(s0 > s for s in s_others), (
            f"shear top-1 ({s0}) did not strictly outscore runners-up "
            f"({s_others}); scores={result.scores}"
        )

    def test_strength_query_returns_strength_chunk_top1(self, synthetic_retriever):
        retriever, embedder = synthetic_retriever
        ctx = {
            "candidate_id": "syn_c_strength",
            "issue_id": "iss_strength_exceeded_c2",
            "issue_type": "strength_exceeded",
            "action_type": "replace_section",
            "target": {"member_type": "column"},
            "proposed_change": {
                "operation": "replace_section",
                "from": "H-300x150",
                "to": "H-400x400",
                "applicable": True,
            },
        }
        q = make_kds_query(ctx)
        assert q.topic == "member_strength"

        result = retriever.retrieve(q, top_k=3)
        assert not result.is_empty, (
            f"retrieval returned 0 chunks for strength; "
            f"warnings={result.warnings}"
        )
        assert result.chunks[0].chunk_id == "syn_strength_0", (
            f"expected syn_strength_0 top-1, got "
            f"{[c.chunk_id for c in result.chunks]}; "
            f"query_text={q.query_text!r}"
        )
        s0 = result.scores[result.chunks[0].chunk_id]
        s_others = [result.scores[c.chunk_id] for c in result.chunks[1:]]
        assert all(s0 > s for s in s_others), (
            f"strength top-1 ({s0}) did not strictly outscore runners-up "
            f"({s_others}); scores={result.scores}"
        )


class TestRetrievalQualityCoverage:
    """Sanity-check the test scaffolding itself.

    These tests guard against silent drift in the fake embedder + chunk
    set — if someone removes one of the synthetic chunks or changes its
    topic, the top-1 tests still need to fail loudly rather than passing
    by accident.
    """

    def test_synthetic_chunk_set_is_complete(self):
        chunks = _synthetic_chunks()
        topics = {c["topic"] for c in chunks}
        assert topics == {"story_drift", "member_shear", "member_strength"}

    def test_basis_vectors_are_orthogonal(self):
        # The first three coordinates form an orthogonal basis; the noise
        # axis is shared (0.05) so cosine still discriminates strongly.
        from core.kds_rag.voyage_retriever import _cosine
        for a, b in (("story_drift", "member_shear"),
                     ("story_drift", "member_strength"),
                     ("member_shear", "member_strength")):
            sim = _cosine(_TOPIC_BASIS[a], _TOPIC_BASIS[b])
            # Only the noise axis overlaps → cosine ≈ 0.0025/1.0025 ≈ 0.0025
            assert sim < 0.05, (
                f"basis vectors {a} vs {b} are too parallel: cos={sim:.4f}"
            )

    def test_expected_topic_map_matches_pipeline(self):
        # If ``ISSUE_TYPE_TO_TOPIC`` drifts, the synthetic basis becomes
        # misleading and the top-1 tests would assert on the wrong target.
        from core.kds_rag.pipeline import ISSUE_TYPE_TO_TOPIC
        for issue_type, expected_topic in _EXPECTED_TOPIC_FOR_ISSUE.items():
            pipeline_topic = ISSUE_TYPE_TO_TOPIC.get(issue_type)
            if pipeline_topic is None:
                # pipeline left it unset → embedder falls back to keyword
                # scan, which is fine. But the basis we synthesize for the
                # query side must still equal the chunk side's topic.
                continue
            assert pipeline_topic == expected_topic, (
                f"ISSUE_TYPE_TO_TOPIC[{issue_type!r}]={pipeline_topic!r} "
                f"diverges from this test's expected topic "
                f"{expected_topic!r}; update both together."
            )
