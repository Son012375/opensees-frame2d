"""Tests for the recommendation explainer module.

These tests do NOT touch OpenSees, FastAPI, or any retriever backend.
They lock down the deterministic explanation surface so a future LLM
provider cannot silently bypass the safety guarantees:

    - explanation has all 8 sections, always
    - no fabricated KDS clause numbers when evidence is empty
    - source.score_method reflects whether reanalysis verified the score
    - rejected / abstract / unverified candidates each produce a
      distinct, honest warning

When evidence IS supplied (via an in-memory retriever), the response
must include kds_evidence with the slim API shape and the source must
flip rag_used=True.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mcp-server"))

from core.kds_rag import (  # noqa: E402
    InMemoryKDSRetriever,
    KDSChunk,
    NoopKDSRetriever,
    NOOP_WARNING,
)
from core.recommendation import (  # noqa: E402
    CandidateEvaluation,
    KdsEvidence,
    RetrofitCandidate,
    ScoreBreakdown,
    STATUS_EVALUATED,
    STATUS_REJECTED_FAILED,
    STATUS_REJECTED_NEW_NG,
    STATUS_SKIPPED_INAPPLICABLE,
    VERIFIED_SCORE_METHOD,
    build_explanation_context,
    deterministic_explanation,
    explain_candidate,
)
from core.recommendation.apply_candidate import ChangeDiff  # noqa: E402
from core.recommendation.llm_explainer import (  # noqa: E402
    BaseExplanationLLMProvider,
    NoopExplanationLLMProvider,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

REQUIRED_KEYS = (
    "summary",
    "issue_interpretation",
    "recommended_change",
    "expected_structural_effect",
    "verified_result",
    "tradeoffs",
    "limitations",
    "next_user_decision",
)


def _applicable_candidate(**overrides) -> RetrofitCandidate:
    base = dict(
        candidate_id="cand_test_X",
        issue_id="strength_exceeded_member_1",
        action_type="increase_section",
        description="Bump column section to H-310x310",
        member_id=1,
        element_id=1,
        expected_effect="DCR ↓ ~0.1",
        tradeoffs="강재 약 0.5톤 증가",
        requires_reanalysis=True,
        confidence="medium",
        code_refs=[],
        target={"scope": "member", "member_id": 1, "element_id": 1,
                "member_type": "column", "story": 1},
        proposed_change={
            "operation": "replace_section",
            "from": "H-300x300",
            "to": "H-310x310",
            "requires_user_selection": False,
            "applicable": True,
            "reason": "strength_exceeded",
        },
        metadata={},
    )
    base.update(overrides)
    return RetrofitCandidate(**base)


def _abstract_candidate() -> RetrofitCandidate:
    return RetrofitCandidate(
        candidate_id="cand_abstract",
        issue_id="drift_exceeded_story_1",
        action_type="add_lateral_resistance",
        description="횡저항 시스템 추가 검토 필요",
        target={"scope": "story", "story": 1, "direction": "X"},
        proposed_change={
            "operation": "add_lateral_resistance",
            "from": None,
            "to": None,
            "requires_user_selection": True,
            "applicable": False,
            "reason": "drift_exceeded",
        },
        metadata={"abstract": True},
    )


def _evaluated_evaluation(*, status: str = STATUS_EVALUATED,
                          verified: bool = True) -> CandidateEvaluation:
    return CandidateEvaluation(
        candidate_id="cand_test_X",
        status=status,
        diff=ChangeDiff(
            candidate_id="cand_test_X",
            operation="replace_section",
            reason="strength_exceeded",
            changed_members=[{
                "element_id": 1, "member_id": 1, "member_label": "C1@story1",
                "section_from": "H-300x300", "section_to": "H-310x310",
                "story": 1, "reason": "strength_exceeded",
            }],
        ),
        metrics={
            "max_interaction_ratio": 0.85,
            "max_drift_ratio": 0.005,
            "ng_member_count": 0,
            "ng_drift_count": 0,
            "weight_proxy": 5e5,
            "changed_member_count": 1,
            "analysis_succeeded": True,
        },
        improvement={"dcr_delta": -0.55, "drift_delta": 0.0,
                     "ng_member_delta": -1, "ng_drift_delta": 0},
        score=ScoreBreakdown(
            safety_gain=1.0, code_compliance=1.0, relative_cost=0.99,
            disruption=0.5, side_effect_risk=1.0, total=0.85,
            method=VERIFIED_SCORE_METHOD, verified=verified,
            status="verified",
        ),
    )


def _drift_chunks() -> list[KDSChunk]:
    return [
        KDSChunk(
            chunk_id="kds_drift_001",
            standard_id="KDS 41 17 00",
            text=(
                "허용 층간변위비는 내진등급에 따라 0.010 hsx (특), "
                "0.015 hsx (I), 0.020 hsx (II)로 한다."
            ),
            clause_id="§8.2.3",
            title="허용 층간변위비",
            topic="story_drift",
            limit_state="drift_limit",
            jurisdiction="KDS",
            source_url=None,
            material=None,
        ),
        KDSChunk(
            chunk_id="kds_drift_002",
            standard_id="KDS 41 17 00",
            text=(
                "지진하중에 의한 비탄성 층간변위는 Cd × δ_elastic / IE 로 "
                "계산하여 허용 층간변위비와 비교한다."
            ),
            clause_id="§8.2.4",
            title="비탄성 층간변위 산정",
            topic="story_drift",
            limit_state="drift_limit",
            jurisdiction="KDS",
            source_url=None,
            material=None,
        ),
    ]


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

class TestBuildExplanationContext:
    def test_context_has_core_fields(self):
        cand = _applicable_candidate()
        ctx = build_explanation_context(cand)
        for key in (
            "candidate_id", "issue_id", "action_type", "target",
            "proposed_change", "applicable", "operation", "diff_summary",
            "evaluation", "verified", "warnings", "issue_type",
        ):
            assert key in ctx, f"missing context key: {key}"
        assert ctx["candidate_id"] == "cand_test_X"
        assert ctx["operation"] == "replace_section"
        assert ctx["applicable"] is True

    def test_context_extracts_issue_type_from_issue_id(self):
        cand = _applicable_candidate()
        ctx = build_explanation_context(cand)
        assert ctx["issue_type"] == "strength_exceeded"

    def test_diff_summary_extracts_changed_members(self):
        cand = _applicable_candidate()
        ev = _evaluated_evaluation()
        ctx = build_explanation_context(cand, diff=ev.diff, evaluation=ev)
        ds = ctx["diff_summary"]
        assert ds is not None
        assert ds["operation"] == "replace_section"
        assert ds["changed_member_count"] == 1
        assert "H-300x300" in ds["sections_from"]
        assert "H-310x310" in ds["sections_to"]

    def test_missing_evaluation_emits_warning(self):
        cand = _applicable_candidate()
        ctx = build_explanation_context(cand)
        assert any("evaluation_missing" in w for w in ctx["warnings"])
        assert ctx["verified"] is False

    def test_abstract_candidate_marks_warning_and_skips_applicable(self):
        cand = _abstract_candidate()
        ctx = build_explanation_context(cand)
        assert ctx["applicable"] is False
        assert any("abstract_candidate" in w for w in ctx["warnings"])

    def test_evaluated_candidate_marks_verified(self):
        cand = _applicable_candidate()
        ev = _evaluated_evaluation()
        ctx = build_explanation_context(cand, evaluation=ev)
        assert ctx["verified"] is True
        assert ctx["score_method"] == VERIFIED_SCORE_METHOD


# ---------------------------------------------------------------------------
# Deterministic explanation guarantees
# ---------------------------------------------------------------------------

class TestDeterministicExplanation:
    def test_fallback_returns_all_required_sections(self):
        cand = _applicable_candidate()
        ctx = build_explanation_context(cand)
        result = deterministic_explanation(ctx, evidence=[])
        for k in REQUIRED_KEYS:
            assert k in result.explanation, f"missing section: {k}"
            assert isinstance(result.explanation[k], str)
            assert result.explanation[k].strip(), f"empty section: {k}"

    def test_no_evidence_warns_kds_unavailable(self):
        cand = _applicable_candidate()
        ctx = build_explanation_context(cand)
        result = deterministic_explanation(ctx, evidence=[])
        joined = " | ".join(result.warnings)
        assert "kds_evidence_missing" in joined
        assert result.kds_evidence == []

    def test_deterministic_does_not_invent_kds_clauses(self):
        """Guardrail: no "KDS XX YY ZZ" should appear in any section when
        evidence is empty. This prevents future regressions where someone
        bakes a code reference into the template."""
        cand = _applicable_candidate()
        ctx = build_explanation_context(cand)
        result = deterministic_explanation(ctx, evidence=[])
        clause_re = re.compile(r"KDS\s+\d{2}\s+\d{2}\s+\d{2}")
        for k, v in result.explanation.items():
            assert not clause_re.search(v), (
                f"deterministic section {k!r} invented a KDS clause: {v!r}"
            )

    def test_verified_evaluation_sets_score_method_and_text(self):
        cand = _applicable_candidate()
        ev = _evaluated_evaluation()
        ctx = build_explanation_context(cand, diff=ev.diff, evaluation=ev)
        result = deterministic_explanation(ctx, evidence=[])
        assert result.source["score_method"] == VERIFIED_SCORE_METHOD
        # Korean engineer-brief mentions reanalysis verification.
        assert "재해석" in result.explanation["verified_result"]

    def test_missing_evaluation_text_says_not_verified(self):
        cand = _applicable_candidate()
        ctx = build_explanation_context(cand)
        result = deterministic_explanation(ctx, evidence=[])
        joined = result.explanation["verified_result"]
        assert "재해석" in joined and ("수행" in joined or "검증" in joined)

    def test_rejected_new_ng_evaluation_warns_against_apply(self):
        cand = _applicable_candidate()
        ev = _evaluated_evaluation(status=STATUS_REJECTED_NEW_NG)
        ctx = build_explanation_context(cand, evaluation=ev)
        result = deterministic_explanation(ctx, evidence=[])
        joined = (
            result.explanation["next_user_decision"]
            + result.explanation["summary"]
        )
        assert "금지" in joined or "거부" in joined

    def test_failed_evaluation_surfaces_error(self):
        cand = _applicable_candidate()
        ev = CandidateEvaluation(
            candidate_id="cand_test_X",
            status=STATUS_REJECTED_FAILED,
            error="ValueError: singular stiffness",
        )
        ctx = build_explanation_context(cand, evaluation=ev)
        result = deterministic_explanation(ctx, evidence=[])
        assert "재해석 실패" in result.explanation["verified_result"]

    def test_abstract_candidate_says_manual_review(self):
        cand = _abstract_candidate()
        ctx = build_explanation_context(cand)
        result = deterministic_explanation(ctx, evidence=[])
        joined = (
            result.explanation["summary"]
            + " " + result.explanation["next_user_decision"]
            + " " + result.explanation["limitations"]
        )
        assert "수동" in joined or "엔지니어" in joined

    def test_confidence_levels(self):
        cand = _applicable_candidate()
        ev = _evaluated_evaluation()

        # neither verified nor evidence → low
        ctx0 = build_explanation_context(cand)
        r0 = deterministic_explanation(ctx0, evidence=[])
        assert r0.confidence == "low"

        # verified only → medium
        ctx1 = build_explanation_context(cand, evaluation=ev)
        r1 = deterministic_explanation(ctx1, evidence=[])
        assert r1.confidence == "medium"

        # verified + 2 evidence → high
        ev_list = [
            KdsEvidence(
                doc_id="KDS 41 17 00", title="Drift", clause="§8.2.3",
                quote="...허용 층간변위...", relevance="match", score=0.9,
            ),
            KdsEvidence(
                doc_id="KDS 41 17 00", title="Cd", clause="§8.2.4",
                quote="...비탄성 변위 Cd ...", relevance="match", score=0.7,
            ),
        ]
        r2 = deterministic_explanation(ctx1, evidence=ev_list)
        assert r2.confidence == "high"


# ---------------------------------------------------------------------------
# explain_candidate orchestrator
# ---------------------------------------------------------------------------

class TestExplainCandidateOrchestrator:
    def test_runs_without_retriever_or_llm(self):
        cand = _applicable_candidate()
        ev = _evaluated_evaluation()
        result = explain_candidate(cand, diff=ev.diff, evaluation=ev)
        assert set(result.explanation.keys()) >= set(REQUIRED_KEYS)
        assert result.source["rag_used"] is False
        assert result.source["llm_used"] is False
        assert result.source["score_method"] == VERIFIED_SCORE_METHOD

    def test_noop_retriever_records_unavailable_warning(self):
        cand = _applicable_candidate()
        ev = _evaluated_evaluation()
        result = explain_candidate(
            cand, diff=ev.diff, evaluation=ev,
            retriever=NoopKDSRetriever(),
        )
        assert result.source["rag_used"] is False
        joined = " | ".join(result.warnings)
        assert NOOP_WARNING in joined

    def test_inmemory_retriever_populates_evidence(self):
        cand = RetrofitCandidate(
            candidate_id="cand_drift",
            issue_id="drift_exceeded_story_1",
            action_type="add_lateral_resistance",
            description="횡저항 시스템 검토",
            target={"scope": "story", "story": 1, "direction": "X"},
            proposed_change={
                "operation": "add_lateral_resistance",
                "applicable": False,
            },
        )
        retriever = InMemoryKDSRetriever(_drift_chunks())
        result = explain_candidate(cand, retriever=retriever)
        assert result.source["rag_used"] is True
        assert len(result.kds_evidence) >= 1
        ev0 = result.kds_evidence[0]
        assert ev0.doc_id == "KDS 41 17 00"
        assert ev0.clause and ev0.quote
        # Score is taken from the retriever's scores dict.
        assert ev0.score > 0.0

    def test_llm_provider_failure_falls_back_to_deterministic(self):
        cand = _applicable_candidate()
        ev = _evaluated_evaluation()

        class FailingProvider(BaseExplanationLLMProvider):
            def generate(self, **_):
                raise RuntimeError("boom")

        result = explain_candidate(
            cand, diff=ev.diff, evaluation=ev,
            llm_provider=FailingProvider(),
        )
        assert result.source["llm_used"] is False
        joined = " | ".join(result.warnings)
        assert "llm_provider_failed" in joined

    def test_noop_llm_provider_does_not_break_explain(self):
        cand = _applicable_candidate()
        result = explain_candidate(
            cand, llm_provider=NoopExplanationLLMProvider(),
        )
        assert result.source["llm_used"] is False
        joined = " | ".join(result.warnings)
        assert "llm_provider_failed" in joined

    def test_retriever_exception_becomes_warning_not_failure(self):
        cand = _applicable_candidate()

        class BoomRetriever:
            def retrieve(self, *args, **kwargs):
                raise RuntimeError("retriever crashed")

        result = explain_candidate(cand, retriever=BoomRetriever())
        assert isinstance(result.explanation, dict)
        joined = " | ".join(result.warnings)
        assert "kds_retriever_exception" in joined

    def test_aisc_chunk_emits_temporary_reference_warning(self):
        # When the retriever returns AISC chunks (stand-ins until KDS
        # 14 31 00 / 41 31 00 are ingested), the explainer must warn the
        # user that the citation is temporary. The warning carries the
        # ``aisc_temporary_reference`` code so the UI translation layer
        # surfaces a Korean disclaimer instead of silently citing AISC
        # as if it were the Korean code.
        cand = _applicable_candidate()  # strength_exceeded candidate
        aisc_chunks = [
            KDSChunk(
                chunk_id="aisc_f2_001",
                standard_id="AISC 360-22",
                text=(
                    "F2 휨강도 — 양축대칭 콤팩트 I형 부재. "
                    "Mn = Mp = Fy · Zx; 조합응력 H1-1a/H1-1b 의 DCR ≤ 1.0."
                ),
                clause_id="F2",
                title="Flexural Strength of Doubly Symmetric Compact I-Shaped Members",
                topic="member_strength",
                limit_state="strength",
                jurisdiction="AISC",
                material="steel",
                source_url=None,
            ),
        ]
        retriever = InMemoryKDSRetriever(aisc_chunks)
        result = explain_candidate(cand, retriever=retriever)
        # rag_used must still be True — the retriever ran and returned a
        # validated chunk; the warning is advisory, not a failure mode.
        assert result.source["rag_used"] is True
        assert len(result.kds_evidence) >= 1
        joined = " | ".join(result.warnings)
        assert "aisc_temporary_reference" in joined, (
            f"expected aisc_temporary_reference in warnings; got: {joined!r}"
        )
        # Standard id is interpolated into the disclaimer so a future
        # second AISC standard (e.g. AISC 341) shows up separately.
        assert "AISC 360-22" in joined

    def test_aisc_warning_dedupes_per_standard_id(self):
        # Two chunks from the same AISC 360-22 standard must produce a
        # SINGLE warning line. Multiple AISC standards in one response
        # would each produce their own line, but per-chunk dedupe within
        # a single standard keeps the warning panel clean.
        cand = _applicable_candidate()
        aisc_chunks = [
            KDSChunk(
                chunk_id="aisc_f2_001",
                standard_id="AISC 360-22",
                text="F2 휨 — Mn = Mp = Fy · Zx.",
                clause_id="F2",
                title="Flexure",
                topic="member_strength",
                limit_state="strength",
                jurisdiction="AISC",
                material="steel",
                source_url=None,
            ),
            KDSChunk(
                chunk_id="aisc_f2_002",
                standard_id="AISC 360-22",
                text="F2-2 횡-비틀림좌굴 보정계수 Cb 적용.",
                clause_id="F2",
                title="LTB",
                topic="member_strength",
                limit_state="strength",
                jurisdiction="AISC",
                material="steel",
                source_url=None,
            ),
        ]
        retriever = InMemoryKDSRetriever(aisc_chunks)
        result = explain_candidate(cand, retriever=retriever)
        aisc_warnings = [
            w for w in result.warnings
            if w.startswith("aisc_temporary_reference")
        ]
        assert len(aisc_warnings) == 1, (
            f"expected exactly one AISC warning (per-standard dedupe); "
            f"got {len(aisc_warnings)}: {aisc_warnings!r}"
        )

    def test_kds_only_evidence_omits_aisc_warning(self):
        # False-positive guard: a pure KDS response must NOT emit the AISC
        # disclaimer. The drift chunks set jurisdiction="KDS" and
        # standard_id="KDS 41 17 00" — neither trigger the AISC heuristic.
        cand = RetrofitCandidate(
            candidate_id="cand_drift",
            issue_id="drift_exceeded_story_1",
            action_type="add_lateral_resistance",
            description="횡저항 시스템 검토",
            target={"scope": "story", "story": 1, "direction": "X"},
            proposed_change={
                "operation": "add_lateral_resistance",
                "applicable": False,
            },
        )
        retriever = InMemoryKDSRetriever(_drift_chunks())
        result = explain_candidate(cand, retriever=retriever)
        assert result.source["rag_used"] is True
        joined = " | ".join(result.warnings)
        assert "aisc_temporary_reference" not in joined, (
            f"AISC warning leaked onto pure-KDS evidence; warnings={joined!r}"
        )

    def test_to_dict_contract(self):
        cand = _applicable_candidate()
        ev = _evaluated_evaluation()
        result = explain_candidate(cand, diff=ev.diff, evaluation=ev)
        d = result.to_dict()
        assert d["candidate_id"] == cand.candidate_id
        assert isinstance(d["explanation"], dict)
        assert isinstance(d["kds_evidence"], list)
        assert d["confidence"] in {"low", "medium", "high"}
        for k in ("deterministic", "rag_used", "llm_used", "score_method"):
            assert k in d["source"]
