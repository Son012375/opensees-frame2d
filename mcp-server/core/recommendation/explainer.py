"""Explanation layer for retrofit recommendations.

The deterministic Python pipeline (issue extraction → candidate generation →
apply → reanalysis → verified scoring) stays the source of truth. This
module is the *explanation* layer: it takes a verified (or unverified)
candidate plus optional diff/evaluation/KDS evidence and produces a
human-readable engineer-brief narrative.

The LLM/RAG layer (if wired) is constrained to:
    - explain deterministic results
    - cite KDS clauses ONLY from supplied evidence
    - flag missing or weak evidence honestly

It is NEVER allowed to:
    - generate new structural candidates
    - mutate model_json
    - rerank verified scores
    - invent KDS clause numbers

This module enforces those invariants by construction: ``explain_candidate``
runs deterministic_explanation as a fallback, and the only way an LLM
can override the text is by being explicitly wired *and* returning a
schema-compatible dict. Validation of LLM citations against retrieved
evidence happens at the call-site (citation validator).
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

from ..kds_rag import (
    KDSRetriever,
    NOOP_WARNING,
    chunk_to_code_reference,
    make_kds_query,
    validate_code_reference,
)
from .apply_candidate import (
    ChangeDiff,
    InapplicableCandidateError,
    apply_candidate_to_model,
)
from .evaluator import (
    STATUS_EVALUATED,
    STATUS_REJECTED_FAILED,
    STATUS_REJECTED_NEW_NG,
    STATUS_SKIPPED_INAPPLICABLE,
    VERIFIED_SCORE_METHOD,
)
from .schemas import CodeReference, RetrofitCandidate


# ---------------------------------------------------------------------------
# Slim API-boundary types
# ---------------------------------------------------------------------------

@dataclass
class KdsEvidence:
    """Slim citation object for the /explain API boundary.

    Internally we manipulate :class:`CodeReference` (which carries hint
    fields + provenance). This dataclass is the JSON contract the
    frontend reads — keep it small and stable.
    """
    doc_id: str
    title: str
    clause: Optional[str]
    quote: str
    relevance: str
    score: float

    @classmethod
    def from_code_reference(
        cls,
        ref: CodeReference,
        *,
        score: float = 1.0,
    ) -> "KdsEvidence":
        return cls(
            doc_id=ref.standard_id or "",
            title=ref.title or "",
            clause=ref.clause_id,
            quote=ref.quote or "",
            relevance=ref.relevance_reason or "",
            score=float(score),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RecommendationExplanation:
    """Top-level result of :func:`explain_candidate`. Serializable as dict."""

    candidate_id: str
    explanation: dict[str, str]
    kds_evidence: list[KdsEvidence] = field(default_factory=list)
    confidence: str = "low"
    source: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "explanation": dict(self.explanation),
            "kds_evidence": [e.to_dict() for e in self.kds_evidence],
            "confidence": self.confidence,
            "source": dict(self.source),
            "warnings": list(self.warnings),
        }


# ---------------------------------------------------------------------------
# Context construction
# ---------------------------------------------------------------------------

_EVAL_STATUS_LABEL = {
    STATUS_EVALUATED: "재해석 검증 통과",
    STATUS_SKIPPED_INAPPLICABLE: "자동 적용 불가 — 수동 검토 필요",
    STATUS_REJECTED_FAILED: "재해석 실패",
    STATUS_REJECTED_NEW_NG: "재해석 결과 새로운 NG 발생 — 적용 거부",
}


def _coerce_diff_dict(diff: Any) -> Optional[dict[str, Any]]:
    if diff is None:
        return None
    if isinstance(diff, ChangeDiff):
        return diff.to_dict()
    if isinstance(diff, dict):
        return diff
    return None


def _coerce_evaluation_dict(evaluation: Any) -> Optional[dict[str, Any]]:
    if evaluation is None:
        return None
    if isinstance(evaluation, dict):
        return evaluation
    if hasattr(evaluation, "to_dict"):
        return evaluation.to_dict()
    return None


def _summarize_diff(diff_dict: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if not diff_dict:
        return None
    members = diff_dict.get("changed_members") or []
    sections_to: list[str] = []
    sections_from: list[str] = []
    member_labels: list[str] = []
    stories: set[int] = set()
    for m in members:
        if not isinstance(m, dict):
            continue
        if m.get("section_from"):
            sections_from.append(str(m["section_from"]))
        if m.get("section_to"):
            sections_to.append(str(m["section_to"]))
        label = m.get("member_label") or m.get("member_id")
        if label is not None:
            member_labels.append(str(label))
        s = m.get("story")
        if isinstance(s, int):
            stories.add(s)
    return {
        "operation": diff_dict.get("operation"),
        "reason": diff_dict.get("reason"),
        "changed_member_count": diff_dict.get("changed_member_count")
        or len(members),
        "member_labels": member_labels[:6],
        "sections_from": sorted(set(sections_from)),
        "sections_to": sorted(set(sections_to)),
        "stories": sorted(stories),
    }


def build_explanation_context(
    candidate: RetrofitCandidate,
    *,
    diff: Any = None,
    evaluation: Any = None,
    analysis_summary: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build a deterministic, machine-readable explanation context.

    The returned dict feeds:
        * :func:`make_kds_query` (RAG query construction)
        * :func:`deterministic_explanation` (text generation)
        * any future LLM provider (which receives this verbatim)

    It is intentionally serializable and side-effect free.
    """
    diff_dict = _coerce_diff_dict(diff)
    eval_dict = _coerce_evaluation_dict(evaluation)

    proposed_change = dict(candidate.proposed_change or {})
    applicable = bool(proposed_change.get("applicable", True))
    operation = proposed_change.get("operation")

    warnings: list[str] = []
    if eval_dict is None:
        warnings.append("evaluation_missing: 재해석 검증이 아직 수행되지 않았습니다.")
    elif eval_dict.get("status") and eval_dict["status"] != STATUS_EVALUATED:
        status_label = _EVAL_STATUS_LABEL.get(
            eval_dict["status"], eval_dict["status"]
        )
        warnings.append(
            f"evaluation_status: 후보 평가 상태가 정상 통과가 아닙니다 — {status_label}."
        )
    if diff_dict is None and applicable:
        warnings.append(
            "diff_missing: 구조 변경 내용(diff)이 제공되지 않았습니다."
        )
    if not applicable:
        warnings.append(
            "abstract_candidate: 이 후보는 자동 적용이 불가하며 "
            "엔지니어의 수동 검토가 필요합니다."
        )

    score_method = "unknown"
    verified = False
    if eval_dict and eval_dict.get("score"):
        sc = eval_dict["score"]
        score_method = sc.get("method") or score_method
        verified = bool(sc.get("verified"))
    elif candidate.metadata and candidate.metadata.get("score"):
        sc = candidate.metadata["score"]
        score_method = sc.get("method") or score_method
        verified = bool(sc.get("verified"))

    context: dict[str, Any] = {
        "candidate_id": candidate.candidate_id,
        "issue_id": candidate.issue_id,
        "action_type": candidate.action_type,
        "description": candidate.description,
        "expected_effect": candidate.expected_effect,
        "tradeoffs": candidate.tradeoffs,
        "target": dict(candidate.target or {}),
        "proposed_change": proposed_change,
        "applicable": applicable,
        "operation": operation,
        "diff_summary": _summarize_diff(diff_dict),
        "evaluation": {
            "status": eval_dict.get("status") if eval_dict else None,
            "status_label": _EVAL_STATUS_LABEL.get(
                eval_dict.get("status") if eval_dict else "",
                "미평가",
            ),
            "metrics": dict(eval_dict.get("metrics") or {}) if eval_dict else {},
            "improvement": (
                dict(eval_dict.get("improvement") or {}) if eval_dict else {}
            ),
            "score": dict(eval_dict.get("score") or {}) if eval_dict else {},
            "error": eval_dict.get("error") if eval_dict else None,
        },
        "score_method": score_method,
        "verified": verified,
        "analysis_summary": dict(analysis_summary or {}),
        "warnings": warnings,
    }

    # Helpful inferred fields for make_kds_query() — pull issue_type
    # from the issue_id slug. The slug commonly carries an "iss_" prefix
    # (issue_extractor's make_issue_id) so strip it before matching.
    issue_id = candidate.issue_id or ""
    issue_type = None
    trimmed = issue_id[4:] if issue_id.startswith("iss_") else issue_id
    for known in (
        "drift_exceeded",
        "strength_exceeded",
        "shear_exceeded",
        "missing_design_check",
        "analysis_warning",
        "compression_capacity",
        "tension_capacity",
    ):
        if trimmed.startswith(known):
            issue_type = known
            break
    context["issue_type"] = issue_type

    return context


# ---------------------------------------------------------------------------
# Deterministic explanation generator
# ---------------------------------------------------------------------------

# Korean section labels for the engineer_brief style.
_KO_SECTION_LABELS = {
    "summary": "요약",
    "issue_interpretation": "이슈 해석",
    "recommended_change": "권장 변경",
    "expected_structural_effect": "예상 구조 효과",
    "verified_result": "검증 결과",
    "tradeoffs": "트레이드오프",
    "limitations": "한계 및 미확보 근거",
    "next_user_decision": "사용자 의사결정",
}


# Korean labels for known issue types. Falls back to "미분류" when the
# explainer cannot infer the type from the issue_id slug.
_ISSUE_TYPE_LABELS_KO = {
    "strength_exceeded": "강도 초과",
    "shear_exceeded": "전단 초과",
    "drift_exceeded": "층간변위 초과",
    "missing_design_check": "설계 검토 누락",
    "analysis_warning": "해석 경고",
    "compression_capacity": "압축내력 부족",
    "tension_capacity": "인장내력 부족",
}

# Korean labels for ``proposed_change.operation`` enum values.
_OPERATION_LABELS_KO = {
    "replace_section": "부재 단면 교체",
    "replace_sections_by_story": "층 전체 단면 교체",
    "change_material": "재료 변경",
    "add_lateral_resistance": "횡저항 시스템 추가",
    "add_member": "부재 추가",
    "change_support": "지지조건 변경",
    "manual_review": "수동 검토",
}


def _localize_operation(op: Optional[str]) -> str:
    if not op:
        return "—"
    label = _OPERATION_LABELS_KO.get(op)
    return f"{label} ({op})" if label else op


def _short_candidate_label(cand_id: str) -> str:
    """Return a short human-readable label for a candidate slug.

    Phase 1 candidate ids are long deterministic slugs
    (e.g. ``cand_iss_shear_exceeded_m10_..._d9c5d7dca8``). Showing the
    full string in narrative text is noisy — we keep the trailing hash
    so the user can still cross-reference it with the rec card.
    """
    if not cand_id:
        return "이 후보"
    s = str(cand_id)
    if len(s) <= 28:
        return s
    return f"{s[:8]}…{s[-8:]}"


def _format_list(items, *, sep: str = ", ", empty: str = "—") -> str:
    """Join a list as a comma-separated string for display.

    Avoids leaking Python's list repr (``['H-300x150']``) into prose.
    """
    if not items:
        return empty
    return sep.join(str(x) for x in items)


_NEW_NG_KO = "재해석 결과 새 NG 발생"
_NEW_NG_EN = "introduced new NG"


def _fmt_ratio(v: Any) -> str:
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return "—"


def _fmt_delta(v: Any) -> str:
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return "—"
    sign = "+" if fv >= 0 else ""
    return f"{sign}{fv:.3f}"


def _engineer_brief_text(context: dict[str, Any],
                         evidence: list[KdsEvidence]) -> dict[str, str]:
    """Pure-Korean engineer-brief text. No LLM, no invented citations."""
    cand_id = context["candidate_id"]
    cand_label = _short_candidate_label(cand_id)
    action = context.get("action_type") or "—"
    operation = context.get("operation") or "—"
    target = context.get("target") or {}
    diff = context.get("diff_summary") or {}
    evaluation = context.get("evaluation") or {}
    metrics = evaluation.get("metrics") or {}
    improvement = evaluation.get("improvement") or {}
    score = evaluation.get("score") or {}
    verified = context.get("verified", False)
    applicable = context.get("applicable", True)
    status = evaluation.get("status")
    status_label = evaluation.get("status_label", "미평가")

    # ── summary ────────────────────────────────────────────────────────────
    if not applicable:
        summary = (
            "이 후보는 자동 적용 가능한 변경안이 아니라 "
            "엔지니어의 수동 검토가 필요한 추상 권고입니다."
        )
    elif status == STATUS_REJECTED_NEW_NG:
        summary = (
            "이 후보는 재해석 결과 새 NG가 발생하여 적용이 거부되었습니다. "
            "그대로 적용하지 마세요."
        )
    elif status == STATUS_REJECTED_FAILED:
        summary = (
            "이 후보는 재해석 단계에서 실패했습니다. "
            "원인 확인 후 재시도가 필요합니다."
        )
    elif verified:
        method = score.get("method", VERIFIED_SCORE_METHOD)
        summary = (
            f"이 후보는 재해석 기반 검증을 통과했습니다 (방식 {method})."
        )
    else:
        summary = (
            "이 후보는 결정론적 휴리스틱으로 도출되었으며 "
            "아직 재해석 검증이 수행되지 않았습니다."
        )

    # ── issue_interpretation ──────────────────────────────────────────────
    issue_type = context.get("issue_type")
    issue_type_label = _ISSUE_TYPE_LABELS_KO.get(issue_type or "", "미분류")
    issue_type_part = (
        f"{issue_type_label}" if issue_type is None
        else f"{issue_type_label} ({issue_type})"
    )

    _scope_label = {
        "member": "단일 부재",
        "story": "층 단위",
        "system": "시스템 단위",
    }
    _member_type_label = {
        "column": "기둥",
        "beam": "보",
        "beam_x": "X 방향 보",
        "beam_y": "Y 방향 보",
        "brace": "가새",
    }
    scope = target.get("scope")
    member_type = target.get("member_type")
    parts: list[str] = [f"이슈 유형: {issue_type_part}."]
    ctx_bits: list[str] = []
    if scope:
        ctx_bits.append(f"범위 {_scope_label.get(scope, scope)}")
    if member_type:
        ctx_bits.append(f"부재 유형 {_member_type_label.get(member_type, member_type)}")
    if target.get("member_id") is not None:
        ctx_bits.append(f"부재 ID {target['member_id']}")
    if target.get("story") is not None:
        ctx_bits.append(f"{target['story']}층")
    if ctx_bits:
        parts.append("대상 부재 컨텍스트 — " + ", ".join(ctx_bits) + ".")
    issue_interpretation = " ".join(parts)

    # ── recommended_change ────────────────────────────────────────────────
    op_label = _localize_operation(operation)
    if diff:
        ops_str = _format_list(diff.get("sections_from") or [])
        new_str = _format_list(diff.get("sections_to") or [])
        nchg = diff.get("changed_member_count", 0)
        stories = diff.get("stories") or []
        story_part = f", 층={_format_list(stories)}" if stories else ""
        recommended_change = (
            f"작업: {op_label}. 변경 부재 수 {nchg}건, "
            f"단면 {ops_str} → {new_str}{story_part}."
        )
    else:
        recommended_change = (
            f"작업: {op_label}. 구체적 변경 diff가 제공되지 않아 "
            "부재 단위 영향 범위는 산출되지 않았습니다."
        )

    # ── expected_structural_effect ────────────────────────────────────────
    if metrics:
        max_dcr = _fmt_ratio(metrics.get("max_interaction_ratio"))
        max_drift = _fmt_ratio(metrics.get("max_drift_ratio"))
        ng_m = metrics.get("ng_member_count", "—")
        ng_d = metrics.get("ng_drift_count", "—")
        expected_structural_effect = (
            f"재해석 후 최대 상관식 비(DCR) {max_dcr}, 최대 층간변위비 {max_drift}, "
            f"NG 부재 {ng_m}건, NG 층 {ng_d}건."
        )
    else:
        expected_structural_effect = (
            f"제안 효과(검증 전): {context.get('expected_effect') or '—'}"
        )

    # ── verified_result ───────────────────────────────────────────────────
    if verified and improvement:
        verified_result = (
            f"재해석 검증 통과 ({status_label}). "
            f"DCR 변화량 {_fmt_delta(improvement.get('dcr_delta'))}, "
            f"층간변위비 변화량 {_fmt_delta(improvement.get('drift_delta'))}, "
            f"NG 부재 변화 {improvement.get('ng_member_delta', '—')}건, "
            f"NG 층 변화 {improvement.get('ng_drift_delta', '—')}건. "
            f"종합 점수 {_fmt_ratio(score.get('total'))} "
            f"(방식 {score.get('method', VERIFIED_SCORE_METHOD)})."
        )
    elif status == STATUS_REJECTED_NEW_NG:
        verified_result = (
            f"재해석 결과 새 NG가 발생함 — 이전에 OK였던 부재/층이 NG로 전환되었습니다. "
            f"개선 지표: {improvement}."
        )
    elif status == STATUS_REJECTED_FAILED:
        err = evaluation.get("error") or "원인 미상"
        # Keep error excerpt short; pure deterministic copy from upstream.
        verified_result = f"재해석 실패: {str(err).splitlines()[0]}"
    else:
        verified_result = "재해석 검증이 아직 수행되지 않았습니다. 현재 표시는 추정치입니다."

    # ── tradeoffs ─────────────────────────────────────────────────────────
    base_trade = context.get("tradeoffs") or ""
    weight_proxy = metrics.get("weight_proxy") if metrics else None
    weight_part = (
        f" 추가 강재 무게 proxy={float(weight_proxy):.0f} mm³."
        if weight_proxy
        else ""
    )
    tradeoffs = (base_trade or "정량적 트레이드오프는 제공되지 않았습니다.") + weight_part

    # ── limitations ───────────────────────────────────────────────────────
    lim_parts: list[str] = []
    if not evidence:
        lim_parts.append("KDS 인용이 확보되지 않았습니다 (RAG 인덱스 미설정 또는 무매칭).")
    if not verified:
        lim_parts.append("재해석 검증이 없는 추정치입니다.")
    if not applicable:
        lim_parts.append("이 후보는 자동 적용이 불가합니다.")
    if not lim_parts:
        lim_parts.append("표시된 결과는 결정론적 분석과 인용된 KDS 근거에 기반합니다.")
    limitations = " ".join(lim_parts)

    # ── next_user_decision ────────────────────────────────────────────────
    if status == STATUS_REJECTED_NEW_NG:
        next_user_decision = (
            "자동 적용 금지. 변경 범위를 줄이거나 다른 후보를 선택해 재평가하세요."
        )
    elif status == STATUS_REJECTED_FAILED:
        next_user_decision = "재해석 실패 원인을 확인하고 모델/조건을 점검 후 재실행하세요."
    elif not applicable:
        next_user_decision = (
            "엔지니어 검토 후 모델에 수동으로 반영하세요. 본 시스템은 자동 적용을 제공하지 않습니다."
        )
    elif verified:
        next_user_decision = (
            "Preview로 변경 내용을 확인한 뒤 Apply로 반영하세요. 적용 후 모델은 자동 재해석됩니다."
        )
    else:
        next_user_decision = (
            "먼저 Evaluate Candidates로 재해석 검증을 수행한 뒤 Preview/Apply 여부를 결정하세요."
        )

    return {
        "summary": summary,
        "issue_interpretation": issue_interpretation,
        "recommended_change": recommended_change,
        "expected_structural_effect": expected_structural_effect,
        "verified_result": verified_result,
        "tradeoffs": tradeoffs,
        "limitations": limitations,
        "next_user_decision": next_user_decision,
    }


# Guards: deterministic text must never contain a fabricated "KDS XX YY ZZ"
# clause when evidence is empty. We do not silently strip — instead we
# assert via a regex at confidence-calc time.
_KDS_CLAUSE_RE = re.compile(r"KDS\s+\d{2}\s+\d{2}\s+\d{2}")


def _contains_kds_citation(text_blocks: dict[str, str]) -> bool:
    return any(_KDS_CLAUSE_RE.search(v or "") for v in text_blocks.values())


def _confidence_level(*, verified: bool, evidence_count: int) -> str:
    if verified and evidence_count >= 2:
        return "high"
    if verified or evidence_count >= 1:
        return "medium"
    return "low"


def deterministic_explanation(
    context: dict[str, Any],
    evidence: list[KdsEvidence],
    *,
    language: str = "ko",
    style: str = "engineer_brief",
) -> RecommendationExplanation:
    """LLM-free explanation built only from context + supplied evidence.

    ``language`` and ``style`` are honored as hints — currently only
    ``ko`` / ``engineer_brief`` are implemented. Other combinations
    fall through to the Korean engineer brief; callers that need a
    different presentation should wire an LLM provider.
    """
    _ = language, style  # keep signature; future variants can dispatch.
    cand_id = context.get("candidate_id", "")
    text_blocks = _engineer_brief_text(context, evidence)

    # Invariant: deterministic text must not invent a KDS clause id when
    # we have no evidence. This is enforced by construction (no KDS
    # numbers appear in the templates above) — assert it here so
    # regressions surface immediately rather than as silent hallucinations.
    if not evidence and _contains_kds_citation(text_blocks):
        raise AssertionError(
            "deterministic_explanation produced a KDS clause citation "
            "without any retrieved evidence — this is a guardrail bug."
        )

    confidence = _confidence_level(
        verified=context.get("verified", False),
        evidence_count=len(evidence),
    )

    warnings = list(context.get("warnings") or [])
    if not evidence:
        warnings.append(
            "kds_evidence_missing: KDS 인용이 확보되지 않아 "
            "결정론적 해석 결과만으로 설명을 구성했습니다."
        )

    return RecommendationExplanation(
        candidate_id=cand_id,
        explanation=text_blocks,
        kds_evidence=list(evidence),
        confidence=confidence,
        source={
            "deterministic": True,
            "rag_used": bool(evidence),
            "llm_used": False,
            "score_method": context.get("score_method", "unknown"),
        },
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Retrieval + orchestration
# ---------------------------------------------------------------------------

def _retrieve_evidence(
    context: dict[str, Any],
    retriever: KDSRetriever,
    *,
    top_k: int = 5,
) -> tuple[list[KdsEvidence], list[str], bool]:
    """Run retrieval and convert chunks → validated KdsEvidence.

    Returns ``(evidence, warnings, rag_used)``. ``rag_used`` is True when
    the retriever was invoked successfully *regardless* of whether any
    chunks came back — Noop retriever sets it False via its NOOP_WARNING.
    """
    warnings: list[str] = []
    rag_used = True
    try:
        query = make_kds_query(context)
    except Exception as exc:  # noqa: BLE001 — retrieval must never fail loud
        warnings.append(f"kds_query_build_failed: {exc!r}")
        return [], warnings, False

    try:
        result = retriever.retrieve(query, top_k=top_k)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"kds_retriever_exception: {exc!r}")
        return [], warnings, False

    # Propagate backend warnings verbatim; if the Noop sentinel appears,
    # downgrade rag_used so the API surface is honest.
    for w in result.warnings or []:
        warnings.append(w)
        if w == NOOP_WARNING:
            rag_used = False

    evidence: list[KdsEvidence] = []
    for rank, chunk in enumerate(result.chunks or []):
        try:
            ref = chunk_to_code_reference(chunk, base_ref=None)
        except Exception as exc:  # noqa: BLE001
            warnings.append(
                f"chunk_to_ref_failed: chunk={chunk.chunk_id} err={exc!r}"
            )
            continue
        ok, reason = validate_code_reference(
            ref,
            chunk_id=chunk.chunk_id,
            matched_topic=chunk.topic,
            matched_limit_state=chunk.limit_state,
        )
        if not ok:
            warnings.append(
                f"retrieved KDS chunk rejected by citation validator: "
                f"chunk={chunk.chunk_id} reason={reason}"
            )
            continue
        score = float(
            (result.scores or {}).get(chunk.chunk_id, 1.0 / (rank + 1))
        )
        evidence.append(KdsEvidence.from_code_reference(ref, score=score))

    return evidence, warnings, rag_used


def explain_candidate(
    candidate: RetrofitCandidate,
    *,
    diff: Any = None,
    evaluation: Any = None,
    analysis_summary: Optional[dict[str, Any]] = None,
    retriever: Optional[KDSRetriever] = None,
    llm_provider: Any = None,
    language: str = "ko",
    style: str = "engineer_brief",
    top_k: int = 5,
) -> RecommendationExplanation:
    """Build an explanation for a single retrofit candidate.

    Orchestration:
        1. Build deterministic context (never raises).
        2. If a retriever is supplied, retrieve+validate KDS evidence.
           Any retrieval failure becomes a warning, not an exception.
        3. If an LLM provider is supplied, try to generate refined text
           from context+evidence. On *any* failure we fall back to
           deterministic_explanation and add a warning.
        4. Otherwise return deterministic_explanation directly.

    The endpoint that calls this must never see a raised exception just
    because RAG/LLM is unavailable.
    """
    context = build_explanation_context(
        candidate,
        diff=diff,
        evaluation=evaluation,
        analysis_summary=analysis_summary,
    )

    evidence: list[KdsEvidence] = []
    retrieval_warnings: list[str] = []
    rag_used = False
    if retriever is not None:
        evidence, retrieval_warnings, rag_used = _retrieve_evidence(
            context, retriever, top_k=top_k,
        )

    # Build the deterministic explanation FIRST — it's our fallback
    # and also the safety floor. LLM (when wired) may overwrite the
    # text blocks but never the source/confidence guarantees.
    det = deterministic_explanation(
        context, evidence, language=language, style=style,
    )
    # Merge retrieval warnings into the result (deterministic_explanation
    # already added kds_evidence_missing when appropriate).
    for w in retrieval_warnings:
        if w not in det.warnings:
            det.warnings.append(w)
    det.source["rag_used"] = bool(rag_used and evidence)

    if llm_provider is None:
        return det

    try:
        llm_blocks = llm_provider.generate(
            context=copy.deepcopy(context),
            evidence=[e.to_dict() for e in evidence],
            language=language,
            style=style,
        )
    except Exception as exc:  # noqa: BLE001
        det.warnings.append(f"llm_provider_failed: {exc!r}")
        return det

    if not isinstance(llm_blocks, dict):
        det.warnings.append(
            "llm_provider_returned_unexpected_type: falling back to deterministic"
        )
        return det

    # Merge: keep deterministic text where the LLM omitted a section.
    merged = dict(det.explanation)
    for k, v in llm_blocks.items():
        if isinstance(v, str) and v.strip():
            merged[k] = v.strip()

    det.explanation = merged
    det.source["llm_used"] = True
    return det


# ---------------------------------------------------------------------------
# Helper for the API endpoint: derive diff lazily from cached model_json
# ---------------------------------------------------------------------------

def derive_diff_if_applicable(
    candidate: RetrofitCandidate,
    model_json: Optional[dict],
) -> tuple[Optional[ChangeDiff], list[str]]:
    """If the candidate is applicable and no diff was provided, compute one.

    The cached ``model_json`` is deep-copied before being passed into
    :func:`apply_candidate_to_model`, which itself deep-copies again.
    The returned diff is detached from the cached state.
    """
    if model_json is None:
        return None, ["diff_derivation_skipped: no cached model_json"]
    applicable = bool((candidate.proposed_change or {}).get("applicable", True))
    if not applicable:
        return None, []
    try:
        _, diff = apply_candidate_to_model(copy.deepcopy(model_json), candidate)
        return diff, []
    except InapplicableCandidateError as e:
        return None, [f"diff_derivation_inapplicable: {e.reason}"]
    except Exception as exc:  # noqa: BLE001
        return None, [f"diff_derivation_failed: {exc!r}"]


__all__ = [
    "KdsEvidence",
    "RecommendationExplanation",
    "build_explanation_context",
    "deterministic_explanation",
    "explain_candidate",
    "derive_diff_if_applicable",
]
