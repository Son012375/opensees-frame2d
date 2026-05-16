"""Placeholder retrofit-candidate generator.

This is *intentionally* minimal. Real recommendation logic lives in
later layers:

    * KDS-RAG retrieval → :class:`CodeReference` for each candidate
    * LLM narration → ``description`` / ``tradeoffs`` polish
    * Optimization / iteration → actual section sizing

For now we only ensure that each :class:`StructuralIssue` produces at
least one well-typed :class:`RetrofitCandidate` so that downstream
components have a stable surface to bind to. The data model itself
encodes "not a final design" via ``requires_reanalysis=True``.
"""
from __future__ import annotations

import uuid
from typing import Iterable

from .schemas import (
    ActionType,
    CodeReference,
    Confidence,
    IssueType,
    RetrofitCandidate,
    StructuralIssue,
)


def _new_id() -> str:
    return f"cand_{uuid.uuid4().hex[:10]}"


def _strength_candidate(issue: StructuralIssue) -> RetrofitCandidate:
    """Default placeholder for strength_exceeded / shear_exceeded."""
    ratio = issue.demand_capacity_ratio or 1.0
    overstress_pct = max(0.0, (ratio - 1.0)) * 100.0
    section = ""
    if isinstance(issue.evidence, dict):
        section = str(issue.evidence.get("section", "") or "")

    description = (
        f"부재 {issue.member_id} 단면 확대 후보 (현재 단면 '{section}' "
        f"기준 D/C={ratio:.2f}, {overstress_pct:.1f}% 초과)."
    )
    return RetrofitCandidate(
        candidate_id=_new_id(),
        issue_id=issue.issue_id,
        action_type=ActionType.INCREASE_SECTION,
        description=description,
        member_id=issue.member_id,
        element_id=issue.element_id,
        expected_effect=(
            "단면 증가로 D/C 비를 1.0 이하로 낮추는 것이 목표. "
            "정확한 신규 단면은 재해석으로 검증 필요."
        ),
        tradeoffs=(
            "자중/비용 증가, 인접 부재 응력 재분배 가능. "
            "공간/건축 계획 영향 검토 필요."
        ),
        requires_reanalysis=True,
        confidence=Confidence.MEDIUM,
        code_refs=[],  # TODO(rag): KDS 41 31 00 단면 분류 / AISC 360 H1 채워질 예정
        metadata={
            "trigger": issue.issue_type,
            "current_ratio": ratio,
            "section": section,
        },
    )


def _drift_candidate(issue: StructuralIssue) -> RetrofitCandidate:
    """Drift exceedance → suggest adding lateral-resistance + engineer review."""
    ratio = issue.demand_capacity_ratio or 1.0
    story = None
    direction = None
    if isinstance(issue.evidence, dict):
        story = issue.evidence.get("story")
        direction = issue.evidence.get("direction")

    description = (
        f"{story}층 {direction or ''}방향 층간변위가 허용치를 "
        f"{(ratio - 1.0) * 100:.1f}% 초과. "
        "횡력 저항요소 추가/보강을 검토 필요."
    )
    return RetrofitCandidate(
        candidate_id=_new_id(),
        issue_id=issue.issue_id,
        action_type=ActionType.ADD_LATERAL_RESISTANCE,
        description=description,
        expected_effect=(
            "가새/전단벽/모멘트프레임 강성 증가로 층간변위비를 "
            "허용치 이내로 감소."
        ),
        tradeoffs=(
            "건축 계획/개구부 제약 발생 가능. "
            "기초 부담 증가, 다른 층의 강성 분포 영향 검토 필요."
        ),
        requires_reanalysis=True,
        confidence=Confidence.LOW,
        code_refs=[],  # TODO(rag): KDS 41 17 00 §8 / KDS 41 12 00 풍하중
        metadata={
            "trigger": issue.issue_type,
            "story": story,
            "direction": direction,
            "current_ratio": ratio,
        },
    )


def _engineer_review_candidate(issue: StructuralIssue, reason: str) -> RetrofitCandidate:
    return RetrofitCandidate(
        candidate_id=_new_id(),
        issue_id=issue.issue_id,
        action_type=ActionType.REQUIRES_ENGINEER_REVIEW,
        description=f"엔지니어 검토 필요: {reason}",
        member_id=issue.member_id,
        element_id=issue.element_id,
        expected_effect="자동 추천이 부적합한 사례 — 수동 판단.",
        tradeoffs="정성적 판단이 요구되므로 자동 후보 생성 보류.",
        requires_reanalysis=True,
        confidence=Confidence.LOW,
        code_refs=[],
        metadata={"trigger": issue.issue_type, "reason": reason},
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_candidates(
    issues: Iterable[StructuralIssue],
) -> list[RetrofitCandidate]:
    """Map each issue to one or more :class:`RetrofitCandidate`.

    Rules (deterministic placeholders):
        * STRENGTH_EXCEEDED / SHEAR_EXCEEDED with a member_id
          → ``INCREASE_SECTION``
        * DRIFT_EXCEEDED → ``ADD_LATERAL_RESISTANCE``
        * MISSING_DESIGN_CHECK → ``REQUIRES_ENGINEER_REVIEW``
        * ANALYSIS_WARNING with severity=error → ``REQUIRES_ENGINEER_REVIEW``
        * ANALYSIS_WARNING with severity=warning → no candidate (informational)
        * Any issue missing the information needed to act on it
          → ``REQUIRES_ENGINEER_REVIEW``
    """
    candidates: list[RetrofitCandidate] = []

    for issue in issues:
        t = issue.issue_type

        if t in (IssueType.STRENGTH_EXCEEDED, IssueType.SHEAR_EXCEEDED):
            if issue.member_id is None:
                candidates.append(_engineer_review_candidate(
                    issue, "강도 초과지만 member_id가 식별되지 않음"
                ))
            else:
                candidates.append(_strength_candidate(issue))

        elif t == IssueType.DRIFT_EXCEEDED:
            candidates.append(_drift_candidate(issue))

        elif t == IssueType.MISSING_DESIGN_CHECK:
            candidates.append(_engineer_review_candidate(
                issue, "설계검토 결과 부재"
            ))

        elif t == IssueType.ANALYSIS_WARNING:
            # Only error-level warnings get a candidate; pure info/warn is
            # surfaced but not actionable at this stage.
            if issue.severity == "error":
                candidates.append(_engineer_review_candidate(
                    issue, "해석 경고가 error 등급 — 결과 신뢰성 재검토 필요"
                ))

        else:
            candidates.append(_engineer_review_candidate(
                issue, f"미분류 이슈 유형: {t}"
            ))

    return candidates
