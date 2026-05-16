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

Two key concepts encoded here:

1. :func:`_should_block_auto_candidate` lists every condition under
   which we refuse to emit a typed candidate and fall back to
   ``REQUIRES_ENGINEER_REVIEW``.
2. ``RetrofitCandidate.proposed_change`` is the *machine-readable*
   contract a re-analysis loop will eventually act on. The placeholder
   layer only fills ``operation`` / ``from`` / ``requires_user_selection``;
   the actual ``to`` is left null because automatic sizing is out of scope.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional

from .ids import make_candidate_id
from .schemas import (
    ActionType,
    ChangeOperation,
    Confidence,
    IssueType,
    RetrofitCandidate,
    Severity,
    StructuralIssue,
)


# ---------------------------------------------------------------------------
# "No auto-candidate" rules
# ---------------------------------------------------------------------------

def _should_block_auto_candidate(issue: StructuralIssue) -> Optional[str]:
    """Return a human-readable reason to refuse auto-candidate, or None.

    Centralizing these guards keeps the per-issue branches below short
    and lets the test suite assert each rule individually.
    """
    t = issue.issue_type

    if t == IssueType.MISSING_DESIGN_CHECK:
        return "설계검토 결과 부재 — 강도/변위 판정 불가"

    if t in (IssueType.STRENGTH_EXCEEDED, IssueType.SHEAR_EXCEEDED):
        if issue.member_id is None:
            return "강도 초과지만 member_id가 식별되지 않음"
        if not (issue.member_type or
                (isinstance(issue.evidence, dict) and issue.evidence.get("type"))):
            return "member_type 정보 부족 — 자동 후보 생성 보류"
        if not (issue.section or
                (isinstance(issue.evidence, dict) and issue.evidence.get("section"))):
            return "현재 section 정보 부족 — 자동 후보 생성 보류"

    if t == IssueType.DRIFT_EXCEEDED:
        # Story is the minimum info needed to point at *which* level
        # needs more lateral system. If we don't have it, defer.
        if issue.story is None and (
            not isinstance(issue.evidence, dict)
            or issue.evidence.get("story") is None
        ):
            return "drift 초과이지만 story 정보 없음 — lateral system 미상"

    if t == IssueType.ANALYSIS_WARNING:
        # Only error-severity warnings ever yield a candidate; other
        # severities should not block, but pure info is filtered out
        # upstream in :func:`generate_candidates`.
        if issue.severity != Severity.ERROR:
            return None

    return None


# ---------------------------------------------------------------------------
# Builders for the typed candidate shapes
# ---------------------------------------------------------------------------

def _strength_candidate(issue: StructuralIssue) -> RetrofitCandidate:
    """Default placeholder for strength_exceeded / shear_exceeded."""
    ratio = issue.demand_capacity_ratio or 1.0
    overstress_pct = max(0.0, (ratio - 1.0)) * 100.0
    section = issue.section or ""
    if not section and isinstance(issue.evidence, dict):
        section = str(issue.evidence.get("section", "") or "")

    description = (
        f"부재 {issue.member_id} 단면 확대 후보 (현재 단면 '{section}' "
        f"기준 D/C={ratio:.2f}, {overstress_pct:.1f}% 초과)."
    )
    target = {
        "member_id": issue.member_id,
        "element_id": issue.element_id,
        "member_type": issue.member_type
        or (issue.evidence.get("type") if isinstance(issue.evidence, dict) else None),
    }
    proposed_change = {
        "operation": ChangeOperation.REPLACE_SECTION,
        "from": section or None,
        "to": None,  # TODO(optimizer/rag): filled later by sizing logic
        "requires_user_selection": True,
        "reason": (
            "strength_exceeded" if issue.issue_type == IssueType.STRENGTH_EXCEEDED
            else "shear_exceeded"
        ),
    }
    return RetrofitCandidate(
        candidate_id=make_candidate_id(
            issue_id=issue.issue_id, action_type=ActionType.INCREASE_SECTION,
        ),
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
        code_refs=[],  # TODO(rag): KDS 41 31 00 단면 분류 / AISC 360 H1
        target=target,
        proposed_change=proposed_change,
        metadata={
            "trigger": issue.issue_type,
            "current_ratio": ratio,
            "section": section,
        },
    )


def _drift_candidate(issue: StructuralIssue) -> RetrofitCandidate:
    ratio = issue.demand_capacity_ratio or 1.0
    story = issue.story
    direction = None
    if isinstance(issue.evidence, dict):
        story = story or issue.evidence.get("story")
        direction = issue.evidence.get("direction")

    description = (
        f"{story}층 {direction or ''}방향 층간변위가 허용치를 "
        f"{(ratio - 1.0) * 100:.1f}% 초과. "
        "횡력 저항요소 추가/보강을 검토 필요."
    )
    target = {
        "scope": "story",
        "story": story,
        "direction": direction,
    }
    proposed_change = {
        "operation": ChangeOperation.ADD_LATERAL_RESISTANCE,
        "from": None,
        "to": None,
        "requires_user_selection": True,
        "reason": "drift_exceeded",
    }
    return RetrofitCandidate(
        candidate_id=make_candidate_id(
            issue_id=issue.issue_id, action_type=ActionType.ADD_LATERAL_RESISTANCE,
        ),
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
        target=target,
        proposed_change=proposed_change,
        metadata={
            "trigger": issue.issue_type,
            "story": story,
            "direction": direction,
            "current_ratio": ratio,
        },
    )


def _engineer_review_candidate(issue: StructuralIssue, reason: str) -> RetrofitCandidate:
    target: dict[str, Any] = {}
    if issue.member_id is not None:
        target["member_id"] = issue.member_id
        if issue.element_id is not None:
            target["element_id"] = issue.element_id
    proposed_change = {
        "operation": ChangeOperation.MANUAL_REVIEW,
        "from": issue.section,
        "to": None,
        "requires_user_selection": True,
        "reason": reason,
    }
    return RetrofitCandidate(
        candidate_id=make_candidate_id(
            issue_id=issue.issue_id, action_type=ActionType.REQUIRES_ENGINEER_REVIEW,
        ),
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
        target=target,
        proposed_change=proposed_change,
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
        * STRENGTH_EXCEEDED / SHEAR_EXCEEDED → ``INCREASE_SECTION``
          (``replace_section``)
        * DRIFT_EXCEEDED → ``ADD_LATERAL_RESISTANCE``
        * MISSING_DESIGN_CHECK → ``REQUIRES_ENGINEER_REVIEW``
        * ANALYSIS_WARNING with severity=error → ``REQUIRES_ENGINEER_REVIEW``
        * ANALYSIS_WARNING with severity=warning/info → no candidate
        * Any issue blocked by :func:`_should_block_auto_candidate`
          → ``REQUIRES_ENGINEER_REVIEW`` (with the reason as metadata)
    """
    candidates: list[RetrofitCandidate] = []

    for issue in issues:
        t = issue.issue_type

        # Info/warn-severity analysis warnings are surfaced as issues but
        # are not actionable here. Skip before guard checks.
        if t == IssueType.ANALYSIS_WARNING and issue.severity != Severity.ERROR:
            continue

        block_reason = _should_block_auto_candidate(issue)
        if block_reason is not None:
            candidates.append(_engineer_review_candidate(issue, block_reason))
            continue

        if t in (IssueType.STRENGTH_EXCEEDED, IssueType.SHEAR_EXCEEDED):
            candidates.append(_strength_candidate(issue))
        elif t == IssueType.DRIFT_EXCEEDED:
            candidates.append(_drift_candidate(issue))
        elif t == IssueType.ANALYSIS_WARNING:
            # Error-level warning that wasn't blocked: still review-only.
            candidates.append(_engineer_review_candidate(
                issue, "해석 경고가 error 등급 — 결과 신뢰성 재검토 필요",
            ))
        else:
            candidates.append(_engineer_review_candidate(
                issue, f"미분류 이슈 유형: {t}",
            ))

    return candidates
