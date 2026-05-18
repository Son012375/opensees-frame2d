"""Deterministic retrofit-candidate generator.

What this module produces for each :class:`StructuralIssue`:

    * strength_exceeded / shear_exceeded → up to 3 candidates
        1. member 1-step section upgrade  (replace_section)
        2. member 2-step section upgrade  (replace_section)
        3. story-wide 1-step upgrade for the same member_type
                                          (replace_sections_by_story)
    * drift_exceeded → up to 3 candidates
        1. story column 1-step upgrade    (replace_sections_by_story)
        2. story column 2-step upgrade    (replace_sections_by_story)
        3. add_lateral_resistance         (abstract, applicable=false)
    * missing_design_check / analysis_warning(error) → engineer review

Each candidate carries a machine-readable ``proposed_change`` contract:

    {
        "operation": ChangeOperation.<...>,
        "from":      current section name (str) | None,
        "to":        target section name (str)  | None,
        "requires_user_selection": False (when ``to`` is fixed),
        "applicable": True | False,    # False = abstract / review-only
        "reason":    issue.issue_type,
    }

``applicable=False`` candidates are surfaced for context but the
evaluator and ``apply_candidate`` executor will refuse to act on them
— they exist to tell the user "this option exists but it's outside
the deterministic loop's scope" (e.g. add bracing, add shear wall).

If section_catalog can't suggest a real ``to`` (e.g. unknown family,
top of ladder), the corresponding candidate is dropped. The handler
never returns a no-op candidate.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional

from .ids import make_candidate_id
from .registry import RuleRegistry, default_registry, normalize_handler_result
from .schemas import (
    ActionType,
    ChangeOperation,
    Confidence,
    IssueType,
    RetrofitCandidate,
    Severity,
    StructuralIssue,
)
from .section_catalog import next_sections


# ---------------------------------------------------------------------------
# "No auto-candidate" rules
# ---------------------------------------------------------------------------

def _should_block_auto_candidate(issue: StructuralIssue) -> Optional[str]:
    """Return a human-readable reason to refuse auto-candidate, or None.

    Centralizes the guards that send an issue to engineer-review instead
    of the typed candidate handlers.
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
        if issue.story is None and (
            not isinstance(issue.evidence, dict)
            or issue.evidence.get("story") is None
        ):
            return "drift 초과이지만 story 정보 없음 — lateral system 미상"

    if t == IssueType.ANALYSIS_WARNING:
        if issue.severity != Severity.ERROR:
            return None

    return None


# ---------------------------------------------------------------------------
# Field-extraction helpers
# ---------------------------------------------------------------------------

def _issue_section(issue: StructuralIssue) -> Optional[str]:
    if issue.section:
        return issue.section
    if isinstance(issue.evidence, dict):
        s = issue.evidence.get("section")
        if s:
            return str(s)
    return None


def _issue_member_type(issue: StructuralIssue) -> Optional[str]:
    if issue.member_type:
        return issue.member_type
    if isinstance(issue.evidence, dict):
        t = issue.evidence.get("type") or issue.evidence.get("member_type")
        if t:
            return str(t)
    return None


def _issue_story(issue: StructuralIssue) -> Optional[int]:
    if issue.story is not None:
        return issue.story
    if isinstance(issue.evidence, dict):
        s = issue.evidence.get("story")
        if s is not None:
            try:
                return int(s)
            except (TypeError, ValueError):
                return None
    return None


# ---------------------------------------------------------------------------
# Strength / shear candidates
# ---------------------------------------------------------------------------

def _strength_candidates(issue: StructuralIssue) -> list[RetrofitCandidate]:
    """Up to 3 candidates for strength/shear overstress."""
    section = _issue_section(issue) or ""
    member_type = _issue_member_type(issue) or ""
    story = _issue_story(issue)
    ratio = issue.demand_capacity_ratio or 1.0
    reason = (
        "strength_exceeded" if issue.issue_type == IssueType.STRENGTH_EXCEEDED
        else "shear_exceeded"
    )

    nxt = next_sections(section, member_type, steps=2) if section else []
    out: list[RetrofitCandidate] = []

    base_target = {
        "scope": "member",
        "member_id": issue.member_id,
        "element_id": issue.element_id,
        "member_type": member_type or None,
        "story": story,
    }

    # 1) Member 1-step / 2-step upgrade — individual candidates
    for step_idx, entry in enumerate(nxt, start=1):
        desc = (
            f"부재 {issue.member_id} 단면 {entry.name} 로 교체 "
            f"({step_idx}단계, 현재 {section} D/C={ratio:.2f})."
        )
        proposed_change = {
            "operation": ChangeOperation.REPLACE_SECTION,
            "from": section or None,
            "to": entry.name,
            "requires_user_selection": False,
            "applicable": True,
            "reason": reason,
        }
        out.append(RetrofitCandidate(
            candidate_id=make_candidate_id(
                issue_id=issue.issue_id,
                action_type=ActionType.INCREASE_SECTION,
                variant=entry.name,
            ),
            issue_id=issue.issue_id,
            action_type=ActionType.INCREASE_SECTION,
            description=desc,
            member_id=issue.member_id,
            element_id=issue.element_id,
            expected_effect=(
                f"단면 증가로 D/C 비를 1.0 이하로 낮추는 것이 목표 "
                f"(검증은 재해석 필요)."
            ),
            tradeoffs="자중/비용 증가, 인접 부재 응력 재분배 가능.",
            requires_reanalysis=True,
            confidence=Confidence.MEDIUM,
            code_refs=[],
            target=dict(base_target),
            proposed_change=proposed_change,
            metadata={
                "trigger": issue.issue_type,
                "current_ratio": ratio,
                "section_from": section,
                "section_to": entry.name,
                "upgrade_step": step_idx,
            },
        ))

    # 3) Story-wide 1-step upgrade for the same member_type — only if
    #    we actually know the story.
    if story is not None and member_type:
        story_target = {
            "scope": "story",
            "story": story,
            "member_type": member_type,
        }
        proposed_change = {
            "operation": ChangeOperation.REPLACE_SECTIONS_BY_STORY,
            "from": section or None,
            "to": None,   # executor walks each member's own family ladder
            "requires_user_selection": False,
            "applicable": True,
            "reason": reason,
            "upgrade_step": 1,
        }
        out.append(RetrofitCandidate(
            candidate_id=make_candidate_id(
                issue_id=issue.issue_id,
                action_type=ActionType.INCREASE_SECTION,
                variant=f"story{story}_step1",
            ),
            issue_id=issue.issue_id,
            action_type=ActionType.INCREASE_SECTION,
            description=(
                f"{story}층 {member_type} 단면 일괄 1단계 증대 "
                f"(부재 {issue.member_id} 보강만으로 부족할 때 검토)."
            ),
            member_id=issue.member_id,
            element_id=issue.element_id,
            expected_effect=(
                "같은 층의 동일 타입 부재 강성 균일 증대. 단일 부재 "
                "교체보다 응력 재분배가 부드러움."
            ),
            tradeoffs="변경 부재 수 증가, 자중/비용 증가.",
            requires_reanalysis=True,
            confidence=Confidence.MEDIUM,
            code_refs=[],
            target=story_target,
            proposed_change=proposed_change,
            metadata={
                "trigger": issue.issue_type,
                "current_ratio": ratio,
                "story": story,
                "member_type": member_type,
                "upgrade_step": 1,
            },
        ))

    return out


# ---------------------------------------------------------------------------
# Drift candidates
# ---------------------------------------------------------------------------

def _drift_candidates(issue: StructuralIssue) -> list[RetrofitCandidate]:
    """Up to 3 candidates for drift overstress.

    Plan: only column story upgrade (1단계/2단계) is verified-applicable
    in Phase 1. ``add_lateral_resistance`` is kept as an abstract
    candidate (applicable=False) so the user sees the option exists
    but the evaluator does not attempt it.
    """
    ratio = issue.demand_capacity_ratio or 1.0
    story = _issue_story(issue)
    direction = None
    if isinstance(issue.evidence, dict):
        direction = issue.evidence.get("direction")

    out: list[RetrofitCandidate] = []

    # 1) & 2) Story column 1-step / 2-step upgrade
    if story is not None:
        for step_idx in (1, 2):
            story_target = {
                "scope": "story",
                "story": story,
                "member_type": "column",
                "direction": direction,
            }
            proposed_change = {
                "operation": ChangeOperation.REPLACE_SECTIONS_BY_STORY,
                "from": None,
                "to": None,
                "requires_user_selection": False,
                "applicable": True,
                "reason": "drift_exceeded",
                "upgrade_step": step_idx,
            }
            out.append(RetrofitCandidate(
                candidate_id=make_candidate_id(
                    issue_id=issue.issue_id,
                    action_type=ActionType.INCREASE_SECTION,
                    variant=f"story{story}_column_step{step_idx}",
                ),
                issue_id=issue.issue_id,
                action_type=ActionType.INCREASE_SECTION,
                description=(
                    f"{story}층 column 단면 일괄 {step_idx}단계 증대 "
                    f"(layer 강성 증가로 층간변위 개선 시도, "
                    f"현재 비={ratio:.2f})."
                ),
                expected_effect=(
                    "층 강성 증대로 층간변위비 감소. 단, 강성 비대칭이 "
                    "심해지면 비틀림 응답 악화 가능 — 재해석 필수."
                ),
                tradeoffs="자중/비용 증가, 기초 부담 증가.",
                requires_reanalysis=True,
                confidence=Confidence.MEDIUM,
                code_refs=[],
                target=story_target,
                proposed_change=proposed_change,
                metadata={
                    "trigger": issue.issue_type,
                    "current_ratio": ratio,
                    "story": story,
                    "direction": direction,
                    "member_type": "column",
                    "upgrade_step": step_idx,
                },
            ))

    # 3) Abstract: add lateral-resistance system. Not auto-applicable.
    abstract_target = {
        "scope": "story",
        "story": story,
        "direction": direction,
    }
    abstract_change = {
        "operation": ChangeOperation.ADD_LATERAL_RESISTANCE,
        "from": None,
        "to": None,
        "requires_user_selection": True,
        "applicable": False,
        "reason": "drift_exceeded",
    }
    out.append(RetrofitCandidate(
        candidate_id=make_candidate_id(
            issue_id=issue.issue_id,
            action_type=ActionType.ADD_LATERAL_RESISTANCE,
        ),
        issue_id=issue.issue_id,
        action_type=ActionType.ADD_LATERAL_RESISTANCE,
        description=(
            f"{story}층 {direction or ''}방향에 가새/전단벽 등 횡력 저항요소 "
            "추가 (정성적 후보 — 자동 적용 대상 아님)."
        ),
        expected_effect=(
            "가새/전단벽/모멘트프레임 강성 추가로 층간변위비 감소. "
            "건축 계획/개구부 제약 검토 필요."
        ),
        tradeoffs=(
            "건축 평면 제약, 기초 부담 증가. 본 후보는 manual review."
        ),
        requires_reanalysis=True,
        confidence=Confidence.LOW,
        code_refs=[],
        target=abstract_target,
        proposed_change=abstract_change,
        metadata={
            "trigger": issue.issue_type,
            "story": story,
            "direction": direction,
            "current_ratio": ratio,
            "abstract": True,
        },
    ))

    return out


# ---------------------------------------------------------------------------
# Engineer review (fallback)
# ---------------------------------------------------------------------------

def _engineer_review_candidate(
    issue: StructuralIssue, reason: str,
) -> RetrofitCandidate:
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
        "applicable": False,
        "reason": reason,
    }
    return RetrofitCandidate(
        candidate_id=make_candidate_id(
            issue_id=issue.issue_id,
            action_type=ActionType.REQUIRES_ENGINEER_REVIEW,
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
    *,
    registry: Optional[RuleRegistry] = None,
) -> list[RetrofitCandidate]:
    """Map each issue to one or more candidates via the registry.

    Behaviour (default registry):
        * STRENGTH_EXCEEDED / SHEAR_EXCEEDED → list of section upgrades
          + story-wide upgrade (handler returns up to 3).
        * DRIFT_EXCEEDED → story column 1단계/2단계 + abstract lateral
          (handler returns up to 3).
        * MISSING_DESIGN_CHECK / ANALYSIS_WARNING(error) → engineer review.

    Handlers may return None / a single candidate / an iterable of them.
    Empty handler output triggers an engineer-review fallback so the
    pipeline always has *something* to surface per issue.
    """
    candidates: list[RetrofitCandidate] = []
    reg = registry if registry is not None else default_registry()

    for issue in issues:
        t = issue.issue_type

        # Info/warn-severity analysis warnings are surfaced as issues but
        # are not actionable. Skip before guard checks.
        if t == IssueType.ANALYSIS_WARNING and issue.severity != Severity.ERROR:
            continue

        block_reason = _should_block_auto_candidate(issue)
        if block_reason is not None:
            candidates.append(_engineer_review_candidate(issue, block_reason))
            continue

        entry = reg.get(t)
        if entry is not None:
            produced = normalize_handler_result(entry.handler(issue))
            if produced:
                candidates.extend(produced)
                continue
            # Handler explicitly opted out — fall back to review.
            candidates.append(_engineer_review_candidate(
                issue, f"등록된 규칙이 후보를 생성하지 않음: {t}",
            ))
            continue

        # No registered rule for this issue_type — fall back to review.
        candidates.append(_engineer_review_candidate(
            issue, f"미분류 이슈 유형: {t}",
        ))

    return candidates


# ---------------------------------------------------------------------------
# Default rule registrations (module-load time)
# ---------------------------------------------------------------------------

def _handle_strength(issue: StructuralIssue) -> list[RetrofitCandidate]:
    cands = _strength_candidates(issue)
    if cands:
        return cands
    # Section ladder ran dry (top of the catalog) — degrade gracefully.
    return [_engineer_review_candidate(
        issue, "단면 ladder 최상단 — 자동 증대 후보 없음",
    )]


def _handle_drift(issue: StructuralIssue) -> list[RetrofitCandidate]:
    return _drift_candidates(issue)


def _handle_missing_design_check(issue: StructuralIssue) -> RetrofitCandidate:
    return _engineer_review_candidate(
        issue, "설계검토 결과 부재 — 강도/변위 판정 불가",
    )


def _handle_analysis_warning(issue: StructuralIssue) -> Optional[RetrofitCandidate]:
    if issue.severity != Severity.ERROR:
        return None
    return _engineer_review_candidate(
        issue, "해석 경고가 error 등급 — 결과 신뢰성 재검토 필요",
    )


def _register_default_rules() -> None:
    reg = default_registry()
    reg.register(
        IssueType.STRENGTH_EXCEEDED, _handle_strength,
        label="strength → increase_section (member 1/2 step + story-wide)",
    )
    reg.register(
        IssueType.SHEAR_EXCEEDED, _handle_strength,
        label="shear → increase_section (member 1/2 step + story-wide)",
    )
    reg.register(
        IssueType.DRIFT_EXCEEDED, _handle_drift,
        label="drift → story column upgrades + abstract lateral",
    )
    reg.register(
        IssueType.MISSING_DESIGN_CHECK, _handle_missing_design_check,
        label="missing_design_check → engineer_review",
    )
    reg.register(
        IssueType.ANALYSIS_WARNING, _handle_analysis_warning,
        label="analysis_warning → engineer_review (error only)",
    )


_register_default_rules()
