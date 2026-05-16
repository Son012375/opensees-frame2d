"""Deterministic issue extractor.

Walks analysis + design_check + warnings dicts and produces a normalized
list of :class:`StructuralIssue`. The recommendation engine reads only
this layer, never the raw heterogeneous shapes upstream.

Defensive by design:
    * Tolerates missing / partial design_check structure.
    * Records its own structural problems as new warnings instead of
      raising — the analysis itself already succeeded by the time we run.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional

from .context import MemberContext, MemberContextIndex, build_context_index, context_to_kwargs
from .ids import make_issue_id, slugify
from .schemas import (
    AnalysisWarning,
    CodeReference,
    IssueExtractionResult,
    IssueSource,
    IssueType,
    Severity,
    StructuralIssue,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _kds_drift_ref() -> CodeReference:
    return CodeReference(
        standard_id="KDS 41 17 00",
        clause_id="§8.2.3",
        title="허용 층간변위비",
        query_hint="KDS 41 17 00 허용 층간변위비 inelastic story drift",
        topic="seismic_story_drift",
        limit_state="story_drift",
        jurisdiction="KDS",
    )


def _aisc_h1_ref() -> CodeReference:
    return CodeReference(
        standard_id="KDS 41 31 00",
        clause_id="H1",
        title="조합응력 상관식 (AISC 360 H1)",
        query_hint="KDS 41 31 00 강구조 조합응력 AISC 360 H1 interaction",
        topic="steel_member_strength",
        material="steel",
        limit_state="combined_force_interaction",
        jurisdiction="KDS",
    )


def _aisc_shear_ref() -> CodeReference:
    return CodeReference(
        standard_id="KDS 41 31 00",
        clause_id="G2",
        title="전단강도 (AISC 360 G2)",
        query_hint="KDS 41 31 00 강구조 전단강도 AISC 360 G2 shear strength",
        topic="steel_member_shear",
        material="steel",
        limit_state="shear_strength",
        jurisdiction="KDS",
    )


# ---------------------------------------------------------------------------
# Per-source extractors
# ---------------------------------------------------------------------------

def _issues_from_member_check(
    member_check: Optional[dict],
    out_warnings: list[AnalysisWarning],
    ctx_index: MemberContextIndex,
) -> list[StructuralIssue]:
    """Extract member strength issues (interaction > 1.0 or shear > 1.0)."""
    if not isinstance(member_check, dict):
        return []

    members = member_check.get("members") or []
    if not isinstance(members, list):
        out_warnings.append(AnalysisWarning(
            code="issue_extract_malformed_members",
            message="design_check.member_check.members is not a list — skipped",
            severity=Severity.WARNING,
            stage="issue_extractor",
            recoverable=True,
        ))
        return []

    issues: list[StructuralIssue] = []

    for m in members:
        if not isinstance(m, dict):
            continue

        ratios = m.get("ratios") or {}
        interaction = _safe_float(
            ratios.get("interaction", ratios.get("H1")), 0.0,
        )
        shear = _safe_float(ratios.get("shear"), 0.0)
        status = m.get("status", "UNKNOWN")
        member_id = m.get("member_id", m.get("element_id"))
        if member_id is None:
            continue
        try:
            member_id_int = int(member_id)
        except (TypeError, ValueError):
            continue

        # Per-member design_check fields take priority; analysis_metadata
        # fills gaps.
        ctx = MemberContext(
            member_type=m.get("type") or m.get("member_type"),
            section=m.get("section"),
            material=m.get("material"),
            story=m.get("story"),
            connected_node_ids=list(m.get("connected_node_ids") or []),
        )
        ctx.merge(ctx_index.get(member_id_int))
        loc_kwargs = context_to_kwargs(ctx)
        section = ctx.section or ""
        member_type = ctx.member_type or ""
        combo = m.get("governing_combo", "")

        # Strength: interaction > 1.0 → strength_exceeded
        if interaction > 1.0:
            issues.append(StructuralIssue(
                issue_id=make_issue_id(
                    issue_type=IssueType.STRENGTH_EXCEEDED,
                    member_id=member_id_int,
                    governing_combo=combo,
                ),
                issue_type=IssueType.STRENGTH_EXCEEDED,
                severity=Severity.ERROR,
                source=IssueSource.DESIGN_CHECK,
                description=(
                    f"부재 {member_id_int} ({member_type}, {section}) "
                    f"상관비 {interaction:.3f} > 1.0 "
                    f"[combo: {combo or 'n/a'}]"
                ),
                member_id=member_id_int,
                element_id=m.get("element_id"),
                governing_combo=combo or None,
                demand_capacity_ratio=interaction,
                status=status,
                **loc_kwargs,
                evidence={
                    "ratios": ratios,
                    "demand": m.get("demand"),
                    "capacity": m.get("capacity"),
                    "section": section,
                    "type": member_type,
                    "story": ctx.story,
                },
                code_refs=[_aisc_h1_ref()],
            ))

        # Shear: separate issue when shear > 1.0
        if shear > 1.0:
            issues.append(StructuralIssue(
                issue_id=make_issue_id(
                    issue_type=IssueType.SHEAR_EXCEEDED,
                    member_id=member_id_int,
                    governing_combo=combo,
                ),
                issue_type=IssueType.SHEAR_EXCEEDED,
                severity=Severity.ERROR,
                source=IssueSource.DESIGN_CHECK,
                description=(
                    f"부재 {member_id_int} ({member_type}, {section}) "
                    f"전단비 {shear:.3f} > 1.0 "
                    f"[combo: {combo or 'n/a'}]"
                ),
                member_id=member_id_int,
                element_id=m.get("element_id"),
                governing_combo=combo or None,
                demand_capacity_ratio=shear,
                status=status,
                **loc_kwargs,
                evidence={
                    "shear_ratio": shear,
                    "demand": m.get("demand"),
                    "capacity": m.get("capacity"),
                    "section": section,
                    "type": member_type,
                },
                code_refs=[_aisc_shear_ref()],
            ))

    return issues


def _issues_from_drift_check(
    drift_check: Optional[dict],
    out_warnings: list[AnalysisWarning],
) -> list[StructuralIssue]:
    if not isinstance(drift_check, dict):
        return []

    checks = drift_check.get("checks") or []
    if not isinstance(checks, list):
        out_warnings.append(AnalysisWarning(
            code="issue_extract_malformed_drift_checks",
            message="design_check.drift_check.checks is not a list — skipped",
            severity=Severity.WARNING,
            stage="issue_extractor",
            recoverable=True,
        ))
        return []

    allowable = _safe_float(drift_check.get("allowable"), 0.0)
    issues: list[StructuralIssue] = []
    for chk in checks:
        if not isinstance(chk, dict):
            continue
        if chk.get("status") != "NG":
            continue
        ratio = _safe_float(chk.get("ratio"), 0.0)
        story = chk.get("story")
        direction = chk.get("direction", "")
        combo = chk.get("combo", "")
        level = _safe_float(chk.get("height_m"), 0.0) or None

        issues.append(StructuralIssue(
            issue_id=make_issue_id(
                issue_type=IssueType.DRIFT_EXCEEDED,
                story=int(story) if isinstance(story, int) else None,
                direction=direction,
                governing_combo=combo,
            ),
            issue_type=IssueType.DRIFT_EXCEEDED,
            severity=Severity.ERROR,
            source=IssueSource.DESIGN_CHECK,
            description=(
                f"{story}층 {direction}방향 비탄성 층간변위비 "
                f"{chk.get('inelastic_drift', 0):.5f} > 허용 {allowable} "
                f"[combo: {combo or 'n/a'}]"
            ),
            governing_combo=combo or None,
            demand_capacity_ratio=ratio,
            status="NG",
            story=int(story) if isinstance(story, int) else None,
            level=level,
            evidence={
                "story": story,
                "direction": direction,
                "elastic_drift": chk.get("elastic_drift"),
                "inelastic_drift": chk.get("inelastic_drift"),
                "allowable": allowable,
                "Cd": drift_check.get("Cd"),
                "IE": drift_check.get("IE"),
            },
            code_refs=[_kds_drift_ref()],
        ))

    return issues


def _issues_from_warnings(
    warnings: Iterable[Any],
) -> list[StructuralIssue]:
    """Lift each AnalysisWarning into a low-severity issue."""
    issues: list[StructuralIssue] = []
    for w in warnings:
        if isinstance(w, AnalysisWarning):
            code, message, stage, severity = w.code, w.message, w.stage, w.severity
            detail = w.detail or {}
        elif isinstance(w, dict):
            code = str(w.get("code", "warning"))
            message = str(w.get("message", ""))
            stage = str(w.get("stage", ""))
            severity = str(w.get("severity", Severity.WARNING))
            detail = w.get("detail") or {}
        else:
            parsed = AnalysisWarning.from_legacy_string(str(w))
            code, message, stage = parsed.code, parsed.message, parsed.stage
            severity = parsed.severity
            detail = {}

        sev = Severity.ERROR if severity == Severity.ERROR else Severity.WARNING
        issues.append(StructuralIssue(
            issue_id=make_issue_id(
                issue_type=IssueType.ANALYSIS_WARNING,
                extra=f"{code}_{slugify(stage) or 'stage'}",
            ),
            issue_type=IssueType.ANALYSIS_WARNING,
            severity=sev,
            source=IssueSource.WARNING,
            description=f"[{code}] {message}",
            evidence={"code": code, "stage": stage, "detail": detail},
        ))
    return issues


def _missing_design_check_issue() -> StructuralIssue:
    return StructuralIssue(
        issue_id=make_issue_id(
            issue_type=IssueType.MISSING_DESIGN_CHECK,
            extra="global",
        ),
        issue_type=IssueType.MISSING_DESIGN_CHECK,
        severity=Severity.WARNING,
        source=IssueSource.ANALYSIS,
        description=(
            "Design check 결과가 없어 부재 강도/변위 판정을 수행할 수 없습니다. "
            "엔지니어 검토가 필요합니다."
        ),
        evidence={"reason": "design_check is None or empty"},
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def extract_issues(
    *,
    design_check: Optional[dict] = None,
    warnings: Optional[Iterable[Any]] = None,
    analysis_metadata: Optional[dict] = None,
    out_warnings: Optional[list[AnalysisWarning]] = None,
    include_missing_design_check: bool = True,
) -> IssueExtractionResult:
    """Build a normalized issue list from heterogeneous analysis output.

    Parameters
    ----------
    design_check : dict | None
        ``run_design_check`` output. May be ``None`` if the design check
        failed or was skipped.
    warnings : iterable
        Either legacy ``list[str]``, list of ``AnalysisWarning`` instances,
        or list of dicts in the ``AnalysisWarning`` shape.
    analysis_metadata : dict | None
        Optional model/analysis context used to enrich issues with the
        member's section, story, connected nodes, etc. Tolerant of
        missing fields. Shapes accepted: ``member_info`` (list of dicts),
        ``updated_model`` (V2 StructuralModel JSON), ``material_name``.
    out_warnings : list | None
        Optional sink — the extractor appends any *new* warnings it
        produces (e.g. "design_check structure is malformed"). The caller
        is responsible for merging these back into the response.
    include_missing_design_check : bool
        When True (default), a ``missing_design_check`` issue is emitted
        if ``design_check`` is None. Set to False from pre-analysis stages
        (e.g. /api/v2/parse-ifc) where the absence is expected.
    """
    sink: list[AnalysisWarning] = out_warnings if out_warnings is not None else []

    ctx_index = build_context_index(analysis_metadata)

    issues: list[StructuralIssue] = []

    if design_check is None or not isinstance(design_check, dict):
        if include_missing_design_check:
            issues.append(_missing_design_check_issue())
    else:
        issues.extend(_issues_from_member_check(
            design_check.get("member_check"), sink, ctx_index,
        ))
        issues.extend(_issues_from_drift_check(
            design_check.get("drift_check"), sink,
        ))

    if warnings:
        issues.extend(_issues_from_warnings(warnings))

    # Summary counts
    by_type: dict[str, int] = {}
    by_severity: dict[str, int] = {}
    for iss in issues:
        by_type[iss.issue_type] = by_type.get(iss.issue_type, 0) + 1
        by_severity[iss.severity] = by_severity.get(iss.severity, 0) + 1

    return IssueExtractionResult(
        issues=issues,
        summary={
            "total": len(issues),
            "by_type": by_type,
            "by_severity": by_severity,
        },
    )
