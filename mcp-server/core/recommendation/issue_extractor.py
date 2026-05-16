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

import uuid
from typing import Any, Iterable, Optional

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

def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


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
    )


def _aisc_h1_ref() -> CodeReference:
    return CodeReference(
        standard_id="KDS 41 31 00",
        clause_id="H1",
        title="조합응력 상관식 (AISC 360 H1)",
    )


# ---------------------------------------------------------------------------
# Per-source extractors
# ---------------------------------------------------------------------------

def _issues_from_member_check(
    member_check: Optional[dict],
    out_warnings: list[AnalysisWarning],
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

        member_type = m.get("type") or m.get("member_type", "")
        section = m.get("section", "")
        combo = m.get("governing_combo", "")

        # Strength: interaction > 1.0 → strength_exceeded
        if interaction > 1.0:
            issues.append(StructuralIssue(
                issue_id=_new_id("iss"),
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
                evidence={
                    "ratios": ratios,
                    "demand": m.get("demand"),
                    "capacity": m.get("capacity"),
                    "section": section,
                    "type": member_type,
                    "story": m.get("story"),
                },
                code_refs=[_aisc_h1_ref()],
            ))

        # Shear: separate issue when shear > 1.0
        if shear > 1.0:
            issues.append(StructuralIssue(
                issue_id=_new_id("iss"),
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
                evidence={
                    "shear_ratio": shear,
                    "demand": m.get("demand"),
                    "capacity": m.get("capacity"),
                    "section": section,
                    "type": member_type,
                },
                code_refs=[_aisc_h1_ref()],
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

        issues.append(StructuralIssue(
            issue_id=_new_id("iss"),
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
    """Lift each AnalysisWarning into a low-severity issue.

    Used so that the recommendation layer can surface "analysis_warning"
    alongside real strength/drift failures.
    """
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
            # legacy plain string
            parsed = AnalysisWarning.from_legacy_string(str(w))
            code, message, stage = parsed.code, parsed.message, parsed.stage
            severity = parsed.severity
            detail = {}

        # Errors propagate as errors; warnings stay warnings.
        sev = Severity.ERROR if severity == Severity.ERROR else Severity.WARNING
        issues.append(StructuralIssue(
            issue_id=_new_id("iss"),
            issue_type=IssueType.ANALYSIS_WARNING,
            severity=sev,
            source=IssueSource.WARNING,
            description=f"[{code}] {message}",
            evidence={"code": code, "stage": stage, "detail": detail},
        ))
    return issues


def _missing_design_check_issue() -> StructuralIssue:
    return StructuralIssue(
        issue_id=_new_id("iss"),
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
) -> IssueExtractionResult:
    """Build a normalized issue list from heterogeneous analysis output.

    Parameters
    ----------
    design_check : dict | None
        The ``run_design_check`` output (``overall_status``, ``drift_check``,
        ``member_check``, ...). May be ``None`` if the design check failed
        or was skipped.
    warnings : iterable
        Either legacy ``list[str]``, list of ``AnalysisWarning`` instances,
        or list of dicts in the ``AnalysisWarning`` shape.
    analysis_metadata : dict | None
        Currently unused but reserved — lets future logic key issues on
        e.g. P-Delta vs. linear.
    out_warnings : list | None
        Optional sink — the extractor appends any *new* warnings it
        produces (e.g. "design_check structure is malformed"). The caller
        is responsible for merging these back into the response.
    """
    sink: list[AnalysisWarning] = out_warnings if out_warnings is not None else []

    issues: list[StructuralIssue] = []

    if design_check is None or not isinstance(design_check, dict):
        issues.append(_missing_design_check_issue())
    else:
        issues.extend(_issues_from_member_check(
            design_check.get("member_check"), sink,
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
