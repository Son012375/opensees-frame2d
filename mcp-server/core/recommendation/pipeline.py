"""Convenience adapter: analysis result → recommendation payload.

A thin orchestrator that:
    1. Normalizes legacy warnings (``list[str]``) into
       :class:`AnalysisWarning` objects.
    2. Runs ``extract_issues``.
    3. Runs ``generate_candidates``.
    4. Returns a JSON-ready dict that the FastAPI layer can splice
       directly into its response.

Keeping this in the core layer (not in ``main_simple.py``) means the
MCP server, tests, and future CLI tools all share the same call site.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Iterable, Optional

from .schemas import (
    AnalysisCaseSummary,
    AnalysisEnvelope,
    AnalysisWarning,
    MemberDesignCheck,
    MemberForceSummary,
    Severity,
)
from .issue_extractor import extract_issues
from .candidate_generator import generate_candidates


# ---------------------------------------------------------------------------
# Warning normalization
# ---------------------------------------------------------------------------

def normalize_warnings(
    raw: Optional[Iterable[Any]],
    stage: str = "",
) -> list[AnalysisWarning]:
    """Coerce legacy/mixed warning shapes into ``AnalysisWarning``."""
    out: list[AnalysisWarning] = []
    if not raw:
        return out
    for w in raw:
        if isinstance(w, AnalysisWarning):
            out.append(w)
        elif isinstance(w, dict):
            out.append(AnalysisWarning(
                code=str(w.get("code", "warning")),
                message=str(w.get("message", "")),
                severity=str(w.get("severity", Severity.WARNING)),
                stage=str(w.get("stage", stage)),
                recoverable=bool(w.get("recoverable", True)),
                detail=w.get("detail"),
            ))
        else:
            out.append(AnalysisWarning.from_legacy_string(str(w), stage=stage))
    return out


def warnings_to_payload(warnings: Iterable[AnalysisWarning]) -> dict[str, Any]:
    """Return the dual representation kept in the API response.

    Returns
    -------
    {
        "warnings": [ {...AnalysisWarning...}, ... ],     # structured
        "warning_messages": ["code: msg", ...],           # legacy strings
    }
    """
    structured: list[dict[str, Any]] = []
    legacy: list[str] = []
    for w in warnings:
        structured.append(w.to_dict())
        legacy.append(f"{w.code}: {w.message}" if w.code else w.message)
    return {"warnings": structured, "warning_messages": legacy}


# ---------------------------------------------------------------------------
# Adapters for the structured Analysis* contracts
# ---------------------------------------------------------------------------

def envelope_from_dict(env: Optional[dict]) -> AnalysisEnvelope:
    if not isinstance(env, dict):
        return AnalysisEnvelope()
    return AnalysisEnvelope(
        max_dx_mm=float(env.get("max_dx_mm", 0.0) or 0.0),
        max_dy_mm=float(env.get("max_dy_mm", 0.0) or 0.0),
        max_dz_mm=float(env.get("max_dz_mm", 0.0) or 0.0),
        max_drift_x=float(env.get("max_drift_x", 0.0) or 0.0),
        max_drift_y=float(env.get("max_drift_y", 0.0) or 0.0),
        max_moment_kNm=float(env.get("max_moment_kNm", 0.0) or 0.0),
        max_axial_kN=float(env.get("max_axial_kN", 0.0) or 0.0),
        max_shear_kN=float(env.get("max_shear_kN", 0.0) or 0.0),
    )


def case_summaries_from_dict(
    case_data: Optional[dict],
    combo_names: Optional[Iterable[str]] = None,
) -> list[AnalysisCaseSummary]:
    if not isinstance(case_data, dict):
        return []
    combos = set(combo_names or [])
    out: list[AnalysisCaseSummary] = []
    for name, cd in case_data.items():
        if not isinstance(cd, dict):
            continue
        s = cd.get("summary") or {}
        out.append(AnalysisCaseSummary(
            case_name=str(name),
            is_combination=name in combos,
            max_dx_mm=float(s.get("max_dx_mm", 0.0) or 0.0),
            max_dy_mm=float(s.get("max_dy_mm", 0.0) or 0.0),
            max_dz_mm=float(s.get("max_dz_mm", 0.0) or 0.0),
            max_drift_x=float(s.get("max_drift_x", 0.0) or 0.0),
            max_drift_y=float(s.get("max_drift_y", 0.0) or 0.0),
            max_moment_kNm=float(s.get("max_moment_kNm", 0.0) or 0.0),
            max_axial_kN=float(s.get("max_axial_kN", 0.0) or 0.0),
            max_shear_kN=float(s.get("max_shear_kN", 0.0) or 0.0),
        ))
    return out


def member_checks_from_design_check(
    design_check: Optional[dict],
    material: str = "",
) -> list[MemberDesignCheck]:
    if not isinstance(design_check, dict):
        return []
    mc = design_check.get("member_check")
    if not isinstance(mc, dict):
        return []
    out: list[MemberDesignCheck] = []
    for m in mc.get("members") or []:
        if not isinstance(m, dict):
            continue
        try:
            mid = int(m.get("member_id", m.get("element_id", 0)))
        except (TypeError, ValueError):
            continue
        ratios = m.get("ratios") or {}
        interaction = float(ratios.get("interaction", ratios.get("H1", 0)) or 0)
        shear = float(ratios.get("shear", 0) or 0)
        out.append(MemberDesignCheck(
            member_id=mid,
            element_id=m.get("element_id"),
            member_type=str(m.get("type") or m.get("member_type", "")),
            section=str(m.get("section", "")),
            material=material,
            story=m.get("story"),
            connected_node_ids=list(m.get("connected_node_ids") or []),
            governing_combo=str(m.get("governing_combo", "")),
            demand_capacity_ratio=max(interaction, shear),
            interaction_ratio=interaction,
            shear_ratio=shear,
            status=str(m.get("status", "OK")),
            raw=m,
        ))
    return out


# ---------------------------------------------------------------------------
# Main entry point used by the API
# ---------------------------------------------------------------------------

def build_recommendation_payload(
    *,
    design_check: Optional[dict] = None,
    raw_warnings: Optional[Iterable[Any]] = None,
    analysis_metadata: Optional[dict] = None,
    stage: str = "analysis",
) -> dict[str, Any]:
    """Build the additive recommendation block for the API response.

    Returns a dict with these top-level keys (all additive to existing
    response shape):

        warnings              : list[AnalysisWarning.to_dict()]
        warning_messages      : list[str]   (legacy mirror)
        issues                : list[StructuralIssue.to_dict()]
        recommendation_candidates : list[RetrofitCandidate.to_dict()]
        recommendation_summary    : dict
    """
    normalized = normalize_warnings(raw_warnings, stage=stage)

    # Extractor may emit additional warnings — collect them into the same list.
    extra_warnings: list[AnalysisWarning] = []
    issue_result = extract_issues(
        design_check=design_check,
        warnings=normalized,
        analysis_metadata=analysis_metadata,
        out_warnings=extra_warnings,
    )
    normalized.extend(extra_warnings)

    candidates = generate_candidates(issue_result.issues)

    payload = warnings_to_payload(normalized)
    payload["issues"] = [i.to_dict() for i in issue_result.issues]
    payload["recommendation_candidates"] = [c.to_dict() for c in candidates]
    payload["recommendation_summary"] = {
        "num_issues": issue_result.summary.get("total", 0),
        "issues_by_type": issue_result.summary.get("by_type", {}),
        "issues_by_severity": issue_result.summary.get("by_severity", {}),
        "num_candidates": len(candidates),
        "rag_enabled": False,
        "llm_enabled": False,
        "note": (
            "Deterministic placeholder pipeline. Candidates are NOT a "
            "final design — re-analysis and engineer review required."
        ),
    }
    return payload
