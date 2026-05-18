"""Reanalysis-driven candidate evaluator.

For each :class:`RetrofitCandidate`, this module:

    1. Applies the candidate via :func:`apply_candidate_to_model`.
    2. Re-runs the 3D frame analyzer on the modified model.
    3. Re-runs ``run_design_check`` against the new analysis.
    4. Computes improvement vs. a *baseline* (the same load cases /
       combinations the caller already ran).
    5. Detects NEW NG members or drift checks that didn't exist in the
       baseline — these candidates are hard-rejected.
    6. Emits a verified :class:`ScoreBreakdown` (``method="reanalyzed_v1"``,
       ``verified=True``) the ranking step then sorts on.

The caller MUST provide the same load_cases / load_combinations /
seismic_report / building_model that produced the baseline; otherwise
the comparison is meaningless. We don't rebuild loads here — that
responsibility stays in the API layer (which cached them at analyze
time).

Heavy lifting (OpenSees analysis, design check) happens through the
existing modules:

    * ``core.frame_3d.analyze_from_model``
    * ``core.design_check.run_design_check``
    * ``core.structural_model.StructuralModel.from_json``

OpenSees has process-global state, so callers must serialize evaluate()
invocations. The API layer enforces ``MAX_PARALLEL_EVALS=1``.
"""
from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import Any, Optional

from .apply_candidate import (
    ChangeDiff,
    InapplicableCandidateError,
    apply_candidate_to_model,
)
from .schemas import RetrofitCandidate
from .scoring import ScoreBreakdown
from .section_catalog import get_section_metadata


# ---------------------------------------------------------------------------
# Status / verified score identity
# ---------------------------------------------------------------------------

VERIFIED_SCORE_METHOD = "reanalyzed_v1"
VERIFIED_STATUS = "verified"

STATUS_EVALUATED = "evaluated"
STATUS_SKIPPED_INAPPLICABLE = "skipped_not_applicable"
STATUS_REJECTED_FAILED = "rejected_analysis_failed"
STATUS_REJECTED_NEW_NG = "rejected_new_ng"


# ---------------------------------------------------------------------------
# Baseline contract
# ---------------------------------------------------------------------------

@dataclass
class BaselineSummary:
    """The minimum baseline info the evaluator needs to compare against.

    Construct one from the response of /api/v2/analyze (held in the
    server-side analysis cache).
    """
    max_interaction_ratio: float = 0.0
    max_drift_ratio: float = 0.0
    ng_member_ids: set[int] = field(default_factory=set)
    ng_drift_keys: set[tuple[int, str]] = field(default_factory=set)
    ok_member_ids: set[int] = field(default_factory=set)
    total_member_count: int = 0

    @classmethod
    def from_design_check(cls, design_check: Optional[dict]) -> "BaselineSummary":
        if not design_check:
            return cls()
        member_check = design_check.get("member_check") or {}
        drift_check = design_check.get("drift_check") or {}
        summary = design_check.get("summary") or {}

        ng_member_ids: set[int] = set()
        ok_member_ids: set[int] = set()
        for m in (member_check.get("members") or []):
            mid = m.get("member_id")
            if mid is None:
                continue
            if m.get("status") == "NG":
                ng_member_ids.add(int(mid))
            else:
                ok_member_ids.add(int(mid))

        ng_drift_keys: set[tuple[int, str]] = set()
        for chk in (drift_check.get("checks") or []):
            if chk.get("status") == "NG":
                ng_drift_keys.add(
                    (int(chk.get("story", -1)),
                     str(chk.get("direction", "")))
                )

        return cls(
            max_interaction_ratio=float(summary.get("max_interaction_ratio") or 0.0),
            max_drift_ratio=float(summary.get("max_drift_ratio") or 0.0),
            ng_member_ids=ng_member_ids,
            ng_drift_keys=ng_drift_keys,
            ok_member_ids=ok_member_ids,
            total_member_count=len(member_check.get("members") or []),
        )


# ---------------------------------------------------------------------------
# Evaluation record
# ---------------------------------------------------------------------------

@dataclass
class CandidateEvaluation:
    """Outcome of one evaluate() call. Serializable as dict."""
    candidate_id: str
    status: str
    diff: Optional[ChangeDiff] = None
    metrics: dict[str, Any] = field(default_factory=dict)
    improvement: dict[str, Any] = field(default_factory=dict)
    score: Optional[ScoreBreakdown] = None
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "status": self.status,
            "diff": self.diff.to_dict() if self.diff else None,
            "metrics": dict(self.metrics),
            "improvement": dict(self.improvement),
            "score": self.score.to_dict() if self.score else None,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------

def _extract_metrics_from_dc(dc: dict) -> tuple[
    float, float, int, int, set[int], set[tuple[int, str]],
]:
    """Pull the summary numbers + NG sets out of a design_check dict."""
    summary = dc.get("summary") or {}
    max_dcr = float(summary.get("max_interaction_ratio") or 0.0)
    max_drift = float(summary.get("max_drift_ratio") or 0.0)

    member_check = dc.get("member_check") or {}
    ng_member_ids: set[int] = set()
    for m in (member_check.get("members") or []):
        if m.get("status") == "NG" and m.get("member_id") is not None:
            ng_member_ids.add(int(m["member_id"]))

    drift_check = dc.get("drift_check") or {}
    ng_drift_keys: set[tuple[int, str]] = set()
    for chk in (drift_check.get("checks") or []):
        if chk.get("status") == "NG":
            ng_drift_keys.add(
                (int(chk.get("story", -1)), str(chk.get("direction", "")))
            )

    return (
        max_dcr, max_drift,
        len(ng_member_ids), len(ng_drift_keys),
        ng_member_ids, ng_drift_keys,
    )


def _weight_proxy_delta(
    diff: ChangeDiff,
    nodes_by_id: dict[int, dict],
    elements_by_id: dict[int, dict],
) -> float:
    """Sum of (new_area − old_area) × length over changed members, in
    mm³ (a stand-in for steel weight when ρ is constant).

    Falls back to 0 if any required catalog metadata is missing.
    """
    total = 0.0
    for row in diff.changed_members:
        eid = row.get("element_id")
        if eid is None:
            continue
        elem = elements_by_id.get(eid)
        if elem is None:
            continue
        ni = nodes_by_id.get(elem.get("node_i"))
        nj = nodes_by_id.get(elem.get("node_j"))
        if ni is None or nj is None:
            continue
        # Length in mm.
        dx = (nj.get("x", 0.0) - ni.get("x", 0.0))
        dy = (nj.get("y", 0.0) - ni.get("y", 0.0))
        dz = (nj.get("z", 0.0) - ni.get("z", 0.0))
        length_mm = (dx * dx + dy * dy + dz * dz) ** 0.5

        old_meta = get_section_metadata(row.get("section_from"))
        new_meta = get_section_metadata(row.get("section_to"))
        if old_meta is None or new_meta is None:
            continue
        # area_mm2 already converted by the catalog
        delta_area = new_meta.area_mm2 - old_meta.area_mm2
        total += delta_area * length_mm
    return total


# ---------------------------------------------------------------------------
# Verified score
# ---------------------------------------------------------------------------

def _verified_score(
    *,
    baseline: BaselineSummary,
    metrics: dict[str, Any],
    new_ng_count: int,
) -> ScoreBreakdown:
    """Compute the reanalyzed-score breakdown.

    All five axes in [0, 1]; weights inherited from
    :data:`scoring.WEIGHTS` for continuity with the heuristic
    pipeline. NEW NG candidates are hard-rejected upstream so they
    never reach this function.
    """
    # Lazy import to avoid circulars at module load.
    from .scoring import WEIGHTS

    baseline_ng = len(baseline.ng_member_ids) + len(baseline.ng_drift_keys)
    new_ng_total = metrics["ng_member_count"] + metrics["ng_drift_count"]

    # safety_gain: fraction of baseline NG resolved.
    if baseline_ng > 0:
        resolved = max(0, baseline_ng - new_ng_total)
        safety_gain = min(1.0, resolved / baseline_ng)
    else:
        # Nothing was wrong to begin with — no gain to demonstrate.
        safety_gain = 0.5

    # code_compliance: 1.0 if both worst-case ratios are ≤ 1.0 after
    # the change; scale linearly otherwise.
    max_ratio_after = max(
        metrics.get("max_interaction_ratio", 0.0),
        metrics.get("max_drift_ratio", 0.0),
    )
    if max_ratio_after <= 1.0:
        code_compliance = 1.0
    else:
        # 1.5x and above gets the floor; tighter overruns recover.
        code_compliance = max(0.0, 1.0 - (max_ratio_after - 1.0) / 0.5)

    # relative_cost: cheaper changes (smaller weight increase) score higher.
    weight_proxy = float(metrics.get("weight_proxy", 0.0))
    # Normalize against an arbitrary 1 ton steel reference
    # (≈ 1e8 mm³ at ρ_steel=7850 kg/m³ → 1000 kg = 1e8/7.85 mm³ ≈ 1.27e7 mm³).
    # Pick 5e7 mm³ as the "expensive" reference for sub-ton changes.
    cost_norm = max(0.0, min(1.0, weight_proxy / 5e7))
    relative_cost = 1.0 - cost_norm

    # disruption: fewer changed members → less disruption.
    changed = float(metrics.get("changed_member_count", 0))
    total_members = max(1, baseline.total_member_count or 1)
    disruption = 1.0 - min(1.0, changed / total_members)

    # side_effect_risk: any new NG (even if upstream catches some)
    # drives this to zero. Here it should be 1.0 because the caller
    # filters rejected_new_ng before calling.
    side_effect_risk = 1.0 if new_ng_count == 0 else 0.0

    total = (
        WEIGHTS["safety_gain"] * safety_gain
        + WEIGHTS["code_compliance"] * code_compliance
        + WEIGHTS["relative_cost"] * relative_cost
        + WEIGHTS["disruption"] * disruption
        + WEIGHTS["side_effect_risk"] * side_effect_risk
    )

    return ScoreBreakdown(
        safety_gain=round(safety_gain, 4),
        code_compliance=round(code_compliance, 4),
        relative_cost=round(relative_cost, 4),
        disruption=round(disruption, 4),
        side_effect_risk=round(side_effect_risk, 4),
        total=round(max(0.0, min(1.0, total)), 4),
        rationale={
            "method": VERIFIED_SCORE_METHOD,
            "baseline_ng_total": str(baseline_ng),
            "new_ng_total": str(new_ng_total),
            "max_ratio_after": f"{max_ratio_after:.3f}",
            "weight_proxy_mm3": f"{weight_proxy:.0f}",
        },
        method=VERIFIED_SCORE_METHOD,
        verified=True,
        status=VERIFIED_STATUS,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def evaluate_candidate(
    model_json: dict,
    load_cases: dict,
    load_combinations: Optional[dict],
    candidate: RetrofitCandidate,
    baseline: BaselineSummary,
    *,
    building_model: Any = None,
    seismic_report: Optional[dict] = None,
    fy_MPa: Optional[float] = None,
    E_MPa: Optional[float] = None,
) -> CandidateEvaluation:
    """Reanalyze ``model_json`` with ``candidate`` applied.

    Inapplicable candidates short-circuit before any OpenSees work.
    Analysis failures are captured as ``rejected_analysis_failed`` with
    a truncated traceback; new-NG outcomes are demoted to
    ``rejected_new_ng``.
    """
    # 1) Apply
    try:
        new_model_json, diff = apply_candidate_to_model(model_json, candidate)
    except InapplicableCandidateError as e:
        return CandidateEvaluation(
            candidate_id=candidate.candidate_id,
            status=STATUS_SKIPPED_INAPPLICABLE,
            error=e.reason,
        )

    # 2) Re-analyze + 3) re-design-check
    try:
        # Local imports — these pull in OpenSees which is expensive.
        from core.frame_3d import analyze_from_model  # type: ignore
        from core.structural_model import StructuralModel  # type: ignore
        from core.design_check import run_design_check  # type: ignore

        model_obj = StructuralModel.from_json(new_model_json)
        multi = analyze_from_model(model_obj, load_cases, load_combinations)
        dc = run_design_check(multi, building_model, seismic_report)
    except Exception as e:  # noqa: BLE001 — surface every kind of failure
        tb = traceback.format_exc(limit=3)
        return CandidateEvaluation(
            candidate_id=candidate.candidate_id,
            status=STATUS_REJECTED_FAILED,
            diff=diff,
            error=f"{type(e).__name__}: {e}\n{tb}",
        )

    # 4) Metric extraction
    (
        max_dcr, max_drift,
        ng_member_count, ng_drift_count,
        new_ng_member_ids, new_ng_drift_keys,
    ) = _extract_metrics_from_dc(dc)

    nodes_by_id = {n["id"]: n for n in new_model_json.get("nodes", [])
                   if isinstance(n, dict) and "id" in n}
    elements_by_id = {e["id"]: e for e in new_model_json.get("elements", [])
                      if isinstance(e, dict) and "id" in e}
    weight_proxy = _weight_proxy_delta(diff, nodes_by_id, elements_by_id)

    metrics = {
        "max_interaction_ratio": max_dcr,
        "max_drift_ratio": max_drift,
        "ng_member_count": ng_member_count,
        "ng_drift_count": ng_drift_count,
        "weight_proxy": weight_proxy,
        "changed_member_count": diff.changed_member_count,
        "analysis_succeeded": True,
    }

    improvement = {
        "dcr_delta": round(max_dcr - baseline.max_interaction_ratio, 6),
        "drift_delta": round(max_drift - baseline.max_drift_ratio, 6),
        "ng_member_delta": ng_member_count - len(baseline.ng_member_ids),
        "ng_drift_delta": ng_drift_count - len(baseline.ng_drift_keys),
    }

    # 5) New-NG detection: members that were OK in baseline now NG,
    # or drift keys that didn't exist as NG in baseline.
    new_ng_members = new_ng_member_ids - baseline.ng_member_ids
    new_ng_drifts = new_ng_drift_keys - baseline.ng_drift_keys
    introduced_new_ng = bool(new_ng_members or new_ng_drifts)

    if introduced_new_ng:
        improvement["new_ng_members"] = sorted(new_ng_members)
        improvement["new_ng_drifts"] = [
            list(k) for k in sorted(new_ng_drifts)
        ]
        return CandidateEvaluation(
            candidate_id=candidate.candidate_id,
            status=STATUS_REJECTED_NEW_NG,
            diff=diff,
            metrics=metrics,
            improvement=improvement,
            error=(
                f"introduced new NG: members={sorted(new_ng_members)}, "
                f"drifts={sorted(new_ng_drifts)}"
            ),
        )

    # 6) Verified score (no new NG)
    score = _verified_score(
        baseline=baseline,
        metrics=metrics,
        new_ng_count=0,
    )

    return CandidateEvaluation(
        candidate_id=candidate.candidate_id,
        status=STATUS_EVALUATED,
        diff=diff,
        metrics=metrics,
        improvement=improvement,
        score=score,
    )
