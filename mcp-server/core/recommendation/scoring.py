"""Deterministic scoring + ranking for retrofit candidates.

Each :class:`RetrofitCandidate` is scored along five axes, all in [0, 1]
where higher = better for the candidate. The final ``total`` is a fixed
weighted sum so that two identical inputs always produce identical
output.

Axes (all heuristic — no external data, no LLM):

    safety_gain        How much the candidate is expected to reduce D/C
                       or drift. Larger overstress → larger headroom for
                       improvement → higher score.
    code_compliance    Strength / serviceability tied to a code clause
                       score higher than generic engineer-review fallbacks.
    relative_cost      *Inverse* cost. Cheap actions (replace one
                       section) score higher than expensive ones
                       (system-wide lateral retrofit).
    disruption         *Inverse* disruption. Single-member changes score
                       higher than story-scope or whole-building changes.
    side_effect_risk   *Inverse* risk. Local member sizing has less
                       side-effect risk than adding lateral resistance.

The weights are intentionally simple. They are also the levers a future
calibration step (real cost data, real safety models, real KDS clauses)
will tune.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable, Optional

from .schemas import (
    ActionType,
    IssueType,
    RetrofitCandidate,
    Severity,
    StructuralIssue,
)
from .taxonomy import (
    IssueCategory,
    classify_issue,
    PRIORITY_CRITICAL,
)


# ---------------------------------------------------------------------------
# Score axis weights — fixed for determinism.
# ---------------------------------------------------------------------------

WEIGHTS: dict[str, float] = {
    "safety_gain": 0.40,
    "code_compliance": 0.20,
    "relative_cost": 0.15,
    "disruption": 0.15,
    "side_effect_risk": 0.10,
}


# Per-action heuristic baselines. Values are in [0, 1] and were chosen so
# that single-member changes outrank system-wide ones, while everything
# beats engineer-review fallbacks.
_ACTION_BASELINES: dict[str, dict[str, float]] = {
    ActionType.INCREASE_SECTION: {
        "code_compliance": 0.85,
        "relative_cost":   0.70,
        "disruption":      0.80,
        "side_effect_risk": 0.75,
    },
    ActionType.CHANGE_MATERIAL: {
        "code_compliance": 0.75,
        "relative_cost":   0.55,
        "disruption":      0.70,
        "side_effect_risk": 0.60,
    },
    ActionType.ADD_MEMBER: {
        "code_compliance": 0.65,
        "relative_cost":   0.45,
        "disruption":      0.55,
        "side_effect_risk": 0.55,
    },
    ActionType.ADD_LATERAL_RESISTANCE: {
        "code_compliance": 0.70,
        "relative_cost":   0.35,
        "disruption":      0.40,
        "side_effect_risk": 0.50,
    },
    ActionType.CHANGE_SUPPORT: {
        "code_compliance": 0.55,
        "relative_cost":   0.45,
        "disruption":      0.50,
        "side_effect_risk": 0.45,
    },
    ActionType.REQUIRES_ENGINEER_REVIEW: {
        # Manual review can never outscore a code-grounded suggestion.
        "code_compliance": 0.30,
        "relative_cost":   0.50,
        "disruption":      0.50,
        "side_effect_risk": 0.50,
    },
}

_DEFAULT_BASELINE = {
    "code_compliance": 0.40,
    "relative_cost":   0.50,
    "disruption":      0.50,
    "side_effect_risk": 0.50,
}


@dataclass
class ScoreBreakdown:
    """Per-axis [0,1] scores plus the weighted total.

    Round-tripped via ``to_dict`` and attached to
    ``RetrofitCandidate.metadata["score"]`` so the API surface stays
    additive.
    """
    safety_gain: float = 0.0
    code_compliance: float = 0.0
    relative_cost: float = 0.0
    disruption: float = 0.0
    side_effect_risk: float = 0.0
    total: float = 0.0
    rationale: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Per-axis helpers
# ---------------------------------------------------------------------------

def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _safety_gain_for(
    candidate: RetrofitCandidate,
    issue: Optional[StructuralIssue],
) -> tuple[float, str]:
    """Estimate how much D/C reduction the candidate could plausibly buy.

    The deterministic layer doesn't actually size a new section, so we
    treat ``D/C`` overshoot as a proxy: the more overstressed, the more
    safety improvement headroom the candidate represents.
    """
    if candidate.action_type == ActionType.REQUIRES_ENGINEER_REVIEW:
        return 0.20, "engineer-review fallback — limited automatic gain"

    if issue is None:
        return 0.35, "no issue context — generic estimate"

    dcr = issue.demand_capacity_ratio or 0.0
    if dcr > 0:
        # Overshoot fraction normalized into ~[0,1]. D/C 1.0→0.0,
        # D/C 2.0→1.0 (capped). Avoids division blowups for tiny ratios.
        overshoot = max(0.0, dcr - 1.0)
        score = _clamp(overshoot / 1.0)
        return score, f"D/C={dcr:.2f} → overshoot={overshoot:.2f}"

    # No quantitative ratio — fall back to severity.
    if issue.severity == Severity.ERROR:
        return 0.50, "severity=error (no D/C)"
    if issue.severity == Severity.WARNING:
        return 0.25, "severity=warning"
    return 0.10, "severity=info"


def _apply_priority_boost(
    base: float,
    issue: Optional[StructuralIssue],
) -> float:
    """Critical issues (D/C ≥ 1.5) get a small additional bump.

    Bounded so it never lifts a non-error finding above an error finding
    by itself.
    """
    if issue is None:
        return base
    cls = classify_issue(issue)
    if cls.priority == PRIORITY_CRITICAL:
        return _clamp(base + 0.10)
    return base


def score_candidate(
    candidate: RetrofitCandidate,
    issue: Optional[StructuralIssue] = None,
) -> ScoreBreakdown:
    """Compute a deterministic [0,1] score breakdown for one candidate."""
    safety_gain, safety_reason = _safety_gain_for(candidate, issue)
    safety_gain = _apply_priority_boost(safety_gain, issue)

    baseline = _ACTION_BASELINES.get(candidate.action_type, _DEFAULT_BASELINE)

    code_compliance = _clamp(baseline["code_compliance"])
    # Engineer-review fallbacks lose a bit more compliance score because
    # they don't ship a typed action.
    if candidate.action_type == ActionType.REQUIRES_ENGINEER_REVIEW:
        code_compliance = max(0.0, code_compliance - 0.05)

    relative_cost = _clamp(baseline["relative_cost"])
    disruption = _clamp(baseline["disruption"])
    side_effect_risk = _clamp(baseline["side_effect_risk"])

    # Data-quality issues (missing_design_check, analysis warnings) can't
    # claim code-compliance benefits; cap them.
    if issue is not None:
        cls = classify_issue(issue)
        if cls.category == IssueCategory.DATA_QUALITY:
            code_compliance = min(code_compliance, 0.35)

    total = (
        WEIGHTS["safety_gain"]      * safety_gain
        + WEIGHTS["code_compliance"]  * code_compliance
        + WEIGHTS["relative_cost"]    * relative_cost
        + WEIGHTS["disruption"]       * disruption
        + WEIGHTS["side_effect_risk"] * side_effect_risk
    )
    total = round(_clamp(total), 4)

    return ScoreBreakdown(
        safety_gain=round(safety_gain, 4),
        code_compliance=round(code_compliance, 4),
        relative_cost=round(relative_cost, 4),
        disruption=round(disruption, 4),
        side_effect_risk=round(side_effect_risk, 4),
        total=total,
        rationale={
            "safety_gain": safety_reason,
            "action_baseline": candidate.action_type,
        },
    )


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def rank_candidates(
    candidates: Iterable[RetrofitCandidate],
    issues: Optional[Iterable[StructuralIssue]] = None,
) -> list[RetrofitCandidate]:
    """Score every candidate, attach the score, and return them sorted.

    Sort key (stable):
        1) ``-total`` (high first)
        2) issue priority (low number first)
        3) ``candidate_id`` (alphabetical tie-breaker)

    The score is attached as ``candidate.metadata["score"]`` so existing
    consumers stay compatible. Candidates are **mutated in place** —
    pass copies if the caller needs the originals untouched.
    """
    issue_by_id: dict[str, StructuralIssue] = {}
    if issues:
        for iss in issues:
            issue_by_id[iss.issue_id] = iss

    annotated: list[tuple[float, int, str, RetrofitCandidate]] = []
    for cand in candidates:
        parent = issue_by_id.get(cand.issue_id)
        score = score_candidate(cand, parent)
        # Mutate metadata so the JSON payload exposes the score.
        if not isinstance(cand.metadata, dict):
            cand.metadata = {}
        cand.metadata["score"] = score.to_dict()
        if parent is not None:
            cand.metadata["issue_classification"] = classify_issue(parent).to_dict()

        priority = (
            classify_issue(parent).priority if parent is not None
            else 3  # PRIORITY_LOW for orphan candidates
        )
        annotated.append((-score.total, priority, cand.candidate_id, cand))

    annotated.sort(key=lambda x: (x[0], x[1], x[2]))
    return [c for _, _, _, c in annotated]
