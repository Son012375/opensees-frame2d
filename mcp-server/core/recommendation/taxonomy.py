"""Issue taxonomy: category + priority for downstream ranking / UI grouping.

The deterministic recommendation engine reasons about :class:`StructuralIssue`
objects produced by :mod:`core.recommendation.issue_extractor`. Those carry
an ``issue_type`` (one of :class:`IssueType.*`) and a ``severity``, but no
higher-level grouping. This module adds two derived classifications:

    * :class:`IssueCategory`  — coarse engineering bucket (strength,
      serviceability, global stability, data quality).
    * priority (int)          — 0 = critical, 1 = high, 2 = medium,
      3 = low. Lower number = act first. The scorer uses this as one of
      its inputs; the UI can group/sort by it.

Both are *derived* from existing :class:`StructuralIssue` fields — no new
fields are added to the schema. The :func:`classify_issue` result is
attached only to ``recommendation_summary`` and (optionally) to
candidate ``metadata``.

Keep the rules deterministic and short. Anything debatable belongs in
the future RAG/LLM layer, not here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .schemas import IssueType, Severity, StructuralIssue


class IssueCategory:
    STRENGTH = "strength"               # member-level capacity exceedance
    SERVICEABILITY = "serviceability"   # drift / deflection / vibration
    GLOBAL_STABILITY = "global_stability"  # P-Δ, base reactions, overall
    DATA_QUALITY = "data_quality"       # missing DC, analysis warnings
    UNCLASSIFIED = "unclassified"       # issue_type we don't recognize


# Priority ordering — lower is more urgent.
PRIORITY_CRITICAL = 0
PRIORITY_HIGH = 1
PRIORITY_MEDIUM = 2
PRIORITY_LOW = 3


@dataclass
class IssueClassification:
    category: str
    priority: int

    def to_dict(self) -> dict:
        return {"category": self.category, "priority": self.priority}


# Type → category mapping. Kept as a module-level dict so callers can
# introspect/extend without monkey-patching a function.
#
# Only the two issue types that genuinely describe data/observability
# problems (``missing_design_check``, ``analysis_warning``) map to
# DATA_QUALITY. Any other unknown issue_type lands in UNCLASSIFIED so
# the UI doesn't mislead the engineer into thinking it's a data issue.
_TYPE_TO_CATEGORY: dict[str, str] = {
    IssueType.STRENGTH_EXCEEDED: IssueCategory.STRENGTH,
    IssueType.SHEAR_EXCEEDED: IssueCategory.STRENGTH,
    IssueType.DRIFT_EXCEEDED: IssueCategory.SERVICEABILITY,
    IssueType.MISSING_DESIGN_CHECK: IssueCategory.DATA_QUALITY,
    IssueType.ANALYSIS_WARNING: IssueCategory.DATA_QUALITY,
}


def classify_issue(issue: StructuralIssue) -> IssueClassification:
    """Return (category, priority) derived from issue type + severity + D/C.

    Unknown issue_types classify as :data:`IssueCategory.UNCLASSIFIED` —
    they are not data-quality problems and must not be silently bucketed
    as such. Downstream code generates an engineer-review candidate for
    them, which is the correct behavior.
    """
    category = _TYPE_TO_CATEGORY.get(
        issue.issue_type, IssueCategory.UNCLASSIFIED,
    )
    priority = _priority_for(issue, category)
    return IssueClassification(category=category, priority=priority)


def _priority_for(issue: StructuralIssue, category: str) -> int:
    """Heuristic priority. Severity is the dominant signal, then D/C.

    Rules (in order):
        * error-severity strength/shear with D/C ≥ 1.5  → CRITICAL
        * error-severity strength/shear or drift        → HIGH
        * warning-severity analysis warning              → MEDIUM
        * info-severity / missing_design_check (warning) → LOW
    """
    sev = issue.severity
    dcr = issue.demand_capacity_ratio or 0.0

    if category == IssueCategory.STRENGTH:
        if sev == Severity.ERROR and dcr >= 1.5:
            return PRIORITY_CRITICAL
        if sev == Severity.ERROR:
            return PRIORITY_HIGH
        return PRIORITY_MEDIUM

    if category == IssueCategory.SERVICEABILITY:
        if sev == Severity.ERROR and dcr >= 1.5:
            return PRIORITY_HIGH      # serviceability is rarely "critical"
        if sev == Severity.ERROR:
            return PRIORITY_MEDIUM
        return PRIORITY_LOW

    if category == IssueCategory.GLOBAL_STABILITY:
        return PRIORITY_HIGH if sev == Severity.ERROR else PRIORITY_MEDIUM

    if category == IssueCategory.DATA_QUALITY:
        if sev == Severity.ERROR:
            return PRIORITY_HIGH
        if issue.issue_type == IssueType.MISSING_DESIGN_CHECK:
            return PRIORITY_LOW
        return PRIORITY_MEDIUM

    # UNCLASSIFIED — be conservative; surface as MEDIUM so the engineer
    # sees it but doesn't get drowned out by real strength/drift NGs.
    if sev == Severity.ERROR:
        return PRIORITY_HIGH
    return PRIORITY_MEDIUM


def category_counts(issues) -> dict[str, int]:
    """Group an iterable of issues by category for summary surfacing."""
    out: dict[str, int] = {}
    for iss in issues:
        c = classify_issue(iss).category
        out[c] = out.get(c, 0) + 1
    return out


def priority_counts(issues) -> dict[str, int]:
    """Group an iterable of issues by priority (string-keyed for JSON)."""
    out: dict[str, int] = {}
    for iss in issues:
        p = classify_issue(iss).priority
        key = str(p)
        out[key] = out.get(key, 0) + 1
    return out
