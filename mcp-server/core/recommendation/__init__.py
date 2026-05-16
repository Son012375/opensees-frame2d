"""Recommendation pipeline foundation.

Deterministic layer that turns analysis + design-check results into a
structured stream of design issues and (placeholder) retrofit candidates
that a downstream KDS-RAG + LLM layer can later enrich.

Scope (intentional):
    * NO RAG retrieval, NO LLM calls, NO vector DB.
    * NO real optimization — candidates are deterministic placeholders.
    * Numerical decisions stay here; LLM will only narrate / cite later.

Public surface:
    from core.recommendation import (
        AnalysisWarning, AnalysisEnvelope, AnalysisCaseSummary,
        MemberForceSummary, MemberDesignCheck,
        StructuralIssue, IssueExtractionResult, extract_issues,
        RetrofitCandidate, generate_candidates,
        CodeReference,
    )
"""
from __future__ import annotations

from .schemas import (
    AnalysisEnvelope,
    AnalysisCaseSummary,
    MemberForceSummary,
    MemberDesignCheck,
    AnalysisWarning,
    StructuralIssue,
    IssueExtractionResult,
    RetrofitCandidate,
    CodeReference,
    Severity,
    IssueSource,
    IssueType,
    ActionType,
    Confidence,
)
from .issue_extractor import extract_issues
from .candidate_generator import generate_candidates
from .pipeline import (
    build_recommendation_payload,
    normalize_warnings,
    warnings_to_payload,
    envelope_from_dict,
    case_summaries_from_dict,
    member_checks_from_design_check,
)

__all__ = [
    "AnalysisEnvelope",
    "AnalysisCaseSummary",
    "MemberForceSummary",
    "MemberDesignCheck",
    "AnalysisWarning",
    "StructuralIssue",
    "IssueExtractionResult",
    "RetrofitCandidate",
    "CodeReference",
    "Severity",
    "IssueSource",
    "IssueType",
    "ActionType",
    "Confidence",
    "extract_issues",
    "generate_candidates",
    "build_recommendation_payload",
    "normalize_warnings",
    "warnings_to_payload",
    "envelope_from_dict",
    "case_summaries_from_dict",
    "member_checks_from_design_check",
]
