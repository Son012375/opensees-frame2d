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
        CodeReference, build_recommendation_payload,
        MODE_ANALYZE, MODE_PARSE_ONLY,
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
    ChangeOperation,
    Confidence,
)
from .ids import make_issue_id, make_candidate_id, slugify
from .context import MemberContext, MemberContextIndex, build_context_index
from .issue_extractor import extract_issues
from .candidate_generator import generate_candidates, _should_block_auto_candidate
from .pipeline import (
    MODE_ANALYZE,
    MODE_PARSE_ONLY,
    build_recommendation_payload,
    normalize_warnings,
    warnings_to_payload,
    envelope_from_dict,
    case_summaries_from_dict,
    member_checks_from_design_check,
)
from .registry import (
    RuleEntry,
    RuleHandler,
    RuleRegistry,
    default_registry,
    get_rule_for_issue,
    list_registered_handlers,
    register_rule,
)
from .scoring import (
    ScoreBreakdown,
    WEIGHTS as SCORING_WEIGHTS,
    rank_candidates,
    score_candidate,
)
from .taxonomy import (
    IssueCategory,
    IssueClassification,
    PRIORITY_CRITICAL,
    PRIORITY_HIGH,
    PRIORITY_LOW,
    PRIORITY_MEDIUM,
    category_counts,
    classify_issue,
    priority_counts,
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
    "ChangeOperation",
    "Confidence",
    "MemberContext",
    "MemberContextIndex",
    "build_context_index",
    "make_issue_id",
    "make_candidate_id",
    "slugify",
    "extract_issues",
    "generate_candidates",
    "_should_block_auto_candidate",
    "build_recommendation_payload",
    "MODE_ANALYZE",
    "MODE_PARSE_ONLY",
    "normalize_warnings",
    "warnings_to_payload",
    "envelope_from_dict",
    "case_summaries_from_dict",
    "member_checks_from_design_check",
    # Rule registry
    "RuleEntry",
    "RuleHandler",
    "RuleRegistry",
    "default_registry",
    "get_rule_for_issue",
    "list_registered_handlers",
    "register_rule",
    # Scoring / ranking
    "ScoreBreakdown",
    "SCORING_WEIGHTS",
    "rank_candidates",
    "score_candidate",
    # Taxonomy
    "IssueCategory",
    "IssueClassification",
    "PRIORITY_CRITICAL",
    "PRIORITY_HIGH",
    "PRIORITY_LOW",
    "PRIORITY_MEDIUM",
    "category_counts",
    "classify_issue",
    "priority_counts",
]
