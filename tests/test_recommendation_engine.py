"""Focused tests for the deterministic recommendation engine foundation.

Scope:
    * Issue taxonomy: category + priority derivation.
    * Rule registry: dispatch and overrideability.
    * Scoring: per-axis breakdown is deterministic, sensible ordering.
    * Ranking: stable order, score attached to candidate.metadata.
    * Pipeline integration: rank=True flag, summary surfaces.
    * No auto KDS-RAG attachment in the deterministic pipeline.
    * Graceful behavior on partial data.

Run:
    pytest tests/test_recommendation_engine.py -q
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mcp-server"))

from core.recommendation import (  # noqa: E402
    ActionType,
    ChangeOperation,
    IssueCategory,
    IssueSource,
    IssueType,
    MODE_ANALYZE,
    PRIORITY_CRITICAL,
    PRIORITY_HIGH,
    PRIORITY_LOW,
    PRIORITY_MEDIUM,
    RetrofitCandidate,
    RuleRegistry,
    SCORE_METHOD,
    SCORE_VERIFIED,
    SCORING_WEIGHTS,
    ScoreBreakdown,
    Severity,
    StructuralIssue,
    build_recommendation_payload,
    category_counts,
    classify_issue,
    default_registry,
    generate_candidates,
    list_registered_handlers,
    make_candidate_id,
    normalize_handler_result,
    priority_counts,
    rank_candidates,
    score_candidate,
)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _strength_issue(**overrides):
    base = dict(
        issue_id="iss_strength_exceeded_m1_dl",
        issue_type=IssueType.STRENGTH_EXCEEDED,
        severity=Severity.ERROR,
        source=IssueSource.DESIGN_CHECK,
        description="…",
        member_id=1,
        member_type="column",
        section="H-300x300",
        demand_capacity_ratio=1.3,
        evidence={"section": "H-300x300", "type": "column"},
    )
    base.update(overrides)
    return StructuralIssue(**base)


def _drift_issue(**overrides):
    base = dict(
        issue_id="iss_drift_exceeded_s3_x",
        issue_type=IssueType.DRIFT_EXCEEDED,
        severity=Severity.ERROR,
        source=IssueSource.DESIGN_CHECK,
        description="…",
        story=3,
        demand_capacity_ratio=1.2,
        evidence={"story": 3, "direction": "X"},
    )
    base.update(overrides)
    return StructuralIssue(**base)


def _missing_dc_issue():
    return StructuralIssue(
        issue_id="iss_missing_design_check_global",
        issue_type=IssueType.MISSING_DESIGN_CHECK,
        severity=Severity.WARNING,
        source=IssueSource.ANALYSIS,
        description="missing",
    )


def _make_design_check(*, members=None, drift_checks=None, drift_allow=0.020):
    return {
        "overall_status": "NG",
        "drift_check": ({
            "status": "NG" if drift_checks else "OK",
            "Cd": 3.0, "IE": 1.0, "allowable": drift_allow,
            "checks": drift_checks or [],
        } if drift_checks is not None else None),
        "member_check": ({
            "status": "NG",
            "members": members or [],
            "summary": {"total": len(members or []), "ok": 0, "ng": len(members or [])},
        } if members is not None else None),
        "summary": {},
    }


def _mk_member(**overrides):
    base = {
        "member_id": 1, "type": "column", "section": "H-300x300",
        "governing_combo": "1.2DL+1.0LL",
        "ratios": {"interaction": 1.35, "shear": 0.4},
        "demand": {"Pu": 200}, "capacity": {"phiPn_kN": 800},
        "status": "NG", "story": 2,
    }
    base.update(overrides)
    return base


# ============================================================
# Taxonomy
# ============================================================

class TestTaxonomy:
    def test_strength_classified_as_strength(self):
        c = classify_issue(_strength_issue())
        assert c.category == IssueCategory.STRENGTH
        assert c.priority == PRIORITY_HIGH

    def test_critical_dcr_bumps_priority(self):
        c = classify_issue(_strength_issue(demand_capacity_ratio=1.6))
        assert c.priority == PRIORITY_CRITICAL

    def test_drift_classified_as_serviceability(self):
        c = classify_issue(_drift_issue())
        assert c.category == IssueCategory.SERVICEABILITY
        # Error-severity drift below 1.5 → MEDIUM
        assert c.priority == PRIORITY_MEDIUM

    def test_high_drift_critical_drift_promotes_to_high(self):
        c = classify_issue(_drift_issue(demand_capacity_ratio=1.6))
        assert c.priority == PRIORITY_HIGH

    def test_missing_design_check_is_data_quality_low(self):
        c = classify_issue(_missing_dc_issue())
        assert c.category == IssueCategory.DATA_QUALITY
        assert c.priority == PRIORITY_LOW

    def test_category_and_priority_counts(self):
        issues = [
            _strength_issue(member_id=1),
            _strength_issue(member_id=2, demand_capacity_ratio=1.6),
            _drift_issue(),
            _missing_dc_issue(),
        ]
        cats = category_counts(issues)
        assert cats[IssueCategory.STRENGTH] == 2
        assert cats[IssueCategory.SERVICEABILITY] == 1
        assert cats[IssueCategory.DATA_QUALITY] == 1

        prios = priority_counts(issues)
        # One CRITICAL strength, one HIGH strength, one MEDIUM drift,
        # one LOW missing-dc.
        assert prios[str(PRIORITY_CRITICAL)] == 1
        assert prios[str(PRIORITY_HIGH)] == 1
        assert prios[str(PRIORITY_MEDIUM)] == 1
        assert prios[str(PRIORITY_LOW)] == 1

    def test_unknown_issue_type_classifies_as_unclassified(self):
        """An issue_type the recommender hasn't seen lands in UNCLASSIFIED."""
        iss = StructuralIssue(
            issue_id="iss_unknown",
            issue_type="brand_new_issue_type",
            severity=Severity.WARNING,
            source=IssueSource.ANALYSIS,
            description="…",
        )
        c = classify_issue(iss)
        # Crucially, unknown types are NOT silently bucketed as data_quality.
        assert c.category == IssueCategory.UNCLASSIFIED
        assert c.category != IssueCategory.DATA_QUALITY

    def test_data_quality_only_for_missing_dc_and_warning(self):
        """DATA_QUALITY must be reserved for the two known data-related types."""
        # Both known data-quality types map to DATA_QUALITY
        assert classify_issue(_missing_dc_issue()).category == IssueCategory.DATA_QUALITY
        warn_iss = StructuralIssue(
            issue_id="iss_w",
            issue_type=IssueType.ANALYSIS_WARNING,
            severity=Severity.WARNING,
            source=IssueSource.WARNING,
            description="…",
        )
        assert classify_issue(warn_iss).category == IssueCategory.DATA_QUALITY

    def test_unclassified_error_severity_routes_to_high(self):
        """An error-severity unknown issue should still surface to engineer."""
        iss = StructuralIssue(
            issue_id="iss_unknown_err",
            issue_type="some_new_type",
            severity=Severity.ERROR,
            source=IssueSource.ANALYSIS,
            description="…",
        )
        assert classify_issue(iss).priority == PRIORITY_HIGH


# ============================================================
# Rule registry
# ============================================================

class TestRuleRegistry:
    def test_default_registry_has_known_handlers(self):
        labels = list_registered_handlers()
        # Defaults populated by candidate_generator at import time.
        for t in (IssueType.STRENGTH_EXCEEDED, IssueType.SHEAR_EXCEEDED,
                  IssueType.DRIFT_EXCEEDED, IssueType.MISSING_DESIGN_CHECK,
                  IssueType.ANALYSIS_WARNING):
            assert t in labels

    def test_registry_returns_none_for_unknown_type(self):
        reg = default_registry()
        assert reg.get("totally_made_up_issue_type") is None

    def test_isolated_registry_does_not_pollute_default(self):
        """RuleRegistry instances are independent — sanity check."""
        local = RuleRegistry()
        assert local.get(IssueType.STRENGTH_EXCEEDED) is None
        # Default registry still has it.
        assert default_registry().has(IssueType.STRENGTH_EXCEEDED)

    def test_generate_candidates_uses_provided_registry(self):
        """``registry=`` overrides the default — no global mutation."""
        empty = RuleRegistry()
        iss = _strength_issue()

        # Default registry → INCREASE_SECTION (typed candidate).
        default_cands = generate_candidates([iss])
        assert default_cands[0].action_type == ActionType.INCREASE_SECTION

        # Custom registry with no rule for STRENGTH_EXCEEDED → fall back
        # to engineer review. Crucially, the default registry must NOT
        # be affected by passing this in.
        empty_cands = generate_candidates([iss], registry=empty)
        assert empty_cands[0].action_type == ActionType.REQUIRES_ENGINEER_REVIEW
        # Default registry untouched.
        assert default_registry().has(IssueType.STRENGTH_EXCEEDED)

    def test_build_payload_accepts_custom_registry(self):
        """``build_recommendation_payload(registry=...)`` forwards the registry."""
        empty = RuleRegistry()
        dc = _make_design_check(members=[_mk_member()])
        payload = build_recommendation_payload(design_check=dc, registry=empty)
        actions = {c["action_type"] for c in payload["recommendation_candidates"]}
        # With no rules registered, every issue falls back to engineer review.
        assert actions == {ActionType.REQUIRES_ENGINEER_REVIEW}
        # Default registry untouched.
        assert default_registry().has(IssueType.STRENGTH_EXCEEDED)

    def test_handler_can_still_return_single_candidate(self):
        """Backward-compat: a handler may return one RetrofitCandidate."""
        def one_option(issue):
            return RetrofitCandidate(
                candidate_id=make_candidate_id(
                    issue_id=issue.issue_id,
                    action_type=ActionType.INCREASE_SECTION,
                ),
                issue_id=issue.issue_id,
                action_type=ActionType.INCREASE_SECTION,
                description="single option",
                member_id=issue.member_id,
            )

        reg = RuleRegistry()
        reg.register(IssueType.STRENGTH_EXCEEDED, one_option)
        cands = generate_candidates([_strength_issue()], registry=reg)
        assert len(cands) == 1
        assert cands[0].action_type == ActionType.INCREASE_SECTION

    def test_handler_may_return_multiple_candidates(self):
        """One handler may yield multiple competing alternatives."""
        def two_options(issue):
            c1 = RetrofitCandidate(
                candidate_id=make_candidate_id(
                    issue_id=issue.issue_id,
                    action_type=ActionType.INCREASE_SECTION,
                ),
                issue_id=issue.issue_id,
                action_type=ActionType.INCREASE_SECTION,
                description="alt: increase section",
                member_id=issue.member_id,
            )
            c2 = RetrofitCandidate(
                candidate_id=make_candidate_id(
                    issue_id=issue.issue_id,
                    action_type=ActionType.CHANGE_MATERIAL,
                ),
                issue_id=issue.issue_id,
                action_type=ActionType.CHANGE_MATERIAL,
                description="alt: change material",
                member_id=issue.member_id,
            )
            return [c1, c2]

        reg = RuleRegistry()
        reg.register(IssueType.STRENGTH_EXCEEDED, two_options)
        iss = _strength_issue()
        cands = generate_candidates([iss], registry=reg)
        # Both alternatives present.
        assert len(cands) == 2
        assert {c.action_type for c in cands} == {
            ActionType.INCREASE_SECTION, ActionType.CHANGE_MATERIAL,
        }
        # Ranking must handle multi-candidate output deterministically.
        ranked = rank_candidates(list(cands), [iss])
        ranked_again = rank_candidates(
            generate_candidates([iss], registry=reg), [iss],
        )
        assert [c.candidate_id for c in ranked] == \
               [c.candidate_id for c in ranked_again]

    def test_handler_can_return_none_to_force_review_fallback(self):
        """Returning None from a handler should fall back to engineer review."""
        def opt_out(issue):
            return None

        reg = RuleRegistry()
        reg.register(IssueType.STRENGTH_EXCEEDED, opt_out)
        cands = generate_candidates([_strength_issue()], registry=reg)
        assert len(cands) == 1
        assert cands[0].action_type == ActionType.REQUIRES_ENGINEER_REVIEW

    def test_normalize_handler_result_helper(self):
        """The dispatcher helper coerces all three return shapes uniformly."""
        cand = RetrofitCandidate(
            candidate_id="cand_x", issue_id="iss_x",
            action_type=ActionType.INCREASE_SECTION, description="…",
        )
        assert normalize_handler_result(None) == []
        assert normalize_handler_result(cand) == [cand]
        assert normalize_handler_result([cand, cand]) == [cand, cand]
        # Generator
        def gen():
            yield cand
            yield cand
        assert len(normalize_handler_result(gen())) == 2


# ============================================================
# Scoring
# ============================================================

class TestScoring:
    def test_score_is_deterministic(self):
        cands = generate_candidates([_strength_issue()])
        a = score_candidate(cands[0], _strength_issue())
        b = score_candidate(cands[0], _strength_issue())
        assert a.to_dict() == b.to_dict()

    def test_higher_overshoot_yields_higher_safety_gain(self):
        iss_low = _strength_issue(demand_capacity_ratio=1.1)
        iss_high = _strength_issue(demand_capacity_ratio=1.9)
        cand_low = generate_candidates([iss_low])[0]
        cand_high = generate_candidates([iss_high])[0]
        s_low = score_candidate(cand_low, iss_low)
        s_high = score_candidate(cand_high, iss_high)
        assert s_high.safety_gain > s_low.safety_gain
        assert s_high.total > s_low.total

    def test_engineer_review_scores_below_typed_candidate(self):
        typed_iss = _strength_issue()
        review_iss = _missing_dc_issue()
        typed_cand = generate_candidates([typed_iss])[0]
        review_cand = generate_candidates([review_iss])[0]
        assert score_candidate(typed_cand, typed_iss).total > \
               score_candidate(review_cand, review_iss).total

    def test_score_axes_all_in_unit_interval(self):
        cand = generate_candidates([_strength_issue()])[0]
        s = score_candidate(cand, _strength_issue())
        for axis in ("safety_gain", "code_compliance", "relative_cost",
                     "disruption", "side_effect_risk", "total"):
            v = getattr(s, axis)
            assert 0.0 <= v <= 1.0, f"{axis}={v}"

    def test_score_serializes_round_trip_json(self):
        cand = generate_candidates([_strength_issue()])[0]
        s = score_candidate(cand, _strength_issue())
        d = s.to_dict()
        assert json.loads(json.dumps(d)) == d

    def test_weights_sum_to_one(self):
        assert abs(sum(SCORING_WEIGHTS.values()) - 1.0) < 1e-9

    def test_score_without_issue_context_does_not_raise(self):
        """Orphan candidate (issue not in the index) → default safety_gain."""
        cand = generate_candidates([_strength_issue()])[0]
        s = score_candidate(cand, None)
        assert isinstance(s, ScoreBreakdown)
        assert 0.0 < s.total <= 1.0

    def test_score_breakdown_advertises_unverified_status(self):
        """ScoreBreakdown carries the honesty markers — not a verified result."""
        cand = generate_candidates([_strength_issue()])[0]
        s = score_candidate(cand, _strength_issue())
        d = s.to_dict()
        assert d["method"] == SCORE_METHOD
        assert d["verified"] is False
        assert d["status"] == "unverified_heuristic"
        # Sanity: the constant itself reads "v1" — bump it whenever
        # weights/axes change in a way that re-orders candidates.
        assert SCORE_METHOD == "heuristic_v1"
        assert SCORE_VERIFIED is False


# ============================================================
# Ranking
# ============================================================

class TestRanking:
    def test_rank_attaches_score_to_metadata(self):
        iss = _strength_issue()
        cands = generate_candidates([iss])
        ranked = rank_candidates(cands, [iss])
        assert "score" in ranked[0].metadata
        # safety_gain key exists in the score breakdown
        assert "safety_gain" in ranked[0].metadata["score"]

    def test_rank_stamps_unverified_marker_on_metadata(self):
        """rank_candidates exposes the honesty markers at the top level too."""
        iss = _strength_issue()
        ranked = rank_candidates(generate_candidates([iss]), [iss])
        md = ranked[0].metadata
        assert md["score_method"] == SCORE_METHOD
        assert md["score_status"] == "unverified_heuristic"
        assert md["score_verified"] is False

    def test_rank_orders_critical_priority_first(self):
        """CRITICAL-priority candidate ranks before HIGH-priority one."""
        iss_high = _strength_issue(
            issue_id="iss_high_m1", member_id=1, demand_capacity_ratio=1.05,
        )
        iss_crit = _strength_issue(
            issue_id="iss_crit_m2", member_id=2, demand_capacity_ratio=1.9,
        )
        cands = generate_candidates([iss_high, iss_crit])
        ranked = rank_candidates(cands, [iss_high, iss_crit])
        # CRITICAL (dcr=1.9, dcr ≥ 1.5) must be first.
        assert ranked[0].issue_id == "iss_crit_m2"

    def test_within_same_priority_total_decides(self):
        """Inside the same priority bucket, higher total ranks first."""
        # Both are HIGH priority (dcr < 1.5, severity=ERROR).
        iss_a = _strength_issue(
            issue_id="iss_a_m1", member_id=1, demand_capacity_ratio=1.05,
        )
        iss_b = _strength_issue(
            issue_id="iss_b_m2", member_id=2, demand_capacity_ratio=1.4,
        )
        cands = generate_candidates([iss_a, iss_b])
        ranked = rank_candidates(cands, [iss_a, iss_b])
        # Higher D/C → higher safety_gain → higher total.
        assert ranked[0].issue_id == "iss_b_m2"

    def test_priority_dominates_over_heuristic_total(self):
        """Regression: priority must NOT be overridden by a higher
        heuristic total from a cheap-action candidate.

        Scenario:
            * Issue A — CRITICAL (D/C=1.6) but member_id unknown →
              ``_should_block_auto_candidate`` blocks it → engineer
              review candidate (low compliance/cost baseline).
            * Issue B — HIGH (D/C=1.1) with full info → typed
              ``INCREASE_SECTION`` candidate (high baselines).

        Heuristic-total alone would push B in front (its action
        baselines are stronger). Priority-first ordering puts A
        first because it's the more urgent safety finding.
        """
        critical_iss = StructuralIssue(
            issue_id="iss_crit_no_member",
            issue_type=IssueType.STRENGTH_EXCEEDED,
            severity=Severity.ERROR,
            source=IssueSource.DESIGN_CHECK,
            description="…",
            member_id=None,            # → blocked → engineer review
            demand_capacity_ratio=1.6,  # CRITICAL priority
        )
        high_iss = _strength_issue(
            issue_id="iss_high_m9", member_id=9,
            demand_capacity_ratio=1.1,  # HIGH priority
        )

        cands = generate_candidates([critical_iss, high_iss])
        by_id = {c.issue_id: c for c in cands}
        assert by_id["iss_crit_no_member"].action_type == ActionType.REQUIRES_ENGINEER_REVIEW
        assert by_id["iss_high_m9"].action_type == ActionType.INCREASE_SECTION

        # Sanity: critical's total is in fact LOWER than high's
        # (typed action vs review fallback).
        crit_total = score_candidate(by_id["iss_crit_no_member"], critical_iss).total
        high_total = score_candidate(by_id["iss_high_m9"], high_iss).total
        assert crit_total < high_total, (
            "Test premise broken — expected critical(review) < high(typed) "
            f"but got {crit_total} vs {high_total}."
        )

        # Yet ranking puts the CRITICAL safety finding first.
        ranked = rank_candidates(cands, [critical_iss, high_iss])
        assert ranked[0].issue_id == "iss_crit_no_member"
        assert ranked[1].issue_id == "iss_high_m9"

    def test_rank_is_stable_for_same_input(self):
        issues = [
            _strength_issue(issue_id=f"iss_m{i}", member_id=i,
                            demand_capacity_ratio=1.2)
            for i in range(1, 5)
        ]
        cands = generate_candidates(issues)
        a = [c.candidate_id for c in rank_candidates(list(cands), issues)]
        # Regenerate to avoid mutated metadata polluting equality.
        cands2 = generate_candidates(issues)
        b = [c.candidate_id for c in rank_candidates(list(cands2), issues)]
        assert a == b

    def test_rank_attaches_issue_classification(self):
        iss = _strength_issue()
        cands = generate_candidates([iss])
        ranked = rank_candidates(cands, [iss])
        cls = ranked[0].metadata.get("issue_classification")
        assert cls is not None
        assert cls["category"] == IssueCategory.STRENGTH


# ============================================================
# Pipeline integration
# ============================================================

class TestPipelineIntegration:
    def test_payload_exposes_category_and_priority_counts(self):
        dc = _make_design_check(members=[_mk_member()])
        payload = build_recommendation_payload(
            design_check=dc, raw_warnings=None,
            stage="t", mode=MODE_ANALYZE,
        )
        s = payload["recommendation_summary"]
        assert "issues_by_category" in s
        assert "issues_by_priority" in s
        assert s["issues_by_category"].get(IssueCategory.STRENGTH, 0) >= 1
        assert s["ranked"] is True

    def test_payload_candidate_carries_score_in_metadata(self):
        dc = _make_design_check(members=[_mk_member()])
        payload = build_recommendation_payload(design_check=dc)
        cand = payload["recommendation_candidates"][0]
        assert "score" in cand["metadata"]
        assert 0.0 <= cand["metadata"]["score"]["total"] <= 1.0
        # Honesty markers must be present at the payload level.
        assert cand["metadata"]["score_method"] == SCORE_METHOD
        assert cand["metadata"]["score_status"] == "unverified_heuristic"
        assert cand["metadata"]["score_verified"] is False

    def test_summary_advertises_ranking_method_and_reanalysis_flag(self):
        """Summary must expose ranking provenance + need-for-reanalysis flag."""
        dc = _make_design_check(members=[_mk_member()])
        payload = build_recommendation_payload(design_check=dc)
        s = payload["recommendation_summary"]
        assert s["ranking_method"] == SCORE_METHOD
        assert s["ranking_verified"] is False
        assert s["requires_reanalysis_for_final_ranking"] is True

    def test_summary_ranking_keys_are_none_when_no_candidates(self):
        """Empty payload — ranking method should be None, not the score id."""
        payload = build_recommendation_payload(
            design_check=_make_design_check(members=[]),
        )
        s = payload["recommendation_summary"]
        assert s["ranked"] is False
        assert s["ranking_method"] is None
        assert s["ranking_verified"] is None
        # Reanalysis flag is always True — it's about the engine's
        # nature, not whether this particular payload had candidates.
        assert s["requires_reanalysis_for_final_ranking"] is True

    def test_rank_false_skips_ordering(self):
        # Build two issues such that low-D/C-then-high order would NOT be
        # rank order. Verify rank=False preserves generation order.
        iss_low = _strength_issue(
            issue_id="iss_a_m1", member_id=1, demand_capacity_ratio=1.05,
        )
        iss_high = _strength_issue(
            issue_id="iss_b_m2", member_id=2, demand_capacity_ratio=1.9,
        )
        # Run through the pipeline by hand to keep generation order.
        dc = _make_design_check(members=[
            _mk_member(member_id=1, ratios={"interaction": 1.05, "shear": 0.0}),
            _mk_member(member_id=2, ratios={"interaction": 1.9, "shear": 0.0}),
        ])
        # rank=True (default) → high overshoot first
        ranked_payload = build_recommendation_payload(design_check=dc, rank=True)
        ranked_first = ranked_payload["recommendation_candidates"][0]
        assert ranked_first["member_id"] == 2

        # rank=False → generation order (member_id 1 first)
        unranked = build_recommendation_payload(design_check=dc, rank=False)
        assert unranked["recommendation_summary"]["ranked"] is False
        assert unranked["recommendation_candidates"][0]["member_id"] == 1
        # No score attached when rank=False
        assert "score" not in (unranked["recommendation_candidates"][0]["metadata"] or {})

    def test_no_auto_kds_rag_attachment(self):
        """Deterministic pipeline must never attach KDS-RAG quotes/urls."""
        dc = _make_design_check(members=[_mk_member()])
        payload = build_recommendation_payload(design_check=dc)
        # Summary flag advertises RAG off
        assert payload["recommendation_summary"]["rag_enabled"] is False
        assert payload["recommendation_summary"]["llm_enabled"] is False
        # No kds_rag_summary key (only added by enrich adapter)
        assert "kds_rag_summary" not in payload
        # Code refs on issues carry hints only — no quote / source_url
        for iss in payload["issues"]:
            for ref in iss.get("code_refs", []):
                assert "quote" not in ref
                assert "source_url" not in ref


# ============================================================
# Graceful behavior on partial data
# ============================================================

class TestGracefulPartialData:
    def test_strength_without_section_falls_back_to_review_with_score(self):
        iss = _strength_issue(section=None, evidence={"type": "column"})
        cands = generate_candidates([iss])
        assert cands[0].action_type == ActionType.REQUIRES_ENGINEER_REVIEW
        ranked = rank_candidates(cands, [iss])
        # Even fallback candidates get a deterministic score.
        assert "score" in ranked[0].metadata
        assert ranked[0].metadata["score"]["total"] > 0.0

    def test_drift_without_story_falls_back_with_score(self):
        iss = _drift_issue(story=None, evidence={})
        cands = generate_candidates([iss])
        assert cands[0].action_type == ActionType.REQUIRES_ENGINEER_REVIEW
        ranked = rank_candidates(cands, [iss])
        assert ranked[0].metadata["score"]["total"] > 0.0

    def test_empty_issue_list_yields_empty_payload(self):
        payload = build_recommendation_payload(
            design_check=_make_design_check(members=[]), raw_warnings=None,
        )
        assert payload["recommendation_candidates"] == []
        assert payload["recommendation_summary"]["num_candidates"] == 0
        # ranked is False when there's nothing to rank
        assert payload["recommendation_summary"]["ranked"] is False

    def test_warning_only_issue_yields_no_candidate_no_crash(self):
        payload = build_recommendation_payload(
            design_check=_make_design_check(members=[]),
            raw_warnings=["legacy: just a warn"],
        )
        # warning was lifted to a low-severity ANALYSIS_WARNING issue,
        # and no candidate emitted for non-error severity.
        types = [i["issue_type"] for i in payload["issues"]]
        assert IssueType.ANALYSIS_WARNING in types
        assert payload["recommendation_candidates"] == []


# ============================================================
# Unverified-heuristic disclosure (Issue 2)
# ============================================================

class TestUnverifiedHeuristicDisclosure:
    def test_score_breakdown_carries_method_and_unverified_flag(self):
        iss = _strength_issue()
        cand = generate_candidates([iss])[0]
        s = score_candidate(cand, iss)
        assert s.method == SCORE_METHOD == "heuristic_v1"
        assert s.verified is SCORE_VERIFIED is False
        assert s.status == "unverified_heuristic"

    def test_candidate_metadata_score_marks_unverified(self):
        iss = _strength_issue()
        cands = rank_candidates(generate_candidates([iss]), [iss])
        meta_score = cands[0].metadata["score"]
        assert meta_score["method"] == "heuristic_v1"
        assert meta_score["verified"] is False
        assert meta_score["status"] == "unverified_heuristic"

    def test_summary_exposes_ranking_method_and_reanalysis_flag(self):
        dc = _make_design_check(members=[_mk_member()])
        payload = build_recommendation_payload(design_check=dc)
        s = payload["recommendation_summary"]
        assert s["ranking_method"] == "heuristic_v1"
        assert s["ranking_verified"] is False
        assert s["requires_reanalysis_for_final_ranking"] is True

    def test_unranked_summary_method_is_null_but_flag_still_true(self):
        """rank=False → no ranking_method, but the re-analysis flag stays.

        Even with no ranking, the caller still must re-analyze before
        treating any candidate as a final design decision.
        """
        dc = _make_design_check(members=[_mk_member()])
        payload = build_recommendation_payload(design_check=dc, rank=False)
        s = payload["recommendation_summary"]
        assert s["ranking_method"] is None
        assert s["ranked"] is False
        assert s["requires_reanalysis_for_final_ranking"] is True


# ============================================================
# Priority-dominant ranking regression (Issue 3)
# ============================================================

class TestPriorityDominantRanking:
    def test_critical_strength_outranks_medium_drift_with_cheaper_action(self):
        """A CRITICAL strength issue must not be ranked below a MEDIUM
        drift issue just because the drift candidate has a cheaper-looking
        heuristic score.

        Construction:
            * CRITICAL strength issue (D/C=1.6) → INCREASE_SECTION
              (priority=CRITICAL via D/C ≥ 1.5)
            * MEDIUM drift issue (D/C=1.05) → ADD_LATERAL_RESISTANCE
              (priority=MEDIUM via error-severity drift below 1.5)
        """
        critical_strength = _strength_issue(
            issue_id="iss_strength_crit", member_id=10,
            demand_capacity_ratio=1.6,
        )
        medium_drift = _drift_issue(
            issue_id="iss_drift_medium", story=3,
            demand_capacity_ratio=1.05,
        )

        # Sanity-check classification matches expectation.
        assert classify_issue(critical_strength).priority == PRIORITY_CRITICAL
        assert classify_issue(medium_drift).priority == PRIORITY_MEDIUM

        cands = generate_candidates([medium_drift, critical_strength])
        ranked = rank_candidates(cands, [medium_drift, critical_strength])

        # Critical strength must be first regardless of action-baseline
        # cost differences. This is the safety invariant of the engine.
        assert ranked[0].issue_id == "iss_strength_crit"
        assert ranked[-1].issue_id == "iss_drift_medium"

    def test_priority_dominance_over_total_within_other_priorities(self):
        """A HIGH-priority candidate must outrank a LOW-priority one even
        if the latter has a higher heuristic total."""
        high = _strength_issue(
            issue_id="iss_high_strength", member_id=20,
            demand_capacity_ratio=1.2,
        )
        low = _missing_dc_issue()
        # MISSING_DC has priority LOW; strength error D/C<1.5 has HIGH.
        assert classify_issue(high).priority == PRIORITY_HIGH
        assert classify_issue(low).priority == PRIORITY_LOW

        cands = generate_candidates([low, high])
        ranked = rank_candidates(cands, [low, high])
        assert ranked[0].issue_id == "iss_high_strength"

    def test_within_same_priority_higher_total_wins(self):
        """Tie-break by score total only within the same priority bucket."""
        iss_a = _strength_issue(
            issue_id="iss_a", member_id=1, demand_capacity_ratio=1.6,
        )  # CRITICAL
        iss_b = _strength_issue(
            issue_id="iss_b", member_id=2, demand_capacity_ratio=1.9,
        )  # CRITICAL, higher overshoot
        assert classify_issue(iss_a).priority == PRIORITY_CRITICAL
        assert classify_issue(iss_b).priority == PRIORITY_CRITICAL

        ranked = rank_candidates(
            generate_candidates([iss_a, iss_b]), [iss_a, iss_b],
        )
        # Higher overshoot first inside the same priority bucket.
        assert ranked[0].issue_id == "iss_b"


# ============================================================
# UNCLASSIFIED category (Issue 4)
# ============================================================

class TestUnclassifiedCategory:
    def test_unknown_issue_type_classifies_as_unclassified(self):
        iss = StructuralIssue(
            issue_id="iss_weird_1",
            issue_type="totally_made_up_issue_type",
            severity=Severity.ERROR,
            source=IssueSource.ANALYSIS,
            description="…",
        )
        cls = classify_issue(iss)
        assert cls.category == IssueCategory.UNCLASSIFIED
        # Not silently bucketed as DATA_QUALITY.
        assert cls.category != IssueCategory.DATA_QUALITY

    def test_unknown_issue_type_falls_back_to_engineer_review(self):
        iss = StructuralIssue(
            issue_id="iss_weird_2",
            issue_type="totally_made_up_issue_type",
            severity=Severity.ERROR,
            source=IssueSource.ANALYSIS,
            description="…",
        )
        cands = generate_candidates([iss])
        assert len(cands) == 1
        assert cands[0].action_type == ActionType.REQUIRES_ENGINEER_REVIEW

    def test_data_quality_is_only_missing_dc_and_warnings(self):
        """DATA_QUALITY must NOT absorb arbitrary unknown issue_types."""
        unknown = StructuralIssue(
            issue_id="iss_xx", issue_type="something_else",
            severity=Severity.WARNING, source=IssueSource.WARNING,
            description="…",
        )
        missing_dc = _missing_dc_issue()
        warning_issue = StructuralIssue(
            issue_id="iss_warn", issue_type=IssueType.ANALYSIS_WARNING,
            severity=Severity.WARNING, source=IssueSource.WARNING,
            description="…",
        )
        assert classify_issue(unknown).category == IssueCategory.UNCLASSIFIED
        assert classify_issue(missing_dc).category == IssueCategory.DATA_QUALITY
        assert classify_issue(warning_issue).category == IssueCategory.DATA_QUALITY

    def test_category_counts_includes_unclassified_bucket(self):
        issues = [
            _strength_issue(),
            StructuralIssue(
                issue_id="iss_z", issue_type="brand_new_type",
                severity=Severity.WARNING, source=IssueSource.ANALYSIS,
                description="…",
            ),
        ]
        cats = category_counts(issues)
        assert cats.get(IssueCategory.UNCLASSIFIED) == 1
        assert cats.get(IssueCategory.STRENGTH) == 1
