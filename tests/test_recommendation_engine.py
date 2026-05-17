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
    IssueCategory,
    IssueSource,
    IssueType,
    MODE_ANALYZE,
    PRIORITY_CRITICAL,
    PRIORITY_HIGH,
    PRIORITY_LOW,
    PRIORITY_MEDIUM,
    RuleRegistry,
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
    priority_counts,
    rank_candidates,
    register_rule,
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

    def test_register_rule_overrides_existing(self):
        """register_rule must overwrite an existing handler — last one wins."""
        sentinel_called = []

        def fake_handler(issue):
            sentinel_called.append(issue.issue_id)
            return None  # signals "no candidate"

        reg = default_registry()
        original = reg.get(IssueType.STRENGTH_EXCEEDED)
        try:
            register_rule(IssueType.STRENGTH_EXCEEDED, fake_handler, label="test_override")
            cands = generate_candidates([_strength_issue()])
            # Handler returned None → engineer-review fallback emitted.
            assert sentinel_called == ["iss_strength_exceeded_m1_dl"]
            assert len(cands) == 1
            assert cands[0].action_type == ActionType.REQUIRES_ENGINEER_REVIEW
        finally:
            # Restore original handler so other tests aren't affected.
            assert original is not None
            register_rule(
                original.issue_type, original.handler,
                label=original.label, priority=original.priority,
            )

    def test_isolated_registry_does_not_pollute_default(self):
        """RuleRegistry instances are independent — sanity check."""
        local = RuleRegistry()
        assert local.get(IssueType.STRENGTH_EXCEEDED) is None
        # Default registry still has it.
        assert default_registry().has(IssueType.STRENGTH_EXCEEDED)


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

    def test_rank_orders_by_total_descending(self):
        iss_low = _strength_issue(
            issue_id="iss_low_m1", member_id=1, demand_capacity_ratio=1.05,
        )
        iss_high = _strength_issue(
            issue_id="iss_high_m2", member_id=2, demand_capacity_ratio=1.9,
        )
        cands = generate_candidates([iss_low, iss_high])
        ranked = rank_candidates(cands, [iss_low, iss_high])
        # The high-overshoot candidate must be first.
        assert ranked[0].issue_id == "iss_high_m2"

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
