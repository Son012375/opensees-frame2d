"""Unit tests for ``core.recommendation.evaluator``.

The full reanalysis loop is exercised end-to-end by the API smoke
test (`test_v2_recommendations_api`). Here we cover only the parts
that DON'T require running OpenSees:

    * ``BaselineSummary.from_design_check`` parsing
    * Inapplicable candidates short-circuit before any analysis
    * Verified score shape (method/verified/status fields)
    * New-NG detection (driven by monkey-patching the analyzer)
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mcp-server"))

from core.recommendation import (  # noqa: E402
    ActionType,
    ChangeOperation,
    Confidence,
    IssueSource,
    IssueType,
    RetrofitCandidate,
    Severity,
    StructuralIssue,
    BaselineSummary,
    CandidateEvaluation,
    STATUS_EVALUATED,
    STATUS_REJECTED_FAILED,
    STATUS_REJECTED_NEW_NG,
    STATUS_SKIPPED_INAPPLICABLE,
    VERIFIED_SCORE_METHOD,
    evaluate_candidate,
    generate_candidates,
    make_issue_id,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_model() -> dict:
    """Same fixture shape as test_apply_candidate.py."""
    nodes = []
    nid = 1
    for s in range(3):
        for ix in range(2):
            for iy in range(2):
                nodes.append({
                    "id": nid, "x": ix * 6000, "y": iy * 6000, "z": s * 3500,
                    "story": s,
                    "support": "fixed" if s == 0 else None, "mass": 0.0,
                })
                nid += 1
    elements = []
    eid = 1
    for s in range(1, 3):
        for ix in range(2):
            for iy in range(2):
                lower = (s - 1) * 4 + ix * 2 + iy + 1
                upper = s * 4 + ix * 2 + iy + 1
                elements.append({
                    "id": eid, "node_i": lower, "node_j": upper,
                    "elem_type": "column",
                    "section": "H-300x300", "material": "SS275",
                    "release_i": None, "release_j": None, "beta_angle": 0.0,
                })
                eid += 1
    return {"version": "2.0", "nodes": nodes, "elements": elements,
            "story_elevations": [0, 3500, 7000]}


def _strength_issue() -> StructuralIssue:
    return StructuralIssue(
        issue_id=make_issue_id(
            issue_type=IssueType.STRENGTH_EXCEEDED, member_id=1,
            governing_combo="X",
        ),
        issue_type=IssueType.STRENGTH_EXCEEDED,
        severity=Severity.ERROR, source=IssueSource.DESIGN_CHECK,
        description="ng", member_id=1, element_id=1,
        governing_combo="X", demand_capacity_ratio=1.4,
        member_type="column", section="H-300x300", story=1,
    )


def _drift_issue() -> StructuralIssue:
    return StructuralIssue(
        issue_id=make_issue_id(
            issue_type=IssueType.DRIFT_EXCEEDED, story=2, direction="X",
            governing_combo="EQX",
        ),
        issue_type=IssueType.DRIFT_EXCEEDED,
        severity=Severity.ERROR, source=IssueSource.DESIGN_CHECK,
        description="drift", demand_capacity_ratio=1.20,
        story=2, evidence={"direction": "X", "story": 2},
    )


# ---------------------------------------------------------------------------
# BaselineSummary parsing
# ---------------------------------------------------------------------------

class TestBaselineSummary:
    def test_from_design_check_extracts_ng_sets(self):
        dc = {
            "summary": {"max_drift_ratio": 0.012, "max_interaction_ratio": 1.35,
                        "ng_members": 1, "ng_stories": 1},
            "drift_check": {
                "status": "NG",
                "checks": [
                    {"story": 2, "direction": "X", "status": "NG",
                     "ratio": 1.2, "combo": "EQX"},
                    {"story": 1, "direction": "X", "status": "OK",
                     "ratio": 0.5, "combo": "EQX"},
                ],
            },
            "member_check": {
                "status": "NG",
                "members": [
                    {"member_id": 1, "status": "NG"},
                    {"member_id": 2, "status": "OK"},
                    {"member_id": 3, "status": "OK"},
                ],
                "summary": {"max_interaction_ratio": 1.35, "ng": 1},
            },
        }
        b = BaselineSummary.from_design_check(dc)
        assert b.max_interaction_ratio == 1.35
        assert b.max_drift_ratio == 0.012
        assert b.ng_member_ids == {1}
        assert b.ok_member_ids == {2, 3}
        assert b.ng_drift_keys == {(2, "X")}
        assert b.total_member_count == 3

    def test_from_design_check_empty(self):
        b = BaselineSummary.from_design_check(None)
        assert b.max_interaction_ratio == 0.0
        assert b.ng_member_ids == set()


# ---------------------------------------------------------------------------
# Inapplicable short-circuit
# ---------------------------------------------------------------------------

class TestInapplicableShortCircuit:
    def test_abstract_candidate_is_skipped(self):
        model = _build_model()
        cands = generate_candidates([_drift_issue()])
        abstract = next(
            c for c in cands
            if c.proposed_change["operation"]
            == ChangeOperation.ADD_LATERAL_RESISTANCE
        )
        result = evaluate_candidate(
            model_json=model,
            load_cases={}, load_combinations={},
            candidate=abstract,
            baseline=BaselineSummary(),
        )
        assert result.status == STATUS_SKIPPED_INAPPLICABLE
        assert result.score is None
        assert result.diff is None
        assert "applicable=False" in (result.error or "")

    def test_unknown_target_is_skipped(self):
        model = _build_model()
        # Build a fake candidate pointing at a missing element.
        cand = RetrofitCandidate(
            candidate_id="cand_missing",
            issue_id="iss_x",
            action_type=ActionType.INCREASE_SECTION,
            description="missing",
            requires_reanalysis=True,
            confidence=Confidence.MEDIUM,
            target={"element_id": 9999},
            proposed_change={
                "operation": ChangeOperation.REPLACE_SECTION,
                "from": "H-300x300", "to": "H-350x350",
                "requires_user_selection": False,
                "applicable": True, "reason": "test",
            },
        )
        result = evaluate_candidate(
            model_json=model,
            load_cases={}, load_combinations={},
            candidate=cand,
            baseline=BaselineSummary(),
        )
        assert result.status == STATUS_SKIPPED_INAPPLICABLE


# ---------------------------------------------------------------------------
# Analysis failure path
# ---------------------------------------------------------------------------

class TestAnalysisFailure:
    def test_analyzer_exception_is_rejected_not_raised(self):
        """An analysis failure must NOT crash the evaluator — it should
        produce a ``rejected_analysis_failed`` evaluation with the
        traceback in ``error``."""
        model = _build_model()
        cands = generate_candidates([_strength_issue()])
        member_cand = next(
            c for c in cands
            if c.proposed_change["operation"] == ChangeOperation.REPLACE_SECTION
        )

        # Patch analyze_from_model to raise inside evaluator.
        with patch("core.frame_3d.analyze_from_model",
                   side_effect=RuntimeError("simulated solver failure")):
            result = evaluate_candidate(
                model_json=model,
                load_cases={}, load_combinations={},
                candidate=member_cand,
                baseline=BaselineSummary(),
            )

        assert result.status == STATUS_REJECTED_FAILED
        assert result.diff is not None
        assert "simulated solver failure" in (result.error or "")
        assert result.score is None


# ---------------------------------------------------------------------------
# Verified-score and new-NG detection via design_check stub
# ---------------------------------------------------------------------------

class _FakeMulti:
    """Minimal stand-in returned by a stubbed analyze_from_model."""
    pass


def _patch_pipeline(dc_after: dict):
    """Patch analyze + run_design_check + StructuralModel.from_json to
    return canned values so the evaluator can run on top of them."""
    return (
        patch("core.frame_3d.analyze_from_model", return_value=_FakeMulti()),
        patch("core.design_check.run_design_check", return_value=dc_after),
        patch("core.structural_model.StructuralModel.from_json",
              return_value=object()),
    )


class TestEvaluatedPath:
    def test_evaluated_emits_verified_score(self):
        model = _build_model()
        cands = generate_candidates([_strength_issue()])
        member_cand = next(
            c for c in cands
            if c.proposed_change["operation"] == ChangeOperation.REPLACE_SECTION
        )

        # baseline: 1 NG member, max D/C 1.4. After change: all OK.
        baseline = BaselineSummary(
            max_interaction_ratio=1.4, max_drift_ratio=0.005,
            ng_member_ids={1},
            ng_drift_keys=set(),
            ok_member_ids={2, 3, 4, 5, 6, 7, 8},
            total_member_count=8,
        )
        dc_after = {
            "overall_status": "OK",
            "summary": {"max_interaction_ratio": 0.9, "max_drift_ratio": 0.005,
                        "ng_members": 0, "ng_stories": 0},
            "drift_check": {"status": "OK", "checks": []},
            "member_check": {
                "status": "OK",
                "members": [{"member_id": i, "status": "OK"}
                            for i in range(1, 9)],
                "summary": {"max_interaction_ratio": 0.9, "ng": 0},
            },
        }

        a, b, c = _patch_pipeline(dc_after)
        with a, b, c:
            result = evaluate_candidate(
                model_json=model,
                load_cases={}, load_combinations={},
                candidate=member_cand,
                baseline=baseline,
            )

        assert result.status == STATUS_EVALUATED, result.error
        assert result.score is not None
        assert result.score.method == VERIFIED_SCORE_METHOD
        assert result.score.verified is True
        assert result.score.status == "verified"
        # D/C dropped: improvement.dcr_delta must be negative.
        assert result.improvement["dcr_delta"] < 0
        assert result.improvement["ng_member_delta"] == -1
        assert result.metrics["analysis_succeeded"] is True
        assert result.metrics["changed_member_count"] == 1


class TestNewNgPath:
    def test_new_ng_is_rejected(self):
        """If a previously-OK member becomes NG after the change, the
        candidate must be hard-rejected as ``rejected_new_ng``."""
        model = _build_model()
        cands = generate_candidates([_strength_issue()])
        member_cand = next(
            c for c in cands
            if c.proposed_change["operation"] == ChangeOperation.REPLACE_SECTION
        )

        # Baseline: only member 1 NG. After change: member 1 OK BUT
        # member 5 (previously OK) is now NG.
        baseline = BaselineSummary(
            max_interaction_ratio=1.4, max_drift_ratio=0.005,
            ng_member_ids={1},
            ng_drift_keys=set(),
            ok_member_ids={2, 3, 4, 5, 6, 7, 8},
            total_member_count=8,
        )
        dc_after = {
            "overall_status": "NG",
            "summary": {"max_interaction_ratio": 1.1, "max_drift_ratio": 0.005,
                        "ng_members": 1, "ng_stories": 0},
            "drift_check": {"status": "OK", "checks": []},
            "member_check": {
                "status": "NG",
                "members": [
                    {"member_id": 1, "status": "OK"},
                    {"member_id": 2, "status": "OK"},
                    {"member_id": 3, "status": "OK"},
                    {"member_id": 4, "status": "OK"},
                    {"member_id": 5, "status": "NG"},   # new NG
                    {"member_id": 6, "status": "OK"},
                    {"member_id": 7, "status": "OK"},
                    {"member_id": 8, "status": "OK"},
                ],
                "summary": {"max_interaction_ratio": 1.1, "ng": 1},
            },
        }

        a, b, c = _patch_pipeline(dc_after)
        with a, b, c:
            result = evaluate_candidate(
                model_json=model,
                load_cases={}, load_combinations={},
                candidate=member_cand,
                baseline=baseline,
            )

        assert result.status == STATUS_REJECTED_NEW_NG, result.error
        assert result.score is None
        assert result.improvement["new_ng_members"] == [5]


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_evaluation_to_dict_round_trips(self):
        import json
        ce = CandidateEvaluation(candidate_id="cand_x", status=STATUS_EVALUATED)
        d = ce.to_dict()
        assert json.loads(json.dumps(d)) == d
        assert d["candidate_id"] == "cand_x"
        assert d["status"] == STATUS_EVALUATED
