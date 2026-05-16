"""Tests for the recommendation pipeline foundation.

Scope: deterministic layer only — issue extraction, candidate generation,
warning normalization. NO RAG/LLM coverage here (none is implemented).

Run from repo root:
    pytest tests/test_recommendation.py -q
"""
from __future__ import annotations

import json
import os
import sys

# mcp-server를 import path에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mcp-server"))

from core.recommendation import (  # noqa: E402
    ActionType,
    AnalysisCaseSummary,
    AnalysisEnvelope,
    AnalysisWarning,
    CodeReference,
    Confidence,
    IssueSource,
    IssueType,
    MemberDesignCheck,
    RetrofitCandidate,
    Severity,
    StructuralIssue,
    build_recommendation_payload,
    case_summaries_from_dict,
    envelope_from_dict,
    extract_issues,
    generate_candidates,
    member_checks_from_design_check,
    normalize_warnings,
    warnings_to_payload,
)


# ============================================================
# AnalysisWarning serialization
# ============================================================

class TestAnalysisWarningSerialization:
    def test_to_dict_roundtrip(self):
        w = AnalysisWarning(
            code="design_check_failed",
            message="Cd not found",
            severity=Severity.WARNING,
            stage="design_check",
            recoverable=True,
            detail={"missing": "Cd"},
        )
        d = w.to_dict()
        assert d["code"] == "design_check_failed"
        assert d["message"] == "Cd not found"
        assert d["severity"] == "warning"
        assert d["stage"] == "design_check"
        assert d["recoverable"] is True
        assert d["detail"] == {"missing": "Cd"}
        # JSON serializable
        assert json.loads(json.dumps(d)) == d

    def test_drops_none_detail(self):
        w = AnalysisWarning(code="x", message="y")
        d = w.to_dict()
        assert "detail" not in d

    def test_from_legacy_string_with_prefix(self):
        w = AnalysisWarning.from_legacy_string("rsa_failed: spectrum 없음")
        assert w.code == "rsa_failed"
        assert w.message == "spectrum 없음"
        assert w.severity == Severity.WARNING
        assert w.recoverable is True

    def test_from_legacy_string_without_prefix(self):
        w = AnalysisWarning.from_legacy_string("free-form text")
        assert w.code == "legacy_warning"
        assert w.message == "free-form text"

    def test_normalize_warnings_mixed_shapes(self):
        raw = [
            "rsa_failed: foo",                                      # legacy str
            {"code": "x", "message": "y", "severity": "error"},     # dict shape
            AnalysisWarning(code="z", message="zz"),                # already typed
        ]
        out = normalize_warnings(raw, stage="t")
        assert len(out) == 3
        assert out[0].code == "rsa_failed"
        assert out[1].severity == "error"
        assert out[2].code == "z"

    def test_warnings_to_payload_keeps_legacy_mirror(self):
        warnings = [AnalysisWarning(code="rsa_failed", message="missing")]
        payload = warnings_to_payload(warnings)
        assert payload["warnings"][0]["code"] == "rsa_failed"
        assert payload["warning_messages"] == ["rsa_failed: missing"]


# ============================================================
# CodeReference
# ============================================================

class TestCodeReference:
    def test_basic_serialization(self):
        ref = CodeReference(
            standard_id="KDS 41 31 00",
            clause_id="H1",
            title="조합응력",
        )
        d = ref.to_dict()
        assert d["standard_id"] == "KDS 41 31 00"
        assert d["clause_id"] == "H1"
        # None fields dropped
        assert "quote" not in d
        assert "relevance_reason" not in d


# ============================================================
# Issue extraction from design_check
# ============================================================

def _make_design_check(*, members=None, drift_checks=None, drift_allow=0.020):
    return {
        "overall_status": "NG",
        "drift_check": {
            "status": "NG" if drift_checks else "OK",
            "code_ref": "KDS 41 17 00 §8.2.3",
            "Cd": 3.0, "IE": 1.0, "importance": "II",
            "allowable": drift_allow,
            "checks": drift_checks or [],
        } if drift_checks is not None else None,
        "member_check": {
            "status": "NG",
            "code_ref": "KDS 41 31 00 / AISC 360 H1",
            "members": members or [],
            "summary": {"total": len(members or []), "ok": 0, "ng": len(members or [])},
        } if members is not None else None,
        "critical_issues": [],
        "summary": {},
    }


class TestIssueExtraction:
    def test_interaction_over_one_yields_strength_exceeded(self):
        dc = _make_design_check(members=[{
            "member_id": 7,
            "type": "column",
            "section": "H-300x300",
            "governing_combo": "1.2DL+1.0LL+1.0EQX",
            "ratios": {"interaction": 1.35, "shear": 0.4, "formula": "H1: 1.35"},
            "demand": {"Pu": 200, "Mux": 80, "Muy": 10, "Vu": 30},
            "capacity": {"phiPn_kN": 800},
            "status": "NG",
        }])

        result = extract_issues(design_check=dc, warnings=None)
        types = [i.issue_type for i in result.issues]
        assert IssueType.STRENGTH_EXCEEDED in types
        s = [i for i in result.issues if i.issue_type == IssueType.STRENGTH_EXCEEDED][0]
        assert s.member_id == 7
        assert s.severity == Severity.ERROR
        assert s.source == IssueSource.DESIGN_CHECK
        assert s.demand_capacity_ratio == 1.35
        assert s.governing_combo == "1.2DL+1.0LL+1.0EQX"
        assert any(c.standard_id == "KDS 41 31 00" for c in s.code_refs)

    def test_shear_over_one_yields_separate_issue(self):
        dc = _make_design_check(members=[{
            "member_id": 9,
            "type": "beam_x",
            "section": "H-400x200",
            "governing_combo": "1.4DL",
            "ratios": {"interaction": 0.6, "shear": 1.2},
            "status": "NG",
        }])
        result = extract_issues(design_check=dc, warnings=None)
        kinds = [i.issue_type for i in result.issues]
        # interaction <= 1.0 → no strength_exceeded; shear > 1.0 → shear_exceeded
        assert IssueType.SHEAR_EXCEEDED in kinds
        assert IssueType.STRENGTH_EXCEEDED not in kinds

    def test_interaction_under_one_no_issue(self):
        dc = _make_design_check(members=[{
            "member_id": 1, "type": "column", "section": "H-300x300",
            "ratios": {"interaction": 0.42, "shear": 0.1}, "status": "OK",
        }])
        result = extract_issues(design_check=dc, warnings=None)
        assert all(i.issue_type != IssueType.STRENGTH_EXCEEDED for i in result.issues)

    def test_drift_ng_yields_issue(self):
        dc = _make_design_check(members=[], drift_checks=[{
            "story": 3, "direction": "X", "combo": "1.0DL+1.0LL+1.0EQX",
            "elastic_drift": 0.0080, "inelastic_drift": 0.0240,
            "allowable": 0.020, "ratio": 1.20, "status": "NG",
            "height_m": 3.5, "drift_inv": "1/41",
        }], drift_allow=0.020)
        result = extract_issues(design_check=dc, warnings=None)
        drift_issues = [i for i in result.issues if i.issue_type == IssueType.DRIFT_EXCEEDED]
        assert len(drift_issues) == 1
        assert drift_issues[0].demand_capacity_ratio == 1.20
        assert drift_issues[0].evidence["story"] == 3
        assert drift_issues[0].evidence["direction"] == "X"

    def test_missing_design_check_emits_warning_issue(self):
        result = extract_issues(design_check=None, warnings=None)
        types = [i.issue_type for i in result.issues]
        assert IssueType.MISSING_DESIGN_CHECK in types
        m = [i for i in result.issues if i.issue_type == IssueType.MISSING_DESIGN_CHECK][0]
        assert m.severity == Severity.WARNING

    def test_warnings_lifted_to_issues(self):
        result = extract_issues(
            design_check=_make_design_check(members=[]),
            warnings=["rsa_failed: missing spectrum"],
        )
        warning_issues = [i for i in result.issues if i.issue_type == IssueType.ANALYSIS_WARNING]
        assert len(warning_issues) == 1
        assert "rsa_failed" in warning_issues[0].description

    def test_summary_counts(self):
        dc = _make_design_check(members=[
            {"member_id": 1, "type": "col", "section": "X",
             "ratios": {"interaction": 1.2, "shear": 0.5}, "status": "NG"},
            {"member_id": 2, "type": "col", "section": "X",
             "ratios": {"interaction": 0.8, "shear": 1.1}, "status": "NG"},
        ])
        result = extract_issues(design_check=dc, warnings=["foo: bar"])
        s = result.summary
        assert s["total"] == len(result.issues)
        assert s["by_type"].get(IssueType.STRENGTH_EXCEEDED, 0) == 1
        assert s["by_type"].get(IssueType.SHEAR_EXCEEDED, 0) == 1
        assert s["by_type"].get(IssueType.ANALYSIS_WARNING, 0) == 1

    def test_malformed_member_check_emits_extractor_warning(self):
        dc = {"member_check": {"members": "not-a-list"}}
        sink: list[AnalysisWarning] = []
        result = extract_issues(design_check=dc, warnings=None, out_warnings=sink)
        assert any(w.code == "issue_extract_malformed_members" for w in sink)
        # No member issues should have been produced
        assert all(i.issue_type != IssueType.STRENGTH_EXCEEDED for i in result.issues)


# ============================================================
# Candidate generation
# ============================================================

def _issue(**kw):
    base = dict(
        issue_id="iss_test",
        issue_type=IssueType.STRENGTH_EXCEEDED,
        severity=Severity.ERROR,
        source=IssueSource.DESIGN_CHECK,
        description="…",
    )
    base.update(kw)
    return StructuralIssue(**base)


class TestCandidateGenerator:
    def test_strength_exceeded_yields_increase_section(self):
        issue = _issue(
            member_id=7,
            demand_capacity_ratio=1.35,
            evidence={"section": "H-300x300"},
        )
        cands = generate_candidates([issue])
        assert len(cands) == 1
        c = cands[0]
        assert c.action_type == ActionType.INCREASE_SECTION
        assert c.member_id == 7
        assert c.requires_reanalysis is True
        assert c.confidence in (Confidence.LOW, Confidence.MEDIUM, Confidence.HIGH)
        # No RAG yet — code_refs must be empty placeholder
        assert c.code_refs == []

    def test_strength_exceeded_without_member_id_falls_back_to_review(self):
        issue = _issue(member_id=None, demand_capacity_ratio=1.5)
        c = generate_candidates([issue])[0]
        assert c.action_type == ActionType.REQUIRES_ENGINEER_REVIEW

    def test_drift_exceeded_yields_lateral_resistance(self):
        issue = _issue(
            issue_type=IssueType.DRIFT_EXCEEDED,
            demand_capacity_ratio=1.20,
            evidence={"story": 3, "direction": "X"},
        )
        c = generate_candidates([issue])[0]
        assert c.action_type == ActionType.ADD_LATERAL_RESISTANCE
        assert c.requires_reanalysis is True

    def test_missing_design_check_yields_engineer_review(self):
        issue = _issue(issue_type=IssueType.MISSING_DESIGN_CHECK, severity=Severity.WARNING)
        c = generate_candidates([issue])[0]
        assert c.action_type == ActionType.REQUIRES_ENGINEER_REVIEW
        assert c.requires_reanalysis is True

    def test_warning_severity_filters_out_info_only(self):
        info_issue = _issue(
            issue_type=IssueType.ANALYSIS_WARNING,
            severity=Severity.WARNING,
        )
        err_issue = _issue(
            issue_type=IssueType.ANALYSIS_WARNING,
            severity=Severity.ERROR,
            issue_id="iss_err",
        )
        cands = generate_candidates([info_issue, err_issue])
        # Only the error-level warning gets a candidate.
        assert len(cands) == 1
        assert cands[0].issue_id == "iss_err"
        assert cands[0].action_type == ActionType.REQUIRES_ENGINEER_REVIEW

    def test_candidate_serializes_to_json(self):
        issue = _issue(
            member_id=7,
            demand_capacity_ratio=1.5,
            evidence={"section": "H-300x300"},
        )
        c = generate_candidates([issue])[0]
        d = c.to_dict()
        assert json.loads(json.dumps(d)) == d
        assert d["requires_reanalysis"] is True


# ============================================================
# Adapters & end-to-end build_recommendation_payload
# ============================================================

class TestAdapters:
    def test_envelope_from_dict(self):
        env = envelope_from_dict({"max_dx_mm": 12.3, "max_drift_x": 0.005})
        assert isinstance(env, AnalysisEnvelope)
        assert env.max_dx_mm == 12.3
        assert env.max_drift_x == 0.005

    def test_case_summaries(self):
        case_data = {
            "DL": {"summary": {"max_dx_mm": 1.0, "max_moment_kNm": 10.0}},
            "1.2DL+1.6LL": {"summary": {"max_dx_mm": 5.0}},
        }
        summaries = case_summaries_from_dict(case_data, combo_names=["1.2DL+1.6LL"])
        names = {s.case_name: s for s in summaries}
        assert names["DL"].is_combination is False
        assert names["1.2DL+1.6LL"].is_combination is True
        assert names["1.2DL+1.6LL"].max_dx_mm == 5.0

    def test_member_checks_extraction(self):
        dc = _make_design_check(members=[{
            "member_id": 3, "type": "column", "section": "H-300x300",
            "governing_combo": "1.2DL", "story": 2,
            "ratios": {"interaction": 0.7, "shear": 0.2}, "status": "OK",
        }])
        out = member_checks_from_design_check(dc, material="SS275")
        assert len(out) == 1
        assert isinstance(out[0], MemberDesignCheck)
        assert out[0].member_id == 3
        assert out[0].material == "SS275"
        assert out[0].demand_capacity_ratio == 0.7


class TestBuildRecommendationPayload:
    def test_end_to_end_payload_shape(self):
        dc = _make_design_check(members=[{
            "member_id": 1, "type": "column", "section": "H-300x300",
            "governing_combo": "1.2DL+1.0LL",
            "ratios": {"interaction": 1.4, "shear": 0.3},
            "status": "NG",
        }])
        payload = build_recommendation_payload(
            design_check=dc,
            raw_warnings=["rsa_failed: missing spectrum"],
            stage="test",
        )
        # Required top-level keys
        for k in ("warnings", "warning_messages", "issues",
                  "recommendation_candidates", "recommendation_summary"):
            assert k in payload

        # Issues include the strength exceedance + the warning lift
        types = [i["issue_type"] for i in payload["issues"]]
        assert IssueType.STRENGTH_EXCEEDED in types
        assert IssueType.ANALYSIS_WARNING in types

        # At least one increase_section candidate
        actions = [c["action_type"] for c in payload["recommendation_candidates"]]
        assert ActionType.INCREASE_SECTION in actions

        # Summary advertises that RAG/LLM are NOT live
        s = payload["recommendation_summary"]
        assert s["rag_enabled"] is False
        assert s["llm_enabled"] is False
        assert s["num_issues"] >= 2
        assert s["num_candidates"] >= 1

        # Legacy mirror retains string view
        assert any("rsa_failed" in m for m in payload["warning_messages"])

    def test_no_design_check_no_warnings(self):
        payload = build_recommendation_payload(
            design_check=None, raw_warnings=None, stage="test",
        )
        # Missing design check → exactly one issue of that type
        types = [i["issue_type"] for i in payload["issues"]]
        assert types.count(IssueType.MISSING_DESIGN_CHECK) == 1
        # Candidate must be engineer review, not increase_section
        actions = {c["action_type"] for c in payload["recommendation_candidates"]}
        assert ActionType.REQUIRES_ENGINEER_REVIEW in actions
        assert ActionType.INCREASE_SECTION not in actions
