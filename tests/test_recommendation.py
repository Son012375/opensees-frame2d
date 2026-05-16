"""Tests for the recommendation pipeline foundation.

Scope: deterministic layer only — issue extraction, candidate generation,
warning normalization, deterministic IDs, parse_only mode. NO RAG/LLM
coverage here (none is implemented).

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
    ChangeOperation,
    CodeReference,
    Confidence,
    IssueSource,
    IssueType,
    MODE_ANALYZE,
    MODE_PARSE_ONLY,
    MemberDesignCheck,
    RetrofitCandidate,
    Severity,
    StructuralIssue,
    _should_block_auto_candidate,
    build_context_index,
    build_recommendation_payload,
    case_summaries_from_dict,
    envelope_from_dict,
    extract_issues,
    generate_candidates,
    make_candidate_id,
    make_issue_id,
    member_checks_from_design_check,
    normalize_warnings,
    slugify,
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
            "rsa_failed: foo",
            {"code": "x", "message": "y", "severity": "error"},
            AnalysisWarning(code="z", message="zz"),
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
# CodeReference (incl. RAG hints)
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
        assert "quote" not in d
        assert "relevance_reason" not in d

    def test_rag_hints_serialize(self):
        ref = CodeReference(
            standard_id="KDS 41 31 00",
            clause_id="H1",
            query_hint="combined force interaction",
            topic="steel_member_strength",
            material="steel",
            limit_state="combined_force_interaction",
            jurisdiction="KDS",
        )
        d = ref.to_dict()
        for k in ("query_hint", "topic", "material",
                  "limit_state", "jurisdiction"):
            assert k in d, f"{k} missing from serialized CodeReference"
        assert d["topic"] == "steel_member_strength"
        assert d["limit_state"] == "combined_force_interaction"


# ============================================================
# Deterministic IDs
# ============================================================

class TestDeterministicIds:
    def test_slugify_strips_specials(self):
        assert slugify("1.2DL+1.0LL") == "1_2dl_1_0ll"
        assert slugify("X 방향") == "x"  # non-ascii dropped
        assert slugify(None) == ""
        assert slugify("") == ""

    def test_issue_id_is_stable_across_calls(self):
        kw = dict(
            issue_type=IssueType.STRENGTH_EXCEEDED,
            member_id=7,
            governing_combo="1.2DL+1.0LL",
        )
        a = make_issue_id(**kw)
        b = make_issue_id(**kw)
        assert a == b
        # Shape: starts with iss_<type>_m7_…
        assert a.startswith("iss_strength_exceeded_m7_")

    def test_issue_id_differs_by_member(self):
        a = make_issue_id(issue_type=IssueType.STRENGTH_EXCEEDED, member_id=7,
                          governing_combo="1.2DL+1.0LL")
        b = make_issue_id(issue_type=IssueType.STRENGTH_EXCEEDED, member_id=8,
                          governing_combo="1.2DL+1.0LL")
        assert a != b

    def test_drift_issue_id_uses_story_and_direction(self):
        a = make_issue_id(
            issue_type=IssueType.DRIFT_EXCEEDED,
            story=3, direction="X", governing_combo="1.0DL+1.0LL+1.0EQX",
        )
        assert a.startswith("iss_drift_exceeded_s3_x_")

    def test_candidate_id_is_stable_and_derives_from_issue(self):
        iid = "iss_strength_exceeded_m7_1_2dl_1_0ll"
        a = make_candidate_id(issue_id=iid, action_type=ActionType.INCREASE_SECTION)
        b = make_candidate_id(issue_id=iid, action_type=ActionType.INCREASE_SECTION)
        assert a == b
        assert iid in a
        assert "increase_section" in a

    def test_candidate_id_url_safe(self):
        cid = make_candidate_id(
            issue_id="iss_strength_exceeded_m7_1_2dl_1_0ll",
            action_type=ActionType.INCREASE_SECTION,
        )
        # ASCII alnum + underscores/dashes only
        import re
        assert re.match(r"^[a-z0-9_\-]+$", cid)


# ============================================================
# Issue extraction
# ============================================================

def _make_design_check(*, members=None, drift_checks=None, drift_allow=0.020):
    return {
        "overall_status": "NG",
        "drift_check": ({
            "status": "NG" if drift_checks else "OK",
            "code_ref": "KDS 41 17 00 §8.2.3",
            "Cd": 3.0, "IE": 1.0, "importance": "II",
            "allowable": drift_allow,
            "checks": drift_checks or [],
        } if drift_checks is not None else None),
        "member_check": ({
            "status": "NG",
            "code_ref": "KDS 41 31 00 / AISC 360 H1",
            "members": members or [],
            "summary": {"total": len(members or []), "ok": 0, "ng": len(members or [])},
        } if members is not None else None),
        "critical_issues": [],
        "summary": {},
    }


def _mk_member(**overrides):
    base = {
        "member_id": 7, "type": "column", "section": "H-300x300",
        "governing_combo": "1.2DL+1.0LL+1.0EQX",
        "ratios": {"interaction": 1.35, "shear": 0.4},
        "demand": {"Pu": 200, "Mux": 80, "Muy": 10, "Vu": 30},
        "capacity": {"phiPn_kN": 800},
        "status": "NG",
        "story": 2,
    }
    base.update(overrides)
    return base


class TestIssueExtraction:
    def test_interaction_over_one_yields_strength_exceeded(self):
        dc = _make_design_check(members=[_mk_member()])
        result = extract_issues(design_check=dc, warnings=None)
        s = [i for i in result.issues if i.issue_type == IssueType.STRENGTH_EXCEEDED]
        assert len(s) == 1
        assert s[0].member_id == 7
        assert s[0].severity == Severity.ERROR
        assert s[0].source == IssueSource.DESIGN_CHECK
        assert s[0].demand_capacity_ratio == 1.35
        assert s[0].governing_combo == "1.2DL+1.0LL+1.0EQX"
        # Code ref carries RAG hints
        assert any(c.standard_id == "KDS 41 31 00" for c in s[0].code_refs)
        assert s[0].code_refs[0].topic == "steel_member_strength"
        assert s[0].code_refs[0].limit_state == "combined_force_interaction"
        assert s[0].code_refs[0].material == "steel"

    def test_shear_over_one_yields_separate_issue_with_shear_ref(self):
        dc = _make_design_check(members=[_mk_member(
            ratios={"interaction": 0.6, "shear": 1.2},
        )])
        result = extract_issues(design_check=dc, warnings=None)
        s = [i for i in result.issues if i.issue_type == IssueType.SHEAR_EXCEEDED]
        assert len(s) == 1
        # shear gets its own G2-style hint
        assert s[0].code_refs[0].topic == "steel_member_shear"
        assert s[0].code_refs[0].limit_state == "shear_strength"

    def test_interaction_under_one_no_issue(self):
        dc = _make_design_check(members=[_mk_member(
            ratios={"interaction": 0.42, "shear": 0.1}, status="OK",
        )])
        result = extract_issues(design_check=dc, warnings=None)
        assert all(i.issue_type != IssueType.STRENGTH_EXCEEDED for i in result.issues)

    def test_drift_ng_yields_issue(self):
        dc = _make_design_check(members=[], drift_checks=[{
            "story": 3, "direction": "X", "combo": "1.0DL+1.0LL+1.0EQX",
            "elastic_drift": 0.0080, "inelastic_drift": 0.0240,
            "allowable": 0.020, "ratio": 1.20, "status": "NG",
            "height_m": 3.5, "drift_inv": "1/41",
        }])
        result = extract_issues(design_check=dc, warnings=None)
        drift = [i for i in result.issues if i.issue_type == IssueType.DRIFT_EXCEEDED]
        assert len(drift) == 1
        assert drift[0].story == 3
        assert drift[0].level == 3.5
        assert drift[0].demand_capacity_ratio == 1.20
        assert drift[0].evidence["direction"] == "X"
        # drift ref hints
        assert drift[0].code_refs[0].topic == "seismic_story_drift"

    def test_missing_design_check_emits_warning_issue(self):
        result = extract_issues(design_check=None, warnings=None)
        types = [i.issue_type for i in result.issues]
        assert IssueType.MISSING_DESIGN_CHECK in types

    def test_parse_only_skips_missing_design_check(self):
        result = extract_issues(
            design_check=None, warnings=None,
            include_missing_design_check=False,
        )
        types = [i.issue_type for i in result.issues]
        assert IssueType.MISSING_DESIGN_CHECK not in types
        assert result.summary["total"] == 0

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
            _mk_member(member_id=1, ratios={"interaction": 1.2, "shear": 0.5}),
            _mk_member(member_id=2, ratios={"interaction": 0.8, "shear": 1.1}),
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
        assert all(i.issue_type != IssueType.STRENGTH_EXCEEDED for i in result.issues)


# ============================================================
# Issue top-level context / analysis_metadata enrichment
# ============================================================

class TestIssueContextFields:
    def test_member_context_serializes_at_top_level(self):
        """member_type/section/story/connected_node_ids must be top-level."""
        dc = _make_design_check(members=[_mk_member()])
        result = extract_issues(design_check=dc, warnings=None)
        d = result.issues[0].to_dict()
        assert d["member_type"] == "column"
        assert d["section"] == "H-300x300"
        assert d["story"] == 2
        # Even when empty, connected_node_ids must be present as a list.
        assert isinstance(d["connected_node_ids"], list)

    def test_member_info_metadata_fills_connected_node_ids(self):
        """member_info from analysis_metadata fills nodes when design_check doesn't."""
        # design_check row lacks node info
        dc = _make_design_check(members=[_mk_member(
            **{"connected_node_ids": None},
        )])
        # member_info knows the endpoints
        meta = {
            "member_info": [{
                "member_id": 7, "type": "column", "section": "H-300x300",
                "ni": 101, "nj": 102, "story": 2,
            }],
            "material_name": "SS275",
        }
        result = extract_issues(
            design_check=dc, warnings=None, analysis_metadata=meta,
        )
        iss = next(i for i in result.issues if i.issue_type == IssueType.STRENGTH_EXCEEDED)
        assert iss.connected_node_ids == [101, 102]
        assert iss.material == "SS275"

    def test_updated_model_fills_member_context(self):
        """When member_info is missing, updated_model.elements is consulted."""
        dc = _make_design_check(members=[_mk_member(
            type=None,  # design_check row missing type
        )])
        # Remove type from row entirely to force lookup
        for m in dc["member_check"]["members"]:
            m.pop("type", None)
            m.pop("section", None)
        meta = {
            "updated_model": {
                "elements": {
                    "7": {
                        "id": 7,
                        "elem_type": "beam_x",
                        "section": "H-400x200",
                        "ni": 11, "nj": 12,
                    },
                },
                "nodes": {
                    "11": {"x": 0.0, "y": 0.0, "z": 3.5},
                    "12": {"x": 8.0, "y": 0.0, "z": 3.5},
                },
            },
            "material_name": "SS275",
        }
        result = extract_issues(
            design_check=dc, warnings=None, analysis_metadata=meta,
        )
        iss = next(i for i in result.issues if i.issue_type == IssueType.STRENGTH_EXCEEDED)
        assert iss.member_type == "beam_x"
        assert iss.section == "H-400x200"
        assert iss.connected_node_ids == [11, 12]
        assert iss.coordinates == {"x": 0.0, "y": 0.0, "z": 3.5}

    def test_context_index_falls_back_to_default_material(self):
        ctx_idx = build_context_index({
            "material_name": "SS275",
            "member_info": [{"member_id": 9, "type": "column", "section": "X"}],
        })
        ctx = ctx_idx.get(9)
        # Default material is injected on access
        assert ctx.material == "SS275"


# ============================================================
# Candidate generation
# ============================================================

def _full_strength_issue(**overrides):
    """StructuralIssue with all the info needed to *not* be blocked."""
    base = dict(
        issue_id="iss_strength_exceeded_m7_dl",
        issue_type=IssueType.STRENGTH_EXCEEDED,
        severity=Severity.ERROR,
        source=IssueSource.DESIGN_CHECK,
        description="…",
        member_id=7,
        member_type="column",
        section="H-300x300",
        demand_capacity_ratio=1.35,
        evidence={"section": "H-300x300", "type": "column"},
    )
    base.update(overrides)
    return StructuralIssue(**base)


class TestCandidateGenerator:
    def test_strength_exceeded_yields_replace_section_with_proposed_change(self):
        issue = _full_strength_issue()
        cands = generate_candidates([issue])
        assert len(cands) == 1
        c = cands[0]
        assert c.action_type == ActionType.INCREASE_SECTION
        assert c.member_id == 7
        assert c.requires_reanalysis is True
        # proposed_change contract
        assert c.proposed_change["operation"] == ChangeOperation.REPLACE_SECTION
        assert c.proposed_change["from"] == "H-300x300"
        assert c.proposed_change["to"] is None
        assert c.proposed_change["requires_user_selection"] is True
        # target
        assert c.target == {
            "member_id": 7, "element_id": None, "member_type": "column"
        }
        # No RAG yet — code_refs empty
        assert c.code_refs == []

    def test_candidate_id_is_deterministic_for_same_issue(self):
        issue = _full_strength_issue(
            issue_id="iss_strength_exceeded_m7_1_2dl_1_0ll",
        )
        a = generate_candidates([issue])[0]
        b = generate_candidates([issue])[0]
        assert a.candidate_id == b.candidate_id
        assert "increase_section" in a.candidate_id
        assert "iss_strength_exceeded_m7_1_2dl_1_0ll" in a.candidate_id

    def test_strength_exceeded_without_member_id_falls_back_to_review(self):
        issue = _full_strength_issue(member_id=None)
        c = generate_candidates([issue])[0]
        assert c.action_type == ActionType.REQUIRES_ENGINEER_REVIEW
        assert c.proposed_change["operation"] == ChangeOperation.MANUAL_REVIEW

    def test_strength_exceeded_without_section_falls_back_to_review(self):
        issue = _full_strength_issue(section=None, evidence={"type": "column"})
        c = generate_candidates([issue])[0]
        assert c.action_type == ActionType.REQUIRES_ENGINEER_REVIEW

    def test_strength_exceeded_without_member_type_falls_back_to_review(self):
        issue = _full_strength_issue(member_type=None,
                                     evidence={"section": "H-300x300"})
        c = generate_candidates([issue])[0]
        assert c.action_type == ActionType.REQUIRES_ENGINEER_REVIEW

    def test_drift_exceeded_yields_add_lateral_resistance(self):
        issue = StructuralIssue(
            issue_id="iss_drift_exceeded_s3_x",
            issue_type=IssueType.DRIFT_EXCEEDED,
            severity=Severity.ERROR,
            source=IssueSource.DESIGN_CHECK,
            description="…",
            demand_capacity_ratio=1.20,
            story=3,
            evidence={"story": 3, "direction": "X"},
        )
        c = generate_candidates([issue])[0]
        assert c.action_type == ActionType.ADD_LATERAL_RESISTANCE
        assert c.proposed_change["operation"] == ChangeOperation.ADD_LATERAL_RESISTANCE
        assert c.requires_reanalysis is True
        assert c.target == {"scope": "story", "story": 3, "direction": "X"}

    def test_drift_exceeded_without_story_falls_back_to_review(self):
        issue = StructuralIssue(
            issue_id="iss_drift_no_story",
            issue_type=IssueType.DRIFT_EXCEEDED,
            severity=Severity.ERROR,
            source=IssueSource.DESIGN_CHECK,
            description="…",
            demand_capacity_ratio=1.10,
            # story is None and evidence empty
        )
        c = generate_candidates([issue])[0]
        assert c.action_type == ActionType.REQUIRES_ENGINEER_REVIEW

    def test_missing_design_check_yields_engineer_review(self):
        issue = StructuralIssue(
            issue_id="iss_missing_design_check_global",
            issue_type=IssueType.MISSING_DESIGN_CHECK,
            severity=Severity.WARNING,
            source=IssueSource.ANALYSIS,
            description="…",
        )
        c = generate_candidates([issue])[0]
        assert c.action_type == ActionType.REQUIRES_ENGINEER_REVIEW
        assert c.proposed_change["operation"] == ChangeOperation.MANUAL_REVIEW

    def test_info_warning_yields_no_candidate(self):
        info = StructuralIssue(
            issue_id="iss_warning_info",
            issue_type=IssueType.ANALYSIS_WARNING,
            severity=Severity.WARNING,
            source=IssueSource.WARNING,
            description="rsa_failed",
        )
        cands = generate_candidates([info])
        assert cands == []

    def test_error_warning_yields_engineer_review(self):
        err = StructuralIssue(
            issue_id="iss_warning_err",
            issue_type=IssueType.ANALYSIS_WARNING,
            severity=Severity.ERROR,
            source=IssueSource.WARNING,
            description="bad",
        )
        c = generate_candidates([err])[0]
        assert c.action_type == ActionType.REQUIRES_ENGINEER_REVIEW

    def test_candidate_serializes_to_json(self):
        issue = _full_strength_issue()
        c = generate_candidates([issue])[0]
        d = c.to_dict()
        assert json.loads(json.dumps(d)) == d
        assert d["requires_reanalysis"] is True
        assert d["proposed_change"]["operation"] == "replace_section"

    def test_block_helper_reports_each_rule(self):
        """_should_block_auto_candidate must explain each refusal."""
        # missing design_check → blocked
        iss = StructuralIssue(
            issue_id="x", issue_type=IssueType.MISSING_DESIGN_CHECK,
            severity=Severity.WARNING, source=IssueSource.ANALYSIS,
            description="",
        )
        assert _should_block_auto_candidate(iss) is not None

        # strength without member_id → blocked
        iss = StructuralIssue(
            issue_id="x", issue_type=IssueType.STRENGTH_EXCEEDED,
            severity=Severity.ERROR, source=IssueSource.DESIGN_CHECK,
            description="", member_type="col", section="X",
        )
        assert _should_block_auto_candidate(iss) is not None

        # strength without member_type → blocked
        iss = StructuralIssue(
            issue_id="x", issue_type=IssueType.STRENGTH_EXCEEDED,
            severity=Severity.ERROR, source=IssueSource.DESIGN_CHECK,
            description="", member_id=1, section="X",
        )
        assert _should_block_auto_candidate(iss) is not None

        # strength without section → blocked
        iss = StructuralIssue(
            issue_id="x", issue_type=IssueType.STRENGTH_EXCEEDED,
            severity=Severity.ERROR, source=IssueSource.DESIGN_CHECK,
            description="", member_id=1, member_type="col",
        )
        assert _should_block_auto_candidate(iss) is not None

        # drift without story → blocked
        iss = StructuralIssue(
            issue_id="x", issue_type=IssueType.DRIFT_EXCEEDED,
            severity=Severity.ERROR, source=IssueSource.DESIGN_CHECK,
            description="",
        )
        assert _should_block_auto_candidate(iss) is not None


# ============================================================
# Adapters
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
        dc = _make_design_check(members=[_mk_member(
            ratios={"interaction": 0.7, "shear": 0.2}, status="OK",
        )])
        out = member_checks_from_design_check(dc, material="SS275")
        assert len(out) == 1
        assert isinstance(out[0], MemberDesignCheck)
        assert out[0].material == "SS275"
        assert out[0].demand_capacity_ratio == 0.7


# ============================================================
# End-to-end build_recommendation_payload
# ============================================================

class TestBuildRecommendationPayload:
    def test_end_to_end_payload_shape(self):
        dc = _make_design_check(members=[_mk_member(
            ratios={"interaction": 1.4, "shear": 0.3}, status="NG",
        )])
        payload = build_recommendation_payload(
            design_check=dc,
            raw_warnings=["rsa_failed: missing spectrum"],
            stage="test",
            mode=MODE_ANALYZE,
        )
        for k in ("warnings", "warning_messages", "issues",
                  "recommendation_candidates", "recommendation_summary"):
            assert k in payload

        # Issues contain the strength + warning lift
        types = [i["issue_type"] for i in payload["issues"]]
        assert IssueType.STRENGTH_EXCEEDED in types
        assert IssueType.ANALYSIS_WARNING in types

        # Strength issue serialized with top-level context
        sissue = next(i for i in payload["issues"]
                      if i["issue_type"] == IssueType.STRENGTH_EXCEEDED)
        assert sissue["member_type"] == "column"
        assert sissue["section"] == "H-300x300"

        # Candidate carries proposed_change
        sc = next(c for c in payload["recommendation_candidates"]
                  if c["action_type"] == ActionType.INCREASE_SECTION)
        assert sc["proposed_change"]["operation"] == "replace_section"
        assert sc["proposed_change"]["from"] == "H-300x300"
        assert sc["proposed_change"]["to"] is None
        assert sc["requires_reanalysis"] is True

        # Summary advertises RAG/LLM disabled + current mode
        s = payload["recommendation_summary"]
        assert s["rag_enabled"] is False
        assert s["llm_enabled"] is False
        assert s["mode"] == MODE_ANALYZE

    def test_parse_only_mode_skips_missing_design_check(self):
        payload = build_recommendation_payload(
            design_check=None, raw_warnings=None,
            stage="parse-ifc", mode=MODE_PARSE_ONLY,
        )
        types = [i["issue_type"] for i in payload["issues"]]
        assert IssueType.MISSING_DESIGN_CHECK not in types
        assert payload["recommendation_candidates"] == []
        assert payload["recommendation_summary"]["mode"] == MODE_PARSE_ONLY

    def test_analyze_mode_still_flags_missing_design_check(self):
        payload = build_recommendation_payload(
            design_check=None, raw_warnings=None,
            stage="analyze", mode=MODE_ANALYZE,
        )
        types = [i["issue_type"] for i in payload["issues"]]
        assert IssueType.MISSING_DESIGN_CHECK in types
        # Yields an engineer-review candidate (manual_review)
        actions = {c["action_type"] for c in payload["recommendation_candidates"]}
        assert ActionType.REQUIRES_ENGINEER_REVIEW in actions

    def test_metadata_enriches_issue_then_unlocks_candidate(self):
        """If design_check is sparse but metadata fills the gaps, candidate
        should escape the no-auto-candidate guard."""
        dc = _make_design_check(members=[{
            "member_id": 7, "governing_combo": "1.2DL+1.6LL",
            "ratios": {"interaction": 1.5, "shear": 0.3},
            "status": "NG",
            # no type/section in design_check row
        }])
        meta = {
            "member_info": [{
                "member_id": 7, "type": "column", "section": "H-300x300",
                "ni": 11, "nj": 12,
            }],
            "material_name": "SS275",
        }
        payload = build_recommendation_payload(
            design_check=dc, raw_warnings=None,
            analysis_metadata=meta, stage="t", mode=MODE_ANALYZE,
        )
        # Strength issue resolved with member_type + section from metadata
        sissue = next(i for i in payload["issues"]
                      if i["issue_type"] == IssueType.STRENGTH_EXCEEDED)
        assert sissue["member_type"] == "column"
        assert sissue["section"] == "H-300x300"
        assert sissue["material"] == "SS275"
        assert sissue["connected_node_ids"] == [11, 12]
        # Candidate not blocked because metadata filled the missing fields
        cand = next(c for c in payload["recommendation_candidates"]
                    if c["issue_id"] == sissue["issue_id"])
        assert cand["action_type"] == ActionType.INCREASE_SECTION
        assert cand["proposed_change"]["from"] == "H-300x300"
