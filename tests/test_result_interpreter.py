"""Result Interpreter 모듈 테스트.

실행: cd mcp-server && python -m pytest ../tests/test_result_interpreter.py -v
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mcp-server"))

from core.result_interpreter import (
    interpret_results,
    _classify_severity,
    _interpret_drift,
    _interpret_members,
    _interpret_modal,
    _diagnose_failure,
    _suggest_improvements,
    _find_next_section,
)


# ============================================================
# Mock 헬퍼
# ============================================================

def _mock_design_check(
    drift_ratio=0.5, interaction_ratio=0.4,
    drift_status="OK", member_status="OK",
    ng_stories=0, ng_members=0,
    drift_checks=None, member_list=None,
):
    """테스트용 design_check dict 생성."""
    dc = {
        "overall_status": "NG" if (drift_status == "NG" or member_status == "NG") else "OK",
        "summary": {
            "max_drift_ratio": drift_ratio,
            "max_interaction_ratio": interaction_ratio,
            "ng_stories": ng_stories,
            "ng_members": ng_members,
        },
    }

    if drift_checks is not None or drift_status is not None:
        dc["drift_check"] = {
            "status": drift_status,
            "max_ratio": drift_ratio,
            "checks": drift_checks or [],
            "critical": {
                "story": 1, "direction": "X", "combo": "1.2DL+1.0EQX",
                "ratio": drift_ratio,
            } if drift_ratio > 1.0 else None,
        }

    if member_list is not None or member_status is not None:
        dc["member_check"] = {
            "status": member_status,
            "members": member_list or [],
            "summary": {
                "total": len(member_list or []),
                "ok": len([m for m in (member_list or []) if m.get("status") == "OK"]),
                "ng": ng_members,
                "max_interaction_ratio": interaction_ratio,
                "max_shear_ratio": 0.3,
            },
        }

    return dc


def _mock_member(member_id, mtype, story, interaction, status="OK", formula="H1-1b",
                  axial=0.1, bending_x=0.2, bending_y=0.05, shear=0.1):
    return {
        "member_id": member_id,
        "type": mtype,
        "story": story,
        "status": status,
        "ratios": {
            "interaction": interaction,
            "axial": axial,
            "bending_x": bending_x,
            "bending_y": bending_y,
            "shear": shear,
            "formula": formula,
        },
    }


class _MockMulti:
    """Frame3DMultiCaseResult 대체 mock."""
    def __init__(self, stories=None, bays_x=None, bays_y=None,
                 column_section="H-300x300", beam_x_section="H-400x200"):
        self.stories = stories or [4.0, 3.5]
        self.bays_x = bays_x or [8.0, 8.0]
        self.bays_y = bays_y or [8.0]
        self.column_section = column_section
        self.beam_x_section = beam_x_section


# ============================================================
# TestSeverity
# ============================================================

class TestSeverity:
    def test_safe(self):
        dc = _mock_design_check(drift_ratio=0.3, interaction_ratio=0.5)
        assert _classify_severity(dc) == "safe"

    def test_marginal(self):
        dc = _mock_design_check(drift_ratio=0.85, interaction_ratio=0.6)
        assert _classify_severity(dc) == "marginal"

    def test_moderate(self):
        dc = _mock_design_check(drift_ratio=1.15, interaction_ratio=0.9)
        assert _classify_severity(dc) == "moderate"

    def test_severe(self):
        dc = _mock_design_check(drift_ratio=1.5, interaction_ratio=1.4)
        assert _classify_severity(dc) == "severe"

    def test_empty_summary(self):
        dc = {"summary": {}}
        assert _classify_severity(dc) == "safe"


# ============================================================
# TestDrift
# ============================================================

class TestDrift:
    def test_no_drift_check(self):
        interp, findings = _interpret_drift(None)
        assert interp is None
        assert findings == []

    def test_drift_ok(self):
        drift_check = {
            "status": "OK",
            "max_ratio": 0.65,
            "checks": [
                {"story": 1, "direction": "X", "ratio": 0.65},
                {"story": 2, "direction": "X", "ratio": 0.55},
                {"story": 1, "direction": "Y", "ratio": 0.40},
                {"story": 2, "direction": "Y", "ratio": 0.35},
            ],
            "critical": None,
        }
        interp, findings = _interpret_drift(drift_check)
        assert interp is not None
        assert interp["soft_story_detected"] is False
        assert any(f["code"] == "R03" for f in findings)

    def test_drift_ng(self):
        drift_check = {
            "status": "NG",
            "max_ratio": 1.17,
            "checks": [
                {"story": 1, "direction": "X", "ratio": 1.17},
                {"story": 2, "direction": "X", "ratio": 0.80},
            ],
            "critical": {"story": 1, "direction": "X", "combo": "1.2DL+1.0EQX", "ratio": 1.17},
        }
        interp, findings = _interpret_drift(drift_check)
        assert interp["critical_story"] == 1
        assert interp["critical_direction"] == "X"
        assert any(f["code"] == "R02" for f in findings)

    def test_soft_story(self):
        drift_check = {
            "status": "NG",
            "max_ratio": 1.2,
            "checks": [
                {"story": 1, "direction": "X", "ratio": 1.2},
                {"story": 2, "direction": "X", "ratio": 0.3},
                {"story": 3, "direction": "X", "ratio": 0.25},
            ],
            "critical": {"story": 1, "direction": "X", "combo": "EQX", "ratio": 1.2},
        }
        interp, findings = _interpret_drift(drift_check)
        assert interp["soft_story_detected"] is True
        assert 1 in interp["soft_story_stories"]
        assert interp["pattern"] == "soft_story"
        assert any(f["code"] == "R01" for f in findings)

    def test_whipping(self):
        drift_check = {
            "status": "OK",
            "max_ratio": 0.9,
            "checks": [
                {"story": 1, "direction": "X", "ratio": 0.3},
                {"story": 2, "direction": "X", "ratio": 0.35},
                {"story": 3, "direction": "X", "ratio": 0.7},
                {"story": 4, "direction": "X", "ratio": 0.9},
            ],
            "critical": None,
        }
        interp, findings = _interpret_drift(drift_check)
        assert interp["pattern"] == "whipping"
        assert any(f["code"] == "R04" for f in findings)


# ============================================================
# TestMember
# ============================================================

class TestMember:
    def test_all_ok(self):
        members = [
            _mock_member(1, "column", 1, 0.5),
            _mock_member(2, "beam_x", 1, 0.3),
        ]
        interp, findings = _interpret_members({
            "status": "OK",
            "members": members,
            "summary": {"total": 2, "ok": 2, "ng": 0, "max_interaction_ratio": 0.5, "max_shear_ratio": 0.1},
        })
        assert interp["status"] == "OK"
        assert any(f["code"] == "R08" for f in findings)

    def test_weakest_link(self):
        members = [
            _mock_member(1, "column", 1, 1.15, status="NG"),
            _mock_member(2, "column", 1, 0.95),
            _mock_member(3, "beam_x", 1, 0.60),
        ]
        interp, findings = _interpret_members({
            "status": "NG",
            "members": members,
            "summary": {"total": 3, "ok": 2, "ng": 1, "max_interaction_ratio": 1.15, "max_shear_ratio": 0.1},
        })
        assert interp["weakest_link"]["member_id"] == 1
        assert interp["weakest_link"]["ratio"] == 1.15
        assert any(f["code"] == "R05" for f in findings)

    def test_ng_grouping(self):
        members = [
            _mock_member(1, "column", 1, 1.1, status="NG"),
            _mock_member(2, "column", 1, 1.05, status="NG"),
            _mock_member(3, "beam_x", 2, 1.02, status="NG"),
        ]
        interp, findings = _interpret_members({
            "status": "NG",
            "members": members,
            "summary": {"total": 3, "ok": 0, "ng": 3, "max_interaction_ratio": 1.1, "max_shear_ratio": 0.1},
        })
        assert interp["ng_by_story"][1] == 2
        assert interp["ng_by_type"]["column"] == 2
        assert any(f["code"] == "R07" for f in findings)

    def test_no_member_check(self):
        interp, findings = _interpret_members(None)
        assert interp is None
        assert findings == []


# ============================================================
# TestModal
# ============================================================

class TestModal:
    def test_no_modal(self):
        interp, findings = _interpret_modal(None, _MockMulti())
        assert interp is None
        assert findings == []

    def test_normal(self):
        modal = {
            "modes": [
                {"mode": 1, "period_s": 0.45, "direction": "TRAN-X", "dominance_pct": 75},
            ],
            "fundamental_periods": {"T1_x_s": 0.45, "T1_y_s": 0.30, "T1_rz_s": 0.20},
        }
        # h=7.5m → Ta = 0.085 * 7.5^0.75 ≈ 0.39
        interp, findings = _interpret_modal(modal, _MockMulti(stories=[4.0, 3.5]))
        assert interp is not None
        assert interp["flexibility_flag"] is False
        assert interp["torsion_risk"] is False
        assert any(f["code"] == "R12" for f in findings)

    def test_flexibility_flag(self):
        modal = {
            "modes": [
                {"mode": 1, "period_s": 1.2, "direction": "TRAN-X", "dominance_pct": 80},
            ],
            "fundamental_periods": {"T1_x_s": 1.2, "T1_y_s": 0.8, "T1_rz_s": 0.5},
        }
        # h=7.5m → Ta ≈ 0.39, T1/Ta ≈ 3.1 > 1.4
        interp, findings = _interpret_modal(modal, _MockMulti(stories=[4.0, 3.5]))
        assert interp["flexibility_flag"] is True
        assert any(f["code"] == "R09" for f in findings)

    def test_torsion_risk(self):
        modal = {
            "modes": [
                {"mode": 1, "period_s": 0.5, "direction": "ROTN-Z", "dominance_pct": 60},
            ],
            "fundamental_periods": {"T1_x_s": 0.4, "T1_y_s": 0.35, "T1_rz_s": 0.5},
        }
        interp, findings = _interpret_modal(modal, _MockMulti(stories=[4.0, 3.5]))
        assert interp["torsion_risk"] is True
        assert any(f["code"] == "R10" for f in findings)


# ============================================================
# TestDiagnosis
# ============================================================

class TestDiagnosis:
    def test_safe_returns_none(self):
        result = _diagnose_failure("safe", None, None, None, _MockMulti())
        assert result is None

    def test_soft_story_primary(self):
        drift_interp = {
            "soft_story_detected": True, "soft_story_stories": [1],
            "max_ratio": 1.2, "pattern": "soft_story",
        }
        result = _diagnose_failure("moderate", drift_interp, None, None, _MockMulti())
        assert result["primary_cause"] == "soft_story"

    def test_member_capacity_primary(self):
        member_interp = {
            "status": "NG", "weakest_link": {"ratio": 1.1, "member_id": 1},
            "ng_by_story": {1: 2}, "ng_by_type": {"column": 2},
        }
        drift_interp = {"soft_story_detected": False, "max_ratio": 0.8}
        result = _diagnose_failure("moderate", drift_interp, member_interp, None, _MockMulti())
        assert result["primary_cause"] == "member_capacity"

    def test_high_story_factor(self):
        multi = _MockMulti(stories=[5.0, 3.5])
        drift_interp = {"soft_story_detected": False, "max_ratio": 1.1}
        result = _diagnose_failure("moderate", drift_interp, None, None, multi)
        assert any("층고" in f for f in result["contributing_factors_ko"])


# ============================================================
# TestSuggestions
# ============================================================

class TestSuggestions:
    def test_safe_no_suggestions(self):
        result = _suggest_improvements("safe", None, None, None, _MockMulti())
        assert result == []

    def test_column_upgrade(self):
        diagnosis = {"primary_cause": "lateral_stiffness"}
        multi = _MockMulti(column_section="H-300x300")
        result = _suggest_improvements("moderate", diagnosis, None, None, multi)
        assert len(result) >= 1
        upgrade = [s for s in result if s["type"] == "section_upgrade" and s["target"] == "column"]
        assert len(upgrade) == 1
        assert upgrade[0]["recommended"] == "H-350x350"

    def test_severe_system_change(self):
        diagnosis = {"primary_cause": "lateral_stiffness"}
        multi = _MockMulti(column_section="H-300x300")
        result = _suggest_improvements("severe", diagnosis, None, None, multi)
        assert any(s["type"] == "system_change" for s in result)

    def test_section_not_in_series(self):
        """시리즈에 없는 단면 — 업그레이드 제안 없이 통과."""
        diagnosis = {"primary_cause": "lateral_stiffness"}
        multi = _MockMulti(column_section="H-999x999")
        result = _suggest_improvements("moderate", diagnosis, None, None, multi)
        # column upgrade not found, but no error
        col_upgrades = [s for s in result if s["target"] == "column"]
        assert len(col_upgrades) == 0


# ============================================================
# TestFindNextSection
# ============================================================

class TestFindNextSection:
    def test_found(self):
        assert _find_next_section("H-300x300", ["H-200x200", "H-300x300", "H-400x400"]) == "H-400x400"

    def test_last_in_series(self):
        assert _find_next_section("H-400x400", ["H-200x200", "H-300x300", "H-400x400"]) is None

    def test_not_in_series(self):
        assert _find_next_section("H-999x999", ["H-200x200", "H-300x300"]) is None

    def test_empty(self):
        assert _find_next_section("", ["H-200x200"]) is None


# ============================================================
# TestIntegration
# ============================================================

class TestIntegration:
    def test_full_ok(self):
        dc = _mock_design_check(
            drift_ratio=0.5, interaction_ratio=0.4,
            drift_checks=[
                {"story": 1, "direction": "X", "ratio": 0.5},
                {"story": 2, "direction": "X", "ratio": 0.4},
            ],
            member_list=[
                _mock_member(1, "column", 1, 0.4),
                _mock_member(2, "beam_x", 1, 0.3),
            ],
        )
        result = interpret_results(dc, _MockMulti())
        assert result["severity"] == "safe"
        assert "OK" in result["summary_en"] or "passed" in result["summary_en"]
        assert result["diagnosis"] is None
        assert result["suggestions"] == []

    def test_full_ng(self):
        dc = _mock_design_check(
            drift_ratio=1.17, interaction_ratio=1.05,
            drift_status="NG", member_status="NG",
            ng_stories=1, ng_members=2,
            drift_checks=[
                {"story": 1, "direction": "X", "ratio": 1.17},
                {"story": 2, "direction": "X", "ratio": 0.8},
            ],
            member_list=[
                _mock_member(1, "column", 1, 1.05, status="NG"),
                _mock_member(2, "column", 1, 1.02, status="NG"),
                _mock_member(3, "beam_x", 2, 0.6),
            ],
        )
        dc["drift_check"]["critical"] = {
            "story": 1, "direction": "X", "combo": "1.2DL+1.0EQX", "ratio": 1.17,
        }

        modal = {
            "modes": [{"mode": 1, "period_s": 0.8, "direction": "TRAN-X", "dominance_pct": 78}],
            "fundamental_periods": {"T1_x_s": 0.8, "T1_y_s": 0.5, "T1_rz_s": 0.3},
        }

        result = interpret_results(dc, _MockMulti(), modal_analysis=modal)
        assert result["severity"] == "moderate"
        assert result["diagnosis"] is not None
        assert result["diagnosis"]["primary_cause"] == "lateral_stiffness"
        assert len(result["suggestions"]) >= 1
        assert "부적합" in result["summary_ko"]
        assert len(result["findings"]) >= 4

    def test_empty_design_check(self):
        result = interpret_results({}, _MockMulti())
        assert result == {}
