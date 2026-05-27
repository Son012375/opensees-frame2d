"""Design Check 모듈 테스트.

KDS 41 17 00 층간변위 검토 + KDS 41 31 00 부재 강도 검토.

실행: cd mcp-server && python -m pytest ../tests/test_design_check.py -v
"""
import math
import os
import sys
import types

# mcp-server를 import path에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mcp-server"))

from core.design_check import (
    _compute_design_props,
    _compression_capacity,
    _bending_capacity,
    _shear_capacity,
    _interaction_H1,
    check_story_drifts,
    check_member_strengths,
    run_design_check,
    DRIFT_LIMITS,
    PHI,
)


# ============================================================
# 테스트 헬퍼: Mock 데이터
# ============================================================

class MockCaseResult:
    """Frame3DCaseResult mock."""
    def __init__(self, story_drifts=None, element_forces=None):
        self.story_drifts = story_drifts or []
        self.element_forces = element_forces or []
        self.max_drift_x = 0.0
        self.max_drift_y = 0.0


class MockMultiResult:
    """Frame3DMultiCaseResult mock."""
    def __init__(self):
        self.stories = [4.0, 3.5, 3.5]
        self.bays_x = [8.0, 8.0]
        self.bays_y = [8.0]
        self.num_stories = 3
        self.fy_MPa = 275.0
        self.E_MPa = 205000.0

        # H-300x300x10x15 (column)
        self.column_section = "H-300x300"
        self.column_A_mm2 = 11980.0
        self.column_Ix_mm4 = 20400e4
        self.column_Iy_mm4 = 6750e4
        self.column_J_mm4 = 128e4
        self.column_h_mm = 300.0
        self.column_b_mm = 300.0
        self.column_tw_mm = 10.0
        self.column_tf_mm = 15.0

        # H-400x200x8x13 (beam_x, beam_y)
        self.beam_x_section = "H-400x200"
        self.beam_x_A_mm2 = 8412.0
        self.beam_x_Ix_mm4 = 23700e4
        self.beam_x_Iy_mm4 = 1740e4
        self.beam_x_J_mm4 = 66e4
        self.beam_x_h_mm = 400.0
        self.beam_x_b_mm = 200.0
        self.beam_x_tw_mm = 8.0
        self.beam_x_tf_mm = 13.0

        self.beam_y_section = "H-400x200"
        self.beam_y_A_mm2 = 8412.0
        self.beam_y_Ix_mm4 = 23700e4
        self.beam_y_Iy_mm4 = 1740e4
        self.beam_y_J_mm4 = 66e4
        self.beam_y_h_mm = 400.0
        self.beam_y_b_mm = 200.0
        self.beam_y_tw_mm = 8.0
        self.beam_y_tf_mm = 13.0

        self.member_info = []
        self.member_forces = {}
        self.combo_results = {}
        self.case_results = {}
        self.analysis_metadata = {}


def _make_member_forces_entry(member_id, mtype, section, length_m,
                               N_vals, Vy_vals, Vz_vals,
                               My_vals, Mz_vals, T_vals=None):
    """단일 부재의 member_forces 항목 생성."""
    n = len(N_vals)
    return {
        "member_id": member_id,
        "type": mtype,
        "ni": 1, "nj": 2,
        "length_m": length_m,
        "section": section,
        "N_kN": N_vals,
        "Vy_kN": Vy_vals,
        "Vz_kN": Vz_vals,
        "T_kNm": T_vals or [0.0] * n,
        "My_kNm": My_vals,
        "Mz_kNm": Mz_vals,
    }


# ============================================================
# T1-T2: 단면 물성 계산
# ============================================================

def test_T1_design_props_with_tw_tf():
    """tw/tf 있는 경우 Sx, Zx, Aw 계산 정확성."""
    # H-300x300x10x15
    props = _compute_design_props(
        A=11980, Ix=20400e4, Iy=6750e4,
        h=300, b=300, tw=10, tf=15,
    )

    # Sx = Ix / (h/2) = 20400e4 / 150 = 1,360,000 mm³
    assert abs(props["Sx"] - 1360000.0) < 1.0

    # rx = sqrt(Ix / A) = sqrt(20400e4 / 11980) ≈ 130.5 mm
    assert abs(props["rx"] - math.sqrt(20400e4 / 11980)) < 0.1

    # Zx = b*tf*(h-tf) + tw*hw²/4 = 300*15*285 + 10*270²/4
    hw = 300 - 2 * 15  # = 270
    expected_Zx = 300 * 15 * (300 - 15) + 10 * hw**2 / 4
    assert abs(props["Zx"] - expected_Zx) < 1.0

    # Aw = hw * tw = 270 * 10 = 2700
    assert abs(props["Aw"] - 2700.0) < 1.0


def test_T2_design_props_without_tw_tf():
    """tw/tf 없는 경우 (=0) 보수적 근사 적용."""
    props = _compute_design_props(
        A=11980, Ix=20400e4, Iy=6750e4,
        h=300, b=300, tw=0, tf=0,
    )

    Sx = 20400e4 / 150
    assert abs(props["Zx"] - Sx * 1.12) < 1.0  # shape factor 1.12
    assert abs(props["Aw"] - 11980 / 3) < 1.0   # A/3 근사


# ============================================================
# T3-T5: 강도 계산 함수
# ============================================================

def test_T3_compression_capacity():
    """AISC E3: KL/r=60, fy=275 MPa, E=205000 MPa."""
    A = 11980.0  # mm²
    fy = 275.0
    E = 205000.0
    r = 130.5     # mm (approximate rx for H-300x300)
    KL = 60 * r   # KL/r = 60

    phiPn = _compression_capacity(fy, E, A, r, KL)

    # 수동 계산:
    Fe = math.pi**2 * E / 60**2  # = 562.2 MPa
    ratio = fy / Fe               # = 0.489
    Fcr = (0.658 ** ratio) * fy   # ≈ 223.8 MPa
    expected = 0.9 * Fcr * A / 1000

    assert abs(phiPn - expected) < 0.1, f"φPn={phiPn:.1f} vs expected={expected:.1f}"
    assert phiPn > 0


def test_T4_bending_capacity():
    """AISC F2: compact section, Mn = Fy × Zx."""
    fy = 275.0
    Zx = 1_464_750.0  # mm³ (H-300x300 approximate)

    phiMn = _bending_capacity(fy, Zx)

    expected = 0.9 * 275 * 1_464_750 / 1e6
    assert abs(phiMn - expected) < 0.01


def test_T5_shear_capacity():
    """AISC G2: Vn = 0.6 × Fy × Aw."""
    fy = 275.0
    Aw = 2700.0  # mm² (270 × 10)

    phiVn = _shear_capacity(fy, Aw)

    expected = 0.9 * 0.6 * 275 * 2700 / 1000
    assert abs(phiVn - expected) < 0.01


# ============================================================
# T6-T7: H1 Interaction
# ============================================================

def test_T6_interaction_H1_1a():
    """H1-1a: Pu/φPn ≥ 0.2 경우."""
    Pu = 500.0
    phiPn = 1000.0  # Pu/phiPn = 0.5 ≥ 0.2
    Mux = 100.0
    phiMnx = 400.0
    Muy = 50.0
    phiMny = 200.0

    ratio, formula = _interaction_H1(Pu, phiPn, Mux, phiMnx, Muy, phiMny)

    # 0.5 + 8/9 * (100/400 + 50/200) = 0.5 + 8/9 * 0.5 = 0.5 + 0.444 = 0.944
    expected = 0.5 + (8.0 / 9.0) * (0.25 + 0.25)
    assert abs(ratio - expected) < 0.001
    assert formula == "H1-1a"


def test_T7_interaction_H1_1b():
    """H1-1b: Pu/φPn < 0.2 경우."""
    Pu = 100.0
    phiPn = 1000.0  # Pu/phiPn = 0.1 < 0.2
    Mux = 100.0
    phiMnx = 400.0
    Muy = 50.0
    phiMny = 200.0

    ratio, formula = _interaction_H1(Pu, phiPn, Mux, phiMnx, Muy, phiMny)

    # 0.1/2 + (100/400 + 50/200) = 0.05 + 0.5 = 0.55
    expected = 0.05 + (0.25 + 0.25)
    assert abs(ratio - expected) < 0.001
    assert formula == "H1-1b"


# ============================================================
# T8-T10: 층간변위 검토
# ============================================================

def test_T8_drift_check_eq_identification():
    """지진 조합만 검토되는지 확인."""
    combo_results = {
        "1.4DL": MockCaseResult(story_drifts=[
            {"story": 1, "height_m": 4.0, "drift_x": 0.001, "drift_y": 0.0005},
        ]),
        "1.2DL+1.0LL+1.0EQX": MockCaseResult(story_drifts=[
            {"story": 1, "height_m": 4.0, "drift_x": 0.005, "drift_y": 0.002},
        ]),
    }

    result = check_story_drifts(combo_results, [4.0], Cd=3.0, IE=1.0, importance="II")

    # 1.4DL은 EQ가 아니므로 제외 → EQ 조합만 검토
    assert all("EQ" in c["combo"] for c in result["checks"])
    assert len(result["checks"]) == 2  # X, Y directions for 1 EQ combo


def test_T9_drift_check_cd_ie_amplification():
    """Cd/IE 증폭 계산 정확성."""
    combo_results = {
        "1.2DL+1.0EQX": MockCaseResult(story_drifts=[
            {"story": 1, "height_m": 4.0, "drift_x": 0.002, "drift_y": 0.001},
        ]),
    }

    result = check_story_drifts(combo_results, [4.0], Cd=5.5, IE=1.2, importance="II")

    # inelastic = 5.5 * 0.002 / 1.2 = 0.009167
    x_check = [c for c in result["checks"] if c["direction"] == "X"][0]
    expected = 5.5 * 0.002 / 1.2
    assert abs(x_check["inelastic_drift"] - round(expected, 6)) < 1e-6


def test_T10_drift_check_importance_grades():
    """중요도 등급별 허용치 확인."""
    for grade, limit in DRIFT_LIMITS.items():
        combo_results = {
            "1.2DL+1.0EQX": MockCaseResult(story_drifts=[
                {"story": 1, "height_m": 4.0, "drift_x": 0.001, "drift_y": 0.0},
            ]),
        }
        result = check_story_drifts(combo_results, [4.0], Cd=3.0, IE=1.0, importance=grade)
        assert result["allowable"] == limit

    # NG case: drift > limit
    combo_results = {
        "1.2DL+1.0EQX": MockCaseResult(story_drifts=[
            {"story": 1, "height_m": 4.0, "drift_x": 0.010, "drift_y": 0.0},
        ]),
    }
    # Cd=3.0, IE=1.0 → inelastic = 0.030 > 0.020 → NG
    result = check_story_drifts(combo_results, [4.0], Cd=3.0, IE=1.0, importance="II")
    assert result["status"] == "NG"


# ============================================================
# T11-T13: 부재 강도 검토
# ============================================================

def test_T11_member_check_ok():
    """정상 단면 → 모든 부재 OK."""
    multi = MockMultiResult()

    # 3개 부재: 1 column + 1 beam_x + 1 beam_y (경미한 하중)
    multi.member_info = [
        {"member_id": 1, "type": "column", "length_m": 4.0,
         "section": "H-300x300", "element_ids": [1, 2, 3, 4]},
        {"member_id": 2, "type": "beam_x", "length_m": 8.0,
         "section": "H-400x200", "element_ids": [5, 6, 7, 8]},
        {"member_id": 3, "type": "beam_y", "length_m": 8.0,
         "section": "H-400x200", "element_ids": [9, 10, 11, 12]},
    ]

    # 경미한 하중 (low forces → OK)
    pts = 5  # num_elements + 1
    multi.member_forces = {
        "1.2DL+1.6LL": [
            _make_member_forces_entry(1, "column", "H-300x300", 4.0,
                                       [-200]*pts, [30]*pts, [20]*pts,
                                       [50]*pts, [30]*pts),
            _make_member_forces_entry(2, "beam_x", "H-400x200", 8.0,
                                       [0]*pts, [40]*pts, [0]*pts,
                                       [80]*pts, [0]*pts),
            _make_member_forces_entry(3, "beam_y", "H-400x200", 8.0,
                                       [0]*pts, [0]*pts, [40]*pts,
                                       [80]*pts, [0]*pts),
        ],
    }
    multi.combo_results = {"1.2DL+1.6LL": MockCaseResult()}

    result = check_member_strengths(multi, 275.0, 205000.0)

    assert result["status"] == "OK"
    assert result["summary"]["ng"] == 0
    assert len(result["members"]) == 3


def test_T12_member_check_ng():
    """약한 단면 (H-200x100) → NG 발생."""
    multi = MockMultiResult()

    # H-200x100 (very small section) for column
    multi.column_A_mm2 = 2716.0
    multi.column_Ix_mm4 = 1840e4
    multi.column_Iy_mm4 = 134e4
    multi.column_h_mm = 200.0
    multi.column_b_mm = 100.0
    multi.column_tw_mm = 5.5
    multi.column_tf_mm = 8.0

    multi.member_info = [
        {"member_id": 1, "type": "column", "length_m": 4.0,
         "section": "H-200x100", "element_ids": [1, 2, 3, 4]},
    ]

    # 매우 큰 하중 → NG
    pts = 5
    multi.member_forces = {
        "1.2DL+1.0EQX": [
            _make_member_forces_entry(1, "column", "H-200x100", 4.0,
                                       [-800]*pts, [100]*pts, [80]*pts,
                                       [200]*pts, [150]*pts),
        ],
    }
    multi.combo_results = {"1.2DL+1.0EQX": MockCaseResult()}

    result = check_member_strengths(multi, 275.0, 205000.0)

    assert result["status"] == "NG"
    assert result["summary"]["ng"] >= 1
    assert result["members"][0]["ratios"]["interaction"] > 1.0


def test_T12b_member_check_per_member_section(monkeypatch):
    """동일 mtype에서 서로 다른 section을 가진 부재가 각자의 단면 기준으로
    capacity를 받는지 확인 (Phase B 단면 변경 후 회귀 방지).

    이전엔 multi.{mtype}_section을 첫 부재 단면으로 일괄 적용해서
    "5번 부재만 H-200x200으로 변경" 같은 시나리오에서 변경 부재의 ratio가
    원래 단면(예: H-400x200) capacity로 계산되는 버그가 있었음.
    """
    # get_section_3d를 가짜 단면 룩업으로 교체 — 두 단면 모두 명시적 props
    fake_sections = {
        "H-400x200": types.SimpleNamespace(
            name="H-400x200",
            A=8412.0, Ix=23700e4, Iy=1740e4, J=66e4,
            h=400.0, b=200.0, tw=8.0, tf=13.0,
        ),
        "H-200x100": types.SimpleNamespace(
            name="H-200x100",
            A=2716.0, Ix=1840e4, Iy=134e4, J=10e4,
            h=200.0, b=100.0, tw=5.5, tf=8.0,
        ),
    }
    import core.section_3d as section_3d_mod
    monkeypatch.setattr(
        section_3d_mod, "get_section_3d",
        lambda name: fake_sections.get(name) or fake_sections["H-400x200"],
    )

    multi = MockMultiResult()
    # multi_result의 mtype 기본 단면은 H-400x200 (큰 단면). 두 번째 부재만
    # 약한 H-200x100로 교체 — 동일 하중에서 capacity가 다르고, 따라서
    # interaction ratio도 명확히 달라야 함.
    multi.member_info = [
        {"member_id": 1, "type": "beam_x", "length_m": 6.0,
         "section": "H-400x200", "element_ids": [1, 2, 3, 4]},
        {"member_id": 2, "type": "beam_x", "length_m": 6.0,
         "section": "H-200x100", "element_ids": [5, 6, 7, 8]},
    ]

    pts = 5
    # 두 부재에 동일한 하중 인가 (단면 차이만 ratio에 반영되도록)
    forces = [
        _make_member_forces_entry(1, "beam_x", "H-400x200", 6.0,
                                   [0]*pts, [60]*pts, [0]*pts,
                                   [120]*pts, [0]*pts),
        _make_member_forces_entry(2, "beam_x", "H-200x100", 6.0,
                                   [0]*pts, [60]*pts, [0]*pts,
                                   [120]*pts, [0]*pts),
    ]
    multi.member_forces = {"1.2DL+1.6LL": forces}
    multi.combo_results = {"1.2DL+1.6LL": MockCaseResult()}

    result = check_member_strengths(multi, 275.0, 205000.0)

    m_by_id = {m["member_id"]: m for m in result["members"]}
    m1 = m_by_id[1]  # H-400x200
    m2 = m_by_id[2]  # H-200x100

    # capacity가 단면별로 다르게 계산되었는지
    assert m1["capacity"]["phiMnx_kNm"] > m2["capacity"]["phiMnx_kNm"], (
        f"큰 단면(H-400x200) capacity가 작은 단면(H-200x100)보다 커야 함: "
        f"{m1['capacity']} vs {m2['capacity']}"
    )
    # 따라서 같은 하중에서 작은 단면의 interaction ratio가 더 커야 함
    assert m2["ratios"]["interaction"] > m1["ratios"]["interaction"], (
        f"같은 하중·작은 단면의 ratio가 더 커야 함: "
        f"m1={m1['ratios']}, m2={m2['ratios']}"
    )
    # 결과 객체에도 부재별 실제 단면이 보고되어야 함
    assert m1["section"] == "H-400x200"
    assert m2["section"] == "H-200x100"


def test_T13_member_check_envelope():
    """다중 조합에서 envelope 힘 추출."""
    multi = MockMultiResult()

    multi.member_info = [
        {"member_id": 1, "type": "column", "length_m": 4.0,
         "section": "H-300x300", "element_ids": [1, 2, 3, 4]},
    ]

    pts = 5
    # Combo 1: N=100, My=50
    # Combo 2: N=300, My=80 (larger → should govern)
    multi.member_forces = {
        "1.2DL+1.6LL": [
            _make_member_forces_entry(1, "column", "H-300x300", 4.0,
                                       [-100]*pts, [20]*pts, [10]*pts,
                                       [50]*pts, [20]*pts),
        ],
        "1.2DL+1.0EQX": [
            _make_member_forces_entry(1, "column", "H-300x300", 4.0,
                                       [-300]*pts, [60]*pts, [40]*pts,
                                       [80]*pts, [50]*pts),
        ],
    }
    multi.combo_results = {
        "1.2DL+1.6LL": MockCaseResult(),
        "1.2DL+1.0EQX": MockCaseResult(),
    }

    result = check_member_strengths(multi, 275.0, 205000.0)

    # 지배 조합은 EQ (더 큰 하중)
    m = result["members"][0]
    assert m["governing_combo"] == "1.2DL+1.0EQX"
    assert m["demand"]["Pu"] == 300.0
    assert m["demand"]["Mux"] == 80.0


# ============================================================
# T14-T16: 통합 설계 검토
# ============================================================

def test_T14_run_design_check_ok():
    """통합 검토: drift OK + member OK → overall OK."""
    multi = MockMultiResult()

    multi.member_info = [
        {"member_id": 1, "type": "column", "length_m": 4.0,
         "section": "H-300x300", "element_ids": [1, 2, 3, 4]},
    ]

    pts = 5
    multi.member_forces = {
        "1.2DL+1.0EQX": [
            _make_member_forces_entry(1, "column", "H-300x300", 4.0,
                                       [-200]*pts, [30]*pts, [20]*pts,
                                       [50]*pts, [30]*pts),
        ],
    }
    multi.combo_results = {
        "1.2DL+1.0EQX": MockCaseResult(story_drifts=[
            {"story": 1, "height_m": 4.0, "drift_x": 0.001, "drift_y": 0.0005},
            {"story": 2, "height_m": 3.5, "drift_x": 0.002, "drift_y": 0.001},
            {"story": 3, "height_m": 3.5, "drift_x": 0.0015, "drift_y": 0.0008},
        ]),
    }

    seismic_report = {"Cd": 3.0, "IE": 1.0}

    class MockModel:
        importance = "II"

    result = run_design_check(multi, MockModel(), seismic_report)

    assert result["overall_status"] == "OK"
    assert result["drift_check"]["status"] == "OK"
    assert result["member_check"]["status"] == "OK"
    assert len(result["critical_issues"]) == 0


def test_T15_run_design_check_ng():
    """통합 검토: drift NG → overall NG."""
    multi = MockMultiResult()
    multi.member_info = []

    multi.combo_results = {
        "1.2DL+1.0EQX": MockCaseResult(story_drifts=[
            {"story": 1, "height_m": 4.0, "drift_x": 0.010, "drift_y": 0.005},
        ]),
    }

    seismic_report = {"Cd": 5.5, "IE": 1.0}
    # inelastic = 5.5 * 0.010 = 0.055 >> 0.020 → NG

    class MockModel:
        importance = "II"

    result = run_design_check(multi, MockModel(), seismic_report)

    assert result["overall_status"] == "NG"
    assert result["drift_check"]["status"] == "NG"
    assert len(result["critical_issues"]) > 0
    assert result["critical_issues"][0]["type"] == "drift_exceeded"


def test_T16_run_design_check_no_seismic():
    """seismic_report 없으면 drift check 건너뜀."""
    multi = MockMultiResult()
    multi.member_info = [
        {"member_id": 1, "type": "column", "length_m": 4.0,
         "section": "H-300x300", "element_ids": [1, 2, 3, 4]},
    ]

    pts = 5
    multi.member_forces = {
        "1.2DL+1.6LL": [
            _make_member_forces_entry(1, "column", "H-300x300", 4.0,
                                       [-100]*pts, [20]*pts, [10]*pts,
                                       [30]*pts, [15]*pts),
        ],
    }
    multi.combo_results = {"1.2DL+1.6LL": MockCaseResult()}

    result = run_design_check(multi, None, None)

    assert result["drift_check"] is None
    assert result["member_check"] is not None
    assert result["overall_status"] == "OK"
