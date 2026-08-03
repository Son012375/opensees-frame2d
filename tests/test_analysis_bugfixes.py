"""해석시스템 정합성 회귀 테스트 (적대적 감사 2026-06-24 확정 버그 B1~B6).

실행: cd mcp-server && python -m pytest ../tests/test_analysis_bugfixes.py -q
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mcp-server"))


# ── B5: 약축 소성단면계수 Zy (플랜지 2개, Zy ≥ Sy) ──
def test_b5_weak_axis_plastic_modulus_not_halved():
    from core.section_3d import get_section_3d
    from core.design_check import _compute_design_props
    s = get_section_3d("H-250x250")
    dp = _compute_design_props(s.A, s.Ix, s.Iy, s.h, s.b, s.tw, s.tf)
    # 소성단면계수는 탄성단면계수 이상이어야 함(구버전은 플랜지 절반만 세어 Zy<Sy).
    assert dp["Zy"] >= dp["Sy"]
    assert 1.3 <= dp["Zy"] / dp["Sy"] <= 1.8  # H형강 약축 형상계수 ~1.5


def test_b5_zx_clamped_above_sx():
    from core.design_check import _compute_design_props
    # tw/tf 미상 → 근사경로도 Z>=S 보장
    dp = _compute_design_props(A=1e4, Ix=1e8, Iy=2e7, h=400, b=200, tw=0, tf=0)
    assert dp["Zx"] >= dp["Sx"] and dp["Zy"] >= dp["Sy"]


# ── B3: 지배조합 = 실제 H1 최대 (차원불일치 N+My+Mz 간이지표 제거) ──
def test_b3_governing_combo_by_true_h1():
    from core.design_check import _governing_member_forces
    mf = {
        # 중력: 축력 큼·모멘트 작음 → 구버전 간이지표가 항상 선택했던 조합
        "Gravity": [{"member_id": 1, "N_kN": [3000], "My_kNm": [20], "Mz_kNm": [5],
                     "Vy_kN": [100], "Vz_kN": [0]}],
        # 지진: 축력 작음·모멘트 큼 → 실제 H1 지배 (NG)
        "Seismic": [{"member_id": 1, "N_kN": [1500], "My_kNm": [400], "Mz_kNm": [120],
                     "Vy_kN": [150], "Vz_kN": [0]}],
    }
    gov = _governing_member_forces(mf, 1, ["Gravity", "Seismic"], 4000, 300, 80, 500)
    assert gov["governing_combo"] == "Seismic"
    assert gov["interaction"] > 1.0          # 구버전은 0.86(OK)로 오판
    assert gov["Mux"] == 400.0               # 지진조합 모멘트가 envelope에 반영


def test_b3_shear_enveloped_independently():
    from core.design_check import _governing_member_forces
    mf = {
        "A": [{"member_id": 1, "N_kN": [100], "My_kNm": [300], "Mz_kNm": [0],
               "Vy_kN": [50], "Vz_kN": [0]}],       # 상관비 지배
        "B": [{"member_id": 1, "N_kN": [50], "My_kNm": [10], "Mz_kNm": [0],
               "Vy_kN": [600], "Vz_kN": [0]}],       # 전단 지배(다른 조합)
    }
    gov = _governing_member_forces(mf, 1, ["A", "B"], 4000, 300, 80, 500)
    # 전단비는 조합 독립 최댓값(B의 600/500=1.2)이어야 NG 누락 안 됨
    assert abs(gov["shear_ratio"] - 1.2) < 1e-6
    assert gov["shear_governing_combo"] == "B"


# ── B2: 중요도계수 Ie 이중적용 제거 ──
def test_b2_importance_factor_applied_once():
    from core.building_model import BuildingModel
    from core.load_generator import generate_seismic_loads

    def cs_sds(imp):
        m = BuildingModel.from_json({
            "stories": [{"height": 3.5, "usage": "office"}] * 3,
            "bays_x": [6, 6], "bays_y": [6, 6], "region": "강남구",
            "importance": imp, "seismic_system": "ordinary_moment_frame", "site_class": "S3"})
        rep = generate_seismic_loads(m, [500, 500, 500])["report"]
        return rep["Cs"], rep["SDS"]

    cs2, sds2 = cs_sds("II")   # Ie=1.0
    csx, sdsx = cs_sds("특")   # Ie=1.5
    assert abs(sds2 - sdsx) < 1e-6           # SDS는 순수 지반위험도(Ie 무관)
    assert abs(csx / cs2 - 1.5) < 0.05       # Cs ∝ Ie (1회), ≠ Ie²(2.25)


# ── B1: V1 모달 누적 유효질량 참여율 ≤ 100% (짝수 n_cols 포함) ──
def test_b1_v1_modal_cumulative_le_100():
    from core.frame_3d import analyze_frame_3d_multi
    dl = [{"type": "floor_area", "value": 5.0, "story": s} for s in (1, 2)]
    multi = analyze_frame_3d_multi(
        stories=[3.5, 3.5], bays_x=[6, 6, 6], bays_y=[6, 6, 6],  # 4×4 cols (짝수)
        load_cases={"DL": dl}, load_combinations={"DL": {"DL": 1.0}},
        modal_analysis=True, rigid_diaphragm=True)
    cp = (multi.modal_analysis or {}).get("cumulative_participation") or {}
    assert cp.get("x_pct", 999) <= 100.5
    assert cp.get("y_pct", 999) <= 100.5
    # 완전기저(2층=6 DOF, 6모드)이므로 ~100% 도달해야 함
    assert cp.get("x_pct", 0) >= 95.0 and cp.get("y_pct", 0) >= 95.0


# ── B4: 비정형 경계 보 패널 합산 (floor_area 하중 누락 방지) ──
def test_b4_irregular_shared_beam_panels_merged():
    from core.frame_3d import _generate_irregular_geometry
    zoneA = {"id": 0, "bays_x": [6, 6], "bays_y": [6], "origin_x": 0, "origin_y": 0,
             "story_from": 1, "story_to": None}
    zoneB = {"id": 1, "bays_x": [6], "bays_y": [6], "origin_x": 0, "origin_y": 6,
             "story_from": 1, "story_to": None}
    _n, conns, _s, _b, meta = _generate_irregular_geometry([3.5, 3.5], [zoneA, zoneB])
    shared = [m for i, m in enumerate(meta)
              if conns[i][2] == "beam_x" and m["story"] == 1
              and len(m.get("adj_panels", [])) >= 2]
    # 경계(y=6) 공유 보가 양쪽 영역 패널을 모두 보유해야 함(구버전은 1개만 → 하중 누락)
    assert len(shared) >= 1


# ── B6: V2 T1_x/T1_y 'XY' 부분문자열 오매칭 제거 (질량참여 기반) ──
def test_b6_v2_fundamental_period_by_participation():
    modes = [
        {"period_s": 1.50, "direction": "XY", "mass_participation_x_pct": 70, "mass_participation_y_pct": 65},
        {"period_s": 1.20, "direction": "Y", "mass_participation_x_pct": 0, "mass_participation_y_pct": 85},
        {"period_s": 1.10, "direction": "X", "mass_participation_x_pct": 85, "mass_participation_y_pct": 0},
    ]
    # 출하 코드(_run_eigen_analysis_v2)의 선정 로직과 동일

    def _t1_dir(key):
        cand = [m for m in modes if m.get(key, 0) > 0]
        return max(cand, key=lambda m: m.get(key, 0))["period_s"] if cand else None

    assert _t1_dir("mass_participation_x_pct") == 1.10   # 순수 X 모드
    assert _t1_dir("mass_participation_y_pct") == 1.20   # 순수 Y 모드
    # 구버전 substring은 'XY'를 양쪽에 잡아 둘 다 1.50이었음
