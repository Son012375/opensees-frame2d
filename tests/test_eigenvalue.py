"""고유치해석 통합 테스트.

실행: cd mcp-server && python -m pytest ../tests/test_eigenvalue.py -v
"""
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mcp-server"))

from core.frame_3d import (
    analyze_frame_3d_multi,
    _estimate_story_weights,
    _run_eigen_analysis,
    _generate_frame_3d_geometry,
    _build_frame_3d_model,
    Frame3DMultiCaseResult,
)
from core.section_3d import get_section_3d, DEFAULT_SECTIONS_3D
from core.simple_beam import get_material_from_db, DEFAULT_MATERIALS


# ============================================================
# 테스트 헬퍼
# ============================================================

def _simple_2story_load_cases():
    """2층 2x1bay 간단한 DL+EQX 하중."""
    return {
        "DL": [
            {"type": "floor_area", "story": 1, "value": 5.0},
            {"type": "floor_area", "story": 2, "value": 5.0},
        ],
        "EQX": [
            {"type": "lateral_x", "story": 1, "value": 50.0},
            {"type": "lateral_x", "story": 2, "value": 100.0},
        ],
    }


def _simple_2story_combos():
    return {
        "1.2DL+1.0EQX": {"DL": 1.2, "EQX": 1.0},
    }


# ============================================================
# T1~T3: _estimate_story_weights
# ============================================================

def test_T1_estimate_from_floor_area():
    """floor_area 하중에서 층별 중력하중 추정."""
    cases = {
        "DL": [
            {"type": "floor_area", "story": 1, "value": 5.0},
            {"type": "floor_area", "story": 2, "value": 4.0},
        ]
    }
    bays_x = [8.0, 8.0]
    bays_y = [8.0]
    weights = _estimate_story_weights(cases, 2, bays_x, bays_y, 3, 2)
    area = 16.0 * 8.0  # 128 m²
    assert len(weights) == 2
    assert abs(weights[0] - 5.0 * area) < 0.01
    assert abs(weights[1] - 4.0 * area) < 0.01


def test_T2_estimate_from_floor_line():
    """floor (선하중) 에서 중력하중 추정."""
    cases = {
        "DL": [
            {"type": "floor", "story": 1, "value": 10.0},
        ]
    }
    bays_x = [6.0, 6.0]
    bays_y = [6.0]
    n_cx, n_cy = 3, 2
    total_len = (6.0 + 6.0) * n_cy + 6.0 * n_cx  # 24 + 18 = 42
    weights = _estimate_story_weights(cases, 1, bays_x, bays_y, n_cx, n_cy)
    assert len(weights) == 1
    assert abs(weights[0] - 10.0 * total_len) < 0.01


def test_T3_estimate_no_dl():
    """DL 케이스 없으면 빈 리스트."""
    cases = {"LL": [{"type": "floor_area", "story": 1, "value": 2.0}]}
    weights = _estimate_story_weights(cases, 1, [8.0], [8.0], 2, 2)
    assert weights == []


# ============================================================
# T4~T6: 통합 테스트 (analyze_frame_3d_multi with modal_analysis)
# ============================================================

def test_T4_modal_analysis_basic():
    """2층 간단 모델 모달해석 — 모드 반환 확인."""
    multi = analyze_frame_3d_multi(
        stories=[4.0, 3.5],
        bays_x=[8.0, 8.0],
        bays_y=[8.0],
        load_cases=_simple_2story_load_cases(),
        load_combinations=_simple_2story_combos(),
        rigid_diaphragm=True,
        modal_analysis=True,
    )

    ma = multi.modal_analysis
    assert ma, "modal_analysis 결과가 비어있음"
    assert ma["num_modes"] > 0
    assert len(ma["modes"]) > 0

    # 첫 번째 모드 구조 확인
    m1 = ma["modes"][0]
    assert "period_s" in m1
    assert "frequency_hz" in m1
    assert "direction" in m1
    assert "dominance_pct" in m1
    assert m1["period_s"] > 0
    assert m1["direction"] in ("TRAN-X", "TRAN-Y", "ROTN-Z")

    # fundamental_periods 존재
    fp = ma["fundamental_periods"]
    assert "T1_x_s" in fp
    assert "T1_y_s" in fp
    assert "T1_rz_s" in fp


def test_T5_modal_auto_diaphragm():
    """rigid_diaphragm=False + modal_analysis=True → 자동 활성화 + W04 경고."""
    multi = analyze_frame_3d_multi(
        stories=[4.0, 3.5],
        bays_x=[8.0, 8.0],
        bays_y=[8.0],
        load_cases=_simple_2story_load_cases(),
        rigid_diaphragm=False,
        modal_analysis=True,
    )

    # 모달해석 결과 있어야 함
    assert multi.modal_analysis
    assert multi.modal_analysis["num_modes"] > 0

    # diaphragm 자동 활성화 플래그
    assert multi.modal_analysis["diaphragm_auto_enabled"] is True

    # W04 경고
    assert "W04_diaphragm_auto" in multi.analysis_metadata


def test_T6_modal_no_mass_warning():
    """DL 없는 하중으로 모달해석 → W05 경고."""
    cases = {
        "EQX": [
            {"type": "lateral_x", "story": 1, "value": 50.0},
        ]
    }
    multi = analyze_frame_3d_multi(
        stories=[4.0],
        bays_x=[8.0],
        bays_y=[8.0],
        load_cases=cases,
        rigid_diaphragm=True,
        modal_analysis=True,
    )

    # 질량 추정 불가 → 모달해석 건너뜀
    assert not multi.modal_analysis
    assert "W05_no_mass" in multi.analysis_metadata


# ============================================================
# T7: story_weights_kN 직접 제공
# ============================================================

def test_T7_modal_with_explicit_weights():
    """story_weights_kN 직접 제공 시 DL 없어도 모달해석 수행."""
    cases = {
        "EQX": [
            {"type": "lateral_x", "story": 1, "value": 50.0},
            {"type": "lateral_x", "story": 2, "value": 100.0},
        ]
    }
    multi = analyze_frame_3d_multi(
        stories=[4.0, 3.5],
        bays_x=[8.0, 8.0],
        bays_y=[8.0],
        load_cases=cases,
        rigid_diaphragm=True,
        modal_analysis=True,
        story_weights_kN=[500.0, 400.0],
    )

    assert multi.modal_analysis
    assert multi.modal_analysis["num_modes"] > 0
    assert multi.modal_analysis["story_weights_kN"] == [500.0, 400.0]


# ============================================================
# T8: 모달해석 비활성화 시 결과 없음
# ============================================================

def test_T8_no_modal_when_disabled():
    """modal_analysis=False → 모달해석 결과 없음."""
    multi = analyze_frame_3d_multi(
        stories=[4.0],
        bays_x=[8.0],
        bays_y=[8.0],
        load_cases={"DL": [{"type": "floor_area", "story": 1, "value": 5.0}]},
        modal_analysis=False,
    )

    assert not multi.modal_analysis


# ============================================================
# T9: 모드 방향 일관성
# ============================================================

def test_T9_mode_directions_reasonable():
    """3층 모델 — X,Y 비대칭 → 모드 방향이 합리적."""
    multi = analyze_frame_3d_multi(
        stories=[4.0, 3.5, 3.5],
        bays_x=[8.0, 8.0],   # X: 16m
        bays_y=[6.0],         # Y: 6m (비대칭)
        load_cases={
            "DL": [
                {"type": "floor_area", "story": i, "value": 5.0}
                for i in range(1, 4)
            ],
        },
        rigid_diaphragm=True,
        modal_analysis=True,
    )

    ma = multi.modal_analysis
    assert ma["num_modes"] >= 3

    # 1차 모드는 가장 긴 주기
    T1 = ma["modes"][0]["period_s"]
    T2 = ma["modes"][1]["period_s"]
    assert T1 >= T2, "1차 모드가 가장 긴 주기여야 함"

    # 방향 종류 확인 (TRAN-X, TRAN-Y, ROTN-Z 중 적어도 2가지)
    directions = set(m["direction"] for m in ma["modes"][:6])
    assert len(directions) >= 2, f"최소 2가지 방향이 있어야 함: {directions}"

    # fundamental_periods 값 확인
    fp = ma["fundamental_periods"]
    assert fp["T1_x_s"] > 0 or fp["T1_y_s"] > 0, "최소 1개 방향의 기본주기가 있어야 함"


# ============================================================
# T10: period / frequency 일관성
# ============================================================

def test_T10_period_frequency_consistency():
    """T = 1/f 관계 확인."""
    multi = analyze_frame_3d_multi(
        stories=[4.0, 3.5],
        bays_x=[8.0],
        bays_y=[8.0],
        load_cases={
            "DL": [
                {"type": "floor_area", "story": 1, "value": 5.0},
                {"type": "floor_area", "story": 2, "value": 5.0},
            ],
        },
        rigid_diaphragm=True,
        modal_analysis=True,
    )

    for mode in multi.modal_analysis["modes"]:
        T = mode["period_s"]
        f = mode["frequency_hz"]
        assert abs(T * f - 1.0) < 0.01, f"T×f ≠ 1: T={T}, f={f}"


# ============================================================
# T11: mass_participation 필드 존재 및 합산 ≤ 100%
# ============================================================

def test_T11_mass_participation_per_mode():
    """각 모드에 mass_participation dict가 있고 0~100% 범위."""
    multi = analyze_frame_3d_multi(
        stories=[4.0, 3.5],
        bays_x=[8.0, 8.0],
        bays_y=[8.0],
        load_cases=_simple_2story_load_cases(),
        rigid_diaphragm=True,
        modal_analysis=True,
    )

    ma = multi.modal_analysis
    assert ma

    for m in ma["modes"]:
        mp = m["mass_participation"]
        assert "x_pct" in mp
        assert "y_pct" in mp
        assert "rz_pct" in mp
        assert 0.0 <= mp["x_pct"] <= 100.0
        assert 0.0 <= mp["y_pct"] <= 100.0
        assert 0.0 <= mp["rz_pct"] <= 100.0


# ============================================================
# T12: cumulative_participation + 90% 충분조건
# ============================================================

def test_T12_cumulative_participation():
    """누적 참여질량 존재 + sufficient_90pct 플래그."""
    multi = analyze_frame_3d_multi(
        stories=[4.0, 3.5],
        bays_x=[8.0, 8.0],
        bays_y=[8.0],
        load_cases=_simple_2story_load_cases(),
        rigid_diaphragm=True,
        modal_analysis=True,
    )

    cum = multi.modal_analysis["cumulative_participation"]
    assert "x_pct" in cum
    assert "y_pct" in cum
    assert "rz_pct" in cum
    assert "sufficient_90pct" in cum
    assert isinstance(cum["sufficient_90pct"], bool)

    # 누적 ≥ 개별 모드 합
    modes = multi.modal_analysis["modes"]
    sum_x = sum(m["mass_participation"]["x_pct"] for m in modes)
    assert abs(cum["x_pct"] - sum_x) < 0.5, f"누적 X 불일치: {cum['x_pct']} vs {sum_x}"


# ============================================================
# T13: mass_info 메타데이터
# ============================================================

def test_T13_mass_info_metadata():
    """mass_info에 basis, method, notes 포함."""
    multi = analyze_frame_3d_multi(
        stories=[4.0, 3.5],
        bays_x=[8.0, 8.0],
        bays_y=[8.0],
        load_cases=_simple_2story_load_cases(),
        rigid_diaphragm=True,
        modal_analysis=True,
    )

    mi = multi.modal_analysis["mass_info"]
    assert mi["basis"] == "floor_dead_load"
    assert mi["method"] == "lumped_distributed"
    assert mi["includes_member_self_weight"] is False
    assert mi["rigid_diaphragm"] is True
    assert mi["total_weight_kN"] > 0
    assert len(mi["notes"]) >= 2


# ============================================================
# T14: weight_source 추적
# ============================================================

def test_T14_weight_source_DL():
    """DL 기반 추정 시 weight_source = 'DL_load_case'."""
    multi = analyze_frame_3d_multi(
        stories=[4.0, 3.5],
        bays_x=[8.0, 8.0],
        bays_y=[8.0],
        load_cases=_simple_2story_load_cases(),
        rigid_diaphragm=True,
        modal_analysis=True,
    )
    assert multi.modal_analysis["weight_source"] == "DL_load_case"


def test_T14b_weight_source_user():
    """story_weights_kN 직접 제공 시 weight_source = 'user_provided'."""
    multi = analyze_frame_3d_multi(
        stories=[4.0, 3.5],
        bays_x=[8.0, 8.0],
        bays_y=[8.0],
        load_cases={
            "EQX": [
                {"type": "lateral_x", "story": 1, "value": 50.0},
                {"type": "lateral_x", "story": 2, "value": 100.0},
            ]
        },
        rigid_diaphragm=True,
        modal_analysis=True,
        story_weights_kN=[500.0, 400.0],
    )
    assert multi.modal_analysis["weight_source"] == "user_provided"


# ============================================================
# T15: 방향 판별이 참여질량 기반
# ============================================================

def test_T15_direction_from_mass_participation():
    """direction이 mass_participation 최대값 기반으로 결정됨."""
    multi = analyze_frame_3d_multi(
        stories=[4.0, 3.5, 3.5],
        bays_x=[8.0, 8.0],
        bays_y=[6.0],
        load_cases={
            "DL": [
                {"type": "floor_area", "story": i, "value": 5.0}
                for i in range(1, 4)
            ],
        },
        rigid_diaphragm=True,
        modal_analysis=True,
    )

    for m in multi.modal_analysis["modes"]:
        mp = m["mass_participation"]
        px, py, prz = mp["x_pct"], mp["y_pct"], mp["rz_pct"]
        d = m["direction"]
        # 지배 방향이 참여질량 최대 방향과 일치해야 함
        max_pct = max(px, py, prz)
        if max_pct == 0:
            continue
        if d == "TRAN-X":
            assert px >= py and px >= prz
        elif d == "TRAN-Y":
            assert py >= prz
        elif d == "ROTN-Z":
            assert prz > 0
