"""Member End Release (Hinge) 검증 테스트.

Test 1: 2D 1층 1경간 포탈프레임 - 강절 vs 보 양단 핀
Test 2: 3D 2층 1x1bay - 강절 vs 보 양단 핀
Test 3: 하위호환성 - member_releases=None
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from core.frame_2d import analyze_frame_2d_multi
from core.frame_3d import analyze_frame_3d_multi


def test_2d_release():
    """Test 1: 2D portal frame - rigid vs beam both-end hinged."""
    print("=" * 60)
    print("TEST 1: 2D Frame - 1-story 1-bay portal (w=20 kN/m)")
    print("=" * 60)

    load_cases = {"DL": [{"type": "floor", "story": 1, "value": 20.0}]}

    # 강절
    r_rigid = analyze_frame_2d_multi(
        stories=[3.0], bays=[6.0],
        load_cases=load_cases,
        supports="fixed",
    )

    # 보 양단 핀
    r_hinged = analyze_frame_2d_multi(
        stories=[3.0], bays=[6.0],
        load_cases=load_cases,
        supports="fixed",
        member_releases={"beam": "both"},
    )

    cr_rigid = r_rigid.case_results["DL"]
    cr_hinged = r_hinged.case_results["DL"]

    print("\n--- Rigid Frame ---")
    print(f"  Max moment: {cr_rigid.max_moment:.2f} kN*m (elem {cr_rigid.max_moment_element})")
    print(f"  Max displacement Y: {cr_rigid.max_displacement_y:.4f} mm")

    print("\n--- Hinged Beam (both ends) ---")
    print(f"  Max moment: {cr_hinged.max_moment:.2f} kN*m (elem {cr_hinged.max_moment_element})")
    print(f"  Max displacement Y: {cr_hinged.max_displacement_y:.4f} mm")

    # 검증: 핀 보의 중앙 모멘트 = wL^2/8 = 20*6^2/8 = 90 kN*m
    expected_midspan = 20.0 * 6.0**2 / 8.0  # 90 kN*m
    print(f"\n  Expected midspan moment (wL²/8): {expected_midspan:.1f} kN*m")

    # 핀 보의 양단 모멘트 = 0 확인 (member_forces에서)
    mf_hinged = r_hinged.member_forces.get("DL", [])
    beam_members = [m for m in mf_hinged if m.get("type") == "beam"]
    assert beam_members, "No beam members found in member_forces"
    bm = beam_members[0]
    M = bm["M_kNm"]
    m_i = M[0]
    m_j = M[-1]
    print(f"  Beam i-end moment: {m_i:.4f} kN*m (expect ~= 0)")
    print(f"  Beam j-end moment: {m_j:.4f} kN*m (expect ~= 0)")
    assert abs(m_i) < 0.1, f"Beam i-end moment should be ~= 0, got {m_i}"
    assert abs(m_j) < 0.1, f"Beam j-end moment should be ~= 0, got {m_j}"
    mid_idx = len(M) // 2
    m_mid = abs(M[mid_idx])
    print(f"  Beam midspan moment: {m_mid:.2f} kN*m (expect ~= {expected_midspan:.1f})")
    assert abs(m_mid - expected_midspan) < 2.0, f"Midspan moment should be ~= {expected_midspan}, got {m_mid}"

    # 핀 보의 최대 모멘트가 더 커야 함 (단순보 90 > 강절 46.25)
    assert cr_hinged.max_moment > cr_rigid.max_moment, \
        f"Hinged beam should have larger midspan moment: {cr_hinged.max_moment} vs {cr_rigid.max_moment}"

    # member_releases 저장 확인
    assert r_hinged.member_releases == {"beam": "both"}, \
        f"Expected releases stored, got {r_hinged.member_releases}"
    assert r_rigid.member_releases == {}, \
        f"Expected empty releases, got {r_rigid.member_releases}"

    print("\nOK TEST 1 PASSED\n")


def test_3d_release():
    """Test 2: 3D frame - rigid vs beam both-end hinged."""
    print("=" * 60)
    print("TEST 2: 3D Frame - 2-story 1x1bay (gravity + lateral)")
    print("=" * 60)

    load_cases = {
        "DL": [
            {"type": "floor_area", "story": 1, "value": 5.0},
            {"type": "floor_area", "story": 2, "value": 5.0},
        ],
        "EQX": [
            {"type": "lateral_x", "story": 1, "value": 50.0},
            {"type": "lateral_x", "story": 2, "value": 100.0},
        ],
    }

    # 강절
    r_rigid = analyze_frame_3d_multi(
        stories=[3.5, 3.5], bays_x=[6.0], bays_y=[6.0],
        load_cases=load_cases,
        supports="fixed",
    )

    # 보 양단 핀
    r_hinged = analyze_frame_3d_multi(
        stories=[3.5, 3.5], bays_x=[6.0], bays_y=[6.0],
        load_cases=load_cases,
        supports="fixed",
        member_releases={"beam_x": "both", "beam_y": "both"},
    )

    cr_rigid_dl = r_rigid.case_results["DL"]
    cr_hinged_dl = r_hinged.case_results["DL"]
    cr_rigid_eq = r_rigid.case_results["EQX"]
    cr_hinged_eq = r_hinged.case_results["EQX"]

    print("\n--- Gravity (DL) ---")
    print(f"  Rigid  max disp Z: {cr_rigid_dl.max_displacement_z:.4f} mm")
    print(f"  Hinged max disp Z: {cr_hinged_dl.max_displacement_z:.4f} mm")

    print("\n--- Lateral (EQX) ---")
    print(f"  Rigid  max disp X: {cr_rigid_eq.max_displacement_x:.4f} mm")
    print(f"  Hinged max disp X: {cr_hinged_eq.max_displacement_x:.4f} mm")
    print(f"  Rigid  max drift: {cr_rigid_eq.max_drift_x:.6f}")
    print(f"  Hinged max drift: {cr_hinged_eq.max_drift_x:.6f}")

    # 핀 보: 횡변위가 더 커야 함
    assert cr_hinged_eq.max_displacement_x > cr_rigid_eq.max_displacement_x, \
        "Hinged beams should result in larger lateral displacement"

    # member_releases 저장 확인
    assert r_hinged.member_releases == {"beam_x": "both", "beam_y": "both"}, \
        f"Expected releases stored, got {r_hinged.member_releases}"

    print("\nOK TEST 2 PASSED\n")


def test_backward_compat():
    """Test 3: member_releases=None → 기존 동작과 동일."""
    print("=" * 60)
    print("TEST 3: Backward Compatibility (member_releases=None)")
    print("=" * 60)

    load_cases = {"DL": [{"type": "floor", "story": 1, "value": 10.0}]}

    r1 = analyze_frame_2d_multi(
        stories=[3.0], bays=[6.0],
        load_cases=load_cases,
    )

    r2 = analyze_frame_2d_multi(
        stories=[3.0], bays=[6.0],
        load_cases=load_cases,
        member_releases=None,
    )

    cr1 = r1.case_results["DL"]
    cr2 = r2.case_results["DL"]

    assert abs(cr1.max_moment - cr2.max_moment) < 0.001, \
        f"Results should be identical: {cr1.max_moment} vs {cr2.max_moment}"
    assert r1.member_releases == {}, f"Default should be empty dict"
    assert r2.member_releases == {}, f"None should become empty dict"

    print("  Rigid (default) max moment:", cr1.max_moment)
    print("  Rigid (explicit None) max moment:", cr2.max_moment)
    print("\nOK TEST 3 PASSED\n")


if __name__ == "__main__":
    test_2d_release()
    test_3d_release()
    test_backward_compat()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
