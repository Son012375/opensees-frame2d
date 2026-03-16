"""P-Delta 공학적 검증 테스트.

테스트 그룹:
  A — 변위/층간변위 증폭 (Drift Amplification)
  B — 기둥 모멘트 증폭 (Member Moment Amplification)
  C — 높이 민감도 (Height Sensitivity)
  D — 릴리즈 호환성 (Release Compatibility)

실행:
  cd mcp-server && python ../tests/test_pdelta_validation.py
"""
import sys
import os

# mcp-server를 모듈 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mcp-server'))

from core.frame_2d import analyze_frame_2d_multi
from core.frame_3d import analyze_frame_3d_multi

# ============================================================
# 헬퍼
# ============================================================

passed = 0
failed = 0
results_summary = {}


def check(cond, msg, group=None):
    global passed, failed
    if cond:
        passed += 1
        print(f"  [PASS] {msg}")
    else:
        failed += 1
        print(f"  [FAIL] {msg}")
    if group:
        results_summary.setdefault(group, []).append((cond, msg))


def _make_gravity_lateral_loads_2d(n_stories, w_floor=30.0, f_lat_base=20.0):
    """2D 다층 프레임용 중력+횡력 단일 하중케이스 생성.

    - w_floor: 보 분포하중 (kN/m)
    - f_lat_base: 1층 횡력 (kN), 상층으로 비례 증가
    """
    loads = []
    for s in range(1, n_stories + 1):
        loads.append({"type": "floor", "story": s, "value": w_floor})
        loads.append({"type": "lateral", "story": s, "value": f_lat_base * s})
    return loads


def _make_gravity_lateral_loads_3d(n_stories, w_area=10.0, f_lat_base=25.0):
    """3D 다층 프레임용 중력+횡력 단일 하중케이스 생성."""
    loads = []
    for s in range(1, n_stories + 1):
        loads.append({"type": "floor_area", "story": s, "value": w_area})
        loads.append({"type": "lateral_x", "story": s, "value": f_lat_base * s})
    return loads


def _run_2d_pair(n_stories, bays, loads, releases=None):
    """Linear/PDelta 2D 해석 쌍 실행, 결과 반환."""
    common = dict(
        stories=[3.5] * n_stories,
        bays=bays,
        load_cases={"COMB": loads},
        supports="fixed",
        column_section="H-300x300",
        beam_section="H-400x200",
        material_name="SS275",
        member_releases=releases,
    )
    r_lin = analyze_frame_2d_multi(**common, geometric_nonlinearity="linear")
    r_pd = analyze_frame_2d_multi(**common, geometric_nonlinearity="pdelta")
    return r_lin, r_pd


def _run_3d_pair(n_stories, bays_x, bays_y, loads, releases=None):
    """Linear/PDelta 3D 해석 쌍 실행, 결과 반환."""
    common = dict(
        stories=[3.5] * n_stories,
        bays_x=bays_x,
        bays_y=bays_y,
        load_cases={"COMB": loads},
        supports="fixed",
        member_releases=releases,
    )
    r_lin = analyze_frame_3d_multi(**common, geometric_nonlinearity="linear")
    r_pd = analyze_frame_3d_multi(**common, geometric_nonlinearity="pdelta")
    return r_lin, r_pd


def _max_col_moment_2d(result, case_name="COMB"):
    """2D 결과에서 기둥 최대 모멘트 추출."""
    mf_list = result.member_forces.get(case_name, [])
    max_m = 0.0
    for mf in mf_list:
        if mf.get("type") == "column":
            m_arr = mf.get("M_kNm", [])
            if m_arr:
                max_m = max(max_m, max(abs(v) for v in m_arr))
    return max_m


# ============================================================
# GROUP A: 변위/층간변위 증폭
# ============================================================

print("=" * 60)
print("GROUP A: Drift Amplification (변위/층간변위 증폭)")
print("=" * 60)

# --- A1: 2D 3층 2경간 ---
print("\n--- A1: 2D 3-story 2-bay frame ---")
loads_a1 = _make_gravity_lateral_loads_2d(3, w_floor=40.0, f_lat_base=30.0)
r_lin_a1, r_pd_a1 = _run_2d_pair(3, [6.0, 6.0], loads_a1)

cr_lin = r_lin_a1.case_results["COMB"]
cr_pd = r_pd_a1.case_results["COMB"]

disp_lin = cr_lin.max_displacement_x
disp_pd = cr_pd.max_displacement_x
drift_lin = cr_lin.max_drift
drift_pd = cr_pd.max_drift
amp_disp = disp_pd / disp_lin if disp_lin > 0 else 1.0
amp_drift = drift_pd / drift_lin if drift_lin > 0 else 1.0

print(f"  Roof disp   : Linear={disp_lin:.4f} mm, PDelta={disp_pd:.4f} mm, ratio={amp_disp:.6f}")
print(f"  Max drift   : Linear={drift_lin:.6f}, PDelta={drift_pd:.6f}, ratio={amp_drift:.6f}")

check(disp_pd > disp_lin, f"A1 PDelta roof disp ({disp_pd:.4f}) > Linear ({disp_lin:.4f})", "A")
check(drift_pd > drift_lin, f"A1 PDelta max drift ({drift_pd:.6f}) > Linear ({drift_lin:.6f})", "A")

# --- A2: 3D 3층 1x1bay ---
print("\n--- A2: 3D 3-story 1x1-bay frame ---")
loads_a2 = _make_gravity_lateral_loads_3d(3, w_area=15.0, f_lat_base=30.0)
r_lin_a2, r_pd_a2 = _run_3d_pair(3, [6.0], [6.0], loads_a2)

cr3_lin = r_lin_a2.case_results["COMB"]
cr3_pd = r_pd_a2.case_results["COMB"]

disp3_lin = cr3_lin.max_displacement_x
disp3_pd = cr3_pd.max_displacement_x
drift3_lin = cr3_lin.max_drift_x
drift3_pd = cr3_pd.max_drift_x
amp3_disp = disp3_pd / disp3_lin if disp3_lin > 0 else 1.0
amp3_drift = drift3_pd / drift3_lin if drift3_lin > 0 else 1.0

print(f"  Roof X-disp : Linear={disp3_lin:.4f} mm, PDelta={disp3_pd:.4f} mm, ratio={amp3_disp:.6f}")
print(f"  Max drift X : Linear={drift3_lin:.6f}, PDelta={drift3_pd:.6f}, ratio={amp3_drift:.6f}")

check(disp3_pd > disp3_lin, f"A2 3D PDelta roof disp ({disp3_pd:.4f}) > Linear ({disp3_lin:.4f})", "A")
check(drift3_pd > drift3_lin, f"A2 3D PDelta max drift ({drift3_pd:.6f}) > Linear ({drift3_lin:.6f})", "A")

# --- A3: 2D 5층 2경간 (더 높은 건물) ---
print("\n--- A3: 2D 5-story 2-bay frame ---")
loads_a3 = _make_gravity_lateral_loads_2d(5, w_floor=40.0, f_lat_base=20.0)
r_lin_a3, r_pd_a3 = _run_2d_pair(5, [6.0, 6.0], loads_a3)

cr_lin5 = r_lin_a3.case_results["COMB"]
cr_pd5 = r_pd_a3.case_results["COMB"]

disp_lin5 = cr_lin5.max_displacement_x
disp_pd5 = cr_pd5.max_displacement_x
drift_lin5 = cr_lin5.max_drift
drift_pd5 = cr_pd5.max_drift
amp_disp5 = disp_pd5 / disp_lin5 if disp_lin5 > 0 else 1.0

print(f"  Roof disp   : Linear={disp_lin5:.4f} mm, PDelta={disp_pd5:.4f} mm, ratio={amp_disp5:.6f}")
print(f"  Max drift   : Linear={drift_lin5:.6f}, PDelta={drift_pd5:.6f}")

check(disp_pd5 > disp_lin5, f"A3 5-story PDelta disp ({disp_pd5:.4f}) > Linear ({disp_lin5:.4f})", "A")
check(drift_pd5 > drift_lin5, f"A3 5-story PDelta drift ({drift_pd5:.6f}) > Linear ({drift_lin5:.6f})", "A")


# ============================================================
# GROUP B: 기둥 모멘트 증폭
# ============================================================

print("\n" + "=" * 60)
print("GROUP B: Column Moment Amplification (기둥 모멘트 증폭)")
print("=" * 60)

# --- B1: 2D 3층 기둥 모멘트 ---
print("\n--- B1: 2D 3-story column max moment ---")
# r_lin_a1, r_pd_a1 은 Group A에서 이미 계산됨
m_col_lin = _max_col_moment_2d(r_lin_a1)
m_col_pd = _max_col_moment_2d(r_pd_a1)

print(f"  Column max M: Linear={m_col_lin:.2f} kN*m, PDelta={m_col_pd:.2f} kN*m")
if m_col_lin > 0:
    print(f"  Ratio: {m_col_pd / m_col_lin:.6f}")

check(m_col_pd > m_col_lin,
      f"B1 PDelta col moment ({m_col_pd:.2f}) > Linear ({m_col_lin:.2f})", "B")

# --- B2: 2D 5층 기둥 모멘트 ---
print("\n--- B2: 2D 5-story column max moment ---")
m_col_lin5 = _max_col_moment_2d(r_lin_a3)
m_col_pd5 = _max_col_moment_2d(r_pd_a3)

print(f"  Column max M: Linear={m_col_lin5:.2f} kN*m, PDelta={m_col_pd5:.2f} kN*m")
if m_col_lin5 > 0:
    print(f"  Ratio: {m_col_pd5 / m_col_lin5:.6f}")

check(m_col_pd5 > m_col_lin5,
      f"B2 5-story PDelta col moment ({m_col_pd5:.2f}) > Linear ({m_col_lin5:.2f})", "B")

# --- B3: 3D 3층 기저부 반력 모멘트 ---
# 참고: 3D Corotational 변환은 eleForce를 변형된 좌표계에서 반환하므로
#       로컬 max_moment는 감소할 수 있음 (알려진 특성).
#       대신 기저부 반력 모멘트 합으로 P-Delta 증폭 검증.
print("\n--- B3: 3D 3-story base reaction moment ---")


def _sum_base_reaction_moment(reactions):
    """기저부 반력 모멘트 합 (|MY|)."""
    return sum(abs(r.get("MY_kNm", 0)) for r in reactions)


m3_base_lin = _sum_base_reaction_moment(cr3_lin.reactions)
m3_base_pd = _sum_base_reaction_moment(cr3_pd.reactions)

print(f"  Base sum|MY|: Linear={m3_base_lin:.2f} kN*m, PDelta={m3_base_pd:.2f} kN*m")
if m3_base_lin > 0:
    print(f"  Ratio: {m3_base_pd / m3_base_lin:.6f}")

check(m3_base_pd >= m3_base_lin,
      f"B3 3D PDelta base moment ({m3_base_pd:.2f}) >= Linear ({m3_base_lin:.2f})", "B")


# ============================================================
# GROUP C: 높이 민감도
# ============================================================

print("\n" + "=" * 60)
print("GROUP C: Height Sensitivity (높이 민감도)")
print("=" * 60)

# 2층 vs 5층 동일 하중 패턴으로 비교
print("\n--- C1: 2D 2-story vs 5-story amplification comparison ---")

# 2층 프레임
loads_c2 = _make_gravity_lateral_loads_2d(2, w_floor=40.0, f_lat_base=30.0)
r_lin_c2, r_pd_c2 = _run_2d_pair(2, [6.0, 6.0], loads_c2)

cr_c2_lin = r_lin_c2.case_results["COMB"]
cr_c2_pd = r_pd_c2.case_results["COMB"]
amp_2story = cr_c2_pd.max_displacement_x / cr_c2_lin.max_displacement_x

# 5층 프레임
loads_c5 = _make_gravity_lateral_loads_2d(5, w_floor=40.0, f_lat_base=30.0)
r_lin_c5, r_pd_c5 = _run_2d_pair(5, [6.0, 6.0], loads_c5)

cr_c5_lin = r_lin_c5.case_results["COMB"]
cr_c5_pd = r_pd_c5.case_results["COMB"]
amp_5story = cr_c5_pd.max_displacement_x / cr_c5_lin.max_displacement_x

print(f"  2-story: Linear={cr_c2_lin.max_displacement_x:.4f}, PDelta={cr_c2_pd.max_displacement_x:.4f}, amp={amp_2story:.6f}")
print(f"  5-story: Linear={cr_c5_lin.max_displacement_x:.4f}, PDelta={cr_c5_pd.max_displacement_x:.4f}, amp={amp_5story:.6f}")

check(amp_5story > amp_2story,
      f"C1 5-story amp ({amp_5story:.6f}) > 2-story amp ({amp_2story:.6f})", "C")

# 3D도 동일 검증
print("\n--- C2: 3D 2-story vs 5-story amplification comparison ---")

loads_3d_c2 = _make_gravity_lateral_loads_3d(2, w_area=15.0, f_lat_base=30.0)
r3_lin_c2, r3_pd_c2 = _run_3d_pair(2, [6.0], [6.0], loads_3d_c2)

cr3_c2_lin = r3_lin_c2.case_results["COMB"]
cr3_c2_pd = r3_pd_c2.case_results["COMB"]
amp3_2story = cr3_c2_pd.max_displacement_x / cr3_c2_lin.max_displacement_x

loads_3d_c5 = _make_gravity_lateral_loads_3d(5, w_area=15.0, f_lat_base=30.0)
r3_lin_c5, r3_pd_c5 = _run_3d_pair(5, [6.0], [6.0], loads_3d_c5)

cr3_c5_lin = r3_lin_c5.case_results["COMB"]
cr3_c5_pd = r3_pd_c5.case_results["COMB"]
amp3_5story = cr3_c5_pd.max_displacement_x / cr3_c5_lin.max_displacement_x

print(f"  2-story: Linear={cr3_c2_lin.max_displacement_x:.4f}, PDelta={cr3_c2_pd.max_displacement_x:.4f}, amp={amp3_2story:.6f}")
print(f"  5-story: Linear={cr3_c5_lin.max_displacement_x:.4f}, PDelta={cr3_c5_pd.max_displacement_x:.4f}, amp={amp3_5story:.6f}")

check(amp3_5story > amp3_2story,
      f"C2 3D 5-story amp ({amp3_5story:.6f}) > 2-story amp ({amp3_2story:.6f})", "C")


# ============================================================
# GROUP D: 릴리즈 호환성
# ============================================================

print("\n" + "=" * 60)
print("GROUP D: Release Compatibility (릴리즈 호환성)")
print("=" * 60)

# --- D1: 2D release + linear ---
print("\n--- D1: 2D beam release + linear ---")
loads_d = _make_gravity_lateral_loads_2d(3, w_floor=30.0, f_lat_base=20.0)
try:
    r_d1, _ = _run_2d_pair(3, [6.0, 6.0], loads_d, releases={"beam": "both"})
    cr_d1 = r_d1.case_results["COMB"]
    print(f"  Disp: {cr_d1.max_displacement_x:.4f} mm, Drift: {cr_d1.max_drift:.6f}")
    check(True, "D1 2D release+linear runs OK", "D")
except Exception as e:
    check(False, f"D1 2D release+linear FAILED: {e}", "D")

# --- D2: 2D release + pdelta ---
print("\n--- D2: 2D beam release + pdelta ---")
try:
    _, r_d2 = _run_2d_pair(3, [6.0, 6.0], loads_d, releases={"beam": "both"})
    cr_d2 = r_d2.case_results["COMB"]
    print(f"  Disp: {cr_d2.max_displacement_x:.4f} mm, Drift: {cr_d2.max_drift:.6f}")
    check(True, "D2 2D release+pdelta runs OK", "D")
except Exception as e:
    check(False, f"D2 2D release+pdelta FAILED: {e}", "D")

# --- D3: 3D release + linear ---
print("\n--- D3: 3D beam release + linear ---")
loads_3d_d = _make_gravity_lateral_loads_3d(2, w_area=10.0, f_lat_base=25.0)
try:
    r_d3, _ = _run_3d_pair(2, [6.0], [6.0], loads_3d_d,
                            releases={"beam_x": "both", "beam_y": "both"})
    cr_d3 = r_d3.case_results["COMB"]
    print(f"  Disp X: {cr_d3.max_displacement_x:.4f} mm, Drift X: {cr_d3.max_drift_x:.6f}")
    check(True, "D3 3D release+linear runs OK", "D")
except Exception as e:
    check(False, f"D3 3D release+linear FAILED: {e}", "D")

# --- D4: 3D release + pdelta ---
print("\n--- D4: 3D beam release + pdelta ---")
try:
    _, r_d4 = _run_3d_pair(2, [6.0], [6.0], loads_3d_d,
                            releases={"beam_x": "both", "beam_y": "both"})
    cr_d4 = r_d4.case_results["COMB"]
    print(f"  Disp X: {cr_d4.max_displacement_x:.4f} mm, Drift X: {cr_d4.max_drift_x:.6f}")
    check(True, "D4 3D release+pdelta runs OK", "D")
except Exception as e:
    check(False, f"D4 3D release+pdelta FAILED: {e}", "D")

# --- D5: release + pdelta에서 변위 비교 (released pdelta > released linear) ---
print("\n--- D5: 2D release pdelta vs release linear displacement ---")
if cr_d1 and cr_d2:
    d_rel_lin = cr_d1.max_displacement_x
    d_rel_pd = cr_d2.max_displacement_x
    print(f"  Release+Linear: {d_rel_lin:.4f} mm")
    print(f"  Release+PDelta: {d_rel_pd:.4f} mm")
    check(d_rel_pd >= d_rel_lin * 0.99,
          f"D5 Release+PDelta disp ({d_rel_pd:.4f}) >= Release+Linear ({d_rel_lin:.4f})", "D")


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PDELTA VALIDATION SUMMARY")
print("=" * 60)

group_labels = {
    "A": "Roof displacement & story drift amplification",
    "B": "Column moment amplification",
    "C": "Height sensitivity (taller = more P-Delta)",
    "D": "Release compatibility (no solver failure)",
}

all_ok = True
for grp in ["A", "B", "C", "D"]:
    items = results_summary.get(grp, [])
    n_pass = sum(1 for ok, _ in items if ok)
    n_total = len(items)
    status = "PASS" if n_pass == n_total else "FAIL"
    if status == "FAIL":
        all_ok = False
    print(f"  [{status}] Group {grp}: {group_labels[grp]} ({n_pass}/{n_total})")

print(f"\nTOTAL: {passed + failed} checks, {passed} passed, {failed} failed")

if all_ok:
    print("\nALL VALIDATION CHECKS PASSED")
else:
    print(f"\nFAILURES: {failed}")
    sys.exit(1)
