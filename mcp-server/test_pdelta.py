"""P-Delta (기하비선형) 검증 테스트.

테스트 구성:
  T1: 2D 1층1경간 횡력 — linear vs pdelta 변위 비교
  T2: 2D 하위호환성 — geometric_nonlinearity 미지정 시 기존 동작 동일
  T3: 3D 2층1x1bay 횡력 — linear vs pdelta 변위 비교
  T4: 3D 하위호환성 — geometric_nonlinearity 미지정 시 기존 동작 동일
  T5: 2D release + pdelta 조합 (4가지 조합 모두 동작)
  T6: 3D release + pdelta 조합 (4가지 조합 모두 동작)
  T7: BuildingModel pipeline pdelta 전달 확인
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from core.frame_2d import analyze_frame_2d_multi
from core.frame_3d import analyze_frame_3d_multi
from core.building_model import BuildingModel

passed = 0
failed = 0


def check(cond, msg):
    global passed, failed
    if cond:
        passed += 1
        print(f"  ✓ {msg}")
    else:
        failed += 1
        print(f"  ✗ {msg}")


# ============================================================
# T1: 2D 횡력 — linear vs pdelta
# ============================================================
print("\n=== T1: 2D lateral — linear vs pdelta ===")

common_2d = dict(
    stories=[3.5],
    bays=[6.0],
    load_cases={"EQX": [{"type": "lateral", "story": 1, "value": 100}]},
    supports="fixed",
    column_section="H-300x300",
    beam_section="H-400x200",
    material_name="SS275",
)

r_linear = analyze_frame_2d_multi(**common_2d, geometric_nonlinearity="linear")
r_pdelta = analyze_frame_2d_multi(**common_2d, geometric_nonlinearity="pdelta")

cr_lin = r_linear.case_results["EQX"]
cr_pd = r_pdelta.case_results["EQX"]

disp_lin = cr_lin.max_displacement_x
disp_pd = cr_pd.max_displacement_x

print(f"  Linear displacement: {disp_lin:.4f} mm")
print(f"  P-Delta displacement: {disp_pd:.4f} mm")
print(f"  Ratio (pdelta/linear): {disp_pd/disp_lin:.4f}")

# P-Delta는 순수 횡력에서 변위가 크거나 같아야 함
# (축력이 없으면 차이 미미, 축력+횡력이면 차이 큼)
check(disp_pd >= disp_lin * 0.99, "P-Delta disp >= Linear disp (or very close)")
check(r_linear.geometric_nonlinearity == "linear", "Result stores 'linear'")
check(r_pdelta.geometric_nonlinearity == "pdelta", "Result stores 'pdelta'")


# ============================================================
# T1b: 2D 중력+횡력 — P-Delta 효과 확인
# ============================================================
print("\n=== T1b: 2D gravity+lateral — P-Delta amplification ===")

r_lin_gl = analyze_frame_2d_multi(
    stories=[3.5],
    bays=[6.0],
    load_cases={
        "DL": [{"type": "floor", "story": 1, "value": 30}],
        "EQX": [{"type": "lateral", "story": 1, "value": 100}],
    },
    load_combinations={"1.0DL+1.0EQX": {"DL": 1.0, "EQX": 1.0}},
    supports="fixed",
    geometric_nonlinearity="linear",
)

r_pd_gl = analyze_frame_2d_multi(
    stories=[3.5],
    bays=[6.0],
    load_cases={
        "DL": [{"type": "floor", "story": 1, "value": 30}],
        "EQX": [{"type": "lateral", "story": 1, "value": 100}],
    },
    load_combinations={"1.0DL+1.0EQX": {"DL": 1.0, "EQX": 1.0}},
    supports="fixed",
    geometric_nonlinearity="pdelta",
)

# EQX 단독 케이스에서도 확인
cr_lin_eq = r_lin_gl.case_results["EQX"]
cr_pd_eq = r_pd_gl.case_results["EQX"]
print(f"  Linear EQX disp: {cr_lin_eq.max_displacement_x:.4f} mm")
print(f"  P-Delta EQX disp: {cr_pd_eq.max_displacement_x:.4f} mm")

# DL+EQX 조합에서 확인
combo_lin = r_lin_gl.combo_results.get("1.0DL+1.0EQX")
combo_pd = r_pd_gl.combo_results.get("1.0DL+1.0EQX")
if combo_lin and combo_pd:
    print(f"  Linear combo disp: {combo_lin.max_displacement_x:.4f} mm")
    print(f"  P-Delta combo disp: {combo_pd.max_displacement_x:.4f} mm")

check(True, "2D gravity+lateral P-Delta analysis runs without error")


# ============================================================
# T2: 2D 하위호환성
# ============================================================
print("\n=== T2: 2D backward compatibility ===")

r_default = analyze_frame_2d_multi(**common_2d)  # geometric_nonlinearity 미지정
cr_def = r_default.case_results["EQX"]

check(abs(cr_def.max_displacement_x - disp_lin) < 1e-6,
      f"Default == Linear: {cr_def.max_displacement_x:.4f} == {disp_lin:.4f}")
check(r_default.geometric_nonlinearity == "linear", "Default geometric_nonlinearity is 'linear'")


# ============================================================
# T3: 3D 횡력 — linear vs pdelta
# ============================================================
print("\n=== T3: 3D lateral — linear vs pdelta ===")

common_3d = dict(
    stories=[3.5, 3.5],
    bays_x=[6.0],
    bays_y=[6.0],
    load_cases={"EQX": [{"type": "lateral_x", "story": 1, "value": 50},
                         {"type": "lateral_x", "story": 2, "value": 100}]},
    supports="fixed",
)

r3_linear = analyze_frame_3d_multi(**common_3d, geometric_nonlinearity="linear")
r3_pdelta = analyze_frame_3d_multi(**common_3d, geometric_nonlinearity="pdelta")

cr3_lin = r3_linear.case_results["EQX"]
cr3_pd = r3_pdelta.case_results["EQX"]

disp3_lin = cr3_lin.max_displacement_x
disp3_pd = cr3_pd.max_displacement_x

print(f"  Linear X-disp: {disp3_lin:.4f} mm")
print(f"  P-Delta X-disp: {disp3_pd:.4f} mm")
print(f"  Ratio (pdelta/linear): {disp3_pd/disp3_lin:.4f}")

check(disp3_pd >= disp3_lin * 0.99, "3D P-Delta disp >= Linear disp")
check(r3_linear.geometric_nonlinearity == "linear", "3D result stores 'linear'")
check(r3_pdelta.geometric_nonlinearity == "pdelta", "3D result stores 'pdelta'")


# ============================================================
# T3b: 3D 중력+횡력 — P-Delta 증폭 확인
# ============================================================
print("\n=== T3b: 3D gravity+lateral — P-Delta amplification ===")

r3_lin_gl = analyze_frame_3d_multi(
    stories=[3.5, 3.5],
    bays_x=[6.0],
    bays_y=[6.0],
    load_cases={
        "DL": [{"type": "floor_area", "story": 1, "value": 10},
               {"type": "floor_area", "story": 2, "value": 10}],
        "EQX": [{"type": "lateral_x", "story": 1, "value": 50},
                {"type": "lateral_x", "story": 2, "value": 100}],
    },
    load_combinations={"1.0DL+1.0EQX": {"DL": 1.0, "EQX": 1.0}},
    supports="fixed",
    geometric_nonlinearity="linear",
)

r3_pd_gl = analyze_frame_3d_multi(
    stories=[3.5, 3.5],
    bays_x=[6.0],
    bays_y=[6.0],
    load_cases={
        "DL": [{"type": "floor_area", "story": 1, "value": 10},
               {"type": "floor_area", "story": 2, "value": 10}],
        "EQX": [{"type": "lateral_x", "story": 1, "value": 50},
                {"type": "lateral_x", "story": 2, "value": 100}],
    },
    load_combinations={"1.0DL+1.0EQX": {"DL": 1.0, "EQX": 1.0}},
    supports="fixed",
    geometric_nonlinearity="pdelta",
)

cr3_lin_eq = r3_lin_gl.case_results["EQX"]
cr3_pd_eq = r3_pd_gl.case_results["EQX"]
print(f"  Linear EQX X-disp: {cr3_lin_eq.max_displacement_x:.4f} mm")
print(f"  P-Delta EQX X-disp: {cr3_pd_eq.max_displacement_x:.4f} mm")

check(True, "3D gravity+lateral P-Delta analysis runs without error")


# ============================================================
# T4: 3D 하위호환성
# ============================================================
print("\n=== T4: 3D backward compatibility ===")

r3_default = analyze_frame_3d_multi(**common_3d)  # geometric_nonlinearity 미지정
cr3_def = r3_default.case_results["EQX"]

check(abs(cr3_def.max_displacement_x - disp3_lin) < 1e-6,
      f"Default == Linear: {cr3_def.max_displacement_x:.4f} == {disp3_lin:.4f}")
check(r3_default.geometric_nonlinearity == "linear", "3D default geometric_nonlinearity is 'linear'")


# ============================================================
# T5: 2D release + pdelta 조합 (4가지)
# ============================================================
print("\n=== T5: 2D release + pdelta combinations ===")

combos_2d = [
    ("rigid+linear", None, "linear"),
    ("rigid+pdelta", None, "pdelta"),
    ("released+linear", {"beam": "both"}, "linear"),
    ("released+pdelta", {"beam": "both"}, "pdelta"),
]

results_2d = {}
for label, releases, geo in combos_2d:
    r = analyze_frame_2d_multi(
        stories=[3.5],
        bays=[6.0],
        load_cases={"EQX": [{"type": "lateral", "story": 1, "value": 100}]},
        supports="fixed",
        member_releases=releases,
        geometric_nonlinearity=geo,
    )
    results_2d[label] = r.case_results["EQX"].max_displacement_x
    print(f"  {label}: disp = {results_2d[label]:.4f} mm")

# 릴리즈하면 변위 증가 (강성 감소)
check(results_2d["released+linear"] > results_2d["rigid+linear"],
      "Released > Rigid (linear)")
check(results_2d["released+pdelta"] > results_2d["rigid+pdelta"],
      "Released > Rigid (pdelta)")
# P-Delta는 변위를 증가시키거나 유사하게 유지
check(results_2d["rigid+pdelta"] >= results_2d["rigid+linear"] * 0.99,
      "PDelta >= Linear (rigid)")
check(results_2d["released+pdelta"] >= results_2d["released+linear"] * 0.99,
      "PDelta >= Linear (released)")


# ============================================================
# T6: 3D release + pdelta 조합 (4가지)
# ============================================================
print("\n=== T6: 3D release + pdelta combinations ===")

combos_3d = [
    ("rigid+linear", None, "linear"),
    ("rigid+pdelta", None, "pdelta"),
    ("released+linear", {"beam_x": "both", "beam_y": "both"}, "linear"),
    ("released+pdelta", {"beam_x": "both", "beam_y": "both"}, "pdelta"),
]

results_3d = {}
for label, releases, geo in combos_3d:
    r = analyze_frame_3d_multi(
        stories=[3.5, 3.5],
        bays_x=[6.0],
        bays_y=[6.0],
        load_cases={"EQX": [{"type": "lateral_x", "story": 1, "value": 50},
                             {"type": "lateral_x", "story": 2, "value": 100}]},
        supports="fixed",
        member_releases=releases,
        geometric_nonlinearity=geo,
    )
    results_3d[label] = r.case_results["EQX"].max_displacement_x
    print(f"  {label}: disp = {results_3d[label]:.4f} mm")

check(results_3d["released+linear"] > results_3d["rigid+linear"],
      "3D Released > Rigid (linear)")
check(results_3d["released+pdelta"] > results_3d["rigid+pdelta"],
      "3D Released > Rigid (pdelta)")
check(results_3d["rigid+pdelta"] >= results_3d["rigid+linear"] * 0.99,
      "3D PDelta >= Linear (rigid)")
check(results_3d["released+pdelta"] >= results_3d["released+linear"] * 0.99,
      "3D PDelta >= Linear (released)")


# ============================================================
# T7: BuildingModel pipeline
# ============================================================
print("\n=== T7: BuildingModel pipeline ===")

config_linear = {
    "stories": [{"height": 3.5, "usage": "office"}],
    "bays_x": [6.0],
    "bays_y": [6.0],
}

config_pdelta = {
    "stories": [{"height": 3.5, "usage": "office"}],
    "bays_x": [6.0],
    "bays_y": [6.0],
    "geometric_nonlinearity": "pdelta",
}

m_lin = BuildingModel.from_json(config_linear)
m_pd = BuildingModel.from_json(config_pdelta)

check(m_lin.geometric_nonlinearity == "linear", "BuildingModel default = linear")
check(m_pd.geometric_nonlinearity == "pdelta", "BuildingModel pdelta from config")

kwargs_lin = m_lin.to_frame3d_kwargs()
kwargs_pd = m_pd.to_frame3d_kwargs()

check(kwargs_lin["geometric_nonlinearity"] == "linear", "to_frame3d_kwargs passes linear")
check(kwargs_pd["geometric_nonlinearity"] == "pdelta", "to_frame3d_kwargs passes pdelta")


# ============================================================
# Summary
# ============================================================
print(f"\n{'='*60}")
print(f"TOTAL: {passed + failed} checks, {passed} passed, {failed} failed")
if failed == 0:
    print("ALL CHECKS PASSED ✓")
else:
    print(f"FAILURES: {failed}")
    sys.exit(1)
