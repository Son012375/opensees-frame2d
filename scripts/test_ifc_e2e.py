"""IFC -> BuildingModel -> Load Generation -> OpenSees 3D Frame E2E Test."""
import sys
import json

sys.path.insert(0, "mcp-server")

from core.building_model import BuildingModel
from core.load_generator import generate_all_loads
from core.frame_3d import analyze_frame_3d_multi

IFC_PATH = r"C:\Users\youm\Documents\카카오톡 받은 파일\2차시도_260121.ifc"

# ── 1. IFC -> BuildingModel ──
print("=" * 60)
print("Phase 1: IFC -> BuildingModel")
print("=" * 60)

config = {
    "stories": [
        {"usage": "retail", "dead_load_finish": 1.5},
        {"usage": "office", "dead_load_finish": 1.0},
        {"usage": "office", "dead_load_finish": 0.5},
    ],
    "region": "\uc11c\uc6b8",
    "site_class": "S3",
    "importance": "II",
    "column_section": "H-300x300",
    "beam_x_section": "H-400x200",
    "beam_y_section": "H-400x200",
    "seismic_system": "ordinary_moment_frame",
    "auto_combinations": True,
}

model = BuildingModel.from_ifc(IFC_PATH, config)
print(f"  Stories: {model.num_stories}, Height: {model.total_height}m")
print(f"  Plan: {model.total_width_x}m x {model.total_width_y}m = {model.floor_area}m2")

# ── 2. Load Generation ──
print("\n" + "=" * 60)
print("Phase 2: Automatic Load Generation")
print("=" * 60)

load_result = generate_all_loads(model)
load_cases = load_result["load_cases"]
combinations = load_result.get("load_combinations", {})
reports = load_result.get("reports", {})

print(f"  Load cases: {list(load_cases.keys())}")
print(f"  Combinations: {len(combinations)}")

# 하중 요약
if "seismic" in reports:
    s = reports["seismic"]
    print(f"  Seismic: V={s.get('V_kN', 0):.1f} kN, Cs={s.get('Cs', 0):.6f}, W={s.get('W_kN', 0):.1f} kN")
if "wind" in reports:
    w = reports["wind"]
    print(f"  Wind: Fx={w.get('total_Fx_kN', 0):.1f} kN, Fy={w.get('total_Fy_kN', 0):.1f} kN, V0={w.get('V0_ms', 0)} m/s")

# ── 3. OpenSees Analysis ──
print("\n" + "=" * 60)
print("Phase 3: OpenSees 3D Frame Analysis")
print("=" * 60)

kwargs = model.to_frame3d_kwargs()
kwargs["load_cases"] = load_cases
kwargs["load_combinations"] = combinations

result = analyze_frame_3d_multi(**kwargs)

print(f"  Nodes: {len(result.nodes)}, Elements: {len(result.elements)}")
print(f"  Cases: {list(result.case_results.keys())}")
print(f"  Combos: {list(result.combo_results.keys())}")

# ── 4. Equilibrium Check ──
print("\n" + "=" * 60)
print("Phase 4: Results & Verification")
print("=" * 60)

ok = True

for case_name, case_data in result.case_results.items():
    reactions = case_data.reactions
    if not reactions:
        continue

    total_Rz = sum(r.get("RZ_kN", 0) for r in reactions)
    total_Rx = sum(r.get("RX_kN", 0) for r in reactions)
    total_Ry = sum(r.get("RY_kN", 0) for r in reactions)

    if case_name in ["DL", "LL"]:
        # 중력하중 검증: floor_area 총합 vs 반력합
        loads = load_cases.get(case_name, [])
        total_applied = 0.0
        for lc in loads:
            if lc.get("type") == "floor_area":
                total_applied += lc.get("value", lc.get("w_area_kNm2", 0)) * model.floor_area
        if total_applied > 0:
            ratio = abs(total_Rz) / total_applied
            status = "PASS" if abs(ratio - 1.0) < 0.01 else "FAIL"
            if status == "FAIL":
                ok = False
            print(f"  {case_name}: Rz={total_Rz:.1f} kN / Applied={total_applied:.1f} kN = {ratio:.6f} [{status}]")
        else:
            print(f"  {case_name}: Rz={total_Rz:.1f} kN (self-weight)")

    elif case_name in ["EQX", "EQY"]:
        # 횡하중 검증: 반력합 = 총 지진력
        V_kN = reports.get("seismic", {}).get("V_kN", 0)
        if "X" in case_name:
            ratio = abs(total_Rx) / V_kN if V_kN > 0 else 0
        else:
            ratio = abs(total_Ry) / V_kN if V_kN > 0 else 0
        status = "PASS" if abs(ratio - 1.0) < 0.02 else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"  {case_name}: R={abs(total_Rx):.1f}/{abs(total_Ry):.1f} kN, V={V_kN:.1f} kN, ratio={ratio:.6f} [{status}]")

    elif case_name in ["WX", "WY"]:
        wind_r = reports.get("wind", {})
        if "X" in case_name:
            applied = wind_r.get("total_Fx_kN", 0)
            R = abs(total_Rx)
        else:
            applied = wind_r.get("total_Fy_kN", 0)
            R = abs(total_Ry)
        ratio = R / applied if applied > 0 else 0
        status = "PASS" if abs(ratio - 1.0) < 0.02 else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"  {case_name}: R={R:.1f} kN / Applied={applied:.1f} kN = {ratio:.6f} [{status}]")

# Envelope
print(f"\n  --- Envelope (across all combos) ---")
best_combo = None
max_m = 0
for cname, cdata in result.combo_results.items():
    if abs(cdata.max_moment) > max_m:
        max_m = abs(cdata.max_moment)
        best_combo = cname

if best_combo:
    bc = result.combo_results[best_combo]
    print(f"  Critical combo: {best_combo}")
    print(f"    max_moment:  {bc.max_moment:.2f} kN*m (elem {bc.max_moment_element})")
    print(f"    max_axial:   {bc.max_axial:.2f} kN (elem {bc.max_axial_element})")
    print(f"    max_shear:   {bc.max_shear:.2f} kN (elem {bc.max_shear_element})")
    print(f"    max_drift_x: {bc.max_drift_x:.6f} (story {bc.max_drift_x_story})")
    print(f"    max_drift_y: {bc.max_drift_y:.6f} (story {bc.max_drift_y_story})")

# 층간변위 검토
drift_limit = 1.0 / 200.0
print(f"\n  --- Drift Check (limit=1/200={drift_limit:.5f}) ---")
max_dx_all, max_dy_all = 0, 0
for cname, cdata in result.combo_results.items():
    if cdata.max_drift_x > max_dx_all:
        max_dx_all = cdata.max_drift_x
    if cdata.max_drift_y > max_dy_all:
        max_dy_all = cdata.max_drift_y

dx_ok = max_dx_all < drift_limit
dy_ok = max_dy_all < drift_limit
print(f"  max_drift_x: {max_dx_all:.6f} [{'PASS' if dx_ok else 'FAIL - exceeds limit'}]")
print(f"  max_drift_y: {max_dy_all:.6f} [{'PASS' if dy_ok else 'FAIL - exceeds limit'}]")

if ok:
    print(f"\n>>> E2E TEST PASSED <<<")
else:
    print(f"\n>>> SOME CHECKS NEED REVIEW <<<")

# 결과 저장
output_path = "data/kds_output/ifc_e2e_result.json"
from dataclasses import asdict
save_data = {
    "source_ifc": "2\ucc28\uc2dc\ub3c4_260121.ifc",
    "model_summary": model.summary(),
    "load_summary": {
        "cases": list(load_cases.keys()),
        "num_combinations": len(combinations),
        "seismic_V_kN": reports.get("seismic", {}).get("V_kN"),
        "wind_Fx_kN": reports.get("wind", {}).get("total_Fx_kN"),
        "wind_Fy_kN": reports.get("wind", {}).get("total_Fy_kN"),
    },
    "analysis_summary": {
        "num_nodes": len(result.nodes),
        "num_elements": len(result.elements),
    },
}
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)
print(f"\nSaved to {output_path}")
