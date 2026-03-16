"""4차시도 IFC -> E2E Test.

10층 비정형 건물. 비정형 그리드이지만 파이프라인 동작 검증 목적.
"""
import sys, json
sys.path.insert(0, "mcp-server")

from core.building_model import BuildingModel
from core.load_generator import generate_all_loads
from core.frame_3d import analyze_frame_3d_multi

IFC_PATH = r"C:\Users\youm\Documents\카카오톡 받은 파일\4차시도_260121.ifc"

# ── 1. IFC -> BuildingModel ──
print("=" * 60)
print("Phase 1: IFC -> BuildingModel (4차시도)")
print("=" * 60)

config = {
    "stories": [
        {"usage": "retail", "dead_load_finish": 1.5},  # 1F
    ] + [
        {"usage": "office", "dead_load_finish": 1.0}   # 2F~10F + Roof
        for _ in range(10)
    ],
    "region": "\uc11c\uc6b8",
    "site_class": "S3",
    "importance": "II",
    "column_section": "H-400x400",
    "beam_x_section": "H-500x200",
    "beam_y_section": "H-500x200",
    "seismic_system": "ordinary_moment_frame",
    "auto_combinations": True,
    "num_elements_per_member": 2,
}

model = BuildingModel.from_ifc(IFC_PATH, config)
print(f"  Stories: {model.num_stories}")
print(f"  Heights: {model.story_heights_list}")
print(f"  bays_x: {model.bays_x}")
print(f"  bays_y: {model.bays_y}")
print(f"  Plan: {model.total_width_x}m x {model.total_width_y}m = {model.floor_area}m2")
print(f"  Total height: {model.total_height}m")

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

if "seismic" in reports:
    s = reports["seismic"]
    print(f"  Seismic: V={s.get('V_kN', 0):.1f} kN, Cs={s.get('Cs', 0):.6f}, "
          f"W={s.get('W_kN', 0):.1f} kN, T={s.get('T_sec', 0):.3f}s")
if "wind" in reports:
    w = reports["wind"]
    print(f"  Wind: Fx={w.get('total_Fx_kN', 0):.1f} kN, Fy={w.get('total_Fy_kN', 0):.1f} kN")

# ── 3. OpenSees Analysis ──
print("\n" + "=" * 60)
print("Phase 3: OpenSees 3D Frame Analysis")
print("=" * 60)

kwargs = model.to_frame3d_kwargs()
kwargs["load_cases"] = load_cases
kwargs["load_combinations"] = combinations

print(f"  Running {model.num_stories}-story analysis...")
result = analyze_frame_3d_multi(**kwargs)
print(f"  Nodes: {len(result.nodes)}, Elements: {len(result.elements)}")

# ── 4. Verification ──
print("\n" + "=" * 60)
print("Phase 4: Results")
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
        loads = load_cases.get(case_name, [])
        total_applied = sum(
            lc.get("value", 0) * model.floor_area
            for lc in loads if lc.get("type") == "floor_area"
        )
        if total_applied > 0:
            ratio = abs(total_Rz) / total_applied
            status = "PASS" if abs(ratio - 1.0) < 0.01 else "FAIL"
            if status == "FAIL":
                ok = False
            print(f"  {case_name}: Rz={total_Rz:.1f} / Applied={total_applied:.1f} = {ratio:.6f} [{status}]")
    elif case_name in ["EQX", "EQY"]:
        V = reports.get("seismic", {}).get("V_kN", 0)
        R = abs(total_Rx) if "X" in case_name else abs(total_Ry)
        ratio = R / V if V > 0 else 0
        status = "PASS" if abs(ratio - 1.0) < 0.02 else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"  {case_name}: R={R:.1f} / V={V:.1f} = {ratio:.6f} [{status}]")
    elif case_name in ["WX", "WY"]:
        wr = reports.get("wind", {})
        applied = wr.get("total_Fx_kN", 0) if "X" in case_name else wr.get("total_Fy_kN", 0)
        R = abs(total_Rx) if "X" in case_name else abs(total_Ry)
        ratio = R / applied if applied > 0 else 0
        status = "PASS" if abs(ratio - 1.0) < 0.02 else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"  {case_name}: R={R:.1f} / Applied={applied:.1f} = {ratio:.6f} [{status}]")

# Envelope
print(f"\n  --- Envelope ---")
max_m, best = 0, None
for cn, cd in result.combo_results.items():
    if abs(cd.max_moment) > max_m:
        max_m = abs(cd.max_moment)
        best = cn
if best:
    bc = result.combo_results[best]
    print(f"  Critical: {best}")
    print(f"    M={bc.max_moment:.1f} kN*m, N={bc.max_axial:.1f} kN, "
          f"V={bc.max_shear:.1f} kN, T={bc.max_torsion:.1f} kN*m")
    print(f"    drift_x={bc.max_drift_x:.6f} (story {bc.max_drift_x_story})")
    print(f"    drift_y={bc.max_drift_y:.6f} (story {bc.max_drift_y_story})")

# Drift check
drift_limit = 1.0 / 200.0
max_dx = max(cd.max_drift_x for cd in result.combo_results.values())
max_dy = max(cd.max_drift_y for cd in result.combo_results.values())
print(f"\n  max_drift_x: {max_dx:.6f} {'PASS' if max_dx < drift_limit else 'OVER LIMIT'}")
print(f"  max_drift_y: {max_dy:.6f} {'PASS' if max_dy < drift_limit else 'OVER LIMIT'}")

print(f"\n>>> {'E2E PASSED' if ok else 'SOME CHECKS NEED REVIEW'} <<<")
