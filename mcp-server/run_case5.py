"""Case 5 벤치마크 모델 실행 + HTML 리포트 저장 스크립트

analyze_frame_3d_multi 입력 단위:
  - stories/bays: m (내부에서 ×1000 → mm 변환)
  - floor_area: kN/m²
  - lateral_x: kN (층 전체)
  - E, A, I 등 단면: mm 계열
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# ── Supabase 없이 동작하도록 단면/재료 주입 ──
from core.section_3d import BeamSection3D, DEFAULT_SECTIONS_3D
from core.simple_beam import Material, DEFAULT_MATERIALS

# Case 5 기둥: H-428x407x20x35 (Supabase DB와 동일 값)
DEFAULT_SECTIONS_3D["H-428x407x20x35"] = BeamSection3D(
    name="H-428x407x20x35",
    A=36070, Ix=119000e4, Iy=39400e4, J=1259e4,
    h=428, b=407, tw=20, tf=35, j_source="benchmark",
)
# Case 5 보: H-400x200x8x13
DEFAULT_SECTIONS_3D["H-400x200x8x13"] = BeamSection3D(
    name="H-400x200x8x13",
    A=8412, Ix=23700e4, Iy=1740e4, J=66e4,
    h=400, b=200, tw=8, tf=13, j_source="benchmark",
)
# SS275 (E=210,000 MPa)
DEFAULT_MATERIALS["SS275"] = Material("SS275", E=210000, fy=275)

from core.frame_3d import analyze_frame_3d_multi
from core.design_check import run_design_check
from core.result_interpreter import interpret_results
from core.visualization_3d import plot_frame_3d_interactive

# ── Case 5: 5층 1x1bay 3D Frame ──
stories = [3.0] * 5             # 5층, 각 3m (단위: m)
bays_x = [6.0]                  # 1bay, 6m (단위: m)
bays_y = [6.0]                  # 1bay, 6m (단위: m)

# ── 하중 ──
# 벤치마크: 노드당 Fz=-70kN × 4노드 = -280kN/층
# 면적: 6m × 6m = 36m² → 280/36 ≈ 7.778 kN/m²
# 벤치마크 수평: 노드당 8/12/16/20/24 kN × 4노드 = 32/48/64/80/96 kN
floor_area_load = 280.0 / (6.0 * 6.0)  # ≈ 7.778 kN/m²

load_cases = {
    "Combined": [
        {"type": "floor_area", "story": s, "value": floor_area_load}
        for s in range(1, 6)
    ] + [
        {"type": "lateral_x", "story": 1, "value": 32.0},
        {"type": "lateral_x", "story": 2, "value": 48.0},
        {"type": "lateral_x", "story": 3, "value": 64.0},
        {"type": "lateral_x", "story": 4, "value": 80.0},
        {"type": "lateral_x", "story": 5, "value": 96.0},
    ],
}

load_combinations = {
    "G+L": {"Combined": 1.0},
}

# ── 해석 (Linear + Modal) ──
print("=== Case 5 해석 시작 ===")
multi = analyze_frame_3d_multi(
    stories=stories,
    bays_x=bays_x,
    bays_y=bays_y,
    column_section="H-428x407x20x35",
    beam_x_section="H-400x200x8x13",
    beam_y_section="H-400x200x8x13",
    material_name="SS275",
    supports="fixed",
    load_cases=load_cases,
    load_combinations=load_combinations,
    geometric_nonlinearity="pdelta",
    num_elements_per_member=1,
    modal_analysis=True,
    story_weights_kN=[280.0] * 5,
)
print(f"  요소 수: {multi.num_elements}")
print(f"  E_MPa: {multi.E_MPa}, fy_MPa: {multi.fy_MPa}")
print(f"  기둥: A={multi.column_A_mm2}, Ix={multi.column_Ix_mm4}")
print(f"  해석 모드: {multi.analysis_metadata.get('actual_transf', 'N/A')}")

# 모달 결과
if multi.modal_analysis:
    ma = multi.modal_analysis
    print(f"\n  모달 해석: {len(ma.get('modes', []))}개 모드")
    for m in ma.get("modes", [])[:5]:
        print(f"    Mode {m['mode']}: T={m['period_s']:.4f}s, dir={m['direction']}")

# ── 설계검토 ──
print("\n=== Design Check ===")
seismic_report = {"Cd": 3.0, "IE": 1.0}
try:
    dc_result = run_design_check(multi, building_model=None, seismic_report=seismic_report)
    drift_status = dc_result.get("drift_check", {}).get("status", "N/A")
    member_status = dc_result.get("member_check", {}).get("status", "N/A")
    print(f"  Drift: {drift_status}")
    print(f"  Member: {member_status}")
    if dc_result.get("member_check"):
        mc = dc_result["member_check"]
        worst = max(mc.get("members", [{}]), key=lambda m: m.get("interaction_ratio", 0))
        print(f"  Max interaction: {worst.get('interaction_ratio', 0):.3f} "
              f"({worst.get('member_type')} #{worst.get('member_id')}, story {worst.get('story')})")
except Exception as e:
    print(f"  Design check error: {e}")
    import traceback; traceback.print_exc()
    dc_result = None

# ── 결과 해석 ──
print("\n=== Interpretation ===")
interpretation = None
if dc_result is not None:
    try:
        interpretation = interpret_results(
            dc_result, multi,
            modal_analysis=multi.modal_analysis or None,
        )
        print(f"  Severity: {interpretation['severity']}")
        print(f"  요약(KO): {interpretation['summary_ko']}")
        for f in interpretation.get("findings", []):
            print(f"  [{f['code']}] {f['message_ko']}")
    except Exception as e:
        print(f"  Interpretation error: {e}")
        import traceback; traceback.print_exc()

# ── HTML 리포트 저장 ──
output_path = os.path.join(os.path.dirname(__file__), "case5_report.html")
html_path = plot_frame_3d_interactive(
    multi,
    design_check=dc_result,
    interpretation=interpretation,
    output_path=output_path,
)
print(f"\n=== HTML 리포트 저장 완료 ===")
print(f"  경로: {html_path}")
print(f"\n브라우저에서 열어보세요:")
print(f"  file:///{html_path.replace(os.sep, '/')}")
