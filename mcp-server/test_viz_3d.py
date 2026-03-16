"""3D 시각화 기능 테스트 스크립트"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from core.frame_3d import analyze_frame_3d_multi
from core.visualization_3d import plot_frame_3d_interactive, prepare_3d_viz_data

# 2층 2x2bay 간단 테스트
stories = [4.0, 4.0]
bays_x = [6.0, 6.0]
bays_y = [6.0, 6.0]

load_cases = {
    "DL": [{"type": "floor_area", "value": 5.0, "story": 1},
           {"type": "floor_area", "value": 5.0, "story": 2}],
    "LL": [{"type": "floor_area", "value": 2.5, "story": 1},
           {"type": "floor_area", "value": 2.5, "story": 2}],
    "EQX": [{"type": "lateral_x", "value": 50.0, "story": 1},
            {"type": "lateral_x", "value": 100.0, "story": 2}],
}

load_combinations = {
    "1.2DL+1.6LL": {"DL": 1.2, "LL": 1.6},
    "1.2DL+1.0EQX": {"DL": 1.2, "EQX": 1.0},
}

print("Running 3D frame analysis...")
multi = analyze_frame_3d_multi(
    stories=stories,
    bays_x=bays_x,
    bays_y=bays_y,
    load_cases=load_cases,
    column_section="H-300x300",
    beam_x_section="H-400x200",
    beam_y_section="H-400x200",
    material_name="SS275",
    load_combinations=load_combinations,
)
print(f"Analysis complete: {len(multi.nodes)} nodes, {multi.num_elements} elements")

# Test data preparation
print("\nTesting prepare_3d_viz_data()...")
data = prepare_3d_viz_data(multi)
print(f"  node_map keys: {len(data['node_map'])}")
print(f"  members: col={len(data['members']['column'])}, bx={len(data['members']['beam_x'])}, by={len(data['members']['beam_y'])}")
print(f"  support_ids: {len(data['support_ids'])}")
print(f"  cases: {data['case_names']}")
print(f"  combos: {data['combo_names']}")

# Test HTML generation
print("\nGenerating HTML report...")
output_path = os.path.join(os.path.dirname(__file__), "test_3d_report.html")
html_path = plot_frame_3d_interactive(multi, output_path=output_path)
print(f"HTML report saved to: {html_path}")

file_size = os.path.getsize(html_path)
print(f"File size: {file_size / 1024:.1f} KB")

# Show some results
for case_name in data["all_names"]:
    cd = data["case_data"][case_name]
    mv = cd["max"]
    print(f"\n  [{case_name}]")
    print(f"    Max Disp X: {mv['disp_x']:.3f} mm (Node {mv['disp_x_node']})")
    print(f"    Max Disp Y: {mv['disp_y']:.3f} mm (Node {mv['disp_y_node']})")
    print(f"    Max Drift X: {mv['drift_x']:.6f}")
    if mv['drift_x'] > 0:
        print(f"    → 1/{1/mv['drift_x']:.0f}")

print("\nDone! Open the HTML file in a browser to verify.")
