"""3D Frame analysis full integration test.

Test 1: 1x1 bay, 1-story, gravity only → column axial = total_load / 4
Test 2: 2x1 bay, 2-story, lateral_x → equilibrium check
"""
import json
import sys
import os

# Ensure module path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.frame_3d import analyze_frame_3d_multi


def test_1_gravity():
    """1x1 bay, 1-story, floor load only."""
    print("=" * 60)
    print("TEST 1: 1x1 bay, 1-story, gravity (floor 20 kN/m)")
    print("=" * 60)

    result = analyze_frame_3d_multi(
        stories=[3.5],
        bays_x=[6.0],
        bays_y=[6.0],
        load_cases={
            "DL": [
                {"type": "floor", "story": 1, "value": 20.0},  # kN/m
            ]
        },
        supports="fixed",
        column_section="H-300x300",
        beam_x_section="H-400x200",
        beam_y_section="H-400x200",
        material_name="SS275",
        num_elements_per_member=4,
    )

    print(f"\nGeometry: {result.num_stories} stories, "
          f"{result.num_bays_x}x{result.num_bays_y} bays")
    print(f"Elements: {result.num_elements}")
    print(f"Column section: {result.column_section}")
    print(f"Beam X section: {result.beam_x_section}")

    cr = result.case_results["DL"]
    print(f"\nMax displacements:")
    print(f"  dx = {cr.max_displacement_x:.4f} mm (node {cr.max_displacement_x_node})")
    print(f"  dy = {cr.max_displacement_y:.4f} mm (node {cr.max_displacement_y_node})")
    print(f"  dz = {cr.max_displacement_z:.4f} mm (node {cr.max_displacement_z_node})")

    print(f"\nMax forces:")
    print(f"  Moment = {cr.max_moment:.2f} kN*m (elem {cr.max_moment_element})")
    print(f"  Axial = {cr.max_axial:.2f} kN (elem {cr.max_axial_element})")
    print(f"  Shear = {cr.max_shear:.2f} kN (elem {cr.max_shear_element})")

    print(f"\nReactions ({len(cr.reactions)} base nodes):")
    total_rz = 0.0
    for r in cr.reactions:
        print(f"  Node {r['node']}: RZ={r['RZ_kN']:.2f} kN, "
              f"MX={r['MX_kNm']:.2f}, MY={r['MY_kNm']:.2f}, MZ={r['MZ_kNm']:.2f}")
        total_rz += r['RZ_kN']

    # Expected: 2 beam_x * 6m * 20 kN/m + 2 beam_y * 6m * 20 kN/m = 480 kN
    # 1x1 bay: 2 beams in X (top/bottom) + 2 beams in Y (left/right) = 4 beams
    # Total = 4 * 6.0 * 20.0 = 480 kN
    print(f"\nTotal RZ (reactions) = {total_rz:.2f} kN")
    expected = 4 * 6.0 * 20.0
    print(f"Expected total load = {expected:.2f} kN")
    print(f"Equilibrium ratio = {total_rz / expected:.6f}")

    if abs(total_rz / expected - 1.0) < 0.001:
        print("PASS: Gravity equilibrium OK")
    else:
        print("FAIL: Gravity equilibrium error!")

    return result


def test_2_lateral():
    """2x1 bay, 2-story, lateral_x load."""
    print("\n" + "=" * 60)
    print("TEST 2: 2x1 bay, 2-story, lateral_x")
    print("=" * 60)

    result = analyze_frame_3d_multi(
        stories=[3.5, 3.2],
        bays_x=[6.0, 8.0],
        bays_y=[6.0],
        load_cases={
            "EQX": [
                {"type": "lateral_x", "story": 1, "value": 50.0},
                {"type": "lateral_x", "story": 2, "value": 100.0},
            ]
        },
        supports="fixed",
        column_section="H-300x300",
        beam_x_section="H-400x200",
        beam_y_section="H-400x200",
        material_name="SS275",
        num_elements_per_member=4,
    )

    print(f"\nGeometry: {result.num_stories} stories, "
          f"{result.num_bays_x}x{result.num_bays_y} bays")
    print(f"Elements: {result.num_elements}")

    cr = result.case_results["EQX"]
    print(f"\nMax displacements:")
    print(f"  dx = {cr.max_displacement_x:.4f} mm (node {cr.max_displacement_x_node})")
    print(f"  dy = {cr.max_displacement_y:.4f} mm (node {cr.max_displacement_y_node})")

    print(f"\nStory drifts:")
    for sd in cr.story_drifts:
        print(f"  Story {sd['story']}: drift_x={sd['drift_x']:.6f}, "
              f"drift_y={sd['drift_y']:.6f}")

    print(f"\nReactions:")
    total_rx = 0.0
    for r in cr.reactions:
        print(f"  Node {r['node']}: RX={r['RX_kN']:.2f}, RY={r['RY_kN']:.2f}, "
              f"RZ={r['RZ_kN']:.2f}")
        total_rx += r['RX_kN']

    expected_rx = -(50.0 + 100.0)  # reactions are opposite to applied loads
    print(f"\nTotal RX (reactions) = {total_rx:.2f} kN")
    print(f"Expected = {expected_rx:.2f} kN")
    if abs(total_rx - expected_rx) < 0.1:
        print("PASS: Lateral equilibrium OK")
    else:
        print(f"FAIL: Lateral equilibrium error (diff={abs(total_rx - expected_rx):.4f})")

    return result


def test_3_combo():
    """1x1 bay, 1-story, gravity + lateral with load combination."""
    print("\n" + "=" * 60)
    print("TEST 3: 1x1 bay, 1-story, DL + EQX with combo")
    print("=" * 60)

    result = analyze_frame_3d_multi(
        stories=[3.5],
        bays_x=[6.0],
        bays_y=[6.0],
        load_cases={
            "DL": [{"type": "floor", "story": 1, "value": 20.0}],
            "EQX": [{"type": "lateral_x", "story": 1, "value": 80.0}],
        },
        load_combinations={
            "1.2DL+1.0EQX": {"DL": 1.2, "EQX": 1.0},
            "1.0DL": {"DL": 1.0},
        },
        supports="fixed",
        column_section="H-300x300",
        beam_x_section="H-400x200",
        beam_y_section="H-400x200",
        material_name="SS275",
        num_elements_per_member=4,
    )

    print(f"Cases: {list(result.case_results.keys())}")
    print(f"Combos: {list(result.combo_results.keys())}")

    # Check DL case
    cr_dl = result.case_results["DL"]
    cr_eq = result.case_results["EQX"]
    cr_combo = result.combo_results["1.2DL+1.0EQX"]

    print(f"\nDL: max_dz = {cr_dl.max_displacement_z:.4f} mm")
    print(f"EQX: max_dx = {cr_eq.max_displacement_x:.4f} mm")
    print(f"Combo: max_dx = {cr_combo.max_displacement_x:.4f} mm")

    # Linear superposition check: combo_dx should be 1.2*DL_dx + 1.0*EQX_dx
    # Find node with max dx in combo and check
    for nd in cr_combo.nodal_displacements:
        node_id = nd['node']
        # find same node in DL and EQX
        dl_nd = next((n for n in cr_dl.nodal_displacements if n['node'] == node_id), None)
        eq_nd = next((n for n in cr_eq.nodal_displacements if n['node'] == node_id), None)
        if dl_nd and eq_nd:
            expected_dx = 1.2 * dl_nd['dx_mm'] + 1.0 * eq_nd['dx_mm']
            if abs(expected_dx) > 0.001:
                ratio = nd['dx_mm'] / expected_dx if expected_dx != 0 else 0
                if abs(ratio - 1.0) > 0.001:
                    print(f"  Node {node_id}: combo_dx={nd['dx_mm']:.4f}, "
                          f"expected={expected_dx:.4f}, ratio={ratio:.6f}")
                    break
    else:
        print("PASS: Linear superposition verified for all nodes")

    return result


if __name__ == "__main__":
    test_1_gravity()
    test_2_lateral()
    test_3_combo()
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)
