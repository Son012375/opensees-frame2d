"""Result extraction and formatting helpers for benchmark comparison.

Extracts key metrics from OpenSees results into a flat table format
suitable for side-by-side comparison with Midas Gen results.
"""
from __future__ import annotations


def extract_case1(results: dict) -> list[dict]:
    """Extract comparison metrics for Case 1 (2D Simple Beam)."""
    return [
        {"metric": "Midspan Vert. Disp.", "unit": "mm",
         "opensees": results["midspan_disp_mm"]},
        {"metric": "Midspan Moment", "unit": "kN*m",
         "opensees": results["midspan_moment_kNm"]},
        {"metric": "Reaction N1 Fy", "unit": "kN",
         "opensees": results["reaction_N1_Fy_kN"]},
        {"metric": "Reaction N3 Fy", "unit": "kN",
         "opensees": results["reaction_N3_Fy_kN"]},
        {"metric": "Reaction N1 Fx", "unit": "kN",
         "opensees": results["reaction_N1_Fx_kN"]},
    ]


def extract_case2(results: dict) -> list[dict]:
    """Extract comparison metrics for Case 2 (2D Portal Frame)."""
    return [
        {"metric": "Top Disp N3 dx", "unit": "mm",
         "opensees": results["top_disp_N3_dx_mm"]},
        {"metric": "Top Disp N4 dx", "unit": "mm",
         "opensees": results["top_disp_N4_dx_mm"]},
        {"metric": "Top Disp N3 dy", "unit": "mm",
         "opensees": results["top_disp_N3_dy_mm"]},
        {"metric": "Top Disp N4 dy", "unit": "mm",
         "opensees": results["top_disp_N4_dy_mm"]},
        {"metric": "Col1 Base Moment", "unit": "kN*m",
         "opensees": results["col1_base_moment_kNm"]},
        {"metric": "Col2 Base Moment", "unit": "kN*m",
         "opensees": results["col2_base_moment_kNm"]},
        {"metric": "Beam Moment (i-end)", "unit": "kN*m",
         "opensees": results["beam_moment_i_kNm"]},
        {"metric": "Beam Moment (j-end)", "unit": "kN*m",
         "opensees": results["beam_moment_j_kNm"]},
        {"metric": "Base Shear", "unit": "kN",
         "opensees": results["base_shear_kN"]},
        {"metric": "Reaction N1 Fx", "unit": "kN",
         "opensees": results["reaction_N1_Fx_kN"]},
        {"metric": "Reaction N1 Fy", "unit": "kN",
         "opensees": results["reaction_N1_Fy_kN"]},
        {"metric": "Reaction N1 Mz", "unit": "kN*m",
         "opensees": results["reaction_N1_Mz_kNm"]},
        {"metric": "Reaction N2 Fx", "unit": "kN",
         "opensees": results["reaction_N2_Fx_kN"]},
        {"metric": "Reaction N2 Fy", "unit": "kN",
         "opensees": results["reaction_N2_Fy_kN"]},
        {"metric": "Reaction N2 Mz", "unit": "kN*m",
         "opensees": results["reaction_N2_Mz_kNm"]},
    ]


def extract_case3(results: dict) -> list[dict]:
    """Extract comparison metrics for Case 3 (2D 3-story Frame)."""
    rows = [
        {"metric": "Roof Disp dx (avg)", "unit": "mm",
         "opensees": results["roof_disp_dx_mm"]},
        {"metric": "Roof Disp N7 dx", "unit": "mm",
         "opensees": results["roof_disp_N7_dx_mm"]},
        {"metric": "Roof Disp N8 dx", "unit": "mm",
         "opensees": results["roof_disp_N8_dx_mm"]},
    ]
    for i in range(3):
        rows.append({
            "metric": f"Story Drift {i+1}",
            "unit": "ratio",
            "opensees": results[f"story_drift_{i+1}"],
        })
    rows.extend([
        {"metric": "Base Shear", "unit": "kN",
         "opensees": results["base_shear_kN"]},
        {"metric": "Col1 Base Moment", "unit": "kN*m",
         "opensees": results["col1_base_moment_kNm"]},
        {"metric": "Col2 Base Moment", "unit": "kN*m",
         "opensees": results["col2_base_moment_kNm"]},
        {"metric": "Top Beam Moment (i)", "unit": "kN*m",
         "opensees": results["top_beam_moment_i_kNm"]},
        {"metric": "Top Beam Moment (j)", "unit": "kN*m",
         "opensees": results["top_beam_moment_j_kNm"]},
        {"metric": "Reaction N1 Fx", "unit": "kN",
         "opensees": results["reaction_N1_Fx_kN"]},
        {"metric": "Reaction N1 Fy", "unit": "kN",
         "opensees": results["reaction_N1_Fy_kN"]},
        {"metric": "Reaction N1 Mz", "unit": "kN*m",
         "opensees": results["reaction_N1_Mz_kNm"]},
        {"metric": "Reaction N2 Fx", "unit": "kN",
         "opensees": results["reaction_N2_Fx_kN"]},
        {"metric": "Reaction N2 Fy", "unit": "kN",
         "opensees": results["reaction_N2_Fy_kN"]},
        {"metric": "Reaction N2 Mz", "unit": "kN*m",
         "opensees": results["reaction_N2_Mz_kNm"]},
    ])
    return rows


def extract_case4(results: dict) -> list[dict]:
    """Extract comparison metrics for Case 4 (3D 2-story Frame)."""
    rows = [
        {"metric": "Roof Max dx", "unit": "mm",
         "opensees": results["roof_max_dx_mm"]},
        {"metric": "Roof Avg dx", "unit": "mm",
         "opensees": results["roof_avg_dx_mm"]},
        {"metric": "Story1 Avg dx", "unit": "mm",
         "opensees": results["story1_avg_dx_mm"]},
        {"metric": "Story Drift 1", "unit": "ratio",
         "opensees": results["story_drift_1"]},
        {"metric": "Story Drift 2", "unit": "ratio",
         "opensees": results["story_drift_2"]},
        {"metric": "Base Reaction X", "unit": "kN",
         "opensees": results["base_reaction_X_kN"]},
        {"metric": "Base Reaction Z", "unit": "kN",
         "opensees": results["base_reaction_Z_kN"]},
        {"metric": "Col1 Base My", "unit": "kN*m",
         "opensees": results["col1_base_My_kNm"]},
        {"metric": "Col1 Base Mz", "unit": "kN*m",
         "opensees": results["col1_base_Mz_kNm"]},
        {"metric": "Roof Beam1 My (i)", "unit": "kN*m",
         "opensees": results["roof_beam1_My_i_kNm"]},
        {"metric": "Roof Beam1 My (j)", "unit": "kN*m",
         "opensees": results["roof_beam1_My_j_kNm"]},
    ]
    # Per-node reactions
    for nid in [1, 2, 3, 4]:
        key = f"N{nid}"
        r = results["reactions"][key]
        for comp in ["Fx_kN", "Fy_kN", "Fz_kN", "Mx_kNm", "My_kNm", "Mz_kNm"]:
            unit = "kN" if comp.endswith("_kN") else "kN*m"
            rows.append({
                "metric": f"Reaction {key} {comp.replace('_kN', '').replace('_kNm', '')}",
                "unit": unit,
                "opensees": r[comp],
            })
    return rows


def extract_case5(results: dict) -> list[dict]:
    """Extract comparison metrics for Case 5 (3D 5-story P-Delta)."""
    lin = results["linear"]
    pdl = results["pdelta"]
    rows = [
        {"metric": "Linear Roof dx", "unit": "mm",
         "opensees": lin["roof_avg_dx_mm"]},
        {"metric": "PDelta Roof dx", "unit": "mm",
         "opensees": pdl["roof_avg_dx_mm"]},
        {"metric": "Disp Amplification", "unit": "ratio",
         "opensees": results["displacement_amplification"]},
    ]
    for i in range(5):
        rows.append({
            "metric": f"Linear Story Drift {i+1}",
            "unit": "ratio",
            "opensees": lin["story_drifts"][i],
        })
    for i in range(5):
        rows.append({
            "metric": f"PDelta Story Drift {i+1}",
            "unit": "ratio",
            "opensees": pdl["story_drifts"][i],
        })
    for i in range(5):
        rows.append({
            "metric": f"Drift Amp Story {i+1}",
            "unit": "ratio",
            "opensees": results["drift_amplification"][i],
        })
    rows.extend([
        {"metric": "Base Rx Linear", "unit": "kN",
         "opensees": results["base_rx_linear_kN"]},
        {"metric": "Base Rx PDelta", "unit": "kN",
         "opensees": results["base_rx_pdelta_kN"]},
        {"metric": "Base Rz Linear", "unit": "kN",
         "opensees": results["base_rz_linear_kN"]},
        {"metric": "Base Rz PDelta", "unit": "kN",
         "opensees": results["base_rz_pdelta_kN"]},
        {"metric": "Total Lateral", "unit": "kN",
         "opensees": results["total_lateral_kN"]},
        {"metric": "Total Gravity", "unit": "kN",
         "opensees": results["total_gravity_kN"]},
        {"metric": "Linear Col1 Base My", "unit": "kN*m",
         "opensees": lin["col1_base_My_kNm"]},
        {"metric": "PDelta Col1 Base My", "unit": "kN*m",
         "opensees": pdl["col1_base_My_kNm"]},
        {"metric": "Linear Roof Beam1 My (i)", "unit": "kN*m",
         "opensees": lin["roof_beam1_My_i_kNm"]},
        {"metric": "PDelta Roof Beam1 My (i)", "unit": "kN*m",
         "opensees": pdl["roof_beam1_My_i_kNm"]},
    ])
    # Per-node reactions (linear)
    for nid in [1, 2, 3, 4]:
        key = f"N{nid}"
        r = lin["reactions"][key]
        for comp in ["Fx_kN", "Fz_kN", "My_kNm"]:
            unit = "kN" if comp.endswith("_kN") else "kN*m"
            rows.append({
                "metric": f"Linear Reaction {key} {comp.replace('_kN', '').replace('_kNm', '')}",
                "unit": unit,
                "opensees": r[comp],
            })
    return rows


def extract_case1_udl(results: dict) -> list[dict]:
    """Extract comparison metrics for Case 1 UDL variant (same structure as case1)."""
    return extract_case1(results)


EXTRACTORS = {
    "case1": extract_case1,
    "case1_udl": extract_case1_udl,
    "case2": extract_case2,
    "case3": extract_case3,
    "case4": extract_case4,
    "case5": extract_case5,
}
