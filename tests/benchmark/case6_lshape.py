"""Case 6 — L-shape decision-gate validation runner.

Reuses build_l_shape_model() from tests/test_v2_irregular.py and applies the
exact load vectors specified in scripts/_case6_lshape_spec.md.

Four independent linear static cases (no combinations):
    DL  = 5.1 kN/m² floor area (Zone A all 5F, Zone B 1F-3F only)
    LL  = 2.5 kN/m² floor area (same coverage)
    EQX = 500 kN total base shear, inverted-triangle, applied as per-node Fx on slaves
    EQY = same, applied as Fy

Outputs:
    - flat metric dict (consumed by extract_case6_lshape in extract.py)
    - tests/benchmark/case6_lshape_loadtables.json — per-element DL/LL line loads
      and per-node EQX/EQY forces + auto-picked diaphragm masters, for the
      Midas Gen handoff doc (Step 3).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Path setup: add mcp-server and tests/ to import path
_THIS = Path(__file__).resolve()
_REPO = _THIS.parent.parent.parent
sys.path.insert(0, str(_REPO / "mcp-server"))
sys.path.insert(0, str(_REPO / "tests"))

import core.frame_3d as _f3d
from core.frame_3d import analyze_from_model, _estimate_tributary_width
from core.simple_beam import (
    get_material_from_db as _orig_get_material_from_db,
    DEFAULT_MATERIALS,
    Material,
)
from core.structural_model import SupportType  # noqa: F401 (used in builder)
from test_v2_irregular import build_l_shape_model


# ── Benchmark E unification (2026-06) ─────────────────────────────────────────
# Cases 1-5 (tests/benchmark/cases.py) explicitly use E = 210 GPa so the
# OpenSeesPy model matches the Young's modulus of the commercial reference (Midas
# Gen / ETABS) models. SS275's KS / application-wide default is 205 GPa, which is
# what Case 6 used previously. _activate_E210() (called inside run_case6_lshape)
# forces Case 6 onto the same 210 GPa as the rest of the suite. It is applied at
# RUN time, not import time, so merely importing this module (e.g. via cases.py)
# does not mutate the shared material globals. (The §4.3 L-shape demonstration
# itself reports only E-independent gravity-equilibrium quantities.)
_SS275_E210 = Material("SS275", E=210000, fy=275)


def get_material_from_db(name):
    """Module-local override: force SS275 to the 210 GPa benchmark value."""
    return _SS275_E210 if name == "SS275" else _orig_get_material_from_db(name)


def _activate_E210():
    """Point frame_3d's material resolution at the 210 GPa SS275 (run-time only)."""
    _f3d.get_material_from_db = get_material_from_db
    DEFAULT_MATERIALS["SS275"] = _SS275_E210


# ── Spec constants (mirrors scripts/_case6_lshape_spec.md) ──
STORY_HEIGHT_M = 3.5
NUM_STORIES = 5
DL_KNM2 = 5.1
LL_KNM2 = 2.5
V_BASE_KN = 500.0  # total base shear for EQX & EQY (each independent)

# Diagnostic node coordinates (from spec §1.1)
NODE_AT = {
    "zone_a_corner_base": (0.0, 0.0, 0.0),       # node 1
    "shared_boundary_base": (12.0, 0.0, 0.0),    # node 13
    "zone_b_far_base": (24.0, 0.0, 0.0),         # node 59
    "zone_a_far_5f": (0.0, 8.0, 17.5),           # node 42
    "zone_b_re_entrant_3f": (24.0, 4.0, 10.5),   # node 70 — Zone B far-corner roof
    "zone_b_far_3f": (24.0, 0.0, 10.5),          # node 62 — secondary diagnostic
    "setback_boundary_3f": (12.0, 4.0, 10.5),    # node 34
    "setback_boundary_4f": (12.0, 4.0, 14.0),    # node 35 — upper end of 3F→4F column
}


# ── Helpers ──

def _node_id_at(model, x, y, z, tol=0.05):
    for n in model.nodes.values():
        if abs(n.x - x) < tol and abs(n.y - y) < tol and abs(n.z - z) < tol:
            return n.id
    return None


def _column_element_at(model, x, y, z_bot, z_top, tol=0.05):
    """Find element ID of the column from (x,y,z_bot) to (x,y,z_top)."""
    ni = _node_id_at(model, x, y, z_bot, tol)
    nj = _node_id_at(model, x, y, z_top, tol)
    if ni is None or nj is None:
        return None
    for e in model.elements.values():
        endpoints = (e.node_i, e.node_j)
        if endpoints == (ni, nj) or endpoints == (nj, ni):
            return e.id
    return None


def _story_node_ids(model, story_idx):
    return sorted([n.id for n in model.nodes.values() if n.story == story_idx])


def _resolved_material(name="SS275"):
    mat = get_material_from_db(name) or DEFAULT_MATERIALS.get(name)
    return mat


def _build_member_info(model):
    """Re-derive (member_id → info) the same way analyze_from_model does, so we
    can replay _estimate_tributary_width and report w_line per element.
    Returns list aligned with member_id = index+1.
    """
    member_info = []
    for e in model.elements.values():
        ni = model.nodes.get(e.node_i)
        nj = model.nodes.get(e.node_j)
        if ni is None or nj is None:
            continue
        dx = abs(nj.x - ni.x)
        dy = abs(nj.y - ni.y)
        dz = abs(nj.z - ni.z)
        if dz > max(dx, dy):
            mtype = "column"
        elif dx >= dy:
            mtype = "beam_x"
        else:
            mtype = "beam_y"
        member_info.append({
            "elem_id": e.id,
            "ni": e.node_i, "nj": e.node_j,
            "type": mtype,
            "x_i": ni.x, "y_i": ni.y, "z_i": ni.z,
            "x_j": nj.x, "y_j": nj.y, "z_j": nj.z,
        })
    return member_info


def _compute_line_load_table(model, w_area_kNm2):
    """Replicate the V2 tributary distribution and return per-element w_line (kN/m).

    Matches the logic in [_apply_loads_v2()](mcp-server/core/frame_3d.py#L2273):
        w_line = 0.5 * w_area * trib_w  (for both beam_x and beam_y, 2-way slab)
    """
    member_info = _build_member_info(model)
    table = []
    for mi, minfo in enumerate(member_info, start=1):
        if minfo["type"] not in ("beam_x", "beam_y"):
            continue
        ni = model.nodes[minfo["ni"]]
        if ni.story is None:
            continue
        # _estimate_tributary_width signature: (model, minfo_dict, member_id)
        trib_w = _estimate_tributary_width(model, minfo, mi)
        w_line = 0.5 * w_area_kNm2 * trib_w  # kN/m
        length_m = (
            (minfo["x_j"] - minfo["x_i"]) ** 2
            + (minfo["y_j"] - minfo["y_i"]) ** 2
            + (minfo["z_j"] - minfo["z_i"]) ** 2
        ) ** 0.5
        table.append({
            "elem_id": minfo["elem_id"],
            "ni": minfo["ni"], "nj": minfo["nj"],
            "type": minfo["type"],
            "story": ni.story,
            "x_i": round(minfo["x_i"], 3), "y_i": round(minfo["y_i"], 3), "z_i": round(minfo["z_i"], 3),
            "x_j": round(minfo["x_j"], 3), "y_j": round(minfo["y_j"], 3), "z_j": round(minfo["z_j"], 3),
            "length_m": round(length_m, 4),
            "trib_w_m": round(trib_w, 4),
            "w_line_kNm": round(w_line, 6),
            "total_kN": round(w_line * length_m, 4),
        })
    return table


def _compute_lateral_table(model, v_base_kN):
    """Return per-story per-node lateral force vector (inverted triangle)."""
    z_levels = [STORY_HEIGHT_M * s for s in range(1, NUM_STORIES + 1)]
    sum_h = sum(z_levels)
    table = []
    for s, hi in enumerate(z_levels, start=1):
        Fi_total = v_base_kN * hi / sum_h
        snodes = _story_node_ids(model, s)
        per_node = Fi_total / len(snodes) if snodes else 0.0
        table.append({
            "story": s,
            "z_m": hi,
            "Fi_total_kN": round(Fi_total, 4),
            "n_nodes": len(snodes),
            "per_node_kN": round(per_node, 6),
            "node_ids": snodes,
        })
    return table


def _pick_master_per_story(model):
    """Mirror [frame_3d.py:510](mcp-server/core/frame_3d.py#L510) master pick:
        master = snodes_sorted[len(snodes)//2]
    """
    masters = {}
    for s in range(1, NUM_STORIES + 1):
        snodes = _story_node_ids(model, s)
        if not snodes:
            continue
        masters[s] = snodes[len(snodes) // 2]
    return masters


# ── Main runner ──

def run_case6_lshape() -> dict:
    """Build L-shape model, run 4 independent linear cases, return metric dict."""
    _activate_E210()  # unify Case 6 Young's modulus with Cases 1-5 (210 GPa)
    m = build_l_shape_model()
    m.rigid_diaphragm = True
    m.geometric_nonlinearity = "linear"

    # Resolve material (for export to handoff doc)
    mat = _resolved_material("SS275")
    E_MPa = mat.E if mat is not None else float("nan")

    # Locate diagnostic node IDs
    n_zone_a_corner_base = _node_id_at(m, *NODE_AT["zone_a_corner_base"])
    n_shared_boundary_base = _node_id_at(m, *NODE_AT["shared_boundary_base"])
    n_zone_b_far_base = _node_id_at(m, *NODE_AT["zone_b_far_base"])
    n_zone_a_far_5f = _node_id_at(m, *NODE_AT["zone_a_far_5f"])
    n_zone_b_far_3f = _node_id_at(m, *NODE_AT["zone_b_far_3f"])
    n_zone_b_re_entrant_3f = _node_id_at(m, *NODE_AT["zone_b_re_entrant_3f"])

    # Locate diagnostic column elements
    col_corner_1f = _column_element_at(m, 0.0, 0.0, 0.0, 3.5)   # base→1F at (0,0)
    col_setback_3f_to_4f = _column_element_at(m, 12.0, 4.0, 10.5, 14.0)

    # Build load cases per spec §7
    DL_loads = [{"type": "floor_area", "story": s, "value": DL_KNM2}
                for s in range(1, NUM_STORIES + 1)]
    LL_loads = [{"type": "floor_area", "story": s, "value": LL_KNM2}
                for s in range(1, NUM_STORIES + 1)]

    z_levels = [STORY_HEIGHT_M * s for s in range(1, NUM_STORIES + 1)]
    sum_h = sum(z_levels)
    EQX_loads = [{"type": "lateral_x", "story": s, "value": V_BASE_KN * z_levels[s - 1] / sum_h}
                 for s in range(1, NUM_STORIES + 1)]
    EQY_loads = [{"type": "lateral_y", "story": s, "value": V_BASE_KN * z_levels[s - 1] / sum_h}
                 for s in range(1, NUM_STORIES + 1)]

    load_cases = {"DL": DL_loads, "LL": LL_loads, "EQX": EQX_loads, "EQY": EQY_loads}

    # Run
    result = analyze_from_model(m, load_cases)

    # ── Export load tables + master picks for Midas handoff (Step 3) ──
    dl_line_loads = _compute_line_load_table(m, DL_KNM2)
    ll_line_loads = _compute_line_load_table(m, LL_KNM2)
    lateral_table = _compute_lateral_table(m, V_BASE_KN)
    masters = _pick_master_per_story(m)

    handoff = {
        "spec_version": "_case6_lshape_spec.md v1",
        "material": {"name": "SS275", "E_MPa_used": E_MPa,
                     "note": "Resolved via get_material_from_db → DEFAULT_MATERIALS fallback. "
                             "Midas modeler must use this E exactly."},
        "geometry_check": {
            "n_nodes": len(m.nodes),
            "n_elements": len(m.elements),
            "n_base_fixed": sum(1 for n in m.nodes.values() if n.z < 0.01),
            "stories_detected": m.num_stories,
        },
        "rigid_diaphragm_master_per_story": masters,
        "diagnostic_nodes": {
            "zone_a_corner_base (0,0,0)": n_zone_a_corner_base,
            "shared_boundary_base (12,0,0)": n_shared_boundary_base,
            "zone_b_far_base (24,0,0)": n_zone_b_far_base,
            "zone_a_far_5f (0,8,17.5)": n_zone_a_far_5f,
            "zone_b_far_3f (24,0,10.5)": n_zone_b_far_3f,
            "zone_b_re_entrant_3f (24,4,10.5)": n_zone_b_re_entrant_3f,
        },
        "diagnostic_columns": {
            "corner_1f (0,0 z=0→3.5)": col_corner_1f,
            "setback_3F_to_4F (12,4 z=10.5→14.0)": col_setback_3f_to_4f,
        },
        "DL_line_loads_kNm": dl_line_loads,
        "LL_line_loads_kNm": ll_line_loads,
        "lateral_force_table_kN": lateral_table,
        "DL_line_load_total_kN": round(sum(r["total_kN"] for r in dl_line_loads), 3),
        "LL_line_load_total_kN": round(sum(r["total_kN"] for r in ll_line_loads), 3),
    }
    out_dir = Path(__file__).parent
    with open(out_dir / "case6_lshape_loadtables.json", "w", encoding="utf-8") as f:
        json.dump(handoff, f, indent=2, ensure_ascii=False)

    # ── Extract metrics for compare.py ──
    metrics = {}

    def _get_disp(case_name, nid):
        for d in result.case_results[case_name].nodal_displacements:
            if d["node"] == nid:
                return d
        return None

    def _get_reaction(case_name, nid):
        for r in result.case_results[case_name].reactions:
            if r["node"] == nid:
                return r
        return None

    def _sum_reaction(case_name, key):
        return sum(r[key] for r in result.case_results[case_name].reactions)

    def _max_drift_envelope(case_name, key):
        drifts = result.case_results[case_name].story_drifts
        if not drifts:
            return 0.0
        return max(abs(d[key]) for d in drifts)

    def _get_member_force(case_name, member_id):
        """부재-레벨 힘분포(member_forces)를 member_id로 조회.

        ⚠ case_results[*].element_forces는 **OS 세분화 요소 id(1..num_sub·N)**로 키가
        매겨진다. col_corner_1f/col_setback은 StructuralModel **부재 id**이므로 그것으로
        element_forces를 조회하면 엉뚱한 sub-element를 읽는다(과거 세트백 -29.09 유령의
        원인). member_forces[case]는 member_id로 올바르게 키되며 per-station My/Mz
        리스트(s=0..L)를 제공하므로, i단=[0]·j단=[-1]로 정확한 부재단 모멘트를 얻는다.
        """
        for mf in result.member_forces.get(case_name, []):
            if mf.get("member_id") == member_id:
                return mf
        return None

    # Gravity metrics (DL, LL)
    for case in ("DL", "LL"):
        metrics[f"{case} Base SumFz (kN)"] = round(_sum_reaction(case, "RZ_kN"), 3)
        for label, nid in (
            ("ZoneA_corner", n_zone_a_corner_base),
            ("SharedBoundary", n_shared_boundary_base),
            ("ZoneB_far", n_zone_b_far_base),
        ):
            r = _get_reaction(case, nid)
            metrics[f"{case} Reaction {label} Fz (kN)"] = round(r["RZ_kN"], 3) if r else None
        # Zone B 3F diagnostic vertical disp (node at 24,0,10.5)
        d = _get_disp(case, n_zone_b_far_3f)
        metrics[f"{case} ZoneB 3F Far Corner dz (mm)"] = round(d["dz_mm"], 4) if d else None

    # Lateral metrics — EQX
    d_eqx_a = _get_disp("EQX", n_zone_a_far_5f)
    metrics["EQX ZoneA Far 5F dx (mm)"] = round(d_eqx_a["dx_mm"], 4) if d_eqx_a else None
    metrics["EQX ZoneA Far 5F dy (mm)"] = round(d_eqx_a["dy_mm"], 4) if d_eqx_a else None
    metrics["EQX Max StoryDrift X (ratio)"] = round(_max_drift_envelope("EQX", "drift_x"), 6)
    metrics["EQX Base Shear Fx (kN)"] = round(_sum_reaction("EQX", "RX_kN"), 3)

    # Lateral metrics — EQY
    d_eqy_a = _get_disp("EQY", n_zone_a_far_5f)
    metrics["EQY ZoneA Far 5F dx (mm)"] = round(d_eqy_a["dx_mm"], 4) if d_eqy_a else None
    metrics["EQY ZoneA Far 5F dy (mm)"] = round(d_eqy_a["dy_mm"], 4) if d_eqy_a else None
    metrics["EQY Max StoryDrift Y (ratio)"] = round(_max_drift_envelope("EQY", "drift_y"), 6)
    metrics["EQY Base Shear Fy (kN)"] = round(_sum_reaction("EQY", "RY_kN"), 3)

    # Torsional diagnostic: relative disp between Zone A far corner (5F) and
    # Zone B re-entrant top corner (3F) under each EQ
    d_eqx_b = _get_disp("EQX", n_zone_b_re_entrant_3f)
    d_eqy_b = _get_disp("EQY", n_zone_b_re_entrant_3f)
    metrics["EQX Torsion (A_5F dx - B_3F dx) (mm)"] = (
        round(d_eqx_a["dx_mm"] - d_eqx_b["dx_mm"], 4)
        if d_eqx_a and d_eqx_b else None
    )
    metrics["EQY Torsion (A_5F dy - B_3F dy) (mm)"] = (
        round(d_eqy_a["dy_mm"] - d_eqy_b["dy_mm"], 4)
        if d_eqy_a and d_eqy_b else None
    )

    # Member forces (under EQX) — read member-level station arrays: i-end=[0], j-end=[-1].
    # (Columns are built i=bottom, j=top: _add_column(z_bot→z_top).)
    if col_corner_1f is not None:
        mf = _get_member_force("EQX", col_corner_1f)
        if mf and mf.get("My_kNm"):
            metrics["EQX CornerCol1F Base My (kNm)"] = round(mf["My_kNm"][0], 3)
            metrics["EQX CornerCol1F Base Mz (kNm)"] = round(mf["Mz_kNm"][0], 3)
    if col_setback_3f_to_4f is not None:
        mf = _get_member_force("EQX", col_setback_3f_to_4f)
        if mf and mf.get("My_kNm"):
            # upper end (z=14.0) = j-end = last station
            metrics["EQX SetbackCol 3F->4F UpperEnd My (kNm)"] = round(mf["My_kNm"][-1], 3)
            metrics["EQX SetbackCol 3F->4F UpperEnd Mz (kNm)"] = round(mf["Mz_kNm"][-1], 3)

    # Stash analysis metadata for post-run inspection (not a compared metric)
    metrics["_meta_n_nodes"] = len(m.nodes)
    metrics["_meta_n_elements"] = len(m.elements)
    metrics["_meta_E_MPa"] = E_MPa
    metrics["_meta_DL_line_total_kN"] = round(sum(r["total_kN"] for r in dl_line_loads), 3)

    return metrics


if __name__ == "__main__":
    import pprint
    pprint.pprint(run_case6_lshape())
