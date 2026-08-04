"""G3 / E2 exporter — freeze the §4 IFC application example as a generic
node-element + per-case load model that an ETABS runner can rebuild verbatim.

Why this exists
---------------
The paper's §4 application example (3-story, 3x2 bay, 348 analysis elements /
87 physical members, rigid_diaphragm=false) is the model whose response
numbers appear in the manuscript (`example_section4_results.json`).  To close
reviewer gap G3 (no commercial cross-validation of the IFC pipeline) we rebuild
*exactly this model and its loads* in ETABS and compare the global response.

To make the ETABS comparison isolate **model + analysis** (not load generation)
we must feed ETABS the SAME loads OpenSees applied.  Rather than re-deriving the
KDS tributary line loads, this script **captures** them: it wraps the OpenSees
`ops` object with a recording proxy and runs the real `analyze_frame_3d_multi`,
recording every `eleLoad` (beam line load) and `load` (nodal lateral) call,
bucketed per load case.  The captured loads, the physical-member node-element
model, and the per-case OpenSees global response are written to three JSON files
under docs/.../validation/ for the ETABS runner.

Outputs (validation/):
  - ifc_example_model.json       node coords, 87 physical members, sections, supports
  - ifc_example_loadtables.json  per-case beam line loads + nodal lateral loads
  - ifc_example_os_response.json  per-case nodal disp/reactions, combos, envelope

Usage:
    python scripts/_export_ifc_example_for_etabs.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mcp-server"))

import core.frame_3d as f3d                      # noqa: E402
from core.building_model import BuildingModel    # noqa: E402
from core.load_generator import generate_all_loads  # noqa: E402
from core.frame_3d import analyze_frame_3d_multi  # noqa: E402

VAL = ROOT / "docs" / "paper1_open_source_alternative" / "validation"
INPUT = VAL / "example_input.json"
REF = VAL / "example_section4_results.json"

MODEL_OUT = VAL / "ifc_example_model.json"
LOADS_OUT = VAL / "ifc_example_loadtables.json"
RESP_OUT = VAL / "ifc_example_os_response.json"


# ─────────────────────────────────────────────────────────────────────────
# Recording proxy around core.ops_compat.ops
# ─────────────────────────────────────────────────────────────────────────

class _RecordingOps:
    """Transparent proxy: forwards everything to the real ops object, but
    records eleLoad / load calls into the current load-case bucket.

    frame_3d looks up ``ops`` in its module globals at call time, so replacing
    ``f3d.ops`` with this proxy intercepts the two load-application calls while
    leaving model build / solve / result extraction untouched.
    """

    def __init__(self, real, state, buckets):
        self._real = real
        self._state = state        # {"case": <name or None>}
        self._buckets = buckets    # {case_name: {"ele": [...], "nodal": [...]}}

    def eleLoad(self, *args, **kwargs):
        case = self._state["case"]
        if case is not None:
            # ('-ele', eid, '-type', '-beamUniform', Wy, Wz, Wx)
            try:
                eid = int(args[1])
                Wz = float(args[5])
                self._buckets[case]["ele"].append((eid, Wz))
            except (IndexError, ValueError, TypeError):
                pass
        return self._real.eleLoad(*args, **kwargs)

    def load(self, *args, **kwargs):
        case = self._state["case"]
        # Skip nodal loads emitted by _apply_beam_udl (gravity consistent-nodal);
        # those are captured as per-element beam UDLs, not as lateral nodal loads.
        if case is not None and not self._state.get("in_udl"):
            # (nid, fx, fy, fz, mx, my, mz)  — N, N·mm
            try:
                nid = int(args[0])
                comps = [float(v) for v in args[1:7]]
                self._buckets[case]["nodal"].append((nid, comps))
            except (IndexError, ValueError, TypeError):
                pass
        return self._real.load(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._real, name)


def _capture_loads(kwargs, load_cases, load_combinations):
    """Run analyze_frame_3d_multi with load capture. Returns (multi, buckets).

    Gravity distributed loads are applied by frame_3d via `_apply_beam_udl`
    (consistent-nodal, post the 2026-07-01 eleLoad-recovery fix), so we capture
    the per-element beam UDL there. Lateral cases still use `ops.load`; we record
    those via the ops proxy but skip the nodal loads that `_apply_beam_udl` itself
    emits (flagged by `state['in_udl']`).
    """
    case_seq = list(load_cases.keys())
    state = {"case": None, "in_udl": False}
    buckets = {name: {"ele": [], "nodal": []} for name in case_seq}

    real_ops = f3d.ops
    proxy = _RecordingOps(real_ops, state, buckets)

    orig_apply = f3d._apply_loads_3d
    orig_apply_irr = f3d._apply_loads_3d_irregular
    orig_udl = f3d._apply_beam_udl
    counter = {"idx": -1}

    def wrapped_apply(loads, *a, **k):
        counter["idx"] += 1
        state["case"] = case_seq[counter["idx"]] if counter["idx"] < len(case_seq) else None
        try:
            return orig_apply(loads, *a, **k)
        finally:
            state["case"] = None

    def wrapped_apply_irr(loads, *a, **k):
        counter["idx"] += 1
        state["case"] = case_seq[counter["idx"]] if counter["idx"] < len(case_seq) else None
        try:
            return orig_apply_irr(loads, *a, **k)
        finally:
            state["case"] = None

    def wrapped_udl(eid, w_Nmm, etype):
        case = state["case"]
        if case is not None and w_Nmm:
            # Store as (eid, Wz) with Wz = -w to match the old eleLoad capture
            # format consumed by _build_load_tables (w_line = |Wz|).
            buckets[case]["ele"].append((int(eid), -float(w_Nmm)))
        state["in_udl"] = True
        try:
            return orig_udl(eid, w_Nmm, etype)
        finally:
            state["in_udl"] = False

    f3d.ops = proxy
    f3d._apply_loads_3d = wrapped_apply
    f3d._apply_loads_3d_irregular = wrapped_apply_irr
    f3d._apply_beam_udl = wrapped_udl
    try:
        multi = analyze_frame_3d_multi(
            load_cases=load_cases,
            load_combinations=load_combinations,
            modal_analysis=False,
            **kwargs,
        )
    finally:
        f3d.ops = real_ops
        f3d._apply_loads_3d = orig_apply
        f3d._apply_loads_3d_irregular = orig_apply_irr
        f3d._apply_beam_udl = orig_udl
    return multi, buckets


# ─────────────────────────────────────────────────────────────────────────
# Assembly of the export payloads
# ─────────────────────────────────────────────────────────────────────────

def _node_coord_map(multi) -> dict:
    return {n["id"]: (n["x_m"], n["y_m"], n["z_m"]) for n in multi.nodes}


def _elem_to_member(multi) -> dict:
    """element_id -> member_info dict."""
    e2m = {}
    for m in multi.member_info:
        for eid in m["element_ids"]:
            e2m[eid] = m
    return e2m


def _build_load_tables(multi, buckets) -> dict:
    coord = _node_coord_map(multi)
    e2m = _elem_to_member(multi)
    tables = {}
    for case, rec in buckets.items():
        # Beam line loads: collapse per member (all sub-elements share w).
        per_member = {}
        for eid, Wz in rec["ele"]:
            m = e2m.get(eid)
            if m is None:
                continue
            # w_line (kN/m) == |Wz| (N/mm) numerically; sign convention: Wz<0 = gravity.
            per_member[m["member_id"]] = (m, abs(Wz))
        line_loads = []
        for mid, (m, w_line) in sorted(per_member.items()):
            xi, yi, zi = coord[m["ni"]]
            xj, yj, zj = coord[m["nj"]]
            line_loads.append({
                "member_id": mid, "type": m["type"], "section": m["section"],
                "x_i": xi, "y_i": yi, "z_i": zi,
                "x_j": xj, "y_j": yj, "z_j": zj,
                "w_line_kNm": round(w_line, 6),
            })
        # Nodal loads (lateral): N -> kN, N·mm -> kN·m.
        nodal_loads = []
        for nid, comps in rec["nodal"]:
            fx, fy, fz, mx, my, mz = comps
            x, y, z = coord[nid]
            nodal_loads.append({
                "node": nid, "x": x, "y": y, "z": z,
                "Fx_kN": round(fx / 1000.0, 6), "Fy_kN": round(fy / 1000.0, 6),
                "Fz_kN": round(fz / 1000.0, 6),
                "Mx_kNm": round(mx / 1e6, 6), "My_kNm": round(my / 1e6, 6),
                "Mz_kNm": round(mz / 1e6, 6),
            })
        tables[case] = {"line_loads": line_loads, "nodal_loads": nodal_loads}
    return tables


def _build_model_payload(multi, base_nodes, config) -> dict:
    members = [
        {"member_id": m["member_id"], "type": m["type"], "section": m["section"],
         "ni": m["ni"], "nj": m["nj"], "story": m.get("story"),
         "length_m": m["length_m"]}
        for m in multi.member_info
    ]
    sections = {
        multi.column_section: {
            "A_mm2": multi.column_A_mm2, "Ix_mm4": multi.column_Ix_mm4,
            "Iy_mm4": multi.column_Iy_mm4, "J_mm4": multi.column_J_mm4,
            "h_mm": multi.column_h_mm, "b_mm": multi.column_b_mm,
            "tw_mm": multi.column_tw_mm, "tf_mm": multi.column_tf_mm,
        },
        multi.beam_x_section: {
            "A_mm2": multi.beam_x_A_mm2, "Ix_mm4": multi.beam_x_Ix_mm4,
            "Iy_mm4": multi.beam_x_Iy_mm4, "J_mm4": multi.beam_x_J_mm4,
            "h_mm": multi.beam_x_h_mm, "b_mm": multi.beam_x_b_mm,
            "tw_mm": multi.beam_x_tw_mm, "tf_mm": multi.beam_x_tf_mm,
        },
    }
    return {
        "_meta": {
            "purpose": "G3/E2 — §4 IFC application example frozen as generic "
                       "node-element model for ETABS cross-validation",
            "source_input": str(INPUT.relative_to(ROOT)),
            "num_analysis_elements": multi.num_elements,
            "num_physical_members": len(members),
            "E_MPa": multi.E_MPa,
            "material": multi.material_name,
            "rigid_diaphragm": bool(config.get("rigid_diaphragm", False)),
            "supports": multi.supports,
        },
        "stories": multi.stories,
        "bays_x": multi.bays_x,
        "bays_y": multi.bays_y,
        "nodes": [{"id": n["id"], "x": n["x_m"], "y": n["y_m"], "z": n["z_m"]}
                  for n in multi.nodes],
        "members": members,
        "base_nodes": sorted(base_nodes),
        "sections": sections,
        "section_map": {
            "column": multi.column_section,
            "beam_x": multi.beam_x_section,
            "beam_y": multi.beam_y_section,
        },
    }


def _story_nodes_by_z(multi) -> dict:
    """{z_level_index: [node ids]} grouped by rounded z, index 0=base."""
    levels = sorted({round(n["z_m"], 3) for n in multi.nodes})
    zidx = {z: i for i, z in enumerate(levels)}
    groups = {i: [] for i in range(len(levels))}
    for n in multi.nodes:
        groups[zidx[round(n["z_m"], 3)]].append(n["id"])
    return {"z_levels_m": levels, "story_nodes": groups}


def _build_response_payload(multi) -> dict:
    per_case = {}
    for cn, cr in multi.case_results.items():
        per_case[cn] = {
            "nodal": [
                {"node": d["node"], "x": d["x_m"], "y": d["y_m"], "z": d["z_m"],
                 "dx_mm": d["dx_mm"], "dy_mm": d["dy_mm"], "dz_mm": d["dz_mm"]}
                for d in cr.nodal_displacements
            ],
            "reactions": [
                {"node": r["node"], "x": r["x_m"], "y": r["y_m"],
                 "RX_kN": r["RX_kN"], "RY_kN": r["RY_kN"], "RZ_kN": r["RZ_kN"]}
                for r in cr.reactions
            ],
            "max_dx_mm": cr.max_displacement_x,
            "max_dy_mm": cr.max_displacement_y,
            "story_drifts": cr.story_drifts,
        }
    per_combo = {}
    for cn, cr in multi.combo_results.items():
        per_combo[cn] = {
            "max_dx_mm": round(cr.max_displacement_x, 4),
            "max_dy_mm": round(cr.max_displacement_y, 4),
            "max_drift_x": round(cr.max_drift_x, 6),
            "max_drift_y": round(cr.max_drift_y, 6),
            "story_drifts": cr.story_drifts,
        }
    return {
        "load_combinations": multi.load_combinations,
        "story_grouping": _story_nodes_by_z(multi),
        "per_case": per_case,
        "per_combo": per_combo,
    }


# ─────────────────────────────────────────────────────────────────────────
# Sanity checks (fail loud if capture is wrong)
# ─────────────────────────────────────────────────────────────────────────

def _sanity(multi, tables, config):
    print("\n" + "=" * 72)
    print("SANITY CHECKS")
    print("=" * 72)
    print(f"  Load cases captured: {list(tables.keys())}")
    print(f"  Analysis elements:   {multi.num_elements} (expect 348)")
    print(f"  Physical members:    {len(multi.member_info)} (expect 87)")
    print(f"  Nodes:               {len(multi.nodes)} (expect 48)")

    floor_area = sum(multi.bays_x) * sum(multi.bays_y)  # m² per floor
    for case in ("DL", "LL"):
        if case not in tables:
            continue
        # Σ(w_line × L) should equal Σ RZ reactions (total vertical load).
        w_total = 0.0
        coord = _node_coord_map(multi)
        for e in tables[case]["line_loads"]:
            L = ((e["x_j"] - e["x_i"]) ** 2 + (e["y_j"] - e["y_i"]) ** 2
                 + (e["z_j"] - e["z_i"]) ** 2) ** 0.5
            w_total += e["w_line_kNm"] * L
        rz_total = sum(r["RZ_kN"] for r in multi.case_results[case].reactions)
        rel = abs(w_total - rz_total) / max(abs(rz_total), 1e-9) * 100
        print(f"  [{case}] Σ(w·L)={w_total:10.3f} kN | ΣRZ={rz_total:10.3f} kN "
              f"| Δ={rel:.4f}%  (per-floor area={floor_area:.1f} m²)")

    # Envelope vs the manuscript reference.
    ref = json.loads(REF.read_text(encoding="utf-8"))
    env = ref["analysis"]["envelope"]
    # recompute envelope from combo_results
    max_dx = max(abs(c.max_displacement_x) for c in multi.combo_results.values())
    max_dy = max(abs(c.max_displacement_y) for c in multi.combo_results.values())
    max_drx = max(c.max_drift_x for c in multi.combo_results.values())
    max_dry = max(c.max_drift_y for c in multi.combo_results.values())
    print(f"\n  Envelope check (this run vs manuscript example_section4_results.json):")
    for label, got, want in (
        ("max_dx_mm", max_dx, env["max_dx_mm"]),
        ("max_dy_mm", max_dy, env["max_dy_mm"]),
        ("max_drift_x", max_drx, env["max_drift_x"]),
        ("max_drift_y", max_dry, env["max_drift_y"]),
    ):
        rel = abs(got - want) / max(abs(want), 1e-12) * 100
        flag = "OK" if rel < 0.5 else "CHECK"
        print(f"    {label:14s}: run={got:.6g}  ref={want:.6g}  Δ={rel:.3f}%  [{flag}]")


def main() -> int:
    print(f"[*] Loading {INPUT.relative_to(ROOT)}")
    config = json.loads(INPUT.read_text(encoding="utf-8"))
    config.pop("_meta", None)

    model = BuildingModel.from_json(config)
    print(f"[*] Model: {model.num_stories} stories, "
          f"{len(model.bays_x)}x{len(model.bays_y)} bays")

    load_result = generate_all_loads(model)
    load_cases = load_result["load_cases"]
    load_combinations = load_result["load_combinations"]
    print(f"[*] Load cases: {list(load_cases.keys())}")
    print(f"[*] Combinations: {len(load_combinations)}")

    kwargs = model.to_frame3d_kwargs()
    # geometry for base_nodes (regular path)
    from core.frame_3d import _generate_frame_3d_geometry
    _, _, _, base_nodes = _generate_frame_3d_geometry(
        model.story_heights_list, model.bays_x, model.bays_y)

    multi, buckets = _capture_loads(kwargs, load_cases, load_combinations)
    print(f"[*] Analysis done: {multi.num_elements} elements, "
          f"{len(multi.case_results)} cases, {len(multi.combo_results)} combos")

    tables = _build_load_tables(multi, buckets)
    model_payload = _build_model_payload(multi, base_nodes, config)
    resp_payload = _build_response_payload(multi)

    MODEL_OUT.write_text(json.dumps(model_payload, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    LOADS_OUT.write_text(json.dumps(
        {"_meta": {"note": "per-case loads captured from OpenSees eleLoad/load; "
                           "w_line_kNm = |Wz|, gravity (-Z); nodal in kN/kN·m"},
         "cases": tables}, ensure_ascii=False, indent=2), encoding="utf-8")
    RESP_OUT.write_text(json.dumps(resp_payload, ensure_ascii=False, indent=2,
                                   default=str), encoding="utf-8")

    _sanity(multi, tables, config)

    print("\n[OK] wrote:")
    for p in (MODEL_OUT, LOADS_OUT, RESP_OUT):
        print(f"     {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
