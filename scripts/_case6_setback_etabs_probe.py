"""Case 6 setback-column diagnostic probe (ETABS 23).

Task C (2026-07-01): the setback column (12,4,10.5)→(12,4,14.0) reports an
upper-end My of −29.09 (OpenSees) vs −55.24 (ETABS) under EQX — ~1.9×, same
sign. This probe re-runs the ETABS Case-6 L-shape model and dumps, for that
column under EQX:
    (1) FrameForce at ALL output stations (sta, M2, M3, V2, V3) — reveals the
        moment gradient and whether a station lands exactly at the j-end (z=14.0),
    (2) both end moments (i-end sta=0, j-end sta=L),
    (3) joint rotations R1/R2/R3 (θx/θy/θz) at both joints (12,4,10.5) & (12,4,14.0),
to separate hypothesis (a) rigid-diaphragm constraint-formulation θy difference
from (b) FrameForce output-station granularity on a double-curvature column.

It also re-extracts the DL/LL gravity reactions to confirm the V2 gravity-load
fix closed the ZoneA/SharedBoundary tributary gap (Task B).

OpenSees reference (from case6_lshape.py, EQX):
    setback col 29: My_i(z=10.5)=+58.46  My_j(z=14.0)=−29.09  Mz_i=1.85  Mz_j=−0.88
    joint_lo(12,4,10.5): ry=+0.001344   joint_hi(12,4,14.0): ry=+0.001568

Usage (ETABS must be installed + licensed):
    .\\opensees-mcp\\Scripts\\python.exe scripts/_case6_setback_etabs_probe.py --launch
"""
from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "mcp-server"))
sys.path.insert(0, str(ROOT / "tests" / "benchmark"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from core.etabs_api import ETABSClient                       # noqa: E402
from etabs_benchmark_case1_2 import (                        # noqa: E402
    _init, _run, _select_case, _displ, _load_pattern,
)
from etabs_benchmark_case6_lshape import (                   # noqa: E402
    _setup_stories, _material_case6, _build_geometry,
    _apply_rigid_diaphragm, _apply_distributed_from_table, _apply_lateral,
    BENCH_DIR, DIAG_NODES, DIAG_COLUMNS,
)


def _frameforce_all(model, elem: str) -> list[dict]:
    """Return every FrameForce output station for `elem` (currently-selected case).

    Each row: {sta, P, V2, V3, T, M2, M3} in kN / kN·m, sta in m from i-end.
    """
    (n, obj, obj_sta, elm, elm_sta, lc, st, sn,
     p, v2, v3, t, m2, m3, ret) = model.Results.FrameForce(
        elem, 0, 0, [], [], [], [], [], [], [], [], [], [], [], [], []
    )
    if ret != 0:
        raise RuntimeError(f"FrameForce '{elem}' failed (ret={ret})")
    sta = list(obj_sta) if obj_sta else []
    rows = []
    for i in range(len(sta)):
        rows.append({
            "sta": round(float(sta[i]), 4),
            "P":  round(float(list(p)[i]), 3),
            "V2": round(float(list(v2)[i]), 3),
            "V3": round(float(list(v3)[i]), 3),
            "T":  round(float(list(t)[i]), 3),
            "M2": round(float(list(m2)[i]), 3),
            "M3": round(float(list(m3)[i]), 3),
        })
    return rows


def _react(model, node: str):
    (n, obj, elm, lc, st, sn,
     f1, f2, f3, m1, m2, m3, ret) = model.Results.JointReact(
        node, 0, 0, [], [], [], [], [], [], [], [], [], [])
    if ret != 0:
        raise RuntimeError(f"JointReact '{node}' failed (ret={ret})")
    g = lambda a: float(list(a)[0]) if a else 0.0
    return g(f1), g(f2), g(f3), g(m1), g(m2), g(m3)


def run_probe(client) -> dict:
    m = client.model
    _init(m)
    _setup_stories(m)
    _material_case6(m)
    nodes_xyz, cols_xyz, beams_xy_z, nodes_by_story = _build_geometry(m)
    _apply_rigid_diaphragm(m, nodes_by_story)

    tables = json.loads((BENCH_DIR / "case6_lshape_loadtables.json").read_text(encoding="utf-8"))
    for lp in ("DL", "LL", "EQX", "EQY"):
        _load_pattern(m, lp)
    _apply_distributed_from_table(m, beams_xy_z, "DL", tables["DL_line_loads_kNm"])
    _apply_distributed_from_table(m, beams_xy_z, "LL", tables["LL_line_loads_kNm"])
    _apply_lateral(m, nodes_by_story, "EQX", "X")
    _apply_lateral(m, nodes_by_story, "EQY", "Y")
    _run(m)

    out = {}

    # ── Task B confirmation: DL/LL gravity reactions ──
    for case in ("DL", "LL"):
        _select_case(m, case)
        grav = {}
        for key, lbl in (("zone_a_corner_base", "ZoneA_corner"),
                         ("shared_boundary_base", "SharedBoundary"),
                         ("zone_b_far_base", "ZoneB_far")):
            node = nodes_xyz[DIAG_NODES[key]]
            _, _, f3, _, _, _ = _react(m, node)
            grav[lbl] = round(f3, 3)
        total = sum(_react(m, node)[2] for node in nodes_by_story.get(0, []))
        grav["BaseSumFz"] = round(total, 3)
        out[f"{case}_gravity_reactions"] = grav

    # ── Task C: setback column diagnostics under EQX ──
    _select_case(m, "EQX")
    setback = cols_xyz[(12.0, 4.0, 10.5, 14.0)]
    corner = cols_xyz[(0.0, 0.0, 0.0, 3.5)]

    out["setback_frameforce_all_stations"] = _frameforce_all(m, setback)
    out["corner_frameforce_all_stations"] = _frameforce_all(m, corner)

    # Joint rotations (θx=R1, θy=R2, θz=R3) at the two column joints
    joints = {}
    for lbl, coord in (("joint_lo(12,4,10.5)", (12.0, 4.0, 10.5)),
                       ("joint_hi(12,4,14.0)", (12.0, 4.0, 14.0))):
        node = nodes_xyz[coord]
        u1, u2, u3, r1, r2, r3 = _displ(m, node)
        joints[lbl] = {
            "dx_mm": round(u1 * 1000, 4), "dy_mm": round(u2 * 1000, 4),
            "rx": round(r1, 6), "ry": round(r2, 6), "rz": round(r3, 6),
        }
    out["setback_joint_rotations"] = joints
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--no-pause", action="store_true",
                    help="Do not wait for Enter before closing (non-interactive).")
    args = ap.parse_args()

    def _pause_close(client):
        if not args.launch:
            return
        if not args.no_pause:
            try:
                input("\n  [Enter] to close ETABS …")
            except EOFError:
                pass
        client.close()

    print("\n" + "=" * 78)
    print("  Case 6 setback-column probe (ETABS 23) — Task C")
    print("=" * 78)
    try:
        client = ETABSClient.launch(visible=True) if args.launch else ETABSClient.attach()
        print("  Connected ✓")
    except Exception as e:
        print(f"  ❌ Cannot connect to ETABS: {e}")
        sys.exit(1)

    try:
        res = run_probe(client)
    except Exception as e:
        print(f"  ❌ Probe failed: {e}")
        import traceback
        traceback.print_exc()
        _pause_close(client)
        sys.exit(1)

    out_path = ROOT / "docs" / "paper1_open_source_alternative" / "validation" / "case6_setback_etabs_probe.json"
    out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n── Task B: gravity reactions (ETABS, kN) ──")
    for case in ("DL", "LL"):
        print(f"  {case}: {res[f'{case}_gravity_reactions']}")

    print("\n── Task C: setback column FrameForce (EQX, all stations) ──")
    print(f"  {'sta':>6} {'P':>9} {'V2':>9} {'V3':>9} {'M2':>9} {'M3':>9}")
    for row in res["setback_frameforce_all_stations"]:
        print(f"  {row['sta']:>6} {row['P']:>9} {row['V2']:>9} {row['V3']:>9} "
              f"{row['M2']:>9} {row['M3']:>9}")
    print("\n  setback joint rotations (θy=R2 drives strong-axis My):")
    for lbl, jr in res["setback_joint_rotations"].items():
        print(f"    {lbl}: {jr}")

    print("\n  OS reference (EQX): setback My_i=+58.46 My_j=−29.09 | "
          "ry_lo=+0.001344 ry_hi=+0.001568")
    print(f"\n  → saved: {out_path}")
    _pause_close(client)


if __name__ == "__main__":
    main()
