"""Case 6 ETABS benchmark — L-shape decision-gate (OS↔ET 2-way).

Replicates the L-shape model from tests/benchmark/case6_lshape.py
(70 nodes, 57 columns, 78 beams, 5 stories with setback) in ETABS 23,
applying 4 independent linear-static load cases (DL, LL, EQX, EQY) and
comparing against OpenSees results.

Midas reference is currently empty (decision-gate deferred);
2-way OS↔ET comparison only.  3-way will be possible once Midas is run.

Material: SS275 with E = 210 GPa, unified with the value used in Cases 1-5
(the commercial-reference Young's modulus). Case 6 previously used the 205 GPa
SS275 application default; both the OpenSees and ETABS Case 6 models now use 210 GPa.

Coordinate system: X, Y horizontal; Z vertical (up).  Units: kN, m.

Usage (ETABS must be running and licensed):
    .\\opensees-mcp\\Scripts\\python.exe tests\\benchmark\\etabs_benchmark_case6_lshape.py --launch
"""
from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "mcp-server"))
sys.path.insert(0, str(Path(__file__).parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from core.etabs_api import ETABSClient   # noqa: E402

# Reuse helpers from the Cases 1-5 file.
from etabs_benchmark_case1_2 import (    # noqa: E402
    _KN_M_C, _STEEL, _LP_OTHER, _KS_LIB,
    _init, _pt, _frame, _restrain,
    _load_pattern, _joint_load, _run, _select_case,
    _displ, _react, _moments_at,
)

BENCH_DIR  = Path(__file__).parent
MIDAS_DIR  = BENCH_DIR / "midas_results"
ETABS_DIR  = BENCH_DIR / "etabs_results"

# ── Spec constants (from scripts/_case6_lshape_spec.md) ──
STORY_HEIGHT_M = 3.5
NUM_STORIES    = 5
DL_KNM2        = 5.1
LL_KNM2        = 2.5
V_BASE_KN      = 500.0    # total base shear for EQX, EQY (independent)
_MAT           = "SS275"

# Diagnostic locations (m)
DIAG_NODES = {
    "zone_a_corner_base":      (0.0,  0.0,  0.0),
    "shared_boundary_base":    (12.0, 0.0,  0.0),
    "zone_b_far_base":         (24.0, 0.0,  0.0),
    "zone_a_far_5f":           (0.0,  8.0,  17.5),
    "zone_b_far_3f":           (24.0, 0.0,  10.5),
    "zone_b_re_entrant_3f":    (24.0, 4.0,  10.5),
}

DIAG_COLUMNS = {
    "corner_1f":         ((0.0, 0.0),  0.0,   3.5),    # base→1F at zone A corner
    "setback_3F_to_4F":  ((12.0, 4.0), 10.5, 14.0),    # 3F→4F at setback boundary
}


# ─────────────────────────────────────────────────────────────────────────
# Material + section overrides (E=210, unified with Cases 1-5)
# ─────────────────────────────────────────────────────────────────────────

def _material_case6(model) -> None:
    """SS275 with E = 210 GPa, unified with the benchmark Cases 1-5 value."""
    if model.PropMaterial.SetMaterial(_MAT, _STEEL) != 0:
        raise RuntimeError("SetMaterial SS275 failed")
    # 210 GPa = 210 000 N/mm² = 2.10×10⁸ kN/m²
    if model.PropMaterial.SetMPIsotropic(_MAT, 210_000_000.0, 0.3, 1.2e-5) != 0:
        raise RuntimeError("SetMPIsotropic SS275 failed")


def _setup_stories(model) -> None:
    """Define 5 stories of 3.5 m each.

    Stories must be defined BEFORE diaphragm assignment — ETABS' diaphragm
    system depends on the story framework being in place.  Without stories,
    Diaphragm.SetDiaphragm returns ret=1.

    cStory.SetStories_2 signature:
      BaseElevation, NumberStories, StoryNames[], StoryHeights[],
      IsMasterStory[], SimilarToStory[], SpliceAbove[], SpliceHeight[], color[]
    """
    n = NUM_STORIES
    story_names   = [f"Story{i+1}" for i in range(n)]
    story_heights = [STORY_HEIGHT_M] * n
    is_master     = [False] * (n - 1) + [True]
    similar_to    = [""] * (n - 1) + ["None"]
    splice_above  = [False] * n
    splice_height = [0.0] * n
    colors        = [-1] * n
    out = model.Story.SetStories_2(
        0.0, n,
        story_names, story_heights,
        is_master, similar_to,
        splice_above, splice_height,
        colors,
    )
    ret = out if isinstance(out, int) else out[-1]
    print(f"  [DBG] Story.SetStories_2 → ret={ret} (raw={out!r})")
    if ret != 0:
        raise RuntimeError(f"Story.SetStories_2 failed (ret={ret})")

    # Diagnostic: list any auto-created diaphragms after story definition
    try:
        out2 = model.Diaphragm.GetNameList()
        print(f"  [DBG] Diaphragm.GetNameList after SetStories → {out2!r}")
    except Exception as e:
        print(f"  [DBG] Diaphragm.GetNameList failed: {e}")
    try:
        out3 = model.Story.GetNameList()
        print(f"  [DBG] Story.GetNameList → {out3!r}")
    except Exception as e:
        print(f"  [DBG] Story.GetNameList failed: {e}")


def _ks_section(model, name: str, ks_label: str) -> None:
    """Import KS D 3502 section + zero As2/As3 for Euler-Bernoulli."""
    ret = model.PropFrame.ImportProp(name, _MAT, _KS_LIB, ks_label)
    if ret != 0:
        raise RuntimeError(
            f"ImportProp '{name}' ← '{ks_label}' from {_KS_LIB} failed (ret={ret})"
        )
    *_, ret2 = model.PropFrame.SetModifiers(
        name, [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    )
    if ret2 != 0:
        raise RuntimeError(f"SetModifiers '{name}' failed (ret={ret2})")


# ─────────────────────────────────────────────────────────────────────────
# Loads — distributed (DL/LL) and lateral (EQX/EQY)
# ─────────────────────────────────────────────────────────────────────────

def _distrib_load(model, frame: str, lp: str, w_line_kNm: float) -> None:
    """Apply uniform downward distributed load on a beam (in -Z direction).

    SetLoadDistributed signature:
      Name, LoadPat, MyType, Dir, Dist1, Dist2, Val1, Val2,
      CSys='Global', RelDist=True, Replace=True, ItemType=Object
        MyType 1 = force per length (kN/m)
        Dir 10  = Gravity (= -Z always)
    """
    out = model.FrameObj.SetLoadDistributed(
        frame, lp, 1, 10, 0.0, 1.0,
        w_line_kNm, w_line_kNm,
        "Global", True, True,
    )
    ret = out if isinstance(out, int) else out[-1]
    if ret != 0:
        raise RuntimeError(
            f"SetLoadDistributed '{frame}' lp={lp} failed (ret={ret})"
        )


# ─────────────────────────────────────────────────────────────────────────
# Rigid diaphragm per story
# ─────────────────────────────────────────────────────────────────────────

def _apply_rigid_diaphragm(model, nodes_by_story: dict) -> None:
    """Define and assign a rigid diaphragm for each story (1-5).

    ETABS API:
      Diaphragm.SetDiaphragm(name, SemiRigid=False)   — define rigid diaphragm
      PointObj.SetDiaphragm(node, DiaphragmOption=3, DiaphragmName=name)
        DiaphragmOption 3 = From Specified Diaphragm Name
    """
    # Try a less reserved name pattern in case "D1" collides with auto-created names
    for s in range(1, NUM_STORIES + 1):
        dname = f"RD{s}"   # was 'D{s}' — that name is reserved/auto-created by ETABS
        ret = model.Diaphragm.SetDiaphragm(dname, False)
        if ret != 0:
            # Try SemiRigid=True (the CHM example uses True)
            ret2 = model.Diaphragm.SetDiaphragm(dname, True)
            print(f"  [DBG] SetDiaphragm('{dname}', False) ret={ret}, "
                  f"retry with True ret={ret2}")
            if ret2 != 0:
                raise RuntimeError(
                    f"Diaphragm.SetDiaphragm '{dname}' failed: "
                    f"ret={ret} (False) and ret={ret2} (True)"
                )
        for node in nodes_by_story.get(s, []):
            ret = model.PointObj.SetDiaphragm(node, 3, dname)
            if ret != 0:
                raise RuntimeError(
                    f"PointObj.SetDiaphragm node='{node}' D='{dname}' failed (ret={ret})"
                )


# ─────────────────────────────────────────────────────────────────────────
# Geometry build — matches OpenSees build_l_shape_model() node/elem order
# ─────────────────────────────────────────────────────────────────────────

def _build_geometry(model) -> tuple:
    """Build L-shape: 70 nodes, 57 columns, 78 beams.

    Returns:
        nodes_xyz:      dict[(x, y, z) → etabs node name]
        cols_xyz:       dict[(x, y, z_bot, z_top) → etabs col frame name]
        beams_xy_z:     dict[((x_i, y_i), (x_j, y_j), z) → etabs beam frame name]
        nodes_by_story: dict[story_index (0=base, 1-5=tops) → list of node names]
    """
    _ks_section(model, "H300x300", "H 300x300x10/15")   # columns
    _ks_section(model, "H400x200", "H 400x200x8/13")    # beams

    nodes_xyz      = {}
    cols_xyz       = {}
    beams_xy_z     = {}
    nodes_by_story = {s: [] for s in range(NUM_STORIES + 1)}   # 0=base, 1..5=tops

    left_cols  = [(0, 0), (6, 0), (12, 0), (0, 4), (6, 4), (12, 4), (0, 8), (6, 8), (12, 8)]
    right_cols = [(18, 0), (24, 0), (18, 4), (24, 4)]

    # Nodes: Zone A first (6 levels), then Zone B (4 levels), to match OpenSees IDs
    node_counter = 1
    for x, y in left_cols:
        for s in range(NUM_STORIES + 1):   # 0..5
            z = s * STORY_HEIGHT_M
            n = _pt(model, f"N{node_counter}", float(x), float(y), float(z))
            nodes_xyz[(float(x), float(y), float(z))] = n
            nodes_by_story[s].append(n)
            node_counter += 1
    for x, y in right_cols:
        for s in range(4):                  # 0..3 (Zone B has 3 stories)
            z = s * STORY_HEIGHT_M
            n = _pt(model, f"N{node_counter}", float(x), float(y), float(z))
            nodes_xyz[(float(x), float(y), float(z))] = n
            nodes_by_story[s].append(n)
            node_counter += 1

    assert len(nodes_xyz) == 70, f"Expected 70 nodes, got {len(nodes_xyz)}"

    # Columns
    col_counter = 1
    for x, y in left_cols:
        for s in range(NUM_STORIES):
            z_bot = s * STORY_HEIGHT_M
            z_top = (s + 1) * STORY_HEIGHT_M
            ni = nodes_xyz[(float(x), float(y), float(z_bot))]
            nj = nodes_xyz[(float(x), float(y), float(z_top))]
            c = _frame(model, f"C{col_counter}", ni, nj, "H300x300")
            cols_xyz[(float(x), float(y), float(z_bot), float(z_top))] = c
            col_counter += 1
    for x, y in right_cols:
        for s in range(3):
            z_bot = s * STORY_HEIGHT_M
            z_top = (s + 1) * STORY_HEIGHT_M
            ni = nodes_xyz[(float(x), float(y), float(z_bot))]
            nj = nodes_xyz[(float(x), float(y), float(z_top))]
            c = _frame(model, f"C{col_counter}", ni, nj, "H300x300")
            cols_xyz[(float(x), float(y), float(z_bot), float(z_top))] = c
            col_counter += 1

    assert len(cols_xyz) == 57, f"Expected 57 columns, got {len(cols_xyz)}"

    # Beams — match build_l_shape_model() iteration order:
    #   X-direction beams first (all stories), then Y-direction beams
    beam_counter = 1
    # X-direction
    for s in range(1, NUM_STORIES + 1):
        z = s * STORY_HEIGHT_M
        for y in (0, 4, 8):                                   # Zone A
            for x1, x2 in ((0, 6), (6, 12)):
                k_i = (float(x1), float(y), float(z))
                k_j = (float(x2), float(y), float(z))
                if k_i in nodes_xyz and k_j in nodes_xyz:
                    b = _frame(model, f"B{beam_counter}",
                               nodes_xyz[k_i], nodes_xyz[k_j], "H400x200")
                    beams_xy_z[((float(x1), float(y)), (float(x2), float(y)), float(z))] = b
                    beam_counter += 1
        if s <= 3:                                            # Zone B (stories 1-3 only)
            for y in (0, 4):
                for x1, x2 in ((12, 18), (18, 24)):
                    k_i = (float(x1), float(y), float(z))
                    k_j = (float(x2), float(y), float(z))
                    if k_i in nodes_xyz and k_j in nodes_xyz:
                        b = _frame(model, f"B{beam_counter}",
                                   nodes_xyz[k_i], nodes_xyz[k_j], "H400x200")
                        beams_xy_z[((float(x1), float(y)), (float(x2), float(y)), float(z))] = b
                        beam_counter += 1
    # Y-direction
    for s in range(1, NUM_STORIES + 1):
        z = s * STORY_HEIGHT_M
        for x in (0, 6, 12):                                  # Zone A
            for y1, y2 in ((0, 4), (4, 8)):
                k_i = (float(x), float(y1), float(z))
                k_j = (float(x), float(y2), float(z))
                if k_i in nodes_xyz and k_j in nodes_xyz:
                    b = _frame(model, f"B{beam_counter}",
                               nodes_xyz[k_i], nodes_xyz[k_j], "H400x200")
                    beams_xy_z[((float(x), float(y1)), (float(x), float(y2)), float(z))] = b
                    beam_counter += 1
        if s <= 3:                                            # Zone B
            for x in (18, 24):
                k_i = (float(x), 0.0, float(z))
                k_j = (float(x), 4.0, float(z))
                if k_i in nodes_xyz and k_j in nodes_xyz:
                    b = _frame(model, f"B{beam_counter}",
                               nodes_xyz[k_i], nodes_xyz[k_j], "H400x200")
                    beams_xy_z[((float(x), 0.0), (float(x), 4.0), float(z))] = b
                    beam_counter += 1

    assert len(beams_xy_z) == 78, f"Expected 78 beams, got {len(beams_xy_z)}"

    # Restraints — fix all base nodes (z = 0)
    for (x, y, z), name in nodes_xyz.items():
        if z == 0.0:
            _restrain(model, name, [True] * 6)

    print(f"  Geometry: {len(nodes_xyz)} nodes, {len(cols_xyz)} columns, "
          f"{len(beams_xy_z)} beams")
    return nodes_xyz, cols_xyz, beams_xy_z, nodes_by_story


# ─────────────────────────────────────────────────────────────────────────
# Distributed load application from the OS-side load table
# ─────────────────────────────────────────────────────────────────────────

def _apply_distributed_from_table(model, beams_xy_z: dict, lp: str,
                                    table: list) -> None:
    """Apply per-element w_line_kNm loads from the load tables JSON."""
    applied = 0
    missed = 0
    for entry in table:
        # ETABS receives positive Val with Dir=10 (gravity) → load points -Z.
        # The OS table reports w_line_kNm as the magnitude of gravity load.
        w_line = entry["w_line_kNm"]
        # Locate corresponding ETABS beam by endpoint coordinates
        key1 = (
            (float(entry["x_i"]), float(entry["y_i"])),
            (float(entry["x_j"]), float(entry["y_j"])),
            float(entry["z_i"]),
        )
        key2 = (
            (float(entry["x_j"]), float(entry["y_j"])),
            (float(entry["x_i"]), float(entry["y_i"])),
            float(entry["z_i"]),
        )
        frame = beams_xy_z.get(key1) or beams_xy_z.get(key2)
        if frame is None:
            missed += 1
            continue
        _distrib_load(model, frame, lp, w_line)
        applied += 1
    print(f"  [{lp}] distributed loads applied to {applied}/{applied + missed} beams")


def _apply_lateral(model, nodes_by_story: dict, lp: str, direction: str) -> None:
    """Apply inverted-triangle lateral nodal loads for V_BASE_KN total.

    direction: 'X' or 'Y' — which global axis to push.
    """
    z_levels = [STORY_HEIGHT_M * s for s in range(1, NUM_STORIES + 1)]
    sum_h    = sum(z_levels)
    for s, hi in enumerate(z_levels, start=1):
        Fi_total = V_BASE_KN * hi / sum_h
        snodes   = nodes_by_story.get(s, [])
        if not snodes:
            continue
        per_node = Fi_total / len(snodes)
        for n in snodes:
            if direction == "X":
                forces = [per_node, 0.0, 0.0, 0.0, 0.0, 0.0]
            elif direction == "Y":
                forces = [0.0, per_node, 0.0, 0.0, 0.0, 0.0]
            else:
                raise ValueError(f"Bad direction '{direction}'")
            _joint_load(model, n, lp, forces)


# ─────────────────────────────────────────────────────────────────────────
# Result extraction
# ─────────────────────────────────────────────────────────────────────────

def _diag_node(nodes_xyz: dict, key: str) -> str:
    return nodes_xyz[DIAG_NODES[key]]


def _extract_one(model, nodes_xyz: dict, cols_xyz: dict, nodes_by_story: dict,
                   case_label: str, prefix: str) -> dict:
    """Extract metrics for the currently selected case.

    prefix: short label inserted into metric names (e.g. 'DL', 'LL', 'EQX').
    """
    out = {}

    # Per-diagnostic-node reactions and displacements
    if prefix in ("DL", "LL"):
        # Vertical reactions at three base nodes
        for diag_key, midas_label in (
            ("zone_a_corner_base",   "ZoneA_corner"),
            ("shared_boundary_base", "SharedBoundary"),
            ("zone_b_far_base",      "ZoneB_far"),
        ):
            n = _diag_node(nodes_xyz, diag_key)
            _, _, f3, _, _, _ = _react(model, n)
            out[f"{prefix} Reaction {midas_label} Fz (kN)"] = f3

        # Base sum Fz
        total_fz = 0.0
        for n in nodes_by_story.get(0, []):   # all base nodes
            _, _, f3, _, _, _ = _react(model, n)
            total_fz += f3
        out[f"{prefix} Base SumFz (kN)"] = total_fz

        # Vertical displacement at ZoneB 3F far corner (24, 0, 10.5)
        n = _diag_node(nodes_xyz, "zone_b_far_3f")
        _, _, u3, _, _, _ = _displ(model, n)
        out[f"{prefix} ZoneB 3F Far Corner dz (mm)"] = u3 * 1000.0

    elif prefix in ("EQX", "EQY"):
        # Roof displacements (Zone A 5F far corner)
        n_a5 = _diag_node(nodes_xyz, "zone_a_far_5f")
        u1_a5, u2_a5, _, _, _, _ = _displ(model, n_a5)
        out[f"{prefix} ZoneA Far 5F dx (mm)"] = u1_a5 * 1000.0
        out[f"{prefix} ZoneA Far 5F dy (mm)"] = u2_a5 * 1000.0

        # Base shear (sum F1 for EQX, sum F2 for EQY at all base nodes)
        total_fx = 0.0
        total_fy = 0.0
        for n in nodes_by_story.get(0, []):
            f1, f2, _, _, _, _ = _react(model, n)
            total_fx += f1
            total_fy += f2
        out[f"{prefix} Base Shear Fx (kN)"] = total_fx if prefix == "EQX" else 0.0
        out[f"{prefix} Base Shear Fy (kN)"] = total_fy if prefix == "EQY" else 0.0
        # only one of the two is meaningful per case, but mirror the Midas key set
        # so the comparison table aligns.

        # Story drifts — average drift per story (max over stories)
        # Use average displacement at top of each story / h
        max_drift_x = 0.0
        max_drift_y = 0.0
        for s in range(1, NUM_STORIES + 1):
            snodes = nodes_by_story.get(s, [])
            if not snodes:
                continue
            if s == 1:
                d_low_x = 0.0
                d_low_y = 0.0
            else:
                lower = nodes_by_story.get(s - 1, [])
                d_low_x = sum(_displ(model, n)[0] for n in lower) / len(lower) if lower else 0.0
                d_low_y = sum(_displ(model, n)[1] for n in lower) / len(lower) if lower else 0.0
            d_up_x = sum(_displ(model, n)[0] for n in snodes) / len(snodes)
            d_up_y = sum(_displ(model, n)[1] for n in snodes) / len(snodes)
            drift_x = abs(d_up_x - d_low_x) / STORY_HEIGHT_M
            drift_y = abs(d_up_y - d_low_y) / STORY_HEIGHT_M
            max_drift_x = max(max_drift_x, drift_x)
            max_drift_y = max(max_drift_y, drift_y)
        out[f"{prefix} Max StoryDrift X (ratio)"] = max_drift_x if prefix == "EQX" else 0.0
        out[f"{prefix} Max StoryDrift Y (ratio)"] = max_drift_y if prefix == "EQY" else 0.0

        # Torsion: difference between zone_a_far_5f and zone_b_far_3f or re_entrant
        n_b3 = _diag_node(nodes_xyz, "zone_b_re_entrant_3f")
        u1_b3, u2_b3, _, _, _, _ = _displ(model, n_b3)
        out[f"{prefix} Torsion (A_5F dx - B_3F dx) (mm)"] = (u1_a5 - u1_b3) * 1000.0
        out[f"{prefix} Torsion (A_5F dy - B_3F dy) (mm)"] = (u2_a5 - u2_b3) * 1000.0

        # Column base/end moments via FrameForce (NOT JointReact — interior
        # joints have zero reactions by definition; we need the column's own
        # member forces).  Default ETABS vertical column orientation has:
        #   local-3 ≈ global +Y → M3 = strong-axis bending (matches OS My)
        #   local-2 ≈ global +X → M2 = weak-axis bending  (matches OS Mz)
        if prefix == "EQX":
            # Corner column C1: base (i-end, sta=0) at (0,0)→(0,0,3.5)
            col_corner = cols_xyz[(0.0, 0.0, 0.0, 3.5)]
            m2_corner, m3_corner = _moments_at(model, col_corner, 0.0)
            # Case 4 convention: negate M3 at column i-end to match OS sign
            out[f"{prefix} CornerCol1F Base My (kNm)"] = -m3_corner
            out[f"{prefix} CornerCol1F Base Mz (kNm)"] =  m2_corner

            # Setback column: 3F→4F is the column from (12,4,10.5) to (12,4,14.0).
            # Upper end = j-end of this column = station 3.5 m from i.
            col_setback = cols_xyz[(12.0, 4.0, 10.5, 14.0)]
            m2_sb, m3_sb = _moments_at(model, col_setback, 3.5)
            # j-end: OS convention reverses signs at j relative to i (per Cases 2-3)
            out[f"{prefix} SetbackCol 3F->4F UpperEnd My (kNm)"] =  m3_sb
            out[f"{prefix} SetbackCol 3F->4F UpperEnd Mz (kNm)"] = -m2_sb

    return out


# ─────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────

def run_case6_lshape_etabs(client) -> dict:
    """Build L-shape model in ETABS, apply 4 cases, extract metrics."""
    m = client.model
    _init(m)
    _setup_stories(m)            # Must precede geometry/diaphragm setup
    _material_case6(m)

    nodes_xyz, cols_xyz, beams_xy_z, nodes_by_story = _build_geometry(m)
    _apply_rigid_diaphragm(m, nodes_by_story)

    # Load the OS-side load tables (used for distributed loads)
    tables_path = BENCH_DIR / "case6_lshape_loadtables.json"
    if not tables_path.exists():
        raise FileNotFoundError(
            f"Missing {tables_path} — run case6_lshape.py first to generate."
        )
    tables = json.loads(tables_path.read_text(encoding="utf-8"))

    # Load patterns
    for lp in ("DL", "LL", "EQX", "EQY"):
        _load_pattern(m, lp)

    # DL & LL distributed loads
    _apply_distributed_from_table(m, beams_xy_z, "DL", tables["DL_line_loads_kNm"])
    _apply_distributed_from_table(m, beams_xy_z, "LL", tables["LL_line_loads_kNm"])

    # EQX & EQY lateral nodal loads (inverted triangle)
    _apply_lateral(m, nodes_by_story, "EQX", "X")
    _apply_lateral(m, nodes_by_story, "EQY", "Y")

    _run(m)

    # Extract metrics per case
    all_metrics = {}
    for lp, prefix in (("DL", "DL"), ("LL", "LL"), ("EQX", "EQX"), ("EQY", "EQY")):
        _select_case(m, lp)
        m_dict = _extract_one(m, nodes_xyz, cols_xyz, nodes_by_story, lp, prefix)
        all_metrics.update(m_dict)

    return all_metrics


def _extract_case6(r: dict) -> list[dict]:
    """Map raw metric dict → comparison rows.  Preserves Midas key order."""
    midas_keys = [
        "DL Base SumFz (kN)",
        "DL Reaction ZoneA_corner Fz (kN)",
        "DL Reaction SharedBoundary Fz (kN)",
        "DL Reaction ZoneB_far Fz (kN)",
        "DL ZoneB 3F Far Corner dz (mm)",
        "LL Base SumFz (kN)",
        "LL Reaction ZoneA_corner Fz (kN)",
        "LL Reaction SharedBoundary Fz (kN)",
        "LL Reaction ZoneB_far Fz (kN)",
        "LL ZoneB 3F Far Corner dz (mm)",
        "EQX ZoneA Far 5F dx (mm)",
        "EQX ZoneA Far 5F dy (mm)",
        "EQX Max StoryDrift X (ratio)",
        "EQX Base Shear Fx (kN)",
        "EQY ZoneA Far 5F dx (mm)",
        "EQY ZoneA Far 5F dy (mm)",
        "EQY Max StoryDrift Y (ratio)",
        "EQY Base Shear Fy (kN)",
        "EQX Torsion (A_5F dx - B_3F dx) (mm)",
        "EQY Torsion (A_5F dy - B_3F dy) (mm)",
        "EQX CornerCol1F Base My (kNm)",
        "EQX CornerCol1F Base Mz (kNm)",
        "EQX SetbackCol 3F->4F UpperEnd My (kNm)",
        "EQX SetbackCol 3F->4F UpperEnd Mz (kNm)",
    ]
    rows = []
    for k in midas_keys:
        rows.append({
            "metric": k,
            "unit": "—",                     # units are encoded in metric name
            "opensees": r.get(k),
        })
    return rows


# ─────────────────────────────────────────────────────────────────────────
# Comparison and output (mirrors etabs_benchmark_case1_2.py style)
# ─────────────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _save_etabs(case_id: str, extracted: list[dict]) -> None:
    ETABS_DIR.mkdir(exist_ok=True)
    fpath = ETABS_DIR / f"{case_id}.json"
    data = {row["metric"]: row["opensees"] for row in extracted}
    fpath.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  → saved: {fpath}")


def _pct(a, b) -> str:
    if a is None or b is None:
        return "      —"
    ref = max(abs(a), abs(b))
    if ref < 1e-12:
        return "  0.00%"
    return f"{abs(a - b) / ref * 100:6.2f}%"


def _status(et_val, midas_val) -> str:
    if midas_val is None or et_val is None:
        return "PENDING"
    ref = max(abs(et_val), abs(midas_val))
    if ref < 1e-12:
        return "OK"
    d = abs(et_val - midas_val) / ref * 100
    return "OK" if d <= 1.0 else ("CHECK" if d <= 5.0 else "FAIL")


def format_3way(case_name: str, extracted: list[dict],
                midas: dict | None, opensees: dict | None) -> str:
    W = 130
    lines = [
        f"\n{'=' * W}",
        f"  {case_name}",
        f"{'=' * W}",
        f"{'Metric':<46} {'OpenSees':>13} {'ETABS':>13} "
        f"{'Midas':>13} {'OS–ET%':>8} {'OS–M%':>8} {'ET–M%':>8} {'ET vs M':>10}",
        "-" * W,
    ]
    ok = check = fail = pending = 0
    for row in extracted:
        metric  = row["metric"]
        et_val  = row["opensees"]                          # ETABS value in "opensees" slot
        m_val   = (midas    or {}).get(metric)
        ops_val = (opensees or {}).get(metric)

        def _fmt(v):
            return f"{'—':>13}" if v is None else f"{v:13.4f}"

        st = _status(et_val, m_val)
        if   st == "OK":      ok      += 1
        elif st == "CHECK":   check   += 1
        elif st == "FAIL":    fail    += 1
        else:                 pending += 1

        lines.append(
            f"{metric:<46} {_fmt(ops_val)} {_fmt(et_val)} {_fmt(m_val)} "
            f"{_pct(ops_val, et_val):>8} {_pct(ops_val, m_val):>8} "
            f"{_pct(et_val, m_val):>8} {st:>10}"
        )
    total = len(extracted)
    lines += [
        "-" * W,
        f"  Total: {total}  |  OK: {ok}  |  CHECK: {check}  |"
        f"  FAIL: {fail}  |  PENDING: {pending}"
        f"   (PENDING = Midas reference is null — decision-gate deferred)",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ETABS 23 benchmark — Case 6 (L-shape decision-gate)",
    )
    parser.add_argument("--launch", action="store_true",
                        help="Launch new ETABS instance (default: attach)")
    args = parser.parse_args()

    sep = "=" * 80
    print(f"\n{sep}")
    print("  ETABS 23 Benchmark — Case 6: L-shape decision-gate (OS↔ET 2-way)")
    print(f"{sep}")

    try:
        if args.launch:
            print("  Launching ETABS …")
            client = ETABSClient.launch(visible=True)
        else:
            print("  Attaching to running ETABS …")
            client = ETABSClient.attach()
        print("  Connected ✓\n")
    except Exception as exc:
        print(f"  ❌ Cannot connect to ETABS: {exc}")
        sys.exit(1)

    case_id = "case6_lshape"
    name    = "Case 6: L-shape (5F+3F setback)"
    print(f"\n  [{case_id.upper()}] {name} …", end=" ", flush=True)
    try:
        raw = run_case6_lshape_etabs(client)
    except Exception as exc:
        print(f"\n  ❌ Failed: {exc}")
        import traceback
        traceback.print_exc()
        if args.launch:
            input("\n  [Enter] to close ETABS …")
            client.close()
        sys.exit(1)
    print("done")

    extracted = _extract_case6(raw)
    midas     = _load_json(MIDAS_DIR / f"{case_id}.json")
    opensees  = _load_json(BENCH_DIR / "opensees_results" / f"{case_id}.json")

    _save_etabs(case_id, extracted)
    print(format_3way(name, extracted, midas, opensees))

    if args.launch:
        input("\n  [Enter] to close ETABS …")
        client.close()

    print(f"\n{sep}")
    print(f"  Done.  Results → tests/benchmark/etabs_results/{case_id}.json")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
