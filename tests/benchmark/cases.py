"""Benchmark case definitions for Midas Gen comparison.

Each case builds a raw OpenSees model with exact node/element/load specs
matching the Midas Gen model. No solver code is modified.

Units: kN, m internally → converted to N, mm for OpenSees.
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'mcp-server'))
from core.ops_compat import ops


# ── Unit conversion helpers (user: kN,m → OpenSees: N,mm) ──
def _kN_to_N(v: float) -> float:
    return v * 1000.0

def _m_to_mm(v: float) -> float:
    return v * 1000.0



# ── Section properties from DB (KS D 3502, h_beam_sections) ──
# H-400x200x8x13: A=8412mm2, Ix=23700e4mm4, Iy=1740e4mm4, J=66e4mm4
# Material: SS275 → E=210GPa

SECTION_H400x200 = {
    "name": "H-400x200x8x13",
    "A_mm2": 8412.0,        # 84.12 cm2
    "Ix_mm4": 23700e4,      # 2.37e8 mm4 (strong axis)
    "Iy_mm4": 1740e4,       # 1.74e7 mm4 (weak axis)
    "J_mm4": 66e4,          # torsional constant (approx)
    "h_mm": 400.0,
    "b_mm": 200.0,
}

MATERIAL_SS275 = {
    "name": "SS275",
    "E_MPa": 210000.0,      # 210 GPa
    "fy_MPa": 275.0,
    "G_MPa": 80769.0,       # E / (2*(1+0.3))
}


# ============================================================
# CASE 1: 2D Simple Beam (Point Load)
# ============================================================

CASE1_SPEC = {
    "name": "2D Simple Beam (Point Load)",
    "description": (
        "60 kN point load at midspan (Node 2). "
        "Section: H-400x200x8x13 (A=8412mm2, Ix=23700e4mm4). "
        "Material: SS275 (E=210GPa)."
    ),
    "nodes": {1: (0.0, 0.0), 2: (3.0, 0.0), 3: (6.0, 0.0)},
    "elements": {1: (1, 2), 2: (2, 3)},
    "boundary": {1: "pin", 3: "roller"},
    "material": MATERIAL_SS275,
    "section": SECTION_H400x200,
    "loads": {"N2": {"Fy_kN": -60.0}},
    "analytical": {
        "R": "P/2 = 60/2 = 30 kN",
        "M_max": "PL/4 = 60*6/4 = 90 kN*m",
        "delta_max": "PL^3/(48EI) = 60e3*6000^3/(48*210000*23700e4) = 5.425 mm",
    },
}


def run_case1() -> dict:
    """2D simple beam: 3 nodes, 2 elements, 60kN point load at midspan.

    Section: H-400x200x8x13 (DB-resolved)
    Material: SS275 (E=210GPa)
    """
    E = MATERIAL_SS275["E_MPa"]               # 210000 MPa
    A = SECTION_H400x200["A_mm2"]             # 8412 mm^2
    Iz = SECTION_H400x200["Ix_mm4"]           # 23700e4 mm^4 (strong axis = Iz in 2D)

    ops.wipe()
    ops.model('basic', '-ndm', 2, '-ndf', 3)

    # Nodes (mm)
    ops.node(1, 0.0, 0.0)
    ops.node(2, 3000.0, 0.0)
    ops.node(3, 6000.0, 0.0)

    # Boundary: pin at N1 (fix x,y), roller at N3 (fix y)
    ops.fix(1, 1, 1, 0)
    ops.fix(3, 0, 1, 0)

    # Transformation
    ops.geomTransf('Linear', 1)

    # Elements
    ops.element('elasticBeamColumn', 1, 1, 2, A, E, Iz, 1)
    ops.element('elasticBeamColumn', 2, 2, 3, A, E, Iz, 1)

    # Point load at midspan (Node 2)
    ops.timeSeries('Linear', 1)
    ops.pattern('Plain', 1, 1)
    ops.load(2, 0.0, _kN_to_N(-60.0), 0.0)

    # Solve
    ops.system('BandGeneral')
    ops.numberer('RCM')
    ops.constraints('Plain')
    ops.integrator('LoadControl', 1.0)
    ops.algorithm('Linear')
    ops.analysis('Static')
    ops.analyze(1)

    # Extract results
    ops.reactions()
    d2 = ops.nodeDisp(2)
    r1 = ops.nodeReaction(1)
    r3 = ops.nodeReaction(3)

    # Element forces at midspan (element 1 j-end = node 2)
    f1 = ops.eleForce(1)  # [Ni, Vi, Mi, Nj, Vj, Mj]

    results = {
        "midspan_disp_mm": d2[1],                # vertical displacement at N2
        "midspan_moment_kNm": -f1[5] / 1e6,      # N*mm -> kN*m
        "reaction_N1_Fy_kN": r1[1] / 1000.0,
        "reaction_N3_Fy_kN": r3[1] / 1000.0,
        "reaction_N1_Fx_kN": r1[0] / 1000.0,
    }

    ops.wipe()
    return results


# ============================================================
# CASE 1_UDL: 2D Simple Beam (UDL) - Optional variant
# ============================================================

CASE1_UDL_SPEC = {
    "name": "2D Simple Beam (UDL)",
    "description": (
        "Uniformly distributed load 10 kN/m over full 6 m span. "
        "Section: H-400x200x8x13. Material: SS275 (E=210GPa). "
        "NOTE: Different from case1 (point load). "
        "Do NOT confuse with case1 - total load is 60kN but deflection/moment differ."
    ),
    "nodes": {1: (0.0, 0.0), 2: (3.0, 0.0), 3: (6.0, 0.0)},
    "elements": {1: (1, 2), 2: (2, 3)},
    "boundary": {1: "pin", 3: "roller"},
    "material": MATERIAL_SS275,
    "section": SECTION_H400x200,
    "loads": {
        "type": "UDL",
        "w_kN_per_m": -10.0,
        "note": "Applied via eleLoad on both elements, NOT as nodal load",
    },
    "analytical": {
        "R": "wL/2 = 10*6/2 = 30 kN",
        "M_max": "wL^2/8 = 10*36/8 = 45 kN*m",
        "delta_max": "5wL^4/(384EI) = 5*10*6000^4/(384*210000*23700e4) = 3.391 mm",
    },
}


def run_case1_udl() -> dict:
    """2D simple beam: UDL 10 kN/m variant (optional benchmark).

    Section: H-400x200x8x13 (DB-resolved)
    Material: SS275 (E=210GPa)
    """
    E = MATERIAL_SS275["E_MPa"]
    A = SECTION_H400x200["A_mm2"]
    Iz = SECTION_H400x200["Ix_mm4"]

    # UDL: 10 kN/m = 10 N/mm (downward = negative in local y)
    w_Nmm = -10.0

    ops.wipe()
    ops.model('basic', '-ndm', 2, '-ndf', 3)

    ops.node(1, 0.0, 0.0)
    ops.node(2, 3000.0, 0.0)
    ops.node(3, 6000.0, 0.0)

    ops.fix(1, 1, 1, 0)
    ops.fix(3, 0, 1, 0)

    ops.geomTransf('Linear', 1)
    ops.element('elasticBeamColumn', 1, 1, 2, A, E, Iz, 1)
    ops.element('elasticBeamColumn', 2, 2, 3, A, E, Iz, 1)

    ops.timeSeries('Linear', 1)
    ops.pattern('Plain', 1, 1)
    ops.eleLoad('-ele', 1, '-type', '-beamUniform', w_Nmm)
    ops.eleLoad('-ele', 2, '-type', '-beamUniform', w_Nmm)

    ops.system('BandGeneral')
    ops.numberer('RCM')
    ops.constraints('Plain')
    ops.integrator('LoadControl', 1.0)
    ops.algorithm('Linear')
    ops.analysis('Static')
    ops.analyze(1)

    ops.reactions()
    d2 = ops.nodeDisp(2)
    r1 = ops.nodeReaction(1)
    r3 = ops.nodeReaction(3)
    f1 = ops.eleForce(1)

    results = {
        "midspan_disp_mm": d2[1],
        "midspan_moment_kNm": -f1[5] / 1e6,
        "reaction_N1_Fy_kN": r1[1] / 1000.0,
        "reaction_N3_Fy_kN": r3[1] / 1000.0,
        "reaction_N1_Fx_kN": r1[0] / 1000.0,
    }

    ops.wipe()
    return results


# ── Case 2 section: H-350x350x12x19 (column) ──
# From Supabase ks3502.h_beam_sections
# A=173.9cm2, Ix=40300cm4, Iy=13600cm4, h=350, b=350, tw=12, tf=19
# J approx: (2*350*19^3 + 312*12^3)/3 = 178e4 mm4
SECTION_H350x350 = {
    "name": "H-350x350x12x19",
    "A_mm2": 17390.0,       # 173.9 cm2
    "Ix_mm4": 40300e4,      # 4.03e8 mm4 (strong axis)
    "Iy_mm4": 13600e4,      # 1.36e8 mm4 (weak axis)
    "J_mm4": 178e4,         # torsional constant (approx)
    "h_mm": 350.0,
    "b_mm": 350.0,
}

# ── Case 3 column section: H-400x400x13x21 ──
# From Supabase ks3502.h_beam_sections
# A=218.7cm2, Ix=66600cm4, Iy=22400cm4, h=400, b=400, tw=13, tf=21
# J approx: (2*400*21^3 + 358*13^3)/3 = 498e4 mm4
SECTION_H400x400 = {
    "name": "H-400x400x13x21",
    "A_mm2": 21870.0,       # 218.7 cm2
    "Ix_mm4": 66600e4,      # 6.66e8 mm4 (strong axis)
    "Iy_mm4": 22400e4,      # 2.24e8 mm4 (weak axis)
    "J_mm4": 498e4,         # torsional constant (approx)
    "h_mm": 400.0,
    "b_mm": 400.0,
}


# ============================================================
# CASE 2: 2D Portal Frame (1 story, 1 bay)
# ============================================================

CASE2_SPEC = {
    "name": "2D Portal Frame (1-story 1-bay)",
    "description": (
        "Frame lies in global X-Y plane (X=horizontal, Y=vertical). "
        "H-beam web in frame plane, flanges point out-of-plane (Z). "
        "Bending occurs about strong axis (Ix). "
        "NOTE: Reversing orientation causes weak-axis bending with much larger displacements. "
        "In Midas Gen, set section Beta Angle = 0 (default)."
    ),
    "nodes": {
        1: (0.0, 0.0), 2: (6.0, 0.0),
        3: (0.0, 3.0), 4: (6.0, 3.0),
    },
    "elements": {"C1": (1, 3), "C2": (2, 4), "B1": (3, 4)},
    "boundary": {1: "fixed", 2: "fixed"},
    "material": MATERIAL_SS275,
    "column_section": SECTION_H350x350,
    "beam_section": SECTION_H400x200,
    "loads": {
        "N3": {"Fx_kN": 25.0, "Fy_kN": -100.0},
        "N4": {"Fx_kN": 25.0, "Fy_kN": -100.0},
    },
}


def run_case2() -> dict:
    """2D portal frame: 1 story, 1 bay, fixed base, lateral + gravity.

    Column: H-350x350x12x19 (DB-resolved)
    Beam: H-400x200x8x13 (DB-resolved)
    Material: SS275 (E=210GPa)
    Section orientation: H-beam web in frame plane (X-Y), flanges out-of-plane (Z).
    Bending about strong axis (Ix). In Midas, Beta Angle = 0.
    """
    E = MATERIAL_SS275["E_MPa"]               # 210000 MPa
    col_A = SECTION_H350x350["A_mm2"]         # 17390 mm2
    col_I = SECTION_H350x350["Ix_mm4"]        # 40300e4 mm4
    bm_A = SECTION_H400x200["A_mm2"]          # 8412 mm2
    bm_I = SECTION_H400x200["Ix_mm4"]         # 23700e4 mm4

    ops.wipe()
    ops.model('basic', '-ndm', 2, '-ndf', 3)

    # Nodes
    ops.node(1, 0.0, 0.0)
    ops.node(2, 6000.0, 0.0)
    ops.node(3, 0.0, 3000.0)
    ops.node(4, 6000.0, 3000.0)

    # Fixed base
    ops.fix(1, 1, 1, 1)
    ops.fix(2, 1, 1, 1)

    ops.geomTransf('Linear', 1)

    # Elements: C1(1-3), C2(2-4), B1(3-4)
    ops.element('elasticBeamColumn', 1, 1, 3, col_A, E, col_I, 1)  # C1
    ops.element('elasticBeamColumn', 2, 2, 4, col_A, E, col_I, 1)  # C2
    ops.element('elasticBeamColumn', 3, 3, 4, bm_A, E, bm_I, 1)   # B1

    # Loads
    ops.timeSeries('Linear', 1)
    ops.pattern('Plain', 1, 1)
    ops.load(3, _kN_to_N(25.0), _kN_to_N(-100.0), 0.0)
    ops.load(4, _kN_to_N(25.0), _kN_to_N(-100.0), 0.0)

    ops.system('BandGeneral')
    ops.numberer('RCM')
    ops.constraints('Plain')
    ops.integrator('LoadControl', 1.0)
    ops.algorithm('Linear')
    ops.analysis('Static')
    ops.analyze(1)

    ops.reactions()
    d3 = ops.nodeDisp(3)
    d4 = ops.nodeDisp(4)
    r1 = ops.nodeReaction(1)
    r2 = ops.nodeReaction(2)

    f_c1 = ops.eleForce(1)  # column 1
    f_c2 = ops.eleForce(2)  # column 2
    f_b1 = ops.eleForce(3)  # beam

    results = {
        "top_disp_N3_dx_mm": d3[0],
        "top_disp_N4_dx_mm": d4[0],
        "top_disp_N3_dy_mm": d3[1],
        "top_disp_N4_dy_mm": d4[1],
        # Column base moments (i-end = base)
        "col1_base_moment_kNm": -f_c1[2] / 1e6,
        "col2_base_moment_kNm": -f_c2[2] / 1e6,
        # Beam end moments
        "beam_moment_i_kNm": -f_b1[2] / 1e6,  # i-end (node 3)
        "beam_moment_j_kNm": -f_b1[5] / 1e6,  # j-end (node 4)
        # Base reactions
        "base_shear_kN": (r1[0] + r2[0]) / 1000.0,
        "reaction_N1_Fx_kN": r1[0] / 1000.0,
        "reaction_N1_Fy_kN": r1[1] / 1000.0,
        "reaction_N1_Mz_kNm": r1[2] / 1e6,
        "reaction_N2_Fx_kN": r2[0] / 1000.0,
        "reaction_N2_Fy_kN": r2[1] / 1000.0,
        "reaction_N2_Mz_kNm": r2[2] / 1e6,
    }

    ops.wipe()
    return results


# ============================================================
# CASE 3: 2D Frame (3-story, 1-bay)
# ============================================================

CASE3_SPEC = {
    "name": "2D Frame (3-story 1-bay)",
    "description": (
        "3-story 1-bay frame in X-Y plane. "
        "Col: H-400x400x13x21 (DB), Beam: H-400x200x8x13 (DB). "
        "Material: SS275 (E=210GPa). "
        "Midas note: XY plane frame cannot use Story definition; "
        "compute drift manually from nodal displacements."
    ),
    "nodes": {
        1: (0, 0), 2: (6, 0),
        3: (0, 3), 4: (6, 3),
        5: (0, 6), 6: (6, 6),
        7: (0, 9), 8: (6, 9),
    },
    "columns": [(1, 3), (3, 5), (5, 7), (2, 4), (4, 6), (6, 8)],
    "beams": [(3, 4), (5, 6), (7, 8)],
    "boundary": {1: "fixed", 2: "fixed"},
    "material": MATERIAL_SS275,
    "column_section": SECTION_H400x400,
    "beam_section": SECTION_H400x200,
    "loads": {
        "N3": {"Fx_kN": 15.0, "Fy_kN": -80.0},
        "N4": {"Fx_kN": 15.0, "Fy_kN": -80.0},
        "N5": {"Fx_kN": 25.0, "Fy_kN": -80.0},
        "N6": {"Fx_kN": 25.0, "Fy_kN": -80.0},
        "N7": {"Fx_kN": 35.0, "Fy_kN": -80.0},
        "N8": {"Fx_kN": 35.0, "Fy_kN": -80.0},
    },
}


def run_case3() -> dict:
    """2D frame: 3-story 1-bay, fixed base, lateral + gravity.

    Column: H-400x400x13x21 (DB-resolved)
    Beam: H-400x200x8x13 (DB-resolved)
    Material: SS275 (E=210GPa)
    """
    E = MATERIAL_SS275["E_MPa"]               # 210000 MPa
    col_A = SECTION_H400x400["A_mm2"]         # 21870 mm2
    col_I = SECTION_H400x400["Ix_mm4"]        # 66600e4 mm4
    bm_A = SECTION_H400x200["A_mm2"]          # 8412 mm2
    bm_I = SECTION_H400x200["Ix_mm4"]         # 23700e4 mm4

    ops.wipe()
    ops.model('basic', '-ndm', 2, '-ndf', 3)

    # Nodes
    coords = {
        1: (0, 0), 2: (6000, 0),
        3: (0, 3000), 4: (6000, 3000),
        5: (0, 6000), 6: (6000, 6000),
        7: (0, 9000), 8: (6000, 9000),
    }
    for nid, (x, y) in coords.items():
        ops.node(nid, float(x), float(y))

    ops.fix(1, 1, 1, 1)
    ops.fix(2, 1, 1, 1)

    ops.geomTransf('Linear', 1)

    eid = 0
    # Columns
    col_pairs = [(1, 3), (3, 5), (5, 7), (2, 4), (4, 6), (6, 8)]
    col_eids = []
    for ni, nj in col_pairs:
        eid += 1
        ops.element('elasticBeamColumn', eid, ni, nj, col_A, E, col_I, 1)
        col_eids.append(eid)

    # Beams
    bm_pairs = [(3, 4), (5, 6), (7, 8)]
    bm_eids = []
    for ni, nj in bm_pairs:
        eid += 1
        ops.element('elasticBeamColumn', eid, ni, nj, bm_A, E, bm_I, 1)
        bm_eids.append(eid)

    # Loads
    ops.timeSeries('Linear', 1)
    ops.pattern('Plain', 1, 1)
    load_data = {
        3: (15.0, -80.0), 4: (15.0, -80.0),
        5: (25.0, -80.0), 6: (25.0, -80.0),
        7: (35.0, -80.0), 8: (35.0, -80.0),
    }
    for nid, (fx, fy) in load_data.items():
        ops.load(nid, _kN_to_N(fx), _kN_to_N(fy), 0.0)

    ops.system('BandGeneral')
    ops.numberer('RCM')
    ops.constraints('Plain')
    ops.integrator('LoadControl', 1.0)
    ops.algorithm('Linear')
    ops.analysis('Static')
    ops.analyze(1)

    # Extract
    ops.reactions()
    d7 = ops.nodeDisp(7)
    d8 = ops.nodeDisp(8)
    r1 = ops.nodeReaction(1)
    r2 = ops.nodeReaction(2)

    # Story drifts (average of left/right column top displacements)
    d3 = ops.nodeDisp(3)
    d4 = ops.nodeDisp(4)
    d5 = ops.nodeDisp(5)
    d6 = ops.nodeDisp(6)

    story_disp = [
        (d3[0] + d4[0]) / 2,                                    # story 1
        ((d5[0] + d6[0]) / 2 - (d3[0] + d4[0]) / 2),          # story 2
        ((d7[0] + d8[0]) / 2 - (d5[0] + d6[0]) / 2),          # story 3
    ]
    story_drift = [sd / 3000.0 for sd in story_disp]  # h=3000mm

    # Column base forces (elements 1=N1-N3, 4=N2-N4)
    f_c1 = ops.eleForce(col_eids[0])  # left col base
    f_c4 = ops.eleForce(col_eids[3])  # right col base

    # Top beam (element for N7-N8)
    f_bm_top = ops.eleForce(bm_eids[2])

    results = {
        "roof_disp_dx_mm": (d7[0] + d8[0]) / 2,
        "roof_disp_N7_dx_mm": d7[0],
        "roof_disp_N8_dx_mm": d8[0],
        "story_drift_1": story_drift[0],
        "story_drift_2": story_drift[1],
        "story_drift_3": story_drift[2],
        "base_shear_kN": (r1[0] + r2[0]) / 1000.0,
        "col1_base_moment_kNm": -f_c1[2] / 1e6,
        "col2_base_moment_kNm": -f_c4[2] / 1e6,
        "top_beam_moment_i_kNm": -f_bm_top[2] / 1e6,
        "top_beam_moment_j_kNm": -f_bm_top[5] / 1e6,
        "reaction_N1_Fx_kN": r1[0] / 1000.0,
        "reaction_N1_Fy_kN": r1[1] / 1000.0,
        "reaction_N1_Mz_kNm": r1[2] / 1e6,
        "reaction_N2_Fx_kN": r2[0] / 1000.0,
        "reaction_N2_Fy_kN": r2[1] / 1000.0,
        "reaction_N2_Mz_kNm": r2[2] / 1e6,
    }

    ops.wipe()
    return results


# ── Case 4 beam section: H-350x175x7x11 ──
# From Supabase ks3502.h_beam_sections
# A=63.14cm2, Ix=13600cm4, Iy=984cm4, h=350, b=175, tw=7, tf=11
# J approx: (2*175*11^3 + 328*7^3)/3 = 19.3e4 mm4
SECTION_H350x175 = {
    "name": "H-350x175x7x11",
    "A_mm2": 6314.0,        # 63.14 cm2
    "Ix_mm4": 13600e4,      # 1.36e8 mm4 (strong axis)
    "Iy_mm4": 984e4,        # 9.84e6 mm4 (weak axis)
    "J_mm4": 19.3e4,        # torsional constant (approx)
    "h_mm": 350.0,
    "b_mm": 175.0,
}

# ── Case 5 column section: H-428x407x20x35 ──
# From Supabase ks3502.h_beam_sections
# A=360.7cm2, Ix=119000cm4, Iy=39400cm4, h=428, b=407, tw=20, tf=35, r=22
# J approx: (2*407*35^3 + 358*20^3)/3 ≈ 1259e4 mm4
SECTION_H428x407 = {
    "name": "H-428x407x20x35",
    "A_mm2": 36070.0,       # 360.7 cm2
    "Ix_mm4": 119000e4,     # 1.19e9 mm4 (strong axis)
    "Iy_mm4": 39400e4,      # 3.94e8 mm4 (weak axis)
    "J_mm4": 1259e4,        # torsional constant (approx)
    "h_mm": 428.0,
    "b_mm": 407.0,
}


# ============================================================
# CASE 4: 3D Frame (2-story, 1x1 bay)
# ============================================================

CASE4_SPEC = {
    "name": "3D Frame (2-story 1x1bay)",
    "description": (
        "3D linear frame, X=lateral, Y=orthogonal horizontal, Z=vertical. "
        "Col: H-400x400x13x21 (DB), Beam: H-350x175x7x11 (DB). "
        "Material: SS275 (E=210GPa). "
        "Section orientation: web vertical for beams (vecxz=0,0,1), "
        "column strong axis resists X-lateral (vecxz=1,0,0). "
        "Beam connectivity normalized: beam_x from smaller X, beam_y from smaller Y. "
        "Midas: shear deformation OFF, Beta Angle=0."
    ),
    "nodes": {
        # Base z=0
        1: (0, 0, 0), 2: (6, 0, 0), 3: (6, 6, 0), 4: (0, 6, 0),
        # Story1 z=3
        5: (0, 0, 3), 6: (6, 0, 3), 7: (6, 6, 3), 8: (0, 6, 3),
        # Roof z=6
        9: (0, 0, 6), 10: (6, 0, 6), 11: (6, 6, 6), 12: (0, 6, 6),
    },
    "columns": [
        "C1=N1-N5", "C2=N5-N9", "C3=N2-N6", "C4=N6-N10",
        "C5=N3-N7", "C6=N7-N11", "C7=N4-N8", "C8=N8-N12",
    ],
    "beams_1f": ["B1=N5-N6(X)", "B2=N6-N7(Y)", "B3=N8-N7(X)", "B4=N5-N8(Y)"],
    "beams_roof": ["B5=N9-N10(X)", "B6=N10-N11(Y)", "B7=N12-N11(X)", "B8=N9-N12(Y)"],
    "boundary": {1: "fixed", 2: "fixed", 3: "fixed", 4: "fixed"},
    "material": MATERIAL_SS275,
    "column_section": SECTION_H400x400,
    "beam_section": SECTION_H350x175,
    "loads": {
        "story1": {"Fx_kN": 10.0, "Fz_kN": -60.0},
        "roof": {"Fx_kN": 20.0, "Fz_kN": -80.0},
    },
}


def run_case4() -> dict:
    """3D frame: 2-story 1x1bay, fixed base, lateral+gravity.

    Column: H-400x400x13x21 (DB-resolved)
    Beam: H-350x175x7x11 (DB-resolved)
    Material: SS275 (E=210GPa, G=80769MPa)

    Coordinate system: X=lateral, Y=in-plane orthogonal, Z=vertical(up)

    Section orientation (OpenSees local axes):
      Columns (vertical, vecxz=1,0,0):
        local y = global Y, local z = -global X
        → Iy_elem = Ix_section (strong axis, resists X-lateral)
        → Iz_elem = Iy_section (weak axis, resists Y-lateral)
      Beams (horizontal, vecxz=0,0,1):
        local y = ±global Y or ±global X, local z = -global Z
        → Iy_elem = Ix_section (strong axis, resists gravity)
        → Iz_elem = Iy_section (weak axis, resists lateral)

    Beam connectivity normalized:
      beam_x: i-node has smaller X than j-node
      beam_y: i-node has smaller Y than j-node
    """
    E = MATERIAL_SS275["E_MPa"]               # 210000 MPa
    G = MATERIAL_SS275["G_MPa"]               # 80769 MPa

    # Column: H-400x400x13x21
    col_A = SECTION_H400x400["A_mm2"]         # 21870 mm2
    col_Iy = SECTION_H400x400["Ix_mm4"]       # 66600e4 (strong axis → Iy_elem)
    col_Iz = SECTION_H400x400["Iy_mm4"]       # 22400e4 (weak axis → Iz_elem)
    col_J = SECTION_H400x400["J_mm4"]         # 498e4

    # Beam: H-350x175x7x11
    bm_A = SECTION_H350x175["A_mm2"]          # 6314 mm2
    bm_Iy = SECTION_H350x175["Ix_mm4"]        # 13600e4 (strong axis → Iy_elem)
    bm_Iz = SECTION_H350x175["Iy_mm4"]        # 984e4 (weak axis → Iz_elem)
    bm_J = SECTION_H350x175["J_mm4"]          # 19.3e4

    ops.wipe()
    ops.model('basic', '-ndm', 3, '-ndf', 6)

    # Nodes (mm): X=lateral, Y=orthogonal, Z=vertical
    node_coords = {
        1: (0, 0, 0), 2: (6000, 0, 0), 3: (6000, 6000, 0), 4: (0, 6000, 0),
        5: (0, 0, 3000), 6: (6000, 0, 3000), 7: (6000, 6000, 3000), 8: (0, 6000, 3000),
        9: (0, 0, 6000), 10: (6000, 0, 6000), 11: (6000, 6000, 6000), 12: (0, 6000, 6000),
    }
    for nid, (x, y, z) in node_coords.items():
        ops.node(nid, float(x), float(y), float(z))

    # Fixed base (N1~N4)
    for nid in [1, 2, 3, 4]:
        ops.fix(nid, 1, 1, 1, 1, 1, 1)

    # Geometric transformations
    # Columns: vertical (along Z), vecxz = (1,0,0) → local z ≈ -global X
    ops.geomTransf('Linear', 1, 1.0, 0.0, 0.0)
    # Beams along X: vecxz = (0,0,1) → local z = -global Z (web vertical)
    ops.geomTransf('Linear', 2, 0.0, 0.0, 1.0)
    # Beams along Y: vecxz = (0,0,1) → local z = -global Z (web vertical)
    ops.geomTransf('Linear', 3, 0.0, 0.0, 1.0)

    eid = 0

    # Columns: C1~C8
    col_pairs = [
        (1, 5), (5, 9),    # C1, C2 (corner 1)
        (2, 6), (6, 10),   # C3, C4 (corner 2)
        (3, 7), (7, 11),   # C5, C6 (corner 3)
        (4, 8), (8, 12),   # C7, C8 (corner 4)
    ]
    col_eids = []
    for ni, nj in col_pairs:
        eid += 1
        ops.element('elasticBeamColumn', eid, ni, nj,
                     col_A, E, G, col_J, col_Iy, col_Iz, 1)
        col_eids.append(eid)

    # 1F beams: normalized connectivity (beam_x: smaller X→larger X, beam_y: smaller Y→larger Y)
    # B1=N5-N6(X), B2=N6-N7(Y), B3=N8-N7(X), B4=N5-N8(Y)
    floor_bm_spec = [
        (5, 6, 2),   # B1: X-dir (X:0→6000)
        (6, 7, 3),   # B2: Y-dir (Y:0→6000)
        (8, 7, 2),   # B3: X-dir (X:0→6000)  — normalized from N8(X=0) to N7(X=6000)
        (5, 8, 3),   # B4: Y-dir (Y:0→6000)
    ]
    floor_bm_eids = []
    for ni, nj, transf in floor_bm_spec:
        eid += 1
        ops.element('elasticBeamColumn', eid, ni, nj,
                     bm_A, E, G, bm_J, bm_Iy, bm_Iz, transf)
        floor_bm_eids.append(eid)

    # Roof beams: normalized connectivity
    # B5=N9-N10(X), B6=N10-N11(Y), B7=N12-N11(X), B8=N9-N12(Y)
    roof_bm_spec = [
        (9, 10, 2),   # B5: X-dir (X:0→6000)
        (10, 11, 3),  # B6: Y-dir (Y:0→6000)
        (12, 11, 2),  # B7: X-dir (X:0→6000)  — normalized from N12(X=0) to N11(X=6000)
        (9, 12, 3),   # B8: Y-dir (Y:0→6000)
    ]
    roof_bm_eids = []
    for ni, nj, transf in roof_bm_spec:
        eid += 1
        ops.element('elasticBeamColumn', eid, ni, nj,
                     bm_A, E, G, bm_J, bm_Iy, bm_Iz, transf)
        roof_bm_eids.append(eid)

    # Loads (equivalent nodal loads only)
    ops.timeSeries('Linear', 1)
    ops.pattern('Plain', 1, 1)

    # 1F nodes (N5~N8): Fx=+10kN, Fz=-60kN each
    for nid in [5, 6, 7, 8]:
        ops.load(nid, _kN_to_N(10.0), 0.0, _kN_to_N(-60.0), 0.0, 0.0, 0.0)

    # Roof nodes (N9~N12): Fx=+20kN, Fz=-80kN each
    for nid in [9, 10, 11, 12]:
        ops.load(nid, _kN_to_N(20.0), 0.0, _kN_to_N(-80.0), 0.0, 0.0, 0.0)

    ops.system('BandGeneral')
    ops.numberer('RCM')
    ops.constraints('Plain')
    ops.integrator('LoadControl', 1.0)
    ops.algorithm('Linear')
    ops.analysis('Static')
    ops.analyze(1)

    # Extract results
    ops.reactions()

    # Roof displacements (X-direction)
    roof_dx = [ops.nodeDisp(n)[0] for n in [9, 10, 11, 12]]

    # Story 1 displacements (X-direction)
    s1_dx = [ops.nodeDisp(n)[0] for n in [5, 6, 7, 8]]

    # Story drifts
    avg_s1_dx = sum(s1_dx) / 4
    avg_roof_dx = sum(roof_dx) / 4
    drift_1 = avg_s1_dx / 3000.0
    drift_2 = (avg_roof_dx - avg_s1_dx) / 3000.0

    # Base reactions (sum over N1~N4)
    base_rx = sum(ops.nodeReaction(n)[0] for n in [1, 2, 3, 4])
    base_rz = sum(ops.nodeReaction(n)[2] for n in [1, 2, 3, 4])

    # Column base forces — C1 (element 1, N1→N5, i-end = base)
    # 3D eleForce: [Nx, Vy, Vz, Tx, My, Mz, Nx, Vy, Vz, Tx, My, Mz]
    f_c1 = ops.eleForce(col_eids[0])

    # Roof beam — B5 (first roof beam, N9→N10, X-direction)
    f_rb1 = ops.eleForce(roof_bm_eids[0])

    # Per-node reactions
    reactions = {}
    for nid in [1, 2, 3, 4]:
        r = ops.nodeReaction(nid)
        reactions[f"N{nid}"] = {
            "Fx_kN": r[0] / 1000.0,
            "Fy_kN": r[1] / 1000.0,
            "Fz_kN": r[2] / 1000.0,
            "Mx_kNm": r[3] / 1e6,
            "My_kNm": r[4] / 1e6,
            "Mz_kNm": r[5] / 1e6,
        }

    results = {
        "roof_max_dx_mm": max(roof_dx),
        "roof_avg_dx_mm": avg_roof_dx,
        "story1_avg_dx_mm": avg_s1_dx,
        "story_drift_1": drift_1,
        "story_drift_2": drift_2,
        "base_reaction_X_kN": base_rx / 1000.0,
        "base_reaction_Z_kN": base_rz / 1000.0,
        # Column 1 base moment (i-end): My resists X-lateral, Mz resists Y-lateral
        "col1_base_My_kNm": f_c1[4] / 1e6,
        "col1_base_Mz_kNm": f_c1[5] / 1e6,
        # Roof beam B5 end moments (My = gravity bending)
        "roof_beam1_My_i_kNm": f_rb1[4] / 1e6,
        "roof_beam1_My_j_kNm": f_rb1[10] / 1e6,
        "reactions": reactions,
    }

    ops.wipe()
    return results


# ============================================================
# CASE 5: 3D Frame (5-story, 1x1 bay, P-Delta)
# ============================================================

CASE5_SPEC = {
    "name": "3D Frame (5-story 1x1bay, P-Delta)",
    "description": (
        "3D 5-story 1x1bay frame, X=lateral, Y=orthogonal, Z=vertical. "
        "Col: H-428x407x20x35 (DB), Beam: H-400x200x8x13 (DB). "
        "Material: SS275 (E=210GPa, G=80769MPa). "
        "Two analysis modes: Linear and P-Delta (Corotational). "
        "Section orientation same as Case 4. "
        "Beam connectivity normalized: beam_x smaller X, beam_y smaller Y. "
        "Midas: shear deformation OFF, Beta Angle=0."
    ),
    "nodes": "24 nodes: 4 corners x 6 levels (z=0,3,6,9,12,15m)",
    "columns": "20 columns: 4 corners x 5 stories",
    "beams": "20 beams: 4 per floor x 5 floors (normalized connectivity)",
    "boundary": {1: "fixed", 2: "fixed", 3: "fixed", 4: "fixed"},
    "material": MATERIAL_SS275,
    "column_section": SECTION_H428x407,
    "beam_section": SECTION_H400x200,
    "loads_per_story": {
        1: {"Fx_kN": 8.0, "Fz_kN": -70.0},
        2: {"Fx_kN": 12.0, "Fz_kN": -70.0},
        3: {"Fx_kN": 16.0, "Fz_kN": -70.0},
        4: {"Fx_kN": 20.0, "Fz_kN": -70.0},
        5: {"Fx_kN": 24.0, "Fz_kN": -70.0},
    },
    "total_loads": {"FX_kN": 320.0, "FZ_kN": -1400.0},
}


def _build_case5_model(geometric_nonlinearity: str = "linear") -> tuple[dict, list, list]:
    """Build 3D 5-story model with DB-resolved sections.

    Returns (node_coords, col_eids, roof_bm_eids).

    Column: H-428x407x20x35 (DB-resolved)
    Beam: H-400x200x8x13 (DB-resolved)
    Material: SS275 (E=210GPa, G=80769MPa)

    Section orientation (same as Case 4):
      Columns (vertical, vecxz=1,0,0):
        Iy_elem = Ix_section (strong, resists X-lateral)
        Iz_elem = Iy_section (weak, resists Y-lateral)
      Beams (horizontal, vecxz=0,0,1):
        Iy_elem = Ix_section (strong, resists gravity)
        Iz_elem = Iy_section (weak, resists lateral)

    Beam connectivity normalized:
      beam_x: i-node has smaller X than j-node
      beam_y: i-node has smaller Y than j-node
    """
    E = MATERIAL_SS275["E_MPa"]               # 210000 MPa
    G = MATERIAL_SS275["G_MPa"]               # 80769 MPa

    # Column: H-428x407x20x35
    col_A = SECTION_H428x407["A_mm2"]         # 36070 mm2
    col_Iy = SECTION_H428x407["Ix_mm4"]       # 119000e4 (strong → Iy_elem)
    col_Iz = SECTION_H428x407["Iy_mm4"]       # 39400e4 (weak → Iz_elem)
    col_J = SECTION_H428x407["J_mm4"]         # 1259e4

    # Beam: H-400x200x8x13
    bm_A = SECTION_H400x200["A_mm2"]          # 8412 mm2
    bm_Iy = SECTION_H400x200["Ix_mm4"]        # 23700e4 (strong → Iy_elem)
    bm_Iz = SECTION_H400x200["Iy_mm4"]        # 1740e4 (weak → Iz_elem)
    bm_J = SECTION_H400x200["J_mm4"]          # 66e4

    n_stories = 5
    h = 3000.0  # mm
    plan_x = 6000.0
    plan_y = 6000.0

    ops.wipe()
    ops.model('basic', '-ndm', 3, '-ndf', 6)

    # Nodes: 4 corners per level, 6 levels (z=0,3000,...,15000)
    node_coords = {}
    corners = [(0.0, 0.0), (plan_x, 0.0), (plan_x, plan_y), (0.0, plan_y)]
    for level in range(n_stories + 1):
        z = level * h
        for ci, (cx, cy) in enumerate(corners):
            nid = level * 4 + ci + 1
            ops.node(nid, cx, cy, z)
            node_coords[nid] = (cx, cy, z)

    # Fixed base (N1~N4)
    for n in range(1, 5):
        ops.fix(n, 1, 1, 1, 1, 1, 1)

    # Geometric transformations
    use_pdelta = (geometric_nonlinearity == "pdelta")
    transf_type = 'Corotational' if use_pdelta else 'Linear'

    # Columns: vertical (along Z), vecxz=(1,0,0) → local z ≈ -global X
    ops.geomTransf(transf_type, 1, 1.0, 0.0, 0.0)
    # Beams X-dir: vecxz=(0,0,1) → local z ≈ -global Z (web vertical)
    ops.geomTransf(transf_type, 2, 0.0, 0.0, 1.0)
    # Beams Y-dir: vecxz=(0,0,1) → local z ≈ -global Z (web vertical)
    ops.geomTransf(transf_type, 3, 0.0, 0.0, 1.0)

    eid = 0
    col_eids = []

    # Columns: 4 corners × 5 stories = 20
    for corner in range(4):
        for story in range(n_stories):
            ni = story * 4 + corner + 1
            nj = (story + 1) * 4 + corner + 1
            eid += 1
            ops.element('elasticBeamColumn', eid, ni, nj,
                         col_A, E, G, col_J, col_Iy, col_Iz, 1)
            col_eids.append(eid)

    # Beams: 4 per floor × 5 floors = 20, normalized connectivity
    bm_eids_all = []
    for level in range(1, n_stories + 1):
        n1 = level * 4 + 1  # (0, 0, z)
        n2 = level * 4 + 2  # (6000, 0, z)
        n3 = level * 4 + 3  # (6000, 6000, z)
        n4 = level * 4 + 4  # (0, 6000, z)
        # Normalized: beam_x from smaller X, beam_y from smaller Y
        bm_spec = [
            (n1, n2, 2),  # B1: X-dir (X:0→6000)
            (n2, n3, 3),  # B2: Y-dir (Y:0→6000)
            (n4, n3, 2),  # B3: X-dir (X:0→6000, normalized n4→n3)
            (n1, n4, 3),  # B4: Y-dir (Y:0→6000)
        ]
        for ni, nj, transf in bm_spec:
            eid += 1
            ops.element('elasticBeamColumn', eid, ni, nj,
                         bm_A, E, G, bm_J, bm_Iy, bm_Iz, transf)
            bm_eids_all.append(eid)

    # Loads
    ops.timeSeries('Linear', 1)
    ops.pattern('Plain', 1, 1)

    story_fx = {1: 8.0, 2: 12.0, 3: 16.0, 4: 20.0, 5: 24.0}
    fz_per_node = _kN_to_N(-70.0)

    for story in range(1, n_stories + 1):
        fx = _kN_to_N(story_fx[story])
        for corner in range(4):
            nid = story * 4 + corner + 1
            ops.load(nid, fx, 0.0, fz_per_node, 0.0, 0.0, 0.0)

    # Roof beam eids (last 4 beams)
    roof_bm_eids = bm_eids_all[-4:]

    return node_coords, col_eids, roof_bm_eids


def _solve_case5(geometric_nonlinearity: str = "linear") -> int:
    """Solve with appropriate algorithm."""
    ops.system('BandGeneral')
    ops.numberer('RCM')
    ops.constraints('Plain')

    if geometric_nonlinearity == "pdelta":
        ops.test('NormDispIncr', 1.0e-8, 50)
        n_steps = 10
        ops.integrator('LoadControl', 1.0 / n_steps)
        ops.algorithm('Newton')
        ops.analysis('Static')
        ok = ops.analyze(n_steps)
        if ok != 0:
            # Fallback
            ops.algorithm('ModifiedNewton')
            ops.integrator('LoadControl', 1.0 / 50)
            ok = ops.analyze(50)
    else:
        ops.integrator('LoadControl', 1.0)
        ops.algorithm('Linear')
        ops.analysis('Static')
        ok = ops.analyze(1)
    return ok


def _extract_case5(node_coords, col_eids, roof_bm_eids) -> dict:
    """Extract results for case 5."""
    n_stories = 5
    h = 3000.0

    # Per-story average X displacements
    story_avg_dx = []
    for story in range(n_stories + 1):
        nodes = [story * 4 + c + 1 for c in range(4)]
        avg_dx = sum(ops.nodeDisp(n)[0] for n in nodes) / 4
        story_avg_dx.append(avg_dx)

    roof_dx = [ops.nodeDisp(n)[0] for n in [n_stories * 4 + c + 1 for c in range(4)]]

    # Story drifts
    story_drifts = []
    for s in range(1, n_stories + 1):
        drift = (story_avg_dx[s] - story_avg_dx[s - 1]) / h
        story_drifts.append(drift)

    # Base reactions
    ops.reactions()
    base_rx = sum(ops.nodeReaction(n)[0] for n in [1, 2, 3, 4])
    base_rz = sum(ops.nodeReaction(n)[2] for n in [1, 2, 3, 4])

    # Column 1 base moment
    f_c1 = ops.eleForce(col_eids[0])

    # Roof beam 1 moment
    f_rb1 = ops.eleForce(roof_bm_eids[0])

    # Per-node reactions
    reactions = {}
    for nid in [1, 2, 3, 4]:
        r = ops.nodeReaction(nid)
        reactions[f"N{nid}"] = {
            "Fx_kN": r[0] / 1000.0, "Fy_kN": r[1] / 1000.0, "Fz_kN": r[2] / 1000.0,
            "Mx_kNm": r[3] / 1e6, "My_kNm": r[4] / 1e6, "Mz_kNm": r[5] / 1e6,
        }

    return {
        "roof_max_dx_mm": max(roof_dx),
        "roof_avg_dx_mm": story_avg_dx[n_stories],
        "story_avg_dx_mm": [story_avg_dx[s] for s in range(1, n_stories + 1)],
        "story_drifts": story_drifts,
        "base_reaction_X_kN": base_rx / 1000.0,
        "base_reaction_Z_kN": base_rz / 1000.0,
        "col1_base_My_kNm": f_c1[4] / 1e6,
        "roof_beam1_My_i_kNm": f_rb1[4] / 1e6,
        "reactions": reactions,
    }


def run_case5() -> dict:
    """3D 5-story P-Delta benchmark: both linear and nonlinear.

    Column: H-428x407x20x35 (DB-resolved)
    Beam: H-400x200x8x13 (DB-resolved)
    Material: SS275 (E=210GPa, G=80769MPa)
    """
    # Linear analysis
    node_coords, col_eids, roof_bm_eids = _build_case5_model("linear")
    _solve_case5("linear")
    linear = _extract_case5(node_coords, col_eids, roof_bm_eids)
    ops.wipe()

    # P-Delta analysis (Corotational)
    node_coords, col_eids, roof_bm_eids = _build_case5_model("pdelta")
    ok = _solve_case5("pdelta")
    pdelta = _extract_case5(node_coords, col_eids, roof_bm_eids)
    ops.wipe()

    # Amplification
    lin_roof = linear["roof_avg_dx_mm"]
    pd_roof = pdelta["roof_avg_dx_mm"]
    amp_disp = pd_roof / lin_roof if abs(lin_roof) > 1e-12 else 0.0

    drift_amps = []
    for i in range(5):
        ld = linear["story_drifts"][i]
        pd = pdelta["story_drifts"][i]
        drift_amps.append(pd / ld if abs(ld) > 1e-12 else 0.0)

    return {
        "linear": linear,
        "pdelta": pdelta,
        "displacement_amplification": amp_disp,
        "drift_amplification": drift_amps,
        "pdelta_solver_ok": (ok == 0),
        # Equilibrium check
        "total_lateral_kN": sum([8, 12, 16, 20, 24]) * 4,  # 320 kN
        "total_gravity_kN": -70.0 * 4 * 5,  # -1400 kN
        "base_rx_linear_kN": linear["base_reaction_X_kN"],
        "base_rx_pdelta_kN": pdelta["base_reaction_X_kN"],
        "base_rz_linear_kN": linear["base_reaction_Z_kN"],
        "base_rz_pdelta_kN": pdelta["base_reaction_Z_kN"],
    }


# ── Registry ──
ALL_CASES = {
    "case1": ("2D Simple Beam (Point Load)", run_case1),
    "case2": ("2D Portal Frame (1S-1B)", run_case2),
    "case3": ("2D Frame (3S-1B)", run_case3),
    "case4": ("3D Frame (2S-1x1B)", run_case4),
    "case5": ("3D Frame (5S-1x1B, P-Delta)", run_case5),
}

# Optional variants (not included in default benchmark suite)
# Decision-gate cases (Case 6+) are also placed here — opt-in via CLI arg.
from case6_lshape import run_case6_lshape  # noqa: E402

OPTIONAL_CASES = {
    "case1_udl": ("2D Simple Beam (UDL)", run_case1_udl),
    "case6_lshape": ("3D L-Shape 5F+3F Setback (Decision-Gate)", run_case6_lshape),
}
