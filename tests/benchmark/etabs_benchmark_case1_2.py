"""ETABS 23 benchmark runner — Case 1 (simple beam) & Case 2 (portal frame).

Builds each model fresh inside ETABS, runs linear static analysis, extracts
key metrics, and compares against saved Midas Gen reference values.

Usage (ETABS must be running and licensed):
    cd d:\\son\\opensees-MCP
    .\\opensees-mcp\\Scripts\\python.exe tests/benchmark/etabs_benchmark_case1_2.py
    .\\opensees-mcp\\Scripts\\python.exe tests/benchmark/etabs_benchmark_case1_2.py --launch
    .\\opensees-mcp\\Scripts\\python.exe tests/benchmark/etabs_benchmark_case1_2.py case1
    .\\opensees-mcp\\Scripts\\python.exe tests/benchmark/etabs_benchmark_case1_2.py case2

Coordinate system: X = span/bay, Y = out-of-plane, Z = vertical (up)
Units: kN, m  (ETABS kN_m_C = 6) — matches PropFrame.ImportProp's internal
       storage of the cm-based KoreanKS21.xml library, so KS sections work
       without modifiers.  Displacements are converted m→mm only at the
       output stage to match Midas/OpenSees reporting units.

Sign convention — ETABS local axes vs Midas 2D convention:
  Horizontal beam along +X:  local2 = +Z, local3 = -Y
    → M3 at j-end is NEGATIVE for sagging  ✓  matches Midas
  Vertical column along +Z:  local2 = +X, local3 = +Y
    → M3 at base is NEGATIVE for +X lateral  ✓  matches Midas
  JointReact M2 (about global Y) = in-plane restraint moment  ≡  Midas Mz
"""
from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "mcp-server"))

# Windows cp949 terminals choke on box-drawing/emoji chars — force UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from core.etabs_api import ETABSClient   # noqa: E402

BENCH_DIR = Path(__file__).parent
MIDAS_DIR = BENCH_DIR / "midas_results"
ETABS_DIR = BENCH_DIR / "etabs_results"

# ── ETABS enum constants ──────────────────────────────────────────────────
# ETABS runs entirely in kN-m-C (the library's SI base):
#   - PropFrame.ImportProp stores cm-library values directly as kN-m
#     regardless of current model units → using kN-m avoids the cm/N-mm
#     unit-conversion bug observed when running in N-mm-C
#   - Result extraction returns m / kN / kN·m → only displacements need
#     a ×1000 conversion to mm at output time; forces and moments already
#     match Midas/OpenSees reporting units (kN, kN·m)
_KN_M_C   = 6     # eUnits: kN, m, °C  — model working units
_STEEL    = 1     # eMatType_Steel
_LP_OTHER = 8     # eLoadPatternType_Other
_MAT      = "SS275"
_KS_LIB   = "KoreanKS21.xml"   # KS D 3502:2021 — ships with ETABS 23



# ─────────────────────────────────────────────────────────────────────────
# Low-level model building helpers
# ─────────────────────────────────────────────────────────────────────────

def _init(model) -> None:
    """Blank model, set kN-m units.

    ETABS API requires InitializeNewModel(units) → File.NewBlank() sequence.
    """
    model.InitializeNewModel(_KN_M_C)   # eUnits = 6 (kN_m_C)
    ret = model.File.NewBlank()
    if ret != 0:
        raise RuntimeError(f"File.NewBlank failed (ret={ret})")


def _material(model) -> None:
    """SS275: E = 2.1×10⁸ kN/m² (≡ 210 GPa ≡ 210 000 N/mm²), ν = 0.3."""
    if model.PropMaterial.SetMaterial(_MAT, _STEEL) != 0:
        raise RuntimeError("SetMaterial SS275 failed")
    if model.PropMaterial.SetMPIsotropic(_MAT, 210_000_000.0, 0.3, 1.2e-5) != 0:
        raise RuntimeError("SetMPIsotropic SS275 failed")


def _ks_section(model, name: str, ks_label: str) -> None:
    """Import a section directly from the KS D 3502:2021 library.

    ks_label format (matches KoreanKS21.xml <LABEL>):
        'H 400x200x8/13'    (h × bf × tw / tf)
        'H 350x350x12/19'

    Works correctly because the model is in kN-m, which is what
    PropFrame.ImportProp stores cm-library values in regardless of
    current units (a quirk verified by scripts/diag_etabs_importprop.py).

    Only modifier applied: As2 = As3 = 0 → Euler-Bernoulli beam theory
    (matches OpenSees elasticBeamColumn / Midas Gen default).  All other
    properties (A, I22, I33, J, S, Z, R) are used as KS-tabulated.
    """
    ret = model.PropFrame.ImportProp(name, _MAT, _KS_LIB, ks_label)
    if ret != 0:
        raise RuntimeError(
            f"ImportProp '{name}' ← '{ks_label}' from {_KS_LIB} failed (ret={ret})"
        )
    # Modifiers: [A, As2, As3, J, I22, I33, mass, weight] — zero shear areas.
    *_, ret2 = model.PropFrame.SetModifiers(
        name, [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    )
    if ret2 != 0:
        raise RuntimeError(f"SetModifiers '{name}' failed (ret={ret2})")


def _pt(model, name: str, x: float, y: float, z: float) -> str:
    """Add a joint and return its name."""
    n, ret = model.PointObj.AddCartesian(x, y, z, name)
    if ret != 0:
        raise RuntimeError(f"AddCartesian '{name}' failed (ret={ret})")
    return n


def _frame(model, name: str, i_node: str, j_node: str, sec: str) -> str:
    """Add a frame element, return its assigned name.

    Also disables ETABS automatic End Length Offsets (default = half of the
    connecting column depth).  Without this, beam moments are reported at the
    column FACE — not at the joint centerline — and differ from Midas/OpenSees
    by V_beam × offset.
    """
    n, ret = model.FrameObj.AddByPoint(i_node, j_node, name, sec, "")
    if ret != 0:
        raise RuntimeError(f"AddByPoint '{name}' failed (ret={ret})")
    # AutoOffset=False, Length1=Length2=0, RigidZoneFactor=0
    ret2 = model.FrameObj.SetEndLengthOffset(n, False, 0.0, 0.0, 0.0)
    if ret2 != 0:
        raise RuntimeError(f"SetEndLengthOffset '{n}' failed (ret={ret2})")
    return n


def _restrain(model, node: str, dofs: list) -> None:
    """Apply restraints.  dofs = [U1, U2, U3, R1, R2, R3], True = fixed."""
    _, ret = model.PointObj.SetRestraint(node, dofs)
    if ret != 0:
        raise RuntimeError(f"SetRestraint '{node}' failed (ret={ret})")


def _load_pattern(model, lp: str) -> None:
    """Add a load pattern + matching linear static load case."""
    ret = model.LoadPatterns.Add(lp, _LP_OTHER, 0.0, True)
    if ret != 0:
        raise RuntimeError(f"LoadPatterns.Add '{lp}' failed (ret={ret})")
    # Explicitly wire the load pattern to the auto-created load case.
    # SetLoads returns (LoadType_out, LoadName_out, SF_out, retval).
    *_, ret2 = model.LoadCases.StaticLinear.SetLoads(lp, 1, ["Load"], [lp], [1.0])
    if ret2 != 0:
        raise RuntimeError(f"StaticLinear.SetLoads '{lp}' failed (ret={ret2})")


def _joint_load(model, node: str, lp: str, forces: list) -> None:
    """Apply joint load.  forces = [F1, F2, F3, M1, M2, M3] in kN / kN·m."""
    _, ret = model.PointObj.SetLoadForce(node, lp, forces)
    if ret != 0:
        raise RuntimeError(f"SetLoadForce '{node}' failed (ret={ret})")


def _run(model) -> None:
    """Save to a temp path then run analysis.

    ETABS writes result files alongside the .edb, so an unsaved blank model
    has no output directory and RunAnalysis returns ret=1.
    """
    import tempfile
    tmp = Path(tempfile.gettempdir()) / "etabs_bench_tmp.edb"
    model.File.Save(str(tmp))       # give ETABS a directory to write results
    model.SetModelIsLocked(False)
    ret = model.Analyze.RunAnalysis()
    if ret != 0:
        raise RuntimeError(f"RunAnalysis failed (ret={ret})")


def _select_case(model, case: str) -> None:
    """Select a single load case for results output."""
    setup = model.Results.Setup
    setup.DeselectAllCasesAndCombosForOutput()
    setup.SetCaseSelectedForOutput(case, True)


# ─────────────────────────────────────────────────────────────────────────
# Result extraction helpers
# ─────────────────────────────────────────────────────────────────────────

def _displ(model, node: str) -> tuple:
    """Return (U1, U2, U3, R1, R2, R3) for the first result row (m / rad)."""
    (n, obj, elm, lc, st, sn,
     u1, u2, u3, r1, r2, r3, ret) = model.Results.JointDispl(
        node, 0, 0, [], [], [], [], [], [], [], [], [], []
    )
    if ret != 0:
        raise RuntimeError(f"JointDispl '{node}' failed (ret={ret})")
    def _f(arr):
        return float(list(arr)[0]) if arr else 0.0
    return _f(u1), _f(u2), _f(u3), _f(r1), _f(r2), _f(r3)


def _react(model, node: str) -> tuple:
    """Return (F1, F2, F3, M1, M2, M3) for the first result row (kN / kN·m)."""
    (n, obj, elm, lc, st, sn,
     f1, f2, f3, m1, m2, m3, ret) = model.Results.JointReact(
        node, 0, 0, [], [], [], [], [], [], [], [], [], []
    )
    if ret != 0:
        raise RuntimeError(f"JointReact '{node}' failed (ret={ret})")
    def _f(arr):
        return float(list(arr)[0]) if arr else 0.0
    return _f(f1), _f(f2), _f(f3), _f(m1), _f(m2), _f(m3)


def _m3_at(model, elem: str, target_sta: float) -> float:
    """M3 (kN·m) at the nearest output station to target_sta (m from i-end).

    ETABS outputs FrameForce at several stations per element; we pick the
    station closest to target_sta.  A warning is printed if the closest
    station is more than 0.1 m away.
    """
    (n, obj, obj_sta, elm, elm_sta, lc, st, sn,
     p, v2, v3, t, m2, m3, ret) = model.Results.FrameForce(
        elem, 0, 0, [], [], [], [], [], [], [], [], [], [], [], [], []
    )
    if ret != 0:
        raise RuntimeError(f"FrameForce '{elem}' failed (ret={ret})")

    sta  = list(obj_sta) if obj_sta else []
    m3l  = list(m3)      if m3      else []
    if not sta:
        return 0.0

    idx  = min(range(len(sta)), key=lambda i: abs(sta[i] - target_sta))
    dist = abs(sta[idx] - target_sta)
    if dist > 0.1:
        print(f"  WARNING: '{elem}' nearest station {sta[idx]:.3f} m is "
              f"{dist:.3f} m from target {target_sta:.3f} m")
    return float(m3l[idx])


# ─────────────────────────────────────────────────────────────────────────
# Case 1: 2D Simple Beam
# ─────────────────────────────────────────────────────────────────────────

def run_case1_etabs(client) -> dict:
    """3-node simple beam, 6 m span, 60 kN point load at midspan.

    Layout (X-axis, Z=0):
        N1(0,0,0) —E1— N2(3,0,0) —E2— N3(6,0,0)         [m]
    BC:   pin at N1  [U1/U2/U3 fixed, R free]
          roller at N3 [U2/U3 fixed, U1/R free]
    Load: Fz = −60 kN at N2

    Expected (analytical):
        Midspan disp  = −5.425 mm
        Midspan M     = −90 kN·m   (sagging → M3 < 0 in ETABS local)
        Reactions     = ±30 kN (Fz), 0 (Fx)
    """
    m = client.model
    _init(m)
    _material(m)
    # H 400x200x8/13 imported from KS D 3502:2021 library (KoreanKS21.xml) —
    # exact KS tabulated values, no manual modifier needed.
    _ks_section(m, "H400x200", "H 400x200x8/13")

    # Coordinates in metres (model is in kN-m).
    n1 = _pt(m, "N1", 0.0, 0.0, 0.0)
    n2 = _pt(m, "N2", 3.0, 0.0, 0.0)
    n3 = _pt(m, "N3", 6.0, 0.0, 0.0)

    e1 = _frame(m, "E1", n1, n2, "H400x200")
    _frame(m, "E2", n2, n3, "H400x200")

    # ETABS auto-pins all nodes at Z=0 (base level). Release the interior node.
    _restrain(m, n2, [False, False, False, False, False, False])  # free midspan
    _restrain(m, n1, [True,  True,  True,  False, False, False])  # pin
    _restrain(m, n3, [False, True,  True,  False, False, False])  # roller

    _load_pattern(m, "CASE1")
    # Load in kN: Fz = −60 kN at midspan.
    _joint_load(m, n2, "CASE1", [0.0, 0.0, -60.0, 0.0, 0.0, 0.0])

    _run(m)
    _select_case(m, "CASE1")

    _, _, u3_N2, _, _, _ = _displ(m, n2)        # m
    f1_N1, _, f3_N1, _, _, _ = _react(m, n1)    # kN
    _, _, f3_N3, _, _, _ = _react(m, n3)        # kN

    # M3 at j-end of E1 (station = 3.0 m = midspan of full beam).  kN·m.
    m3_mid = _m3_at(m, e1, 3.0)

    return {
        # Displacements in mm: ×1000 from m
        "midspan_disp_mm":    u3_N2 * 1000.0,
        # Moments and forces already in kN·m / kN — match Midas/OpenSees report units.
        # M3 sign flip: ETABS local-3 = −Y for horizontal +X beam → negate to match.
        "midspan_moment_kNm": -m3_mid,
        "reaction_N1_Fy_kN":  f3_N1,            # Fz (global) ≡ Midas Fy
        "reaction_N3_Fy_kN":  f3_N3,
        "reaction_N1_Fx_kN":  f1_N1,
    }


# ─────────────────────────────────────────────────────────────────────────
# Case 2: 2D Portal Frame (1-story, 1-bay)
# ─────────────────────────────────────────────────────────────────────────

def run_case2_etabs(client) -> dict:
    """1-story 1-bay portal frame, fixed base, lateral + gravity loads.

    Layout (X-Z plane, Y=0; coordinates in m):
        N3(0,0,3) ——B1—— N4(6,0,3)
          |                  |
         C1                 C2
          |                  |
        N1(0,0,0)         N2(6,0,0)

    BC:   fully fixed at N1 and N2
    Load: Fx = +25 kN, Fz = −100 kN at N3 and N4 (CASE2)

    Column: H 350x350x12/19  Beam: H 400x200x8/13  (both SS275, from KS D 3502)

    Sign convention for X-Z frame:
      Column (along +Z):  local2=+X, local3=+Y  → M3 in [kN·m], negative at
                          base under +X lateral  ≡  Midas Mz
      Beam   (along +X):  local2=+Z, local3=−Y  → M3 in [kN·m]  ≡  Midas Mz
      JointReact M2 (about global +Y) ≡  Midas Mz reaction
    """
    m = client.model
    _init(m)
    _material(m)
    # Imported from KS D 3502:2021 library — exact tabulated values.
    _ks_section(m, "H400x200", "H 400x200x8/13")    # beam
    _ks_section(m, "H350x350", "H 350x350x12/19")   # column

    # Coordinates in metres.
    n1 = _pt(m, "N1", 0.0, 0.0, 0.0)
    n2 = _pt(m, "N2", 6.0, 0.0, 0.0)
    n3 = _pt(m, "N3", 0.0, 0.0, 3.0)
    n4 = _pt(m, "N4", 6.0, 0.0, 3.0)

    c1 = _frame(m, "C1", n1, n3, "H350x350")  # column 1
    c2 = _frame(m, "C2", n2, n4, "H350x350")  # column 2
    b1 = _frame(m, "B1", n3, n4, "H400x200")  # beam

    _restrain(m, n1, [True] * 6)
    _restrain(m, n2, [True] * 6)

    _load_pattern(m, "CASE2")
    # Loads in kN: Fx = +25 kN, Fz = −100 kN at each top node.
    _joint_load(m, n3, "CASE2", [25.0, 0.0, -100.0, 0.0, 0.0, 0.0])
    _joint_load(m, n4, "CASE2", [25.0, 0.0, -100.0, 0.0, 0.0, 0.0])

    _run(m)
    _select_case(m, "CASE2")

    u1_N3, _, u3_N3, _, _, _ = _displ(m, n3)        # m
    u1_N4, _, u3_N4, _, _, _ = _displ(m, n4)        # m

    f1_N1, _, f3_N1, _, m2_N1, _ = _react(m, n1)    # kN, kN·m
    f1_N2, _, f3_N2, _, m2_N2, _ = _react(m, n2)    # kN, kN·m

    # Column base moments: M3 at station 0 (i-end = base).  kN·m.
    m3_c1_base = _m3_at(m, c1, 0.0)
    m3_c2_base = _m3_at(m, c2, 0.0)

    # Beam end moments: M3 at i-end (sta=0) and j-end (sta=6.0 m).
    m3_b1_i = _m3_at(m, b1, 0.0)
    m3_b1_j = _m3_at(m, b1, 6.0)

    # Sign convention (X-Z plane, Y-out-of-plane):
    #   Column (+Z local axis): ETABS M3 at i-end (base) = -(Midas Mz)  → negate
    #   Beam (+X local axis):   ETABS M3 at i-end = +(Midas Mz),
    #                           ETABS M3 at j-end = -(Midas Mz)          → negate j
    #   JointReact M2 (about global +Y) = -(Midas Mz)                    → negate
    return {
        # Displacements: ×1000 (m → mm).  Forces / moments already kN, kN·m.
        "top_disp_N3_dx_mm":    u1_N3 * 1000.0,           # U1 (Ux) ≡ Midas dx
        "top_disp_N4_dx_mm":    u1_N4 * 1000.0,
        "top_disp_N3_dy_mm":    u3_N3 * 1000.0,           # U3 (Uz) ≡ Midas dy
        "top_disp_N4_dy_mm":    u3_N4 * 1000.0,
        "col1_base_moment_kNm": -m3_c1_base,              # sign-corrected
        "col2_base_moment_kNm": -m3_c2_base,              # sign-corrected
        "beam_moment_i_kNm":     m3_b1_i,                 # same sign as Midas
        "beam_moment_j_kNm":    -m3_b1_j,                 # sign-corrected
        "base_shear_kN":         f1_N1 + f1_N2,
        "reaction_N1_Fx_kN":    f1_N1,
        "reaction_N1_Fy_kN":    f3_N1,                    # Fz(global) ≡ Midas Fy
        "reaction_N1_Mz_kNm":  -m2_N1,                    # sign-corrected
        "reaction_N2_Fx_kN":    f1_N2,
        "reaction_N2_Fy_kN":    f3_N2,
        "reaction_N2_Mz_kNm":  -m2_N2,                    # sign-corrected
    }


# ─────────────────────────────────────────────────────────────────────────
# Case 3: 2D 3-Story 1-Bay Frame (vertical extension of Case 2)
# ─────────────────────────────────────────────────────────────────────────

def run_case3_etabs(client) -> dict:
    """2D 3-story 1-bay portal frame, fixed base, lateral + gravity loads.

    Layout (X-Z plane, Y=0; coordinates in m):
        N7(0,0,9) ——B3—— N8(6,0,9)       ← roof (story 3 top)
          |                  |
         C3                 C6
          |                  |
        N5(0,0,6) ——B2—— N6(6,0,6)       ← story 2 top
          |                  |
         C2                 C5
          |                  |
        N3(0,0,3) ——B1—— N4(6,0,3)       ← story 1 top
          |                  |
         C1                 C4
          |                  |
        N1(0,0,0)        N2(6,0,0)        ← base (fixed)

    BC:   fully fixed at N1, N2
    Loads (at each top node):
      Story 1 (N3, N4): Fx = +15 kN, Fz = −80 kN
      Story 2 (N5, N6): Fx = +25 kN, Fz = −80 kN
      Story 3 (N7, N8): Fx = +35 kN, Fz = −80 kN

    Column: H 400x400x13/21   Beam: H 400x200x8/13  (both SS275, from KS D 3502)
    """
    m = client.model
    _init(m)
    _material(m)
    _ks_section(m, "H400x400", "H 400x400x13/21")    # columns
    _ks_section(m, "H400x200", "H 400x200x8/13")     # beams

    # 8 nodes (m): 2 columns × 4 levels
    n1 = _pt(m, "N1", 0.0, 0.0, 0.0)
    n2 = _pt(m, "N2", 6.0, 0.0, 0.0)
    n3 = _pt(m, "N3", 0.0, 0.0, 3.0)
    n4 = _pt(m, "N4", 6.0, 0.0, 3.0)
    n5 = _pt(m, "N5", 0.0, 0.0, 6.0)
    n6 = _pt(m, "N6", 6.0, 0.0, 6.0)
    n7 = _pt(m, "N7", 0.0, 0.0, 9.0)
    n8 = _pt(m, "N8", 6.0, 0.0, 9.0)

    # 6 columns (left stack: C1-C2-C3 / right stack: C4-C5-C6)
    c1 = _frame(m, "C1", n1, n3, "H400x400")
    _frame(m, "C2", n3, n5, "H400x400")
    _frame(m, "C3", n5, n7, "H400x400")
    c4 = _frame(m, "C4", n2, n4, "H400x400")
    _frame(m, "C5", n4, n6, "H400x400")
    _frame(m, "C6", n6, n8, "H400x400")

    # 3 beams
    _frame(m, "B1", n3, n4, "H400x200")
    _frame(m, "B2", n5, n6, "H400x200")
    b3 = _frame(m, "B3", n7, n8, "H400x200")   # roof beam (we read moments)

    _restrain(m, n1, [True] * 6)
    _restrain(m, n2, [True] * 6)

    _load_pattern(m, "CASE3")
    # Story 1: Fx = +15 kN | Story 2: +25 kN | Story 3: +35 kN  (all Fz = −80 kN)
    _joint_load(m, n3, "CASE3", [15.0, 0.0, -80.0, 0.0, 0.0, 0.0])
    _joint_load(m, n4, "CASE3", [15.0, 0.0, -80.0, 0.0, 0.0, 0.0])
    _joint_load(m, n5, "CASE3", [25.0, 0.0, -80.0, 0.0, 0.0, 0.0])
    _joint_load(m, n6, "CASE3", [25.0, 0.0, -80.0, 0.0, 0.0, 0.0])
    _joint_load(m, n7, "CASE3", [35.0, 0.0, -80.0, 0.0, 0.0, 0.0])
    _joint_load(m, n8, "CASE3", [35.0, 0.0, -80.0, 0.0, 0.0, 0.0])

    _run(m)
    _select_case(m, "CASE3")

    # Lateral displacements at all top nodes (m)
    u1_N3, _, _, _, _, _ = _displ(m, n3)
    u1_N4, _, _, _, _, _ = _displ(m, n4)
    u1_N5, _, _, _, _, _ = _displ(m, n5)
    u1_N6, _, _, _, _, _ = _displ(m, n6)
    u1_N7, _, _, _, _, _ = _displ(m, n7)
    u1_N8, _, _, _, _, _ = _displ(m, n8)

    # Base reactions at N1, N2 (kN, kN·m)
    f1_N1, _, f3_N1, _, m2_N1, _ = _react(m, n1)
    f1_N2, _, f3_N2, _, m2_N2, _ = _react(m, n2)

    # Column base moments (M3 at i-end = base, kN·m)
    m3_c1_base = _m3_at(m, c1, 0.0)
    m3_c4_base = _m3_at(m, c4, 0.0)

    # Top (roof) beam moments at i (sta=0) and j (sta=6 m)
    m3_b3_i = _m3_at(m, b3, 0.0)
    m3_b3_j = _m3_at(m, b3, 6.0)

    # Story drifts: average lateral disp / story height (h = 3 m)
    avg34 = (u1_N3 + u1_N4) / 2.0
    avg56 = (u1_N5 + u1_N6) / 2.0
    avg78 = (u1_N7 + u1_N8) / 2.0

    return {
        "roof_disp_avg_dx_mm":   avg78 * 1000.0,
        "roof_disp_N7_dx_mm":    u1_N7 * 1000.0,
        "roof_disp_N8_dx_mm":    u1_N8 * 1000.0,
        "story_drift_1":         avg34 / 3.0,
        "story_drift_2":         (avg56 - avg34) / 3.0,
        "story_drift_3":         (avg78 - avg56) / 3.0,
        "base_shear_kN":         f1_N1 + f1_N2,
        "col1_base_moment_kNm": -m3_c1_base,            # sign-corrected (column i-end)
        "col2_base_moment_kNm": -m3_c4_base,
        "top_beam_moment_i_kNm": m3_b3_i,               # beam i-end matches Midas sign
        "top_beam_moment_j_kNm":-m3_b3_j,               # beam j-end sign-corrected
        "reaction_N1_Fx_kN":     f1_N1,
        "reaction_N1_Fy_kN":     f3_N1,
        "reaction_N1_Mz_kNm":   -m2_N1,                 # sign-corrected
        "reaction_N2_Fx_kN":     f1_N2,
        "reaction_N2_Fy_kN":     f3_N2,
        "reaction_N2_Mz_kNm":   -m2_N2,
    }


# ─────────────────────────────────────────────────────────────────────────
# Extraction (same metric names as extract.py / Midas JSON keys)
# ─────────────────────────────────────────────────────────────────────────

def _extract_case1(r: dict) -> list[dict]:
    return [
        {"metric": "Midspan Vert. Disp.", "unit": "mm",   "opensees": r["midspan_disp_mm"]},
        {"metric": "Midspan Moment",      "unit": "kN*m", "opensees": r["midspan_moment_kNm"]},
        {"metric": "Reaction N1 Fy",      "unit": "kN",   "opensees": r["reaction_N1_Fy_kN"]},
        {"metric": "Reaction N3 Fy",      "unit": "kN",   "opensees": r["reaction_N3_Fy_kN"]},
        {"metric": "Reaction N1 Fx",      "unit": "kN",   "opensees": r["reaction_N1_Fx_kN"]},
    ]


def _extract_case2(r: dict) -> list[dict]:
    return [
        {"metric": "Top Disp N3 dx",      "unit": "mm",   "opensees": r["top_disp_N3_dx_mm"]},
        {"metric": "Top Disp N4 dx",      "unit": "mm",   "opensees": r["top_disp_N4_dx_mm"]},
        {"metric": "Top Disp N3 dy",      "unit": "mm",   "opensees": r["top_disp_N3_dy_mm"]},
        {"metric": "Top Disp N4 dy",      "unit": "mm",   "opensees": r["top_disp_N4_dy_mm"]},
        {"metric": "Col1 Base Moment",    "unit": "kN*m", "opensees": r["col1_base_moment_kNm"]},
        {"metric": "Col2 Base Moment",    "unit": "kN*m", "opensees": r["col2_base_moment_kNm"]},
        {"metric": "Beam Moment (i-end)", "unit": "kN*m", "opensees": r["beam_moment_i_kNm"]},
        {"metric": "Beam Moment (j-end)", "unit": "kN*m", "opensees": r["beam_moment_j_kNm"]},
        {"metric": "Base Shear",          "unit": "kN",   "opensees": r["base_shear_kN"]},
        {"metric": "Reaction N1 Fx",      "unit": "kN",   "opensees": r["reaction_N1_Fx_kN"]},
        {"metric": "Reaction N1 Fy",      "unit": "kN",   "opensees": r["reaction_N1_Fy_kN"]},
        {"metric": "Reaction N1 Mz",      "unit": "kN*m", "opensees": r["reaction_N1_Mz_kNm"]},
        {"metric": "Reaction N2 Fx",      "unit": "kN",   "opensees": r["reaction_N2_Fx_kN"]},
        {"metric": "Reaction N2 Fy",      "unit": "kN",   "opensees": r["reaction_N2_Fy_kN"]},
        {"metric": "Reaction N2 Mz",      "unit": "kN*m", "opensees": r["reaction_N2_Mz_kNm"]},
    ]


def _extract_case3(r: dict) -> list[dict]:
    return [
        {"metric": "Roof Disp dx (avg)",   "unit": "mm",   "opensees": r["roof_disp_avg_dx_mm"]},
        {"metric": "Roof Disp N7 dx",      "unit": "mm",   "opensees": r["roof_disp_N7_dx_mm"]},
        {"metric": "Roof Disp N8 dx",      "unit": "mm",   "opensees": r["roof_disp_N8_dx_mm"]},
        {"metric": "Story Drift 1",        "unit": "-",    "opensees": r["story_drift_1"]},
        {"metric": "Story Drift 2",        "unit": "-",    "opensees": r["story_drift_2"]},
        {"metric": "Story Drift 3",        "unit": "-",    "opensees": r["story_drift_3"]},
        {"metric": "Base Shear",           "unit": "kN",   "opensees": r["base_shear_kN"]},
        {"metric": "Col1 Base Moment",     "unit": "kN*m", "opensees": r["col1_base_moment_kNm"]},
        {"metric": "Col2 Base Moment",     "unit": "kN*m", "opensees": r["col2_base_moment_kNm"]},
        {"metric": "Top Beam Moment (i)",  "unit": "kN*m", "opensees": r["top_beam_moment_i_kNm"]},
        {"metric": "Top Beam Moment (j)",  "unit": "kN*m", "opensees": r["top_beam_moment_j_kNm"]},
        {"metric": "Reaction N1 Fx",       "unit": "kN",   "opensees": r["reaction_N1_Fx_kN"]},
        {"metric": "Reaction N1 Fy",       "unit": "kN",   "opensees": r["reaction_N1_Fy_kN"]},
        {"metric": "Reaction N1 Mz",       "unit": "kN*m", "opensees": r["reaction_N1_Mz_kNm"]},
        {"metric": "Reaction N2 Fx",       "unit": "kN",   "opensees": r["reaction_N2_Fx_kN"]},
        {"metric": "Reaction N2 Fy",       "unit": "kN",   "opensees": r["reaction_N2_Fy_kN"]},
        {"metric": "Reaction N2 Mz",       "unit": "kN*m", "opensees": r["reaction_N2_Mz_kNm"]},
    ]


# ─────────────────────────────────────────────────────────────────────────
# Comparison and output
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
    """Percentage difference string, or '—' if either value is missing."""
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
    if d <= 1.0:
        return "OK"
    if d <= 5.0:
        return "CHECK"
    return "FAIL"


def format_3way(case_name: str, extracted: list[dict],
                midas: dict | None, opensees: dict | None) -> str:
    """Three-column comparison: OpenSees | ETABS | Midas."""
    W = 108
    lines = [
        f"\n{'=' * W}",
        f"  {case_name}",
        f"{'=' * W}",
        f"{'Metric':<30} {'Unit':<7} {'OpenSees':>13} {'ETABS':>13} "
        f"{'Midas':>13} {'OS–M%':>8} {'ET–M%':>8} {'ET Status':>10}",
        "-" * W,
    ]

    ok = check = fail = pending = 0
    for row in extracted:
        metric  = row["metric"]
        unit    = row["unit"]
        et_val  = row["opensees"]       # ETABS value stored in "opensees" slot
        m_val   = (midas    or {}).get(metric)
        ops_val = (opensees or {}).get(metric)

        def _fmt(v):
            if v is None:
                return f"{'—':>13}"
            return f"{v:13.6f}"

        st = _status(et_val, m_val)
        if   st == "OK":      ok      += 1
        elif st == "CHECK":   check   += 1
        elif st == "FAIL":    fail    += 1
        else:                 pending += 1

        lines.append(
            f"{metric:<30} {unit:<7} {_fmt(ops_val)} {_fmt(et_val)} "
            f"{_fmt(m_val)} {_pct(ops_val, m_val):>8} {_pct(et_val, m_val):>8} "
            f"{st:>10}"
        )

    total = len(extracted)
    lines += [
        "-" * W,
        f"  Total: {total}  |  OK: {ok}  |  CHECK: {check}  |"
        f"  FAIL: {fail}  |  PENDING: {pending}",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────
# Case registry & main
# ─────────────────────────────────────────────────────────────────────────

_CASES: dict[str, tuple] = {
    "case1": ("Case 1: 2D Simple Beam",         run_case1_etabs, _extract_case1),
    "case2": ("Case 2: 2D Portal Frame",        run_case2_etabs, _extract_case2),
    "case3": ("Case 3: 2D 3-Story Frame",       run_case3_etabs, _extract_case3),
}


def _run_case(case_id: str, client) -> None:
    name, runner, extractor = _CASES[case_id]
    print(f"\n  [{case_id.upper()}] {name} …", end=" ", flush=True)
    try:
        raw = runner(client)
    except Exception as exc:
        print(f"\n  ❌ Failed: {exc}")
        import traceback
        traceback.print_exc()
        return
    print("done")

    extracted = extractor(raw)
    midas     = _load_json(MIDAS_DIR  / f"{case_id}.json")
    opensees  = _load_json(BENCH_DIR  / "opensees_results" / f"{case_id}.json")

    _save_etabs(case_id, extracted)
    print(format_3way(name, extracted, midas, opensees))

    # Sign-check reminder if moments are off by ≈200%
    for row in extracted:
        if "moment" in row["metric"].lower() or "Mz" in row["metric"]:
            m_ref = (midas or {}).get(row["metric"])
            if m_ref is not None and abs(row["opensees"]) > 1e-6:
                ratio = row["opensees"] / m_ref
                if ratio < -0.8:  # signs are opposite (ratio ≈ −1)
                    print(f"  ⚠  Sign flip detected for '{row['metric']}' "
                          f"(ETABS={row['opensees']:.3f}, Midas={m_ref:.3f}). "
                          f"Check local-axis orientation in ETABS GUI.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ETABS 23 benchmark — Case 1, 2, 3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "cases", nargs="*",
        help="Case IDs to run: case1 case2 case3  (default: all)",
    )
    parser.add_argument(
        "--launch", action="store_true",
        help="Launch a new ETABS instance (default: attach to running instance)",
    )
    args = parser.parse_args()

    case_ids = [c for c in args.cases if c in _CASES] or list(_CASES)

    sep = "=" * 80
    print(f"\n{sep}")
    print("  ETABS 23 Benchmark -- Case 1, 2, 3")
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
        print("  Start ETABS (any model) and re-run, or use --launch.")
        sys.exit(1)

    try:
        for cid in case_ids:
            _run_case(cid, client)
    finally:
        if args.launch:
            input("\n  [Enter] to close ETABS …")
            client.close()

    print(f"\n{sep}")
    print("  Done.  Results → tests/benchmark/etabs_results/")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
