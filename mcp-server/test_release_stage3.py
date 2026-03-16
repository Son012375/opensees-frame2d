"""Stage 3 Member Release Verification Tests.

Tests:
  T1: beam_x orientation - gravity UDL - rigid vs both-end hinged
  T2: beam_y orientation - gravity UDL - rigid vs both-end hinged
  T3: 3D lateral EQX - rigid vs beam_x+beam_y hinged, member forces
  T4: 3D lateral EQY - rigid vs beam_x+beam_y hinged, member forces
  T5: Column release warning - ensure analysis works but flags experimental
  T6: Release axis mapping validation summary

IMPORTANT NOTE on force component labels:
  eleForce() returns forces in GLOBAL coordinates, but _extract_member_forces_3d
  labels them as if they were in element LOCAL coordinates. This causes:
  - beam_x (spans X): labels mostly correct (local ~= global)
    -> gravity bending = My (forces[4] = My_global)
  - beam_y (spans Y): gravity bending appears in "T" field, not "My"
    -> gravity bending = "T" (forces[3] = Mx_global)
  - column (spans Z): bending appears in wrong fields
    -> EQX bending = "My" (forces[4] = My_global)  <- correct by coincidence
    -> EQY bending = "T" (forces[3] = Mx_global)  <- mislabeled

  This is a pre-existing force extraction issue, NOT a release bug.
  The equalDOF release correctly frees the right global DOFs:
    beam_x: frees DOFs 5(RY), 6(RZ) -> forces[4], forces[5] -> "My", "Mz"
    beam_y: frees DOFs 4(RX), 6(RZ) -> forces[3], forces[5] -> "T", "Mz"
    column: frees DOFs 4(RX), 5(RY) -> forces[3], forces[4] -> "T", "My"
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from core.frame_3d import analyze_frame_3d_multi

PASS = 0
FAIL = 0


def check(cond, msg):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"    [OK] {msg}")
    else:
        FAIL += 1
        print(f"    [FAIL] {msg}")
    return cond


def find_member_forces(mf_list, member_type):
    """Return all member force dicts matching given type."""
    return [m for m in mf_list if m["type"] == member_type]


# =====================================================================
# T1: beam_x + gravity - rigid vs hinged
# =====================================================================
def test_beam_x_gravity():
    """Single-story 1x1 bay, beam_x only hinged, gravity load.

    beam_x local axes: local x = +X, local y = -Y, local z = +Z
    Gravity bending (XZ plane) -> moment about Y -> forces[4] -> "My" field
    Torsion about X -> forces[3] -> "T" field
    Release frees DOFs 5(RY) and 6(RZ) -> zeroes My and Mz at hinged ends
    """
    print("=" * 65)
    print("T1: beam_x orientation - gravity (floor_area 10 kN/m2)")
    print("=" * 65)

    lc = {"DL": [{"type": "floor_area", "story": 1, "value": 10.0}]}

    r_rigid = analyze_frame_3d_multi(
        stories=[3.5], bays_x=[6.0], bays_y=[6.0],
        load_cases=lc, supports="fixed",
    )
    r_hinge = analyze_frame_3d_multi(
        stories=[3.5], bays_x=[6.0], bays_y=[6.0],
        load_cases=lc, supports="fixed",
        member_releases={"beam_x": "both"},
    )

    mf_rigid = find_member_forces(r_rigid.member_forces["DL"], "beam_x")
    mf_hinge = find_member_forces(r_hinge.member_forces["DL"], "beam_x")
    bx_r = mf_rigid[0]
    bx_h = mf_hinge[0]

    print(f"\n  --- beam_x gravity bending = My field (= My_global) ---")
    print(f"    Rigid  My: {bx_r['My_kNm']}")
    print(f"    Hinged My: {bx_h['My_kNm']}")

    # beam_x: gravity bending is in My field (forces[4] = My_global)
    my_i = bx_h["My_kNm"][0]
    my_j = bx_h["My_kNm"][-1]
    check(abs(my_i) < 0.5, f"beam_x hinged My(i) ~= 0 (got {my_i:.4f})")
    check(abs(my_j) < 0.5, f"beam_x hinged My(j) ~= 0 (got {my_j:.4f})")

    mid = len(bx_h["My_kNm"]) // 2
    my_mid = abs(bx_h["My_kNm"][mid])
    check(my_mid > 1.0, f"beam_x hinged My(mid) significant ({my_mid:.2f} kN*m)")

    # Rigid should have nonzero end moments
    my_r_i = abs(bx_r["My_kNm"][0])
    check(my_r_i > 1.0, f"beam_x rigid My(i) nonzero ({my_r_i:.2f})")

    # Mz (lateral bending) should be ~0 for both
    mz_i = bx_h["Mz_kNm"][0]
    check(abs(mz_i) < 0.5, f"beam_x hinged Mz(i) ~= 0 (got {mz_i:.4f})")

    print("\n  T1 DONE\n")


# =====================================================================
# T2: beam_y + gravity - rigid vs hinged
# =====================================================================
def test_beam_y_gravity():
    """Single-story 1x1 bay, beam_y only hinged, gravity load.

    beam_y local axes: local x = +Y, local y = -X, local z = +Z
    Gravity bending (YZ plane) -> moment about X -> forces[3] -> "T" field(!)
    Torsion about Y -> forces[4] -> "My" field(!)
    Release frees DOFs 4(RX) and 6(RZ) -> zeroes "T" and "Mz" at hinged ends
    """
    print("=" * 65)
    print("T2: beam_y orientation - gravity (floor_area 10 kN/m2)")
    print("=" * 65)

    lc = {"DL": [{"type": "floor_area", "story": 1, "value": 10.0}]}

    r_rigid = analyze_frame_3d_multi(
        stories=[3.5], bays_x=[6.0], bays_y=[6.0],
        load_cases=lc, supports="fixed",
    )
    r_hinge = analyze_frame_3d_multi(
        stories=[3.5], bays_x=[6.0], bays_y=[6.0],
        load_cases=lc, supports="fixed",
        member_releases={"beam_y": "both"},
    )

    mf_rigid = find_member_forces(r_rigid.member_forces["DL"], "beam_y")
    mf_hinge = find_member_forces(r_hinge.member_forces["DL"], "beam_y")
    by_r = mf_rigid[0]
    by_h = mf_hinge[0]

    # beam_y gravity bending is in "T" field (forces[3] = Mx_global)
    print(f"\n  --- beam_y gravity bending = T field (= Mx_global) ---")
    print(f"    Rigid  T: {by_r['T_kNm']}")
    print(f"    Hinged T: {by_h['T_kNm']}")

    t_i = by_h["T_kNm"][0]
    t_j = by_h["T_kNm"][-1]
    print(f"    Hinged T(i) = {t_i:.4f} (expect ~0)")
    print(f"    Hinged T(j) = {t_j:.4f} (expect ~0)")
    check(abs(t_i) < 0.5, f"beam_y hinged T(i) ~= 0 (got {t_i:.4f})")
    check(abs(t_j) < 0.5, f"beam_y hinged T(j) ~= 0 (got {t_j:.4f})")

    mid = len(by_h["T_kNm"]) // 2
    t_mid = abs(by_h["T_kNm"][mid])
    check(t_mid > 1.0, f"beam_y hinged T(mid) significant ({t_mid:.2f} kN*m)")

    # Rigid should have nonzero end "T" (actually gravity bending moment)
    t_r_i = abs(by_r["T_kNm"][0])
    check(t_r_i > 1.0, f"beam_y rigid T(i) nonzero ({t_r_i:.2f})")

    # Mz (forces[5] = Mz_global, DOF 6 freed) should be ~0 at hinged ends
    mz_i = by_h["Mz_kNm"][0]
    check(abs(mz_i) < 0.5, f"beam_y hinged Mz(i) ~= 0 (got {mz_i:.4f})")

    print("\n  T2 DONE\n")


# =====================================================================
# T3: Lateral EQX - rigid vs all beams hinged
# =====================================================================
def test_lateral_eqx():
    """2-story 1x1 bay, EQX lateral, rigid vs all beams hinged."""
    print("=" * 65)
    print("T3: Lateral EQX - rigid vs beam_x+beam_y hinged")
    print("=" * 65)

    lc = {
        "EQX": [
            {"type": "lateral_x", "story": 1, "value": 50.0},
            {"type": "lateral_x", "story": 2, "value": 100.0},
        ],
    }

    r_rigid = analyze_frame_3d_multi(
        stories=[3.5, 3.5], bays_x=[6.0], bays_y=[6.0],
        load_cases=lc, supports="fixed",
    )
    r_hinge = analyze_frame_3d_multi(
        stories=[3.5, 3.5], bays_x=[6.0], bays_y=[6.0],
        load_cases=lc, supports="fixed",
        member_releases={"beam_x": "both", "beam_y": "both"},
    )

    cr_r = r_rigid.case_results["EQX"]
    cr_h = r_hinge.case_results["EQX"]

    print(f"\n  --- Displacement ---")
    print(f"    Rigid  X: {cr_r.max_displacement_x:.4f} mm")
    print(f"    Hinged X: {cr_h.max_displacement_x:.4f} mm")
    ratio_x = cr_h.max_displacement_x / cr_r.max_displacement_x
    check(ratio_x > 2, f"Hinged X-disp >> Rigid ({ratio_x:.1f}x)")

    print(f"\n  --- Story Drift X ---")
    print(f"    Rigid:  {cr_r.max_drift_x:.6f}")
    print(f"    Hinged: {cr_h.max_drift_x:.6f}")
    check(cr_h.max_drift_x > cr_r.max_drift_x * 2,
          f"Hinged drift_x >> Rigid ({cr_h.max_drift_x/cr_r.max_drift_x:.1f}x)")

    # beam_x end moments (EQX bending: My field = My_global)
    bx_r = find_member_forces(r_rigid.member_forces["EQX"], "beam_x")
    bx_h = find_member_forces(r_hinge.member_forces["EQX"], "beam_x")
    if bx_r and bx_h:
        print(f"\n  --- beam_x forces (EQX) ---")
        print(f"    Rigid  My: i={bx_r[0]['My_kNm'][0]:.2f}, j={bx_r[0]['My_kNm'][-1]:.2f}")
        print(f"    Hinged My: i={bx_h[0]['My_kNm'][0]:.4f}, j={bx_h[0]['My_kNm'][-1]:.4f}")
        check(abs(bx_h[0]["My_kNm"][0]) < 0.5, f"beam_x hinged My(i) ~= 0 under EQX")
        check(abs(bx_h[0]["My_kNm"][-1]) < 0.5, f"beam_x hinged My(j) ~= 0 under EQX")

    # Column forces: EQX bending = My field (My_global, about Y) - correct by coincidence
    col_r = find_member_forces(r_rigid.member_forces["EQX"], "column")
    col_h = find_member_forces(r_hinge.member_forces["EQX"], "column")
    if col_r and col_h:
        max_my_r = max(abs(v) for c in col_r for v in c["My_kNm"])
        max_my_h = max(abs(v) for c in col_h for v in c["My_kNm"])
        print(f"\n  --- Column max bending (EQX -> My field) ---")
        print(f"    Rigid:  {max_my_r:.2f} kN*m")
        print(f"    Hinged: {max_my_h:.2f} kN*m")
        check(max_my_h > max_my_r,
              f"Column bending increases ({max_my_h:.1f} > {max_my_r:.1f})")

    print("\n  T3 DONE\n")


# =====================================================================
# T4: Lateral EQY - rigid vs all beams hinged
# =====================================================================
def test_lateral_eqy():
    """2-story 1x1 bay, EQY lateral, rigid vs all beams hinged.

    Column EQY bending: moment about X -> forces[3] -> "T" field
    beam_y EQY bending: moment about X -> forces[3] -> "T" field
    beam_x under EQY: mostly axial (spanning perpendicular to load)
    """
    print("=" * 65)
    print("T4: Lateral EQY - rigid vs beam_x+beam_y hinged")
    print("=" * 65)

    lc = {
        "EQY": [
            {"type": "lateral_y", "story": 1, "value": 50.0},
            {"type": "lateral_y", "story": 2, "value": 100.0},
        ],
    }

    r_rigid = analyze_frame_3d_multi(
        stories=[3.5, 3.5], bays_x=[6.0], bays_y=[6.0],
        load_cases=lc, supports="fixed",
    )
    r_hinge = analyze_frame_3d_multi(
        stories=[3.5, 3.5], bays_x=[6.0], bays_y=[6.0],
        load_cases=lc, supports="fixed",
        member_releases={"beam_x": "both", "beam_y": "both"},
    )

    cr_r = r_rigid.case_results["EQY"]
    cr_h = r_hinge.case_results["EQY"]

    print(f"\n  --- Displacement ---")
    print(f"    Rigid  Y: {cr_r.max_displacement_y:.4f} mm")
    print(f"    Hinged Y: {cr_h.max_displacement_y:.4f} mm")
    ratio_y = cr_h.max_displacement_y / cr_r.max_displacement_y
    check(ratio_y > 2, f"Hinged Y-disp >> Rigid ({ratio_y:.1f}x)")

    print(f"\n  --- Story Drift Y ---")
    print(f"    Rigid:  {cr_r.max_drift_y:.6f}")
    print(f"    Hinged: {cr_h.max_drift_y:.6f}")
    check(cr_h.max_drift_y > cr_r.max_drift_y * 2,
          f"Hinged drift_y >> Rigid ({cr_h.max_drift_y/cr_r.max_drift_y:.1f}x)")

    # beam_y end forces: EQY frame action puts bending in "T" field (Mx_global)
    # Also check that hinged beam_y has T ~= 0 at ends
    by_r = find_member_forces(r_rigid.member_forces["EQY"], "beam_y")
    by_h = find_member_forces(r_hinge.member_forces["EQY"], "beam_y")
    if by_r and by_h:
        # Under EQY, beam_y acts in portal frame action (lateral sway)
        # The "frame action" bending moment may appear in T for beam_y
        # beam_x is the primary lateral frame resisting member for EQY (perpendicular beams)
        # beam_y primarily carries axial from EQY
        print(f"\n  --- beam_y forces (EQY) ---")
        print(f"    Rigid  T:  i={by_r[0]['T_kNm'][0]:.4f}, j={by_r[0]['T_kNm'][-1]:.4f}")
        print(f"    Hinged T:  i={by_h[0]['T_kNm'][0]:.4f}, j={by_h[0]['T_kNm'][-1]:.4f}")
        check(abs(by_h[0]["T_kNm"][0]) < 0.5, f"beam_y hinged T(i) ~= 0 under EQY")
        check(abs(by_h[0]["T_kNm"][-1]) < 0.5, f"beam_y hinged T(j) ~= 0 under EQY")

    # Column bending under EQY: appears in "T" field (forces[3] = Mx_global)
    col_r = find_member_forces(r_rigid.member_forces["EQY"], "column")
    col_h = find_member_forces(r_hinge.member_forces["EQY"], "column")
    if col_r and col_h:
        # For columns, EQY bending is about global X -> forces[3] -> "T" field
        max_t_r = max(abs(v) for c in col_r for v in c["T_kNm"])
        max_t_h = max(abs(v) for c in col_h for v in c["T_kNm"])
        print(f"\n  --- Column max bending (EQY -> T field = Mx_global) ---")
        print(f"    Rigid:  {max_t_r:.2f} kN*m")
        print(f"    Hinged: {max_t_h:.2f} kN*m")
        check(max_t_h > max_t_r,
              f"Column bending increases ({max_t_h:.1f} > {max_t_r:.1f})")

    print("\n  T4 DONE\n")


# =====================================================================
# T5: Column release - experimental, should work but with warning
# =====================================================================
def test_column_release_experimental():
    """Column release should still run but is experimental."""
    print("=" * 65)
    print("T5: Column release (experimental) - 1-story fixed base")
    print("=" * 65)

    lc = {"DL": [{"type": "floor_area", "story": 1, "value": 5.0}]}

    try:
        r = analyze_frame_3d_multi(
            stories=[3.5], bays_x=[6.0], bays_y=[6.0],
            load_cases=lc, supports="fixed",
            member_releases={"column": "i"},
        )
        cr = r.case_results["DL"]
        print(f"    Analysis OK, max disp Z: {cr.max_displacement_z:.4f} mm")
        check(True, "Column release analysis completes")

        # Column base bending should be ~0 (i-end hinged)
        # For columns: bending moments appear in "My" (about Y, from X-forces)
        # and "T" (about X, from Y-forces). Under gravity, both should be ~0.
        mf_col = find_member_forces(r.member_forces["DL"], "column")
        if mf_col:
            c = mf_col[0]
            my_base = c["My_kNm"][0]
            t_base = c["T_kNm"][0]
            print(f"    Col base My = {my_base:.4f}, T = {t_base:.4f} (expect ~0)")
            check(abs(my_base) < 0.5, f"Col base My ~= 0 ({my_base:.4f})")
            check(abs(t_base) < 0.5, f"Col base T ~= 0 ({t_base:.4f})")
    except Exception as e:
        print(f"    Analysis failed: {e}")
        check(False, f"Column release should not crash: {e}")

    print("\n  T5 DONE\n")


# =====================================================================
# T6: Axis mapping summary (documentation/verification)
# =====================================================================
def test_axis_mapping_summary():
    """Print and validate release axis mapping."""
    print("=" * 65)
    print("T6: Release Axis Mapping - Documentation & Validation")
    print("=" * 65)

    print("""
  3D Release Implementation: equalDOF node splitting
  ====================================================
  Reason: OpenSees 0.1.26 silently ignores -releasey/-releasez flags

  geomTransf definitions:
    column:  geomTransf('Linear', 1, 1.0, 0.0, 0.0)  vecxz = +X
    beam_x:  geomTransf('Linear', 2, 0.0, 0.0, 1.0)  vecxz = +Z
    beam_y:  geomTransf('Linear', 3, 0.0, 0.0, 1.0)  vecxz = +Z

  Local axis derivation (e2 = vecxz x e1, e3 = e1 x e2):
  ---------------------------------------------------------------
  Member  | e1(local x) | e2(local y)  | e3(local z)  | Span
  --------|-------------|--------------|--------------|------
  beam_x  | +X (1,0,0)  | -Y (0,-1,0)  | +Z (0,0,1)   | X
  beam_y  | +Y (0,1,0)  | -X (-1,0,0)  | +Z (0,0,1)   | Y
  column  | +Z (0,0,1)  | -Y (0,-1,0)  | +X (1,0,0)   | Z

  Torsion/bending DOF mapping (GLOBAL DOF numbers):
  ---------------------------------------------------------------
  Member  | Torsion axis | Torsion DOF | Bending DOFs (freed)
  --------|-------------|-------------|---------------------
  beam_x  | +X (RX)     | 4           | 5 (RY), 6 (RZ)
  beam_y  | +Y (RY)     | 5           | 4 (RX), 6 (RZ)
  column  | +Z (RZ)     | 6           | 4 (RX), 5 (RY)

  equalDOF pattern per member type:
    beam_x:  equalDOF(master, hinge, 1, 2, 3, 4)  -> frees 5, 6
    beam_y:  equalDOF(master, hinge, 1, 2, 3, 5)  -> frees 4, 6
    column:  equalDOF(master, hinge, 1, 2, 3, 6)  -> frees 4, 5

  Force extraction note (eleForce returns GLOBAL coords):
  ---------------------------------------------------------------
  eleForce index | Global meaning | beam_x label | beam_y label | column label
  -------------- |---------------|-------------|-------------|------------
  [0]            | Fx (global X) | N (OK)      | N (wrong)   | N (wrong)
  [1]            | Fy (global Y) | Vy (flip)   | Vy (wrong)  | Vy (flip)
  [2]            | Fz (global Z) | Vz (OK)     | Vz (OK)     | Vz (wrong=N)
  [3]            | Mx (about X)  | T  (OK)     | T (BENDING!)| T (BENDING!)
  [4]            | My (about Y)  | My (flip)   | My (TORSION)| My (OK for EQX)
  [5]            | Mz (about Z)  | Mz (OK)     | Mz (OK)     | Mz (TORSION)

  Key: "OK" = correct physical meaning, "wrong"/"flip" = label mismatch
  For RELEASE VERIFICATION, the freed DOFs map directly to force indices:
    beam_x freed 5,6 -> forces[4](My), forces[5](Mz) -> both ~0 at hinge
    beam_y freed 4,6 -> forces[3](T),  forces[5](Mz) -> both ~0 at hinge
    column freed 4,5 -> forces[3](T),  forces[4](My) -> both ~0 at hinge
""")

    from core.frame_3d import _TORSION_DOF
    check(_TORSION_DOF["beam_x"] == 4, "beam_x torsion DOF = 4 (RX)")
    check(_TORSION_DOF["beam_y"] == 5, "beam_y torsion DOF = 5 (RY)")
    check(_TORSION_DOF["column"] == 6, "column torsion DOF = 6 (RZ)")

    print("\n  T6 DONE\n")


# =====================================================================
# Run all
# =====================================================================
if __name__ == "__main__":
    test_beam_x_gravity()
    test_beam_y_gravity()
    test_lateral_eqx()
    test_lateral_eqy()
    test_column_release_experimental()
    test_axis_mapping_summary()

    print("=" * 65)
    total = PASS + FAIL
    print(f"RESULTS: {PASS}/{total} passed, {FAIL} failed")
    if FAIL > 0:
        print("SOME TESTS FAILED")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")
    print("=" * 65)
