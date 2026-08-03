"""Reproduce the paper's §4/§5 IFC example through the ACTUAL V2 product path.

Where `_rerun_section4_analysis.py` reproduces the §4.2 numbers through the V1
grid path (`BuildingModel.from_json` -> regular-grid generator), this script
reproduces them through the REAL node-element product pipeline that the webapp
exercises (`webapp/backend/app/main_simple.py`, /api/v2/parse-ifc ->
/api/v2/snap-joints -> /api/v2/analyze):

    parse_ifc_v2(ifc)            # extract free node-element StructuralModel
      -> IFC_DISCONNECTED_JOINTS # validation flags beam ends off the column grid
    snap_beams_to_columns(model) # connectivity repair -> 48-node column grid
    model.to_building_model()    # duck-typed BuildingModel for load generation
    generate_all_loads(bm)       # KDS auto loads (DL/LL/EQ/Wind + combinations)
    analyze_from_model(model,..) # V2 node-element 3D analysis (348 sub-elements)
    run_design_check(...)        # KDS drift + AISC member strength

Input IFC is the synthetic, IP-free artifact authored by
`_generate_synthetic_ifc_example.py`. The supplementary engineering config that
the IFC geometry cannot carry (per-story usage/finish, region, site class,
importance, seismic system, exposure) is taken from `example_input.json`
(minus `_meta`), exactly as a webapp user would supply it after upload.

The §5 governing results reproduced here (roof displacement X/Y, max member
interaction ratio) match the manuscript reference `example_section4_results.json`
to within a few tenths of a percent — the residual comes from the V2
node-element load application differing infinitesimally from the V1 grid path,
not from any modeling difference (post-snap topology is identical).

Usage:
    python scripts/_reproduce_ifc_example_v2.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mcp-server"))

VAL = ROOT / "docs" / "paper1_open_source_alternative" / "validation"
IFC = VAL / "ifc_example.ifc"
INPUT = VAL / "example_input.json"
REF = VAL / "example_section4_results.json"

# Tolerances for the §5 reproduction check (V2 vs manuscript V1 reference).
DISP_TOL_PCT = 1.0     # roof displacement
RATIO_TOL_PCT = 1.0    # max interaction ratio


def _apply_supplementary_config(model, cfg: dict) -> None:
    """Overlay the engineering config the IFC can't carry onto the StructuralModel.

    Mirrors webapp/backend/app/main_simple.py::analyze_v2_api (lines ~980-1001).
    example_input.json stores `stories` as a POSITIONAL list without a "story"
    key, so map entries onto story indices 1..N by position.
    """
    for key in ("region", "site_class", "importance", "seismic_system",
                "exposure_category", "geometric_nonlinearity"):
        if cfg.get(key):
            setattr(model, key, cfg[key])
    if cfg.get("importance"):
        model.importance_factor = {"특": 1.5, "I": 1.2, "II": 1.0}.get(model.importance, 1.0)
    if cfg.get("rigid_diaphragm") is not None:
        model.rigid_diaphragm = cfg["rigid_diaphragm"]

    for i, sc in enumerate(cfg.get("stories", []), start=1):
        if not isinstance(sc, dict):
            continue
        if sc.get("usage"):
            model.story_usages[i] = sc["usage"]
        if sc.get("slab_thickness"):
            model.story_slab_thickness[i] = sc["slab_thickness"]
        if sc.get("dead_load_finish") is not None:
            model.story_dead_load_finish[i] = sc["dead_load_finish"]

    for i in range(1, model.num_stories + 1):
        model.story_usages.setdefault(i, "office")
        model.story_slab_thickness.setdefault(i, 0.15)
        model.story_dead_load_finish.setdefault(i, 1.0)


def main() -> int:
    from core.ifc_parser_v2 import parse_ifc_v2, snap_beams_to_columns
    from core.load_generator import generate_all_loads
    from core.frame_3d import analyze_from_model
    from core.design_check import run_design_check

    print("=" * 72)
    print("V2 node-element reproduction of the §4/§5 IFC example")
    print(f"  IFC: {IFC.relative_to(ROOT)}")
    print("=" * 72)

    # ── Step 1: parse_ifc_v2 → free node-element StructuralModel ──
    model, validation = parse_ifc_v2(str(IFC))
    codes = [i.code for i in validation.issues]
    disc = "IFC_DISCONNECTED_JOINTS" in codes
    from collections import Counter
    types_pre = dict(Counter(e.elem_type.value for e in model.elements.values()))
    print("\n[1] parse_ifc_v2:")
    print(f"    nodes          = {len(model.nodes)}")
    print(f"    elements       = {len(model.elements)}  (types: {types_pre})")
    print(f"    is_valid       = {validation.is_valid}")
    print(f"    issue codes    = {codes}")
    print(f"    IFC_DISCONNECTED_JOINTS raised = {disc}")
    if not disc:
        print("    [FAIL] expected IFC_DISCONNECTED_JOINTS diagnostic")
        return 1
    if len(model.elements) != 87:
        print(f"    [FAIL] expected 87 elements, got {len(model.elements)}")
        return 1

    # ── Step 2: snap_beams_to_columns → connectivity repair ──
    snapped = snap_beams_to_columns(model)
    xs = sorted(set(round(n.x, 3) for n in model.nodes.values()))
    ys = sorted(set(round(n.y, 3) for n in model.nodes.values()))
    zs = sorted(set(round(n.z, 3) for n in model.nodes.values()))
    types_post = dict(Counter(e.elem_type.value for e in model.elements.values()))
    print("\n[2] snap_beams_to_columns:")
    print(f"    snapped beam joints = {snapped}")
    print(f"    nodes after         = {len(model.nodes)}  (expect 48)")
    print(f"    elements after      = {len(model.elements)}  (expect 87; types {types_post})")
    print(f"    grid X = {[float(v) for v in xs]}")
    print(f"    grid Y = {[float(v) for v in ys]}")
    print(f"    grid Z = {[float(v) for v in zs]}")
    grid_ok = (xs == [0.0, 4.0, 8.0, 12.0] and ys == [0.0, 5.0, 10.0]
               and zs == [0.0, 3.0, 6.0, 9.0])
    if not (len(model.nodes) == 48 and len(model.elements) == 87 and grid_ok):
        print("    [FAIL] post-snap geometry does not match the 48/87 column grid")
        return 1
    print("    [OK] node-element extraction + connectivity repair -> 48-node / 87-element grid")

    # ── Step 3: analysis + loads + design-check through the V2 product path ──
    cfg = json.loads(INPUT.read_text(encoding="utf-8"))
    cfg.pop("_meta", None)
    _apply_supplementary_config(model, cfg)
    print("\n[3] V2 analysis pipeline (loads -> analyze_from_model -> design_check):")
    print(f"    stories        = {model.num_stories}  heights={model.story_heights}")
    print(f"    usages         = {[model.story_usages[i] for i in range(1, model.num_stories + 1)]}")
    print(f"    finishes       = {[model.story_dead_load_finish[i] for i in range(1, model.num_stories + 1)]}")
    print(f"    region/site/IE = {model.region} / {model.site_class} / {model.importance}")

    building_model = model.to_building_model()
    print(f"    detected bays  = X{[round(float(b), 3) for b in building_model.bays_x]}"
          f" Y{[round(float(b), 3) for b in building_model.bays_y]}")

    load_result = generate_all_loads(building_model)
    load_cases = load_result["load_cases"]
    load_combinations = load_result.get("load_combinations", {})
    seismic_rpt = load_result.get("reports", {}).get("seismic", {})

    multi = analyze_from_model(model, load_cases, load_combinations)
    # V2 analyze_from_model does not populate multi.num_elements; the true analysis
    # sub-element count is the sum of per-member meshed element_ids.
    n_analysis = sum(len(m.get("element_ids", [])) for m in multi.member_info)
    print(f"    physical members = {len(multi.member_info)}  (expect 87)")
    print(f"    analysis elements = {n_analysis}  (expect 348)")
    print(f"    load combinations = {len(multi.combo_results)}")

    max_dx = max(abs(c.max_displacement_x) for c in multi.combo_results.values())
    max_dy = max(abs(c.max_displacement_y) for c in multi.combo_results.values())

    dc = run_design_check(multi, building_model, seismic_rpt)
    max_ratio = dc["member_check"]["summary"]["max_interaction_ratio"]

    # ── Step 4: compare against the manuscript reference ──
    ref = json.loads(REF.read_text(encoding="utf-8"))
    ref_dx = ref["analysis"]["envelope"]["max_dx_mm"]
    ref_dy = ref["analysis"]["envelope"]["max_dy_mm"]
    ref_ratio = ref["design_check"]["member_check"]["summary"]["max_interaction_ratio"]

    def _pct(got, want):
        return abs(got - want) / max(abs(want), 1e-12) * 100.0

    dx_pct, dy_pct, ratio_pct = _pct(max_dx, ref_dx), _pct(max_dy, ref_dy), _pct(max_ratio, ref_ratio)

    print("\n[4] §5 governing results — V2 (this run) vs manuscript reference:")
    print(f"    {'metric':22s} {'V2':>10s} {'ref':>10s} {'Δ%':>8s}  flag")
    rows = [
        ("roof disp X (mm)", max_dx, ref_dx, dx_pct, DISP_TOL_PCT),
        ("roof disp Y (mm)", max_dy, ref_dy, dy_pct, DISP_TOL_PCT),
        ("max interaction",  max_ratio, ref_ratio, ratio_pct, RATIO_TOL_PCT),
    ]
    all_ok = True
    for label, got, want, pct, tol in rows:
        ok = pct <= tol
        all_ok = all_ok and ok
        print(f"    {label:22s} {got:10.4f} {want:10.4f} {pct:7.3f}%  [{'OK' if ok else 'CHECK'}]")

    n_ok = (len(multi.member_info) == 87 and n_analysis == 348)

    print("\n" + "=" * 72)
    if all_ok and n_ok:
        print("RESULT: FULL reproduction — the V2 node-element + connectivity-repair path")
        print("        reproduces the §5 governing numbers (roof disp X/Y ≈ 5.3/11.9 mm,")
        print("        max interaction ≈ 0.467) AND the 48-node / 87-member / 348-element model.")
    elif n_ok:
        print("RESULT: PARTIAL — geometry (48/87/348) reproduced, but the §5 numbers")
        print("        drifted beyond tolerance. See the table above.")
    else:
        print("RESULT: PARTIAL — see failures above.")
    print("=" * 72)
    return 0 if (all_ok and n_ok) else 2


if __name__ == "__main__":
    raise SystemExit(main())
