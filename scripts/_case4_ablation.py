"""Case 4 discrepancy ablation (paper gap G2 / Table 9 / Fig 8).

The 12 CHECK metrics of the benchmark are all in Case 4 (3D 2-story 1x1-bay
frame): OpenSees Story Drift 1 = 0.000666 vs Midas Gen 0.000640 (3.9%), while
OpenSees-vs-ETABS is 0.00% on the identical metrics. So the gap is a
Midas-specific stiffening effect. The benchmark spec already fixes shear
deformation OFF and Beta = 0 in BOTH programs, so those are not candidates.

This script rebuilds Case 4 and toggles the remaining candidate drivers one at a
time, reporting Story Drift 1 (and roof / story-1 X-displacement) for each, so
the 3.9% gap can be attributed quantitatively instead of hand-waved.

Drivers tested:
  baseline       : as benchmarked (elasticBeamColumn, centerline, no offsets)
  rigid_offset   : rigid end zones / panel zone (geomTransf -jntOffset; column
                   gets half-beam-depth offsets, beams get half-column-depth)
  shear          : Timoshenko shear deformation ON (ElasticTimoshenkoBeam3d) —
                   a sensitivity check; expected to INCREASE drift (wrong way)
  colI_match     : scale of the column strong-axis I needed to hit Midas exactly
                   (quantifies the gap as an equivalent section-stiffness delta)

Outputs:
  - console table
  - docs/paper1_open_source_alternative/figures/fig8_case4_ablation.{png,pdf}
  - docs/paper1_open_source_alternative/validation/case4_ablation.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mcp-server"))

from core.ops_compat import ops  # noqa: E402

# ── Case 4 spec (mirrors tests/benchmark/cases.py) ──
E = 210000.0          # MPa (N/mm^2) — SS275 benchmark value
G = 80769.0           # MPa
# Column H-400x400x13x21
COL = dict(A=21870.0, Iy=66600e4, Iz=22400e4, J=498e4, h=400.0, b=400.0, tw=13.0, tf=21.0)
# Beam H-350x175x7x11
BM = dict(A=6314.0, Iy=13600e4, Iz=984e4, J=19.3e4, h=350.0, b=175.0, tw=7.0, tf=11.0)

NODES = {
    1: (0, 0, 0), 2: (6000, 0, 0), 3: (6000, 6000, 0), 4: (0, 6000, 0),
    5: (0, 0, 3000), 6: (6000, 0, 3000), 7: (6000, 6000, 3000), 8: (0, 6000, 3000),
    9: (0, 0, 6000), 10: (6000, 0, 6000), 11: (6000, 6000, 6000), 12: (0, 6000, 6000),
}
COL_PAIRS = [(1, 5), (5, 9), (2, 6), (6, 10), (3, 7), (7, 11), (4, 8), (8, 12)]
# (ni, nj, axis) axis in {'x','y'}
BM_SPEC = [
    (5, 6, "x"), (6, 7, "y"), (8, 7, "x"), (5, 8, "y"),       # 1F
    (9, 10, "x"), (10, 11, "y"), (12, 11, "x"), (9, 12, "y"),  # roof
]
BASE_NODES = {1, 2, 3, 4}

MIDAS = {"story_drift_1": 0.000640, "roof_avg_dx_mm": 5.201, "story1_avg_dx_mm": 1.921}


def _kN(v):
    return v * 1000.0


def build_run(offset_factor=0.0, shear=False, colI_scale=1.0, beamI_scale=1.0):
    offset = offset_factor > 0.0
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)
    for nid, (x, y, z) in NODES.items():
        ops.node(nid, float(x), float(y), float(z))
    for nid in BASE_NODES:
        ops.fix(nid, 1, 1, 1, 1, 1, 1)

    col_Iy = COL["Iy"] * colI_scale
    col_Iz = COL["Iz"] * colI_scale
    bm_Iy = BM["Iy"] * beamI_scale
    bm_Iz = BM["Iz"] * beamI_scale

    # rigid-zone offsets, scaled by offset_factor (1.0 = full half-joint, the
    # geometric panel-zone limit; Midas Gen's rigid-zone factor defaults to ~0.5)
    half_bd = (BM["h"] / 2.0) * offset_factor    # column offset at a beam joint
    half_cd = (COL["b"] / 2.0) * offset_factor   # beam offset at a column joint

    # shear areas (H-section): web for strong-axis shear, flanges for weak-axis
    def shear_areas(s):
        avy = s["h"] * s["tw"]                 # web area (strong-axis shear)
        avz = (5.0 / 6.0) * 2.0 * s["b"] * s["tf"]  # flange area (weak-axis shear)
        return avy, avz

    tag = [0]

    def transf(vecxz, oi=(0, 0, 0), oj=(0, 0, 0)):
        tag[0] += 1
        if offset and (any(oi) or any(oj)):
            ops.geomTransf("Linear", tag[0], *vecxz, "-jntOffset", *oi, *oj)
        else:
            ops.geomTransf("Linear", tag[0], *vecxz)
        return tag[0]

    eid = 0
    col_eids = []
    for ni, nj in COL_PAIRS:  # ni bottom, nj top
        oi = (0, 0, half_bd) if ni not in BASE_NODES else (0, 0, 0)
        oj = (0, 0, -half_bd)
        t = transf((1.0, 0.0, 0.0), oi, oj)
        eid += 1
        if shear:
            avy, avz = shear_areas(COL)
            ops.element("ElasticTimoshenkoBeam", eid, ni, nj, E, G, COL["A"],
                        COL["J"], col_Iy, col_Iz, avy, avz, t)
        else:
            ops.element("elasticBeamColumn", eid, ni, nj, COL["A"], E, G,
                        COL["J"], col_Iy, col_Iz, t)
        col_eids.append(eid)

    for ni, nj, axis in BM_SPEC:
        if axis == "x":
            oi, oj = (half_cd, 0, 0), (-half_cd, 0, 0)
        else:
            oi, oj = (0, half_cd, 0), (0, -half_cd, 0)
        t = transf((0.0, 0.0, 1.0), oi, oj)
        eid += 1
        if shear:
            avy, avz = shear_areas(BM)
            ops.element("ElasticTimoshenkoBeam", eid, ni, nj, E, G, BM["A"],
                        BM["J"], bm_Iy, bm_Iz, avy, avz, t)
        else:
            ops.element("elasticBeamColumn", eid, ni, nj, BM["A"], E, G,
                        BM["J"], bm_Iy, bm_Iz, t)

    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    for nid in [5, 6, 7, 8]:
        ops.load(nid, _kN(10.0), 0.0, _kN(-60.0), 0.0, 0.0, 0.0)
    for nid in [9, 10, 11, 12]:
        ops.load(nid, _kN(20.0), 0.0, _kN(-80.0), 0.0, 0.0, 0.0)

    ops.system("BandGeneral")
    ops.numberer("RCM")
    ops.constraints("Plain")
    ops.integrator("LoadControl", 1.0)
    ops.algorithm("Linear")
    ops.analysis("Static")
    ops.analyze(1)

    s1 = sum(ops.nodeDisp(n)[0] for n in [5, 6, 7, 8]) / 4.0
    rf = sum(ops.nodeDisp(n)[0] for n in [9, 10, 11, 12]) / 4.0
    ops.wipe()
    return {"story1_avg_dx_mm": s1, "roof_avg_dx_mm": rf,
            "story_drift_1": s1 / 3000.0, "story_drift_2": (rf - s1) / 3000.0}


def _bisect(fn, lo, hi, target, n=40):
    """fn is monotonically DECREASING in its argument; find arg with fn==target."""
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        if fn(mid) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def solve_colI_match():
    return _bisect(lambda s: build_run(colI_scale=s)["story_drift_1"],
                   1.0, 1.30, MIDAS["story_drift_1"])


def solve_offset_match():
    return _bisect(lambda f: build_run(offset_factor=f)["story_drift_1"],
                   0.0, 1.0, MIDAS["story_drift_1"])


def make_fig8(base, offset_factor, colI_factor):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.family": "serif", "font.size": 10,
                         "savefig.dpi": 300, "axes.unicode_minus": False})
    factors = np.linspace(0, 1, 21)
    drifts = [build_run(offset_factor=f)["story_drift_1"] for f in factors]
    shear_d = build_run(shear=True)["story_drift_1"]
    midas = MIDAS["story_drift_1"]

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.plot(factors, np.array(drifts) * 1e3, "-o", ms=3, color="#1565C0",
            label="OpenSeesPy (rigid-zone sweep)")
    ax.axhline(midas * 1e3, color="#C62828", ls="--", lw=1.3,
               label=f"Midas Gen reference ({midas*1e3:.4f}×10⁻³)")
    ax.axhline(base["story_drift_1"] * 1e3, color="0.6", ls=":", lw=1.0)
    ax.plot([offset_factor], [midas * 1e3], "s", ms=9, color="#C62828",
            markerfacecolor="none", mew=1.8,
            label=f"match at rigid-zone factor = {offset_factor:.2f}")
    ax.axvline(offset_factor, color="#C62828", ls=":", lw=0.8)

    arrow = dict(arrowstyle="->", color="0.4", lw=0.9)
    ax.annotate(f"centerline (factor 0)\n+{(base['story_drift_1']/midas-1)*100:.1f}% vs Midas\n"
                "(OpenSees ≡ ETABS)",
                xy=(0.0, base["story_drift_1"] * 1e3),
                xytext=(0.13, base["story_drift_1"] * 1e3 - 0.001),
                fontsize=8.5, color="0.3", ha="left", va="top", arrowprops=arrow)
    ax.annotate(f"full panel zone (factor 1)\n{(drifts[-1]/midas-1)*100:+.1f}% vs Midas",
                xy=(1.0, drifts[-1] * 1e3),
                xytext=(0.60, drifts[-1] * 1e3 + 0.004),
                fontsize=8.5, color="0.3", ha="left", va="bottom", arrowprops=arrow)

    ax.set_xlabel("beam–column rigid-zone (end-offset) factor")
    ax.set_ylabel("Story Drift 1  (×10⁻³)")
    ax.set_title("Case 4 discrepancy attribution: beam–column joint rigid-zone modeling\n"
                 "OpenSeesPy ≡ ETABS (centerline, 0.00%); Midas reproduced at factor ≈ 0.5",
                 fontsize=10, pad=8)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=8.5)

    txt = (f"Shear deformation ON → {shear_d*1e3:.4f}×10⁻³ "
           f"({(shear_d/midas-1)*100:+.1f}%, wrong direction; "
           f"spec fixes shear OFF in both)\n"
           f"Equivalent column-stiffness delta to match Midas: "
           f"+{(colI_factor-1)*100:.1f}%")
    ax.text(0.02, 0.03, txt, transform=ax.transAxes, fontsize=8,
            va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.7", alpha=0.92))

    fig.tight_layout()
    fig_dir = ROOT / "docs" / "paper1_open_source_alternative" / "figures"
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"fig8_case4_ablation.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {fig_dir / 'fig8_case4_ablation.png'} (+ .pdf)")


def main():
    base = build_run()
    variants = {
        "baseline (centerline)": base,
        "rigid_offset x0.50": build_run(offset_factor=0.50),
        "rigid_offset x1.00 (full)": build_run(offset_factor=1.00),
        "shear_timoshenko": build_run(shear=True),
    }
    offset_factor = solve_offset_match()
    colI_factor = solve_colI_match()
    variants["rigid_offset matched"] = build_run(offset_factor=offset_factor)
    variants["colI scaled to match"] = build_run(colI_scale=colI_factor)

    rows = []
    print(f"\n{'variant':24} {'drift_1':>11} {'vs base':>9} {'vs Midas':>9} "
          f"{'roof_dx':>9} {'s1_dx':>8}")
    print("-" * 76)
    for name, r in variants.items():
        d1 = r["story_drift_1"]
        vs_base = (d1 - base["story_drift_1"]) / base["story_drift_1"] * 100
        vs_mid = (d1 - MIDAS["story_drift_1"]) / MIDAS["story_drift_1"] * 100
        print(f"{name:24} {d1:11.6f} {vs_base:+8.2f}% {vs_mid:+8.2f}% "
              f"{r['roof_avg_dx_mm']:9.3f} {r['story1_avg_dx_mm']:8.3f}")
        rows.append({"variant": name, **r, "pct_vs_baseline": vs_base,
                     "pct_vs_midas": vs_mid})
    print("-" * 76)
    print(f"{'MIDAS (reference)':24} {MIDAS['story_drift_1']:11.6f} "
          f"{'':>9} {'':>9} {MIDAS['roof_avg_dx_mm']:9.3f} "
          f"{MIDAS['story1_avg_dx_mm']:8.3f}")
    print(f"\nrigid-zone (offset) factor needed to match Midas drift_1: "
          f"{offset_factor:.3f}  (1.0 = full geometric panel zone)")
    print(f"column-I scale needed to match Midas drift_1 exactly: "
          f"{colI_factor:.4f}  (i.e. +{(colI_factor-1)*100:.2f}% column stiffness)")

    out = {
        "midas_reference": MIDAS,
        "baseline_vs_midas_pct": rows[0]["pct_vs_midas"],
        "rigid_zone_factor_to_match": offset_factor,
        "colI_scale_to_match": colI_factor,
        "variants": rows,
        "note": ("Shear deformation and Beta=0 are fixed OFF in both programs per "
                 "the benchmark spec, so the residual is a Midas-side stiffening "
                 "(rigid end zones / panel-zone) effect; the rigid_offset variant "
                 "quantifies it."),
    }
    out_json = ROOT / "docs" / "paper1_open_source_alternative" / "validation" / "case4_ablation.json"
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[OK] {out_json.relative_to(ROOT)}")
    make_fig8(base, offset_factor, colI_factor)
    return variants, base, colI_factor


if __name__ == "__main__":
    main()
