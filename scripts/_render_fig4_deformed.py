"""Fig 4 (manuscript): 3D deformed shape of the IFC-derived example building
under the governing seismic load combination.

Rebuilds the frozen §4 example (example_input.json) through the real pipeline
(BuildingModel -> generate_all_loads -> analyze_frame_3d_multi), extracts the
per-node displacement field for the governing combination via
visualization_3d.prepare_3d_viz_data, and renders an undeformed (gray) +
amplified-deformed (blue) wireframe consistent with the other paper figures.

Outputs:
  docs/paper1_open_source_alternative/figures/fig4_ifc_deformed.{png,pdf}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mcp-server"))

from core.building_model import BuildingModel  # noqa: E402
from core.load_generator import generate_all_loads  # noqa: E402
from core.frame_3d import analyze_frame_3d_multi  # noqa: E402
from core.visualization_3d import prepare_3d_viz_data  # noqa: E402

INPUT = ROOT / "docs" / "paper1_open_source_alternative" / "validation" / "example_input.json"
OUT_DIR = ROOT / "docs" / "paper1_open_source_alternative" / "figures"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "savefig.dpi": 300,
    "axes.unicode_minus": False,
})


def _node_disp_mag(viz, combo, nid):
    d = viz["case_data"][combo]["disp"].get(str(nid))
    if not d:
        return 0.0
    return (d[0] ** 2 + d[1] ** 2 + d[2] ** 2) ** 0.5  # mm


def pick_governing_combo(viz):
    """Combo with the largest single-node displacement magnitude (mm)."""
    best, best_mag = None, -1.0
    for combo in viz["combo_names"]:
        disp = viz["case_data"][combo]["disp"]
        mag = max((d[0] ** 2 + d[1] ** 2 + d[2] ** 2) ** 0.5 for d in disp.values()) if disp else 0.0
        if mag > best_mag:
            best, best_mag = combo, mag
    return best, best_mag


def main() -> int:
    config = json.loads(INPUT.read_text(encoding="utf-8"))
    config.pop("_meta", None)
    model = BuildingModel.from_json(config)
    print(f"[*] model: {model.num_stories} stories, "
          f"{len(model.bays_x)}x{len(model.bays_y)} bays, H={model.total_height} m")

    load_result = generate_all_loads(model)
    kwargs = model.to_frame3d_kwargs()
    kwargs["load_cases"] = load_result["load_cases"]
    kwargs["load_combinations"] = load_result["load_combinations"]
    kwargs["modal_analysis"] = False  # not needed for the deformed-shape figure

    multi = analyze_frame_3d_multi(**kwargs)
    print(f"[*] analysis: {multi.num_elements} elements, "
          f"{len(multi.combo_results)} combos")

    viz = prepare_3d_viz_data(multi)
    node_map = {str(k): v for k, v in viz["node_map"].items()}
    combo, max_mag_mm = pick_governing_combo(viz)
    disp = viz["case_data"][combo]["disp"]
    print(f"[*] governing combo: {combo}  (max nodal disp {max_mag_mm:.2f} mm)")

    # amplification: scale max displacement to ~12% of the largest plan/height span
    span = max(viz["model_info"]["total_width_x"],
               viz["model_info"]["total_width_y"],
               viz["model_info"]["total_height"])  # m
    max_mag_m = max_mag_mm / 1000.0
    scale = (0.12 * span) / max_mag_m if max_mag_m > 0 else 1.0

    def deformed(nid):
        x, y, z = node_map[str(nid)]
        d = disp.get(str(nid), [0.0, 0.0, 0.0])
        return (x + d[0] / 1000.0 * scale,
                y + d[1] / 1000.0 * scale,
                z + d[2] / 1000.0 * scale)

    undef_segs, def_segs = [], []
    for mlist in viz["members"].values():
        for ni, nj in mlist:
            pi, pj = node_map[str(ni)], node_map[str(nj)]
            undef_segs.append([tuple(pi), tuple(pj)])
            def_segs.append([deformed(ni), deformed(nj)])

    fig = plt.figure(figsize=(8.0, 6.6))
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(Line3DCollection(undef_segs, colors="#BDBDBD",
                                         linewidths=1.0, alpha=0.8))
    ax.add_collection3d(Line3DCollection(def_segs, colors="#1565C0",
                                         linewidths=1.6, alpha=0.95))

    # supports
    sx = [node_map[str(i)][0] for i in viz["support_ids"]]
    sy = [node_map[str(i)][1] for i in viz["support_ids"]]
    sz = [node_map[str(i)][2] for i in viz["support_ids"]]
    ax.scatter(sx, sy, sz, c="#333", marker="^", s=28, depthshade=False)

    # axis ranges from undeformed geometry
    xs = [p[0] for seg in undef_segs for p in seg]
    ys = [p[1] for seg in undef_segs for p in seg]
    zs = [p[2] for seg in undef_segs for p in seg]
    ax.set_xlim(min(xs), max(xs))
    ax.set_ylim(min(ys), max(ys))
    ax.set_zlim(0, max(zs))
    ax.set_box_aspect((max(xs) - min(xs), max(ys) - min(ys), max(zs)))
    ax.view_init(elev=20, azim=-60)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    from matplotlib.lines import Line2D
    legend = [
        Line2D([0], [0], color="#BDBDBD", lw=2, label="Undeformed"),
        Line2D([0], [0], color="#1565C0", lw=2,
               label=f"Deformed (×{scale:.0f})"),
    ]
    ax.legend(handles=legend, loc="upper left", fontsize=9, framealpha=0.9)
    ax.set_title("3D deformed shape of the IFC-derived example building\n"
                 f"governing combination {combo}  "
                 f"(max nodal displacement {max_mag_mm:.1f} mm, "
                 f"deformation scaled ×{scale:.0f})", fontsize=10, pad=8)

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"fig4_ifc_deformed.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] -> {OUT_DIR / 'fig4_ifc_deformed.png'} (+ .pdf)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
