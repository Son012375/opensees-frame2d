"""§4.3 L-shape demonstration data extraction for paper.

Builds the L-shape model (5F + 3F setback) using the same pattern as
tests/test_v2_irregular.py, runs full analysis (DL/LL/EQX/EQY), and extracts
key response metrics for the paper §4.3 demonstration section.

Outputs:
- docs/paper1_open_source_alternative/validation/lshape_demonstration.json
  (numeric results — node/element counts, modal periods if available, base
   reactions, story-wise displacements, drift, member utilization summary)
- docs/paper1_open_source_alternative/figures/fig6_lshape_zones.png
  (top-view: column positions + detected zones color-coded)
- docs/paper1_open_source_alternative/figures/fig7_lshape_deformed.png
  (3D deformed shape under governing seismic combo, schematic)

Usage:
    python scripts/_extract_lshape_for_paper.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mcp-server"))

from core.structural_model import StructuralModel, SupportType, ElementType
from core.frame_3d import analyze_from_model


# ============================================================
# Build L-shape model (5F left + 3F setback right)
# ============================================================

def _find_or_add(m, x, y, z, tol=0.01):
    for n in m.nodes.values():
        if abs(n.x - x) < tol and abs(n.y - y) < tol and abs(n.z - z) < tol:
            return n.id
    return m.add_node(x, y, z).id


def _add_column(m, x, y, z_bot, z_top, section="H-300x300"):
    ni = _find_or_add(m, x, y, z_bot)
    nj = _find_or_add(m, x, y, z_top)
    return m.add_element(ni, nj, elem_type="column", section=section)


def _add_beam(m, x1, y1, x2, y2, z, section="H-400x200"):
    ni = _find_or_add(m, x1, y1, z)
    nj = _find_or_add(m, x2, y2, z)
    return m.add_element(ni, nj, elem_type="beam", section=section)


def build_l_shape_model() -> StructuralModel:
    """L-shape: left wing 12m×8m (5 stories), right wing 12m×4m (3 stories).

    Plan view (in m):
        Y
        8 ┌──┬──┐
          │  │  │  ← left wing only (5 stories)
        4 ├──┼──┼──┬──┐
          │  │  │  │  │  ← both wings (3 stories)
        0 └──┴──┴──┴──┘
          0  6  12 18 24 X
    """
    m = StructuralModel()
    story_h = 3.5

    # Left wing columns (5 stories)
    left_cols = [(0, 0), (6, 0), (12, 0), (0, 4), (6, 4), (12, 4), (0, 8), (6, 8), (12, 8)]
    for x, y in left_cols:
        for s in range(5):
            _add_column(m, x, y, s * story_h, (s + 1) * story_h)

    # Right wing columns (3 stories only)
    right_cols = [(18, 0), (24, 0), (18, 4), (24, 4)]
    for x, y in right_cols:
        for s in range(3):
            _add_column(m, x, y, s * story_h, (s + 1) * story_h)

    # X-direction beams
    for s in range(1, 6):
        z = s * story_h
        for y in [0, 4, 8]:
            _add_beam(m, 0, y, 6, y, z)
            _add_beam(m, 6, y, 12, y, z)
        if s <= 3:
            for y in [0, 4]:
                _add_beam(m, 12, y, 18, y, z)
                _add_beam(m, 18, y, 24, y, z)

    # Y-direction beams
    for s in range(1, 6):
        z = s * story_h
        for x in [0, 6, 12]:
            _add_beam(m, x, 0, x, 4, z)
            _add_beam(m, x, 4, x, 8, z)
        if s <= 3:
            for x in [18, 24]:
                _add_beam(m, x, 0, x, 4, z)

    for n in m.nodes.values():
        if abs(n.z) < 0.01:
            n.support = SupportType.FIXED

    m.detect_stories()
    return m


# ============================================================
# Run analysis with realistic KDS-style loads
# ============================================================

def make_lshape_load_cases(model: StructuralModel) -> dict:
    """DL/LL/EQX/EQY for 5-story L-shape with setback (right wing only stories 1-3).

    Floor area is reduced for stories 4-5 (left wing only), and this is
    naturally handled by the floor_area load distribution.
    """
    # DL: office-style 5.1 kN/m² per story
    # LL: 2.5 kN/m² (office)
    # EQX/EQY: triangular lateral force per KDS pattern, magnitudes calibrated
    # to give plausible mid-rise lateral demand (5-story, ~17.5 m height)
    DL_kpm2 = 5.1
    LL_kpm2 = 2.5
    # Gravity-only demonstration: lateral analysis with diaphragm/member-strength
    # setup parallel to §3 benchmark cases is reserved for the future irregular
    # benchmark campaign (Option A in the revision roadmap).
    cases = {
        "DL": [{"type": "floor_area", "story": s, "value": DL_kpm2}
               for s in range(1, model.num_stories + 1)],
        "LL": [{"type": "floor_area", "story": s, "value": LL_kpm2}
               for s in range(1, model.num_stories + 1)],
    }
    return cases


# ============================================================
# Extract metrics for paper
# ============================================================

def extract_metrics(model: StructuralModel, result) -> dict:
    """Pull together the numbers needed for §4.3 narrative and Table."""
    out = {
        "geometry": {
            "left_wing": {"x_span_m": 12.0, "y_span_m": 8.0, "stories": 5},
            "right_wing": {"x_span_m": 12.0, "y_span_m": 4.0, "stories": 3},
            "story_height_m": 3.5,
            "total_height_m": 5 * 3.5,
        },
        "model_size": {
            "nodes": len(model.nodes),
            "elements": len(model.elements),
            "columns": len(model.elements_by_type(ElementType.COLUMN)),
            "beams": len(model.elements_by_type(ElementType.BEAM)),
            "base_nodes_fixed": sum(1 for n in model.nodes.values()
                                    if abs(n.z) < 0.01),
            "stories": model.num_stories,
        },
        "load_cases": list(result.case_results.keys()),
    }

    # Per-case max displacement + base reactions
    case_summary = {}
    for case_name, cr in result.case_results.items():
        try:
            max_disp_x = abs(cr.max_displacement_x)
        except AttributeError:
            max_disp_x = None
        try:
            max_disp_y = abs(cr.max_displacement_y)
        except AttributeError:
            max_disp_y = None

        # Base reactions sum
        try:
            total_rx = sum(r.get("RX_kN", 0) for r in cr.reactions)
            total_ry = sum(r.get("RY_kN", 0) for r in cr.reactions)
            total_rz = sum(r.get("RZ_kN", 0) for r in cr.reactions)
        except (AttributeError, KeyError, TypeError):
            total_rx = total_ry = total_rz = None

        case_summary[case_name] = {
            "max_disp_x_mm": float(max_disp_x * 1000) if max_disp_x is not None else None,
            "max_disp_y_mm": float(max_disp_y * 1000) if max_disp_y is not None else None,
            "base_reaction_X_kN": float(total_rx) if total_rx is not None else None,
            "base_reaction_Y_kN": float(total_ry) if total_ry is not None else None,
            "base_reaction_Z_kN": float(total_rz) if total_rz is not None else None,
        }
    out["case_summary"] = case_summary

    # Story-wise drift if available
    drift_summary = {}
    for case_name, cr in result.case_results.items():
        story_drift = getattr(cr, "story_drift", None)
        if story_drift:
            drift_summary[case_name] = {
                f"story_{s}": float(d) for s, d in story_drift.items()
            }
    out["story_drift"] = drift_summary

    return out


# ============================================================
# Figures
# ============================================================

def plot_zone_decomposition_top_view(model: StructuralModel, output_path: Path):
    """Top view: column positions, zone rectangles, X/Y axes."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    # Two zones (rectangles)
    # Left wing: x[0,12] y[0,8] active stories 1-5
    # Right wing: x[12,24] y[0,4] active stories 1-3
    ax.add_patch(Rectangle((0, 0), 12, 8, facecolor="#a8d5e2", edgecolor="#2c5f7c",
                           linewidth=1.5, alpha=0.55, label="Zone A (left wing, stories 1–5)"))
    ax.add_patch(Rectangle((12, 0), 12, 4, facecolor="#f5cbcc", edgecolor="#a13d3d",
                           linewidth=1.5, alpha=0.55, label="Zone B (right wing, stories 1–3)"))

    # Plot column positions at base
    col_xy = set()
    for n in model.nodes.values():
        if abs(n.z) < 0.01:
            col_xy.add((n.x, n.y))
    xs = [p[0] for p in col_xy]
    ys = [p[1] for p in col_xy]
    ax.scatter(xs, ys, c="black", s=70, marker="s", zorder=5, label="Column base (auto-detected)")

    # Boundary node merging callout
    # Shared boundary: x=12, y in {0, 4}
    ax.scatter([12, 12], [0, 4], c="red", s=140, marker="o", facecolors="none",
               edgecolors="red", linewidths=2, zorder=6, label="Zone boundary merged nodes")

    # Grid lines
    for x in [0, 6, 12, 18, 24]:
        ax.axvline(x, color="#ddd", linewidth=0.5, linestyle="--", zorder=1)
    for y in [0, 4, 8]:
        ax.axhline(y, color="#ddd", linewidth=0.5, linestyle="--", zorder=1)

    ax.set_xlim(-2, 26)
    ax.set_ylim(-2, 10)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    # In-image title carries NO "Figure N." prefix — the Word caption supplies the
    # figure number (avoids the number going stale on renumbering; this plot is the
    # manuscript's Figure 7).
    ax.set_title("Zone-based decomposition of the L-shaped plan\n"
                 "(left: 5-story 12×8 m wing; right: 3-story 12×4 m wing)",
                 fontsize=10)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(Path(output_path).with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved figure: {output_path} (+ .pdf)")


def plot_lshape_3d_schematic(model: StructuralModel, output_path: Path):
    """3D wireframe of the L-shape model (undeformed)."""
    fig = plt.figure(figsize=(9, 6), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    # Draw all elements
    for e in model.elements.values():
        ni = model.nodes[e.node_i]
        nj = model.nodes[e.node_j]
        if e.elem_type == ElementType.COLUMN:
            color = "#444"
            lw = 1.2
        elif e.elem_type == ElementType.BEAM:
            color = "#777"
            lw = 0.8
        else:
            color = "#a13d3d"
            lw = 1.0
        ax.plot3D([ni.x, nj.x], [ni.y, nj.y], [ni.z, nj.z],
                  color=color, linewidth=lw, alpha=0.85)

    # Color the wings differently by drawing translucent floor patches
    # (drawn schematically; not deformed)
    ax.view_init(elev=18, azim=-55)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    # No "Figure N." prefix — Word caption numbers it (this plot is the manuscript's
    # Figure 8).
    ax.set_title("L-shaped node-element model assembled from the\n"
                 "zone decomposition (5-story left wing + 3-story right wing)",
                 fontsize=10)
    ax.set_box_aspect([2.4, 1.0, 1.6])
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(Path(output_path).with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved figure: {output_path} (+ .pdf)")


# ============================================================
# Main
# ============================================================

def main() -> int:
    print("[*] Building L-shape model...")
    model = build_l_shape_model()
    print(f"    nodes: {len(model.nodes)}, elements: {len(model.elements)},"
          f" stories: {model.num_stories}")

    print("[*] Running analysis (DL, LL, EQX, EQY)...")
    cases = make_lshape_load_cases(model)
    result = analyze_from_model(model, cases)
    print(f"    cases analyzed: {list(result.case_results.keys())}")

    print("[*] Extracting metrics...")
    metrics = extract_metrics(model, result)

    out_json = ROOT / "docs" / "paper1_open_source_alternative" / "validation" / "lshape_demonstration.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[OK] Saved metrics: {out_json.relative_to(ROOT)}")

    print("[*] Generating figures...")
    fig_dir = ROOT / "docs" / "paper1_open_source_alternative" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_zone_decomposition_top_view(model, fig_dir / "fig6_lshape_zones.png")
    plot_lshape_3d_schematic(model, fig_dir / "fig7_lshape_3d.png")

    # Brief summary print
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Nodes: {metrics['model_size']['nodes']}")
    print(f"  Elements: {metrics['model_size']['elements']} "
          f"({metrics['model_size']['columns']} cols + "
          f"{metrics['model_size']['beams']} beams)")
    print(f"  Base nodes (fixed): {metrics['model_size']['base_nodes_fixed']}")
    print(f"  Stories: {metrics['model_size']['stories']}")
    print()
    for case, summary in metrics["case_summary"].items():
        print(f"  [{case}]")
        for k, v in summary.items():
            if v is None:
                continue
            print(f"    {k}: {v:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
