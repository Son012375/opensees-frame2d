# -*- coding: utf-8 -*-
"""Rebuild Figure 1 (workflow) at a printable aspect ratio.

The original fig1_open_workflow was a single ultra-wide row per pipeline (6.7:1),
so at column width the box text became microscopic and Word cropped the sides.
This rebuild keeps the two-pipeline comparison but wraps the longer open-source
chain onto two rows, giving a ~2.1:1 figure whose text is legible at full width.
300-dpi PNG + vector PDF, matching the manuscript figure style.

Usage: python scripts/_render_fig1_workflow_2026-07-09.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "paper1_open_source_alternative" / "figures"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "savefig.dpi": 300,
})

BOX_FC = "#e9e9f7"     # light lavender box
BOX_EC = "#8a8ac8"
OUT_FC = "#dfe9f7"     # outcome box (slightly blue)
REGION_FC = "#fffde7"  # pale-yellow region background
REGION_EC = "#e3dca0"

fig, ax = plt.subplots(figsize=(10.0, 4.7))
ax.set_xlim(0, 100)
ax.set_ylim(0, 47)
ax.axis("off")


def region(x, y, w, h, title):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                                fc=REGION_FC, ec=REGION_EC, lw=1.2, zorder=1))
    ax.text(x + w / 2, y + h - 1.6, title, ha="center", va="center",
            fontsize=11, fontweight="bold", color="#555", zorder=3)


def box(cx, cy, w, text, fc=BOX_FC, h=6.2):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.15,rounding_size=0.8",
                                fc=fc, ec=BOX_EC, lw=1.1, zorder=4))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=9.2, zorder=5)
    return (cx, cy, w, h)


def arrow(a, b, style="-", color="#333", lw=1.5, rad=0.0):
    ax.add_patch(FancyArrowPatch((a[0] + a[2] / 2, a[1]), (b[0] - b[2] / 2, b[1]),
                                 arrowstyle="-|>", mutation_scale=13, lw=lw,
                                 color=color, linestyle=style,
                                 connectionstyle=f"arc3,rad={rad}", zorder=6))


def elbow(pts, style="-", color="#333", lw=1.5):
    """Orthogonal polyline with an arrowhead on the final segment (clean routing)."""
    if len(pts) > 2:
        xs = [p[0] for p in pts[:-1]]
        ys = [p[1] for p in pts[:-1]]
        ax.plot(xs, ys, color=color, lw=lw, ls=style, solid_capstyle="round", zorder=6)
    ax.annotate("", xy=pts[-1], xytext=pts[-2],
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, linestyle=style,
                                shrinkA=0, shrinkB=0, mutation_scale=13), zorder=6)


# ---- Top: typical commercial-centered workflow -----------------------------
region(2, 31, 96, 14.5, "Typical commercial-centered workflow")
ty = 36.5
c1 = box(13, ty, 18, "BIM / IFC\nmodel")
c2 = box(37, ty, 24, "Manual remodeling or\nproprietary preprocessing")
c3 = box(62, ty, 20, "Commercial\nanalysis model")
c4 = box(86, ty, 18, "Analysis\nresults")
for a, b in [(c1, c2), (c2, c3), (c3, c4)]:
    arrow(a, b)

# ---- Bottom: proposed open-source workflow ---------------------------------
region(2, 2, 96, 26, "Proposed open-source workflow")
r1 = 20.0   # row 1 y
r2 = 8.5    # row 2 y
o1 = box(13, r1, 18, "BIM / IFC\nmodel")
o2 = box(35, r1, 20, "Node-element\nIFC parser")
o3 = box(60, r1, 24, "StructuralModel graph\n(nodes + elements +\nsupports + sections)", h=8.0)
o4 = box(86, r1, 18, "KDS load\nautomation")
for a, b in [(o1, o2), (o2, o3), (o3, o4)]:
    arrow(a, b)

o5 = box(13, r2, 18, "OpenSeesPy\nanalysis")
o6 = box(37, r2, 22, "Displacement, drift,\nreactions, member forces")
o7 = box(62, r2, 18, "Metric-level\ncomparison")
o8 = box(86, r2, 18, "Open-source\nalternative potential", fc=OUT_FC)
for a, b in [(o5, o6), (o6, o7), (o7, o8)]:
    arrow(a, b)

# wrap arrow: end of row1 (KDS load automation) -> start of row2 (OpenSeesPy),
# routed through the clear gap band at y=14 (below row1, above row2).
elbow([(86, r1 - 3.1), (86, 14.2), (13, 14.2), (13, r2 + 3.1)], lw=1.5)

# dotted commercial-reference-baseline link: commercial model -> metric comparison,
# routed down the clear x=74 corridor (graph|KDS gap above, Metric|outcome gap below),
# then a short left jog into the TOP of Metric so it avoids every box and the solid
# Metric->outcome arrow.
elbow([(62, ty - 3.1), (62, 28.6), (74, 28.6), (74, 12.6), (64, 12.6), (64, r2 + 3.1)],
      style=(0, (4, 3)), color="#b06a00", lw=1.4)
ax.text(53, 29.9, "commercial reference baseline", ha="center", va="center",
        fontsize=8.2, style="italic", color="#b06a00", zorder=7)

fig.tight_layout(pad=0.4)
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig1_open_workflow.{ext}", bbox_inches="tight")
print("wrote", OUT / "fig1_open_workflow.png", "and .pdf")

from PIL import Image
im = Image.open(OUT / "fig1_open_workflow.png")
print("PNG size:", im.size, "aspect %.2f" % (im.size[0] / im.size[1]))
