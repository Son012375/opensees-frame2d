"""Publication-quality figures for the open-source-alternative paper (G1).

Data-driven: reads the real benchmark JSONs under tests/benchmark/ and
reproduces the comparison logic of tests/benchmark/compare.py, so the figures
are auditable and cannot drift from the reported numbers. Self-verifies the
reconstructed counts against opensees_results/summary.json (112/100/12/0).

Outputs English-labelled 300-dpi PNG + vector PDF into
docs/paper1_open_source_alternative/figures/:
  - fig6_parity_plot.{png,pdf}        OpenSeesPy vs Midas Gen metric-level parity
  - fig7_ok_check_distribution.{png,pdf}  per-case OK/CHECK + diff% distribution

Commercial reference = Midas Gen only. (ETABS case4.json matches OpenSees to
~15 significant figures, i.e. it is seeded from the OpenSees run rather than an
independent extraction, so it is intentionally NOT used as a parity baseline.)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "tests" / "benchmark"
OPS_DIR = BENCH / "opensees_results"
MIDAS_DIR = BENCH / "midas_results"
OUT_DIR = ROOT / "docs" / "paper1_open_source_alternative" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CASES = ["case1", "case2", "case3", "case4", "case5"]
CASE_SHORT = {
    "case1": "Case 1\n2D simple beam",
    "case2": "Case 2\n2D portal",
    "case3": "Case 3\n2D 3-story",
    "case4": "Case 4\n3D 2-story",
    "case5": "Case 5\n3D 5-story (P-Δ)",
}

TOL_OK = 1.0        # diff% <= 1% -> OK
TOL_FAIL = 5.0      # diff% > 5% -> FAIL
NEAR_ZERO = 1e-9    # |value| below this is an exact (essentially-zero) match

C_OK, C_CHECK, C_FAIL = "#2E7D32", "#EF6C00", "#C62828"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.unicode_minus": False,
})


def _load(d: Path, case: str) -> dict:
    with open(d / f"{case}.json", encoding="utf-8") as f:
        return json.load(f)


def build_rows() -> list[dict]:
    """Pair every OpenSees metric with its Midas reference (non-null) and
    classify exactly as compare.py does."""
    rows: list[dict] = []
    for case in CASES:
        ops = _load(OPS_DIR, case)
        mid = _load(MIDAS_DIR, case)
        for metric, ov in ops.items():
            mv = mid.get(metric)
            if mv is None:
                continue
            ref = max(abs(ov), abs(mv))
            if ref <= 1e-12:
                diff, status = 0.0, "OK"
            else:
                diff = abs(ov - mv) / ref * 100.0
                if diff <= TOL_OK:
                    status = "OK"
                elif diff > TOL_FAIL:
                    status = "FAIL"
                else:
                    status = "CHECK"
            rows.append({
                "case": case, "metric": metric, "ops": ov, "mid": mv,
                "ref": ref, "diff": diff, "status": status,
            })
    return rows


def verify(rows: list[dict]) -> None:
    summ = json.loads((OPS_DIR / "summary.json").read_text(encoding="utf-8"))
    n = len(rows)
    ok = sum(r["status"] == "OK" for r in rows)
    chk = sum(r["status"] == "CHECK" for r in rows)
    fail = sum(r["status"] == "FAIL" for r in rows)
    print(f"reconstructed: total={n} ok={ok} check={chk} fail={fail}")
    print(f"summary.json : total={summ['total_metrics']} ok={summ['ok']} "
          f"check={summ['check']} fail={summ['fail']}")
    assert n == summ["total_metrics"], "metric count mismatch vs summary.json"
    assert ok == summ["ok"] and chk == summ["check"] and fail == summ["fail"], \
        "status counts mismatch vs summary.json"
    print("OK: reconstructed counts match summary.json\n")


def fig_parity(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 6.0))

    plotted = [r for r in rows if r["ref"] > NEAR_ZERO]
    zeros = [r for r in rows if r["ref"] <= NEAR_ZERO]

    lo = min(r["ref"] for r in plotted)
    hi = max(r["ref"] for r in plotted)
    lo, hi = lo / 2.0, hi * 2.0
    line = np.array([lo, hi])

    # tolerance bands around the y = x diagonal (in log space)
    ax.fill_between(line, line * (1 - TOL_FAIL / 100), line * (1 + TOL_FAIL / 100),
                    color=C_CHECK, alpha=0.10, lw=0, zorder=0,
                    label=f"±{TOL_FAIL:.0f}% band")
    ax.fill_between(line, line * (1 - TOL_OK / 100), line * (1 + TOL_OK / 100),
                    color=C_OK, alpha=0.16, lw=0, zorder=1,
                    label=f"±{TOL_OK:.0f}% band")
    ax.plot(line, line, color="0.35", lw=1.0, ls="--", zorder=2, label="y = x")

    for status, color, marker, z in [("OK", C_OK, "o", 3),
                                     ("CHECK", C_CHECK, "s", 4),
                                     ("FAIL", C_FAIL, "^", 5)]:
        xs = [abs(r["mid"]) for r in plotted if r["status"] == status]
        ys = [abs(r["ops"]) for r in plotted if r["status"] == status]
        if not xs:
            continue
        ax.scatter(xs, ys, s=34, c=color, marker=marker, edgecolors="white",
                   linewidths=0.4, alpha=0.9, zorder=z,
                   label=f"{status} (n={sum(r['status']==status for r in rows)})")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Midas Gen reference  |value|  (mm, kN, kN·m, drift)")
    ax.set_ylabel("OpenSeesPy  |value|  (mm, kN, kN·m, drift)")
    ax.set_title("Metric-level parity: OpenSeesPy vs Midas Gen\n"
                 "5 benchmark cases, 112 response metrics", pad=10)
    ax.grid(True, which="both", alpha=0.15)
    ax.legend(loc="upper left", framealpha=0.9, ncol=1)

    n_ok = sum(r["status"] == "OK" for r in rows)
    n_chk = sum(r["status"] == "CHECK" for r in rows)
    note = (f"{n_ok} of {len(rows)} metrics agree within {TOL_OK:.0f}%\n"
            f"{n_chk} CHECK (all Case 4, ≤{max(r['diff'] for r in rows):.2f}%); "
            f"0 FAIL\n"
            f"{len(zeros)} near-zero metrics (|value|<1e-9) omitted from log axes")
    ax.text(0.97, 0.03, note, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.5, bbox=dict(boxstyle="round,pad=0.4", fc="white",
                                    ec="0.7", alpha=0.92))

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"fig6_parity_plot.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {OUT_DIR / 'fig6_parity_plot.png'} (+ .pdf)")


def fig_distribution(rows: list[dict]) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.8),
                                   gridspec_kw={"width_ratios": [1.0, 1.15]})

    # ---- Left: per-case stacked OK/CHECK counts ----
    ok = [sum(r["status"] == "OK" for r in rows if r["case"] == c) for c in CASES]
    chk = [sum(r["status"] == "CHECK" for r in rows if r["case"] == c) for c in CASES]
    maxd = [max([r["diff"] for r in rows if r["case"] == c], default=0.0) for c in CASES]
    y = np.arange(len(CASES))

    ax1.barh(y, ok, color=C_OK, edgecolor="white", label="OK (≤1%)")
    ax1.barh(y, chk, left=ok, color=C_CHECK, edgecolor="white", label="CHECK (1–5%)")
    for i, (o, c, d) in enumerate(zip(ok, chk, maxd)):
        tot = o + c
        tag = "all ≤1%" if d < TOL_OK else f"max {d:.2f}%"
        ax1.text(tot + 0.6, i, f"{tot}  ({tag})", va="center", fontsize=9)
    ax1.set_yticks(y)
    ax1.set_yticklabels([CASE_SHORT[c] for c in CASES], fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlabel("number of compared metrics")
    ax1.set_xlim(0, max(o + c for o, c in zip(ok, chk)) * 1.45)
    ax1.set_title("Per-case agreement (112 metrics, 100 OK / 12 CHECK / 0 FAIL)")
    ax1.legend(loc="lower right")
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.grid(axis="x", alpha=0.2)

    # ---- Right: diff% of every metric, by case ----
    rng = np.random.default_rng(42)
    cidx = {c: i for i, c in enumerate(CASES)}
    for status, color, marker in [("OK", C_OK, "o"), ("CHECK", C_CHECK, "s"),
                                  ("FAIL", C_FAIL, "^")]:
        xs = [cidx[r["case"]] + rng.uniform(-0.18, 0.18)
              for r in rows if r["status"] == status]
        ys = [r["diff"] for r in rows if r["status"] == status]
        if xs:
            ax2.scatter(xs, ys, s=26, c=color, marker=marker, edgecolors="white",
                        linewidths=0.3, alpha=0.85, label=status, zorder=3)
    ax2.axhline(TOL_OK, color=C_OK, ls="--", lw=1.0, zorder=1,
                label=f"{TOL_OK:.0f}% (OK threshold)")
    ax2.axhline(TOL_FAIL, color=C_FAIL, ls=":", lw=1.0, zorder=1,
                label=f"{TOL_FAIL:.0f}% (FAIL threshold)")
    ax2.set_xticks(list(cidx.values()))
    ax2.set_xticklabels([CASE_SHORT[c] for c in CASES], fontsize=9)
    ax2.set_ylabel("relative difference vs Midas Gen (%)")
    ax2.set_ylim(-0.25, max(TOL_FAIL + 0.4, max(r["diff"] for r in rows) + 0.4))
    ax2.set_title("Per-metric difference distribution")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"fig7_ok_check_distribution.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {OUT_DIR / 'fig7_ok_check_distribution.png'} (+ .pdf)")


def fig_screening() -> None:
    """Fig 5 (manuscript): summary of preliminary review outputs for the
    IFC-derived example building — member-strength utilisation + story drift."""
    src = ROOT / "docs" / "paper1_open_source_alternative" / "validation" / "example_section4_results.json"
    data = json.loads(src.read_text(encoding="utf-8"))
    dc = data["design_check"]
    members = dc["member_check"]["members"]
    msum = dc["member_check"]["summary"]
    checks = dc["drift_check"]["checks"]
    drift_limit_ratio = 1.0  # checks store ratio = demand/allowable

    def _is_col(m):
        return str(m.get("type", "")).lower().startswith("col")
    inter = [m["ratios"]["interaction"] for m in members]
    beams = [m["ratios"]["interaction"] for m in members if not _is_col(m)]
    cols = [m["ratios"]["interaction"] for m in members if _is_col(m)]
    n_ok = msum.get("ok", sum(m["status"] == "OK" for m in members))
    n_ng = msum.get("ng", sum(m["status"] != "OK" for m in members))
    max_int = msum.get("max_interaction_ratio", max(inter))
    print(f"  [Fig5] members={len(members)} ok={n_ok} ng={n_ng} "
          f"max_interaction={max_int:.3f} max_beam={max(beams) if beams else 0:.3f} "
          f"max_col={max(cols) if cols else 0:.3f}")

    # governing drift ratio per (story, direction)
    gov: dict[tuple, float] = {}
    for c in checks:
        key = (int(c["story"]), str(c["direction"]).upper())
        gov[key] = max(gov.get(key, 0.0), float(c["ratio"]))
    stories = sorted({k[0] for k in gov})
    dirs = ["X", "Y"]
    print(f"  [Fig5] governing drift ratios: "
          + ", ".join(f"S{s}{d}={gov.get((s,d),0):.2f}" for s in stories for d in dirs))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.6),
                                   gridspec_kw={"width_ratios": [1.1, 1.0]})

    # ---- Left: member interaction-ratio histogram ----
    bins = np.linspace(0, 1.0, 21)
    ax1.hist([cols, beams], bins=bins, stacked=True,
             color=["#1565C0", "#2E7D32"], edgecolor="white",
             label=[f"columns (n={len(cols)})", f"beams (n={len(beams)})"])
    ax1.axvline(1.0, color=C_FAIL, ls="--", lw=1.3, label="utilisation limit = 1.0")
    ax1.axvline(max_int, color="#EF6C00", ls=":", lw=1.3)
    # governing (max-interaction) member descriptor — data-driven so the label
    # cannot drift from the JSON (e.g. after the 2026-07 gravity fix moved the
    # governing member from a 2nd-story H1-1b column to a 1st-story H1-1a column).
    gov_m = max(members, key=lambda m: m["ratios"]["interaction"])
    _story = int(gov_m.get("story", 0))
    _ord = {1: "1st", 2: "2nd", 3: "3rd"}.get(_story, f"{_story}th")
    _typ = "column" if _is_col(gov_m) else "beam"
    _formula = gov_m["ratios"].get("formula", "")
    _desc = f"{_ord}-story {_typ}" + (f",\n{_formula}" if _formula else "")
    ax1.annotate(f"max = {max_int:.3f}\n({_desc})",
                 xy=(max_int, 0), xytext=(max_int - 0.02, ax1.get_ylim()[1] * 0.55),
                 ha="right", fontsize=8.5, color="#EF6C00",
                 arrowprops=dict(arrowstyle="->", color="#EF6C00", lw=1.0))
    ax1.set_xlabel("AISC 360-22 interaction ratio (demand / capacity)")
    ax1.set_ylabel("number of members")
    ax1.set_xlim(0, 1.05)
    ax1.set_title(f"Member-strength screening — {len(members)} members, "
                  f"all OK ({n_ng} NG)")
    ax1.legend(loc="upper right", fontsize=8.5)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.grid(axis="y", alpha=0.2)

    # ---- Right: governing story-drift ratio per story/direction ----
    y = np.arange(len(stories))
    w = 0.38
    rx = [gov.get((s, "X"), 0.0) for s in stories]
    ry = [gov.get((s, "Y"), 0.0) for s in stories]
    ax2.barh(y - w / 2, rx, w, color="#1565C0", edgecolor="white", label="X direction")
    ax2.barh(y + w / 2, ry, w, color="#EF6C00", edgecolor="white", label="Y direction")
    ax2.axvline(drift_limit_ratio, color=C_FAIL, ls="--", lw=1.3,
                label="KDS drift limit")
    for i, (vx, vy) in enumerate(zip(rx, ry)):
        ax2.text(vx + 0.01, i - w / 2, f"{vx:.2f}", va="center", fontsize=8.5)
        ax2.text(vy + 0.01, i + w / 2, f"{vy:.2f}", va="center", fontsize=8.5)
    ax2.set_yticks(y)
    ax2.set_yticklabels([f"Story {s}" for s in stories])
    ax2.invert_yaxis()
    ax2.set_xlabel("story-drift ratio (amplified demand / allowable)")
    ax2.set_xlim(0, max(1.1, max(rx + ry) * 1.25))
    ax2.set_title("Story-drift screening vs KDS limit")
    ax2.legend(loc="lower right", fontsize=8.5)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="x", alpha=0.2)

    fig.suptitle("Preliminary review outputs — IFC-derived example building "
                 "(review support, not full code compliance)", fontsize=11, y=1.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"fig5_screening_summary.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {OUT_DIR / 'fig5_screening_summary.png'} (+ .pdf)")


def main() -> None:
    rows = build_rows()
    verify(rows)
    fig_parity(rows)
    fig_distribution(rows)
    fig_screening()
    print("\ndone.")


if __name__ == "__main__":
    main()
