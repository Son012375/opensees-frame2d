"""Bake the landing-page demo bundle.

Runs the real analysis pipeline once per column-section variant and writes a
trimmed, self-contained JSON bundle the landing page can render with no backend.

Nothing here re-implements engineering logic: it calls the same
``generate_all_loads`` / ``analyze_frame_3d_multi`` / ``run_design_check`` path
the web app uses, then selects fields. Every number on the landing page must be
traceable to a file written by this script.

Usage (run against the branch that carries the 10-group design_check):

    python scripts/bake_demo_bundle.py \
        --mcp-root d:/tmp/os-paper/mcp-server \
        --out data/demo_bundle

Requires network: the KDS tables (seismic zone, section properties) are read
from Supabase at bake time. That is fine — baking is offline work; the *output*
is what ships.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── The demo model: the editor's default 3-story preset, unchanged ───────────
BASE_CONFIG: dict[str, Any] = {
    "stories": [
        {"height": 4.0, "usage": "office"},
        {"height": 3.5, "usage": "office"},
        {"height": 3.5, "usage": "office"},
    ],
    "bays_x": [8, 8],
    "bays_y": [8, 8],
    "column_section": "H-300x300",
    "beam_x_section": "H-400x200",
    "beam_y_section": "H-400x200",
    "material_name": "SS275",
    "supports": "fixed",
    "region": "서울",
    "importance": "II",
    "auto_combinations": True,
    "geometric_nonlinearity": "linear",
    "seismic_method": "ELF",
}

DEFAULT_LADDER = ["H-250x250", "H-300x300", "H-350x350", "H-400x400"]

# Check groups in the order the landing page shows them, with display labels.
CHECK_GROUPS: list[tuple[str, str]] = [
    ("member_check", "부재강도"),
    ("drift_check", "층간변위"),
    ("wind_check", "풍하중 사용성"),
    ("pdelta_check", "P-Delta 안정"),
    ("deflection_check", "처짐"),
    ("stability_check", "전도"),
    ("torsion_check", "비틀림 비정형"),
    ("vertical_check", "수직 비정형"),
    ("system_check", "내진시스템"),
    ("accidental_torsion", "우발 비틀림"),
]

# Scalars lifted verbatim from the load reports for the "where did this come
# from" receipt. (key_in_report, label) per section.
LOAD_CHAIN_FIELDS: dict[str, list[str]] = {
    "seismic": [
        "method", "code", "region", "z", "IE", "site_class", "Fa", "Fv",
        "SDS", "SD1", "seismic_system", "R", "Cd", "Ct", "hn_m", "Ta_sec",
        "T_sec", "Cs", "Cs_min", "Cs_max", "Cs_governed_by", "W_kN", "V_kN",
    ],
    "wind": [
        "method", "code", "region", "V0_ms", "V0_fallback", "exposure",
        "Kzt", "Gf", "Cp_total", "Bx_m", "By_m", "total_Fx_kN", "total_Fy_kN",
    ],
    "snow": [
        "method", "code", "region", "Sg_kNm2", "Cb", "Ce", "Ct", "Is",
        "S_formula_kNm2", "S_min_kNm2", "S_design_kNm2", "governed_by",
        "roof_story", "roof_area_m2",
    ],
}


def _round(v: Any, nd: int = 4) -> Any:
    """Round floats for a compact bundle; leave everything else alone."""
    if isinstance(v, float):
        return round(v, nd)
    if isinstance(v, list):
        return [_round(x, nd) for x in v]
    if isinstance(v, dict):
        return {k: _round(x, nd) for k, x in v.items()}
    return v


def _ratio_of(group: dict) -> float | None:
    """The headline number for a check group, whatever it is called."""
    for key in ("max_ratio", "max_theta", "max_interaction_ratio"):
        if group.get(key) is not None:
            try:
                return float(group[key])
            except (TypeError, ValueError):
                pass
    summary = group.get("summary")
    if isinstance(summary, dict) and summary.get("max_interaction_ratio") is not None:
        return float(summary["max_interaction_ratio"])
    return None


def _limit_of(group: dict) -> Any:
    for key in ("allowable", "theta_limit", "limit", "limit_inv"):
        if group.get(key) is not None:
            return group[key]
    return None


def extract_checks(dc: dict) -> list[dict]:
    """The 10-row check table. code_ref is copied, never hand-written."""
    rows = []
    for key, label in CHECK_GROUPS:
        g = dc.get(key)
        if not isinstance(g, dict):
            continue
        rows.append(_round({
            "key": key,
            "label": label,
            "status": g.get("status"),
            "code_ref": g.get("code_ref"),
            "ratio": _ratio_of(g),
            "limit": _limit_of(g),
            "ng_count": g.get("ng") or (g.get("summary") or {}).get("ng"),
        }))
    return rows


def extract_members(dc: dict) -> list[dict]:
    """Per-member result — this is what colours the frame drawing."""
    out = []
    for m in (dc.get("member_check") or {}).get("members") or []:
        ratios = m.get("ratios") or {}
        out.append(_round({
            "id": m.get("member_id"),
            "type": m.get("type"),
            "story": m.get("story"),
            "section": m.get("section"),
            "status": m.get("status"),
            "interaction": ratios.get("interaction"),
            "axial": ratios.get("axial"),
            "shear": ratios.get("shear"),
            "governing_combo": m.get("governing_combo"),
        }))
    return out


def extract_geometry(multi: Any) -> dict:
    """Node coordinates + member end nodes, taken from the analysis result.

    Schemas are frame_3d's, not guessed:
      nodes       -> {"id", "x_m", "y_m", "z_m"}          (frame_3d.py:128)
      member_info -> {"member_id","type","ni","nj","story","section","length_m"}
                                                          (frame_3d.py:810)
    """
    nodes: dict[str, list[float]] = {}
    for n in getattr(multi, "nodes", None) or []:
        if not isinstance(n, dict) or "id" not in n:
            continue
        nodes[str(n["id"])] = [
            round(float(n.get("x_m") or 0.0), 4),
            round(float(n.get("y_m") or 0.0), 4),
            round(float(n.get("z_m") or 0.0), 4),
        ]

    members = []
    for info in getattr(multi, "member_info", None) or []:
        if not isinstance(info, dict):
            continue
        members.append({
            "id": info.get("member_id"),
            "type": info.get("type"),
            "story": info.get("story"),
            "i": info.get("ni"),
            "j": info.get("nj"),
            "length_m": info.get("length_m"),
        })

    # Base nodes = supports, same rule visualization_3d.py:36 uses.
    support_ids = [
        int(n["id"]) for n in getattr(multi, "nodes", None) or []
        if isinstance(n, dict) and abs(float(n.get("z_m") or 0.0)) < 0.001
    ]

    return {
        "nodes": nodes,
        "members": members,
        "support_ids": support_ids,
        "support_type": getattr(multi, "supports", None),
        "num_nodes": len(nodes),
        "num_members": len(members),
    }


def extract_modal(multi: Any) -> dict:
    """Modal summary. Mode *shapes* are deliberately dropped — they are the
    bulk of the payload and the landing page does not animate them.

    ``mass_info`` is carried verbatim because it states, in the software's own
    words, that member self-weight is excluded from the seismic mass. The
    limitations section quotes it rather than paraphrasing.
    """
    ma = getattr(multi, "modal_analysis", None)
    if not isinstance(ma, dict):
        return {}
    keep: dict[str, Any] = {}
    for k in ("num_modes", "fundamental_periods", "cumulative_participation",
              "mass_info", "story_weights_kN", "weight_source",
              "diaphragm_auto_enabled", "mode_count_auto_increased"):
        if k in ma:
            keep[k] = _round(ma[k])
    keep["modes"] = [
        _round({
            "mode": m.get("mode"),
            "period_s": m.get("period_s"),
            "direction": m.get("direction"),
            "dominance_pct": m.get("dominance_pct"),
            "mass_participation": m.get("mass_participation"),
        })
        for m in (ma.get("modes") or [])[:6]
        if isinstance(m, dict)
    ]
    return keep


def run_variant(section: str, mcp_root: str) -> dict:
    """One full pipeline run. Returns the trimmed variant record."""
    sys.path.insert(0, mcp_root)
    from core.building_model import BuildingModel
    from core.load_generator import generate_all_loads
    from core.frame_3d import analyze_frame_3d_multi
    from core.design_check import run_design_check

    cfg = dict(BASE_CONFIG)
    cfg["column_section"] = section

    model = BuildingModel.from_json(cfg)
    load_result = generate_all_loads(model)

    kwargs = model.to_frame3d_kwargs()
    kwargs["load_cases"] = load_result["load_cases"]
    kwargs["load_combinations"] = load_result["load_combinations"]
    kwargs["modal_analysis"] = True
    multi = analyze_frame_3d_multi(**kwargs)

    dc = run_design_check(multi, model, load_result["reports"].get("seismic"))

    reports = load_result["reports"]
    load_chain = {}
    for sec, fields in LOAD_CHAIN_FIELDS.items():
        r = reports.get(sec) or {}
        load_chain[sec] = _round({f: r.get(f) for f in fields if f in r})
    seismic = reports.get("seismic") or {}
    load_chain["seismic_story_forces"] = _round(seismic.get("story_forces") or [])

    mc_summary = _round((dc.get("member_check") or {}).get("summary") or {})

    return {
        "variant": section,
        "overall_status": dc.get("overall_status"),
        "checks": extract_checks(dc),
        "member_summary": mc_summary,
        "members": extract_members(dc),
        "governing_combo": ((dc.get("member_check") or {}).get("critical_members") or [{}])[0].get("governing_combo"),
        "load_chain": load_chain,
        "combos": list(load_result["load_combinations"].keys())
        if isinstance(load_result["load_combinations"], dict)
        else [str(c) for c in load_result["load_combinations"]],
        "modal": extract_modal(multi),
        "model_info": {
            "num_stories": getattr(multi, "num_stories", None),
            "stories": list(getattr(multi, "stories", []) or []),
            "bays_x": list(getattr(multi, "bays_x", []) or []),
            "bays_y": list(getattr(multi, "bays_y", []) or []),
            "total_height": getattr(multi, "total_height", None),
            "column_section": getattr(multi, "column_section", None),
            "beam_x_section": getattr(multi, "beam_x_section", None),
            "beam_y_section": getattr(multi, "beam_y_section", None),
            "material_name": getattr(multi, "material_name", None),
            "fy_MPa": getattr(multi, "fy_MPa", None),
            "E_MPa": getattr(multi, "E_MPa", None),
            "column_A_mm2": getattr(multi, "column_A_mm2", None),
            "column_Ix_mm4": getattr(multi, "column_Ix_mm4", None),
            "column_Iy_mm4": getattr(multi, "column_Iy_mm4", None),
            "column_J_mm4": getattr(multi, "column_J_mm4", None),
            "supports": getattr(multi, "supports", None),
            "geometric_nonlinearity": getattr(multi, "geometric_nonlinearity", None),
        },
        "_geometry": extract_geometry(multi),
    }


# ── SVG frame drawing ────────────────────────────────────────────────────────
# A fixed isometric projection baked at build time. The browser never projects
# anything: it only toggles CSS classes on <line> elements keyed by member id.
# No WebGL, no canvas, no drag handling — so nothing can fight page scrolling.

import math

# Azimuth is deliberately NOT 45°. At exactly 45° the columns on the x == y
# diagonal collapse onto one screen column and read as a single tall member.
_AZIMUTH_DEG = 32.0
_ELEVATION_DEG = 26.0

_CA = math.cos(math.radians(_AZIMUTH_DEG))
_SA = math.sin(math.radians(_AZIMUTH_DEG))
_SE = math.sin(math.radians(_ELEVATION_DEG))
_CE = math.cos(math.radians(_ELEVATION_DEG))


def _project(x: float, y: float, z: float) -> tuple[float, float]:
    """Axonometric projection. Z is up on screen (SVG y grows downward)."""
    px = x * _CA - y * _SA
    py = (x * _SA + y * _CA) * _SE - z * _CE
    return px, py


def build_frame_svg(geometry: dict, width: int = 720, pad: int = 28) -> str:
    nodes = {nid: _project(*c) for nid, c in geometry["nodes"].items()}
    xs = [p[0] for p in nodes.values()]
    ys = [p[1] for p in nodes.values()]
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    span_x, span_y = (maxx - minx) or 1.0, (maxy - miny) or 1.0
    scale = (width - 2 * pad) / span_x
    height = round(span_y * scale + 2 * pad)

    def to_px(p: tuple[float, float]) -> tuple[float, float]:
        return (round((p[0] - minx) * scale + pad, 2),
                round((p[1] - miny) * scale + pad, 2))

    depth = {nid: (geometry["nodes"][nid][0] + geometry["nodes"][nid][1])
             for nid in geometry["nodes"]}

    members = [m for m in geometry["members"]
               if str(m.get("i")) in nodes and str(m.get("j")) in nodes]
    # Painter's algorithm: far members first.
    members.sort(key=lambda m: depth[str(m["i"])] + depth[str(m["j"])])

    lines = []
    for m in members:
        x1, y1 = to_px(nodes[str(m["i"])])
        x2, y2 = to_px(nodes[str(m["j"])])
        lines.append(
            f'<line class="mb mb--{m["type"]}" data-m="{m["id"]}" '
            f'data-type="{m["type"]}" data-story="{m["story"]}" '
            f'x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"/>'
        )

    marks = []
    for nid in geometry.get("support_ids", []):
        px, py = to_px(nodes[str(nid)])
        marks.append(
            f'<path class="sup" d="M{px - 6} {py + 6} L{px + 6} {py + 6} '
            f'L{px} {py} Z"/>'
        )

    return (
        f'<svg class="frame-svg" viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg" role="img" '
        f'aria-label="3층 강구조 골조 — 절점 {geometry["num_nodes"]}개, 부재 '
        f'{geometry["num_members"]}개">\n'
        f'  <g class="members">\n    ' + "\n    ".join(lines) + "\n  </g>\n"
        f'  <g class="supports">\n    ' + "\n    ".join(marks) + "\n  </g>\n"
        "</svg>\n"
    )


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcp-root",
                    help="path to the mcp-server dir carrying the 10-group design_check")
    ap.add_argument("--out", default="data/demo_bundle")
    ap.add_argument("--sections", default=",".join(DEFAULT_LADDER))
    ap.add_argument("--svg-only", action="store_true",
                    help="redraw frame.svg from the existing geometry.json "
                         "without re-running any analysis")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ladder = [s.strip() for s in args.sections.split(",") if s.strip()]

    if args.svg_only:
        geo_path = out / "geometry.json"
        if not geo_path.exists():
            print(f"[bake] {geo_path} not found — run a full bake first", file=sys.stderr)
            return 1
        geometry = json.loads(geo_path.read_text(encoding="utf-8"))
        svg = build_frame_svg(geometry)
        (out / "frame.svg").write_text(svg, encoding="utf-8")
        print(f"[bake] frame.svg {(out / 'frame.svg').stat().st_size / 1024:.1f} KB "
              f"({geometry['num_members']} members)")
        return 0

    if not args.mcp_root:
        ap.error("--mcp-root is required unless --svg-only is given")

    geometry: dict | None = None
    index: list[dict] = []

    for section in ladder:
        print(f"[bake] {section} ...", flush=True)
        rec = run_variant(section, args.mcp_root)
        geo = rec.pop("_geometry")
        if geometry is None:
            geometry = geo
        elif geo["num_nodes"] != geometry["num_nodes"]:
            print(f"  ! geometry differs for {section} — bundle assumes one geometry",
                  file=sys.stderr)

        slug = section.replace("-", "").replace("x", "X").lower()
        path = out / f"variant_{slug}.json"
        path.write_text(json.dumps(rec, ensure_ascii=False, separators=(",", ":")),
                        encoding="utf-8")

        summary = rec["member_summary"]
        index.append({
            "variant": section,
            "file": path.name,
            "overall_status": rec["overall_status"],
            "max_interaction_ratio": summary.get("max_interaction_ratio"),
            "ng": summary.get("ng"),
            "total": summary.get("total"),
            "ng_groups": [c["label"] for c in rec["checks"] if c["status"] == "NG"],
        })
        print(f"  -> {rec['overall_status']}  max_ratio={summary.get('max_interaction_ratio')}"
              f"  NG {summary.get('ng')}/{summary.get('total')}"
              f"  ({path.stat().st_size/1024:.0f} KB)", flush=True)

    (out / "geometry.json").write_text(
        json.dumps(geometry, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    (out / "frame.svg").write_text(build_frame_svg(geometry), encoding="utf-8")

    manifest = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "mcp_root": args.mcp_root,
        "base_config": BASE_CONFIG,
        "ladder": index,
        "note": "모든 값은 이 스크립트가 실행한 실제 해석 결과입니다. 브라우저에서 재계산하지 않습니다.",
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")

    print("\n[bake] ladder")
    for row in index:
        print(f"  {row['variant']:12s} {row['overall_status']:3s} "
              f"ratio={row['max_interaction_ratio']}  NG={row['ng']}/{row['total']}  "
              f"groups={row['ng_groups']}")
    total_kb = sum(p.stat().st_size for p in out.glob("*.json")) / 1024
    print(f"[bake] bundle {total_kb:.0f} KB -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
