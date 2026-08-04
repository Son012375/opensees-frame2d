"""§4 응용예제 재해석 — V_base = 223.5 kN 정합 결과 산출.

example_input.json의 3층 3×2베이 건물을 현재 코드로 재해석하여
§4.2 본문에 들어갈 응답값을 재산출한다:
  - 최대 옥상 변위 X/Y
  - 최대 층간변위 (탄성 → Cd 증폭 → 설계 drift)
  - 지배 하중조합
  - 부재강도 interaction ratio 요약

결과는 example_section4_results.json으로 저장하고 콘솔에 §4.2 paragraph
update 초안을 출력한다.

Usage:
    python scripts/_rerun_section4_analysis.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mcp-server"))

from core.building_model import BuildingModel
from core.load_generator import generate_all_loads
from core.frame_3d import analyze_frame_3d_multi


INPUT = ROOT / "docs" / "paper1_open_source_alternative" / "validation" / "example_input.json"
OUTPUT = ROOT / "docs" / "paper1_open_source_alternative" / "validation" / "example_section4_results.json"


def main() -> int:
    print(f"[*] Loading input: {INPUT.relative_to(ROOT)}")
    config = json.loads(INPUT.read_text(encoding="utf-8"))
    config.pop("_meta", None)

    model = BuildingModel.from_json(config)
    print(f"[*] Model: {model.num_stories} stories, "
          f"{len(model.bays_x)}x{len(model.bays_y)} bays, "
          f"H={model.total_height} m")

    # 1. Loads
    load_result = generate_all_loads(model)
    seismic_rpt = load_result["reports"].get("seismic", {})
    print(f"[*] Seismic V_base = {seismic_rpt.get('V_kN')} kN, "
          f"Cs = {seismic_rpt.get('Cs')}, W = {seismic_rpt.get('W_kN')} kN")

    # 2. Analysis
    kwargs = model.to_frame3d_kwargs()
    kwargs["load_cases"] = load_result["load_cases"]
    kwargs["load_combinations"] = load_result["load_combinations"]
    kwargs["modal_analysis"] = True

    print(f"[*] Running analyze_frame_3d_multi ...")
    multi = analyze_frame_3d_multi(**kwargs)
    print(f"[*] Analysis done: {multi.num_elements} elements, "
          f"{len(multi.case_results)} cases, {len(multi.combo_results)} combos")

    # 3. Envelope across combos
    env = {
        "max_dx_mm": 0.0, "gov_combo_dx": "",
        "max_dy_mm": 0.0, "gov_combo_dy": "",
        "max_drift_x": 0.0, "gov_combo_drift_x": "",
        "max_drift_y": 0.0, "gov_combo_drift_y": "",
        "drift_x_story": None, "drift_y_story": None,
    }
    per_combo = {}

    for cn, cr in multi.combo_results.items():
        dx = abs(cr.max_displacement_x)
        dy = abs(cr.max_displacement_y)
        drx = cr.max_drift_x
        dry = cr.max_drift_y
        per_combo[cn] = {
            "max_dx_mm": round(cr.max_displacement_x, 3),
            "max_dy_mm": round(cr.max_displacement_y, 3),
            "max_drift_x": round(drx, 6),
            "max_drift_y": round(dry, 6),
            "story_drifts": cr.story_drifts,
        }
        if dx > env["max_dx_mm"]:
            env["max_dx_mm"] = dx
            env["gov_combo_dx"] = cn
        if dy > env["max_dy_mm"]:
            env["max_dy_mm"] = dy
            env["gov_combo_dy"] = cn
        if drx > env["max_drift_x"]:
            env["max_drift_x"] = drx
            env["gov_combo_drift_x"] = cn
            # which story
            for sd in cr.story_drifts:
                if abs(sd.get("drift_x", 0) - drx) < 1e-9:
                    env["drift_x_story"] = sd["story"]
                    break
        if dry > env["max_drift_y"]:
            env["max_drift_y"] = dry
            env["gov_combo_drift_y"] = cn
            for sd in cr.story_drifts:
                if abs(sd.get("drift_y", 0) - dry) < 1e-9:
                    env["drift_y_story"] = sd["story"]
                    break

    # round display values
    env["max_dx_mm"] = round(env["max_dx_mm"], 3)
    env["max_dy_mm"] = round(env["max_dy_mm"], 3)
    env["max_drift_x"] = round(env["max_drift_x"], 6)
    env["max_drift_y"] = round(env["max_drift_y"], 6)

    # 4. Cd amplification for design drift
    Cd = seismic_rpt.get("Cd", 3.0)
    IE = seismic_rpt.get("IE", 1.0)
    drift_limit = 0.020  # KDS 41 17 00 §6.8 importance II
    design_drift_x = env["max_drift_x"] * Cd / IE
    design_drift_y = env["max_drift_y"] * Cd / IE
    ratio_x = design_drift_x / drift_limit
    ratio_y = design_drift_y / drift_limit

    print()
    print("=" * 72)
    print(f"RESULTS SUMMARY (envelope across {len(multi.combo_results)} combinations)")
    print("=" * 72)
    print(f"  Max displacement X:   {env['max_dx_mm']:.3f} mm   (gov: {env['gov_combo_dx']})")
    print(f"  Max displacement Y:   {env['max_dy_mm']:.3f} mm   (gov: {env['gov_combo_dy']})")
    print(f"  Max elastic drift X:  {env['max_drift_x']*100:.4f} %   "
          f"(Story {env['drift_x_story']}, gov: {env['gov_combo_drift_x']})")
    print(f"  Max elastic drift Y:  {env['max_drift_y']*100:.4f} %   "
          f"(Story {env['drift_y_story']}, gov: {env['gov_combo_drift_y']})")
    print()
    print(f"  Cd = {Cd}, IE = {IE}, drift limit = {drift_limit*100:.1f}%")
    print(f"  Design drift X (= elastic × Cd/IE): {design_drift_x*100:.4f} %   "
          f"(ratio {ratio_x:.3f})")
    print(f"  Design drift Y:                      {design_drift_y*100:.4f} %   "
          f"(ratio {ratio_y:.3f})")
    print(f"  Limit check: X = {'OK' if ratio_x < 1 else 'NG'}, "
          f"Y = {'OK' if ratio_y < 1 else 'NG'}")

    # 5. Design check (member strength)
    design_check = None
    try:
        from core.design_check import run_design_check
        design_check = run_design_check(multi, model, seismic_rpt)
        if design_check:
            print()
            print(f"[*] Design check ran: status = {design_check.get('overall_status', '?')}")

            # Drift check section
            drift_chk = design_check.get("drift_check", {})
            if drift_chk:
                print(f"  Drift check: {drift_chk.get('overall_status', '?')}")

            # Member check section
            member_chk = design_check.get("member_check", {})
            if member_chk:
                summary = member_chk.get("summary", {})
                print(f"  Member check: {member_chk.get('overall_status', '?')}, "
                      f"{summary.get('num_design_members', '?')} design members, "
                      f"max ratio = {summary.get('max_interaction_ratio', '?')}")
                gov = member_chk.get("governing_member", {})
                if gov:
                    print(f"  Governing member: id={gov.get('member_id', '?')}, "
                          f"ratio={gov.get('interaction_ratio', '?')}, "
                          f"equation={gov.get('equation', '?')}")
    except Exception as exc:
        print(f"[!] Design check error: {exc}")

    # 6. Save full results
    out = {
        "_meta": {
            "input_file": str(INPUT.relative_to(ROOT)),
            "purpose": f"Section 4.2 re-analysis (V_base = {seismic_rpt.get('V_kN')} kN, "
                       f"{len(multi.combo_results)} combinations, KDS Kzr wind)",
        },
        "loads": {
            "V_base_kN": seismic_rpt.get("V_kN"),
            "W_kN": seismic_rpt.get("W_kN"),
            "Cs": seismic_rpt.get("Cs"),
            "Ta_sec": seismic_rpt.get("Ta_sec"),
            "Cd": Cd,
            "IE": IE,
        },
        "analysis": {
            "num_elements": multi.num_elements,
            "envelope": env,
            "design_drift_x_pct": round(design_drift_x * 100, 4),
            "design_drift_y_pct": round(design_drift_y * 100, 4),
            "design_drift_x_ratio": round(ratio_x, 4),
            "design_drift_y_ratio": round(ratio_y, 4),
            "drift_limit_pct": drift_limit * 100,
        },
        "design_check": design_check,
        "per_combo": per_combo,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str),
                      encoding="utf-8")
    print()
    print(f"[OK] Full results saved: {OUTPUT.relative_to(ROOT)}")

    # 7. §4.2 paragraph draft (for docx patching)
    print()
    print("=" * 72)
    print("DRAFT §4.2 paragraph (for docx update):")
    print("=" * 72)
    print(
        f"Under the governing seismic load combinations, the maximum roof displacement "
        f"is {env['max_dx_mm']:.1f} mm in the X-direction and {env['max_dy_mm']:.1f} mm "
        f"in the Y-direction. The maximum design drift demand, obtained by amplifying "
        f"the elastic inter-story drift using the deflection amplification factor "
        f"(Cd = {Cd}) and the importance factor (IE = {IE}) as described in Section 2.5.1, "
        f"occurs at Story {env['drift_y_story']}: "
        f"{design_drift_x*100:.2f}% in the X-direction (ratio {ratio_x:.3f}) and "
        f"{design_drift_y*100:.2f}% in the Y-direction (ratio {ratio_y:.3f}), both well "
        f"below the {drift_limit*100:.1f}% allowable limit for Importance Class II buildings "
        f"(KDS 41 17 00)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
