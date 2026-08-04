"""KDS 하중 자동생성값 추출 스크립트 (Paper 1, Table A1 검증용).

example_input.json의 빌딩에 대해 generate_all_loads()를 실행하고,
DL/LL/EQ/Wind의 모든 중간 계산값(SDS, SD1, Fa, Fv, Ta, Cs, V, Kz, qz, p, story forces)을
example_auto_values.json으로 떨군다.

손계산표(Table A1)의 "Auto-gen" 열을 채우기 위한 reproducible artifact.

Usage:
    python scripts/_hand_check_loads.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# mcp-server를 import path에 추가
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mcp-server"))

from core.building_model import BuildingModel
from core.load_generator import (
    generate_all_loads,
    _calculate_story_weights,
)


INPUT_PATH = ROOT / "docs" / "paper1_open_source_alternative" / "validation" / "example_input.json"
OUTPUT_PATH = ROOT / "docs" / "paper1_open_source_alternative" / "validation" / "example_auto_values.json"


def main() -> int:
    print(f"[*] Reading input: {INPUT_PATH.relative_to(ROOT)}")
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    # _meta 키 제거 (BuildingModel.from_json은 이를 무시하지만 명시적으로 제거)
    config.pop("_meta", None)

    model = BuildingModel.from_json(config)

    print(f"[*] Model built: {model.num_stories} stories, "
          f"{len(model.bays_x)}x{len(model.bays_y)} bays, "
          f"H={model.total_height} m, A={model.floor_area} m^2")

    result = generate_all_loads(model)

    # 층별 무게 직접 추출 (V_base 추적용)
    dl_loads = result["load_cases"].get("DL", [])
    story_weights_kN = _calculate_story_weights(model, dl_loads)
    floor_area_per_story = [model.floor_area_at_story(i + 1) for i in range(model.num_stories)]

    # 표준화된 출력 구조
    out = {
        "_meta": {
            "input_file": str(INPUT_PATH.relative_to(ROOT)),
            "purpose": "Auto-generated load values for Paper 1 Table A1 hand-check",
            "generator": "scripts/_hand_check_loads.py",
        },
        "geometry": {
            "num_stories": model.num_stories,
            "story_heights_m": model.story_heights_list,
            "cumulative_heights_m": model.cumulative_heights,
            "total_height_m": model.total_height,
            "bays_x_m": model.bays_x,
            "bays_y_m": model.bays_y,
            "total_width_x_m": model.total_width_x,
            "total_width_y_m": model.total_width_y,
            "floor_area_per_story_m2": floor_area_per_story,
        },
        "dead_load": {
            "rc_unit_weight_kN_m3": 24.0,
            "mep_allowance_kN_m2": 0.5,
            "per_story": result["reports"]["gravity"],
        },
        "live_load": {
            "per_story": [
                {"story": ld["story"], "value_kN_m2": ld["value"], "usage": s.usage}
                for ld, s in zip(result["load_cases"]["LL"], model.stories)
            ],
        },
        "story_weights": {
            "method": "W_i = DL_i x floor_area_i (LL excluded for general buildings)",
            "per_story_kN": [
                {
                    "story": i + 1,
                    "DL_kN_m2": dl_loads[i]["value"],
                    "floor_area_m2": floor_area_per_story[i],
                    "W_kN": story_weights_kN[i],
                }
                for i in range(model.num_stories)
            ],
            "total_W_kN": sum(story_weights_kN),
        },
        "seismic": result["reports"].get("seismic", {}),
        "wind": result["reports"].get("wind", {}),
        "load_combinations": {
            "count": len(result["load_combinations"]),
            "combinations": result["load_combinations"],
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[OK] Auto-gen values written: {OUTPUT_PATH.relative_to(ROOT)}")

    # 핵심 숫자를 콘솔에 요약
    print()
    print("=" * 70)
    print("KEY AUTO-GENERATED VALUES (for Table A1 'Auto-gen' column)")
    print("=" * 70)

    print("\n[Dead Load]")
    for r in result["reports"]["gravity"]:
        bd = r["DL_breakdown"]
        print(f"  Story {r['story']} ({r['usage']:>10s}): "
              f"slab={bd['slab_self']:.2f} + finish={bd['finish']:.2f} + mep={bd['mep']:.2f} "
              f"= DL={r['DL_kNm2']:.2f} kN/m^2,  LL={r['LL_kNm2']:.2f} kN/m^2")

    print("\n[Story Weights]")
    for sw in out["story_weights"]["per_story_kN"]:
        print(f"  Story {sw['story']}: W = {sw['DL_kN_m2']:.2f} kN/m^2 x "
              f"{sw['floor_area_m2']:.1f} m^2 = {sw['W_kN']:.2f} kN")
    print(f"  Total W = {out['story_weights']['total_W_kN']:.2f} kN")

    seismic = result["reports"].get("seismic", {})
    if seismic and "error" not in seismic:
        print("\n[Seismic]")
        print(f"  z = {seismic['z']},  Fa = {seismic['Fa']},  Fv = {seismic['Fv']}")
        print(f"  SDS = {seismic['SDS']},  SD1 = {seismic['SD1']}")
        print(f"  R = {seismic['R']},  Cd = {seismic['Cd']},  IE = {seismic['IE']}")
        print(f"  Ct = {seismic['Ct']},  x = {seismic['x']},  hn = {seismic['hn_m']} m")
        print(f"  Ta = {seismic['Ta_sec']} s,  Cu = {seismic['Cu']},  T = {seismic['T_sec']} s")
        print(f"  Cs = {seismic['Cs']}  (min={seismic['Cs_min']}, max={seismic['Cs_max']})")
        print(f"  k  = {seismic['k']}")
        print(f"  W  = {seismic['W_kN']} kN,  V_base = {seismic['V_kN']} kN")
        print(f"  Story forces:")
        for sf in seismic["story_forces"]:
            print(f"    Story {sf['story']}: w={sf['weight_kN']:.1f} kN, "
                  f"h={sf['height_m']:.2f} m, Cvx={sf['Cvx']:.4f}, Fx={sf['Fx_kN']:.2f} kN")
    else:
        print(f"\n[Seismic] error: {seismic.get('error', 'unknown')}")

    wind = result["reports"].get("wind", {})
    if wind and "error" not in wind:
        print("\n[Wind]")
        print(f"  V0 = {wind['V0_ms']} m/s,  exposure = {wind['exposure']}")
        print(f"  Gf = {wind['Gf']},  Cp_total = {wind['Cp_total']}")
        print(f"  Bx = {wind['Bx_m']} m,  By = {wind['By_m']} m")
        print(f"  Total Fx = {wind['total_Fx_kN']} kN,  Total Fy = {wind['total_Fy_kN']} kN")
        print(f"  Story detail:")
        for sd in wind["story_detail"]:
            print(f"    Story {sd['story']}: h_mid={sd['h_mid_m']:.2f} m, "
                  f"Kz={sd['Kz']:.4f}, qz={sd['qz_kNm2']:.4f} kN/m^2, "
                  f"p={sd['p_kNm2']:.4f}, Fx={sd['Fx_kN']:.2f}, Fy={sd['Fy_kN']:.2f}")
    else:
        print(f"\n[Wind] error: {wind.get('error', 'unknown')}")

    print("\n[Load Combinations]")
    print(f"  Count = {len(result['load_combinations'])}")
    for name in list(result["load_combinations"].keys()):
        print(f"    - {name}")

    print()
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
