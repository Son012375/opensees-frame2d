"""OpenSees-MCP 사용성 테스트 실행기.

11개 테스트 케이스를 순차 실행하고 결과를 JSON + Markdown 리포트로 출력.
"""
import json
import sys
import os
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mcp-server'))

from core.building_model import BuildingModel
from core.load_generator import generate_all_loads
from core.frame_3d import analyze_frame_3d_multi
from core.design_check import run_design_check
from core.result_interpreter import interpret_results
from core.assumption_tracker import build_assumption_summary
from core.visualization_3d import plot_frame_3d_interactive


def run_test(test_id: str, description: str, config: dict, expect: dict) -> dict:
    """단일 테스트 실행, 결과 반환."""
    result = {
        "id": test_id,
        "description": description,
        "config_input": config,
        "expect": expect,
        "status": "FAIL",
        "error": None,
        "timings": {},
    }

    try:
        # 1. Build model
        t0 = time.time()
        model = BuildingModel.from_json(config)
        result["timings"]["model_build"] = round(time.time() - t0, 3)

        result["model_summary"] = model.summary()

        # 2. Load generation
        t0 = time.time()
        load_result = generate_all_loads(model)
        result["timings"]["load_gen"] = round(time.time() - t0, 3)

        result["load_cases"] = list(load_result["load_cases"].keys())
        result["load_combos"] = list(load_result["load_combinations"].keys())
        result["num_load_cases"] = len(result["load_cases"])
        result["num_combos"] = len(result["load_combos"])

        # 3. Analysis
        t0 = time.time()
        kwargs = model.to_frame3d_kwargs()
        kwargs["load_cases"] = load_result["load_cases"]
        kwargs["load_combinations"] = load_result["load_combinations"]
        kwargs["modal_analysis"] = True
        multi = analyze_frame_3d_multi(**kwargs)
        result["timings"]["analysis"] = round(time.time() - t0, 3)

        result["num_nodes"] = len(multi.nodes) if multi.nodes else 0
        result["num_elements"] = multi.num_elements

        # 4. Modal analysis
        if multi.modal_analysis:
            ma = multi.modal_analysis
            fp = ma.get("fundamental_periods", {})
            result["modal"] = {
                "num_modes": ma.get("num_modes"),
                "T1_x_s": round(fp.get("T1_x_s", 0), 4),
                "T1_y_s": round(fp.get("T1_y_s", 0), 4),
                "T1_rz_s": round(fp.get("T1_rz_s", 0), 4),
                "modes": [
                    {"mode": m["mode"], "period_s": round(m["period_s"], 4), "direction": m["direction"]}
                    for m in ma.get("modes", [])[:5]
                ],
            }
        else:
            result["modal"] = None

        # 5. Design check
        t0 = time.time()
        seismic_rpt = load_result["reports"].get("seismic")
        dc_result = run_design_check(multi, model, seismic_rpt)
        result["timings"]["design_check"] = round(time.time() - t0, 3)

        result["design_check"] = {
            "overall_status": dc_result["overall_status"],
            "max_drift_ratio": dc_result["summary"]["max_drift_ratio"],
            "max_interaction_ratio": dc_result["summary"]["max_interaction_ratio"],
            "ng_stories": dc_result["summary"].get("ng_stories", 0),
            "ng_members": dc_result["summary"].get("ng_members", 0),
        }

        # Top critical members
        cm = dc_result.get("member_check", {}).get("critical_members", [])
        result["critical_members_top3"] = [
            {"id": m["member_id"], "type": m["type"], "ratio": round(m["ratios"]["interaction"], 3), "status": m["status"]}
            for m in cm[:3]
        ]

        # 6. Interpretation
        t0 = time.time()
        interp = interpret_results(dc_result, multi, modal_analysis=multi.modal_analysis)
        result["timings"]["interpretation"] = round(time.time() - t0, 3)

        result["interpretation"] = {
            "severity": interp["severity"],
            "severity_label_ko": interp["severity_label"]["ko"],
            "summary_ko": interp["summary_ko"],
            "summary_en": interp["summary_en"],
            "findings_count": len(interp["findings"]),
            "findings": [
                {"code": f["code"], "severity": f["severity"], "message_en": f["message_en"]}
                for f in interp["findings"]
            ],
            "diagnosis": interp.get("diagnosis"),
            "suggestions": [
                {"type": s["type"], "target": s.get("target"), "current": s.get("current"),
                 "recommended": s.get("recommended"), "impact": s["expected_impact"],
                 "message_en": s["message_en"]}
                for s in interp.get("suggestions", [])
            ],
        }
        if interp.get("drift_interpretation"):
            di = interp["drift_interpretation"]
            result["interpretation"]["drift_pattern"] = di.get("pattern")
            result["interpretation"]["critical_story"] = di.get("critical_story")
            result["interpretation"]["critical_direction"] = di.get("critical_direction")
        if interp.get("modal_interpretation"):
            mi = interp["modal_interpretation"]
            result["interpretation"]["T1_Ta_ratio"] = mi.get("T1_Ta_ratio")
            result["interpretation"]["flexibility_flag"] = mi.get("flexibility_flag")
            result["interpretation"]["torsion_risk"] = mi.get("torsion_risk")
            result["interpretation"]["first_mode"] = mi.get("first_mode_direction")

        # 7. HTML Report
        t0 = time.time()
        html_path = plot_frame_3d_interactive(
            multi, assumptions=build_assumption_summary(model, config, load_result["summary"]),
            design_check=dc_result, interpretation=interp,
        )
        result["timings"]["html_report"] = round(time.time() - t0, 3)
        result["html_report_path"] = html_path

        # Total time
        result["timings"]["total"] = round(sum(result["timings"].values()), 3)

        # Verify expectations
        result["status"] = "PASS"
        result["verification"] = {}

        if "expect_status" in expect:
            ok = result["design_check"]["overall_status"] == expect["expect_status"]
            result["verification"]["status_match"] = ok
            if not ok:
                result["status"] = "CHECK"

        if "expect_severity" in expect:
            ok = result["interpretation"]["severity"] == expect["expect_severity"]
            result["verification"]["severity_match"] = ok
            if not ok:
                result["status"] = "CHECK"

        if "expect_has_modal" in expect:
            ok = (result["modal"] is not None) == expect["expect_has_modal"]
            result["verification"]["modal_exists"] = ok
            if not ok:
                result["status"] = "CHECK"

    except Exception as e:
        result["status"] = "ERROR"
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()

    return result


# ─── Test Case Definitions ───

TESTS = [
    # E2E 정상
    {
        "id": "TC-01",
        "desc": "5층 오피스 (서울 강남, 기본 단면)",
        "config": {
            "stories": [{"height": 4, "usage": "office"}, {"height": 3.5, "usage": "office"},
                         {"height": 3.5, "usage": "office"}, {"height": 3.5, "usage": "office"},
                         {"height": 3.5, "usage": "office"}],
            "bays_x": [8, 8], "bays_y": [8, 8], "region": "강남구",
        },
        "expect": {"expect_has_modal": True},
    },
    {
        "id": "TC-02",
        "desc": "10층 주상복합 (부산 해운대, 1F 근생/2-9F 오피스/10F 기계실)",
        "config": {
            "stories": [{"height": 4.5, "usage": "retail"},
                         *[{"height": 3.5, "usage": "office"} for _ in range(8)],
                         {"height": 3, "usage": "mechanical_room", "dead_load_finish": 2.0}],
            "bays_x": [8, 8, 8], "bays_y": [8, 8], "region": "해운대구",
        },
        "expect": {"expect_has_modal": True},
    },
    {
        "id": "TC-03",
        "desc": "3층 주거 (대전 서구, 소규모)",
        "config": {
            "stories": [{"height": 3, "usage": "residential"}, {"height": 3, "usage": "residential"},
                         {"height": 3, "usage": "residential"}],
            "bays_x": [6, 6], "bays_y": [6], "region": "서구",
        },
        "expect": {"expect_has_modal": True},
    },
    # Robustness
    {
        "id": "TC-04",
        "desc": "1층 단일 건물 (최소 규모)",
        "config": {
            "stories": [{"height": 4, "usage": "retail"}],
            "bays_x": [8], "bays_y": [6], "region": "강남구",
        },
        "expect": {"expect_has_modal": True},
    },
    {
        "id": "TC-05",
        "desc": "지역 미지정 (지진/풍하중 없이)",
        "config": {
            "stories": [{"height": 4, "usage": "office"}, {"height": 3.5, "usage": "office"},
                         {"height": 3.5, "usage": "office"}],
            "bays_x": [8, 8], "bays_y": [8],
        },
        "expect": {"expect_has_modal": True},
    },
    {
        "id": "TC-06",
        "desc": "대경간+중하중 (창고, 12m경간)",
        "config": {
            "stories": [{"height": 6, "usage": "storage"}, {"height": 5, "usage": "storage"}],
            "bays_x": [12, 12], "bays_y": [12], "region": "강남구",
        },
        "expect": {"expect_has_modal": True},
    },
    {
        "id": "TC-07",
        "desc": "비표준 용도 (gymnasium → gym 매핑 확인)",
        "config": {
            "stories": [{"height": 5, "usage": "gym"}, {"height": 3.5, "usage": "office"}],
            "bays_x": [10, 10], "bays_y": [8], "region": "강남구",
        },
        "expect": {"expect_has_modal": True},
    },
    {
        "id": "TC-08",
        "desc": "극소 단면 + 15층 (H-200x200 기둥 → NG 예상)",
        "config": {
            "stories": [{"height": 4, "usage": "office"}]
                       + [{"height": 3.5, "usage": "office"} for _ in range(14)],
            "bays_x": [8, 8], "bays_y": [8, 8], "region": "강남구",
            "column_section": "H-200x200",
            "beam_x_section": "H-200x100",
            "beam_y_section": "H-200x100",
        },
        "expect": {"expect_status": "NG", "expect_has_modal": True},
    },
    # Sensitivity
    {
        "id": "TC-09a",
        "desc": "[Sensitivity 단면] 5층 오피스 - 소단면 H-200",
        "config": {
            "stories": [{"height": 4, "usage": "office"}]
                       + [{"height": 3.5, "usage": "office"} for _ in range(4)],
            "bays_x": [8, 8], "bays_y": [8, 8], "region": "강남구",
            "column_section": "H-200x200",
            "beam_x_section": "H-200x100",
            "beam_y_section": "H-200x100",
        },
        "expect": {"expect_has_modal": True},
    },
    {
        "id": "TC-09b",
        "desc": "[Sensitivity 단면] 5층 오피스 - 대단면 H-400",
        "config": {
            "stories": [{"height": 4, "usage": "office"}]
                       + [{"height": 3.5, "usage": "office"} for _ in range(4)],
            "bays_x": [8, 8], "bays_y": [8, 8], "region": "강남구",
            "column_section": "H-400x400",
            "beam_x_section": "H-400x200",
            "beam_y_section": "H-400x200",
        },
        "expect": {"expect_has_modal": True},
    },
    {
        "id": "TC-10a",
        "desc": "[Sensitivity 지역] 5층 오피스 - 서울 (zone I)",
        "config": {
            "stories": [{"height": 4, "usage": "office"}]
                       + [{"height": 3.5, "usage": "office"} for _ in range(4)],
            "bays_x": [8, 8], "bays_y": [8, 8], "region": "강남구",
        },
        "expect": {"expect_has_modal": True},
    },
    {
        "id": "TC-10b",
        "desc": "[Sensitivity 지역] 5층 오피스 - 울산 (zone I, 높은 PGA)",
        "config": {
            "stories": [{"height": 4, "usage": "office"}]
                       + [{"height": 3.5, "usage": "office"} for _ in range(4)],
            "bays_x": [8, 8], "bays_y": [8, 8], "region": "울주군",
        },
        "expect": {"expect_has_modal": True},
    },
    {
        "id": "TC-11a",
        "desc": "[Sensitivity 층고] 5층 오피스 - 표준 층고 3.5m",
        "config": {
            "stories": [{"height": 3.5, "usage": "office"} for _ in range(5)],
            "bays_x": [8, 8], "bays_y": [8, 8], "region": "강남구",
        },
        "expect": {"expect_has_modal": True},
    },
    {
        "id": "TC-11b",
        "desc": "[Sensitivity 층고] 5층 오피스 - 고층고 5.0m",
        "config": {
            "stories": [{"height": 5, "usage": "office"} for _ in range(5)],
            "bays_x": [8, 8], "bays_y": [8, 8], "region": "강남구",
        },
        "expect": {"expect_has_modal": True},
    },
]


def main():
    results = []
    for t in TESTS:
        print(f"\n{'='*60}")
        print(f"Running {t['id']}: {t['desc']}")
        print(f"{'='*60}")
        r = run_test(t["id"], t["desc"], t["config"], t["expect"])
        results.append(r)

        # Print summary
        if r["status"] == "ERROR":
            print(f"  STATUS: ERROR - {r['error']}")
        else:
            dc = r.get("design_check", {})
            interp = r.get("interpretation", {})
            modal = r.get("modal", {})
            print(f"  STATUS: {r['status']}")
            dr = dc.get('max_drift_ratio') or 0
            ir = dc.get('max_interaction_ratio') or 0
            print(f"  Design: {dc.get('overall_status','N/A')} | drift={dr:.3f} | interact={ir:.3f} | NG={dc.get('ng_members',0)}")
            print(f"  Severity: {interp.get('severity','N/A')} | findings={interp.get('findings_count',0)} | suggestions={len(interp.get('suggestions',[]))}")
            if modal:
                print(f"  Modal: T1_x={modal.get('T1_x_s',0):.3f}s T1_y={modal.get('T1_y_s',0):.3f}s ({modal.get('num_modes',0)} modes)")
            print(f"  Nodes={r.get('num_nodes',0)} Elements={r.get('num_elements',0)}")
            print(f"  Time: {r.get('timings',{}).get('total',0):.1f}s")

    # Save JSON
    out_path = os.path.join(os.path.dirname(__file__), "usability_test_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n\nResults saved to {out_path}")

    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'ID':<8} {'Status':<7} {'DC':<4} {'Drift':>7} {'Inter':>7} {'Sev':<10} {'Find':>5} {'Sugg':>5} {'Time':>6}")
    print("-" * 80)
    for r in results:
        if r["status"] == "ERROR":
            print(f"{r['id']:<8} {'ERROR':<7} {r['error'][:50]}")
        else:
            dc = r.get("design_check", {})
            interp = r.get("interpretation", {})
            dr = dc.get('max_drift_ratio') or 0
            ir = dc.get('max_interaction_ratio') or 0
            print(f"{r['id']:<8} {r['status']:<7} {dc.get('overall_status','N/A'):<4} "
                  f"{dr:>7.3f} {ir:>7.3f} "
                  f"{interp.get('severity','N/A'):<10} {interp.get('findings_count',0):>5} "
                  f"{len(interp.get('suggestions',[])):>5} {r.get('timings',{}).get('total',0):>5.1f}s")

    # Pass/Fail count
    passed = sum(1 for r in results if r["status"] == "PASS")
    checked = sum(1 for r in results if r["status"] == "CHECK")
    errors = sum(1 for r in results if r["status"] == "ERROR")
    print(f"\nTotal: {len(results)} | PASS: {passed} | CHECK: {checked} | ERROR: {errors}")


if __name__ == "__main__":
    main()
