"""문서형 구조계산서 리포트 라이브 스모크.

실제 건물 파이프라인(BuildingModel → 하중 → 3D해석 → 설계검토 → 해설)을 돌려
plot_frame_3d_calc_report()로 문서형 HTML을 생성한다. 결과를 브라우저로 열어 확인.

실행:
    python scripts/_smoke_calc_report.py
출력:
    docs/report_mockups/_live_calc_report.html  (브라우저로 열기)

NARRATOR_API_KEY(.env)가 있으면 §10 종합결론에 Opus 산문이 들어간다.
"""
import os
import sys
import traceback

_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(_ROOT, "mcp-server"))

# .env (NARRATOR_API_KEY 등)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(os.path.join(_ROOT, ".env"))
except Exception:
    pass


CONFIG = {
    "stories": [
        {"height": 3.5, "usage": "office"},
        {"height": 3.5, "usage": "office"},
        {"height": 3.5, "usage": "office"},
        {"height": 3.5, "usage": "office"},
    ],
    "bays_x": [6.0, 6.0],
    "bays_y": [6.0, 6.0],
    # Stage 1B: 유효 지역 + 내진 입력 → 지진/풍하중 생성 (§1/§4 실측치)
    "region": "강남구",                       # hazard DB sigungu 키 (서울)
    "seismic_system": "ordinary_moment_frame",  # 철골 보통모멘트골조 (R=3.5)
    "importance": "II",                       # Ie=1.0
    "site_class": "S3",
    "exposure_category": "B",
}


def main() -> int:
    from core.building_model import BuildingModel
    from core.load_generator import generate_all_loads
    from core.frame_3d import analyze_frame_3d_multi
    from core.design_check import run_design_check
    from core.result_interpreter import interpret_results
    from core.narrative_interpreter import narrate_interpretation
    from core.narrative_llm import make_narrator_from_env
    from core.visualization_calc_report import plot_frame_3d_calc_report

    print("1) BuildingModel + 하중 생성 ...")
    model = BuildingModel.from_json(CONFIG)
    load_result = generate_all_loads(model)

    print("2) 3D 해석 (modal 포함) ...")
    kwargs = model.to_frame3d_kwargs()
    kwargs["load_cases"] = load_result["load_cases"]
    kwargs["load_combinations"] = load_result["load_combinations"]
    kwargs["modal_analysis"] = True
    multi = analyze_frame_3d_multi(**kwargs)

    print("3) 설계검토 + 해석 ...")
    dc_result = None
    interpretation = None
    try:
        seismic_rpt = load_result["reports"].get("seismic")
        dc_result = run_design_check(multi, model, seismic_rpt)
        interpretation = interpret_results(
            dc_result, multi, modal_analysis=multi.modal_analysis or None,
        )
        narrator = make_narrator_from_env()
        print(f"   narration: {'ON (게이트웨이)' if narrator else 'OFF (키 없음 → 템플릿)'}")
        interpretation = narrate_interpretation(interpretation, dc_result, llm=narrator)
    except Exception:
        print("   [!] 설계검토/해설 단계 예외:")
        traceback.print_exc()

    print("4) 문서형 구조계산서 리포트 생성 ...")
    out = os.path.abspath(os.path.join(_ROOT, "docs", "report_mockups", "_live_calc_report.html"))
    path = plot_frame_3d_calc_report(
        multi, output_path=out,
        design_check=dc_result, interpretation=interpretation,
        load_result=load_result,
    )
    print("\n완료. 브라우저로 열어 확인:")
    print("  " + path)
    if interpretation is not None:
        meta = interpretation.get("narration_meta", {})
        print(f"  narration_meta: {meta}")
        print(f"  severity: {interpretation.get('severity')}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        print("\n[!] 스모크 실패 — 트레이스백:")
        traceback.print_exc()
        raise SystemExit(1)
