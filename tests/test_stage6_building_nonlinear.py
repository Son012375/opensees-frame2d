"""Stage 6 회귀 테스트: 건물 스케일 비선형 통합 + 전역 응답 검증.

검증 항목:
  T1: BuildingModel config → geometric_nonlinearity 전달
  T2: 파이프라인 E2E (config → analyze_frame_3d_multi → metadata/warnings)
  T3: nonlinear_summary 조건부 생성 (pdelta 시만)
  T4: 3D Corotational 용어 정확성 (PDelta 허위 표기 없음)
  T5: linear 모드에서 비선형 텍스트 미출현
  T6: validate_3d_global_response 헬퍼 동작
  T7: force_interpretation policy 정확성
  T8: analysis_metadata building 응답 포함 확인
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mcp-server'))

from core.frame_3d import analyze_frame_3d_multi
from core.building_model import BuildingModel

passed = 0
failed = 0


def check(cond, msg):
    global passed, failed
    if cond:
        passed += 1
        print(f"  [PASS] {msg}")
    else:
        failed += 1
        print(f"  [FAIL] {msg}")


# ============================================================
# 공용 BuildingModel 설정
# ============================================================
config_pdelta = {
    "stories": [
        {"height": 3.5, "usage": "office"},
        {"height": 3.5, "usage": "office"},
        {"height": 3.5, "usage": "office"},
    ],
    "bays_x": [6.0],
    "bays_y": [6.0],
    "geometric_nonlinearity": "pdelta",
}

config_linear = {
    "stories": [
        {"height": 3.5, "usage": "office"},
        {"height": 3.5, "usage": "office"},
        {"height": 3.5, "usage": "office"},
    ],
    "bays_x": [6.0],
    "bays_y": [6.0],
}


# ============================================================
# T1: BuildingModel config → geometric_nonlinearity 전달
# ============================================================
print("\n=== T1: BuildingModel config propagation ===")

m_pd = BuildingModel.from_json(config_pdelta)
m_lin = BuildingModel.from_json(config_linear)

check(m_pd.geometric_nonlinearity == "pdelta",
      f"pdelta config → model.geometric_nonlinearity == 'pdelta' (got '{m_pd.geometric_nonlinearity}')")
check(m_lin.geometric_nonlinearity == "linear",
      f"default config → model.geometric_nonlinearity == 'linear' (got '{m_lin.geometric_nonlinearity}')")

kwargs_pd = m_pd.to_frame3d_kwargs()
kwargs_lin = m_lin.to_frame3d_kwargs()
check(kwargs_pd["geometric_nonlinearity"] == "pdelta",
      "to_frame3d_kwargs passes 'pdelta'")
check(kwargs_lin["geometric_nonlinearity"] == "linear",
      "to_frame3d_kwargs passes 'linear'")


# ============================================================
# T2: E2E 파이프라인 (config → analyze → metadata/warnings)
# ============================================================
print("\n=== T2: E2E pipeline integration ===")

load_cases = {
    "DL": [{"type": "floor_area", "story": s, "value": 5} for s in range(1, 4)],
    "EQX": [{"type": "lateral_x", "story": s, "value": 30 * s} for s in range(1, 4)],
}

r_pd = analyze_frame_3d_multi(
    **kwargs_pd, load_cases=load_cases,
    load_combinations={"1.0DL+1.0EQX": {"DL": 1.0, "EQX": 1.0}},
)

r_lin = analyze_frame_3d_multi(
    **kwargs_lin, load_cases=load_cases,
    load_combinations={"1.0DL+1.0EQX": {"DL": 1.0, "EQX": 1.0}},
)

check(r_pd.geometric_nonlinearity == "pdelta",
      "E2E result.geometric_nonlinearity == 'pdelta'")
check(r_lin.geometric_nonlinearity == "linear",
      "E2E result.geometric_nonlinearity == 'linear'")

meta_pd = r_pd.analysis_metadata
meta_lin = r_lin.analysis_metadata
check(meta_pd.get("actual_transf") == "Corotational",
      f"E2E pdelta actual_transf == 'Corotational' (got '{meta_pd.get('actual_transf')}')")
check(meta_lin.get("actual_transf") == "Linear",
      f"E2E linear actual_transf == 'Linear' (got '{meta_lin.get('actual_transf')}')")


# ============================================================
# T3: nonlinear_summary 조건부 생성
# ============================================================
print("\n=== T3: nonlinear_summary conditional ===")

from core.visualization_3d import _build_report_summary

summary_pd = _build_report_summary(r_pd)
summary_lin = _build_report_summary(r_lin)

check("nonlinear_summary" in summary_pd,
      "nonlinear_summary present when pdelta enabled")
check("nonlinear_summary" not in summary_lin,
      "nonlinear_summary absent when linear mode")

ns = summary_pd.get("nonlinear_summary", {})
check(ns.get("requested_mode") == "pdelta",
      f"nonlinear_summary.requested_mode == 'pdelta' (got '{ns.get('requested_mode')}')")
check(ns.get("transformation") == "Corotational",
      f"nonlinear_summary.transformation == 'Corotational' (got '{ns.get('transformation')}')")


# ============================================================
# T4: Corotational 용어 정확성 (PDelta 허위 표기 없음)
# ============================================================
print("\n=== T4: Corotational terminology correctness ===")

from core.visualization_3d import _get_model_warnings, _get_model_settings

warnings_pd = _get_model_warnings(r_pd)
settings_pd = _get_model_settings(r_pd)

# W01에서 Corotational 언급
w01_text = next((w["text"] for w in warnings_pd if w["code"] == "W01"), "")
check("Corotational" in w01_text,
      f"W01 mentions Corotational (got '{w01_text}')")
check("PDelta" not in settings_pd["analysis_type"],
      f"analysis_type does not claim pure PDelta (got '{settings_pd['analysis_type']}')")

# W03 경고 존재
w_codes = [w["code"] for w in warnings_pd]
check("W03" in w_codes,
      "W03 Corotational force interpretation warning present")


# ============================================================
# T5: linear 모드에서 비선형 텍스트 미출현
# ============================================================
print("\n=== T5: No nonlinear text in linear mode ===")

warnings_lin = _get_model_warnings(r_lin)
settings_lin = _get_model_settings(r_lin)

w_codes_lin = [w["code"] for w in warnings_lin]
check("W03" not in w_codes_lin,
      "W03 absent in linear mode")
check("W04" not in w_codes_lin,
      "W04 absent in linear mode")
check(settings_lin["analysis_type"] == "Linear Static",
      f"linear analysis_type == 'Linear Static' (got '{settings_lin['analysis_type']}')")
check("Disabled" in settings_lin["geometric_nonlinearity"],
      f"linear geo_nl starts with 'Disabled' (got '{settings_lin['geometric_nonlinearity']}')")


# ============================================================
# T6: validate_3d_global_response 헬퍼 동작
# ============================================================
print("\n=== T6: validate_3d_global_response ===")

from core.visualization_3d import validate_3d_global_response

gr_pd = validate_3d_global_response(r_pd)
gr_lin = validate_3d_global_response(r_lin)

# max_displacement 존재 및 합리적 값
check("max_displacement" in gr_pd, "global_response has max_displacement")
check(abs(gr_pd["max_displacement"]["x_mm"]) > 0,
      f"pdelta max_disp_x > 0 (got {gr_pd['max_displacement']['x_mm']})")

# story_drift_summary 존재 및 층 수 일치
check(len(gr_pd["story_drift_summary"]) == 3,
      f"story_drift_summary has 3 stories (got {len(gr_pd['story_drift_summary'])})")

# base_reaction_summary 존재
check("EQX" in gr_pd["base_reaction_summary"],
      "base_reaction_summary contains EQX case")
eq_reactions = gr_pd["base_reaction_summary"]["EQX"]
total_lateral = sum(30 * s for s in range(1, 4))  # 30+60+90 = 180 kN
check(abs(eq_reactions["sum_RX_kN"] + total_lateral) < 0.2,
      f"EQX base reaction X ≈ -{total_lateral} kN (got {eq_reactions['sum_RX_kN']})")

# overturning_check 존재
check(len(gr_pd["overturning_check"]) > 0,
      "overturning_check has at least one entry")
ot_eqx = next((c for c in gr_pd["overturning_check"] if c["case"] == "EQX"), None)
check(ot_eqx is not None and ot_eqx["status"] == "OK",
      f"EQX overturning check status == 'OK' (got {ot_eqx['status'] if ot_eqx else 'None'})")

# global_response는 summary 안에도 포함됨
check("global_response" in summary_pd,
      "global_response included in _build_report_summary")


# ============================================================
# T7: force_interpretation policy
# ============================================================
print("\n=== T7: force_interpretation policy ===")

fi = ns.get("force_interpretation", {})
check(fi.get("displacement_drift") == "reliable",
      f"displacement_drift == 'reliable' (got '{fi.get('displacement_drift')}')")
check(fi.get("base_reactions") == "reliable",
      f"base_reactions == 'reliable' (got '{fi.get('base_reactions')}')")
check("caution" in fi.get("local_element_forces", ""),
      f"local_element_forces contains 'caution' (got '{fi.get('local_element_forces')}')")


# ============================================================
# T8: analysis_metadata → building 응답 포함 확인
# ============================================================
print("\n=== T8: analysis_metadata in building response ===")

# BuildingModel 파이프라인 결과에 metadata가 존재하는지 확인
check(meta_pd.get("dimension") == "3D",
      f"building pipeline dimension == '3D' (got '{meta_pd.get('dimension')}')")
check(meta_pd.get("solver_algorithm") in ("Newton", "ModifiedNewton"),
      f"building pipeline solver is Newton variant (got '{meta_pd.get('solver_algorithm')}')")
check(isinstance(meta_pd.get("fallback_used"), bool),
      "building pipeline fallback_used is bool")


# ============================================================
# Summary
# ============================================================
print(f"\n{'='*60}")
print(f"TOTAL: {passed + failed} checks, {passed} passed, {failed} failed")
if failed == 0:
    print("ALL STAGE 6 CHECKS PASSED")
else:
    print(f"FAILURES: {failed}")
    sys.exit(1)
