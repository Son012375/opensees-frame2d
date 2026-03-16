"""Stage 5 회귀 테스트: 메타데이터 정확성 및 리포트 일관성 검증.

검증 항목:
  T1: analysis_metadata 존재 및 필드 정합성 (2D/3D)
  T2: 3D pdelta → actual_transf == "Corotational" (PDelta 아님)
  T3: 2D pdelta → actual_transf == "PDelta"
  T4: linear 기본값 → actual_transf == "Linear"
  T5: 3D visualization warnings에 W03 Corotational 경고 포함
  T6: 3D visualization model_settings에 정확한 분석 유형 표시
  T7: solver_algorithm/fallback 메타데이터 기록
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mcp-server'))

from core.frame_2d import analyze_frame_2d_multi
from core.frame_3d import analyze_frame_3d_multi

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
# T1: analysis_metadata 필드 존재 (2D & 3D)
# ============================================================
print("\n=== T1: analysis_metadata field presence ===")

r2d = analyze_frame_2d_multi(
    stories=[3.5], bays=[6.0],
    load_cases={"EQX": [{"type": "lateral", "story": 1, "value": 100}]},
    supports="fixed",
    geometric_nonlinearity="pdelta",
)
meta_2d = r2d.analysis_metadata
check(isinstance(meta_2d, dict) and len(meta_2d) > 0, "2D analysis_metadata exists and non-empty")
check("dimension" in meta_2d, "2D metadata has 'dimension' field")
check("requested_mode" in meta_2d, "2D metadata has 'requested_mode' field")
check("actual_transf" in meta_2d, "2D metadata has 'actual_transf' field")
check("solver_algorithm" in meta_2d, "2D metadata has 'solver_algorithm' field")
check("fallback_used" in meta_2d, "2D metadata has 'fallback_used' field")

r3d = analyze_frame_3d_multi(
    stories=[3.5], bays_x=[6.0], bays_y=[6.0],
    load_cases={"EQX": [{"type": "lateral_x", "story": 1, "value": 100}]},
    supports="fixed",
    geometric_nonlinearity="pdelta",
)
meta_3d = r3d.analysis_metadata
check(isinstance(meta_3d, dict) and len(meta_3d) > 0, "3D analysis_metadata exists and non-empty")

# ============================================================
# T2: 3D pdelta → Corotational (NOT PDelta)
# ============================================================
print("\n=== T2: 3D pdelta uses Corotational ===")
check(meta_3d["actual_transf"] == "Corotational",
      f"3D actual_transf == 'Corotational' (got '{meta_3d['actual_transf']}')")
check(meta_3d["requested_mode"] == "pdelta",
      f"3D requested_mode == 'pdelta' (got '{meta_3d['requested_mode']}')")
check(meta_3d["dimension"] == "3D",
      f"3D dimension == '3D' (got '{meta_3d['dimension']}')")

# ============================================================
# T3: 2D pdelta → PDelta
# ============================================================
print("\n=== T3: 2D pdelta uses PDelta ===")
check(meta_2d["actual_transf"] == "PDelta",
      f"2D actual_transf == 'PDelta' (got '{meta_2d['actual_transf']}')")
check(meta_2d["dimension"] == "2D",
      f"2D dimension == '2D' (got '{meta_2d['dimension']}')")

# ============================================================
# T4: linear 기본값 → Linear
# ============================================================
print("\n=== T4: linear default → Linear transf ===")
r3d_lin = analyze_frame_3d_multi(
    stories=[3.5], bays_x=[6.0], bays_y=[6.0],
    load_cases={"EQX": [{"type": "lateral_x", "story": 1, "value": 100}]},
    supports="fixed",
)
meta_lin = r3d_lin.analysis_metadata
check(meta_lin.get("actual_transf") == "Linear",
      f"3D linear actual_transf == 'Linear' (got '{meta_lin.get('actual_transf')}')")
check(meta_lin.get("requested_mode") == "linear",
      f"3D linear requested_mode == 'linear' (got '{meta_lin.get('requested_mode')}')")

# ============================================================
# T5: 3D visualization warnings W03 Corotational 경고
# ============================================================
print("\n=== T5: 3D report W03 warning for Corotational ===")
try:
    from core.visualization_3d import _get_model_warnings
    warnings = _get_model_warnings(r3d)
    w_codes = [w["code"] for w in warnings]
    w_texts = [w["text"] for w in warnings]

    check("W03" in w_codes,
          "W03 Corotational warning present in 3D pdelta warnings")
    w03_text = next((w["text"] for w in warnings if w["code"] == "W03"), "")
    check("Corotational" in w03_text,
          f"W03 mentions 'Corotational' in text")
    check("base reactions" in w03_text.lower(),
          "W03 mentions base reactions for verification")

    # linear 모드에서는 W03 없어야 함
    warnings_lin = _get_model_warnings(r3d_lin)
    w_codes_lin = [w["code"] for w in warnings_lin]
    check("W03" not in w_codes_lin,
          "W03 NOT present in linear mode (no false positive)")
except ImportError:
    check(False, "Could not import visualization_3d._get_model_warnings")

# ============================================================
# T6: 3D model_settings 정확한 분석 유형
# ============================================================
print("\n=== T6: 3D model_settings analysis type ===")
try:
    from core.visualization_3d import _get_model_settings
    settings_pd = _get_model_settings(r3d)
    settings_lin = _get_model_settings(r3d_lin)

    check("Corotational" in settings_pd["analysis_type"],
          f"pdelta analysis_type contains 'Corotational' (got '{settings_pd['analysis_type']}')")
    check("PDelta" not in settings_pd["analysis_type"],
          f"pdelta analysis_type does NOT claim pure 'PDelta' (got '{settings_pd['analysis_type']}')")
    check("Linear Static" == settings_lin["analysis_type"],
          f"linear analysis_type == 'Linear Static' (got '{settings_lin['analysis_type']}')")
    check("Enabled" in settings_pd["geometric_nonlinearity"],
          f"pdelta geometric_nonlinearity starts with 'Enabled' (got '{settings_pd['geometric_nonlinearity']}')")
    check("Disabled" in settings_lin["geometric_nonlinearity"],
          f"linear geometric_nonlinearity starts with 'Disabled' (got '{settings_lin['geometric_nonlinearity']}')")
except ImportError:
    check(False, "Could not import visualization_3d._get_model_settings")

# ============================================================
# T7: solver_algorithm 및 fallback 메타데이터
# ============================================================
print("\n=== T7: solver metadata ===")
check(meta_2d["solver_algorithm"] in ("Newton", "ModifiedNewton"),
      f"2D pdelta solver_algorithm is Newton variant (got '{meta_2d['solver_algorithm']}')")
check(isinstance(meta_2d["fallback_used"], bool),
      f"2D fallback_used is bool (got {type(meta_2d['fallback_used']).__name__})")
check(meta_2d["n_steps"] > 1,
      f"2D pdelta n_steps > 1 (got {meta_2d['n_steps']})")

check(meta_3d["solver_algorithm"] in ("Newton", "ModifiedNewton"),
      f"3D pdelta solver_algorithm is Newton variant (got '{meta_3d['solver_algorithm']}')")
check(isinstance(meta_3d["fallback_used"], bool),
      f"3D fallback_used is bool (got {type(meta_3d['fallback_used']).__name__})")

# linear 모드는 Linear 알고리즘
check(meta_lin.get("solver_algorithm") == "Linear",
      f"3D linear solver_algorithm == 'Linear' (got '{meta_lin.get('solver_algorithm')}')")

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*60}")
print(f"TOTAL: {passed + failed} checks, {passed} passed, {failed} failed")
if failed == 0:
    print("ALL STAGE 5 CHECKS PASSED")
else:
    print(f"FAILURES: {failed}")
    sys.exit(1)
