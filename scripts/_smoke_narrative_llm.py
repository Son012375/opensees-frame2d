"""결과 해설 LLM(역할 #3) 라이브 스모크 — ChatKHU 게이트웨이 / claude-opus-4-8.

전체 OpenSees/IFC 파이프라인 없이, 대표 해석결과(safe/moderate/severe) 3건을
게이트웨이로 보내 summary_ko/en 산문 + narration_meta(적용/폴백)를 확인한다.

사용:
    set NARRATOR_API_KEY=<ChatKHU 키>      (PowerShell: $env:NARRATOR_API_KEY="...")
    python scripts/_smoke_narrative_llm.py

선택 env: NARRATOR_MODEL(기본 claude-opus-4-8), NARRATOR_BASE_URL(기본 게이트웨이).
"""
import os
import sys
import json
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mcp-server"))

# .env 로드 (NARRATOR_API_KEY 등) — soft dep.
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except Exception:
    pass

from core.narrative_interpreter import build_facts, narrate_interpretation  # noqa: E402
from core.narrative_llm import make_narrator_from_env, _DEFAULT_MODEL, _DEFAULT_BASE_URL  # noqa: E402


# ============================================================
# 대표 케이스 (interpret_results 출력 형태 mock)
# ============================================================

def _safe():
    interp = {
        "severity": "safe",
        "severity_label": {"ko": "안전", "en": "Safe"},
        "drift_interpretation": {"max_ratio": 0.48, "critical_story": 3, "critical_direction": "X"},
        "member_interpretation": {"governing_check": "interaction", "weakest_link":
                                  {"member_id": 214, "type": "beam_x", "story": 3, "ratio": 0.62},
                                  "ng_by_story": {}, "ng_by_type": {}, "status": "OK"},
        "modal_interpretation": {"T1_actual_s": 1.05, "Ta_empirical_s": 1.10,
                                 "T1_Ta_ratio": 0.95, "first_mode_direction": "TRAN-X",
                                 "torsion_risk": False, "flexibility_flag": False},
        "diagnosis": None, "suggestions": [],
        "summary_ko": "[TEMPLATE] 모든 설계검토 통과. 최대 활용률 62%.",
        "summary_en": "[TEMPLATE] All design checks passed. Max utilization 62%.",
    }
    dc = {"summary": {"max_drift_ratio": 0.48, "max_interaction_ratio": 0.62,
                      "ng_stories": 0, "ng_members": 0}}
    return interp, dc


def _moderate():
    interp = {
        "severity": "moderate",
        "severity_label": {"ko": "보통 — 단면 보강 필요", "en": "Moderate — section upgrade needed"},
        "drift_interpretation": {"max_ratio": 0.92, "critical_story": 3, "critical_direction": "X"},
        "member_interpretation": {"governing_check": "interaction", "weakest_link":
                                  {"member_id": 214, "type": "beam_x", "story": 3, "ratio": 1.12},
                                  "ng_by_story": {3: 4}, "ng_by_type": {"beam_x": 4}, "status": "NG"},
        "modal_interpretation": None,
        "diagnosis": {"primary_cause": "member_capacity", "primary_cause_ko": "부재 내력 부족",
                      "primary_cause_en": "Insufficient member capacity", "contributing_factors_ko": []},
        "suggestions": [{"type": "section_upgrade", "target": "beam_x",
                         "current": "H-400x200", "recommended": "H-450x200"}],
        "summary_ko": "[TEMPLATE] 설계검토 부적합 (부재 NG 4개). 주 원인: 부재 내력 부족. 단면 보강으로 해결 가능.",
        "summary_en": "[TEMPLATE] Design check NG (4 members). Resolvable with section upgrade.",
    }
    dc = {"summary": {"max_drift_ratio": 0.92, "max_interaction_ratio": 1.12,
                      "ng_stories": 0, "ng_members": 4}}
    return interp, dc


def _severe():
    interp = {
        "severity": "severe",
        "severity_label": {"ko": "심각 — 구조시스템 변경 검토", "en": "Severe"},
        "drift_interpretation": {"max_ratio": 1.45, "critical_story": 1, "critical_direction": "X",
                                 "soft_story_detected": True, "soft_story_stories": [1]},
        "member_interpretation": {"governing_check": "interaction", "weakest_link":
                                  {"member_id": "C03", "type": "column", "story": 1, "ratio": 1.28},
                                  "ng_by_story": {1: 5, 2: 4}, "ng_by_type": {"column": 9}, "status": "NG"},
        "modal_interpretation": None,
        "diagnosis": {"primary_cause": "soft_story", "primary_cause_ko": "연약층 (Soft Story)",
                      "primary_cause_en": "Soft story", "contributing_factors_ko": ["횡강성 부족"]},
        "suggestions": [{"type": "system_change", "target": "structural_system",
                         "current": None, "recommended": None}],
        "summary_ko": "[TEMPLATE] 설계검토 부적합 (변위 NG 2층, 부재 NG 9개). 주 원인: 연약층. 구조시스템 변경 검토 필요.",
        "summary_en": "[TEMPLATE] Design check NG. Structural system change needed.",
    }
    dc = {"summary": {"max_drift_ratio": 1.45, "max_interaction_ratio": 1.28,
                      "ng_stories": 2, "ng_members": 9}}
    return interp, dc


CASES = [("safe", _safe), ("moderate", _moderate), ("severe", _severe)]


def main() -> int:
    print(f"base_url = {os.environ.get('NARRATOR_BASE_URL', _DEFAULT_BASE_URL)}")
    print(f"model    = {os.environ.get('NARRATOR_MODEL', _DEFAULT_MODEL)}\n")

    narrator = make_narrator_from_env()
    if narrator is None:
        print("[!] NARRATOR_API_KEY 미설정 → narrator 없음. 키를 설정하고 다시 실행하세요.")
        return 1

    # 1) 원시 호출 프로브 — 게이트웨이 오류(404/인증/경로)를 그대로 노출
    interp0, dc0 = _moderate()
    facts0 = build_facts(interp0, dc0)
    print("=== RAW PROBE (moderate facts) ===")
    print("facts:", json.dumps(facts0, ensure_ascii=False))
    try:
        raw = narrator(facts0)
        print("raw candidate:", json.dumps(raw, ensure_ascii=False))
    except Exception:
        print("[!] 게이트웨이 호출 예외 — 전체 트레이스백:")
        traceback.print_exc()
        print("\n경로(끝 슬래시)·인증(x-api-key vs Bearer)·모델 id를 확인하세요.")
        return 2
    print()

    # 2) 전체 경로 (검증+적용/폴백)
    for name, builder in CASES:
        interp, dc = builder()
        out = narrate_interpretation(interp, dc, llm=narrator)
        meta = out.get("narration_meta", {})
        print(f"=== [{name}] ===")
        print("  narration_meta:", json.dumps(meta, ensure_ascii=False))
        print("  summary_ko:", out.get("summary_ko"))
        print("  summary_en:", out.get("summary_en"))
        print()

    print("완료. fallback=false/reason=applied 이면 LLM 산문 적용됨.")
    print("'[TEMPLATE]'로 시작하면 폴백된 것 — meta.reason 확인.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
