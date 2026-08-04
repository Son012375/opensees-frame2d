"""결과 해설 LLM (역할 #3) — Narrative Interpreter.

규칙기반 해석기(result_interpreter.interpret_results)와 결정론적 설계검토
(design_check)가 숫자·판정을 모두 확정한 뒤, 그 결과를 한국 구조 실무의
"종합검토의견" 문체로 자연스럽게 풀어주는 계층.

원칙 (report_ux): **LLM is explanation-only, not judgment.**
    - LLM은 숫자·판정을 만들지 않는다. 받은 facts를 문장으로만 엮는다.
    - 모든 숫자는 facts에 "문자열로 미리 포맷"되어 들어가고, LLM은 그것을 복사만 한다.
    - 출력 후 숫자누출 self-check를 통과하지 못하면 기존 템플릿(summary_ko/en)으로 폴백.

이 모듈은 단계적으로 구현된다:
    P0 (현재): 계약(build_facts) + 숫자누출 self-check + 판정모순 검사 + 폴백 골격.
               LLM 호출 없음 — narrate_interpretation(llm=None)은 identity(리포트 무변화).
    P1: claude-opus-4-8 콜러블을 llm 인자로 주입 → summary_ko/en 생성.
    P2: diagnosis_narrative_ko/en + 프롬프트 캐싱 + 감사로그.

few-shot/register 가이드: docs/narrative_interpreter/fewshot_register_ko.md
머신리더블 자산:          docs/narrative_interpreter/fewshot_examples.json

Usage:
    from core.narrative_interpreter import narrate_interpretation
    interpretation = narrate_interpretation(interpretation, design_check)   # P0: identity
    # P1: narrate_interpretation(interpretation, design_check, llm=claude_narrator)
"""
from __future__ import annotations

import re
from typing import Any, Callable

# ============================================================
# 상수
# ============================================================

# 보고서 산문에서 "측정값/주장"이 아닌 보편 상수 — self-check 화이트리스트.
# "1.0"(내력비 허용한계), 0, 1 은 측정값이 아니라 코드상 고정 기준/자명한 정수.
_ALLOWED_CONSTANTS = {"0", "1"}

# 부재 지배 파괴 모드 한국어 라벨 (result_interpreter._interpret_members와 일치).
_GOVERNING_KO = {
    "axial": "축력",
    "bending_x": "강축 휨",
    "bending_y": "약축 휨",
    "shear": "전단",
    "interaction": "축력-휨 상관",
}

# 부재 타입 한국어 라벨 (내부 키 → 실무 표기).
_MEMBER_TYPE_KO = {
    "beam_x": "X방향 보",
    "beam_y": "Y방향 보",
    "beam": "보",
    "column": "기둥",
    "brace": "가새",
}

# 모달 1차 모드 방향 라벨.
_MODE_DIR_KO = {
    "TRAN-X": "X방향 병진",
    "TRAN-Y": "Y방향 병진",
    "ROTN-Z": "Z축 비틀림",
}

# 내진시스템 key → 한글 (구조형식 서술용).
_SEISMIC_SYSTEM_KO = {
    "special_moment_frame": "철골 특수모멘트골조",
    "intermediate_moment_frame": "철골 중간모멘트골조",
    "ordinary_moment_frame": "철골 보통모멘트골조",
    "rc_special_moment_frame": "RC 특수모멘트골조",
    "rc_intermediate_moment_frame": "RC 중간모멘트골조",
    "rc_ordinary_moment_frame": "RC 보통모멘트골조",
    "special_concentric_braced": "철골 특수중심가새골조",
    "ordinary_concentric_braced": "철골 보통중심가새골조",
}

# 검토 항목별 적용 KDS 기준 (grounded clause — LLM이 지어내지 않고 복사만).
_APPLICABLE_CODE_DRIFT = "KDS 41 17 00"   # 허용층간변위
_APPLICABLE_CODE_MEMBER = "KDS 14 31 00"  # 강구조 강도

# 판정모순 검사: moderate/severe인데 산문이 "안전 종결"로 단정하면 차단.
_SAFE_CONCLUSION_PATTERNS_KO = [
    "안전한 것으로 판단",
    "이상이 없는 것으로 판단",
    "안전성에 이상이 없",
]
_SAFE_CONCLUSION_PATTERNS_EN = [
    "judged structurally safe",
    "is structurally safe",
    "no structural concern",
]

# LLM 산문이 채울 수 있는 필드 (prose-only). 이외 필드는 절대 LLM이 건드리지 않음.
_NARRATIVE_FIELDS = (
    "summary_ko",
    "summary_en",
    "diagnosis_narrative_ko",
    "diagnosis_narrative_en",
)

# 숫자 토큰 정규식: 정수 또는 소수.
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")


# ============================================================
# 메인 진입점
# ============================================================

def narrate_interpretation(
    interpretation: dict | None,
    design_check: dict | None = None,
    *,
    llm: Callable[[dict], dict] | None = None,
    load_result: dict | None = None,
    model_info: dict | None = None,
    analysis_metadata: dict | None = None,
    seismic_method: str | None = None,
    analysis_id: str | None = None,
) -> dict | None:
    """해석 dict의 summary 산문을 LLM으로 다듬는다 (안 되면 그대로 둠).

    Args:
        interpretation: result_interpreter.interpret_results() 결과 dict.
        design_check:   run_design_check() 결과 dict (facts 보강용, 선택).
        llm:            facts(dict) -> 후보(dict) 콜러블. None이면 identity
                        (P0). 후보 dict 형식: {summary_ko, summary_en,
                        diagnosis_narrative_ko?, diagnosis_narrative_en?,
                        confidence: "high"|"low", used_only_given_facts: bool}.

    Returns:
        interpretation (얕은 복사) — summary_ko/en이 LLM 산문으로 덮어쓰였거나,
        검증 실패/LLM 미사용 시 원본 템플릿 그대로. narration_meta 키로 경로 기록.
    """
    if not interpretation:
        return interpretation

    facts = build_facts(
        interpretation, design_check,
        load_result=load_result, model_info=model_info,
        analysis_metadata=analysis_metadata, seismic_method=seismic_method,
    )

    # P0: LLM 미주입 → identity (템플릿 summary 유지).
    if llm is None:
        return _with_meta(
            interpretation,
            {"llm_used": False, "fallback": True, "reason": "no_llm"},
        )

    # C2′: facts-hash 캐시 조회 (동일 facts·동일 프롬프트면 LLM 재호출 없이 결정론적 재사용).
    cache_key = None
    prompt_hash = getattr(llm, "prompt_hash", None)
    if prompt_hash:
        try:
            from core import narration_cache
            cache_key = narration_cache.cache_key(facts, prompt_hash)
            cached = narration_cache.get(cache_key)
        except Exception:  # noqa: BLE001 — 캐시 장애는 라이브 생성으로 폴백
            cached = None
        if cached is not None:
            # 재사용 후보도 apply_narration으로 재검증(동일 facts → 동일 allowlist).
            result = apply_narration(interpretation, cached, facts)
            if isinstance(result.get("narration_meta"), dict):
                result["narration_meta"]["cached"] = True
            _audit_narration(analysis_id, facts, cached, result, cached=True)
            return result

    # P1+: LLM 호출 → 검증 → 적용 또는 폴백.
    try:
        candidate = llm(facts)
    except Exception as exc:  # noqa: BLE001 — 어떤 실패든 템플릿으로 폴백
        result = _with_meta(
            interpretation,
            {"llm_used": True, "fallback": True,
             "reason": f"llm_exception:{type(exc).__name__}"},
        )
        _audit_narration(analysis_id, facts, None, result)
        return result

    result = apply_narration(interpretation, candidate, facts)
    # 검증 통과(필드 1개 이상 적용)한 후보만 캐시 — 폴백 후보는 저장하지 않음.
    meta = result.get("narration_meta", {}) if isinstance(result, dict) else {}
    if cache_key and meta.get("applied_fields"):
        try:
            from core import narration_cache
            narration_cache.put(cache_key, candidate)
        except Exception:  # noqa: BLE001 — 캐시 저장 실패는 무시
            pass
    _audit_narration(analysis_id, facts, candidate, result)
    return result


def _audit_narration(
    analysis_id: str | None,
    facts: dict,
    candidate: dict | None,
    result: dict,
    *,
    cached: bool = False,
) -> None:
    """C3: AI 생성문 추적 기록 (append-only JSONL). 디스크 실패는 warning만."""
    try:
        from core.narration_audit import append_narration_audit
    except Exception:  # noqa: BLE001 — 감사 모듈 부재 시 조용히 무시
        return
    cand = candidate if isinstance(candidate, dict) else {}
    meta = result.get("narration_meta", {}) if isinstance(result, dict) else {}
    record = {
        "analysis_id": analysis_id or "",
        "model": meta.get("model") or cand.get("_model"),
        "prompt_hash": cand.get("_prompt_hash"),
        "cached": cached,
        "facts": facts,
        "reason": meta.get("reason"),
        "fallback": meta.get("fallback"),
        "applied_fields": meta.get("applied_fields"),
        "fallback_fields": meta.get("fallback_fields"),
        # 적용된 최종 산문(폴백 시 템플릿) + 거부된 경우 추적용 원본 후보.
        "summary_ko": result.get("summary_ko") if isinstance(result, dict) else None,
        "summary_en": result.get("summary_en") if isinstance(result, dict) else None,
        "diagnosis_narrative_ko": result.get("diagnosis_narrative_ko") if isinstance(result, dict) else None,
        "candidate_summary_ko": cand.get("summary_ko"),
        "candidate_summary_en": cand.get("summary_en"),
    }
    append_narration_audit(record)


# ============================================================
# 1. facts 빌드 (숫자 = 문자열로 사전 포맷)
# ============================================================

def build_facts(
    interpretation: dict,
    design_check: dict | None = None,
    *,
    load_result: dict | None = None,
    model_info: dict | None = None,
    analysis_metadata: dict | None = None,
    seismic_method: str | None = None,
) -> dict:
    """해석/검토 dict에서 LLM에 넘길 prose-safe facts를 만든다.

    모든 수치는 **문자열로 미리 포맷**한다 — LLM은 계산하지 않고 복사만 한다.
    선택 입력(load_result/model_info/analysis_metadata/seismic_method)이 주어지면
    지진근거(V·Cs·R·Ie)·하중요약·구조개요·해석방법 facts를 추가한다. 없으면 종전과 동일.
    """
    di = interpretation.get("drift_interpretation") or {}
    mi = interpretation.get("member_interpretation") or {}
    mo = interpretation.get("modal_interpretation") or {}
    diag = interpretation.get("diagnosis") or {}
    sugg = interpretation.get("suggestions") or []
    label = interpretation.get("severity_label") or {}
    dc_summary = (design_check or {}).get("summary") or {}

    reports = (load_result or {}).get("reports", {}) if isinstance(load_result, dict) else {}
    seismic_report = reports.get("seismic") if isinstance(reports, dict) else None
    wind_report = reports.get("wind") if isinstance(reports, dict) else None
    snow_report = reports.get("snow") if isinstance(reports, dict) else None
    gravity_report = reports.get("gravity") if isinstance(reports, dict) else None

    def _ok(r):
        return isinstance(r, dict) and not r.get("error")

    facts: dict[str, Any] = {
        "lang": "ko",
        "severity": interpretation.get("severity"),
        "severity_label_ko": label.get("ko"),
        "severity_label_en": label.get("en"),
    }

    # Tier1-4: 해석 실패(솔버 비수렴) — 0 결과를 '안전'으로 서술하지 않도록 명시.
    # 숫자 없는 순수 텍스트(allowlist 영향 없음).
    if (design_check or {}).get("analysis_error"):
        facts["analysis_error_note"] = (
            "구조해석 미수렴/실패로 변위·부재력이 0으로 산출됨 — 설계검토 결과 신뢰 불가, "
            "모델 점검 후 재해석 필요"
        )

    # 최대 층간변위비 (허용기준 대비) + 위치
    max_dr = dc_summary.get("max_drift_ratio")
    if max_dr is None:
        max_dr = di.get("max_ratio")
    if max_dr is not None:
        loc = _drift_location_ko(di)
        facts["max_drift_ratio"] = f"{max_dr:.2f}" + (f" ({loc})" if loc else "")
        # A2: 지배 하중조합 (조합 문자열의 계수는 allowlist 자기씨앗으로 근거화됨)
        gov_dr = dc_summary.get("drift_governing_combo")
        if gov_dr:
            facts["max_drift_ratio"] += f", 지배조합 {gov_dr}"

    # B4: 방향별 비대칭 (X/Y가 유의미하게 다를 때만 — "X 만족·Y 보강" 식 서술 근거)
    mrd = di.get("max_ratio_by_dir") or {}
    dx, dy = mrd.get("X"), mrd.get("Y")
    if isinstance(dx, (int, float)) and isinstance(dy, (int, float)):
        hi, lo = max(dx, dy), min(dx, dy)
        # 실제 횡변위가 있고(≥0.3), 방향 간 격차가 허용기준의 15% 이상일 때만 비대칭으로 본다
        if hi >= 0.3 and (hi - lo) >= 0.15:
            facts["drift_by_direction"] = f"X방향 {dx:.2f} / Y방향 {dy:.2f}"

    # 최대 부재 내력비 + 최약 부재
    max_ir = dc_summary.get("max_interaction_ratio")
    if max_ir is not None:
        wk = mi.get("weakest_link") or {}
        loc = _member_location_ko(wk)
        facts["max_interaction_ratio"] = f"{max_ir:.2f}" + (f" ({loc})" if loc else "")
        # A2: 지배 하중조합 (B3 수정으로 실제 H1 지배조합)
        gov_ir = dc_summary.get("member_governing_combo")
        if gov_ir:
            facts["max_interaction_ratio"] += f", 지배조합 {gov_ir}"

    # 부재 지배 파괴 모드 (NG일 때만 의미)
    gov = mi.get("governing_check")
    if gov and mi.get("status") == "NG":
        facts["governing_member_mode"] = _GOVERNING_KO.get(gov, gov)

    # B1: 보 수직처짐(사용성) — 미검토 시 명시(과대주장 방지), 검토 시 비율 노출
    max_df = dc_summary.get("max_deflection_ratio")
    dstat = dc_summary.get("deflection_status")
    if dstat == "not_checked" or (max_df is None and dstat not in ("OK", "NG")):
        facts["deflection_note"] = "보 수직처짐(사용성) 미검토 — 본 검토는 처짐을 포함하지 않음"
    elif max_df is not None:
        facts["max_deflection_ratio"] = f"{max_df:.2f} (활하중 L/360·전체 L/240)"

    # 모달 (고유주기 / 1차 모드 방향)
    if mo:
        t1 = mo.get("T1_actual_s")
        ta = mo.get("Ta_empirical_s")
        ratio = mo.get("T1_Ta_ratio")
        if t1 is not None and ta is not None and ratio is not None:
            facts["T1_Ta"] = f"{t1:.2f}s / {ta:.2f}s (비 {ratio:.2f})"
        fmd = mo.get("first_mode_direction")
        if fmd:
            facts["first_mode_direction"] = _MODE_DIR_KO.get(fmd, fmd)
        if mo.get("torsion_risk"):
            facts["torsion_risk"] = True
        # A4: 누적 유효질량 참여율 (KDS 41 17 00 충분조건 ≥90%)
        mx = mo.get("cumulative_mass_x_pct")
        my = mo.get("cumulative_mass_y_pct")
        if isinstance(mx, (int, float)) and isinstance(my, (int, float)):
            suf = "충분조건(≥90%) 만족" if mo.get("mass_participation_sufficient") else "90% 미만 — 모드 수 확대 검토"
            facts["mass_participation"] = f"X {mx:.0f}% / Y {my:.0f}% ({suf})"

    # NG 개수 (정수 그대로 — 표현용 '4개 부재')
    facts["ng_stories"] = int(dc_summary.get("ng_stories", 0) or 0)
    facts["ng_members"] = int(dc_summary.get("ng_members", 0) or 0)

    # 진단 (NG일 때) — B2: 진단 서사용 재료 보강 (연약층·층별 NG 분포)
    if diag:
        diag_facts: dict[str, Any] = {
            "primary_cause_ko": diag.get("primary_cause_ko"),
            "contributing_factors_ko": diag.get("contributing_factors_ko") or [],
        }
        if di.get("soft_story_detected") and di.get("soft_story_stories"):
            ss = di["soft_story_stories"]
            diag_facts["soft_story_ko"] = (
                ", ".join(f"{s}층" for s in ss) + " 연약층(인접 층 대비 변위 집중) 감지"
            )
        ng_by_story = mi.get("ng_by_story") or {}
        if ng_by_story:
            top_stories = sorted(ng_by_story, key=lambda k: ng_by_story[k], reverse=True)[:3]
            diag_facts["ng_by_story_ko"] = ", ".join(
                f"{s}층 {ng_by_story[s]}개" for s in top_stories
            )
        facts["diagnosis"] = diag_facts

    # B3: 개선 제안 다건 (상위 2~3건 문자열 리스트 — 단면치수는 allowlist 자기씨앗)
    if sugg:
        sugg_strings = [s for s in (_format_suggestion(x) for x in sugg[:3]) if s]
        if sugg_strings:
            facts["suggestions"] = sugg_strings

    # ── A6: 부재 그룹별 NG 분포 + 전단비 (인자 불필요 — interpretation/dc에 이미 존재) ──
    ng_by_type = mi.get("ng_by_type") or {}
    # member_check 키가 존재하되 값이 None일 수 있어 default 인자 대신 `or {}` 사용.
    mc_summary = ((design_check or {}).get("member_check") or {}).get("summary", {}) if isinstance(design_check, dict) else {}
    member_groups: dict[str, Any] = {}
    grp_parts = [
        f"{_MEMBER_TYPE_KO.get(k, k)} {ng_by_type[k]}개"
        for k in ("column", "beam_x", "beam_y")
        if ng_by_type.get(k)
    ]
    if grp_parts:
        member_groups["ng_distribution"] = ", ".join(grp_parts)
    msr = mc_summary.get("max_shear_ratio")
    if isinstance(msr, (int, float)):
        member_groups["max_shear_ratio"] = f"{msr:.2f}"
    if member_groups:
        facts["member_groups"] = member_groups

    # ── Tier1-1/2/5: 기둥 K / 보 LTB / RSA 미검토 (숫자는 facts에 자기씨앗) ──
    col_K = mc_summary.get("column_K")
    if isinstance(col_K, (int, float)) and col_K > 1.0:
        facts["effective_length_K"] = (
            f"기둥 유효좌굴길이계수 K={col_K:g} (비가새 모멘트골조 sway, AISC)"
        )
    n_ltb = mc_summary.get("n_ltb_governed")
    if isinstance(n_ltb, int) and n_ltb > 0:
        facts["ltb_note"] = (
            f"보 {n_ltb}개 횡-비틀림좌굴(LTB) 지배 — 비지지길이 기준 강축 휨내력 감소(AISC F2)"
        )
    rsa_unchecked = mc_summary.get("rsa_unchecked_combos") or []
    if rsa_unchecked:
        facts["rsa_member_check_note"] = (
            f"RSA 조합 {len(rsa_unchecked)}개 부재강도 미검토 (RSA는 변위만 산출) — "
            f"ELF 조합 기준 부재검토만 수행"
        )
    # Tier2-6/7/15: 세장비 / 비콤팩트 / 인장 순단면파단 (숫자 자기씨앗)
    n_sl = mc_summary.get("n_slenderness_ng")
    if isinstance(n_sl, int) and n_sl > 0:
        facts["slenderness_note"] = (
            f"세장비 한계 초과 부재 {n_sl}개 (압축 KL/r≤200·인장 L/r≤300, AISC E2/D1)"
        )
    n_nc = mc_summary.get("n_noncompact")
    if isinstance(n_nc, int) and n_nc > 0:
        facts["compactness_note"] = (
            f"비콤팩트/세장 단면 {n_nc}개 — 플랜지국부좌굴(F3.2)로 휨내력 감소 적용"
        )
    n_cslen = mc_summary.get("n_compression_slender")
    if isinstance(n_cslen, int) and n_cslen > 0:
        facts["compression_slender_note"] = (
            f"압축 세장요소 단면 {n_cslen}개 — E7 추가감소 미반영(보수검토 필요)"
        )

    # ── A1: 지진 설계근거 (밑면전단 V·Cs·R·Ie·SDS·W) ──
    if _ok(seismic_report):
        sr = seismic_report
        seis: dict[str, str] = {}

        def _fmt(key, spec, suffix=""):
            v = sr.get(key)
            return (format(v, spec) + suffix) if isinstance(v, (int, float)) else None

        for fk, (key, spec, suffix) in {
            "base_shear_V": ("V_kN", ".0f", " kN"),
            "Cs": ("Cs", ".4f", ""),
            "R": ("R", ".1f", ""),
            "Ie": ("IE", ".1f", ""),
            "SDS": ("SDS", ".3f", " g"),
            "SD1": ("SD1", ".3f", " g"),
            "W_eff": ("W_kN", ".0f", " kN"),
            "T_design": ("T_sec", ".2f", " s"),
            "Ta": ("Ta_sec", ".2f", " s"),
        }.items():
            val = _fmt(key, spec, suffix)
            if val is not None:
                seis[fk] = val
        if all(isinstance(sr.get(k), (int, float)) for k in ("Cs", "W_kN", "V_kN")):
            seis["formula"] = f"V = Cs·W = {sr['Cs']:.4f} × {sr['W_kN']:.0f} = {sr['V_kN']:.0f} kN"
        if sr.get("region"):
            seis["region"] = sr["region"]
        if seis:
            facts["seismic"] = seis

    # ── B2: 내진설계범주(SDC) + 횡력저항시스템 적격성 ──
    sysc = (design_check or {}).get("system_check") if isinstance(design_check, dict) else None
    if isinstance(sysc, dict):
        sb: dict[str, Any] = {"sdc": sysc.get("sdc")}
        hl = sysc.get("height_limit") or {}
        if hl.get("limit_desc"):
            sb["height_limit"] = hl["limit_desc"]
        sb["status"] = "적합" if sysc.get("status") == "OK" else "부적격"
        if sysc.get("issues"):
            sb["issues"] = sysc["issues"][:2]
        facts["seismic_design_category"] = sb

    # ── C1/C2/C3: 전역안정(전도)·비틀림 비정형·P-Delta θ ──
    stab = (design_check or {}).get("stability_check") if isinstance(design_check, dict) else None
    if isinstance(stab, dict) and isinstance(stab.get("governing_FS"), (int, float)):
        req = stab.get("required_FS", 1.5)
        st = "만족" if stab.get("status") == "OK" else "부족"
        facts["overturning"] = f"전도 안전율 {stab['governing_FS']:.2f} (허용 ≥ {req:g}, {st})"
    pd = (design_check or {}).get("pdelta_check") if isinstance(design_check, dict) else None
    if isinstance(pd, dict) and isinstance(pd.get("max_theta"), (int, float)):
        tl = pd.get("theta_limit")
        st = "안정" if pd.get("status") == "OK" else "불안정"
        facts["pdelta_theta"] = (
            f"P-Delta 안정계수 θ_max {pd['max_theta']:.3f}"
            + (f" (한계 {tl:.3f}, {st})" if isinstance(tl, (int, float)) else "")
        )
    tor = (design_check or {}).get("torsion_check") if isinstance(design_check, dict) else None
    torsion_ran = isinstance(tor, dict) and isinstance(tor.get("max_ratio"), (int, float))
    if torsion_ran:
        cls_ko = {"regular": "정형", "torsional": "비틀림 비정형",
                  "extreme": "극단 비틀림 비정형"}.get(tor.get("classification"), "")
        facts["torsion"] = f"δmax/δavg {tor['max_ratio']:.2f} ({cls_ko})"
    # T4-5: 비틀림 비정형 검토 수행 여부 플래그 (bool → allowlist 무영향).
    # drift·torsion 둘 다 None(횡력해석/변위극값 부재 → 중력전용)이면 '미평가'를 명시한다
    # (Tier1-5 RSA·Tier2-13 not_checked 패턴 — 미수행을 '안전'으로 침묵시키지 않음).
    facts["torsion_performed"] = bool(torsion_ran)
    if not torsion_ran and max_dr is None:
        facts["torsion_note"] = (
            "비틀림 비정형(δmax/δavg) 미평가 — 횡력 해석/변위 극값이 없어 본 검토에서 "
            "비틀림을 산정하지 않음(미검토 항목, '안전'으로 단정 불가)"
        )

    # #10: 풍하중 사용성 층간변위 (H/400). 미검토 시 명시(과대주장 방지).
    max_wd = dc_summary.get("max_wind_drift_ratio")
    wd_stat = dc_summary.get("wind_drift_status")
    wind_check = (design_check or {}).get("wind_check") if isinstance(design_check, dict) else None
    if isinstance(max_wd, (int, float)) and isinstance(wind_check, dict):
        lim = wind_check.get("limit_inv", 400)
        st = "만족" if wd_stat == "OK" else "초과"
        facts["wind_drift"] = f"풍하중 사용성 층간변위비 {max_wd:.2f} (허용 1/{lim}, {st})"

    # #12: 수직 비정형 (KDS 41 17 00 표 5.3-2). 비정형일 때만 서술 근거.
    vert = (design_check or {}).get("vertical_check") if isinstance(design_check, dict) else None
    if isinstance(vert, dict) and vert.get("irregular"):
        types_ko = {"stiffness_soft_story": "강성(연층)", "mass": "중량", "geometric": "기하"}
        kinds = ", ".join(types_ko.get(t, t) for t in vert.get("types", []))
        pre = "극단 " if vert.get("extreme") else ""
        facts["vertical_irregularity"] = (
            f"{pre}수직 비정형 감지({kinds}) — KDS 41 17 00 표 5.3-2, 동적해석/내진상세 검토 필요"
        )

    # #8: 우발 비틀림 — 등가정적 모델 미적용 명시 (숫자 자기씨앗).
    accid = (design_check or {}).get("accidental_torsion") if isinstance(design_check, dict) else None
    if isinstance(accid, dict) and not accid.get("applied_in_model"):
        ecc = accid.get("eccentricity_ratio") or 0.05   # None-safe (key 존재+None 대비)
        mx, my = accid.get("total_Mt_x_kNm"), accid.get("total_Mt_y_kNm")
        tail = (f" (총 Mt_x {mx:.0f}·Mt_y {my:.0f} kN·m)"
                if isinstance(mx, (int, float)) and isinstance(my, (int, float)) else "")
        facts["accidental_torsion"] = (
            f"우발 비틀림 ±{ecc * 100:g}% 편심 산정{tail}, 등가정적 모델 미적용 — 최종 설계 시 명시 반영"
        )

    # #13: 지진입력 손상 + 미검토 항목 (숫자 없는 텍스트 — allowlist 무영향).
    if dc_summary.get("seismic_input_error"):
        facts["seismic_input_error_note"] = (
            "지진하중 입력 손상/누락으로 지진검토(층간변위·내진설계범주·전도·P-Delta) 미수행 — "
            "결과를 '안전'으로 단정 불가, 입력 정정 후 재검토 필요"
        )
    nci = dc_summary.get("not_checked_items") or []
    if nci:
        facts["not_checked_note"] = "본 검토 미수행(미포함) 항목: " + ", ".join(str(x) for x in nci)

    # ── A5: 하중조건 요약 (대표층 1개만) ──
    loads: dict[str, str] = {}
    if isinstance(gravity_report, list) and gravity_report:
        g0 = gravity_report[0]
        dl, ll = g0.get("DL_kNm2"), g0.get("LL_kNm2")
        if isinstance(dl, (int, float)) and isinstance(ll, (int, float)):
            usage = g0.get("usage")
            loads["gravity"] = (
                f"고정하중 {dl:.2f} kN/m², 활하중 {ll:.2f} kN/m²"
                + (f" ({usage})" if usage else "")
            )
    if _ok(wind_report) and isinstance(wind_report.get("V0_ms"), (int, float)):
        fx = wind_report.get("total_Fx_kN", 0) or 0
        fy = wind_report.get("total_Fy_kN", 0) or 0
        loads["wind"] = f"기본풍속 {wind_report['V0_ms']:.0f} m/s, 총 풍력 {fx:.0f}/{fy:.0f} kN"
    if _ok(snow_report) and isinstance(snow_report.get("S_design_kNm2"), (int, float)):
        loads["snow"] = f"지붕적설하중 {snow_report['S_design_kNm2']:.2f} kN/m²"
    if loads:
        facts["loads"] = loads

    # ── A2: 구조 개요 (규모·형식·재료·단면·높이) ──
    mio = model_info or {}
    ov: list[str] = []
    if mio.get("num_stories"):
        ov.append(f"지상 {int(mio['num_stories'])}층")
    sys_key = (seismic_report or {}).get("seismic_system") if isinstance(seismic_report, dict) else None
    sys_ko = _SEISMIC_SYSTEM_KO.get(sys_key) if sys_key else None
    if sys_ko:
        ov.append(sys_ko)
    if mio.get("material"):
        ov.append(str(mio["material"]))
    col, bx = mio.get("column_section"), mio.get("beam_x_section")
    if col and bx:
        ov.append(f"기둥 {col} / 보 {bx}")
    th = mio.get("total_height_m")
    if isinstance(th, (int, float)):
        ov.append(f"총높이 {th:.1f} m")
    if ov:
        facts["structure_overview"] = ", ".join(ov)

    # ── A3: 해석 방법 ──
    am = analysis_metadata or {}
    method_parts = ["3차원 골조해석"]
    sm = (seismic_method or "").upper()
    if sm == "RSA":
        method_parts.append("응답스펙트럼해석(RSA)")
    elif _ok(seismic_report):
        method_parts.append("등가정적해석법(ELF)")
    transf = (am.get("actual_transf") or am.get("requested_mode") or "")
    if str(transf).lower() not in ("", "linear"):
        method_parts.append("기하비선형(P-Delta)")
    if mio.get("rigid_diaphragm") or am.get("rigid_diaphragm"):
        method_parts.append("강체 다이어프램")
    if len(method_parts) > 1:
        facts["analysis_method_ko"] = ", ".join(method_parts)

    # ── A7: 적용 설계기준 (동적 — 검토/하중 항목이 있을 때만, 하위호환 유지) ──
    codes: list[str] = []

    def _addcode(c):
        if c and c not in codes:
            codes.append(c)

    if _ok(seismic_report):
        _addcode("KDS 41 10 00")  # 설계일반·하중조합 (지진검토 동반)
        _addcode(seismic_report.get("code") or _APPLICABLE_CODE_DRIFT)
    if (isinstance(gravity_report, list) and gravity_report) or _ok(wind_report) or _ok(snow_report):
        _addcode("KDS 41 12 00")  # 설계하중 (중력·풍·적설)
    if max_dr is not None:
        _addcode(_APPLICABLE_CODE_DRIFT)
    if max_ir is not None:
        _addcode(_APPLICABLE_CODE_MEMBER)
    if codes:
        facts["applicable_codes"] = codes

    # ── A8: 해석 가정·한계 (숫자 없는 라벨, 상위 1~2건 — 마무리 1문장용) ──
    mc = (design_check or {}).get("member_check") if isinstance(design_check, dict) else None
    lims = _limitations_ko((mc or {}).get("assumptions") or [])
    if lims:
        facts["limitations_ko"] = lims

    return facts


def _format_suggestion(s: dict) -> str | None:
    """개선 제안 dict을 prose-safe 한 문장 라벨로 포맷. 숫자는 단면명만(자기씨앗)."""
    typ = s.get("type")
    tgt = _MEMBER_TYPE_KO.get(s.get("target"), s.get("target") or "")
    cur, rec = s.get("current"), s.get("recommended")
    if typ == "section_upgrade" and cur and rec:
        return f"{tgt} 단면 증대: {cur} → {rec}"
    if typ == "system_change":
        return "횡력저항시스템 변경·추가(가새·전단벽 등)"
    if typ == "geometry_review":
        if isinstance(s.get("target"), str) and s["target"].startswith("story"):
            return "연약층 횡강성 보강(가새 추가·기둥 단면 증대) 검토"
        return "형상·강성 재검토"
    return None


# A8: member_check.assumptions → 숫자 없는 한계 라벨.
def _limitations_ko(assumptions: list) -> list[str]:
    joined = " ".join(str(a) for a in assumptions)
    out: list[str] = []
    if "좌굴길이계수" in joined or "유효좌굴" in joined:
        # Tier1-1: K는 골조형식별 적용(모멘트골조 sway 시 증가). '보수가정' 표현 정정.
        # 숫자 없는 라벨 유지(자기씨앗 불필요).
        out.append("유효좌굴길이계수 K는 골조형식별로 적용함(비가새 모멘트골조는 sway로 K 증가)")
    if "LTB" in joined or "횡비틀림" in joined or "횡지지" in joined:
        # Tier1-2: LTB는 비지지길이 기준으로 검토(슬래브 연속횡지지 가정 시 Mp).
        out.append("보 횡비틀림좌굴(LTB)은 비지지길이 기준 검토(슬래브 연속횡지지 가정 시 소성모멘트)")
    return out[:2]


def _drift_location_ko(di: dict) -> str:
    cs = di.get("critical_story")
    cd = di.get("critical_direction")
    parts = []
    if cs is not None:
        parts.append(f"{cs}층")
    if cd:
        parts.append(f"{cd}방향")
    return ", ".join(parts)


def _member_location_ko(wk: dict) -> str:
    if not wk:
        return ""
    mtype = _MEMBER_TYPE_KO.get(wk.get("type"), wk.get("type") or "")
    mid = wk.get("member_id")
    story = wk.get("story")
    parts = []
    head = mtype
    if mid is not None:
        head = f"{mtype} #{mid}".strip()
    if head.strip():
        parts.append(head.strip())
    if story is not None:
        parts.append(f"{story}층")
    return ", ".join(parts)


# ============================================================
# 2. 숫자누출 self-check (결정론적)
# ============================================================

def _norm_num(token: str) -> str:
    """숫자 토큰 정규화. 소수는 끝자리 0 제거, 정수는 int 정규화.

    예: '1.40'->'1.4', '1.0'->'1', '0.85'->'0.85', '00'->'0', '003'->'3'.
    """
    if "." in token:
        s = token.rstrip("0").rstrip(".")
        return s if s else "0"
    try:
        return str(int(token))
    except ValueError:
        return token


def _numbers_in(text: str) -> list[str]:
    """문자열에서 정규화된 숫자 토큰 목록."""
    return [_norm_num(t) for t in _NUM_RE.findall(text or "")]


def build_number_allowlist(facts: dict) -> set[str]:
    """facts의 모든 값에서 추출한 정규화 숫자 + 보편 상수 집합."""
    allow: set[str] = set(_ALLOWED_CONSTANTS)

    def walk(v: Any) -> None:
        if isinstance(v, str):
            allow.update(_numbers_in(v))
        elif isinstance(v, bool):
            return
        elif isinstance(v, (int, float)):
            allow.update(_numbers_in(str(v)))
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                walk(x)

    walk(facts)
    return allow


def find_ungrounded_numbers(text: str, allowlist: set[str]) -> list[str]:
    """text의 숫자 중 allowlist에 없는 것(= 근거 없는 숫자) 목록."""
    return [n for n in _numbers_in(text) if n not in allowlist]


def passes_number_check(text: str, allowlist: set[str]) -> bool:
    """text의 모든 숫자가 근거(allowlist) 내에 있으면 True."""
    return not find_ungrounded_numbers(text, allowlist)


# ============================================================
# 3. 판정모순 검사
# ============================================================

def contradicts_verdict(text: str, severity: str | None, lang: str = "ko") -> bool:
    """moderate/severe인데 산문이 '안전 종결'로 단정하면 True (모순)."""
    if severity not in ("moderate", "severe") or not text:
        return False
    patterns = _SAFE_CONCLUSION_PATTERNS_KO if lang == "ko" else _SAFE_CONCLUSION_PATTERNS_EN
    return any(p in text for p in patterns)


# ============================================================
# 4. 검증 + 적용
# ============================================================

def apply_narration(interpretation: dict, candidate: dict | None, facts: dict) -> dict:
    """LLM 후보 산문을 검증하고, **필드 단위로** 통과분만 적용한다 (C4).

    각 prose 필드는 (1) 숫자누출 (2) 판정모순 (3) ko/en 숫자 불일치(C5) 중
    하나라도 걸리면 해당 필드만 폐기(템플릿 유지)하고, 깨끗한 필드는 적용한다.
    narration_meta에 필드별 적용/폴백 내역을 기록한다.
    """
    # 신뢰도 게이트 / 빈 출력
    if not candidate or candidate.get("confidence") == "low":
        return _with_meta(
            interpretation,
            {"llm_used": True, "fallback": True, "reason": "low_confidence_or_empty"},
        )

    allow = build_number_allowlist(facts)
    severity = facts.get("severity")

    leaks: dict[str, list[str]] = {}
    contradictions: list[str] = []
    bilingual_fields: list[str] = []

    # Pass 1: 필드별 숫자누출 / 판정모순.
    field_issue: dict[str, str] = {}
    for field in _NARRATIVE_FIELDS:
        txt = candidate.get(field)
        if not txt:
            continue
        ung = find_ungrounded_numbers(txt, allow)
        if ung:
            leaks[field] = ung
            field_issue[field] = "number_leak"
            continue
        lang = "en" if field.endswith("_en") else "ko"
        if contradicts_verdict(txt, severity, lang):
            contradictions.append(field)
            field_issue[field] = "verdict_contradiction"

    # Pass 2 (C5): summary_ko ↔ summary_en 숫자 교차 일관성 (KDS 조항·보편상수 제외).
    # 두 요약이 모두 개별적으로 깨끗할 때만 비교한다 — 한쪽이 이미 누출/모순으로
    # 폐기되면(그 쪽은 템플릿으로 폴백) 남은 깨끗한 요약까지 버리지 않는다.
    sko_ok = bool(candidate.get("summary_ko")) and "summary_ko" not in field_issue
    sen_ok = bool(candidate.get("summary_en")) and "summary_en" not in field_issue
    if sko_ok and sen_ok:
        exclude = _clause_and_const_numbers(facts)
        if not _bilingual_consistent(candidate["summary_ko"], candidate["summary_en"], exclude):
            bilingual_fields = ["summary_ko", "summary_en"]

    # 적용 가능한(깨끗하고 비대칭검사도 통과한) 필드만 수집.
    dropped = set(field_issue) | set(bilingual_fields)
    clean: dict[str, str] = {
        field: candidate[field]
        for field in _NARRATIVE_FIELDS
        if candidate.get(field) and field not in dropped
    }

    # 적용 가능한 필드가 하나도 없으면 전량 폴백 (메타 형태 하위호환 유지).
    if not clean:
        reason = (
            "number_leak" if leaks else
            "verdict_contradiction" if contradictions else
            "bilingual_mismatch" if bilingual_fields else
            "empty_output"
        )
        meta = {"llm_used": True, "fallback": True, "reason": reason}
        if leaks:
            meta["leaked"] = leaks
        if contradictions:
            meta["contradictions"] = contradictions
        if bilingual_fields:
            meta["bilingual_fields"] = bilingual_fields
        return _with_meta(interpretation, meta)

    # 깨끗한 필드만 덮어쓰기.
    result = dict(interpretation)
    for field, txt in clean.items():
        result[field] = txt

    fallback_fields: dict[str, str] = {}
    for f in leaks:
        fallback_fields[f] = "number_leak"
    for f in contradictions:
        fallback_fields[f] = "verdict_contradiction"
    for f in bilingual_fields:
        fallback_fields[f] = "bilingual_mismatch"

    meta = {
        "llm_used": True,
        "fallback": bool(fallback_fields),
        "reason": "partial_fallback" if fallback_fields else "applied",
        "applied_fields": list(clean.keys()),
        "model": candidate.get("_model") if isinstance(candidate, dict) else None,
    }
    if fallback_fields:
        meta["fallback_fields"] = fallback_fields
        if leaks:
            meta["leaked"] = leaks
        if contradictions:
            meta["contradictions"] = contradictions
        if bilingual_fields:
            meta["bilingual_fields"] = bilingual_fields
    result["narration_meta"] = meta
    return result


def _clause_and_const_numbers(facts: dict) -> set[str]:
    """C5 교차검사에서 제외할 숫자: KDS 조항 번호 + 보편상수(0/1).

    조항 번호(41/17/00 등)는 ko·en 양쪽에 동일하게 나타나며 측정값이 아니므로
    교차검사 대상에서 뺀다.
    """
    nums: set[str] = set(_ALLOWED_CONSTANTS)
    for c in facts.get("applicable_codes") or []:
        nums.update(_numbers_in(str(c)))
    return nums


def _bilingual_consistent(ko: str, en: str, exclude: set[str]) -> bool:
    """ko/en의 '측정값'(소수) 집합이 동일하면 True.

    비교 대상은 **소수(소수점이 살아있는 토큰)뿐**이다. 비·가속도·주기·하중 등
    핵심 측정값은 두 언어 모두 숫자로 표기되지만, 정수 개수·서수는 한국어가 '4개',
    영어가 'four'/'Story 3'처럼 단어/자릿수 표기가 언어별로 갈려 정수 집합 비교는
    거짓 불일치(예: 한국어 '4' vs 영어 'four')를 만들어 깨끗한 산문을 통째로
    폐기시킨다. 측정값(소수)에만 교차검사를 한정해 이 거짓 폴백을 제거한다.
    (정수든 소수든 '근거 없는 숫자' 누출은 Pass 1 allowlist 검사가 양 언어에서
    이미 차단하므로 anti-hallucination 안전성은 그대로 유지된다.)
    """
    nk = {t for t in _numbers_in(ko) if "." in t} - exclude
    ne = {t for t in _numbers_in(en) if "." in t} - exclude
    return nk == ne


# ============================================================
# 헬퍼
# ============================================================

def _with_meta(interpretation: dict, meta: dict) -> dict:
    """interpretation 얕은 복사 + narration_meta 부착 (원본 prose 보존)."""
    result = dict(interpretation)
    result["narration_meta"] = meta
    return result
