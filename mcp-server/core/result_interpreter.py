"""구조해석 결과 해석 모듈.

설계 검토(design_check) 결과와 해석 데이터를 종합 분석하여
심각도 분류, 패턴 인식, 진단, 개선 제안을 자동 생성한다.

Finding 코드:
    R01  drift   critical  Soft story 감지
    R02  drift   warning   층간변위 초과
    R03  drift   info      층간변위 OK
    R04  drift   caution   Whipping 패턴
    R05  member  warning   부재 NG + weakest link
    R06  member  caution   지배 파괴 모드
    R07  member  warning   특정 층 NG 집중
    R08  member  info      부재 전부 OK
    R09  modal   warning   T1/Ta > 1.4 (과유연)
    R10  modal   warning   1차 모드 비틀림
    R11  modal   caution   모달-변위 방향 상관
    R12  modal   info      모달 결과 정상
    R13  overall info      심각도 분류
    R14  overall caution   여유 부족 (marginal)
    R15  overall critical  심각 — 시스템 변경 필요
    R16  modal   caution   누적 유효질량 참여율 < 90% (모드 수 부족)
    R17  deflect warning   보 처짐 사용성 NG (L/360·L/240)
    R18  deflect info      보 처짐 사용성 OK
    R19  system  critical  내진설계범주 — 횡력저항시스템 부적격
    R20  system  info      내진설계범주 — 시스템 높이제한 충족
    R21  member  caution   순인장 부재 — 연결부 별도검토 권고
    R22  stab    critical  전도 안정 부족
    R23  stab    crit/caut P-Delta 불안정(NG) / 2차효과 고려(caution)
    R24  torsion warn/caut (극단)비틀림 비정형 — Ax 변위증폭
    R25  torsion caution   비틀림 비정형 미평가
    R26  analysis critical 해석 미수렴(솔버 실패) — 결과 신뢰 불가
    R27  member  caution   RSA 조합 부재검토 미수행
    R28  member  info      기둥 유효좌굴길이계수 K>1 적용
    R29  member  caution   보 LTB 지배(AISC F2)
    R30  member  warning   세장비 한계 초과(KL/r·L/r)
    R31  member  caution   비콤팩트/세장 단면(F3.2)
    R32  member  caution   압축 세장요소(E7 미반영)
    R33  member  caution   인장 순단면파단(D2-b) 지배
    R34  wind    warn/info 풍하중 사용성 층간변위 NG/OK (1/400)   [bundle B #10]
    R35  vert    warn/caut (극단)수직 비정형 (KDS 표 5.3-2)        [bundle B #12]
    R36  torsion caution   우발 비틀림 ±5% 편심 모델 미적용         [bundle B #8]
    R37  analysis warning  지진입력 손상 — 지진검토 미수행          [bundle B #13]

    주의: R23(NG/caution)·R34(NG/OK)는 if/elif/else로 상호배타 — 1회 실행당 1개만 방출.
    새 검토 추가 시 R38부터 할당(위 표 갱신).

Usage:
    from core.result_interpreter import interpret_results
    interp = interpret_results(design_check, multi_result, modal_analysis)
"""
from __future__ import annotations

import math
from collections import defaultdict

# ============================================================
# 상수
# ============================================================

_SEVERITY_THRESHOLDS = {"safe": 0.7, "marginal": 1.0, "moderate": 1.3}

_SEVERITY_LABELS = {
    "safe": ("안전", "Safe"),
    "marginal": ("경계", "Marginal"),
    "moderate": ("보통 — 단면 보강 필요", "Moderate — section upgrade needed"),
    "severe": ("심각 — 구조시스템 변경 검토", "Severe — structural system review needed"),
}

_SOFT_STORY_RATIO = 1.5
_T1_TA_FLEXIBILITY = 1.4

_EMPIRICAL_PERIOD = {
    "steel_moment_frame": (0.085, 0.75),
    "rc_moment_frame": (0.073, 0.75),
    "braced_frame": (0.049, 0.75),
}

_COLUMN_UPGRADE_SERIES = [
    "H-200x200", "H-250x250", "H-300x300", "H-350x350",
    "H-400x400", "H-414x405", "H-428x407", "H-458x417",
    "H-498x432",
]
_BEAM_UPGRADE_SERIES = [
    "H-200x100", "H-250x125", "H-300x150", "H-350x175",
    "H-400x200", "H-450x200", "H-500x200", "H-600x200",
    "H-700x300", "H-800x300", "H-900x300",
]

_CAUSE_LABELS = {
    "lateral_stiffness": ("횡강성 부족", "Insufficient lateral stiffness"),
    "member_capacity": ("부재 내력 부족", "Insufficient member capacity"),
    "soft_story": ("연약층 (Soft Story)", "Soft story irregularity"),
    "torsion": ("비틀림 비정형", "Torsional irregularity"),
}


# ============================================================
# 메인 진입점
# ============================================================

def interpret_results(
    design_check: dict,
    multi_result,
    modal_analysis: dict | None = None,
    building_config: dict | None = None,
) -> dict:
    """구조해석 결과 종합 해석.

    Args:
        design_check: run_design_check() 결과 dict
        multi_result: Frame3DMultiCaseResult 객체
        modal_analysis: modal_analysis dict (None이면 모달 미수행)
        building_config: 원본 building config (선택적)

    Returns:
        Interpretation dict — severity, findings, diagnosis, suggestions, summary
    """
    if not design_check:
        return {}

    findings: list[dict] = []

    # 0. Tier1-4: 해석 실패 게이트 — 0 결과를 '안전'으로 해석하지 않음.
    if design_check.get("analysis_error"):
        ae = design_check.get("analysis_error") or {}
        findings.append({
            "code": "R26",
            "category": "analysis",
            "severity": "critical",
            "message_ko": (
                "구조해석 미수렴/실패 — 변위·부재력 0 산출, 설계검토 결과 신뢰 불가. "
                "모델 점검 후 재해석 필요"
            ),
            "message_en": (
                "Structural analysis did not converge — zero response; design check "
                "results unreliable. Re-run after model review"
            ),
            "data": {"failed_cases": ae.get("failed_cases", [])},
        })

    # 1. 심각도 분류
    severity = _classify_severity(design_check)
    ko, en = _SEVERITY_LABELS[severity]
    findings.append({
        "code": "R13" if severity in ("safe", "marginal") else "R15" if severity == "severe" else "R13",
        "category": "overall",
        "severity": "info" if severity == "safe" else "caution" if severity == "marginal" else "critical",
        "message_ko": f"종합 심각도: {ko}",
        "message_en": f"Overall severity: {en}",
        "data": {"severity": severity},
    })
    if severity == "marginal":
        findings.append({
            "code": "R14",
            "category": "overall",
            "severity": "caution",
            "message_ko": "설계검토 통과하였으나 여유가 부족합니다. 하중 변경 시 재검토 필요.",
            "message_en": "Design checks passed but with limited margin. Re-check if loads change.",
            "data": None,
        })

    # 2. 층간변위 해석
    drift_interp, drift_findings = _interpret_drift(
        design_check.get("drift_check"),
    )
    findings.extend(drift_findings)

    # 3. 부재 강도 해석
    member_interp, member_findings = _interpret_members(
        design_check.get("member_check"),
    )
    findings.extend(member_findings)

    # 3b. 보 처짐(사용성) 해석 (B1)
    deflection_interp, deflection_findings = _interpret_deflection(
        design_check.get("deflection_check"),
    )
    findings.extend(deflection_findings)

    # 3c. 내진설계범주(SDC) + 시스템 적격성 (B2)
    system_check = design_check.get("system_check")
    if system_check:
        if system_check["status"] == "NG":
            findings.append({
                "code": "R19",
                "category": "seismic_system",
                "severity": "critical",
                "message_ko": (
                    f"내진설계범주 {system_check['sdc']} — 횡력저항시스템 부적격: "
                    + "; ".join(system_check.get("issues", []))
                ),
                "message_en": f"SDC {system_check['sdc']} — seismic system inadequate",
                "data": {"sdc": system_check["sdc"], "issues": system_check.get("issues", [])},
            })
        else:
            hl = (system_check.get("height_limit") or {}).get("limit_desc", "")
            findings.append({
                "code": "R20",
                "category": "seismic_system",
                "severity": "info",
                "message_ko": (
                    f"내진설계범주 {system_check['sdc']} — 횡력저항시스템 높이제한 충족 ({hl})"
                ),
                "message_en": f"SDC {system_check['sdc']} — system within height limit",
                "data": {"sdc": system_check["sdc"]},
            })

    # 3d. 전역안정 전도 (C1)
    stability_check = design_check.get("stability_check")
    if stability_check and stability_check["status"] == "NG":
        gfs = stability_check.get("governing_FS")
        findings.append({
            "code": "R22",
            "category": "stability",
            "severity": "critical",
            "message_ko": f"전도 안정 부족 — 안전율 {gfs} < {stability_check.get('required_FS', 1.5)}",
            "message_en": f"Overturning instability — FS {gfs} < {stability_check.get('required_FS', 1.5)}",
            "data": {"governing_FS": gfs},
        })

    # 3e. P-Delta 안정계수 θ (C3)
    pdelta_check = design_check.get("pdelta_check")
    if pdelta_check:
        if pdelta_check["status"] == "NG":
            findings.append({
                "code": "R23",
                "category": "stability",
                "severity": "critical",
                "message_ko": (
                    f"P-Delta 불안정 — θ_max {pdelta_check['max_theta']} > "
                    f"한계 {pdelta_check['theta_limit']}"
                ),
                "message_en": f"P-Delta unstable — θ_max {pdelta_check['max_theta']} > {pdelta_check['theta_limit']}",
                "data": {"max_theta": pdelta_check["max_theta"]},
            })
        elif pdelta_check.get("needs_amplification"):
            findings.append({
                "code": "R23",
                "category": "stability",
                "severity": "caution",
                "message_ko": (
                    f"P-Delta 2차효과 고려 필요 — θ_max {pdelta_check['max_theta']} "
                    f"(>0.1, 증폭 1/(1−θ))"
                ),
                "message_en": f"P-Delta 2nd-order effects — θ_max {pdelta_check['max_theta']} (>0.1)",
                "data": {"max_theta": pdelta_check["max_theta"]},
            })

    # 3f. 비틀림 비정형 (C2)
    torsion_check = design_check.get("torsion_check")
    # 지진검토(drift)는 있는데 비틀림이 미평가(예: V2/RSA 경로 — 다이어프램 변위 극값
    # 미기록)면 누락을 명시(은폐 방지).
    if torsion_check is None and design_check.get("drift_check"):
        findings.append({
            "code": "R25",
            "category": "torsion",
            "severity": "caution",
            "message_ko": "비틀림 비정형(δmax/δavg) 미평가 — 변위증폭(Ax) 미반영 가능",
            "message_en": "Torsional irregularity not evaluated — Ax amplification may be omitted",
            "data": None,
        })
    if torsion_check and torsion_check.get("irregular"):
        extreme = torsion_check.get("extreme")
        findings.append({
            "code": "R24",
            "category": "torsion",
            "severity": "warning" if extreme else "caution",
            "message_ko": (
                f"{'극단 ' if extreme else ''}비틀림 비정형 — δmax/δavg "
                f"{torsion_check['max_ratio']} (Ax 변위증폭 적용)"
            ),
            "message_en": (
                f"{'Extreme ' if extreme else ''}torsional irregularity — "
                f"δmax/δavg {torsion_check['max_ratio']} (Ax applied)"
            ),
            "data": {"max_ratio": torsion_check["max_ratio"], "extreme": extreme},
        })

    # 3g. 풍하중 사용성 층간변위 (#10)
    wind_check = design_check.get("wind_check")
    if wind_check:
        if wind_check["status"] == "NG":
            crit = wind_check.get("critical") or {}
            findings.append({
                "code": "R34",
                "category": "wind_drift",
                "severity": "warning",
                "message_ko": (
                    f"풍하중 사용성 층간변위 초과 — {crit.get('story', '?')}층 "
                    f"{crit.get('direction', '?')}방향 {crit.get('drift_inv', '')} "
                    f"> 1/{wind_check['limit_inv']} (비율 {wind_check['max_ratio']:.2f})"
                ),
                "message_en": (
                    f"Wind serviceability drift exceeds 1/{wind_check['limit_inv']} — "
                    f"story {crit.get('story', '?')} {crit.get('direction', '?')}, "
                    f"ratio {wind_check['max_ratio']:.2f}"
                ),
                "data": {"max_ratio": wind_check["max_ratio"], "limit_inv": wind_check["limit_inv"]},
            })
        else:
            findings.append({
                "code": "R34",
                "category": "wind_drift",
                "severity": "info",
                "message_ko": f"풍하중 사용성 층간변위 OK — 최대 비율 {wind_check['max_ratio']:.2f} (1/{wind_check['limit_inv']})",
                "message_en": f"Wind serviceability drift OK — max ratio {wind_check['max_ratio']:.2f}",
                "data": {"max_ratio": wind_check["max_ratio"]},
            })

    # 3h. 수직 비정형 (#12) — 분류·경고 (overall 미게이트)
    vertical_check = design_check.get("vertical_check")
    if vertical_check and vertical_check.get("irregular"):
        ext = vertical_check.get("extreme")
        types_ko = {"stiffness_soft_story": "강성(연층)", "mass": "중량", "geometric": "기하"}
        kinds = ", ".join(types_ko.get(t, t) for t in vertical_check.get("types", []))
        findings.append({
            "code": "R35",
            "category": "vertical_irregularity",
            "severity": "warning" if ext else "caution",
            "message_ko": (
                f"{'극단 ' if ext else ''}수직 비정형 감지({kinds}) — KDS 41 17 00 표 5.3-2. "
                f"동적해석/내진상세 적용 검토 필요"
            ),
            "message_en": (
                f"{'Extreme ' if ext else ''}vertical irregularity detected ({kinds}) — "
                f"KDS 41 17 00 Table 5.3-2; dynamic analysis / detailing review needed"
            ),
            "data": {"types": vertical_check.get("types"), "extreme": ext},
        })

    # 3i. 우발 비틀림 미적용 (#8) — 등가정적 모델은 ±5% 편심 직접 적용 안 함
    accidental = design_check.get("accidental_torsion")
    if accidental and not accidental.get("applied_in_model"):
        ecc = accidental.get("eccentricity_ratio") or 0.05   # None-safe
        findings.append({
            "code": "R36",
            "category": "torsion",
            "severity": "caution",
            "message_ko": (
                f"우발 비틀림(평면 ±{ecc * 100:g}% 편심) 모멘트 산정됨이나 등가정적 해석모델 미적용 "
                f"— 최종 설계 시 ±{ecc * 100:g}% 우발편심 명시적 반영 권장(KDS 41 17 00 §7.2.6.4)"
            ),
            "message_en": (
                f"Accidental torsion (±{ecc * 100:g}% eccentricity) computed but not applied "
                f"in the equivalent-static model — apply explicitly in final design"
            ),
            "data": {"eccentricity_ratio": ecc,
                     "total_Mt_x_kNm": accidental.get("total_Mt_x_kNm"),
                     "total_Mt_y_kNm": accidental.get("total_Mt_y_kNm")},
        })

    # 3j. 지진입력 손상 → 지진검토 미수행 (#13, 게이트 구멍 정정)
    sumry = design_check.get("summary") or {}
    if sumry.get("seismic_input_error"):
        findings.append({
            "code": "R37",
            "category": "analysis",
            "severity": "warning",
            "message_ko": (
                "지진하중 입력 손상/누락 — 층간변위·내진설계범주·전도·P-Delta 등 지진검토 미수행. "
                "본 결과를 '안전'으로 단정 불가, 지진입력 정정 후 재검토 필요"
            ),
            "message_en": (
                "Seismic load input corrupted/missing — drift, SDC, overturning, P-Delta "
                "checks not performed; cannot conclude 'safe', re-check after fixing input"
            ),
            "data": {"not_checked": sumry.get("not_checked_items", [])},
        })

    # 4. 모달 해석
    modal_interp, modal_findings = _interpret_modal(
        modal_analysis, multi_result, drift_interp,
    )
    findings.extend(modal_findings)

    # 5. 진단 (NG일 때)
    diagnosis = _diagnose_failure(
        severity, drift_interp, member_interp, modal_interp, multi_result,
    )

    # 6. 개선 제안
    suggestions = _suggest_improvements(
        severity, diagnosis, member_interp, drift_interp, multi_result,
    )

    # 7. 요약 생성
    summary_ko, summary_en = _generate_summary(
        severity, design_check, drift_interp, member_interp, diagnosis,
    )

    return {
        "severity": severity,
        "severity_label": {"ko": ko, "en": en},
        "findings": findings,
        "drift_interpretation": drift_interp,
        "member_interpretation": member_interp,
        "deflection_interpretation": deflection_interp,
        "modal_interpretation": modal_interp,
        "diagnosis": diagnosis,
        "suggestions": suggestions,
        "summary_ko": summary_ko,
        "summary_en": summary_en,
    }


# ============================================================
# 1. 심각도 분류
# ============================================================

def _classify_severity(dc: dict) -> str:
    """design_check의 최대 비율로 심각도 판별."""
    # Tier1-4: 해석 실패(솔버 비수렴)면 결과 신뢰 불가 → 최고 심각도.
    if dc.get("analysis_error"):
        return "severe"
    s = dc.get("summary", {})
    dr = s.get("max_drift_ratio") or 0.0
    ir = s.get("max_interaction_ratio") or 0.0
    df = s.get("max_deflection_ratio") or 0.0  # B1: 보 처짐(사용성)
    max_ratio = max(dr, ir, df)

    if max_ratio < _SEVERITY_THRESHOLDS["safe"]:
        sev = "safe"
    elif max_ratio < _SEVERITY_THRESHOLDS["marginal"]:
        sev = "marginal"
    elif max_ratio < _SEVERITY_THRESHOLDS["moderate"]:
        sev = "moderate"
    else:
        sev = "severe"

    # 비율로 잡히지 않는 NG(예: B2 시스템 부적격)도 overall_status가 NG면 최소 moderate로
    # 격상한다 — 그래야 §10 verdict(severity 기반)가 overall_status와 모순되지 않는다.
    if dc.get("overall_status") == "NG" and sev in ("safe", "marginal"):
        sev = "moderate"
    # Tier1-5: RSA 조합 부재검토 미수행이면 '안전(green)'으로 단정하지 않는다.
    # (부재설계가 일부 미수행 → 검토 필요. NG는 아니므로 최소 marginal로만 격상.)
    if sev == "safe" and s.get("rsa_unchecked_combos"):
        sev = "marginal"
    # #13: 지진입력 손상으로 지진검토 미수행 → '안전'으로 단정 금지(게이트 구멍 정정).
    if sev == "safe" and s.get("seismic_input_error"):
        sev = "marginal"
    return sev


# ============================================================
# 2. 층간변위 해석
# ============================================================

def _interpret_drift(drift_check: dict | None) -> tuple[dict | None, list[dict]]:
    """층간변위 패턴 분석."""
    if not drift_check:
        return None, []

    checks = drift_check.get("checks", [])
    if not checks:
        return None, []

    findings: list[dict] = []
    status = drift_check.get("status", "OK")

    # 방향별 층간변위 수집
    drift_by_dir: dict[str, dict[int, float]] = defaultdict(dict)
    for c in checks:
        story = c["story"]
        direction = c["direction"]
        ratio = c.get("ratio", 0.0)
        # 같은 층/방향이 여러 조합에서 나오면 최대값 사용
        if story not in drift_by_dir[direction] or ratio > drift_by_dir[direction][story]:
            drift_by_dir[direction][story] = ratio

    # 방향별 최대 층간변위비 (X/Y 비대칭 판정용 — B4)
    max_ratio_by_dir: dict[str, float] = {
        direction: max(story_ratios.values())
        for direction, story_ratios in drift_by_dir.items()
        if story_ratios
    }

    # 패턴 판별
    pattern = "negligible"
    soft_stories: list[int] = []
    critical_story = drift_check.get("critical", {}).get("story") if drift_check.get("critical") else None
    critical_dir = drift_check.get("critical", {}).get("direction") if drift_check.get("critical") else None

    for direction, story_ratios in drift_by_dir.items():
        sorted_stories = sorted(story_ratios.keys())
        if len(sorted_stories) < 2:
            continue

        ratios = [story_ratios[s] for s in sorted_stories]
        max_ratio = max(ratios)

        if max_ratio < 0.3:
            continue
        pattern = "uniform"

        # Soft story 판별: 해당 층 > 인접 층 평균의 1.5배
        for i, s in enumerate(sorted_stories):
            neighbors = []
            if i > 0:
                neighbors.append(ratios[i - 1])
            if i < len(ratios) - 1:
                neighbors.append(ratios[i + 1])
            if neighbors:
                avg_neighbor = sum(neighbors) / len(neighbors)
                if avg_neighbor > 0 and ratios[i] / avg_neighbor > _SOFT_STORY_RATIO:
                    if s not in soft_stories:
                        soft_stories.append(s)

        # Whipping 판별: 상부 집중
        if len(ratios) >= 4:
            lower_avg = sum(ratios[:2]) / 2
            upper_avg = sum(ratios[-2:]) / 2
            if lower_avg > 0 and upper_avg / lower_avg > _SOFT_STORY_RATIO:
                pattern = "whipping"

    if soft_stories:
        pattern = "soft_story"

    # Findings 생성
    if soft_stories:
        findings.append({
            "code": "R01",
            "category": "drift",
            "severity": "critical",
            "message_ko": f"연약층 감지: {soft_stories}층 — 인접 층 대비 변위 집중",
            "message_en": f"Soft story detected at story {soft_stories} — drift concentration",
            "data": {"soft_stories": soft_stories},
        })

    if status == "NG":
        crit = drift_check.get("critical", {})
        max_r = drift_check.get("max_ratio", 0)
        findings.append({
            "code": "R02",
            "category": "drift",
            "severity": "warning",
            "message_ko": (
                f"층간변위 초과: {crit.get('story', '?')}층 {crit.get('direction', '?')}방향, "
                f"비율 {max_r:.2f} (허용 1.00)"
            ),
            "message_en": (
                f"Drift exceeds limit: story {crit.get('story', '?')} {crit.get('direction', '?')}-dir, "
                f"ratio {max_r:.2f} (limit 1.00)"
            ),
            "data": {"critical": crit, "max_ratio": max_r},
        })
    else:
        max_r = drift_check.get("max_ratio", 0)
        findings.append({
            "code": "R03",
            "category": "drift",
            "severity": "info",
            "message_ko": f"층간변위 OK — 최대 비율 {max_r:.2f}",
            "message_en": f"Drift OK — max ratio {max_r:.2f}",
            "data": {"max_ratio": max_r},
        })

    if pattern == "whipping" and not soft_stories:
        findings.append({
            "code": "R04",
            "category": "drift",
            "severity": "caution",
            "message_ko": "상부 층 변위 집중 (Whipping 패턴) — 상부 강성 재검토 권장",
            "message_en": "Upper story drift concentration (whipping) — review upper stiffness",
            "data": None,
        })

    return {
        "pattern": pattern,
        "critical_story": critical_story,
        "critical_direction": critical_dir,
        "soft_story_detected": len(soft_stories) > 0,
        "soft_story_stories": soft_stories,
        "max_ratio": drift_check.get("max_ratio", 0),
        "max_ratio_by_dir": {k: round(v, 4) for k, v in max_ratio_by_dir.items()},
    }, findings


# ============================================================
# 3. 부재 강도 해석
# ============================================================

def _interpret_members(member_check: dict | None) -> tuple[dict | None, list[dict]]:
    """부재 강도 검토 결과 해석."""
    if not member_check:
        return None, []

    findings: list[dict] = []
    members = member_check.get("members", [])
    summary = member_check.get("summary", {})
    status = member_check.get("status", "OK")

    # Weakest link
    weakest = None
    if members:
        by_ratio = sorted(members, key=lambda m: m.get("ratios", {}).get("interaction", 0), reverse=True)
        top = by_ratio[0]
        weakest = {
            "member_id": top.get("member_id"),
            "type": top.get("type"),
            "story": top.get("story"),
            "ratio": top.get("ratios", {}).get("interaction", 0),
            "formula": top.get("ratios", {}).get("formula", ""),
        }

    # NG 그룹핑
    ng_members = [m for m in members if m.get("status") == "NG"]
    ng_by_story: dict[int, int] = defaultdict(int)
    ng_by_type: dict[str, int] = defaultdict(int)
    for m in ng_members:
        ng_by_story[m.get("story", 0)] += 1
        ng_by_type[m.get("type", "unknown")] += 1

    # 지배 파괴 모드
    governing = "interaction"
    if ng_members:
        max_components = {"axial": 0, "bending_x": 0, "bending_y": 0, "shear": 0}
        for m in ng_members:
            r = m.get("ratios", {})
            for k in max_components:
                max_components[k] = max(max_components[k], r.get(k, 0))
        governing = max(max_components, key=max_components.get)
    elif members:
        # All OK — check which component is closest to limit
        max_components = {"axial": 0, "bending_x": 0, "bending_y": 0, "shear": 0}
        for m in members:
            r = m.get("ratios", {})
            for k in max_components:
                max_components[k] = max(max_components[k], r.get(k, 0))
        governing = max(max_components, key=max_components.get)

    # Findings
    if status == "NG":
        sev = "critical" if summary.get("max_interaction_ratio", 0) > 1.3 else "warning"
        findings.append({
            "code": "R05",
            "category": "member",
            "severity": sev,
            "message_ko": (
                f"부재 {summary.get('ng', 0)}개 NG — "
                f"최대 상관비 {summary.get('max_interaction_ratio', 0):.2f} "
                f"({weakest['type']} #{weakest['member_id']}, {weakest['story']}층)"
            ),
            "message_en": (
                f"{summary.get('ng', 0)} members NG — "
                f"max interaction {summary.get('max_interaction_ratio', 0):.2f} "
                f"({weakest['type']} #{weakest['member_id']}, story {weakest['story']})"
            ),
            "data": {"weakest": weakest, "ng_count": summary.get("ng", 0)},
        })

        _GOVERNING_KO = {
            "axial": "축력", "bending_x": "강축 휨", "bending_y": "약축 휨",
            "shear": "전단", "interaction": "축력-휨 상관",
        }
        findings.append({
            "code": "R06",
            "category": "member",
            "severity": "caution",
            "message_ko": f"지배 파괴 모드: {_GOVERNING_KO.get(governing, governing)}",
            "message_en": f"Governing failure mode: {governing}",
            "data": {"governing": governing},
        })

        # 층 집중 확인
        if ng_by_story:
            worst_story = max(ng_by_story, key=ng_by_story.get)
            if ng_by_story[worst_story] > 1:
                findings.append({
                    "code": "R07",
                    "category": "member",
                    "severity": "warning",
                    "message_ko": f"NG 부재 {worst_story}층 집중 ({ng_by_story[worst_story]}개)",
                    "message_en": f"NG members concentrated at story {worst_story} ({ng_by_story[worst_story]})",
                    "data": {"story": worst_story, "count": ng_by_story[worst_story]},
                })
    else:
        findings.append({
            "code": "R08",
            "category": "member",
            "severity": "info",
            "message_ko": f"부재 강도 OK — 최대 상관비 {summary.get('max_interaction_ratio', 0):.2f}",
            "message_en": f"Members OK — max interaction {summary.get('max_interaction_ratio', 0):.2f}",
            "data": {"max_ratio": summary.get("max_interaction_ratio", 0)},
        })

    # B3: 순인장 부재 존재 시 연결부 순단면파단(D2-b)·블록전단 별도검토 권고
    n_tension = summary.get("n_tension_members", 0)
    if n_tension:
        findings.append({
            "code": "R21",
            "category": "member",
            "severity": "caution",
            "message_ko": (
                f"순인장 부재 {n_tension}개 — 총단면항복(D2-a)+순단면파단(D2-b, Ae/Ag 가정) "
                f"min으로 검토됨. 블록전단·실연결부 상세는 별도 검토 필요"
            ),
            "message_en": (
                f"{n_tension} net-tension members — checked by min(gross yield D2-a, "
                f"net rupture D2-b with assumed Ae/Ag); block shear / connection details "
                f"require separate review"
            ),
            "data": {"n_tension_members": n_tension},
        })

    # Tier1-1: 기둥 유효좌굴길이계수 K>1 (비가새 모멘트골조 sway) 적용 명시.
    col_K = summary.get("column_K")
    if isinstance(col_K, (int, float)) and col_K > 1.0:
        findings.append({
            "code": "R28",
            "category": "member",
            "severity": "info",
            "message_ko": (
                f"기둥 유효좌굴길이계수 K={col_K:g} 적용 (비가새 모멘트골조 sway, AISC) — "
                f"세장비·압축내력에 반영"
            ),
            "message_en": (
                f"Column effective length factor K={col_K:g} applied "
                f"(unbraced moment frame, AISC)"
            ),
            "data": {"column_K": col_K},
        })

    # Tier1-2: 보 횡-비틀림좌굴(LTB) 지배 — 비지지길이 기준 휨내력 감소.
    n_ltb = summary.get("n_ltb_governed", 0)
    if n_ltb:
        findings.append({
            "code": "R29",
            "category": "member",
            "severity": "caution",
            "message_ko": (
                f"보 {n_ltb}개 횡-비틀림좌굴(LTB) 지배 — 비지지길이 Lb 기준 "
                f"강축 휨내력 감소(AISC F2) 적용"
            ),
            "message_en": (
                f"{n_ltb} beams governed by lateral-torsional buckling — reduced "
                f"strong-axis flexural capacity (AISC F2) applied"
            ),
            "data": {"n_ltb_governed": n_ltb},
        })

    # Tier2-6: 세장비 한계 초과 (압축 KL/r>200 / 인장 L/r>300).
    n_sl_ng = summary.get("n_slenderness_ng", 0)
    if n_sl_ng:
        findings.append({
            "code": "R30",
            "category": "member",
            "severity": "warning",
            "message_ko": (
                f"세장비 한계 초과 부재 {n_sl_ng}개 (압축 KL/r≤200·인장 L/r≤300, AISC E2/D1) "
                f"— 최대 세장비 {summary.get('max_slenderness', 0):.0f}"
            ),
            "message_en": (
                f"{n_sl_ng} members exceed slenderness limit (compression KL/r≤200, "
                f"tension L/r≤300) — max {summary.get('max_slenderness', 0):.0f}"
            ),
            "data": {"n": n_sl_ng, "max_slenderness": summary.get("max_slenderness")},
        })

    # Tier2-7: 비콤팩트/세장 단면 (휨 국부좌굴 F3.2 감소 적용).
    n_nc = summary.get("n_noncompact", 0)
    if n_nc:
        findings.append({
            "code": "R31",
            "category": "member",
            "severity": "caution",
            "message_ko": (
                f"비콤팩트/세장 단면 {n_nc}개 — 플랜지국부좌굴(AISC F3.2)로 휨내력 감소 적용"
            ),
            "message_en": (
                f"{n_nc} non-compact/slender sections — flange local buckling (F3.2) "
                f"capacity reduction applied"
            ),
            "data": {"n_noncompact": n_nc},
        })

    # Tier2-7: 압축 세장요소 (E7 추가감소 미반영 — 보수검토 권고).
    n_cs = summary.get("n_compression_slender", 0)
    if n_cs:
        findings.append({
            "code": "R32",
            "category": "member",
            "severity": "caution",
            "message_ko": (
                f"압축 세장요소 단면 {n_cs}개 — 세장판요소 압축감소(AISC E7) 미반영, "
                f"압축내력 보수검토 필요"
            ),
            "message_en": (
                f"{n_cs} sections with slender compression elements — E7 reduction not "
                f"applied; compression capacity needs conservative review"
            ),
            "data": {"n_compression_slender": n_cs},
        })

    # Tier2-15: 인장 순단면파단(D2-b)이 총단면항복보다 지배 (Ae/Ag 가정 기반).
    n_rupt = sum(1 for m in members
                 if (m.get("capacity") or {}).get("tension_rupture_governs"))
    if n_rupt:
        findings.append({
            "code": "R33",
            "category": "member",
            "severity": "caution",
            "message_ko": (
                f"인장 순단면파단(D2-b)이 지배하는 부재 {n_rupt}개 — Ae/Ag 보수 가정 기반, "
                f"실연결부(볼트공·블록전단) 검토 필요"
            ),
            "message_en": (
                f"{n_rupt} members governed by net-section rupture (D2-b) — based on "
                f"assumed Ae/Ag; verify actual connection (bolt holes, block shear)"
            ),
            "data": {"n_rupture_governed": n_rupt},
        })

    # Tier1-5: RSA(응답스펙트럼) 조합 부재검토 미수행 명시 (은폐 방지).
    rsa_unchecked = summary.get("rsa_unchecked_combos") or []
    if rsa_unchecked:
        findings.append({
            "code": "R27",
            "category": "member",
            "severity": "caution",
            "message_ko": (
                f"RSA 조합 {len(rsa_unchecked)}개 부재강도 미검토 — RSA는 변위만 산출, "
                f"부재력 미생성. ELF 조합 기준 부재검토만 수행됨"
            ),
            "message_en": (
                f"{len(rsa_unchecked)} RSA combos not member-checked — RSA gives "
                f"displacement only; member check from ELF combos only"
            ),
            "data": {"rsa_unchecked_combos": rsa_unchecked},
        })

    return {
        "governing_check": governing,
        "weakest_link": weakest,
        "ng_by_story": dict(ng_by_story),
        "ng_by_type": dict(ng_by_type),
        "status": status,
    }, findings


# ============================================================
# 3b. 보 처짐(사용성) 해석 (B1)
# ============================================================

def _interpret_deflection(deflection_check: dict | None) -> tuple[dict | None, list[dict]]:
    """보 수직처짐 사용성 검토 결과 해석. None=미검토."""
    if not deflection_check:
        return None, []

    summary = deflection_check.get("summary", {})
    status = deflection_check.get("status", "OK")
    max_r = summary.get("max_deflection_ratio", 0) or 0
    ng = summary.get("ng", 0)
    findings: list[dict] = []

    if status == "NG":
        beams = [b for b in deflection_check.get("beams", []) if b.get("status") == "NG"]
        worst = beams[0] if beams else {}
        findings.append({
            "code": "R17",
            "category": "deflection",
            "severity": "critical" if max_r > 1.5 else "warning",
            "message_ko": (
                f"보 처짐 사용성 NG — {ng}개 보, 최대 처짐비 {max_r:.2f} (허용 1.00, "
                f"활하중 L/360·전체 L/240)"
            ),
            "message_en": (
                f"Beam deflection serviceability NG — {ng} beams, max ratio {max_r:.2f} "
                f"(limit 1.00)"
            ),
            "data": {"ng": ng, "max_ratio": max_r, "worst_member": worst.get("member_id")},
        })
    else:
        findings.append({
            "code": "R18",
            "category": "deflection",
            "severity": "info",
            "message_ko": f"보 처짐 사용성 OK — 최대 처짐비 {max_r:.2f} (L/360·L/240)",
            "message_en": f"Beam deflection OK — max ratio {max_r:.2f}",
            "data": {"max_ratio": max_r},
        })

    return {"status": status, "max_ratio": round(max_r, 4), "ng": ng}, findings


# ============================================================
# 4. 모달 해석
# ============================================================

def _interpret_modal(
    modal_analysis: dict | None,
    multi_result,
    drift_interp: dict | None = None,
) -> tuple[dict | None, list[dict]]:
    """모달 해석 결과 해석."""
    if not modal_analysis or not modal_analysis.get("modes"):
        return None, []

    findings: list[dict] = []

    # 총 높이 (m)
    total_height_m = sum(getattr(multi_result, 'stories', []) or [])
    if total_height_m <= 0:
        return None, []

    # 경험적 고유주기 (강재 모멘트골조 기본)
    Ct, x = _EMPIRICAL_PERIOD.get("steel_moment_frame", (0.085, 0.75))
    Ta = Ct * (total_height_m ** x)

    # 실제 1차 주기 (X, Y 중 큰 값)
    fp = modal_analysis.get("fundamental_periods", {})
    T1_x = fp.get("T1_x_s", 0)
    T1_y = fp.get("T1_y_s", 0)
    T1 = max(T1_x, T1_y)
    T1_Ta_ratio = T1 / Ta if Ta > 0 else 0

    # 1차 모드 방향
    modes = modal_analysis.get("modes", [])
    first_dir = modes[0]["direction"] if modes else "N/A"

    flexibility_flag = T1_Ta_ratio > _T1_TA_FLEXIBILITY
    torsion_risk = first_dir == "ROTN-Z"

    # 모달-변위 상관관계
    drift_modal_corr = None
    if drift_interp and drift_interp.get("critical_direction"):
        crit_dir = drift_interp["critical_direction"]
        mode_dir_mapped = "X" if first_dir == "TRAN-X" else "Y" if first_dir == "TRAN-Y" else None
        if mode_dir_mapped == crit_dir:
            drift_modal_corr = f"{crit_dir}-direction"

    # Findings
    if flexibility_flag:
        findings.append({
            "code": "R09",
            "category": "modal",
            "severity": "warning",
            "message_ko": (
                f"구조물 과유연: T1={T1:.3f}s, 경험식 Ta={Ta:.3f}s, "
                f"비율 {T1_Ta_ratio:.2f} (> {_T1_TA_FLEXIBILITY})"
            ),
            "message_en": (
                f"Structure too flexible: T1={T1:.3f}s, empirical Ta={Ta:.3f}s, "
                f"ratio {T1_Ta_ratio:.2f} (> {_T1_TA_FLEXIBILITY})"
            ),
            "data": {"T1": T1, "Ta": Ta, "ratio": round(T1_Ta_ratio, 2)},
        })

    if torsion_risk:
        findings.append({
            "code": "R10",
            "category": "modal",
            "severity": "warning",
            "message_ko": "1차 모드가 비틀림(ROTN-Z) — 비틀림 비정형 위험",
            "message_en": "1st mode is torsional (ROTN-Z) — torsional irregularity risk",
            "data": None,
        })

    if drift_modal_corr:
        findings.append({
            "code": "R11",
            "category": "modal",
            "severity": "caution",
            "message_ko": f"1차 모드 방향({first_dir})이 층간변위 NG 방향({drift_modal_corr})과 일치",
            "message_en": f"1st mode direction ({first_dir}) correlates with drift NG ({drift_modal_corr})",
            "data": {"mode_dir": first_dir, "drift_dir": drift_modal_corr},
        })

    if not flexibility_flag and not torsion_risk:
        findings.append({
            "code": "R12",
            "category": "modal",
            "severity": "info",
            "message_ko": f"모달 결과 정상 — T1={T1:.3f}s (경험식 Ta={Ta:.3f}s)",
            "message_en": f"Modal results normal — T1={T1:.3f}s (empirical Ta={Ta:.3f}s)",
            "data": {"T1": T1, "Ta": Ta, "ratio": round(T1_Ta_ratio, 2)},
        })

    # 누적 유효질량 참여율 (KDS 41 17 00 충분조건 ≥90%) — V1: cumulative_participation,
    # V2: 마지막 모드의 cumulative_*_pct. 둘 다 방어적으로 처리.
    cum = modal_analysis.get("cumulative_participation")
    if not cum and modes:
        last = modes[-1]
        cum = {"x_pct": last.get("cumulative_x_pct"), "y_pct": last.get("cumulative_y_pct")}
    cum = cum or {}
    cx = cum.get("x_pct")
    cy = cum.get("y_pct")
    mass_x = round(cx, 1) if isinstance(cx, (int, float)) else None
    mass_y = round(cy, 1) if isinstance(cy, (int, float)) else None
    mass_ok = (mass_x is not None and mass_y is not None and mass_x >= 90 and mass_y >= 90)

    # A3: 누적 유효질량 참여율 < 90% → 모드 수 부족 경고 (KDS 41 17 00 충분조건 미달).
    if mass_x is not None and mass_y is not None and not mass_ok:
        findings.append({
            "code": "R16",
            "category": "modal",
            "severity": "caution",
            "message_ko": (
                f"누적 유효질량 참여율 부족: X {mass_x:.0f}% / Y {mass_y:.0f}% "
                f"(< 90%) — 모드 수 확대 검토 필요"
            ),
            "message_en": (
                f"Cumulative effective mass participation low: X {mass_x:.0f}% / "
                f"Y {mass_y:.0f}% (< 90%) — increase number of modes"
            ),
            "data": {"mass_x_pct": mass_x, "mass_y_pct": mass_y},
        })

    return {
        "T1_actual_s": round(T1, 4),
        "Ta_empirical_s": round(Ta, 4),
        "T1_Ta_ratio": round(T1_Ta_ratio, 2),
        "flexibility_flag": flexibility_flag,
        "torsion_risk": torsion_risk,
        "first_mode_direction": first_dir,
        "drift_modal_correlation": drift_modal_corr,
        "cumulative_mass_x_pct": mass_x,
        "cumulative_mass_y_pct": mass_y,
        "mass_participation_sufficient": mass_ok,
    }, findings


# ============================================================
# 5. 진단
# ============================================================

def _diagnose_failure(
    severity: str,
    drift_interp: dict | None,
    member_interp: dict | None,
    modal_interp: dict | None,
    multi_result,
) -> dict | None:
    """NG 원인 진단. safe이면 None 반환."""
    if severity == "safe":
        return None

    primary = None
    factors: list[str] = []
    factors_ko: list[str] = []

    # 주 원인 결정 (우선순위: soft_story > torsion > lateral_stiffness > member_capacity)
    if drift_interp and drift_interp.get("soft_story_detected"):
        primary = "soft_story"
    elif modal_interp and modal_interp.get("torsion_risk"):
        primary = "torsion"
    elif drift_interp and drift_interp.get("max_ratio", 0) > 1.0:
        primary = "lateral_stiffness"
    elif member_interp and member_interp.get("status") == "NG":
        primary = "member_capacity"
    else:
        # marginal — drift가 지배적이면 lateral, 아니면 member
        dr = drift_interp.get("max_ratio", 0) if drift_interp else 0
        mr = member_interp.get("weakest_link", {}).get("ratio", 0) if member_interp else 0
        primary = "lateral_stiffness" if dr >= mr else "member_capacity"

    # Contributing factors
    stories = getattr(multi_result, 'stories', []) or []
    if any(h > 4.5 for h in stories):
        high_stories = [i + 1 for i, h in enumerate(stories) if h > 4.5]
        factors.append(f"High story height at story {high_stories} (> 4.5m)")
        factors_ko.append(f"{high_stories}층 높은 층고 (> 4.5m)")

    bays_x = getattr(multi_result, 'bays_x', []) or []
    bays_y = getattr(multi_result, 'bays_y', []) or []
    all_bays = bays_x + bays_y
    if all_bays and max(all_bays) > 1.5 * min(all_bays):
        factors.append("Asymmetric bay widths")
        factors_ko.append("비대칭 경간 폭")

    if modal_interp and modal_interp.get("flexibility_flag") and primary != "lateral_stiffness":
        factors.append(f"Structure too flexible (T1/Ta = {modal_interp['T1_Ta_ratio']:.2f})")
        factors_ko.append(f"과유연 구조 (T1/Ta = {modal_interp['T1_Ta_ratio']:.2f})")

    if member_interp and member_interp.get("ng_by_story"):
        worst = max(member_interp["ng_by_story"], key=member_interp["ng_by_story"].get)
        if member_interp["ng_by_story"][worst] >= 2 and primary != "soft_story":
            factors.append(f"Member failures concentrated at story {worst}")
            factors_ko.append(f"{worst}층 부재 파괴 집중")

    cause_ko, cause_en = _CAUSE_LABELS.get(primary, (primary, primary))

    return {
        "primary_cause": primary,
        "primary_cause_ko": cause_ko,
        "primary_cause_en": cause_en,
        "contributing_factors": factors,
        "contributing_factors_ko": factors_ko,
    }


# ============================================================
# 6. 개선 제안
# ============================================================

def _suggest_improvements(
    severity: str,
    diagnosis: dict | None,
    member_interp: dict | None,
    drift_interp: dict | None,
    multi_result,
) -> list[dict]:
    """개선 제안 생성. safe이면 빈 리스트."""
    if severity == "safe":
        return []

    suggestions: list[dict] = []
    cause = diagnosis.get("primary_cause") if diagnosis else None

    col_section = getattr(multi_result, 'column_section', None) or ""
    beam_x_section = getattr(multi_result, 'beam_x_section', None) or ""

    # NG 집중 층 / weakest link 정보 수집
    _ng_detail_ko = ""
    _ng_detail_en = ""
    if member_interp:
        wk = member_interp.get("weakest_link")
        ngs = member_interp.get("ng_by_story", {})
        details = []
        details_en = []
        if ngs:
            worst_s = max(ngs, key=ngs.get)
            details.append(f"{worst_s}층 NG {ngs[worst_s]}개 집중")
            details_en.append(f"{ngs[worst_s]} NG at story {worst_s}")
        if wk and wk.get("ratio", 0) > 0:
            details.append(f"최대 상관비 {wk['ratio']:.2f}")
            details_en.append(f"max ratio {wk['ratio']:.2f}")
        if details:
            _ng_detail_ko = " (" + ", ".join(details) + ")"
            _ng_detail_en = " (" + ", ".join(details_en) + ")"

    # Drift NG 상세
    _drift_detail_ko = ""
    _drift_detail_en = ""
    if drift_interp and drift_interp.get("max_ratio", 0) > 1.0:
        cs = drift_interp.get("critical_story", "?")
        cd = drift_interp.get("critical_direction", "?")
        dr = drift_interp.get("max_ratio", 0)
        _drift_detail_ko = f" ({cs}층 {cd}방향, 비율 {dr:.2f})"
        _drift_detail_en = f" (story {cs} {cd}-dir, ratio {dr:.2f})"

    # 단면 업그레이드 — 변위 또는 부재 NG 시
    if cause in ("lateral_stiffness", "soft_story", "member_capacity", "torsion", None):
        # 기둥 업그레이드 (횡강성/부재강도 모두에 효과적)
        next_col = _find_next_section(col_section, _COLUMN_UPGRADE_SERIES)
        if next_col:
            extra_ko = _drift_detail_ko if cause == "lateral_stiffness" else _ng_detail_ko
            extra_en = _drift_detail_en if cause == "lateral_stiffness" else _ng_detail_en
            suggestions.append({
                "priority": 1,
                "type": "section_upgrade",
                "target": "column",
                "current": col_section,
                "recommended": next_col,
                "message_ko": f"기둥 단면 업그레이드: {col_section} → {next_col}{extra_ko}",
                "message_en": f"Upgrade column section: {col_section} → {next_col}{extra_en}",
                "expected_impact": "high",
            })

    # 부재 NG가 보에서 집중된 경우
    if member_interp and member_interp.get("ng_by_type", {}).get("beam_x", 0) > 0:
        next_bm = _find_next_section(beam_x_section, _BEAM_UPGRADE_SERIES)
        if next_bm:
            ng_bx = member_interp["ng_by_type"]["beam_x"]
            suggestions.append({
                "priority": 2,
                "type": "section_upgrade",
                "target": "beam_x",
                "current": beam_x_section,
                "recommended": next_bm,
                "message_ko": f"X방향 보 단면 업그레이드: {beam_x_section} → {next_bm} (beam_x NG {ng_bx}개)",
                "message_en": f"Upgrade beam-X section: {beam_x_section} → {next_bm} ({ng_bx} beam_x NG)",
                "expected_impact": "medium",
            })

    # Soft story — 해당 층 보강
    if cause == "soft_story" and drift_interp:
        ss = drift_interp.get("soft_story_stories", [])
        if ss:
            suggestions.append({
                "priority": 1,
                "type": "geometry_review",
                "target": f"story_{ss}",
                "current": None,
                "recommended": None,
                "message_ko": f"{ss}층 보강: 가새 추가 또는 기둥 단면 증가 검토",
                "message_en": f"Reinforce story {ss}: consider adding bracing or increasing column size",
                "expected_impact": "high",
            })

    # Severe — 구조시스템 변경
    if severity == "severe":
        suggestions.append({
            "priority": 3,
            "type": "system_change",
            "target": "structural_system",
            "current": None,
            "recommended": None,
            "message_ko": "단면 보강만으로 부족 — 가새골조/전단벽 추가 등 구조시스템 변경 검토",
            "message_en": "Section upgrade insufficient — consider adding bracing or shear walls",
            "expected_impact": "high",
        })

    # Marginal — 일반 경고
    if severity == "marginal" and not suggestions:
        suggestions.append({
            "priority": 3,
            "type": "geometry_review",
            "target": "general",
            "current": None,
            "recommended": None,
            "message_ko": "여유 부족 — 향후 하중 변경 시 단면 검토 권장",
            "message_en": "Limited margin — review sections if loads change",
            "expected_impact": "low",
        })

    suggestions.sort(key=lambda s: s["priority"])
    return suggestions


def _find_next_section(current: str, series: list[str]) -> str | None:
    """시리즈에서 현재 단면의 다음(더 큰) 단면을 찾는다.

    정확 매칭 실패 시 prefix 매칭 시도.
    예: "H-428x407x20x35"는 시리즈의 "H-428x407"과 매칭.
    """
    if not current:
        return None
    # 정확 매칭
    try:
        idx = series.index(current)
        if idx < len(series) - 1:
            return series[idx + 1]
        return None
    except ValueError:
        pass
    # prefix 매칭: current가 시리즈 항목으로 시작하거나, 시리즈 항목이 current로 시작
    for i, s in enumerate(series):
        if current.startswith(s) or s.startswith(current):
            if i < len(series) - 1:
                return series[i + 1]
            return None
    return None


# ============================================================
# 7. 요약 생성
# ============================================================

def _generate_summary(
    severity: str,
    design_check: dict,
    drift_interp: dict | None,
    member_interp: dict | None,
    diagnosis: dict | None,
) -> tuple[str, str]:
    """한국어/영어 1~2문장 요약."""
    s = design_check.get("summary", {})
    dr = s.get("max_drift_ratio") or 0.0
    ir = s.get("max_interaction_ratio") or 0.0
    df = s.get("max_deflection_ratio") or 0.0  # B1: 보 처짐
    util = max(dr, ir, df)
    ng_stories = s.get("ng_stories", 0)
    ng_members = s.get("ng_members", 0)

    if severity == "safe":
        ko = f"모든 설계검토 통과. 최대 활용률 {util:.0%}."
        en = f"All design checks passed. Max utilization {util:.0%}."
    elif severity == "marginal":
        ko = f"설계검토 통과, 최대 활용률 {util:.0%}으로 여유 부족. 하중 변경 시 재검토 필요."
        en = f"Design checks passed, max utilization {util:.0%} with limited margin."
    elif severity == "moderate":
        cause_ko = diagnosis.get("primary_cause_ko", "") if diagnosis else ""
        ko = (
            f"설계검토 부적합 (변위 NG {ng_stories}층, 부재 NG {ng_members}개). "
            f"주 원인: {cause_ko}. 단면 보강으로 해결 가능."
        )
        cause_en = diagnosis.get("primary_cause_en", "") if diagnosis else ""
        en = (
            f"Design check NG (drift NG {ng_stories} stories, member NG {ng_members}). "
            f"Primary cause: {cause_en}. Resolvable with section upgrade."
        )
    else:  # severe
        cause_ko = diagnosis.get("primary_cause_ko", "") if diagnosis else ""
        ko = (
            f"설계검토 부적합 (변위 NG {ng_stories}층, 부재 NG {ng_members}개). "
            f"주 원인: {cause_ko}. 구조시스템 변경 검토 필요."
        )
        cause_en = diagnosis.get("primary_cause_en", "") if diagnosis else ""
        en = (
            f"Design check NG (drift NG {ng_stories} stories, member NG {ng_members}). "
            f"Primary cause: {cause_en}. Structural system change needed."
        )

    return ko, en
