"""KDS 설계 검토 모듈.

KDS 41 17 00 (내진설계) 층간변위 검토 + KDS 41 31 00 (강구조) 부재 강도 검토.
AISC 360 H1 조합응력비 (Interaction) 포함.

Usage:
    from core.design_check import run_design_check
    result = run_design_check(multi_result, building_model, seismic_report)

반환 형식:
    {
        "overall_status": "OK" | "NG",
        "drift_check": {...},
        "member_check": {...},
        "critical_issues": [...],
        "summary": {...},
    }
"""
from __future__ import annotations

import logging
import math
from typing import Optional

_log = logging.getLogger(__name__)

# ============================================================
# 상수
# ============================================================

PHI = 0.9  # 강도감소계수 (KDS 41 31 00: φc=φb=φv=0.9)

# KDS 41 17 00 §8.2.3 표 8.2-1: 허용 층간변위비
DRIFT_LIMITS = {
    "특": 0.010,
    "I": 0.015,
    "II": 0.020,
}


# ============================================================
# 단면 설계 물성 계산
# ============================================================

def _compute_design_props(
    A: float, Ix: float, Iy: float,
    h: float, b: float,
    tw: float, tf: float,
) -> dict:
    """설계용 단면 물성 계산.

    Args:
        A: 단면적 (mm²)
        Ix, Iy: 단면2차모멘트 (mm⁴)
        h, b: 단면 높이/폭 (mm)
        tw, tf: 웹/플랜지 두께 (mm), 0이면 미확인

    Returns:
        dict with rx, ry, Sx, Sy, Zx, Zy, Aw (모두 mm 단위 기반)
    """
    if A <= 0 or h <= 0:
        return {"rx": 0, "ry": 0, "Sx": 0, "Sy": 0, "Zx": 0, "Zy": 0, "Aw": 0}

    rx = math.sqrt(Ix / A)
    ry = math.sqrt(Iy / A) if Iy > 0 else 0.0

    Sx = Ix / (h / 2)                     # 탄성 단면계수 (강축, mm³)
    Sy = Iy / (b / 2) if b > 0 else 0.0   # 탄성 단면계수 (약축, mm³)

    if tw > 0 and tf > 0:
        hw = h - 2 * tf
        if hw < 0:
            hw = 0.0
        Zx = b * tf * (h - tf) + tw * hw ** 2 / 4      # 소성 단면계수 (강축)
        Zy = 2 * tf * (b / 2) ** 2 / 2 + hw * tw ** 2 / 4  # 소성 단면계수 (약축)
        Aw = hw * tw                                     # 전단면적
    else:
        # tw/tf 미확인 — 보수적 근사
        Zx = Sx * 1.12   # H형강 평균 shape factor ≈ 1.12
        Zy = Sy * 1.12
        Aw = A / 3.0     # 근사 (A의 약 1/3이 웹)

    return {
        "rx": rx, "ry": ry,
        "Sx": Sx, "Sy": Sy,
        "Zx": Zx, "Zy": Zy,
        "Aw": Aw,
    }


# ============================================================
# 강도 계산 함수 (KDS 41 31 00 / AISC 360)
# ============================================================

def _compression_capacity(
    fy: float, E: float, A: float,
    r_min: float, KL_mm: float,
) -> float:
    """AISC E3: 압축 공칭 강도 φPn (kN).

    λ = KL/r → Fe = π²E/λ²
    Fy/Fe ≤ 2.25 → Fcr = 0.658^(Fy/Fe) × Fy
    Fy/Fe > 2.25 → Fcr = 0.877 × Fe
    φPn = 0.9 × Fcr × A / 1000
    """
    if r_min <= 0 or A <= 0 or KL_mm <= 0:
        return 0.0

    lam = KL_mm / r_min  # 세장비
    Fe = math.pi ** 2 * E / lam ** 2  # 오일러 좌굴 응력

    if Fe <= 0:
        return 0.0

    ratio = fy / Fe
    if ratio <= 2.25:
        Fcr = (0.658 ** ratio) * fy
    else:
        Fcr = 0.877 * Fe

    return PHI * Fcr * A / 1000  # kN


def _bending_capacity(fy: float, Zx: float) -> float:
    """AISC F2: 휨 공칭 강도 φMn (kN·m).

    compact section + 횡지지 가정.
    Mn = Fy × Zx → φMn = 0.9 × Fy × Zx / 1e6
    """
    if Zx <= 0:
        return 0.0
    return PHI * fy * Zx / 1e6  # kN·m


def _shear_capacity(fy: float, Aw: float) -> float:
    """AISC G2: 전단 공칭 강도 φVn (kN).

    Vn = 0.6 × Fy × Aw → φVn = 0.9 × 0.6 × Fy × Aw / 1000
    """
    if Aw <= 0:
        return 0.0
    return PHI * 0.6 * fy * Aw / 1000  # kN


def _interaction_H1(
    Pu: float, phiPn: float,
    Mux: float, phiMnx: float,
    Muy: float, phiMny: float,
) -> tuple[float, str]:
    """AISC H1.1: 축력-휨 상관비.

    Pu/φPn ≥ 0.2 → H1-1a: Pu/φPn + 8/9 × (Mux/φMnx + Muy/φMny) ≤ 1.0
    Pu/φPn < 0.2 → H1-1b: Pu/(2φPn) + (Mux/φMnx + Muy/φMny) ≤ 1.0

    Returns:
        (interaction_ratio, formula_name)
    """
    if phiPn <= 0:
        # 압축 강도 0 → 휨만 검토
        bending = 0.0
        if phiMnx > 0:
            bending += abs(Mux) / phiMnx
        if phiMny > 0:
            bending += abs(Muy) / phiMny
        return (bending, "bending_only")

    axial_ratio = abs(Pu) / phiPn
    bending_x = abs(Mux) / phiMnx if phiMnx > 0 else 0.0
    bending_y = abs(Muy) / phiMny if phiMny > 0 else 0.0

    if axial_ratio >= 0.2:
        # H1-1a
        ratio = axial_ratio + (8.0 / 9.0) * (bending_x + bending_y)
        return (ratio, "H1-1a")
    else:
        # H1-1b
        ratio = axial_ratio / 2.0 + (bending_x + bending_y)
        return (ratio, "H1-1b")


# ============================================================
# Phase 1: 층간변위 검토
# ============================================================

def check_story_drifts(
    combo_results: dict,
    stories: list[float],
    Cd: float,
    IE: float,
    importance: str = "II",
) -> dict:
    """KDS 41 17 00 §8.2.3 층간변위 검토.

    지진 하중조합에 대해 비탄성 변위를 계산하고 허용치와 비교.

    Args:
        combo_results: {combo_name: Frame3DCaseResult}
        stories: 층고 리스트 (m)
        Cd: 변위증폭계수
        IE: 중요도계수
        importance: 중요도 등급 ("특", "I", "II")

    Returns:
        {status, code_ref, Cd, IE, allowable, checks, critical, max_ratio}
    """
    allowable = DRIFT_LIMITS.get(importance, 0.020)
    checks = []
    max_ratio = 0.0
    critical = None

    # 지진 조합만 검토 (combo 이름에 "EQ" 포함)
    eq_combos = [name for name in combo_results if "EQ" in name]

    for combo_name in eq_combos:
        cr = combo_results[combo_name]
        if not hasattr(cr, "story_drifts"):
            continue

        for sd in cr.story_drifts:
            story = sd["story"]
            height_m = sd["height_m"]

            for direction, drift_key in [("X", "drift_x"), ("Y", "drift_y")]:
                elastic_drift = sd.get(drift_key, 0.0)
                inelastic_drift = Cd * elastic_drift / IE if IE > 0 else 0.0
                ratio = inelastic_drift / allowable if allowable > 0 else 0.0
                status = "OK" if ratio <= 1.0 else "NG"

                # 사람이 읽기 쉬운 형식
                drift_inv = f"1/{int(1/inelastic_drift)}" if inelastic_drift > 1e-8 else "0"

                check = {
                    "story": story,
                    "height_m": height_m,
                    "direction": direction,
                    "combo": combo_name,
                    "elastic_drift": round(elastic_drift, 6),
                    "inelastic_drift": round(inelastic_drift, 6),
                    "allowable": allowable,
                    "ratio": round(ratio, 4),
                    "status": status,
                    "drift_inv": drift_inv,
                    "message": (
                        f"{story}층 {direction}방향: "
                        f"Cd×δe/IE = {Cd}×{elastic_drift:.6f}/{IE} = "
                        f"{inelastic_drift:.6f} "
                        f"{'>' if ratio > 1.0 else '<'} {allowable} "
                        f"({status})"
                    ),
                }
                checks.append(check)

                if ratio > max_ratio:
                    max_ratio = ratio
                    critical = {
                        "story": story,
                        "direction": direction,
                        "combo": combo_name,
                        "ratio": round(ratio, 4),
                        "inelastic_drift": round(inelastic_drift, 6),
                    }

    overall = "OK" if max_ratio <= 1.0 else "NG"

    return {
        "status": overall,
        "code_ref": "KDS 41 17 00 §8.2.3",
        "Cd": Cd,
        "IE": IE,
        "importance": importance,
        "allowable": allowable,
        "checks": checks,
        "critical": critical,
        "max_ratio": round(max_ratio, 4),
    }


# ============================================================
# Phase 2: 부재 강도 검토
# ============================================================

def _get_section_props_for_type(multi, member_type: str) -> dict:
    """Frame3DMultiCaseResult에서 부재 타입별 단면 물성 조회."""
    prefix = member_type  # "column", "beam_x", "beam_y"
    return {
        "A": getattr(multi, f"{prefix}_A_mm2", 0),
        "Ix": getattr(multi, f"{prefix}_Ix_mm4", 0),
        "Iy": getattr(multi, f"{prefix}_Iy_mm4", 0),
        "h": getattr(multi, f"{prefix}_h_mm", 0),
        "b": getattr(multi, f"{prefix}_b_mm", 0),
        "tw": getattr(multi, f"{prefix}_tw_mm", 0),
        "tf": getattr(multi, f"{prefix}_tf_mm", 0),
    }


def _envelope_member_forces(
    member_forces: dict,
    member_id: int,
    combo_names: list[str],
) -> tuple[dict, str]:
    """모든 조합에서 부재별 envelope 힘과 지배 조합 반환.

    Returns:
        ({Pu, Mux, Muy, Vu}, governing_combo)
    """
    max_interaction = 0.0
    governing_combo = ""
    envelope = {"Pu": 0.0, "Mux": 0.0, "Muy": 0.0, "Vu": 0.0}

    for combo_name in combo_names:
        mf_list = member_forces.get(combo_name, [])
        for mf in mf_list:
            if mf["member_id"] != member_id:
                continue

            # 배열에서 최대 절대값 추출
            N_max = max(abs(v) for v in mf.get("N_kN", [0]))
            My_max = max(abs(v) for v in mf.get("My_kNm", [0]))
            Mz_max = max(abs(v) for v in mf.get("Mz_kNm", [0]))
            Vy_max = max(abs(v) for v in mf.get("Vy_kN", [0]))
            Vz_max = max(abs(v) for v in mf.get("Vz_kN", [0]))

            # 전단: 양방향 중 최대
            V_max = max(Vy_max, Vz_max)

            # 간이 상관비로 지배 조합 판정
            combined = N_max + My_max + Mz_max
            if combined > max_interaction:
                max_interaction = combined
                governing_combo = combo_name
                envelope = {
                    "Pu": round(N_max, 2),
                    "Mux": round(My_max, 4),
                    "Muy": round(Mz_max, 4),
                    "Vu": round(V_max, 2),
                }
            break  # member_id는 combo당 1개

    return envelope, governing_combo


def _infer_member_story(member_info: dict, stories: list[float]) -> int:
    """부재의 층 번호 추정.

    Column: member_id 순서 → (n_cols_x × n_cols_y) 그룹별 층 배정
    Beam: 유사하게 추정, 정확하지 않을 수 있음
    """
    # member_info에 직접적인 story 정보 없으므로 간단 추정:
    # column은 0-based로 n_cols 개씩 묶이고, beam은 1층부터 시작
    # 정확한 매핑을 위해서는 member_id와 node_grid 매핑이 필요하지만,
    # 여기서는 간단히 member_id 기반으로 추정
    mid = member_info.get("member_id", 0)
    mtype = member_info.get("type", "")

    if mtype == "column":
        # 기둥은 stories 수 × grid 크기로 배정
        # 정확한 값은 아니지만, 검토 보고서용으로 충분
        return (mid % len(stories)) + 1 if stories else 1
    else:
        return (mid % len(stories)) + 1 if stories else 1


def check_member_strengths(multi_result, fy_MPa: float, E_MPa: float) -> dict:
    """KDS 41 31 00 / AISC 360 부재 강도 검토.

    Args:
        multi_result: Frame3DMultiCaseResult
        fy_MPa: 항복강도 (MPa)
        E_MPa: 탄성계수 (MPa)

    Returns:
        {status, code_ref, members, critical_members, summary}
    """
    members = []
    member_info_list = multi_result.member_info
    stories = multi_result.stories

    # 모든 조합 이름 (case + combo)
    combo_names = list(multi_result.combo_results.keys())
    if not combo_names:
        # combo가 없으면 case_results 사용
        combo_names = list(multi_result.case_results.keys())

    # 단면별 설계 물성 캐시 (member section이 부재마다 다를 수 있어 mtype-cache로는
    # 부족함. Phase B 단면 변경 후 변경 부재만 capacity가 새 단면 기준이어야 한다).
    # mtype 폴백은 section_name이 비었을 때만 사용 (정상 분석에선 일어나지 않음).
    section_props_cache: dict[str, dict] = {}
    mtype_fallback_cache: dict[str, dict] = {}
    for mtype in ["column", "beam_x", "beam_y"]:
        sp = _get_section_props_for_type(multi_result, mtype)
        mtype_fallback_cache[mtype] = _compute_design_props(
            sp["A"], sp["Ix"], sp["Iy"],
            sp["h"], sp["b"], sp["tw"], sp["tf"],
        )
        mtype_fallback_cache[mtype]["A"] = sp["A"]
        mtype_fallback_cache[mtype]["section_name"] = getattr(
            multi_result, f"{mtype}_section", ""
        )

    def _props_for(section_name: str, mtype: str) -> dict:
        if section_name and section_name in section_props_cache:
            return section_props_cache[section_name]
        if section_name:
            try:
                from core.section_3d import get_section_3d
                sec = get_section_3d(section_name)
                dp = _compute_design_props(
                    sec.A, sec.Ix, sec.Iy,
                    sec.h, sec.b, sec.tw, sec.tf,
                )
                dp["A"] = sec.A
                dp["section_name"] = section_name
                section_props_cache[section_name] = dp
                return dp
            except Exception as exc:
                _log.warning(
                    "design_check: 단면 %s 조회 실패 (%s) — mtype fallback",
                    section_name, exc,
                )
        return mtype_fallback_cache.get(mtype, {})

    for minfo in member_info_list:
        mid = minfo["member_id"]
        mtype = minfo["type"]
        length_m = minfo.get("length_m", 3.5)
        section_name = minfo.get("section", "")
        KL_mm = length_m * 1000  # K = 1.0

        dp = _props_for(section_name, mtype)
        A = dp.get("A", 0)

        if A <= 0:
            continue

        # Envelope 힘
        envelope, governing_combo = _envelope_member_forces(
            multi_result.member_forces, mid, combo_names,
        )

        # 강도 계산
        # Column: 양축 + 압축, Beam: 강축 휨 + 전단
        rx = dp.get("rx", 0)
        ry = dp.get("ry", 0)
        Zx = dp.get("Zx", 0)
        Zy = dp.get("Zy", 0)
        Aw = dp.get("Aw", 0)

        if mtype == "column":
            # 기둥: 약축 좌굴 지배
            r_gov = ry if ry > 0 else rx
            phiPn = _compression_capacity(fy_MPa, E_MPa, A, r_gov, KL_mm)
            phiMnx = _bending_capacity(fy_MPa, Zx)
            phiMny = _bending_capacity(fy_MPa, Zy)
        else:
            # 보: 축력 무시 (횡하중 시 소량 존재하지만 미미)
            # 슬래브 구속으로 약축 휨 무시
            phiPn = _compression_capacity(fy_MPa, E_MPa, A, ry if ry > 0 else rx, KL_mm)
            phiMnx = _bending_capacity(fy_MPa, Zx)
            phiMny = _bending_capacity(fy_MPa, Zy)

        phiVn = _shear_capacity(fy_MPa, Aw)

        # 개별 비율
        axial_ratio = abs(envelope["Pu"]) / phiPn if phiPn > 0 else 0.0
        bending_x_ratio = abs(envelope["Mux"]) / phiMnx if phiMnx > 0 else 0.0
        bending_y_ratio = abs(envelope["Muy"]) / phiMny if phiMny > 0 else 0.0
        shear_ratio = abs(envelope["Vu"]) / phiVn if phiVn > 0 else 0.0

        # H1 interaction
        interaction, formula = _interaction_H1(
            envelope["Pu"], phiPn,
            envelope["Mux"], phiMnx,
            envelope["Muy"], phiMny,
        )

        # 지배 비율 (interaction vs shear 중 큰 것)
        max_ratio = max(interaction, shear_ratio)
        status = "OK" if max_ratio <= 1.0 else "NG"

        story = _infer_member_story(minfo, stories)

        members.append({
            "member_id": mid,
            "type": mtype,
            "section": section_name,
            "story": story,
            "governing_combo": governing_combo,
            "demand": envelope,
            "capacity": {
                "phiPn_kN": round(phiPn, 1),
                "phiMnx_kNm": round(phiMnx, 2),
                "phiMny_kNm": round(phiMny, 2),
                "phiVn_kN": round(phiVn, 1),
            },
            "ratios": {
                "axial": round(axial_ratio, 4),
                "bending_x": round(bending_x_ratio, 4),
                "bending_y": round(bending_y_ratio, 4),
                "shear": round(shear_ratio, 4),
                "interaction": round(interaction, 4),
                "formula": formula,
            },
            "status": status,
        })

    # 정렬: interaction 비율 내림차순
    members.sort(key=lambda m: m["ratios"]["interaction"], reverse=True)

    # Critical members: 상위 5개
    critical_members = members[:5]

    # Summary
    ng_count = sum(1 for m in members if m["status"] == "NG")
    max_ratio = members[0]["ratios"]["interaction"] if members else 0.0
    max_shear_ratio = max(
        (m["ratios"]["shear"] for m in members), default=0.0
    )

    overall = "OK" if ng_count == 0 else "NG"

    return {
        "status": overall,
        "code_ref": "KDS 41 31 00 / AISC 360 H1",
        "members": members,
        "critical_members": critical_members,
        "summary": {
            "total": len(members),
            "ok": len(members) - ng_count,
            "ng": ng_count,
            "max_interaction_ratio": round(max_ratio, 4),
            "max_shear_ratio": round(max_shear_ratio, 4),
        },
        "assumptions": [
            "K = 1.0 (유효좌굴길이계수: 보수적 가정, 비가새 골조는 K>1 필요)",
            "Compact section 가정 (LTB 미검토, 슬래브 횡지지)",
            "φ = 0.9 (KDS 41 31 00 표준 강도감소계수)",
        ],
    }


# ============================================================
# Phase 3: 통합 설계 검토
# ============================================================

def run_design_check(
    multi_result,
    building_model=None,
    seismic_report: Optional[dict] = None,
) -> dict:
    """전체 설계 검토 실행 (층간변위 + 부재강도).

    Args:
        multi_result: Frame3DMultiCaseResult
        building_model: BuildingModel (importance, seismic_system 등)
        seismic_report: load_result["reports"]["seismic"] (Cd, IE 포함)

    Returns:
        {overall_status, drift_check, member_check, critical_issues, summary}
    """
    fy = multi_result.fy_MPa
    E = multi_result.E_MPa
    stories = multi_result.stories

    # ── 1. 층간변위 검토 ──
    drift_check = None
    if seismic_report and multi_result.combo_results:
        Cd = seismic_report.get("Cd", 3.0)
        IE = seismic_report.get("IE", 1.0)
        importance = "II"
        if building_model:
            importance = getattr(building_model, "importance", "II")

        drift_check = check_story_drifts(
            multi_result.combo_results, stories, Cd, IE, importance,
        )

    # ── 2. 부재 강도 검토 ──
    member_check = None
    if multi_result.member_info and (multi_result.member_forces or multi_result.combo_results):
        member_check = check_member_strengths(multi_result, fy, E)

    # ── 3. Critical issues 취합 ──
    critical_issues = []

    if drift_check and drift_check["status"] == "NG":
        for chk in drift_check["checks"]:
            if chk["status"] == "NG":
                critical_issues.append({
                    "type": "drift_exceeded",
                    "description": (
                        f"{chk['story']}층 {chk['direction']}방향 "
                        f"비탄성 층간변위비 {chk['inelastic_drift']:.5f} > "
                        f"허용 {chk['allowable']} ({chk['drift_inv']})"
                    ),
                    "code_ref": "KDS 41 17 00 §8.2.3",
                    "combo": chk["combo"],
                })

    if member_check and member_check["status"] == "NG":
        for m in member_check["members"]:
            if m["status"] == "NG":
                critical_issues.append({
                    "type": "member_overstressed",
                    "description": (
                        f"부재 {m['member_id']} ({m['type']}, {m['section']}) "
                        f"상관비 {m['ratios']['interaction']:.3f} > 1.0 "
                        f"({m['ratios']['formula']})"
                    ),
                    "code_ref": "KDS 41 31 00 / AISC 360 H1",
                    "combo": m["governing_combo"],
                })

    # ── 4. Overall status ──
    drift_ok = drift_check is None or drift_check["status"] == "OK"
    member_ok = member_check is None or member_check["status"] == "OK"
    overall = "OK" if (drift_ok and member_ok) else "NG"

    # ── 5. Summary ──
    summary = {
        "max_drift_ratio": drift_check["max_ratio"] if drift_check else None,
        "max_interaction_ratio": (
            member_check["summary"]["max_interaction_ratio"]
            if member_check else None
        ),
        "ng_stories": (
            len([c for c in drift_check["checks"] if c["status"] == "NG"])
            if drift_check else 0
        ),
        "ng_members": member_check["summary"]["ng"] if member_check else 0,
    }

    return {
        "overall_status": overall,
        "drift_check": drift_check,
        "member_check": member_check,
        "critical_issues": critical_issues,
        "summary": summary,
    }
