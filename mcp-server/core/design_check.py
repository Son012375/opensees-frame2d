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
import re
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
# Tier2-14: JSON 직렬화 안전 — NaN/Inf 정리
# ============================================================

def _sanitize_nonfinite(obj):
    """dict/list 내 NaN/Inf float를 None으로 치환 (재귀, in-place 비변경).

    웹앱이 json.dumps(design_check) + FastAPI 직렬화를 타므로 NaN/Inf가 있으면
    비표준 JSON('NaN' 토큰)으로 500을 유발할 수 있다. 결과 dict 경계에서 정리한다.
    bool은 보존(int 하위형이지만 그대로), 키는 그대로 둔다(상위에서 문자열 보장).
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _sanitize_nonfinite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_nonfinite(v) for v in obj]
    return obj


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
        # 약축 소성단면계수: 플랜지 2개(각 tf·b²/4) + 웹(hw·tw²/4).
        # (구버전 2·tf·(b/2)²/2 = tf·b²/4 는 플랜지 1개만 세어 약 50% 과소 → Zy<Sy 비물리.)
        Zy = tf * b ** 2 / 2 + hw * tw ** 2 / 4        # 소성 단면계수 (약축)
        Aw = hw * tw                                     # 전단면적
    else:
        # tw/tf 미확인 — 보수적 근사
        Zx = Sx * 1.12   # H형강 평균 shape factor ≈ 1.12
        Zy = Sy * 1.12
        Aw = A / 3.0     # 근사 (A의 약 1/3이 웹)

    # 소성단면계수는 탄성단면계수 이상이어야 함(물리 제약) — 형상 근사 오차 방어.
    Zx = max(Zx, Sx)
    Zy = max(Zy, Sy)

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
    """AISC F2: 휨 공칭 강도 φMn (kN·m) — 완전소성(Mp), 완전횡지지 가정.

    compact section + 연속 횡지지 가정.
    Mn = Fy × Zx → φMn = 0.9 × Fy × Zx / 1e6

    주의: 횡-비틀림좌굴(LTB)은 미고려. 비지지길이가 있는 보는
    `_ltb_bending_capacity`를 사용한다(슬래브 연속횡지지가 없으면 LTB 지배).
    """
    if Zx <= 0:
        return 0.0
    return PHI * fy * Zx / 1e6  # kN·m


# ============================================================
# Tier1-1: 유효좌굴길이계수 K (AISC) — 모멘트(비가새)골조 비보수 정정
# ============================================================

# 가새골조·전단벽(비횡요동)은 K≈1.0. 비가새 모멘트골조(sway)는 K>1
# (정렬도표 K≈1.2~2.0+). K=1.0 고정은 모멘트골조에서 세장비 과소→φPn 과대→
# 비보수. AISC 보수 하한으로 모멘트(미상 포함)골조에 K=1.2를 적용한다.
_BRACED_SYSTEMS = {
    "special_braced_frame", "ordinary_braced_frame",
    "rc_special_shear_wall", "rc_ordinary_shear_wall",
}
K_MOMENT_FRAME = 1.2   # 비가새(sway) 모멘트골조 보수 하한 (AISC)
K_BRACED = 1.0         # 가새골조·전단벽 (비횡요동)


def _effective_length_factor(seismic_system: Optional[str], member_type: str) -> float:
    """기둥 유효좌굴길이계수 K. 보/가새는 1.0, 기둥은 골조형식별.

    가새골조·전단벽 → K=1.0(비횡요동). 모멘트골조 또는 미상 → K=1.2
    (비가새 sway 보수 하한; K=1.0은 비보수이므로 미상도 sway로 본다).
    """
    if member_type != "column":
        return 1.0
    sys_key = (seismic_system or "").strip().lower()
    if sys_key in _BRACED_SYSTEMS:
        return K_BRACED
    return K_MOMENT_FRAME


# ============================================================
# Tier1-2: 보 횡-비틀림좌굴 LTB (AISC F2) — Lb/Lp/Lr/Cb
# ============================================================

def _ltb_bending_capacity(
    fy: float, E: float, Lb_mm: float,
    Zx: float, Sx: float, ry: float,
    Iy: float, J: float, ho: float,
    Cb: float = 1.0,
) -> tuple[float, dict]:
    """AISC F2 강축 휨 공칭강도 φMn (kN·m) + LTB 진단 dict (2축대칭 I형강).

    Lb=비지지길이. Lp/Lr 산정 후 구간별 Mn = min(Mp, Mcr).
      Lb ≤ Lp        : Mn = Mp = Fy·Zx (소성, 횡지지 충분)
      Lp < Lb ≤ Lr   : Mn = Cb[Mp − (Mp−0.7Fy·Sx)(Lb−Lp)/(Lr−Lp)] ≤ Mp (비탄성 LTB)
      Lb > Lr        : Mn = Fcr·Sx ≤ Mp (탄성 LTB)
    rts² = √(Iy·Cw)/Sx, 2축대칭 I형강 Cw = Iy·ho²/4 → rts = √(Iy·ho/(2Sx)).
    ho = 플랜지 도심간 거리(≈ h−tf). Cb=1.0 보수(균등모멘트). c=1.0(2축대칭).

    Returns:
        (phiMn_kNm, info). 입력 부족/비물리 시 Mp(φMp)로 폴백(zone="undetermined").
        info의 모든 키/값은 JSON-safe (문자열 키, 유한 float).
    """
    Mp = fy * Zx  # N·mm
    phiMp = PHI * Mp / 1e6
    if Zx <= 0:
        return 0.0, {"zone": "undetermined", "governs": False}
    # 비지지길이 없음(연속 횡지지) 또는 단면물성 부족 → Mp
    if (Lb_mm is None or Lb_mm <= 0 or ry <= 0 or Sx <= 0
            or Iy <= 0 or ho <= 0):
        return phiMp, {"zone": "braced", "governs": False,
                       "phiMn_kNm": round(phiMp, 2), "phiMp_kNm": round(phiMp, 2)}

    c = 1.0
    Lp = 1.76 * ry * math.sqrt(E / fy)
    rts = math.sqrt(Iy * ho / (2.0 * Sx))
    term = (J * c) / (Sx * ho)          # 1/mm²
    fyr = 0.7 * fy
    Lr = (1.95 * rts * (E / fyr)
          * math.sqrt(term + math.sqrt(term * term + 6.76 * (fyr / E) ** 2)))

    _undet = {"zone": "undetermined", "governs": False,
              "phiMn_kNm": round(phiMp, 2), "phiMp_kNm": round(phiMp, 2)}
    # 물리적으로 Lr>Lp(실강재). 퇴화 단면(데이터 이상)으로 Lr≤Lp면 비탄성구간 분모가
    # 음/0 → 신뢰 불가 → 보수적으로 Mp 폴백(0/Inf/NaN 직렬화 위험 차단).
    if not math.isfinite(Lr) or not math.isfinite(Lp) or Lr <= Lp:
        return phiMp, _undet

    if Lb_mm <= Lp:
        Mn, zone = Mp, "plastic"
    elif Lb_mm <= Lr:
        Mn = min(Cb * (Mp - (Mp - fyr * Sx) * (Lb_mm - Lp) / (Lr - Lp)), Mp)
        zone = "inelastic_LTB"
    else:
        lam = Lb_mm / rts
        Fcr = (Cb * math.pi ** 2 * E / (lam * lam)) * math.sqrt(1.0 + 0.078 * term * lam * lam)
        Mn = min(Fcr * Sx, Mp)
        zone = "elastic_LTB"
    Mn = max(Mn, 0.0)
    phiMn = PHI * Mn / 1e6
    if not math.isfinite(phiMn):    # 어떤 경로로든 비유한값이면 보수적 폴백
        return phiMp, _undet
    return phiMn, {
        "zone": zone,
        "governs": zone != "plastic",
        "Lb_m": round(Lb_mm / 1000.0, 3),
        "Lp_m": round(Lp / 1000.0, 3),
        "Lr_m": round(Lr / 1000.0, 3),
        "Cb": Cb,
        "phiMn_kNm": round(phiMn, 2),
        "phiMp_kNm": round(phiMp, 2),
    }


def _resolve_beam_Lb_mm(minfo: dict, length_mm: float, beam_unbraced: bool):
    """보 비지지길이 Lb(mm) 결정.

    1) member_info["unbraced_length_m"] 가 양수면 그 값.
    2) member_info["laterally_braced"] is False 또는 beam_unbraced=True → 부재 전길이(보수).
    3) 그 외(기본) → None = 슬래브 연속 횡지지 가정(LTB 미감소, Mp).

    슬래브 연속횡지지가 기본인 이유: 바닥보 상부플랜지는 슬래브가 연속 구속.
    중간 횡지지가 없거나 보수검토가 필요하면 beam_unbraced=True로 전길이 LTB.
    """
    ub = minfo.get("unbraced_length_m")
    if isinstance(ub, (int, float)) and ub > 0:
        return ub * 1000.0
    if minfo.get("laterally_braced") is False or beam_unbraced:
        return length_mm
    return None


# ============================================================
# Tier2-6: 세장비 한계 (AISC E2 압축 / D1 인장)
# ============================================================

SLENDER_LIMIT_COMP = 200   # AISC E2 사용자주석: 압축재 KL/r 권고상한
SLENDER_LIMIT_TENS = 300   # AISC D1 사용자주석: 인장재 L/r 권고상한


# ============================================================
# Tier2-7: 단면 콤팩트 분류 (AISC B4) + 플랜지 국부좌굴 (F3.2)
# ============================================================

def _classify_compactness(fy: float, E: float, b: float, tf: float,
                          h: float, tw: float) -> dict:
    """AISC B4 휨 콤팩트 분류 (2축대칭 I형강 플랜지·웹). 입력부족 시 compact 가정.

    플랜지(B4.1b case 10): λ=bf/2tf, λpf=0.38√(E/Fy), λrf=1.0√(E/Fy).
    웹(case 15): λ=h/tw(h≈d−2tf 순간격), λpw=3.76√(E/Fy), λrw=5.70√(E/Fy).
    압축 세장요소(B4.1a): 플랜지 λr=0.56√, 웹 λr=1.49√.
    반환 dict는 JSON-safe(문자열/유한값).
    """
    if fy <= 0 or E <= 0 or tf <= 0 or tw <= 0 or b <= 0 or h <= 0:
        return {"flange": "compact", "web": "compact", "section": "compact",
                "flange_lambda": None, "web_lambda": None,
                "compression_slender": False}
    sq = math.sqrt(E / fy)
    lam_f = b / (2.0 * tf)
    lpf, lrf = 0.38 * sq, 1.0 * sq
    hw = max(h - 2.0 * tf, 0.0)
    lam_w = hw / tw if tw > 0 else 0.0
    lpw, lrw = 3.76 * sq, 5.70 * sq

    def _cls(l, lp, lr):
        return "compact" if l <= lp else ("noncompact" if l <= lr else "slender")

    fclass, wclass = _cls(lam_f, lpf, lrf), _cls(lam_w, lpw, lrw)
    order = {"compact": 0, "noncompact": 1, "slender": 2}
    section = fclass if order[fclass] >= order[wclass] else wclass
    comp_slender = (lam_f > 0.56 * sq) or (lam_w > 1.49 * sq)
    return {
        "flange": fclass, "web": wclass, "section": section,
        "flange_lambda": round(lam_f, 2), "web_lambda": round(lam_w, 2),
        "lambda_pf": round(lpf, 2), "lambda_rf": round(lrf, 2),
        "compression_slender": comp_slender,
    }


def _flb_capacity(fy: float, E: float, Zx: float, Sx: float,
                  b: float, tf: float, h: float, tw: float, comp: dict) -> float:
    """AISC F3.2 압축플랜지 국부좌굴 φMn (kN·m). compact 플랜지면 φMp.

    noncompact: Mn = Mp − (Mp−0.7Fy·Sx)(λ−λpf)/(λrf−λpf).
    slender:    Mn = 0.9·E·kc·Sx/λ² (kc=4/√(h/tw), 0.35≤kc≤0.76).
    Mn ≤ Mp. 비유한/입력부족 시 보수적으로 φMp 폴백.
    """
    Mp = fy * Zx
    phiMp = PHI * Mp / 1e6
    if not comp or comp.get("flange") == "compact" or Sx <= 0 or tf <= 0 or b <= 0 or fy <= 0:
        return phiMp
    sq = math.sqrt(E / fy)
    lam = b / (2.0 * tf)
    lpf, lrf = 0.38 * sq, 1.0 * sq
    if comp.get("flange") == "noncompact":
        if lrf <= lpf:
            return phiMp
        Mn = Mp - (Mp - 0.7 * fy * Sx) * ((lam - lpf) / (lrf - lpf))
    else:  # slender
        hw = max(h - 2.0 * tf, 0.0)
        kc = 4.0 / math.sqrt(hw / tw) if (tw > 0 and hw > 0) else 0.76
        kc = min(max(kc, 0.35), 0.76)
        Mn = 0.9 * E * kc * Sx / (lam * lam) if lam > 0 else Mp
    Mn = max(min(Mn, Mp), 0.0)
    val = PHI * Mn / 1e6
    return val if math.isfinite(val) else phiMp


# ============================================================
# Tier2-15: 인장 순단면 파단 (AISC D2-b) — 연결부 정보 부재 시 보수 가정
# ============================================================

DEFAULT_NET_AREA_RATIO = 0.85   # Ae/Ag 기본(연결부 미상 → 보수 가정, KDS/AISC 통상범위)
_FU_FROM_GRADE = {
    "SS235": 330, "SS275": 410, "SS315": 490, "SS410": 540, "SS400": 400,
    "SM275": 410, "SM355": 490, "SM420": 520, "SM460": 570, "SM490": 490,
    "SN275": 410, "SN355": 490, "SN460": 570,
    "SHN275": 410, "SHN355": 490, "SHN420": 520,
    "SMA275": 410, "SMA355": 490,
}


def _estimate_fu(fy: float, material_name=None) -> float:
    """인장강도 Fu(MPa). 강종표 우선, 실패 시 Fu≈1.5·Fy 보수 추정."""
    if material_name:
        import re
        key = str(material_name).upper().replace("-", "").replace(" ", "")
        m = re.match(r"(S[A-Z]+\d{3})", key)
        if m and m.group(1) in _FU_FROM_GRADE:
            return float(_FU_FROM_GRADE[m.group(1)])
    return round(fy * 1.5, 1) if fy > 0 else 0.0


def _tension_rupture_capacity(fu: float, A: float, ae_ag: float) -> float:
    """AISC D2-b 순단면 파단 인장내력 φt·Fu·Ae (kN). φt=0.75, Ae=ae_ag·Ag."""
    if fu <= 0 or A <= 0 or ae_ag <= 0:
        return 0.0
    return 0.75 * fu * (ae_ag * A) / 1000.0


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
    phiPn_tens: float = 0.0,
) -> tuple[float, str]:
    """AISC H1.1/H1.2: 축력-휨 상관비.

    Pu/Pc ≥ 0.2 → H1-1a: Pu/Pc + 8/9 × (Mux/φMnx + Muy/φMny) ≤ 1.0
    Pu/Pc < 0.2 → H1-1b: Pu/(2Pc) + (Mux/φMnx + Muy/φMny) ≤ 1.0

    B3: Pu는 **부호있는 축력**(압축>0, 인장<0). 순인장(Pu<0)이고 인장내력
    phiPn_tens가 주어지면 Pc=phiPn_tens(=φt·Fy·Ag, AISC H1.2)를 사용한다.
    그 외(압축 또는 phiPn_tens 미제공)는 Pc=phiPn(압축 Fcr). phiPn_tens 기본 0.0 →
    하위호환(기존 압축 전용 동작 유지).

    Returns:
        (interaction_ratio, formula_name)  — 인장이면 formula에 "(T)" 표기
    """
    is_tension = Pu < 0 and phiPn_tens > 0
    Pc = phiPn_tens if is_tension else phiPn
    tsuf = "(T)" if is_tension else ""

    if Pc <= 0:
        # 축강도 0 → 휨만 검토
        bending = 0.0
        if phiMnx > 0:
            bending += abs(Mux) / phiMnx
        if phiMny > 0:
            bending += abs(Muy) / phiMny
        return (bending, "bending_only")

    axial_ratio = abs(Pu) / Pc
    bending_x = abs(Mux) / phiMnx if phiMnx > 0 else 0.0
    bending_y = abs(Muy) / phiMny if phiMny > 0 else 0.0

    if axial_ratio >= 0.2:
        # H1-1a
        ratio = axial_ratio + (8.0 / 9.0) * (bending_x + bending_y)
        return (ratio, "H1-1a" + tsuf)
    else:
        # H1-1b
        ratio = axial_ratio / 2.0 + (bending_x + bending_y)
        return (ratio, "H1-1b" + tsuf)


# ============================================================
# Phase 1: 층간변위 검토
# ============================================================

def check_story_drifts(
    combo_results: dict,
    stories: list[float],
    Cd: float,
    IE: float,
    importance: str = "II",
    ax_map: Optional[dict] = None,
) -> dict:
    """KDS 41 17 00 §8.2.3 층간변위 검토.

    지진 하중조합에 대해 비탄성 변위를 계산하고 허용치와 비교.

    Args:
        combo_results: {combo_name: Frame3DCaseResult}
        stories: 층고 리스트 (m)
        Cd: 변위증폭계수
        IE: 중요도계수
        importance: 중요도 등급 ("특", "I", "II")
        ax_map: C2 비틀림 변위증폭계수 {(combo, story, dir): Ax}. 비틀림 비정형 시
            inelastic_drift = Ax·Cd·δe/Ie 로 증폭한다. None이면 Ax=1.0.

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
                # C2: 비틀림 변위증폭계수 Ax (해당 조합·층·방향)
                ax = ax_map.get(_ax_key(combo_name, story, direction), 1.0) if ax_map else 1.0
                if ax and ax > 1.0:
                    inelastic_drift *= ax
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
                    "ax": round(ax, 3),  # C2: 적용된 비틀림 변위증폭계수
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
# Phase C2: 비틀림 비정형 + 변위증폭계수 Ax (KDS 41 17 00 표 5.3-1 H-1)
# ============================================================

TORSION_IRREGULAR_RATIO = 1.2   # 비틀림 비정형 (δmax/δavg)
TORSION_EXTREME_RATIO = 1.4     # 극단 비틀림 비정형


def _ax_key(combo: str, story: int, direction: str) -> str:
    """비틀림 Ax 맵 키 (JSON 직렬화 가능한 문자열 — 튜플 키 금지)."""
    return f"{combo}␟{story}␟{direction}"


def _torsion_ratio(d_max: float, d_min: float) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """두 극단 다이어프램 변위 → (δmax, δavg, ratio). 우세 방향을 +로 정렬.

    δavg = (δmax+δmin)/2 (KDS/ASCE 양단 평균). δavg≤0이면 (None,None,None).
    """
    a, b = d_max, d_min
    if a + b < 0:    # ±EQ 방향 정렬 (우세 변위를 +로)
        a, b = -a, -b
    dmax = max(a, b)
    dmin = min(a, b)
    avg = (dmax + dmin) / 2.0
    if avg <= 1e-9:
        return None, None, None
    return dmax, avg, dmax / avg


def check_torsional_irregularity(combo_results: dict) -> Optional[dict]:
    """비틀림 비정형 정량 + 변위증폭계수 Ax (KDS 41 17 00 표 5.3-1 H-1).

    각 지진조합·층·하중방향에서 다이어프램 양단 변위 δmax/δavg 를 산정한다.
    >1.2 비틀림 비정형, >1.4 극단. Ax = clamp((δmax/(1.2·δavg))², 1, 3).
    반환 ax_map[(combo,story,dir)]=Ax 를 check_story_drifts가 비탄성변위 증폭에 사용.
    story_drifts에 disp_* 극값이 없으면(예: V2 경로) None(미검토).

    Returns:
        {status, code_ref, checks, max_ratio, classification, irregular, ax_map, assumptions} | None
    """
    eq_combos = [name for name in combo_results if "EQ" in name]
    if not eq_combos:
        return None

    checks = []
    ax_map: dict = {}
    max_ratio = 0.0
    any_disp = False
    for combo_name in eq_combos:
        dir_ = "X" if "EQX" in combo_name else ("Y" if "EQY" in combo_name else None)
        if dir_ is None:
            continue
        cr = combo_results[combo_name]
        for sd in getattr(cr, "story_drifts", []) or []:
            kmax = "disp_x_max" if dir_ == "X" else "disp_y_max"
            kmin = "disp_x_min" if dir_ == "X" else "disp_y_min"
            if sd.get(kmax) is None:
                continue
            any_disp = True
            dmax, davg, ratio = _torsion_ratio(sd[kmax], sd[kmin])
            if ratio is None:
                continue
            Ax = min(max((ratio / TORSION_IRREGULAR_RATIO) ** 2, 1.0), 3.0)
            if ratio >= TORSION_EXTREME_RATIO:
                cls = "extreme"
            elif ratio >= TORSION_IRREGULAR_RATIO:
                cls = "torsional"
            else:
                cls = "regular"
            # JSON 직렬화 위해 문자열 키 사용 (튜플 키 금지 — 리포트/API 직렬화 호환).
            ax_map[_ax_key(combo_name, sd["story"], dir_)] = Ax
            checks.append({
                "combo": combo_name, "story": sd["story"], "direction": dir_,
                "delta_max": round(dmax, 4), "delta_avg": round(davg, 4),
                "ratio": round(ratio, 3), "classification": cls, "Ax": round(Ax, 3),
            })
            max_ratio = max(max_ratio, ratio)

    if not any_disp or not checks:
        return None

    checks.sort(key=lambda c: c["ratio"], reverse=True)
    if max_ratio >= TORSION_EXTREME_RATIO:
        classification = "extreme"
    elif max_ratio >= TORSION_IRREGULAR_RATIO:
        classification = "torsional"
    else:
        classification = "regular"

    return {
        "status": "OK",  # 비틀림 분류 자체는 판정이 아님 — 효과는 Ax→증폭변위→층간변위 검토
        "code_ref": "KDS 41 17 00 표 5.3-1 H-1 (비틀림 비정형)",
        "checks": checks,
        "critical": checks[0],
        "max_ratio": round(max_ratio, 3),
        "classification": classification,
        "irregular": max_ratio >= TORSION_IRREGULAR_RATIO,
        "extreme": max_ratio >= TORSION_EXTREME_RATIO,
        "ax_map": ax_map,
        "assumptions": [
            "δmax/δavg: 다이어프램 양단(절점 변위 극값) 기준 (KDS 41 17 00 H-1)",
            "Ax = clamp((δmax/1.2·δavg)², 1, 3) → 비탄성 층간변위에 증폭 적용",
            "우발편심 0.05L 별도 미반영 (해석모델 실변위 기반)",
        ],
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
        "J": getattr(multi, f"{prefix}_J_mm4", 0),
        "h": getattr(multi, f"{prefix}_h_mm", 0),
        "b": getattr(multi, f"{prefix}_b_mm", 0),
        "tw": getattr(multi, f"{prefix}_tw_mm", 0),
        "tf": getattr(multi, f"{prefix}_tf_mm", 0),
    }


def _governing_member_forces(
    member_forces: dict,
    member_id: int,
    combo_names: list[str],
    phiPn: float,
    phiMnx: float,
    phiMny: float,
    phiVn: float,
    phiPn_tens: float = 0.0,
) -> dict:
    """조합별 **실제 H1 상관비**로 지배조합을 선정한다.

    구버전은 N(kN)+My(kN·m)+Mz(kN·m)를 더한 차원불일치 간이지표로 지배조합을
    골라, 축력이 큰 중력조합이 거의 항상 선택되고 모멘트가 큰 지진조합의 H1이
    검토되지 않았다(부재가 실제 NG여도 OK로 보고). 여기서는 각 조합의 H1 상관비를
    실제 능력(phi*)으로 계산해 최댓값 조합을 지배조합으로 삼고, 전단비는 조합별
    독립 최댓값으로 별도 집계한다(상관·전단은 서로 다른 조합이 지배할 수 있음).

    A5: demand(Pu/Mux/Muy)는 H1지배 '단일조합 스냅샷'이라 축력·강축·약축의 최댓값이
    서로 다른 조합에 있을 때 성분별 worst를 놓칠 수 있다. 이를 보강하기 위해 조합
    전체에서 성분별 최댓값(+해당 조합)을 `demand_envelope`로 함께 반환한다(보수적
    참고값 — 판정에 쓰이는 interaction은 B3대로 조합별 실제 H1을 유지한다).

    Returns:
        {Pu, Mux, Muy, Vu, governing_combo, interaction, formula,
         axial, bending_x, bending_y, shear_ratio, shear_governing_combo,
         demand_envelope}
    """
    best = {
        "Pu": 0.0, "Mux": 0.0, "Muy": 0.0, "Vu": 0.0,
        "governing_combo": "", "interaction": 0.0, "formula": "",
        "axial": 0.0, "bending_x": 0.0, "bending_y": 0.0,
        "N_signed": 0.0, "axial_mode": "compression",
    }
    max_shear = 0.0
    shear_combo = ""
    # A5: 성분별 worst envelope (각 성분의 최댓값과 그 조합)
    env = {"N": 0.0, "N_combo": "", "My": 0.0, "My_combo": "",
           "Mz": 0.0, "Mz_combo": "", "V": 0.0, "V_combo": ""}

    for combo_name in combo_names:
        for mf in member_forces.get(combo_name, []):
            if mf["member_id"] != member_id:
                continue
            Nvals = mf.get("N_kN", [0]) or [0]
            # B3: 최대 크기 스테이션의 '부호있는' 축력 (압축>0, 인장<0)
            N_signed = max(Nvals, key=abs)
            N_max = abs(N_signed)
            My_max = max(abs(v) for v in mf.get("My_kNm", [0]))
            Mz_max = max(abs(v) for v in mf.get("Mz_kNm", [0]))
            V_max = max(
                max(abs(v) for v in mf.get("Vy_kN", [0])),
                max(abs(v) for v in mf.get("Vz_kN", [0])),
            )

            inter, formula = _interaction_H1(
                N_signed, phiPn, My_max, phiMnx, Mz_max, phiMny, phiPn_tens=phiPn_tens,
            )
            if inter > best["interaction"]:
                is_tension = N_signed < 0 and phiPn_tens > 0
                Pc = phiPn_tens if is_tension else phiPn
                best = {
                    "Pu": round(N_max, 2), "Mux": round(My_max, 4),
                    "Muy": round(Mz_max, 4), "Vu": round(V_max, 2),
                    "N_signed": round(N_signed, 2),
                    "axial_mode": "tension" if N_signed < 0 else "compression",
                    "governing_combo": combo_name,
                    "interaction": inter, "formula": formula,
                    "axial": N_max / Pc if Pc > 0 else 0.0,
                    "bending_x": abs(My_max) / phiMnx if phiMnx > 0 else 0.0,
                    "bending_y": abs(Mz_max) / phiMny if phiMny > 0 else 0.0,
                }
            sr = V_max / phiVn if phiVn > 0 else 0.0
            if sr > max_shear:
                max_shear = sr
                shear_combo = combo_name
            # A5: 성분별 worst 갱신 (서로 다른 조합일 수 있음)
            if N_max > env["N"]:
                env["N"], env["N_combo"] = N_max, combo_name
            if My_max > env["My"]:
                env["My"], env["My_combo"] = My_max, combo_name
            if Mz_max > env["Mz"]:
                env["Mz"], env["Mz_combo"] = Mz_max, combo_name
            if V_max > env["V"]:
                env["V"], env["V_combo"] = V_max, combo_name
            break  # member_id는 combo당 1개

    best["shear_ratio"] = max_shear
    best["shear_governing_combo"] = shear_combo
    best["demand_envelope"] = {
        "N_max_kN": round(env["N"], 2), "N_combo": env["N_combo"],
        "Mux_max_kNm": round(env["My"], 4), "Mux_combo": env["My_combo"],
        "Muy_max_kNm": round(env["Mz"], 4), "Muy_combo": env["Mz_combo"],
        "Vu_max_kN": round(env["V"], 2), "Vu_combo": env["V_combo"],
    }
    return best


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


def check_member_strengths(multi_result, fy_MPa: float, E_MPa: float,
                           pdelta_amp_by_story: Optional[dict] = None,
                           seismic_system: Optional[str] = None,
                           beam_unbraced: bool = False) -> dict:
    """KDS 41 31 00 / AISC 360 부재 강도 검토.

    Args:
        multi_result: Frame3DMultiCaseResult
        fy_MPa: 항복강도 (MPa)
        E_MPa: 탄성계수 (MPa)
        pdelta_amp_by_story: C3 P-Delta 2차효과 증폭계수 {story: 1/(1−θ)}. 해당 층
            부재의 상관비·전단비에 증폭 적용 (KDS 41 17 00 §7.2.7). None이면 미적용.
        seismic_system: 횡력저항시스템 키 (Tier1-1: 기둥 유효좌굴길이계수 K 산정).
            가새/전단벽 → K=1.0, 모멘트골조·미상 → K=1.2.
        beam_unbraced: True면 모든 보를 비지지(Lb=부재길이)로 보아 LTB 검토(보수).
            기본 False(슬래브 연속 횡지지). member_info의 unbraced_length_m/
            laterally_braced로 부재별 재정의 가능 (Tier1-2).

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
    if not combo_names:
        # 둘 다 비었으나 member_forces가 있으면 그 키를 사용 (부재력 무시 방지).
        combo_names = list((multi_result.member_forces or {}).keys())

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
        mtype_fallback_cache[mtype].update({
            "A": sp["A"], "Iy": sp["Iy"], "Ix": sp["Ix"], "J": sp.get("J", 0),
            "h": sp["h"], "b": sp["b"], "tf": sp["tf"], "tw": sp["tw"],
            "section_name": getattr(multi_result, f"{mtype}_section", ""),
        })

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
                dp.update({
                    "A": sec.A, "Iy": sec.Iy, "Ix": sec.Ix,
                    "J": getattr(sec, "J", 0) or 0,
                    "h": getattr(sec, "h", 0) or 0,
                    "b": getattr(sec, "b", 0) or 0,
                    "tf": getattr(sec, "tf", 0) or 0,
                    "tw": getattr(sec, "tw", 0) or 0,
                    "section_name": section_name,
                })
                section_props_cache[section_name] = dp
                return dp
            except Exception as exc:
                _log.warning(
                    "design_check: 단면 %s 조회 실패 (%s) — mtype fallback",
                    section_name, exc,
                )
        return mtype_fallback_cache.get(mtype, {})

    n_ltb_governed = 0
    n_slenderness_ng = 0
    n_noncompact = 0
    n_comp_slender = 0
    max_slenderness = 0.0
    # Tier2-15: 인장강도 Fu (강종/추정) — 순단면파단 내력용 (부재 공통).
    fu_MPa = _estimate_fu(fy_MPa, getattr(multi_result, "material_name", None))
    for minfo in member_info_list:
        mid = minfo["member_id"]
        mtype = minfo["type"]
        length_m = minfo.get("length_m", 3.5)
        section_name = minfo.get("section", "")
        length_mm = length_m * 1000
        # Tier1-1: 유효좌굴길이계수 K (기둥: 골조형식별, 보/가새: 1.0)
        K = _effective_length_factor(seismic_system, mtype)
        KL_mm = K * length_mm

        dp = _props_for(section_name, mtype)
        A = dp.get("A", 0)

        if A <= 0:
            continue

        # 강도 계산 (단면·길이 기반 — 조합과 무관하므로 지배조합 선정 전에 산정)
        # Column: 양축 + 압축, Beam: 강축 휨 + 전단
        rx = dp.get("rx", 0)
        ry = dp.get("ry", 0)
        Zx = dp.get("Zx", 0)
        Zy = dp.get("Zy", 0)
        Sx = dp.get("Sx", 0)
        Aw = dp.get("Aw", 0)
        Iy = dp.get("Iy", 0) or 0
        Jc = dp.get("J", 0) or 0
        h_mm = dp.get("h", 0) or 0
        tf_mm = dp.get("tf", 0) or 0
        b_mm = dp.get("b", 0) or 0
        tw_mm = dp.get("tw", 0) or 0
        # 플랜지 도심간 거리 ho ≈ h−tf. tf 미상(폴백 단면)이면 ho=h로 과대평가하면
        # rts·Lr 과대→LTB 과소(비보수). 보수적으로 0.95·h 사용.
        ho = (h_mm - tf_mm) if (h_mm > 0 and tf_mm > 0) else (0.95 * h_mm if h_mm > 0 else 0)

        # Tier2-7: 콤팩트 분류 (B4) + 플랜지국부좌굴 휨내력 감소 (F3.2).
        # _classify_compactness/_flb_capacity는 2축대칭 I형강(H/I) 가정 — 채널·T·ㄱ형강
        # 등 비대칭 단면엔 b/2tf·h/tw 룰이 부정확하므로 I/H 단면에만 적용(그 외 compact).
        _sn = (section_name or "").upper()
        is_i_shape = (_sn == "" or _sn.startswith("H-") or _sn.startswith("I-"))
        if is_i_shape:
            compact = _classify_compactness(fy_MPa, E_MPa, b_mm, tf_mm, h_mm, tw_mm)
            phiMnx_flb = _flb_capacity(fy_MPa, E_MPa, Zx, Sx, b_mm, tf_mm, h_mm, tw_mm, compact)
        else:
            compact = {"flange": "n/a", "web": "n/a", "section": "n/a",
                       "flange_lambda": None, "web_lambda": None,
                       "compression_slender": False, "note": "non-I section: B4/F3.2 미적용"}
            phiMnx_flb = _bending_capacity(fy_MPa, Zx)  # FLB 미적용 → Mp
        if compact.get("section") in ("noncompact", "slender"):
            n_noncompact += 1
        # 압축 세장요소(E7)는 '압축재(기둥)'에만 의미. 보(휨지배,축력≈0)의 웹
        # 압축세장은 비대상 — 정상건물 보를 오플래그하지 않도록 기둥만 집계.
        if mtype == "column" and compact.get("compression_slender"):
            n_comp_slender += 1

        ltb_info = None
        if mtype == "column":
            # 기둥: 약축 좌굴 지배. 강축 휨은 층레벨 보가 횡지지 → Mp (LTB 미적용,
            # 가정에 명기). 기둥 압축 세장비는 K(Tier1-1)로 반영.
            r_gov = ry if ry > 0 else rx
            phiPn = _compression_capacity(fy_MPa, E_MPa, A, r_gov, KL_mm)
            phiMnx = _bending_capacity(fy_MPa, Zx)
            phiMny = _bending_capacity(fy_MPa, Zy)
        else:
            # 보: 축력 무시 (횡하중 시 소량 존재하지만 미미)
            phiPn = _compression_capacity(fy_MPa, E_MPa, A, ry if ry > 0 else rx, KL_mm)
            # Tier1-2: 강축 휨에 LTB 반영 (비지지길이 Lb 기준).
            Lb_mm = _resolve_beam_Lb_mm(minfo, length_mm, beam_unbraced)
            if Lb_mm is None:
                phiMnx = _bending_capacity(fy_MPa, Zx)  # 슬래브 연속횡지지 → Mp
                ltb_info = {"zone": "braced(slab)", "governs": False}
            else:
                phiMnx, ltb_info = _ltb_bending_capacity(
                    fy_MPa, E_MPa, Lb_mm, Zx, Sx, ry, Iy, Jc, ho,
                )
            if ltb_info and ltb_info.get("governs"):
                n_ltb_governed += 1
            # 약축 휨내력 phiMny = φ·Fy·Zy (약축은 LTB 미발생 → 소성단면).
            # T4-2(보수적 가정): 보의 약축 모멘트 Muy(3D 해석상 소량 발생)를 H1 상관식
            # (Muy/phiMny 항)에 그대로 포함한다. 보는 통상 강축 전용 설계이고 약축내력
            # phiMny(작은 Zy)는 작으므로, 약축 demand를 더하는 것은 상관비를 키우는
            # **보수측** 처리다(약축 휨을 무시하지 않음). 약축 demand가 0이면 무영향.
            phiMny = _bending_capacity(fy_MPa, Zy)

        # Tier2-7: 강축 휨내력에 플랜지국부좌굴(F3.2) 반영 — LTB와 함께 min (compact면 무변화).
        phiMnx = min(phiMnx, phiMnx_flb)

        phiVn = _shear_capacity(fy_MPa, Aw)
        # B3/Tier2-15: 인장내력 = min(총단면항복 φt·Fy·Ag(D2-a), 순단면파단 φt·Fu·Ae(D2-b)).
        # 연결부 미상이면 Ae/Ag 기본(보수). member_info["net_area_ratio"]로 재정의 가능.
        ae_ag = minfo.get("net_area_ratio", DEFAULT_NET_AREA_RATIO)
        phiPn_yield = PHI * fy_MPa * A / 1000.0  # kN (D2-a)
        phiPn_rupture = _tension_rupture_capacity(fu_MPa, A, ae_ag)  # kN (D2-b)
        phiPn_tens = min(phiPn_yield, phiPn_rupture) if phiPn_rupture > 0 else phiPn_yield
        tension_governs_rupture = phiPn_rupture > 0 and phiPn_rupture < phiPn_yield

        # 실제 H1 상관비로 지배조합 선정 (조합별 H1 + 독립 전단비 집계)
        gov = _governing_member_forces(
            multi_result.member_forces, mid, combo_names,
            phiPn, phiMnx, phiMny, phiVn, phiPn_tens=phiPn_tens,
        )
        envelope = {k: gov[k] for k in ("Pu", "Mux", "Muy", "Vu")}
        governing_combo = gov["governing_combo"]

        axial_ratio = gov["axial"]
        bending_x_ratio = gov["bending_x"]
        bending_y_ratio = gov["bending_y"]
        shear_ratio = gov["shear_ratio"]
        interaction, formula = gov["interaction"], gov["formula"]

        # Tier1-3: 실제 층정보(frame_3d가 member_info["story"]에 기록)를 우선 사용.
        # member_id 기반 추정(_infer_member_story)은 층이 누락(None)일 때만 폴백.
        # 정확한 층 매핑은 P-Delta 부재증폭(아래)·NG 층보고의 정확도와 직결된다.
        story_raw = minfo.get("story")
        story_resolved = isinstance(story_raw, int) and story_raw >= 1
        story = story_raw if story_resolved else _infer_member_story(minfo, stories)

        # C3: P-Delta 2차효과 증폭 (해당 층) — 상관비·전단비 증폭 (KDS §7.2.7).
        # 층이 실제로 확인되면 해당 층 증폭; 추정(불확실)이면 오매핑으로 과소증폭하지
        # 않도록 보수적으로 전 층 최대 증폭을 적용한다(안전측).
        if pdelta_amp_by_story:
            if story_resolved:
                pdelta_amp = pdelta_amp_by_story.get(story, 1.0)
            else:
                pdelta_amp = max([1.0] + list(pdelta_amp_by_story.values()))
        else:
            pdelta_amp = 1.0
        if pdelta_amp and pdelta_amp > 1.0:
            interaction *= pdelta_amp
            shear_ratio *= pdelta_amp

        # Tier2-6: 세장비 한계 (AISC E2 압축 KL/r≤200 / D1 인장 L/r≤300).
        #  - 기둥: 압축재 → 항상 KL/r≤200 (지배조합이 일시 인장이어도 중력 압축을
        #    받으므로 압축한계 적용; 인장한계 300은 더 느슨해 기둥엔 미적용).
        #  - 비기둥(보·가새): 유의 압축(축비≥0.05) 시 KL/r≤200, 순인장 시 L/r≤300.
        #    축력≈0 휨지배 보는 비대상(None) — 정상건물 장경간 보 오플래그 방지.
        r_sl = ry if ry > 0 else rx
        axial_mode = gov.get("axial_mode", "compression")
        slenderness = None
        if r_sl > 0:
            cands = []
            if mtype == "column" or (axial_mode == "compression" and axial_ratio >= 0.05):
                cands.append((KL_mm / r_sl, SLENDER_LIMIT_COMP, "compression"))
            if axial_mode == "tension" and mtype != "column":
                cands.append((length_mm / r_sl, SLENDER_LIMIT_TENS, "tension"))
            if cands:
                sl, sl_lim, sl_mode = max(cands, key=lambda c: c[0] / c[1])
                sl_ok = sl <= sl_lim
                slenderness = {"ratio": round(sl, 1), "limit": sl_lim,
                               "mode": sl_mode, "status": "OK" if sl_ok else "NG"}
                max_slenderness = max(max_slenderness, sl)
                if not sl_ok:
                    n_slenderness_ng += 1

        # 지배 비율 (interaction vs shear 중 큰 것)
        max_ratio = max(interaction, shear_ratio)
        slender_ng = bool(slenderness and slenderness["status"] == "NG")
        status = "OK" if (max_ratio <= 1.0 and not slender_ng) else "NG"

        members.append({
            "member_id": mid,
            "type": mtype,
            "section": section_name,
            "story": story,
            "K": round(K, 2),  # Tier1-1: 적용된 유효좌굴길이계수
            "ltb": ltb_info,   # Tier1-2: 보 LTB 진단 (기둥은 None)
            "slenderness": slenderness,    # Tier2-6: 세장비 (압축/인장, None=비대상)
            "compactness": compact,        # Tier2-7: 콤팩트 분류 (B4)
            "governing_combo": governing_combo,
            "demand": envelope,
            "demand_envelope": gov.get("demand_envelope"),  # A5: 성분별 worst
            "axial_mode": gov.get("axial_mode", "compression"),  # B3: 인장/압축
            "N_signed_kN": gov.get("N_signed", 0.0),
            "capacity": {
                "phiPn_kN": round(phiPn, 1),
                "phiPnt_kN": round(phiPn_tens, 1),  # B3/T2-15: min(항복,파단)
                "phiPnt_yield_kN": round(phiPn_yield, 1),    # D2-a 총단면항복
                "phiPnt_rupture_kN": round(phiPn_rupture, 1),  # D2-b 순단면파단
                "ae_ag_used": round(ae_ag, 3),  # T2-15: 적용된 유효순단면비(기본 0.85 가정)
                "tension_rupture_governs": tension_governs_rupture,
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
                "pdelta_amp": round(pdelta_amp, 3),  # C3: 적용된 2차효과 증폭계수
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

    # Tier1-5: RSA(응답스펙트럼) 조합은 부재력(N/V/M) 미생성 → 부재강도 미검토.
    # combo에는 있으나 member_forces가 비어 '안전'으로 침묵되는 것을 명시(은폐 방지).
    mf_all = multi_result.member_forces or {}

    def _is_rsa_name(c: str) -> bool:
        # None-safe: u는 (c or "")로 가드 — 한글은 upper() 무영향이므로 u로 검사.
        u = (c or "").upper()
        return ("RSA" in u or "SPECTR" in u or "RESPSPEC" in u or "응답스펙트럼" in u)

    rsa_unchecked = sorted({
        c for c in combo_names if _is_rsa_name(c) and not mf_all.get(c)
    })

    # Tier1-1: 적용된 K (모든 기둥 동일 골조형식이므로 대표값 1개).
    K_applied = _effective_length_factor(seismic_system, "column")
    k_sway = K_applied > 1.0

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
            # B3: 순인장 부재 수 (있으면 연결부 순단면파단·블록전단 별도검토 권고)
            "n_tension_members": sum(1 for m in members if m.get("axial_mode") == "tension"),
            # Tier1-2: LTB 지배 보 수 / Tier1-1: 기둥 K / Tier1-5: RSA 미검토 조합
            "n_ltb_governed": n_ltb_governed,
            "column_K": round(K_applied, 2),
            "rsa_unchecked_combos": rsa_unchecked,
            # Tier2-6/7: 세장비 초과 / 비콤팩트 / 압축 세장요소
            "n_slenderness_ng": n_slenderness_ng,
            "max_slenderness": round(max_slenderness, 1),
            "n_noncompact": n_noncompact,
            "n_compression_slender": n_comp_slender,
            "Fu_MPa": round(fu_MPa, 1),
            # T3-9: 기둥 좌굴 비지지길이 = 층고 가정(중간 횡지지 입력 없음).
            "column_unbraced_basis": "story_height",
        },
        "assumptions": [
            (f"유효좌굴길이계수 K: 기둥 K={K_applied:g} "
             + ("(비가새 모멘트골조 sway, AISC 보수 하한; K=1.0은 비보수)"
                if k_sway else "(가새골조·전단벽, 비횡요동)")),
            "기둥 좌굴 비지지길이 = 층고(부재 길이)로 가정 — 중간 횡지지(가새·중간보)는 "
            "입력하지 않음. member_info.unbraced_length_m로 재정의 가능(좌굴/세장비는 이 가정에 민감)",
            "보 강축 휨내력: 슬래브 연속횡지지 시 Mp, 비지지 시 AISC F2 LTB(Lb/Lp/Lr, Cb=1.0). "
            "기둥 강축 휨은 층레벨 횡지지로 Mp 가정",
            "보 약축 휨내력 φMny=φ·Fy·Zy(약축 LTB 없음). 보의 약축 모멘트는 H1 상관식에 "
            "포함하여 검토함 — 보수적 처리(약축 휨을 무시하지 않음, 약축 demand 0이면 무영향)",
            f"세장비 한계: 압축 KL/r≤{SLENDER_LIMIT_COMP}(AISC E2), 인장 L/r≤{SLENDER_LIMIT_TENS}(D1) — "
            "휨지배 보는 비대상",
            "단면 콤팩트 분류(AISC B4): 비콤팩트 플랜지는 국부좌굴(F3.2)로 휨내력 감소. "
            "세장 압축요소(E7) 추가감소는 미반영(R-code 경고) — 롤드 H형강은 통상 비대상",
            f"인장내력: min(총단면항복 φt·Fy·Ag(D2-a), 순단면파단 φt·Fu·Ae(D2-b)). "
            f"Fu={round(fu_MPa, 0):g} MPa, 연결부 미상 시 Ae/Ag={DEFAULT_NET_AREA_RATIO} 보수 가정 "
            "— 블록전단·실연결부 검토는 별도",
            "φ = 0.9 (KDS 41 31 00 표준 강도감소계수, 인장파단 φt=0.75)",
        ],
    }


# ============================================================
# Phase B1: 보 수직처짐 / 사용성 검토 (KDS 41 31 00)
# ============================================================

# KDS 41 31 00 사용성 처짐 한계 (분모)
DEFL_LIMIT_LIVE = 360   # 활하중 처짐 L/360
DEFL_LIMIT_TOTAL = 240  # 전체(고정+활) 처짐 L/240


def _upsample_quadratic(y: list[float], refine: int = 8) -> list[float]:
    """모멘트 다이어그램을 조각별 2차보간으로 세분한다(분포하중 포물선 모멘트에 적합).

    각 구간을 인접 3점 라그랑주 2차식으로 보간 → refine배 세분. 단일 포물선이면
    어느 3점 스텐실로도 동일 포물선이 복원되어 모멘트값 자체는 정확히 재현되므로,
    이후 사다리꼴 이중적분(O(h²))의 거친-메시 과소평가가 크게 줄어든다(refine=8,
    n≥5에서 해석해 대비 <2%). 적분법 자체는 사다리꼴이라 완전 정확은 아니다.
    """
    n = len(y)
    if n < 3 or refine <= 1:
        return list(y)
    m = n - 1
    out: list[float] = []
    for k in range(m):  # 구간 [k, k+1]
        if k == 0:
            i0, i1, i2 = 0, 1, 2
        elif k == m - 1:
            i0, i1, i2 = m - 2, m - 1, m
        else:
            i0, i1, i2 = k - 1, k, k + 1
        for s in range(refine):
            t = k + s / refine  # 전역 인덱스 좌표
            val = (
                y[i0] * ((t - i1) * (t - i2)) / ((i0 - i1) * (i0 - i2))
                + y[i1] * ((t - i0) * (t - i2)) / ((i1 - i0) * (i1 - i2))
                + y[i2] * ((t - i0) * (t - i1)) / ((i2 - i0) * (i2 - i1))
            )
            out.append(val)
    out.append(y[-1])
    return out


def _chord_relative_max_deflection(M_kNm_list: list[float], L_mm: float, EI_Nmm2: float) -> float:
    """모멘트 다이어그램을 이중적분해 '현(chord) 기준' 최대 처짐(mm)을 구한다.

    보 양단을 잇는 직선(현)을 기준으로 한 상대 처짐이므로 지점침하(부동침하)가
    자동 차감된다(KDS 41 31 사용성 — 지점 상대처짐). 강축 모멘트(My_kNm) 스테이션을
    조각별 2차보간으로 세분한 뒤 사다리꼴 이중적분하고, v(0)=v(L)=0(현) 경계를 부여.

    Args:
        M_kNm_list: 강축 휨모멘트 스테이션값 (kN·m), 등간격, n≥3
        L_mm: 부재 길이 (mm)
        EI_Nmm2: 강성 E·I (N·mm²)

    Returns:
        현 기준 최대 처짐 크기 (mm). 데이터 부족/비물리 입력 시 0.0.
    """
    if len(M_kNm_list) < 3 or EI_Nmm2 <= 0 or L_mm <= 0:
        return 0.0
    M = _upsample_quadratic(M_kNm_list, refine=8)  # 거친 메시 정확도 보강
    n = len(M)
    h = L_mm / (n - 1)  # 스테이션 간격 (mm)
    # 곡률 φ = M/EI  (M: kN·m → N·mm 는 ×1e6)
    phi = [m * 1e6 / EI_Nmm2 for m in M]
    # 기울기 적분 (θ0=0 가정)
    theta = [0.0] * n
    for k in range(1, n):
        theta[k] = theta[k - 1] + 0.5 * (phi[k - 1] + phi[k]) * h
    # 처짐 적분 (v0=0)
    v = [0.0] * n
    for k in range(1, n):
        v[k] = v[k - 1] + 0.5 * (theta[k - 1] + theta[k]) * h
    # 현 기준 보정: v_chord(x) = v(x) − (x/L)·v(L)
    vn = v[-1]
    max_def = 0.0
    for k in range(n):
        x = k * h
        vc = v[k] - (x / L_mm) * vn
        if abs(vc) > max_def:
            max_def = abs(vc)
    return max_def


def _beam_strong_axis_Ix(multi_result, section_name: str, mtype: str) -> float:
    """보 강축 단면2차모멘트 Ix (mm⁴) — 단면명 우선, 실패 시 mtype 폴백."""
    if section_name:
        try:
            from core.section_3d import get_section_3d
            return get_section_3d(section_name).Ix
        except Exception:
            pass
    return (getattr(multi_result, f"{mtype}_Ix_mm4", 0)
            or getattr(multi_result, "beam_x_Ix_mm4", 0) or 0)


def check_beam_deflections(multi_result, building_model=None) -> Optional[dict]:
    """KDS 41 31 00 보 수직처짐(사용성) 검토.

    중력 케이스(DL, LL)의 강축 모멘트 다이어그램을 이중적분해 현 기준 최대 처짐을
    구하고, 활하중 처짐 ≤ L/360, 전체(고정+활) 처짐 ≤ L/240 을 검토한다.
    개별 하중케이스(DL/LL) 부재력이 없으면 None(미검토) 반환 → §10에 명시.

    Returns:
        {status, code_ref, beams, critical_beams, summary, assumptions} | None
    """
    E = getattr(multi_result, "E_MPa", 0) or 0
    member_info = getattr(multi_result, "member_info", None) or []
    mf_all = getattr(multi_result, "member_forces", None) or {}
    if E <= 0 or not member_info or not mf_all:
        return None

    # 케이스 키 탐색 (대소문자 무관)
    up = {k.upper(): k for k in mf_all}
    ll_key = up.get("LL") or up.get("LIVE") or up.get("L")
    dl_key = up.get("DL") or up.get("DEAD") or up.get("D")
    if not ll_key and not dl_key:
        return None  # 중력 케이스 부재력 없음 → 미검토

    def _index(case_key):
        return {m["member_id"]: m for m in mf_all.get(case_key, [])} if case_key else {}

    ll_idx = _index(ll_key)
    dl_idx = _index(dl_key)

    beams = []
    for minfo in member_info:
        mtype = minfo.get("type", "")
        if mtype not in ("beam", "beam_x", "beam_y"):
            continue
        mid = minfo["member_id"]
        L_m = minfo.get("length_m", 0) or 0
        if L_m <= 0:
            continue
        L_mm = L_m * 1000.0
        Ix = _beam_strong_axis_Ix(multi_result, minfo.get("section", ""), mtype)
        EI = E * Ix
        if EI <= 0:
            continue

        mf_ll = ll_idx.get(mid)
        mf_dl = dl_idx.get(mid)
        My_ll = mf_ll.get("My_kNm") if mf_ll else None
        My_dl = mf_dl.get("My_kNm") if mf_dl else None
        if not My_ll and not My_dl:
            continue

        # 활하중 처짐 (LL only)
        ratio_live = None
        defl_live = None
        if My_ll:
            defl_live = _chord_relative_max_deflection(My_ll, L_mm, EI)
            ratio_live = defl_live / (L_mm / DEFL_LIMIT_LIVE)
        # 전체 처짐 (DL+LL 비계수 중첩)
        ratio_total = None
        defl_total = None
        if My_dl or My_ll:
            nd = len(My_dl) if My_dl else len(My_ll)
            My_tot = []
            for i in range(nd):
                a = My_dl[i] if (My_dl and i < len(My_dl)) else 0.0
                b = My_ll[i] if (My_ll and i < len(My_ll)) else 0.0
                My_tot.append(a + b)
            defl_total = _chord_relative_max_deflection(My_tot, L_mm, EI)
            ratio_total = defl_total / (L_mm / DEFL_LIMIT_TOTAL)

        ratios = [r for r in (ratio_live, ratio_total) if r is not None]
        if not ratios:
            continue
        max_ratio = max(ratios)
        governing = "live" if (ratio_live is not None and ratio_live >= (ratio_total or 0)) else "total"
        status = "OK" if max_ratio <= 1.0 else "NG"

        beams.append({
            "member_id": mid,
            "type": mtype,
            "section": minfo.get("section", ""),
            "length_m": round(L_m, 3),
            "defl_live_mm": round(defl_live, 3) if defl_live is not None else None,
            "defl_total_mm": round(defl_total, 3) if defl_total is not None else None,
            "ratio_live": round(ratio_live, 4) if ratio_live is not None else None,
            "ratio_total": round(ratio_total, 4) if ratio_total is not None else None,
            "ratio": round(max_ratio, 4),
            "governing": governing,
            "limit_live": f"L/{DEFL_LIMIT_LIVE}",
            "limit_total": f"L/{DEFL_LIMIT_TOTAL}",
            "status": status,
        })

    if not beams:
        return None

    beams.sort(key=lambda b: b["ratio"], reverse=True)
    ng = sum(1 for b in beams if b["status"] == "NG")
    max_ratio = beams[0]["ratio"]
    overall = "OK" if ng == 0 else "NG"

    return {
        "status": overall,
        "code_ref": "KDS 41 31 00 (사용성: 활하중 L/360, 전체 L/240)",
        "beams": beams,
        "critical_beams": beams[:5],
        "summary": {
            "total": len(beams),
            "ok": len(beams) - ng,
            "ng": ng,
            "max_deflection_ratio": round(max_ratio, 4),
            "live_load_case": ll_key,
            "dead_load_case": dl_key,
        },
        "assumptions": [
            "처짐 = 부재 양단 현(chord) 기준 상대처짐 (지점침하 차감)",
            "강축 모멘트 다이어그램 이중적분(사다리꼴) — 메시 스테이션 수에 따른 근사",
            "전체 처짐은 비계수 DL+LL (캠버 미고려, 보수적)",
            f"활하중 L/{DEFL_LIMIT_LIVE}, 전체 L/{DEFL_LIMIT_TOTAL} (KDS 41 31 00 사용성)",
        ],
    }


# ============================================================
# Phase B2: 내진설계범주(SDC) + 횡력저항시스템 적격성/높이제한 (KDS 41 17 00)
# ============================================================

# 내진등급 → 표 5.2-1/5.2-2 등급 키
_IMPORTANCE_GRADE = {"특": "special", "I": "grade_I", "II": "grade_II", "III": "grade_II"}

# KDS 41 17 00 표 6.2-1 횡력저항시스템 높이제한 (db_key 기준).
# 값: None=NL(제한없음), 숫자=높이상한(m), "NP"=해당 SDC 미허용. A/B는 모두 NL.
SEISMIC_HEIGHT_LIMITS: dict[str, dict[str, object]] = {
    "1-a": {"C": None, "D": None},   # RC 특수전단벽
    "1-b": {"C": None, "D": 60},     # RC 보통전단벽
    "2-a": {"C": None, "D": None},   # 철골 특수중심가새골조
    "2-q": {"C": "NP", "D": "NP"},   # 철골 보통중심가새골조(예시)
    "3-a": {"C": None, "D": None},   # 철골 특수모멘트골조
    "3-b": {"C": None, "D": None},   # 철골 중간모멘트골조
    "3-c": {"C": None, "D": None},   # 철골 보통모멘트골조
    "4-a": {"C": None, "D": None},   # RC 특수모멘트골조
    "4-b": {"C": None, "D": None},   # RC 중간모멘트골조
    "4-c": {"C": None, "D": None},   # RC 보통모멘트골조
}


def _sdc_from_value(val: float, grade: str, thresholds: tuple[float, float, float]) -> str:
    """SDS 또는 SD1 + 내진등급 → 내진설계범주 (KDS 41 17 00 표 5.2-1/5.2-2).

    thresholds=(t_hi, t_mid, t_lo): val≥t_hi→D / t_mid≤val<t_hi→(특 D, 그외 C) /
    t_lo≤val<t_mid→(특 C, 그외 B) / val<t_lo→A.
    """
    t_hi, t_mid, t_lo = thresholds
    special = grade == "special"
    if val >= t_hi:
        return "D"
    if val >= t_mid:
        return "D" if special else "C"
    if val >= t_lo:
        return "C" if special else "B"
    return "A"


def _governing_sdc(a: str, b: str) -> str:
    """둘 중 더 엄격한(높은) 내진설계범주."""
    order = {"A": 0, "B": 1, "C": 2, "D": 3}
    return a if order.get(a, 0) >= order.get(b, 0) else b


def seismic_design_category(sds: float, sd1: float, importance: str = "II") -> Optional[str]:
    """SDS·SD1·내진등급 → 내진설계범주(SDC) (KDS 41 17 00 표 5.2-1/5.2-2).

    SDS·SD1 둘 다 유효해야 판정. 하나라도 None/비수치면 None(미상). 외부(load_generator의
    직교지진 100/30 게이트 등)에서도 동일 기준을 쓰도록 공개 헬퍼로 노출한다.
    """
    if not isinstance(sds, (int, float)) or not isinstance(sd1, (int, float)):
        return None
    grade = _IMPORTANCE_GRADE.get(importance or "II", "grade_II")
    sdc_sds = _sdc_from_value(sds, grade, (0.50, 0.33, 0.17))
    sdc_sd1 = _sdc_from_value(sd1, grade, (0.20, 0.14, 0.07))
    return _governing_sdc(sdc_sds, sdc_sd1)


def check_seismic_system(building_model=None, seismic_report: Optional[dict] = None) -> Optional[dict]:
    """내진설계범주(SDC) 판정 + 횡력저항시스템 높이제한/적격성 검토.

    SDS/SD1·내진등급 → SDC (표 5.2-1/5.2-2), 시스템 db_key·높이제한 (표 6.2-1) 조회.
    높이제한 초과 또는 해당 SDC 미허용(NP) 시 NG. seismic_report가 없으면 None(미검토).

    주의: 본 검토는 표 6.2-1 '높이제한'만 평가한다. 시스템별 상세 적용성(KDS 41 31
    부재상세, 추가 SDC 제한)은 별도이며 가정에 명기한다(과대주장 방지).
    """
    if not isinstance(seismic_report, dict) or seismic_report.get("error"):
        return None
    sds = seismic_report.get("SDS")
    sd1 = seismic_report.get("SD1")
    if not isinstance(sds, (int, float)) or not isinstance(sd1, (int, float)):
        return None

    importance = getattr(building_model, "importance", None) or "II"
    grade = _IMPORTANCE_GRADE.get(importance, "grade_II")

    sdc_sds = _sdc_from_value(sds, grade, (0.50, 0.33, 0.17))
    sdc_sd1 = _sdc_from_value(sd1, grade, (0.20, 0.14, 0.07))
    sdc = _governing_sdc(sdc_sds, sdc_sd1)  # == seismic_design_category(sds, sd1, importance)

    system_key = (getattr(building_model, "seismic_system", None)
                  or seismic_report.get("seismic_system") or "ordinary_moment_frame")
    try:
        from core.load_generator import SEISMIC_SYSTEM_MAP
        db_key = SEISMIC_SYSTEM_MAP.get(system_key, "3-c")
    except Exception:
        db_key = "3-c"

    total_height = getattr(building_model, "total_height", None)
    if not isinstance(total_height, (int, float)) or total_height <= 0:
        total_height = seismic_report.get("hn_m") or 0.0

    # 높이제한 조회 (A/B는 NL)
    limits = SEISMIC_HEIGHT_LIMITS.get(db_key, {})
    limit = None if sdc in ("A", "B") else limits.get(sdc)

    issues = []
    if limit == "NP":
        height_ok = False
        system_permitted = False
        limit_desc = "해당 내진설계범주 미허용(NP)"
        issues.append(f"{system_key}({db_key})는 내진설계범주 {sdc}에서 미허용(KDS 41 17 00 표 6.2-1)")
    elif isinstance(limit, (int, float)):
        height_ok = total_height <= limit
        system_permitted = True
        limit_desc = f"{limit:g} m"
        if not height_ok:
            issues.append(
                f"건물 높이 {total_height:.1f} m > 허용 {limit:g} m "
                f"(내진설계범주 {sdc}, {system_key})"
            )
    else:  # None → NL
        height_ok = True
        system_permitted = True
        limit_desc = "제한없음(NL)"

    status = "OK" if (system_permitted and height_ok) else "NG"

    return {
        "status": status,
        "code_ref": "KDS 41 17 00 §5.2(표 5.2-1/5.2-2), §6.1(표 6.2-1)",
        "sdc": sdc,
        "sdc_basis": {"from_SDS": sdc_sds, "from_SD1": sdc_sd1, "governing": sdc},
        "SDS": round(sds, 4), "SD1": round(sd1, 4),
        "importance": importance, "importance_grade": grade,
        "seismic_system": system_key, "db_key": db_key,
        "total_height_m": round(total_height, 2),
        "height_limit": {"sdc": sdc, "limit": limit, "limit_desc": limit_desc},
        "height_ok": height_ok,
        "system_permitted": system_permitted,
        "issues": issues,
        "assumptions": [
            "SDC = max(SDS기준, SD1기준) (KDS 41 17 00 표 5.2-1/5.2-2)",
            "높이제한은 표 6.2-1만 평가 — 시스템별 상세 적용성(KDS 41 31 부재상세·추가제한)은 미검토",
        ],
    }


# ============================================================
# Phase C1: 전역 안정 — 전도(overturning)·활동(sliding)
# ============================================================

OVERTURNING_FS_MIN = 1.5       # 전도 안전율
SLIDING_FS_MIN = 1.5           # 활동 안전율
SLIDING_FRICTION_DEFAULT = 0.4  # 기초-지반 마찰계수 (간이 가정)


def check_global_stability(building_model=None, seismic_report: Optional[dict] = None) -> Optional[dict]:
    """전역 전도/활동 안정 검토 (등가정적 지진력 기준).

    전도모멘트 M_ot = Σ(Fx_i·h_i), 저항모멘트 M_r = 0.9·W·(평면폭/2).
    전도 안전율 = M_r/M_ot ≥ 1.5 (방향별, 좁은 평면폭이 지배).
    활동은 기초-지반 영역이라 간이지표(μ·0.9W/V)로 '참고'만 제공한다.
    seismic_report(story_forces, W, V)가 없으면 None(미검토).
    """
    if not isinstance(seismic_report, dict) or seismic_report.get("error"):
        return None
    sf = seismic_report.get("story_forces")
    W = seismic_report.get("W_kN")
    V = seismic_report.get("V_kN")
    if not sf or not isinstance(W, (int, float)) or W <= 0:
        return None

    # 전도모멘트 (밑면 기준)
    M_ot = sum((s.get("Fx_kN", 0) or 0) * (s.get("height_m", 0) or 0) for s in sf)

    # 데이터 정합성: 밑면전단 V>0 인데 story_forces의 Fx가 비어 M_ot≈0 이면
    # (잘린/손상된 seismic_report) 전도를 '안전'으로 오판하지 않도록 미검토 처리.
    if M_ot <= 1e-9 and isinstance(V, (int, float)) and V > 1e-6:
        return None

    Bx = getattr(building_model, "total_width_x", 0) or 0
    By = getattr(building_model, "total_width_y", 0) or 0

    overturning = {}
    over_status = "OK"
    gov_fs = None
    for dir_, B in (("X", Bx), ("Y", By)):
        if B <= 0:
            continue
        Mr = 0.9 * W * (B / 2.0)
        fs = Mr / M_ot if M_ot > 1e-9 else None  # None = 횡력 없음(무한 안전)
        st = "OK" if (fs is None or fs >= OVERTURNING_FS_MIN) else "NG"
        if st == "NG":
            over_status = "NG"
        overturning[dir_] = {
            "width_m": round(B, 2), "M_resist_kNm": round(Mr, 1),
            "M_overturn_kNm": round(M_ot, 1),
            "FS": round(fs, 2) if fs is not None else None, "status": st,
        }
        if fs is not None and (gov_fs is None or fs < gov_fs):
            gov_fs = fs

    # 활동 (참고: 기초도메인 — 고정단 상부구조 모델로는 지표만)
    mu = SLIDING_FRICTION_DEFAULT
    N = 0.9 * W
    slide_fs = (mu * N) / V if isinstance(V, (int, float)) and V > 0 else None
    sliding = {
        "mu": mu, "N_kN": round(N, 1),
        "V_kN": round(V, 2) if isinstance(V, (int, float)) else None,
        "FS": round(slide_fs, 2) if slide_fs is not None else None,
        "status": ("OK" if (slide_fs is None or slide_fs >= SLIDING_FS_MIN) else "check"),
        "note": "기초-지반 마찰 간이 가정(μ=0.4, 0.9D) — 지반/근입/수동저항 별도검토 필요",
    }

    return {
        "status": over_status,   # overall은 전도만 게이트 (활동은 참고)
        "code_ref": "전역안정: 전도 M_r/M_ot ≥ 1.5 (0.9D 저항)",
        "overturning": overturning,
        "governing_FS": round(gov_fs, 2) if gov_fs is not None else None,
        "sliding": sliding,
        "required_FS": OVERTURNING_FS_MIN,
        "assumptions": [
            "전도저항 = 0.9·W·(평면폭/2) (균등질량·강체 가정)",
            "전도모멘트 = Σ(Fx·h) 등가정적 지진력 기준",
            "활동(sliding)은 기초-지반 영역 — 본 검토는 참고지표만(overall 판정 제외)",
        ],
    }


# ============================================================
# Phase C3: P-Delta 안정계수 θ (KDS 41 17 00 §7.2.7)
# ============================================================

def check_pdelta_stability(seismic_report: Optional[dict] = None,
                           drift_check: Optional[dict] = None) -> Optional[dict]:
    """P-Delta 안정계수 θ 검토 (KDS 41 17 00).

    θ = (Px·Δ)/(Vx·hsx·Cd) — Δ=비탄성 층간변위, hsx 소거 후 θ=Px·δratio_inel/(Vx·Cd).
      Px: 해당 층 이상 누적 중력하중, Vx: 해당 층 이상 누적 지진전단.
    θ≤0.1 무시 / 0.1<θ≤θmax 증폭 1/(1−θ) / θ>θmax=min(0.5/(β·Cd),0.25) 불안정(NG).
    (β=1.0 보수가정.) seismic_report/drift_check 없으면 None(미검토).
    """
    if not isinstance(seismic_report, dict) or seismic_report.get("error"):
        return None
    sf = seismic_report.get("story_forces")
    Cd = seismic_report.get("Cd")
    if not sf or not isinstance(Cd, (int, float)) or Cd <= 0:
        return None
    if not drift_check or not drift_check.get("checks"):
        return None

    # 층별 비탄성 층간변위비 (방향/조합 중 최대)
    drift_by_story: dict[int, float] = {}
    for c in drift_check["checks"]:
        st = c["story"]
        r = c.get("inelastic_drift", 0) or 0
        if st not in drift_by_story or r > drift_by_story[st]:
            drift_by_story[st] = r

    w_by_story = {s["story"]: (s.get("weight_kN", 0) or 0) for s in sf}
    fx_by_story = {s["story"]: (s.get("Fx_kN", 0) or 0) for s in sf}
    stories_sorted = sorted(w_by_story)

    theta_limit = min(0.5 / Cd, 0.25)  # β=1.0 보수
    checks = []
    max_theta = 0.0
    status = "OK"
    needs_amp = False
    for st in stories_sorted:
        Px = sum(w_by_story[s] for s in stories_sorted if s >= st)
        Vx = sum(fx_by_story[s] for s in stories_sorted if s >= st)
        drift = drift_by_story.get(st, 0.0)
        if Vx <= 0:
            continue
        theta = Px * drift / (Vx * Cd)
        if theta > theta_limit:
            st_status = "NG"
            status = "NG"
            amp = None
        elif theta > 0.1:
            st_status = "amplify"
            needs_amp = True
            amp = round(1.0 / (1.0 - theta), 3)
        else:
            st_status = "OK"
            amp = 1.0
        checks.append({
            "story": st, "theta": round(theta, 4),
            "Px_kN": round(Px, 1), "Vx_kN": round(Vx, 1),
            "inelastic_drift": round(drift, 6),
            "amplification": amp, "status": st_status,
        })
        max_theta = max(max_theta, theta)

    if not checks:
        return None

    # T3-5: 1/(1−θ)는 '층(전역) 단위' 2차효과 증폭(AISC B2형)으로, 부재 상관비·층간변위에
    # 동일 적용한다. 매우 세장한 개별 기둥(큰 Pr/Pe1·부재 곡률)은 부재 단위 B1=Cm/(1−Pr/Pe1)이
    # 별도 지배할 수 있으나 본 검토는 1/(1−θ) 선형근사만 적용 — 과대주장 방지용 명시.
    member_level_note = (
        bool(needs_amp) or max_theta > 0.1
    )
    return {
        "status": status,
        "code_ref": "KDS 41 17 00 §7.2.7 (P-Delta 안정계수 θ)",
        "Cd": Cd,
        "theta_limit": round(theta_limit, 4),
        "max_theta": round(max_theta, 4),
        "needs_amplification": needs_amp,
        "amplification_method": "story_level_B2_1/(1-theta)",  # T3-5
        "member_level_b1_note": member_level_note,             # T3-5
        "checks": checks,
        "assumptions": [
            "θ = Px·Δ/(Vx·hsx·Cd), Δ=비탄성 층간변위 (hsx 소거형)",
            "θmax = min(0.5/(β·Cd), 0.25), β=1.0 (보수)",
            "0.1<θ≤θmax: 2차효과 증폭 1/(1−θ), θ>θmax: 불안정(NG)",
            "증폭 1/(1−θ)는 층(전역) 단위 B2형 근사 — 매우 세장한 개별 기둥은 부재 B1"
            "(Cm/(1−Pr/Pe1))이 별도 지배 가능(본 검토 미반영, 상세설계 확인 권장)",
        ],
    }


# ============================================================
# Tier2-10: 풍하중 사용성 층간변위 (H/400, 탄성)
# ============================================================

WIND_DRIFT_LIMIT_INV = 400   # 풍하중 사용성 층간변위 한계 1/400 (KDS 41 12 사용성 통상)


def check_wind_drifts(combo_results: dict, stories: list[float]) -> Optional[dict]:
    """풍하중 사용성 층간변위 검토 (탄성 δe/h ≤ 1/400).

    풍하중 조합(WX/WY)의 **탄성** 층간변위비를 1/400과 비교한다. 지진(KDS 41 17 §8.2.3)은
    비탄성변위(Cd·δe/Ie)를 쓰지만 풍은 사용성(탄성)이므로 증폭하지 않는다.
    풍하중 조합이 없으면 None(미검토) → §10에 명시. 처짐검토와 동일하게 overall을 게이트.

    Returns:
        {status, code_ref, limit_inv, allowable, checks, critical, max_ratio, ng} | None
    """
    # 풍하중 케이스 토큰만 매칭: '...1.0WX...' 형태(부호/계수 뒤 WX/WY, 뒤에 영문 없음).
    # 단순 substring("WX" in name)은 외부/사용자정의 조합('NEWX_Load' 등)을 오탐할 수 있어
    # H/400 게이트가 비풍하중 조합에 잘못 적용될 위험 → 토큰 경계 정규식으로 한정.
    _wind_re = re.compile(r"[\d.]W[XY](?![A-Za-z])")
    wind_combos = [name for name in combo_results if _wind_re.search(name or "")]
    if not wind_combos:
        return None
    allowable = 1.0 / WIND_DRIFT_LIMIT_INV
    checks = []
    max_ratio = 0.0
    critical = None
    any_drift = False
    for combo_name in wind_combos:
        cr = combo_results[combo_name]
        for sd in getattr(cr, "story_drifts", []) or []:
            story = sd.get("story")
            height_m = sd.get("height_m")
            for direction, key in (("X", "drift_x"), ("Y", "drift_y")):
                de = sd.get(key)
                if de is None:
                    continue
                any_drift = True
                de = abs(de)
                ratio = de / allowable if allowable > 0 else 0.0
                status = "OK" if ratio <= 1.0 else "NG"
                drift_inv = f"1/{int(1 / de)}" if de > 1e-9 else "0"
                checks.append({
                    "story": story, "height_m": height_m, "direction": direction,
                    "combo": combo_name, "elastic_drift": round(de, 6),
                    "allowable": round(allowable, 6), "ratio": round(ratio, 4),
                    "status": status, "drift_inv": drift_inv,
                })
                if ratio > max_ratio:
                    max_ratio = ratio
                    critical = {"story": story, "direction": direction,
                                "combo": combo_name, "ratio": round(ratio, 4),
                                "elastic_drift": round(de, 6), "drift_inv": drift_inv}
    if not any_drift or not checks:
        return None
    checks.sort(key=lambda c: c["ratio"], reverse=True)
    ng = sum(1 for c in checks if c["status"] == "NG")
    overall = "OK" if max_ratio <= 1.0 else "NG"
    return {
        "status": overall,
        "code_ref": f"풍하중 사용성 층간변위 δe/h ≤ 1/{WIND_DRIFT_LIMIT_INV} (탄성)",
        "limit_inv": WIND_DRIFT_LIMIT_INV,
        "allowable": round(allowable, 6),
        "checks": checks,
        "critical": critical,
        "max_ratio": round(max_ratio, 4),
        "ng": ng,
        "assumptions": [
            f"풍하중 사용성 한계 δe/h ≤ 1/{WIND_DRIFT_LIMIT_INV} (탄성변위 — Cd·Ie 미적용)",
            "풍하중 강도조합(1.0W)의 탄성 층간변위 기준 — 사용성 재현주기 풍속 미환산(보수)",
        ],
    }


# ============================================================
# Tier2-12: 수직 비정형 자동탐지 (KDS 41 17 00 표 5.3-2)
# ============================================================

VERT_STIFFNESS_SOFT = 0.70      # 연층(1a): K_i < 0.7·K_상층
VERT_STIFFNESS_EXTREME = 0.60   # 극단 연층(1b): K_i < 0.6·K_상층
VERT_STIFFNESS_AVG3 = 0.80      # 연층(1a): K_i < 0.8·상위3층 평균
VERT_STIFFNESS_AVG3_EXT = 0.70  # 극단 연층(1b): K_i < 0.7·상위3층 평균
VERT_MASS_RATIO = 1.50          # 중량 비정형(2): m_i > 1.5·인접층
VERT_GEOM_RATIO = 1.30          # 기하 비정형(3): 평면치수_i > 1.3·인접층


def check_vertical_irregularity(building_model=None, seismic_report: Optional[dict] = None,
                                drift_check: Optional[dict] = None) -> Optional[dict]:
    """수직 비정형 자동탐지 (KDS 41 17 00 표 5.3-2).

    - 강성(연층) 1a/1b: 층강성 K_i = V_i/Δ_i (탄성 층전단/층간변위). K_i<0.7·K_상층 또는
      <0.8·상위3층평균 → 연층(1a). <0.6 또는 <0.7·평균 → 극단연층(1b).
    - 중량(질량) 2: 층중량 m_i > 1.5·인접층(최상층=지붕은 가벼움이 정상이라 비교 제외).
    - 기하 3: 층 평면치수(바닥면적 대용) > 1.3·인접층.
    비틀림 비정형과 동일하게 **분류·경고만** 수행하고 overall 판정은 게이트하지 않는다
    (효과는 동적해석·내진상세 요구). seismic_report(story_forces)가 없으면 None(미검토).

    Returns:
        {status, code_ref, irregular, extreme, types, findings, n_stories, assumptions} | None
    """
    if not isinstance(seismic_report, dict) or seismic_report.get("error"):
        return None
    sf = seismic_report.get("story_forces")
    if not sf:
        return None

    w_by_story = {s["story"]: (s.get("weight_kN", 0) or 0) for s in sf}
    fx_by_story = {s["story"]: (s.get("Fx_kN", 0) or 0) for s in sf}
    stories_sorted = sorted(w_by_story)
    if len(stories_sorted) < 2:
        return None  # 단층은 수직 비정형 비대상

    # 층 높이 (drift_check checks에서 우선, 없으면 building_model)
    height_by_story: dict[int, float] = {}
    if drift_check and drift_check.get("checks"):
        for c in drift_check["checks"]:
            h = c.get("height_m")
            if isinstance(h, (int, float)) and h > 0:
                height_by_story.setdefault(c["story"], h)
    if not height_by_story and building_model is not None:
        for i, st in enumerate(getattr(building_model, "stories", []) or [], start=1):
            h = getattr(st, "height", None)
            if isinstance(h, (int, float)) and h > 0:
                height_by_story[i] = h

    # 층별 탄성 층간변위비 (방향 최대) — 강성 산정용
    drift_ratio_by_story: dict[int, float] = {}
    if drift_check and drift_check.get("checks"):
        for c in drift_check["checks"]:
            r = c.get("elastic_drift", 0) or 0
            st = c["story"]
            if st not in drift_ratio_by_story or r > drift_ratio_by_story[st]:
                drift_ratio_by_story[st] = r

    # 층강성 K_i = V_i / Δ_i,  V_i=Σ(Fx, s≥i),  Δ_i=ratio_i·h_i (탄성 층간변위)
    stiffness: dict[int, float] = {}
    for st in stories_sorted:
        ratio = drift_ratio_by_story.get(st, 0.0)
        h = height_by_story.get(st, 0.0)
        Vi = sum(fx_by_story[s] for s in stories_sorted if s >= st)
        delta = ratio * h
        if Vi > 0 and delta > 1e-9:
            stiffness[st] = Vi / delta

    findings: list[dict] = []
    types: set = set()
    extreme = False

    # ── 1. 강성(연층) 비정형 ──
    for idx, st in enumerate(stories_sorted):
        Ki = stiffness.get(st)
        if Ki is None:
            continue
        above = [s for s in stories_sorted if s > st]
        if not above:
            continue
        K_above = stiffness.get(above[0])
        above3 = [stiffness[s] for s in above[:3] if s in stiffness]
        avg3 = sum(above3) / len(above3) if above3 else None
        is_soft = is_ext = False
        if K_above and K_above > 0:
            if Ki < VERT_STIFFNESS_EXTREME * K_above:
                is_soft = is_ext = True
            elif Ki < VERT_STIFFNESS_SOFT * K_above:
                is_soft = True
        if avg3 and avg3 > 0:
            if Ki < VERT_STIFFNESS_AVG3_EXT * avg3:
                is_soft = is_ext = True
            elif Ki < VERT_STIFFNESS_AVG3 * avg3:
                is_soft = True
        if is_soft:
            types.add("stiffness_soft_story")
            extreme = extreme or is_ext
            findings.append({
                "type": "stiffness_soft_story",
                "kds_type": "1b(극단연층)" if is_ext else "1a(연층)",
                "story": st,
                "K_ratio_above": round(Ki / K_above, 3) if K_above else None,
                "K_ratio_avg3": round(Ki / avg3, 3) if avg3 else None,
            })

    # ── 2. 중량(질량) 비정형 ── (최상층=지붕은 '가벼움 정상'이라 heavy 비교 제외)
    roof = stories_sorted[-1]
    for st in stories_sorted:
        if st == roof:
            continue
        mi = w_by_story.get(st, 0)
        # 인접층 중량 (지붕은 '가벼움 정상'이므로 비교 대상에서 제외)
        adj = [w_by_story[s] for s in (st - 1, st + 1)
               if s in w_by_story and s != roof]
        adj_min = min(adj) if adj else None
        if adj_min and adj_min > 0 and mi > VERT_MASS_RATIO * adj_min:
            types.add("mass")
            findings.append({
                "type": "mass", "kds_type": "2(중량)", "story": st,
                "mass_ratio": round(mi / adj_min, 3),
            })

    # ── 3. 기하(평면치수) 비정형 ── (바닥면적비를 평면치수 대용)
    if building_model is not None and hasattr(building_model, "floor_area_at_story"):
        area_by_story: dict[int, float] = {}
        for st in stories_sorted:
            try:
                a = building_model.floor_area_at_story(st)
            except Exception:
                a = 0.0
            if isinstance(a, (int, float)) and a > 0:
                area_by_story[st] = a
        for st in stories_sorted:
            ai = area_by_story.get(st)
            if ai is None:
                continue
            adj = [area_by_story[s] for s in (st - 1, st + 1) if s in area_by_story]
            adj_min = min(adj) if adj else None
            # 평면치수 ∝ √면적 → 치수비 1.3 ↔ 면적비 1.69
            if adj_min and adj_min > 0 and ai > (VERT_GEOM_RATIO ** 2) * adj_min:
                types.add("geometric")
                findings.append({
                    "type": "geometric", "kds_type": "3(기하)", "story": st,
                    "dim_ratio": round((ai / adj_min) ** 0.5, 3),
                })

    irregular = bool(types)
    findings.sort(key=lambda f: f.get("story", 0))
    return {
        "status": "OK",  # 분류 자체는 판정 아님 (효과는 동적해석·내진상세 요구)
        "code_ref": "KDS 41 17 00 표 5.3-2 (수직 비정형)",
        "irregular": irregular,
        "extreme": extreme,
        "types": sorted(types),
        "findings": findings,
        "n_stories": len(stories_sorted),
        "assumptions": [
            "강성(연층) K_i=V_i/Δ_i (탄성 층전단/층간변위) — 0.7·상층 또는 0.8·상위3층평균 미만",
            "중량 비정형 m_i>1.5·인접층(지붕 제외), 기하 비정형 평면치수>1.3·인접층(√면적 대용)",
            "분류·경고만 — 동적해석/내진상세 요구는 책임기술자 판단",
        ],
    }


# ============================================================
# Phase 3: 통합 설계 검토
# ============================================================

def _is_geometric_nonlinear(multi_result) -> bool:
    """해석이 기하비선형(P-Delta/Corotational)이면 2차효과가 이미 반영됨."""
    gn = str(getattr(multi_result, "geometric_nonlinearity", "") or "").lower()
    am = getattr(multi_result, "analysis_metadata", {}) or {}
    transf = str(am.get("actual_transf", "") or am.get("requested_mode", "") or "").lower()
    blob = gn + " " + transf
    return any(k in blob for k in ("pdelta", "corotational", "nonlinear"))


def _apply_pdelta_amp_to_drift(drift_check: dict, amp_by_story: dict) -> None:
    """C3: P-Delta 2차효과 증폭 1/(1−θ)을 층간변위에 적용(제자리 갱신, KDS §7.2.7)."""
    allowable = drift_check.get("allowable", 0.02)
    max_ratio = 0.0
    critical = None
    for chk in drift_check["checks"]:
        amp = amp_by_story.get(chk["story"], 1.0)
        if amp and amp > 1.0:
            chk["inelastic_drift"] = round(chk["inelastic_drift"] * amp, 6)
            chk["pdelta_amp"] = round(amp, 3)
            chk["ratio"] = round(chk["inelastic_drift"] / allowable, 4) if allowable > 0 else 0.0
            chk["status"] = "OK" if chk["ratio"] <= 1.0 else "NG"
            chk["drift_inv"] = (f"1/{int(1 / chk['inelastic_drift'])}"
                                if chk["inelastic_drift"] > 1e-8 else "0")
        if chk["ratio"] > max_ratio:
            max_ratio = chk["ratio"]
            critical = {
                "story": chk["story"], "direction": chk["direction"],
                "combo": chk["combo"], "ratio": chk["ratio"],
                "inelastic_drift": chk["inelastic_drift"],
            }
    drift_check["max_ratio"] = round(max_ratio, 4)
    drift_check["critical"] = critical
    drift_check["status"] = "OK" if max_ratio <= 1.0 else "NG"


# ============================================================
# Tier1-4: 솔버 실패 게이트 — 0(영) 결과를 '안전'으로 보고하지 않음
# ============================================================

def _analysis_failed(multi_result) -> bool:
    """해석(정적 솔버)이 비수렴/특이행렬로 실패했는지.

    V2(analyze_from_model)는 솔버 실패 시 빈 케이스결과로 진행하므로
    analysis_metadata에 solver_failed 플래그를 싣는다. 그 플래그를 확인한다.
    """
    am = getattr(multi_result, "analysis_metadata", None) or {}
    if am.get("solver_failed") or am.get("nonfinite_response"):
        return True
    # 솔버 메타가 ok!=0을 직접 들고 있는 경우(드묾)도 실패로 본다.
    ok = am.get("ok")
    if isinstance(ok, int) and ok != 0:
        return True
    # Tier2-14(안전): 솔버 ok=0이어도 결과(변위/층간변위)가 비유한이면 신뢰불가.
    # frame_3d가 케이스/조합 결과에 nonfinite_response 플래그를 단다.
    for coll in ("case_results", "combo_results"):
        for r in (getattr(multi_result, coll, None) or {}).values():
            if getattr(r, "nonfinite_response", False):
                return True
    return False


def _analysis_error_result(multi_result) -> dict:
    """해석 실패 시 '안전' 오판을 막는 게이트 결과(overall NG + analysis_error).

    모든 하위검토는 not_checked, overall_status는 NG로 둔다(0 변위·0 부재력을
    근거로 OK를 내지 않는다). 반환 dict는 JSON-safe(문자열 키, 유한값).
    """
    am = getattr(multi_result, "analysis_metadata", None) or {}
    failed_cases = am.get("failed_cases") or []
    detail = (f" (케이스: {', '.join(map(str, failed_cases))})" if failed_cases else "")
    issue = {
        "type": "analysis_solver_failed",
        "description": (
            "구조해석 솔버가 비수렴/특이행렬/비유한(NaN·Inf) 응답으로 실패했습니다" + detail
            + " — 변위·부재력이 0 또는 비유한으로 산출되어 설계검토 결과를 신뢰할 수 없습니다. "
            "모델(지점·강막·좌표병합)을 점검 후 재해석하십시오."
        ),
        "code_ref": "해석 무결성 게이트",
        "combo": "전체",
    }
    return {
        "overall_status": "NG",
        "analysis_error": {
            "solver_failed": True,
            "failed_cases": list(failed_cases),
            "message": issue["description"],
        },
        "drift_check": None,
        "member_check": None,
        "deflection_check": None,
        "system_check": None,
        "stability_check": None,
        "pdelta_check": None,
        "torsion_check": None,
        "wind_check": None,
        "vertical_check": None,
        "accidental_torsion": None,
        "critical_issues": [issue],
        "summary": {
            "analysis_error": True,
            "max_drift_ratio": None,
            "max_interaction_ratio": None,
            "drift_governing_combo": None,
            "member_governing_combo": None,
            "max_deflection_ratio": None,
            "deflection_status": "not_checked",
            "sdc": None,
            "seismic_system_status": "not_checked",
            "overturning_FS": None,
            "stability_status": "not_checked",
            "max_theta": None,
            "pdelta_status": "not_checked",
            "torsion_ratio": None,
            "torsion_classification": "not_checked",
            "max_wind_drift_ratio": None,
            "wind_drift_status": "not_checked",
            "vertical_irregularity": None,
            "vertical_irregular": False,
            "vertical_irregularity_status": "not_checked",
            "accidental_torsion_applied": None,
            "not_checked_items": ["전체(해석 실패)"],
            "seismic_input_error": False,
            "n_slenderness_ng": 0,
            "max_slenderness": 0,
            "n_noncompact": 0,
            "n_compression_slender": 0,
            "ng_stories": 0,
            "ng_members": 0,
            "ng_beams_deflection": 0,
        },
    }


def run_design_check(
    multi_result,
    building_model=None,
    seismic_report: Optional[dict] = None,
    beam_unbraced: bool = False,
) -> dict:
    """전체 설계 검토 실행 (층간변위 + 부재강도).

    Args:
        multi_result: Frame3DMultiCaseResult
        building_model: BuildingModel (importance, seismic_system 등)
        seismic_report: load_result["reports"]["seismic"] (Cd, IE 포함)
        beam_unbraced: True면 보를 비지지(Lb=부재길이)로 보아 LTB 보수검토 (Tier1-2).

    Returns:
        {overall_status, drift_check, member_check, critical_issues, summary}
    """
    # ── Tier1-4: 해석 실패 게이트 — 0 결과를 '안전'으로 보고하지 않음 ──
    if _analysis_failed(multi_result):
        return _sanitize_nonfinite(_analysis_error_result(multi_result))

    fy = multi_result.fy_MPa
    E = multi_result.E_MPa
    stories = multi_result.stories
    seismic_system = getattr(building_model, "seismic_system", None)

    # ── 0. 비틀림 비정형 + Ax (C2) — 층간변위 증폭에 사용하므로 먼저 산정 ──
    torsion_check = None
    if multi_result.combo_results:
        torsion_check = check_torsional_irregularity(multi_result.combo_results)

    # #13: 손상된 seismic_report({error:...})는 '안전'으로 새지 않도록 게이트한다.
    # (지진 의도됐으나 입력 실패 → drift를 빈 OK로 흘리지 않고 not_checked로 둠.)
    seismic_input_error = bool(isinstance(seismic_report, dict) and seismic_report.get("error"))
    seismic_ok = isinstance(seismic_report, dict) and not seismic_report.get("error")

    # ── 1. 층간변위 검토 (C2 Ax 증폭 반영) ──
    drift_check = None
    if seismic_ok and multi_result.combo_results:
        Cd = seismic_report.get("Cd", 3.0)
        IE = seismic_report.get("IE", 1.0)
        importance = "II"
        if building_model:
            importance = getattr(building_model, "importance", "II")

        ax_map = torsion_check["ax_map"] if torsion_check else None
        drift_check = check_story_drifts(
            multi_result.combo_results, stories, Cd, IE, importance, ax_map=ax_map,
        )

    # ── 1b. P-Delta 안정계수 θ (C3) — 부재/변위 2차효과 증폭에 쓰므로 먼저 산정 ──
    pdelta_check = check_pdelta_stability(seismic_report, drift_check)
    pdelta_amp_by_story: dict = {}
    # 선형해석일 때만 증폭 적용 (기하비선형은 2차효과 이미 반영 → 이중계산 방지).
    if (pdelta_check and pdelta_check.get("needs_amplification")
            and not _is_geometric_nonlinear(multi_result)):
        for c in pdelta_check["checks"]:
            if c.get("status") == "amplify" and c.get("amplification"):
                pdelta_amp_by_story[c["story"]] = c["amplification"]
    # 층간변위에 2차효과 증폭 적용 (KDS 41 17 00 §7.2.7)
    if pdelta_amp_by_story and drift_check:
        _apply_pdelta_amp_to_drift(drift_check, pdelta_amp_by_story)

    # ── 2. 부재 강도 검토 (P-Delta 2차효과 증폭 반영) ──
    member_check = None
    if multi_result.member_info and (multi_result.member_forces or multi_result.combo_results):
        member_check = check_member_strengths(
            multi_result, fy, E, pdelta_amp_by_story=pdelta_amp_by_story or None,
            seismic_system=seismic_system, beam_unbraced=beam_unbraced,
        )

    # ── 2b. 보 수직처짐(사용성) 검토 (B1) ──
    deflection_check = check_beam_deflections(multi_result, building_model)

    # ── 2c. 내진설계범주(SDC) + 시스템 적격성/높이제한 (B2) ──
    system_check = check_seismic_system(building_model, seismic_report)

    # ── 2d. 전역안정: 전도/활동 (C1) ──
    stability_check = check_global_stability(building_model, seismic_report)

    # ── 2e. 풍하중 사용성 층간변위 (H/400, #10) ──
    wind_check = None
    if multi_result.combo_results:
        wind_check = check_wind_drifts(multi_result.combo_results, stories)

    # ── 2f. 수직 비정형 자동탐지 (#12, 분류·경고만 — overall 미게이트) ──
    vertical_check = check_vertical_irregularity(building_model, seismic_report, drift_check)

    # ── 2g. 우발 비틀림 (#8) — 등가정적 모델은 우발편심 미적용 → 산정값 명시 ──
    # 손상된 지진입력(error)이면 우발 비틀림도 무의미 → 다른 지진검토와 동일하게 게이트.
    accidental_torsion = None
    if isinstance(seismic_report, dict) and not seismic_report.get("error"):
        at = seismic_report.get("accidental_torsion")
        if isinstance(at, dict) and at.get("story_moments"):
            accidental_torsion = at

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
            if m["status"] != "NG":
                continue
            interaction_r = m["ratios"]["interaction"]
            shear_r = m["ratios"]["shear"]
            # Tier2-6: 세장비로 NG면 전용 이슈로 보고(상관비<1인데 '상관비>1.0' 오문구 방지).
            sl = m.get("slenderness")
            if sl and sl.get("status") == "NG":
                critical_issues.append({
                    "type": "slenderness_exceeded",
                    "description": (
                        f"부재 {m['member_id']} ({m['type']}, {m['section']}) "
                        f"세장비 {sl['ratio']} > {sl['limit']} "
                        f"({'압축 KL/r' if sl['mode'] == 'compression' else '인장 L/r'}, AISC E2/D1)"
                    ),
                    "code_ref": "AISC 360 E2(압축)/D1(인장)",
                    "combo": m["governing_combo"],
                })
            # 강도(상관/전단) 초과 — 세장비만 NG인 부재는 여기 미해당(중복·오문구 방지).
            if interaction_r > 1.0 or shear_r > 1.0:
                if shear_r >= interaction_r:
                    desc = (
                        f"부재 {m['member_id']} ({m['type']}, {m['section']}) "
                        f"전단비 {shear_r:.3f} > 1.0 (Vu/φVn, AISC G2)"
                    )
                else:
                    ltb = m.get("ltb") or {}
                    ltb_tag = (f", LTB 지배 Lb={ltb.get('Lb_m')}m"
                               if ltb.get("governs") else "")
                    desc = (
                        f"부재 {m['member_id']} ({m['type']}, {m['section']}) "
                        f"상관비 {interaction_r:.3f} > 1.0 "
                        f"({m['ratios']['formula']}{ltb_tag})"
                    )
                critical_issues.append({
                    "type": "member_overstressed",
                    "description": desc,
                    "code_ref": "KDS 41 31 00 / AISC 360 H1",
                    "combo": m["governing_combo"],
                })

    # Tier1-5: RSA(응답스펙트럼) 조합 부재력 미생성 → 부재검토 미수행 명시 (은폐 방지).
    if member_check and member_check["summary"].get("rsa_unchecked_combos"):
        rsa_combos = member_check["summary"]["rsa_unchecked_combos"]
        critical_issues.append({
            "type": "rsa_member_check_not_performed",
            "description": (
                f"RSA(응답스펙트럼) 조합 {len(rsa_combos)}개의 부재강도 미검토 "
                f"({', '.join(rsa_combos[:3])}) — RSA는 변위만 산출, 부재력(N/V/M) 미생성. "
                f"등가정적(ELF) 조합 기준 부재검토만 수행됨"
            ),
            "code_ref": "RSA 부재검토 한계",
            "combo": ", ".join(rsa_combos[:3]),
        })

    if deflection_check and deflection_check["status"] == "NG":
        for b in deflection_check["beams"]:
            if b["status"] == "NG":
                lim = b["limit_live"] if b["governing"] == "live" else b["limit_total"]
                kind = "활하중" if b["governing"] == "live" else "전체"
                critical_issues.append({
                    "type": "deflection_exceeded",
                    "description": (
                        f"보 {b['member_id']} ({b['type']}, {b['section']}) "
                        f"{kind} 처짐비 {b['ratio']:.3f} > 1.0 (한계 {lim})"
                    ),
                    "code_ref": "KDS 41 31 00 사용성",
                    "combo": kind,
                })

    if system_check and system_check["status"] == "NG":
        for issue in system_check["issues"]:
            critical_issues.append({
                "type": "seismic_system_inadequate",
                "description": issue,
                "code_ref": "KDS 41 17 00 표 6.2-1",
                "combo": f"SDC {system_check['sdc']}",
            })

    if stability_check and stability_check["status"] == "NG":
        for dir_, ov in stability_check["overturning"].items():
            if ov["status"] == "NG":
                critical_issues.append({
                    "type": "overturning_instability",
                    "description": (
                        f"{dir_}방향 전도 안전율 {ov['FS']} < {OVERTURNING_FS_MIN} "
                        f"(M_r {ov['M_resist_kNm']} / M_ot {ov['M_overturn_kNm']} kN·m)"
                    ),
                    "code_ref": "전역안정 전도검토",
                    "combo": f"{dir_} 지진",
                })

    if pdelta_check and pdelta_check["status"] == "NG":
        for c in pdelta_check["checks"]:
            if c["status"] == "NG":
                critical_issues.append({
                    "type": "pdelta_instability",
                    "description": (
                        f"{c['story']}층 P-Delta 안정계수 θ={c['theta']} > "
                        f"θmax={pdelta_check['theta_limit']} (불안정)"
                    ),
                    "code_ref": "KDS 41 17 00 §7.2.7",
                    "combo": "지진",
                })

    # #10: 풍하중 사용성 층간변위 NG
    if wind_check and wind_check["status"] == "NG":
        for c in wind_check["checks"]:
            if c["status"] == "NG":
                critical_issues.append({
                    "type": "wind_drift_exceeded",
                    "description": (
                        f"{c['story']}층 {c['direction']}방향 풍하중 탄성 층간변위비 "
                        f"{c['elastic_drift']:.5f} ({c['drift_inv']}) > 허용 1/{wind_check['limit_inv']}"
                    ),
                    "code_ref": wind_check["code_ref"],
                    "combo": c["combo"],
                })

    # #12: 수직 비정형 (분류·경고 — overall 미게이트, '미반영' 은폐 방지)
    if vertical_check and vertical_check.get("irregular"):
        kinds = ", ".join(vertical_check.get("types", []))
        critical_issues.append({
            "type": "vertical_irregularity",
            "description": (
                f"수직 비정형 감지({kinds}) — KDS 41 17 00 표 5.3-2. "
                f"동적해석/내진상세 적용 검토 필요"
                + (" (극단 연층 포함)" if vertical_check.get("extreme") else "")
            ),
            "code_ref": "KDS 41 17 00 표 5.3-2",
            "combo": "지진",
        })

    # #8: 우발 비틀림 — 등가정적 모델은 ±5% 편심을 직접 적용하지 않음(은폐 방지)
    if accidental_torsion and not accidental_torsion.get("applied_in_model"):
        # None-safe: key가 None이어도 폴백 (get default는 key 부재만 처리).
        _ecc = (accidental_torsion.get("eccentricity_ratio") or 0.05) * 100
        _mtx = accidental_torsion.get("total_Mt_x_kNm") or 0.0
        _mty = accidental_torsion.get("total_Mt_y_kNm") or 0.0
        critical_issues.append({
            "type": "accidental_torsion_not_modeled",
            "description": (
                f"우발 비틀림(평면치수 ±{_ecc:g}% 편심) 모멘트 산정됨"
                f"(총 Mt_x={_mtx:.2f}·Mt_y={_mty:.2f} kN·m)이나 등가정적 해석모델엔 "
                f"미적용 — C2 비틀림은 해석 실변위(내재 비틀림)만 반영. 최종 설계는 ±5% 편심 "
                f"명시적 적용 권장"
            ),
            "code_ref": "KDS 41 17 00 §7.2.6.4",
            "combo": "지진",
        })

    # #13: 지진 입력 손상(error)으로 지진검토 미수행 — '안전' 단정 차단 (게이트 구멍 정정)
    if seismic_input_error:
        critical_issues.append({
            "type": "seismic_input_error",
            "description": (
                "지진하중 입력이 손상/누락되어(seismic_report error) 층간변위·내진설계범주·"
                "전도·P-Delta 등 지진검토가 수행되지 않았습니다 — 결과를 '안전'으로 단정할 수 "
                f"없습니다. 입력: {str(seismic_report.get('error'))[:120]}"
            ),
            "code_ref": "해석 입력 무결성",
            "combo": "지진",
        })

    # ── 4. Overall status ──
    drift_ok = drift_check is None or drift_check["status"] == "OK"
    member_ok = member_check is None or member_check["status"] == "OK"
    deflection_ok = deflection_check is None or deflection_check["status"] == "OK"
    system_ok = system_check is None or system_check["status"] == "OK"
    stability_ok = stability_check is None or stability_check["status"] == "OK"
    pdelta_ok = pdelta_check is None or pdelta_check["status"] == "OK"
    wind_ok = wind_check is None or wind_check["status"] == "OK"  # #10 사용성 게이트
    overall = "OK" if (drift_ok and member_ok and deflection_ok
                       and system_ok and stability_ok and pdelta_ok and wind_ok) else "NG"

    # ── 5. Summary ──
    # 지배 하중조합 (A2): 데이터는 이미 존재 — drift는 critical, member는 최대
    # 상관비 부재(members[0], interaction 내림차순 정렬)의 governing_combo.
    drift_governing_combo = None
    if drift_check and drift_check.get("critical"):
        drift_governing_combo = drift_check["critical"].get("combo")
    member_governing_combo = None
    if member_check and member_check.get("members"):
        member_governing_combo = member_check["members"][0].get("governing_combo")

    # #13: 미검토(not_checked) 항목 명시 — '전부 통과' 과대주장 방지.
    not_checked_items = []
    if seismic_input_error or drift_check is None:
        not_checked_items.append("층간변위(지진)" if seismic_input_error else "층간변위")
    if member_check is None:
        not_checked_items.append("부재강도")
    if deflection_check is None:
        not_checked_items.append("보 처짐(사용성)")
    if system_check is None:
        not_checked_items.append("내진설계범주/시스템")
    if stability_check is None:
        not_checked_items.append("전역안정(전도)")
    if pdelta_check is None:
        not_checked_items.append("P-Delta 안정")
    if torsion_check is None:
        not_checked_items.append("비틀림 비정형")
    if wind_check is None:
        not_checked_items.append("풍하중 사용성 변위")
    if vertical_check is None:
        not_checked_items.append("수직 비정형")

    summary = {
        "max_drift_ratio": drift_check["max_ratio"] if drift_check else None,
        "max_interaction_ratio": (
            member_check["summary"]["max_interaction_ratio"]
            if member_check else None
        ),
        "drift_governing_combo": drift_governing_combo,
        "member_governing_combo": member_governing_combo,
        # B1: 보 처짐(사용성). None=미검토 → §10에 "처짐 미검토" 명시.
        "max_deflection_ratio": (
            deflection_check["summary"]["max_deflection_ratio"] if deflection_check else None
        ),
        "deflection_status": deflection_check["status"] if deflection_check else "not_checked",
        # B2: 내진설계범주 + 시스템 적격성
        "sdc": system_check["sdc"] if system_check else None,
        "seismic_system_status": system_check["status"] if system_check else "not_checked",
        # C1: 전역안정(전도) / C3: P-Delta θ / C2: 비틀림 비정형
        "overturning_FS": stability_check["governing_FS"] if stability_check else None,
        "stability_status": stability_check["status"] if stability_check else "not_checked",
        "max_theta": pdelta_check["max_theta"] if pdelta_check else None,
        "pdelta_status": pdelta_check["status"] if pdelta_check else "not_checked",
        # T3-5: 1/(1−θ) 층단위 근사 — 세장 기둥은 부재 B1 별도 지배 가능(명시).
        "pdelta_amplification_method": (
            pdelta_check.get("amplification_method") if pdelta_check else None
        ),
        "pdelta_member_level_note": (
            bool(pdelta_check.get("member_level_b1_note")) if pdelta_check else False
        ),
        "torsion_ratio": torsion_check["max_ratio"] if torsion_check else None,
        "torsion_classification": torsion_check["classification"] if torsion_check else "not_checked",
        # #10: 풍하중 사용성 층간변위 (H/400)
        "max_wind_drift_ratio": wind_check["max_ratio"] if wind_check else None,
        "wind_drift_status": wind_check["status"] if wind_check else "not_checked",
        # #12: 수직 비정형 (분류)
        "vertical_irregularity": (vertical_check.get("types") if vertical_check else None),
        "vertical_irregular": bool(vertical_check and vertical_check.get("irregular")),
        "vertical_irregularity_status": (
            ("irregular" if vertical_check.get("irregular") else "regular")
            if vertical_check else "not_checked"
        ),
        # #8: 우발 비틀림 (등가정적 모델 미적용 → applied=False 명시)
        "accidental_torsion_applied": (
            bool(accidental_torsion.get("applied_in_model")) if accidental_torsion else None
        ),
        # #13: 미검토 항목 + 지진입력 손상 게이트
        "not_checked_items": not_checked_items,
        "seismic_input_error": seismic_input_error,
        "ng_stories": (
            # 고유 NG 층수 (물리 층). 행(층×방향×조합) 개수가 아니라 층 단위로 집계.
            len({c["story"] for c in drift_check["checks"] if c["status"] == "NG"})
            if drift_check else 0
        ),
        "ng_members": member_check["summary"]["ng"] if member_check else 0,
        "ng_beams_deflection": deflection_check["summary"]["ng"] if deflection_check else 0,
        # Tier1-1/2/5: 기둥 K / LTB 지배 보 수 / RSA 미검토 조합
        "column_K": member_check["summary"].get("column_K") if member_check else None,
        "n_ltb_governed": member_check["summary"].get("n_ltb_governed", 0) if member_check else 0,
        "rsa_unchecked_combos": (
            member_check["summary"].get("rsa_unchecked_combos", []) if member_check else []
        ),
        # Tier2-6/7: 세장비/콤팩트 — §10 리포트(renderSec10)가 dc.summary에서 직접 읽으므로
        # member_check.summary에서 최상위 summary로 전파(미전파 시 §10 경고 침묵 — 은폐 방지).
        "n_slenderness_ng": member_check["summary"].get("n_slenderness_ng", 0) if member_check else 0,
        "max_slenderness": member_check["summary"].get("max_slenderness", 0) if member_check else 0,
        "n_noncompact": member_check["summary"].get("n_noncompact", 0) if member_check else 0,
        "n_compression_slender": (
            member_check["summary"].get("n_compression_slender", 0) if member_check else 0
        ),
    }

    # Tier2-14: 결과 dict 경계에서 NaN/Inf → None 정리 (json.dumps 500 방지).
    return _sanitize_nonfinite({
        "overall_status": overall,
        "drift_check": drift_check,
        "member_check": member_check,
        "deflection_check": deflection_check,
        "system_check": system_check,
        "stability_check": stability_check,
        "pdelta_check": pdelta_check,
        "torsion_check": torsion_check,
        "wind_check": wind_check,                # #10
        "vertical_check": vertical_check,        # #12
        "accidental_torsion": accidental_torsion,  # #8
        "critical_issues": critical_issues,
        "summary": summary,
    })
