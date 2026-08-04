"""자동 하중 생성 모듈.

BuildingModel에서 KDS DB를 조회하여 설계하중을 자동 생성한다.
- 중력하중 (DL/LL): KDS 41 12 00 §2, §3
- 지진하중 (EQX/EQY): KDS 41 17 00 등가정적해석법
- 풍하중 (WX/WY): KDS 41 12 00 §5
- 하중조합: KDS 41 12 00 §1.7

Usage:
    from core.building_model import BuildingModel
    from core.load_generator import generate_all_loads

    model = BuildingModel.from_json(config)
    load_cases, combinations, report = generate_all_loads(model)
"""
from __future__ import annotations

import math
from typing import Optional

from core.building_model import BuildingModel, StoryInfo
from core.kds_loads import query_design_loads, query_hazard_values
from core.design_spectrum import (
    _get_zone_coefficient,
    _get_site_coefficients,
    _compute_site_spectrum_params,
    _linear_interp_z,
)


# ============================================================
# 용도 → DB primary_key 매핑
# ============================================================

# usage → live_load DB primary_key (기본 sub-key)
USAGE_TO_LIVE_LOAD_KEY = {
    "office": ("office", "office_general"),
    "residential": ("residential", "residential_general"),
    "retail": ("retail", "retail_general"),
    "parking": ("parking_passenger", "parking_passenger_general"),
    "parking_heavy": ("parking", "parking_truck"),
    "assembly": ("assembly", None),
    "assembly_fixed": ("assembly", "assembly_fixed_seats"),
    "assembly_movable": ("assembly", "assembly_movable_seats"),
    "storage": ("storage_light", None),
    "storage_light": ("storage_light", None),
    "storage_heavy": ("storage_heavy", None),
    "hospital": ("hospital", "hospital_ward"),
    "school": ("school", None),
    "library": ("library", "library_reading"),
    "library_reading": ("library", "library_reading"),
    "library_stack": ("library", "library_stack"),
    "corridor": ("corridor", "corridor_above_1f"),
    "restaurant": ("restaurant", None),
    "hotel": ("hotel", None),
    "factory": ("factory", None),
    "factory_light": ("factory", "factory_light"),
    "factory_heavy": ("factory", "factory_heavy"),
    "gym": ("gym", None),
    "roof": ("roof", None),
    "balcony": ("balcony", None),
    "mechanical_room": ("mechanical_room", None),
}

# usage → 기본 활하중 (DB 조회 실패 시 fallback, kN/m²)
FALLBACK_LIVE_LOADS = {
    "office": 2.5,
    "residential": 2.0,
    "retail": 4.0,
    "parking": 3.0,
    "assembly": 5.0,
    "storage": 6.0,
    "hospital": 2.0,
    "school": 3.0,
    "library": 3.0,
    "corridor": 5.0,
    "restaurant": 3.0,
    "hotel": 2.0,
    "factory": 6.0,
    "gym": 5.0,
    "roof": 1.0,
    "balcony": 3.0,
    "mechanical_room": 5.0,
}

# 내진설계 시스템 → DB primary_key 매핑
SEISMIC_SYSTEM_MAP = {
    # 철골 모멘트골조
    "special_moment_frame": "3-a",       # R=8, Ω₀=3, Cd=5.5
    "intermediate_moment_frame": "3-b",  # R=4.5, Ω₀=3, Cd=4
    "ordinary_moment_frame": "3-c",      # R=3.5, Ω₀=3, Cd=3
    # RC 모멘트골조
    "rc_special_moment_frame": "4-a",    # R=8, Ω₀=2.5, Cd=4
    "rc_intermediate_moment_frame": "4-b",  # R=7, Ω₀=2.5, Cd=5.5
    "rc_ordinary_moment_frame": "4-c",   # R=8, Ω₀=2.5, Cd=4
    # RC 전단벽
    "rc_special_shear_wall": "1-a",      # R=5, Ω₀=2.5, Cd=5
    "rc_ordinary_shear_wall": "1-b",     # R=4, Ω₀=2.5, Cd=4
    # 철골 가새골조
    "special_braced_frame": "2-a",       # R=8, Ω₀=2, Cd=4
    "ordinary_braced_frame": "2-q",      # R=1.5
}

# 근사고유주기 계수 → DB primary_key
PERIOD_CT_MAP = {
    "special_moment_frame": "steel_moment_frame",
    "intermediate_moment_frame": "steel_moment_frame",
    "ordinary_moment_frame": "steel_moment_frame",
    "rc_special_moment_frame": "rc_moment_frame",
    "rc_intermediate_moment_frame": "rc_moment_frame",
    "rc_ordinary_moment_frame": "rc_moment_frame",
    "rc_special_shear_wall": "rc_shear_wall",
    "rc_ordinary_shear_wall": "rc_shear_wall",
    "special_braced_frame": "steel_braced_frame",
    "ordinary_braced_frame": "steel_braced_frame",
}


# ============================================================
# 1. 중력하중 (DL/LL) 생성
# ============================================================

def _query_live_load(usage: str) -> float:
    """용도에 따른 활하중 DB 조회 (kN/m²)."""
    usage = (usage or "").lower()  # 키는 소문자 — 혼합대소문자 입력 정규화
    key_info = USAGE_TO_LIVE_LOAD_KEY.get(usage)
    if key_info is None:
        return FALLBACK_LIVE_LOADS.get(usage, 2.5)

    primary_key, secondary_key = key_info
    result = query_design_loads("live_load", "distributed", primary_key)

    if result["status"] == "success":
        records = result["records"]
        # secondary_key가 지정되면 해당 레코드 우선
        if secondary_key:
            for rec in records:
                if rec.get("secondary_key") == secondary_key and rec.get("value") is not None:
                    return rec["value"]
        # secondary_key 매칭 실패 시, value가 있는 첫 레코드
        for rec in records:
            if rec.get("value") is not None:
                return rec["value"]

    return FALLBACK_LIVE_LOADS.get(usage, 2.5)


def _query_live_load_traced(usage: str) -> dict:
    """활하중 DB 조회 + 출처 추적 (assumption_tracker 전용).

    _query_live_load와 동일 로직이나 value + source 정보를 함께 반환한다.

    Returns:
        {"value": float, "source": str, "db_primary_key": str|None, "db_secondary_key": str|None}
    """
    usage = (usage or "").lower()  # 키는 소문자 — 혼합대소문자 입력 정규화
    key_info = USAGE_TO_LIVE_LOAD_KEY.get(usage)
    if key_info is None:
        return {
            "value": FALLBACK_LIVE_LOADS.get(usage, 2.5),
            "source": "fallback_unmapped",
            "db_primary_key": None,
            "db_secondary_key": None,
        }

    primary_key, secondary_key = key_info
    result = query_design_loads("live_load", "distributed", primary_key)

    if result["status"] == "success":
        records = result["records"]
        if secondary_key:
            for rec in records:
                if rec.get("secondary_key") == secondary_key and rec.get("value") is not None:
                    return {
                        "value": rec["value"],
                        "source": "db_lookup",
                        "db_primary_key": primary_key,
                        "db_secondary_key": secondary_key,
                    }
        for rec in records:
            if rec.get("value") is not None:
                return {
                    "value": rec["value"],
                    "source": "db_lookup",
                    "db_primary_key": primary_key,
                    "db_secondary_key": rec.get("secondary_key"),
                }

    return {
        "value": FALLBACK_LIVE_LOADS.get(usage, 2.5),
        "source": "fallback_db_miss",
        "db_primary_key": primary_key,
        "db_secondary_key": secondary_key,
    }


# ============================================================
# 활하중 면적저감 (KDS 41 12 00 §3.5) — opt-in (기본 off, 하위호환)
# ============================================================
# 데이터: data/kds_output/03_live_reduction_normalized.json (§3.5 등분포활하중 저감).
# 본 시스템은 기본적으로 저감을 적용하지 않아 보수적이다 — 본 기능은 명시적으로
# 켤 때만(model.live_load_reduction=True) 동작하며, 기본 호출경로는 무변경이다.

LIVE_REDUCTION_MIN_AREA_M2 = 36.0           # 저감 적용 최소 영향면적 (§3.5.1)
LIVE_REDUCTION_HEAVY_THRESHOLD = 5.0        # 저감 불가 활하중 기준 (kN/m², §3.5.3)
LIVE_REDUCTION_MIN_1FLOOR = 0.5             # 1개 층 지지 최소 저감계수 (§3.5.3)
LIVE_REDUCTION_MIN_MULTIFLOOR = 0.4         # 2개 층 이상 지지 최소 저감계수 (§3.5.3)
LIVE_REDUCTION_HEAVY_MULTIFLOOR_MIN = 0.8   # >5kN/m²·승용주차장 + 2층 이상 지지 시 최소 (§3.5.3)


def live_load_reduction_factor(
    influence_area_m2: float,
    *,
    floors_supported: int = 1,
    live_load_kNm2: float | None = None,
    occupancy: str | None = None,
) -> dict:
    """KDS 41 12 00 §3.5 등분포활하중 저감계수 C를 산정한다 (opt-in).

    C = 0.3 + 4.2/√A (영향면적 A ≥ 36 m²), 최소 0.5(1층 지지)/0.4(2층 이상).
    예외(§3.5.3): 활하중 > 5 kN/m²는 저감 불가(단 2층 이상 지지 시 C ≥ 0.8);
    공중집회(assembly, ≤5 kN/m²)는 저감 불가; 승용차 전용 주차장은 저감 불가(2층
    이상 지지 시 C ≥ 0.8). 지붕활하중(§3.6, L=Lo·R1·R2)은 별도식 — 본 함수 미적용.

    기본값(미적용)은 C=1.0 — 본 함수가 호출되지 않는 한 시스템은 보수적(저감 없음)
    으로 동작한다(하위호환).

    Args:
        influence_area_m2: 영향면적 A (= 부하면적 × 4(기둥/기초)·2(보/벽)·1(슬래브), §3.5.2).
        floors_supported:  부재가 지지하는 층 수 (최소 저감계수 결정).
        live_load_kNm2:    기본 등분포활하중 (kN/m²) — 5 초과 시 저감 예외 판정.
        occupancy:         canonical usage 키 (assembly/parking 등) — 저감 예외 판정.

    Returns:
        {"factor": C(0~1), "applied": bool, "reason": str,
         "influence_area_m2": A, "floors_supported": n, "clause": "KDS 41 12 00 3.5"}
    """
    A = float(influence_area_m2 or 0.0)
    n_floors = max(1, int(floors_supported or 1))
    multi = n_floors >= 2
    usage = (occupancy or "").lower()
    base = {"influence_area_m2": round(A, 2), "floors_supported": n_floors,
            "clause": "KDS 41 12 00 3.5"}

    def _result(factor: float, applied: bool, reason: str) -> dict:
        return {**base, "factor": round(min(max(factor, 0.0), 1.0), 4),
                "applied": applied, "reason": reason}

    # 예외 1: 영향면적 미달 → 저감 없음 (§3.5.1)
    if A < LIVE_REDUCTION_MIN_AREA_M2:
        return _result(1.0, False,
                       f"영향면적 {A:.1f}m² < {LIVE_REDUCTION_MIN_AREA_M2:.0f}m² (저감 미적용)")

    is_heavy = (isinstance(live_load_kNm2, (int, float))
                and live_load_kNm2 > LIVE_REDUCTION_HEAVY_THRESHOLD)

    # 예외 2: 공중집회(≤5kN/m²) → 저감 불가 (§3.5.3)
    if usage.startswith("assembly") and not is_heavy:
        return _result(1.0, False, "공중집회 용도(≤5kN/m²) — 저감 불가(§3.5.3)")

    # 저감식 C = 0.3 + 4.2/√A (상한 1.0), 최소 0.5(1층)/0.4(2층 이상)
    C = min(0.3 + 4.2 / math.sqrt(A), 1.0)

    # 예외 3: 중하중(>5kN/m²) 또는 승용차 주차장 → 1층 지지 저감 불가, 2층 이상 C≥0.8
    restricted = is_heavy or usage.startswith("parking")
    if restricted:
        label = "활하중>5kN/m²" if is_heavy else "승용차 주차장"
        if not multi:
            return _result(1.0, False, f"{label}, 1층 지지 → 저감 불가(§3.5.3)")
        C = max(C, LIVE_REDUCTION_HEAVY_MULTIFLOOR_MIN)
        return _result(C, True, f"{label}, 2층 이상 지지 → C≥{LIVE_REDUCTION_HEAVY_MULTIFLOOR_MIN:g}")

    floor_min = LIVE_REDUCTION_MIN_MULTIFLOOR if multi else LIVE_REDUCTION_MIN_1FLOOR
    C = max(C, floor_min)
    return _result(C, True, f"C=0.3+4.2/√{A:.0f}, 최소 {floor_min:g}({n_floors}층 지지)")


def _representative_live_influence_area(model: BuildingModel, story: int) -> float:
    """대표 보(beam) 영향면적 추정 (m²) — 평균 경간 패널 × 2(보/벽 기준, §3.5.2).

    정형 그리드는 평균 경간 패널(b̄x·b̄y)을 부하면적으로 보고 보 기준 배수 ×2를
    적용한다(하중이 보 선하중으로 분배되므로 보 영향면적이 대표적). 경간 미상(비정형
    등)이면 층 바닥면적으로 폴백한다(근사 — 본 저감은 opt-in 보조기능).
    """
    bx = list(model.bays_x or [])
    by = list(model.bays_y or [])
    if bx and by:
        panel = (sum(bx) / len(bx)) * (sum(by) / len(by))   # 평균 패널 부하면적 (m²)
        return panel * 2.0                                  # 보/벽 영향면적 (§3.5.2 ×2)
    return model.floor_area_at_story(story)


def generate_gravity_loads(
    model: BuildingModel, apply_live_reduction: bool = False
) -> dict[str, list[dict]]:
    """BuildingModel → DL/LL 하중케이스 생성.

    DL = 슬래브 자중 + 마감재 + 설비
    LL = 용도별 DB 조회

    Args:
        model: 건물 모델.
        apply_live_reduction: True면 등분포활하중 면적저감(KDS 41 12 00 §3.5)을
            적용한다. **기본 False(보수, 하위호환)** — 기본 경로는 저감 없이 동작한다.
            저감 적용 시 report[*]["live_reduction"]에 산정 근거를 기록한다.

    Returns:
        {"DL": [...], "LL": [...]}  각 항목은 frame_3d의 load entry 형식
    """
    dl_loads = []
    ll_loads = []
    report = []

    for story_info in model.stories:
        s = story_info.story

        # DL: 슬래브 자중 + 마감재 + 설비
        rc_unit_weight = 24.0  # kN/m³ (철근콘크리트)
        slab_self = rc_unit_weight * story_info.slab_thickness  # kN/m²
        finish = story_info.dead_load_finish  # kN/m²
        mep = 0.5  # kN/m² (설비 경험치)
        dl_total = slab_self + finish + mep

        dl_loads.append({
            "type": "floor_area",
            "story": s,
            "value": round(dl_total, 2),
        })

        # LL: 용도별 활하중 (opt-in 면적저감)
        ll_value = _query_live_load(story_info.usage)
        ll_reduction = None
        if apply_live_reduction:
            A_inf = _representative_live_influence_area(model, s)
            red = live_load_reduction_factor(
                A_inf, floors_supported=1,
                live_load_kNm2=ll_value, occupancy=story_info.usage)
            if red["applied"] and red["factor"] < 1.0:
                ll_orig = ll_value
                ll_value = ll_value * red["factor"]
                ll_reduction = {**red,
                                "LL_original_kNm2": round(ll_orig, 2),
                                "LL_reduced_kNm2": round(ll_value, 2)}
        ll_loads.append({
            "type": "floor_area",
            "story": s,
            "value": round(ll_value, 2),
        })

        rpt = {
            "story": s,
            "usage": story_info.usage,
            "DL_kNm2": round(dl_total, 2),
            "DL_breakdown": {
                "slab_self": round(slab_self, 2),
                "finish": finish,
                "mep": mep,
            },
            "LL_kNm2": round(ll_value, 2),
        }
        if ll_reduction:
            rpt["live_reduction"] = ll_reduction
        report.append(rpt)

    return {
        "load_cases": {"DL": dl_loads, "LL": ll_loads},
        "report": report,
    }


# ============================================================
# 2. 지진하중 (등가정적해석법, KDS 41 17 00)
# ============================================================

def _get_seismic_system_params(system_key: str) -> dict:
    """내진시스템 R, Ω₀, Cd 조회."""
    db_key = SEISMIC_SYSTEM_MAP.get(system_key, "3-c")  # default: 보통모멘트골조
    result = query_design_loads("seismic_design", "seismic_force_resisting_system")

    if result["status"] == "success":
        for rec in result["records"]:
            if rec["primary_key"] == db_key:
                cond = rec.get("conditions", [])
                params = {}
                for c in cond:
                    if "=" in c:
                        k, v = c.split("=", 1)
                        try:
                            params[k] = float(v)
                        except ValueError:
                            params[k] = v
                return {
                    "R": params.get("R", 3.5),
                    "omega_0": params.get("omega_0", 3.0),
                    "Cd": params.get("Cd", 3.0),
                    "db_key": db_key,
                }

    # Fallback for ordinary moment frame
    return {"R": 3.5, "omega_0": 3.0, "Cd": 3.0, "db_key": db_key}


def _get_period_coefficients(system_key: str) -> tuple[float, float]:
    """근사고유주기 계수 Ct, x 조회."""
    ct_key = PERIOD_CT_MAP.get(system_key, "steel_moment_frame")
    result = query_design_loads("seismic_design", "approximate_period_Ct", ct_key)

    if result["status"] == "success" and result["records"]:
        rec = result["records"][0]
        Ct = rec.get("value", 0.0724)
        # x 값은 conditions에서 추출
        cond = rec.get("conditions", [])
        x = 0.8  # default
        for c in cond:
            if c.startswith("x="):
                try:
                    x = float(c.split("=")[1])
                except ValueError:
                    pass
        return Ct, x

    # Fallback
    return 0.0724, 0.8


# DB primary_key → 표 7.2-1 anchor SD1 값 (KDS 41 17 00 §7.2.3 표 7.2-1)
_CU_SD1_ANCHORS: tuple[tuple[str, float], ...] = (
    ("Sd1_le_0.1", 0.10),
    ("Sd1_0.15",   0.15),
    ("Sd1_0.2",    0.20),
    ("Sd1_0.3",    0.30),
    ("Sd1_ge_0.4", 0.40),
)

# KDS 41 17 00 표 7.2-1 fallback values (DB 조회 실패 시)
_CU_FALLBACK_ANCHORS: list[tuple[float, float]] = [
    (0.10, 1.7), (0.15, 1.6), (0.20, 1.5), (0.30, 1.4), (0.40, 1.4),
]


def _get_period_upper_limit(SD1: float) -> float:
    """주기상한계수 Cu 조회 (KDS 41 17 00 표 7.2-1 직선보간).

    KDS 41 17 00 §7.2.3 (1) 표 7.2-1 끝부분이 명시한다:
        "SD1의 중간값에 해당할 경우 주기상한계수 Cu는 직선보간한다."

    이전 구현은 SD1을 anchor SD1 값(0.10/0.15/0.20/0.30/0.40)으로 버킷화하여
    anchor의 Cu를 그대로 반환했으나, 본 함수는 anchor 5행을 DB에서 모두 조회한
    뒤 실제 SD1로 직선보간한다. (Fa/Fv 보간 fix와 동일 패턴.)
    범위 밖(SD1 < 0.1 또는 SD1 > 0.4)은 표 경계값으로 clamp.
    """
    result = query_design_loads("seismic_design", "period_upper_limit_Cu")
    if result["status"] != "success":
        return _linear_interp_z(SD1, _CU_FALLBACK_ANCHORS)

    Cu_map = {rec["primary_key"]: rec.get("value", 1.4) for rec in result["records"]}

    anchors: list[tuple[float, float]] = []
    for sec_key, sd1_anchor in _CU_SD1_ANCHORS:
        if sec_key in Cu_map:
            anchors.append((sd1_anchor, Cu_map[sec_key]))

    if not anchors:
        anchors = _CU_FALLBACK_ANCHORS

    return _linear_interp_z(SD1, anchors)


# ============================================================
# 유효지진중량 활하중 분담 (KDS 41 17 00)
# ============================================================

# 창고/공장류 — 유효지진중량 W에 활하중의 25%를 포함하는 용도 (KDS 41 17 00 §4.2).
# 그 외 일반 용도(사무실·주거 등)는 활하중 미포함(0%)으로 단순화한다.
SEISMIC_STORAGE_USAGES = frozenset({
    "storage", "storage_light", "storage_heavy",
    "factory", "factory_light", "factory_heavy",
})


def seismic_live_load_fraction(usage: str) -> float:
    """유효지진중량 W에 포함할 활하중 분담률.

    KDS 41 17 00 §4.2: 창고류(storage/factory)는 활하중의 25%를 유효지진중량에
    포함한다. 일반 용도는 0%(관례적 단순화). modal 질량(_estimate_story_weights)도
    동일 규칙을 따라 일관성을 유지한다.
    """
    return 0.25 if (usage or "").lower() in SEISMIC_STORAGE_USAGES else 0.0


def _calculate_story_weights(model: BuildingModel, dl_loads: list[dict]) -> list[float]:
    """층별 유효 중량 계산 (kN).

    W = DL × floor_area + f_LL × LL × floor_area
       (f_LL: 창고류 0.25, 그 외 0 — KDS 41 17 00 §4.2)
    비정형 건물은 층별 바닥면적이 다를 수 있음 (setback 등).
    """
    usage_by_story = {si.story: si.usage for si in model.stories}
    weights = []
    for ld in dl_loads:
        story = ld.get("story", 1)
        area = model.floor_area_at_story(story)
        w = ld["value"] * area  # DL 분담 (kN)
        # 창고류 활하중 25% (KDS 41 17 00) — 비창고는 frac=0 → 기존과 동일.
        usage = usage_by_story.get(story, "office")
        frac = seismic_live_load_fraction(usage)
        if frac > 0:
            ll_value = _query_live_load(usage)  # kN/m²
            w += frac * ll_value * area
        weights.append(round(w, 2))
    return weights


def generate_seismic_loads(
    model: BuildingModel,
    story_weights: list[float],
) -> dict:
    """등가정적 지진하중 생성 (KDS 41 17 00).

    Args:
        model: BuildingModel
        story_weights: 층별 유효 중량 (kN), DL 기반

    Returns:
        {"load_cases": {"EQX": [...], "EQY": [...]}, "report": {...}}
    """
    if not model.region:
        return {"load_cases": {}, "report": {"error": "region not specified"}}

    # 1. 구역계수 z
    z, zone_desc = _get_zone_coefficient(model.region)

    # 2. 지반계수 Fa, Fv
    Fa, Fv = _get_site_coefficients(model.site_class, z)

    # 3. 스펙트럼 파라미터
    # SDS/SD1은 '순수 지반위험도'(z 기반)이며 중요도계수 Ie는 포함하지 않는다.
    # Ie는 아래 Cs = SDS/(R/Ie)에서만 한 번 적용한다. (구버전은 z_eff=z·Ie로 SDS에
    # Ie를 먼저 곱하고 Cs에서 또 곱해 V·Fx·Ev가 Ie배(특 1.5/I 1.2) 과다 산정되었다.)
    IE = model.importance_factor
    params = _compute_site_spectrum_params(z, Fa, Fv)
    SDS = params["SDS"]
    SD1 = params["SD1"]

    # 4. 내진시스템 R, Ω₀, Cd
    sys_params = _get_seismic_system_params(model.seismic_system)
    R = sys_params["R"]
    Omega_0 = sys_params["omega_0"]
    Cd_sys = sys_params["Cd"]

    # 5. 근사 고유주기 Ta
    Ct, x = _get_period_coefficients(model.seismic_system)
    hn = model.total_height  # m
    Ta = Ct * (hn ** x)

    # 주기 상한
    Cu = _get_period_upper_limit(SD1)
    T = min(Ta, Cu * Ta)  # 근사주기 사용 (고유치 미수행)

    # 6. 지진응답계수 Cs
    Cs = SDS / (R / IE)
    # 하한
    Cs_min = max(0.044 * SDS * IE, 0.01)
    # 상한
    if T > 0:
        Cs_max = SD1 / (T * (R / IE))
    else:
        Cs_max = Cs
    Cs = max(Cs, Cs_min)
    Cs = min(Cs, Cs_max)

    # 7. 밑면 전단력 V
    W = sum(story_weights)
    V = Cs * W

    # 8. 층별 횡력 분배 Fx
    # Cvx = (wx × hx^k) / Σ(wi × hi^k)
    # k = 1 if T ≤ 0.5, k = 2 if T ≥ 2.5, linear interpolation between
    if T <= 0.5:
        k = 1.0
    elif T >= 2.5:
        k = 2.0
    else:
        k = 1.0 + (T - 0.5) / 2.0

    cumulative_h = model.cumulative_heights
    sum_wh = sum(w * (h ** k) for w, h in zip(story_weights, cumulative_h))

    eq_x_loads = []
    eq_y_loads = []
    story_forces = []

    for i, (w, h) in enumerate(zip(story_weights, cumulative_h)):
        if sum_wh > 0:
            Cvx = (w * (h ** k)) / sum_wh
        else:
            Cvx = 1.0 / len(story_weights)
        Fx = Cvx * V

        story_num = i + 1
        story_forces.append({
            "story": story_num,
            "weight_kN": round(w, 2),
            "height_m": round(h, 2),
            "Cvx": round(Cvx, 4),
            "Fx_kN": round(Fx, 2),
        })

        if model.seismic_direction in ("x", "both"):
            eq_x_loads.append({
                "type": "lateral_x",
                "story": story_num,
                "value": round(Fx, 2),
            })
        if model.seismic_direction in ("y", "both"):
            eq_y_loads.append({
                "type": "lateral_y",
                "story": story_num,
                "value": round(Fx, 2),
            })

    load_cases = {}
    if eq_x_loads:
        load_cases["EQX"] = eq_x_loads
    if eq_y_loads:
        load_cases["EQY"] = eq_y_loads

    report = {
        "method": "equivalent_lateral_force",
        "code": "KDS 41 17 00",
        "region": zone_desc,
        "z": z,
        "z_effective": z,  # SDS/SD1는 z 기반(중요도 Ie는 Cs에서만 1회 적용)
        "IE": IE,
        "site_class": model.site_class,
        "Fa": Fa,
        "Fv": Fv,
        "SDS": SDS,
        "SD1": SD1,
        "seismic_system": model.seismic_system,
        "R": R,
        "omega_0": Omega_0,
        "Cd": Cd_sys,
        "Ct": Ct,
        "x": x,
        "hn_m": hn,
        "Ta_sec": round(Ta, 4),
        "Cu": Cu,
        "T_sec": round(T, 4),
        "k": round(k, 2),
        "Cs": round(Cs, 6),
        "Cs_min": round(Cs_min, 6),
        "Cs_max": round(Cs_max, 6),
        # T3-7: Cs_min에 IE 적용 — 조항·해석 명시. KDS 41 17 00은 Cs=SDS·IE/R 형태로
        # IE를 분자에 두므로, 일관성을 위해 하한 Cs_min=0.044·SDS·IE에도 IE를 곱한다
        # (지진력이 IE배 증가하면 하한도 동일 비례). Cs_min이 V를 지배하면 밑면전단이
        # IE에 선형 민감 — 중요도 등급 변경 시 재검토 권장.
        "Cs_min_basis": "0.044·SDS·IE (lower-bound, IE applied consistently with Cs)",
        "Cs_governed_by": (
            "Cs_min" if abs(Cs - Cs_min) < 1e-9 else
            ("Cs_max" if abs(Cs - Cs_max) < 1e-9 else "Cs")
        ),
        "IE_applied_to_Cs_min": True,
        "W_kN": round(W, 2),
        "V_kN": round(V, 2),
        "story_forces": story_forces,
    }

    # 우발 비틀림 모멘트 (KDS 41 17 00 §7.2.6.4) — 각 층 지진력에 평면치수의 ±5% 편심.
    #   EQX(X방향 횡력) ⟂ 치수 = total_width_y → Mt_x = Fx·0.05·By
    #   EQY(Y방향 횡력) ⟂ 치수 = total_width_x → Mt_y = Fy·0.05·Bx (Fy=Fx 동일 크기)
    # 등가정적 모델은 우발편심을 하중으로 직접 적용하지 않으므로(applied_in_model=False)
    # 본 값은 '산정·문서화'이며 design_check가 '미반영' 주의를 띄운다(C2 비틀림은 해석
    # 실변위(내재 비틀림)만 사용 — 우발편심 별도).
    ecc = 0.05
    By = model.total_width_y or 0.0
    Bx = model.total_width_x or 0.0
    story_moments = [
        {
            "story": sfo["story"],
            "Fx_kN": sfo["Fx_kN"],
            "Mt_x_kNm": round(sfo["Fx_kN"] * ecc * By, 2),
            "Mt_y_kNm": round(sfo["Fx_kN"] * ecc * Bx, 2),
        }
        for sfo in story_forces
    ]
    report["accidental_torsion"] = {
        "code_ref": "KDS 41 17 00 §7.2.6.4 (우발 비틀림 — 평면치수 ±5% 편심)",
        "eccentricity_ratio": ecc,
        "plan_dim_x_m": round(Bx, 2),
        "plan_dim_y_m": round(By, 2),
        "applied_in_model": False,
        "story_moments": story_moments,
        "total_Mt_x_kNm": round(sum(s["Mt_x_kNm"] for s in story_moments), 2),
        "total_Mt_y_kNm": round(sum(s["Mt_y_kNm"] for s in story_moments), 2),
    }

    return {"load_cases": load_cases, "report": report}


# ============================================================
# 3. 풍하중 (KDS 41 12 00 §5)
# ============================================================

# KDS 41 12 00 §5.5.4 지표면조도구분별 풍속고도분포계수 Kzr (표 5.5-2 / 표 5.5-3)
# 설계풍속 Vz = V0·Kd·Kzr·Kzt·Iw (식 5.5-2), 속도압 qz = 0.5·ρ·Vz² (식 5.5-1)
#   → Kzr은 속도에 곱해지는 고도분포계수이므로 qz에서 제곱된다.
#   z ≤ zb        : Kzr = kzr_zb (고정)
#   zb < z ≤ Zg   : Kzr = coeff · z^alpha
KDS_ROUGHNESS_PARAMS = {
    # 조도구분: zb(경계층시작높이 m), zg(기준경도풍높이 m), alpha(고도분포지수), coeff(z>zb 계수), kzr_zb(z≤zb 고정 Kzr)
    "A": {"zb": 20.0, "zg": 550.0, "alpha": 0.33, "coeff": 0.22, "kzr_zb": 0.58},  # 대도시 중심부
    "B": {"zb": 15.0, "zg": 450.0, "alpha": 0.22, "coeff": 0.45, "kzr_zb": 0.81},  # 수목·중층건물(4~9층) 산재
    "C": {"zb": 10.0, "zg": 350.0, "alpha": 0.15, "coeff": 0.71, "kzr_zb": 1.00},  # 저층 장애물 산재 (V0 산정 기준 조도)
    "D": {"zb": 5.0,  "zg": 250.0, "alpha": 0.10, "coeff": 0.98, "kzr_zb": 1.13},  # 해안·초원·비행장
}


def _velocity_pressure_coeff(z_height: float, exposure: str) -> float:
    """KDS 41 12 00 §5.5.4 풍속고도분포계수 Kzr (표 5.5-2 / 표 5.5-3).

    Vz = V0·Kzr·Kzt (식 5.5-2, Kd=Iw=1 간략화), qz = 0.5·ρ·Vz² (식 5.5-1).
    반환하는 Kzr은 속도에 곱해지는 계수이므로 호출부의 qz 계산에서 제곱된다.
      z ≤ zb      : Kzr = kzr_zb (고정)
      zb < z ≤ Zg : Kzr = coeff · z^alpha
    """
    p = KDS_ROUGHNESS_PARAMS.get(exposure, KDS_ROUGHNESS_PARAMS["C"])
    if z_height <= p["zb"]:
        return p["kzr_zb"]
    z = min(z_height, p["zg"])
    return p["coeff"] * z ** p["alpha"]


def generate_wind_loads(model: BuildingModel) -> dict:
    """풍하중 생성 (KDS 41 12 00 §5, 간략화).

    간소화 가정:
    - Kzt = 1.0 (평탄지)
    - Gf = 가스트계수 (노풍도별 약식)
    - Cp = 0.8 (풍상) + 0.5 (풍하) = 1.3 (총 풍력계수)

    Returns:
        {"load_cases": {"WX": [...], "WY": [...]}, "report": {...}}
    """
    if not model.region:
        return {"load_cases": {}, "report": {"error": "region not specified"}}

    # 1. 기본풍속 V₀ 조회
    hazard = query_hazard_values(model.region, "wind_v0")
    if hazard["status"] != "success" or not hazard["regions"]:
        return {"load_cases": {}, "report": {"error": f"wind_v0 not found for {model.region}"}}

    # T3-6: wind_v0 키가 실제로 있는지 확인 — 누락 시 기본값(26 m/s) 폴백을 경고로 명시
    # (침묵 폴백이 임의 풍속으로 설계되는 것을 방지).
    _v0_raw = hazard["regions"][0].get("wind_v0")
    v0_fallback = not isinstance(_v0_raw, (int, float)) or _v0_raw <= 0
    V0 = 26.0 if v0_fallback else _v0_raw  # m/s

    # 2. 풍하중 계수
    exposure = model.exposure_category
    Kzt = 1.0  # 평탄지
    rho = 1.225  # kg/m³ (공기밀도)

    # 가스트영향계수 (간략화)
    Gf_map = {"A": 2.50, "B": 2.20, "C": 1.85, "D": 1.65}
    Gf = Gf_map.get(exposure, 2.20)

    # 풍력계수 (폐쇄형 건물, 주골조)
    Cp_total = 1.3  # 풍상면 0.8 + 풍하면 0.5

    # 3. 층별 풍하중 계산
    wx_loads = []
    wy_loads = []
    story_detail = []

    cumulative_h = model.cumulative_heights
    Bx = model.total_width_x  # 수풍면폭 (WY 풍향)
    By = model.total_width_y  # 수풍면폭 (WX 풍향)

    for i, story_info in enumerate(model.stories):
        h_mid = cumulative_h[i] - story_info.height / 2.0  # 층 중앙높이
        Kz = _velocity_pressure_coeff(h_mid, exposure)

        # 속도압 qz (kN/m²) — KDS 41 12 00 §5.5
        # Vz = V0·Kzr·Kzt (식 5.5-2), qz = 0.5·ρ·Vz² (식 5.5-1)
        # Kz(=Kzr, 표 5.5-2/5.5-3 풍속고도분포계수)는 속도계수이므로 속도항 안에서 제곱된다.
        qz = 0.5 * rho * (V0 * Kz * Kzt) ** 2 * 1e-3  # Pa → kN/m²

        # 풍압 p = qz × Gf × Cp (kN/m²)
        p = qz * Gf * Cp_total

        # X방향 풍력: p × 층높이 × Y폭 (수풍면)
        Fx = p * story_info.height * By
        # Y방향 풍력: p × 층높이 × X폭 (수풍면)
        Fy = p * story_info.height * Bx

        s = story_info.story
        wx_loads.append({
            "type": "lateral_x",
            "story": s,
            "value": round(Fx, 2),
        })
        wy_loads.append({
            "type": "lateral_y",
            "story": s,
            "value": round(Fy, 2),
        })

        story_detail.append({
            "story": s,
            "h_mid_m": round(h_mid, 2),
            "Kz": round(Kz, 4),
            "qz_kNm2": round(qz, 4),
            "p_kNm2": round(p, 4),
            "Fx_kN": round(Fx, 2),
            "Fy_kN": round(Fy, 2),
        })

    report = {
        "method": "simplified_wind",
        "code": "KDS 41 12 00 §5",
        "region": model.region,
        "V0_ms": V0,
        "V0_fallback": v0_fallback,  # T3-6: True면 기본값(26 m/s) 사용
        "exposure": exposure,
        "Kzt": Kzt,
        "Gf": Gf,
        "Cp_total": Cp_total,
        "Bx_m": Bx,
        "By_m": By,
        "total_Fx_kN": round(sum(l["value"] for l in wx_loads), 2),
        "total_Fy_kN": round(sum(l["value"] for l in wy_loads), 2),
        "story_detail": story_detail,
    }
    if v0_fallback:
        report["warnings"] = [
            f"기본풍속 V0가 '{model.region}' 위험도 DB에 없어 기본값 26 m/s로 폴백 — "
            "지역 실제 V0 확인 필요(풍하중은 V0²에 비례)."
        ]

    return {
        "load_cases": {"WX": wx_loads, "WY": wy_loads},
        "report": report,
    }


# ============================================================
# 3b. 적설하중 (KDS 41 12 00 §4)
# ============================================================

# 적설 중요도계수 Is (KDS 41 12 00 — 특 1.2 / 1등급 1.1 / 2등급 1.0 / 3등급 0.8)
_SNOW_IMPORTANCE = {"특": 1.2, "I": 1.1, "II": 1.0, "III": 0.8}


def generate_snow_loads(model: BuildingModel) -> dict:
    """지붕 적설하중 생성 (KDS 41 12 00 §4, 평지붕 약식).

    평지붕 적설하중:  S = C_b · C_e · C_t · I_s · S_g
      S_g : 기본지상적설하중 (지역별, hazard DB)
      C_b : 기본지붕적설하중계수 = 0.7
      C_e : 노출계수 = 1.0 (부분노출 가정)
      C_t : 온도계수 = 1.0 (난방건물 가정)
      I_s : 중요도계수 (특 1.2 / I 1.1 / II 1.0 / III 0.8)
    최소 지붕적설하중:  S_m = I_s·S_g (S_g ≤ 1.0) 또는 I_s·1.0 (S_g > 1.0) — 저경사 지붕.
    설계값 = max(S, S_m). 적설은 최상층(지붕)에만 floor_area 로 적용한다.

    Returns:
        {"load_cases": {"S": [...]} | {}, "report": {...}}
    """
    if not model.region:
        return {"load_cases": {}, "report": {"error": "region not specified"}}

    hazard = query_hazard_values(model.region, "snow_sg")
    if hazard["status"] != "success" or not hazard["regions"]:
        return {"load_cases": {}, "report": {"error": f"snow_sg not found for {model.region}"}}

    Sg = hazard["regions"][0].get("snow_sg")
    if not isinstance(Sg, (int, float)) or Sg <= 0:
        return {"load_cases": {}, "report": {"error": f"snow_sg invalid for {model.region}"}}

    Cb, Ce, Ct = 0.7, 1.0, 1.0
    Is = _SNOW_IMPORTANCE.get(model.importance, 1.0)
    S_flat = Cb * Ce * Ct * Is * Sg
    S_min = Is * Sg if Sg <= 1.0 else Is * 1.0
    S_design = max(S_flat, S_min)
    governed_by = "minimum" if S_min > S_flat else "formula"

    roof_story = max(si.story for si in model.stories)
    roof_area = model.floor_area_at_story(roof_story)

    s_loads = [{
        "type": "floor_area",
        "story": roof_story,
        "value": round(S_design, 3),
    }]

    report = {
        "method": "flat_roof_snow",
        "code": "KDS 41 12 00 §4",
        "region": model.region,
        "Sg_kNm2": round(Sg, 3),
        "Cb": Cb, "Ce": Ce, "Ct": Ct, "Is": Is,
        "S_formula_kNm2": round(S_flat, 3),
        "S_min_kNm2": round(S_min, 3),
        "S_design_kNm2": round(S_design, 3),
        "governed_by": governed_by,
        "roof_story": roof_story,
        "roof_area_m2": round(roof_area, 2),
        "total_S_kN": round(S_design * roof_area, 2),
        # T3-6: 평지붕 균형적설만 모델 — 불균형/표류(drift)·미끄럼·돌출부 적설은 미반영.
        "limitations": [
            "평지붕 균형적설(balanced)만 산정 — 불균형 적설·표류(drift, KDS 41 12 00 §4.5·4.6)·"
            "지붕 단차/돌출부·미끄럼 적설은 미모델링(경사·다단·차양 지붕은 별도 검토 필요).",
            "노출계수 Ce=1.0(보통노출)·온도계수 Ct=1.0(난방건물) 가정.",
        ],
    }

    return {"load_cases": {"S": s_loads}, "report": report}


# ============================================================
# 4. 하중조합 (KDS 41 12 00 §1.7)
# ============================================================

def generate_load_combinations(
    available_cases: list[str],
    sds: float = 0.0,
    orthogonal_seismic: bool = False,
) -> dict[str, dict[str, float]]:
    """KDS 41 12 00 §1.7 주요 하중조합 생성.

    사용 가능한 하중케이스에 따라 적절한 조합만 생성한다.

    Args:
        available_cases: 생성된 하중케이스 이름 목록 (예: ["DL", "LL", "EQX", "EQY", "WX", "WY"])
        sds: 설계스펙트럼가속도 S_DS (g). >0이면 수직지진성분 E_v=0.2·S_DS·D 를
            지진 하중조합의 고정하중 계수에 반영(KDS 41 17 00):
            하향 (1.2+0.2·S_DS)D, 양압 (0.9−0.2·S_DS)D.
            기본 0.0 → 1.2/0.9로 환원(하위호환: 벤치마크/직접호출 영향 없음).
        orthogonal_seismic: True면 직교방향 지진 100%+30% 조합을 추가 생성
            (KDS 41 17 00 §8.1.3 — 내진설계범주 C/D 의무). EQX·EQY 둘 다 존재할 때만
            적용. 기본 False → 하위호환(단방향·SDC A/B는 미생성, 정상건물 무영향).

    Returns:
        {"1.2DL+1.6LL": {"DL": 1.2, "LL": 1.6}, ...}
    """
    combos = {}
    has = set(available_cases)

    # 기본 중력 조합
    if "DL" in has and "LL" in has:
        combos["1.4DL"] = {"DL": 1.4}
        combos["1.2DL+1.6LL"] = {"DL": 1.2, "LL": 1.6}

    elif "DL" in has:
        combos["1.4DL"] = {"DL": 1.4}

    # 적설 조합 (KDS 41 10 00 LRFD — 적설은 지붕에만 작용)
    if "S" in has and "DL" in has:
        if "LL" in has:
            combos["1.2DL+1.6LL+0.5S"] = {"DL": 1.2, "LL": 1.6, "S": 0.5}
            combos["1.2DL+1.0LL+1.6S"] = {"DL": 1.2, "LL": 1.0, "S": 1.6}
        else:
            combos["1.2DL+1.6S"] = {"DL": 1.2, "S": 1.6}

    # 지진 조합 — 수직지진성분 Ev=0.2·SDS·D 를 D 계수에 포함 (KDS 41 17 00).
    #   하향: (1.2+0.2·SDS)D + 1.0L + 1.0E (+0.2S)
    #   양압: (0.9−0.2·SDS)D + 1.0E
    # 적설이 있으면 하향조합에만 동반하중 +0.2S (KDS 41 10 00). 양압엔 미포함.
    # sds=0.0(기본)이면 1.2/0.9로 환원 → 하위호환.
    ev = 0.2 * sds
    Dd = round(1.2 + ev, 4)   # 지진 하향조합 DL 계수
    Du = round(0.9 - ev, 4)   # 지진 양압조합 DL 계수
    dd_tag = ("%g" % Dd) + "DL"
    du_tag = ("%g" % Du) + "DL"
    snow_seis = {"S": 0.2} if "S" in has else {}
    snow_tag = "+0.2S" if "S" in has else ""
    for eq in ("EQX", "EQY"):
        if eq not in has:
            continue
        if "LL" in has:
            combos[f"{dd_tag}+1.0LL+1.0{eq}{snow_tag}"] = {"DL": Dd, "LL": 1.0, eq: 1.0, **snow_seis}
            combos[f"{dd_tag}+1.0LL-1.0{eq}{snow_tag}"] = {"DL": Dd, "LL": 1.0, eq: -1.0, **snow_seis}
            combos[f"{du_tag}+1.0{eq}"] = {"DL": Du, eq: 1.0}
            combos[f"{du_tag}-1.0{eq}"] = {"DL": Du, eq: -1.0}
        else:
            combos[f"{dd_tag}+1.0{eq}{snow_tag}"] = {"DL": Dd, eq: 1.0, **snow_seis}
            combos[f"{dd_tag}-1.0{eq}{snow_tag}"] = {"DL": Dd, eq: -1.0, **snow_seis}
            combos[f"{du_tag}+1.0{eq}"] = {"DL": Du, eq: 1.0}
            combos[f"{du_tag}-1.0{eq}"] = {"DL": Du, eq: -1.0}

    # 직교방향 지진 100%+30% (KDS 41 17 00 §8.1.3 — 내진설계범주 C/D 의무).
    # 한 방향 100% + 직교방향 30%를 모든 부호조합(±100%, ±30%)으로 생성한다.
    # 모서리 기둥의 양축 동시 수요(상관비·인장)를 포착 → 단방향 검토 누락 보정.
    # EQX·EQY 둘 다 있고 orthogonal_seismic(SDC C/D)일 때만 — 정상건물(SDC A/B) 무영향.
    if orthogonal_seismic and "EQX" in has and "EQY" in has:
        for primary, orth in (("EQX", "EQY"), ("EQY", "EQX")):
            for sp in (1.0, -1.0):       # 주방향(100%) 부호
                for so in (0.3, -0.3):   # 직교방향(30%) 부호
                    psign = "+" if sp > 0 else "-"
                    osign = "+" if so > 0 else "-"
                    eq_part = f"{psign}1.0{primary}{osign}0.3{orth}"
                    if "LL" in has:
                        combos[f"{dd_tag}+1.0LL{eq_part}{snow_tag}"] = {
                            "DL": Dd, "LL": 1.0, primary: sp, orth: so, **snow_seis}
                    else:
                        combos[f"{dd_tag}{eq_part}{snow_tag}"] = {
                            "DL": Dd, primary: sp, orth: so, **snow_seis}
                    combos[f"{du_tag}{eq_part}"] = {"DL": Du, primary: sp, orth: so}

    # 풍하중 조합 — KDS 41 12 00 §1.7 식1.7-4: 1.2D + 1.0W + 1.0L + 0.5(Lr | S | R).
    # 적설이 있으면 1.2DL 풍조합(1.0W)에만 동반하중 +0.5S 포함(지붕적설). 양압
    # 0.9DL±1.0W(식1.7-6)는 동반 중력하중 미포함. (지진 하향조합의 +0.2S와 동일 패턴.)
    wsnow = {"S": 0.5} if "S" in has else {}
    wsnow_tag = "+0.5S" if "S" in has else ""
    for w in ("WX", "WY"):
        if w not in has:
            continue
        if "LL" in has:
            combos[f"1.2DL+1.0LL+1.0{w}{wsnow_tag}"] = {"DL": 1.2, "LL": 1.0, w: 1.0, **wsnow}
            combos[f"1.2DL+1.0LL-1.0{w}{wsnow_tag}"] = {"DL": 1.2, "LL": 1.0, w: -1.0, **wsnow}
            combos[f"0.9DL+1.0{w}"] = {"DL": 0.9, w: 1.0}
            combos[f"0.9DL-1.0{w}"] = {"DL": 0.9, w: -1.0}
        else:
            combos[f"1.2DL+1.0{w}{wsnow_tag}"] = {"DL": 1.2, w: 1.0, **wsnow}
            combos[f"1.2DL-1.0{w}{wsnow_tag}"] = {"DL": 1.2, w: -1.0, **wsnow}
            combos[f"0.9DL+1.0{w}"] = {"DL": 0.9, w: 1.0}
            combos[f"0.9DL-1.0{w}"] = {"DL": 0.9, w: -1.0}

    return combos


# ============================================================
# 5. 통합 하중 생성
# ============================================================

def generate_all_loads(model: BuildingModel) -> dict:
    """BuildingModel에서 전체 하중을 자동 생성.

    Returns:
        {
            "load_cases": {"DL": [...], "LL": [...], "EQX": [...], ...},
            "load_combinations": {"1.2DL+1.6LL": {...}, ...},
            "reports": {
                "gravity": {...},
                "seismic": {...},
                "wind": {...},
            },
            "summary": {...},
        }
    """
    all_load_cases = {}
    reports = {}

    # 1. 중력하중 (활하중 면적저감은 model.live_load_reduction=True일 때만 opt-in)
    gravity = generate_gravity_loads(
        model, apply_live_reduction=getattr(model, "live_load_reduction", False))
    all_load_cases.update(gravity["load_cases"])
    reports["gravity"] = gravity["report"]

    # 2. 지진하중
    if model.region:
        story_weights = _calculate_story_weights(model, gravity["load_cases"]["DL"])
        seismic = generate_seismic_loads(model, story_weights)
        all_load_cases.update(seismic["load_cases"])
        reports["seismic"] = seismic["report"]

    # 3. 풍하중
    if model.region:
        wind = generate_wind_loads(model)
        all_load_cases.update(wind["load_cases"])
        reports["wind"] = wind["report"]

    # 3b. 적설하중 (지붕)
    if model.region:
        snow = generate_snow_loads(model)
        if snow["load_cases"]:
            all_load_cases.update(snow["load_cases"])
        reports["snow"] = snow["report"]

    # 4. 하중조합 (수직지진 Ev 반영: seismic report의 SDS 전달)
    combinations = {}
    if model.auto_combinations:
        sds = 0.0
        sd1 = 0.0
        seis_rpt = reports.get("seismic")
        if isinstance(seis_rpt, dict):
            sds = seis_rpt.get("SDS", 0.0) or 0.0
            sd1 = seis_rpt.get("SD1", 0.0) or 0.0
        # 직교지진 100/30 의무 여부: 내진설계범주 C/D (KDS 41 17 00 §8.1.3).
        orthogonal = False
        if sds and sd1:
            try:
                from core.design_check import seismic_design_category
                sdc = seismic_design_category(sds, sd1, model.importance)
                orthogonal = sdc in ("C", "D")
            except Exception:
                orthogonal = False
        combinations = generate_load_combinations(
            list(all_load_cases.keys()), sds=sds, orthogonal_seismic=orthogonal)

    # 5. 요약
    summary = {
        "load_cases": list(all_load_cases.keys()),
        "num_load_cases": len(all_load_cases),
        "num_combinations": len(combinations),
        "total_weight_kN": round(sum(
            _calculate_story_weights(model, all_load_cases.get("DL", []))
        ), 2) if "DL" in all_load_cases else 0,
    }

    return {
        "load_cases": all_load_cases,
        "load_combinations": combinations,
        "reports": reports,
        "summary": summary,
    }
