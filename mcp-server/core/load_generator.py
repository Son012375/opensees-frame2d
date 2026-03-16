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


def generate_gravity_loads(model: BuildingModel) -> dict[str, list[dict]]:
    """BuildingModel → DL/LL 하중케이스 생성.

    DL = 슬래브 자중 + 마감재 + 설비
    LL = 용도별 DB 조회

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

        # LL: 용도별 활하중
        ll_value = _query_live_load(story_info.usage)
        ll_loads.append({
            "type": "floor_area",
            "story": s,
            "value": round(ll_value, 2),
        })

        report.append({
            "story": s,
            "usage": story_info.usage,
            "DL_kNm2": round(dl_total, 2),
            "DL_breakdown": {
                "slab_self": round(slab_self, 2),
                "finish": finish,
                "mep": mep,
            },
            "LL_kNm2": round(ll_value, 2),
        })

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


def _get_period_upper_limit(SD1: float) -> float:
    """주기상한계수 Cu 조회."""
    result = query_design_loads("seismic_design", "period_upper_limit_Cu")
    if result["status"] != "success":
        return 1.4

    Cu_map = {}
    for rec in result["records"]:
        pk = rec["primary_key"]
        val = rec.get("value", 1.4)
        Cu_map[pk] = val

    # SD1 기반 Cu 선정
    if SD1 <= 0.1:
        return Cu_map.get("Sd1_le_0.1", 1.7)
    elif SD1 <= 0.15:
        return Cu_map.get("Sd1_0.15", 1.6)
    elif SD1 <= 0.2:
        return Cu_map.get("Sd1_0.2", 1.5)
    elif SD1 <= 0.3:
        return Cu_map.get("Sd1_0.3", 1.4)
    else:
        return Cu_map.get("Sd1_ge_0.4", 1.4)


def _calculate_story_weights(model: BuildingModel, dl_loads: list[dict]) -> list[float]:
    """층별 유효 중량 계산 (kN).

    W = DL × floor_area (활하중은 창고 25%만 포함, 일반 건물은 미포함)
    """
    floor_area = model.floor_area
    weights = []
    for ld in dl_loads:
        w = ld["value"] * floor_area  # kN
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
    IE = model.importance_factor
    z_eff = z * IE
    params = _compute_site_spectrum_params(z_eff, Fa, Fv)
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
        "z_effective": z_eff,
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
        "W_kN": round(W, 2),
        "V_kN": round(V, 2),
        "story_forces": story_forces,
    }

    return {"load_cases": load_cases, "report": report}


# ============================================================
# 3. 풍하중 (KDS 41 12 00 §5)
# ============================================================

# 노풍도별 Kz 계수 (표 5.2-4, 간략화)
# Kz = 2.01 × (z/zg)^(2/α)
EXPOSURE_PARAMS = {
    "A": {"zg": 250, "alpha": 7.0},   # 대도시 중심
    "B": {"zg": 365, "alpha": 9.5},   # 도시/교외
    "C": {"zg": 275, "alpha": 7.0},   # 개활지
    "D": {"zg": 215, "alpha": 5.0},   # 해안
}


def _velocity_pressure_coeff(z_height: float, exposure: str) -> float:
    """속도압 높이분포계수 Kz 계산 (KDS 41 12 00 표 5.2-4)."""
    params = EXPOSURE_PARAMS.get(exposure, EXPOSURE_PARAMS["B"])
    zg = params["zg"]
    alpha = params["alpha"]

    z_min = 5.0  # 최소 높이 (m)
    z_eff = max(z_height, z_min)
    z_eff = min(z_eff, zg)

    Kz = 2.01 * (z_eff / zg) ** (2.0 / alpha)
    return Kz


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

    V0 = hazard["regions"][0].get("wind_v0", 26.0)  # m/s

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

        # 속도압 qz (kN/m²)
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

    return {
        "load_cases": {"WX": wx_loads, "WY": wy_loads},
        "report": report,
    }


# ============================================================
# 4. 하중조합 (KDS 41 12 00 §1.7)
# ============================================================

def generate_load_combinations(
    available_cases: list[str],
) -> dict[str, dict[str, float]]:
    """KDS 41 12 00 §1.7 주요 하중조합 생성.

    사용 가능한 하중케이스에 따라 적절한 조합만 생성한다.

    Args:
        available_cases: 생성된 하중케이스 이름 목록 (예: ["DL", "LL", "EQX", "EQY", "WX", "WY"])

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

    # 지진 조합
    for eq in ("EQX", "EQY"):
        if eq not in has:
            continue
        if "LL" in has:
            combos[f"1.2DL+1.0LL+1.0{eq}"] = {"DL": 1.2, "LL": 1.0, eq: 1.0}
            combos[f"1.2DL+1.0LL-1.0{eq}"] = {"DL": 1.2, "LL": 1.0, eq: -1.0}
            combos[f"0.9DL+1.0{eq}"] = {"DL": 0.9, eq: 1.0}
            combos[f"0.9DL-1.0{eq}"] = {"DL": 0.9, eq: -1.0}
        else:
            combos[f"1.2DL+1.0{eq}"] = {"DL": 1.2, eq: 1.0}
            combos[f"1.2DL-1.0{eq}"] = {"DL": 1.2, eq: -1.0}
            combos[f"0.9DL+1.0{eq}"] = {"DL": 0.9, eq: 1.0}
            combos[f"0.9DL-1.0{eq}"] = {"DL": 0.9, eq: -1.0}

    # 풍하중 조합
    for w in ("WX", "WY"):
        if w not in has:
            continue
        if "LL" in has:
            combos[f"1.2DL+1.0LL+1.0{w}"] = {"DL": 1.2, "LL": 1.0, w: 1.0}
            combos[f"1.2DL+1.0LL-1.0{w}"] = {"DL": 1.2, "LL": 1.0, w: -1.0}
            combos[f"0.9DL+1.0{w}"] = {"DL": 0.9, w: 1.0}
            combos[f"0.9DL-1.0{w}"] = {"DL": 0.9, w: -1.0}
        else:
            combos[f"1.2DL+1.0{w}"] = {"DL": 1.2, w: 1.0}
            combos[f"1.2DL-1.0{w}"] = {"DL": 1.2, w: -1.0}
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

    # 1. 중력하중
    gravity = generate_gravity_loads(model)
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

    # 4. 하중조합
    combinations = {}
    if model.auto_combinations:
        combinations = generate_load_combinations(list(all_load_cases.keys()))

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
