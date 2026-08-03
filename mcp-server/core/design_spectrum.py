"""KDS 17 10 00 설계응답스펙트럼 계산 모듈.

KDS 17 10 00 §4.2.1.4에 따른 가속도표준설계응답스펙트럼을 생성한다.

암반지반(S1) 표준스펙트럼:
  - 표 4.2-5: α₄=2.8, T₀=0.06s, Ts=0.3s, TL=3.0s
  - 표 4.2-6: Sa(T) 수식 (4개 구간)

토사지반(S2~S5) 스펙트럼:
  - 그림 4.2-2: Fa, Fv로 보정
  - S = z × Fa (유효수평지반가속도)
  - 전이주기: T₀ = 0.2×Fv×S / (Fa×S) = 0.2×Fv/Fa×T₀_rock
  - Ts = Fv×S / (2.5×Fa×S) 등

Supabase DB에서 Fa, Fv, 구역계수(z)를 조회하여 계산한다.
"""
from __future__ import annotations

import math
from typing import Optional

from supabase import create_client, Client

# Supabase 설정 (kds_loads.py와 동일)
SUPABASE_URL = "https://kuqoajdqvfsobhhlnyyg.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imt1cW9hamRxdmZzb2JoaGxueXlnIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjkzNjA1MjksImV4cCI6MjA4NDkzNjUyOX0.5-rSgPXFrMi4xxdrmxgqZdQnl1U2ta-qDYd2IpLMHzM"

_supabase_client: Optional[Client] = None


def _get_supabase() -> Client:
    global _supabase_client
    if _supabase_client is None:
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _supabase_client


# ── 암반지반(S1) 표준 파라미터 (표 4.2-5) ──
ROCK_PARAMS = {
    "alpha_4": 2.8,   # 단주기 스펙트럼 증폭계수
    "T0": 0.06,       # 전이주기 T₀ (sec)
    "Ts": 0.3,        # 전이주기 Ts (sec)
    "TL": 3.0,        # 전이주기 TL (sec)
}


def _get_zone_coefficient(region_name: str) -> tuple[float, str]:
    """지역명으로 구역계수(z) 조회.

    Returns:
        (z, zone_description)
    """
    supabase = _get_supabase()
    query = supabase.table("hazard_region_values").select("*").eq(
        "hazard_type", "seismic_zone"
    ).or_(
        f"region_sido.ilike.%{region_name}%,"
        f"region_sigungu.ilike.%{region_name}%"
    )
    result = query.execute()

    if not result.data:
        raise ValueError(f"지역 '{region_name}'을 찾을 수 없습니다.")

    row = result.data[0]
    z = row["value"]
    notes = row.get("notes", "")
    desc = f"{row['region_sido']} {row['region_sigungu']} (z={z}g, {notes})"
    return z, desc


# DB secondary_key → 표 4.2-8 anchor point (유효수평지반가속도 S)
_Z_ANCHORS: tuple[tuple[str, float], ...] = (
    ("z_le_0.1", 0.1),
    ("z_0.2", 0.2),
    ("z_0.3", 0.3),
)


def _linear_interp_z(z: float, anchors: list[tuple[float, float]]) -> float:
    """KDS 17 10 00 §4.2.1 ②: 표 4.2-8 값을 z 기준 직선보간.

    KDS 원문 ("S의 값이 중간 값에 해당할 경우 직선보간하여 결정한다") 준수.
    anchors: [(z_value, coefficient), ...] (z 오름차순).
    범위 밖(z < 0.1 또는 z > 0.3)은 표 경계값으로 clamp (KDS 외삽 미규정).
    """
    anchors = sorted(anchors, key=lambda t: t[0])
    if z <= anchors[0][0]:
        return anchors[0][1]
    if z >= anchors[-1][0]:
        return anchors[-1][1]
    for (z_lo, v_lo), (z_hi, v_hi) in zip(anchors[:-1], anchors[1:]):
        if z_lo <= z <= z_hi:
            t = (z - z_lo) / (z_hi - z_lo)
            return v_lo + t * (v_hi - v_lo)
    return anchors[-1][1]  # unreachable, defensive


def _get_site_coefficients(site_class: str, z: float) -> tuple[float, float]:
    """지반종류와 구역계수로 Fa, Fv 조회 (KDS 17 10 00 표 4.2-8 직선보간).

    KDS 17 10 00 §4.2.1 ②는 단주기/장주기 지반증폭계수를 표 4.2-8 anchor
    (S ≤ 0.1, S = 0.2, S = 0.3) 사이에서 직선보간하도록 명시한다. 이전
    구현은 z를 버킷화하여 anchor 값을 그대로 사용했으나, 본 함수는 세 anchor
    행을 DB에서 모두 조회한 뒤 실제 z로 보간한다.

    Args:
        site_class: "S1"~"S5"
        z: 구역계수 (예: 서울 0.11, 부산 0.07)

    Returns:
        (Fa, Fv) — z에서 직선보간된 값.
    """
    site_class = site_class.upper()
    if site_class not in ("S1", "S2", "S3", "S4", "S5"):
        raise ValueError(f"지반종류 '{site_class}'은 S1~S5 중 하나여야 합니다.")

    supabase = _get_supabase()

    fa_anchors: list[tuple[float, float]] = []
    fv_anchors: list[tuple[float, float]] = []

    for sec_key, z_anchor in _Z_ANCHORS:
        fa_row = supabase.table("load_params").select("value").eq(
            "param_subtype", "site_coefficient_Fa"
        ).eq("primary_key", site_class).eq("secondary_key", sec_key).execute()
        if not fa_row.data:
            raise ValueError(f"Fa anchor 조회 실패: {site_class}, {sec_key}")
        fa_anchors.append((z_anchor, fa_row.data[0]["value"]))

        fv_row = supabase.table("load_params").select("value").eq(
            "param_subtype", "site_coefficient_Fv"
        ).eq("primary_key", site_class).eq("secondary_key", sec_key).execute()
        if not fv_row.data:
            raise ValueError(f"Fv anchor 조회 실패: {site_class}, {sec_key}")
        fv_anchors.append((z_anchor, fv_row.data[0]["value"]))

    Fa = _linear_interp_z(z, fa_anchors)
    Fv = _linear_interp_z(z, fv_anchors)
    return Fa, Fv


def _compute_site_spectrum_params(z: float, Fa: float, Fv: float) -> dict:
    """토사지반 스펙트럼 파라미터 계산 (KDS 17 10 00 그림 4.2-2).

    암반지반 S1일 때는 Fa=1.12, Fv=0.84로 표준값과 동일해짐.

    Args:
        z: 구역계수
        Fa: 단주기 지반증폭계수
        Fv: 1초주기 지반증폭계수

    Returns:
        S, SDS, SD1, T0, Ts, TL 등 스펙트럼 정의 파라미터
    """
    S = z * Fa                    # 유효수평지반가속도 (단주기)
    S1 = z * Fv                   # 유효수평지반가속도 (1초주기) -- KDS 41 17 00 기준

    # 설계스펙트럼가속도
    SDS = 2.5 * S                 # 단주기 설계스펙트럼가속도 = α₄/α₄_rock × Fa × S ≈ 2.5S
    SD1 = S1                      # 1초주기 설계스펙트럼가속도

    # 전이주기 (토사지반)
    # T₀ = 0.2 × SD1 / SDS
    T0 = 0.2 * SD1 / SDS if SDS > 0 else ROCK_PARAMS["T0"]
    # Ts = SD1 / SDS
    Ts = SD1 / SDS if SDS > 0 else ROCK_PARAMS["Ts"]
    # TL = 고정값 3.0초 (KDS 17 10 00)
    TL = ROCK_PARAMS["TL"]

    return {
        "S": round(S, 4),
        "S1": round(S1, 4),
        "SDS": round(SDS, 4),
        "SD1": round(SD1, 4),
        "T0": round(T0, 4),
        "Ts": round(Ts, 4),
        "TL": TL,
    }


def _spectral_acceleration(T: float, S: float, SDS: float, SD1: float,
                           T0: float, Ts: float, TL: float) -> float:
    """주기 T에서의 설계스펙트럼가속도 Sa(T) 계산.

    KDS 17 10 00 표 4.2-6:
      구간 1 (0 ≤ T ≤ T₀):    Sa = S + (SDS - S) × (T / T₀)
      구간 2 (T₀ ≤ T ≤ Ts):   Sa = SDS
      구간 3 (Ts ≤ T ≤ TL):   Sa = SD1 / T
      구간 4 (TL ≤ T):         Sa = SD1 × TL / T²
    """
    if T <= 0:
        return S
    elif T <= T0:
        return S + (SDS - S) * (T / T0)
    elif T <= Ts:
        return SDS
    elif T <= TL:
        return SD1 / T
    else:
        return SD1 * TL / (T * T)


def compute_design_spectrum(
    region: str,
    site_class: str = "S3",
    importance_factor: float = 1.0,
    damping_ratio: float = 0.05,
    period_start: float = 0.0,
    period_end: float = 5.0,
    period_step: float = 0.01,
    return_period: int = 500,
) -> dict:
    """KDS 17 10 00 설계응답스펙트럼 계산.

    Args:
        region: 지역명 (예: "서울", "강남구", "부산")
        site_class: 지반종류 "S1"~"S5"
        importance_factor: 위험도계수 I (기본 1.0)
        damping_ratio: 감쇠비 (기본 5% = 0.05)
        period_start: 시작 주기 (sec)
        period_end: 끝 주기 (sec)
        period_step: 주기 간격 (sec)
        return_period: 재현주기 (년), PGA 조회용

    Returns:
        스펙트럼 데이터 (periods, Sa, parameters, metadata)
    """
    try:
        # 1. 구역계수 조회
        z, region_desc = _get_zone_coefficient(region)

        # 위험도계수 적용
        z_eff = z * importance_factor

        # 2. 지반계수 조회
        Fa, Fv = _get_site_coefficients(site_class, z)

        # 3. 스펙트럼 파라미터 계산
        params = _compute_site_spectrum_params(z_eff, Fa, Fv)

        # 4. 감쇠보정 (5% 이외)
        Cd = 1.0
        if abs(damping_ratio - 0.05) > 0.001:
            # KDS 17 10 00 표 4.2-7 감쇠보정계수
            # Cd = 1.0 for 5%, 근사식: Cd = (10 / (5 + 100ξ))^0.5
            xi_pct = damping_ratio * 100
            Cd = math.sqrt(10.0 / (5.0 + xi_pct))
            Cd = round(Cd, 4)

        # 5. 스펙트럼 곡선 생성
        periods = []
        sa_values = []

        # 주요 전이점 포함
        key_periods = {0.0, params["T0"], params["Ts"], params["TL"]}

        T = period_start
        while T <= period_end + period_step / 2:
            periods.append(round(T, 4))
            T += period_step

        # 전이점 삽입 (누락 시)
        for kp in key_periods:
            if period_start <= kp <= period_end and kp not in periods:
                periods.append(round(kp, 4))
        periods = sorted(set(periods))

        for T in periods:
            Sa = _spectral_acceleration(
                T, params["S"], params["SDS"], params["SD1"],
                params["T0"], params["Ts"], params["TL"]
            )
            Sa *= Cd  # 감쇠보정
            sa_values.append(round(Sa, 6))

        # 6. PGA 정보 조회 (가능하면)
        pga_info = None
        try:
            supabase = _get_supabase()
            rp_key = f"seismic_pga_{return_period:04d}"
            pga_result = supabase.table("hazard_region_values").select(
                "value,region_sido,region_sigungu"
            ).eq("hazard_type", rp_key).or_(
                f"region_sido.ilike.%{region}%,"
                f"region_sigungu.ilike.%{region}%"
            ).limit(1).execute()
            if pga_result.data:
                pga_row = pga_result.data[0]
                pga_info = {
                    "return_period_years": return_period,
                    "pga_pct_g": pga_row["value"],
                    "pga_g": round(pga_row["value"] / 100.0, 4),
                    "region": f"{pga_row['region_sido']} {pga_row['region_sigungu']}",
                }
        except Exception:
            pass

        # 7. 결과 조립
        return {
            "status": "success",
            "spectrum": {
                "periods": periods,
                "Sa": sa_values,
                "unit": "g",
                "count": len(periods),
            },
            "parameters": {
                "z": z,
                "z_effective": z_eff,
                "importance_factor": importance_factor,
                "site_class": site_class,
                "Fa": Fa,
                "Fv": Fv,
                "S": params["S"],
                "SDS": params["SDS"],
                "SD1": params["SD1"],
                "T0": params["T0"],
                "Ts": params["Ts"],
                "TL": params["TL"],
                "damping_ratio": damping_ratio,
                "damping_correction_Cd": Cd,
            },
            "pga": pga_info,
            "region": region_desc,
            "code_reference": {
                "code": "KDS 17 10 00",
                "clause": "§4.2.1.4",
                "tables": ["표 4.2-5", "표 4.2-6", "표 4.2-8"],
                "figures": ["그림 4.2-1 (암반)", "그림 4.2-2 (토사)"],
            },
            "opensees_input": {
                "description": "OpenSeesPy responseSpectrumAnalysis 입력용 데이터",
                "usage": (
                    "ops.timeSeries('Path', tsTag, '-time', *periods, '-values', *Sa)\n"
                    "ops.responseSpectrumAnalysis(tsTag, 'SRSS', '-dir', 1)"
                ),
                "periods": periods,
                "Sa": sa_values,
            },
        }

    except ValueError as e:
        return {"status": "error", "message": str(e)}
    except Exception as e:
        return {"status": "error", "message": f"스펙트럼 계산 오류: {str(e)}"}
