"""
KDS 설계하중 조회 모듈
Supabase load_params / hazard_region_values 테이블에서 KDS 하중 데이터를 조회한다.

Tables:
  - load_params: 설계하중 파라미터 (고정/활/설/풍하중, 하중조합 등) 454건+
  - hazard_region_values: 지역별 위험계수 (Sg, V₀) 458건
"""
from __future__ import annotations

from typing import Optional
from supabase import create_client, Client

# Supabase 설정 (simple_beam.py와 동일, openseespy 의존성 없이 독립 연결)
SUPABASE_URL = "https://kuqoajdqvfsobhhlnyyg.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imt1cW9hamRxdmZzb2JoaGxueXlnIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjkzNjA1MjksImV4cCI6MjA4NDkzNjUyOX0.5-rSgPXFrMi4xxdrmxgqZdQnl1U2ta-qDYd2IpLMHzM"

_supabase_client: Optional[Client] = None


def _get_supabase() -> Client:
    """Supabase 클라이언트 (싱글톤)."""
    global _supabase_client
    if _supabase_client is None:
        from core.kds_cache import wrap as _kds_wrap
        # Wrapped so a paused/unreachable database falls back to the local
        # snapshot instead of raising. Transparent while the DB answers.
        _supabase_client = _kds_wrap(create_client(SUPABASE_URL, SUPABASE_KEY))
    return _supabase_client


def query_design_loads(
    param_type: str,
    param_subtype: Optional[str] = None,
    keyword: Optional[str] = None,
) -> dict:
    """설계하중 파라미터 조회.

    Args:
        param_type: 하중 유형 (dead_load, live_load, snow_load, wind_load 등)
        param_subtype: 세부 유형 (unit_weight, distributed, base_coefficient 등)
        keyword: primary_key 부분 매칭 검색어 (예: "office", "steel")

    Returns:
        매칭 레코드 목록 또는 에러
    """
    try:
        supabase = _get_supabase()
        query = supabase.table("load_params").select("*").eq("param_type", param_type)

        if param_subtype:
            query = query.eq("param_subtype", param_subtype)

        if keyword:
            query = query.ilike("primary_key", f"%{keyword}%")

        result = query.order("primary_key").execute()

        if not result.data:
            return {
                "status": "not_found",
                "message": f"No records for param_type='{param_type}'"
                + (f", param_subtype='{param_subtype}'" if param_subtype else "")
                + (f", keyword='{keyword}'" if keyword else ""),
                "hint": "Available param_types: dead_load, live_load, snow_load, wind_load, load_combo, live_reduction, roof_live_reduction, similar_live_load",
            }

        records = []
        for row in result.data:
            rec = {
                "primary_key": row["primary_key"],
                "secondary_key": row.get("secondary_key"),
                "display_name_ko": row.get("display_name_ko"),
                "display_name_en": row.get("display_name_en"),
                "param_subtype": row.get("param_subtype"),
                "unit": row.get("unit"),
                "code_id": row.get("code_id"),
                "clause_id": row.get("clause_id"),
                "table_id": row.get("table_id"),
            }
            if row.get("value") is not None:
                rec["value"] = row["value"]
            if row.get("value_min") is not None:
                rec["value_min"] = row["value_min"]
            if row.get("value_max") is not None:
                rec["value_max"] = row["value_max"]
            if row.get("conditions"):
                rec["conditions"] = row["conditions"]
            if row.get("notes"):
                rec["notes"] = row["notes"]
            records.append(rec)

        return {
            "status": "success",
            "param_type": param_type,
            "count": len(records),
            "records": records,
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


def query_load_combinations(
    limit_state: Optional[str] = None,
) -> dict:
    """하중조합 조회.

    Args:
        limit_state: "uls" 또는 "sls". None이면 전체 반환.

    Returns:
        하중조합 목록
    """
    try:
        supabase = _get_supabase()
        query = supabase.table("load_params").select("*").eq("param_type", "load_combo")

        if limit_state:
            query = query.eq("param_subtype", limit_state.lower())

        result = query.order("primary_key").execute()

        if not result.data:
            return {
                "status": "not_found",
                "message": f"No load combinations found"
                + (f" for limit_state='{limit_state}'" if limit_state else ""),
            }

        combos = []
        for row in result.data:
            combos.append({
                "combo_id": row["primary_key"],
                "limit_state": row.get("param_subtype", "").upper(),
                "display_name_ko": row.get("display_name_ko"),
                "display_name_en": row.get("display_name_en"),
                "conditions": row.get("conditions"),
                "notes": row.get("notes"),
                "code_id": row.get("code_id"),
                "clause_id": row.get("clause_id"),
            })

        return {
            "status": "success",
            "count": len(combos),
            "combinations": combos,
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


def query_hazard_values(
    region_name: str,
    hazard_type: Optional[str] = None,
) -> dict:
    """지역 위험계수 조회 (지상적설하중 Sg, 기본풍속 V₀).

    Args:
        region_name: 지역명 (시/군/구). 부분 매칭 지원 (예: "서울", "강남구")
        hazard_type: "snow_sg" 또는 "wind_v0". None이면 둘 다 반환.

    Returns:
        매칭 지역의 Sg, V₀ 값
    """
    try:
        supabase = _get_supabase()

        query = supabase.table("hazard_region_values").select("*")

        # 지역명 부분 매칭: region_sido 또는 region_sigungu에서 검색
        # Supabase에서 OR 조건은 .or_() 사용
        query = query.or_(
            f"region_sido.ilike.%{region_name}%,region_sigungu.ilike.%{region_name}%"
        )

        if hazard_type:
            query = query.eq("hazard_type", hazard_type)

        result = query.order("region_sido").order("region_sigungu").execute()

        if not result.data:
            return {
                "status": "not_found",
                "message": f"No hazard data found for region '{region_name}'",
                "hint": "지역명을 시/도 또는 시/군/구 단위로 입력하세요 (예: '서울', '강남구', '부산')",
            }

        # 지역별로 그룹핑
        regions = {}
        for row in result.data:
            key = f"{row['region_sido']} {row['region_sigungu']}"
            if key not in regions:
                regions[key] = {
                    "region_sido": row["region_sido"],
                    "region_sigungu": row["region_sigungu"],
                }
            if row["hazard_type"] == "snow_sg":
                regions[key]["snow_sg"] = row["value"]
                regions[key]["snow_sg_unit"] = row.get("unit", "kN/m²")
            elif row["hazard_type"] == "wind_v0":
                regions[key]["wind_v0"] = row["value"]
                regions[key]["wind_v0_unit"] = row.get("unit", "m/s")

        records = list(regions.values())

        return {
            "status": "success",
            "query": region_name,
            "count": len(records),
            "regions": records,
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}
