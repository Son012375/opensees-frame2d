"""해석 엔진 래퍼 모듈.

mcp-server/core/ 모듈을 임포트하여 Streamlit에서 사용 가능하게 한다.
기존 webapp과 동일한 sys.path 주입 패턴 사용.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

# mcp-server 경로 주입
MCP_SERVER_PATH = str(Path(__file__).resolve().parent.parent.parent / "mcp-server")


def _ensure_path():
    """mcp-server를 sys.path에 추가."""
    if MCP_SERVER_PATH not in sys.path:
        sys.path.insert(0, MCP_SERVER_PATH)


def run_building_analysis(config: dict) -> tuple:
    """Building Pipeline 전체 실행.

    Args:
        config: BuildingModel.from_json() 스키마에 맞는 dict

    Returns:
        (BuildingModel, load_result_dict, Frame3DMultiCaseResult)
    """
    _ensure_path()
    from core.building_model import BuildingModel
    from core.load_generator import generate_all_loads
    from core.frame_3d import analyze_frame_3d_multi

    model = BuildingModel.from_json(config)
    load_result = generate_all_loads(model)

    kwargs = model.to_frame3d_kwargs()
    kwargs["load_cases"] = load_result["load_cases"]
    kwargs["load_combinations"] = load_result["load_combinations"]

    multi = analyze_frame_3d_multi(**kwargs)
    return model, load_result, multi


def run_frame3d_direct(
    stories: list[float],
    bays_x: list[float],
    bays_y: list[float],
    load_cases: dict,
    load_combinations: Optional[dict] = None,
    **kwargs,
):
    """frame_3d 직접 해석 (수동 하중케이스).

    Returns:
        Frame3DMultiCaseResult
    """
    _ensure_path()
    from core.frame_3d import analyze_frame_3d_multi

    return analyze_frame_3d_multi(
        stories=stories,
        bays_x=bays_x,
        bays_y=bays_y,
        load_cases=load_cases,
        load_combinations=load_combinations or {},
        **kwargs,
    )


def get_usage_keys() -> list[str]:
    """사용 가능한 용도 키 목록 반환."""
    _ensure_path()
    from core.load_generator import USAGE_TO_LIVE_LOAD_KEY
    return list(USAGE_TO_LIVE_LOAD_KEY.keys())


def get_seismic_system_keys() -> list[str]:
    """사용 가능한 내진시스템 키 목록 반환."""
    _ensure_path()
    from core.load_generator import SEISMIC_SYSTEM_MAP
    return list(SEISMIC_SYSTEM_MAP.keys())


def merge_geometry_and_usage(geometry_config: dict, usage_config: dict) -> dict:
    """Phase 1 geometry + Phase 2 usage → 완전한 analysis config.

    Args:
        geometry_config: {"stories": [{"height", "slab_thickness"}], "bays_x", "bays_y"}
        usage_config: {"stories_usage": [{"usage", "dead_load_finish"}],
                       "column_section", "region", ...}

    Returns:
        run_building_analysis()에 전달할 완전한 config dict
    """
    geo_stories = geometry_config.get("stories", [])
    usage_stories = usage_config.get("stories_usage", [])

    stories = []
    for i, geo_s in enumerate(geo_stories):
        merged = dict(geo_s)
        if i < len(usage_stories):
            merged["usage"] = usage_stories[i].get("usage", "office")
            merged["dead_load_finish"] = usage_stories[i].get("dead_load_finish", 1.0)
        else:
            merged["usage"] = "office"
            merged["dead_load_finish"] = 1.0
        stories.append(merged)

    # 단면/재료: usage_config → geometry_config(IFC 감지) → 기본값
    col_sec = (usage_config.get("column_section")
               or geometry_config.get("column_section")
               or "H-300x300")
    beam_sec = (usage_config.get("beam_x_section")
                or geometry_config.get("beam_section")
                or "H-400x200")

    return {
        "stories": stories,
        "bays_x": geometry_config["bays_x"],
        "bays_y": geometry_config["bays_y"],
        "column_section": col_sec,
        "beam_x_section": usage_config.get("beam_x_section", beam_sec),
        "beam_y_section": usage_config.get("beam_y_section", beam_sec),
        "material_name": (usage_config.get("material_name")
                          or geometry_config.get("material_name")
                          or "SS275"),
        "region": usage_config.get("region", ""),
        "site_class": usage_config.get("site_class", "S3"),
        "importance": usage_config.get("importance", "II"),
        "seismic_system": usage_config.get("seismic_system", "ordinary_moment_frame"),
        "supports": usage_config.get("supports", "fixed"),
        "auto_combinations": usage_config.get("auto_combinations", True),
    }


def get_section_info(section_name: str) -> dict:
    """단면 물성 정보 조회."""
    _ensure_path()
    from core.section_3d import get_section_3d
    sec = get_section_3d(section_name)
    return {
        "name": sec.name,
        "A_mm2": sec.A,
        "Ix_mm4": sec.Ix,
        "Iy_mm4": sec.Iy,
        "J_mm4": sec.J,
        "h_mm": sec.h,
        "b_mm": sec.b,
    }
