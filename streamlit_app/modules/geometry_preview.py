"""해석 없이 3D 모델 미리보기 생성.

_generate_frame_3d_geometry()로 노드/연결만 계산하고,
Frame3DMultiCaseResult의 기하 필드만 채운 mock을 생성하여
기존 create_3d_model_figure()를 재사용한다.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

MCP_SERVER_PATH = str(Path(__file__).resolve().parent.parent.parent / "mcp-server")


def _ensure_path():
    if MCP_SERVER_PATH not in sys.path:
        sys.path.insert(0, MCP_SERVER_PATH)


def build_geometry_preview(
    stories: list[float],
    bays_x: list[float],
    bays_y: list[float],
    column_section: str = "H-300x300",
    beam_x_section: str = "H-400x200",
    beam_y_section: str = "H-400x200",
):
    """해석 없이 3D 미리보기용 mock Frame3DMultiCaseResult 생성.

    Args:
        stories: 층고 리스트 (m)
        bays_x: X방향 경간 폭 리스트 (m)
        bays_y: Y방향 경간 폭 리스트 (m)
        column_section: 기둥 단면명
        beam_x_section: X보 단면명
        beam_y_section: Y보 단면명

    Returns:
        Frame3DMultiCaseResult (기하 필드만 채워짐)
    """
    _ensure_path()
    from core.frame_3d import _generate_frame_3d_geometry, Frame3DMultiCaseResult

    nodes, connections, node_grid, base_nodes = _generate_frame_3d_geometry(
        stories, bays_x, bays_y
    )

    mock = Frame3DMultiCaseResult()
    mock.num_stories = len(stories)
    mock.num_bays_x = len(bays_x)
    mock.num_bays_y = len(bays_y)
    mock.stories = list(stories)
    mock.bays_x = list(bays_x)
    mock.bays_y = list(bays_y)
    mock.total_height = sum(stories)
    mock.total_width_x = sum(bays_x)
    mock.total_width_y = sum(bays_y)
    mock.column_section = column_section
    mock.beam_x_section = beam_x_section
    mock.beam_y_section = beam_y_section

    # 노드를 dict 형태로 (create_3d_model_figure가 기대하는 형식)
    mock.nodes = [
        {"id": n.id, "x_m": n.x, "y_m": n.y, "z_m": n.z}
        for n in nodes
    ]

    # member_info 생성 (부재별 ni, nj, type, section)
    section_map = {
        "column": column_section,
        "beam_x": beam_x_section,
        "beam_y": beam_y_section,
    }

    member_info = []
    for member_id, (ni, nj, etype) in enumerate(connections, start=1):
        ni_node = next(n for n in nodes if n.id == ni)
        nj_node = next(n for n in nodes if n.id == nj)
        length = math.sqrt(
            (nj_node.x - ni_node.x) ** 2
            + (nj_node.y - ni_node.y) ** 2
            + (nj_node.z - ni_node.z) ** 2
        )
        member_info.append({
            "member_id": member_id,
            "type": etype,
            "ni": ni,
            "nj": nj,
            "length_m": round(length, 4),
            "section": section_map[etype],
            "element_ids": [],
        })

    mock.member_info = member_info
    return mock
