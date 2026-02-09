"""
2D 골조 해석 모듈
OpenSeesPy를 사용한 2D 프레임 정적 해석
- 멀티 하중케이스 / 하중조합 지원
- 부재력 다이어그램 (N/V/M) 생성
- 층별 분석 (변위, 층간변위, 층전단)
"""
from __future__ import annotations

import math
import openseespy.opensees as ops
from dataclasses import dataclass, field
from typing import Literal, Optional

from core.simple_beam import (
    get_section_from_db,
    get_material_from_db,
    DEFAULT_SECTIONS,
    DEFAULT_MATERIALS,
)


# ============================================================
# 데이터 구조
# ============================================================

@dataclass
class Node2D:
    """2D 노드 정보"""
    id: int
    x: float  # m
    y: float  # m (높이)


@dataclass
class Element2D:
    """2D 요소 정보"""
    id: int
    ni: int  # 시작 노드 ID
    nj: int  # 끝 노드 ID
    elem_type: str  # "column" or "beam"
    section_name: str


@dataclass
class Frame2DCaseResult:
    """단일 하중케이스/조합 결과"""
    nodal_displacements: list[dict] = field(default_factory=list)
    element_forces: list[dict] = field(default_factory=list)
    reactions: list[dict] = field(default_factory=list)
    story_drifts: list[dict] = field(default_factory=list)
    story_data: dict = field(default_factory=dict)  # 층별 상세 (변위, 전단)

    # 최대값
    max_displacement_x: float = 0.0
    max_displacement_y: float = 0.0
    max_displacement_x_node: int = 0
    max_displacement_y_node: int = 0
    max_drift: float = 0.0
    max_drift_story: int = 0
    max_moment: float = 0.0
    max_moment_element: int = 0
    max_axial: float = 0.0
    max_axial_element: int = 0
    max_shear: float = 0.0
    max_shear_element: int = 0


@dataclass
class Frame2DMultiCaseResult:
    """멀티케이스 2D 골조 해석 결과"""
    # 기하 정보 (모든 케이스 공유)
    num_stories: int = 0
    num_bays: int = 0
    total_height: float = 0.0
    total_width: float = 0.0
    stories: list[float] = field(default_factory=list)
    bays: list[float] = field(default_factory=list)
    nodes: list[dict] = field(default_factory=list)
    elements: list[dict] = field(default_factory=list)
    column_section: str = ""
    beam_section: str = ""
    material_name: str = ""
    E_MPa: float = 0.0
    num_elements: int = 0
    supports: str = "fixed"
    num_elements_per_member: int = 4

    # 단면/재료 물성 (시각화용)
    column_A_mm2: float = 0.0
    column_I_mm4: float = 0.0
    column_h_mm: float = 0.0
    beam_A_mm2: float = 0.0
    beam_I_mm4: float = 0.0
    beam_h_mm: float = 0.0
    fy_MPa: float = 0.0

    # 부재 매핑 (부재력 다이어그램용)
    member_info: list[dict] = field(default_factory=list)

    # 하중케이스 정의 및 결과
    load_cases: dict = field(default_factory=dict)
    case_results: dict = field(default_factory=dict)

    # 하중조합 정의 및 결과
    load_combinations: dict = field(default_factory=dict)
    combo_results: dict = field(default_factory=dict)

    # 부재력 다이어그램 데이터: {case_name: [member_force_data, ...]}
    member_forces: dict = field(default_factory=dict)

    # 하위호환용
    loads_info: list[dict] = field(default_factory=list)


@dataclass
class Frame2DResult:
    """2D 골조 해석 결과 (하위호환용, 단일 케이스)"""
    # 기하 정보
    num_stories: int = 0
    num_bays: int = 0
    total_height: float = 0.0
    total_width: float = 0.0
    stories: list[float] = field(default_factory=list)
    bays: list[float] = field(default_factory=list)

    # 노드/요소 정보
    nodes: list[dict] = field(default_factory=list)
    elements: list[dict] = field(default_factory=list)

    # 지점 반력
    reactions: list[dict] = field(default_factory=list)

    # 전체 최대값
    max_displacement_x: float = 0.0
    max_displacement_y: float = 0.0
    max_displacement_x_node: int = 0
    max_displacement_y_node: int = 0
    max_drift: float = 0.0
    max_drift_story: int = 0

    # 요소별 결과
    element_forces: list[dict] = field(default_factory=list)
    max_moment: float = 0.0
    max_moment_element: int = 0
    max_axial: float = 0.0
    max_axial_element: int = 0
    max_shear: float = 0.0
    max_shear_element: int = 0

    # 노드 변위
    nodal_displacements: list[dict] = field(default_factory=list)

    # 단면/재료 정보
    column_section: str = ""
    beam_section: str = ""
    material_name: str = ""

    # 해석 메타정보
    E_MPa: float = 0.0
    num_elements: int = 0

    # 시각화용 추가 정보
    loads_info: list[dict] = field(default_factory=list)
    supports: str = "fixed"


# ============================================================
# 기하 생성
# ============================================================

def generate_frame_geometry(
    stories: list[float],
    bays: list[float],
) -> tuple[list[Node2D], list[tuple[int, int, str]]]:
    """
    그리드 파라미터로부터 노드와 요소 연결 정보 생성

    Parameters
    ----------
    stories : list[float]
        각 층의 높이 (m), 아래에서 위로. 예: [3.5, 3.2, 3.2]
    bays : list[float]
        각 경간의 폭 (m), 왼쪽에서 오른쪽으로. 예: [6.0, 8.0, 6.0]

    Returns
    -------
    nodes : list[Node2D]
    connections : list[tuple[ni, nj, elem_type]]
    """
    n_stories = len(stories)
    n_bays = len(bays)
    n_cols = n_bays + 1

    nodes: list[Node2D] = []
    x_positions = [0.0]
    for bay in bays:
        x_positions.append(x_positions[-1] + bay)

    y_positions = [0.0]
    for story in stories:
        y_positions.append(y_positions[-1] + story)

    node_id = 1
    node_grid: dict[tuple[int, int], int] = {}

    for story_idx in range(n_stories + 1):
        for col_idx in range(n_cols):
            nodes.append(Node2D(
                id=node_id,
                x=x_positions[col_idx],
                y=y_positions[story_idx],
            ))
            node_grid[(story_idx, col_idx)] = node_id
            node_id += 1

    connections: list[tuple[int, int, str]] = []

    # 기둥: 각 층의 수직 요소
    for story_idx in range(n_stories):
        for col_idx in range(n_cols):
            ni = node_grid[(story_idx, col_idx)]
            nj = node_grid[(story_idx + 1, col_idx)]
            connections.append((ni, nj, "column"))

    # 보: 각 층의 수평 요소 (기초 제외)
    for story_idx in range(1, n_stories + 1):
        for bay_idx in range(n_bays):
            ni = node_grid[(story_idx, bay_idx)]
            nj = node_grid[(story_idx, bay_idx + 1)]
            connections.append((ni, nj, "beam"))

    return nodes, connections


# ============================================================
# 내부 함수: 모델 빌드, 하중 적용, 해석, 결과 추출
# ============================================================

def _build_frame_model(
    nodes: list[Node2D],
    connections: list[tuple[int, int, str]],
    base_nodes: list[int],
    supports: str,
    col_section_props,
    beam_section_props,
    E: float,
    column_section_name: str,
    beam_section_name: str,
    num_elements_per_member: int,
) -> tuple[list[Element2D], dict, list[int], int]:
    """
    OpenSees 모델 빌드 (노드, BC, 기하변환, 요소)

    Returns
    -------
    elements_info, member_to_elements, internal_nodes, total_elements
    """
    ops.model('basic', '-ndm', 2, '-ndf', 3)

    # 노드 생성
    for node in nodes:
        ops.node(node.id, node.x * 1000, node.y * 1000)  # m -> mm

    # 경계조건
    for bn in base_nodes:
        if supports == "fixed":
            ops.fix(bn, 1, 1, 1)
        else:
            ops.fix(bn, 1, 1, 0)

    # 기하변환
    ops.geomTransf('Linear', 1)  # 기둥용
    ops.geomTransf('Linear', 2)  # 보용

    # 요소 생성 (부재별 분할)
    elem_id = 0
    elements_info: list[Element2D] = []
    member_to_elements: dict[int, list[int]] = {}
    internal_nodes: list[int] = []
    next_node_id = len(nodes) + 1

    for member_idx, (ni, nj, elem_type) in enumerate(connections):
        member_id = member_idx + 1
        member_to_elements[member_id] = []

        if elem_type == "column":
            sec = col_section_props
            transf = 1
            section_name = column_section_name
        else:
            sec = beam_section_props
            transf = 2
            section_name = beam_section_name

        A = sec.A
        I = sec.I

        if num_elements_per_member == 1:
            elem_id += 1
            ops.element('elasticBeamColumn', elem_id, ni, nj, A, E, I, transf)
            elements_info.append(Element2D(elem_id, ni, nj, elem_type, section_name))
            member_to_elements[member_id].append(elem_id)
        else:
            node_i = next((n for n in nodes if n.id == ni), None)
            node_j = next((n for n in nodes if n.id == nj), None)
            if node_i is None or node_j is None:
                continue

            sub_nodes = [ni]
            for k in range(1, num_elements_per_member):
                ratio = k / num_elements_per_member
                x_new = node_i.x + ratio * (node_j.x - node_i.x)
                y_new = node_i.y + ratio * (node_j.y - node_i.y)
                ops.node(next_node_id, x_new * 1000, y_new * 1000)
                sub_nodes.append(next_node_id)
                internal_nodes.append(next_node_id)
                next_node_id += 1
            sub_nodes.append(nj)

            for k in range(num_elements_per_member):
                elem_id += 1
                ops.element('elasticBeamColumn', elem_id, sub_nodes[k], sub_nodes[k + 1], A, E, I, transf)
                elements_info.append(Element2D(elem_id, sub_nodes[k], sub_nodes[k + 1], elem_type, section_name))
                member_to_elements[member_id].append(elem_id)

    return elements_info, member_to_elements, internal_nodes, elem_id


def _apply_loads(
    loads: list[dict],
    n_stories: int,
    n_bays: int,
    n_cols: int,
    member_to_elements: dict[int, list[int]],
):
    """하중 적용 (timeSeries + pattern + loads)"""
    ops.timeSeries('Linear', 1)
    ops.pattern('Plain', 1, 1)

    for ld in loads:
        ld_type = ld.get("type", "floor")
        story = ld.get("story", 1)

        if ld_type == "floor":
            w = ld.get("value", 0.0) * 1000 / 1000  # kN/m -> N/mm
            story_beam_start = n_stories * n_cols + (story - 1) * n_bays + 1
            for member_id in range(story_beam_start, story_beam_start + n_bays):
                if member_id in member_to_elements:
                    for eid in member_to_elements[member_id]:
                        ops.eleLoad('-ele', eid, '-type', '-beamUniform', -w)

        elif ld_type == "lateral":
            fx = ld.get("fx", ld.get("value", 0.0)) * 1000  # kN -> N
            target_node = story * n_cols + 1
            ops.load(target_node, fx, 0, 0)

        elif ld_type == "nodal":
            node_id = ld.get("node", 1)
            fx = ld.get("fx", 0.0) * 1000
            fy = ld.get("fy", 0.0) * 1000
            mz = ld.get("mz", 0.0) * 1e6
            ops.load(node_id, fx, -fy, mz)


def _solve():
    """정적 해석 수행"""
    ops.system('BandGen')
    ops.numberer('Plain')
    ops.constraints('Plain')
    ops.integrator('LoadControl', 1.0)
    ops.algorithm('Linear')
    ops.analysis('Static')
    ops.analyze(1)
    ops.reactions()


def _extract_case_results(
    nodes: list[Node2D],
    elements_info: list[Element2D],
    base_nodes: list[int],
    stories: list[float],
    n_stories: int,
    n_cols: int,
    supports: str,
    member_to_elements: dict[int, list[int]],
    connections: list[tuple[int, int, str]],
    case_loads: list[dict] | None = None,
) -> Frame2DCaseResult:
    """현재 OpenSees 상태에서 결과 추출"""

    # 1. 노드 변위
    nodal_displacements = []
    max_dx, max_dy = 0.0, 0.0
    max_dx_node, max_dy_node = 0, 0

    for node in nodes:
        dx = ops.nodeDisp(node.id, 1)
        dy = ops.nodeDisp(node.id, 2)
        rz = ops.nodeDisp(node.id, 3)
        nodal_displacements.append({
            "node": node.id,
            "x_m": node.x,
            "y_m": node.y,
            "dx_mm": round(dx, 6),
            "dy_mm": round(dy, 6),
            "rz_rad": round(rz, 6),
        })
        if abs(dx) > abs(max_dx):
            max_dx = dx
            max_dx_node = node.id
        if abs(dy) > abs(max_dy):
            max_dy = dy
            max_dy_node = node.id

    # 2. 층간변위각
    story_drifts = []
    for story_idx in range(1, n_stories + 1):
        lower_nodes = [(story_idx - 1) * n_cols + i + 1 for i in range(n_cols)]
        upper_nodes = [story_idx * n_cols + i + 1 for i in range(n_cols)]
        lower_dx = sum(ops.nodeDisp(n, 1) for n in lower_nodes) / n_cols
        upper_dx = sum(ops.nodeDisp(n, 1) for n in upper_nodes) / n_cols
        story_height = stories[story_idx - 1] * 1000  # mm
        drift = (upper_dx - lower_dx) / story_height if story_height > 0 else 0
        story_drifts.append({
            "story": story_idx,
            "height_m": stories[story_idx - 1],
            "drift": round(abs(drift), 6),
        })

    max_drift = 0.0
    max_drift_story = 0
    for sd in story_drifts:
        if sd["drift"] > max_drift:
            max_drift = sd["drift"]
            max_drift_story = sd["story"]

    # 3. 지점 반력
    reactions = []
    for bn in base_nodes:
        rx = ops.nodeReaction(bn, 1) / 1000
        ry = ops.nodeReaction(bn, 2) / 1000
        mz = ops.nodeReaction(bn, 3) / 1e6
        node_info = next((n for n in nodes if n.id == bn), None)
        reactions.append({
            "node": bn,
            "x_m": node_info.x if node_info else 0,
            "RX_kN": round(rx, 2),
            "RY_kN": round(ry, 2),
            "MZ_kNm": round(mz, 2) if supports == "fixed" else 0,
        })

    # 4. 요소력
    element_forces = []
    max_M, max_N, max_V = 0.0, 0.0, 0.0
    max_M_elem, max_N_elem, max_V_elem = 0, 0, 0

    for elem in elements_info:
        forces = ops.eleForce(elem.id)
        N_i = forces[0] / 1000
        V_i = forces[1] / 1000
        M_i = forces[2] / 1e6
        N_j = forces[3] / 1000
        V_j = forces[4] / 1000
        M_j = forces[5] / 1e6

        element_forces.append({
            "element": elem.id,
            "type": elem.elem_type,
            "section": elem.section_name,
            "ni": elem.ni,
            "nj": elem.nj,
            "axial_kN": round(N_i, 2),
            "shear_i_kN": round(V_i, 2),
            "moment_i_kNm": round(M_i, 2),
            "shear_j_kN": round(V_j, 2),
            "moment_j_kNm": round(M_j, 2),
        })

        if abs(M_i) > abs(max_M):
            max_M = M_i
            max_M_elem = elem.id
        if abs(M_j) > abs(max_M):
            max_M = M_j
            max_M_elem = elem.id
        if abs(N_i) > abs(max_N):
            max_N = N_i
            max_N_elem = elem.id
        if abs(V_i) > abs(max_V):
            max_V = V_i
            max_V_elem = elem.id

    # 5. 층별 분석
    n_bays = n_cols - 1
    lat_forces = _get_lateral_forces(case_loads) if case_loads else {}
    story_data = _compute_story_data(
        nodal_displacements, stories, n_stories, n_cols,
        element_forces, elements_info, member_to_elements, connections,
        reactions=reactions,
        lateral_forces_by_story=lat_forces,
    )

    return Frame2DCaseResult(
        nodal_displacements=nodal_displacements,
        element_forces=element_forces,
        reactions=reactions,
        story_drifts=story_drifts,
        story_data=story_data,
        max_displacement_x=round(max_dx, 3),
        max_displacement_y=round(max_dy, 3),
        max_displacement_x_node=max_dx_node,
        max_displacement_y_node=max_dy_node,
        max_drift=round(max_drift, 6),
        max_drift_story=max_drift_story,
        max_moment=round(abs(max_M), 2),
        max_moment_element=max_M_elem,
        max_axial=round(abs(max_N), 2),
        max_axial_element=max_N_elem,
        max_shear=round(abs(max_V), 2),
        max_shear_element=max_V_elem,
    )


def _extract_member_forces(
    connections: list[tuple[int, int, str]],
    member_to_elements: dict[int, list[int]],
    elements_info: list[Element2D],
    nodes: list[Node2D],
    num_elements_per_member: int,
    n_stories: int,
    n_bays: int,
    n_cols: int,
) -> list[dict]:
    """
    각 부재의 부재력 다이어그램 데이터 추출
    s(0~L) 좌표에 대한 N(s), V(s), M(s) 곡선
    """
    elem_by_id = {e.id: e for e in elements_info}
    node_by_id = {n.id: n for n in nodes}
    member_forces_list = []

    for member_idx, (ni, nj, elem_type) in enumerate(connections):
        member_id = member_idx + 1
        sub_elem_ids = member_to_elements.get(member_id, [])
        if not sub_elem_ids:
            continue

        node_i = node_by_id.get(ni)
        node_j = node_by_id.get(nj)
        if not node_i or not node_j:
            continue

        member_length = math.sqrt(
            (node_j.x - node_i.x) ** 2 + (node_j.y - node_i.y) ** 2
        )
        sub_length = member_length / len(sub_elem_ids)

        # 부재 위치 정보 (story/bay/col)
        if elem_type == "column":
            story_idx = member_idx // n_cols  # 0-based
            col_idx = member_idx % n_cols
            location = {"story": story_idx + 1, "col": col_idx}
        else:
            beam_offset = member_idx - n_stories * n_cols
            story_idx = beam_offset // n_bays  # 0-based (0 = 1층)
            bay_idx = beam_offset % n_bays
            location = {"story": story_idx + 1, "bay": bay_idx}

        s_arr = []
        N_arr = []
        V_arr = []
        M_arr = []

        # 부재력 다이어그램: 부재 기준 연속 곡선 생성
        # OpenSees eleForce 반환값: [N_i, V_i, M_i, N_j, V_j, M_j] (절점 반력 관점)
        # - i-end 값: 요소 시작점에서의 내부력 (부재력 다이어그램용 그대로 사용)
        # - j-end 값: 절점 반력이므로 내부력 관점에서는 부호 반전 필요
        #
        # 연속성 보장: 각 sub-element의 i-end만 추가하고, 마지막 j-end만 별도 추가
        # 이렇게 하면 한 부재 내에서 N+1개의 점이 생성됨 (N = sub-element 개수)

        for k, eid in enumerate(sub_elem_ids):
            forces = ops.eleForce(eid)
            # i-end 내부력 (부재력 관점: 그대로 사용)
            N_i = forces[0] / 1000
            V_i = forces[1] / 1000
            M_i = forces[2] / 1e6

            s_start = k * sub_length
            s_arr.append(round(s_start, 6))
            N_arr.append(round(N_i, 4))
            V_arr.append(round(V_i, 4))
            M_arr.append(round(M_i, 4))

        # 마지막 sub-element의 j-end (부호 반전: 절점 반력 → 내부력)
        last_forces = ops.eleForce(sub_elem_ids[-1])
        N_j = -last_forces[3] / 1000
        V_j = -last_forces[4] / 1000
        M_j = -last_forces[5] / 1e6
        s_arr.append(round(member_length, 6))
        N_arr.append(round(N_j, 4))
        V_arr.append(round(V_j, 4))
        M_arr.append(round(M_j, 4))

        member_forces_list.append({
            "member_id": member_id,
            "type": elem_type,
            "ni": ni,
            "nj": nj,
            "length_m": round(member_length, 4),
            "sub_element_ids": sub_elem_ids,
            "location": location,
            "s": s_arr,
            "N_kN": N_arr,
            "V_kN": V_arr,
            "M_kNm": M_arr,
        })

    return member_forces_list


def _get_lateral_forces(case_loads: list[dict]) -> dict[int, float]:
    """하중 케이스 정의에서 층별 수평력 추출 (story → fx_kN)"""
    forces: dict[int, float] = {}
    for load in case_loads:
        if load.get("type") == "lateral":
            story = load.get("story", 0)
            fx = load.get("fx", 0.0)
            forces[story] = forces.get(story, 0.0) + fx
    return forces


def _compute_story_data(
    nodal_displacements: list[dict],
    stories: list[float],
    n_stories: int,
    n_cols: int,
    element_forces: list[dict],
    elements_info: list[Element2D],
    member_to_elements: dict[int, list[int]],
    connections: list[tuple[int, int, str]],
    reactions: list[dict] | None = None,
    lateral_forces_by_story: dict[int, float] | None = None,
) -> dict:
    """
    층별 분석 데이터 (변위, 전단)

    Parameters
    ----------
    reactions : 지점 반력 (reaction-based story shear 계산용)
    lateral_forces_by_story : 층별 수평력 {story_idx: fx_kN}

    Returns
    -------
    {
        "story_displacements": [{story, y_m, dx_left, dx_right, dx_avg}],
        "story_shears": [{story, shear_kN, shear_kN_signed, shear_kN_abs,
                          shear_rxn_kN, has_lateral}],
    }
    """
    # 노드 변위를 ID로 인덱싱
    disp_by_node = {d["node"]: d for d in nodal_displacements}

    # 층별 변위
    story_displacements = []
    y_cum = 0.0
    for story_idx in range(1, n_stories + 1):
        y_cum += stories[story_idx - 1]
        left_node = story_idx * n_cols + 1
        right_node = story_idx * n_cols + n_cols

        dx_left = disp_by_node.get(left_node, {}).get("dx_mm", 0.0)
        dx_right = disp_by_node.get(right_node, {}).get("dx_mm", 0.0)

        # 중간 노드들 평균
        all_dx = []
        for col in range(n_cols):
            nid = story_idx * n_cols + col + 1
            d = disp_by_node.get(nid, {})
            all_dx.append(d.get("dx_mm", 0.0))
        dx_avg = sum(all_dx) / len(all_dx) if all_dx else 0.0

        story_displacements.append({
            "story": story_idx,
            "y_m": round(y_cum, 2),
            "dx_left_mm": round(dx_left, 4),
            "dx_right_mm": round(dx_right, 4),
            "dx_avg_mm": round(dx_avg, 4),
        })

    # 층전단력 = 각 층 기둥들의 전단력 합
    # 기둥 부재: connections 앞부분 n_stories * n_cols 개
    # signed = ΣVx (부호 포함), abs = Σ|Vx|
    elem_force_by_id = {ef["element"]: ef for ef in element_forces}
    story_shears = []

    # 수평력 존재 여부
    lat = lateral_forces_by_story or {}
    has_lateral = bool(lat and any(abs(v) > 1e-10 for v in lat.values()))

    # Reaction-based base shear
    base_shear_rxn = 0.0
    if reactions:
        base_shear_rxn = sum(r.get("RX_kN", 0.0) for r in reactions)

    # Reaction-based story shear (위에서 아래로 수평력 누적)
    # story_shear_rxn(k) = Σ lateral forces at story k, k+1, ..., n_stories
    rxn_shears: dict[int, float] = {}
    cumulative = 0.0
    for s_idx in range(n_stories, 0, -1):
        cumulative += lat.get(s_idx, 0.0)
        rxn_shears[s_idx] = cumulative

    for story_idx in range(1, n_stories + 1):
        shear_signed = 0.0
        shear_abs = 0.0
        for col_idx in range(n_cols):
            # 기둥 member_id (0-based story_idx - 1)
            member_id = (story_idx - 1) * n_cols + col_idx + 1
            sub_eids = member_to_elements.get(member_id, [])
            if sub_eids:
                # 기둥 하단(i-end) 전단력
                ef = elem_force_by_id.get(sub_eids[0], {})
                v = ef.get("shear_i_kN", 0.0)
                shear_signed += v
                shear_abs += abs(v)
        story_shears.append({
            "story": story_idx,
            "shear_kN": round(shear_abs, 2),
            "shear_kN_signed": round(shear_signed, 2),
            "shear_kN_abs": round(shear_abs, 2),
            "shear_rxn_kN": round(rxn_shears.get(story_idx, 0.0), 2),
            "has_lateral": has_lateral,
        })

    return {
        "story_displacements": story_displacements,
        "story_shears": story_shears,
    }


# ============================================================
# 선형 중첩 (Load Combination)
# ============================================================

def _superpose_case_results(
    case_results: dict[str, Frame2DCaseResult],
    factors: dict[str, float],
    stories: list[float],
    n_stories: int,
    n_cols: int,
    base_nodes: list[int],
    supports: str,
    elements_info: list[Element2D],
    member_to_elements: dict[int, list[int]],
    connections: list[tuple[int, int, str]],
    load_cases_raw: dict[str, list[dict]] | None = None,
) -> Frame2DCaseResult:
    """하중케이스 결과를 선형 중첩하여 조합 결과 생성"""

    # 노드 변위 중첩
    first_case = next(iter(case_results.values()))
    node_ids = [d["node"] for d in first_case.nodal_displacements]

    # 노드별 변위 인덱싱
    case_disp_maps = {}
    for cname, cr in case_results.items():
        case_disp_maps[cname] = {d["node"]: d for d in cr.nodal_displacements}

    combined_disps = []
    max_dx, max_dy = 0.0, 0.0
    max_dx_node, max_dy_node = 0, 0

    for nid in node_ids:
        ref = case_disp_maps[next(iter(factors))].get(nid, {})
        cx = sum(
            factors.get(cn, 0) * case_disp_maps[cn].get(nid, {}).get("dx_mm", 0)
            for cn in factors if cn in case_results
        )
        cy = sum(
            factors.get(cn, 0) * case_disp_maps[cn].get(nid, {}).get("dy_mm", 0)
            for cn in factors if cn in case_results
        )
        crz = sum(
            factors.get(cn, 0) * case_disp_maps[cn].get(nid, {}).get("rz_rad", 0)
            for cn in factors if cn in case_results
        )
        combined_disps.append({
            "node": nid,
            "x_m": ref.get("x_m", 0),
            "y_m": ref.get("y_m", 0),
            "dx_mm": round(cx, 6),
            "dy_mm": round(cy, 6),
            "rz_rad": round(crz, 6),
        })
        if abs(cx) > abs(max_dx):
            max_dx = cx
            max_dx_node = nid
        if abs(cy) > abs(max_dy):
            max_dy = cy
            max_dy_node = nid

    # 요소력 중첩
    case_force_maps = {}
    for cname, cr in case_results.items():
        case_force_maps[cname] = {ef["element"]: ef for ef in cr.element_forces}

    first_forces = first_case.element_forces
    combined_forces = []
    max_M, max_N, max_V = 0.0, 0.0, 0.0
    max_M_elem, max_N_elem, max_V_elem = 0, 0, 0

    for ref_ef in first_forces:
        eid = ref_ef["element"]
        axial = sum(
            factors.get(cn, 0) * case_force_maps[cn].get(eid, {}).get("axial_kN", 0)
            for cn in factors if cn in case_results
        )
        si = sum(
            factors.get(cn, 0) * case_force_maps[cn].get(eid, {}).get("shear_i_kN", 0)
            for cn in factors if cn in case_results
        )
        mi = sum(
            factors.get(cn, 0) * case_force_maps[cn].get(eid, {}).get("moment_i_kNm", 0)
            for cn in factors if cn in case_results
        )
        sj = sum(
            factors.get(cn, 0) * case_force_maps[cn].get(eid, {}).get("shear_j_kN", 0)
            for cn in factors if cn in case_results
        )
        mj = sum(
            factors.get(cn, 0) * case_force_maps[cn].get(eid, {}).get("moment_j_kNm", 0)
            for cn in factors if cn in case_results
        )

        combined_forces.append({
            "element": eid,
            "type": ref_ef["type"],
            "section": ref_ef["section"],
            "ni": ref_ef["ni"],
            "nj": ref_ef["nj"],
            "axial_kN": round(axial, 2),
            "shear_i_kN": round(si, 2),
            "moment_i_kNm": round(mi, 2),
            "shear_j_kN": round(sj, 2),
            "moment_j_kNm": round(mj, 2),
        })

        if abs(mi) > abs(max_M):
            max_M = mi
            max_M_elem = eid
        if abs(mj) > abs(max_M):
            max_M = mj
            max_M_elem = eid
        if abs(axial) > abs(max_N):
            max_N = axial
            max_N_elem = eid
        if abs(si) > abs(max_V):
            max_V = si
            max_V_elem = eid

    # 반력 중첩
    case_rxn_maps = {}
    for cname, cr in case_results.items():
        case_rxn_maps[cname] = {r["node"]: r for r in cr.reactions}

    combined_rxns = []
    for bn in base_nodes:
        ref_r = case_rxn_maps[next(iter(factors))].get(bn, {})
        crx = sum(
            factors.get(cn, 0) * case_rxn_maps[cn].get(bn, {}).get("RX_kN", 0)
            for cn in factors if cn in case_results
        )
        cry = sum(
            factors.get(cn, 0) * case_rxn_maps[cn].get(bn, {}).get("RY_kN", 0)
            for cn in factors if cn in case_results
        )
        cmz = sum(
            factors.get(cn, 0) * case_rxn_maps[cn].get(bn, {}).get("MZ_kNm", 0)
            for cn in factors if cn in case_results
        )
        combined_rxns.append({
            "node": bn,
            "x_m": ref_r.get("x_m", 0),
            "RX_kN": round(crx, 2),
            "RY_kN": round(cry, 2),
            "MZ_kNm": round(cmz, 2) if supports == "fixed" else 0,
        })

    # 층간변위 재계산 (중첩 변위 기반)
    disp_by_node = {d["node"]: d for d in combined_disps}
    story_drifts = []
    for story_idx in range(1, n_stories + 1):
        lower_dx_sum = 0.0
        upper_dx_sum = 0.0
        for col in range(n_cols):
            lower_nid = (story_idx - 1) * n_cols + col + 1
            upper_nid = story_idx * n_cols + col + 1
            lower_dx_sum += disp_by_node.get(lower_nid, {}).get("dx_mm", 0)
            upper_dx_sum += disp_by_node.get(upper_nid, {}).get("dx_mm", 0)
        lower_avg = lower_dx_sum / n_cols
        upper_avg = upper_dx_sum / n_cols
        story_height_mm = stories[story_idx - 1] * 1000
        drift = (upper_avg - lower_avg) / story_height_mm if story_height_mm > 0 else 0
        story_drifts.append({
            "story": story_idx,
            "height_m": stories[story_idx - 1],
            "drift": round(abs(drift), 6),
        })

    max_drift = 0.0
    max_drift_story = 0
    for sd in story_drifts:
        if sd["drift"] > max_drift:
            max_drift = sd["drift"]
            max_drift_story = sd["story"]

    # 층별 분석 재계산 (조합의 수평력 = 구성 케이스 수평력의 선형 중첩)
    lateral_combo: dict[int, float] = {}
    if load_cases_raw:
        for cname, factor in factors.items():
            case_lat = _get_lateral_forces(load_cases_raw.get(cname, []))
            for story, fx in case_lat.items():
                lateral_combo[story] = lateral_combo.get(story, 0.0) + factor * fx
    story_data = _compute_story_data(
        combined_disps, stories, n_stories, n_cols,
        combined_forces, elements_info, member_to_elements, connections,
        reactions=combined_rxns,
        lateral_forces_by_story=lateral_combo,
    )

    return Frame2DCaseResult(
        nodal_displacements=combined_disps,
        element_forces=combined_forces,
        reactions=combined_rxns,
        story_drifts=story_drifts,
        story_data=story_data,
        max_displacement_x=round(max_dx, 3),
        max_displacement_y=round(max_dy, 3),
        max_displacement_x_node=max_dx_node,
        max_displacement_y_node=max_dy_node,
        max_drift=round(max_drift, 6),
        max_drift_story=max_drift_story,
        max_moment=round(abs(max_M), 2),
        max_moment_element=max_M_elem,
        max_axial=round(abs(max_N), 2),
        max_axial_element=max_N_elem,
        max_shear=round(abs(max_V), 2),
        max_shear_element=max_V_elem,
    )


def _superpose_member_forces(
    all_member_forces: dict[str, list[dict]],
    factors: dict[str, float],
) -> list[dict]:
    """부재력 다이어그램 선형 중첩"""
    first_case = next(iter(all_member_forces.values()))
    combined = []

    for m_idx, ref_mf in enumerate(first_case):
        s_arr = ref_mf["s"]  # 동일 기하 → 동일 s 배열
        N_arr = [0.0] * len(s_arr)
        V_arr = [0.0] * len(s_arr)
        M_arr = [0.0] * len(s_arr)

        for cname, factor in factors.items():
            if cname not in all_member_forces:
                continue
            cmf = all_member_forces[cname][m_idx]
            for i in range(len(s_arr)):
                N_arr[i] += factor * cmf["N_kN"][i]
                V_arr[i] += factor * cmf["V_kN"][i]
                M_arr[i] += factor * cmf["M_kNm"][i]

        combined.append({
            "member_id": ref_mf["member_id"],
            "type": ref_mf["type"],
            "ni": ref_mf["ni"],
            "nj": ref_mf["nj"],
            "length_m": ref_mf["length_m"],
            "sub_element_ids": ref_mf["sub_element_ids"],
            "location": ref_mf["location"],
            "s": s_arr,
            "N_kN": [round(v, 4) for v in N_arr],
            "V_kN": [round(v, 4) for v in V_arr],
            "M_kNm": [round(v, 4) for v in M_arr],
        })

    return combined


# ============================================================
# 메인 해석 함수
# ============================================================

def _build_member_info(
    connections: list[tuple[int, int, str]],
    member_to_elements: dict[int, list[int]],
    nodes: list[Node2D],
    n_stories: int,
    n_bays: int,
    n_cols: int,
) -> list[dict]:
    """부재 매핑 정보 생성 (시각화/CSV용)"""
    node_by_id = {n.id: n for n in nodes}
    info_list = []

    for member_idx, (ni, nj, elem_type) in enumerate(connections):
        member_id = member_idx + 1
        node_i = node_by_id.get(ni)
        node_j = node_by_id.get(nj)
        length = math.sqrt(
            (node_j.x - node_i.x) ** 2 + (node_j.y - node_i.y) ** 2
        ) if node_i and node_j else 0

        if elem_type == "column":
            story_idx = member_idx // n_cols
            col_idx = member_idx % n_cols
            location = {"story": story_idx + 1, "col": col_idx}
        else:
            beam_offset = member_idx - n_stories * n_cols
            story_idx = beam_offset // n_bays
            bay_idx = beam_offset % n_bays
            location = {"story": story_idx + 1, "bay": bay_idx}

        info_list.append({
            "id": member_id,
            "ni": ni,
            "nj": nj,
            "type": elem_type,
            "sub_element_ids": member_to_elements.get(member_id, []),
            "length_m": round(length, 4),
            "location": location,
        })

    return info_list


def analyze_frame_2d_multi(
    stories: list[float],
    bays: list[float],
    load_cases: dict[str, list[dict]],
    supports: Literal["fixed", "pinned"] = "fixed",
    column_section: str = "H-300x300",
    beam_section: str = "H-400x200",
    material_name: str = "SS275",
    num_elements_per_member: int = 4,
    load_combinations: Optional[dict[str, dict[str, float]]] = None,
) -> Frame2DMultiCaseResult:
    """
    멀티 하중케이스 2D 골조 정적 해석

    Parameters
    ----------
    stories : list[float]
        각 층의 높이 (m), 아래에서 위로
    bays : list[float]
        각 경간의 폭 (m), 왼쪽에서 오른쪽으로
    load_cases : dict[str, list[dict]]
        하중케이스 딕셔너리. 예: {"DL": [...], "EQX": [...]}
    supports : "fixed" or "pinned"
    column_section, beam_section, material_name : str
    num_elements_per_member : int
    load_combinations : dict, optional
        하중조합. 예: {"1.0DL+1.0EQX": {"DL": 1.0, "EQX": 1.0}}
    """
    n_stories = len(stories)
    n_bays = len(bays)

    if n_stories < 1 or n_stories > 10:
        raise ValueError(f"층수는 1~10 범위여야 합니다. 입력: {n_stories}")
    if n_bays < 1 or n_bays > 5:
        raise ValueError(f"경간 수는 1~5 범위여야 합니다. 입력: {n_bays}")

    # 단면/재료 조회
    col_section = get_section_from_db(column_section)
    if col_section is None:
        col_section = DEFAULT_SECTIONS.get(column_section)
    if col_section is None:
        raise ValueError(f"Unknown column section: {column_section}")

    beam_sec = get_section_from_db(beam_section)
    if beam_sec is None:
        beam_sec = DEFAULT_SECTIONS.get(beam_section)
    if beam_sec is None:
        raise ValueError(f"Unknown beam section: {beam_section}")

    material = get_material_from_db(material_name)
    if material is None:
        material = DEFAULT_MATERIALS.get(material_name)
    if material is None:
        raise ValueError(f"Unknown material: {material_name}")

    # 기하 생성
    nodes, connections = generate_frame_geometry(stories, bays)
    n_cols = n_bays + 1
    base_nodes = [i + 1 for i in range(n_cols)]
    E = material.E

    # 노드/요소 출력용
    nodes_output = [{"id": n.id, "x_m": n.x, "y_m": n.y} for n in nodes]

    # 케이스별 해석
    case_results: dict[str, Frame2DCaseResult] = {}
    member_forces_all: dict[str, list[dict]] = {}
    elements_info = None
    member_to_elements = None
    total_elements = 0

    for case_name, case_loads in load_cases.items():
        ops.wipe()
        elements_info, member_to_elements, internal_nodes, total_elements = _build_frame_model(
            nodes, connections, base_nodes, supports,
            col_section, beam_sec, E,
            column_section, beam_section, num_elements_per_member,
        )
        _apply_loads(case_loads, n_stories, n_bays, n_cols, member_to_elements)
        _solve()

        case_results[case_name] = _extract_case_results(
            nodes, elements_info, base_nodes, stories,
            n_stories, n_cols, supports, member_to_elements, connections,
            case_loads=case_loads,
        )
        member_forces_all[case_name] = _extract_member_forces(
            connections, member_to_elements, elements_info, nodes,
            num_elements_per_member, n_stories, n_bays, n_cols,
        )

    # 요소 출력용
    elements_output = []
    if elements_info:
        elements_output = [
            {"id": e.id, "ni": e.ni, "nj": e.nj, "type": e.elem_type, "section": e.section_name}
            for e in elements_info
        ]

    # 부재 매핑 정보
    member_info = []
    if member_to_elements:
        member_info = _build_member_info(
            connections, member_to_elements, nodes, n_stories, n_bays, n_cols,
        )

    # 하중조합
    combo_results: dict[str, Frame2DCaseResult] = {}
    combo_member_forces: dict[str, list[dict]] = {}
    if load_combinations and elements_info:
        for combo_name, factors in load_combinations.items():
            combo_results[combo_name] = _superpose_case_results(
                case_results, factors, stories, n_stories, n_cols,
                base_nodes, supports, elements_info, member_to_elements, connections,
                load_cases_raw=load_cases,
            )
            combo_member_forces[combo_name] = _superpose_member_forces(
                member_forces_all, factors,
            )

    # 모든 member_forces 합치기
    all_member_forces = {**member_forces_all, **combo_member_forces}

    # 첫 번째 케이스의 하중을 loads_info로 (하위호환용)
    first_loads = next(iter(load_cases.values())) if load_cases else []

    return Frame2DMultiCaseResult(
        num_stories=n_stories,
        num_bays=n_bays,
        total_height=sum(stories),
        total_width=sum(bays),
        stories=stories,
        bays=bays,
        nodes=nodes_output,
        elements=elements_output,
        column_section=column_section,
        beam_section=beam_section,
        material_name=material_name,
        E_MPa=E,
        num_elements=total_elements,
        supports=supports,
        num_elements_per_member=num_elements_per_member,
        column_A_mm2=col_section.A,
        column_I_mm4=col_section.I,
        column_h_mm=col_section.h,
        beam_A_mm2=beam_sec.A,
        beam_I_mm4=beam_sec.I,
        beam_h_mm=beam_sec.h,
        fy_MPa=material.fy,
        member_info=member_info,
        load_cases=load_cases,
        case_results=case_results,
        load_combinations=load_combinations or {},
        combo_results=combo_results,
        member_forces=all_member_forces,
        loads_info=first_loads,
    )


def _multi_to_legacy_result(
    multi: Frame2DMultiCaseResult,
    case_name: str,
    loads: list[dict],
) -> Frame2DResult:
    """MultiCaseResult → 기존 Frame2DResult 변환"""
    cr = multi.case_results.get(case_name)
    if cr is None:
        raise ValueError(f"Case '{case_name}' not found")

    return Frame2DResult(
        num_stories=multi.num_stories,
        num_bays=multi.num_bays,
        total_height=multi.total_height,
        total_width=multi.total_width,
        stories=multi.stories,
        bays=multi.bays,
        nodes=multi.nodes,
        elements=multi.elements,
        reactions=cr.reactions,
        max_displacement_x=cr.max_displacement_x,
        max_displacement_y=cr.max_displacement_y,
        max_displacement_x_node=cr.max_displacement_x_node,
        max_displacement_y_node=cr.max_displacement_y_node,
        max_drift=cr.max_drift,
        max_drift_story=cr.max_drift_story,
        element_forces=cr.element_forces,
        max_moment=cr.max_moment,
        max_moment_element=cr.max_moment_element,
        max_axial=cr.max_axial,
        max_axial_element=cr.max_axial_element,
        max_shear=cr.max_shear,
        max_shear_element=cr.max_shear_element,
        nodal_displacements=cr.nodal_displacements,
        column_section=multi.column_section,
        beam_section=multi.beam_section,
        material_name=multi.material_name,
        E_MPa=multi.E_MPa,
        num_elements=multi.num_elements,
        loads_info=loads,
        supports=multi.supports,
    )


def analyze_frame_2d(
    stories: list[float],
    bays: list[float],
    loads: list[dict],
    supports: Literal["fixed", "pinned"] = "fixed",
    column_section: str = "H-300x300",
    beam_section: str = "H-400x200",
    material_name: str = "SS275",
    num_elements_per_member: int = 4,
) -> Frame2DResult:
    """
    2D 골조 정적 해석 (하위호환 wrapper)

    Parameters
    ----------
    stories : 각 층의 높이 (m)
    bays : 각 경간의 폭 (m)
    loads : 하중 리스트
    supports : 기초 지점 조건 ("fixed" / "pinned")
    column_section, beam_section, material_name : 단면/재료
    num_elements_per_member : 부재당 요소 분할 수

    Returns
    -------
    Frame2DResult
    """
    multi = analyze_frame_2d_multi(
        stories=stories,
        bays=bays,
        load_cases={"LC1": loads},
        supports=supports,
        column_section=column_section,
        beam_section=beam_section,
        material_name=material_name,
        num_elements_per_member=num_elements_per_member,
    )
    return _multi_to_legacy_result(multi, "LC1", loads)
