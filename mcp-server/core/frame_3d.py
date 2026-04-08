"""
3D 골조 해석 모듈
OpenSeesPy를 사용한 3D 프레임 정적 해석
- 멀티 하중케이스 / 하중조합 지원
- X/Y 양방향 다경간
- 6-DOF 노드 변위, 12성분 요소력
- 층간변위각 (X/Y 양방향)

===================================================================================
COORDINATE SYSTEM
===================================================================================
  X = 수평 (bay_x 방향)
  Y = 수평 (bay_y 방향)
  Z = 수직 (위, 높이 방향)

SIGN CONVENTION
  - eleForce(tag) returns 12 values:
    [N_i, Vy_i, Vz_i, T_i, My_i, Mz_i, N_j, Vy_j, Vz_j, T_j, My_j, Mz_j]
  - Local coordinates defined by element orientation + geomTransf vecxz
  - Results stored in OpenSees local convention
===================================================================================
"""
from __future__ import annotations

import math
from core.ops_compat import ops
from dataclasses import dataclass, field
from typing import Literal, Optional

from core.section_3d import get_section_3d, BeamSection3D, DEFAULT_SECTIONS_3D
from core.simple_beam import get_material_from_db, DEFAULT_MATERIALS


# ============================================================
# 데이터 구조
# ============================================================

@dataclass
class Node3D:
    """3D 노드 정보"""
    id: int
    x: float  # m (bay_x 방향)
    y: float  # m (bay_y 방향)
    z: float  # m (높이)


@dataclass
class Element3D:
    """3D 요소 정보"""
    id: int
    ni: int
    nj: int
    elem_type: str  # "column", "beam_x", "beam_y"
    section_name: str


@dataclass
class Frame3DCaseResult:
    """단일 하중케이스/조합 결과"""
    nodal_displacements: list[dict] = field(default_factory=list)
    element_forces: list[dict] = field(default_factory=list)
    reactions: list[dict] = field(default_factory=list)
    story_drifts: list[dict] = field(default_factory=list)
    story_data: dict = field(default_factory=dict)

    # 최대값
    max_displacement_x: float = 0.0
    max_displacement_y: float = 0.0
    max_displacement_z: float = 0.0
    max_displacement_x_node: int = 0
    max_displacement_y_node: int = 0
    max_displacement_z_node: int = 0
    max_drift_x: float = 0.0
    max_drift_y: float = 0.0
    max_drift_x_story: int = 0
    max_drift_y_story: int = 0
    max_moment: float = 0.0
    max_moment_element: int = 0
    max_axial: float = 0.0
    max_axial_element: int = 0
    max_shear: float = 0.0
    max_shear_element: int = 0
    max_torsion: float = 0.0
    max_torsion_element: int = 0


@dataclass
class Frame3DMultiCaseResult:
    """멀티케이스 3D 골조 해석 결과"""
    # 기하 정보
    num_stories: int = 0
    num_bays_x: int = 0
    num_bays_y: int = 0
    total_height: float = 0.0
    total_width_x: float = 0.0
    total_width_y: float = 0.0
    stories: list[float] = field(default_factory=list)
    bays_x: list[float] = field(default_factory=list)
    bays_y: list[float] = field(default_factory=list)
    nodes: list[dict] = field(default_factory=list)
    elements: list[dict] = field(default_factory=list)
    supports: str = "fixed"
    num_elements_per_member: int = 4

    # 단면/재료
    column_section: str = ""
    beam_x_section: str = ""
    beam_y_section: str = ""
    material_name: str = ""
    E_MPa: float = 0.0
    G_MPa: float = 0.0
    num_elements: int = 0

    # 단면 물성
    column_A_mm2: float = 0.0
    column_Ix_mm4: float = 0.0
    column_Iy_mm4: float = 0.0
    column_J_mm4: float = 0.0
    beam_x_A_mm2: float = 0.0
    beam_x_Ix_mm4: float = 0.0
    beam_x_Iy_mm4: float = 0.0
    beam_x_J_mm4: float = 0.0
    beam_y_A_mm2: float = 0.0
    beam_y_Ix_mm4: float = 0.0
    beam_y_Iy_mm4: float = 0.0
    beam_y_J_mm4: float = 0.0
    fy_MPa: float = 0.0

    # 단면 치수 (design check용)
    column_h_mm: float = 0.0
    column_b_mm: float = 0.0
    column_tw_mm: float = 0.0
    column_tf_mm: float = 0.0
    beam_x_h_mm: float = 0.0
    beam_x_b_mm: float = 0.0
    beam_x_tw_mm: float = 0.0
    beam_x_tf_mm: float = 0.0
    beam_y_h_mm: float = 0.0
    beam_y_b_mm: float = 0.0
    beam_y_tw_mm: float = 0.0
    beam_y_tf_mm: float = 0.0

    # 부재 매핑
    member_info: list[dict] = field(default_factory=list)

    # 하중케이스/조합 결과
    load_cases: dict = field(default_factory=dict)
    case_results: dict = field(default_factory=dict)
    load_combinations: dict = field(default_factory=dict)
    combo_results: dict = field(default_factory=dict)
    member_forces: dict = field(default_factory=dict)

    # 부재 릴리즈 (힌지) 설정
    member_releases: dict = field(default_factory=dict)

    # 기하비선형 설정
    geometric_nonlinearity: str = "linear"  # "linear" or "pdelta"

    # 해석 메타데이터 (추적 및 리포트용)
    analysis_metadata: dict = field(default_factory=dict)

    # 고유치해석 결과
    modal_analysis: dict = field(default_factory=dict)


# ============================================================
# 부재 릴리즈 헬퍼 (3D)
# ============================================================

_RELEASE_MAP = {"i": 1, "j": 2, "both": 3}


def _get_release_code_3d(elem_type: str, member_releases: dict | None) -> int | None:
    """부재 유형에 대한 release code 반환 (1=i, 2=j, 3=both). None=릴리즈 없음."""
    if not member_releases:
        return None
    release_str = member_releases.get(elem_type)
    if release_str is None:
        return None
    return _RELEASE_MAP.get(release_str)


# 부재 방향별 비틀림(torsion) DOF → equalDOF에서 연결할 회전 DOF
# beam_x(X축): torsion=RX(4), bending free=RY(5),RZ(6)
# beam_y(Y축): torsion=RY(5), bending free=RX(4),RZ(6)
# column(Z축): torsion=RZ(6), bending free=RX(4),RY(5)
_TORSION_DOF = {"beam_x": 4, "beam_y": 5, "column": 6}


# ============================================================
# 기하 생성
# ============================================================

_COORD_TOL = 3  # 좌표 병합 소수점 자릿수 (mm 단위, 0.001m)


def _get_story_nodes_from_grid(node_grid: dict, n_cols_x: int, n_cols_y: int,
                                n_stories: int) -> dict[int, list[int]]:
    """정형 node_grid에서 story_nodes 딕셔너리 생성."""
    story_nodes = {}
    for s in range(n_stories + 1):
        story_nodes[s] = [
            node_grid[(s, cx, cy)]
            for cy in range(n_cols_y)
            for cx in range(n_cols_x)
        ]
    return story_nodes


def _generate_irregular_geometry(
    stories: list[float],
    zones: list[dict],
) -> tuple[list[Node3D], list[tuple[int, int, str]], dict[int, list[int]],
           list[int], list[dict]]:
    """비정형 프레임 기하 생성 (존 기반, 노드 병합).

    Args:
        stories: 층고 리스트 (m)
        zones: 존 정보 dict 리스트
            각 존: {id, bays_x, bays_y, origin_x, origin_y, story_from, story_to}

    Returns:
        nodes: 노드 목록
        connections: [(ni, nj, elem_type), ...]
        story_nodes: {story_level: [node_ids]}
        base_nodes: 기초 노드 ID 목록
        member_metadata: 부재별 메타데이터 (story, zone_id, trib_width, direction)
    """
    n_stories = len(stories)
    z_coords = [0.0]
    for sh in stories:
        z_coords.append(z_coords[-1] + sh)

    def rkey(x, y, z):
        return (round(x, _COORD_TOL), round(y, _COORD_TOL), round(z, _COORD_TOL))

    coord_to_node: dict[tuple, int] = {}
    nodes: list[Node3D] = []
    story_nodes: dict[int, list[int]] = {s: [] for s in range(n_stories + 1)}
    node_id = 1

    def _zone_active_at_level(zone, level):
        """level=0은 base, level>=1은 해당 층 상단."""
        if level == 0:
            return zone.get("story_from", 1) <= 1
        return (zone.get("story_from", 1) <= level and
                (zone.get("story_to") is None or zone["story_to"] >= level))

    def _zone_x_coords(zone):
        coords = [zone.get("origin_x", 0.0)]
        for bx in zone["bays_x"]:
            coords.append(coords[-1] + bx)
        return coords

    def _zone_y_coords(zone):
        coords = [zone.get("origin_y", 0.0)]
        for by in zone["bays_y"]:
            coords.append(coords[-1] + by)
        return coords

    # 1단계: 모든 존에서 노드 생성 (좌표 기반 병합)
    for zone in zones:
        x_crds = _zone_x_coords(zone)
        y_crds = _zone_y_coords(zone)

        for s in range(n_stories + 1):
            if not _zone_active_at_level(zone, s):
                continue
            z = z_coords[s]
            for y in y_crds:
                for x in x_crds:
                    key = rkey(x, y, z)
                    if key not in coord_to_node:
                        node = Node3D(id=node_id, x=x, y=y, z=z)
                        nodes.append(node)
                        coord_to_node[key] = node_id
                        story_nodes[s].append(node_id)
                        node_id += 1

    base_nodes = list(story_nodes[0])

    # 2단계: 부재 연결 (중복 방지)
    connections: list[tuple[int, int, str]] = []
    member_metadata: list[dict] = []
    conn_set: set[tuple[int, int, str]] = set()

    def _add_conn(ni, nj, etype, meta):
        nonlocal connections
        key = (min(ni, nj), max(ni, nj), etype)
        if key not in conn_set:
            conn_set.add(key)
            connections.append((ni, nj, etype))
            member_metadata.append(meta)

    for zone in zones:
        x_crds = _zone_x_coords(zone)
        y_crds = _zone_y_coords(zone)
        bays_x = zone["bays_x"]
        bays_y = zone["bays_y"]
        zid = zone["id"]
        n_cx = len(x_crds)
        n_cy = len(y_crds)

        # 기둥: story s → s+1
        for s in range(n_stories):
            low_active = _zone_active_at_level(zone, s)
            up_active = _zone_active_at_level(zone, s + 1)
            if not (low_active and up_active):
                continue
            for y in y_crds:
                for x in x_crds:
                    ni = coord_to_node[rkey(x, y, z_coords[s])]
                    nj = coord_to_node[rkey(x, y, z_coords[s + 1])]
                    _add_conn(ni, nj, "column", {
                        "story": s + 1, "zone_id": zid, "trib_width": 0.0,
                    })

        # Beam_X: story ≥ 1, 인접 X 노드 연결
        for s in range(1, n_stories + 1):
            if not _zone_active_at_level(zone, s):
                continue
            z = z_coords[s]
            for iy, y in enumerate(y_crds):
                # tributary width (Y방향)
                trib_y = 0.0
                if iy > 0:
                    trib_y += bays_y[iy - 1] / 2.0
                if iy < n_cy - 1:
                    trib_y += bays_y[iy] / 2.0
                if trib_y == 0.0:
                    trib_y = bays_y[0] if bays_y else 1.0

                for ix in range(n_cx - 1):
                    ni = coord_to_node[rkey(x_crds[ix], y, z)]
                    nj = coord_to_node[rkey(x_crds[ix + 1], y, z)]
                    _add_conn(ni, nj, "beam_x", {
                        "story": s, "zone_id": zid, "trib_width": trib_y,
                    })

        # Beam_Y: story ≥ 1, 인접 Y 노드 연결
        for s in range(1, n_stories + 1):
            if not _zone_active_at_level(zone, s):
                continue
            z = z_coords[s]
            for ix, x in enumerate(x_crds):
                trib_x = 0.0
                if ix > 0:
                    trib_x += bays_x[ix - 1] / 2.0
                if ix < n_cx - 1:
                    trib_x += bays_x[ix] / 2.0
                if trib_x == 0.0:
                    trib_x = bays_x[0] if bays_x else 1.0

                for iy in range(n_cy - 1):
                    ni = coord_to_node[rkey(x, y_crds[iy], z)]
                    nj = coord_to_node[rkey(x, y_crds[iy + 1], z)]
                    _add_conn(ni, nj, "beam_y", {
                        "story": s, "zone_id": zid, "trib_width": trib_x,
                    })

    return nodes, connections, story_nodes, base_nodes, member_metadata


def _generate_frame_3d_geometry(
    stories: list[float],
    bays_x: list[float],
    bays_y: list[float],
) -> tuple[list[Node3D], list[tuple[int, int, str]], dict, list[int]]:
    """3D 프레임 그리드 노드와 부재 연결 생성.

    Returns:
        nodes: 노드 목록
        connections: [(ni, nj, elem_type), ...]
        node_grid: {(story, cx, cy): node_id}
        base_nodes: 기초 노드 ID 목록
    """
    n_stories = len(stories)
    n_cols_x = len(bays_x) + 1
    n_cols_y = len(bays_y) + 1

    # X, Y 좌표 계산
    x_coords = [0.0]
    for bx in bays_x:
        x_coords.append(x_coords[-1] + bx)

    y_coords = [0.0]
    for by in bays_y:
        y_coords.append(y_coords[-1] + by)

    # Z 좌표 (층 높이 누적)
    z_coords = [0.0]
    for sh in stories:
        z_coords.append(z_coords[-1] + sh)

    # 노드 생성
    nodes = []
    node_grid = {}
    node_id = 1

    for s in range(n_stories + 1):
        for cy in range(n_cols_y):
            for cx in range(n_cols_x):
                node = Node3D(id=node_id, x=x_coords[cx], y=y_coords[cy], z=z_coords[s])
                nodes.append(node)
                node_grid[(s, cx, cy)] = node_id
                node_id += 1

    # 기초 노드 (story=0)
    base_nodes = [node_grid[(0, cx, cy)] for cy in range(n_cols_y) for cx in range(n_cols_x)]

    # 부재 연결
    connections = []

    # 1. 기둥: 모든 (cx, cy) 위치에서 story s → s+1
    for s in range(n_stories):
        for cy in range(n_cols_y):
            for cx in range(n_cols_x):
                ni = node_grid[(s, cx, cy)]
                nj = node_grid[(s + 1, cx, cy)]
                connections.append((ni, nj, "column"))

    # 2. Beam_X: story≥1에서, 각 Y라인마다 인접 X노드 연결
    for s in range(1, n_stories + 1):
        for cy in range(n_cols_y):
            for cx in range(n_cols_x - 1):
                ni = node_grid[(s, cx, cy)]
                nj = node_grid[(s, cx + 1, cy)]
                connections.append((ni, nj, "beam_x"))

    # 3. Beam_Y: story≥1에서, 각 X라인마다 인접 Y노드 연결
    for s in range(1, n_stories + 1):
        for cx in range(n_cols_x):
            for cy in range(n_cols_y - 1):
                ni = node_grid[(s, cx, cy)]
                nj = node_grid[(s, cx, cy + 1)]
                connections.append((ni, nj, "beam_y"))

    return nodes, connections, node_grid, base_nodes


# ============================================================
# 모델 구축
# ============================================================

def _build_frame_3d_model(
    nodes: list[Node3D],
    connections: list[tuple[int, int, str]],
    base_nodes: list[int],
    supports: str,
    col_sec: BeamSection3D,
    beam_x_sec: BeamSection3D,
    beam_y_sec: BeamSection3D,
    E: float,
    G: float,
    num_elements_per_member: int,
    rigid_diaphragm: bool = False,
    node_grid: dict | None = None,
    n_cols_x: int = 0,
    n_cols_y: int = 0,
    num_stories: int = 0,
    member_releases: dict | None = None,
    geometric_nonlinearity: str = "linear",
    story_nodes_map: dict[int, list[int]] | None = None,
) -> tuple[list[Element3D], dict[int, list[int]], list[dict], int]:
    """OpenSees 3D 모델 구축.

    Returns:
        elements_info: 요소 목록
        member_to_elements: {member_id: [elem_ids]}
        member_info_list: 부재 정보 목록
        next_node_id: 다음 사용 가능 노드 ID
    """
    ops.wipe()
    ops.model('basic', '-ndm', 3, '-ndf', 6)

    # 노드 생성 (m → mm)
    for node in nodes:
        ops.node(node.id, node.x * 1000, node.y * 1000, node.z * 1000)

    # 경계조건
    for bn in base_nodes:
        if supports == "fixed":
            ops.fix(bn, 1, 1, 1, 1, 1, 1)
        else:  # pinned
            ops.fix(bn, 1, 1, 1, 0, 0, 0)

    # 강체 다이어프램 (층별 수평면 구속)
    if rigid_diaphragm:
        if story_nodes_map:
            # 비정형: story_nodes_map에서 직접 사용
            for s in range(1, num_stories + 1):
                snodes = story_nodes_map.get(s, [])
                if len(snodes) < 2:
                    continue
                master_nid = snodes[len(snodes) // 2]
                slave_nids = [n for n in snodes if n != master_nid]
                if slave_nids:
                    ops.rigidDiaphragm(3, master_nid, *slave_nids)
        elif node_grid:
            # 정형: 기존 방식
            cx_m = n_cols_x // 2
            cy_m = n_cols_y // 2
            for s in range(1, num_stories + 1):
                master_nid = node_grid[(s, cx_m, cy_m)]
                slave_nids = [
                    node_grid[(s, cx, cy)]
                    for cx in range(n_cols_x) for cy in range(n_cols_y)
                    if node_grid[(s, cx, cy)] != master_nid
                ]
                if slave_nids:
                    ops.rigidDiaphragm(3, master_nid, *slave_nids)

    # 기하변환 (vecxz 벡터)
    # 참고: opensees 0.1.x에서 3D 'PDelta'는 기하강성 미반영 (silently ignored).
    #       대안으로 'Corotational' 변환 사용 (3D P-Delta + 대변위 효과 포함).
    transf_type = 'Corotational' if geometric_nonlinearity == "pdelta" else 'Linear'
    # 기둥 (수직, Z방향): vecxz = global X
    ops.geomTransf(transf_type, 1, 1.0, 0.0, 0.0)
    # Beam_X (X방향): vecxz = global Z (up)
    ops.geomTransf(transf_type, 2, 0.0, 0.0, 1.0)
    # Beam_Y (Y방향): vecxz = global Z (up)
    ops.geomTransf(transf_type, 3, 0.0, 0.0, 1.0)

    # 요소 생성
    elements_info = []
    member_to_elements = {}
    member_info_list = []
    elem_id = 1
    next_node_id = max(n.id for n in nodes) + 1

    for member_id, (ni, nj, etype) in enumerate(connections, start=1):
        # 단면 물성 및 변환 결정
        if etype == "column":
            sec = col_sec
            transf = 1
            # 기둥: 로컬 x=Z방향, 로컬 y~Y, 로컬 z~X
            # Iy(로컬y축 주위 휨) = X방향 평면내 → section.Ix
            # Iz(로컬z축 주위 휨) = Y방향 평면내 → section.Iy (or Ix if symmetric)
            os_Iy = sec.Ix
            os_Iz = sec.Iy
        elif etype == "beam_x":
            sec = beam_x_sec
            transf = 2
            # beam_x: 로컬 x=X방향, 로컬 z~Z(up), 로컬 y~-Y
            # Iy(로컬y축 주위 휨) = XZ 평면내 중력방향 = section.Ix (강축)
            # Iz(로컬z축 주위 휨) = XY 평면내 수평방향 = section.Iy (약축)
            os_Iy = sec.Ix
            os_Iz = sec.Iy
        else:  # beam_y
            sec = beam_y_sec
            transf = 3
            # beam_y: 로컬 x=Y방향, 로컬 z~Z(up), 로컬 y~X
            os_Iy = sec.Ix
            os_Iz = sec.Iy

        # 부재 세분화
        ni_node = _get_node_by_id(nodes, ni)
        nj_node = _get_node_by_id(nodes, nj)
        member_elem_ids = []

        # 부재 릴리즈 코드 결정 (1=i, 2=j, 3=both)
        rel_code = _get_release_code_3d(etype, member_releases)

        # 릴리즈 힌지 노드 생성 (equalDOF 방식)
        # 변환(1,2,3) + 비틀림 DOF 연결, 나머지 회전 DOF 자유
        # 비틀림 DOF: beam_x=4(RX), beam_y=5(RY), column=6(RZ)
        torsion_dof = _TORSION_DOF[etype]
        actual_ni = ni
        if rel_code in (1, 3):
            hinge_ni = next_node_id
            ops.node(hinge_ni, ni_node.x * 1000, ni_node.y * 1000, ni_node.z * 1000)
            ops.equalDOF(ni, hinge_ni, 1, 2, 3, torsion_dof)
            actual_ni = hinge_ni
            next_node_id += 1

        actual_nj = nj
        if rel_code in (2, 3):
            hinge_nj = next_node_id
            ops.node(hinge_nj, nj_node.x * 1000, nj_node.y * 1000, nj_node.z * 1000)
            ops.equalDOF(nj, hinge_nj, 1, 2, 3, torsion_dof)
            actual_nj = hinge_nj
            next_node_id += 1

        if num_elements_per_member <= 1:
            # 세분화 없이 단일 요소
            ops.element('elasticBeamColumn', elem_id, actual_ni, actual_nj,
                        sec.A, E, G, sec.J, os_Iy, os_Iz, transf)
            elements_info.append(Element3D(elem_id, actual_ni, actual_nj, etype, sec.name))
            member_elem_ids.append(elem_id)
            elem_id += 1
        else:
            # 내부 노드 생성 및 서브요소 생성
            sub_nodes = [actual_ni]
            for k in range(1, num_elements_per_member):
                ratio = k / num_elements_per_member
                sx = ni_node.x + ratio * (nj_node.x - ni_node.x)
                sy = ni_node.y + ratio * (nj_node.y - ni_node.y)
                sz = ni_node.z + ratio * (nj_node.z - ni_node.z)
                ops.node(next_node_id, sx * 1000, sy * 1000, sz * 1000)
                sub_nodes.append(next_node_id)
                next_node_id += 1
            sub_nodes.append(actual_nj)

            for k in range(num_elements_per_member):
                ops.element('elasticBeamColumn', elem_id,
                            sub_nodes[k], sub_nodes[k + 1],
                            sec.A, E, G, sec.J, os_Iy, os_Iz, transf)
                elements_info.append(Element3D(elem_id, sub_nodes[k], sub_nodes[k + 1],
                                               etype, sec.name))
                member_elem_ids.append(elem_id)
                elem_id += 1

        member_to_elements[member_id] = member_elem_ids

        # 부재 정보 저장
        length = math.sqrt(
            (nj_node.x - ni_node.x) ** 2
            + (nj_node.y - ni_node.y) ** 2
            + (nj_node.z - ni_node.z) ** 2
        )
        member_info_list.append({
            "member_id": member_id,
            "type": etype,
            "ni": ni,
            "nj": nj,
            "length_m": round(length, 4),
            "section": sec.name,
            "element_ids": member_elem_ids,
        })

    return elements_info, member_to_elements, member_info_list, next_node_id


def _get_node_by_id(nodes: list[Node3D], node_id: int) -> Node3D:
    """노드 ID로 노드 객체 반환."""
    for n in nodes:
        if n.id == node_id:
            return n
    raise ValueError(f"Node {node_id} not found")


# ============================================================
# 하중 적용
# ============================================================

def _apply_loads_3d(
    loads: list[dict],
    n_stories: int,
    n_cols_x: int,
    n_cols_y: int,
    node_grid: dict,
    connections: list[tuple[int, int, str]],
    member_to_elements: dict[int, list[int]],
    bays_x: list[float],
    bays_y: list[float],
):
    """3D 하중 적용."""
    ops.timeSeries('Linear', 1)
    ops.pattern('Plain', 1, 1)

    n_bays_x = n_cols_x - 1
    n_bays_y = n_cols_y - 1

    for ld in loads:
        ld_type = ld.get("type", "floor")
        story = ld.get("story", 1)

        if ld_type == "floor":
            # 보에 선하중 (kN/m) - story의 모든 보에 중력방향
            w_kNm = ld.get("value", 0.0)
            w_Nmm = w_kNm * 1000.0 / 1000.0  # kN/m → N/mm

            # 해당 story의 beam_x, beam_y 부재 찾기
            _apply_floor_load(story, w_Nmm, n_stories, n_cols_x, n_cols_y,
                              connections, member_to_elements)

        elif ld_type == "floor_area":
            # 면적하중 (kN/m²) → tributary width로 보 선하중 변환
            w_area = ld.get("value", 0.0)  # kN/m²
            _apply_floor_area_load(story, w_area, n_stories, n_cols_x, n_cols_y,
                                   connections, member_to_elements, bays_x, bays_y)

        elif ld_type == "lateral_x":
            # X방향 횡하중 (kN)
            fx_kN = ld.get("value", ld.get("fx", 0.0))
            fx_N = fx_kN * 1000.0
            # story 전체 노드에 균등 분배
            story_nodes = [node_grid[(story, cx, cy)]
                           for cy in range(n_cols_y) for cx in range(n_cols_x)]
            fx_per_node = fx_N / len(story_nodes)
            for nid in story_nodes:
                ops.load(nid, fx_per_node, 0.0, 0.0, 0.0, 0.0, 0.0)

        elif ld_type == "lateral_y":
            # Y방향 횡하중 (kN)
            fy_kN = ld.get("value", ld.get("fy", 0.0))
            fy_N = fy_kN * 1000.0
            story_nodes = [node_grid[(story, cx, cy)]
                           for cy in range(n_cols_y) for cx in range(n_cols_x)]
            fy_per_node = fy_N / len(story_nodes)
            for nid in story_nodes:
                ops.load(nid, 0.0, fy_per_node, 0.0, 0.0, 0.0, 0.0)

        elif ld_type == "nodal":
            # 절점하중 (6-DOF)
            nid = ld.get("node", 1)
            fx = ld.get("fx", 0.0) * 1000.0
            fy = ld.get("fy", 0.0) * 1000.0
            fz = ld.get("fz", 0.0) * 1000.0  # 양수=위, 음수=아래
            mx = ld.get("mx", 0.0) * 1e6
            my = ld.get("my", 0.0) * 1e6
            mz = ld.get("mz", 0.0) * 1e6
            ops.load(nid, fx, fy, fz, mx, my, mz)


def _apply_floor_load(story, w_Nmm, n_stories, n_cols_x, n_cols_y,
                       connections, member_to_elements):
    """보에 선하중 적용 (중력방향 하향)."""
    # 기둥 수 = n_stories * n_cols_x * n_cols_y
    n_cols_total = n_cols_x * n_cols_y
    # beam_x 시작 member_id 계산
    # connections 순서: 기둥 → beam_x → beam_y
    col_count = n_stories * n_cols_total
    beam_x_per_story = (n_cols_x - 1) * n_cols_y
    beam_y_per_story = n_cols_x * (n_cols_y - 1)

    # story에 해당하는 beam_x 부재
    beam_x_start = col_count + (story - 1) * beam_x_per_story
    beam_y_start = col_count + n_stories * beam_x_per_story + (story - 1) * beam_y_per_story

    for i in range(beam_x_per_story):
        mid = beam_x_start + i + 1  # 1-based member_id
        if mid in member_to_elements:
            for eid in member_to_elements[mid]:
                # beam_x: 로컬 z ~ global Z → wz = -w (하향)
                ops.eleLoad('-ele', eid, '-type', '-beamUniform', 0.0, -w_Nmm, 0.0)

    for i in range(beam_y_per_story):
        mid = beam_y_start + i + 1
        if mid in member_to_elements:
            for eid in member_to_elements[mid]:
                # beam_y: 로컬 z ~ global Z → wz = -w (하향)
                ops.eleLoad('-ele', eid, '-type', '-beamUniform', 0.0, -w_Nmm, 0.0)


def _apply_floor_area_load(story, w_area_kNm2, n_stories, n_cols_x, n_cols_y,
                            connections, member_to_elements, bays_x, bays_y):
    """면적하중을 tributary width로 보 선하중으로 변환 적용.

    2방향 슬래브 가정: 면적하중을 X/Y 양방향 보에 균등 분배 (각 방향 50%).
    총 반력 = w_area × floor_area 가 되도록 보장.
    """
    n_cols_total = n_cols_x * n_cols_y
    col_count = n_stories * n_cols_total
    beam_x_per_story = (n_cols_x - 1) * n_cols_y
    beam_y_per_story = n_cols_x * (n_cols_y - 1)

    # 2방향 분배: 각 방향에 50%씩
    w_area_half = w_area_kNm2 * 0.5

    # Beam_X: 각 보의 tributary width = 인접 bay_y의 평균 절반
    beam_x_start = col_count + (story - 1) * beam_x_per_story
    bx_idx = 0
    for cy in range(n_cols_y):
        # tributary width for this Y-line
        trib_y = 0.0
        if cy > 0:
            trib_y += bays_y[cy - 1] / 2.0
        if cy < n_cols_y - 1:
            trib_y += bays_y[cy] / 2.0
        if trib_y == 0.0:
            trib_y = bays_y[0] if bays_y else 1.0  # single bay_y fallback

        w_line = w_area_half * trib_y  # kN/m
        w_Nmm = w_line * 1000.0 / 1000.0

        for cx in range(n_cols_x - 1):
            mid = beam_x_start + bx_idx + 1
            if mid in member_to_elements:
                for eid in member_to_elements[mid]:
                    ops.eleLoad('-ele', eid, '-type', '-beamUniform', 0.0, -w_Nmm, 0.0)
            bx_idx += 1

    # Beam_Y: 각 보의 tributary width = 인접 bay_x의 평균 절반
    beam_y_start = col_count + n_stories * beam_x_per_story + (story - 1) * beam_y_per_story
    by_idx = 0
    for cx in range(n_cols_x):
        trib_x = 0.0
        if cx > 0:
            trib_x += bays_x[cx - 1] / 2.0
        if cx < n_cols_x - 1:
            trib_x += bays_x[cx] / 2.0
        if trib_x == 0.0:
            trib_x = bays_x[0] if bays_x else 1.0

        w_line = w_area_half * trib_x
        w_Nmm = w_line * 1000.0 / 1000.0

        for cy in range(n_cols_y - 1):
            mid = beam_y_start + by_idx + 1
            if mid in member_to_elements:
                for eid in member_to_elements[mid]:
                    ops.eleLoad('-ele', eid, '-type', '-beamUniform', 0.0, -w_Nmm, 0.0)
            by_idx += 1


def _apply_loads_3d_irregular(
    loads: list[dict],
    story_nodes: dict[int, list[int]],
    connections: list[tuple[int, int, str]],
    member_to_elements: dict[int, list[int]],
    member_metadata: list[dict],
):
    """비정형 건물 하중 적용 (story_nodes + member_metadata 기반)."""
    ops.timeSeries('Linear', 1)
    ops.pattern('Plain', 1, 1)

    for ld in loads:
        ld_type = ld.get("type", "floor")
        story = ld.get("story", 1)

        if ld_type == "floor":
            w_kNm = ld.get("value", 0.0)
            w_Nmm = w_kNm  # kN/m → N/mm (×1000/1000)
            for mid_0, meta in enumerate(member_metadata):
                mid = mid_0 + 1
                if meta["story"] != story:
                    continue
                etype = connections[mid_0][2]
                if etype not in ("beam_x", "beam_y"):
                    continue
                if mid in member_to_elements:
                    for eid in member_to_elements[mid]:
                        ops.eleLoad('-ele', eid, '-type', '-beamUniform',
                                    0.0, -w_Nmm, 0.0)

        elif ld_type == "floor_area":
            w_area = ld.get("value", 0.0)  # kN/m²
            w_area_half = w_area * 0.5
            for mid_0, meta in enumerate(member_metadata):
                mid = mid_0 + 1
                if meta["story"] != story:
                    continue
                etype = connections[mid_0][2]
                if etype not in ("beam_x", "beam_y"):
                    continue
                trib = meta.get("trib_width", 0.0)
                if trib <= 0:
                    continue
                w_line_Nmm = w_area_half * trib  # kN/m² × m × 0.5 → kN/m → N/mm
                if mid in member_to_elements:
                    for eid in member_to_elements[mid]:
                        ops.eleLoad('-ele', eid, '-type', '-beamUniform',
                                    0.0, -w_line_Nmm, 0.0)

        elif ld_type == "lateral_x":
            fx_N = ld.get("value", ld.get("fx", 0.0)) * 1000.0
            snodes = story_nodes.get(story, [])
            if snodes:
                fx_per = fx_N / len(snodes)
                for nid in snodes:
                    ops.load(nid, fx_per, 0.0, 0.0, 0.0, 0.0, 0.0)

        elif ld_type == "lateral_y":
            fy_N = ld.get("value", ld.get("fy", 0.0)) * 1000.0
            snodes = story_nodes.get(story, [])
            if snodes:
                fy_per = fy_N / len(snodes)
                for nid in snodes:
                    ops.load(nid, 0.0, fy_per, 0.0, 0.0, 0.0, 0.0)

        elif ld_type == "nodal":
            nid = ld.get("node", 1)
            fx = ld.get("fx", 0.0) * 1000.0
            fy = ld.get("fy", 0.0) * 1000.0
            fz = ld.get("fz", 0.0) * 1000.0
            mx = ld.get("mx", 0.0) * 1e6
            my = ld.get("my", 0.0) * 1e6
            mz = ld.get("mz", 0.0) * 1e6
            ops.load(nid, fx, fy, fz, mx, my, mz)


# ============================================================
# 해석
# ============================================================

def _solve(rigid_diaphragm: bool = False, geometric_nonlinearity: str = "linear") -> dict:
    """정적 해석 수행. 해석 메타데이터 dict 반환 (ok 코드 포함)."""
    ops.system('BandGen')
    ops.numberer('Plain')
    if rigid_diaphragm:
        ops.constraints('Transformation')
    else:
        ops.constraints('Plain')
    solver_meta = {
        "requested_mode": geometric_nonlinearity,
        "actual_transf": "Corotational" if geometric_nonlinearity == "pdelta" else "Linear",
        "algorithm": "Linear",
        "fallback_used": False,
        "n_steps": 1,
        "ok": 0,
    }
    if geometric_nonlinearity == "pdelta":
        # P-Delta: Newton 알고리즘 + 다단계 하중 적용
        n_steps = 10
        ops.test('NormDispIncr', 1.0e-8, 50)
        ops.integrator('LoadControl', 1.0 / n_steps)
        ops.algorithm('Newton')
        ops.analysis('Static')
        solver_meta["algorithm"] = "Newton"
        solver_meta["n_steps"] = n_steps
        ok = ops.analyze(n_steps)
        if ok != 0:
            # 수렴 실패 시 더 작은 스텝으로 재시도
            ops.test('NormDispIncr', 1.0e-6, 100)
            ops.algorithm('ModifiedNewton')
            n_sub = 50
            ops.integrator('LoadControl', 1.0 / n_sub)
            ok = ops.analyze(n_sub)
            solver_meta["fallback_used"] = True
            solver_meta["algorithm"] = "ModifiedNewton"
            solver_meta["n_steps"] = n_sub
    else:
        ops.integrator('LoadControl', 1.0)
        ops.algorithm('Linear')
        ops.analysis('Static')
        ok = ops.analyze(1)
    ops.reactions()
    solver_meta["ok"] = ok
    return solver_meta


# ============================================================
# 결과 추출
# ============================================================

def _extract_case_results_3d(
    nodes: list[Node3D],
    elements_info: list[Element3D],
    base_nodes: list[int],
    stories: list[float],
    node_grid: dict,
    n_cols_x: int,
    n_cols_y: int,
    supports: str,
    story_nodes_map: dict[int, list[int]] | None = None,
) -> Frame3DCaseResult:
    """현재 OpenSees 상태에서 결과 추출.

    story_nodes_map이 제공되면 (비정형) 그것을 사용, 없으면 node_grid에서 생성.
    """
    result = Frame3DCaseResult()
    n_stories = len(stories)

    # 1. 노드 변위 (6-DOF)
    for node in nodes:
        dx = ops.nodeDisp(node.id, 1)  # mm
        dy = ops.nodeDisp(node.id, 2)
        dz = ops.nodeDisp(node.id, 3)
        rx = ops.nodeDisp(node.id, 4)  # rad
        ry = ops.nodeDisp(node.id, 5)
        rz = ops.nodeDisp(node.id, 6)

        result.nodal_displacements.append({
            "node": node.id,
            "x_m": round(node.x, 4),
            "y_m": round(node.y, 4),
            "z_m": round(node.z, 4),
            "dx_mm": round(dx, 4),
            "dy_mm": round(dy, 4),
            "dz_mm": round(dz, 4),
            "rx_rad": round(rx, 6),
            "ry_rad": round(ry, 6),
            "rz_rad": round(rz, 6),
        })

        # 최대값 갱신
        if abs(dx) > abs(result.max_displacement_x):
            result.max_displacement_x = round(dx, 4)
            result.max_displacement_x_node = node.id
        if abs(dy) > abs(result.max_displacement_y):
            result.max_displacement_y = round(dy, 4)
            result.max_displacement_y_node = node.id
        if abs(dz) > abs(result.max_displacement_z):
            result.max_displacement_z = round(dz, 4)
            result.max_displacement_z_node = node.id

    # 2. 요소력 (12성분)
    for elem in elements_info:
        forces = ops.eleForce(elem.id)
        N_i = forces[0] / 1000       # N → kN
        Vy_i = forces[1] / 1000
        Vz_i = forces[2] / 1000
        T_i = forces[3] / 1e6        # N·mm → kN·m
        My_i = forces[4] / 1e6
        Mz_i = forces[5] / 1e6
        N_j = forces[6] / 1000
        Vy_j = forces[7] / 1000
        Vz_j = forces[8] / 1000
        T_j = forces[9] / 1e6
        My_j = forces[10] / 1e6
        Mz_j = forces[11] / 1e6

        result.element_forces.append({
            "element": elem.id,
            "type": elem.elem_type,
            "section": elem.section_name,
            "ni": elem.ni, "nj": elem.nj,
            "N_i_kN": round(N_i, 2), "Vy_i_kN": round(Vy_i, 2),
            "Vz_i_kN": round(Vz_i, 2), "T_i_kNm": round(T_i, 2),
            "My_i_kNm": round(My_i, 2), "Mz_i_kNm": round(Mz_i, 2),
            "N_j_kN": round(N_j, 2), "Vy_j_kN": round(Vy_j, 2),
            "Vz_j_kN": round(Vz_j, 2), "T_j_kNm": round(T_j, 2),
            "My_j_kNm": round(My_j, 2), "Mz_j_kNm": round(Mz_j, 2),
        })

        # 최대값 갱신
        max_M = max(abs(My_i), abs(My_j), abs(Mz_i), abs(Mz_j))
        if max_M > abs(result.max_moment):
            result.max_moment = round(max_M, 2)
            result.max_moment_element = elem.id
        max_N = max(abs(N_i), abs(N_j))
        if max_N > abs(result.max_axial):
            result.max_axial = round(max_N, 2)
            result.max_axial_element = elem.id
        max_V = max(abs(Vy_i), abs(Vy_j), abs(Vz_i), abs(Vz_j))
        if max_V > abs(result.max_shear):
            result.max_shear = round(max_V, 2)
            result.max_shear_element = elem.id
        max_T = max(abs(T_i), abs(T_j))
        if max_T > abs(result.max_torsion):
            result.max_torsion = round(max_T, 2)
            result.max_torsion_element = elem.id

    # 3. 반력 (6성분)
    for bn in base_nodes:
        rx_val = ops.nodeReaction(bn, 1) / 1000  # N → kN
        ry_val = ops.nodeReaction(bn, 2) / 1000
        rz_val = ops.nodeReaction(bn, 3) / 1000
        mx_val = ops.nodeReaction(bn, 4) / 1e6   # N·mm → kN·m
        my_val = ops.nodeReaction(bn, 5) / 1e6
        mz_val = ops.nodeReaction(bn, 6) / 1e6

        bn_node = _get_node_by_id(nodes, bn)
        result.reactions.append({
            "node": bn,
            "x_m": round(bn_node.x, 4),
            "y_m": round(bn_node.y, 4),
            "RX_kN": round(rx_val, 2),
            "RY_kN": round(ry_val, 2),
            "RZ_kN": round(rz_val, 2),
            "MX_kNm": round(mx_val, 2),
            "MY_kNm": round(my_val, 2),
            "MZ_kNm": round(mz_val, 2),
        })

    # 4. 층간변위각 (X/Y 양방향)
    # story_nodes_map 사용 (비정형 지원) 또는 node_grid에서 생성
    if story_nodes_map is None:
        story_nodes_map = _get_story_nodes_from_grid(node_grid, n_cols_x, n_cols_y, n_stories)

    for s in range(1, n_stories + 1):
        story_height_mm = stories[s - 1] * 1000

        lower_nodes = story_nodes_map.get(s - 1, [])
        upper_nodes = story_nodes_map.get(s, [])

        if not lower_nodes or not upper_nodes:
            continue

        lower_dx = sum(ops.nodeDisp(n, 1) for n in lower_nodes) / len(lower_nodes)
        upper_dx = sum(ops.nodeDisp(n, 1) for n in upper_nodes) / len(upper_nodes)
        drift_x = (upper_dx - lower_dx) / story_height_mm if story_height_mm > 0 else 0

        lower_dy = sum(ops.nodeDisp(n, 2) for n in lower_nodes) / len(lower_nodes)
        upper_dy = sum(ops.nodeDisp(n, 2) for n in upper_nodes) / len(upper_nodes)
        drift_y = (upper_dy - lower_dy) / story_height_mm if story_height_mm > 0 else 0

        drift_r = math.sqrt(drift_x ** 2 + drift_y ** 2)

        result.story_drifts.append({
            "story": s,
            "height_m": stories[s - 1],
            "drift_x": round(abs(drift_x), 6),
            "drift_y": round(abs(drift_y), 6),
            "drift_resultant": round(drift_r, 6),
        })

        if abs(drift_x) > abs(result.max_drift_x):
            result.max_drift_x = round(abs(drift_x), 6)
            result.max_drift_x_story = s
        if abs(drift_y) > abs(result.max_drift_y):
            result.max_drift_y = round(abs(drift_y), 6)
            result.max_drift_y_story = s

    return result


# ============================================================
# 부재력 다이어그램
# ============================================================

def _extract_member_forces_3d(
    member_info_list: list[dict],
    member_to_elements: dict[int, list[int]],
    num_elements_per_member: int,
) -> list[dict]:
    """부재별 6성분 내력 다이어그램 추출."""
    member_forces = []

    for minfo in member_info_list:
        mid = minfo["member_id"]
        elem_ids = member_to_elements.get(mid, [])
        if not elem_ids:
            continue

        length = minfo["length_m"]
        n_sub = len(elem_ids)
        sub_len = length / n_sub if n_sub > 0 else length

        s_vals = []
        N_vals, Vy_vals, Vz_vals, T_vals, My_vals, Mz_vals = [], [], [], [], [], []

        for k, eid in enumerate(elem_ids):
            forces = ops.eleForce(eid)
            s_start = k * sub_len

            if k == 0:
                s_vals.append(round(s_start, 4))
                N_vals.append(round(forces[0] / 1000, 4))
                Vy_vals.append(round(forces[1] / 1000, 4))
                Vz_vals.append(round(forces[2] / 1000, 4))
                T_vals.append(round(forces[3] / 1e6, 4))
                My_vals.append(round(forces[4] / 1e6, 4))
                Mz_vals.append(round(forces[5] / 1e6, 4))

            # j-end: 내력 = -반력
            s_end = (k + 1) * sub_len
            s_vals.append(round(s_end, 4))
            N_vals.append(round(-forces[6] / 1000, 4))
            Vy_vals.append(round(-forces[7] / 1000, 4))
            Vz_vals.append(round(-forces[8] / 1000, 4))
            T_vals.append(round(-forces[9] / 1e6, 4))
            My_vals.append(round(-forces[10] / 1e6, 4))
            Mz_vals.append(round(-forces[11] / 1e6, 4))

        member_forces.append({
            "member_id": mid,
            "type": minfo["type"],
            "ni": minfo["ni"],
            "nj": minfo["nj"],
            "length_m": length,
            "s": s_vals,
            "N_kN": N_vals,
            "Vy_kN": Vy_vals,
            "Vz_kN": Vz_vals,
            "T_kNm": T_vals,
            "My_kNm": My_vals,
            "Mz_kNm": Mz_vals,
        })

    return member_forces


# ============================================================
# 하중조합 (선형 중첩)
# ============================================================

def _superpose_case_results_3d(
    case_results: dict[str, Frame3DCaseResult],
    factors: dict[str, float],
    nodes: list[Node3D],
    elements_info: list[Element3D],
    base_nodes: list[int],
    stories: list[float],
    node_grid: dict,
    n_cols_x: int,
    n_cols_y: int,
    story_nodes_map: dict[int, list[int]] | None = None,
) -> Frame3DCaseResult:
    """하중조합 결과를 선형 중첩으로 생성."""
    combo = Frame3DCaseResult()
    n_stories = len(stories)

    # 노드 변위 중첩
    node_disp_map = {}
    for cn, factor in factors.items():
        if cn not in case_results:
            continue
        for nd in case_results[cn].nodal_displacements:
            nid = nd["node"]
            if nid not in node_disp_map:
                node_disp_map[nid] = {
                    "node": nid, "x_m": nd["x_m"], "y_m": nd["y_m"], "z_m": nd["z_m"],
                    "dx_mm": 0, "dy_mm": 0, "dz_mm": 0,
                    "rx_rad": 0, "ry_rad": 0, "rz_rad": 0,
                }
            for key in ("dx_mm", "dy_mm", "dz_mm", "rx_rad", "ry_rad", "rz_rad"):
                node_disp_map[nid][key] += nd[key] * factor

    for nid, nd in node_disp_map.items():
        for key in ("dx_mm", "dy_mm", "dz_mm"):
            nd[key] = round(nd[key], 4)
        for key in ("rx_rad", "ry_rad", "rz_rad"):
            nd[key] = round(nd[key], 6)
        combo.nodal_displacements.append(nd)

        if abs(nd["dx_mm"]) > abs(combo.max_displacement_x):
            combo.max_displacement_x = nd["dx_mm"]
            combo.max_displacement_x_node = nid
        if abs(nd["dy_mm"]) > abs(combo.max_displacement_y):
            combo.max_displacement_y = nd["dy_mm"]
            combo.max_displacement_y_node = nid
        if abs(nd["dz_mm"]) > abs(combo.max_displacement_z):
            combo.max_displacement_z = nd["dz_mm"]
            combo.max_displacement_z_node = nid

    # 요소력 중첩
    elem_force_map = {}
    force_keys = [
        "N_i_kN", "Vy_i_kN", "Vz_i_kN", "T_i_kNm", "My_i_kNm", "Mz_i_kNm",
        "N_j_kN", "Vy_j_kN", "Vz_j_kN", "T_j_kNm", "My_j_kNm", "Mz_j_kNm",
    ]
    for cn, factor in factors.items():
        if cn not in case_results:
            continue
        for ef in case_results[cn].element_forces:
            eid = ef["element"]
            if eid not in elem_force_map:
                elem_force_map[eid] = {
                    "element": eid, "type": ef["type"], "section": ef["section"],
                    "ni": ef["ni"], "nj": ef["nj"],
                }
                for fk in force_keys:
                    elem_force_map[eid][fk] = 0.0
            for fk in force_keys:
                elem_force_map[eid][fk] += ef[fk] * factor

    for eid, ef in elem_force_map.items():
        for fk in force_keys:
            ef[fk] = round(ef[fk], 2)
        combo.element_forces.append(ef)

        max_M = max(abs(ef.get("My_i_kNm", 0)), abs(ef.get("My_j_kNm", 0)),
                     abs(ef.get("Mz_i_kNm", 0)), abs(ef.get("Mz_j_kNm", 0)))
        if max_M > abs(combo.max_moment):
            combo.max_moment = round(max_M, 2)
            combo.max_moment_element = eid

        max_N = max(abs(ef.get("N_i_kN", 0)), abs(ef.get("N_j_kN", 0)))
        if max_N > abs(combo.max_axial):
            combo.max_axial = round(max_N, 2)
            combo.max_axial_element = eid

        max_V = max(abs(ef.get("Vy_i_kN", 0)), abs(ef.get("Vy_j_kN", 0)),
                     abs(ef.get("Vz_i_kN", 0)), abs(ef.get("Vz_j_kN", 0)))
        if max_V > abs(combo.max_shear):
            combo.max_shear = round(max_V, 2)
            combo.max_shear_element = eid

        max_T = max(abs(ef.get("T_i_kNm", 0)), abs(ef.get("T_j_kNm", 0)))
        if max_T > abs(combo.max_torsion):
            combo.max_torsion = round(max_T, 2)
            combo.max_torsion_element = eid

    # 반력 중첩
    rxn_map = {}
    rxn_keys = ["RX_kN", "RY_kN", "RZ_kN", "MX_kNm", "MY_kNm", "MZ_kNm"]
    for cn, factor in factors.items():
        if cn not in case_results:
            continue
        for r in case_results[cn].reactions:
            nid = r["node"]
            if nid not in rxn_map:
                rxn_map[nid] = {"node": nid, "x_m": r["x_m"], "y_m": r["y_m"]}
                for rk in rxn_keys:
                    rxn_map[nid][rk] = 0.0
            for rk in rxn_keys:
                rxn_map[nid][rk] += r[rk] * factor

    for nid, r in rxn_map.items():
        for rk in rxn_keys:
            r[rk] = round(r[rk], 2)
        combo.reactions.append(r)

    # 층간변위각 재계산
    if story_nodes_map is None:
        story_nodes_map = _get_story_nodes_from_grid(node_grid, n_cols_x, n_cols_y, n_stories)

    for s in range(1, n_stories + 1):
        story_height_mm = stories[s - 1] * 1000
        lower_nids = story_nodes_map.get(s - 1, [])
        upper_nids = story_nodes_map.get(s, [])

        if not lower_nids or not upper_nids:
            continue

        lower_dx = sum(node_disp_map.get(n, {}).get("dx_mm", 0) for n in lower_nids) / len(lower_nids)
        upper_dx = sum(node_disp_map.get(n, {}).get("dx_mm", 0) for n in upper_nids) / len(upper_nids)
        drift_x = abs(upper_dx - lower_dx) / story_height_mm if story_height_mm > 0 else 0

        lower_dy = sum(node_disp_map.get(n, {}).get("dy_mm", 0) for n in lower_nids) / len(lower_nids)
        upper_dy = sum(node_disp_map.get(n, {}).get("dy_mm", 0) for n in upper_nids) / len(upper_nids)
        drift_y = abs(upper_dy - lower_dy) / story_height_mm if story_height_mm > 0 else 0

        combo.story_drifts.append({
            "story": s, "height_m": stories[s - 1],
            "drift_x": round(drift_x, 6), "drift_y": round(drift_y, 6),
            "drift_resultant": round(math.sqrt(drift_x ** 2 + drift_y ** 2), 6),
        })

        if drift_x > abs(combo.max_drift_x):
            combo.max_drift_x = round(drift_x, 6)
            combo.max_drift_x_story = s
        if drift_y > abs(combo.max_drift_y):
            combo.max_drift_y = round(drift_y, 6)
            combo.max_drift_y_story = s

    return combo


def _superpose_member_forces_3d(
    all_member_forces: dict[str, list[dict]],
    factors: dict[str, float],
) -> list[dict]:
    """부재력 다이어그램 선형 중첩."""
    combined = {}
    force_arrays = ["N_kN", "Vy_kN", "Vz_kN", "T_kNm", "My_kNm", "Mz_kNm"]

    for cn, factor in factors.items():
        if cn not in all_member_forces:
            continue
        for mf in all_member_forces[cn]:
            mid = mf["member_id"]
            if mid not in combined:
                combined[mid] = {
                    "member_id": mid, "type": mf["type"],
                    "ni": mf["ni"], "nj": mf["nj"],
                    "length_m": mf["length_m"], "s": mf["s"],
                }
                for fa in force_arrays:
                    combined[mid][fa] = [0.0] * len(mf["s"])

            for fa in force_arrays:
                for i, val in enumerate(mf[fa]):
                    combined[mid][fa][i] += val * factor

    result = []
    for mid in sorted(combined.keys()):
        mf = combined[mid]
        for fa in force_arrays:
            mf[fa] = [round(v, 4) for v in mf[fa]]
        result.append(mf)

    return result


# ============================================================
# 고유치해석 (Eigenvalue Analysis)
# ============================================================

def _estimate_story_weights(
    load_cases: dict, num_stories: int,
    bays_x: list[float], bays_y: list[float],
    n_cols_x: int, n_cols_y: int,
) -> list[float]:
    """DL 하중케이스에서 층별 중력하중(kN) 추정.

    Returns:
        story_weights_kN (길이 = num_stories). 빈 리스트 = 추정 불가.
    """
    dl_loads = None
    for case_name, loads in load_cases.items():
        if case_name.upper() in ("DL", "DEAD", "D"):
            dl_loads = loads
            break

    if not dl_loads:
        return []

    floor_area = sum(bays_x) * sum(bays_y)  # m²
    total_beam_len = sum(bays_x) * n_cols_y + sum(bays_y) * n_cols_x  # m

    weights = [0.0] * num_stories
    for ld in dl_loads:
        ld_type = ld.get("type", "")
        story = ld.get("story", 0)
        if story < 1 or story > num_stories:
            continue
        idx = story - 1

        if ld_type == "floor_area":
            weights[idx] += ld.get("value", 0.0) * floor_area
        elif ld_type == "floor":
            weights[idx] += ld.get("value", 0.0) * total_beam_len

    return weights if any(w > 0 for w in weights) else []


def _run_eigen_analysis(
    nodes: list,
    connections: list,
    base_nodes: list[int],
    supports: str,
    col_sec, bx_sec, by_sec,
    E: float, G: float,
    node_grid: dict,
    n_cols_x: int, n_cols_y: int,
    num_stories: int,
    stories: list[float],
    story_weights_kN: list[float],
    bays_x: list[float] | None = None,
    bays_y: list[float] | None = None,
    num_modes: int = 0,
    geometric_nonlinearity: str = "linear",
    member_releases: dict | None = None,
    story_nodes_map: dict[int, list[int]] | None = None,
) -> dict:
    """고유치해석 수행.

    정적해석과 별도로 모델을 재구축 (num_elements_per_member=1, 질량 배정).
    rigid_diaphragm은 항상 True (모달해석 전제조건).

    Args:
        story_weights_kN: 층별 중력하중 (kN), 길이 = num_stories

    Returns:
        {"num_modes": int, "modes": [...], "fundamental_periods": {...}}
    """
    if not story_weights_kN or len(story_weights_kN) != num_stories:
        return {}

    # story_nodes_map 생성 (없으면)
    is_irreg = story_nodes_map is not None
    if not is_irreg and node_grid:
        story_nodes_map = _get_story_nodes_from_grid(node_grid, n_cols_x, n_cols_y, num_stories)

    # 1. 모델 재구축 (num_elements_per_member=1 → 유령모드 방지)
    _build_frame_3d_model(
        nodes, connections, base_nodes, supports,
        col_sec, bx_sec, by_sec, E, G,
        num_elements_per_member=1,
        rigid_diaphragm=True,
        node_grid=node_grid if not is_irreg else None,
        n_cols_x=n_cols_x,
        n_cols_y=n_cols_y,
        num_stories=num_stories,
        member_releases=member_releases,
        geometric_nonlinearity=geometric_nonlinearity,
        story_nodes_map=story_nodes_map if is_irreg else None,
    )

    # 2. 분산 질량 배정 + 층별 질량/회전관성 기록
    g_acc = 9810.0  # mm/s²
    node_map = {n.id: n for n in nodes}

    # 마스터 절점 결정
    master_nodes = {}
    floor_masses = []

    for s in range(1, num_stories + 1):
        snodes = story_nodes_map.get(s, [])
        if not snodes:
            continue
        nodes_per_floor = len(snodes)

        # 마스터 = 중앙 노드
        master_nid = snodes[len(snodes) // 2]
        master_nodes[s] = master_nid
        mx_mm = node_map[master_nid].x * 1000
        my_mm = node_map[master_nid].y * 1000

        W_N = story_weights_kN[s - 1] * 1000.0
        m_per_node = W_N / g_acc / nodes_per_floor
        m_floor = W_N / g_acc

        # 실효 회전관성
        I_eff = 0.0
        for nid in snodes:
            ops.mass(nid, m_per_node, m_per_node, 1e-6, 0.0, 0.0, 0.0)
            dx = node_map[nid].x * 1000 - mx_mm
            dy = node_map[nid].y * 1000 - my_mm
            I_eff += m_per_node * (dx ** 2 + dy ** 2)

        floor_masses.append((m_floor, I_eff))

    total_weight_kN = sum(story_weights_kN)

    # 3. 모드 수 결정
    if num_modes <= 0:
        num_modes = min(3 * num_stories, 15)

    # 4. 고유치 풀이 (fallback 체인)
    eigenvalues = None
    try:
        eigenvalues = ops.eigen(num_modes)
    except Exception:
        try:
            eigenvalues = ops.eigen('-genBandArpack', num_modes)
        except Exception:
            try:
                eigenvalues = ops.eigen('-fullGenLapack', num_modes)
            except Exception:
                return {}

    if not eigenvalues:
        return {}

    # 5. 총 질량 (참여질량 비율 계산 기준)
    total_mass_x = sum(fm[0] for fm in floor_masses)
    total_mass_y = total_mass_x  # 동일 (병진질량)
    total_mass_rz = sum(fm[1] for fm in floor_masses)

    # 6. 모드별 결과 추출 + 참여질량 계산
    modes = []
    T1_x, T1_y, T1_rz = None, None, None
    cum_x, cum_y, cum_rz = 0.0, 0.0, 0.0

    for i, lam in enumerate(eigenvalues):
        mode_num = i + 1
        if lam <= 0:
            continue

        omega = math.sqrt(lam)
        T = 2.0 * math.pi / omega
        f = 1.0 / T

        # 모드 형상 추출 (마스터 절점)
        phi = []
        for s in range(1, num_stories + 1):
            nid = master_nodes[s]
            ux = ops.nodeEigenvector(nid, mode_num, 1)
            uy = ops.nodeEigenvector(nid, mode_num, 2)
            rz = ops.nodeEigenvector(nid, mode_num, 6)
            phi.append((ux, uy, rz))

        # 일반화 질량: φ^T M φ
        gen_mass = 0.0
        for s_idx in range(num_stories):
            m_t, i_r = floor_masses[s_idx]
            ux, uy, rz = phi[s_idx]
            gen_mass += m_t * ux ** 2 + m_t * uy ** 2 + i_r * rz ** 2

        # 방향별 참여질량
        mp = {}
        for dir_name, dof_idx, total_m in [
            ("x", 0, total_mass_x),
            ("y", 1, total_mass_y),
            ("rz", 2, total_mass_rz),
        ]:
            L = 0.0
            for s_idx in range(num_stories):
                m_t, i_r = floor_masses[s_idx]
                if dof_idx < 2:
                    L += m_t * phi[s_idx][dof_idx]
                else:
                    L += i_r * phi[s_idx][2]
            m_eff = L ** 2 / gen_mass if gen_mass > 1e-30 else 0.0
            pct = m_eff / total_m * 100 if total_m > 1e-30 else 0.0
            mp[f"{dir_name}_pct"] = round(pct, 2)

        # 누적 참여질량
        cum_x += mp["x_pct"]
        cum_y += mp["y_pct"]
        cum_rz += mp["rz_pct"]

        # 지배 방향 판별 (참여질량 기반으로 개선)
        px, py, prz = mp["x_pct"], mp["y_pct"], mp["rz_pct"]
        if px >= py and px >= prz:
            direction, dominance = "TRAN-X", px
        elif py >= prz:
            direction, dominance = "TRAN-Y", py
        elif prz > 0:
            direction, dominance = "ROTN-Z", prz
        else:
            direction, dominance = "N/A", 0.0

        # 전체 노드 모드형상 추출 (3D 시각화용)
        mode_shape = {}
        all_nids = set()
        for s_nids in story_nodes_map.values():
            all_nids.update(s_nids)
        for nid in all_nids:
            try:
                ux_e = ops.nodeEigenvector(nid, mode_num, 1)
                uy_e = ops.nodeEigenvector(nid, mode_num, 2)
                uz_e = ops.nodeEigenvector(nid, mode_num, 3)
            except Exception:
                ux_e = uy_e = uz_e = 0.0
            mode_shape[nid] = [round(ux_e, 6), round(uy_e, 6), round(uz_e, 6)]

        # 정규화: 최대 변위 = 1.0
        max_disp = max((math.sqrt(v[0]**2 + v[1]**2 + v[2]**2) for v in mode_shape.values()), default=1.0)
        if max_disp > 1e-12:
            for nid in mode_shape:
                mode_shape[nid] = [round(c / max_disp, 6) for c in mode_shape[nid]]

        modes.append({
            "mode": mode_num,
            "period_s": round(T, 4),
            "frequency_hz": round(f, 4),
            "direction": direction,
            "dominance_pct": round(dominance, 2),
            "mass_participation": mp,
            "shape": mode_shape,
        })

        # 방향별 1차 고유주기 기록
        if direction == "TRAN-X" and T1_x is None:
            T1_x = round(T, 4)
        elif direction == "TRAN-Y" and T1_y is None:
            T1_y = round(T, 4)
        elif direction == "ROTN-Z" and T1_rz is None:
            T1_rz = round(T, 4)

    # 7. 누적 참여질량 충분조건 (≥90%)
    sufficient = cum_x >= 90.0 and cum_y >= 90.0

    return {
        "num_modes": len(modes),
        "modes": modes,
        "fundamental_periods": {
            "T1_x_s": T1_x or 0.0,
            "T1_y_s": T1_y or 0.0,
            "T1_rz_s": T1_rz or 0.0,
        },
        "cumulative_participation": {
            "x_pct": round(cum_x, 1),
            "y_pct": round(cum_y, 1),
            "rz_pct": round(cum_rz, 1),
            "sufficient_90pct": sufficient,
        },
        "mass_info": {
            "basis": "floor_dead_load",
            "method": "lumped_distributed",
            "includes_member_self_weight": False,
            "rigid_diaphragm": True,
            "total_weight_kN": round(total_weight_kN, 1),
            "notes": [
                "Floor-load-based lumped mass used",
                "Member self-weight not included separately",
                "Rigid diaphragm applied (Z-perpendicular)",
            ],
        },
    }


# ============================================================
# 메인 해석 함수
# ============================================================

def analyze_frame_3d_multi(
    stories: list[float],
    bays_x: list[float],
    bays_y: list[float],
    load_cases: dict[str, list[dict]],
    supports: Literal["fixed", "pinned"] = "fixed",
    column_section: str = "H-300x300",
    beam_x_section: str = "H-400x200",
    beam_y_section: str = "H-400x200",
    material_name: str = "SS275",
    num_elements_per_member: int = 4,
    load_combinations: Optional[dict[str, dict[str, float]]] = None,
    rigid_diaphragm: bool = False,
    member_releases: dict | None = None,
    geometric_nonlinearity: str = "linear",
    modal_analysis: bool = False,
    story_weights_kN: list[float] | None = None,
    zones: list[dict] | None = None,
) -> Frame3DMultiCaseResult:
    """3D 골조 멀티 하중케이스 정적 해석.

    Args:
        stories: 각 층 높이 (m), 아래→위
        bays_x: X방향 경간 폭 (m)
        bays_y: Y방향 경간 폭 (m)
        load_cases: 하중케이스 {"DL": [...], "EQX": [...]}
        supports: "fixed" or "pinned"
        column_section: 기둥 단면명
        beam_x_section: X방향 보 단면명
        beam_y_section: Y방향 보 단면명
        material_name: 재료명
        num_elements_per_member: 부재당 서브요소 수
        load_combinations: 하중조합 {"1.2DL+1.0EQX": {"DL": 1.2, "EQX": 1.0}}
        geometric_nonlinearity: "linear" (기본) 또는 "pdelta" (P-Delta 기하비선형)
        modal_analysis: True → 정적해석 후 고유치해석 수행 (rigid_diaphragm 자동 활성화)
        story_weights_kN: 층별 중력하중 (kN). None이면 DL 하중에서 자동 추정.
    """
    # 비정형 여부 판별
    is_irregular = bool(zones)

    # 입력 검증
    if not (1 <= len(stories) <= 20):
        raise ValueError(f"stories: 1~20층 지원 (입력: {len(stories)}층)")
    if not is_irregular:
        if not (1 <= len(bays_x) <= 10):
            raise ValueError(f"bays_x: 1~10경간 지원 (입력: {len(bays_x)}경간)")
        if not (1 <= len(bays_y) <= 10):
            raise ValueError(f"bays_y: 1~10경간 지원 (입력: {len(bays_y)}경간)")

    # 결과 컨테이너
    multi = Frame3DMultiCaseResult()
    multi.num_stories = len(stories)
    multi.num_bays_x = len(bays_x)
    multi.num_bays_y = len(bays_y)
    multi.stories = stories
    multi.bays_x = bays_x
    multi.bays_y = bays_y
    multi.total_height = sum(stories)
    multi.total_width_x = sum(bays_x)
    multi.total_width_y = sum(bays_y)
    multi.supports = supports
    multi.num_elements_per_member = num_elements_per_member
    multi.rigid_diaphragm = rigid_diaphragm
    multi.column_section = column_section
    multi.beam_x_section = beam_x_section
    multi.beam_y_section = beam_y_section
    multi.material_name = material_name
    multi.load_cases = load_cases
    multi.load_combinations = load_combinations or {}
    multi.member_releases = member_releases or {}
    multi.geometric_nonlinearity = geometric_nonlinearity
    multi.analysis_metadata = {
        "dimension": "3D",
        "requested_mode": geometric_nonlinearity,
        "actual_transf": "Corotational" if geometric_nonlinearity == "pdelta" else "Linear",
        "solver_algorithm": "Linear",
        "fallback_used": False,
        "n_steps": 1,
    }

    # 단면 조회
    col_sec = get_section_3d(column_section) or DEFAULT_SECTIONS_3D.get("H-300x300")
    bx_sec = get_section_3d(beam_x_section) or DEFAULT_SECTIONS_3D.get("H-400x200")
    by_sec = get_section_3d(beam_y_section) or DEFAULT_SECTIONS_3D.get("H-400x200")

    if not col_sec or not bx_sec or not by_sec:
        raise ValueError("단면을 찾을 수 없습니다.")

    # 재료 조회
    mat = get_material_from_db(material_name) or DEFAULT_MATERIALS.get("SS275")
    if not mat:
        raise ValueError(f"재료 '{material_name}'을 찾을 수 없습니다.")

    E = mat.E    # MPa = N/mm²
    G = E / (2.0 * (1.0 + 0.3))  # ν = 0.3
    multi.E_MPa = E
    multi.G_MPa = round(G, 1)
    multi.fy_MPa = mat.fy

    # 단면 물성 기록
    multi.column_A_mm2 = col_sec.A
    multi.column_Ix_mm4 = col_sec.Ix
    multi.column_Iy_mm4 = col_sec.Iy
    multi.column_J_mm4 = col_sec.J
    multi.beam_x_A_mm2 = bx_sec.A
    multi.beam_x_Ix_mm4 = bx_sec.Ix
    multi.beam_x_Iy_mm4 = bx_sec.Iy
    multi.beam_x_J_mm4 = bx_sec.J
    multi.beam_y_A_mm2 = by_sec.A
    multi.beam_y_Ix_mm4 = by_sec.Ix
    multi.beam_y_Iy_mm4 = by_sec.Iy
    multi.beam_y_J_mm4 = by_sec.J

    # 단면 치수 기록 (design check용)
    multi.column_h_mm = col_sec.h
    multi.column_b_mm = col_sec.b
    multi.column_tw_mm = col_sec.tw
    multi.column_tf_mm = col_sec.tf
    multi.beam_x_h_mm = bx_sec.h
    multi.beam_x_b_mm = bx_sec.b
    multi.beam_x_tw_mm = bx_sec.tw
    multi.beam_x_tf_mm = bx_sec.tf
    multi.beam_y_h_mm = by_sec.h
    multi.beam_y_b_mm = by_sec.b
    multi.beam_y_tw_mm = by_sec.tw
    multi.beam_y_tf_mm = by_sec.tf

    # 기하 생성
    story_nodes_map = None
    member_metadata = None

    if is_irregular:
        nodes, connections, story_nodes_map, base_nodes, member_metadata = \
            _generate_irregular_geometry(stories, zones)
        node_grid = {}  # 비정형에서는 미사용
        n_cols_x = 0
        n_cols_y = 0
        multi.analysis_metadata["irregular"] = True
        multi.analysis_metadata["num_zones"] = len(zones)
    else:
        nodes, connections, node_grid, base_nodes = _generate_frame_3d_geometry(
            stories, bays_x, bays_y
        )
        n_cols_x = len(bays_x) + 1
        n_cols_y = len(bays_y) + 1
        story_nodes_map = _get_story_nodes_from_grid(node_grid, n_cols_x, n_cols_y, len(stories))

    # 노드/요소 정보 저장
    multi.nodes = [{"id": n.id, "x_m": n.x, "y_m": n.y, "z_m": n.z} for n in nodes]

    # 하중케이스별 해석
    all_member_forces = {}

    for case_name, case_loads in load_cases.items():
        # 모델 구축 (매 케이스마다 초기화)
        elements_info, member_to_elements, member_info_list, _ = _build_frame_3d_model(
            nodes, connections, base_nodes, supports,
            col_sec, bx_sec, by_sec, E, G, num_elements_per_member,
            rigid_diaphragm=rigid_diaphragm,
            node_grid=node_grid if not is_irregular else None,
            n_cols_x=n_cols_x,
            n_cols_y=n_cols_y,
            num_stories=len(stories),
            member_releases=member_releases,
            geometric_nonlinearity=geometric_nonlinearity,
            story_nodes_map=story_nodes_map if is_irregular else None,
        )

        if case_name == list(load_cases.keys())[0]:
            multi.member_info = member_info_list
            multi.num_elements = len(elements_info)
            multi.elements = [
                {"id": e.id, "ni": e.ni, "nj": e.nj, "type": e.elem_type}
                for e in elements_info
            ]

        # 하중 적용
        if is_irregular:
            _apply_loads_3d_irregular(
                case_loads, story_nodes_map, connections,
                member_to_elements, member_metadata,
            )
        else:
            _apply_loads_3d(
                case_loads, len(stories), n_cols_x, n_cols_y,
                node_grid, connections, member_to_elements, bays_x, bays_y,
            )

        # 해석
        solver_meta = _solve(rigid_diaphragm=rigid_diaphragm, geometric_nonlinearity=geometric_nonlinearity)
        if solver_meta["ok"] != 0:
            raise RuntimeError(f"해석 실패 (case: {case_name}, rc={solver_meta['ok']})")

        # solver 메타데이터 갱신 (마지막 케이스 기준, fallback 발생 시 유지)
        if solver_meta.get("fallback_used"):
            multi.analysis_metadata["fallback_used"] = True
        multi.analysis_metadata["solver_algorithm"] = solver_meta.get("algorithm", "Linear")
        multi.analysis_metadata["n_steps"] = solver_meta.get("n_steps", 1)

        # 결과 추출
        case_result = _extract_case_results_3d(
            nodes, elements_info, base_nodes, stories,
            node_grid, n_cols_x, n_cols_y, supports,
            story_nodes_map=story_nodes_map,
        )
        multi.case_results[case_name] = case_result

        # 부재력 다이어그램
        mf = _extract_member_forces_3d(member_info_list, member_to_elements,
                                        num_elements_per_member)
        multi.member_forces[case_name] = mf
        all_member_forces[case_name] = mf

    # 하중조합
    if multi.load_combinations:
        for combo_name, combo_factors in multi.load_combinations.items():
            combo_result = _superpose_case_results_3d(
                multi.case_results, combo_factors,
                nodes, elements_info, base_nodes, stories,
                node_grid, n_cols_x, n_cols_y,
                story_nodes_map=story_nodes_map,
            )
            multi.combo_results[combo_name] = combo_result

            combo_mf = _superpose_member_forces_3d(all_member_forces, combo_factors)
            multi.member_forces[combo_name] = combo_mf

    # 고유치해석 (정적해석 완료 후)
    if modal_analysis:
        # rigid_diaphragm 자동 활성화 경고
        diaphragm_auto = False
        if not rigid_diaphragm:
            diaphragm_auto = True
            multi.analysis_metadata["W04_diaphragm_auto"] = (
                "모달해석 요청으로 rigid diaphragm이 자동 활성화되었습니다. "
                "Rigid diaphragm 없이는 모달 결과가 부정확합니다."
            )

        # 층별 중력하중 결정
        sw = story_weights_kN
        weight_source = "user_provided" if sw else None
        if not sw:
            sw = _estimate_story_weights(
                load_cases, len(stories), bays_x, bays_y, n_cols_x, n_cols_y,
            )
            weight_source = "DL_load_case" if sw else None

        if sw:
            eigen_result = _run_eigen_analysis(
                nodes, connections, base_nodes, supports,
                col_sec, bx_sec, by_sec, E, G,
                node_grid, n_cols_x, n_cols_y,
                len(stories), stories, sw,
                bays_x=bays_x, bays_y=bays_y,
                geometric_nonlinearity=geometric_nonlinearity,
                member_releases=member_releases,
                story_nodes_map=story_nodes_map if is_irregular else None,
            )
            if eigen_result:
                eigen_result["diaphragm_auto_enabled"] = diaphragm_auto
                eigen_result["story_weights_kN"] = [round(w, 1) for w in sw]
                eigen_result["weight_source"] = weight_source
                multi.modal_analysis = eigen_result
        else:
            multi.analysis_metadata["W05_no_mass"] = (
                "DL 하중케이스에서 층별 중력하중을 추정할 수 없어 고유치해석을 건너뛰었습니다. "
                "story_weights_kN 파라미터를 직접 제공해주세요."
            )

    return multi


# ============================================================
# V2 진입점: StructuralModel → 해석
# ============================================================

def _get_geom_transf_id(elem_type: str, direction_vector: tuple[float, float, float]) -> int:
    """요소 방향에 따른 기하변환 ID 결정.

    V2에서는 elem_type이 "beam"(방향 무관)일 수 있으므로,
    실제 방향벡터로 적절한 변환 ID를 결정한다.

    Returns:
        1=column(수직), 2=beam_x(X방향), 3=beam_y(Y방향)
    """
    if elem_type == "column":
        return 1

    # beam/brace: 부재 방향의 X, Y 성분으로 주방향 결정
    dx, dy, _dz = direction_vector
    if abs(dx) >= abs(dy):
        return 2  # X방향 보 (beam_x 변환)
    else:
        return 3  # Y방향 보 (beam_y 변환)


def _get_torsion_dof_v2(transf_id: int) -> int:
    """V2 기하변환 ID에 따른 비틀림 DOF."""
    return {1: 6, 2: 4, 3: 5}[transf_id]


def _release_type_to_code(release_val) -> int | None:
    """StructuralElement의 release 값 → 릴리즈 코드.

    V1 호환: 1=i, 2=j, 3=both
    V2에서는 요소별 개별 지정이므로 i/j를 분리 처리.
    """
    if release_val is None:
        return None
    # ReleaseType enum의 value 또는 문자열
    val = release_val.value if hasattr(release_val, 'value') else str(release_val)
    if val in ("moment_y", "moment_z", "moment_yz", "all"):
        return True  # 해당 끝점 릴리즈 있음
    return None


def _build_model_v2(
    model,
    E: float,
    G: float,
    section_cache: dict[str, "BeamSection3D"],
) -> tuple[list[Node3D], list[Element3D], dict[int, list[int]], list[dict], list[int], int]:
    """StructuralModel에서 OpenSees 모델 직접 구축.

    기존 _build_frame_3d_model()과 달리 요소별 개별 단면/릴리즈를 지원한다.

    Returns:
        nodes_3d, elements_info, member_to_elements,
        member_info_list, base_node_ids, next_node_id
    """
    ops.wipe()
    ops.model('basic', '-ndm', 3, '-ndf', 6)

    # 노드 생성 (m → mm)
    nodes_3d = []
    base_node_ids = []
    for n in sorted(model.nodes.values(), key=lambda n: n.id):
        ops.node(n.id, n.x * 1000, n.y * 1000, n.z * 1000)
        nodes_3d.append(Node3D(id=n.id, x=n.x, y=n.y, z=n.z))

        # 경계조건
        if n.support is not None:
            sup = n.support.value if hasattr(n.support, 'value') else str(n.support)
            if sup == "fixed":
                ops.fix(n.id, 1, 1, 1, 1, 1, 1)
                base_node_ids.append(n.id)
            elif sup == "pinned":
                ops.fix(n.id, 1, 1, 1, 0, 0, 0)
                base_node_ids.append(n.id)
            elif sup == "roller_x":
                ops.fix(n.id, 0, 1, 1, 0, 0, 0)
                base_node_ids.append(n.id)
            elif sup == "roller_y":
                ops.fix(n.id, 1, 0, 1, 0, 0, 0)
                base_node_ids.append(n.id)

    # 강체 다이어프램
    if model.rigid_diaphragm and model.story_elevations:
        for s_idx in range(1, len(model.story_elevations)):
            snodes = [n.id for n in model.nodes_at_story(s_idx)]
            if len(snodes) < 2:
                continue
            master_nid = snodes[len(snodes) // 2]
            slave_nids = [n for n in snodes if n != master_nid]
            if slave_nids:
                ops.rigidDiaphragm(3, master_nid, *slave_nids)

    # 기하변환
    transf_type = 'Corotational' if model.geometric_nonlinearity == "pdelta" else 'Linear'
    ops.geomTransf(transf_type, 1, 1.0, 0.0, 0.0)  # column
    ops.geomTransf(transf_type, 2, 0.0, 0.0, 1.0)  # beam_x
    ops.geomTransf(transf_type, 3, 0.0, 0.0, 1.0)  # beam_y

    # 요소 생성
    elements_info = []
    member_to_elements = {}
    member_info_list = []
    elem_id = 1
    next_node_id = max(n.id for n in model.nodes.values()) + 1
    num_sub = model.num_elements_per_member

    for member_id, se in enumerate(
        sorted(model.elements.values(), key=lambda e: e.id), start=1
    ):
        ni_node = model.nodes[se.node_i]
        nj_node = model.nodes[se.node_j]

        # 단면 조회
        sec = section_cache.get(se.section)
        if sec is None:
            sec = get_section_3d(se.section)
            section_cache[se.section] = sec

        # 기하변환 결정 (방향벡터 기반)
        dir_vec = se.direction_vector(model.nodes)
        transf_id = _get_geom_transf_id(se.elem_type.value, dir_vec)
        torsion_dof = _get_torsion_dof_v2(transf_id)

        os_Iy = sec.Ix
        os_Iz = sec.Iy

        # 요소별 개별 릴리즈
        actual_ni = se.node_i
        if _release_type_to_code(se.release_i):
            hinge_ni = next_node_id
            ops.node(hinge_ni, ni_node.x * 1000, ni_node.y * 1000, ni_node.z * 1000)
            ops.equalDOF(se.node_i, hinge_ni, 1, 2, 3, torsion_dof)
            actual_ni = hinge_ni
            next_node_id += 1

        actual_nj = se.node_j
        if _release_type_to_code(se.release_j):
            hinge_nj = next_node_id
            ops.node(hinge_nj, nj_node.x * 1000, nj_node.y * 1000, nj_node.z * 1000)
            ops.equalDOF(se.node_j, hinge_nj, 1, 2, 3, torsion_dof)
            actual_nj = hinge_nj
            next_node_id += 1

        member_elem_ids = []

        # V1 호환 elem_type 이름 ("beam" → "beam_x" or "beam_y")
        v1_etype = se.elem_type.value
        if v1_etype == "beam":
            v1_etype = "beam_x" if transf_id == 2 else "beam_y"
        elif v1_etype == "brace":
            v1_etype = "beam_x" if transf_id == 2 else "beam_y"

        if num_sub <= 1:
            ops.element('elasticBeamColumn', elem_id, actual_ni, actual_nj,
                        sec.A, E, G, sec.J, os_Iy, os_Iz, transf_id)
            elements_info.append(Element3D(elem_id, actual_ni, actual_nj, v1_etype, sec.name))
            member_elem_ids.append(elem_id)
            elem_id += 1
        else:
            sub_nodes = [actual_ni]
            for k in range(1, num_sub):
                ratio = k / num_sub
                sx = ni_node.x + ratio * (nj_node.x - ni_node.x)
                sy = ni_node.y + ratio * (nj_node.y - ni_node.y)
                sz = ni_node.z + ratio * (nj_node.z - ni_node.z)
                ops.node(next_node_id, sx * 1000, sy * 1000, sz * 1000)
                sub_nodes.append(next_node_id)
                next_node_id += 1
            sub_nodes.append(actual_nj)

            for k in range(num_sub):
                ops.element('elasticBeamColumn', elem_id,
                            sub_nodes[k], sub_nodes[k + 1],
                            sec.A, E, G, sec.J, os_Iy, os_Iz, transf_id)
                elements_info.append(Element3D(
                    elem_id, sub_nodes[k], sub_nodes[k + 1], v1_etype, sec.name))
                member_elem_ids.append(elem_id)
                elem_id += 1

        member_to_elements[member_id] = member_elem_ids

        length = ni_node.distance_to(nj_node) if hasattr(ni_node, 'distance_to') else math.sqrt(
            (nj_node.x - ni_node.x) ** 2
            + (nj_node.y - ni_node.y) ** 2
            + (nj_node.z - ni_node.z) ** 2
        )
        member_info_list.append({
            "member_id": member_id,
            "type": v1_etype,
            "ni": se.node_i,
            "nj": se.node_j,
            "length_m": round(length, 4),
            "section": sec.name,
            "element_ids": member_elem_ids,
        })

    return nodes_3d, elements_info, member_to_elements, member_info_list, base_node_ids, next_node_id


def _apply_loads_v2(
    loads: list[dict],
    model,
    member_to_elements: dict[int, list[int]],
    member_info_list: list[dict],
):
    """V2 하중 적용 — 노드/요소 기반.

    지원 하중 타입:
        - floor_area: kN/m² → 보 line load (tributary 기반)
        - lateral_x, lateral_y: kN → 층별 노드 분배
        - nodal: 직접 노드 하중 (6-DOF)
    """
    ops.timeSeries('Linear', 1)
    ops.pattern('Plain', 1, 1)

    for load in loads:
        ltype = load.get("type", "")
        story = load.get("story")
        value = load.get("value", 0.0)

        if ltype == "floor_area" and story is not None:
            # kN/m² → 보 line load
            # 각 보에 tributary width 기반으로 분배
            w_area = value  # kN/m²

            for mi, minfo in enumerate(member_info_list, start=1):
                if minfo["type"] not in ("beam_x", "beam_y"):
                    continue
                # 부재의 층 확인
                ni_node = model.nodes.get(minfo["ni"])
                if ni_node is None or ni_node.story != story:
                    continue

                # Tributary width 추정: 인접 보 간격의 절반
                trib_w = _estimate_tributary_width(model, minfo, mi)

                # 50% X방향, 50% Y방향 분배 (2-way slab)
                w_line = 0.5 * w_area * trib_w  # kN/m
                w_Nmm = w_line * 1000 / 1000  # kN/m → N/mm

                for eid in member_to_elements[mi]:
                    # 수직 방향 등분포하중 (로컬 z축 = 중력방향)
                    ops.eleLoad('-ele', eid, '-type', '-beamUniform',
                                0.0, -w_Nmm, 0.0)

        elif ltype in ("lateral_x", "lateral_y") and story is not None:
            # 층별 수평하중 → 해당 층 노드에 균등 분배
            story_idx = story
            snodes = [n for n in model.nodes.values() if n.story == story_idx]
            if not snodes:
                continue
            f_per_node = value / len(snodes) * 1000  # kN → N

            for n in snodes:
                if ltype == "lateral_x":
                    ops.load(n.id, f_per_node, 0.0, 0.0, 0.0, 0.0, 0.0)
                else:
                    ops.load(n.id, 0.0, f_per_node, 0.0, 0.0, 0.0, 0.0)

        elif ltype == "nodal":
            # 직접 노드 하중
            nid = load.get("node")
            if nid is None:
                continue
            fx = load.get("fx", 0.0) * 1000  # kN → N
            fy = load.get("fy", 0.0) * 1000
            fz = load.get("fz", 0.0) * 1000
            mx = load.get("mx", 0.0) * 1e6   # kN·m → N·mm
            my = load.get("my", 0.0) * 1e6
            mz = load.get("mz", 0.0) * 1e6
            ops.load(nid, fx, fy, fz, mx, my, mz)


def _estimate_tributary_width(model, minfo: dict, member_id: int) -> float:
    """보의 tributary width 추정 (m).

    인접 평행 보까지의 평균 거리로 추정.
    정확한 값은 슬래브 요소 추가 시 개선 예정.
    """
    ni = model.nodes.get(minfo["ni"])
    nj = model.nodes.get(minfo["nj"])
    if ni is None or nj is None:
        return 3.0  # 기본값

    # 부재 방향 판별 (X or Y)
    dx = abs(nj.x - ni.x)
    dy = abs(nj.y - ni.y)
    is_x_dir = dx >= dy

    # 직교 방향 좌표
    if is_x_dir:
        my_perp = (ni.y + nj.y) / 2
        # 같은 층, X방향 보의 Y좌표 수집
        perp_coords = set()
        for e in model.elements.values():
            n_i = model.nodes.get(e.node_i)
            n_j = model.nodes.get(e.node_j)
            if n_i is None or n_j is None:
                continue
            if n_i.story != ni.story:
                continue
            e_dx = abs(n_j.x - n_i.x)
            e_dy = abs(n_j.y - n_i.y)
            if e_dx >= e_dy:  # X방향 보
                perp_coords.add(round((n_i.y + n_j.y) / 2, 3))
        # 인접 Y좌표 거리
        sorted_perp = sorted(perp_coords)
    else:
        my_perp = (ni.x + nj.x) / 2
        perp_coords = set()
        for e in model.elements.values():
            n_i = model.nodes.get(e.node_i)
            n_j = model.nodes.get(e.node_j)
            if n_i is None or n_j is None:
                continue
            if n_i.story != ni.story:
                continue
            e_dx = abs(n_j.x - n_i.x)
            e_dy = abs(n_j.y - n_i.y)
            if e_dy >= e_dx:  # Y방향 보
                perp_coords.add(round((n_i.x + n_j.x) / 2, 3))
        sorted_perp = sorted(perp_coords)

    if len(sorted_perp) < 2:
        return 3.0  # 기본값

    # 인접 좌표까지의 반거리 합
    idx = min(range(len(sorted_perp)), key=lambda i: abs(sorted_perp[i] - my_perp))
    left = (sorted_perp[idx] - sorted_perp[idx - 1]) / 2 if idx > 0 else 0
    right = (sorted_perp[idx + 1] - sorted_perp[idx]) / 2 if idx < len(sorted_perp) - 1 else 0

    trib = left + right
    return max(trib, 0.5)  # 최소 0.5m


def analyze_from_model(
    model,
    load_cases: dict[str, list[dict]],
    load_combinations: dict[str, dict[str, float]] | None = None,
) -> Frame3DMultiCaseResult:
    """V2 진입점: StructuralModel에서 직접 3D 해석.

    기존 analyze_frame_3d_multi()와 동일한 결과 형식을 반환하되,
    격자(bays_x/y) 대신 자유 노드-요소 그래프를 입력으로 사용한다.

    Args:
        model: StructuralModel (core.structural_model)
        load_cases: 하중 케이스 딕셔너리
        load_combinations: 하중 조합 딕셔너리 (None이면 조합 없음)

    Returns:
        Frame3DMultiCaseResult
    """
    # 재료 물성 조회
    # 모델의 첫 번째 요소 재료 사용 (동일 재료 가정 — 향후 멀티 재료 확장)
    materials_used = set(e.material for e in model.elements.values())
    primary_material = sorted(materials_used)[0] if materials_used else "SS275"

    mat = get_material_from_db(primary_material)
    if mat is None:
        mat = DEFAULT_MATERIALS.get(primary_material, DEFAULT_MATERIALS["SS275"])
    E = mat.E    # MPa
    G = E / (2.0 * (1.0 + 0.3))  # ν = 0.3
    fy = mat.fy  # MPa

    # 단면 캐시
    section_cache: dict[str, BeamSection3D] = {}

    # story_nodes_map 구축
    story_nodes_map: dict[int, list[int]] = {}
    for n in model.nodes.values():
        if n.story is not None:
            story_nodes_map.setdefault(n.story, []).append(n.id)

    stories = model.story_heights
    n_stories = model.num_stories

    # 케이스별 해석
    case_results = {}
    member_forces = {}
    analysis_metadata = {}

    for case_name, case_loads in load_cases.items():
        # 모델 구축
        nodes_3d, elements_info, member_to_elements, member_info_list, \
            base_node_ids, _next_nid = _build_model_v2(model, E, G, section_cache)

        # 하중 적용
        _apply_loads_v2(case_loads, model, member_to_elements, member_info_list)

        # 해석
        solver_meta = _solve(model.rigid_diaphragm, model.geometric_nonlinearity)
        if case_name not in analysis_metadata:
            analysis_metadata = solver_meta

        if solver_meta["ok"] != 0:
            case_results[case_name] = Frame3DCaseResult()
            continue

        # 결과 추출
        case_result = _extract_case_results_3d(
            nodes=nodes_3d,
            elements_info=elements_info,
            base_nodes=base_node_ids,
            stories=stories,
            node_grid=None,
            n_cols_x=0,
            n_cols_y=0,
            supports="fixed",
            story_nodes_map=story_nodes_map,
        )
        case_results[case_name] = case_result

        # 부재력
        mf = _extract_member_forces_3d(
            member_info_list, member_to_elements,
            model.num_elements_per_member,
        )
        member_forces[case_name] = mf

    # 하중 조합
    combo_results = {}
    if load_combinations:
        for combo_name, factors in load_combinations.items():
            combo_results[combo_name] = _superpose_case_results_3d(
                case_results=case_results,
                factors=factors,
                nodes=nodes_3d,
                elements_info=elements_info,
                base_nodes=base_node_ids,
                stories=stories,
                node_grid=None,
                n_cols_x=0,
                n_cols_y=0,
                story_nodes_map=story_nodes_map,
            )

    # 결과 조립
    # 대표 단면 정보 (첫 번째 기둥, 보)
    from core.structural_model import ElementType as ET
    col_sections = [e.section for e in model.elements.values() if e.elem_type == ET.COLUMN]
    beam_sections = [e.section for e in model.elements.values() if e.elem_type == ET.BEAM]
    col_sec_name = col_sections[0] if col_sections else "H-300x300"
    beam_sec_name = beam_sections[0] if beam_sections else "H-400x200"

    col_sec = section_cache.get(col_sec_name, get_section_3d(col_sec_name))
    beam_sec = section_cache.get(beam_sec_name, get_section_3d(beam_sec_name))

    multi = Frame3DMultiCaseResult(
        num_stories=n_stories,
        num_bays_x=0,
        num_bays_y=0,
        total_height=model.total_height,
        total_width_x=0.0,
        total_width_y=0.0,
        stories=stories,
        bays_x=[],
        bays_y=[],
        nodes=[{"id": n.id, "x": n.x, "y": n.y, "z": n.z,
                "x_m": n.x, "y_m": n.y, "z_m": n.z}
               for n in sorted(model.nodes.values(), key=lambda n: n.id)],
        elements=[{"id": e.id, "ni": e.node_i, "nj": e.node_j,
                   "type": e.elem_type.value, "section": e.section}
                  for e in sorted(model.elements.values(), key=lambda e: e.id)],
        supports="mixed",
        num_elements_per_member=model.num_elements_per_member,
        column_section=col_sec_name,
        beam_x_section=beam_sec_name,
        beam_y_section=beam_sec_name,
        material_name=primary_material,
        E_MPa=E,
        G_MPa=G,
        fy_MPa=fy,
        column_A_mm2=col_sec.A,
        column_Ix_mm4=col_sec.Ix,
        column_Iy_mm4=col_sec.Iy,
        column_J_mm4=col_sec.J,
        beam_x_A_mm2=beam_sec.A,
        beam_x_Ix_mm4=beam_sec.Ix,
        beam_x_Iy_mm4=beam_sec.Iy,
        beam_x_J_mm4=beam_sec.J,
        beam_y_A_mm2=beam_sec.A,
        beam_y_Ix_mm4=beam_sec.Ix,
        beam_y_Iy_mm4=beam_sec.Iy,
        beam_y_J_mm4=beam_sec.J,
        column_h_mm=getattr(col_sec, 'h', 0),
        column_b_mm=getattr(col_sec, 'b', 0),
        column_tw_mm=getattr(col_sec, 'tw', 0) or 0,
        column_tf_mm=getattr(col_sec, 'tf', 0) or 0,
        beam_x_h_mm=getattr(beam_sec, 'h', 0),
        beam_x_b_mm=getattr(beam_sec, 'b', 0),
        beam_x_tw_mm=getattr(beam_sec, 'tw', 0) or 0,
        beam_x_tf_mm=getattr(beam_sec, 'tf', 0) or 0,
        beam_y_h_mm=getattr(beam_sec, 'h', 0),
        beam_y_b_mm=getattr(beam_sec, 'b', 0),
        beam_y_tw_mm=getattr(beam_sec, 'tw', 0) or 0,
        beam_y_tf_mm=getattr(beam_sec, 'tf', 0) or 0,
        member_info=member_info_list if member_info_list else [],
        load_cases=load_cases,
        case_results=case_results,
        load_combinations=load_combinations or {},
        combo_results=combo_results,
        member_forces=member_forces,
        geometric_nonlinearity=model.geometric_nonlinearity,
        analysis_metadata=analysis_metadata,
    )

    # ── Modal Analysis ──
    try:
        # DL 케이스에서 층별 중량 추정
        dl_result = case_results.get("DL")
        if dl_result and dl_result.reactions:
            total_rz = sum(r["RZ_kN"] for r in dl_result.reactions)
            if total_rz > 0 and n_stories > 0:
                # 층별 중량 = 총 중량 / 층 수 (간이 추정)
                sw = [total_rz / n_stories] * n_stories

                eigen_result = _run_eigen_analysis_v2(
                    model, E, G, section_cache, sw, story_nodes_map,
                )
                if eigen_result:
                    multi.modal_analysis = eigen_result
    except Exception as e:
        print(f"V2 modal analysis skipped: {e}")

    return multi


def _run_eigen_analysis_v2(
    model,
    E: float,
    G: float,
    section_cache: dict,
    story_weights_kN: list[float],
    story_nodes_map: dict[int, list[int]],
    num_modes: int = 0,
) -> dict:
    """V2 고유치해석 — StructuralModel 기반.

    _build_model_v2로 모델 재구축 후 질량 배정 + eigen 풀이.
    """
    n_stories = model.num_stories
    if not story_weights_kN or len(story_weights_kN) != n_stories:
        return {}

    # 1. 모델 재구축 (num_elements_per_member=1, rigid_diaphragm=True)
    orig_num_elem = model.num_elements_per_member
    orig_rigid = model.rigid_diaphragm
    model.num_elements_per_member = 1
    model.rigid_diaphragm = True

    nodes_3d, _ei, _mte, _mil, _bni, _nni = _build_model_v2(model, E, G, section_cache)

    model.num_elements_per_member = orig_num_elem
    model.rigid_diaphragm = orig_rigid

    # 2. 질량 배정
    g_acc = 9810.0  # mm/s²
    node_map_3d = {n.id: n for n in nodes_3d}
    floor_masses = []

    for s in range(1, n_stories + 1):
        snodes = story_nodes_map.get(s, [])
        if not snodes:
            continue
        nodes_per_floor = len(snodes)

        master_nid = snodes[len(snodes) // 2]
        mx_mm = node_map_3d[master_nid].x * 1000
        my_mm = node_map_3d[master_nid].y * 1000

        W_N = story_weights_kN[s - 1] * 1000.0
        m_per_node = W_N / g_acc / nodes_per_floor
        m_floor = W_N / g_acc

        I_eff = 0.0
        for nid in snodes:
            ops.mass(nid, m_per_node, m_per_node, 1e-6, 0.0, 0.0, 0.0)
            if nid in node_map_3d:
                dx = node_map_3d[nid].x * 1000 - mx_mm
                dy = node_map_3d[nid].y * 1000 - my_mm
                I_eff += m_per_node * (dx ** 2 + dy ** 2)

        floor_masses.append((m_floor, I_eff))

    # 3. 모드 수
    if num_modes <= 0:
        num_modes = min(3 * n_stories, 15)

    # 4. 고유치 풀이
    eigenvalues = None
    for solver in [lambda: ops.eigen(num_modes),
                   lambda: ops.eigen('-genBandArpack', num_modes),
                   lambda: ops.eigen('-fullGenLapack', num_modes)]:
        try:
            eigenvalues = solver()
            break
        except Exception:
            continue

    if not eigenvalues:
        return {}

    # 5. 참여질량 계산
    total_mass_x = sum(fm[0] for fm in floor_masses)
    total_mass_rz = sum(fm[1] for fm in floor_masses)
    stories = model.story_heights

    modes = []
    cum_x, cum_y, cum_rz = 0.0, 0.0, 0.0

    for i, lam in enumerate(eigenvalues):
        mode_num = i + 1
        if lam <= 0:
            continue

        omega = math.sqrt(lam)
        T = 2.0 * math.pi / omega
        f = 1.0 / T

        # 모드 형상 추출
        shape = {}
        for n3d in nodes_3d:
            try:
                phi = [ops.nodeEigenvector(n3d.id, mode_num, dof) for dof in range(1, 7)]
                shape[str(n3d.id)] = phi[:3]  # dx, dy, dz
            except Exception:
                shape[str(n3d.id)] = [0, 0, 0]

        # 참여질량 (간이)
        Lx, Ly, Lrz = 0.0, 0.0, 0.0
        Mx, My, Mrz = 0.0, 0.0, 0.0
        for s_idx in range(1, n_stories + 1):
            snodes = story_nodes_map.get(s_idx, [])
            if not snodes or s_idx - 1 >= len(floor_masses):
                continue
            m_f, I_f = floor_masses[s_idx - 1]
            avg_phi_x = sum(shape.get(str(nid), [0, 0, 0])[0] for nid in snodes) / max(len(snodes), 1)
            avg_phi_y = sum(shape.get(str(nid), [0, 0, 0])[1] for nid in snodes) / max(len(snodes), 1)
            Lx += m_f * avg_phi_x
            Ly += m_f * avg_phi_y
            Mx += m_f * avg_phi_x ** 2
            My += m_f * avg_phi_y ** 2

        mp_x = (Lx ** 2 / Mx / total_mass_x * 100) if Mx > 0 and total_mass_x > 0 else 0
        mp_y = (Ly ** 2 / My / total_mass_x * 100) if My > 0 and total_mass_x > 0 else 0

        direction = "X" if abs(Lx) > abs(Ly) * 1.5 else ("Y" if abs(Ly) > abs(Lx) * 1.5 else "XY")
        cum_x += mp_x
        cum_y += mp_y

        modes.append({
            "mode_num": mode_num,
            "eigenvalue": round(lam, 4),
            "frequency_Hz": round(f, 4),
            "period_s": round(T, 4),
            "direction": direction,
            "mass_participation_x_pct": round(mp_x, 2),
            "mass_participation_y_pct": round(mp_y, 2),
            "cumulative_x_pct": round(cum_x, 2),
            "cumulative_y_pct": round(cum_y, 2),
            "shape": shape,
        })

    if not modes:
        return {}

    return {
        "num_modes": len(modes),
        "modes": modes,
        "fundamental_periods": {
            "T1_x": next((m["period_s"] for m in modes if "X" in m["direction"]), None),
            "T1_y": next((m["period_s"] for m in modes if "Y" in m["direction"]), None),
        },
    }
