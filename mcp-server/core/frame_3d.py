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
  - ops.eleForce(tag) returns 12 values in GLOBAL coordinates.
  - For member-design quantities (N, Vy, Vz, T, My, Mz) we need LOCAL coords,
    so this module uses ops.eleResponse(tag, 'localForce'):
      [N_i, Vy_i, Vz_i, T_i, My_i, Mz_i, N_j, Vy_j, Vz_j, T_j, My_j, Mz_j]
  - Local axes defined by element orientation + geomTransf vecxz
    (column local x = vertical, beam local x = beam axis).
===================================================================================
"""
from __future__ import annotations

import math
from core.ops_compat import ops
from dataclasses import dataclass, field
from typing import Literal, Optional

from core.section_3d import get_section_3d, BeamSection3D, DEFAULT_SECTIONS_3D
from core.simple_beam import get_material_from_db, DEFAULT_MATERIALS

import logging
_log = logging.getLogger(__name__)


# ============================================================
# Tier2-14: JSON 직렬화 안전 — 비유한값(NaN/Inf) 방어
# ============================================================

def _safe_round(v, n=6):
    """유한값이면 round, 아니면 0.0 (NaN/Inf → JSON 직렬화 안전)."""
    try:
        return round(v, n) if isinstance(v, (int, float)) and math.isfinite(v) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _safe_extreme(vals, fn, n=4):
    """유한값만 골라 max/min 후 round. 전부 비유한/빈 리스트면 0.0."""
    finite = [v for v in vals if isinstance(v, (int, float)) and math.isfinite(v)]
    if not finite:
        return 0.0
    return round(fn(finite), n)


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
    # Tier2-14(안전): 솔버 ok=0이어도 변위/층간변위가 비유한(NaN/Inf)이면 True →
    # design_check가 '안전' 대신 해석실패로 게이트.
    nonfinite_response: bool = False

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

# 영길이(zero-length) 요소 판정 좌표 허용오차 (mm) — V1 _degenerate_connections /
# V2 analyze_from_model 공통 사용(동형 보장).
_ZERO_LEN_TOL_MM = 1.0


# ============================================================
# Tier-3 견고성/모델링 보조 (특이행렬·None 폴백·근사 가정 명시)
# ============================================================

def _z_story_map(z_values) -> dict:
    """고유 z레벨 → 층 인덱스(0=base, 1..N=층) 매핑.

    T3-1: node_grid/story_nodes_map 누락으로 member story가 None일 때, 절점
    z좌표만으로 층을 복원하는 기하 폴백. 항상 산정 가능하므로 '층 정보 없음'
    누락 버그를 제거한다.
    """
    levels = sorted({round(float(z), 3) for z in z_values})
    return {z: i for i, z in enumerate(levels)}


def _story_from_z(z, z_map) -> "int | None":
    """z좌표(m)를 _z_story_map으로 층 인덱스 조회 (없으면 None)."""
    return z_map.get(round(float(z), 3))


def _degenerate_connections(nodes, connections, tol_mm: float = _ZERO_LEN_TOL_MM) -> list[int]:
    """영길이(ni==nj 또는 두 절점 좌표 일치) 연결의 1-based member_id 목록.

    T3-2: 비정형 좌표병합으로 두 절점이 사실상 겹치면 elasticBeamColumn 길이≈0 →
    강성 특이/발산. V2(analyze_from_model)는 이미 제거하지만 V1 경로엔 없었다.
    좌표는 m 단위, tol_mm는 mm 허용오차(기본 1mm).
    """
    coord = {n.id: (n.x, n.y, n.z) for n in nodes}
    tol_m = tol_mm / 1000.0
    tol_sq = tol_m * tol_m
    bad: list[int] = []
    for mid, conn in enumerate(connections, start=1):
        ni, nj = conn[0], conn[1]
        if ni == nj:
            bad.append(mid)
            continue
        ci, cj = coord.get(ni), coord.get(nj)
        if ci is not None and cj is not None:
            d2 = (ci[0] - cj[0]) ** 2 + (ci[1] - cj[1]) ** 2 + (ci[2] - cj[2]) ** 2
            if d2 <= tol_sq:
                bad.append(mid)
    return bad


def _diaphragm_rank_warnings(story_nodes_map: dict, rigid_diaphragm: bool) -> list[dict]:
    """강막(rigidDiaphragm) 절점 수가 부족한 층의 rank 결손 경고 (T3-3).

    강체 다이어프램은 면내 평면을 정의하려면 비공선 절점 ≥3개가 필요하다. 절점이
    2개면 면내 회전 DOF가 구속되지 못해 rank 결손/특이행렬·허위 0주기 모드 위험.
    1개(또는 0)면 다이어프램 자체가 형성되지 않는다. 분류·경고만 (해석 자체는
    빈 결과/솔버게이트가 별도 차단). 층 0(지점)은 제외.
    """
    if not rigid_diaphragm or not story_nodes_map:
        return []
    warns: list[dict] = []
    for s in sorted(k for k in story_nodes_map if k is not None and k >= 1):
        n = len(story_nodes_map[s])
        if n < 1:
            continue
        if n < 2:
            warns.append({
                "code": "W07", "story": int(s), "node_count": int(n),
                "message": (f"{s}층 강막 절점 {n}개(<2) — 강체 다이어프램 미형성"
                            "(층 수평구속 부족)"),
            })
        elif n < 3:
            warns.append({
                "code": "W07", "story": int(s), "node_count": int(n),
                "message": (f"{s}층 강막 절점 {n}개(<3) — 평면 미정의로 면내 회전 "
                            "rank 결손/특이행렬 위험. 절점 추가 또는 강막 해제 권장"),
            })
    return warns


def _modeling_assumptions(rigid_diaphragm: bool, has_releases: bool) -> list[dict]:
    """모델링 근사·가정 명시 (T3-4/T3-8) — 과대주장 방지용 리포트 노트.

    - A06: 바닥을 완전 강체로 가정(유연/반강성 다이어프램 미모델).
    - A08: 부재 릴리즈는 완전 핀(휨모멘트 0); 반강접(부분강성) 미지원 + 비틀림 DOF는
      elem_type 라벨 기준 → 실제 방향 불일치 시 특이행렬 위험.
    """
    notes: list[dict] = []
    if rigid_diaphragm:
        notes.append({
            "code": "A06",
            "message": ("바닥 다이어프램을 완전 강체(rigid)로 가정 — 유연/반강성 "
                        "다이어프램은 미모델링. 장경간 슬래브·데크 등 면내 변형이 큰 "
                        "경우 층력 분배·비틀림이 달라질 수 있음(KDS 41 17 00)."),
        })
    if has_releases:
        notes.append({
            "code": "A08",
            "message": ("부재 릴리즈(힌지)는 완전 핀(휨모멘트 0)으로 모델링 — "
                        "반강접(부분강성) 미지원. 비틀림 DOF는 부재 elem_type 기준 "
                        "방향으로 연결되며, 실제 부재 방향이 라벨과 다르면 특이행렬 "
                        "위험(좌표 기반 분류 권장)."),
        })
    return notes


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

    # 2단계: 부재 연결 (중복 방지 + 경계 공유부재 패널 합산)
    connections: list[tuple[int, int, str]] = []
    member_metadata: list[dict] = []
    conn_index: dict[tuple[int, int, str], int] = {}

    def _add_conn(ni, nj, etype, meta):
        key = (min(ni, nj), max(ni, nj), etype)
        idx = conn_index.get(key)
        if idx is None:
            conn_index[key] = len(connections)
            connections.append((ni, nj, etype))
            member_metadata.append(meta)
        else:
            # 두 영역이 공유하는 경계 보: 각 영역이 자기쪽 트리뷰터리 패널만 등록하므로
            # 반대편 영역 등록 시 패널/트리뷰터리를 합산해야 floor_area 하중이 누락되지
            # 않는다(구버전은 중복키를 통째로 버려 경계 패널 하중이 사라졌다).
            # 연결(요소)은 한 번만 생성해 강성 중복을 막는다.
            existing = member_metadata[idx]
            new_panels = meta.get("adj_panels")
            if new_panels:
                existing.setdefault("adj_panels", [])
                existing["adj_panels"].extend(new_panels)
            if meta.get("trib_width"):
                existing["trib_width"] = existing.get("trib_width", 0.0) + meta["trib_width"]

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
                    # 인접 패널 정보 (2-way 분배용)
                    adj_panels = []
                    if iy > 0:
                        adj_panels.append((bays_x[ix], bays_y[iy - 1]))
                    if iy < n_cy - 1:
                        adj_panels.append((bays_x[ix], bays_y[iy]))
                    _add_conn(ni, nj, "beam_x", {
                        "story": s, "zone_id": zid, "trib_width": trib_y,
                        "adj_panels": adj_panels,
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
                    adj_panels = []
                    if ix > 0:
                        adj_panels.append((bays_x[ix - 1], bays_y[iy]))
                    if ix < n_cx - 1:
                        adj_panels.append((bays_x[ix], bays_y[iy]))
                    _add_conn(ni, nj, "beam_y", {
                        "story": s, "zone_id": zid, "trib_width": trib_x,
                        "adj_panels": adj_panels,
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

    # Reverse map node_id → story (0 = base, 1..N = floor index). Used to
    # tag member_info entries with a ``story`` field that the chat
    # inspect_selection tool surfaces when the user asks "이거 어느 층?".
    # Without this Node3D has no .story attribute so the cache builder
    # always returned None — the LLM correctly said "층 정보 없음" but
    # that was a missing-data bug, not a real "unknown story" case.
    node_to_story: dict[int, int] = {}
    if node_grid:
        for key, nid in node_grid.items():
            node_to_story[nid] = key[0]  # key = (story, cx, cy)
    elif story_nodes_map:
        for s, nids in story_nodes_map.items():
            for nid in nids:
                node_to_story[nid] = s
    # T3-1: node_grid/story_nodes_map가 모두 없을 때(혹은 일부 절점 누락) member story가
    # None이 되던 버그. 절점 z좌표로 층을 복원하는 기하 폴백(항상 가용).
    _z_map = _z_story_map(n.z for n in nodes)

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
        # T3-4: 예상 외 etype이어도 KeyError로 죽지 않게 RZ(6) 폴백 (모델링 노트로 별도 명시).
        torsion_dof = _TORSION_DOF.get(etype, 6)
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
        # ``story`` tagged from the nj side: columns span (story-1) → story,
        # beams sit at their story — using nj's story gives the floor the
        # member supports (column at story 1 = "1층 기둥", beam_x at story 1
        # = "1층 보"). Falls back to ni for safety.
        # None-aware 해결: story 0(base)이 falsy로 버려지지 않게 명시적 None 체크
        # (구 'a or b'는 0을 falsy로 보아 base층 부재를 오매핑 — 적대적 리뷰 confirmed).
        member_story = node_to_story.get(nj)
        if member_story is None:
            member_story = node_to_story.get(ni)
        if member_story is None:  # T3-1: 매핑 누락 → z좌표 기반 폴백
            member_story = _story_from_z(nj_node.z, _z_map)
            if member_story is None:
                member_story = _story_from_z(ni_node.z, _z_map)
        member_info_list.append({
            "member_id": member_id,
            "type": etype,
            "ni": ni,
            "nj": nj,
            "length_m": round(length, 4),
            "section": sec.name,
            "element_ids": member_elem_ids,
            "story": member_story,
        })

    return elements_info, member_to_elements, member_info_list, next_node_id


def _get_node_by_id(nodes: list[Node3D], node_id: int) -> Node3D:
    """노드 ID로 노드 객체 반환."""
    for n in nodes:
        if n.id == node_id:
            return n
    raise ValueError(f"Node {node_id} not found")


# ============================================================
# 2-way slab 하중 분배
# ============================================================

def _panel_beam_contribution(w: float, panel_x: float, panel_y: float,
                             beam_dir: str) -> float:
    """한 패널이 인접 보에 기여하는 등가 UDL (kN/m) 반환.

    45° yield line method, shear-equivalent UDL.
    - 단변 보 (삼각형 하중): w_eq = w × a / 4
    - 장변 보 (사다리꼴 하중): w_eq = w × a × (2b − a) / (4b)
    총 하중 보존: 2×w_short×a + 2×w_long×b = w×a×b
    """
    if panel_x <= 0 or panel_y <= 0:
        return 0.0
    a = min(panel_x, panel_y)
    b = max(panel_x, panel_y)
    if beam_dir == "beam_x":
        beam_span = panel_x
    else:
        beam_span = panel_y
    if beam_span <= a:  # 단변 보 (삼각형)
        return w * a / 4.0
    else:  # 장변 보 (사다리꼴)
        return w * a * (2.0 * b - a) / (4.0 * b)


# ============================================================
# 하중 적용
# ============================================================

# ============================================================
# 분포하중 적용 — 등가절점하중 방식 (eleLoad recovery 버그 회피)
# ============================================================
#
# 배경(2026-07-01, G3/E2 ETABS 교차검증에서 발견): OpenSees `eleLoad -beamUniform`을
# 모멘트접합(프레임) 보에 적용하면 고정단력을 절점에 비대칭(i-end 편향)으로 오귀속하여,
# 전 보가 i→j 동일방향인 정형골조에서 중력해가 최대 ~42% 비대칭이 된다(정정 단순보/절점
# 하중은 무영향 → 벤치마크 case1~5는 미영향). 상용(ETABS)은 대칭 정답.
#
# 해결: 분포하중을 **등가(consistent) 절점하중**으로 적용하면 변위/반력이 대칭·정확해진다
# (ETABS와 <1%). 단 등가절점하중만으로는 요소 내부력이 고정단항(q0)만큼 어긋나므로,
# 요소력 추출 시 q0를 더해 복원한다(_localforce_with_q0). q0 보정은 아래 캘리브레이션값:
#   Vz_i += wL/2,  My_i += -wL²/12,  Vz_j += wL/2,  My_j += +wL²/12  (local, N·mm)
# beam_x/beam_y 동일(둘 다 local z ≡ global Z). 결과적으로 (등가절점해 + q0) = eleLoad의
# 올바른 요소력이며, 프레임에서는 등가절점해가 정확하므로 요소력도 정확해진다.
#
# _ELEM_UDL: {elem_id: w_Nmm(local-z UDL 크기, +아래방향)} — 케이스마다 초기화.

_ELEM_UDL: dict[int, float] = {}


def _apply_beam_udl(eid: int, w_Nmm: float, etype: str) -> None:
    """보 등분포하중을 등가절점하중으로 적용하고 q0 복원용으로 등록.

    w_Nmm: local-z 방향 하중 크기(양수, 아래방향). etype: 'beam_x'/'beam_y'.
    """
    if not w_Nmm:
        return
    ni, nj = ops.eleNodes(eid)
    ci = ops.nodeCoord(ni)
    cj = ops.nodeCoord(nj)
    L = ((cj[0] - ci[0]) ** 2 + (cj[1] - ci[1]) ** 2 + (cj[2] - ci[2]) ** 2) ** 0.5
    S = w_Nmm * L / 2.0           # 등가 절점 전단(N, 아래방향)
    m0 = w_Nmm * L * L / 12.0     # 고정단 모멘트(N·mm)
    # 방향-강건화: 등가 고정단모멘트의 전역-축 부호는 부재의 물리 위치(좌/우단)에
    # 묶여야 하며 i/j 라벨링에 의존해선 안 된다. V1 격자 빌더는 보를 항상 i→j = +X/+Y로
    # 생성하므로 sgn=+1(무변경)이지만, V2(임의 노드-요소 그래프: IFC/웹 에디터)는 보가
    # -X/-Y로 정렬될 수 있어 sgn 없이는 반대편 모멘트가 적용돼 대칭이 깨진다.
    # (등가절점해가 아닌 전역 모멘트 적용이므로 좌표차 부호로 결정. localForce/q0는
    #  국부좌표라 방향-불변 → _localforce_with_q0는 수정 불필요.)
    if etype == "beam_x":         # global Y축 휨(DOF 5)
        sgn = 1.0 if (cj[0] - ci[0]) >= 0 else -1.0
        ops.load(ni, 0.0, 0.0, -S, 0.0, +m0 * sgn, 0.0)
        ops.load(nj, 0.0, 0.0, -S, 0.0, -m0 * sgn, 0.0)
    else:                          # beam_y: global X축 휨(DOF 4)
        sgn = 1.0 if (cj[1] - ci[1]) >= 0 else -1.0
        ops.load(ni, 0.0, 0.0, -S, +m0 * sgn, 0.0, 0.0)
        ops.load(nj, 0.0, 0.0, -S, -m0 * sgn, 0.0, 0.0)
    _ELEM_UDL[eid] = w_Nmm


def _localforce_with_q0(eid: int) -> list:
    """요소 localForce(12성분) + 등분포하중 고정단항(q0) 복원.

    등가절점하중으로 적용된 요소는 K·v만 리포트(q0 누락)하므로, 등록된 하중으로
    q0를 더해 실제 분포하중 내부력을 복원한다. eleLoad 미사용 요소(기둥·비등록)는 원본 반환.
    """
    forces = list(ops.eleResponse(eid, 'localForce'))
    w = _ELEM_UDL.get(eid)
    if w and len(forces) >= 12:
        ni, nj = ops.eleNodes(eid)
        ci = ops.nodeCoord(ni)
        cj = ops.nodeCoord(nj)
        L = ((cj[0] - ci[0]) ** 2 + (cj[1] - ci[1]) ** 2 + (cj[2] - ci[2]) ** 2) ** 0.5
        S = w * L / 2.0
        m0 = w * L * L / 12.0
        forces[2] += S        # Vz_i
        forces[4] += -m0      # My_i
        forces[8] += S        # Vz_j
        forces[10] += m0      # My_j
    return forces


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
    slab_distribution: str = "2way",
):
    """3D 하중 적용."""
    _ELEM_UDL.clear()
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
            # kN/m → N/mm (모델 단위계 N·mm): 1 kN/m = 1000 N / 1000 mm = 1 N/mm →
            # 환산계수 = 1000(N/kN)/1000(mm/m) = 1.0 (수치 동일·항등). ×1000.0/1000.0은
            # 의도된 단위 명시이므로 '불필요'로 단순화하지 말 것.
            w_Nmm = w_kNm * 1000.0 / 1000.0

            # 해당 story의 beam_x, beam_y 부재 찾기
            _apply_floor_load(story, w_Nmm, n_stories, n_cols_x, n_cols_y,
                              connections, member_to_elements)

        elif ld_type == "floor_area":
            # 면적하중 (kN/m²) → tributary width로 보 선하중 변환
            w_area = ld.get("value", 0.0)  # kN/m²
            _apply_floor_area_load(story, w_area, n_stories, n_cols_x, n_cols_y,
                                   connections, member_to_elements, bays_x, bays_y,
                                   slab_distribution=slab_distribution)

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
                # 등가절점하중 방식(eleLoad recovery 버그 회피). beam_x: local z ~ global Z.
                _apply_beam_udl(eid, w_Nmm, "beam_x")

    for i in range(beam_y_per_story):
        mid = beam_y_start + i + 1
        if mid in member_to_elements:
            for eid in member_to_elements[mid]:
                _apply_beam_udl(eid, w_Nmm, "beam_y")


def _apply_floor_area_load(story, w_area_kNm2, n_stories, n_cols_x, n_cols_y,
                            connections, member_to_elements, bays_x, bays_y,
                            slab_distribution="2way"):
    """면적하중을 보 선하중으로 변환 적용.

    slab_distribution:
      - "2way": 45° yield line method (패널 종횡비 반영, 기본값)
      - "uniform": 기존 50/50 균등 분배 (하위 호환)
    총 반력 = w_area × floor_area 가 되도록 보장.
    """
    n_cols_total = n_cols_x * n_cols_y
    col_count = n_stories * n_cols_total
    beam_x_per_story = (n_cols_x - 1) * n_cols_y
    beam_y_per_story = n_cols_x * (n_cols_y - 1)

    # Beam_X: 각 보에 인접 패널 기여 합산
    beam_x_start = col_count + (story - 1) * beam_x_per_story
    bx_idx = 0
    for cy in range(n_cols_y):
        for cx in range(n_cols_x - 1):
            if slab_distribution == "2way":
                # 인접 패널별 기여 합산
                w_line = 0.0
                if cy > 0:
                    w_line += _panel_beam_contribution(
                        w_area_kNm2, bays_x[cx], bays_y[cy - 1], "beam_x")
                if cy < n_cols_y - 1:
                    w_line += _panel_beam_contribution(
                        w_area_kNm2, bays_x[cx], bays_y[cy], "beam_x")
                if w_line == 0.0 and bays_y:
                    # 단일 bay fallback (edge)
                    w_line = _panel_beam_contribution(
                        w_area_kNm2, bays_x[cx], bays_y[0], "beam_x")
            else:
                # uniform: 기존 50/50
                trib_y = 0.0
                if cy > 0:
                    trib_y += bays_y[cy - 1] / 2.0
                if cy < n_cols_y - 1:
                    trib_y += bays_y[cy] / 2.0
                if trib_y == 0.0:
                    trib_y = bays_y[0] if bays_y else 1.0
                w_line = w_area_kNm2 * 0.5 * trib_y

            w_Nmm = w_line  # kN/m = N/mm (단위계 N·mm: 환산계수 1.0 항등 — 위 floor 참조)
            mid = beam_x_start + bx_idx + 1
            if mid in member_to_elements:
                for eid in member_to_elements[mid]:
                    _apply_beam_udl(eid, w_Nmm, "beam_x")
            bx_idx += 1

    # Beam_Y: 각 보에 인접 패널 기여 합산
    beam_y_start = col_count + n_stories * beam_x_per_story + (story - 1) * beam_y_per_story
    by_idx = 0
    for cx in range(n_cols_x):
        for cy in range(n_cols_y - 1):
            if slab_distribution == "2way":
                w_line = 0.0
                if cx > 0:
                    w_line += _panel_beam_contribution(
                        w_area_kNm2, bays_x[cx - 1], bays_y[cy], "beam_y")
                if cx < n_cols_x - 1:
                    w_line += _panel_beam_contribution(
                        w_area_kNm2, bays_x[cx], bays_y[cy], "beam_y")
                if w_line == 0.0 and bays_x:
                    w_line = _panel_beam_contribution(
                        w_area_kNm2, bays_x[0], bays_y[cy], "beam_y")
            else:
                trib_x = 0.0
                if cx > 0:
                    trib_x += bays_x[cx - 1] / 2.0
                if cx < n_cols_x - 1:
                    trib_x += bays_x[cx] / 2.0
                if trib_x == 0.0:
                    trib_x = bays_x[0] if bays_x else 1.0
                w_line = w_area_kNm2 * 0.5 * trib_x

            w_Nmm = w_line  # kN/m = N/mm (단위계 N·mm: 환산계수 1.0 항등 — 위 floor 참조)
            mid = beam_y_start + by_idx + 1
            if mid in member_to_elements:
                for eid in member_to_elements[mid]:
                    _apply_beam_udl(eid, w_Nmm, "beam_y")
            by_idx += 1


def _apply_loads_3d_irregular(
    loads: list[dict],
    story_nodes: dict[int, list[int]],
    connections: list[tuple[int, int, str]],
    member_to_elements: dict[int, list[int]],
    member_metadata: list[dict],
    slab_distribution: str = "2way",
):
    """비정형 건물 하중 적용 (story_nodes + member_metadata 기반)."""
    _ELEM_UDL.clear()
    ops.timeSeries('Linear', 1)
    ops.pattern('Plain', 1, 1)

    for ld in loads:
        ld_type = ld.get("type", "floor")
        story = ld.get("story", 1)

        if ld_type == "floor":
            w_kNm = ld.get("value", 0.0)
            # kN/m → N/mm: 환산계수 = 1000(N/kN)/1000(mm/m) = 1.0 (단위계 N·mm, 수치 항등).
            w_Nmm = w_kNm
            for mid_0, meta in enumerate(member_metadata):
                mid = mid_0 + 1
                if meta["story"] != story:
                    continue
                etype = connections[mid_0][2]
                if etype not in ("beam_x", "beam_y"):
                    continue
                if mid in member_to_elements:
                    for eid in member_to_elements[mid]:
                        _apply_beam_udl(eid, w_Nmm, etype)

        elif ld_type == "floor_area":
            w_area = ld.get("value", 0.0)  # kN/m²
            for mid_0, meta in enumerate(member_metadata):
                mid = mid_0 + 1
                if meta["story"] != story:
                    continue
                etype = connections[mid_0][2]
                if etype not in ("beam_x", "beam_y"):
                    continue
                if slab_distribution == "2way":
                    adj_panels = meta.get("adj_panels", [])
                    if adj_panels:
                        w_line_Nmm = sum(
                            _panel_beam_contribution(w_area, px, py, etype)
                            for px, py in adj_panels
                        )
                    else:
                        # fallback: adj_panels 미제공 시 50/50
                        trib = meta.get("trib_width", 0.0)
                        w_line_Nmm = w_area * 0.5 * trib
                else:
                    trib = meta.get("trib_width", 0.0)
                    w_line_Nmm = w_area * 0.5 * trib
                if w_line_Nmm <= 0:
                    continue
                if mid in member_to_elements:
                    for eid in member_to_elements[mid]:
                        _apply_beam_udl(eid, w_line_Nmm, etype)

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

    # 2. 요소력 (12성분, LOCAL coords — eleForce는 global이므로 localForce 사용)
    #    분포하중 요소는 q0(고정단항) 복원 포함 (등가절점하중 방식 보정).
    for elem in elements_info:
        forces = _localforce_with_q0(elem.id)
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

        ux_up = [ops.nodeDisp(n, 1) for n in upper_nodes]
        uy_up = [ops.nodeDisp(n, 2) for n in upper_nodes]
        lower_dx = sum(ops.nodeDisp(n, 1) for n in lower_nodes) / len(lower_nodes)
        upper_dx = sum(ux_up) / len(ux_up)
        drift_x = (upper_dx - lower_dx) / story_height_mm if story_height_mm > 0 else 0

        lower_dy = sum(ops.nodeDisp(n, 2) for n in lower_nodes) / len(lower_nodes)
        upper_dy = sum(uy_up) / len(uy_up)
        drift_y = (upper_dy - lower_dy) / story_height_mm if story_height_mm > 0 else 0

        drift_r = math.sqrt(drift_x ** 2 + drift_y ** 2)

        # Tier2-14(안전): 솔버가 ok=0이어도 변위/층간변위가 비유한이면 해석 신뢰불가 →
        # 플래그(아래 _safe_*는 JSON 위해 0으로 정리하지만, design_check 게이트가 작동하도록 표시).
        if not all(math.isfinite(v) for v in
                   (drift_x, drift_y, drift_r, *ux_up, *uy_up)):
            result.nonfinite_response = True

        result.story_drifts.append({
            "story": s,
            "height_m": stories[s - 1],
            "drift_x": _safe_round(abs(drift_x), 6),
            "drift_y": _safe_round(abs(drift_y), 6),
            "drift_resultant": _safe_round(drift_r, 6),
            # C2: 다이어프램 절점 변위 극값 (비틀림 비정형 δmax/δavg 산정용).
            # Tier2-14: NaN/Inf 방어 (비수렴 시 0 → JSON 직렬화 안전).
            "disp_x_max": _safe_extreme(ux_up, max), "disp_x_min": _safe_extreme(ux_up, min),
            "disp_y_max": _safe_extreme(uy_up, max), "disp_y_min": _safe_extreme(uy_up, min),
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
            # 부재 설계용 12성분은 LOCAL 좌표여야 함 — eleForce는 global이라
            # 컬럼에서 Vz가 축력으로 오인되는 등 부재력 라벨이 뒤바뀜.
            # 분포하중 요소는 q0(고정단항) 복원 포함.
            forces = _localforce_with_q0(eid)
            s_start = k * sub_len

            # Tier2-14: _safe_round로 NaN/Inf(비수렴) → 0 정리 (raw member_forces도
            # 웹앱 응답에 그대로 직렬화되므로 비표준 JSON 토큰 방지).
            if k == 0:
                s_vals.append(_safe_round(s_start, 4))
                N_vals.append(_safe_round(forces[0] / 1000, 4))
                Vy_vals.append(_safe_round(forces[1] / 1000, 4))
                Vz_vals.append(_safe_round(forces[2] / 1000, 4))
                T_vals.append(_safe_round(forces[3] / 1e6, 4))
                My_vals.append(_safe_round(forces[4] / 1e6, 4))
                Mz_vals.append(_safe_round(forces[5] / 1e6, 4))

            # j-end: 내력 = -반력
            s_end = (k + 1) * sub_len
            s_vals.append(_safe_round(s_end, 4))
            N_vals.append(_safe_round(-forces[6] / 1000, 4))
            Vy_vals.append(_safe_round(-forces[7] / 1000, 4))
            Vz_vals.append(_safe_round(-forces[8] / 1000, 4))
            T_vals.append(_safe_round(-forces[9] / 1e6, 4))
            My_vals.append(_safe_round(-forces[10] / 1e6, 4))
            Mz_vals.append(_safe_round(-forces[11] / 1e6, 4))

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

        ux_up = [node_disp_map.get(n, {}).get("dx_mm", 0) for n in upper_nids]
        uy_up = [node_disp_map.get(n, {}).get("dy_mm", 0) for n in upper_nids]
        lower_dx = sum(node_disp_map.get(n, {}).get("dx_mm", 0) for n in lower_nids) / len(lower_nids)
        upper_dx = sum(ux_up) / len(ux_up)
        drift_x = abs(upper_dx - lower_dx) / story_height_mm if story_height_mm > 0 else 0

        lower_dy = sum(node_disp_map.get(n, {}).get("dy_mm", 0) for n in lower_nids) / len(lower_nids)
        upper_dy = sum(uy_up) / len(uy_up)
        drift_y = abs(upper_dy - lower_dy) / story_height_mm if story_height_mm > 0 else 0

        # Tier2-14(안전): 비유한 응답 → 게이트 플래그 (ok=0이어도 신뢰불가).
        if not all(math.isfinite(v) for v in (drift_x, drift_y, *ux_up, *uy_up)):
            combo.nonfinite_response = True

        combo.story_drifts.append({
            "story": s, "height_m": stories[s - 1],
            "drift_x": _safe_round(drift_x, 6), "drift_y": _safe_round(drift_y, 6),
            "drift_resultant": _safe_round(math.sqrt(drift_x ** 2 + drift_y ** 2), 6),
            # C2: 다이어프램 절점 변위 극값 (비틀림 비정형 δmax/δavg 산정용).
            # Tier2-14: NaN/Inf 방어 (비수렴 시 0 → JSON 직렬화 안전).
            "disp_x_max": _safe_extreme(ux_up, max), "disp_x_min": _safe_extreme(ux_up, min),
            "disp_y_max": _safe_extreme(uy_up, max), "disp_y_min": _safe_extreme(uy_up, min),
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
    usages: list[str] | None = None,
) -> list[float]:
    """DL 하중케이스에서 층별 중력하중(kN) 추정.

    usages(층별 용도, 1층→상층 순)가 주어지면 창고류 층은 유효지진중량 일관성을
    위해 활하중의 25%를 추가한다(KDS 41 17 00, load_generator.seismic_live_load_fraction
    과 동일 규칙). usages 미제공 시 DL만 사용(하위호환).

    Returns:
        story_weights_kN (길이 = num_stories). 빈 리스트 = 추정 불가.
    """
    dl_loads = None
    ll_loads = None
    for case_name, loads in load_cases.items():
        up = case_name.upper()
        if up in ("DL", "DEAD", "D") and dl_loads is None:
            dl_loads = loads
        elif up in ("LL", "LIVE", "L") and ll_loads is None:
            ll_loads = loads

    if not dl_loads:
        return []

    floor_area = sum(bays_x) * sum(bays_y)  # m²
    total_beam_len = sum(bays_x) * n_cols_y + sum(bays_y) * n_cols_x  # m

    def _area_for(ld_type: str) -> float:
        if ld_type == "floor_area":
            return floor_area
        if ld_type == "floor":
            return total_beam_len
        return 0.0

    weights = [0.0] * num_stories
    for ld in dl_loads:
        story = ld.get("story", 0)
        if story < 1 or story > num_stories:
            continue
        weights[story - 1] += ld.get("value", 0.0) * _area_for(ld.get("type", ""))

    # 창고류 활하중 25% (유효지진중량 일관성, KDS 41 17 00) — 비창고는 frac=0.
    if usages and ll_loads:
        from core.load_generator import seismic_live_load_fraction
        for ld in ll_loads:
            story = ld.get("story", 0)
            if story < 1 or story > num_stories:
                continue
            idx = story - 1
            usage = usages[idx] if idx < len(usages) else ""
            frac = seismic_live_load_fraction(usage)
            if frac > 0:
                weights[idx] += frac * ld.get("value", 0.0) * _area_for(ld.get("type", ""))

    return weights if any(w > 0 for w in weights) else []


# #11: 누적 유효질량 참여율 <90% 시 모드 수 자동 증대 상한(면내 기저 3·층수와 min).
_EIGEN_MODE_CAP = 30


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
    node_mass: dict[int, float] = {}  # 절점기반 유효모달질량용 (실제 배정 질량)

    for s in range(1, num_stories + 1):
        snodes = story_nodes_map.get(s, [])
        if not snodes:
            continue
        nodes_per_floor = len(snodes)

        # 마스터 = 강막(rigidDiaphragm) retained 절점. _build_frame_3d_model과 반드시
        # 동일해야 master phi가 '순수 병진(ux,uy)'을 읽는다. 정형격자에서 빌드는
        # node_grid[(s, ncx//2, ncy//2)]를 쓰는데, snodes[len//2]는 n_cols가 짝수면
        # 다른 절점을 가리켜 비틀림(rz)이 ux/uy에 섞여 누적참여율이 100%를 넘었다.
        if not is_irreg and node_grid:
            master_nid = node_grid.get((s, n_cols_x // 2, n_cols_y // 2), snodes[len(snodes) // 2])
        else:
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
            node_mass[nid] = node_mass.get(nid, 0.0) + m_per_node
            dx = node_map[nid].x * 1000 - mx_mm
            dy = node_map[nid].y * 1000 - my_mm
            I_eff += m_per_node * (dx ** 2 + dy ** 2)

        floor_masses.append((m_floor, I_eff))

    total_weight_kN = sum(story_weights_kN)

    # 3. 모드 수 결정 + 총 질량(참여질량 기준 — num_modes와 무관, 미리 계산)
    if num_modes <= 0:
        num_modes = min(3 * num_stories, 15)
    mode_cap = min(3 * num_stories, _EIGEN_MODE_CAP)  # #11 자동증대 상한(면내 기저)

    total_mass_x = sum(fm[0] for fm in floor_masses)
    total_mass_y = total_mass_x  # 동일 (병진질량)
    total_mass_rz = sum(fm[1] for fm in floor_masses)

    all_nids = set()
    for s_nids in story_nodes_map.values():
        all_nids.update(s_nids)

    def _solve_and_extract(nm: int):
        """eigen(nm) 풀이 + 모드별 참여질량/형상 추출. 실패 시 None.

        모델·질량은 이미 배정돼 있으므로 ops.eigen만 재호출하면 nm개 모드를 재산정한다(#11).
        """
        eigenvalues = None
        for solver in (lambda: ops.eigen(nm),
                       lambda: ops.eigen('-genBandArpack', nm),
                       lambda: ops.eigen('-fullGenLapack', nm)):
            try:
                eigenvalues = solver()
                break
            except Exception:
                continue
        if not eigenvalues:
            return None

        modes_ = []
        T1x = T1y = T1rz = None
        cx = cy = crz = 0.0
        for i, lam in enumerate(eigenvalues):
            mode_num = i + 1
            if lam <= 0:
                continue

            omega = math.sqrt(lam)
            T = 2.0 * math.pi / omega
            f = 1.0 / T

            # 참여질량: 절점기반 유효모달질량.
            # 질량은 x·y 병진에만 배정되므로 일반화질량 M_n = Σ m(φx²+φy²).
            # L_x=Σmφx, L_y=Σmφy, L_rz=Σm(Δx·φy−Δy·φx)(각 층 마스터 기준 비틂영향).
            # 마스터≠질량중심이어도 절점합이므로 정확 → 누적참여율 ≤100% 보장.
            M_n = 0.0
            Lx = Ly = Lrz = 0.0
            for s in range(1, num_stories + 1):
                mnid = master_nodes.get(s)
                mxx = node_map[mnid].x * 1000 if mnid in node_map else 0.0
                myy = node_map[mnid].y * 1000 if mnid in node_map else 0.0
                for nid in story_nodes_map.get(s, []):
                    m_nd = node_mass.get(nid, 0.0)
                    if m_nd <= 0:
                        continue
                    try:  # nodeEigenvector 실패(과구속·수치이상) 시 0으로 폴백(V2와 동일)
                        ux = ops.nodeEigenvector(nid, mode_num, 1)
                        uy = ops.nodeEigenvector(nid, mode_num, 2)
                    except Exception:
                        ux = uy = 0.0
                    M_n += m_nd * (ux * ux + uy * uy)
                    Lx += m_nd * ux
                    Ly += m_nd * uy
                    Lrz += m_nd * ((node_map[nid].x * 1000 - mxx) * uy
                                   - (node_map[nid].y * 1000 - myy) * ux)

            def _pct(L, tot):
                if M_n <= 1e-30 or tot <= 1e-30:
                    return 0.0
                return (L * L / M_n) / tot * 100

            mp = {
                "x_pct": round(_pct(Lx, total_mass_x), 2),
                "y_pct": round(_pct(Ly, total_mass_y), 2),
                "rz_pct": round(_pct(Lrz, total_mass_rz), 2),
            }

            cx += mp["x_pct"]
            cy += mp["y_pct"]
            crz += mp["rz_pct"]

            # 지배 방향 판별 (참여질량 기반)
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

            modes_.append({
                "mode": mode_num,
                "period_s": round(T, 4),
                "frequency_hz": round(f, 4),
                "direction": direction,
                "dominance_pct": round(dominance, 2),
                "mass_participation": mp,
                "shape": mode_shape,
            })

            if direction == "TRAN-X" and T1x is None:
                T1x = round(T, 4)
            elif direction == "TRAN-Y" and T1y is None:
                T1y = round(T, 4)
            elif direction == "ROTN-Z" and T1rz is None:
                T1rz = round(T, 4)

        return modes_, T1x, T1y, T1rz, cx, cy, crz

    # 4. 초기 풀이 + 누적참여 <90% 시 모드 수 자동 증대 재시도 (#11)
    res = _solve_and_extract(num_modes)
    if res is None or not res[0]:   # 유효 모드 0개 → 실패(V2와 동일하게 빈 dict 반환)
        return {}
    n_retry = 0
    while (not (res[4] >= 90.0 and res[5] >= 90.0)
           and num_modes < mode_cap and n_retry < 4):
        num_modes = min(num_modes + max(num_stories, 3), mode_cap)
        nxt = _solve_and_extract(num_modes)
        if nxt is None or not nxt[0]:
            break
        improved = (nxt[4] > res[4] + 0.5) or (nxt[5] > res[5] + 0.5)
        res = nxt
        n_retry += 1
        if not improved:   # 모드 추가가 누적참여를 개선 못하면 조기 종료(불필요한 풀이 방지)
            break
    modes, T1_x, T1_y, T1_rz, cum_x, cum_y, cum_rz = res
    if not modes:
        return {}

    # 7. 누적 참여질량 충분조건 (≥90%)
    sufficient = cum_x >= 90.0 and cum_y >= 90.0

    return {
        "num_modes": len(modes),
        "mode_count_auto_increased": n_retry > 0,  # #11
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
    slab_distribution: str = "2way",
    story_usages: list[str] | None = None,
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
        "slab_distribution": slab_distribution,
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

    # T3-2: 영길이(좌표 일치) 연결 제거 — 비정형 좌표병합 시 특이행렬 방지(V2와 동형).
    # 제거 후 member 번호가 재정렬되지만 모든 케이스가 동일 connections를 쓰므로 일관.
    # member_metadata(비정형 하중 적용)는 connections와 1:1 평행이므로 동일 인덱스로 함께
    # 필터해야 하중 매핑이 어긋나지 않는다.
    _robustness_warnings: list[dict] = []
    _degen = _degenerate_connections(nodes, connections)
    if _degen:
        _bad = set(_degen)
        connections = [c for i, c in enumerate(connections, start=1) if i not in _bad]
        if member_metadata is not None:
            member_metadata = [mm for i, mm in enumerate(member_metadata, start=1)
                               if i not in _bad]
        _robustness_warnings.append({
            "code": "W06", "removed": len(_degen),
            "message": (f"영길이/좌표중복 연결 {len(_degen)}개 제거 "
                        f"(특이행렬 방지, member_id {_degen[:10]})"),
        })

    # T3-3: 강막 절점 부족(rank 결손) 경고.
    _robustness_warnings.extend(_diaphragm_rank_warnings(story_nodes_map, rigid_diaphragm))
    if _robustness_warnings:
        multi.analysis_metadata["robustness_warnings"] = _robustness_warnings
    # T3-4/T3-8: 모델링 근사·가정(강막 강체·릴리즈 완전핀) 명시.
    _mnotes = _modeling_assumptions(rigid_diaphragm, bool(member_releases))
    if _mnotes:
        multi.analysis_metadata["modeling_assumptions"] = _mnotes

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
            # T3-1: z-폴백 후에도 층 미해결 부재가 남으면 명시(설계검토 층보고 정확도 영향).
            _none_story = sum(1 for m in member_info_list if m.get("story") is None)
            if _none_story:
                multi.analysis_metadata.setdefault("robustness_warnings", []).append({
                    "code": "W08", "count": _none_story,
                    "message": (f"부재 {_none_story}개의 층 정보를 좌표로도 복원하지 못함 "
                                "— 설계검토 층보고/P-Delta 증폭 매핑이 부정확할 수 있음"),
                })

        # 하중 적용
        if is_irregular:
            _apply_loads_3d_irregular(
                case_loads, story_nodes_map, connections,
                member_to_elements, member_metadata,
                slab_distribution=slab_distribution,
            )
        else:
            _apply_loads_3d(
                case_loads, len(stories), n_cols_x, n_cols_y,
                node_grid, connections, member_to_elements, bays_x, bays_y,
                slab_distribution=slab_distribution,
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
                usages=story_usages,
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

    A-3: vecxz와 요소 방향이 평행하면 singular matrix → fallback 변환 사용.

    Returns:
        1=column(vecxz=X), 2=beam_x(vecxz=Z), 3=beam_y(vecxz=Z), 4=brace_fallback(vecxz=Y)
    """
    _VECXZ = {1: (1, 0, 0), 2: (0, 0, 1), 3: (0, 0, 1), 4: (0, 1, 0)}

    if elem_type == "column":
        candidate = 1
    elif elem_type in ("beam", "beam_x", "brace"):
        dx, dy, _dz = direction_vector
        candidate = 2 if abs(dx) >= abs(dy) else 3
    else:
        # beam_y
        candidate = 3

    # A-3: vecxz 평행 체크 (cross product ≈ 0 → parallel)
    vx, vy, vz = _VECXZ[candidate]
    dx, dy, dz = direction_vector
    cross_sq = (
        (dy * vz - dz * vy) ** 2
        + (dz * vx - dx * vz) ** 2
        + (dx * vy - dy * vx) ** 2
    )
    dir_sq = dx * dx + dy * dy + dz * dz
    if dir_sq > 1e-12 and cross_sq / dir_sq < 1e-6:
        # 평행! fallback: 다른 vecxz 시도
        for alt in (4, 1, 2, 3):
            if alt == candidate:
                continue
            avx, avy, avz = _VECXZ[alt]
            alt_cross_sq = (
                (dy * avz - dz * avy) ** 2
                + (dz * avx - dx * avz) ** 2
                + (dx * avy - dy * avx) ** 2
            )
            if alt_cross_sq / dir_sq > 1e-6:
                import logging
                logging.getLogger(__name__).info(
                    f"[A-3] geomTransf {candidate}→{alt} for elem dir=({dx:.3f},{dy:.3f},{dz:.3f})"
                )
                return alt
    return candidate


def _get_torsion_dof_v2(transf_id: int) -> int:
    """V2 기하변환 ID에 따른 비틀림 DOF."""
    return {1: 6, 2: 4, 3: 5, 4: 5}[transf_id]


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

    # 기하변환 (A-3: 경사 부재의 vecxz 평행 방지용 추가 변환 포함)
    transf_type = 'Corotational' if model.geometric_nonlinearity == "pdelta" else 'Linear'
    ops.geomTransf(transf_type, 1, 1.0, 0.0, 0.0)  # column (vecxz=X)
    ops.geomTransf(transf_type, 2, 0.0, 0.0, 1.0)  # beam_x (vecxz=Z)
    ops.geomTransf(transf_type, 3, 0.0, 0.0, 1.0)  # beam_y (vecxz=Z)
    ops.geomTransf(transf_type, 4, 0.0, 1.0, 0.0)  # brace fallback (vecxz=Y)

    # 요소 생성
    elements_info = []
    member_to_elements = {}
    member_info_list = []
    elem_id = 1
    next_node_id = max(n.id for n in model.nodes.values()) + 1
    num_sub = model.num_elements_per_member
    # T3-1: .story 미할당 절점 대비 z좌표 기반 층 폴백 맵 (항상 가용).
    _z_map_v2 = _z_story_map(n.z for n in model.nodes.values())

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
        # Story tag for chat inspect_selection — see legacy path comment
        # above for the same reasoning. V2 nodes are StructuralNode which
        # carries .story directly, so we can read it without a reverse
        # lookup. nj first (column nj = supported floor), fall back to ni.
        v2_story = getattr(nj_node, "story", None)
        if v2_story is None:
            v2_story = getattr(ni_node, "story", None)
        if v2_story is None:  # T3-1: .story 누락 → z좌표 폴백 (0=base 보존, None-aware)
            v2_story = _story_from_z(nj_node.z, _z_map_v2)
            if v2_story is None:
                v2_story = _story_from_z(ni_node.z, _z_map_v2)
        member_info_list.append({
            "member_id": member_id,
            "type": v1_etype,
            "ni": se.node_i,
            "nj": se.node_j,
            "length_m": round(length, 4),
            "section": sec.name,
            "element_ids": member_elem_ids,
            "story": v2_story,
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
    # 2026-07-01 중력버그 픽스(V1→V2 이식): eleLoad -beamUniform은 모멘트접합 보에서
    # 고정단력을 i-end 편향 오귀속(대칭중력→비대칭해). 등가절점하중(_apply_beam_udl)
    # + 요소력 q0 복원(공유 추출기 _extract_*_3d가 이미 _localforce_with_q0 사용)으로
    # 정정. _ELEM_UDL은 케이스마다(모델 재구축 시 elem_id 재발급) 초기화한다.
    _ELEM_UDL.clear()
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
                # kN/m → N/mm: 환산계수 = 1000(N/kN)/1000(mm/m) = 1.0 (단위계 N·mm, 항등).
                w_Nmm = w_line * 1000 / 1000

                for eid in member_to_elements[mi]:
                    # 수직 등분포하중 → 등가절점하중(중력버그 픽스). minfo["type"]은
                    # beam_x/beam_y, w_Nmm은 양수(아래방향) 크기. 요소력은 추출단계에서
                    # q0 복원(_localforce_with_q0)되어 실제 분포하중 내력과 일치.
                    _apply_beam_udl(eid, w_Nmm, minfo["type"])

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

    # ── B-1: 사전체크 + 자동치유 (한 번만 실행) ──
    import logging
    _log = logging.getLogger(__name__)

    # T3-2: 영길이(ni==nj 또는 좌표 일치) 요소를 **고아 절점 제거보다 먼저** 정리한다.
    # (역순이면 영길이 요소 삭제로 새로 고립된 절점이 정리되지 않아, 무구속 자유절점이
    #  OpenSees에 추가돼 특이행렬을 유발한다 — 적대적 리뷰 confirmed.) 제거 목록은
    # robustness_warnings로 surfacing(침묵 0결과 방지). 좌표일치도 함께 검출.
    _coord_v2 = {nid: (n.x, n.y, n.z) for nid, n in model.nodes.items()}
    _tol_sq_v2 = (_ZERO_LEN_TOL_MM / 1000.0) ** 2  # m²

    def _is_zero_len(e):
        if e.node_i == e.node_j:
            return True
        ci, cj = _coord_v2.get(e.node_i), _coord_v2.get(e.node_j)
        if ci is not None and cj is not None:
            d2 = (ci[0]-cj[0])**2 + (ci[1]-cj[1])**2 + (ci[2]-cj[2])**2
            return d2 <= _tol_sq_v2
        return False

    zero_elems = [eid for eid, e in model.elements.items() if _is_zero_len(e)]
    if zero_elems:
        _log.warning(f"[B-1] Removing {len(zero_elems)} zero-length elements: {zero_elems[:10]}")
        for eid in zero_elems:
            del model.elements[eid]

    # 고아 절점 제거 (영길이 요소 제거 후 새로 고립된 절점까지 포착).
    conn_count: dict[int, int] = {nid: 0 for nid in model.nodes}
    for e in model.elements.values():
        if e.node_i in conn_count:
            conn_count[e.node_i] += 1
        if e.node_j in conn_count:
            conn_count[e.node_j] += 1
    orphans = [nid for nid, cnt in conn_count.items()
               if cnt == 0 and not model.nodes[nid].support]
    if orphans:
        _log.warning(f"[B-1] Removing {len(orphans)} orphan nodes: {orphans[:10]}")
        for nid in orphans:
            del model.nodes[nid]

    has_support = any(n.support for n in model.nodes.values())
    if not has_support:
        raise ValueError(
            "[B-1] 모델에 지점(support)이 없습니다. 최소 1개의 고정/힌지 지점이 필요합니다."
        )

    # 요소 분류 안전장치: 모든 요소가 동일 타입이면 자동 분류
    type_set = set(e.elem_type for e in model.elements.values())
    if len(type_set) == 1 and len(model.elements) > 3:
        _log.info("[B-1] All elements same type → auto classify_elements()")
        model.classify_elements()

    # 케이스별 해석
    case_results = {}
    member_forces = {}
    analysis_metadata = {}
    failed_cases = []  # Tier1-4: 솔버 실패 케이스 추적 (0 결과 '안전' 오판 방지)

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
            # Tier1-4: 비수렴/특이행렬 — 빈 결과로 진행하되 실패를 기록한다.
            # design_check가 이 플래그를 보고 '안전'이 아닌 NG/미검토로 게이트한다.
            _log.warning(
                "[Tier1-4] V2 solver failed (case=%s, rc=%s) — 결과 0 처리, design_check 게이트 대상",
                case_name, solver_meta["ok"],
            )
            failed_cases.append(case_name)
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
            # 조합 부재력 (부재 강도검토용)
            combo_mf = _superpose_member_forces_3d(member_forces, factors)
            if combo_mf:
                member_forces[combo_name] = combo_mf

    # 결과 조립
    # 대표 단면 정보 (첫 번째 기둥, 보)
    from core.structural_model import ElementType as ET
    col_sections = [e.section for e in model.elements.values() if e.elem_type == ET.COLUMN]
    beam_sections = [e.section for e in model.elements.values() if e.elem_type == ET.BEAM]
    col_sec_name = col_sections[0] if col_sections else "H-300x300"
    beam_sec_name = beam_sections[0] if beam_sections else "H-400x200"

    col_sec = section_cache.get(col_sec_name, get_section_3d(col_sec_name))
    beam_sec = section_cache.get(beam_sec_name, get_section_3d(beam_sec_name))

    # V2 bay 추정: 기둥 노드의 X/Y 좌표 클러스터링
    from core.structural_model import ElementType as _ET2
    col_node_ids = set()
    _type_counts = {}
    for e in model.elements.values():
        t = e.elem_type.value if hasattr(e.elem_type, 'value') else str(e.elem_type)
        _type_counts[t] = _type_counts.get(t, 0) + 1
        if e.elem_type == _ET2.COLUMN:
            col_node_ids.add(e.node_i)
            col_node_ids.add(e.node_j)
    _log.info(f"[Bay] element types: {_type_counts}, col_nodes: {len(col_node_ids)}")
    # 최하층(z≈0) 기둥 노드만 사용
    base_col_nodes = [model.nodes[nid] for nid in col_node_ids if nid in model.nodes]
    if base_col_nodes:
        min_z = min(n.z for n in base_col_nodes)
        base_col_nodes = [n for n in base_col_nodes if abs(n.z - min_z) < 0.5]
    xs = sorted(set(round(n.x, 2) for n in base_col_nodes)) if base_col_nodes else []
    ys = sorted(set(round(n.y, 2) for n in base_col_nodes)) if base_col_nodes else []
    est_bays_x = [round(xs[i+1] - xs[i], 3) for i in range(len(xs)-1)] if len(xs) > 1 else []
    est_bays_y = [round(ys[i+1] - ys[i], 3) for i in range(len(ys)-1)] if len(ys) > 1 else []
    est_width_x = (xs[-1] - xs[0]) if len(xs) > 1 else 0.0
    est_width_y = (ys[-1] - ys[0]) if len(ys) > 1 else 0.0

    # Tier1-4: 솔버 실패가 있으면 메타데이터에 플래그를 실어 design_check가 게이트.
    if not isinstance(analysis_metadata, dict):
        analysis_metadata = {}
    analysis_metadata = dict(analysis_metadata)
    if failed_cases:
        analysis_metadata["solver_failed"] = True
        analysis_metadata["failed_cases"] = list(failed_cases)

    # ── Tier-3 견고성/모델링 surfacing (V2) ──
    _rob: list[dict] = []
    if zero_elems:  # T3-2: 영길이 요소 제거 기록
        _rob.append({
            "code": "W06", "removed": len(zero_elems),
            "message": (f"영길이/좌표중복 요소 {len(zero_elems)}개 제거 "
                        f"(특이행렬 방지, elem_id {list(zero_elems)[:10]})"),
        })
    # T3-3: 강막 절점 부족(rank 결손) — story_nodes_map은 .story 기반(층 0=지점 제외).
    _rob.extend(_diaphragm_rank_warnings(story_nodes_map, model.rigid_diaphragm))
    # T3-1: z-폴백 후에도 층 미해결 부재
    try:
        _none_story = sum(1 for m in member_info_list if m.get("story") is None)
    except NameError:
        _none_story = 0
    if _none_story:
        _rob.append({
            "code": "W08", "count": _none_story,
            "message": (f"부재 {_none_story}개의 층 정보를 좌표로도 복원하지 못함 "
                        "— 설계검토 층보고/P-Delta 증폭 매핑이 부정확할 수 있음"),
        })
    if _rob:
        analysis_metadata["robustness_warnings"] = _rob
    # T3-4/T3-8: 모델링 근사·가정(강막 강체·릴리즈 완전핀) 명시.
    _has_rel = any(getattr(e, "release_i", None) or getattr(e, "release_j", None)
                   for e in model.elements.values())
    _mnotes = _modeling_assumptions(model.rigid_diaphragm, _has_rel)
    if _mnotes:
        analysis_metadata["modeling_assumptions"] = _mnotes

    multi = Frame3DMultiCaseResult(
        num_stories=n_stories,
        num_bays_x=len(est_bays_x),
        num_bays_y=len(est_bays_y),
        total_height=model.total_height,
        total_width_x=est_width_x,
        total_width_y=est_width_y,
        stories=stories,
        bays_x=est_bays_x,
        bays_y=est_bays_y,
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
        ns = n_stories or (len(story_nodes_map) - 1 if story_nodes_map else 0)
        if ns > 0:
            # 층별 중량 추정: 반력 > 하중 데이터 > 기본값 순서
            sw = None
            dl_result = case_results.get("DL")
            if dl_result and dl_result.reactions:
                total_rz = sum(r["RZ_kN"] for r in dl_result.reactions)
                if total_rz > 0:
                    sw = [total_rz / ns] * ns

            if sw is None:
                # 반력이 없으면 DL 하중 데이터에서 직접 추정
                dl_loads = load_cases.get("DL", [])
                total_load_kN = 0.0
                for ld in dl_loads:
                    lt = ld.get("type", "")
                    if lt == "floor_area":
                        total_load_kN += abs(ld.get("qz_kN_m2", 0)) * ld.get("area_m2", 0)
                    elif lt == "floor":
                        total_load_kN += abs(ld.get("w_kN_m", 0)) * ld.get("L_m", 1)
                    elif lt == "nodal":
                        total_load_kN += abs(ld.get("Fz_kN", 0))
                if total_load_kN > 0:
                    sw = [total_load_kN / ns] * ns

            if sw is None:
                # 최후 수단: 단위면적 기본하중으로 추정
                area = 0.0
                for e in model.elements.values():
                    if e.elem_type.value in ("beam", "beam_x", "beam_y"):
                        ni, nj = model.nodes[e.node_i], model.nodes[e.node_j]
                        dx, dy, dz = nj.x-ni.x, nj.y-ni.y, nj.z-ni.z
                        area += (dx**2+dy**2+dz**2)**0.5 * 4.0  # 보 길이 × 가정 폭 4m
                if area > 0:
                    sw = [area * 5.0 / ns] * ns  # 5 kN/m² 기본 가정

            # A1: 창고류 활하중 25% (modal 질량 V1/V2 일관성, KDS 41 17 00).
            # base sw(DL)에 창고층만 0.25·LL을 가산한다. 비창고/LL부재 없으면 무변경.
            story_usages = getattr(model, "story_usages", None)
            if sw and story_usages:
                from core.load_generator import seismic_live_load_fraction
                ll_loads = None
                for cn, loads in load_cases.items():
                    if cn.upper() in ("LL", "LIVE", "L"):
                        ll_loads = loads
                        break
                if ll_loads:
                    def _story_plan_area(st):
                        # value(kN/m²) 포맷용 평면면적 — 해당 층 절점 bbox 근사
                        nids = story_nodes_map.get(st, []) if story_nodes_map else []
                        xs = [model.nodes[n].x for n in nids if n in model.nodes]
                        ys = [model.nodes[n].y for n in nids if n in model.nodes]
                        if len(xs) < 2 or len(ys) < 2:
                            return 0.0
                        return (max(xs) - min(xs)) * (max(ys) - min(ys))

                    add = [0.0] * ns
                    for ld in ll_loads:
                        st = ld.get("story", 0)
                        if st < 1 or st > ns:
                            continue
                        frac = seismic_live_load_fraction(story_usages.get(st, "office"))
                        if frac <= 0:
                            continue
                        lt = ld.get("type", "")
                        if lt == "floor_area":
                            area = ld.get("area_m2") or _story_plan_area(st)
                            w = abs(ld.get("qz_kN_m2", ld.get("value", 0))) * area
                        elif lt == "floor":
                            w = abs(ld.get("w_kN_m", ld.get("value", 0))) * ld.get("L_m", 1)
                        elif lt == "nodal":
                            w = abs(ld.get("Fz_kN", ld.get("value", 0)))
                        else:
                            w = 0.0
                        add[st - 1] += frac * w
                    if any(a > 0 for a in add):
                        sw = [sw[i] + add[i] for i in range(ns)]

            if sw:
                eigen_result = _run_eigen_analysis_v2(
                    model, E, G, section_cache, sw, story_nodes_map,
                )
                if eigen_result:
                    multi.modal_analysis = eigen_result
    except Exception as e:
        import traceback
        print(f"V2 modal analysis skipped: {e}")
        traceback.print_exc()

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

    # 1. 모델 재구축 (num_elements_per_member=1, rigid_diaphragm 시도)
    orig_num_elem = model.num_elements_per_member
    orig_rigid = model.rigid_diaphragm
    model.num_elements_per_member = 1

    # rigid diaphragm 시도, 실패 시 without
    nodes_3d = None
    for try_rigid in [True, False]:
        try:
            model.rigid_diaphragm = try_rigid
            nodes_3d, _ei, _mte, _mil, _bni, _nni = _build_model_v2(model, E, G, section_cache)
            break
        except Exception:
            continue

    model.num_elements_per_member = orig_num_elem
    model.rigid_diaphragm = orig_rigid

    if nodes_3d is None:
        return {}

    # 2. 질량 배정
    g_acc = 9810.0  # mm/s²
    node_map_3d = {n.id: n for n in nodes_3d}
    floor_masses = []
    node_mass: dict[int, float] = {}  # 절점기반 유효모달질량 계산용 (실제 배정 질량)
    # T3-A: 층별 rz 참여 기준점 = 질량중심(mm). 기존 master=snodes[len//2]는 짝수 격자에서
    # 기하중심을 벗어나 Σm·Δy≠0 → 순수병진모드에 허위 비틀림(Lrz≠0)이 섞여 누적 rz
    # 참여율이 100%를 넘는 비보수 결과를 냈다. 질량중심을 쓰면 병진모드 Lrz=0이 되고
    # I_eff(극관성)도 같은 기준으로 산정돼 모달 분해가 정합(누적 rz ≤100%).
    story_ref: dict[int, tuple] = {}

    for s in range(1, n_stories + 1):
        snodes = story_nodes_map.get(s, [])
        if not snodes:
            continue
        nodes_per_floor = len(snodes)

        W_N = story_weights_kN[s - 1] * 1000.0
        m_per_node = W_N / g_acc / nodes_per_floor
        m_floor = W_N / g_acc

        # 질량중심(질량 균일 → 좌표 산술평균). node_map_3d에 있는 절점만 사용.
        cxs = [node_map_3d[nid].x * 1000 for nid in snodes if nid in node_map_3d]
        cys = [node_map_3d[nid].y * 1000 for nid in snodes if nid in node_map_3d]
        cx_mm = sum(cxs) / len(cxs) if cxs else 0.0
        cy_mm = sum(cys) / len(cys) if cys else 0.0
        story_ref[s] = (cx_mm, cy_mm)

        I_eff = 0.0
        for nid in snodes:
            ops.mass(nid, m_per_node, m_per_node, 1e-6, 0.0, 0.0, 0.0)
            node_mass[nid] = node_mass.get(nid, 0.0) + m_per_node
            if nid in node_map_3d:
                dx = node_map_3d[nid].x * 1000 - cx_mm
                dy = node_map_3d[nid].y * 1000 - cy_mm
                I_eff += m_per_node * (dx ** 2 + dy ** 2)

        floor_masses.append((m_floor, I_eff))

    # 3. 모드 수 + 총 질량(참여질량 기준 — num_modes와 무관)
    if num_modes <= 0:
        num_modes = min(3 * n_stories, 15)
    mode_cap = min(3 * n_stories, _EIGEN_MODE_CAP)  # #11 자동증대 상한(면내 기저)

    total_mass_x = sum(fm[0] for fm in floor_masses)
    total_mass_rz = sum(fm[1] for fm in floor_masses)
    stories = model.story_heights

    def _solve_and_extract_v2(nm: int):
        """eigen(nm) 풀이 + 모드별 참여질량/형상 추출. 실패 시 None (#11)."""
        eigenvalues = None
        for solver in [lambda: ops.eigen(nm),
                       lambda: ops.eigen('-genBandArpack', nm),
                       lambda: ops.eigen('-fullGenLapack', nm)]:
            try:
                eigenvalues = solver()
                break
            except Exception:
                continue
        if not eigenvalues:
            return None

        modes_ = []
        cx = cy = crz = 0.0
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

            # 참여질량: 절점기반 유효모달질량 (V1 _run_eigen_analysis와 동일 정식화).
            # m_eff,dir = L_dir² / M_n,  M_n = Σ m·(φx²+φy²) (전체 일반화질량).
            # A4/T3-A: L_rz = Σ m·(Δx·φy − Δy·φx) (각 층 질량중심 기준)로 비틀림참여 산정.
            Lx = Ly = Lrz = M_n = 0.0
            for s in range(1, n_stories + 1):
                ref = story_ref.get(s)
                if ref is None:
                    continue
                mxx, myy = ref  # 질량중심(mm) 기준 (T3-A)
                for nid in story_nodes_map.get(s, []):
                    m_nd = node_mass.get(nid, 0.0)
                    if m_nd <= 0 or nid not in node_map_3d:
                        continue
                    ph = shape.get(str(nid)) or (0.0, 0.0, 0.0)
                    phx, phy = ph[0], ph[1]
                    M_n += m_nd * (phx * phx + phy * phy)
                    Lx += m_nd * phx
                    Ly += m_nd * phy
                    Lrz += m_nd * ((node_map_3d[nid].x * 1000 - mxx) * phy
                                   - (node_map_3d[nid].y * 1000 - myy) * phx)

            def _pct(L, tot):
                if M_n <= 1e-30 or tot <= 1e-30:
                    return 0.0
                return (L * L / M_n) / tot * 100

            mp_x = _pct(Lx, total_mass_x)
            mp_y = _pct(Ly, total_mass_x)  # total_mass_y == total_mass_x (병진질량)
            mp_rz = _pct(Lrz, total_mass_rz)

            # 지배 방향 (V1과 통일: TRAN-X / TRAN-Y / ROTN-Z).
            if mp_x >= mp_y and mp_x >= mp_rz:
                direction = "TRAN-X"
            elif mp_y >= mp_rz:
                direction = "TRAN-Y"
            elif mp_rz > 0:
                direction = "ROTN-Z"
            else:
                direction = "N/A"

            cx += mp_x
            cy += mp_y
            crz += mp_rz

            modes_.append({
                "mode_num": mode_num,
                "mode": mode_num,  # V1 호환
                "eigenvalue": round(lam, 4),
                "frequency_Hz": round(f, 4),
                "frequency_hz": round(f, 4),  # V1 호환
                "period_s": round(T, 4),
                "direction": direction,
                "mass_participation_x_pct": round(mp_x, 2),
                "mass_participation_y_pct": round(mp_y, 2),
                "mass_participation_rz_pct": round(mp_rz, 2),  # T3-B: rz 주기 산정용
                "mass_participation": {  # V1 호환 nested 형식
                    "x_pct": round(mp_x, 2),
                    "y_pct": round(mp_y, 2),
                    "rz_pct": round(mp_rz, 2),
                },
                "dominance_pct": round(max(mp_x, mp_y, mp_rz), 2),  # V1 호환
                "cumulative_x_pct": round(cx, 2),
                "cumulative_y_pct": round(cy, 2),
                "shape": shape,
            })

        return modes_, cx, cy, crz

    # 4+5. 초기 풀이 + 누적참여 <90% 시 모드 수 자동 증대 재시도 (#11)
    res = _solve_and_extract_v2(num_modes)
    if res is None or not res[0]:
        return {}
    n_retry = 0
    while (not (res[1] >= 90.0 and res[2] >= 90.0)
           and num_modes < mode_cap and n_retry < 4):
        num_modes = min(num_modes + max(n_stories, 3), mode_cap)
        nxt = _solve_and_extract_v2(num_modes)
        if nxt is None or not nxt[0]:
            break
        improved = (nxt[1] > res[1] + 0.5) or (nxt[2] > res[2] + 0.5)
        res = nxt
        n_retry += 1
        if not improved:   # 모드 추가가 누적참여를 개선 못하면 조기 종료
            break
    modes, cum_x, cum_y, cum_rz = res

    if not modes:
        return {}

    # 방향별 기본주기: 'X' in 'XY' 같은 부분문자열 매칭은 결합모드를 X·Y 양쪽에
    # 잡아 둘 다 같은(엉뚱한) 주기를 돌려준다. 해당 방향 질량참여가 최대인 모드를
    # 그 방향 기본모드로 본다(결합모드는 우세 방향에만 귀속).
    def _t1_dir(key: str):
        cand = [m for m in modes if m.get(key, 0) > 0]
        return max(cand, key=lambda m: m.get(key, 0))["period_s"] if cand else None

    t1x = _t1_dir("mass_participation_x_pct")
    t1y = _t1_dir("mass_participation_y_pct")
    t1rz = _t1_dir("mass_participation_rz_pct")  # T3-B: 비틀림(rz) 1차주기

    # A3: 누적 유효질량 참여율 충분조건 (KDS 41 17 00 ≥90%) — V1과 동형.
    sufficient = cum_x >= 90.0 and cum_y >= 90.0

    return {
        "num_modes": len(modes),
        "mode_count_auto_increased": n_retry > 0,  # #11
        "modes": modes,
        "fundamental_periods": {
            "T1_x": t1x,
            "T1_y": t1y,
            # V1 호환 키 (리포트/interpreter에서 사용)
            "T1_x_s": t1x,
            "T1_y_s": t1y,
            "T1_rz_s": t1rz,  # T3-B: V1과 동일 키 (구버전 V2는 누락)
        },
        "cumulative_participation": {
            "x_pct": round(cum_x, 1),
            "y_pct": round(cum_y, 1),
            "rz_pct": round(cum_rz, 1),
            "sufficient_90pct": sufficient,
        },
    }
