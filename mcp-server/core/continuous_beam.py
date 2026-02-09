"""
다경간 연속보 해석 모듈
OpenSeesPy를 사용한 2D 연속보 정적 해석

===================================================================================
SIGN CONVENTION (OpenSees Local Coordinate System)
===================================================================================

This module stores internal forces in OpenSees local coordinate convention:
- eleForce(tag) returns [N_i, V_i, M_i, N_j, V_j, M_j]
- These are nodal equilibrating forces (reaction forces on nodes)

For visualization, the sign_convention.py module transforms to textbook convention:
- Shear V: + upward on left cut face
- Moment M: + sagging (tension at bottom)

Transformation (applied in visualization.py):
    V_textbook = -V_opensees
    M_textbook = -M_opensees

NOTE: The ContinuousBeamResult.moments and .shears arrays store OpenSees convention.
      Transformation is applied only at the visualization layer for plotting.
===================================================================================
"""

import openseespy.opensees as ops
from dataclasses import dataclass, field
from typing import Optional

from core.simple_beam import (
    get_section_from_db,
    get_material_from_db,
    DEFAULT_SECTIONS,
    DEFAULT_MATERIALS,
    get_available_sections,
    get_available_materials,
)


@dataclass
class ContinuousBeamResult:
    """연속보 해석 결과"""
    num_spans: int
    total_length: float                         # m
    spans: list[float] = field(default_factory=list)
    supports: list[str] = field(default_factory=list)

    # 전체 최대값
    max_displacement: float = 0.0               # mm
    max_displacement_location: float = 0.0      # m
    max_moment: float = 0.0                     # kN·m
    max_moment_location: float = 0.0            # m
    max_shear: float = 0.0                      # kN
    max_shear_location: float = 0.0             # m
    max_stress: float = 0.0                     # MPa

    # 지점별 반력
    reactions: list[dict] = field(default_factory=list)

    # 경간별 결과
    span_results: list[dict] = field(default_factory=list)

    # 단면/재료 정보
    section_name: str = ""
    material_name: str = ""

    # 시각화용 배열
    node_positions: list[float] = field(default_factory=list)  # m
    displacements: list[float] = field(default_factory=list)   # mm
    rotations: list[float] = field(default_factory=list)       # rad (RZ)
    moments: list[float] = field(default_factory=list)         # kN·m
    shears: list[float] = field(default_factory=list)          # kN
    # 하중 정보 (시각화용) — 전체 좌표 기준
    load_info: list[dict] = field(default_factory=list)
    # 새깅/호깅 분리
    max_moment_positive: float = 0.0       # M+(sagging) kN·m
    max_moment_positive_location: float = 0.0
    max_moment_negative: float = 0.0       # M-(hogging) kN·m (음수)
    max_moment_negative_location: float = 0.0
    # 해석 메타정보
    E_MPa: float = 0.0
    Ix_mm4: float = 0.0
    h_mm: float = 0.0
    Zx_mm3: float = 0.0
    fy_MPa: float = 0.0
    num_elements_per_span: int = 0
    deflection_limit_ratio: int = 300  # 허용처짐 기준 분모 (L/N)
    # 힌지 정보
    hinges: list[int] = field(default_factory=list)  # 힌지가 있는 지점 인덱스
    hinge_locations: list[float] = field(default_factory=list)  # 힌지 위치 (m)


def analyze_continuous_beam(
    spans: list[float],
    loads: list[dict],
    supports: list[str] | None = None,
    hinges: list[int] | None = None,
    section_name: str = "H-400x200x8x13",
    material_name: str = "SS275",
    num_elements_per_span: int = 20,
    deflection_limit: int = 300,
) -> ContinuousBeamResult:
    """
    다경간 연속보 해석

    Parameters
    ----------
    spans : list[float]
        각 경간 길이 (m), 예: [6, 8, 6]
    loads : list[dict]
        하중 리스트. 각 항목:
        - span_index (int, optional): 적용 경간 (0-based). 생략 시 전 경간
        - type: "uniform", "point", "triangular", "partial_uniform"
        - value: 하중 크기 (kN/m 또는 kN)
        - location (float, optional): 집중하중 위치 (경간 내 m)
        - value_end (float, optional): 삼각분포 끝단값
        - start, end (float, optional): 부분등분포 구간 (경간 내 m)
    supports : list[str], optional
        지점 조건 리스트 (len = len(spans)+1)
        "pin", "roller", "fixed", "free"
        기본: ["pin"] + ["pin"]*(n-1) + ["roller"] 의 마지막을 roller
    hinges : list[int], optional
        내부 힌지를 추가할 지점 인덱스 리스트 (1-based, 중간 지점만 허용)
        예: [1] → 지점 B에 힌지, [1, 2] → 지점 B, C에 힌지
        힌지가 있으면 해당 위치에서 모멘트 전달 안 됨 (회전 자유)
    section_name : str
    material_name : str
    num_elements_per_span : int

    Returns
    -------
    ContinuousBeamResult
    """
    n_spans = len(spans)
    if n_spans < 2 or n_spans > 5:
        raise ValueError(f"경간 수는 2~5개여야 합니다. 입력: {n_spans}")

    n_supports = n_spans + 1

    # 기본 지점 조건
    if supports is None:
        supports = ["pin"] * n_supports
        supports[-1] = "roller"

    if len(supports) != n_supports:
        raise ValueError(f"지점 수({len(supports)})가 경간 수+1({n_supports})과 불일치")

    # 힌지 검증
    if hinges is None:
        hinges = []
    hinge_set = set()
    for h in hinges:
        if h < 1 or h >= n_supports - 1:
            raise ValueError(f"힌지 인덱스 {h}는 중간 지점(1~{n_supports-2})에만 허용됩니다.")
        hinge_set.add(h)

    # 단면/재료
    section = get_section_from_db(section_name)
    if section is None:
        section = DEFAULT_SECTIONS.get(section_name)
    if section is None:
        raise ValueError(f"Unknown section: {section_name}")

    material = get_material_from_db(material_name)
    if material is None:
        material = DEFAULT_MATERIALS.get(material_name)
    if material is None:
        raise ValueError(f"Unknown material: {material_name}")

    # OpenSees 모델 초기화
    ops.wipe()
    ops.model('basic', '-ndm', 2, '-ndf', 3)

    # 노드 생성
    # 각 경간을 num_elements_per_span개로 분할
    # support_node_pairs[i] = (left_node, right_node) for i-th support
    #   - 힌지 없으면 left_node == right_node
    #   - 힌지 있으면 left_node != right_node (같은 좌표, equalDOF로 연결)
    node_id = 1
    x_pos = 0.0  # mm
    support_node_pairs = [(1, 1)]  # 첫 지점: 노드 1
    support_x_positions = [0.0]    # 각 지점의 x 좌표 (mm)
    span_element_ranges = []  # [(start_elem_id, end_elem_id), ...] per span

    ops.node(1, 0.0, 0.0)

    for span_idx, span_len in enumerate(spans):
        dx = span_len * 1000 / num_elements_per_span  # mm
        start_node = node_id + 1
        for j in range(num_elements_per_span):
            node_id += 1
            x_pos += dx
            ops.node(node_id, x_pos, 0.0)
        end_node = node_id

        # 경간 끝 = 다음 지점
        support_idx = span_idx + 1
        support_x_positions.append(x_pos)

        if support_idx in hinge_set:
            # 힌지 지점: 같은 좌표에 추가 노드 생성
            node_id += 1
            hinge_right_node = node_id
            ops.node(hinge_right_node, x_pos, 0.0)
            support_node_pairs.append((end_node, hinge_right_node))
        else:
            support_node_pairs.append((end_node, end_node))

        # 경간별 요소 범위는 나중에 계산
        span_element_ranges.append((start_node - 1, end_node - 1))  # placeholder

    total_nodes = node_id

    # 기존 방식과 호환을 위해 support_nodes 리스트 생성 (left node 기준)
    support_nodes = [pair[0] for pair in support_node_pairs]

    # 경계조건
    # 첫 지점은 수평 구속 보장 (안정성)
    for i, (st, (left_node, right_node)) in enumerate(zip(supports, support_node_pairs)):
        # 힌지 지점에서는 left_node에 경계조건 적용
        sn = left_node
        if st == "pin":
            if i == 0:
                ops.fix(sn, 1, 1, 0)   # 첫 핀: 수평+수직
            else:
                ops.fix(sn, 0, 1, 0)   # 중간/끝 핀: 수직만 (수평 자유)
        elif st == "roller":
            ops.fix(sn, 0, 1, 0)       # 수직만
        elif st == "fixed":
            ops.fix(sn, 1, 1, 1)       # 모두 구속
        # "free" → 구속 없음

    # 힌지: equalDOF로 변위만 연결 (회전 독립)
    hinge_locations = []
    for i in hinge_set:
        left_node, right_node = support_node_pairs[i]
        if left_node != right_node:
            ops.equalDOF(left_node, right_node, 1, 2)  # DOF 1(x), 2(y) 연결
            hinge_locations.append(support_x_positions[i] / 1000)  # mm → m

    # 기하변환
    ops.geomTransf('Linear', 1)

    # 요소 생성
    E = material.E
    A = section.A
    I = section.I

    # 요소 생성 - 힌지 처리 포함
    eid = 0
    span_element_ranges = []
    for span_idx, span_len in enumerate(spans):
        # 경간 시작 노드 결정 (이전 지점의 right_node)
        if span_idx == 0:
            prev_node = 1
        else:
            _, prev_node = support_node_pairs[span_idx]  # 이전 지점의 right_node

        # 경간 끝 노드 (현재 지점의 left_node)
        end_left_node, _ = support_node_pairs[span_idx + 1]

        start_elem = eid + 1
        dx = span_len * 1000 / num_elements_per_span
        for j in range(num_elements_per_span):
            eid += 1
            next_node = prev_node + 1
            # 마지막 요소는 end_left_node에 연결
            if j == num_elements_per_span - 1:
                next_node = end_left_node
            ops.element('elasticBeamColumn', eid, prev_node, next_node, A, E, I, 1)
            prev_node = next_node
        end_elem = eid
        span_element_ranges.append((start_elem, end_elem))

    total_elements = eid

    # 하중 적용
    ops.timeSeries('Linear', 1)
    ops.pattern('Plain', 1, 1)

    # 집중하중이 적용된 노드 집합 (SFD 불연속점 처리용)
    point_load_nodes = set()

    for ld in loads:
        ld_type = ld.get("type", "uniform")
        ld_value = ld.get("value", 0.0)
        span_index = ld.get("span_index", None)

        # 적용 대상 경간 결정
        if span_index is not None:
            target_spans = [span_index]
        else:
            target_spans = list(range(n_spans))

        for si in target_spans:
            if si < 0 or si >= n_spans:
                continue
            start_elem, end_elem = span_element_ranges[si]
            n_elem = num_elements_per_span
            span_len = spans[si]
            dx = span_len * 1000 / n_elem  # mm

            if ld_type == "uniform":
                w = ld_value * 1000 / 1000  # kN/m → N/mm
                for eid in range(start_elem, end_elem + 1):
                    ops.eleLoad('-ele', eid, '-type', '-beamUniform', -w)

            elif ld_type == "point":
                loc = ld.get("location", span_len / 2)  # 경간 내 위치 (m)
                loc_mm = loc * 1000
                # 경간 시작 노드 기준으로 노드 인덱스 계산
                span_start_node = support_nodes[si]
                local_node_offset = int(loc_mm / dx)
                local_node_offset = max(0, min(local_node_offset, n_elem))
                target_node = span_start_node + local_node_offset
                P = ld_value * 1000  # kN → N
                ops.load(target_node, 0, -P, 0)
                # SFD 불연속점 처리를 위해 노드 추적
                point_load_nodes.add(target_node)

            elif ld_type == "triangular":
                w_start = ld_value
                w_end = ld.get("value_end", 0.0)
                span_mm = span_len * 1000
                for j, eid in enumerate(range(start_elem, end_elem + 1)):
                    x_mid = (j + 0.5) * dx
                    ratio = x_mid / span_mm
                    w_local = w_start + (w_end - w_start) * ratio
                    w_nmm = w_local * 1000 / 1000
                    ops.eleLoad('-ele', eid, '-type', '-beamUniform', -w_nmm)

            elif ld_type == "partial_uniform":
                a_mm = ld.get("start", 0.0) * 1000
                b_mm = ld.get("end", span_len) * 1000
                w = ld_value * 1000 / 1000
                for j, eid in enumerate(range(start_elem, end_elem + 1)):
                    x1 = j * dx
                    x2 = (j + 1) * dx
                    if x2 > a_mm and x1 < b_mm:
                        ops.eleLoad('-ele', eid, '-type', '-beamUniform', -w)

    # 해석
    ops.system('BandGen')
    ops.numberer('Plain')
    ops.constraints('Plain')
    ops.integrator('LoadControl', 1.0)
    ops.algorithm('Linear')
    ops.analysis('Static')
    ops.analyze(1)
    ops.reactions()

    # 결과 추출
    # 전체 변위
    all_disps = []
    for i in range(total_nodes):
        all_disps.append(ops.nodeDisp(i + 1, 2))

    min_disp_idx = all_disps.index(min(all_disps))
    global_max_disp = abs(min(all_disps))  # mm
    # 전역 좌표 계산
    # 노드 i+1의 x 좌표
    global_max_disp_loc = ops.nodeCoord(min_disp_idx + 1, 1) / 1000  # mm → m

    # 지점별 반력
    reactions_list = []
    for i, sn in enumerate(support_nodes):
        x_loc = ops.nodeCoord(sn, 1) / 1000  # m
        vert = abs(ops.nodeReaction(sn, 2)) / 1000  # N → kN
        moment = abs(ops.nodeReaction(sn, 3)) / 1e6 if supports[i] == "fixed" else 0.0  # N·mm → kN·m
        is_hinge = i in hinge_set
        reactions_list.append({
            "support_index": i,
            "location": round(x_loc, 3),
            "type": supports[i],
            "vertical_kN": round(vert, 2),
            "moment_kNm": round(moment, 2),
            "has_hinge": is_hinge,
        })

    # 요소력에서 모멘트/전단력 추출
    all_moments = []
    all_moment_locs = []
    all_shears = []
    all_shear_locs = []

    eid = 0
    node_cursor = 1
    for span_idx, span_len in enumerate(spans):
        dx = span_len * 1000 / num_elements_per_span
        for j in range(num_elements_per_span):
            eid += 1
            forces = ops.eleForce(eid)
            x_start = ops.nodeCoord(node_cursor, 1) / 1000  # m
            x_end = ops.nodeCoord(node_cursor + 1, 1) / 1000
            all_shears.extend([abs(forces[1]) / 1000, abs(forces[4]) / 1000])
            all_shear_locs.extend([x_start, x_end])
            all_moments.extend([abs(forces[2]) / 1e6, abs(forces[5]) / 1e6])
            all_moment_locs.extend([x_start, x_end])
            node_cursor += 1

    global_max_moment = max(all_moments) if all_moments else 0.0
    max_moment_idx = all_moments.index(global_max_moment) if all_moments else 0
    global_max_moment_loc = all_moment_locs[max_moment_idx] if all_moment_locs else 0.0

    global_max_shear = max(all_shears) if all_shears else 0.0
    max_shear_idx = all_shears.index(global_max_shear) if all_shears else 0
    global_max_shear_loc = all_shear_locs[max_shear_idx] if all_shear_locs else 0.0

    # 최대 응력
    y_max = section.h / 2
    global_max_stress = (global_max_moment * 1e6) * y_max / section.I  # MPa

    # 경간별 결과
    span_results_list = []
    for span_idx in range(n_spans):
        start_elem, end_elem = span_element_ranges[span_idx]
        n_elem = end_elem - start_elem + 1

        # 경간 내 변위
        span_start_node = support_nodes[span_idx]
        span_end_node = support_nodes[span_idx + 1]
        span_disps = [ops.nodeDisp(n, 2) for n in range(span_start_node, span_end_node + 1)]
        span_max_disp = abs(min(span_disps))
        span_min_idx = span_disps.index(min(span_disps))
        span_max_disp_loc = ops.nodeCoord(span_start_node + span_min_idx, 1) / 1000

        # 경간 내 요소력
        span_moments = []
        span_shears = []
        for eid in range(start_elem, end_elem + 1):
            forces = ops.eleForce(eid)
            span_shears.extend([abs(forces[1]) / 1000, abs(forces[4]) / 1000])
            span_moments.extend([abs(forces[2]) / 1e6, abs(forces[5]) / 1e6])

        span_max_moment = max(span_moments) if span_moments else 0.0
        span_max_shear = max(span_shears) if span_shears else 0.0

        # 경간별 새깅/호깅 (부호 있는 값으로 추출)
        span_signed_moments = []
        for eid in range(start_elem, end_elem + 1):
            forces = ops.eleForce(eid)
            span_signed_moments.append(forces[2] / 1e6)
            span_signed_moments.append(-forces[5] / 1e6)
        # 부호 반전: 배열 음수 = sagging(M+), 배열 양수 = hogging(M-)
        span_sagging = [-m for m in span_signed_moments if m < 0]
        span_hogging = [-m for m in span_signed_moments if m > 0]
        span_m_pos = max(span_sagging) if span_sagging else 0.0  # M+ (양수)
        span_m_neg = min(span_hogging) if span_hogging else 0.0  # M- (음수)

        span_x_start = ops.nodeCoord(span_start_node, 1) / 1000

        # 처짐 판정 (경간별 L/N 기준)
        delta_allow = spans[span_idx] * 1000 / deflection_limit  # mm
        defl_status = "OK" if span_max_disp <= delta_allow else "NG"

        span_results_list.append({
            "span_index": span_idx,
            "span_length": spans[span_idx],
            "start_location": round(span_x_start, 3),
            "max_displacement_mm": round(span_max_disp, 3),
            "max_displacement_location_m": round(span_max_disp_loc, 3),
            "max_moment_kNm": round(span_max_moment, 2),
            "max_moment_positive_kNm": round(span_m_pos, 2),
            "max_moment_negative_kNm": round(span_m_neg, 2),
            "max_shear_kN": round(span_max_shear, 2),
            "delta_allow_mm": round(delta_allow, 1),
            "deflection_status": defl_status,
        })

    # 시각화용 배열
    # 중간 지점에서 전단력 불연속(수직 점프)을 표현하기 위해
    # 지점 노드에서 x좌표를 중복하여 두 개의 전단값(좌/우)을 저장
    # 힌지 지점에서는 left_node만 중간 지점으로 처리 (right_node는 다음 경간의 시작 노드)
    intermediate_support_set = set()
    for i in range(1, n_supports - 1):
        left_node, _ = support_node_pairs[i]
        intermediate_support_set.add(left_node)

    viz_node_positions = []
    viz_displacements = []
    viz_rotations = []
    viz_moments = []
    viz_shears = []

    # 요소별 i-end, j-end 값을 노드 기준으로 수집
    # node_shear_left[nid] = j-end of element ending at nid (좌측에서 오는 값)
    # node_shear_right[nid] = i-end of element starting at nid (우측으로 가는 값)
    node_moment_left = {}
    node_moment_right = {}
    node_shear_left = {}
    node_shear_right = {}

    # 요소의 실제 노드 ID를 추적 (힌지로 인해 연속적이지 않을 수 있음)
    for eid in range(1, total_elements + 1):
        nodes = ops.eleNodes(eid)
        i_node, j_node = nodes[0], nodes[1]
        forces = ops.eleForce(eid)
        m_i = forces[2] / 1e6
        m_j = -forces[5] / 1e6
        # 전단력: eleForce → 내부 전단력 변환
        # i-end: 부호 반전 필요 (eleForce는 노드에 가하는 힘)
        # j-end: 부호 유지
        v_i = -forces[1] / 1000
        v_j = forces[4] / 1000

        # i-node의 right 값 (요소 시작점에서 오른쪽으로 가는 값)
        # 항상 덮어쓰기: 집중하중 노드에서 올바른 불연속점 처리를 위해
        node_shear_right[i_node] = v_i
        node_moment_right[i_node] = m_i

        # j-node의 left 값 (요소 끝단에서 오는 값)
        node_shear_left[j_node] = v_j
        node_moment_left[j_node] = m_j

        # j-node의 right 값이 없으면 left 값으로 초기화
        if j_node not in node_shear_right:
            node_shear_right[j_node] = v_j
            node_moment_right[j_node] = m_j

    # 경간별 첫 요소의 i-end로 right 값 덮어쓰기 (힌지 지점의 right_node 처리)
    for span_idx in range(n_spans):
        start_elem, _ = span_element_ranges[span_idx]
        nodes = ops.eleNodes(start_elem)
        i_node = nodes[0]
        forces = ops.eleForce(start_elem)
        v_i = -forces[1] / 1000
        m_i = forces[2] / 1e6
        node_shear_right[i_node] = v_i
        node_moment_right[i_node] = m_i

    # 힌지 right_node 집합 (시각화에서 건너뛸 노드)
    hinge_right_nodes = set()
    for i in hinge_set:
        _, right_node = support_node_pairs[i]
        hinge_right_nodes.add(right_node)

    # 불연속점 노드 집합: 내부 지점 + 집중하중 노드
    # 이 노드들에서는 좌측값, 우측값을 각각 삽입하여 SFD에서 수직 점프 표현
    discontinuity_nodes = intermediate_support_set | point_load_nodes

    # 배열 구성: 불연속점에서 x좌표 중복, 힌지 right_node는 건너뜀
    for nid in range(1, total_nodes + 1):
        if nid in hinge_right_nodes:
            continue  # 힌지 right_node는 건너뜀 (같은 좌표의 중복 노드)

        x = ops.nodeCoord(nid, 1) / 1000
        d = ops.nodeDisp(nid, 2)
        r = ops.nodeDisp(nid, 3)  # 회전각 (rad)

        if nid in discontinuity_nodes:
            # 불연속점: 좌측값, 우측값 두 개 삽입하여 수직 점프 표현
            # 내부 지점 힌지인 경우 특수 처리
            is_hinge_left = False
            if nid in intermediate_support_set:
                for i in hinge_set:
                    left_node, right_node = support_node_pairs[i]
                    if nid == left_node:
                        is_hinge_left = True
                        m_right = node_moment_right.get(right_node, 0.0)
                        v_right = node_shear_right.get(right_node, 0.0)
                        r_right = ops.nodeDisp(right_node, 3)
                        break

            # 첫 번째 점: 좌측 값 (좌측 요소의 j-end)
            viz_node_positions.append(x)
            viz_displacements.append(d)
            viz_rotations.append(r)
            viz_moments.append(node_moment_left.get(nid, 0.0))
            viz_shears.append(node_shear_left.get(nid, 0.0))

            # 두 번째 점: 우측 값 (우측 요소의 i-end)
            viz_node_positions.append(x)
            viz_displacements.append(d)
            if is_hinge_left:
                viz_rotations.append(r_right)
                viz_moments.append(m_right)
                viz_shears.append(v_right)
            else:
                viz_rotations.append(r)
                viz_moments.append(node_moment_right.get(nid, 0.0))
                viz_shears.append(node_shear_right.get(nid, 0.0))
        else:
            # 일반 노드: 단일 값
            viz_node_positions.append(x)
            viz_displacements.append(d)
            viz_rotations.append(r)
            viz_moments.append(node_moment_left.get(nid, node_moment_right.get(nid, 0.0)))
            viz_shears.append(node_shear_left.get(nid, node_shear_right.get(nid, 0.0)))

    # 내부지점 좌/우 모멘트를 reactions_list에 추가
    # 비대칭 하중에서 내부지점 좌/우 모멘트가 다를 수 있음.
    # 연속보의 내부지점에서 좌측 경간 끝단(j-end)과 우측 경간 시작단(i-end)의
    # 휨모멘트는 평형 조건상 같아야 하나, 수치해석에서는 미세한 차이 발생 가능.
    # 힌지 지점에서는 모멘트가 0에 가까워야 함.
    for i, (left_node, right_node) in enumerate(support_node_pairs):
        # 부호 반전: OpenSees 배열 음수 = sagging(M+), 양수 = hogging(M-)
        m_left = -node_moment_left.get(left_node, 0.0)   # 구조공학 관례로 변환
        # 힌지인 경우 right_node에서 right 값을 가져옴
        if i in hinge_set and left_node != right_node:
            m_right = -node_moment_right.get(right_node, 0.0)
        else:
            m_right = -node_moment_right.get(left_node, 0.0)
        reactions_list[i]["moment_left_kNm"] = round(m_left, 2)
        reactions_list[i]["moment_right_kNm"] = round(m_right, 2)

    # 전체 새깅/호깅 분리 (부호 있는 viz_moments에서 추출)
    # OpenSees 2D beamColumn: 배열 음수 = sagging(하부인장), 배열 양수 = hogging(상부인장)
    # 구조공학 관례: M+ = sagging, M- = hogging → 부호 반전
    sagging = [(-m, viz_node_positions[i]) for i, m in enumerate(viz_moments) if m < 0]
    hogging = [(-m, viz_node_positions[i]) for i, m in enumerate(viz_moments) if m > 0]
    global_max_m_pos = max(sagging, key=lambda t: t[0]) if sagging else (0.0, 0.0)
    global_max_m_neg = min(hogging, key=lambda t: t[0]) if hogging else (0.0, 0.0)

    # 하중 정보 구성 (시각화용 — 전체 좌표 변환)
    _load_info = []
    for ld in loads:
        ld_type = ld.get("type", "uniform")
        ld_value = ld.get("value", 0.0)
        span_index = ld.get("span_index", None)
        target_spans = [span_index] if span_index is not None else list(range(n_spans))
        for si in target_spans:
            if si < 0 or si >= n_spans:
                continue
            span_start = sum(spans[:si])
            span_len = spans[si]
            if ld_type == "uniform":
                _load_info.append({"type": "uniform", "value": ld_value, "start": span_start, "end": span_start + span_len})
            elif ld_type == "point":
                loc = ld.get("location", span_len / 2)
                _load_info.append({"type": "point", "value": ld_value, "location": span_start + loc})
            elif ld_type == "triangular":
                _load_info.append({"type": "triangular", "value": ld_value, "value_end": ld.get("value_end", 0.0), "start": span_start, "end": span_start + span_len})
            elif ld_type == "partial_uniform":
                s = ld.get("start", 0.0)
                e = ld.get("end", span_len)
                _load_info.append({"type": "uniform", "value": ld_value, "start": span_start + s, "end": span_start + e})

    return ContinuousBeamResult(
        num_spans=n_spans,
        total_length=sum(spans),
        spans=spans,
        supports=supports,
        section_name=section_name,
        material_name=material_name,
        max_displacement=global_max_disp,
        max_displacement_location=global_max_disp_loc,
        max_moment=global_max_moment,
        max_moment_location=global_max_moment_loc,
        max_shear=global_max_shear,
        max_shear_location=global_max_shear_loc,
        max_stress=global_max_stress,
        reactions=reactions_list,
        span_results=span_results_list,
        load_info=_load_info,
        max_moment_positive=global_max_m_pos[0],
        max_moment_positive_location=global_max_m_pos[1],
        max_moment_negative=global_max_m_neg[0],
        max_moment_negative_location=global_max_m_neg[1],
        E_MPa=material.E,
        Ix_mm4=section.I,
        h_mm=section.h,
        Zx_mm3=section.Zx,
        fy_MPa=material.fy,
        num_elements_per_span=num_elements_per_span,
        deflection_limit_ratio=deflection_limit,
        node_positions=viz_node_positions,
        displacements=viz_displacements,
        rotations=viz_rotations,
        moments=viz_moments,
        shears=viz_shears,
        hinges=list(hinge_set),
        hinge_locations=sorted(hinge_locations),
    )
