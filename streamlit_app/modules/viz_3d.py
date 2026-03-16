"""3D 시각화 모듈 (Plotly).

opensees-ai-agent/app/plots/model_viz.py 및 model_defo.py에서
검증된 Mesh3d 패턴을 포팅하고, Frame3DMultiCaseResult 데이터에 적용.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# --- 색상 팔레트 ---
COLUMN_COLOR = "#4A90D9"      # 스틸 블루
BEAM_X_COLOR = "#E8524A"      # 코랄 레드
BEAM_Y_COLOR = "#50C878"      # 에메랄드 그린
SUPPORT_COLOR = "#333333"     # 지점
DEFORM_COLORSCALE = "Jet_r"   # 변형
FORCE_COLORSCALE = "RdYlBu"   # 부재력

TYPE_COLORS = {
    "column": COLUMN_COLOR,
    "beam_x": BEAM_X_COLOR,
    "beam_y": BEAM_Y_COLOR,
}

TYPE_LABELS = {
    "column": "Column",
    "beam_x": "Beam (X)",
    "beam_y": "Beam (Y)",
}

Vec3 = np.ndarray


# ============================================================
# 기하 프리미티브 (opensees-ai-agent에서 포팅)
# ============================================================

def compute_beam_vertices_rect(A: Vec3, B: Vec3, width: float, height: float) -> np.ndarray:
    """두 점 A, B 사이의 직육면체 프리즘 8개 꼭짓점 계산.

    opensees-ai-agent/app/plots/model_viz.py:14-43 포팅.
    """
    v = B - A
    length = np.linalg.norm(v)
    if length == 0:
        return np.array([A] * 8)
    v_hat = v / length

    axes = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
    helper = min(axes, key=lambda ax: abs(np.dot(v_hat, ax)))

    local_y = np.cross(v_hat, helper)
    local_y /= np.linalg.norm(local_y)
    local_z = np.cross(v_hat, local_y)
    local_z /= np.linalg.norm(local_z)

    local_y *= width / 2.0
    local_z *= height / 2.0

    v0 = A + local_y + local_z
    v1 = A + local_y - local_z
    v2 = A - local_y - local_z
    v3 = A - local_y + local_z
    v4 = B + local_y + local_z
    v5 = B + local_y - local_z
    v6 = B - local_y - local_z
    v7 = B - local_y + local_z
    return np.stack([v0, v1, v2, v3, v4, v5, v6, v7])


def add_beam_mesh(
    fig: go.Figure,
    verts: np.ndarray,
    color: str,
    hover_text: str = "",
    opacity: float = 1.0,
) -> None:
    """직육면체 프리즘을 Mesh3d로 추가.

    opensees-ai-agent/app/plots/model_viz.py:45-87 포팅.
    6개 쿼드 → 12개 삼각형 + 12개 역면 = 24 삼각형.
    """
    quads = [
        (0, 1, 2, 3),  # face at A
        (4, 5, 6, 7),  # face at B
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ]
    i_list, j_list, k_list = [], [], []
    for a, b, c, d in quads:
        i_list.extend([a, a, a, a])
        j_list.extend([b, c, c, d])
        k_list.extend([c, d, b, c])

    fig.add_trace(
        go.Mesh3d(
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            i=i_list,
            j=j_list,
            k=k_list,
            color=color,
            flatshading=False,
            opacity=opacity,
            hovertext=hover_text,
            hoverinfo="text" if hover_text else "skip",
            lighting=dict(ambient=0.5, diffuse=0.7, specular=0.3, roughness=0.9),
            showscale=False,
            showlegend=False,
        )
    )


def _add_bounding_cube(fig: go.Figure, nodes: list[dict]) -> None:
    """투명 바운딩 큐브로 aspectmode='data' 안정화.

    opensees-ai-agent/app/plots/model_viz.py:168-193 포팅.
    """
    xs = [n["x_m"] for n in nodes]
    ys = [n["y_m"] for n in nodes]
    zs = [n["z_m"] for n in nodes]

    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    cz = (min(zs) + max(zs)) / 2
    half = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)) / 2.0
    if half == 0:
        half = 1.0

    xb = [cx - half, cx + half]
    yb = [cy - half, cy + half]
    zb = [cz - half, cz + half]

    fig.add_trace(go.Scatter3d(
        x=[xb[0], xb[1], xb[0], xb[1], xb[0], xb[1], xb[0], xb[1]],
        y=[yb[0], yb[0], yb[1], yb[1], yb[0], yb[0], yb[1], yb[1]],
        z=[zb[0], zb[0], zb[0], zb[0], zb[1], zb[1], zb[1], zb[1]],
        mode="markers",
        marker=dict(size=0, color="rgba(0,0,0,0)"),
        showlegend=False,
        hoverinfo="skip",
    ))


def _display_dims(h_mm: float, b_mm: float, min_span_m: float) -> tuple[float, float]:
    """가시성을 위한 단면 표시 크기 스케일링.

    실제 단면(mm)을 건물 스케일(m)에서 볼 수 있도록 조정.
    """
    h_m = h_mm / 1000.0
    b_m = b_mm / 1000.0
    target = min_span_m * 0.04  # 최소 경간의 4%
    if h_m < target:
        factor = target / h_m
        h_m *= factor
        b_m *= factor
    return h_m, b_m


def _get_default_layout() -> dict:
    """3D 시각화 기본 레이아웃."""
    return dict(
        scene=dict(
            aspectmode="data",
            xaxis_visible=False,
            yaxis_visible=False,
            zaxis_visible=False,
            bgcolor="white",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
        ),
        paper_bgcolor="white",
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(
            x=0.95, y=0.05,
            xanchor="right", yanchor="bottom",
            bgcolor="rgba(255,255,255,0.85)",
            borderwidth=1,
            itemsizing="constant",
            font=dict(size=14),
        ),
    )


def _build_node_map(nodes: list[dict]) -> dict[int, dict]:
    """노드 ID → 좌표 매핑."""
    return {n["id"]: n for n in nodes}


def _get_min_span(result) -> float:
    """결과에서 최소 경간 추출."""
    spans = []
    if hasattr(result, "bays_x") and result.bays_x:
        spans.extend(result.bays_x)
    if hasattr(result, "bays_y") and result.bays_y:
        spans.extend(result.bays_y)
    if hasattr(result, "stories") and result.stories:
        spans.extend(result.stories)
    return min(spans) if spans else 3.0


# ============================================================
# 메인 시각화 함수
# ============================================================

def create_3d_model_figure(result) -> go.Figure:
    """비변형 3D 모델 시각화.

    Args:
        result: Frame3DMultiCaseResult (nodes, elements, member_info 포함)

    Returns:
        Plotly Figure with Mesh3d beams/columns
    """
    fig = go.Figure()
    nodes = result.nodes
    node_map = _build_node_map(nodes)
    min_span = _get_min_span(result)

    # 바운딩 큐브
    _add_bounding_cube(fig, nodes)

    # 단면 치수
    section_dims = {}
    for sec_attr, sec_type in [
        ("column_section", "column"),
        ("beam_x_section", "beam_x"),
        ("beam_y_section", "beam_y"),
    ]:
        sec_name = getattr(result, sec_attr, "")
        h_attr = f"{sec_type.replace('beam_', 'beam_' if 'beam' in sec_type else '')}"
        # 실제 단면 높이/폭은 section_3d에서 가져오는 것이 이상적이지만
        # result에 저장된 값 사용
        h_key = f"{'column' if sec_type == 'column' else sec_type}_Ix_mm4"
        # 단순하게 h_mm, b_mm 계산 (result에서)
        if sec_type == "column":
            # H-NxN 형식에서 추출
            h_mm, b_mm = _parse_section_dims(sec_name)
        else:
            h_mm, b_mm = _parse_section_dims(sec_name)
        disp_h, disp_b = _display_dims(h_mm, b_mm, min_span)
        section_dims[sec_type] = (disp_h, disp_b)

    # 부재 그리기
    drawn_types = set()
    for mem in result.member_info:
        ni_id = mem["ni"]
        nj_id = mem["nj"]
        etype = mem["type"]  # "column", "beam_x", "beam_y"
        color = TYPE_COLORS.get(etype, "#999999")

        ni = node_map.get(ni_id)
        nj = node_map.get(nj_id)
        if ni is None or nj is None:
            continue

        A = np.array([ni["x_m"], ni["y_m"], ni["z_m"]], dtype=float)
        B = np.array([nj["x_m"], nj["y_m"], nj["z_m"]], dtype=float)

        disp_h, disp_b = section_dims.get(etype, (0.2, 0.2))
        verts = compute_beam_vertices_rect(A, B, disp_b, disp_h)

        sec_name = mem.get("section", etype)
        hover = f"{TYPE_LABELS.get(etype, etype)}: {sec_name}"
        add_beam_mesh(fig, verts, color, hover_text=hover)
        drawn_types.add(etype)

    # 지점 마커
    base_nodes = [n for n in nodes if n["z_m"] == 0.0]
    if base_nodes:
        fig.add_trace(go.Scatter3d(
            x=[n["x_m"] for n in base_nodes],
            y=[n["y_m"] for n in base_nodes],
            z=[n["z_m"] for n in base_nodes],
            mode="markers",
            marker=dict(size=6, color=SUPPORT_COLOR, symbol="diamond"),
            name="Supports",
            hoverinfo="skip",
            showlegend=True,
        ))

    # 범례 (부재 유형별)
    for etype in sorted(drawn_types):
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode="markers",
            marker=dict(symbol="square", size=10, color=TYPE_COLORS[etype]),
            name=f"{TYPE_LABELS[etype]} ({getattr(result, f'{etype}_section' if etype != 'column' else 'column_section', '')})",
            hoverinfo="skip",
            showlegend=True,
        ))

    fig.update_layout(**_get_default_layout())
    fig.update_layout(height=700)
    return fig


def create_deformed_figure(
    result,
    case_result,
    scale: float = 50.0,
) -> go.Figure:
    """변형 형상 시각화 (Jet_r 컬러맵).

    Args:
        result: Frame3DMultiCaseResult
        case_result: Frame3DCaseResult (특정 하중케이스/조합)
        scale: 변형 확대 배율

    Returns:
        변위 크기별 컬러맵이 적용된 3D Figure
    """
    fig = go.Figure()
    nodes = result.nodes
    min_span = _get_min_span(result)

    # 노드별 변위 매핑
    disp_map = {}
    for d in case_result.nodal_displacements:
        nid = d["node"]
        dx = d.get("dx_mm", 0.0) / 1000.0  # mm → m
        dy = d.get("dy_mm", 0.0) / 1000.0
        dz = d.get("dz_mm", 0.0) / 1000.0
        mag = (dx**2 + dy**2 + dz**2) ** 0.5 * 1000.0  # magnitude in mm
        disp_map[nid] = {"dx": dx, "dy": dy, "dz": dz, "mag_mm": mag}

    # 변형된 노드 좌표
    def_nodes = {}
    for n in nodes:
        nid = n["id"]
        d = disp_map.get(nid, {"dx": 0, "dy": 0, "dz": 0, "mag_mm": 0})
        def_nodes[nid] = {
            "x_m": n["x_m"] + d["dx"] * scale,
            "y_m": n["y_m"] + d["dy"] * scale,
            "z_m": n["z_m"] + d["dz"] * scale,
            "mag_mm": d["mag_mm"],
        }

    # 변위 범위
    all_mags = [v["mag_mm"] for v in def_nodes.values()]
    dmin = min(all_mags) if all_mags else 0.0
    dmax = max(all_mags) if all_mags else 0.0

    # 컬러맵
    jet_r = px.colors.sequential.Jet_r

    def map_color(val: float) -> str:
        if dmax == dmin:
            return jet_r[0]
        ratio = (val - dmin) / (dmax - dmin)
        return jet_r[int(ratio * (len(jet_r) - 1))]

    # 바운딩 큐브 (변형된 좌표 기준)
    def_list = [{"x_m": v["x_m"], "y_m": v["y_m"], "z_m": v["z_m"]} for v in def_nodes.values()]
    _add_bounding_cube(fig, def_list)

    # 단면 치수
    section_dims = {}
    for sec_type in ["column", "beam_x", "beam_y"]:
        sec_name = getattr(result, f"{sec_type}_section" if sec_type == "column" else f"{sec_type}_section", "")
        h_mm, b_mm = _parse_section_dims(sec_name)
        disp_h, disp_b = _display_dims(h_mm, b_mm, min_span)
        section_dims[sec_type] = (disp_h, disp_b)

    # 부재 그리기 (변형 좌표 + 변위 컬러)
    for mem in result.member_info:
        ni_id = mem["ni"]
        nj_id = mem["nj"]
        etype = mem["type"]

        ni = def_nodes.get(ni_id)
        nj = def_nodes.get(nj_id)
        if ni is None or nj is None:
            continue

        A = np.array([ni["x_m"], ni["y_m"], ni["z_m"]], dtype=float)
        B = np.array([nj["x_m"], nj["y_m"], nj["z_m"]], dtype=float)

        mean_mag = (ni["mag_mm"] + nj["mag_mm"]) / 2.0
        color = map_color(mean_mag)

        disp_h, disp_b = section_dims.get(etype, (0.2, 0.2))
        verts = compute_beam_vertices_rect(A, B, disp_b, disp_h)
        add_beam_mesh(fig, verts, color)

    # 컬러바
    fig.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode="markers",
        marker=dict(
            colorscale=DEFORM_COLORSCALE,
            cmin=dmin, cmax=dmax,
            color=[dmin, dmax],
            showscale=True,
            colorbar=dict(
                title="Disp. (mm)",
                len=0.45, lenmode="fraction",
                thickness=25, thicknessmode="pixels",
                y=0.5, yanchor="middle",
                x=1.02, xanchor="left",
                outlinewidth=1,
                ticks="outside",
                tickfont=dict(size=12),
            ),
        ),
        showlegend=False,
        hoverinfo="skip",
    ))

    # 최대 변위 주석
    max_mag = max(abs(dmin), abs(dmax))
    fig.add_annotation(
        text=f"<b>Max Displacement</b><br>{max_mag:.2f} mm (scale: {scale:.0f}x)",
        xref="paper", yref="paper",
        x=0.5, y=1,
        showarrow=False,
        bgcolor="rgba(255,255,255,0.85)",
        borderwidth=1,
        font=dict(size=14),
    )

    fig.update_layout(**_get_default_layout())
    fig.update_layout(height=700)
    return fig


def create_force_colored_figure(
    result,
    case_result,
    force_component: str = "My",
) -> go.Figure:
    """부재력 컬러맵 3D 시각화.

    Args:
        result: Frame3DMultiCaseResult
        case_result: Frame3DCaseResult
        force_component: "N", "Vy", "Vz", "T", "My", "Mz"

    Returns:
        부재력 크기별 RdYlBu 컬러맵 Figure
    """
    fig = go.Figure()
    nodes = result.nodes
    node_map = _build_node_map(nodes)
    min_span = _get_min_span(result)

    # 바운딩 큐브
    _add_bounding_cube(fig, nodes)

    # 요소별 최대 부재력 매핑
    # element_forces 키 형식: "My_i_kNm", "N_i_kN" 등
    force_map = {}
    unit_suffix = "_kNm" if force_component in ("My", "Mz", "T") else "_kN"
    key_i = f"{force_component}_i{unit_suffix}"
    key_j = f"{force_component}_j{unit_suffix}"
    for ef in case_result.element_forces:
        eid = ef["element"]
        val_i = abs(ef.get(key_i, 0.0))
        val_j = abs(ef.get(key_j, 0.0))
        force_map[eid] = max(val_i, val_j)

    all_forces = list(force_map.values())
    fmin = min(all_forces) if all_forces else 0.0
    fmax = max(all_forces) if all_forces else 0.0

    # RdYlBu 컬러스케일
    rdylbu = px.colors.diverging.RdYlBu

    def map_force_color(val: float) -> str:
        if fmax == fmin:
            return rdylbu[len(rdylbu) // 2]
        ratio = (val - fmin) / (fmax - fmin)
        return rdylbu[int(ratio * (len(rdylbu) - 1))]

    # 단면 치수
    section_dims = {}
    for sec_type in ["column", "beam_x", "beam_y"]:
        sec_name = getattr(result, f"{sec_type}_section" if sec_type == "column" else f"{sec_type}_section", "")
        h_mm, b_mm = _parse_section_dims(sec_name)
        disp_h, disp_b = _display_dims(h_mm, b_mm, min_span)
        section_dims[sec_type] = (disp_h, disp_b)

    # member_info에서 해당 요소의 부재력 찾기
    # member_info의 각 항목은 물리적 부재이고, elements는 sub-elements
    # element_forces는 sub-element 단위 → 부재별 최대값 취합
    member_max_force = {}
    for mem in result.member_info:
        elem_ids = mem.get("element_ids", [])
        max_f = 0.0
        for eid in elem_ids:
            max_f = max(max_f, force_map.get(eid, 0.0))
        member_max_force[mem["member_id"]] = max_f

    # 전체 범위 재계산
    all_mf = list(member_max_force.values())
    fmin = min(all_mf) if all_mf else 0.0
    fmax = max(all_mf) if all_mf else 0.0

    # 부재 그리기
    for mem in result.member_info:
        ni_id = mem["ni"]
        nj_id = mem["nj"]
        etype = mem["type"]

        ni = node_map.get(ni_id)
        nj = node_map.get(nj_id)
        if ni is None or nj is None:
            continue

        A = np.array([ni["x_m"], ni["y_m"], ni["z_m"]], dtype=float)
        B = np.array([nj["x_m"], nj["y_m"], nj["z_m"]], dtype=float)

        force_val = member_max_force.get(mem["member_id"], 0.0)
        color = map_force_color(force_val)

        disp_h, disp_b = section_dims.get(etype, (0.2, 0.2))
        verts = compute_beam_vertices_rect(A, B, disp_b, disp_h)

        # 부재력 단위
        unit = "kN" if force_component in ("N", "Vy", "Vz") else "kNm"
        hover = f"{TYPE_LABELS.get(etype, etype)}<br>|{force_component}| = {force_val:.1f} {unit}"
        add_beam_mesh(fig, verts, color, hover_text=hover)

    # 컬러바
    unit = "kN" if force_component in ("N", "Vy", "Vz") else "kNm"
    fig.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode="markers",
        marker=dict(
            colorscale=FORCE_COLORSCALE,
            cmin=fmin, cmax=fmax,
            color=[fmin, fmax],
            showscale=True,
            colorbar=dict(
                title=f"|{force_component}| ({unit})",
                len=0.45, lenmode="fraction",
                thickness=25, thicknessmode="pixels",
                y=0.5, yanchor="middle",
                x=1.02, xanchor="left",
                outlinewidth=1,
                ticks="outside",
                tickfont=dict(size=12),
            ),
        ),
        showlegend=False,
        hoverinfo="skip",
    ))

    fig.update_layout(**_get_default_layout())
    fig.update_layout(height=700)
    return fig


# ============================================================
# 유틸리티
# ============================================================

def _parse_section_dims(section_name: str) -> tuple[float, float]:
    """단면 이름에서 h, b(mm) 추출.

    Examples:
        "H-400x400" → (400, 400)
        "H-500x200" → (500, 200)
        "H-300x300" → (300, 300)
    """
    if not section_name:
        return 300.0, 300.0
    try:
        # "H-NNNxMMM" or "H-NNNxMMM..." 형식
        parts = section_name.split("-", 1)
        if len(parts) < 2:
            return 300.0, 300.0
        dims = parts[1].split("x")
        h = float(dims[0])
        b = float(dims[1]) if len(dims) > 1 else h
        return h, b
    except (ValueError, IndexError):
        return 300.0, 300.0
