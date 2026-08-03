"""다중 도면 정합 — 픽셀 좌표를 공통 world 좌표(m)로 매핑.

핵심 흐름:
1. 각 평면 시트마다 라벨된 GridSet 보유 (그리드 라인 좌표 + 라벨)
2. 사용자가 그리드 간격(`grid_spacing_x_m`, `grid_spacing_y_m`)을 명시
   → 각 라벨에 world 좌표 부여 (예: A→0.0m, B→8.0m, C→16.0m)
3. 라벨된 교점들로 픽셀→world affine을 시트별로 계산 (least-squares)
4. RegisteredFrame 으로 모음

평면 ↔ 입면 정합은 별도(`link_elevation_to_plan`)로 처리하며, MVP에선 stub.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .schemas import GridIntersection, GridSet, RegisteredFrame


# ─────────────────────────────────────────────────────────────
# 1) 라벨 → world 좌표
# ─────────────────────────────────────────────────────────────

def build_world_grid_from_spacing(
    labels: list[str],
    spacing_m: float,
    start_m: float = 0.0,
) -> dict[str, float]:
    """라벨 시퀀스에 균등 간격 world 좌표를 부여.

    예: labels=["A","B","C"], spacing_m=8.0 → {"A":0, "B":8, "C":16}
    """
    return {label: start_m + i * spacing_m for i, label in enumerate(labels)}


def build_world_grid_from_explicit(coords: dict[str, float]) -> dict[str, float]:
    """비균등 간격(예: {"A":0, "B":7.5, "C":18.2})을 그대로 통과.

    함수 호출의 의도(명시적 비균등)를 코드에서 드러내기 위한 thin wrapper.
    """
    return dict(coords)


# ─────────────────────────────────────────────────────────────
# 2) 단일 시트 affine
# ─────────────────────────────────────────────────────────────

def compute_affine_from_grid(
    grid_set: GridSet,
    world_grid_x: dict[str, float],
    world_grid_y: dict[str, float],
) -> tuple[np.ndarray, float]:
    """라벨된 GridSet과 world 좌표 매핑으로 픽셀→world affine 계산.

    Returns:
        (M3, rmse_m) — M3 is 3x3 homogeneous affine matrix.
        [wx, wy, 1].T ≈ M3 @ [px, py, 1].T

    Raises:
        ValueError: 라벨 매칭된 교점이 3개 미만 (affine 결정 불가)
    """
    src: list[tuple[float, float]] = []
    dst: list[tuple[float, float]] = []
    for inter in grid_set.intersections:
        if inter.x_label and inter.y_label:
            if inter.x_label in world_grid_x and inter.y_label in world_grid_y:
                src.append((inter.px, inter.py))
                dst.append((world_grid_x[inter.x_label], world_grid_y[inter.y_label]))

    if len(src) < 3:
        raise ValueError(
            f"Not enough labeled intersections to fit affine: {len(src)} < 3"
        )

    src_arr = np.array(src, dtype=np.float32)
    dst_arr = np.array(dst, dtype=np.float32)
    M, _ = cv2.estimateAffine2D(src_arr, dst_arr, method=cv2.LMEDS)
    if M is None:
        raise ValueError("cv2.estimateAffine2D failed (degenerate point set?)")

    M3 = np.vstack([M, [0.0, 0.0, 1.0]])

    # residual RMSE (m)
    ones = np.ones((src_arr.shape[0], 1), dtype=np.float32)
    src_h = np.hstack([src_arr, ones])              # (N,3)
    predicted = (M3 @ src_h.T).T[:, :2]              # (N,2)
    rmse_m = float(np.sqrt(np.mean(np.sum((predicted - dst_arr) ** 2, axis=1))))
    return M3, rmse_m


def transform_points(M3: np.ndarray, points_px: np.ndarray) -> np.ndarray:
    """affine 변환 적용. points_px shape: (N, 2). 반환: world (N, 2)."""
    pts = np.atleast_2d(points_px).astype(np.float32)
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    homo = np.hstack([pts, ones])
    world = (M3 @ homo.T).T[:, :2]
    return world


# ─────────────────────────────────────────────────────────────
# 3) 다중 평면 정합
# ─────────────────────────────────────────────────────────────

def register_plans(
    plan_grids: dict[str, GridSet],
    grid_spacing_x_m: float,
    grid_spacing_y_m: float,
    story_elevations_m: list[float],
    world_grid_x_override: Optional[dict[str, float]] = None,
    world_grid_y_override: Optional[dict[str, float]] = None,
) -> RegisteredFrame:
    """다중 평면(시트 id → GridSet)을 공통 world 좌표로 정합.

    각 평면이 같은 라벨 시스템(A,B,C / 1,2,3)을 사용한다고 가정.
    그리드 간격은 사용자 지정. 라벨된 모든 교점에서 픽셀→world affine을 계산.

    Args:
        plan_grids: {sheet_id: 라벨링된 GridSet}
        grid_spacing_x_m: vertical 그리드(A,B,C…) 간격
        grid_spacing_y_m: horizontal 그리드(1,2,3…) 간격
        story_elevations_m: 층별 elevation Z 좌표 (m)
        world_grid_x_override: 비균등 spacing을 명시할 때 사용 ({"A":0,"B":7.5,...})
        world_grid_y_override: 동일

    Returns:
        RegisteredFrame (plan_affines, world_grid_x/y/z, rmse_px=시트별 잔차[m])
    """
    if not plan_grids:
        raise ValueError("plan_grids is empty")

    # 모든 평면에서 사용된 라벨 합집합 (좌표 정렬 순서 보장)
    vertical_labels_by_coord: dict[str, float] = {}
    horizontal_labels_by_coord: dict[str, float] = {}
    for gs in plan_grids.values():
        for gl in gs.vertical_lines:
            if gl.label and gl.label not in vertical_labels_by_coord:
                vertical_labels_by_coord[gl.label] = gl.coord_px
        for gl in gs.horizontal_lines:
            if gl.label and gl.label not in horizontal_labels_by_coord:
                horizontal_labels_by_coord[gl.label] = gl.coord_px

    if world_grid_x_override is not None:
        world_grid_x = build_world_grid_from_explicit(world_grid_x_override)
    else:
        sorted_v = sorted(vertical_labels_by_coord, key=lambda l: vertical_labels_by_coord[l])
        world_grid_x = build_world_grid_from_spacing(sorted_v, grid_spacing_x_m)

    if world_grid_y_override is not None:
        world_grid_y = build_world_grid_from_explicit(world_grid_y_override)
    else:
        sorted_h = sorted(horizontal_labels_by_coord, key=lambda l: horizontal_labels_by_coord[l])
        world_grid_y = build_world_grid_from_spacing(sorted_h, grid_spacing_y_m)

    plan_affines: dict[str, np.ndarray] = {}
    rmse: dict[str, float] = {}
    for sheet_id, gs in plan_grids.items():
        M3, err = compute_affine_from_grid(gs, world_grid_x, world_grid_y)
        plan_affines[sheet_id] = M3
        rmse[sheet_id] = err

    return RegisteredFrame(
        plan_affines=plan_affines,
        elevation_affines={},
        world_grid_x=world_grid_x,
        world_grid_y=world_grid_y,
        world_grid_z=list(story_elevations_m),
        rmse_px=rmse,
    )


# ─────────────────────────────────────────────────────────────
# 4) 입면 ↔ 평면 정합
# ─────────────────────────────────────────────────────────────

def link_elevation_to_plan(
    elevation_grid: GridSet,
    elevation_orth_axis: str,
    registered: RegisteredFrame,
    story_labels: Optional[list[int]] = None,
) -> np.ndarray:
    """입면 시트 그리드를 이미 정합된 평면 world 좌표계와 연결.

    입면도는 X-Z 또는 Y-Z 평면을 본 그림. 입면도의 수직 라인은 평면의 한 축 그리드에
    대응하고, 수평 라인은 층 라인.

    Args:
        elevation_grid: 입면도에서 검출된 GridSet — vertical_lines는 라벨 부여돼 있어야 함
            (입면도가 본 평면 축의 라벨; 예: "A열 입면"이면 vertical_lines 라벨은 1,2,3…)
        elevation_orth_axis: 입면도가 본 평면 축
            - "vertical_grid":   입면의 수직 라인이 평면의 horizontal 라벨(1,2,3…)에 대응
                                 (즉 "A열 입면" 같은 경우 — A 그리드를 따라간 vertical slice)
            - "horizontal_grid": 입면의 수직 라인이 평면의 vertical 라벨(A,B,C…)에 대응
                                 (즉 "1통 입면" 같은 경우)
        story_labels: 입면 horizontal_lines의 층 인덱스 (좌표 오름차순).
            예: [0, 1, 2, 3, 4, 5]은 base + 5층. None이면 horizontal_lines 좌표 오름차순 자동.

    Returns:
        3×3 affine M3 such that [w_axis, w_z, 1].T ≈ M3 @ [px, py, 1].T
        - w_axis: 평면의 한 축 world 좌표 (X 또는 Y)
        - w_z:    elevation Z

    Raises:
        ValueError: 매칭 가능한 (수직 라벨, 수평 층) 페어가 3개 미만
    """
    if elevation_orth_axis == "vertical_grid":
        world_axis_map = registered.world_grid_y
    elif elevation_orth_axis == "horizontal_grid":
        world_axis_map = registered.world_grid_x
    else:
        raise ValueError(
            f"elevation_orth_axis must be 'vertical_grid' or 'horizontal_grid', "
            f"got {elevation_orth_axis!r}"
        )

    n_horiz = len(elevation_grid.horizontal_lines)
    if story_labels is None:
        # 좌표 오름차순 → 상→하 (이미지 좌표는 y가 위에서 아래로 증가)
        # 가장 아래(큰 y)가 base, 가장 위(작은 y)가 최상층
        sorted_h = sorted(
            range(n_horiz), key=lambda i: elevation_grid.horizontal_lines[i].coord_px,
            reverse=True,  # 큰 y(아래) → 작은 y(위) = base → 상층
        )
        story_labels = [None] * n_horiz
        for story, idx in enumerate(sorted_h):
            story_labels[idx] = story

    if len(story_labels) != n_horiz:
        raise ValueError(
            f"story_labels length {len(story_labels)} ≠ horizontal_lines count {n_horiz}"
        )

    # 라벨된 vertical × horizontal 교점에 매핑 빌드
    src: list[tuple[float, float]] = []
    dst: list[tuple[float, float]] = []
    for vl in elevation_grid.vertical_lines:
        if not vl.label or vl.label not in world_axis_map:
            continue
        w_axis = world_axis_map[vl.label]
        for i, hl in enumerate(elevation_grid.horizontal_lines):
            story = story_labels[i]
            if story is None or story >= len(registered.world_grid_z):
                continue
            w_z = registered.world_grid_z[story]
            src.append((vl.coord_px, hl.coord_px))
            dst.append((w_axis, w_z))

    if len(src) < 3:
        raise ValueError(
            f"Not enough matched (label×story) points for elevation affine: {len(src)} < 3"
        )

    src_arr = np.array(src, dtype=np.float32)
    dst_arr = np.array(dst, dtype=np.float32)
    M, _ = cv2.estimateAffine2D(src_arr, dst_arr, method=cv2.LMEDS)
    if M is None:
        raise ValueError("cv2.estimateAffine2D failed for elevation registration")
    return np.vstack([M, [0.0, 0.0, 1.0]])
