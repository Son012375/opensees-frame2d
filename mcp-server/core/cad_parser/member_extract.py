"""부재(컬럼/보) 후보 추출.

MVP W3: 평면도에서 **컬럼**만 검출.
- vectorize.extract_polygons → 폴리곤 후보 리스트
- 각 폴리곤 중심점이 그리드 intersection 근처에 있으면 컬럼으로 판정
- 평면이 어느 층을 표현하는지(sheet_id → story 매핑)는 호출자가 제공
- story_from / story_to는 "이 평면이 표현하는 층(s)"으로 채워짐

보 추출은 W5에서 (`extract_beam_candidates`).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np

from .schemas import ColumnCandidate, GridIntersection, GridSet, Polygon


def _polygon_centroid(poly: Polygon) -> tuple[float, float]:
    """폴리곤의 area-weighted centroid (간단 평균 — convex/small box에 충분)."""
    pts = poly.points
    return float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))


def assign_polygons_to_grid(
    polygons: list[Polygon],
    grid_set: GridSet,
    max_dist_px: float = 25.0,
) -> dict[tuple[str, str], list[Polygon]]:
    """폴리곤을 가장 가까운 그리드 intersection에 할당.

    Args:
        polygons: vectorize.extract_polygons 결과
        grid_set: 라벨된 GridSet (intersections 채워짐)
        max_dist_px: 이 거리 초과는 미할당(컬럼 아님으로 간주)

    Returns:
        {(x_label, y_label): [polygon, …]} — 같은 교점에 여러 후보가 떨어지면 모두 보존
    """
    out: dict[tuple[str, str], list[Polygon]] = defaultdict(list)
    intersections = [i for i in grid_set.intersections if i.x_label and i.y_label]
    if not intersections:
        return out

    inter_pts = np.array([[i.px, i.py] for i in intersections], dtype=np.float32)
    inter_keys = [(i.x_label, i.y_label) for i in intersections]

    for poly in polygons:
        cx, cy = _polygon_centroid(poly)
        dists = np.linalg.norm(inter_pts - np.array([cx, cy], dtype=np.float32), axis=1)
        nearest = int(np.argmin(dists))
        if dists[nearest] <= max_dist_px:
            out[inter_keys[nearest]].append(poly)
    return out


def extract_column_candidates(
    polygons_per_plan: dict[str, list[Polygon]],
    plan_grid_per_plan: dict[str, GridSet],
    sheet_id_to_stories: dict[str, list[int]],
    max_dist_px: float = 25.0,
    min_area_px: float = 80.0,
    max_area_px: Optional[float] = None,
) -> list[ColumnCandidate]:
    """평면들에서 컬럼 후보 추출.

    Args:
        polygons_per_plan: {sheet_id: 폴리곤 리스트}
        plan_grid_per_plan: {sheet_id: 라벨된 GridSet}
        sheet_id_to_stories: {sheet_id: [story_n, …]}
            이 평면이 어느 층(들, 1-based)을 표현하는지. 보통 한 평면 = 한 층.
            typical floor 평면은 여러 층을 가질 수 있음.
            story_n=1 평면 → 노드 elev_idx (0,1), 즉 base→1F 컬럼.
        max_dist_px: 교점-폴리곤 매칭 거리 임계치
        min_area_px / max_area_px: 면적 필터 (컬럼은 너무 작거나 크지 않음)

    Returns:
        ColumnCandidate 리스트. 같은 (grid_x, grid_y, story) 조합은 1개로 dedup.
    """
    # (x_label, y_label, story) 등장 여부
    occurrences: set[tuple[str, str, int]] = set()
    candidates: list[ColumnCandidate] = []

    for sheet_id, polys in polygons_per_plan.items():
        gs = plan_grid_per_plan.get(sheet_id)
        if gs is None or not gs.intersections:
            continue
        stories = sheet_id_to_stories.get(sheet_id, [])
        if not stories:
            continue

        filtered = [
            p for p in polys
            if p.area >= min_area_px and (max_area_px is None or p.area <= max_area_px)
        ]
        per_grid = assign_polygons_to_grid(filtered, gs, max_dist_px=max_dist_px)

        for (xl, yl), poly_list in per_grid.items():
            if not poly_list:
                continue
            for story in stories:
                key = (xl, yl, story)
                if key in occurrences:
                    continue
                occurrences.add(key)
                # 같은 교점에 여러 polygon이 떨어지면 가장 큰 것의 score(=1.0 fixed for now)
                candidates.append(
                    ColumnCandidate(
                        grid_x_label=xl,
                        grid_y_label=yl,
                        story_from=story,
                        story_to=story,
                        confidence=1.0,
                    )
                )
    return candidates


def extract_beam_candidates(
    elevation_horiz_segments: list,           # list[LineSegment] 수평 long lines
    elevation_grid: GridSet,                  # 라벨된 입면 GridSet
    elevation_orth_axis: str,                 # "vertical_grid" or "horizontal_grid"
    transverse_label: str,                    # 입면이 본 평면 축의 라벨 (예: "A" 또는 "1")
    story_labels: list[int],                  # horizontal_lines i → story index (1-based)
    floor_tolerance_px: float = 30.0,
    min_span_ratio: float = 0.5,
) -> list["BeamCandidate"]:
    """입면도의 long horizontal segments → BeamCandidate.

    각 long horizontal segment가:
      - 검출된 floor line ±floor_tolerance_px 안에 있고
      - 두 수직 그리드 라벨 사이를 적어도 min_span_ratio 만큼 가로지를 때
    BeamCandidate를 1개 생성.

    Args:
        elevation_horiz_segments: vectorize.split_by_orientation 의 horizontal 부분
        elevation_grid: detect_grid + 라벨 부여된 입면 GridSet (vertical_lines 라벨 필요)
        elevation_orth_axis: 입면이 본 축. "vertical_grid"=A열 입면(=Y-Z 평면),
            "horizontal_grid"=1통 입면(=X-Z 평면)
            beam의 span_along은 입력의 반대 — A열 입면은 수직 그리드가 1,2,3 라벨이므로
            보는 horizontal_grid 따라감 → span_along="horizontal_grid"
        transverse_label: 입면이 본 평면 축의 라벨 (예: "A")
        story_labels: 입면 horizontal_lines[i] → story index. 길이 일치.
        floor_tolerance_px: floor line 근처 라인만 보로 인정
        min_span_ratio: 그리드 간격 대비 최소 span 비율 (예: 0.8 권장)

    Returns:
        BeamCandidate 리스트. 동일 (span_along, from, to, transverse, story) 조합 dedup.
    """
    from .schemas import BeamCandidate

    if elevation_orth_axis == "vertical_grid":
        beam_span = "horizontal_grid"
    elif elevation_orth_axis == "horizontal_grid":
        beam_span = "vertical_grid"
    else:
        raise ValueError(f"unknown elevation_orth_axis: {elevation_orth_axis!r}")

    # 입면의 수직 그리드 라벨 + 좌표 (좌표 오름차순)
    labeled_v = [(gl.label, gl.coord_px) for gl in elevation_grid.vertical_lines if gl.label]
    labeled_v.sort(key=lambda kv: kv[1])
    if len(labeled_v) < 2:
        return []

    # 입면 horizontal_lines i → story
    if len(story_labels) != len(elevation_grid.horizontal_lines):
        raise ValueError("story_labels length mismatch with horizontal_lines")

    floor_y_to_story = {
        gl.coord_px: story_labels[i]
        for i, gl in enumerate(elevation_grid.horizontal_lines)
        if story_labels[i] is not None and story_labels[i] > 0
    }

    out_set: set[tuple] = set()
    candidates: list[BeamCandidate] = []

    for seg in elevation_horiz_segments:
        y_mid = (seg.y1 + seg.y2) / 2
        x_lo, x_hi = sorted([seg.x1, seg.x2])

        # floor line 매칭
        nearest_y = min(floor_y_to_story.keys(), key=lambda y: abs(y - y_mid))
        if abs(nearest_y - y_mid) > floor_tolerance_px:
            continue
        story = floor_y_to_story[nearest_y]

        # 어떤 그리드 라벨 두 개를 잇는가
        for (l_from, x_from), (l_to, x_to) in zip(labeled_v[:-1], labeled_v[1:]):
            grid_span = x_to - x_from
            if grid_span <= 0:
                continue
            # segment가 [x_from, x_to] 구간을 min_span_ratio 이상 커버하는지
            overlap = max(0.0, min(x_hi, x_to) - max(x_lo, x_from))
            if overlap / grid_span >= min_span_ratio:
                key = (beam_span, l_from, l_to, transverse_label, story)
                if key in out_set:
                    continue
                out_set.add(key)
                candidates.append(BeamCandidate(
                    span_along=beam_span,
                    from_label=l_from,
                    to_label=l_to,
                    transverse_label=transverse_label,
                    story=story,
                ))

    return candidates


def merge_columns_across_stories(
    candidates: list[ColumnCandidate],
) -> list[ColumnCandidate]:
    """동일 (grid_x, grid_y) 위치의 연속된 story 후보들을 단일 ColumnCandidate로 통합.

    예: [(A,1,story=0), (A,1,story=1), (A,1,story=2)] → (A,1, story_from=0, story_to=2)

    효용: 빌더 단계에서 컬럼 1개당 노드 페어를 한 번만 만들 수 있음.
    """
    by_grid: dict[tuple[str, str], list[int]] = defaultdict(list)
    for c in candidates:
        by_grid[(c.grid_x_label, c.grid_y_label)].append(c.story_from)

    merged: list[ColumnCandidate] = []
    for (xl, yl), stories in by_grid.items():
        sorted_stories = sorted(set(stories))
        # 연속 구간으로 분할
        run_start = sorted_stories[0]
        prev = run_start
        for s in sorted_stories[1:]:
            if s == prev + 1:
                prev = s
                continue
            merged.append(
                ColumnCandidate(
                    grid_x_label=xl, grid_y_label=yl,
                    story_from=run_start, story_to=prev,
                )
            )
            run_start = s
            prev = s
        merged.append(
            ColumnCandidate(
                grid_x_label=xl, grid_y_label=yl,
                story_from=run_start, story_to=prev,
            )
        )
    return merged
