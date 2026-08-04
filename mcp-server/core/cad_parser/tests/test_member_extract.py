"""W3 acceptance test — 컬럼 후보 추출 (≥83% recall)."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

_MCP_SERVER = Path(__file__).resolve().parents[3]
if str(_MCP_SERVER) not in sys.path:
    sys.path.insert(0, str(_MCP_SERVER))

from core.cad_parser import grid_detector, member_extract, vectorize  # noqa: E402
from core.cad_parser.schemas import (  # noqa: E402
    ColumnCandidate, GridLine, GridSet, Polygon,
)


# ───────────────────────── fixtures ─────────────────────────

def _make_plan_with_painted_columns(
    width: int = 800,
    height: int = 800,
    grid_xs: list[int] = [200, 400, 600],
    grid_ys: list[int] = [200, 400, 600],
    column_size_px: int = 16,
) -> tuple[np.ndarray, list[tuple[str, str, int, int]]]:
    """그리드 + 모든 교점에 칠해진 컬럼 정사각형이 있는 합성 평면.

    컬럼이 그리드 라인과 connected component로 합쳐지지 않도록
    그리드 라인을 컬럼 영역 주변에서 끊어서 그림.

    Returns: (image, [(x_label, y_label, px, py), …])
    """
    img = np.full((height, width), 255, dtype=np.uint8)
    half = column_size_px // 2
    gap = half + 4  # 컬럼 주변 4px 갭

    # vertical 그리드: 각 라인을 horizontal 그리드 교점에서 갭만큼 끊어 그림
    for x in grid_xs:
        x_int = int(x)
        prev_y = 100
        for gy in grid_ys:
            cv2.line(img, (x_int, prev_y), (x_int, int(gy) - gap), 0, 1)
            prev_y = int(gy) + gap
        cv2.line(img, (x_int, prev_y), (x_int, height - 100), 0, 1)

    # horizontal 그리드: 동일 처리
    for y in grid_ys:
        y_int = int(y)
        prev_x = 100
        for gx in grid_xs:
            cv2.line(img, (prev_x, y_int), (int(gx) - gap, y_int), 0, 1)
            prev_x = int(gx) + gap
        cv2.line(img, (prev_x, y_int), (width - 100, y_int), 0, 1)

    truth: list[tuple[str, str, int, int]] = []
    x_labels = [chr(ord("A") + i) for i in range(len(grid_xs))]
    y_labels = [str(i + 1) for i in range(len(grid_ys))]
    for xi, gx in enumerate(grid_xs):
        for yi, gy in enumerate(grid_ys):
            cv2.rectangle(
                img,
                (gx - half, gy - half),
                (gx + half, gy + half),
                0, -1,
            )
            truth.append((x_labels[xi], y_labels[yi], gx, gy))
    return img, truth


def _binarize(img: np.ndarray) -> np.ndarray:
    _, b = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    return b


def _labeled_grid_from_truth(truth: list[tuple[str, str, int, int]]) -> GridSet:
    """truth 리스트에서 라벨된 GridSet 직접 생성 (OCR 우회)."""
    by_x: dict[str, int] = {}
    by_y: dict[str, int] = {}
    for xl, yl, px, py in truth:
        by_x[xl] = px
        by_y[yl] = py
    gs = GridSet(
        vertical_lines=[
            GridLine("vertical", float(px), xl)
            for xl, px in sorted(by_x.items(), key=lambda kv: kv[1])
        ],
        horizontal_lines=[
            GridLine("horizontal", float(py), yl)
            for yl, py in sorted(by_y.items(), key=lambda kv: kv[1])
        ],
    )
    gs.intersections = grid_detector.compute_intersections(gs)
    return gs


# ───────────────────────── assign_polygons_to_grid ─────────────────────────

class TestAssignPolygonsToGrid:
    def test_polygon_near_intersection_assigned(self):
        gs = _labeled_grid_from_truth([("A", "1", 200, 200)])
        # 교점에서 5px 떨어진 곳에 polygon
        poly = Polygon(points=np.array([[195, 195], [205, 195], [205, 205], [195, 205]], dtype=np.float32), area=100.0)
        out = member_extract.assign_polygons_to_grid([poly], gs, max_dist_px=20.0)
        assert ("A", "1") in out
        assert out[("A", "1")][0] is poly

    def test_polygon_too_far_unassigned(self):
        gs = _labeled_grid_from_truth([("A", "1", 200, 200)])
        poly = Polygon(points=np.array([[400, 400], [410, 400], [410, 410], [400, 410]], dtype=np.float32), area=100.0)
        out = member_extract.assign_polygons_to_grid([poly], gs, max_dist_px=20.0)
        assert ("A", "1") not in out


# ───────────────────────── E2E: 합성 평면 → 컬럼 검출 ─────────────────────────

class TestExtractColumnCandidatesE2E:
    def test_3x3_grid_recovers_at_least_83pct_columns(self):
        """3x3 그리드 = 9 컬럼 → ≥8 검출 (≥88%, plan의 ≥83% 기준 통과)."""
        img, truth = _make_plan_with_painted_columns()
        binary = _binarize(img)

        polygons = vectorize.extract_polygons(binary, min_area=100.0, max_area=1000.0)
        gs = _labeled_grid_from_truth(truth)

        candidates = member_extract.extract_column_candidates(
            polygons_per_plan={"plan_1F": polygons},
            plan_grid_per_plan={"plan_1F": gs},
            sheet_id_to_stories={"plan_1F": [0]},
            max_dist_px=25.0,
            min_area_px=100.0,
            max_area_px=1000.0,
        )

        # 같은 (xl, yl, story) 조합으로 dedup된 후 9개 기대
        labels_found = {(c.grid_x_label, c.grid_y_label) for c in candidates}
        labels_truth = {(t[0], t[1]) for t in truth}

        recall = len(labels_found & labels_truth) / len(labels_truth)
        assert recall >= 0.83, (
            f"column recall {recall:.0%} < 83%; "
            f"found={labels_found}, truth={labels_truth}"
        )

    def test_typical_floor_applied_to_multiple_stories(self):
        """한 평면 → 여러 층(2,3,4,5층) 적용 시 모든 층에 컬럼 후보 생성."""
        img, truth = _make_plan_with_painted_columns(
            grid_xs=[200, 400], grid_ys=[200, 400]
        )
        binary = _binarize(img)
        polygons = vectorize.extract_polygons(binary, min_area=100.0, max_area=1000.0)
        gs = _labeled_grid_from_truth(truth)

        candidates = member_extract.extract_column_candidates(
            polygons_per_plan={"typical": polygons},
            plan_grid_per_plan={"typical": gs},
            sheet_id_to_stories={"typical": [2, 3, 4, 5]},
        )
        # 4 컬럼 × 4 층 = 16 후보 기대 (recall 100% 시)
        stories_covered = {c.story_from for c in candidates}
        assert stories_covered == {2, 3, 4, 5}


# ───────────────────────── merge_columns_across_stories ─────────────────────────

class TestMergeAcrossStories:
    def test_consecutive_stories_merge_into_single_run(self):
        candidates = [
            ColumnCandidate("A", "1", 0, 0),
            ColumnCandidate("A", "1", 1, 1),
            ColumnCandidate("A", "1", 2, 2),
            ColumnCandidate("B", "1", 0, 0),
        ]
        merged = member_extract.merge_columns_across_stories(candidates)
        a1 = [m for m in merged if (m.grid_x_label, m.grid_y_label) == ("A", "1")]
        assert len(a1) == 1
        assert a1[0].story_from == 0 and a1[0].story_to == 2
        b1 = [m for m in merged if (m.grid_x_label, m.grid_y_label) == ("B", "1")]
        assert len(b1) == 1
        assert b1[0].story_from == 0 and b1[0].story_to == 0

    def test_gap_in_stories_creates_two_runs(self):
        candidates = [
            ColumnCandidate("A", "1", 0, 0),
            ColumnCandidate("A", "1", 1, 1),
            # gap at story 2
            ColumnCandidate("A", "1", 3, 3),
        ]
        merged = member_extract.merge_columns_across_stories(candidates)
        assert len(merged) == 2
        runs = sorted([(m.story_from, m.story_to) for m in merged])
        assert runs == [(0, 1), (3, 3)]
