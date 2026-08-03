"""W3 acceptance test — 다중 평면 정합 + world 좌표 복원."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

_MCP_SERVER = Path(__file__).resolve().parents[3]
if str(_MCP_SERVER) not in sys.path:
    sys.path.insert(0, str(_MCP_SERVER))

from core.cad_parser import grid_detector, registration  # noqa: E402
from core.cad_parser.schemas import GridIntersection, GridLine, GridSet  # noqa: E402


def _labeled_grid(
    xs_px: list[float], ys_px: list[float],
    x_labels: list[str], y_labels: list[str],
) -> GridSet:
    """라벨이 모두 채워진 GridSet 합성 (테스트 fixture)."""
    gs = GridSet(
        vertical_lines=[
            GridLine("vertical", x, lab) for x, lab in zip(xs_px, x_labels)
        ],
        horizontal_lines=[
            GridLine("horizontal", y, lab) for y, lab in zip(ys_px, y_labels)
        ],
    )
    gs.intersections = grid_detector.compute_intersections(gs)
    return gs


# ───────────────────────── build_world_grid ─────────────────────────

class TestBuildWorldGrid:
    def test_uniform_spacing(self):
        out = registration.build_world_grid_from_spacing(["A", "B", "C"], 8.0)
        assert out == {"A": 0.0, "B": 8.0, "C": 16.0}

    def test_explicit_passthrough(self):
        explicit = {"A": 0.0, "B": 7.5, "C": 18.2}
        assert registration.build_world_grid_from_explicit(explicit) == explicit


# ───────────────────────── compute_affine_from_grid ─────────────────────────

class TestComputeAffine:
    def test_axis_aligned_recovery(self):
        """그리드가 축 정렬이고 균등 간격이면 affine은 단순 scale + translation."""
        gs = _labeled_grid(
            xs_px=[100, 200, 300],
            ys_px=[150, 250, 350],
            x_labels=["A", "B", "C"],
            y_labels=["1", "2", "3"],
        )
        world_x = {"A": 0.0, "B": 8.0, "C": 16.0}
        world_y = {"1": 0.0, "2": 6.0, "3": 12.0}
        M3, rmse = registration.compute_affine_from_grid(gs, world_x, world_y)

        # 픽셀 (200, 250) → world (8, 6)
        result = registration.transform_points(M3, np.array([[200, 250]]))
        assert np.allclose(result[0], [8.0, 6.0], atol=1e-3), result[0]
        # 픽셀 (100, 150) → world (0, 0)
        result0 = registration.transform_points(M3, np.array([[100, 150]]))
        assert np.allclose(result0[0], [0.0, 0.0], atol=1e-3), result0[0]
        # rmse 거의 0
        assert rmse < 1e-3

    def test_too_few_intersections_raises(self):
        gs = GridSet(
            vertical_lines=[GridLine("vertical", 100, "A")],
            horizontal_lines=[GridLine("horizontal", 100, "1")],
        )
        gs.intersections = grid_detector.compute_intersections(gs)
        with pytest.raises(ValueError, match="Not enough"):
            registration.compute_affine_from_grid(gs, {"A": 0}, {"1": 0})


# ───────────────────────── register_plans (multi-plan) ─────────────────────────

class TestRegisterPlans:
    def test_two_plans_same_grid_recover_same_world(self):
        """두 평면이 같은 그리드를 다른 위치/스케일로 그렸어도 라벨로 정합."""
        # plan 1F: 그리드가 픽셀 (100,150) ~ (300,350)
        plan_1 = _labeled_grid(
            xs_px=[100, 200, 300], ys_px=[150, 250, 350],
            x_labels=["A", "B", "C"], y_labels=["1", "2", "3"],
        )
        # plan 2F: 그리드가 픽셀 (50,80) ~ (450,480) (다른 시트 스케일)
        plan_2 = _labeled_grid(
            xs_px=[50, 250, 450], ys_px=[80, 280, 480],
            x_labels=["A", "B", "C"], y_labels=["1", "2", "3"],
        )

        rf = registration.register_plans(
            plan_grids={"1F": plan_1, "2F": plan_2},
            grid_spacing_x_m=8.0,
            grid_spacing_y_m=6.0,
            story_elevations_m=[0.0, 4.0],
        )

        assert set(rf.plan_affines.keys()) == {"1F", "2F"}
        assert rf.world_grid_x == {"A": 0.0, "B": 8.0, "C": 16.0}
        assert rf.world_grid_y == {"1": 0.0, "2": 6.0, "3": 12.0}
        assert rf.world_grid_z == [0.0, 4.0]
        # 두 시트에서 라벨 B-2 위치는 둘 다 world (8, 6)
        for sheet_id, b2_px in [("1F", [200, 250]), ("2F", [250, 280])]:
            w = registration.transform_points(rf.plan_affines[sheet_id], np.array([b2_px]))
            assert np.allclose(w[0], [8.0, 6.0], atol=1e-3), (sheet_id, w[0])
        # RMSE는 두 시트 모두 거의 0
        assert all(r < 1e-3 for r in rf.rmse_px.values())

    def test_explicit_override_uneven_spacing(self):
        plan = _labeled_grid(
            xs_px=[100, 200, 350], ys_px=[100, 200, 350],
            x_labels=["A", "B", "C"], y_labels=["1", "2", "3"],
        )
        rf = registration.register_plans(
            plan_grids={"plan": plan},
            grid_spacing_x_m=8.0,    # 무시됨 — override 우선
            grid_spacing_y_m=6.0,
            story_elevations_m=[0.0],
            world_grid_x_override={"A": 0.0, "B": 8.0, "C": 20.0},
            world_grid_y_override={"1": 0.0, "2": 6.0, "3": 15.0},
        )
        assert rf.world_grid_x["C"] == 20.0
        assert rf.world_grid_y["3"] == 15.0
        # 픽셀 (350, 350) = C-3 → world (20, 15)
        w = registration.transform_points(rf.plan_affines["plan"], np.array([[350, 350]]))
        assert np.allclose(w[0], [20.0, 15.0], atol=1e-3), w[0]

    def test_empty_plans_raises(self):
        with pytest.raises(ValueError):
            registration.register_plans(
                {}, grid_spacing_x_m=8.0, grid_spacing_y_m=6.0, story_elevations_m=[0]
            )


# link_elevation_to_plan은 W5에서 구현 완료. 상세 테스트는 test_elevation.py 참조.
