"""W5 — 입면 정합 + 보 추출 + builder 통합."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

_MCP_SERVER = Path(__file__).resolve().parents[3]
if str(_MCP_SERVER) not in sys.path:
    sys.path.insert(0, str(_MCP_SERVER))

from core.cad_parser import (  # noqa: E402
    builder, grid_detector, member_extract, registration, vectorize,
)
from core.cad_parser.schemas import (  # noqa: E402
    BeamCandidate, ColumnCandidate, GridLine, GridSet,
    LineSegment, RegisteredFrame, TypicalSectionSpec,
)


def _registered_3x3_5story() -> RegisteredFrame:
    """3×3 그리드, base+5층 = 6 elevation."""
    return RegisteredFrame(
        plan_affines={"1F": np.eye(3, dtype=np.float32)},
        elevation_affines={},
        world_grid_x={"A": 0.0, "B": 8.0, "C": 16.0},
        world_grid_y={"1": 0.0, "2": 6.0, "3": 12.0},
        world_grid_z=[0.0, 4.0, 7.5, 11.0, 14.5, 18.0],
        rmse_px={"1F": 0.0},
    )


# ───────────────────────── link_elevation_to_plan ─────────────────────────

class TestLinkElevationToPlan:
    """입면도 vertical=horizontal grid 라벨, horizontal=층 라인."""

    def _make_elev_grid(self) -> GridSet:
        """A열 입면도 — vertical_lines는 1,2,3 라벨 (Y 그리드)."""
        gs = GridSet(
            vertical_lines=[
                GridLine("vertical", 100.0, "1"),
                GridLine("vertical", 300.0, "2"),
                GridLine("vertical", 500.0, "3"),
            ],
            horizontal_lines=[
                # y 픽셀: 50(상=5F) … 350(하=base)
                GridLine("horizontal", 50.0),
                GridLine("horizontal", 110.0),
                GridLine("horizontal", 170.0),
                GridLine("horizontal", 230.0),
                GridLine("horizontal", 290.0),
                GridLine("horizontal", 350.0),
            ],
        )
        gs.intersections = grid_detector.compute_intersections(gs)
        return gs

    def test_a_column_elevation_recovery(self):
        # 균등 elevation으로 affine 정확 복원 검증 (비균등이면 LSQ 잔차 발생)
        rf = RegisteredFrame(
            plan_affines={}, elevation_affines={},
            world_grid_x={"A": 0.0, "B": 8.0, "C": 16.0},
            world_grid_y={"1": 0.0, "2": 6.0, "3": 12.0},
            world_grid_z=[0.0, 4.0, 8.0, 12.0, 16.0, 20.0],
            rmse_px={},
        )
        gs = self._make_elev_grid()
        # story_labels 자동: horizontal_lines 좌표 큰순(아래) → 0,1,2…
        M3 = registration.link_elevation_to_plan(gs, "vertical_grid", rf)
        # 픽셀 (300, 350) = (라벨 2, base) → world Y=6, Z=0
        w = registration.transform_points(M3, np.array([[300, 350]]))
        assert np.allclose(w[0], [6.0, 0.0], atol=0.01), w[0]
        # 픽셀 (500, 50) = (라벨 3, 5F) → world Y=12, Z=20
        w = registration.transform_points(M3, np.array([[500, 50]]))
        assert np.allclose(w[0], [12.0, 20.0], atol=0.01), w[0]

    def test_invalid_orth_axis_raises(self):
        rf = _registered_3x3_5story()
        gs = self._make_elev_grid()
        with pytest.raises(ValueError, match="orth_axis"):
            registration.link_elevation_to_plan(gs, "diagonal_grid", rf)

    def test_too_few_matched_raises(self):
        rf = _registered_3x3_5story()
        # 매칭 가능한 라벨이 1개뿐
        gs = GridSet(
            vertical_lines=[GridLine("vertical", 100.0, "1")],
            horizontal_lines=[
                GridLine("horizontal", 50.0),
                GridLine("horizontal", 200.0),
            ],
        )
        gs.intersections = grid_detector.compute_intersections(gs)
        with pytest.raises(ValueError, match="Not enough"):
            registration.link_elevation_to_plan(gs, "vertical_grid", rf)


# ───────────────────────── extract_beam_candidates ─────────────────────────

class TestExtractBeamCandidates:
    def _elev_grid_a_col(self) -> GridSet:
        """A열 입면 — vertical(1,2,3) × horizontal(base + 2F)."""
        gs = GridSet(
            vertical_lines=[
                GridLine("vertical", 100.0, "1"),
                GridLine("vertical", 300.0, "2"),
                GridLine("vertical", 500.0, "3"),
            ],
            horizontal_lines=[
                GridLine("horizontal", 50.0),    # 2F 상부 슬래브
                GridLine("horizontal", 200.0),   # 1F 상부 슬래브
                GridLine("horizontal", 350.0),   # base
            ],
        )
        gs.intersections = grid_detector.compute_intersections(gs)
        return gs

    def test_horizontal_segments_become_beams(self):
        gs = self._elev_grid_a_col()
        # story_labels: 큰 y=base → 0, 작은 y → 위층
        story_labels = [2, 1, 0]

        segments = [
            # 2F 위치 (y=50)에서 1→2, 2→3 보
            LineSegment(110, 50, 290, 51),
            LineSegment(310, 50, 490, 50),
            # 1F 위치 (y=200)에서 1→2 보만
            LineSegment(110, 200, 290, 200),
        ]
        beams = member_extract.extract_beam_candidates(
            elevation_horiz_segments=segments,
            elevation_grid=gs,
            elevation_orth_axis="vertical_grid",
            transverse_label="A",
            story_labels=story_labels,
            floor_tolerance_px=15.0,
            min_span_ratio=0.5,
        )
        # 기대: 2F에서 1↔2, 2↔3 + 1F에서 1↔2 = 3개
        assert len(beams) == 3
        # 모두 horizontal_grid 따라 (입면 vertical_grid의 반대)
        assert all(b.span_along == "horizontal_grid" for b in beams)
        assert all(b.transverse_label == "A" for b in beams)
        # 2F 보 라벨/story 확인
        f2 = [b for b in beams if b.story == 2]
        assert {(b.from_label, b.to_label) for b in f2} == {("1", "2"), ("2", "3")}
        # base(story=0) 보는 없음
        assert all(b.story > 0 for b in beams)

    def test_short_segment_below_min_span_ratio_excluded(self):
        gs = self._elev_grid_a_col()
        story_labels = [2, 1, 0]
        # 1→2 구간(폭=200) 중 50%만 커버 (overlap=80px) → ratio 0.4 < 0.5
        segments = [LineSegment(120, 50, 200, 50)]
        beams = member_extract.extract_beam_candidates(
            elevation_horiz_segments=segments,
            elevation_grid=gs,
            elevation_orth_axis="vertical_grid",
            transverse_label="A",
            story_labels=story_labels,
            floor_tolerance_px=15.0,
            min_span_ratio=0.5,
        )
        assert beams == []

    def test_segment_far_from_floor_excluded(self):
        gs = self._elev_grid_a_col()
        story_labels = [2, 1, 0]
        # floor line 50, 200, 350 — y=125는 어느 floor에서도 75px 떨어짐
        segments = [LineSegment(110, 125, 290, 125)]
        beams = member_extract.extract_beam_candidates(
            elevation_horiz_segments=segments,
            elevation_grid=gs,
            elevation_orth_axis="vertical_grid",
            transverse_label="A",
            story_labels=story_labels,
            floor_tolerance_px=30.0,
            min_span_ratio=0.5,
        )
        assert beams == []


# ───────────────────────── builder integration ─────────────────────────

class TestBuilderIntegratesBeams:
    def test_beams_become_beam_elements_with_correct_section(self):
        rf = _registered_3x3_5story()
        # 컬럼: A,B,C × 1,2,3 × 1~5층
        cols = [
            ColumnCandidate(xl, yl, s, s)
            for xl in ["A", "B", "C"] for yl in ["1", "2", "3"]
            for s in range(1, 6)
        ]
        # 보: 2층 슬래브에서 A-1↔A-2 (Y 방향, horizontal_grid)
        beams = [
            BeamCandidate(
                span_along="horizontal_grid",
                from_label="1", to_label="2",
                transverse_label="A",
                story=2,
            ),
            # 2층 슬래브에서 A-1↔B-1 (X 방향, vertical_grid)
            BeamCandidate(
                span_along="vertical_grid",
                from_label="A", to_label="B",
                transverse_label="1",
                story=2,
            ),
        ]
        typical = TypicalSectionSpec(
            column="H-400x400", beam_x="H-500x200", beam_y="H-400x200", material="SS275"
        )
        d = builder.build_structural_model_dict(rf, cols, beams, typical)

        # 보 element 2개
        beam_elems = [e for e in d["elements"] if e["elem_type"] == "beam"]
        assert len(beam_elems) == 2
        # 단면: vertical_grid → beam_x, horizontal_grid → beam_y
        sections = {e["section"] for e in beam_elems}
        assert sections == {"H-500x200", "H-400x200"}

        # 보 양 끝 노드 좌표가 grid 좌표와 일치
        for e in beam_elems:
            ni = next(n for n in d["nodes"] if n["id"] == e["node_i"])
            nj = next(n for n in d["nodes"] if n["id"] == e["node_j"])
            # 둘 다 같은 Z (story=2 → world_grid_z[2] = 7.5)
            assert ni["z"] == 7.5
            assert nj["z"] == 7.5

    def test_beam_with_unknown_label_skipped(self):
        rf = _registered_3x3_5story()
        beams = [
            BeamCandidate(
                span_along="vertical_grid",
                from_label="Z", to_label="A",      # Z 미존재
                transverse_label="1",
                story=1,
            ),
        ]
        typical = TypicalSectionSpec(
            column="H-400x400", beam_x="H-500x200", beam_y="H-400x200"
        )
        d = builder.build_structural_model_dict(rf, [], beams, typical)
        # 보가 element로 들어가지 않음
        assert all(e["elem_type"] != "beam" for e in d["elements"])
