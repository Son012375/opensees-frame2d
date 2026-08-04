"""W4 acceptance test — builder + CLI + StructuralModel round-trip."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

_MCP_SERVER = Path(__file__).resolve().parents[3]
if str(_MCP_SERVER) not in sys.path:
    sys.path.insert(0, str(_MCP_SERVER))

from core.cad_parser import builder, fallback, registration  # noqa: E402
from core.cad_parser.schemas import (  # noqa: E402
    ColumnCandidate,
    RegisteredFrame,
    TypicalSectionSpec,
)


# ───────────────────────── fixtures ─────────────────────────

def _make_simple_registered() -> RegisteredFrame:
    """3×3 그리드, 4층 건물 (base + 3F) 의 RegisteredFrame."""
    return RegisteredFrame(
        plan_affines={"1F": np.eye(3, dtype=np.float32)},
        elevation_affines={},
        world_grid_x={"A": 0.0, "B": 8.0, "C": 16.0},
        world_grid_y={"1": 0.0, "2": 6.0, "3": 12.0},
        world_grid_z=[0.0, 4.0, 8.0, 12.0],   # base + 3 stories
        rmse_px={"1F": 0.0},
    )


def _typical() -> TypicalSectionSpec:
    return TypicalSectionSpec(
        column="H-400x400", beam_x="H-500x200", beam_y="H-400x200", material="SS275"
    )


# ───────────────────────── builder unit tests ─────────────────────────

class TestBuildStructuralModelDict:
    def test_basic_4_columns_3_stories(self):
        registered = _make_simple_registered()
        # A1, A3, C1, C3 의 1~3층 컬럼 (1-based)
        columns = [
            ColumnCandidate("A", "1", s, s) for s in range(1, 4)
        ] + [
            ColumnCandidate("A", "3", s, s) for s in range(1, 4)
        ] + [
            ColumnCandidate("C", "1", s, s) for s in range(1, 4)
        ] + [
            ColumnCandidate("C", "3", s, s) for s in range(1, 4)
        ]
        d = builder.build_structural_model_dict(
            registered, columns, [], _typical(),
        )

        # 4 그리드 위치 × 4 elevation(base + 3F) = 16 노드
        assert len(d["nodes"]) == 16
        # 4 컬럼 × 3 층 = 12 element
        assert len(d["elements"]) == 12
        for e in d["elements"]:
            assert e["elem_type"] == "column"
            assert e["section"] == "H-400x400"
            assert e["material"] == "SS275"
        # base 노드(elev_idx=0) = fixed
        base_nodes = [n for n in d["nodes"] if n["story"] == 0]
        assert len(base_nodes) == 4
        assert all(n["support"] == "fixed" for n in base_nodes)
        top_nodes = [n for n in d["nodes"] if n["story"] > 0]
        assert all(n["support"] is None for n in top_nodes)

    def test_node_coordinates_match_world_grid(self):
        registered = _make_simple_registered()
        # 1층 컬럼 → 노드 elev_idx=0,1 → Z=0, 4
        columns = [ColumnCandidate("B", "2", 1, 1)]
        d = builder.build_structural_model_dict(registered, columns, [], _typical())
        b2_base = next(n for n in d["nodes"] if n["story"] == 0)
        assert (b2_base["x"], b2_base["y"], b2_base["z"]) == (8.0, 6.0, 0.0)
        b2_1 = next(n for n in d["nodes"] if n["story"] == 1)
        assert (b2_1["x"], b2_1["y"], b2_1["z"]) == (8.0, 6.0, 4.0)

    def test_unknown_grid_label_skipped(self):
        registered = _make_simple_registered()
        # "Z" 라벨이 world_grid_x에 없음
        columns = [ColumnCandidate("Z", "1", 1, 1)]
        d = builder.build_structural_model_dict(registered, columns, [], _typical())
        assert len(d["nodes"]) == 0
        assert len(d["elements"]) == 0

    def test_empty_story_elevations_raises(self):
        registered = RegisteredFrame(
            plan_affines={}, elevation_affines={},
            world_grid_x={"A": 0}, world_grid_y={"1": 0},
            world_grid_z=[], rmse_px={},
        )
        with pytest.raises(ValueError, match="story elevations"):
            builder.build_structural_model_dict(registered, [], [], _typical())

    def test_environment_override(self):
        registered = _make_simple_registered()
        d = builder.build_structural_model_dict(
            registered, [], [], _typical(),
            environment={"region": "서울 강남", "importance": "II"},
        )
        assert d["environment"]["region"] == "서울 강남"
        assert d["environment"]["importance"] == "II"
        # 기본값 유지
        assert d["environment"]["site_class"] == "S3"


# ───────────────────────── round-trip with StructuralModel.from_json ─────────────────────────

class TestRoundTripWithStructuralModel:
    """builder 출력을 실제 StructuralModel.from_json()이 받을 수 있는지 검증.

    이게 W4의 핵심 약속 — 기존 시스템에 0건 수정으로 통합 가능함.
    """

    def test_from_json_accepts_builder_output(self):
        from core.structural_model import StructuralModel  # noqa

        registered = _make_simple_registered()
        # 9 그리드 × 1~3층 컬럼
        columns = [
            ColumnCandidate(xl, yl, s, s)
            for xl in ["A", "B", "C"]
            for yl in ["1", "2", "3"]
            for s in range(1, 4)
        ]
        d = builder.build_structural_model_dict(registered, columns, [], _typical())

        model = StructuralModel.from_json(d)
        # 9 grid × 4 elevation = 36 nodes; 9 × 3 = 27 columns
        assert len(model.nodes) == 36
        assert len(model.elements) == 27
        bases = [n for n in model.nodes.values() if n.story == 0]
        assert all(n.support is not None and n.support.value == "fixed" for n in bases)


# ───────────────────────── wrap_v2proj ─────────────────────────

class TestWrapV2Proj:
    def test_wrapper_has_required_keys(self):
        d = builder.build_structural_model_dict(
            _make_simple_registered(), [], [], _typical()
        )
        wrapped = builder.wrap_v2proj(d)
        assert wrapped["version"] == 3
        assert "timestamp" in wrapped
        assert wrapped["model"] is d
        # V2 UI 검증: project.model.nodes 가 존재
        assert "nodes" in wrapped["model"]


# ───────────────────────── fallback.affine_from_keypoints ─────────────────────────

class TestFallbackKeypoints:
    def test_4_keypoints_recover_translation_scale(self):
        pixel = [(100, 100), (300, 100), (300, 300), (100, 300)]
        world = [(0.0, 0.0), (20.0, 0.0), (20.0, 20.0), (0.0, 20.0)]
        M3, rmse = fallback.affine_from_keypoints(pixel, world)
        # (200, 200) → (10, 10)
        from core.cad_parser.registration import transform_points
        w = transform_points(M3, np.array([[200, 200]]))
        assert np.allclose(w[0], [10.0, 10.0], atol=1e-3)
        assert rmse < 1e-3

    def test_too_few_keypoints_raises(self):
        with pytest.raises(ValueError, match="≥3"):
            fallback.affine_from_keypoints([(0, 0), (1, 1)], [(0, 0), (1, 1)])

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length mismatch"):
            fallback.affine_from_keypoints([(0, 0), (1, 1), (2, 2)], [(0, 0)])


# ───────────────────────── CLI E2E (subprocess) ─────────────────────────

class TestCLIEndToEnd:
    """`python -m core.cad_parser ...` 호출이 .v2proj.json 을 만들고
    그 결과를 StructuralModel.from_json()이 받는지 검증."""

    def _make_synthetic_plan(self, path: Path) -> None:
        img = np.full((800, 800), 255, dtype=np.uint8)
        # 3x3 그리드 + 9 컬럼, 그리드 라인은 컬럼 영역을 우회
        grid_xs = [200, 400, 600]
        grid_ys = [200, 400, 600]
        col_size = 16
        half = col_size // 2
        gap = half + 4

        for x in grid_xs:
            prev = 100
            for gy in grid_ys:
                cv2.line(img, (x, prev), (x, gy - gap), 0, 1)
                prev = gy + gap
            cv2.line(img, (x, prev), (x, 700), 0, 1)
        for y in grid_ys:
            prev = 100
            for gx in grid_xs:
                cv2.line(img, (prev, y), (gx - gap, y), 0, 1)
                prev = gx + gap
            cv2.line(img, (prev, y), (700, y), 0, 1)
        # columns
        for gx in grid_xs:
            for gy in grid_ys:
                cv2.rectangle(img, (gx - half, gy - half), (gx + half, gy + half), 0, -1)
        cv2.imwrite(str(path), img)

    def test_cli_produces_loadable_v2proj(self, tmp_path):
        plan_path = tmp_path / "plan_typical.png"
        self._make_synthetic_plan(plan_path)
        out_path = tmp_path / "building.v2proj.json"

        cmd = [
            sys.executable, "-m", "core.cad_parser",
            "--plan", f"{plan_path}:1,2,3",
            "--grid-spacing-x", "8.0",
            "--grid-spacing-y", "8.0",
            "--grid-labels-x", "A,B,C",
            "--grid-labels-y", "1,2,3",
            "--story-elevations", "0,4,8,12",
            "--typical-column", "H-400x400",
            "--typical-beam-x", "H-500x200",
            "--typical-beam-y", "H-400x200",
            "--output", str(out_path),
        ]
        env = {**__import__("os").environ, "PYTHONPATH": str(_MCP_SERVER)}
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)
        assert result.returncode == 0, f"CLI failed: stderr={result.stderr}"
        assert out_path.exists(), "v2proj.json not created"

        # .v2proj.json 형식 검증
        proj = json.loads(out_path.read_text(encoding="utf-8"))
        assert proj["version"] == 3
        assert "model" in proj
        assert "nodes" in proj["model"]
        assert len(proj["model"]["nodes"]) > 0

        # round-trip with StructuralModel
        from core.structural_model import StructuralModel  # noqa
        model = StructuralModel.from_json(proj["model"])
        assert len(model.nodes) > 0
        assert len(model.elements) > 0
