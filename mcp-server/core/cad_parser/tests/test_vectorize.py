"""W1 acceptance test — 합성 그리드 이미지에서 라인 추출 검증.

cad_parser 패키지가 mcp-server/core/ 하위에 있어 import 시 sys.path 조정 필요.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

# mcp-server/ 를 sys.path 에 추가 (server.py 와 동일한 패턴)
_MCP_SERVER = Path(__file__).resolve().parents[3]
if str(_MCP_SERVER) not in sys.path:
    sys.path.insert(0, str(_MCP_SERVER))

from core.cad_parser import vectorize  # noqa: E402
from core.cad_parser.schemas import LineSegment  # noqa: E402


# ---------- 합성 fixture ----------

def _make_grid_image(
    width: int = 600,
    height: int = 600,
    n_x: int = 4,
    n_y: int = 4,
    margin: int = 50,
    line_width: int = 2,
) -> np.ndarray:
    """라인=0(검정), 배경=255(흰색) 합성 그리드 이미지 (grayscale).

    그리드는 (n_x × n_y) intersection을 갖는다.
    바이너리 변환을 거치면 라인=255가 됨 (THRESH_BINARY_INV).
    """
    import cv2
    img = np.full((height, width), 255, dtype=np.uint8)
    xs = np.linspace(margin, width - margin, n_x, dtype=int)
    ys = np.linspace(margin, height - margin, n_y, dtype=int)
    for x in xs:
        cv2.line(img, (int(x), margin), (int(x), height - margin), 0, line_width)
    for y in ys:
        cv2.line(img, (margin, int(y)), (width - margin, int(y)), 0, line_width)
    return img


def _binarize_simple(image: np.ndarray) -> np.ndarray:
    """전처리 단순 버전 — 라인=255 인 binary 반환 (테스트 전용)."""
    import cv2
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    return binary


# ---------- 테스트 ----------

class TestExtractLineSegments:
    def test_returns_list_of_segments(self):
        img = _make_grid_image()
        bin_ = _binarize_simple(img)
        segs = vectorize.extract_line_segments(bin_, min_length=100, threshold=80)
        assert isinstance(segs, list)
        assert len(segs) > 0
        assert all(isinstance(s, LineSegment) for s in segs)

    def test_detects_at_least_n_lines_for_4x4_grid(self):
        """4x4 그리드면 수평 4 + 수직 4 = 최소 8개 라인을 기대.

        Hough가 하나의 라인을 여러 세그먼트로 쪼개기도 하므로 하한만 검증.
        """
        img = _make_grid_image(n_x=4, n_y=4)
        bin_ = _binarize_simple(img)
        segs = vectorize.extract_line_segments(bin_, min_length=400, threshold=100)
        assert len(segs) >= 8, f"expected ≥8 long segments, got {len(segs)}"

    def test_empty_image_returns_empty(self):
        empty = np.zeros((200, 200), dtype=np.uint8)
        segs = vectorize.extract_line_segments(empty, min_length=20)
        assert segs == []


class TestSplitByOrientation:
    def test_grid_lines_split_cleanly_into_horizontal_and_vertical(self):
        img = _make_grid_image(n_x=4, n_y=4)
        bin_ = _binarize_simple(img)
        segs = vectorize.extract_line_segments(bin_, min_length=200, threshold=80)
        horiz, vert, diag = vectorize.split_by_orientation(segs, tolerance_deg=3.0)
        assert len(horiz) >= 4, f"expected ≥4 horizontal lines, got {len(horiz)}"
        assert len(vert) >= 4, f"expected ≥4 vertical lines, got {len(vert)}"
        assert len(diag) == 0, f"expected 0 diagonals for clean grid, got {len(diag)}"

    def test_diagonal_line_classified_correctly(self):
        seg = LineSegment(0, 0, 100, 100)   # 45°
        horiz, vert, diag = vectorize.split_by_orientation([seg], tolerance_deg=5.0)
        assert horiz == []
        assert vert == []
        assert len(diag) == 1


class TestExtractPolygons:
    def test_finds_painted_rectangles(self):
        """6개 페인트된 사각형 (컬럼 후보 모의) → 6개 polygon 검출."""
        import cv2
        img = np.full((400, 400), 255, dtype=np.uint8)
        centers = [(100, 100), (200, 100), (300, 100), (100, 300), (200, 300), (300, 300)]
        for cx, cy in centers:
            cv2.rectangle(img, (cx - 10, cy - 10), (cx + 10, cy + 10), 0, -1)
        bin_ = _binarize_simple(img)
        polys = vectorize.extract_polygons(bin_, min_area=50.0)
        # 각 페인트 사각형이 contour 하나씩 + 가장자리 contour 1개 가능
        assert len(polys) >= 6, f"expected ≥6 polygons, got {len(polys)}"
