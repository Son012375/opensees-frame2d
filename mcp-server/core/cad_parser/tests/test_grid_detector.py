"""W2 acceptance tests — 그리드 클러스터링 + 라벨 부여 + (옵션) OCR."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

_MCP_SERVER = Path(__file__).resolve().parents[3]
if str(_MCP_SERVER) not in sys.path:
    sys.path.insert(0, str(_MCP_SERVER))

from core.cad_parser import grid_detector, vectorize  # noqa: E402
from core.cad_parser.schemas import GridLine, GridSet, LineSegment  # noqa: E402


# ───────────────────────── Synthetic fixtures ─────────────────────────

def _draw_text_pil(img: np.ndarray, text: str, xy: tuple[int, int], font_size: int = 36) -> None:
    """PIL TTF 폰트로 텍스트 그리기. cv2.putText의 Hershey 폰트는 OCR이 헷갈림.

    img는 grayscale numpy 배열 (in-place 수정).
    xy는 텍스트 좌상단 (PIL 관행).
    """
    from PIL import Image, ImageDraw, ImageFont
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    # Windows 기본 폰트 우선, 없으면 PIL default
    for font_name in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf"):
        try:
            font = ImageFont.truetype(font_name, font_size)
            break
        except (OSError, IOError):
            continue
    else:
        font = ImageFont.load_default()
    draw.text(xy, text, fill=0, font=font)
    img[:] = np.array(pil_img)


def make_grid_image(
    width: int = 800,
    height: int = 800,
    n_vertical: int = 4,
    n_horizontal: int = 4,
    margin: int = 100,
    line_width: int = 2,
    with_labels: bool = False,
    label_font_size: int = 36,
) -> tuple[np.ndarray, list[float], list[float]]:
    """라인=0 / 배경=255 합성 그리드 이미지.

    `with_labels=True`이면 PIL TTF 폰트로 각 그리드 라인 양 끝에 라벨 표기.

    Returns: (image, vertical_x_coords, horizontal_y_coords)
    """
    img = np.full((height, width), 255, dtype=np.uint8)
    xs = np.linspace(margin, width - margin, n_vertical, dtype=int)
    ys = np.linspace(margin, height - margin, n_horizontal, dtype=int)

    for x in xs:
        cv2.line(img, (int(x), margin), (int(x), height - margin), 0, line_width)
    for y in ys:
        cv2.line(img, (margin, int(y)), (width - margin, int(y)), 0, line_width)

    if with_labels:
        labels_v = [chr(ord("A") + i) for i in range(n_vertical)]
        labels_h = [str(i + 1) for i in range(n_horizontal)]
        half_font = label_font_size // 2
        for x, lab in zip(xs, labels_v):
            _draw_text_pil(img, lab, (int(x) - half_font // 2, margin // 4), label_font_size)
            _draw_text_pil(
                img, lab, (int(x) - half_font // 2, height - margin + margin // 4), label_font_size
            )
        for y, lab in zip(ys, labels_h):
            _draw_text_pil(img, lab, (margin // 4, int(y) - half_font), label_font_size)
            _draw_text_pil(
                img, lab, (width - margin + margin // 4, int(y) - half_font), label_font_size
            )

    return img, xs.tolist(), ys.tolist()


def _binarize(img: np.ndarray) -> np.ndarray:
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    return binary


# ───────────────────────── cluster_grid_lines ─────────────────────────

class TestClusterGridLines:
    def test_vertical_segments_become_distinct_grid_lines(self):
        # x=100, 200, 300 각각에 2개의 segment 노이즈
        segs = [
            LineSegment(100, 0, 100, 200),
            LineSegment(101, 50, 101, 250),    # 거의 같은 위치
            LineSegment(200, 10, 200, 220),
            LineSegment(201, 80, 199, 280),
            LineSegment(300, 0, 300, 200),
        ]
        out = grid_detector.cluster_grid_lines(segs, "vertical", gap_threshold=5.0)
        assert len(out) == 3, [g.coord_px for g in out]
        # 좌표가 오름차순 + 평균값 근접
        coords = [g.coord_px for g in out]
        assert coords == sorted(coords)
        assert abs(coords[0] - 100) < 2
        assert abs(coords[1] - 200) < 2
        assert abs(coords[2] - 300) < 2
        assert all(g.orientation == "vertical" for g in out)

    def test_horizontal_orientation(self):
        segs = [
            LineSegment(0, 100, 200, 100),
            LineSegment(50, 200, 250, 201),
        ]
        out = grid_detector.cluster_grid_lines(segs, "horizontal", gap_threshold=5.0)
        assert len(out) == 2
        assert all(g.orientation == "horizontal" for g in out)
        # coord_px = y 평균
        assert abs(out[0].coord_px - 100) < 1
        assert abs(out[1].coord_px - 200.5) < 1

    def test_empty_input(self):
        assert grid_detector.cluster_grid_lines([], "vertical") == []

    def test_invalid_orientation_raises(self):
        with pytest.raises(ValueError):
            grid_detector.cluster_grid_lines([LineSegment(0, 0, 1, 1)], "diagonal")


# ───────────────────────── detect_grid ─────────────────────────

class TestDetectGrid:
    def test_4x4_grid_yields_4_and_4(self):
        img, _, _ = make_grid_image(n_vertical=4, n_horizontal=4)
        binary = _binarize(img)
        grid = grid_detector.detect_grid(binary, min_line_length=400, gap_threshold=8.0)
        assert len(grid.vertical_lines) == 4, [g.coord_px for g in grid.vertical_lines]
        assert len(grid.horizontal_lines) == 4, [g.coord_px for g in grid.horizontal_lines]
        # 4x4 → 16 교점
        assert len(grid.intersections) == 16

    def test_5x3_asymmetric_grid(self):
        img, _, _ = make_grid_image(n_vertical=5, n_horizontal=3)
        binary = _binarize(img)
        grid = grid_detector.detect_grid(binary, min_line_length=400)
        assert len(grid.vertical_lines) == 5
        assert len(grid.horizontal_lines) == 3


# ───────────────────────── Manual label assignment ─────────────────────────

class TestAssignLabelsManual:
    def test_labels_attached_in_coordinate_order(self):
        gs = GridSet(
            vertical_lines=[
                GridLine("vertical", 100),
                GridLine("vertical", 200),
                GridLine("vertical", 300),
            ],
            horizontal_lines=[
                GridLine("horizontal", 50),
                GridLine("horizontal", 250),
            ],
        )
        out = grid_detector.assign_labels_manual(
            gs, vertical_labels=["A", "B", "C"], horizontal_labels=["1", "2"]
        )
        assert [g.label for g in out.vertical_lines] == ["A", "B", "C"]
        assert [g.label for g in out.horizontal_lines] == ["1", "2"]
        # intersections에 라벨도 채워졌는지
        assert any(i.x_label == "A" and i.y_label == "1" for i in out.intersections)
        assert any(i.x_label == "C" and i.y_label == "2" for i in out.intersections)

    def test_label_count_mismatch_raises(self):
        gs = GridSet(vertical_lines=[GridLine("vertical", 0)])
        with pytest.raises(ValueError):
            grid_detector.assign_labels_manual(gs, ["A", "B"], [])


# ───────────────────────── Monotonic check ─────────────────────────

class TestValidateLabelMonotonic:
    def test_clean_alphanumeric_sequence_passes(self):
        gs = GridSet(
            vertical_lines=[
                GridLine("vertical", 100, "A"),
                GridLine("vertical", 200, "B"),
                GridLine("vertical", 300, "C"),
            ],
            horizontal_lines=[
                GridLine("horizontal", 50, "1"),
                GridLine("horizontal", 150, "2"),
            ],
        )
        assert grid_detector.validate_label_monotonic(gs) == []

    def test_reversed_labels_flagged(self):
        gs = GridSet(
            vertical_lines=[
                GridLine("vertical", 100, "C"),
                GridLine("vertical", 200, "B"),
                GridLine("vertical", 300, "A"),
            ],
        )
        warnings = grid_detector.validate_label_monotonic(gs)
        assert len(warnings) == 1
        assert "vertical" in warnings[0]


# ───────────────────────── E2E: 그리드 검출 → manual 라벨 → 교점 ─────────────────────────

class TestE2EManualPath:
    def test_synthetic_4x4_recovers_grid_with_labels(self):
        img, true_xs, true_ys = make_grid_image(n_vertical=4, n_horizontal=4)
        binary = _binarize(img)

        grid = grid_detector.detect_grid(binary, min_line_length=400)
        grid = grid_detector.assign_labels_manual(
            grid,
            vertical_labels=["A", "B", "C", "D"],
            horizontal_labels=["1", "2", "3", "4"],
        )

        # 라벨 sanity
        assert grid_detector.validate_label_monotonic(grid) == []

        # 라벨된 교점에서 A-1 → 좌측 상단, D-4 → 우측 하단
        a1 = next(i for i in grid.intersections if i.x_label == "A" and i.y_label == "1")
        d4 = next(i for i in grid.intersections if i.x_label == "D" and i.y_label == "4")
        assert a1.px < d4.px
        assert a1.py < d4.py

        # 그리드 사이 간격이 균등한지 (4점 등간격)
        v_coords = sorted([g.coord_px for g in grid.vertical_lines])
        gaps = np.diff(v_coords)
        assert np.std(gaps) < 5.0


# ───────────────────────── (Slow) EasyOCR ─────────────────────────
# EasyOCR 첫 호출이 무거우므로 슬로우 마커로 격리.
# 실행: pytest -m slow cad_parser/tests/test_grid_detector.py

@pytest.mark.slow
class TestOCR:
    """OCR pipeline 작동 + 일부 라벨 인식 검증.

    합성 폰트(PIL TTF Arial 등)는 실제 도면 폰트와 다를 수 있어 100% 인식은 어려움.
    W2 acceptance: pipeline이 작동하고 양 축에서 최소 ≥1개 라벨 인식.
    실제 도면 정확도(recall 90%+) 검증은 W6 실 도면 테스트로.
    """

    def test_ocr_recognizes_some_labels(self):
        img, _, _ = make_grid_image(
            n_vertical=3,
            n_horizontal=3,
            margin=140,
            with_labels=True,
            label_font_size=40,
        )
        binary = _binarize(img)
        grid = grid_detector.detect_grid(binary, min_line_length=400)
        labeled = grid_detector.ocr_grid_labels(
            img, grid, score_threshold=0.4, margin_px=130, roi_thickness_px=100
        )

        labels_v = [gl.label for gl in labeled.vertical_lines]
        labels_h = [gl.label for gl in labeled.horizontal_lines]

        n_v = sum(1 for lab in labels_v if lab and lab in {"A", "B", "C"})
        n_h = sum(1 for lab in labels_h if lab and lab in {"1", "2", "3"})
        # 양 축 각각 ≥1개 인식
        assert n_v >= 1, f"vertical OCR recognized nothing: {labels_v}"
        assert n_h >= 1, f"horizontal OCR recognized nothing: {labels_h}"
        # 누적 ≥3 (총 6 라벨 중 절반 이상)
        assert n_v + n_h >= 3, (
            f"total recall too low: vertical={labels_v}, horizontal={labels_h}"
        )

    def test_paddle_grid_image_persists_for_visual_inspection(self, tmp_path):
        """디버그 PNG 산출 — Done 기준 확인용 (육안 검토 가능)."""
        img, _, _ = make_grid_image(
            n_vertical=4, n_horizontal=4, margin=140, with_labels=True, label_font_size=40
        )
        debug = Path("outputs/cad_debug")
        debug.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug / "w2_grid_synth.png"), img)
        assert (debug / "w2_grid_synth.png").exists()
