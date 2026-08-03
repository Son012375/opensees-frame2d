"""그리드 라인 클러스터링 + 버블 라벨 OCR.

핵심 흐름:
1. vectorize.split_by_orientation으로 분리된 수평/수직 LineSegment
2. cluster_grid_lines: 같은 좌표대의 라인들을 평균 좌표 GridLine 하나로 통합
3. compute_intersections: 수직×수평 모든 교점 생성
4. ocr_grid_labels: 버블 영역(가장자리) ROI OCR → 각 GridLine에 라벨 부여
5. label_intersections: 라벨된 GridLine 쌍으로 GridIntersection.x_label/y_label 채움

EasyOCR Reader는 모듈 레벨에서 lazy 캐싱 (생성/모델 로드 비용 큼).
(이전 시도였던 PaddleOCR는 Windows + paddlepaddle 3.x의 oneDNN 호환 버그로 제외)
"""
from __future__ import annotations

import re
from dataclasses import replace
from typing import Optional

import cv2
import numpy as np

from .schemas import GridIntersection, GridLine, GridSet, LineSegment
from .vectorize import split_by_orientation


# ─────────────────────────────────────────────────────────────
# 1) Line clustering
# ─────────────────────────────────────────────────────────────

def cluster_grid_lines(
    segments: list[LineSegment],
    orientation: str,
    gap_threshold: float = 8.0,
    min_segments_per_cluster: int = 1,
) -> list[GridLine]:
    """orientation별 LineSegment들을 좌표 클러스터로 묶어 GridLine 리스트 생성.

    Args:
        segments: 같은 방향(수평 또는 수직)으로 사전 필터링된 segment들
        orientation: "vertical" or "horizontal"
        gap_threshold: 같은 클러스터로 묶는 좌표 차 (픽셀)
        min_segments_per_cluster: 이 수 미만의 segment로 구성된 클러스터는 제거

    Returns:
        클러스터 1개 = GridLine 1개 (coord_px = 평균 좌표). 좌표 오름차순.
    """
    if not segments:
        return []

    if orientation == "vertical":
        coords = np.array([(s.x1 + s.x2) / 2 for s in segments])
    elif orientation == "horizontal":
        coords = np.array([(s.y1 + s.y2) / 2 for s in segments])
    else:
        raise ValueError(f"orientation must be 'vertical' or 'horizontal', got {orientation!r}")

    sort_idx = np.argsort(coords)
    sorted_coords = coords[sort_idx]

    clusters: list[list[float]] = [[float(sorted_coords[0])]]
    for c in sorted_coords[1:]:
        if c - clusters[-1][-1] <= gap_threshold:
            clusters[-1].append(float(c))
        else:
            clusters.append([float(c)])

    grid_lines: list[GridLine] = []
    for cluster in clusters:
        if len(cluster) < min_segments_per_cluster:
            continue
        coord_mean = float(np.mean(cluster))
        grid_lines.append(GridLine(orientation=orientation, coord_px=coord_mean))
    return grid_lines


def compute_intersections(grid_set: GridSet) -> list[GridIntersection]:
    """현재 라벨된(또는 미라벨) vertical × horizontal 모든 교점 생성."""
    intersections: list[GridIntersection] = []
    for v in grid_set.vertical_lines:
        for h in grid_set.horizontal_lines:
            intersections.append(
                GridIntersection(
                    x_label=v.label or "",
                    y_label=h.label or "",
                    px=v.coord_px,
                    py=h.coord_px,
                )
            )
    return intersections


def detect_grid(
    binary: np.ndarray,
    min_line_length: int = 80,
    hough_threshold: int = 60,
    gap_threshold: float = 8.0,
    angle_tol_deg: float = 3.0,
) -> GridSet:
    """binary 이미지에서 그리드 라인을 검출하여 GridSet 반환 (라벨 없음).

    내부적으로 vectorize.extract_line_segments → split_by_orientation
    → cluster_grid_lines 흐름을 한 번에 수행.
    """
    from .vectorize import extract_line_segments
    segments = extract_line_segments(
        binary,
        min_length=min_line_length,
        threshold=hough_threshold,
        max_gap=8,
    )
    horiz, vert, _diag = split_by_orientation(segments, tolerance_deg=angle_tol_deg)

    vert_lines = cluster_grid_lines(vert, "vertical", gap_threshold=gap_threshold)
    horiz_lines = cluster_grid_lines(horiz, "horizontal", gap_threshold=gap_threshold)

    grid_set = GridSet(vertical_lines=vert_lines, horizontal_lines=horiz_lines)
    grid_set.intersections = compute_intersections(grid_set)
    return grid_set


# ─────────────────────────────────────────────────────────────
# 2) Label OCR
# ─────────────────────────────────────────────────────────────

_LABEL_PATTERN = re.compile(r"^[A-Za-z0-9]{1,2}$")
# 축별 권장 패턴 — 한국 구조도면 관행 (vertical=A,B,C / horizontal=1,2,3) 기반
_LABEL_PATTERN_VERTICAL = re.compile(r"^[A-Za-z]$")
_LABEL_PATTERN_HORIZONTAL = re.compile(r"^[0-9]{1,2}$")

_easyocr_reader = None


def _get_ocr_reader():
    """EasyOCR Reader 싱글톤 — 생성 비용이 크므로 1회만."""
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        _easyocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _easyocr_reader


def _ocr_roi(
    roi: np.ndarray, min_side_for_ocr: int = 200
) -> list[tuple[str, float]]:
    """EasyOCR를 ROI에 적용하고 (text, score) 리스트 반환.

    좁은 ROI(예: '1' 단일 숫자)에 대비해:
    - 짧은 변이 min_side_for_ocr 미만이면 upscale (INTER_CUBIC)
    - EasyOCR detection threshold를 살짝 낮춤 (text_threshold, low_text, min_size)
    - 그래도 빈 결과면 ROI 전체를 단일 박스로 강제 recognize 시도

    EasyOCR.readtext output: [(bbox, text, confidence), ...]
    """
    if roi.size == 0:
        return []
    if roi.ndim == 2:
        roi_input = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    else:
        roi_input = roi

    h, w = roi_input.shape[:2]
    short_side = min(h, w)
    if short_side < min_side_for_ocr and short_side > 0:
        scale = min_side_for_ocr / short_side
        new_w = int(w * scale)
        new_h = int(h * scale)
        roi_input = cv2.resize(roi_input, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        h, w = roi_input.shape[:2]

    reader = _get_ocr_reader()
    results = reader.readtext(
        roi_input,
        text_threshold=0.5,
        low_text=0.3,
        link_threshold=0.3,
        min_size=5,
    )

    if not results:
        # ROI 전체를 단일 박스로 보고 recognize만 시도 (detection 우회)
        try:
            full_box = [[0, 0], [w, 0], [w, h], [0, h]]
            recog = reader.recognize(roi_input, horizontal_list=[[0, w, 0, h]], free_list=[])
            results = [(full_box, text, float(conf)) for _bbox, text, conf in recog]
        except Exception:
            results = []

    return [(text.strip(), float(conf)) for _bbox, text, conf in results]


def _extract_label_candidates(
    image: np.ndarray,
    grid_line: GridLine,
    image_shape: tuple[int, int],
    margin_px: int = 100,
    roi_thickness_px: int = 80,
) -> list[np.ndarray]:
    """단일 GridLine 양 끝쪽 버블 영역의 ROI 2장 반환 (top/bottom 또는 left/right).

    그리드 라인 영역 침범을 피하기 위해 ROI 깊이를 정확히 margin_px로 제한.
    이렇게 하면 vertical 라인 ROI에 horizontal 라벨이 들어오는 것을 방지.

    image_shape = (H, W)
    """
    H, W = image_shape
    rois: list[np.ndarray] = []
    half = roi_thickness_px // 2
    if grid_line.orientation == "vertical":
        cx = int(grid_line.coord_px)
        x0 = max(0, cx - half)
        x1 = min(W, cx + half)
        top = image[0 : min(H, margin_px), x0:x1]
        bottom = image[max(0, H - margin_px) : H, x0:x1]
        rois = [top, bottom]
    else:
        cy = int(grid_line.coord_px)
        y0 = max(0, cy - half)
        y1 = min(H, cy + half)
        left = image[y0:y1, 0 : min(W, margin_px)]
        right = image[y0:y1, max(0, W - margin_px) : W]
        rois = [left, right]
    return [r for r in rois if r.size > 0]


def ocr_grid_labels(
    image: np.ndarray,
    grid_set: GridSet,
    score_threshold: float = 0.5,
    margin_px: int = 100,
    roi_thickness_px: int = 80,
    vertical_pattern: re.Pattern = _LABEL_PATTERN_VERTICAL,
    horizontal_pattern: re.Pattern = _LABEL_PATTERN_HORIZONTAL,
) -> GridSet:
    """각 GridLine의 양 끝 버블 영역에 OCR → label 채워진 GridSet 반환.

    Args:
        image: 원본 grayscale 또는 BGR 이미지
        grid_set: detect_grid 결과
        score_threshold: EasyOCR confidence 하한
        margin_px: 그리드 영역 바깥쪽 ROI 깊이 (그리드 라인 침범 방지)
        roi_thickness_px: 라인 방향 ROI 폭
        vertical_pattern: 수직 라인 라벨 정규식 (default: 영문 1자)
        horizontal_pattern: 수평 라인 라벨 정규식 (default: 숫자 1-2자)

    Returns:
        새 GridSet (입력 미변경). 인식 실패한 라인은 label=None 유지.
    """
    image_shape = image.shape[:2]

    def _ocr_line(gl: GridLine, pattern: re.Pattern) -> GridLine:
        rois = _extract_label_candidates(
            image, gl, image_shape, margin_px=margin_px, roi_thickness_px=roi_thickness_px
        )
        best: Optional[tuple[str, float]] = None
        for roi in rois:
            for text, score in _ocr_roi(roi):
                if not pattern.match(text):
                    continue
                if score < score_threshold:
                    continue
                if best is None or score > best[1]:
                    best = (text.upper(), score)
        return replace(gl, label=best[0]) if best else gl

    new_vert = [_ocr_line(gl, vertical_pattern) for gl in grid_set.vertical_lines]
    new_horiz = [_ocr_line(gl, horizontal_pattern) for gl in grid_set.horizontal_lines]
    out = GridSet(vertical_lines=new_vert, horizontal_lines=new_horiz)
    out.intersections = compute_intersections(out)
    return out


# ─────────────────────────────────────────────────────────────
# 3) Manual label assignment (OCR 우회)
# ─────────────────────────────────────────────────────────────

def assign_labels_manual(
    grid_set: GridSet,
    vertical_labels: list[str],
    horizontal_labels: list[str],
) -> GridSet:
    """OCR 없이 수동으로 라벨 부여 (좌표 오름차순 매칭).

    vertical_labels: x좌표 오름차순으로 라벨 부여 (예: ["A","B","C","D"])
    horizontal_labels: y좌표 오름차순으로 라벨 부여 (예: ["1","2","3","4"])
    길이가 라인 수와 다르면 ValueError.
    """
    if len(vertical_labels) != len(grid_set.vertical_lines):
        raise ValueError(
            f"vertical_labels count {len(vertical_labels)} ≠ "
            f"vertical_lines count {len(grid_set.vertical_lines)}"
        )
    if len(horizontal_labels) != len(grid_set.horizontal_lines):
        raise ValueError(
            f"horizontal_labels count {len(horizontal_labels)} ≠ "
            f"horizontal_lines count {len(grid_set.horizontal_lines)}"
        )

    new_vert = [replace(gl, label=lab) for gl, lab in zip(grid_set.vertical_lines, vertical_labels)]
    new_horiz = [
        replace(gl, label=lab) for gl, lab in zip(grid_set.horizontal_lines, horizontal_labels)
    ]
    out = GridSet(vertical_lines=new_vert, horizontal_lines=new_horiz)
    out.intersections = compute_intersections(out)
    return out


# ─────────────────────────────────────────────────────────────
# 4) Sanity checks
# ─────────────────────────────────────────────────────────────

def validate_label_monotonic(grid_set: GridSet) -> list[str]:
    """라벨 시퀀스가 단조 증가하는지 검증.

    영문자(A,B,C…)와 숫자(1,2,3…)는 각각 ord/숫자값으로 비교.
    위반 사항을 문자열 리스트로 반환 (비어 있으면 정상).
    """
    warnings: list[str] = []

    def _label_key(label: Optional[str]) -> Optional[float]:
        if not label:
            return None
        if label.isdigit():
            return float(int(label))
        if len(label) == 1 and label.isalpha():
            return float(ord(label.upper()))
        return None  # 무시

    for axis_name, lines in [
        ("vertical", grid_set.vertical_lines),
        ("horizontal", grid_set.horizontal_lines),
    ]:
        keys = [(_label_key(gl.label), gl.coord_px) for gl in lines if gl.label]
        sorted_keys = sorted(keys, key=lambda x: x[1])  # 좌표 오름차순
        last_k: Optional[float] = None
        for k, _ in sorted_keys:
            if k is None:
                continue
            if last_k is not None and k <= last_k:
                warnings.append(
                    f"{axis_name} 라벨이 단조 증가가 아님 (좌표순으로 {sorted_keys})"
                )
                break
            last_k = k
    return warnings
