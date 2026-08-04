"""벡터화 — Hough 라인 + contour 폴리곤 추출."""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .schemas import LineSegment, Polygon


def extract_line_segments(
    binary: np.ndarray,
    min_length: int = 30,
    max_gap: int = 8,
    threshold: int = 80,
) -> list[LineSegment]:
    """probabilistic Hough → LineSegment 리스트.

    Args:
        binary: cv2.adaptiveThreshold 결과 (라인=255, 배경=0)
        min_length: 최소 라인 길이 (픽셀)
        max_gap: 라인 내 허용 갭 (픽셀)
        threshold: Hough 누적 임계치
    """
    if binary.dtype != np.uint8:
        binary = binary.astype(np.uint8)

    lines = cv2.HoughLinesP(
        binary,
        rho=1,
        theta=np.pi / 180,
        threshold=threshold,
        minLineLength=min_length,
        maxLineGap=max_gap,
    )
    if lines is None:
        return []
    return [
        LineSegment(float(x1), float(y1), float(x2), float(y2))
        for x1, y1, x2, y2 in lines.reshape(-1, 4)
    ]


def split_by_orientation(
    segments: list[LineSegment],
    tolerance_deg: float = 5.0,
) -> tuple[list[LineSegment], list[LineSegment], list[LineSegment]]:
    """수평/수직/사선으로 분리.

    수평: |angle| < tolerance 또는 180 - tolerance < |angle|
    수직: |angle - 90| < tolerance
    그 외: 사선
    """
    horizontal: list[LineSegment] = []
    vertical: list[LineSegment] = []
    diagonal: list[LineSegment] = []
    for seg in segments:
        ang = abs(seg.angle_deg) % 180
        if ang < tolerance_deg or ang > 180 - tolerance_deg:
            horizontal.append(seg)
        elif abs(ang - 90) < tolerance_deg:
            vertical.append(seg)
        else:
            diagonal.append(seg)
    return horizontal, vertical, diagonal


def extract_polygons(
    binary: np.ndarray,
    min_area: float = 100.0,
    max_area: Optional[float] = None,
    approx_epsilon: float = 0.02,
) -> list[Polygon]:
    """closed contour → Polygon (컬럼 후보 등).

    OpenCV findContours + approxPolyDP. 너무 작거나(min_area 미만)
    너무 큰(max_area 초과) contour는 버림.
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[Polygon] = []
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, approx_epsilon * peri, True)
        pts = approx.reshape(-1, 2).astype(np.float32)
        polygons.append(Polygon(points=pts, area=area))
    return polygons
