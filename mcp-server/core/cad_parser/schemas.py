"""cad_parser 자료구조 정의.

전 파이프라인이 공유하는 dataclass 모음. 단계별 모듈은 이 타입들을 입출력으로 사용.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np


SheetKind = Literal["plan", "elevation"]


@dataclass
class DrawingSheet:
    """단일 도면 시트 (raster 이미지 + 메타)."""

    sheet_id: str
    kind: SheetKind
    source_path: Path
    image: np.ndarray          # grayscale ndarray (H, W) uint8
    dpi: Optional[float] = None
    label: Optional[str] = None   # 예: "1F plan", "elevation A"


@dataclass
class LineSegment:
    """벡터화 1차 결과 — Hough 검출 라인."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def length(self) -> float:
        return float(np.hypot(self.x2 - self.x1, self.y2 - self.y1))

    @property
    def angle_deg(self) -> float:
        return float(np.degrees(np.arctan2(self.y2 - self.y1, self.x2 - self.x1)))


@dataclass
class Polygon:
    """벡터화 2차 결과 — 컬럼 후보 폐곡선."""

    points: np.ndarray   # (N, 2)
    area: float


@dataclass
class GridLine:
    """그리드 라인 (방향 + 좌표).

    orientation="vertical":   수직 라인 (column 그리드). coord_px = x 픽셀 좌표.
    orientation="horizontal": 수평 라인 (row 그리드). coord_px = y 픽셀 좌표.
    """

    orientation: Literal["vertical", "horizontal"]
    coord_px: float
    label: Optional[str] = None


@dataclass
class GridIntersection:
    """그리드 교점."""

    x_label: str
    y_label: str
    px: float
    py: float


@dataclass
class GridSet:
    """단일 시트에서 검출된 그리드.

    vertical_lines:   수직 그리드 라인들 (보통 A,B,C 라벨)
    horizontal_lines: 수평 그리드 라인들 (보통 1,2,3 라벨)
    """

    vertical_lines: list[GridLine] = field(default_factory=list)
    horizontal_lines: list[GridLine] = field(default_factory=list)
    intersections: list[GridIntersection] = field(default_factory=list)


@dataclass
class RegisteredFrame:
    """다중 시트 정합 결과 — 공유 world 좌표계."""

    plan_affines: dict[str, np.ndarray]        # sheet_id → 3x3 affine
    elevation_affines: dict[str, np.ndarray]
    world_grid_x: dict[str, float]             # 라벨(A,B,…) → world X (m)
    world_grid_y: dict[str, float]             # 라벨(1,2,…) → world Y (m)
    world_grid_z: list[float]                  # 층별 elevation (m)
    rmse_px: dict[str, float] = field(default_factory=dict)


@dataclass
class ColumnCandidate:
    grid_x_label: str
    grid_y_label: str
    story_from: int
    story_to: int
    confidence: float = 1.0


@dataclass
class BeamCandidate:
    """입면도에서 검출된 보 후보.

    span_along="vertical_grid"   → 수직 그리드(A↔B)를 따라가는 보 (X 방향 보)
    span_along="horizontal_grid" → 수평 그리드(1↔2)를 따라가는 보 (Y 방향 보)

    Fields:
        span_along:        보 방향
        from_label:        보가 시작하는 그리드 라벨 (예: "A")
        to_label:          보가 끝나는 그리드 라벨 (예: "B")
        transverse_label:  직교 방향 라벨 — 어느 평면 위치인가
            예) span_along=vertical_grid, transverse_label="2"
                → A↔B 보가 2-라인 위치 (즉 노드 (A,2), (B,2) 사이)
        story:             1-based 층 번호 (story=N → world_grid_z[N] elevation)
    """
    span_along: Literal["vertical_grid", "horizontal_grid"]
    from_label: str
    to_label: str
    transverse_label: str
    story: int
    confidence: float = 1.0


@dataclass
class TypicalSectionSpec:
    """사용자가 form으로 입력한 typical 단면."""

    column: str        # 예: "H-400x400"
    beam_x: str        # X 방향 보 (예: "H-500x200")
    beam_y: str
    material: str = "SM355"


@dataclass
class CADExtractionReport:
    """파싱 1회분 진단 리포트 — UI/CLI에 노출."""

    detected_columns: int = 0
    detected_beams: int = 0
    ocr_label_count: int = 0
    registration_rmse_px: float = 0.0
    warnings: list[str] = field(default_factory=list)
    sheets_processed: list[str] = field(default_factory=list)
