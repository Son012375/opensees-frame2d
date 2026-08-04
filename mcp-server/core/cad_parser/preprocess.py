"""도면 raster 전처리 — PDF→PNG, grayscale, denoise, binarize."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .schemas import DrawingSheet, SheetKind


def load_sheet(
    path: str | Path,
    sheet_id: str,
    kind: SheetKind,
    label: Optional[str] = None,
    pdf_page: int = 0,
    pdf_zoom: float = 2.0,
) -> DrawingSheet:
    """이미지 또는 PDF 1페이지를 DrawingSheet으로 로드.

    PDF는 pymupdf로 zoom 배율을 적용해 PNG 렌더 후 grayscale 변환.
    raster는 OpenCV로 직접 로드 후 grayscale 변환.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        image, dpi = _render_pdf_page(path, pdf_page, pdf_zoom)
    elif suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"OpenCV failed to decode: {path}")
        dpi = None
    else:
        raise ValueError(f"Unsupported drawing format: {suffix}")

    return DrawingSheet(
        sheet_id=sheet_id,
        kind=kind,
        source_path=path,
        image=image,
        dpi=dpi,
        label=label,
    )


def _render_pdf_page(pdf_path: Path, page_index: int, zoom: float) -> tuple[np.ndarray, float]:
    import fitz  # pymupdf
    doc = fitz.open(str(pdf_path))
    try:
        page = doc.load_page(page_index)
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csGRAY, alpha=False)
        buf = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
        return buf.copy(), 72.0 * zoom
    finally:
        doc.close()


def binarize(sheet: DrawingSheet, block_size: int = 35, c: int = 10) -> np.ndarray:
    """adaptive threshold → 흑백 binary (라인=255, 배경=0).

    block_size는 홀수여야 함. 도면 해상도에 따라 조정.
    """
    if block_size % 2 == 0:
        block_size += 1
    blurred = cv2.GaussianBlur(sheet.image, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        blurred,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=block_size,
        C=c,
    )
    return binary


def deskew(image: np.ndarray, max_angle_deg: float = 5.0) -> tuple[np.ndarray, float]:
    """Hough 기반 미세 회전 보정.

    검출 라인의 dominant 각도가 max_angle_deg 이내일 때만 보정.
    반환: (회전된 이미지, 적용된 각도°)
    """
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    edges = cv2.Canny(image, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=120,
        minLineLength=min(image.shape) // 4,
        maxLineGap=8,
    )
    if lines is None or len(lines) == 0:
        return image, 0.0

    angles = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # 0/90 근방으로 wrap
        ang = ang % 90
        if ang > 45:
            ang -= 90
        if abs(ang) <= max_angle_deg:
            angles.append(ang)

    if not angles:
        return image, 0.0

    skew = float(np.median(angles))
    if abs(skew) < 0.05:
        return image, 0.0

    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), skew, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated, skew
