"""OCR 실패 시 manual 모드 — 사용자가 keypoint를 직접 제공.

핵심 사용 시나리오:
- 그리드 OCR이 라벨을 못 찾아서 register_plans가 실패
- 또는 노후/리모델링 도면처럼 라벨 표기가 부실

이 모듈은 두 가지 경로를 제공:
1. `manual_label_grid(grid_set, labels_x, labels_y)`: 좌표 오름차순으로 라벨 부여
   (grid_detector.assign_labels_manual의 thin wrapper — 일관성 위해 재export)
2. `affine_from_keypoints(pixel_kps, world_kps)`: 4점(또는 그 이상) 매핑으로 affine 직접
"""
from __future__ import annotations

import cv2
import numpy as np

from .grid_detector import assign_labels_manual
from .schemas import GridSet


__all__ = ["manual_label_grid", "affine_from_keypoints"]


# grid_detector.assign_labels_manual의 별칭 (네이밍 일관성)
manual_label_grid = assign_labels_manual


def affine_from_keypoints(
    pixel_kps: list[tuple[float, float]],
    world_kps: list[tuple[float, float]],
) -> tuple[np.ndarray, float]:
    """사용자가 직접 클릭한 keypoint 쌍으로 픽셀→world affine 직접 계산.

    Args:
        pixel_kps: [(px1, py1), (px2, py2), …] (≥3개 필수)
        world_kps: [(wx1, wy1), (wx2, wy2), …] (m, 동일 길이)

    Returns:
        (M3, rmse_m). M3는 3×3 homogeneous affine.

    Raises:
        ValueError: keypoint 수가 3개 미만이거나 길이 불일치
    """
    if len(pixel_kps) != len(world_kps):
        raise ValueError(
            f"pixel_kps and world_kps length mismatch: "
            f"{len(pixel_kps)} vs {len(world_kps)}"
        )
    if len(pixel_kps) < 3:
        raise ValueError(f"Need ≥3 keypoints, got {len(pixel_kps)}")

    src = np.array(pixel_kps, dtype=np.float32)
    dst = np.array(world_kps, dtype=np.float32)
    M, _ = cv2.estimateAffine2D(src, dst, method=cv2.LMEDS)
    if M is None:
        raise ValueError("cv2.estimateAffine2D failed (degenerate keypoints?)")

    M3 = np.vstack([M, [0.0, 0.0, 1.0]])
    ones = np.ones((src.shape[0], 1), dtype=np.float32)
    src_h = np.hstack([src, ones])
    pred = (M3 @ src_h.T).T[:, :2]
    rmse = float(np.sqrt(np.mean(np.sum((pred - dst) ** 2, axis=1))))
    return M3, rmse
