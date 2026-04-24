from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class ColorSegConfig:
    hsv_lower: Tuple[int, int, int]
    hsv_upper: Tuple[int, int, int]
    hsv_lower_2: Tuple[int, int, int]
    hsv_upper_2: Tuple[int, int, int]
    min_area_px: int


class LowPassVec3:
    def __init__(self, alpha: float) -> None:
        self.alpha = alpha
        self.value: Optional[np.ndarray] = None

    def update(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).reshape(3)
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1.0 - self.alpha) * self.value
        return self.value.copy()


def segment_object(
    frame_bgr: np.ndarray,
    cfg: ColorSegConfig,
) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, np.array(cfg.hsv_lower), np.array(cfg.hsv_upper))
    mask2 = cv2.inRange(hsv, np.array(cfg.hsv_lower_2), np.array(cfg.hsv_upper_2))
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask, None

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area < cfg.min_area_px:
        return mask, None

    x, y, w, h = cv2.boundingRect(contour)
    return mask, (x, y, w, h)


def transform_point(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    p_h = np.ones(4, dtype=np.float64)
    p_h[:3] = p
    return (T @ p_h)[:3]


def deproject_pixel_to_camera(u: np.ndarray, v: np.ndarray, z: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.stack([x, y, z], axis=-1)


def estimate_object_position_from_depth(
    mask: np.ndarray,
    depth_m: np.ndarray,
    camera_matrix: np.ndarray,
    T_camera_to_board: np.ndarray,
    *,
    min_valid_depth_m: float = 0.05,
    max_valid_depth_m: float = 5.0,
    sample_stride: int = 2,
) -> Optional[np.ndarray]:
    """
    Estimate a 3D object position from the segmented mask and aligned depth.

    Returns the median 3D point of valid visible object pixels in board coordinates.
    This is a visible-surface centroid proxy, not a true rigid-body center of mass.
    """
    if mask is None or depth_m is None:
        return None

    ys, xs = np.nonzero(mask > 0)
    if xs.size == 0:
        return None

    if sample_stride > 1:
        xs = xs[::sample_stride]
        ys = ys[::sample_stride]

    z = depth_m[ys, xs].astype(np.float64)
    valid = np.isfinite(z) & (z >= min_valid_depth_m) & (z <= max_valid_depth_m)
    if not np.any(valid):
        return None

    xs = xs[valid].astype(np.float64)
    ys = ys[valid].astype(np.float64)
    z = z[valid]

    p_cam = deproject_pixel_to_camera(xs, ys, z, camera_matrix)
    ones = np.ones((p_cam.shape[0], 1), dtype=np.float64)
    p_cam_h = np.concatenate([p_cam, ones], axis=1)
    p_board = (T_camera_to_board @ p_cam_h.T).T[:, :3]

    return np.median(p_board, axis=0)


def identity_quat_wxyz() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
