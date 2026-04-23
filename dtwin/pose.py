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


def segment_red_object(frame_bgr: np.ndarray, cfg: ColorSegConfig) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]:
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


def pixel_to_camera_ray(u: float, v: float, camera_matrix: np.ndarray) -> np.ndarray:
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    x = (u - cx) / fx
    y = (v - cy) / fy
    ray = np.array([x, y, 1.0], dtype=np.float64)
    return ray / np.linalg.norm(ray)


def intersect_ray_with_plane(ray_cam: np.ndarray, plane_z_in_board: float, T_camera_board: np.ndarray) -> Optional[np.ndarray]:
    # We assume the object lies on the board plane z = plane_z_in_board in board/world coordinates.
    T_board_camera = np.linalg.inv(T_camera_board)

    cam_origin_board = T_board_camera[:3, 3]
    ray_dir_board = T_board_camera[:3, :3] @ ray_cam

    if abs(ray_dir_board[2]) < 1e-8:
        return None

    t = (plane_z_in_board - cam_origin_board[2]) / ray_dir_board[2]
    if t <= 0:
        return None

    p_board = cam_origin_board + t * ray_dir_board
    return p_board


def estimate_object_position_on_board(
    bbox: Tuple[int, int, int, int],
    camera_matrix: np.ndarray,
    T_camera_board: np.ndarray,
    object_height_m: float,
) -> Optional[np.ndarray]:
    x, y, w, h = bbox
    u = x + 0.5 * w
    v = y + 0.5 * h

    ray_cam = pixel_to_camera_ray(u, v, camera_matrix)
    # Approximate object center as lying half-height above the board plane.
    return intersect_ray_with_plane(ray_cam, plane_z_in_board=object_height_m / 2.0, T_camera_board=T_camera_board)


def identity_quat_wxyz() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)