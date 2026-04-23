from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class BoardConfig:
    dictionary: str
    markers_x: int
    markers_y: int
    marker_length_m: float
    marker_separation_m: float
    axis_length_m: float = 0.05


def create_board(cfg: BoardConfig):
    dict_id = getattr(cv2.aruco, cfg.dictionary)
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
    board = cv2.aruco.GridBoard(
        (cfg.markers_x, cfg.markers_y),
        cfg.marker_length_m,
        cfg.marker_separation_m,
        dictionary,
    )
    return dictionary, board


def generate_board_image(
    cfg: BoardConfig,
    out_path: str,
    pixels_per_meter: int,
    margin_px: int,
) -> None:
    _, board = create_board(cfg)
    board_width_m = cfg.markers_x * cfg.marker_length_m + (cfg.markers_x - 1) * cfg.marker_separation_m
    board_height_m = cfg.markers_y * cfg.marker_length_m + (cfg.markers_y - 1) * cfg.marker_separation_m

    out_w = int(board_width_m * pixels_per_meter) + 2 * margin_px
    out_h = int(board_height_m * pixels_per_meter) + 2 * margin_px

    img = board.generateImage((out_w, out_h), marginSize=margin_px, borderBits=1)
    cv2.imwrite(out_path, img)


class ArucoBoardTracker:
    def __init__(self, cfg: BoardConfig) -> None:
        self.cfg = cfg
        self.dictionary, self.board = create_board(cfg)
        self.detector = cv2.aruco.ArucoDetector(
            self.dictionary,
            cv2.aruco.DetectorParameters(),
        )

    def estimate_pose(
        self,
        frame_bgr: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        vis = frame_bgr.copy()
        corners, ids, _ = self.detector.detectMarkers(frame_bgr)

        if ids is None or len(ids) == 0:
            return vis, None, None, None

        cv2.aruco.drawDetectedMarkers(vis, corners, ids)

        obj_points, img_points = self.board.matchImagePoints(corners, ids)
        if obj_points is None or img_points is None or len(obj_points) < 4:
            return vis, None, None, ids

        ok, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs)
        if not ok:
            return vis, None, None, ids

        cv2.drawFrameAxes(
            vis,
            camera_matrix,
            dist_coeffs,
            rvec,
            tvec,
            self.cfg.axis_length_m,
        )
        return vis, rvec, tvec, ids


def rvec_tvec_to_transform(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    rot, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rot
    T[:3, 3] = tvec.reshape(3)
    return T