from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pyorbbecsdk as ob


@dataclass
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


class GeminiCamera:
    def __init__(self, timeout_ms: int = 100) -> None:
        self.timeout_ms = timeout_ms
        self.pipeline = ob.Pipeline()
        self.config = ob.Config()
        self._started = False
        self._intrinsics: Optional[CameraIntrinsics] = None

    def start(self) -> None:
        profile_list = self.pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
        color_profile = profile_list.get_default_video_stream_profile()
        self.config.enable_stream(color_profile)
        self.pipeline.start(self.config)
        self._started = True

        frames = self.pipeline.wait_for_frames(1000)
        if frames is None:
            raise RuntimeError("No frames received while initializing camera.")

        color_frame = frames.get_color_frame()
        if color_frame is None:
            raise RuntimeError("No color frame received while initializing camera.")

        profile = color_frame.get_stream_profile().as_video_stream_profile()
        intr = profile.get_intrinsic()
        self._intrinsics = CameraIntrinsics(
            width=intr.width,
            height=intr.height,
            fx=intr.fx,
            fy=intr.fy,
            cx=intr.cx,
            cy=intr.cy,
        )

    def stop(self) -> None:
        if self._started:
            self.pipeline.stop()
            self._started = False

    @property
    def intrinsics(self) -> CameraIntrinsics:
        if self._intrinsics is None:
            raise RuntimeError("Camera intrinsics are not available. Call start() first.")
        return self._intrinsics

    def read_color(self) -> Optional[np.ndarray]:
        frames = self.pipeline.wait_for_frames(self.timeout_ms)
        if frames is None:
            return None

        color_frame = frames.get_color_frame()
        if color_frame is None:
            return None

        vf = color_frame.as_video_frame()
        w, h = vf.get_width(), vf.get_height()
        img = np.frombuffer(vf.get_data(), dtype=np.uint8).reshape((h, w, 3))
        return img.copy()

    def get_camera_matrix(self) -> np.ndarray:
        intr = self.intrinsics
        return np.array(
            [
                [intr.fx, 0.0, intr.cx],
                [0.0, intr.fy, intr.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def get_dist_coeffs(self) -> np.ndarray:
        # Minimal starter version: zero distortion.
        # Replace with mapped Gemini distortion coefficients later if needed.
        return np.zeros((5, 1), dtype=np.float64)