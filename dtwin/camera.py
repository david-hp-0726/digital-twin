from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    import pyorbbecsdk2 as ob
except ImportError:
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
        self._color_format = None
        self._depth_enabled = False
        self._depth_scale_m = 0.001

    def _try_enable_color(self) -> None:
        profile_list = self.pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
        color_profile = profile_list.get_default_video_stream_profile()
        self.config.enable_stream(color_profile)

    def _try_enable_depth(self) -> None:
        try:
            profile_list = self.pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR)
            depth_profile = profile_list.get_default_video_stream_profile()
            self.config.enable_stream(depth_profile)
            self._depth_enabled = True
        except Exception as exc:
            print(f"[camera] depth stream unavailable: {exc}", flush=True)
            self._depth_enabled = False

    def _try_enable_align_to_color(self) -> None:
        # Best-effort only. Different SDK builds expose different APIs.
        try:
            if hasattr(self.config, "set_align_mode") and hasattr(ob, "OBAlignMode"):
                # Try common enum names across SDK versions.
                for name in (
                    "SW_MODE",
                    "ALIGN_D2C_SW_MODE",
                    "HW_MODE",
                    "ALIGN_D2C_HW_MODE",
                ):
                    if hasattr(ob.OBAlignMode, name):
                        self.config.set_align_mode(getattr(ob.OBAlignMode, name))
                        print(f"[camera] enabled align mode: {name}", flush=True)
                        return
        except Exception as exc:
            print(f"[camera] depth-color alignment unavailable: {exc}", flush=True)

    def start(self) -> None:
        self._try_enable_color()
        self._try_enable_depth()
        if self._depth_enabled:
            self._try_enable_align_to_color()

        self.pipeline.start(self.config)
        self._started = True

        frames = None
        color_frame = None
        depth_frame = None

        # Warm up camera streams. Some SDK/device combos need a few frames
        # before color becomes available, especially when depth+align are enabled.
        for i in range(30):
            frames = self.pipeline.wait_for_frames(500)
            if frames is None:
                print(f"[camera] warmup {i}: no frames", flush=True)
                continue

            try:
                color_frame = frames.get_color_frame()
            except Exception:
                color_frame = None

            try:
                depth_frame = frames.get_depth_frame()
            except Exception:
                depth_frame = None

            print(
                f"[camera] warmup {i}: color={'yes' if color_frame is not None else 'no'} "
                f"depth={'yes' if depth_frame is not None else 'no'}",
                flush=True,
            )

            if color_frame is not None:
                break

        if color_frame is None:
            raise RuntimeError(
                "Camera started, but no color frame was received after warmup. "
                "Try disabling depth/alignment first to verify color-only streaming works."
            )

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

        try:
            self._color_format = color_frame.get_format()
            print(f"[camera] color format: {self._color_format}", flush=True)
        except Exception:
            self._color_format = None

        if self._depth_enabled and depth_frame is not None:
            self._depth_scale_m = self._get_depth_scale_m(depth_frame)
            print(f"[camera] depth enabled, scale={self._depth_scale_m:.6f} m/unit", flush=True)
        elif self._depth_enabled:
            print("[camera] depth enabled but no initial depth frame received", flush=True)

    def stop(self) -> None:
        if self._started:
            self.pipeline.stop()
            self._started = False

    @property
    def intrinsics(self) -> CameraIntrinsics:
        if self._intrinsics is None:
            raise RuntimeError("Camera intrinsics are not available. Call start() first.")
        return self._intrinsics

    def _decode_color_frame(self, color_frame) -> np.ndarray:
        vf = color_frame.as_video_frame()
        w, h = vf.get_width(), vf.get_height()

        raw = np.frombuffer(vf.get_data(), dtype=np.uint8)
        n = raw.size

        if n == h * w * 3:
            img = raw.reshape((h, w, 3))
            return img.copy()

        if n == h * w:
            gray = raw.reshape((h, w))
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        if n == h * w * 2:
            yuyv = raw.reshape((h, w, 2))
            return cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUY2)

        img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if img is not None:
            return img

        raise RuntimeError(
            f"Unsupported color frame layout: data_size={n}, width={w}, height={h}, "
            f"format={self._color_format}"
        )

    def _get_depth_scale_m(self, depth_frame) -> float:
        for attr in ("get_depth_scale", "get_value_scale"):
            if hasattr(depth_frame, attr):
                try:
                    return float(getattr(depth_frame, attr)())
                except Exception:
                    pass
        return 0.001

    def _decode_depth_frame(self, depth_frame, target_shape=None) -> np.ndarray:
        vf = depth_frame.as_video_frame()
        w, h = vf.get_width(), vf.get_height()

        raw = np.frombuffer(vf.get_data(), dtype=np.uint16).reshape(h, w)

        # Most likely Gemini raw depth is in millimeters
        depth_m = raw.astype(np.float32) / 1000.0

        if target_shape is not None and (depth_m.shape[0] != target_shape[0] or depth_m.shape[1] != target_shape[1]):
            depth_m = cv2.resize(depth_m, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)

        # turn zeros into invalids
        depth_m[depth_m <= 0] = np.nan
        return depth_m

    def read(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        for _ in range(5):
            frames = self.pipeline.wait_for_frames(self.timeout_ms)
            if frames is None:
                continue

            try:
                color_frame = frames.get_color_frame()
            except Exception:
                color_frame = None

            if color_frame is None:
                continue

            color = self._decode_color_frame(color_frame)

            depth = None
            if self._depth_enabled:
                try:
                    depth_frame = frames.get_depth_frame()
                except Exception:
                    depth_frame = None

                if depth_frame is not None:
                    try:
                        depth = self._decode_depth_frame(depth_frame, target_shape=color.shape[:2])
                    except Exception as exc:
                        print(f"[camera] failed to decode depth frame: {exc}", flush=True)
                        depth = None

            return color, depth

        return None, None

    def read_color(self) -> Optional[np.ndarray]:
        color, _ = self.read()
        return color

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
        return np.zeros((5, 1), dtype=np.float64)
