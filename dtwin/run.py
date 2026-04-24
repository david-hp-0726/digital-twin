from __future__ import annotations

import pathlib
import traceback

import cv2
import numpy as np
import yaml

from dtwin.board import ArucoBoardTracker, BoardConfig, camera_to_board_transform
from dtwin.camera import GeminiCamera
from dtwin.mujoco import MujocoObjectViewer
from dtwin.pose import (
    ColorSegConfig,
    LowPassVec3,
    estimate_object_position_from_depth,
    identity_quat_wxyz,
    segment_object,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_config():
    print("[run] loading config", flush=True)
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def board_to_mujoco(pos_board: np.ndarray, cfg) -> np.ndarray:
    R = np.array(cfg["world"]["board_to_mujoco_rotation"], dtype=np.float64)
    t = np.array(cfg["world"]["board_to_mujoco_translation_m"], dtype=np.float64)
    return R @ pos_board + t


def depth_to_vis(depth_m: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth_m) & (depth_m > 0)
    if not np.any(valid):
        return np.zeros((*depth_m.shape, 3), dtype=np.uint8)
    d = depth_m.copy()
    near = np.percentile(d[valid], 5)
    far = np.percentile(d[valid], 95)
    if far <= near:
        far = near + 1e-3
    d = np.clip((d - near) / (far - near), 0.0, 1.0)
    d_u8 = (255.0 * (1.0 - d)).astype(np.uint8)
    d_u8[~valid] = 0
    return cv2.applyColorMap(d_u8, cv2.COLORMAP_TURBO)


def main() -> None:
    cfg = load_config()

    print("[run] constructing camera", flush=True)
    cam = GeminiCamera(timeout_ms=cfg["camera"]["timeout_ms"])

    cam.start()
    print("[run] camera started", flush=True)

    print("[run] constructing board tracker", flush=True)
    board_tracker = ArucoBoardTracker(
        BoardConfig(
            dictionary=cfg["board"]["dictionary"],
            markers_x=cfg["board"]["markers_x"],
            markers_y=cfg["board"]["markers_y"],
            marker_length_m=cfg["board"]["marker_length_m"],
            marker_separation_m=cfg["board"]["marker_separation_m"],
            axis_length_m=cfg["board"]["axis_length_m"],
        )
    )
    board_w = (
        cfg["board"]["markers_x"] * cfg["board"]["marker_length_m"]
        + (cfg["board"]["markers_x"] - 1) * cfg["board"]["marker_separation_m"]
    )
    board_h = (
        cfg["board"]["markers_y"] * cfg["board"]["marker_length_m"]
        + (cfg["board"]["markers_y"] - 1) * cfg["board"]["marker_separation_m"]
    )

    seg_cfg = ColorSegConfig(
        hsv_lower=tuple(cfg["object"]["hsv_lower"]),
        hsv_upper=tuple(cfg["object"]["hsv_upper"]),
        hsv_lower_2=tuple(cfg["object"]["hsv_lower_2"]),
        hsv_upper_2=tuple(cfg["object"]["hsv_upper_2"]),
        min_area_px=cfg["object"]["min_area_px"],
    )

    position_filter = LowPassVec3(alpha=cfg["tracking"]["position_alpha"])

    print("[run] constructing mujoco viewer", flush=True)
    viewer = MujocoObjectViewer(
        xml_path=str(ROOT / cfg["mujoco"]["xml_path"]),
        body_name=cfg["mujoco"]["body_name"],
    )
    print("[run] mujoco viewer ready", flush=True)

    obj_size = np.array(cfg["object"]["size_m"], dtype=np.float64)
    # Treat size_m[2] as half-height if that is how your MuJoCo geom is defined.
    initial_z = float(obj_size[2])

    print("[run] reading camera intrinsics", flush=True)
    camera_matrix = cam.get_camera_matrix()
    dist_coeffs = cam.get_dist_coeffs()
    print(f"[run] camera_matrix=\n{camera_matrix}", flush=True)

    latest_pos = np.array([0.0, 0.0, initial_z], dtype=np.float64)
    latest_quat = identity_quat_wxyz()

    def update_fn(mjv: MujocoObjectViewer) -> None:
        nonlocal latest_pos, latest_quat

        frame, depth_m = cam.read()
        if frame is None:
            print("[run] no color frame", flush=True)
            mjv.set_body_pose(latest_pos, latest_quat)
            return

        vis, rvec, tvec, ids = board_tracker.estimate_pose(frame, camera_matrix, dist_coeffs)

        mask, bbox = segment_object(frame, seg_cfg)
        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 2)

        if rvec is not None and tvec is not None and bbox is not None and depth_m is not None:
            T_camera_to_board = camera_to_board_transform(rvec, tvec)

            pos_board = estimate_object_position_from_depth(
                mask=mask,
                depth_m=depth_m,
                camera_matrix=camera_matrix,
                T_camera_to_board=T_camera_to_board,
                min_valid_depth_m=0.05,
                max_valid_depth_m=5.0,
                sample_stride=2,
            )
            # pos_board = pos_board - np.array([0.082, 0.082, 0.0], dtype=np.float64)

            if pos_board is not None:
                pos_mujoco = board_to_mujoco(pos_board, cfg)
                latest_pos = position_filter.update(pos_mujoco)

                cv2.putText(
                    vis,
                    f"Board pos: [{pos_board[0]:.3f}, {pos_board[1]:.3f}, {pos_board[2]:.3f}]",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                )

                cv2.putText(
                    vis,
                    f"MuJoCo pos: [{latest_pos[0]:.3f}, {latest_pos[1]:.3f}, {latest_pos[2]:.3f}]",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

        elif depth_m is None:
            cv2.putText(
                vis,
                "Depth unavailable: pose falls back to last estimate",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

        mjv.set_body_pose(latest_pos, latest_quat)

        cv2.imshow("DTwin Debug", vis)
        # cv2.imshow("Mask", mask)
        # if depth_m is not None:
        #     cv2.imshow("Depth", depth_to_vis(depth_m))

        key = cv2.waitKey(1)
        if key == 27 or key == ord("q"):
            raise KeyboardInterrupt

    try:
        print("[run] launching mujoco viewer loop", flush=True)
        viewer.run(update_fn)
    except KeyboardInterrupt:
        print("[run] interrupted by user", flush=True)
    finally:
        print("[run] stopping camera", flush=True)
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("[run] fatal exception:", flush=True)
        traceback.print_exc()
        raise
