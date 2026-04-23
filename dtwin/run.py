from __future__ import annotations

import pathlib

import cv2
import numpy as np
import yaml

from dtwin.board import ArucoBoardTracker, BoardConfig, rvec_tvec_to_transform
from dtwin.camera import GeminiCamera
from dtwin.mujoco import MujocoObjectViewer
from dtwin.pose import (
    ColorSegConfig,
    LowPassVec3,
    estimate_object_position_on_board,
    identity_quat_wxyz,
    segment_red_object,
)


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_config():
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    cfg = load_config()

    cam = GeminiCamera(timeout_ms=cfg["camera"]["timeout_ms"])
    cam.start()

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

    seg_cfg = ColorSegConfig(
        hsv_lower=tuple(cfg["object"]["hsv_lower"]),
        hsv_upper=tuple(cfg["object"]["hsv_upper"]),
        hsv_lower_2=tuple(cfg["object"]["hsv_lower_2"]),
        hsv_upper_2=tuple(cfg["object"]["hsv_upper_2"]),
        min_area_px=cfg["object"]["min_area_px"],
    )

    position_filter = LowPassVec3(alpha=cfg["tracking"]["position_alpha"])

    viewer = MujocoObjectViewer(
        xml_path=str(ROOT / cfg["mujoco"]["xml_path"]),
        body_name=cfg["mujoco"]["body_name"],
    )

    obj_size = np.array(cfg["object"]["size_m"], dtype=np.float64)
    object_height_m = float(obj_size[2] * 2.0)

    camera_matrix = cam.get_camera_matrix()
    dist_coeffs = cam.get_dist_coeffs()

    latest_pos = np.array([0.0, 0.0, object_height_m / 2.0], dtype=np.float64)
    latest_quat = identity_quat_wxyz()

    def update_fn(mjv: MujocoObjectViewer) -> None:
        nonlocal latest_pos, latest_quat

        frame = cam.read_color()
        if frame is None:
            mjv.set_body_pose(latest_pos, latest_quat)
            return

        vis, rvec, tvec, ids = board_tracker.estimate_pose(frame, camera_matrix, dist_coeffs)

        mask, bbox = segment_red_object(frame, seg_cfg)
        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 2)

        if rvec is not None and tvec is not None and bbox is not None:
            T_camera_board = rvec_tvec_to_transform(rvec, tvec)
            pos_board = estimate_object_position_on_board(
                bbox=bbox,
                camera_matrix=camera_matrix,
                T_camera_board=T_camera_board,
                object_height_m=object_height_m,
            )
            if pos_board is not None:
                latest_pos = position_filter.update(pos_board)

        mjv.set_body_pose(latest_pos, latest_quat)

        cv2.putText(
            vis,
            f"Object pos (board frame): [{latest_pos[0]:.3f}, {latest_pos[1]:.3f}, {latest_pos[2]:.3f}]",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.imshow("DTwin Debug", vis)
        cv2.imshow("Mask", mask)

        key = cv2.waitKey(1)
        if key == 27 or key == ord("q"):
            raise KeyboardInterrupt

    try:
        viewer.run(update_fn)
    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()