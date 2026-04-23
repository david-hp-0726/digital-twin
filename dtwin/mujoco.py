from __future__ import annotations

import mujoco
import mujoco.viewer
import numpy as np


class MujocoObjectViewer:
    def __init__(self, xml_path: str, body_name: str) -> None:
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            raise ValueError(f"Body '{body_name}' not found in MuJoCo model.")
        self.body_id = body_id

        joint_adr = self.model.body_jntadr[self.body_id]
        if joint_adr < 0:
            raise ValueError(f"Body '{body_name}' has no joint.")
        self.joint_id = joint_adr
        self.qpos_adr = self.model.jnt_qposadr[self.joint_id]

    def set_body_pose(self, pos_xyz: np.ndarray, quat_wxyz: np.ndarray) -> None:
        self.data.qpos[self.qpos_adr:self.qpos_adr + 3] = pos_xyz
        self.data.qpos[self.qpos_adr + 3:self.qpos_adr + 7] = quat_wxyz
        mujoco.mj_forward(self.model, self.data)

    def run(self, update_fn):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running():
                update_fn(self)
                viewer.sync()