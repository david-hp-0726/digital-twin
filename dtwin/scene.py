from __future__ import annotations

import pathlib
from typing import Any


def board_size_m(cfg: dict[str, Any]) -> tuple[float, float]:
    w = (
        cfg["board"]["markers_x"] * cfg["board"]["marker_length_m"]
        + (cfg["board"]["markers_x"] - 1) * cfg["board"]["marker_separation_m"]
    )
    h = (
        cfg["board"]["markers_y"] * cfg["board"]["marker_length_m"]
        + (cfg["board"]["markers_y"] - 1) * cfg["board"]["marker_separation_m"]
    )
    return float(w), float(h)


def generate_scene_xml(cfg: dict[str, Any], out_path: pathlib.Path) -> pathlib.Path:
    board_w, board_h = board_size_m(cfg)
    board_half_w = board_w / 2.0
    board_half_h = board_h / 2.0

    obj_x, obj_y, obj_z = [float(v) for v in cfg["object"]["size_m"]]
    obj_half_x = obj_x / 2.0
    obj_half_y = obj_y / 2.0
    obj_half_z = obj_z / 2.0

    board_img = cfg["board"]["board_image_path"]

    xml = f"""
<mujoco model="object_tracker">
  <option timestep="0.01"/>

  <asset>
    <texture name="board_tex" type="2d" file="{board_img}"/>
    <material name="board_mat" texture="board_tex" texrepeat="1 1" texuniform="false"/>

    <texture name="table_tex" type="2d" builtin="flat" rgb1="0.7 0.7 0.7" width="32" height="32"/>
    <material name="table_mat" texture="table_tex"/>
  </asset>

  <visual>
    <headlight diffuse="0.8 0.8 0.8" ambient="0.4 0.4 0.4" specular="0.1 0.1 0.1"/>
  </visual>

  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1"/>

    <geom
      name="aruco_board"
      type="box"
      pos="0 0 0.001"
      size="{board_half_w:.6f} {board_half_h:.6f} 0.001"
      material="board_mat"
      rgba="1 1 1 1"/>

    <geom
      name="table"
      type="box"
      pos="0 0 -0.02"
      size="1 1 0.02"
      material="table_mat"
      rgba="1 1 1 1"/>

    <body name="{cfg["mujoco"]["body_name"]}" pos="0 0 {obj_half_z:.6f}">
      <freejoint/>
      <geom
        name="tracked_object_geom"
        type="box"
        size="{obj_half_x:.6f} {obj_half_y:.6f} {obj_half_z:.6f}"
        rgba="1 0.5 0 1"/>
    </body>

    <site name="world_origin" pos="0 0 0" size="0.01" rgba="1 1 1 1"/>
    <geom name="world_x_axis" type="capsule" fromto="0 0 0 0.12 0 0" size="0.003" rgba="1 0 0 1"/>
    <geom name="world_y_axis" type="capsule" fromto="0 0 0 0 0.12 0" size="0.003" rgba="0 1 0 1"/>
    <geom name="world_z_axis" type="capsule" fromto="0 0 0 0 0 0.12" size="0.003" rgba="0 0 1 1"/>
  </worldbody>
</mujoco>
"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(xml, encoding="utf-8")
    return out_path